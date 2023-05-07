import pickle

import sklearn.metrics
import tensorflow as tf

import mmsdk
import numpy as np
import os
from mmsdk import mmdatasdk as md
import torch

with open('./pre-processed-data/train.data', 'rb') as f:
    train = pickle.load(f)
with open('./pre-processed-data/test.data', 'rb') as f:
    test = pickle.load(f)
with open('./pre-processed-data/dev.data', 'rb') as f:
    dev = pickle.load(f)
import re
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm_notebook
from collections import defaultdict

batch_sz = 128
train_flag = False
test_flag = True


def multi_collate(batch):
    batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)

    # I eat food _ _ _ -> 1
    # I eat a lot of food -> 1.2

    labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
    sentences = pad_sequence([torch.FloatTensor(sample[0][0]) for sample in batch])
    visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
    acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])

    lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])
    return sentences, visual, acoustic, labels, lengths


train_loader = DataLoader(train, shuffle=True, batch_size=batch_sz, collate_fn=multi_collate)
dev_loader = DataLoader(dev, shuffle=False, batch_size=batch_sz * 3, collate_fn=multi_collate)
test_loader = DataLoader(test, shuffle=False, batch_size=batch_sz * 3, collate_fn=multi_collate)


class LFLSTM(nn.Module):
    def __init__(self, input_sizes, hidden_sizes, fc1_size, output_size, dropout_rate):
        super(LFLSTM, self).__init__()
        self.input_size = input_sizes
        self.hidden_size = hidden_sizes
        self.fc1_size = fc1_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        # defining modules - two layer bidirectional LSTM with layer norm in between
        self.embed = nn.Embedding(2196017, input_sizes[0])
        self.trnn1 = nn.LSTM(input_sizes[0], hidden_sizes[0], bidirectional=True)
        self.trnn2 = nn.LSTM(2 * hidden_sizes[0], hidden_sizes[0], bidirectional=True)

        self.vrnn1 = nn.LSTM(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = nn.LSTM(2 * hidden_sizes[1], hidden_sizes[1], bidirectional=True)

        self.arnn1 = nn.LSTM(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = nn.LSTM(2 * hidden_sizes[2], hidden_sizes[2], bidirectional=True)

        self.lstm1 = nn.LSTM(sum(hidden_sizes) * 4, 512,
                             bidirectional=True)
        self.lstm2 = nn.LSTM(1024, 512,
                             bidirectional=True)
        self.lstm3 = nn.LSTM(1024, 512,
                             bidirectional=True)
        self.fc1 = nn.Linear(1024, fc1_size)
        self.fc2 = nn.Linear(fc1_size, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0] * 2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1] * 2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2] * 2,))
        self.bn = nn.BatchNorm1d(sum(hidden_sizes) * 4)

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths)
        packed_h1, (final_h1, _) = rnn1(packed_sequence)
        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)
        _, (final_h2, _) = rnn2(packed_normed_h1)
        return final_h1, final_h2

    def fusion(self, sentences, visual, acoustic, lengths):
        batch_size = lengths.size(0)
        # sentences = self.embed(sentences)

        # extract features from text modality
        final_h1t, final_h2t = self.extract_features(sentences, lengths, self.trnn1, self.trnn2, self.tlayer_norm)

        # extract features from visual modality
        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)

        # extract features from acoustic modality
        final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)

        # simple late fusion -- concatenation + normalization
        h = torch.cat((final_h1t, final_h2t, final_h1v, final_h2v, final_h1a, final_h2a),
                      dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        return self.bn(h)

    def forward(self, sentences, visual, acoustic, lengths):
        batch_size = lengths.size(0)
        h = self.fusion(sentences, visual, acoustic, lengths.cpu())
        h, (_, _) = self.lstm1(h)
        h, (_, _) = self.lstm2(h)
        h, (_, _) = self.lstm3(h)
        h = self.fc1(h)
        h = self.dropout(h)
        h = self.relu(h)
        o = self.fc2(h)
        return o


from tqdm import tqdm_notebook
from torch.optim import Adam, SGD
from sklearn.metrics import accuracy_score, f1_score

torch.manual_seed(2023)
torch.cuda.manual_seed_all(2023)

CUDA = torch.cuda.is_available()
print('cuda is available: ' + str(CUDA))

MAX_EPOCH = 1000

text_size = 300
visual_size = 47
acoustic_size = 74

input_sizes = [text_size, visual_size, acoustic_size]
hidden_sizes = [int(text_size * 1.5), int(visual_size * 1.5), int(acoustic_size * 1.5)]
fc1_size = sum(hidden_sizes) // 2
dropout = 0.1
output_size = 1
curr_patience = patience = 5
num_trials = 1
grad_clip_value = 1.0
weight_decay = 0.001
pretrained_emb = None
model = LFLSTM(input_sizes, hidden_sizes, fc1_size, output_size, dropout)

model.embed.requires_grad = False
optimizer = Adam([param for param in model.parameters() if param.requires_grad], weight_decay=weight_decay)

if CUDA:
    model.cuda()
criterion = nn.L1Loss(reduction='sum')
criterion_test = nn.L1Loss(reduction='sum')
best_valid_loss = float('inf')
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
lr_scheduler.step()  # for some reason it seems the StepLR needs to be stepped once first
if train_flag:
    train_losses = []
    valid_losses = []
    for e in range(MAX_EPOCH):
        model.train()
        train_iter = tqdm_notebook(train_loader)
        train_loss = 0.0
        for batch in train_iter:
            model.zero_grad()
            t, v, a, y, l = batch
            batch_size = t.size(0)
            if CUDA:
                t = t.cuda()
                v = v.cuda()
                a = a.cuda()
                y = y.cuda()
                l = l.cuda()
            y_tilde = model(t, v, a, l)
            loss = criterion(y_tilde, y)
            loss.backward()
            torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad],
                                            grad_clip_value)
            optimizer.step()
            train_iter.set_description(
                f"Epoch {e}/{MAX_EPOCH}, current batch loss: {round(loss.item() / batch_size, 4)}")
            train_loss += loss.item()
        train_loss = train_loss / len(train)
        train_losses.append(train_loss)
        print(f"Training loss: {round(train_loss, 4)}")

        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            for batch in dev_loader:
                model.zero_grad()
                t, v, a, y, l = batch
                if CUDA:
                    t = t.cuda()
                    v = v.cuda()
                    a = a.cuda()
                    y = y.cuda()
                    # l = l.cuda()
                y_tilde = model(t, v, a, l)
                loss = criterion(y_tilde, y)
                valid_loss += loss.item()

        valid_loss = valid_loss / len(dev)
        valid_losses.append(valid_loss)
        print(f"Validation loss: {round(valid_loss, 4)}")
        print(f"Current patience: {curr_patience}, current trial: {num_trials}.")
        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            print("Found new best model on dev set!")
            torch.save(model.state_dict(), 'model.std')
            torch.save(optimizer.state_dict(), 'optim.std')
            curr_patience = patience
        else:
            curr_patience -= 1
            if curr_patience <= -1:
                print("Running out of patience, loading previous best model.")
                num_trials -= 1
                curr_patience = patience
                model.load_state_dict(torch.load('model.std'))
                optimizer.load_state_dict(torch.load('optim.std'))
                lr_scheduler.step()
                print(f"Current learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")

        if num_trials <= 0:
            print("Running out of patience, early stopping.")
            break

if test_flag:
    from keras import backend


    def coeff_determination(y_true, y_pred):
        SS_res = np.sum(np.square(y_true - y_pred))
        SS_tot = np.sum(np.square(y_true - np.mean(y_true)))
        return (1 - SS_res / (SS_tot + backend.epsilon()))


    def get_accuracy_2(y_real, y_predict):
        true_pos = 0
        for i in range(len(y_real)):
            if y_real[i] < 0.5 and y_predict[i] < 0.5:
                true_pos += 1
            elif y_real[i] >= 0.5 and y_predict[i] >= 0.5:
                true_pos += 1
        return true_pos / len(y_real)


    def get_range_1_to_7(num):
        return round(float(num) / (1 / 7.0))


    def get_accuracy_7(y_real, y_predict):
        true_pos = 0
        for i in range(len(y_real)):
            if get_range_1_to_7(y_real[i]) == get_range_1_to_7(y_predict[i]):
                true_pos += 1
        return true_pos / len(y_real)


    def get_mae(y_real, y_predict):
        mae = tf.keras.losses.MeanAbsoluteError()
        return mae(y_real * 6 - 3, y_predict * 6 - 3).numpy()


    def get_classification_lists(y_real, y_predict):
        true_list = []
        predict_list = []
        for i in range(len(y_real)):
            if y_real[i] < 0.5:
                true_list.append(0)
            if y_predict[i] < 0.5:
                predict_list.append(0)
            if y_real[i] >= 0.5:
                true_list.append(1)
            if y_predict[i] >= 0.5:
                predict_list.append(1)
        return true_list, predict_list


    def get_f1(y_real, y_predict):
        true_list, predict_list = get_classification_lists(y_real, y_predict)
        return f1_score(true_list, predict_list, average='weighted')


    model.load_state_dict(torch.load('trained_model/model.std'))
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for batch in test_loader:
            model.zero_grad()
            t, v, a, y, l = batch
            if CUDA:
                t = t.cuda()
                v = v.cuda()
                a = a.cuda()
                y = y.cuda()
                l = l.cuda()
            y_tilde = model(t, v, a, l)
            loss = criterion_test(y_tilde, y)
            y_true.append(y_tilde.detach().cpu().numpy())
            y_pred.append(y.detach().cpu().numpy())
            test_loss += loss.item()
    y_real = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    y_dev_refined = y_real / 6 + 0.5
    y_dev_pred = y_pred / 6 + 0.5
    with open('../../results/d3_multimodal.txt', 'w') as w:
        w.write("Multimodal neural network F1: " + str(get_f1(y_dev_refined, y_dev_pred)) + '\n')
        w.write("Multimodal neural network MAE: " + str(get_mae(y_dev_refined, y_dev_pred)) + '\n')
        w.write("Multimodal neural network accuracy (binary): " + str(get_accuracy_2(y_dev_refined, y_dev_pred)) + '\n')
        w.write(
            "Multimodal neural network accuracy (7 class classification): " + str(
                get_accuracy_7(y_dev_refined, y_dev_pred)) + '\n')
        w.write("Multimodal neural network R^2: " + str(sklearn.metrics.r2_score(y_dev_pred, y_dev_refined)) + '\n\n')
