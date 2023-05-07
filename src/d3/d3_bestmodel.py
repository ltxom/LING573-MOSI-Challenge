import pickle

import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVR

import numpy as np
from keras.layers import Dense, LSTM
from keras import Sequential

from keras import backend as K

seed_value = 2023

import os

import random
import tensorflow as tf

os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

with open('./pre-processed-data/x_train_ave.data', 'rb') as f:
    x_train_ave = np.load(f)

with open('./pre-processed-data/y_train_refined.data', 'rb') as f:
    y_train_refined = np.load(f)

with open('./pre-processed-data/x_dev_ave.data', 'rb') as f:
    x_dev_ave = np.load(f)

with open('./pre-processed-data/y_dev_refined.data', 'rb') as f:
    y_dev_refined = np.load(f)

with open('./pre-processed-data/x_val_ave.data', 'rb') as f:
    x_val_ave = np.load(f)

with open('./pre-processed-data/y_val_refined.data', 'rb') as f:
    y_val_refined = np.load(f)

x_train_new = []
x_val_new = []
x_test_new = []
for i in range(len(x_train_ave)):
    x_train_new.append(x_train_ave[i].reshape((1, 300)))
for i in range(len(x_val_ave)):
    x_val_new.append(x_val_ave[i].reshape((1, 300)))
for i in range(len(x_dev_ave)):
    x_test_new.append(x_dev_ave[i].reshape((1, 300)))

x_train_new = np.asarray(x_train_new)
x_val_new = np.asarray(x_val_new)
x_test_new = np.asarray(x_test_new)


def coeff_determination(y_true, y_pred):
    SS_res = np.sum(K.square(y_true - y_pred))
    SS_tot = np.sum(K.square(y_true - np.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def get_accuracy_2(y_real, y_predict):
    true_pos = 0
    for i in range(len(y_real)):
        if y_real[i] < 0.5 and y_predict[i] < 0.5:
            true_pos += 1
        elif y_real[i] > 0.5 and y_predict[i] > 0.5:
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


lstm_layers = 416
model = Sequential(
    [LSTM(lstm_layers, activation='relu', input_shape=(1, 300), dropout=0.5),
     Dense(lstm_layers / 2, input_shape=(lstm_layers,), activation='relu'),
     Dense(1, activation='sigmoid')])
model.compile()

model.compile('adam', 'mean_squared_error', metrics=['mse'])
history_model = model.fit(x_train_new, y_train_refined, batch_size=218, epochs=21, verbose=1,
                          validation_data=(x_val_new, y_val_refined))

y_dev_pred = model.predict(x_test_new)

Cs = 200
epsilons = 1
gammas = 'auto'

m = SVR(kernel='rbf')
m.fit(x_train_ave, y_train_refined.ravel())

train_score = m.score(x_train_ave, y_train_refined)
val_score = m.score(x_val_ave, y_val_refined)
test_score = m.score(x_dev_ave, y_dev_refined)

print('Please check results in the \'results\' directory.')
with open('../../results/d3.txt', 'w') as w:
    w.write("neural network F1: " + str(get_f1(y_dev_refined, y_dev_pred)) + '\n')
    w.write("neural network MAE: " + str(get_mae(y_dev_refined, y_dev_pred)) + '\n')
    w.write("neural network accuracy (binary): " + str(get_accuracy_2(y_dev_refined, y_dev_pred)) + '\n')
    w.write(
        "neural network accuracy (7 class classification): " + str(get_accuracy_7(y_dev_refined, y_dev_pred)) + '\n')
    w.write("neural network R^2: " + str(coeff_determination(y_dev_refined, y_dev_pred)) + '\n\n')

    w.write("SVR network F1: " + str(get_f1(y_dev_refined, m.predict(x_dev_ave))) + '\n')
    w.write("SVR network MAE: " + str(get_mae(y_dev_refined, m.predict(x_dev_ave))) + '\n')
    w.write("SVR network accuracy (binary): " + str(get_accuracy_2(y_dev_refined, m.predict(x_dev_ave))) + '\n')
    w.write("SVR network accuracy (7 segments classification): " + str(
        get_accuracy_7(y_dev_refined, m.predict(x_dev_ave))) + '\n')
    w.write('SVR network R^2:' + str(test_score))

print('Please check visualization outputs in the \'output\' directory.')

nn_true_list, nn_predict_list = get_classification_lists(y_dev_refined, y_dev_pred)
confusion_matrix_org = sklearn.metrics.confusion_matrix(nn_true_list, nn_predict_list)
bal_plt = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_org, display_labels=[True, False])
bal_plt.plot(cmap=plt.cm.Blues)
bal_plt.figure_.savefig('../../outputs/d3/neural_network_conf_mat.png')
plt.clf()

svr_true_list, svr_predict_list = get_classification_lists(y_dev_refined, m.predict(x_dev_ave))
confusion_matrix_org = sklearn.metrics.confusion_matrix(svr_true_list, svr_predict_list)
bal_plt = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_org, display_labels=[True, False])
bal_plt.plot(cmap=plt.cm.Blues)
bal_plt.figure_.savefig('../../outputs/d3/svr_conf_mat.png')
plt.clf()

fpr_nn, tpr_nn, _ = sklearn.metrics.roc_curve(nn_true_list, nn_predict_list)

fpr_svr, tpr_svr, _ = sklearn.metrics.roc_curve(svr_true_list, svr_predict_list)

nn_auc = sklearn.metrics.roc_auc_score(nn_true_list, nn_predict_list)
svr_auc = sklearn.metrics.roc_auc_score(svr_true_list, svr_predict_list)

plt.plot(fpr_nn, tpr_nn, label="Neural_Network_AUC=" + str(nn_auc))
plt.plot(fpr_svr, tpr_svr, label="SVR_AUC=" + str(svr_auc))

plt.title('ROC (receiver operating characteristic)')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.legend(loc=4)
plt.savefig('../../outputs/d3/ROC.png')
plt.clf()

hist = pd.DataFrame(history_model.history)
fig, ax = plt.subplots()
ax.plot(hist.index, hist['loss'], label='Training Loss')
ax.plot(hist.index, hist['val_loss'], label='Validation Loss')
ax.set_ylabel('loss')
ax.set_xlabel('epochs')
plt.title('Loss vs. Epochs')
ax.legend(loc='upper left')

plt.savefig('../../outputs/d3/neural_network_loss.png')
