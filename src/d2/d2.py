# from mmsdk import mmdatasdk
# from mmsdk import mmdatasdk as md
#
# from sklearn.metrics import accuracy_score
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.svm import SVR

import numpy as np
# import tensorflow
from keras.layers import Dense
from keras import Sequential

from keras import backend as K

# Seed value
# Apparently you may use different seed values at each stage
seed_value = 2023

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf

tf.random.set_seed(seed_value)
# for later versions:
# tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
# for later versions:
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

# # input data
# text_field = './data/CMU_MOSI_TimestampedWordVectors'
# # output data (labels)
# label_field = './data/CMU_MOSI_Opinion_Labels'
#
# # load input text data
# dataset = md.mmdataset({text_field: text_field + '.csd'})
#
# # load labels data
# labels = {label_field: label_field + '.csd'}
#
#
# # define an average function to align X and y
#
# def avg(intervals: np.array, features: np.array) -> np.array:
#     return np.average(features, axis=0)
#
#
# dataset.align(text_field, collapse_functions=[avg])
#
# dataset.add_computational_sequences(labels, destination=None)
# dataset.align(label_field)
#
# ## Train test split
# DATASET = md.cmu_mosi
# train_split = DATASET.standard_folds.standard_train_fold
# dev_split = DATASET.standard_folds.standard_valid_fold
# test_split = DATASET.standard_folds.standard_test_fold
#
#
# def get_split(dataset, split, isX=True):
#     if isX:
#         field = text_field
#     else:
#         field = label_field
#     result = {}
#     for i in split:
#         counter = 0
#         while True:
#             key_1 = i + '[' + str(counter) + ']'
#             counter += 1
#             if key_1 not in dataset[field].keys():
#                 break
#             if i not in result:
#                 result[i] = []
#             result[i].append(dataset[field][key_1])
#     return result
#
#
# x_train = get_split(dataset, train_split)
# x_dev = get_split(dataset, dev_split)
# x_val = get_split(dataset, test_split)
# y_train = get_split(dataset, train_split, False)
# y_dev = get_split(dataset, dev_split, False)
# y_val = get_split(dataset, test_split, False)
#
#
# # sum the vectors and get average
# def get_ave(x):
#     result_x = []
#     for key in x:
#         for item in x[key]:
#             result_x.append(np.mean(item['features'], axis=0))
#     return np.asarray(result_x)
#
#
# def get_refined_y(y):
#     result_y = []
#     for key in y:
#         for item in y[key]:
#             result_y.append(item['features'][0] / 6 + 0.5)
#     return np.asarray(result_y)
#
#
# x_train_ave = get_ave(x_train)
# y_train_refined = get_refined_y(y_train)
# x_dev_ave = get_ave(x_dev)
# y_dev_refined = get_refined_y(y_dev)
# x_val_ave = get_ave(x_val)
# y_val_refined = get_refined_y(y_val)
#
# with open('./pre-processed-data/x_train_ave.data', 'wb') as f:
#     np.save(f, x_train_ave)
#
# with open('./pre-processed-data/y_train_refined.data', 'wb') as f:
#     np.save(f, y_train_refined)
#
# with open('./pre-processed-data/x_dev_ave.data', 'wb') as f:
#     np.save(f, x_dev_ave)
#
# with open('./pre-processed-data/y_dev_refined.data', 'wb') as f:
#     np.save(f, y_dev_refined)
#
# with open('./pre-processed-data/x_val_ave.data', 'wb') as f:
#     np.save(f, x_val_ave)
#
# with open('./pre-processed-data/y_val_refined.data', 'wb') as f:
#     np.save(f, y_val_refined)


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

Cs = 200
epsilons = 1
gammas = 'auto'

m = SVR(kernel='rbf')
m.fit(x_train_ave, y_train_refined.ravel())
# training_result = m.predict(x_train_ave)
# val_result = m.predict(x_val_ave)
# test_result = m.predict(x_dev_ave)

train_score = m.score(x_train_ave, y_train_refined)
val_score = m.score(x_val_ave, y_val_refined)
test_score = m.score(x_dev_ave, y_dev_refined)

# print(c, epsilon, gamma, end='\t')

# print('Scores of predictions from the primal SVM on the training data:')
# print('Training sets:', train_score, end='\n')
# print('Val: ', val_score, end='\n')
# print('Test sets:', test_score)

model = Sequential([Dense(512, input_shape=(300,), activation='relu'), Dense(256, activation='relu'),
                    Dense(1, activation='sigmoid')])
model.compile()
output = model.predict(x_train_ave[0].reshape(1, -1))

model.compile('adam', 'mean_squared_error', metrics=['mse'])
history_model = model.fit(x_train_ave, y_train_refined, batch_size=256, epochs=20, verbose=1, validation_data=(x_val_ave, y_val_refined))


def coeff_determination(y_true, y_pred):
    SS_res = np.sum(K.square(y_true - y_pred))
    SS_tot = np.sum(K.square(y_true - np.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


y_dev_pred = model.predict(x_dev_ave)


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


print('Please check results in the \'results\' directory.')
with open('../../results/d2.txt', 'w') as w:
    w.write("neural network F1: " + str(get_f1(y_dev_refined, y_dev_pred)) + '\n')
    w.write("neural network MAE: " + str(get_mae(y_dev_refined, y_dev_pred)) + '\n')
    w.write("neural network accuracy (binary): " + str(get_accuracy_2(y_dev_refined, y_dev_pred)) + '\n')
    w.write("neural network accuracy (7 class classification): " + str(get_accuracy_7(y_dev_refined, y_dev_pred)) + '\n')
    w.write("neural network R^2: " + str(coeff_determination(y_dev_refined, y_dev_pred)) + '\n\n')

    w.write("SVR network F1: " + str(get_f1(y_dev_refined, m.predict(x_dev_ave))) + '\n')
    w.write("SVR network MAE: " + str(get_mae(y_dev_refined, m.predict(x_dev_ave))) + '\n')
    w.write("SVR network accuracy (binary): " + str(get_accuracy_2(y_dev_refined, m.predict(x_dev_ave))) + '\n')
    w.write("SVR network accuracy (7 segments classification): " + str(get_accuracy_7(y_dev_refined, m.predict(x_dev_ave))) + '\n')
    w.write('SVR network R^2:' + str(test_score))


print('Please check visualization outputs in the \'output\' directory.')

nn_true_list, nn_predict_list = get_classification_lists(y_dev_refined, y_dev_pred)
confusion_matrix_org = sklearn.metrics.confusion_matrix(nn_true_list, nn_predict_list)
bal_plt = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_org, display_labels=[True, False])
bal_plt.plot(cmap=plt.cm.Blues)
bal_plt.figure_.savefig('../../outputs/neural_network_conf_mat.png')
plt.clf()

svr_true_list, svr_predict_list = get_classification_lists(y_dev_refined, m.predict(x_dev_ave))
confusion_matrix_org = sklearn.metrics.confusion_matrix(svr_true_list, svr_predict_list)
bal_plt = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_org, display_labels=[True, False])
bal_plt.plot(cmap=plt.cm.Blues)
bal_plt.figure_.savefig('../../outputs/svr_conf_mat.png')
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
plt.savefig('../../outputs/ROC.png')
plt.clf()

hist = pd.DataFrame(history_model.history)
fig, ax = plt.subplots()
ax.plot(hist.index, hist['loss'], label='Training Loss')
ax.plot(hist.index, hist['val_loss'], label='Validation Loss')
ax.set_ylabel('loss')
ax.set_xlabel('epochs')
plt.title('Loss vs. Epochs')
ax.legend(loc='upper left')

plt.savefig('../../outputs/neural_network_loss.png')

