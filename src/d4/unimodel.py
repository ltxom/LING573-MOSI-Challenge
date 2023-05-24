import pickle

import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVR
from mmsdk import mmdatasdk as md
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

with open('./pre-processed-data/train.data', 'rb') as f:
    train = np.load(f)

with open('./pre-processed-data/test.data', 'rb') as f:
    test = np.load(f)

with open('./pre-processed-data/dev.data', 'rb') as f:
    dev = np.load(f)

# # input data
text_field = './data_mosei/CMU_MOSEI_TimestampedWordVectors'
# # output data (labels)
label_field = './data_mosei/CMU_MOSEI_Labels'

def get_split(dataset, split, isX=True):
    if isX:
        field = text_field
    else:
        field = label_field
    result = {}
    for i in split:
        counter = 0
        while True:
            key_1 = i + '[' + str(counter) + ']'
            counter += 1
            if key_1 not in dataset[field].keys():
                break
            if i not in result:
                result[i] = []
            result[i].append(dataset[field][key_1])
    return result

with open('./pre-processed-data/dataset.data', 'rb') as f:
    dataset = np.load(f)

DATASET = md.cmu_mosei
train_split = DATASET.standard_folds.standard_train_fold
dev_split = DATASET.standard_folds.standard_valid_fold
test_split = DATASET.standard_folds.standard_test_fold

x_train = get_split(dataset, train_split)
x_dev = get_split(dataset, dev_split)
x_val = get_split(dataset, test_split)
y_train = get_split(dataset, train_split, False)
y_dev = get_split(dataset, dev_split, False)
y_val = get_split(dataset, test_split, False)

with open('./pre-processed-data/x_train.data', 'wb') as f:
    np.save(f, x_train)

with open('./pre-processed-data/y_train.data', 'wb') as f:
    np.save(f, y_train)

with open('./pre-processed-data/x_dev.data', 'wb') as f:
    np.save(f, x_dev)

with open('./pre-processed-data/y_dev.data', 'wb') as f:
    np.save(f, y_dev)

with open('./pre-processed-data/x_val.data', 'wb') as f:
    np.save(f, x_val)

with open('./pre-processed-data/y_val.data', 'wb') as f:
    np.save(f, y_val)