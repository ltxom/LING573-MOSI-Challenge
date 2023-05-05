import numpy as np
import os
import mmsdk
from mmsdk import mmdatasdk as md
import re
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm_notebook
from collections import defaultdict

# visual_field = '../data/CMU_MOSI_Visual_Facet_41'
# acoustic_field = '../data/CMU_MOSI_COVAREP'
# text_field = '../data/CMU_MOSI_TimestampedWordVectors'

# features = [
#     text_field,
#     visual_field,
#     acoustic_field
# ]

# recipe = {feat: feat + '.csd' for feat in features}
# dataset = md.mmdataset(recipe)

# def avg(intervals: np.array, features: np.array) -> np.array:
#     return np.average(features, axis=0)

# dataset.align(text_field, collapse_functions=[avg])
# label_field = '../data/CMU_MOSI_Opinion_Labels'

# label_recipe = {label_field:  label_field + '.csd'}
# dataset.add_computational_sequences(label_recipe, destination=None)
# dataset.align(label_field)

# train_split = md.cmu_mosi.standard_folds.standard_train_fold
# dev_split = md.cmu_mosi.standard_folds.standard_valid_fold
# test_split = md.cmu_mosi.standard_folds.standard_test_fold


# # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
# EPS = 0

# # place holders for the final train/dev/test dataset
# train = []
# dev = []
# test = []

# # define a regular expression to extract the video ID out of the keys
# pattern = re.compile('(.*)\[.*\]')
# num_drop = 0 # a counter to count how many data points went into some processing issues

# for segment in dataset[label_field].keys():

#     # get the video ID and the features out of the aligned dataset
#     vid = re.search(pattern, segment).group(1)
#     label = dataset[label_field][segment]['features']
#     _words = dataset[text_field][segment]['features']
#     _visual = dataset[visual_field][segment]['features']
#     _acoustic = dataset[acoustic_field][segment]['features']

#     # if the sequences are not same length after alignment, there must be some problem with some modalities
#     # we should drop it or inspect the data again
#     if not _words.shape[0] == _visual.shape[0] == _acoustic.shape[0]:
#         print(f"Encountered datapoint {vid} with text shape {_words.shape}, visual shape {_visual.shape}, acoustic shape {_acoustic.shape}")
#         num_drop += 1
#         continue

#     # remove nan values
#     label = np.nan_to_num(label)
#     _visual = np.nan_to_num(_visual)
#     _acoustic = np.nan_to_num(_acoustic)

#     words = np.asarray(_words)
#     visual = np.asarray(_visual)
#     acoustic = np.asarray(_acoustic)

#     # z-normalization per instance and remove nan/infs
#     visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
#     acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))

#     if vid in train_split:
#         train.append(((words, visual, acoustic), label, segment))
#     elif vid in dev_split:
#         dev.append(((words, visual, acoustic), label, segment))
#     elif vid in test_split:
#         test.append(((words, visual, acoustic), label, segment))
#     else:
#         print(f"Found video that doesn't belong to any splits: {vid}")

# print(f"Total number of {num_drop} datapoints have been dropped.")

# with open('./pre-processed-data/train.data', 'wb') as f:
#     np.save(f, train)

# with open('./pre-processed-data/dev.data', 'wb') as f:
#     np.save(f, dev)

# with open('./pre-processed-data/test.data', 'wb') as f:
#     np.save(f, test)

with open('./pre-processed-data/train.data', 'rb') as f:
    train = np.load(f,allow_pickle=True)

with open('./pre-processed-data/dev.data', 'rb') as f:
    dev = np.load(f,allow_pickle=True)

with open('./pre-processed-data/test.data', 'rb') as f:
    test = np.load(f,allow_pickle=True)


print(len(train))
print(len(dev))
print(len(test))

print(train[0][0][1].shape)
print(train[0][1].shape)
print(train[0][1])