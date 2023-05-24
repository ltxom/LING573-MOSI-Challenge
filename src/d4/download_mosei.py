import os

from mmsdk import mmdatasdk
from mmsdk import mmdatasdk as md

# download dataset
# cmumosei_highlevel = mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.highlevel, './data_mosei/')
cmumosei_labels = mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.labels, './data_mosei/')