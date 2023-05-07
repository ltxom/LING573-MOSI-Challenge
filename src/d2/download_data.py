import os


from mmsdk import mmdatasdk
from mmsdk import mmdatasdk as md

# download dataset
mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel, './data/')
mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.labels, './data/')
mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.raw, '../data_raw/')