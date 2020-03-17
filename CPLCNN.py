# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:28:30 2020

@author: Dani
"""
import numpy as np
import collections
import glob
import os
import h5py 
#from bokeh.plotting import figure, output_file, show
import mutable_data_structs as mds
import immutable_data_structs as ids
from sklearn.model_selection import train_test_split 
import tensorflow as tf
from matplotlib import pyplot as plt
import pdb
import neural_nets
import tensorflow as tf
from scipy import signal
import skimage

from skimage.transform import rescale, resize, downscale_local_mean
    
#n = 3
#   
#arr = np.array([[2,4,6,8,10,12,14,16,18], [12,14,16,18,110,112,114,116,118]])
#window = (1.0 / n) * np.ones((1, n))
##res = np.convolve(arr, window, mode='valid')[::n]
#res = signal.convolve2d(arr, window, mode='valid')[:, ::n]

raw_filename = "C:\\Users\\Dani\\Desktop\\CPL\\08Aug19\\220_0000.cls"

CLS_meta = mds.define_CLS_meta_struct(256)
raw_engineering_data = np.fromfile(raw_filename, dtype=ids.CLS_raw_engineering_struct)

nbins = 833
nchans = 4

CLS_struct = mds.define_CLS_structure(nchans, nbins, CLS_meta)

raw_cls_data = np.fromfile(raw_filename, dtype=CLS_struct)


"""
Inputs Data
"""
#raw_filename = "C:\\Users\\Dani\\Desktop\\CPL\\08Aug19\\220_0000.cls"
#header = np.fromfile(raw_filename, dtype=CLS_raw_header_struct)
#raw_data = np.fromfile(raw_filename, dtype=CLS_struct)

directory = "C:\\Users\\Dani\\Desktop\\CPL\\08Aug19"

#Importing all .cls files from the specified directory
#
tensor_dict = collections.OrderedDict([])


for file in glob.glob('{}/*.cls'.format(directory)):
    raw_data = np.fromfile(file, dtype=CLS_struct)
    file_num = os.path.basename(os.path.normpath(file))[:-4]
    tensor_dict["photon_count_{}".format(file_num)] = raw_data['counts']

"""
Combine all Laser Energy Monitor readings into a rank
3 tensor of shape nbins x num_cpl_files x 3
Will act as input tensor for neural network
"""

input_tensor = np.transpose(np.concatenate([tensor_dict[x] for x in tensor_dict], 0), axes = (1, 0, 2))
pads = np.ones((4,54000,67))
input_tensor = np.concatenate((input_tensor, pads), axis=2)



#
##Targets
##
##0: earth, 1:unknown
##2:H2O cloud, 3: unknown cloud
##4: Ice cloud  
##"""

filename = "C:\\Users\\Dani\\Desktop\\CPL\\CPL_FIREX_L2\\CPL_L2_V1-02_01kmPro_19911_20aug19.hdf5"

hdf5 = h5py.File(filename, 'r')

target = np.array(hdf5['profile/Feature_Type'][:,:])
pads = np.ones((900,21))
target = np.array([np.concatenate((target, pads), axis=1)])
target = np.moveaxis(np.moveaxis(target, 1,2),0,2)
                
               
    
d = downscale_local_mean(input_tensor, (1, 18,1))
d = np.moveaxis(d,0,2)



CNN = neural_nets.UNet(np.array(d))

CNN.model.fit(np.array([d]),np.array([target]), validation_split=0.1, epochs=10)
#aprint("DONE!")
#
##print(image_downscaled.shape)
##
#
#
##CNN.model.fit(np.array([input_tensor]), np.array([target]))
#
#    
#
##C
##
##filename = "C:\\Users\\Dani\\Desktop\\CPL\\CPL_FIREX_L2\\CPL_L2_V1-02_01kmLay_19901_01aug19.hdf5"
##
##hdf5 = h5py.File(filename, 'r')
##
##hdf5['layer_descriptor']
##
###x1 = raw_data['VoltageMeasurements'][:,0]
###x2 = raw_data['VoltageMeasurements'][:,1]
###x3 = raw_data['VoltageMeasurements'][:,2]
##y = np.arange(0,460800,1)
##
##
###p = figure(plot_width=5000, plot_height=2000)
##
###p.line(x2, y, line_width=2)
##
###show(p)
##
##
##
##
