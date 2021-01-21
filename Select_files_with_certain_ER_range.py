#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Purpose    : To load and save sources whose Einstein Radii in range [1.5,4]
# Created by : Bharath Nagam, Jan 19 2021.



from astropy.io import fits
from DataGenerator import DataGenerator
from functools import partial
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Network import Network
import numpy as np
import os
from Parameters import Parameters
import pandas as pd
import random
import tensorflow as tf
import shutil
import pathlib
from tensorflow.keras import backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from utils import show2Imgs, calc_RMS, get_model_paths, get_h5_path_dialog, binary_dialog, set_experiment_folder, set_models_folders, load_settings_yaml, normalize_img, get_samples, normalize_function, compute_PSF_r, load_normalize_img
from utils import bytes2gigabyes, create_dir_if_not_exists

#1.0 - Fix memory leaks if running on tensorflow 2
tf.compat.v1.disable_eager_execution()


# 2.0 - Model Selection from directory
model_paths = get_model_paths()
print("**************")
print(model_paths)

# 2.1 - Select a weights file. There are 2 for each model. Selected based on either validation loss or validation metric. The metric can differ per model.
#h5_paths = get_h5_path_dialog(model_paths)


# 3.0 - Load params - used for normalization etc -
#yaml_path = glob.glob(os.path.join(model_paths[0], "test_beta_best_val_metric.yaml"))[0]                      # Only choose the first one for now
yaml_path = glob.glob(os.path.join(model_paths[0], "run.yaml"))[0]
settings_yaml = load_settings_yaml(yaml_path)
settings_yaml['model_name']='alpha'# Returns a dictionary object.
params = Parameters(settings_yaml, yaml_path, mode="no_training")                       # Create Parameter object
params.data_type = np.float32 if params.data_type == "np.float32" else np.float32       # This must be done here, due to the json, not accepting this kind of if statement in the parameter class.


# 4.0 - Select random sample from the data (with replacement)
sample_size = int(input("How many samples do you want to create and run (int): "))
#sources_fnames, lenses_fnames, negatives_fnames = get_samples(size=sample_size, type_data="train", deterministic=False)
sources_fnames=get_source_samples(size=sample_size, type_data="train", deterministic=False)

# 5.0 - Initialize and fill a pandas dataframe to store Source parameters
df = get_empty_dataframe()
df = fill_dataframe(df, sources_fnames)

# 6.0 - Select ER in the range of 1.5 and 4
df_er_processed=df[df['LENSER'].between(1.5, 4, inclusive=True)]

# 7.0 - Convert dataframe to a list
df_er_processed_path_as_list=df_er_processed['path'].tolist()

# 8.0 - Process and save files to destination


destination_er_train=os.path.join("data","train_refined_total","train")
destination_er_validation=os.path.join("data","train_refined_total","validation")
destination_er_test=os.path.join("data","train_refined_total","test")
create_dir_if_not_exists(destination_er)
folderPath=os.path.join("data","train","sources")
files=df_er_processed_path_as_list
for file in os.listdir(folderPath): 
    
        print(len(os.listdir(destination_er)))
        for file in files:
            if len(os.listdir(destination_er))!=(df_er_processed.shape)[0]:    
                parent_folder=pathlib.PurePath(file)
                parent_folder_name=parent_folder.parent.name
                rand_nr=random.randrange(10)
                if rand_nr<8:
                    destination_add=os.path.join(destination_er_train,parent_folder_name)
                elif rand_nr==8:
                    destination_add=os.path.join(destination_er_validation,parent_folder_name)
                elif rand_nr==9:
                    destination_add=os.path.join(destination_er_test,parent_folder_name)
                #print(destination_add)                             
                create_dir_if_not_exists(destination_add)
                shutil.copy(file, destination_add)
            
# 9.0 check:
if len(os.listdir(destination_er))==(df_er_processed.shape)[0]:
    print("Process successful")

