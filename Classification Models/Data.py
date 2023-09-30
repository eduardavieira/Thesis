import pandas as pd
import numpy as np

def load_features(path):
	x_features, y_label, numeric_features = load_dataset(path)
	x_features_normal = data_normalization(path, numeric_features, x_features) #TODOEduarda ReviewThis
	
	return x_features_normal, y_label #Atualizar com o data augmentation
	

''' ************************************** DATASET LOADING *************************************************'''
# Data Upload based on https://www.tensorflow.org/tutorials/load_data/csv

def load_dataset(path):
	hrv_data_path = path
	numeric_features = ['hr_mean','nni_mean','nni_min','nni_max','sdnn','rmssd','nn50','pnn50','LF/HF ratio','Total power']
	#print(hrv_data_path, flush=True)
	data = pd.read_csv(hrv_data_path)
	#print(data)

	x_features_df = data.iloc[:,2:12]
	#print(x)
	y_label_df = data.iloc[:,1]
	#print(y)

	x_features = x_features_df.to_numpy()
	y_label = y_label_df.to_numpy()
	print(x_features)
	print(y_label)
	return x_features, y_label, numeric_features

''' ************************************** DATA NORMALIZATION *************************************************'''

def data_normalization(hrv_data_path, numeric_features, x_features):
	desc = pd.read_csv(hrv_data_path)[numeric_features].describe()
	desc
	mean = np.array(desc.T['mean'])
	std = np.array(desc.T['std'])
	  
	x_features_normal = (x_features-mean)/std
	#print(x_features_normal)
	return x_features_normal


''' ************************************** DATA AUGMENTATION *************************************************'''
#The Random Over Sampling technique was used
