import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_features(path):
	x_features, y_label, numeric_features = load_dataset(path)
	#x_features_normal = data_normalization(path, numeric_features, x_features) #Can't normalize here
	
	return x_features, y_label, numeric_features #TODOEduarda Atualizar com o data augmentation
	

''' ************************************** DATASET LOADING *************************************************'''
# Data Upload based on https://www.tensorflow.org/tutorials/load_data/csv

def load_dataset(path):
    """Load dataset from CSV and clean NaN values"""
    numeric_features = ['hr_mean','nni_mean','nni_min','nni_max','sdnn','rmssd','nn50','pnn50']
    
    data = pd.read_csv(path)
    
    print(f"Original data shape: {data.shape}")
    
    # Check for NaN in labels
    label_col = data.iloc[:,1]
    nan_labels = label_col.isna().sum()
    print(f"Rows with NaN labels: {nan_labels}")
    
    # Remove rows with NaN labels
    data = data.dropna(subset=[data.columns[1]])  # Drop rows where label is NaN
    
    # Check for NaN in features
    feature_cols = data.iloc[:,2:12]
    nan_features = feature_cols.isna().any(axis=1).sum()
    print(f"Rows with NaN features: {nan_features}")
    
    # Remove rows with NaN features
    data = data.dropna(subset=data.columns[2:12])
    
    print(f"Clean data shape: {data.shape}\n")
    
    x_features_df = data.iloc[:,2:12]
    y_label_df = data.iloc[:,1]
    
    x_features = x_features_df.to_numpy()
    y_label = y_label_df.to_numpy()
    
    # Final verification
    assert not np.isnan(x_features).any(), "Still have NaN in features!"
    assert not np.isnan(y_label).any(), "Still have NaN in labels!"
    
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

def normalize_data(x_train, x_test):
    """Normalize with outlier handling"""
    
    # Option A: Remove outliers BEFORE normalization
    print("\nOutlier Detection:")
    
    # Find outliers using IQR method
    Q1 = np.percentile(x_train, 25, axis=0)
    Q3 = np.percentile(x_train, 75, axis=0)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    # Clip to these bounds
    x_train_clipped = np.clip(x_train, lower_bound, upper_bound)
    x_test_clipped = np.clip(x_test, lower_bound, upper_bound)
    
    print(f"  Original train range: [{np.min(x_train):.2f}, {np.max(x_train):.2f}]")
    print(f"  Clipped train range: [{np.min(x_train_clipped):.2f}, {np.max(x_train_clipped):.2f}]")
    
    # NOW normalize
    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train_clipped)
    x_test_norm = scaler.transform(x_test_clipped)
    
    print(f"\nAfter normalization:")
    print(f"  Train range: [{np.min(x_train_norm):.2f}, {np.max(x_train_norm):.2f}]")
    print(f"  Test range: [{np.min(x_test_norm):.2f}, {np.max(x_test_norm):.2f}]")
    
    # Final safety clip
    x_train_norm = np.clip(x_train_norm, -5, 5)
    x_test_norm = np.clip(x_test_norm, -5, 5)
    
    assert not np.isnan(x_train_norm).any(), "NaN in training data!"
    assert not np.isnan(x_test_norm).any(), "NaN in test data!"
    
    return x_train_norm, x_test_norm


''' ************************************** DATA AUGMENTATION *************************************************'''
#The Random Over Sampling technique was used
#from imblearn.over_sampling import RandomOverSampler
#ros = RandomOverSampler(random_state=0)
#X_train_ros, y_train_ros = ros.fit_resample(X_train_pca, y_train)