from Models import * #Contains all the models
from Data import * #Contains data upload and processing
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam, SGD #Adagrad, Adadelta, Adamax, Nadam

if __name__ == '__main__':
    #Dataset Info
    #path = "/home/gsd/hrv_data/ProcessedECGData.csv"
    path = "/home/eduarda/Desktop/ProcessedECGData.csv"
    x_features, y_label, numeric_features = load_features(path) #vetor com vetores com as 12 features, corresponde a uma row do CSV [9.18096477e+01 7.18307692e+02 ... 6.80000000e+01 2.17479711e-01]
    
    print(f"Loaded data: {x_features.shape}")
    print(f"Features: {numeric_features}")
    print(f"Unique labels: {np.unique(y_label)}")
    #print(f"Label distribution: {np.bincount(y_label.astype(int))}\n")

    
    # Spliting Data
    x_train, x_test, y_train, y_test = train_test_split(
        x_features, 
        y_label, 
        test_size=0.2,
        random_state=42,
        stratify=y_label
    )

    #Spliting Data
    #x_train, x_test, y_train, y_test = train_test_split(x_features_normal, y_label, test_size = 0.2) # para test_size = 0.2 há 5277 entradas para treino e 1321 para teste

    print(f"After split:")
    print(f"  Train: {x_train.shape}")
    print(f"  Test: {x_test.shape}\n")

    # Normalize
    x_train_norm, x_test_norm = normalize_data(x_train, x_test)


    ''' ************************************** logica do claude *************************************************'''

    # Reshape for Conv1D/LSTM (samples, timesteps, features)
    nrows_train, ncols = x_train_norm.shape
    nrows_test = x_test_norm.shape[0]
    
    x_train_reshape = x_train_norm.reshape(nrows_train, ncols, 1)
    x_test_reshape = x_test_norm.reshape(nrows_test, ncols, 1)
    
    print(f"\nReshaped:")
    print(f"  Train: {x_train_reshape.shape}")
    print(f"  Test: {x_test_reshape.shape}\n")
    
    # Build model
    model_name = "model2_new"  # Try "model1_new", "model2_new", or "model3_new"
    input_shape = (ncols, 1)  # (10, 1) for your 10 features
    model = build_model(model_name, input_shape)
    
    # Compile with Adam optimizer
    opt = Adam(learning_rate=0.001, clipnorm=1.0)
    
    metrics = [
        'accuracy',
        tensorflow.keras.metrics.Precision(name="precision"),
        tensorflow.keras.metrics.Recall(name='recall')
    ]
    
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=metrics)
    
    print(f"Using {model_name}:")
    model.summary()
    print()


    ''' ************************************** logica do claude *************************************************'''
    
    #Train Settings
    batch_size = 32
    num_epochs = 100 #Test Accuracy 0.4431818127632141
            
    #Create Model
    #model = build_model(model_name)
    
    #Train Model
    #opt = SGD(learning_rate=0.0001, decay=0.005, momentum=0.9, nesterov=True) #decay is deprecated in the new Keras optimizer
    
#opt = SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)
#   metrics = ['accuracy', tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name='recall')]
#   loss = "binary_crossentropy"
#    model.compile(optimizer=opt, loss=loss, metrics=metrics)
#    nrows, ncols = x_train.shape # (entries,features)
#    x_train_reshape = x_train.reshape(nrows, ncols, 1) # (features, entries, 1)
#train_evalute_model(model, batch_size, num_epochs, x_train_reshape, x_test, y_train, y_test)
#    model.summary()

    history = train_evaluate_model(
        model, 
        batch_size, 
        num_epochs, 
        x_train_reshape, 
        x_test_reshape, 
        y_train, 
        y_test
    )
