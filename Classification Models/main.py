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
    x_features_normal, y_label = load_features(path)
    
    #Model used
    model_name = "model2" #model1, model2, model3, weightsAlreadyCalculated
    
    #Train Settings
    batch_size = 64
    num_epochs = 100
            
    #Spliting Data
    x_train, x_test, y_train, y_test = train_test_split(x_features_normal, y_label, test_size = 0.2)
            
    #Create Model
    model = build_model(model_name)
    
    #Train Model
    #opt = SGD(learning_rate=0.0001, decay=0.005, momentum=0.9, nesterov=True) #decay is deprecated in the new Keras optimizer
    opt = SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)
    metrics = ['accuracy', tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name='recall')]
    loss = "binary_crossentropy"
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    model.fit(model, batch_size, num_epochs, x_train, x_test, y_train, y_test) #TODOEduarda 'Sequential' object has no attribute 'train_evalute_model'/ etc
    model.summary()
