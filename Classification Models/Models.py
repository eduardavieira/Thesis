import tensorflow
#import keras
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Dropout, Flatten, LSTM
import tensorflow as tf
from livelossplot import PlotLossesKeras
from sklearn.model_selection import train_test_split

def build_model(model_name):
    if model_name == "model1":
        model = model1()
    elif model_name == "model2":
        model = model2()
    return model

def train_evalute_model(self, model, batch_size, num_epochs, x_train, x_test, y_train, y_test):
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=num_epochs, batch_size=batch_size, verbose=0, callbacks=[PlotLossesKeras()])
    #model.save_weights(("{}/{}_{}_{}").format(os.path.join(self.save_path, "WEIGHTS"), mode, prefix, "finalw.h5"))
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))




'''      *********************      MODELS            *****************'''
    
def model3(): #CNN
    model = tensorflow.keras.Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


def model1(): #CNN
    model = tensorflow.keras.Sequential()
    model.add(Conv1D(128, kernel_size=2, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(36, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


def model2(): #LSTM
    model = tensorflow.keras.Sequential()
    model.add(LSTM(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model
