import tensorflow
#import keras
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Dropout, Flatten, LSTM
import tensorflow as tf
from livelossplot import PlotLossesKeras
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, 
    MaxPooling1D, 
    LSTM, 
    Dense, 
    Dropout, 
    Flatten, 
    BatchNormalization
)
import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, 
    MaxPooling1D, 
    LSTM, 
    Dense, 
    Dropout, 
    Flatten, 
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def build_model(model_name, input_shape):
    try:
        model_fn = globals()[model_name] #in this way i dont have to put here the name of every model created, just need to call it on main
        model = model_fn(input_shape)
    except KeyError:
        raise ValueError(f"Unknown model: {model_name}")
    return model


#outdated version
#def train_evalute_model(model, batch_size, num_epochs, x_train, x_test, y_train, y_test):
#    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=num_epochs, batch_size=batch_size, verbose=0) #callbacks=[PlotLossesKeras()]
    #model.save_weights(("{}/{}_{}_{}").format(os.path.join(self.save_path, "WEIGHTS"), mode, prefix, "finalw.h5"))
#    test_loss, test_accuracy, *anything_else = model.evaluate(x_test, y_test, verbose=0)
#    print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

def train_evaluate_model(model, batch_size, num_epochs, x_train, x_test, y_train, y_test):
    """Train and evaluate with proper validation"""
    
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Training samples: {len(x_train)}")
    print(f"  Test samples: {len(x_test)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {num_epochs}")
    print(f"  Input shape: {x_train.shape}")
    print(f"{'='*60}\n")
    
    # Check labels
    print(f"Label distribution:")
    print(f"  Train: {np.bincount(y_train.astype(int))}")
    print(f"  Test: {np.bincount(y_test.astype(int))}\n")
    
    # Class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}\n")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train with validation split from training data
    print("Starting training...\n")
    history = model.fit(
        x_train, 
        y_train,
        validation_split=0.2,  # NOT validation_data=(x_test, y_test)!
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1  # Show progress!
    )
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60 + "\n")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = model.evaluate(x_test, y_test, verbose=1)
    
    print(f"\n{'='*60}")
    print("TEST SET RESULTS:")
    print(f"{'='*60}")
    print(f"  Loss:      {test_results[0]:.4f}")
    print(f"  Accuracy:  {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
    if len(test_results) > 2:
        print(f"  Precision: {test_results[2]:.4f}")
        print(f"  Recall:    {test_results[3]:.4f}")
    print(f"{'='*60}\n")
    
    # Detailed metrics
    y_pred_proba = model.predict(x_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Stress', 'Stress']))
    
    # Plot training history
    plot_history(history)
    
    return history


def plot_history(history):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss Over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Acc')
    axes[1].plot(history.history['val_accuracy'], label='Val Acc')
    axes[1].set_title('Accuracy Over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("\nTraining plot saved as 'training_history.png'\n")
    plt.show()



'''      *********************      MODELS FIXED            *****************'''
def model1_new(input_shape):
    """Improved CNN Model"""
    model = Sequential([
        # First conv block
        Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Second conv block
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Dense layers
        Flatten(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    return model


def model2_new(input_shape):
    """Improved LSTM Model"""
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model


def model3_new(input_shape):
    """Fixed CNN Model"""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Flatten(),  # CRITICAL: Must flatten before Dense!
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    return model


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
