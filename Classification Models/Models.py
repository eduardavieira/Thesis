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
from tensorflow.keras.layers import Bidirectional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

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
    #class_weight_dict[1] *= 1.5  # Boost stress detection
    print(f"Class weights: {class_weight_dict}\n")

    original_stress_weight = class_weight_dict.get(1, 1.0)
    class_weight_dict[1] = original_stress_weight * 1.2
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=30, #Use 20 for cnn/lstm models
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15, #Use 10 for cnn/lstm models
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

def build_model(model_name, input_shape):
    """Build model by name"""
    models = {
        'model1_new': model1_new,
        'model2_new': model2_new,
        'model3_new': model3_new,
        'train_and_ensemble': train_and_ensemble,
        'model4_cnn_lstm_hybrid': model4_cnn_lstm_hybrid,
        'model2_simple': model2_simple,
        'model5_interpretable_transformer': model5_interpretable_transformer,
        'model5_interpretable_transformer_lite': model5_interpretable_transformer_lite,
        'model5_cnn_transformer_hybrid': model5_cnn_transformer_hybrid,
        'model5_cnn_transformer_hybrid_v2': model5_cnn_transformer_hybrid_v2,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    model_fn = models[model_name]
    model = model_fn(input_shape)
    
    return model

'''      *********************      MODEL TRANSFORMER (JANUARY 2026)           *****************'''
class PositionalEncoding(Layer):
    """
    Positional encoding for temporal information
    Essential for transformers to understand sequence order
    """
    def __init__(self, d_model, max_len=100, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        
    def build(self, input_shape):
        # Create positional encoding matrix
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pos_encoding = np.zeros((self.max_len, self.d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        self.pos_encoding = tf.constant(pos_encoding[np.newaxis, :, :], dtype=tf.float32)
        
    def call(self, x):
        # Add positional encoding to input
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

def model5_cnn_transformer_hybrid_v2(input_shape):
    """
    Optimized CNN-Transformer Hybrid
    Slightly more capacity, better regularization
    """
    from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout
    from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
    from tensorflow.keras.layers import GlobalAveragePooling1D, Dense, MaxPooling1D
    from tensorflow.keras.models import Model
    
    inputs = Input(shape=input_shape, name='hrv_input')
    
    # ============ ENHANCED CNN FEATURE EXTRACTION ============
    # First conv block
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same', name='conv1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = Dropout(0.2, name='dropout_conv1')(x)
    
    # Second conv block (deeper)
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Dropout(0.2, name='dropout_conv2')(x)
    
    # Third conv block (even deeper)
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same', name='conv3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = Dropout(0.25, name='dropout_conv3')(x)
    
    # ============ TRANSFORMER ATTENTION ============
    # Add positional encoding (matching dimension)
    x = PositionalEncoding(128, max_len=input_shape[0], name='positional_encoding')(x)
    
    # Multi-head attention
    attn_output = MultiHeadAttention(
        num_heads=8,  # ← Increased from 4
        key_dim=16,
        dropout=0.2,
        name='attention'
    )(x, x)
    
    # Add & Norm
    x = LayerNormalization(epsilon=1e-6, name='norm1')(x + attn_output)
    
    # Feed-forward network (wider)
    ff_output = Dense(256, activation='relu', name='ffn1')(x)  # ← Increased from 128
    ff_output = Dropout(0.3, name='dropout_ffn')(ff_output)
    ff_output = Dense(128, name='ffn2')(ff_output)
    
    # Add & Norm
    x = LayerNormalization(epsilon=1e-6, name='norm2')(x + ff_output)
    x = Dropout(0.3, name='dropout_transformer')(x)
    
    # ============ CLASSIFICATION HEAD ============
    # Global average pooling
    x = GlobalAveragePooling1D(name='global_pooling')(x)
    
    # Deeper classification head
    x = Dense(64, activation='relu', name='dense1')(x)  # ← Increased from 32
    x = BatchNormalization(name='bn_dense')(x)
    x = Dropout(0.4, name='dropout_dense1')(x)
    
    x = Dense(32, activation='relu', name='dense2')(x)
    x = Dropout(0.4, name='dropout_dense2')(x)
    
    outputs = Dense(1, activation='sigmoid', name='stress_prediction')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='CNN_Transformer_Hybrid_V2')
    
    return model

def model5_cnn_transformer_hybrid(input_shape):
    """
    CNN-Transformer Hybrid
    CNN extracts features, Transformer adds attention
    Best for small datasets with short sequences
    """
    from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout
    from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
    from tensorflow.keras.layers import GlobalAveragePooling1D, Dense
    from tensorflow.keras.models import Model
    
    inputs = Input(shape=input_shape, name='hrv_input')
    
    # ============ CNN FEATURE EXTRACTION ============
    # First conv block
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same', name='conv1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = Dropout(0.2, name='dropout_conv1')(x)
    
    # Second conv block
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Dropout(0.2, name='dropout_conv2')(x)
    
    # ============ TRANSFORMER ATTENTION ============
    # Add positional encoding
    x = PositionalEncoding(64, max_len=input_shape[0], name='positional_encoding')(x)
    
    # Multi-head attention
    attn_output = MultiHeadAttention(
        num_heads=4,
        key_dim=16,
        dropout=0.2,
        name='attention'
    )(x, x)
    
    # Add & Norm
    x = LayerNormalization(epsilon=1e-6, name='norm1')(x + attn_output)
    
    # Feed-forward network
    ff_output = Dense(128, activation='relu', name='ffn1')(x)
    ff_output = Dropout(0.3, name='dropout_ffn')(ff_output)
    ff_output = Dense(64, name='ffn2')(ff_output)
    
    # Add & Norm
    x = LayerNormalization(epsilon=1e-6, name='norm2')(x + ff_output)
    x = Dropout(0.3, name='dropout_transformer')(x)
    
    # ============ CLASSIFICATION HEAD ============
    # Global average pooling
    x = GlobalAveragePooling1D(name='global_pooling')(x)
    
    # Dense layers
    x = Dense(32, activation='relu', name='dense1')(x)
    x = Dropout(0.4, name='dropout_dense')(x)
    
    outputs = Dense(1, activation='sigmoid', name='stress_prediction')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='CNN_Transformer_Hybrid')
    
    return model


def model5_interpretable_transformer_lite(input_shape):
    """
    Lighter transformer for small datasets
    """
    
    # Reduced dimensions
    d_model = 32  # Down from 64
    num_heads = 2  # Down from 4
    num_layers = 1  # Down from 2
    
    # Input
    inputs = Input(shape=input_shape, name='hrv_input')
    
    # Project to d_model
    x = Dense(d_model, name='input_projection')(inputs)
    
    # Positional encoding
    x = PositionalEncoding(d_model, max_len=input_shape[0], name='positional_encoding')(x)
    
    # Single transformer block
    attn_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=0.2,  # More dropout
        name='attention_layer'
    )(x, x)
    
    x = LayerNormalization(epsilon=1e-6, name='norm1')(x + attn_output)
    
    # Feed-forward
    ff = Dense(d_model * 2, activation='relu', name='ffn1')(x)
    ff = Dropout(0.3, name='dropout_ffn')(ff)
    ff = Dense(d_model, name='ffn2')(ff)
    
    x = LayerNormalization(epsilon=1e-6, name='norm2')(x + ff)
    x = Dropout(0.4, name='dropout_main')(x)
    
    # Global pooling
    x = GlobalAveragePooling1D(name='global_pooling')(x)
    
    # Simpler classification head
    x = Dense(16, activation='relu', name='dense1')(x)
    x = Dropout(0.5, name='dropout_final')(x)
    outputs = Dense(1, activation='sigmoid', name='stress_prediction')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='TransformerLite')
    
    return model

def model5_interpretable_transformer(input_shape, num_heads=4, d_model=64, num_layers=2):
    """
    Interpretable Transformer for Stress Detection
    
    Args:
        input_shape: (sequence_length, features) e.g., (8, 1)
        num_heads: Number of attention heads
        d_model: Model dimension
        num_layers: Number of transformer blocks
    """
    
    # Input
    inputs = Input(shape=input_shape, name='hrv_input')
    
    # Project to d_model dimensions
    x = Dense(d_model, name='input_projection')(inputs)
    
    # Add positional encoding
    x = PositionalEncoding(d_model, max_len=input_shape[0], name='positional_encoding')(x)
    
    # Transformer blocks
    for i in range(num_layers):
        # Multi-head self-attention
        attn_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=0.1,
            name=f'attention_layer_{i}'
        )(x, x)
        
        # Add & Norm
        x = LayerNormalization(epsilon=1e-6, name=f'norm1_{i}')(x + attn_output)
        
        # Feed-forward network
        ff_output = Dense(d_model * 4, activation='relu', name=f'ffn1_{i}')(x)
        ff_output = Dropout(0.1, name=f'dropout_ffn1_{i}')(ff_output)
        ff_output = Dense(d_model, name=f'ffn2_{i}')(ff_output)
        
        # Add & Norm
        x = LayerNormalization(epsilon=1e-6, name=f'norm2_{i}')(x + ff_output)
        x = Dropout(0.3, name=f'dropout_block_{i}')(x)
    
    # Global pooling
    x = GlobalAveragePooling1D(name='global_pooling')(x)
    
    # Classification
    x = Dense(32, activation='relu', name='dense1')(x)
    x = Dropout(0.4, name='dropout_final')(x)
    outputs = Dense(1, activation='sigmoid', name='stress_prediction')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='InterpretableTransformer')
    
    return model

'''      *********************      MODELS FIXED (DECEMBER 2025)           *****************'''
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
    """Improved LSTM with more capacity"""
    model = Sequential([
        # Stacked LSTM layers
        LSTM(128, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, activation='relu', return_sequences=True),
        Dropout(0.3),
        LSTM(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    return model

def model2_simple(input_shape):
    """
    Simplified LSTM - easier to train, more stable
    """
    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape),
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

def model4_cnn_lstm_hybrid(input_shape):
    """
    CNN-LSTM Hybrid: CNN extracts features, LSTM models sequences
    """
    model = Sequential([
        # CNN layers for feature extraction
        Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.3),
        
        # LSTM for temporal modeling
        LSTM(64, activation='relu'),
        Dropout(0.4),
        
        # Classification
        Dense(32, activation='relu'),
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

def train_and_ensemble(x_train, x_test, y_train, y_test, input_shape):
    """
    Train all three models and create ensemble predictions
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    import tensorflow
    
    print("\n" + "="*60)
    print("TRAINING ENSEMBLE OF 3 MODELS")
    print("="*60)
    
    # Configuration for each model
    models_config = {
        'Model1_CNN': {
            'builder': model1_new,
            'lr': 0.001,
            'epochs': 150,
            'batch_size': 32
        },
        'Model2_LSTM': {
            'builder': model2_new,
            'lr': 0.0003,
            'epochs': 150,
            'batch_size': 32
        },
        'Model3_CNN': {
            'builder': model3_new,
            'lr': 0.001,
            'epochs': 150,
            'batch_size': 32
        }
    }
    
    trained_models = {}
    test_predictions = {}
    test_accuracies = {}
    
    # Train each model
    for model_name, config in models_config.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}\n")
        
        # Build model
        model = config['builder'](input_shape)
        
        # Compile
        opt = Adam(learning_rate=config['lr'], clipnorm=1.0)
        metrics = [
            'accuracy',
            tensorflow.keras.metrics.Precision(name="precision"),
            tensorflow.keras.metrics.Recall(name='recall')
        ]
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=metrics)
        
        # Class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        
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
        
        # Train
        history = model.fit(
            x_train, y_train,
            validation_split=0.2,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Evaluate
        test_results = model.evaluate(x_test, y_test, verbose=0)
        test_loss = test_results[0]
        test_acc = test_results[1]
        test_prec = test_results[2] if len(test_results) > 2 else None
        test_rec = test_results[3] if len(test_results) > 3 else None
        
        print(f"\n{model_name} Test Results:")
        print(f"  Loss:      {test_loss:.4f}")
        print(f"  Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
        if test_prec:
            print(f"  Precision: {test_prec:.4f}")
        if test_rec:
            print(f"  Recall:    {test_rec:.4f}")
        
        # Store
        trained_models[model_name] = model
        test_predictions[model_name] = model.predict(x_test, verbose=0)
        test_accuracies[model_name] = test_acc
    
    # Ensemble predictions
    print(f"\n{'='*60}")
    print("CREATING ENSEMBLE PREDICTIONS")
    print(f"{'='*60}\n")
    
    # Show prediction statistics
    for model_name, preds in test_predictions.items():
        print(f"{model_name}:")
        print(f"  Mean: {preds.mean():.3f}, Std: {preds.std():.3f}")
        print(f"  Min: {preds.min():.3f}, Max: {preds.max():.3f}")
        print(f"  Accuracy: {test_accuracies[model_name]:.4f}\n")
    
    # Different ensemble strategies
    pred1 = test_predictions['Model1_CNN']
    pred2 = test_predictions['Model2_LSTM']
    pred3 = test_predictions['Model3_CNN']
    
    ensemble_strategies = {
        'Simple Average': np.mean([pred1, pred2, pred3], axis=0),
        
        'Weighted by Accuracy': (
            test_accuracies['Model1_CNN'] * pred1 +
            test_accuracies['Model2_LSTM'] * pred2 +
            test_accuracies['Model3_CNN'] * pred3
        ) / sum(test_accuracies.values()),
        
        'Best Model Double Weight (Model2)': (pred1 + 2*pred2 + pred3) / 4,
        
        'Majority Vote': (
            (pred1 > 0.5).astype(float) +
            (pred2 > 0.5).astype(float) +
            (pred3 > 0.5).astype(float)
        ) / 3,
        
        'Max Confidence': np.maximum(np.maximum(pred1, pred2), pred3),
        
        'Min Confidence': np.minimum(np.minimum(pred1, pred2), pred3)
    }
    
    # Evaluate each strategy
    print("="*60)
    print("EVALUATING ENSEMBLE STRATEGIES")
    print("="*60)
    
    results = []
    
    for strategy_name, ensemble_pred in ensemble_strategies.items():
        y_pred = (ensemble_pred > 0.5).astype(int).flatten()
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'strategy': strategy_name,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'predictions': ensemble_pred
        })
        
        print(f"\n{strategy_name}:")
        print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
    
    # Find best strategy
    best_by_accuracy = max(results, key=lambda x: x['accuracy'])
    best_by_f1 = max(results, key=lambda x: x['f1'])
    
    print(f"\n{'='*60}")
    print("🏆 BEST ENSEMBLE RESULTS")
    print(f"{'='*60}")
    
    print(f"\nBest by Accuracy: {best_by_accuracy['strategy']}")
    print(f"  Accuracy:  {best_by_accuracy['accuracy']:.4f} ({best_by_accuracy['accuracy']*100:.2f}%)")
    print(f"  Precision: {best_by_accuracy['precision']:.4f}")
    print(f"  Recall:    {best_by_accuracy['recall']:.4f}")
    print(f"  F1-Score:  {best_by_accuracy['f1']:.4f}")
    
    print(f"\nBest by F1-Score: {best_by_f1['strategy']}")
    print(f"  Accuracy:  {best_by_f1['accuracy']:.4f} ({best_by_f1['accuracy']*100:.2f}%)")
    print(f"  Precision: {best_by_f1['precision']:.4f}")
    print(f"  Recall:    {best_by_f1['recall']:.4f}")
    print(f"  F1-Score:  {best_by_f1['f1']:.4f}")
    
    # Show confusion matrix for best
    y_pred_best = (best_by_accuracy['predictions'] > 0.5).astype(int).flatten()
    
    print(f"\n{'='*60}")
    print(f"Confusion Matrix ({best_by_accuracy['strategy']}):")
    print(f"{'='*60}")
    print(confusion_matrix(y_test, y_pred_best))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best, target_names=['No Stress', 'Stress']))
    
    return trained_models, results, best_by_accuracy, best_by_f1