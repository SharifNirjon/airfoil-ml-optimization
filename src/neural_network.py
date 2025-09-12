"""
Airfoil Neural Network Optimization - 8 Architecture Comparison
Author: Sharif Nirjon
Date: September 2025
Framework: TensorFlow/Keras
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class AirfoilNeuralNetworks:
    """
    Comprehensive comparison of 8 neural network architectures for airfoil performance prediction
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.results = []
        self.models = {}
        self.histories = {}
        
        # Create directories
        os.makedirs('models/saved_models', exist_ok=True)
        os.makedirs('logs/tensorboard', exist_ok=True)
        
    def build_dense_network(self, input_shape):
        """
        1. Dense Neural Network (DNN) - Standard fully connected
        """
        model = models.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(128, activation='relu', name='dense1'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu', name='dense2'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu', name='dense3'),
            layers.Dense(1, activation='linear', name='output')
        ], name='Dense_Network')
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def build_wide_deep_network(self, input_shape):
        """
        2. Wide & Deep Network - Combines wide linear and deep components
        """
        # Input
        input_layer = layers.Input(shape=(input_shape,), name='input')
        
        # Wide component (linear)
        wide = layers.Dense(1, activation='linear', name='wide_output')(input_layer)
        
        # Deep component
        deep = layers.Dense(128, activation='relu')(input_layer)
        deep = layers.BatchNormalization()(deep)
        deep = layers.Dropout(0.3)(deep)
        deep = layers.Dense(64, activation='relu')(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.Dropout(0.3)(deep)
        deep = layers.Dense(32, activation='relu')(deep)
        deep_output = layers.Dense(1, activation='linear')(deep)
        
        # Combine wide and deep
        output = layers.Add(name='wide_deep_output')([wide, deep_output])
        
        model = models.Model(inputs=input_layer, outputs=output, name='Wide_Deep_Network')
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def build_residual_network(self, input_shape):
        """
        3. Residual Network (ResNet) - Skip connections for deeper networks
        """
        input_layer = layers.Input(shape=(input_shape,))
        
        # First block
        x = layers.Dense(128, activation='relu')(input_layer)
        x = layers.BatchNormalization()(x)
        
        # Residual block 1
        residual = x
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='linear')(x)
        x = layers.Add()([x, residual])  # Skip connection
        x = layers.ReLU()(x)
        
        # Residual block 2
        residual = x
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='linear')(x)
        x = layers.Add()([x, residual])  # Skip connection
        x = layers.ReLU()(x)
        
        # Output
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1, activation='linear')(x)
        
        model = models.Model(inputs=input_layer, outputs=output, name='Residual_Network')
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def build_dropout_network(self, input_shape):
        """
        4. Heavy Dropout Network - Strong regularization
        """
        model = models.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='linear')
        ], name='Dropout_Network')
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def build_cnn_network(self, input_shape):
        """
        5. 1D Convolutional Neural Network - For sequence-like features
        """
        model = models.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Reshape((input_shape, 1)),  # Add channel dimension
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(32, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalMaxPooling1D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='linear')
        ], name='CNN_Network')
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def build_lstm_network(self, input_shape):
        """
        6. LSTM Network - For temporal patterns (treating features as sequence)
        """
        model = models.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Reshape((input_shape, 1)),  # Add time dimension
            layers.LSTM(64, return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='linear')
        ], name='LSTM_Network')
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def build_transformer_network(self, input_shape):
        """
        7. Transformer Network - Self-attention mechanisms
        """
        # Multi-head attention layer
        class MultiHeadAttention(layers.Layer):
            def __init__(self, d_model, num_heads):
                super(MultiHeadAttention, self).__init__()
                self.num_heads = num_heads
                self.d_model = d_model
                self.depth = d_model // num_heads
                
                self.wq = layers.Dense(d_model)
                self.wk = layers.Dense(d_model)
                self.wv = layers.Dense(d_model)
                self.dense = layers.Dense(d_model)
                
            def call(self, inputs):
                batch_size = tf.shape(inputs)[0]
                seq_len = tf.shape(inputs)[1]
                
                q = self.wq(inputs)
                k = self.wk(inputs)
                v = self.wv(inputs)
                
                # Scaled dot-product attention
                attention_weights = tf.nn.softmax(
                    tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.d_model, tf.float32))
                )
                attention_output = tf.matmul(attention_weights, v)
                
                return self.dense(attention_output)
        
        input_layer = layers.Input(shape=(input_shape,))
        x = layers.Reshape((input_shape, 1))(input_layer)
        x = layers.Dense(64)(x)  # Project to d_model
        
        # Multi-head attention
        attention_output = MultiHeadAttention(64, 4)(x)
        x = layers.Add()([x, attention_output])  # Residual connection
        x = layers.LayerNormalization()(x)
        
        # Feed forward
        ff_output = layers.Dense(128, activation='relu')(x)
        ff_output = layers.Dense(64)(ff_output)
        x = layers.Add()([x, ff_output])  # Residual connection
        x = layers.LayerNormalization()(x)
        
        # Global pooling and final layers
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(1, activation='linear')(x)
        
        model = models.Model(inputs=input_layer, outputs=output, name='Transformer_Network')
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def build_ensemble_network(self, input_shape):
        """
        8. Ensemble Network - Multiple sub-networks combined
        """
        input_layer = layers.Input(shape=(input_shape,))
        
        # Sub-network 1: Dense
        sub1 = layers.Dense(64, activation='relu')(input_layer)
        sub1 = layers.Dense(32, activation='relu')(sub1)
        sub1 = layers.Dense(1, activation='linear')(sub1)
        
        # Sub-network 2: Wide
        sub2 = layers.Dense(128, activation='relu')(input_layer)
        sub2 = layers.Dropout(0.3)(sub2)
        sub2 = layers.Dense(1, activation='linear')(sub2)
        
        # Sub-network 3: Deep narrow
        sub3 = layers.Dense(32, activation='relu')(input_layer)
        sub3 = layers.Dense(32, activation='relu')(sub3)
        sub3 = layers.Dense(32, activation='relu')(sub3)
        sub3 = layers.Dense(1, activation='linear')(sub3)
        
        # Ensemble combination
        ensemble_output = layers.Average()([sub1, sub2, sub3])
        
        model = models.Model(inputs=input_layer, outputs=ensemble_output, name='Ensemble_Network')
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def evaluate_architecture(self, model, X_train, X_val, y_train, y_val, name, epochs=100):
        """
        Train and evaluate a single neural network architecture
        """
        print(f"   🧠 Training {name}...")
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )
        
        tensorboard_callback = callbacks.TensorBoard(
            log_dir=f'logs/tensorboard/{name}', histogram_freq=1
        )
        
        # Training
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr, tensorboard_callback],
            verbose=0
        )
        training_time = time.time() - start_time
        
        # Predictions
        start_time = time.time()
        y_pred = model.predict(X_val, verbose=0)
        prediction_time = time.time() - start_time
        
        # Metrics
        r2 = r2_score(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        
        # Model complexity
        total_params = model.count_params()
        
        # Save model
        model.save(f'models/saved_models/{name}.h5')
        
        # Store history
        self.histories[name] = history
        
        result = {
            'Architecture': name,
            'R2_Score': r2,
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'Training_Time': training_time,
            'Prediction_Time': prediction_time,
            'Total_Parameters': total_params,
            'Final_Epoch': len(history.history['loss']),
            'Best_Val_Loss': min(history.history['val_loss'])
        }
        
        print(f"      ✅ {name}: R² = {r2:.4f}, RMSE = {rmse:.4f}, "
              f"Params = {total_params:,}, Time = {training_time:.1f}s")
        
        return result
    
    def compare_architectures(self, X, y, test_size=0.2, val_size=0.2):
        """
        Compare all 8 neural network architectures
        """
        print("🧠 Starting Neural Network Architecture Comparison...")
        print(f"📊 Dataset shape: {X.shape}")
        print(f"🎯 Target variable (L/D ratio) range: {y.min():.2f} to {y.max():.2f}")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        input_shape = X_train_scaled.shape[1]
        
        # Define all architectures
        architectures = {
            'Dense_Network': self.build_dense_network,
            'Wide_Deep_Network': self.build_wide_deep_network,
            'Residual_Network': self.build_residual_network,
            'Dropout_Network': self.build_dropout_network,
            'CNN_Network': self.build_cnn_network,
            'LSTM_Network': self.build_lstm_network,
            'Transformer_Network': self.build_transformer_network,
            'Ensemble_Network': self.build_ensemble_network
        }
        
        results = []
        print(f"\n🚀 Training {len(architectures)} neural network architectures...")
        
        for name, build_func in architectures.items():
            try:
                # Build model
                model = build_func(input_shape)
                self.models[name] = model
                
                # Train and evaluate
                result = self.evaluate_architecture(
                    model, X_train_scaled, X_val_scaled, y_train, y_val, name
                )
                results.append(result)
                
            except Exception as e:
                print(f"      ❌ Error training {name}: {str(e)}")
                continue
        
        # Convert to DataFrame and sort
        self.results_df = pd.DataFrame(results)
        self.results_df = self.results_df.sort_values('R2_Score', ascending=False)
        
        print(f"\n🏆 Top performing architecture: {self.results_df.iloc[0]['Architecture']} "
              f"(R² = {self.results_df.iloc[0]['R2_Score']:.4f})")
        
        return self.results_df
    
    def save_results(self, filename='results/architecture_performance.csv'):
        """
        Save results to CSV file
        """
        os.makedirs('results', exist_ok=True)
        self.results_df.to_csv(filename, index=False)
        print(f"📁 Results saved to {filename}")

if __name__ == "__main__":
    # Test with synthetic data
    print("🧪 Testing Neural Network Architectures with synthetic data...")
    
    # Generate synthetic airfoil data
    np.random.seed(42)
    n_samples = 2000
    n_features = 8
    
    X = np.random.randn(n_samples, n_features)
    y = 2 * X[:, 0] - 1.5 * X[:, 1] + 0.8 * X[:, 2] + np.random.normal(0, 0.1, n_samples)
    
    # Initialize and run comparison
    nn_comp = AirfoilNeuralNetworks()
    results = nn_comp.compare_architectures(X, y)
    
    print("\n📊 Final Architecture Comparison:")
    print(results[['Architecture', 'R2_Score', 'RMSE', 'Total_Parameters', 'Training_Time']].round(4))
