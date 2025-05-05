import os
import numpy as np
import tensorflow as tf
import yaml
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import h5py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate mean absolute percentage error, handling zero values appropriately
    """
    # Create a mask for non-zero values to avoid division by zero
    non_zero_mask = tf.not_equal(y_true, 0)
    
    # Calculate absolute percentage error only for non-zero values
    absolute_percentage_error = tf.abs((y_true - y_pred) / (y_true + 1e-10))* 100.0  # Add small epsilon
    
    # Apply the mask and calculate mean
    return tf.reduce_mean(tf.boolean_mask(absolute_percentage_error, non_zero_mask))

def kl_divergence(y_true, y_pred):
    """
    Calculate KL divergence between y_true and y_pred, treating them as probability distributions
    with improved numerical stability
    """
    # Add small epsilon to prevent numerical issues
    epsilon = 1e-7
    
    # Check if inputs contain only zeros or negative values
    y_true_sum = tf.reduce_sum(y_true, axis=-1, keepdims=True)
    y_pred_sum = tf.reduce_sum(y_pred, axis=-1, keepdims=True)
    
    # Only normalize if sum is positive
    y_true_normalized = tf.where(
        y_true_sum > epsilon,
        y_true / (y_true_sum + epsilon),
        tf.ones_like(y_true) / tf.cast(tf.shape(y_true)[-1], tf.float32)  # Uniform distribution
    )
    
    y_pred_normalized = tf.where(
        y_pred_sum > epsilon,
        y_pred / (y_pred_sum + epsilon),
        tf.ones_like(y_pred) / tf.cast(tf.shape(y_pred)[-1], tf.float32)  # Uniform distribution
    )
    
    # TensorFlow has a more numerically stable KL implementation
    kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)(
        y_true_normalized, y_pred_normalized
    )
    
    # Return mean across batch
    return tf.reduce_mean(kl)

class MLModelTrainer:
    def __init__(self, config_path=None):
        if config_path is None:
            # Default to yaml/configML.yaml relative to the script's directory
            config_path = os.path.join(os.path.dirname(__file__), "../yaml/configML.yaml")
        
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.data_path = self.config["data"]["path"]
        self.h5_file = self.config["data"]["file"]
        self.x_dataset = self.config["data"]["x_dataset"]
        self.y_dataset = self.config["data"]["y_dataset"]
        
        # Get test size from config or use default
        self.test_size = self.config["data"].get("test_size", 0.1)
        
        # Optional: Add config for photons_per_event if you want to use it
        self.use_photon_counts = self.config["model"].get("use_photon_counts", False)

        self.learning_rate = self.config["model"]["learning_rate"]
        self.epochs = self.config["model"]["epochs"]
        self.batch_size = self.config["model"]["batch_size"]
        self.model_path = self.config["model"]["save_path"]
        
        # Get model architecture parameters
        self.hidden_layers = self.config["model"].get("hidden_layers", [64, 128, 128, 128])
        self.activation = self.config["model"].get("activation", "relu")

        self.output_dir = self.config["output"]["dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        self.X_train, self.X_test = None, None
        self.Y_train, self.Y_test = None, None
        self.photons_train, self.photons_test = None, None
        
        self.scaler_X, self.scaler_Y = StandardScaler(), StandardScaler()
        self.scaler_photons = StandardScaler() if self.use_photon_counts else None
        self.model = None
        
        print(f"Model will be trained using:")
        print(f" - X data from: {self.x_dataset}")
        print(f" - Y data from: {self.y_dataset}")

        if self.use_photon_counts:
            print(f" - Including photons per event as additional input feature")

    def load_data(self):
        print("\nLoading data")
        with h5py.File(os.path.join(self.data_path, self.h5_file), "r") as f:
            # Check if direct ML data is available in ml_data group
            if "ml_data" in f and "X" in f["ml_data"] and "Y" in f["ml_data"]:
                print(" Using pre-formatted ML data from ml_data group")
                X = f["ml_data/X"][:]
                Y = f["ml_data/Y"][:]
                print(f"Number of events in dataset: {X.shape[0]}")
                print(f"Number of channels in Y data: {Y.shape[1]}")
            else:
                # Use the specified datasets
                print(f" Using specified datasets: {self.x_dataset} and {self.y_dataset}")
                X = f[self.x_dataset][:]
                Y = f[self.y_dataset][:]
            
            # Optionally load photons per event if needed for the model
            if self.use_photon_counts:
                photons = f["events/photons_per_event"][:]
                photons = photons.reshape(-1, 1)  # Reshape to 2D for sklearn
                print(f" Also loaded photons_per_event with shape {photons.shape}")
        
        print(f" Data loaded: X shape {X.shape}, Y shape {Y.shape}")
        
        # Check if Y contains normalized ratios (values should be small)
        y_max = np.max(Y)
        if y_max > 1.0:
            print(f" Warning: Y data has maximum value of {y_max}, which suggests it might not be normalized ratios")
        else:
            print(f" Y data appears to be normalized (max value: {y_max:.4f})")
        
        if self.use_photon_counts:
            # Split including photons per event
            (self.X_train, self.X_test, 
             self.Y_train, self.Y_test,
             self.photons_train, self.photons_test) = train_test_split(
                X, Y, photons, test_size=self.test_size, random_state=42
            )
            # Scale photons per event if used
            self.photons_train = self.scaler_photons.fit_transform(self.photons_train)
            self.photons_test = self.scaler_photons.transform(self.photons_test)
        else:
            # Standard split
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
                X, Y, test_size=self.test_size, random_state=42
            )
        
        # Standard scaling for X and Y
        self.X_train = self.scaler_X.fit_transform(self.X_train)
        self.X_test = self.scaler_X.transform(self.X_test)
        
        # For ratio data, we might want to avoid scaling Y since ratios are already normalized
        # but this depends on the specific requirements of the model
        if self.config["model"].get("scale_y", True):
            self.Y_train = self.scaler_Y.fit_transform(self.Y_train)
            self.Y_test = self.scaler_Y.transform(self.Y_test)
            print(" Applied scaling to Y data")
        else:
            print(" Skipped scaling for Y data as it's already normalized")
            
        print(f" Train/test split: {len(self.X_train)} training samples, {len(self.X_test)} test samples")

    def save_scalers(self):
        scaler_path = os.path.join(self.output_dir, "scalers")
        os.makedirs(scaler_path, exist_ok=True)
        
        joblib.dump(self.scaler_X, os.path.join(scaler_path, "scaler_X.pkl"))
        
        if self.config["model"].get("scale_y", True):
            joblib.dump(self.scaler_Y, os.path.join(scaler_path, "scaler_Y.pkl"))
            
        if self.use_photon_counts:
            joblib.dump(self.scaler_photons, os.path.join(scaler_path, "scaler_photons.pkl"))
            
        print(f" Scalers saved to {scaler_path}")

    def build_model(self):
        print("\nBuilding model")
        
        # Get the output dimension from Y shape
        output_dim = self.Y_train.shape[1]
        
        # Get L2 regularization parameter
        l2_reg = self.config["model"].get("l2_regularization", 0)
        
        if self.use_photon_counts:
            # Create a model that takes both position and photon count as input
            input_pos = Input(shape=(3,), name="position_input")
            input_photons = Input(shape=(1,), name="photon_count_input")
            
            # Position branch with L2 regularization
            x = Dense(self.hidden_layers[0], activation=self.activation, 
                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(input_pos)
            for units in self.hidden_layers[1:-1]:
                x = Dense(units, activation=self.activation,
                         kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
            
            # Combine with photon count
            combined = Concatenate()([x, input_photons])
            
            # Additional layers after combining
            x = Dense(self.hidden_layers[-1], activation=self.activation,
                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(combined)
            outputs = Dense(output_dim)(x)
            
            self.model = Model(inputs=[input_pos, input_photons], outputs=outputs)
        else:
            # Standard sequential model with just position input
            layers = [Dense(self.hidden_layers[0], input_dim=3, activation=self.activation,
                            kernel_regularizer=tf.keras.regularizers.l2(l2_reg))]
            
            # Get dropout rate from config
            dropout_rate = self.config["model"].get("dropout_rate", 0)
            
            # Add remaining hidden layers with dropout
            for units in self.hidden_layers[1:]:
                layers.append(Dense(units, activation=self.activation,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
                if dropout_rate > 0:
                    layers.append(tf.keras.layers.Dropout(dropout_rate))
                    
            # Final output layer
            layers.append(Dense(output_dim))
            
            self.model = Sequential(layers)

        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, 
                           loss='mse', 
                           metrics=['mae', mean_absolute_percentage_error, kl_divergence])
        self.model.summary()
        print(f" Model built with {len(self.hidden_layers)} hidden layers and {output_dim} output neurons")

    def train_model(self):
        print("\nTraining model")
        
        # Create callbacks list
        callbacks = []
        
        # Add learning rate scheduler if enabled
        if self.config["model"].get("learning_rate_scheduler", False):
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config["model"].get("lr_reduction_factor", 0.5),
                patience=self.config["model"].get("lr_patience", 10),
                min_lr=self.config["model"].get("min_learning_rate", 0.00001),
                verbose=1
            )
            callbacks.append(reduce_lr)
        
        # Add callbacks to fit function
        if self.use_photon_counts:
            history = self.model.fit(
                [self.X_train, self.photons_train], self.Y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.15,
                verbose=1,
                callbacks=callbacks  # Add this line
            )
        else:
            history = self.model.fit(
                self.X_train, self.Y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.15,
                verbose=1,
                callbacks=callbacks  # Add this line
            )
            
        print(" Training completed")
        
        # Plot training history
        self.plot_history(history)
        
        return history

    def plot_history(self, history):
        plot_dir = os.path.join(os.path.dirname(__file__), "../plot")
        os.makedirs(plot_dir, exist_ok=True)

        plt.figure(figsize=(12, 10))
        
        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        
        # Plot MAE
        plt.subplot(2, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.yscale('log')
        plt.legend()
        
        # Plot MAPE
        plt.subplot(2, 2, 3)
        plt.plot(history.history['mean_absolute_percentage_error'], label='Training MAPE')
        plt.plot(history.history['val_mean_absolute_percentage_error'], label='Validation MAPE')
        plt.title('Mean Absolute Percentage Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAPE (%)')
        plt.yscale('log')
        plt.legend()
        
        # Plot KL Divergence
        plt.subplot(2, 2, 4)
        plt.plot(history.history['kl_divergence'], label='Training KL Div')
        plt.plot(history.history['val_kl_divergence'], label='Validation KL Div')
        plt.title('KL Divergence')
        plt.xlabel('Epoch')
        plt.ylabel('KL Divergence')
        plt.yscale('log')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'training_history.png'))
        print(" Training history plot saved")

    def evaluate_model(self):
        print("\nEvaluating model")
        
        if self.use_photon_counts:
            # Unpack all four metrics: loss, mae, mape, and kl_divergence
            test_results = self.model.evaluate(
                [self.X_test, self.photons_test], self.Y_test
            )
            
            # Make predictions
            y_pred = self.model.predict([self.X_test, self.photons_test])
        else:
            # Unpack all four metrics: loss, mae, mape, and kl_divergence
            test_results = self.model.evaluate(self.X_test, self.Y_test)
            
            # Make predictions
            y_pred = self.model.predict(self.X_test)
        
        # Extract individual metrics from results
        test_loss = test_results[0]
        test_mae = test_results[1]
        test_mape = test_results[2]
        test_kl = test_results[3]
            
        print(f" Test MAE: {test_mae:.4f}, Test MAPE: {test_mape:.2f}%, Test Loss: {test_loss:.4f}, KL Div: {test_kl:.4f}")
        
        # Plot some example predictions vs actual
        self.plot_predictions(self.Y_test, y_pred)
        
        # Plot relative errors
        self.plot_relative_errors(self.Y_test, y_pred)

    def plot_predictions(self, y_true, y_pred, num_samples=5, num_channels=10):
        plot_dir = os.path.join(os.path.dirname(__file__), "../plot")
        os.makedirs(plot_dir, exist_ok=True)

        plt.figure(figsize=(15, 10))
        
        for i in range(min(num_samples, len(y_true))):
            plt.subplot(num_samples, 1, i+1)
            
            # Select first few channels to display
            x = np.arange(num_channels)
            plt.plot(x, y_true[i][:num_channels], 'b-', label='Actual')
            plt.plot(x, y_pred[i][:num_channels], 'r--', label='Predicted')
            
            plt.title(f'Sample {i+1}: Actual vs Predicted')
            plt.xlabel('Channel')
            plt.ylabel('Photon Ratio')
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'prediction_samples.png'))
        print(f" Prediction plot saved with {num_samples} samples and {num_channels} channels each")

    def plot_relative_errors(self, y_true, y_pred):
        plot_dir = os.path.join(os.path.dirname(__file__), "../plot")
        os.makedirs(plot_dir, exist_ok=True)

        # Calculate relative errors, avoiding division by zero
        non_zero_mask = y_true != 0
        relative_errors = np.zeros_like(y_true)
        relative_errors[non_zero_mask] = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
        
        # Calculate average relative error per channel
        avg_channel_errors = np.mean(relative_errors, axis=0) * 100  # Convert to percentage
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(avg_channel_errors)), avg_channel_errors)
        plt.xlabel('Channel')
        plt.ylabel('Average Percentage Error (%)')
        plt.title('Mean Percentage Error by Channel')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'channel_percentage_errors.png'))
        print(" Channel percentage error plot saved")

    def save_model(self):
        # Save model architecture and weights
        self.model.save(self.model_path)
        
        # Save model config
        model_info = {
            'input_shape': (3,) if not self.use_photon_counts else [(3,), (1,)],
            'output_shape': (self.Y_train.shape[1],),
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'use_photon_counts': self.use_photon_counts,
            'scale_y': self.config["model"].get("scale_y", True)
        }
        
        with open(os.path.join(self.output_dir, 'model_config.yaml'), 'w') as f:
            yaml.dump(model_info, f)
            
        print(f" Model saved at {self.model_path}")
        print(f" Model configuration saved at {os.path.join(self.output_dir, 'model_config.yaml')}")

    def run(self):
        self.load_data()
        self.build_model()
        history = self.train_model()
        self.evaluate_model()
        self.save_model()
        self.save_scalers()

if __name__ == "__main__":
    trainer = MLModelTrainer()
    trainer.run()