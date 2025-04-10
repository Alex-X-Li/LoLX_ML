import os
import sys

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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.post_analysis import sipmid_to_merci
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define a metric for position error (Euclidean distance)
def euclidean_distance(y_true, y_pred):
    """
    Calculate average Euclidean distance between predicted and true positions
    """
    return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=1))

class InverseMLModelTrainer:
    def __init__(self, config_path=None):
        if config_path is None:
            # Default to yaml/configML_inverse.yaml relative to the script's directory
            config_path = os.path.join(os.path.dirname(__file__), "../yaml/configML_inverse.yaml")
        
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.data_path = self.config["data"]["path"]
        self.h5_file = self.config["data"]["file"]
        self.x_dataset = self.config["data"]["x_dataset"]  # This will be photon ratios (Y in original)
        self.y_dataset = self.config["data"]["y_dataset"]  # This will be positions (X in original)
        
        # Get test size from config or use default
        self.test_size = self.config["data"].get("test_size", 0.1)
        
        # Optional: Add config for photons_per_event if you want to use it
        self.use_photon_counts = self.config["model"].get("use_photon_counts", False)

        self.learning_rate = self.config["model"]["learning_rate"]
        self.epochs = self.config["model"]["epochs"]
        self.batch_size = self.config["model"]["batch_size"]
        self.model_path = self.config["model"]["save_path"]
        
        # Get model architecture parameters
        self.hidden_layers = self.config["model"].get("hidden_layers", [256, 128, 64, 32])
        self.activation = self.config["model"].get("activation", "relu")

        self.output_dir = self.config["output"]["dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        self.X_train, self.X_test = None, None  # Will hold photon ratios (channels)
        self.Y_train, self.Y_test = None, None  # Will hold positions (3D coordinates)
        self.photons_train, self.photons_test = None, None
        
        self.scaler_X, self.scaler_Y = StandardScaler(), StandardScaler()
        self.scaler_photons = StandardScaler() if self.use_photon_counts else None
        self.model = None
        
        # Add new configuration options
        self.use_merci_chan = self.config["data"].get("use_merci_chan", False)
        self.bad_sipms = self.config["data"].get("bad_sipms", None)
        
        print(f"Inverse model will be trained using:")
        print(f" - X data (input) from: {self.x_dataset} (photon ratios)")
        print(f" - Y data (output) from: {self.y_dataset} (positions)")
        if self.use_merci_chan:
            print(f" - Converting SiPM channels to MERCI channels")
            if self.bad_sipms:
                print(f" - Excluding bad SiPMs: {self.bad_sipms}")

        if self.use_photon_counts:
            print(f" - Including photons per event as additional input feature")

    def load_data(self):
        print("\nLoading data for inverse mapping")
        with h5py.File(os.path.join(self.data_path, self.h5_file), "r") as f:
            # For inverse mapping, we swap X and Y
            Y = f[self.y_dataset][:]  # Position data (target)
            X_sipm = f[self.x_dataset][:]  # Photon ratios data (input)
            
            # Check input data before processing
            print(f" Input data statistics before processing:")
            print(f" - Number of zeros: {np.sum(np.sum(X_sipm, axis=1) == 0)}")
            
            # Convert to MERCI channels first if configured
            if self.use_merci_chan:
                print(f" Converting raw SiPM data (shape: {X_sipm.shape}) to MERCI channels...")
                X = sipmid_to_merci(X_sipm, bad_sipm=self.bad_sipms)
                print(f" Converted to MERCI channels shape: {X.shape}")
            else:
                X = X_sipm
            
            # Then normalize data safely
            X = X / np.sum(X, axis=1, keepdims=True)
            ## make NaN values zero
            X = np.nan_to_num(X, nan=0.0)
            
            # Verify normalized data
            print(f" Normalized data statistics:")
            print(f" - Min value: {np.min(X):.4f}")
            print(f" - Max value: {np.max(X):.4f}")
            print(f" - Mean value: {np.mean(X):.4f}")
            
            # Optionally load photons per event if needed for the model
            if self.use_photon_counts:
                photons = f["events/photons_per_event"][:]
                photons = photons.reshape(-1, 1)  # Reshape to 2D for sklearn
                print(f" Also loaded photons_per_event with shape {photons.shape}")
        
        print(f" Data loaded: X (photon ratios) shape {X.shape}, Y (positions) shape {Y.shape}")
        
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
        
        # Scale inputs (photon ratios) if configured to do so
        if self.config["model"].get("scale_x", False):
            self.X_train = self.scaler_X.fit_transform(self.X_train)
            self.X_test = self.scaler_X.transform(self.X_test)
            print(" Applied scaling to X data (photon ratios)")
        else:
            print(" Skipped scaling for X data as it's already normalized")
            
        # Scale outputs (positions)
        self.Y_train = self.scaler_Y.fit_transform(self.Y_train)
        self.Y_test = self.scaler_Y.transform(self.Y_test)
        print(" Applied scaling to Y data (positions)")
            
        print(f" Train/test split: {len(self.X_train)} training samples, {len(self.X_test)} test samples")

    def save_scalers(self):
        scaler_path = os.path.join(self.output_dir, "scalers_inverse")
        os.makedirs(scaler_path, exist_ok=True)
        
        if self.config["model"].get("scale_x", False):
            joblib.dump(self.scaler_X, os.path.join(scaler_path, "scaler_X_inv.pkl"))
            
        joblib.dump(self.scaler_Y, os.path.join(scaler_path, "scaler_Y_inv.pkl"))
            
        if self.use_photon_counts:
            joblib.dump(self.scaler_photons, os.path.join(scaler_path, "scaler_photons_inv.pkl"))
            
        print(f" Scalers saved to {scaler_path}")

    def build_model(self):
        print("\nBuilding inverse model")
        
        # Output dimension is 3 (x, y, z coordinates)
        output_dim = 3
        
        # Get L2 regularization parameters from config
        use_l2 = self.config["model"].get("use_l2_regularization", False)
        l2_lambda = self.config["model"].get("l2_lambda", 0.01)
        
        # Configure regularizer if enabled
        regularizer = tf.keras.regularizers.l2(l2_lambda) if use_l2 else None
        
        if self.use_photon_counts:
            # Create a model that takes both channel ratios and photon count as input
            input_channels = Input(shape=(self.X_train.shape[1],), name="channel_ratios_input")
            input_photons = Input(shape=(1,), name="photon_count_input")
            
            # Channels branch with regularization
            x = Dense(self.hidden_layers[0], 
                     activation=self.activation, 
                     kernel_regularizer=regularizer)(input_channels)
            for units in self.hidden_layers[1:-1]:
                x = Dense(units, 
                         activation=self.activation, 
                         kernel_regularizer=regularizer)(x)
            
            # Combine with photon count
            combined = Concatenate()([x, input_photons])
            
            # Additional layers after combining
            x = Dense(self.hidden_layers[-1], 
                     activation=self.activation, 
                     kernel_regularizer=regularizer)(combined)
            outputs = Dense(output_dim, 
                           kernel_regularizer=regularizer)(x)
            
            self.model = Model(inputs=[input_channels, input_photons], outputs=outputs)
        else:
            # Standard sequential model with just channel ratios input
            input_dim = self.X_train.shape[1]  # Number of channels
            layers = [Dense(self.hidden_layers[0], 
                           input_dim=input_dim, 
                           activation=self.activation,
                           kernel_regularizer=regularizer)]
            
            for units in self.hidden_layers[1:]:
                layers.append(Dense(units, 
                                  activation=self.activation,
                                  kernel_regularizer=regularizer))
                
            # Final output layer (3 neurons for x, y, z coordinates)
            layers.append(Dense(output_dim, 
                              kernel_regularizer=regularizer))
            
            self.model = Sequential(layers)

        # Use Adam optimizer
        optimizer = Adam(learning_rate=self.learning_rate)
        
        # Compile with appropriate metrics for position prediction
        self.model.compile(optimizer=optimizer, 
                        loss='mse', 
                        metrics=['mae', euclidean_distance])
        
        self.model.summary()
        print(f" Model built with {len(self.hidden_layers)} hidden layers and {output_dim} output neurons")
        if use_l2:
            print(f" Using L2 regularization with lambda={l2_lambda}")

    def train_model(self):
        print("\nTraining inverse model")
        
        if self.use_photon_counts:
            # Training with both inputs
            history = self.model.fit(
                [self.X_train, self.photons_train], self.Y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.15,
                verbose=1
            )
        else:
            # Standard training
            history = self.model.fit(
                self.X_train, self.Y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.15,
                verbose=1
            )
            
        print(" Training completed")
        
        # Plot training history
        self.plot_history(history)
        
        return history

    def plot_history(self, history):
        plt.figure(figsize=(12, 8))  # Reduced size since we have 3 plots now
        
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
        
        # Plot Euclidean Distance
        plt.subplot(2, 2, 3)
        plt.plot(history.history['euclidean_distance'], label='Training Euclidean Dist')
        plt.plot(history.history['val_euclidean_distance'], label='Validation Euclidean Dist')
        plt.title('Euclidean Distance Error')
        plt.xlabel('Epoch')
        plt.ylabel('Distance Error')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'inverse_training_history.png'))
        print(" Training history plot saved")

    def evaluate_model(self):
        print("\nEvaluating inverse model")
        
        if self.use_photon_counts:
            # Evaluate model with both inputs
            test_results = self.model.evaluate(
                [self.X_test, self.photons_test], self.Y_test
            )
            
            # Make predictions
            y_pred = self.model.predict([self.X_test, self.photons_test])
        else:
            # Standard evaluation
            test_results = self.model.evaluate(self.X_test, self.Y_test)
            
            # Make predictions
            y_pred = self.model.predict(self.X_test)
        
        # Extract individual metrics from results
        test_loss = test_results[0]
        test_mae = test_results[1]
        test_euclidean = test_results[2]
            
        print(f" Test MAE: {test_mae:.4f}, Test Loss: {test_loss:.4f}")
        print(f" Test Euclidean Distance Error: {test_euclidean:.4f}")
        
        # Calculate and print errors for each coordinate
        y_true_unscaled = self.scaler_Y.inverse_transform(self.Y_test)
        y_pred_unscaled = self.scaler_Y.inverse_transform(y_pred)
        
        coord_errors = np.abs(y_true_unscaled - y_pred_unscaled)
        mean_errors = np.mean(coord_errors, axis=0)
        
        print(f" Mean absolute errors by coordinate:")
        print(f"  X: {mean_errors[0]:.4f}, Y: {mean_errors[1]:.4f}, Z: {mean_errors[2]:.4f}")
        
        # Plot position errors and distributions
        self.plot_position_errors(y_true_unscaled, y_pred_unscaled)
        
        # Plot 3D scatter of actual vs predicted positions
        self.plot_3d_positions(y_true_unscaled, y_pred_unscaled)

    def plot_position_errors(self, y_true, y_pred):
        """Plot histograms of position errors for each coordinate"""
        errors = y_pred - y_true
        
        plt.figure(figsize=(15, 5))
        
        # X coordinate errors
        plt.subplot(1, 3, 1)
        plt.hist(errors[:, 0], bins=50, alpha=0.7)
        plt.title('X Coordinate Errors')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.axvline(x=0, color='r', linestyle='--')
        
        # Y coordinate errors
        plt.subplot(1, 3, 2)
        plt.hist(errors[:, 1], bins=50, alpha=0.7)
        plt.title('Y Coordinate Errors')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.axvline(x=0, color='r', linestyle='--')
        
        # Z coordinate errors
        plt.subplot(1, 3, 3)
        plt.hist(errors[:, 2], bins=50, alpha=0.7)
        plt.title('Z Coordinate Errors')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.axvline(x=0, color='r', linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'position_errors.png'))
        print(" Position error histograms saved")
        
        # Calculate Euclidean distance errors
        euclidean_errors = np.sqrt(np.sum(np.square(errors), axis=1))
        
        # Plot histogram of Euclidean distance errors
        plt.figure(figsize=(10, 6))
        plt.hist(euclidean_errors, bins=100, alpha=0.7)
        plt.title('Euclidean Distance Errors')
        plt.xlabel('Error (distance in mm)')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.axvline(x=np.mean(euclidean_errors), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(euclidean_errors):.4f}')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'euclidean_errors.png'))
        print(" Euclidean distance error histogram saved")

    def plot_3d_positions(self, y_true, y_pred, num_samples=100):
        """Plot 3D scatter of actual vs predicted positions"""
        # Sample a subset of the data to avoid overcrowding
        indices = np.random.choice(len(y_true), min(num_samples, len(y_true)), replace=False)
        
        true_sample = y_true[indices]
        pred_sample = y_pred[indices]
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot actual positions
        ax.scatter(true_sample[:, 0], true_sample[:, 1], true_sample[:, 2], 
                  c='blue', marker='o', label='Actual', alpha=0.6)
        
        # Plot predicted positions
        ax.scatter(pred_sample[:, 0], pred_sample[:, 1], pred_sample[:, 2], 
                  c='red', marker='^', label='Predicted', alpha=0.6)
        
        # Connect actual and predicted positions with lines
        for i in range(len(true_sample)):
            ax.plot([true_sample[i, 0], pred_sample[i, 0]],
                   [true_sample[i, 1], pred_sample[i, 1]],
                   [true_sample[i, 2], pred_sample[i, 2]], 'k-', alpha=0.2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Actual vs Predicted Positions')
        ax.legend()
        
        plt.savefig(os.path.join(self.output_dir, '3d_position_comparison.png'))
        print(f" 3D position comparison plot saved with {num_samples} samples")

    def save_model(self):
        # Save model architecture and weights
        self.model.save(self.model_path)
        
        # Save model config
        model_info = {
            'input_shape': (self.X_train.shape[1],) if not self.use_photon_counts else [(self.X_train.shape[1],), (1,)],
            'output_shape': (3,),  # 3D coordinates
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'use_photon_counts': self.use_photon_counts,
            'scale_x': self.config["model"].get("scale_x", False)
        }
        
        with open(os.path.join(self.output_dir, 'inverse_model_config.yaml'), 'w') as f:
            yaml.dump(model_info, f)
            
        print(f" Model saved at {self.model_path}")
        print(f" Model configuration saved at {os.path.join(self.output_dir, 'inverse_model_config.yaml')}")

    def run(self):
        self.load_data()
        self.build_model()
        history = self.train_model()
        self.evaluate_model()
        self.save_model()
        self.save_scalers()

if __name__ == "__main__":
    trainer = InverseMLModelTrainer()
    trainer.run()