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
    Calculate Kullback-Leibler divergence between true and predicted distributions.
    Both inputs are assumed to be probability distributions (sum to 1 along axis 1).
    """
    # Add small epsilon to avoid log(0) and division by zero
    epsilon = 1e-10
    
    # Ensure inputs are normalized to sum to 1 along axis 1 (each sample is a distribution)
    y_true_sum = tf.reduce_sum(y_true, axis=1, keepdims=True)
    y_pred_sum = tf.reduce_sum(y_pred, axis=1, keepdims=True)
    
    y_true_normalized = y_true / (y_true_sum + epsilon)
    y_pred_normalized = y_pred / (y_pred_sum + epsilon)
    
    # Calculate KL divergence: sum(y_true * log(y_true / y_pred))
    kl_div = y_true_normalized * tf.math.log(y_true_normalized / (y_pred_normalized + epsilon) + epsilon)
    
    # Sum over distribution dimension (axis 1) and take mean over batch
    return tf.reduce_mean(tf.reduce_sum(kl_div, axis=1))

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
        print(f" - Using KL Divergence as optimization metric")

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
        y_sums = np.sum(Y, axis=1)
        print(f" Y data distribution sums: min={np.min(y_sums):.4f}, max={np.max(y_sums):.4f}, mean={np.mean(y_sums):.4f}")
        if np.abs(np.mean(y_sums) - 1.0) > 0.01:
            print(f" Warning: Y data doesn't seem to sum to 1.0 along rows. For KL divergence, distributions should sum to 1.")
            print(f" Consider normalizing the Y data to ensure it represents proper probability distributions.")
        
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
        
        # For KL divergence, we should not scale Y as it needs to remain a probability distribution
        print(" Skipping scaling for Y data to preserve probability distribution properties for KL divergence")
            
        print(f" Train/test split: {len(self.X_train)} training samples, {len(self.X_test)} test samples")

    def save_scalers(self):
        scaler_path = os.path.join(self.output_dir, "scalers")
        os.makedirs(scaler_path, exist_ok=True)
        
        joblib.dump(self.scaler_X, os.path.join(scaler_path, "scaler_X.pkl"))
            
        if self.use_photon_counts:
            joblib.dump(self.scaler_photons, os.path.join(scaler_path, "scaler_photons.pkl"))
            
        print(f" Scalers saved to {scaler_path}")

    def build_model(self):
        print("\nBuilding model")
        
        # Get the output dimension from Y shape
        output_dim = self.Y_train.shape[1]
        
        if self.use_photon_counts:
            # Create a model that takes both position and photon count as input
            input_pos = Input(shape=(3,), name="position_input")
            input_photons = Input(shape=(1,), name="photon_count_input")
            
            # Position branch
            x = Dense(self.hidden_layers[0], activation=self.activation)(input_pos)
            for units in self.hidden_layers[1:-1]:
                x = Dense(units, activation=self.activation)(x)
            
            # Combine with photon count
            combined = Concatenate()([x, input_photons])
            
            # Additional layers after combining
            x = Dense(self.hidden_layers[-1], activation=self.activation)(combined)
            # Change the final layer to use softmax activation to ensure output is a probability distribution
            outputs = Dense(output_dim, activation='softmax')(x)
            
            self.model = Model(inputs=[input_pos, input_photons], outputs=outputs)
        else:
            # Standard sequential model with just position input
            layers = [Dense(self.hidden_layers[0], input_dim=3, activation=self.activation)]
            
            for units in self.hidden_layers[1:]:
                layers.append(Dense(units, activation=self.activation))
                
            # Final output layer with softmax activation for probability distribution
            layers.append(Dense(output_dim, activation='softmax'))
            
            self.model = Sequential(layers)

        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, 
                           loss=kl_divergence, 
                           metrics=['mae', mean_absolute_percentage_error])
        self.model.summary()
        print(f" Model built with {len(self.hidden_layers)} hidden layers and {output_dim} output neurons")
        print(f" Using KL Divergence as the loss function with softmax output activation")

    def train_model(self):
        print("\nTraining model")
        
        # Create a history callback for plotting
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
        plt.figure(figsize=(15, 5))
        
        # Plot KL Divergence loss
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Training KL Divergence')
        plt.plot(history.history['val_loss'], label='Validation KL Divergence')
        plt.title('KL Divergence Loss')
        plt.xlabel('Epoch')
        plt.ylabel('KL Divergence')
        plt.yscale('log')
        plt.legend()
        
        # Plot MAE
        plt.subplot(1, 3, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.yscale('log')
        plt.legend()
        
        # Plot MAPE
        plt.subplot(1, 3, 3)
        plt.plot(history.history['mean_absolute_percentage_error'], label='Training MAPE')
        plt.plot(history.history['val_mean_absolute_percentage_error'], label='Validation MAPE')
        plt.title('Mean Absolute Percentage Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAPE (%)')
        plt.yscale('log')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'))
        print(" Training history plot saved")

    def evaluate_model(self):
        print("\nEvaluating model")
        
        if self.use_photon_counts:
            # Capture all metrics: kl_div, mae, and mape
            test_kl_div, test_mae, test_mape = self.model.evaluate(
                [self.X_test, self.photons_test], self.Y_test
            )
            
            # Make predictions
            y_pred = self.model.predict([self.X_test, self.photons_test])
        else:
            # Capture all metrics: kl_div, mae, and mape
            test_kl_div, test_mae, test_mape = self.model.evaluate(self.X_test, self.Y_test)
            
            # Make predictions
            y_pred = self.model.predict(self.X_test)
            
        print(f" Test KL Divergence: {test_kl_div:.4f}, Test MAE: {test_mae:.4f}, Test MAPE: {test_mape:.2f}%")
        
        # Calculate additional KL divergence statistics
        individual_kl = self.calculate_individual_kl(self.Y_test, y_pred)
        print(f" KL Divergence stats - Min: {np.min(individual_kl):.4f}, Max: {np.max(individual_kl):.4f}, Median: {np.median(individual_kl):.4f}")
        
        # Plot some example predictions vs actual
        self.plot_predictions(self.Y_test, y_pred)
        
        # Plot KL divergence distribution
        self.plot_kl_distribution(individual_kl)
        
        # Plot relative errors
        self.plot_relative_errors(self.Y_test, y_pred)

    def calculate_individual_kl(self, y_true, y_pred):
        """Calculate KL divergence for each individual sample"""
        epsilon = 1e-10
        individual_kl = []
        
        for i in range(len(y_true)):
            # Ensure distributions sum to 1
            p_true = y_true[i] / (np.sum(y_true[i]) + epsilon)
            p_pred = y_pred[i] / (np.sum(y_pred[i]) + epsilon)
            
            # Calculate KL for this sample
            kl = np.sum(p_true * np.log((p_true + epsilon) / (p_pred + epsilon)))
            individual_kl.append(kl)
            
        return np.array(individual_kl)

    def plot_kl_distribution(self, kl_values):
        """Plot the distribution of KL divergence values across samples"""
        plt.figure(figsize=(10, 6))
        plt.hist(kl_values, bins=50, alpha=0.75)
        plt.xlabel('KL Divergence')
        plt.ylabel('Frequency')
        plt.title('Distribution of KL Divergence Values')
        plt.axvline(np.median(kl_values), color='r', linestyle='--', label=f'Median: {np.median(kl_values):.4f}')
        plt.axvline(np.mean(kl_values), color='g', linestyle='--', label=f'Mean: {np.mean(kl_values):.4f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'kl_distribution.png'))
        print(" KL divergence distribution plot saved")

    def plot_predictions(self, y_true, y_pred, num_samples=5, num_channels=10):
        """Plot predictions vs actual for a few samples"""
        plt.figure(figsize=(15, 10))
        
        for i in range(min(num_samples, len(y_true))):
            plt.subplot(num_samples, 1, i+1)
            
            # Select first few channels to display
            x = np.arange(num_channels)
            plt.plot(x, y_true[i][:num_channels], 'b-', label='Actual')
            plt.plot(x, y_pred[i][:num_channels], 'r--', label='Predicted')
            
            # Calculate KL for this example
            epsilon = 1e-10
            p_true = y_true[i] / (np.sum(y_true[i]) + epsilon)
            p_pred = y_pred[i] / (np.sum(y_pred[i]) + epsilon)
            kl = np.sum(p_true * np.log((p_true + epsilon) / (p_pred + epsilon)))
            
            plt.title(f'Sample {i+1}: Actual vs Predicted (KL Div: {kl:.4f})')
            plt.xlabel('Channel')
            plt.ylabel('Photon Ratio')
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'prediction_samples.png'))
        print(f" Prediction plot saved with {num_samples} samples and {num_channels} channels each")

    def plot_relative_errors(self, y_true, y_pred):
        """Plot the relative errors as percentages"""
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
        plt.savefig(os.path.join(self.output_dir, 'channel_percentage_errors.png'))

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
            'loss_function': 'kl_divergence',
            'output_activation': 'softmax'
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