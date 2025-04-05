import os
import numpy as np
import tensorflow as tf
import yaml
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import h5py

class MLModelInference:
    def __init__(self, config_path=None):
        if config_path is None:
            # Default to yaml/configPred.yaml relative to the script's directory
            config_path = os.path.join(os.path.dirname(__file__), "../yaml/configPred.yaml")
        
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.data_path = self.config["data"]["path"]
        
        # Data can come from either H5 file or separate npy files
        self.use_h5 = self.config["data"].get("use_h5", False)
        
        if self.use_h5:
            self.h5_file = self.config["data"]["h5_file"]
            self.x_dataset = self.config["data"]["x_dataset"]
            self.y_dataset = self.config["data"].get("y_dataset", None)  # Optional for evaluation
        else:
            self.x_file = self.config["data"]["x_file"]
            self.y_file = self.config["data"].get("y_file", None)  # Optional for evaluation
        
        # Model path and output configuration
        self.model_path = self.config["model"]["save_path"]
        self.output_dir = self.config["output"]["dir"]
        
        # Check if predictions_file is defined in the YAML config
        predictions_file_key = self.config["output"].get("predictions_file", None)
        if predictions_file_key:
            self.predictions_file = os.path.join(self.output_dir, predictions_file_key)
        else:
            self.predictions_file = None  # Set to None if not defined
            # print(" Warning: 'predictions_file' not defined in the YAML configuration. Predictions will not be saved.")

        os.makedirs(self.output_dir, exist_ok=True)

        print("\nLoading model")
        self.model = tf.keras.models.load_model(self.model_path)
        print(f" Model loaded from {self.model_path}")

        # Try to load model config if available
        self.model_config = None
        config_path = os.path.join(os.path.dirname(self.model_path), 'model_config.yaml')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self.model_config = yaml.safe_load(f)
                print(f" Model configuration loaded")
            except yaml.constructor.ConstructorError as e:
                print(f" Warning: Could not load model config due to YAML parsing error: {e}")
                print(f" Continuing with default configuration settings")
        
        self.load_scalers()
        self.load_data()

    def load_scalers(self):
        print("\nLoading scalers")
        
        scaler_path = os.path.join(self.output_dir, "scalers")
        if os.path.exists(scaler_path):
            # New structure with scalers in a dedicated directory
            self.scaler_X = joblib.load(os.path.join(scaler_path, "scaler_X.pkl"))
            print(f" X scaler loaded from: {os.path.join(scaler_path, 'scaler_X.pkl')}")
            
            # Check if we need to scale Y based on model config
            scale_y = True
            if self.model_config and 'scale_y' in self.model_config:
                scale_y = self.model_config['scale_y']
            
            if scale_y and os.path.exists(os.path.join(scaler_path, "scaler_Y.pkl")):
                self.scaler_Y = joblib.load(os.path.join(scaler_path, "scaler_Y.pkl"))
                print(f" Y scaler loaded from: {os.path.join(scaler_path, 'scaler_Y.pkl')}")
            else:
                self.scaler_Y = None
                print(f" No Y scaler found or not needed (working with normalized ratios)")
            
            # Check if model uses photon counts
            use_photon_counts = False
            if self.model_config and 'use_photon_counts' in self.model_config:
                use_photon_counts = self.model_config['use_photon_counts']
            
            if use_photon_counts and os.path.exists(os.path.join(scaler_path, "scaler_photons.pkl")):
                self.scaler_photons = joblib.load(os.path.join(scaler_path, "scaler_photons.pkl"))
                print(f" Photon count scaler loaded")
                self.use_photon_counts = True
            else:
                self.use_photon_counts = False
        else:
            # Legacy structure with scalers in the main directory
            self.scaler_X = joblib.load(os.path.join(self.output_dir, "scaler_X.pkl"))
            
            if os.path.exists(os.path.join(self.output_dir, "scaler_Y.pkl")):
                self.scaler_Y = joblib.load(os.path.join(self.output_dir, "scaler_Y.pkl"))
            else:
                self.scaler_Y = None
                print(f" No Y scaler found (working with normalized ratios)")
            
            self.use_photon_counts = False
            
        print(" Scalers loaded successfully")

    def load_data(self):
        print("\nLoading test data")
        
        if self.use_h5:
            with h5py.File(os.path.join(self.data_path, self.h5_file), "r") as f:
                self.X = f[self.x_dataset][:]
                
                if self.y_dataset and self.y_dataset in f:
                    self.Y_true = f[self.y_dataset][:]
                    have_y = True
                else:
                    self.Y_true = None
                    have_y = False
                
                # Load source photons count data
                if "events/source_photons" in f:
                    self.source_photons = f["events/source_photons"][:]
                    print(f" Source photons loaded with shape {self.source_photons.shape}")
                else:
                    # Try alternate location
                    if "events/photons_per_event" in f:
                        self.source_photons = f["events/photons_per_event"][:]
                        print(f" Source photons loaded from photons_per_event with shape {self.source_photons.shape}")
                    else:
                        self.source_photons = None
                        print(" No source photon count data found in the file")

                # If model requires photon counts, try to load them
                if self.use_photon_counts and "events/photons_per_event" in f:
                    self.photons = f["events/photons_per_event"][:]
                    self.photons = self.photons.reshape(-1, 1)  # Reshape for model input
                    self.photons_scaled = self.scaler_photons.transform(self.photons)
        else:
            self.X = np.load(os.path.join(self.data_path, self.x_file))
            
            if self.y_file:
                try:
                    self.Y_true = np.load(os.path.join(self.data_path, self.y_file))
                    have_y = True
                except:
                    self.Y_true = None
                    have_y = False
            else:
                self.Y_true = None
                have_y = False
                
            # For NPY files, we don't support photon counts
            self.use_photon_counts = False
        
        # Scale the X input data
        self.X_scaled = self.scaler_X.transform(self.X)
        
        print(f" Test data loaded: X shape {self.X.shape}")
        if have_y:
            print(f" Y shape {self.Y_true.shape}")
    def get_source_data(self):
        """Retrieve the original source positions and photon counts from the loaded data"""
        
        # Check if data has been loaded
        if not hasattr(self, 'X') or self.X is None:
            print("Warning: No position data (X) loaded yet")
            origins = None
        else:
            origins = self.X
            print(f"Origins shape: {origins.shape}")
        
        # Check if source photon data was loaded
        if not hasattr(self, 'source_photons') or self.source_photons is None:
            print("Warning: No source photon data loaded")
            source_photons = None
        else:
            source_photons = self.source_photons
            print(f"Source photons shape: {source_photons.shape}")
            print(f"Mean photons per event: {np.mean(source_photons):.2f}")
            
        return origins, source_photons
        
    def predict(self):
        print("\nGenerating predictions")
        
        # Make predictions with the model
        if self.use_photon_counts:
            Y_pred_scaled = self.model.predict([self.X_scaled, self.photons_scaled])
        else:
            Y_pred_scaled = self.model.predict(self.X_scaled)
        
        # Inverse transform if needed
        if self.scaler_Y:
            Y_pred = self.scaler_Y.inverse_transform(Y_pred_scaled)
        else:
            Y_pred = Y_pred_scaled
        
        # Ensure all values are non-negative (ratios can't be negative)
        Y_pred = np.maximum(0, Y_pred)
        
        # Save ratio predictions only if predictions_file is defined
        if self.predictions_file:
            np.save(self.predictions_file, Y_pred)
            print(f" Ratio predictions saved to {self.predictions_file}")
        return Y_pred

    def evaluate(self, Y_pred):
        if self.Y_true is not None:
            print("\nEvaluating model predictions")
            
            # For ratios, we use the raw values
            mae = mean_absolute_error(self.Y_true, Y_pred)
            mse = mean_squared_error(self.Y_true, Y_pred)
            print(f" MAE (photon ratios): {mae:.6f}")
            print(f" MSE (photon ratios): {mse:.6f}")
            
            # Create evaluation visualizations
            self.visualize_predictions(Y_pred)
        else:
            print(" No ground truth data available for evaluation")

    def visualize_predictions(self, Y_pred, num_samples=3):
        """Visualize sample predictions as bar charts comparing to ground truth"""
        if self.Y_true is None:
            return
            
        # Get visualization settings from config or use defaults
        vis_config = self.config.get("visualization", {})
        num_samples = vis_config.get("num_samples", num_samples)
        
        # Get number of channels
        num_channels = Y_pred.shape[1]
        
        # Select random sample indices
        total_samples = len(self.Y_true)
        if total_samples <= num_samples:
            # If we have fewer samples than requested, use all of them
            sample_indices = np.arange(total_samples)
        else:
            # Randomly select sample indices without replacement
            sample_indices = np.random.choice(total_samples, size=num_samples, replace=False)
        
        plt.figure(figsize=(14, 4 * num_samples))
        
        for idx, sample_idx in enumerate(sample_indices):
            plt.subplot(num_samples, 1, idx+1)
            
            # Create positions for bars
            x = np.arange(num_channels)
            width = 0.35
            
            # Create bar chart
            plt.bar(x - width/2, self.Y_true[sample_idx], width, label='Actual', alpha=0.7)
            plt.bar(x + width/2, Y_pred[sample_idx], width, label='Predicted', alpha=0.7)
            
            # Use original X position data directly instead of converting from scaled
            if self.X.shape[1] >= 3:
                x_true_str = np.array2string(self.X[sample_idx][:3], precision=3, separator=', ')
                plt.title(f'Random event {sample_idx} at Position: {x_true_str}')
            else:
                plt.title(f'Random event {sample_idx}')
            
            plt.xlabel('Channel ID')
            plt.ylabel('Photon Ratio')
            plt.legend()
            
            # Add grid for better readability
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # If too many channels, limit x-axis ticks to avoid overcrowding
            if num_channels > 20:
                step = max(1, num_channels // 20)
                plt.xticks(np.arange(0, num_channels, step))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'prediction_comparison.png'))
        print(f" Prediction visualization saved to {os.path.join(self.output_dir, 'prediction_comparison.png')}")

    def visualize_input_positions(self):
        """Create histograms and plots of X, Y, Z positions from X_scaled data"""
        print("\nGenerating position distributions")
        
        # Check if we have at least 3 dimensions for positions
        if self.X_scaled.shape[1] < 3:
            print(" Warning: X_scaled has fewer than 3 dimensions, cannot plot 3D positions")
            return
        
        # Extract X, Y, Z positions (assuming they're the first 3 columns)
        x_positions = self.X_scaled[:, 0]
        y_positions = self.X_scaled[:, 1]
        z_positions = self.X_scaled[:, 2]
        
        # Create histograms for each position dimension
        plt.figure(figsize=(15, 5))
        
        # X position histogram
        plt.subplot(1, 3, 1)
        plt.hist(x_positions, bins=50, alpha=0.7, color='red')
        plt.title('X Position Distribution')
        plt.xlabel('X Position (scaled)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # Y position histogram
        plt.subplot(1, 3, 2)
        plt.hist(y_positions, bins=50, alpha=0.7, color='green')
        plt.title('Y Position Distribution')
        plt.xlabel('Y Position (scaled)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # Z position histogram
        plt.subplot(1, 3, 3)
        plt.hist(z_positions, bins=50, alpha=0.7, color='blue')
        plt.title('Z Position Distribution')
        plt.xlabel('Z Position (scaled)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'position_histograms.png'))
        
        # Create 2D scatter plots to visualize spatial relationships
        plt.figure(figsize=(15, 5))
        
        # X vs Y scatter plot
        plt.subplot(1, 3, 1)
        plt.scatter(x_positions, y_positions, alpha=0.3, s=1, color='purple')
        plt.title('X vs Y Positions')
        plt.xlabel('X Position (scaled)')
        plt.ylabel('Y Position (scaled)')
        plt.grid(True, alpha=0.3)
        
        # X vs Z scatter plot
        plt.subplot(1, 3, 2)
        plt.scatter(x_positions, z_positions, alpha=0.3, s=1, color='orange')
        plt.title('X vs Z Positions')
        plt.xlabel('X Position (scaled)')
        plt.ylabel('Z Position (scaled)')
        plt.grid(True, alpha=0.3)
        
        # Y vs Z scatter plot
        plt.subplot(1, 3, 3)
        plt.scatter(y_positions, z_positions, alpha=0.3, s=1, color='teal')
        plt.title('Y vs Z Positions')
        plt.xlabel('Y Position (scaled)')
        plt.ylabel('Z Position (scaled)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'position_scatterplots.png'))
        
        print(f" Position distribution plots saved to {self.output_dir}")

    def run(self):
        # self.visualize_input_positions()


        Y_pred = self.predict()
        self.evaluate(Y_pred)

if __name__ == "__main__":
    inference = MLModelInference()
    inference.run()