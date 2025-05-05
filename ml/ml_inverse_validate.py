import os
import sys
import numpy as np
import tensorflow as tf
import yaml
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import h5py
from keras.saving import register_keras_serializable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.post_analysis import sipmid_to_merci

@register_keras_serializable()
def euclidean_distance(y_true, y_pred):
    """
    Calculate average Euclidean distance between predicted and true positions
    """
    return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=1))

class InverseModelValidator:
    def __init__(self, config_path=None):
        if config_path is None:
            # Default to yaml/configInverseValidate.yaml relative to the script's directory
            config_path = os.path.join(os.path.dirname(__file__), "../yaml/configInverseValidate.yaml")
        
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.data_path = self.config["data"]["path"]
        
        # Data should come from H5 file
        self.use_h5 = self.config["data"].get("use_h5", True)
        
        if self.use_h5:
            self.h5_file = self.config["data"]["h5_file"]
            self.x_dataset = self.config["data"]["x_dataset"]
            self.true_positions_dataset = self.config["data"].get("true_positions_dataset", "events/origins")
        else:
            self.x_file = self.config["data"]["x_file"]
            self.true_positions_file = self.config["data"].get("true_positions_file", None)
        
        # Model path and output configuration
        self.model_path = self.config["model"]["save_path"]
        self.output_dir = self.config["output"]["dir"]
        
        # Check if predictions_file is defined in the YAML config
        predictions_file_key = self.config["output"].get("predictions_file", None)
        if predictions_file_key:
            self.predictions_file = os.path.join(self.output_dir, predictions_file_key)
        else:
            self.predictions_file = None

        os.makedirs(self.output_dir, exist_ok=True)

        print("\nLoading inverse model")
        self.model = tf.keras.models.load_model(
            self.model_path,
            custom_objects={'euclidean_distance': euclidean_distance}
        )
        print(f" Model loaded from {self.model_path}")

        # Try to load model config if available
        self.model_config = None
        config_path = os.path.join(os.path.dirname(self.model_path), 'inverse_model_config.yaml')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self.model_config = yaml.safe_load(f)
                print(f" Model configuration loaded")
            except yaml.constructor.ConstructorError as e:
                print(f" Warning: Could not load model config due to YAML parsing error: {e}")
                print(f" Continuing with default configuration settings")
        
        # Add MERCI configuration
        self.use_merci_chan = self.config["data"].get("use_merci_chan", False)
        self.bad_sipms = self.config["data"].get("bad_sipms", None)
        
        if self.use_merci_chan:
            print(" Using MERCI channel conversion")
            if self.bad_sipms:
                print(f" Excluding bad SiPMs: {self.bad_sipms}")

        self.load_scalers()
        self.load_data()

    def load_scalers(self):
        print("\nLoading scalers")
        
        # Use model's scaler_dir if specified, otherwise use output_dir
        scaler_path = self.config["model"].get("scaler_dir", None)
        if not scaler_path:
            scaler_path = os.path.join(self.output_dir, "scalers")
        
        if os.path.exists(scaler_path):
            # If path is just a directory without 'scalers' folder
            if not os.path.exists(os.path.join(scaler_path, "scaler_X.pkl")) and os.path.exists(os.path.join(scaler_path, "scalers", "scaler_X.pkl")):
                scaler_path = os.path.join(scaler_path, "scalers")
                
            self.scaler_X = joblib.load(os.path.join(scaler_path, "scaler_X.pkl"))
            print(f" X scaler loaded from: {os.path.join(scaler_path, 'scaler_X.pkl')}")
            
            # For inverse model, we need Y scaler for input data (detector responses)
            if os.path.exists(os.path.join(scaler_path, "scaler_Y.pkl")):
                self.scaler_Y = joblib.load(os.path.join(scaler_path, "scaler_Y.pkl"))
                print(f" Y scaler loaded from: {os.path.join(scaler_path, 'scaler_Y.pkl')}")
            else:
                self.scaler_Y = None
                print(f" No Y scaler found (working with normalized ratios)")
            
            # Check if model uses photon counts
            self.use_photon_counts = False
            if self.model_config and 'use_photon_counts' in self.model_config:
                self.use_photon_counts = self.model_config['use_photon_counts']
            
            if self.use_photon_counts and os.path.exists(os.path.join(scaler_path, "scaler_photons.pkl")):
                self.scaler_photons = joblib.load(os.path.join(scaler_path, "scaler_photons.pkl"))
                print(f" Photon count scaler loaded")
            
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
        print("\nLoading validation data")
        
        if self.use_h5:
            with h5py.File(os.path.join(self.data_path, self.h5_file), "r") as f:
                # Load SiPM data first
                X_sipm = f[self.x_dataset][:]
                # Convert to MERCI channels first if configured
                if self.use_merci_chan:
                    print(f" Converting SiPM data (shape: {X_sipm.shape}) to MERCI channels...")
                    self.X = sipmid_to_merci(X_sipm, bad_sipm=self.bad_sipms)
                    print(f" Converted to MERCI channels shape: {self.X.shape}")
                else:
                    self.X = X_sipm
                    
                # Then normalize data safely
                self.X = self.X / np.sum(self.X, axis=1, keepdims=True)
                self.X = np.nan_to_num(self.X, nan=0.0)
                
                # Load true positions for validation
                if self.true_positions_dataset in f:
                    self.true_positions = f[self.true_positions_dataset][:]
                    print(f" True positions loaded with shape {self.true_positions.shape}")
                else:
                    self.true_positions = None
                    print(" No true positions found in the file")
                
                # Load source photons count data if model uses it
                if self.use_photon_counts:
                    if "events/source_photons" in f:
                        self.source_photons = f["events/source_photons"][:]
                        self.photons = self.source_photons.reshape(-1, 1)  # Reshape for model input
                        print(f" Source photons loaded with shape {self.source_photons.shape}")
                    elif "events/photons_per_event" in f:
                        self.source_photons = f["events/photons_per_event"][:]
                        self.photons = self.source_photons.reshape(-1, 1)  # Reshape for model input
                        print(f" Source photons loaded from photons_per_event with shape {self.source_photons.shape}")
                    else:
                        self.photons = None
                        print(" No source photon count data found in the file")
        else:
            self.X = np.load(os.path.join(self.data_path, self.x_file))
            
            if self.true_positions_file:
                try:
                    self.true_positions = np.load(os.path.join(self.data_path, self.true_positions_file))
                    print(f" True positions loaded with shape {self.true_positions.shape}")
                except:
                    self.true_positions = None
                    print(" Could not load true positions file")
            else:
                self.true_positions = None
                
            self.photons = None
            
        # For inverse model, the input is the detector responses (Y in forward model)
        # So we need to scale with Y scaler
        if self.scaler_Y:
            self.X_scaled = self.scaler_Y.transform(self.X)
        else:
            # If no scaler provided, assume data is already normalized
            self.X_scaled = self.X
            
        # Scale photon counts if needed
        if self.use_photon_counts and self.photons is not None and hasattr(self, 'scaler_photons'):
            self.photons_scaled = self.scaler_photons.transform(self.photons)
        
        print(f" Validation data loaded: X shape {self.X.shape}")
        if self.true_positions is not None:
            print(f" True positions shape {self.true_positions.shape}")
            
    def predict(self):
        print("\nGenerating position predictions")
        
        # Make predictions with the model
        if self.use_photon_counts and hasattr(self, 'photons_scaled'):
            positions_pred_scaled = self.model.predict([self.X_scaled, self.photons_scaled])
        else:
            positions_pred_scaled = self.model.predict(self.X_scaled)
        
        # Inverse transform if we used a scaler for positions
        if hasattr(self, 'scaler_X'):
            positions_pred = self.scaler_X.inverse_transform(positions_pred_scaled)
        else:
            positions_pred = positions_pred_scaled
        
        # Save predictions if output file is defined
        if self.predictions_file:
            np.save(self.predictions_file, positions_pred)
            print(f" Position predictions saved to {self.predictions_file}")
            
        return positions_pred

    def evaluate(self, positions_pred):
        if self.true_positions is not None:
            print("\nEvaluating inverse model predictions")
            
            # Calculate overall MAE and MSE
            mae = mean_absolute_error(self.true_positions, positions_pred)
            mse = mean_squared_error(self.true_positions, positions_pred)
            rmse = np.sqrt(mse)
            
            # Calculate Euclidean distance error
            euclidean_dist = np.sqrt(np.sum((self.true_positions - positions_pred)**2, axis=1))
            mean_euclidean = np.mean(euclidean_dist)
            
            print(f" Overall MAE: {mae:.4f}")
            print(f" RMSE: {rmse:.4f}")
            print(f" Mean Euclidean Distance Error: {mean_euclidean:.4f}")
            
            # Calculate per-coordinate errors
            mae_x = mean_absolute_error(self.true_positions[:, 0], positions_pred[:, 0])
            mae_y = mean_absolute_error(self.true_positions[:, 1], positions_pred[:, 1])
            mae_z = mean_absolute_error(self.true_positions[:, 2], positions_pred[:, 2])
            
            print(" Mean absolute errors by coordinate:")
            print(f"  X: {mae_x:.4f}, Y: {mae_y:.4f}, Z: {mae_z:.4f}")
            
            # Calculate mean absolute percentage error (MAPE) for each coordinate
            # Add epsilon to avoid division by zero
            epsilon = 1e-10
            mape_x = np.mean(np.abs((self.true_positions[:, 0] - positions_pred[:, 0]) / (np.abs(self.true_positions[:, 0]) + epsilon))) * 100
            mape_y = np.mean(np.abs((self.true_positions[:, 1] - positions_pred[:, 1]) / (np.abs(self.true_positions[:, 1]) + epsilon))) * 100
            mape_z = np.mean(np.abs((self.true_positions[:, 2] - positions_pred[:, 2]) / (np.abs(self.true_positions[:, 2]) + epsilon))) * 100
            
            print(" Mean absolute percentage errors by coordinate:")
            print(f"  X: {mape_x:.2f}%, Y: {mape_y:.2f}%, Z: {mape_z:.2f}%")
            
            # Create evaluation visualizations
            self.visualize_predictions(positions_pred, euclidean_dist)
        else:
            print(" No ground truth positions available for evaluation")

    def visualize_predictions(self, positions_pred, euclidean_dist=None):
        """Visualize position predictions vs ground truth"""
        if self.true_positions is None:
            return
            
        # Get visualization settings from config or use defaults
        vis_config = self.config.get("visualization", {})
        num_samples = vis_config.get("num_samples", 3)
        
        # Create scatter plots for each coordinate pair
        plt.figure(figsize=(18, 6))
        
        # X true vs X predicted
        plt.subplot(1, 3, 1)
        plt.scatter(self.true_positions[:, 0], positions_pred[:, 0], alpha=0.5, s=3)
        plt.plot([-20, 20], [-20, 20], 'r--', linewidth=2)  # Perfect prediction line
        plt.title('X Coordinate: True vs Predicted')
        plt.xlabel('True X Position')
        plt.ylabel('Predicted X Position')
        plt.grid(True, alpha=0.3)
        
        # Y true vs Y predicted
        plt.subplot(1, 3, 2)
        plt.scatter(self.true_positions[:, 1], positions_pred[:, 1], alpha=0.5, s=3)
        plt.plot([-20, 20], [-20, 20], 'r--', linewidth=2)  # Perfect prediction line
        plt.title('Y Coordinate: True vs Predicted')
        plt.xlabel('True Y Position')
        plt.ylabel('Predicted Y Position')
        plt.grid(True, alpha=0.3)
        
        # Z true vs Z predicted
        plt.subplot(1, 3, 3)
        plt.scatter(self.true_positions[:, 2], positions_pred[:, 2], alpha=0.5, s=3)
        plt.plot([-20, 20], [-20, 20], 'r--', linewidth=2)  # Perfect prediction line
        plt.title('Z Coordinate: True vs Predicted')
        plt.xlabel('True Z Position')
        plt.ylabel('Predicted Z Position')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'position_prediction_scatter.png'))
        print(f" Position scatter plots saved to {os.path.join(self.output_dir, 'position_prediction_scatter.png')}")
        
        # Histogram of Euclidean distance errors
        if euclidean_dist is not None:
            plt.figure(figsize=(10, 6))
            plt.hist(euclidean_dist, bins=100, alpha=0.7, histtype='step')
            plt.yscale('log')
            plt.axvline(np.mean(euclidean_dist), color='r', linestyle='dashed', linewidth=2, label='Mean = {:.3g} mm'.format(np.mean(euclidean_dist)))
            plt.title('Distribution of Distance Errors')
            plt.xlabel('Distance Error [mm]')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, 'euclidean_error_histogram.png'))
            print(f" Error histogram saved to {os.path.join(self.output_dir, 'euclidean_error_histogram.png')}")
            
        # Create 3D visualization of selected samples
        if num_samples > 0:
            # Select random samples for 3D visualization
            sample_indices = np.random.choice(len(self.true_positions), size=min(num_samples, len(self.true_positions)), replace=False)
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            for idx in sample_indices:
                true_pos = self.true_positions[idx]
                pred_pos = positions_pred[idx]
                
                # Plot true position (blue), predicted position (red), and connecting line
                ax.scatter(true_pos[0], true_pos[1], true_pos[2], color='blue', s=100, label='True' if idx == sample_indices[0] else "")
                ax.scatter(pred_pos[0], pred_pos[1], pred_pos[2], color='red', s=100, label='Predicted' if idx == sample_indices[0] else "")
                ax.plot([true_pos[0], pred_pos[0]], 
                        [true_pos[1], pred_pos[1]], 
                        [true_pos[2], pred_pos[2]], 'k-', alpha=0.5)
                
                # Add text labels with positions and error
                error = np.sqrt(np.sum((true_pos - pred_pos)**2))
                ax.text(true_pos[0], true_pos[1], true_pos[2], f'T{idx}', size=8)
                ax.text(pred_pos[0], pred_pos[1], pred_pos[2], f'P{idx} (E={error:.2f})', size=8)
            
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_zlabel('Z Position')
            ax.set_title('3D Visualization of Position Reconstruction')
            ax.legend()
            
            # Set equal aspect ratio
            max_range = np.array([
                positions_pred[:, 0].max() - positions_pred[:, 0].min(),
                positions_pred[:, 1].max() - positions_pred[:, 1].min(),
                positions_pred[:, 2].max() - positions_pred[:, 2].min()
            ]).max() / 2.0
            
            mid_x = (positions_pred[:, 0].max() + positions_pred[:, 0].min()) * 0.5
            mid_y = (positions_pred[:, 1].max() + positions_pred[:, 1].min()) * 0.5
            mid_z = (positions_pred[:, 2].max() + positions_pred[:, 2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            plt.savefig(os.path.join(self.output_dir, 'position_reconstruction_3d.png'))
            print(f" 3D visualization saved to {os.path.join(self.output_dir, 'position_reconstruction_3d.png')}")

    def visualize_error_distribution(self, positions_pred):
        """Create heatmaps and plots showing spatial distribution of errors"""
        if self.true_positions is None:
            return
            
        # Calculate per-coordinate absolute errors
        abs_errors = np.abs(self.true_positions - positions_pred)
        euclidean_dist = np.sqrt(np.sum(abs_errors**2, axis=1))
        
        # Create mask for errors > 5mm
        error_mask = euclidean_dist > 5.0
        
        # Filter data using the mask
        filtered_positions = self.true_positions[error_mask]
        filtered_errors = euclidean_dist[error_mask]
        
        if len(filtered_positions) == 0:
            print(" No events found with distance error > 5mm")
            return
            
        print(f" Found {len(filtered_positions)} events with distance error > 5mm")
        
        # Create 3D scatter plot with color indicating error magnitude
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            filtered_positions[:, 0],
            filtered_positions[:, 1],
            filtered_positions[:, 2],
            c=filtered_errors,
            cmap='viridis',
            s=30,
            alpha=0.7
        )
        
        # Add color bar to show error magnitudes
        plt.colorbar(scatter, ax=ax, label='Euclidean Distance Error (mm)')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title('Spatial Distribution of Large Errors (>5mm)')
        
        plt.savefig(os.path.join(self.output_dir, 'error_spatial_distribution_large_errors.png'))
        print(f" Error spatial distribution saved to {os.path.join(self.output_dir, 'error_spatial_distribution_large_errors.png')}")
        
        # Create 2D slice views of error distributions
        plt.figure(figsize=(18, 6))
        
        # XY plane
        plt.subplot(1, 3, 1)
        plt.scatter(
            filtered_positions[:, 0],
            filtered_positions[:, 1],
            c=filtered_errors,
            cmap='viridis',
            s=20,
            alpha=0.7
        )
        plt.colorbar(label='Error (mm)')
        plt.title('Error Distribution in XY Plane (Errors >5mm)')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True, alpha=0.3)
        
        # XZ plane
        plt.subplot(1, 3, 2)
        plt.scatter(
            filtered_positions[:, 0],
            filtered_positions[:, 2],
            c=filtered_errors,
            cmap='viridis',
            s=20,
            alpha=0.7
        )
        plt.colorbar(label='Error (mm)')
        plt.title('Error Distribution in XZ Plane (Errors >5mm)')
        plt.xlabel('X Position')
        plt.ylabel('Z Position')
        plt.grid(True, alpha=0.3)
        
        # YZ plane
        plt.subplot(1, 3, 3)
        plt.scatter(
            filtered_positions[:, 1],
            filtered_positions[:, 2],
            c=filtered_errors,
            cmap='viridis',
            s=20,
            alpha=0.7
        )
        plt.colorbar(label='Error (mm)')
        plt.title('Error Distribution in YZ Plane (Errors >5mm)')
        plt.xlabel('Y Position')
        plt.ylabel('Z Position')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'error_distribution_slices_large_errors.png'))
        print(f" Error distribution slices saved to {os.path.join(self.output_dir, 'error_distribution_slices_large_errors.png')}")
        
        # Add histogram of large errors
        plt.figure(figsize=(10, 6))
        plt.hist(filtered_errors, bins=50, alpha=0.7, histtype='step')
        plt.title('Distribution of Large Distance Errors (>5mm)')
        plt.xlabel('Distance Error (mm)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'large_errors_histogram.png'))
        print(f" Large errors histogram saved to {os.path.join(self.output_dir, 'large_errors_histogram.png')}")

    def run(self):
        positions_pred = self.predict()
        self.evaluate(positions_pred)
        self.visualize_error_distribution(positions_pred)
        
        print("\nInverse model validation complete!")

if __name__ == "__main__":
    validator = InverseModelValidator()
    validator.run()