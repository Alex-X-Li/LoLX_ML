import numpy as np
import tensorflow as tf
import os
import joblib
import yaml

# Add this function definition
def euclidean_distance(y_true, y_pred):
    """Calculate Euclidean distance between predicted and true positions"""
    return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1))

class MLPositionReconstructor:
    """
    Position reconstruction using pre-trained inverse neural network model.
    Takes MERCI channel photon ratios as input and predicts 3D positions.
    """
    
    def __init__(self, model_path=None, scalers_dir=None, config_path=None):
        """
        Initialize the position reconstructor with model and scalers.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to the trained inverse model. Default is 'training_data/inverse_model.keras'
        scalers_dir : str, optional
            Path to directory containing scalers. Default is 'training_data/inverse_model'
        config_path : str, optional
            Path to model configuration file. Default is 'training_data/inverse_model/inverse_model_config.yaml'
        """
        # Set default paths if not provided
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                    "training_data/inverse_model.keras")
        
        if scalers_dir is None:
            scalers_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     "training_data/inverse_model")
        
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     "training_data/inverse_model/inverse_model_config.yaml")
        
        print(f"Loading model from {model_path}")
        # Add custom_objects parameter with our function
        self.model = tf.keras.models.load_model(model_path, 
                                              custom_objects={'euclidean_distance': euclidean_distance})
        
        # Load model configuration if it exists
        self.use_photon_counts = False
        if os.path.exists(config_path):
            with open(config_path, "r") as file:
                try:
                    self.model_config = yaml.safe_load(file)
                except yaml.constructor.ConstructorError:
                    print("Warning: Safe YAML loading failed, trying unsafe loader for Python objects")
                    # Reopen the file since the reader position is at the end
                    file.seek(0)
                    self.model_config = yaml.unsafe_load(file)
                if 'use_photon_counts' in self.model_config:
                    self.use_photon_counts = self.model_config['use_photon_counts']
        
        # Load scalers
        self.load_scalers(scalers_dir)
        
        print("ML Position Reconstructor initialized")
        
    def load_scalers(self, scalers_dir):
        """Load scalers for input/output normalization"""
        print("Loading scalers")
        
        # Check for scalers with both naming conventions
        x_scaler_path = os.path.join(scalers_dir, "scaler_X.pkl")
        x_inv_scaler_path = os.path.join(scalers_dir, "scaler_X_inv.pkl")
        
        if os.path.exists(x_scaler_path):
            self.scaler_X = joblib.load(x_scaler_path)
            print(f"Position scaler loaded from: {x_scaler_path}")
        elif os.path.exists(x_inv_scaler_path):
            self.scaler_X = joblib.load(x_inv_scaler_path)
            print(f"Position scaler loaded from: {x_inv_scaler_path}")
        else:
            self.scaler_X = None
            print("WARNING: No position scaler found!")
        
        # For inverse model, Y scaler is used for input (photon ratios)
        y_scaler_path = os.path.join(scalers_dir, "scaler_Y.pkl")
        y_inv_scaler_path = os.path.join(scalers_dir, "scaler_Y_inv.pkl")
        
        if os.path.exists(y_scaler_path):
            self.scaler_Y = joblib.load(y_scaler_path)
            print(f"Input scaler loaded from: {y_scaler_path}")
        elif os.path.exists(y_inv_scaler_path):
            self.scaler_Y = joblib.load(y_inv_scaler_path)
            print(f"Input scaler loaded from: {y_inv_scaler_path}")
        else:
            self.scaler_Y = None
            print("No input scaler found (assuming normalized ratios)")
    
    def reconstruct_position(self, photon_ratios, photon_counts=None):
        """
        Reconstruct 3D position from photon ratios.
        
        Parameters:
        -----------
        photon_ratios : array-like
            Photon ratios in MERCI channel format (array with size 27)
        photon_counts : float or array-like, optional
            Total photon counts for the event(s) if the model uses them
            
        Returns:
        --------
        numpy.ndarray
            Reconstructed 3D position [x, y, z]
        """
        # Ensure input is numpy array
        photon_ratios = np.array(photon_ratios, dtype=float)
        
        # Handle single event vs batch of events
        single_event = False
        if photon_ratios.ndim == 1:
            single_event = True
            photon_ratios = photon_ratios.reshape(1, -1)
        
        # Always normalize again to match validator logic
        photon_ratios = photon_ratios / np.sum(photon_ratios, axis=1, keepdims=True)
        photon_ratios = np.nan_to_num(photon_ratios, nan=0.0)
        
        # Scale input if scaler exists
        if self.scaler_Y is not None:
            X_scaled = self.scaler_Y.transform(photon_ratios)
        else:
            X_scaled = photon_ratios
            
        # Make prediction
        if self.use_photon_counts:
            if photon_counts is None:
                print("Warning: Model requires photon counts but none provided. Using default value.")
                photon_counts = np.ones(photon_ratios.shape[0]) * 10000
                
            # Handle single value for photon_counts
            if np.isscalar(photon_counts):
                photon_counts = np.array([photon_counts])
                
            # Reshape for scaler
            photon_counts = np.array(photon_counts).reshape(-1, 1)
                
            # Scale photon counts
            if hasattr(self, 'scaler_photons') and self.scaler_photons is not None:
                photons_scaled = self.scaler_photons.transform(photon_counts)
            else:
                photons_scaled = photon_counts
                
            # Predict with both inputs
            positions_pred_scaled = self.model.predict([X_scaled, photons_scaled], verbose=0)
        else:
            # Standard prediction with just photon ratios
            positions_pred_scaled = self.model.predict(X_scaled, verbose=0)
            
        # Inverse transform positions if needed
        if hasattr(self, 'scaler_X') and self.scaler_X is not None:
            positions = self.scaler_X.inverse_transform(positions_pred_scaled)
        else:
            positions = positions_pred_scaled
            
        # Return single position or batch based on input
        if single_event:
            return positions[0]
        else:
            return positions


# Example usage

# if __name__ == "__main__":
#     # Initialize the reconstructor
#     reconstructor = MLPositionReconstructor()
    
#     # Example photon ratios in MERCI channel format (normalized)
#     example_ratios = np.random.rand(27)
#     example_ratios /= np.sum(example_ratios)  # Normalize
    
#     # Reconstruct position
#     position = reconstructor.reconstruct_position(example_ratios)
#     print(f"Reconstructed position: {position}")
