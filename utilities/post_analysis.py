import numpy as np
import csv
from ml.ml_Prediction import MLModelInference
import os
import numpy as np
import tensorflow as tf
import joblib


def sipmid_to_merci(photon_ratios, optimize_channels=True):
    """
    Convert photon ratios indexed by SiPM ID to ratios indexed by MERCI channel,
    summing values for SiPMs that map to the same MERCI channel.
    
    Args:
        photon_ratios: numpy array of shape (num_events, num_sipms) where
                       photon_ratios[event, i] corresponds to SiPM ID = i+1
        optimize_channels: If True, remap channels to eliminate empty ones
                
    Returns:
        numpy array of shape (num_events, num_merci_channels) with summed ratios
        and a dictionary mapping original to new channel numbers
    """
    # First, create a mapping from SiPM ID to MERCI channel
    sipm_to_merci = {}
    config_path = '/home/alexli/LoLX_ML/ml/SiPMid_vs_chans.csv'
    
    with open(config_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            sipm_range = row['SiPM ID']
            merci_chan = int(row['MERCI Channel #'])
            
            if "-" in sipm_range:  # Handle ranges
                start, end = map(int, sipm_range.split('-'))
                for sipm_id in range(start, end + 1):
                    sipm_to_merci[sipm_id] = merci_chan
            else:  # Handle single values
                sipm_id = int(sipm_range)
                sipm_to_merci[sipm_id] = merci_chan
    
    # Create output array
    num_events = photon_ratios.shape[0]
    
    # Determine which channels are actually used
    used_channels = set(sipm_to_merci.values())
    all_channels = list(sorted(used_channels))
    
    # Create a mapping from original MERCI channels to optimized indices
    if optimize_channels:
        # Find empty channels (26-30)
        empty_channels = [ch for ch in range(26, 31) if ch not in used_channels]
        print(f"Empty channels that will be skipped: {empty_channels}")
        
        # Create optimized mapping
        channel_map = {}
        new_idx = 0
        
        for ch in range(max(used_channels) + 1):
            if ch in used_channels:
                # If it's channel 31+, see if we can move it down
                if ch >= 31 and empty_channels:
                    # Use the lowest available empty channel
                    new_ch = empty_channels.pop(0)
                    channel_map[ch] = new_ch
                    print(f"Remapped channel {ch} → {new_ch}")
                else:
                    channel_map[ch] = ch
    else:
        # No optimization, use original channels
        channel_map = {ch: ch for ch in range(max(used_channels) + 1)}
    
    # Determine the maximum output channel after remapping
    max_output_channel = max(channel_map.values())
    merci_ratios = np.zeros((num_events, max_output_channel + 1))
    
    # Fill in the ratios with the new mapping
    for sipm_id, merci_chan in sipm_to_merci.items():
        # Get the new channel number after remapping
        new_chan = channel_map.get(merci_chan, merci_chan)
        
        # sipm_id is 1-based, so subtract 1 to get the index in photon_ratios
        sipm_idx = sipm_id - 1
        
        # Make sure sipm_idx is within the bounds of photon_ratios
        if sipm_idx < photon_ratios.shape[1]:
            # Sum the values for this remapped channel
            merci_ratios[:, new_chan] += photon_ratios[:, sipm_idx]
    
    return merci_ratios, channel_map
# Usage with your data



class FastPredictor:
    """
    A lightweight wrapper for efficient predictions from pre-trained models.
    Loads the model and scalers once, then provides fast predictions for new positions.
    """
    
    def __init__(self, model_path=None, scalers_dir=None, config_path=None):
        """
        Initialize the predictor by loading the model and scalers.
        
        Parameters:
        model_path (str, optional): Path to the saved TensorFlow model
        scalers_dir (str, optional): Directory containing the scalers
        config_path (str, optional): Path to model config if available
        """
        # If no paths are provided, try to get them from MLModelInference
        if config_path is None and (model_path is None or scalers_dir is None):
            temp_inference = MLModelInference()
            
            if model_path is None:
                model_path = temp_inference.model_path
                
            if scalers_dir is None:
                scalers_dir = os.path.join(temp_inference.output_dir, "scalers")
                if not os.path.exists(scalers_dir):
                    scalers_dir = temp_inference.output_dir  # Legacy structure fallback
        
        print(f"Loading model from {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        
        # Load X scaler (required)
        self.scaler_X = joblib.load(os.path.join(scalers_dir, "scaler_X.pkl"))
        print("X scaler loaded")
        
        # Check for Y scaler (for inverse transform)
        y_scaler_path = os.path.join(scalers_dir, "scaler_Y.pkl")
        if os.path.exists(y_scaler_path):
            self.scaler_Y = joblib.load(y_scaler_path)
            print("Y scaler loaded")
        else:
            self.scaler_Y = None
            print("No Y scaler found (working with normalized ratios)")
        
        # Check for photon counts scaler
        photon_scaler_path = os.path.join(scalers_dir, "scaler_photons.pkl")
        if os.path.exists(photon_scaler_path):
            self.scaler_photons = joblib.load(photon_scaler_path)
            self.use_photon_counts = True
            print("Photon scaler loaded")
        else:
            self.use_photon_counts = False
        
        print("Predictor initialized and ready for fast predictions")
    
    def predict(self, positions, photon_counts=None):
        """
        Make predictions for the given positions.
        
        Parameters:
        positions (numpy.ndarray or list): Either a single position [x, y, z] 
                                          or multiple positions [[x1, y1, z1], ...]
        photon_counts (numpy.ndarray or list or float, optional): Photon counts if the model uses them
        
        Returns:
        numpy.ndarray: Predictions with shape (n_samples, n_channels)
        """
        # Convert to numpy array
        positions = np.array(positions, dtype=float)
        
        # Handle single position
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        
        # Scale positions
        X_scaled = self.scaler_X.transform(positions)
        
        # Handle photon counts if needed
        if self.use_photon_counts:
            if photon_counts is None:
                # If not provided but required, use a default value
                photon_counts = np.ones(positions.shape[0]) * 1000
            else:
                # Convert to numpy array
                photon_counts = np.array(photon_counts, dtype=float)
                
                # Handle single value
                if photon_counts.ndim == 0:
                    photon_counts = np.array([photon_counts])
                
                # Repeat single value for multiple positions if needed
                if len(photon_counts) == 1 and positions.shape[0] > 1:
                    photon_counts = np.full(positions.shape[0], photon_counts[0])
            
            # Scale photon counts
            photons_scaled = self.scaler_photons.transform(photon_counts.reshape(-1, 1))
            
            # Make predictions with both inputs
            Y_pred_scaled = self.model.predict([X_scaled, photons_scaled], verbose=0)
        else:
            # Make predictions with just positions
            Y_pred_scaled = self.model.predict(X_scaled, verbose=0)
        
        # Inverse transform if needed
        if self.scaler_Y is not None:
            Y_pred = self.scaler_Y.inverse_transform(Y_pred_scaled)
        else:
            Y_pred = Y_pred_scaled
        
        # Ensure non-negative values
        Y_pred = np.maximum(0, Y_pred)
        
        return Y_pred