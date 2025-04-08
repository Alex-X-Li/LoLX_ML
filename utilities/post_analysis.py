import numpy as np
import csv
from ml.ml_Prediction import MLModelInference
import os
import numpy as np
import tensorflow as tf
import joblib


def sipmid_to_merci(photon_ratios, bad_sipm=None, config_path=None):
    """
    Convert photon ratios indexed by SiPM ID to ratios indexed by MERCI channel.
    
    Args:
        photon_ratios (np.ndarray): Array of shape (num_events, num_sipms)
        bad_sipm (list): List of SiPM IDs to ignore (1-based indexing)
        config_path (str): Optional path to SiPM mapping config file
    
    Returns:
        numpy.ndarray: Array of shape (num_events, 27) with MERCI channel mapping
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../ml/SiPMid_vs_chans.csv')
    
    # Convert bad_sipm to set for faster lookup
    bad_sipm_set = set(bad_sipm) if bad_sipm is not None else set()
    
    # Load SiPM to MERCI channel mapping
    sipm_to_merci = {}
    with open(config_path, mode='r') as file:
        for row in csv.DictReader(file):
            sipm_range = row['SiPM ID']
            merci_chan = int(row['MERCI Channel #'])
            
            # Handle both single values and ranges
            if "-" in sipm_range:
                start, end = map(int, sipm_range.split('-'))
                sipm_to_merci.update({i: merci_chan for i in range(start, end + 1) 
                                    if i not in bad_sipm_set})
            else:
                sipm_id = int(sipm_range)
                if sipm_id not in bad_sipm_set:
                    sipm_to_merci[sipm_id] = merci_chan

    # Initialize output array with 27 channels (0-26)
    merci_ratios = np.zeros((photon_ratios.shape[0], 27))

    # Map SiPM ratios to MERCI channels, skipping bad SiPMs
    valid_sipms = [(sipm_id-1, merci_chan if merci_chan != 31 else 26)
                   for sipm_id, merci_chan in sipm_to_merci.items()
                   if sipm_id-1 < photon_ratios.shape[1]]
    
    for sipm_idx, new_chan in valid_sipms:
        merci_ratios[:, new_chan] += photon_ratios[:, sipm_idx]

    return merci_ratios



class FastPredictor:
    """
    A lightweight wrapper for efficient predictions from pre-trained models.
    Loads the model and scalers once, then provides fast predictions for new positions.
    """
    
    def __init__(self, bad_sipm = None , model_path=None, scalers_dir=None, config_path=None):
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
        self.bad_sipm = bad_sipm
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
    
    def predict(self, positions, photon_counts=None, use_merci=False):
        """
        Make predictions for the given positions.
        
        Parameters:
        positions (numpy.ndarray or list): Either a single position [x, y, z] 
                                          or multiple positions [[x1, y1, z1], ...]
        photon_counts (numpy.ndarray or list or float, optional): Photon counts if the model uses them
        use_merci (bool): If True, convert predictions to MERCI channel mapping
        
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
        
        # Mask bad SiPMs with zeros (bad_sipm uses 1-based indexing)
        if self.bad_sipm is not None:
            mask = np.ones(Y_pred.shape[1], dtype=bool)
            for sipm_id in self.bad_sipm:
                mask[sipm_id-1] = False
            Y_pred = Y_pred * mask  # Broadcasting will apply to all positions
        
        # Convert to MERCI channels if requested
        if use_merci:
            Y_pred_merci = sipmid_to_merci(Y_pred, bad_sipm=self.bad_sipm)
            return Y_pred_merci
        else:
            return Y_pred