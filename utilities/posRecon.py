import numpy as np
from scipy.optimize import minimize
from .post_analysis import FastPredictor

class PositionReconstructor:
    def __init__(self, model_path=None, scalers_dir=None, bad_sipm=None):
        """
        Initialize reconstructor with MERCI channel positions and ML model
        
        Parameters:
        channel_map: Object with MERCIchan_pos method that returns channel positions
        model_path: str, path to the saved ML model
        scalers_dir: str, directory containing the scalers
        bad_sipm: array-like, list of bad SiPM channels
        """
        # Initialize active channels array
        self.active_channels = []
        
        # Get channels 0-25 (excluding 4 and 18)
        for ichan in range(26):
            if ichan not in [4, 18]:  # Skip disabled channels
                self.active_channels.append(ichan)
            
        # Add PMT channel (31 maps to index 26)
        self.active_channels.append(26)
        
        # Convert to numpy array
        self.active_channels = np.array(self.active_channels)
        
        # Initialize ML predictor
        self.predictor = FastPredictor(
            model_path=model_path, 
            scalers_dir=scalers_dir,
            bad_sipm=bad_sipm
        )
        print("ML predictor initialized")
    
    def get_active_intensities(self, charge_pe_norm):
        """Extract intensities for active channels from the full charge array"""
        return charge_pe_norm[:, self.active_channels]
    
    def expected_intensity(self, source_position):
        """
        Get ML predictions for light intensity distribution
        
        Parameters:
        source_position: array-like, shape (3,)
            The x,y,z coordinates of the position
            
        Returns:
        array of shape (n_active_detectors,)
            Predicted intensity at each active detector
        """
        position = np.array(source_position).reshape(1, -1)
        predictions = self.predictor.predict(position, use_merci=True)[0]
        norm = np.sum(predictions)
        return predictions[self.active_channels] / norm 
    
    def negative_log_likelihood(self, params, measured_intensities):
        """
        Compute the negative log likelihood (KL divergence) between 
        measured and expected intensities plus a regularization term.
        """
        source_position = params[:3]
        expected = self.expected_intensity(source_position)
        
        # KL divergence calculation
        epsilon = 1e-10  # Numerical stability
        kl_div = np.sum(
            (measured_intensities + epsilon) * 
            np.log((measured_intensities + epsilon) / (expected + epsilon))
        )
        
        # Weak regularization
        reg_strength = 0.00001  
        reg_term = reg_strength * np.sum(source_position**2) / 400
        
        return kl_div + reg_term

    def mae_loss(self, params, measured_intensities):
        """
        Compute the mean absolute error (MAE) between measured and expected intensities.
        A weak regularization term is added similar to the KL-based loss.
        """
        source_position = params[:3]
        expected = self.expected_intensity(source_position)
        
        # Mean Absolute Error calculation
        mae = np.mean(np.abs(measured_intensities - expected))
        
        # Weak regularization
        reg_strength = 0.00001
        reg_term = reg_strength * np.sum(source_position**2) / 400
        
        return mae + reg_term

    def reconstruct_position(self, charge_pe_norm, event_idx, initial_guess=None, loss_metric="kl"):
        """
        Reconstruct source position from normalized charge data.
        
        Parameters:
        - charge_pe_norm: numpy.ndarray
          The array containing normalized charge data.
        - event_idx: int
          Index of the event to be reconstructed.
        - initial_guess: array-like, optional
          Initial guess for the reconstruction [x, y, z]. Default is [0, 0, 0].
        - loss_metric: str, optional
          Loss function to use for optimization. Options:
            - "kl": (default) negative log likelihood (KL divergence)
            - "mae": mean absolute error (MAE)
        
        Returns:
        - dict:
            'position': Reconstructed [x, y, z] position,
            'success': Boolean flag indicating if optimization was successful,
            'nll': Final value of the loss function.
        """
        measured_intensities = charge_pe_norm[event_idx, self.active_channels]
        epsilon = 1e-10
        measured_intensities = measured_intensities + epsilon
        measured_intensities = measured_intensities / np.sum(measured_intensities)
        
        # Set initial guess
        initial_guess = np.array([0, 0, 0]) if initial_guess is None else np.array(initial_guess)
        
        # Select the loss function based on loss_metric parameter
        if loss_metric.lower() == "mae":
            loss_fn = self.mae_loss
        else:
            loss_fn = self.negative_log_likelihood
        
        best_result = None
        best_loss = np.inf
        
        methods = ['L-BFGS-B', 'Nelder-Mead']
        
        for method in methods:
            bounds = [(-20, 20), (-20, 20), (-20, 20)] if method == 'L-BFGS-B' else None
            
            # Different options for different methods
            if method == 'L-BFGS-B':
                options = {
                    'maxiter': 5000,
                    'maxfun': 10000,
                    'ftol': 1e-12
                }
            else:
                options = {
                    'maxiter': 5000,
                    'maxfev': 10000,
                    'xatol': 1e-12,
                    'fatol': 1e-12
                }
            
            try:
                result = minimize(
                    loss_fn,
                    initial_guess,
                    args=(measured_intensities,),
                    method=method,
                    bounds=bounds,
                    options=options
                )
                
                if result.success and result.fun < best_loss:
                    best_loss = result.fun
                    best_result = result
                    
            except Exception as e:
                continue
        
        if best_result is None:
            return {
                'position': np.zeros(3),
                'success': False,
                'nll': np.inf
            }
        
        return {
            'position': best_result.x,
            'success': best_result.success,
            'nll': best_result.fun
        }

# Example usage:
# if __name__ == "__main__":
#     # Example assuming you have ChannelMap class and data
#     reconstructor = PositionReconstructor(ChannelMap)
    
#     # Reconstruct single event
#     event_idx = 3
#     result = reconstructor.reconstruct_position(charge_pe_norm, event_idx)
    
#     print("Reconstructed position:", result['position'])
#     print("Total intensity:", result['total_intensity'])
#     print("Success:", result['success'])
#     print("Final NLL:", result['nll'])