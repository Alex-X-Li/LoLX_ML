import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

class PositionEnergyPredictor:
    def __init__(self, model_path, x_mean_path, x_scale_path, y_mean_path, y_scale_path):
        self.model = load_model(model_path)
        self.X_mean = np.load(x_mean_path)
        self.X_scale = np.load(x_scale_path)
        self.Y_mean = np.load(y_mean_path)
        self.Y_scale = np.load(y_scale_path)

    def scale_input(self, X):
        return (X - self.X_mean) / self.X_scale

    def inverse_transform_Y(self, Y_scaled):
        return Y_scaled * self.Y_scale + self.Y_mean

    def predict_event(self, photon_counts):
        x_input = self.scale_input(photon_counts)
        y_pred_scaled = self.model.predict(x_input[np.newaxis])[0]
        return self.inverse_transform_Y(y_pred_scaled)

    def evaluate(self, X, Y, label):
        Y_pred = self.model.predict(X)
        Y_true = self.inverse_transform_Y(Y)
        Y_pred = self.inverse_transform_Y(Y_pred)

        mae = mean_absolute_error(Y_true, Y_pred, multioutput='raw_values')
        mse = mean_squared_error(Y_true, Y_pred, multioutput='raw_values')

        print(f"{label} MAE (x, y, z, E):", mae)
        print(f"{label} MSE (x, y, z, E):", mse)

        return Y_true, Y_pred

    def plot_history(self, history):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label="Train Loss")
        plt.plot(history.history['val_loss'], label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE)")
        plt.title("Loss Over Epochs")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label="Train MAE")
        plt.plot(history.history['val_mae'], label="Val MAE")
        plt.xlabel("Epochs")
        plt.ylabel("MAE")
        plt.title("MAE Over Epochs")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_predictions(self, Y_true, Y_pred):
        labels = ['x', 'y', 'z', 'Energy']
        plt.figure(figsize=(14, 10))
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.scatter(Y_true[:, i], Y_pred[:, i], s=5, alpha=0.3)
            plt.plot([Y_true[:, i].min(), Y_true[:, i].max()],
                     [Y_true[:, i].min(), Y_true[:, i].max()], 'r--')
            plt.xlabel(f"True {labels[i]}")
            plt.ylabel(f"Predicted {labels[i]}")
            plt.title(f"{labels[i]} Prediction vs Ground Truth")
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_position_error(self, Y_true, Y_pred):
        true_pos = Y_true[:, :3]
        pred_pos = Y_pred[:, :3]
        errors = np.linalg.norm(pred_pos - true_pos, axis=1)

        plt.figure(figsize=(8, 6))
        plt.hist(errors, bins=100, color='dodgerblue', alpha=0.8)
        plt.xlabel("Position Error (mm or cm)")
        plt.ylabel("Number of Events")
        plt.title("Histogram of Position Reconstruction Error")
        plt.grid(True)
        plt.show()

    def plot_energy_error(self, Y_true, Y_pred):
        true_energy = Y_true[:, 3]
        pred_energy = Y_pred[:, 3]
        errors = (pred_energy - true_energy) / true_energy

        plt.figure(figsize=(8, 6))
        plt.hist(errors, bins=100, color='orange', alpha=0.8)
        plt.xlabel("Normalized Energy Error")
        plt.ylabel("Number of Events")
        plt.title("Histogram of Normalized Energy Reconstruction Error")
        plt.grid(True)
        plt.show()

    def plot_energy_distribution(self, Y_true, Y_pred):
        true_energy = Y_true[:, 3]
        pred_energy = Y_pred[:, 3]

        plt.figure(figsize=(8, 5))
        bins = np.linspace(min(true_energy.min(), pred_energy.min()),
                           max(true_energy.max(), pred_energy.max()), 50)

        plt.hist(true_energy, bins=bins, alpha=0.6, label='True Photon Count', color='blue', edgecolor='black')
        plt.hist(pred_energy, bins=bins, alpha=0.6, label='Predicted Photon Count', color='orange', edgecolor='black')

        plt.title("Photon Count Distribution: True vs Predicted")
        plt.xlabel("Photon Count (Energy)")
        plt.ylabel("Number of Events")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
