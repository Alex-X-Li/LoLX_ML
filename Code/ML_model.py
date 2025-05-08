import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class PhotonModelTrainer:
    def __init__(self, x_path="X_input.npy",y_path="Y_target.npy"):
        self.x_path=x_path
        self.y_path=y_path
        self.X_scaler=StandardScaler()
        self.Y_scaler=StandardScaler()
        self.model=None

    def load_and_prepare_data(self):
        X=np.load(self.x_path)
        Y=np.load(self.y_path)
        X=self.X_scaler.fit_transform(X)
        Y=self.Y_scaler.fit_transform(Y)
        X_train,X_temp,Y_train, Y_temp =train_test_split(X, Y, test_size=0.2, random_state=42)
        X_val,X_test, Y_val,Y_test= train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)
        return X_train, X_val,X_test, Y_train, Y_val, Y_test

    def build_model(self, input_dim=81, output_dim=4):
        inp = Input(shape=(input_dim,))
        x = Dense(128, activation='relu')(inp)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        out = Dense(output_dim)(x)
        self.model = Model(inputs=inp, outputs=out)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def train(self,X_train,Y_train, X_val, Y_val):
        history = self.model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=50,
            batch_size=32,
            verbose=1
        )
        return history

    def evaluate_and_save(self, X_test, Y_test):
        loss,mae=self.model.evaluate(X_test, Y_test)
        print(f"Test Loss: {loss:.4f}, MAE: {mae:.4f}")
        self.model.save("position_energy_model.keras")
        np.save("X_scaler_mean.npy", self.X_scaler.mean_)
        np.save("X_scaler_scale.npy", self.X_scaler.scale_)
        np.save("Y_scaler_mean.npy", self.Y_scaler.mean_)
        np.save("Y_scaler_scale.npy", self.Y_scaler.scale_)