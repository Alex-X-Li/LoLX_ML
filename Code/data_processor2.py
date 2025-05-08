import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LoLXDataProcessor:
    def __init__(self,  data_path , file_name, output_dir="plots" , num_channels=81):
        self.data_path=data_path
        self.file_name= file_name
        self.file_path= os.path.join(data_path, file_name)
        self.num_channels =num_channels
        self.output_dir =output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_datasets(self):
        with h5py.File(self.file_path, "r") as f:
            self.channel_ids =f["ChannelIDs/ChannelIDs"][:]
            self.channel_charges= f["ChannelCharges/ChannelCharges"][:]
            self.num_hit_channels= f["NumHitChannels/NumHitChannels"][:]
            self.num_photons =f["NumPhotons/NumPhotons"][:]
            self.num_detected =f["NumDetected/NumDetected"][:]
            self.origins=f["Origin/Origin"][:]
        print("Data loaded successfully")

    def create_dataframe(self):
        self.df =pd.DataFrame({"ChannelID": self.channel_ids, "Charge": self.channel_charges})
        self.df_grouped= self.df.groupby("ChannelID")["Charge"].apply(list).reset_index()
        print("DataFrame created and grouped by ChannelID")

    def plot_charge_distributions(self, max_channels=80, batch_size=20):
        print(f"Creating plots in batches of {batch_size}...")
        channels= self.df_grouped["ChannelID"].unique()[:max_channels]
        
        for i in range(0, len(channels), batch_size):
            batch_channels = channels[i:i+batch_size]
            fig,axs = plt.subplots(nrows=batch_size, figsize=(10, 4 * batch_size))
            if batch_size == 1:
                axs = [axs]
            for ax, channel_id in zip(axs, batch_channels):
                charges =self.df_grouped[self.df_grouped["ChannelID"] == channel_id]["Charge"].values[0]
                ax.hist(charges, bins=100, range=(0, 200), histtype='step', linewidth=1, color='blue')
                ax.set_title(f"Charge Distribution for Channel {channel_id}", fontsize=12)
                ax.set_xlabel("Charge (PE)")
                ax.set_ylabel("Count")
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, f"channel_plots_{i//batch_size + 1}.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved: {plot_path}")

    def process_event_data(self):
        num_events =self.origins.shape[0]
        self.photon_counts_matrix = np.zeros((num_events, self.num_channels), dtype=np.float32)
        
        split_indices=np.cumsum(self.num_hit_channels)
        channel_ids_split = np.split(self.channel_ids, split_indices[:-1])
        channel_charges_split = np.split(self.channel_charges, split_indices[:-1])
        
        for event_id, (channels, charges) in enumerate(zip(channel_ids_split, channel_charges_split)):
            self.photon_counts_matrix[event_id, channels - 1] = charges
        
        self.X_input = self.photon_counts_matrix
        self.Y_target = np.hstack([self.origins, self.num_photons.reshape(-1, 1)])
        print("Event data processed into fixed-size input and target arrays")

    def save_numpy_arrays(self, x_filename="X_input.npy", y_filename="Y_target.npy"):
        np.save(x_filename, self.X_input)
        np.save(y_filename, self.Y_target)
        print(f"Saved: {x_filename} (shape: {self.X_input.shape}), {y_filename} (shape: {self.Y_target.shape})")