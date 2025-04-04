import h5py
import numpy as np
import os
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

class LoLXDataProcessor:
    def __init__(self, config_path=None):
        if config_path is None:
            # Default to yaml/config.yaml relative to the script's directory
            config_path = os.path.join(os.path.dirname(__file__), "../yaml/config.yaml")
        
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.data_path = self.config["data"]["path"]
        self.file_name = self.config["data"]["file"]
        self.output_dir = self.config["output"]["dir"]
        self.x_filename = self.config["output"]["x_file"]
        self.y_filename = self.config["output"]["y_file"]
        os.makedirs(self.output_dir, exist_ok=True)
        self.file_path = os.path.join(self.data_path, self.file_name)
        self.origins = None
        self.photon_counts_matrix = None

    def load_data(self):
        print("\n Loading Data...")
        with h5py.File(self.file_path, "r") as f:
            self.origins = f["Origin/Origin"][:]
            channel_ids = f["ChannelIDs/ChannelIDs"][:]
            channel_charges = f["ChannelCharges/ChannelCharges"][:]
            num_hit_channels = f["NumHitChannels/NumHitChannels"][:]

        split_indices = np.cumsum(num_hit_channels)
        channel_ids_per_event = np.split(channel_ids, split_indices[:-1])
        channel_charges_per_event = np.split(channel_charges, split_indices[:-1])

        num_events = self.origins.shape[0]
        num_channels = 81
        self.photon_counts_matrix = np.zeros((num_events, num_channels))

        for event_id in tqdm(range(num_events), desc="Processing Events", unit="event"):
            channels = channel_ids_per_event[event_id]
            charges = channel_charges_per_event[event_id]
            self.photon_counts_matrix[event_id, channels - 1] = charges

        print(f" Loaded data: X shape {self.origins.shape}, Y shape {self.photon_counts_matrix.shape}")

    def save_data(self):
        print("\n Saving Data...")
        h5_path = os.path.join(self.output_dir, "processed_data.h5")
        
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("origins", data=self.origins, compression="gzip")
            f.create_dataset("photon_counts", data=self.photon_counts_matrix, compression="gzip")
        
        print(f" Saved processed data to {h5_path}")

    def plot_data(self):
        print("\n Create plots")

        df_origin = pd.DataFrame(self.origins, columns=['x', 'y', 'z'])
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df_origin["x"], df_origin["y"], df_origin["z"], alpha=0.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Event Origin Distribution")
        plt.savefig(os.path.join(self.output_dir, "event_origins.png"))
        plt.close()
        print(" Saved event_origins.png")

        channel_81_charges = self.photon_counts_matrix[:, 80]
        plt.figure(figsize=(8, 6))
        plt.hist(channel_81_charges, bins=50, range=(200, 2000), histtype='step', linewidth=1.5, color='red')
        plt.xlabel("Charge (PE)")
        plt.ylabel("Event Count")
        plt.title("Charge Distribution for Channel 81")
        plt.savefig(os.path.join(self.output_dir, "channel_81.png"))
        plt.close()
        print(" Saved channel_81.png")

        batch_size = 20
        num_batches = 4
        
        for batch in tqdm(range(num_batches), desc="Batch Plotting", unit="batch"):
            fig, axes = plt.subplots(5, 4, figsize=(15, 12))
            fig.suptitle(f"Charge Distributions for Channels {batch * batch_size + 1} to {(batch + 1) * batch_size}")

            for i, channel in enumerate(range(batch * batch_size, (batch + 1) * batch_size)):
                ax = axes[i // 4, i % 4]
                charges = self.photon_counts_matrix[:, channel]
                ax.hist(charges, bins=50, range=(0, 200), histtype='step', linewidth=1.5, color='blue')
                ax.set_title(f"Channel {channel+1}")
                ax.set_xlabel("Charge (PE)")
                ax.set_ylabel("Count")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(self.output_dir, f"channels_{batch+1}.png"))
            plt.close()
            print(f"\n Saved channels_{batch+1}.png")

        print("\n All plots are saved successfully")

    def run(self):
        self.load_data()
        self.save_data()
        self.plot_data()

if __name__ == "__main__":
    processor = LoLXDataProcessor()
    processor.run()
