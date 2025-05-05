import h5py
import numpy as np
import os
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import glob

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now you can import ChannelMap
from ChannelMap import GetSiPMsByType

class LoLXDataProcessor:
    def __init__(self, config_path=None):
        if config_path is None:
            # Default to yaml/config.yaml relative to the script's directory
            config_path = os.path.join(os.path.dirname(__file__), "../yaml/config.yaml")
        
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.data_path = self.config["data"]["path"]
        self.file_pattern = self.config["data"].get("file_pattern", None)  # New line
        self.output_dir = self.config["output"]["path"]
        self.output_h5_file = self.config["output"]["file"]

        # If a file pattern is specified, gather all matching files. Otherwise, read from "files".
        if self.file_pattern:
            pattern_path = os.path.join(self.data_path, self.file_pattern)
            matching_files = glob.glob(pattern_path)
            self.file_names = [os.path.basename(f) for f in matching_files]
        else:
            self.file_names = self.config["data"]["files"]
        
        # Get correction factors from config or use defaults
        self.correction_factors = self.config.get("correction_factors", {})
        self.fbk_correction = self.correction_factors.get("fbk", 1.0)
        self.hpk_correction = self.correction_factors.get("hpk", 1.0)
        self.pmt_correction = self.correction_factors.get("pmt", 1.0)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.origins = None
        self.photon_counts_matrix = None
        self.photon_ratio_matrix = None
        self.photons_per_event = None
        
        # Get channel IDs for different SiPM types
        self.fbk_id, self.fbk_chan = GetSiPMsByType('FBK')
        self.hpk_id, self.hpk_chan = GetSiPMsByType('HPK')
        self.pmt_id, self.pmt_chan = GetSiPMsByType('PMT')
        
        print(f"FBK channels: {len(self.fbk_id)}")
        print(f"HPK channels: {len(self.hpk_id)}")
        print(f"PMT channels: {len(self.pmt_id)}")
        print(f"Processing {len(self.file_names)} input files")

    def load_data(self):
        print("\n Loading Data from multiple files...")
        
        # Lists to hold data from all files
        all_origins = []
        all_photon_counts = []
        all_photons_per_event = []
        
        for file_name in self.file_names:
            file_path = os.path.join(self.data_path, file_name)
            print(f"Processing file: {file_name}")
            
            with h5py.File(file_path, "r") as f:
                origins = f["Origin/Origin"][:]
                channel_ids = f["ChannelIDs/ChannelIDs"][:]
                channel_charges = f["ChannelCharges/ChannelCharges"][:]
                num_hit_channels = f["NumHitChannels/NumHitChannels"][:]
                photons_per_event = f["NumPhotons/NumPhotons"][:]

            split_indices = np.cumsum(num_hit_channels)
            channel_ids_per_event = np.split(channel_ids, split_indices[:-1])
            channel_charges_per_event = np.split(channel_charges, split_indices[:-1])

            num_events = origins.shape[0]
            num_channels = 81
            photon_counts_matrix = np.zeros((num_events, num_channels))

            for event_id in tqdm(range(num_events), desc=f"Processing Events in {file_name}", unit="event"):
                channels = channel_ids_per_event[event_id]
                charges = channel_charges_per_event[event_id]
                photon_counts_matrix[event_id, channels - 1] = charges

            # Append data from this file to our lists
            all_origins.append(origins)
            all_photon_counts.append(photon_counts_matrix)
            all_photons_per_event.append(photons_per_event)
            
            print(f" Loaded {num_events} events from {file_name}")
        
        # Combine data from all files
        self.origins = np.concatenate(all_origins, axis=0)
        self.photon_counts_matrix = np.concatenate(all_photon_counts, axis=0)
        self.photons_per_event = np.concatenate(all_photons_per_event, axis=0)
        
        print(f" Combined data: Total events {self.origins.shape[0]}")
        print(f" X shape {self.origins.shape}, Y shape {self.photon_counts_matrix.shape}")

    def apply_corrections(self):
        print("\n Applying correction factors to different SiPM types...")
        
        # Create a copy of the uncorrected data
        self.uncorrected_matrix = self.photon_counts_matrix.copy()
        
        # Convert IDs to 0-based indices for the matrix
        fbk_indices = [id - 1 for id in self.fbk_id]
        hpk_indices = [id - 1 for id in self.hpk_id]
        pmt_indices = [id - 1 for id in self.pmt_id]
        
        # Apply correction factors
        for idx in fbk_indices:
            self.photon_counts_matrix[:, idx] /= self.fbk_correction
        
        for idx in hpk_indices:
            self.photon_counts_matrix[:, idx] /= self.hpk_correction
        
        for idx in pmt_indices:
            self.photon_counts_matrix[:, idx] /= self.pmt_correction
        
        print(f" Applied correction factors: FBK: {self.fbk_correction}, HPK: {self.hpk_correction}, PMT: {self.pmt_correction}")

    def normalize_photon_counts(self):
        """Normalize photon counts by total photons per event to create ratio matrix"""
        print("\n Normalizing photon counts to create ratio matrix...")
        
        # Create a new matrix for the normalized ratios
        self.photon_ratio_matrix = np.zeros_like(self.photon_counts_matrix)
        
        # Calculate ratios by dividing each channel's count by total photons per event
        for event_id in range(self.photon_counts_matrix.shape[0]):
            total_photons = self.photons_per_event[event_id]
            
            # Avoid division by zero
            if total_photons > 0:
                self.photon_ratio_matrix[event_id, :] = self.photon_counts_matrix[event_id, :] / total_photons
        
        # Also create uncorrected ratios
        self.uncorrected_ratio_matrix = np.zeros_like(self.uncorrected_matrix)
        for event_id in range(self.uncorrected_matrix.shape[0]):
            total_photons = self.photons_per_event[event_id]
            if total_photons > 0:
                self.uncorrected_ratio_matrix[event_id, :] = self.uncorrected_matrix[event_id, :] / total_photons
        
        print(f" Created photon ratio matrix with shape {self.photon_ratio_matrix.shape}")
        print(f" Ratio values range from {np.min(self.photon_ratio_matrix)} to {np.max(self.photon_ratio_matrix)}")

    def save_data(self):
        print("\n Saving combined data...")
        h5_path = os.path.join(self.output_dir, self.output_h5_file)
        
        with h5py.File(h5_path, "w") as f:
            # Create groups for better organization
            ml_group = f.create_group("ml_data")
            event_group = f.create_group("events")
            channel_group = f.create_group("channels")
            config_group = f.create_group("config")
            
            # Save ML-ready data
            ml_group.create_dataset("X", data=self.origins, compression="gzip")
            ml_group.create_dataset("Y", data=self.photon_ratio_matrix, compression="gzip")
            
            # Save event data
            event_group.create_dataset("origins", data=self.origins, compression="gzip")
            event_group.create_dataset("photon_counts", data=self.photon_counts_matrix, compression="gzip")
            event_group.create_dataset("uncorrected_counts", data=self.uncorrected_matrix, compression="gzip")
            event_group.create_dataset("photon_ratios", data=self.photon_ratio_matrix, compression="gzip")
            event_group.create_dataset("uncorrected_ratios", data=self.uncorrected_ratio_matrix, compression="gzip")
            event_group.create_dataset("photons_per_event", data=self.photons_per_event, compression="gzip")
            event_group.create_dataset("source_photons", data=self.photons_per_event, compression="gzip")
            
            # Save channel type information
            channel_group.create_dataset("fbk_channels", data=np.array(self.fbk_id))
            channel_group.create_dataset("hpk_channels", data=np.array(self.hpk_id))
            channel_group.create_dataset("pmt_channels", data=np.array(self.pmt_id))
            
            # Save correction factors
            config_group.attrs["fbk_correction"] = self.fbk_correction
            config_group.attrs["hpk_correction"] = self.hpk_correction
            config_group.attrs["pmt_correction"] = self.pmt_correction
            
            # Save additional metadata
            f.attrs["created_by"] = "LoLXDataProcessor"
            f.attrs["date_processed"] = np.string_(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
            f.attrs["source_files"] = np.string_(", ".join(self.file_names))
            f.attrs["num_files_processed"] = len(self.file_names)
            f.attrs["ml_x_feature"] = np.string_("Event origins (x, y, z)")
            f.attrs["ml_y_feature"] = np.string_("Normalized photon ratios per channel")
        
        print(f" Saved combined data from {len(self.file_names)} files to: {h5_path}")

    def plot_data(self):
        print("\n Create plots")

        plot_config = self.config.get("plot", {})
        dpi = plot_config.get("dpi", 300)
        show_plots = plot_config.get("show_plots", False)

        df_origin = pd.DataFrame(self.origins, columns=['x', 'y', 'z'])
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df_origin["x"], df_origin["y"], df_origin["z"], alpha=0.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Event Origin Distribution")
        plt.savefig(os.path.join(self.output_dir, "event_origins.png"), dpi=dpi)
        if show_plots:
            plt.show()
        plt.close()
        print(" Saved event_origins.png")

        # Plot normalized ratios for channel 81
        channel_81_ratios = self.photon_ratio_matrix[:, 80]
        plt.figure(figsize=(8, 6))
        plt.hist(channel_81_ratios, bins=50, range=(0, 0.2), histtype='step', linewidth=1.5, color='red')
        plt.xlabel("Photon Ratio")
        plt.ylabel("Event Count")
        plt.title("Normalized Photon Ratio Distribution for Channel 81")
        plt.savefig(os.path.join(self.output_dir, "channel_81_ratio.png"), dpi=dpi)
        if show_plots:
            plt.show()
        plt.close()
        print(" Saved channel_81_ratio.png")

        # Plot distributions by SiPM type
        self.plot_by_sipm_type(show_plots, dpi)

        # Plot photon ratios by channel groups
        batch_size = 20
        num_batches = 4
        
        for batch in tqdm(range(num_batches), desc="Batch Plotting", unit="batch"):
            fig, axes = plt.subplots(5, 4, figsize=(15, 12))
            fig.suptitle(f"Photon Ratio Distributions for Channels {batch * batch_size + 1} to {(batch + 1) * batch_size}")

            for i, channel in enumerate(range(batch * batch_size, (batch + 1) * batch_size)):
                ax = axes[i // 4, i % 4]
                ratios = self.photon_ratio_matrix[:, channel]
                ax.hist(ratios, bins=50, range=(0, 0.05), histtype='step', linewidth=1.5, color='blue')
                ax.set_title(f"Channel {channel+1}")
                ax.set_xlabel("Photon Ratio")
                ax.set_ylabel("Count")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(self.output_dir, f"channel_ratios_{batch+1}.png"), dpi=dpi)
            if show_plots:
                plt.show()
            plt.close()
            print(f" Saved channel_ratios_{batch+1}.png")

        print("\n All plots are saved successfully")

    def plot_by_sipm_type(self, show_plots, dpi):
        """Plot photon ratio distributions by SiPM type"""
        
        plt.figure(figsize=(10, 8))
        
        # Plot average photon ratio by SiPM type
        fbk_indices = [id - 1 for id in self.fbk_id]
        hpk_indices = [id - 1 for id in self.hpk_id]
        pmt_indices = [id - 1 for id in self.pmt_id]
        
        fbk_ratios = np.sum(self.photon_ratio_matrix[:, fbk_indices], axis=1)
        hpk_ratios = np.sum(self.photon_ratio_matrix[:, hpk_indices], axis=1)
        pmt_ratios = np.sum(self.photon_ratio_matrix[:, pmt_indices], axis=1)
        sum_ratios = fbk_ratios + hpk_ratios + pmt_ratios
        
        plt.hist(fbk_ratios, bins=50, histtype='step', linewidth=1.5, 
                 color='blue', label=f'FBK (corr: {self.fbk_correction:.2f})')
        plt.hist(hpk_ratios, bins=50,  histtype='step', linewidth=1.5, 
                 color='red', label=f'HPK (corr: {self.hpk_correction:.2f})')
        plt.hist(pmt_ratios, bins=50,  histtype='step', linewidth=1.5, 
                 color='green', label=f'PMT (corr: {self.pmt_correction:.2f})')
        plt.hist(sum_ratios, bins=50, histtype='step', linewidth=1.5,
                 color='orange', label='Sum')
        
        plt.xlabel("Average Photon Ratio")
        plt.ylabel("Event Count")
        plt.title("Average Photon Ratio Distribution by SiPM Type (After Correction)")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "sipm_type_ratio_comparison.png"), dpi=dpi)
        if show_plots:
            plt.show()
        plt.close()
        print(" Saved sipm_type_ratio_comparison.png")

    def run(self):
        self.load_data()
        self.apply_corrections()
        self.normalize_photon_counts()
        self.save_data()
        self.plot_data()

if __name__ == "__main__":
    processor = LoLXDataProcessor()
    processor.run()