import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import numpy as np

# corrected channel mapping
ChannelMap = {
    (-21.01, -12.64, 12.60): 1,
    (-21.01, -4.24, 12.60): 2,
    (-21.01, -12.64, 4.20): 3,
    (-21.01, -4.24, 4.20): 4,
    (-21.01, 5.15, 11.65): 5,
    (-21.01, 11.65, 11.65): 6,
    (-21.01, 5.15, 5.15): 7,
    (-21.01, 11.65, 5.15): 8,
    (-21.01, 4.24, -4.20): 9,
    (-21.01, 12.64, -4.20): 10,
    (-21.01, 4.24, -12.60): 11,
    (-21.01, 12.64, -12.60): 12,
    (-21.01, -11.65, -5.15): 13,
    (-21.01, -5.15, -5.15): 14,
    (-21.01, -11.65, -11.65): 15,
    (-21.01, -5.15, -11.65): 16,
    (-12.64, 12.61, -21.00): 17,
    (-4.24, 12.61, -21.00): 18,
    (-12.64, 4.21, -21.01): 19,
    (-4.24, 4.21, -21.01): 20,
    (5.15, 11.66, -21.00): 21,
    (11.65, 11.66, -21.00): 22,
    (5.15, 5.16, -21.01): 23,
    (11.65, 5.16, -21.01): 24,
    (4.24, -4.19, -21.01): 25,
    (12.64, -4.19, -21.01): 26,
    (4.24, -12.59, -21.01): 27,
    (12.64, -12.59, -21.01): 28,
    (-11.65, -5.14, -21.01): 29,
    (-5.15, -5.14, -21.01): 30,
    (-11.65, -11.64, -21.01): 31,
    (-5.15, -11.64, -21.01): 32,
    (21.01, 12.64, 12.60): 33,
    (21.01, 4.24, 12.60): 34,
    (21.01, 12.64, 4.20): 35,
    (21.01, 4.24, 4.20): 36,
    (21.01, -5.15, 11.65): 37,
    (21.01, -11.65, 11.65): 38,
    (21.01, -5.15, 5.15): 39,
    (21.01, -11.65, 5.15): 40,
    (21.01, -4.24, -4.20): 41,
    (21.01, -12.64, -4.20): 42,
    (21.01, -4.24, -12.60): 43,
    (21.01, -12.64, -12.60): 44,
    (21.01, 11.65, -5.15): 45,
    (21.01, 5.15, -5.15): 46,
    (21.01, 11.65, -11.65): 47,
    (21.01, 5.15, -11.65): 48,
    (-12.64, 21.00, 12.61): 49,
    (-4.24, 21.00, 12.61): 50,
    (-12.64, 21.01, 4.21): 51,
    (-4.24, 21.01, 4.21): 52,
    (5.15, 21.00, 11.66): 53,
    (11.65, 21.00, 11.66): 54,
    (5.15, 21.01, 5.16): 55,
    (11.65, 21.01, 5.16): 56,
    (4.24, 21.01, -4.19): 57,
    (12.64, 21.01, -4.19): 58,
    (4.24, 21.01, -12.59): 59,
    (12.64, 21.01, -12.59): 60,
    (-11.65, 21.01, -5.14): 61,
    (-5.15, 21.01, -5.14): 62,
    (-11.65, 21.01, -11.64): 63,
    (-5.15, 21.01, -11.64): 64,
    (12.64, -21.01, 12.59): 65,
    (4.24, -21.01, 12.59): 66,
    (12.64, -21.01, 4.19): 67,
    (4.24, -21.01, 4.19): 68,
    (-5.15, -21.01, 11.64): 69,
    (-11.65, -21.01, 11.64): 70,
    (-5.15, -21.01, 5.14): 71,
    (-11.65, -21.01, 5.14): 72,
    (-4.24, -21.01, -4.21): 73,
    (-12.64, -21.01, -4.21): 74,
    (-4.24, -21.00, -12.61): 75,
    (-12.64, -21.00, -12.61): 76,
    (11.65, -21.01, -5.16): 77,
    (5.15, -21.01, -5.16): 78,
    (11.65, -21.00, -11.66): 79,
    (5.15, -21.00, -11.66): 80,
    (-0.00, -0.14, 21.05): 81
    }

bad_sipm1 = np.array([5,6,7,8,11,25, 36,42,44, 64,66])

def GetSiPM_pos(sipm_id):
    for position, id in ChannelMap.items():
        if id == sipm_id:
            return position
    return None


SiPM_Map = {v: k for k, v in ChannelMap.items()}

## test plot
x_coords = [coords[0] for coords in SiPM_Map.values()]
y_coords = [coords[1] for coords in SiPM_Map.values()]
z_coords = [coords[2] for coords in SiPM_Map.values()]
channels = list(SiPM_Map.keys())

# Set up a 3D plot
scatter = go.Scatter3d(
    x=x_coords,
    y=y_coords,
    z=z_coords,
    mode='markers',
    marker=dict(size=5, color='blue'),
    text=channels,
    hoverinfo='text'
)

# Create the figure and add the scatter plot to it
# fig = go.Figure(data=[scatter])

# # Customize the layout
# fig.update_layout(
#     title='3D Plot of SiPM Map',
#     scene=dict(
#         xaxis_title='X [mm]',
#         yaxis_title='Y [mm]',
#         zaxis_title='Z [mm]'
#     )
# )

# # Display the plot
# fig.show()


def SiPM_to_DAQchan(sipm_id):
    config_path = '/home/alexli/merci/lolx/datafiles/sipm_info_lolx2_aug2023.csv'  # Adjust the path as necessary
    with open(config_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            sipm_range = row['SiPM ID']
            if "-" in sipm_range:  # Handle ranges
                start, end = map(int, sipm_range.split('-'))
                if start <= sipm_id <= end:
                    return row['DAQ Channel (0-31)']
            else:  # Handle single values
                if sipm_id == int(sipm_range):
                    return row['DAQ Channel (0-31)']
    return "DAQ channel not found for SiPM ID {}".format(sipm_id)
 

def DAQchan_to_SiPM(daq_channel):
    config_path = '/home/alexli/merci/lolx/datafiles/sipm_info_lolx2_aug2023.csv'
    daq_to_sipm_map = {}
    with open(config_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Wavelength Filter'] == 'empty':  # Ignore rows with "empty" in "Wavelength Filter"
                continue
            row_daq_channel = int(row['DAQ Channel (0-31)'])
            sipm_id = row['SiPM ID']
            if '-' in sipm_id:
                sipm_id = list(range(int(sipm_id.split('-')[0]), int(sipm_id.split('-')[1]) + 1))
            else:
                sipm_id = [int(sipm_id)]
            daq_to_sipm_map[row_daq_channel] = sipm_id

    return daq_to_sipm_map.get(daq_channel, None)

def MERCIchan_to_SiPM(merci_chan):
    config_path = '/home/alexli/merci/scripts/lolx/SiPMid_vs_chans.csv'
    sipmID = []
    with open(config_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if int(row['MERCI Channel #']) == merci_chan:
                sipmID.append(int(row['SiPM ID']))
        return sipmID

def SiPM_to_MERCIchan(sipm_id):
    config_path = '/home/alexli/merci/scripts/lolx/SiPMid_vs_chans.csv'
    with open(config_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            sipm_range = row['SiPM ID']
            if "-" in sipm_range:  # Handle ranges
                start, end = map(int, sipm_range.split('-'))
                if start <= sipm_id <= end:
                    return int(row['MERCI Channel #'])
            else:  # Handle single values
                if sipm_id == int(sipm_range):
                    return int(row['MERCI Channel #'])
    return "MERCI channel not found for SiPM ID {}".format(sipm_id)

def MERCIchan_pos(merci_chan):
    sipmID = MERCIchan_to_SiPM(merci_chan)
    if not sipmID:
        return None

    sipm_positions = [GetSiPM_pos(sipm) for sipm in sipmID if sipm not in bad_sipm1]
    if not sipm_positions:
        return None

    avg_x = sum(pos[0] for pos in sipm_positions) / len(sipm_positions)
    avg_y = sum(pos[1] for pos in sipm_positions) / len(sipm_positions)
    avg_z = sum(pos[2] for pos in sipm_positions) / len(sipm_positions)

    return (avg_x, avg_y, avg_z)

def MERCIchan_area(merci_chan):
    sipm_area = 6*6 # mm^2
    pmt_area = 20.5*20.5 # mm^2

    if merci_chan == 31:
        return pmt_area
    
    sipmID = MERCIchan_to_SiPM(merci_chan)
    if not sipmID:
        return None
    
    valid_sipmID = [sipm for sipm in sipmID if sipm not in bad_sipm1]
    total_area = len(valid_sipmID) * sipm_area

    return total_area
    
def GetSiPMsByType(sipm_type, csv_file_path='SiPMid_vs_chans.csv'):
    """
    Get all SiPM IDs and MERCI channels of a specific type (FBK, HPK, or PMT).
    
    Args:
        sipm_type (str): The SiPM type to look up (FBK, HPK, or PMT)
        csv_file_path (str): Path to the CSV file with SiPM data
            
    Returns:
        list: List of tuples (sipm_id, merci_channel) for all SiPMs of the specified type,
              or empty list if no SiPMs are found for this type
    """
    # Load numeric data (SiPM IDs and MERCI channels)
    data = np.loadtxt(
        csv_file_path, 
        delimiter=',', 
        skiprows=1,
        usecols=(0, 1),
        dtype=int
    )
    
    # Load the type column
    types = np.loadtxt(
        csv_file_path,
        delimiter=',',
        skiprows=1,
        usecols=(2,),
        dtype=str
    )
    
    # Extract columns
    all_sipm_ids = data[:, 0]
    all_merci_channels = data[:, 1]
    
    # Find all indices where type matches
    indices = np.where(types == sipm_type)[0]
    
    # Get the matching SiPM IDs and MERCI channels
    sipm_ids = all_sipm_ids[indices]
    merci_channels = all_merci_channels[indices]
    merci_channels = np.unique(merci_channels)  # Remove duplicates
    
    return sipm_ids.astype(int), merci_channels.astype(int)
## test
# print(DAQchan_to_SiPM(8))
# print(MERCIchan_to_SiPM(9))
# print(SiPM_to_DAQchan(72))
# print(SiPM_Map[for i_sipm in DAQchan_to_SiPM(8): i_sipm])