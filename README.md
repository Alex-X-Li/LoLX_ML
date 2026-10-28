# LoLX_ML Project

## Overview
LoLX_ML is a machine learning project designed for Light Only Liquid Xenon (LoLX) experiments. It provides tools for position reconstruction and charge prediction using neural networks.

## Project Structure
```
LoLX_ML
├── ml
│   ├── Beta_data_processor.py       # Processes beta simulation data for ML
│   ├── ChannelMap.py                # Maps SiPM IDs to channels and positions
│   ├── ml_Prediction.py             # Main inference class for model predictions
│   ├── ml_trainer.py                # Model training script
│   ├── ml_trainer_kl.py             # Model training script with KL divergence
│   ├── photonBomb_data_processor.py # Processes photon bomb data
│   ├── SiPMid_vs_chans.csv          # SiPM ID to channel mapping
├── utilities
│   ├── posRecon.py                  # Position reconstruction utilities
│   ├── posRecon_ml.py               # ML-based position reconstruction
├── test                             # Directory for test scripts
├── training_data                    # Directory for training data
│   ├── scalers                      # Directory for data scalers
│   ├── trained_model.keras          # Trained forward model
│   ├── inverse_model.keras          # Trained inverse model
├── yaml
│   ├── config.yaml                  # Configuration for data processing
│   ├── configML.yaml                # Configuration for postion to charge distribution
│   ├── configML_inverse.yaml        # Configuration for charge distribution to position
│   ├── configPred.yaml              # Configuration for model inference
│   ├── configInverseValidate.yaml   # Configuration for inverse model validation
├── README.md                        # Project documentation
```

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd LoLX_ML
```

## Usage
```bash
python ml/Beta_data_processor.py --config yaml/config.yaml
python ml/ml_trainer.py --config yaml/configML.yaml
python ml/ml_Prediction.py --config yaml/configPred.yaml
```

### Data Processing
- Use `ml/Beta_data_processor.py` or `ml/photonBomb_data_processor.py` to process raw data into a format suitable for training or inference.
- Configuration files for data processing are located in the `yaml/` directory.

### Model Training
- Train forward models using `ml/ml_trainer.py` or `ml/ml_trainer_kl.py`.
- Train inverse models using `ml/ml_trainer.py` with `yaml/configML_inverse.yaml`.
- Update YAML configuration files with appropriate paths and parameters for training.

### Model Inference
- Run predictions using `ml/ml_Prediction.py`.
- Ensure the appropriate config YAML is properly configured with the paths to the model and data.

### Position Reconstruction
- Use `utilities/posRecon.py` and `utilities/posRecon_ml.py` for position reconstruction.
- Jupyter notebooks demonstrate position reconstruction workflow.
- The MLPositionReconstructor class handles loading models and scalers for fast predictions.

### Digitizer Saturation (Run-1 data)
Bright events clip the digitizer (-1.05 V). The lolx_ana PE files mark each SiPM MERCI channel as `sat` 0 (unsaturated),
1 (clipped, recovered `pe_rec`) or 2 (deeply clipped, only `pe_lo <= true <= pe_hi`). `ml/saturation.py` brings this
into training and reconstruction:
- `SaturationEmulator`: simulated per-SiPM counts -> MERCI channels (Run-1 turned-off SiPMs removed) -> the same
  saturated fields as data. Each channel saturates at its own PE level, `|clip voltage| / (pulse height per PE)`
  measured from data (83-139 PE for HPK, 228-365 PE for FBK channels).
- `build_features`: the inverse-model input from simulation or data alike (27 ratios with `sat` 2 at `pe_lo`,
  48 saturation flags, log light outside the `sat` 2 channels).
- `npe_central`: emitted photons from the channels without `sat` 2 (forward-model LCE), so clipped channels do not
  pull the energy low.
- Calibration: `python scripts/calibrate_saturation.py --pe-dir <lolx_ana run1/pe>` writes
  `yaml/saturation_calib.yaml` (per-channel saturation PE level, plateau and recovery fits) and
  `yaml/saturation_hi_over_lo.npz` (data deep-clip table). The committed files come from Run-1 runs 31056, 31057,
  31060 and 31061.
- Test: `python -m unittest discover -s test -p "test_saturation.py"`.

### Visualization
- Visualization scripts are integrated into the data processors and inference scripts to generate plots for analysis.

## Configuration
All configurations are stored in the `yaml/` directory:
- `config.yaml`: For processing raw simulation data from Chroma.
- `configML.yaml`: For forward model training (position → light pattern).
- `configML_inverse.yaml`: For inverse model training (light pattern → position).
- `configPred.yaml`: For validating the forward model.
- `configInverseValidate.yaml`: For validating the inverse model.

