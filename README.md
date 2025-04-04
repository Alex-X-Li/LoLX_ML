# LoLX_ML Project

## Overview
LoLX_ML is a machine learning project designed for Light Only Liquid Xenon (LoLX) experiments.

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
├── test                             # Directory for test scripts
├── training_data                    # Directory for training data
├── yaml
│   ├── config.yaml                  # Configuration for data processing
│   ├── configML.yaml                # Configuration for model training
│   ├── configPred.yaml              # Configuration for model inference
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
python Beta_data_processor.py --config config.yaml
python ml_trainer.py --config configML.yaml
python ml_Prediction.py --config configPred.yaml
```

### Data Processing
- Use `ml/Beta_data_processor.py` or `ml/photonBomb_data_processor.py` to process raw data into a format suitable for training or inference.
- Configuration files for data processing are located in the `yaml/` directory.

### Model Training
- Train models using `ml/ml_trainer.py` or `ml/ml_trainer_kl.py`.
- Update `yaml/configML.yaml` with the appropriate paths and parameters for training.

### Model Inference
- Run predictions using `ml/ml_Prediction.py`.
- Ensure `yaml/configPred.yaml` is properly configured with the paths to the model and data.

### Visualization
- Visualization scripts are integrated into the data processors and inference scripts to generate plots for analysis.

## Configuration
All configurations are stored in the `yaml/` directory:
- `config.yaml`: For precessing raw simulation data from Chroma.
- `configML.yaml`: For model training.
- `configPred.yaml`: For validation the model.

