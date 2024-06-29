
# MCSE Module 8

This repository contains Python scripts for detecting malware in IoT devices using various machine learning models as described in the paper 'Smart IoT Security: Utilizing AI for Effective Anomaly Detection'. Each script serves a unique purpose in the process of loading, transforming, and modeling data to predict potential security threats.

## Description

- `ann.py`: Implements an Artificial Neural Network (ANN) for selecting features.
- `lstm.py`: Uses Long Short-Term Memory (LSTM) networks to handle sequence data for anomaly detection.
- `svm.py`: Applies Support Vector Machine (SVM) for classifying IoT scenario data into categories.
- `transform_csv.py`: Prepares and cleans data files by transforming CSV formats and content.
- `helper_funcs.py`: Contains helper functions used across various scripts for data processing and model evaluation.
- `combine_data.py`: Preprocessing and creation of combined dataset.
- `optuna_ann.py`: Uses Optuna for hyperparameter tuning of the ANN model.
- `optuna_lstm.py`: Uses Optuna for hyperparameter tuning of the LSTM model.
- `optuna_svm.py`: Uses Optuna for hyperparameter tuning of the SVM model.
- `own_models.py`: Contains custom model implementations.
- `stacked.py`: Implements stacked ensemble learning models.
- `config.py`: Configuration file for setting up paths and other constants.
  
## Datasets

The datasets used for training can be found at [Zenodo](https://zenodo.org/records/4743746).

## Getting Started

### Prerequisites

Ensure you have `conda` installed and set up correctly.

```bash
conda activate base
conda install nvidia/label/cuda-12.4.1::cuda-toolkit
conda create -n pytorch --python=3.11 pytorch torchvision torchaudio pytorch-cuda=12.1 scikit-learn pandas scipy -c anaconda -c pytorch -c nvidia
```

### Dependencies

- `Python 3.11`
- `PyTorch`
- `Pandas`
- `Numpy`
- `Scikit-Learn`
- `Optuna`
- `SciPy`

### Installation

Install the dependencies using pip:

```bash
pip install torch pandas numpy scikit-learn optuna scipy
```

### Usage

To run each script, navigate to the directory and use:

```bash
python ann.py
python lstm.py
python svm.py
python transform_csv.py
python combine_data.py
python optuna_ann.py
python optuna_lstm.py
python optuna_svm.py
python own_models.py
python stacked.py
```

### Common Issues

Ensure CUDA is available for PyTorch:

```python
import torch
print(torch.cuda.is_available())
```

For further assistance, refer to the respective script files and ensure all dependencies are correctly installed.

### Additional Notes

- Ensure all paths and configurations are correctly set in `config.py`.
- Check and update helper functions in `helper_funcs.py` as needed for your specific use case.

For more detailed instructions and troubleshooting, please refer to the comments and documentation within each script.

