# Enhancing Automated Seizure Detection via Self-Calibrating Spatial-Temporal EEG Features with SC-LSTM
## Model architecture
![image](https://github.com/Ivan020121/EpilepsyEEG/blob/main/Figures/SC-SLTM.svg)

## Setup and dependencies
Dependencies:
+ python 3.9
+ pytorch >= 2.0.1
+ numpy
+ pandas
+ h5py

Setup using conda:
```
conda env create -f environment.yml
```
##  Source codes:
+ SCNet.py: SC-LSTM model file
+ datasets.py: Generate training datatset
+ utils.py: General functions
+ train.py: train SC-LSTM model


## Run

````
python train.py --window [window size] --chunk [chunk size] --device 0
````
+ window (required): Window size should be [1, 2, 5, 10, 20].
+ chunk (required): Chunk size should be [1, 2, 5, 10, 20, 10000].
+ device (optional): Cuda device.
## Author
Wenhao Li

Guan Ning Lin
