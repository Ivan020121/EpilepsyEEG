# Enhancing Automated Seizure Detection via Self-Calibrating Spatial-Temporal EEG Features with SC-LSTM
## Model architecture
![image](https://github.com/NENUBioCompute/TopoPharmDTI/blob/main/Figure/Model%20architecture.png)

## Setup and dependencies
dependencies:
+ python 3.9
+ pytorch >= 2.0.1
+ numpy
+ pandas
+ h5py


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