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
+ scipy
+ scikit-learn

Setup using conda:
```
conda env create -f environment.yml
```

## Dataset
Raw EDF files and CSV annotations files are available at https://drive.google.com/drive/folders/1y2A7-mKEG7qpw1Hh-RuDzKcJH71f2tN9?usp=drive_link. 

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

## Deployment
### Server Dependency
+ fastapi\[standard\]
+ cachetools
+ matplotlib
+ scikit-learn
+ pyedflib

````
fastapi run server.py --no-reload --port [port]
````


## Author
Wenhao Li

Liujinxiang Zhu

Guan Ning Lin
