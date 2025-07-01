# Deployment  
The **`EpiSight`** directory contains a ready-to-deploy server application.  

## Server Directory Structure  
+ **`model/`**: Stores the SC-LSTM model architecture files and trained weights.  
+ **`static/`**: Static web resources (CSS, JavaScript, images, fonts, etc.).  
+ **`templates/`**: HTML template files for server-side rendering.  
+ **`predict.py`**: Core prediction module — loads the model, processes uploaded EDF files, and generates predictions.  
+ **`server.py`**: Main service entry point — initializes the web server, defines API endpoints, and handles request routing.  
+ **`utils.py`**: Utility functions (data preprocessing, logging, auxiliary computations, etc.). 

## Server Dependency
+ python 3.9
+ pytorch >= 2.0.1
+ numpy
+ pandas
+ h5py
+ scipy
+ scikit-learn
+ fastapi\[standard\]
+ cachetools
+ matplotlib
+ scikit-learn
+ pyedflib

````
fastapi run server.py --no-reload --port [port]
````