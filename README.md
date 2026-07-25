# ASGAE
This repository contains the official implementation of the ASGAE model, introduced in our paper Unsupervised Adaptive Sampling Graph Autoencoder for 3D Surface Encoding and Mesh Representation Transfer. 

######
In terminal you need to type 
- docker build -t my-model .

After that
- docker run --gpus all -p 8000:8000 -v "D:\OneDrive - The University of Colorado Denver\DataMeshCHCO4_25:/data" my-model

it is needed to mount the folder with the data inside the docker image to have data access