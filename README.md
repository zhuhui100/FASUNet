# FASUNet

This repository provides the code for paper: "FAS-UNet: A Novel FAS-driven Unet to Learn Variational Image Segmentation".

First, you can download the dataset at https://competitions.codalab.org/competitions/21145 (SegTHOR), or 
http://segchd.csail.mit.edu/index.html (HVSMR 2016), or 
https://chaos.grand-challenge.org/ (CHAOS CT)

Preprocess the dataset and save as ".npy" according to our paper. Split the preprocessed data into train data and validation data.

Place the ".npy" files in the "dataset_2D"(SegTHOR) or "01-chaos-overlap"(CHAOS CT) directory.


Run "main_train.py" to train and test.