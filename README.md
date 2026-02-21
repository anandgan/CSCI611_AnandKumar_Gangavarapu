#CSCI611 – Assignment 2

This project implements a custom Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset. The model is built from scratch using PyTorch and consists of four convolutional layers followed by Batch Normalization, ReLU activation functions, and MaxPooling to progressively extract and refine spatial features. Dropout regularization is applied to reduce overfitting, and the extracted features are passed into a fully connected classifier for final prediction across the ten CIFAR-10 classes. The network is trained end-to-end, learning hierarchical image representations directly from the dataset without using any pretrained weights.

Requirements:

Python 3.10+

Install dependencies:by 
pip install torch torchvision matplotlib numpy
##How to Run :
From the Assignment_2 directory- python cnn.py
then the script will, The script will:

-Download CIFAR-10
-Train the CNN
-Save the best model
-Generate required plots and visualizations

Outputs
-All generated outputs are saved in the outputs/ folder:

-Training & validation loss curves

-Training & validation accuracy curves

-Feature maps from first convolutional layer (Task 2A)

-Top-5 maximally activating images per filter (Task 2B)

Files Included

-cnn.py – Source code

-CNN REPORT.pdf – Assignment report

-outputs/ – Generated figures
