# Brain-Tumor-Segmentation-using-BraTS-MRI-
Developed an end-to-end brain tumor segmentation pipeline using the BraTS MRI dataset. Built preprocessing, training, evaluation, and ONNX-based inference pipelines for U-Net segmentation of WT, TC, and ET regions. Gradio and AWS deployment are in progress.


# Brain-Tumor-Segmentation-using-BraTS-MRI-
Developed an end-to-end brain tumor segmentation pipeline using the BraTS MRI dataset. Built preprocessing, training, evaluation, and ONNX-based inference pipelines for U-Net segmentation of WT, TC, and ET regions. Gradio and AWS deployment are in progress.


## Overview

Brain tumor segmentation is an important task in medical image analysis that helps clinicians identify tumor regions from MRI scans for diagnosis and treatment planning.

This project implements an end-to-end deep learning pipeline for automatic brain tumor segmentation using the BraTS MRI dataset. The pipeline covers MRI preprocessing, data augmentation, model training, evaluation, ONNX optimization, and deployment.

The model segments three clinically important regions:

- Whole Tumor (WT)
- Tumor Core (TC)
- Enhancing Tumor (ET)

Deployment using Gradio and AWS is currently in progress.


## Features

- End-to-end segmentation pipeline
- Multi-modal MRI preprocessing
- Slice-wise training pipeline
- Data augmentation using Albumentations
- U-Net semantic segmentation model
- Dice Score and HD95 evaluation
- ONNX Runtime inference
- Modular inference pipeline
- FastAPI Inference API (In Progress)
- Interactive Gradio application (In Progress)
- AWS Deployment (In Progress)


## Dataset

Dataset: BraTS (Brain Tumor Segmentation Challenge)

Files: *.nii.gz

Input modalities:

- T1
- T1ce
- T2
- FLAIR

Output classes:

- Whole Tumor (WT)
- Tumor Core (TC)
- Enhancing Tumor (ET)


## Project Structure
project/

├── dataset/
│   └── Data_Analysis.ipynb
│   └── Data_Preparation.ipynb
│   └── split.json
├── code/
│   └── train.py
│   └── Config.yaml
├── inference/
│   └── api/
├── evaluate/
├── output/
└── README.md


## MRI Preprocessing

- Intensity normalization
- Dynamic padding
- Slice extraction
- Channel-wise processing
- Mask preparation
- Data augmentation

## Model

Architecture:

- U-Net

Loss Function:

- BCE + Dice Loss

Framework:

- PyTorch Lightning


## Evaluation Metrics

- Dice Score
- HD95

Results:
| Region | Dice  |
| ------ | ----- |
| WT     | 0.918 |
| TC     | 0.876 |
| ET     | 0.821 |


## ONNX Runtime

The trained PyTorch model was exported to ONNX for deployment-ready inference.

Implemented:

- ONNX export
- ONNX Runtime inference
- GPU inference support
- Modular preprocessing and post-processing pipeline

## Deployment

Deployment is currently under development.

Planned features:

- Upload MRI (.npz)
- Automatic segmentation
- Overlay visualization
- Gradio Web UI
- AWS Deployment


## Tech Stack

- Python
- PyTorch
- PyTorch Lightning
- NumPy
- Albumentations
- ONNX Runtime
- MedPy
- Gradio
- AWS (In Progress)


## Future Improvements

- 3D U-Net implementation
- MONAI integration
- TensorRT optimization
- Docker deployment
- SageMaker endpoint
- Intgration of MRI (.nii.gz) / DICOM Medical Imaging File
- Batch inference
