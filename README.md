# Brain Tumor Segmentation using BraTS MRI

Developed an end-to-end brain tumor segmentation pipeline using the BraTS MRI dataset. Built preprocessing, training, evaluation, and ONNX-based inference pipelines for U-Net segmentation of Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET). Gradio and AWS deployment are currently in progress.

---

## Overview

Brain tumor segmentation is an important task in medical image analysis that helps clinicians identify tumor regions from MRI scans for diagnosis and treatment planning.

This project implements an end-to-end deep learning pipeline for automatic brain tumor segmentation using the BraTS MRI dataset. The pipeline covers MRI preprocessing, data augmentation, model training, evaluation, ONNX optimization, and deployment.

The model segments three clinically important regions:

- Whole Tumor (WT)
- Tumor Core (TC)
- Enhancing Tumor (ET)

Deployment using Gradio and AWS is currently in progress.

---

## Features

- End-to-end brain tumor segmentation pipeline
- Multi-modal MRI preprocessing
- Slice-wise training pipeline
- Data augmentation using Albumentations
- U-Net semantic segmentation model
- Dice Score and HD95 evaluation
- ONNX Runtime inference
- Modular preprocessing and post-processing pipeline
- FastAPI inference API *(In Progress)*
- Interactive Gradio application *(In Progress)*
- AWS deployment *(In Progress)*

---

## Dataset

**Dataset:** BraTS (Brain Tumor Segmentation Challenge)

**Input Format:** `.nii.gz`

### MRI Modalities

- T1
- T1ce
- T2
- FLAIR

### Output Classes

- Whole Tumor (WT)
- Tumor Core (TC)
- Enhancing Tumor (ET)

---

## Project Structure

```text
project/
│
├── dataset/
│   ├── Data_Analysis.ipynb
│   ├── Data_Preparation.ipynb
│   └── split.json
│
├── code/
│   ├── train.py
│   └── config.yaml
│
├── inference/
│   └── api/
│
├── evaluate/
│
├── output/
│
└── README.md
```

---

## MRI Preprocessing

- Intensity normalization
- Dynamic padding
- Slice extraction
- Channel-wise processing
- Mask preparation
- Data augmentation

---

## Model

**Architecture**

- U-Net

**Loss Function**

- BCE + Dice Loss

**Framework**

- PyTorch Lightning

---

## Evaluation Metrics

- Dice Score
- HD95

### Results

| Region | Dice Score |
|---------|-----------:|
| WT | **0.918** |
| TC | **0.876** |
| ET | **0.821** |

---

## ONNX Runtime Optimization

The trained PyTorch model was exported to ONNX for deployment-ready inference.

### Implemented

- ONNX model export
- ONNX Runtime inference
- GPU inference support
- Modular preprocessing pipeline
- Modular post-processing pipeline

---

## Deployment

Deployment is currently under development.

### Planned Features

- Upload MRI (`.npz`)
- Automatic tumor segmentation
- Segmentation overlay visualization
- Interactive Gradio web application
- AWS deployment

---

## Tech Stack

- Python
- PyTorch
- PyTorch Lightning
- NumPy
- Albumentations
- ONNX Runtime
- MedPy
- Gradio *(In Progress)*
- AWS *(In Progress)*

---

## Future Improvements

- 3D U-Net implementation
- MONAI integration
- TensorRT optimization
- Docker deployment
- AWS SageMaker endpoint
- Support for MRI `.nii.gz` and DICOM files
- Batch inference

---

## Acknowledgements

- BraTS (Brain Tumor Segmentation Challenge)
- PyTorch
- PyTorch Lightning
- ONNX Runtime
- Albumentations
- MedPy
