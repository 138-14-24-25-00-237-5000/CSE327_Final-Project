# CSE327: Respiratory Sound Classification with CNN and ViT

This repository contains two deep learning projects for respiratory sound classification using audio spectrograms. The tasks are implemented in PyTorch and involve both Convolutional Neural Networks (CNNs) and Vision Transformers (ViT).

## Files

### 📄 `CSE327_CNNs.ipynb`
- Loads respiratory sound data and converts it into log-mel spectrograms.
- Trains and compares different CNN-based architectures.
- Plots training/validation accuracy and loss.
- Evaluates performance using accuracy, F1-score, and ROC-AUC.

### 📄 `CSE327_Data_Preprocessing_&_ViT.ipynb`
- Applies data preprocessing including **SpecAugment** (frequency/time masking).
- Fine-tunes a **ViT (Vision Transformer)** model on spectrograms.
- Implements training loops with class imbalance handling and multiple augmentation strategies.
- Tracks performance metrics across 30–50 epochs.

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- timm
- torchaudio
- librosa
- scikit-learn
- matplotlib

Install via:
```bash
pip install torch torchvision timm torchaudio librosa scikit-learn matplotlib
```

## Author
Sanghoon Lee  
Chaeeun Kyung

Stony Brook University
