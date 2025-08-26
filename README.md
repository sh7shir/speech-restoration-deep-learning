# Comparative Study of Diffusion-Based vs Inception-Based Neural Speech Decoders

This repository contains the implementation and experimental results of **DiffusionUNet** (a diffusion-based neural speech decoder) compared against the **NeuroIncept Decoder** (an inception-based baseline).  
The project focuses on reconstructing **log-Mel spectrograms** from **stereotactic EEG (sEEG)** signals, contributing to research in **speech neuroprosthetics** and **brain-computer interfaces (BCIs)**.

---

## 📖 Abstract
We present a comparative study between two neural architectures for EEG-to-speech decoding:
- **NeuroIncept Decoder** → CNN + GRU hybrid using Inception modules.  
- **DiffusionUNet** → A novel U-Net–like denoising diffusion probabilistic model (DDPM).  

Results show that **DiffusionUNet** outperforms the baseline in terms of **Mean Squared Error (MSE)** and **Pearson Correlation Coefficient (PCC)**, while maintaining competitive performance on the **Spectro-Temporal Glimpsing Index (STGI)**.

---

## 🧩 Methodology
### Dataset
We use the **SingleWordProductionDutch-iBIDS** dataset [Verwoert et al., 2022], consisting of paired sEEG and speech recordings from 10 Dutch-speaking participants.  
- EEG sampled at **1024/2048 Hz** → downsampled to **1024 Hz**.  
- Speech audio at **48 kHz** → downsampled to **16 kHz**.  
- Ground truth represented as **log-Mel spectrograms** (128 bins).  

### Preprocessing
- **EEG:** Bandpass filtering (70–170 Hz), notch filtering (50/100/150 Hz), Hilbert transform, windowing (50 ms, 10 ms shift), z-score normalization.  
- **Audio:** Log-Mel spectrogram extraction (LibROSA).  

### Models
- **NeuroIncept Decoder:** Inception-style CNNs + GRU layers.  
- **DiffusionUNet:** DDPM-based conditional generative model with dilated 1D Conv U-Net.  

### Training
- Optimizer: **Adam (lr=1e-4)**  
- Batch size: **128**  
- Early stopping with patience = 5  
- Subject-specific training with **80/20 train-test split**  

### Evaluation Metrics
- **Mean Squared Error (MSE)** – reconstruction error  
- **Pearson Correlation Coefficient (PCC)** – signal fidelity  
- **Spectro-Temporal Glimpsing Index (STGI)** – perceptual intelligibility  

---

## 📊 Results
| Model            | MSE ↓       | PCC ↑       | STGI ↑     |
|------------------|-------------|-------------|------------|
| **DiffusionUNet** | **0.458 ± 0.085** | **0.928 ± 0.014** | 0.512 ± 0.014 |
| NeuroIncept      | 0.618 ± 0.141 | 0.901 ± 0.023 | **0.521 ± 0.021** |

- DiffusionUNet achieves **better accuracy and stronger correlation** with ground truth.  
- NeuroIncept has a slight advantage in STGI.  
- Visual comparisons confirm **DiffusionUNet produces spectrograms closer to the original**.  

---

## 🚀 Getting Started
### Prerequisites
- Python 3.9+  
- Install dependencies:
```bash
pip install -r requirements.txt
