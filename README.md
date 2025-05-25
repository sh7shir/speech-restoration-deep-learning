# A Comparative Study of Diffusion-Based vs Inception-Based Neural Speech Decoders

![Example Spectrograms](Images/DiffusionUNetspectrogram_comparison.png)  
*Example reconstructed spectrograms from different models.*

## Description
This project compares neural architectures for reconstructing speech from intracranial EEG (iEEG) data. It implements and evaluates four models:  
- **DiffusionUNet**: A U-Net with dilated convolutions and diffusion-based denoising.  
- **inceptDecoder**: Inception modules combined with GRU layers.  
- **FCN**: Fully Connected Network with batch normalization.  
- **CNN**: Convolutional Neural Network.  

The pipeline includes feature extraction from EEG/audio, model training, spectrogram reconstruction, and visualization of results (correlation, STGI scores, and spectrograms).

## Installation
### Dependencies
- Python 3.8+
- TensorFlow 2.10+
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn
- PyNWB (for loading EEG data)

Install packages:  
```bash
pip install -r requirements.txt
