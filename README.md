# A Comparative Study of Diffusion-Based vs Inception-Based Neural Speech Decoders

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


project_root/
├── SingleWordProductionDutch-iBIDS/
│   ├── sub-01/  
│   │   └── ieeg/  
│   │       └── sub-01_task-wordProduction_ieeg.nwb  
│   ├── ...  
│   └── participants.tsv  
├── src/  
├── ...

Usage
1. Configuration
Modify config.py to:

Set modelName to one of: DiffusionUNet, inceptDecoder, CNN, FCN.

Adjust hyperparameters (epochs, batch size, etc.).

2. Feature Extraction
Run to extract Mel spectrograms and EEG features:

bash
python main.py
Note: Set extract_features = True in config.py for this step.

3. Model Training & Reconstruction
Set construct = True in config.py, then run:

bash
python main.py
Reconstructed spectrograms and evaluation metrics will be saved in results/.

4. Visualization
Set visualization = True in config.py, then run:

bash
python main.py
Plots for correlation, STGI, and spectrograms will be saved in Images/.
