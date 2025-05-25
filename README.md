```markdown
# A Comparative Study of Diffusion-Based vs Inception-Based Neural Speech Decoders

This repository contains code for comparing different neural architectures (DiffusionUNet, inceptDecoder, FCN, and CNN) to reconstruct speech from EEG signals. The study evaluates performance using spectro-temporal metrics like Pearson correlation and STGI (Spectro-Temporal Glimpsing Index).
```
## Project Structure

.
├── main.py              # Entry point for feature extraction, reconstruction, and visualization
├── README.md
└── src/
    ├── audio_constructor.py   # Handles audio reconstruction using trained models
    ├── config.py              # Configuration parameters (paths, hyperparameters)
    ├── feature_extractor.py   # Processes EEG/audio data and extracts Mel spectrograms
    ├── MelFilterBank.py       # Implements Mel filter bank for spectrogram conversion
    ├── models.py              # Defines neural architectures (DiffusionUNet, inceptDecoder, etc.)
    ├── reconstructWave.py     # Griffin-Lim algorithm for waveform reconstruction
    └── vis.py                 # Generates plots for results (spectrograms, correlations)
```

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/neural-speech-decoder-comparison.git
   cd neural-speech-decoder-comparison
   ```

2. **Install dependencies**:
   ```bash
   pip install tensorflow numpy scipy pandas matplotlib seaborn pynwb
   ```

3. **Dataset Setup**:
   - Download the `SingleWordProductionDutch-iBIDS` dataset and place it in the project root.

## Configuration
Modify `src/config.py` to adjust:
- `modelName`: Switch between `DiffusionUNet`, `inceptDecoder`, `CNN`, or `FCN`.
- `extract_features`, `construct`, `visualization`: Toggle pipeline stages.
- `epochs`, `batch_size`, `num_folds`: Training/evaluation parameters.

## Usage
1. **Feature Extraction** (if disabled in config):
   ```python
   # Set extract_features = True in config.py
   python main.py
   ```

2. **Train Models & Reconstruct Audio**:
   ```python
   # Set construct = True in config.py
   python main.py
   ```
   Reconstructed spectrograms are saved in `Reconstructed/`.

3. **Generate Visualizations**:
   ```python
   # Set visualization = True in config.py
   python main.py
   ```
   Output plots (spectrograms, correlations, STGI) are saved in `Images/`.

## Results
Key evaluation metrics:
- **Pearson Correlation**: Measures similarity between original and reconstructed spectrograms.
- **STGI**: Quantifies spectro-temporal reconstruction accuracy.
- **MSE Loss**: Training convergence metric.

Example outputs:
- `spectrogram_comparison.png`: Side-by-side spectrogram comparison across models.
- `correlations.png`: Per-subject Pearson correlation scores.
- `stgis.png`: Box plots of STGI values.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation
If you use this code, please cite:
```plaintext
[Your Publication Title]
Authors: [Your Name et al.]
Year: 2023
```

---

**Note**: Ensure the dataset directory `SingleWordProductionDutch-iBIDS` follows the BIDS format. Adjust paths in `config.py` if needed.
```
