import os
from pathlib import Path

num_jobs = 20

extract_features = False
construct = True
visualization = False


current_dir = os.getcwd()
dataset_dir = Path(current_dir, "SingleWordProductionDutch-iBIDS")
features_dir = Path(current_dir, "features")
results_dir = Path(current_dir, "results")
no_of_mel_spectrograms = 128
epochs = 200
batch_size = 128
num_folds = 10

# model names: inceptDecoder, DiffusionUNet
modelNameArray =['inceptDecoder', 'DiffusionUNet', 'CNN', 'FCN']
modelName = outputFolderName = modelNameArray[1]