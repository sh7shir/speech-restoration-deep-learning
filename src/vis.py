import os
from pathlib import Path
import src.config as config
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import src.config as config
import pandas as pd
import pdb
fontsize = 16


def plot_history():
    print('*****************Plotting History******************')

    folder = config.modelName
    destination = Path(config.current_dir, 'Images')
    os.makedirs(destination, exist_ok=True)
    
    files = os.listdir(folder)
    files = [Path(folder, item) for item in files if item.endswith('.csv') and item.startswith('sub')]
    names = []
    for item in files:
        name = str(item).split('/')[-1].split('.')[0]
        names.append(name)
        data = pd.read_csv(item)
        plt.plot(data['val_loss'])
    
    plt.legend(names)
    plt.xlabel('Epochs', fontsize=fontsize+2, fontweight='bold')
    plt.ylabel('Mean Squared Error', fontsize=fontsize+2, fontweight='bold')
    plt.xticks(fontsize=fontsize, fontweight='bold')
    plt.yticks(fontsize=fontsize, fontweight='bold')

    plt.tight_layout()
   
    plt.savefig(Path(destination, config.modelName + 'history.png'), dpi=600)
    plt.clf()
    
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import src.config as config

def plot_spectrograms():
    start, end = 1100, 1600
    # load all five
    original        = np.load('NeuroIncept/original.npy')
    predicted_neuro = np.load('NeuroIncept/predicted.npy')
    predicted_diff  = np.load('DiffusionUNet/predicted.npy')   # assuming you saved it here
    predicted_CNN   = np.load('CNN/predicted.npy')
    predicted_FCN   = np.load('FCN/predicted.npy')

    # collapse the 9-frame blocks into single time points
    num_samples_per_time = 9
    num_time_points = original.shape[1] // num_samples_per_time

    def collapse(x):
        x = x[:, :num_time_points * num_samples_per_time]
        x = x.reshape(x.shape[0], num_time_points, num_samples_per_time)
        return x.mean(axis=2)

    orig_m = collapse(original)
    neuro_m = collapse(predicted_neuro)
    diff_m = collapse(predicted_diff)
    cnn_m = collapse(predicted_CNN)
    fcn_m = collapse(predicted_FCN)

    # set up 5×1 grid
    fig, axes = plt.subplots(5, 1, figsize=(6, 12), sharex=True, sharey=True)

    titles = [
        'Original Log Mel Spectrogram',
        'NeuroIncept Decoder',
        'DiffusionUNet',
        'CNN',
        'FCN'
    ]
    data = [orig_m, neuro_m, diff_m, cnn_m, fcn_m]
    labels = ['a)', 'b)', 'c)', 'd)', 'e)']

    for ax, spec, title, label in zip(axes, data, titles, labels):
        im = ax.imshow(spec[start:end].T, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=6)
        ax.tick_params(labelleft=False, labelbottom=False)
        ax.text(0.02, 0.95, label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    destination = Path(config.current_dir, 'Images')
    destination.mkdir(parents=True, exist_ok=True)
    plt.savefig(destination / config.modelName + 'spectrogram_comparison.png', dpi=600)
    plt.clf()


def plot_correlation():
    print('*****************Plotting Correlations******************')

    folder = Path(config.current_dir, config.modelName)
    destination = Path(config.current_dir, 'Images')
    os.makedirs(destination, exist_ok=True)
    
    files = os.listdir(folder)
    files = [Path(folder, item) for item in files if item.endswith('.npy') and 'corr' in item]
    num_folds = config.num_folds
    num_subjects = 10
    data = []
    names = []
    for file in files:
        temp = np.load(file)
        data.append(temp)
        names.append(str(file).split('/')[-1].split('.')[0].split('-')[1].split('_')[0])
    data = np.array(data)

    
    mean_correlations = np.mean(data, axis=2)  # Shape: (10, 10)
    std_errors = np.std(data, axis=2)  # Shape: (10, 10)

    mean_per_subject = np.mean(mean_correlations, axis=1)  # Shape: (10,)
    std_per_subject = np.std(mean_correlations, axis=1)  # Shape: (10,)
    sorted_indices = np.argsort(mean_per_subject)
    sorted_mean_per_subject = mean_per_subject[sorted_indices]
    sorted_std_per_subject = std_per_subject[sorted_indices]
    sorted_mean_correlations = mean_correlations[sorted_indices]
    sorted_dot_colors = sns.color_palette("coolwarm", n_colors=num_subjects)
    print(sorted_mean_per_subject)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    bar_colors = sns.color_palette("viridis", n_colors=num_subjects)

    bars = ax.bar(range(num_subjects), sorted_mean_per_subject, yerr=sorted_std_per_subject, capsize=8, color=[bar_colors[i] for i in range(num_subjects)], edgecolor='black', alpha=0.8, label='Mean Correlation')

    for i in range(num_subjects):
        ax.plot(np.full(num_folds, i), sorted_mean_correlations[i], '.', color=sorted_dot_colors[i], alpha=0.7, label=f'Folds (Subject {i + 1})', markersize=8, linestyle='None')

    ax.set_ylabel('Correlation', fontsize=16, fontweight='bold')
    ax.set_xticks(range(num_subjects))
    ax.set_xticklabels([f'sub-{names[i]}' for i in sorted_indices], fontsize=14, fontweight='bold')
    #ax.set_yticklabels(ax.get_yticks(), fontsize=14, fontweight='bold')

    
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.ylim([0.8,0.95])
    plt.xticks(fontsize=fontsize, fontweight='bold')
    plt.yticks(fontsize=fontsize, fontweight='bold')
    fig.tight_layout()

    # Show the plot
    plt.savefig(Path(destination, 'correlations.png'), dpi=600)
    plt.clf()

def plot_stgis():
    print('*****************Plotting STGIS******************')
    folder = Path(config.current_dir, config.modelName)
    destination = Path(config.current_dir, 'Images')
    os.makedirs(destination, exist_ok=True)
    
    files = os.listdir(folder)
    files = [Path(folder, item) for item in files if item.endswith('.npy') and 'stgi' in item]
    num_folds = config.num_folds
    num_subjects = 10
    data = []
    names = []
    for file in files:
        temp = np.load(file)
        data.append(temp)
        names.append(str(file).split('/')[-1].split('.')[0].split('-')[1].split('_')[0])
    data = np.array(data)
    
    mean_correlations = np.mean(data, axis=2)  # Shape: (10, 10)
    std_errors = np.std(data, axis=2)  # Shape: (10, 10)

    mean_per_subject = np.mean(mean_correlations, axis=1)  # Shape: (10,)
    std_per_subject = np.sqrt(np.mean(std_errors**2, axis=1) / num_folds)  # Shape: (10,)

    sorted_indices = np.argsort(mean_per_subject)
    sorted_mean_correlations = mean_correlations[sorted_indices]
    print(np.mean(sorted_mean_correlations, axis=1))
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    meanprops = dict(marker='s', markerfacecolor='green', markeredgecolor='red', markersize=10)
    box_data = [sorted_mean_correlations[i] for i in range(num_subjects)]
    
    box = ax.boxplot(box_data, patch_artist=False, notch=False, showmeans=True, 
                     meanprops=meanprops, showfliers=False)

    
    
    for i in range(num_subjects):
        y = sorted_mean_correlations[i]
        x = np.random.normal(i + 1, 0.04, size=len(y))  # Add some jitter to the x-axis
        ax.scatter(x, y, color='black', alpha=0.7, s=20, edgecolor='black', zorder=3)

    ax.set_ylabel('STGI', fontsize=16, fontweight='bold')
   
    labels = [f'sub-{names[i]}' for i in sorted_indices]
    print(labels)
    ax.set_xticklabels(labels, fontsize=fontsize, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.ylim([0.45, 0.57])
    fig.tight_layout()
    plt.xticks(fontsize=fontsize, fontweight='bold')
    plt.yticks(fontsize=fontsize, fontweight='bold')
    plt.subplots_adjust(left=0.08)
    # Save the plot
    plt.savefig(Path(destination, 'stgis.png'), dpi=600)
    plt.clf()
