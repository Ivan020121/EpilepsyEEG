import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

'''
Plotting script for visualization of the ablation study (fig. 7),
using the results from the *.npy files in the IN_DIR directory, so modify IN_DIR if necessary.
'''
IN_DIR = './embeddingData/'
OUT_DIR = './embeddingPlot/'

def drawEmbedding(model, expert):
    modelMap = {'nolstm': 'BiLSTM', 'noself': 'Self-Calibrated', 'nor': 'Reconstruction'}
    fontSize = 22
    plt.rcParams['font.family'] = 'Times New Roman'
    embedding = np.load(f'{IN_DIR}{model}_{expert}.npy')
    labels = np.load(f'{IN_DIR}{model}_{expert}_y.npy')
    pca = PCA(n_components=2)
    embedding_2d = pca.fit_transform(embedding)
    plt.figure(figsize=(10, 8))
    colors = {0: '#feb039', 1: '#638394'}
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=[colors[label.item()] for label in labels], alpha=0.3)
    handles = [plt.Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor='#feb039', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Epilepsy', markerfacecolor='#638394', markersize=10)]
    plt.legend(handles=handles,title = f'Without {modelMap[model]} (Expert={expert})', title_fontsize = fontSize, fontsize=fontSize, ncol = 2)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('Principal Component 1', fontsize=fontSize)
    plt.ylabel('Principal Component 2', fontsize=fontSize)
    plt.xticks(fontsize=fontSize)
    plt.yticks(fontsize=fontSize)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}{model}_{expert}.svg')

if __name__ == "__main__":
    if IN_DIR == './embeddingData/':
        print(f"Please ensure you set the correct directory for the input files, now using default {IN_DIR}")
    else:
        print(f"Reading data from {IN_DIR}")
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    for model in ['nolstm', 'noself', 'nor']:
        for expert in ['A', 'B', 'C']:
            drawEmbedding(model, expert)
    print(f"All plots saved to {OUT_DIR}")

