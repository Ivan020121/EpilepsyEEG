import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
import warnings
from scipy.interpolate import make_interp_spline

'''
Plotting script for the simulated channel lost experiments (fig. 8),
using the results from the scc_results.xlsx and channel.xlsx in the IN_DIR directory, so modify IN_DIR if necessary.
'''
IN_DIR = '../../EpilepsyEEGResults/Results/'
OUT_DIR = './channelsPlot/'
warnings.filterwarnings("ignore", category=UserWarning, message=".*set_ticklabels.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Downcasting behavior in `replace`.*")

def getData(IN_DIR) -> tuple[DataFrame, DataFrame]:
    dfChannel = pd.read_excel(f'{IN_DIR}channel.xlsx', usecols="A:H")
    dfScc = pd.read_excel(f'{IN_DIR}scc_results.xlsx', usecols="A:I")
    dfScc.columns = [col.lower() for col in dfScc.columns]
    dfStandard = dfScc[(dfScc['window'] == 20) & (dfScc['chunk'] == 20)].reset_index()
    dfStandard.drop(columns=['window', 'chunk'], inplace=True)
    dfStandard['channel'] = -1
    dfChannel = dfChannel.replace('avg', 0).reset_index()
    return dfChannel, dfStandard


def draw2DNew(dfChannels, dfStandard, metric, ylim, expert) -> None:
    plt.rcParams['font.family'] = 'Times New Roman'
    fontSize = 32
    colorPalette = {'F4-C4': '#a7d7bb', 'T6-O2': '#abcdef', 'avg': '#feb039', 'noDamage': '#638394',
                    'default': '#374649', 'line': '#fa7f6f'}
    channels = ['Undamaged', 'Average', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F8',
                'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fz-Cz', 'Cz-Pz']
    metricMap = {'acc': 'ACC', 'auc': 'AUC', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1-Score'}
    dfChannelNew = dfChannels[dfChannels['expert'] == expert].drop(columns=['fold', 'index'])
    dfStandardNew = dfStandard[dfStandard['expert'] == expert].drop(columns=['fold', 'index'])
    dfFull = pd.concat([dfChannelNew, dfStandardNew], axis=0).drop(columns=['expert'])
    dfFullAvg = dfFull.groupby(['channel']).mean(metric).reset_index()
    fig, ax = plt.subplots(figsize=(15, 9))
    dfFull['color'] = 'default'
    dfFull.loc[dfFull['channel'] == 0, 'color'] = 'avg'
    dfFull.loc[dfFull['channel'] == -1, 'color'] = 'noDamage'
    dfFull.loc[dfFull['channel'] == 2, 'color'] = 'F4-C4'
    dfFull.loc[dfFull['channel'] == 12, 'color'] = 'T6-O2'
    sns.barplot(data=dfFull, x='channel', y=metric, color='b', alpha=1, capsize=0.4, dodge=True, ax=ax)
    sns.scatterplot(data=dfFull, x=dfFull['channel'] + 1, y=metric, hue='color', palette=colorPalette, s=60, alpha=0.8, ax=ax)
    for i, patch in enumerate(plt.gca().patches):
        patch.set_edgecolor(colorPalette['default'])
        patch.set_linewidth(4)
        patch.set_facecolor('none')
        if i == 0:
            patch.set_edgecolor(colorPalette['noDamage'])
        elif i == 1:
            patch.set_edgecolor(colorPalette['avg'])
        elif i == 3:
            patch.set_edgecolor(colorPalette['F4-C4'])
        elif i == 13:
            patch.set_edgecolor(colorPalette['T6-O2'])
    x = dfFullAvg['channel']
    y = dfFullAvg[metric]
    x_new = np.linspace(x.min(), x.max(), 3000)
    spl = make_interp_spline(x, y, k=3)
    y_new = spl(x_new)
    for line in ax.get_lines():
        line.set_alpha(0.85)
        line.set_linewidth(4)
    sns.lineplot(x=x_new + 1, y=y_new, alpha=0.85, color=colorPalette['line'], linewidth=4)
    plt.ylim(ylim, 1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().set_xticklabels(channels, rotation=60, fontsize=fontSize)
    ax.get_legend().remove()
    ax.tick_params(axis='both', labelsize=fontSize)
    plt.xlabel('Channel', fontsize=fontSize)
    plt.ylabel(metricMap[metric], fontsize=fontSize)
    ax.yaxis.set_label_coords(0.0, 1.03)
    ax.yaxis.label.set_rotation(0)
    plt.title(f'Window=20, Chunk=20, Expert={expert}', fontsize=fontSize, pad=20)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}{metric}_{expert}.svg')


if __name__ == '__main__':
    if IN_DIR == '../../EpilepsyEEGResults/Results/':
        print(f"Please ensure you set the correct directory for the input files, now using default {IN_DIR}")
    else:
        print(f"Reading data from {IN_DIR}")
    dfChannels, dfStandard = getData(IN_DIR)
    ylimMap = {'acc': 0.8, 'auc': 0.95, 'precision': 0.95, 'recall': 0.65, 'f1': 0.7}
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    for expert in ['A', 'B', 'C']:
        for metric in list(ylimMap.keys()):
            draw2DNew(dfChannels, dfStandard, metric, ylimMap[metric], expert)
    print(f"All plots saved to {OUT_DIR}")