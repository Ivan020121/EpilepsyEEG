import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

IN_DIR = '../../EpilepsyEEGResults/Results/'
OUT_DIR = './violinPlot/'

def read_grouped_data(model_name):
    file_path = f'{IN_DIR}{model_name}_results.xlsx'
    df = pd.read_excel(file_path, usecols="A:I")
    df = df[df['Chunk'] != 10000]
    df_group = df.groupby(['Expert','Window','Chunk']).agg({'AUC': 'std'})
    df_group = df_group.reset_index()
    return df_group

if __name__ == "__main__":
    dataMap = {}
    dataStdMap = {}
    for model in ['cnn', 'cnnlstm', 'resnet50', 'scc']:
        dataMap[model] = read_grouped_data(model)
        dataStdMap[model] = dataMap[model]['AUC'].values
    n = len(dataStdMap['cnn'])
    print(dataStdMap)
    std_data = pd.DataFrame({
        'AUC STD': np.concatenate([dataStdMap[model] for model in ['cnn', 'cnnlstm', 'resnet50', 'scc']]),
        'Models': ['CNN']*n + ['CNN LSTM']*n + ['ResNet50']*n + ['SCC']*n
    })
    colorMap = {'CNN': '#528FAD', 'CNN LSTM': '#FFD06F', 'ResNet50': '#EF8A47', 'SCC': '#E76254'}
    sns.violinplot(x='Models', y='AUC STD', data=std_data, palette=colorMap, cut=0)
    sampleFraction = 0.3
    subset = std_data.groupby('Models').apply(lambda x: x.sample(frac=sampleFraction)).reset_index(drop=True)
    sns.stripplot(x='Models', y='AUC STD', data=subset, color='white', size=2, jitter=True, alpha=0.8)
    plt.ylim((-0.01, 0.22))
    legendElements = [plt.Line2D([0], [0], color=colorMap[key], label=key, marker='o', linestyle='') for key in colorMap]
    plt.legend(handles=legendElements, loc='upper right')
    plt.title('Violin Plot of AUC STD for Four Models')
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    plt.savefig(f'{OUT_DIR}violinPlot.svg')