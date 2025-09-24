from pathlib import Path
import random

import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
import config.config as config
sns.set(style="whitegrid")

pd.set_option('display.max_columns', 100)

bin_columns = [
    'chr10.136_NC', 'chr10.137_NC', 'chr10.138_NC', 'chr10.139_NC', 'chr10.140_NC', 
    'chr10.141_NC', 'chr10.142_NC', 'chr10.143_NC', 'chr10.144_NC', 'chr10.145_NC', 
    'chr10.146_NC', 'chr10.147_NC', 'chr10.148_NC', 'chr10.149_NC', 'chr10.150_NC', 
    'chr10.151_NC', 'chr10.152_NC', 'chr10.153_NC', 'chr10.154_NC', 'chr10.155_NC', 
    'chr10.156_NC', 'chr10.157_NC', 'chr10.158_NC', 'chr10.159_NC', 'chr10.160_NC', 
    'chr10.161_NC', 'chr10.162_NC', 'chr10.163_NC', 'chr10.164_NC', 'chr10.165_NC', 
    'chr10.166_NC', 'chr10.167_NC', 'chr10.168_NC', 'chr10.169_NC', 'chr10.170_NC', 
    'chr10.171_NC', 'chr10.172_NC', 'chr10.173_NC', 'chr10.174_NC'
]

def custom_log_plot_seaborn(data, title='PTEN', bin_columns=config.BIN_COLUMNS, xlabel='Bins', ylabel='Normalized Count'):
    """
    Plots data with a custom logarithmic y-axis scale using Seaborn and adds reference lines.

    Parameters:
    data (dict): Dictionary where keys are bin names and values are counts.
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    """
    df_bins = pd.DataFrame(list(data[bin_columns].items()), columns=[xlabel, ylabel])

    fig, ax = plt.subplots(figsize=(4, 8))
    sns.scatterplot(ax=ax, data=df_bins, x=xlabel, y=ylabel, marker='s', s=13, color=[69/255, 57/255, 71/255])

    ax.set_xticklabels([])
    ax.set_yscale('log')
    ax.set_ylim(0.01, 5)
    y_ticks = [0.01, 0.1, 0.5, 1.0, 1.5, 2.5, 5.0]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(y) for y in y_ticks])

    poly_degrees = [6]
    colors = [[254/255, 215/255, 137/255], [114/255, 135/255, 78/255], [71/255, 111/255, 132/255]]
    x_numeric = np.arange(len(df_bins))

    for degree, color in zip(poly_degrees, colors):
        coeffs = np.polyfit(x_numeric, df_bins[ylabel], degree)
        poly_eqn = np.poly1d(coeffs)
        y_fitted = poly_eqn(x_numeric)
        ax.plot(df_bins[xlabel], y_fitted, color=color)

    data = data[['AN number', 'PID', 'H-Score', 'FC', 'TCC']]
    metadata_entries = [f"$\\mathbf{{{key}}}$: {value:.2f}" if isinstance(value, float) else f"$\\mathbf{{{key}}}$: {value}"
                        for key, value in data.items() if key not in bin_columns]
    for entry in metadata_entries:
        ax.plot([], [], ' ', label=entry)

    ax.grid(True, which='both', linestyle='--', linewidth=0.1)
    lines_c = [164/255, 190/255, 213/255]
    ax.axhline(y=0.5, color=lines_c, linestyle='--', linewidth=0.5)
    ax.axhline(y=1.5, color=lines_c, linestyle='--', linewidth=0.5)

    pid = data['PID']
    ax.set_title(f"{title}: {data['PID']}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(handlelength=0, handletextpad=0, fancybox=True, ncol=2, loc='lower center')
    return fig

def bins_plot(data, title='PTEN', bin_columns=config.BIN_COLUMNS, xlabel='Bins', ylabel='Normalized Count'):
    """Plots data with a custom logarithmic y-axis scale using Seaborn and adds reference lines.
    """
    df_bins = pd.DataFrame(list(data[bin_columns].items()), columns=[xlabel, ylabel])
    plt.figure(figsize=(4, 8))
    plot = sns.scatterplot(data=df_bins, x=xlabel, y=ylabel, marker='s', s=13, color=[69/255, 57/255, 71/255])
    plot.set_xticklabels([])
    plt.grid(True, which='both', linestyle='--', linewidth=0.1)

    lines_c = [164/255, 190/255, 213/255]
    plt.axhline(y=0.5, color=lines_c, linestyle='--', linewidth=0.6)
    plt.axhline(y=1.5, color=lines_c, linestyle='--', linewidth=0.6)

    pid = data['PID']
    plt.title(f"{title}: {pid}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return plt

if __name__ == "__main__":

    dataset = pd.read_csv(config.DATASET_PATH)
    dataset = dataset.rename(columns={"FC_PTEN_Illumina": "FC", "TCC_GEPADO": "TCC",})

    sns.set_theme(style='whitegrid')
    for index, row in dataset.iterrows():
        fig = custom_log_plot_seaborn(row)
        fig.savefig(f'./output/19Sep/bin_plots/{row["AN number"]}_bins.jpg')
        plt.close(fig)




