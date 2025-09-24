import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

def plot_samples(X, y, bin_columns, indices=None, figsize=(15, 10)):
    """
    Plot selected rows from X based on provided indices with color based on y values.
    
    Parameters:
        X (pd.DataFrame): DataFrame containing the feature data.
        y (pd.Series): Series containing the true values.
        bin_columns (list): List of column names in X to be used for the plot.
        indices (list, optional): List of indices to plot. If None, all rows are plotted.
        figsize (tuple): Size of the figure.
    """
    target_name = y.name
    if indices is None:
        indices = X.index.tolist()

    x_data = X.loc[indices, bin_columns]
    y_true = y.loc[indices]
    x_data[target_name] = y_true

    fig, ax = plt.subplots(figsize=figsize)
    data_melted = x_data.reset_index().melt(id_vars=['index', target_name], var_name='order', value_name='value')
    sns.lineplot(data=data_melted, x='order', y='value', hue=target_name, palette='viridis', legend='brief', linewidth=0.5, alpha=0.3, ax=ax)
    plt.xticks(rotation=90)
    plt.xlabel('Bins')
    plt.ylabel('Values')
    plt.tight_layout()
    ax.legend(title=target_name, bbox_to_anchor=(1.05, 1), loc='upper left')
    return fig, ax

def custom_log_plot(data, title='PTEN', bin_columns=bin_columns, xlabel='Bins', ylabel='Normalized Count'):
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

    # [X]
    ax.set_xticklabels([])

    # [Y] Set custom logarithmic scale for y-axis
    ax.set_yscale('log')
    ax.set_ylim(0.01, 5)
    y_ticks = [0.01, 0.1, 0.5, 1.0, 1.5, 2.5, 5.0]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(y) for y in y_ticks])

    ax.grid(True, which='both', linestyle='--', linewidth=0.1)

    # --- Horizontal lines
    lines_c = [164/255, 190/255, 213/255]
    ax.axhline(y=0.5, color=lines_c, linestyle='--', linewidth=0.9)
    ax.axhline(y=1.5, color=lines_c, linestyle='--', linewidth=0.9)

    pid = data['PID']
    ax.set_title(f"{title}: {data['Pseudonym']}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def plot_predictions(pred_df):
    """Generates a swarmplot
    """
    pred_df['H-Score'] = pd.to_numeric(pred_df['H-Score'], errors='coerce')
    
    conditions = [
        pred_df['H-Score'].isna(),  # If the H-Score is missing, mark as "Not stained"
        pred_df['H-Score'] < 25,      # For scores less than 25, assign "Del"
        pred_df['H-Score'] > 90       # For scores greater than 90, assign "WT"
    ]
    choices = ["Not stained", "Del", "WT"]
    
    pred_df['H_color'] = np.select(conditions, choices, default="Grey")
    pred_df['Category'] = 'Predictions'
    
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(7, 10))
    
    palette = {
        "WT": "#1f77b4",          # blue
        "Del": "#d62728",         # red
        "Grey": "#7f7f7f",        # gray
        "Not stained": "#ffffff"  # white
    }
    
    ax = sns.swarmplot(
        x='Category',
        y='Bioinf pred',
        data=pred_df,
        hue='H_color',
        palette=palette,
        size=10,
        edgecolor='black',
        linewidth=1,
        alpha=0.7,
    )
    
    ax.axhline(y=0.68, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(0.092, 0.66, '0.68', color='red', fontsize=11, ha='left')
    
    ax.violinplot(
        [pred_df['Bioinf pred']],
        positions=[0],
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.2
    )
    
    ax.set_title("Cohort B: Model 1 Predictions", fontsize=20)
    ax.set_xlabel("")
    ax.set_ylabel("Predicted WT prob", fontsize=16)
    
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for lab in labels:
        if lab == "WT":
            new_labels.append("WT (>90)")
        elif lab == "Del":
            new_labels.append("Del (<25)")
        elif lab == "Grey":
            new_labels.append("Grey area (25-90)")
        else:
            new_labels.append(lab)
    ax.legend(handles=handles, labels=new_labels, fontsize=14)
    ax.get_legend().set_title('')
    
    ax.grid(True, which='both', linestyle='--', linewidth=1.3)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()


def plot_scatter_with_thresholds(pred_df, palette=None, annotate=False):
    """Scatter plot of 'H-Score' vs 'Bioinf pred'.
    
    The function draws a horizontal threshold line at y = 0.68 and vertical threshold lines at x = 25 and x = 90.
    It also optionally annotates data points based on conditions:
      - Annotates if H-Score < 100 and Bioinf pred > 0.68, or if H-Score >= 90 and Bioinf pred < 0.68.
    
    Parameters:
        pred_df (pd.DataFrame): DataFrame with at least 'H-Score' and 'Bioinf pred' columns.
                                  Optionally, include a 'Pseudonym' column for annotations.
        palette (dict, optional): A dictionary defining colors for the scatter plot.
                                  Default palette is provided if None.
        annotate (bool, optional): If True, annotates points based on predefined conditions.
    
    Returns:
        None: The plot is displayed using plt.show().
    """
    pred_df = pred_df.copy()
    if palette is None:
        palette = {
            "WT": "#1f77b4",       # blue
            "Del": "#d62728",      # red
            "Grey": "#7f7f7f",     # gray
            "Not stained": "#ffffff"  # white
        }
    
    plt.figure(figsize=(6, 8))
    ax = plt.gca()
    
    ax.axhline(y=0.68, color='red', linestyle='--', linewidth=1.7, alpha=0.7)
    ax.text(275, 0.69, '0.68', color='red', fontsize=11, ha='left', zorder=1, alpha=0.7)
    
    ax.axvline(x=25, color='red', linestyle='--', linewidth=1.7, alpha=0.7)
    ax.text(27, 0.01, '25', color='red', fontsize=11, ha='left', zorder=1, alpha=0.7)
    
    ax.axvline(x=90, color='red', linestyle='--', linewidth=1.7, alpha=0.7)
    ax.text(92, 0.01, '90', color='red', fontsize=11, ha='left', zorder=1, alpha=0.7)
    
    ax = sns.scatterplot(
        x='H-Score',
        y='Bioinf pred',
        data=pred_df,
        # hue='H_color',
        palette=palette,
        s=160,
        edgecolor='black',
        linewidth=1,
        alpha=0.7,
        zorder=4
    )
    
    if annotate:
        for _, row in pred_df.iterrows():
            if row['H-Score'] < 100 and row['Bioinf pred'] > 0.68:
                ax.annotate(
                    row['Pseudonym'],
                    xy=(row['H-Score'], row['Bioinf pred']),
                    xytext=(32, 25),  # Text offset
                    textcoords='offset points',
                    ha='right',
                    fontsize=12,
                    arrowprops=dict(arrowstyle='-', color='black')
                )
            elif row['H-Score'] >= 90 and row['Bioinf pred'] < 0.68:
                ax.annotate(
                    row['Pseudonym'],
                    xy=(row['H-Score'], row['Bioinf pred']),
                    xytext=(32, 25),  # Text offset
                    textcoords='offset points',
                    ha='right',
                    fontsize=12,
                    arrowprops=dict(arrowstyle='-', color='black')
                )
    
    ax.set_title("Cohort B: Model 1 Predictions", fontsize=20)
    ax.set_xlabel("H-Score", fontsize=16)
    ax.set_ylabel("Predicted WT prob", fontsize=16)
    
    ax.set_xlim(-10, 300)
    ax.set_ylim(0, 1)
    ax.grid(True, which='both', linestyle='--', linewidth=1.3)
    
    plt.tight_layout()
    plt.show()

# plot_scatter_with_thresholds(pred_df, annotate=True)