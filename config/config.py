
import os
import random
import numpy as np
from pathlib import Path

import pandas as pd

SEED = 67

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = Path('./data/processed/train_val_pids_09052025.csv')
TEST_COL = 'Test set'

BIN_COLUMNS = [
    'chr10.136_NC', 'chr10.137_NC', 'chr10.138_NC', 'chr10.139_NC', 'chr10.140_NC', 
    'chr10.141_NC', 'chr10.142_NC', 'chr10.143_NC', 'chr10.144_NC', 'chr10.145_NC', 
    'chr10.146_NC', 'chr10.147_NC', 'chr10.148_NC', 'chr10.149_NC', 'chr10.150_NC', 
    'chr10.151_NC', 'chr10.152_NC', 'chr10.153_NC', 'chr10.154_NC', 'chr10.155_NC', 
    'chr10.156_NC', 'chr10.157_NC', 'chr10.158_NC', 'chr10.159_NC', 'chr10.160_NC', 
    'chr10.161_NC', 'chr10.162_NC', 'chr10.163_NC', 'chr10.164_NC', 'chr10.165_NC', 
    'chr10.166_NC', 'chr10.167_NC', 'chr10.168_NC', 'chr10.169_NC', 'chr10.170_NC', 
    'chr10.171_NC', 'chr10.172_NC', 'chr10.173_NC', 'chr10.174_NC'
]

# https://www.ensembl.org/Homo_sapiens/Gene/Summary?db=core;g=ENSG00000171862;r=10:87862638-87971930
BINS_INFO = {
    'chr10.136_NC': ['Exon'], # EMAR
    'chr10.137_NC': [], # EMAR
    'chr10.138_NC': [], # EMAR
    'chr10.139_NC': ['Enhancer'], # EMAR
    'chr10.140_NC': ['Enhancer'], # EMAR
    'chr10.141_NC': [],
    'chr10.142_NC': [],
    'chr10.143_NC': [],
    'chr10.144_NC': [],
    'chr10.145_NC': ['Exon'], 
    'chr10.146_NC': [],
    'chr10.147_NC': [],
    'chr10.148_NC': ['Enhancer'], # EMAR
    'chr10.149_NC': [],
    'chr10.150_NC': [], 
    'chr10.151_NC': [],
    'chr10.152_NC': [],
    'chr10.153_NC': [],
    'chr10.154_NC': [],
    'chr10.155_NC': ['Exon'], 
    'chr10.156_NC': ['Enhancer'], # EMAR
    'chr10.157_NC': [],
    'chr10.158_NC': ['Exon'],
    'chr10.159_NC': ['Exon'],
    'chr10.160_NC': [], 
    'chr10.161_NC': [],
    'chr10.162_NC': [],
    'chr10.163_NC': [],
    'chr10.164_NC': [],
    'chr10.165_NC': [], 
    'chr10.166_NC': [],
    'chr10.167_NC': [],
    'chr10.168_NC': ['Exon'],
    'chr10.169_NC': ['Exon'],
    'chr10.170_NC': [], 
    'chr10.171_NC': [],
    'chr10.172_NC': ['Exon'],
    'chr10.173_NC': ['Exon'],
    'chr10.174_NC': ['Exon'],
}

def get_norm_reference():
    file_path = './data/PTEN/baseline_normalizedBinCount.tsv'
    nct_df = pd.read_csv(file_path, sep='\t')
    nct_df = nct_df[nct_df['Gene'] == 'PTEN'].reset_index(drop=True)
    nct_df = nct_df[['binName',	'NormalizedCount_MEAN']]
    nct_df = nct_df.sort_values(by='binName')
    return nct_df['NormalizedCount_MEAN'].values

BIN_NORM_REFERENCE = [] # get_norm_reference()


# --- XGBoost
EARLY_STOPPING_ROUNDS = 10


_default_config = {
    # --- General
    'exp_name': 'tune_default',
    'output_dir': str(Path('./output/runs')),
    # 'dataset_path': str(Path('./data/clean/dataset_upd_1410.csv')),
    'train_on_cols': 'bins+FC+TCC',  # Default choice for columns

    # --- XGBoost
    'add_poly_degree': 4,  # No polynomial features by default
    'learning_rate': 0.01,  # Default learning rate
    'max_depth': 3,  # Maximum depth of trees
    'min_child_weight': 3,  # Minimum child weight
    'subsample': 0.8,  # Subsample ratio
    'colsample_bytree': 0.8,  # Column subsample ratio
    'reg_alpha': 1e-3,  # L1 regularization
    'reg_lambda': 1.0,  # L2 regularization
    'num_boost_round': 3,  # Number of boosting rounds

    # --- Target
    'objective': 'multi:softprob', #'binary:logistic',
    'num_class': 3,
    'eval_metric': 'mlogloss', # 'logloss',
    'target_col': 'Bioinf Zygosity',
}


BIOCALL_CONFIG = {
    # --- General
    'exp_name': 'retrain_1219',
    'output_dir': str(Path('./output/selected_Bioinf/1015_1718_19_p7_508_loss')),
    'dataset_path': str(DATASET_PATH.absolute()),
    'train_on_cols': 'bins+FC+TCC',

    # --- XGBoost
    'add_poly_degree': 7,                   # No polynomial features by default
    'learning_rate': 0.19744472029067167,   # Default learning rate
    'max_depth': 6,                         # Maximum depth of trees
    'min_child_weight': 2,                  # Minimum child weight
    'subsample': 0.6060047115752186,        # Subsample ratio
    'colsample_bytree': 0.699597883923433,  # Column subsample ratio
    'reg_alpha': 0.04667969832175275,       # L1 regularization
    'reg_lambda': 0.00014671564346191828,   # L2 regularization
    'num_boost_round': 100,                 # Number of boosting rounds
    'best_num_boost_round': 16,             # Best number of boosting rounds
    
    # --- Target
    'objective': 'binary:logistic',         # 'binary:logistic', multi:softprob
    'eval_metric': 'logloss',               
    'target_col': 'Bioinf called as',
}


ZYGOSITY_CONFIG = {
    # --- General
    'exp_name': 'retrain_1219',
    'output_dir': str(Path('./output/selected_Zygosity/1015_2349_13_p2_432_best')),
    'dataset_path': str(DATASET_PATH.absolute()),
    'train_on_cols': 'bins+FC+TCC',

    # --- XGBoost
    'add_poly_degree': 2,                   # No polynomial features by default
    'learning_rate': 0.1164813305114614,    # Default learning rate
    'max_depth': 3,                         # Maximum depth of trees
    'min_child_weight': 2,                  # Minimum child weight
    'subsample': 0.629273564088229,         # Subsample ratio
    'colsample_bytree': 0.6,                # Column subsample ratio
    'reg_alpha': 0.0007652055082149236,     # L1 regularization
    'reg_lambda': 0.0010087795232010305,    # L2 regularization
    'num_boost_round': 100,                 # Number of boosting rounds

    # --- Target
    'objective': 'multi:softprob',          # 'binary:logistic',
    'num_class': 3,
    'eval_metric': 'mlogloss',          
    'target_col': 'Bioinf Zygosity',
}
