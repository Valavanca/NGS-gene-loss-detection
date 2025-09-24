import os
from pathlib import Path

from syne_tune.config_space import randint, uniform, loguniform, choice

SPACE_ZYG = {
    'exp_name': 'tune_Zygosity',
    'dataset_path': None,
    'output_dir': None,
    'train_on_cols': 'bins+FC+TCC',

    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'target_col': 'Bioinf Zygosity',
    'num_class': 3,

    'tree_method': choice(['auto', 'exact', 'approx', 'hist']),
    'num_boost_round': 100,
    'learning_rate': loguniform(0.0001, 0.25),
    'max_depth': randint(2, 6),
    'min_child_weight': randint(2, 9),
    'subsample': uniform(0.6, 1.0),
    'colsample_bytree': uniform(0.6, 1.0),
    'gamma': uniform(0, 5),
    'reg_alpha': loguniform(1e-5, 10),
    'reg_lambda': loguniform(1e-5, 10),

    # + TRANSFORMATIONS FIXME
}

SPACE_BIOCALL = {
    'exp_name': 'tune_Biocall',
    'dataset_path': None,
    'output_dir': None,
    'train_on_cols': 'bins+FC+TCC',

    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'target_col': 'Bioinf called as',

    'tree_method': choice(['auto', 'exact', 'approx', 'hist']),
    'num_boost_round': 100,
    'learning_rate': loguniform(0.0001, 0.25),
    'max_depth': randint(2, 6),
    'min_child_weight': randint(2, 9),
    'subsample': uniform(0.6, 1.0),
    'colsample_bytree': uniform(0.6, 1.0),
    'gamma': uniform(0, 5),
    'reg_alpha': loguniform(1e-5, 10),
    'reg_lambda': loguniform(1e-5, 10),

    # + TRANSFORMATIONS FIXME
}

TRANSFORMATIONS = {
    'None': None,
    'log': {
        'log_transform': {}
    },
    'PCA': {
        'add_pca': {
            'n_components': randint(1, 10)
        }
    },
    'log+PCA': {
        'log_transform': {},
        'add_pca': {
            'n_components': randint(1, 10)  # PCA components to keep
        }
    },
    'poly+der+log': {
        'poly_deriv_log': {
            'degree': randint(0, 12), # 1-3 # Degree of polynomial + deriv feature augmentation
        }
    },
    'pca+poly+der+log': {
        'add_pca': {
            'n_components': randint(1, 10)
        },
        'poly_deriv_log': {
            'degree': randint(0, 12), # 1-3 # Degree of polynomial + deriv feature augmentation
        }
    },
    'norm+poly+der+log': {
        'normalize': {},
        'poly_deriv_log': {
            'degree': randint(0, 12), # 1-3 # Degree of polynomial + deriv feature augmentation
        }
    }

}