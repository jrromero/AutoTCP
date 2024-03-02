# J.R. Romero, A. Ramírez, A. Fuentes-Almoguera, C. García.
# "Automated machine learning for test case prioritisation".
# 2024.

# Script to prepare the datasets and separate them by build

import pandas as pd
import numpy as np
from pathlib import Path 
import Dataset

# Settings
root_path = '../../datasets'
repo_name = 'apache@airavata'
path = root_path + '/datasets/' + repo_name
output_path = Path(root_path + '/data/' + repo_name)
output_path.parent.mkdir(parents=True, exist_ok=True)

# Open dataset
dataset = Dataset.Dataset(path)
df_repo = pd.read_csv(path + '/dataset.csv')
feature_names = df_repo.columns
feature_names = feature_names[~np.isin(feature_names, ['Build', 'Test', 'Verdict', 'Duration'])]

# Convert data for lightgbm
dataset.save_feature_id_map()
df_converted = dataset.convert_to_lightGBM_dataset(df_repo)
feature_ids = df_converted.columns

# This function creates the train/test splits by build (the folder name is the build id)
dataset.create_ranklib_training_sets(df_converted, output_path)