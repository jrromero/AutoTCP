# J.R. Romero, A. Ramírez, A. Fuentes-Almoguera, C. García.
# "Automated machine learning for test case prioritisation".
# 2024.

# Script to prepare the data for the RQ1 analysis

import pandas as pd
import numpy as np
import os

# Settings
dir = os.getcwd() + "/../../tests/rq1/data/"
confs = os.listdir(dir)

# Matrix to store APFD values (rows=configurations, columns=datasets)
n_confs = len(confs)
df = pd.read_csv(dir + "/" + confs[0])
dataset_names = df['project_build']
n_datasets = len(dataset_names)
apfd = np.zeros((n_confs, n_datasets))
data_confs = []

# For each file with average results of one configuration
for i in range(0, n_confs):
    
    conf_results = pd.read_csv(dir + "/" + confs[i])
    conf_name = str(conf_results['conf'].values[0])
    apfd_values = conf_results['apfd_avg'].values
    data_confs.append(conf_name)
    apfd[i] = apfd_values

# Transpose matrix
apfd = np.transpose(apfd)

# Save summary results in CSV file
df_rq1 = pd.DataFrame(data=apfd, columns=[data_confs])
df_rq1.insert(0, 'dataset', dataset_names)

df_name = dir + "/rq1_summary.csv"
df_rq1.to_csv(df_name, index=False)