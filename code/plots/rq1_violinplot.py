# J.R. Romero, A. Ramírez, A. Fuentes-Almoguera, C. García.
# "Automated machine learning for test case prioritisation".
# 2024.

# Script to plot the violin plot of the RQ1 results

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Settings
dir_data = os.getcwd() + "/results/rq1"
dir_res = os.getcwd() + "/plots/rq1"
if not os.path.exists(dir_res):
    os.makedirs(dir_res)

# Open results
confs = os.listdir(dir_data)
n_rows = len(confs)
n_cols = 10 * 5 * 5 # (seeds * projects * builds) 

matrix_apfd = np.zeros((n_rows, n_cols))
i = 0

# For each algorithm configuration, fill the matrix row with the apfd score of all projects, builds and seeds
for c in confs:
    subdir = dir_data + "/" + c
    files = os.listdir(subdir)
    j = 0
    for f in files:
        df = pd.read_csv(subdir + "/" + f)
        matrix_apfd[i, j:j+5] = df['apfd_score'].values
        j = j + 5
    i = i + 1    

# Transpose the matrix, so each configuration is a column
matrix_apfd = np.transpose(matrix_apfd)

# Boxplot
plt.figure(figsize=(15, 10))
x_ticks_labels = ["(APFD, 100%,\n100%, Double)", "(APFD, 100%,\n100%, Single)", "(APFD, 50%,\n100%, Double)", "(APFD, 75%,\n100%, Double)", 
                  "(APFD, 75%,\n50%, Double)", "(APFD, 75%,\n75%, Double)", "(AUC, 100%,\n100%, Double)", "(AUC, 100%,\n100%, Single)",
                  "(AUC, 50%,\n100%, Double)", "(AUC, 75%,\n100%, Double)", "(AUC, 75%,\n50%, Double)", "(AUC, 75%,\n75%, Double)"]
plt.grid()
fontproperties = {'weight': 'bold', 'size': 16}
plt.ylabel("APFDc", fontdict=fontproperties)
plt.xlabel("AutoTCP configuration", fontdict=fontproperties)
plt.xticks(ticks=range(1, len(x_ticks_labels)+1), labels=x_ticks_labels)
plt.tick_params(axis='x', labelsize=8.5)
plt.violinplot(matrix_apfd, showmeans=True)

plt.savefig(dir_res+"/rq1_violinplot.pdf") # or .svg
plt.clf()
