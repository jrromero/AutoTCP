# J.R. Romero, A. Ramírez, A. Fuentes-Almoguera, C. García.
# "Automated machine learning for test case prioritisation".
# 2024.

# Script to process the results for the RQ1 analysis

import pandas as pd
import numpy as np
import os
import glob

# Settings
dir = os.getcwd() + "/../../results/rq1/"
projects = ["Angel-ML@angel", "apache@logging-log4j2", "apache@rocketmq", "CompEvol@beast2", "eclipse@paho.mqtt.java"]
confs = os.listdir(dir)

# Output directory
dir_res = os.getcwd() + "/../../tests/rq1/data"
if not os.path.exists(dir_res):
    os.makedirs(dir_res)

# For each algorithm configuration
for c in confs:
    
    data_confs = []
    data_projects = []
    data_avg = []
    data_std = []
    
    # All files from this configuration
    subdir = dir + "/" + c
    files = os.listdir(subdir)

    # Process files by project
    for p in projects:
        
        # The CSV files of this project (one per seed)
        data_files = glob.glob("".join([subdir, "\\" + p + "*.csv"]))
        n_seeds = len(data_files)
        builds = pd.read_csv(data_files[0])['version']
        n_builds = len(builds)
        
        # Matrix to store APFD values (rows=seeds, columns=builds)
        apfd = np.zeros((n_seeds, n_builds))
        i = 0
        
        # Get APFD value for each seed and build
        for f in data_files:
            df = pd.read_csv(f)
            apfd_values = df['apfd_score'].values
            apfd[i] = apfd_values
            i = i + 1

        # Average and standard deviation by build
        avg_by_build = apfd.mean(axis=0)
        stdev_by_build = apfd.std(axis=0)

        # Save the statistics of each build together with the configuration and project id
        for i in range(0, n_builds):
            data_confs.append(c)
            data_projects.append(p+"-build-"+str(builds[i]))
            data_avg.append(avg_by_build[i])
            data_std.append(stdev_by_build[i])

    # Save summary results in CSV file
    df_rq1 = pd.DataFrame({"conf": data_confs, "project_build": data_projects, "apfd_avg": data_avg, "apfd_stdev": data_std})

    df_name = dir_res+"/rq1_" + c + "_summary.csv"
    df_rq1.to_csv(df_name, index=False)