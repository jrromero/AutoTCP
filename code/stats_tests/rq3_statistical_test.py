#    Aurora Ramirez, PhD
#    Knowledge Discovery and Intelligent Systems (KDIS)
#    University of Cordoba, Spain

import pandas as pd
import numpy as np
import os
from scipy import stats
from cliffs_delta import cliffs_delta

# Settings
dir = os.getcwd() + "/tests/rq3/"
dir_data = dir + "/data"
alpha = 0.05
res_filename = dir + "rq3_statistical_analysis.txt"

# Open results
df_rq3 = pd.read_csv(dir_data+"/rq3_results.csv")
conf_names = df_rq3.columns
conf_names = np.delete(conf_names, 0)

# Separate result samples
rf_pair = df_rq3[conf_names[0]]
bt_g_rf_pair = df_rq3[conf_names[1]]
bt_b_rf_pair = df_rq3[conf_names[2]]
pycaret = df_rq3[conf_names[3]]
autotcp = df_rq3[conf_names[4]]

# Output to file
f=open(res_filename, 'w')

# Kruskal-Wallis test
kruskal_result, kruskall_pvalue = stats.kruskal(rf_pair, bt_g_rf_pair, bt_b_rf_pair, pycaret, autotcp)
f.write("== Kruskal-Wallis test ==\n")
f.write("\tStatistics: " + str(kruskal_result) + "\n")
f.write("\tp-value:" + str(kruskall_pvalue) + "\n")
if kruskall_pvalue < alpha:
    f.write("\tThe null hypothesis can be rejected\n\n")
else:
    f.write("\tThe null hypothesis cannot be rejected\n\n")

# Friedman test
#friedman_res = stats.friedmanchisquare(rf_pair, bt_g_rf_pair, bt_b_rf_pair, autotcp)
#f.write("== Friedman test ==\n")
#f.write("\tStatistics: " + str(friedman_res.statistic) + "\n")
#f.write("\tp-value:" + str(friedman_res.pvalue) + "\n")
#if friedman_res.pvalue < alpha:
#    f.write("\tThe null hypothesis can be rejected\n\n")
#else:
#    f.write("\tThe null hypothesis cannot be rejected\n\n")

# Wilcoxon test (pairwise comparisons)
f.write("== Wilcoxon test (pairwise comparisons) ==\n")
all_results = [rf_pair, bt_g_rf_pair, bt_b_rf_pair, pycaret, autotcp]
for i in range(0, len(conf_names)):
    for j in range(i+1, len(conf_names)):
        rank_result, rank_pvalue = stats.ranksums(all_results[i], all_results[j])
        f.write("\t-- " + conf_names[i] + " vs. " + conf_names[j] + " --\n")
        f.write("\t\tStatistics: " + str(rank_result) + "\n")
        f.write("\t\tp-value: " + str(rank_pvalue) + "\n")
        if rank_pvalue < alpha:
            f.write("\t\tThe null hypothesis can be rejected\n")
        else:
            f.write("\t\tThe null hypothesis cannot be rejected\n")

# Cliff's Delta test (effect size, pairwise comparisons)
f.write("\n== Cliff's Delta (pairwise comparisons) ==\n")
for i in range(0, len(conf_names)):
    for j in range(i+1, len(conf_names)):
        cdelta_result, cdelta_label = cliffs_delta(all_results[i], all_results[j])
        f.write("\t-- " + conf_names[i] + " vs. " + conf_names[j] + " --\n")
        f.write("\t\tEffect size: " + cdelta_label + "\n")
        f.write("\t\tp-value: " + str(cdelta_result) + "\n")

f.close()