# J.R. Romero, A. Ramírez, A. Fuentes-Almoguera, C. García.
# "Automated machine learning for test case prioritisation".
# 2024.

# Statistical analysis for RQ1

import pandas as pd
import numpy as np
import os
from scipy import stats
from cliffs_delta import cliffs_delta

# Settings
dir = os.getcwd() + "/../../tests/rq1/"
dir_data = dir + "/data"
alpha = 0.05
res_filename = dir + "rq1_statistical_analysis.txt"

# Open results
df_rq1 = pd.read_csv(dir_data+"/rq1_summary.csv")
conf_names = df_rq1.columns
conf_names = np.delete(conf_names, 0)

# Separate result samples
res1 = df_rq1[conf_names[0]]
res2 = df_rq1[conf_names[1]]
res3 = df_rq1[conf_names[2]]
res4 = df_rq1[conf_names[3]]
res5 = df_rq1[conf_names[4]]
res6 = df_rq1[conf_names[5]]
res7 = df_rq1[conf_names[6]]
res8 = df_rq1[conf_names[7]]
res9 = df_rq1[conf_names[8]]
res10 = df_rq1[conf_names[9]]
res11 = df_rq1[conf_names[10]]
res12 = df_rq1[conf_names[11]]

# Output to file
f=open(res_filename, 'w')

# Kruskal-Wallis test
kruskal_result, kruskall_pvalue = stats.kruskal(res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, res11, res12)
f.write("== Kruskal-Wallis test ==\n")
f.write("\tStatistics: " + str(kruskal_result) + "\n")
f.write("\tp-value:" + str(kruskall_pvalue) + "\n")
if kruskall_pvalue < alpha:
    f.write("\tThe null hypothesis can be rejected\n\n")
else:
    f.write("\tThe null hypothesis cannot be rejected\n\n")

# Friedman test
#friedman_res = stats.friedmanchisquare(res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, res11, res12)
#f.write("== Friedman test ==\n")
#f.write("\tStatistics: " + str(friedman_res.statistic) + "\n")
#f.write("\tp-value:" + str(friedman_res.pvalue) + "\n")
#if friedman_res.pvalue < alpha:
#    f.write("\tThe null hypothesis can be rejected\n\n")
#else:
#    f.write("\tThe null hypothesis cannot be rejected\n\n")

# Wilcoxon test (pairwise comparisons)
f.write("== Wilcoxon test (pairwise comparisons) ==\n")
all_results = [res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, res11, res12]
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