# J.R. Romero, A. Ramírez, A. Fuentes-Almoguera, C. García.
# "Automated machine learning for test case prioritisation".
# 2024.

# Script used to merge the .csv files generated by apfdc_score.py

import pandas as pd
import os

curdir = os.path.dirname(os.path.abspath(__file__))

results_folder = '../results/rq2/auc_reduce_75_75_double_mut_apfdc/apfdc_scores_s%s'
output_folder = results_folder.replace('apfdc_scores_s%s', 'apfdc_scores_dataset')
executions = ['1', '2', '3', '4', '5']
seeds = {
    '1': 123,
    '2': 897,
    '3': 564,
    '4': 000,
    '5': 789
}
datasets = ['Angel-ML@angel', 'apache@airavata', 'apache@curator', 'apache@logging-log4j2', 'apache@rocketmq', 'apache@shardingsphere', 'apache@sling', 'b2ihealthcare@snow-owl', 'camunda@camunda-bpm-platform', 'cantaloupe-project@cantaloupe', 'CompEvol@beast2', 'eclipse@jetty.project', 'eclipse@paho.mqtt.java', 'eclipse@steady', 'EMResearch@EvoMaster', 'facebook@buck', 'Graylog2@graylog2-server', 'jcabi@jcabi-github', 'JMRI@JMRI', 'optimatika@ojAlgo', 'SonarSource@sonarqube', 'spring-cloud@spring-cloud-dataflow', 'thinkaurelius@titan', 'yamcs@Yamcs', 'zolyfarkas@spf4j']

if not os.path.exists(os.path.join(curdir, output_folder)):
    os.mkdir(os.path.join(curdir, output_folder))

for d in datasets:
    print(f'Dataset {d}')
    results_dict = {}

    for e in executions:
        data = pd.read_csv(os.path.join(curdir, results_folder % seeds[e], d + '.csv'))
        data = data.sort_values(by=['version'], ignore_index=True)
        results_dict[e] = data

    # Merge all executions by version
    results = results_dict['1']

    for e in executions[1:]:
        results = results.merge(results_dict[e], on='version', suffixes=('', '_' + e))

    results.columns = ['version'] + [f'Seed{e}' for e in executions]
    results = results.sort_values(by=['version'], ignore_index=True)
    results.to_csv(os.path.join(curdir, output_folder, d + '.csv'), index=False)