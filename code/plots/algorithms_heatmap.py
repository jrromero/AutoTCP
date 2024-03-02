# J.R. Romero, A. Ramírez, A. Fuentes-Almoguera, C. García.
# "Automated machine learning for test case prioritisation".
# 2024.

# Script used to get the normalized count of the algorithms used in the ensembles trained in the ahmadreza dataset and plot a heatmap

from joblib import load

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

curdir = os.path.dirname(os.path.abspath(__file__))

results_folder = 'out_%s_1seed_auc_reduce_25_25_%s'
output_folder = results_folder.replace('out_%s', 'algorithms_heatmap').replace('_%s', '')
datasets_sets = {
	'palma': ['chart', 'closure', 'joda', 'math', 'lang'],
	'ahmadreza': ['Angel-ML@angel', 'apache@airavata', 'apache@curator', 'apache@logging-log4j2', 'apache@rocketmq', 'apache@shardingsphere', 'apache@sling', 'b2ihealthcare@snow-owl', 'camunda@camunda-bpm-platform', 'cantaloupe-project@cantaloupe', 'CompEvol@beast2', 'eclipse@jetty.project', 'eclipse@paho.mqtt.java', 'eclipse@steady', 'EMResearch@EvoMaster', 'facebook@buck', 'Graylog2@graylog2-server', 'jcabi@jcabi-github', 'JMRI@JMRI', 'optimatika@ojAlgo', 'SonarSource@sonarqube', 'spring-cloud@spring-cloud-dataflow', 'thinkaurelius@titan', 'yamcs@Yamcs', 'zolyfarkas@spf4j']
}
executions = ['1', '2', '3', '4', '5']
seeds = {
    '1': 123,
    '2': 897,
    '3': 564,
    '4': 000,
    '5': 789
}

if not os.path.exists(os.path.join(curdir, output_folder)):
    os.mkdir(os.path.join(curdir, output_folder))

for e in executions:
    results_df_dict = {}

    preprocess_algorithms_set = set()
    classification_algorithms_set = set()

    columns = ['algorithm']

    for paper, datasets in datasets_sets.items():
        for dataset in datasets:
            print(dataset)
            os.chdir(os.path.join(curdir, results_folder % (paper, e), 'output/1h', dataset))
            columns.append(dataset)

            preprocess_algorithms_df = pd.DataFrame(columns=['algorithm', 'count'])
            classification_algorithms_df = pd.DataFrame(columns=['algorithm', 'count'])

            preprocess_algorithms_count = {}
            classification_algorithms_count = {}

            folders = [f for f in os.listdir(os.curdir) if f.endswith('_s%s' % seeds[e])]

            for f in folders:
                os.chdir(os.path.join(curdir, results_folder % (paper, e), 'output/1h', dataset, f))
                #print(os.path.abspath(os.curdir))
                try:
                    model = load('ensemble.pkl')
                except Exception:
                    print(f'Error loading model in dataset {dataset} version {f}')
                    continue

                for _, pipeline in model.named_estimators_.items():
                    # Preprocess algorithms
                    for i in range(len(pipeline.steps) - 1):
                        if pipeline.steps[i][1].__class__.__name__ in preprocess_algorithms_count:
                            preprocess_algorithms_count[pipeline.steps[i][1].__class__.__name__] += 1
                        else:
                            preprocess_algorithms_count[pipeline.steps[i][1].__class__.__name__] = 1

                    # Classification algorithm
                    if pipeline.steps[-1][1].__class__.__name__ in classification_algorithms_count:
                        classification_algorithms_count[pipeline.steps[-1][1].__class__.__name__] += 1
                    else:
                        classification_algorithms_count[pipeline.steps[-1][1].__class__.__name__] = 1

            preprocess_algorithms_set.update(preprocess_algorithms_count.keys())
            classification_algorithms_set.update(classification_algorithms_count.keys())

            for algorithm, count in preprocess_algorithms_count.items():
                preprocess_algorithms_df = preprocess_algorithms_df.append({'algorithm': algorithm, 'count': count}, ignore_index=True)

            for algorithm, count in classification_algorithms_count.items():
                classification_algorithms_df = classification_algorithms_df.append({'algorithm': algorithm, 'count': count}, ignore_index=True)

            # Normalize the counts and round to 4 decimals
            preprocess_algorithms_df['count'] = preprocess_algorithms_df['count'] / preprocess_algorithms_df['count'].sum()
            preprocess_algorithms_df['count'] = preprocess_algorithms_df['count'].apply(lambda x: round(x, 4))
            classification_algorithms_df['count'] = classification_algorithms_df['count'] / classification_algorithms_df['count'].sum()
            classification_algorithms_df['count'] = classification_algorithms_df['count'].apply(lambda x: round(x, 4))

            preprocess_algorithms_df = preprocess_algorithms_df.set_index('algorithm')
            classification_algorithms_df = classification_algorithms_df.set_index('algorithm')
            results_df_dict[dataset] = (preprocess_algorithms_df, classification_algorithms_df)

    datasets = columns[1:]

    # Initialize a dataframe having as columns the datasets and as rows the algorithms
    preprocess_algorithms_df = pd.DataFrame(columns=columns)
    classification_algorithms_df = pd.DataFrame(columns=columns)

    preprocess_algorithms_set = sorted(preprocess_algorithms_set)
    classification_algorithms_set = sorted(classification_algorithms_set)

    for algorithm in preprocess_algorithms_set:
        preprocess_algorithms_df = preprocess_algorithms_df.append({'algorithm': algorithm}, ignore_index=True)

    for algorithm in classification_algorithms_set:
        classification_algorithms_df = classification_algorithms_df.append({'algorithm': algorithm}, ignore_index=True)

    preprocess_algorithms_df = preprocess_algorithms_df.set_index('algorithm')
    classification_algorithms_df = classification_algorithms_df.set_index('algorithm')

    for dataset in datasets:
        preprocess_algorithms_df[dataset] = results_df_dict[dataset][0]['count']
        classification_algorithms_df[dataset] = results_df_dict[dataset][1]['count']

    preprocess_algorithms_df = preprocess_algorithms_df.fillna(0)
    classification_algorithms_df = classification_algorithms_df.fillna(0)

    preprocess_algorithms_df.to_csv(os.path.join(curdir, output_folder, 'preprocess_algorithms_%s.csv' % e))
    classification_algorithms_df.to_csv(os.path.join(curdir, output_folder, 'classification_algorithms_%s.csv' % e))

    # Plot the heatmap
    plt.figure(figsize=(20, 10))
    plt.title(f'Preprocess algorithms heatmap seed {seeds[e]}')
    sns.heatmap(preprocess_algorithms_df, annot=True, cmap='Blues', fmt='g')
    plt.savefig(os.path.join(curdir, output_folder, 'preprocess_algorithms_%s.png' % e))
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.title(f'Classification algorithms heatmap seed {seeds[e]}')
    sns.heatmap(classification_algorithms_df, annot=True, cmap='Blues', fmt='g')
    plt.savefig(os.path.join(curdir, output_folder, 'classification_algorithms_%s.png' % e))
    plt.close()