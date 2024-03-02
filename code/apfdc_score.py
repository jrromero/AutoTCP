# J.R. Romero, A. Ramírez, A. Fuentes-Almoguera, C. García.
# "Automated machine learning for test case prioritisation".
# 2024.

# Script used to calculate the APFDC score of the ensembles trained in the ahmadreza dataset

import pandas as pd
import numpy as np
import joblib
import os
import re

from evoflow.extension.metrics import apfdc_score

curdir = os.path.dirname(os.path.abspath(__file__))
pattern = re.compile(r'.*\_(\d+)\_s.*')

results_folder = 'out_ahmadreza_1seed_auc_reduce_25_25_%s'
output_folder = 'apfdc_scores_%s'
execution = 3 # From 1 to 5, each execution has a different seed
datasets_path = 'dataset/datasets/data/versions/ahmadreza'

datasets = os.listdir(os.path.join(curdir, results_folder % execution, 'output/1h'))

results = pd.DataFrame(columns=['dataset', 'apfdc'])

if os.path.exists(os.path.join(curdir, output_folder % execution)) == False:
	os.mkdir(os.path.join(curdir, output_folder % execution))

for dataset in datasets:
	print(dataset)
	apfdc_scores = []
	apfdc_scores_df = pd.DataFrame(columns=['version', 'apfdc'])

	for version_folder in os.listdir(os.path.join(curdir, results_folder % execution, 'output/1h', dataset)):
		version = pattern.match(version_folder).group(1)
		file = os.path.join(curdir, results_folder % execution, 'output/1h', dataset, version_folder, 'ensemble.pkl')

		if os.path.exists(file):
			model = joblib.load(file)
			data = pd.read_csv(os.path.join(curdir, datasets_path, dataset, version, 'test.csv'))
			X_test = data[[f'f{i}' for i in range(1, 151)]].to_numpy()
			y_test = data[['i_verdict', 'i_duration']].to_numpy()
			y_score = model.predict_proba(X_test)[:, 1]
			apfdc = apfdc_score(y_test, y_score)
			apfdc_scores.append(apfdc)
			apfdc_scores_df = apfdc_scores_df.append({'version': version, 'apfdc': apfdc}, ignore_index=True)

	results = results.append({'dataset': dataset, 'apfdc': str(np.mean(apfdc_scores)) + '+-' + str(np.std(apfdc_scores))}, ignore_index=True)
	apfdc_scores_df = apfdc_scores_df.sort_values(by=['version'], ignore_index=True)
	apfdc_scores_df.to_csv(os.path.join(curdir, output_folder % execution, dataset + '.csv'), index=False)

results = results.sort_values(by=['dataset'], ignore_index=True, key=lambda x: x.str.lower())
results.to_csv(os.path.join(curdir, output_folder % execution, 'apfdc_scores.csv'), index=False)