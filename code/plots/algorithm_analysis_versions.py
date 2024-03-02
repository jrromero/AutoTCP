# J.R. Romero, A. Ramírez, A. Fuentes-Almoguera, C. García.
# "Automated machine learning for test case prioritisation".
# 2024.

# Script used to get the normalized count of an algorithm used in the ensembles trained in the ahmadreza dataset and plot a lineplot

import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp1d
import numpy as np
from joblib import load
import matplotlib.ticker as mtick

curdir = os.path.dirname(os.path.abspath(__file__))

results_folder = 'out_%s_1seed_auc_reduce_25_25_%s'
output_folder = results_folder.replace('out_%s', 'algorithm_analysis').replace('_%s', '')

# Algorithm to analyze, either preprocessing (e.g. SMOTE) or classification (e.g. RandomForestClassifier)
algorithms = ['SMOTE', 'RandomOverSampler', 'RandomForestClassifier']

# Datasets in which the algorithm will be analyzed
datasets_sets = {
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


project_labels = {
'$S_1$': 'Angel-ML@angel',
'$S_2$': 'apache@airavata',
'$S_3$': 'apache@curator',
'$S_4$': 'apache@logging-log4j2',
'$S_5$': 'apache@rocketmq',
'$S_6$': 'apache@shardingsphere',
'$S_7$': 'apache@sling',
'$S_8$': 'b2ihealthcare@snow-owl',
'$S_9$': 'camunda@camunda-bpm-platform',
'$S_{10}$': 'cantaloupe-project@cantaloupe',
'$S_{11}$': 'CompEvol@beast2',
'$S_{12}$': 'eclipse@jetty.project',
'$S_{13}$': 'eclipse@paho.mqtt.java',
'$S_{14}$': 'eclipse@steady',
'$S_{15}$': 'EMResearch@EvoMaster',
'$S_{16}$': 'facebook@buck',
'$S_{17}$': 'Graylog2@graylog2-server',
'$S_{18}$': 'jcabi@jcabi-github',
'$S_{19}$': 'JMRI@JMRI',
'$S_{20}$': 'optimatika@ojAlgo',
'$S_{21}$': 'SonarSource@sonarqube',
'$S_{22}$': 'spring-cloud@spring-cloud-dataflow',
'$S_{23}$': 'thinkaurelius@titan',
'$S_{24}$': 'yamcs@Yamcs',
'$S_{25}$': 'zolyfarkas@spf4j'
}

labels_projects = {project_labels[i]: i for i in project_labels}

for algorithm in algorithms:

	if not os.path.exists(os.path.join(curdir, f'data_{algorithm}.pickle')):
		data = {i: pd.DataFrame() for i in datasets}

		for e in executions:
			if not os.path.exists(os.path.join(curdir, results_folder % ('ahmadreza', e))):
				print(f'Execution {e} not found')
				continue

			results_df = pd.DataFrame()

			for paper, datasets in datasets_sets.items():
				for dataset in datasets:
					print(algorithm, e, dataset)
					os.chdir(os.path.join(curdir, results_folder % (paper, e), 'output/1h', dataset))

					folders = [f for f in os.listdir(os.curdir) if f.endswith('_s%s' % seeds[e])]
					folders.sort()
					versions = folders

					# Get the results for every version
					results = []

					for v in versions:
						os.chdir(os.path.join(curdir, results_folder % (paper, e), 'output/1h', dataset, v))

						try:
							model = load('ensemble.pkl')
						except Exception:
							print(f'Error loading model in dataset {dataset} version {v}')
							continue

						# Check percent of occurrences of the algorithm
						count = 0

						for _, pipeline in model.named_estimators_.items():
							for i in range(len(pipeline.steps)):
								if algorithm in pipeline.steps[i][1].__class__.__name__:
									count += 1

						results.append(count / len(model.named_estimators_))

					data[dataset][str(e)] = results

		os.chdir(curdir)

		with open(f'data_{algorithm}.pickle', 'wb') as f:
			pickle.dump(data, f)

	data = pickle.load(open(f'data_{algorithm}.pickle', 'rb'))

	# he leído todas las versiones

	new_results = pd.DataFrame()
	new_results_sd = pd.DataFrame()

	for paper, datasets in datasets_sets.items():
		for dataset in datasets:
			# Paso 1, hacer la media sobre todas las ejecuciones
			# results = data[dataset].mean(axis=1)
			results = data[dataset]

			# Paso 2, convertir todas las versiones en 0-100
			results.index = pd.date_range('2021/10/01', periods=len(results), freq='1D')
			step = (results.index[-1] - results.index[0]) / 100

			# Paso 3, concatenar las columnas para tener una única serie con todas las semillas
			results = pd.concat([results[i] for i in results.columns])

			# print(results)
			# print('------------------------------')

			# results.reindex(list(range(0,100)), method='ffill')
			results_sd = results.resample(step).std()
			# print(dataset, results.shape)
			results_mean = results.resample(step).mean()
			results_mean.index = list(range(0,101))
			results_sd.index = list(range(0,101))

			# Paso 3, hacer una media sobre una ventana del 10 por ciento
			window_size = 5
			results_mean = results_mean.rolling(window_size, min_periods=int(window_size/2)).mean()
			results_mean.index = list(range(0, 101))
			results_sd = results_sd.rolling(window_size, min_periods=int(window_size/2)).mean()
			results_mean = results_mean[(~ np.isnan(results_mean))]
			results_sd = results_sd[(~ np.isnan(results_sd))]

			new_results[dataset] = results_mean
			new_results_sd[dataset] = results_sd

	fig,ax = plt.subplots(figsize=(10,10))
	markers = ['D', 'o', 's', 'x', '']
	markers_on = pd.Series(list(range(0,100,10)))
	for paper, datasets in datasets_sets.items():
		for index, dataset in enumerate(datasets):
			x = pd.Series(list(range(0, len(new_results[dataset]))))
			y = pd.Series(new_results[dataset])
			cubic_interpolation_model = interp1d(x, y, kind="cubic")
			try:
				cubic_interpolation_model_mas_sd = interp1d(x, y + new_results_sd[dataset], kind="cubic")
				cubic_interpolation_model_menos_sd = interp1d(x, y - new_results_sd[dataset], kind="cubic")
			except ValueError as e:
				raise ValueError('Sizes do not match: ' + str(y.shape) + ' != ' + str(new_results_sd[dataset].shape)) from e
			num_points = 500
			x_ = pd.Series(np.linspace(x.min()+1, x.max()-1, num_points))
			y_ = cubic_interpolation_model(x_)
			y_mas_sd = cubic_interpolation_model_mas_sd(x_)
			y_menos_sd = cubic_interpolation_model_menos_sd(x_)
			new_markers_on = np.trunc(markers_on / 101 * num_points).astype(int)
			ax.plot(x_, 100*y_, markevery=new_markers_on, ls='-', marker=markers[index], label=labels_projects[dataset], ms=8)
			ax.fill_between(x_, 100*y_menos_sd, 100*y_mas_sd, alpha=0.2)

	fontsize = 30
	ax.legend(fontsize=fontsize-4)
	plt.xlabel('% Builds', fontsize=fontsize-2)
	plt.ylabel('% Occurrences', fontsize=fontsize-2)
	plt.title(f'Usage percent of {algorithm}', fontsize=fontsize)
	plt.yticks(fontsize=fontsize-6)
	plt.xticks(fontsize=fontsize-6)
	ax.yaxis.set_major_formatter(mtick.PercentFormatter())
	ax.xaxis.set_major_formatter(mtick.PercentFormatter())
	plt.savefig(os.path.join(curdir, f'{algorithm}.png'), bbox_inches='tight')
	plt.close()


