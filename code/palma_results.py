# J.R. Romero, A. Ramírez, A. Fuentes-Almoguera, C. García.
# "Automated machine learning for test case prioritisation".
# 2024.

# Script used to train EvoFlow on a the palma dataset

import os
import subprocess


base_path = os.path.dirname(os.path.abspath(__file__))
versions_path = os.path.join(base_path, 'dataset/datasets/data/versions/palma')

if __name__ == '__main__':
	#train_program = 'evoflow_train_v3.py' # To train in all versions
	train_program = 'percent_versions_train.py' # To train in 20%, 40%, 60%, 80% and 100% of the versions
	datasets_folder = versions_path.replace(' ', '\ ')
	results_folder = 'out_palma_2seeds_auc_reduce_25_25'
	config_file = os.path.join(base_path, 'evoflow/configs/evoflow_1h.py')
	cv = 5
	seed = 123
	processes = 8

	# Command line to train in all versions
	#cmd = f'python3 {train_program} {datasets_folder} {results_folder} {config_file} -cv {cv} -s {seed} -p {processes}'
	# Command line to train in 20%, 40%, 60%, 80% and 100% of the versions
	cmd = f'python3 {train_program} {datasets_folder} {results_folder} {config_file} -cv {cv} -p {processes}'
	p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

	stdout, stderr = p.communicate()
	print(stdout.decode('utf-8'))
	print(stderr.decode('utf-8'))