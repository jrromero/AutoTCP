# J.R. Romero, A. Ramírez, A. Fuentes-Almoguera, C. García.
# "Automated machine learning for test case prioritisation".
# 2024.

# Script used to train EvoFlow on a few versions of a dataset with multiple seeds

import pandas as pd
import os
import re

from argparse import ArgumentParser
from multiprocessing import Pool
from subprocess import Popen, PIPE

parser = ArgumentParser(description='Train EvoFlow on all datasets located in a given folder.')
parser.add_argument('datasets_folder', help='Path to the folder with the datasets', type=str)
parser.add_argument('results_folder', help='Path to the folder where the results will be stored', type=str)
parser.add_argument('config', help='Path to the configuration file', type=str)
parser.add_argument('-cv', dest='cross_validation', help='Number of folds for cross validation', type=int, default=5)
parser.add_argument('-p', dest='processes', help='Number of processes to use', type=int, default=4)

# Script used to train EvoFlow on a dataset
train_program = 'dataset_test.py'

seeds = [000, 123, 456, 789, 231, 564, 897, 321, 654, 987]

def prepare_datasets_executions(datasets_path, results_folder, config, cv, seed):
    datasets = os.listdir(datasets_path)
    datasets.sort()

    executions = []

    for dataset in datasets:
        versions = os.listdir(os.path.join(datasets_path, dataset))
        versions = sorted([int(v) for v in versions])
        
        # Get the total number of versions
        total_versions = len(versions)

        # Get the index of the versions that represent the 20%, 40%, 60%, 80% and 100% of the total number of versions
        percent_20_index = next(i for i, v in enumerate(versions) if ((i+1) / total_versions) >= 0.2)
        percent_40_index = next(i for i, v in enumerate(versions) if ((i+1) / total_versions) >= 0.4)
        percent_60_index = next(i for i, v in enumerate(versions) if ((i+1) / total_versions) >= 0.6)
        percent_80_index = next(i for i, v in enumerate(versions) if ((i+1) / total_versions) >= 0.8)
        percent_100_index = next(i for i, v in enumerate(versions) if ((i+1) / total_versions) >= 1.00)

        # Get the versions that represent the 20%, 40%, 60%, 80% and 100% of the total number of versions
        percent_20 = versions[percent_20_index]
        percent_40 = versions[percent_40_index]
        percent_60 = versions[percent_60_index]
        percent_80 = versions[percent_80_index]
        percent_100 = versions[percent_100_index]

        executions.append((datasets_path, results_folder, dataset, percent_20, cv, seed, config))
        executions.append((datasets_path, results_folder, dataset, percent_40, cv, seed, config))
        executions.append((datasets_path, results_folder, dataset, percent_60, cv, seed, config))
        executions.append((datasets_path, results_folder, dataset, percent_80, cv, seed, config))
        executions.append((datasets_path, results_folder, dataset, percent_100, cv, seed, config))

    return executions

def execute(execution):
    datasets_path = execution[0]
    results_folder = execution[1]
    dataset = execution[2]
    version = execution[3]
    cv = execution[4]
    seed = execution[5]
    config_file = execution[6]

    # Load the configuration file within a dictionary
    with open(config_file, 'r') as f:
        config_dic = eval(f.read().replace('\n', ''))
    
    outdir = config_dic["outdir"]

    train_file = os.path.join(datasets_path, dataset, str(version), 'train.csv')
    test_file = os.path.join(datasets_path, dataset, str(version), 'test.csv')

    results_path = os.path.join(results_folder, outdir, dataset, f'{dataset}_{version}')
    
    cmd = f'python3 {train_program} {config_file} {train_file} {test_file} {results_path} -s {seed} -cv {cv}'
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    print(f'Dataset {dataset} version {version}')
    print(out.decode('utf-8'))
    print(err.decode('utf-8'))

    out_file = os.path.join(results_folder, f'{dataset}_s{seed}.out')

    with open(out_file, 'a') as file:
        file.write(out.decode('utf-8'))
        file.write(err.decode('utf-8'))

    if p.returncode == 1:
        return (dataset, version, -1, False)

    relative_position = float(re.search(r'Relative position: (.+)\n', out.decode('utf-8')).group(1))
    apfd_score = float(re.search(r'APFD: (.+)\n', out.decode('utf-8')).group(1))
    correct_prediction = (re.search(r'Correct prediction: (.+)\n', out.decode('utf-8')).group(1)).lower() == 'true'

    return (dataset, version, seed, relative_position, apfd_score, correct_prediction)

if __name__ == '__main__':

    args = parser.parse_args()

    # Get the command line arguments
    datasets_folder = args.datasets_folder
    results_folder = args.results_folder
    config = args.config
    cross_validation = args.cross_validation
    processes = args.processes

    results_folder = results_folder.strip('/')

    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    executions = []
    
    for seed in seeds:
        executions.extend(prepare_datasets_executions(datasets_folder, results_folder, config, cross_validation, seed))

    with Pool(processes) as p:
        results = p.map(execute, executions)

    results = [r for r in results if r is not None and r[2] != -1]

    relative_positions = pd.DataFrame(columns=['dataset', 'version', 'seed', 'relative_position', 'apfd_score', 'correct_prediction'])

    for r in results:
        relative_positions.loc[len(relative_positions.index)] = r

    datasets = relative_positions['dataset'].unique().tolist()
    datasets.sort()

    for d in datasets:
        dataset_result = relative_positions[relative_positions['dataset'] == d]
        
        for s in seeds:
            dataset_seed_result = dataset_result[dataset_result['seed'] == s]
            dataset_seed_result = dataset_seed_result.sort_values(by='version', ascending=True, ignore_index=True)
            dataset_seed_result.to_csv(os.path.join(results_folder, f'{d}_s{s}.csv'), index=False)