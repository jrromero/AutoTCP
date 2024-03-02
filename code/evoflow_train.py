# J.R. Romero, A. Ramírez, A. Fuentes-Almoguera, C. García.
# "Automated machine learning for test case prioritisation".
# 2024.

# Script used to train EvoFlow on all datasets located in a folder

import os
import re
import pandas as pd

from argparse import ArgumentParser
from multiprocessing import Pool
from subprocess import Popen, PIPE

parser = ArgumentParser()
parser.add_argument('datasets_folder', help='Path to the folder with the datasets', type=str)
parser.add_argument('results_folder', help='Path to the folder where the results will be stored', type=str)
parser.add_argument('config', help='Path to the configuration file', type=str)
parser.add_argument('-cv', dest='cross_validation', help='Number of folds for cross validation', type=int, default=5)
parser.add_argument('-s', dest='seed', help='Seed for the random number generator', type=int, default=None)
parser.add_argument('-p', dest='processes', help='Number of processes to use', type=int, default=4)

train_program = 'dataset_test.py'

def prepare_datasets_executions(datasets_path, results_folder, config, cv, seed):
    datasets = os.listdir(datasets_path)
    datasets.sort()

    executions = []

    for dataset in datasets:
        files = os.listdir(os.path.join(datasets_path, dataset))
        files.sort()
        versions = set()

        for i in range(len(files)):
            version = re.search(r'(\d+)', files[i]).group(1)
            versions.add(int(version))

        versions = sorted(versions)
        executions.extend([(datasets_path, results_folder, dataset, v, cv, seed, config) for v in versions])

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

    train_file = f'{datasets_path}/{dataset}/{version}/train.csv'
    test_file = f'{datasets_path}/{dataset}/{version}/test.csv'
    
    cmd = f'python3 {train_program} {config_file} {train_file} {test_file} {results_folder}/{outdir}/{dataset}/{dataset}_{version} -s {seed} -cv {cv}'
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    print(f'Dataset {dataset} version {version}')
    print(out.decode('utf-8'))
    print(err.decode('utf-8'))

    with open(results_folder + '/' + dataset + '.out', 'a') as file:
        file.write(out.decode('utf-8'))

    if p.returncode == 1:
        return (dataset, version, -1, False)

    relative_position = float(re.search(r'Relative position: (.+)\n', out.decode('utf-8')).group(1))
    apfd_score = float(re.search(r'APFD: (.+)\n', out.decode('utf-8')).group(1))
    correct_prediction = (re.search(r'Correct prediction: (.+)\n', out.decode('utf-8')).group(1)).lower() == 'true'

    return (dataset, version, relative_position, apfd_score, correct_prediction)

if __name__ == '__main__':
    args = parser.parse_args()
    args.results_folder = args.results_folder.strip('/')

    if not os.path.exists(args.results_folder):
        os.mkdir(args.results_folder)
    else:
        for file in os.listdir(args.results_folder):
            path = args.results_folder + '/' + file

            if os.path.isfile(path):
                os.remove(path)

    executions = prepare_datasets_executions(args.datasets_folder, args.results_folder, args.config, args.cross_validation, args.seed)

    with Pool(args.processes) as p:
        results = p.map(execute, executions)

    results = [r for r in results if r is not None and r[2] != -1]

    relative_positions = pd.DataFrame(columns=['dataset', 'version', 'relative_position', 'apfd_score', 'correct_prediction'])

    for r in results:
        relative_positions.loc[len(relative_positions.index)] = r

    datasets = relative_positions['dataset'].unique().tolist()
    datasets.sort()

    for d in datasets:
        dataset_result = relative_positions[relative_positions['dataset'] == d]
        dataset_result = dataset_result.sort_values(by=['version'], ignore_index=True)
        dataset_result.to_csv(f'{args.results_folder}/{d}.csv', index=False)
