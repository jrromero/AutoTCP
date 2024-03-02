# J.R. Romero, A. Ramírez, A. Fuentes-Almoguera, C. García.
# "Automated machine learning for test case prioritisation".
# 2024.

# Script used to train EvoFlow on a dataset
# Used by evoflow_train.py for each dataset

import os
import sys
import pandas as pd
import joblib
import numpy as np
from argparse import ArgumentParser

from evoflow.base import EvoFlow
from evoflow.utils import log_elite

from evoflow.utils import compute_performance

parser = ArgumentParser()
parser.add_argument('config', help='Path to the configuration file', type=str)
parser.add_argument('train_file', help='Path to the training file', type=str)
parser.add_argument('test_file', help='Path to the test file', type=str)
parser.add_argument('outdir_folder', help='Path to the output folder', type=str)
parser.add_argument('-cv', dest='cross_validation', help='Number of folds for cross validation', type=int, default=5)
parser.add_argument('-s', dest='seed', help='Seed for the random number generator', type=int, default=None)

def apfd_score(data):
    tmp = data.sort_values(by=['C1', 'i_duration'], inplace=False, ascending=[False, True], ignore_index=True)

    if tmp['C1'].unique().shape[0] == 1 and tmp.shape[0] > 1:
        return 0.5
    
    n = tmp.shape[0]
    
    if n <= 1:
        return 1.0
    
    m = tmp[tmp['i_verdict'] > 0].shape[0]
    fault_pos_sum = np.sum(tmp[tmp['i_verdict'] > 0].index + 1)
    apfd = 1 - fault_pos_sum / (n * m) + 1 / (2 * n)
    return apfd

if __name__ == '__main__':

    args = parser.parse_args()

    # Get the command line arguments
    seed = args.seed
    config_file = args.config
    train_file = args.train_file
    test_file = args.test_file
    cv = args.cross_validation
    outdir_folder = args.outdir_folder

    # Load the configuration file within a dictionary
    with open(config_file, 'r') as f:
        config_dic = eval(f.read().replace('\n', ''))

    train_data = pd.read_csv(train_file)
    train_data = train_data.sample(frac=1, random_state=0)
    X_train = train_data[[f'f{i}' for i in range(1, 150 + 1)]].to_numpy()
    y_train = train_data['i_verdict'].to_numpy()
    
    test_data = pd.read_csv(test_file)
    test_data = test_data.sample(frac=1, random_state=0)
    X_test = test_data[[f'f{i}' for i in range(1, 150 + 1)]].to_numpy()
    y_test = test_data['i_verdict'].to_numpy()
    

    cv = min(cv, np.count_nonzero(y_train == 1))

    if cv < 2:
        print(f'Trainig dataset {train_file} does not contain enough positive samples for cross validation. Only {cv} positive samples found. Skipping dataset.')
        sys.exit(1)

    # Run the G3P algorithm and log the results
    outdir = config_dic["outdir"]
    del config_dic["outdir"]

    if outdir not in outdir_folder:
        outdir = os.path.join(outdir, outdir_folder + "_s" + str(seed))
    else:
        outdir = outdir_folder + "_s" + str(seed)
    
    automl = EvoFlow(seed=seed, outdir=outdir, cv=cv, **config_dic)

    automl.fit(X_train, y_train)
    log_elite(automl, X_train, y_train, X_test, y_test)

    model = joblib.load(os.path.join(outdir, "ensemble.pkl"))
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[::, 1]

    performance = compute_performance(y_test, y_pred, y_proba)
    print(performance)

    test_data['Pred'] = y_pred
    test_data[['C0', 'C1']] = model.predict_proba(X_test)

    position = 0
    relative_position = 0
    correct_prediction = False

    tmp = test_data.sort_values(by=['C1', 'i_duration'], inplace=False, ascending=[False, True], ignore_index=True)

    for i, d in tmp.iterrows():
        if d['i_verdict'] == 1:
            position = i + 1
            relative_position = position / len(tmp)
            print('First occurrence in', i + 1, 'out of', len(tmp), 'tests')
            
            if d['i_verdict'] == d['Pred']:
                correct_prediction = True

            break

    apfd = apfd_score(test_data)

    print('Position:', position)
    print('Relative position:', relative_position)
    print('APFD:', apfd)
    print('Correct prediction:', correct_prediction)

    with open(os.path.join(outdir, "test_performance.txt"), "w") as f:
        f.write(str(y_pred.tolist()) + '\n')
        f.write(str(performance) + '\n')
        f.write('Position: ' + str(position) + '\n')
        f.write('Relative position: ' + str(relative_position) + '\n')
        f.write('APFD: ' + str(apfd) + '\n')
        f.write('Correct prediction: ' + str(correct_prediction))
    
    for ind in automl.elite:
        del ind.pipeline

    del model