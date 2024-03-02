import numpy as np
import pandas as pd

def apfd_score(y_true, y_score):
    """
    Computes the APFD (Average Percent of Faults Detected) score for a given set of predictions and true labels.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels

    y_score : array-like of shape (n_samples,)
        Target scores

    Returns
    -------
    apfd : float
        The APFD score
    """
    pred = pd.DataFrame(columns=['verdict', 'score'])
    pred['verdict'] = y_true
    pred['score'] = y_score
    pred.sort_values('score', ascending=False, inplace=True, ignore_index=True)

    if pred['score'].unique().shape[0] == 1 and pred.shape[0] > 1:
        return 0.5
    
    n = pred.shape[0]
    
    if n <= 1:
        return 1.0
    
    m = pred[pred['verdict'] > 0].shape[0]
    fault_pos_sum = np.sum(pred[pred['verdict'] > 0].index + 1)
    apfd = 1 - fault_pos_sum / (n * m) + 1 / (2 * n)
    return apfd

def apfdc_score(y_true, y_score):
    """
    Computes the APFDC (Average Percent of Faults Detected with Cost) score for a given set of predictions and true labels.

    Parameters
    ----------
    y_true : array-like of shape (n_samples, 2)
        True labels and costs. Expects that the first column is the label and the second column is the cost

    y_score : array-like of shape (n_samples,)
        Target scores

    Returns
    -------
    apfdc : float
        The APFDC score
    """
    pred = pd.DataFrame(columns=['verdict', 'cost', 'score'])
    pred['verdict'] = y_true[:, 0]
    pred['cost'] = y_true[:, 1]
    pred['score'] = y_score
    pred.sort_values('score', ascending=False, inplace=True, ignore_index=True)

    if pred['score'].unique().shape[0] == 1 and pred.shape[0] > 1:
        return 0.5
    
    n = pred.shape[0]

    if n <= 1:
        return 1.0
    
    m = pred[pred['verdict'] > 0].shape[0]
    costs = pred['cost'].values.tolist()
    failed_costs = 0.0

    for tfi in pred[pred['verdict'] > 0].index:
        failed_costs += sum(costs[tfi:]) - (costs[tfi] / 2)
    
    apfdc = failed_costs / (sum(costs) * m)
    return apfdc