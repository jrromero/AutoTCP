# -*- coding: utf-8 -*-

#    Rafael Barbudo Lunar, PhD Student
#    Knowledge and Discovery Systems (KDIS)
#    University of Cordoba, Spain

import numpy as np
from multiprocessing import Process, Manager
from os.path import join
import warnings
from time import time

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def evaluate(ind, timeout, eval_timeout, metric, is_classification, use_predict_proba, X, y, cv, outdir, seed, pset, start, output_variables_ignore_fit=None):

	if time() - start > timeout:
		with open(join(outdir, "individuals.tsv"), 'a') as f:
			f.write(str(ind) + '\t' + 'timeout_err' + '\t' + 'timeout_err' + '\n')
		
		return (np.nan,), None, None

	start_time = time()
	fitness, predictions = _evaluate_cv(ind, metric, is_classification, use_predict_proba, X, y, cv, eval_timeout, seed, pset, output_variables_ignore_fit)
	elapsed = time() - start_time

	with open(join(outdir, "individuals.tsv"), 'a') as f:

		if isinstance(fitness, str):
			f.write(str(ind) + '\t' + fitness + '\t' + fitness + '\n')
			return (np.nan,), None, None
		else:
			#f.write(str(ind) + '\t' + str(fitness) + '\t' + str(elapsed) + '\t' + ",".join(str(v) for v in predictions.tolist()) + '\n')
			f.write(str(ind) + '\t' + str(fitness) + '\t' + str(elapsed) + '\n')
			return (fitness,), predictions, elapsed

'''
def evaluate(ind, timeout, return_dict, metric, use_predict_proba, X, y, cv, outdir, seed, pset):
	start_time = time.time()
	fitness, predictions = _evaluate_cv(ind, metric, use_predict_proba, X, y, cv, timeout, seed, pset)
	elapsed = time.time() - start_time

	with open(join(outdir, "individuals.tsv"), 'a') as f:

		if isinstance(fitness, str):
			f.write(str(ind) + '\t' + fitness + '\t' + fitness + '\n')
			return_dict[getpid()] = (0.0,), None, None
		else:
			#f.write(str(ind) + '\t' + str(fitness) + '\t' + str(elapsed) + '\t' + ",".join(str(v) for v in predictions.tolist()) + '\n')
			f.write(str(ind) + '\t' + str(fitness) + '\t' + str(elapsed) + '\n')
			return_dict[getpid()] = (fitness,), predictions, elapsed
'''

def _evaluate_cv(ind, metric, is_classification, use_predict_proba, X, y, cv, timeout, seed, pset, output_variables_ignore_fit=None, memory_limit=3072):

	# Check for pipelines with duplicated operators
	pipe_str = str(ind).split(';')
	ops_str = [op[:op.index('(')] for op in pipe_str]
	
	if len(ops_str) != len(set(ops_str)):
		return "invalid_ind (duplicated operator)", None

	# Generate the folds
	k_folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
	
	# Check whether there is a timeout for an evaluation
	if timeout is not None:
		fold_timeout = int(timeout/cv)

	# Train and test the model for each fold
	fitneses = []
	predictions = np.empty(0)
	mng = Manager()

	# Delete the output variables that are not used in the fit
	if output_variables_ignore_fit is not None and len(output_variables_ignore_fit) > 0:
		cost = y[:, output_variables_ignore_fit]
		y = np.delete(y, output_variables_ignore_fit, axis=1)

	for train_index, test_index in k_folds.split(X, y):

		# Get the train and test sets
		x_train, x_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		if timeout is None:
			'''
			try:
				if len(ind.pipeline) == 1: # only a classifier
					ind.pipeline.fit(x_train, y_train)
				else:
					# apply only the preprocessing
					for step in ind.pipeline[:-1]:

						if getattr(step, "fit_transform", None) is not None:
							x_train = step.fit_transform(x_train, y_train)
						elif getattr(step, "fit_resample", None) is not None:
							x_train, y_train = step.fit_resample(x_train, y_train)

					# raise an exception if the new dataset is too large
					if x_train.nbytes/1024**2 > memory_limit:
						raise MemoryError

					ind.pipeline[-1].fit(x_train, y_train)

				y_pred = ind.pipeline.predict(x_test) if not use_predict_proba else ind.pipeline.predict_proba(x_test)

			except ValueError as v:
				return "invalid_ind (" + str(v) + ")", None
			except MemoryError as m:
				return "mem_err (" + str(m) + ")", None
			'''
			return_dict = mng.dict()
			p = Process(target=_fit_predict, args=(ind, pset, is_classification, use_predict_proba, x_train, y_train, x_test, return_dict, memory_limit))
			p.start()
			p.join()

			if isinstance(return_dict['y_pred'], str):
				return return_dict['y_pred'], None
			
			y_pred = return_dict['y_pred']
		else:
			# Create a new process to control the timeout
			return_dict = mng.dict()
			p = Process(target=_fit_predict, args=(ind, pset, is_classification, use_predict_proba, x_train, y_train, x_test, return_dict, memory_limit))
			p.start()
			p.join(fold_timeout)

			if p.is_alive():
				p.terminate()
				p.join()
				return "eval_timeout_err", None
			elif isinstance(return_dict['y_pred'], str):
				return return_dict['y_pred'], None

			y_pred = return_dict['y_pred']

		# Store the fitness of the fold
		try:
			y_pred = y_pred[:, -1] if y_pred.ndim == 2 and y_pred.shape[1] <= 2 else y_pred
			predictions = np.concatenate([predictions, y_pred])

			if output_variables_ignore_fit is not None and len(output_variables_ignore_fit) > 0:
				y_test = np.insert(y_test, output_variables_ignore_fit, cost[test_index], axis=1)

			fitness = metric(y_test, y_pred)
			fitneses.append(fitness)
		except ValueError as v:
			return "metric_err (" + str(v) + ")", None
		except Exception as e:
			return "metric_err (" + str(e) + ")", None

	# Return the evaluated individual
	fitness = sum(fitneses)/len(fitneses)
	return fitness, predictions


def _fit_predict(ind, pset, is_classification, use_predict_proba, x_train, y_train, x_test, ret_dict, memory_limit):
	if not hasattr(ind, 'pipeline') or ind.pipeline is None:
		ind.create_sklearn_pipeline(pset)

	try:
		if len(ind.pipeline) == 1:
			ind.pipeline.fit(x_train, y_train)
		else:
			for step in ind.pipeline[:-1]:

				if getattr(step, "fit_transform", None) is not None:
					x_train = step.fit_transform(x_train, y_train)
				elif getattr(step, "fit_resample", None) is not None:
					x_train, y_train = step.fit_resample(x_train, y_train)

			if x_train.nbytes/1024**2 > memory_limit:
				raise MemoryError

			ind.pipeline[-1].fit(x_train, y_train)

		#ret_dict['y_pred'] = ind.pipeline.predict(x_test) if not use_predict_proba else ind.pipeline.predict_proba(x_test)
		ret_dict['y_pred'] = ind.pipeline.predict_proba(x_test) if is_classification and use_predict_proba else ind.pipeline.predict(x_test)

	except ValueError as v:
		ret_dict['y_pred'] = "invalid_ind (" + str(v) + ")"
	except IndexError as i:
		ret_dict['y_pred'] = "invalid_ind (" + str(i) + ")"
	except MemoryError as m:
		ret_dict['y_pred'] = "mem_err (" + str(m) + ")"
	except np.linalg.LinAlgError as l:
		ret_dict['y_pred'] = "invalid_ind (" + str(l) + ")"
	except Exception as e:
		ret_dict['y_pred'] = "invalid_ind (" + str(e) + ")"