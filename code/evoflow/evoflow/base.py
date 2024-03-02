# -*- coding: utf-8 -*-

#    Rafael Barbudo Lunar, PhD Student
#    Knowledge and Discovery Systems (KDIS)
#    University of Cordoba, Spain

#    Edited by Ángel Fuentes Almoguera, Bachelor Student

import numpy as np
from os.path import join, exists
import pathlib
import random as rand
import shutil
import sys
import time

from deap import base, creator, tools

from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import VotingClassifier, VotingRegressor

from multiprocessing import cpu_count

import evoflow.extension.metrics
import sklearn.metrics
from evoflow.g3p.crossover import cx_multi
from evoflow.g3p.encoding import SyntaxTreeSchema, SyntaxTreePipeline
from evoflow.g3p.evaluation import evaluate
from evoflow.g3p.grammar import parse_pipe_grammar
from evoflow.g3p.mutation import mut_multi, mut_hps
from evoflow.g3p.support import DiverseElite
from evoflow.g3p.multiprocessing import EvaluationPool


class EvoFlow:
	"""
	Class responsible for optimizing the machine learning pipelines

	Parameters
	----------
	grammar: str
		Path of the file containing the grammar used to generate the pipelines

	pop_size: int, default=100
		Size of the population used in the evolutionary process.
		Must be greater than 0
	
	generations: int, default=10
		Number of generations used in the evolutionary process.
		Must be greater or equal than 0.
		If 0, only the initial population will be evaluated
	
	fitness: str or function, default=sklearn.metrics.balanced_accuracy_score
		Fitness function used to evaluate the pipelines.
		When using the function name, it will be searched first in the evoflow.extension.metrics module and then in the sklearn.metrics module.
		If the function is not found in any of the modules, the balanced_accuracy_score function from sklearn.metrics will be used by default
	
	nderiv: int, default=13
		Number of derivations used to generate the pipelines.
		Must be greater than 0
	
	crossover: str or function, default=evoflow.g3p.crossover.cx_multi
		Crossover function used to generate new individuals.
		When using the function name, it will be searched in the evoflow.g3p.crossover module
	
	mutation: str or function, default=evoflow.g3p.mutation.mut_multi
		Mutation function used to generate new individuals.
		When using the function name, it will be searched in the evoflow.g3p.mutation module

	mutation_elite: str or function, default=evoflow.g3p.mutation.mut_hps
		Mutation function used to generate new individuals from the elite.
		When using the function name, it will be searched in the evoflow.g3p.mutation module

	use_double_mutation: bool, default=False
		Indicates if the mutation_elite function should be used to generate new individuals from the elite.
		No crossover will be applied.
		If False, the mutation function will be used instead and the crossover function will be applied.
	
	cxpb: float, default=0.8
		Probability of applying the crossover operator.
		Must be in the range [0.0, 1.0]
	
	mutpb: float, default=0.2
		Probability of applying the mutation operator.
		Must be in the range [0.0, 1.0]
	
	elite_size: int, default=10
		Maximun number of pipelines to be stored in the elite.
		must be in the range [0, pop_size]
	
	timeout: int, default=3600
		Maximun time allowed for the evolutionary process.
		Must be greater than 0
	
	eval_timeout: int, default=360
		Maximun time allowed for the evaluation of a pipeline.
		It can be None, in which case the evaluation of the individuals will not be stopped, or greater than 0.
	
	seed: int, default=None
		Seed used to initialize the random number generator.
		If None, a random seed will be used
	
	outdir: str, default="results"
		Path of the directory where the results will be stored
	
	cv: int, default=5
		Number of folds used in the cross-validation process.
		Must be greater or equal than 2

	use_predict_proba: bool, default=False
		Indicates if the fitness function should use the predict_proba method instead of the predict method.
		Some fitness functions require probabilities instead of class labels. E.g. roc_auc_score

	maximization: bool, default=True
		Indicates if the fitness function should be maximized or minimized

	output_variables_ignore_fit: list, default=None
		List of output variables that should be ignored when fitting the pipeline.
		List should contain the indexes of the columns of the output variables.
		When using indexes, the indexes should be in the range [0, n_outputs-1].
		When using None or an empty list, all output variables will be used to fit the pipeline.
		Can be used to define cost-sensitive learning problems

	processes: int, default=1
		Number of processes used to evaluate the population.
		Must be greater than 0.
		If -1, the number of processes will be equal to the number of cores of the machine

	diversity_weight: float, default=0.0
		Weight of the diversity in the fitness function.
		Must be in the range [0.0, 1.0]. 
		If 0.0, the diversity will not be considered.
		If 1.0, the fitness will not be considered and the diversity will be maximized.
		Applicable if elite_size is greater than 1

	intergenerational_elite: int, default=1
		Number of individuals from the elite that will be included in the next generation.
		Must be in the range [0, elite_size].
		If 0, no individuals from the elite will be included in the next generation

	is_classification: bool, default=True
		Indicates if the problem is a classification problem or a regression problem

	early_stopping_threshold: float, default=0.0001
		Threshold used to determine if there is improvement in the elite average fitness.
		Must be greater or equal than 0.0

	early_stopping_generations: int, default=None
		Number of generations without improvement in the elite average fitness before stopping the evolutionary process.
		It can be None, in which case the evolutionary process will not stop based on generations without improvement, or greater than 0

	early_stopping_time: int, default=None
		Maximun time without improvement in the elite average fitness before stopping the evolutionary process.
		It can be None, in which case the evolutionary process will not stop based on time without improvement, or greater than 0
	"""
	def __init__(self, grammar, pop_size=100, generations=10,
				 fitness=balanced_accuracy_score, nderiv=13,
				 crossover=cx_multi, mutation=mut_multi,
				 mutation_elite=mut_hps, use_double_mutation=False,
				 cxpb=0.8, mutpb=0.2, elite_size=10, timeout=3600,
				 eval_timeout=360, seed=None, outdir="results", cv=5,
				 use_predict_proba=False, maximization=True,
				 output_variables_ignore_fit=None, processes=1,
				 diversity_weight=0.0, intergenerational_elite=1,
				 is_classification=True, early_stopping_threshold=0.0001,
				 early_stopping_generations=None, early_stopping_time=None):

		# Set the parameters for training
		if not isinstance(grammar, str):
			raise ValueError("grammar must be a string")
		self.grammar = grammar

		if not isinstance(pop_size, int) or pop_size <= 0:
			raise ValueError("pop_size must be an integer greater than 0")
		self.pop_size = pop_size
		
		if not isinstance(generations, int) or generations < 0:
			raise ValueError("generations must be an integer greater or equal than 0")
		self.generations = generations

		if not isinstance(fitness, (str, type(balanced_accuracy_score))):
			raise ValueError("fitness must be a string or a function")
		if isinstance(fitness, str):
			if hasattr(evoflow.extension.metrics, fitness):
				self.fitness = getattr(evoflow.extension.metrics, fitness)
			elif hasattr(sklearn.metrics, fitness):
				self.fitness = getattr(sklearn.metrics, fitness)
			else:
				print("WARNING: Fitness function not found. Using balanced_accuracy_score by default")
				self.fitness = balanced_accuracy_score
		else:
			self.fitness = fitness

		if not isinstance(nderiv, int) or nderiv <= 0:
			raise ValueError("nderiv must be an integer greater than 0")
		self.nderiv = nderiv		
		
		if not isinstance(crossover, (str, type(cx_multi))):
			raise ValueError("crossover must be a string or a function")
		if isinstance(crossover, str):
			if hasattr(evoflow.g3p.crossover, crossover):
				self.crossover = getattr(evoflow.g3p.crossover, crossover)
			else:
				print("WARNING: Crossover function not found. Using cx_multi by default")
				self.crossover = cx_multi
		else:
			self.crossover = crossover

		if not isinstance(mutation, (str, type(mut_multi))):
			raise ValueError("mutation must be a string or a function")
		if isinstance(mutation, str):
			if hasattr(evoflow.g3p.mutation, mutation):
				self.mutation = getattr(evoflow.g3p.mutation, mutation)
			else:
				print("WARNING: Mutation function not found. Using mut_multi by default")
				self.mutation = mut_multi
		else:
			self.mutation = mutation

		if not isinstance(mutation_elite, (str, type(mut_hps))):
			raise ValueError("mutation_elite must be a string or a function")
		if isinstance(mutation_elite, str):
			if hasattr(evoflow.g3p.mutation, mutation_elite):
				self.mutation_elite = getattr(evoflow.g3p.mutation, mutation_elite)
			else:
				print("WARNING: Elite mutation function not found. Using mut_hps by default")
				self.mutation_elite = mut_hps
		else:
			self.mutation_elite = mutation_elite

		if not isinstance(use_double_mutation, bool):
			raise ValueError("use_double_mutation must be a boolean")
		self.use_double_mutation = use_double_mutation

		if not isinstance(cxpb, float) or cxpb < 0.0 or cxpb > 1.0:
			raise ValueError("cxpb must be a float in the range [0.0, 1.0]")
		self.cxpb = cxpb
		
		if not isinstance(mutpb, float) or mutpb < 0.0 or mutpb > 1.0:
			raise ValueError("mutpb must be afloat in the range [0.0, 1.0]")
		self.mutpb = mutpb
		
		if not isinstance(elite_size, int) or elite_size < 0 or elite_size > self.pop_size:
			raise ValueError("elite_size must be an integer in the range [0, pop_size]")
		self.elite_size = elite_size
		
		if not isinstance(timeout, int) or timeout <= 0:
			raise ValueError("timeout must be an integer greater than 0")
		self.timeout = timeout
		
		if not isinstance(eval_timeout, (int, type(None))) or (isinstance(eval_timeout, int) and eval_timeout <= 0):
			raise ValueError("eval_timeout must be None or an integer greater than 0")
		self.eval_timeout = eval_timeout
		
		if not isinstance(seed, (int, type(None))):
			raise ValueError("seed must be None or an integer")
		self.seed = seed

		if not isinstance(outdir, str):
			raise ValueError("outdir must be a string")
		self.outdir = outdir

		if not isinstance(cv, int) or cv < 2:
			raise ValueError("cv must be an integer greater or equal than 2")
		self.cv = cv
		
		if not isinstance(use_predict_proba, bool):
			raise ValueError("use_predict_proba must be a boolean")
		self.use_predict_proba = use_predict_proba

		if not isinstance(maximization, bool):
			raise ValueError("maximization must be a boolean")
		self.maximization = maximization
		
		if not isinstance(output_variables_ignore_fit, (list, type(None))):
			raise ValueError("output_variables_ignore_fit must be None or a list")
		self.output_variables_ignore_fit = output_variables_ignore_fit

		if not isinstance(processes, int) or (processes <= 0 and processes != -1):
			raise ValueError("processes must be an integer greater than 0 or -1")
		self.processes = cpu_count() if processes == -1 else processes
		
		if not isinstance(diversity_weight, float) or diversity_weight < 0.0 or diversity_weight > 1.0:
			raise ValueError("diversity_weight must be a float in the range [0.0, 1.0]")
		self.diversity_weight = diversity_weight

		if not isinstance(intergenerational_elite, int) or intergenerational_elite < 0 or intergenerational_elite > self.elite_size:
			raise ValueError("intergenerational_elite must be an integer in the range [0, elite_size]")
		self.intergenerational_elite = intergenerational_elite
		
		if not isinstance(is_classification, bool):
			raise ValueError("is_classification must be a boolean")
		self.is_classification = is_classification

		if not isinstance(early_stopping_threshold, float) or early_stopping_threshold < 0.0:
			raise ValueError("early_stopping_threshold must be a float greater or equal than 0.0")
		self.early_stopping_threshold = early_stopping_threshold
		
		if not isinstance(early_stopping_generations, (int, type(None))) or (isinstance(early_stopping_generations, int) and early_stopping_generations <= 0):
			raise ValueError("early_stopping_generations must be None or an integer greater than 0")
		self.early_stopping_generations = early_stopping_generations
		
		if not isinstance(early_stopping_time, (int, type(None))) or (isinstance(early_stopping_time, int) and early_stopping_time <= 0):
			raise ValueError("early_stopping_time must be None or an integer greater than 0")
		self.early_stopping_time = early_stopping_time

		# Set the seed if it is None
		if self.seed is None:
			self.seed = np.random.randint(low=0, high=2**32)
		
		rand.seed(self.seed)
		np.random.seed(self.seed)

		# Create the logging file for the individual
		if exists(self.outdir):
			shutil.rmtree(self.outdir)
		
		pathlib.Path(join(self.outdir)).mkdir(parents=True, exist_ok=True)

		with open(join(self.outdir, "config.txt"), "a") as log:
			log.write("Grammar: "+ self.grammar + "\n")
			log.write("Generations: "+ str(self.generations) + "\n")
			log.write("Population size: " + str(self.pop_size) + "\n")
			log.write("Is classification: " + str(self.is_classification) + "\n")
			log.write("Derivations: " + str(self.nderiv) + "\n")
			log.write("Fitness: " + self.fitness.__name__ + "\n")
			log.write("Maximization: " + str(self.maximization) + "\n")
			log.write("Use predict_proba: " + str(self.use_predict_proba) + "\n")
			log.write("Number of derivations: " + str(self.nderiv) + "\n")
			log.write("Crossover probability: " + str(self.cxpb) + "\n")
			log.write("Mutation probability: " + str(self.mutpb) + "\n")
			log.write("Timeout: " + str(self.timeout) + "\n")
			log.write("Evaluation time: " + str(self.eval_timeout) + "\n")
			log.write("Elite size: " + str(self.elite_size) + "\n")
			log.write("Intergenerational elite: " + str(self.intergenerational_elite) + "\n")
			log.write("Diversity weight: " + str(self.diversity_weight) + "\n")
			log.write("Cross-validation: " + str(self.cv) + "\n")
			log.write("Use double mutation: " + str(self.use_double_mutation) + "\n")
			log.write("Crossover: " + self.crossover.__name__ + "\n")
			log.write("Mutation: " + self.mutation.__name__ + "\n")
			log.write("Mutation elite: " + self.mutation_elite.__name__ + "\n")
			log.write("Early stopping threshold: " + str(self.early_stopping_threshold) + "\n")
			log.write("Early stopping generations: " + str(self.early_stopping_generations) + "\n")
			log.write("Early stopping time: " + str(self.early_stopping_time) + "\n")
			log.write("Seed: " + str(self.seed))
		
		with open(join(self.outdir, "individuals.tsv"), "a") as f:
			f.write("pipeline\tfitness\tfit_time\n")

		# Configure the G3P logging
		# Statistics for the fitness and size of the individuals in the population
		stat_fit = tools.Statistics(lambda ind: ind.fitness.values)
		stat_size = tools.Statistics(lambda ind: len(str(ind).split(";")))
		stats = tools.MultiStatistics(fitness=stat_fit, size=stat_size)
		stats.register("min     ", np.nanmin)  # For better visualization
		stats.register("max     ", np.nanmax)
		stats.register("avg     ", np.nanmean)
		stats.register("std     ", np.nanstd)
		self.stats = stats

		# Statistics for the fitness and size of the individuals in the elite
		stat_fit_elite = tools.Statistics(lambda ind: ind.fitness.values)
		stat_size_elite = tools.Statistics(lambda ind: len(str(ind).split(";")))
		stats_elite = tools.MultiStatistics(fitness_elite=stat_fit_elite, size_elite=stat_size_elite)
		stats_elite.register("min     ", np.min)  # For better visualization
		stats_elite.register("max     ", np.max)
		stats_elite.register("avg     ", np.mean)
		stats_elite.register("std     ", np.std)
		self.stats_elite = stats_elite

		# Load the grammar
		root, terms, non_terms, self.pset, terms_families = parse_pipe_grammar(grammar, seed)
		self.schema = SyntaxTreeSchema(nderiv, root, terms, non_terms, self.pset, terms_families)

		# Configure the creator (from DEAP)
		creator.create("FitnessMax", base.Fitness, weights=(1.0,) if maximization else (-1.0,))
		creator.create("Individual", SyntaxTreePipeline, fitness=creator.FitnessMax)

		# Configure the toolbox (from DEAP)
		toolbox = base.Toolbox()
		toolbox.register("expr", self.schema.createSyntaxTree)
		toolbox.register("ind", tools.initIterate, creator.Individual, toolbox.expr)
		toolbox.register("population", tools.initRepeat, list, toolbox.ind)
		toolbox.register("select", tools.selTournament, tournsize=2)
		toolbox.register("mate", self.crossover, schema=self.schema)
		toolbox.register("mutate", self.mutation, schema=self.schema)
		toolbox.register("mutate_elite", self.mutation_elite, schema=self.schema)
		self.toolbox = toolbox

	def fit(self, X, y):
		"""
		Fit the evolutionary process

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			The training input samples, where `n_samples` is the number of samples
			and `n_features` is the number of features.

		y : array-like of shape (n_samples,)
			The target values.
		"""
		# Create the evolutionary process
		start_time = time.time()
		self._evolve(X, y)

		# Save the final execution time
		exec_time = str(time.time() - start_time)
		print("--- " + exec_time + " seconds ---")

		with open(join(self.outdir, "evolution.txt"), "a") as log:
			log.write("--- " + exec_time + " sec ---\n")

		# Stop the execution if there are not individuals in the elite
		if self.elite is None or len(self.elite) == 0:
			print('ERR: The elite is empty.')
			sys.exit()

		# Delete the output variables that are not used in the fit
		if self.output_variables_ignore_fit is not None and len(self.output_variables_ignore_fit) > 0:
			y = np.delete(y, self.output_variables_ignore_fit, axis=1)

		# Build ensemble for later use
		for ind in self.elite:
			if not hasattr(ind, 'pipeline') or ind.pipeline is None:
				ind.create_sklearn_pipeline(self.pset)

		best_ind = self.elite.best_ind()
		best_ind.pipeline.fit(X, y)
		best_fitness = best_ind.fitness.values[0]
		weights = [ind.fitness.values[0]/best_fitness for ind in self.elite] if self.maximization else [best_fitness/ind.fitness.values[0] for ind in self.elite]
		estimators = [(str(idx), est.pipeline) for idx, est in enumerate(self.elite)]
		self.vt = VotingClassifier(estimators, voting="soft", weights=weights) if self.is_classification else VotingRegressor(estimators, weights=weights)
		self.vt.fit(X, y)

	def _reset_ind(self, ind):

		del ind.fitness.values
		del ind.pipeline

	def _apply_operators_cx_mut(self, population):

		# Select the parents
		offspring = self.toolbox.select(population, self.pop_size)
		# Clone the parents
		offspring = [self.toolbox.clone(ind) for ind in offspring]

		# Apply the crossover
		for i in range(1, len(offspring), 2):
			if rand.random() < self.cxpb:
				offspring[i - 1], offspring[i], modified = self.toolbox.mate(offspring[i-1], offspring[i])
				
				if modified:
					self._reset_ind(offspring[i-1])
					self._reset_ind(offspring[i])

		# Apply the mutation
		for i, _ in enumerate(offspring):
			if rand.random() < self.mutpb:
				offspring[i], modified = self.toolbox.mutate(offspring[i])

				# Properties may do not exist if they have been deleted during crossover
				if modified and hasattr(offspring[i], "pipeline"):
					self._reset_ind(offspring[i])

		return offspring
	
	def _apply_operators_double_mut(self, population):

		# Clone the parents
		offspring = [self.toolbox.clone(ind) for ind in population]

		# Apply the mutation
		for i, _ in enumerate(offspring):
			if rand.random() < self.mutpb:
				mutate = self.toolbox.mutate_elite if offspring[i] in self.elite else self.toolbox.mutate
				offspring[i], modified = mutate(offspring[i])

				# Properties may do not exist if they have been deleted during crossover
				if modified and hasattr(offspring[i], "pipeline"):
					self._reset_ind(offspring[i])

		return offspring

	def _evolve(self, X, y):

		# To control the timeout
		start = time.time()

		# Configure the rest of the evolutionary algorithm
		self.toolbox.register("evaluate", evaluate, timeout=self.timeout, eval_timeout=self.eval_timeout, metric=self.fitness, is_classification=self.is_classification, use_predict_proba=self.use_predict_proba,
							  seed=self.seed, X=X, y=y, outdir=self.outdir, cv=self.cv, pset=self.pset, start=start, output_variables_ignore_fit=self.output_variables_ignore_fit)

		# Configure additional logging
		logbook = tools.Logbook()
		logbook.header = ['gen', 'nevals'] + (self.stats.fields if self.stats else []) + (self.stats_elite.fields if self.stats_elite else [])

		for category in self.stats.fields:
			logbook.chapters[category].header = self.stats[category].fields

		for category in self.stats_elite.fields:
			logbook.chapters[category].header = self.stats_elite[category].fields

		# Elite with the best individuals
		self.elite = DiverseElite(self.elite_size, div_weight=self.diversity_weight)

		# Create the initial population
		population = self.toolbox.population(n=self.pop_size)

		# Early stopping variables
		if self.early_stopping_generations is not None:
			generations_without_improvement = 0

		if self.early_stopping_time is not None:
			time_without_improvement = time.time()

		with EvaluationPool(processes=self.processes) as pool:
			results = pool.map(self.toolbox.evaluate, population)

		for ind, result in zip(population, results):
			ind.fitness.values, ind.prediction, ind.runtime = result
		
		# Update the elite
		pop_valid = [ind for ind in population if not np.isnan(ind.fitness.values[0])]
		self.elite.update(pop_valid)

		# Append the current generation statistics to the logbook
		record = self.stats.compile(population) if self.stats else {}
		record_elite = self.stats_elite.compile(self.elite) if self.stats_elite else {}
		logbook.record(gen=0, nevals=len(population), **record, **record_elite)
		report = logbook.stream
		print(report)

		with open(join(self.outdir, "evolution.txt"), "a") as log:
			log.write(report + "\n")

		if self.early_stopping_generations is not None or self.early_stopping_time is not None:
			elite_avg_fitness = np.mean([ind.fitness.values[0] for ind in self.elite])

		# Specify which operators will be applied
		apply_operators = self._apply_operators_double_mut if self.use_double_mutation else self._apply_operators_cx_mut

		# Begin the generational process
		for gen in range(1, self.generations + 1):
			# Stop the evaluation if timeout has been reached
			if (time.time() - start) > self.timeout:
				return

			# Apply the genetic operators
			offspring = apply_operators(population)

			# Evaluate the offspring
			invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

			with EvaluationPool(processes=self.processes) as pool:
				results = pool.map(self.toolbox.evaluate, invalid_ind)

			for ind, result in zip(invalid_ind, results):
				ind.fitness.values, ind.prediction, ind.runtime = result
			
			# Update the elite
			off_valid = [ind for ind in offspring if not np.isnan(ind.fitness.values[0])]
			self.elite.update(off_valid)
			
			# Put invalid individuals at the beginning of the list			
			offspring = sorted(offspring, key=lambda ind: not np.isnan(ind.fitness.values[0]))
			idx = next((i for i, ind in enumerate(offspring) if not np.isnan(ind.fitness.values[0])), None)
			# Sort the valid individuals considering if the problem is a minimization or maximization problem
			offspring = offspring[:idx] + sorted(offspring[idx:], key=lambda ind: ind.fitness.values[0], reverse=not self.maximization)
			idx = 0

			# Replace the current population ensuring that the best individuals are kept
			for i in range(min(len(self.elite), self.intergenerational_elite)):
				if self.elite[i] not in offspring:
					offspring[idx] = self.elite[i]
					idx += 1
			
			population[:] = offspring

			# Append the current generation statistics to the logbook
			record = self.stats.compile(population) if self.stats else {}
			record_elite = self.stats_elite.compile(self.elite) if self.stats_elite else {}
			logbook.record(gen=gen, nevals=len(invalid_ind), **record, **record_elite)
			report = logbook.stream
			print(report)

			with open(join(self.outdir, "evolution.txt"), "a") as log:
				log.write(report + "\n")

			# Check if should early stop
			if self.early_stopping_generations is not None or self.early_stopping_time is not None:
				current_elite_avg_fitness = np.mean([ind.fitness.values[0] for ind in self.elite])

				if (current_elite_avg_fitness - elite_avg_fitness) < self.early_stopping_threshold:
					# Check if should early stop due to no improvement in the last generations
					if self.early_stopping_generations is not None:
						generations_without_improvement += 1

						if generations_without_improvement >= self.early_stopping_generations:
							print(f'Early stopping due to no improvement in the last {self.early_stopping_generations} generations.')
							return

					# Check if should early stop due to no improvement in the last time
					if self.early_stopping_time is not None:
						diff = time.time() - time_without_improvement

						if diff >= self.early_stopping_time:
							print(f'Early stopping due to no improvement in the last {diff} seconds.')
							return
				else:
					generations_without_improvement = 0
					time_without_improvement = time.time()
					elite_avg_fitness = current_elite_avg_fitness
	
	def predict(self, X):
		"""
		Predict class labels for X.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			The input samples.

		Returns
		-------
		y : array-like of shape (n_samples,)
			The predicted classes.
		"""
		return self.vt.predict(X)

	def predict_proba(self, X):
		"""
		Compute probabilities of possible outcomes for samples in X.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			The input samples.

		Returns
		-------
		y : array-like of shape (n_samples, n_classes)
			The class probabilities for each class per sample.
		"""
		return self.vt.predict_proba(X)
