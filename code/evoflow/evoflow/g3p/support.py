# -*- coding: utf-8 -*-

#    Rafael Barbudo Lunar, PhD Student
#    Knowledge and Discovery Systems (KDIS)
#    University of Cordoba, Spain


from bisect import bisect_right
from deap.tools.support import HallOfFame
from copy import deepcopy
import numpy as np
from operator import eq


class DiverseElite(HallOfFame):
	"""
	Class responsible for maintaining the elite of the population and
	ensuring that is composed of diverse individuals. The elite is sorted
	based on the weighted sum of the diversity and the fitness of the individual.

	Parameters
	----------
	maxsize : int
		The maximum size of the elite
	div_weight : float
		How much weight to give to the diversity of the individual.
		0.0 means that only the fitness is considered, 1.0 means that only the diversity is considered.
		If the elite size is 1, the diversity is not considered
	similar : function, default: operator.eq
		The function used to compare individuals
	"""

	def __init__(self, maxsize, div_weight, similar=eq):
		HallOfFame.__init__(self, maxsize, similar)
		self.div_weight = div_weight if maxsize > 1 else 0.0
		self.diversities = []


	def insert(self, item):
		"""
		Individuals are inserted ensuring the elite is sorted
		against their diversity value
		"""
		item = deepcopy(item)
		i = bisect_right(self.keys, item.diversity)
		self.items.insert(len(self) - i, item)
		self.keys.insert(i, item.diversity)


	def update(self, population):
		"""
		Update the elite with the individuals from the population.
		
		Parameters
		----------
		population : list
			The population used to update the elite
		"""
		# first time adding individuals to the elite
		if len(self) == 0 and self.maxsize != 0:
			# we cannot compute a diversity measure yet, so
			# we used the update from the parent class
			for ind in population:
				ind.diversity = ind.fitness.values[0]*ind.fitness.weights[0]

			super(DiverseElite, self).update(population)

			# now we can calculate the diversity and sort
			# the individuals
			self.update_diversities()
			self.sort()
		else:
			for ind in population:
				# compute the diversity of the individual
				div = self.compute_diversity(ind)
				ind.diversity = self.div_weight*div + (1-self.div_weight)*ind.fitness.values[0]*ind.fitness.weights[0]
				# the diversity should be better than the worst from the elite
				if ind.diversity > self[-1].diversity or len(self) < self.maxsize:
					for hofer in self:
						if self.similar(ind, hofer):
							break
					else:
						if len(self) >= self.maxsize:
							self.remove(-1)
						self.insert(ind)


	def sort(self):
		"""
		Sort the elite based on the diversity
		"""
		# the insert function ensure elite is sorted
		aux_items = deepcopy(self.items)
		self.clear()
		self.insert(aux_items[0])
		for item in aux_items[1:]:
			self.insert(item)


	def update_diversities(self):
		"""
		Compute the diversity of each individual from the elite
		"""
		for ind in self:
			div = self.compute_diversity(ind)
			ind.diversity = self.div_weight*div + (1-self.div_weight)*ind.fitness.values[0]*ind.fitness.weights[0]


	def compute_diversity(self, ind):
		"""
		Compute the diversity against the elite

		Parameters
		----------
		ind : evoflow.g3p.encoding.SyntaxTreePipeline
			The individual to compute the diversity against the elite
		"""
		diversity = 0.0
		for ind_elite in self:
			diff = np.sum(ind.prediction != ind_elite.prediction)
			diff /= len(ind.prediction)
			diversity += diff
		return (diversity/len(self))


	def best_ind(self):
		"""
		Get the best individual based only in its fitness
		"""
		maxfit = self[0].fitness.values[0]
		maxidx = 0
		for idx, ind in enumerate(self[1:]):
			if ind.fitness.values[0] > maxfit:
				maxfit = ind.fitness.values[0]
				maxidx = idx + 1
		return deepcopy(self[maxidx])