# -*- coding: utf-8 -*-

#    Rafael Barbudo Lunar, PhD Student
#    Knowledge and Discovery Systems (KDIS)
#    University of Cordoba, Spain

import copy
import numpy as np
import random
import hashlib
import sys

from inspect import isclass
from deap import gp

from evoflow.utils import is_number

from imblearn.pipeline import Pipeline

#######################################
# G3P Data structure                  #
#######################################


class SyntaxTree(list):
	"""
	Gramatically based genetic programming tree.

	Tree specifically formatted for optimization of G3P operations. The tree
	is represented with a list where the nodes (terminals and non-terminals)
	are appended in a depth-first order. The nodes appended to the tree are
	required to have an attribute *arity* which defines the arity of the
	primitive. An arity of 0 is expected from terminals nodes.
	"""

	def __init__(self, content):
		list.__init__(self, content)

	def __deepcopy__(self, memo):
		new = self.__class__(self)
		new.__dict__.update(copy.deepcopy(self.__dict__, memo))
		return new

	def __str__(self):
		"""Return the syntax tree in a human readable string."""
		return ' '.join([elem.__str__() for elem in self])

	def primitive_tree(self):
		"""
		Convert the SyntaxTree into a PrimitiveTree, which can
		be directly evaluated.

		:returns: PrimitiveTree containing the terminals of the SyntaxTree.
		"""
		ptree = gp.PrimitiveTree([])
		for elem in self:
			# Only terminals are considered. These terminals refers to both
			# terminals and arguments in the GP vocabulary
			if isinstance(elem, TerminalNode):
				ptree.append(elem.code)
		return ptree

	def search_subtree(self, begin):
		"""
		Return a slice object that corresponds to the
		range of values that defines the subtree which has the
		element with index *begin* as its root.
		"""
		end = begin + 1
		total = self[begin].arity()
		while total > 0:
			total += self[end].arity() - 1
			end += 1
		return slice(begin, end)


class TerminalNode:
	"""
	Terminal node of a SyntaxTree. It correspond to both primitives and
	arguments in GP. Terminal nodes have 0 arity and include the code being
	executed when the SyntaxTree is evaluated.
	"""
	__slots__ = ('symbol', 'code', 'family')

	def __init__(self, symbol, code, family=None):
		self.symbol = symbol
		self.code = code
		self.family = family

	def __str__(self):
		return self.symbol

	def arity(self):
		"""Return the arity of the terminal node, i.e. 0"""
		return 0


class NonTerminalNode:
	"""
	Non-terminal node of a SyntaxTree. Each non-terminal node correspond
	to a production rule of a grammar. Thus, each node has a symbol
	(production rule left-sided) and the production itself
	(production rule right-sided)

	:Example:

	>>> <example> :: <a> <b> <c>
	self.symbol = example
	self.production = "a;b;c"
	self.prodList = ["a", "b", "c"]
	"""
	__slots__ = ('symbol', 'production', 'prodList')

	def __init__(self, symbol, production):
		self.symbol = symbol
		# Python does not allow to use a list as the key of a dictionary
		self.production = production
		self.prodList = production.split(';')

	def __str__(self):
		return self.symbol

	def arity(self):
		return len(self.prodList)


#######################################
# G3P Program generation functions    #
#######################################

class SyntaxTreeSchema:
	"""
	Schema used to guarantee valid individuals are created. It constains the
	set of terminals and non-terminals that can be used to construct a
	SyntaxTree, among others. It also ensure the construction of trees with a
	bounded-size
	"""
	__slots__ = ('pset', 'terminals', 'nonTerminals', 'maxDerivSize',
				 'minDerivSize', 'root', 'terminals_map',
				 'nonTerminals_map', 'terms_families', 'cardinality_map')

	def __init__(self, maxDerivSize, root, terminals,
				 nonTerminals, pset, terms_families):
		# Initialize the variables of the schema
		self.pset = pset
		self.terminals = terminals
		self.nonTerminals = nonTerminals
		self.maxDerivSize = maxDerivSize
		self.terms_families = terms_families
		self.minDerivSize = -1
		self.root = root
		self.terminals_map = {}
		self.nonTerminals_map = {}
		self.cardinality_map = {}
		# Configure the schema
		self.configure()

	def setTerminalsDic(self):
		"""Build and set the terminals dictionary."""
		self.terminals_map = {}
		for terminal in self.terminals:
			self.terminals_map[terminal.symbol] = terminal

	def setNonTerminalsDic(self):
		"""Build and set the non terminals dictionary."""
		# Used to classify symbols
		aux_map = {}
		# Classify non-term symbols
		for node in self.nonTerminals:
			symbol = node.symbol
			if symbol in aux_map:
				aux_map.get(symbol).append(node)
			else:
				aux_map[symbol] = [node]

		# Create non-term symbols map
		self.nonTerminals_map = {}
		for symbol in aux_map:
			# Put array in non terminals map
			self.nonTerminals_map[symbol] = aux_map.get(symbol)

	def setCardinalityDic(self):
		"""
		Build and set the cardinality dictionary. This dictionary contains
		cardinality of all production rules (from cero to max number of
		derivations)
		"""
		# Cardinality map
		self.cardinality_map = {}
		for nonTerm in self.nonTerminals:
			# Allocate space for cardinalities array
			list1 = [-1] * (1+self.maxDerivSize)
			# Put array in cardinality map
			self.cardinality_map[nonTerm.production] = list1

	def setMinDerivSize(self):
		"""Calculate and set the minimum number of derivations."""
		for i in range(self.maxDerivSize+1):
			if self.symbolCardinality(self.root, i) != 0:
				self.minDerivSize = i
				break

	def createSyntaxTree(self):
		"""
		Create a new syntax tree of a random size in range
		(minDerivSize, maxDerivSize)

		:returns: SyntaxTree conformant with the grammar defined.
		"""
		# Create resulting tree
		stree = SyntaxTree([])
		# Randomly determine the number of derivarion
		nOfDer = random.randint(self.minDerivSize, self.maxDerivSize)
		# Fill result branch
		self.fillTreeBranch(stree, self.root, nOfDer)
		# Return resulting tree
		return stree

	def fillTreeBranch(self, tree, symbol, nOfDer):
		"""
		Fill a SyntaxTree using the symbol and the allowed number of
		derivations

		:param symbol: The new symbol (terminal or non-terminal) to add
		:param nOfDer: The number of derivations
		"""
		if symbol in self.terminals_map:
			term = self.terminals_map.get(symbol)
			if isclass(term.code):
				tree.append(TerminalNode(symbol, term.code()))
			else:
				tree.append(TerminalNode(symbol, term.code))
		else:
			# Select a production rule
			selProd = self.selectProduction(symbol, nOfDer)
			# Expand the non terminal node
			self.expandNonTerminal(tree, symbol, nOfDer, selProd)

	def expandNonTerminal(self, tree, symbol, nOfDer, selProd):
		if selProd is not None:
			# Add this node
			tree.append(selProd)
			# Select a partition for this production rule
			selPart = self.selectPartition(selProd.prodList, nOfDer-1)
			# Apply partition, expanding production symbols
			selProdSize = len(selPart)

			for i in range(selProdSize):
				self.fillTreeBranch(tree, selProd.prodList[i], selPart[i])
		else:
			self.fillTreeBranch(tree, symbol, nOfDer-1)

	def selectProduction(self, symbol, nOfDer, selProd=None):
		"""
		Select a production rule for a symbol of the grammar, given the number
		of derivations available.

		:param symbol: Symbol to expand
		:param nOfDer: Number of derivations available
		:param selProd: The production might be selected beforehand. Then the 
		 method is called to compute cardinalities

		:returns: A production rule for the given symbol or None if this symbol
		cannot be expanded using exactly such number of derivations
		"""

		# Get the productions
		if selProd is None:
			prodRules = self.nonTerminals_map.get(symbol)
		else:
			prodRules = [selProd]

		# Number of productions
		nOfProdRules = len(prodRules)
		# Create productions roulette
		roulette = [0] * nOfProdRules

		# Fill roulette
		for i in range(nOfProdRules):
			cardinalities = self.cardinality_map.get(prodRules[i].production)
			# If this cardinality is not calculated, it will be calculated
			if cardinalities[nOfDer-1] == -1:
				cardinalities[nOfDer-1] = self.pRuleDerivCardinality(
					prodRules[i].prodList, nOfDer-1)
				self.cardinality_map[prodRules[i].production] = cardinalities

			roulette[i] = cardinalities[nOfDer-1]
			if i != 0:
				roulette[i] += roulette[i-1]

		# Choose a production at random
		rand_val = roulette[nOfProdRules-1] * random.uniform(0, 1)

		return next((prodRule for index, prodRule in enumerate(prodRules)
					 if rand_val < roulette[index]), None)

	def selectPartition(self, prodRule, nOfDer):
		"""
		Select a partition to expand a symbol using a production rule.

		:param prodRule: Production rule to expand
		:param nOfDer: Number of derivations available

		:returns: A partition
		"""
		# Obtain all partitions for this production rule
		partitions = self.partitions(nOfDer, len(prodRule))
		# Number of partitions
		nOfPart = len(partitions)
		# Create partitions roulette
		roulette = [0] * nOfPart

		# Set roulette values
		for i in range(nOfPart):
			roulette[i] = self.pRulePartCardinality(prodRule, partitions[i])
			if i != 0:
				roulette[i] = roulette[i] + roulette[i-1]

		# Choose a production at random
		rand_val = roulette[nOfPart-1] * random.uniform(0, 1)

		for i in range(nOfPart):
			if rand_val < roulette[i]:
				return partitions[i]

		# This point shouldn't be reached
		return None

	def symbolCardinality(self, symbol, nOfDer):
		"""
		Cardinality of a grammar symbol for the given number of derivs

		:param symbol: The grammar symbol (terminal or non-terminal)
		:param nOfDer: Number of derivations allowed

		:returns: Cardinality of the symbol
		"""
		if symbol in self.terminals_map:
			if nOfDer == 0:
				return 1
			return 0

		result = 0
		prodRules = self.nonTerminals_map.get(symbol)
		for pRule in prodRules:
			cardinalities = self.cardinality_map.get(pRule.production)
			if nOfDer <= 0:
				result = result + self.pRuleDerivCardinality(
					pRule.prodList, nOfDer-1)
			else:
				# If this cardinality is not calculated, calculate it
				if cardinalities[nOfDer-1] == -1:
					cardinalities[nOfDer-1] = self.pRuleDerivCardinality(
						pRule.prodList, nOfDer-1)
					self.cardinality_map[pRule.production] = cardinalities

				result = result + cardinalities[nOfDer-1]
		return result

	def pRuleDerivCardinality(self, pRule, nOfDer):
		"""
		Cardinality of a production rule for the given number of derivations.

		:param pRule: Production rule
		:param nOfDer: Number of derivations allowed

		:returns: Cardinality of the production rule
		"""
		# Resulting cardinality
		result = 0
		# Obtain all partitions
		partitions = self.partitions(nOfDer, len(pRule))
		# For all partitions of nOfDer...
		for partition in partitions:
			result += self.pRulePartCardinality(pRule, partition)
		# Return result
		return result

	def pRulePartCardinality(self, prodRule, partition):
		"""
		Cardinality of a production rule for the given partition.

		:param pRule: Production rule
		:param nOfDer: The given partition

		:returns: Cardinality of the production rule for the partition
		"""
		result = 1

		for prod, part in zip(prodRule, partition):
			factor = self.symbolCardinality(prod, part)
			if factor == 0:
				return 0
			result *= factor

		return result

	def partitions(self, total, dimension):
		result = []

		if dimension == 1:
			result.append([total])
		else:
			for i in range(total+1):
				pi = self.partitions(total-i, dimension-1)
				result += self.insertBefore(i, pi)

		return result

	def insertBefore(self, previous, strings):

		result = []
		for string in strings:
			tmp = [previous]
			for i in string:
				tmp.append(i)
			result.append(tmp)

		return result

	def configure(self):
		"""
		Configure the different dictionaries and set the minimum
		derivation size given the set of terminals and non terminals
		"""
		self.setTerminalsDic()
		self.setNonTerminalsDic()
		self.setCardinalityDic()
		self.setMinDerivSize()

	def minDerivations(self, symbol):
		"""
		Compute the minimum number of derivations of a given node
		"""
		for i in range(self.maxDerivSize):
			if self.symbolCardinality(symbol, i) != 0:
				return i

		# This point should never be reached
		return -1


#######################################
# Specific purpose code               #
#######################################

class SyntaxTreePipeline(SyntaxTree):

	def __init__(self, content):
		super().__init__(content)
		self.pipeline = None


	def __str__(self):
		# Get the terminals (i.e. functions and its hyper-parameters)
		terms = [elem for elem in self if isinstance(elem, TerminalNode)]
		# Build the string
		pipe_str = ""
		term_index = 0
		while term_index < len(terms):
			# The first node is always a function (transformer or classifier)
			pipe_str += terms[term_index].symbol + "("
			term_index += 1
			# Search its hyper-parameters
			h_params = []
			while term_index < len(terms) and "::" in terms[term_index].symbol:
				# Get the value of the hyper-parameter
				arg_val = terms[term_index].code.name
				# Check if value is a string
				if arg_val not in ["True", "False"] and not is_number(arg_val):
					arg_val = "'" + arg_val + "'"
				h_params.append(arg_val)
				term_index += 1
			# Append the hyper-parameters to the string
			pipe_str += ",".join(h_params) + ");"
		# Remove the last ";"
		return pipe_str[:-1]

	def __eq__(self, other):
		return str(self) == str(other) and self.fitness == other.fitness


	def create_sklearn_pipeline(self, pset):
		# Transform the strings into the steps of the pipeline
		steps = []
		steps_str = str(self).replace("'None'", "None").split(";")
		for index, step_str in enumerate(steps_str):
			steps.append((str(index), eval(step_str, pset.context, {})))
		self.pipeline = Pipeline(steps)




def tostring(nodes):
	# Get the terminals (i.e. functions and its hyper-parameters)
	terms = [elem for elem in nodes if isinstance(elem, TerminalNode)]
	# Build the string
	pipe_str = ""
	term_index = 0
	while term_index < len(terms):
		# The first node is always a function (transformer or classifier)
		pipe_str += terms[term_index].symbol + "("
		term_index += 1
		# Search its hyper-parameters
		h_params = []
		while term_index < len(terms) and "::" in terms[term_index].symbol:
			# Get the value of the hyper-parameter
			arg_val = terms[term_index].code.name
			# Check if value is a string
			if arg_val not in ["True", "False"] and not is_number(arg_val):
				arg_val = "'" + arg_val + "'"
			h_params.append(arg_val)
			term_index += 1
		# Append the hyper-parameters to the string
		pipe_str += ",".join(h_params) + ");"
	# Remove the last ";"
	return pipe_str[:-1]


def create_pipeline(pipestr, pset):
	# Transform the strings into the steps of the pipeline
	steps = []
	steps_str = pipestr.replace("'None'", "None").split(";")
	for index, step_str in enumerate(steps_str):
		steps.append((str(index), eval(step_str, pset.context, {})))
	return Pipeline(steps)
