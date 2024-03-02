# -*- coding: utf-8 -*-

#    Rafael Barbudo Lunar, PhD Student
#    Knowledge and Discovery Systems (KDIS)
#    University of Cordoba, Spain

#    Edited by Ángel Fuentes Almoguera, Bachelor Student

"""
The :mod:`mutation` for Grammar Guided Genetic Programming (G3P).

This module provides the methods to mutate SyntaxTree.
"""
import random
from copy import deepcopy
import random as rand
from evoflow.g3p.encoding import TerminalNode, NonTerminalNode
from random import shuffle


def rebuild_branch(ind, schema, start):
	# Make a copy just in case
	ind_bak = deepcopy(ind)

	# Get the subtree to mutate and its end
	tree_slice = ind.search_subtree(start)
	p0_branchEnd = tree_slice.stop

	# Get branch depth (to check maximum size)
	p0_branchDepth = sum(1 for node in ind if isinstance(node, NonTerminalNode))

	# Determine the maximum size to fill (to check maximum size)
	p0_swapBranch = sum(1 for node in ind[tree_slice] if isinstance(node, NonTerminalNode))

	# Get the symbol
	symbol = ind[start].symbol

	# Save the fragment at the right of the subtree
	aux = ind[p0_branchEnd:]

	# Remove the subtree and the fragment at its right
	del ind[start:]

	# Create the son (second fragment) controlling the number of derivations
	max_derivations = schema.maxDerivSize - p0_branchDepth + p0_swapBranch
	min_derivations = schema.minDerivations(symbol)
	try:
		derivations = rand.randint(min_derivations, max_derivations)
	except ValueError:
		return ind_bak
	schema.fillTreeBranch(ind, symbol, derivations)

	# Restore the fragment at the right of the subtree
	ind += aux
	return ind


def mut_multi(ind, schema):
	# get the string representing the parent
	par_str = str(ind)

	# apply one of the mutators
	randval = rand.random()
	if randval < 0.2:
		son = _mut_struct(ind, schema)
	else:
		son = _mut_hps(ind, schema)

	# check whether the new individual is equal
	if str(son) == par_str:
		return son, False
	return son, True


def _mut_hps(ind, schema):
	# get the start position of the classifier
	pos_classifier = [idx for idx, node in enumerate(ind) if node.symbol == "classifier"][0]

	# get the number of hyperparameters for the preprocessing and the classifier
	num_prep_hps = sum([1 for node in ind[:pos_classifier] if "::" in node.symbol])
	num_class_hps = sum([1 for node in ind[pos_classifier:] if "::" in node.symbol])

	# compute the probability of changing one hyperparameter
	mutpb_class = 0.0
	mutpb_prep = 0.0

	if num_prep_hps > 0:
		mutpb_prep = 1.0 / num_prep_hps

	if num_class_hps > 0:
		mutpb_class = 1.0 / num_class_hps

	# mutate each hyperparameter with its corresponding probability
	for idx, node in enumerate(ind):

		if idx < pos_classifier:
			mutpb = mutpb_prep
		else:
			mutpb = mutpb_class

		if "::" in node.symbol and rand.random() < mutpb:
			term = schema.terminals_map.get(node.symbol)
			ind[idx] = TerminalNode(node.symbol, term.code())

	# return the individual after modifying it
	return ind


def _mut_struct(ind, schema):
	# we want to mutate high level non-terminals
	target_symbols = ['workflow', 'preprocessingBranch']
	# get the possible start points to start the mutation
	possible_starts = [index for index, node in enumerate(ind) if node.symbol in target_symbols]
	# get one start point at random
	start = rand.choice(possible_starts)
	# perform the mutation
	return rebuild_branch(ind, schema, start)


def mut_branch(ind, schema):
	# get the string representing the parent
	par_str = str(ind)
	# get the possible start points to start the mutation
	possible_starts = [index for index, node in enumerate(ind) if isinstance(node, NonTerminalNode)]
	# get one start point at random
	start = rand.choice(possible_starts)
	# perform the mutation
	son = rebuild_branch(ind, schema, start)
	# check whether the new individual is equal
	if str(son) == par_str:
		return son, False
	return son, True


def mut_hps(ind, schema):
	# get the string representing the parent
	par_str = str(ind)
	# mutate the hyperparameters
	son = _mut_hps(ind, schema)
	# check whether the new individual is equal
	if str(son) == par_str:
		return son, False
	return son, True


def _permut_branch(ind, preprocess_nodes_indexes, classifier_node_index):
	# make a copy
	ind_bak = deepcopy(ind)
	# create list of node indexes
	nodes = list(preprocess_nodes_indexes)
	nodes.append(classifier_node_index)
	# list of segments
	segments = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
	# permute the list
	shuffle(segments)

	# retrieve the preprocesses nodes ordered
	permuted_preprocesses = []
	for i in range(len(segments)):
		permuted_preprocesses.extend(ind_bak[segments[i][0]:segments[i][1]])

	# create the new ind
	ind[:nodes[0]] = ind_bak[:nodes[0]]
	ind[nodes[0]:classifier_node_index] = permuted_preprocesses
	ind[classifier_node_index:] = ind_bak[classifier_node_index:]
	return ind


def mut_permut(ind, schema):
	# get the string representing the parent
	par_str = str(ind)
	# get the possible preprocess nodes to permute
	preprocess_nodes_indexes = [index for index, node in enumerate(ind) if node.symbol == 'preprocess' or node.symbol == 'classifier']
	# get the classification node
	#classifier_node_index = next((index for index, node in enumerate(ind) if node.symbol == 'classifier'))
	classifier_node_index = preprocess_nodes_indexes.pop()
	# check if there are preprocess nodes to permute
	if len(preprocess_nodes_indexes) == 0:
		return ind, False
	# perform the mutation
	son = _permut_branch(ind, preprocess_nodes_indexes, classifier_node_index)
	# check whether the new individual is equal
	if str(son) == par_str:
		return son, False
	return son, True


def _exchange_node(ind, schema, selected_node):
	# we want to mutate non-terminals
	target_symbols = ['preprocess', 'classifier']
	# make a copy
	ind_bak = deepcopy(ind)
	# get the length of the node to exchange
	next_index = next((index for index, node in enumerate(ind) if node.symbol in target_symbols and index > selected_node[0]), None)
	# probability to exchange the whole node or only its hyperparameters
	if rand.random() < 0.8:
		# create new node
		new_node = []
		# 2 derivations, one for de algorithm and the other for its hyperparameters
		schema.fillTreeBranch(new_node, selected_node[1], 2)

		# replace the old node with the new one
		ind[selected_node[0]:next_index] = new_node
	else:
		num_hps = sum([1 for node in ind[selected_node[0]:next_index] if "::" in node.symbol])
		mutpb = 0.0

		if num_hps > 0:
			mutpb = 1.0 / num_hps

		hps_pos = 0
		prob = rand.random()

		# look for hyperparameters and exchange one of them
		for index, node in enumerate(ind[selected_node[0]:next_index]):
			if "::" in node.symbol:
				hps_pos += 1

				if prob < mutpb * hps_pos:
					term = schema.terminals_map.get(node.symbol)
					ind[selected_node[0] + index] = TerminalNode(node.symbol, term.code())
					break

	return ind


def mut_exchange(ind, schema):
	# we want to mutate non-terminals
	target_symbols = ['preprocess', 'classifier']
	# get the string representing the parent
	par_str = str(ind)
	# possible nodes to exchange
	possible_nodes = [(index, node.symbol) for index, node in enumerate(ind) if node.symbol in target_symbols]
	# select a node to exchange randomly
	selected_node = random.choice(possible_nodes)
	# perform the mutation
	son = _exchange_node(ind, schema, selected_node)
	# check whether the new individual is equal
	if str(son) == par_str:
		return son, False
	return son, True


def mut_random_mutation(ind, schema):
	# all mutations type
	mutations = [mut_multi, mut_branch, mut_hps, mut_permut, mut_exchange]
	# select a random mutation type
	mutation = random.choice(mutations)
	# perform the selected mutation and return the new ind
	return mutation(ind, schema)
