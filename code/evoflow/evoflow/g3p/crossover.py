# -*- coding: utf-8 -*-

#    Rafael Barbudo Lunar, PhD Student
#    Knowledge and Discovery Systems (KDIS)
#    University of Cordoba, Spain

"""
The :mod:`crossover` for Grammar Guided Genetic Programming (G3P).

This module provides the methods to combine two or more SyntaxTree.
"""

import random as rand
from evoflow.g3p.encoding import NonTerminalNode


def _swap_branches(ind1, ind2, schema, start1, start2):

	# Get the slice to swap
	slice1 = ind1.search_subtree(start1)
	slice2 = ind2.search_subtree(start2)

	# Get branch depth (to check maximum size)
	p0_non_terms = sum(1 for node in ind1 if isinstance(node, NonTerminalNode))
	p0_swapBranch = sum([1 for node in ind1[slice1] if isinstance(node, NonTerminalNode)])
	p1_non_terms = sum(1 for node in ind2 if isinstance(node, NonTerminalNode))
	p1_swapBranch = sum([1 for node in ind2[slice2] if isinstance(node, NonTerminalNode)])

	# Check maximum number of derivation conditions
	max_derivations = schema.maxDerivSize
	cond0 = (p0_non_terms - p0_swapBranch + p1_swapBranch > max_derivations)
	cond1 = (p1_non_terms - p1_swapBranch + p0_swapBranch > max_derivations)
	if cond0 or cond1:
		return ind1, ind2

	# Swap and return the individuals
	ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]
	return ind1, ind2


def _cx_struct(ind1, ind2, schema):

	# get the common nodes (symbols)
	incl = ['classifier', 'preprocessingBranch']

	# Select a common non terminal node
	start1, start2 = _select_symbol_incl(ind1, ind2, incl)

	return _swap_branches(ind1, ind2, schema, start1, start2)


def _swap_hp(ind1, ind2, hp_name):
	for idx1, node1 in enumerate(ind1):
		if node1.symbol == hp_name:
			for idx2, node2 in enumerate(ind2):
				if node2.symbol == hp_name:
					aux = ind1[idx1]
					ind1[idx1] = ind2[idx2]
					ind2[idx2] = aux
	return ind1, ind2


def _cx_hps(ind1, ind2, common_hps):

	# randomly select a crossover point
	cxpoint = rand.randint(1, len(common_hps)-1)

	# swap the hyper-parameters
	for idx, hp_name in enumerate(common_hps):
		if idx < cxpoint:
			ind1, ind2 = _swap_hp(ind1, ind2, hp_name)

	return ind1, ind2


def cx_multi(ind1, ind2, schema):

	par1_str = str(ind1)
	par2_str = str(ind2)

	# Get the common hyper-parameters
	hps1 = [node.symbol for node in ind1 if "::" in node.symbol]
	hps2 = [node.symbol for node in ind2 if "::" in node.symbol]
	common_hps = sorted(list(set(hps1) & set(hps2)))

	# Apply one of the crossover operators
	if len(common_hps) > 1:
		ind1, ind2 = _cx_hps(ind1, ind2, common_hps)
	else:
		ind1, ind2 = _cx_struct(ind1, ind2, schema)

	# Ensure sons are different
	son1_str = str(ind1)
	son2_str = str(ind2)
	if son1_str == par1_str and son2_str == par2_str:
		return ind1, ind2, False
	return ind1, ind2, True


def _select_symbol_incl(ind1, ind2, incl=[]):

	# Individual length
	ind1_len = len(ind1)

	# Generate a tree position at random
	start_pos = rand.randint(0, ind1_len)
	act_pos = start_pos

	for index1 in range(ind1_len):
		# Update the current position
		act_pos = (start_pos + index1) % ind1_len
		# Get the node
		node1 = ind1[act_pos]
		# Check symbol is an included one
		if node1.symbol in incl:
			# Find the ocurrences of the symbol in the other individual
			indexes2 = [index2 for index2, node2 in enumerate(ind2)
						if node1.symbol == node2.symbol]
			# Return a coincidence at random
			if len(indexes2) != 0:
				return act_pos, rand.choice(indexes2)

	# There are not occurrences
	return None, None


def cx_branches(ind1, ind2, schema):

	par1_str = str(ind1)
	par2_str = str(ind2)

	# Select a common non terminal node (excluding the root)
	start1, start2 = _select_symbol_excl(ind1, ind2, [schema.root, "classificationBranch"])
	
	# Apply the crossover
	ind1, ind2 = _swap_branches(ind1, ind2, schema, start1, start2)

	# Ensure sons are different
	son1_str = str(ind1)
	son2_str = str(ind2)
	if son1_str == par1_str and son2_str == par2_str:
		return ind1, ind2, False
	return ind1, ind2, True


def _select_symbol_excl(ind1, ind2, excl=None):

	# Individual length
	ind1_len = len(ind1)

	# Generate a tree position at random
	start_pos = rand.randint(0, ind1_len)
	act_pos = start_pos

	for index1 in range(ind1_len):
		# Update the current position
		act_pos = (start_pos + index1) % ind1_len
		# Get the node
		node1 = ind1[act_pos]
		# Check symbol is non termianl
		if isinstance(node1, NonTerminalNode) != 0 and (excl is None or node1.symbol not in excl):
			# Find the ocurrences of the symbol in the other individual
			indexes2 = [index2 for index2, node2 in enumerate(ind2)
						if node1.symbol == node2.symbol]
			# Return a coincidence at random
			if len(indexes2) != 0:
				return act_pos, rand.choice(indexes2)

	# There are not occurrences
	return None, None