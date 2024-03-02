# -*- coding: utf-8 -*-

#    Rafael Barbudo Lunar, PhD Student
#    Knowledge and Discovery Systems (KDIS)
#    University of Cordoba, Spain

from deap.gp import PrimitiveSet
import importlib
import random as rand
from scipy.stats import loguniform
import xml.etree.ElementTree as ETree
#import imblearn
#import sklearn
#import xgboost
#import evoflow

from evoflow.g3p.encoding import NonTerminalNode, TerminalNode


#######################################
# Sklearn parser methods              #
#######################################

class TerminalCode:
	"""
	This class is used to generate the code of the terminal nodes
	
	Parameters
	----------
	tCode : str
		The code of the terminal node
	hp_list : list
		The list of hyper-parameters of the terminal node
	seed : int
		Seed used to initialize the random number generator of the terminal node
	label : list, default=None
		TODO: Add description
	"""

	def __init__(self, tCode, hp_list, seed, label=None):
		self.tCode = tCode
		self.hp_list = hp_list
		self.seed = seed
		self.label = label

	def generate_term_code(self):
		return self.__call__

	def __call__(self, *argv):
		package_name = self.tCode[:self.tCode.find(".")]
		module_name = self.tCode[:self.tCode.rfind(".")]

		if package_name not in globals():
			globals()[package_name] = importlib.import_module(package_name)
		
		if module_name not in globals():
			globals()[module_name] = importlib.import_module(module_name)

		# Get the class given its name
		estimator_class = eval(self.tCode)
		
		if self.label is None:
			estimator = estimator_class()
		else:
			estimator = estimator_class(argv[3:])
			estimator.fit(None, self.label)

		# Assign the hyper-parameters
		for index, hp in enumerate(self.hp_list):
			hp = hp.split("::")[1]
			setattr(estimator, hp, argv[index])

		setattr(estimator, "random_state", self.seed)
		return estimator


def parse_pipe_grammar(filename, seed):
	"""
	Parse the grammar. This function is specially designed
	work with parametrized function, e.g. sklearn estimators

	:param filename: The name of the file containing the grammar
	:param filename: The global random seed

	:returns: the root, terminals, non terminals and the primitive
			  set conforming the grammar
	"""
	# Load the grammar file
	root, p_data_terms, p_func_terms, p_non_terms = load_grammar(filename)

	# Create the primitive set
	pset = PrimitiveSet("grammar_pset", len(p_data_terms))

	# Add the data terminals
	terms = []
	non_terms = []
	terms_families = {}

	# Add the function terminals
	for func_term in p_func_terms:
		# Get the hyper-parameters of the function
		hpo_list = [hp.get('name') for hp in func_term.findall('hparam')]
		# Get the node properties
		term_name = func_term.get('name')
		term_family = func_term.get('type')
		term_code = func_term.get('code')
		# Add an unexistent parameter to ensure arity > 0
		term_args = len(hpo_list) + 1
		# Add the non terminals representing the hyper-parameters
		production = ";".join(hpo_list)
		if production:
			non_terms.append(NonTerminalNode(term_name + "_hp", production))

		# Dinamically import required modules
		#module_name = term_code[:term_code.rfind(".")]
		#globals()[module_name] = importlib.import_module(module_name)
		
		# Add the primitive and terminal
		code = TerminalCode(term_code, hpo_list, seed).generate_term_code()
		pset.addPrimitive(code, term_args, name=term_name)
		term = TerminalNode(term_name, pset.primitives[pset.ret][-1], term_family)
		terms.append(term)
		
		# Add the symbol and family to the dictionary
		if term_family in terms_families:
			terms_families[term_family].append(term_name)
		else:
			terms_families[term_family] = [term_name]

		# Add the posible hyper parameter
		for hparam in func_term.findall('hparam'):
			hp_name = hparam.get('name')

			# Differentiate between the types of hyper-parameters
			hp_type = hparam.get('type')

			# A predefined set of categorical values
			if hp_type == "categorical":
				vals = hparam.get('values').split(";")
				is_none = hparam.get('default')
				if is_none == "None":
					vals.append(None)
				pset.addEphemeralConstant(hp_name, lambda values=vals:
										  rand.choice(values))
			# Boolean value
			elif "bool" in hp_type:
				if hp_type == "fix_bool":
					hp_val = hparam.get('value')
					hp_val = True if hp_val == 'True' else False
					pset.addEphemeralConstant(hp_name, lambda val=hp_val: val)
				else:
					pset.addEphemeralConstant(hp_name, lambda:
											  bool(rand.getrandbits(1)))
			# Integer value
			elif "int" in hp_type:
				if hp_type == "fix_int":
					hp_val = int(hparam.get('value'))
					pset.addEphemeralConstant(hp_name, lambda val=hp_val: val)
				elif hp_type == "uni_int":
					lb = hparam.get('lower')
					ub = hparam.get('upper')
					is_log = hparam.get('log')
					if is_log == "True":
						pset.addEphemeralConstant(hp_name, lambda lower=lb, upper=ub:
												  int(loguniform.rvs(int(lower), int(upper), size=1,
																	 random_state=rand.randrange(99999))[0]))
					else:
						pset.addEphemeralConstant(hp_name, lambda lower=lb, upper=ub:
												  rand.randint(int(lower), int(upper)))
			# Float value
			elif "float" in hp_type:
				if hp_type == "fix_float":
					hp_val = float(hparam.get('value'))
					pset.addEphemeralConstant(hp_name, lambda val=hp_val: val)
				elif hp_type == "uni_float":
					lb = hparam.get('lower')
					ub = hparam.get('upper')
					is_log = hparam.get('log')
					if is_log == "True":
						pset.addEphemeralConstant(hp_name, lambda lower=lb, upper=ub:
												  loguniform.rvs(float(lower), float(upper), size=1,
																 random_state=rand.randrange(99999))[0])
					else:
						pset.addEphemeralConstant(hp_name, lambda lower=lb, upper=ub:
												  rand.uniform(float(lower), float(upper)))
			else:
				print("ERROR: ", hp_type)

			# Add the terminal to the primitive set
			term = TerminalNode(hp_name, pset.terminals[pset.ret][-1])
			terms.append(term)

	# Parse the non terminal nodes (i.e. the production rules)
	non_terms.extend(load_productions(p_non_terms))

	# Return the elements composing the grammar
	return root, terms, non_terms, pset, terms_families


#######################################
# General methods                     #
#######################################

def load_grammar(filename):
	"""
	Parse the grammar filename grammar, returning the elements
	that will be latter processed to conform the PrimitiveSet

	:param filename: The name of the file containing the grammar

	:returns: the root, terminals (data and functions) and non
	terminals, as they are read from the grammar file
	"""
	# Parse the grammar file
	grammar = ETree.parse(filename).getroot()

	# Parse the root, terminals and non terminals
	root = grammar.find('root-symbol').text
	terms = grammar.findall('terminals/terminal')
	non_terms = grammar.findall('non-terminals/non-terminal')

	# Differentiate between functions and data in terminals
	data_terms = [term for term in terms
				  if term.get('code') is None]
	func_terms = [term for term in terms
				  if term.get('code') is not None]

	return root, data_terms, func_terms, non_terms


def load_productions(p_non_terms):
	non_terms = []
	for non_term in p_non_terms:
		name = non_term.get('name')
		for production in non_term.findall('production-rule'):
			non_terms.append(NonTerminalNode(name, production.text))
	return non_terms


def load_data_terminals(p_data_terms, pset):
	terms = []
	for index, data_term in enumerate(p_data_terms):
		# Get the node data
		term_name = data_term.get('name')
		# Rename the argument
		eval("pset.renameArguments(ARG" + str(index) + "=term_name)")
		# Add the terminal
		term = TerminalNode(term_name, pset.terminals[pset.ret][index])
		terms.append(term)
	return terms
