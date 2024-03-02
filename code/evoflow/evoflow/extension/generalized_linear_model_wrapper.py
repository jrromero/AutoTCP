from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from statsmodels.api import add_constant, GLM
from statsmodels.genmod.families import Binomial
from numpy import column_stack
from numpy.random import RandomState

class StatsmodelsGeneralizedLinearModel(BaseEstimator, ClassifierMixin):
	"""
	Statsmodel Generalized Linear Model wrapper.

	Parameters
	----------
	family : str, default='binomial'
		The family of the GLM.

	fit_intercept : bool, default=True
		Whether to fit an intercept.

	max_iter : int, default=100
		The maximum number of iterations.
	
	tol : float, default=1e-8
		Convergence tolerance.
	
	penalty : str, default=None
		The penalty to use. Must be None, 'l1', 'l2' or 'elastic-net'.

	alpha : float, default=0
		The penalty weight. Only used if penalty is not None.

	l1_ratio : float, default=None
		The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. Only used if penalty='elastic-net'.
		L1 penalty is l1_ratio * C and L2 penalty is (1 - l1_ratio) * C.

	random_state : int or None, default=None
		The seed of the pseudo random number generator to use when initializing the weights.

	Attributes
	----------
	model_ : statsmodels.genmod.generalized_linear_model.GLM
		The model object.

	result_ : statsmodels.genmod.generalized_linear_model.GLMResults
		The model results.

	params_ : array, shape (n_features,)
		The model parameters.

	intercept_ : float
		The intercept.

	coef_ : array, shape (n_features,)
		The coefficients.
	"""
	def __init__(self, family='binomial', fit_intercept=True, max_iter=100, tol=1e-8, penalty=None, alpha=0, l1_ratio=0.5, random_state=None):
		self.family = family
		self.fit_intercept = fit_intercept
		self.max_iter = max_iter
		self.tol = tol
		self.penalty = penalty
		self.alpha = alpha
		self.l1_ratio = l1_ratio
		self.random_state = random_state

	def fit(self, X, y):
		"""
		Fit the model according to the given training data.

		Parameters
		----------
		X : array-like, shape (n_samples, n_features)
			Training vector, where n_samples is the number of samples and
			n_features is the number of features.

		y : array-like, shape (n_samples,)
			Target vector relative to X.

		Returns
		-------
		self : object
			Returns fitted estimator.
		"""
		if self.fit_intercept:
			X = add_constant(X) # add intercept

		if self.family == 'binomial':
			family = Binomial()
		else:
			raise ValueError("Currently only 'binomial' is supported as family; got (family=%r)" % self.family)

		self.model_ = GLM(y, X, family=family)

		if self.penalty == None:
			start_params = RandomState(self.random_state).uniform(-1, 1, X.shape[1])
			self.result_ = self.model_.fit(maxiter=self.max_iter, tol=self.tol, start_params=start_params)
		else:
			if self.penalty == 'l1':
				L1_wt = 1.0
			elif self.penalty == 'l2':
				L1_wt = 1e-8
			elif self.penalty == 'elastic-net':
				if self.l1_ratio < 0.0 or self.l1_ratio > 1.0:
					raise ValueError("l1_ratio must be between 0 and 1; got (l1_ratio=%r)" % self.l1_ratio)
				
				if self.l1_ratio < 1e-8:
					L1_wt = 1e-8
				else:
					L1_wt = self.l1_ratio
			else:
				raise ValueError("penalty must be None, 'l1', 'l2' or 'elastic-net'; got (penalty=%r)" % self.penalty)

			start_params = RandomState(self.random_state).uniform(-1, 1, X.shape[1])
			self.result_ = self.model_.fit_regularized(alpha=self.alpha, L1_wt=L1_wt, cnvrg_tol=self.tol, maxiter=self.max_iter, start_params=start_params)

		self.params_ = self.result_.params
		self.intercept_, self.coef_ = self.result_.params[0], self.result_.params[1:]
		return self

	def predict(self, X):
		"""
		Predict class labels for samples in X.

		Parameters
		----------
		X : array-like, shape (n_samples, n_features)
			Vector of samples.

		Returns
		-------
		C : array, shape (n_samples,)
			The predicted class label per sample.
		"""
		if self.fit_intercept:
			has_constant = self.params_.shape[0] == X.shape[1] # avoid confusion with a feature with 0 variance
			X = add_constant(X, has_constant='add' if not has_constant else 'skip') # add intercept
		
		return (self.model_.predict(self.params_, X) > 0.5).astype(int) # return 0 or 1

	def predict_proba(self, X):
		"""
		Estimates the probability for each sample in X to belong to each of the two classes.

		Parameters
		----------
		X : array-like, shape (n_samples, n_features)
			Vector of samples.

		Returns
		-------
		T : array-like, shape (n_samples, n_classes)
			Probabilities of the sample for each class in the model.
		"""
		if self.fit_intercept:
			has_constant = self.params_.shape[0] == X.shape[1] # avoid confusion with a feature with 0 variance
			X = add_constant(X, has_constant='add' if not has_constant else 'skip') # add intercept

		y_pred = self.model_.predict(self.params_, X)
		return column_stack((y_pred, 1 - y_pred))

	def score(self, X, y, sample_weight=None):
		"""
		Score the model according to the given test data and labels.

		Parameters
		----------
		X : array-like, shape (n_samples, n_features)
			Test samples.

		y : array-like, shape (n_samples,)
			True labels for X.

		sample_weight : array-like, shape (n_samples,), default=None
			Sample weights.

		Returns
		-------
		score : float
			Mean accuracy of self.predict(X) wrt. y.
		"""
		if self.fit_intercept:
			has_constant = self.params_.shape[0] == X.shape[1] # avoid confusion with a feature with 0 variance
			X = add_constant(X, has_constant='add' if not has_constant else 'skip') # add intercept

		return accuracy_score(y, self.predict(X), sample_weight=sample_weight)