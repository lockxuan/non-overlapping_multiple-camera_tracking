
import numpy as np




def cosine_distance(a, b, normalize=True):
	""" compute cosine diatance between vectors 'a' and 'b'

	Parameters
	----------
	a: array
		An NxM matrix of N samples of M dimensions.
	b: array
		An LxM matrix of L samples of M dimensions.

	normalize: Optional[bool]
	
	Return
	------
	ndarray
		Returns a matrix of size len(a) x len(b)

	"""


	if normalize:
		a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
		b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)

	return (1 - np.dot(a, b.T)).min(axis=0)


def euclidean_distance(a, b):
	""" compute euclidean diatance between vectors 'a' and 'b'

	Parameters
	----------
	a: array
		An NxM matrix of N samples of M dimensions.
	b: array
		An LxM matrix of L samples of M dimensions.

	normalize: Optional[bool]
	
	Return
	------
	ndarray
		Returns a matrix of size len(a) x len(b)

	"""

	#a, b = np.asarray(a), np.asarray(b)
	a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
	b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)

	if len(a) == 0 or len(b) == 0:
		return np.zeros((len(a), len(b)))

	a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
	r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
	distances = np.clip(r2, 0., float(np.inf))

	return np.maximum(0.0, distances.min(axis=0))







class NearestNeighborDistanceMetric(object):
	"""
	A nearest neighbor distance metric that, for each target, returns
	the closest distance to any sample that has been observed so far.
	Parameters
	----------
	metric : str
		Either "euclidean" or "cosine".
	matching_threshold: float
		The matching threshold. Samples with larger distance are considered an
		invalid match.
	budget : Optional[int]
		If not None, fix samples per class to at most this number. Removes
		the oldest samples when the budget is reached.
	Attributes
	----------
	samples : Dict[int -> List[ndarray]]
		A dictionary that maps from target identities to the list of samples
		that have been observed so far.
	"""

	def __init__(self, metric, matching_threshold=0.2, budget=None): #0.4


		if metric == "euclidean":
			self._metric = euclidean_distance
		elif metric == "cosine":
			self._metric = cosine_distance
		else:
			raise ValueError(
				"Invalid metric; must be either 'euclidean' or 'cosine'")
		self.matching_threshold = matching_threshold
		self.budget = budget
		self.samples = {}

	def partial_fit(self, features, targets, active_targets):
		"""Update the distance metric with new data.
		Parameters
		----------
		features : ndarray
			An NxM matrix of N features of dimensionality M.
		targets : ndarray
			An integer array of associated target identities.
		active_targets : List[int]
			A list of targets that are currently present in the scene.
		"""
		#print(features.shape)
		#print(targets)
		for feature, target in zip(features, targets):
			self.samples.setdefault(target, []).append(feature)
			if self.budget is not None:
				self.samples[target] = self.samples[target][-self.budget:]
		self.samples = {k: self.samples[k] for k in active_targets}

	def distance(self, features, targets):
		"""Compute distance between features and targets.
		Parameters
		----------
		features : ndarray
			An NxM matrix of N features of dimensionality M.
		targets : List[int]
			A list of targets to match the given `features` against.
		Returns
		-------
		ndarray
			Returns a cost matrix of shape len(targets), len(features), where
			element (i, j) contains the closest squared distance between
			`targets[i]` and `features[j]`.
		"""
		cost_matrix = np.zeros((len(targets), len(features)))
		for i, target in enumerate(targets):
			cost_matrix[i, :] = self._metric(self.samples[target], features)

		return cost_matrix






