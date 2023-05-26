import numpy as np


class Splitter:
    def __init__(self, criterion, max_features, min_samples_leaf, min_weight_leaf, 
                 # random_state
                ):
        self.criterion = criterion
        
        self.n_samples = 0
        self.n_features = 0
        
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        # self.random_state = random_state
        
    def init(self, x, y, sample_weight):
        # self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        
        n_samples = x.shape[0]
        
        self.samples = np.empty(n_samples)
        
        samples = self.samples
        weighted_n_samples = 0.0
        j = 0
        
        for i in range(n_samples):
            # only work with positively weighted samples
            if sample_weight is None or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight is not None:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0
        
        # number of samples is the number of positively weighted samples
        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples

        n_features = x.shape[1]
        self.features = np.arange(n_features)
        self.n_features = n_features

        self.feature_values = np.empty(n_samples)
        self.constant_features = np.empty(n_features)

        self.y = y

        self.sample_weight = sample_weight
        if feature_has_missing is not None:
            self.criterion.init_sum_missing()
        return 0
    
    
    def node_split(self, impurity, split, n_constant_features):
        """Find the best split on node samples[start:end].

        This is a placeholder method. The majority of computation will be done
        here.

        It should return -1 upon errors.
        """
        pass

    def node_value(self, dest):
        """Copy the value of node samples[start:end] into dest."""
        self.criterion.node_value(dest)

    def node_impurity(self):
        """Return the impurity of the current node."""
        return self.criterion.node_impurity()
        
