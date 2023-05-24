class Splitter:
    def __init__(self, criterion, max_features, min_samples_leaf, min_weight_leaf, random_state):
        self.criterion = criterion
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state
