class Node:
    __init__(self, left_child, right_child, feature, threshold, impurity, n_node_samples=None, weighted_n_node_samples=None, missing_go_to_left=None):
        self.left_child = left_child
        self.right_child = right_child
        self.feature = feature
        self.threshold = threshold
        self.impurity = impurity
        self.n_node_samples = n_node_samples
        self.weighted_n_node_samples = weighted_n_node_samples
        self.missing_go_to_left = missing_go_to_left
