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
        
        
class Tree:
    __init__(self, n_features, n_classes, n_outputs, max_n_classes, max_depth, node_count, capacity, nodes, value, value_stride):
        # input and output layout
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_outputs = n_outputs
        self.max_n_classes = max_n_classes
        
        # inner structure
        self.max_depth = max_depth
        self.node_count = node_count
        self.capacity = capacity
        self.nodes = nodes
        self.value = value
        self.value_stride = value_stride
        
        
class TreeBuilder:
    __init__(self, tree, x, y, sample_weight=None, feature_has_missing=None):
        pass
    
