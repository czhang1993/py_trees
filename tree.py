import numpy as np

is_first = 1
is_not_first = 0
is_left = 1
is_not_left = 0

tree_leaf = -1
tree_undefined = -2


class Node:
    def __init__(self, left_child, right_child, feature, threshold, impurity, n_node_samples=None,
                 weighted_n_node_samples=None):
        self.left_child = left_child
        self.right_child = right_child
        self.feature = feature
        self.threshold = threshold
        self.impurity = impurity
        self.n_node_samples = n_node_samples
        self.weighted_n_node_samples = weighted_n_node_samples


class Tree:
    def __init__(self, n_features, n_classes, n_outputs, max_n_classes, max_depth, node_count, capacity, nodes, value,
                 value_stride):
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

    def add_node(self, parent, is_left, is_leaf, feature, threshold, impurity, n_node_samples, weighted_n_node_samples):
        # calculate the new node ID
        node_id = self.node_count

        # check whether adding this node would exceed the specified tree capacity
        if node_id >= self.capacity:
            return 0

        # specify this node and its attributes
        node = self.nodes[node_id]
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples

        # if its node's parent node has been defined
        if parent != tree_undefined:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].left_right = node_id

        # if this node is a leaf
        if is_leaf:
            node.left_child = tree_leaf
            node.right_child = tree_leaf
            node.feature = tree_undefined
            node.threshold = tree_undefined

        else:
            # the left child and the right child will be set later
            node.feature = feature
            node.threshold = threshold

    # finds the leaf node for each sample in x
    def apply(self, x):
        # sample size
        n_samples = x.shape[0]

        # initialise output
        out = np.zeros(n_samples)

        # initialise auxiliary data-structure
        node = None
        i = 0

        # for i in 1: sample size
        for i in range(n_samples):
            # initialise the node as the root node
            node = self.nodes
            # while node is not a leaf
            while node.left_child != tree_leaf:
                # define x_i's node feature as x[i, node.feature]
                x_i_node_feature = x[i, node.feature]
                # if x_i's node feature <= the node's threshold
                if x_i_node_feature <= node.threshold:
                    # define the node as the node's left child
                    node = self.nodes[node.left_child]
                else:
                    # define the node as the node's right child
                    node = self.nodes[node.right_child]
            # node offset
            out[i] = (node - self.nodes)
        return np.asarray(out)

    def predict(self, x):
        out = self.get_value_ndarray()
        out = out.take(
            self.apply(x),
            axis=0,
            mode='clip'
        )
        if self.n_outputs == 1:
            out = out.reshape(x.shape[0], self.max_n_classes)
        return out


# class TreeBuilder:
#     __init__(self, tree, x, y, sample_weight=None):
#     pass
