import numpy as np

infinity = np.inf


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
        
        self.samples = np.zeros(n_samples)
        
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

        # calculate the total of dimensions of the features
        n_features = x.shape[1]
        # define the indices of the features
        self.features = np.arange(n_features)
        self.n_features = n_features
        
        # initialise a feature's values as an array of zeros
        self.feature_values = np.zeros(n_samples)
        # initialise a constant feature's values as an array of zeros
        self.constant_features = np.zeros(n_features)

        self.y = y

        self.sample_weight = sample_weight

    def init_split(self, start_pos):
        self.impurity_left = infinity
        self.impurity_right = infinity
        self.pos = start_pos
        self.feature = 0
        self.threshold = 0.
        self.improvement = -infinity
        
    def node_value(self, dest):
        """Copy the value of node samples[start:end] into dest."""
        self.criterion.node_value(dest)

    def node_impurity(self):
        """Return the impurity of the current node."""
        return self.criterion.node_impurity()
    
    def node_reset(self, start, end, weighted_n_node_samples):
        """Reset splitter on node samples[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        start : The index of the first sample to consider
        end : The index of the last sample to consider
        weighted_n_node_samples : ndarray, The total weight of those samples
        """

        self.start = start
        self.end = end

        self.criterion.init(
            self.y,
            self.sample_weight,
            self.weighted_n_samples,
            self.samples,
            start,
            end
        )

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples
        return 0
    
    def node_split_best(self, partitioner, criterion, impurity, split, n_constant_features):
        """Find the best split on node samples[start:end]

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # Find the best split
        start = self.start
        end = self.end
        n_searches
        n_left, n_right

        samples = self.samples
        features = self.features
        constant_features = self.constant_features
        n_features = self.n_features

        feature_values = self.feature_values
        max_features = self.max_features
        min_samples_leaf = self.min_samples_leaf
        min_weight_leaf = self.min_weight_leaf
        # random_state = splitter.rand_r_state

        current_proxy_improvement = -infinity
        best_proxy_improvement = -infinity

        f_i = n_features
        
        n_visited_features = 0
        # Number of features discovered to be constant during the split search
        n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        n_drawn_constants = 0
        n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        n_total_constants = n_known_constants

        init_split(best_split, end)

        partitioner.init_node_split(start, end)

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):

            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           # random_state
                          )

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]

                n_drawn_constants += 1
                continue

            # f_j in the interval [n_known_constants, f_i - n_found_constants[
            f_j += n_found_constants
            # f_j in the interval [n_total_constants, f_i[
            current_split.feature = features[f_j]
            partitioner.sort_samples_and_feature_values(current_split.feature)

            if (
                # All values for this feature are missing, or
                end_non_missing == start or
                # This feature is considered constant (max - min <= FEATURE_THRESHOLD)
                feature_values[end_non_missing - 1] <= feature_values[start] + FEATURE_THRESHOLD
            ):
                # We consider this feature constant in this case.
                # Since finding a split among constant feature is not valuable,
                # we do not consider this feature for splitting.
                features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]

                n_found_constants += 1
                n_total_constants += 1
                continue

            f_i -= 1
            features[f_i], features[f_j] = features[f_j], features[f_i]
            has_missing = n_missing != 0
            if has_missing:
                criterion.init_missing(n_missing)
            # Evaluate all splits

            # If there are missing values, then we search twice for the most optimal split.
            # The first search will have all the missing values going to the right node.
            # The second search will have all the missing values going to the left node.
            # If there are no missing values, then we search only once for the most
            # optimal split.
            n_searches = 2 if has_missing else 1

            for i in range(n_searches):
                missing_go_to_left = i == 1
                criterion.missing_go_to_left = missing_go_to_left
                criterion.reset()

                p = start

                while p < end_non_missing:
                    partitioner.next_p(p_prev, p)

                    if p >= end_non_missing:
                        continue

                    if missing_go_to_left:
                        n_left = p - start + n_missing
                        n_right = end_non_missing - p
                    else:
                        n_left = p - start
                        n_right = end_non_missing - p + n_missing

                    # Reject if min_samples_leaf is not guaranteed
                    if n_left < min_samples_leaf or n_right < min_samples_leaf:
                        continue

                    current_split.pos = p
                    criterion.update(current_split.pos)

                    # Reject if min_weight_leaf is not satisfied
                    if ((criterion.weighted_n_left < min_weight_leaf) or
                            (criterion.weighted_n_right < min_weight_leaf)):
                        continue

                    current_proxy_improvement = criterion.proxy_impurity_improvement()

                    if current_proxy_improvement > best_proxy_improvement:
                        best_proxy_improvement = current_proxy_improvement
                        # sum of halves is used to avoid infinite value
                        current_split.threshold = (
                            feature_values[p_prev] / 2.0 + feature_values[p] / 2.0
                        )

                        if (
                            current_split.threshold == feature_values[p] or
                            current_split.threshold == infinity or
                            current_split.threshold == -infinity
                        ):
                            current_split.threshold = feature_values[p_prev]

                        # copy
                        best_split = current_split

        # Reorganize into samples[start:best_split.pos] + samples[best_split.pos:end]
        if best_split.pos < end:
            partitioner.partition_samples_final(
                best_split.pos,
                best_split.threshold,
                best_split.feature,
                best_split.n_missing
            )
            if best_split.n_missing != 0:
                criterion.init_missing(best_split.n_missing)
            criterion.missing_go_to_left = best_split.missing_go_to_left

            criterion.reset()
            criterion.update(best_split.pos)
            criterion.children_impurity(
                best_split.impurity_left, best_split.impurity_right
            )
            best_split.improvement = criterion.impurity_improvement(
                impurity,
                best_split.impurity_left,
                best_split.impurity_right
            )

            shift_missing_values_to_left_if_required(best_split, samples, end)

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features[0], constant_features[0], sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(constant_features[n_known_constants],
               features[n_known_constants],
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        split[0] = best_split
        n_constant_features[0] = n_total_constants
        return 0
