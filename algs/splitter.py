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
        
    def init(self, x, y, sample_weight):
        
        n_samples = x.shape[0]
        
        self.samples = np.zeros(n_samples)
        
        samples = self.samples
        weighted_n_samples = 0.0
        j = 0
        
        for i in range(n_samples):
            if sample_weight is None or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight is not None:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0
        
        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples
        n_features = x.shape[1]
        self.features = np.arange(n_features)
        self.n_features = n_features
        
        self.feature_values = np.zeros(n_samples)
        self.constant_features = np.zeros(n_features)

        self.y = y

        self.sample_weight = sample_weight

    def node_value(self, dest):
        self.criterion.node_value(dest)

    def node_impurity(self):
        return self.criterion.node_impurity()
    
    def node_reset(self, start, end, weighted_n_node_samples):
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
        start = self.start
        end = self.end
        end_non_missing
        n_missing = 0
        has_missing = 0
        n_searches
        n_left, n_right
        missing_go_to_left

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
        n_found_constants = 0
        n_drawn_constants = 0
        n_known_constants = n_constant_features[0]
        n_total_constants = n_known_constants

        init_split(best_split, end)

        partitioner.init_node_split(start, end)

        while (f_i > n_total_constants and
            (n_visited_features < max_features or n_visited_features <= n_found_constants + n_drawn_constants)
        ):
            n_visited_features += 1

            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            if f_j < n_known_constants:
                features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]

                n_drawn_constants += 1
                continue

            f_j += n_found_constants
            current_split.feature = features[f_j]
            partitioner.sort_samples_and_feature_values(current_split.feature)
            n_missing = partitioner.n_missing
            end_non_missing = end - n_missing

            if (
                end_non_missing == start or
                feature_values[end_non_missing - 1] <= feature_values[start] + FEATURE_THRESHOLD
            ):
                features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]

                n_found_constants += 1
                n_total_constants += 1
                continue

            f_i -= 1
            features[f_i], features[f_j] = features[f_j], features[f_i]
            has_missing = n_missing != 0
            if has_missing:
                criterion.init_missing(n_missing)

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

                    if n_left < min_samples_leaf or n_right < min_samples_leaf:
                        continue

                    current_split.pos = p
                    criterion.update(current_split.pos)

                    if ((criterion.weighted_n_left < min_weight_leaf) or
                            (criterion.weighted_n_right < min_weight_leaf)):
                        continue

                    current_proxy_improvement = criterion.proxy_impurity_improvement()

                    if current_proxy_improvement > best_proxy_improvement:
                        best_proxy_improvement = current_proxy_improvement
                        current_split.threshold = (
                            feature_values[p_prev] / 2.0 + feature_values[p] / 2.0
                        )

                        if (
                            current_split.threshold == feature_values[p] or
                            current_split.threshold == infinity or
                            current_split.threshold == -infinity
                        ):
                            current_split.threshold = feature_values[p_prev]

                        current_split.n_missing = n_missing
                        if n_missing == 0:
                            current_split.missing_go_to_left = n_left > n_right
                        else:
                            current_split.missing_go_to_left = missing_go_to_left

                        # copy
                        best_split = current_split

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

        memcpy(features[0], constant_features[0], sizeof(SIZE_t) * n_known_constants)

        memcpy(constant_features[n_known_constants],
               features[n_known_constants],
               sizeof(SIZE_t) * n_found_constants)

        split[0] = best_split
        n_constant_features[0] = n_total_constants
        return 0
