# Config for Decision Tree Regressor


criterion = 'mse' # Function to measure quality of split.
# Possible criterions are
# 'mse' for L2 error 
# 'mae' for L1 error
# 'friedman_mse' which uses mean squared error with Friedman’s improvement score for potential splits

max_depth = None # Max depth of the tree

min_samples_split = 2 # The minimum number of samples required to split an internal node

min_samples_leaf = 1 # The minimum number of samples required to be at a leaf node

random_state = None # seed
