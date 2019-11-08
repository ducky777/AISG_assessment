import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import SGDRegressor

class MLAlgorithms:
    def __init__(self):
        return

    def feature_importance(self):
        clf = ExtraTreesClassifier(n_estimators=5).fit(self.x, self.y)
        f_importance = clf.feature_importances_
        importance_idx = np.argsort(f_importance)[::-1]
        return f_importance, importance_idx

    def decision_tree_regressor(self, x, y, criterion, max_depth, min_samples_split
                                , min_samples_leaf, random_state):
        model = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf
                                      , min_samples_split=min_samples_split, random_state=random_state)

        print('Fitting Decision Tree...')
        model.fit(x, y)
        print('Fitting Decision Tree done!')
        return model

    def kernel_ridge_regression(self, x, y, alpha, kernel, gamma, degree, coef0):
        model = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
        print('Fitting Kernel Ridge Regression...')
        model.fit(x, y)
        print('Fitting Kernel Ridge Regression Done!')
        return

    def k_nearest_neighbour(self, x, y, n_neighbors, p):
        model = KNeighborsRegressor(n_neighbors=n_neighbors, p=p)
        print('Fitting k-NN...')
        model.fit(x, y)
        print('Fitting k-NN done!')
        return model

    def sgd_regressor(self, x, y, loss, penalty, alpha, l1_ratio
                      , fit_intercept, max_iter, tol, shuffle, verbose
                      , epsilon, random_state, learning_rate, eta0
                      , power_t, early_stopping, validation_fraction
                      , n_iter_no_change, warm_start, average):

        model = SGDRegressor(loss, penalty, alpha, l1_ratio
                             , fit_intercept, max_iter, tol, shuffle, verbose
                             , epsilon, random_state, learning_rate, eta0
                             , power_t, early_stopping, validation_fraction
                             , n_iter_no_change, warm_start, average)

        model.fit(x, y)

        return model
