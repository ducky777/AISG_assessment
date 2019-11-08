# Config for Kernel Ridge Regression

alpha =  1
kernel =  'linear'
# Possible kernels are
# Linear, Polynomial, Sigmoid, RBF, Laplacian
gamma = None # applicable for RBF, laplacian, polynomial and sigmoid kernels
degree = 3 # Degree for polynomial. Ignored by other kernels
coef0 = 1 # Zero coefficient for polynomial and sigmoid kernels. Ignored by the rest