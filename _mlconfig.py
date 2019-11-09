
''' Example for using Randomforest
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(50, 'mse')
'''

# ===================
# USAGE INSTRUCTIONS
# ===================
# 1. Create whatever model you want in this page
# 2. Make sure to name your model variable as final_model


# Example for using a simple keras dense model
from keras.layers import Dense, LeakyReLU
from keras.models import Sequential
import numpy as np


def create_model():
    xshape = int(np.loadtxt('save.txt'))
    model = Sequential()
    model.add(Dense(128, input_shape=(xshape, )))
    model.add(LeakyReLU())
    model.add(Dense(256))
    model.add(LeakyReLU())
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model


# Define scale type of inputs and outputs
# Scale types available are:
    # 1. minmax - normalize values between 0-1
    # 2. normalize - normalize values using mean
    # 3. standardize - standardize using mean and standard deviation
x_scale_type = 'minmax'
y_scale_type = 'minmax'

# Set model and keyword arguments
kwargs = {'verbose': 1, 'epochs': 2, 'shuffle': False, 'batch_size': 128}
final_model = create_model()
