
# Example for using RandomForest
from sklearn.ensemble import RandomForestRegressor


def create_model():
    model = RandomForestRegressor(50, 'mse')
    return model

# ===================
# USAGE INSTRUCTIONS
# ===================
# 1. Create whatever model you want in this page
# 2. Make sure to name your model variable as final_model


# Example for using a simple keras dense model

# from keras.layers import Dense, LeakyReLU
# from keras.models import Sequential
# import os
#
# def create_model():
#     xshape1 = int(os.environ['x_shape_1'])
#     print('Input shape: ', xshape1)
#     model = Sequential()
#     model.add(Dense(128, input_shape=(xshape1,)))
#     model.add(LeakyReLU())
#     model.add(Dense(256))
#     model.add(LeakyReLU())
#     model.add(Dense(1, activation='linear'))
#     print('Model created')
#     model.compile(optimizer='adam', loss='mse')
#     print('Model compiled')
#     return model

# ========================================
# Define scale type of inputs and outputs
# ========================================
# Scale types available are:
    # 1. minmax - normalize values between 0-1
    # 2. normalize - normalize values using mean
    # 3. standardize - standardize using mean and standard deviation

x_scale_type = 'minmax'
y_scale_type = 'minmax'
metric = 'mse'
save_model_filename_prefix = ''  # Leave it blank if you do not wish to save your model using this pipeline
validation_split = 0.3  # Ratio of data to be used as validation set
#fit_kwargs = {'verbose': 1, 'epochs': 2, 'shuffle': False, 'batch_size': 128}  # kwargs for your model's fit function
fit_kwargs = {}
preprocess_fn = None

























#========================================================
#===================DO NOT TOUCH!========================
#========================================================
import os
try:
    xshape = int(os.environ['x_shape_1'])
except KeyError:
    xshape = -1
    pass
if xshape > 0:
    final_model = create_model()
