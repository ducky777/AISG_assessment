
''' Example for using Randomforest
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(50, 'mse')
'''

# ===================
# USAGE INSTRUCTIONS
# ===================
# 1. Create whatever model you want in this page
# 2. Make sure to name your model variable as model


# Example for using a simple keras dense model
from keras.layers import Dense, LeakyReLU
from keras.models import Sequential


def create_model():
    model = Sequential()
    model.add(Dense(128, input_dim=()))
    model.add(LeakyReLU())
    model.add(Dense(256))
    model.add(LeakyReLU())
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model


model = create_model()
