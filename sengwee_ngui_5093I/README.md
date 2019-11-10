# AIAP Technical Assessment ML Pipeline

Submission for technical assessment

## Installing

Use python 3.6.7 or python 3.6.8\
\
Required packages are in requirements.txt

## Usage

1. Open the file _mlconfig.py in the __mlp folder__
2. Choose your x variables scaling function
3. Import/Write/Define your desired model here
4. Set the kwargs for your model's fit function if any
5. Assign your model to the variable final_model
6. Type in your model save file prefix in save_model_filename_prefix if\
you want the pipeline to save your model. Otherwise leave it blank.
7. Run main.py

## Custom Preprocess Function
You can write your own preprocess function in\
_mlconfig.py file and pass it to preprocess_fn variable.\
Your preprocess function needs to take in the dataframe as\
an input and output dataframe, x, y.

Example:
```python
def own_preprocess(df):
    x = df/2
    y = df*2
    df = df-1
    return x, y, df

preprocess_fn = own_preprocess
```

## Explaination of the Pipeline
The pipeline will take in the dataframe and
1. Create a categorical representation of weather data
2. Create one hot encoding for the weather data
3. Create Hour, Day, DayOfWeek and Month data from\
the date_time column
4. Remove holiday and date_time from the dataframe
5. Fit the processed data into the machine learning model of your choice
6. Print out the evaluation metrics of the model

## Usage example of _mlconfig.py
Create your model within **create_model()** function. It has\
to return the model.

The following shows an example for using a Keras model.

#### READ FIRST - FOR MODELS THAT REQUIRES INPUT DIM
For models that require the shape of the input for model creation,\
i.e. keras models, use **int(os.environ['x_shape_1'])** to fetch\
the input size.

```python
from keras.layers import Dense, LeakyReLU
from keras.models import Sequential
import os

def create_model():
    xshape1 = int(os.environ['x_shape_1'])
    print('Input shape: ', xshape1)
    model = Sequential()
    model.add(Dense(128, input_shape=(xshape1,)))
    model.add(LeakyReLU())
    model.add(Dense(256))
    model.add(LeakyReLU())
    model.add(Dense(1, activation='linear'))
    print('Model created')
    model.compile(optimizer='adam', loss='mse')
    print('Model compiled')
    return model

fit_kwargs = {'verbose': 1, 'epochs': 2, 'shuffle': False, 'batch_size': 128}
```

Another example using sklearn's random forest.

```python
from sklearn.ensemble import RandomForestRegressor
def create_model():
    model = RandomForestRegressor(50, 'mse')
    return model

fit_kwargs = {}
```

## _mlconfig.py variables
1. x_scale_type - method to scale x variables. Available methods are:
    - minmax -- scale between 0 to 1
    - normalize -- normalize values using mean
    - standardize -- standardize using mean and standard deviation
    - None --  no scaling needed
2. y_scale_type = 'minmax'
    - minmax -- scale between 0 to 1
    - None -- No scaling needed
3. metric = 'mse' -- metric for evaluating model
    - mae --  L1 Mean Absolute Error
    - mse -- L2 Mean Squared Error
4. save_model_filename_prefix 
    - Save model's prefix
    - Leave it blank if you do not wish to save your model using this pipeline
5. validation_split = 0.3
    - Split ratio for training/validation set
6. fit_kwargs
    - Keyword arguments for your model's fit function
    - Only applicable if your model's fit function has arguments needed
    - Otherwise leave it blank
    
7. preprocess_fn
    - Add your own preprocess function. Leave it blank if\
    you are not using your own custom function.
    - Requires function to take dataframe as input and\
    output **dataframe, x, y**