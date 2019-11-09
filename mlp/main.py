from AISGPipeline import MLPipeline
import pandas as pd
from importlib import reload
import os
os.environ['x_shape_1'] = '-1'

if __name__ == '__main__':
    df = pd.read_csv('https://aisgaiap.blob.core.windows.net/aiap5-assessment-data/traffic_data.csv')
    model_pipeline = MLPipeline(df)

    import _mlconfig
    reload(_mlconfig)
    model_pipeline.add_model(_mlconfig.final_model)
    metrics = model_pipeline.fit(**_mlconfig.fit_kwargs)

