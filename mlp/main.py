from AISGPipeline import MLPipeline
import pandas as pd
import numpy as np
from importlib import reload
import os
os.environ['x_shape_1'] = '-1'
import _mlconfig

if __name__ == '__main__':
    print('Starting Pipeline...')

    df = pd.read_csv('https://aisgaiap.blob.core.windows.net/aiap5-assessment-data/traffic_data.csv')
    model_pipeline = MLPipeline(df)

    reload(_mlconfig)
    model_pipeline.add_model(_mlconfig.final_model)
    metrics = model_pipeline.fit(**_mlconfig.fit_kwargs)
    if _mlconfig.save_model_filename_prefix != '':
        filename = _mlconfig.save_model_filename_prefix + '_' + str(metrics[0]) + '.npy'
        np.save(filename, model_pipeline.model)
