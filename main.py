from mlp.AISGPipeline import MLPipeline
import pandas as pd
from _mlconfig import final_model

df = pd.read_csv('https://aisgaiap.blob.core.windows.net/aiap5-assessment-data/traffic_data.csv')
valid_split = 0.7
model_pipeline = MLPipeline(df, valid_split, final_model)
metrics = model_pipeline.fit()

