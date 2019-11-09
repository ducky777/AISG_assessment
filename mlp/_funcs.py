import pandas as pd

def feature_engineering(df):

    # Convert date_time into DayOfWeek, Day, Month, Hour
    df.date_time = pd.to_datetime(df.date_time)
    hour = df.date_time.dt.hour
    dayofweek = df.date_time.dt.dayofweek
    day = df.date_time.dt.day
    month = df.date_time.dt.month

    df['Month'] = month.copy()
    df['Hour'] = hour.copy()
    df['DayOfWeek'] = dayofweek.copy()
    df['Day'] = day.copy()

    # Dropping holiday and date_time
    df = df.drop('holiday', axis=1)
    df = df.drop('date_time', axis=1)
    return df

