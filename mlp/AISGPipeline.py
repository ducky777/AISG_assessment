import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import r2_score

class MLPipeline:
    def __init__(self, df, valid_split, model):
        self.valid_split = valid_split
        self.valid_idx = int(round((1-valid_split)*len(df)))

        self.df = self.feature_engineering(df)
        self.weather_main_numerical, self.weather_main_categories = \
            self._convert_categorical(self.df.weather_main)
        self.weather_main_onehot = self._convert_onehot(self.df.weather_main)
        self.weather_desc_numerical, self.weather_desc_categories = \
            self._convert_categorical(self.df.weather_description)
        self.weather_desc_onehot = self._convert_onehot(self.df.weather_description)
        self.x = self._create_x()
        self.weather_categories_idx = self.weather_main_categories + self.weather_desc_categories
        self.weather_categories_idx = self.weather_categories_idx + ['Temperature', 'Rain_1h', 'Snow_1h'
                                                                     , 'Clouds_all', 'Hour', 'Month'
                                                                     , 'Day Of Week', 'Day']
        self._create_y()
        self.x_train = self.x[:self.valid_idx]
        self.y_train = self.y[:self.valid_idx]
        self.x_valid = self.x[self.valid_idx:]
        self.y_valid = self.y[self.valid_idx:]
        self.model = model
        return

    def feature_engineering(self, df):
        df.date_time = pd.to_datetime(df.date_time)
        hour = df.date_time.dt.hour
        dayofweek = df.date_time.dt.dayofweek
        day = df.date_time.dt.day
        month = df.date_time.dt.month

        df['Month'] = month.copy()
        df['Hour'] = hour.copy()
        df['DayOfWeek'] = dayofweek.copy()
        df['Day'] = day.copy()
        return df

    def _convert_categorical(self, x):
        # =======================================
        # Convert data into categorical numerics
        # =======================================
        x = np.array(x)
        categories = list(set(x))
        rx = []
        for i in x:
            rx.append(categories.index(i))
        return rx, categories

    def _convert_onehot(self, x):
        # =======================================
        # Convert data into onehot encoding
        # =======================================
        x = np.array(x)
        categories = list(set(x))
        rx = np.zeros((len(x), len(categories)))
        for i in range(len(rx)):
            idx = categories.index(x[i])
            rx[i, idx] = 1
        return rx

    def _create_x(self):
        temp = np.array(self.df.temp)
        rain1h = np.array(self.df.rain_1h)
        snow1h = np.array(self.df.snow_1h)
        clouds = np.array(self.df.clouds_all)
        weathermain = self.weather_main_onehot.copy()
        weatherdesc = self.weather_desc_onehot.copy()
        hour = np.array(self.df.Hour)
        month = np.array(self.df.Month)
        dayofweek = np.array(self.df.DayOfWeek)
        day = np.array(self.df.Day)
        x = np.stack((temp, rain1h
                      , snow1h, clouds, hour, month, dayofweek
                      , day), axis=1)
        x = np.hstack((weathermain, weatherdesc, x))
        return x

    def _create_y(self):
        self.y = np.array(self.df.traffic_volume)
        return

    def minmax_scaling(self, x):
        x_min = []
        x_range = []
        rx = np.zeros_like(x)
        for i in range(x.shape[1]):
            xmin = x[:, i].min()
            xmax = x[:, i].max()
            xrange = xmax-xmin
            x_min.append(xmin)
            x_range.append(xrange)
            rx[:, i] = (x[:, i]-xmin)/xrange
        return rx, x_min, x_range

    def normalize(self, x):
        x_mean = []
        x_range = []
        rx = np.zeros_like(x)
        for i in range(x.shape[1]):
            xmin = x[:, i].min()
            xmax = x[:, i].max()
            xrange = xmax-xmin
            xmean = x[:, i].mean()
            x_mean.append(xmean)
            x_range.append(xrange)
            rx[:, i] = (x[:, i]-xmean)/xrange
        return rx, x_mean, x_range

    def standardize(self, x):
        x_mean = []
        x_std = []
        rx = np.zeros_like(x)
        for i in range(x.shape[1]):
            xmean = x[:, i].mean()
            xstd = x[:, i].std()
            x_mean.append(xmean)
            x_std.append(xstd)
            rx[:, i] = (x[:, i] - xmean) / xstd
        return rx, x_mean, x_std

    def mse(self, y_true, y_pred):
        return np.mean(np.square(y_pred-y_true))

    def mae(self, y_true, y_pred):
        return np.mean(np.abs(y_pred-y_true))

    def r2(self, y_true, y_pred):
        return r2_score(y_true, y_pred)

    def fit(self):
        self.model.fit(self.x_train, self.y_train)
        pr = self.model.predict(self.x, self.y)
        mse_valid = np.mean(np.square(self.y_valid-pr[self.valid_idx:]))
        mse_train = np.mean(np.square(self.y_train-pr[:self.valid_idx]))
        mse_total = np.mean(np.square(self.y-pr))
        metrics = [mse_total, mse_train, mse_valid]
        print('MSE Valid Set: ', mse_valid, sep='')
        print('MSE Train Set: ', mse_train, sep='')
        print('MSE Total: ', mse_total, sep='')
        return metrics

    def reverse_minmax(self):
        return

    def create_lookbacks(self):
        return

    def feature_importance(self):
        clf = ExtraTreesClassifier(n_estimators=5).fit(self.x, self.y)
        f_importance = clf.feature_importances_
        importance_idx = np.argsort(f_importance)[::-1]
        return f_importance, importance_idx
