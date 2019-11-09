import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import r2_score

class MLPipeline:
    def __init__(self, df, valid_split):
        self.x_mean = None
        self.x_min = None
        self.x_range = None
        self.x_std = None
        self.y_min = None
        self.y_range = None
        self.x_idx = None

        self.valid_split = valid_split
        self.valid_idx = int(round(valid_split*len(df)))

        self.df = self.feature_engineering(df)
        self.weather_main_numerical, self.weather_main_categories = \
            self._convert_categorical(self.df.weather_main)
        self.weather_main_onehot = self._convert_onehot(self.df.weather_main)
        self.weather_desc_numerical, self.weather_desc_categories = \
            self._convert_categorical(self.df.weather_description)
        self.weather_desc_onehot = self._convert_onehot(self.df.weather_description)

        self.weather_categories_idx = self.weather_main_categories + self.weather_desc_categories
        self.weather_categories_idx = self.weather_categories_idx + ['Temperature', 'Rain_1h', 'Snow_1h'
                                                                     , 'Clouds_all', 'Hour', 'Month'
                                                                     , 'Day Of Week', 'Day']

        self.x = self._create_x()

        self._create_y()
        self.x_train = self.x[:self.valid_idx]
        self.y_train = self.y[:self.valid_idx]

        self.x_train, self.x_idx = self._check_data(self.x_train)
        self.x = self.x[:, self.x_idx]

        self.save_settings()

        from _mlconfig import x_scale_type

        self.x_scale(x_scale_type)
        self.y_scale()

        self.x_valid = self.x[self.valid_idx:]
        self.y_valid = self.y[self.valid_idx:]

        self.model = None
        return

    def _check_data(self, x):
        print('Checking and removing variables with 0 variance')
        x_idx = []
        for i in range(x.shape[1]):
            xmin = x[:, i].min()
            xmax = x[:, i].max()
            if xmax - xmin != 0:
                x_idx.append(i)
        rx = x[:, x_idx]
        return rx, x_idx

    def x_scale(self, x_scale_type):
        if x_scale_type == 'minmax':
            self.x_train, self.x_min, self.x_range = self.minmax_scaling(self.x_train)
            np.savetxt('xrange.txt', self.x_range)
            self.x, dummy1, dummy2 = self.minmax_scaling(self.x, self.x_min, self.x_range)
        elif x_scale_type == 'normalize':
            self.x_train, self.x_mean, self.x_range = self.normalize(self.x_train)
            self.x, dummy1, dummy2 = self.normalize(self.x, self.x_mean, self.x_range)
        elif x_scale_type == 'standardize':
            self.x_train, self.x_mean, self.x_std = self.standardize(self.x_train)
            self.x, dummy1, dummy2 = self.standardize(self.x, self.x_mean, self.x_std)
        return

    def y_scale(self):
        self.y_train, self.y_min, self.y_range = self.minmax_scaling(self.y_train)
        self.y, dummy1, dummy2 = self.minmax_scaling(self.y, self.y_min, self.y_range)
        return

    def save_settings(self):
        f = open('save.txt', 'w')
        f.write(str(self.x.shape[1]))
        f.close()
        # os.environ['x_shape_1'] = str(self.x.shape[1])
        return

    def add_model(self, model):
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

        df = df.drop('holiday', axis=1)
        df = df.drop('date_time', axis=1)
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
        self.y = self.y.reshape(len(self.y), 1)
        return

    def minmax_scaling(self, x, xmin0=None, xrange0=None):
        x_min = []
        x_range = []
        rx = np.zeros_like(x)
        for i in range(x.shape[1]):
            if xmin0 is None:
                xmin = x[:, i].min()
                xmax = x[:, i].max()
                xrange = xmax - xmin
                x_min.append(xmin)
                x_range.append(xrange)
            else:
                xmin = xmin0[i]
                xrange = xrange0[i]
            rx[:, i] = (x[:, i]-xmin)/xrange
            if xrange == 0:
                print(i)
        return rx, x_min, x_range

    def normalize(self, x, xmean0=None, xrange0=None):
        x_mean = []
        x_range = []
        rx = np.zeros_like(x)
        for i in range(x.shape[1]):
            if xmean0 is None:
                xmin = x[:, i].min()
                xmax = x[:, i].max()
                xrange = xmax-xmin
                xmean = x[:, i].mean()
                x_mean.append(xmean)
                x_range.append(xrange)
            else:
                xmean = xmean0[i]
                xrange = xrange0[i]
            rx[:, i] = (x[:, i]-xmean)/xrange
        return rx, x_mean, x_range

    def standardize(self, x, xmean0=None, xstd0=None):
        x_mean = []
        x_std = []
        rx = np.zeros_like(x)
        for i in range(x.shape[1]):
            if xmean0 is None:
                xmean = x[:, i].mean()
                xstd = x[:, i].std()
                x_mean.append(xmean)
                x_std.append(xstd)
            else:
                xmean = xmean0[i]
                xstd = xstd0[i]
            rx[:, i] = (x[:, i] - xmean) / xstd
        return rx, x_mean, x_std

    def mse(self, y_true, y_pred):
        return np.mean(np.square(y_pred-y_true))

    def mae(self, y_true, y_pred):
        return np.mean(np.abs(y_pred-y_true))

    def r2(self, y_true, y_pred):
        return r2_score(y_true, y_pred)

    def evaluate(self, y_true, y_pred):
        
        return

    def fit(self, **kwargs):
        self.model.fit(self.x_train, self.y_train, **kwargs)
        pr = self.model.predict(self.x)
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
