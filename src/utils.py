import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


from datetime import datetime
from datetime import timedelta

from . import config as c


def load_data(data_dir):
    df = pd.read_csv(data_dir)
    return df


def generate_time_series_data(df, window_size, stride_pred):
    # except for the feature: `time`
    feature_num = len(df.columns) - 1

    def decomposes_into_valid_sub_df_list():
        list_df = []
        mark = 0

        for i in range(1, len(df)):
            prev = datetime.strptime(df['time'][i - 1], '%Y-%m-%d %H:%S')
            curr = datetime.strptime(df['time'][i], '%Y-%m-%d %H:%S')

            if curr - prev != timedelta(hours=1):
                list_df.append(df.iloc[mark:i, :])
                mark = i

        return list_df

    sub_df_list = decomposes_into_valid_sub_df_list()
    # print(f'Before remove sub_df which has length <= `time_step+1`: {len(sub_df_list)}')

    sub_df_list = [ df_i for df_i in sub_df_list if len(df_i) >= window_size + stride_pred ]
    # print(f'After remove sub_df which has length >= `time_step+1`: {len(sub_df_list)}')


    X, y = [], []

    for df_i in sub_df_list:
        df_i = df_i[['temperature','dewpoint_temperature','pressure','humidity','wind_speed','wind_direction','vision','clouds','PM25_Concentration']]
        # convert dataframe to numpy array
        df_i = df_i.values.reshape(-1, feature_num)
        # print('len df_i:', len(df_i))

        s, e = 0, len(df_i) - stride_pred - window_size + 1
        # print(s, " :", e)

        for i in range(s, e):
            X.append(df_i[i: i+12])
            y.append(df_i[i+window_size+stride_pred-1][-1])

    X, y = np.array(X), np.array(y)

    # # check shape X, y
    # print(f'Shape: X{X.shape}, y{y.shape}')
    # # check first sample ~ first column in csv
    # print(X[0][-1], y[0])
    # # check last sample ~ last column in csv
    # print(X[-1][-1], y[-1])

    return X, y

def normalize_data(X):
    window_size, feature_num = X[0].shape
    sc = MinMaxScaler()

    X = sc.fit_transform(X.reshape(-1, window_size * feature_num))
    X = X.reshape(-1, window_size, feature_num)

    return X


def split_data(X, y, train_ratio):

    split_point = round(len(X) * train_ratio)
    x_train, x_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    return x_train, y_train, x_test, y_test


def compute_weight_sharing(y_pred1, y_pred2, y_true):

    print(y_pred1.shape, y_pred2.shape)

    E1 = np.abs(y_pred1 - y_true)
    E2 = np.abs(y_pred2 - y_true)
    E = np.array([[E1.dot(E1), E1.dot(E2)], [E2.dot(E1), E2.dot(E2)]])

    R = np.array([1, 1])
    E_inv = np.linalg.pinv(E)

    w = E_inv.dot(R) / R.T.dot(E_inv).dot(R)

    w1, w2 = w
    return w1, w2

def calculate_metrics(y_p, y_t):
    mae = mean_absolute_error(y_p, y_t)
    rmse = math.sqrt(mean_squared_error(y_p, y_t))
    r2 = r2_score(y_p, y_t)
    R = np.corrcoef(y_p, y_t)[0][1]
    # print(f"Metrics of {model_name}: \nmae = {mae} \nrmse = {rmse} \nr^2 = {r2} \nR = {R}\n")

    return mae, rmse, r2, R


def create_metrics_report_table(model_names, y_preds, y_true):

    df = pd.DataFrame(columns=['Model', 'MAE', 'RMSE', 'r2', 'R'])

    for i in range(len(model_names)):
        model_names_i, y_pred_i = model_names[i], y_preds[i]
        mae_i, rmse_i, r2_i, R_i = calculate_metrics(y_pred_i, y_true)

        model_names_i = model_names_i + '-' + str(c.stride_pred) + 'h'

        df.loc[len(df.index)] = [model_names_i, mae_i, rmse_i, r2_i, R_i]

    print(df)
    return df








