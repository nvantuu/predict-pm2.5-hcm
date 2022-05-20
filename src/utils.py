import os
import numpy as np
import pandas as pd
import math

from datetime import datetime
from datetime import timedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import scipy
import matplotlib.pyplot as plt

import json

from . import config as c


def create_folders_for_output(parent_dir):
    """ create an output folder,
    to store all the outputs of the trained model:
    weights of models, metrics of models """


    def create_folder(folder_path):
        folder_path = os.path.abspath(folder_path)
        project_folder = os.path.dirname(os.getcwd())

        if os.path.commonpath([folder_path, project_folder]) != project_folder:
            return -1
        if folder_path == project_folder:
            return 0
        if os.path.isdir(folder_path):
            return 0
        create_folder(os.path.dirname(folder_path))
        os.mkdir(folder_path)
        return 0

    create_folder(os.path.join(parent_dir, 'output'))

    create_folder(os.path.join(parent_dir, 'output', 'config'))
    create_folder(os.path.join(parent_dir, 'output', 'img'))
    create_folder(os.path.join(parent_dir, 'output', 'lgbm_model'))
    create_folder(os.path.join(parent_dir, 'output', 'lstm_model'))
    create_folder(os.path.join(parent_dir, 'output', 'results'))

    create_folder(os.path.join(parent_dir, 'output', 'img', 'comp_graph'))
    create_folder(os.path.join(parent_dir, 'output', 'img', 'scatter_plot'))

    create_folder(os.path.join(parent_dir, 'output', 'lstm_model', 'loggers'))


def load_data(data_dir):
    """ read csv file data to dataframe """
    df = pd.read_csv(data_dir)
    return df


def generate_time_series_data(df, window_size, stride_pred):
    """ generate time series data with existing dataframe
    with number of data points in one sampling = window_size and
    prediction distance = stride_pred respectively """

    def decomposes_into_valid_sub_df_list():
        """ subdivide the parent df into a list of dataframes containing continuous data points
        discontinuous data points are adjacent points and the distance is > 1 hour """

        list_df = []
        mark = 0

        for i in range(1, len(df)):
            prev = datetime.strptime(df['time'][i - 1], '%Y-%m-%d %H:%S')
            curr = datetime.strptime(df['time'][i], '%Y-%m-%d %H:%S')

            if curr - prev != timedelta(hours=1):
                list_df.append(df.iloc[mark:i, :])
                mark = i

        return list_df

    # list of data frames containing continuous data
    sub_df_list = decomposes_into_valid_sub_df_list()
    # print(f'Before remove sub_df which has length <= `time_step+1`: {len(sub_df_list)}')

    # filter out a list of data frames of suitable length to sample the data
    sub_df_list = [ df_i for df_i in sub_df_list if len(df_i) >= window_size + stride_pred ]
    # print(f'After remove sub_df which has length >= `time_step+1`: {len(sub_df_list)}')


    # X_training, y_training list
    X, y = [], []

    # list columns of dataframe
    feature_num = len(c.features_list)

    for df_i in sub_df_list:
        # list contain all feature in dataframe, except for the feature: `time`
        df_i = df_i[c.features_list]
        # convert dataframe to numpy array
        df_i = df_i.values.reshape(-1, feature_num)
        # print('len df_i:', len(df_i))

        # create sample for data
        s, e = 0, len(df_i) - stride_pred - window_size + 1
        # print(s, " :", e)
        for i in range(s, e):
            X.append(df_i[i: i+window_size])
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
    """ Min Max Scaler for time series data in np.array format,
     have shape (length, window_size, feature_num) """
    # auto window_size, feature_num
    ws, fn = X[0].shape
    sc = MinMaxScaler()

    X = sc.fit_transform(X.reshape(-1, ws * fn))
    X = X.reshape(-1, ws, fn)
    return X


def split_data(X, y, train_ratio):
    split_point = round(len(X) * train_ratio)
    x_train, x_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    return x_train, y_train, x_test, y_test


def compute_weight_sharing(ytr_pred1, ytr_pred2, ytr_true):
    """ Compute the shared weights based on the results
    of the two models predicting the results of the training dataset """
    E1 = np.abs(ytr_pred1 - ytr_true)
    E2 = np.abs(ytr_pred2 - ytr_true)
    E = np.array([[E1.dot(E1), E1.dot(E2)], [E2.dot(E1), E2.dot(E2)]])

    R = np.array([1, 1])
    E_inv = np.linalg.pinv(E)

    w = E_inv.dot(R) / R.T.dot(E_inv).dot(R)
    w1, w2 = w
    return w1, w2

def store_weight_sharing(w1, w2):
    c.model_hyperparams['w1'] = w1
    c.model_hyperparams['w2'] = w2

def calculate_metrics(y_p, y_t):
    """ calculate metrics based on the predicted output and actual output of the testset """
    mae = mean_absolute_error(y_p, y_t)
    rmse = math.sqrt(mean_squared_error(y_p, y_t))
    r2 = r2_score(y_p, y_t)
    R = np.corrcoef(y_p, y_t)[0][1]
    print(f"Metrics of: \nmae = {round(mae,2)} \nrmse = {round(rmse,2)} \nr^2 = {round(r2,2)} \nR = {round(R,2)}\n")

    return round(mae,2), round(rmse,2), round(r2,2), round(R,2)


def create_metrics_report_table(model_names, y_preds, y_trues):
    """ calculate metrics and create a dataframe
    that stores metrics of the list of models """

    df = pd.DataFrame(columns=['Model', 'MAE', 'RMSE', 'r2', 'R'])

    for i in range(len(model_names)):
        model_names_i, y_pred_i = model_names[i], y_preds[i]
        mae_i, rmse_i, r2_i, R_i = calculate_metrics(y_pred_i, y_trues[i])
        model_names_i = model_names_i + '-' + c.unique_name
        df.loc[len(df.index)] = [model_names_i, mae_i, rmse_i, r2_i, R_i]

    df.loc[len(df.index)] = [None, None, None, None, None]
    # print(df)
    return df


def auto_correct_config(window_size, stride_pred, train_ratio=0.8, feature_list=None):
    """ automatically correct value in config.py file
    when window_size and stride_pred change value """
    if feature_list is not None:
        c.features_list = feature_list
        c.num_feature = len(feature_list)

    c.stride_pred = stride_pred
    c.window_size = window_size
    c.unique_name = str(stride_pred) + 'h-' + str(window_size) + 'T'

    c.model_hyperparams['window_size'] = window_size
    c.model_hyperparams['stride_pred'] = stride_pred
    c.model_hyperparams['unique_name'] = str(stride_pred) + 'h-' + str(window_size) + 'T'
    c.model_hyperparams['train_ratio'] = train_ratio
    return

def draw_comparison_graph(y_pred, y_true):
    plt.plot(y_pred[:200], color='r')
    plt.plot(y_true[:200], color='b')
    plt.title(f'LSTM-TSLightGBM-{c.unique_name} predict of PM2.5 concentration')
    plt.xlabel("Time")
    plt.ylabel("ug/m3")
    plt.legend(('prediction', 'reality'), loc='upper right')
    plt.savefig(os.path.join(c.img_output, 'comp_graph', c.unique_name + '.png'))
    plt.clf()


def draw_scatter_plot(y_pred, y_true):
    slope, intercept, r, p, stderr = scipy.stats.linregress(y_pred, y_true)
    line = f'Regression line y:{intercept:.2f}+{slope:.2f}, r={r:.2f}'

    fig, ax = plt.subplots()
    ax.plot(y_pred, y_true, linewidth=0, marker='s', label='Data points')
    ax.plot(y_pred, intercept + slope * y_pred, label=line)
    ax.set_xlabel('y_pred')
    ax.set_ylabel('y_true')
    ax.legend(facecolor='white')
    plt.savefig(os.path.join(c.img_output, 'scatter_plot', c.unique_name + '.png'))
    plt.clf()


def store_model_configuration():
    config_dict = {'model_hyperparams': c.model_hyperparams, 'lstm_params': c.lstm_params, 'lgbm_params': c.lgbm_params}

    file_config_dir = os.path.join(c.config_dir, c.unique_name + '.json')
    with open(file_config_dir, 'w') as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=4)
















