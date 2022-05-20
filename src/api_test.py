import os
import argparse
import json

from numpy import copy
import keras
import lightgbm as lgb


from . import config as c
from .utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Main pipeline for predict PM2.5 Concentration HCM city')

    parser.add_argument('--model_name', type=str, default='2h-32T', help='model name')

    args = parser.parse_args()
    return args


def main():
    # parse command line arguments
    args = parse_args()

    # model default
    model_name = '2h-32T'
    model_name = args.model_name
    print(f'MODEL NAME: {model_name}')


    """ Prepare data for test api """
    sp = int(model_name[0])
    ws = int(model_name[3:len(model_name)-1])

    # read data in file .csv
    data_dir = os.path.join(c.parent_dir, 'data', 'training_data', c.fname_data)
    df = load_data(data_dir=data_dir)
    print(df.columns)

    # generate time series data
    X, y = generate_time_series_data(df, ws, sp)
    X = normalize_data(X)
    print("- Checking correctness of sample generation:")
    print(f" + the shape of a sample is {X[0].shape}, the correct shape is ({ws}, {9}))\n")

    _, fn = X[0].shape

    # split data in to train/validation
    x_train, y_train, x_test, y_test = split_data(X, y, c.train_ratio)
    # data for LSTM model
    x_test1 = copy(x_test)
    # data for LightGBM model
    x_test2 = copy(x_test).reshape(-1, ws * fn)


    """ =============== Load LSTM and Predict =================== """
    lstm_model_file_path = os.path.join(c.lstm_output, model_name + ".h5")
    print(lstm_model_file_path)
    lstm_model = keras.models.load_model(lstm_model_file_path)
    print('Successfully loaded lstm model')
    # predict x_test
    y_pred1 = lstm_model.predict(x_test1).flatten()


    """ ============ Load LightGBM and Predict ================== """
    lgbm_model_file_path = os.path.join(c.lgbm_output, model_name + ".txt")
    print(lgbm_model_file_path)
    lgbm_model = lgb.Booster(model_file=lgbm_model_file_path)
    print('Successfully loaded lgbm model')
    # predict x_test
    y_pred2 = lgbm_model.predict(x_test2).flatten()


    """ =============== Load weight sharing store in file config json ================= """
    file_config_path = os.path.join(c.config_dir, model_name + '.json')
    with open(file_config_path) as json_file:
        config_dict = json.load(json_file)

    model_params = config_dict['model_hyperparams']
    w1, w2 = model_params['w1'], model_params['w2']


    """ ================= LSTM-TSLightGBM predict ====================== """
    y_pred = w1 * y_pred1 + w2 * y_pred2


    """ ==================== Predict result============================== """
    # The results will be the same as the test results during the training process
    calculate_metrics(y_pred, y_test)

if __name__ == '__main__':
    main()











