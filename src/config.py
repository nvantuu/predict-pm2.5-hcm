import os

parent_dir = os.getcwd()

lgbm_output = os.path.join(parent_dir, 'output', 'lgbm_model')
lstm_output = os.path.join(parent_dir, 'output', 'lstm_model')
img_output = os.path.join(parent_dir, 'output', 'img')


# process data config
fname_data = "clean_data.csv"
features_list = ['temperature','dewpoint_temperature','pressure','humidity',
                'wind_speed','wind_direction','vision','clouds','PM25_Concentration']
num_feature = len(features_list)


window_size = 8                     # length in one sample
stride_pred = 2                     # stride of prediction
train_ratio = 0.8

# get unique
unique_name = str(stride_pred) + 'h-' + str(window_size) + 'T'

# lightgbm config
lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'max_depth': 10,
    'learning_rate': 0.01,
    'n_estimators': 700,
    'verbose': 1,
    'force_col_wise': True
}


# LSTM config
lstm_params = {
    'validation_split': 0.25,
    'epochs': 40,
    'lr': 0.01,
    'drop_rate': 0.5,
    'epochs_drop': 5,
    'verbose': 1,
    'batch_size': 32,
    'output': os.path.join(parent_dir, 'output', 'lstm_model')
}






