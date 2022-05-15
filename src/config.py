import os

parent_dir = os.path.dirname(os.getcwd())
lgbm_output = os.path.join(parent_dir, 'output', 'lgbm_model')
lstm_output = os.path.join(parent_dir, 'output', 'lstm_model')


# process data config
fname_data = "clean_data.csv"
window_size = 12
hop_size = 6
train_ratio = 0.8


# lightgbm config
lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'max_depth': 10,
    'learning_rate': 0.01,
    'n_estimators': 1000,
    'verbose': 0,
    'force_col_wise': True
}


# LSTM config
lstm_params = {
    'validation_split': 0.25,
    'window_size': 12,
    'feature_num': 9,
    'epochs': 2,
    'lr': 0.01,
    'drop_rate': 0.5,
    'epochs_drop': 20,
    'verbose': 1,
    'batch_size': 32,
    'output': os.path.join(parent_dir, 'output', 'lstm_model')
}






