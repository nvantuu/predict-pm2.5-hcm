from utils import *
from model import *
import src.config as c

def main():
    parent_dir = c.parent_dir
    data_dir = os.path.join(parent_dir, 'data', 'training_data', c.fname_data)


    df = load_data(data_dir=data_dir)
    # print(df.columns)


    X, y = generate_time_series_data(df, c.window_size, c.hop_size)
    X = normalize_data(X)


    # split data in to train/validation
    x_train, y_train, x_test, y_test = split_data(X, y, c.train_ratio)


    """ ================== LSTM ====================== """
    lstm = LTSMModel(params=c.lstm_params)
    lstm.fit(X=x_train, y=y_train)
    y_pred1 = lstm.predict(x_test).flatten()



    """ ================= LGBM ======================== """
    window_size, feature_num = x_train[0].shape
    lgbm = LightGBMModel(c.lgbm_params)
    x_train, x_test = x_train.reshape(-1, window_size * feature_num), x_test.reshape(-1, window_size * feature_num)
    lgbm.fit(X=x_train, y=y_train)
    lgbm.save_model(c.lgbm_output)
    y_pred2 = lgbm.predict(x_test).flatten()


    """ ============== LSTM-TSLightGBM ================== """
    w1, w2 = compute_weight_sharing(y_pred1, y_pred2, y_test)
    y_pred = w1 * y_pred1 + w2 * y_pred2


    """ =================== METRIC ====================== """
    create_metrics_report_table(['LSTM', 'LightGBM', 'LSTM-TSLightGBM'],
                                [y_pred1, y_pred2, y_pred], y_test)



if __name__ == '__main__':
    main()
