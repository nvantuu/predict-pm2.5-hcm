from .utils import *
from .model import *

from . import config as c

def main():
    parent_dir = c.parent_dir
    create_folders_for_output(parent_dir)

    data_dir = os.path.join(parent_dir, 'data', 'training_data', c.fname_data)
    df = load_data(data_dir=data_dir)
    # print(df.columns)

    # """ ================ PREPARE DATA TRAINING ================ """
    # X, y = generate_time_series_data(df, c.window_size, c.stride_pred)
    # X = normalize_data(X)
    #
    # # split data in to train/validation
    # x_train, y_train, x_test, y_test = split_data(X, y, c.train_ratio)
    #
    # """ ================== LSTM ====================== """
    # lstm = LTSMModel(params=c.lstm_params)
    # lstm.fit(X=x_train, y=y_train)
    #
    # y_pred1 = lstm.predict(x_test).flatten()
    #
    # """ ================= LGBM ======================== """
    # ws, fn = x_train[0].shape
    # lgbm = LightGBMModel(c.lgbm_params)
    # x_train, x_test = x_train.reshape(-1, ws * fn), x_test.reshape(-1, ws * fn)
    # lgbm.fit(X=x_train, y=y_train)
    # lgbm.save_model(c.lgbm_output)
    #
    # y_pred2 = lgbm.predict(x_test).flatten()
    #
    # """ ============== LSTM-TSLightGBM ================== """
    # w1, w2 = compute_weight_sharing(y_pred1, y_pred2, y_test)
    # y_pred = w1 * y_pred1 + w2 * y_pred2
    #
    # """ =================== METRIC ====================== """
    # df_metric = create_metrics_report_table(['LSTM', 'LightGBM', 'LSTM-TSLightGBM'],
    #                                         [y_pred1, y_pred2, y_pred], [y_test, y_test, y_test])
    # df_metric.to_csv(os.path.join(parent_dir, 'output', 'results', 'metrics.csv'), index=False)





    """ Tested on different parameters,
     Save only the metric of the combined model"""

    # stride_preds = [1, 2, 4, 8]
    # window_sizes = [4, 8, 10, 12, 16, 18, 24, 32]

    stride_preds = [1, 2]
    window_sizes = [4, 8]

    df_metrics = pd.DataFrame(columns=['Model', 'MAE', 'RMSE', 'r2', 'R'])

    for sp in stride_preds:
        for ws in window_sizes:
            auto_correct_config(window_size=ws, stride_pred=sp)

            """ ================ PREPARE DATA TRAINING ================ """
            X, y = generate_time_series_data(df, c.window_size, c.stride_pred)
            X = normalize_data(X)

            # split data in to train/validation
            x_train, y_train, x_test, y_test = split_data(X, y, c.train_ratio)

            """ ================== LSTM ====================== """
            lstm = LTSMModel(params=c.lstm_params)
            lstm.fit(X=x_train, y=y_train)

            y_pred1 = lstm.predict(x_test).flatten()


            """ ================= LGBM ======================== """
            _, fn = x_train[0].shape
            lgbm = LightGBMModel(c.lgbm_params)
            x_train, x_test = x_train.reshape(-1, ws * fn), x_test.reshape(-1, ws * fn)
            lgbm.fit(X=x_train, y=y_train)
            lgbm.save_model(c.lgbm_output)

            y_pred2 = lgbm.predict(x_test).flatten()


            """ ============== LSTM-TSLightGBM ================== """
            w1, w2 = compute_weight_sharing(y_pred1, y_pred2, y_test)
            y_pred = w1 * y_pred1 + w2 * y_pred2

            """ =================== METRIC ====================== """
            df_metric = create_metrics_report_table(['LSTM-TSLightGBM'],
                                                    [y_pred], [y_test])
            # df_metric.to_csv(os.path.join(parent_dir, 'output', 'results', c.unique_name + '.csv'), index=False)
            df_metrics = pd.concat([df_metrics, df_metric], axis=0, ignore_index=True)

    df_metrics.to_csv(os.path.join(parent_dir, 'output', 'results', 'metrics.csv'), index=False)

if __name__ == '__main__':
    main()
