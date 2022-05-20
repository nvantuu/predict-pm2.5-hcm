from .utils import *
from .model import *

from . import config as c

def main():
    # get global path to this folders `predict-pm2.5-hcm`
    parent_dir = c.parent_dir
    create_folders_for_output(parent_dir)

    # read training data to dataframe
    data_dir = os.path.join(parent_dir, 'data', 'training_data', c.fname_data)
    df = load_data(data_dir=data_dir)
    # print(df.columns)

    # """ ================ PREPARE DATA TRAINING ================ """
    # X, y = generate_time_series_data(df, c.window_size, c.stride_pred)
    # X = normalize_data(X)
    # print("Checking correctness of sample generation ::")
    # print(f"The shape of a sample is {X[0].shape}, the correct shape is ({ws}, {c.num_feature}))")
    #
    # # split data in to train/validation
    # x_train, y_train, x_test, y_test = split_data(X, y, c.train_ratio)
    #
    # """ ================== LSTM ====================== """
    # lstm = LTSMModel(params=c.lstm_params)
    # lstm.fit_and_save(X=x_train, y=y_train)
    #
    # # predict testset using the trained LSTM model
    # y_pred1 = lstm.predict(x_test).flatten()
    #
    # # store predict x_train, using for compute weight sharing of two model
    # ytr_pred1 = lstm.predict(x_train).flatten()
    #
    # """ ================= LGBM ======================== """
    # # prepare data for lgbm: flatten data point
    # _, fn = x_train[0].shape
    # lgbm = LightGBMModel(c.lgbm_params)
    # x_train, x_test = x_train.reshape(-1, ws * fn), x_test.reshape(-1, ws * fn)
    #
    # lgbm.fit(X=x_train, y=y_train)
    # lgbm.save_model(c.lgbm_output)
    #
    # # predict testset using the trained LGBM model
    # y_pred2 = lgbm.predict(x_test).flatten()
    #
    # # store predict x_train, using for compute weight sharing of two model
    # ytr_pred2 = lgbm.predict(x_train).flatten()
    #
    # """ ============== LSTM-TSLightGBM ================== """
    # w1, w2 = compute_weight_sharing(ytr_pred1, ytr_pred2, y_train)
    # y_pred = w1 * y_pred1 + w2 * y_pred2
    #
    # """ =================== METRIC ====================== """
    # df_metric = create_metrics_report_table(['LSTM', 'LightGBM', 'LSTM-TSLightGBM'],
    #                                         [y_pred1, y_pred2, y_pred], [y_test] * 3)
    # df_metric.to_csv(os.path.join(parent_dir, 'output', 'results', 'metrics.csv'), index=False)




    """ Tested on different hyperparams: stride_pred and window_size,
     Save only the metric of the combined model"""

    # stride_preds = [1, 2]
    # window_sizes = [4, 8]

    stride_preds = [1, 2, 4, 8]
    window_sizes = [4, 8, 10, 12, 16, 18, 24, 32]

    df_metrics = pd.DataFrame(columns=['Model', 'MAE', 'RMSE', 'r2', 'R'])

    for sp in stride_preds:
        for ws in window_sizes:

            # correct config: all config variables related to
            # `stride_pred` = `sp` and `window_size` =  `ws`
            auto_correct_config(window_size=ws, stride_pred=sp)
            print(f'- MODEL NAME: {c.unique_name}\n')


            """ ================ PREPARE DATA TRAINING ================ """
            X, y = generate_time_series_data(df, c.window_size, c.stride_pred)
            X = normalize_data(X)
            print("- Checking correctness of sample generation:")
            print(f" + the shape of a sample is {X[0] .shape}, the correct shape is ({ws}, {c.num_feature}))\n")

            # split data in to train/validation
            x_train, y_train, x_test, y_test = split_data(X, y, c.train_ratio)
            print(f'- All {len(X)} samples:')
            print(f' + train test split rate: {c.train_ratio}/{round(1-c.train_ratio, 2)}\n')


            """ ================== LSTM ====================== """
            lstm = LTSMModel(params=c.lstm_params)
            lstm.fit_and_save(X=x_train, y=y_train, path_save=c.lstm_output)

            # store predict x_train, using for compute weight sharing of two model
            ytr_pred1 = lstm.predict(x_train).flatten()

            # predict testset using the trained LSTM model
            y_pred1 = lstm.predict(x_test).flatten()


            """ ================= LGBM ======================== """
            # prepare data for lgbm: flatten data point
            _, fn = x_train[0].shape
            x_train, x_test = x_train.reshape(-1, ws * fn), x_test.reshape(-1, ws * fn)

            lgbm = LightGBMModel(c.lgbm_params)
            lgbm.fit(X=x_train, y=y_train)
            lgbm.save_model(path_save=c.lgbm_output)

            # store predict x_train, using for compute weight sharing of two model
            ytr_pred2 = lgbm.predict(x_train).flatten()

            # predict testset using the trained LGBM model
            y_pred2 = lgbm.predict(x_test).flatten()


            """ ============== LSTM-TSLightGBM ================== """
            w1, w2 = compute_weight_sharing(ytr_pred1, ytr_pred2, y_train)
            store_weight_sharing(w1, w2)
            store_model_configuration()


            """ ========== Predict PM2.5 Concentration ========== """
            # predict testset of LSTM-TSLightGBM model
            y_pred = w1 * y_pred1 + w2 * y_pred2

            # graphing the results visualization
            draw_comparison_graph(y_pred, y_test)
            draw_scatter_plot(y_pred, y_test)


            """ =================== METRIC ====================== """
            df_metric = create_metrics_report_table(['LSTM', 'LightGBM', 'LSTM-TSLightGBM'],
                                                    [y_pred1, y_pred2, y_pred], [y_test]*3)
            # df_metric.to_csv(os.path.join(parent_dir, 'output', 'results', c.unique_name + '.csv'), index=False)
            df_metrics = pd.concat([df_metrics, df_metric], axis=0, ignore_index=True)

    # save metrics of hyper models to file.csv
    df_metrics.to_csv(os.path.join(parent_dir, 'output', 'results', 'metrics.csv'), index=False)

if __name__ == '__main__':
    main()
