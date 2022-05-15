import os
from datetime import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import lightgbm as lgb

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow import keras
import tensorflow as tf

import config as c


class LightGBMModel:
    def __init__(self, params):
        self.model = lgb.LGBMModel(**params)

    def fit(self, X, y):
        self.model = self.model.fit(X=X, y=y)

    def predict(self, xte):
        return self.model.predict(xte)

    def save_model(self, path):
        time_now = str(c.hop_size) + 'h ' + datetime.now().strftime('%Y-%m-%d %H-%M')
        path = os.path.join(path, 'models', time_now + ".txt")
        self.model.booster_.save_model(path)


class LTSMModel:
    def __init__(self, params):
        self.params = params

        self.model = Sequential()

        self.model.add(LSTM(units=128,
                            input_shape=(self.params['window_size'], self.params['feature_num']),
                            return_sequences=True))
        self.model.add(LSTM(units=128, input_shape=(self.params['window_size'], 128)))
        self.model.add(Dense(units=320, activation="relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=160, activation="relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1, activation="linear"))

        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=self.params['lr']),
                      loss='mse',
                      metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])

        print(self.model.summary())

    def scheduler(self, epoch, lr):
        epochs_drop = self.params['epochs_drop']
        drop_rate = self.params['drop_rate']

        print(f"epoch: {epoch:03d} - learning_rate: {lr:.05f}")
        if epoch % epochs_drop == 0 and epoch > 0:
            return lr * drop_rate
        else:
            return lr

    def fit(self, X, y):
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(self.scheduler)

        time_now = str(c.hop_size) + 'h ' + datetime.now().strftime('%Y-%m-%d %H-%M')

        log_dir = os.path.join(self.params['output'], 'loggers', time_now)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Save the best model to .h5 file
        model_path = os.path.join(self.params['output'], 'models', time_now + '.h5')
        best_model = keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                                     save_best_only=True, verbose=self.params['verbose'])

        self.model.fit(x=X, y=y,
                       epochs=self.params['epochs'],
                       validation_split=self.params['validation_split'],
                       batch_size=self.params['batch_size'],
                       callbacks=[best_model, tensorboard_callback, lr_scheduler])

    def predict(self, xte):
        return self.model.predict(xte)
















