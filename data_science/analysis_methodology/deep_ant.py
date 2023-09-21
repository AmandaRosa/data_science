#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 7:26:28 2020

@author: tuliotorezan
"""
import os

import numpy as np
import pandas as pd
import tensorflow.keras.models as models
from numpy import array
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from tensorflow.keras.models import Sequential

from ..preprocessing_data.FFT import *


class DeepAnT_CNN:
    """Convolutional Neural Network (CNN) class to predict the signal captured by sensors, based on the last signal window. It compares the original signal, with the CNN's prediction. The data is acquired as "blocks", and compared between each other. The variation among the reconstruction error of each block will be used to decide whether the behavior of the signal is normal or anomalous.

    The higher the variation in the reconstruction error of consecutive samples, higher the variation in the signal waveform from the sensors.

    The input domain used is frequency.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
        Default is 8.
    threshold_calibration_window : int
        Number of samples which will be used for calibrating the threshold.
        Default is 300.
    model : DeepAnT_CNN object, optional
        An already trained model of this class.
        If this parameter is not passed, the class will create and train a new model.
    threshold : numpy.ndarray, optional
        Array with the threshold limit for the classification.
        If this parameter is not passed, the class will set the threshold by itself.
    """

    NAME = "deep_ant"

    def __init__(
        self,
        epochs=8,
        threshold_calibration_window=300,
        model=None,
        threshold=None,
    ):
        self.p_w = None
        self.n_features = 1
        self.sensors_group = 1
        self.num_sensors = None

        self.anomaly_neighborhood = 8  # Number of neighboring windows to wich the current one is compared, minimum is 3

        if self.anomaly_neighborhood < 3:
            self.anomaly_neighborhood = 5

        ### Hyperparameters
        self.kernel_size = 2  # Size of filter in conv layers
        self.num_filt_1 = 32  # Number of filters in first conv layer
        self.num_filt_2 = 32  # Number of filters in second conv layer
        self.num_nrn_dl = 40  # Number of neurons in dense layer
        self.conv_strides = 1
        self.pool_size_1 = 2  # Length of window of pooling layer 1
        self.pool_size_2 = 2  # Length of window of pooling layer 2
        self.pool_strides_1 = 2  # Stride of window of pooling layer 1
        self.pool_strides_2 = 2  # Stride of window of pooling layer 2
        self.epochs = epochs
        self.dropout_rate = 0.5
        self.learning_rate = 2e-5

        tr_base = 1.1  # multiplier over the threshold set during training to be the base for the threshold on anomaly detection (i.e. for tr_base of 1.2, the threshold will be based on 120% of the highest score value observed during the threshold calibration process)

        self.signal_buffer = np.array([None, None, None])  # np.array for ease of use
        self.is_MAE_initialized = False
        self.MAE_buffer = list()  # standard python list to use append and pop methods
        self.moving_avg_buffer = [
            None  # standard python list to use append and pop methods
        ]
        self.do_moving_avg = True
        self.moving_avg_size = 4  # size of the moving average window

        self.last_anomaly_score = 0
        self.last_sample_threshold = 0
        self.last_sample_anomaly = True

        self.threshold = []
        self.threshold_const = 0

        if self.do_moving_avg:
            self.tr_base = 1 + (tr_base - 1) / 2
        else:
            self.tr_base = tr_base

        self.threshold_mult = 0.2
        self.threshold_calibration_window = threshold_calibration_window
        self.tr_aux = 0
        self.threshold_smooth = 3  ##5

        if threshold is not None:
            self.set_threshold(threshold)

        if model is not None:
            self.model = model
            self.flag_load_model = True
        else:
            self.flag_load_model = False

        self.flag_first_sample = True

    def _set_pw(self, p_w):
        """Set the parameters based on the acquired sample shape.

        Parameters
        ----------
        p_w : float
            Number of points of the signal

        Attributes
        ----------
        self.w : float
            History window (number of time stamps taken into account)
        self.p_w: int
            Prediction window (number of time stampes required to be predicted)
        self.num_nrn_ol : int
            Number of neurons in output dense layer
        """

        self.w = p_w
        self.p_w = int(p_w / 2)
        self.num_nrn_ol = self.p_w

    def _split_training_sequence(self, sequence):
        """Splits the given array based on the parameters defined in the `_set_pw` method.

        Parameters
        ----------
        sequence : numpy.ndarray
            Array to be splitted.

        Returns
        -------
        x : numpy.ndarray
            Array with the data to be passed as an input to the CNN layers.
        y : numpy.ndarray
            Array with the data to be passed as an output to the CNN layers.
        """

        x, y = list(), list()

        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + self.w
            out_end_ix = end_ix + self.p_w
            # check if we are beyond the sequence
            if out_end_ix > len(sequence):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
            x.append(seq_x)
            y.append(seq_y)

        return array(x), array(y)

    def _initialize_neural_network(self):  # (self, time, signal)
        """Assembles the CNN and starts the training process. The model is defined based on the parameters in the constructor of the class.

        The model is passed to the attribute self.model.
        """
        single_signal = np.concatenate(
            (
                self.signal_buffer[0][0],
                self.signal_buffer[1][0],
                self.signal_buffer[2][0],
            )
        )

        ##Even if we are working with many sensors, the training is done with data from a single one,
        # its not necessary to train for every sensor as it would also increase the chances of overfitting
        df_sig = pd.DataFrame({"signal": single_signal})

        ###Data preprocessing
        # split a univariate sequence into samples
        # define input sequence
        raw_seq = list(df_sig["signal"])

        # split into samples
        batch_sample, batch_label = self._split_training_sequence(raw_seq)

        # need to convert batch into 3D tensor of the form [batch_size, input_seq_len, n_features]
        batch_sample = batch_sample.reshape(
            (batch_sample.shape[0], batch_sample.shape[1], self.n_features)
        )

        ###Generate model for the predictor
        model = Sequential()

        # Convolutional Layer #1
        # Computes 32 features using a 1D filter(kernel) of with w with ReLU activation.
        # Padding is added to preserve width.
        # Input Tensor Shape: [batch_size, w, 1] / batch_size = len(batch_sample)
        # Output Tensor Shape: [batch_size, w, num_filt_1] (num_filt_1 = 32 feature vectors)
        model.add(
            Conv1D(
                filters=self.num_filt_1,
                kernel_size=self.kernel_size,
                strides=self.conv_strides,
                padding="valid",
                activation="relu",
                input_shape=(self.w, self.n_features),
            )
        )

        # Pooling Layer #1
        # First max pooling layer with a filter of length 2 and stride of 2
        # Input Tensor Shape: [batch_size, w, num_filt_1]
        # Output Tensor Shape: [batch_size, 0.5 * w, num_filt_1]
        model.add(MaxPooling1D(pool_size=self.pool_size_1))

        model.add(
            Conv1D(
                filters=self.num_filt_2,
                kernel_size=self.kernel_size,
                strides=self.conv_strides,
                padding="valid",
                activation="relu",
            )
        )

        # Max Pooling Layer #2
        # Second max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 0.5 * w, num_filt_1 * num_filt_2]
        # Output Tensor Shape: [batch_size, 0.25 * w, num_filt_1 * num_filt_2]
        model.add(MaxPooling1D(pool_size=self.pool_size_2))

        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, 0.25 * w, num_filt_1 * num_filt_2]
        # Output Tensor Shape: [batch_size, 0.25 * w * num_filt_1 * num_filt_2]
        model.add(Flatten())

        # Dense Layer (Output layer)
        # Densely connected layer with 1024 neurons
        # Input Tensor Shape: [batch_size, 0.25 * w * num_filt_1 * num_filt_2]
        # Output Tensor Shape: [batch_size, 1024]
        model.add(Dense(units=self.num_nrn_dl, activation="relu"))

        # Dropout
        # Prevents overfitting in deep neural networks
        model.add(Dropout(self.dropout_rate))

        # Output layer
        # Input Tensor Shape: [batch_size, 1024]
        # Output Tensor Shape: [batch_size, p_w]
        model.add(Dense(units=self.num_nrn_ol * self.n_features))

        # Summarize model structure
        model.summary()

        ##Configuring the model
        model.compile(optimizer="adam", loss="mean_absolute_error")

        ###Training
        model.fit(batch_sample, batch_label, epochs=self.epochs, verbose=0)

        """Save Weights (DeepAnT)"""
        # save it to disk so we can load it back up or compare to another training if necessary
        model.save("/home/amanda/Documents/Github_LMEst/edge_lmest/data_science/data_benchmark/da_model_teste_1000.h5")

        self.model = model

    def load_model(self, path_load_model):
        """Loads the model.

        Parameters
        ----------
        path_load_model : str
            Path to the model .h5 file.
        """
        self.model = models.load_model(path_load_model)

    def set_threshold(self, threshold):
        """Sets the threshold value for the future comparisons.

        Parameters
        ----------
        threshold : float, numpy.ndarray
            Limit value for the analysis.
        """

        self.threshold_const = threshold
        self.threshold_calibration_window = 0

    def _predict_window(self):
        """Estimates an signal to compare with the one acquired.

        Returns
        -------
        predicted_window : numpy.ndarray
            Predicted signal for all signals given by the CNN model.
        """

        predicted_window = None
        # aux = self.signal_buffer
        for i in range(int(self.num_sensors / self.sensors_group)):
            raw_seq = np.concatenate(
                (
                    self.signal_buffer[0][i],
                    self.signal_buffer[1][i],
                    self.signal_buffer[2][i],
                )
            )

            endix = len(raw_seq) - self.w - self.p_w

            input_seq = array(raw_seq[endix : endix + self.w])

            input_seq = input_seq.reshape((1, self.w, self.n_features))

            # Predict the next time stamps of the sampled sequence
            predicted_sensor = self.model.predict(input_seq, verbose=0)

            if predicted_window is None:
                predicted_window = predicted_sensor
            else:
                predicted_window = np.append(
                    predicted_window,
                    predicted_sensor,
                    0,
                )

        return predicted_window

    def _window_MAE(self, predicted_window):
        """Calculates the Mean Absolute Error for the predicted window."""

        for i in range(self.num_sensors):
            # calculates the MAE between the predicted window for sensor "i" and the real window for sensor "i" (position 2 is always the newest sample from the signal buffer)
            self.MAE_buffer[i].append(
                mean_absolute_error(self.signal_buffer[2][i], predicted_window[i])
            )
            self.MAE_buffer[i].pop(0)  # pops the oldest MAE value

            if self.moving_avg_buffer[i][0] is None:
                for j in range(len(self.moving_avg_buffer[i])):
                    self.moving_avg_buffer[i][j] = self.MAE_buffer[i][-1]

            if self.is_MAE_initialized == False:
                for k in range(int(self.anomaly_neighborhood / 2)):
                    self.MAE_buffer[i].append(
                        mean_absolute_error(
                            self.signal_buffer[2][i], predicted_window[i]
                        )
                    )
                    self.MAE_buffer[i].pop(0)  # pops the oldest MAE value

                self.is_MAE_initialized = True

    def _MAE_Moving_Avg(self):
        """Calculates the Mean Absolute Error considering the moving average analysis."""

        for i in range(self.num_sensors):
            self.moving_avg_buffer[i].append(
                self.MAE_buffer[i][-1]
            )  # pushing the newest MAE value
            self.moving_avg_buffer[i].pop(0)  # pops the oldest MAE value
            self.MAE_buffer[i][-1] = (
                sum(self.moving_avg_buffer[i]) / self.moving_avg_size
            )

    def set_sample(self, sample):
        """Alternative method to trigger the anomaly analyses.

        Parameters
        ----------
        sample : numpy.ndarray
            The last acquired signal. This array's shape is (number of points, number of sensors).
        """

        self.send_sample(sample)

    # from abstractmethod
    def send_sample(self, sample):
        """Starts the classification of the last acquired signal. If the training process did not occurred, it will do it. After been trained, it tries to predict each subsequent data window, hence it is required a small data buffer containing the Mean Absolute Error (MAE) values. An adaptable threshold is calculated for the predicted signal window and it is compared with the input signal to decide if it is anomalous or not.

        Parameters
        ----------
        sample : numpy.ndarray
            The last acquired signal. This array's shape is (number of points, number of sensors).

        """

        if self.p_w is None:
            self._set_pw(len(sample))

        if self.num_sensors is None:
            self.num_sensors = len(sample[0])

        # This signal_buffer has 3 positions. The first 2 are the previous windows and the last is the actual window. Each window has number of lines equal to the number of sensors and number of columns equal to the number of points divided by 2, because the analysis is made in the frequency domain.
        self.signal_buffer[0] = self.signal_buffer[1]
        self.signal_buffer[1] = self.signal_buffer[2]
        self.signal_buffer[2] = np.ones((self.num_sensors, self.p_w)) * -1

        for i in range(self.num_sensors):
            _, self.signal_buffer[2][i] = FFT(sample[:, i], self.fs)

        if self.signal_buffer[0] is not None:
            if self.flag_first_sample:
                if not self.flag_load_model:
                    self._initialize_neural_network()
                self.flag_first_sample = False

                # Creates a list containing n lists, each of m items, all set to 0
                # where n is the number os sensors, given by self.num_sensors

                # for MAE_buffer m is the self.anomaly_neighborhood
                self.MAE_buffer = [
                    [-1] * self.anomaly_neighborhood for y in range(self.num_sensors)
                ]

                # for moving_avg_buffer m is the self.moving_avg_size
                self.moving_avg_buffer = [
                    [None] * self.moving_avg_size for y in range(self.num_sensors)
                ]
            else:
                predicted_window = self._predict_window()
                self._window_MAE(predicted_window)
                if self.do_moving_avg:
                    self._MAE_Moving_Avg()
                if self.threshold_calibration_window > 0:
                    if (
                        self.threshold_const
                        < self._sample_anomaly_score()
                        / (np.mean(np.abs(self.signal_buffer[2])) / len(sample[0]))
                        and self.tr_aux >= 5
                    ):
                        self.threshold_const = self._sample_anomaly_score() / (
                            np.mean(np.abs(self.signal_buffer[2])) / len(sample[0])
                        )
                    self.tr_aux += 1
                    self.threshold_calibration_window -= 1

                self.last_sample_anomaly = self._sample_is_anomaly()

        self.threshold.append(
            self.threshold_const
            * self.threshold_mult
            * np.mean(np.abs(self.signal_buffer[2]))
            / len(sample[0])
        )

        if len(self.threshold) > self.threshold_smooth:
            self.threshold.pop(0)

        np.save("/home/amanda/Documents/Github_LMEst/edge_lmest/data_science/data_benchmark/da_threshold_teste_1000.npy", np.array(self.threshold_const))

    def set_dt(self, dt):
        """Sets the time step of the acquired signal.

        Parameters
        ----------
        dt : float
            Time step of the signal that will be passed to the CNN.
        """

        self.dt = dt
        self.fs = 1 / dt

    def are_samples_distinct(self):
        """Returns wether or not the last samples is an anomaly.

        Returns
        -------
        self.last_sample_anomaly : bool
            Boolean value for classification. Options are:
                True : the samples are distinct, or the network used the sample for training, or it still needs posterior data to assert on normality.
                False : the samples are similar.
        """

        return self.last_sample_anomaly

    # from abstractmethod
    def _sample_is_anomaly(self):
        """Compare the calculated variation in the reconstruction error with the threshold.

        Returns
        -------
        self.last_sample_anomaly : bool
            Boolean value for classification. Options are:
                True : the samples are distinct, or the network used the sample for training, or it still needs posterior data to assert on normality.
                False : the samples are similar.
        """
        if self.model == None:
            # print("Neural network still training")
            return True
        elif self.threshold == []:
            # print("Network trained, but no signal recieved afterwards")
            return True
        elif self.MAE_buffer[0][0] == -1:
            # print("Awaiting for neighboring values to assert the normality of current data")
            return True
        else:
            self.last_anomaly_score = self._sample_anomaly_score()
            self.last_sample_threshold = 0

            for i in range(self.threshold_smooth):
                self.last_sample_threshold = self.last_sample_threshold + (
                    self.threshold[i] / self.threshold_smooth
                )

            if (
                self.threshold_calibration_window > 0
            ):  # is always anomaly while calibrating the threshold, to analyze while calibrating, remove this if and its contents
                return True

            if self.last_anomaly_score >= self.last_sample_threshold:
                self.threshold_mult = self.tr_base * 1.2

                return True
            else:
                self.threshold_mult = self.tr_base * 1

                return False

    def _sample_anomaly_score(self):
        """Calculates a score to compare with the threshold value.

        Returns
        -------
        anomaly_score : float
            Metric for evaluating the signal.
        """

        anomaly_score = 0

        for i in range(self.num_sensors):
            anomaly_score += np.absolute(
                self.MAE_buffer[i][int((self.anomaly_neighborhood / 2))]
                - ((sum(self.MAE_buffer[i])) / self.anomaly_neighborhood)
            )

        anomaly_score = anomaly_score / self.num_sensors

        return anomaly_score
