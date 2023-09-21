import os

import numpy as np
from sklearn.metrics import mean_absolute_error as MAE
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, Input

from .abstract import Methodology


class Autoencoder(Methodology):
    """This uses the Autoencoder Neural Network method. It compares the original signal captured by sensors, signal will aquired as "blocks", that will be compared with each other. The variation among the reconstruction error of each block (or sample) will tell about behavior of the signal.

    The higher the variation in the reconstruction error of consecutive samples,
    the higher the variation in the waveform signal from the sensors.

    Parameters
    ----------
    size_buffer_thrs : int
        Number of threshold inside the buffer.
    min_signals_train : int
        Minimum number of signals for the training process of the Autoencoder Neural Network.
    model : Autoencoder object, optional
        An already trained model of this class.
        If this parameter is not passed, the class will create and train a new model.
    threshold : numpy.ndarray, optional
        Array with the threshold limit for the classification.
        If this parameter is not passed, the class will set the threshold by itself.
    """

    NAME = "auto_encoder"

    def __init__(
        self,
        model=None,
        threshold=None,
        min_signals_train=5,
        size_buffer_thrs=20,
    ):
        ### NN Hyperparameters
        self.conv_filters = 16
        self.conv_kernel_size = 2
        self.wavenet_residual_layers = 4
        self.num_wavenet_blocks = 1
        self.optimizer_function = "adam"
        self.loss_function = "mse"

        self.total_epochs = 0
        self.max_epochs = 10000
        self.size_sub_pack = 512
        self.size_buffer_thrs = size_buffer_thrs
        self.std_threshold = 5
        self.min_signals_train = min_signals_train

        self.num_signals_train = 0
        self.num_signals_op = 0

        self.flag_first_sample = True
        self.flag_start_operating = False

        self.buffer_var_errors_auto_threshold = list()
        self.min_values = list()
        self.max_values = list()

        # Check if the threshold inserted is "auto" or a number
        if threshold is None:
            self.threshold = 0.1
            self.flag_auto_threshold = True
        elif (
            isinstance(threshold, float)
            or isinstance(threshold, int)
            or isinstance(threshold, list)
            or isinstance(threshold, np.ndarray)
        ):
            self.threshold = np.array(threshold)
            self.flag_auto_threshold = False
            if np.ndim(self.threshold) == 0:
                super().__init__(self.threshold)
            elif np.ndim(self.threshold) > 0:
                super().__init__(self.threshold[0])
            # Set flag start_operating to True
            self.flag_start_operating = True
        else:
            raise Exception("Invalid threshold inserted.")

        if model is not None:
            self.model = model
            self.flag_model_loaded = True
            self.status_training = False
        else:
            self.model = None
            self.flag_model_loaded = False
            self.status_training = True

    def set_sample(self, sample):
        """Alternative method to trigger the anomaly analyses.

        Parameters
        ----------
        sample : numpy.ndarray
            The last acquired signal. This array's shape is (number of points, number of sensors).
        """

        self.send_sample(sample)

    def send_sample(self, sample_compare):
        """Triggers the comparison routine.

        Parameters
        ----------
        sample_compare : numpy.ndarray
            The last acquired signal. This array's shape is (number of points, number of sensors).
        """
        # Preprocessing the input samples
        self.preprocessing_sample_compare(sample_compare)

        # For the first signal, use the loaded model or initialize a new one (random)
        if self.flag_first_sample:
            self.flag_first_sample = False
            if not self.flag_model_loaded:
                self.initialize_model()

            self.var_error = self.threshold * 1.01
            ##########
            ### Fot the compare_data class
            self.get_compare_limit()
            ##########

        # For the following signals
        else:
            if self.status_training:
                self.train_model()
            elif not self.status_training:
                self.operation_phase()

        ### Update the baseline if an anomaly is detected
        if self.num_signals_op > 0 and self.are_samples_distinct():
            self.base_sample_errors = self.current_sample_errors.copy()

    def preprocessing_sample_compare(self, sample_compare):
        """Prepare the acquired sample for the anomaly identification analysis.

        Parameters
        ----------
        sample_compare : numpy.ndarray
            The last acquired signal.
        """

        ### Adjust the input sample's shape to best fit for the Neural Network
        # If the variable sample_compare has 1 dimension
        if sample_compare.ndim == 1:
            sample_compare = sample_compare.reshape(-1, 1)
        else:
            if sample_compare.shape[1] > sample_compare.shape[0]:
                sample_compare = sample_compare.T
        self.num_sensors = sample_compare.shape[1]

        # If the ANN is still training or adjusting the threshold
        if self.status_training:
            
            self.input_samples = list()
            min_values = np.zeros(self.num_sensors)
            max_values = np.zeros(self.num_sensors)

            for sensor_index in range(self.num_sensors):
                ### Slice the input into smaller fragments
                fragments = self.slice_vector(
                    sample_compare[:, sensor_index],
                    slice_size=self.size_sub_pack,
                    overlap=int(
                        self.size_sub_pack * 0.9
                    ),  # 90% overlapping for the training
                )
                self.num_samples = fragments.shape[1]

                ## Normalize between 0 and 1 with min and max amplitude of the training set
                min_values[sensor_index] = np.min(fragments)
                max_values[sensor_index] = np.max(fragments)
                fragments = self.normalize(
                    fragments,
                    np.min(fragments),
                    np.max(fragments),
                )
                self.input_samples.append(fragments)

            self.min_values.append(min_values)
            self.max_values.append(max_values)
            self.input_samples = np.array(self.input_samples).T
            # # Reshape to match the Keras input shape
            # self.input_samples = self.input_samples.reshape(
            #     self.input_samples.shape[2],
            #     self.input_samples.shape[1],
            #     self.input_samples.shape[0],
            # )

        # If the ANN is in the operation phase
        else:
            if self.flag_first_sample and self.flag_model_loaded:
                self.min_values = np.zeros(self.num_sensors)
                self.max_values = np.zeros(self.num_sensors)
                ### Check if the min and max values were passed inside the threshold array
                if len(self.threshold) == self.num_sensors:
                    # Define the min and max values
                    for sensor_index in range(self.num_sensors):
                        self.min_values[sensor_index] = np.min(
                            sample_compare[:, sensor_index]
                        )
                        self.max_values[sensor_index] = np.max(
                            sample_compare[:, sensor_index]
                        )
                else:
                    self.min_values = self.threshold[
                        self.num_sensors : 2 * self.num_sensors
                    ]
                    self.max_values = self.threshold[
                        2 * self.num_sensors : 3 * self.num_sensors
                    ]
                    self.threshold = self.threshold[: self.num_sensors]

            self.input_samples = list()
            for sensor_index in range(self.num_sensors):
                ### Slice the input into smaller fragments
                fragments = self.slice_vector(
                    sample_compare[:, sensor_index],
                    slice_size=self.size_sub_pack,
                    overlap=int(
                        self.size_sub_pack * 0.1
                    ),  # 10% overlapping for the operation
                )
                self.num_samples = fragments.shape[1]

                ## Normalize between 0 and 1 based on the training set min and max values
                fragments = self.normalize(
                    fragments,
                    self.min_values[sensor_index],
                    self.max_values[sensor_index],
                )
                self.input_samples.append(fragments)

            self.input_samples = np.array(self.input_samples).T

    def get_compare_limit(self):
        """
        Gets the updated value for the threshold.

        Method required by the benchmark analysis.
        """

        if self.num_signals_op == 0 and self.flag_auto_threshold:
            self.compare_limit = self.threshold
        else:
            self.compare_limit = self.threshold[0]

    def initialize_model(self):
        """Initializes the model of the autoencoder neural network."""

        input_dimension = self.input_samples.shape[1]

        latent_dim = int(input_dimension / 2**self.num_wavenet_blocks)

        wavenet_layers = np.zeros(self.wavenet_residual_layers)
        wavenet_layers[0] = 1
        for layer in range(1, self.wavenet_residual_layers):
            wavenet_layers[layer] = wavenet_layers[layer - 1] * 2

        ### ===============================
        ### ENCODER
        ### ===============================
        encoder_input = Input(shape=(input_dimension, self.num_sensors))

        flag_fist_layer_encoder = True
        for block in range(self.num_wavenet_blocks):
            for rate in wavenet_layers:
                if flag_fist_layer_encoder:
                    encoded = Conv1D(
                        filters=self.conv_filters,
                        kernel_size=self.conv_kernel_size,
                        padding="causal",
                        activation="relu",
                        dilation_rate=int(rate),
                    )(encoder_input)
                    flag_fist_layer_encoder = False
                else:
                    encoded = Conv1D(
                        filters=self.conv_filters,
                        kernel_size=self.conv_kernel_size,
                        padding="causal",
                        activation="relu",
                        dilation_rate=int(rate),
                    )(encoded)
            encoded = Conv1D(
                filters=self.conv_filters,
                kernel_size=self.conv_kernel_size,
                strides=2,
                padding="same",
            )(encoded)
        encoded = Conv1D(self.num_sensors, 1, name="Last_encoder")(encoded)

        encoder = Model(encoder_input, encoded)

        ### ===============================
        ### DECODER
        ### ===============================
        decoder_input = Input(shape=(latent_dim, self.num_sensors))

        flag_fist_layer_decoder = True
        for block in range(self.num_wavenet_blocks):
            if flag_fist_layer_decoder:
                decoded = Conv1DTranspose(
                    filters=self.conv_filters,
                    kernel_size=self.conv_kernel_size,
                    strides=2,
                    padding="same",
                )(decoder_input)
                flag_fist_layer_decoder = False
            else:
                decoded = Conv1DTranspose(
                    filters=self.conv_filters,
                    kernel_size=self.conv_kernel_size,
                    strides=2,
                    padding="same",
                )(decoded)

            for rate in wavenet_layers:
                decoded = Conv1D(
                    filters=self.conv_filters,
                    kernel_size=self.conv_kernel_size,
                    padding="causal",
                    activation="relu",
                    dilation_rate=int(rate),
                )(decoded)

        output_layer = Conv1D(self.num_sensors, 1, name="Last_decoder")(decoded)

        decoder = Model(decoder_input, output_layer)

        ### ===============================
        ### AUTOENCODER
        ### ===============================
        ae_input = Input(shape=(input_dimension, self.num_sensors))

        encoded = encoder(ae_input)
        decoded = decoder(encoded)

        wavenet_autoencoder = Model(ae_input, decoded)

        wavenet_autoencoder.compile(
            loss=self.loss_function, optimizer=self.optimizer_function
        )

        wavenet_autoencoder.summary()

        self.model = wavenet_autoencoder
        self.encoder_model = encoder
        self.decoder_model = decoder

    def are_samples_distinct(self):
        """Compare calculated variation in the reconstruction error with the threshold.

        Returns
        -------
        bool
            True if the samples are distinct, and False if they are similar.
        """

        if self.num_signals_op == 0 and self.flag_auto_threshold:
            return self.var_error > self.threshold
        else:
            return any(self.var_error > self.threshold)

    def compare_value(self):
        """Gives the variation in the reconstruction error's buffer.

        Returns
        -------
        var_error : float
            Reconstruction error value.
        """

        if self.num_signals_op == 0 and self.flag_auto_threshold:
            return self.var_error
        else:
            return self.var_error[0]

    def get_auto_threshold(self):
        """Function to get the threshold value automatically. The Threshold is calculated with the formula 'mean(x) + 5 x std(x)', being x the vector of anomaly scores. This means that 9.997% of the data is considered normal and every sample that is beyond this value (i.e. has anomaly scores higher than the threshold) are placed outside the normal distribution of the training data.

        Returns
        -------
        threshold : float
            Threshold value.
        """

        var_errors = np.array(self.buffer_var_errors_auto_threshold[1:])
        self.threshold = np.zeros(self.num_sensors)

        for sensor_index in range(self.num_sensors):
            self.threshold[sensor_index] = np.mean(
                var_errors[:, sensor_index]
            ) + self.std_threshold * np.std(var_errors[:, sensor_index])

        print("Threshold defined!!")
        self.thresholds_min_max_values = np.concatenate(
            (
                self.threshold,
                self.min_values,
                self.max_values,
            )
        )

        np.save("/home/amanda/Documents/Github_LMEst/edge_lmest/data_science/data_benchmark/ae_thresholds_teste_1000.npy",self.threshold)

        return self.threshold

    def train_model(self):
        """Starts the training process of the autoencoder neural network. This function automatically detects if the training must stop by the bias variation."""

        self.num_signals_train += 1

        print("================================")
        print("Training with the " + str(self.num_signals_train) + "th signal.")

        # Callback for the Early Stopping regularization
        early_stopping = [
            EarlyStopping(
                monitor="val_loss",
                min_delta=10**-4,
                patience=10,
                verbose=0,
                mode="auto",
                baseline=None,
                restore_best_weights=True,
            )
        ]

        num_train_samples = int(0.9 * len(self.input_samples))

        inputs_train = self.input_samples[:num_train_samples]
        outputs_train = inputs_train
        inputs_val = self.input_samples[num_train_samples:]
        outputs_val = inputs_val

        hist = self.model.fit(
            x=inputs_train, 
            y=outputs_train,
            epochs=self.max_epochs,
            batch_size=32,
            validation_data=(inputs_val, outputs_val),
            callbacks=early_stopping,
            initial_epoch=self.total_epochs,
            verbose=1,
        )
        self.total_epochs += len(hist.history["val_loss"])

        # Stop training only after n signals
        if self.num_signals_train == self.min_signals_train:
            print("================================")
            print("Training ended at the " + str(self.num_signals_train) + "th signal.")
            print("MAE val = " + str(hist.history["val_loss"][-1]))
            print("================================")

            ### Calculate min and max values
            ### based on the mean of the min and max values of the training set
            min_values = np.array(self.min_values)
            max_values = np.array(self.max_values)
            self.min_values = np.mean(min_values, axis=0)
            self.max_values = np.mean(max_values, axis=0)

            self.status_training = False

            self.model.save("/home/amanda/Documents/Github_LMEst/edge_lmest/data_science/data_benchmark/ae_model_teste_1000.h5")

    def operation_phase(self):
        """Starts the anomaly analysis.

        Returns
        -------
        var_error : float
            The reconstruction error value calculated.
        """

        if self.num_signals_op == 0 and self.flag_auto_threshold:
            print("Setting up the threshold...")

        self.operate_model()

        if self.num_signals_op < self.size_buffer_thrs and self.flag_auto_threshold:
            self.threshold = np.ones(self.num_sensors) * self.threshold
            self.var_error = self.threshold * 1.01
        elif self.num_signals_op == self.size_buffer_thrs and self.flag_auto_threshold:
            self.get_auto_threshold()
            self.flag_start_operating = True
        elif not self.flag_auto_threshold:
            self.threshold = np.ones(self.num_sensors) * self.threshold

        self.get_compare_limit()  # required by the benchmak analysis.

    def operate_model(self):
        """Function to calculate the reconstruction error of each sample and each sensor after the autoencoder is trained."""

        # Limit this value number to not exceed the memory
        if self.num_signals_op <= self.size_buffer_thrs:
            self.num_signals_op += 1

        # Get reconstruction error of all subblocks
        current_errors = np.zeros((self.num_samples, self.num_sensors))

        reconstructed_samples = self.model.predict(self.input_samples, verbose=0)

        for sensor_index in range(self.num_sensors):
            current_errors[:, sensor_index] = MAE(
                self.input_samples[:, :, sensor_index].T,
                reconstructed_samples[:, :, sensor_index].T,
                multioutput="raw_values",
            )

        self.current_sample_errors = current_errors

        self.calculate_errors()

    def calculate_errors(self):
        """Calculate the required errors for the analysis."""

        if self.num_signals_op == 1:
            self.base_sample_errors = self.current_sample_errors.copy()
            self.first_sample_errors = self.current_sample_errors.copy()

        self.var_error = (
            abs(
                (
                    np.sum(self.current_sample_errors, axis=0)
                    - np.sum(self.base_sample_errors, axis=0)
                )
                / np.sum(self.base_sample_errors, axis=0)
            )
            * 100
        )

        if self.flag_auto_threshold and not self.flag_start_operating:
            self.buffer_var_errors_auto_threshold.append(self.var_error)

    def normalize(self, data, min_value, max_value):
        """Normalization in the the range [0,1]

        Returns
        -------
        data_normalized : numpy.ndarray
            Normalized vector.
        """
        return (data - min_value) / (max_value - min_value)

    def slice_vector(self, vector, slice_size: int, overlap=0):
        """This function slices a signal into many samples in sequence, with a given overlap.

        Parameters
        ----------
        vector : numpy.ndarray
            Vector to be sliced.
        slice_size : int
            Size of each slice, or sample.
        overlap : int
            Number of data points overlapping.

        Returns
        -------
        slices : numpy.ndarray
            All slices.

        Examples
        --------
        >>> samples = self.slice_vector(data, 512, 50)
        """

        if vector.ndim == 2:
            vector = vector.reshape(-1)
        elif vector.ndim > 2:
            raise Exception("Number of dimensions must be less than 2.")

        indexes_list = np.arange(0, len(vector) - overlap, slice_size - overlap)

        slices = [vector[i : i + slice_size] for i in indexes_list]

        # Check if there are missing values in the last slice windows
        if len(slices[-1]) != slice_size:
            # If so, the last windows will be the last 'slice_size' points
            slices[-1] = vector[-slice_size:]

        slices = np.array(slices).T

        return slices
