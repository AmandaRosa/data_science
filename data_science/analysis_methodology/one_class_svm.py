import sys

import numpy as np

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

from sklearn.preprocessing import *
from sklearn.svm import OneClassSVM

from data_science.clustering import EstimateResourcesOCSVM

from .abstract import Methodology


class One_Class_SVM(Methodology):
    """This class uses One Class Support Vector Machines for detecting anomalies."""

    NAME = "ocsvm"

    def __init__(
        self,
        size_sub_sample=512,
        oversampling=True,
        steps_to_overlap=None,
        size_buffer_training=10,
        size_buffer_adjust_threshold=10,
        discriminators=["rms"],
        reduction=None,
        nb_components=3,
        kernel="sigmoid",
        nu=0.025,
        gamma=1,
        threshold="mean",
        sigma=5,
        training=True,
        wavelet_decomposition=False,
        wavelet_levels=5,
        flag_slicing=False,
        flag_overlap=False,
    ):
        self.size_sub_sample = size_sub_sample
        self.oversampling = oversampling
        self.steps_to_overlap = steps_to_overlap
        self.size_buffer_training = size_buffer_training
        self.size_buffer_adjust_threshold = size_buffer_adjust_threshold
        self.flag_slicing = flag_slicing
        self.flag_overlap = flag_overlap
        self.discriminators = discriminators
        self.wavelet_decomposition = wavelet_decomposition
        self.wavelet_levels = wavelet_levels
        self.reduction = reduction
        self.nb_components = nb_components
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.threshold = threshold
        self.sigma = sigma
        self.training = training

        if self.threshold is not float:
            self.threshold = 0

        self.counter_train = 0
        self.counter_threshold = 0
        self.data_base = list()
        self.all_features_training = list()
        self.sample_base = None

        self.global_counter = 0

        super().__init__(self.threshold)

    def set_sample(self, sample):
        self.send_sample(sample)

    def send_sample(self, sample_compare):
        # self.global_counter += sample_compare.shape[0]
        # self.global_counter += 1
        # print(f'Global counter = {self.global_counter}')

        if sample_compare.ndim == 1:
            sample_compare = sample_compare.reshape((len(sample_compare), 1))

        if self.sample_base is None:
            self.sample_base = sample_compare
            self.dt = 1 / sample_compare.shape[0]

        if self.sample_base.shape != sample_compare.shape:
            raise Exception(
                f"Samples must have the same shape. Sample Base: {self.sample_base.shape} != Sample Compare: {sample_compare.shape}"
            )

        self.sample_compare = sample_compare

        ### ===========================
        # For the first input sample, the class EstimateResources is defined
        ### ===========================
        if self.counter_train == 0:
            self.estimate_resouces = EstimateResourcesOCSVM(
                dt=self.dt,
                slice_size=self.size_sub_sample,
                steps_to_overlap=self.steps_to_overlap,
                discriminators=self.discriminators,
                wavelet_decomposition=self.wavelet_decomposition,
                wavelet_levels=self.wavelet_levels,
            )
            self.num_sensors = self.sample_compare.shape[1]

            self.num_total_features = len(self.discriminators)
            if self.wavelet_decomposition:
                self.num_total_features *= self.wavelet_levels + 1
            self.all_parameters_base = [None] * self.num_sensors
            self.decision_values = [1] * self.num_sensors
            self.anomaly_scores = [1] * self.num_sensors
            print("Defining the frontiers...")

        ### ===========================
        ### Gathering signals to define the frontier
        ### ===========================
        if self.counter_train < self.size_buffer_training:
            self.counter_train += 1
            ### Add the current features to a buffer (list)
            self.input_features = self.estimate_resouces.get_resources(
                self.sample_compare,
                slicing=False,
                overlapping=False,  ########
            )
            self.all_features_training.append(self.input_features)

        ### ===========================
        ### Defining the frontier!!
        ### ===========================
        elif self.counter_train == self.size_buffer_training:
            self.all_features_training = self.preprocessing_features(
                self.all_features_training
            )

            self.defining_ocsv_models()
            self.training_ocsvm_models(self.all_features_training)
            print("Frontiers defined!!!")

            self.counter_train += 1
            self.counter_threshold += 1

            print("Computing threshold...")

            # self.all_data_training = [] # Clean buufer
            self.all_features_adjust_threshold = list()
            self.threshold = [1] * self.num_sensors

            self.input_features = self.estimate_resouces.get_resources(
                self.sample_compare,
                slicing=False,
                overlapping=False,  ########
            )
            self.input_features = self.preprocessing_features(self.input_features)

            self.decision_values = self.get_decision_values(self.input_features)
            self.baseline_decision_values = self.decision_values.copy()

        elif (self.counter_threshold > 0) and (
            self.counter_threshold < self.size_buffer_adjust_threshold
        ):
            self.input_features = self.estimate_resouces.get_resources(
                self.sample_compare,
                slicing=False,
                overlapping=False,  ########
            )
            self.input_features = self.preprocessing_features(self.input_features)

            self.decision_values = self.get_decision_values(self.input_features)
            self.anomaly_scores = self.get_anomaly_scores()

            self.all_features_adjust_threshold.append(self.anomaly_scores)
            self.counter_threshold += 1

        elif self.counter_threshold == self.size_buffer_adjust_threshold:
            self.threshold = self.compute_threshold2()
            print("Threshold defined!!!")
            super().__init__(self.threshold[0])

            self.input_features = self.estimate_resouces.get_resources(
                self.sample_compare,
                slicing=False,
                overlapping=False,  ########
            )
            self.input_features = self.preprocessing_features(self.input_features)

            self.decision_values = self.get_decision_values(self.input_features)
            self.anomaly_scores = self.get_anomaly_scores()
            self.counter_threshold += 1

        elif self.counter_threshold > self.size_buffer_adjust_threshold:
            self.input_features = self.estimate_resouces.get_resources(
                self.sample_compare,
                slicing=False,
                overlapping=False,  ########
            )
            self.input_features = self.preprocessing_features(self.input_features)

            self.decision_values = self.get_decision_values(self.input_features)
            self.anomaly_scores = self.get_anomaly_scores()

            # """
            ### Update the baseline ==================
            if self.are_samples_distinct():
                self.baseline_decision_values = self.decision_values.copy()
            ### ======================================
            # """

        """
        ### ===========================
        ### After defining the frontier, calculating the distance of the input data to this frontier
        ### ===========================
        elif self.counter_train > self.size_buffer_training:
            self.data_compare = self.estimate_resouces.get_resources(
                self.sample_compare,
                slicing=False,
                overlapping=False, ########
            ) 
            self.data_compare = self.preprocessing_data(self.data_compare)
            self.decision_values = self.get_decision_values(self.data_compare)
            self.anomaly_scores = self.get_anomaly_scores()

            if self.counter_threshold < self.size_buffer_adjust_threshold:                
                self.all_parameters_base.append(self.anomaly_scores)
                self.anomaly_scores = [1] * self.num_sensors
                self.counter_threshold += 1

            elif self.counter_threshold == self.size_buffer_adjust_threshold:
                self.compute_threshold()
                self.counter_threshold += 1
                super().__init__(self.threshold[0])
        """

    def are_samples_distinct(self):
        if self.counter_threshold <= self.size_buffer_adjust_threshold:
            return True
        elif self.counter_threshold > self.size_buffer_adjust_threshold:
            return any(np.array(self.anomaly_scores) >= np.array(self.threshold))

    def compare_value(self):
        return self.anomaly_scores[0]

    def defining_ocsv_models(self):
        """
        This function defines a list of ocsvm models being one model per sensor
        """
        self.ocsvm_models_list = [None] * self.num_sensors
        for sensor_index in range(self.num_sensors):
            self.ocsvm_models_list[sensor_index] = OneClassSVM(
                kernel=self.kernel,
                gamma=self.gamma,
                nu=self.nu,
            )

    def training_ocsvm_models(self, input_data):
        """
        This function adjusts the frontier of each one of the models, being one model per sensor
        """
        if self.reduction is None:
            input_data = input_data
        elif self.reduction == "pca":
            self.reduct = PCA(n_components=self.nb_components).fit(input_data)
            for sensor_index in range(self.num_sensors):
                input_data[sensor_index] = self.reduct.transform(
                    input_data[sensor_index]
                )
        elif self.reduction == "kernelPCA":
            self.reduct = KernelPCA(
                n_components=self.nb_components,
                kernel="rbf",
            ).fit(input_data)
            for sensor_index in range(self.num_sensors):
                input_data[sensor_index] = self.reduct.transform(
                    input_data[sensor_index]
                )

        for sensor_index in range(self.num_sensors):
            self.ocsvm_models_list[sensor_index].fit(input_data[sensor_index])

    def compute_threshold(self):
        all_scores_np = np.array(self.all_features_adjust_threshold)
        threshold = np.mean(all_scores_np, axis=0) + self.sigma * np.std(
            all_scores_np, axis=0
        )

        """
        decision_value_np = np.array(self.LISTA_APAGAR)
        plt.figure()
        index=0
        for index in range(self.num_sensors):
            plt.subplot(4,1,index+1)
            plt.plot(decision_value_np[:,index])
        plt.show()

        plt.figure()        
        index=0
        for index in range(self.num_sensors):
            plt.subplot(4,1,index+1)
            plt.plot(all_scores_np[:,index])
            plt.plot(np.ones(len(all_scores_np[:,index]))*threshold[index])
        plt.show()
        """

        return threshold

    def compute_threshold2(self):
        all_scores_np = np.array(self.all_features_adjust_threshold)

        threshold = np.mean(all_scores_np, axis=0) + self.sigma * np.std(
            all_scores_np, axis=0
        )

        ##############
        """
        threshold2 = np.percentile(all_scores_np, 99, axis=0) 

        sensor_index = 0

        all_data_np = np.array(self.LIST_SAMPLES_TRAINING)
        
        scores_training = self.ocsvm_models_list[sensor_index].decision_function(self.all_features_training[sensor_index])
        answers_training = self.ocsvm_models_list[sensor_index].predict(self.all_features_training[sensor_index])
        
        plt.figure()
        plt.subplot(4,1,1)
        plt.plot(all_data_np[:,:,0].reshape(-1))
        plt.subplot(4,1,2)
        plt.plot(scores_training)
        plt.ylabel('Scores')
        plt.subplot(4,1,3)
        plt.plot(answers_training)
        plt.ylabel('Answers')

        decision_value_np = np.array(self.LISTA_APAGAR)
        plt.figure()
        index=0
        for index in range(self.num_sensors):
            plt.subplot(4,1,index+1)
            plt.plot(decision_value_np[:,index])
        plt.show()

        plt.figure()        
        index=0
        for index in range(self.num_sensors):
            plt.subplot(4,1,index+1)
            plt.plot(all_scores_np[:,index])
            plt.plot(np.ones(len(all_scores_np[:,index]))*threshold[index],label='Threhold 1')
            plt.plot(np.ones(len(all_scores_np[:,index]))*threshold2[index],label='Threhold 2')
            plt.legend()
        plt.show()
        """
        ##############

        return threshold

    def get_decision_values(self, input_data):
        decision_values = [None] * self.num_sensors
        for sensor_index in range(self.num_sensors):
            decision_values[sensor_index] = np.mean(
                self.ocsvm_models_list[sensor_index].score_samples(
                    input_data[sensor_index]
                )
                # self.ocsvm_models_list[sensor_index].decision_function(
                #     input_data[sensor_index]
                # )
            )
        return decision_values

    def get_anomaly_scores(self):
        anomaly_scores = [None] * self.num_sensors
        for sensor_index in range(self.num_sensors):
            anomaly_scores[sensor_index] = abs(
                (
                    self.decision_values[sensor_index]
                    - self.baseline_decision_values[sensor_index]
                )
                / self.baseline_decision_values[sensor_index]
            )
        return anomaly_scores

    def set_dt(self, dt):
        self.dt = dt

    def preprocessing_features2(self, data):
        processed_features_all_sensors = list()
        data_np = np.array(data)

        if self.counter_train == self.size_buffer_training:
            self.mean_std_values = np.zeros(
                (2, self.num_total_features, self.num_sensors)
            )
            for sensor_index in range(self.num_sensors):
                data_current_sensor = data_np[:, sensor_index, 0, :].copy()
                for feature_index in range(self.num_total_features):
                    mean_value = np.mean(data_current_sensor[:, feature_index])
                    std_value = np.std(data_current_sensor[:, feature_index])
                    self.mean_std_values[0, feature_index, sensor_index] = mean_value
                    self.mean_std_values[1, feature_index, sensor_index] = std_value

                    data_current_sensor[:, feature_index] = self.standardization(
                        data_current_sensor[:, feature_index],
                        self.mean_std_values[0, feature_index, sensor_index],
                        self.mean_std_values[1, feature_index, sensor_index],
                    )
                processed_features_all_sensors.append(data_current_sensor)

        elif self.counter_train > self.size_buffer_training:
            for sensor_index in range(self.num_sensors):
                data_current_sensor = data_np[sensor_index, :, :].copy()
                for feature_index in range(self.num_total_features):
                    data_current_sensor[:, feature_index] = self.standardization(
                        data_current_sensor[:, feature_index],
                        self.mean_std_values[0, feature_index, sensor_index],
                        self.mean_std_values[1, feature_index, sensor_index],
                    )
                processed_features_all_sensors.append(data_current_sensor)

        return processed_features_all_sensors

    def preprocessing_features(self, data):
        processed_features_all_sensors = list()
        data_np = np.array(data)

        if self.counter_train == self.size_buffer_training:
            self.min_max_values = np.zeros(
                (2, self.num_total_features, self.num_sensors)
            )
            for sensor_index in range(self.num_sensors):
                data_current_sensor = data_np[:, sensor_index, 0, :].copy()
                for feature_index in range(self.num_total_features):
                    self.min_max_values[0, feature_index, sensor_index] = np.min(
                        data_current_sensor[:, feature_index]
                    )
                    self.min_max_values[1, feature_index, sensor_index] = np.max(
                        data_current_sensor[:, feature_index]
                    )

                    data_current_sensor[:, feature_index] = self.normalization(
                        data_current_sensor[:, feature_index],
                        self.min_max_values[0, feature_index, sensor_index],
                        self.min_max_values[1, feature_index, sensor_index],
                    )
                processed_features_all_sensors.append(data_current_sensor)

        elif self.counter_train > self.size_buffer_training:
            for sensor_index in range(self.num_sensors):
                data_current_sensor = data_np[sensor_index, :, :].copy()
                for feature_index in range(self.num_total_features):
                    data_current_sensor[:, feature_index] = self.normalization(
                        data_current_sensor[:, feature_index],
                        self.min_max_values[0, feature_index, sensor_index],
                        self.min_max_values[1, feature_index, sensor_index],
                    )
                processed_features_all_sensors.append(data_current_sensor)

        return processed_features_all_sensors

    def preprocessing_data(self, data):
        processed_data_all_sensors = list()

        if self.counter_train == self.size_buffer_training:
            self.mean_std_values = np.zeros((2, self.num_sensors))
            data_base_np = np.array(data)

            for sensor_index in range(self.num_sensors):
                # Max and Min values of the training set will be calculated by the median
                mean_values = np.mean(data_base_np[:, :, sensor_index], axis=1)
                std_values = np.std(data_base_np[:, :, sensor_index], axis=1)
                self.mean_std_values[0, sensor_index] = np.median(mean_values)
                self.mean_std_values[1, sensor_index] = np.median(std_values)

                data_base_np[:, :, sensor_index] = self.min_max_transform(
                    data_base_np[:, :, sensor_index],
                    self.mean_std_values[0, sensor_index],
                    self.mean_std_values[1, sensor_index],
                )
                processed_data_all_sensors.append(data_base_np[:, :, sensor_index])

        elif self.counter_train > self.size_buffer_training:
            for sensor_index in range(self.num_sensors):
                data[:, sensor_index] = self.min_max_transform(
                    data[:, sensor_index],
                    self.mean_std_values[0, sensor_index],
                    self.mean_std_values[1, sensor_index],
                )
                processed_data_all_sensors.append(data[:, sensor_index].reshape(1, -1))

        return processed_data_all_sensors

    def normalization(self, data, min_value, max_value):
        # return data #(Uncomment for disabling normalization)

        # Normalize between 0 and 1

        # data = (data - min_value) / (max_value - min_value)

        # Normalize between -1 and 1
        # data = (data-0.5)*2
        return data

    def standardization(self, data, mean, std):
        # return data #(Uncomment for disabling standardization)
        return (data - mean) / std
