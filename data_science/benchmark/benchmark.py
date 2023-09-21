import json
import os
import platform
import time
from datetime import datetime

import keras
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ..analysis_methodology import (
    ArchitectureWithDeepAnt,
    Autoencoder,
    ClusterAnalysisDivergence,
    DeepAnT_CNN,
    HistogramConsistencyTest,
    One_Class_SVM,
)



class Benchmark:
    """Class to run benchmarks using the deployed techniques in the EDGE R&D project.

    This class was build for the linux environment.

    Parameters
    ----------
    data : dict, str
        If it is a dictionary, it must be have the following structure:

        ::

            data = {
                "filenames": {
                    "data": "",
                    "target": ""
                },
                "models": [
                    {
                        "name": "",
                        "filename": "",
                        "threshold": ""
                    },
                    {
                        "name": "",
                        "info": {}
                    }
                ]
            }

        where:

            - filenames - stores a dictionary with the template names for:

                - data - template name of the `.npy` file to be used as samples to the benchmark. If it has more than one file, you must replace the number with an `@`.
                - target - template name of the `.npy` file to calculate the metrics for the benchmark. If it has more than one file, you must replace the number with an `@`.

            - models - list with dictionaries for each model that you want to run the benchmark on. Each model dict must have the following keys:

                - name - name of the technique.
                - filename - template name of the `.h5` file. This is specific for the `Autoencoder` or `DeepAnT_CNN` classes. If it has more than one file, you must replace the number with an `@`.
                - threshold - template name of the `.npy` file. This is specific for the `Autoencoder` or `DeepAnT_CNN` classes. If it has more than one file, you must replace the number with an `@`.
                - info - a dict with the required parameters for each class. You must consult the model documentation to see which parameters must be passed.

        If it is str, you can specify the name of existent/runned benchmarks. The current options is:

            - default

    path : str
        The path to where the data to run the benchmark is.

    config : dict, optional
        Similar to the data, it contains the specific information for each test that you have in your database. The structure must follow:

        ::

            {
                "test_1": {
                    "fs":  1000,
                    "size_sample": 4000,
                    "size_sub_pack": 512
                },
                "test_2": {
                    "fs":  19638.6,
                    "size_sample": 20000,
                    "size_sub_pack": 512
                }
            }

        The number of inner dictionaries must be equal to the total number of tests you have in your database. Each test must have the following parameters:

            - fs - sample rate of the dataset.
            - size_sample - number of points of each 1 second sample.
            - size_sub_pack - number of points when reducing the sample.

    Raises
    ------
    Exception
        If the user did not pass the config parameter, but passed a personalized data.

    Example
    -------
    >>> from data_science.benchmark import benchmark_example
    >>> path = "home/user/data"
    >>> test_number = 3
    >>> benchmark = benchmark_example(path, test_number)
    """

    def __init__(self, data, path, config=None):
        self.path = path

        if isinstance(data, dict):
            self.filenames = data["filenames"]
            self.models = data["models"]
            self.benchmark_data = data

            if config is not None:
                self.config = config
            else:
                raise Exception(
                    "Since you are passing the data dictionary you must pass the config parameter also!"
                )

        elif isinstance(data, str):
            benchmark_data, benchmark_config = self.load_benchmark(data)
            self.filenames = benchmark_data["filenames"]
            self.models = benchmark_data["models"]
            self.benchmark_data = benchmark_data
            self.config = benchmark_config

    @staticmethod
    def load_benchmark(data):
        """Method to load saved benchmarks and its configurations.

        Parameters
        ----------
        data : str
            Name of the benchmark to be loaded. Current option is:

                - default

        Returns
        -------
        benchmark_data : dict
            Dictionary with the benchmark data info.
        config : dict
            Dictionary with the benchmark configuration info.
        """
        local_path = (
            os.path.dirname(os.path.abspath(__file__)) + "/utils/"
        )

        with open(local_path + "benchmarks.json") as file:
            benchmark_dict = json.load(file)

        with open(local_path + "configs.json") as file:
            config_dict = json.load(file)

        try:
            benchmark_data = benchmark_dict[data]
            config = config_dict[data]

            return benchmark_data, config

        except KeyError as error:
            print("\nThe following benchmark does not exist in the database:")
            print(error, "\n")

            return None, None

    def save_benchmark(self, name):
        """Method to save the current benchmark data and configuration.

        Parameters
        ----------
        name : str
            Name to save the benchmark.
        """
        if self.benchmark_data is not None:
            local_path = (
                os.path.dirname(os.path.abspath(__file__)) + "/utils/"
            )

            with open(local_path + "benchmarks.json") as file:
                benchmark_dict = json.load(file)

            with open(local_path + "configs.json") as file:
                config_dict = json.load(file)

            benchmark_dict[f"{name}"] = self.benchmark_data
            config_dict[f"{name}"] = self.config

            with open(local_path + "benchmarks.json", "w") as file:
                json.dump(benchmark_dict, file, indent=4, separators=(",", ": "))

            with open(local_path + "configs.json", "w") as file:
                json.dump(config_dict, file, indent=4, separators=(",", ": "))

            print("The benchmark and its configuration was saved succesfully!")

        else:
            print("There is not a benchmark and configuration to be saved!")

    def get_model(self, name, path, test_number, filenames, info=None):
        """Get the model from the `analysis_methodology` package. The options are:

            - Autoencoder - can be called by the following name:

                - auto_encoder

            - DeepAnT_CNN - can be called by the following name:

                - deep_ant

            - HistogramConsistencyTest - can be called by the following name:

                - histogram

            - ClusterAnalysisDivergence - can be called by the following name:

                - clustering

            - One_Class_SVM - can be called by the following names:

                - ocsvm

        Parameters
        ----------
        name : str
            Name of the model.
        path : str
            Path to where the `.h5` file of the model is.
        test_number : int
            Number of the running test.
        filenames : dict
            Contains the template name for each model and threshold.
        info : dict, optional
            Contains the additional parameters necessary to create the model. Default is None.

        Returns
        -------
        obj
            The object of the choosen model.

        Raises
        ------
        Exception
            If any parameter for the model declaration was not passed correctly.
        """
        name = str(name).lower()

        if name in "auto_encoder":
            try:
                model = keras.models.load_model(
                    path + filenames["filename"].replace("@", str(test_number))
                )
                threshold = np.load(
                    path + filenames["threshold"].replace("@", str(test_number))
                )

                return Autoencoder(
                    model=model,
                    threshold=threshold,
                )
            except:
                return Autoencoder()

        elif name in "deep_ant":
            try:
                model = keras.models.load_model(
                    path + filenames["filename"].replace("@", str(test_number))
                )
                threshold = np.load(
                    path + filenames["threshold"].replace("@", str(test_number))
                )

                return DeepAnT_CNN(
                    model=model,
                    threshold=threshold,
                )
            except:
                return DeepAnT_CNN()

        elif name in "histogram":
            if info is not None:
                return HistogramConsistencyTest(
                    nominal_rotation=info["nominal_rotation"],
                    p_value_limit=info["p_value_limit"],
                    n_bins=info["n_bins"],
                    filter_type=info["filter_type"],
                    rotational_speed_estimation=info["rotational_speed_estimation"],
                )
            else:
                raise Exception(
                    "To use Histogram in the benchmark you must provide the parameters to instantiate the class!"
                )

        elif name in "clustering":
            if info is not None:
                return ClusterAnalysisDivergence(
                    size_sub_sample=info["size_sub_sample"],
                    step_to_overlaps=info["step_to_overlaps"],
                    discriminators=info["discriminators"],
                    clusterer=info["clusterer"],
                    decision_type=info["decision_type"],
                    percent=info["percent"],
                )
            else:
                raise Exception(
                    "To use Clusters in the benchmark you must provide the parameters to instantiate the class!"
                )

        elif name in "ocsvm":
            if info is not None:
                return One_Class_SVM(
                    discriminators=info["discriminators"],
                    size_buffer_training=info["size_buffer_training"],
                    size_buffer_adjust_threshold=info["size_buffer_adjust_threshold"],
                    kernel=info["kernel"],
                    nu=info["nu"],
                    gamma=info["gamma"],
                    wavelet_decomposition=info["wavelet_decomposition"],
                    wavelet_levels=info["wavelet_levels"],
                )
            else:
                raise Exception(
                    "To use SVM in the benchmark you must provide the parameters to instantiate the class!"
                )

    def load_data(self, path, test_number, filenames):
        """Method to load the data required to run the benchmark.

        Parameters
        ----------
        path : str
            Path to where the `.npy` file of the test is.
        test_number : int
            Number of the running test.
        filenames : dict
            Contains the template name for the data and target.

        Returns
        -------
        data : numpy.ndarray
            Sample data.
        target : numpy.ndarray
            Target data.
        """
        data = np.load(path + filenames["data"].replace("@", str(test_number)))

        config = self.config[f"test_{test_number}"]

        try:
            target = np.load(path + filenames["target"].replace("@", str(test_number)))
        except:
            target = np.zeros(len(data))

        self.fs = config["fs"]
        self.size_sample = config["size_sample"]
        self.size_sub_sample = config["size_sub_pack"]

        self.dt = 1 / self.fs

        if test_number == 6:
            data = data[int(config["data_len"][0]) :, : int(config["data_len"][1])]
            target = target[: len(data)]

        return data, target

    def run(self, test_number, get_individual=False):
        """Runs the benchmark for a specific test.

        Parameters
        ----------
        test_number : int
            Number of the running test.
        get_individual: bool
            Set to verify the individual votes from models. By default is False.
        """
        self.get_individual = get_individual
        self.test_number = test_number
        self._record = []
        self._compare_value = []
        self._compare_limit = []
        self.methodologies_record = dict()

        methodologies = dict()
        self.methodologies_record = dict()
        for model in self.models:

            if model['name'] == 'gates':
                self.threshold_gates = model["info"]["threshold_gates"]
                self.nominal_rotation = model["info"]["nominal_rotation"]
                self.aquisition_frequency = model["info"]["aquisition_frequency"]

            filenames = model.copy()
            filenames.pop("name")
            try:
                methodologies[model["name"]] = self.get_model(
                    name=model["name"],
                    path=self.path,
                    test_number=test_number,
                    filenames=filenames,
                    info=model["info"],
                )
            except:
                methodologies[model["name"]] = self.get_model(
                    name=model["name"],
                    path=self.path,
                    test_number=test_number,
                    filenames=filenames,
                )
                
            self.methodologies_record[model["name"]] = [] 
        

        methodology = ArchitectureWithDeepAnt(
            methodologies["deep_ant"],
            [methodologies[key] for key in methodologies if key != "deep_ant" and key != "gates"],
            self.threshold_gates,
            self.aquisition_frequency, 
            self.nominal_rotation
        )

        data, target = self.load_data(
            path=self.path, test_number=test_number, filenames=self.filenames
        )

        methodology.set_dt(self.dt)

        n_samples = int(data.shape[0] / self.size_sample)
        t0 = time.time()

        # filtrar as amostras para treinamento
        for i in range(0, n_samples):
            if i % 100 == 0:
                print(i, "of", n_samples, "samples analyzed...")
                print("Time for 100 samples = " + str(time.time() - t0) + " s")
                t0 = time.time()
            
            sample_compare = data[i * self.size_sample : (i + 1) * self.size_sample]

            buffer_samples_to_train = methodology.send_sample(sample_compare, train=True)


        # chama o treinamento
        for i in range(0, int(len(buffer_samples_to_train)/self.size_sample)):

            if i % 100 == 0:
                print(i, "of", int(len(buffer_samples_to_train)/self.size_sample), "samples will be used for training...")
                print("Time for 100 samples = " + str(time.time() - t0) + " s")
                t0 = time.time()

            sample_compare = buffer_samples_to_train[i * self.size_sample : (i + 1) * self.size_sample]

            if len(sample_compare.shape) == 1:
                sample_compare = sample_compare.reshape((len(sample_compare), 1))

            if get_individual:
                votes = methodology.send_sample(sample_compare)
            else:
                methodology.send_sample(sample_compare)

        ## analisar os samples com o treinamento já realizado
        for i in range(0, n_samples):
            if i % 100 == 0:
                print(i, "of", n_samples, "samples analyzed...")
                print("Time for 100 samples = " + str(time.time() - t0) + " s")
                t0 = time.time()

            sample_compare = data[i * self.size_sample : (i + 1) * self.size_sample]


            if len(sample_compare.shape) == 1:
                sample_compare = sample_compare.reshape((len(sample_compare), 1))

            if get_individual:
                votes = methodology.send_sample(sample_compare)
            else:
                methodology.send_sample(sample_compare)

            if methodology.are_samples_distinct() is not None:
                
                if get_individual:

                    for model in self.models:
                        
                        if (model["name"] not in votes) or isinstance(votes[model["name"]], float):                         
                            votes[model["name"]] = False
                        
                        self.methodologies_record[model["name"]].append(votes[model["name"]])

                self._record.append(methodology.are_samples_distinct())
                self._compare_value.append(methodology.compare_value())
                self._compare_limit.append(methodology.compare_limit)

        for i in range(n_samples - len(self._record)):
            self._record.append(True)

        print(n_samples, "of", n_samples, "samples analyzed!")

        self.data = data
        self.target = target

    def _get_target_per_sample(self, target):
        """Inner method to mount the target array for each sample in the dataset.

        Parameters
        ----------
        target : numpy.ndarray
            Target data.

        Returns
        -------
        target_per_sample : numpy.ndarray
            Target array for each sample.
        """
        n_samples = int(self.data.shape[0] / self.size_sample)
        target_per_sample = np.zeros(n_samples)

        for i in range(0, n_samples):
            target_i = target[i * self.size_sample : (i + 1) * self.size_sample]
            if any(target_i):
                target_per_sample[i] = 1

        return target_per_sample

    def _get_metrics(self):
        """Calculates the metrics of the benchmark. The default metrics used are:

            - Confusion matrix params:

                - True positives
                - False positives
                - False negatives
                - True negatives

            - Accuracy.
            - Recall.
            - Precision.
            - F1 score

        Returns
        -------
        dict
            Containing the metrics
        """
        target_per_sample = self._get_target_per_sample(self.target)
        tn, fp, fn, tp = confusion_matrix(target_per_sample, self.record).ravel()
        accuracy = accuracy_score(target_per_sample, self.record)
        recall = recall_score(target_per_sample, self.record)
        precision = precision_score(target_per_sample, self.record)
        f1 = f1_score(target_per_sample, self.record)

        return {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "accuracy_score": accuracy,
            "recall_score": recall,
            "precision_score": precision,
            "f1_score": f1,
        }

    @property
    def record(self):
        return np.array(self._record)

    @property
    def compare_value(self):
        return np.array(self._compare_value)

    @property
    def compare_limit(self):
        return np.array(self._compare_limit)

    def full_record_array(self):
        """Mounts the record array with the shape equal to the test data.

        Returns
        -------
        numpy.ndarray
            Record array.
        """
        return np.concatenate(
            [item * np.ones(self.size_sample) for item in self._record]
        )
    
    def full_record_model(self, model_name):
        """Mounts the individual model array with the shape equal to the test data.

        Returns individual responses from models
        -------
        numpy.ndarray
            Methodologies record array.
        """
        return np.concatenate(
            [item * np.ones(self.size_sample) for item in self.methodologies_record[model_name]]
        )

    def full_compare_value_array(self):
        """Mounts the compare value array with the shape equal to the test data.

        Returns
        -------
        numpy.ndarray
            Compare value array.
        """
        return np.concatenate(
            [item * np.ones(self.size_sample) for item in self._compare_value]
        )

    def full_compare_limit_array(self):
        """Mounts the compare limit array with the shape equal to the test data.

        Returns
        -------
        numpy.ndarray
            Compare limit array.
        """
        return np.concatenate(
            [item * np.ones(self.size_sample) for item in self._compare_limit]
        )

    def plot(self, show=True, save=False, path=None):
        """Plots the benchmark. Figures can be saved also.

        If the `get_individual` parameter is True in the method `run`, this method will render two figures:
        
            - Figure 1 : Sensors signal, Target and Global Architecture response.
            - Figure 2 : Signal of first sensor, Target, Global Architecture Response and Individuals models response.

        Parameters
        ----------
        show : bool, optional
            Wheter or not to show the figure. By default True.
        save : bool, optional
            Wheter or not to save the figure. By default False.
        path : str, optional
            Path where to save the figure. Required only if save is True. By default None.
        """
        if len(self.data.shape) == 1:
            self.data = self.data.reshape((len(self.data), 1))

        n_sensors = self.data.shape[1]

        title_list = []

        for index in range(n_sensors):
            title_list.append(f"<b>Sensor n°{index + 1}</b>")

        title_list.append("<b>Anomalia</b>")

        fig = make_subplots(
            rows=n_sensors + 1,
            cols=1,
            shared_xaxes=True,
            subplot_titles=title_list,
        )

        for index in range(n_sensors):
            fig.add_trace(
                go.Scatter(
                    y=self.data[:, index],
                    line=dict(color="#3947FB"),
                    name=f"Sensor n°{index + 1}",
                ),
                row=(index + 1),
                col=1,
            )
            fig.update_yaxes(title_text=f"Amplitude [\u03BCm]", row=(index + 1), col=1)

        fig.add_trace(
            go.Scatter(
                y=self.target,
                line=dict(
                    color="#2FAA5C",
                    width=3,
                ),
                name="Gabarito",
            ),
            row=n_sensors + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                y=self.full_record_array(),
                line=dict(
                    color="#EA4519",
                    width=1.5,
                ),
                name="Arquitetura de Processamento",
            ),
            row=n_sensors + 1,
            col=1,
        )
        fig.update_yaxes(
            ticktext=["Não Salva", "Salva"],
            tickvals=[0, 1],
            row=n_sensors + 1,
            col=1,
        )

        metrics = self._get_metrics()
        x_pos = 1.02
        y_pos = 0.8
        dy = 0.04

        fig.add_annotation(
            font=dict(size=14),
            text="<b>Métricas</b>",
            x=x_pos,
            y=y_pos,
            showarrow=False,
            textangle=0,
            xref="paper",
            yref="paper",
            xanchor="left",
        )

        for key in metrics.keys():
            y_pos -= dy

            if len(key) > 2:
                name = str(key[:-6])
            else:
                name = str(key)

            fig.add_annotation(
                text=f"<b>{name.upper()}</b> : {metrics[key]:.2f}",
                x=x_pos,
                y=y_pos,
                showarrow=False,
                textangle=0,
                xref="paper",
                yref="paper",
                xanchor="left",
            )

        now = datetime.now()

        fig.update_layout(
            title_text=f"<b>Resultados</b> - Benchmark <b>n° {self.test_number}</b> - {now} || Threshold Gates: {self.threshold_gates} || Nominal Rotation: {self.nominal_rotation} || Aquisition Frequency: {self.aquisition_frequency}",
            width=1920, height=1080
        )
        
        if self.get_individual:

            title_list2 = ["Sensor n° 1","Gabarito", "Arquitetura de Processamento"]

            for key in self.methodologies_record:
                name = key.replace("_", " ")
                title_list2.append(f"<b>{name.title()}</b>")

            fig2 = make_subplots(
                rows=len(self.methodologies_record)+3, 
                cols=1,
                shared_xaxes=True,
                subplot_titles=title_list2,
            )
            title_list2.pop(0)

            colors = ["#01E6AE", "#E5AF06", "#E333ff" ,"#1316DC","#016DE6", "#7401E6", "#E301E6", "#E60158"]

            fig2.add_trace(
                go.Scatter(
                    y=self.data[:, 0],
                    line=dict(color="#3947FB"),
                    name="Sensor n° 1",
                ),
                row=1,
                col=1,
            )
            fig2.update_yaxes(title_text=f"Amplitude [\u03BCm]", row=1, col=1)


            fig2.add_trace(
                go.Scatter(
                    y=self.target,
                    line=dict(
                        color="#2FAA5C",
                        width=2,
                    ),
                    name="Gabarito",
                ),
                row=2,
                col=1,
            )
            title_list2.pop(0)
            
            fig2.update_yaxes(
                ticktext=["Não Salva", "Salva"],
                tickvals=[0, 1],
                row=2,
                col=1,
            )

            fig2.add_trace(
                go.Scatter(
                    y=self.full_record_array(),
                    line=dict(
                        color="#EA4519",
                        width=3,
                    ),
                    name="Arquitetura de Processamento",
                ),
                row=3,
                col=1,
            )
            title_list2.pop(0)

            fig2.update_yaxes(
                ticktext=["Não Salva", "Salva"],
                tickvals=[0, 1],
                row=3,
                col=1,
            )

            for index, key in enumerate(self.methodologies_record):

                fig2.add_trace(
                    go.Scatter(
                        y=self.full_record_model(key),
                        line=dict(
                            color=colors[index],
                            width=2,
                        ),
                        name=title_list2[index],
                    ),
                    row=index+4,
                    col=1,
                )

                fig2.update_yaxes(
                ticktext=["Não Salva", "Salva"],
                tickvals=[0, 1],
                row=index+4,
                col=1,
                )

            fig2.update_layout(
                title_text=f"<b>Resultados</b> - Benchmark <b>n° {self.test_number}</b> - {now} - Resposta Final e Resposta de cada Método || Threshold Gates: {self.threshold_gates} || Nominal Rotation: {self.nominal_rotation} || Aquisition Frequency: {self.aquisition_frequency}",
                width=1920, height=1080
        )
            
        metrics = self._get_metrics()
        x_pos = 1.02
        y_pos = 0.8
        dy = 0.04

        fig2.add_annotation(
            font=dict(size=14),
            text="<b>Métricas</b>",
            x=x_pos,
            y=y_pos,
            showarrow=False,
            textangle=0,
            xref="paper",
            yref="paper",
            xanchor="left",
        )

        for key in metrics.keys():
            y_pos -= dy

            if len(key) > 2:
                name = str(key[:-6])
            else:
                name = str(key)

            fig2.add_annotation(
                text=f"<b>{name.upper()}</b> : {metrics[key]:.2f}",
                x=x_pos,
                y=y_pos,
                showarrow=False,
                textangle=0,
                xref="paper",
                yref="paper",
                xanchor="left",
            )

        if save:
            if path is None:
                path = os.getcwd() + "/"
            else:
                if path[-1] != "/":
                    path += "/"


            date = now.strftime("%d_%m_%Y_%H_%M_%S")

            fig.write_image(path + f"result_{date}.png")

            if self.get_individual:
                fig2.write_image(path + f"individual_result_{date}.png")

            if self.get_individual:
                fig2.write_image(path + f"individual_result_{date}.png")

        if show:
            fig.show()

            if self.get_individual:
                fig2.show()



def benchmark_example(path, test_number):
    """Benchmark example using the default benchmark in the EDGE R&D project.

    Parameters
    ----------
    path : str
        Path to where the `.npy` file of the test is.
    test_number : int
        Number of the running test.
    get_individual: bool
        To get the plot from individual response from models set get_individual=True. If it is not set up the default is False.


    Returns
    -------
    obj
        Benchmark class object.
    """
    data = "default"

    benchmark = Benchmark(data, path)

    benchmark.run(test_number=test_number, get_individual=True)

    return benchmark
