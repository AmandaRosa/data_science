import logging
import time

import numpy as np

from .abstract import Methodology

from ..analysis_methodology import (
    Autoencoder,
    DeepAnT_CNN,
    SimpleAnalysis,
)


class ProcessingArchitecture(Methodology):
    """
    This class combines different models to produce better results. The results are combined using a majority voting approach, where the final result is decided by the majority obtained by combining the models. This class requires the following steps:
    - Results for all models in use available for the signal under analysis. The class only starts to work when the result for all the models used is available (this is because, some models have a delay due to training or moving window).
    - Same signal. The analysis is performed for the results of the models of the same signal.

    Parameters
    ----------
    methodogies : Array of methodogies
        Methodologies to be used to calculate the final result using a majority voting approach.

    Examples
    --------
    >>> import numpy as np
    >>> from data_science import CompareData
    >>> from data_science.analysis_methodology import (
    ...    Autoencoder,
    ...    ClusterAnalysisDivergence,
    ...    DeepAnT_CNN,
    ...    HistogramConsistencyTest,
    ...    ProcessingArchitecture,
    ... )

    Load Data

    >>> data = np.load("data/TESTE01/RespNoiseFiveTEST01.npy")
    >>> dt = data[1, 0] - data[0, 0]
    >>> data = data[:, 1:]

    Define Methodologies

    >>> histogram = HistogramConsistencyTest(
    ...     nominal_rotation=60, filter_type="frequency",
    ...     p_value_limit=0.05
    ... )

    >>> auto_encoder = Autoencoder(
    ...     percent_diff_bias_train=10,
    ...     threshold=5e-4
    ... )

    >>> deep_ant = DeepAnT_CNN(epochs=2)

    >>> adaptative_cluster = ClusterAnalysis(
    ...     slice_size=256,
    ...     nominal_rotation=int(3600 / 60),
    ...     nmaxcluster=2,
    ...     discriminators=["rms", "kurtosis", "peak value"],
    ...     clusterizer="gaussian mixture",
    ...     threshold=0.525,
    ... )

    Define Processing Architecture Methodologie

    >>> processing_architecture = ProcessingArchitecture(
    ...     [histogram, auto_encoder, deep_ant, adaptative_cluster,]
    ... )

    Run comparison

    >>> compare_test = CompareData(data, dt, processing_architecture, slice_size=4000)

    View results

    >>> compare_test.plot()
    >>> processing_architecture.plot()
    >>> metrics = compare_test.get_evaluation_metrics(target)
    """

    def __init__(self, methodogies):
        super().__init__(0.5)

        self.methodologies = methodogies
        self.methodologies_record = {}
        self.processing_architecture_record = []

        for methodology in self.methodologies:
            self.methodologies_record[methodology.NAME] = []

        self.position = 0

    def set_dt(self, dt):
        self.dt = dt
        for methodology in self.methodologies:
            methodology.set_dt(dt)

    def send_sample(self, sample):
        for methodology in self.methodologies:
            methodology.send_sample(sample)
            record = methodology.are_samples_distinct()
            if record is not None:
                self.methodologies_record[methodology.NAME].append(record)

        min_record = np.min(
            [
                len(self.methodologies_record[methodology.NAME])
                for methodology in self.methodologies
            ]
        )
        if self.position < min_record:
            self.record = self.mean_processing_architecture()
            self.processing_architecture_record.append(self.record)
            self.position += 1

        else:
            self.record = None

    def mean_processing_architecture(self):
        result = []

        for record_value in self.methodologies_record.values():
            result.append(record_value[self.position])

        self.predict_mean = np.mean(result)

        return self.predict_mean >= 0.5

    def are_samples_distinct(self):
        return self.record

    def compare_value(self):
        return self.predict_mean


class MethodologiesVotes:
    def __init__(self, methodologies):
        self.methodologies = methodologies
        self.votes = {}
        self.create_vote_list()
        self.majority = len(methodologies) / 2

    def create_vote_list(self):
        for methodology in self.methodologies:
            self.votes[methodology.NAME] = []

    def add_vote(self, methodology, vote):
        if vote is not None:
            self.votes[methodology.NAME].append(vote)

    def get_votes(self, methodology):
        return self.votes[methodology.NAME]

    def get_first_vote(self, methodology):
        methodology_votes = self.votes[methodology.NAME]
        return methodology_votes[0] if len(methodology_votes) > 0 else None

    def get_lowest_number_of_votes(self):
        number_of_votes = [
            len(self.get_votes(methodology)) for methodology in self.methodologies
        ]
        return min(number_of_votes)

    def get_votes_array(self):
        votes = [self.get_first_vote(methodology) for methodology in self.methodologies]
        return np.array(votes)

    def remove_computed_vote(self):
        for methodology in self.methodologies:
            self.votes[methodology.NAME] = self.votes[methodology.NAME][1:]

    def compute_result(self):
        votes = self.get_votes_array()
        if None in votes:
            return None
        else:
            number_of_true_votes = len(votes[votes == True])
            self.remove_computed_vote()
            return number_of_true_votes >= self.majority


class ProcessingArchitectureSimple(Methodology):
    def __init__(self, methodogies):
        self.methodologies = methodogies
        self.votes_handler = MethodologiesVotes(methodogies)
        super().__init__(self.votes_handler.majority)

    def set_dt(self, dt):
        self.dt = dt
        for methodology in self.methodologies:
            methodology.set_dt(dt)

    def send_sample(self, sample):
        logging.info("Starting sample analysis")
        for methodology in self.methodologies:
            ti = time.time()
            methodology.send_sample(sample)
            record = methodology.are_samples_distinct()
            tf = time.time()
            print(f" {methodology.NAME}: {tf-ti:.3f} s")
            self.votes_handler.add_vote(methodology, record)

        self.record = self.votes_handler.compute_result()

    def are_samples_distinct(self):
        return self.record

    def compare_value(self):
        return self.compare_limit


class ArchitectureWithDeepAnt(Methodology):
    
    def __init__(self, deep_ant, methodologies, threshold_gates, aquisition_frequency, nominal_rotation,):
        self.methodologies = methodologies
        self.deep_ant = deep_ant
        self.majority = (len(methodologies) + 1) / 2
        self.samples_buffer = []
        self.threshold_gates = threshold_gates
        self.nominal_rotation = nominal_rotation
        self.aquisition_frequency = aquisition_frequency
        self.preprocessing_buffer_train_samples = list()
        super().__init__(self.majority)

        self.simple = SimpleAnalysis(threshold_for_anomaly = self.threshold_gates, aquisition_frequency=self.aquisition_frequency, nominal_rotation=self.nominal_rotation)

    
    def preprocessing_buffer_train(self, sample, gates_use = True):

        if gates_use:
           
            self.simple.send_sample(sample[:]) 
            
            t0 = time.time()
            self.record = self.simple.are_samples_distinct()

            if self.record == False:
                if len(self.preprocessing_buffer_train_samples)==0:
                    self.preprocessing_buffer_train_samples = sample
                else:
                    self.preprocessing_buffer_train_samples = np.concatenate((self.preprocessing_buffer_train_samples, sample), axis=0)
                    print(self.preprocessing_buffer_train_samples.shape)

            print("             ")
            print(" ======== * Starting pre processing analysis for buffer samples to train * ======== ")
            print(f" {self.simple.NAME}: {time.time()-t0:.3f} s {self.record}")
         
            return self.preprocessing_buffer_train_samples
    
    def set_dt(self, dt):
        self.dt = dt
        self.deep_ant.set_dt(dt)
        for methodology in self.methodologies:
            methodology.set_dt(dt)

    def _create_methodology_votes(self, gates, votes_dict = None):
            
            if gates:
                votes_dict = {}
                votes_dict[self.simple.NAME] = 0
                return votes_dict
        
            else:
                votes_dict = {}
                votes_dict[self.deep_ant.NAME] = 0.5
                for methodology in self.methodologies:
                    votes_dict[methodology.NAME] = 0.5
                return votes_dict
    
    def anomaly_classification(self, sample, gates_use = True):

        if gates_use:
           
            self.simple.send_sample(sample[:]) 
            
            t0 = time.time()
            self.record = self.simple.are_samples_distinct()

            votes_dict = self._create_methodology_votes(gates=True)

            votes_dict[self.simple.NAME] = self.record

            print("             ")
            print(" ======== * Starting gates analysis * ======== ")
            print(f" {self.simple.NAME}: {time.time()-t0:.3f} s {self.record}")
         
            return votes_dict
        else:
            pass

    def send_sample(self, sample, train = False):

        if train == True:

            preprocessing_buffer_train_samples = self.preprocessing_buffer_train(sample)      

            return preprocessing_buffer_train_samples
        
        else:

            print("             ")
            print(" ======== * Starting sample analysis * ======== ")

            votes_dict = self._create_methodology_votes(gates=False)
            self.methodologies_record = votes_dict

            self.samples_buffer.append(sample)

            ti = time.time()
            self.deep_ant.send_sample(self.samples_buffer[0][:])
            record = self.deep_ant.are_samples_distinct()
            votes_dict[self.deep_ant.NAME] = record
            tf = time.time()

            print(f" {self.deep_ant.NAME}: {tf-ti:.3f} s {record}")

            if record is None:
                self.record = None
                return
            else:
                n_votes_true = 1 if record else 0
                n_votes_false = 1 - n_votes_true
                for methodology in self.methodologies:
                    ti = time.time()
                    methodology.send_sample(self.samples_buffer[0][:])
                    record = methodology.are_samples_distinct()
                    tf = time.time()

                    print(f" {methodology.NAME}: {tf-ti:.3f} s {record}")

                    votes_dict[methodology.NAME] = record
                    if record:
                        n_votes_true += 1
                    else:
                        n_votes_false += 1
                    if n_votes_true >= self.majority:
                        self.record = True
                        index = self.methodologies.index(methodology)
                        self._set_sample(
                            self.samples_buffer[0][:], self.methodologies[index + 1 :]
                        )
                        self.samples_buffer = self.samples_buffer[1:]
                        self._votes = n_votes_true
                        return votes_dict
                    elif n_votes_false > self.majority:
                        self.record = False
                        self.samples_buffer = self.samples_buffer[1:]
                        self._votes = n_votes_true
                        return votes_dict

    def _set_sample(self, sample, methodologies):
        for methodology in methodologies:
            methodology.set_sample(sample)

    def are_samples_distinct(self):
        return self.record

    def compare_value(self):
        return self._votes


##############################################################################################################################################


class ArchitectureWithGatesAndDeepAnt(Methodology):
    """
    This class combines different models to produce better results. The results are combined using a majority voting approach, where the final result is decided by the majority obtained by combining the models. This class requires the following steps:
    - Results for all models in use available for the signal under analysis. The class only starts to work when the result for all the models used is available (this is because, some models have a delay due to training or moving window).
    - Same signal. The analysis is performed for the results of the models of the same signal.

    Parameters
    ----------
    methodogies : Array of methodogies
        Methodologies to be used to calculate the final result using a majority voting approach.

    Examples
    --------
    >>> import numpy as np
    >>> from data_science import CompareData
    >>> from data_science.analysis_methodology import (
    ...    Autoencoder,
    ...    ClusterAnalysisDivergence,
    ...    DeepAnT_CNN,
    ...    HistogramConsistencyTest,
    ...    ProcessingArchitecture,
    ... )

    Load Data

    >>> data = np.load("data/TESTE01/RespNoiseFiveTEST01.npy")
    >>> dt = data[1, 0] - data[0, 0]
    >>> data = data[:, 1:]

    Define Methodologies

    >>> histogram = HistogramConsistencyTest(
    ...     nominal_rotation=60, filter_type="frequency",
    ...     p_value_limit=0.05
    ... )

    >>> auto_encoder = Autoencoder(
    ...     percent_diff_bias_train=10,
    ...     threshold=5e-4
    ... )

    >>> adaptative_cluster = ClusterAnalysis(
    ...     slice_size=256,
    ...     nominal_rotation=int(3600 / 60),
    ...     nmaxcluster=2,
    ...     discriminators=["rms", "kurtosis", "peak value"],
    ...     clusterizer="gaussian mixture",
    ...     threshold=0.525,
    ... )

    Define Processing Architecture Methodologie

    >>> processing_architecture = ProcessingArchitecture(
    ...     deep_ant,
    ...     gates,
    ...     training_gates,
    ...     methodologies,
    ...     rpm_threshold (optional)
    ... )

    Run comparison

    >>> compare_test = CompareData(data, dt, processing_architecture, slice_size=4000)

    View results

    >>> compare_test.plot()
    >>> processing_architecture.plot()
    >>> metrics = compare_test.get_evaluation_metrics(target)
    """

    def __init__(self, deep_ant, gates, training_gates, methodologies, rpm_threshold=0):
        self.methodologies = methodologies
        self.simple_analysis = gates
        self.training_gates = training_gates
        self.rpm_threshold = rpm_threshold
        self.deep_ant = deep_ant
        self.majority = (len(methodologies) + 1) / 2
        self.samples_buffer = []
        self.samples_training = []
        self.training_done = False
        super().__init__(self.majority)

    def set_dt(self, dt):
        self.dt = dt
        self.deep_ant.set_dt(dt)
        for methodology in self.methodologies:
            methodology.set_dt(dt)

    def _create_methodology_votes(self):
        votes_dict = {}
        votes_dict[self.simple_analysis.NAME] = 0
        votes_dict[self.deep_ant.NAME] = 0.5
        for methodology in self.methodologies:
            votes_dict[methodology.NAME] = 0.5

        return votes_dict

    def gates_check(self, sample, votes_dict):
        self.simple_analysis.send_sample(sample[:])
        record = self.simple_analysis.are_samples_distinct()
        votes_dict[self.simple_analysis.NAME] = record
        return record

    def send_sample(self, sample):
        print(" ======== * Starting sample analysis * ======== ")
        votes_dict = self._create_methodology_votes()
        self.methodologies_record = votes_dict
        self.samples_buffer.append(sample)
        index = 0
        ti = time.time()
        record = self.gates_check(self.samples_buffer[0][:], votes_dict)
        tf = time.time()
        print(f" {self.simple_analysis.NAME}: {tf-ti:.3f}s {record}")
        if self.simple_analysis.sample_rpm() < self.rpm_threshold:
            self.record = False
            votes_dict[self.simple_analysis.NAME] = False
            self._votes = 0
            return votes_dict
        if record:
            self.record = True
            self.deep_ant.set_sample(sample[:])
            self._set_sample(self.samples_buffer[0][:], self.methodologies[0:])
            self.samples_buffer = self.samples_buffer[1:]
            self._votes = 1
            return votes_dict

        else:
            ti = time.time()
            self.deep_ant.send_sample(sample[:])
            record = self.deep_ant.are_samples_distinct()
            votes_dict[self.deep_ant.NAME] = record
            tf = time.time()

            print(f" {self.deep_ant.NAME}: {tf-ti:.3f} s {record}")

            if record is None:
                self.record = None
                return
            else:
                n_votes_true = 1 if record else 0
                n_votes_false = 1 - n_votes_true
                for methodology in self.methodologies:
                    ti = time.time()
                    methodology.send_sample(self.samples_buffer[0][:])
                    record = methodology.are_samples_distinct()
                    tf = time.time()

                    print(f" {methodology.NAME}: {tf-ti:.3f} s {record}")

                    votes_dict[methodology.NAME] = record
                    if record:
                        n_votes_true += 1
                    else:
                        n_votes_false += 1
                    if n_votes_true >= self.majority:
                        self.record = True
                        index = self.methodologies.index(methodology)
                        self._set_sample(
                            self.samples_buffer[0][:], self.methodologies[index + 1 :]
                        )
                        self.samples_buffer = self.samples_buffer[1:]
                        self._votes = n_votes_true
                        return votes_dict
                    elif n_votes_false > self.majority:
                        self.record = False
                        self.samples_buffer = self.samples_buffer[1:]
                        self._votes = n_votes_true
                        return votes_dict

    def _set_sample(self, sample, methodologies):
        for methodology in methodologies:
            methodology.set_sample(sample)

    def are_samples_distinct(self):
        return self.record

    def compare_value(self):
        return self._votes

    def gather_training_set(self, sample):
        self.record = None
        self.training_gates.send_sample(sample[:])
        answer_gates = self.training_gates.are_samples_distinct()
        for methodology in self.methodologies:
            if methodology.NAME == "histogram":
                methodology.send_sample(sample[:])
                answer_histogram = methodology.are_samples_distinct()
                break
        if answer_gates == False and answer_histogram == False:
            self.samples_training.append(sample)
            if len(self.samples_training) == self.deep_ant.threshold_calibration_window:
                # conversa com a fila que nao precisa mais de amostras por que esta treinando
                self.training_done = True
                for samp in self.samples_training:
                    self.send_sample(samp)
            return True
        else:
            return False


##############################################################################################################################################


class ArchitectureWithGatesAndDeepAntIndividualSensor(Methodology):
    def __init__(self, deep_ant, simple_analysis, methodologies, rpm_threshold=0):
        self.methodologies = methodologies
        self.simple_analysis = simple_analysis
        self.rpm_threshold = rpm_threshold
        self.deep_ant = deep_ant
        self.majority = (len(methodologies) + 1) / 2
        self.samples_buffer = []
        super().__init__(self.majority)

    def set_dt(self, dt):
        self.dt = dt
        for da in self.deep_ant:
            da.set_dt(dt)
        for methodology in self.methodologies:
            for m in methodology:
                m.set_dt(dt)

    def _create_methodology_votes(self):
        votes_dict = {}
        votes_dict[self.simple_analysis.NAME] = 0
        votes_dict[self.deep_ant[0].NAME] = 0.5
        for methodology in self.methodologies:
            votes_dict[methodology[0].NAME] = 0.5

        return votes_dict

    def gates_check(self, sample, votes_dict):
        self.simple_analysis.send_sample(sample[:])
        record = self.simple_analysis.are_samples_distinct()
        votes_dict[self.simple_analysis.NAME] = record
        return record

    def send_sample(self, Sample):
        print(" ======== * Starting sample analysis * ======== ")
        votes_dict = self._create_methodology_votes()
        self.methodologies_record = votes_dict
        self.samples_buffer.append(Sample)
        count = -1
        for sample, deep_ant, methodologies in zip(
            np.transpose(self.samples_buffer[0][:], (-1, 0)),
            self.deep_ant,
            np.transpose(self.methodologies, (-1, 0)),
        ):
            sample = sample.reshape(len(sample), 1)
            count += 1
            ti = time.time()
            record = self.gates_check(sample[:], votes_dict)
            tf = time.time()
            print(
                f" sensor {count}, {self.simple_analysis.NAME}: {tf-ti:.3f}s {record}"
            )
            if self.simple_analysis.sample_rpm() < self.rpm_threshold:
                self.record = False
                votes_dict[self.simple_analysis.NAME] = False
                self._votes = 0
                return votes_dict
            if record:
                self.record = True
                self.deep_ant[count].set_sample(sample[:])
                self._set_sample(method_index=0, sig_count=0)
                self.samples_buffer = self.samples_buffer[1:]
                self._votes = 1
                return votes_dict

            else:
                ti = time.time()
                self.deep_ant[count].send_sample(sample[:])
                record = self.deep_ant[count].are_samples_distinct()
                votes_dict[self.deep_ant[count].NAME] = record
                tf = time.time()

                print(
                    f" sensor {count}, {self.deep_ant[0].NAME}: {tf-ti:.3f} s {record}"
                )

                if record is None:
                    self.record = None
                    return
                else:
                    n_votes_true = 1 if record else 0
                    n_votes_false = 1 - n_votes_true
                    for methodology in self.methodologies:
                        ti = time.time()
                        methodology[count].send_sample(sample[:])
                        record = methodology[count].are_samples_distinct()
                        tf = time.time()

                        print(
                            f" sensor {count}, {methodology[0].NAME}: {tf-ti:.3f} s {record}"
                        )

                        votes_dict[methodology[count].NAME] = record
                        if record:
                            n_votes_true += 1
                        else:
                            n_votes_false += 1
                        if n_votes_true >= self.majority:
                            self.record = True
                            index = self.methodologies.index(methodology)
                            self._set_sample(method_index=index + 1, sig_count=count)
                            self.samples_buffer = self.samples_buffer[1:]
                            self._votes = n_votes_true
                            return votes_dict
                        elif n_votes_false > self.majority:
                            self.record = False

        self.samples_buffer = self.samples_buffer[1:]
        self._votes = n_votes_true
        return votes_dict

    def _set_sample(self, method_index, sig_count):
        c = 0
        mi = 0
        for sample, methodologies in zip(
            np.transpose(self.samples_buffer[0][:], (-1, 0)),
            np.transpose(self.methodologies, (-1, 0)),
        ):
            if c >= sig_count:
                sample = sample.reshape(len(sample), 1)
                for methodology in methodologies:
                    if mi >= method_index:
                        methodology.set_sample(sample[:])
                    else:
                        mi += 1
            else:
                c += 1

    def are_samples_distinct(self):
        return self.record

    def compare_value(self):
        return self._votes
