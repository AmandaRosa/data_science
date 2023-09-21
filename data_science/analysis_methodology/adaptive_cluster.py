import sys

import numpy as np

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

from scipy.spatial.distance import jensenshannon
from sklearn import cluster, mixture
from sklearn.decomposition import *
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestCentroid, kneighbors_graph
from sklearn.preprocessing import *

from data_science.clustering import EstimateResources

from .abstract import Methodology


class ClusterAnalysisDivergence(Methodology):
    """This class explores the particularity of each cluster in detecting anomalies. Decision functions based on distance, scores and Kulback Leibler divergence are explored. The reference and comparison samples of the vibrational signal captured by the sensors are divided into sub-samples and subjected to an oversampling. The resources are calculated and the groupers are applied to the dataset referring to the reference(i-1) and comparison(i) sample. Based on a threshold, either user-defined or automatically calculated, and voting on the responses of the decision functions it is inferred whether the two samples are normal or anomalous.

    The following assumptions are made:
        - Observations in each sample are independent and identically distributed;
        - Observations in each sample are normally distributed.

    Parameters
    ----------
    n_channels : bool, optional
        Sensors insertion. If set as True, number of sensors is greater than 1. If it is False, number of sensors is equal to 1. Default is True.
    nominal_rotation : float, optional
        Rotation speed. Defaults to None.
    size_sub_sample : int, optional
        Size of each input sub-sample. Default value of 256.
    oversampling : bool, optional
        Function for oversampling based in overlap. Default is True.
    step_to_overlaps : int, optional
        Time step for overlap. Default value is 50.
    discriminators : list, optional
        Vibration signal discriminators set by the user. Defaults to "rms".
    reduction : str, optional
        Method for dimensionality reduction. Defaults is None. Options are:
            - 'pca' : Principal Components Analysis,
            - 'kernelPCA' : Principal Components Analysis to non linear data,
    nb_components : float, optional
        Number of principal components. Default value is 3.
    nb_cluster : int, optional
        Number of clusters, needs to be an value equal or greater than 2. Default is 4.
    clusterer : str, optional
        Clusterer used. Default is 'gaussian'. Options are:
            - 'gaussian' uses aussian mixture model probability distribution,
            - 'hierarchical' uses agglomerative Clustering by linkage distance,
            - 'hierarchical birch' uses BIRCH clustering algorithm,
            - 'spectral' uses clustering to a projection of the normalized Laplacian.
    decision_type : str, optional
        Decision function type. Default is'distance'. Options are:
            - 'None' uses distances and divergence for decision function,
            - 'distance' uses only distance for decision function,
            - 'divergence' uses only scores for decision function.
    percent : int, optional
        Percentage to be calculated. Must be a value between 0 and 100 inclusive. Default is 80.
    threshold : float, optional
        Divergence value determined by user. Default is 0.6.
    sigma : float, optional
        Parameter for threshold based on mean and standard deviation, as Threshold = mu(X) + sigma*std(X). Default is 1.
    size_buffer : int, optional
        Number of samples to estimate the threshold, must be equal or greater than 1. Default is 2.
        Obs.: If threshold is None, size_buffer needs to defined.
    """

    NAME = "clustering"

    def __init__(
        self,
        n_channels=True,
        nominal_rotation=None,
        size_sub_sample=256,
        oversampling=False,
        step_to_overlaps=50,
        discriminators=["rms"],
        reduction=None,
        nb_components=3,
        nb_cluster=4,
        clusterer="gaussian",
        decision_type="divergence",
        percent=80,
        threshold=None,
        sigma=5,
        size_buffer=10,
    ):
        self.n_channels = n_channels
        self.nominal_rotation = nominal_rotation
        self.size_sub_sample = size_sub_sample
        self.oversampling = oversampling
        self.step_to_overlaps = step_to_overlaps
        self.discriminators = discriminators
        self.reduction = reduction
        self.nb_components = nb_components
        self.nb_cluster = nb_cluster
        self.clusterer = clusterer
        self.decision_type = decision_type
        self.percent = percent
        self.threshold = threshold
        self.sigma = sigma
        self.size_buffer = size_buffer
        # self.cust_computational = []

        if self.threshold is None:
            self.flag_compute_threshold = True
            self.size_buffer = self.size_buffer
            self.buffer_anomaly_score = []
            self.threshold = 0

        else:
            self.flag_compute_threshold = False
            self.size_buffer = None

        super().__init__(self.threshold)
        self.sample_base = None

        self.compare_value_scaler = 0

        self.all_data_previous = None

    def set_dt(self, dt):
        """Sets the time step of the acquired signal.

        Parameters
        ----------
        dt : float
            Time step of the signal that will be passed to the CNN.
        """
        self.dt = dt

    def send_sample(self, sample_compare):
        """Starts the classification of the last acquired signal. If the training process did not occurred, it will do it. After been trained, it tries to predict each subsequent data window, hence it is required a small data buffer containing the Mean Absolute Error (MAE) values. An adaptable threshold is calculated for the predicted signal window and it is compared with the input signal to decide if it is anomalous or not.

        Parameters
        ----------
        sample_compare : numpy.ndarray
            The last acquired signal. This array's shape is (number of points, number of sensors).

        """
        # t0 = time.time()

        # sample_compare_shape = sample_compare.shape
        # sample_compare = np.ones(sample_compare_shape)

        if sample_compare.ndim == 1:
            sample_compare = sample_compare.reshape((len(sample_compare), 1))

        if self.sample_base is None:
            self.sample_base = sample_compare
            self.decision_value = 1
            return self.decision_value

        if self.sample_base.shape != sample_compare.shape:
            raise Exception(
                f"Samples must have the same shape. Sample Base: {self.sample_base.shape} != Sample Compare: {sample_compare.shape}"
            )

        self.sample_compare = sample_compare
        self.pre_processing_model_(self.sample_base, self.sample_compare)
        self.decision_model_()

        if self.are_samples_distinct():
            self.sample_base = self.sample_compare

        # self.cust_computational.append(time.time() - t0)

        self.all_data_previous = self.all_data.copy()

    def set_sample(self, sample_compare):
        """Alternative method to trigger the anomaly analyses.

        Parameters
        ----------
        sample_compare : numpy.ndarray
            The last acquired signal. This array's shape is (number of points, number of sensors).
        """
        self.sample_base = sample_compare

    def are_samples_distinct(self):
        """Returns wether or not the last samples is an anomaly.

        Returns
        -------
        bool
            Boolean value for classification. Options are:
                True : the samples are distinct.
                False : the samples are similar.
        """
        return self.decision_value >= self.threshold

    def compare_value(self):
        """Gives the classification value.

        Returns
        -------
        compare_value_scaler : float
            Classification value.
        """
        return self.compare_value_scaler
        # return self.decision_value

    def pre_processing_model_(self, base, compare):
        """Estimate the resources chosen by the user to represent each of the vibration signals,l normalization and reduction type to be defined by user."""
        estimate_resources = EstimateResources(
            dt=self.dt,
            multiple_channels=self.n_channels,
            nominal_rotation=self.nominal_rotation,
            slice_size=self.size_sub_sample,
            oversampling=self.oversampling,
            step_to_overlaps=self.step_to_overlaps,
            discriminators=self.discriminators,
        )
        self.data_base = estimate_resources.get_resources(self.sample_base)

        estimate_resources = EstimateResources(
            dt=self.dt,
            multiple_channels=self.n_channels,
            nominal_rotation=self.nominal_rotation,
            slice_size=self.size_sub_sample,
            oversampling=self.oversampling,
            step_to_overlaps=self.step_to_overlaps,
            discriminators=self.discriminators,
        )
        self.data_compare = estimate_resources.get_resources(self.sample_compare)

        self.data_base = np.reshape(
            self.data_base, (len(self.data_base) * len(self.discriminators), -1)
        )
        self.data_compare = np.reshape(
            self.data_compare, (len(self.data_compare) * len(self.discriminators), -1)
        )

        self.scaler = MinMaxScaler().fit(self.data_base)
        self.data_base = self.scaler.transform(self.data_base)
        self.data_compare = self.scaler.transform(self.data_compare)

        # self.data_base = self.resources_base
        # self.data_compare = self.resources_compare

        if self.reduction == "pca":
            self.reduct = PCA(n_components=self.nb_components).fit(self.data_base)
            self.data_base = self.reduct.transform(self.data_base)
            self.data_compare = self.reduct.transform(self.data_compare)

        elif self.reduction == "kernelPCA":
            self.reduct = KernelPCA(
                n_components=self.nb_components,
                kernel="rbf",
            ).fit(self.data_base)
            self.data_base = self.reduct.transform(self.data_base)
            self.data_compare = self.reduct.transform(self.data_compare)

        self.all_data = np.concatenate((self.data_base, self.data_compare), axis=0)

        ### TESTE
        # if len(self.buffer_anomaly_score) == self.size_buffer and self.flag_compute_threshold:
        #     self.all_data[0,:] = np.nan

        if np.any(~np.isfinite(self.all_data)):
            if self.all_data_previous is not None:
                self.all_data_previous = np.ones(self.all_data.shape)
            self.all_data = self.all_data_previous

    def decision_model_(
        self,
    ):
        """Applied decision functions based in distances, scores and Kulback Leibler divergence."""
        if self.decision_type is None:
            self.decision_distance = self.decision_function_by_distances_()
            self.decision_divergence = self.decision_function_by_divergence_()
            self.decision_value = np.mean(
                (0.6 * self.decision_distance, 0.4 * self.decision_divergence), axis=0
            )

        elif self.decision_type == "distance":
            self.decision_distance = self.decision_function_by_distances_()
            self.decision_value = self.decision_distance

        elif self.decision_type == "divergence":
            self.decision_divergence = self.decision_function_by_divergence_()
            self.decision_value = self.decision_divergence

    def calculate_n_neighbors_(self, data):
        """Calculates the number of neighbors and passes it to variable self.n_neighbors.

        Parameters
        ----------
        data : numpy.ndarray
            Array with the data.
        """
        self.n_neighbors = int(np.log(len(data)))

    def hierarchical_clusterer_(self, data):
        """Mount the clusters based on the agglomerative method.

        Parameters
        ----------
        data : numpy.ndarray
            Array with the data.
        """
        self.clustering_model = cluster.AgglomerativeClustering(
            n_clusters=self.nb_cluster,
            connectivity=self.connectivity_matrix,
            linkage="ward",
            affinity="euclidean",
        ).fit(data)

    def connectivity_matrix_(self, data):
        """Mount the conectivity matrix based on the calculated neighborhood.

        Parameters
        ----------
        data : numpy.ndarray
            Array with the data.
        """
        connectivity_matrix = kneighbors_graph(
            data, n_neighbors=self.n_neighbors, include_self=False
        )
        self.connectivity_matrix = 0.5 * (
            connectivity_matrix + connectivity_matrix.T
        )  # symmetric

    def hierarchical_birch_clusterer_(self, data):
        """Mount the clusters based on the BIRCH algorithm.

        Parameters
        ----------
        data : numpy.ndarray
            Array with the data.
        """
        self.clustering_model = cluster.Birch(
            n_clusters=self.nb_cluster,
            branching_factor=50,
        ).fit(data)

    def spectral_clusterer_(self, data):
        """Mount the clusters based on the spectral method.

        Parameters
        ----------
        data : numpy.ndarray
            Array with the data.
        """
        self.clustering_model = cluster.SpectralClustering(
            n_clusters=self.nb_cluster,
            eigen_solver="arpack",
            n_neighbors=self.n_neighbors,
            affinity="nearest_neighbors",
            assign_labels="discretize",
        ).fit(data)

    def gaussian_clusterer_(self, data):
        """Mount the clusters based on the Gaussian mixture.

        Parameters
        ----------
        data : numpy.ndarray
            Array with the data.
        """
        self.clustering_model = mixture.GaussianMixture(
            n_components=self.nb_cluster,
            init_params="kmeans",
            n_init=1,
            random_state=0,
            covariance_type="full",
            reg_covar=1e-05,
            max_iter=10000,
            tol=0.0001,
        ).fit(data)

    def decision_function_by_distances_(
        self,
    ):
        """Compare the distances for instances of the acquired sample, based in centers to the base sample."""
        if self.clusterer == "hierarchical":
            self.calculate_n_neighbors_(self.data_base)
            self.connectivity_matrix_(self.data_base)
            self.hierarchical_clusterer_(self.data_base)

        elif self.clusterer == "hierarchical birch":
            self.hierarchical_birch_clusterer_(self.data_base)

        elif self.clusterer == "spectral":
            self.calculate_n_neighbors_(self.data_base)
            self.spectral_clusterer_(self.data_base)

        elif self.clusterer == "gaussian":
            self.gaussian_clusterer_(self.data_base)

        cntrd = NearestCentroid()
        cntrd.fit(self.data_base, self.clustering_model.fit_predict(self.data_base))
        centers = cntrd.centroids_
        dist_base = euclidean_distances(self.data_base, centers)
        max_base = np.max(dist_base, axis=0)
        dist_compare = euclidean_distances(self.data_compare, centers)

        return np.percentile(
            (np.percentile(dist_compare, self.percent) >= max_base), self.percent
        ) >= (self.percent / 100)

    def decision_function_by_divergence_(
        self,
    ):
        """Estimation of the Janson Shanon divergence test to compare the labels."""
        if self.clusterer == "hierarchical":
            self.calculate_n_neighbors_(self.all_data)
            self.connectivity_matrix_(self.all_data)
            self.hierarchical_clusterer_(self.all_data)

        elif self.clusterer == "hierarchical birch":
            self.hierarchical_birch_clusterer_(self.all_data)

        elif self.clusterer == "spectral":
            self.calculate_n_neighbors_(self.all_data)
            self.spectral_clusterer_(self.all_data)

        elif self.clusterer == "gaussian":
            self.gaussian_clusterer_(self.all_data)

        return self.compute_divergence_(
            self.clustering_model.fit_predict(self.all_data)
        )

    def compute_divergence_(self, labels):
        """Calculate the divergence.

        Parameters
        ----------
        labels : numpy.ndarray
            Clustering model prediction array.

        Returns
        -------
        boolean
            True if the samples are distinct, and False if they are similar.
        """

        aux = int(len(labels) / 2)
        preds_base = labels[:aux]
        preds_compare = labels[aux:]
        aux = np.unique(labels)
        count_base = np.zeros(len(aux))
        count_compare = np.zeros(len(aux))

        for i in range(len(aux)):
            count_base[i] = np.count_nonzero((preds_base == aux[i])) / len(preds_base)
            count_compare[i] = np.count_nonzero((preds_compare == aux[i])) / len(
                preds_compare
            )

        p = count_base
        q = count_compare

        self.compare_value_scaler = jensenshannon(p, q, base=2)

        if self.flag_compute_threshold:
            if len(self.buffer_anomaly_score) < self.size_buffer:
                self.buffer_anomaly_score.append(self.compare_value_scaler)

            elif len(self.buffer_anomaly_score) == self.size_buffer:
                self.threshold = self.compute_threshold(self.buffer_anomaly_score)
                super().__init__(self.threshold)
                print(self.threshold)
                self.flag_compute_threshold = False

        return self.compare_value_scaler >= self.threshold

    def compute_threshold(self, buffer):
        """This function estimates threshold value based on buffer.

        Parameters
        ----------
        buffer : list
            List with the compare value scaler.

        Returns
        -------
        mu_anomaly_score : float
            Threshold value.
        """
        self.mu_anomaly_score = np.mean(buffer) + self.sigma * np.std(buffer)

        return self.mu_anomaly_score
