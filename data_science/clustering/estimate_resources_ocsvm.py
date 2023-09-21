import numpy as np
import pandas as pd
import pywt
from scipy.stats import kurtosis, skew
from sklearn.metrics import mean_squared_error as rmse


class EstimateResourcesOCSVM:
    """This class estimates the resources chosen by the user to represent each of the vibration signals."""

    def __init__(
        self,
        dt,
        slice_size=4096,
        steps_to_overlap=None,
        discriminators=["rms"],
        wavelet_decomposition=False,
        mother_wavelet="db4",
        wavelet_levels=5,
    ):
        self.dt = dt
        self.slice_size = slice_size
        if steps_to_overlap is None:
            self.steps_to_overlap = int(self.slice_size / 10)
        else:
            self.steps_to_overlap = steps_to_overlap
        self.discriminators = discriminators
        self.wavelet_decomposition = wavelet_decomposition
        self.mother_wavelet = mother_wavelet
        self.wavelet_levels = wavelet_levels

        if self.wavelet_levels == "max":
            self.wavelet_levels = pywt.dwt_max_level(
                int(1 / self.dt), self.mother_wavelet
            )

    def get_resources(self, signals, slicing=False, overlapping=False):
        resources_list = list()
        self.n_channels = signals.shape[1]

        for i in range(self.n_channels):
            if slicing is True:
                if overlapping is True:
                    samples = self.slice_vector(
                        signals[:, i],
                        slice_size=self.slice_size,
                        overlap_size=self.steps_to_overlap,
                    )
                else:
                    samples = self.slice_vector(
                        signals[:, i],
                        slice_size=self.slice_size,
                        overlap_size=0,
                    )
            else:
                samples = signals[:, i : i + 1]

            if self.wavelet_decomposition:
                num_samples = samples.shape[1]
                resources = list()
                for sample_index in range(num_samples):
                    wv_coeffs = self.get_wavelet_coefficients(
                        samples[:, sample_index].reshape(-1, 1)
                    )
                    resources_slice = self.estime_resources_with_wavelet(wv_coeffs)
                    resources.append(resources_slice)
                resources = np.array(resources)
            elif not self.wavelet_decomposition:
                resources = self.estime_resources(samples)
            resources_list.append(resources)

        return resources_list

    def slice_vector(self, vector, slice_size: int, overlap_size=0):
        """
        This function slices a signal into many samples in sequence, with a given overlap.

        Parameters
        ----------
        vector : numpy array
            Vector to be sliced.
        slice_size : int
            Size of each slice, or sample.
        overlap_size : int
            Number of data points overlapping.

        Returns
        -------
        slices : numpy array
            All slices.

        Examples
        --------
        >>> samples = slice_vector_sequentially(data, 6720)
        """

        if vector.ndim == 2:
            vector = vector.reshape(-1)
        elif vector.ndim > 2:
            raise Exception("Number of dimensions must be less than 2.")

        indexes_list = np.arange(0, len(vector), slice_size - overlap_size)

        slices = [vector[i : i + slice_size] for i in indexes_list]

        # Check if there are missing values in the last slice windows
        if len(slices[-1]) != slice_size:
            # If so, the last windows will be the last 'slice_size' points
            slices[-1] = vector[-slice_size:]

        slices = np.array(slices).T

        return slices

    def get_wavelet_coefficients(self, data):
        # coeffs_list = list()
        num_slices = data.shape[1]
        for slice_index in range(num_slices):
            coeffs = pywt.wavedec(
                data[:, slice_index], self.mother_wavelet, level=self.wavelet_levels
            )
            # coeffs_list.append(coeffs)

            ### Add the raw signal to the coeffs
            # coeffs.append(data[:,slice_index])

        return coeffs

    def get_wavelet_coefficient2(self, data):
        coeffs_list = list()
        num_slices = data.shape[1]
        for slice_index in range(num_slices):
            cA = data[:, slice_index]
            for level_index in range(self.wavelet_levels):
                (cA, cD) = pywt.dwt(cA, self.mother_wavelet)

                coeffs_list.append((cA, cD))

        return coeffs_list

    def estime_resources(self, sample):
        """This function estimates resources for each sample.

        Parameters
        ----------
        sample : numpy.array
            Any sample (a vibration signal)

        Returns
        -------
        feature_matrix : numpy.array
            Resources
        """

        # Remove the mean from signal
        sample -= np.mean(sample)

        features = []

        for i in range(len(self.discriminators)):
            features.append(self.discriminators[i])

        feature_matrix = pd.DataFrame(
            columns=features, index=range(sample.shape[1]), dtype="int"
        )

        ### RMS
        if "rms" in feature_matrix.columns:
            zero_vector = np.zeros((sample.shape))
            RMS = rmse(sample, zero_vector, multioutput="raw_values", squared=False)
            feature_matrix["rms"] = RMS

        ### Peak value
        if "peak value" in feature_matrix.columns:
            peak_value = RMS * np.sqrt(2)
            feature_matrix["peak value"] = peak_value

        ### Kurtosis
        if "kurtosis" in feature_matrix.columns:
            kurt = kurtosis(sample)
            feature_matrix["kurtosis"] = kurt

        ### Skewness
        if "skewness" in feature_matrix.columns:
            skewness = skew(sample)
            feature_matrix["skewness"] = skewness

        feature_matrix = np.array(feature_matrix)

        return feature_matrix

    def estime_resources_with_wavelet(self, sample):
        num_coeffs = len(sample)
        feature_matrix = list()
        for level in range(num_coeffs):
            current_data = np.array(sample[level])
            zero_vector = np.zeros_like(current_data)
            RMS = rmse(
                current_data, zero_vector, multioutput="raw_values", squared=False
            )
            feature_matrix.append(RMS)
        feature_matrix = np.array(feature_matrix).reshape(-1)

        # self.PLOT_APAGAR_WAVELET_COEFFS(sample,feature_matrix)

        return feature_matrix
