import math

import numpy as np
import pandas as pd
import scipy as sc
from scipy import signal
from scipy.fftpack import fft
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
from sklearn.metrics import mean_squared_error as rmse


class EstimateResources:
    """This class estimates the resources chosen by the user to represent each of the vibration signals [1]_.

    Parameters
    ----------
    signals : numpy.array
        Vibration signal.
    dt : float
        Time step.
    multiple_channels : bool
        Sensors insertion. If multiple_channels = True, number sensors > 1. If multiple_channels = False, number sensors = 1.
    nominal_rotation : float, optional
        Nominal rotation. Defaults to None.
    slice_size : int, optional
        Number of points of each sample. Defaults to 4096.
    oversampling : bool
        Function for oversampling based in overlap. Default to False.
    step_to_overlaps : int, optional
        Time step for overlap. Default to 50.
    discriminators : list, optional
        Resourses set by the user. Defaults to ['rms'].

    References
    ----------
    .. [1] Chegini, S. N., Bagheri, A., & Najafi, F. (2019). A new intelligent fault diagnosis method for bearing in different speeds based on the FDAF-score algorithm, binary particle swarm optimization, and support vector machine. Soft Computing, 1-19.
    """

    def __init__(
        self,
        # signals,
        dt,
        multiple_channels=True,
        nominal_rotation=None,
        slice_size=4096,
        oversampling=False,
        step_to_overlaps=50,
        discriminators=["rms"],
    ):
        self.dt = dt
        self.multiple_channels = multiple_channels
        self.nominal_rotation = nominal_rotation
        self.slice_size = slice_size
        self.oversampling = oversampling
        self.step_to_overlaps = step_to_overlaps
        self.discriminators = discriminators

        # self.filter_rpm = [] # Digital coefficients of the band-pass filter used in rotation speed estimate
        self.filter_number = 0

    def get_resources(self, signals):
        self.resources = list()

        if self.multiple_channels is True:
            self.n_channels = signals.shape[1]

            for i in range(self.n_channels):
                if self.oversampling is True:
                    self.samples = self.slice_vector_with_overlap(
                        signals[:, i],
                        self.slice_size,
                        self.step_to_overlaps,
                    )

                else:
                    self.samples = self._get_slice_vector_sequentially(
                        signals[:, i],
                        self.slice_size,
                    )
                resources = self.estime_resources(self.samples)
                self.resources.append(resources)

            self.resources = np.concatenate(self.resources, axis=1)

        else:
            if self.oversampling is True:
                self.samples = self.slice_vector_with_overlap(
                    signals, self.slice_size, self.step_to_overlaps
                )

            else:
                self.samples = self._get_slice_vector_sequentially(
                    signals, self.slice_size
                )
            resources = self.estime_resources(self.samples.reshape(self.slice_size, -1))
            self.resources.append(resources)
            self.resources = np.concatenate(self.resources, axis=1)

        return self.resources

    def _get_slice_vector_sequentially(self, sinal, slice_size):
        """This function slices a signal into many samples in sequence, with no overlap.

        Parameters
        ----------
        sinal : numpy.array)
            Signal to be sliced.
        slice_size : int
            Size of each slice, or sample.

        Returns
        -------
        slices: numpy.array
            Sliced signal.
        """
        slice_size: int
        num_slices = math.floor((len(sinal) / slice_size))
        last_index = slice_size * num_slices
        index_slice = np.arange(0, last_index, slice_size)
        slices = np.array([sinal[index : index + slice_size] for index in index_slice])
        return slices.T

    def slice_vector_with_overlap(self, vector, slice_size: int, steps: int):
        """
        This function slices a signal into many samples in sequence, with a given overlap.

        Parameters
        ----------
        vector : numpy array
            Vector to be sliced.
        slice_size : int
            Size of each slice, or sample.
        steps : int
            Size of the step between the beginning of two consecutive windows.
            The overlap is equal to the slice size minus the steps.

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
        num_slices = math.floor((len(vector) - slice_size) / steps) + 1
        # last_index = slice_size + (num_slices*steps)
        last_index = num_slices * steps
        index_slice = np.arange(0, last_index, steps)
        slices = [vector[index : index + slice_size] for index in index_slice]
        slices = np.array(slices)

        return slices.T

    def _my_fft(self, sample, fs):
        """This function estimates the Fast Fourier Transform to time series.

        Args:
            sample (np.array): Signal to compute its FFT.
            fs (float): Sampling frequency.

        Returns:
            np.array: values of the frequency domain signal
            np.array: power of each frequency
            float: frequency step
        """

        n = len(sample)
        T = 1 / fs
        f_spect = np.linspace(0, 1 / (2 * T), int(n / 2))
        df = f_spect[1] - f_spect[0]
        yf = sc.fft.fft(sample)
        spectrum = 2 / n * np.abs(yf[: n // 2])

        return f_spect, spectrum, df

    def _find_nearest(self, sample, value):
        """This function finds the nearest value in an array.

        Args:
            sample (np.array): Any sample.
            value (float): Interest value.

        Returns:
            float: nearest value
        """

        sample = np.asarray(sample)
        idx = (np.abs(sample - value)).argmin()

        return sample[idx]

    def _mu(self, sample):
        """This function calculates mean for the sample.

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: mean
        """
        return np.mean(sample)

    def _std(self, sample):
        """This function calculates standard deviation for the sample.

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: standard deviation
        """
        return np.std(sample)

    def _median(self, sample):
        """This function calculates median for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: median
        """
        return np.median(sample)

    def _max(self, sample):
        """This function calculates maximum for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: maximum value
        """
        return np.max(sample)

    def _min(self, sample):
        """This function calculates minimum for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: minimum value
        """
        return np.min(sample)

    def _range(self, sample):
        """This function calculates range (max - min) for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: range value
        """
        return np.max(sample) - np.min(sample)

    def _mu_peak(self, sample):
        """This function calculates mean of local maximums for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: mean for the peaks value
        """
        peaks, _ = find_peaks(sample, height=0)
        return np.mean(peaks)

    def _rms(self, sample):
        """This function calculates RMS for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: rms value
        """
        return np.sqrt(np.sum(np.array(sample) ** 2) / len(sample))

    def _skewness(self, sample):
        """This function calculates skewness for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: skewness value
        """
        return skew(sample)

    def _kurt(self, sample):
        """This function calculates kurtosis for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: kurtosis value
        """
        return float(kurtosis(sample))

    def _k4(self, sample):
        """This function calculates k4 for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: k4 value
        """
        rms = self._rms(sample)
        kurt = self._kurt(sample)

        return rms * kurt

    def _energy(self, sample):
        """This function calculates energy for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: energy value
        """
        return np.sum((np.abs(sample)) ** 2)

    def _entropy(self, sample):
        """This function calculates entropy for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: entropy value
        """
        return -(np.sum(((sample) ** 2) * np.log((sample) ** 2)))

    def _peak_value(self, sample):
        """This function calculates peak value for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: peak value
        """
        rms = self._rms(sample)

        return rms * np.sqrt(2)

    def _crest_factor(self, sample):
        """This function calculates crest factor for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: crest factor value
        """
        rms = self._rms(sample)

        return (np.max(np.abs(sample))) / (rms)  # 20 * np.log()

    def _shape_factor(self, sample):
        """This function calculates shape factor for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: shape factor value
        """
        rms = self._rms(sample)

        return (rms) / (np.mean(np.abs(sample)))

    def _impulse_factor(self, sample):
        """This function calculates impulse factor for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: impulse factor value
        """
        rms = self._rms(sample)

        return (rms * np.sqrt(2)) / (np.mean(np.abs(sample)))

    def _margin_factor(self, sample):
        """This function calculates margin factor for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: margin factor value
        """
        rms = self._rms(sample)

        return (rms * np.sqrt(2)) / (np.mean(np.sqrt(np.abs(sample)))) ** 2

    def _estimate_rotation_speed(self, x):
        """This function estimates rotation speed for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: rotation speed
        """
        filter_rpm = []
        nyquist = 0.5 / self.dt
        band = np.array([5 / nyquist, 200 / nyquist])
        self.filter_rpm = signal.butter(
            4,
            band,
            btype="bandpass",
            analog=False,
            output="sos",
        )
        x_filter = signal.sosfilt(self.filter_rpm, x)
        m = x.shape[0]
        r = np.correlate(x_filter, x_filter, mode="full")
        r = r[m:0:-1]
        exp4 = 1.0 / np.exp(np.linspace(0, (m - 1) * self.dt, num=m) * 4)
        new_sample = np.zeros(16 * m)
        new_sample[0:m] = r * exp4
        X = np.abs(fft(new_sample))
        position = np.argmax(X[0 : 8 * m])
        rotation_frq = position / (16 * m * self.dt)

        return 60 * rotation_frq

    def _stationarity_criterium(self, sample):
        """This function calculates stationarity criterium for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: stationarity criterium value
        """
        z = np.copy(sample)
        z = np.correlate(z, z, "full")
        z = z[len(sample) - 1 : 2 * len(sample)]
        rms_full = np.sum(z * z)
        rms_end = np.sum(z[len(sample) - int(len(sample) / 4) : len(sample)] ** 2)

        return 100 * rms_end / (rms_full + 1e-12)

    def _max_value_spectrum(self, sample):
        """This function estimates amplitude variation of the 1x component for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: speed variation value
        """
        frq, spec, _ = self._my_fft(sample, (1 / self.dt))

        return frq[spec.argmax()]

    def _1per3x(self, sample, omega):
        """This function estimates 1/3X frequency component for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: 1/3X frequency component value
        """
        frq, spec, _ = self._my_fft(sample, (1 / self.dt))
        value_component = (1 / 3) * omega

        return spec[np.where(frq == self._find_nearest(frq, value_component))]

    def _0_42x(self, sample, omega):
        """This function estimates 0.42X frequency component for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: 0.42X frequency component value
        """
        frq, spec, _ = self._my_fft(sample, (1 / self.dt))
        value_component = 0.42 * omega

        return spec[np.where(frq == self._find_nearest(frq, value_component))]

    def _0_48x(self, sample, omega):
        """This function estimates 0.48X frequency component for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: 0.48X frequency component value
        """
        frq, spec, _ = self._my_fft(sample, (1 / self.dt))
        value_component = 0.48 * omega

        return spec[np.where(frq == self._find_nearest(frq, value_component))]

    def _1per2x(self, sample, omega):
        """This function estimates 1/2X frequency component for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: 1/2X frequency component value
        """
        frq, spec, _ = self._my_fft(sample, (1 / self.dt))
        value_component = 0.5 * omega

        return spec[np.where(frq == self._find_nearest(frq, value_component))]

    def _1x(self, sample, omega):
        """This function estimates 1X frequency component for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: 1X frequency component value
        """
        frq, spec, _ = self._my_fft(sample, (1 / self.dt))
        value_component = 1 * omega

        return spec[np.where(frq == self._find_nearest(frq, value_component))]

    def _3per2x(self, sample, omega):
        """This function estimates 3/2X frequency component for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: 3/2X frequency component value
        """

        frq, spec, _ = self._my_fft(sample, (1 / self.dt))
        value_component = (3 / 2) * omega

        return spec[np.where(frq == self._find_nearest(frq, value_component))]

    def _2x(self, sample, omega):
        """This function estimates 2X frequency component for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: 2X frequency component value
        """
        frq, spec, _ = self._my_fft(sample, (1 / self.dt))
        value_component = 2.0 * omega

        return spec[np.where(frq == self._find_nearest(frq, value_component))]

    def _5per2x(self, sample, omega):
        """This function estimates 5/2X frequency component for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: 5/2X frequency component value
        """
        frq, spec, _ = self._my_fft(sample, (1 / self.dt))
        value_component = (5 / 2) * omega

        return spec[np.where(frq == self._find_nearest(frq, value_component))]

    def _3x(self, sample, omega):
        """This function estimates 3X frequency component for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: 3X frequency component value
        """
        frq, spec, _ = self._my_fft(sample, (1 / self.dt))
        value_component = 3.0 * omega

        return spec[np.where(frq == self._find_nearest(frq, value_component))]

    def _7per2x(self, sample, omega):
        """This function estimates 7/2X frequency component for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: 7/2X frequency component value
        """
        frq, spec, _ = self._my_fft(sample, (1 / self.dt))
        value_component = (7 / 2) * omega

        return spec[np.where(frq == self._find_nearest(frq, value_component))]

    def _4x(self, sample, omega):
        """This function estimates 4X frequency component for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: 4X frequency component value
        """
        frq, spec, _ = self._my_fft(sample, (1 / self.dt))
        value_component = 4 * omega

        return spec[np.where(frq == self._find_nearest(frq, value_component))]

    def _center_freq(self, sample):
        """This function estimates center frequency of FFt for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: center frequency component value
        """
        frq, spec, _ = self._my_fft(sample, (1 / self.dt))

        return (np.sum(frq * spec)) / (np.sum(spec))

    def _energy_freq(self, sample):
        """This function estimates energy for FFt for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: energy value
        """
        frq, spec, _ = self._my_fft(sample, (1 / self.dt))

        return (np.sum(np.abs(spec) ** 2)) / len(spec)

    def _rms_freq(self, sample):
        """This function estimates RMS for FFt for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: rms value
        """
        frq, spec, _ = self._my_fft(sample, (1 / self.dt))

        return (np.sqrt(np.mean(np.abs(spec) ** 2))) / np.sqrt(np.sum(spec))

    def _std_freq(self, sample):
        """This function estimates standard deviation for FFt for the sample

        Args:
            sample (np.array): Vibration signal vector.

        Returns:
            float: standard deviation value
        """
        frq, spec, _ = self._my_fft(sample, (1 / self.dt))

        return np.sqrt(np.sum((frq**2) * spec)) / np.sqrt(len(spec))

    def estime_resources(self, sample):
        """This function estimates resources for each sample.

        Args:
            sample (np.array): Any sample (a vibration signal)

        Returns:
            np.array: resources
        """
        features = []

        for i in range(len(self.discriminators)):
            features.append(self.discriminators[i])

        feature_matrix = pd.DataFrame(
            columns=features, index=range(sample.shape[1]), dtype="int"
        )

        # for i in range(0, sample.shape[1]):

        # This way the for loop is deactivated
        for i in range(0, 0):
            sample_compare = sample[:, i]

            if "mean" in feature_matrix.columns:
                muv = self._mu(sample_compare)
                feature_matrix["mean"].loc[i] = muv

            if "std" in feature_matrix.columns:
                stdv = self._std(sample_compare)
                feature_matrix["std"].loc[i] = stdv

            if "median" in feature_matrix.columns:
                medianv = self._median(sample_compare)
                feature_matrix["median"].loc[i] = medianv

            if "max" in feature_matrix.columns:
                maxv = self._max(sample_compare)
                feature_matrix["max"].loc[i] = maxv

            if "min" in feature_matrix.columns:
                minv = self._min(sample_compare)
                feature_matrix["min"].loc[i] = minv

            if "range" in feature_matrix.columns:
                rangev = self._range(sample_compare)
                feature_matrix["range"].loc[i] = rangev

            if "mean peak" in feature_matrix.columns:
                mpv = self._mu_peak(sample_compare)
                feature_matrix["mean peak"].loc[i] = mpv

            if "skewness" in feature_matrix.columns:
                skv = self._skewness(sample_compare)
                feature_matrix["skewness"].loc[i] = skv

            if "kurtosis" in feature_matrix.columns:
                kuv = self._kurt(sample_compare)
                feature_matrix["kurtosis"].loc[i] = kuv

            if "k4" in feature_matrix.columns:
                k4 = self._k4(sample_compare)
                feature_matrix["k4"].loc[i] = k4

            if "rms" in feature_matrix.columns:
                rmsv = self._rms(sample_compare)
                feature_matrix["rms"].loc[i] = rmsv

            if "energy" in feature_matrix.columns:
                energyv = self._energy(sample_compare)
                feature_matrix["energy"].loc[i] = energyv

            if "entropy" in feature_matrix.columns:
                entropyv = self._entropy(sample_compare)
                feature_matrix["entropy"].loc[i] = entropyv

            if "peak value" in feature_matrix.columns:
                pv = self._peak_value(sample_compare)
                feature_matrix["peak value"].loc[i] = pv

            if "crest factor" in feature_matrix.columns:
                cf = self._crest_factor(sample_compare)
                feature_matrix["crest factor"].loc[i] = cf

            if "shape factor" in feature_matrix.columns:
                sf = self._shape_factor(sample_compare)
                feature_matrix["shape factor"].loc[i] = sf

            if "impulse factor" in feature_matrix.columns:
                ifv = self._impulse_factor(sample_compare)
                feature_matrix["impulse factor"].loc[i] = ifv

            if "margin factor" in feature_matrix.columns:
                clf = self._margin_factor(sample_compare)
                feature_matrix["margin factor"].loc[i] = clf

            if "max value spectrum" in feature_matrix.columns:
                value_maxpeaks = self._max_value_spectrum(sample_compare)
                feature_matrix["max value spectrum"].loc[i] = value_maxpeaks

            if "1per3x" in feature_matrix.columns:
                value_1per3x = self._1per3x(sample_compare, self.nominal_rotation)
                feature_matrix["1per3x"].loc[i] = value_1per3x

            if "042x" in feature_matrix.columns:
                value_042x = self._0_42x(sample_compare, self.nominal_rotation)
                feature_matrix["042x"].loc[i] = value_042x

            if "048x" in feature_matrix.columns:
                value_048x = self._0_48x(sample_compare, self.nominal_rotation)
                feature_matrix["048x"].loc[i] = value_048x

            if "1per2x" in feature_matrix.columns:
                value_1per2x = self._1per2x(sample_compare, self.nominal_rotation)
                feature_matrix["1per2x"].loc[i] = value_1per2x

            if "1x" in feature_matrix.columns:
                value_1x = self._1x(sample_compare, self.nominal_rotation)
                feature_matrix["1x"].loc[i] = value_1x

            if "3per2x" in feature_matrix.columns:
                value_3per2x = self._3per2x(sample_compare, self.nominal_rotation)
                feature_matrix["3per2x"].loc[i] = value_3per2x

            if "2x" in feature_matrix.columns:
                value_2x = self._2x(sample_compare, self.nominal_rotation)
                feature_matrix["2x"].loc[i] = value_2x

            if "5per2x" in feature_matrix.columns:
                value_5per2x = self._5per2x(sample_compare, self.nominal_rotation)
                feature_matrix["5per2x"].loc[i] = value_5per2x

            if "3x" in feature_matrix.columns:
                value_3x = self._3x(sample_compare, self.nominal_rotation)
                feature_matrix["3x"].loc[i] = value_3x

            if "7per2x" in feature_matrix.columns:
                value_7per2x = self._7per2x(sample_compare, self.nominal_rotation)
                feature_matrix["7per2x"].loc[i] = value_7per2x

            if "4x" in feature_matrix.columns:
                value_4x = self._4x(sample_compare, self.nominal_rotation)
                feature_matrix["4x"].loc[i] = value_4x

            if "5per2x" in feature_matrix.columns:
                value_5per2x = self._5per2x(sample_compare, self.nominal_rotation)
                feature_matrix["5per2x"].loc[i] = value_5per2x

            if "3x" in feature_matrix.columns:
                value_3x = self._3x(sample_compare, self.nominal_rotation)
                feature_matrix["3x"].loc[i] = value_3x

            if "7per2x" in feature_matrix.columns:
                value_7per2x = self._7per2x(sample_compare, self.nominal_rotation)
                feature_matrix["7per2x"].loc[i] = value_7per2x

            if "4x" in feature_matrix.columns:
                value_4x = self._4x(sample_compare, self.nominal_rotation)
                feature_matrix["4x"].loc[i] = value_4x

            if "center frequency" in feature_matrix.columns:
                cntf = self._center_freq(sample_compare)
                feature_matrix["center frequency"].loc[i] = cntf

            if "energy frequency" in feature_matrix.columns:
                energyf = self._energy_freq(sample_compare)
                feature_matrix["energy frequency"].loc[i] = energyf

            if "rms frequency" in feature_matrix.columns:
                rmsf = self._rms_freq(sample_compare)
                feature_matrix["rms frequency"].loc[i] = rmsf

            if "std frequency" in feature_matrix.columns:
                stdf = self._std_freq(sample_compare)
                feature_matrix["std frequency"].loc[i] = stdf

            if "rpm variation" in feature_matrix.columns:
                rpmvar = self._estimate_rotation_speed(sample_compare)
                feature_matrix["rpm variation"].loc[i] = rpmvar

            if "stationarity" in feature_matrix.columns:
                stat = self._stationarity_criterium(sample_compare)
                feature_matrix["stationarity"].loc[i] = stat

            if "rms in lowpass" in feature_matrix.columns:
                self.nb_filters = 1

                if self.nominal_rotation is None:
                    rpm_base = self._estimate_rotation_speed(sample_compare)

                else:
                    rpm_base = self.nominal_rotation

                rmslp = self._frequency_filter_features(
                    sample_compare, rpm_base=rpm_base
                )
                feature_matrix["rms in lowpass"].loc[i] = rmslp

            if "rms in bandpass" in feature_matrix.columns:
                self.nb_filters = 2

                if self.nominal_rotation is None:
                    rpm_base = self._estimate_rotation_speed(sample_compare)

                else:
                    rpm_base = self.nominal_rotation

                rmsbp = self._frequency_filter_features(
                    sample_compare, rpm_base=rpm_base
                )
                feature_matrix["rms in bandpass"].loc[i] = rmsbp

            if "rms in highpass" in feature_matrix.columns:
                self.nb_filters = 3

                if self.nominal_rotation is None:
                    rpm_base = self._estimate_rotation_speed(sample_compare)

                else:
                    rpm_base = self.nominal_rotation

                rmshp = self._frequency_filter_features(
                    sample_compare, rpm_base=rpm_base
                )
                feature_matrix["rms in highpass"].loc[i] = rmshp

            if "rms in 1rpm" in feature_matrix.columns:
                self.nb_filters = 4

                if self.nominal_rotation is None:
                    rpm_base = self._estimate_rotation_speed(sample_compare)

                else:
                    rpm_base = self.nominal_rotation

                rms1 = self._frequency_filter_features(
                    sample_compare, rpm_base=rpm_base
                )
                feature_matrix["rms in 1rpm"].loc[i] = rms1

            if "rms in 2rpm" in feature_matrix.columns:
                self.nb_filters = 5

                if self.nominal_rotation is None:
                    rpm_base = self._estimate_rotation_speed(sample_compare)

                else:
                    rpm_base = self.nominal_rotation

                rms2 = self._frequency_filter_features(
                    sample_compare, rpm_base=rpm_base
                )
                feature_matrix["rms in 2rpm"].loc[i] = rms2

        # """
        ### RMS
        if "rms" in feature_matrix.columns:
            zero_vector = np.zeros((self.samples.shape))
            RMS = rmse(
                self.samples, zero_vector, multioutput="raw_values", squared=False
            )
            feature_matrix["rms"] = RMS

        ### Peak value
        if "peak value" in feature_matrix.columns:
            peak_value = RMS * np.sqrt(2)
            feature_matrix["peak value"] = peak_value

        ### Kurtosis
        if "kurtosis" in feature_matrix.columns:
            kurt = kurtosis(self.samples)
            feature_matrix["kurtosis"] = kurt

        ### Skewness
        if "skewness" in feature_matrix.columns:
            skewness = skew(self.samples)
            feature_matrix["skewness"] = skewness

        # feature_matrix = np.concatenate((
        #     RMS.reshape(-1,1),
        #     peak_value.reshape(-1,1),
        #     kurt.reshape(-1,1),
        #     skewness.reshape(-1,1),
        #     ),axis=1)
        # """

        feature_matrix = np.array(feature_matrix)

        return feature_matrix

    def _frequency_filter_features(self, sample, rpm_base):
        """This function filters noise with different frequency filters.
        Uses Butterworth filters.

         Args:
             sample (np.array): Any sample to be filtered (a vibration signal).
             rpm_base (float): nominal rotation

         Returns:
             float: RMS for different frequency filters selected
        """
        nyquist = 0.5 / self.dt
        omega = rpm_base / (60 * nyquist)  # Adimentional rpm base

        # Low Pass filter

        if self.nb_filters == 1:
            sos = signal.butter(
                4, [1.35 * omega], btype="lowpass", analog=False, output="sos"
            )
            sign_filtered = signal.sosfilt(sos, sample)

        # Band Pass filter

        if self.nb_filters == 2:
            if nyquist < 3500:
                sos = signal.butter(
                    4, [900 / nyquist], btype="highpass", analog=False, output="sos"
                )
                sign_filtered = signal.sosfilt(sos, sample)

            elif nyquist > 3500:
                sos = signal.butter(
                    4,
                    [900 / nyquist, 3500 / nyquist],
                    btype="bandpass",
                    analog=False,
                    output="sos",
                )
                sign_filtered = signal.sosfilt(sos, sample)

        # High Pass filter

        if self.nb_filters == 3:
            if (3400 / nyquist) > 0 and (3400 / nyquist) < 1:
                sos = signal.butter(
                    4, [3400 / nyquist], btype="highpass", analog=False, output="sos"
                )
                sign_filtered = signal.sosfilt(sos, sample)

            else:
                print("High-pass frequency higher than the acquisition frequency")

        # Fundamental Harmonic Band 1*RPM

        if self.nb_filters == 4:
            sos = signal.butter(
                2,
                [0.65 * omega, 1.35 * omega],
                btype="bandpass",
                analog=False,
                output="sos",
            )
            sign_filtered = signal.sosfilt(sos, sample)

        # Second Harmonic Band 2*RPM

        if self.nb_filters == 5:
            sos = signal.butter(
                4,
                [1.30 * omega, 2.70 * omega],
                btype="bandpass",
                analog=False,
                output="sos",
            )
            sign_filtered = signal.sosfilt(sos, sample)

        return np.sqrt(np.sum(np.array(sign_filtered) ** 2) / len(sign_filtered))
