import numpy as np
import scipy.stats as st
from pywt import dwt
from scipy import fftpack, signal

from .abstract import Methodology


class HistogramConsistencyTest(Methodology):
    """
    This class uses "Histogram Consistency Test" statistical method. It compares the original signal captured by sensors, signal will aquired as "blocks", that will be compared with each other. The P-value of the Histogram consistency will determinate the percentage of the following hypotesis test:

        - Whether the means of two independent samples are significantly different.

    The following assumptions are made:

        - Observations in each sample are independent and identically distributed (iid);
        - Observations in each sample are normally distributed;
        - Observations in each sample have the same variance.

    The results given can be interpretated as follows:

        - H0 : the means of the samples are equal;
        - H1 : the means of the samples are unequal.

    Whether the U and V histograms of two signals with the same number of k classes, and same limits. If the number of occurrences in each class is very large, it can be assumed that the distribution of values within each class is normal. Therefore, if the two histograms are equal, we have under H0:

    .. math::

        u_{i} = v_{i} = u_{(i)}, i = 1, \\dotsc, k

    .. math::

        T = \\sum_{i=0}^{k} \\frac{{\\delta_{i}}^{2}}{\\sigma^{2}}

    has distribution :math:`{X_{k}}^{2}`. With :math:`\\delta = u_{(i)} - v_{i}`. The variance is estimated by:

    .. math::

        \\sigma^{2} \\approx | u_{(i)} - v_{i} |

    Parameters
    ----------
    nominal_rotation : float
        It's the machine nominal rotation that the sesnsor are conected to.
    n_bins : int, optional
        This is for the Histogram configuiration, the number of bins that will be used in the Histogram constancy test calculation. Default value is 40.
    p_value_limit : float, optional
        Determined by user, is the limit value (percentage) for record. The default is 0.10.
    rotational_speed_estimation : float, optional
        This option is for defining possible variation of rotation during the signal. The default is "1" rpm.
    filter_type : str, optional
        Type of filter to be applied before compare. Default is None.
        The options are:

            - Wavelet: As with Fourier analysis there are three basic steps to filtering signals using wavelets:
                - Decompose the signal using the DWT.
                - Filter the signal in the wavelet space using thresholding.
                - Invert the filtered signal to reconstruct the original, now filtered signal, using the inverse DWT.

                Briefly, the filtering of signals using wavelets is based on the idea that as the DWT decomposes the signal into details and approximation parts, at some scale the details contain mostly the insignificant noise and can be removed or zeroed out using thresholding without affecting the signal.

            - Frequency: This option will pass the signal over the following filters:
                - Nyquist filter.
                - 0.5 rotation filter.
                - First harmonic filter (1X).
                - Second harmonic filter (2X).
                - Third harmonic filter (3X).
                - Fourth harmonic filter (4X).
                - High pass filter.

            - None: It will pass the signal without any kind of filter and run the Hipothesis Test.

    Examples
    --------
    >>> import numpy as np
    >>> from data_science.statistic import HistogramConsistencyTest

    Define sample arrays:

    >>> dt = 0.001
    >>> time = np.arange(0, (dt)*4000, dt)
    >>> sample_1_X = np.cos(2 * time)
    >>> sample_1_Y = np.sin(2 * time)
    >>> sample_1 = np.array([sample_1_X, sample_1_Y]).T
    >>> sample_2_X = np.cos( 2.5* time)
    >>> sample_2_Y = np.sin( 2.5* time)
    >>> sample_2 = np.array([sample_2_X, sample_2_Y]).T

    Run analysis:

    >>> hct = HistogramConsistencyTest(
    ... n_bins=40,
    ... p_value_limit=0.10,
    ... rotational_speed_estimation=0,
    ... nominal_rotation=60,
    ... filter_type=None)
    >>> hct.compare(
    ... sample_base=sample_1,
    ... sample_compare=sample_2,
    ... dt=dt,)

    >>> hct.compare_value()
    0.06409476804918346
    >>> hct.are_samples_distinct()
    True
    >>> hct.plot()
    """

    NAME = "histogram"

    def __init__(
        self,
        nominal_rotation,
        n_bins=40,
        p_value_limit=0.10,
        rotational_speed_estimation=0,
        filter_type=None,
    ):
        super().__init__(p_value_limit)
        self.nominal_rotation = nominal_rotation
        self.n_bins = n_bins
        self.p_value_limit = p_value_limit
        self.rotational_speed_estimation = rotational_speed_estimation / 60
        self.filter_type = filter_type
        self.sample_base = None

        ## Define Comparison Method
        # wavelets
        if self.filter_type == "wavelets":
            self._comparison_method = self._evaluate_wavelet_signal

        # frequency
        elif self.filter_type == "frequency":
            self._comparison_method = self._evaluate_frequecy_signal

        # linear
        elif self.filter_type is None:
            self._comparison_method = self._evaluate_linear_signal

        else:
            raise Exception("Invalid filter_type")

    def send_sample(self, sample_compare):
        """
        Start comparison.

        Parameters
        ----------
        sample_base : numpy.ndarray
            Sample base (signal aquired by sensor) is the first block of 4096 points that will be compared with the next sample. Multiple signals must be passed as columns.
        sample_compare : numpy.ndarray
            Is the second sample (signal aquired by sensor) with the same lenght (4096 point) of the "sample_base" that must be compared. Multiple signals must be passed as columns.
        dt : float
            Time resolution = 1/Sample Frequency.

        Returns
        -------
        hypothesis_test : float
            Value of comparison between the samples.
        """

        # Inicialization

        if self.sample_base is None:
            self.sample_base = sample_compare
            self.hypothesis_test = 0
            return self.hypothesis_test

        if self.sample_base.shape != sample_compare.shape:
            raise Exception(
                f"Samples must have the same shape. Sample Base: {self.sample_base.shape} != Sample Compare: {sample_compare.shape}"
            )

        self.sample_compare = sample_compare

        self.sample_size, self.n_channels = self.sample_base.shape

        # Compare
        self.hypothesis_test = self._comparison_method()
        if self.rotational_speed_estimation > 0:
            hypothesis_test_rotac = self._evaluate_rpm_variation()
            self.hypothesis_test = self.hypothesis_test * hypothesis_test_rotac

        if self.are_samples_distinct():
            self.sample_base = self.sample_compare

        return self.hypothesis_test

    def set_sample(self, sample_compare):
        self.sample_base = sample_compare

    def _evaluate_linear_signal(self):
        """
        It calculates the P-value between sample_base and sample_compare

        Returns
        -------
        hypothesis_test : float
            The p-value as H1 or H0.

        """
        number_points = self.sample_size * (self.n_channels)

        x1 = self.sample_base[:, : self.n_channels]
        x1 = np.reshape(x1, number_points)

        x2 = np.reshape(self.sample_compare[:, : self.n_channels], number_points)
        hypothesis_test = self._compute_histogram_pvalue(x1, x2)

        return hypothesis_test

    def _evaluate_wavelet_signal(self):
        """
        It calculates the P-value between sample_base and sample_compare

        Returns
        -------
        hypothesis_test : float
            the p-value as H1 or H0.

        """

        x1 = self.sample_base[:, : self.n_channels]
        cA, cD = dwt(x1, "dB4", axis=0)
        cA1, cD1 = dwt(cA, "dB4", axis=0)
        cA2, cD2 = dwt(cA1, "dB4", axis=0)
        cD_rows, j = cD.shape
        cD1_rows, j = cD1.shape
        cD2_rows, j = cD2.shape
        cA2_rows, j = cA2.shape
        x1_1 = np.reshape(cD, cD_rows * self.n_channels)
        x1_2 = np.reshape(cD1, cD1_rows * self.n_channels)
        x1_3 = np.reshape(cD2, cD2_rows * self.n_channels)
        x1_4 = np.reshape(cA2, cA2_rows * self.n_channels)

        x2 = self.sample_compare[:, : self.n_channels]
        p_value = 1
        cA, cD = dwt(x2, "dB4", axis=0)
        x = np.reshape(cD, cD_rows * self.n_channels)
        p_value = p_value * self._compute_histogram_pvalue(x1_1, x)
        x1_1 = x

        cA1, cD1 = dwt(cA, "dB4", axis=0)
        x = np.reshape(cD1, cD1_rows * self.n_channels)
        p_value = p_value * self._compute_histogram_pvalue(x1_2, x)
        x1_2 = x

        cA2, cD2 = dwt(cA1, "dB4", axis=0)
        x = np.reshape(cD2, cD2_rows * self.n_channels)
        p_value = p_value * self._compute_histogram_pvalue(x1_3, x)
        x1_3 = x

        x = np.reshape(cA2, cA2_rows * self.n_channels)
        p_value = p_value * self._compute_histogram_pvalue(x1_4, x)
        x1_4 = x
        hypothesis_test = p_value**0.25

        return hypothesis_test

    def _evaluate_frequecy_signal(self):
        """
        It calculates the P-value between sample_base and sample_compare

        Returns
        -------
        hypothesis_test : float
            The p-value as H1 or H0.

        """
        pvalue = []
        # Low Pass Filter
        f_cut = 2 * self.nominal_rotation * self.dt * 0.5
        b, a = signal.butter(1, f_cut, btype="lowpass")
        pvalue.append(self._evaluate_filter_signal(a, b))

        # Rotation Harmonic
        band = 2 * self.nominal_rotation * self.dt * np.array([0.65, 1.35])
        b, a = signal.butter(2, band, btype="bandpass")
        pvalue.append(self._evaluate_filter_signal(a, b))

        # Second Harmonic
        band = 2 * 2 * self.nominal_rotation * self.dt * np.array([0.65, 1.35])
        b, a = signal.butter(2, band, btype="bandpass")
        pvalue.append(self._evaluate_filter_signal(a, b))

        # Third Harmonic
        band = 3 * 2 * self.nominal_rotation * self.dt * np.array([0.65, 1.35])
        b, a = signal.butter(2, band, btype="bandpass")
        pvalue.append(self._evaluate_filter_signal(a, b))

        # High Pass Filter
        b, a = signal.butter(2, band[1], btype="highpass")
        pvalue.append(self._evaluate_filter_signal(a, b))

        pvalue = np.array(pvalue)
        hypothesis_test = np.prod(pvalue, axis=0) ** 0.2

        return hypothesis_test

    def _evaluate_filter_signal(
        self,
        a_filter,
        b_filter,
    ):
        """
        It calculates the P-value between filtered sample_base and filtered
        sample_compare.

        Parameters
        ----------
        a_filter : numpy.ndarray
            The denominator coefficient vector in a 1-D sequence. If a[0] is not 1, then both a and b are normalized by a[0].
        b_filter : numpy.ndarray
            The numerator coefficient vector in a 1-D sequence.

        Returns
        -------
        hypothesis_test : float
            The p-value as H1 or H0.
        """
        # Inicialization
        number_points = self.sample_size * (self.n_channels)

        x1 = self.sample_base[:, : self.n_channels]
        x1 = signal.lfilter(b_filter, a_filter, x1)
        x1 = np.reshape(x1, number_points)

        x2 = self.sample_compare[:, : self.n_channels]
        x2 = signal.lfilter(b_filter, a_filter, x2)
        x2 = np.reshape(x2, number_points)
        hypothesis_test = self._compute_histogram_pvalue(x1, x2)

        return hypothesis_test

    def _compute_histogram_pvalue(self, x1, x2):
        """
        Computes the histogram pvalue.

        Parameters
        ----------
        x1 : numpy.ndarray
            Sample_base.
        x2 : numpy.ndarray
            Sample_compare.

        Returns
        -------
        hypothesis_test : Float
            It calculates the P-value between sample_base and sample_compare it will return the p-value as H1 or H0.
        """

        bins = np.linspace(x1.min(), x1.max(), num=self.n_bins + 1)

        h1 = self._hist_user(x1, bins)
        h2 = self._hist_user(x2, bins)
        T = np.sum(((h2 - h1) ** 2) / (h1 + h2 + 1e-12))
        p_value = 1 - st.chi2.cdf(T, self.n_bins)
        return p_value

    def _hist_user(self, x, bins):
        nbins = bins.size - 1
        hist = np.zeros(nbins)
        for i in range(0, nbins - 1):
            hist[i] = x[x < bins[i + 1]].size
        hist[i + 1] = x[x > bins[i + 1]].size
        hist[1 : nbins - 1] = hist[1 : nbins - 1] - hist[0 : nbins - 2]
        return hist

    def _evaluate_rpm_variation(self):
        # Inicialization
        sensability_frequency = self.rotational_speed_estimation

        x = self.sample_base[:, 0]
        actual_frequency = self._speed_calculation(x)

        # second data
        freq = self._speed_calculation(self.sample_compare[:, 0])
        if abs(actual_frequency - freq) > sensability_frequency:
            actual_frequency = freq
            hypothesis_test = 0
        else:
            hypothesis_test = 1

        return hypothesis_test

    def _speed_calculation(self, x):
        m = x.shape[0]
        r = np.correlate(x, x, mode="full")
        r = r[m:0:-1]
        exp4 = 1.0 / np.exp(np.linspace(0, (m - 1) * self.dt, num=m) * 4)
        r = r * exp4
        x = np.zeros(16 * m)
        x[0:m] = r
        X = np.abs(fftpack.fft(x))
        position = np.argmax(X)
        rotation_frequency = position / (16 * m * self.dt)
        return rotation_frequency

    def are_samples_distinct(self):
        """
        Compare calculated P-Value with the limit.

        Returns
        -------
        bool
            - True for distinct samples
            - False for similar samples

        """
        return self.hypothesis_test <= self.p_value_limit

    def compare_value(self):
        """
        Get compare value.

        Returns
        -------
        hypothesis_test : float
            Value of comparison between the samples

        """
        return self.hypothesis_test
