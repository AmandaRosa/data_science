import numpy as np
from scipy import signal as sig
from scipy import fftpack

from .abstract import Methodology


class SimpleAnalysis(Methodology):
    """
    This analysis check the mean value, the peak to peak value and the RMS

    Paramethers:
        threshold_for_anomaly: float from 0 to 1
            The percentual variation value between 2 samples which will be needed in order for those samples to be considered anomalous
            ex: with a threshold_for_anomaly = 0.3 either the rms, peak to peak or rmp variation between two consecutive samples will
            have to be os at least 30% in order for those samples to be considered anomalous by the gate filter and thus be saved without
            being analysed by the other methodologies
        use_rms: bool
            wether or not to use rms paramether in the analysis
        use_peak_to_peak: bool
            wether or not to use peak to peak paramether in the analysis
        use_rpm: bool
            wether or not to use rpm paramether in the analysis
        encoder_V: float
            The value of the encoder peak when computing a full rotation
    """

    NAME = "gates"

    def __init__(
        self,
        threshold_for_anomaly=0.3,
        use_rms=True,
        use_peak_to_peak=True,
        use_rpm=True,
        encoder_V=-0.5,
        aquisition_frequency = None,
        nominal_rotation = None
    ):
        self.threshold = threshold_for_anomaly
        self.use_rms = use_rms
        self.use_peak_to_peak = use_peak_to_peak
        self.use_rpm = use_rpm
        self.num_signals = 0
        self.V = encoder_V
        self.rpm = 1
        self.nominal_rotation = nominal_rotation
        self.aquisition_frequency = aquisition_frequency

        self.compare_limit = self.threshold

    def set_sample(self, sample):
        self.send_sample(sample)

    def send_sample(self, sample_compare):
        """
        Starts comparison.


        Parameters
        ----------
        sample_compare : Array of Float64
            It is the last sample (signal acquired from the sensors).

        Returns
        -------
        float
            Value of variation in the reconstruction error among the last N samples.
            Being N the size of the buffer.

        """

        self.num_signals += 1

        self.n_channels = sample_compare.shape[1]

        if self.use_rpm:
            self.encoder = sample_compare[:, -1]
            sample_compare = sample_compare[:-1, :]

        self.current_sample = sample_compare

        self.calculate_parameters()

        self.calculate_variation()

        if self.are_samples_distinct():
            self.base_sample_parameters = self.current_sample_parameters.copy()

    def are_samples_distinct(self):
        """
        Compare calculated variation in the reconstruction error with the threshold.

        Returns
        -------
        bool
          -	True for distinct samples
          -	False for similar samples

        """

        return any((self.var_errors > self.threshold).flatten())

    def sample_rpm(self):
        """
        Get rpm from current sample

        Returns
        -------
        float
          - rpm

        """

        return self.rpm

    def compare_value(self, sensor=0, paramether=0):
        """
        Get variation in the reconstruction error's buffer

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        var_error : TYPE
            DESCRIPTION.

        """
        if paramether == "peak_to_peak":
            paramether = 0
        elif paramether == "rms":
            paramether = 1
        elif paramether == "rpm":
            paramether = 2

        return self.var_errors[sensor, paramether]

    def calculate_parameters(self):
        self.peak_to_peak_values = np.zeros(self.n_channels)
        self.rms_values = np.zeros(self.n_channels)
        self.rpm_values = np.zeros(self.n_channels)

        if self.use_peak_to_peak:
            for channel in range(self.n_channels):
                self.peak_to_peak_values[channel] = np.max(
                    self.current_sample[:, channel]
                ) - np.min(self.current_sample[:, channel])

        if self.use_rms:
            for channel in range(self.n_channels):
                self.rms_values[channel] = np.sqrt(
                    np.sum(self.current_sample[:, channel] ** 2)
                    / len(self.current_sample[:, channel])
                )

        if self.use_rpm:
            for channel in range(self.n_channels):
                self.x = self.current_sample[:, channel]
                self.rpm = self.calculate_RPM()
                self.rpm_values[channel] = self.rpm

        self.current_sample_parameters = np.concatenate(
            (
                self.peak_to_peak_values.reshape(-1, 1),
                self.rms_values.reshape(-1, 1),
                self.rpm_values.reshape(-1, 1),
            ),
            axis=1,
        )

    def calculate_variation(self):
        if self.num_signals == 1:
            self.base_sample_parameters = self.current_sample_parameters.copy()

        self.var_errors = abs(
            (self.current_sample_parameters - self.base_sample_parameters)
            / self.base_sample_parameters
        )

    def set_dt(self, dt):
        self.dt = dt
        self.fs = 1 / dt

    def calculate_RPM(self): 
        """
        Calculate the RPM from signal current_sample using nominal rotation (RPM) and aquisition frequency (Hz). 
        First, it is applied a lowpass filter on the fourth harmonic of the signal.
        Then, it calculates the autocorrelation vector of the filtered signal.
        It applies an exponential window 4 to ensure zeros at the end of r.
        Add zeros to r to increase frequency resolution.
        Calculates the FFT of the transient signal.
        Removes the influence the DC signal.
        Performs the product between the harmonics to mitigate the influence of undesirable components.

        Returns
        -------
        float
            Value of RPM from current_sample.

        """
        x = np.copy(self.x) 
        x = x-np.mean(x)
        m = len(x)

        dt = 1/self.aquisition_frequency
        nominal_frequency = self.nominal_rotation / 60
        pos_max = int(1.2*nominal_frequency*16*m*dt)
        fc = 2* 4*nominal_frequency*dt

        b_rot, a_rot = sig.butter(4, fc, btype="lowpass")
        x_filt = sig.lfilter( b_rot, a_rot, x)   

        r = np.correlate(x_filt, x_filt, mode="full")
        r = r[m:0:-1]

        exp4 = 1.0 / np.exp(np.linspace(0, (m - 1) * dt, num=m) * 4)
        r = r * exp4

        x = np.zeros(16 * m)
        x[0:m] = r

        X = np.abs(fftpack.fft(x))

        pos_max2 = int(pos_max/2)
        for j in range( 1,pos_max2 ) :
            if X[j] > X[j-1] : 
                break 
        X[0:j] = 0         

        for i in range( j,pos_max2 ) : 
            X[i] = X[2*i]+X[i]*X[2*i]*X[3*i]
        rotation_frequency = 2*np.argmax(X[0:pos_max2]) / (16 * m * dt)

        return rotation_frequency*60