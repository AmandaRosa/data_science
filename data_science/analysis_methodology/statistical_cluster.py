import numpy as np
import scipy.stats as st
from scipy import signal
from scipy.signal import hilbert
from sklearn.cluster import KMeans

from .abstract import Methodology


class StatisticalCluster(Methodology):

    NAME = "statistical_cluster"

    def __init__(self, significance=10, n_samples=250):
        super().__init__(significance)
        self.significance = significance
        self.n_samples = n_samples
        self.sample_base = None
        self._comparison_method = self._evaluate_signal

    def send_sample(self, sample_compare):
        """
        Start comparison.

        Parameters
        ----------
        sample_base: Array of Float64
            Sample base (signal aquired by sensor) is the first block of 4096 points
            that will be compared with the next sample.

            Multiple signals must be passed as columns.

        sample_compare: Array of Float64
            Is the second sample (signal aquired by sensor) with the same lenght
            (4096 point) of the "sample_base" that must be compared.

            Multiple signals must be passed as columns.

        dt: Float
            Time resolution = 1/Sample Frequency.

        Returns
        -------
        float
            Value of comparison between the samples
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
        if self.are_samples_distinct():
            self.sample_base = self.sample_compare

        return self.hypothesis_test

    def set_sample(self, sample_compare):
        self.sample_base = sample_compare

    def are_samples_distinct(self):
        """
        Compare calculated P-Value with the limit.

        Returns
        -------
        bool
          -	True for distinct samples
          -	False for similar samples

        """

        if self.hypothesis_test >= 100 - (
            self.significance / 2
        ) or self.hypothesis_test <= (
            self.significance / 2
        ):  # significancia/2 porque é bicaudal
            classi_final_cg = True  # recusa hipotese nula (media diferente 0)
        else:
            classi_final_cg = False  # aceita hipotese nula (media = 0)

        return classi_final_cg

    def compare_value(self):
        """
        Returns
        -------
        float
            Value of comparison between the samples

        """
        return self.hypothesis_test

    def _evaluate_signal(self):
        """
        It calculates the P-value between sample_base and sample_compare

        Returns
        -------
        hypothesis_test : Float
            the p-value as H1 or H0.

        """
        number_points = self.sample_size * (self.n_channels)

        x1 = self.sample_base[:, : self.n_channels]
        x1 = np.reshape(x1, number_points)

        x2 = np.reshape(self.sample_compare[:, : self.n_channels], number_points)
        hypothesis_test = self._compute_cg_cluster_pvalue(x1, x2)

        return hypothesis_test * 100

    def _filtro(self, x, freq_low, freq_hig, type):
        """
        A filter funciton

        Returns
        -------
        array
            Filtered signal

        """
        freq_nyquist = (1 / self.dt) / 2
        if freq_hig < 0.1:  # passa-baixas ou passa-altas
            Wn = np.array([freq_low / freq_nyquist])  # Filtro passa baixas
            sos = signal.butter(N=4, Wn=Wn, btype=type, analog=False, output="sos")
            filtrado = signal.sosfilt(sos, x)
            return filtrado
        Wn = np.array(
            [freq_low / freq_nyquist, freq_hig / freq_nyquist]
        )  # Filtro passa baixas
        sos = signal.butter(N=4, Wn=Wn, btype=type, analog=False, output="sos")
        filtrado = signal.sosfilt(sos, x)

        return filtrado

    def _avalia_features(self, x, x_filt):
        """
        Evaluate features to automatic input in cluster analysis.

        Returns
        -------
        integer
          Index: 0 or 1, where 0 = relevant and 1 = irrelevant

        """

        nfft = (1 / self.dt) / 4  # modificar depois so para teste
        x_aux = x
        x_aux_filt = x_filt

        f, Pxx_den = signal.welch(
            x_aux,
            1 / self.dt,
            window="hann",
            nperseg=nfft,
            noverlap=50,
            nfft=nfft,
            scaling="spectrum",
        )  # 2048 usando
        f, Pxx_den_filt = signal.welch(
            x_aux_filt,
            1 / self.dt,
            window="hann",
            nperseg=nfft,
            noverlap=50,
            nfft=nfft,
            scaling="spectrum",
        )  # 2048 usando

        Pxx_den = Pxx_den / np.max(Pxx_den)
        Pxx_den_filt = Pxx_den_filt / np.max(Pxx_den_filt)

        P = Pxx_den / len(Pxx_den)
        S = -sum(P * np.log(P))
        P = Pxx_den_filt / len(Pxx_den_filt)
        S_filt = -sum(P * np.log(P))

        S_base = 10 * np.log10(np.abs(S))
        S_filt = 10 * np.log10(np.abs(S_filt))
        S_final = S_base - S_filt

        if S_base >= -3:
            # print('sinal base só tem ruído')
            result_idx = 1
        else:
            if S_final > 0:
                result_idx = 0

            else:
                if np.abs(S_final) >= 3:
                    result_idx = 1

                else:
                    result_idx = 1

        return result_idx

    def _compute_cg_cluster_pvalue(self, x_base, x_comp):
        """
        It calculates the P-value between sample_base and sample_compare based on statistical cluster approach

        Returns
        -------
        hypothesis_test : Float
            the p-value as H1 or H0.

        """

        N = x_base.shape[0]

        amostra_base = np.zeros([N, self.n_samples])
        amostra_comp = np.zeros([N, self.n_samples])
        for i in range(self.n_samples):
            I = np.random.choice(N, size=N, replace=True)
            amostra_base[:, i] = x_base[I]
            amostra_comp[:, i] = x_comp[I]

        # criar as features

        feat_base = np.zeros([self.n_samples, 8])
        feat_comp = np.zeros([self.n_samples, 8])

        feat_base[:, 0] = np.max(amostra_base, axis=0) - np.min(amostra_base, axis=0)
        feat_base[:, 1] = np.sqrt(np.sum(amostra_base**2, axis=0) / N)
        feat_base[:, 2] = st.kurtosis(amostra_base, axis=0, fisher=True)
        feat_base[:, 3] = np.quantile(amostra_base, 0.75, axis=0) - np.quantile(
            amostra_base, 0.25, axis=0
        )

        freq_nyquist = (1 / self.dt) / 2

        if freq_nyquist <= 1000:
            f_max = freq_nyquist - 1
        else:
            f_max = 1000

        # espectro passa banda 10-100 Hz
        sig_filt1_base = self._filtro(x_base, 10, 100, "bandpass")
        # espectro passa banda 90-1000 Hz
        sig_filt2_base = self._filtro(x_base, 90, f_max, "bandpass")
        # espectro passa altas 900 Hz
        sig_filt3_base = self._filtro(x_base, f_max, 0, "highpass")
        # envelope passa banda 500 - fs/2 - 1
        sig_filt4_base = self._filtro(x_base, 300, ((1 / self.dt) / 2 - 1), "bandpass")

        for i in range(self.n_samples):
            I = np.random.choice(N, size=N, replace=True)
            x_aux = sig_filt1_base[I]
            feat_base[i, 4] = np.sqrt(np.sum(x_aux * x_aux) / N)
            x_aux = sig_filt2_base[I]
            feat_base[i, 5] = np.sqrt(np.sum(x_aux * x_aux) / N)
            x_aux = sig_filt3_base[I]
            feat_base[i, 6] = np.sqrt(np.sum(x_aux * x_aux) / N)
            x_aux = sig_filt4_base[I]
            analytical_signal = hilbert(x_aux)
            amplitude_envelope = np.abs(analytical_signal)
            amplitude_envelope[0:20] = 0
            amplitude_envelope = amplitude_envelope - np.mean(amplitude_envelope)
            feat_base[i, 7] = np.sqrt(
                np.sum(amplitude_envelope * amplitude_envelope)
                / len(amplitude_envelope)
            )

        feat_comp[:, 0] = np.max(amostra_comp, axis=0) - np.min(amostra_comp, axis=0)
        feat_comp[:, 1] = np.sqrt(np.sum(amostra_comp**2, axis=0) / N)
        feat_comp[:, 2] = st.kurtosis(amostra_comp, axis=0, fisher=True)
        feat_comp[:, 3] = np.quantile(amostra_comp, 0.75, axis=0) - np.quantile(
            amostra_comp, 0.25, axis=0
        )

        # espectro passa banda 10-100 Hz
        sig_filt1_comp = self._filtro(x_comp, 10, 100, "bandpass")
        # espectro passa banda 90-1000 Hz
        sig_filt2_comp = self._filtro(x_comp, 90, f_max, "bandpass")
        # espectro passa altas 900 Hz
        sig_filt3_comp = self._filtro(x_comp, f_max, 0, "highpass")
        # envelope passa banda 500 - fs/2 - 1
        sig_filt4_comp = self._filtro(x_comp, 300, ((1 / self.dt) / 2 - 1), "bandpass")

        for i in range(self.n_samples):
            I = np.random.choice(N, size=N, replace=True)
            x_aux = sig_filt1_comp[I]
            feat_comp[i, 4] = np.sqrt(np.sum(x_aux * x_aux) / N)
            x_aux = sig_filt2_comp[I]
            feat_comp[i, 5] = np.sqrt(np.sum(x_aux * x_aux) / N)
            x_aux = sig_filt3_comp[I]
            feat_comp[i, 6] = np.sqrt(np.sum(x_aux * x_aux) / N)
            x_aux = sig_filt4_comp[I]
            analytical_signal = hilbert(x_aux)
            amplitude_envelope = np.abs(analytical_signal)
            amplitude_envelope[0:20] = 0
            amplitude_envelope = amplitude_envelope - np.mean(amplitude_envelope)
            feat_comp[i, 7] = np.sqrt(
                np.sum(amplitude_envelope * amplitude_envelope)
                / len(amplitude_envelope)
            )

        # avalia as bandas de freq.

        indx = []
        indx_4 = self._avalia_features(x_comp, sig_filt1_comp)  # feat
        if indx_4 == 1:
            indx.append(4)
        indx_5 = self._avalia_features(x_comp, sig_filt2_comp)  # feat
        if indx_5 == 1:
            indx.append(5)
        indx_6 = self._avalia_features(x_comp, sig_filt3_comp)  # feat
        if indx_6 == 1:
            indx.append(6)
        indx_7 = self._avalia_features(x_comp, sig_filt4_comp)  # feat
        if indx_7 == 1:
            indx.append(7)

        feat_base = np.delete(feat_base, indx, axis=1)
        feat_comp = np.delete(feat_comp, indx, axis=1)

        # Cluster: K_means
        features = np.concatenate([feat_base, feat_comp], axis=0)
        I = np.random.choice(
            (self.n_samples * 2), size=(self.n_samples * 2), replace=None
        )
        features = features[I, :]
        kmeans = KMeans(n_clusters=2, random_state=0, init="k-means++").fit(features)
        cg = kmeans.cluster_centers_

        # Teste t - CG
        cg = np.array(cg)
        cg = np.abs(cg)
        cg_max = np.max(cg, axis=0)
        cg_max[np.abs(cg_max) < 1e-3] = 1
        cg = cg / cg_max
        x = cg[0, :] - cg[1, :]
        teste_t_statis, test_t_pvalue = st.ttest_1samp(x, 0)

        return test_t_pvalue
