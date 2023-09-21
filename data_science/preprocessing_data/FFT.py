import numpy as np
import scipy


def FFT(signal, Fs: int):
    """
    Fast Fourier Transform function.

    Parameters
    ----------
    signal : numpy array
        Signal to compute its FFT.
    Fs : int
        Sample frequency of the signal.

    Returns
    -------
    x_f : numpy array
        x values of the frequency domain signal.
    y_f : numpy array
        x values of the frequency domain signal (correspondent power of each frequency).

    Examples
    --------
    Get the values of FFT x-axis and y-axis

    >>> x_f, y_f = FFT(signal_sample, 1000)

    Plot the signal in the frequency domain

    >>> plt.figure()
    >>> plt.plot(x_f, y_f)

    """

    # Number of samplepoints
    N = len(signal)
    # sample spacing
    T = 1.0 / Fs
    # x_f = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))
    x_f = scipy.fft.fftfreq(N, T)[: N // 2]
    y_f = scipy.fft.fft(signal)
    y_f = 2.0 / N * np.abs(y_f[: N // 2])

    return x_f, y_f


def FFT_with_hanning(signal, Fs: int, df: float):
    """
    Fast Fourier Transform function with hanning window and adjustable Df.

    Parameters
    ----------
    signal : numpy array
        Signal to compute its FFT.
    Fs : int
        Sample frequency of the signal.
    df : float
        Frequency resolution of the FFT plot, the smaller the df, the higher the resolution.

    Returns
    -------
    x_f : numpy array
        x values of the frequency domain signal.
    y_f : numpy array
        x values of the frequency domain signal (correspondent power of each frequency).

    Examples
    --------
    Get the values of FFT x-axis and y-axis

    >>> x_f, y_f = FFT_with_hanning(signal_sample, 1000, 0.01)

    Plot the signal in the frequency domain

    >>> plt.figure()
    >>> plt.plot(x_f, y_f)
    """

    n = len(signal)
    h = np.hanning(n)

    y = np.multiply(signal, h)  # Product of two 1D arrays

    x_f = np.arange(0, Fs - df, df)
    N = len(x_f)

    dt = 1 / Fs
    t = [(index - 1) * dt for index in range(n)]
    t = np.array(t)

    # Compute frequency spectrum by Fourier Series
    y_f = [
        4
        * (
            np.sum(np.multiply(y, np.cos(2 * np.pi * x_f[index] * t)))
            + 1j * np.sum(np.multiply(y, np.sin(2 * np.pi * x_f[index] * t)))
        )
        / n
        for index in range(N)
    ]
    y_f = np.abs(np.array(y_f))

    return x_f, y_f
