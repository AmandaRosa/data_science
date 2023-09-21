import math

import numpy as np


def slice_vector_randomly(vector, slice_number: int, slice_size: int):
    """
    This function extracts randomly a certain number of samples from a given data population.

    Parameters
    ----------
    vector: numpy.ndarray
        A set of numerical data, specified as a vector.
    slice_number: int
        Number of samples that will be randomly extracted from the data, specified as a positive integer.
    slice_size: int
        Size of each sample, specified as a positive integer.

    Returns
    -------
    samples: numpy.ndarray
        samples extracted randomly, from a given data set.

    Examples
    --------
    >>> data=np.loadtxt('data.txt')
    >>> sample_number=50
    >>> sample_size=512
    >>> samples=slice_vector(data,sample_number,sample_size)

    """
    n = len(vector) - slice_size
    index = np.random.permutation(n)[:50]
    samples = np.zeros((slice_size, slice_number))
    for i in range(slice_number):
        samples[:, i] = vector[index[i] : index[i] + slice_size]
    return samples


def get_peak_positions(signal, amp_trig):
    """

    Parameters
    ----------
    signal : TYPE
        DESCRIPTION.
    amp_trig : TYPE
        DESCRIPTION.
    init : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    arround_peaks_position : TYPE
        DESCRIPTION.

    """

    arround_peaks_position = np.where(signal < amp_trig)[0]
    i = 1
    n = len(arround_peaks_position)
    ii = []

    while i < n:
        if arround_peaks_position[i] - arround_peaks_position[i - 1] < 5:
            ii.append(i - 1)
        i += 1
    peaks_position = np.delete(arround_peaks_position, ii)
    return peaks_position


def slice_vector(signal, n, init):
    signal = signal[init:]
    n_cols = n
    n_rows = int(len(signal) / n)
    signal = signal[: n_rows * n_cols]
    return np.reshape(signal, (n_rows, n_cols))


def slice_vector_sequentially(vector, slice_size: int):
    """
    This function slices a signal into many samples in sequence, with no overlap.

    Parameters
    ----------
    vector : numpy array
        Vector to be sliced.
    slice_size : int
        Size of each slice, or sample.

    Returns
    -------
    slices : numpy array
        All slices.

    Examples
    --------
    >>> samples = slice_vector_sequentially(data, 6720)

    """

    num_slices = math.floor((len(vector) / slice_size))
    last_index = slice_size * num_slices
    index_slice = np.arange(0, last_index, slice_size)

    slices = [vector[index : index + slice_size] for index in index_slice]
    slices = np.array(slices)

    return slices


def slice_vector_manually(vector, slice_size: int, steps: int):
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

    return slices


def find_encoder_pulses(vector):
    """
    This function finds the pulses of the encoder signal and returns the
    digital conversion of its signal.

    Parameters
    ----------
    vector : numpy array
        Encoder signal.

    Returns
    -------
    digital_encoder : numpy array
        Digital signal of the encoder. The pulses are converted to ones and the rest is set to zero.

    Examples
    --------
    >>> digital_encoder = find_encoder_pulses(encoder)

    """

    # Create a digital encoder
    digital_encoder = np.zeros([len(vector)])

    # Get region of pulses
    indexes_pulse = np.where(vector < -1)[0]

    # Get border of each pulses
    diff_pulses = np.diff(indexes_pulse)
    indexes_end_pulse = np.where(diff_pulses > 1)[0]

    # Create a vector with the index of the borders of each pulse
    indexes_border_pulses = np.zeros([2, len(indexes_end_pulse)])
    indexes_border_pulses[0, 1:] = indexes_end_pulse[:-1]
    indexes_border_pulses[1, :] = indexes_end_pulse[:]
    indexes_border_pulses = indexes_border_pulses.astype(int)
    indexes_border_pulses = indexes_border_pulses.transpose()

    # Get the indexes of the min value of each pulse
    index_peaks = [
        indexes_pulse[np.argmin(vector[indexes_pulse[index[0] : index[1]]]) + index[0]]
        for index in indexes_border_pulses
    ]

    # Attribute the min values to the digital encoder
    digital_encoder[index_peaks] = 1

    return digital_encoder


def slice_vector_peaks(vector, digital_encoder, num_turns=1):
    """
    This function slices a signal into many samples with a specific number of turns, according to the encoder signal.

    Parameters
    ----------
    vector : numpy array
        Vector to be sliced.
    digital_encoder : numpy array
        Digital encoder signal.
    num_turns : int
        Number of complete turns of each sample.

    Returns
    -------
    samples_multi_turns : numpy array
        All samples.


    Examples
    --------
    >>> samples = slice_vector_peaks(data, digital_encoder, 12)

    """

    # Find index of the encoder pulses
    index_pulses = np.where(digital_encoder == 1)
    index_pulses = index_pulses[0]

    # Number of points of each turn
    pitch = np.min(np.diff(index_pulses))  # Number of points of each turn

    samples_one_turn = [vector[index : index + pitch] for index in index_pulses]
    samples_one_turn = np.array(samples_one_turn)

    indexes_turns = np.arange(0, len(samples_one_turn), num_turns)[:-1]

    # Gather many turns in one sample
    samples_multi_turns = [
        np.array(samples_one_turn[index : index + num_turns, :].flatten())
        for index in indexes_turns
    ]
    samples_multi_turns = np.array(samples_multi_turns)

    return samples_multi_turns


def get_mean_signal_of_all_samples(samples):
    """


    Parameters
    ----------
    samples : numpy array
        All samples with the same size.

    Returns
    -------
    mean_signal : numpy array
        Mean signal of all samples.

    Examples
    --------
    >>> mean_signal = get_mean_signal_of_all_samples(samples)

    """
    samples = samples.T
    mean_signal = [np.mean(position) for position in samples]
    mean_signal = np.array(mean_signal)

    return mean_signal
