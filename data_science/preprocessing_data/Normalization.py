import numpy as np


def add_amp_to_samples(samples):
    """
    Add the information of max amplitude in the end of each sample.

    Parameters
    ----------
    samples : numpy 2-D array.
        Each column of the 2-D array must be one sample.

    Returns
    -------
    samples_with_amp : numpy 2-D array
        Samples with one additional data point in the end of each column.

    Examples
    --------
    >>> samples_with_amp = add_amp_to_samples(samples)

    """

    num_samples, size_samples = np.shape(samples)
    vector_amp = np.multiply(np.ones((1, num_samples)), np.max(samples, axis=1)).T
    samples_with_amp = np.append(samples, vector_amp, axis=1)
    return samples_with_amp


def normalize_for_relu(samples):
    """
    Normalize values to positive only, to fit in the 'relu' function.

    Parameters
    ----------
    samples : numpy array
        Samples to be normalized.

    Returns
    -------
    samples_norm : numpy array
        Samples normalized to values politive only.

    Examples
    --------
    >>> samples_normalized = normalize_for_relu(samples)

    """

    min_value = np.min(samples)
    if min_value < 0:
        samples_norm = samples - min_value
    else:
        samples_norm = samples
    return samples_norm


def normalize_for_sigmoid(samples):
    """
    Normalize values to the range (0,1), to fit in the 'sigmoid' function.

    Parameters
    ----------
    samples : numpy array
        Samples to be normalized.

    Returns
    -------
    samples_norm : numpy array
        Samples normalized in the range (0,1).

    Examples
    --------
    >>> samples_normalized = normalize_for_sigmoid(samples)
    """

    samples_norm = (samples - np.min(samples)) / (np.max(samples) - np.min(samples))
    return samples_norm


def denormalize_from_sigmoid(samples_norm, max_value: float, min_value: float):
    """
    Denormalize array, to return the amplitudes to the original values, before normalization.

    Parameters
    ----------
    samples_norm : numpy array
        Samples to be denormalized.
    max_value : float
        The maximum value of the original samples, before normalization.
    min_value : float
        The minimum value of the original samples, before normalization.

    Returns
    -------
    samples : numpy array
        Samples denormalized.

    Examples
    --------
    >>> samples_denormalized_sigmoid = denormalize_from_sigmoid
        (
            samples_normalized_sigmoid,
            np.max(original_samples),
            np.min(original_samples)
        )

    """

    samples = (samples_norm * (max_value - min_value)) + min_value
    return samples


def denormalize_from_relu(samples_norm, min_value: float):
    """
    Denormalize array, to return the amplitudes to the original values, before normalization.

    Parameters
    ----------
    samples_norm : numpy array
        Samples to be denormalized.
    min_value : float
        The minimum value of the original samples, before normalization.

    Returns
    -------
    samples : numpy array
        Samples denormalized.

    Examples
    --------
    >>> samples_denormalized_relu = denormalize_from_relu
        (
            samples_normalized_relu,
            np.min(original_samples)
        )

    """

    diff = np.min(samples_norm) - min_value
    samples = samples_norm - diff
    return samples


if __name__ == "__main__":
    import scipy.io as sio
    from sklearn.metrics import mean_squared_error
    from slice_vector_ import *

    aux1 = sio.loadmat("Data/Desbalanceamento/data1.mat")
    xx18_1 = aux1["xx18"][0]

    # Slice vector into many samples
    samples = slice_vector_sequentially(xx18_1, 1024).T * 10**6

    samples_with_amp = add_amp_to_samples(samples)

    samples_normalized_sigmoid = normalize_for_sigmoid(samples)
    samples_normalized_relu = normalize_for_relu(samples)

    samples_denormalized_sigmoid = denormalize_from_sigmoid(
        samples_normalized_sigmoid, np.max(samples), np.min(samples)
    )

    samples_denormalized_relu = denormalize_from_relu(
        samples_normalized_relu, np.min(samples)
    )

    print(
        "MSE normalization and denormalization, with sigmoid function = "
        + str(mean_squared_error(samples, samples_denormalized_sigmoid))
    )
    print(
        "MSE normalization and denormalization, with relu function = "
        + str(mean_squared_error(samples, samples_denormalized_relu))
    )
