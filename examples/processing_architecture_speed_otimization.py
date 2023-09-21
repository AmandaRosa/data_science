#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 16:57:27 2021

@author: tuliotorezan
"""
import os
import sys
import time

sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np

from data_science import CompareData
from data_science.analysis_methodology import (
    Autoencoder,
    ClusterAnalysisDivergence,
    DeepAnT_CNN,
    HistogramConsistencyTest,
    ProcessingArchitecture,
)
from data_science.tools import CompressWavelet

# load data (download https://1drv.ms/u/s!AuAnQXzLmsvjihFVtI28bX0y0OyW?e=dDug9Y)
data = np.load("C:\\Users\\user\\Documents\\data_science\\examples\\data\\dataA.npy")[
    :, :4
]
# get dt
dt = 2.546e-4

data = data[140000:, :]

t0 = time.perf_counter()
# define methodologies to be used
histogram = HistogramConsistencyTest(
    nominal_rotation=60, filter_type="frequency", p_value_limit=0.05
)

auto_encoder = Autoencoder(
    size_sub_pack=512,
)

deep_ant = DeepAnT_CNN(epochs=2)

adaptative_cluster = ClusterAnalysisDivergence(
    n_channels=True,
    size_sub_sample=256,
    oversampling=True,
    step_to_overlaps=50,
    discriminators=["rms", "kurtosis", "peak value"],
    nb_cluster=6,
    threshold=None,
    sigma=2.5,
    size_buffer=8,
)

processing_architecture = ProcessingArchitecture(
    [
        auto_encoder,
        deep_ant,
        histogram,
        adaptative_cluster,
    ]
)

# # start comparison

compare_test = CompareData(data, dt, processing_architecture, slice_size=4000)

tf = time.perf_counter()
tt = tf - t0

target = np.load(
    "C:\\Users\\user\\Documents\\data_science\\examples\\data\\labels_tstA.npy"
)
target = target[140000:]
compare_test.plot(target=target)

processing_architecture.plot()


metrics = compare_test.get_evaluation_metrics(target)
metrics

# Compression
record_array = compare_test.record
abs_path = os.path.dirname(os.getcwd())
file_name = "REC_test_A"
save_path = f"C:\\Users\\user\\Documents\\data_science\\examples\\data\\"
cw = CompressWavelet(save_path, file_name, record_array)
start_time_wvl = time.time()

compare_test = CompareData(data, dt, cw, slice_size=4000)

spend_time_wvl = time.time() - start_time_wvl

# If you want to load
rec_data = cw.decode()
