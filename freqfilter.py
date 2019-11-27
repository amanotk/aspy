# -*- coding: utf-8 -*-

""" Frequency filtering tools

"""

import copy

import numpy as np
import scipy as sp
from scipy import signal

import pandas as pd
import xarray as xr


def apply_filter(b, a, data):
    "apply filter frequency filter for given data"
    if isinstance(data, np.ndarray):
        return signal.filtfilt(b, a, data, axis=0)
    elif isinstance(data, xr.DataArray) or isinstance(data, pd.DataFrame):
        x = copy.deepcopy(data)
        x.values = signal.filtfilt(b, a, x.values, axis=0)
        return x
    else:
        raise ValueError('datatype %s is not supported' % (type(data)))


def bandpass(x, flc, fhc, fs, order=5):
    """apply bandpass filter of Butterworth

    Parameters
    ----------
    x : array-like
        input data
    flc :
        lower cutoff frequency
    fhc : float
        higher cutoff freuquency
    fs : float
        sampling frequency
    order : int
        order of Butterworth filter (default 5)

    Returns
    -------
    filtered data
    """
    nyq = 0.5 * fs
    b, a = signal.butter(order, [flc/nyq, fhc/nyq], btype='band')
    return apply_filter(b, a, x)


def highpass(x, fc, fs, order=5):
    """apply highpass filter of Butterworth

    Parameters
    ----------
    x : array-like
        input data
    fc : float
        cutoff freuquency
    fs : float
        sampling frequency
    order : int
        order of Butterworth filter (default 5)

    Returns
    -------
    filtered data
    """
    nyq = 0.5 * fs
    b, a = signal.butter(order, fc/nyq, btype='high')
    return apply_filter(b, a, x)


def lowpass(x, fc, fs, order=5):
    """apply lowpass filter of Butterworth

    Parameters
    ----------
    x  : array-like
        input data
    fc : float
        cutoff freuquency
    fs : float
        sampling frequency
    order : int
        order of Butterworth filter (default 5)

    Returns
    -------
    filtered data
    """
    nyq = 0.5 * fs
    b, a = signal.butter(order, fc/nyq, btype='low')
    return apply_filter(b, a, x)
