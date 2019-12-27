# -*- coding: utf-8 -*-

"""Tools for In-Situ Spacecraft Data Analysis

"""

import numpy as np
import scipy as sp
from scipy import constants
np.seterr(divide = 'ignore')

import pandas as pd
import xarray as xr

from .utils import *
from .ply.generate import generate_stack as ply_generate_stack
from .mpl.generate import generate_stack as mpl_generate_stack
from .tplot2netcdf import save as ncsave
from .tplot2netcdf import load as ncload

from .attrdict import AttrDict

const = AttrDict( {
    'qme' : constants.e / constants.electron_mass,
    'qmp' : constants.e / constants.proton_mass,
    'me'  : constants.electron_mass,
    'mp'  : constants.proton_mass,
    'mu0' : constants.mu_0,
    'Re'  : 6378.1,
    })


def ply_tplot(var, **kwargs):
    return ply_generate_stack(var, **kwargs)


def mpl_tplot(var, **kwargs):
    return mpl_generate_stack(var, **kwargs)


def tplot(var, **kwargs):
    backend = kwargs.get('backend', 'mpl')

    if backend == 'mpl' or backend == 'matplotlib':
        # matplotlib
        return mpl_tplot(var, **kwargs)
    elif backend == 'ply' or backend == 'plotly':
        # plotly
        return ply_tplot(var, **kwargs)
    else:
        raise ValueError('Unknown backend: %s' % (backend,))
        return None
