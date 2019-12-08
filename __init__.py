# -*- coding: utf-8 -*-

""" Tools for In-Situ Spacecraft Data Analysis

"""

import numpy as np
np.seterr(divide = 'ignore')

import pandas as pd
import xarray as xr

from .utils import cast_xarray
from .utils import cast_list
from .utils import process_kwargs
from .utils import set_plot_option
from .utils import time_clip
from .ply.generate import generate_stack as ply_generate_stack
from .mpl.generate import generate_stack as mpl_generate_stack
from .tplot2netcdf import save as ncsave
from .tplot2netcdf import load as ncload


def ply_tplot(var, **kwargs):
    return ply_generate_stack(var)


def mpl_tplot(var, **kwargs):
    return mpl_generate_stack(var)


#
tplot = ply_tplot
