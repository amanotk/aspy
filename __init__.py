# -*- coding: utf-8 -*-

""" Tools for In-Situ Spacecraft Data Analysis

 $Id$
"""

import pandas as pd
import xarray as xr

from .utils import _cast_xarray
from .utils import _process_kwargs
from .utils import time_clip
from .generate import generate_stack
from .tplot2netcdf import save as ncsave
from .tplot2netcdf import load as ncload


def tplot(var, **kwargs):
    figure_opt = dict(shared_xaxes=True)
    layout_opt = dict()

    # figure options
    _process_kwargs(figure_opt, kwargs, 'vertical_spacing')

    # layout options
    _process_kwargs(layout_opt, kwargs, 'width')
    _process_kwargs(layout_opt, kwargs, 'height')

    return generate_stack(var, figure_opt, layout_opt)

