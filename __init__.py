# -*- coding: utf-8 -*-

""" Tools for In-Situ Spacecraft Data Analysis

"""

import pandas as pd
import xarray as xr

from .utils import _cast_xarray
from .utils import _cast_list
from .utils import _process_kwargs
from .utils import set_plot_option
from .utils import time_clip
from .ply.generate import generate_stack as ply_generate_stack
from .mpl.generate import generate_stack as mpl_generate_stack
from .tplot2netcdf import save as ncsave
from .tplot2netcdf import load as ncload


def ply_tplot(var, **kwargs):
    figure_opt = dict(shared_xaxes=True)
    layout_opt = dict()

    # figure options
    _process_kwargs(figure_opt, kwargs, 'vertical_spacing')

    # layout options
    _process_kwargs(layout_opt, kwargs, 'width')
    _process_kwargs(layout_opt, kwargs, 'height')

    return ply_generate_stack(var, figure_opt, layout_opt, kwargs)


def mpl_tplot(var, **kwargs):
    return mpl_generate_stack(var)


#
tplot = ply_tplot
