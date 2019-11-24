# -*- coding: utf-8 -*-

""" Tools for In-Situ Spacecraft Data Analysis

 $Id$
"""

import xarray as xr

try:
    import pytplot
except:
    pytplot = None

from .generate import generate_stack
from .tplot2netcdf import save as ncsave
from .tplot2netcdf import load as ncload


def _process_kwargs(opt, kwargs, key, newkey=None):
    if newkey is None:
        newkey = key
    if key in kwargs:
        opt[key] = kwargs[key]


def tplot(var, **kwargs):
    figure_opt = dict(shared_xaxes=True)
    layout_opt = dict()

    # figure options
    _process_kwargs(figure_opt, kwargs, 'vertical_spacing')

    # layout options
    _process_kwargs(layout_opt, kwargs, 'width')
    _process_kwargs(layout_opt, kwargs, 'height')

    return generate_stack(var, figure_opt, layout_opt)
