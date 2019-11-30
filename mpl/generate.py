# -*- coding: utf-8 -*-

""" Generate stack of plots

"""

import re
import numpy as np
import xarray as xr
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt

from ..utils import _cast_xarray
from ..utils import _cast_list
from ..utils import get_figure_class
from ..utils import get_figure_layout
from ..utils import get_layout_option
from ..utils import bbox_to_rect
from .mplfigure import FigureLine, FigureSpec, FigureAlt, FigureMap

try:
    import pytplot
except:
    pytplot = None


def generate_stack(var, layout=None, options=None):
    classdict = {
        'Line' : FigureLine,
        'Spec' : FigureSpec,
        'Alt'  : FigureAlt,
        'Map'  : FigureMap,
    }

    var = _cast_list(_cast_xarray(var))

    # get figure layout
    if layout is None:
        layout = dict()
    bbox_pixels, bbox_relative = get_figure_layout(var, **layout)

    # create figure
    dpi    = get_layout_option(layout, 'dpi')
    width  = get_layout_option(layout, 'width')
    height = get_layout_option(layout, 'height')
    figure = plt.figure(dpi=dpi, figsize=(width/dpi, height/dpi))
    axs    = list()

    # plot
    if options is None:
        options = {
            'dpi'      : dpi,
            'fontsize' : get_layout_option(layout, 'fontsize'),
        }

    num_plots = len(var)
    for j in range(num_plots):
        bbox = {
            'x0' : bbox_relative['x0'][j],
            'x1' : bbox_relative['x1'][j],
            'y0' : bbox_relative['y0'][j],
            'y1' : bbox_relative['y1'][j],
        }
        rect = bbox_to_rect(bbox)
        axs.append(plt.axes(rect))
        if isinstance(var[j], xr.DataArray):
            cls = get_figure_class(var[j], classdict)
            obj = cls(var[j], figure, axs[j], **options)
            obj.buildfigure()
        elif hasattr(var[j], '__iter__'):
            dat = var[j]
            for k in range(len(dat)):
                cls = get_figure_class(dat[k], classdict)
                obj = cls(dat[k], figure, axs[j], **options)
                obj.buildfigure()
    axs[0].get_shared_x_axes().join(*tuple(axs))

    return figure
