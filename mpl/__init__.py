# -*- coding: utf-8 -*-

"""Plot Module via Matplotlib

"""

import re
import numpy as np
import xarray as xr
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt

from ..utils import *
from .mplfigure import \
    FigureLine, FigureSpec, FigureAlt, FigureMap, get_point_size


def _matplotlib_change_style(params):
    from matplotlib.cbook import _suppress_matplotlib_deprecation_warning
    with _suppress_matplotlib_deprecation_warning():
        current = matplotlib.rcParams.copy()
        params  = params.copy()
        if 'backend' in params:
            params.pop('backend')
        matplotlib.rcParams.update(params)
    return current


def generate_stack(var, **options):
    classdict = {
        'Line' : FigureLine,
        'Spec' : FigureSpec,
        'Alt'  : FigureAlt,
        'Map'  : FigureMap,
    }

    # use deafult matplotlib style
    style = _matplotlib_change_style(matplotlib.rcParamsDefault)

    var = cast_list(cast_xarray(var))

    # get figure layout
    layout = get_figure_layout(var, **options)
    options.update(layout)

    # create figure
    dpi    = options['dpi']
    width  = options['width']
    height = options['height']
    bbox   = options['bbox_relative']
    figure = plt.figure(dpi=dpi, figsize=(width/dpi, height/dpi))
    axs    = list()

    # create axes
    num_plots = len(var)
    for j in range(num_plots):
        bb = {
            'x0' : bbox['x0'][j],
            'x1' : bbox['x1'][j],
            'y0' : bbox['y0'][j],
            'y1' : bbox['y1'][j],
        }
        rect = bbox_to_rect(bb)
        axs.append(plt.axes(rect))
    axs[0].get_shared_x_axes().join(*tuple(axs))

    # plot
    for j in range(num_plots):
        if isinstance(var[j], xr.DataArray):
            cls = get_figure_class(var[j], classdict)
            obj = cls(var[j], figure, axs[j],
                      numaxes=j, numplot=0, **options)
            obj.buildfigure()
        elif hasattr(var[j], '__iter__'):
            # first plot
            k = 0
            dat = var[j]
            cls = get_figure_class(dat[k], classdict)
            obj = cls(dat[k], figure, axs[j],
                      numaxes=j, numplot=k, **options)
            obj.buildfigure()
            ymin, ymax = obj.get_yrange()
            # second, third...
            for k in range(1, len(dat)):
                # plot
                cls = get_figure_class(dat[k], classdict)
                obj = cls(dat[k], figure, axs[j],
                          numaxes=j, numplot=k, **options)
                obj.buildfigure()
                # set yrange
                yrange = obj.get_yrange()
                ymin = np.minimum(ymin, yrange[0])
                ymax = np.maximum(ymax, yrange[1])
                obj.set_yrange([ymin, ymax])

    # show ticks only for the bottom
    for j in range(num_plots-1):
        axs[j].xaxis.set_ticklabels([])

    if 'title' in options:
        fontsize = get_point_size(options['fontsize'], options['dpi'])
        title = {
            'label'    : options['title'],
            'pad'      : fontsize,
            'fontdict' : dict(fontsize=fontsize),
        }
        axs[0].set_title(**title)
    if 'trange' in options:
        axs[0].set_xlim(pd_to_datetime(options['trange']))

    # restore
    style = _matplotlib_change_style(style)

    plt.show()
    return figure
