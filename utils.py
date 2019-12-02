# -*- coding: utf-8 -*-

""" Utilities

"""

import warnings

import numpy as np
import pandas as pd
import xarray as xr

try:
    import pytplot
except:
    pytplot = None


default_layout = {
    'dpi'           : 300,
    'width'         : 800,
    'height'        : 800,
    'vspace'        : 25,
    'margin_top'    : 40,
    'margin_bottom' : 80,
    'margin_left'   : 100,
    'margin_right'  : 140,
    'linewidth'     : 1,
    'fontsize'      : 14,
    'labelsize'     : 14,
    'ticklength'    : 6,
    'tickwidth'     : 1,
    'tickpad'       : 2,
    'colorbar_sep'  : 50,
    'colorbar_size' : 100,
}

_option_table = {
    # x axis
    'xlabel'        : ('xaxis_opt', 'axis_label', ),
    'x_label'       : ('xaxis_opt', 'axis_label', ),
    'xtype'         : ('xaxis_opt', 'x_axis_type', ),
    'x_type'        : ('xaxis_opt', 'x_axis_type', ),
    'xrange'        : ('xaxis_opt', 'x_range', ),
    'x_range'       : ('xaxis_opt', 'x_range', ),
    # y axis
    'ylabel'        : ('yaxis_opt', 'axis_label', ),
    'y_label'       : ('yaxis_opt', 'axis_label', ),
    'ytype'         : ('yaxis_opt', 'y_axis_type', ),
    'y_type'        : ('yaxis_opt', 'y_axis_type', ),
    'yrange'        : ('yaxis_opt', 'y_range', ),
    'y_range'       : ('yaxis_opt', 'y_range', ),
    'legend'        : ('yaxis_opt', 'legend_names', ),
    # z axis
    'zlabel'        : ('zaxis_opt', 'axis_label', ),
    'z_label'       : ('zaxis_opt', 'axis_label', ),
    'ztype'         : ('zaxis_opt', 'z_axis_type', ),
    'z_type'        : ('zaxis_opt', 'z_axis_type', ),
    'zrange'        : ('zaxis_opt', 'z_range', ),
    'z_range'       : ('zaxis_opt', 'z_range', ),
    # other
    'trange'        : ('trange', ),
    't_range'       : ('trange', ),
    'fontsize'      : ('extras', 'char_size', ),
    'char_size'     : ('extras', 'char_size', ),
    'linecolor'     : ('extras', 'line_color', ),
    'line_color'    : ('extras', 'line_color', ),
    'colormap'      : ('extras', 'colormap', ),
    'panelsize'     : ('extras', 'panel_size',),
}


def cast_xarray(var):
    "cast input (scalar or sequence) into xarray's DataArray"
    if isinstance(var, str) and pytplot is not None:
        return pytplot.data_quants[var]
    elif isinstance(var, xr.DataArray):
        return var
    elif hasattr(var, '__iter__'):
        return list([cast_xarray(v) for v in var])
    else:
        raise ValueError('Unrecognized input')


def cast_list(var):
    if not isinstance(var, list):
        return list([var])
    else:
        return var


def process_kwargs(opt, kwargs, key, newkey=None):
    if newkey is None:
        newkey = key
    if key in kwargs:
        opt[key] = kwargs[key]


def set_plot_option(data, **kwargs):
    option_table = _option_table
    option_keys = option_table.keys()

    # check
    plot_options = data.attrs.get('plot_options', None)
    if plot_options is None:
        raise ValueError('Invalid input DataArray')

    # set options
    for key in kwargs.keys():
        if key in option_keys:
            try:
                table  = option_table[key]
                option = plot_options
                for i in range(len(table)-1):
                    option = option.get(table[i])
                option[table[-1]] = kwargs[key]
            except:
                raise warnings.warn('Error in setting option : %s' % (key))
        else:
            pass


def get_plot_option(data, key, val=None):
    option_table = _option_table
    option_keys = option_table.keys()

    # check
    plot_options = data.attrs.get('plot_options', None)
    if plot_options is None:
        raise ValueError('Invalid input DataArray')

    # set options
    if key in option_keys:
        try:
            table  = option_table[key]
            option = plot_options
            for i in range(len(table)-1):
                option = option.get(table[i])
            return option[table[-1]]
        except:
            pass

    return val


def get_layout_option(kwargs, key):
    return kwargs.get(key, default_layout[key])


def get_figure_class(var, classdict):
    opt = var.attrs['plot_options'].get('extras')

    if opt.get('spec', False):
        cls = classdict.get('Spec')
    elif opt.get('alt', False):
        cls = classdict.get('Alt')
    elif opt.get('map', False):
        cls = classdict.get('Map')
    else:
        cls = classdict.get('Line')

    return cls


def get_figure_layout(var, **kwargs):
    var = cast_list(cast_xarray(var))

    # work in unit of pixels
    fig_h = get_layout_option(kwargs, 'height')
    fig_w = get_layout_option(kwargs, 'width')

    # margin
    margin_t = get_layout_option(kwargs, 'margin_top')
    margin_b = get_layout_option(kwargs, 'margin_bottom')
    margin_l = get_layout_option(kwargs, 'margin_left')
    margin_r = get_layout_option(kwargs, 'margin_right')

    # var_label
    if 'var_label' in kwargs:
        print('Warning: var_label functionality has not yet been implemented')

    # get unit size for each panel in pixels
    N  = len(var)
    ps = np.array([get_plot_option(v, 'panelsize') for v in var])
    vs = get_layout_option(kwargs, 'vspace')
    ph = (fig_h - (margin_t + margin_b + vs*(N-1))) / N
    pw = (fig_w - (margin_l + margin_r))
    hh = ph * ps
    ww = pw * np.ones((N,))
    vv = vs * np.ones((N,))

    # bounding box in pixels
    x0 = np.zeros_like(hh)
    x1 = np.zeros_like(hh)
    y0 = np.zeros_like(hh)
    y1 = np.zeros_like(hh)
    x0[ : ] = margin_l
    x1[ : ] = x0[:] + ww
    y0[  0] = margin_b
    y0[1:N] = y0[0] + np.cumsum(hh + vv)[0:N-1]
    y1[ : ] = y0[:] + hh

    # reverse order
    x0 = x0[::-1]
    x1 = x1[::-1]
    y0 = y0[::-1]
    y1 = y1[::-1]

    bbox_pixels = {
        'x0' : x0,
        'x1' : x1,
        'y0' : y0,
        'y1' : y1,
    }

    bbox_relative = {
        'x0' : x0 / fig_w,
        'x1' : x1 / fig_w,
        'y0' : y0 / fig_h,
        'y1' : y1 / fig_h,
    }

    return bbox_pixels, bbox_relative


def bbox_to_rect(bbox):
    l = bbox['x0']
    b = bbox['y0']
    w = bbox['x1'] - bbox['x0']
    h = bbox['y1'] - bbox['y0']
    return l, b, w, h


def interpolate_spectrogram(ybin, data, **kwargs):
    from scipy import interpolate
    def interp(x, y, newx):
        f = interpolate.interp1d(x, y, axis=0, kind='nearest',
                                 bounds_error=False, fill_value=None)
        return f(newx)

    nx = ybin.shape[0]

    # check ybin
    if ybin.ndim == 1:
        y0 = ybin[ 0]
        y1 = ybin[-1]
        ny = ybin.shape[-1]
        yy = np.tile(ybin, (nx,1))
    elif ybin.ndim == 2:
        y0 = ybin[:, 0].min()
        y1 = ybin[:,-1].max()
        ny = ybin.shape[-1]
        yy = ybin
    else:
        raise ValueError('Invalid input')

    # interpolation points
    my = 2*ny
    if 'ylog' in kwargs and kwargs['ylog']:
        bine = np.logspace(np.log10(y0), np.log10(y1), my+1)
    else:
        print('error')
        bine = np.linspace(y0, y1, my+1)
    binc = 0.5*(bine[+1:] + bine[:-1])

    zz = np.zeros((nx, my), np.float64)
    for ii in range(nx):
        zz[ii,:] = interp(yy[ii,:], data[ii,:], binc)

    return y0, y1, zz


def time_clip(var, t1, t2):
    t1 = pd.Timestamp(t1).timestamp()
    t2 = pd.Timestamp(t2).timestamp()

    var = cast_list(cast_xarray(var))
    ret = [v.loc[t1:t2] for v in var]
    if len(ret) == 1:
        return ret[0]
    else:
        return ret
