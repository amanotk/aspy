# -*- coding: utf-8 -*-

"""Utilities

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
    'fontsize'      : 12,
    'labelsize'     : 12,
    'ticklength'    : 6,
    'tickwidth'     : 1,
    'tickpad'       : 2,
    'colorbar_sep'  : 20,
    'colorbar_size' : 25,
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


def is_ipython():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except:
        return False


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


def get_figure_class(var, classdict):
    opt = var.attrs['plot_options'].get('extras')

    if opt.get('plotter', None) is not None:
        cls = opt.get('plotter')
    elif opt.get('spec', False):
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

    layout = default_layout.copy()
    for key in layout.keys():
        if key in kwargs:
            layout[key] = kwargs[key]

    # work in unit of pixels
    fig_h    = layout['height']
    fig_w    = layout['width']
    margin_t = layout['margin_top']
    margin_b = layout['margin_bottom']
    margin_l = layout['margin_left']
    margin_r = layout['margin_right']
    vspace   = layout['vspace']

    # var_label
    if 'var_label' in kwargs:
        print('Warning: var_label functionality has not yet been implemented')

    # get unit size for each panel in pixels
    N  = len(var)
    ps = [0] * N
    for i in range(N):
        if isinstance(var[i], xr.DataArray):
            ps[i] = get_plot_option(var[i], 'panelsize')
        elif hasattr(var[i], '__iter__'):
            ps[i] = get_plot_option(var[i][0], 'panelsize')
    ps = np.array(ps)
    ph = (fig_h - (margin_t + margin_b + vspace*(N-1))) / N
    pw = (fig_w - (margin_l + margin_r))
    hh = ph * ps
    ww = np.ones((N,)) * pw
    vv = np.ones((N,)) * vspace

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

    layout['bbox_pixels']   = bbox_pixels
    layout['bbox_relative'] = bbox_relative
    return layout


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

    opt = dict(y0=y0, y1=y1, bine=bine, binc=binc)

    return zz, opt


def time_slice(var, t1, t2):
    t1 = pd.Timestamp(t1).timestamp()
    t2 = pd.Timestamp(t2).timestamp()

    var = cast_list(cast_xarray(var))
    ret = [v.loc[t1:t2] for v in var]
    if len(ret) == 1:
        return ret[0]
    else:
        return ret


def pd_to_datetime(t):
    tt = np.atleast_1d(t)
    dt = tt.dtype
    if dt.type == np.str_ or dt.type == np.string_:
        tt = pd.to_datetime(tt)
    elif dt == np.float32 or dt == np.float64:
        tt = pd.to_datetime(tt, unit='s')
    else:
        raise ValueError('Unrecognized time format : ', dt)
    return tt


def to_scalar_or_array(t):
    if np.isscalar(t):
        return t
    elif t.size > 1:
        return t
    else:
        return t[0]


def to_unixtime(t):
    tt = pd_to_datetime(t).values.astype(np.int64) * 1.0e-9
    return to_scalar_or_array(tt)


def to_datetime64(t):
    tt = pd_to_datetime(t).values
    return to_scalar_or_array(tt)


def to_pydatetime(t):
    tt = pd_to_datetime(t).to_pydatetime()
    return to_scalar_or_array(tt)


def to_datestring(t, fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d %H:%M:%S'
    tt = pd_to_datetime(t).strftime(fmt)
    return to_scalar_or_array(tt)


def create_xarray(**data):
    # default attribute
    default_attrs = {
        'plot_options' : {
            'xaxis_opt' : {
                'axis_label' : 'Time',
                'x_axis_type' : 'linear',
            },
            'yaxis_opt' : {
                'axis_label' : 'Y',
                'y_axis_type' : 'linear',
            },
            'zaxis_opt' : {
                'axis_label' : 'Z',
                'z_axis_type' : 'linear',
            },
            'extras' : {
                'spec' : 0,
                'colormap' : ['viridis'],
                'panel_size' : 1,
                'char_size' : 10,
            },
        },
    }

    if 'x' in data and 'y' in data:
        x = np.array(data['x'])
        y = np.array(data['y'])
        # check compatibility
        if x.ndim == 1 and y.ndim == 1 and x.size == y.size:
            dims = ('time',)
        elif x.ndim == 1 and y.ndim == 2 and x.size == y.shape[0]:
            dims = ('time', 'vdim')
            v = np.arange(y.shape[1])
        else:
            raise ValueError('Error: incompatible input')
    else:
            raise ValueError('Error: incompatible input')

    # create DataArray object
    obj = xr.DataArray(y, dims=dims,
                       coords={'time' : ('time', x), 'v' : ('vdim', v)})
    obj.attrs = default_attrs.copy()

    return obj


def sph2xyz(r, t, p, degree=True):
    if degree:
        t = np.deg2rad(t)
        p = np.deg2rad(p)
    x = r * np.sin(t) * np.cos(p)
    y = r * np.sin(t) * np.sin(p)
    z = r * np.cos(t)
    return x, y, z


def xyz2sph(x, y, z, degree=True):
    r = np.sqrt(x**2 + y**2 + z**2)
    t = np.arccos(z / r)
    p = np.arctan2(y, x)
    if degree:
        t = np.rad2deg(t)
        p = np.rad2deg(p)
    return r, t, p
