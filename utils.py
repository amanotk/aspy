# -*- coding: utf-8 -*-

""" Utilities

 $Id$
"""

import warnings

import pandas as pd
import xarray as xr

try:
    import pytplot
except:
    pytplot = None


def _cast_xarray(var):
    "cast input (scalar or sequence) into xarray's DataArray"
    if isinstance(var, str) and pytplot is not None:
        return pytplot.data_quants[var]
    elif isinstance(var, xr.DataArray):
        return var
    elif hasattr(var, '__iter__'):
        return list([_cast_xarray(v) for v in var])
    else:
        raise ValueError('Unrecognized input')


def _cast_list(var):
    if not isinstance(var, list):
        return list([var])
    else:
        return var


def _process_kwargs(opt, kwargs, key, newkey=None):
    if newkey is None:
        newkey = key
    if key in kwargs:
        opt[key] = kwargs[key]


def set_plot_options(data, **kwargs):
    option_table = {
        # x axis
        'xlabel'        : ('xaxis_opt', 'axis_label', ),
        'x_label'       : ('xaxis_opt', 'axis_label', ),
        'x_type'        : ('xaxis_opt', 'x_axis_type', ),
        'x_range'       : ('xaxis_opt', 'x_range', ),
        # y axis
        'ylabel'        : ('yaxis_opt', 'axis_label', ),
        'y_label'       : ('yaxis_opt', 'axis_label', ),
        'y_type'        : ('yaxis_opt', 'y_axis_type', ),
        'y_range'       : ('yaxis_opt', 'y_range', ),
        'legend'        : ('yaxis_opt', 'legend_names', ),
        # z axis
        'zlabel'        : ('zaxis_opt', 'axis_label', ),
        'z_label'       : ('zaxis_opt', 'axis_label', ),
        'z_type'        : ('zaxis_opt', 'z_axis_type', ),
        'z_range'       : ('zaxis_opt', 'z_range', ),
        # other
        'trange'        : ('trange', ),
        't_range'       : ('trange', ),
        'fontsize'      : ('extras', 'char_size', ),
        'char_size'     : ('extras', 'char_size', ),
        'linecolor'     : ('extras', 'line_color', ),
        'line_color'    : ('extras', 'line_color', ),
    }
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


def time_clip(var, t1, t2):
    t1 = pd.Timestamp(t1).timestamp()
    t2 = pd.Timestamp(t2).timestamp()

    var = _cast_list(_cast_xarray(var))
    ret = [v.loc[t1:t2] for v in var]
    if len(ret) == 1:
        return ret[0]
    else:
        return ret
