# -*- coding: utf-8 -*-

""" Utilities

 $Id$
"""

import pandas as pd
import xarray as xr

try:
    import pytplot
except:
    pytplot = None


def _cast_xarray(var):
    "cast input (scalar or sequence) into list of xarray's DataArray"
    if isinstance(var, str) and pytplot is not None:
        return pytplot.data_quants[var]
    elif isinstance(var, xr.DataArray):
        return var
    elif hasattr(var, '__iter__'):
        return list([_cast_xarray(v) for v in var])
    else:
        raise ValueError('Unrecognized input')


def _process_kwargs(opt, kwargs, key, newkey=None):
    if newkey is None:
        newkey = key
    if key in kwargs:
        opt[key] = kwargs[key]


def time_clip(var, t1, t2):
    t1 = pd.Timestamp(t1).timestamp()
    t2 = pd.Timestamp(t2).timestamp()

    var = _cast_xarray(var)
    ret = [v.loc[t1:t2] for v in var]
    if len(ret) == 1:
        return ret[0]
    else:
        return ret
