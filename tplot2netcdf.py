# -*- coding: utf-8 -*-

""" tplot2netcdf

"""

import os
import pickle
import base64
import xarray as xr

try:
    import pytplot
except:
    pytplot = None

TPLOT_ATTRS = 'tplot_attrs'


def _encode_attrs(attrs):
    return base64.b64encode(pickle.dumps(attrs)).decode('utf-8')


def _decode_attrs(attrs):
    return pickle.loads(base64.b64decode(attrs.encode()))


def _get_encoded_xarray(da):
    attrs = { TPLOT_ATTRS : _encode_attrs(da.attrs) }
    temp = xr.DataArray(name=da.name,
                        data=da.values,
                        coords=da.coords,
                        dims=da.dims,
                        attrs=attrs)
    return temp


def save(tplotvars, filename, replace=False, verbose=True):
    if not isinstance(tplotvars, list):
        tplotvars = list([tplotvars])

    # mode
    if not os.path.exists(filename) or replace:
        mode = 'w'
    else:
        mode = 'a'

    # save for each variables
    for var in tplotvars:
        temp = None
        if isinstance(var, str) and pytplot is not None:
            temp = _get_encoded_xarray(pytplot.get_data(var, xarray=True))
        elif isinstance(var, xr.DataArray):
            temp = _get_encoded_xarray(var)
        else:
            print('Error: ignoring unknown input variables : ', var)
            continue

        # write to disk
        if isinstance(temp, xr.DataArray) and hasattr(temp, 'name'):
            temp.to_netcdf(filename, group=temp.name, mode=mode)
            mode = 'a'

            if verbose:
                print('DataArray %s was saved to %s ...' % (temp.name, filename))


def load(filename, tplot=True, verbose=True):
    # get list of groups
    import netCDF4
    with netCDF4.Dataset(filename) as nc:
        tplotvars = [group for group in nc.groups]

    dadict = dict()
    for var in tplotvars:
        da = xr.open_dataarray(filename, group=var)
        if TPLOT_ATTRS in da.attrs:
            da.attrs = _decode_attrs(da.attrs[TPLOT_ATTRS])
            dadict[var] = da
            if verbose:
                print('DataArray %s was loaded from %s ...'
                      % (da.name, filename))

    if tplot and pytplot is not None:
        for key, item in dadict.items():
            pytplot.data_quants[key] = item
        if verbose:
            print('Loaded data are stored in tplot variables')

    return dadict
