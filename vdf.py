# -*- coding: utf-8 -*-

"""Velocity Distribution Function Analysis Tools

"""

import numpy as np
import scipy as sp
import xarray as xr

from . import const
from .utils import *
from .attrdict import AttrDict


class VDF(object):
    """Velocity Distribution Function Object
    """
    def __init__(self, dist, **kwargs):
        self.dataset = create_dataset(dist, **kwargs)

    def slice(self, time, **kwargs):
        return slice_plane(self.dataset, time, **kwargs)


def _extend_mesh_interp(fv, vr, vt, vp):
    # assume spherical coordinate
    r = np.concatenate([[vr[0] * 0.99], vr, [vr[-1]*1.01]])
    t = np.concatenate([[0.0], vt, [np.pi]])
    p = np.concatenate([[vp[-1]-2*np.pi], vp, [vp[0]+2*np.pi]])
    f = np.zeros((p.size, t.size, r.size), dtype=np.float64)
    f[+1:-1,+1:-1,+1:-1] = fv
    # radial
    f[:,:, 0] = f[:,:,+1]
    f[:,:,-1] = f[:,:,-2]
    # theta
    f[:, 0,:] = f[:,+1,:].mean(axis=0)[None,None,:]
    f[:,-1,:] = f[:,-2,:].mean(axis=0)[None,None,:]
    # phi
    f[ 0,:,:] = f[-2,:,:]
    f[-1,:,:] = f[+1,:,:]
    return f, r, t, p


def _get_mesh_plane(a, b, c, origin, vrmin, vrmax, n1, n2):
    t  = np.linspace(0.0, 2*np.pi, n2)[:,None]
    r  = np.logspace(np.log10(vrmin), np.log10(vrmax), n1)[None,:]
    va = (r * np.cos(t)).ravel()
    vb = (r * np.sin(t)).ravel()
    vc = np.ones_like(va) * origin
    vx = va * a[0] + vb * b[0] + vc * c[0]
    vy = va * a[1] + vb * b[1] + vc * c[1]
    vz = va * a[2] + vb * b[2] + vc * c[2]

    vx = vx.reshape((n2, n1))
    vy = vy.reshape((n2, n1))
    vz = vz.reshape((n2, n1))
    va = va.reshape((n2, n1))
    vb = vb.reshape((n2, n1))

    return vx, vy, vz, va, vb


def _get_interpolator(fv, vr, vt, vp):
    from scipy.interpolate import RegularGridInterpolator
    f, r, t, p = _extend_mesh_interp(fv, vr, vt, vp)
    points = (p, t, r)
    kwargs = dict(bounds_error=False,
                  fill_value=0.0)
    return RegularGridInterpolator(points, f, **kwargs)


def create_dataset(dist, **kwargs):
    """Create Dataset of 3D distribution function with attached coordinates

    Parameters
    ----------
    dist : xarray.DataArray
        DataArray for velocity distribution function

    Returns
    -------
    xarray.Dataset containing velocity distribution function and three
    coordiante axes with aligned time index.
    """
    tindex = dist.time.values
    bvec = kwargs.get('bvec', None)
    cvec = kwargs.get('cvec', None)
    evec = kwargs.get('evec', None)
    if bvec is not None:
        # bvec is parallel to bfield
        tt = bvec.coords['time']
        bb = bvec.groupby_bins(tt, tindex).mean().values[:,0:3]
        bb = bb / np.linalg.norm(bb, axis=-1)[:,None]
        if cvec is not None:
            # cvec is parallel to bulk velocity
            tt = cvec.coords['time']
            cc = cvec.groupby_bins(tt, tindex).mean().values[:,0:3]
            cc = cc - np.sum(cc*bb, axis=-1)[:,None]*bb
            cc = cc / np.linalg.norm(cc, axis=-1)[:,None]
            ee = np.cross(bb, cc)
        elif evec is not None:
            # evec is parallel to efield
            tt = evec.coords['time']
            ee = evec.groupby_bins(tt, tindex).mean().values[:,0:3]
            ee = ee - np.sum(ee*bb, axis=-1)[:,None]*bb
            ee = ee / np.linalg.norm(ee, axis=-1)[:,None]
            cc = np.cross(ee, bb)
        else:
            # cvec is perpendicular to B and lies in x-z plane
            cc = np.zeros_like(bb)
            cz = -bb[...,0]/(bb[...,2] + 1.0e-32)
            cc[...,0] = 1.0 / np.sqrt(1.0 + cz**2)
            cc[...,1] = 0.0
            cc[...,2] = cz / np.sqrt(1.0 + cz**2)
            # evec = bvec x cvec
            ee = np.cross(bb, cc)
    else:
        # use spacecraft frame as bfield is not given
        bb = np.zeros((tindex.size-1, 3))
        cc = np.zeros((tindex.size-1, 3))
        ee = np.zeros((tindex.size-1, 3))
        bb[:,2] = 1.0
        cc[:,0] = 1.0
        ee[:,1] = 1.0

    qm = kwargs.get('qm', const.qme)
    fv = dist[:-1,...]
    vp = np.deg2rad(fv.v1.values)
    vt = np.deg2rad(fv.v2.values[None,:].repeat(fv.shape[0], axis=0))
    vr = np.sqrt(2*qm*fv.v3.values) * 1.0e-3

    distarray = xr.DataArray(np.array(fv.values, dtype=np.float64),
                             name='dist',
                             dims=['time', 'vp_dims', 'vt_dims', 'vr_dims'],
                             coords={'time' : fv.time,
                                     'vp'   : (('time', 'vp_dims'), vp),
                                     'vt'   : (('time', 'vt_dims'), vt),
                                     'vr'   : (('time', 'vr_dims'), vr)},
                             attrs={'qm' : qm})
    bvecarray = xr.DataArray(bb,
                             name='bvec',
                             dims=['time', 'xyz'],
                             coords={'time' : fv.time, 'xyz' : np.arange(3)})
    cvecarray = xr.DataArray(cc,
                             name='cvec',
                             dims=['time', 'xyz'],
                             coords={'time' : fv.time, 'xyz' : np.arange(3)})
    evecarray = xr.DataArray(ee,
                             name='cvec',
                             dims=['time', 'xyz'],
                             coords={'time' : fv.time, 'xyz' : np.arange(3)})

    # dist error
    disterr = kwargs.get('disterr', None)
    if disterr is not None:
        gv = disterr.interp(time=fv.time)
        disterrarray = xr.DataArray(np.array(gv.values, dtype=np.float64),
                                    name='disterr',
                                    dims=distarray.dims,
                                    coords=distarray.coords,
                                    attrs=distarray.attrs)
    else:
        disterrarray = None


    return xr.Dataset({'dist' : distarray,
                       'bvec' : bvecarray,
                       'cvec' : cvecarray,
                       'evec' : evecarray,
                       'disterr' : disterrarray,})


def interp(fv, vr, vt, vp, ux, uy, uz, method='nearest'):
    """Interpolation of 3D distribution funciton at arbitrary velocities

    Parameters
    ----------
    fv : 3D array
        velocity distribution function
    vr : 1D array
        radial velocity coordiante
    vt : 1D array
        polar angle coordiante (theta) in radian
    vp : 1D array
        azimuthal angle coordiante (phi) in radian
    ux, uy, uz : array-like
        cartesian velocities at which interpolated values are calculated

    Returns
    -------
    interpolated values of distribution function
    """
    # check input
    if ux.shape == uy.shape and ux.shape == uz.shape:
        shape = ux.shape
        ur, ut, up = xyz2sph(ux, uy, uz, degree=False)
        up = np.where(up < 0, 2*np.pi + up, up)
    else:
        raise ValueError('Invalid input')

    # get interpolator
    interpfunc = _get_interpolator(fv, vr, vt, vp)

    return interpfunc((up.flat, ut.flat, ur.flat), method=method).reshape(shape)


def slice_plane(data, time, **kwargs):
    """Calculate slice of distribution function on a specified plane

    Parameters
    ----------
    data : xarray.Dataset
       Dataset contains distribution function and three coordiante axes
    time : int or str or object that can be converted to unixtime
       interpolation is calculated for the time snapshot

    Returns
    -------
    AttrDict object containing the result of interpolation
    """
    if type(time) == int:
        ds = dta.isel(time=tt)
    else:
        tt = to_unixtime(time)
        ds = data.sel(time=tt, method='nearest')
    fv = ds.dist.values[...]
    bb = ds.bvec.values[...]
    cc = ds.cvec.values[...]
    ee = ds.evec.values[...]
    vr = ds.vr.values[...]
    vt = ds.vt.values[...]
    vp = ds.vp.values[...]

    # select velocity plane
    normdir = kwargs.get('normdir', None)
    if normdir is None:
        av, bv, cv = cc, ee, bb
        labels = ('C', 'B', )
    elif normdir == 'c':
        av, bv, cv = ee, bb, cc
        labels = ('E', 'B', )
    elif normdir == 'e':
        av, bv, cv = bb, cc, ee
        labels = ('B', 'C', )
    elif normdir == 'b':
        av, bv, cv = cc, ee, bb
        labels = ('C', 'E', )
    elif normdir == 'x':
        av = np.array([0.0, 1.0, 0.0])
        bv = np.array([0.0, 0.0, 1.0])
        cv = np.array([1.0, 0.0, 0.0])
        labels = ('Y', 'Z', )
    elif normdir == 'y':
        av = np.array([0.0, 0.0, 1.0])
        bv = np.array([1.0, 0.0, 0.0])
        cv = np.array([0.0, 1.0, 0.0])
        labels = ('Z', 'X', )
    elif normdir == 'z':
        av = np.array([1.0, 0.0, 0.0])
        bv = np.array([0.0, 1.0, 0.0])
        cv = np.array([0.0, 0.0, 1.0])
        labels = ('X', 'Y', )
    else:
        raise ValueError('Invalid input')

    # interpolation on velocity plane
    opts = {
        'origin'  : kwargs.get('origin', 0.0),
        'vrmin'   : kwargs.get('vrmin', vr[ 0]),
        'vrmax'   : kwargs.get('vrmax', vr[-1]),
        'n1'      : kwargs.get('n1', ds.vr_dims.size),
        'n2'      : kwargs.get('n2', ds.vt_dims.size),
    }
    ux, uy, uz, ua, ub = _get_mesh_plane(av, bv, cv, **opts)
    method  = kwargs.get('method', 'linear')
    if kwargs.get('look_direction', False):
        gv = interp(fv, vr, vt, vp, ux, uy, uz, method)
    else:
        gv = interp(fv, vr, vt, vp,-ux,-uy,-uz, method)

    # return result as dict
    result = AttrDict({
        'dist'     : gv,
        'v1'       : ua,
        'v2'       : ub,
        'v1_label' : labels[0],
        'v2_label' : labels[1],
        'time'     : ds.dist.time.values,
    })

    return result


def get_vtk(data, time):
    # temporary function generating vtk data structure for experiment
    from tvtk.api import tvtk
    tt = to_unixtime(time)
    ds = data.sel(time=tt, method='nearest')
    fv = ds.dist.values[0,...]
    vr = ds.vr.values[0,...]
    vt = ds.vt.values[0,...]
    vp = ds.vp.values[0,...]
    f, r, t, p = _extend_mesh_interp(fv, vr, vt, vp)
    fmax = f.max()
    fmin = fmax * 1.0e-15
    f  = np.clip(f[:-1,...], fmin, fmax) # eliminate zeros
    p  = p[:-1,...]
    rr = r[None,None,:]
    tt = t[None,:,None]
    pp = p[:,None,None]
    dims = f.shape
    mesh = np.zeros((np.prod(dims), 3), dtype=np.float64)
    mesh[:,0] = (rr*np.sin(tt)*np.cos(pp)).ravel()
    mesh[:,1] = (rr*np.sin(tt)*np.sin(pp)).ravel()
    mesh[:,2] = (rr*np.cos(tt)*np.ones_like(pp)).ravel()
    sgrid = tvtk.StructuredGrid(dimensions=dims[::-1])
    sgrid.points = np.zeros((np.prod(dims), 3), dtype=np.float64)
    sgrid.points = mesh
    sgrid.point_data.scalars = np.log10(f.ravel())
    sgrid.point_data.scalars.name = 'VDF'
    return tvtk.to_vtk(sgrid)

