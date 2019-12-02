# -*- coding: utf-8 -*-

""" Wave analysis tools

References
----------
- Santolik et al., J. Geophys. Res., 107, 1444, 2002
- Santolik et al., Rdaio Sci., 38(1), 1010, 2003
- Santolik et al., J. Geophys. Res., 115, A00F13, 2010
"""

import numpy as np
import scipy as sp
from scipy import fftpack, signal, ndimage, constants

import xarray as xr
import pandas as pd

from insitu import cast_list
from insitu import set_plot_option


def _default_attrs_spectrogram():
    default_attrs = {
        'plot_options' : {
            'xaxis_opt' : {
                'axis_label' : 'Time',
                'x_axis_type' : 'linear',
            },
            'yaxis_opt' : {
                'axis_label' : 'Freq [Hz]',
                'y_axis_type' : 'log',
                'y_range' : [0.0, 1.0],
            },
            'zaxis_opt' : {
                'axis_label' : '',
                'z_axis_type' : 'linear',
            },
            'trange' : [0.0, 1.0],
            'extras' : {
                'spec' : True,
                'colormap' : ['viridis'],
                'panel_size' : 1,
                'char_size' : 10,
            },
        },
    }
    return default_attrs


def get_mfa_unit_vector(bx, by, bz):
    """calculate unit vectors for Magnetic-Field-Aligned coordinate

    e1 : perpendicular to B and lies in the x-z plane
    e2 : e3 x e1
    e3 : parallel to B

    Parameters
    ----------
    bx, by, bz : array-like
        three components of magnetic field

    Returns
    -------
    e1, e2, e3 : array-like
        unit vectors
    """
    bx = np.atleast_1d(bx)
    by = np.atleast_1d(by)
    bz = np.atleast_1d(bz)
    bb = np.sqrt(bx**2 + by**2 + bz**2) + 1.0e-32
    sh = bb.shape + (3,)
    e1 = np.zeros(sh, np.float64)
    e2 = np.zeros(sh, np.float64)
    e3 = np.zeros(sh, np.float64)
    # e3 parallel to B
    e3[...,0] = bx / bb
    e3[...,1] = by / bb
    e3[...,2] = bz / bb
    # e1 is perpendicular to B and in x-z plane
    e1z = -e3[...,0]/(e3[...,2] + 1.0e-32)
    e1[...,0] = 1.0 / np.sqrt(1.0 + e1z**2)
    e1[...,1] = 0.0
    e1[...,2] = e1z / np.sqrt(1.0 + e1z**2)
    # e2 = e3 x e1
    e2 = np.cross(e3, e1, axis=-1)
    # back to scalar
    if bx.size == 1 and by.size == 1 and bz.size == 1:
        e1 = e1[0,:]
        e2 = e2[0,:]
        e3 = e3[0,:]
    return e1, e2, e3


def transform_vector(vx, vy, vz, e1, e2, e3):
    """transform vector (vx, vy, vz) to given coordinate system (e1, e2, e3)

    Parameters
    ----------
    vx, vy, vz : array-like
        input vector
    e1, e2, e3 : array-like
        unit vectors for the new coordinate system

    Returns
    -------
    v1, v2, v3 : array-like
        each vector component in the new coordinate system
    """
    if e1.ndim == 1 and e2.ndim == 1 and e2.ndim == 1:
        v1 = vx*e1[0] + vy*e1[1] + vz*e1[2]
        v2 = vx*e2[0] + vy*e2[1] + vz*e2[2]
        v3 = vx*e3[0] + vy*e3[1] + vz*e3[2]
    else:
        v1 = vx*e1[:,None,0] + vy*e1[:,None,1] + vz*e1[:,None,2]
        v2 = vx*e2[:,None,0] + vy*e2[:,None,1] + vz*e2[:,None,2]
        v3 = vx*e3[:,None,0] + vy*e3[:,None,1] + vz*e3[:,None,2]
    return v1, v2, v3


def segmentalize(x, nperseg, noverlap):
    """segmentalize the input array

    This may be useful for custom spectrogram calculation.

    Parameters
    ----------
    x : array-like
        input array
    nperseg : int
        data size for each segment
    noverlap : int
        data size for for overlap interval (default nperseg/2)

    Returns
    -------
    segmentalized data
    """
    step = nperseg - noverlap
    sh = x.shape[:-1] + ((x.shape[-1]-noverlap)//step, nperseg)
    st = x.strides[:-1] + (step*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=sh, strides=st,
                                             writeable=False)
    return result


def spectrogram(x, fs, nperseg, noverlap=None, window='blackman'):
    """calculate spectrogram

    Parameters
    ----------
    x : array-like or list of array-like
        time series data
    fs : float
        sampling frequency
    nperseg : int
        number of data points for each segment
    noverlap : int
        number of overlapped data points
    window : str
        window applied for each segment

    Returns
    -------
    """
    if noverlap is None:
        noverlap = nperseg // 2
        args = {
            'nperseg'  : nperseg,
            'noverlap' : noverlap,
            'fs'       : fs,
            'window'   : window,
        }

    # calculate sum of all input
    x = cast_list(x)
    f, t, s = signal.spectrogram(x[0], **args)
    for i in range(1, len(x)):
        ff, tt, ss = signal.spectrogram(x[i], **args)
        if s.shape == ss.shape:
            s[...] = s[...] + ss
        else:
            raise ValueError('Invalid input data')

    # discard zero frequency
    f = f[1:]
    s = s[1:,:]

    # return xarray if input is xarray
    is_xarray = np.all([isinstance(xx, xr.DataArray) for xx in x])
    if is_xarray:
        t = t + x[0].time.values[0]
        s = s.transpose()
        f = np.repeat(f[np.newaxis,:], s.shape[0], axis=0)
        bins = xr.DataArray(f, dims=('time', 'f'), coords={'time' : t})

        # DataArray
        args = {
            'dims'   : ('time', 'f'),
            'coords' : {
                'time' : t,
                'spec_bins' : bins
            },
        }
        data = xr.DataArray(s, **args)

        # set attribute
        data.attrs = _default_attrs_spectrogram()
        set_plot_option(data,
                        yrange=[f[0], f[-1]],
                        trange=[t[0], t[-1]],
                        z_type='log',
                        colormap='viridis')

        return data
    else:
        # otherwise simple sum of all spectra
        return f, t, s


class SVD:
    """ Magnetic Singular Value Decomposition Method

    """
    def __init__(self, **kwargs):
        self.sps_ace  = 8192
        self.sps_acb  = 8192
        self.sps_dcb  = 128
        self.nperseg  = self.sps_acb // 8
        self.noverlap = self.nperseg // 8
        self.window   = np.blackman
        self.wsmooth  = np.blackman(7)
        # update attribute given by keyword arguments
        for key, item in kwargs.items():
            setattr(self, key, item)

    def calc_mfa_coord(self, dcb, tb):
        # magnetic field averaged over bins
        time = dcb.coords['time']
        bb = dcb.groupby_bins(time, tb).mean().values
        bt = np.sqrt(np.sum(bb[:,0:3]**2, axis=1))
        bx = np.array(bb[:,0] / bt)
        by = np.array(bb[:,1] / bt)
        bz = np.array(bb[:,2] / bt)
        return get_mfa_unit_vector(bx, by, bz)

    def spectral_matrix(self, acb, dcb):
        # spectral matrix
        convolve = ndimage.filters.convolve1d
        sps_acb  = float(self.sps_acb)
        sps_dcb  = float(self.sps_dcb)
        nperseg  = self.nperseg
        noverlap = self.noverlap
        nsegment = nperseg - noverlap
        nfreq    = nperseg // 2
        window   = self.window
        wsmooth  = self.wsmooth
        # data
        nt = acb.shape[0]
        bx = acb.values[:,0]
        by = acb.values[:,1]
        bz = acb.values[:,2]
        ww = window(nperseg)
        ww = ww / ww.sum()
        mt = (nt - noverlap)//nsegment
        # segmentalize
        Bx = segmentalize(bx, nperseg, noverlap) * ww[None,:]
        By = segmentalize(by, nperseg, noverlap) * ww[None,:]
        Bz = segmentalize(bz, nperseg, noverlap) * ww[None,:]
        # time and frequency coordinate
        dt = nsegment / sps_acb
        tb = acb.time[::nsegment].values # bin edges
        tt = 0.5*(tb[+1:] + tb[:-1])     # bin center
        ff = np.arange(1, nfreq+1)/(nperseg/sps_acb)
        # coordinate transformation and FFT
        e1, e2, e3 = self.calc_mfa_coord(dcb, tb)
        B1, B2, B3 = transform_vector(Bx, By, Bz, e1, e2, e3)
        B1 = fftpack.fft(B1, axis=-1)[:,1:nfreq+1].T
        B2 = fftpack.fft(B2, axis=-1)[:,1:nfreq+1].T
        B3 = fftpack.fft(B3, axis=-1)[:,1:nfreq+1].T
        # calculate 3x3 spectral matrix with smoothing
        ws  = wsmooth / wsmooth.sum()
        Q00 = B1 * np.conj(B1)
        Q01 = B1 * np.conj(B2)
        Q02 = B1 * np.conj(B3)
        Q11 = B2 * np.conj(B2)
        Q12 = B2 * np.conj(B3)
        Q22 = B3 * np.conj(B3)
        Q00_re = convolve(Q00.real, ws, mode='nearest')
        Q00_im = convolve(Q00.imag, ws, mode='nearest')
        Q01_re = convolve(Q01.real, ws, mode='nearest')
        Q01_im = convolve(Q01.imag, ws, mode='nearest')
        Q02_re = convolve(Q02.real, ws, mode='nearest')
        Q02_im = convolve(Q02.imag, ws, mode='nearest')
        Q11_re = convolve(Q11.real, ws, mode='nearest')
        Q11_im = convolve(Q11.imag, ws, mode='nearest')
        Q12_re = convolve(Q12.real, ws, mode='nearest')
        Q12_im = convolve(Q12.imag, ws, mode='nearest')
        Q22_re = convolve(Q22.real, ws, mode='nearest')
        Q22_im = convolve(Q22.imag, ws, mode='nearest')
        # real representation (6x3) for spectral matrix
        N = B1.shape[0]
        M = B1.shape[1]
        S = np.zeros((N, M, 6, 3), np.float64)
        S[:,:,0,0] = Q00_re
        S[:,:,0,1] = Q01_re
        S[:,:,0,2] = Q02_re
        S[:,:,1,0] = Q01_re
        S[:,:,1,1] = Q11_re
        S[:,:,1,2] = Q12_re
        S[:,:,2,0] = Q02_re
        S[:,:,2,1] = Q12_re
        S[:,:,2,2] = Q22_re
        S[:,:,3,0] = 0
        S[:,:,3,1] = Q01_im
        S[:,:,3,2] = Q02_im
        S[:,:,4,0] =-Q01_im
        S[:,:,4,1] = 0
        S[:,:,4,2] = Q12_im
        S[:,:,5,0] =-Q02_im
        S[:,:,5,1] =-Q12_im
        S[:,:,5,2] = 0
        return tt, ff, S

    def poynting(self, ace, acb, dcb):
        # calculate Poynting vector
        convolve = ndimage.filters.convolve1d
        sps_ace  = float(self.sps_ace)
        sps_acb  = float(self.sps_acb)
        sps_dcb  = float(self.sps_dcb)
        nperseg  = self.nperseg
        noverlap = self.noverlap
        nsegment = nperseg - noverlap
        nfreq    = nperseg // 2
        window   = self.window
        wsmooth  = self.wsmooth
        # data size and window
        nt = acb.shape[0]
        bx = acb.values[:,0]
        by = acb.values[:,1]
        bz = acb.values[:,2]
        ww = window(nperseg)
        ww = ww / ww.sum()
        mt = (nt - noverlap)//nsegment
        # interpolate electric field
        ef = ace.interp(time=acb.time)
        ex = ef.values[:,0]
        ey = ef.values[:,1]
        ez = ef.values[:,2]
        # segmentalize
        Ex = segmentalize(ex, nperseg, noverlap) * ww[None,:]
        Ey = segmentalize(ey, nperseg, noverlap) * ww[None,:]
        Ez = segmentalize(ez, nperseg, noverlap) * ww[None,:]
        Bx = segmentalize(bx, nperseg, noverlap) * ww[None,:]
        By = segmentalize(by, nperseg, noverlap) * ww[None,:]
        Bz = segmentalize(bz, nperseg, noverlap) * ww[None,:]
        # time and frequency coordinate
        dt = nsegment / sps_acb
        tb = acb.time[::nsegment].values # bin edges
        tt = 0.5*(tb[+1:] + tb[:-1])     # bin center
        ff = np.arange(1, nfreq+1)/(nperseg/sps_acb)
        # coordinate transformation and FFT
        e1, e2, e3 = self.calc_mfa_coord(dcb, tb)
        E1, E2, E3 = transform_vector(Ex, Ey, Ez, e1, e2, e3)
        B1, B2, B3 = transform_vector(Bx, By, Bz, e1, e2, e3)
        # FFT
        E1 = fftpack.fft(E1, axis=-1)[:,1:nfreq+1].T
        E2 = fftpack.fft(E2, axis=-1)[:,1:nfreq+1].T
        E3 = fftpack.fft(E3, axis=-1)[:,1:nfreq+1].T
        B1 = fftpack.fft(B1, axis=-1)[:,1:nfreq+1].T
        B2 = fftpack.fft(B2, axis=-1)[:,1:nfreq+1].T
        B3 = fftpack.fft(B3, axis=-1)[:,1:nfreq+1].T
        # calculate Poynting flux from cross spectral matrix
        ws  = wsmooth / wsmooth.sum()
        S1  = (E2 * np.conj(B3) - E3 * np.conj(B2)).real
        S2  = (E3 * np.conj(B1) - E1 * np.conj(B3)).real
        S3  = (E1 * np.conj(B2) - E2 * np.conj(B1)).real
        # E [mV/m] * B [nT] => unit conversion factor = 1.0e-12
        mu0 = constants.mu_0
        s1  = convolve(S1, ws, mode='nearest') / mu0 * 1.0e-12
        s2  = convolve(S2, ws, mode='nearest') / mu0 * 1.0e-12
        s3  = convolve(S3, ws, mode='nearest') / mu0 * 1.0e-12
        ss  = np.sqrt(S1**2 + S2**2 + S3**2)
        tsb = np.rad2deg(np.abs(np.arctan2(np.sqrt(s1**2 + s2**2), s3)))
        psb = np.rad2deg(np.arctan2(s2, s1))
        # store result
        r  = dict()
        r['s1']       = s1 / ss
        r['s2']       = s2 / ss
        r['s3']       = s3 / ss
        r['theta_sb'] = tsb
        r['phi_sb']   = psb
        return r

    def svd(self, acb, dcb):
        # calculate spectral matrix
        t, f, S = self.spectral_matrix(acb, dcb)
        # perform SVD only for valid data
        N, M, _, _ = S.shape
        T = S.reshape(N*M, 6, 3)
        I = np.argwhere(np.isfinite(np.sum(T, axis=(-2, -1))))[:,0]
        UU = np.zeros((N*M, 6, 6), np.float64)
        WW = np.zeros((N*M, 3), np.float64)
        VV = np.zeros((N*M, 3, 3), np.float64)
        UU[I], WW[I], VV[I] = np.linalg.svd(T[I])
        U = UU.reshape(N, M, 6, 6)
        W = WW.reshape(N, M, 3)
        V = VV.reshape(N, M, 3, 3)
        self.svd_result = dict(t=t, f=f, S=S, U=U, W=W, V=V)
        return t, f, self._process_svd_result(S, U, W, V)

    def _process_svd_result(self, S, U, W, V):
        eps = 1.0e-34
        Tr  = lambda x: np.trace(x, axis1=2, axis2=3)
        SS  = S[...,0:3,0:3] + S[...,3:6,0:3]*1j

        r = dict()

        ### power spectral density
        r['psd'] = Tr(np.abs(SS))

        ### degree of polarization
        r['degpol'] = 1.5*(Tr(np.matmul(SS,SS))/(Tr(SS)**2+eps)).real - 0.5

        ### planarity
        r['planarity'] = 1 - np.sqrt(W[...,2]/(W[...,0]+eps))

        ### ellipticity
        r['ellipticity'] = W[...,1]/(W[...,0]+eps) * np.sign(SS[...,0,1].imag)

        ### k vector
        k1  = np.sign(V[...,2,2])*V[...,2,0]
        k2  = np.sign(V[...,2,2])*V[...,2,1]
        k3  = np.sign(V[...,2,2])*V[...,2,2]
        kk  = np.sqrt(k1**2 + k2**2 + k3**2)
        tkb = np.rad2deg(np.abs(np.arctan2(np.sqrt(k1**2 + k2**2), k3)))
        pkb = np.rad2deg(np.arctan2(k2, k1))
        r['n1']       = k1 / kk
        r['n2']       = k2 / kk
        r['n3']       = k3 / kk
        r['theta_kb'] = tkb
        r['phi_kb']   = pkb

        return r

    def _setup_arrays(self, t, f, result):
        default_args = {
            'dims'   : ('time', 'f'),
            'coords' : {
                'time' : t,
                'spec_bins' : ('f', f),
            },
        }

        # construct DataArray and store in dict
        dadict = dict()
        for key in result.keys():
            try:
                data = xr.DataArray(result[key].transpose(), **default_args)
                data.name = key
                data.attrs = _default_attrs_spectrogram()
                set_plot_option(data,
                                yrange=[f[0], f[-1]],
                                trange=[t[0], t[-1]])
                dadict[key] = data
            except Exception as e:
                print('Error in creating spectrogram for : %s' % (key))
                print(e)

        # power spectral density
        if 'psd' in dadict:
            zmax = np.ceil(np.log10(np.max(dadict['psd'])))
            zmin = zmax - 7
            set_plot_option(dadict['psd'],
                            zlabel='log10(PSD [nT^2/Hz])',
                            zrange=[zmin, zmax],
                            colormap='jet',
                            ztype='log')

        # degree of polarization
        if 'degpol' in dadict:
            set_plot_option(dadict['degpol'],
                            zlabel='Deg. Pol',
                            zrange=[0.0, +1.0],
                            colormap='greens')

        # planarity
        if 'planarity' in dadict:
            set_plot_option(dadict['planarity'],
                            zlabel='Planarity',
                            zrange=[0.0, +1.0],
                            colormap='greens')

        # ellipticity
        if 'ellipticity' in dadict:
            set_plot_option(dadict['ellipticity'],
                            zlabel='Ellipticity',
                            zrange=[-1.0, +1.0],
                            colormap='bwr')

        # k vector
        for nn in ('n1', 'n2', 'n3'):
            if nn in dadict:
                set_plot_option(dadict[nn],
                                zlabel=nn,
                                zrange=[-1, +1],
                                colormap='bwr')

        if 'theta_kb' in dadict:
            set_plot_option(dadict['theta_kb'],
                            zlabel='theta_kb',
                            zrange=[0.0, 90.0],
                            colormap='bwr')

        if 'phi_kb' in dadict:
            set_plot_option(dadict['phi_kb'],
                            zlabel='phi_kb',
                            zrange=[0.0, 180.0],
                            colormap='bwr')

        # poynting flux
        for ss in ('s1', 's2', 's3'):
            if ss in dadict:
                set_plot_option(dadict[ss],
                                zlabel=ss,
                                zrange=[-1, +1],
                                colormap='bwr')

        if 'theta_sb' in dadict:
            set_plot_option(dadict['theta_sb'],
                            zlabel='theta_sb',
                            zrange=[0.0, 180.0],
                            colormap='bwr')

        if 'phi_sb' in dadict:
            set_plot_option(dadict['phi_sb'],
                            zlabel='phi_sb',
                            zrange=[0.0, 180.0],
                            colormap='bwr')


        return dadict

    def analyze(self, ace, acb, dcb):
        t, f, result1 = self.svd(acb, dcb)
        result2 = self.poynting(ace, acb, dcb)

        result = dict()
        result.update(result1)
        result.update(result2)
        dadict = self._setup_arrays(t, f, result)

        return dadict

    def smooth_result(self, xx):
        convolve = ndimage.filters.convolve1d
        ws  = self.wsmooth / self.wsmooth.sum()
        return tuple([convolve(x, ws, mode='nearest') for x in xx])

