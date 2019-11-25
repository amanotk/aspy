# -*- coding: utf-8 -*-

""" wave analysis tools

"""

import numpy as np
import scipy as sp
from scipy import signal
from scipy import fftpack
from scipy import ndimage

import xarray as xr
import pandas as pd

from insitu import _cast_list


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
    x = _cast_list(x)
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
        bins = xr.DataArray(f, dims=('time', 'v'), coords={'time' : t})

        # DataArray
        args = {
            'dims'   : ('time', 'v'),
            'coords' : {
                'time' : t,
                'spec_bins' : bins
            },
        }
        data = xr.DataArray(s, **args)

        # set attribute
        data.attrs = {
            'plot_options' : {
                'xaxis_opt' : {
                    'axis_label' : 'Time',
                    'x_axis_type' : 'linear',
                },
                'yaxis_opt' : {
                    'axis_label' : 'Frequency [Hz]',
                    'y_axis_type' : 'log',
                    'y_range' : [f[0], f[-1]],
                },
                'zaxis_opt' : {
                    'axis_label' : 'PSD',
                    'z_axis_type' : 'log',
                },
                'trange' : [t[0], t[-1]],
                'extras' : {
                    'spec' : True,
                    'panel_size' : 1,
                    'char_size' : 10,
                },
            },
        }

        return data
    else:
        # otherwise simple sum of all spectra
        return f, t, s


class SVD:
    """Magnetic SVD
    """
    def __init__(self, **kwargs):
        self.sps_efd  = 8192
        self.sps_acb  = 8192
        self.sps_dcb  = 128
        self.nperseg  = self.sps_acb // 8
        self.noverlap = self.nperseg // 8
        self.naverage = 32
        self.window   = np.blackman
        self.wsmooth  = np.blackman(7)
        # update attribute given by keyword arguments
        for key, item in kwargs.items():
            setattr(self, key, item)

    def calc_mfa_coord(self, dcb, tt, navg):
        # averaged magnetic field
        bb = dcb.rolling(navg, center=True).mean()
        bb = mms.reindex_interpolate(bb, tt)
        bx = np.array(bb.x / bb.t)
        by = np.array(bb.y / bb.t)
        bz = np.array(bb.z / bb.t)
        return get_mfa_unit_vector(bx, by, bz)

    def spectral_matrix(self, acb, dcb):
        # spectral matrix
        convolve = ndimage.filters.convolve1d
        sps_acb  = float(self.sps_acb)
        sps_dcb  = float(self.sps_dcb)
        delt_acb = 1.0 / sps_acb
        delt_dcb = 1.0 / sps_dcb
        nperseg  = self.nperseg
        noverlap = self.noverlap
        nsegment = nperseg - noverlap
        naverage = self.naverage
        nfreq    = nperseg // 2
        window   = self.window
        wsmooth  = self.wsmooth
        # data
        nt = acb.index.size
        bx = acb.x
        by = acb.y
        bz = acb.z
        ww = window(nperseg)
        ww = ww / ww.sum()
        mt = (nt - noverlap)//nsegment
        # segmentalize
        Bx = segmentalize(bx, nperseg, noverlap) * ww[None,:]
        By = segmentalize(by, nperseg, noverlap) * ww[None,:]
        Bz = segmentalize(bz, nperseg, noverlap) * ww[None,:]
        # time and frequency coordinate
        dt = pd.Timedelta(nsegment/sps_acb, unit='s')
        t0 = acb.index[0] + 0.5*dt
        tt = pd.TimedeltaIndex(np.arange(mt)*dt, unit='s') + t0
        ff = np.arange(1, nfreq+1)/(nperseg/sps_acb)
        # coordinate transformation and FFT
        e1, e2, e3 = self.calc_mfa_coord(dcb, tt, naverage)
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

    def poynting(self, efd, acb, dcb):
        # calculate Poynting vector
        convolve = ndimage.filters.convolve1d
        sps_efd  = float(self.sps_efd)
        sps_acb  = float(self.sps_acb)
        sps_dcb  = float(self.sps_dcb)
        delt_efd = 1.0 / sps_efd
        delt_acb = 1.0 / sps_acb
        delt_dcb = 1.0 / sps_dcb
        nperseg  = self.nperseg
        noverlap = self.noverlap
        nsegment = nperseg - noverlap
        naverage = self.naverage
        nfreq    = nperseg // 2
        window   = self.window
        wsmooth  = self.wsmooth
        # data size and window
        nt = acb.index.size
        ww = window(nperseg)
        ww = ww / ww.sum()
        mt = (nt - noverlap)//nsegment
        # segmentalize
        Ex = segmentalize(efd.x, nperseg, noverlap) * ww[None,:]
        Ey = segmentalize(efd.y, nperseg, noverlap) * ww[None,:]
        Ez = segmentalize(efd.z, nperseg, noverlap) * ww[None,:]
        Bx = segmentalize(acb.x, nperseg, noverlap) * ww[None,:]
        By = segmentalize(acb.y, nperseg, noverlap) * ww[None,:]
        Bz = segmentalize(acb.z, nperseg, noverlap) * ww[None,:]
        # time and frequency coordinate
        dt = pd.Timedelta(nsegment/sps_acb, unit='s')
        t0 = acb.index[0] + 0.5*dt
        tt = pd.TimedeltaIndex(np.arange(mt)*dt, unit='s') + t0
        ff = np.arange(1, nfreq+1)/(nperseg/sps_acb)
        # FFT
        Ex = fftpack.fft(Ex, axis=-1)[:,1:nfreq+1]
        Ey = fftpack.fft(Ey, axis=-1)[:,1:nfreq+1]
        Ez = fftpack.fft(Ez, axis=-1)[:,1:nfreq+1]
        Bx = fftpack.fft(Bx, axis=-1)[:,1:nfreq+1]
        By = fftpack.fft(By, axis=-1)[:,1:nfreq+1]
        Bz = fftpack.fft(Bz, axis=-1)[:,1:nfreq+1]
        # calculate Poynting flux from cross spectral matrix
        ws  = wsmooth / wsmooth.sum()
        Sx1 = convolve((Ey * np.conj(Bz)).real, ws, mode='nearest')
        Sx2 = convolve((Ez * np.conj(By)).real, ws, mode='nearest')
        Sy1 = convolve((Ez * np.conj(Bx)).real, ws, mode='nearest')
        Sy2 = convolve((Ex * np.conj(Bz)).real, ws, mode='nearest')
        Sz1 = convolve((Ex * np.conj(By)).real, ws, mode='nearest')
        Sz2 = convolve((Ey * np.conj(Bx)).real, ws, mode='nearest')
        # E [mV/m] * B [nT] => unit conversion factor = 1.0e-12
        Sx  = (Sx1 - Sx2)/mms.mu0 * 1.0e-12
        Sy  = (Sy1 - Sy2)/mms.mu0 * 1.0e-12
        Sz  = (Sz1 - Sz2)/mms.mu0 * 1.0e-12
        # coordinate transformation
        e1, e2, e3 = self.calc_mfa_coord(dcb, tt, naverage)
        S1, S2, S3 = vector_transform(Sx, Sy, Sz, e1, e2, e3)
        S1 = S1.T
        S2 = S2.T
        S3 = S3.T
        pfx = np.sqrt(S1**2 + S2**2 + S3**2)
        tsb = np.rad2deg(np.abs(np.arctan2(np.sqrt(S1**2 + S2**2), S3)))
        psb = np.rad2deg(np.arctan2(S2, S1))
        pfx, tsb, psd = self.smooth_result((pfx, tsb, psb))
        # magnitude
        spec_pfx = tseries.Spectrogram(tt, ff, pfx, log=True, logy=True)
        spec_pfx.set(vmin=-12, vmax=-4, cmap=mms.cm_jet,
                     ylabel='Freq. [Hz]',
                     clabel=r'Poynting Flux [W/Hz/m$^2$/s]')
        # theta
        spec_tsb = tseries.Spectrogram(tt, ff, tsb, log=False, logy=True)
        spec_tsb.set(vmin=0, vmax=180, cmap=mms.cm_b2r,
                     ylabel='Freq. [Hz]',
                     clabel=r'$\theta_{sb}$')
        # phi
        spec_psb = tseries.Spectrogram(tt, ff, psb, log=False, logy=True)
        spec_psb.set(vmin=0, vmax=180, cmap=mms.cm_none,
                     ylabel='Freq. [Hz]',
                     clabel=r'$\phi_{sb}$')
        return spec_pfx, spec_tsb, spec_psb

    def svd(self):
        # perform SVD only for valid data
        S = self.S
        N, M, _, _ = S.shape
        T = S.reshape(N*M, 6, 3)
        I = np.argwhere(np.isfinite(np.sum(T, axis=(-2, -1))))[:,0]
        UU = np.zeros((N*M, 6, 6), np.float64)
        WW = np.zeros((N*M, 3), np.float64)
        VV = np.zeros((N*M, 3, 3), np.float64)
        UU[I], WW[I], VV[I] = np.linalg.svd(T[I])
        self.U = UU.reshape(N, M, 6, 6)
        self.W = WW.reshape(N, M, 3)
        self.V = VV.reshape(N, M, 3, 3)

    def get(self, *args):
        trace = lambda x: np.trace(x, axis1=2, axis2=3)
        SS = self.S[...,0:3,0:3] + self.S[...,3:6,0:3]*1j
        S  = self.S
        U  = self.U
        W  = self.W
        V  = self.V
        r  = dict()
        # power spectral density
        if 'psd' in args:
            psd = trace(np.abs(SS))
            r['psd'] = psd
        # degree of polarization
        if 'pol' in args or 'degpol' in args:
            pol = 1.5*(trace(np.matmul(SS,SS))/trace(SS)**2).real - 0.5
            r['pol'] = pol
        # planarity
        if 'pla' in args or 'planarity' in args:
            pla = 1 - np.sqrt(W[...,2]/(W[...,0]+1.0e-32))
            r['pla'] = pla
        # elipticity
        if 'elp' in args or 'elipticity' in args:
            elp = W[...,1]/W[...,0] * np.sign(SS[...,0,1].imag)
            r['elp'] = elp
        # n vector
        if 'n' in args or 'nvec' in args:
            kx  = np.sign(V[...,2,2])*V[...,2,0]
            ky  = np.sign(V[...,2,2])*V[...,2,1]
            kz  = np.sign(V[...,2,2])*V[...,2,2]
            tkb = np.rad2deg(np.abs(np.arctan2(np.sqrt(kx**2 + ky**2), kz)))
            pkb = np.rad2deg(np.arctan2(ky, kx))
            r['tkb'] = tkb
            r['pkb'] = pkb
        return r

    def analyze(self, acb, dcb):
        # polarization analysis via SVD for spectral matrix
        tt, ff, S = self.spectral_matrix(acb, dcb)
        self.tt = tt
        self.ff = ff
        self.S  = S
        self.svd()
        result = self.get('psd', 'pol', 'pla', 'elp', 'n')
        # apply smoothing
        keys = ('psd', 'pol', 'pla', 'elp', 'tkb', 'pkb')
        data = tuple([result[key] for key in keys])
        data = self.smooth_result(data)
        # power spectral density
        spec_psd = tseries.Spectrogram(tt, ff, data[0], log=True, logy=True)
        spec_psd.set(vmin=-6, vmax=+2, cmap=mms.cm_jet,
                     ylabel='Freq. [Hz]', clabel=r'PSD [nT $^2$ / Hz]')
        # degree of polarization
        spec_pol = tseries.Spectrogram(tt, ff, data[1], log=False, logy=True)
        spec_pol.set(vmin=0, vmax=+1, cmap=mms.cm_jet,
                     ylabel='Freq. [Hz]', clabel='Deg. Pol.')
        # planarity
        spec_pla = tseries.Spectrogram(tt, ff, data[2], log=False, logy=True)
        spec_pla.set(vmin=0, vmax=+1, cmap=mms.cm_jet,
                     ylabel='Freq. [Hz]', clabel='Planarity')
        # elipticity
        spec_elp = tseries.Spectrogram(tt, ff, data[3], log=False, logy=True)
        spec_elp.set(vmin=-1, vmax=+1, cmap=mms.cm_b2r,
                     ylabel='Freq. [Hz]', clabel='Elipticity')
        # theta
        spec_tkb = tseries.Spectrogram(tt, ff, data[4], log=False, logy=True)
        spec_tkb.set(vmin=0, vmax=90, cmap=mms.cm_jet,
                     ylabel='Freq. [Hz]', clabel=r'$\theta_{kb}$')
        # phi
        spec_pkb = tseries.Spectrogram(tt, ff, data[5], log=False, logy=True)
        spec_pkb.set(vmin=0, vmax=180, cmap=mms.cm_jet,
                     ylabel='Freq. [Hz]', clabel=r'$\phi_{kb}$')
        return spec_psd, spec_pol, spec_pla, spec_elp, spec_tkb, spec_pkb

    def smooth_result(self, xx):
        convolve = ndimage.filters.convolve1d
        ws  = self.wsmooth / self.wsmooth.sum()
        return tuple([convolve(x, ws, mode='nearest') for x in xx])


