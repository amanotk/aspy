# -*- coding: utf-8 -*-

""" Figures for plotly

"""

import re
import numpy as np
from numpy import ma
import pandas as pd

import plotly.graph_objects as go

from ..utils import *

# suffix
_axis_suffix_cb = '000'

_mpl_jet = \
[
    [ 0.0000, "rgb(  0,  0,127)"],
    [ 0.1100, "rgb(  0,  0,255)"],
    [ 0.1250, "rgb(  0,  0,254)"],
    [ 0.3400, "rgb(  0,219,255)"],
    [ 0.3500, "rgb(  0,229,246)"],
    [ 0.3750, "rgb( 20,255,226)"],
    [ 0.6400, "rgb(238,255,  8)"],
    [ 0.6500, "rgb(246,245,  0)"],
    [ 0.6600, "rgb(255,236,  0)"],
    [ 0.8900, "rgb(255, 18,  0)"],
    [ 0.9100, "rgb(231,  0,  0)"],
    [ 1.0000, "rgb(127,  0,  0)"],
]
_mpl_bwr = \
[
    [ 0.0000, "rgb(  0,  0,255)"],
    [ 0.5000, "rgb(255,255,255)"],
    [ 1.0000, "rgb(255,  0,  0)"],
]
_mpl_seismic = \
[
    [ 0.0000, "rgb(  0,  0, 76)"],
    [ 0.2500, "rgb(  0,  0,255)"],
    [ 0.5000, "rgb(255,255,255)"],
    [ 0.7500, "rgb(255,  0,  0)"],
    [ 1.0000, "rgb(127,  0,  0)"],
]


def _convert_color(color):
    colortable = {
        'r' : 'red',
        'g' : 'green',
        'b' : 'blue',
        'c' : 'cyan',
        'y' : 'yellow',
        'm' : 'magenta',
        'w' : 'white',
        'k' : 'black',
    }
    if not isinstance(color, str):
        return color

    if color in colortable:
        return colortable[color]
    elif re.fullmatch(r'^#([0-f][0-f][0-f]){1,2}', color):
        return color
    else:
        raise ValueError('unrecognized color format: %s' % (color))


def _get_colormap(cmap, table=True):
    cmaptable = {
        'jet'      : _mpl_jet,
        'bwr'      : _mpl_bwr,
        'seismic'  : _mpl_seismic,
    }
    if isinstance(cmap, list) and len(cmap) == 1:
        cmap = cmap[0]
    if table and isinstance(cmap, str) and cmap in cmaptable:
        cmap = cmaptable[cmap]
    return cmap


def _get_utcoffset():
    import dateutil.tz
    return dateutil.tz.tzlocal().utcoffset(0).total_seconds() * 1.0e+3


def _get_domain(figure, axis):
    return figure.layout[axis]['domain']


def _get_legend_label(x, y, text, **opts):
    r = go.layout.Annotation(x=x, y=y, text=text, showarrow=False,
                             xref='paper', yref='paper',
                             xanchor='left', yanchor='middle', **opts)
    return r


def _get_legend_line(x0, y0, x1, y1, **opts):
    r = go.layout.Shape(type='line', x0=x0, x1=x1, y0=y0, y1=y1,
                        xref='paper', yref='paper', **opts)
    return r


class BaseFigure(object):
    def __init__(self, data, figure, axes, **options):
        self.data   = data
        self.figure = figure
        self.axes   = axes
        self.setup_options(options)
        self.setup_default_axes()

    def setup_options(self, options):
        self.opt = options.copy()
        self.opt['webgl']   = self.opt.get('webgl', False)
        self.opt['xtime']   = self.opt.get('xtime', True)
        self.opt['numplot'] = self.opt.get('numplot', 0)
        self.opt['numaxes'] = self.opt.get('numaxes', 0)

    def setup_default_axes(self):
        xaxis = dict(
            linewidth=self.opt['line_width'],
            linecolor='#000',
            ticks='outside',
            mirror='allticks',
            showline=True,
            showticklabels=True,
            showgrid=False,
        )
        yaxis = dict(
            linewidth=self.opt['line_width'],
            linecolor='#000',
            ticks='outside',
            mirror='allticks',
            showline=True,
            showticklabels=True,
            showgrid=False,
        )
        if self.opt['xtime']:
            xaxis.update(self.get_date_options())

        ####
        # now update axes
        layout = {
            self.axes['xaxis'] : xaxis,
            self.axes['yaxis'] : yaxis,
        }
        self.figure.update_layout(**layout)

    def get_opt(self, key, val=None):
        return get_plot_option(self.data, key, val)

    def get_date_options(self):
        opt = {
            'type' : 'date',
            'tickformatstops' : [
                dict(dtickrange=[None, 1000], value="%M:%S.%L"),
                dict(dtickrange=[1000, 60000], value="%H:%M:%S"),
                dict(dtickrange=[60000, 3600000], value="%H:%M"),
                dict(dtickrange=[3600000, 86400000], value="%Y-%m-%d %H"),
                dict(dtickrange=[86400000, 604800000], value="%Y-%m-%d"),
                dict(dtickrange=[604800000, "M1"], value="%Y-%m"),
                dict(dtickrange=["M1", "M12"], value="%Y"),
                dict(dtickrange=["M12", None], value="%Y")
            ]
        }
        return opt

    def get_ylog_options(self, tickvals):
        tick1 = np.ceil(np.log10(tickvals))
        tick2 = np.floor(np.log10(tickvals))
        ticks = np.sort(np.unique(np.array(np.concatenate([tick1, tick2]),
                                           dtype=np.int32)))
        nt = len(ticks)
        tickvals = [0]*nt
        ticktext = [0]*nt
        for i in range(nt):
            tickvals[i] = 10.0**ticks[i]
            ticktext[i] = '10<sup>%+d</sup>' % (ticks[i])
        opt = {
            'type' : 'log',
            'tickvals' : tickvals,
            'ticktext' : ticktext,
        }
        return opt

    def get_xrange(self):
        return self.figure.layout[self.axes['xaxis']]['range']

    def get_yrange(self):
        return self.figure.layout[self.axes['yaxis']]['range']

    def set_xrange(self, xrange):
        layout = {
            self.axes['xaxis'] : dict(range=xrange),
        }
        self.figure.update_layout(**layout)

    def set_yrange(self, yrange):
        layout = {
            self.axes['yaxis'] : dict(range=yrange),
        }
        self.figure.update_layout(**layout)

    def add_legend(self, label, line):
        # for the first legend
        na   = self.opt['numaxes']
        if self.figure._legend[na] is None:
            self.figure._legend[na] = []

        nl = len(self.figure._legend[na])
        ts = self.opt['ticklength']/self.opt['width']
        fs = self.opt['fontsize']/self.opt['width']
        xx = _get_domain(self.figure, self.axes['xaxis'])[1] + 2.0*ts
        yy = _get_domain(self.figure, self.axes['yaxis'])[1] - 0.5*fs
        x0 = xx
        y0 = yy - nl*fs
        x1 = x0 + 2.0*fs
        y1 = y0
        x2 = x1 + 0.5*fs
        y2 = y1
        r1 = _get_legend_line(x0, y0, x1, y1, **line)
        r2 = _get_legend_label(x2, y2, **label)
        self.figure._legend[na].append(dict(line=r1, label=r2))
        # render
        line = list(self.figure.layout.shapes)
        text = list(self.figure.layout.annotations)
        line.append(r1)
        text.append(r2)
        self.figure.update_layout(annotations=text, shapes=line)

    def buildfigure(self):
        pass

    def update_axes(self):
        pass


class FigureLine(BaseFigure):
    def buildfigure(self):
        data = self.data

        # use WebGL version or not
        if self.opt['webgl']:
            scatter = go.Scattergl
        else:
            scatter = go.Scatter

        x = pd_to_datetime(data.time)

        # ensure 2D array
        if data.values.ndim == 1:
            y = data.values[:,np.newaxis]
        elif data.values.ndim == 2:
            y = data.values
        else:
            print('Error: input must be either 1D or 2D array')
            return None

        legend_names = self.get_opt('legend')
        layout = self.figure.layout
        legend = list()
        N = y.shape[1]
        for i in range(N):
            # line options
            lopt = dict(line_width=self.opt['line_width'])
            lc = self.get_opt('line_color')
            if lc is not None and len(lc) == N:
                lopt['line_color'] = _convert_color(lc[i])
            else:
                lopt['line_color'] = _convert_color('k')
            # legend
            opt = dict(xaxis=self.axes['x'],
                       yaxis=self.axes['y'],
                       showlegend=False)
            if legend_names is not None:
                label = {
                    'text'      : legend_names[i],
                    'font_size' : self.opt['fontsize'],
                }
                self.add_legend(label, lopt)
            # plot
            opt.update(lopt)
            plot = scatter(x=x, y=y[:,i], mode='lines', **opt)
            self.figure.add_trace(plot)

        # update axes
        self.update_axes()

    def update_axes(self):
        font = dict(titlefont_size=self.opt['fontsize'],
                    tickfont_size=self.opt['fontsize'])

        xaxis = dict(font)
        if self.get_opt('trange', None) is not None:
            xaxis['range'] = pd_to_datetime(self.get_opt('trange'))

        yaxis = dict(font)
        if self.get_opt('yrange', None) is not None:
            yaxis['range'] = self.get_opt('yrange')

        if self.opt['numplot'] == 0:
            yaxis['title_text'] = self.get_opt('ylabel')

        # now update axes
        layout = {
            self.axes['xaxis'] : xaxis,
            self.axes['yaxis'] : yaxis,
        }
        self.figure.update_layout(**layout)


class FigureSpec(BaseFigure):
    def setup_default_axes(self):
        # setup primary axes
        BaseFigure.setup_default_axes(self)

        # setup colorbar axes
        self.axes_cb = dict(
            x='x%d%s' % (self.axes['numaxes'], _axis_suffix_cb),
            y='y%d%s' % (self.axes['numaxes'], _axis_suffix_cb),
            xaxis='xaxis%d%s' % (self.axes['numaxes'], _axis_suffix_cb),
            yaxis='yaxis%d%s' % (self.axes['numaxes'], _axis_suffix_cb),
        )

        cb_size = self.opt['colorbar_size'] / self.opt['width']
        cb_sep  = self.opt['colorbar_sep']  / self.opt['width']
        xdom = _get_domain(self.figure, self.axes['xaxis'])
        ydom = _get_domain(self.figure, self.axes['yaxis'])
        xcb0 = xdom[1] + cb_sep
        xcb1 = cb_size + xcb0
        ycb0 = ydom[0]
        ycb1 = ydom[1]
        xaxis_cb = dict(
            domain=[xcb0, xcb1],
            range=[0, 1],
            tickvals=[],
            linewidth=self.opt['line_width'],
            linecolor='#000',
            mirror=True,
            showline=True,
            showticklabels=False,
            showgrid=False,
            fixedrange=True,
            anchor=self.axes_cb['y'],
        )
        yaxis_cb = dict(
            domain=[ycb0, ycb1],
            range=[0, 1],
            tickvals=[],
            linewidth=self.opt['line_width'],
            linecolor='#000',
            mirror=True,
            ticks='outside',
            side='right',
            showline=True,
            showticklabels=True,
            showgrid=False,
            fixedrange=True,
            anchor=self.axes_cb['x'],
        )

        # update layout
        layout_option = {
            self.axes_cb['xaxis'] : xaxis_cb,
            self.axes_cb['yaxis'] : yaxis_cb,
        }
        self.figure.update_layout(**layout_option)

    def buildfigure(self):
        data = self.data

        x = data.time.values
        t = pd_to_datetime(x)
        y = data.coords['spec_bins'].values
        z = data.values
        ylog = self.get_opt('ytype', 'linear') == 'log'
        zlog = self.get_opt('ztype', 'linear') == 'log'

        # prepare for data to be rasterized
        kwargs = {
            'ylog' : ylog,
            'zlog' : zlog,
        }
        self.raster_data = prepare_raster_spectrogram(x, y, z, **kwargs)

        # colormap
        cmap = _get_colormap(self.get_opt('colormap'), False)

        # xrange
        x_range = self.get_opt('trange')
        if x_range is None:
            x_range = data.attrs['xmin'], data.attrs['xmax']
        else:
            x_range = to_unixtime(x_range)

        # yrange
        y_range = self.get_opt('yrange')
        if y_range is None:
            y_range = data.attrs['ymin'], data.attrs['ymax']
        if ylog:
            y_range = np.log10(y_range[0]), np.log10(y_range[1])

        # zrange
        z_range = self.get_opt('zrange')
        if z_range is None:
            z_range = data.attrs['zmin'], data.attrs['zmax']
        if zlog:
            z_range = np.log10(z_range[0]), np.log10(z_range[1])

        # bounding box and width/height in pixels
        numaxes = self.opt['numaxes']
        bbox_x0 = self.opt['bbox_pixels']['x0'][numaxes]
        bbox_x1 = self.opt['bbox_pixels']['x1'][numaxes]
        bbox_y0 = self.opt['bbox_pixels']['y0'][numaxes]
        bbox_y1 = self.opt['bbox_pixels']['y1'][numaxes]
        width   = int(bbox_x1 - bbox_x0)
        height  = int(bbox_y1 - bbox_y0)

        # rasterize
        kwargs = {
            'x_range' : x_range,
            'y_range' : y_range,
            'z_range' : z_range,
            'cmap'    : cmap,
            'width'   : width,
            'height'  : height,
        }
        img = do_raster_spectrogram(self.raster_data, **kwargs)

        # show image
        t_range = pd_to_datetime(x_range)
        tmin = t_range[0]
        tmax = t_range[1]
        delt = (tmax - tmin).total_seconds() * 1.0e+3

        image_opt = {
            'source'  : img,
            'xref'    : self.axes['x'],
            'yref'    : self.axes['y'],
            'x'       : tmin,
            'sizex'   : delt,
            'y'       : y_range[0],
            'sizey'   : y_range[1] - y_range[0],
            'sizing'  : 'stretch',
            'opacity' : 1.0,
            'layer'   : 'below',
            'xanchor' : 'left',
            'yanchor' : 'bottom',
        }
        image = go.layout.Image(**image_opt)
        self.figure.add_layout_image(image)

        # update axes
        self.xlim = x_range

        if ylog:
            self.ylim = 10**y_range[0], 10**y_range[1]
        else:
            self.ylim = y_range

        if zlog:
            self.zlim = 10**z_range[0], 10**z_range[1]
        else:
            self.zlim = z_range

        self.update_axes()

        # colorbar
        colorbar_opt = {
            'zlabel' : self.get_opt('zlabel', ''),
            'zmin'   : self.zlim[0],
            'zmax'   : self.zlim[1],
            'zlog'   : zlog,
            'cmap'   : cmap,
        }
        self.set_colorbar(**colorbar_opt)

    def set_colorbar(self, zmin, zmax, zlabel, zlog, cmap):
        font = dict(titlefont_size=self.opt['fontsize'],
                    tickfont_size=self.opt['fontsize'])

        xaxis = dict()
        yaxis = dict(font)
        if zlog:
            yaxis['type'] = 'log'
            zmin = np.log10(zmin)
            zmax = np.log10(zmax)

        # rasterized colorbar image
        img_colorbar = get_raster_colorbar(cmap=cmap)
        image_opt = {
            'source'  : img_colorbar,
            'xref'    : self.axes_cb['x'],
            'yref'    : self.axes_cb['y'],
            'x'       : 0,
            'y'       : zmax,
            'sizex'   : 1,
            'sizey'   : zmax - zmin,
            'sizing'  : 'stretch',
            'opacity' : 1.0,
            'layer'   : 'below',
        }
        image = go.layout.Image(**image_opt)
        self.figure.add_layout_image(image)

        # label
        yaxis['title_text'] = zlabel

        # ticks
        if self.get_opt('colorbar_ticks', None) is None:
            if zlog:
                z0 = np.ceil(zmin)
                z1 = np.floor(zmax)

                # TODO: need a cleverer way
                ntick = int(np.rint(z1 - z0)) + 1
                ticks = np.linspace(z0, z1, ntick)

                tickvals = [0]*ntick
                ticktext = [0]*ntick
                for i in range(ntick):
                    tickvals[i] = 10.0**ticks[i]
                    ticktext[i] = '10<sup>%+d</sup>' % (int(ticks[i]))

                yaxis['tickvals'] = tickvals
                yaxis['ticktext'] = ticktext
                yaxis['range'] = [zmin, zmax]
            else:
                # leave it handled by plotly
                pass
        else:
            # ticks are provided
            opt = self.get_opt('colorbar_ticks')
            if 'tickvals' in opt and 'ticktext' in opt:
                tickvals = np.atleast_1d(opt['tickvals'])
                ticktext = np.atleast_1d(opt['ticktext'])
                # check
                if tickvals.shape == ticktext.shape and tickvals.ndim == 1:
                    yaxis['tickvals'] = tickvals
                    yaxis['ticktext'] = ticktext
                    yaxis['range'] = [zmin, zmax]
                else:
                    print('Error: tickvals and ticktext are not consistent')
            else:
                print('Error: tickvals or ticktext are not given')

        layout = {
            self.axes_cb['xaxis'] : xaxis,
            self.axes_cb['yaxis'] : yaxis,
        }
        self.figure.update_layout(**layout)

    def update_axes(self):
        font = dict(titlefont_size=self.opt['fontsize'],
                    tickfont_size=self.opt['fontsize'])

        xaxis = dict(font)
        if self.get_opt('trange', None) is not None:
            xaxis['range'] = pd_to_datetime(self.get_opt('trange'))

        yaxis = dict(font)
        if self.get_opt('ytype', 'linear') == 'log':
            # log scale in y
            if self.get_opt('yrange', None) is not None:
                yrange = self.get_opt('yrange')
                yrange = [np.log10(yrange[0]), np.log10(yrange[1])]
            else:
                yrange = [np.log10(self.ylim[0]), np.log10(self.ylim[1])]
            # ticks
            ylogmin = int(np.ceil(yrange[0]))
            ylogmax = int(np.floor(yrange[1]))
            ticks   = 10**np.arange(ylogmin, ylogmax+1)
            yaxis['range'] = yrange
            yaxis.update(self.get_ylog_options(ticks))
        else:
            # linear scale in y
            if self.get_opt('yrange', None) is not None:
                yaxis['range'] = self.get_opt('yrange')
            else:
                yaxis['range'] = [y.min(), y.max()]

        if self.opt['numplot'] == 0:
            yaxis['title_text'] = self.get_opt('ylabel')

        # now update axes
        layout = {
            self.axes['xaxis'] : xaxis,
            self.axes['yaxis'] : yaxis,
        }
        self.figure.update_layout(**layout)


class FigureAlt(BaseFigure):
    pass


class FigureMap(BaseFigure):
    pass
