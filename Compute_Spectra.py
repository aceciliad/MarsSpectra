#!/usr/bin/env python
# -*- coding: utf-8 -*-

#==========================================================================
# Compute spectra
#==========================================================================

import numpy as np
from obspy.core import UTCDateTime
from obspy import Stream, read, Trace
import datetime, os, argparse, glob, ast
from phasecorr.phasecorr_seismic import acorr,xcorr

import warnings
warnings.filterwarnings("ignore")

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import ConnectionPatch
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from utils_processdata import *



class PLOT_SPECTRA():

    def __init__(self,
                 component='Z'):

        self.component  = component
        self.plot_data()

        return


    def plot_data(self,
                  data_path='Data'):

        ax, axn = self.set_fig()

        #----------------------------------------
        self.entire_day = True

        sols    = np.arange(235, 245, 1)

        self.plot_spectra(ax, axn, sols,
                          plot_single=True,
                          idx=0.1,
                          ampi=1.)



        plt.savefig(os.path.join(path_figs,
                                 f'Fig_Spectra.pdf'),
                    dpi=600)

        import ipdb; ipdb.set_trace()  # noqa
        return



    def plot_spectra(self, ax, ax2,
                     sols,
                     idx=0,
                     ampi=1,
                     plot_single=False):

        # make list of usable sols
        sols_str    = np.copy(sols).astype(str)


        # remove manually selected sols where problems were observed:
        sols_clean  = remove_events(sols_str)

        # initialize streams
        stream_stack= Stream()
        stream_stack_whiten = Stream()
        stream_stack_pcc    = Stream()

        count   = 0

        specs_arr= []
        freqs_arr= []

        sols_final  = []
        for sol_id in sols_clean:
            self.sol_id = sol_id
            print(f'Sol {self.sol_id}')

            try:
                tr_ac   = self.read_cc(whiten=False)
                tr_acw  = self.read_cc(whiten=True)
                tr_pcc  = self.read_pcc(whiten=True)
            except:
                try:
                    self.get_data()
                    stream_twi, tref = trim_traces(self.sol_id,
                                                   self.stream_twi,
                                                   entire_sol=self.entire_day)
                    self.tref   = tref
                    tr_twi  = stream_twi.select(component=self.component)[0]
                except:
                    print(f'     skip {self.sol_id}')
                    continue

                # without spectral whitening
                tr_ac   = self.compute_cc(tr_twi,
                                          whiten=False)
                # with spectral whitening
                tr_acw   = self.compute_cc(tr_twi,
                                           whiten=True)
                # PCC with spectral whitening
                tr_pcc  = self.compute_pcc(tr_twi,
                                           whiten=True)

            amps, freqs = compute_spectrum_general(tr_ac,
                                                   type_data='correlation',
                                                   input_data='vel',
                                                   db=False)

            if (count>10):
                if (tr_ac.stats.npts!=stream_stack[0].stats.npts):
                    print(f'     skip {self.sol_id}')
                    continue

            count   += 1


            sols_final.append(sol_id)
            # stream without spectral whitening
            stream_stack.append(tr_ac)
            stream_stack_whiten.append(tr_acw)
            stream_stack_pcc.append(tr_pcc)

            specs_arr.append(np.sqrt(amps))
            freqs_arr.append(freqs*1.e3)


            #if plot_single:
            #    ax.plot(freqs*1.e3, np.sqrt(amps),
            #            color='#a9a1b4', alpha=0.4, lw=0.4,
            #            zorder=15, rasterized=True)

        if plot_single:
            frequency_bins = np.linspace(1.,11, 50)  # Linearly spaced bins for frequency
            amplitude_bins = np.logspace(-11, -6, num=100)  # Logarithmically spaced bins for amplitude
            hist, xedges, yedges = np.histogram2d(np.array(freqs_arr).flatten(),
                                                  np.array(specs_arr).flatten(),
                                                  bins=[frequency_bins, amplitude_bins])

            midpoints = (xedges[1:] + xedges[:-1])/2, (yedges[1:] + yedges[:-1])/2

            #ax0.imshow(*midpoints, hist.T,#, aspect='auto',
            #           extent=[xedges[0], xedges[-1],
            #                   yedges[0], yedges[-1]],
            #           cmap='viridis')


            c_white_trans = mpl.colors.colorConverter.to_rgba('w')

            colors  = [cm.plasma(idx) for idx in np.linspace(0.15,.7, 5)]

            cmap_rb = mpl.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white_trans, *colors[:]],512)

            cnt = ax.contourf(*midpoints, hist.T, levels=200,
                         #extend=[xedges[0], xedges[-1],
                         #        yedges[0], yedges[-1]],
                         cmap=cmap_rb, alpha=1, zorder=1,
                         vmin=0,
                         vmax=hist.max()-hist.max()/20)
            for c in cnt.collections:
                c.set_edgecolor("face")


        self.sols_final = sols_final

        # Stack for upper plot
        if plot_single:
            stream_stack_copy   = stream_stack.copy()
            tr_stack    = stream_stack_copy.stack(stack_type='linear')[0]
            amps, freqs = compute_spectrum_general(tr_stack,
                                                   type_data='correlation',
                                                   input_data='vel',
                                                   db=False)
            ax.plot(freqs*1.e3, np.sqrt(amps),
                    lw=0.5, color='gray',
                    alpha=1, zorder=20)


        # Stack for lower plot:
        # other cases: ('pw', 1), ('root', 1.5)
        stack_type  = 'linear'
        amp_fact    = 7
        for stream, color, ypos, label_cc in zip([stream_stack_whiten, stream_stack_pcc],
                                                 ['#820300', '#154d92'],
                                                 [idx, idx+1],
                                                 ['CCGN', 'PCC']):

            stream_stack_copy   = stream.copy()
            tr_stack    = stream_stack_copy.stack(stack_type=stack_type)[0]

            freqs, amps = compute_mtspec(tr_stack,
                                         time_bandwidth=2.5,
                                         number_of_tapers=3)

            idx0    = np.argwhere(freqs>=ax2.get_xlim()[0])[0][0]
            idx1    = np.argwhere(freqs>ax2.get_xlim()[1])[0][0]
            freqs   = np.copy(freqs[idx0:idx1])
            amps    = np.copy(amps[idx0:idx1])

            amps_plot   = (amps-amps.min())/(amps.max()-amps.min())

            #if label_cc=='PCC':
            #    amps_plot *= ampi

            ax2.plot(freqs, (amps_plot)+(ypos),
                    lw=0.5, color=color,
                    alpha=0.9, zorder=20)

            ax2.text(freqs[-1],ypos, label_cc,
                     color=color, va='top', ha='right')

        return




    def get_data(self,
                 data_path='Data',
                 data_deglitched='preprocessed'):
        '''
        get deglitched data from seisglitch
        '''

        fname   = f'*_ZNE_{self.sol_id}.mseed'

        stream  = read(os.path.join(data_path,
                                    data_deglitched,
                                    fname))

        self.stream_twi = stream.copy()

        self.stream_twi.resample(sampling_rate=0.2, window='hann')

        return



    def define_path_cc(self,
                       whiten=False,
                        data_base='Data',
                        directory='CrossCorrelations_CC'):

        label   = f'CC_{self.sol_id}_{self.component}'

        if whiten:
            label   = label+'_SW'
        file_save   = label+'.mseed'

        path2save   =  os.path.join(data_base,
                                    directory,
                                    file_save)

        os.makedirs(os.path.join(data_base,
                                 directory),
                    exist_ok=True)

        return path2save


    def read_cc(self,*,
                whiten=False):

        path2save   = self.define_path_cc(whiten=whiten)

        try:
            tr_spe  = read(path2save)[0]
        except:
            raise FileNotFoundError

        return tr_spe


    def compute_cc(self,
                   tr_raw,*,
                   whiten=False):

        path2save   = self.define_path_cc(whiten=whiten)

        try:
            tr  = tr_raw.copy()
            # Process
            #tr2  = one_bit_normalization(tr2.copy())
            if whiten:
                tr = spectral_whitening(tr)
        except:
            print(' Error again')

        # compute cross-correlation
        tr_spe  = self.correlate_traces(tr, tr)
        tr_spe  = preprocessing_spectra(tr_spe.copy(),
                                        pad=True,
                                        taper=True)

        tr_spe.write(path2save, format='MSEED')

        return tr_spe


    def correlate_traces(self,tr1, tr2):

        # Get the data from the traces
        data1 = tr1.data
        data2 = tr2.data

        # Compute the cross-correlation using scipy's correlate function

        cross_corr = np.correlate(data1, data2, 'full')[len(data1)-1:]
        # normalize cross-correlation to follow CCGN
        term1   = np.sqrt(np.sum(data1**2))
        term2   = np.sqrt(np.sum(data2**2))

        #cross_corr  /= (term1*term2)

        # Determine the time shift values corresponding to each correlation value
        dt = tr1.stats.delta
        correlation_times = np.arange(-len(data1) + 1, len(data1)) * dt

        # Create a new trace to store the cross-correlation values
        tr_corr     = Trace(data=cross_corr,
                            header={'delta': dt})

        return tr_corr


    def define_path_pcc(self,
                        whiten=False,
                        data_base='Data',
                        directory='CrossCorrelations_PCC'):

        label   = f'PCC_{self.sol_id}_{self.component}'

        if whiten:
            label   = label+'_SW'
        file_save   = label+'.mseed'

        path2save   =  os.path.join(data_base,
                                    directory,
                                    file_save)

        os.makedirs(os.path.join(data_base,
                                 directory),
                    exist_ok=True)

        return path2save


    def read_pcc(self,*,
                 whiten=False):

        path2save   = self.define_path_pcc(whiten=whiten)

        try:
            tr_spe  = read(path2save)[0]
        except:
            raise FileNotFoundError

        return tr_spe


    def compute_pcc(self, tr_raw=None,
                    whiten=False):

        path2save   = self.define_path_pcc(whiten=whiten)

        tr  = tr_raw.copy()
        # process
        if whiten:
            tr = spectral_whitening(tr)

        # compute cross-correlation
        delta   = tr.stats.delta

        tr_spe  = acorr(tr).select(component=self.component)[0]
        tr_spe  = preprocessing_spectra(tr_spe.copy(),
                                        pad=True,
                                        taper=True)
        tr_spe.write(path2save,
                     format='MSEED')

        return tr_spe


    def set_fig(self):

        fig = plt.figure()
        cmi = 1/2.54
        fig.set_size_inches(13*cmi,15.*cmi)


        gs  = gridspec.GridSpec(ncols=1, nrows=4,
                                bottom=0.09, top=0.87,
                                left=0.15, right=0.98,
                                figure=fig,
                                wspace=0.1, hspace=.3)

        ax0 = fig.add_subplot(gs[:2,:])
        axn = fig.add_subplot(gs[2:4,:])


        ax0.set_ylim([1.e-10, .5e-7])
        ax0.set_xlim([1.5,8])
        ax0.set_yscale('log')
        ax0.tick_params(bottom=False, labelbottom=False)
        ax0.set_ylabel(r'Amplitude (m/s$^2$/$\sqrt{Hz}$)')
        #self.plot_selfnoise(ax, labels=True)
        ax0.spines[['right', 'top', 'bottom']].set_visible(False)

        axn.set_ylim([-0.2, 2.0])
        axn.set_xlim(ax0.get_xlim())
        axn.tick_params(left=False, labelleft=False)
        axn.spines[['right', 'top']].set_visible(False)

        axn.set_xlabel('Frequency (mHz)')# labelpad=-0.1)


        pos1    =  ax0.get_position()
        gs_cb   = gridspec.GridSpec(ncols=1, nrows=1,
                                bottom=pos1.y1+0.07,
                                top=pos1.y1+0.095,
                                left=pos1.x0+0.3,
                                right=pos1.x1-0.3,
                                figure=fig)

        cb_spe  = fig.add_subplot(gs_cb[:])

        norm    = mpl.colors.Normalize(vmin=0,vmax=1)

        c_white_trans = mpl.colors.colorConverter.to_rgba('white')

        colors  = [cm.plasma(idx) for idx in np.linspace(0.05,.7, 5)]

        cmap_rb = mpl.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white_trans, *colors[:]],512)

        cb = mpl.colorbar.ColorbarBase(cb_spe,
                                       orientation='horizontal',
                                       cmap=cmap_rb,
                                       norm=norm)
        cb.set_label('Probability density')
        cb.ax.xaxis.set_major_locator(MultipleLocator(0.5))
        cb.ax.xaxis.set_label_position('top')

        return ax0, axn


#--------------------------------------------------------------------------
if __name__=='__main__':

    # Global variables
    dir_figures = 'Figures'
    path_figs   = os.path.join(dir_figures)
    os.makedirs(path_figs, exist_ok=True)

    PLOT_SPECTRA()



