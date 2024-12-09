#!/usr/bin/env python
# -*- coding: utf-8 -*-

#==========================================================================
# Utility functions for downlaoding and processing data
#==========================================================================

# Third party imports
import numpy as np
from mtspec import mtspec

from obspy import Trace, Stream
from obspy.core import UTCDateTime
from obspy.signal.filter import envelope

from scipy.fft import next_fast_len
import scipy.fft as sf
from scipy.signal import hilbert
from datetime import datetime, timedelta
import re

from get_times import *


def trim_traces(sol_id,
                stream,
                hr_min=10.,
                hr_max=22.,
                entire_sol=False):

    # for afternoon: hr_min =10, hr_max=22
    #hr_min=0
    #hr_max=12

    # Trim
    tref, tend = get_sol_start_end_utc(int(sol_id), printing=False)
    tref   = UTCDateTime(tref)

    if entire_sol:
        tstart  = UTCDateTime(tref)
        tend    = UTCDateTime(tend)
    else:
        tstart  = UTCDateTime(tref + timedelta(0, hours=hr_min))
        tend    = UTCDateTime(tref + timedelta(0, hours=hr_max))

    stream.trim(tstart, tend)
    starttime=UTCDateTime(2019,1,1,0,0,0)

    for tr in stream:
        tr.stats.starttime  = starttime

    return stream, tref




# Functions
def preprocessing(user,
                  password,
                    stream,
                  fname_save,*,
                  type_st='VEL',
                  rotate=False,
                  bazi=None,
                  system='ZNE'):

    if system not in ['ZNE', 'UVW']:
        raise ValueError

    stream.detrend(type='demean')
    stream.taper(0.1)
    stream.detrend()

    info    = SEIS(user=user,
                   password=password)

    # Remove instrument response
    print('     Removing instrument response')
    pre_filt    = [0.0005, 0.001, 45, 50]
    print(f'        going to {type_st}')
    stream.remove_response(inventory=info.inventory,
                            pre_filt=pre_filt,
                            output=type_st,
                            water_level=None)

    # Rotate to ZNE system
    if system=='ZNE':
        print('     Rotating from UVW to ZNE')
        stream.rotate('->ZNE', components=['UVW'],
                      inventory=info.inventory)


    # Write stream to .mseed file
    stream.write(fname_save,
                 format='mseed')

    if rotate:
        stream.rotate(method='NE->RT', back_azimuth=bazi)

    return stream


def preprocessing_spectra(tr,
                          taper=True,
                          pad=False,
                          starttime=UTCDateTime(2019,1,1,0,0,0)):
    '''
    This function applies a quick preprocessing to the data
    Data can be tapered and padded to the next power of 2

    Args:
        tr (Obspy.Trace): an Obspy Trace object containing seismic data
        taper (bool): a boolean flag to taper the data
        pad (bool): a boolean flag to pad the data to the next power of 2
    Returns:
        tr (Obspy.Trace): a preprocessed Obspy Trace object containing seismic data
    '''
    if taper:
        tr.taper(max_percentage=0.2)

    if pad:
        data    = np.copy(tr.data)
        npts    = len(data)
        np2 = 2**np.ceil(np.log2(npts))

        # Compute the required zero padding length on each side
        padding_length  = np2 - npts
        left_padding    = int(padding_length // 2)
        right_padding   = int(padding_length - left_padding)

        data_pad    = np.pad(data,
                             pad_width=(left_padding, right_padding),
                             mode='constant')

        tr_pad      = Trace(data=data_pad,
                            header={'starttime': starttime,
                                    'delta': tr.stats.delta})
        return tr_pad

    return tr



def one_bit_normalization(input_data):
    '''
    Perform one-bit normalization on ObsPy Trace or Stream.

    This function applies one-bit normalization to seismic data in an ObsPy Trace
    or a Stream. One-bit normalization replaces all nonzero values with 1 and
    zeros with -1. The function can handle both single traces and streams of traces.

    Args:
        input_data (obspy.Trace or obspy.Stream): The input data to be normalized.

    Returns:
        obspy.Trace or obspy.Stream: Normalized data in the form of an ObsPy Trace
        or Stream, depending on the input type.

    Raises:
        ValueError: If the input data is neither an ObsPy Trace nor an ObsPy Stream.

    Example:
        # Normalize a single Trace
        trace = one_bit_normalization(trace_data)

        # Normalize a Stream
        normalized_stream = one_bit_normalization(stream_data)
    '''
    if isinstance(input_data, Trace):
        # One-bit normalization for a single trace
        trace_data = input_data.data
        normalized_data = np.sign(trace_data)
        normalized_data[trace_data == 0] = -1  # Set zeros to -1

    elif isinstance(input_data, Stream):
        # One-bit normalization for a stream
        normalized_stream = Stream()
        for trace in input_data:
            trace_data = trace.data
            normalized_data = np.sign(trace_data)
            normalized_data[trace_data == 0] = -1  # Set zeros to -1
            normalized_trace = Trace(data=normalized_data, header=trace.stats)
            normalized_stream.append(normalized_trace)

        return normalized_stream

    else:
        raise ValueError("Input data must be an ObsPy trace or stream.")

    return Trace(data=normalized_data, header=input_data.stats)



def spectral_whitening(tr,
                       freqmin=1.e-3,
                       freqmax=22.e-3,
                       delta=None,
                       returntime=False,
                       Nfft=None):
    '''
    Perform spectral whitening on seismic data in an ObsPy Trace object.
    Modified version from spectral whitening of MSNoise.

    This function applies spectral whitening to seismic data in an ObsPy Trace object.
    Spectral whitening aims to flatten the amplitude spectrum of the signal in a specified frequency range.

    Args:
        tr (obspy.Trace): ObsPy Trace containing seismic data to be whitened.
        freqmin (float, optional): Minimum frequency for whitening (default is 0.001 Hz).
        freqmax (float, optional): Maximum frequency for whitening (default is 0.022 Hz).
        delta (float, optional): Time sampling interval (delta) for the trace data. If not provided,
            it's automatically determined from the trace.
        Nfft (int, optional): Number of points to use in the Fast Fourier Transform (FFT) (default is None).
            If not provided, it's determined automatically using the `next_fast_len` function.

    Returns:
        obspy.Trace or numpy.ndarray: If 'returntime' is True, the whitened time-domain signal as a NumPy array.
        If 'returntime' is False, a new ObsPy Trace with whitened data.

    Raises:
        ValueError: If the input data is not a valid ObsPy Trace.

    Example:
        # Apply spectral whitening to a trace and return a new ObsPy Trace
        whitened_trace = whiten2(trace_data, freqmin=0.001, freqmax=0.02)

        # Apply spectral whitening to a trace and return the whitened time-domain signal
        whitened_data = whiten2(trace_data, freqmin=0.001, freqmax=0.02, returntime=True)
    '''

    data    = np.copy(tr.data)

    if not Nfft:
        Nfft    = next_fast_len(tr.stats.npts)
    if not delta:
        delta   = tr.stats.delta

    Napod = 100
    Nfft = int(Nfft)
    freqVec = sf.fftfreq(Nfft, d=delta)[:Nfft // 2]
    J = np.where((freqVec >= freqmin) & (freqVec <= freqmax))[0]
    low = J[0] - Napod
    if low <= 0:
        low = 1

    porte1 = J[0]
    porte2 = J[-1]
    high = J[-1] + Napod
    if high > Nfft / 2:
        high = int(Nfft // 2)

    FFTRawSign = sf.fft(data, Nfft)

    # Left tapering:
    FFTRawSign[0:low] *= 0
    FFTRawSign[low:porte1] = np.cos(
        np.linspace(np.pi / 2., np.pi, porte1 - low)) ** 2 * np.exp(
        1j * np.angle(FFTRawSign[low:porte1]))
    # Pass band:
    FFTRawSign[porte1:porte2] = np.exp(1j * np.angle(FFTRawSign[porte1:porte2]))
    # Right tapering:
    FFTRawSign[porte2:high] = np.cos(
        np.linspace(0., np.pi / 2., high - porte2)) ** 2 * np.exp(
        1j * np.angle(FFTRawSign[porte2:high]))
    FFTRawSign[high:Nfft + 1] *= 0

    # Hermitian symmetry (because the input is real)
    FFTRawSign[-(Nfft // 2) + 1:] = FFTRawSign[1:(Nfft // 2)].conjugate()[::-1]

    if returntime:
        return np.real(sf.ifft(FFTRawSign, Nfft))[:len(data)]

    # Create a new ObsPy Trace using the whitened data
    whitened_data = np.real(sf.ifft(FFTRawSign, Nfft))[:len(data)]
    tr2 = Trace(data=whitened_data)
    # Copy necessary metadata from the original trace 'tr' to 'tr2'
    tr2.stats = tr.stats

    return tr2



def check_nans(tr):
    '''
    Check for NaN (Not-a-Number) values in an ObsPy Trace's data and handle them.

    Args:
        tr (obspy.Trace): The ObsPy Trace object to check for NaN values.

    Returns:
        obspy.Trace: If NaN values are found, a new ObsPy Trace with NaN values replaced by zeros.
                     If no NaN values are found, the original ObsPy Trace is returned.

    Example:
        # Check for NaN values in an ObsPy Trace and replace them with zeros if found
        cleaned_trace = check_nans(trace_data)
    '''
    if np.any(np.isnan(tr.data)):
        tr_filled   = tr.copy()
        data    = tr.data
        data_filled = np.nan_to_num(data)
        tr_filled.data  = data_filled

        return tr_filled

    else:
        return tr



def compute_fft(tr):
    '''
    This functions computes the fft of an obspy trace

    Args:
        tr (Obspy.Trace) : an Obspy Trace object containing seismic data

    Returns:
        positive_frequencies (np.array): numpy array with the positive frequencies of the fft
        positive_amplitudes (np.array): numpy array with the positive amplitudes of the fft in mHz
    '''
    Nsamp   = tr.stats.npts
    amplitude_spectrum = np.fft.fft(tr.data)
    frequencies = np.fft.fftfreq(Nsamp,
                                 tr.stats.delta)

    positive_frequencies = frequencies[:Nsamp//2]*1.e3  # freqs in mHz
    positive_amplitudes = 2.0/Nsamp * np.abs(amplitude_spectrum[:Nsamp//2])

    return positive_frequencies, positive_amplitudes



def compute_spectrum_general(tr, *,
                             sampling_rate=None,
                             resample=False,
                             type_data='single',
                             input_data='acc',
                             db=False):

    if sampling_rate and resample:
        tr.resample(sampling_rate=2, window='hann')
    if not sampling_rate:
        sampling_rate = tr.stats.sampling_rate

    # Set the power based on the data type
    power = 1 if type_data == 'correlation' else 2

    delta = 1. / sampling_rate
    Nsamp = tr.stats.npts

    # Compute spectrum and frequencies
    spectrum    = np.fft.fft(tr.data)
    frequencies = np.fft.fftfreq(Nsamp, delta)

    positive_frequencies    = frequencies[:Nsamp // 2]
    accel_spectral_density  = (2.0 / (Nsamp * delta)) * np.abs(spectrum[:Nsamp // 2]) ** power

    # Reverse arrays for consistency
    freqs   = positive_frequencies[1:]
    accel_spectrum = accel_spectral_density[1:]
    ang_freqs   = 2.0 * np.pi * freqs

    # Modify the spectrum based on the input_data
    if input_data == 'vel':
        accel_spectrum = (ang_freqs ** 2) * accel_spectrum
    elif input_data == 'acc':
        pass  # No modification needed for acceleration input

    # Apply logarithmic scaling if required
    if db:
        accel_spectrum = 10 * np.log10(accel_spectrum)

    return accel_spectrum, freqs



# Try not to use the functions below for computing amplitude spectrum
def compute_amplitude_spectrum(tr):
    '''
    This function computes the amplitude spectrum of an Obspy Trace object in acceleration units.

    Args:
        tr (Obspy.Trace): An Obspy Trace object containing seismic data in acceleration units (m/s²).

    Returns:
        positive_frequencies (np.array): Numpy array with the positive frequencies of the amplitude spectrum in Hz.
        amplitude_spectrum (np.array): Numpy array with the amplitude spectrum values in m/s/√Hz.
    '''
    Nsamp = tr.stats.npts
    accel_spectrum = np.fft.fft(tr.data)
    frequencies = np.fft.fftfreq(Nsamp, tr.stats.delta)

    positive_frequencies = frequencies[:Nsamp // 2]

    # Compute acceleration spectral density
    accel_spectral_density = (2.0 / (Nsamp * tr.stats.delta)) * np.abs(accel_spectrum[:Nsamp // 2])**2

    # Take the square root to get the amplitude spectrum in m/s/√Hz
    amplitude_spectrum = np.sqrt(accel_spectral_density)

    return positive_frequencies*1.e3, amplitude_spectrum


def compute_acceleration_psd(tr):
    '''
    This function computes the power spectral density (PSD) of acceleration from an Obspy Trace object in velocity units.

    Args:
        tr (Obspy.Trace): An Obspy Trace object containing seismic data in velocity units (m/s).

    Returns:
        positive_frequencies (np.array): Numpy array with the positive frequencies of the acceleration PSD in Hz.
        accel_psd (np.array): Numpy array with the acceleration PSD values in m/s²/√Hz.
    '''
    Nsamp = tr.stats.npts
    velocity_spectrum = np.fft.fft(tr.data)
    frequencies = np.fft.fftfreq(Nsamp, tr.stats.delta)

    positive_frequencies = frequencies[:Nsamp // 2]

    # Compute acceleration spectral density by scaling velocity spectrum
    accel_spectral_density = (2.0 / (Nsamp * tr.stats.delta)) * np.abs(velocity_spectrum[:Nsamp // 2])**2 * (2 * np.pi * positive_frequencies)**2

    # Take the square root to get the acceleration amplitude spectrum in m/s²/√Hz
    accel_amplitude_spectrum = np.sqrt(accel_spectral_density)

    return positive_frequencies, accel_amplitude_spectrum



def compute_mtspec(tr,
                   time_bandwidth=1.5,
                   number_of_tapers=3):
    '''
    This functions computes the adaptive weighted multitaper spectrum (Priet et al., 2009)

    Args:
        tr (Obspy.Trace) : an Obspy Trace object containing seismic data
        time_bandwidth (float, optional): float Time-bandwidth product. Common values are 2, 3, 4 and numbers in between
        numer_of_tapers (int, optional): integer, optional Number of tapers to use

    Returns:
        frequencies (np.array): numpy array with the positive frequencies of the spectrum in mHz
        amplitudes (np.array): numpy array with the positive amplitudes of the spectrum
    '''
    amplitudes, frequencies = mtspec(data=tr,
                                    delta=tr.stats.delta,
                                    time_bandwidth=time_bandwidth,
                                    number_of_tapers=number_of_tapers)

    return frequencies*1.e3, amplitudes


def remove_events(events):
    '''
    This functions remove from the list of processed stream (one per sol)
    the ones that correspond to corrupted sols, i.e., saturation, large gltiches, gaps, etc.

    Args:
        events (list of str): a list of str, where each one correspond to the name of
        one file, where each file corresponds to one sol

    Returns:
        events_clean (list of str): a list of str, with the corrupted sol removed
    '''

    events_skip = ['73','76', '83', '85', '88', '89', '91', '94', '95', '96',
                   '100','182', '183',
                   '105', '116', '121', '126', '128', '129', '130', '132',
                   '152', '153', '154', '155', '160', '168', '171', '175',
                   '192', '199', '217', '231', '232', '235', '236', '240',
                   '248', '254', '255', '267','288', '289', '292', '293',
                   '295', '297', '299', '308', '311', '315', '318', '320',
                   '322', '325', '331', '346', '349', '351', '355', '366',
                   '367', '370', '373', '380', '381', '398', '407', '412',
                   '415', '434', '442', '450', '457', '458', '469', '470',
                   '472', '487', '489', '499', '803', '829', '852', '855',
                   '860', '891', '893', '894', '895', '897', '911', '1003',
                   '1064', '1082']

    events_clean= [event for event in events if event not in events_skip]

    return events_clean


def stack_data(data, stack_type='linear'):
    '''
    Modified from Obspy stack function to stack numpy arrays
    Stack data by first axis.

    :type stack_type: str or tuple
    :param stack_type: Type of stack, one of the following:
        ``'linear'``: average stack (default),
        ``('pw', order)``: phase weighted stack of given order
        (see [Schimmel1997]_, order 0 corresponds to linear stack),
        ``('root', order)``: root stack of given order
        (order 1 corresponds to linear stack).
    '''

    if stack_type == 'linear':
        stack   = np.mean(data, axis=0)

    elif stack_type[0] == 'pw':
        npts    = np.shape(data)[1]
        nfft    = next_fast_len(npts)
        anal_sig= hilbert(data, N=nfft)[:, :npts]
        norm_anal_sig   = anal_sig / np.abs(anal_sig)
        phase_stack     = np.abs(np.mean(norm_anal_sig, axis=0)) ** stack_type[1]
        stack = np.mean(data, axis=0) * phase_stack

    elif stack_type[0] == 'root':
        r = np.mean(np.sign(data) * np.abs(data)
                    ** (1 / stack_type[1]), axis=0)
        stack = np.sign(r) * np.abs(r) ** stack_type[1]

    else:
        raise ValueError('stack type is not valid.')

    return stack



def compute_fft_data(data,delta):
    '''
    This functions computes the fft of a numpy array

    Args:
        tr (Obspy.Trace) : an Obspy Trace object containing seismic data

    Returns:
        positive_frequencies (np.array): numpy array with the positive frequencies of the fft
        positive_amplitudes (np.array): numpy array with the positive amplitudes of the fft in mHz
    '''
    Nsamp   = len(data)
    amplitude_spectrum = np.fft.fft(data)
    frequencies = np.fft.fftfreq(Nsamp, delta)

    positive_frequencies = frequencies[:Nsamp//2]*1.e3  # freqs in mHz
    positive_amplitudes = 2.0/Nsamp * np.abs(amplitude_spectrum[:Nsamp//2])

    return positive_frequencies, positive_amplitudes**2


def compute_mtspec_data(data,
                        delta,
                        time_bandwidth=1.5,
                        number_of_tapers=3):
    '''
    This functions computes the afaptive weighted multitaper spectrum (Priet et al., 2009)

    Args:
        tr (Obspy.Trace) : an Obspy Trace object containing seismic data
        time_bandwidth (float, optional): float Time-bandwidth product. Common values are 2, 3, 4 and numbers in between
        numer_of_tapers (int, optional): integer, optional Number of tapers to use

    Returns:
        frequencies (np.array): numpy array with the positive frequencies of the spectrum in mHz
        amplitudes (np.array): numpy array with the positive amplitudes of the spectrum
    '''
    amplitudes, frequencies = mtspec(data=data,
                                    delta=delta,
                                    time_bandwidth=time_bandwidth,
                                    number_of_tapers=number_of_tapers)

    return frequencies*1.e3, amplitudes


def check_nans_data(data):
    '''
    Check for NaN (Not-a-Number) values in a numpy array and handle them

    Args:
        tr (obspy.Trace): The ObsPy Trace object to check for NaN values.

    Returns:
        obspy.Trace: If NaN values are found, a new ObsPy Trace with NaN values replaced by zeros.
                     If no NaN values are found, the original ObsPy Trace is returned.

    Example:
        # Check for NaN values in an ObsPy Trace and replace them with zeros if found
        cleaned_trace = check_nans(trace_data)
    '''
    if np.any(np.isnan(data)):
        data_filled = np.nan_to_num(data)

        return data_filled

    else:
        return data



def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def envelope_smooth(envelope_window_in_sec, tr, mode='valid'):
    tr_env = tr.copy()
    tr_env.data = envelope(tr_env.data)

    w = np.ones(int(envelope_window_in_sec / tr.stats.delta))
    w /= w.sum()
    tr_env.data = np.convolve(tr_env.data, w, mode=mode)

    return tr_env


def find_nearest(arr, val):
    '''
    find nearest value to val in arr
    return index and value
    '''
    idx = np.abs(arr-val).argmin()
    return arr[idx], idx


