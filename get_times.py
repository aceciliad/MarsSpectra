#!/usr/bin/env python
# -*- coding: utf-8 -*-

#==========================================================================
# Given a Martian sol (day after landing of InSight), obtain the time
# stamps that define the start and end of the sol, useful to download
# data from IRIS
#==========================================================================

import numpy as np
import os, argparse
import datetime
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, read, read_inventory

import warnings
warnings.filterwarnings("ignore")

def arguments():
    '''
    arguments
    '''
    ap = argparse.ArgumentParser(description='Plot event')

    ap.add_argument('--sol', type=str, dest='sol_data', help='id data', required=True)

    return ap.parse_args()


def read_timestamp(id_data,
                   data_path='Data'):
    '''
    time stamps for downloading seismic data
    '''

    os.makedirs(data_path, exist_ok=True)

    tref,_   = get_sol_start_end_utc(int(id_data))

    tref    = UTCDateTime(tref)

    np.save(os.path.join(data_path,
                         f'ReferenceTime_{id_data}'),
            tref)

    # TO READ THIS TIME FROM OTHER SCRIPTS YOU CAN DO:
    tref    = np.load(os.path.join(data_path,
                                   f'ReferenceTime_{id_data}.npy'),
                      allow_pickle=True).item()


    return


def get_sol_start_end_utc(sol, printing=True):
    #Mars Local Mean Solar Time (LMST) offset from UTC in hours
    mars_offset = 0.0  # Replace with the actual offset value

    # Length of a Martian sol (day) in seconds
    sol_length_seconds = 88775.244

    # UTC start time of Sol 0
    sol0_utc_datetime = datetime.datetime(2018, 11, 26, 5, 10, 50, 335080)

    # Calculate the start time of the given sol in UTC
    sol_start_utc_datetime = sol0_utc_datetime + datetime.timedelta(seconds=int(sol)*sol_length_seconds)

    # Calculate the end time of the given sol in UTC
    sol_end_utc_datetime = sol_start_utc_datetime + datetime.timedelta(seconds=sol_length_seconds)

    if printing:
        print(f'{UTCDateTime(sol_start_utc_datetime)}--{UTCDateTime(sol_end_utc_datetime)}')
    return UTCDateTime(sol_start_utc_datetime), UTCDateTime(sol_end_utc_datetime)


if __name__=='__main__':

    results = arguments()
    id_data = results.sol_data

    # To download raw waveforms and get event info:
    read_timestamp(id_data)





