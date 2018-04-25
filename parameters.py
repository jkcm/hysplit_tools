# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:46:03 2015

@author: Hans
"""
"""
Plotting parameters for SOCRATES automated plot generation
"""

import socket
machine = socket.gethostname()

if machine == 'Hans-PC':
    raise Exception('not configured for Hans-PC!')
#    # should be either 'windows' or 'linux'
#    OS = 'windows'
#    # HYSPLIT-related parameters
#    HYSPLIT_workdir = 'E:\\Data\\HYSPLIT\\working'  # storing CONTROL file
#    HYSPLIT_tdumpdir = 'E:\\Data\\HYSPLIT\\tdump'  # storing tdumps
#    HYSPLIT_call = 'C:\hysplit4\exec\hyts_std.exe'  # to run HYSPLIT
#
#    # Data storage params
#    plot_loc = 'E:\\plots'  # basedir for storing plots
#    sausage_dir = 'E:\\Data\\CSET\\sausage'
#    gfs_data_source = 'E:\\Data\\GFS'
#    HYSPLIT_source = 'E:\\Data\\HYSPLIT\\source'  # storing HYSPlIT sourcefiles
#    GOES_source = 'E:\\Data\\GOES\\2015'
#    flight_plans = 'E:\\Data\\CSET\\flight_plans'
#    flight_data = 'E:\\Data\\CSET\\flight_data'
#    flight_trajs = 'E:\\Data\\CSET\\Trajectories'
#    HYSPLIT_runhgt = 500

if machine == 'fog':
    # should be either 'windows' or 'linux'
    OS = 'linux'
    # HYSPLIT-related parameters
    HYSPLIT_workdir = '/home/disk/eos4/jkcm/Data/HYSPLIT/working'  # storing CONTROL
    HYSPLIT_tdumpdir = '/home/disk/eos4/jkcm/Data/HYSPLIT/tdump/socrates'  # storing tdumps
    HYSPLIT_call = '/home/disk/p/jkcm/hysplit/trunk/exec/hyts_std'  # to run HYSPLIT
    data_dir = '/home/disk/eos4/jkcm/Data/SOCRATES/'

    # Data storage params
    plot_loc = '/home/disk/p/jkcm/public_html/SOCRATES_plots'  # basedir for storing plots
#    sausage_dir = '/home/disk/eos4/jkcm/CSET/sausage'
    gfs_data_source = '/home/disk/eos4/jkcm/Data/GFS'
    HYSPLIT_source = '/home/disk/eos4/jkcm/Data/HYSPLIT/source'
    HIMAWARI_source = '/home/disk/eos4/jkcm/Data/SOCRATES/HIMAWARI'
#    flight_plans = '/home/disk/eos4/jkcm/CSET/flight_plans'
#    flight_data = '/home/disk/eos4/jkcm/CSET/flight_data'
    HYSPLIT_runhgt = 500

# Plotting parameters
latlon_range = {'lat': (-65, -20), 'lon': (100, 180)}
met_offsets = ['f006', 'f012', 'f018', 'f024', 'f030', 'f048', 'f072']
