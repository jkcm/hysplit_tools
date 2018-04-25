# -*- coding: utf-8 -*-
"""
config file for HYSPLIT trajectory running and plotting
Created on Wed Apr 4 2018
@author: Johannes Mohrmann
"""

# should be either 'windows' or 'linux'
OS = 'linux'

# working directory for HYSPLIT to write CONTROL file. need write access
HYSPLIT_working_dir = '/home/disk/eos4/jkcm/Data/HYSPLIT/working' 
# write directory for HYSPLIT output files. need write access (can be anywhere)
HYSPLIT_tdump_dir = '/home/disk/eos4/jkcm/Data/HYSPLIT/tdump/demo'
# pathname for HYSPLIT executable. need execute access
HYSPLIT_call = '/home/disk/p/jkcm/hysplit/trunk/exec/hyts_std'
# write directory for saving plots
plot_dir = '~'
# read directory for HYSPLIT data. This shouldn't need changing unless you're downloading analysis I don't have
HYSPLIT_source_dir = '/home/disk/eos4/jkcm/Data/HYSPLIT/source'
# write directory for geostationary imagery
imagery_dir = '/home/disk/eos4/jkcm/Data/Minnis'