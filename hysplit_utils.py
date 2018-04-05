# -*- coding: utf-8 -*-
"""
Utility functions for HYSPLIT trajectory running and plotting
Created on Wed Apr 4 2018

@author: Johannes Mohrmann
Functions write_control_file, read_tdump,
"""

import os
import sys
import glob
import re
import pandas as pd
import netCDF4 as nc4
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, rc
from urllib.request import urlopen
from urllib.error import HTTPError
from ftplib import FTP
from mpl_toolkits.basemap import Basemap
import parameters as params
from time import sleep
import matplotlib.image as mping
from matplotlib.colors import LinearSegmentedColormap


rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica'],
              'size': 14})
rc('text', usetex=True)

# ---!
# ---data grab/push methods
# ---!


def get_imagery(platform, datetime, 
                filetype="visst-pixel-netcdf", saveloc=None):
    """
    Platform can be GOES-W, GOES-E, Meteosat-8, Meteosat-9, HIMAWARI-8
    Product can be 
    
    filetype 
    """
    plat_dict = {"HIMAWARI-8": ['HIMWARI-FD', 'HM8V03'],
                 "Meteosat-8": ['msg_fd/met8-fd', 'MT8V04'], 
                 "Meteosat-10": ['msg_fd', 'MT10V03'],
                 "GOES-W": ['gw_fd', 'G15V03'],
                 "GOES-E": ['goes-east', 'G13V03']}
    if saveloc is None:
        saveloc = os.path.join('/home/disk/eos4/jkcm/Data/CLOGS/test/')
    if filetype == "visst-pixel-netcdf":
        if platform == 'HIMAWARI-8':
            filename = 'HM8V03.0.HEM.{0:%Y%j}.{0:%H%M}.PX.06K.NC'.format(datetime)
        elif platform in ['Meteosat-8', 'Meteosat-10']:
            filename = '{1}.0.HEM.{0:%Y%j}.{0:%H%M}.PX.09K.NC'.format(datetime, plat_dict[platform][1])
        else:
            if platform == 'GOES-E':
                datetime = datetime.replace(minute=45)
            filename = '{1}.0.HEM.{0:%Y%j}.{0:%H%M}.PX.08K.NC'.format(datetime, plat_dict[platform][1])            
#            f_pre = '{1}.0.HEM.{0:%Y%j}.{0:%H}'.format(datetime, plat_dict[platform][1])
#            filename_rex = re.compile(f_pre+'\d{2}.PX.08K.NC')

    url_base = "https://clouds.larc.nasa.gov/prod/"
    hems = ['NH', 'SH']
    saved = []
    for hem in hems:
        url = hem.join(os.path.join(url_base, plat_dict[platform][0], filetype, 
                           "{:%Y/%m/%d}".format(datetime), filename).split('HEM'))
        save_name = os.path.join(saveloc, os.path.basename(url))
        saved.append(get_HTTP(url, save_name, maxtries=3))
    return saved
    
def get_HTTP(url, savename, forcedownload=False, maxtries=10):
    if os.path.exists(savename) and not forcedownload:
        print('{} already downloaded, not downloading again'.format(os.path.basename(savename)))
        return savename
    tries = 0
    while(True):
        print('Looking for file {}'.format(os.path.basename(savename)))
        if tries > maxtries:
            raise IOError('Something went very wrong with HTTP download;' +
                     'perhaps website is down?. Try this url: ' + url)
        try:
            response = urlopen(url)
        except HTTPError:
            #            print('could not file file {} in location {}'.format(os.path.basename(url), os.path.dirname(url)))
            tries = tries + 1
            continue
        else:
            print('saving to ' + savename) 
            with open(savename, 'wb') as fp:
                while True:
                    chunk = response.read(16384)
                    if not chunk:
                        print('File successfully downloaded')
                        break
                    fp.write(chunk)
            break
    return savename


def get_HIMAWARI_files(date):
    HIMAWARI_file = ("http://rammb.cira.colostate.edu/ramsdis/online/images/hi_res/himawari-8/" +
    "full_disk_ahi_true_color/full_disk_ahi_true_color_{:%Y%m%d%H%M}00.jpg".format(date))
    floc = os.path.join(params.HIMAWARI_source, os.path.basename(HIMAWARI_file))

    if os.path.exists(floc):
        print('HIMAWARI for {:%Y%m%d%H%M} already downloaded.'.format(date))
        return floc
    tries = 0
    while(True):
        print('Looking for HIMAWARI file for date {:%Y%m%d%H%M}'.format(date))
        if tries > 20:
            sys.exit('Something went very wrong with get_HIMAWARI_files;' +
                     'perhaps colostate.edu is down?. Try this url: ' + HIMAWARI_file)
        try:
            response = urlopen(HIMAWARI_file)
        except HTTPError:
            # return to the loop
            print('could not find HIMAWARI file for date {:%Y%m%d%H%M}.jpg'.format(date))
            tries = tries + 1
            continue
        else:
            with open(floc, 'wb') as fp:
                while True:
                    chunk = response.read(16384)
                    if not chunk:
                        break
                    fp.write(chunk)
            break
    return floc
    

def get_nomads_gfs_files(date=None, hour=18):
    """
    Gets GFS forecast files from NOMADS server.
    CSET-specific: will grab all forecast files specified in
    params.met_offsets for the the forecast hour specified in hour on date
    provided. Default behavior is to look for the most recent forecast
    (looking at today, at 18Z). If no forecast is found, will look 6 hours
    earlier. save_dir specified in params.gfs_data_source
    """
    if hour not in [0, 6, 12, 18]:
        raise ValueError('hour must be multiple of 6!')
    if date is None:
        date = dt.datetime.utcnow()
    date_try = date.replace(hour=hour, minute=00)
    tries = 0

    while(True):
        print(('GFS: trying date {:%Y-%m-%d %H}Z'.format(date_try)))
        if tries > 20:
            sys.exit('Something went very wrong with get_nomads_gfs_files;' +
                     'perhaps NOMADS is down?')
        url_str = 'http://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?' +\
                  'file=gfs.t{:%H}z.pgrb2.0p25.anl&all_lev=on&all_var=on&s' +\
                  'ubregion=&leftlon=190&rightlon=250&toplat=60&bottomlat=' +\
                  '5&dir=%2Fgfs.{:%Y%m%d%H}'
        url_try = url_str.format(date_try, date_try)
        # check for the .anl file here
        savedir = os.path.join(params.gfs_data_source, 'forecast',
                               '{:%Y%m%d}'.format(date_try))
        if os.path.isfile(os.path.join(savedir, url_try[60:83]+'.grb2')):
            print((os.path.join(savedir, url_try[60:83]+'.grb2')))
            print("GFS: File already acquired, not downloading it again.")
            break

        try:
            response = urlopen(url_try)
        except HTTPError:
            # return to the loop
            print('GFS: remote file not found, trying 6 hours earlier')
            date_try = date_try - dt.timedelta(hours=6)
            tries = tries + 1
            continue
        else:
            print(('GFS: we found the most recent model output! {:%Y-%m-%d %H}Z'
                  .format(date_try)))
            savedir = os.path.join(params.gfs_data_source, 'forecast',
                                   '{:%Y%m%d}'.format(date_try))
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            floc = os.path.join(savedir, url_try[60:83]+'.grb2')
            with open(floc, 'wb') as fp:
                while True:
                    chunk = response.read(16384)
                    if not chunk:
                        break
                    fp.write(chunk)
            break

    forecastlist = params.met_offsets
    for f in forecastlist:
        print(('GFS: acquiring {} forecast...'.format(f)))
        sys.stdout.flush()
        floc = os.path.join(savedir, url_try[60:80]+f+'.grb2')
        if os.path.isfile(floc):
            print("GFS: file already acquired, not downloading it again.")
        else:
            url_f = url_try[:80]+f+url_try[83:]
            response = urlopen(url_f)
            with open(floc, 'wb') as fp:
                while True:
                    chunk = response.read(16384)
                    if not chunk:
                        break
                    fp.write(chunk)
            print('GFS: File successfully downloaded')
        sys.stdout.flush()
    return(savedir, date_try)


def get_GDAS_data(date, addforecast=True, force_nonewestanalysis=False):
    print(date)
    if addforecast:
        forecast_file, for_date = get_hysplit_forecast_files(date=date)
        appended_file, app_date = get_hysplit_appended_files(date=date)
    if force_nonewestanalysis:
        date = date - dt.timedelta(days=1)
    print(date)
    analysis_file, ana_date = get_latest_hysplit_analysis(date)
    analysis_prev, date_2 = get_hysplit_analysis(date=ana_date-dt.timedelta(days=1))
    analysis_prev2, date_3 = get_hysplit_analysis(date=ana_date-dt.timedelta(days=2))
    analysis_prev3, date_4 = get_hysplit_analysis(date=ana_date-dt.timedelta(days=3))
    analysis_prev4, date_5 = get_hysplit_analysis(date=ana_date-dt.timedelta(days=4))
    analysis_prev5, date_6 = get_hysplit_analysis(date=ana_date-dt.timedelta(days=5))
    analysis_prev6, date_7 = get_hysplit_analysis(date=ana_date-dt.timedelta(days=6))

    hyfile_list = [analysis_prev6, analysis_prev5, analysis_prev4, analysis_prev3, analysis_prev2, analysis_prev, analysis_file]
    if addforecast:
        hyfile_list.append(appended_file)
        hyfile_list.append(forecast_file)

    return ana_date, hyfile_list


def get_latest_hysplit_analysis(today=dt.datetime.utcnow()):
    x = get_hysplit_analysis(today)
    offset = 0
    while x is None:
        offset = offset + 1
        date = today - dt.timedelta(days=offset)
        x = get_hysplit_analysis(date)
    return x


def get_hysplit_analysis(date):
    """
    gets hysplit analysis file for day in date.
    if the file is already acquired, will not download it again.
    if the file does not exist yet, returns None.
    """
    ftp = FTP('arlftp.arlhq.noaa.gov')
    ftp.login()
    ftp.cwd('/archives/gdas0p5')
    rx = re.compile('{:%Y%m%d}_gdas0p5\Z'.format(date))
    files = sorted(filter(rx.match, ftp.nlst()))
    if len(files) == 0:
        print("ARL: No analysis available for {:%Y%m%d} yet...".format(date))
        return None
    newest = files[-1]
    savedir = os.path.join(params.HYSPLIT_source, 'analysis')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    print(("ARL: Attempting to find analysis file {} locally...".format(newest)))
    if os.path.isfile(os.path.join(savedir, newest)):
        print("ARL: File already acquired, not downloading it again.")
    else:
        print("ARL: File not found, will grab it from archives.")
        try:
            ftp.retrbinary("RETR " + newest,
                           open(os.path.join(savedir, newest), 'wb').write)
        except:
            print("ARL: Error in ftp transfer.")
            raise
        print('ARL: Analysis file successfully downloaded')

    savedfile = os.path.join(savedir, newest)
    print(('ARL: {}'.format(savedfile)))
    ret_date = dt.datetime.strptime(os.path.basename(savedfile)[:8], "%Y%m%d")
    return savedfile, ret_date


def get_hysplit_appended_files(date=None):
    """
    Gets most recent HYSPLIT appended files on date.
    Returns file location and initialization time (in the appended
    case that means the end of the file, so gfsa for 18Z on the 12th
    is relevant from 18Z on the 10th through the 12th, for instance)
    """
    f, d = get_hysplit_forecast_files(date, model='gfsa')
    return f, d


def get_hysplit_forecast_files(date=None, model='gfsf'):
    """
    Gets most recent HYSPLIT forecast files on date.
    Finds most recent file on ARL server. If it already exists on disk,
    does nothing and returns location on disk and initialization date.
    If it does not exist on disk, downloads and then returns the same.
    Arrival times:  00z gfsf - 10:07PM PDT (-1) = 05z
                    00z gfsa - 10:12PM PDT (-1) = 05z 
                    06z gfsf - 04:07AM PDT      = 11z
                    06z gfsa - 04:12AM PDT      = 11z
                    12z gfsf - 10:02AM PDT      = 17z
                    12z gfsa - 10:07AM PDT      = 17z
                    18z gfsf - 16:07PM PDT      = 23z
                    18z gfsa - 16:12PM PDT      = 23z
                    
    """
    def try_FTP_connect(ftpname):
        counter = 0
        while True:
            try:
                ftp = FTP(ftpname)
                return ftp
            except Exception as e:
                counter += 1
                sleep(1)
                if counter > 20:
                    raise e

    if date is None:
        date = dt.datetime.utcnow()

    try:
        ftp = try_FTP_connect('arlftp.arlhq.noaa.gov')
        ftp.login()
        ftp.cwd('/forecast/{:%Y%m%d/}'.format(date))
        rx = re.compile('hysplit.*.{}\Z'.format(model))
        files = list(filter(rx.match, ftp.nlst()))
        if len(files) == 0:  # too early in the day
            print(('ARL: no recent {} matches, looking at yesterday instead'.format(model)))
            date = date - dt.timedelta(days=1)
            ftp.cwd('/forecast/{:%Y%m%d/}'.format(date))
            files = list(filter(rx.match, ftp.nlst()))
        newest = files[-1]
    
        savedir = os.path.join(params.HYSPLIT_source, 'forecast',
                               '{:%Y%m%d}'.format(date))
        if not os.path.exists(savedir):
            os.makedirs(savedir)
    
        print(("ARL: Attempting to find {} for {:%Y-%m-%d}...".format(newest, date)))
        if os.path.isfile(os.path.join(savedir, newest)):
            print("ARL: File already acquired, not downloading it again.")
        else:
            print("ARL: File not found, will grab it from server.")
            try:
                ftp.retrbinary("RETR " + newest,
                               open(os.path.join(savedir, newest), 'wb').write)
            except:
                print("ARL Error in ftp transfer.")
                raise
            print('ARL: File successfully downloaded')
    
        inittime = int(newest.split('.')[-2][1:3])
        initdate = date.replace(hour=inittime, minute=00, second=00,
                                microsecond=00)
        savedfile = os.path.join(savedir, newest)
        print(("ARL: file saves as {}".format(savedfile)))
    except Exception as e:
        print("Caught Exception in FTP transfer, looking locally...({})".format(e))
        savedir = os.path.join(params.HYSPLIT_source, 'forecast', '{:%Y%m%d}'.format(date))
        if not os.path.exists(savedir):
            raise IOError("no local {} data found, FTP not available for {}".format(model, date))
        else:
            files = sorted(glob.glob(os.path.join(savedir, "hysplit.*.{}".format(model))))
            if len(files) == 0:
                raise IOError("no local {} data found, FTP not available for {}".format(model, date))
            else:
                savedfile = files[-1]
                print('found local forecast file: {}'.format(savedfile))
                inittime = int(savedfile.split('.')[-2][1:3])
                initdate = date.replace(hour=inittime, minute=00, second=00,
                                        microsecond=00)
    return(savedfile, initdate)

# ---!
# ---local file I/O methods
# ---!


def grib_2_netcdf(gribfile, ncfile, ncfn=None):
    """converts grib 2 netcdf file
    """
    if params.OS == 'windows':
        ifile = w_2_cygfile(gribfile)
        ofile = w_2_cygfile(ncfile)
        os.system(r'C:\cygwin64\bin\bash.exe --login -c "cdo -f nc copy {} {}"'
                  .format(ifile, ofile))
    elif params.OS == 'linux':
        os.system(r'cdo -f nc copy {} {}'.format(gribfile, ncfile))


def grib_folder_2_netcdf(gribfolder, ncfolder=None):

    if ncfolder is None:
        ncfolder = gribfolder

    g = glob.glob(os.path.join(gribfolder, '*.grb2'))

    for f in g:
        ncname = os.path.basename(f)[:-4]+'ncdf'
        print(('looking for '+ncname+'...'))
        if os.path.isfile(os.path.join(ncfolder, ncname)):
            print("netcdf already exists")
            sys.stdout.flush()
        else:
            print('not found; converting to netcdf...')
            sys.stdout.flush()
            grib_2_netcdf(f, os.path.join(ncfolder, ncname))
    return ncfolder


def w_2_cygfile(fname):
    """convert windows filename to cygwin filename
    """
    return('/cygdrive/' + fname.split('\\')[0][0].lower() + '/' +
           '/'.join(fname.split('\\')[1:]))


def write_control_file(start_time, coords, hyfile_list, hours, vertical_type, init_height,
                       tdumpdir=params.HYSPLIT_tdumpdir, tdump_prefix=''):
    """
    This file generates the CONTROL files used for running the trajectories.
    start_time - the datetime object of when the trajectory should start
    coords - list of decimal [lat, lon] pairs. N and E are positive.
    hyfile_list - list of HYSPLIT source files on which to run model
    hours- negative hours means backwards run
    vertical_type:
        0 'data' ie vertical velocity fields
        1 isobaric
        2 isentropic
        3 isopycnal (constant density)
        4 isohypsic (constant internal sigma coord)
        5 from velocity divergence
        6 something wacky to convert from msl to HYSPLIT's above ground level
        7 spatially averaged vertical velocity
    """

    fl = os.path.join(params.HYSPLIT_workdir, 'CONTROL')
    f = open(fl, 'w')

    f.write(start_time.strftime('%y %m %d %H\n'))
    f.writelines([str(len(coords)), '\n'])
    for j in coords:
        f.write('{} {} {}\n'.format(str(j[0]), str(j[1]), init_height))
    f.writelines([str(hours), '\n'])

    f.writelines([str(vertical_type), '\n', '10000.0\n'])

    f.write('{}\n'.format(len(hyfile_list)))
    for hyfile in hyfile_list:
        f.writelines([
            os.path.dirname(hyfile), os.sep, '\n',
            os.path.basename(hyfile), '\n'])

    filename = '{}tdump{}'.format(tdump_prefix, start_time.strftime('%Y%m%dH%H%M'))
    
    f.writelines([tdumpdir, os.sep, '\n', filename, '\n'])
    f.close()
    return os.path.join(tdumpdir, filename)


def read_tdump(tdump, icept=True):
    """
    Read a tdump file as output by the HYSPLIT Trajectory Model
        Returns a pandas DataFrame object.
    """

    def parseFunc(s):
        print(s)
        return dt.strptime('-'.join([i.zfill(2) for i in s.split()]),
                           '%y-%m-%d-%H-%M')

    def parseFunc_icept(y, m, d, H, M):
        return dt.datetime(int('20'+y), int(m), int(d), int(H), int(M))

    columns = ['tnum', 'gnum', 'y', 'm', 'd', 'H', 'M', 'fhour', 'age', 'lat',
               'lon', 'height', 'pres']

    tmp = pd.read_table(tdump, nrows=100, header=None)
    l = [len(i[0]) for i in tmp.values]
    skiprows = l.index(max(l))
    pF = parseFunc_icept if icept else parseFunc
    D = pd.read_table(tdump, names=columns,
                      skiprows=skiprows,
                      engine='python',
                      sep=r'\s*',
                      parse_dates={'dtime': ['y', 'm', 'd', 'H', 'M']},
                      date_parser=pF,
                      index_col='dtime')

    return D


# ---!
# ---sausage/trajectory methods
# ---!


def gridder(SW, NW, NE, SE, numlats=6, numlons=6):
    """each point is a [lat lon] corner of the desired area"""
    lat_starts = np.linspace(SW[0], NW[0], numlats)
    lon_starts = np.linspace(SW[1], SE[1], numlons)
    lat_ends = np.linspace(SE[0], NE[0], numlats)
    lon_ends = np.linspace(NW[1], NE[1], numlons)
    lat_weight = np.linspace(0., 1., numlats)
    lon_weight = np.linspace(0., 1., numlons)
    lat = (1. - lon_weight[:, None])*lat_starts[None, :] +\
        lon_weight[:, None]*lat_ends[None, :]
    lon = (1. - lat_weight[:, None])*lon_starts[None, :] +\
        lat_weight[:, None]*lon_ends[None, :]
    l = []
    for i in range(numlats):
        for j in range(numlons):
            l.append((lat[j, i], lon[i, j]))
    return(l)


def make_sausage(coord_list, inbetween=5):

    numpoints = 1+(len(coord_list)-1)*(inbetween+1)

    lats = [0] * numpoints
    lons = [0] * numpoints

    skip = inbetween+1
    for i in range(len(coord_list)-1):
        start = skip*i
        lats[start] = coord_list[i][0]
        lons[start] = coord_list[i][1]
        latspace = np.linspace(coord_list[i][0], coord_list[i+1][0], inbetween+2)
        lats[start:start+skip+1] = latspace
        lonspace = np.linspace(coord_list[i][1], coord_list[i+1][1], inbetween+2)
        lons[start:start+skip+1] = lonspace
    lats[-1] = coord_list[-1][0]
    lons[-1] = coord_list[-1][1]

    out = [(x, y) for x, y in zip(lats, lons)]
    return (out)


# ---!
# ---Plotting methods
# ---!

def add_HIM_netcdf_to_plot(ncfile, m_ax, latlon_range=params.latlon_range, var=None):
    
    if m_ax is None:
        lat_range = latlon_range['lat']
        lon_range = latlon_range['lon']

        m_ax = Basemap(llcrnrlon=lon_range[0], llcrnrlat=lat_range[0],
                       urcrnrlon=lon_range[1],  urcrnrlat=lat_range[1],
                       rsphere=(6378137.00, 6356752.3142),
                       projection='cyl', resolution='i')
    
    start_time = dt.datetime.strptime(os.path.basename(ncfile)[-22:-10], '%Y%j.%H%M')
    
#    if 
#    vis = True if (starttime >= 1700 or starttime <= 200) else False
    with nc4.Dataset(ncfile, 'r') as img_nc:
        glats = img_nc.variables['latitude'][:].copy()
        glons = img_nc.variables['longitude'][:].copy()
        if vis:
            gdata = img_nc.variables['gvar_ch1'][0, :].copy()
        else:
            gdata = img_nc.variables['gvar_ch2'][0, :].copy()

    goesname = '\n GOES({}) {:%Y-%m-%d %H%M}Z'.format('VIS' if vis else 'IR',
                                                      goes_date)


def add_HIMAWARI_to_plot(imfile, m_ax, latlon_range=params.latlon_range):
    
    if m_ax is None:
        lat_range = latlon_range['lat']
        lon_range = latlon_range['lon']

        m_ax = Basemap(llcrnrlon=lon_range[0], llcrnrlat=lat_range[0],
                       urcrnrlon=lon_range[1],  urcrnrlat=lat_range[1],
                       rsphere=(6378137.00, 6356752.3142),
                       projection='cyl', resolution='i')
    
    def rgb_to_24bit(rgb):
        return rgb[0]*65536 + rgb[1]*256 + rgb[2]
    
    # Read in image, remove black borders    
    img = mping.imread(imfile)
    ymask = (np.max(np.sum(img, axis=2)[2600:2900,:], axis=0))>50
    xmask = (np.max(np.sum(img, axis=2)[:, 2600:2900], axis=1))>50
    img_mask = img[np.ix_(xmask, ymask)]
    
#    imgplot = ax.imshow(img_mask)
    # render in greyscale
    img_grey = np.array(np.sum(img_mask, axis=2),dtype='float')

    #extract lats and lons using the m basemap; mask pixels outside globe
    m = Basemap(projection='geos', lon_0=140.7, satellite_height=35793*1000) #, fix_aspect=False, rsphere=(63781370, 63567523))
    xx, yy = np.meshgrid(np.linspace(0, m.urcrnry, img_mask.shape[1]), 
                         np.linspace(0, m.urcrnrx, img_mask.shape[0]))
    lonpix, latpix = m(xx, yy, inverse=True)
    latpix = np.flipud(latpix)
    badmask = lonpix>100000
    lonpix[badmask] = np.nan
    latpix[badmask] = np.nan
    img_grey[badmask] = np.nan

    lons_shift = (lonpix+360)%360             
    b_indx = min(np.argwhere(np.nanmax(latpix, axis=1) 
                                < latlon_range['lat'][0]).flatten())
    t_indx = max(np.argwhere(np.nanmin(latpix, axis=1) 
                                > latlon_range['lat'][1]).flatten())
    l_indx = max(np.argwhere(np.nanmax(lons_shift[t_indx:b_indx,:], axis=0)
                                < (latlon_range['lon'][0]+360)%360).flatten())
    r_indx = min(np.argwhere(np.nanmin(lons_shift[t_indx:b_indx,:], axis=0) 
                                > (latlon_range['lon'][1]+360)%360).flatten())
    chop = slice(t_indx, b_indx), slice(l_indx, r_indx)
    m_ax.pcolormesh(lons_shift[chop] ,latpix[chop], img_grey[chop], 
                    latlon=True, cmap=plt.cm.gray, vmin=0, vmax=700)
    "Garbage below is for trying to plot in color. TODO: try alpha colorbars again"
#    img_col = img_mask/255.
#    img_col[badmask] = np.nan
#    img_rgb = np.apply_along_axis(rgb_to_24bit, 2, img_col)       
#    colormap = [((i//65536)/256, ((i//256)%256)/256, (i%256)/256) for i in range(256**3)]
#    my_cmap = LinearSegmentedColormap.from_list('my_cmap', colormap, N=256**3)
#    
#    m2 = Basemap(projection='cyl', llcrnrlon=110, llcrnrlat=-65, urcrnrlon=180, urcrnrlat=-20,
#                 resolution='i')
#    m2.pcolormesh(lonpix[3000:, :4500] ,latpix[3000:, :4500], img_rgb[3000:, :4500], 
#                  latlon=True, cmap=my_cmap, vmin=0, vmax=256**3)
#    m2.drawparallels(np.arange(-90.,120.,10.))
#    m2.drawmeridians(np.arange(0.,420.,10.))
#    m2.drawcoastlines(color='g')


def bmap(ax=None, proj='cyl', drawlines=True):

    if ax is None:
        fig, ax = plt.subplots()
    latlon_range = params.latlon_range
    lat_range = latlon_range['lat']
    lon_range = latlon_range['lon']

    m = Basemap(llcrnrlon=lon_range[0], llcrnrlat=lat_range[0],
                urcrnrlon=lon_range[1],  urcrnrlat=lat_range[1],
                rsphere=(6378137.00, 6356752.3142),
                projection=proj, ax=ax, resolution='i')
    if drawlines:
        m.drawparallels(np.arange(-90., 99., 15.), labels=[1, 1, 0, 0])
        m.drawmeridians(np.arange(-180., 180., 20.), labels=[0, 0, 1, 0])
    m.drawcoastlines()

    return m


def plot_gridpoints(outfile):
    coords = gridder([23.25, -140], [32.25, -146.75], [41, -128], [32, -125],
                     numlats=5, numlons=6)
    coords = gridder([23.25, -140], [34.5, -148.4375], [43.25, -128.75],
                     [32, -125], numlats=6, numlons=6)

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    m = bmap(ax=ax, proj='merc', drawlines=False)

    m.drawgreatcircle(-121.3, 38.6, -156, 19.8, linestyle='--', c='black')
#    m.plot(-121.3, 38.6, 's', ms=8, c='black')
#    m.plot(-156, 19.8, '*', ms=12, c='black')
#    ax.set_title('Coordinates for gridpoints of trajectory start locations',
#                 y=1.08)

    colors = cm.rainbow(np.linspace(0, 1, len(coords)))
    for i, crd in enumerate(coords):
        m.plot(crd[1], crd[0], '*', c=colors[i], latlon=True, ms=12, label=i)
        x, y = m(crd[1]+.5, crd[0]+.5)
        ax.annotate(str(i), xy=(x, y), xytext=(x, y), xycoords='data',
                    textcoords='data', fontsize=6)

    handles, labels = ax.get_legend_handles_labels()
    labels = [str(i)+":"+str(X) for i, X in enumerate(coords)]
    ax.legend(handles, labels, fontsize=8, loc='upper left', ncol=1,
              numpoints=1)
    ax.patch.set_visible(False)
    fig.savefig(outfile, dpi=300, transparent=True, bbox_inches='tight',
                pad_inches=0)
    fig.clf()
    plt.close('all')

    return


def plot_sausage(sausage_tdump, outfile, goes_source, latlon_range):

    def plot_single(t, m=None, ax=None, c=None, highlight=None):

        m.plot(t.lon.values, t.lat.values, c=c, latlon=True, label=t.tnum[0])
        m.plot(t.lon.values[::6], t.lat.values[::6], '.', c=c, latlon=True)
        m.plot(t.lon.values[0], t.lat.values[0], '*', c=c, latlon=True, ms=12)
        m.plot(t.lon.values[-1], t.lat.values[-1], 's', c=c, latlon=True, ms=8)
        if highlight is None:
            pass
        elif (highlight == 0):
            m.plot(t.lon.values[0], t.lat.values[0], '*', c='black',
                   latlon=True, ms=12)  # start stars
        elif (highlight == t.age[-1]):
            m.plot(t.lon.values[-1], t.lat.values[-1], 's', c='black',
                   latlon=True, ms=8)  # end squares
        else:
            offset = np.where(t.age == highlight)[0][0]
            m.plot(t.lon.values[offset], t.lat.values[offset], '.', c='black',
                   latlon=True, ms=10)

        ax.plot(t.age, t.height, '-', c=c)
        ax.plot(t.age[::6], t.height[::6], '.', c=c)
        ax.plot(t.age[0], t.height[0], '*', c=c)
        ax.grid('on')

        return m, ax

    T = read_tdump(sausage_tdump)
    start = T.index.min().to_pydatetime().strftime('%Y-%m-%d %HZ')
    end = T.index.max().to_pydatetime().strftime('%Y-%m-%d %HZ')

    goes_date = dt.datetime.strptime(goes_source.split(os.sep)[-1],
                                     'g15.%Y%j.%H%M.nc')
    goes_delay = goes_date - T.index.min().to_pydatetime()
    highlight = int(round((goes_delay.total_seconds()/3600)/6)*6)

    if highlight > 48 or highlight < 0:
        highlight = None

    fig = plt.figure(figsize=(7, 8))
    ax = fig.add_axes([0.1, 0.3, 0.8, 0.6])

    m = bmap(ax=ax)
    axh = fig.add_axes([0.15, 0.1, 0.7, 0.15])

    t = T.groupby('tnum')

    goesname = add_goes_to_map(m, goes_source)

    colors = cm.rainbow(np.linspace(0, 1, len(list(t.groups.keys()))))

    for i, k in enumerate(t.groups.keys()):
            m, axh = plot_single(t.get_group(k), m=m, ax=axh, c=colors[i],
                                 highlight=highlight)

    m.drawgreatcircle(-121.3, 38.6, -156, 19.8, linestyle='--', c='black')
    m.plot(-121.3, 38.6, 's', ms=8, c='black')
    m.plot(-156, 19.8, '*', ms=12, c='black')
    hrs = np.ptp(T.age.values)
    start = T.index.min().to_pydatetime().strftime('%Y-%m-%d %HZ')
    end = T.index.max().to_pydatetime().strftime('%Y-%m-%d %HZ')
    plotname = '{:2.0f}-hour trajectories for flight plan from {} to {}'\
               .format(hrs, start, end) + goesname
    print(plotname)
    sys.stdout.flush()

    ax.set_title('{}'.format(plotname), y=1.08)
    axh.grid('on')
    axh.set_xlabel('Trajectory Age (Hours)')
    axh.set_ylabel('Alt (m)')
    axh.set_yticks(axh.get_yticks()[::2])
    axh.set_xlim(left=T.age.min()-5, right=T.age.max()+5)

    fig.savefig(outfile, dpi=300)
    fig.clf()
    plt.close('all')

    return


def plot_tdump_clear(tdump, outfile, latlon_range):

    def plot_single(t, m=None, c=None, i=None):
        m.plot(t.lon.values, t.lat.values, c=c, latlon=True)
        m.plot(t.lon.values[::6], t.lat.values[::6], '.', c=c, latlon=True)
        m.plot(t.lon.values[0], t.lat.values[0], '*', c=c, latlon=True, ms=12)
        m.plot(t.lon.values[-1], t.lat.values[-1], 's', c=c, latlon=True, ms=8)
        if i is not None:
            x, y = m(t.lon.values[0]+.5, t.lat.values[0]+.5)
            ax.annotate(str(i), xy=(x, y), xytext=(x, y), xycoords='data',
                        textcoords='data', fontsize=6)
        return m

    T = read_tdump(tdump)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    m = bmap(ax=ax)
    t = T.groupby('tnum')
    colors = cm.rainbow(np.linspace(0, 1, len(list(t.groups.keys()))))
    for i, k in enumerate(t.groups.keys()):
        m = plot_single(t.get_group(k), m=m, c=colors[i], i=i)
#    ax.patch.set_visible(False)
#    ax.text(-5.7, -15.9, 'H', ms=8, c='black')
#    ax.text(-14.3, -7.95, 'A', c='black')
    ax.annotate('H', xy=(-5.7, -15.9), xycoords='data', color='black')
    ax.annotate('A', xy=(-14.3, -7.95), xycoords='data', color='black')
    ax.annotate('W', xy=(14.5, -22.9), xycoords='data', color='black')
#    fig.savefig(outfile, dpi=150, transparent=True, bbox_inches='tight',
#                pad_inches=0)
    hrs = np.ptp(T.age.values)
    start = T.index.min().to_pydatetime().strftime('%Y-%m-%d %HZ')
    end = T.index.max().to_pydatetime().strftime('%Y-%m-%d %HZ')
    plotname = '{:2.0f}-hour trajectories from {} to {}'\
               .format(hrs, start, end)
    ax.set_title('{}'.format(plotname), y=1.08)
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    fig.clf()
    plt.close('all')

    return


def add_Minnis_to_map(m, minnis_file, variable='reflectance_vis', **plargs):
    meteosat_date = dt.datetime.strptime(os.path.basename(minnis_file)[-22:-10], '%Y%j.%H%M')

    with nc4.Dataset(minnis_file, 'r') as img_nc:
        glats = img_nc.variables['latitude'][:].copy()#.filled(float('nan'))
        glons = (img_nc.variables['longitude'][:].copy()% 360)#.filled(float('nan'))
        gdata = img_nc.variables[variable][:].copy()

    if False: ## this is the old method, trying to deprecate hard-coded vals and masking
        high = 700
        low = 100
        left = 100
        right = 1300
        glats = glats.filled(-100)[low:high, left:right]
        nan_mask = glons.mask
        left_mask = nan_mask.copy()
        right_mask = nan_mask.copy()
        left_mask[:,600:] = False
        right_mask[:,:600] = False
        glons = glons.filled(float('nan'))
        glons[left_mask] = 1
        glons[right_mask] = 359
        glons = glons[low:high, left:right]
        gdata = gdata.filled(float('nan'))[low:high, left:right]
    else:
        # glons[glons<=0] = glons[glons<=0] + 360
        box = params.latlon_range
        lat_upper = box['lat'][1]
        lat_lower = box['lat'][0]
        lon_upper = box['lon'][1]
        lon_lower = box['lon'][0]
        good_array = np.logical_and.reduce(
            [glons > lon_lower, glons < lon_upper, glats > lat_lower, glats < lat_upper])
        left_i = np.nonzero(np.sum(good_array, axis=0))[0][0]
        right_i = np.nonzero(np.sum(good_array, axis=0))[0][-1]
        top_i = np.nonzero(np.sum(good_array, axis=1))[0][0]
        bottom_i = np.nonzero(np.sum(good_array, axis=1))[0][-1]

        gdata = gdata[top_i:bottom_i, left_i:right_i]
        glats = glats[top_i:bottom_i, left_i:right_i].filled(float('nan'))
        glons = glons[top_i:bottom_i, left_i:right_i].filled(float('nan'))


    Csigma = np.sqrt(np.var(gdata))
    Cmean = np.mean(gdata)
    Cmin = Cmean - 2.*Csigma
    Cmax = Cmean + 2.*Csigma
    
    Cmin=0
    Cmax=1
    plargs['shading'] = 'gouraud'
#    plargs['alpha'] = 0.5
    plargs['linewidth'] = 0
    plargs['rasterized'] = True

    def clean(string):
        return '\_'.join(string.split('_'))
    
    pcol = m.pcolor(glons, glats, gdata, cmap=cm.gray, vmin=Cmin,
                     vmax=Cmax, latlon=True, **plargs)
    pcol.set_edgecolor('face')

    goes_name = 'HIMAWARI({}) {:%Y-%m-%d %H%M}Z'.format(clean(variable), meteosat_date)
    return goes_name


def build_basemap_with_heights():
    fig = plt.figure()
    ax = fig.add_axes([0, 0.3, 1, 0.6])
    m = bmap(ax=ax)
    axh = fig.add_axes([0.15, 0.1, 0.7, 0.15])
    
    axh.grid('on')
    axh.set_xlabel('Trajectory Age (Hours)')
    axh.set_ylabel('Alt (km)')
    
    m.drawgreatcircle(-121.3, 38.6, -156, 19.8, linestyle='--', c='w')
    m.plot(-121.3, 38.6, 's', ms=8, c='black', latlon=True)
    m.plot(-156, 19.8, '*', ms=12, c='black', latlon=True)
    return fig, ax, m, axh


def add_tdump_to_map_with_heights(m, axh, tdump, highlight=None, fixed_color=None):

    def plot_single(t, m, axh, c, i=None, highlight=None):
        m.plot(t.lon.values, t.lat.values, c=c, latlon=True, label=t.tnum[0])
        m.plot(t.lon.values[::6], t.lat.values[::6], '.', c=c, latlon=True)
        m.plot(t.lon.values[0], t.lat.values[0], '*', c=c, latlon=True, ms=12)
        m.plot(t.lon.values[-1], t.lat.values[-1], 's', c=c, latlon=True, ms=8)
        if i is not None:
            x, y = m(t.lon.values[0]+.5, t.lat.values[0]+.5)
            plt.annotate(str(i), xy=(x, y), xytext=(x, y), xycoords='data',
                        textcoords='data', fontsize=6)
    
        axh.plot(t.age, t.height, '-', c=c)
        axh.plot(t.age[::6], t.height[::6], '.', c=c)
        axh.plot(t.age[0], t.height[0], '*', c=c)
        axh.grid('on')
        
        if highlight is None:
            pass
        elif (highlight == 0):
            m.plot(t.lon.values[0], t.lat.values[0], '*', c='black',
                   latlon=True, ms=12)  # start stars
        elif (highlight == t.age[-1]):
            m.plot(t.lon.values[-1], t.lat.values[-1], 's', c='black',
                   latlon=True, ms=8)  # end squares
        else:
            try:
                offset = np.where(t.age == highlight)[0][0]
                m.plot(t.lon.values[offset], t.lat.values[offset], '.', c='black', latlon=True, ms=10)
                axh.plot(t.age[offset], t.height[offset], '.', c='black')
            except IndexError:
                print('nothing to highlight!')
                pass
    
    T = read_tdump(tdump)
    start = T.index.min().to_pydatetime()
    end = T.index.max().to_pydatetime()
    t = T.groupby('tnum')
    
    if fixed_color is None:
        colors = cm.rainbow(np.linspace(0, 1, len(t.groups.keys())))
    else:
        colors = [fixed_color]*len(t.groups.keys())

    for i, k in enumerate(t.groups.keys()):
        plot_single(t=t.get_group(k), m=m, axh=axh, c=colors[i], 
                        highlight=highlight, i=i)

    hrs = np.ptp(T.age.values)
    tdump_name = '{:2.0f}-hour trajectories ({} to {})'.format(hrs, start.strftime('%Y-%m-%d %HZ'), end.strftime('%Y-%m-%d %HZ'))
    return tdump_name

def plot_tdump_with_heights(tdump, him_file=None, annotation='', axh_ylims=None):

    fig, ax, m, axh = build_basemap_with_heights()
    T = read_tdump(tdump)
    start = T.index[0].to_pydatetime()
    goesname= ''
    goes_offset = None
    if him_file is not None:
            goesname = add_Minnis_to_map(m, him_file)        
            goes_date = dt.datetime.strptime(os.path.basename(him_file)[-22:-10], '%Y%j.%H%M')
            goes_offset = int((goes_date - start).total_seconds()/3600)
    print(axh)
    tdump_name = add_tdump_to_map_with_heights(m=m, axh=axh, tdump=tdump, highlight=goes_offset)

    plotname = tdump_name+annotation+'\n'+goesname
    ax.set_title(plotname, y=1.08)

    if axh_ylims is not None:
        axh.set_ylim(axh_ylims)
    axh.set_yticks(axh.get_yticks()[::2])
    axh.set_yticklabels(['{:0.2f}'.format(i/1000.) for i in axh.get_yticks()])
    axh.set_xlim(left=T.age.min()-5, right=T.age.max()+5)
    axh.set_ylim(bottom=0)

    return fig


#def _DEP_plot_tdump_with_heights(tdump, outfile, latlon_range, him_file=None, annotation='', axh_ylims=None):
#
#    def plot_single(t, m=None, ax=None, axh=None, c=None, i=None):
#        print(t.lon.values[::6], t.lat.values[::6])
#        m.plot(t.lon.values, t.lat.values, c=c, latlon=True, label=t.tnum[0])
#        m.plot(t.lon.values[::6], t.lat.values[::6], '.', c=c, latlon=True)
#        m.plot(t.lon.values[0], t.lat.values[0], '*', c=c, latlon=True, ms=12)
#        m.plot(t.lon.values[-1], t.lat.values[-1], 's', c=c, latlon=True, ms=8)
#        if i is not None:
#            x, y = m(t.lon.values[0]+.5, t.lat.values[0]+.5)
#            ax.annotate(str(i), xy=(x, y), xytext=(x, y), xycoords='data',
#                        textcoords='data', fontsize=6)
#
#        axh.plot(t.age, t.height, '-', c=c)
#        axh.plot(t.age[::6], t.height[::6], '.', c=c)
#        axh.plot(t.age[0], t.height[0], '*', c=c)
#        axh.grid('on')
#        return m, axh
#
#    T = read_tdump(tdump)
#    fig = plt.figure()
#    ax = fig.add_axes([0, 0.3, 1, 0.6])
#    m = bmap(ax=ax)
#    goesname= ''
#    if him_file is not None:
#        add_HIMAWARI_to_plot(him_file, m)
#        himdate = dt.datetime.strptime(him_file[-18:-6], '%Y%m%d%H%M')
#        goesname = ', HIMAWARI from {:%Y-%m-%d %HZ}'.format(himdate)
#        
#    axh = fig.add_axes([0.15, 0.1, 0.7, 0.15])
#    t = T.groupby('tnum')
#    colors = cm.rainbow(np.linspace(0, 1, len(list(t.groups.keys()))))
#    for i, k in enumerate(t.groups.keys()):
#        m, axh = plot_single(t.get_group(k), m=m, ax=ax, axh=axh, c=colors[i], i=i)
##    ax.annotate('H', xy=(-5.7, -15.9), xycoords='data', color='black')
##    ax.annotate('A', xy=(-14.3, -7.95), xycoords='data', color='b1lack')
##    ax.annotate('W', xy=(14.5, -22.9), xycoords='data', color='black')
#    hrs = np.ptp(T.age.values)
#
#    start = T.index.min().to_datetime().strftime('%Y-%m-%d %HZ')
#    end = T.index.max().to_datetime().strftime('%Y-%m-%d %HZ')
#    plotname = '{:2.0f}-hour trajectories ({} to {})'.format(hrs, start, end)+annotation+goesname
#    ax.set_title(plotname, y=1.08)
#    axh.grid('on')
#    axh.set_xlabel('Trajectory Age (Hours)')
#    axh.set_ylabel('Alt (km)')
#    if axh_ylims is not None:
#        axh.set_ylim(axh_ylims)
#    axh.set_yticks(axh.get_yticks()[::2])
#    axh.set_yticklabels(['{:0.2f}'.format(i/1000.) for i in axh.get_yticks()])
#    axh.set_xlim(left=T.age.min()-5, right=T.age.max()+5)
#
#    fig.savefig(outfile, dpi=300, bbox_inches='tight')
#    fig.clf()
#    plt.close('all')



def plot_tdump(tdump, outfile, goes_source, latlon_range):

    def plot_single(t, m=None, c=None, highlight=None, i=None):

        m.plot(t.lon.values, t.lat.values, c=c, latlon=True, label=t.tnum[0])
        m.plot(t.lon.values[::6], t.lat.values[::6], '.', c=c, latlon=True)
        m.plot(t.lon.values[0], t.lat.values[0], '*', c=c, latlon=True, ms=12)
        m.plot(t.lon.values[-1], t.lat.values[-1], 's', c=c, latlon=True, ms=8)
        if i is not None:
            plt.annotate(str(i), xy=(t.lon.values[0]+.5, t.lat.values[0]+.5))

        if highlight is None:
            pass
        elif (highlight == 0):
            m.plot(t.lon.values[0], t.lat.values[0], '*', c='black',
                   latlon=True, ms=12)  # start stars
        elif (highlight == t.age[-1]):
            m.plot(t.lon.values[-1], t.lat.values[-1], 's', c='black',
                   latlon=True, ms=8)  # end squares
        else:
            try:
                offset = np.where(t.age == highlight)[0][0]
            except IndexError:
                print ('nothing to highlight!')
                pass
            m.plot(t.lon.values[offset], t.lat.values[offset], '.', c='black',
                   latlon=True, ms=10)
        return m

    T = read_tdump(tdump)

    start = T.index.min().to_pydatetime()
    end = T.index.max().to_pydatetime()
    tlen = (end-start).total_seconds()/3600

    goes_date = dt.datetime.strptime(goes_source.split(os.sep)[-1],
                                     'g15.%Y%j.%H%M.nc')

    goes_offset = int((goes_date - start).total_seconds()/3600)
    highlight = goes_offset if 0 <= goes_offset <= tlen else None

    fig = plt.figure(figsize=(7, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    m = bmap(ax=ax)

    t = T.groupby('tnum')

    goesname = add_goes_to_map(m, goes_source)

    colors = cm.rainbow(np.linspace(0, 1, len(list(t.groups.keys()))))

    for i, k in enumerate(t.groups.keys()):
        m = plot_single(t.get_group(k), m=m, c=colors[i],
                        highlight=highlight, i=i)

    m.drawgreatcircle(-121.3, 38.6, -156, 19.8, linestyle='--', c='black')

    m.plot(-121.3, 38.6, 's', ms=8, c='black', latlon=True)
    m.plot(-156, 19.8, '*', ms=12, c='black', latlon=True)
    m.plot(-118.2, 33.77, 's', ms=8, c='red', latlon=True)
    hrs = np.ptp(T.age.values)

    plotname = '{:2.0f}-hour trajectories from {} to {}'\
        .format(hrs, start.strftime('%Y-%m-%d %HZ'),
                end.strftime('%Y-%m-%d %HZ'))+goesname

    ax.set_title('{}'.format(plotname), y=1.08)

    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    fig.clf()
    plt.close('all')
    return


def add_goes_to_map(m, goes_source, addname=True):
    starttime = int(goes_source[-7:-3])
    goes_date = dt.datetime.strptime(goes_source.split(os.sep)[-1],
                                     'g15.%Y%j.%H%M.nc')
    vis = True if (starttime >= 1700 or starttime <= 200) else False

    with nc4.Dataset(goes_source, 'r') as img_nc:
        glats = img_nc.variables['latitude'][:].copy()
        glons = img_nc.variables['longitude'][:].copy()
        if vis:
            gdata = img_nc.variables['gvar_ch1'][0, :].copy()
        else:
            gdata = img_nc.variables['gvar_ch2'][0, :].copy()

    goesname = '\n GOES({}) {:%Y-%m-%d %H%M}Z'.format('VIS' if vis else 'IR',
                                                      goes_date)
    if not addname:
        goesname = ''
    Csigma = np.sqrt(np.var(gdata))
    Cmean = np.mean(gdata)
    Cmin = Cmean - 2.*Csigma
    Cmax = Cmean + 2.*Csigma

    if vis:
        m.pcolormesh(glons, glats, gdata, cmap=cm.gray, vmin=Cmin,
                     vmax=Cmax, alpha=0.5, shading='gouraud', latlon=True)
    else:
        m.pcolormesh(glons, glats, gdata, cmap=cm.gray_r, vmin=Cmin,
                     vmax=Cmax, alpha=0.5, shading='gouraud', latlon=True)
    return goesname
