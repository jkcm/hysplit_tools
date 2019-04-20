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
import config 
from time import sleep
import matplotlib.image as mping
from matplotlib.colors import LinearSegmentedColormap
from subprocess import check_call

# rc('font', **{'family': 'sans-serif',
#               'sans-serif': ['Helvetica'],
#               'size': 14})
# rc('text', usetex=True)

# ---!
# ---data grab/push methods
# ---!
def get_imagery(platform, datetime, 
                filetype="visst-pixel-netcdf", save_dir=None):
    """
    Download NASA SatCORPS geostationary imagery if not found on disk.
    Platform (str): which satellite. be GOES-W, GOES-E, Meteosat-8, Meteosat-9, HIMAWARI-8
    datetime (obj): datetime object representing time to download imagery
    filetype (str): type of SatCORPS product. can be altered to download gif or gridded instead.
    save_dir (str): location directory where to save files (one per hemisphere)
    returns (NH_file, SH_file): tuple containing strings of both NH and SH file locations on disk.
    """
    plat_dict = {"HIMAWARI-8": ['HIMWARI-FD', 'HM8V03'],
                 "Meteosat-8": ['msg_fd/met8-fd', 'MT8V04'], 
                 "Meteosat-10": ['msg_fd', 'MT10V03'],
                 "GOES-W": ['gw_fd', 'G15V03'],
                 "GOES-E": ['goes-east', 'G13V03']}
    if save_dir is None:
        save_dir = os.path.join(config.imagery_dir, platform)
        if not os.path.exists(save_dir):
                os.makedirs(save_dir)
    if filetype == "visst-pixel-netcdf":
        if platform == 'HIMAWARI-8':
            filename = 'HM8V03.0.HEM.{0:%Y%j}.{0:%H%M}.PX.06K.NC'.format(datetime)
        elif platform in ['Meteosat-8', 'Meteosat-10']:
            filename = '{1}.0.HEM.{0:%Y%j}.{0:%H%M}.PX.09K.NC'.format(datetime, plat_dict[platform][1])
        else:
            filename = '{1}.0.HEM.{0:%Y%j}.{0:%H%M}.PX.08K.NC'.format(datetime, plat_dict[platform][1])            
    url_base = "https://clouds.larc.nasa.gov/prod/"
    saved = []
    for hem in ['NH', 'SH']:
        url = hem.join(os.path.join(url_base, plat_dict[platform][0], filetype, 
                           "{:%Y/%m/%d}".format(datetime), filename).split('HEM'))
        save_name = os.path.join(save_dir, os.path.basename(url))
        saved.append(get_HTTP(url, save_name, maxtries=3))
    return tuple(saved)

    
def get_HTTP(url, savename, forcedownload=False, maxtries=10):
    """download file over HTTP given URL. Wrapper around urlllib.
    url (string): url (e.g. https://website.com/file) of file to be downloaded
    savename (string): filename on disk (e.g. /home/me/file.jpg) to save file
    forcedownload (bool): force overwrite of existing file
    maxtries (int): number of attempts before giving up
    
    returns (savename): location of file on disk if successful
    
    Throws Exception IOError if maxtries exeeded and HTTPError still detected"""
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
    savedir = os.path.join(config.HYSPLIT_source_dir, 'analysis')
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

def get_GDAS_data(date, addforecast=False):
    if addforecast:
        forecast_file, for_date = get_hysplit_forecast_files(date=date)
        appended_file, app_date = get_hysplit_appended_files(date=date)
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

    return (hyfile_list, ana_date)


def get_latest_hysplit_analysis(today=dt.datetime.utcnow()):
    x = get_hysplit_analysis(today)
    offset = 0
    while x is None:
        offset = offset + 1
        date = today - dt.timedelta(days=offset)
        x = get_hysplit_analysis(date)
    return x


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
    
        savedir = os.path.join(config.HYSPLIT_source_dir, 'forecast',
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
        savedir = os.path.join(config.HYSPLIT_source_dir, 'forecast', '{:%Y%m%d}'.format(date))
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


def write_control_file(start_time, coords, hyfile_list, hours, vertical_type, init_height,
                       tdumpdir=config.HYSPLIT_tdump_dir, tdump_prefix=''):
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

    fl = os.path.join(config.HYSPLIT_working_dir, 'CONTROL')
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


def run_HYSPLIT(log=None):
    """Run HYSPLIT. Make sure CONTROL file has already been written.
    returns True upon successful completion.
    raises CalledProcessError if not."""
    check_call(config.HYSPLIT_call, shell=False, stdout=log, cwd=config.HYSPLIT_working_dir)
    print("finished running HYSPLIT.")
    return True

"""Hey you! reading this code! Hello. Here's some synthpop for you. https://www.youtube.com/watch?v=KNHxwSp-6Og
"""

def read_tdump(tdump):
    """
    Read a tdump file as output by the HYSPLIT Trajectory Model
        Returns a pandas DataFrame object.
    """
    def parseFunc(y, m, d, H, M):
        return dt.datetime(int('20'+y), int(m), int(d), int(H), int(M))

    columns = ['tnum', 'gnum', 'y', 'm', 'd', 'H', 'M', 'fhour', 'age', 'lat',
               'lon', 'height', 'pres']

    tmp = pd.read_table(tdump, nrows=100, header=None)
    l = [len(i[0]) for i in tmp.values]
    skiprows = l.index(max(l))
    D = pd.read_table(tdump, names=columns,
                      skiprows=skiprows,
                      engine='python',
                      sep=r'\s+',
                      parse_dates={'dtime': ['y', 'm', 'd', 'H', 'M']},
                      date_parser=parseFunc,
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
    """Linearly interpolate between coordinates in coord_list to add intermediate points.
    Legacy naming from CSET, leaving here for laughs.
    """

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


def bmap(ax=None, latlon_range=None, proj='cyl', drawlines=True):
    """Build basemap onto ax, drawing some default lines."""
    if ax is None:
        fig, ax = plt.subplots()
    if latlon_range is None:
        latlon_range = {'lat': (-90, 90), 'lon': (-180, 180)}
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

def add_satellite_to_map(m, sat_file, variable='reflectance_vis', **plargs):
    """Add SatCORPS satellite data to a basemap object.
    Currently not working, as SatCORPS data is down.
    """
    meteosat_date = dt.datetime.strptime(os.path.basename(minnis_file)[-22:-10], '%Y%j.%H%M')

    with nc4.Dataset(sat_file, 'r') as img_nc:
        glats = img_nc.variables['latitude'][:].copy()#.filled(float('nan'))
        glons = (img_nc.variables['longitude'][:].copy()% 360)#.filled(float('nan'))
        gdata = img_nc.variables[variable][:].copy()

    box = config.latlon_range
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

    goes_name = 'Satellite({}) {:%Y-%m-%d %H%M}Z'.format(clean(variable), meteosat_date)
    return goes_name


def build_basemap_with_heights(latlon_range):
    fig = plt.figure()
    ax = fig.add_axes([0, 0.3, 1, 0.7])
    m = bmap(ax=ax, latlon_range=latlon_range)
    axh = fig.add_axes([0, 0, 1, 0.2])
    
    axh.grid(True)
    axh.set_xlabel('Trajectory Age (Hours)')
    axh.set_ylabel('Alt (km)')
    return fig, ax, m, axh


def add_tdump_to_map_with_heights(m, axh, tdump, highlight=None, fixed_color=None):

    def plot_single(t, m, axh, c, i=None, highlight=None):
        m.plot(t.lon.values, t.lat.values, c=c, latlon=True, label=t.tnum[0])
        m.plot(t.lon.values[::6], t.lat.values[::6], '.', c=c, latlon=True)
        m.plot(t.lon.values[0], t.lat.values[0], '*', c=c, latlon=True, ms=12)
        m.plot(t.lon.values[-1], t.lat.values[-1], 's', c=c, latlon=True, ms=8)
        if i is not None:
            x, y = m(t.lon.values[0]+.5, t.lat.values[0]+.5)
            m.ax.annotate(str(i), xy=(x, y), xytext=(x, y), xycoords='data',
                        textcoords='data', fontsize=6)
    
        axh.plot(t.age, t.height, '-', c=c)
        axh.plot(t.age[::6], t.height[::6], '.', c=c)
        axh.plot(t.age[0], t.height[0], '*', c=c)
        
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


def plot_tdump_with_heights(tdump, latlon_range, him_file=None, annotation='', axh_ylims=None):

    fig, ax, m, axh = build_basemap_with_heights(latlon_range)
    T = read_tdump(tdump)
    start = T.index[0].to_pydatetime()
    goesname= ''
    goes_offset = None
    if him_file is not None:
            goesname = add_satellite_to_map(m, him_file)        
            goes_date = dt.datetime.strptime(os.path.basename(him_file)[-22:-10], '%Y%j.%H%M')
            goes_offset = int((goes_date - start).total_seconds()/3600)
    tdump_name = add_tdump_to_map_with_heights(m=m, axh=axh, tdump=tdump, highlight=goes_offset)

    plotname = tdump_name+annotation+'\n'+goesname
    ax.set_title(plotname, y=1.08)

    if axh_ylims is not None:
        axh.set_ylim(axh_ylims)
    axh.set_yticks(axh.get_yticks()[::2])
    axh.set_yticklabels(['{:0.1f}'.format(i/1000.) for i in axh.get_yticks()])
    axh.set_xlim(left=T.age.min()-5, right=T.age.max()+5)
    axh.set_ylim(bottom=0)

    return fig


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
    m = bmap(ax=ax, latlon_range=latlon_range)

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
