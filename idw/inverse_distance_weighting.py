"""
prism - Inverse Distance Weighting interpolation
__date__ = '20240513'
__version__ = '1.0.0'
__author__ =
        'Andrea Libertino (andrea.libertino@cimafoundation.org',
__library__ = 'prism'
General command line:
### python inverse_distance_weight.py -settings_file "settings.json" -time "YYYY-MM-DD HH:MM"
Version(s):

20240513 (1.0.0) --> Beta release
"""
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Complete library
import os
import copy
import logging
from os.path import join
from datetime import datetime, timedelta
from argparse import ArgumentParser
import numpy as np
import xarray as xr
import pandas as pd
import json
import time
import fnmatch
import netrc

from prism.libs import libs_idw
from prism.libs.libs_generic_io import importDropsData, importWebDropsData, importTimeSeries, write_raster, read_file_tiff, read_point_data
# -------------------------------------------------------------------------------------
# Script Main
def main():

    # -------------------------------------------------------------------------------------
    # Version and algorithm information
    alg_name = 'prism - Inverse Distance Weighting interpolation'
    alg_version = '1.0.0'
    alg_release = '2024-05-13'
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Get algorithm settings
    alg_settings, alg_time = get_args()

    # Set algorithm settings
    data_settings = read_file_json(alg_settings)

    # Set algorithm logging
    os.makedirs(data_settings['data']['log']['folder'], exist_ok=True)
    set_logging(logger_file=join(data_settings['data']['log']['folder'], data_settings['data']['log']['filename']))
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Get algorithm settings
    settings_file, alg_time = get_args()
    dateRun = datetime.strptime(alg_time, "%Y-%m-%d %H:%M")

    startRun = dateRun - data_settings['data']['dynamic']['time']['time_observed_period'] * pd.Timedelta(
        data_settings['data']['dynamic']['time']['time_frequency'])
    endRun = dateRun + data_settings['data']['dynamic']['time']['time_forecast_period'] * pd.Timedelta(
        data_settings['data']['dynamic']['time']['time_frequency'])
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Info algorithm
    logging.info(' ============================================================================ ')
    logging.info(' ==> START ... ')
    logging.info(' ')

    # Time algorithm information
    start_time = time.time()

    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Check computation setting
    logging.info(' --> Check settings...')

    # Gauge data sources
    if not 'use_webdrops' in data_settings['algorithm']['flags']["sources"]:
        data_settings['algorithm']['flags']["sources"]['use_webdrops'] = False                  # Check for retrocompatibility
        logging.warning(' ----> Setting for WebDrops not found in the settings file. WebDrops can not be used!')
    computation_settings = [data_settings['algorithm']['flags']["sources"]['use_timeseries'], data_settings['algorithm']['flags']["sources"]['use_webdrops'], data_settings['algorithm']['flags']["sources"]['use_drops2'], data_settings['algorithm']['flags']["sources"]['use_point_data']]
    if len([x for x in computation_settings if x]) > 1 or len([x for x in computation_settings if x]) == 0:
        logging.error(' ----> ERROR! Please choose if use local data or download stations trough drops2 or webdrops!')
        raise ValueError("Data sources flags are mutually exclusive!")

    # Output format
    if data_settings['data']['outcome']['format'].lower() == 'netcdf' or data_settings['data']['outcome'][
        'format'].lower() == 'nc':
        format_out = 'nc'
    elif fnmatch.fnmatch(data_settings['data']['outcome']['format'], '*tif*'):
        format_out = 'GTiff'
    elif fnmatch.fnmatch(data_settings['data']['outcome']['format'], '*txt*'):
        format_out = 'AAIGrid'
    else:
        logging.error('ERROR! Unknown or unsupported output format! ')
        raise ValueError("Supported output formats are netcdf and GTiff")

    logging.info(' ---> Format for output : ' + format_out)

    # Final correlation
    exponent_idw = data_settings['algorithm']['settings']['exponent_idw']
    logging.info(' --> IDW exponent: ' + str(exponent_idw))

    logging.info(' --> Check settings...DONE')
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Import data for drops and time series setup
    if data_settings['algorithm']['flags']["sources"]['use_drops2']:
        logging.info(' --> Station data source: drops2 database')
        drops_settings = data_settings['data']['dynamic']['source_stations']['drops2']
        if not all([drops_settings['DropsUser'], drops_settings['DropsPwd']]):
            netrc_handle = netrc.netrc()
            try:
                drops_settings['DropsUser'], _, drops_settings['DropsPwd'] = netrc_handle.authenticators(drops_settings['DropsAddress'])
            except:
                logging.error(' --> Valid netrc authentication file not found in home directory! Generate it or provide user and password in the settings!')
                raise FileNotFoundError(
                    'Verify that your .netrc file exists in the home directory and that it includes proper credentials!')
        drops_settings["time_aggregation"] = data_settings['data']['dynamic']['time']['time_frequency']
        dfData, dfStations = importDropsData(drops_settings=drops_settings, start_time=dateRun - timedelta(hours=data_settings['data']['dynamic']['time']['time_observed_period']), end_time=dateRun, time_frequency= data_settings['data']['dynamic']['time']['time_frequency'])
    elif data_settings['algorithm']['flags']["sources"]['use_webdrops']:
        logging.info(' --> Station data source: webdrops database')
        drops_settings = data_settings['data']['dynamic']['source_stations']['webdrops']
        if not all([drops_settings['DropsUser'], drops_settings['DropsPwd']]):
            netrc_handle = netrc.netrc()
            try:
                drops_settings['DropsUser'], _, drops_settings['DropsPwd'] = netrc_handle.authenticators(drops_settings['DropsAddress'])
            except:
                logging.error(
                    ' --> Valid netrc authentication file not found in home directory! Generate it or provide user and password in the settings!')
                raise FileNotFoundError('Verify that your .netrc file exists in the home directory and that it includes proper credentials!')
        drops_settings["time_aggregation"] = data_settings['data']['dynamic']['time']['time_frequency']
        dfData, dfStations = importWebDropsData(webdrops_settings=drops_settings, start_time=dateRun - timedelta(
            hours=data_settings['data']['dynamic']['time']['time_observed_period']), end_time=dateRun,
                                             time_frequency=data_settings['data']['dynamic']['time']['time_frequency'])

    elif data_settings['algorithm']['flags']["sources"]['use_timeseries']:
        logging.info(' --> Station data source: station time series')
        dfData, dfStations = importTimeSeries(timeseries_settings=data_settings['data']['dynamic']['source_stations']['time_series'], start_time=startRun, end_time=dateRun, time_frequency= data_settings['data']['dynamic']['time']['time_frequency'])
    else:
        logging.info(' --> Station data source: station point files')
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Loop across time steps
    for timeNow in pd.date_range(start=startRun, end=endRun, closed='right', freq=data_settings['data']['dynamic']['time']['time_frequency']):
        logging.info(' ---> Computing time step ' + timeNow.strftime("%Y-%m-%d %H:%M:00"))

        # Compute time step file names
        file_out_time_step = os.path.join(data_settings['data']['outcome']['folder'], data_settings['data']['outcome']['filename'])
        ancillary_out_folder = data_settings['data']['ancillary']['folder']
        gridded_in_time_step = os.path.join(data_settings['data']['dynamic']['source_gridded']['folder'], data_settings['data']['dynamic']['source_gridded']['filename'])
        point_in_time_step = os.path.join(data_settings['data']['dynamic']['source_stations']['point_files']['folder'], data_settings['data']['dynamic']['source_stations']['point_files']['filename'])

        for i in data_settings['algorithm']['template']:
            file_out_time_step = file_out_time_step.replace("{" + i + "}", timeNow.strftime(data_settings['algorithm']['template'][i]))
            gridded_in_time_step = gridded_in_time_step.replace("{" + i + "}", timeNow.strftime(data_settings['algorithm']['template'][i]))
            ancillary_out_folder = ancillary_out_folder.replace("{" + i + "}", timeNow.strftime(data_settings['algorithm']['template'][i]))
            point_in_time_step = point_in_time_step.replace("{" + i + "}", timeNow.strftime(data_settings['algorithm']['template'][i]))

        if os.path.isfile(file_out_time_step) or os.path.isfile(file_out_time_step + '.gz'):
            if data_settings['algorithm']['flags']['overwrite_existing']:
                pass
            else:
                logging.info(' ---> Time step ' + timeNow.strftime("%Y-%m-%d %H:%M:00") + ' already exist, skipping...')
                continue

        # Make output dir
        os.makedirs(os.path.dirname(file_out_time_step), exist_ok=True)

        # Import grid
        basic_grid = read_file_tiff(data_settings['data']['static']['grid'], var_name='precip', time=[timeNow], \
                             coord_name_x='lon', coord_name_y='lat', dim_name_y='lat', dim_name_x='lon')
        gridded_in_time_step = data_settings['data']['static']['grid']

        grid_out = copy.deepcopy(basic_grid)
        # Import point gauge data for point_data setup
        try:
            if data_settings['algorithm']['flags']["sources"]['use_point_data']:
                logging.info(' ----> Load time step point data ...')
                if data_settings['algorithm']['flags']["sources"]['non_standard_tab_fields']:
                    fields_dict = data_settings['data']['dynamic']['source_stations']['point_files']['non_standard_tab_fields']
                    dfStations, data = read_point_data(point_in_time_step, st_code=fields_dict["station_code"], st_name=fields_dict["station_name"], st_lon=fields_dict["longitude"],
                                                       st_lat=fields_dict["latitude"], st_data=fields_dict["data"], sep=fields_dict["separator"], header=fields_dict["header"])
                else:
                    dfStations, data = read_point_data(point_in_time_step, st_code='code', st_name='name', st_lon='longitude', st_lat='latitude', st_data='data')
            else:
                data = dfData.loc[timeNow.strftime("%Y-%m-%d %H:%M:00")].values
            if len(data) == 0:
                raise ValueError
            elif len(data[~np.isnan(data)]) == 0:
                raise ValueError
        except (FileNotFoundError, ValueError) as err:
            if not data_settings['algorithm']['flags']['raise_error_if_no_station_available']:
                logging.warning(' ----> WARNING! No valid gauge data available for time step, skip')
                continue
            else:
                logging.error(' ----> ERROR! No valid data available for time step')
                raise err

        dfData[dfData<0] = np.nan

        if not data_settings['algorithm']['flags']["sources"]['use_point_data']:
            # Check if station data with nan values are present and, in the case, remove station from Station dataframe
            # (only if not point data are used, in this case, no further cleaning is needed)
            logging.info(' ----> Check null values for time step ...')
            avail_time_step = len(dfData.columns[~np.isnan(data)])
            logging.info(' ----> Available data for the time step ...' + str(avail_time_step))
            if avail_time_step <=1:
                logging.warning(' ----> WARNING! Not enought valid gauge data available for time step (at least 2 points required), skip')
                continue
            #dfStations_available = dfStations.loc[dfStations["name"]==dfData.columns[~np.isnan(data)]]    #mod_andrea
            # delete the rows of the dfStations dataframe if dfStations["name"] is in dfData.columns[~np.isnan(data)]
            dfStations_available = dfStations[dfStations["name"].isin(dfData.columns[~np.isnan(data)])]   #mod_andrea

            data = data[~np.isnan(data)]
        else:
            dfStations_available = copy.deepcopy(dfStations)


        dfStations_available.loc[:,"data"] = data


        os.makedirs(ancillary_out_folder, exist_ok=True)
        dfStations_available.to_csv(os.path.join(ancillary_out_folder, timeNow.strftime("%Y%m%d%H%M_point_data.csv")))
        logging.info(' ----> Saving point data for time step ...DONE')

        logging.info(' ----> Interpolating data ...')
        XY_obs_coords = np.vstack((dfStations_available.lon.array, dfStations_available.lat.array)).T
        z_arr = dfStations_available["data"].array
        # returns a function that is trained (the tree setup) for the interpolation on the grid
        idw_tree = libs_idw.tree(XY_obs_coords, z_arr)
        if data_settings['algorithm']['settings']['npoints_idw'] is None:
            numpoints_idw = len(z_arr)
        else:
            numpoints_idw = data_settings['algorithm']['settings']['npoints_idw']
        if data_settings['algorithm']['settings']['exponent_idw'] is None:
            exponent_idw = 2
        else:
            exponent_idw = data_settings['algorithm']['settings']['exponent_idw']
        x_y_grid_pairs = np.meshgrid(grid_out.lon.values, grid_out.lat.values)
        x_y_grid_pairs_list = np.reshape(x_y_grid_pairs, (2, -1)).T
        z = idw_tree(x_y_grid_pairs_list, exp_idw= exponent_idw, k=numpoints_idw).reshape((int(len(grid_out.lat.values)), int(len(grid_out.lon.values))))

        logging.info(' ----> Interpolating data ...DONE')

        logging.info(' ---> Saving outfile...')
        ok_out = copy.deepcopy(grid_out)
        try:
            ok_out.precip.values = z
        except:
            ok_out.precip.values[0,:,:] = z

        write_output(ok_out["precip"], ok_out, format_out, file_out_time_step, gridded_in_time_step)

        if data_settings['algorithm']['flags']['compress_output']:
            os.system('gzip -f ' + file_out_time_step)
        logging.info(' ---> Saving outfile...DONE')

        #Clear ancillary data if flag clear_ancillary is checked
        if data_settings['algorithm']['flags']['clear_ancillary']:
            logging.info(' ---> Clearing ancillary data ...')
            os.remove(os.path.join(ancillary_out_folder, timeNow.strftime("%Y%m%d%H%M_point_data.csv")))
            # if the ancillary folder is empty, remove it
            if not os.listdir(ancillary_out_folder):
                os.rmdir(ancillary_out_folder)
            logging.info(' ---> Clearing ancillary data ...DONE')
    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    # Info algorithm
    time_elapsed = round(time.time() - start_time, 1)

    logging.info(' ')
    logging.info(' ==> ' + alg_name + ' (Version: ' + alg_version + ' Release_Date: ' + alg_release + ')')
    logging.info(' ==> TIME ELAPSED: ' + str(time_elapsed) + ' seconds')
    logging.info(' ==> ... END')
    logging.info(' ==> Bye, Bye')
    logging.info(' ============================================================================ ')
    # ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Method to write output files
def write_output(array, dataarray, format, file_out_time_step, gridded_in_time_step):
    if format == 'nc':
        logging.info(' ---> Saving outfile in netcdf format ' + os.path.basename(file_out_time_step))
        dataarray.to_netcdf(file_out_time_step)
    else:
        grid = xr.open_rasterio(gridded_in_time_step)
        write_raster(array, grid, file_out_time_step, driver=format)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Method to read file json
def read_file_json(file_name):

    env_ws = {}
    for env_item, env_value in os.environ.items():
        env_ws[env_item] = env_value

    with open(file_name, "r") as file_handle:
        json_block = []
        for file_row in file_handle:

            for env_key, env_value in env_ws.items():
                env_tag = '$' + env_key
                if env_tag in file_row:
                    env_value = env_value.strip("'\\'")
                    file_row = file_row.replace(env_tag, env_value)
                    file_row = file_row.replace('//', '/')

            # Add the line to our JSON block
            json_block.append(file_row)

            # Check whether we closed our JSON block
            if file_row.startswith('}'):
                # Do something with the JSON dictionary
                json_dict = json.loads(''.join(json_block))
                # Start a new block
                json_block = []

    return json_dict
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Method to get script argument(s)
def get_args():
    parser_handle = ArgumentParser()
    parser_handle.add_argument('-settings_file', action="store", dest="alg_settings")
    parser_handle.add_argument('-time', action="store", dest="alg_time")
    parser_values = parser_handle.parse_args()

    if parser_values.alg_settings:
        alg_settings = parser_values.alg_settings
    else:
        alg_settings = 'configuration.json'

    if parser_values.alg_time:
        alg_time = parser_values.alg_time
    else:
        alg_time = None

    return alg_settings, alg_time
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Method to set logging information
def set_logging(logger_file='log.txt', logger_format=None):
    if logger_format is None:
        logger_format = '%(asctime)s %(name)-12s %(levelname)-8s ' \
                        '%(filename)s:[%(lineno)-6s - %(funcName)20s()] %(message)s'

    # Remove old logging file
    if os.path.exists(logger_file):
        os.remove(logger_file)

    # Set level of root debugger
    logging.root.setLevel(logging.INFO)

    # Open logging basic configuration
    logging.basicConfig(level=logging.INFO, format=logger_format, filename=logger_file, filemode='w')

    # Set logger handle
    logger_handle_1 = logging.FileHandler(logger_file, 'w')
    logger_handle_2 = logging.StreamHandler()
    # Set logger level
    logger_handle_1.setLevel(logging.INFO)
    logger_handle_2.setLevel(logging.INFO)
    # Set logger formatter
    logger_formatter = logging.Formatter(logger_format)
    logger_handle_1.setFormatter(logger_formatter)
    logger_handle_2.setFormatter(logger_formatter)
    # Add handle to logging
    logging.getLogger('').addHandler(logger_handle_1)
    logging.getLogger('').addHandler(logger_handle_2)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def distance_matrix(x0, y0, x1, y1):
    """ Make a distance matrix between pairwise observations.
    Note: from <http://stackoverflow.com/questions/1871536>
    """

    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    d0 = np.subtract.outer(obs[:, 0], interp[:, 0])
    d1 = np.subtract.outer(obs[:, 1], interp[:, 1])

    # calculate hypotenuse
    return np.hypot(d0, d1)

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def simple_idw(x, y, z, xi, yi, power=1):
    """ Simple inverse distance weighted (IDW) interpolation
    Weights are proportional to the inverse of the distance, so as the distance
    increases, the weights decrease rapidly.
    The rate at which the weights decrease is dependent on the value of power.
    As power increases, the weights for distant points decrease rapidly.
    """

    dist = distance_matrix(x, y, xi, yi)

    # In IDW, weights are 1 / distance
    weights = 1.0 / (dist + 1e-12) ** power

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    return np.dot(weights.T, z)

# Call script from external library
if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------