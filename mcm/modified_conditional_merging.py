"""
prism - Modified Conditional Merging with GRISO
__date__ = '20230223'
__version__ = '2.5.5'
__author__ =
        'Flavio Pignone (flavio.pignone@cimafoundation.org',
        'Andrea Libertino (andrea.libertino@cimafoundation.org',
        'Fabio Delogu (fabio.delogu@cimafoundation.org',
__library__ = 'prism'
General command line:
### python hyde_data_dynamic_modified_conditional_merging.py -settings_file "settings.json" -time "YYYY-MM-DD HH:MM"
Version(s):
20230223 (2.5.5) --> Added support for WebDrops data download
20221103 (2.5.4) --> Fixed bug for backup griso
20220726 (2.5.3) --> Added griso backup if gridded data not available
                     Fixed bug for sub-hourly drops2 aggregation
20220704 (2.5.2) --> Added support to sub-hourly merging
                     Moved to prism repository
20220302 (2.5.1) --> Fixed management of absence of point measurements
20211026 (2.5.0) --> Added theoretical kernel estimation
                     Bug fixes, major system optimizations
20210602 (2.2.0) --> Added support for AAIGrid inputs/outputs
20210602 (2.1.0) --> Added beta support for radar rainfall products
                     Implemented tif inputs, implemented not-standard point files input.
                     Add netrc support for drops2. Bug fixes.
20210425 (2.0.0) --> Dynamic radius for GRISO implemented
                     Added support to point rain files (for FloodProofs compatibility)
                     Script structure fully revised and bug fixes
20210312 (1.5.0) --> Geotiff output implementation, script structure updates, various bug fixes and improvements
20201209 (1.2.0) --> Integrated with local station data configuration. Settings file implemented.
20200716 (1.1.0) --> Integrated with drops2 libraries. Updates and bug fixes.
20200326 (1.0.0) --> Beta release for FloodProofs Bolivia
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
import matplotlib.pyplot as plt
import fnmatch
import netrc
import rasterio as rio

from prism.libs.libs_model_griso_exec import GrisoCorrel, GrisoInterpola, GrisoPreproc
from prism.libs.libs_generic_io import importDropsData, importWebDropsData, importTimeSeries, check_and_write_dataarray, write_raster, read_file_tiff, read_point_data
# -------------------------------------------------------------------------------------
# Script Main
def main():

    # -------------------------------------------------------------------------------------
    # Version and algorithm information
    alg_name = 'prism - Modified Conditional Merging '
    alg_version = '2.5.5'
    alg_release = '2023-02-23'
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

    try:
        griso_only = data_settings['algorithm']['flags']['perform_griso_only']
    except:
        griso_only = False
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

    # Griso correlation type
    computation_settings = [data_settings['algorithm']['flags']["mcm"]['fixed_correlation'], data_settings['algorithm']['flags']["mcm"]['dynamic_correlation']]
    if len([x for x in computation_settings if x]) > 1 or len([x for x in computation_settings if x]) == 0:
        logging.error(' ----> ERROR! Please choose if use fixed or dynamic correlation!')
        raise ValueError("Correlation type settings are mutually exclusive!")

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

    # Debug mode
    try:
        debug_mode = data_settings['algorithm']['flags']['debug_mode']
    except:
        debug_mode = False

    if debug_mode is True:
        logging.info(' --> Debug mode ACTIVE. All the figures and the ancillary maps will be saved.')
        data_settings['algorithm']['flags']['save_griso_ancillary_maps'] = True
        data_settings['algorithm']['flags']['save_figures'] = True

    # Final correlation
    corrFin = data_settings['algorithm']['settings']['radius_GRISO_km'] * 2
    logging.info(' --> Final correlation: ' + str(corrFin) + ' km')

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

        # Reset settigns for missing data
        if griso_only:
            backup_griso = True
            missing_radar = True
        else:
            backup_griso = False
            missing_radar = False

        if data_settings['algorithm']['flags']["mcm"]['fixed_correlation']:
            corr_type = 'fixed'
        else:
            corr_type = 'dynamic'

        # Compute time step file names
        file_out_time_step = os.path.join(data_settings['data']['outcome']['folder'], data_settings['data']['outcome']['filename'])
        ancillary_out_time_step = os.path.join(data_settings['data']['ancillary']['folder'], data_settings['data']['ancillary']['filename'])
        gridded_in_time_step = os.path.join(data_settings['data']['dynamic']['source_gridded']['folder'], data_settings['data']['dynamic']['source_gridded']['filename'])
        point_in_time_step = os.path.join(data_settings['data']['dynamic']['source_stations']['point_files']['folder'], data_settings['data']['dynamic']['source_stations']['point_files']['filename'])

        for i in data_settings['algorithm']['template']:
            file_out_time_step = file_out_time_step.replace("{" + i + "}", timeNow.strftime(data_settings['algorithm']['template'][i]))
            gridded_in_time_step = gridded_in_time_step.replace("{" + i + "}", timeNow.strftime(data_settings['algorithm']['template'][i]))
            ancillary_out_time_step = ancillary_out_time_step.replace("{" + i + "}", timeNow.strftime(data_settings['algorithm']['template'][i]))
            point_in_time_step = point_in_time_step.replace("{" + i + "}", timeNow.strftime(data_settings['algorithm']['template'][i]))

        if os.path.isfile(file_out_time_step) or os.path.isfile(file_out_time_step + '.gz'):
            if data_settings['algorithm']['flags']['overwrite_existing']:
                pass
            else:
                logging.info(' ---> Time step ' + timeNow.strftime("%Y-%m-%d %H:%M:00") + ' already exist, skipping...')
                continue

        # Make output dir
        os.makedirs(os.path.dirname(file_out_time_step), exist_ok=True)

        # Import gridded source
        if not griso_only:
            logging.info(' ----> Importing gridded source')
            try:
                if data_settings['algorithm']['flags']['compressed_gridded_input']:
                    os.system('gunzip ' + gridded_in_time_step + '.gz')
                if data_settings['data']['dynamic']['source_gridded']['file_type'] == "netcdf" or data_settings['data']['dynamic']['source_gridded']['file_type'] == "nc":
                    logging.info(' ---> Grids in netcdf format')
                    data_settings['data']['dynamic']['source_gridded']['file_type'] = "netcdf"
                    sat = xr.open_dataset(gridded_in_time_step, decode_times=False)
                    sat = sat.rename(
                        {data_settings['data']['dynamic']['source_gridded']['nc_settings']['var_name']: 'precip',
                         data_settings['data']['dynamic']['source_gridded']['nc_settings']['lon_name']: 'lon',
                         data_settings['data']['dynamic']['source_gridded']['nc_settings']['lat_name']: 'lat'})
                elif (fnmatch.fnmatch(data_settings['data']['dynamic']['source_gridded']['file_type'],'*tif*')
                      or fnmatch.fnmatch(data_settings['data']['dynamic']['source_gridded']['file_type'],'*txt*')
                      or fnmatch.fnmatch(data_settings['data']['dynamic']['source_gridded']['file_type'],'AAIGrid')):
                    logging.info(' ---> Grids in tif format')
                    data_settings['data']['dynamic']['source_gridded']['file_type'] = "tif"
                    sat = read_file_tiff(gridded_in_time_step, var_name= 'precip', time=[timeNow],\
                                         coord_name_x='lon', coord_name_y='lat', dim_name_y='lat', dim_name_x='lon')
                    sat['precip'] = sat.where(sat['precip']>=0)['precip']
                else:
                    logging.error(" ---> ERROR! Only netcdf or tif inputs are supported for grid files")
                    raise NotImplementedError("Please, choose 'netcdf' or 'tif' in the setting file!")
            except (FileNotFoundError, rio.rasterio.errors.RasterioIOError) as e:
                missing_radar = True
                if not data_settings['algorithm']['flags']['raise_error_if_no_gridded_available']:
                    sat = read_file_tiff(data_settings['data']['static']['backup_grid'], var_name='precip', time=[timeNow], \
                                         coord_name_x='lon', coord_name_y='lat', dim_name_y='lat', dim_name_x='lon')
                    backup_griso = True
                    logging.error('----> WARNING! File ' + os.path.basename(gridded_in_time_step) + ' not found! Backup on griso only!')
                    gridded_in_time_step = data_settings['data']['static']['backup_grid']
                else:
                    logging.error('----> ERROR! File ' + os.path.basename(gridded_in_time_step) + ' not found!')
                    raise FileNotFoundError

            if data_settings['data']['dynamic']['source_gridded']['file_type'] == "netcdf":
                sat = sat.rename({data_settings['data']['dynamic']['source_gridded']['nc_settings']['var_name']:'precip', data_settings['data']['dynamic']['source_gridded']['nc_settings']['lon_name']:'lon', data_settings['data']['dynamic']['source_gridded']['nc_settings']['lat_name']:'lat'})
        else:
            sat = read_file_tiff(data_settings['data']['static']['backup_grid'], var_name='precip', time=[timeNow], \
                                 coord_name_x='lon', coord_name_y='lat', dim_name_y='lat', dim_name_x='lon')
            backup_griso = True
            gridded_in_time_step = data_settings['data']['static']['backup_grid']

        sat_out = copy.deepcopy(sat)
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
        # If no data available for the actual time step just copy the input
            if not data_settings['algorithm']['flags']['raise_error_if_no_station_available']:
                if missing_radar is True:
                    logging.error(' ----> ERROR! Both gridded and point data are not available for the time step! Skipping time step!')
                    continue
                logging.warning(' ----> WARNING! No valid gauge data available for time step')
                logging.warning(' ----> Use unconditioned radar map as output')
                sat_out_value = sat_out['precip'].squeeze().values
                write_output(sat_out_value, sat_out, format_out, file_out_time_step, gridded_in_time_step)
                continue
            else:
                logging.error(' ----> ERROR! No valid data available for time step')
                raise err

        if not data_settings['algorithm']['flags']["sources"]['use_point_data']:
            # Check if station data with nan values are present and, in the case, remove station from Station dataframe
            # (only if not point data are used, in this case, no further cleaning is needed)
            logging.info(' ----> Check null values for time step ...')
            avail_time_step = len(dfData.columns[~np.isnan(data)])
            logging.info(' ----> Available data for the time step ...' + str(avail_time_step))
            dfStations_available = dfStations.loc[dfData.columns[~np.isnan(data)]]
            data = data[~np.isnan(data)]
        else:
            dfStations_available = copy.deepcopy(dfStations)

        # Extract gridded product values at point locations
        sat_gauge = np.array([sat['precip'].sel({'lon':Lon, 'lat':Lat}, method='nearest').values for Lon,Lat in zip(dfStations_available.lon.astype(np.float),dfStations_available.lat.astype(np.float))])

        ##  Save txt with station data - FOR DEBUGGING
        if debug_mode:
            os.makedirs(os.path.dirname(ancillary_out_time_step), exist_ok=True)
            temp = np.vstack((dfStations_available.lon.astype(np.float),dfStations_available.lat.astype(np.float) , data)).T
            np.savetxt(os.path.join(os.path.dirname(ancillary_out_time_step), 'raingauge_cell_value_' + timeNow.strftime("%Y%m%d_%H%M" + '.txt')), temp)
            temp2 = np.vstack((dfStations_available.lon.astype(np.float),dfStations_available.lat.astype(np.float) , np.squeeze(sat_gauge))).T
            np.savetxt(os.path.join(os.path.dirname(ancillary_out_time_step), 'remotesensing_cell_value_' + timeNow.strftime("%Y%m%d_%H%M" + '.txt')), temp2)

        # Data preprocessing for GRISO
        logging.info(' ---> GRISO: Data preprocessing...')
        point_data, a2dPosizioni, grid_rain, grid, passoKm = GrisoPreproc(corrFin,dfStations_available.lon.astype(np.float),dfStations_available.lat.values.astype(np.float), data, sat_gauge, Grid=sat)
        logging.info(' ---> GRISO: Data preprocessing... DONE')

        # Calculate GRISO correlation features
        logging.info(' ---> GRISO: Calculate correlation...')
        if missing_radar is True:
            corr_type = 'fixed'
        correl_features = GrisoCorrel(point_data["rPluvio"], point_data["cPluvio"], corrFin, grid_rain, passoKm, a2dPosizioni, point_data["gauge_value"], corr_type= corr_type)
        logging.info(' ---> GRISO: Data preprocessing...DONE')

        # GRISO interpolator on observed data
        logging.info(' ---> GRISO: Observed data interpolation...')
        griso_obs = GrisoInterpola(point_data["rPluvio"], point_data["cPluvio"], point_data["gauge_value"], correl_features)

        griso_out = copy.deepcopy(sat)
        try:
            griso_out.precip.values = griso_obs
        except:
            griso_out.precip.values[0,:,:] = griso_obs

        # Save ancillary griso maps
        if data_settings['algorithm']['flags']['save_griso_ancillary_maps']:
            os.makedirs(os.path.dirname(ancillary_out_time_step), exist_ok=True)
            os.system('rm ' + ancillary_out_time_step)
            griso_out.to_netcdf(ancillary_out_time_step)
            if data_settings['algorithm']['flags']['compress_output']:
                os.system('gzip ' + ancillary_out_time_step)
        logging.info(' ---> GRISO: Data preprocessing...DONE')

        if missing_radar is False:
            # GRISO interpolator on satellite data
            logging.info(' ---> GRISO: Gridded data interpolation...')
            griso_sat = GrisoInterpola(point_data["rPluvio"], point_data["cPluvio"], point_data["grid_value"], correl_features)
            logging.info(' ---> GRISO: Gridded data interpolation...DONE')

            ##  Save griso of gridded field - FOR DEBUGGING
            if debug_mode:
                os.makedirs(os.path.dirname(ancillary_out_time_step), exist_ok=True)
                griso_sat_out = copy.deepcopy(sat)
                griso_sat_out.precip.values[0, :, :] = griso_sat
                griso_sat_out.to_netcdf(os.path.join(os.path.dirname(ancillary_out_time_step), 'griso_gridded_' + timeNow.strftime("%Y%m%d_%H%M" + '.nc')))

            # Modified Conditional Merging
            logging.info(' ---> Perform Modified Conditional Merging...')
            error_sat = np.squeeze(grid_rain) - griso_sat

            conditioned_sat = griso_obs + error_sat
            conditioned_sat[conditioned_sat < 0] = 0
            conditioned_sat= np.where(np.isnan(conditioned_sat), griso_obs, conditioned_sat)    # where there is no radar, use griso stations
            mcm_out = check_and_write_dataarray(conditioned_sat, sat, var_name='precip', lat_var_name='lat',
                                      lon_var_name='lon')
            logging.info(' ---> Perform Modified Conditional Merging...DONE')

            # Plot output and save figure
            logging.info(' ---> Make and save figure...')
            if data_settings['algorithm']['flags']['save_figures']:
                os.makedirs(os.path.dirname(ancillary_out_time_step), exist_ok=True)

                fig, axs = plt.subplots(2, 2)

                maxVal = np.nanmax(np.concatenate((sat["precip"].values.flatten(), griso_out["precip"].values.flatten(),
                                                   conditioned_sat.flatten(), data)))
                griso_out["precip"].plot(x="lon", y="lat", cmap='gray', ax=axs[0, 0], alpha=0.4, add_colorbar=False)
                OB = axs[0, 0].scatter(dfStations_available.lon.astype(np.float),
                                       dfStations_available.lat.astype(np.float), 3, c=data, vmin=0, vmax=maxVal)
                axs[0, 0].set_title("OBS")
                axs[0, 0].set_xlabel("")
                axs[0, 0].set_ylabel("")
                plt.colorbar(OB, ax=axs[0, 0])

                SAT = sat["precip"].plot(x="lon", y="lat", ax=axs[0, 1], add_colorbar=False, vmin=0, vmax=maxVal)
                axs[0, 1].set_title("SAT")
                axs[0, 1].set_xlabel("")
                axs[0, 1].set_ylabel("")
                plt.colorbar(SAT, ax=axs[0, 1])

                GR = griso_out["precip"].plot(x="lon", y="lat", ax=axs[1, 0], add_colorbar=False, vmin=0, vmax=maxVal)
                axs[1, 0].set_title("GRISO")
                axs[1, 0].set_xlabel("")
                axs[1, 0].set_ylabel("")
                plt.colorbar(GR, ax=axs[1, 0])

                MCM = mcm_out.plot(x="lon", y="lat", ax=axs[1, 1], add_colorbar=False, vmin=0, vmax=maxVal)
                axs[1, 1].set_xlabel("")
                axs[1, 1].set_ylabel("")
                axs[1, 1].set_title("MCM")
                plt.colorbar(MCM, ax=axs[1, 1])

                plt.suptitle(timeNow.strftime("%Y-%m-%d %H:%M"))
                # plt.show(block=True)

                plt.tight_layout()
                plt.savefig(os.path.join(os.path.dirname(ancillary_out_time_step),
                                         'figure_mcm_' + timeNow.strftime("%Y%m%d_%H%M" + '.png')))
                plt.close()
            logging.info(' ---> Make and save figure...DONE')

        else:
            if backup_griso is True:
                logging.info(' ---> Take rainfall griso as output...')
                conditioned_sat = griso_obs.copy()
                mcm_out = griso_out.copy()
            else:
                raise NotImplementedError("Backup griso is not active, this condition should not subsist!")

        # Save output
        logging.info(' ---> Save output map in ' + format_out + ' format')

        write_output(conditioned_sat, mcm_out, format_out, file_out_time_step, gridded_in_time_step)

        if data_settings['algorithm']['flags']['compress_output']:
            os.system('gzip -f ' + file_out_time_step)

        logging.info(' ---> Saving outfile...DONE')

        if data_settings['algorithm']['flags']['compressed_gridded_input']:
            os.system('gzip ' + gridded_in_time_step)
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
# Call script from external library
if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------