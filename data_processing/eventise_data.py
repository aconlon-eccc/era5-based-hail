import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import xarray as xr
import os
# local imports
import constants_and_variables as cv
import data_processing.load_eventised_observations as load_data
import new_dir as nd
from itertools import cycle

event_time_window=cv.event_time_window
event_spatial_window=cv.event_spatial_window
earth_rad=cv.earth_rad
boundary_dist=cv.boundary_distance
num_level_steps=len(cv.pressure_levels)

def observations(file_name, destination_file_name=''):

    if destination_file_name == '':
        destination_file_name = 'eventised_obs.csv'

    def cosine_dist(lon_1, lat_1, lon_2, lat_2):
        # convert long/lat information from degrees to radians
        lon_1 = np.deg2rad(lon_1)
        lat_1 = np.deg2rad(lat_1)
        lon_2 = np.deg2rad(lon_2)
        lat_2 = np.deg2rad(lat_2)

        dlon = lon_2 - lon_1
        a = np.sin(lat_1) * np.sin(lat_2) + np.cos(lat_1) * np.cos(lat_2) * np.cos(dlon)
        a = round(a, 14)  # round to the 14 decimal place, otherwise we get values larger than 1
        ca = np.arccos(a)

        return earth_rad * ca

    def group_reports():
        data = pd.read_csv(file_name, index_col=0, na_values=float('nan'))
        data = data.reset_index(drop=True)

        # create timedelta objects from report 'Start Time' and 'End Time' to help us compare times of reports
        dt_format = '%m/%d/%Y %H:%M'  # set dt_format to the format printed above
        data['Temp Start Timedelta'] = [dt.strptime(startDate, dt_format) for startDate in
                                        data[
                                            'Start Time']]  # create new column in our data table called 'Start timedelta'
        na_index = data['End Time'].index[
            data['End Time'].apply(pd.isna)]  # list of indexes of NaN values in 'End Time' column
        data['End Time'] = np.where(pd.isna(data['End Time']), data['Start Time'],
                                    data['End Time'])  # replacing NaN values by 'Start Time' values
        data['Temp End Timedelta'] = [dt.strptime(endDate, dt_format) for endDate in
                                      data['End Time']]  # create new column in our data table called 'End timedelta'
        data['Start Timedelta'] = np.where((data['Temp Start Timedelta'] <= data['Temp End Timedelta']),
                                           data['Temp Start Timedelta'],
                                           data['Temp End Timedelta'])
        data['End Timedelta'] = np.where((data['Temp Start Timedelta'] >= data['Temp End Timedelta']),
                                         data['Temp Start Timedelta'],
                                         data['Temp End Timedelta'])
        data.loc[na_index, 'End Time'] = float('nan')  # reset appropriate 'End Time' values back to NaN
        # sort data by 'End Timedelta'
        data = data.sort_values(by='End Timedelta')
        data = data.reset_index(drop=True)

        dist = 0
        dists = [dist]
        for i in range(len(data) - 1):
            sdist = cosine_dist(data['Longitude'][i], data['Latitude'][i], data['Longitude'][i + 1],
                                data['Latitude'][i + 1])
            dists.append(sdist)
        data['Cosine Distances'] = dists

        # parse hail data into seperate events according to temporal and spatial windows
        event = 0  # initializing the event counter
        events = [event]  # initializing the list of event IDs
        for i in range(len(data) - 1):
            tdist = data['Start Timedelta'][i + 1] - data['End Timedelta'][i]
            if tdist > timedelta(hours=event_time_window):
                event += 1
            else:
                sdist = cosine_dist(data['Longitude'][i], data['Latitude'][i], data['Longitude'][i + 1],
                                    data['Latitude'][i + 1])
                if sdist > event_spatial_window:
                    event += 1
            events.append(event)
        data['Event'] = events
        print('Number of events: ', max(events) + 1)

        data.to_csv(destination_file_name, index=False)

        return data

    return group_reports()

def era5(year, eventised_observations_path, sl_file_path, pl_dir_path, destination_dir_path='', init_event=0, fin_event=0):

    try:
        sl_ds = xr.open_dataset(sl_file_path)
    except ValueError as p:
        raise Exception("Path provided for 'sl_file_path' does not exist.").with_traceback(p.__traceback__)

    eventised_observations = load_data.load_obs(eventised_observations_path)

    eventised_dir = 'era5_{}.by_event'.format(year)
    dest = os.path.join(destination_dir_path, eventised_dir)
    dest = nd.make_dir(dest)

    obs_data = eventised_observations.where(eventised_observations['Year'] == year).dropna(how='all')
# check if we are starting from a specific event within the year (useful if program was interrupted)
    obs_data = obs_data.where(obs_data['Event'] >= init_event).dropna(how='all')
# check if we are working until the final event of the specified year
    if fin_event != 0:
        obs_data = obs_data.where(obs_data['Event'] <= init_event).dropna(how='all')

 # retrieve event IDs from user-specified year
    event_list = obs_data['Event']

    for event in event_list:
        # time
        ini_time = min(obs_data['Start Timedelta'].where(obs_data['Event'] == event).dropna()) - timedelta(
            hours=event_time_window)
        fin_time = max(obs_data['End Timedelta'].where(obs_data['Event'] == event).dropna()) + timedelta(
            hours=event_time_window)

        # longitude
        event_lons = obs_data['Longitude'].where(obs_data['Event'] == event).dropna()
        min_lon = min(event_lons) - np.rad2deg(boundary_dist)
        max_lon = max(event_lons) + np.rad2deg(boundary_dist)

        # latitude
        event_lats = obs_data['Latitude'].where(obs_data['Event'] == event).dropna()
        min_lat = min(event_lats) - np.rad2deg(boundary_dist)
        max_lat = max(event_lats) + np.rad2deg(boundary_dist)

        # ensure bounds do not go out of range of era5 sl data
        ini_time = max(ini_time, min(sl_ds.time))
        fin_time = min(fin_time, max(sl_ds.time))

        min_lon = max(min_lon, min(sl_ds.longitude))
        max_lon = min(max_lon, max(sl_ds.longitude))

        min_lat = max(min_lat, min(sl_ds.latitude))
        max_lat = min(max_lat, max(sl_ds.latitude))

        # slice single dataset according to bounds
        sl_ds = sl_ds.sel(time=slice(ini_time, fin_time), longitude=slice(min_lon, max_lon),
                          latitude=slice(max_lat, min_lat))

        # open first pressure level dataset
        pl_file_path = os.path.join(pl_dir_path, 'era5_pl.{}.{}.h5'.format(year, 0))

        # open pressure level dataset
        try:
            pl_ds = xr.open_dataset(pl_file_path)
        except ValueError as p:
            raise Exception("Path provided for 'pl_dir_path' does not exist.").with_traceback(p.__traceback__)

        # slice pressure level dataset according to bounds
        pl_ds = pl_ds.sel(time=slice(ini_time, fin_time), longitude=slice(min_lon, max_lon),
                          latitude=slice(max_lat, min_lat))

        # merge single level and pressure level sliced datasets
        new_ds = xr.merge([sl_ds, pl_ds])

        for pl_step in range(1, num_level_steps):
            # open first pressure level dataset
            pl_file_path = os.path.join(pl_dir_path, 'era5_pl.{}.{}.h5'.format(year, pl_step))

            # open pressure level dataset
            try:
                pl_ds = xr.open_dataset(pl_file_path)
            except ValueError as p:
                raise Exception("Path provided for 'pl_dir_path' does not exist.").with_traceback(p.__traceback__)

            # slice pressure level dataset according to bounds
            pl_ds = pl_ds.sel(time=slice(ini_time, fin_time), longitude=slice(min_lon, max_lon),
                        latitude=slice(max_lat, min_lat))

            # merge single level and pressure level sliced datasets
            new_ds = xr.merge([new_ds, pl_ds])
            new_file_path = os.path.join(dest, 'era5.event_{}.h5'.format(int(event)))
            new_ds.to_netcdf(new_file_path, engine='h5netcdf', invalid_netcdf=True)

            new_ds.close()

