import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import constants_and_variables as cv
import xarray as xr
import os

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

def era5(era5_dataset, year, eventised_observations, era5_dir_path, destination_dir_path='', init_event=0,
         init_level_step=0):

    dir = era5_dir_path
    era5ds = era5_dataset
    data = eventised_observations
    from_event = init_event

    eventised_dir = 'era5_{}.{}.by_event'.format(era5ds, year)
    dest = os.path.join(destination_dir_path, eventised_dir)

    def make_dir(path, nf=0):
        new_path = path
        if nf > 0:
            new_path = '{}.{}'.format(path, nf)
        try:
            os.mkdir(new_path)
            return new_path
        except FileExistsError:
            nf+=1
            return make_dir(path, nf=nf)

    dest = make_dir(dest)

    def breakup_era5(level_step):
        # change type from str to datetime
        dt_format = '%Y-%m-%d %H:%M:%S'  # set dt_format to the format printed above
        data['Temp Start Timedelta'] = [dt.strptime(str(startDate), dt_format) for startDate in data['Start Timedelta']]
        data['Start Timedelta'] = data['Temp Start Timedelta']

        data['Temp End Timedelta'] = [dt.strptime(str(endDate), dt_format) for endDate in data['End Timedelta']]
        data['End Timedelta'] = data['Temp End Timedelta']

        # finding indices of reports in the year of interest (ev_num)
        ev_inds = data['Year'].index[data['Year'] == year]
        # check if we are starting from a specific event within the year (useful if program was interrupted)
        if from_event != 0:
            ini_ind = min(data['Event'].index[data['Event'] == from_event])
            ev_inds = ev_inds[ev_inds.values.tolist().index(ini_ind):]
        # retrieve event numbers from year of interest
        event_list = data['Event'].iloc[ev_inds]
        # flatten list of event numbers (removes repeated numbers)
        event_dict = dict.fromkeys(event_list)

        event_keys = list(event_dict)

        for event in event_keys:
            # time
            ini_time = min(data['Start Timedelta'].where(data['Event'] == event).dropna()) - timedelta(
                hours=event_time_window)
            fin_time = max(data['End Timedelta'].where(data['Event'] == event).dropna()) + timedelta(
                hours=event_time_window)

            # longitude
            event_lons = data['Longitude'].where(data['Event'] == event).dropna()
            min_lon = min(event_lons) - np.rad2deg(boundary_dist)
            max_lon = max(event_lons) + np.rad2deg(boundary_dist)

            # latitude
            event_lats = data['Latitude'].where(data['Event'] == event).dropna()
            min_lat = min(event_lats) - np.rad2deg(boundary_dist)
            max_lat = max(event_lats) + np.rad2deg(boundary_dist)

            # open dataset
            filename = 'era5_{}.{}.h5'.format(year, level_step)
            filename = os.path.join(dir, filename)
            newfile = 'era5_pl.event_{}.{}.h5'.format(event, level_step)
            newfile = os.path.join(dest, newfile)
            if era5ds == 'sl':
                filename = 'era5_sl.{}.h5'.format(year)
                filename = os.path.join(dir, filename)
                newfile = 'era5.sl.event_{}.h5'.format(event)
                newfile = os.path.join(dest, newfile)

            ds = xr.open_dataset(filename)

            # ensure bounds do not go out of range of era5 data
            ini_time = max(ini_time, min(ds.time))
            fin_time = min(fin_time, max(ds.time))

            min_lon = max(min_lon, min(ds.longitude))
            max_lon = min(max_lon, max(ds.longitude))

            min_lat = max(min_lat, min(ds.latitude))
            max_lat = min(max_lat, max(ds.latitude))

            ds = ds.sel(time=slice(ini_time, fin_time), longitude=slice(min_lon, max_lon),
                        latitude=slice(max_lat, min_lat))

            ds.to_netcdf(newfile, engine='h5netcdf', invalid_netcdf=True)

            ds.close()

    if era5ds == 'pl':
        for ls in range(init_level_step, num_level_steps):
            breakup_era5(ls)
            print('done eventising {}.{}.{}'.format(era5ds, year, ls))
    else:
        breakup_era5(0)
    return print('done eventising {}.{}'.format(era5ds, year))