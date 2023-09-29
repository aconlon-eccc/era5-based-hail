import constants_and_variables as cv
import numpy as np
from datetime import timedelta
import cdsapi
import os

event_time_window = cv.event_time_window
event_spatial_window = cv.event_spatial_window
earth_rad = cv.earth_rad
boundary_dist = cv.boundary_distance
era5_datasets = cv.era5_datasets
hours = cv.hours
pressure_levels = cv.pressure_levels
pressure_level_variables = cv.pressure_level_variables
single_level_variables = cv.single_level_variables

# function to make ERA5 data request
def era5_req(era5ds, days, months, years, spatial_bounds, filename, pressure_levels=[]):
    if era5ds == 'pl':
        attr = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'pressure_level': pressure_levels,
            'year': years,
            'month': months,
            'day': days,
            'time': hours,
            'variable': pressure_level_variables,
            'area': spatial_bounds,
        }
    else:
        attr = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': single_level_variables,
            'year': years,
            'month': months,
            'day': days,
            'time': hours,
            'area': spatial_bounds,
        }

    c = cdsapi.Client()
    c.retrieve(era5_datasets[era5ds], attr, filename)

# make ERA5 data request according to a specified event number 'ev_num'.
def submit_request(era5_dataset, year, eventised_observations, destination_dir='', init_pressure_level=0, init_event=0, fin_event=0):
    data = eventised_observations

    if init_event != 0:
        data = data.where(data['Event'] >= init_event)
    if fin_event != 0:
        data = data.where(data['Event'] <= fin_event)

    era5_dir = 'era5_{}'.format(era5_dataset)
    destination_path = os.path.join(destination_dir, era5_dir)
    try:
        os.mkdir(destination_path)
    except FileExistsError:
        pass

    if era5_dataset == 'pl':
        destination_path = os.path.join(destination_path, 'era5_pl.{}'.format(year))

    try:
        os.mkdir(destination_path)
    except FileExistsError:
        pass

    era5ds = era5_dataset
    # finding min and max indices in data of reports of the specified year
    year_inds = data['Year'].index[data['Year'] == year]
    min_yr_ind = min(year_inds)
    max_yr_ind = max(year_inds)
    # finding min and max longitudes of reports in the event 'ev_num'
    lons = data['Longitude'].iloc[year_inds]
    max_lon = max(lons) + np.rad2deg(boundary_dist)
    min_lon = min(lons) - np.rad2deg(boundary_dist)
    # finding min and max latitudes of reports in the event 'ev_num'
    lats = data['Latitude'].iloc[year_inds]
    max_lat = max(lats) + np.rad2deg(boundary_dist)
    min_lat = min(lats) - np.rad2deg(boundary_dist)
    # defining the spatial boundary for the ERA5 data request
    spatial_bounds = [max_lat, min_lon, min_lat, max_lon]

    # computing the initial time of the temporal window
    ini_rep_time = data['Start Timedelta'][min_yr_ind] - timedelta(hours=event_time_window)

    # computing the final time of the temporal window
    fin_time = data['End Timedelta'][max_yr_ind] + timedelta(hours=event_time_window + 1)

    # taking the floor and ceiling in days of the initial and final times, respectively, of the temporal window
    # that is, ini_time = '15:00, April 15, 2023' goes to '00:00, April 15, 2023'
    # and fin_time = '02:00, April 16, 2023' goes to '23:00, April 16, 2023'
    # the ERA5 request will be done for all hours of each day in the event (ERA5 data requests do not allow for a mix
    # of partial days and full days). Once the ERA5 data is downloaded, we
    # will then remove data that lies outside our desired temporal window.
    ini_rep_time_floor = ini_rep_time - timedelta(hours=ini_rep_time.hour)
    if fin_time.hour < 23:
        fin_time_ceil = fin_time + timedelta(hours=(23 - fin_time.hour))
    else:
        fin_time_ceil = fin_time

    # define days, months, and years that we will pass in the ERA5 request
    days = []
    months = []
    years = []
    curr_time = ini_rep_time_floor
    while curr_time <= fin_time_ceil:
        days.append(curr_time.day)
        months.append(curr_time.month)
        years.append(curr_time.year)
        curr_time = curr_time + timedelta(days=1)
    months = list(dict.fromkeys(months))
    years = list(dict.fromkeys(years))
    for d in range(len(days)):
        if days[d] < 10:
            days[d] = '0' + str(days[d])
        else:
            days[d] = str(days[d])
    for m in range(len(months)):
        if months[m] < 10:
            months[m] = '0' + str(months[m])
        else:
            months[m] = str(months[m])
    years = [str(y) for y in years]

    if era5ds == 'pl':
        for lvl in range(init_pressure_level, len(pressure_levels)):
            file_name = 'era5_pl.{}.{}.h5'.format(year, lvl)
            file_path = os.path.join(destination_path, file_name)
            era5_req(era5ds, days, months, years, spatial_bounds, file_path, pressure_levels[lvl])

    else:
        file_name = 'era5_sl.{}.h5'.format(year)
        file_path = os.path.join(destination_path, file_name)
        era5_req(era5ds, days, months, years, spatial_bounds, file_path)

