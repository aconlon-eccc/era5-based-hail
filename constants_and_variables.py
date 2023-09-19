event_time_window = 7 # amount of time in hours between reports to be considered the same event
event_spatial_window = 256 # distance in km between reports to be considered the same event
earth_rad = 6371.0088 # arithmetic mean radius taken from Wikipedia
boundary_distance = event_spatial_window / (2 * earth_rad)  # spatial window in radians around reports for training data

era5_datasets = {'sl': 'reanalysis-era5-single-levels', 'pl': 'reanalysis-era5-pressure-levels'}

hours = [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ]
# when downloading from the 'reanalysis-era5-pressure-levels' ERA5 dataset, requests are broken up into 10 requests
# by pressure levels for each year. This helps ensure we do not go over the maximum limit of files requested from ERA5.
pl0 = ['300', '350', ]
pl1 = ['400', '450', ]
pl2 = ['500', '550', ]
pl3 = ['600', '650', ]
pl4 = ['700', '750', ]
pl5 = ['775', '800', ]
pl6 = ['825', '850', ]
pl7 = ['875', '900', ]
pl8 = ['925', '950', ]
pl9 = ['975', '1000', ]
pressure_levels = [pl0, pl1, pl2, pl3, pl4, pl5, pl6, pl7, pl8, pl9]

# variables requested when submitting an ERA5 request from the 'reanalysis-era5-single-levels' dataset
single_level_variables = [
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            '2m_temperature',
            '2m_dewpoint_temperature',
            'surface_pressure',
            'total_column_water',
            'total_precipitation',
            'convective_precipitation',
            'total_column_water_vapour',
            'total_column_rain_water',
            'total_column_snow_water',
            'total_column_cloud_ice_water',
            'total_column_cloud_liquid_water',
            'total_cloud_cover']

# variables requested when submitting an ERA5 request from the 'reanalysis-era5-pressure-levels' dataset
pressure_level_variables = ['geopotential', 'relative_humidity', 'temperature', 'u_component_of_wind',
                            'v_component_of_wind', ]
