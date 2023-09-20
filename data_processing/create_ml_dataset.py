import pandas as pd
from datetime import datetime as dt
import numpy as np
import xarray as xr
import glob
from metpy.calc import bulk_shear, wind_speed, mixed_layer_cape_cin, geopotential_to_height, most_unstable_cape_cin
from metpy.calc import dewpoint_from_relative_humidity, wet_bulb_temperature
from metpy.units import units
from scipy.interpolate import interp1d
import re, os
import time

def build_hail_ds_nans(obs_path, era5_dir, destination_dir='', ini_ev=0, fin_ev=0, time_limit=5.5, severe=20, save_freq=50):
    file_name_format = 'era5.event_*.h5'
    tic = time.perf_counter()
    # get list of file paths
    files = glob.glob(era5_dir + file_name_format)
    files.sort()

    # extract first and last event from list of file paths
    era5_data_events = [int(re.search('(.*)era5.event_(.*).h5', file).group(2)) for file in files]
    era5_data_events.sort()

    #era5_data_events = [int(files[i][-7:-3]) for i in range(len(files))]
    # first_event = era5_data_events[0]

    # load clean eventised data
    data = pd.read_csv(obs_path)

    if fin_ev == 0:
        fin_ev = max(data['Event'])

    event_range = np.arange(ini_ev, fin_ev + 1)
    event_list = data['Event'].where((data['Event'] >= ini_ev)).dropna()
    event_list = event_list.where(event_list <= fin_ev).dropna().astype(int).unique()

    # init_ev_ind = np.where(event_list == ini_ev)[0][0]
    # fin_ev_ind = np.where(event_list == fin_ev)[0][0]

    # create dictionary with appropriate headers to be turned into pandas dataframe
    ## pressure level dependent variables
    plvar = ['r', 't', 'u', 'v', 'z']

    ## single level variables
    slvar = ['cp', 'd2m', 'sp', 't2m', 'tcc', 'tciw', 'tclw', 'tcrw', 'tcsw', 'tcw', 'tcwv', 'tp', 'u10', 'v10']

    ## pressure levels
    lvls = ['300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '775', '800', '825', '850', '875',
            '900', '925', '950', '975', '1000']

    ## additional indices calculated below
    indices = ['bulk_shear_0_6_km', 'bulk_shear_0_3_km', 'bulk_shear_0_1_km', 'mean_wind_1_3_km', 'lapse_rate_0_3_km',
               'lapse_rate_2_6_km', 'wet_bulb_temperature_0_deg_height', 'cape', 'cin', 'cape_depth_90', 'cin_depth_90']

    ## indicators from hail data csv
    hail_x_data = ['LD']
    hail_y_data = ['hail_size']

    ## create dictionaries
    event_dict = {'event': [], 'year': [], 'start_time': [], 'end_time': [], 'latitude': [], 'longitude': []}
    x_dict = {}
    y_dict = {}
    for t in range(10):
        for lvl in lvls:
            for vr in plvar:
                column = '{}_{}.t{}'.format(vr, lvl, t)
                x_dict[column] = []
        for slvr in slvar:
            x_dict['{}.t{}'.format(slvr, t)] = []
        for ind in indices:
            x_dict['{}.t{}'.format(ind, t)] = []
        for hdi in hail_x_data:
            x_dict['{}.t{}'.format(hdi, t)] = []

    for y in hail_y_data:
        y_dict[y] = []
    y_dict['severe'] = []

    # mu_cape_trouble = {'event': [], 'report': [], 'lat': [], 'lon': [], 'time': []}
    # populate dictionaries
    file_count = 0
    for event in event_list:
        last_ev = event
        toc = time.perf_counter()
        tictoc = (toc - tic)/(60*60)
        print('Elapsed time {:0.2f}hrs'.format(tictoc))
        if tictoc < time_limit:
            # get file name for ERA5 data of event
            file_name = 'era5.event_{}.h5'.format(event)
            file = os.path.join(era5_dir, file_name)

            # load ERA5 data of event
            ds = xr.open_dataset(file, engine='h5netcdf')

            # extract hail report data for event
            sub_data = data.where(data['Event'] == event).dropna(how='all')

            # extract ERA5 data for reports within event
            for ind in range(len(sub_data.index)):

                # get latitude window
                rep_lat = sub_data['Latitude'].iloc[ind]
                # s_lat = sub_data['Latitude'].iloc[ind] - 0.25
                # e_lat = s_lat + 0.5

                # get longitude window
                rep_lon = sub_data['Longitude'].iloc[ind]
                # s_lon = sub_data['Longitude'].iloc[ind] - 0.25
                # e_lon = s_lon + 0.5

                # get time window
                rep_s_tme = dt.strptime(sub_data['Start Timedelta'].iloc[ind], '%Y-%m-%d %H:%M:%S')
                rep_e_tme = dt.strptime(sub_data['End Timedelta'].iloc[ind], '%Y-%m-%d %H:%M:%S')

                tme = ds.sel(time=rep_s_tme, method='nearest').time

                s_tme = tme - np.timedelta64(6, 'h')
                e_tme = tme + np.timedelta64(3, 'h')

                # slice ERA5 data to associated report
                sub_ds = ds.sel(latitude=rep_lat, longitude=rep_lon, method='nearest').sel(time=slice(s_tme, e_tme))

                tmes = sub_ds.time

                # insert event
                event_dict['event'].append(event)

                event_dict['start_time'].append(rep_s_tme)
                event_dict['end_time'].append(rep_e_tme)
                event_dict['year'].append(rep_e_tme.year)
                event_dict['latitude'].append(rep_lat)
                event_dict['longitude'].append(rep_lon)

                # insert max hail size for event
                hs = sub_data['Hail Diameter (mm) FINAL'].iloc[ind]
                y_dict['hail_size'].append(hs)

                # insert severity
                if (hs > severe):
                    y_dict['severe'].append(1)
                else:
                    y_dict['severe'].append(0)


                t_count = 0
                for t in tmes:
                    print('Working on event {}, report {} of {} at time step {} of {}'.format(event, ind + 1, len(sub_data.index), t_count+1, len(sub_ds.time)))


                    rep_ds = sub_ds.sel(time=t)

                    # insert lightning data for report
                    x_dict['LD.t{}'.format(t_count)].append(sub_data['LD'].iloc[ind])

                    # flatten sliced dataset
                    DA = rep_ds

                    for plvr in plvar:
                        for lvl in lvls:
                            k = '{}_{}.t{}'.format(plvr, lvl, t_count)
                            x_dict[k].append(DA[plvr].sel(level=int(lvl)).values)
                    for slvr in slvar:
                        r = '{}.t{}'.format(slvr, t_count)
                        x_dict[r].append(DA[slvr].values)

                    # calculate indices and populate row
                    p = DA.level * units.hPa
                    u = DA.u * units.m / units.s
                    v = DA.v * units.m / units.s
                    z = DA.z * units.m ** 2 / units.s ** 2
                    height = geopotential_to_height(z)

                    u_shr, v_shr = bulk_shear(p, u, v, height, bottom=height[-1], depth=6000 * units.m)
                    bulk_shear_0_6_km = wind_speed(u_shr, v_shr)
                    x_dict['bulk_shear_0_6_km.t{}'.format(t_count)].append(bulk_shear_0_6_km.magnitude)

                    u_shr, v_shr = bulk_shear(p, u, v, height, bottom=height[-1], depth=3000 * units.m)
                    bulk_shear_0_3_km = wind_speed(u_shr, v_shr)
                    x_dict['bulk_shear_0_3_km.t{}'.format(t_count)].append(bulk_shear_0_3_km.magnitude)

                    u_shr, v_shr = bulk_shear(p, u, v, height, bottom=height[-1], depth=1000 * units.m)
                    bulk_shear_0_1_km = wind_speed(u_shr, v_shr)
                    x_dict['bulk_shear_0_1_km.t{}'.format(t_count)].append(bulk_shear_0_1_km.magnitude)

                    u_1_3_km = u.where((height >= 1000 * units.m) & (height <= 3000 * units.m)).dropna(dim='level')
                    v_1_3_km = v.where((height >= 1000 * units.m) & (height <= 3000 * units.m)).dropna(dim='level')
                    mean_wind_1_3_km = wind_speed(u_1_3_km, v_1_3_km).mean()
                    x_dict['mean_wind_1_3_km.t{}'.format(t_count)].append(mean_wind_1_3_km.values)

                    temperature = DA.t * units.kelvins
                    f = interp1d(height.values, temperature.values)
                    lapse_rate_0_3_km = temperature[-1].values - f(3000)
                    x_dict['lapse_rate_0_3_km.t{}'.format(t_count)].append(lapse_rate_0_3_km)

                    f = interp1d(height.values, temperature.values)
                    step = 100
                    LR_max = 0
                    for h in range(2000, 4000 + step, step):
                        LR_curr = f(h) - f(h + 2000)
                        if LR_curr > LR_max:
                            LR_max = LR_curr
                    # lapse_rate_2_6_km = f(2000) - f(6000)
                    x_dict['lapse_rate_2_6_km.t{}'.format(t_count)].append(LR_max)

                    rh = DA.r * units.percent
                    dewpoint = dewpoint_from_relative_humidity(temperature, rh)
                    wet_bulb = wet_bulb_temperature(p, temperature, dewpoint)
                    idxl = np.where(wet_bulb.values < 273.15)[0]
                    idx = idxl[-1]
                    if len(idxl) < 2:
                        x_dict['wet_bulb_temperature_0_deg_height.t{}'.format(t_count)].append(height.values[0])
                    elif idx == 19:
                        x_dict['wet_bulb_temperature_0_deg_height.t{}'.format(t_count)].append(height.values[19])
                    else:
                        idx = idxl[-1]
                        try:
                            f = interp1d(wet_bulb.values[idx:idx + 2], height.values[idx:idx + 2])
                        except:
                            print('idxl')
                            print(len(idxl))
                            print(idxl)
                        wet_bulb_temperature_0_deg_height = f(273.15)
                        x_dict['wet_bulb_temperature_0_deg_height.t{}'.format(t_count)].append(wet_bulb_temperature_0_deg_height)

                    p = DA.level[::-1] * units.hPa
                    temperature = DA.t[::-1] * units.kelvins
                    rh = DA.r[::-1] * units.percent
                    dewpoint = dewpoint_from_relative_humidity(temperature, rh)

                    try:
                        mu_cape, mu_cin = most_unstable_cape_cin(p, temperature, dewpoint)
                        x_dict['cape.t{}'.format(t_count)].append(mu_cape.magnitude)
                        x_dict['cin.t{}'.format(t_count)].append(mu_cin.magnitude)
                    except:
                        mu_cape = float('nan')
                        mu_cin = float('nan')
                        x_dict['cape.t{}'.format(t_count)].append(mu_cape)
                        x_dict['cin.t{}'.format(t_count)].append(mu_cin)
                    try:
                        cape_depth_90, cin_depth_90 = mixed_layer_cape_cin(p, temperature, dewpoint, depth=90 * units.hPa)
                        x_dict['cape_depth_90.t{}'.format(t_count)].append(cape_depth_90.magnitude)
                        x_dict['cin_depth_90.t{}'.format(t_count)].append(cin_depth_90.magnitude)
                    except:
                        cape_depth_90 = float('nan')
                        cin_depth_90 = float('nan')
                        x_dict['cape_depth_90.t{}'.format(t_count)].append(cape_depth_90)
                        x_dict['cin_depth_90.t{}'.format(t_count)].append(cin_depth_90)


                    t_count+=1

                file_count += 1

                if (file_count % save_freq) == 0:
                    # merge dictionnaries
                    x_dict.update(y_dict)
                    event_dict.update(x_dict)

                    # create dataframe containing all data

                    df = pd.DataFrame(data=event_dict)
                    fn = destination_dir + 'partial_ml_dataset.{}_{}.csv'.format(ini_ev, fin_ev)
                    df.to_csv(fn, index=False)
                    print('saved partial hail ds')
                    print(fn)


        else:
            # create dataframes for train/test split
            # x_df = pd.DataFrame(data=x_dict)
            # x_df.to_csv(destination_dir + 'partial_x_hail_nans_dataset.{}_{}.csv'.format(ini_ev, fin_ev), index=False)
            # y_df = pd.DataFrame(data=y_dict)
            # y_df.to_csv(destination_dir + 'partial_y_hail_nans_dataset.{}_{}.csv'.format(ini_ev, fin_ev), index=False)

            # merge dictionnaries
            x_dict.update(y_dict)
            event_dict.update(x_dict)

            # create dataframe containing all data
            df = pd.DataFrame(data=event_dict)
            df.to_csv(destination_dir + 'partial_ml_dataset.{}_{}.csv'.format(ini_ev, fin_ev), index=False)
    # mu_cape_trouble_df = pd.DataFrame(data=mu_cape_trouble)
    # mu_cape_trouble_df.to_csv(destination_dir + 'mu_cape_trouble2.csv', index=False)

    # create dataframes for train/test split
    x_df = pd.DataFrame(data=x_dict)
    # x_df.to_csv(destination_dir + 'partial_x_hail_nans_dataset.{}_{}.csv'.format(ini_ev, fin_ev), index=False)
    y_df = pd.DataFrame(data=y_dict)
    # y_df.to_csv(destination_dir + 'partial_y_hail_nans_dataset.{}_{}.csv'.format(ini_ev, fin_ev), index=False)

    # merge dictionnaries
    x_dict.update(y_dict)
    event_dict.update(x_dict)

    # create dataframe containing all data
    df = pd.DataFrame(data=event_dict)
    df.to_csv(destination_dir + 'ml_dataset.{}_{}.csv'.format(ini_ev, fin_ev), index=False)