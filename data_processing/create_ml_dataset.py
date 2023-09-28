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
import time, random

def hail_ds(obs_path, era5_dir, destination_dir='', ini_ev=0, fin_ev=0, time_limit=5.5, severe=20, save_freq=50):
    file_name_format = 'era5.event_*.h5'
    tic = time.perf_counter()

    # get list of file paths
    files = glob.glob(era5_dir + file_name_format)
    files.sort()

    # extract first and last event from list of file paths
    era5_data_events = [int(re.search('(.*)era5.event_(.*).h5', file).group(2)) for file in files]
    era5_data_events.sort()

    # load clean eventised data
    data = pd.read_csv(obs_path)

    event_list = data['Event'].where((data['Event'] >= ini_ev)).dropna()
    if fin_ev > 0:
        event_list = event_list.where(event_list <= fin_ev).dropna().astype(int).unique()
    else:
        fin_ev = max(event_list)

    # create dictionary with appropriate headers to be turned into pandas dataframe
    # pressure level dependent variables
    plvar = ['r', 't', 'u', 'v', 'z']

    # single level variables
    slvar = ['cp', 'd2m', 'sp', 't2m', 'tcc', 'tciw', 'tclw', 'tcrw', 'tcsw', 'tcw', 'tcwv', 'tp', 'u10', 'v10']

    # pressure levels
    lvls = ['300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '775', '800', '825', '850', '875',
            '900', '925', '950', '975', '1000']

    # additional indices calculated below
    indices = ['bulk_shear_0_6_km', 'bulk_shear_0_3_km', 'bulk_shear_0_1_km', 'mean_wind_1_3_km', 'lapse_rate_0_3_km',
               'lapse_rate_2_6_km', 'wet_bulb_temperature_0_deg_height', 'cape', 'cin', 'cape_depth_90', 'cin_depth_90']

    # indicators from hail data csv
    hail_x_data = ['LD']
    hail_y_data = ['hail_size']

    # create dictionaries
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

    # populate dictionaries
    file_count = 0
    for event in event_list:
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

                # get longitude window
                rep_lon = sub_data['Longitude'].iloc[ind]

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

                    # set mu-cape and -cin to 'NAN' if calculation raises error
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
                    fn = destination_dir + 'partial_ml_hail_dataset.{}_{}.csv'.format(ini_ev, fin_ev)
                    df.to_csv(fn, index=False)
                    print('saved partial hail ds')
                    print(fn)

        # if time limit is reached
        else:
            # merge x, y, and event dictionnaries
            x_dict.update(y_dict)
            event_dict.update(x_dict)

            # create dataframe containing all data
            df = pd.DataFrame(data=event_dict)
            df.to_csv(destination_dir + 'partial_ml_hail_dataset.{}_{}.csv'.format(ini_ev, fin_ev), index=False)

    # run completed, save csv
    # merge dictionnaries
    x_dict.update(y_dict)
    event_dict.update(x_dict)

    # create dataframe containing all data
    df = pd.DataFrame(data=event_dict)
    fn = os.path.join(destination_dir, 'ml_hail_dataset.{}_{}.csv'.format(ini_ev, fin_ev))
    df.to_csv(fn, index=False)

    return df

def null_ds(sl_file_path, pl_dir_path, num_reports, destination_dir='', time_limit=5.5, save_freq=50):
    tic = time.perf_counter()

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

    # open single level dataset
    try:
        sl_ds = xr.open_dataset(sl_file_path)
    except ValueError as p:
        raise Exception("Path provided for 'sl_file_path' does not exist.").with_traceback(p.__traceback__)

    for null_rep in range(num_reports):
        # file_count +=1
        toc = time.perf_counter()
        tictoc = (toc - tic) / (60 * 60)
        print('Elapsed time {:0.2f}hrs'.format(tictoc))
        if tictoc > time_limit:
            # merge x, y, and event dictionnaries
            x_dict.update(y_dict)
            event_dict.update(x_dict)

            # create dataframe containing all data
            df = pd.DataFrame(data=event_dict)
            df.to_csv(destination_dir + 'partial_null_ml_dataset.{}.csv'.format(num_reports), index=False)

        # choose random lat/lon/times
        t_len = 0
        while (t_len != 10):
            t_ind = random.choice(range(len(sl_ds.time[6:-5])))
            tmes = sl_ds.time[t_ind - 6:t_ind + 4]
            t_len = len(tmes)
        rep_s_tme = tmes[0]
        rep_e_tme = tmes[-1]
        lat = random.choice(sl_ds.latitude)
        lon = random.choice(sl_ds.longitude)

        # get year
        year = sl_ds.time[t_ind].values.astype('datetime64[Y]').astype(int) + 1970

        # slice single-level dataset
        sub_ds = sl_ds.sel(latitude=lat, longitude=lon, time=slice(rep_s_tme, rep_e_tme))

        # get file name for ERA5 data of event
        pl_files = glob.glob(pl_dir_path + '/era5_pl.{}.*.h5'.format(year))

        for pl_file in pl_files:
            pl_ds = xr.open_dataset(pl_file)
            pl_ds = pl_ds.sel(latitude=lat, longitude=lon, time=slice(rep_s_tme, rep_e_tme))
            sub_ds = xr.merge([pl_ds, sub_ds])

        # insert event
        event_dict['event'].append('null')
        event_dict['start_time'].append(rep_s_tme.values)
        event_dict['end_time'].append(rep_e_tme.values)
        event_dict['year'].append(rep_e_tme.values.astype('datetime64[Y]').astype(int) + 1970)
        event_dict['latitude'].append(lat.values)
        event_dict['longitude'].append(lon.values)

        # insert max hail size for event
        hs = 0
        y_dict['hail_size'].append(hs)
        y_dict['severe'].append(0)

        t_count = 0
        for t in tmes:
            print('Working on null report {} of {}, at time step {} of {}'.format(null_rep + 1, num_reports, t_count + 1, len(sub_ds.time)))

            # insert lightning data for report
            x_dict['LD.t{}'.format(t_count)].append(0)

            # flatten sliced dataset
            DA = sub_ds.sel(time=t)

            for plvr in plvar:
                for lvl in lvls:
                    k = '{}_{}.t{}'.format(plvr, lvl, t_count)
                    try:
                        x_dict[k].append(DA[plvr].sel(level=int(lvl)).values)
                    except KeyError:
                        dbhelp = ''
                        raise
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
                x_dict['wet_bulb_temperature_0_deg_height.t{}'.format(t_count)].append(
                    wet_bulb_temperature_0_deg_height)

            p = DA.level[::-1] * units.hPa
            temperature = DA.t[::-1] * units.kelvins
            rh = DA.r[::-1] * units.percent
            dewpoint = dewpoint_from_relative_humidity(temperature, rh)

            # set mu-cape and -cin to 'NAN' if calculation raises error
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
                cape_depth_90, cin_depth_90 = mixed_layer_cape_cin(p, temperature, dewpoint,
                                                                   depth=90 * units.hPa)
                x_dict['cape_depth_90.t{}'.format(t_count)].append(cape_depth_90.magnitude)
                x_dict['cin_depth_90.t{}'.format(t_count)].append(cin_depth_90.magnitude)
            except:
                cape_depth_90 = float('nan')
                cin_depth_90 = float('nan')
                x_dict['cape_depth_90.t{}'.format(t_count)].append(cape_depth_90)
                x_dict['cin_depth_90.t{}'.format(t_count)].append(cin_depth_90)

            t_count += 1

        if (null_rep > 0) and (null_rep % save_freq) == 0:
            # merge dictionnaries
            x_dict.update(y_dict)
            event_dict.update(x_dict)

            # create dataframe containing all data
            df = pd.DataFrame(data=event_dict)
            fn = destination_dir + 'partial_null_ml_dataset.{}.csv'.format(num_reports)
            df.to_csv(fn, index=False)
            print('saved partial hail ds')
            print(fn)


    # run completed, save csv
    # merge dictionnaries
    x_dict.update(y_dict)
    event_dict.update(x_dict)

    # create dataframe containing all data
    df = pd.DataFrame(data=event_dict)
    fn = os.path.join(destination_dir, 'null_ml_dataset.{}.csv'.format(num_reports))
    df.to_csv(fn, index=False)

    return df

def full_ds(obs_path, eventised_era5_dir_path, sl_file_path, pl_dir_path, num_reports, destination_dir='', ini_ev=0, fin_ev=0, time_limit=5.5, severe=20, save_freq=50):
    print('Working on hail dataset.')
    hail_df = hail_ds(obs_path=obs_path, era5_dir=eventised_era5_dir_path, destination_dir=destination_dir, ini_ev=ini_ev, fin_ev=fin_ev, time_limit=time_limit, severe=severe, save_freq=save_freq)
    print('Hail dataset done.')

    print('Working on null reports dataset.')
    null_df = null_ds(sl_file_path=sl_file_path, pl_dir_path=pl_dir_path, num_reports=num_reports, destination_dir=destination_dir, ini_ev=ini_ev, fin_ev=fin_ev, time_limit=time_limit, severe=severe, save_freq=save_freq)
    print('Null dataset done.')

    print('Merging hail and null datasets.')
    full_df = pd.concat([null_df, hail_df])
    fn = os.path.join(destination_dir, 'full_ml_dataset.{}.csv'.format(num_reports))
    full_df.to_csv(fn, index=False)

    return full_df


def hail_ds_no_event(sl_file_path, pl_dir_path, obs_path, destination_dir='', ini_ev=0, fin_ev=0, time_limit=5.5, severe=20, save_freq=50):
    tic = time.perf_counter()

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

    # open single level dataset
    try:
        sl_ds = xr.open_dataset(sl_file_path)
    except ValueError as p:
        raise Exception("Path provided for 'sl_file_path' does not exist.").with_traceback(p.__traceback__)

    # load eventised data
    data = pd.read_csv(obs_path)

    # get event list
    event_list = data['Event'].where((data['Event'] >= ini_ev)).dropna()
    if fin_ev > 0:
        event_list = event_list.where(event_list <= fin_ev).dropna().astype(int).unique()
    else:
        fin_ev = max(event_list)

    for event in event_list:
        toc = time.perf_counter()
        tictoc = (toc - tic) / (60 * 60)
        print('Elapsed time {:0.2f}hrs'.format(tictoc))
        if tictoc > time_limit:
            # merge x, y, and event dictionnaries
            x_dict.update(y_dict)
            event_dict.update(x_dict)

            # create dataframe containing all data
            df = pd.DataFrame(data=event_dict)
            df.to_csv(destination_dir + 'partial_hail_ml_dataset.{}_{}.csv'.format(ini_ev, fin_ev), index=False)

        # choose random lat/lon/times
        t_len = 0
        while (t_len != 10):
            t_ind = random.choice(range(len(sl_ds.time[6:-5])))
            tmes = sl_ds.time[t_ind - 6:t_ind + 4]
            t_len = len(tmes)
        rep_s_tme = tmes[0]
        rep_e_tme = tmes[-1]
        lat = random.choice(sl_ds.latitude)
        lon = random.choice(sl_ds.longitude)

        # get year
        year = sl_ds.time[t_ind].values.astype('datetime64[Y]').astype(int) + 1970

        # slice single-level dataset
        sub_ds = sl_ds.sel(latitude=lat, longitude=lon, time=slice(rep_s_tme, rep_e_tme))

        # get file name for ERA5 data of event
        pl_files = glob.glob(pl_dir_path + '/era5_pl.{}.*.h5'.format(year))

        for pl_file in pl_files:
            pl_ds = xr.open_dataset(pl_file)
            pl_ds = pl_ds.sel(latitude=lat, longitude=lon, time=slice(rep_s_tme, rep_e_tme))
            sub_ds = xr.merge([pl_ds, sub_ds])

        # insert event
        event_dict['event'].append('null')
        event_dict['start_time'].append(rep_s_tme.values)
        event_dict['end_time'].append(rep_e_tme.values)
        event_dict['year'].append(rep_e_tme.values.astype('datetime64[Y]').astype(int) + 1970)
        event_dict['latitude'].append(lat.values)
        event_dict['longitude'].append(lon.values)

        # insert max hail size for event
        hs = 0
        y_dict['hail_size'].append(hs)
        y_dict['severe'].append(0)

        t_count = 0
        for t in tmes:
            print('Working on null report {} of {}, at time step {} of {}'.format(null_rep + 1, num_reports, t_count + 1, len(sub_ds.time)))

            # insert lightning data for report
            x_dict['LD.t{}'.format(t_count)].append(0)

            # flatten sliced dataset
            DA = sub_ds.sel(time=t)

            for plvr in plvar:
                for lvl in lvls:
                    k = '{}_{}.t{}'.format(plvr, lvl, t_count)
                    try:
                        x_dict[k].append(DA[plvr].sel(level=int(lvl)).values)
                    except KeyError:
                        dbhelp = ''
                        raise
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
                x_dict['wet_bulb_temperature_0_deg_height.t{}'.format(t_count)].append(
                    wet_bulb_temperature_0_deg_height)

            p = DA.level[::-1] * units.hPa
            temperature = DA.t[::-1] * units.kelvins
            rh = DA.r[::-1] * units.percent
            dewpoint = dewpoint_from_relative_humidity(temperature, rh)

            # set mu-cape and -cin to 'NAN' if calculation raises error
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
                cape_depth_90, cin_depth_90 = mixed_layer_cape_cin(p, temperature, dewpoint,
                                                                   depth=90 * units.hPa)
                x_dict['cape_depth_90.t{}'.format(t_count)].append(cape_depth_90.magnitude)
                x_dict['cin_depth_90.t{}'.format(t_count)].append(cin_depth_90.magnitude)
            except:
                cape_depth_90 = float('nan')
                cin_depth_90 = float('nan')
                x_dict['cape_depth_90.t{}'.format(t_count)].append(cape_depth_90)
                x_dict['cin_depth_90.t{}'.format(t_count)].append(cin_depth_90)

            t_count += 1

        if (null_rep > 0) and (null_rep % save_freq) == 0:
            # merge dictionnaries
            x_dict.update(y_dict)
            event_dict.update(x_dict)

            # create dataframe containing all data
            df = pd.DataFrame(data=event_dict)
            fn = destination_dir + 'partial_null_ml_dataset.{}.csv'.format(num_reports)
            df.to_csv(fn, index=False)
            print('saved partial hail ds')
            print(fn)


    # run completed, save csv
    # merge dictionnaries
    x_dict.update(y_dict)
    event_dict.update(x_dict)

    # create dataframe containing all data
    df = pd.DataFrame(data=event_dict)
    fn = os.path.join(destination_dir, 'null_ml_dataset.{}.csv'.format(num_reports))
    df.to_csv(fn, index=False)

    return df
































