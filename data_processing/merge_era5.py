import glob, os, re
import xarray as xr
import numpy as np
import constants_and_variables as cv

num_level_steps=len(cv.pressure_levels)

def make_dir(path, nf=0):
    new_path = path
    if nf > 0:
        new_path = '{}.{}'.format(path, nf)
    try:
        os.mkdir(new_path)
        return new_path
    except FileExistsError:
        nf += 1
        return make_dir(path, nf=nf)

def merge_pl(folder):
    files = glob.glob(folder + '/*')
    files.sort()
    dest_dir = folder + '.merged_levels'

    dest_dir = make_dir(dest_dir)

    num_of_files = len(files)
    for c in range(0, num_of_files, num_level_steps):
        ds_list = []
        for i in range(num_level_steps):
            try:
                file = files[c + i]
                event = re.search('(.*)era5_pl.event_(.*).(.).h5', file).group(2)
                ds = xr.open_dataset(file)
                ds_list.append(ds)
            except ValueError:
                print('Problem with event #' + event)
                pass
        new_file = xr.merge(ds_list)
        file_name = 'event_{}.h5'.format(event)
        name = os.path.join(dest_dir, file_name)
        new_file.to_netcdf(name, engine='h5netcdf', invalid_netcdf=True)

def merge_eventised(eventised_pressure_levels_dir, eventised_single_level_dir, destination_dir='era5.by_event', init_event=0, fin_event=0):
    pl_folder = eventised_pressure_levels_dir
    sl_folder = eventised_single_level_dir

    destination_dir = make_dir(destination_dir)

# load file names from 'eventised_pressure_levels_dir' and create list of events
    files = glob.glob(pl_folder + '/*.h5')
    files.sort()
    events = np.array([int(re.search('(.*)event_(.*).h5', file).group(2)) for file in files])
    events.sort()
    events = np.delete(events, np.where(events < init_event))
    events = np.delete(events, np.where(events > fin_event))

# for each event, open corresponding files in each directory and merge. save to 'destination_dir'
    for event in events:
        pl_files = glob.glob('{}/event_{}.h5'.format(pl_folder, event))
        ds_list = [xr.open_dataset(pl_file) for pl_file in pl_files]
        sl_files = glob.glob('{}/*era5.sl.event_{}.h5'.format(sl_folder, event))
        ds_list.append(xr.open_dataset(sl_files[0]))

        merged_file = xr.merge(ds_list)
        file_name = 'event_{}.h5'.format(event)
        merged_file_name = os.path.join(destination_dir, file_name)
        merged_file.to_netcdf(merged_file_name, engine='h5netcdf', invalid_netcdf=True)


