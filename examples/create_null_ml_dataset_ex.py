from data_processing import create_ml_dataset as cd

cd.full_ds(obs_path='eventised_obs.csv',
           eventised_era5_dir_path='era5_2022.by_event.10',
           sl_file_path='era5_sl.2022.3/era5_sl.2022.h5',
           pl_dir_path='era5_pl.2022.1',
           num_reports=4,
           ini_ev=2898)