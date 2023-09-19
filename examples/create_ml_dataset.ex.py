import data_processing.create_ml_dataset as cmd

cmd.build_hail_ds_nans('eventised_obs.csv', 'era5.by_event', ini_ev=2898)