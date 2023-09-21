"""
**Creating a dataset for machine-learning**

Use 'data_processing/create_ml_dataset.build_hail_ds_nans'

Here we use the eventised ERA5 data to populate a csv file that describes hourly conditions up to six hours before and
three after the start time of each hail report. The csv file created has 1268 columns which includes location and hail
severity information.

Building the csv file for all 7000 hail reports is a long process, so I included a 'time_limit' argument to
'build_hail_ds_nans' to make sure processes are saved before being timed out on whatever system you are using. Note
that 'time_limit' is set to 5.5 hours by default because of the system I use. You will need to find out the time limit
of your system (if it has one) and set 'time_limit' accordingly.

One can also split this job into chunks by using the initial and final event arguments 'ini_ev' and 'fin_ev',
respectively.
"""

import data_processing.create_ml_dataset as cmd

cmd.hail_ds('eventised_obs.csv', 'era5_2022.by_event.10', ini_ev=2898)