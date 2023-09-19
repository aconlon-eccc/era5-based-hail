"""
------------------
Eventise ERA5 data
------------------
Use 'ERA5_Based_Hail.data_processing.eventise_data.era5'.

In this example we breakup the ERA5 data we downloaded in the 'era5_request.ex.py' example into events according to
the events we created for the hail reports in the 'eventise_observations.ex.py' example.

Opening the '.h5' files acquired from ERA5 may require installing additional dependencies to xarray. I use PyCharm on
my local machine and had to pass the following installation command:

'pip install "xarray[io]"'

If you think this may be necessary, open the 'ERA5_Based_Hail' directory in your command prompt and enter the command:

'venv\Scripts\activate'

Then enter the installation command above. For more information on Xarray installation see:

https://docs.xarray.dev/en/latest/getting-started-guide/installing.html
"""

import data_processing.eventise_data as eventise
import data_processing.load_eventised_observations as load_obs

# variables that require assignments, others are optional
era5_dataset = 'sl'
year = 2022
eventised_observations = load_obs.load_obs('eventised_obs.csv')
era5_dir_path = 'era5_sl.2022'

# eventise the single-level ('sl') era5 data for 2022
eventise.era5(era5_dataset=era5_dataset, year=year, eventised_observations=eventised_observations,
              era5_dir_path=era5_dir_path, destination_dir_path='', init_event=2898, init_level_step=0)

# eventise the pressure-level ('pl') era5 data for 2022
era5_dataset = 'pl'
era5_dir_path = 'era5_pl.2022'
eventise.era5(era5_dataset=era5_dataset, year=year, eventised_observations=eventised_observations,
              era5_dir_path=era5_dir_path, destination_dir_path='', init_event=2898, init_level_step=0)