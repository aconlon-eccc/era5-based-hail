"""
Eventise ERA5 data

==================

Use 'ERA5_Based_Hail.data_processing.eventise_data.era5'.

In this example we breakup the ERA5 data we downloaded in the 'era5_request_ex.py' example into events according to
the events we created for the hail reports in the 'eventise_observations_ex.py' example. The data surrounding an event
in the single level and pressure level ERA5 data are merged into a single file and saved to a directory called
'era5.by_event'. A parent directory can be specified for 'era5.by_event' using the argument
'destination_dir_path' of the 'ERA5_Based_Hail.data_processing.eventise_data.era5' function.

Opening the '.h5' files acquired from ERA5 may require installing additional dependencies to xarray. I use PyCharm on
my local machine and had to pass the following installation command:

' pip install "xarray[io]" '

If you think this may be necessary, open the 'ERA5_Based_Hail' directory in your command prompt and enter the command:

'venv\Scripts\activate' (if you are a Windows user)

Then enter the installation command above. For more information on Xarray installation see:

https://docs.xarray.dev/en/latest/getting-started-guide/installing.html
"""

import data_processing.eventise_data as eventise

year=2022
obs_path='eventised_obs.csv'
init_event=2898

eventise.era5(year=year, eventised_observations_path=obs_path, init_event=2898)

