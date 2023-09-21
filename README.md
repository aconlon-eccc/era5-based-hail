# era5-based-hail
Hail prediction using machine learning trained on `ERA5` data

## Workflow
This README has been aranged in the indended workflow. Here it is at a glance: 

> 1. eventise hail observations
> 2. submit ERA5 requests 
> 3. eventise ERA5 data 
> 4. create machine-learning dataset

## Eventise hail observation data
Use 
['data_processing/eventise_data.observations'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/eventise_data.py#L18)
.

We have a csv file containing 7000 hail reports from all over Canada between 2005 and 2022 called 
['hail_db_with_LD.csv'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/examples/hail_db_with_LD.csv)
.

Some of these reports are from the same hail event, that is a single hail storm produces multiple reports from different
locations. We would like to bundle reports into hail 'events' based on the time of the report and the distances between reports. Our function 
['data_processing/eventise_data.observations'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/eventise_data.py#L18) 
does this for us. I have provided the example
['examples/eventise_observations_ex](https://github.com/aconlon-eccc/era5-based-hail/blob/master/examples/eventise_observations_ex.py)
.

Different reports are considered to be of the same hail event according to time and space windows which are defined in 
['constants_and_variables'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/constants_and_variables.py)
. Specifically, 
['data_processing/eventise_data.observations'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/eventise_data.py#L18) 
uses the constants 
['event_time_window'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/constants_and_variables.py#L1), 
['event_spatial_window'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/constants_and_variables.py#L2), 
['earth_rad'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/constants_and_variables.py#L3), 
and 
['boundary_distance'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/constants_and_variables.py#L4)
. If different time and space constants are desired, changes should be made to the 
['constants_and_variables'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/constants_and_variables.py) 
file, keeping in mind units.


## Submitting an ERA5 request
Use 
['era5_request/submit_request'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/era5_request/era5_request.py#L47)
. Requires an eventised observation file created using 
['data_processing/eventise_data.observations'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/eventise_data.py#L18)
see
['examples/eventise_observations_ex](https://github.com/aconlon-eccc/era5-based-hail/blob/master/examples/eventise_observations_ex.py) 
for an example of how to use.

The total ERA5 data download for 2022 is ~13GB. This could take a several hours to complete depending on how busy the ERA5 servers
are with requests from other users. The code in 
['examples/era5_request_ex'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/examples/era5_request.ex.py) 
downloads ~4.4MB which takes about 20 minutes to complete depending on ERA5 server activity. The example downloads ERA5 data for only two events in 2022 instead of downloading data for all 287 events in 2022.


### Install CDS API key
Before anything, see 
['How to use the CDS API'](https://cds.climate.copernicus.eu/api-how-to) 
for instructions on installing your unique CDS API key.


### ERA5 datasets
There are two datasets to choose from, 
[`reanalysis-era5-single-levels`](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)
and 
[`reanalysis-era5-pressure-levels`](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview)
, which have been abbreviated to 'sl' and 'pl', respectively.


### Data requests
Receiving the data you have requested from ERA5 can take some time. It depends on a couple things; how many other users
are requesting data at that time and how much data you have requested. Your request will first be queued with other
users, then ERA5 will run your request, and finally the download will begin. The request made by our function 
['era5_request.submit_request'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/era5_request/era5_request.py#L47)
is somewhat of a greedy one. We request data for all points in time and space *between* events as well as for those for the events themselves. This ensures that we gather gathering a healthy amount of null-event data (no-hail) for training.

Below is an example of the output for a ~1.3GB request submitted to the 
['reanalysis-era5-single-levels'](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview) 
ERA5 dataset for year 2022 of our hail observations, note the amount of time between timestamps of each step.

> 2023-09-14 10:24:12,294 INFO Welcome to the CDS
> 
> 2023-09-14 10:24:12,294 INFO Sending request to reanalysis-era5-single-levels
> 
> 2023-09-14 10:24:12,495 INFO Request is queued
> 
> 2023-09-14 10:24:17,674 INFO Request is running
> 
> 2023-09-14 11:00:42,448 INFO Request is completed
> 
> 2023-09-14 11:00:42,448 INFO Downloading
> 
> 2023-09-14 11:10:18,343 INFO Download rate 2.8M/s

Requests to the 
['reanalysis-era5-pressure-levels'](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview) 
ERA5 dataset are broken-up into 10 requests split by pressure levels to ensure maximum file limits set by ERA5 on data requests are respected. See 
['constants_and_variables.pressure_levels'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/constants_and_variables.py#L30) 
for the pressure level breakdown.


## Eventise ERA5 data

Use 
['data_processing/eventise_data.era5'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/eventise_data.py#L93)
.

In the example 
['examples/eventise_era5_ex'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/examples/eventise_era5_ex.py) 
we breakup the ERA5 data we downloaded in the 
['examples/era5_request_ex'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/examples/era5_request_ex.py) 
example into events according to the events we created for the hail reports in the 
['examples/eventise_observations_ex'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/examples/eventise_observations_ex.py) 
example. The data surrounding an event in the single-level and pressure-level ERA5 data are merged into a single event file and saved to a directory called 'era5.by_event'. A parent directory to 'era5.by_event' can be specified by using the argument 'destination_dir_path' of the 
['data_processing/eventise_data.era5'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/eventise_data.py#L93) 
function.

Opening the '.h5' files acquired from ERA5 may require installing additional dependencies to xarray. I use PyCharm on
my local machine and had to pass the following installation command:

    ' pip install "xarray[io]" '

If you think this may be necessary, open the 'ERA5_Based_Hail' directory in your command prompt and enter the command (if you are a Windows user):

    'venv\Scripts\activate'

Then enter the installation command above. For more information see `Xarray` 
[installation](https://docs.xarray.dev/en/latest/getting-started-guide/installing.html)
.

## Creating a dataset for machine-learning

Use 
['data_processing/create_ml_dataset.hail_ds'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py)

In the example 
['examples/create_ml_dataset_ex'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/examples/create_ml_dataset_ex.py) 
, we use the eventised ERA5 data to populate a csv file that describes hourly conditions up to six hours before and three after the start time of each hail report. The csv file created has 1268 columns which includes location and hail severity information.

Building the csv file for all 7000 hail reports is a long process, so 
[hail_ds'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py)
saves its progress to a csv file called 'partial_ml_dataset.{ini_ev}_{fin_ev}.csv' in the user-specified directory 
['destination_ir'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L13) 
for every 5 reports it processes. To change the frequency at which progress is saved, set the argument
'save_freq'
to desired frequency.

so I included a 
['time_limit'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L13) 
argument to 
['hail_ds'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py) 
. When the run-time reaches the time limit, it saves the csv file as-is, to a file called 
'partial_ml_dataset.{ini_ev}__{fin_ev}.csv' 
to make sure processes are saved. The process can be picked up by checking the partially completed csv file for the last completed event (all reports in the event were processed) and setting 
['ini_ev'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L13) 
to that event + 1. 
['build_hail_ds_nans'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py) 
also saves its progress to a csv file called 'partial_ml_dataset.{ini_ev}_{fin_ev}.csv' for every 50 reports it processes. To change the frequency at which progress is saved, change to desired frequency 
[here](https://github.com/aconlon-eccc/era5-based-hail/blob/e0beeae18fb9b6b8c882ec937440f6fd247970bf/data_processing/create_ml_dataset.py#L254C17-L254C20)
.
Note that 
['time_limit'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L13) 
is set to 5.5 hours by default because the system I use has a six hour job-time limit. You will need to find out the time limit of your system (if it has one) and set 
['time_limit'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L13) 
accordingly. 

One can also split this job into chunks by using the initial and final event arguments 
['ini_ev'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L13) 
and 
['fin_ev'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L13)
, respectively. 
