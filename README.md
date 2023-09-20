# era5-based-hail
Hail prediction using machine learning trained on ERA5 data

--------------------------

## Eventise hail observation data
Use 
['data_processing/eventise_data.observations'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/eventise_data.py#L18)
.

We have a csv file of 7000 hail reports from all over Canada between 2005 and 2022 called 
['hail_db_with_LD.csv'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/examples/hail_db_with_LD.csv)
.

Some of these reports are from the same hail event, i.e. a single hail storm produces multiple reports from different
locations throughout its life. So we would like to bundle reports into hail 'events' based on the time of the report
and the distances between reports. Our function 
['data_processing/eventise_data.observations'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/eventise_data.py#L18) 
does this for us.

Different reports are considered to be of the same hail event according to time and space windows which are defined in 
['constants_and_variables.py'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/constants_and_variables.py)
. Specifically, 
['data_processing/eventise_data.observations'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/eventise_data.py#L18) 
uses the constants 
'event_time_window', 
'event_spatial_window', 
'earth_rad', 
and 
['boundary_distance'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/constants_and_variables.py#L4)
. If different time and space constants are desired, changes should be made to the 
['constants_and_variables.py'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/constants_and_variables.py) 
file, keeping in mind units.


## Submitting an ERA5 request
Use ['era5_request/submit_request'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/era5_request/era5_request.py#L47). Requires an eventised observation file created using 
['data_processing/eventise_data.observations'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/eventise_data.py#L18)
.

The total ERA5 data for 2022 is ~13GB. This could take a several hours to complete depending on how busy the ERA5 servers
are with requests from other users. The code in 
['examples/era5_request.ex'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/examples/era5_request.ex.py) 
downloads ~4.4MB which takes about 20 minutes to complete depending on ERA5 server activity.


### Install CDS API key
Before anything, please see 
['How to use the CDS API'](https://cds.climate.copernicus.eu/api-how-to) 
for instructions on installing your unique CDS API key.


### ERA5 datasets
There are two datasets to choose from, 
['reanalysis-era5-single-levels'](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)
and 
['reanalysis-era5-pressure-levels'](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview)
, which have been abbreviated to 'sl' and 'pl', respectively.


### Data requests
Receiving the data you have requested from ERA5 can take some time. It depends on a couple things; how many other users
are requesting data at that time and how much data you have requested. Your request will first be queued with other
users, then ERA5 will process your request, and finally the download will begin. Each request can take about an hour
from start to finish. The request made by our function 
['era5_request.submit_request'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/era5_request/era5_request.py#L47)
is somewhat of a greedy one. We request data for all points in time and space *between* events as well as for the events themselves. This ensures gathering a healthy amount of data for examples of null-events (no hail).

Below is an example of the output for a request to the 
['reanalysis-era5-single-levels'](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview) 
ERA5 dataset for year 2022 of our hail observations, note the amount of time between timestamps of each step.

2023-09-14 10:24:12,294 INFO Welcome to the CDS

2023-09-14 10:24:12,294 INFO Sending request to reanalysis-era5-single-levels

2023-09-14 10:24:12,495 INFO Request is queued

2023-09-14 10:24:17,674 INFO Request is running

2023-09-14 11:00:42,448 INFO Request is completed

2023-09-14 11:00:42,448 INFO Downloading

2023-09-14 11:10:18,343 INFO Download rate 2.8M/s

Requests to the 
['reanalysis-era5-pressure-levels'](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview) 
ERA5 dataset are broken-up into 10 requests split by pressure levels
to ensure maximum file limits set by ERA5 on data requests are respected. See 
['constants_and_variables.py'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/constants_and_variables.py) 
for the pressure level breakdown.


## Eventise ERA5 data

Use 
['data_processing/eventise_data.era5'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/eventise_data.py#L93)
.

In this example we breakup the ERA5 data we downloaded in the 
['examples/era5_request.ex'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/examples/era5_request.ex.py) 
example into events according to
the events we created for the hail reports in the 
['examples/eventise_observations.ex.py'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/examples/eventise_observations.ex.py) 
example. The data surrounding an event
in the single level and pressure level ERA5 data are merged into a single file and saved to a directory called 'era5.by_event'. A parent directory to 'era5.by_event' can be specified by using the argument 'destination_dir_path' of the 
['data_processing/eventise_data.era5'](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/eventise_data.py#L93) 
function.

Opening the '.h5' files acquired from ERA5 may require installing additional dependencies to xarray. I use PyCharm on
my local machine and had to pass the following installation command:

' pip install "xarray[io]" '

If you think this may be necessary, open the 'ERA5_Based_Hail' directory in your command prompt and enter the command:

'venv\Scripts\activate' (if you are a Windows user)

Then enter the installation command above. For more information see [Xarray installation](https://docs.xarray.dev/en/latest/getting-started-guide/installing.html).
