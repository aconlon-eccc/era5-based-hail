# era5-based-hail
Hail prediction using machine learning (ML) trained on ERA5 data.

In 2023 I was tasked with building a hail prediction and detection model from scratch for the High Impact Weather Research team of the Atmospheric Science and Technology directorate at Environment and Climate Change Canada. My task was heavy; organize our observational data, retrieve, clean and organize large datasets from ECMWF for training and testing, choosing ML models to build, and being responsible for the design of every aspect of my approach along the way. This repository contains all the relevant code that I wrote to accomplish my task, complete with examples.

In this README, the first section is aranged in the indended workflow to go from unorganized hail observation data to having structured observational data and large meteorological datasets for ML training and testing. The second section describes potentially useful functions that I created in my exploratory phase of processing data but that are no longer part of my main workflow.

---
## Table of contents
 1. Main workflow 
     1. [Eventise hail observations](#eventise-hail-observation-data)
     2. [ERA5 data download](#era5-data-download)
         1. [Install CDS API key](#install-cds-api-key)
         2. [ERA5 datasets](#era5-datasets)
         3. [Submiting ERA5 data requests](#submiting-era5-data-requests)
         4. [Details on ERA5 data](#details-on-era5-data)
     3. Create machine-learning datasets
 2. Other functions
     1. Eventise ERA5 data

---
## Eventise hail observation data

We have a CSV file containing 7000 hail reports from all over Canada between 2005 and 2022 called 
[`integrated_canadian_hail_db.csv`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/examples/integrated_canadian_hail_db.csv) that looks like the following:

| Start | Time	| Year	| Month	| Day	| Hour	| Longitude	| Latitude	| Province Code	| Reference Object |	Hail Diameter (mm) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|
|5/15/2005 |21:15	|2005	|5	|15	|21	|-109.37	|52.7	|SK	|pea	|8|
|5/23/2005 |21:05	|2005	|5	|23	|21	|-104	|53.45	|SK	|pea	|8|
|5/31/2005 |1:45	|2005	|5	|31	|1	|-118.43	|55.17	|AB	|pea	|8|
|6/1/2005 |21:40	|2005	|6	|1	|21	|-100.38	|50.58	|MB	|pea	|8|
|6/1/2005 |22:00	|2005	|6	|1	|22	|-100.22	|50.58	|MB	|pea	|8|
|6/25/2005 |20:30	|2005	|6	|25	|20	|-108.1	|50.38	|SK	|quarter	|24|
|7/3/2005 |1:20	|2005	|7	|3	|1	|-98.9	|49.2	|MB	|apple	|70|
|7/13/2005 |23:05	|2005	|7	|13	|23	|-111.12	|52.02	|AB	|pea	|8|
|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|



Some of these reports are from the same hail event, that is, a single hail storm produces multiple reports varying slightly in time and location. We would like to bundle reports into hail events based on the time of the report and the distances between reports - hence, 'eventise'. The function 
```python
def observations(file_name, destination_file_name=''):
```
found here [`data_processing/eventise_data.observations`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/eventise_data.py#L18), does this by first ordering the observational data by `Start Time` then calculating the (arc) distance, `Cosine Distances`, between a report location and the location of the previous location. It then uses the variables [`event_time_window`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/constants_and_variables.py#L1) and [`event_spatial_window`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/constants_and_variables.py#L2) to determine wether or not a report and the next report are part of the same hail event and assigns an event number to each report; reports have overlapping time-and-space windows are deemed to be of the same event and receive the same event number;

| Start | Time	| Year	| Month	| Day	| Hour	| Longitude	| Latitude	| Province Code	| Reference Object |	Hail Diameter (mm) | Cosine Distances | Event |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|4/18/2005 | 9:15 | 2005 | 4 | 18 | 9 | -96.7 |49.72 | MB | marble | 16.0 | 0.0 | 0 |
|5/7/2005 | 2:40 | 2005 | 5 | 7 | 2 | -113.55 | 51.65 | AB | quarter | 24.0 | 1203.5186663512313 | 1 |
|5/15/2005 | 21:00 | 2005 | 5 | 15 | 21 | -109.47 | 53.1 | SK | pea | 8.0 | 320.40683317623564 | 2 |
|5/15/2005 | 21:15 | 2005 | 5 | 15 | 21 | -109.37 | 52.7 | SK | pea | 8.0 | 44.980919111703585 | 2 |
|5/17/2005 | 0:15 | 2005 | 5 | 17 | 0 | -113.05 | 53.02 | AB | dime | 18.0 | 249.57974229868645 | 3 |
|5/19/2005 | 19:52 | 2005 | 5 | 19 | 19 | -109.53 | 54.32 | SK | nickel | 21.0 | 273.20214854901093 | 4 |
|5/19/2005 | 19:55 | 2005 | 5 | 19 | 19 | -109.61 | 54.51 | SK | estimated 19mm | 19.0 | 21.751971807038505 | 4 |
|5/20/2005 | 0:41 | 2005 | 5 | 20 | 0 | -113.5 | 53.6 | AB | pea | 8.0 | 273.2833425267738 | 5 | 
|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|

The result is returned as a `pandas.DataFrame` and a CSV copy is saved under the `destination_file_name` which defaults to `eventised_obs.csv` if no file name is specified.

---
## ERA5 data download
### Install CDS API key
Before starting anything, see 
['CDSAPI setup'](https://cds.climate.copernicus.eu/how-to-api)
for instructions on installing your unique CDS API key.

---
### ERA5 datasets
In this workflow we download from the two ERA5 datasets:
[`reanalysis-era5-single-levels`](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)
and 
[`reanalysis-era5-pressure-levels`](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview) which I abbreviated to `sl` and `pl`, respectively. The variables we download from the `sl` dataset are:
```python
# variables requested when submitting an ERA5 request from the 'reanalysis-era5-single-levels' dataset
single_level_variables = [
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            '2m_temperature',
            '2m_dewpoint_temperature',
            'surface_pressure',
            'total_column_water',
            'total_precipitation',
            'convective_precipitation',
            'total_column_water_vapour',
            'total_column_rain_water',
            'total_column_snow_water',
            'total_column_cloud_ice_water',
            'total_column_cloud_liquid_water',
            'total_cloud_cover']
```
and the variables download from the `pl` dataset are: 
```python
# variables requested when submitting an ERA5 request from the 'reanalysis-era5-pressure-levels' dataset
pressure_level_variables = [
            'geopotential', 
            'relative_humidity', 
            'temperature', 
            'u_component_of_wind',
            'v_component_of_wind', 
]
```
accross twenty pressure levels from 300 to 1000 hPa: 
```python
# pressure levels
lvls = ['300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '775', '800', '825', '850', '875',
        '900', '925', '950', '975', '1000']
```
The code snippets above are taken from [constants_and_variables.py](https://github.com/aconlon-eccc/era5-based-hail/blob/master/constants_and_variables.py).

---
### Submiting ERA5 data requests
The function used for submitting ERA5 data download requests is 
[`era5_request/era5_request.submit_request`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/era5_request/era5_request.py#L47)
:
```python
def submit_request(era5_dataset, year, eventised_observations, destination_dir='', init_pressure_level=0, init_event=0, fin_event=0):
```
The download acquires data in netcdf format. Note that the `submit_request` function requires an eventised observation file created using 
[`data_processing/eventise_data.observations`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/eventise_data.py#L18) as described in section 1.i. 

Receiving the requested data can take some time and depends on a couple things: 
- the number of current requests from other users
- how much data you have requested.

A request goes through the following steps: 
1. the request is put into a queue
2. the ERA5 server then runs your request once it has reached the front of the queue - running the request can take the server some time depending on the size of your request
3. finally, the download begins.

There are few approaches we can take when it comes to submitting requests: 
1. The granular approach: submit requests for data bound by each event;
   - leaner data, faster data prep on the server side, smaller download sizes, but results in many queues (users can only make one request at a time) and fewer null-event data (no-hail) to build training and testing datasets.
3. The greedy approach: submit a request data covering an entire season (year) and all event locations;
   - more null-event data, fewer queues, but larger download sizes, and slower data prep on the server side. 

From what I learned, it's faster to make a few greedy requests than it is to go the granular route - the queues can take a long time. 

The `submit_request` function builds a box around the min/max lat/long and the min/max times of the reports for a given year and submits a request to the ERA5 servers for that dataset. This means we gather all data points in time and space *between* events as well as for those for the events themselves.  

Here is the example I provided in the [examples](https://github.com/aconlon-eccc/era5-based-hail/tree/master/examples) folder where two requests are made; where a request is made for the dataset related to the final two events of year 2022: 
```python
from data_processing import load_eventised_observations as load_obs
from era5_request import era5_request as er

# load eventised observations data
eventised_observations = load_obs.load_obs('eventised_obs.csv') # for details see 'eventise_observations_ex.py'

# request data from the ERA5 'reanalysis-era5-single-levels' ('sl') for year 2022
er.submit_request('sl', 2022, eventised_observations, init_event=2898)

# request data from the ERA5 'reanalysis-era5-pressure-levels' ('pl') for year 2022
er.submit_request('pl', 2022, eventised_observations, init_event=2898)
```
When making a request to the `pl` database, the request is broken up into 10 seperate requests according to pressure level in order to respect ERA5's limits on dataset size.

Below is an example of the output for a request submitted to the [`reanalysis-era5-single-levels`](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview) dataset covering all events of year 2022. The total size of the request was about 1.3GB. Note the amount of time elapsed between timestamps.
```
2023-09-14 10:24:12,294 INFO Welcome to the CDS
2023-09-14 10:24:12,294 INFO Sending request to reanalysis-era5-single-levels
2023-09-14 10:24:12,495 INFO Request is queued
2023-09-14 10:24:17,674 INFO Request is running
2023-09-14 11:00:42,448 INFO Request is completed
2023-09-14 11:00:42,448 INFO Downloading
2023-09-14 11:10:18,343 INFO Download rate 2.8M/s
```

The total ERA5 data download for 2022 is **~13GB**. This can take a several hours to complete depending on how busy the ERA5 servers
are. Requests to the 
[`reanalysis-era5-pressure-levels`](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview) 
ERA5 dataset are broken-up into 10 requests split by pressure levels to ensure maximum file limits set by ERA5 on data requests are respected. See 
[`constants_and_variables.pressure_levels`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/constants_and_variables.py#L30) 
for the pressure level breakdown. 

In the example 
[`examples/era5_request_ex`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/examples/era5_request_ex.py) 
seen below, we limit the download to ~4.4MB which takes about 20 minutes to complete depending on ERA5 server activity. The example downloads ERA5 data only for the last two events of 2022, `Event 2898` and `Event 2899`, instead of all 287 events in 2022.
```python
from data_processing import load_eventised_observations as load_obs
from era5_request import era5_request as er

# load eventised observations data
eventised_observations = load_obs.load_obs('eventised_obs.csv') # for details see 'eventise_observations_ex.py'

# request data from the ERA5 'reanalysis-era5-single-levels' ('sl') for year 2022
er.submit_request('sl', 2022, eventised_observations, init_event=2898)

# request data from the ERA5 'reanalysis-era5-single-levels' ('pl') for year 2022
er.submit_request('pl', 2022, eventised_observations, init_event=2898)
```
<!--
---
### Details on ERA5 data

The hail events created `eventise_data.observations` serve as temporal boundaries for each year of interest. For example, in 2022 the first and last events are `Event 2612` and `Event 2899` and are comprised of a single hail reports with the following relavent information:

| Event | Start Time | End Time | Latitude | Longitude | Year |
| ----- | ---------- | -------- | -------- | --------- | ---- |
| 2612 | 4/22/2022  4:00:00 PM | 4/22/2022  6:00:00 PM | 44.37701 | -64.31884 | 2022 | 
| $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ |
| 2899 | 8/28/2022  12:00:00 AM | 8/28/2022  3:00:00 AM | 51.5285 | -111.20515 | 2022 | 

The `submit_request` function will obtain data for every day from `4/22/2022` to `8/28/2022`, inclusive. If `Start Time` of the first event is within 7 hours of the previous day, `submit_request` will set the previous day as the first day of data to obtain. Similarly, if `End Time` of the final event is within 7 hours of the next day, `submit request` will set the next day as the final day of data to obtain. 

Spatial boundaries for the data request as obtained in the following way: 
```python
# finding min and max longitudes of reports in the event 'ev_num'
lons = eventised_hail_data['Longitude'].iloc[year_inds]
max_lon = max(lons) + np.rad2deg(boundary_dist)
min_lon = min(lons) - np.rad2deg(boundary_dist)
# finding min and max latitudes of reports in the event 'ev_num'
lats = eventised_hail_data['Latitude'].iloc[year_inds]
max_lat = max(lats) + np.rad2deg(boundary_dist)
min_lat = min(lats) - np.rad2deg(boundary_dist)
```
where `boundary_dist` the spatial window given in radians calculated using:
```python
boundary_distance = event_spatial_window / (2 * earth_rad)  # spatial window in radians around reports for training data
```
and where `event_spatial_window` and `earth_rad` are defined in `km` in `constants_and_variables`. 
-->
---
## Creating datasets for machine-learning
Opening the `.h5` files acquired from ERA5 may require installing additional dependencies to `Xarray`. I use `PyCharm` on
my local machine and had to pass the following installation command:
```
pip install "xarray[io]"
```
Here is an exerpt from the `Xarray` 
[installation](https://docs.xarray.dev/en/latest/getting-started-guide/installing.html)
page that mentions the potential need for the line above

> We also maintain other dependency sets for different subsets of functionality:
> ```
> python -m pip install "xarray[io]"        # Install optional dependencies for handling I/O
> python -m pip install "xarray[accel]"     # Install optional dependencies for accelerating xarray
> python -m pip install "xarray[parallel]"  # Install optional dependencies for dask arrays
> python -m pip install "xarray[viz]"       # Install optional dependencies for visualization
> python -m pip install "xarray[complete]"  # Install all the above
> ```


If you think this may be necessary, open the 'ERA5_Based_Hail' directory in your command prompt and enter the command (for Windows users):
```
venv\Scripts\activate
```
Then enter the installation command above. For more information see `Xarray` 
[installation](https://docs.xarray.dev/en/latest/getting-started-guide/installing.html)
.

The [`create_ml_dataset`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py) functions use the ERA5 data is to compute the following meteorological values:
```python
# calculated indices
indices = ['bulk_shear_0_6_km', 'bulk_shear_0_3_km', 'bulk_shear_0_1_km', 'mean_wind_1_3_km', 'lapse_rate_0_3_km',
           'lapse_rate_2_6_km', 'wet_bulb_temperature_0_deg_height', 'cape', 'cin', 'cape_depth_90', 'cin_depth_90']
```
They also use the observational data to associate a Boolean value to each event to indicate the severity of the hail (larger than 20 cm is considered severe by default); 1 for hail size larger than 20 cm, 0 for hail size smaller or equal to 20 cm. These computed values along with the ERA5 data, and our observation data, are used to populate a CSV file. It describes hourly conditions up to six hours before (`T-6`) and three hours after (`T+3`) the start time (`T`) of each hail report, producing a table with 1268 columns.
<!--
Below are the variables we collect for each time step, `[T-6, T+3]` directly from the downloaded ERA5 data:
```python
# pressure-level dependent variables
plvar = ['r', 't', 'u', 'v', 'z']

# single-level variables
slvar = ['cp', 'd2m', 'sp', 't2m', 'tcc', 'tciw', 'tclw', 'tcrw', 'tcsw', 'tcw', 'tcwv', 'tp', 'u10', 'v10']
```

Recall that the `plvar` variables have a value for each pressure level in `lvls`: 
```python
# pressure levels
lvls = ['300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '775', '800', '825', '850', '875',
        '900', '925', '950', '975', '1000']
```
Variables that are collected for each timestep `[T-6, T+3]` are given the suffix `.ti` where `i = 0, 1, ..., 9`. For example, the vlaues for `relative_humidity` at pressure level `500 hPa` and timestep `2` are in column `r_500.t2`. These variables account for 1260 columns. The remaining 8 are: 
```python
remaining_eight = ['event', 'year', 'start_time', 'end_time', 'latitude', 'longitude', 'hail_size', 'severe']
```
-->

### Create a dataset for predicting hail size and severity classification given a hail event
Use 
[`data_processing/create_ml_dataset.hail_ds`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py)
;
```python
def hail_ds(sl_dir_loc='', pl_dir_loc='', obs_file_path='', destination_dir='', ini_year=0, fin_year=0, ini_ev=0, fin_ev=0, time_limit=5.5, severe=20, save_freq=50)
```

The example 
[`examples/create_hail_ml_dataset_ex`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/examples/create_hail_ml_dataset_ex.py) 
is shown here:
```python
import data_processing.create_ml_dataset as cd
cd.hail_ds(ini_ev=2898)
```

Building the CSV file for all 7000 hail reports is a long process, so 
[`hail_ds`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py)
saves its progress to a csv file called 'partial_ml_dataset.{ini_ev}_{fin_ev}.csv' in the user-specified directory 
[`destination_dir`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L13) 
for every 50 reports it processes. To change the frequency at which progress is saved, set the argument
[`save_freq`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L13)
to desired frequency. Note that 
[`time_limit`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L13) 
is set to 5.5 hours by default because the system I use has a six hour job-time limit. You will need to find out the time limit of your system (if it has one) and set 
[`time_limit`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L13) 
accordingly. When the run-time reaches the time limit, a CSV file saves the progress as-is, to a file called 
[`partial_ml_dataset.{ini_ev}__{fin_ev}.csv`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/examples/ml_dataset.2898_2899.csv) 
to make sure processes are saved. The process can be picked up again by checking the partially completed CSV file for the last completed event (all reports in the event were processed) and setting 
[`ini_ev`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L13) 
to that event + 1.  

One can also split this job into chunks by using the initial and final event arguments 
[`ini_ev`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L13) 
and 
[`fin_ev`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L13)
, respectively. 

---
### Datasets for hail detection from meteorological data
#### Null-case dataset
Use 
[`data_processing/create_ml_dataset.null_ds`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L275)
;
```python
def null_ds(num_reports, sl_dir_loc='', pl_dir_loc='', destination_dir='', time_limit=5.5, save_freq=50)
```

The `null_ds` function randomly samples a `latitude`, a `longitude`, and a consecutive set of ten timestamps from the single-level ERA5 NetCDF through the user-provided `sl_file_path`. It then creates sub-datasets of the single-level and pressure-level ERA5 data using the randomly sampled information and merges these sub-datasets. The merged sub-dataset is used to fill-in a dataframe with the same columns as the one in `hail_ds` (see above). The attribute `num_reports` lets the user specify how many null-case reports they would like to create. Similar to `hail_ds`, `null_ds` saves it's progress according to `save_freq`,  as `partial_null_ml_dataset.{num_reports}.csv` in `destination_dir`, and has a `time_limit` variable for those of us with run-time limits on our machines. `null_ds` returns the dataframe when finished and saves the final dataframe as a CSV file called `null_ml_dataset.{num_reports}.csv`.

---
#### Full dataset (hail & null cases)
Use
[`data_processing/create_ml_dataset.full_ds`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L518)
;
```python
def full_ds(num_reports, sl_dir_loc='', pl_dir_loc='', obs_file_path='', destination_dir='', ini_year=0, fin_year=0, ini_ev=0, fin_ev=0, time_limit=5.5, severe=20, save_freq=50)
```
The `full_ds` function combines `hail_ds` and `null_ds` functions to create a dataset of both hail and null cases. It returns the dataset as a `pandas.DataFrame` when finished and saves the full dataset as `full_ml_dataset{num_reports}.csv` in `destination_dir`, where `num_reports` is the number of null-cases. 

---
## Other functions
### Eventise ERA5 data

Use 
[`data_processing/eventise_data.era5`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/eventise_data.py#L93)
.
```python
def era5(year, eventised_observations_path, sl_file_path, pl_dir_path, destination_dir_path='', init_event=0, fin_event=0):
```

In the example 
[`examples/eventise_era5_ex`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/examples/eventise_era5_ex.py) 
we breakup the downloaded ERA5 data from
[`examples/era5_request_ex`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/examples/era5_request_ex.py) 
into events according to the events we created for the hail reports in the 
[`examples/eventise_observations_ex`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/examples/eventise_observations_ex.py) 
example. The data surrounding an event in the single-level and pressure-level ERA5 data are merged into a single event file and saved to a directory called 'era5_{year}.by_event'. A parent directory to 'era5_{year}.by_event' can be specified by using the argument 'destination_dir_path' of the 
[`data_processing/eventise_data.era5`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/eventise_data.py#L93) 
function.


