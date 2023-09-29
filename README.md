# era5-based-hail
Hail prediction using machine learning trained on `ERA5` data.

In this README, the first section titled `Main workflow` has been aranged in the indended workflow to get from hail observation data to creating a large dataset for machine-learning applications. The second section title `Other functions` describes other potentially usefull functions that I created while deciding the best course of action for processing data.

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
Use 
[`data_processing/eventise_data.observations`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/eventise_data.py#L18)
;
```python
def observations(file_name, destination_file_name=''):
```

We have a CSV file containing 7000 hail reports from all over Canada between 2005 and 2022 called 
[`hail_db_with_LD.csv`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/examples/hail_db_with_LD.csv)
.

Some of these reports are from the same hail event, that is a single hail storm produces multiple reports from different
locations. We would like to bundle reports into hail 'events' based on the time of the report and the distances between reports. Our function 
[`data_processing/eventise_data.observations`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/eventise_data.py#L18) 
does this for us by adding an `Event` column to `hail_db_with_LD.csv` and saving the result as `eventised_obs.csv`. The result filename can be specified by the user by specifying `destination_file_name`. This function also returns the result as a `pandas.DataFrame`. I have provided the example
[`examples/eventise_observations_ex`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/examples/eventise_observations_ex.py)
:
```python
import data_processing.eventise_data as ed

eventised_hail_data = ed.observations('hail_db_with_LD.csv')
```

Different reports are considered to be of the same hail event if they are within a specified time and space windows, defined in 
[`constants_and_variables`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/constants_and_variables.py)
. Specifically, the
[`observations`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/eventise_data.py#L18) 
function uses the constants 
[`event_time_window`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/constants_and_variables.py#L1), 
[`event_spatial_window`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/constants_and_variables.py#L2), 
[`earth_rad`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/constants_and_variables.py#L3), 
and 
[`boundary_distance`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/constants_and_variables.py#L4)
. If different time and space constants are desired, changes can be made to the 
[`constants_and_variables`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/constants_and_variables.py) 
file, keeping in mind units. 

---
## ERA5 data download
### Install CDS API key
Before staring anything, see 
['How to use the CDS API'](https://cds.climate.copernicus.eu/api-how-to) 
for instructions on installing your unique CDS API key.

---
### ERA5 datasets
There are two datasets that we download from, 
[`reanalysis-era5-single-levels`](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)
and 
[`reanalysis-era5-pressure-levels`](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview)
, which have been abbreviated to `sl` and `pl`, respectively.

---
### Submiting ERA5 data requests
Use 
[`era5_request/era5_request.submit_request`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/era5_request/era5_request.py#L47)
;
```python
def submit_request(era5_dataset, year, eventised_observations, destination_dir='', init_pressure_level=0, init_event=0, fin_event=0):
```
This requires an eventised observation file created using 
[`data_processing/eventise_data.observations`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/eventise_data.py#L18)
.

Receiving the data you have requested from ERA5 can take some time. It depends on a couple things; how many other users
are requesting data at that time and how much data you have requested. Your request will first be queued with other
users, then ERA5 will run your request, and finally the download will begin. The request made by our function 
[`era5_request.submit_request`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/era5_request/era5_request.py#L47)
is somewhat of a greedy one; we request data for all points in time and space *between* events as well as for those for the events themselves. This ensures that we gather gathering a healthy amount of null-event data (no-hail) for training, and other future applications.

Below is an example of the output for a ~1.3GB request submitted to the 
[`reanalysis-era5-single-levels`](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview) 
ERA5 dataset for year 2022 of our hail observations, note the amount of time between timestamps of each step.
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
### For hail size and classification
Use 
[`data_processing/create_ml_dataset.hail_ds`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py)
;
```python
def hail_ds(obs_path, era5_dir, destination_dir='', ini_ev=0, fin_ev=0, time_limit=5.5, severe=20, save_freq=50):
```

In the example 
[`examples/create_hail_ml_dataset_ex`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/examples/create_hail_ml_dataset_ex.py) 
,
```python
import data_processing.create_ml_dataset as cmd

cmd.hail_ds('eventised_obs.csv', 'era5_2022.by_event.10', ini_ev=2898)
```
, we use the eventised ERA5 data to populate a csv file that describes hourly conditions up to six hours before and three after the start time of each hail report. The csv file created has 1268 columns which includes location and hail severity information.

Building the csv file for all 7000 hail reports is a long process, so 
[`hail_ds`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py)
saves its progress to a csv file called 'partial_ml_dataset.{ini_ev}_{fin_ev}.csv' in the user-specified directory 
[`destination_dir`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L13) 
for every 50 reports it processes. To change the frequency at which progress is saved, set the argument
[`save_freq`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L13)
to desired frequency. Note that 
[`time_limit`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L13) 
is set to 5.5 hours by default because the system I use has a six hour job-time limit. You will need to find out the time limit of your system (if it has one) and set 
[`time_limit`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L13) 
accordingly.

I also included a 
[`time_limit`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L13) 
argument. When the run-time reaches the time limit, a csv file saves the progress as-is, to a file called 
[`partial_ml_dataset.{ini_ev}__{fin_ev}.csv`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/examples/ml_dataset.2898_2899.csv) 
to make sure processes are saved. The process can be picked up again by checking the partially completed csv file for the last completed event (all reports in the event were processed) and setting 
[`ini_ev`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L13) 
to that event + 1.  

One can also split this job into chunks by using the initial and final event arguments 
[`ini_ev`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L13) 
and 
[`fin_ev`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L13)
, respectively. 

---
### For hail detection
#### Null-case dataset
Use 
[`data_processing/create_ml_dataset.null_ds`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L275)
;
```python
def null_ds(sl_file_path, pl_dir_path, num_reports, destination_dir='', time_limit=5.5, save_freq=50):
```

The `null_ds` function randomly samples a latitude, a longitude, and a consecutive set of ten timestamps from the single-level ERA5 NetCDF through the user-provided `sl_file_path`. It then creates sub-datasets of the single-level and pressure-level ERA5 data using the randomly sampled information and merges these sub-datasets. The merged sub-dataset is used to fill-in a dataframe with the same columns as the one in `hail_ds` (see above). The attribute `num_reports` lets the user specify how many null-case reports they would like to create. Similar to `hail_ds`, `null_ds` saves it's progress according to `save_freq`,  as `partial_null_ml_dataset.{num_reports}.csv` in `destination_dir`, and has a `time_limit` variable for those of us with run-time limits on our machines. `null_ds` returns the dataframe when finished and saves the final dataframe as a CSV file called `null_ml_dataset.{num_reports}.csv`.

---
#### Full dataset (hail & null cases)
Use
[`data_processing/create_ml_dataset.full_ds`](https://github.com/aconlon-eccc/era5-based-hail/blob/master/data_processing/create_ml_dataset.py#L518)
;
```python
def full_ds(obs_path, eventised_era5_dir_path, sl_file_path, pl_dir_path, num_reports, destination_dir='', ini_ev=0, fin_ev=0, time_limit=5.5, severe=20, save_freq=50):
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


