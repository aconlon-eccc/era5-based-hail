"""
--------------------------
Submitting an ERA5 request
--------------------------
The total data for 2022 is ~13GB. This could take a several hours to complete depending on how busy the ERA5 servers
are with requests from other users. The code below downloads ~4.4MB which takes about 20 minutes to complete, depending
on ERA5 server activity.

-------------------
Install CDS API key
-------------------
Before anything, please see ['How to use the CDS API'](https://cds.climate.copernicus.eu/api-how-to) for instructions
on how to install your unique CDS API key.

-------------
ERA5 datasets
-------------
There are two datasets to choose from, 'reanalysis-era5-single-levels' and 'reanalysis-era5-pressure-levels', which
have been abbreviated to 'sl' and 'pl', respectively.

---------------
Data requests
---------------
Receiving the data you have requested from ERA5 can take some time. It depends on a couple things; how many other users
are requesting data at that time and how much data you have requested. Your request will first be queued with other
users, then ERA5 will process your request, and finally the download will begin. Each request can take about an hour
from start to finish. The request made by our function 'era5_request.submit_request' is somewhat of a greedy one. We
request data for all points in time and space *between* events as well as for the events themselves. This ensures
gathering a healthy amount of data for examples of null-events (no hail).

Below is an example of the output for a request to the 'reanalysis-era5-single-levels' ERA5 dataset for year 2022 of
our hail observations, note the amount of time between timestamps of each step.

2023-09-14 10:24:12,294 INFO Welcome to the CDS
2023-09-14 10:24:12,294 INFO Sending request to reanalysis-era5-single-levels
2023-09-14 10:24:12,495 INFO Request is queued
2023-09-14 10:24:17,674 INFO Request is running
2023-09-14 11:00:42,448 INFO Request is completed
2023-09-14 11:00:42,448 INFO Downloading
2023-09-14 11:10:18,343 INFO Download rate 2.8M/s

Requests to the 'reanalysis-era5-pressure-levels' ERA5 dataset are broken-up into 10 requests split by pressure levels
to ensure maximum file limits set by ERA5 on data requests are respected. See 'constants_and_variables.py' for the
pressure level breakdown.
"""

from data_processing import load_eventised_observations as load_obs
from era5_request import era5_request as er

# load eventised observations data
eventised_observations = load_obs.load_obs('eventised_obs.csv') # for details see 'eventise_observations_ex.py'

# request data from the ERA5 'reanalysis-era5-single-levels' ('sl') for year 2022
er.submit_request('sl', 2022, eventised_observations, init_event=2898)

# request data from the ERA5 'reanalysis-era5-single-levels' ('pl') for year 2022
er.submit_request('pl', 2022, eventised_observations, init_event=2898)