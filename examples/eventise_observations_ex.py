"""
------------------------------
Eventise hail observation data
------------------------------

Use 'ERA5_Based_Hail.data_processing.eventise_data.observations'.

We have a csv file of 7000 hail reports from all over Canada between 2005 and 2022 called 'hail_db_with_LD.csv'.

Some of these reports are from the same hail event, i.e. a single hail storm produces multiple reports from different
locations throughout its life. So we would like to bundle reports into hail 'events' based on the time of the report
and the distances between reports. Our function 'ERA5_Based_Hail.data_processing.eventise_data.observations' does this
for us.

Different reports are considered to be of the same hail event according to time and space windows which are defined in
'ERA5_Based_Hail.constants_and_variables.py'. Specifically, 'ERA5_Based_Hail.data_processing.eventise_data.observations'
uses the constants 'event_time_window', 'event_spatial_window', 'earth_rad', and 'boundary_distance'. If different time
and space constants are desired, changes should be made to the 'constants_and_variables.py' file, keeping in mind units.
"""

import data_processing.eventise_data as ed

eventised_hail_data = ed.observations('integrated_canadian_hail_db.csv')



