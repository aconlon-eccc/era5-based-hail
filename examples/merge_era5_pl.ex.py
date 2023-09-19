""" merge_era5_pl.ex.py
-------------------------------------------------------
Merging pressure levels into single file for each event
-------------------------------------------------------
Once the ERA5 pressure level datasets have been broken up into events, we want to merge the event files into single
files for each event.
"""

import data_processing.merge_era5 as mrg

# merge pressure levels into single file for each event
mrg.merge_pl('era5_pl.2022.by_event')
