""" merge_era5.ex.py
-----------------------
Merging ERA5 event data
-----------------------
Once the ERA5 pressure level event files have been merged into single event files (see merge_era5_pl.ex.py) and the
ERA5 single level data has been broken up into events (see eventise_era5.ex.py), we need to merge the events from the
different datasets.
"""

import data_processing.merge_era5 as mrg

# merge events in pressure level event folder with events in single level event folder
mrg.merge_eventised('era5_pl.2022.by_event.merged_levels', 'era5_sl.2022.by_event', init_event=2898, fin_event=2899)