"""Script for fresh download of all census data,
and preprocessing of all optimization data structures

NOTE: must manually download all election data from Box repository
"""

from gerrypy.data import shapefiles, acs, preprocess, precinct_state_wrappers
from gerrypy import constants
import pandas as pd
import os

# Download census tract shapefiles for all states in most recent year.
os.makedirs(constants.CENSUS_SHAPE_PATH, exist_ok=True)
shapefiles.download_state_shapes()

# Download county and census acs data for all recent election years

os.makedirs(constants.TRACT_DATA_PATH, exist_ok=True)
os.makedirs(constants.COUNTY_DATA_PATH, exist_ok=True)
DATA_YEARS = ['2010', '2012', '2016', '2018']

acs.download_all_county_data(years=DATA_YEARS)
acs.download_all_tract_data(years=DATA_YEARS)

# Preprocess
os.makedirs(constants.COUNTY_DATA_PATH, exist_ok=True)
for state, seats_dict in constants.seats.items():
    if seats_dict['house'] > 1:
        preprocess.preprocess_tracts(state)

# Process precinct data to match to tracts
for state, wrapper in precinct_state_wrappers.wrappers.items():
    print(state)
    try:
        election_df = wrapper().get_data()
        save_path = os.path.join(constants.OPT_DATA_PATH, state, 'election_df.csv')
        election_df.to_csv(save_path, index=False)
    except NotImplementedError:
        print('Not implemented')
