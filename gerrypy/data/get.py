"""Script for fresh download of all data

NOTE: must manually download the county dataset here
https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/VOQCHQ/HEIJCQ&version=6.0#"""

from gerrypy.data import shapefiles, acs, preprocess
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

