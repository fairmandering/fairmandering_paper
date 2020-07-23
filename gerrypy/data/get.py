"""Script for fresh download of all data

NOTE: must manually download the county dataset here
https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/VOQCHQ/HEIJCQ&version=6.0#"""

from gerrypy.data import shapefiles, acs, preprocess
from gerrypy import constants
import pandas as pd

# Download census tract shapefiles for all states in most recent year.
shapefiles.download_state_shapes()

# Download county and census acs data for all recent election years
DATA_YEARS = ['2010', '2012', '2016', '2018']

acs.download_all_county_data(years=DATA_YEARS)
acs.download_all_tract_data(years=DATA_YEARS)

# Preprocess
n_district_df = pd.read_csv('states.csv')
for ix, row in n_district_df.iterrows():
    if row['cong_districts'] > 1:
        preprocess.preprocess_tracts(row['STUSAB'])

