"""This script downloads selected census tables from selected datasets
from selected counties.

DATASET and TABLES should be left to default values unless there is a
good reason to change. This script will create a parent data directory
for each DATASET and YEAR combination. Within this parent directory each
county will get its own directory with all tables from TABLES."""

import os
import json
import requests
import pandas as pd
from gerrypy import constants

########################### PARAMETERS #############################
YEAR = "2016"
DATASET = "acs/acs5"

# Selected social, economic, housing, and demographic indicators
TABLES = ["DP02", "DP03", "DP04", "DP05"]

# Location of the data
BASE_URL = "https://api.census.gov/data/%s/%s" % (YEAR, DATASET)

# List of 2 digit state codes and/or 5 digit county FIPS codes
# https://en.wikipedia.org/wiki/List_of_United_States_FIPS_codes_by_county
AREAS = ['25']


def download_census_tables(AREAS, YEAR):
    """Download census tables for every census tract is AREAS.
    
    AREAS : str list - state or county fips codes to download tracts for
    YEAR : str - year of census data to download from"""

    AREAS = [str(a) for a in AREAS]
    YEAR = str(YEAR)
    BASE_URL = "https://api.census.gov/data/%s/%s" % (YEAR, DATASET)

    dirname = os.path.dirname(__file__)

    data_name = YEAR + "_" + DATASET.split("/")[1]
    path_name = os.path.join(dirname, data_name)
    try:
        os.mkdir(path_name)
    except FileExistsError:
        pass

    # Download variable mapping
    var_path = os.path.join(dirname, "%s_acs5" % YEAR, "profile_variables.json")
    if not (os.path.exists(var_path) and os.stat(var_path).st_size > 10):
        vars_request = requests.get(BASE_URL + "/profile/variables.json")
        vars_json = vars_request.json()
        with open(os.path.join(path_name, "profile_variables.json"), "w") as f:
            json.dump(vars_json, f)
        print("Successfully downloaded variable mapping")

    # Download census tract data for all counties and all tables
    for table in TABLES:
        for fips_code in AREAS:

            table_url = BASE_URL + "/profile?get=group(%s)" % table
            area_path = os.path.join(path_name, fips_code)

            # If file exists
            if (
                os.path.exists(os.path.join(area_path, table + ".csv"))
                and os.stat(os.path.join(area_path, table + ".csv")).st_size > 10
            ):
                continue

            try:
                os.mkdir(area_path)
            except FileExistsError:
                pass

            if len(fips_code) == 2:  # State
                table_url += "&for=tract:*&in=state:%s" % fips_code

            elif len(fips_code) == 5:  # County
                state = fips_code[:2]
                county = fips_code[2:]
                table_url += "&for=tract:*&in=state:%s&in=county:%s" % (state, county)

            if constants.CENSUS_API_KEY:
                table_url += "&key=" + constants.CENSUS_API_KEY

            r = requests.get(table_url)
            try:
                json_df = r.json()
            except:
                print(table_url)
                print(r.text)

            df = pd.DataFrame(json_df[1:], columns=json_df[0])
            df.rename(columns={"GEO_ID": "GEOID"})
            save_name = os.path.join(area_path, table + ".csv")
            df.to_csv(save_name, index=False)
            print("Successfully downloaded", table, "for area(s)", AREAS)


if __name__ == "__main__":
    download_census_tables(AREAS, YEAR)
