import os
import json
import requests
import pandas as pd
from gerrypy import constants


def variable_mapping(year, columns=None):
    dirname = constants.TRACT_DATA_PATH
    file_path = os.path.join(dirname, '%d_acs5' % year, 'profile_variables.json')
    with open(file_path, 'r') as f:
        var_map = json.load(f)['variables']

        if columns is None:
            return var_map

        else:
            return {v: var_map.get(v, 'None').get('label',
                                                  'Not a census variable')
                    for v in columns}


def download_all_county_data(states=None, years=['2018']):
    if not states:
        states = [str(state_abbrev) for _, state_abbrev, _ in constants.STATE_IDS]
    for y in years:
        for state in states:
            download_census_tables(state=state, year=y, county=True)


def download_all_tract_data(states=None, years=['2018']):
    if not states:
        states = [str(state_abbrev) for _, state_abbrev, _ in constants.STATE_IDS]
    for y in years:
        for state in states:
            download_census_tables(state=state, year=y, county=False)


def download_census_tables(state=None, year=None, county=False):
    """Download census tables for every census tract is areas.

    state : str - state abbreviation (Two capital letters)
    year : str - year of census data to download from
    county: bool - download county or census tract data"""
    DATASET = "acs/acs5"

    # Selected social, economic, housing, and demographic indicators
    TABLES = ["DP02", "DP03", "DP04", "DP05"]

    year = str(year)
    BASE_URL = "https://api.census.gov/data/%s/%s" % (year, DATASET)

    base_save_dir = constants.COUNTY_DATA_PATH if county else constants.TRACT_DATA_PATH

    data_name = year + "_" + DATASET.split("/")[1]
    path_name = os.path.join(base_save_dir, data_name)
    try:
        os.mkdir(path_name)
    except FileExistsError:
        pass
    state_fips = constants.ABBREV_DICT[state][constants.FIPS_IX]
    granularity = 'county' if county else 'tract'

    save_path = os.path.join(path_name, state + '_%s.csv' % granularity)
    if os.path.exists(save_path):
        return

    # Download variable mapping
    var_path = os.path.join(path_name, "profile_variables.json")
    if not (os.path.exists(var_path) and os.stat(var_path).st_size > 10):
        vars_request = requests.get(BASE_URL + "/profile/variables.json")
        vars_json = vars_request.json()
        with open(os.path.join(path_name, "profile_variables.json"), "w") as f:
            json.dump(vars_json, f)
        print("Successfully downloaded variable mapping")

    # Download census tract data for all counties and all tables
    table_dfs = []
    for table in TABLES:

        table_url = BASE_URL + "/profile?get=group(%s)" % table
        if county:
            table_url += "&for=county:*&in=state:%s" % state_fips
        else:
            table_url += "&for=tract:*&in=state:%s" % state_fips

        if constants.CENSUS_API_KEY:
            table_url += "&key=" + constants.CENSUS_API_KEY

        r = requests.get(table_url)
        try:
            json_df = r.json()
        except:
            print('WARNING: table download failure')
            print('url:', table_url)
            print('text', r.text)

        df = pd.DataFrame(json_df[1:], columns=json_df[0])

        if int(year) < 2011:
            df.rename(columns={"GEO_ID": "GEOID"}, inplace=True)
            if county:
                df['GEOID'] = df['GEOID'].apply(lambda x: x.split("US")[1]) \
                    .astype(str).apply(lambda x: x.zfill(5))
            else:
                df['GEOID'] = df['GEOID'].apply(lambda x: x.split("US")[1]) \
                    .astype(str).apply(lambda x: x.zfill(11))
        else:
            df['state'] = df['state'].astype(str).apply(lambda x: x.zfill(2))
            df['county'] = df['county'].astype(str).apply(lambda x: x.zfill(3))
            if county:
                df['GEOID'] = df['state'] + df['county']
            else:
                df['tract'] = df['tract'].astype(str).apply(lambda x: x.zfill(6))
                df['GEOID'] = df['state'] + df['county'] + df['tract']

        df = df.set_index('GEOID')

        table_dfs.append(df)

    state_year_df = pd.concat(table_dfs, axis=1)

    state_year_df.to_csv(save_path)
    print("Successfully downloaded acs for", state, 'for', year)


if __name__ == '__main__':
    download_all_tract_data()
