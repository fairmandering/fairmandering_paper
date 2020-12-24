"""Constants file for mapping census names to human readable names."""


NAME_MAP = {
    'population': {
        '2010': 'DP02_0122E',
        '2012': 'DP02_0122E',
        '2016': 'DP02_0122E',
        '2018': 'DP02_0122E',
    },
    'p_public_transportation_commute': {
        '2010': 'DP03_0021PE',
        '2012': 'DP03_0021PE',
        '2016': 'DP03_0021PE',
        '2018': 'DP03_0021PE',
    },
    'p_walk_commute': {
        '2010': 'DP03_0022PE',
        '2012': 'DP03_0022PE',
        '2016': 'DP03_0022PE',
        '2018': 'DP03_0022PE',
    },
    'mean_commute_time': {
        '2010': 'DP03_0025E',
        '2012': 'DP03_0025E',
        '2016': 'DP03_0025E',
        '2018': 'DP03_0025E',
    },
    'p_bachelors_degree_or_higher': {
        '2010': 'DP02_0067PE',
        '2012': 'DP02_0067PE',
        '2016': 'DP02_0067PE',
        '2018': 'DP02_0067PE',
    },
    'unemployment_rate': {
        '2010': 'DP03_0009PE',
        '2012': 'DP03_0009PE',
        '2016': 'DP03_0009PE',
        '2018': 'DP03_0009PE',
    },
    'p_GRAPI<15%': {  # Gross rent as a percentage of household income
        '2010': 'DP04_0135PE',
        '2012': 'DP04_0135PE',
        '2016': 'DP04_0137PE',
        '2018': 'DP04_0137PE',
    },
    'p_GRAPI>35%': {  # Gross rent as a percentage of household income
        '2010': 'DP04_0140PE',
        '2012': 'DP04_0140PE',
        '2016': 'DP04_0142PE',
        '2018': 'DP04_0142PE',
    },
    'p_without_health_insurance': {
        '2010': 'DP03_0099PE',
        '2012': 'DP03_0099PE',
        '2016': 'DP03_0099PE',
        '2018': 'DP03_0099PE',
    },
    'p_nonfamily_household': {
        '2010': 'DP02_0010PE',
        '2012': 'DP02_0010PE',
        '2016': 'DP02_0010PE',
        '2018': 'DP02_0010PE',
    },
    'p_vacant_housing_units': {
        '2010': 'DP04_0003PE',
        '2012': 'DP04_0003PE',
        '2016': 'DP04_0003PE',
        '2018': 'DP04_0003PE',
    },
    'p_renter_occupied': {
        '2010': 'DP04_0046PE',
        '2012': 'DP04_0046PE',
        '2016': 'DP04_0047PE',
        '2018': 'DP04_0047PE',
    },
    'median_household_income': {
        '2010': 'DP03_0062E',
        '2012': 'DP03_0062E',
        '2016': 'DP03_0062E',
        '2018': 'DP03_0062E',
    },
    'p_SNAP_benefits': {
        '2010': 'DP03_0074PE',
        '2012': 'DP03_0074PE',
        '2016': 'DP03_0074PE',
        '2018': 'DP03_0074PE',
    },
    'p_below_poverty_line': {
        '2010': 'DP03_0128PE',
        '2012': 'DP03_0128PE',
        '2016': 'DP03_0128PE',
        '2018': 'DP03_0128PE',
    },
    'p_white': {
        '2010': 'DP05_0032PE',
        '2012': 'DP05_0032PE',
        '2016': 'DP05_0032PE',
        '2018': 'DP05_0037PE'
    },
    'p_age_students': {
        '2010': 'DP02_0057PE',
        '2012': 'DP02_0057PE',
        '2016': 'DP02_0057PE',
        '2018': 'DP02_0057PE',
    },
    'median_age': {
        '2010': 'DP05_0017E',
        '2012': 'DP05_0017E',
        '2016': 'DP05_0017E',
        '2018': 'DP05_0018E',
    },
    'p_mobile_homes': {
        '2010': 'DP04_0014PE',
        '2012': 'DP04_0014PE',
        '2016': 'DP04_0014PE',
        '2018': 'DP04_0014PE',
    },
    'p_without_person_vehicle': {
        '2010': 'DP04_0057PE',
        '2012': 'DP04_0057PE',
        '2016': 'DP04_0058PE',
        '2018': 'DP04_0058PE',
    },
    'p_veterans': {
        '2010': 'DP02_0069PE',
        '2012': 'DP02_0069PE',
        '2016': 'DP02_0069PE',
        '2018': 'DP02_0069PE',
    }
}

CENSUS_VARIABLE_TO_NAME = {y: {} for y in NAME_MAP['population']}
for name, year_to_variable_map in NAME_MAP.items():
    for y, v in year_to_variable_map.items():
        CENSUS_VARIABLE_TO_NAME[y][v] = name
