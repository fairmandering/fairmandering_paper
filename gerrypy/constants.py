import os

GERRYPY_BASE_PATH = os.path.dirname(os.path.abspath(__file__))
PRECINCT_PATH = os.path.join(GERRYPY_BASE_PATH, 'data', 'precincts')
OPT_DATA_PATH = os.path.join(GERRYPY_BASE_PATH, 'data', 'optimization_data')
CENSUS_SHAPE_PATH = os.path.join(GERRYPY_BASE_PATH, 'data', 'shapes')
TRACT_DATA_PATH = os.path.join(GERRYPY_BASE_PATH, 'data', 'acs_tract_data')
COUNTY_DATA_PATH = os.path.join(GERRYPY_BASE_PATH, 'data', 'acs_county_data')
COLUMNS_PATH = os.path.join(GERRYPY_BASE_PATH, 'results', 'columns')
APP_PATH = os.path.join(GERRYPY_BASE_PATH, 'analyze', 'viz', 'gerryapp')

ACS_BASE_YEAR = 2018

NAME_IX = 0
ABBREV_IX = 1
FIPS_IX = 2
STATE_IDS = [
    ('Alabama', 'AL', '01'),
    ('Alaska', 'AK', '02'),
    ('Arizona', 'AZ', '04'),
    ('Arkansas', 'AR', '05'),
    ('California', 'CA', '06'),
    ('Colorado', 'CO', '08'),
    ('Connecticut', 'CT', '09'),
    ('Delaware', 'DE', '10'),
    ('Florida', 'FL', '12'),
    ('Georgia', 'GA', '13'),
    ('Hawaii', 'HI', '15'),
    ('Idaho', 'ID', '16'),
    ('Illinois', 'IL', '17'),
    ('Indiana', 'IN', '18'),
    ('Iowa', 'IA', '19'),
    ('Kansas', 'KS', '20'),
    ('Kentucky', 'KY', '21'),
    ('Louisiana', 'LA', '22'),
    ('Maine', 'ME', '23'),
    ('Maryland', 'MD', '24'),
    ('Massachusetts', 'MA', '25'),
    ('Michigan', 'MI', '26'),
    ('Minnesota', 'MN', '27'),
    ('Mississippi', 'MS', '28'),
    ('Missouri', 'MO', '29'),
    ('Montana', 'MT', '30'),
    ('Nebraska', 'NE', '31'),
    ('Nevada', 'NV', '32'),
    ('New Hampshire', 'NH', '33'),
    ('New Jersey', 'NJ', '34'),
    ('New Mexico', 'NM', '35'),
    ('New York', 'NY', '36'),
    ('North Carolina', 'NC', '37'),
    ('North Dakota', 'ND', '38'),
    ('Ohio', 'OH', '39'),
    ('Oklahoma', 'OK', '40'),
    ('Oregon', 'OR', '41'),
    ('Pennsylvania', 'PA', '42'),
    ('Rhode Island', 'RI', '44'),
    ('South Carolina', 'SC', '45'),
    ('South Dakota', 'SD', '46'),
    ('Tennessee', 'TN', '47'),
    ('Texas', 'TX', '48'),
    ('Utah', 'UT', '49'),
    ('Vermont', 'VT', '50'),
    ('Virginia', 'VA', '51'),
    ('Washington', 'WA', '53'),
    ('West Virginia', 'WV', '54'),
    ('Wisconsin', 'WI', '55'),
    ('Wyoming', 'WY', '56'),
]
NAME_DICT = {state_info[NAME_IX]: state_info for state_info in STATE_IDS}
FIPS_DICT = {state_info[FIPS_IX]: state_info for state_info in STATE_IDS}
ABBREV_DICT = {state_info[ABBREV_IX]: state_info for state_info in STATE_IDS}


seats = {
    'AL': {'house': 7, 'state_senate': 35, 'state_house': 105},
    'AK': {'house': 1, 'state_senate': 20, 'state_house': 40},
    'AZ': {'house': 9, 'state_senate': 30, 'state_house': 60},
    'AR': {'house': 4, 'state_senate': 35, 'state_house': 100},
    'CA': {'house': 53, 'state_senate': 40, 'state_house': 80},
    'CO': {'house': 7, 'state_senate': 35, 'state_house': 65},
    'CT': {'house': 5, 'state_senate': 36, 'state_house': 151},
    'DE': {'house': 1, 'state_senate': 21, 'state_house': 41},
    'FL': {'house': 27, 'state_senate': 40, 'state_house': 120},
    'GA': {'house': 14, 'state_senate': 56, 'state_house': 180},
    'HI': {'house': 2, 'state_senate': 25, 'state_house': 51},
    'ID': {'house': 2, 'state_senate': 35, 'state_house': 70},
    'IL': {'house': 18, 'state_senate': 59, 'state_house': 118},
    'IN': {'house': 9, 'state_senate': 50, 'state_house': 100},
    'IA': {'house': 4, 'state_senate': 50, 'state_house': 100},
    'KS': {'house': 4, 'state_senate': 40, 'state_house': 125},
    'KY': {'house': 6, 'state_senate': 38, 'state_house': 100},
    'LA': {'house': 6, 'state_senate': 39, 'state_house': 105},
    'ME': {'house': 2, 'state_senate': 35, 'state_house': 151},
    'MD': {'house': 8, 'state_senate': 47, 'state_house': 141},
    'MA': {'house': 9, 'state_senate': 40, 'state_house': 160},
    'MI': {'house': 14, 'state_senate': 38, 'state_house': 110},
    'MN': {'house': 8, 'state_senate': 67, 'state_house': 134},
    'MS': {'house': 4, 'state_senate': 52, 'state_house': 122},
    'MO': {'house': 8, 'state_senate': 34, 'state_house': 163},
    'MT': {'house': 1, 'state_senate': 50, 'state_house': 100},
    'NE': {'house': 3, 'state_senate': 49, 'state_house': 0},
    'NV': {'house': 4, 'state_senate': 21, 'state_house': 42},
    'NH': {'house': 2, 'state_senate': 24, 'state_house': 400},
    'NJ': {'house': 12, 'state_senate': 40, 'state_house': 80},
    'NM': {'house': 3, 'state_senate': 42, 'state_house': 70},
    'NY': {'house': 27, 'state_senate': 63, 'state_house': 150},
    'NC': {'house': 13, 'state_senate': 50, 'state_house': 120},
    'ND': {'house': 1, 'state_senate': 47, 'state_house': 94},
    'OH': {'house': 16, 'state_senate': 33, 'state_house': 99},
    'OK': {'house': 5, 'state_senate': 48, 'state_house': 101},
    'OR': {'house': 5, 'state_senate': 30, 'state_house': 60},
    'PA': {'house': 18, 'state_senate': 50, 'state_house': 203},
    'RI': {'house': 2, 'state_senate': 38, 'state_house': 75},
    'SC': {'house': 7, 'state_senate': 46, 'state_house': 124},
    'SD': {'house': 1, 'state_senate': 35, 'state_house': 70},
    'TN': {'house': 9, 'state_senate': 33, 'state_house': 99},
    'TX': {'house': 36, 'state_senate': 31, 'state_house': 150},
    'UT': {'house': 4, 'state_senate': 29, 'state_house': 75},
    'VT': {'house': 1, 'state_senate': 30, 'state_house': 150},
    'VA': {'house': 11, 'state_senate': 40, 'state_house': 100},
    'WA': {'house': 10, 'state_senate': 49, 'state_house': 98},
    'WV': {'house': 3, 'state_senate': 34, 'state_house': 100},
    'WI': {'house': 8, 'state_senate': 33, 'state_house': 99},
    'WY': {'house': 1, 'state_senate': 30, 'state_house': 60}
}



