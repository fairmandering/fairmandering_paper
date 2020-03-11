import os

NAME_IX = 0
FIPS_IX = 1
ABBREV_IX = 2
STATE_IDS = [
    ('Florida', 12, 'fl'),
    ('Illinois', 17, 'il'),
    ('Massachusetts', 25, 'ma'),
    ('Michigan', 26, 'mi'),
    ('North Carolina', 37, 'nc')
]
NAME_DICT = {state_info[NAME_IX]: state_info for state_info in STATE_IDS}
FIPS_DICT = {state_info[FIPS_IX]: state_info for state_info in STATE_IDS}
ABBREV_DICT = {state_info[ABBREV_IX]: state_info for state_info in STATE_IDS}



GERRYPY_BASE_PATH = os.path.dirname(os.path.abspath(__file__))
PRECINCT_PATH = os.path.join(GERRYPY_BASE_PATH, 'data', 'precincts')
OPT_DATA_PATH = os.path.join(GERRYPY_BASE_PATH, 'data', 'optimization_data')
COLUMNS_PATH = os.path.join(GERRYPY_BASE_PATH, 'results', 'columns')
APP_PATH = os.path.join(GERRYPY_BASE_PATH, 'analyze', 'viz', 'gerryapp')
