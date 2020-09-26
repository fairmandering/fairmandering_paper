import os
from gerrypy import constants
from gerrypy.data.precinct_state_wrappers import wrappers

for state, wrapper in wrappers.items():
    print(state)
    try:
        election_df = wrapper().get_data()
        save_path = os.path.join(constants.OPT_DATA_PATH, state, 'election_df.csv')
        election_df.to_csv(save_path, index=False)
    except NotImplementedError:
        print('Not implemented')
