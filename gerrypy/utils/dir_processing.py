import os
import time
import pickle
from gerrypy import constants
from gerrypy.data.load import load_state_df, load_election_df
from gerrypy.analyze.districts import make_bdm, create_district_df


def district_df_of_tree_dir(dir_path):
    tree_files = os.listdir(dir_path)
    for state_file in tree_files:
        if state_file[-2:] != '.p':
            continue
        start_t = time.time()
        state_abbrev = state_file[:2]
        tree_data = pickle.load(open(os.path.join(dir_path, state_file), 'rb'))
        state_df = load_state_df(state_abbrev)
        try:
            election_df = load_election_df(state_abbrev)
        except FileNotFoundError:
            election_df = None

        block_district_matrix = make_bdm(tree_data['leaf_nodes'], len(state_df))
        district_df = create_district_df(block_district_matrix, state_df, election_df)
        
        district_df.to_csv(os.path.join(dir_path, state_abbrev + '.csv'), index=False)
        elapsed_t = str(round((time.time() - start_t) / 60, 2))
        print('Built and saved', state_abbrev, 'in', elapsed_t, 'mins')


if __name__ == '__main__':
    paths_to_process = [
        os.path.join(constants.GERRYPY_BASE_PATH, 'results', 'allstates', 'aaai_columns1595891125'),
        os.path.join(constants.GERRYPY_BASE_PATH, 'results', 'allstates', 'aaai_columns1595813019')
    ]
    for path in paths_to_process:
        district_df_of_tree_dir(path)
