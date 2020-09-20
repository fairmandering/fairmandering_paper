import os
import time
import pickle
import numpy as np
from gerrypy import constants
from gerrypy.data.load import load_state_df, load_election_df
from gerrypy.analyze.districts import make_bdm, create_district_df


def district_df_of_tree_dir(dir_path):
    os.makedirs(os.path.join(dir_path, 'district_dfs'), exist_ok=True)
    tree_files = os.listdir(dir_path)
    for state_file in tree_files:
        if state_file[-2:] == '.p':
            save_name = state_file[:-2] + '_district_df.csv'
            tree_data = pickle.load(open(os.path.join(dir_path, state_file), 'rb'))
        elif state_file[-4:] == '.npy':
            save_name = state_file[:-4] + '_district_df.csv'
            tree_data = np.load(os.path.join(dir_path, state_file),
                                allow_pickle=True)[()]
        else:
            continue
        start_t = time.time()
        state_abbrev = state_file[:2]

        state_df = load_state_df(state_abbrev)
        try:
            election_df = load_election_df(state_abbrev)
        except FileNotFoundError:
            election_df = None

        block_district_matrix = make_bdm(tree_data['leaf_nodes'], len(state_df))
        district_df = create_district_df(block_district_matrix, state_df, election_df)
        
        district_df.to_csv(os.path.join(dir_path, 'district_dfs', save_name), index=False)
        elapsed_t = str(round((time.time() - start_t) / 60, 2))
        print('Built and saved', state_abbrev, 'in', elapsed_t, 'mins')


if __name__ == '__main__':
    paths_to_process = [
        os.path.join(constants.RESULTS_PATH, 'PNAS', 'PNAS_generation_trials_results_1599960957'),
        os.path.join(constants.RESULTS_PATH, 'PNAS', 'PNAS_population_tolerance_results_1600392876')
    ]
    for path in paths_to_process:
        district_df_of_tree_dir(path)
