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

        block_district_matrix = make_bdm(tree_data['leaf_nodes'], len(state_df))
        district_df = create_district_df(state_abbrev, block_district_matrix,
                                         calculate_compactness=True)
        
        district_df.to_csv(os.path.join(dir_path, 'district_dfs', save_name), index=False)
        elapsed_t = str(round((time.time() - start_t) / 60, 2))
        print('Built and saved', state_abbrev, 'in', elapsed_t, 'mins')


if __name__ == '__main__':
    paths_to_process = [
        os.path.join(constants.RESULTS_PATH, 'allstates', 'aaai_columns1595813019'),
    ]
    for path in paths_to_process:
        district_df_of_tree_dir(path)
