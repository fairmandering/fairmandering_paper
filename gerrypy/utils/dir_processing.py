import os
import pickle
from gerrypy.data.load import load_state_df
from gerrypy.analyze.districts import make_bdm, create_district_df

def district_df_of_tree_dir(dir_path):
    tree_files = os.listdir(dir_path)
    for state_file in tree_files:
        state_abbrev = state_file[:2]
        tree_data = pickle.load(open(os.path.join(dir_path, state_file), 'rb'))
        state_df = load_state_df(state_abbrev)
        
        block_district_matrix = make_bdm(tree_data['leaf_nodes'], len(state_df))
        district_df = create_district_df(block_district_matrix, state_df)
        
        district_df.to_csv(os.path.join(dir_path, state_abbrev))
        print('Built and saved', state_abbrev)
        
if __name__ == '__name__':
    paths_to_process = [
        os.path.abspath('/gerrypy/gerrypy/experiments/aaai_columns1595813019'),
        os.path.abspath('/gerrypy/gerrypy/experiments/aaai_columns1595891125')
    ]
    for path in paths_to_process:
        district_df_of_tree_dir(path)
