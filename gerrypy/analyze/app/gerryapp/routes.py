from flask import render_template, url_for, flash, jsonify, request
import gerrypy.constants as consts
import os
import json

from flask import Flask
app = Flask(__name__)


US_BB = {
    'w': -124.848974,
    's': 24.396308,
    'e': -66.885444,
    'n': 49.384358
}


@app.route('/', methods=['GET'])
def home():
    print('reached home')
    run_info = request.args.get('run_info')
    print('$$$$ri$$$$', run_info)
    if run_info:
        state_abbr, run_data = get_geographic_data(run_info)
    else:
        run_data = []
        state_abbr = 'nc'

    path_dict = make_selector()

    return render_template('layout.html', path_dict=path_dict, bb=US_BB,
                           state=state_abbr, run_data=run_data, select_value=run_info)


def make_selector():
    state_dirs = [d for d in os.listdir(consts.COLUMNS_PATH)
                  if os.path.isdir(os.path.join(consts.COLUMNS_PATH, d))]

    path_dict = {
        consts.ABBREV_DICT[state][consts.NAME_IX]:
            os.listdir(os.path.join(consts.COLUMNS_PATH, state))
        for state in state_dirs
    }
    return path_dict


def get_geographic_data(run_name):
    state_abbr = run_name.split('_')[0]
    fpath = os.path.join(consts.COLUMNS_PATH, state_abbr, run_name)
    with open(fpath, 'r') as f:
        trial_run = json.load(f)
    return state_abbr, trial_run


# // $.ajax({
#           // url: '/get_geographic_data',
# // data: {'run_info': opt.value},
# // async: false,
# // success: function(result)
# {
# // var
# run_data = result;
# // var
# run_index = 0;
# // var
# run_len = run_data.length
#           //}
# //})