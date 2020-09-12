from gerrypy import constants
from gerrypy.data.load import *
from gerrypy.data.precinct_state_wrappers import wrappers


def get_state_partisanship():
    partisanship = {}
    for state, wrapper in wrappers.items():
        if state == 'WV':
            state_df = load_state_df(state)
            shares = state_df[['2008', '2012', '2016']].values.T
            weights = state_df.population.values / state_df.population.values.sum()
            state_partisanship = (shares @ weights).mean()
        else:
            election_df = load_election_df(state).fillna(0)
            election_columns = wrapper().election_columns()
            state_vote_dict = election_df[election_columns].sum(axis=0).to_dict()
            elections = set([c[2:] for c in election_columns])
            state_partisanship = np.array([
                state_vote_dict['R_' + e] / (state_vote_dict['R_' + e] + state_vote_dict['D_' + e])
                for e in elections
            ]).mean()
        partisanship[state] = state_partisanship
    return partisanship


def get_state_election_results():
    election_results = {}
    for state, wrapper in wrappers.items():
        if state == 'WV':
            state_df = load_state_df(state)
            elections = ['2008', '2012', '2016']
            state_elections = {e: np.average(state_df[e].values,
                                             weights=state_df.population.values) for e in elections}
        else:
            election_df = load_election_df(state).fillna(0)
            election_columns = wrapper().election_columns()
            state_vote_dict = election_df[election_columns].sum(axis=0).to_dict()
            elections = set([c[2:] for c in election_columns])
            state_elections = {
                e:state_vote_dict['R_' + e] / (state_vote_dict['R_' + e] + state_vote_dict['D_' + e])
                for e in elections
            }
        election_results[state] = state_elections
    return election_results


def get_state_population():
    return {state: load_state_df(state).population.sum()
            for state in constants.seats}
