from gerrypy import constants
from gerrypy.data.load import *
from gerrypy.data.load import *
from gerrypy.analyze.viz import *

from scipy.stats import t
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.errors import TopologicalError
from scipy.spatial.distance import cdist


class StatePrecinctWrapper:
    def __init__(self):
        self.state = None
        self.main_sources = []
        self.county_inference = {}

    def get_data(self):
        precinct_gdfs = self.load_precincts()
        tract_coverages = [self.compute_tract_coverage(precincts)
                           for precincts in precinct_gdfs]
        tract_votes = [self.compute_tract_votes(precincts, coverage)
                       for precincts, coverage in zip(precinct_gdfs, tract_coverages)]
        tract_votes = pd.concat(tract_votes, axis=1)
        inferred_elections = self.infer_w_county_data(tract_votes)
        return pd.concat([tract_votes, inferred_elections], axis=1)

    def load_precincts(self):
        precincts_gdfs = []
        for source in self.main_sources:
            d_columns = {d_col: '_'.join(['D', office, str(year)])
                         for (office, year), (d_col, _) in source['elections'].items()}
            r_columns = {r_col: '_'.join(['R', office, str(year)])
                         for (office, year), (_, r_col) in source['elections'].items()}
            name_dict = {**d_columns, **r_columns,
                         'geometry': 'geometry'}

            precinct_gdf = gpd.read_file(source['path'])
            precinct_gdf.rename(columns={c: c.strip() for c in precinct_gdf.columns})
            precinct_gdf = precinct_gdf.rename(columns=name_dict).to_crs(epsg=constants.CRS)
            precincts_gdfs.append(precinct_gdf[list(name_dict.values())])
        return precincts_gdfs

    def create_probability_state_df(self):
        precinct_gdfs = self.load_precincts()
        source_data = [self.compute_tract_results(precincts, source_ix)
                       for source_ix, precincts in enumerate(precinct_gdfs)]
        election_df = pd.concat(source_data)
        mu = election_df.mean(axis=1)
        stddev = election_df.std(ddof=1, axis=1)
        n_election = len(election_df.columns)
        degrees_of_freedom = n_election - election_df.isna().sum(axis=1)
        return pd.DataFrame({
            'mean': mu,
            'stddev': stddev,
            'DoF': degrees_of_freedom
        })

    def compute_tract_coverage(self, precincts):
        tracts = load_tract_shapes(self.state).to_crs(epsg=constants.CRS)

        p_centers = np.stack([precincts.centroid.x, precincts.centroid.y]).T
        t_centers = np.stack([tracts.centroid.x, tracts.centroid.y]).T

        dists = cdist(t_centers, p_centers).argsort()
        # Calculate the overlap of tracts and precincts
        tract_coverage = {}
        for tix, row in tracts.iterrows():
            tract_coverage[tix] = []
            ratio_tract_covered = 0
            tgeo = row.geometry
            tarea = tgeo.area
            pix = 0
            while ratio_tract_covered < .99 and pix < len(p_centers) / 10:
                precinct_id = dists[tix, pix]
                precinct_row = precincts.iloc[precinct_id]
                pgeo = precinct_row.geometry
                try:
                    overlap_area = tgeo.intersection(pgeo).area
                except TopologicalError:  # In case polygon crosses itself
                    try:
                        overlap_area = tgeo.buffer(0).intersection(pgeo.buffer(0)).area
                    except TopologicalError:
                        overlap_area = tgeo.convex_hull.buffer(0) \
                            .intersection(pgeo.convex_hull.buffer(0)).area
                if overlap_area > 0:
                    ratio_tract_covered += overlap_area / tarea

                    tract_coverage[tix].append((precinct_id,
                                                overlap_area / pgeo.area))
                pix += 1
        return tract_coverage

    def compute_tract_votes(self, precincts, tract_coverage):
        # Estimate tract vote shares
        tract_election_results = {}
        election_columns = list(precincts.columns)
        election_columns.remove('geometry')
        try:
            precincts[election_columns] = precincts[election_columns].astype(np.float64)
        except ValueError:
            for column in election_columns:
                precincts[column] = precincts[column].str.replace(',', '').astype(np.float64)
        for t, plist in tract_coverage.items():
            try:
                tract_precincts, coverage_ratio = zip(*plist)
                results_mat = precincts.loc[tract_precincts, election_columns].values
                tract_election_results[t] = pd.Series(np.array(coverage_ratio) @ results_mat)
            except ValueError:  # If tract_coverage empty
                mock = np.empty(len(election_columns))
                mock[:] = np.nan
                tract_election_results[t] = pd.Series(mock)

        col_names = {ix: estr for ix, estr in enumerate(election_columns)}
        tract_election_df = pd.DataFrame(tract_election_results).T.rename(columns=col_names).fillna(0)

        # election_vote_shares = {e: tract_election_df['R_' + e] /
        #                            (tract_election_df['R_' + e] + tract_election_df['D_' + e])
        #                         for e in self.election_strings(source_ix)}
        #
        # tract_vote_shares = pd.DataFrame(election_vote_shares)

        return tract_election_df

    def infer_w_county_data(self, votes_df):
        state_df = load_state_df(self.state)
        votes_df['county'] = state_df['GEOID'].astype(str).apply(lambda x: x.zfill(11)[:5])
        votes_df['population'] = state_df['population']
        inferred_dfs = []
        for election_unknown, election_known in self.county_inference.items():
            known_election_str = '_'.join([str(f) for f in election_known])
            unknown_election_str = '_'.join([str(f) for f in election_unknown])
            known_election_series = votes_df[
                ['county', 'population', 'D_' + known_election_str, 'R_' + known_election_str]]
            county_total = known_election_series.groupby('county').sum()
            total_votes = county_total[['D_' + known_election_str, 'R_' + known_election_str]].sum(axis=1)
            known_election_county_share = county_total['R_' + known_election_str] / total_votes
            state_df['county'] = votes_df['county']
            unknown_election_county_share = state_df.groupby('county')[str(election_unknown[1])].mean()
            county_adjustment = (unknown_election_county_share - known_election_county_share).to_dict()
            tract_adjustment = votes_df['county'].map(lambda x: county_adjustment[x])
            tract_total = votes_df[['D_' + known_election_str, 'R_' + known_election_str]].sum(axis=1)
            known_tract_ratio = (votes_df['R_' + known_election_str] / tract_total).values
            estimated_tract_ratio = known_tract_ratio + tract_adjustment
            inferred_dfs.append(pd.DataFrame({
                'D_' + unknown_election_str: (1 - estimated_tract_ratio) * tract_total,
                'R_' + unknown_election_str: estimated_tract_ratio * tract_total
            }))
        # Clean up mutable data structure
        votes_df.drop(columns=['county', 'population'], inplace=True)
        return pd.concat(inferred_dfs, axis=1)

    def election_strings(self):
        return [office + '_' + str(year) for s in self.main_sources
                for office, year in s['elections']]

    def data_report(self):
        precincts = self.load_precincts()
        for gdf in precincts:
            for column in gdf.columns:
                print(column, (gdf[column] == 0).sum())
            coverage, shares = self.compute_tract_results(gdf)
            shares.isna()

    def test_self(self):
        print('Testing %s precinct wrapper' % self.state)
        for source in self.main_sources:
            if os.path.exists(source['path']):
                required_election_columns = set([
                    column
                    for election in source['elections'].values()
                    for column in election
                ])
                precinct_file = gpd.read_file(source['path'])
                available_columns = set([c.strip() for c in precinct_file.columns])
                col_intersection = available_columns.intersection(required_election_columns)
                if len(required_election_columns) != len(col_intersection):
                    missing_columns = required_election_columns - col_intersection
                    print('ERROR: missing columns', missing_columns)
                if (source['county_column'] is not None) and \
                        (not source['county_column'] in available_columns):
                    print('ERROR: county column %s does not exist' % source['county_column'])
            else:
                print('ERROR:', source['path'], 'does not exist')
