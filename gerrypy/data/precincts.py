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
    """
    Parent class to do precinct to tract processing.
    """
    def __init__(self):
        # Two letter state abbreviation
        self.state = None
        # (list) of (dict)
        # Each dict should have keys 'path' and 'elections'
        self.main_sources = []
        # (dict) to do basic county inference
        self.county_inference = {}

    def get_data(self):
        """
        Main pipeline to gather precincts, aggregate to tract totals, and do county inference

        Returns: (pd.DataFrame) of election vote totals by election by party.
        """
        precinct_gdfs = self.load_precincts()
        tract_coverages = [self.compute_tract_coverage(precincts)
                           for precincts in precinct_gdfs]
        tract_votes = [self.compute_tract_votes(precincts, coverage)
                       for precincts, coverage in zip(precinct_gdfs, tract_coverages)]
        tract_votes = pd.concat(tract_votes, axis=1).fillna(0)
        if self.county_inference is None:
            return tract_votes
        inferred_elections = self.infer_w_county_data(tract_votes)
        return pd.concat([tract_votes, inferred_elections], axis=1).fillna(0)

    def load_precincts(self):
        """Load precinct data from file system"""
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
            precinct_gdf = precinct_gdf[list(name_dict.values())]
            numeric_columns = [c for c in precinct_gdf if c != 'geometry']
            try:
                precinct_gdf[numeric_columns] = precinct_gdf[numeric_columns].astype(np.float64)
            except ValueError:
                for column in numeric_columns:
                    precinct_gdf[column] = precinct_gdf[column].str.replace(',', '').astype(np.float64)
            precincts_gdfs.append(precinct_gdf)
        return precincts_gdfs

    def create_probability_state_df(self):
        """Compute probability statistics for the state_df"""
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
        """
        Computes area overlap between tracts and precincts.
        Args:
            precincts: (gpd.GeoDataFrame) of precinct shapes

        Returns: (dict) {tract id: [(precinct id, coverage)]}
        """
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
        """
        Compute tract vote totals based on precinct vote totals
        Args:
            precincts: (gpd.GeoDataFrame) of vote totals by precinct
            tract_coverage: (dict) {tract id: [(precinct id, coverage)]}

        Returns: (pd.DataFrame) of tract vote totals

        """
        # Estimate tract vote shares
        tract_election_results = {}
        election_columns = list(precincts.columns)
        election_columns.remove('geometry')
        try:
            precincts[election_columns] = precincts[election_columns].astype(np.float64).fillna(0)
        except ValueError:
            for column in election_columns:
                precincts[column] = precincts[column].str.replace(',', '').astype(np.float64).fillna(0)
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

        return tract_election_df

    def infer_w_county_data(self, votes_df):
        """
        Uses mix of county and precinct data to infer tract voting data.
        Args:
            votes_df: (pd.DataFrame) of election vote totals for all precinct elections

        Returns: (pd.DataFrame) of inferred tract totals

        """
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

    def election_columns(self, include_party=True):
        """
        Helper function to get list of keys for election columns
        Args:
            include_party: (bool) whether to include the party prefix in column name

        Returns: (list) of strings usable to index specific election results

        """
        election_list = []
        for source in self.main_sources:
            for election in source['elections']:
                election_list.append(election)
        if self.county_inference is not None:
            for election in self.county_inference:
                election_list.append(election)
        election_strs = [office + '_' + str(year) for office, year in election_list]
        if include_party:
            return list(map(lambda x: 'D_' + x, election_strs))\
                   + list(map(lambda x: 'R_' + x, election_strs))
        else:
            return election_strs

    def check_state_mean_and_std(self):
        """Sanity check to print mean and std"""
        precincts = self.load_precincts()
        results = {}
        for precinct_gdf in precincts:
            results.update(precinct_gdf.sum(axis=0).to_dict())
        columns = self.election_columns(include_party=False)
        election_shares = {c: results['R_' + c] / (results['D_' + c] + results['R_' + c])
                           for c in columns if 'R_' + c in results}
        shares = np.array(list(election_shares.values()))
        print(election_shares)
        print('Mean', round(shares.mean(), 4), 'STD', round(shares.std(ddof=1), 4))

    def test_self(self):
        """Test function to make sure all paths and columns exist"""
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
