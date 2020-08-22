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
        self.main_source = {}
        self.county_inference = {}

    def load_precincts(self):
        d_columns = {d_col: '_'.join(['D', office, str(year)])
                     for (office, year), (d_col, _) in self.main_source['elections'].items()}
        r_columns = {r_col: '_'.join(['R', office, str(year)])
                     for (office, year), (_, r_col) in self.main_source['elections'].items()}
        name_dict = {**d_columns, **r_columns, 'geometry': 'geometry'}

        precinct_gdf = gpd.read_file(self.main_source['path']).rename(columns=name_dict).to_crs(epsg=constants.CRS)
        return precinct_gdf[list(name_dict.values())]

    def compute_tract_results(self):
        precincts = self.load_precincts()
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
            while ratio_tract_covered < .99 and pix < len(p_centers) / 4:
                if pix % 10 == 0:
                    print(tix, pix)
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

        # Estimate tract vote shares
        tract_election_results = {}
        election_columns = list(precincts.columns)
        election_columns.remove('geometry')
        for t, plist in tract_coverage.items():
            tract_precincts, coverage_ratio = zip(*plist)
            results_mat = precincts.loc[tract_precincts, election_columns].values
            tract_election_results[t] = pd.Series(np.average(results_mat, weights=coverage_ratio, axis=0))

        col_names = {ix: estr for ix, estr in enumerate(election_columns)}
        tract_election_df = pd.DataFrame(tract_election_results).T.rename(columns=col_names)

        election_vote_shares = {e: tract_election_df['R_' + e] /
                                   (tract_election_df['R_' + e] + tract_election_df['D_' + e])
                                for e in self.election_strings()}

        return pd.DataFrame(election_vote_shares)

    def infer_w_county_data(self):
        pass

    def election_strings(self):
        return [office + '_' + str(year) for office, year in self.main_source['elections']]
