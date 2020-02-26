import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.errors import TopologicalError
from scipy.spatial.distance import cdist


def annotate_precincts(precinct_shape_path, census_shape_path,
                       census_data_path, census_column_path):
    precincts = gpd.read_file(precinct_shape_path).to_crs(epsg=4326)
    precincts.columns = [c.upper() if c != 'geometry' else c
                         for c in precincts.columns]
    tracts = gpd.read_file(census_shape_path).to_crs(epsg=4326)

    tracts['center_x'] = tracts.centroid.x
    tracts['center_y'] = tracts.centroid.y
    precincts['center_x'] = precincts.centroid.x
    precincts['center_y'] = precincts.centroid.y

    p_centers = precincts[['center_x', 'center_y']].values
    t_centers = tracts[['center_x', 'center_y']].values

    dists = cdist(p_centers, t_centers).argsort()

    precinct_coverage = {}
    for pix, row in precincts.iterrows():
        precinct_coverage[pix] = []
        percent_precinct_covered = 0
        pgeo = row.geometry
        parea = pgeo.area
        tix = 0
        while percent_precinct_covered < .99 and tix < len(t_centers) / 2:
            tract_id = dists[pix, tix]
            tract_row = tracts.iloc[tract_id]
            tgeo = tract_row.geometry
            try:
                overlap_area = pgeo.intersection(tgeo).area
            except TopologicalError:  # In case polygon crosses itself
                try:
                    overlap_area = pgeo.buffer(0).intersection(tgeo.buffer(0)).area
                except TopologicalError:
                    overlap_area = pgeo.convex_hull.buffer(0)\
                        .intersection(tgeo.convex_hull.buffer(0)).area
            if overlap_area > 0:
                percent_precinct_covered += overlap_area / parea

                precinct_coverage[pix].append((tract_id,
                                               overlap_area / tgeo.area))
            tix += 1

    csvs = []
    for csv_path in os.listdir(census_data_path):
        csv = pd.read_csv(os.path.join(census_data_path, csv_path))
        csv[['state', 'county', 'tract']] = csv[
            ['state', 'county', 'tract']].astype(str)
        csv['state'] = csv['state'].apply(lambda x: x.zfill(2))
        csv['county'] = csv['county'].apply(lambda x: x.zfill(3))
        csv['tract'] = csv['tract'].apply(lambda x: x.zfill(6))
        csv['GEOID'] = csv['state'] + csv['county'] + csv['tract']
        csv = csv.set_index('GEOID')
        csvs.append(csv)

    state_csv = pd.concat(csvs, axis=1)
    keeps_cols = json.load(open(census_column_path, 'r'))
    state_csv = state_csv[keeps_cols]
    state_csv[(state_csv < 0)] = np.nan
    state_csv = state_csv.fillna(state_csv.median())

    population_name = 'DP02_0122E'
    population_column = state_csv[population_name]
    state_csv = state_csv.drop(columns=[population_name])

    precinct_populations = []
    precinct_data = np.zeros((len(precinct_coverage), len(state_csv.columns)))
    for pix, tract_list in precinct_coverage.items():
        precinct_population = []
        for tract, coverage_percent in tract_list:
            tract_geoid = tracts['GEOID'].loc[tract]
            population = population_column[tract_geoid]
            precinct_population.append((tract_geoid,
                                        population * coverage_percent))

        total_precinct_population = sum(pop for _, pop in precinct_population)
        precinct_populations.append(total_precinct_population)

        for tract_geoid, tract_population in precinct_population:
            weight = tract_population / total_precinct_population
            precinct_data[pix] += state_csv.loc[tract_geoid] * weight

    precinct_df = pd.DataFrame(precinct_data, columns=state_csv.columns)
    precinct_df['population'] = precinct_populations

    total_voters = (precincts['G16PRERTRU'] + precincts['G16PREDCLI'])
    precinct_df['p_dem'] = (precincts['G16PREDCLI'] / total_voters)
    precinct_df = precinct_df.fillna(precinct_df.median())
    return precinct_df


# coverage = annotate_precincts('nc_2016_precincts',
#                               'shapes/37_2018',
#                               'data/2016_acs5/37',
#                               '2016_acs5/politics_columns.json')
