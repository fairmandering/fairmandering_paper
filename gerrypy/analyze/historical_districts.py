from scipy.spatial.distance import cdist
import numpy as np
from shapely.errors import TopologicalError


def district_tract_map(tract_gdf, district_gdf):
    """
    Compute the map and inverse map of tracts to districts based on
    maximum area overlap. Used to approximate enacted districts with
    districts strictly composed of census tracts.

    Args:
        tract_gdf: (gpd.GeoDataFrame) of all tracts in state
        district_gdf: (gpd.GeoDataFrame) of all districts in state

    Returns: A mapping between tracts and districts

    """
    def match_tract_to_district(tract, search_order):
        overlap_ratio = {}
        tgeo = tract.geometry
        for cdix in search_order:
            cdgeo = cd_shapes[cdix]
            try:
                overlap_area = tgeo.intersection(cdgeo).area
            except TopologicalError:  # In case polygon crosses itself
                try:
                    overlap_area = tgeo.buffer(0).intersection(cdgeo.buffer(0)).area
                except TopologicalError:
                    overlap_area = tgeo.convex_hull.buffer(0) \
                        .intersection(cdgeo.convex_hull.buffer(0)).area

            if (overlap_area / tgeo.area) > .5:
                assert overlap_area / tgeo.area <= 1
                return cdix
            else:
                overlap_ratio[cdix] = overlap_area / tgeo.area
        print('WARNING: No majority overlap for tract', tract.name)
        return max(list(overlap_ratio.items()), key=lambda x: x[1])[0]

    # fn start
    cd_centers = np.stack([district_gdf.centroid.x, district_gdf.centroid.y]).T
    t_centers = np.stack([tract_gdf.centroid.x, tract_gdf.centroid.y]).T

    cd_shapes = list(district_gdf.geometry)

    dists = cdist(t_centers, cd_centers).argsort()
    # Calculate the overlap of tracts and districts
    tract_to_district = {tix: match_tract_to_district(row, dists[tix])
                         for tix, row in tract_gdf.iterrows()}
    district_to_tract = {}
    for tract, district in tract_to_district.items():
        try:
            district_to_tract[district].append(tract)
        except KeyError:
            district_to_tract[district] = [tract]
    return district_to_tract, tract_to_district