import numpy as np

def vecdist(s_lat, s_lng, e_lat, e_lng):

    # approximate radius of earth in km
    R = 6373

    s_lat = s_lat*np.pi/180.0
    s_lng = np.deg2rad(s_lng)
    e_lat = np.deg2rad(e_lat)
    e_lng = np.deg2rad(e_lng)

    d = np.sin((e_lat - s_lat)/2)**2 + np.cos(s_lat)*np.cos(e_lat) * np.sin((e_lng - s_lng)/2)**2

    return 2 * R * np.arcsin(np.sqrt(d))


def geo_to_euclidean_coords(lat, lon):
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    R = 6373
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return x, y, z


def euclidean_coords_to_geo(x, y, z):
    R = 6373
    lat = np.arcsin(z / R)
    lon = np.arctan2(y, x)
    lat = np.rad2deg(lat)
    lon = np.rad2deg(lon)
    return np.array([lon, lat]).T