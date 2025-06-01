from sklearn.cluster import OPTICS, HDBSCAN
import numpy as np
import fast_tsp
import pandas as pd


def get_cluster(combined_gdf):
    data = combined_gdf["geometry"].centroid.get_coordinates().to_numpy()

    min_cluster_size = max(10, int(len(combined_gdf) / 200))

    clustering = HDBSCAN(min_cluster_size=min_cluster_size).fit(data)
    print(len(clustering.labels_))
    combined_gdf["clust"] = clustering.labels_
    print(len(set(clustering.labels_)))
    combined_gdf = combined_gdf.reindex(
        columns=[
            "type",
            "id",
            "geometry",
            "tourism",
            "name",
            "building",
            "addr:city",
            "building:use",
            "amenity",
            "smoking",
            "wheelchair",
            "clust",
        ]
    )
    return combined_gdf  # .explore(column='clust',cmap="Spectral",tiles="CartoDB positron")


def get_distance_matrix(combined_gdf):
    if -1 in combined_gdf["clust"].to_list():
        combined_gdf = combined_gdf[combined_gdf["clust"] != -1]

    clust_centroid = combined_gdf.dissolve(by="clust")[
        "geometry"
    ].centroid.reset_index()
    # clust_centroid

    # Extract x, y as NumPy arrays
    coords = np.array([[geom.x, geom.y] for geom in clust_centroid.geometry])

    # Efficient vectorized distance matrix
    diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt((diffs**2).sum(axis=2))

    # Optional: convert to DataFrame
    distance_matrix = pd.DataFrame(dists)
    return clust_centroid, distance_matrix


def get_route(distance_matrix):
    dists = (distance_matrix.values * 100).astype(int)
    tour = fast_tsp.find_tour(dists)
    return tour
