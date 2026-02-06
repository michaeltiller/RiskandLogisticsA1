#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 11:00:03 2026

@author: michael
"""
import pandas as pd
from sklearn.cluster import KMeans
from pyproj import Transformer
from shapely.geometry import MultiPoint, Point
from geopy.distance import great_circle
import geopandas as gpd
import folium
from folium import CircleMarker


num_clusters = 60


data_dir = "CaseStudyDataPY"

PostcodeDistricts = pd.read_csv(f"{data_dir}/PostcodeDistricts.csv", index_col=0)
Candidates_df = pd.read_csv(f"{data_dir}/Candidates.csv")
demand_df = pd.read_csv(f"{data_dir}/Demand.csv")


# map demand to the candidate location 


demand_grouped = demand_df.groupby('Customer')['Demand'].sum().reset_index()


Candidates_df = pd.merge(Candidates_df, demand_grouped, left_on = 'Candidate ID', right_on = 'Customer', how ='left')


transformer = Transformer.from_crs(
    "EPSG:27700",
    "EPSG:4326",
    always_xy=True
)

Candidates_df["lon"], Candidates_df["lat"] = transformer.transform(
    Candidates_df["X (Easting)"].values,
    Candidates_df["Y (Northing)"].values
)


arr = Candidates_df[['lat', 'lon']].to_numpy()
kmeans_weights = Candidates_df['Demand'].to_numpy()



# cand do weights as raw weights or could do z score

kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(arr, sample_weight=kmeans_weights)

cluster_labels = kmeans.labels_ 


clusters = pd.Series([arr[cluster_labels == n] for n in range(num_clusters)])

def get_centermost_point(cluster):
    points = [(lon, lat) for lat, lon in cluster]
    centroid = MultiPoint(points).centroid
    centermost_point = min(
        cluster,
        key=lambda point: great_circle(point, (centroid.y, centroid.x)).m
    )
    return centermost_point  # 

centermost_points = clusters.map(get_centermost_point)

temp = gpd.GeoDataFrame(
    {
        "lat": centermost_points.map(lambda x: x[0]),
        "lon": centermost_points.map(lambda x: x[1]),
    },
    geometry=[Point(lon, lat) for lat, lon in centermost_points],
    crs="EPSG:4326"
)


m = folium.Map(
    location=[temp['lat'].mean(), temp['lon'].mean()],
    zoom_start=13
)

for _, row in temp.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=2,
        color="green",
        fill=True,
        fill_color="green",
        fill_opacity=0.6,
    ).add_to(m)

m.save("clusteredloc.html")


