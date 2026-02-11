import pandas as pd
from sklearn.cluster import KMeans
from pyproj import Transformer
from shapely.geometry import MultiPoint, Point
from geopy.distance import great_circle
import geopandas as gpd
import folium
from folium import CircleMarker
import numpy as np


num_clusters = 60


data_dir = "CaseStudyDataPY"

PostcodeDistricts = pd.read_csv(f"{data_dir}/PostcodeDistricts.csv", index_col=0)
Candidates_df = pd.read_csv(f"{data_dir}/Candidates.csv")
Demand_df = pd.read_csv(f"{data_dir}/Demand.csv")


# map demand to the candidate location 

def calcClusters(Demand_df, Candidates_df, num_clusters):


    demand_grouped = Demand_df.groupby('Customer')["Demand"].sum()
    
    Candidates_df = pd.merge(Candidates_df, demand_grouped, left_on = 'Candidate ID', right_on = demand_grouped.index, how ='left')
    Candidates_df = Candidates_df.rename(columns = {"Demand" : "Total Demand"})
    

    transformer = Transformer.from_crs(
        "EPSG:27700",
        "EPSG:4326",
        always_xy=True
    )
    
    #Transfrom from Easting and Northing values to lat and lon 
    Candidates_df["lon"], Candidates_df["lat"] = transformer.transform(
        Candidates_df["X (Easting)"].values,
        Candidates_df["Y (Northing)"].values
    )
    
    #Create array of lat and lon as well as the weight array of demand 
    arr = Candidates_df[['lat', 'lon']].to_numpy()
    kmeans_weights = Candidates_df['Total Demand'].to_numpy()

        
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
        return centermost_point
    
    centermost_points = clusters.map(get_centermost_point)
    
    centermost_points = pd.DataFrame(
        {
            "lat": centermost_points.map(lambda x: x[0]),
            "lon": centermost_points.map(lambda x: x[1]),
            "is_cluster_centre": 1
        }
    )
    All_Candidates_df = Candidates_df.merge(right=centermost_points, how="left", on=["lon", "lat"])
    # one hot encode the clustre centres
    All_Candidates_df["is_cluster_centre"] = All_Candidates_df["is_cluster_centre"].fillna(0).astype(bool)
    
    assert num_clusters == All_Candidates_df["is_cluster_centre"].sum()
    print(All_Candidates_df.head())
    
    reduced_Candidates_df = All_Candidates_df.loc[ All_Candidates_df["is_cluster_centre"] ]
    print(reduced_Candidates_df.head())
    
    reduced_ids = list(reduced_Candidates_df['Candidate ID'])
    
    reduced_demand_df = Demand_df[Demand_df['Customer'].isin(reduced_ids)]
    
    
    return All_Candidates_df, reduced_Candidates_df, reduced_demand_df
    
    
    # creates candidates, then also need seperate df which has demand per product type for each candidate # so are we still using 400 customers and just 60 candidate locations to build?
    
All_Candidates_df, reduced_Candidates_df, reduced_demand_df = calcClusters(Demand_df, Candidates_df, num_clusters) 




m = folium.Map(
    location=[All_Candidates_df['lat'].mean(), All_Candidates_df['lon'].mean()],
    zoom_start=13
)

for _, row in All_Candidates_df.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=4 if row["is_cluster_centre"] else 2,
        color= "red" if row["is_cluster_centre"] else "blue",
        fill=True,
        fill_color="green",
        fill_opacity=1 if row["is_cluster_centre"] else 0.6,
    ).add_to(m)
m.show_in_browser()
# m.save("clusteredloc.html")


