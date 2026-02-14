#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 15:14:53 2026

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
import numpy as np
num_clusters = 60
data_dir = "CaseStudyDataPY"


def calcClustersv2(Demand_df: pd.DataFrame, Candidates_df:pd.DataFrame, DemandPeriodsScenarios_df:pd.DataFrame, num_clusters:int):


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
    All_Candidates_df["cluster label"] = cluster_labels
    
    reduced_Candidates_df = All_Candidates_df.loc[ All_Candidates_df["is_cluster_centre"] ]
    
    reduced_ids = list(reduced_Candidates_df['Candidate ID'])
    
    reduced_demand_df = Demand_df[Demand_df['Customer'].isin(reduced_ids)]

    # we need to aggregate demand per cluster across time
    DemandPeriods_df = pd.read_csv(f"CaseStudyDataPY/DemandPeriods.csv")
    #go from candidate to their cluster
    map_cand_id_to_cluster_label = {
         cand_id:label 
         for cand_id, label in zip(All_Candidates_df["Candidate ID"], cluster_labels)
    }
    #get the cluster for each entry i.e customer time, period 
    DemandPeriods_df["cluster label"] = DemandPeriods_df["Customer"].apply(lambda x: map_cand_id_to_cluster_label[x])
   
    DemandPeriods_df = DemandPeriods_df.astype(int)

    #aggregate by CLUSTER, product and time
    DemandPeriods_df = DemandPeriods_df.groupby(["cluster label", "Product", "Period"]).sum()

    #then we switch from cluster to the cluster centre
    DemandPeriods_df = DemandPeriods_df.rename(
        level=0,
        index = lambda cluster_label:reduced_Candidates_df.index.array[cluster_label] ) 
    
    DemandPeriods_df = DemandPeriods_df.drop(columns="Customer")
    
    DemandPeriods_df.index.get_level_values('cluster label').unique()

    
    #make a dictionary out of it
    DemandPeriods = DemandPeriods_df.to_dict()["Demand"]
    
    
    DemandPeriodsScenarios_df = pd.read_csv(f"{data_dir}/DemandPeriodScenarios.csv")
    DemandPeriodsScenarios_df["cluster label"] = DemandPeriodsScenarios_df["Customer"].apply(lambda x: map_cand_id_to_cluster_label[x])
    DemandPeriodsScenarios_df = DemandPeriodsScenarios_df.astype(int)
    DemandPeriodsScenarios_df = DemandPeriodsScenarios_df.groupby(["cluster label", "Product", "Period", "Scenario"]).sum()
    
    # scenarios = DemandPeriodsScenarios_df.index.get_level_values("Scenario")

    # DemandPeriodsScenarios_df = DemandPeriodsScenarios_df[
    #     (scenarios >= 1) & (scenarios <= 10)
    #     ]
    
    
    DemandPeriodsScenarios_df = DemandPeriodsScenarios_df.rename(
        level=0,
        index = lambda cluster_label:reduced_Candidates_df.index.array[cluster_label] ) 
    DemandPeriodsScenarios_df = DemandPeriodsScenarios_df.drop(columns="Customer")
    
    
    DemandPeriodsScenarios = DemandPeriodsScenarios_df.to_dict()["Demand"]
    
    return All_Candidates_df, reduced_Candidates_df, reduced_demand_df, DemandPeriods, DemandPeriodsScenarios