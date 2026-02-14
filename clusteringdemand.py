import pandas as pd
from sklearn.cluster import KMeans
from pyproj import Transformer
from shapely.geometry import MultiPoint, Point
from geopy.distance import great_circle
import geopandas as gpd
import folium
from folium import CircleMarker
import numpy as np
from time import perf_counter
from helper_funcs import *
import platform

#I hope moving this wont break your code michael. Apologies in advance
# num_clusters = 60


# data_dir = "CaseStudyDataPY"

# PostcodeDistricts = pd.read_csv(f"{data_dir}/PostcodeDistricts.csv", index_col=0)
# Candidates_df = pd.read_csv(f"{data_dir}/Candidates.csv")
# Demand_df = pd.read_csv(f"{data_dir}/Demand.csv")


# map demand to the candidate location 

def calcClusters(Demand_df: pd.DataFrame, Candidates_df:pd.DataFrame, num_clusters:int):


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

    #make a dictionary out of it
    DemandPeriods = DemandPeriods_df.to_dict()["Demand"]

    All_Candidates_df = All_Candidates_df.set_index("Candidate ID") # i think this got unset somehow
    
    return All_Candidates_df, reduced_Candidates_df, reduced_demand_df, DemandPeriods
    
    
    # creates candidates, then also need seperate df which has demand per product type for each candidate # so are we still using 400 customers and just 60 candidate locations to build?

def get_weighted_travel_costs(reduced_customers_df, cost_ware_cust, all_cust_df):

    #initialisation
    agg_cost_ware_cust = {
        (j,central_cust): 0
        for j,_ in cost_ware_cust 
        for central_cust in reduced_customers_df.index 
    }


    for (j,i), cost in cost_ware_cust.items():

        central_cust_for_i = reduced_customers_df.index[ all_cust_df.loc[i,"cluster label"] ]

        agg_cost_ware_cust[j, central_cust_for_i] += cost

    return agg_cost_ware_cust

def aggregate_warehouses_subproblem(num_warehouses:int, candidates_index, customer_index, travel_costs):
    """
    Select ``num_warehouses`` warehouses that have minimal transport costs by solving an IP allocation problem.
    This Docstring thing autocompleted, never seen that.
    
    :param num_warehouses: number of warehouses we want to reduce to
    :type num_warehouses: int
    :param candidates_index: index set for all 400 warehouses
    :param customer_index: index set for the REDUCED set of customers
    :param travel_costs: dictionary of travel costs for going from warehouse to customer
    """
    if platform.system()== "Windows":
        xp.init('c:/xpressmp/bin/xpauth.xpr')

    subprob = xp.problem("transport cost subproblem")
    Warehouses = candidates_index
    Customers = customer_index

    # customer allocations
    x = { 
        (i,j): subprob.addVariable(name=f"X_{i},{j}", vartype=xp.binary) 
        for j in Warehouses for i in Customers
    }
    #warehouses open or closed
    y = {
        j: subprob.addVariable(name=f"Y{j}", vartype=xp.binary)
        for j in Warehouses
    }
    
    # All customers must be allocated a warehouse(s)
    subprob.addConstraint(
        xp.Sum(
            x[i,j]
            for j in Warehouses
        )
        ==1
        for i in Customers
    )

    #We can only allocate from open Warehouses
    subprob.addConstraint(
        x[i,j] <= y[j]
        for i in Customers for j in Warehouses
    )

    # We want there to be num_warehouses open
    subprob.addConstraint(
        xp.Sum(
            y[j]
            for j in Warehouses
        )
        == num_warehouses
    )

    # we dont need the solution of the subproblem to be that good
    # It is better than k means if the original problem has better objvals, in the same configurations
    # when using this subproblem
    subprob.setControl('miprelstop', .05) # stop once the mip gap is below 5%
    subprob.controls.maxtime = -60*5 # stops after 3 mins
    xp.setOutputEnabled(False)

    # minimise transport costs
    subprob.setObjective(
        xp.Sum(
            travel_costs[j,i] * x[i,j]
            for i in Customers for j in Warehouses
        ),
        sense = xp.minimize
    )

    subprob.solve()

    print_sol_status(subprob)

    sol_status = subprob.attributes.solstatus
    if sol_status not in ( xp.SolStatus.OPTIMAL, xp.SolStatus.FEASIBLE):
        raise "Fuck. Why the fuck did easy allocation model fail?"
    
    y = subprob.getSolution(y)
    chosen_warehouses = [
        j for j in Warehouses if y[j] == 1
    ]

    return chosen_warehouses



if __name__ == "__main__":

    if platform.system()== "Windows":
        xp.init('c:/xpressmp/bin/xpauth.xpr')

    # num_clusters = 60


    # data_dir = "CaseStudyDataPY"

    # PostcodeDistricts = pd.read_csv(f"{data_dir}/PostcodeDistricts.csv", index_col=0)
    # Candidates_df = pd.read_csv(f"{data_dir}/Candidates.csv")
    # Demand_df = pd.read_csv(f"{data_dir}/Demand.csv")

    # # this was running when i imported calcClusters
    # All_Candidates_df, reduced_Candidates_df, reduced_demand_df, _ = calcClusters(Demand_df, Candidates_df, num_clusters) 

    # m = folium.Map(
    #     location=[All_Candidates_df['lat'].mean(), All_Candidates_df['lon'].mean()],
    #     zoom_start=13
    # )

    # for _, row in All_Candidates_df.iterrows():
    #     folium.CircleMarker(
    #         location=[row['lat'], row['lon']],
    #         radius=4 if row["is_cluster_centre"] else 2,
    #         color= "red" if row["is_cluster_centre"] else "blue",
    #         fill=True,
    #         fill_color="green",
    #         fill_opacity=1 if row["is_cluster_centre"] else 0.6,
    #     ).add_to(m)
    # m.show_in_browser()
    # m.save("clusteredloc.html")
    (
        PostcodeDistricts_df, Candidates_df, Suppliers_df,
        Demand_df, DemandPeriods, DemandPeriodsScenarios_df,
        Operating_costs_df, DistanceSupplierDistrict_df, DistanceDistrictDistrict_df,
        nbPeriods, nbScenarios
    ) = get_all_data("CaseStudyDataPY")

    #cluster the customer locations and take the aggregated demand
    _, reduced_Customers_df, _, _  = calcClusters(Demand_df, Candidates_df, num_clusters=200)

    reduced_Customers = reduced_Customers_df.index
    all_Candidates = Candidates_df.index 


    # Vehicle capacity in tonnes
    VehicleCapacity = {
        1: 9.0,
        2: 2.4,
        3: 1.5
    }

    # Cost in pounds per mile travelled (fixed cost)
    VehicleCostPerMileOverall = {
        1: 1.666,
        2: 1.727,
        3: 1.285
    }

    # Cost in pounds per mile and tonne transported (variable cost)
    VehicleCostPerMileAndTonneOverall = {
        1: 0.185,
        2: 0.720,
        3: 0.857
    }


    CostCandidateCustomers = {
        (j, i): 2
        * DistanceDistrictDistrict_df.loc[j, i]
        * VehicleCostPerMileAndTonneOverall[3]
        / 1000
        for j in all_Candidates
        for i in reduced_Customers
    }



    num_warehouses = 60

    sub_start = perf_counter() 
    print(f"solving subproblem of finding {num_warehouses} warehouses which have minimal transport cost")

    reduced_warehouses_index = aggregate_warehouses_subproblem(num_warehouses, all_Candidates, reduced_Customers, CostCandidateCustomers )

    sub_end = perf_counter()
    print(f"subproblem took {pretty_print_seconds(sub_end-sub_start)}")
    # print(np.array(reduced_warehouses_index))

    m = folium.Map(
        location=[Candidates_df['lat'].mean(), Candidates_df['lon'].mean()],
        zoom_start=7
    )

    
    for i in reduced_Customers:
        row = Candidates_df.loc[i]

        folium.CircleMarker(
            location= (row["lat"], row["lon"]),
            radius=4,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=1
        ).add_to(m)
    
    for j in reduced_warehouses_index:
            row = Candidates_df.loc[j]

            col = "red"
            rad = 6
        
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=rad ,
                color=col ,
                fill=True,
                fill_color=col,
                fill_opacity=1 ,
            ).add_to(m)


    m.show_in_browser()








