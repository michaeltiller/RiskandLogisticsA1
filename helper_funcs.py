import xpress as xp
import folium, pyproj
import pandas as pd
import geopandas as gpd
import numpy as np
from pyproj import Transformer


def get_basic_summary_sol(probs, xs,ys,zs, time_index, product_index, costs):

    setup, operating, sup_ware, ware_cust = costs
    print("t\twarehouses operating, sup_ware, ware_cust")

    for t in time_index:
        print(t, end="\t")
        n_ware_t = sum( v for k, v in ys.items() if k[1]==t )
        print(n_ware_t, end="\t")

        # the >8 means put it to the right within 8 spaces 
        # the , means use comma seperation
        # the .0f means no decimals
        print(f"{operating[t]:>10,.0f} {sup_ware[t]:>8,.0f} {ware_cust[t]:>8,.0f}")
    
    print(f"setup costs were {setup:,.0f}")

def put_solution_on_map(probs, xs, ys, zs, cand_gdf:gpd.GeoDataFrame, cust_gdf, supp_gdf,
                         time_index=range(1,10+1), product_index=[1,2,3,4]):
    
    t =max(time_index)
    m = folium.Map(location=[cand_gdf['lat'].mean(), cand_gdf['lon'].mean()], zoom_start=7)

    cand_jitter = (0,-.1) # need to move warehouses as they overlap with customers

    #show the warehouses 
    for j in cand_gdf.index:

        cust = cand_gdf.loc[j]

        folium.Marker(
            location= ( cust["lat"]+cand_jitter[0], cust["lon"]+cand_jitter[1] ),
            icon=folium.Icon(
                icon="warehouse",
                prefix="fa",
                color= "red" if ys[j,t] else "white", 
                icon_color="grey"
            )
        ).add_to(m)

    #show the customers
    for i in cust_gdf.index:
        cust = cust_gdf.loc[i]
        cust_loc = cust_gdf.loc[i,["lat","lon"]].values 
        folium.Marker(
            location= cust_loc,
            icon=folium.Icon(
                icon="house",
                prefix="fa",
                color= "blue",
                icon_color="grey"
            )
        ).add_to(m)   

        #show warehouse to customer links
        for j in cand_gdf.index:
            if max(xs[i,j,t,p] for p in product_index) > 0 :
                
                ware_loc  = cand_gdf.loc[j, ["lat", "lon"]].values + cand_jitter
                txt = f"W{j} -> C{i}"
                txt += "P"+ ",".join(str(p) for p in product_index if xs[i,j,t,p] )

                folium.PolyLine(
                    locations=[cust_loc, ware_loc],
                    color="blue",
                    weight=3,
                    tooltip=txt
                ).add_to(m)

    m.show_in_browser()




def print_sol_status(solved_prob):
    sol_status = solved_prob.attributes.solstatus

    if sol_status == xp.SolStatus.OPTIMAL:
        print("Optimal solution found")
        best_obj = solved_prob.attributes.objval
        best_bound = solved_prob.attributes.bestbound
        mip_gap = abs(best_obj - best_bound) / (1e-10 +abs(best_obj))
        print(f"Objval: {best_obj:,.0}\t MIP Gap: {mip_gap*100:.2f}%")
        
    elif sol_status == xp.SolStatus.FEASIBLE:
        print("Feasible solution (not proven optimal)")
        best_obj = solved_prob.attributes.objval
        best_bound = solved_prob.attributes.bestbound
        mip_gap = abs(best_obj - best_bound) / (1e-10 +abs(best_obj))
        print(f"Objval: {int(best_obj):,}\t MIP Gap: {mip_gap*100:.2f}%")

    elif sol_status == xp.SolStatus.INFEASIBLE:
        print("Model is infeasible")
    elif sol_status == xp.SolStatus.UNBOUNDED:
        print("Model is unbounded")
    else:
        print("No solution available")

def get_all_data(data_dir="CaseStudyDataPY"):
    #for converting coords
    transformer = Transformer.from_crs(
        "EPSG:27700",
        "EPSG:4326",
        always_xy=True
    )

    Suppliers_df = pd.read_csv(f"{data_dir}/Suppliers.csv", index_col=0)

    PostcodeDistricts_df = pd.read_csv(f"{data_dir}/PostcodeDistricts.csv", index_col=0)

    PostcodeDistricts_df["lon"], PostcodeDistricts_df["lat"] = transformer.transform(
        PostcodeDistricts_df["X (Easting)"].values,
        PostcodeDistricts_df["Y (Northing)"].values
    )

    Candidates_df = pd.read_csv(f"{data_dir}/Candidates.csv", index_col=0)

    Candidates_df["lon"], Candidates_df["lat"] = transformer.transform(
        Candidates_df["X (Easting)"].values,
        Candidates_df["Y (Northing)"].values
    )


    # -----------------------------------------------------------------------------
    # Read distance matrices
    # Supplier → District distances
    # District → District distances
    # Column names are converted from strings to integers for correct .loc indexing
    # -----------------------------------------------------------------------------
    DistanceSupplierDistrict_df = pd.read_csv(
        f"{data_dir}/Distance Supplier-District.csv", index_col=0
    )
    DistanceSupplierDistrict_df.columns = DistanceSupplierDistrict_df.columns.astype(int)

    DistanceDistrictDistrict_df = pd.read_csv(
        f"{data_dir}/Distance District-District.csv", index_col=0
    )
    DistanceDistrictDistrict_df.columns = DistanceDistrictDistrict_df.columns.astype(int)


    # -----------------------------------------------------------------------------
    # Read aggregate demand data (no time dimension)
    # Creates a dictionary keyed by (Customer, Product)
    # -----------------------------------------------------------------------------
    Demand_df = pd.read_csv(f"{data_dir}/Demand.csv")
    Operating_costs_df = pd.read_csv(f"{data_dir}/Operating.csv", index_col=0)["Operating cost"].to_dict()

    # -----------------------------------------------------------------------------
    # Read demand data with time periods
    # Creates a dictionary keyed by (Customer, Product, Period)
    # -----------------------------------------------------------------------------
    DemandPeriods_df = pd.read_csv(f"{data_dir}/DemandPeriods.csv")
    nbPeriods = DemandPeriods_df["Period"].max()
    DemandPeriods_df = (
        DemandPeriods_df
            .set_index(["Customer", "Product", "Period"])["Demand"]
            .to_dict()
    )


    # -----------------------------------------------------------------------------
    # Read demand data with time periods and scenarios
    # Creates a dictionary keyed by (Customer, Product, Period, Scenario)
    # -----------------------------------------------------------------------------
    DemandPeriodsScenarios_df = pd.read_csv(f"{data_dir}/DemandPeriodScenarios.csv")
    nbScenarios = DemandPeriodsScenarios_df["Scenario"].max()
    DemandPeriodsScenarios_df = (
        DemandPeriodsScenarios_df
            .set_index(["Customer", "Product", "Period", "Scenario"])["Demand"]
            .to_dict()
    )

    return (
        PostcodeDistricts_df, Candidates_df, Suppliers_df,
        Demand_df, DemandPeriods_df, DemandPeriodsScenarios_df,
        Operating_costs_df, DistanceSupplierDistrict_df, DistanceDistrictDistrict_df,
        nbPeriods, nbScenarios
    )