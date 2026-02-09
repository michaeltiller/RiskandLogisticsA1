import xpress as xp
import folium, pyproj
import pandas as pd
import geopandas as gpd
import numpy as np

def get_basic_summary_sol(probs, xs,ys,zs):

    n_ware = len(ys.values())
    print(f"{sum(v for v in ys.values()):,}/{n_ware} warehouses built")

def put_solution_on_map(probs, xs, ys, zs, cand_gdf:gpd.GeoDataFrame, cust_gdf, supp_gdf,
                         time_index=range(1,10+1), product_index=[1,2,3,4]):
    
    t =max(time_index)
    m = folium.Map(location=[cand_gdf['lat'].mean(), cand_gdf['lon'].mean()], zoom_start=7)

    #show the warehouses 
    for j in cand_gdf.index:

        cust = cand_gdf.loc[j]
        folium.Marker(
            location= (cust["lat"],cust["lon"]),
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
                # cust_lat, cust_lon  = cust_gdf.loc[i, ["lat", "lon"]]
                ware_loc  = cand_gdf.loc[j, ["lat", "lon"]].values

                folium.PolyLine(
                    locations=[cust_loc, ware_loc],
                    color="blue",
                    weight=3,
                    tooltip=f"W{j} -> C{i}"
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
        print(f"Objval: {best_obj:,.0}\t MIP Gap: {mip_gap*100:.2f}%")

    elif sol_status == xp.SolStatus.INFEASIBLE:
        print("Model is infeasible")
    elif sol_status == xp.SolStatus.UNBOUNDED:
        print("Model is unbounded")
    else:
        print("No solution available")


    