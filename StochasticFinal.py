import numpy as np
import pandas as pd
import xpress as xp
import platform
from helper_funcs import *
from clusteringdemand import calcClusters
from demandfuncstochastic import calcClustersv2
from time import perf_counter

(
    PostcodeDistricts_df, Candidates_df, Suppliers_df,
    Demand_df, DemandPeriods_df, DemandPeriodsScenarios_df,
    Operating_costs_df, DistanceSupplierDistrict_df, DistanceDistrictDistrict_df,
    nbPeriods, nbScenarios
) = get_all_data("CaseStudyDataPY")

# =============================================================================
# Index sets
# =============================================================================
#put the aggregation function here
# and then aggregate demand from the clustered customers
# rng = np.random.RandomState(2026)
# Customers = rng.choice(PostcodeDistricts_df.index, size=40, replace=False)
# Candidates = rng.choice(Candidates_df.index, size =40, replace=False)

########## this is where it gets  confusing
#cluster the warehouse locations 
_, reduced_Candidates_df, _, _, _  = calcClustersv2(Demand_df, Candidates_df,DemandPeriodsScenarios_df, num_clusters=30)
Candidates = reduced_Candidates_df.index

#cluster the customer locations and take the aggregated demand
_, reduced_Customers_df, _, _, DemandPeriodsScenarios  = calcClustersv2(Demand_df, Candidates_df,DemandPeriodsScenarios_df, num_clusters=30)
Customers = reduced_Customers_df.index


Suppliers = Suppliers_df.index

Times = range(1, nbPeriods + 1)
nbScenarios = 5
nbCustomers = len(Customers)
nbSuppliers = len(Suppliers)
nbCandidates = len(Candidates)
print(f"{nbCustomers=:,}\t{nbCandidates=:,}\t{nbSuppliers=:,}\t{nbScenarios=:,}")



Scenarios = range(1, nbScenarios + 1)
Products = (1,2,3,4) #hardcoding
final_t = max(Times)



# =============================================================================
# Vehicle-related data
# Vehicles are indexed as:
#   1 = 18t trucks
#   2 = 7.5t lorries
#   3 = 3.5t vans
# =============================================================================

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


# =============================================================================
# Transport cost calculations
# =============================================================================

# Cost from suppliers to candidate facilities
# Round-trip distance (factor 2)
# Cost depends on supplier vehicle type
# Division by 1000 converts from kg to tonnes
CostSupplierCandidate = {
    (k, j): 2
    * DistanceSupplierDistrict_df.loc[k, j]
    * VehicleCostPerMileAndTonneOverall[
        Suppliers_df.loc[k, "Vehicle type"]
    ]
    / 1000
    for j in Candidates
    for k in Suppliers
}

# Cost from candidate facilities to customers
# All transports use 3.5t vans (vehicle type 3)
CostCandidateCustomers = {
    (j, i): 2
    * DistanceDistrictDistrict_df.loc[j, i]
    * VehicleCostPerMileAndTonneOverall[3]
    / 1000
    for j in Candidates
    for i in Customers
}


# =============================================================================
# Build optimization model
# =============================================================================

if platform.system()== "Windows":
    xp.init('c:/xpressmp/bin/xpauth.xpr')
else:
    print("lmk if that annoying message is coming up")
prob = xp.problem("Assignment 1")

######## Decision variables 

x = {
    (i,j,t,p, s): prob.addVariable(name=f"X__C{i}_W{j}_T{t}_P{p}_S{s}", vartype=xp.binary)
    for i in Customers for j in Candidates for t in Times for p in Products for s in Scenarios
}

y = {
    (j,t): prob.addVariable(name=f"Y__W{j}_T{t}", vartype = xp.binary)
    for j in Candidates for t in Times
}

z = {
    (k,j,t,p, s): prob.addVariable(name=f"Z__S{k}_W{j}_T{t}_P{p}_S{s}", ub=1) #idk if this helps the solver
    for k in Suppliers for j in Candidates for t in Times for p in Products for s in Scenarios
}

########### Constraints
# we can only supply from a warehouse if it is built
prob.addConstraint(
    x[i,j,t,p, s] <= y[j,t]
    for i in Customers for j in Candidates for t in Times for p in Products for s in Scenarios
)
#if we build a warehouse it stays open 
prob.addConstraint(
    y[j,t] <= y[j,t+1]
    for j in Candidates for t in Times if t != max(Times)
)




# We must meet all customer demands, each year
prob.addConstraint(
    xp.Sum(
        x[i,j,t,p, s]
        for j in Candidates
    )
    == 1
    for i in Customers for p in Products for t in Times for s in Scenarios
)

# the z decision variables are percentages of supplier k's total stock of p sent to warehouse j at time t 
# constrain them less than one and summing less than one
# and force them to zero if the supplier doesnt supply that product
prob.addConstraint(
    z[k,j,t,p, s] <= int( p == Suppliers_df["Product group"][k] )
    for k in Suppliers for j in Candidates for p in Products for t in Times for s in Scenarios
)
# we can supply out 100% of stock at most
prob.addConstraint(
    xp.Sum(
        z[k,j,t,p, s]
        for j in Candidates
    )
    <= 1
    for k in Suppliers for p in Products for t in Times for s in Scenarios
)
# we only supply to open warehouses - this helps to do 30,30 10scen in 10 mins
prob.addConstraint(
    z[k,j,t,p,s] <= y[j,t]
    for t in Times for k in Suppliers for j in Candidates for p in Products for s in Scenarios
)

######link warehouse stock supplier
# a warehouse can deliver no more than what it has in stock
# assuming no stock gets carried over into the next year because fuck that its too complicated
#Double check where the for s goes #ND looks good
prob.addConstraint(
    xp.Sum(
        Suppliers_df["Capacity"][k] * z[k,j,t,p, s]        #Into warehouse from suppliers
        for k in Suppliers for p in Products
    )
    ==
    1*(xp.Sum(
        DemandPeriodsScenarios[i,p,t,s] * x[i,j,t,p, s]             #Out of warehouse to customers
        for i in Customers  for p in Products               
    ) )
    for j in Candidates for t in Times for s in Scenarios
)

# a warehouse has a capacity
prob.addConstraint(
    xp.Sum(
        Suppliers_df["Capacity"][k] * z[k,j,t,p, s]        #Into warehouse from suppliers
        for k in Suppliers for p in Products
    )
    <= Candidates_df["Capacity"][j]                     #Warehouse capacity
    for j in Candidates for t in Times for s in Scenarios
)


######### Objective function
#minimise costs
#we know that if we build a warehouse it will be open in year 10
# so we can use year 10 to calculate fixed costs
warehouse_setup_costs = xp.Sum(
    Candidates_df["Setup cost"][j]*y[j,final_t]
    for j in Candidates
)
warehouse_operating_costs = {
    t: xp.Sum(
        Operating_costs_df[j]*y[j,t] 
        for j in Candidates
    )
    for t in Times
}


supplier_to_warehouse_costs = {
    t: xp.Sum(
        1/nbScenarios * (
        CostSupplierCandidate[k,j] * Suppliers_df["Capacity"][k] * z[k,j,t,p,s] ) 
        for k in Suppliers for j in Candidates for p in Products for s in Scenarios
    )
    for t in Times
}


warehouse_to_customer_costs = {
    t: xp.Sum(
        1/nbScenarios * (
        CostCandidateCustomers[j,i] * DemandPeriodsScenarios[i,p,t,s] * x[i,j,t,p,s] ) 
        for i in Customers for j in Candidates for p in Products for s in Scenarios
    )
    for t in Times
}

prob.setControl('miprelstop', .05) # stop once the mip gap is below 5%
prob.controls.maxtime = -10*60 # stops after 3 mins
prob.setObjective(
    warehouse_setup_costs + xp.Sum(
    warehouse_operating_costs[t] + supplier_to_warehouse_costs[t] + warehouse_to_customer_costs[t]
    for t in Times
    )
    ,sense = xp.minimize
)

xp.setOutputEnabled(False)
start_time = perf_counter()
print(f"Solving a problem with {prob.getAttrib('rows'):,} rows and {prob.getAttrib("cols"):,} columns")
prob.solve()
end_time = perf_counter()
print(f"took {pretty_print_seconds(end_time-start_time)} for a problem with {prob.getAttrib('rows'):,} rows and {prob.getAttrib("cols"):,} columns")


# =============================================================================
# Post-processing and data visualisation
# =============================================================================
print_sol_status(prob)

#x = { k:int(v) for k,v in prob.getSolution(x).items() }   # if x is not binary change this !!!
y = { k:int(v) for k,v in prob.getSolution(y).items() }
#z = prob.getSolution(z)
costs = (warehouse_setup_costs, warehouse_operating_costs, supplier_to_warehouse_costs, warehouse_to_customer_costs)
costs = map(
    lambda v: prob.getSolution(v),
    costs
)



setup, operating, sup_ware, ware_cust = costs
#print(f"t\tware\t{"operating":>10} {"supp->ware":>10} {"ware->cust":>10}")
print("t\twarehouses operating, sup_ware, ware_cust")

for t in Times:

    n_ware_t = sum( v for k, v in y.items() if k[1]==t )

    # the >10 means put it to the right within 10 spaces 
    # the , means use comma seperation
    # the .0f means no decimals
    print(f"{t}\t{n_ware_t:>3}\t{operating[t]:>10,.0f} {sup_ware[t]:>10,.0f} {ware_cust[t]:>10,.0f}")
    
    #print(f"setup costs were {setup:,.0f}")





probs=prob
ys = prob.getSolution(y)
zs = prob.getSolution(z)
ys = prob.getSolution(y)
zs = prob.getSolution(z)
cand_gdf=Candidates_df.loc[Candidates]
cust_gdf=PostcodeDistricts_df.loc[Customers] 
supp_gdf=Suppliers_df
time_index=Times 
product_index=Products

Scenarios


t = max(time_index)
s=nbScenarios
m = folium.Map(location=[cand_gdf['lat'].mean(), cand_gdf['lon'].mean()], zoom_start=7)

cand_jitter = np.array([0,-.1]) # need to move warehouses as they overlap with customers


#show the suppliers
for k in supp_gdf.index:

    supp = supp_gdf.loc[k]
    supp_loc = supp["lat"], supp["lon"]
    folium.Marker(
        location= supp_loc,
        icon=folium.Icon(
            icon="industry",
            prefix="fa",
            color= "green" if max(zs[k,j,t,p,s] for p in product_index for j in cand_gdf.index) else "white",
            icon_color="black"
        )
    ).add_to(m)
    


#show the warehouses 
for j in cand_gdf.index:

    ware = cand_gdf.loc[j]

    folium.Marker(
        location= ( ware["lat"]+cand_jitter[0], ware["lon"]+cand_jitter[1] ),
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
 

m.show_in_browser()




