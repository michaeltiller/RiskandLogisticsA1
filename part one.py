import numpy as np
import pandas as pd
import xpress as xp
import platform
from helper_funcs import *
from clusteringdemand import *
from time import perf_counter

(
    PostcodeDistricts_df, Candidates_df, Suppliers_df,
    Demand_df, DemandPeriods, DemandPeriodsScenarios_df,
    Operating_costs_df, DistanceSupplierDistrict_df, DistanceDistrictDistrict_df,
    nbPeriods, nbScenarios
) = get_all_data("CaseStudyDataPY")


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
    for j in Candidates_df.index
    for k in Suppliers_df.index
}

# Cost from candidate facilities to customers
# All transports use 3.5t vans (vehicle type 3)
CostCandidateCustomers = {
    (j, i): 2
    * DistanceDistrictDistrict_df.loc[j, i]
    * VehicleCostPerMileAndTonneOverall[3]
    / 1000
    for j in Candidates_df.index
    for i in Candidates_df.index
}

# =============================================================================
# Index sets
# =============================================================================
#put the aggregation function here
# and then aggregate demand from the clustered customers
# rng = np.random.RandomState(2026)
# Customers = rng.choice(PostcodeDistricts_df.index, size=40, replace=False)
# Candidates = rng.choice(Candidates_df.index, size =40, replace=False)

#cluster the customer locations and take the aggregated demand
all_Customers_df, reduced_Customers_df, _, DemandPeriods, _, _  = calcClusters(Demand_df, Candidates_df,DemandPeriodsScenarios_df, num_clusters=30)
Customers = reduced_Customers_df.index

########## this is where it gets  confusing
#cluster the warehouse locations 
num_warehouses = 30

# sub_start = perf_counter() 
# print(f"solving subproblem of finding {num_warehouses} warehouses which have minimal transport cost")
# reduced_warehouses_index = aggregate_warehouses_subproblem(num_warehouses, Candidates_df.index, Customers, CostCandidateCustomers )
# sub_end = perf_counter()
# print(f"subproblem took {pretty_print_seconds(sub_end-sub_start)}")
# Candidates = reduced_warehouses_index

_, reduced_Candidates_df, _, _, _, _  = calcClusters(Demand_df, Candidates_df, DemandPeriodsScenarios_df, num_clusters=num_warehouses)
Candidates = reduced_Candidates_df.index


Suppliers = Suppliers_df.index


nbCustomers = len(Customers)
nbSuppliers = len(Suppliers)
nbCandidates = len(Candidates)
print(f"{nbCustomers=:,}\t{nbCandidates=:,}\t{nbSuppliers=:,}")

Times = range(1, nbPeriods + 1)
Scenarios = range(1, nbScenarios + 1)
Products = (1,2,3,4) #hardcoding
final_t = max(Times)


agg_ware_cust_travel_costs = get_weighted_travel_costs(reduced_Customers_df, CostCandidateCustomers, all_Customers_df)

# =============================================================================
# Build optimization model
# =============================================================================

if platform.system()== "Windows":
    xp.init('c:/xpressmp/bin/xpauth.xpr')
else:
    print("lmk if that annoying message is coming up") #they did not lmk

prob = xp.problem("Assignment 1")

######## Decision variables 
x = {
    (i,j,t,p): prob.addVariable(name=f"X__C{i}_W{j}_T{t}_P{p}", vartype=xp.binary)
    for i in Customers for j in Candidates for t in Times for p in Products
}
y = {
    (j,t): prob.addVariable(name=f"Y__W{j}_T{t}", vartype = xp.binary)
    for j in Candidates for t in Times
}

z = {
    (k,j,t,p): prob.addVariable(name=f"Z__S{k}_W{j}_T{t}_P{p}", ub=1) #maybe specifying to xpress that the upper bound is one will help things
    for k in Suppliers for j in Candidates for t in Times for p in Products
}

########### Constraints
# we can only supply from a warehouse if it is built
prob.addConstraint(
    x[i,j,t,p] <= y[j,t]
    for i in Customers for j in Candidates for t in Times for p in Products
)
#if we build a warehouse it stays open 
prob.addConstraint(
    y[j,t] <= y[j,t+1]
    for j in Candidates for t in Times if t != max(Times)
)
if nbCandidates >30:
    prob.addConstraint(
        xp.Sum( y[j,t] for j in Candidates)
        <=
        nbCandidates//2
        for t in Times 
    )

# We must meet all customer demands, each year
prob.addConstraint(
    xp.Sum(
        x[i,j,t,p]
        for j in Candidates
    )
    == 1
    for i in Customers for p in Products for t in Times
)

# the z decision variables are percentages of supplier k's total stock of p sent to warehouse j at time t 
# constrain them less than one
prob.addConstraint(
    z[k,j,t,p] <= int( p == Suppliers_df["Product group"][k] )
    for k in Suppliers for j in Candidates for p in Products for t in Times
)

# we can supply out 100% of stock at most
prob.addConstraint(
    xp.Sum(
        z[k,j,t,p]
        for j in Candidates
    )
    <= 1
    for k in Suppliers for p in Products for t in Times
)
# # we only supply to open warehouses
# prob.addConstraint(
#     z[k,j,t,p] <= y[j,t]
#     for t in Times for k in Suppliers for j in Candidates for p in Products
# )



######link warehouse stock supplier
# a warehouse can deliver no more than what it has in stock
# assuming no stock gets carried over into the next year because fuck that its too complicated
prob.addConstraint(
    xp.Sum(
        Suppliers_df["Capacity"][k] * z[k,j,t,p]        #Into warehouse from suppliers
        for k in Suppliers for p in Products
    )
    >=
    xp.Sum(
        DemandPeriods[i,p,t] * x[i,j,t,p]              #Out of warehouse to customers
        for i in Customers  for p in Products                    
    )
    for j in Candidates for t in Times
)

# a warehouse has a capacity
prob.addConstraint(
    xp.Sum(
        Suppliers_df["Capacity"][k] * z[k,j,t,p]        #Into warehouse from suppliers
        for k in Suppliers for p in Products
    )
    <= Candidates_df["Capacity"][j]                     #Warehouse capacity
    for j in Candidates for t in Times
)


################################
# Objective function
#minimise costs
#################################
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
        CostSupplierCandidate[k,j] * Suppliers_df["Capacity"][k] * z[k,j,t,p]
        for k in Suppliers for j in Candidates for p in Products
    )
    for t in Times
}

warehouse_to_customer_costs = {
    t: xp.Sum(
        agg_ware_cust_travel_costs[j,i] * DemandPeriods[i,p,t] * x[i,j,t,p]
        for i in Customers for j in Candidates for p in Products
    )
    for t in Times
}

##################################
#Solving
##################################


prob.setControl('miprelstop', .05) # stop once the mip gap is below 5%
prob.controls.maxtime = -60*20 
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

x = { k:int(v) for k,v in prob.getSolution(x).items() }   # if x is not binary change this !!!
y = { k:int(v) for k,v in prob.getSolution(y).items() }
z = prob.getSolution(z)
costs = (warehouse_setup_costs, warehouse_operating_costs, supplier_to_warehouse_costs, warehouse_to_customer_costs)
costs = map(
    lambda v: prob.getSolution(v),
    costs
)

get_basic_summary_sol(prob,xs=x, ys = y, zs=y, time_index=Times, product_index=Products, costs=costs)

put_solution_on_map(
    probs=prob,
    xs=x, ys = y, zs=z,
    cand_gdf=Candidates_df.loc[Candidates], cust_gdf=PostcodeDistricts_df.loc[Customers], supp_gdf=Suppliers_df
    ,time_index=Times, product_index=Products
)