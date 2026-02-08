import numpy as np
import pandas as pd
import xpress as xp
import platform
from helper_funcs import *


data_dir = "CaseStudyDataPY"
Suppliers_df = pd.read_csv(f"{data_dir}/Suppliers.csv", index_col=0)

PostcodeDistricts = pd.read_csv(f"{data_dir}/PostcodeDistricts.csv", index_col=0)

Candidates_df = pd.read_csv(f"{data_dir}/Candidates.csv", index_col=0)
# Maximum candidate index
nbCandidates = Candidates_df.index.max()


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
Demand = (
    Demand_df
        .set_index(["Customer", "Product"])["Demand"]
        .to_dict()
)
Operating_costs = pd.read_csv(f"{data_dir}/Operating.csv", index_col=0)["Operating cost"].to_dict()

# -----------------------------------------------------------------------------
# Read demand data with time periods
# Creates a dictionary keyed by (Customer, Product, Period)
# -----------------------------------------------------------------------------
DemandPeriods_df = pd.read_csv(f"{data_dir}/DemandPeriods.csv")
DemandPeriods = (
    DemandPeriods_df
        .set_index(["Customer", "Product", "Period"])["Demand"]
        .to_dict()
)

# Number of time periods
nbPeriods = DemandPeriods_df["Period"].max()


# -----------------------------------------------------------------------------
# Read demand data with time periods and scenarios
# Creates a dictionary keyed by (Customer, Product, Period, Scenario)
# -----------------------------------------------------------------------------
DemandPeriodsScenarios_df = pd.read_csv(f"{data_dir}/DemandPeriodScenarios.csv")
DemandPeriodsScenarios = (
    DemandPeriodsScenarios_df
        .set_index(["Customer", "Product", "Period", "Scenario"])["Demand"]
        .to_dict()
)

# Number of scenarios
nbScenarios = DemandPeriodsScenarios_df["Scenario"].max()


# =============================================================================
# Index sets
# =============================================================================
rng = np.random.RandomState(2026)
Customers = rng.choice(PostcodeDistricts.index, size=60, replace=False)
Candidates = rng.choice(Candidates_df.index, size =60, replace=False)
Suppliers = Suppliers_df.index
# Suppliers  = rng.choice(Suppliers_df.index, size=10, replace=False)

nbCustomers = len(Customers)
nbSuppliers = len(Suppliers)
nbCandidates = len(Candidates)
print(f"{nbCustomers=:,}\t{nbCandidates=:,}\t{nbSuppliers=:,}")

Times = range(1, nbPeriods + 1)
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
    (i,j,t,p): prob.addVariable(name=f"X__C{i}_W{j}_T{t}_P{p}", vartype=xp.binary)
    for i in Customers for j in Candidates for t in Times for p in Products
}
y = {
    (j,t): prob.addVariable(name=f"Y__W{j}_T{t}", vartype = xp.binary)
    for j in Candidates for t in Times
}

z = {
    (k,j,t,p): prob.addVariable(name=f"Z__S{k}_W{j}_T{t}_P{p}")
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


# We must meet all customer demands each year
prob.addConstraint(
    xp.Sum(
        DemandPeriods[i,p,t]*x[i,j,t,p] 
        for i in Customers for j in Candidates for t in Times for p in Products
    )
    >= 
    sum(
        DemandPeriods[i,p,t] 
        for i in Customers for t in Times for p in Products
    )
)
# constrain that suppliers only supply their product type
prob.addConstraint(
    z[k,j,t,p] == 0 
    for k in Suppliers for j in Candidates for p in Products for t in Times
    if p != Suppliers_df["Product group"][k]
)
######link warehouse stock supplier
# a warehouse can deliver no more than what it has in stock
prob.addConstraint(
    xp.Sum(
        Suppliers_df["Capacity"][k] * z[k,j,t,p]        #Into warehouse from suppliers
        for k in Suppliers 
        if Suppliers_df["Product group"][k] == p
    )
    >=
    xp.Sum(
        DemandPeriods[i,p,t]  * x[i,j,t,p] for i in Customers      #Out of warehouse to customers
    )
    for p in Products for j in Candidates for t in Times
)
######### Objective function
#minimise costs
#we know that if we build a warehouse it will be open in year 10
# so we can use year 10 to calculate fixed costs
warehouse_setup_costs = xp.Sum(
    Candidates_df["Setup cost"][j]*y[j,final_t]
    for j in Candidates
)
warehouse_operating_costs = xp.Sum(
    Operating_costs[j]*y[j,t] 
    for j in Candidates for t in Times
)
supplier_to_warehouse_costs = xp.Sum(
    CostSupplierCandidate[k,j]*z[k,j,t,p]
    for k in Suppliers for j in Candidates for t in Times for p in Products
)
warehouse_to_customer_costs = xp.Sum(
    CostCandidateCustomers[j,i]*x[i,j,t,p]
    for i in Customers for j in Candidates for t in Times for p in Products
)

prob.setObjective(
    warehouse_setup_costs + warehouse_operating_costs + supplier_to_warehouse_costs + warehouse_to_customer_costs
)

xp.setOutputEnabled(False)
print("Solving")
prob.controls.maxtime = -60*3 # stops after 3 mins
prob.solve()

# =============================================================================
# Post-processing and data visualisation
# =============================================================================

print_sol_status(prob)