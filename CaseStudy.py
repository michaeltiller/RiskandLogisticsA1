# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 13:58:41 2026

@author: ksearle
"""

import pandas as pd
import xpress as xp
import os
import zipfile


zip_path = "CaseStudyDataPY.zip"
data_dir = "CaseStudyDataPY"

# -----------------------------------------------------------------------------
# Create directory and extract CSVs if it doesn't exist
# -----------------------------------------------------------------------------
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)


# -----------------------------------------------------------------------------
# Read supplier data
# The first column is used as the supplier index
# -----------------------------------------------------------------------------
Suppliers_df = pd.read_csv(f"{data_dir}/Suppliers.csv", index_col=0)
# Maximum supplier index (assumed to be integer-indexed)
nbSuppliers = Suppliers_df.index.max()


# -----------------------------------------------------------------------------
# Read postcode district data (used to define customers)
# -----------------------------------------------------------------------------
PostcodeDistricts = pd.read_csv(f"{data_dir}/PostcodeDistricts.csv", index_col=0)


# -----------------------------------------------------------------------------
# Read candidate facility data
# -----------------------------------------------------------------------------
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
Customers  = PostcodeDistricts.index
Candidates = Candidates_df.index
Suppliers  = Suppliers_df.index


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

# CO₂ emissions in kg per mile and tonne transported
VehicleCO2PerMileAndTonne = {
    1: 0.11,
    2: 0.31,
    3: 0.30
}


# -----------------------------------------------------------------------------
# Time periods and scenarios
# -----------------------------------------------------------------------------
Times = range(1, nbPeriods + 1)
Scenarios = range(1, nbScenarios + 1)


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
prob = xp.problem("Assignment 1")

# To turn on and off the solver log
xp.setOutputEnabled(True)


xp.setOutputEnabled(True)
prob.solve()

# =============================================================================
# Post-processing and data visualisation
# =============================================================================

sol_status = prob.attributes.solstatus

if sol_status == xp.SolStatus.OPTIMAL:
    print("Optimal solution found")
    best_obj = prob.attributes.objval
    best_bound = prob.attributes.bestbound
    mip_gap = abs(best_obj - best_bound) / (1e-10 +abs(best_obj))
    print(f"MIP Gap: {mip_gap*100:.2f}%")
    
elif sol_status == xp.SolStatus.FEASIBLE:
    print("Feasible solution (not proven optimal)")
    best_obj = prob.attributes.objval
    best_bound = prob.attributes.bestbound
    mip_gap = abs(best_obj - best_bound) / (1e-10 +abs(best_obj))
    print(f"MIP Gap: {mip_gap*100:.2f}%")
elif sol_status == xp.SolStatus.INFEASIBLE:
    print("Model is infeasible")
elif sol_status == xp.SolStatus.UNBOUNDED:
    print("Model is unbounded")
else:
    print("No solution available")