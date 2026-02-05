import xpress as xp

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