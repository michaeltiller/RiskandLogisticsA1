## requirements.txt
the python packages needed to run the other code

# clusterdemand.py
Contains the functions for the two methods of aggregation considered in this report: Weighted kmeans and for the candidates, solving an IP subproblem (see report)

# part a.py
Runs the code for the deterministic MECLWP in part b ( part a was actually the aggregation step but none of us noticed that). This contains all the model formulations given to xpress.

# StochasticFinal.py
Runs the code for the stochastactic MECWLP in part c. This constains all the model formulations given to xpress.

## running part a many times.py
This is essentially the code in part a.py. It is reworked to run the deterministic model on different inputs in order to compare two aggregation methods (see report). It's output is dumped to "part a comparison Subprob.txt".

## helper_funcs.py
Utility functions for loading data and analysing the results of the deterministic problem e.g. solution status, cost breakdown, plotting the solution.

## barplots.py
Used to plot comparison results in report. Uses the data in "part a comparison Subprob.txt".

## .gitignore
Some files, like folium maps and roughwork code, we didnt want to circulate on git.

## assignment.pdf
The assignment description.

## part a comparison Subprob.txt
Contains the data from running "running part a many times".py