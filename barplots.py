import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_and_prepare_data(file_path):
    """Load warehouse data and sort it for plotting."""
    df = pd.read_csv(file_path)
    df = df.sort_values(["warehouses", "method"])
    df["time"] = df["time"]//60
    df["objval"] = df["objval"]/(1_000_000)
    df.loc[ df["method"]=="subprob", "method"] = "Sub Problem"
    
    return df

def create_comparison_chart(df, value_column, y_label, chart_title):
    """Create a grouped bar chart comparing methods across warehouse counts."""
    warehouses = sorted(df["warehouses"].unique())
    methods = df["method"].unique()
    
    x_positions = np.arange(len(warehouses))
    bar_width = 0.35
    small_bit_up = min(df[value_column].loc[df[value_column] !=0]) *.1

    plt.figure(figsize=(10, 6))
    
    for i, method in enumerate(methods):
        method_data = df[df["method"] == method].sort_values("warehouses")
        plt.bar(
            x_positions + i * bar_width,
            method_data[value_column],
            width=bar_width,
            label=method,
            alpha=0.8
        )
        plt.scatter(
            x=x_positions + i * bar_width,
            y=[small_bit_up for _ in warehouses],
            alpha= method_data[value_column] == 0,
            c = "red",
            s = 200,
            marker = "X"
        )
    
    plt.xticks(x_positions + bar_width / 2, warehouses)
    plt.xlabel("Number of Warehouses", fontsize=11)
    plt.ylabel(y_label, fontsize=11)
    plt.title(chart_title, fontsize=13, fontweight='bold')
    plt.legend(title="Method")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load the data
    data_file =  "part a comparison Subprob.txt"
    warehouse_data = load_and_prepare_data(data_file)
    
    print(warehouse_data)
    # Chart 1: Computation time comparison
    create_comparison_chart(
        warehouse_data,
        value_column="time",
        y_label="Solve Time (minutes)",
        chart_title="Solve time for increasing numbers of candidate locations"
    )
    
    # Chart 2: Objective value comparison
    create_comparison_chart(
        warehouse_data,
        value_column="objval",
        y_label="Total Cost (£ millions)",
        chart_title="Objective Value for increasing numbers of candidate locations"
    )
    
