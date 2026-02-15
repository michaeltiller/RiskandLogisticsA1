import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_and_prepare_data(file_path):
    """Load warehouse data and sort it for plotting."""
    df = pd.read_csv(file_path)
    df["warehouses"] = pd.to_numeric(df["warehouses"], errors="coerce")
    df = df.sort_values(["warehouses", "method"])
    return df

def create_comparison_chart(df, value_column, y_label, chart_title):
    """Create a grouped bar chart comparing methods across warehouse counts."""
    warehouses = sorted(df["warehouses"].unique())
    methods = df["method"].unique()
    
    x_positions = np.arange(len(warehouses))
    bar_width = 0.35
    
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
    data_file = Path.home() / "Desktop" / "part_a_comparison_Subprob.txt"
    warehouse_data = load_and_prepare_data(data_file)
    
    # Chart 1: Computation time comparison
    create_comparison_chart(
        warehouse_data,
        value_column="time",
        y_label="Computation Time (seconds)",
        chart_title="How Computation Time Changes with Warehouse Count"
    )
    
    # Chart 2: Objective value comparison
    create_comparison_chart(
        warehouse_data,
        value_column="objval",
        y_label="Objective Value",
        chart_title="Objective Value Comparison Across Methods"
    )
    
    print("Analysis complete!")
