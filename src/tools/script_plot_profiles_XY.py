import os
import sys
import json
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend instead of QtAgg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # ajoute le dossier 'src' au PYTHONPATH

import shutil
from datetime import datetime
from datetime import date

# Parameters
target_path = "/home/loi/flagAutomata/data_plots/simulationAnalysis/sliding_puzzle_coordinates_2025-06-23_23-08-40_rgb-italian-flag_16x16/learning/data_all_runs/data_env0_flag_target.csv"
flag_path = "/home/loi/flagAutomata/data_plots/simulationAnalysis/sliding_puzzle_coordinates_2025-06-23_23-08-40_rgb-italian-flag_16x16/learning/run_000/data/data_env0_flag/data_env_flag_run_000_gen_00001_eval_0000020.csv"
nb_rows, nb_cols = 16, 16
first_elem_size = 2
row_indexes_to_inspect = [0, 7, 15]
cols_indexes_to_inspect = [0, 7, 15]

# Build a new folder
today_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
output_dir = f"profilesXY_{today_str}"
os.makedirs(output_dir, exist_ok=True)

# Save a copy of the originals flags and their paths
shutil.copy(flag_path, os.path.join(output_dir, os.path.basename(flag_path)))
shutil.copy(target_path, os.path.join(output_dir, os.path.basename(target_path)))

with open(os.path.join(output_dir, "used_paths.txt"), "w") as f:
    f.write("Original file paths used in this analysis:\n")
    f.write(f"Target path: {os.path.abspath(target_path)}\n")
    f.write(f"Flag path:   {os.path.abspath(flag_path)}\n")


# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
colors = {'Target': 'crimson', 'Flag': 'black'}
line_styles = {'Target': '-', 'Flag': '-'}

# Data gathering
target_dataset = pd.read_csv(target_path)
target_list = eval(target_dataset.loc[target_dataset.Step == 0, 'Flag'].values[0])
target_array = np.array(target_list).reshape(nb_rows, nb_cols, first_elem_size)

flag_dataset = pd.read_csv(flag_path)
flag_list = eval(flag_dataset.loc[flag_dataset.Step == 49, 'Flag'].values[0])
flag_array = np.array(flag_list).reshape(nb_rows, nb_cols, first_elem_size)

# Plot x values of a specific row in the grid
for index in row_indexes_to_inspect:
    plt.figure(figsize=(8, 4), dpi=300)

    # Target
    target_x_values = target_array[index, :, 0] # 'index' selects a specific row, ':' selects all columns in that row, '0' selects the first value (x) in each [x, y] pair
    plt.plot(target_x_values, label='Target', color=colors['Target'], linestyle=line_styles['Target'], linewidth=1)

    # Flag
    flag_x_values = flag_array[index, :, 0]
    plt.plot(flag_x_values, label='Flag step 49', color=colors['Flag'], linestyle=line_styles['Flag'], linewidth=1)


    plt.title(f"Row {index} – component x", fontsize=14)
    plt.xlabel("grid column index", fontsize=12)
    plt.ylabel("x value", fontsize=12)
    plt.ylim(-0.1, 1.1) # 0 and 1 are respectively min and max values of flag distance (fitness)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/profileXY_row{index}_componentX.png")
    plt.clf()
    plt.close()


# Plot y values of a specific column in the grid
for index in cols_indexes_to_inspect:
    plt.figure(figsize=(8, 4), dpi=300)

    # Target
    target_y_values = target_array[:, index, 1]  # ':' selects all rows, 'index' selects a specific grid column, '1' selects the second value (y) in each [x, y] pair
    plt.plot(target_y_values, label='Target', color=colors['Target'], linestyle=line_styles['Target'], linewidth=1)

    # Flag
    flag_y_values = flag_array[:, index, 1]
    plt.plot(flag_y_values, label='Flag step 49', color=colors['Flag'], linestyle=line_styles['Flag'], linewidth=1)

    plt.title(f"Column {index} – component y", fontsize=14)
    plt.xlabel("grid row index", fontsize=12)
    plt.ylabel("y value", fontsize=12)
    plt.ylim(-0.1, 1.1)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(f"{output_dir}/profileXY_col{index}_componentY.png")
    plt.clf()
    plt.close()
