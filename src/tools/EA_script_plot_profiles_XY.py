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
target_path = "/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_NxN_280kEvals/sliding_puzzle_coordinates_2025-10-27_15-05-06_centered-half-discs_16x16/learning/data_all_runs/data_env0_flag_target.csv"
flagSingleEnv_path = "/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_NxN_280kEvals/sliding_puzzle_coordinates_2025-10-27_15-05-06_centered-half-discs_16x16/learning/run_011/data/data_env0_flag/data_env_flag_run_011_gen_03817_eval_0045811.csv"
flagMultiEnv_path = "/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_NxN_280kEvals/sliding_puzzle_multiEnvs_coordinates_2025-10-17_02-23-54_centered-half-discs_NxN/learning/run_000/data/data_env2_flag/data_env_flag_run_000_gen_02040_eval_0024481.csv"
nb_rows, nb_cols = 16, 16
first_elem_size = 2 # [x,y]
row_indexes_to_inspect = [0, 7, 15]
cols_indexes_to_inspect = [0, 7, 15]

# Build a new folder
today_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
output_dir = f"profilesXY_{today_str}"
os.makedirs(output_dir, exist_ok=True)

# Save a copy of the originals flags and their paths
shutil.copy(target_path, os.path.join(output_dir, os.path.basename(target_path)))
shutil.copy(flagSingleEnv_path, os.path.join(output_dir, os.path.basename(flagSingleEnv_path)))
shutil.copy(flagMultiEnv_path, os.path.join(output_dir, os.path.basename(flagMultiEnv_path)))

with open(os.path.join(output_dir, "used_paths.txt"), "w") as f:
    f.write("Original file paths used in this analysis:\n")
    f.write(f"Target path: {os.path.abspath(target_path)}\n")
    f.write(f"Flag singleEnv path:   {os.path.abspath(flagSingleEnv_path)}\n")
    f.write(f"Flag multiEnv path:   {os.path.abspath(flagMultiEnv_path)}\n")


# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
colors = {'target': 'tab:red', 'flagSingleEnv': 'black', 'flagMultiEnv': 'tab:blue'} # red ou crimson
line_styles = {'target': '-', 'flagSingleEnv': '-', 'flagMultiEnv': '-'}

# Data gathering
target_dataset = pd.read_csv(target_path)
target_list = eval(target_dataset.loc[target_dataset.Step == 0, 'Flag'].values[0])
target_array = np.array(target_list).reshape(nb_rows, nb_cols, first_elem_size)

flagSingleEnv_dataset = pd.read_csv(flagSingleEnv_path)
flagSingleEnv_list = eval(flagSingleEnv_dataset.loc[flagSingleEnv_dataset.Step == 49, 'Flag'].values[0])
flagSingleEnv_array = np.array(flagSingleEnv_list).reshape(nb_rows, nb_cols, first_elem_size)

flagMultiEnv_dataset = pd.read_csv(flagMultiEnv_path)
flagMultiEnv_list = eval(flagMultiEnv_dataset.loc[flagMultiEnv_dataset.Step == 49, 'Flag'].values[0])
flagMultiEnv_array = np.array(flagMultiEnv_list).reshape(nb_rows, nb_cols, first_elem_size)

# Plot x values of a specific row in the grid
for index in row_indexes_to_inspect:
    fig = plt.figure(figsize=(12, 11), dpi=300)
    fig.subplots_adjust(left=0.25, bottom=0.25, right=1.1, top=0.87)

    # Target
    target_x_values = target_array[index, :, 0] # 'index' selects a specific row, ':' selects all columns in that row, '0' selects the first value (x) in each [x, y] pair
    plt.plot(target_x_values, label='Target', color=colors['target'], linestyle=line_styles['target'], linewidth=3)

    # Flag singleEnv
    flagSingleEnv_x_values = flagSingleEnv_array[index, :, 0]
    plt.plot(flagSingleEnv_x_values, label='Best flag step 49\nsingleEnv', color=colors['flagSingleEnv'], linestyle=line_styles['flagSingleEnv'], linewidth=3)

    # Flag multiEnv
    flagMultiEnv_x_values = flagMultiEnv_array[index, :, 0]
    plt.plot(flagMultiEnv_x_values, label='Best flag step 49\nmultiEnvs', color=colors['flagMultiEnv'], linestyle=line_styles['flagMultiEnv'], linewidth=3)


    # plt.suptitle(f"Profile XY - Row {index}, component X", fontsize=18)
    # plt.title("Pattern: coordinates 16x16; 20 runs; 280,000 evaluations", fontsize=18)
    plt.title(f"Row {index}, component X", fontsize=65, pad=20)
    plt.xlabel("Grid column index", fontsize=70)
    plt.ylabel("X value", fontsize=70)
    plt.xticks([0, 5, 10, 15], fontsize=60)
    for label in plt.gca().get_xticklabels()[:1]:
        label.set_horizontalalignment('left')
    plt.yticks([0.0, 0.5, 1.0], fontsize=60)
    plt.xlim(0, 15)
    plt.ylim(-0.01, 1.01) # 0 and 1 are respectively min and max values of flag distance (fitness)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=55, ncol=1, frameon=False)
    # plt.legend(frameon=True, fontsize=50)
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.tight_layout()
    
    plt.savefig(f"{output_dir}/profileXY_row{index}_componentX.png", bbox_inches='tight')
    plt.clf()
    plt.close()


# Plot y values of a specific column in the grid
for index in cols_indexes_to_inspect:
    fig = plt.figure(figsize=(12, 11), dpi=300)
    fig.subplots_adjust(left=0.25, bottom=0.25, right=1.1, top=0.87)

    # Target
    target_y_values = target_array[:, index, 1]  # ':' selects all rows, 'index' selects a specific grid column, '1' selects the second value (y) in each [x, y] pair
    plt.plot(target_y_values, label='Target', color=colors['target'], linestyle=line_styles['target'], linewidth=3)

    # Flag singleEnv
    flagSingleEnv_y_values = flagSingleEnv_array[:, index, 1]
    plt.plot(flagSingleEnv_y_values, label='Best flag step 49\nsingleEnv', color=colors['flagSingleEnv'], linestyle=line_styles['flagSingleEnv'], linewidth=3)

    # Flag multiEnv
    flagMultiEnv_y_values = flagMultiEnv_array[:, index, 1]
    plt.plot(flagMultiEnv_y_values, label='Best flag step 49\nmultiEnvs', color=colors['flagMultiEnv'], linestyle=line_styles['flagMultiEnv'], linewidth=3)

    # plt.suptitle(f"Profile XY - Column {index}, component Y", fontsize=18)
    # plt.title("Pattern: coordinates 16x16; 20 runs; 280,000 evaluations", fontsize=18)
    plt.title(f"Column {index}, component Y", fontsize=65, pad=20)
    plt.xlabel("Grid row index", fontsize=70)
    plt.ylabel("Y value", fontsize=70)
    plt.xticks([0, 5, 10, 15], fontsize=60)
    for label in plt.gca().get_xticklabels()[:1]:
        label.set_horizontalalignment('left')
    plt.yticks([0.0, 0.5, 1.0], fontsize=60)
    plt.xlim(0, 15)
    plt.ylim(-0.01, 1.01)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=55, ncol=1, frameon=False)
    # plt.legend(frameon=True, fontsize=50)
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.tight_layout()

    plt.savefig(f"{output_dir}/profileXY_col{index}_componentY.png", bbox_inches='tight')
    plt.clf()
    plt.close()
