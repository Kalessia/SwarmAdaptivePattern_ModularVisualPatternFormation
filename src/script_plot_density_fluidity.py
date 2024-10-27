import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from learning_initializations import save_data_to_csv





# def plot_density_fluidity(data_flag_dir, setup_name, run, grid_size, switch_step, analysis_dir_plots):
def plot_sliding_puzzle_incremental_learning_density_fluidity(experiences_dirs_to_plot):

    y_labels = [0.20, 0.15, 0.10, 0.05, 0.01, 0.0] # fluidity (p_move)
    x_labels = [0.80, 0.85, 0.90, 0.95, 1.0] # density ( (grid_size - nb_deletions) / grid_size)
    data = pd.DataFrame(np.nan, index=y_labels, columns=x_labels)

    for experience_dir in experiences_dirs_to_plot:

        # Get parameters from config files
        with open(experience_dir+"/learning/learning_params.json", "r") as f:
            learning_params = json.load(f)

            if learning_params['evolutionary_settings']['env_name'] != "sliding_puzzle_incremental":
                print(f"plot_sliding_puzzle_incremental_learning_density_fluidity stopped because {experience_dir} has env_name is {learning_params['evolutionary_settings']['env_name']} and not 'sliding_puzzle_incremental'")
                exit()

            p_move = learning_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_proba_move'] # fluidity ticks
            density = round( (learning_params['grid']['grid_size'] - learning_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_ticks'][1] ) / learning_params['grid']['grid_size'], 2) # density ticks

            dataset = pd.read_csv(experience_dir+"/learning/data_all_runs/data_evo_all_runs_best_ind_per_run_per_phase.csv")
            data.loc[p_move, density] = dataset.loc[dataset.Learning_phase==2, 'Fitness'].min()


    sns.set_theme(style='dark')
    sns.heatmap(data, annot=True, cmap="viridis", cbar=True, linewidths=0.5, linecolor='white')

    exp_setup_and_dims = experience_dir.split('_')
    plt.xlabel("Density of the system", fontsize=12)
    plt.ylabel("Fluidity of the system", fontsize=12)
    plt.title(f"Incremental learning flag distances\nin sliding puzzle {exp_setup_and_dims[-2]} {exp_setup_and_dims[-1]} experiences overview", fontsize=12)

    data.to_csv(f"simulationAnalysis/plot_sliding_puzzle_incremental_{exp_setup_and_dims[-2]}_{exp_setup_and_dims[-1]}_learning_density_fluidity.csv") # write data
    plt.savefig(f"simulationAnalysis/plot_sliding_puzzle_incremental_{exp_setup_and_dims[-2]}_{exp_setup_and_dims[-1]}_learning_density_fluidity.png")

    plt.clf()
    plt.close()





experiences_dirs_to_plot = [
    "simulationAnalysis/sliding_puzzle_incremental_2024-10-25_03-08-11_two_bands_16x16"
]

plot_sliding_puzzle_incremental_learning_density_fluidity(experiences_dirs_to_plot)