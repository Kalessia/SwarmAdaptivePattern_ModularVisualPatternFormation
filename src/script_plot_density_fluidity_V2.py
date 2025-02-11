import os
import json
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend instead of QtAgg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


#def plot_sliding_puzzle_incremental_generalization_density_fluidity(experiences_dir_to_plot):



def plot_sliding_puzzle_incremental_learning_density_fluidity(experiences_dir_to_plot):

    experiences_dirs = os.listdir(experiences_dir_to_plot)

    # max_fitnesses = 0.0
    y_labels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0] # fluidity (p_move)
    # x_labels = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0] # density ( (grid_size - nb_deletions) / grid_size)
    x_labels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # density ( (grid_size - nb_deletions) / grid_size)
    data = pd.DataFrame(np.nan, index=y_labels, columns=x_labels)

    for dir in experiences_dirs:

        # Get parameters from config files
        with open(f"{experiences_dir_to_plot}/{dir}/learning/learning_params.json", "r") as f:
            learning_params = json.load(f)

            if learning_params['evolutionary_settings']['env_name'] != "sliding_puzzle_incremental":
                print(f"plot_sliding_puzzle_incremental_learning_density_fluidity stopped because {dir} env_name is {learning_params['evolutionary_settings']['env_name']} and not 'sliding_puzzle_incremental'")
                exit()

            p_move = learning_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_proba_move'] # fluidity ticks
            
            if learning_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_switch_eval'] > 13000:
                density = round( (learning_params['grid']['grid_size'] - learning_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_ticks'][0] ) / learning_params['grid']['grid_size'], 2) # density ticks
                phase = 1
            else:
                density = round( (learning_params['grid']['grid_size'] - learning_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_ticks'][1] ) / learning_params['grid']['grid_size'], 2) # density ticks
                phase = 2 # normally, we look for the best individual at the end of learning

            if density == 1.0:
                p_move = 0 # if density is max, agents can't move

            dataset = pd.read_csv(f"{experiences_dir_to_plot}/{dir}/learning/data_all_runs/data_evo_all_runs_best_ind_per_run_per_phase.csv")
            filtered_data = dataset[dataset['Learning_phase'] == phase]
            best_fit = filtered_data['Fitness'].min()
            min_fitness_rows = filtered_data[filtered_data['Fitness'] == best_fit]            
            best_row = min_fitness_rows.loc[min_fitness_rows['Nb_eval'].idxmin()] # If there's a tie, select the row with the minimum number of evaluations
            best_run = best_row['Run']

            dataset_file_source = f"{experiences_dir_to_plot}/{dir}/swarm/run_000/best_ind_{best_run:03}/data/data_all_repetitions/setup_sliding_puzzle/data_setup_sliding_puzzle_stats_per_repetition.csv"
            df = pd.read_csv(dataset_file_source)
            deletions = learning_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_ticks'][1]
            mean_fit_over_repetitions = df.loc[(df.Deletions==deletions) & (df.Fluidity==p_move), 'Flags_distance'].mean()
            # print(density, deletions, mean_fit_)
            data.loc[p_move, density] = mean_fit_over_repetitions



    sns.set_theme(style='dark')
    cmap = sns.color_palette("Reds", as_cmap=True)
    cmap.set_bad(color='lightgrey')
    heatmap_plot = sns.heatmap(data, annot=False, annot_kws={"size": 9}, fmt=".3f", cmap=cmap, cbar=True, linewidths=0.5, linecolor='white', vmin=0.0, vmax=0.5) # annot=True to show fit values on cells
    

    # Add a crossed rectangle for each not evaluated case
    for case in [(9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9)]: # cases density 1.0, fluidity > 0.0 are crossed
        crossed_box = plt.Rectangle(case, 1, 1, facecolor='none', edgecolor='grey', hatch='//')
        heatmap_plot.add_patch(crossed_box)

    exp_setup_and_dims = dir.split('_')
    plt.xlabel(f"Density ($\\rho$)\n _", fontsize=16)
    plt.ylabel(f"Fluidity ($\\Phi$)", fontsize=16)
    plt.title(f"Performance of the best-ever controllers\ntwo-bands {exp_setup_and_dims[-1]}, 11 runs per condition", fontsize=18)

    data.to_csv(f"simulationAnalysis/plot_sliding_puzzle_incremental_{exp_setup_and_dims[-2]}_{exp_setup_and_dims[-1]}_learning_density_fluidity.csv") # write data
    plt.savefig(f"simulationAnalysis/plot_sliding_puzzle_incremental_{exp_setup_and_dims[-2]}_{exp_setup_and_dims[-1]}_learning_density_fluidity.png")

    plt.clf()
    plt.close()



experiences_dir_to_plot = "simulationAnalysis"
plot_sliding_puzzle_incremental_learning_density_fluidity(experiences_dir_to_plot)