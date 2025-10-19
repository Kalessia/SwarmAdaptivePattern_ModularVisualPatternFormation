import os
import json
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend instead of QtAgg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_boxplots_allRunsFitnesses_per_exp_NOcoordinates(experiences_dir_to_plot):

    experiences_dirs = [e for e in os.listdir(experiences_dir_to_plot) if "coordinates" not in e]

    experiences_data = []
    for dir in experiences_dirs:
        dataset_file_source = f"{experiences_dir_to_plot}/{dir}/learning/data_all_runs/data_evo_all_runs_best_ind_per_run.csv"    
        df = pd.read_csv(dataset_file_source)
        df_run_fitness = df[['Run', 'Fitness']]

        for _, row in df_run_fitness.iterrows():
            experiences_data.append({
                'Experiment': (dir.split('_'))[-1],
                'Run': row['Run'],
                'Fitness': row['Fitness']
            })


    df_gne = pd.DataFrame(experiences_data)
    df_gne.head()

    sns.set_theme(style='dark')
    plt.figure(figsize=(14, 7), dpi=300)
    sns.set_theme(style='darkgrid')
    _, ax = plt.subplots(figsize=(12, 7), dpi=300)
    sns.boxplot(x='Experiment',
                y='Fitness',
                data=df_gne,
                color='skyblue',
                medianprops={'color': 'red', 'linewidth': 0.5},
                width=0.7,
                ax=ax) # individuals of differents runs with same generation have also same nb_evaluation, so dataset is grouped by "nb_eval" in the "Evaluation" column


    plt.xlabel("Distances", fontsize=14)
    plt.ylabel("Experiment", fontsize=14)
    exp_setup_and_dims = dir.split('_')
    plt.title(f"All runs fitnesses per experiment\n{exp_setup_and_dims[-3]}, {exp_setup_and_dims[-2]}", fontsize=12)

    df_gne.to_csv(f"../data_plots/learning_boxplots_sliding_puzzle_NOcoordinates.csv") # write data
    plt.savefig(f"../data_plots/learning_boxplots_sliding_puzzle_NOcoordinates.png")
    plt.ylim(-0.1, 1.1) # 0 and 1 are respectively min and max values of flag distance (fitness)

    plt.clf()
    plt.close()




def plot_boxplots_allRunsFitnesses_per_exp_coordinatesONLY(experiences_dir_to_plot):

    experiences_dirs = os.listdir(experiences_dir_to_plot)

    experiences_data = []
    for dir in experiences_dirs:
        dataset_file_source = f"{experiences_dir_to_plot}/{dir}/data_all_runs/data_evo_all_runs_best_ind_per_run.csv"    
        df = pd.read_csv(dataset_file_source)
        df_run_fitness = df[['Run', 'Fitness']]

        for _, row in df_run_fitness.iterrows():
            exp_setup_and_dims = dir.split('_')
            experiences_data.append({
                'Experiment': f"{exp_setup_and_dims[-1]}",
                'Run': row['Run'],
                'Fitness': row['Fitness']
            })


    df_gne = pd.DataFrame(experiences_data)
    df_gne.head()

    sns.set_theme(style='dark')
    plt.figure(figsize=(12, 7), dpi=300)
    sns.set_theme(style='darkgrid')
    _, ax = plt.subplots(figsize=(12, 7), dpi=300)
    sns.boxplot(x='Experiment',
                y='Fitness',
                data=df_gne,
                color='skyblue',
                medianprops={'color': 'red', 'linewidth': 0.5},
                width=0.7,
                ax=ax) # individuals of differents runs with same generation have also same nb_evaluation, so dataset is grouped by "nb_eval" in the "Evaluation" column


    plt.xlabel("Distances", fontsize=14)
    plt.ylabel("Experiment", fontsize=14)
    exp_setup_and_dims = experiences_dir_to_plot.split('/')
    plt.title(f"All runs fitnesses per experiment\n{exp_setup_and_dims[-1]}", fontsize=11)

    df_gne.to_csv(f"../data_plots/learning_boxplots_sliding_puzzle_coordinatesONLY_{exp_setup_and_dims[-1]}.csv") # write data
    plt.savefig(f"../data_plots/learning_boxplots_sliding_puzzle_coordinatesONLY_{exp_setup_and_dims[-1]}.png")
    plt.ylim(-0.1, 1.1) # 0 and 1 are respectively min and max values of flag distance (fitness)
    ax.tick_params(axis='x', labelsize=4)

    plt.clf()
    plt.close()




# experiences_dir_to_plot = "../data_plots/simulationAnalysis_20250527_flagAutomata_results"
# plot_boxplots_allRunsFitnesses_per_exp_NOcoordinates(experiences_dir_to_plot)

coordinates_experiences_dir_to_plot = "../data_plots/simulationAnalysis/sliding_puzzle_coordinates_2025-05-22_14-07-36_centered-half-discs_16x16"
plot_boxplots_allRunsFitnesses_per_exp_coordinatesONLY(coordinates_experiences_dir_to_plot)