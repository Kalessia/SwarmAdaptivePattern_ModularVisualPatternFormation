import os
import json
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend instead of QtAgg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np




def add_experiment_to_experiences_data(experiences_data, experiences_dir_to_plot, dir, flag, ctrl):
    dataset_file_source = f"{experiences_dir_to_plot}/{dir}/data_all_runs/data_evo_all_runs_best_ind_per_run.csv"
    df = pd.read_csv(dataset_file_source)
    df_run_fitness = df[['Run', 'Fitness']]

    for _, row in df_run_fitness.iterrows():
        experiences_data.append({
            'Experiment': f"{flag}_{ctrl}",
            'Run': row['Run'],
            'Fitness': row['Fitness']
        })


def plot_boxplots_allRunsFitnesses_per_exp(experiences_dir_to_plot, coordinates_experiences_dir_to_plot):

    controllers_bn = ["modelGECCO_4-[3]-2", "modelGECCO_4-[5,5]-2", "modelD_8-[3]-3", "modelD_8-[5,5]-3"]
    controllers_rgb = ["modelGECCO_4-[3]-4", "modelGECCO_4-[5,5]-4", "modelD_8-[3]-5", "modelD_8-[5,5]-5"]

    flag_patterns_bn = ['centered-half-discs', 'bn-SU', 'bn-smile']
    flag_patterns_rgb = ['rgb-italian-flag', 'rgb-french-cockade', 'rgb-rainbow-full']

    experiences_dirs = [e for e in os.listdir(experiences_dir_to_plot) if not any(x in e for x in ["multiEnvs", coordinates_experiences_dir_to_plot])]
    experiences_data = []

    for flag_bn in flag_patterns_bn:
        for ctrl_bn in controllers_bn:
            dir = [e for e in experiences_dirs if all(x in e for x in [flag_bn, ctrl_bn])]

            if len(dir)==1:
                add_experiment_to_experiences_data(
                        experiences_data=experiences_data,
                        experiences_dir_to_plot=experiences_dir_to_plot,
                        dir=f"{dir[0]}/learning", 
                        flag=f"{flag_bn}",
                        ctrl=f"{ctrl_bn}")
            else:
                print(f"Warning: experience {flag_bn}, {ctrl_bn} not found or multiple. Dir: {dir}")
            

    for flag_rgb in flag_patterns_rgb:
        for ctrl_rgb in controllers_rgb:
            dir = [e for e in experiences_dirs if all(x in e for x in [flag_rgb, ctrl_rgb])]

            if len(dir)==1:
                add_experiment_to_experiences_data(
                        experiences_data=experiences_data,
                        experiences_dir_to_plot=experiences_dir_to_plot,
                        dir=f"{dir[0]}/learning", 
                        flag=f"{flag_rgb}",
                        ctrl=f"{ctrl_rgb}")
            else:
                print(f"Warning: experience {flag_rgb}, {ctrl_rgb} not found or multiple. Dir: {dir}")


    controllers_bn = ["modelA_4-[]-3_2-[3]-1", "modelA_4-[]-3_2-[5,5]-1", "modelC_4-[]-3_6-[3]-1", "modelC_4-[]-3_6-[5,5]-1"]
    controllers_rgb = ["modelA_4-[]-3_2-[3]-3", "modelA_4-[]-3_2-[5,5]-3", "modelC_4-[]-3_6-[3]-3", "modelC_4-[]-3_6-[5,5]-3"]

    experiences_dirs = [e for e in os.listdir(coordinates_experiences_dir_to_plot) if not any(x in e for x in ["multiEnvs"])]

    for flag_bn in flag_patterns_bn:
        for ctrl_bn in controllers_bn:
            dir = [e for e in experiences_dirs if all(x in e for x in [flag_bn, ctrl_bn])]

            if len(dir)==1:            
                add_experiment_to_experiences_data(
                        experiences_data=experiences_data,
                        experiences_dir_to_plot=coordinates_experiences_dir_to_plot,
                        dir=f"{dir[0]}", 
                        flag=f"{flag_bn}",
                        ctrl=f"{ctrl_bn}")
            else:
                print(f"Warning: experience {flag_bn}, {ctrl_bn} not found or multiple. Dir: {dir}")
            

    for flag_rgb in flag_patterns_rgb:
        for ctrl_rgb in controllers_rgb:
            dir = [e for e in experiences_dirs if all(x in e for x in [flag_rgb, ctrl_rgb])]

            if len(dir)==1:    
                add_experiment_to_experiences_data(
                        experiences_data=experiences_data,
                        experiences_dir_to_plot=coordinates_experiences_dir_to_plot,
                        dir=f"{dir[0]}", 
                        flag=f"{flag_rgb}",
                        ctrl=f"{ctrl_rgb}")
            else:
                print(f"Warning: experience {flag_rgb}, {ctrl_rgb} not found or multiple. Dir: {dir}")


    df_new = pd.DataFrame(experiences_data)
    df_new['Experiment_label'] = df_new['Experiment_label'] = df_new['Experiment'].str.split(pat='_', n=1).str[1].str.replace('_', '\n')  # Example of label: modelC_4-[]-3\n6-[5,5]-3

    palette = sns.color_palette("Set2", n_colors=8)
    # palette = ["skyblue", "lightgreen", "lightcoral", "orange", "plum", "khaki", "salmon", "turquoise"]

    for flag in flag_patterns_bn + flag_patterns_rgb:
        df_flag = df_new[df_new['Experiment'].str.split('_').str[0] == flag]
        sns.set_theme(style='darkgrid')
        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
        fig.subplots_adjust(bottom=0.15)
        sns.boxplot(x='Experiment_label',
                    y='Fitness',
                    data=df_flag,
                    color='mediumaquamarine',
                    # hue='Experiment_label',  # set 'hue' in order to use the palette
                    # palette=palette,  # each box has a different color
                    medianprops={'color': 'red', 'linewidth': 0.8},
                    width=0.7,
                    ax=ax) # individuals of differents runs with same generation have also same nb_evaluation, so dataset is grouped by "nb_eval" in the "Evaluation" column

        plt.xlabel("Experiment", fontsize=18)
        plt.ylabel("Distance", fontsize=18, labelpad=20)
        plt.suptitle(f"All runs fitnesses per experiment", fontsize=18)
        plt.title(f"Pattern: {flag} 16x16; 20 runs; 28,000 evaluations", fontsize=18)
        # ax.set_autoscaley_on(False)
        ax.set_ylim(0.00, 0.25) # 0 and 1 are respectively min and max values of flag distance (fitness)

        df_flag.to_csv(f"/home/loi/flagAutomata/src/tools/learning_boxplots_{flag}.csv") # write data
        plt.savefig(f"/home/loi/flagAutomata/src/tools/learning_boxplots_{flag}.png")

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


    df_new = pd.DataFrame(experiences_data)
    df_new.head()

    sns.set_theme(style='dark')
    plt.figure(figsize=(12, 7), dpi=300)
    sns.set_theme(style='darkgrid')
    _, ax = plt.subplots(figsize=(12, 7), dpi=300)
    sns.boxplot(x='Experiment',
                y='Fitness',
                data=df_new,
                color='skyblue',
                medianprops={'color': 'red', 'linewidth': 0.5},
                width=0.7,
                ax=ax) # individuals of differents runs with same generation have also same nb_evaluation, so dataset is grouped by "nb_eval" in the "Evaluation" column


    plt.xlabel("Distances", fontsize=14)
    plt.ylabel("Experiment", fontsize=14)
    exp_setup_and_dims = experiences_dir_to_plot.split('/')
    plt.title(f"All runs fitnesses per experiment\n{exp_setup_and_dims[-1]}", fontsize=11)

    df_new.to_csv(f"../data_plots/learning_boxplots_sliding_puzzle_coordinatesONLY_{exp_setup_and_dims[-1]}.csv") # write data
    plt.savefig(f"../data_plots/learning_boxplots_sliding_puzzle_coordinatesONLY_{exp_setup_and_dims[-1]}.png")
    plt.ylim(0, 1.1) # 0 and 1 are respectively min and max values of flag distance (fitness)
    ax.tick_params(axis='x', labelsize=4)

    plt.clf()
    plt.close()





experiences_dir_to_plot = "/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_models-A-C-D-GECCO"
coordinates_experiences_dir_to_plot = "/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_models-A-C-D-GECCO/sliding_puzzle_coordinates_2025-05-22_14-07-36_centered-half-discs_16x16"
plot_boxplots_allRunsFitnesses_per_exp(experiences_dir_to_plot, coordinates_experiences_dir_to_plot)