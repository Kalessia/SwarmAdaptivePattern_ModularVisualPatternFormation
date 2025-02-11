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
    # y_la!*bels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0] # fluidity (p_move)
    # x_labels = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0] # density ( (grid_size - nb_deletions) / grid_size)
    x_labels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # density ( (grid_size - nb_deletions) / grid_size)
    phi = 0.0
    target = [(0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0), (0.5, 0.0), (0.6, 0.0), (0.7, 0.0), (0.8, 0.0), (0.9, 0.0), (1.0, 0.0)]
    # data = pd.DataFrame(np.nan, index=y_labels, columns=x_labels)

    experiences_data = []
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


            if (density, p_move) in target:
                dataset = pd.read_csv(f"{experiences_dir_to_plot}/{dir}/learning/data_all_runs/data_evo_all_runs_best_ind_per_run_per_phase.csv")
                filtered_data = dataset[dataset['Learning_phase'] == phase]
                best_fit = filtered_data['Fitness'].min()
                min_fitness_rows = filtered_data[filtered_data['Fitness'] == best_fit]     
                best_row = min_fitness_rows.loc[min_fitness_rows['Nb_eval'].idxmin()] # If there's a tie, select the row with the minimum number of evaluations
                best_run = best_row['Run']

                dataset_file_source = f"{experiences_dir_to_plot}/{dir}/swarm/run_000/best_ind_{best_run:03}/data/data_all_repetitions/setup_sliding_puzzle/data_setup_sliding_puzzle_stats_per_repetition.csv"
                df = pd.read_csv(dataset_file_source)
                deletions = learning_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_ticks'][1]

                # mean_fit_over_repetitions = df.loc[(df.Deletions==deletions) & (df.Fluidity==p_move), 'Flags_distance'].mean()
                fitnesses = df.loc[(df.Deletions==deletions) & (df.Fluidity==p_move), 'Flags_distance']
                
                # Ajout des données de cette expérience à la liste
                for fitness in fitnesses:
                    experiences_data.append({
                        'Experience': (density, p_move),
                        'Density': density,
                        'Fluidity': p_move,
                        'Fitness': fitness
                    })

                # print(density, deletions, fitnesses.tolist())
                # data.loc[p_move, density] = mean_fit_over_repetitions


    df_gne = pd.DataFrame(experiences_data)
    df_gne.head()

    sns.set_theme(style='dark')
    # heatmap_plot = sns.heatmap(data, annot=False, annot_kws={"size": 9}, fmt=".3f", cmap="Blues", cbar=True, linewidths=0.5, linecolor='white', vmin=0.0, vmax=0.4) # annot=True to show fit values on cells

    # # Add a crossed rectangle for each not evaluated case
    # for case in [(9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9)]: # cases density 1.0, fluidity > 0.0 are crossed
    #     crossed_box = plt.Rectangle(case, 1, 1, facecolor='none', edgecolor='grey', hatch='//')
    #     heatmap_plot.add_patch(crossed_box)

    plt.figure(figsize=(12, 7))
    sns.set_theme(style='darkgrid')
    _, ax = plt.subplots()
    sns.boxplot(x='Experience',
                y='Fitness',
                data=df_gne,
                color='skyblue',
                medianprops={'color': 'red', 'linewidth': 0.5},
                width=0.7,
                ax=ax) # individuals of differents runs with same generation have also same nb_evaluation, so dataset is grouped by "nb_eval" in the "Evaluation" column

    exp_setup_and_dims = dir.split('_')
    plt.xlabel("Distances", fontsize=14)
    plt.ylabel("Fluidity of the system", fontsize=14)
    plt.title(f"Averaged distances of the best-ever controller\nin sliding puzzle {exp_setup_and_dims[-2]} {exp_setup_and_dims[-1]}, 11 runs", fontsize=16)

    df_gne.to_csv(f"simulationAnalysis/plot_sliding_puzzle_incremental_{exp_setup_and_dims[-2]}_{exp_setup_and_dims[-1]}_learning_boxplots.csv") # write data
    plt.savefig(f"simulationAnalysis/plot_sliding_puzzle_incremental_{exp_setup_and_dims[-2]}_{exp_setup_and_dims[-1]}_learning_boxplots.png")

    # plt.clf()
    # plt.close()



    #     dataset = pd.read_csv(dataset_path)
    # dataset = dataset.loc[dataset.Run==run]

    # nb_gens_to_plot = 20 # number of boxes to display
    # generations = dataset['Generation'].unique()
    # pop_size = int(nb_evals/len(generations))
    # gens_to_plot = list(range(generations[0], generations[-1], max(1, int(len(generations)/nb_gens_to_plot))))

    # filtered_dataset = dataset[dataset['Generation'].isin(gens_to_plot)]
    # filtered_dataset = filtered_dataset.copy() # create a copy to avoid SettingWithCopyWarning
    # filtered_dataset['Max_evaluation_per_gen'] = pd.to_numeric(filtered_dataset['Generation'].apply(lambda gen: pop_size * (gen + 1))) # nb_eval is here the last evaluation of a given generation, even if the best ind per run per gen was registered in different nb_eval within the same generation
    # evaluations = filtered_dataset['Max_evaluation_per_gen'].unique()


    # Plot phase1-phase2 delimeter and max fitness limit
    # x_end = len(evaluations)
    # y = 1
    # if switch_eval is not None:
    #     index = bisect.bisect_left(evaluations, switch_eval) # bisect_left returns the 1st index where switch_eval would be inserted to maintain the list order
    #     plt.axvline(x=index-0.5, color='r', linestyle='--')
    #     y = 1 - (nb_deletions[1]/grid_size)
    #     plt.plot([index-0.5, x_end], [y, y], linestyle=':', linewidth=2.0, color='tab:blue') # max fitness limit phase2
    #     x_end = index-0.5
    #     y = 1 - (nb_deletions[0]/grid_size) 
    
    # plt.plot([0, x_end], [y, y], linestyle=':', linewidth=2.0, color='tab:blue', label=f"worst flags dist") # max fitness limit phase1

    # plt.title("Flags distance over generations\nall individuals generated", fontsize=14)
    # plt.xlabel("Evaluations", fontsize=12)
    # plt.ylabel("Flags distance", fontsize=12)

    plt.ylim(-0.1, 1.1) # 0 and 1 are respectively min and max values of flag distance (fitness)
    ax.set_xticks(range(0, len(evaluations))) # important: boxplot boxes locations are incremental number from 0 to N=len(evaluations)
    # pace = int(len(evaluations) / 7) # number of labels to display, for lisibility
    # evaluations_labels = [str(evaluations[i]) if i%pace == 0 else "" for i in range(0, len(evaluations))]
    ax.set_xticklabels(evaluations_labels)

    plt.savefig(save_filename)
    plt.clf()
    plt.close()



experiences_dir_to_plot = "simulationAnalysis"
plot_sliding_puzzle_incremental_learning_density_fluidity(experiences_dir_to_plot)