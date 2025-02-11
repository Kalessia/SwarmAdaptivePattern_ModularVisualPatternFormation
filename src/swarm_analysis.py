import os
import time
import shutil
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend instead of QtAgg
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import pandas as pd
import numpy as np
import re

import argparse

import json

from swarm_initializations import *
from environments import swarmGrid


###########################################################################
# Analysis folders creation
###########################################################################

def init_all_runs_analysis(learning_analysis_dir_root, params):

    # Create the 'analysis_dir' folder
    params['analysis_dir'] = {}
    params['analysis_dir']['root'] = learning_analysis_dir_root.replace("/learning", "/swarm")

    if os.path.exists(params['analysis_dir']['root']):
        shutil.rmtree(params['analysis_dir']['root'])
    
    os.makedirs(params['analysis_dir']['root']+"/data_all_runs", exist_ok=True)
    os.makedirs(params['analysis_dir']['root']+"/plots_all_runs", exist_ok=True)

    if os.path.exists(learning_analysis_dir_root+"/plots_all_runs/plot_env_flag_target.png"): # forse meglio copiare la flag e ricrearla?
        shutil.copyfile(learning_analysis_dir_root+"/plots_all_runs/plot_env_flag_target.png", params['analysis_dir']['root']+"/plots_all_runs/plot_env_flag_target.png")
                    
    save_data_to_csv(params['analysis_dir']['root']+"/data_all_runs/data_all_runs_time.csv", [], header = ["Run", "Time(s)", "Time(min)", "Time(h)"])
    
    return params

#---------------------------------------------------

def init_one_run_analysis(run, best_ind, best_ind_run, params):

    # Create the data and plot directories tree from the 'analysis_dir' folder
    params['analysis_dir']['data'] = params['analysis_dir']['root']+ f"/run_{run:03}/best_ind_{best_ind_run:03}/data"
    params['analysis_dir']['plots'] = params['analysis_dir']['root']+ f"/run_{run:03}/best_ind_{best_ind_run:03}/plots"
    os.makedirs(params['analysis_dir']['data'], exist_ok=True)
    os.makedirs(params['analysis_dir']['plots'], exist_ok=True)

    # Copy of the original best individual flag (taken from learning data) in data/original_flag_copied_from_learning
    os.makedirs(params['analysis_dir']['data']+"/original_flag_copied_from_learning", exist_ok=True)
    dataset = pd.read_csv(params['analysis_dir']['root'].replace("/swarm", "/learning") + "/data_all_runs/data_evo_all_runs_best_ind_per_run.csv")
    gen = dataset.loc[(dataset.Run==best_ind_run),['Generation']].values.tolist()[0][0]
    nb_eval = dataset.loc[(dataset.Run==best_ind_run),['Nb_eval']].values.tolist()[0][0]
    source_path = params['analysis_dir']['root'].replace("/swarm", "/learning") + f"/run_{best_ind_run:03}/data/data_env_flag/data_env_flag_run_{best_ind_run:03}_gen_{gen:05}_eval_{nb_eval:07}.csv"
    shutil.copyfile(source_path, params['analysis_dir']['data']+ f"/original_flag_copied_from_learning/data_original_flag_copied_from_learning_flag_n_000.csv")

    file_path = params['analysis_dir']['root']+ f"/run_{run:03}/best_ind_{best_ind_run:03}/flag_individual.txt"
    if not os.path.exists(file_path):
        with open (file_path, 'w') as f:
            f.write(str(best_ind))

    return params


# ###########################################################################
# Plot data functions
# ###########################################################################
                
def plot_single_run_single_ind_data(run, best_ind_run, params):

    # if best_ind_run != 4:
    #     return

    time_run_ind = time.time()
    print(f"swarm_analysis plots run n.{run}, best_ind_{best_ind_run} - Starting")

    params['analysis_dir']['data'] = params['analysis_dir']['root']+ f"/run_{run:03}/best_ind_{best_ind_run:03}/data"
    params['analysis_dir']['plots'] = params['analysis_dir']['root']+ f"/run_{run:03}/best_ind_{best_ind_run:03}/plots"

    os.makedirs(params['analysis_dir']['plots']+"/original_flag_copied_from_learning", exist_ok=True)
    setups = params['setups'] + ["original_flag_copied_from_learning"]

    # setups = [f"setup_sliding_puzzle_phase1_VS_phase2_control_0_phase1",
    #           f"setup_sliding_puzzle_phase1_VS_phase2_control_0_phase2",
    #           f"setup_sliding_puzzle_phase1_VS_phase2_control_0_phase3",
    #           f"setup_sliding_puzzle_phase1_VS_phase2_control_25_phase1",
    #           f"setup_sliding_puzzle_phase1_VS_phase2_control_25_phase2",
    #           f"setup_sliding_puzzle_phase1_VS_phase2_control_25_phase3"]

    for setup_name in setups:

        for n in range(params['nb_repetitions']):
            data_flag_file = params['analysis_dir']['data']+ f"/{setup_name}/data_{setup_name}_flag_n_{n:03}.csv"
            dataset = pd.read_csv(data_flag_file)
            steps = [0, params['switch_step']-1, params['switch_step'], params['time_steps']-1]
            switch_step = params['switch_step']
            
            if setup_name == "original_flag_copied_from_learning" or setup_name.startswith("setup_sliding_puzzle"):
                # steps = dataset['Step'].unique()
                steps = [0, params['time_steps']-5, params['time_steps']-4, params['time_steps']-3, params['time_steps']-2, params['time_steps']-1]

            for step in steps:
                flag_list = dataset.loc[(dataset.Step==step),['Flag']].values.tolist()[0][0]
                flag_list = str(flag_list).replace('[', '').replace(']', '').strip()
                flag_list = np.asarray(flag_list.split(','), dtype=np.float32)
                fitness = dataset.loc[(dataset.Step==step),['Flags_distance']].values.tolist()[0][0]

                permutated_pos = []
                if setup_name.startswith("setup_permutation"):
                    permutated_pos = dataset.loc[(dataset.Step==step),['Permutated_agents_positions']].values.tolist()[0][0]
                    permutated_pos = eval(permutated_pos)

                deleted_pos = []
                if setup_name.startswith("setup_deletion") or setup_name.startswith("setup_sliding_puzzle"):
                    deleted_pos = dataset.loc[(dataset.Step==step),['Deleted_agents_positions']].values.tolist()[0][0]
                    deleted_pos = eval(deleted_pos)
                    nb_moves_per_step = dataset.loc[(dataset.Step==step),['Nb_moves']].values.tolist()[0][0]

                nb_rows = params['grid_nb_rows']
                nb_cols = params['grid_nb_cols']
                if setup_name.startswith("setup_scalability"):
                    setup_name_chunks = re.split(r'[_x]', setup_name) # original setup_name: setup_scalability_NxM
                    nb_rows = int(setup_name_chunks[2])
                    nb_cols = int(setup_name_chunks[3])
                    switch_step = None

                # swarmGrid.plot_flag(grid_nb_rows=nb_rows,
                #                     grid_nb_cols=nb_cols,
                #                     setup_name=setup_name,
                #                     run=run,
                #                     nb_ind=best_ind_run,
                #                     gen=None,
                #                     nb_eval=None,
                #                     n=n,
                #                     step=step,
                #                     flag=flag_list,
                #                     fitness=fitness,
                #                     permutated_pos=permutated_pos,
                #                     deleted_pos=deleted_pos,
                #                     nb_moves_per_step=nb_moves_per_step,
                #                     analysis_dir_plots=params['analysis_dir']['plots'])

            if setup_name == "original_flag_copied_from_learning":
                break # there is only one repetition for this file to plot
        
        swarmGrid.plot_multi_flag_fitnesses_from_file(data_flag_dir=params['analysis_dir']['data']+"/"+setup_name,
                                                    setup_name=setup_name,
                                                    run=run,
                                                    switch_step=switch_step,
                                                    analysis_dir_plots=params['analysis_dir']['plots'])
    
        # if setup_name.startswith("setup_deletion") or setup_name.startswith("setup_sliding_puzzle"):
        #     swarmGrid.plot_nb_moves_from_file(data_flag_dir=params['analysis_dir']['data']+"/"+setup_name,
        #                                     setup_name=setup_name,
        #                                     run=run,
        #                                     grid_size=params['grid_nb_rows']*params['grid_nb_cols'],
        #                                     switch_step=switch_step,
        #                                     analysis_dir_plots=params['analysis_dir']['plots'])

        #     swarmGrid.plot_multi_nb_moves_from_file(data_flag_dir=params['analysis_dir']['data']+"/"+setup_name,
        #                                             setup_name=setup_name,
        #                                             run=run,
        #                                             grid_size=params['grid_nb_rows']*params['grid_nb_cols'],
        #                                             switch_step=switch_step,
        #                                             analysis_dir_plots=params['analysis_dir']['plots'])


    if params['setup_sliding_puzzle']['setup_sliding_puzzle_bool']:

        df = pd.read_csv(f"{params['analysis_dir']['data']}/data_all_repetitions/setup_sliding_puzzle/data_setup_sliding_puzzle_stats_per_repetition.csv")

        y_labels = sorted(params['setup_sliding_puzzle']['setup_sliding_puzzle_probas_move'], reverse=True) # fluidity (p_move)
        ticks_percent = sorted(params['setup_sliding_puzzle']['setup_sliding_puzzle_ticks_percent'], reverse=True)
        x_labels = [round((1-x), 2) for x in ticks_percent] # density = ( (grid_size - nb_deletions) / grid_size)
        data = pd.DataFrame(np.nan, index=y_labels, columns=x_labels)

        for tick_percent in ticks_percent:
            for p_move in y_labels:

                density = round((1.0-tick_percent), 2)
                if density == 1.0:
                    p_move = 0.0 # if density is max, agents can't move

                deletions = int( (params['grid_nb_rows'] * params['grid_nb_cols']) * tick_percent)
                mean_fit_over_repetitions = df.loc[(df.Deletions==deletions) & (df.Fluidity==p_move), 'Flags_distance'].mean()
                # print(density, deletions, mean_fit_)
                data.loc[p_move, density] = mean_fit_over_repetitions

        sns.set_theme(style='dark')

        # Customize with darker settings
        # plt.rcParams.update({
        #     # 'axes.facecolor': '#181818',   # Darker plot background
        #     'axes.facecolor': '#808080',   # Darker plot background
        #     # 'figure.facecolor': '#101010', # Darker figure background
        #     # 'axes.edgecolor': '#333333',   # Slightly lighter edges for contrast
        #     # 'grid.color': '#303030',       # Darker grid lines
        #     # 'xtick.color': '#CCCCCC',      # Light gray for x-axis ticks
        #     # 'ytick.color': '#CCCCCC',      # Light gray for y-axis ticks
        #     # 'axes.labelcolor': '#FFFFFF',  # White for axis labels
        #     # 'text.color': '#FFFFFF',       # White for any text
        #     # 'lines.color': '#FFDD44',      # Bright color for lines
        #     # 'lines.linewidth': 2.0,        # Slightly thicker lines for visibility
        #     # 'legend.facecolor': '#202020', # Legend background
        #     # 'legend.edgecolor': '#444444', # Legend edge color
        # })

        heatmap_plot = sns.heatmap(data, annot=False, annot_kws={"size": 9}, fmt=".3f", cmap="Blues", cbar=True, linewidths=0.5, linecolor='white', vmin=0.0, vmax=1.0) # annot=True to show fit values on cells
        
        # Add a crossed rectangle for each not evaluated case
        for case in [(9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9)]: # cases density 1.0, fluidity > 0.0 are crossed
            crossed_box = plt.Rectangle(case, 1, 1, facecolor='none', edgecolor='grey', hatch='//')
            heatmap_plot.add_patch(crossed_box)

        # Add a red rectangle around the learning setup parameters case (ideal generalization environment) (se c'éééééééééééééééééééééééééééééééééééééééééééééé)
        rect_pos_col = data.columns.get_loc(round(1-params['setup_sliding_puzzle_phase1_VS_phase2']['learning_nb_deletions_percent'][1], 2))
        rect_pos_row = data.index.get_loc(params['setup_sliding_puzzle_phase1_VS_phase2']['learning_proba_move'])
        rect = plt.Rectangle((rect_pos_col, rect_pos_row), 1, 1, fill=False, edgecolor='red', linewidth=3)
        heatmap_plot.add_patch(rect)

        plt.xlabel("Density of the system", fontsize=12)
        plt.ylabel("Fluidity of the system", fontsize=12)
        plt.title(f"Generalization of learning $\\rho$={round(1-params['setup_sliding_puzzle_phase1_VS_phase2']['learning_nb_deletions_percent'][1], 2)}, $\\Phi$={params['setup_sliding_puzzle_phase1_VS_phase2']['learning_proba_move']}, best_ind_{best_ind_run:03}" + f"\nsliding puzzle {params['flag_pattern']} {params['grid_nb_rows']}x{params['grid_nb_cols']}, 11 runs", fontsize=12)

        dir_name = f"{params['analysis_dir']['plots']}/plot_all_repetitions"
        if not (os.path.exists(dir_name)):
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(f"{dir_name}/plot_sliding_puzzle_incremental_{params['flag_pattern']}_{params['grid_nb_rows']}x{params['grid_nb_cols']}_best_ind_{best_ind_run:03}_generalization_density_fluidity.png")

        plt.clf()
        plt.close()


    # if params['setup_sliding_puzzle_phase1_VS_phase2']['setup_sliding_puzzle_phase1_VS_phase2_bool']:
    #     data_flag_dirs = []
    #     learning_ticks = params['setup_sliding_puzzle_phase1_VS_phase2']['learning_ticks_units']
    #     data_flag_dirs.append([f"{params['analysis_dir']['data']}/{setup_name}" for setup_name in setups if setup_name.startswith(f"setup_sliding_puzzle_phase1_VS_phase2_{learning_ticks[0]}")])
    #     # data_flag_dirs.append([f"{params['analysis_dir']['data']}/{setup_name}" for setup_name in setups if setup_name.startswith(f"setup_sliding_puzzle_phase1_VS_phase2_control_{learning_ticks[0]}")])
    #     setups_names = [f"setup_sliding_puzzle_phase1_VS_phase2_{learning_ticks[0]}"]
    #     # setups_names = [f"setup_sliding_puzzle_phase1_VS_phase2_control_{learning_ticks[0]}"]
    #     if learning_ticks[1] != learning_ticks[0]:
    #         data_flag_dirs.append([f"{params['analysis_dir']['data']}/{setup_name}" for setup_name in setups if setup_name.startswith(f"setup_sliding_puzzle_phase1_VS_phase2_{learning_ticks[1]}")])
    #         # data_flag_dirs.append([f"{params['analysis_dir']['data']}/{setup_name}" for setup_name in setups if setup_name.startswith(f"setup_sliding_puzzle_phase1_VS_phase2_control_{learning_ticks[1]}")])
    #         setups_names.append(f"setup_sliding_puzzle_phase1_VS_phase2_{learning_ticks[1]}")
    #         # setups_names.append(f"setup_sliding_puzzle_phase1_VS_phase2_control_{learning_ticks[1]}")

    #     swarmGrid.plot_merged_multi_flag_fitnesses_from_file(data_flag_dirs=data_flag_dirs,
    #                                                         setups_names=setups_names,
    #                                                         run=run,
    #                                                         analysis_dir_plots=params['analysis_dir']['plots'])

    time_run = time.time() - time_run_ind
    print(f"swarm_analysis plots run n.{run}, best_ind_{best_ind_run} - Completed. Execution time: {time_run} seconds")


###########################################################################
# Parallelization
###########################################################################

def worker(task):

    run, best_ind_run, params = task
    return plot_single_run_single_ind_data(run=run, best_ind_run=best_ind_run, params=params)

#---------------------------------------------------

def parallelize_processes(task_queue, with_parallelization_nb_free_cores):

    # Create a Pool with the number of available cores
    available_cores = cpu_count() - with_parallelization_nb_free_cores
    with Pool(processes=available_cores) as pool:
        pool.map(worker, task_queue)
        pool.close() # no more tasks will be submitted to the pool
        pool.join() # wait for all processes to complete

    
###########################################################################
# MAIN
###########################################################################

if (__name__ == "__main__"):

    # Get parameters from the bash launcher
    parser = argparse.ArgumentParser()
    parser.add_argument("--swarm_analysis_dir", default="", type=str)
    parser.add_argument("--with_parallelization_bool", default=False, type=lambda x:x=="True")
    parser.add_argument("--with_parallelization_nb_free_cores", default=0, type=int)
    parser.add_argument("--plot_with_animation_bool", default=False, type=lambda x:x=="True")
    args = parser.parse_args()

    # Get parameters from the learning simulation
    with open(args.swarm_analysis_dir+"/swarm_params.json", "r") as f:
        params = json.load(f)
    
    params['with_parallelization_bool'] = args.with_parallelization_bool
    params['with_parallelization_nb_free_cores'] = args.with_parallelization_nb_free_cores
    params['plot_with_animation_bool'] = args.plot_with_animation_bool

    best_ind_per_run_dict = get_best_ind_per_run_dict(dataset_path=params['analysis_dir']['root'].replace("/swarm", "/learning")+"/data_all_runs/data_evo_all_runs_best_ind_per_run_per_phase.csv")

    # Launch plots
    if params['with_parallelization_bool']:
        task_queue = [] # create a queue of tasks to execute
        for run in range(params['nb_runs']):
            for best_ind_run in best_ind_per_run_dict:
                task_queue.append((run, best_ind_run, params.copy()))
        swarm_params = parallelize_processes(task_queue, params['with_parallelization_nb_free_cores'])
    else:
        for run in range(params['nb_runs']):
            for best_ind_run in best_ind_per_run_dict:
                plot_single_run_single_ind_data(run=run, best_ind_run=best_ind_run, params=params)