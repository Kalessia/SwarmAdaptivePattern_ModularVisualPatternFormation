import os
import time
import shutil
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np

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

    save_data_to_csv(params['analysis_dir']['root']+"/data_all_runs/data_all_runs_time.csv", [], header = ["Run", "Time"])
    
    return params

#---------------------------------------------------

def init_one_run_analysis(run, best_ind, best_ind_run, params):

    # Create the data and plot directories tree from the 'analysis_dir' folder
    params['analysis_dir']['data'] = params['analysis_dir']['root']+"/run_"+str(run)+"/best_ind_"+str(best_ind_run)+"/data"
    params['analysis_dir']['plots'] = params['analysis_dir']['root']+"/run_"+str(run)+"/best_ind_"+str(best_ind_run)+"/plots"
    os.makedirs(params['analysis_dir']['data'], exist_ok=True)
    os.makedirs(params['analysis_dir']['plots'], exist_ok=True)

    file_path = params['analysis_dir']['root']+"/run_"+str(run)+"/best_ind_"+str(best_ind_run)+"/flag_individual.txt"
    if not os.path.exists(file_path):
        with open (file_path, 'w') as f:
            f.write(str(best_ind))

    return params


# ###########################################################################
# Plot data functions
# ###########################################################################
                
def plot_single_run_single_ind_data(run, best_ind_run, params):

    time_run_ind = time.time()
    print(f"swarm_analysis plots run n.{run}, best_ind_{best_ind_run} - Starting")

    params['analysis_dir']['data'] = params['analysis_dir']['root']+"/run_"+str(run)+"/best_ind_"+str(best_ind_run)+"/data"
    params['analysis_dir']['plots'] = params['analysis_dir']['root']+"/run_"+str(run)+"/best_ind_"+str(best_ind_run)+"/plots"

    for setup_name in params['setups']:
        for n in range(params['nb_repetitions']):
            data_flag_file = params['analysis_dir']['data']+"/"+setup_name+"/data_"+setup_name+"_flag_run_"+str(run)+"_n_"+str(n)+".csv"
            dataset = pd.read_csv(data_flag_file)

            for step in [0, params['switch_step']-1, params['switch_step'], params['time_steps']-1]:
                flag_list = dataset.loc[(dataset.Step==step),['Flag']].values.tolist()[0][0]
                flag_list = str(flag_list).replace('[', '').replace(']', '').strip()
                flag_list = np.asarray(flag_list.split(','), dtype=np.float32)
                fitness = dataset.loc[(dataset.Step==step),['Flags_distance']].values.tolist()[0][0]

                permutated_pos = []
                if setup_name.startswith("setup_permutation"):
                    permutated_pos = dataset.loc[(dataset.Step==step),['Permutated_agents_positions']].values.tolist()[0][0]
                    permutated_pos = eval(permutated_pos)

                deleted_pos = []
                if setup_name.startswith("setup_deletion"):
                    deleted_pos = dataset.loc[(dataset.Step==step),['Deleted_agents_positions']].values.tolist()[0][0]
                    deleted_pos = eval(deleted_pos)

                swarmGrid.plot_flag(grid_nb_rows=params['grid_nb_rows'],
                                    grid_nb_cols=params['grid_nb_cols'],
                                    setup_name=setup_name,
                                    run=run,
                                    nb_ind=best_ind_run,
                                    gen=None,
                                    n=n,
                                    step=step,
                                    flag=flag_list,
                                    fitness=fitness,
                                    permutated_pos=permutated_pos,
                                    deleted_pos=deleted_pos,
                                    analysis_dir_plots=params['analysis_dir']['plots'])

        swarmGrid.plot_multi_flag_fitnesses_from_file(data_flag_dir=params['analysis_dir']['data']+"/"+setup_name,
                                                      setup_name=setup_name,
                                                      run=run,
                                                      switch_step=params['switch_step'],
                                                      analysis_dir_plots=params['analysis_dir']['plots'])

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

    best_ind_per_run_dict = get_best_ind_per_run_dict(dataset_path=params['analysis_dir']['root'].replace("/swarm", "/learning")+"/data_all_runs/data_evo_all_runs_best_ind_per_run.csv")

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