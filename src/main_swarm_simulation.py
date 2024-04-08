import os
import time
import shutil
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np

import json

from swarm import swarmGrid
from simulation_analysis import *

sep = "\n################################################\n"




def swarm_simulation(run, simulation_params):
    print(f"swarm_simulation run n.{run} - Starting")

    # Initializations
    time_run = time.time()
    simulation_params = init_one_run_analysis(run, simulation_params)
    env = swarmGrid(grid_nb_rows=9, grid_nb_cols=9, agent_controller_weights=simulation_params['best_ind_ever'], flag_target=simulation_params['flag_target'])
    flags = []

    time_steps = simulation_params['time_steps']
    for step in range(time_steps):

        if step == 100:
            print(env.grid_map_pos_agent)
            agent1 = env.grid_map_pos_agent[tuple((2,2))]
            agent2 = env.grid_map_pos_agent[tuple((8,8))]
            env.exchange_agents(agent1, agent2)

        flags.append(env.get_flag_from_grid())
        env.plot_flag(run, step, analysis_dir_plots=simulation_params['analysis_dir']['plots']+"/env")
        env.step()

    time_run = time.time() - time_run
    print("swarm_simulation run n." + str(run) + " - Execution time:", time_run, "seconds") # kale change save time for each run
    write_single_run_data(env, run, time_run, time_steps, flags, analysis_dir=simulation_params['analysis_dir'])

    # Plot data for one single run
    if simulation_params['plot_analysis_bool']:
        plot_single_run_data(run, simulation_params)

    return simulation_params

#---------------------------------------------------

def get_best_ind_ever(dataset_path=None):

    dataset = pd.read_csv(dataset_path)
    best_fitness = dataset['Fitness'].min()
    best_ind_ever = dataset.loc[dataset.Fitness==best_fitness, 'Individual'].values.tolist()[0]

    best_ind_ever = dataset.loc[(dataset.Fitness==best_fitness),['Individual']].values.tolist()[0][0]
    best_ind_ever = str(best_ind_ever).replace('[', '').replace(']', '').strip()
    best_ind_ever = list(np.asarray(best_ind_ever.split(','), dtype=np.float32))

    return best_ind_ever

#---------------------------------------------------

def get_flag_target(dataset_path=None):

    dataset = pd.read_csv(dataset_path)
    flag_target = dataset.iloc[0]['Flag']
    flag_target = str(flag_target).replace('[', '').replace(']', '').strip()
    flag_target = list(np.asarray(flag_target.split(','), dtype=np.float32))

    return flag_target

#---------------------------------------------------

def worker(task):
    run, simulation_params = task
    return swarm_simulation(run, simulation_params)

#---------------------------------------------------

def parallelize_processes(nb_runs, simulation_params):
    
    # Create a queue of tasks to execute
    task_queue = [(run, simulation_params.copy()) for run in range(nb_runs)]

    # Create a Pool with the number of available cores
    available_cores = cpu_count() - simulation_params['with_parallelization_nb_free_cores']
    with Pool(processes=min(nb_runs, available_cores)) as pool:
        results_list = pool.map(worker, task_queue) # results_list contains the 'simulation_params' of each run, respecting the ascending order from 0 to nb_runs

    return results_list[-1]


###########################################################################
# MAIN
###########################################################################

if (__name__ == "__main__"):

    # Initialization
    # analysis_dir = sys.argv[1]
    analysis_dir = "simulationAnalysis/flag_automata_simulation_2024-04-07_20-49-44" # exemple
    with open(analysis_dir+"/learning/learning_params.json", "r") as f:
        learning_params = json.load(f)

    if os.path.exists(analysis_dir+"/simulation"):
        shutil.rmtree(analysis_dir+"/simulation")
    
    with open("src/simulation_params.json", "r") as f:
        simulation_params = json.load(f)

    simulation_params = init_all_runs_analysis(learning_params['analysis_dir']['root'], simulation_params)
    simulation_params['best_ind_ever'] = get_best_ind_ever(dataset_path=learning_params['analysis_dir']['root']+"/learning/data_all_runs/data_evo_all_runs_best_ind_per_run.csv")
    simulation_params['flag_target'] = get_flag_target(dataset_path=learning_params['analysis_dir']['root']+"/learning/data_all_runs/data_env_flag_target.csv")

    if simulation_params['with_parallelization_bool']:
        # Launch the Swarm simulation with parallelization over runs
        simulation_params = parallelize_processes(simulation_params['nb_runs'], simulation_params)
    else:
        # Launch the Swarm simulation sequentially
        for run in range(simulation_params['nb_runs']):
            simulation_params = swarm_simulation(run, simulation_params)

    # Save a trace of used parameters for this simulation in simulation_params.json
    # del simulation_params['env']['eval_function'] # not JSON serializable object
    del simulation_params['best_ind_ever'] # à insererrrrrrrr
    del simulation_params['flag_target']
    with open(simulation_params['analysis_dir']['root'] + "/simulation/simulation_params.json", "w") as f:
        json.dump({k:v for k,v in simulation_params.items()}, f, indent=2)

    # # Plot data for all the runs
    # if simulation_params['plot_analysis_bool']:
    #     write_all_runs_data(simulation_params['analysis_dir']['root'])
    #     plot_all_runs_data(simulation_params)