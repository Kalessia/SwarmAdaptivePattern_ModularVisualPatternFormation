import os
import time

import pandas as pd
import numpy as np

import json

from swarm import swarmGrid
from simulation_analysis import *

sep = "\n################################################\n"




def swarm_simulation(run, simulation_params, best_ind_ever, flag_target):

    # Initializations
    time_run = time.time()
    simulation_params = init_one_run_analysis(run, simulation_params)
    env = swarmGrid(grid_nb_rows=3, grid_nb_cols=5, agent_controller_weights=best_ind_ever, flag_target=flag_target)
    flags = []

    time_steps = simulation_params['time_steps']
    for _ in range(time_steps):
        flags.append(env.get_flag_from_grid())
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



###########################################################################
# MAIN
###########################################################################

if (__name__ == "__main__"):

    # Initialization
    # analysis_dir = sys.argv[1]
    analysis_dir = "simulationAnalysis/flag_automata_simulation_2024-04-04_20-08-00" # exemple
    with open(analysis_dir+"/learning/learning_params.json", "r") as f:
        learning_params = json.load(f)

    if os.path.exists(analysis_dir+"/simulation"):
        shutil.rmtree(analysis_dir+"/simulation")
    
    with open("src/simulation_params.json", "r") as f:
        simulation_params = json.load(f)
 
    simulation_params = init_all_runs_analysis(learning_params['analysis_dir']['root'], simulation_params)
    # simulation_params = set_env(simulation_params)

    best_ind_ever = get_best_ind_ever(dataset_path=learning_params['analysis_dir']['root']+"/learning/data_all_runs/data_evo_all_runs_best_ind_per_run.csv")
    flag_target = get_flag_target(dataset_path=learning_params['analysis_dir']['root']+"/learning/data_all_runs/data_env_flag_target.csv")

    # Launch the Swarm simulation
    for run in range(simulation_params['nb_runs']):
        simulation_params = swarm_simulation(run, simulation_params, best_ind_ever, flag_target)

    # Save a trace of used parameters for this simulation in simulation_params.json
    # del simulation_params['env']['eval_function'] # not JSON serializable object
    with open(simulation_params['analysis_dir']['root'] + "/simulation/simulation_params.json", "w") as f:
        json.dump({k:v for k,v in simulation_params.items()}, f, indent=2)

    # # Plot data for all the runs
    # if simulation_params['plot_analysis_bool']:
    #     write_all_runs_data(simulation_params['analysis_dir']['root'])
    #     plot_all_runs_data(simulation_params)