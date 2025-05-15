import os
import time
import random
from multiprocessing import Pool, cpu_count

import argparse
import json

from learning_main import cmaES_EvoAlgorithm, worker, parallelize_processes
from coordinates_learning_initializations import copy_params_from_learning_x, get_best_ind_ever, set_env
from coordinates_learning_analysis import init_all_runs_analysis

sep = "\n################################################\n"


###########################################################################
# MAIN
###########################################################################

if (__name__ == "__main__"):

    print("\ncoordinates_learning_main running...")

    # Get parameters from the bash launcher
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_analysis_dir", default="", type=str)
    args = parser.parse_args()

    # Get parameters from config files
    with open(args.learning_analysis_dir+"/learning/learning_params.json", "r") as f:
        learning_params = json.load(f)
    
    with open(os.getcwd()+"/coordinates_learning_params.json", "r") as f: # os.getcwd() should be the 'src' directory
        coordinates_params = json.load(f)

    # Check previous learning for compatibility
    if learning_params['evolutionary_settings']['env_name'] not in ['sliding_puzzle_coordinates', 'sliding_puzzle_multiEnvs_coordinates']:
        print(f"Error in coordinates_learning_initializations.py - Coordinate learning cannot occur without first constructing a coordinate system, i.e. learning ['evolutionary_settings']['env_name'] should be in ['sliding_puzzle_coordinates', 'sliding_puzzle_multiEnvs_coordinates'].\nlearning_main stopped. Please correct the entry parameter in swarm_params.json before restart.")
        exit()

    # Initializations
    learning_analysis_dir_root=learning_params['analysis_dir']['root']
    coordinates_params = init_all_runs_analysis(learning_analysis_dir_root=learning_analysis_dir_root, params=coordinates_params)
    coordinates_params = copy_params_from_learning_x(learning_gradient_params=learning_params, coordinates_params=coordinates_params)
    coordinates_params['learning_best_ind_ever'], coordinates_params['best_ind_ever_fitness'] = get_best_ind_ever(dataset_path=learning_analysis_dir_root+"/data_all_runs/data_evo_all_runs_best_ind_per_run.csv")
    coordinates_params = set_env(coordinates_params)

    # Launch the Evolutionary Algorithm with parallelization over runs OR sequentially
    if coordinates_params['with_parallelization_bool']:
        coordinates_params = parallelize_processes(coordinates_params['evolutionary_settings']['nb_runs'], coordinates_params)
    else:
        for run in range(coordinates_params['evolutionary_settings']['nb_runs']):
            coordinates_params = cmaES_EvoAlgorithm(run, coordinates_params)

    # Save a trace of used parameters for this simulation in coordinates_params.json
    del coordinates_params['env']['eval_function'] # not JSON serializable object
    del coordinates_params['env']['eval_function_params']['controller'] # not JSON serializable object
    del coordinates_params['env']['eval_function_params']['agent_type'] # not JSON serializable object
    with open(coordinates_params['analysis_dir']['root']+"/coordinates_learning_params.json", "w") as f:
        json.dump({k:v for k,v in coordinates_params.items()}, f, indent=2)

    # Plots: this line allows the learning_launch.sh script to plot figures
    print(coordinates_params['analysis_dir']['root'].replace("/learning_coordinates", ""))