import os
import time
import random
from multiprocessing import Pool, cpu_count

import argparse
import json

import numpy as np
import copy

from environments import sliding_puzzle, sliding_puzzle_multiEnvs
from agents import agent2Outputs, agent3Outputs, agent3Outputs_Devert2011, agentCoordinates_gradient, agent2Outputs_RGB, agent3Outputs_RGB 
from nn import NeuralNetwork

from swarm_rollout_analysis import init_all_runs_analysis, init_one_run_analysis, write_single_run_analysis
from coordinates_learning_initializations import get_best_ind_ever, get_best_ind_per_run_dict

sep = "\n################################################\n"


###########################################################################
# CMAES
###########################################################################

def swarm_rollout_simulation(run, best_ind, best_ind_run, swarm_rollout_params):
    print(f"swarm_rollout simulation run n.{run}, best_ind_{best_ind_run} [{best_ind[0]}, ...] - Starting")

    # Initializations
    time_run = time.time()
    swarm_rollout_params = init_one_run_analysis(run, best_ind, best_ind_run, swarm_rollout_params)
    swarm_rollout_params = get_env(swarm_rollout_params)

    swarm_rollout_params['env']['eval_function'](
            env_eval_function_params=swarm_rollout_params['env']['eval_function_params'],
            analysis_dir=swarm_rollout_params['analysis_dir'],
            run=run,
            gen=0,
            nb_eval=0,
            nb_ind=best_ind_run,
            best_fit=1.0, # 1.0 is the maximal fitness value, this allows the writing of data in environments.py
            weights=best_ind,
            sliding_puzzle_nb_deletions=0,
            sliding_puzzle_proba_move=0.0)


    time_run = time.time() - time_run
    # write_single_run_analysis(run, time_run, swarm_rollout_params['analysis_dir'])
    print(f"swarm_rollout simulation run n.{run}, best_ind_{best_ind_run} [{best_ind[0]}, ...] - Execution time:", time_run, "seconds")

    return swarm_rollout_params

#---------------------------------------------------

def get_env(swarm_rollout_params):

    environments = {
        'sliding_puzzle': {
            'eval_function': sliding_puzzle,
            'eval_function_params': {
                'grid_nb_rows': swarm_rollout_params['grid']['grid_nb_rows'],
                'grid_nb_cols': swarm_rollout_params['grid']['grid_nb_cols'],
                'flags_distance_mode': None, # not implemented
                'flag_pattern': swarm_rollout_params['grid']['flag_pattern'],
                'flag_target': None,
                'init_cell_state_value': swarm_rollout_params['grid']['init_cell_state_value'],
                'agent_type': swarm_rollout_params['nn_controller']['agent_type'],
                'controller': swarm_rollout_params['nn_controller']['controller'],
                'nn_controller_stacking_mode': swarm_rollout_params['ann2_nn_controller']['nn_controller_stacking_mode'],
                'nb_intrasteps': swarm_rollout_params['swarm_rollout_settings']['sliding_puzzle_nb_intrasteps'],
                'time_steps': swarm_rollout_params['environment']['time_steps'],
                'time_window_start': swarm_rollout_params['environment']['time_window_start'],
                'time_window_end': swarm_rollout_params['environment']['time_window_end'],
                'learning_modes': [],
                'noise_std': None,
                'verbose_debug': swarm_rollout_params['verbose_debug'],
                'analysis_dir': swarm_rollout_params['analysis_dir']
            },
        },
        'sliding_puzzle_multiEnvs': {
            'eval_function': sliding_puzzle_multiEnvs,
            'eval_function_params': {
                'env_dims_list': swarm_rollout_params['swarm_rollout_settings']['sliding_puzzle_multiEnvs']['env_dims_list'], # list of [grid_nb_rows, grid_nb_cols]
                'flags_distance_mode': None, # not implemented
                'flag_pattern': swarm_rollout_params['grid']['flag_pattern'],
                'flag_target': None,
                'init_cell_state_value': swarm_rollout_params['grid']['init_cell_state_value'],
                'agent_type': swarm_rollout_params['nn_controller']['agent_type'],
                'controller': swarm_rollout_params['nn_controller']['controller'],
                'nn_controller_stacking_mode': swarm_rollout_params['ann2_nn_controller']['nn_controller_stacking_mode'],
                'nb_intrasteps': swarm_rollout_params['swarm_rollout_settings']['sliding_puzzle_nb_intrasteps'],
                'time_steps': swarm_rollout_params['environment']['time_steps'],
                'time_window_start': swarm_rollout_params['environment']['time_window_start'],
                'time_window_end': swarm_rollout_params['environment']['time_window_end'],
                'learning_modes': [],
                'noise_std': None,
                'verbose_debug': swarm_rollout_params['verbose_debug'],
                'analysis_dir': swarm_rollout_params['analysis_dir']
            }
        }
    }

    return copy.deepcopy(environments[swarm_rollout_params['swarm_rollout_settings']['env_name']])

#---------------------------------------------------

def deduce_agent_type(params, ann_index=1):

    flag_pattern = params['grid']['flag_pattern']

    # ann_index = 1 → ANN1 (direct learning or coordinate system)
    if ann_index == 1:

        if flag_pattern.startswith("rgb"):
            if params['ann1_nn_controller']['nb_neuronsPerOutputs'] == 5:
                return agent3Outputs_RGB
            if params['ann1_nn_controller']['nb_neuronsPerOutputs'] == 4:
                return agent2Outputs_RGB

        else:
            if params['ann1_nn_controller']['nb_neuronsPerOutputs'] == 3:
                return agent3Outputs
            if params['ann1_nn_controller']['nb_neuronsPerOutputs'] == 2:
                return agent2Outputs
                
        # NB: the case agent_type = agentCoordinates_gradient should never happen

    # ann_index = 2 → ANN1+ANN2 (coordinates system + pixel ANN to guess the pattern)
    elif ann_index == 2:

        stacking_mode = params['ann2_nn_controller']['nn_controller_stacking_mode']
        if flag_pattern.startswith("rgb"):

            if stacking_mode == "ann1_ann2_modelA": # Model A: 4-x-3_2-y-1
                return agent2Outputs_RGB # size_chemicals_to_spread = 1, size_phenotype = 1 (r, g, b)

            elif stacking_mode == "ann1_ann2_modelB": # Model B: 4-x-3_6-y-2
                return agent3Outputs_RGB # size_chemicals_to_spread = 2, size_phenotype = 1 (r, g, b)

            elif stacking_mode == "ann1_ann2_modelC":  # Model C: 4-x-3_6-y-1
                return agent2Outputs_RGB  # size_chemicals_to_spread = 1, size_phenotype = 1 (r, g, b)

            elif stacking_mode == "ann1_ann2_modelE":  # Model E: 4-x-3_10-y-2
                return agent3Outputs_RGB  # size_chemicals_to_spread = 2, size_phenotype = 1 (r, g, b)

        else:

            if stacking_mode == "ann1_ann2_modelA":  # Model A: 4-x-3_2-y-1
                return agent2Outputs  # size_chemicals_to_spread = 1, size_phenotype = 1

            elif stacking_mode == "ann1_ann2_modelB":  # Model B: 4-x-3_6-y-2
                return agent3Outputs  # size_chemicals_to_spread = 2, size_phenotype = 1

            elif stacking_mode == "ann1_ann2_modelC":  # Model C: 4-x-3_6-y-1
                return agent2Outputs  # size_chemicals_to_spread = 1, size_phenotype = 1

            elif stacking_mode == "ann1_ann2_modelE":  # Model E: 4-x-3_10-y-2
                return agent3Outputs  # size_chemicals_to_spread = 2, size_phenotype = 1


###########################################################################
# Parallelization
###########################################################################

def worker(task):
    run, best_ind, best_ind_run, swarm_rollout_params = task
    return swarm_rollout_simulation(run=run, best_ind=best_ind, best_ind_run=best_ind_run, swarm_rollout_params=swarm_rollout_params)

#---------------------------------------------------

def parallelize_processes(nb_runs, best_ind_per_run_dict, swarm_rollout_params):

    # Create a queue of tasks to execute
    task_queue = []
    for run in range(nb_runs):
        for best_ind_run in best_ind_per_run_dict:
            task_queue.append((run, best_ind_per_run_dict[best_ind_run], best_ind_run, copy.deepcopy(swarm_rollout_params)))

    # Create a Pool with the number of available cores
    available_cores = cpu_count() - swarm_rollout_params['with_parallelization_nb_free_cores']
    with Pool(processes=available_cores) as pool:
        results_list = pool.map(worker, task_queue) # results_list contains the 'swarm_rollout_params' of each run, respecting the ascending order from 0 to nb_runs

    return results_list[-1]


###########################################################################
# MAIN
###########################################################################

if (__name__ == "__main__"):

    print("\nswarm_rollout_main running...")

    with open(os.getcwd()+"/swarm_rollout_params.json", "r") as f: # os.getcwd() should be the 'src' directory
        swarm_rollout_params = json.load(f)

    # Get parameters from config files
    with open(swarm_rollout_params['ann1_learning_path']+"/learning/learning_params.json", "r") as f:
        learning_params = json.load(f)

    swarm_rollout_params['ann1_nn_controller'] = learning_params['nn_controller']
    ann1 = NeuralNetwork(input_size=swarm_rollout_params['ann1_nn_controller']['nb_neuronsPerInputs'],
                        hidden_layers=swarm_rollout_params['ann1_nn_controller']['hidden_layers'],
                        output_size=swarm_rollout_params['ann1_nn_controller']['nb_neuronsPerOutputs'],
                        activation_function=swarm_rollout_params['ann1_nn_controller']['activation_function'])
                        # find agent type   
    swarm_rollout_params['nn_controller'] = {}

    if swarm_rollout_params['ann2_coordinates_learning_path'] is not None and os.path.exists(swarm_rollout_params['ann2_coordinates_learning_path']): # we build a double ANN (coordinate system + pixel ANN)

        if learning_params['evolutionary_settings']['env_name'] not in ['sliding_puzzle_coordinates', 'sliding_puzzle_multiEnvs_coordinates']:
            print(f"\nError in swarm_rollout_main.py - Learning folder is not a coordinate system (pattern coordinates): can't connect a second ANN. swarm_rollout_main stopped. Please correct the 'learning_analysis_dir' parameter in swarm_rollout_params.json before restart.")
            exit()

        with open(swarm_rollout_params['ann2_coordinates_learning_path']+"/coordinates_learning_params.json", "r") as f:
            coordinates_params = json.load(f)

        # We set ANN1 to be the best coordinate system
        swarm_rollout_params['ann1_best_ind_ever'], swarm_rollout_params['ann1_best_ind_ever_fitness'] = get_best_ind_ever(dataset_path=swarm_rollout_params['ann1_learning_path']+"/learning/data_all_runs/data_evo_all_runs_best_ind_per_run.csv")
        ann1.set_weights_biases_vectors_from_list(swarm_rollout_params['ann1_best_ind_ever'])

        # We set ANN2, we will test the best_ind for each coordinate learning run
        swarm_rollout_params['ann2_nn_controller'] = coordinates_params['coordinates_nn_controller']
        ann2 = NeuralNetwork(input_size=swarm_rollout_params['ann2_nn_controller']['nb_neuronsPerInputs'],
                            hidden_layers=swarm_rollout_params['ann2_nn_controller']['hidden_layers'],
                            output_size=swarm_rollout_params['ann2_nn_controller']['nb_neuronsPerOutputs'],
                            activation_function=swarm_rollout_params['ann2_nn_controller']['activation_function'])
        
        swarm_rollout_params['best_ind_per_run_dict'] = get_best_ind_per_run_dict(dataset_path=swarm_rollout_params['ann2_coordinates_learning_path']+"/data_all_runs/data_evo_all_runs_best_ind_per_run_per_phase.csv")
        swarm_rollout_params = init_all_runs_analysis(learning_analysis_dir_root=swarm_rollout_params['ann2_coordinates_learning_path'], params=swarm_rollout_params)
        swarm_rollout_params['nn_controller']['agent_type'] = deduce_agent_type(params=swarm_rollout_params, ann_index=2) # NB: the outputs from ANN1 are managed in environments -> compute_agent_state, no need of agent_type for ANN1 if we link ANN1+ANN2
        swarm_rollout_params['nn_controller']['controller'] = [ann1, ann2]

    else:  # we build a single ANN (pixel ANN) to test a pattern directly without coordinate system 

        swarm_rollout_params['ann2_nn_controller'] = {"nn_controller_stacking_mode": None}
        swarm_rollout_params['best_ind_per_run_dict'] = get_best_ind_per_run_dict(dataset_path=swarm_rollout_params['ann1_learning_path']+"/learning/data_all_runs/data_evo_all_runs_best_ind_per_run_per_phase.csv")
        swarm_rollout_params = init_all_runs_analysis(learning_analysis_dir_root=swarm_rollout_params['ann1_coordinates_learning_path'], params=swarm_rollout_params)
        swarm_rollout_params['nn_controller']['agent_type'] = deduce_agent_type(params=swarm_rollout_params, ann_index=1)
        swarm_rollout_params['nn_controller']['controller'] = [ann1]


    # Launch the swarm_rollout simulation with parallelization over runs and individuals OR sequentially
    if swarm_rollout_params['with_parallelization_bool']:
        swarm_rollout_params = parallelize_processes(nb_runs=swarm_rollout_params['swarm_rollout_settings']['nb_runs'], best_ind_per_run_dict=swarm_rollout_params['best_ind_per_run_dict'], swarm_rollout_params=swarm_rollout_params)
    else:
        for run in range(swarm_rollout_params['swarm_rollout_settings']['nb_runs']):
            for best_ind_run in best_ind_per_run_dict:
                swarm_rollout_params = swarm_rollout_simulation(run=run, best_ind=swarm_rollout_params['best_ind_per_run_dict'][best_ind_run], best_ind_run=best_ind_run, swarm_rollout_params=copy.deepcopy(swarm_rollout_params))

    # Plot ANNs used in this simulation
    for i, ann in enumerate(swarm_rollout_params['nn_controller']['controller']):
        ann.plot_neural_network(env_name=f"ann{i+1}", analysis_dir=swarm_rollout_params['analysis_dir']['root']+"/plots_all_runs")

    # Save a trace of used parameters for this simulation in swarm_rollout_params.json
    del swarm_rollout_params['env']['eval_function'] # not JSON serializable object
    del swarm_rollout_params['env']['eval_function_params']['controller'] # not JSON serializable object
    del swarm_rollout_params['env']['eval_function_params']['agent_type'] # not JSON serializable object
    del swarm_rollout_params['nn_controller']['controller'] # not JSON serializable object
    del swarm_rollout_params['nn_controller']['agent_type'] # not JSON serializable object
    with open(swarm_rollout_params['analysis_dir']['root']+"/swarm_rollout_params.json", "w") as f:
        json.dump({k:v for k,v in swarm_rollout_params.items()}, f, indent=2)

    # Plots: this line allows the launch.sh script to plot figures
    print(swarm_rollout_params['analysis_dir']['root'])