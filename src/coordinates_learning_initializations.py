import warnings

from deap import creator
from deap import base
from deap import cma

import pandas as pd
import numpy as np
import random

import csv

from environments import sliding_puzzle, sliding_puzzle_multiEnvs
from agents import agent2Outputs, agent3Outputs, agent2Outputs_RGB, agent3Outputs_RGB 
from nn import NeuralNetwork



###########################################################################
# Environment initialization
###########################################################################

def check_params_validity(params): # params = coordinates_params

    exit_bool = False

    # params['nb_runs'] = int(params['nb_runs'])
    # params['time_steps'] = int(params['time_steps'])
    # params['time_window_start'] = int(params['time_window_start'])
    # params['time_window_end'] = int(params['time_window_end'])
    # params['switch_step'] = int(params['switch_step'])
    # params['nb_repetitions'] = int(params['nb_repetitions'])
    # params['with_parallelization_nb_free_cores'] = int(params['with_parallelization_nb_free_cores'])

    # if params['nb_runs'] <= 0 or params['time_steps'] <= 0 or params['nb_repetitions'] <= 0:
    #     print(f"Error in swarm_initializations.py - Parameters nb_runs, time_steps and nb_repetitions must be > 0")
    #     exit_bool = True

    # if params['setup_ind_consistency']['setup_ind_consistency_bool']:
    #     setup_ind_consistency_options = []
    #     if params['setup_ind_consistency']['setup_ind_consistency_learning_conditions_bool']:
    #         setup_ind_consistency_options.append("setup_ind_consistency_learning_conditions")
    #     if params['setup_ind_consistency']['setup_ind_consistency_random_init_states_bool']:
    #         setup_ind_consistency_options.append("setup_ind_consistency_random_init_states")
    #     if params['setup_ind_consistency']['setup_ind_consistency_random_async_update_states_bool']:
    #         setup_ind_consistency_options.append("setup_ind_consistency_random_async_update_states")
    #     params['setup_ind_consistency']['setup_ind_consistency_options'] = setup_ind_consistency_options

    # if params['setup_noise']['setup_noise_bool'] and not isinstance(params['setup_noise']['setup_noise_std_ticks'], list):
    #     print(f"Error in swarm_initializations.py - Parameter setup_noise_std_ticks must be a list")
    #     exit_bool = True

    # if params['setup_permutation']['setup_permutation_bool']:
    #     if not isinstance(params['setup_permutation']['setup_permutation_density_ticks'], list):
    #         print(f"Error in swarm_initializations.py - Parameter setup_permutation_density_ticks must be a list")
    #         params['setup_permutation']['setup_permutation_density_ticks'] = []
    #         exit_bool = True
    #     permutation_ticks = [int(grid_size*(1.0-tick)) for tick in params['setup_permutation']['setup_permutation_density_ticks']]
    #     params['setup_permutation']['permutation_ticks'] = [val if val % 2 == 0 else val - 1 for val in permutation_ticks] # permutation_ticks must be even

    # if params['setup_deletion']['setup_deletion_bool']:
    #     if not isinstance(params['setup_deletion']['setup_deletion_density_ticks'], list):
    #         print(f"Error in swarm_initializations.py - Parameter setup_deletion_density_ticks must be a list")
    #         params['setup_deletion']['setup_deletion_density_ticks'] = []
    #         exit_bool = True
    #     params['setup_deletion']['deletion_ticks'] = [int(grid_size*(1.0-tick)) for tick in params['setup_deletion']['setup_deletion_density_ticks']]

    # if params['setup_sliding_puzzle']['setup_sliding_puzzle_bool']:
    #     if not isinstance(params['setup_sliding_puzzle']['setup_sliding_puzzle_density_ticks'], list) \
    #     or not isinstance(params['setup_sliding_puzzle']['setup_sliding_puzzle_probas_move'], list):
    #         print(f"Error in swarm_initializations.py - Parameter setup_sliding_puzzle_density_ticks and setup_sliding_puzzle_probas_move must be lists")
    #         params['setup_sliding_puzzle']['setup_sliding_puzzle_density_ticks'] = []
    #         exit_bool = True
    #     params['setup_sliding_puzzle']['deletion_ticks'] = [int(grid_size*(1.0-tick)) for tick in params['setup_sliding_puzzle']['setup_sliding_puzzle_density_ticks']]

    #     for p, proba_move in enumerate(params['setup_sliding_puzzle']['setup_sliding_puzzle_probas_move']):
    #         params['setup_sliding_puzzle']['setup_sliding_puzzle_probas_move'][p] = float(proba_move)
    #         if params['setup_sliding_puzzle']['setup_sliding_puzzle_probas_move'][p] < 0.0 or params['setup_sliding_puzzle']['setup_sliding_puzzle_probas_move'][p] > 1.0:
    #             print(f"Error in swarm_initializations.py - Each parameter in setup_sliding_puzzle_probas_move must be in [0.0, 1.0]")
    #             exit_bool = True
    
    if exit_bool:
        print("learning_main stopped. Please correct the entry parameter in swarm_params.json before restart.")
        exit()


#---------------------------------------------------

def set_env(params):

    # check_params_validity(params)
    
    ann1 = NeuralNetwork(input_size=params['learning_nn_controller']['nb_neuronsPerInputs'],
                        hidden_layers=params['learning_nn_controller']['hidden_layers'],
                        output_size=params['learning_nn_controller']['nb_neuronsPerOutputs'],
                        activation_function=params['learning_nn_controller']['activation_function'])
    ann1.set_weights_biases_vectors_from_list(params['learning_best_ind_ever'])

    #---------------------------------------------------

    if params['grid']['flag_pattern'].startswith("rgb"):
        # # Model A: 4-x-3_2-y-1
        # ann2 = NeuralNetwork(input_size=2, # ann2 input = [ x, y ]
        #                     hidden_layers=[5,5],
        #                     output_size=3, # one grayscale phenotype. ann2 output = [ pR, pG, pB ]
        #                     activation_function='tanh')
        # agent_type = agent2Outputs_RGB # size_chemicals_to_spread = 1, size_phenotype = 1 (r, g, b)
        # params['coordinates_nn_controller']['nn_controller_stacking_mode'] = "ann1_ann2_modelA"

        # # Model B: 4-x-3_6-y-2
        # ann2 = NeuralNetwork(input_size=6, # ann2 input = [ x, y, signal_p_N, signal_p_W, signal_p_E, signal_p_S ]
        #                     hidden_layers=[5,5],
        #                     output_size=4, # one signal_p, one grayscale phenotype. ann2 output = [ signal_p, pR, pG, pB ]
        #                     activation_function='tanh')
        # agent_type = agent3Outputs_RGB # size_chemicals_to_spread = 2, size_phenotype = 1 (r, g, b)
        # params['coordinates_nn_controller']['nn_controller_stacking_mode'] = "ann1_ann2_modelB"

        # Model C: 4-x-3_6-y-1
        ann2 = NeuralNetwork(input_size=6, # ann2 input = [ x, y, signal_xy_N, signal_xy_W, signal_xy_E, signal_xy_S ]
                            hidden_layers=[5,5],
                            output_size=3, # one grayscale phenotype. ann2 output = [ pR, pG, pB ]
                            activation_function='tanh')
        agent_type = agent2Outputs_RGB # size_chemicals_to_spread = 1, size_phenotype = 1 (r, g, b)
        params['coordinates_nn_controller']['nn_controller_stacking_mode'] = "ann1_ann2_modelC"

        # # Model E: 4-x-3_10-y-2
        # ann2 = NeuralNetwork(input_size=10, # ann2 input = [ x, y, signal_xy_N, signal_xy_W, signal_xy_E, signal_xy_S, signal_p_N, signal_p_W, signal_p_E, signal_p_S ]
        #                     hidden_layers=[5,5],
        #                     output_size=4, # one signal_p, one grayscale phenotype. ann2 output = [ signal_p, pR, pG, pB ]
        #                     activation_function='tanh')
        # agent_type = agent3Outputs_RGB # size_chemicals_to_spread = 2, size_phenotype = 1 (r, g, b)
        # params['coordinates_nn_controller']['nn_controller_stacking_mode'] = "ann1_ann2_modelE"

    else:
        # # Model A: 4-x-3_2-y-1
        # ann2 = NeuralNetwork(input_size=2, # ann2 input = [ x, y ]
        #                     hidden_layers=[5,5],
        #                     output_size=1, # one grayscale phenotype. ann2 output = [ p ]
        #                     activation_function='tanh')
        # agent_type = agent2Outputs # size_chemicals_to_spread = 1, size_phenotype = 1
        # params['coordinates_nn_controller']['nn_controller_stacking_mode'] = "ann1_ann2_modelA"

        # # Model B: 4-x-3_6-y-2
        # ann2 = NeuralNetwork(input_size=6, # ann2 input = [ x, y, signal_p_N, signal_p_W, signal_p_E, signal_p_S ]
        #                     hidden_layers=[5,5],
        #                     output_size=2, # one signal_p, one grayscale phenotype. ann2 output = [ signal_p, p ]
        #                     activation_function='tanh')
        # agent_type = agent3Outputs # size_chemicals_to_spread = 2, size_phenotype = 1
        # params['coordinates_nn_controller']['nn_controller_stacking_mode'] = "ann1_ann2_modelB"

        # Model C: 4-x-3_6-y-1
        ann2 = NeuralNetwork(input_size=6, # ann2 input = [ x, y, signal_xy_N, signal_xy_W, signal_xy_E, signal_xy_S ]
                            hidden_layers=[5,5],
                            output_size=1, # one grayscale phenotype. ann2 output = [ p ]
                            activation_function='tanh')
        agent_type = agent2Outputs # size_chemicals_to_spread = 1, size_phenotype = 1
        params['coordinates_nn_controller']['nn_controller_stacking_mode'] = "ann1_ann2_modelC"

        # # Model E: 4-x-3_10-y-2
        # ann2 = NeuralNetwork(input_size=10, # ann2 input = [ x, y, signal_xy_N, signal_xy_W, signal_xy_E, signal_xy_S, signal_p_N, signal_p_W, signal_p_E, signal_p_S ]
        #                     hidden_layers=[5,5],
        #                     output_size=2, # one signal_p, one grayscale phenotype. ann2 output = [ signal_p, p ]
        #                     activation_function='tanh')
        # agent_type = agent3Outputs # size_chemicals_to_spread = 2, size_phenotype = 1
        # params['coordinates_nn_controller']['nn_controller_stacking_mode'] = "ann1_ann2_modelE"
    
    #---------------------------------------------------

    params['coordinates_nn_controller']['nb_neuronsPerInputs'] = ann2.input_size
    params['coordinates_nn_controller']['hidden_layers'] = ann2.hidden_layers
    params['coordinates_nn_controller']['nb_neuronsPerOutputs'] = ann2.output_size
    params['coordinates_nn_controller']['activation_function'] = ann2.activation_function
    params['evolutionary_settings']['coordinates_ind_size'] = ann2.weights_biases_size
    
    environments = {
        'sliding_puzzle': {
            'eval_function': sliding_puzzle,
            'eval_function_params': {
                'grid_nb_rows': params['grid']['grid_nb_rows'],
                'grid_nb_cols': params['grid']['grid_nb_cols'],
                'flags_distance_mode': params['evolutionary_settings']['flags_distance_mode'],
                'flag_pattern': params['grid']['flag_pattern'],
                'flag_target': None,
                'init_cell_state_value': params['grid']['init_cell_state_value'],
                'agent_type': agent_type,
                'controller': [ann1, ann2],
                'nn_controller_stacking_mode': params['coordinates_nn_controller']['nn_controller_stacking_mode'],
                'nb_intrasteps': params['evolutionary_settings']['sliding_puzzle_nb_intrasteps'],
                'time_steps': params['environment']['time_steps'],
                'time_window_start': params['environment']['time_window_start'],
                'time_window_end': params['environment']['time_window_end'],
                'learning_modes': params['learning_modes'],
                'noise_std': params['learning_options']['learning_with_noise']['learning_with_noise_std'],
                'verbose_debug': params['verbose_debug'],
                'analysis_dir': params['analysis_dir']
            },
            'toolbox_cmaes': {
                'centroid': list(np.random.uniform(-1, 1, params['evolutionary_settings']['coordinates_ind_size'])),
                'sigma': 0.5
            }
        },
        'sliding_puzzle_multiEnvs': {
            'eval_function': sliding_puzzle_multiEnvs,
            'eval_function_params': {
                'env_dims_list': params['evolutionary_settings']['sliding_puzzle_multiEnvs']['env_dims_list'], # list of [grid_nb_rows, grid_nb_cols]
                'flags_distance_mode': params['evolutionary_settings']['flags_distance_mode'],
                'flag_pattern': params['grid']['flag_pattern'],
                'flag_target': None,
                'init_cell_state_value': params['grid']['init_cell_state_value'],
                'agent_type': agent_type,
                'controller': [ann1, ann2],
                'nn_controller_stacking_mode': params['coordinates_nn_controller']['nn_controller_stacking_mode'],
                'nb_intrasteps': params['evolutionary_settings']['sliding_puzzle_nb_intrasteps'],
                'time_steps': params['environment']['time_steps'],
                'time_window_start': params['environment']['time_window_start'],
                'time_window_end': params['environment']['time_window_end'],
                'learning_modes': params['learning_modes'],
                'noise_std': params['learning_options']['learning_with_noise']['learning_with_noise_std'],
                'verbose_debug': params['verbose_debug'],
                'analysis_dir': params['analysis_dir']
            },
            'toolbox_cmaes': {
                'centroid': list(np.random.uniform(-1, 1, params['evolutionary_settings']['coordinates_ind_size'])),
                'sigma': 0.5
            }
        }
    }

    params['env'] = environments['sliding_puzzle_multiEnvs'] if params['evolutionary_settings']['env_name'] in ['sliding_puzzle_multiEnvs_coordinates'] else environments['sliding_puzzle']
    return params

#---------------------------------------------------

def get_best_ind_per_run_dict(dataset_path):

    best_ind_per_run_dict = {}
    dataset = pd.read_csv(dataset_path)

    learning_phase = dataset['Learning_phase'].max() # the last phase CHECK MEILLEURE FACON DE LE FAIRE FONCTIONNER
    runs = sorted(dataset['Run'].unique())
    for run in runs:
        ind = dataset.loc[(dataset.Run==run) & (dataset.Learning_phase==learning_phase), 'Individual'].values.tolist()[0]
        ind = str(ind).replace('[', '').replace(']', '').strip()
        ind = list(np.asarray(ind.split(','), dtype=np.float64))
        best_ind_per_run_dict[run] = ind

    return best_ind_per_run_dict

#---------------------------------------------------

def get_best_ind_per_run_per_phase_dict(dataset_path=None):

    best_ind_per_run_per_phase_dict = {}
    dataset = pd.read_csv(dataset_path)

    runs = sorted(dataset['Run'].unique())
    phases = sorted(dataset['Learning_phase'].unique())
    for run in runs:
        best_ind_per_run_per_phase_dict[run] = []
        for phase in phases:
            ind = dataset.loc[(dataset.Run == run) & (dataset.Learning_phase == phase), 'Individual'].values.tolist()[0]
            ind = str(ind).replace('[', '').replace(']', '').strip()
            ind = list(np.asarray(ind.split(','), dtype=np.float64))
            best_ind_per_run_per_phase_dict[run].append(ind)

    return best_ind_per_run_per_phase_dict

#---------------------------------------------------

def get_best_ind_ever(dataset_path=None):

    dataset = pd.read_csv(dataset_path)

    best_ind_ever_fitness = dataset['Fitness'].min()
    best_ind_ever = dataset.loc[dataset.Fitness==best_ind_ever_fitness, 'Individual'].values.tolist()[0]
    best_ind_ever = str(best_ind_ever).replace('[', '').replace(']', '').strip()
    best_ind_ever = np.asarray(best_ind_ever.split(','), dtype=np.float32)
    best_ind_ever = best_ind_ever.tolist()

    return best_ind_ever, best_ind_ever_fitness

#---------------------------------------------------

def get_flag_target(dataset_path=None):

    dataset = pd.read_csv(dataset_path)
    flag_target = dataset.iloc[0]['Flag']
    flag_target = str(flag_target).replace('[', '').replace(']', '').strip()
    flag_target = list(np.asarray(flag_target.split(','), dtype=np.float32))

    return flag_target

#---------------------------------------------------

def copy_params_from_learning_x(learning_gradient_params, coordinates_params):

    coordinates_params['evolutionary_settings']['sliding_puzzle_proba_move'] = learning_gradient_params['evolutionary_settings']['sliding_puzzle_proba_move']
    coordinates_params['evolutionary_settings']['learning_ind_size'] = learning_gradient_params['evolutionary_settings']['ind_size']

    coordinates_params['learning_nn_controller'] = learning_gradient_params['nn_controller']
    coordinates_params['coordinates_nn_controller'] = {}

    # coordinates_params['evolutionary_settings']['sliding_puzzle_multiEnvs'] = {}
    # coordinates_params['evolutionary_settings']['sliding_puzzle_multiEnvs']['env_dims_list'] = None
    if coordinates_params['evolutionary_settings']['same_as_learning_gradient']:
        coordinates_params['evolutionary_settings']['nb_runs'] = learning_gradient_params['evolutionary_settings']['nb_runs']
        coordinates_params['evolutionary_settings']['nb_evals'] = learning_gradient_params['evolutionary_settings']['nb_evals']
        coordinates_params['evolutionary_settings']['env_name'] = learning_gradient_params['evolutionary_settings']['env_name']
        coordinates_params['evolutionary_settings']['flags_distance_mode'] = learning_gradient_params['evolutionary_settings']['flags_distance_mode']
        coordinates_params['evolutionary_settings']['nb_intrasteps'] = learning_gradient_params['evolutionary_settings']['sliding_puzzle_nb_intrasteps']
        if learning_gradient_params['evolutionary_settings']['env_name'] == "sliding_puzzle_multiEnvs_coordinates" and coordinates_params['evolutionary_settings']['sliding_puzzle_multiEnvs']['same_as_learning_gradient']:
            coordinates_params['evolutionary_settings']['sliding_puzzle_multiEnvs']['env_dims_list'] = learning_gradient_params['evolutionary_settings']['sliding_puzzle_multiEnvs']['env_dims_list']

    if coordinates_params['environment']['same_as_learning_gradient']:
        coordinates_params['environment']['time_steps'] = learning_gradient_params['environment']['time_steps']
        coordinates_params['environment']['time_window_start'] = learning_gradient_params['environment']['time_window_start']
        coordinates_params['environment']['time_window_end'] = learning_gradient_params['environment']['time_window_end']

    if learning_gradient_params['evolutionary_settings']['env_name'] == "sliding_puzzle_incremental":
        coordinates_params['evolutionary_settings']['sliding_puzzle_incremental'] = {}
        coordinates_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_switch_eval'] = learning_gradient_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_switch_eval']
        coordinates_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_ticks'] = learning_gradient_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_ticks']
        coordinates_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_density_ticks'] = learning_gradient_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_density_ticks']
    else:
        coordinates_params['evolutionary_settings']['sliding_puzzle_nb_deletions_ticks'] = learning_gradient_params['evolutionary_settings']['sliding_puzzle_nb_deletions_ticks']
        coordinates_params['evolutionary_settings']['sliding_puzzle_density'] = learning_gradient_params['evolutionary_settings']['sliding_puzzle_density']

    coordinates_params['grid']['grid_nb_rows'] = learning_gradient_params['grid']['grid_nb_rows']
    coordinates_params['grid']['grid_nb_cols'] = learning_gradient_params['grid']['grid_nb_cols']
    coordinates_params['grid']['init_cell_state_value'] = learning_gradient_params['grid']['init_cell_state_value']
    coordinates_params['grid']['grid_size'] = learning_gradient_params['grid']['grid_size']
    coordinates_params['learning_modes'] = learning_gradient_params['env']['eval_function_params']['learning_modes']

    return coordinates_params