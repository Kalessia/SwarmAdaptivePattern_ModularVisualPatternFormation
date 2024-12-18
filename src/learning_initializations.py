import warnings

from deap import creator
from deap import base
from deap import cma

import numpy as np
import random

import csv

from environments import flag_automata, sliding_puzzle_incremental
from nn import NeuralNetwork


###########################################################################
# Environment initialization
###########################################################################

def check_params_validity(params):

    exit_bool = False

    params['evolutionary_settings']['nb_runs'] = int(params['evolutionary_settings']['nb_runs'])
    params['evolutionary_settings']['nb_evals'] = int(params['evolutionary_settings']['nb_evals'])

    if params['evolutionary_settings']['nb_runs'] <= 0 or params['grid']['grid_nb_rows'] <= 0 or params['grid']['grid_nb_cols'] <= 0 \
        or params['nn_controller']['nb_neuronsPerInputs'] <= 0 or params['nn_controller']['nb_neuronsPerOutputs'] <= 0 or params['environment']['time_steps'] <= 0:
        print(f"Error in learning_initializations.py - Parameters nb_runs, grid_nb_rows, grid_nb_cols, nb_neuronsPerInputs, nb_neuronsPerOutputs and time_steps must be > 0")
        exit_bool = True
    
    #---------------------------------------------------

    params['nn_controller']['nb_neuronsPerInputs'] = int(params['nn_controller']['nb_neuronsPerInputs'])
    params['nn_controller']['nb_neuronsPerOutputs'] = int(params['nn_controller']['nb_neuronsPerOutputs'])
    
    if not isinstance( params['nn_controller']['hidden_layers'], list):
        print(f"Error in learning_initializations.py - Parameter hidden_layers must be a list")
        exit_bool = True

    #---------------------------------------------------

    params['grid']['grid_nb_rows'] = int(params['grid']['grid_nb_rows'])
    params['grid']['grid_nb_cols'] = int(params['grid']['grid_nb_cols'])
    params['grid']['grid_size'] = params['grid']['grid_nb_rows'] * params['grid']['grid_nb_cols']

    if params['grid']['init_cell_state_value'] is not None:
        params['grid']['init_cell_state_value'] = float(params['grid']['init_cell_state_value'])
        if params['grid']['init_cell_state_value'] < 0.0 or params['grid']['init_cell_state_value'] > 1.0:
            print(f"Error in learning_initializations.py - Parameter init_cell_state_value must be in [0.0, 1.0]")
            exit_bool = True

    #---------------------------------------------------

    if params['evolutionary_settings']['env_name'] != "sliding_puzzle_incremental":
        params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_switch_eval'] = None # used in learning_analysis.py to plot the switch eval line (setups delimiter)

    if params['evolutionary_settings']['env_name'] == "sliding_puzzle_incremental":
        if params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_units'] is None:
            if not isinstance(params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_percent'], list):
                print(f"Error in learning_initializations.py - Parameter sliding_puzzle_incremental_nb_deletions_percent must be a list")
                params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_percent'] = []
                exit_bool = True
            params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_ticks'] = [int(params['grid']['grid_size']*tick_percent) for tick_percent in params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_percent']]
        else:
            if params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_percent'] is not None:
                print(f"Error in learning_initializations.py - Either 'sliding_puzzle_incremental_nb_deletions_units' or 'sliding_puzzle_incremental_nb_deletions_percent' must be null.")
                exit_bool = True
            if not isinstance(params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_units'], list):
                print(f"Error in learning_initializations.py - Parameter sliding_puzzle_incremental_nb_deletions_units must be a list")
                params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_units'] = []
                exit_bool = True
            params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_ticks'] = [int(tick_unit) for tick_unit in params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_units']]
    
    #---------------------------------------------------

    params['environment']['time_steps'] = int(params['environment']['time_steps'])
    params['environment']['time_window_start'] = int(params['environment']['time_window_start'])
    params['environment']['time_window_end'] = int(params['environment']['time_window_end'])

    if params['environment']['time_window_start'] < 0 or params['environment']['time_window_start'] >= params['environment']['time_steps']:
        print(f"Error in learning_initializations.py - The time_window_start parameter value must be in [0,{params['environment']['time_steps']}[")
        exit_bool = True

    if params['environment']['time_window_end'] < 0 or params['environment']['time_window_end'] <= params['environment']['time_window_start'] or params['environment']['time_window_end'] > params['environment']['time_steps']:
        print(f"Error in learning_initializations.py - The time_window_end parameter value must be in ]{params['environment']['time_window_start']},{params['environment']['time_steps']}]")
        exit_bool = True

    #---------------------------------------------------

    params['learning_modes'] = [learning_option for learning_option in params['learning_options'].keys() if params['learning_options'][learning_option][f"{learning_option}_bool"] == True]

    if "learning_with_noise" not in params['learning_modes']:
        params['learning_options']['learning_with_noise']['learning_with_noise_std'] = None

    #---------------------------------------------------

    params['with_parallelization_nb_free_cores'] = int(params['with_parallelization_nb_free_cores'])

    if params['with_parallelization_nb_free_cores'] < 0:
        params['with_parallelization_nb_free_cores'] = 0

    #---------------------------------------------------

    patterns = ['flag_automata', 'sliding_puzzle_incremental']
    if params['evolutionary_settings']['env_name'] not in patterns:
        print(f"Error in learning_initializations.py - The env_name parameter must be one of the following: {patterns}")
        exit_bool = True

    patterns = ['two-bands', 'three-bands', 'centered-disc', 'not-centered-disc', 'centered-half-discs', 'not-centered-half-discs']
    if params['grid']['flag_pattern'] not in patterns:
        print(f"Error in learning_initializations.py - The flag_pattern parameter must be one of the following: {patterns}")
        exit_bool = True

    #---------------------------------------------------

    if exit_bool:
        print("learning_main stopped. Please correct the entry parameter in learning_params.json before restart.")
        exit()


#---------------------------------------------------

def set_env(params):

    check_params_validity(params)

    nn_controller = NeuralNetwork(input_size=params['nn_controller']['nb_neuronsPerInputs'],
                                  hidden_layers=params['nn_controller']['hidden_layers'],
                                  output_size=params['nn_controller']['nb_neuronsPerOutputs'],
                                  activation_function='tanh')
    params['evolutionary_settings']['ind_size'] = nn_controller.weights_biases_size

    if params['nn_controller']['nb_neuronsPerOutputs'] == 3: # Devert 2011
        params['evolutionary_settings']['ind_size'] = params['evolutionary_settings']['ind_size'] + 4 # Devert 2011 +4 weights for the Expression function

    environments = {
        'flag_automata': {
            'eval_function': flag_automata, 
            'eval_function_params': {
                'grid_nb_rows': params['grid']['grid_nb_rows'],
                'grid_nb_cols': params['grid']['grid_nb_cols'],
                'flags_distance_mode': params['evolutionary_settings']['flags_distance_mode'],
                'flag_pattern': params['grid']['flag_pattern'],
                'flag_target': None,
                'init_cell_state_value': params['grid']['init_cell_state_value'],
                'controller': nn_controller,
                'time_steps': params['environment']['time_steps'],
                'time_window_start': params['environment']['time_window_start'],
                'time_window_end': params['environment']['time_window_end'],
                'learning_modes': params['learning_modes'],
                'noise_std': params['learning_options']['learning_with_noise']['learning_with_noise_std'],
                'verbose_debug': params['verbose_debug'],
                'analysis_dir': params['analysis_dir']
            },
            'env_boundaries': None,
            'toolbox_cmaes': {
                'centroid': list(np.random.uniform(-1, 1, params['evolutionary_settings']['ind_size'])),
                'sigma': 0.5
            }
        },
        'sliding_puzzle_incremental': {
            'eval_function': sliding_puzzle_incremental,
            'eval_function_params': {
                'grid_nb_rows': params['grid']['grid_nb_rows'],
                'grid_nb_cols': params['grid']['grid_nb_cols'],
                'flags_distance_mode': params['evolutionary_settings']['flags_distance_mode'],
                'flag_pattern': params['grid']['flag_pattern'],
                'flag_target': None,
                'init_cell_state_value': params['grid']['init_cell_state_value'],
                'controller': nn_controller,
                'time_steps': params['environment']['time_steps'],
                'time_window_start': params['environment']['time_window_start'],
                'time_window_end': params['environment']['time_window_end'],
                'learning_modes': params['learning_modes'],
                'noise_std': params['learning_options']['learning_with_noise']['learning_with_noise_std'],
                'verbose_debug': params['verbose_debug'],
                'analysis_dir': params['analysis_dir']
            },
            'env_boundaries': None,
            'toolbox_cmaes': {
                'centroid': list(np.random.uniform(-1, 1, params['evolutionary_settings']['ind_size'])),
                'sigma': 0.5
            }
        }
    }

    params['env'] = environments[params['evolutionary_settings']['env_name']]
    return params


###########################################################################
# DEAP initialization
# Preparation of the EA with the DEAP framework.
# See https://deap.readthedocs.io for more details.
###########################################################################

def init_toolbox(params):

    np.random.seed(random.randint(1, 100))
    
    toolbox = base.Toolbox()

    # Ignore the known warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="A class named 'FitnessMin' has already been created")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="A class named 'Individual' has already been created")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="A class named 'Strategy' has already been created")
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    creator.create("Strategy", list)
    
    toolbox.register("evaluate", params['env']['eval_function'], params['env']['eval_function_params'], params['analysis_dir'])
    toolbox.register("map", map)

    c, s = params['env']['toolbox_cmaes']['centroid'], params['env']['toolbox_cmaes']['sigma']
    strategy = cma.Strategy(centroid=c, sigma=s) # default lambda_ = int(4 + 3 * log(N)) with N the individual’s size (integer)

    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    return toolbox, strategy


###########################################################################
# Save data functions
###########################################################################

def save_data_to_csv(fichier_name, data, header=None):
    f = open(fichier_name, 'a', newline='')
    writer = csv.writer(f)

    if header:
        writer.writerow(header)

    if data:
        writer.writerows(data)

    f.close()

#---------------------------------------------------