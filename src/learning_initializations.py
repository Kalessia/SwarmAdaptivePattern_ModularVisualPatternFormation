import warnings

from deap import creator
from deap import base
from deap import cma

import numpy as np
import random

import csv

from environments import sliding_puzzle, sliding_puzzle_multiEnvs
from agents import agent2Outputs, agent3Outputs, agent3Outputs_Devert2011, agentCoordinates_gradient, agent2Outputs_RGB, agent3Outputs_RGB 
from nn import NeuralNetwork


###########################################################################
# Environment initialization
###########################################################################

def check_params_validity(params):

    exit_bool = False

    params['evolutionary_settings']['nb_runs'] = int(params['evolutionary_settings']['nb_runs'])
    params['evolutionary_settings']['nb_evals'] = int(params['evolutionary_settings']['nb_evals'])

    if params['evolutionary_settings']['nb_runs'] <= 0 or params['grid']['grid_nb_rows'] <= 0 or params['grid']['grid_nb_cols'] <= 0 or params['environment']['time_steps'] <= 0:
        print(f"Error in learning_initializations.py - Parameters nb_runs, grid_nb_rows, grid_nb_cols and time_steps must be > 0")
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

    if not params['evolutionary_settings']['sliding_puzzle_nb_intrasteps']:
        params['evolutionary_settings']['sliding_puzzle_nb_intrasteps'] = None

    if params['evolutionary_settings']['env_name'] == "sliding_puzzle_incremental":
        if not isinstance(params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_density_ticks'], list):
            print(f"Error in learning_initializations.py - Parameter sliding_puzzle_incremental_density_ticks must be a list")
            params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_density_ticks'] = []
            exit_bool = True
        params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_ticks'] = [int(params['grid']['grid_size']*(1.0-tick)) for tick in params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_density_ticks']]
    else:
        params['evolutionary_settings']['sliding_puzzle_nb_deletions_ticks'] = int(params['grid']['grid_size']*(1.0-params['evolutionary_settings']['sliding_puzzle_density']))

    if params['evolutionary_settings']['env_name'] in ['sliding_puzzle_multiEnvs', 'sliding_puzzle_multiEnvs_coordinates']: # to name the learning folder and to cause a crush in case of wrong environment executed
        params['grid']['grid_nb_rows'] = "N"
        params['grid']['grid_nb_cols'] = "N"

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

    patterns = ['sliding_puzzle', 'sliding_puzzle_incremental', 'sliding_puzzle_coordinates', 'sliding_puzzle_multiEnvs', 'sliding_puzzle_multiEnvs_coordinates']
    if params['evolutionary_settings']['env_name'] not in patterns:
        print(f"Error in learning_initializations.py - The env_name parameter must be one of the following: {patterns}")
        exit_bool = True

    patterns = ['two-bands', 'three-bands', 'centered-disc', 'not-centered-disc', 'centered-half-discs', 'not-centered-half-discs', 'bn-SU', 'bn-smile1', 'bn-smile2', 'rgb-italian-flag', 'rgb-french-cockade', 'rgb-rainbow-full', 'rgb-rainbow-arrow']
    if params['grid']['flag_pattern'] not in patterns:
        print(f"Error in learning_initializations.py - The flag_pattern parameter must be one of the following: {patterns}")
        exit_bool = True

    #---------------------------------------------------

    if exit_bool:
        print("learning_main stopped. Please correct the entry parameter in learning_params.json before restart.")
        exit()

    return params


#---------------------------------------------------

def set_env(params):

    if params['evolutionary_settings']['env_name'] in ["sliding_puzzle_coordinates", "sliding_puzzle_multiEnvs_coordinates"]:
        nn_controller = NeuralNetwork(input_size=4, # ann input = [ signal_xy_N, signal_xy_W, signal_xy_E, signal_xy_S ] 
                                    hidden_layers=[],
                                    output_size=3, # ann output = [ signal_xy, x, y ]
                                    activation_function='tanh')
        agent_type = agentCoordinates_gradient # size_chemicals_to_spread = 1, size_phenotype = 2
        flag_pattern = "coordinates" # 2D flag representing a 2D coordinates system (x,y)
    
    else: # one only learning phase (no coordinates system) to learn directly the flag target 
    
        if params['grid']['flag_pattern'].startswith("rgb"):
            # Control experience 1 (one big ann instead of 2 distinct learning phases (coordinate system method), with similar research space)
            nn_controller = NeuralNetwork(input_size=8, # ann input = [ signal_xy_N, signal_xy_W, signal_xy_E, signal_xy_S, signal_p_N, signal_p_W, signal_p_E, signal_p_S ] 
                                        hidden_layers=[3],
                                        output_size=5, # ann output = [signal_xy, signal_p, pR, pG, pB ]
                                        activation_function='tanh')
            agent_type = agent3Outputs_RGB # size_chemicals_to_spread = 2, size_phenotype = 1 (r, g, b)

            # Control experience 2 (GECCO model: one signal from each neighbor, one signal to spread and one phenotype to evaluate as output)
            # nn_controller = NeuralNetwork(input_size=4, # ann input = [ signal_N, signal_W, signal_E, signal_S ] 
            #                             hidden_layers=[3],
            #                             output_size=4, # ann output = [signal, pR, pG, pB ]
            #                             activation_function='tanh')
            # agent_type = agent2Outputs_RGB # size_chemicals_to_spread = 1, size_phenotype = 1 (r, g, b)

        else:
            # Control experience 1 (one big ANN instead of 2 distinct learning phases (coordinate system method), with similar research space)
            nn_controller = NeuralNetwork(input_size=8, # ann input = [ signal_xy_N, signal_xy_W, signal_xy_E, signal_xy_S, signal_p_N, signal_p_W, signal_p_E, signal_p_S ] 
                                        hidden_layers=[3],
                                        output_size=3, # ann output = [signal_xy, signal_p, p ]
                                        activation_function='tanh')
            agent_type = agent3Outputs # size_chemicals_to_spread = 2, size_phenotype = 1

            # Control experience 2 (GECCO model: one signal from each neighbor, one signal to spread and one phenotype to evaluate as output)
            # nn_controller = NeuralNetwork(input_size=4, # ann input = [ signal_N, signal_W, signal_E, signal_S ] 
            #                             hidden_layers=[3],
            #                             output_size=2, # ann output = [signal, p ]
            #                             activation_function='tanh')
            # agent_type = agent2Outputs # size_chemicals_to_spread = 1, size_phenotype = 1

        flag_pattern = params['grid']['flag_pattern']

    params['nn_controller'] = {}
    params['nn_controller']['nb_neuronsPerInputs'] = nn_controller.input_size
    params['nn_controller']['hidden_layers'] = nn_controller.hidden_layers
    params['nn_controller']['nb_neuronsPerOutputs'] = nn_controller.output_size
    params['nn_controller']['activation_function'] = nn_controller.activation_function

    params['evolutionary_settings']['ind_size'] = nn_controller.weights_biases_size
    if agent_type == agent3Outputs_Devert2011: # Devert 2011
        params['evolutionary_settings']['ind_size'] = params['evolutionary_settings']['ind_size'] + 4 # Devert 2011 +4 weights for the Expression function

    environments = {
        'sliding_puzzle': {
            'eval_function': sliding_puzzle,
            'eval_function_params': {
                'grid_nb_rows': params['grid']['grid_nb_rows'],
                'grid_nb_cols': params['grid']['grid_nb_cols'],
                'flags_distance_mode': params['evolutionary_settings']['flags_distance_mode'],
                'flag_pattern': flag_pattern,
                'flag_target': None,
                'init_cell_state_value': params['grid']['init_cell_state_value'],
                'agent_type': agent_type,
                'controller': [nn_controller],
                'nn_controller_stacking_mode': None,
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
                'centroid': list(np.random.uniform(-1, 1, params['evolutionary_settings']['ind_size'])),
                'sigma': 0.5
            }
        },
        'sliding_puzzle_multiEnvs': {
            'eval_function': sliding_puzzle_multiEnvs,
            'eval_function_params': {
                'env_dims_list': params['evolutionary_settings']['sliding_puzzle_multiEnvs']['env_dims_list'], # list of [grid_nb_rows, grid_nb_cols]
                'flags_distance_mode': params['evolutionary_settings']['flags_distance_mode'],
                'flag_pattern': flag_pattern,
                'flag_target': None,
                'init_cell_state_value': params['grid']['init_cell_state_value'],
                'agent_type': agent_type,
                'controller': [nn_controller],
                'nn_controller_stacking_mode': None,
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
                'centroid': list(np.random.uniform(-1, 1, params['evolutionary_settings']['ind_size'])),
                'sigma': 0.5
            }
        }
    }

    params['env'] = environments['sliding_puzzle_multiEnvs'] if params['evolutionary_settings']['env_name'] in ['sliding_puzzle_multiEnvs', 'sliding_puzzle_multiEnvs_coordinates'] else environments['sliding_puzzle']
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