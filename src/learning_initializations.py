import warnings

from deap import creator
from deap import base
from deap import cma

import numpy as np
import random

from learning_environments import *
from nn import NeuralNetwork




###########################################################################
# Environment initialization
###########################################################################

def check_params_validity(params):

    exit_bool = False

    params['nb_runs'] = int(params['nb_runs'])
    params['nb_generations'] = int(params['nb_generations'])
    params['automata_nb_rows'] = int(params['automata_nb_rows'])
    params['automata_nb_cols'] = int(params['automata_nb_cols'])
    params['init_cell_state_value'] = float(params['init_cell_state_value'])
    params['time_steps'] = int(params['time_steps'])
    params['time_window_start'] = int(params['time_window_start'])
    params['time_window_end'] = int(params['time_window_end'])
    params['nb_neuronsPerInputs'] = int(params['nb_neuronsPerInputs'])
    params['nb_hiddenLayers'] = int(params['nb_hiddenLayers'])
    params['nb_neuronsPerHidden'] = int(params['nb_neuronsPerHidden'])
    params['nb_neuronsPerOutputs'] = int(params['nb_neuronsPerOutputs'])
    params['with_parallelization_nb_free_cores'] = int(params['with_parallelization_nb_free_cores'])

    if params['nb_runs'] <= 0 or params['nb_generations'] <= 0 or params['automata_nb_rows'] <= 0 or params['automata_nb_cols'] <= 0 \
        or params['nb_neuronsPerInputs'] <= 0 or params['nb_neuronsPerHidden'] <= 0 or params['nb_neuronsPerOutputs'] <= 0 or params['time_steps'] <= 0:
        print(f"Error in learning_initializations.py - Parameters nb_runs, nb_generations, automata_nb_rows, automata_nb_cols, nb_neuronsPerInputs, nb_neuronsPerHidden, nb_neuronsPerOutputs and time_steps must be > 0")
        exit_bool = True

    if params['nb_hiddenLayers'] < 0 or params['with_parallelization_nb_free_cores'] < 0:
        print(f"Error in learning_initializations.py - Parameters nb_hiddenLayers and with_parallelization_nb_free_cores must be >= 0")
        exit_bool = True

    if params['init_cell_state_value'] < 0.0 or params['init_cell_state_value'] > 1.0:
        print(f"Error in learning_initializations.py - Parameter init_cell_state_value must be in [0.0, 1.0]")
        exit_bool = True

    if params['time_window_start'] < 0 or params['time_window_start'] >= params['time_steps']:
        print(f"Error in learning_initializations.py - The time_window_start parameter value must be in [0,{params['time_steps']}[")
        exit_bool = True

    if params['time_window_end'] < 0 or params['time_window_end'] <= params['time_window_start'] or params['time_window_end'] > params['time_steps']:
        print(f"Error in learning_initializations.py - The time_window_end parameter value must be in ]{params['time_window_start']},{params['time_steps']}]")
        exit_bool = True

    patterns = ['flag_automata']
    if params['env_name'] not in patterns:
        print(f"Error in learning_initializations.py - The env_name parameter must be one of the following: {patterns}")
        exit_bool = True

    patterns = ['2_stripes', '3_stripes', 'circle', 'half_circle']
    if params['flag_pattern'] not in patterns:
        print(f"Error in learning_initializations.py - The flag_pattern parameter must be one of the following: {patterns}")
        exit_bool = True

    if exit_bool:
        print("learning_main stopped. Please correct the entry parameter in learning_params.json before restart.")
        exit()

#---------------------------------------------------

def set_env(params):

    check_params_validity(params)

    nn_controller = NeuralNetwork(nb_neuronsPerInputs=params['nb_neuronsPerInputs'],
                                    nb_hiddenLayers=params['nb_hiddenLayers'],
                                    nb_neuronsPerHidden=params['nb_neuronsPerHidden'],
                                    nb_neuronsPerOutputs=params['nb_neuronsPerOutputs'])
    ind_size = len(nn_controller.getWeightsList())
    params['ind_size'] = ind_size

    environments = {
        'flag_automata': {
            'eval_function': flag_automata, 
            'eval_function_params': {
                'automata_nb_rows': params['automata_nb_rows'], 
                'automata_nb_cols': params['automata_nb_cols'],
                'flag_pattern': params['flag_pattern'],
                'init_cell_state_value': params['init_cell_state_value'],
                'controller': nn_controller,
                'time_steps': params['time_steps'],
                'time_window_start': params['time_window_start'],
                'time_window_end': params['time_window_end']
            },
            'env_boundaries': None,
            'toolbox_cmaes': {
                'centroid': list(np.random.uniform(params['ind_min_value'], params['ind_max_value'], ind_size)),
                'sigma': 0.5
            }
        }
    }

    params['env'] = environments[params['env_name']]
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

    return toolbox

