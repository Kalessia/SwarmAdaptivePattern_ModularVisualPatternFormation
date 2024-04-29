import warnings

from deap import creator
from deap import base
# from deap import algorithms
from deap import cma

import numpy as np
import random

from learning_environments import *

import json




###########################################################################
# Parameters initialization
###########################################################################

def get_parameters_from_json(file_name=None):

    if file_name is None:
        file_name = os.getcwd() +"/learning_params.json"

    with open(file_name, "r") as f:
        params = json.load(f)

    return params


###########################################################################
# Environment initialization
###########################################################################

def check_params_validity(params):

    flag_patterns = ['2_stripes', '3_stripes', 'circle', 'half_circle']
    if params['flag_pattern'] not in flag_patterns:
        print("Error in learning_initializations.py - The flag_pattern must be one of the following:", flag_patterns)
        exit()

    # time_window_end check validity

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
                'controller': {
                    'nn_controller': nn_controller,
                    'nb_neuronsPerInputs': params['nb_neuronsPerInputs'],
                    'nb_hiddenLayers': params['nb_hiddenLayers'],
                    'nb_neuronsPerHidden': params['nb_neuronsPerHidden'],
                    'nb_neuronsPerOutputs': params['nb_neuronsPerOutputs']
                },
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

    if params['env_name'] in environments.keys():
        params['env'] = environments[params['env_name']]
        return params

    print("Error in learning_initializations.py - The env_name must be one of the following:", list(environments.keys()))
    exit()


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

