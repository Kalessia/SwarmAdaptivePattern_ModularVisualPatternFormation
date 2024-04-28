import warnings

from deap import creator
from deap import base
from deap import algorithms
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
        file_name = os.getcwd() +"/src/learning_params.json" # ???
        print("sta a vedere uuuh", file_name)

    with open(file_name, "r") as f:
        params = json.load(f)

    return params


###########################################################################
# Environment initialization
###########################################################################

def set_env(params):

    if params['env_name'] == 'flag_automata':
        nn_controller = NeuralNetwork(nb_neuronsPerInputs=params['nb_neuronsPerInputs'],
                                      nb_hiddenLayers=params['nb_hiddenLayers'],
                                      nb_neuronsPerHidden=params['nb_neuronsPerHidden'],
                                      nb_neuronsPerOutputs=params['nb_neuronsPerOutputs'])
        ind_size = len(nn_controller.getWeightsList())
        params['ind_size'] = ind_size

    environments = {
                    'sphere': {
                        'eval_function': sphere_function, 
                        'eval_function_params': None,
                        'env_boundaries': None,
                        'toolbox_cmaes': {
                            'centroid': [10.0]*params['ind_size'], # starting individual
                            'sigma': 3.0,
                            'lambda_': 20*params['ind_size'] # number of offspring to generate from the centroid individual
                        },
                        'env_boundaries_2D': {
                            'theta_space_min_x': -15,
                            'theta_space_min_y': -15,
                            'theta_space_max_x': 15,
                            'theta_space_max_y': 15
                        },
                    },
                    'rastrigin': {
                        'eval_function': rastrigin_function,
                        'eval_function_params': None,
                        'toolbox_cmaes': {
                            'centroid': [5.0]*params['ind_size'],
                            'sigma': 5.0,
                            'lambda_': 20*params['ind_size']
                        },
                        'env_boundaries_2D': {
                            'theta_space_min_x': -5.12,
                            'theta_space_min_y': -5.12,
                            'theta_space_max_x': 5.12,
                            'theta_space_max_y': 5.12
                        },
                    },
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
                            'sigma': 0.5,
                            'lambda_': None
                        }
                    }
    }

    if params['env_name'] in environments.keys():
        params['env'] = environments[params['env_name']]
        return params

    print("gnééé") # Kale
    exit() 


###########################################################################
# DEAP initialization
# Preparation of the EA with the DEAP framework.
# See https://deap.readthedocs.io for more details.
###########################################################################

def init_toolbox(params):

    np.random.seed(random.randint(1, 100))
    
    toolbox = base.Toolbox()

    warnings.filterwarnings("ignore", category=RuntimeWarning, message="A class named 'FitnessMin' has already been created")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="A class named 'Individual' has already been created")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="A class named 'Strategy' has already been created")
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    creator.create("Strategy", list)
    
    toolbox.register("evaluate", params['env']['eval_function'], params['env']['eval_function_params'], params['analysis_dir'])

    toolbox.register("map", map)

    # strategy = cma.Strategy(centroid=numpy.random.uniform(-5, 5, N), sigma=0.5, lambda_=params['off_lambda'])
    c, s, l = params['env']['toolbox_cmaes']['centroid'], params['env']['toolbox_cmaes']['sigma'], params['env']['toolbox_cmaes']['lambda_']
    # strategy = cma.Strategy(centroid=c, sigma=s, lambda_=l) l=none foctionne?
    strategy = cma.Strategy(centroid=c, sigma=s)
    # default lambda_ = int(4 + 3 * log(N)) with N the individual’s size (integer)

    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    return toolbox


