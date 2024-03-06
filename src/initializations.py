import multiprocessing

from deap import creator
from deap import base
from deap import algorithms
from deap import cma

import numpy as np
import random

from environments import *

import json




###########################################################################
# Parameters initialization
###########################################################################

def get_parameters_from_json():

    with open("/home/kalessia/flagAutomata/src/params.json", "r") as f:
        params = json.load(f)

    return params




###########################################################################
# Environment initialization
###########################################################################

def set_env(params):

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
                            'centroid': [5.0]*params['ind_size'], # starting individual
                            'sigma': 5.0,
                            'lambda_': 20*params['ind_size'] # number of offspring to generate from the centroid individual
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
                            'init_cell_state_value': params['init_cell_state_value'],
                            'time_steps': params['time_steps'],
                            'time_window_start': params['time_window_start'],
                            'time_window_end': params['time_window_end']
                        },
                        'env_boundaries': None,
                        'toolbox_cmaes': {
                            # 'centroid': [10.0]*params['ind_size'], 
                            # 'centroid': list(np.random.uniform(-1, 1, params['ind_size'])), # good! sigma: 0.5
                            'centroid': list(np.random.uniform(-1, 1, params['ind_size'])),
                            'sigma': 0.5,
                            'lambda_': params['off_lambda']
                        },
                        'env_boundaries_2D': {
                            'theta_space_min_x': -15,
                            'theta_space_min_y': -15,
                            'theta_space_max_x': 15,
                            'theta_space_max_y': 15
                        },
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

# Individual generator
# def generateES(icls, scls, size, imin, imax, smin, smax):
#     """
#     icls: the class of individual to instantiate
#     """
#     print(size, imin, imax, smin, smax)
#     ind = icls(random.uniform(imin, imax) for _ in range(size))
#     ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
#     return ind

#---------------------------------------------------

# def convert_list_to_ind(ind):
#     return creator.Individual(ind)

#---------------------------------------------------

# def clip_deap_ind(ind, min_val, max_val):
#     for i_gene, gene in enumerate(ind):
#         ind[i_gene] = max(min(ind[i_gene], max_val), min_val)
#         return ind

#---------------------------------------------------

# def update_ds_success_zone_radius(ds_success_zone_radius_step):
#     global_params['env_DS_params']['success_zone_radius'] += ds_success_zone_radius_step
#     global_params['eval_fit_params']['best_fitness_lim'] += ds_success_zone_radius_step
#     toolbox.register("evaluate", global_params['eval_functiontion'], **global_params['eval_params'], **global_params['env_DS_params'], **global_params['eval_fit_params'], **global_params['noise_params'])
#     toolbox.register("evaluate_det", global_params['eval_functiontion'], **global_params['eval_params'], **global_params['env_DS_params'], **global_params['eval_fit_params'], noise_bool=False)

#---------------------------------------------------

def init_toolbox(params):

    # numpy.random.seed(14)
    np.random.seed(random.randint(1, 100))
    
    toolbox = base.Toolbox()

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    creator.create("Strategy", list)
    
    toolbox.register("evaluate", params['env']['eval_function'], params['env']['eval_function_params']['automata_nb_rows'], params['env']['eval_function_params']['automata_nb_cols'], params['env']['eval_function_params']['init_cell_state_value'], params['env']['eval_function_params']['time_steps'], params['env']['eval_function_params']['time_window_start'], params['env']['eval_function_params']['time_window_end'], params['analysis_dir_run'])
    # kale solve acquisition

    if params['with_parallelization_bool']:
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)
    else:
        toolbox.register("map", map)

    # strategy = cma.Strategy(centroid=numpy.random.uniform(-5, 5, N), sigma=0.5, lambda_=params['off_lambda'])
    c, s, l = params['env']['toolbox_cmaes']['centroid'], params['env']['toolbox_cmaes']['sigma'], params['env']['toolbox_cmaes']['lambda_']
    strategy = cma.Strategy(centroid=c, sigma=s, lambda_=l)

    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    return toolbox


