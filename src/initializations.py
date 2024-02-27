import os
from datetime import datetime

import multiprocessing

from deap import creator
from deap import base
from deap import tools
from deap import algorithms
from deap import cma

import numpy

from environments import *

import csv
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
                            'time_steps': 200,
                            'time_window_start': 140,
                            'time_window_end': 200
                        },
                        'env_boundaries': None,
                        'toolbox_cmaes': {
                            'centroid': [10.0]*params['ind_size'], 
                            'sigma': 3.0,
                            'lambda_': 20
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

    numpy.random.seed(14)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    creator.create("Strategy", list)
    
    toolbox = base.Toolbox()
    toolbox.register("evaluate", params['env']['eval_function'], params['env']['eval_function_params']['time_steps'], params['env']['eval_function_params']['time_window_start'], params['env']['eval_function_params']['time_window_end'], params['analysis_dir'])
    # kale solve acquisition

    if params['withParallelization_bool']:
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)
    else:
        toolbox.register("map", map)

    # strategy = cma.Strategy(centroid=numpy.random.uniform(-5, 5, N), sigma=0.5, lambda_=params['off_lambda'])
    c, s, l = params['env']['toolbox_cmaes']['centroid'], params['env']['toolbox_cmaes']['sigma'], params['env']['toolbox_cmaes']['lambda_']
    strategy = cma.Strategy(centroid=c, sigma=s, lambda_=l)

    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)





    # if global_params['algo'] not in implemented_algos:
    #     print("main.py --- Error in DEAP initialization: unknown QD algorithm in global_params['algo'].")
    #     exit()

    # elif global_params['algo'] in ["NS+FIT", "NSLC"]: # multi-objectives to maximize by Pareto front
    #     # NSGA-II uses a binary tournament mating selection: each individual is first compared by rank and then by crowding distance.
    #     # rank = individuals are selected frontwise. By doing so, there will be the situation
    #     # where a front needs to be split because not all individuals are allowed to survive.
    #     # In this splitting front, solutions are selected based on crowding distance (Manhatten Distance in the objective space).
    #     creator.create("MyFitness", base.Fitness, weights=(1.0,1.0)) 
    #     toolbox.register("select", tools.selNSGA2)

    # else: # [NS, MAP-Elites, MAP-Elites-S] # mono-objective to maximize
    #     # Select the k 'best' individuals among the input individuals, following 'fitness' or 'novelty'
    #     creator.create("MyFitness", base.Fitness, weights=(1.0,))
    #     toolbox.register("select", tools.selBest) 

    # NB: [DeepGrid, DeepGrid-S] selection is made with the fitness_proportionate_selection function, in archive.py


    # creator.create("Individual", array.array, typecode="d", fitness=creator.MyFitness, strategy=None)
    # creator.create("Strategy", array.array, typecode="d")

    # toolbox.register("individual", generateES, creator.Individual, creator.Strategy, 
    #                  global_params["ind_size"],
    #                  global_params["min_value"],
    #                  global_params["max_value"],
    #                  global_params["min_strategy"],
    #                  global_params["max_strategy"])

    # toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # toolbox.register("mate", tools.cxESBlend, alpha=0.1)

    # if global_params['fitness_strategy'] == "DS":
    #     toolbox.register("mutate", tools.mutGaussian, mu=0., sigma=global_params['mut_sigma'], indpb=global_params['mut_indpb'])
    # else:
    #     toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=global_params['mut_indpb']) # default: c=1.0, indpb=0.3


    # toolbox.register("map", futures.map)

    # # toolbox.decorate("mate", checkStrategy(global_params["min_strategy"]))
    # # toolbox.decorate("mutate", checkStrategy(global_params["min_strategy"]))

    # toolbox.register("evaluate", global_params['eval_functiontion'], **global_params['eval_params'], **global_params['env_DS_params'], **global_params['eval_fit_params'], **global_params['noise_params'])
    # toolbox.register("evaluate_det", global_params['eval_functiontion'], **global_params['eval_params'], **global_params['env_DS_params'], **global_params['eval_fit_params'], noise_bool=False)

    return toolbox


