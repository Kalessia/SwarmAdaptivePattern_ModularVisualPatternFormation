import time

import numpy as np

from deap import tools

from initializations import *
from analysis import *

import json

sep = "\n################################################\n"




def cmaES_EvoAlgorithm(run, params):

    params = init_one_run_analysis(run, params)
    toolbox = init_toolbox(params)


    # ind_x = [] 
    # ind_y = []

    # minall= np.inf
    # best_ind = None

    for gen in range(params['nb_generations']):




        # OPTIMIZATION POUR BLOQUER AVANT L OPTIMIZATION
        # while not any(conditions.values())
        # https://deap.readthedocs.io/en/master/examples/bipop_cmaes.html

        # http://deap.gel.ulaval.ca/doc/default/examples/eda.html

        population = toolbox.generate() # generate a new population of λ individuals of type ind_init from the current strategy
        # best_ind_gen = None
        # mingen = np.inf



        eval_results = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, eval_results):

            # final individual evaluation
            ind.fitness.values = fit

            # if fit[0] < mingen:
            #     print("nex best gen", gen, fit[0])
            #     best_ind_gen = ind
            #     mingen = fit[0]

            #     if fit[0] < minall:
            #         best_ind = ind
            #         minall = fit[0]

            # if True:
            #     print("evaluate: ind =", ind, "ind.fit =", fit)
        # print("gen", gen, "best", best_ind_gen[0], best_ind_gen[1], mingen)

        write_single_run_data(run, gen, population, params['analysis_dir_data'])

        toolbox.update(population) # update the current covariance matrix strategy from the population; update the strategy with the evaluated individuals

        # print("")

        # plot_bla_bla(params['env'][ 'env_boundaries_2D'], gen, ind_x, ind_y, params['env']['eval_function'], best_ind)
                
        # ind_x = [] 
        # ind_y = []


    # print("run", run, "best", best_ind[0], best_ind[1], minall)
    plot_single_run_data(params['analysis_dir_data'], params['analysis_dir_plots'])
    #plot(params['analysis_dir'])

    #---------------------------------------------------
    # Evolutionary Algorithm initialization
    #---------------------------------------------------

    # population = toolbox.population(n=global_params["mu"])
    # population = [clip_deap_ind(ind=ind, min_val=global_params['min_value'], max_val=global_params['max_value']) for ind in population]

    # dict_fit, dict_bd_dimA, dict_bd_dimB, dict_joints_x, dict_joints_y, population = evaluate(toolbox, population, global_params['sampling_n'])
    # archive.init_archive(population, dict_fit, dict_bd_dimA, dict_bd_dimB, dict_joints_x, dict_joints_y)


    # if write_analyses_bool:
    #     t_gen_start = time.time()
    #     data_outcome_archive, data_outcome_archive_det, data_all_pop, data_best_inds_fit, data_best_inds_nov, data_coverage, data_coverage_det, data_time = collect_write_data(0, global_params, population, 0., data_outcome_archive, data_outcome_archive_det, data_all_pop, data_best_inds_fit, data_best_inds_nov, data_coverage, data_coverage_det, data_time, analysis_dir)

    # if verbose_debug:
    #     print("\nmain.py --- Evolutionary algorithm initialization terminated. Population size:", len(population), "\n", sep)
    # else:
    #     pbar.write("Generation n.0 done. Outcome archive size: " + str(global_params['archive'].get_nb_elites()) + "; Success archive size: " + str(global_params['archive'].get_nb_elites_success(env_fit=global_params['eval_fit_params']['env_fit'], ds_success_threshold=0.0, best_fitness=global_params['eval_fit_params']['best_fitness']-global_params['eval_fit_params']['best_fitness_lim'])))
    #     pbar.update(1)



    return params










###########################################################################
# MAIN
###########################################################################

if (__name__ == "__main__"):

    params = get_parameters_from_json()
    params = init_all_runs_analysis(params)
    params = set_env(params)

    for run in range(params['nb_runs']):
        # Launch the Evolutionary Algorithm
        t = time.time()
        params = cmaES_EvoAlgorithm(run, params)
        t = time.time() - t
        print("\ncmaES run n." + str(run) + " - Execution time:", t, "seconds") # kale change save time for each run

    # Save a trace of used parameters for this simulation in run_params.json
    params['simulation_execution_time'] = t # kale useful?
    del params['env']['eval_function'] # not JSON serializable object
    with open(params['analysis_dir'] + "/run_params.json", "w") as f:
        json.dump({k:v for k,v in params.items()}, f, indent=2)

    write_all_runs_data(params['analysis_dir'])
    # plot_all_runs_data(params['analysis_dir'])