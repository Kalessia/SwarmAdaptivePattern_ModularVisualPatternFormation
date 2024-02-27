import time

import json

from initializations import *
from analysis import *

import numpy as np

sep = "\n################################################\n"




def cmaES_EvoAlgorithm():


    params = get_parameters_from_json()
    params = init_analysis(params)
    params = set_env(params)
    toolbox = init_toolbox(params)

    save_data_to_csv(params['analysis_dir'] + "/data/data_all_pop.csv", [], header = ["Generation", "Fitness", "Individual"])



    


    ind_x = [] 
    ind_y = []


    for gen in range(1):




        # OPTIMIZATION POUR BLOQUER AVANT L OPTIMIZATION
        # while not any(conditions.values())
        # https://deap.readthedocs.io/en/master/examples/bipop_cmaes.html

        # http://deap.gel.ulaval.ca/doc/default/examples/eda.html

        population = toolbox.generate() # generate a new population of λ individuals of type ind_init from the current strategy
        minfit = np.inf
        eval_results = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, eval_results):

            # final individual evaluation
            ind.fitness.values = fit
            # ind.fit = fit
            # ind_x.append(ind[0])
            # ind_y.append(ind[1])
            # print(ind, ind[0], ind[1])
            if fit[0] < minfit:
                best_ind = ind
                minfit = fit[0]

            # if True:
            #     print("evaluate: ind =", ind, "ind.fit =", fit)
                

        write_EvoAlgorithm_data(gen, population)

        toolbox.update(population) # update the current covariance matrix strategy from the population; update the strategy with the evaluated individuals

        # print("")

        # plot_bla_bla(params['env'][ 'env_boundaries_2D'], gen, ind_x, ind_y, params['env']['eval_function'], best_ind)
                
        # ind_x = [] 
        # ind_y = []


    print("best", best_ind[0], best_ind[1], minfit )
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

    # Launch the Evolutionary Algorithm
    t = time.time()
    params = cmaES_EvoAlgorithm()
    t = time.time() - t
    print("\nExecution time:", t, "seconds")

    # Save a trace of used parameters for this run in run_params.json
    params['execution_time'] = t
    del params['env']['eval_function'] # not JSON serializable object
    with open(params['analysis_dir'] + "/run_params.json", "w") as f:
        json.dump({k:v for k,v in params.items()}, f, indent=2)


        
    # if global_params['write_analyses_bool']:
    #     write_success_archives(analysis_dir, env_fit=global_params['eval_fit_params']['env_fit'], ds_success_threshold=0.0, best_fitness=global_params['eval_fit_params']['best_fitness']-global_params['eval_fit_params']['best_fitness_lim'])
    #     print("Results saved in " + analysis_dir)

    #     if global_params['plot_analyses']:
    #         plot_all_metrics(analysis_dir, global_params, global_params['show_plot_bool'])