import time
from multiprocessing import Pool, cpu_count

from learning_initializations import *
from learning_analysis import *

import json

sep = "\n################################################\n"


###########################################################################
# CMAES
###########################################################################

def cmaES_EvoAlgorithm(run, learning_params):
    print(f"cmaES_EvoAlgorithm run n.{run} - Started")

    # Initialization
    time_run = time.time()
    learning_params = init_one_run_analysis(run, learning_params)
    toolbox = init_toolbox(learning_params)

    for gen in range(learning_params['nb_generations']):

        # DEAP CMAES
        # To control the stopping criteria: while not any(conditions.values())
        # https://deap.readthedocs.io/en/master/examples/bipop_cmaes.html
        # http://deap.gel.ulaval.ca/doc/default/examples/eda.html

        population = toolbox.generate() # generate a new population of λ individuals of type ind_init from the current strategy

        eval_results = toolbox.map(toolbox.evaluate, [run]*len(population), [gen]*len(population), population)
        for ind, fit in zip(population, eval_results):
            ind.fitness.values = fit

        write_single_gen_data(run, gen, population, learning_params['analysis_dir']['data'])

        toolbox.update(population) # update the current covariance matrix strategy from the population; update the strategy with the evaluated individuals

    time_run = time.time() - time_run
    print(f"cmaES_EvoAlgorithm run n.{run} - Finished. Execution time: {time_run} seconds") # kale change save time for each run

    write_single_run_data(run, time_run, learning_params['analysis_dir'])

    # Plot data for one single run
    if learning_params['plot_analysis_bool']:
        plot_single_run_data(run, learning_params)

    return learning_params


###########################################################################
# Parallelization
###########################################################################

def worker(task):
    run, learning_params = task
    return cmaES_EvoAlgorithm(run, learning_params)

#---------------------------------------------------

def parallelize_processes(nb_runs, learning_params):
    
    # Create a queue of tasks to execute
    task_queue = [(run, learning_params.copy()) for run in range(nb_runs)]

    # Create a Pool with the number of available cores
    available_cores = cpu_count() - learning_params['with_parallelization_nb_free_cores']
    with Pool(processes=min(nb_runs, available_cores)) as pool:
        results_list = pool.map(worker, task_queue) # results_list contains the 'learning_params' of each run, respecting the ascending order from 0 to nb_runs

    return results_list[-1]


###########################################################################
# MAIN
###########################################################################

if (__name__ == "__main__"):

    # Initialization
    learning_params = get_parameters_from_json()
    learning_params = init_all_runs_analysis(learning_params)
    learning_params = set_env(learning_params)

    if learning_params['with_parallelization_bool']:
        # Launch the Evolutionary Algorithm with parallelization over runs
        learning_params = parallelize_processes(learning_params['nb_runs'], learning_params)
    else:
        # Launch the Evolutionary Algorithm sequentially
        for run in range(learning_params['nb_runs']):
            learning_params = cmaES_EvoAlgorithm(run, learning_params)

    # Save a trace of used parameters for this simulation in learning_params.json
    del learning_params['env']['eval_function'] # not JSON serializable object
    del learning_params['env']['eval_function_params']['controller']['nn_controller'] # not JSON serializable object
    with open(learning_params['analysis_dir']['root']+"/learning/learning_params.json", "w") as f:
        json.dump({k:v for k,v in learning_params.items()}, f, indent=2)

    # Plot data for all the runs
    if learning_params['plot_analysis_bool']:
        write_all_runs_data(learning_params['analysis_dir']['root']+"/learning")
        plot_all_runs_data(learning_params)