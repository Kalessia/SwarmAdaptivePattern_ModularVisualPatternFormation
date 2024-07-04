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
    best_fit = np.inf

    # Main evolutionary loop
    for gen in range(learning_params['nb_generations']):

        # DEAP CMAES
        # To control the stopping criteria: while not any(conditions.values())
        # https://deap.readthedocs.io/en/master/examples/bipop_cmaes.html
        # http://deap.gel.ulaval.ca/doc/default/examples/eda.html

        population = toolbox.generate() # generate a new population of λ individuals of type ind_init from the current strategy

        eval_results = toolbox.map(toolbox.evaluate, [run]*len(population), [gen]*len(population), [best_fit]*len(population), population)
        for ind, fit in zip(population, eval_results):
            ind.fitness.values = fit

            # In the 'flag_automata' env, save the best flags only (memory optimization)
            if fit[0] < best_fit:
                best_fit = fit[0]

        write_single_gen_data(run, gen, population, learning_params['analysis_dir']['data'])

        toolbox.update(population) # update the current covariance matrix strategy from the population; update the strategy with the evaluated individuals

    time_run = time.time() - time_run
    print(f"cmaES_EvoAlgorithm run n.{run} - Completed. Execution time: {time_run} seconds")

    write_single_run_data(run, time_run, learning_params['analysis_dir'])

    return learning_params


###########################################################################
# Parallelization
###########################################################################

def worker(task):
    run, learning_params = task
    time.sleep(random.random()) # sleep between 0.0 and 1.0 seconds, avoid identical initial conditions
    return cmaES_EvoAlgorithm(run, learning_params)

#---------------------------------------------------

def parallelize_processes(nb_runs, learning_params):
    
    # Create a queue of tasks to execute
    task_queue = [(run, learning_params.copy()) for run in range(nb_runs)]

    # Create a Pool with the number of available cores
    available_cores = cpu_count() - learning_params['with_parallelization_nb_free_cores']
    with Pool(processes=available_cores) as pool:
        results_list = pool.map(worker, task_queue) # results_list contains the 'learning_params' of each run, respecting the ascending order from 0 to nb_runs
        pool.close() # no more tasks will be submitted to the pool
        pool.join() # wait for all processes to complete

    return results_list[0]


###########################################################################
# MAIN
###########################################################################

if (__name__ == "__main__"):

    # Get parameters from config files
    with open(os.getcwd() +"/learning_params.json", "r") as f:
        learning_params = json.load(f)

    # Initializations
    learning_params = init_all_runs_analysis(learning_params)
    learning_params = set_env(learning_params)

    # Launch the Evolutionary Algorithm with parallelization over runs OR sequentially
    if learning_params['with_parallelization_bool']:
        learning_params = parallelize_processes(learning_params['nb_runs'], learning_params)
    else:
        for run in range(learning_params['nb_runs']):
            learning_params = cmaES_EvoAlgorithm(run, learning_params)

    # Save a trace of used parameters for this simulation in learning_params.json
    del learning_params['env']['eval_function'] # not JSON serializable object
    del learning_params['env']['eval_function_params']['controller'] # not JSON serializable object
    with open(learning_params['analysis_dir']['root']+"/learning_params.json", "w") as f:
        json.dump({k:v for k,v in learning_params.items()}, f, indent=2)

    # Plots: this line allows the learning_launch.sh script to plot figures
    print(learning_params['analysis_dir']['root'])