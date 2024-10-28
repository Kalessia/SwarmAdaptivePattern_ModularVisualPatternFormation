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
    toolbox, strategy = init_toolbox(learning_params)
    best_fit = np.inf
    best_pop = None
    best_covariance_matrix = None
    gen = -1
    nb_eval = 1
    switch_gen = None
    sliding_puzzle_nb_deletions = None
    sliding_puzzle_proba_move = None
    executed_once_bool = False

    # Sliding_puzzle incremental learning parameters
    if learning_params['evolutionary_settings']['env_name'] == "sliding_puzzle_incremental":
        sliding_puzzle_incremental_nb_deletions_ticks = learning_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_ticks']
        sliding_puzzle_nb_deletions = sliding_puzzle_incremental_nb_deletions_ticks[0]
        sliding_puzzle_proba_move = learning_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_proba_move']


    # Main evolutionary loop
    while(nb_eval < learning_params['evolutionary_settings']['nb_evals']): # while the max budjet of allowed evaluations is not reached

        gen += 1 # each while iteration correspond to a new evolutionary generation, and 1 gen = pop_size evals. 
        population = toolbox.generate() # generate a new population of λ individuals of type ind_init from the current strategy
        pop_size = len(population)

        # In case of sliding_puzzle incremental learning, we switch from the 1st to the 2nd setup
        # At this point, nb_eval is the last eval of the previous generation. The 'switch_gen' is the generation following the one where switch_eval actually occured
        if learning_params['evolutionary_settings']['env_name'] == "sliding_puzzle_incremental" and not(executed_once_bool) \
        and (nb_eval >= learning_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_switch_eval'] \
        or ((learning_params['evolutionary_settings']['nb_evals'] - nb_eval) < pop_size)):
            learning_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_switch_eval'] = nb_eval # switch_eval is the 1st evaluation of this new generation
            switch_gen = gen
            sliding_puzzle_nb_deletions = sliding_puzzle_incremental_nb_deletions_ticks[1]
            best_fit = np.inf # reset best_fit to save best_individuals data for the 2nd setup starting at this generation
            population = best_pop if best_pop is not None else population # NB: the cmaes covariance matrix has changed from this older best population
            reset_covariance_matrix = best_covariance_matrix if best_covariance_matrix is not None else np.copy(strategy.C)
            strategy.C = np.copy(reset_covariance_matrix)
            executed_once_bool = True

        nb_evals = list(range(nb_eval, nb_eval+pop_size))
        nb_eval += pop_size # important: this line has to be executed after the 'incremental' condition above, to catch correctly nb_eval>switch_eval in the current generation et not in the following one


        # DEAP CMAES
        # To control the stopping criteria: while not any(conditions.values())
        # https://deap.readthedocs.io/en/master/examples/bipop_cmaes.html
        # http://deap.gel.ulaval.ca/doc/default/examples/eda.html

        eval_results = toolbox.map(toolbox.evaluate, [run]*pop_size, [gen]*pop_size, nb_evals, list(range(pop_size)), [best_fit]*pop_size, population, [sliding_puzzle_nb_deletions]*pop_size, [sliding_puzzle_proba_move]*pop_size)
        for ind, fit in zip(population, eval_results):
            ind.fitness.values = fit

            # In the 'flag_automata' env (see set_env in learning_initializations.py), save the best flags only (memory optimization)
            if fit[0] < best_fit:
                best_fit = fit[0]
                best_pop = population
                best_covariance_matrix = np.copy(strategy.C)

        write_single_gen_data(run, gen, nb_evals, population, learning_params['analysis_dir']['data'])

        toolbox.update(population) # update the current covariance matrix strategy from the population; update the strategy with the evaluated individuals


    time_run = time.time() - time_run
    print(f"cmaES_EvoAlgorithm run n.{run} - Completed. Execution time: {time_run} seconds")

    write_single_run_data(run, switch_gen, time_run, learning_params['analysis_dir'])

    return learning_params


###########################################################################
# Parallelization
###########################################################################

def worker(task):
    run, learning_params = task
    time.sleep(3*random.random()) # sleep between 0.0 and 3.0 seconds, avoid identical initial conditions and genomes
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
        learning_params = parallelize_processes(learning_params['evolutionary_settings']['nb_runs'], learning_params)
    else:
        for run in range(learning_params['evolutionary_settings']['nb_runs']):
            learning_params = cmaES_EvoAlgorithm(run, learning_params)

    # Save a trace of used parameters for this simulation in learning_params.json
    del learning_params['env']['eval_function'] # not JSON serializable object
    del learning_params['env']['eval_function_params']['controller'] # not JSON serializable object
    with open(learning_params['analysis_dir']['root']+"/learning_params.json", "w") as f:
        json.dump({k:v for k,v in learning_params.items()}, f, indent=2)

    # Plots: this line allows the learning_launch.sh script to plot figures
    print(learning_params['analysis_dir']['root'])