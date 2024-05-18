import time
from multiprocessing import Pool, cpu_count

import numpy as np

import json

from swarm_environments import init_swarmGrid_env
from swarm_initializations import *
from swarm_analysis import *

sep = "\n################################################\n"




def swarm_simulation(run, best_ind, best_ind_run, swarm_params):
    print(f"swarm_simulation run n.{run}, best_ind_{best_ind_run} [{best_ind[0]}, ...] - Starting")

    # Initializations
    time_run = time.time()
    swarm_params = init_one_run_analysis(run, best_ind, best_ind_run, swarm_params)

    env = init_swarmGrid_env(grid_nb_rows=swarm_params['grid_nb_rows'],
                             grid_nb_cols=swarm_params['grid_nb_cols'],
                             init_cell_state_value=swarm_params['init_cell_state_value'],
                             nn_controller=swarm_params['controller'],
                             flag_target=swarm_params['flag_target'],
                             agent_controller_weights=best_ind)

    # setup_ind_consistency: initialization of the environment and test of the current individual 'nb_repetitions' times
    if swarm_params['setup_ind_consistency_bool']:
        setup_name = "setup_ind_consistency"
        env.setup_ind_consistency(run=run,
                                  setup_name=setup_name,
                                  nb_repetitions=swarm_params['nb_repetitions'],
                                  time_steps=swarm_params['time_steps'],
                                  analysis_dir=swarm_params['analysis_dir'])



















    # cas 1: introduire du bruit continu de differente entité
    # param: entités de bruit souhaités
    # initialize env: init grid? ou nouvelle instance
    # quand on calcule les entrées du reseau in compute_agent_state(self, agent), add bruit selon l entite demandée en parametre
    # repeter le procedé N fois, un plot par N avec barre au 'step de changement regime' et un plot complexive box plot, du coup data "Run, step, fitness"

    # noise_ticks = swarm_params['noise_ticks']
    # if noise_ticks:
    #     for n in range(swarm_params['nb_repetitions']):
    #         for tick in noise_ticks:
    #             env.setup_noise1(run, n, tick, time_steps, switch_step, best_ind_run, best_ind, swarm_params)

    # # cas 2: introduire du bruit continu de differente entité
    # # pareil, mais avec noise type 2

    # # noise_ticks = swarm_params['noise_ticks']
    # # if noise_ticks:
    # #     for n in range(swarm_params['nb_repetitions']):
    # #         for tick in noise_ticks:
    # #             env.setup_noise2(run, n, tick, time_steps, switch_step, best_ind_run, best_ind, swarm_params)
           
    # # cas 2: eliminer n agents
    # # params: liste de nombre N de agents à eliminer, sans retirage, choisi au hazard
    # # initialize env: init grid? ou nouvelle instance
    # # on elimine 2 agents des listes, on refrech les voisins
    # # repeter le procedé N fois, un plot par N avec barre au 'step de changement regime' et un plot complexive box plot, du coup data "Run, step, fitness"
    # # les agents enleves seront reinseres

    # deletion_ticks = swarm_params['deletion_ticks']
    # if deletion_ticks:
    #     for n in range(swarm_params['nb_repetitions']):

    #         # deletion_ticks = random.sample(range(env.size-1), swarm_params['nb_agents_to_delete']) # random.sample choose elements without replacement (don't allow duplicates)

    #         # if swarm_params['nb_agents_to_delete'] >= env.grid_size:
    #         #     print("gneee conlacca")
    #         #     exit()

    #         deleted_map_pos_agent = {}
    #         for tick in deletion_ticks:
    #             deleted_map_pos_agent = env.setup_deletion(run, n, tick, time_steps, switch_step, best_ind_run, best_ind, deleted_map_pos_agent, swarm_params)
            
    #         env.restore_deleted_agents(deleted_map_pos_agent=deleted_map_pos_agent)












    time_run = time.time() - time_run
    print(f"swarm_simulation run n.{run}, best_ind_{best_ind_run} [{best_ind[0]}, ...] - Execution time:", time_run, "seconds") # kale change save time for each run

    return swarm_params


###########################################################################
# Parallelization
###########################################################################

def worker(task):
    run, best_ind, best_ind_run, swarm_params = task
    # print("launching with ", run, best_ind, swarm_params, "\n")
    return swarm_simulation(run=run, best_ind=best_ind, best_ind_run=best_ind_run, swarm_params=swarm_params)

#---------------------------------------------------

def parallelize_processes(nb_runs, best_ind_per_run_dict, swarm_params):
    
    # Create a queue of tasks to execute
    task_queue = []
    for run in range(nb_runs):
        for best_ind_run in best_ind_per_run_dict:
            task_queue.append((run, best_ind_per_run_dict[best_ind_run], best_ind_run, swarm_params.copy()))

    # Create a Pool with the number of available cores
    available_cores = cpu_count() - swarm_params['with_parallelization_nb_free_cores']
    with Pool(processes=min(nb_runs, available_cores)) as pool:
        results_list = pool.map(worker, task_queue) # results_list contains the 'swarm_params' of each run, respecting the ascending order from 0 to nb_runs

    return results_list[-1]


###########################################################################
# MAIN
###########################################################################

if (__name__ == "__main__"):

    # Get parameters from the bash launcher
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_analysis_dir", default="", type=str)
    args = parser.parse_args()

    # Get parameters from config files
    with open(args.learning_analysis_dir+"/learning_params.json", "r") as f:
        learning_params = json.load(f)
    
    with open(os.getcwd()+"/swarm_params.json", "r") as f:
        swarm_params = json.load(f)

    # Initializations
    check_params_validity(params=swarm_params)
    swarm_params = init_all_runs_analysis(learning_analysis_dir_root=learning_params['analysis_dir']['root'], params=swarm_params)
    swarm_params = copy_params_from_learning(learning_params=learning_params, swarm_params=swarm_params)
    best_ind_per_run_dict = get_best_ind_per_run_dict(dataset_path=learning_params['analysis_dir']['root']+"/data_all_runs/data_evo_all_runs_best_ind_per_run.csv")

    # Launch the Swarm simulation with parallelization over runs and individuals OR sequentially
    if swarm_params['with_parallelization_bool']:
        swarm_params = parallelize_processes(nb_runs=swarm_params['nb_runs'], best_ind_per_run_dict=best_ind_per_run_dict, swarm_params=swarm_params)
    else:
        for run in range(swarm_params['nb_runs']):
            for best_ind_run in best_ind_per_run_dict:
                swarm_params = swarm_simulation(run=run, best_ind=best_ind_per_run_dict[best_ind_run], best_ind_run=best_ind_run, swarm_params=swarm_params.copy())

    # Save a trace of used parameters for this simulation in swarm_params.json
    swarm_params['best_ind_ever'] = np.array(swarm_params['best_ind_ever']).tolist()
    swarm_params['flag_target'] = np.array(swarm_params['flag_target']).tolist()
    del swarm_params['controller'] # not JSON serializable object
    with open(swarm_params['analysis_dir']['root']+"/swarm_params.json", "w") as f:
        json.dump({k:v for k,v in swarm_params.items()}, f, indent=2)

    # Plots: this line allows the swarm_launch.sh script to plot figures
    print(swarm_params['analysis_dir']['root'])