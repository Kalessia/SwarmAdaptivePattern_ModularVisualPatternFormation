import time
from multiprocessing import Pool, cpu_count

import numpy as np
import itertools

import json

from environments import init_swarmGrid_env
from swarm_initializations import *
from swarm_analysis import *

sep = "\n################################################\n"


###########################################################################
# Swarm simulation
###########################################################################

def swarm_simulation(run, best_ind, best_ind_run, best_inds_per_run_per_phase, swarm_params):
    print(f"swarm_simulation run n.{run}, best_ind_{best_ind_run} [{best_ind[0]}, ...] - Starting")

    # print(best_ind[0])

    # Initializations
    time_run = time.time()
    swarm_params = init_one_run_analysis(run, best_ind, best_ind_run, swarm_params)
    setups = []

    env = init_swarmGrid_env(grid_nb_rows=swarm_params['grid_nb_rows'],
                             grid_nb_cols=swarm_params['grid_nb_cols'],
                             learning_modes=swarm_params['learning_modes'],
                             flags_distance_mode=None,
                             learning_with_noise_std=swarm_params['learning_with_noise_std'],
                             flag_pattern=None,
                             flag_target=swarm_params['flag_target'],
                             init_cell_state_value=swarm_params['init_cell_state_value'],
                             nn_controller=swarm_params['controller'],
                             agent_controller_weights=best_ind,
                             verbose_debug_bool=swarm_params['verbose_debug'],
                             analysis_dir=swarm_params['analysis_dir'])

    # setup_ind_consistency
    if swarm_params['setup_ind_consistency']['setup_ind_consistency_bool']:
        setups += swarm_params['setup_ind_consistency']['setup_ind_consistency_options']
        for setup_name in swarm_params['setup_ind_consistency']['setup_ind_consistency_options']:
            env.setup_ind_consistency(run=run,
                                    setup_name=setup_name,
                                    nb_repetitions=swarm_params['nb_repetitions'],
                                    time_steps=swarm_params['time_steps'],
                                    switch_step=swarm_params['switch_step'],
                                    switch_step_with_reset_env_bool=swarm_params['switch_step_with_reset_env_bool'],
                                    analysis_dir=swarm_params['analysis_dir'])
    
    # setup_noise
    if swarm_params['setup_noise']['setup_noise_bool']:
        setup_name = "setup_noise"
        setups += [setup_name+"_"+str(tick) for tick in swarm_params['setup_noise']['setup_noise_std_ticks']]
        env.setup_noise(run=run,
                        setup_name=setup_name,
                        nb_repetitions=swarm_params['nb_repetitions'],
                        setup_noise_std_ticks=swarm_params['setup_noise']['setup_noise_std_ticks'],
                        time_steps=swarm_params['time_steps'],
                        switch_step=swarm_params['switch_step'],
                        switch_step_with_reset_env_bool=swarm_params['switch_step_with_reset_env_bool'],
                        switch_step_with_random_async_update_bool=swarm_params['switch_step_with_random_async_update_bool'],
                        analysis_dir=swarm_params['analysis_dir'])
        
    # setup_permutation
    if swarm_params['setup_permutation']['setup_permutation_bool']:
        setup_name = "setup_permutation"
        permutation_ticks = swarm_params['setup_permutation']['permutation_ticks']
        setups += [setup_name+"_"+str(tick) for tick in permutation_ticks]
        env.setup_permutation(run=run,
                              setup_name=setup_name,
                              nb_repetitions=swarm_params['nb_repetitions'],
                              setup_permutation_ticks=permutation_ticks,
                              time_steps=swarm_params['time_steps'],
                              switch_step=swarm_params['switch_step'],
                              switch_step_with_reset_env_bool=swarm_params['switch_step_with_reset_env_bool'],
                              switch_step_with_random_async_update_bool=swarm_params['switch_step_with_random_async_update_bool'],
                              analysis_dir=swarm_params['analysis_dir'])

    # setup_deletion
    if swarm_params['setup_deletion']['setup_deletion_bool']:
        setup_name = "setup_deletion"
        deletion_ticks = swarm_params['setup_deletion']['deletion_ticks']
        setups += [setup_name+"_"+str(tick) for tick in deletion_ticks]
        env.setup_deletion(run=run,
                           setup_name=setup_name,
                           nb_repetitions=swarm_params['nb_repetitions'],
                           deletion_ticks=deletion_ticks,
                           time_steps=swarm_params['time_steps'],
                           switch_step=swarm_params['switch_step'],
                           switch_step_with_reset_env_bool=swarm_params['switch_step_with_reset_env_bool'],
                           switch_step_with_random_async_update_bool=swarm_params['switch_step_with_random_async_update_bool'],
                           analysis_dir=swarm_params['analysis_dir'])

    # setup_sliding_puzzle
    if swarm_params['setup_sliding_puzzle']['setup_sliding_puzzle_bool']:
        setup_name = "setup_sliding_puzzle"
        sliding_puzzle_ticks = swarm_params['setup_sliding_puzzle']['deletion_ticks']
        setups += [ f"{setup_name}_deletions{tick}_fluidity{proba_move}" for tick, proba_move in list(itertools.product(sliding_puzzle_ticks, swarm_params['setup_sliding_puzzle']['setup_sliding_puzzle_probas_move']))]
       
        env.setup_sliding_puzzle(run=run,
                                 setup_name=setup_name,
                                 nb_repetitions=swarm_params['nb_repetitions'],
                                 sliding_puzzle_ticks=sliding_puzzle_ticks,
                                 sliding_puzzle_probas_move=swarm_params['setup_sliding_puzzle']['setup_sliding_puzzle_probas_move'],
                                 time_steps=swarm_params['time_steps'],
                                 time_window_start=swarm_params['time_window_start'],
                                 time_window_end=swarm_params['time_window_end'],
                                 analysis_dir=swarm_params['analysis_dir'])

    # setup_sliding_puzzle_phase1_VS_phase2
    if swarm_params['setup_sliding_puzzle_phase1_VS_phase2']['setup_sliding_puzzle_phase1_VS_phase2_bool'] and swarm_params['setup_sliding_puzzle_phase1_VS_phase2']['learning_bool']:
        setup_name = "setup_sliding_puzzle_phase1_VS_phase2"
        sliding_puzzle_learning_ticks = [int(tick_unit) for tick_unit in swarm_params['setup_sliding_puzzle_phase1_VS_phase2']['learning_ticks_units']]
        setups += [f"{setup_name}_{tick}_phase{phase}" for tick, phase in list(itertools.product(sliding_puzzle_learning_ticks, [1,2]))]
        swarmGrid.setup_sliding_puzzle_phase1_VS_phase2(run=run,
                                                        setup_name=setup_name,
                                                        nb_repetitions=swarm_params['nb_repetitions'],
                                                        sliding_puzzle_learning_best_inds_per_phase=best_inds_per_run_per_phase,
                                                        sliding_puzzle_learning_ticks=sliding_puzzle_learning_ticks,
                                                        sliding_puzzle_learning_proba_move=swarm_params['setup_sliding_puzzle_phase1_VS_phase2']['learning_proba_move'],
                                                        grid_nb_rows=swarm_params['grid_nb_rows'],
                                                        grid_nb_cols=swarm_params['grid_nb_cols'],
                                                        learning_modes=swarm_params['learning_modes'],
                                                        learning_with_noise_std=swarm_params['learning_with_noise_std'],
                                                        flag_pattern=swarm_params['flag_pattern'],
                                                        init_cell_state_value=swarm_params['init_cell_state_value'],
                                                        nn_controller=swarm_params['controller'],
                                                        time_steps=swarm_params['time_steps'],
                                                        analysis_dir=swarm_params['analysis_dir'])
        
    # setup scalability
    if swarm_params['setup_scalability']['setup_scalability_bool']:
        setup_name = "setup_scalability"
        rows = swarm_params['grid_nb_rows']
        cols = swarm_params['grid_nb_cols']
        scalability_ticks = [[rows, cols], [rows*2, cols*2], [rows*2, cols], [rows, cols*2], [int(rows/2), int(cols/2)], [int(rows/2), cols], [rows, int(cols/2)]]
        setups += [setup_name+"_"+str(tick[0])+"x"+str(tick[1]) for tick in scalability_ticks]
        swarmGrid.setup_scalability(run=run,
                                    setup_name=setup_name,
                                    nb_repetitions=swarm_params['nb_repetitions'],
                                    scalability_ticks=scalability_ticks,
                                    learning_modes=swarm_params['learning_modes'],
                                    learning_with_noise_std=swarm_params['learning_with_noise_std'],
                                    flag_pattern=swarm_params['flag_pattern'],
                                    init_cell_state_value=swarm_params['init_cell_state_value'],
                                    nn_controller=swarm_params['controller'],
                                    agent_controller_weights=best_ind,
                                    time_steps=swarm_params['time_steps'],
                                    analysis_dir=swarm_params['analysis_dir'])


    swarm_params['setups'] = setups

    time_run = time.time() - time_run
    print(f"swarm_simulation run n.{run}, best_ind_{best_ind_run} [{best_ind[0]}, ...] - Execution time:", time_run, "seconds") # kale change save time for each run

    return swarm_params


###########################################################################
# Parallelization
###########################################################################

def worker(task):
    run, best_ind, best_ind_run, best_inds_per_run_per_phase, swarm_params = task
    return swarm_simulation(run=run, best_ind=best_ind, best_ind_run=best_ind_run, best_inds_per_run_per_phase=best_inds_per_run_per_phase, swarm_params=swarm_params)

#---------------------------------------------------

def parallelize_processes(nb_runs, best_ind_per_run_dict, best_ind_per_run_per_phase_dict, swarm_params):
    
    # Create a queue of tasks to execute
    task_queue = []
    for run in range(nb_runs):
        for best_ind_run in best_ind_per_run_dict:
            task_queue.append((run, best_ind_per_run_dict[best_ind_run], best_ind_run, best_ind_per_run_per_phase_dict[best_ind_run], swarm_params.copy()))

    # Create a Pool with the number of available cores
    available_cores = cpu_count() - swarm_params['with_parallelization_nb_free_cores']
    with Pool(processes=available_cores) as pool:
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
    check_params_validity(grid_size=learning_params['grid']['grid_nb_rows']*learning_params['grid']['grid_nb_cols'], params=swarm_params)
    swarm_params = init_all_runs_analysis(learning_analysis_dir_root=learning_params['analysis_dir']['root'], params=swarm_params)
    swarm_params = copy_params_from_learning(learning_params=learning_params, swarm_params=swarm_params)
    best_ind_per_run_dict = get_best_ind_per_run_dict(dataset_path=learning_params['analysis_dir']['root']+"/data_all_runs/data_evo_all_runs_best_ind_per_run_per_phase.csv")
    best_ind_per_run_per_phase_dict = get_best_ind_per_run_per_phase_dict(dataset_path=learning_params['analysis_dir']['root']+"/data_all_runs/data_evo_all_runs_best_ind_per_run_per_phase.csv")

    # Launch the Swarm simulation with parallelization over runs and individuals OR sequentially
    if swarm_params['with_parallelization_bool']:
        swarm_params = parallelize_processes(nb_runs=swarm_params['nb_runs'], best_ind_per_run_dict=best_ind_per_run_dict, best_ind_per_run_per_phase_dict=best_ind_per_run_per_phase_dict, swarm_params=swarm_params)
    else:
        for run in range(swarm_params['nb_runs']):
            for best_ind_run in best_ind_per_run_dict:
                swarm_params = swarm_simulation(run=run, best_ind=best_ind_per_run_dict[best_ind_run], best_ind_run=best_ind_run, best_inds_per_run_per_phase=best_ind_per_run_per_phase_dict[best_ind_run], swarm_params=swarm_params.copy())

    # Save a trace of used parameters for this simulation in swarm_params.json
    swarm_params['best_ind_ever'] = np.array(swarm_params['best_ind_ever']).tolist()
    swarm_params['flag_target'] = np.array(swarm_params['flag_target']).tolist()
    del swarm_params['controller'] # not JSON serializable object
    with open(swarm_params['analysis_dir']['root']+"/swarm_params.json", "w") as f:
        json.dump({k:v for k,v in swarm_params.items()}, f, indent=2)

    # Plots: this line allows the swarm_launch.sh script to plot figures
    print(swarm_params['analysis_dir']['root'])