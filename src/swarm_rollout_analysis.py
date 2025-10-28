import os
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
import shutil

import argparse
import json

import pandas as pd

from learning_initializations import save_data_to_csv
from learning_analysis import write_all_runs_data, plot_all_runs_data
from environments import swarmGrid


###########################################################################
# Analysis folders creation
###########################################################################

def init_all_runs_analysis(learning_analysis_dir_root, params):

    # Create the 'analysis_dir' folder
    params['analysis_dir'] = {}
    params['analysis_dir']['root'] = learning_analysis_dir_root+"_swarm_rollout_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if os.path.exists(params['analysis_dir']['root']):
        shutil.rmtree(params['analysis_dir']['root'])
    
    os.makedirs(params['analysis_dir']['root']+"/data_all_runs", exist_ok=True)
    os.makedirs(params['analysis_dir']['root']+"/plots_all_runs", exist_ok=True)

    save_data_to_csv(params['analysis_dir']['root']+"/data_all_runs/data_evo_all_runs_time.csv", [], header = ["Run", "Time(s)", "Time(min)", "Time(h)"])
    save_data_to_csv(params['analysis_dir']['root']+"/data_all_runs/data_evo_all_runs_best_ind_per_run.csv", [], header = ["Run", "Generation", "Nb_eval", "Learning_phase", "Fitness", "Individual"])
    save_data_to_csv(params['analysis_dir']['root']+"/data_all_runs/data_evo_all_runs_best_ind_per_run_per_phase.csv", [], header = ["Run", "Generation", "Nb_eval", "Learning_phase", "Fitness", "Individual"])
    
    return params

#---------------------------------------------------

def init_one_run_analysis(run, best_ind, best_ind_run, params):

    # Create the data and plot directories tree from the 'analysis_dir' folder
    params['analysis_dir']['data'] = params['analysis_dir']['root']+ f"/run_{run:03}/best_ind_{best_ind_run:03}/data"
    params['analysis_dir']['plots'] = params['analysis_dir']['root']+ f"/run_{run:03}/best_ind_{best_ind_run:03}/plots"
    os.makedirs(params['analysis_dir']['data'], exist_ok=True)
    os.makedirs(params['analysis_dir']['plots'], exist_ok=True)

    file_path = params['analysis_dir']['root']+ f"/run_{run:03}/best_ind_{best_ind_run:03}/flag_individual.txt"
    if not os.path.exists(file_path):
        with open (file_path, 'w') as f:
            f.write(str(best_ind))

    return params

#---------------------------------------------------

def write_single_run_analysis(run, time_run, params):
    pass

#---------------------------------------------------

def plot_single_run_single_ind_data(run, best_ind_run, params):

    time_run_ind = time.time()
    print(f"swarm_analysis plots run n.{run}, best_ind_{best_ind_run} - Starting")

    time_steps = params['environment']['time_steps']
    env_dims_list = [[params['grid']['grid_nb_rows'], params['grid']['grid_nb_cols']]]
    if params['swarm_rollout_settings']['env_name'] in ['sliding_puzzle_multiEnvs', 'sliding_puzzle_multiEnvs_coordinates']:
        env_dims_list = params['swarm_rollout_settings']['sliding_puzzle_multiEnvs']['env_dims_list']

    for env_id, env_dims in enumerate(env_dims_list):
        path = params['analysis_dir']['root']+ f"/run_{run:03}/best_ind_{best_ind_run:03}/data/data_env{env_id}_flag/data_env_flag_run_{run:03}_gen_00000_eval_0000000.csv"
        if not os.path.exists(str(path)):
            continue

        # Here, it is certain that 'path' exists
        dataset = pd.read_csv(path)

        steps = dataset['Step'].unique()
        for step in range(time_steps):
            # flag_list = get_flag_list_from_dataset_step(dataset, step) # flag_list is a list of floats (1D) or a list of tuples (nD)
            flag_list = dataset.loc[(dataset.Step==step),['Flag']].values.tolist()[0][0]
            flag_list = eval(flag_list)
            fitness = dataset.loc[(dataset.Step==step),['Flags_distance']].values.tolist()[0][0]            
            deleted_pos = dataset.loc[(dataset.Step==step),['Deleted_agents_positions']].values.tolist()[0][0]
            deleted_pos = eval(deleted_pos)
            nb_moves_per_step = dataset.loc[(dataset.Step==step),['Nb_moves']].values.tolist()[0][0]

            flag_signals_list = dataset.loc[(dataset.Step==step),['Flag_signals']].values.tolist()[0][0]
            flag_signals_list = eval(flag_signals_list)

            if step in steps:
                swarmGrid.plot_flag(grid_nb_rows=env_dims[0],
                                    grid_nb_cols=env_dims[1],
                                    setup_name="sliding_puzzle_test",
                                    run=run,
                                    nb_ind=best_ind_run,
                                    gen=0,
                                    nb_eval=0,
                                    n="",
                                    step=step,
                                    flag_list=flag_list,
                                    fitness=fitness,
                                    env_id=env_id,
                                    deleted_pos=deleted_pos,
                                    nb_moves_per_step=nb_moves_per_step,
                                    analysis_dir_plots=params['analysis_dir']['root']+ f"/run_{run:03}/best_ind_{best_ind_run:03}/plots/env")

                # signals
                swarmGrid.plot_flag(grid_nb_rows=env_dims[0],
                                    grid_nb_cols=env_dims[1],
                                    setup_name="sliding_puzzle_test",
                                    run=run,
                                    nb_ind=best_ind_run,
                                    gen=0,
                                    nb_eval=0,
                                    n="",
                                    step=step,
                                    flag_list=flag_signals_list,
                                    fitness=fitness,
                                    env_id=env_id,
                                    deleted_pos=deleted_pos,
                                    nb_moves_per_step=nb_moves_per_step,
                                    analysis_dir_plots=params['analysis_dir']['root']+ f"/run_{run:03}/best_ind_{best_ind_run:03}/plots/env/signals")
            
        path_to_flag_individual_txt = f"{params['analysis_dir']['root']}/run_{run:03}/best_ind_{best_ind_run:03}/flag_individual.txt"
        with open(path_to_flag_individual_txt, "r") as f:
            ind = f.read().strip()

        swarmGrid.plot_flag_fitnesses_from_file(data_flag_file=params['analysis_dir']['root']+ f"/run_{run:03}/best_ind_{best_ind_run:03}/data/data_env{env_id}_flag/data_env_flag_run_{run:03}_gen_00000_eval_0000000.csv",
                                                setup_name="sliding_puzzle_test",
                                                time_window_start=params['environment']['time_window_start'],
                                                time_window_length= params['environment']['time_window_end'] - params['environment']['time_window_start'] + 1,
                                                run=run,
                                                nb_ind=best_ind_run,
                                                ind=ind,
                                                n="",
                                                gen=0,
                                                nb_eval=0,
                                                switch_step=None,
                                                analysis_dir_plots=params['analysis_dir']['root']+ f"/run_{run:03}/best_ind_{best_ind_run:03}/plots/env")
    

    time_run = time.time() - time_run_ind
    print(f"swarm_analysis plots run n.{run}, best_ind_{best_ind_run} - Completed. Execution time: {time_run} seconds")


###########################################################################
# Parallelization
###########################################################################

def worker(task):

    run, best_ind_run, params = task
    return plot_single_run_single_ind_data(run=run, best_ind_run=best_ind_run, params=params)

#---------------------------------------------------

def parallelize_processes(task_queue, with_parallelization_nb_free_cores):

    # Create a Pool with the number of available cores
    available_cores = cpu_count() - with_parallelization_nb_free_cores
    with Pool(processes=available_cores) as pool:
        pool.map(worker, task_queue)
        pool.close() # no more tasks will be submitted to the pool
        pool.join() # wait for all processes to complete


###########################################################################
# MAIN
###########################################################################

if (__name__ == "__main__"):

    print("\nswarm_rollout_analysis running...")

    # Get parameters from the bash launcher
    parser = argparse.ArgumentParser()
    parser.add_argument("--swarm_rollout_analysis_dir", default="", type=str)
    parser.add_argument("--with_parallelization_bool", default=False, type=lambda x:x=="True")
    parser.add_argument("--with_parallelization_nb_free_cores", default=0, type=int)
    parser.add_argument("--plot_with_animation_bool", default=False, type=lambda x:x=="True")
    args = parser.parse_args()

    # Get parameters from the swarm rollout simulation
    with open(args.swarm_rollout_analysis_dir+"/swarm_rollout_params.json", "r") as f:
        params = json.load(f)
    
    params['with_parallelization_bool'] = args.with_parallelization_bool
    params['with_parallelization_nb_free_cores'] = args.with_parallelization_nb_free_cores
    params['plot_with_animation_bool'] = args.plot_with_animation_bool

    # Launch plots
    if params['with_parallelization_bool']:
        task_queue = [] # create a queue of tasks to execute
        for run in range(params['swarm_rollout_settings']['nb_runs']):
            for best_ind_run in params['best_ind_per_run_dict']:
                task_queue.append((int(run), int(best_ind_run), params.copy()))
        swarm_params = parallelize_processes(task_queue, params['with_parallelization_nb_free_cores'])
    else:
        for run in range(params['swarm_rollout_settings']['nb_runs']):
            for best_ind_run in params['best_ind_per_run_dict']:
                plot_single_run_single_ind_data(run=int(run), best_ind_run=int(best_ind_run), params=params)

    # write_all_runs_data(args.swarm_rollout_analysis_dir)
    # plot_all_runs_data(params)