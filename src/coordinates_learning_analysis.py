import os
import time
from datetime import datetime
import shutil

import argparse
import json

from learning_initializations import save_data_to_csv
from learning_analysis import worker, parallelize_processes, write_all_runs_data, plot_all_runs_data, write_best_inds_ever_and_best_ind_per_run
from environments import swarmGrid


###########################################################################
# Analysis folders creation
###########################################################################

def init_all_runs_analysis(learning_analysis_dir_root, params):

    # Create the 'analysis_dir' folder
    params['analysis_dir'] = {}
    params['analysis_dir']['root'] = learning_analysis_dir_root.replace("/learning", "/learning_coordinates")+"_"+params['grid']['flag_pattern']+"_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if os.path.exists(params['analysis_dir']['root']):
        shutil.rmtree(params['analysis_dir']['root'])
    
    os.makedirs(params['analysis_dir']['root']+"/data_all_runs", exist_ok=True)
    os.makedirs(params['analysis_dir']['root']+"/plots_all_runs", exist_ok=True)

    save_data_to_csv(params['analysis_dir']['root']+"/data_all_runs/data_evo_all_runs_time.csv", [], header = ["Run", "Time(s)", "Time(min)", "Time(h)"])
    save_data_to_csv(params['analysis_dir']['root']+"/data_all_runs/data_evo_all_runs_best_ind_per_run.csv", [], header = ["Run", "Generation", "Nb_eval", "Learning_phase", "Fitness", "Individual"])
    save_data_to_csv(params['analysis_dir']['root']+"/data_all_runs/data_evo_all_runs_best_ind_per_run_per_phase.csv", [], header = ["Run", "Generation", "Nb_eval", "Learning_phase", "Fitness", "Individual"])
    
    return params

    
###########################################################################
# MAIN
###########################################################################

if (__name__ == "__main__"):

    print("\ncoordinates_learning_analysis running...")

    # Get parameters from the bash launcher
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinates_learning_analysis_dir", default="", type=str)
    parser.add_argument("--with_parallelization_bool", default=False, type=lambda x:x=="True")
    parser.add_argument("--with_parallelization_nb_free_cores", default=0, type=int)
    parser.add_argument("--plot_with_animation_bool", default=False, type=lambda x:x=="True")
    args = parser.parse_args()

    # Get parameters from the coordinates learning simulation
    with open(args.coordinates_learning_analysis_dir+"/coordinates_learning_params.json", "r") as f:
        params = json.load(f)
    
    # Save best ever individual data for this single run
    # analysis_dir = params['analysis_dir']
    # for run in range(11):
    #     dataset_path = analysis_dir['data'].replace("run_000", f"run_{run:03}")+ f"/data_evo_run_{run:03}_best_inds_per_gen.csv"
    #     save_best_inds_ever_filename = analysis_dir['data'].replace("run_000", f"run_{run:03}")+ f"/data_evo_run_{run:03}_best_inds_ever.csv"
    #     save_best_ind_per_run_filename = analysis_dir['root']+"/data_all_runs/data_evo_all_runs_best_ind_per_run.csv"
    #     save_best_ind_per_run_per_phase_filename = analysis_dir['root']+"/data_all_runs/data_evo_all_runs_best_ind_per_run_per_phase.csv"
    #     write_best_inds_ever_and_best_ind_per_run(dataset_path=dataset_path, switch_gen=None, save_best_inds_ever_filename=save_best_inds_ever_filename, save_best_ind_per_run_filename=save_best_ind_per_run_filename, save_best_ind_per_run_per_phase_filename=save_best_ind_per_run_per_phase_filename)

    params['with_parallelization_bool'] = args.with_parallelization_bool
    params['with_parallelization_nb_free_cores'] = args.with_parallelization_nb_free_cores
    params['plot_with_animation_bool'] = args.plot_with_animation_bool

    # Launch plots
    if params['with_parallelization_bool']:
        task_queue = [] # create a queue of tasks to execute
        for run in range(params['evolutionary_settings']['nb_runs']):
            task_queue.append((run, params.copy()))
        parallelize_processes(task_queue, params['with_parallelization_nb_free_cores'])
    else:
        for run in range(params['evolutionary_settings']['nb_runs']):
            plot_single_run_data(run, params)

    write_all_runs_data(args.coordinates_learning_analysis_dir)
    # write_all_runs_data(args.coordinates_learning_analysis_dir+"/learning_coordinates") qual'é giusto?
    plot_all_runs_data(params)