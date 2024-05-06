import os
import shutil
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import imageio.v2 as iio

import argparse

import csv
import json





###########################################################################
# Analysis folders creation
###########################################################################

def init_all_runs_analysis(learning_analysis_dir_root, params):

    # Create the 'analysis_dir' folder
    params['analysis_dir'] = {}
    params['analysis_dir']['root'] = learning_analysis_dir_root.replace("/learning", "/swarm")

    if os.path.exists(params['analysis_dir']['root']):
        shutil.rmtree(params['analysis_dir']['root'])
    
    os.makedirs(params['analysis_dir']['root']+"/data_all_runs", exist_ok=True)
    os.makedirs(params['analysis_dir']['root']+"/plots_all_runs", exist_ok=True)

    save_data_to_csv(params['analysis_dir']['root']+"/data_all_runs/data_all_runs_time.csv", [], header = ["Run", "Time"])
    
    return params

#---------------------------------------------------

def init_one_run_analysis(run, params):

    # Create the data and plot directories tree from the 'analysis_dir' folder
    params['analysis_dir']['data'] = params['analysis_dir']['root']+"/run_"+str(run)+"/data"
    params['analysis_dir']['plots'] = params['analysis_dir']['root']+"/run_"+str(run)+"/plots"
    os.makedirs(params['analysis_dir']['data'], exist_ok=True)
    os.makedirs(params['analysis_dir']['plots'], exist_ok=True)

    return params


###########################################################################
# Save data functions
###########################################################################

def save_data_to_csv(fichier_name, data, header=None):
    f = open(fichier_name, 'a', newline='')
    writer = csv.writer(f)

    if header:
        writer.writerow(header)

    if data:
        writer.writerows(data)

    f.close()
 
#---------------------------------------------------
    
def write_single_run_data(env, run, time_run, time_steps, flags, analysis_dir):
    pass


# ###########################################################################
# Plot data functions
# ###########################################################################
                
def plot_single_run_data(run, params):
    pass


###########################################################################
# Parallelization
###########################################################################

def worker(task):

    plot_flag_func, plot_flag_fitnesses_func, run, gen, ind, steps, save_dir, params = task
    plot_flag_func(env_eval_function_params=params['env']['eval_function_params'], data_flag_file=params['analysis_dir']['root']+"/run_"+str(run)+"/data/data_env_flag/data_env_flag_run_"+str(run)+"_gen_"+str(gen)+".csv", run=run, gen=gen, ind=ind, steps=steps, analysis_dir_plots=save_dir)
    plot_flag_fitnesses_func(env_eval_function_params=params['env']['eval_function_params'], data_flag_file=params['analysis_dir']['root']+"/run_"+str(run)+"/data/data_env_flag/data_env_flag_run_"+str(run)+"_gen_"+str(gen)+".csv", run=run, gen=gen, ind=ind, analysis_dir_plots=save_dir)

#---------------------------------------------------

def parallelize_processes(task_queue, params):

    # Create a Pool with the number of available cores
    available_cores = cpu_count() - params['with_parallelization_nb_free_cores']
    with Pool(processes=available_cores) as pool:
        pool.map(worker, task_queue)
        pool.close() # no more tasks will be submitted to the pool
        pool.join() # wait for all processes to complete

    
###########################################################################
# MAIN
###########################################################################

if (__name__ == "__main__"):

    # Get parameters from the bash launcher
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_analysis_dir", default="", type=str)
    parser.add_argument("--with_parallelization_bool", default=False, type=lambda x:x=="True")
    parser.add_argument("--with_parallelization_nb_free_cores", default=0, type=int)
    parser.add_argument("--plot_with_animation_bool", default=False, type=lambda x:x=="True")
    args = parser.parse_args()

    # Get parameters from the learning simulation
    with open(args.learning_analysis_dir+"/learning_params.json", "r") as f:
        params = json.load(f)
    
    params['with_parallelization_bool'] = args.with_parallelization_bool
    params['with_parallelization_nb_free_cores'] = args.with_parallelization_nb_free_cores
    params['plot_with_animation_bool'] = args.plot_with_animation_bool

    # Launch plots
    for run in range(params['nb_runs']):
        plot_single_run_data(run, params)