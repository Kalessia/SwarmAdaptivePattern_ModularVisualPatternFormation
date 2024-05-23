import os
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from operator import attrgetter

import matplotlib.pyplot as plt
import seaborn as sns
import imageio.v2 as iio

import argparse

import json

from learning_initializations import save_data_to_csv
from learning_environments import flagAutomata




###########################################################################
# Analysis folders creation
###########################################################################

def init_all_runs_analysis(params):

    # Create the 'analysis_dir' folder
    params['analysis_dir'] = {}
    params['analysis_dir']['root'] = os.getcwd() +"/simulationAnalysis/"+params['env_name']+"_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+"_"+params['flag_pattern']+"_"+str(params['automata_nb_rows'])+"x"+str(params['automata_nb_cols'])+"/learning"
    os.makedirs(params['analysis_dir']['root'], exist_ok=True)
    os.makedirs(params['analysis_dir']['root']+"/data_all_runs", exist_ok=True)
    os.makedirs(params['analysis_dir']['root']+"/plots_all_runs", exist_ok=True)

    save_data_to_csv(params['analysis_dir']['root']+"/data_all_runs/data_evo_all_runs_time.csv", [], header = ["Run", "Time"])
    save_data_to_csv(params['analysis_dir']['root']+"/data_all_runs/data_evo_all_runs_best_ind_per_run.csv", [], header = ["Run", "Generation", "Fitness", "Individual"])
    
    return params

#---------------------------------------------------

def init_one_run_analysis(run, params):

    # Create the data and plot directories tree from the 'analysis_dir' folder
    params['analysis_dir']['data'] = params['analysis_dir']['root']+"/run_"+str(run)+"/data"
    params['analysis_dir']['plots'] = params['analysis_dir']['root']+"/run_" + str(run) + "/plots"
    os.makedirs(params['analysis_dir']['data'], exist_ok=True)
    os.makedirs(params['analysis_dir']['plots'], exist_ok=True)

    # Create headers for files to update at each generation of each run
    save_data_to_csv(params['analysis_dir']['data']+"/data_evo_run_"+str(run)+"_all_pop.csv", [], header = ["Run", "Generation", "Fitness", "Individual"])
    save_data_to_csv(params['analysis_dir']['data']+"/data_evo_run_"+str(run)+"_best_inds_per_gen.csv", [], header = ["Run", "Generation", "Fitness", "Individual"])
    save_data_to_csv(params['analysis_dir']['data']+"/data_evo_run_"+str(run)+"_best_inds_ever.csv", [], header = ["Run", "Generation", "Fitness", "Individual"])

    return params


###########################################################################
# Save data functions
###########################################################################

def write_single_gen_data(run, gen, population, analysis_dir_data):

    # Save data of all individuals in the population for this single gen
    data_all_pop = []
    for ind in population:
        data_all_pop.append([str(run), str(gen), str(ind.fitness.values[0]).strip(), str(ind).strip()]) # [0]: en fonction de len fitness Kale
    save_data_to_csv(analysis_dir_data+"/data_evo_run_"+str(run)+"_all_pop.csv", data_all_pop)

    # Save best individual data for this single gen. NB: 'max' is the best fitness, deap manages if 'max' is maximization or minimization
    best_ind = max(population, key=attrgetter("fitness"))
    data_best_ind = [[str(run), str(gen), str(best_ind.fitness.values[0]).strip(), str(best_ind).strip()]] #check
    save_data_to_csv(analysis_dir_data+"/data_evo_run_"+str(run)+"_best_inds_per_gen.csv", data_best_ind)
 
#---------------------------------------------------
    
def write_single_run_data(run, time_run, analysis_dir):

    # Save best ever individual data for this single run
    dataset_path = analysis_dir['data']+"/data_evo_run_"+str(run)+"_best_inds_per_gen.csv"
    save_best_inds_ever_filename =  analysis_dir['data']+"/data_evo_run_"+str(run)+"_best_inds_ever.csv"
    save_best_ind_per_run_filename = analysis_dir['root']+"/data_all_runs/data_evo_all_runs_best_ind_per_run.csv"
    write_best_inds_ever_and_best_ind_per_run(dataset_path=dataset_path, save_best_inds_ever_filename=save_best_inds_ever_filename, save_best_ind_per_run_filename=save_best_ind_per_run_filename)

    # Save time information for this run
    data_evo_all_runs_time = [[str(run), str(time_run)]]
    save_data_to_csv(analysis_dir['root']+"/data_all_runs/data_evo_all_runs_time.csv", data_evo_all_runs_time)

#---------------------------------------------------

def write_all_runs_data(analysis_dir):

    # Collect the names of the 'run' folders
    dirs = os.listdir(analysis_dir)
    analysis_single_run_dirs = [dirs[i] for i in range(len(dirs)) if dirs[i].startswith("run_")]
    analysis_single_run_dirs.sort()

    # Concatenate 'data_evo_best_inds_ever' files of each run in one single file named 'data_evo_all_runs_best_inds_ever.csv' placed in 'data_all_runs' directory
    with open(analysis_dir+"/data_all_runs/data_evo_all_runs_best_inds_ever.csv", 'w') as all_runs_file:
        for i, single_run_dir in enumerate(analysis_single_run_dirs):
            path = analysis_dir+"/"+single_run_dir+"/data/data_evo_"+single_run_dir+"_best_inds_ever.csv"
            with open(path, 'r') as single_run_file:
                if i != 0:
                    next(single_run_file) # ignore the csv headers, keep just the 1st one
                all_runs_file.write(single_run_file.read())
                
#---------------------------------------------------

def write_best_inds_ever_and_best_ind_per_run(dataset_path, save_best_inds_ever_filename, save_best_ind_per_run_filename):

    dataset = pd.read_csv(dataset_path)

    generations = []
    data_best_inds_ever = []
    best_fit = np.inf

    for index in dataset.index:
        run = dataset.loc[index, 'Run']
        gen = dataset.loc[index, 'Generation']
        fit = dataset.loc[index, 'Fitness']

        generations.append(gen)
        if fit < best_fit:
            best_fit = fit
            data_best_inds_ever.append([str(run).strip(), str(gen).strip(), str(fit).strip(), str(dataset.loc[index, 'Individual']).strip()])

    save_data_to_csv(save_best_inds_ever_filename, data_best_inds_ever)
    save_data_to_csv(save_best_ind_per_run_filename, [data_best_inds_ever[-1]])


###########################################################################
# Plot data functions
###########################################################################
                
def plot_single_run_data(run, params):

    time_run = time.time()
    print(f"learning_analysis plots for the single run n.{run} - Started")
    os.makedirs(params['analysis_dir']['root']+"/run_"+str(run)+"/plots/evo", exist_ok=True)

    # Plot_all_pop_fitnesses_boxplot
    dataset_path = params['analysis_dir']['root']+"/run_"+str(run)+"/data/data_evo_run_"+str(run)+"_all_pop.csv"
    save_filename = params['analysis_dir']['root']+"/run_"+str(run)+"/plots/evo/plot_evo_run_"+str(run)+"_all_pop_fitnesses_boxplot.png"
    plot_all_pop_fitnesses_boxplot(run, dataset_path=dataset_path, save_filename=save_filename)

    # Plot_best_inds_ever
    dataset_path = params['analysis_dir']['root']+"/run_"+str(run)+"/data/data_evo_run_"+str(run)+"_best_inds_ever.csv"
    save_filename = params['analysis_dir']['root']+"/run_"+str(run)+"/plots/evo/plot_evo_run_"+str(run)+"_best_inds_ever.png"
    plot_best_inds_ever(dataset_path=dataset_path, save_filename=save_filename)

    # Plot_flag_from_file for best individuals ever in a defined range of steps
    save_dir = params['analysis_dir']['root']+"/run_"+str(run)+"/plots/env"
    dataset_path = params['analysis_dir']['root']+"/run_"+str(run)+"/data/data_evo_run_"+str(run)+"_best_inds_ever.csv"
    dataset = pd.read_csv(dataset_path)
    plot_env_step_start = params['time_window_start']
    plot_env_step_end = min(params['time_steps'], params['time_window_end'])
    steps = range(plot_env_step_start, plot_env_step_end)

    for index in dataset.index[:-1]: # for each best individual ever less the last one
        gen = dataset.loc[index, 'Generation']
        ind = dataset.loc[index, 'Individual']
        
        flagAutomata.plot_flag_from_file(env_eval_function_params=params['env']['eval_function_params'], data_flag_file=params['analysis_dir']['root']+"/run_"+str(run)+"/data/data_env_flag/data_env_flag_run_"+str(run)+"_gen_"+str(gen)+".csv", run=run, gen=gen, ind=ind, steps=steps, analysis_dir_plots=save_dir)
        flagAutomata.plot_flag_fitnesses_from_file(env_eval_function_params=params['env']['eval_function_params'], data_flag_file=params['analysis_dir']['root']+"/run_"+str(run)+"/data/data_env_flag/data_env_flag_run_"+str(run)+"_gen_"+str(gen)+".csv", run=run, gen=gen, ind=ind, analysis_dir_plots=save_dir)

    # Plot all the flag steps of the last best individual ever
    gen = dataset.loc[dataset.index[-1], 'Generation']
    ind = dataset.loc[dataset.index[-1], 'Individual']
    flagAutomata.plot_flag_from_file(env_eval_function_params=params['env']['eval_function_params'], data_flag_file=params['analysis_dir']['root']+"/run_"+str(run)+"/data/data_env_flag/data_env_flag_run_"+str(run)+"_gen_"+str(gen)+".csv", run=run, gen=gen, ind=ind, steps=None, analysis_dir_plots=save_dir)
    flagAutomata.plot_flag_fitnesses_from_file(env_eval_function_params=params['env']['eval_function_params'], data_flag_file=params['analysis_dir']['root']+"/run_"+str(run)+"/data/data_env_flag/data_env_flag_run_"+str(run)+"_gen_"+str(gen)+".csv", run=run, gen=gen, ind=ind, analysis_dir_plots=save_dir)

    # Animation of: plot all the flag steps of the last best individual ever
    if params['plot_with_animation_bool']:
        dir = params['analysis_dir']['root']+"/run_"+str(run)+"/plots/env/"
        flag_last_best_ind_ever_dirs = os.listdir(dir)
        flag_last_best_ind_ever_dir = sorted(flag_last_best_ind_ever_dirs, key=get_gen_ind_from_file_name)[-1]
        images = sorted(os.listdir(dir+flag_last_best_ind_ever_dir+"/flag"))
        frames = np.stack([iio.imread(dir+flag_last_best_ind_ever_dir+"/flag/"+img) for img in images], axis = 0)
        iio.mimwrite(dir+flag_last_best_ind_ever_dir+".gif", frames, format='GIF', duration=0.5*len(frames), subrectangles=True)
        print(f"Animation for the single run n.{run} - Saved in {dir}")

    time_run = time.time() - time_run
    print(f"learning_analysis plots for the single run n.{run} - Completed. Execution time: {time_run} seconds")
    
#---------------------------------------------------

def get_gen_ind_from_file_name(file_name):
    file_name_chunks = file_name.split('_')
    gen = int(file_name_chunks[3])
    nb_ind = int(file_name_chunks[5])
    return gen, nb_ind

#---------------------------------------------------

def plot_all_runs_data(params):

    # Plot_best_inds_ever
    dataset_path = params['analysis_dir']['root']+"/data_all_runs/data_evo_all_runs_best_inds_ever.csv"
    save_filename = params['analysis_dir']['root']+"/plots_all_runs/plot_evo_all_runs_best_inds_ever.png"
    plot_best_inds_ever(dataset_path=dataset_path, save_filename=save_filename)

    # Plot_flag_from_file for the target flag
    flagAutomata.plot_flag_from_file(env_eval_function_params=params['env']['eval_function_params'], data_flag_file=params['analysis_dir']['root']+"/data_all_runs/data_env_flag_target.csv", run=None, gen=None, ind=None, steps=None, analysis_dir_plots=params['analysis_dir']['root']+"/plots_all_runs")

    print(f"Plots for all the runs completed.")

#---------------------------------------------------

def plot_all_pop_fitnesses_boxplot(run, dataset_path, save_filename):

    dataset = pd.read_csv(dataset_path)
    dataset = dataset.loc[dataset.Run==run]

    sns.set_theme()
    sns.boxplot(x='Generation', y='Fitness', data=dataset, color='skyblue')

    plt.ylim(0, 1) # 0 and 1 are respectively min and max values of flag distance (fitness)
    plt.title("Fitnesses over generations\nall individuals generated", fontsize=14)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    generations = dataset['Generation'].unique()
    if max(generations) > 14:
        
        positions = [value for value in generations if value % (int(max(generations)/14))==0]
        plt.xticks(positions, fontsize=10)

    plt.savefig(save_filename)
    plt.clf()
    plt.close()

#---------------------------------------------------

def plot_best_inds_ever(dataset_path, save_filename):

    dataset = pd.read_csv(dataset_path)

    max_generation = dataset['Generation'].max()
    runs = dataset['Run'].unique()
    for run in runs:
        generations = dataset.loc[dataset.Run==run, 'Generation'].tolist()
        best_fitnesses_ever = dataset.loc[dataset.Run==run,'Fitness'].tolist()

        if generations[-1] != max_generation:
            generations.append(max_generation)
            best_fitnesses_ever.append(best_fitnesses_ever[-1])

        plt.step(generations, best_fitnesses_ever, where='post', label='run '+str(run)) # plot

    plt.ylim(0, 1) # 0 and 1 are respectively min and max values of flag distance (fitness)
    plt.title("Fitnesses over generations\nbest individuals ever", fontsize=14)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)
    plt.legend()

    plt.savefig(save_filename)
    plt.clf()
    plt.close()


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

#---------------------------------------------------

def worker2(task):

    run, params = task
    plot_single_run_data(run, params)

#---------------------------------------------------

def parallelize_processes2(task_queue, with_parallelization_nb_free_cores):

    # Create a Pool with the number of available cores
    available_cores = cpu_count() - with_parallelization_nb_free_cores
    with Pool(processes=available_cores) as pool:
        pool.map(worker2, task_queue)
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
    if params['with_parallelization_bool']:
        task_queue = [] # create a queue of tasks to execute
        for run in range(params['nb_runs']):
            task_queue.append((run, params.copy()))
        parallelize_processes2(task_queue, params['with_parallelization_nb_free_cores'])
    else:
        for run in range(params['nb_runs']):
            plot_single_run_data(run, params)

    write_all_runs_data(args.learning_analysis_dir)
    plot_all_runs_data(params)