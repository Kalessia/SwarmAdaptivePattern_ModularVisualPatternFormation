import os
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from operator import attrgetter
import bisect

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend instead of QtAgg
import matplotlib.pyplot as plt
import seaborn as sns
import imageio.v2 as iio

import argparse

import json

from learning_initializations import save_data_to_csv
from environments import swarmGrid


###########################################################################
# Analysis folders creation
###########################################################################

def init_all_runs_analysis(params):

    # Create the 'analysis_dir' folder
    params['analysis_dir'] = {}
    params['analysis_dir']['root'] = os.getcwd() +"/simulationAnalysis/"+params['evolutionary_settings']['env_name']+"_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+"_"+params['grid']['flag_pattern']+"_"+str(params['grid']['grid_nb_rows'])+"x"+str(params['grid']['grid_nb_cols'])+"/learning"
    os.makedirs(params['analysis_dir']['root'], exist_ok=True)
    os.makedirs(params['analysis_dir']['root']+"/data_all_runs", exist_ok=True)
    os.makedirs(params['analysis_dir']['root']+"/plots_all_runs", exist_ok=True)

    save_data_to_csv(params['analysis_dir']['root']+"/data_all_runs/data_evo_all_runs_time.csv", [], header = ["Run", "Time(s)", "Time(min)", "Time(h)"])
    save_data_to_csv(params['analysis_dir']['root']+"/data_all_runs/data_evo_all_runs_best_ind_per_run.csv", [], header = ["Run", "Generation", "Nb_eval", "Learning_phase", "Fitness", "Individual"])
    save_data_to_csv(params['analysis_dir']['root']+"/data_all_runs/data_evo_all_runs_best_ind_per_run_per_phase.csv", [], header = ["Run", "Generation", "Nb_eval", "Learning_phase", "Fitness", "Individual"])
    
    return params

#---------------------------------------------------

def init_one_run_analysis(run, params):

    # Create the data and plot directories tree from the 'analysis_dir' folder
    params['analysis_dir']['data'] = params['analysis_dir']['root']+ f"/run_{run:03}/data"
    params['analysis_dir']['plots'] = params['analysis_dir']['root']+ f"/run_{run:03}/plots"
    os.makedirs(params['analysis_dir']['data'], exist_ok=True)
    os.makedirs(params['analysis_dir']['plots'], exist_ok=True)

    # Create headers for files to update at each generation of each run
    save_data_to_csv(params['analysis_dir']['data']+ f"/data_evo_run_{run:03}_all_pop.csv", [], header = ["Run", "Generation", "Nb_eval", "Fitness", "Individual"])
    save_data_to_csv(params['analysis_dir']['data']+ f"/data_evo_run_{run:03}_best_inds_per_gen.csv", [], header = ["Run", "Generation", "Nb_eval", "Fitness", "Individual"])
    save_data_to_csv(params['analysis_dir']['data']+ f"/data_evo_run_{run:03}_best_inds_ever.csv", [], header = ["Run", "Generation", "Nb_eval", "Learning_phase", "Fitness", "Individual"])

    return params


###########################################################################
# Save data functions
###########################################################################

def write_single_gen_data(run, gen, nb_evals, population, analysis_dir_data):

    # Save data of all individuals in the population for this single gen
    data_all_pop = []
    for i, ind in enumerate(population):
        data_all_pop.append([str(run).strip(), str(gen).strip(), str(nb_evals[i]).strip(), str(ind.fitness.values[0]).strip(), str(ind).strip()])
    save_data_to_csv(analysis_dir_data+ f"/data_evo_run_{run:03}_all_pop.csv", data_all_pop)

    # Save best individual data for this single gen. NB: 'max' is the best fitness, deap manages if 'max' is maximization or minimization
    best_ind = max(population, key=attrgetter("fitness"))
    best_ind_nb_eval = nb_evals[population.index(best_ind)]
    data_best_ind = [[str(run).strip(), str(gen).strip(), str(best_ind_nb_eval).strip(), str(best_ind.fitness.values[0]).strip(), str(best_ind).strip()]] #check
    save_data_to_csv(analysis_dir_data+ f"/data_evo_run_{run:03}_best_inds_per_gen.csv", data_best_ind)
 
#---------------------------------------------------
    
def write_single_run_data(run, switch_gen, time_run, analysis_dir):

    # Save best ever individual data for this single run
    dataset_path = analysis_dir['data']+ f"/data_evo_run_{run:03}_best_inds_per_gen.csv"
    save_best_inds_ever_filename =  analysis_dir['data']+ f"/data_evo_run_{run:03}_best_inds_ever.csv"
    save_best_ind_per_run_filename = analysis_dir['root']+"/data_all_runs/data_evo_all_runs_best_ind_per_run.csv"
    save_best_ind_per_run_per_phase_filename = analysis_dir['root']+"/data_all_runs/data_evo_all_runs_best_ind_per_run_per_phase.csv"
    write_best_inds_ever_and_best_ind_per_run(dataset_path=dataset_path, switch_gen=switch_gen, save_best_inds_ever_filename=save_best_inds_ever_filename, save_best_ind_per_run_filename=save_best_ind_per_run_filename, save_best_ind_per_run_per_phase_filename=save_best_ind_per_run_per_phase_filename)

    # Save time information for this run
    data_evo_all_runs_time = [[str(run), str(time_run), str((time_run+1)/60), str((time_run+1)/3600)]]
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

    # Concatenate 'data_evo_best_inds_per_gen' files of each run in one single file named 'data_evo_all_runs_best_inds_per_gen.csv' placed in 'data_all_runs' directory
    with open(analysis_dir+"/data_all_runs/data_evo_all_runs_best_inds_per_gen.csv", 'w') as all_runs_file:
        for i, single_run_dir in enumerate(analysis_single_run_dirs):
            path = analysis_dir+"/"+single_run_dir+"/data/data_evo_"+single_run_dir+"_best_inds_per_gen.csv"
            with open(path, 'r') as single_run_file:
                if i != 0:
                    next(single_run_file) # ignore the csv headers, keep just the 1st one
                all_runs_file.write(single_run_file.read())

    # Save fitnesses stats over all runs in data_all_runs/data_evo_all_runs_fitnesses_stats_per_run_per_gen.csv
    data_all_runs_fitnesses_stats = []
    for i, single_run_dir in enumerate(analysis_single_run_dirs):
        path = analysis_dir+"/"+single_run_dir+"/data/data_evo_"+single_run_dir+"_all_pop.csv"
        dataset = pd.read_csv(path)
        generations = dataset['Generation'].unique()
        for gen in generations:
            nb_eval = dataset.loc[dataset.Generation==gen, 'Nb_eval'].tolist()[-1] # nb_eval is the last evaluation for this generation, as stats referes to all generation fitnesses 
            fitnesses_gen = dataset.loc[dataset.Generation==gen, 'Fitness'].tolist()
            data_all_runs_fitnesses_stats.append([str(i).strip(), str(gen).strip(), str(nb_eval).strip(), str(np.mean(fitnesses_gen)).strip(), str(np.min(fitnesses_gen)).strip(), str(np.quantile(fitnesses_gen, 0.25)).strip(), str(np.quantile(fitnesses_gen, 0.50)).strip(), str(np.quantile(fitnesses_gen, 0.75)).strip(), str(np.max(fitnesses_gen)).strip(), str(np.std(fitnesses_gen)).strip()])
    save_data_to_csv(analysis_dir+"/data_all_runs/data_evo_all_runs_fitnesses_stats_per_run_per_gen.csv", data_all_runs_fitnesses_stats, header = ["Run", "Gen", "Nb_eval", "Fitnesses_mean", "Fitnesses_min", "Fitnesses_quantile25", "Fitnesses_median", "Fitnesses_quantile75", "Fitnesses_max", "Fitnesses_std"])

    # Save fitnesses mean and fitnesses std of the best inds per run
    if not (os.path.exists(analysis_dir+"/data_all_runs/data_evo_all_runs_mean_std.csv")):
        dataset = pd.read_csv(analysis_dir+"/data_all_runs/data_evo_all_runs_best_ind_per_run.csv")
        save_data_to_csv(analysis_dir+"/data_all_runs/data_evo_all_runs_best_ind_per_run_mean_std.csv", [[str(dataset['Fitness'].mean()), str(dataset['Fitness'].std())]], header = ["Best_inds_per_run_mean_fitnesses", "Best_inds_per_run_std_fitnesses"])

#---------------------------------------------------

def write_best_inds_ever_and_best_ind_per_run(dataset_path, switch_gen, save_best_inds_ever_filename, save_best_ind_per_run_filename, save_best_ind_per_run_per_phase_filename):

    dataset = pd.read_csv(dataset_path) # this dataset contains a line per gen

    generations = []
    data_best_inds_ever = []
    best_fit = np.inf
    learning_phase = 1

    for index in dataset.index:
        run = dataset.loc[index, 'Run']
        gen = dataset.loc[index, 'Generation']
        nb_eval = dataset.loc[index, 'Nb_eval']
        fit = dataset.loc[index, 'Fitness']

        generations.append(gen)
        
        if switch_gen is not None and gen == switch_gen:
            if switch_gen == 0:
                best_fit_ever_phase1 = fit
                data_best_inds_ever_phase1 = [[str(run).strip(), str(gen).strip(), str(nb_eval).strip(), str(1).strip(), str(fit).strip(), str(dataset.loc[index, 'Individual']).strip()]]
            else:
                best_fit_ever_phase1 = best_fit
                data_best_inds_ever_phase1 = [data_best_inds_ever[-1]]
            
            learning_phase = 2
            best_fit = np.inf # reset best_fit for phase2


        if fit < best_fit:
            best_fit = fit
            data_best_inds_ever.append([str(run).strip(), str(gen).strip(), str(nb_eval).strip(), str(learning_phase).strip(), str(fit).strip(), str(dataset.loc[index, 'Individual']).strip()])


    # Select and write the data about the individual with the best fitness per run 
    best_fit_per_run = [data_best_inds_ever[-1]]
    if switch_gen is not None:
        save_data_to_csv(save_best_ind_per_run_per_phase_filename, data_best_inds_ever_phase1)
        if best_fit_ever_phase1 < best_fit:
            best_fit_per_run = data_best_inds_ever_phase1

    save_data_to_csv(save_best_inds_ever_filename, data_best_inds_ever) # "/data_evo_run_{run:03}_best_inds_ever.csv"
    save_data_to_csv(save_best_ind_per_run_filename, best_fit_per_run) # "/data_all_runs/data_evo_all_runs_best_ind_per_run.csv"
    save_data_to_csv(save_best_ind_per_run_per_phase_filename, [data_best_inds_ever[-1]]) # "/data_all_runs/data_evo_all_runs_best_ind_per_run_per_phase.csv"


###########################################################################
# Plot data functions
###########################################################################
                
def plot_single_run_data(run, params):  # TODO: si on a un autre setup que "incremental" cela crushes???

    # if run != 4 :
    #     return

    time_run = time.time()
    print(f"learning_analysis plots for the single run n.{run} - Started")
    os.makedirs(params['analysis_dir']['root']+ f"/run_{run:03}/plots/evo", exist_ok=True)
    # os.makedirs(params['analysis_dir']['root']+ f"/run_{run:03}/data/flags_best_inds_ever", exist_ok=True)

    nb_evals = params['evolutionary_settings']['nb_evals']
    switch_eval = params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_switch_eval']
    grid_size = params['grid']['grid_size']
    nb_deletions = params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_ticks']
    density = round(1.0-params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_percent'][1], 2)
    fluidity = params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_proba_move']

    # Plot_all_pop_fitnesses_boxplot
    dataset_path = params['analysis_dir']['root']+ f"/run_{run:03}/data/data_evo_run_{run:03}_all_pop.csv"
    save_filename = params['analysis_dir']['root']+ f"/run_{run:03}/plots/evo/plot_evo_run_{run:03}_all_pop_fitnesses_boxplot.png"
    plot_all_pop_fitnesses_boxplot(run, dataset_path=dataset_path, nb_evals=nb_evals, grid_size=grid_size, nb_deletions=nb_deletions, switch_eval=switch_eval, save_filename=save_filename)

    # Plot_best_inds_ever
    dataset_path = params['analysis_dir']['root']+ f"/run_{run:03}/data/data_evo_run_{run:03}_best_inds_ever.csv"
    save_filename = params['analysis_dir']['root']+ f"/run_{run:03}/plots/evo/plot_evo_run_{run:03}_best_inds_ever.png"
    plot_best_inds_ever(dataset_path=dataset_path, nb_evals=nb_evals, grid_size=grid_size, nb_deletions=nb_deletions, density=density, fluidity=fluidity, switch_eval=switch_eval, save_filename=save_filename, params=params)

    # Plot_flag_from_file for best individuals ever in a defined range of steps
    dataset_path = params['analysis_dir']['root']+ f"/run_{run:03}/data/data_evo_run_{run:03}_best_inds_ever.csv"
    best_inds_ever_dataset = pd.read_csv(dataset_path)
    time_steps = params['environment']['time_steps']
    # steps = list(range(params['environment']['time_window_start'], params['environment']['time_window_end'])) # we plot this steps interval for all individual less the best
    steps = [0, time_steps-5, time_steps-4, time_steps-3, time_steps-2, time_steps-1]

    for index in best_inds_ever_dataset.index: # for each best individual ever less the last one <--- perché ho scritto cio?
        gen = best_inds_ever_dataset.loc[index, 'Generation']
        nb_eval = best_inds_ever_dataset.loc[index, 'Nb_eval']
        ind = best_inds_ever_dataset.loc[index, 'Individual']
        dataset = pd.read_csv(params['analysis_dir']['root']+ f"/run_{run:03}/data/data_env_flag/data_env_flag_run_{run:03}_gen_{gen:05}_eval_{nb_eval:07}.csv")
        
        dataset_gen = dataset.loc[(dataset.Generation==gen)]
        dataset = dataset_gen.loc[(dataset_gen.Individual==str(ind))]
        nb_ind = dataset['Nb_ind'].unique()[0]

        if index == best_inds_ever_dataset.index[-3]: # we plot all steps for the last (the best) individual
            steps = dataset['Step'].unique()

        # data_env_flag = []
        for step in range(time_steps):
            flag_list = get_flag_list_from_dataset_step(dataset, step)
            fitness = dataset.loc[(dataset.Step==step),['Flags_distance']].values.tolist()[0][0]
            # time_window_zone = dataset.loc[(dataset.Step==step),['Time_window_zone']].values.tolist()[0][0]
            
            # TODO if deleted pos in file, if incremental learning ?
            deleted_pos = dataset.loc[(dataset.Step==step),['Deleted_agents_positions']].values.tolist()[0][0]
            deleted_pos = eval(deleted_pos)
            nb_moves_per_step = dataset.loc[(dataset.Step==step),['Nb_moves']].values.tolist()[0][0]

            if step in steps:
                # swarmGrid.plot_flag(grid_nb_rows=params['grid']['grid_nb_rows'],
                #                     grid_nb_cols=params['grid']['grid_nb_cols'],
                #                     setup_name=None,
                #                     run=run,
                #                     nb_ind=nb_ind,
                #                     gen=gen,
                #                     nb_eval=nb_eval,
                #                     n="",
                #                     step=step,
                #                     flag=flag_list,
                #                     fitness=fitness,
                #                     deleted_pos=deleted_pos,
                #                     nb_moves_per_step=nb_moves_per_step,
                #                     analysis_dir_plots=params['analysis_dir']['root']+ f"/run_{run:03}/plots/env")
            
                # Write this individual
                if step == steps[0]:
                    file_path = params['analysis_dir']['root']+ f"/run_{run:03}/plots/env/run_{run:03}_gen_{gen:05}_eval_{nb_eval:07}_individual_{nb_ind:03}/flag_individual.txt"
                    if not os.path.exists(file_path):
                        with open (file_path, 'w') as f:
                            f.write(str(ind))

            # data_env_flag.append([str(gen), str(step), str(fitness).strip(), str(time_window_zone).strip(), str(flag_list.tolist()).strip(), str(ind).strip()])

        # save_data_to_csv(params['analysis_dir']['root']+ f"/run_{run:03}/data/flags_best_inds_ever/run_{run:03}_gen_{gen:05}_eval_{nb_eval:07}_individual_{nb_ind:03}.csv", data_env_flag, header = ["Generation", "Step", "Flags_distance", "Time_window_zone", "Flag", "Individual"])

        swarmGrid.plot_flag_fitnesses_from_file(data_flag_file=params['analysis_dir']['root']+ f"/run_{run:03}/data/data_env_flag/data_env_flag_run_{run:03}_gen_{gen:05}_eval_{nb_eval:07}.csv",
                                                setup_name=None,
                                                time_window_start=params['environment']['time_window_start'],
                                                time_window_length= params['environment']['time_window_end'] - params['environment']['time_window_start'] + 1,
                                                run=run,
                                                nb_ind=nb_ind,
                                                ind=ind,
                                                n="",
                                                gen=gen,
                                                nb_eval=nb_eval,
                                                switch_step=None,
                                                analysis_dir_plots=params['analysis_dir']['root']+ f"/run_{run:03}/plots/env")
        
        # TODO: le nom du dossier de sauvegarde n'est pas bon, il faudrait un plot par flag??? Evitabile
        # if params['evolutionary_settings']['env_name'] == "sliding_puzzle_incremental":
        #     swarmGrid.plot_learning_sliding_puzzle_nb_moves_from_file(data_flag_file=params['analysis_dir']['root']+ f"/run_{run:03}/data/data_env_flag/data_env_flag_run_{run:03}_gen_{gen:05}_eval_{nb_eval:07}.csv",
        #                                                run=run,
        #                                                grid_size=params['grid']['grid_nb_rows']*params['grid']['grid_nb_cols'],
        #                                                analysis_dir_plots=params['analysis_dir']['root']+ f"/run_{run:03}/plots/env")
            
            
    # Animation of: plot all the flag steps of the last best individual ever
    if params['plot_with_animation_bool']:
        dir = params['analysis_dir']['root']+ f"/run_{run:03}/plots/env/"
        flag_last_best_ind_ever_dirs = os.listdir(dir)
        flag_last_best_ind_ever_dir = sorted(flag_last_best_ind_ever_dirs, key=get_gen_ind_from_file_name)[-1]
        images = sorted(os.listdir(dir+flag_last_best_ind_ever_dir+"/flag"))
        frames = np.stack([iio.imread(dir+flag_last_best_ind_ever_dir+"/flag/"+img) for img in images], axis = 0)
        iio.mimwrite(dir+flag_last_best_ind_ever_dir+".gif", frames, format='GIF', duration=0.5, subrectangles=True)
        print(f"Animation for the single run n.{run} - Saved in {dir}")

    # shutil.rmtree(params['analysis_dir']['root']+ f"/run_{run:03}/data/data_env_flag")
    time_run = time.time() - time_run
    print(f"learning_analysis plots for the single run n.{run} - Completed. Execution time: {time_run} seconds")

#---------------------------------------------------

def get_gen_ind_from_file_name(file_name):
    file_name_chunks = file_name.split('_')
    gen = int(file_name_chunks[3])
    nb_ind = int(file_name_chunks[5])
    return gen, nb_ind

#---------------------------------------------------

def get_flag_list_from_dataset_step(dataset, step):
    flag_list = dataset.loc[(dataset.Step==step),['Flag']].values.tolist()[0][0]
    flag_list = str(flag_list).replace('[', '').replace(']', '').strip()
    flag_list = np.asarray(flag_list.split(','), dtype=np.float32)

    return flag_list

#---------------------------------------------------

def plot_all_runs_data(params):

    nb_evals = params['evolutionary_settings']['nb_evals']
    switch_eval = params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_switch_eval'] # switch_eval is the 1st evaluation of the new generation starting after the switch_eval defined in parameters
    grid_size = params['grid']['grid_size']
    nb_deletions = params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_ticks']
    density = round(1.0-params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_percent'][1], 2)
    fluidity = params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_proba_move']

    # Plot_best_inds_ever
    dataset_path = params['analysis_dir']['root']+"/data_all_runs/data_evo_all_runs_best_inds_ever.csv"
    save_filename = params['analysis_dir']['root']+"/plots_all_runs/plot_evo_all_runs_best_inds_ever.png"
    plot_best_inds_ever(dataset_path=dataset_path, nb_evals=nb_evals, grid_size=grid_size, nb_deletions=nb_deletions, density=density, fluidity=fluidity, switch_eval=switch_eval, save_filename=save_filename, params=params)

    # Plot_best_inds_per_gen
    dataset_path = params['analysis_dir']['root']+"/data_all_runs/data_evo_all_runs_best_inds_per_gen.csv"
    save_filename = params['analysis_dir']['root']+"/plots_all_runs/plot_evo_all_runs_best_inds_per_gen.png"
    plot_best_inds_per_gen(dataset_path=dataset_path, nb_evals=nb_evals, grid_size=grid_size, nb_deletions=nb_deletions, switch_eval=switch_eval, with_best_ever_bool=True, save_filename=save_filename)

    # Plot fitnesses mean
    dataset_path = params['analysis_dir']['root']+"/data_all_runs/data_evo_all_runs_fitnesses_stats_per_run_per_gen.csv"
    save_filename = params['analysis_dir']['root']+"/plots_all_runs/plot_evo_all_runs_fitnesses_stats_per_run_per_gen_mean.png"
    # plot_all_pop_fitnesses_mean(dataset_path=dataset_path, nb_evals=nb_evals, grid_size=grid_size, nb_deletions=nb_deletions, switch_eval=switch_eval, save_filename=save_filename)

    # Plot fitnesses mean
    dataset_path = params['analysis_dir']['root']+"/data_all_runs/data_evo_all_runs_fitnesses_stats_per_run_per_gen.csv"
    save_filename = params['analysis_dir']['root']+"/plots_all_runs/plot_evo_all_runs_fitnesses_stats_per_run_per_gen_median.png"
    # plot_all_pop_fitnesses_median(dataset_path=dataset_path, nb_evals=nb_evals, grid_size=grid_size, nb_deletions=nb_deletions, switch_eval=switch_eval, save_filename=save_filename)

    # Plot flag target
    dataset = pd.read_csv(params['analysis_dir']['root']+"/data_all_runs/data_env_flag_target.csv")
    flag_list = get_flag_list_from_dataset_step(dataset, 0)

    # swarmGrid.plot_flag(grid_nb_rows=params['grid']['grid_nb_rows'],
    #                 grid_nb_cols=params['grid']['grid_nb_cols'],
    #                 setup_name=None,
    #                 run=run,
    #                 nb_ind=None,
    #                 gen=0,
    #                 nb_eval=0,
    #                 n="",
    #                 step=0,
    #                 flag=flag_list,
    #                 fitness=0,
    #                 deleted_pos=[],
    #                 analysis_dir_plots=params['analysis_dir']['root']+"/plots_all_runs")
    
    print(f"Plots for all the runs completed.")

#---------------------------------------------------

def plot_all_pop_fitnesses_boxplot(run, dataset_path, nb_evals, grid_size, nb_deletions, switch_eval, save_filename):

    dataset = pd.read_csv(dataset_path)
    dataset = dataset.loc[dataset.Run==run]

    nb_gens_to_plot = 20 # number of boxes to display
    generations = dataset['Generation'].unique()
    pop_size = int(nb_evals/len(generations))
    gens_to_plot = list(range(generations[0], generations[-1], max(1, int(len(generations)/nb_gens_to_plot))))

    filtered_dataset = dataset[dataset['Generation'].isin(gens_to_plot)]
    filtered_dataset = filtered_dataset.copy() # create a copy to avoid SettingWithCopyWarning
    filtered_dataset['Max_evaluation_per_gen'] = pd.to_numeric(filtered_dataset['Generation'].apply(lambda gen: pop_size * (gen + 1))) # nb_eval is here the last evaluation of a given generation, even if the best ind per run per gen was registered in different nb_eval within the same generation
    evaluations = filtered_dataset['Max_evaluation_per_gen'].unique()

    plt.figure(figsize=(12, 7))
    sns.set_theme(style='darkgrid')
    _, ax = plt.subplots()
    sns.boxplot(x='Max_evaluation_per_gen',
                y='Fitness',
                data=filtered_dataset,
                color='skyblue',
                medianprops={'color': 'red', 'linewidth': 0.5},
                width=0.7,
                ax=ax) # individuals of differents runs with same generation have also same nb_evaluation, so dataset is grouped by "nb_eval" in the "Evaluation" column

    # Plot phase1-phase2 delimeter and max fitness limit
    x_end = len(evaluations)
    y = 1
    if switch_eval is not None:
        index = bisect.bisect_left(evaluations, switch_eval) # bisect_left returns the 1st index where switch_eval would be inserted to maintain the list order
        plt.axvline(x=index-0.5, color='r', linestyle='--')
        y = 1 - (nb_deletions[1]/grid_size)
        plt.plot([index-0.5, x_end], [y, y], linestyle=':', linewidth=2.0, color='tab:blue') # max fitness limit phase2
        x_end = index-0.5
        y = 1 - (nb_deletions[0]/grid_size) 
    
    plt.plot([0, x_end], [y, y], linestyle=':', linewidth=2.0, color='tab:blue', label=f"worst flags dist") # max fitness limit phase1

    plt.title("Flags distance over generations\nall individuals generated", fontsize=14)
    plt.xlabel("Evaluations", fontsize=12)
    plt.ylabel("Flags distance", fontsize=12)

    plt.ylim(-0.1, 1.1) # 0 and 1 are respectively min and max values of flag distance (fitness)
    ax.set_xticks(range(0, len(evaluations))) # important: boxplot boxes locations are incremental number from 0 to N=len(evaluations)
    pace = int(len(evaluations) / 7) # number of labels to display, for lisibility
    evaluations_labels = [str(evaluations[i]) if i%pace == 0 else "" for i in range(0, len(evaluations))]
    ax.set_xticklabels(evaluations_labels)

    plt.savefig(save_filename)
    plt.clf()
    plt.close()

#---------------------------------------------------

def plot_best_inds_ever(dataset_path, nb_evals, grid_size, nb_deletions, density, fluidity, switch_eval, save_filename, params):

    dataset = pd.read_csv(dataset_path)

    runs = dataset['Run'].unique()
    for run in runs:
        evals = dataset.loc[dataset.Run==run, 'Nb_eval'].tolist()
        best_fitnesses_ever = dataset.loc[dataset.Run==run,'Fitness'].tolist()

        # Make all plot lines last until x=nb_evals
        if evals[-1] != nb_evals:
            evals.append(nb_evals)
            best_fitnesses_ever.append(best_fitnesses_ever[-1])

        plt.step(evals, best_fitnesses_ever, where='post', label= f"run {run}") # plot

    # Plot phase1-phase2 delimeter and max fitness limit
    x_end = nb_evals
    y = 1
    # if switch_eval is not None:
    #     plt.axvline(x=switch_eval, color='r', linestyle='--')
    #     y = 1 - (nb_deletions[1]/grid_size)
    #     plt.plot([switch_eval, x_end], [y, y], linestyle=':', linewidth=2.0, color='tab:blue') # max fitness limit phase2
    #     x_end = switch_eval
    #     y = 1 - (nb_deletions[0]/grid_size)
    
    # plt.plot([0, x_end], [y, y], linestyle=':', linewidth=2.0, color='tab:blue', label=f"worst flags dist") # max fitness limit phase1


    plt.ylim(-0.1, 1.1) # 0 and 1 are respectively min and max values of flag distance (fitness)
    plt.xlim(0, nb_evals)
    plt.yticks([0, 0.5, 1.0], fontsize=18)
    plt.xticks([0, 7000, 14000], fontsize=18)
    plt.title(f"Learning $\\rho$={density}, $\\Phi$={fluidity}" + f"\n{params['grid']['flag_pattern']} {params['grid']['grid_nb_rows']}x{params['grid']['grid_nb_cols']}, 11 runs, only best ever", fontsize=18)
    plt.xlabel("Evaluations", fontsize=18)
    plt.ylabel("Distance", fontsize=18)
    plt.legend(loc="upper right") 
    plt.tight_layout()

    plt.savefig(save_filename)
    plt.clf()
    plt.close()

#---------------------------------------------------

def plot_best_inds_per_gen(dataset_path, nb_evals, grid_size, nb_deletions, switch_eval, with_best_ever_bool, save_filename):

    dataset = pd.read_csv(dataset_path)

    nb_gens_to_plot = 20 # number of boxes to display
    generations = dataset['Generation'].unique()
    pop_size = int(nb_evals/len(generations))
    gens_to_plot = list(range(generations[0], generations[-1], max(1, int(len(generations)/nb_gens_to_plot))))

    filtered_dataset = dataset[dataset['Generation'].isin(gens_to_plot)]
    filtered_dataset = filtered_dataset.copy() # create a copy to avoid SettingWithCopyWarning
    filtered_dataset['Max_evaluation_per_gen'] = pd.to_numeric(filtered_dataset['Generation'].apply(lambda gen: pop_size * (gen + 1))) # nb_eval is here the last evaluation of a given generation, even if the best ind per run per gen was registered in different nb_eval within the same generation
    evaluations = filtered_dataset['Max_evaluation_per_gen'].unique()

    plt.figure(figsize=(12, 7))
    # sns.set_theme(style='whitegrid')
    sns.set_theme(style='darkgrid')
    _, ax = plt.subplots()
    sns.boxplot(x='Max_evaluation_per_gen',
                y='Fitness',
                data=filtered_dataset,
                color='skyblue',
                medianprops={'color': 'red', 'linewidth': 0.5},
                width=0.7,
                ax=ax) # individuals of differents runs with same generation have also same nb_evaluation, so dataset is grouped by "nb_eval" in the "Evaluation" column

    # Plot phase1-phase2 delimeter and max fitness limit
    x_end = len(evaluations)
    y = 1
    if switch_eval is not None:
        index = bisect.bisect_left(evaluations, switch_eval) # bisect_left returns the 1st index where switch_eval would be inserted to maintain the list order
        plt.axvline(x=index-0.5, color='r', linestyle='--')
        y = 1 - (nb_deletions[1]/grid_size)
        plt.plot([index-0.5, x_end], [y, y], linestyle=':', linewidth=2.0, color='tab:blue') # max fitness limit phase2
        x_end = index-0.5
        y = 1 - (nb_deletions[0]/grid_size)

    plt.plot([0, x_end], [y, y], linestyle=':', linewidth=2.0, color='tab:blue', label=f"worst flags dist") # max fitness limit phase1

    # Plot best inds ever
    if with_best_ever_bool:
        best_fitnesses_per_gen_ever = []
        best_fitnesses_per_gen_dataset = filtered_dataset.loc[filtered_dataset.groupby('Max_evaluation_per_gen')['Fitness'].idxmin()]
        best_fitnesses_per_gen_list = best_fitnesses_per_gen_dataset['Fitness'].tolist() # list of the min fitness for a same gen/nb_eval, among all runs
        best_fit = np.inf
        index = bisect.bisect_left(evaluations, switch_eval) # bisect_left returns the 1st index where switch_eval would be inserted to maintain the list order
        for i, fit in enumerate(best_fitnesses_per_gen_list):
            if fit < best_fit or i == index:
                best_fit = fit
            best_fitnesses_per_gen_ever.append(best_fit)

        plt.step(range(0, len(evaluations)), best_fitnesses_per_gen_ever, where='post', label= f"run {run}") # plot

    plt.title("Flags distance over generations\nbest individuals distributions over evolution", fontsize=14)
    plt.xlabel("Evaluations", fontsize=12)
    plt.ylabel("Flags distance", fontsize=12)
    plt.ylim(-0.1, 1.1) # 0 and 1 are respectively min and max values of flag distance (fitness)
    ax.set_xticks(range(0, len(evaluations))) # important: boxplot boxes locations are incremental number from 0 to N=len(evaluations)
    pace = int(len(evaluations) / 7) # number of labels to display, for lisibility
    evaluations_labels = [str(evaluations[i]) if i%pace == 0 else "" for i in range(0, len(evaluations))]
    ax.set_xticklabels(evaluations_labels)

    plt.savefig(save_filename)
    plt.clf()
    plt.close()

#---------------------------------------------------

def plot_all_pop_fitnesses_mean(dataset_path, nb_evals, grid_size, nb_deletions, switch_eval, save_filename):

    dataset = pd.read_csv(dataset_path)

    runs = dataset['Run'].unique()
    for run in runs:
        evals = dataset.loc[dataset.Run==run, 'Nb_eval'].tolist()
        fitnesses_means = dataset.loc[dataset.Run==run,'Fitnesses_mean'].tolist()
        fitnesses_quantile25 = np.quantile(fitnesses_means, 0.25)
        fitnesses_quantile75 = np.quantile(fitnesses_means, 0.75)
        plt.plot(evals, fitnesses_means, label= f"run {run}")
        plt.fill_between(x=evals, y1=fitnesses_quantile25, y2=fitnesses_quantile75, alpha=0.6)

    # Plot phase1-phase2 delimeter and max fitness limit
    x_end = nb_evals
    y = 1
    if switch_eval is not None:
        plt.axvline(x=switch_eval, color='r', linestyle='--')
        y = 1 - (nb_deletions[1]/grid_size)
        plt.plot([switch_eval, x_end], [y, y], linestyle=':', linewidth=2.0, color='tab:blue') # max fitness limit phase2
        x_end = switch_eval
        y = 1 - (nb_deletions[0]/grid_size)
    
    plt.plot([0, x_end], [y, y], linestyle=':', linewidth=2.0, color='tab:blue', label=f"worst flags dist") # max fitness limit phase1

    plt.ylim(-0.1, 1.1) # 0 and 1 are respectively min and max values of flag distance (fitness)
    plt.xlim(0, nb_evals)
    plt.title("Flags distance over generations\nmean, quantile25 and quantile75 of all individuals fitnesses per generation", fontsize=12)
    plt.xlabel("Evaluations", fontsize=12)
    plt.ylabel("Flags distance", fontsize=12)
    plt.legend()

    plt.savefig(save_filename)
    plt.clf()
    plt.close()

#---------------------------------------------------

def plot_all_pop_fitnesses_median(dataset_path, nb_evals, grid_size, nb_deletions, switch_eval, save_filename):

    dataset = pd.read_csv(dataset_path)

    runs = dataset['Run'].unique()
    for run in runs:
        evals = dataset.loc[dataset.Run==run, 'Nb_eval'].tolist()
        fitnesses_medians = dataset.loc[dataset.Run==run,'Fitnesses_median'].tolist()
        fitnesses_quantile25 = np.quantile(fitnesses_medians, 0.25)
        fitnesses_quantile75 = np.quantile(fitnesses_medians, 0.75)
        plt.plot(evals, fitnesses_medians, label= f"run {run}")
        plt.fill_between(x=evals, y1=fitnesses_quantile25, y2=fitnesses_quantile75, alpha=0.6)

    # Plot phase1-phase2 delimeter and max fitness limit
    x_end = nb_evals
    y = 1
    if switch_eval is not None:
        plt.axvline(x=switch_eval, color='r', linestyle='--')
        y = 1 - (nb_deletions[1]/grid_size)
        plt.plot([switch_eval, x_end], [y, y], linestyle=':', linewidth=2.0, color='tab:blue') # max fitness limit phase2
        x_end = switch_eval
        y = 1 - (nb_deletions[0]/grid_size)
    
    plt.plot([0, x_end], [y, y], linestyle=':', linewidth=2.0, color='tab:blue', label=f"worst flags dist") # max fitness limit phase1


    plt.ylim(-0.1, 1.1) # 0 and 1 are respectively min and max values of flag distance (fitness)
    plt.xlim(0, nb_evals)
    plt.title("Flags distance over generations\nmedian, quantile25 and quantile75 of all individuals fitnesses per generation", fontsize=12)
    plt.xlabel("Evaluations", fontsize=12)
    plt.ylabel("Flags distance", fontsize=12)
    plt.legend()

    plt.savefig(save_filename)
    plt.clf()
    plt.close()


###########################################################################
# Parallelization
###########################################################################

def worker(task):

    run, params = task
    plot_single_run_data(run, params)

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
        for run in range(params['evolutionary_settings']['nb_runs']):
            task_queue.append((run, params.copy()))
        parallelize_processes(task_queue, params['with_parallelization_nb_free_cores'])
    else:
        for run in range(params['evolutionary_settings']['nb_runs']):
            plot_single_run_data(run, params)

    write_all_runs_data(args.learning_analysis_dir)
    plot_all_runs_data(params)