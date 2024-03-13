import os
from datetime import datetime

import numpy as np
import pandas as pd
from operator import attrgetter

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import imageio.v2 as iio

import csv





###########################################################################
# Analysis folders creation
###########################################################################

def init_all_runs_analysis(params):

    # Create the 'analysis_dir' folder
    analysis_dir = "simulationAnalysis/"+params['env_name']+"_simulation_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(analysis_dir+"/data_all_runs", exist_ok=True)
    os.makedirs(analysis_dir+"/plots_all_runs", exist_ok=True)
    params['analysis_dir'] = {}
    params['analysis_dir']['root'] = analysis_dir

    save_data_to_csv(params['analysis_dir']['root']+"/data_all_runs/data_time_runs.csv", [], header = ["Run", "Time"])
    
    return params

#---------------------------------------------------

def init_one_run_analysis(run, params):

    # Create the data and plot directories tree from the 'analysis_dir' folder
    params['analysis_dir']['data'] = params['analysis_dir']['root']+"/run_"+str(run)+"/data"
    params['analysis_dir']['plots'] = params['analysis_dir']['root'] + "/run_" + str(run) + "/plots"
    os.makedirs(params['analysis_dir']['data'], exist_ok=True)
    os.makedirs(params['analysis_dir']['plots'], exist_ok=True)

    # Create headers for files to update at each generation of each run
    save_data_to_csv(params['analysis_dir']['data']+"/data_evo_all_pop.csv", [], header = ["Run", "Generation", "Fitness", "Individual"])
    save_data_to_csv(params['analysis_dir']['data']+"/data_evo_best_inds.csv", [], header = ["Run", "Generation", "Fitness", "Individual"])
    save_data_to_csv(params['analysis_dir']['data']+"/data_evo_best_inds_ever.csv", [], header = ["Run", "Generation", "Fitness", "Individual"])

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

def write_single_gen_data(run, gen, population, analysis_dir_data):

    # Save data of all individuals in the population for this single gen
    data_all_pop = []
    for ind in population:
        data_all_pop.append([str(run), str(gen), str(ind.fitness.values[0]).strip(), str(ind).strip()]) # [0]: en fonction de len fitness Kale
    save_data_to_csv(analysis_dir_data+"/data_evo_all_pop.csv", data_all_pop)

    # Save best individual data for this single gen. NB: 'max' is the best fitness, deap manages if 'max' is maximization or minimization
    # data_best_ind = []
    best_ind = max(population, key=attrgetter("fitness"))
    data_best_ind = [[str(run), str(gen), str(best_ind.fitness.values[0]).strip(), str(best_ind).strip()]] #check
    save_data_to_csv(analysis_dir_data+"/data_evo_best_inds.csv", data_best_ind)
 
#---------------------------------------------------
    
def write_single_run_data(run, time_run, analysis_dir):

    # Save best ever individual data for this single run
    dataset_path = analysis_dir['data']+"/data_evo_best_inds.csv"
    save_filename =  analysis_dir['data']+"/data_evo_best_inds_ever.csv"
    write_best_inds_ever(dataset_path=dataset_path, save_filename=save_filename)

    # Save time information for this run
    data_time_runs = [[str(run), str(time_run)]]
    save_data_to_csv(analysis_dir['root']+"/data_all_runs/data_time_runs.csv", data_time_runs)

#---------------------------------------------------

def write_all_runs_data(analysis_dir):

    # Collect the names of the 'run' folders
    dirs = os.listdir(analysis_dir)
    analysis_single_run_dirs = [dirs[i] for i in range(len(dirs)) if dirs[i].startswith("run_")]
    analysis_single_run_dirs.sort()

    # Concatenate 'data_evo_best_inds_ever' files of each run in one single file named 'data_all_runs_best_inds_ever.csv' placed in 'data_all_runs' directory
    with open(analysis_dir+"/data_all_runs/data_all_runs_best_inds_ever.csv", 'w') as all_runs_file:
        for i, single_run_dir in enumerate(analysis_single_run_dirs):
            path = analysis_dir+"/"+single_run_dir+"/data/data_evo_best_inds_ever.csv"
            with open(path, 'r') as single_run_file:
                if i != 0:
                    next(single_run_file) # ignore the csv headers, keep just the 1st one
                all_runs_file.write(single_run_file.read())

#---------------------------------------------------

def write_best_inds_ever(dataset_path, save_filename):

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

    save_data_to_csv(save_filename, data_best_inds_ever)


###########################################################################
# Plot data functions
###########################################################################
                
def plot_single_run_data(params):

    os.makedirs(params['analysis_dir']['plots']+"/evo", exist_ok=True)

    # plot_all_pop_fitnesses_boxplot
    dataset_path = params['analysis_dir']['data']+"/data_evo_all_pop.csv"
    save_filename = params['analysis_dir']['plots']+"/evo/plot_all_pop_fitnesses_boxplot.png"
    plot_all_pop_fitnesses_boxplot(dataset_path=dataset_path, save_filename=save_filename)

    # plot_best_inds_ever
    dataset_path = params['analysis_dir']['data']+"/data_evo_best_inds_ever.csv"
    save_filename = params['analysis_dir']['plots']+"/evo/plot_best_inds_ever.png"
    plot_best_inds_ever(dataset_path=dataset_path, save_filename=save_filename)

    # plot_flag_from_file for best individuals ever in a defined range of steps
    from environments import flagAutomata
    save_dir = params['analysis_dir']['plots']+"/env"
    dataset_path = params['analysis_dir']['data']+"/data_evo_best_inds_ever.csv"
    dataset = pd.read_csv(dataset_path)
    plot_env_step_start = params['time_window_start']
    plot_env_step_end = params['time_window_end']
    steps = range(plot_env_step_start, plot_env_step_end+1) # steps is a not empty list of values, of type interval or spaced values
    for index in dataset.index[:-1]:        
        gen = dataset.loc[index, 'Generation']
        ind = dataset.loc[index, 'Individual']
        flagAutomata.plot_flag_from_file(env_eval_function_params=params['env']['eval_function_params'], data_flag_file=params['analysis_dir']['data']+"/data_env_flag.csv", gen=gen, ind=ind, steps=steps, analysis_dir_plots=save_dir)
        flagAutomata.plot_flag_fitnesses_from_file(env_eval_function_params=params['env']['eval_function_params'], data_flag_file=params['analysis_dir']['data']+"/data_env_flag.csv", gen=gen, ind=ind, analysis_dir_plots=save_dir)

    # Plot all the flag steps of the last best individual ever
    gen = dataset.loc[dataset.index[-1], 'Generation']
    ind = dataset.loc[dataset.index[-1], 'Individual']
    flagAutomata.plot_flag_from_file(env_eval_function_params=params['env']['eval_function_params'], data_flag_file=params['analysis_dir']['data']+"/data_env_flag.csv", gen=gen, ind=ind, steps=None, analysis_dir_plots=save_dir)
    flagAutomata.plot_flag_fitnesses_from_file(env_eval_function_params=params['env']['eval_function_params'], data_flag_file=params['analysis_dir']['data']+"/data_env_flag.csv", gen=gen, ind=ind, analysis_dir_plots=save_dir)

    # Animation of: plot all the flag steps of the last best individual ever
    # flag_last_best_ind_ever_dirs = os.listdir(params['analysis_dir']['plots']+"/env")
    # flag_last_best_ind_ever_dir = sorted(flag_last_best_ind_ever_dirs, key=get_gen_ind_from_file_name)[-1]
    # images = sorted(os.listdir(params['analysis_dir']['plots']+"/env/"+flag_last_best_ind_ever_dir+"/flag"))
    # path = params['analysis_dir']['plots']+"/env/"+flag_last_best_ind_ever_dir+"/flag/"
    # frames = np.stack([iio.imread(path+img) for img in images], axis = 0)
    # iio.mimwrite(params['analysis_dir']['plots']+"/env/"+flag_last_best_ind_ever_dir+".gif", frames, format='GIF', duration=0.5*len(frames), subrectangles=True)

#---------------------------------------------------

def get_gen_ind_from_file_name(file_name):
    file_name_chunks = file_name.split('_')
    gen = int(file_name_chunks[1])
    nb_ind = int(file_name_chunks[3])
    return gen, nb_ind


def plot_all_runs_data(params):

    # plot_best_inds_ever
    dataset_path = params['analysis_dir']['root']+"/data_all_runs/data_all_runs_best_inds_ever.csv"
    save_filename = params['analysis_dir']['root']+"/plots_all_runs/plot_all_runs_best_inds_ever.png"
    plot_best_inds_ever(dataset_path=dataset_path, save_filename=save_filename)

    # plot_flag_from_file for the target flag
    from environments import flagAutomata
    flagAutomata.plot_flag_from_file(env_eval_function_params=params['env']['eval_function_params'], data_flag_file=params['analysis_dir']['root']+"/data_all_runs/data_env_flag_target.csv", analysis_dir_plots=params['analysis_dir']['root']+"/plots_all_runs")

#---------------------------------------------------

# La valeur centrale du graphique est la médiane (il existe autant de valeur supérieures qu'inférieures à cette valeur dans l'échantillon).
# Les bords du rectangle sont les quartiles (Pour le bord inférieur, un quart des observations ont des valeurs plus petites et trois quart ont des valeurs plus grandes, le bord supérieur suit le même raisonnement).
# Les extrémités des moustaches sont calculées en utilisant 1.5 fois l'espace interquartile (la distance entre le 1er et le 3ème quartile).
# On peut remarquer que 50% des observations se trouvent à l'intérieur de la boîte.

# Les valeurs à l'extérieur des moustaches sont représentées par des points. On ne peut pas dire que si une observation est à l'extérieur des moustaches alors elles est une valeur aberrante. Par contre, cela indique qu'il faut étudier plus en détail cette observation.

def plot_all_pop_fitnesses_boxplot(dataset_path, save_filename):

    dataset = pd.read_csv(dataset_path)

    sns.set_theme()
    sns.boxplot(x='Generation', y='Fitness', data=dataset, color='skyblue')

    plt.title("Fitnesses over generations\nall individuals generated", fontsize=14)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)

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

    plt.title("Fitnesses over generations\nbest individuals ever", fontsize=14)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)
    plt.legend()

    plt.savefig(save_filename)
    plt.clf()
    plt.close()

#---------------------------------------------------

# def build_animation(images_source_path, images_source_file_name, animation_path, animation_file_name):
#     nb_images = os.listdir(images_source_path)
#     frames = np.stack([iio.imread(images_source_path+images_source_file_name+str(i)+".png") for i in range(len(nb_images))], axis = 0)
#     os.makedirs(animation_path, exist_ok=True)
#     iio.mimwrite(animation_path+animation_file_name, frames, duration=0.5, subrectangles = True) # duration 0.5 = ~2fps gif


























def plot_bla_bla(env_boundaries_2D, g, ind_x, ind_y, eval_function, best_ind): # amelioration : eviter de recalculer la matrice
 
    # Create a grid of points
    x = np.linspace(env_boundaries_2D['theta_space_min_x'], env_boundaries_2D['theta_space_max_x'], 100)
    y = np.linspace(env_boundaries_2D['theta_space_min_y'], env_boundaries_2D['theta_space_max_y'], 100)
    X, Y = np.meshgrid(x, y)

    # Compute the function values for each point in the grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = eval_function(np.array([X[i, j], Y[i, j]]))[0]

    # Plot
    # plt.contourf(X, Y, Z, cmap='viridis', levels=100)
    plt.imshow(Z, extent=[env_boundaries_2D['theta_space_min_x'], env_boundaries_2D['theta_space_max_x'], env_boundaries_2D['theta_space_min_y'], env_boundaries_2D['theta_space_max_y']], origin='lower', cmap='viridis', norm=LogNorm())
    plt.scatter(ind_x, ind_y, c='r', linewidths=1)
    plt.scatter(best_ind[0], best_ind[1], c='g', linewidths=1)

    # Colorbar
    cbar = plt.colorbar()
    cbar.set_label('Evaluation function value', fontsize=9)

    # Set x-axis limits
    plt.xlim(env_boundaries_2D['theta_space_min_x'], env_boundaries_2D['theta_space_max_x'])
    plt.ylim(env_boundaries_2D['theta_space_min_y'], env_boundaries_2D['theta_space_max_y'])
    
    plt.xticks([env_boundaries_2D['theta_space_min_x'], 0, env_boundaries_2D['theta_space_max_x']]) # x axis
    plt.yticks([env_boundaries_2D['theta_space_min_y'], 0, env_boundaries_2D['theta_space_max_y']]) # y axis

    # Label axes
    plt.xlabel('x')
    plt.ylabel('y')

    plt.title("cmaes optimization on "+str(eval_function.__name__)+", gen="+str(g), fontsize=14)

    # Add a legend
    # plt.legend()

    # if g%50==0 or g==249:
    plt.savefig("src/a"+str(g)+".png")

    # Clear the figure before the next plot
    plt.clf()
    plt.close()




# def plot(data_path):
#     df_archive = pd.read_csv(data_path+"/data_all_pop.csv")
#     plot_archive_fitnesses_curves(df_archive, data_path+"/plots/plot_archive_fitnesses_curves")


# def plot_archive_fitnesses_curves(dataset, save_dir=None):
#     # algo_name = simulation_params['algo']
#     # with_sampling = simulation_params['sampling_bool']
#     # with_noise = simulation_params['noise_params']['noise_bool']
#     # show_plot_time = simulation_params['show_plot_time']

#     time_gens = []
#     medians = []
#     quartiles_25 = []
#     means = []
#     quartiles_75 = []

#     generations = dataset['Generation'].unique()
#     for gen in generations: 

#         fitnesses_gen = dataset.loc[dataset.Generation==gen, 'Fitness'].values.tolist() # check
#         # time_gens.append(dataset_time.loc[(dataset_time.Generation==gen), ['Time_gen']].values.tolist()[0][0])
#         means.append(np.mean(fitnesses_gen))
#         q25, q50, q75 = np.quantile(fitnesses_gen, [0.25, 0.5, 0.75])
#         quartiles_25.append(q25)
#         medians.append(q50)
#         quartiles_75.append(q75)

    
#     # plt.plot(time_gens, means, color='black', marker='.', label='mean')
#     # plt.plot(time_gens, medians, color='red', marker='.', label='median')
#     # plt.fill_between(time_gens, quartiles_25, quartiles_75, alpha=0.3)
        
#     plt.plot(generations, means, color='black', marker='.', label='mean')
#     plt.plot(generations, medians, color='red', marker='.', label='median')
#     plt.fill_between(generations, quartiles_25, quartiles_75, alpha=0.3)

#     # plt.suptitle(str(algo_name)+" - Archive Fitnesses ('final' evaluations only)")
#     # plt.title("with sampling: "+str(with_sampling)+";   with noise: "+str(with_noise), fontsize=9)
#     plt.xlabel("Time (s)")
#     plt.ylabel("Fitness")
#     # plt.legend()
    
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#         plt.savefig(save_dir+"/archive_fitnesses_curves.png")
    
#     plt.close()


