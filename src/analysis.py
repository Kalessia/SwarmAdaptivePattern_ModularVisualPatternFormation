import os
from datetime import datetime

import numpy as np
import pandas as pd
from operator import attrgetter

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
import seaborn as sns

import csv


analysis_dir = ""


def init_all_runs_analysis(params):

    # Analysis folders creation
    global analysis_dir
    analysis_dir = "simulationAnalysis/" + params['env_name'] + "_simulation_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(analysis_dir, exist_ok=True)
    params['analysis_dir'] = analysis_dir

    return params


def init_one_run_analysis(run, params):

    # Analysis folders creation
    params['analysis_dir_run'] = params['analysis_dir'] + "/run_" + str(run)
    params['analysis_dir_data'] = params['analysis_dir'] + "/run_" + str(run) + "/data"
    params['analysis_dir_plots'] = params['analysis_dir'] + "/run_" + str(run) + "/plots"
    os.makedirs(params['analysis_dir_data'], exist_ok=True)
    os.makedirs(params['analysis_dir_plots'], exist_ok=True)

    save_data_to_csv(params['analysis_dir_data'] + "/data_evo_all_pop.csv", [], header = ["Run", "Generation", "Fitness", "Individual"])
    save_data_to_csv(params['analysis_dir_data'] + "/data_evo_best_inds.csv", [], header = ["Run", "Generation", "Fitness", "Individual"])

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



def write_single_run_data(run, gen, population, analysis_dir_data):

    data_all_pop = [] # data_all_pop: Generation, Fitness, Individual
    data_best_ind = [] # data_best_ind: Generation, Fitness, Individual
    for ind in population:
        data_all_pop.append([str(run), str(gen), str(ind.fitness.values[0]).strip(), str(ind).strip()]) # [0]: en fonction de len fitness Kale

    best_ind = max(population, key=attrgetter("fitness")) # 'max' is the best fitness, deap manage if 'max' is maximization or minimization
    data_best_ind.append([str(run), str(gen), str(best_ind.fitness.values[0]).strip(), str(best_ind).strip()])

    save_data_to_csv(analysis_dir_data + "/data_evo_all_pop.csv", data_all_pop)
    save_data_to_csv(analysis_dir_data + "/data_evo_best_inds.csv", data_best_ind)



def write_all_runs_data(analysis_dir):

    analysis_single_run_dirs = os.listdir(analysis_dir)
    analysis_single_run_dirs.remove('run_params.json')
    analysis_single_run_dirs.sort()

    data_all_runs_best_inds = []

    for single_run_dir in analysis_single_run_dirs:

        path = analysis_dir + "/" + single_run_dir + "/data/data_evo_best_inds.csv"
        df = pd.read_csv(path)

        single_run_best_fitness = df['Fitness'].min()
        single_run_best_inds = df.loc[(df.Fitness==single_run_best_fitness)]

        for index in single_run_best_inds.index:
            data_all_runs_best_inds.append([str(df.loc[index, 'Run']).strip(), str(df.loc[index, 'Generation']).strip(), str(df.loc[index, 'Fitness']).strip(), str(df.loc[index, 'Individual']).strip()])


    os.makedirs(analysis_dir+"/data_all_runs", exist_ok=True)
    save_data_to_csv(analysis_dir+"/data_all_runs/data_all_runs_best_inds.csv", data_all_runs_best_inds, header = ["Run", "Generation", "Fitness", "Individual"])

#---------------------------------------------------
    
def plot_single_run_data(analysis_dir_data, analysis_dir_plots):

    os.makedirs(analysis_dir_plots+"/evo", exist_ok=True)

    path = analysis_dir_data+"/data_evo_all_pop.csv"
    dataset = pd.read_csv(path)
    plot_fitnesses_boxplot(dataset, analysis_dir_plots+"/evo")

    path = analysis_dir_data+"/data_evo_best_inds.csv"
    dataset = pd.read_csv(path)
    plot_best_fitness_ever(dataset, analysis_dir_plots+"/evo")

#---------------------------------------------------

# La valeur centrale du graphique est la médiane (il existe autant de valeur supérieures qu'inférieures à cette valeur dans l'échantillon).
# Les bords du rectangle sont les quartiles (Pour le bord inférieur, un quart des observations ont des valeurs plus petites et trois quart ont des valeurs plus grandes, le bord supérieur suit le même raisonnement).
# Les extrémités des moustaches sont calculées en utilisant 1.5 fois l'espace interquartile (la distance entre le 1er et le 3ème quartile).
# On peut remarquer que 50% des observations se trouvent à l'intérieur de la boîte.

# Les valeurs à l'extérieur des moustaches sont représentées par des points. On ne peut pas dire que si une observation est à l'extérieur des moustaches alors elles est une valeur aberrante. Par contre, cela indique qu'il faut étudier plus en détail cette observation.

def plot_fitnesses_boxplot(dataset, analysis_dir_plots):

    # list_gen_fitnesses = []
    # generations = dataset['Generation'].unique()
    # for gen in generations:
    #     dataset_gen = dataset.loc[dataset.Generation==gen]
    #     list_gen_fitnesses.append(dataset_gen['Fitness'].tolist())

    # plt.boxplot(list_gen_fitnesses, positions=generations)
    sns.set_theme()
    sns.boxplot(x='Generation', y='Fitness', data=dataset, color='skyblue')
    
    # plt.suptitle("Fitnesses")
    # plt.xlabel("Generation")
    # plt.ylabel("Fitness")

    plt.savefig(analysis_dir_plots + "/data_all_pop_fitnesses_boxplot.png")
    plt.clf()
    plt.close()


def plot_best_fitness_ever(dataset, analysis_dir_plots):

    # generations = dataset['Generation'].to_list()
    # fitnesses = dataset['Fitness'].to_list()
    generations = []
    best_fitnesses_ever = [] # to plot
    data_best_inds_best_fitness_ever = [] # to write
    best_fit = np.inf
    # for fit in fitnesses:
    #     if fit < best_fit:
    #         best_fit = fit
    #     best_fitnesses_ever.append(best_fit)

    for index in dataset.index:
        gen = dataset.loc[index, 'Generation']
        fit = dataset.loc[index, 'Fitness']

        generations.append(gen)
        if fit < best_fit:
            best_fit = fit
            data_best_inds_best_fitness_ever.append([str(gen).strip(), str(fit).strip(), str(dataset.loc[index, 'Individual']).strip()])
        best_fitnesses_ever.append(best_fit)

    save_data_to_csv(analysis_dir_plots+"/data_best_inds_best_fitness_ever.csv", data_best_inds_best_fitness_ever, header = ["Generation", "Fitness", "Individual"])

    plt.step(generations, best_fitnesses_ever, where='post') # plot

    plt.suptitle("Fitnesses")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")

    plt.savefig(analysis_dir_plots+"/plot_best_inds_best_fitness_ever.png")
    plt.clf()
    plt.close()


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

    plt.title("cmaes optimization on " + str(eval_function.__name__) + ", gen=" + str(g), fontsize=9)

    # Add a legend
    # plt.legend()

    # if g%50==0 or g==249:
    plt.savefig("src/a" + str(g) + ".png")

    # Clear the figure before the next plot
    plt.clf()
    plt.close()


# def plot(data_path):
#     df_archive = pd.read_csv(data_path + "/data_all_pop.csv")
#     plot_archive_fitnesses_curves(df_archive, data_path+"/plots/plot_archive_fitnesses_curves")


# def plot_archive_fitnesses_curves(dataset, save_dir=None):
#     # algo_name = run_params['algo']
#     # with_sampling = run_params['sampling_bool']
#     # with_noise = run_params['noise_params']['noise_bool']
#     # show_plot_time = run_params['show_plot_time']

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

#     # plt.suptitle(str(algo_name) + " - Archive Fitnesses ('final' evaluations only)")
#     # plt.title("with sampling: " + str(with_sampling) + ";   with noise: " + str(with_noise), fontsize=9)
#     plt.xlabel("Time (s)")
#     plt.ylabel("Fitness")
#     # plt.legend()
    
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#         plt.savefig(save_dir + "/archive_fitnesses_curves.png")
    
#     plt.close()


