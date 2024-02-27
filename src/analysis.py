import os
import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import csv


analysis_dir = ""


def init_analysis(params):

    # Analysis folders creation
    global analysis_dir
    analysis_dir = "simulationAnalysis/" + params['env_name'] + "_simulation_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(analysis_dir + "/data", exist_ok=True)
    os.makedirs(analysis_dir + "/plots", exist_ok=True)
    params['analysis_dir'] = analysis_dir

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



def write_EvoAlgorithm_data(gen, population):

    # population data: Generation, Behavior_descriptor_x,Behavior_descriptor_y, Novelty, Fitness, Individual
    data_all_pop = []
    for ind in population:
        data_all_pop.append([str(gen), str(ind.fitness.values[0]).strip(), str(ind).strip()]) # [0]: en fonction de len fitness Kale

    save_data_to_csv(analysis_dir + "/data/data_all_pop.csv", data_all_pop)

    # data_best_inds_fit = save_data_to_csv(analysis_dir + "/data_best_inds_fit.csv", data_best_inds_fit)







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


def plot(data_path):
    df_archive = pd.read_csv(data_path + "/data_all_pop.csv")
    plot_archive_fitnesses_curves(df_archive, data_path+"/plots/plot_archive_fitnesses_curves")


def plot_archive_fitnesses_curves(dataset, save_dir=None):
    # algo_name = run_params['algo']
    # with_sampling = run_params['sampling_bool']
    # with_noise = run_params['noise_params']['noise_bool']
    # show_plot_time = run_params['show_plot_time']

    time_gens = []
    medians = []
    quartiles_25 = []
    means = []
    quartiles_75 = []

    generations = dataset['Generation'].unique()
    for gen in generations: 

        fitnesses_gen = dataset.loc[dataset.Generation==gen, 'Fitness'].values.tolist() # check
        # time_gens.append(dataset_time.loc[(dataset_time.Generation==gen), ['Time_gen']].values.tolist()[0][0])
        means.append(np.mean(fitnesses_gen))
        q25, q50, q75 = np.quantile(fitnesses_gen, [0.25, 0.5, 0.75])
        quartiles_25.append(q25)
        medians.append(q50)
        quartiles_75.append(q75)

    
    # plt.plot(time_gens, means, color='black', marker='.', label='mean')
    # plt.plot(time_gens, medians, color='red', marker='.', label='median')
    # plt.fill_between(time_gens, quartiles_25, quartiles_75, alpha=0.3)
        
    plt.plot(generations, means, color='black', marker='.', label='mean')
    plt.plot(generations, medians, color='red', marker='.', label='median')
    plt.fill_between(generations, quartiles_25, quartiles_75, alpha=0.3)

    # plt.suptitle(str(algo_name) + " - Archive Fitnesses ('final' evaluations only)")
    # plt.title("with sampling: " + str(with_sampling) + ";   with noise: " + str(with_noise), fontsize=9)
    plt.xlabel("Time (s)")
    plt.ylabel("Fitness")
    # plt.legend()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir + "/archive_fitnesses_curves.png")
    
    plt.close()


#---------------------------------------------------