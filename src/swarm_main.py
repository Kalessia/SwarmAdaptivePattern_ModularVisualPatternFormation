import os
import time
import shutil
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np

import json

from src.swarm_environments import swarmGrid
from src.swarm_analysis import *

sep = "\n################################################\n"




def swarm_simulation(run, best_ind, best_ind_run, simulation_params):
    print(f"swarm_simulation run n.{run}, best_ind_{best_ind_run} [{best_ind[0]}, ...] - Starting")

    # Initializations
    time_run = time.time()
    simulation_params = init_one_run_analysis(run, simulation_params)
    env = swarmGrid(grid_nb_rows=simulation_params['grid_nb_rows'],
                    grid_nb_cols=simulation_params['grid_nb_cols'],
                    nb_neuronsPerInputs=simulation_params['controller']['nb_neuronsPerInputs'],
                    nb_hiddenLayers=simulation_params['controller']['nb_hiddenLayers'],
                    nb_neuronsPerHidden=simulation_params['controller']['nb_neuronsPerHidden'],
                    nb_neuronsPerOutputs=simulation_params['controller']['nb_neuronsPerOutputs'],
                    agent_controller_weights=best_ind,
                    flag_target=simulation_params['flag_target'])
    
    # kale on defini un step en simuparams a partir du quel on change de regime
    time_steps = simulation_params['time_steps']
    switch_step = simulation_params['switch_step']

    # cas 0: test de l'individu
    for n in range(simulation_params['nb_repetitions']):
        env.setup_test_ind(run, n, time_steps, switch_step, best_ind_run, best_ind, simulation_params)

    # cas 1: introduire du bruit continu de differente entité
    # param: entités de bruit souhaités
    # initialize env: init grid? ou nouvelle instance
    # quand on calcule les entrées du reseau in compute_agent_state(self, agent), add bruit selon l entite demandée en parametre
    # repeter le procedé N fois, un plot par N avec barre au 'step de changement regime' et un plot complexive box plot, du coup data "Run, step, fitness"

    noise_ticks = simulation_params['noise_ticks']
    if noise_ticks:
        for n in range(simulation_params['nb_repetitions']):
            for tick in noise_ticks:
                env.setup_noise1(run, n, tick, time_steps, switch_step, best_ind_run, best_ind, simulation_params)

    # cas 2: introduire du bruit continu de differente entité
    # pareil, mais avec noise type 2

    # noise_ticks = simulation_params['noise_ticks']
    # if noise_ticks:
    #     for n in range(simulation_params['nb_repetitions']):
    #         for tick in noise_ticks:
    #             env.setup_noise2(run, n, tick, time_steps, switch_step, best_ind_run, best_ind, simulation_params)
           
    # cas 2: eliminer n agents
    # params: liste de nombre N de agents à eliminer, sans retirage, choisi au hazard
    # initialize env: init grid? ou nouvelle instance
    # on elimine 2 agents des listes, on refrech les voisins
    # repeter le procedé N fois, un plot par N avec barre au 'step de changement regime' et un plot complexive box plot, du coup data "Run, step, fitness"
    # les agents enleves seront reinseres

    deletion_ticks = simulation_params['deletion_ticks']
    if deletion_ticks:
        for n in range(simulation_params['nb_repetitions']):

            # deletion_ticks = random.sample(range(env.size-1), simulation_params['nb_agents_to_delete']) # random.sample choose elements without replacement (don't allow duplicates)

            # if simulation_params['nb_agents_to_delete'] >= env.grid_size:
            #     print("gneee conlacca")
            #     exit()

            deleted_map_pos_agent = {}
            for tick in deletion_ticks:
                deleted_map_pos_agent = env.setup_deletion(run, n, tick, time_steps, switch_step, best_ind_run, best_ind, deleted_map_pos_agent, simulation_params)
            
            env.restore_deleted_agents(deleted_map_pos_agent=deleted_map_pos_agent)



    
    # flags = []

    # time_steps = simulation_params['time_steps']

    # for step in range(time_steps):

    #     if step == 100:
    #         # print(env.grid_map_pos_agent)
    #         agent1 = env.grid_map_pos_agent[tuple((0,0))]
    #         agent2 = env.grid_map_pos_agent[tuple((2,4))]
    #         env.exchange_agents(agent1, agent2)

    #     flags.append(env.get_flag_from_grid())
    #     env.plot_flag(run, step, analysis_dir_plots=simulation_params['analysis_dir']['plots']+"/env")
    #     env.step()



    time_run = time.time() - time_run
    print(f"swarm_simulation run n.{run}, best_ind_{best_ind_run} [{best_ind[0]}, ...] - Execution time:", time_run, "seconds") # kale change save time for each run
    # write_single_run_data(env, run, time_run, time_steps, flags, analysis_dir=simulation_params['analysis_dir'])

    # Plot data for one single run
    if simulation_params['plot_analysis_bool']:
        plot_single_run_data(run, simulation_params)

    return simulation_params

#---------------------------------------------------

def get_best_ind_per_run_dict(dataset_path=None):

    best_ind_per_run_dict = {}
    dataset = pd.read_csv(dataset_path)

    runs = sorted(dataset['Run'].unique())
    for run in runs:
        ind = dataset.loc[(dataset.Run==run), 'Individual'].values.tolist()[0]
        ind = str(ind).replace('[', '').replace(']', '').strip()
        ind = list(np.asarray(ind.split(','), dtype=np.float64))
        best_ind_per_run_dict[run] = ind

    return best_ind_per_run_dict

#---------------------------------------------------

def get_best_ind_ever(dataset_path=None):

    dataset = pd.read_csv(dataset_path)

    best_ind_ever_fitness = dataset['Fitness'].min()
    best_ind_ever = dataset.loc[dataset.Fitness==best_ind_ever_fitness, 'Individual'].values.tolist()[0]
    best_ind_ever = str(best_ind_ever).replace('[', '').replace(']', '').strip()
    best_ind_ever = list(np.asarray(best_ind_ever.split(','), dtype=np.float32))

    return best_ind_ever, best_ind_ever_fitness

#---------------------------------------------------

def get_flag_target(dataset_path=None):

    dataset = pd.read_csv(dataset_path)
    flag_target = dataset.iloc[0]['Flag']
    flag_target = str(flag_target).replace('[', '').replace(']', '').strip()
    flag_target = list(np.asarray(flag_target.split(','), dtype=np.float32))

    return flag_target

#---------------------------------------------------

def worker(task):
    run, best_ind, best_ind_run, simulation_params = task
    # print("launching with ", run, best_ind, simulation_params, "\n")
    return swarm_simulation(run=run, best_ind=best_ind, best_ind_run=best_ind_run, simulation_params=simulation_params)

#---------------------------------------------------

def parallelize_processes(nb_runs, best_ind_per_run_dict, simulation_params):
    
    # Create a queue of tasks to execute
    task_queue = []
    for run in range(nb_runs):
        for best_ind_run in best_ind_per_run_dict:
            task_queue.append((run, best_ind_per_run_dict[best_ind_run], best_ind_run, simulation_params.copy()))
    # task_queue = [(run, simulation_params.copy()) for run in range(nb_runs)]

    # Create a Pool with the number of available cores
    available_cores = cpu_count() - simulation_params['with_parallelization_nb_free_cores']
    with Pool(processes=min(nb_runs, available_cores)) as pool:
        results_list = pool.map(worker, task_queue) # results_list contains the 'simulation_params' of each run, respecting the ascending order from 0 to nb_runs

    return results_list[-1]


###########################################################################
# MAIN
###########################################################################

if (__name__ == "__main__"):

    # Initialization
    # analysis_dir = sys.argv[1]
    analysis_dir = "simulationAnalysis/flag_automata_simulation_2024-04-16_02-25-46_circle_9x9"
    with open(analysis_dir+"/learning/learning_params.json", "r") as f:
        learning_params = json.load(f)

    if os.path.exists(analysis_dir+"/simulation"):
        shutil.rmtree(analysis_dir+"/simulation")
    
    with open("src/simulation_params.json", "r") as f:
        simulation_params = json.load(f)

    simulation_params = init_all_runs_analysis(learning_params['analysis_dir']['root'], simulation_params)
    simulation_params['grid_nb_rows'] = learning_params['automata_nb_rows'] 
    simulation_params['grid_nb_cols'] = learning_params['automata_nb_cols']

    simulation_params['controller'] = learning_params['env']['eval_function_params']['controller']

    best_ind_per_run_dict = get_best_ind_per_run_dict(dataset_path=learning_params['analysis_dir']['root']+"/learning/data_all_runs/data_evo_all_runs_best_ind_per_run.csv")
    simulation_params['best_ind_ever'], simulation_params['best_ind_ever_fitness'] = get_best_ind_ever(dataset_path=learning_params['analysis_dir']['root']+"/learning/data_all_runs/data_evo_all_runs_best_ind_per_run.csv")
    simulation_params['flag_target'] = get_flag_target(dataset_path=learning_params['analysis_dir']['root']+"/learning/data_all_runs/data_env_flag_target.csv")

    if simulation_params['with_parallelization_bool']:
        # Launch the Swarm simulation with parallelization over runs
        simulation_params = parallelize_processes(simulation_params['nb_runs'], best_ind_per_run_dict, simulation_params)
    else:
        # Launch the Swarm simulation sequentially
        for run in range(simulation_params['nb_runs']):
            for best_ind_run in best_ind_per_run_dict:
                simulation_params = swarm_simulation(run=run, best_ind=best_ind_per_run_dict[best_ind_run], best_ind_run=best_ind_run, simulation_params=simulation_params.copy())

    # Save a trace of used parameters for this simulation in simulation_params.json
    simulation_params['best_ind_ever'] = np.array(simulation_params['best_ind_ever']).tolist()
    simulation_params['flag_target'] = np.array(simulation_params['flag_target']).tolist()
    
    with open(simulation_params['analysis_dir']['root'] + "/simulation/simulation_params.json", "w") as f:
        json.dump({k:v for k,v in simulation_params.items()}, f, indent=2)

    # # Plot data for all the runs
    # if simulation_params['plot_analysis_bool']:
    #     write_all_runs_data(simulation_params['analysis_dir']['root'])
    #     plot_all_runs_data(simulation_params)