import pandas as pd
import numpy as np

import csv

from nn import NeuralNetwork


###########################################################################
# Environment initialization
###########################################################################

def check_params_validity(params):

    exit_bool = False

    params['nb_runs'] = int(params['nb_runs'])
    params['time_steps'] = int(params['time_steps'])
    params['switch_step'] = int(params['switch_step'])
    params['nb_repetitions'] = int(params['nb_repetitions'])
    params['with_parallelization_nb_free_cores'] = int(params['with_parallelization_nb_free_cores'])

    if params['nb_runs'] <= 0 or params['time_steps'] <= 0 or params['nb_repetitions'] <= 0:
        print(f"Error in swarm_initializations.py - Parameters nb_runs, time_steps and nb_repetitions must be > 0")
        exit_bool = True

    if params['setup_ind_consistency']['setup_ind_consistency_bool']:
        setup_ind_consistency_options = []
        if params['setup_ind_consistency']['setup_ind_consistency_random_init_states_bool']:
            setup_ind_consistency_options.append("setup_ind_consistency_random_init_states")
        if params['setup_ind_consistency']['setup_ind_consistency_random_async_update_states_bool']:
            setup_ind_consistency_options.append("setup_ind_consistency_random_async_update_states")
        params['setup_ind_consistency']['setup_ind_consistency_options'] = setup_ind_consistency_options

    if params['setup_noise']['setup_noise_bool'] and not isinstance(params['setup_noise']['setup_noise_std_ticks'], list):
        print(f"Error in swarm_initializations.py - Parameter setup_noise_std_ticks must be a list")
        exit_bool = True

    if params['setup_permutation']['setup_permutation_bool'] and not isinstance(params['setup_permutation']['setup_permutation_ticks_percent'], list):
        print(f"Error in swarm_initializations.py - Parameter setup_permutation_ticks_percent must be a list")
        exit_bool = True

    if params['setup_deletion']['setup_deletion_bool'] and not isinstance(params['setup_deletion']['setup_deletion_ticks_percent'], list):
        print(f"Error in swarm_initializations.py - Parameter setup_deletion_ticks_percent must be a list")
        exit_bool = True

    if exit_bool:
        print("learning_main stopped. Please correct the entry parameter in swarm_params.json before restart.")
        exit()

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

def copy_params_from_learning(learning_params, swarm_params):
    
    swarm_params['learning_nb_runs'] = learning_params['nb_runs']
    swarm_params['grid_nb_rows'] = learning_params['grid_nb_rows']
    swarm_params['grid_nb_cols'] = learning_params['grid_nb_cols']
    swarm_params['init_cell_state_value'] = learning_params['init_cell_state_value']

    swarm_params['controller'] = NeuralNetwork(nb_neuronsPerInputs=learning_params['nb_neuronsPerInputs'],
                                        nb_hiddenLayers=learning_params['nb_hiddenLayers'],
                                        nb_neuronsPerHidden=learning_params['nb_neuronsPerHidden'],
                                        nb_neuronsPerOutputs=learning_params['nb_neuronsPerOutputs'])
    
    swarm_params['learning_mode'] = learning_params['learning_mode']
    swarm_params['learning_with_noise_std'] = learning_params['learning_with_noise_std']

    swarm_params['best_ind_ever'], swarm_params['best_ind_ever_fitness'] = get_best_ind_ever(dataset_path=learning_params['analysis_dir']['root']+"/data_all_runs/data_evo_all_runs_best_ind_per_run.csv")
    swarm_params['flag_target'] = get_flag_target(dataset_path=learning_params['analysis_dir']['root']+"/data_all_runs/data_env_flag_target.csv")

    return swarm_params


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