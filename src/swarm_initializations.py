import pandas as pd
import numpy as np

import csv

from nn import NeuralNetwork


###########################################################################
# Environment initialization
###########################################################################

def check_params_validity(grid_size, params):

    exit_bool = False

    params['nb_runs'] = int(params['nb_runs'])
    params['time_steps'] = int(params['time_steps'])
    params['time_window_start'] = int(params['time_window_start'])
    params['time_window_end'] = int(params['time_window_end'])
    params['switch_step'] = int(params['switch_step'])
    params['nb_repetitions'] = int(params['nb_repetitions'])
    params['with_parallelization_nb_free_cores'] = int(params['with_parallelization_nb_free_cores'])

    if params['nb_runs'] <= 0 or params['time_steps'] <= 0 or params['nb_repetitions'] <= 0:
        print(f"Error in swarm_initializations.py - Parameters nb_runs, time_steps and nb_repetitions must be > 0")
        exit_bool = True

    if params['setup_ind_consistency']['setup_ind_consistency_bool']:
        setup_ind_consistency_options = []
        if params['setup_ind_consistency']['setup_ind_consistency_learning_conditions_bool']:
            setup_ind_consistency_options.append("setup_ind_consistency_learning_conditions")
        if params['setup_ind_consistency']['setup_ind_consistency_random_init_states_bool']:
            setup_ind_consistency_options.append("setup_ind_consistency_random_init_states")
        if params['setup_ind_consistency']['setup_ind_consistency_random_async_update_states_bool']:
            setup_ind_consistency_options.append("setup_ind_consistency_random_async_update_states")
        params['setup_ind_consistency']['setup_ind_consistency_options'] = setup_ind_consistency_options

    if params['setup_noise']['setup_noise_bool'] and not isinstance(params['setup_noise']['setup_noise_std_ticks'], list):
        print(f"Error in swarm_initializations.py - Parameter setup_noise_std_ticks must be a list")
        exit_bool = True

    if params['setup_permutation']['setup_permutation_bool']:
        if params['setup_permutation']['setup_permutation_ticks_units'] is None:
            if not isinstance(params['setup_permutation']['setup_permutation_ticks_percent'], list):
                print(f"Error in swarm_initializations.py - Parameter setup_permutation_ticks_percent must be a list")
                params['setup_permutation']['setup_permutation_ticks_percent'] = []
                exit_bool = True
            permutation_ticks = [int(grid_size*tick_percent) for tick_percent in params['setup_permutation']['setup_permutation_ticks_percent']]
        else:
            if params['setup_permutation']['setup_permutation_ticks_percent'] is not None:
                print(f"Error in learning_initializations.py - Either 'setup_permutation_ticks_units' or 'setup_permutation_ticks_percent' must be null.")
                exit_bool = True
            if not isinstance(params['setup_permutation']['setup_permutation_ticks_units'], list):
                print(f"Error in swarm_initializations.py - Parameter setup_permutation_ticks_units must be a list")
                params['setup_permutation']['setup_permutation_ticks_units'] = []
                exit_bool = True
            permutation_ticks = [int(tick_unit) for tick_unit in params['setup_permutation']['setup_permutation_ticks_units']]
        params['setup_permutation']['permutation_ticks'] = [val if val % 2 == 0 else val - 1 for val in permutation_ticks] # permutation_ticks must be even

    if params['setup_deletion']['setup_deletion_bool']:
        if params['setup_deletion']['setup_deletion_ticks_units'] is None:
            if not isinstance(params['setup_deletion']['setup_deletion_ticks_percent'], list):
                print(f"Error in swarm_initializations.py - Parameter setup_deletion_ticks_percent must be a list")
                params['setup_deletion']['setup_deletion_ticks_percent'] = []
                exit_bool = True
            params['setup_deletion']['deletion_ticks'] = [int(grid_size*tick_percent) for tick_percent in params['setup_deletion']['setup_deletion_ticks_percent']]
        else:
            if params['setup_deletion']['setup_deletion_ticks_percent'] is not None:
                print(f"Error in learning_initializations.py - Either 'setup_permutation_ticks_units' or 'setup_permutation_ticks_percent' must be null.")
                exit_bool = True
            if not isinstance(params['setup_deletion']['setup_deletion_ticks_units'], list):
                print(f"Error in swarm_initializations.py - Parameter setup_deletion_ticks_units must be a list")
                params['setup_deletion']['setup_deletion_ticks_units'] = []
                exit_bool = True
            params['setup_deletion']['deletion_ticks'] = [int(tick_unit) for tick_unit in params['setup_deletion']['setup_deletion_ticks_units']]

    if params['setup_sliding_puzzle']['setup_sliding_puzzle_bool']:
        if not isinstance(params['setup_sliding_puzzle']['setup_sliding_puzzle_ticks_percent'], list) \
        or not isinstance(params['setup_sliding_puzzle']['setup_sliding_puzzle_probas_move'], list):
            print(f"Error in swarm_initializations.py - Parameter setup_sliding_puzzle_ticks_percent and setup_sliding_puzzle_probas_move must be lists")
            params['setup_sliding_puzzle']['setup_sliding_puzzle_ticks_percent'] = []
            exit_bool = True
        params['setup_sliding_puzzle']['deletion_ticks'] = [int(grid_size*tick_percent) for tick_percent in params['setup_sliding_puzzle']['setup_sliding_puzzle_ticks_percent']]

        for p, proba_move in enumerate(params['setup_sliding_puzzle']['setup_sliding_puzzle_probas_move']):
            params['setup_sliding_puzzle']['setup_sliding_puzzle_probas_move'][p] = float(proba_move)
            if params['setup_sliding_puzzle']['setup_sliding_puzzle_probas_move'][p] < 0.0 or params['setup_sliding_puzzle']['setup_sliding_puzzle_probas_move'][p] > 1.0:
                print(f"Error in swarm_initializations.py - Each parameter in setup_sliding_puzzle_probas_move must be in [0.0, 1.0]")
                exit_bool = True
    
    if exit_bool:
        print("learning_main stopped. Please correct the entry parameter in swarm_params.json before restart.")
        exit()

#---------------------------------------------------

def get_best_ind_per_run_dict(dataset_path):

    best_ind_per_run_dict = {}
    dataset = pd.read_csv(dataset_path)

    learning_phase = dataset['Learning_phase'].max() # the last phase CHECK MEILLEURE FACON DE LE FAIRE FONCTIONNER
    runs = sorted(dataset['Run'].unique())
    for run in runs:
        ind = dataset.loc[(dataset.Run==run) & (dataset.Learning_phase==learning_phase), 'Individual'].values.tolist()[0]
        ind = str(ind).replace('[', '').replace(']', '').strip()
        ind = list(np.asarray(ind.split(','), dtype=np.float64))
        best_ind_per_run_dict[run] = ind

    return best_ind_per_run_dict

#---------------------------------------------------

def get_best_ind_per_run_per_phase_dict(dataset_path=None):

    best_ind_per_run_per_phase_dict = {}
    dataset = pd.read_csv(dataset_path)

    runs = sorted(dataset['Run'].unique())
    phases = sorted(dataset['Learning_phase'].unique())
    for run in runs:
        best_ind_per_run_per_phase_dict[run] = []
        for phase in phases:
            ind = dataset.loc[(dataset.Run == run) & (dataset.Learning_phase == phase), 'Individual'].values.tolist()[0]
            ind = str(ind).replace('[', '').replace(']', '').strip()
            ind = list(np.asarray(ind.split(','), dtype=np.float64))
            best_ind_per_run_per_phase_dict[run].append(ind)

    return best_ind_per_run_per_phase_dict

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
    
    swarm_params['learning_nb_runs'] = learning_params['evolutionary_settings']['nb_runs']
    swarm_params['grid_nb_rows'] = learning_params['grid']['grid_nb_rows']
    swarm_params['grid_nb_cols'] = learning_params['grid']['grid_nb_cols']
    swarm_params['init_cell_state_value'] = learning_params['grid']['init_cell_state_value']
    swarm_params['flag_pattern'] = learning_params['grid']['flag_pattern']
    
    swarm_params['controller'] = NeuralNetwork(input_size=learning_params['nn_controller']['nb_neuronsPerInputs'],
                                        hidden_layers=learning_params['nn_controller']['hidden_layers'],
                                        output_size=learning_params['nn_controller']['nb_neuronsPerOutputs'],
                                        activation_function='tanh')

    swarm_params['learning_modes'] = learning_params['learning_modes']
    swarm_params['learning_with_noise_std'] = learning_params['learning_options']['learning_with_noise']['learning_with_noise_std']

    # if swarm_params['setup_sliding_puzzle_phase1_VS_phase2']['setup_sliding_puzzle_phase1_VS_phase2_bool'] and learning_params['evolutionary_settings']['env_name'] == "sliding_puzzle_incremental":
    if learning_params['evolutionary_settings']['env_name'] == "sliding_puzzle_incremental":
        swarm_params['setup_sliding_puzzle_phase1_VS_phase2'].update({
            'learning_bool': True,
            'learning_nb_deletions_percent': learning_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_percent'],
            'learning_ticks_units': learning_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_nb_deletions_ticks'],
            'learning_proba_move': learning_params['evolutionary_settings']['sliding_puzzle_incremental']['sliding_puzzle_incremental_proba_move'], 
        })

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