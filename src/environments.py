import os

import random
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg') # Use TkAgg backend instead of QtAgg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
#from flags_distance_methods import convert_flag_to_image, get_images_distance_MSE, get_images_distance_SSIM, get_images_distance_CLIP

from agents import *


###########################################################################
# Global variables
###########################################################################

verbose_debug = False
verbose_str = ""

env = None # one single env
multiEnvs = None # list of envs (swarmGrid) of different sizes, used in sliding_puzzle_multiEnvs


###########################################################################
# Activation functions
###########################################################################

def init_swarmGrid_env(grid_nb_rows, grid_nb_cols, learning_modes, flags_distance_mode, learning_with_noise_std, flag_pattern, flag_target, init_cell_state_value, agent_type, nn_controller, nn_controller_stacking_mode, agent_controller_weights, nb_intrasteps, verbose_debug_bool, analysis_dir):
    
    global verbose_debug, env
    verbose_debug = verbose_debug_bool

    if env is None:
        env = swarmGrid(grid_nb_rows=grid_nb_rows,
                        grid_nb_cols=grid_nb_cols,
                        learning_modes=learning_modes,
                        flags_distance_mode=flags_distance_mode,
                        learning_with_noise_std=learning_with_noise_std,
                        flag_pattern=flag_pattern,
                        flag_target=flag_target,
                        init_cell_state_value=init_cell_state_value,
                        agent_type=agent_type,
                        nn_controller=nn_controller,
                        nn_controller_stacking_mode=nn_controller_stacking_mode,
                        nb_intrasteps=nb_intrasteps)
    
    # Some genomes could have 2 components, one dedicated to the controller NN and one for other purpose. Ex: in Devert 2011, 4 weights are required for the expression function
    agent_additional_weights = None
    nn_controller_weights_size = nn_controller[-1].weights_biases_size
    ind_size = len(agent_controller_weights)
    if (ind_size > nn_controller_weights_size):
        agent_controller_weights = agent_controller_weights[:nn_controller_weights_size]
        agent_additional_weights = agent_controller_weights[nn_controller_weights_size:]
    
    # grid and agent re-initialization are required for each new individual (new agent_controller_weights) before evaluation
    env.set_agent_controller_weights(agent_controller_weights=agent_controller_weights, agent_additional_weights=agent_additional_weights)
    env.init_agents_state() # to execute at last in this function
    return env


###########################################################################
# Learning evaluation function
###########################################################################

def sliding_puzzle(env_eval_function_params, analysis_dir, run, gen, nb_eval, nb_ind, best_fit, weights, sliding_puzzle_nb_deletions, sliding_puzzle_proba_move):
    time_steps = env_eval_function_params['time_steps']
    time_window_start = env_eval_function_params['time_window_start']
    time_window_end = env_eval_function_params['time_window_end']

    init_cell_state_value = env_eval_function_params['init_cell_state_value'] # init_cell_state_value is None or float depending on user settings in learning_params.json
    if "learning_random_init_states_bool" in env_eval_function_params['learning_modes']: # if this bool is True, init_cell_state_value is ignored
        init_cell_state_value = None # None = random state initialization

    random_async_update_bool = False
    if "learning_random_async_update_states_bool" in env_eval_function_params['learning_modes']:
        random_async_update_bool = True

    with_noise_bool = False
    noise_std = None
    if "learning_with_noise_bool" in env_eval_function_params['learning_modes']:
        with_noise_bool = True
        noise_std = env_eval_function_params['noise_std']

    env = init_swarmGrid_env(grid_nb_rows=env_eval_function_params['grid_nb_rows'],
                             grid_nb_cols=env_eval_function_params['grid_nb_cols'],
                             learning_modes=[],
                             flags_distance_mode=env_eval_function_params['flags_distance_mode'],
                             learning_with_noise_std=None,
                             flag_pattern=env_eval_function_params['flag_pattern'],
                             flag_target=env_eval_function_params['flag_target'],
                             init_cell_state_value=init_cell_state_value,
                             agent_type=env_eval_function_params['agent_type'],
                             nn_controller=env_eval_function_params['controller'],
                             nn_controller_stacking_mode=env_eval_function_params['nn_controller_stacking_mode'],
                             agent_controller_weights=weights,
                             nb_intrasteps=env_eval_function_params['nb_intrasteps'],
                             verbose_debug_bool=env_eval_function_params['verbose_debug'],
                             analysis_dir=env_eval_function_params['analysis_dir'])

    env.write_flag_target_data(analysis_dir=env_eval_function_params['analysis_dir'])

    in_t_window_zone_bools = []
    flags = []
    flags_signals = []

    flags_distance = 0.0
    sum_flags_distances = 0.0
    flags_distances = []

    agents_to_delete = []
    deleted_agents_per_step = []
    nb_moves_per_step = []

    agents = env.get_agents()
    agents_to_delete = random.sample(agents, sliding_puzzle_nb_deletions)
    deleted_map_pos_agent = env.delete_agent(agents_to_delete=agents_to_delete)

    for step in range(time_steps):
        in_t_window_zone_bool = False
        flag = env.get_flag_from_grid()
        flags.append(env.convert_flag_to_list(flag, env.size_phenotype))
        flags_distance = env.eval_flags_distance(flag)
        
        # To write and plot ANN learning mechanism
        flag_signals = env.get_flag_signals_from_grid()
        flags_signals.append(env.convert_flag_to_list(flag_signals, env.size_chemicals_to_spread))
        
        deleted_agents_per_step.append([a.pos for a in agents_to_delete])
        
        if step >= time_window_start and step <= time_window_end:
            in_t_window_zone_bool = True
            sum_flags_distances += flags_distance

        flags_distances.append(flags_distance)
        in_t_window_zone_bools.append(in_t_window_zone_bool)
        
        nb_moves = env.step_random_async_update_sliding_puzzle(agents_to_delete=agents_to_delete,
                                                                sliding_puzzle_proba_move=sliding_puzzle_proba_move,
                                                                with_noise_bool=with_noise_bool,
                                                                noise_std=noise_std)
        nb_moves_per_step.append(nb_moves)

    mean_tw_flags_distances = sum_flags_distances/(time_window_end - time_window_start)
    if mean_tw_flags_distances < best_fit:
        env.write_flag_data_learning(run=run, gen=gen, nb_eval=nb_eval, nb_ind=nb_ind, time_steps=time_steps, flags_distances=flags_distances, in_t_window_zone_bools=in_t_window_zone_bools, flags=flags, flags_signals=flags_signals, weights=weights, deleted_agents_per_step=deleted_agents_per_step, nb_moves_per_step=nb_moves_per_step, analysis_dir=analysis_dir)
        env.write_controller_data_for_pogobots(run=run, gen=gen, nb_eval=nb_eval, nb_ind=nb_ind, analysis_file=analysis_dir['data']+ f"/data_env_run_{run:03}_individual_controller_pogobots.txt") # overwrite previous saved files, to keep the best ind controller

    # Restore original grid
    env.restore_deleted_agents(deleted_map_pos_agent=deleted_map_pos_agent) # i have a new grid each time?

    # Verbose debug
    global verbose_str
    with open(analysis_dir['root']+"/verbose_debug.txt", 'a') as f:
        f.write(verbose_str)
        verbose_str = ""

    return (mean_tw_flags_distances,) # it is important to return a tuple (deap framework)

#---------------------------------------------------

def sliding_puzzle_multiEnvs(env_eval_function_params, analysis_dir, run, gen, nb_eval, nb_ind, best_fit, weights, sliding_puzzle_nb_deletions, sliding_puzzle_proba_move):
    time_steps = env_eval_function_params['time_steps']
    time_window_start = env_eval_function_params['time_window_start']
    time_window_end = env_eval_function_params['time_window_end']

    init_cell_state_value = env_eval_function_params['init_cell_state_value'] # init_cell_state_value is None or float depending on user settings in learning_params.json
    if "learning_random_init_states_bool" in env_eval_function_params['learning_modes']: # if this bool is True, init_cell_state_value is ignored
        init_cell_state_value = None # None = random state initialization

    random_async_update_bool = False
    if "learning_random_async_update_states_bool" in env_eval_function_params['learning_modes']:
        random_async_update_bool = True

    with_noise_bool = False
    noise_std = None
    if "learning_with_noise_bool" in env_eval_function_params['learning_modes']:
        with_noise_bool = True
        noise_std = env_eval_function_params['noise_std']

    env_dims_list = env_eval_function_params['env_dims_list']

    multiEnvs_mean_tw_flags_distances = []
    multiEnvs_flags_distances = []
    multiEnvs_flags = []
    multiEnvs_flags_signals = []
    multiEnvs_deleted_agents_per_step = []
    multiEnvs_nb_moves_per_step = []

    global env, multiEnvs
    if multiEnvs is None:
        multiEnvs = [None] * len(env_dims_list)

    for env_id, env_dims in enumerate(env_dims_list):

        env = multiEnvs[env_id] # this line triggers "init_swarmGrid_env" to create a new env, if env==None, i.e. if an env of this size has never been created before.
        multiEnvs[env_id] = init_swarmGrid_env(grid_nb_rows=env_dims[0],
                            grid_nb_cols=env_dims[1],
                            learning_modes=[],
                            flags_distance_mode=env_eval_function_params['flags_distance_mode'],
                            learning_with_noise_std=None,
                            flag_pattern=env_eval_function_params['flag_pattern'],
                            flag_target=env_eval_function_params['flag_target'],
                            init_cell_state_value=init_cell_state_value,
                            agent_type=env_eval_function_params['agent_type'],
                            nn_controller=env_eval_function_params['controller'],
                            nn_controller_stacking_mode=env_eval_function_params['nn_controller_stacking_mode'],
                            agent_controller_weights=weights,
                            nb_intrasteps=env_eval_function_params['nb_intrasteps'],
                            verbose_debug_bool=env_eval_function_params['verbose_debug'],
                            analysis_dir=env_eval_function_params['analysis_dir'])

        multiEnvs[env_id].write_flag_target_data(analysis_dir=env_eval_function_params['analysis_dir'], env_id=env_id)

        in_t_window_zone_bools = []
        flags = []
        flags_signals = []

        flags_distance = 0.0
        sum_flags_distances = 0.0
        flags_distances = []

        agents_to_delete = []
        deleted_agents_per_step = []
        nb_moves_per_step = []

        agents = multiEnvs[env_id].get_agents()
        agents_to_delete = random.sample(agents, sliding_puzzle_nb_deletions)
        deleted_map_pos_agent = multiEnvs[env_id].delete_agent(agents_to_delete=agents_to_delete)

        for step in range(time_steps):
            in_t_window_zone_bool = False
            flag = multiEnvs[env_id].get_flag_from_grid()
            flags.append(multiEnvs[env_id].convert_flag_to_list(flag, multiEnvs[env_id].size_phenotype))
            flags_distance = multiEnvs[env_id].eval_flags_distance(flag)
            
            # To write and plot ANN learning mechanism
            flag_signals = multiEnvs[env_id].get_flag_signals_from_grid()
            flags_signals.append(multiEnvs[env_id].convert_flag_to_list(flag_signals, multiEnvs[env_id].size_chemicals_to_spread))
            
            deleted_agents_per_step.append([a.pos for a in agents_to_delete])
            
            if step >= time_window_start and step <= time_window_end:
                in_t_window_zone_bool = True
                sum_flags_distances += flags_distance

            flags_distances.append(flags_distance)
            in_t_window_zone_bools.append(in_t_window_zone_bool)
            
            nb_moves = multiEnvs[env_id].step_random_async_update_sliding_puzzle(agents_to_delete=agents_to_delete,
                                                                    sliding_puzzle_proba_move=sliding_puzzle_proba_move,
                                                                    with_noise_bool=with_noise_bool,
                                                                    noise_std=noise_std)
            nb_moves_per_step.append(nb_moves)

        mean_tw_flags_distances = sum_flags_distances/(time_window_end - time_window_start)
        
        # We save one-single-env information for each env
        multiEnvs_mean_tw_flags_distances.append(mean_tw_flags_distances)
        multiEnvs_flags_distances.append(flags_distances)
        multiEnvs_flags.append(flags)
        multiEnvs_flags_signals.append(flags_signals)
        multiEnvs_deleted_agents_per_step.append(deleted_agents_per_step)
        multiEnvs_nb_moves_per_step.append(nb_moves_per_step)

        # Restore original grid
        multiEnvs[env_id].restore_deleted_agents(deleted_map_pos_agent=deleted_map_pos_agent) # problem?


    averaged_multiEnvs_mean_tw_flags_distances = sum(multiEnvs_mean_tw_flags_distances)/len(multiEnvs_mean_tw_flags_distances)
    if averaged_multiEnvs_mean_tw_flags_distances < best_fit:
        multiEnvs[env_id].write_multiEnvs_flag_data_learning(run=run, gen=gen, nb_eval=nb_eval, nb_ind=nb_ind, time_steps=time_steps, flags_distances=multiEnvs_flags_distances, in_t_window_zone_bools=in_t_window_zone_bools, flags=multiEnvs_flags, flags_signals=multiEnvs_flags_signals, weights=weights, deleted_agents_per_step=multiEnvs_deleted_agents_per_step, nb_moves_per_step=multiEnvs_nb_moves_per_step, analysis_dir=analysis_dir)
        multiEnvs[env_id].write_controller_data_for_pogobots(run=run, gen=gen, nb_eval=nb_eval, nb_ind=nb_ind, analysis_file=analysis_dir['data']+ f"/data_env_run_{run:03}_individual_controller_pogobots.txt") # overwrite previous saved files, to keep the best ind controller

    # Verbose debug
    global verbose_str
    with open(analysis_dir['root']+"/verbose_debug.txt", 'a') as f:
        f.write(verbose_str)
        verbose_str = ""

    return (averaged_multiEnvs_mean_tw_flags_distances,) # it is important to return a tuple (deap framework)

###########################################################################
# Environment swarmGrid
###########################################################################

class swarmGrid:
    def __init__(self, grid_nb_rows, grid_nb_cols, learning_modes, flags_distance_mode, learning_with_noise_std, flag_pattern, flag_target, init_cell_state_value, agent_type, nn_controller, nn_controller_stacking_mode, nb_intrasteps) -> None:
      
        self.grid_nb_rows = grid_nb_rows
        self.grid_nb_cols = grid_nb_cols
        self.grid_size = grid_nb_rows * grid_nb_cols

        self.agent_controller = nn_controller # list of one or more nn_controllers
        self.agent_controller_stacking_mode = nn_controller_stacking_mode
        self.agent_controller_weights = None
        self.agent_additional_weights = None
        self.agent_type = agent_type
        self.size_phenotype = None
        self.size_chemicals_to_spread = None
        self.nb_intrasteps = nb_intrasteps

        # parameters useful in the swarm application, to restore learning initialization parameters 
        self.learning_random_async_update_states_bool = True if "learning_random_async_update_states_bool" in learning_modes else False
        self.learning_with_noise_bool = True if "learning_with_noise_bool" in learning_modes else False
        self.learning_with_noise_std = learning_with_noise_std
        self.flags_distance_mode = flags_distance_mode
        
        self.default_missing_neighbor_state = 0.0
        self.grid_map_pos_agent = None
        self.init_grid(init_cell_state_value)
        
        if flag_target is not None:
            self.flag_target = flag_target
        else:
            self.flag_target = self.build_flag(flag_pattern) # flag_target is a dict pos:phenotype

    #---------------------------------------------------

    def set_agent_controller_weights(self, agent_controller_weights, agent_additional_weights=None):
        self.agent_controller_weights = agent_controller_weights
        self.agent_controller[-1].set_weights_biases_vectors_from_list(agent_controller_weights)

        if self.agent_type == agent3Outputs_Devert2011:
            self.agent_additional_weights = agent_additional_weights
            agents = self.get_agents()
            for agent in agents:
                agent.agent_additional_weights=agent_additional_weights # check si ça pose probleme en cas de deletion et reinsertion (agents sans additional_weights) ?

    #---------------------------------------------------

    def init_grid(self, init_cell_state_value):
    
        grid_map_pos_agent = {}
        for row in range(self.grid_nb_rows):
            for col in range(self.grid_nb_cols):
                agent = self.agent_type(pos=tuple((row, col)), init_cell_state_value=init_cell_state_value) # agent instance creation
                grid_map_pos_agent[tuple((row, col))] = agent
        
        self.grid_map_pos_agent = grid_map_pos_agent
        self.update_agent_neighbors()
        self.size_phenotype = agent.size_phenotype
        self.size_chemicals_to_spread = agent.size_chemicals_to_spread

    #---------------------------------------------------
    
    def get_agents(self):
        return [a for a in self.grid_map_pos_agent.values() if a != None]

    #---------------------------------------------------

    def update_agent_neighbors(self):
        agents = self.get_agents()
        for agent in agents:
            l_tmp = self.get_neighbors(agent=agent)
            agent.neighbors_NWES = l_tmp
    
    #---------------------------------------------------

    def init_agents_state(self, random_init_bool=False):
        agents = self.get_agents()
        for agent in agents:
            agent.init_state(random_init_bool)

    #---------------------------------------------------
            
    def build_flag(self, flag_pattern):
        flag_target = {}

        if flag_pattern == "two-bands":
            vertical_threshold = np.floor(self.grid_nb_cols*2/5)

            for cell in self.grid_map_pos_agent.keys():
                if cell[1] < vertical_threshold:
                    flag_target[cell] = 0.0 # black
                else:
                    flag_target[cell] = 1.0 # white


        elif flag_pattern == "three-bands":
            band_thickness = int(self.grid_nb_rows/3)
            horizontal_threshold_upper = band_thickness
            horizontal_threshold_lower = self.grid_nb_rows - 1 - band_thickness

            for cell in self.grid_map_pos_agent.keys():
                if cell[0] <= horizontal_threshold_upper:
                    flag_target[cell] = 0.0 # black in the upper region
                elif cell[0] >= horizontal_threshold_lower:
                    flag_target[cell] = 0.5 # grey in the lower region
                else:
                    flag_target[cell] = 1.0 # white in the middle region


        elif flag_pattern == "centered-disc" or flag_pattern == "not-centered-disc":
            center = (np.floor(self.grid_nb_rows/2), np.floor(self.grid_nb_cols/2))
            if flag_pattern == "not-centered-disc":
                center = (center[0]+1, center[1]+1)

            for cell in self.grid_map_pos_agent.keys():
                if self.grid_nb_rows%2 == 0 or self.grid_nb_cols%2 == 0:
                    radius = np.floor(min((self.grid_nb_rows/2)-1, (self.grid_nb_cols/2)-1))
                    if ((cell[0] - center[0])**2 + (cell[1] - center[1])**2 <= radius**2) \
                    or ((cell[0] - center[0]+1)**2 + (cell[1] - center[1])**2 <= radius**2) \
                    or ((cell[0] - center[0])**2 + (cell[1] - center[1]+1)**2 <= radius**2) \
                    or ((cell[0] - center[0]+1)**2 + (cell[1] - center[1]+1)**2 <= radius**2):
                        flag_target[cell] = 0.0 # black
                    else:
                        flag_target[cell] = 1.0 # white
                else:
                    radius = np.floor(min((self.grid_nb_rows/2), (self.grid_nb_cols/2)))
                    if ((cell[0] - center[0])**2 + (cell[1] - center[1])**2 <= radius**2):
                        flag_target[cell] = 0.0 # black
                    else:
                        flag_target[cell] = 1.0 # white


        elif flag_pattern == "half-discs":
            center = (np.floor(self.grid_nb_rows/2), np.floor(self.grid_nb_cols/2))
            radius = np.floor(min(self.grid_nb_rows/2, self.grid_nb_cols/2))

            for cell in self.grid_map_pos_agent.keys():
                if ((cell[0] - center[0])**2 + (cell[1] - center[1])**2 <= radius**2): # the cell is inside the disc area
                    if cell[1] <= center[1]:
                        flag_target[cell] = 0.0 # black
                    else:
                        flag_target[cell] = 1.0 # white
                else:  # the cell is outside the disc area
                    if cell[1] < center[1]:
                        flag_target[cell] = 1.0 # white
                    else:
                        flag_target[cell] = 0.0 # black


        elif flag_pattern == "centered-half-discs" or flag_pattern == "not-centered-half-discs":
            center = (np.floor(self.grid_nb_rows/2), np.floor(self.grid_nb_cols/2))
            if flag_pattern == "not-centered-half-discs":
                center = (center[0]+1, center[1]+1)

            for cell in self.grid_map_pos_agent.keys():
                if self.grid_nb_rows%2 == 0 or self.grid_nb_cols%2 == 0:
                    radius = np.floor(min((self.grid_nb_rows/2)-1, (self.grid_nb_cols/2)-1))
                    if ((cell[0] - center[0])**2 + (cell[1] - center[1])**2 <= radius**2) \
                    or ((cell[0] - center[0]+1)**2 + (cell[1] - center[1])**2 <= radius**2) \
                    or ((cell[0] - center[0])**2 + (cell[1] - center[1]+1)**2 <= radius**2) \
                    or ((cell[0] - center[0]+1)**2 + (cell[1] - center[1]+1)**2 <= radius**2):
                        if cell[1] < center[1]:
                            flag_target[cell] = 0.0 # black
                        else:
                            flag_target[cell] = 1.0 # white
                    else:  # the cell is outside the disc area
                        if cell[1] < center[1]+1:
                            flag_target[cell] = 1.0 # white
                        else:
                            flag_target[cell] = 0.0 # black
                else:
                    radius = np.floor(min((flag_pattern/2), (self.grid_nb_cols/2)))
                    if ((cell[0] - center[0])**2 + (cell[1] - center[1])**2 <= radius**2):
                        if cell[1] <= center[1]:
                            flag_target[cell] = 0.0 # black
                        else:
                            flag_target[cell] = 1.0 # white
                    else:  # the cell is outside the disc area
                        if cell[1] < center[1]:
                            flag_target[cell] = 1.0 # white
                        else:
                            flag_target[cell] = 0.0 # black


        elif flag_pattern == "coordinates":
            assert self.size_phenotype == 2
            for cell in self.grid_map_pos_agent.keys():
                flag_target[cell] = [round((1.0 / (self.grid_nb_cols-1)) * cell[1], 2), # linear_gradient_left_right = x component
                                    round((1.0 / (self.grid_nb_rows-1)) * cell[0], 2)] # linear_gradient_up_down = y component


        elif flag_pattern == "bn-SU":

            scales = {
                "bn-SU-4x4": 0.5,
                "bn-SU-8x8": 1,
                "bn-SU-16x16": 2,
                "bn-SU-24x24": 3,
                "bn-SU-32x32": 4
            }

            flag_pattern = flag_pattern + f"-{self.grid_nb_rows}x{self.grid_nb_cols}"
            assert flag_pattern in scales, f"Error in environments.py, build_flag - Pattern {flag_pattern} not supported."
            scale = scales[flag_pattern]

            pattern_black_cells_8x8_list = [(0,1), (0,2), (0,5), (0,7),
                                            (1,0), (1,3), (1,5), (1,7),
                                            (2,0), (2,5), (2,7),
                                            (3,1), (3,5), (3,7),
                                            (4,2), (4,5), (4,7),
                                            (5,3), (5,5), (5,7),
                                            (6,0), (6,3), (6,5), (6,7),
                                            (7,1), (7,2), (7,5), (7,6), (7,7)]

            black_cells = []
            for row, col in pattern_black_cells_8x8_list:
                for shift_row in range(scale):
                    for shift_col in range(scale):
                        black_cells.append((row * scale + shift_row, col * scale + shift_col))

            for cell in self.grid_map_pos_agent.keys():
                flag_target[cell] = 0.0 if cell in black_cells else 1.0


        elif flag_pattern == "bn-smile1":

            scales = {
                "bn-smile1-4x4": 0.5,
                "bn-smile1-8x8": 1,
                "bn-smile1-16x16": 2,
                "bn-smile1-24x24": 3,
                "bn-smile1-32x32": 4
            }

            flag_pattern = flag_pattern + f"-{self.grid_nb_rows}x{self.grid_nb_cols}"
            assert flag_pattern in scales, f"Error in environments.py, build_flag - Pattern {flag_pattern} not supported."
            scale = scales[flag_pattern]

            pattern_black_cells_8x8_list = [(1,2), (1,5),
                                            (2,2), (2,5),
                                            (4,0), (4,7),
                                            (5,1), (5,6),
                                            (6,2), (6,3), (6,4), (6,5)]

            black_cells = []
            for row, col in pattern_black_cells_8x8_list:
                for shift_row in range(scale):
                    for shift_col in range(scale):
                        black_cells.append((row * scale + shift_row, col * scale + shift_col))

            for cell in self.grid_map_pos_agent.keys():
                flag_target[cell] = 0.0 if cell in black_cells else 1.0


        elif flag_pattern == "bn-smile2":

            scales = {
                "bn-smile2-4x4": 0.5,
                "bn-smile2-8x8": 1,
                "bn-smile2-16x16": 2,
                "bn-smile2-24x24": 3,
                "bn-smile2-32x32": 4
            }

            flag_pattern = flag_pattern + f"-{self.grid_nb_rows}x{self.grid_nb_cols}"
            assert flag_pattern in scales, f"Error in environments.py, build_flag - Pattern {flag_pattern} not supported."
            scale = scales[flag_pattern]

            pattern_black_cells_8x8_list = [(1,1), (1,2), (1,5), (1,6),
                                            (2,1), (2,2), (2,5), (2,6),
                                            (4,0), (4,7),
                                            (5,0), (5,1), (5,6), (5,7),
                                            (6,1), (6,2), (6,3), (6,4), (6,5), (6,6),
                                            (7,2), (7,3), (7,4), (7,5)]

            black_cells = []
            for row, col in pattern_black_cells_8x8_list:
                for shift_row in range(scale):
                    for shift_col in range(scale):
                        black_cells.append((row * scale + shift_row, col * scale + shift_col))

            for cell in self.grid_map_pos_agent.keys():
                flag_target[cell] = 0.0 if cell in black_cells else 1.0


        elif flag_pattern == "rgb-italian-flag":
            assert self.size_phenotype == 3
            band_thickness = int(self.grid_nb_cols/3)
            vertical_threshold_left = band_thickness
            vertical_threshold_right = self.grid_nb_cols - 1 - band_thickness

            for cell in self.grid_map_pos_agent.keys():
                if cell[1] <= vertical_threshold_left:
                    flag_target[cell] = [0.0, 0.60, 0.20] # green in the region left
                elif cell[1] >= vertical_threshold_right:
                    flag_target[cell] = [1.0, 0.0, 0.0] # red in the region right
                else:
                    flag_target[cell] = [1.0, 1.0, 1.0] # white in the middle region


        elif flag_pattern == "rgb-french-cockade":
            assert self.size_phenotype == 3
            center = (np.floor(self.grid_nb_rows/2), np.floor(self.grid_nb_cols/2))
            band_thickness = int((min(self.grid_nb_rows/2, self.grid_nb_cols/2) -1)/3)

            for cell in self.grid_map_pos_agent.keys():
                if self.grid_nb_rows%2 == 0 or self.grid_nb_cols%2 == 0:
                    radius_outer = np.floor(min((self.grid_nb_rows/2)-1, (self.grid_nb_cols/2)-1))
                    radius_middle = radius_outer - band_thickness
                    radius_inner = radius_middle - band_thickness
                    if ((cell[0] - center[0])**2 + (cell[1] - center[1])**2 <= radius_inner**2) \
                    or ((cell[0] - center[0]+1)**2 + (cell[1] - center[1])**2 <= radius_inner**2) \
                    or ((cell[0] - center[0])**2 + (cell[1] - center[1]+1)**2 <= radius_inner**2) \
                    or ((cell[0] - center[0]+1)**2 + (cell[1] - center[1]+1)**2 <= radius_inner**2):
                        flag_target[cell] = [0.0, 0.0, 1.0] # blue in the inner region

                    elif ((cell[0] - center[0])**2 + (cell[1] - center[1])**2 <= radius_middle**2) \
                    or ((cell[0] - center[0]+1)**2 + (cell[1] - center[1])**2 <= radius_middle**2) \
                    or ((cell[0] - center[0])**2 + (cell[1] - center[1]+1)**2 <= radius_middle**2) \
                    or ((cell[0] - center[0]+1)**2 + (cell[1] - center[1]+1)**2 <= radius_middle**2):
                        flag_target[cell] = [1.0, 1.0, 1.0] # white in the middle region

                    elif ((cell[0] - center[0])**2 + (cell[1] - center[1])**2 <= radius_outer**2) \
                    or ((cell[0] - center[0]+1)**2 + (cell[1] - center[1])**2 <= radius_outer**2) \
                    or ((cell[0] - center[0])**2 + (cell[1] - center[1]+1)**2 <= radius_outer**2) \
                    or ((cell[0] - center[0]+1)**2 + (cell[1] - center[1]+1)**2 <= radius_outer**2):
                        flag_target[cell] = [1.0, 0.0, 0.0] # red in the outer region
                    else:
                        flag_target[cell] = [1.0, 1.0, 1.0] # grey

                else:
                    radius_outer = np.floor(min((self.grid_nb_rows/2), (self.grid_nb_cols/2)))
                    radius_middle = radius_outer - band_thickness
                    radius_inner = radius_middle - band_thickness
                    if ((cell[0] - center[0])**2 + (cell[1] - center[1])**2 <= radius_inner**2):
                        flag_target[cell] = [0.0, 0.0, 1.0] # blue in the inner region
                    elif ((cell[0] - center[0])**2 + (cell[1] - center[1])**2 <= radius_middle**2):
                        flag_target[cell] = [1.0, 1.0, 1.0] # white in the middle region
                    elif ((cell[0] - center[0])**2 + (cell[1] - center[1])**2 <= radius_outer**2):
                        flag_target[cell] = [1.0, 0.0, 0.0] # red in the outer region
                    else:
                        flag_target[cell] = [1.0, 1.0, 1.0] # white in the middle region
            

        elif flag_pattern == "rgb-rainbow-full":
            assert self.size_phenotype == 3
            
            colors_list = [
                [1.0, 0.0, 0.0],     # red
                [1.0, 0.6, 0.0],     # orange
                [1.0, 1.0, 0.0],     # yellow
                [0.0, 0.5, 0.0],     # green
                [0.0, 1.0, 1.0],     # cyan
                [0.0, 0.0, 1.0],     # blue
                [0.5, 0.0, 0.5]]     # purple
            
            arrow_size = max(self.grid_nb_rows, self.grid_nb_cols)
            band_thickness = max(1, min(self.grid_nb_rows, self.grid_nb_cols) // 8) # scaling pattern from grid 8x8

            # Initialize grid to avoid not colored regions
            for cell in self.grid_map_pos_agent.keys():
                flag_target[cell] = [1.0, 1.0, 1.0] # white

            for row_index in range(self.grid_nb_rows):
                col_index = self.grid_nb_cols - 1 - row_index # x in {N-1, N-2, ..., 0}. (row_index,col_index) identifies the diagonal of the grid: {(0, N-1), (1, N-2), ..., (N-1, 0)}
                if not (0 <= col_index < self.grid_nb_cols):
                    continue
                
                color_index = (row_index // band_thickness) % len(colors_list) # loop color choice during 'band_thickness' cases
                flag_target[(row_index, col_index)] = colors_list[color_index]

                # Color all pixels in the same row and col as the diagonal cell
                for offset in range(1, arrow_size):
                    off = col_index - offset # horizontal shift
                    if 0 <= off < self.grid_nb_cols: # if this position is valid
                        flag_target[(row_index, off)] = colors_list[color_index]

                    off = row_index + offset # vertical shift
                    if 0 <= off < self.grid_nb_rows: # if this position is valid
                        flag_target[(off, col_index)] = colors_list[color_index]


        elif flag_pattern == "rgb-rainbow-arrow": # same code as "rgb-rainbow-full", just different "arrow_size"
            assert self.size_phenotype == 3
            
            colors_list = [
                [1.0, 0.0, 0.0],     # red
                [1.0, 0.6, 0.0],     # orange
                [1.0, 1.0, 0.0],     # yellow
                [0.0, 0.5, 0.0],     # green
                [0.0, 1.0, 1.0],     # cyan
                [0.0, 0.0, 1.0],     # blue
                [0.5, 0.0, 0.5]]     # purple
            
            arrow_size = max(1, min(self.grid_nb_rows, self.grid_nb_cols) // 4)
            band_thickness = max(1, min(self.grid_nb_rows, self.grid_nb_cols) // 8) # scaling pattern from grid 8x8

            # Initialize grid to avoid not colored regions
            for cell in self.grid_map_pos_agent.keys():
                flag_target[cell] = [1.0, 1.0, 1.0] # white

            for row_index in range(self.grid_nb_rows):
                col_index = self.grid_nb_cols - 1 - row_index # x in {N-1, N-2, ..., 0}. (row_index,col_index) identifies the diagonal of the grid: {(0, N-1), (1, N-2), ..., (N-1, 0)}
                if not (0 <= col_index < self.grid_nb_cols):
                    continue
                
                color_index = (row_index // band_thickness) % len(colors_list) # loop color choice during 'band_thickness' cases
                flag_target[(row_index, col_index)] = colors_list[color_index]

                # Color all pixels in the same row and col as the diagonal cell
                for offset in range(1, arrow_size):
                    off = col_index - offset # horizontal shift
                    if 0 <= off < self.grid_nb_cols: # if this position is valid
                        flag_target[(row_index, off)] = colors_list[color_index]

                    off = row_index + offset # vertical shift
                    if 0 <= off < self.grid_nb_rows: # if this position is valid
                        flag_target[(off, col_index)] = colors_list[color_index]


        # Verbose debug
        if verbose_debug:
            global verbose_str
            verbose_str += f"\n<build_flag> - Flag pattern just built: {flag_pattern}.\n{self.convert_flag_to_list(flag_target, self.size_phenotype)}"

        return self.convert_flag_to_list(flag_target, self.size_phenotype)

    #---------------------------------------------------

    @staticmethod
    def is_pos_valid(grid_nb_rows, grid_nb_cols, pos):
        return pos[0]>=0 and pos[1]>=0 and pos[0]<grid_nb_rows and pos[1]<grid_nb_cols

    #---------------------------------------------------

    def get_neighbors(self, agent):
        l_tmp = []
        row, col = agent.pos
        for pos in [(row-1, col), (row, col-1), (row, col+1), (row+1, col)]:
            if swarmGrid.is_pos_valid(self.grid_nb_rows, self.grid_nb_cols, pos) and self.grid_map_pos_agent[tuple(pos)] != None:
                l_tmp.append(self.grid_map_pos_agent[tuple(pos)])
            else:
                l_tmp.append(None)
        
        return l_tmp

    #---------------------------------------------------

    def step(self, random_async_update_bool=False, with_noise_bool=False, noise_std=None):
        global verbose_str
        agents = self.get_agents() # ordered update

        if random_async_update_bool: # async_update
            np.random.shuffle(agents) # random order

            for agent in agents:
                state = self.compute_agent_state(agent=agent)
                agent.set_state(state, with_noise_bool, noise_std)

        else: # sync_update
            grid_map_agent_state_tmp = {}
            for agent in agents:
                state = self.compute_agent_state(agent=agent)
                grid_map_agent_state_tmp[agent] = state

            if verbose_debug:
                verbose_str += f"\n<step> - Synchronous update of the states. Grid states at t-1: {self.get_flag_from_grid()}"
                verbose_str += f"\n<step> - Synchronous update of the states. Temporary grid states at t: {grid_map_agent_state_tmp.values()}"

            for agent in agents:
                agent.set_state(grid_map_agent_state_tmp[agent], with_noise_bool, noise_std)

            if verbose_debug:
                verbose_str += f"\n<step> - Synchronous update of the states. Grid states at t+1: {self.get_flag_from_grid()}"

    #---------------------------------------------------

    def step_random_async_update_sliding_puzzle(self, agents_to_delete, sliding_puzzle_proba_move, with_noise_bool=False, noise_std=None):
        global verbose_str
        nb_moves_per_step = 0
        agents = self.get_agents()
        np.random.shuffle(agents) # random update order (async update)

        for agent in agents:

            # Check if there are empty cells in the neighbouring cells around this agent 
            row, col = agent.pos
            neighbouring_cells = [pos for pos in [(row-1, col), (row, col-1), (row, col+1), (row+1, col)] if swarmGrid.is_pos_valid(self.grid_nb_rows, self.grid_nb_cols, pos)]
            empty_neighbouring_cells_list = [pos for pos in neighbouring_cells if self.grid_map_pos_agent[pos] == None]
            np.random.shuffle(empty_neighbouring_cells_list) # random order

            # For each empty cell in the agent neighborhood, this agent tries to occupe its position
            # If this agent moves in an empty cell, than the grid is updated and the other empty cells to try are ignored
            for empty_pos in empty_neighbouring_cells_list:
                if random.random() < sliding_puzzle_proba_move:
                    nb_moves_per_step += 1

                    # Movement of the deleted/hidden agent, from the empty cell to the this-agent position
                    agent_to_delete = [a for a in agents_to_delete if a.pos == empty_pos][0] 
                    agent_to_delete.pos = agent.pos

                    # Movement of this agent, from its position to the empty cell
                    self.grid_map_pos_agent[agent.pos] = None
                    self.grid_map_pos_agent[empty_pos] = agent
                    agent.pos = empty_pos

                    # Update the neighbors list of each agent
                    self.update_agent_neighbors()
                    break # the agent has already moved once, other empty cells to try are ignored
            
            # Random async update of this agent state
            state = self.compute_agent_state(agent=agent)
            agent.set_state(state, with_noise_bool, noise_std)
        
        if verbose_debug:
            verbose_str += f"\n<step_random_async_update_sliding_puzzle> - Asynchronous update of the states done"


        # Computing the remaining intrasteps
        if self.nb_intrasteps is not None:
            agents = self.get_agents()

            if verbose_debug:
                verbose_str += f"\n<step_random_async_update_sliding_puzzle> - Intrastep 1/{self.nb_intrasteps} already computed"

            for intrastep in range(2, self.nb_intrasteps+1): # one state computation per agent has already been done
                # Random asynchronous update of the agent states (random update order)
                np.random.shuffle(agents)
                for agent in agents:
                    state = self.compute_agent_state(agent=agent)
                    agent.set_state(state, with_noise_bool, noise_std)

                if verbose_debug:
                    verbose_str += f"\n<step_random_async_update_sliding_puzzle> - Intrastep {intrastep}/{self.nb_intrasteps} done"


        return nb_moves_per_step

    #---------------------------------------------------

    def get_swarmGrid_energy(self):
        if self.agent_type == agent3Outputs_Devert2011:
            organism_energy = 0
            agents = self.get_agents()
            for agent in agents:
                organism_energy += agent.get_agent_energy()
            return organism_energy
        raise ValueError("Error: <get_swarmGrid_energy>: energy not available for this agent. Set a agent3Outputs_Devert2011 agent.")

    #---------------------------------------------------

    def compute_agent_state(self, agent):
        global verbose_str

        if agent is None: # check
            print("ça arrive?")
            return

        # Get controller inputs (chemicals_to_spread)
        neighbors_states = []
        for neighbor in agent.neighbors_NWES: # est il à jour? calcolarlo ora?
            if neighbor is not None: # si il a un id, il y a l'agent? tjr?
                neighbors_states += neighbor.get_external_chemicals_to_spread()
            else:
                neighbors_states += [self.default_missing_neighbor_state] * agent.size_chemicals_to_spread

        if (self.agent_type == agent3Outputs_Devert2011):
            neighbors_states += agent.get_internal_chemicals()
        

        # Compute controller outputs (state)
        if len(self.agent_controller) == 1: # one only ANN is used
            state = self.agent_controller[0].predict(neighbors_states) # forwardPropagation, stableSigmoid on the last layer       
            
            if verbose_debug:
                verbose_str += f"\n<compute_agent_state> - Agent at pos {agent.pos}, neighbors_states = {neighbors_states}, final state = {state}"
        

        # In the following options, we combine more than one ANN
        elif (self.agent_type == agent2Outputs or self.agent_type == agent2Outputs_RGB) and self.agent_controller_stacking_mode == "ann1_ann2_modelA":

            # Model A: 4-x-3_2-y-1
            # The 1st ANN (4-x-3), used for the learning phase (coordinates system, flag 2D), has:
            #   - inputs: a signal (chemicals_to_spread) from each neighbor. ann1 input = neighbors_states = [ signal_xy_N, signal_xy_W, signal_xy_E, signal_xy_S ]
            #   - output: a signal to spread to neighbors, and two phenotype values x and y. ann1 output = [ signal_xy, x, y ]
            # The 2nd ANN (2-y-1), used to learn the target flag (two-bands, centered-half-discs, ...), has:
            #   - inputs: x and y from the coordinate system. ann2 input = [ x, y ]
            #   - output: one phenotype. ann2 output = [ p ]
            # The final state for an agent, is [signal_xy from ann1, p from ann2]
            # NB: all phenotypes (x, y, p) are rescaled from (-1,1) to (0,1); ann2 phenotypes (p) will be rescaled in agent2Outputs

            # print("model A: ann1 input =", neighbors_states)
            ann1_output = list(self.agent_controller[0].predict(neighbors_states)) # ann1 input = [ signal_xy_N, signal_xy_W, signal_xy_E, signal_xy_S ], ann1 output = [ signal_xy, x, y ]
            # print("model A: ann1 output =", ann1_output)
            ann1_output[1] = (ann1_output[1] + 1) / 2 # rescale phenotype x in (-1,1) to (0,1)
            ann1_output[2] = (ann1_output[2] + 1) / 2 # rescale phenotype y in (-1,1) to (0,1)
            # print("model A: ann1 input after rescale =", ann1_output)
            # print("model A: ann2 input =", ann1_output[-2:])
            ann2_output = list(self.agent_controller[-1].predict(ann1_output[-2:])) # ann2 input = [ x, y ], ann2 output = [ p ]
            # print("model A: ann2 output =", ann2_output)
            state = [ann1_output[0]] + ann2_output # state = [signal_xy from ann1, p from ann2]. ann2 phenotypes (p) will be rescaled in agent2Outputs
            # print("model A: state = [signal_xy from ann1, p from ann2] =", state, "\n-----")


        elif (self.agent_type == agent3Outputs or self.agent_type == agent3Outputs_RGB) and self.agent_controller_stacking_mode == "ann1_ann2_modelB":
            
            # Model B: 4-x-3_6-y-2
            # The 1st ANN (4-x-3), used for the learning phase (coordinates system, flag 2D), has:
            #   - inputs: a signal (chemicals_to_spread) from each neighbor. ann1 input = 1st quartet of neighbors_states = [ signal_xy_N, signal_xy_W, signal_xy_E, signal_xy_S ]
            #   - output: a signal to spread to neighbors, and two phenotype values x and y. ann1 output = [ signal_xy, x, y ]
            # The 2nd ANN (6-y-2), used to learn the target flag (two-bands, centered-half-discs, ...), has:
            #   - inputs: x and y from the coordinate system, plus the 2nd quartet of neighbors_states. ann2 input = [ x, y, signal_p_N, signal_p_W, signal_p_E, signal_p_S ]
            #   - output: one signal to spread to neighbors and one phenotype. ann2 output = [ signal_p, p ]
            # The final state for an agent, is [signal_xy from ann1, signal_p from ann2, p from ann2]
            # NB: all phenotypes (x, y, p) are rescaled from (-1,1) to (0,1); ann2 phenotypes (p) will be rescaled in agent3Outputs
            
            signals_xy = neighbors_states[::2] # even indexes = signal_xy
            signals_p = neighbors_states[1::2] # odd indexes = signal_p
            # print("model B: ann1 input =", neighbors_states)
            # print("model B: signals_xy =", signals_xy)
            # print("model B: signals_p =", signals_p)
            ann1_output = list(self.agent_controller[0].predict(signals_xy)) # ann1 input = [ signal_xy_N, signal_xy_W, signal_xy_E, signal_xy_S ], ann1 output = [ signal_xy, x, y ]
            # print("model B: ann1 output =", ann1_output)
            ann1_output[1] = (ann1_output[1] + 1) / 2 # rescale phenotype x in (-1,1) to (0,1)
            ann1_output[2] = (ann1_output[2] + 1) / 2 # rescale phenotype y in (-1,1) to (0,1)
            # print("model B: ann1 input after rescale =", ann1_output)
            ann2_input = list(ann1_output[-2:]) + signals_p # ann2 input = [ x, y, signal_p_N, signal_p_W, signal_p_E, signal_p_S ]
            # print("model B: ann2 input =", ann2_input)
            ann2_output = list(self.agent_controller[-1].predict(ann2_input)) # ann2 output = [ signal_p, p ]
            # print("model B: ann2 output =", ann2_output)
            state = [ann1_output[0]] + ann2_output # state = [signal_xy from ann1, signal_p from ann2, p from ann2]. ann2 phenotypes (p) will be rescaled in agent3Outputs
            # print("model B: state = [signal_xy from ann1, signal_p from ann2, p from ann2] =", state, "\n-----")
            

        elif (self.agent_type == agent2Outputs or self.agent_type == agent2Outputs_RGB) and self.agent_controller_stacking_mode == "ann1_ann2_modelC":

            # Model C: 4-x-3_6-y-1
            # The 1st ANN (4-x-3), used for the learning phase (coordinates system, flag 2D), has:
            #   - inputs: a signal (chemicals_to_spread) from each neighbor. ann1 input = neighbors_states = [ signal_xy_N, signal_xy_W, signal_xy_E, signal_xy_S ]
            #   - output: a signal to spread to neighbors, and two phenotype values x and y. ann1 output = [ signal_xy, x, y ]
            # The 2nd ANN (6-y-1), used to learn the target flag (two-bands, centered-half-discs, ...), has:
            #   - inputs: x and y from the coordinate system, plus the ann1 neighbors_states. ann2 input = [ x, y, signal_xy_N, signal_xy_W, signal_xy_E, signal_xy_S ]
            #   - output: one phenotype. ann2 output = [ p ]
            # The final state for an agent, is [signal_xy from ann1, p from ann2]
            # NB: ann1 phenotypes (x, y) are rescaled from (-1,1) to (0,1); ann2 phenotypes (p) will be rescaled in agent2Outputs
            
            # print("model C: ann1 input =", neighbors_states)
            ann1_output = list(self.agent_controller[0].predict(neighbors_states)) # ann1 input = [ signal_xy_N, signal_xy_W, signal_xy_E, signal_xy_S ], ann1 output = [ signal_xy, x, y ]
            # print("model C: ann1 output =", ann1_output)
            ann1_output[1] = (ann1_output[1] + 1) / 2 # rescale phenotype x in (-1,1) to (0,1)
            ann1_output[2] = (ann1_output[2] + 1) / 2 # rescale phenotype y in (-1,1) to (0,1)
            # print("model C: ann1 input after rescale =", ann1_output)
            ann2_input = list(ann1_output[-2:]) + neighbors_states # ann2 input = [ x, y, signal_xy_N, signal_xy_W, signal_xy_E, signal_xy_S ]
            # print("model C: ann2 input =", ann2_input)
            ann2_output = list(self.agent_controller[-1].predict(ann2_input)) # ann2 output = [ p ]
            # print("model C: ann2 output =", ann2_output)
            state = [ann1_output[0]] + ann2_output # state = [signal_xy from ann1, p from ann2]. ann2 phenotypes (p) will be rescaled in agent2Outputs
            # print("model C: state = [signal_xy from ann1, p from ann2] =", state, "\n-----")


        elif (self.agent_type == agent3Outputs or self.agent_type == agent3Outputs_RGB) and self.agent_controller_stacking_mode == "ann1_ann2_modelE":
            
            # Model E: 4-x-3_10-y-2
            # The 1st ANN (4-x-3), used for the learning phase (coordinates system, flag 2D), has:
            #   - inputs: a signal (chemicals_to_spread) from each neighbor. ann1 input = 1st quartet of neighbors_states = [ signal_xy_N, signal_xy_W, signal_xy_E, signal_xy_S ]
            #   - output: a signal to spread to neighbors, and two phenotype values x and y. ann1 output = [ signal_xy, x, y ]
            # The 2nd ANN (10-y-2), used to learn the target flag (two-bands, centered-half-discs, ...), has:
            #   - inputs: x and y from the coordinate system, plus the 2nd quartet of neighbors_states, plus the 2nd quartet of neighbors_states. ann2 input = [ x, y, signal_xy_N, signal_xy_W, signal_xy_E, signal_xy_S, signal_p_N, signal_p_W, signal_p_E, signal_p_S ]
            #   - output: one signal to spread to neighbors and one phenotype. ann2 output = [ signal_p, p ]
            # The final state for an agent, is [signal_xy from ann1, signal_p from ann2, p from ann2]
            # NB: all phenotypes (x, y, p) are rescaled from (-1,1) to (0,1); ann2 phenotypes (p) will be rescaled in agent3Outputs
            
            signals_xy = neighbors_states[::2] # even indexes = signal_xy
            signals_p = neighbors_states[1::2] # odd indexes = signal_p
            # print("model E: ann1 input =", neighbors_states)
            # print("model E: signals_xy =", signals_xy)
            # print("model E: signals_p =", signals_p)
            ann1_output = list(self.agent_controller[0].predict(signals_xy)) # ann1 input = [ signal_xy_N, signal_xy_W, signal_xy_E, signal_xy_S ], ann1 output = [ signal_xy, x, y ]
            # print("model E: ann1 output =", ann1_output)
            ann1_output[1] = (ann1_output[1] + 1) / 2 # rescale phenotype x in (-1,1) to (0,1)
            ann1_output[2] = (ann1_output[2] + 1) / 2 # rescale phenotype y in (-1,1) to (0,1)
            # print("model E: ann1 input after rescale =", ann1_output)
            ann2_input = list(ann1_output[-2:]) + signals_xy + signals_p # ann2 input = [ x, y, signal_xy_N, signal_xy_W, signal_xy_E, signal_xy_S, signal_p_N, signal_p_W, signal_p_E, signal_p_S ]
            # print("model E: ann2 input =", ann2_input)
            ann2_output = list(self.agent_controller[-1].predict(ann2_input)) # ann2 output = [ signal_p, p ]
            # print("model E: ann2 output =", ann2_output)
            state = [ann1_output[0]] + ann2_output # state = [signal_xy from ann1, signal_p from ann2, p from ann2]. ann2 phenotypes (p) will be rescaled in agent3Outputs
            # print("model E: state = [signal_xy from ann1, signal_p from ann2, p from ann2] =", state, "\n-----")

        return state

    #---------------------------------------------------

    def setup_ind_consistency(self, run, setup_name, nb_repetitions, time_steps, switch_step, switch_step_with_reset_env_bool, analysis_dir):
        global verbose_str
        for n in range(nb_repetitions):
            switch_step_with_random_async_update_bool = self.learning_random_async_update_states_bool
            with_noise_bool = self.learning_with_noise_bool
            noise_std = self.learning_with_noise_std
            self.init_agents_state()
            flags = []

            for step in range(time_steps):

                if setup_name != "setup_ind_consistency_learning_conditions" and step == switch_step-1:
                
                    if setup_name == "setup_ind_consistency_random_init_states": # this setup represents noisy perturbation on initialization
                        self.init_agents_state(random_init_bool=True)
                        flag = self.get_flag_from_grid()
                        flags.append(flag)
                        continue # don't execute the following self.step update line, as we just updated the grid in this condition

                    if setup_name == "setup_ind_consistency_random_async_update_states": # this setup represents perturbation on the update order
                        switch_step_with_random_async_update_bool = True
                        if switch_step_with_reset_env_bool:
                            self.init_agents_state()
                            flag = self.get_flag_from_grid()
                            flags.append(flag)
                            continue # don't execute the following self.step update line, as we just updated the grid in this condition

                self.step(random_async_update_bool=switch_step_with_random_async_update_bool,
                          with_noise_bool=with_noise_bool,
                          noise_std=noise_std)
                
                flag = self.get_flag_from_grid()
                flags.append(flag)

            
            # Save flags for this run
            self.write_flag_data_swarm(setup_name=setup_name, run=run, n=n, time_steps=time_steps, flags=flags, permutated_agents_per_step=[], deleted_agents_per_step=[], nb_moves_per_step=[], analysis_dir=analysis_dir)

        with open(analysis_dir['root']+"/verbose_debug.txt", 'a') as f:
            f.write(verbose_str)
            verbose_str = ""

    #---------------------------------------------------

    def setup_noise(self, run, setup_name, nb_repetitions, setup_noise_std_ticks, time_steps, switch_step, switch_step_with_reset_env_bool, switch_step_with_random_async_update_bool, analysis_dir):
        global verbose_str
        for n in range(nb_repetitions):
            for tick in setup_noise_std_ticks:
                setup_name_tick = setup_name+"_"+str(tick)
                switch_step_with_random_async_update_bool = self.learning_random_async_update_states_bool
                with_noise_bool = self.learning_with_noise_bool
                noise_std = self.learning_with_noise_std
                self.init_agents_state()
                flags = []

                for step in range(time_steps):

                    if step == switch_step-1:
                        with_noise_bool = True
                        noise_std = tick

                        if switch_step_with_reset_env_bool:
                            self.init_agents_state()
                            flag = self.get_flag_from_grid()
                            flags.append(flag)
                            continue # don't execute the following self.step update line, as we just updated the grid in this condition

                    self.step(random_async_update_bool=switch_step_with_random_async_update_bool,
                              with_noise_bool=with_noise_bool,
                              noise_std=noise_std)
                    
                    flag = self.get_flag_from_grid()
                    flags.append(flag)

                
                # Save flags for this run
                self.write_flag_data_swarm(setup_name=setup_name_tick, run=run, n=n, time_steps=time_steps, flags=flags, permutated_agents_per_step=[], deleted_agents_per_step=[], nb_moves_per_step=[], analysis_dir=analysis_dir)
        
        with open(analysis_dir['root']+"/verbose_debug.txt", 'a') as f:
            f.write(verbose_str)
            verbose_str = ""

    #---------------------------------------------------

    def setup_permutation(self, run, setup_name, nb_repetitions, setup_permutation_ticks, time_steps, switch_step, switch_step_with_reset_env_bool, switch_step_with_random_async_update_bool, analysis_dir):
        global verbose_str
        for n in range(nb_repetitions):
            for tick in setup_permutation_ticks:
                setup_name_tick = setup_name+"_"+str(tick)
                switch_step_with_random_async_update_bool = self.learning_random_async_update_states_bool
                with_noise_bool = self.learning_with_noise_bool
                noise_std = self.learning_with_noise_std
                self.init_agents_state()
                flags = []
                agents_to_permutate = []
                permutated_agents_per_step = []

                if verbose_debug:
                    verbose_str += f"\nStarting setup_permutation n={n} tick={tick}"

                for step in range(time_steps):

                    if step == switch_step-1:
                        agents = self.get_agents()
                        agents_to_permutate = random.sample(agents, tick) # random.sample returns a list in selection order
                        self.permutate_agents(agents_to_permutate=agents_to_permutate)

                        if verbose_debug:
                            verbose_str += f"\nStep {step}. Check the content of the permutated agents list in the last column of the data_setup_permutation file. Ex path: swarm/run_000/best_ind_000/data/setup_permutation_6/data_setup_permutation_6_flag_run_000_n_000.csv"
                        
                        if switch_step_with_reset_env_bool:
                            self.init_agents_state()
                            permutated_agents_per_step.append([a.pos for a in agents_to_permutate])                    
                            flag = self.get_flag_from_grid()
                            flags.append(flag)
                            continue # don't execute the following self.step update line, as we just updated the grid with the permutation and eventually the reset_env

                    self.step(random_async_update_bool=switch_step_with_random_async_update_bool,
                              with_noise_bool=with_noise_bool,
                              noise_std=noise_std)
                    
                    permutated_agents_per_step.append([a.pos for a in agents_to_permutate])                    
                    flag = self.get_flag_from_grid()
                    flags.append(flag)


                # Save flags for this run
                self.write_flag_data_swarm(setup_name=setup_name_tick, run=run, n=n, time_steps=time_steps, flags=flags, permutated_agents_per_step=permutated_agents_per_step, deleted_agents_per_step=[], nb_moves_per_step=[], analysis_dir=analysis_dir)

        with open(analysis_dir['root']+"/verbose_debug.txt", 'a') as f:
            f.write(verbose_str)
            verbose_str = ""

    #---------------------------------------------------

    def setup_deletion(self, run, setup_name, nb_repetitions, deletion_ticks, time_steps, switch_step, switch_step_with_reset_env_bool, switch_step_with_random_async_update_bool, analysis_dir):
        global verbose_str
        for n in range(nb_repetitions):
            deleted_map_pos_agent = {}

            for tick in deletion_ticks:
                setup_name_tick = setup_name+"_"+str(tick)
                switch_step_with_random_async_update_bool = self.learning_random_async_update_states_bool
                with_noise_bool = self.learning_with_noise_bool
                noise_std = self.learning_with_noise_std
                self.init_agents_state()
                flags = []
                agents_to_delete = []
                deleted_agents_per_step = []

                for step in range(time_steps):

                    if step == switch_step-1:
                        agents = self.get_agents()
                        nb_deletions = tick - len(deleted_map_pos_agent)
                        agents_to_delete = list(deleted_map_pos_agent.values()) + random.sample(agents, nb_deletions) # old + new chosen agents
                        deleted_map_pos_agent = self.delete_agent(agents_to_delete=agents_to_delete)

                        if switch_step_with_reset_env_bool:
                            self.init_agents_state()
                            deleted_agents_per_step.append(agents_to_delete)
                            flag = self.get_flag_from_grid()
                            flags.append(flag)
                            continue # don't execute the following self.step update line, as we just updated the grid in this condition

                    self.step(random_async_update_bool=switch_step_with_random_async_update_bool,
                              with_noise_bool=with_noise_bool,
                              noise_std=noise_std)
                    
                    deleted_agents_per_step.append([a.pos for a in agents_to_delete])
                    flag = self.get_flag_from_grid()
                    flags.append(flag)
                
                
                # Save flags for this run
                self.write_flag_data_swarm(setup_name=setup_name_tick, run=run, n=n, time_steps=time_steps, flags=flags, permutated_agents_per_step=[], deleted_agents_per_step=deleted_agents_per_step, nb_moves_per_step=[], analysis_dir=analysis_dir)
            
                # Restore original grid
                self.restore_deleted_agents(deleted_map_pos_agent=deleted_map_pos_agent)

        with open(analysis_dir['root']+"/verbose_debug.txt", 'a') as f:
            f.write(verbose_str)
            verbose_str = ""

    #---------------------------------------------------

    def setup_sliding_puzzle(self, run, setup_name, nb_repetitions, sliding_puzzle_ticks, sliding_puzzle_probas_move, time_steps, time_window_start, time_window_end, analysis_dir):
        global verbose_str
        for proba_move in sliding_puzzle_probas_move:
            for n in range(nb_repetitions):
                deleted_map_pos_agent = {}

                # print("controller:", self.agent_controller_weights)

                
                # This code is similar to setup_deletion as it manages the deletion of tiles:
                # Their position will be modified in step_random_async_update_sliding_puzzle.
                # In this setup, the automata update is always asynchrone (random_async_update)
                for tick in sliding_puzzle_ticks:
                    
                    setup_name_tick = f"{setup_name}_deletions{tick}_fluidity{proba_move}"
                    with_noise_bool = self.learning_with_noise_bool
                    noise_std = self.learning_with_noise_std
                    self.init_agents_state()
                    flags = []
                    flags_distance = 0.0
                    sum_flags_distances = 0.0
                    agents_to_delete = []
                    deleted_agents_per_step = []
                    nb_moves_per_step = []

                    agents = self.get_agents()
                    agents_to_delete = random.sample(agents, tick)
                    deleted_map_pos_agent = self.delete_agent(agents_to_delete=agents_to_delete)

                    for step in range(time_steps):
                        deleted_agents_per_step.append([a.pos for a in agents_to_delete])
                        flag = self.get_flag_from_grid()
                        flags.append(flag)
                        # print("env.eval_flags_distance(flag)", env.eval_flags_distance(flag))
                        # print("\t", flag)
                        # if step == 3:
                        #     break

                        if step >= time_window_start and step <= time_window_end:
                            sum_flags_distances += env.eval_flags_distance(flag)

                        nb_moves = self.step_random_async_update_sliding_puzzle(agents_to_delete=agents_to_delete,
                                                                                sliding_puzzle_proba_move=proba_move,
                                                                                with_noise_bool=with_noise_bool,
                                                                                noise_std=noise_std)
                        nb_moves_per_step.append(nb_moves)


                    mean_tw_flags_distances = sum_flags_distances/(time_window_end - time_window_start)
                    # if tick == 230 and proba_move == 0.5:
                    #     print("mean_tw_flags_distances = sum_flags_distances/(time_window_end - time_window_start)", mean_tw_flags_distances, sum_flags_distances, time_window_end, time_window_start)

                    # Save flags for this run
                    self.write_flag_data_swarm(setup_name=setup_name_tick, run=run, n=n, time_steps=time_steps, flags=flags, permutated_agents_per_step=[], deleted_agents_per_step=deleted_agents_per_step, analysis_dir=analysis_dir, nb_moves_per_step=nb_moves_per_step)
                
                    # Save stats for this repetition
                    self.write_sliding_puzzle_repetitions(setup_name=setup_name, setup_name_tick=setup_name_tick, run=run, deletions=tick, proba_move=proba_move, n=n, mean_tw_flags_distances=mean_tw_flags_distances, analysis_dir=analysis_dir)

                    # Restore original grid
                    self.restore_deleted_agents(deleted_map_pos_agent=deleted_map_pos_agent)


        with open(analysis_dir['root']+"/verbose_debug.txt", 'a') as f:
            f.write(verbose_str)
            verbose_str = ""

    #---------------------------------------------------

    def write_sliding_puzzle_repetitions(self, setup_name, setup_name_tick, run, deletions, proba_move, n, mean_tw_flags_distances, analysis_dir):

        from learning_initializations import save_data_to_csv

        dir_name = f"{analysis_dir['data']}/data_all_repetitions/{setup_name}"
        file_name = f"/data_{setup_name}_stats_per_repetition.csv"
        if not (os.path.exists(dir_name)):
            os.makedirs(dir_name, exist_ok=True)
            save_data_to_csv(dir_name+file_name, [], header=["Run", "Setup", "Deletions", "Fluidity", "N", "Flags_distance"])

        data_one_repetition = [str(run), setup_name_tick, str(deletions), str(proba_move), str(n), str(mean_tw_flags_distances).strip()]
        # if deletions == 230 and proba_move == 0.5:
        #     print("mean_tw_flags_distances", n, mean_tw_flags_distances)
        
        save_data_to_csv(dir_name+file_name, [data_one_repetition])

    #---------------------------------------------------

    @staticmethod
    def setup_sliding_puzzle_phase1_VS_phase2(run, setup_name, nb_repetitions, sliding_puzzle_learning_best_inds_per_phase, sliding_puzzle_learning_ticks, sliding_puzzle_learning_proba_move, 
                                              grid_nb_rows, grid_nb_cols, learning_modes, learning_with_noise_std, flag_pattern, init_cell_state_value, nn_controller, time_steps,analysis_dir):
        global verbose_str

        # best_ind del=0 file 20-04-58 : 10,791,9493,1,0.0,"[-0.012560697342902754, -2.4218532064886547, -11.284764996702298, -3.83551410035861, 0.083622246643351, 5.5113126967703066, -0.0027867433240647416, -11.629436788302101, 5.009320936447007, -17.704798393117848, -10.896845332860472, -24.493805009057088, 2.5093953686045802, 5.108077537144663, 13.560729551675559, 1.3333431480092919]"
        # best_ind del=0.1 file 22-47-36 : 3,723,8677,2,0.0210600459373937,"[0.7551013499973216, -0.8099165077998569, 0.39403386311516464, -3.0751653808831234, -1.5472970783237112, 0.12214498389552143, 0.4242082294164468, -1.004712399587744, -4.648831756659906, -0.6800924749548437, -0.7739346754676364, 0.6693057234261393, -0.6723684639597962, 1.755774979565761, -1.171903351840752, 0.07096918320204165]"
        # best_ind del=[0.0, 0.1] file 22-46-52 : 6,792,9506,2,0.0115501975350683,"[-0.9489165395619286, -0.5070084158891089, -3.307401823857753, 1.2321234062873245, -0.6437943140104423, -0.5373127755880184, -1.2178955303840306, -0.9360873065469761, 1.9483769709668306, 0.581890195712445, -2.7724818107110933, -4.384791894596162, 0.7172862086219275, -0.04168444524174454, 2.29416315533213, -0.7267384958587466]"

        # sliding_puzzle_learning_best_inds_per_phase = [
        #     [-0.012560697342902754, -2.4218532064886547, -11.284764996702298, -3.83551410035861, 0.083622246643351, 5.5113126967703066, -0.0027867433240647416, -11.629436788302101, 5.009320936447007, -17.704798393117848, -10.896845332860472, -24.493805009057088, 2.5093953686045802, 5.108077537144663, 13.560729551675559, 1.3333431480092919],
        #     [0.7551013499973216, -0.8099165077998569, 0.39403386311516464, -3.0751653808831234, -1.5472970783237112, 0.12214498389552143, 0.4242082294164468, -1.004712399587744, -4.648831756659906, -0.6800924749548437, -0.7739346754676364, 0.6693057234261393, -0.6723684639597962, 1.755774979565761, -1.171903351840752, 0.07096918320204165],
        #     [-0.9489165395619286, -0.5070084158891089, -3.307401823857753, 1.2321234062873245, -0.6437943140104423, -0.5373127755880184, -1.2178955303840306, -0.9360873065469761, 1.9483769709668306, 0.581890195712445, -2.7724818107110933, -4.384791894596162, 0.7172862086219275, -0.04168444524174454, 2.29416315533213, -0.7267384958587466]
        # ]

        # best_ind del=[0.0, 0.15] file 04-09-13 : 9,1085,13025,2,0.0203556016817295,"[-2.3281762977773464, 1.0897984741831388, -3.4380823318850102, -5.364419682576576, 0.6450558907808008, 7.277072552073939, -2.3197060681368376, -0.2268261857899258, -2.682882037659171, -4.3759456639819065, -2.8980748544702246, 3.508211315465028, 2.0592799284359655, 1.7388241842799008, -0.8434797482496187, 4.358683203841405]"
        # best_ind del=0.15 file 04-13-33 : 0,1166,13998,2,0.045280784863846,"[-0.7838250959482946, -0.7064742791885482, -1.2267676994042034, -1.8622634802345577, -0.4974216044177394, 0.1125867910423979, 0.26435862083172573, -1.0407262077118675, 0.965503030713558, 0.8680916996522677, 0.12551901135563304, -0.46385517149270317, -1.0230676163591752, -0.7711729763800076, 0.6248420517282114, 0.1715900060404594]"

        # sliding_puzzle_learning_best_inds_per_phase = [
        #     [-2.3281762977773464, 1.0897984741831388, -3.4380823318850102, -5.364419682576576, 0.6450558907808008, 7.277072552073939, -2.3197060681368376, -0.2268261857899258, -2.682882037659171, -4.3759456639819065, -2.8980748544702246, 3.508211315465028, 2.0592799284359655, 1.7388241842799008, -0.8434797482496187, 4.358683203841405],
        #     [-0.7838250959482946, -0.7064742791885482, -1.2267676994042034, -1.8622634802345577, -0.4974216044177394, 0.1125867910423979, 0.26435862083172573, -1.0407262077118675, 0.965503030713558, 0.8680916996522677, 0.12551901135563304, -0.46385517149270317, -1.0230676163591752, -0.7711729763800076, 0.6248420517282114, 0.1715900060404594]
        # ]
        
        for n in range(nb_repetitions):
            # for best_ind_per_phase, phase in zip(sliding_puzzle_learning_best_inds_per_phase, [1, 2, 3]): # sliding_puzzle_learning_best_inds_per_phase = [best ind phase1, best ind phase2]
            for best_ind_per_phase, phase in zip(sliding_puzzle_learning_best_inds_per_phase, [1, 2]): # sliding_puzzle_learning_best_inds_per_phase = [best ind phase1, best ind phase2]

                # Grid creation with a new best ind
                new_env = init_swarmGrid_env(grid_nb_rows=grid_nb_rows,
                                             grid_nb_cols=grid_nb_cols,
                                             learning_modes=learning_modes,
                                             learning_with_noise_std=learning_with_noise_std,
                                             flag_pattern=flag_pattern,
                                             flag_target=None,
                                             init_cell_state_value=init_cell_state_value,
                                             agent_type=None,
                                             nn_controller=nn_controller,
                                             agent_controller_weights=best_ind_per_phase,
                                             nb_intrasteps=None,
                                             verbose_debug_bool=False,
                                             analysis_dir=analysis_dir)

                # This code is similar to setup_deletion as it manages the deletion of tiles:
                # Their position will be modified in step_random_async_update_sliding_puzzle.
                # In this setup, the automata update is always asynchrone (random_async_update)
                for tick in sliding_puzzle_learning_ticks:
                    # setup_name_tick_phase = f"{setup_name}_control_{tick}_phase{phase}"
                    setup_name_tick_phase = f"{setup_name}_{tick}_phase{phase}"
                    with_noise_bool = new_env.learning_with_noise_bool
                    noise_std = new_env.learning_with_noise_std
                    new_env.init_agents_state()
                    flags = []
                    deleted_agents_per_step = []
                    nb_moves_per_step = []

                    agents = new_env.get_agents()
                    agents_to_delete = random.sample(agents, tick)
                    deleted_map_pos_agent = new_env.delete_agent(agents_to_delete=agents_to_delete)

                    for _ in range(time_steps):
                        nb_moves = new_env.step_random_async_update_sliding_puzzle(agents_to_delete=agents_to_delete,
                                                                                   sliding_puzzle_proba_move=sliding_puzzle_learning_proba_move,
                                                                                   with_noise_bool=with_noise_bool,
                                                                                   noise_std=noise_std)
                        nb_moves_per_step.append(nb_moves)
                        deleted_agents_per_step.append([a.pos for a in agents_to_delete])
                        flag = new_env.get_flag_from_grid()
                        flags.append(flag)

                    # Save flags for this run
                    new_env.write_flag_data_swarm(setup_name=setup_name_tick_phase, run=run, n=n, time_steps=time_steps, flags=flags, permutated_agents_per_step=[], deleted_agents_per_step=deleted_agents_per_step, nb_moves_per_step=nb_moves_per_step, analysis_dir=analysis_dir)
                
                    # Restore original grid
                    new_env.restore_deleted_agents(deleted_map_pos_agent=deleted_map_pos_agent)


        with open(analysis_dir['root']+"/verbose_debug.txt", 'a') as f:
            f.write(verbose_str)
            verbose_str = ""

    #---------------------------------------------------

    @staticmethod
    def setup_scalability(run, setup_name, nb_repetitions, scalability_ticks, learning_modes, flags_distance_mode, learning_with_noise_std, flag_pattern, init_cell_state_value, nn_controller,
                         agent_controller_weights, time_steps, analysis_dir):

        global verbose_str

        for n in range(nb_repetitions):
            for tick in scalability_ticks:
                setup_name_tick = setup_name+"_"+str(tick[0])+"x"+str(tick[1])
                
                # Grid creation with a new size
                new_env = init_swarmGrid_env(grid_nb_rows=tick[0],
                                            grid_nb_cols=tick[1],
                                            learning_modes=learning_modes,
                                            flags_distance_mode=flags_distance_mode,
                                            learning_with_noise_std=learning_with_noise_std,
                                            flag_pattern=flag_pattern,
                                            flag_target=None,
                                            init_cell_state_value=init_cell_state_value,
                                            agent_type=None,
                                            nn_controller=nn_controller,
                                            agent_controller_weights=agent_controller_weights,
                                            nb_intrasteps=None,
                                            verbose_debug_bool=False,
                                            analysis_dir=analysis_dir)

                switch_step_with_random_async_update_bool = new_env.learning_random_async_update_states_bool
                with_noise_bool = new_env.learning_with_noise_bool
                noise_std = new_env.learning_with_noise_std
                new_env.init_agents_state()
                flags = []

                for _ in range(time_steps):
                    new_env.step(random_async_update_bool=switch_step_with_random_async_update_bool,
                                 with_noise_bool=with_noise_bool,
                                 noise_std=noise_std)
                    
                    flag = new_env.get_flag_from_grid()
                    flags.append(flag)
                
                # Save flags for this run
                new_env.write_flag_data_swarm(setup_name=setup_name_tick, run=run, n=n, time_steps=time_steps, flags=flags, permutated_agents_per_step=[], deleted_agents_per_step=[], analysis_dir=analysis_dir)

        with open(analysis_dir['root']+"/verbose_debug.txt", 'a') as f:
            f.write(verbose_str)
            verbose_str = ""

    #---------------------------------------------------

    def permutate_agents(self, agents_to_permutate):
        global verbose_str
        for i in range(0, len(agents_to_permutate), 2): # agents_to_permutate is a list of agents in shuffled order, even size

            if verbose_debug:
                verbose_str += f"\nPermutating the positions of 2 agents: agent1={agents_to_permutate[i]} (agent1.pos={agents_to_permutate[i].pos}) and agent2={agents_to_permutate[i+1]} (agent2.pos={agents_to_permutate[i+1].pos})"
                verbose_str += f"\ngrid_map_pos_agent before permutation: {self.grid_map_pos_agent}"
            
            # Permutation at the grid level
            self.grid_map_pos_agent[agents_to_permutate[i].pos] = agents_to_permutate[i+1]
            self.grid_map_pos_agent[agents_to_permutate[i+1].pos] = agents_to_permutate[i]
            
            # Permutation at the agent level (agent.pos)
            agent1_pos = agents_to_permutate[i].pos
            agents_to_permutate[i].pos = agents_to_permutate[i+1].pos
            agents_to_permutate[i+1].pos = agent1_pos
            
            if verbose_debug:
                verbose_str += f"\npermutated positions: agent1={agents_to_permutate[i]} (agent1.pos={agents_to_permutate[i].pos}) and agent2={agents_to_permutate[i+1]} (agent2.pos={agents_to_permutate[i+1].pos})"
                verbose_str += f"\ngrid_map_pos_agent after permutation: {self.grid_map_pos_agent}"

        # Update the neighbors list of each agent
        self.update_agent_neighbors()

    #---------------------------------------------------

    def delete_agent(self, agents_to_delete=None):
        
        deleted_map_pos_agent = {}
        for agent in agents_to_delete:
            deleted_map_pos_agent[agent.pos] = agent # save agent
            self.grid_map_pos_agent[agent.pos] = None # eliminate agent from grid

        # Update the neighbors list of each agent
        self.update_agent_neighbors()

        return deleted_map_pos_agent

    #---------------------------------------------------

    def restore_deleted_agents(self, deleted_map_pos_agent=None):
        empty_cells_list = [pos for pos in self.grid_map_pos_agent.keys() if self.grid_map_pos_agent[pos] == None]
        agents_to_restore = [a for a in deleted_map_pos_agent.values()]
        assert len(empty_cells_list) == len(agents_to_restore), f"\nrestore_deleted_agents, len(empty_cells_list) {len(empty_cells_list)} != len(agents_to_restore) {len(agents_to_restore)}"

        for i, empty_pos in enumerate(empty_cells_list):
            agents_to_restore[i].pos = empty_pos
            self.grid_map_pos_agent[empty_pos] = agents_to_restore[i] # check

        # for pos in deleted_map_pos_agent:
        #     self.grid_map_pos_agent[pos] = deleted_map_pos_agent[pos]

        # Update the neighbors list of each agent
        agents = self.get_agents()
        assert len(agents) == self.grid_size, f"\nrestore_deleted_agents, len(agents) {len(agents)} != self.grid_size {self.grid_size}"
        self.update_agent_neighbors()

    #---------------------------------------------------

    def get_flag_from_grid(self):
        flag = {}
        for pos in self.grid_map_pos_agent.keys(): # salvare la lista per ogni utilizzo
            if self.grid_map_pos_agent[pos] is not None:
                flag[pos] = self.grid_map_pos_agent[pos].get_phenotype()
            else:
                # flag[pos] = self.default_missing_neighbor_state
                flag[pos] = None

        return flag
    
    #---------------------------------------------------

    def get_flag_signals_from_grid(self):
        flag_signals = {}
        for pos in self.grid_map_pos_agent.keys(): # salvare la lista per ogni utilizzo
            if self.grid_map_pos_agent[pos] is not None:
                flag_signals[pos] = self.grid_map_pos_agent[pos].get_external_chemicals_to_spread()[0]
            else:
                # flag_signals[pos] = self.default_missing_neighbor_state
                flag_signals[pos] = None

        return flag_signals

    #---------------------------------------------------

    # def eval_flags_distance(self, flag):

    #     if self.flags_distance_mode == "MSE":
    #         sum_states = 0.0
    #         positions = self.grid_map_pos_agent.keys() # all positions in the grid
    #         assert len(positions) == self.grid_size, f"\eval_flags_distance, len(positions) {len(positions)} != grid_size {self.grid_size}"
    #         for p, pos in enumerate(positions):
    #             if self.grid_map_pos_agent[pos] is not None:
    #                 sum_states += (self.flag_target[p] - flag[pos])**2
    #         flags_distance = sum_states/self.grid_size

    #     elif self.flags_distance_mode == "SSIM": # flags_distance in [0 = most similar, 2 = strong dissimilarity]
    #         flag_list = self.convert_flag_to_list(flag)
    #         img1 = convert_flag_to_image(flag=flag_list, grid_nb_rows=self.grid_nb_rows, grid_nb_cols=self.grid_nb_cols)
    #         img2 = convert_flag_to_image(flag=self.flag_target, grid_nb_rows=self.grid_nb_rows, grid_nb_cols=self.grid_nb_cols)
    #         flags_distance = get_images_distance_SSIM(image_generated=img1, image_ref=img2)

    #     elif self.flags_distance_mode == "CLIP": # flags_distance in [0 = most similar, 2 = strong dissimilarity]
    #         flag_list = self.convert_flag_to_list(flag)
    #         img1 = convert_flag_to_image(flag=flag_list, grid_nb_rows=self.grid_nb_rows, grid_nb_cols=self.grid_nb_cols)
    #         img2 = convert_flag_to_image(flag=self.flag_target, grid_nb_rows=self.grid_nb_rows, grid_nb_cols=self.grid_nb_cols)
    #         flags_distance = get_images_distance_CLIP(image_generated=img1, image_ref=img2)
        
    #     else:
    #         print("Error in eval_flags_distance: invalid flags_distance_mode.")

    #     return flags_distance
    
    #---------------------------------------------------

    def eval_flags_distance(self, flag):

        sum_states = 0.0
        nb_agents = 0

        positions = self.grid_map_pos_agent.keys() # all positions in the grid
        # assert len(positions) == self.grid_size, f"\eval_flags_distance, len(positions) {len(positions)} != grid_size {self.grid_size}"

        if self.size_phenotype > 1: # flag coordinates (2D) or rgb (3D)
            for p, pos in enumerate(positions):
                if flag[pos] is not None:
                    for coordinate in range(self.size_phenotype):
                        sum_states += (self.flag_target[p][coordinate] - flag[pos][coordinate])**2
                    nb_agents += 1
        else:
            for p, pos in enumerate(positions):
                # if self.grid_map_pos_agent[pos] is not None:
                if flag[pos] is not None:
                    # print("self.flag_target[p]", self.flag_target[p]) #OK!
                    # print("flag[pos]", flag[pos]) # NOK c'est une liste!
                    sum_states += (self.flag_target[p] - flag[pos])**2
                    # print("eval_flags_distance:", "pos=", pos, self.flag_target[p], flag[pos], "-->",  ((self.flag_target[p] - flag[pos])**2))
                    nb_agents += 1

        if nb_agents == 0:
            return 0.0

        # flags_distance = sum_states/self.grid_size # deprecated
        flags_distance = (sum_states/self.size_phenotype)/nb_agents # agregation = sum composante x+y in [0,2] / nb agents presents
        # print("eval_flags_distance:", sum_states, "/", nb_agents, "= flags distance", flags_distance) 
        # print("flags_distance", flags_distance)
        return flags_distance
    
    #---------------------------------------------------

    def convert_flag_to_list(self, flag, size_pixel):
        
        if size_pixel == 1:
            flag_tmp = {k: (v if v is not None else self.default_missing_neighbor_state) for k, v in flag.items()}
        else:
            flag_tmp = {k: (v if v is not None else [self.default_missing_neighbor_state] * size_pixel) for k, v in flag.items()}

        return list(flag_tmp.values())

    #---------------------------------------------------
    
    def write_flag_target_data(self, analysis_dir, env_id=0):

        from learning_initializations import save_data_to_csv

        if not (os.path.exists(f"{analysis_dir['root']}/data_all_runs/data_env{env_id}_flag_target.csv")):
            save_data_to_csv(f"{analysis_dir['root']}/data_all_runs/data_env{env_id}_flag_target.csv", [[0, 0, 0, 0,  str(self.flag_target).strip(), 0]], header = ["Generation", "Step", "Flags_distance", "Time_window_zone", "Flag", "Individual"])

    #---------------------------------------------------
    
    def write_flag_data_learning(self, run, gen, nb_eval, nb_ind, time_steps, flags_distances, in_t_window_zone_bools, flags, flags_signals, weights, deleted_agents_per_step, nb_moves_per_step, analysis_dir):

        from learning_initializations import save_data_to_csv

        file_path = analysis_dir['data']+ f"/data_env0_flag/data_env_flag_run_{run:03}_gen_{gen:05}_eval_{nb_eval:07}.csv"
        if not (os.path.exists(file_path)):
            os.makedirs(analysis_dir['data']+"/data_env0_flag/", exist_ok=True)
            save_data_to_csv(file_path, [], header = ["Generation", "Nb_eval", "Nb_ind", "Step", "Flags_distance", "Time_window_zone", "Flag", "Flag_signals", "Individual", "Deleted_agents_positions", "Nb_moves"])
        
        self.clean_data_env_flag(analysis_dir=analysis_dir['data'])

        if deleted_agents_per_step is None:
            deleted_agents_per_step = [[] for _ in range(time_steps)]
            nb_moves_per_step = [[] for _ in range(time_steps)]
        
        data_env_flag = []
        for step in range(time_steps):
            data_env_flag.append([str(gen), str(nb_eval), str(nb_ind), str(step), str(flags_distances[step]).strip(), str(in_t_window_zone_bools[step]).strip(), str(flags[step]).strip(), str(flags_signals[step]).strip(), str(weights).strip(), str(deleted_agents_per_step[step]).strip(), str(nb_moves_per_step[step]).strip()])

        save_data_to_csv(file_path, data_env_flag)

    #---------------------------------------------------

    def write_multiEnvs_flag_data_learning(self, run, gen, nb_eval, nb_ind, time_steps, flags_distances, in_t_window_zone_bools, flags, flags_signals, weights, deleted_agents_per_step, nb_moves_per_step, analysis_dir):

        from learning_initializations import save_data_to_csv

        for env_id, _ in enumerate(flags_distances):
            file_path = analysis_dir['data']+ f"/data_env{env_id}_flag/data_env_flag_run_{run:03}_gen_{gen:05}_eval_{nb_eval:07}.csv"
            if not (os.path.exists(file_path)):
                os.makedirs(f"{analysis_dir['data']}/data_env{env_id}_flag/", exist_ok=True)
                save_data_to_csv(file_path, [], header = ["Generation", "Nb_eval", "Nb_ind", "Step", "Flags_distance", "Time_window_zone", "Flag", "Flag_signals", "Individual", "Deleted_agents_positions", "Nb_moves"])
            
            self.clean_data_env_flag(analysis_dir=analysis_dir['data'], env_id=env_id)

            if deleted_agents_per_step is None:
                deleted_agents_per_step = [[] for _ in range(time_steps)]
                nb_moves_per_step = [[] for _ in range(time_steps)]
            
            data_env_flag = []
            for step in range(time_steps):
                data_env_flag.append([str(gen), str(nb_eval), str(nb_ind), str(step), str(flags_distances[env_id][step]).strip(), str(in_t_window_zone_bools[step]).strip(), str(flags[env_id][step]).strip(), str(flags_signals[env_id][step]).strip(), str(weights).strip(), str(deleted_agents_per_step[env_id][step]).strip(), str(nb_moves_per_step[env_id][step]).strip()])

            save_data_to_csv(file_path, data_env_flag)

    #---------------------------------------------------

    def clean_data_env_flag(self, analysis_dir, max_len_dir=5, env_id=0):

        path = f"{analysis_dir}/data_env{env_id}_flag"
        files = sorted(os.listdir(path))
        if len(files) <= (max_len_dir*3): # to avoid erasing one by one
            return

        files = files[1:-(max_len_dir-1)] # always keep the 1st and the last flags
        if len(files) > 0:
            for f in files:
                os.remove(path+"/"+f)

    #---------------------------------------------------

    def write_controller_data_for_pogobots(self, run, gen, nb_eval, nb_ind, analysis_file):
        
        with open (analysis_file, 'w') as f:
            s = f"// flagAutomata individual controller run {run:03}, gen {gen:05}, eval {nb_eval:07}, nb_ind {nb_ind:03}\n"
            s += self.agent_controller[-1].get_weights_biases_for_pogobots()
            if self.agent_additional_weights:
                s += f"const double additional_weights[] = {(str(self.agent_additional_weights).replace('[','{')).replace(']','}')};\n"
            s += f"#define {self.agent_type.__name__}"
            f.write(s)

    #---------------------------------------------------

    def write_flag_data_swarm(self, setup_name, run, n, time_steps, flags, permutated_agents_per_step, deleted_agents_per_step, nb_moves_per_step, analysis_dir):

        from learning_initializations import save_data_to_csv

        dir_name = analysis_dir['data']+"/"+str(setup_name)
        file_name = f"/data_{setup_name}_flag_n_{n:03}.csv"
        if os.path.exists(dir_name+file_name):
            return
        if not (os.path.exists(dir_name)):
            os.makedirs(dir_name, exist_ok=True)

        data_env_flag = []
        for step in range(time_steps):
            flags_distance = self.eval_flags_distance(flags[step])
            # print(step, flags_distance)
            # print("\t", flags[step])

            if permutated_agents_per_step:
                #TODO: perché convert flag to list? Non é già list?
                data_env_flag.append([str(run), setup_name, str(n), str(step), str(flags_distance).strip(), str(self.convert_flag_to_list(flags[step], self.size_phenotype)).strip(), str(permutated_agents_per_step[step]).strip()])
                header = ["Run", "Setup", "N", "Step", "Flags_distance", "Flag", "Permutated_agents_positions"]
            elif deleted_agents_per_step:
                data_env_flag.append([str(run), setup_name, str(n), str(step), str(flags_distance).strip(), str(self.convert_flag_to_list(flags[step], self.size_phenotype)).strip(), str(deleted_agents_per_step[step]).strip(), str(nb_moves_per_step[step]).strip()])
                header = ["Run", "Setup", "N", "Step", "Flags_distance", "Flag", "Deleted_agents_positions", "Nb_moves"]
            else:
                data_env_flag.append([str(run), setup_name, str(n), str(step), str(flags_distance).strip(), str(self.convert_flag_to_list(flags[step], self.size_phenotype)).strip()])
                header = ["Run", "Setup", "N", "Step", "Flags_distance", "Flag"]

        save_data_to_csv(dir_name+file_name, data_env_flag, header=header)

    #---------------------------------------------------

    @staticmethod
    def plot_flag(grid_nb_rows, grid_nb_cols, setup_name, run, nb_ind, gen, nb_eval, n, step, flag_list, fitness, env_id, permutated_pos=[], deleted_pos=[], nb_moves_per_step=0, analysis_dir_plots=None):

        if isinstance(flag_list[0], (list, tuple)): # this means that we have a flag with N dimensions (called components)
            flag_components = swarmGrid.get_flag_components(flag_list)

            if len(flag_list[0]) == 3: # flag 3D, phenotype = [r,g,b] and r, g, b are floats in [0,1]
                flag_list = flag_components + [flag_list]
            else: # flag 2D, phenotype = [x,y] and x, y are floats in [0,1]
                flag_list = flag_components
        else: # flag 1D, phenotype = p and p is a float in [0,1]
            flag_list = [flag_list]

        for n_flag, flag in enumerate(flag_list):

            # Color profile detection: 
            first_elem_flag = flag[0]
            is_rgb_flag = isinstance(first_elem_flag, (list, tuple)) and len(first_elem_flag) == 3
            is_2d_or_1d_flag = (isinstance(first_elem_flag, float) or (isinstance(first_elem_flag, (list, tuple)) and len(first_elem_flag) == 2))

            if is_rgb_flag:
                color_mode = 'rgb'
            elif is_2d_or_1d_flag:
                if "signals" in analysis_dir_plots:
                    color_mode = 'monochrome2'
                else:
                    color_mode = 'monochrome1'
            else:
                raise ValueError("Error in evironments.py, plot_flag. Unrecognized flag structure: expected float, 1D/2D list, or RGB triplet.")

            fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

            grid_pos = []
            for row in range(grid_nb_rows):
                for col in range(grid_nb_cols):
                    grid_pos.append(tuple((row, col)))

            if permutated_pos:
                x_permutated = [pos[1] for pos in permutated_pos]
                y_permutated = [-pos[0] for pos in permutated_pos]

            if deleted_pos:
                x_deleted = [pos[1] for pos in deleted_pos]
                y_deleted = [-pos[0] for pos in deleted_pos]

            if grid_nb_rows <= 10 and grid_nb_cols <= 10:
                for row, col in [pos for pos in grid_pos if pos not in deleted_pos]:
                    for neighbor_pos in [(row-1, col), (row, col-1), (row, col+1), (row+1, col)]:
                        if swarmGrid.is_pos_valid(grid_nb_rows, grid_nb_cols, neighbor_pos) and neighbor_pos not in deleted_pos:
                            ax.plot([col, neighbor_pos[1]], [-row, -neighbor_pos[0]], color='black', linestyle=':', zorder=1)
                        
                circle_radius = 0.4


            x = []
            y = []
            grey_values = []
            for p, pos in enumerate(grid_pos):
                grey_value = flag[p]

                if grid_nb_rows > 10 or grid_nb_cols > 10:
                    x.append(pos[1])
                    y.append(-pos[0])
                    grey_values.append(grey_value)

                else:
                    edgecolor_color = swarmGrid.map_color(grey_value, color_mode)
                    facecolor_color = 'white'

                    if color_mode != 'rgb' and grey_value > 0.9: # close to white
                        circle = patches.Circle((pos[1], -pos[0]), circle_radius, edgecolor=edgecolor_color, facecolor=facecolor_color, linestyle='--', linewidth=1.0, zorder=2)
                    else:
                        circle = patches.Circle((pos[1], -pos[0]), circle_radius, edgecolor=edgecolor_color, facecolor=facecolor_color, linewidth=6.0, zorder=2)

                    if pos in permutated_pos:
                        circle = patches.Circle((pos[1], -pos[0]), circle_radius, edgecolor='tab:green', facecolor=facecolor_color, linestyle='--', linewidth=2.0, zorder=2)    

                    if pos in deleted_pos:
                        circle = patches.Circle((pos[1], -pos[0]), circle_radius, edgecolor='tab:red', facecolor=facecolor_color, linestyle='--', linewidth=2.0, zorder=2)    
                    
                    ax.add_patch(circle)

                    if grid_nb_rows < 6 and grid_nb_cols < 6:
                        ax.text(pos[1], -pos[0], "(" + str(pos[0]) +"," + str(pos[1]) + ")\n"+ str(round(grey_value,2)), color='black', va='center', ha='center')
                    
                    # Hide axis lines and ticks, but still show labels
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)


            if grid_nb_rows > 10 or grid_nb_cols > 10:
                if color_mode == 'monochrome1':
                    colors = [(0.0, 0.0, 0.0), (0.7, 0.9, 1.0)]  # black → light blue
                elif color_mode == 'monochrome2':
                    colors = [(0.0, 0.0, 0.0), (1.0, 0.4, 0.4)]  # black → light red
                else:  # RGB
                    colors = None

                if colors:
                    cmap = ListedColormap(np.linspace(colors[0], colors[1], 100))
                    plt.scatter(x, y, c=grey_values, cmap=cmap)
                else:
                    plt.scatter(x, y, c=grey_values) # use RGB values (already in 0–1 range)
                
                if permutated_pos:
                    plt.scatter(x_permutated, y_permutated, c='tab:green') # agents permutated

                if deleted_pos:
                    plt.scatter(x_deleted, y_deleted, c='tab:red') # deleted agents
        

            ax.set_aspect('equal')
            plt.xlim(-0.5, grid_nb_cols-0.5)
            plt.ylim(-grid_nb_rows+0.5, 0.5)
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.xlabel(f"nb moves during this step = {nb_moves_per_step}", fontsize=12)


            if setup_name:
                plt.title(f"Flag states - {setup_name}\nRun {run}, best individual {nb_ind}, step {step}.\nFlags distance = {fitness}", fontsize=10)
                dir_name = f"{analysis_dir_plots}/{setup_name}/flag/component{n_flag}"
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name, exist_ok=True)
                plt.savefig(f"{dir_name}/{setup_name}_flag_run_{run:03}_best_ind_{nb_ind:03}_n_{n:03}_step_{step:03}.png")
            else:
                if nb_ind is not None:
                    file_name = f"run_{run:03}_gen_{gen:05}_eval_{nb_eval:07}_individual_{nb_ind:03}"
                    plt.title(f"Flag states - learning.\nRun {run}, gen {gen}, nb_eval {nb_eval}, individual {nb_ind}, step {step}.\nFlags distance = {fitness}", fontsize=12)       
                    dir_name = f"{analysis_dir_plots}/{file_name}/flag/component{n_flag}/env{env_id}"
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name, exist_ok=True)
                    plt.savefig(f"{dir_name}/plot_env_flag_{file_name}_step_{step:03}.png")
                else:
                    plt.suptitle(f"Flag target {grid_nb_rows}x{grid_nb_cols}", fontsize=12)
                    plt.savefig(f"{analysis_dir_plots}/plot_env{env_id}_flag_target_component{n_flag}.png")

            plt.clf()
            plt.close()

    #---------------------------------------------------

    @staticmethod
    def map_color(value, mode):

        # clip values (color encoding). phenotypes are always in [0,1], but signals could be negatives
        if isinstance(value, list) or isinstance(value, tuple):
            value = [max(0.0, min(1.0, v)) for v in value]
        else:
            value = max(0.0, min(1.0, value))

        if mode == 'monochrome1':
            return (0.0, 0.5 * value, value)  # black to light blue
        elif mode == 'monochrome2':
            return (value, 0.3 * value, 0.3 * value)  # black to reddish
        elif mode == 'rgb':
            return value  # already (r, g, b)
        else:
            return (value, value, value)

    #---------------------------------------------------

    # flag_list becomes a list of flag components. [ [ x1, y1 ], [ x2, y2 ] ] -> [ [ x1, x2 ], [ y1, y2 ] ]
    @staticmethod
    def get_flag_components(flag_list):
        flag_components = []
        first_elem_size = len(flag_list[0])
        if isinstance(flag_list[0], (list, tuple)):
            for component in range(first_elem_size):
                tmp = [flag_list[i][component] for i in range(len(flag_list))]
                flag_components.append(tmp)
            return flag_components
        return None

    #---------------------------------------------------

    # We build a new flag, by computing the sum of its N components, representing the final flag 
    @staticmethod
    def merge_flag_components(flag_list):
        merged_flag = [] 
        nb_components = len(flag_list[0])
        for l in range(len(flag_list)):
            sum_components = 0
            for c in range(nb_components):
                sum_components += flag_list[l][c]/(nb_components)
            merged_flag.append(sum_components)
            
        return merged_flag

    #---------------------------------------------------

    @staticmethod
    def plot_flag_fitnesses_from_file(data_flag_file, setup_name, time_window_start, time_window_length, run, nb_ind, ind, n, gen, nb_eval, switch_step, analysis_dir_plots):

        df = pd.read_csv(data_flag_file)

        dataset_gen = df.loc[(df.Generation==gen)]
        dataset = dataset_gen.loc[(dataset_gen.Individual==str(ind))]

        x = dataset['Step'].tolist()
        y = dataset['Flags_distance'].tolist()

        _, ax = plt.subplots(figsize=(12, 7), dpi=300)

        plt.plot(x, y)

        if setup_name is not None:
            plt.axvline(x=switch_step, color='r', linestyle='--')
        else:
            rectangle = patches.Rectangle((time_window_start, 0), time_window_length, 1, linewidth=1, edgecolor=None, facecolor='lemonchiffon', alpha=0.5)
            ax.add_patch(rectangle)

        plt.ylim(-0.1, 1) # 0 and 1 are respectively min and max values of flag distance
        plt.yticks([0, 0.5, 1.0], fontsize=18)
        plt.xticks([0, 25, 49], fontsize=18)
        plt.xlabel("Steps", fontsize=18)
        plt.ylabel("Distance", fontsize=18)

        if setup_name:
            plt.title(f"Flags distance related to the flag development over steps\n{setup_name}, {n} repetitions", fontsize=14)
            dir_name = analysis_dir_plots+"/"+setup_name
            if not (os.path.exists(dir_name)):
                os.makedirs(dir_name, exist_ok=True)
            plt.savefig(f"{dir_name}/{setup_name}_flag_fitnesses_run_{run:03}_n_{n:03}.png")
        else:
            # plt.title(f"Flags distance related to the flag development over steps. Gen {gen}, individual {nb_ind}\nTime window zone from step {time_window_start} to step {time_window_start+time_window_length-1} (included).", fontsize=10)
            plt.title(f"Dynamic of the best candidate solution\nFlag development over time", fontsize=18)
            # plt.suptitle(f"Flag development over time", fontsize=18)
            plt.tight_layout()
            plt.savefig(f"{analysis_dir_plots}/run_{run:03}_gen_{gen:05}_eval_{nb_eval:07}_individual_{nb_ind:03}/flag_fitnesses_run_{run:03}_gen_{gen:05}_eval_{nb_eval:07}_individual_{nb_ind:03}.png")

        plt.clf()
        plt.close()

    #---------------------------------------------------

    @staticmethod
    def plot_multi_flag_fitnesses_from_file(data_flag_dir, setup_name, run, switch_step, analysis_dir_plots):

        data_flag_files = os.listdir(data_flag_dir)
        for data_flag_file in data_flag_files:
            dataset = pd.read_csv(data_flag_dir+"/"+data_flag_file)
            x = dataset['Step'].tolist()
            y = dataset['Flags_distance'].tolist()
            n = int(data_flag_file.split("n_")[1].split(".csv")[0])
            plt.plot(x, y, label=f"n_{n}")

            if switch_step and not(setup_name.startswith("setup_sliding_puzzle_phase1_VS_phase2")):
                plt.axvline(x=switch_step, color='r', linestyle='--')

        plt.ylim(-0.1, 1) # 0 and 1 are respectively min and max values of flag distance
        plt.xlabel("Steps", fontsize=12)
        plt.ylabel("Flags distance", fontsize=12)
        plt.title(f"Flags distance related to the flag development over steps. Run {run}\n{setup_name}, {len(data_flag_files)} repetitions", fontsize=12)
        plt.legend()

        dir_name = analysis_dir_plots+"/"+setup_name
        if not (os.path.exists(dir_name)):
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(f"{dir_name}/{setup_name}_flag_fitnesses_run_{run:03}.png")

        plt.clf()
        plt.close()

    #---------------------------------------------------

    # @staticmethod # TODO: not used for now
    # def plot_learning_sliding_puzzle_nb_moves_from_file(data_flag_file, run, grid_size, analysis_dir_plots):

    #     setup_name = "learning_sliding_puzzle"
    #     dataset = pd.read_csv(data_flag_file)
    #     x = dataset['Step'].tolist()
    #     y = dataset['Nb_moves'].tolist()
    #     plt.plot(x, y)

    #     plt.ylim(-0.1, grid_size) # 0 and 1 are respectively min and max values of flag distance
    #     plt.xlabel("Steps", fontsize=12)
    #     plt.ylabel("Nb moves", fontsize=12)
    #     plt.title(f"Number of agents' moves related to the flag development over steps. Run {run}\n{setup_name}", fontsize=12)

    #     dir_name = analysis_dir_plots+"/"+data_flag_file.replace()
    #     if not (os.path.exists(dir_name)):
    #         os.makedirs(dir_name, exist_ok=True)
    #     plt.savefig(f"{dir_name}/{setup_name}_flag_nb_moves_run_{run:03}.png")

    #     plt.clf()
    #     plt.close()

    #---------------------------------------------------

    @staticmethod
    def plot_nb_moves_from_file(data_flag_dir, setup_name, run, grid_size, switch_step, analysis_dir_plots):

        data_flag_files = os.listdir(data_flag_dir)
        for data_flag_file in data_flag_files:
            dataset = pd.read_csv(data_flag_dir+"/"+data_flag_file)
            x = dataset['Step'].tolist()
            y = dataset['Nb_moves'].tolist()
            n = int(data_flag_file.split("n_")[1].split(".csv")[0])
            plt.plot(x, y, label=f"n_{n}")


            if switch_step and not(setup_name.startswith("setup_sliding_puzzle_phase1_VS_phase2")):
                plt.axvline(x=switch_step, color='r', linestyle='--')

            plt.ylim(-0.1, grid_size) # 0 and 1 are respectively min and max values of flag distance
            plt.xlabel("Steps", fontsize=12)
            plt.ylabel("Nb moves", fontsize=12)
            plt.title(f"Number of agents' moves related to the flag development over steps. Run {run}\n{setup_name}, {len(data_flag_files)} repetitions", fontsize=12)
            plt.legend()

            dir_name = analysis_dir_plots+"/"+setup_name
            if not (os.path.exists(dir_name)):
                os.makedirs(dir_name, exist_ok=True)
            plt.savefig(f"{dir_name}/{setup_name}_flag_nb_moves_run_{run:03}_n_{n:03}.png")

            plt.clf()
            plt.close()

    #---------------------------------------------------

    @staticmethod
    def plot_multi_nb_moves_from_file(data_flag_dir, setup_name, run, grid_size, switch_step, analysis_dir_plots):

        data_flag_files = os.listdir(data_flag_dir)
        for data_flag_file in data_flag_files:
            dataset = pd.read_csv(data_flag_dir+"/"+data_flag_file)
            x = dataset['Step'].tolist()
            y = dataset['Nb_moves'].tolist()
            n = int(data_flag_file.split("n_")[1].split(".csv")[0])
            plt.plot(x, y, label=f"n_{n}")

        if switch_step and not(setup_name.startswith("setup_sliding_puzzle_phase1_VS_phase2")):
            plt.axvline(x=switch_step, color='r', linestyle='--')

        plt.ylim(-0.1, grid_size) # 0 and 1 are respectively min and max values of flag distance
        plt.xlabel("Steps", fontsize=12)
        plt.ylabel("Nb moves", fontsize=12)
        plt.title(f"Number of agents' moves related to the flag development over steps. Run {run}\n{setup_name}, {len(data_flag_files)} repetitions", fontsize=12)
        plt.legend()

        dir_name = analysis_dir_plots+"/"+setup_name
        if not (os.path.exists(dir_name)):
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(f"{dir_name}/{setup_name}_flag_nb_moves_run_{run:03}.png")

        plt.clf()
        plt.close()

    #---------------------------------------------------

    @staticmethod
    def plot_merged_multi_flag_fitnesses_from_file(data_flag_dirs, setups_names, run, analysis_dir_plots):

        plot_colors = ['#0173b2', '#de8f05'] # blue, orange
        for setup, data_flag_dirs_phase1_and_phase2 in enumerate(data_flag_dirs): # data_flag_dirs = [[data setup0 phase1, data setup0 phase2], [data setup1 phase1, data setup1 phase2]]
            for phase, data_flag_dir in enumerate(data_flag_dirs_phase1_and_phase2):
                data_flag_files = os.listdir(data_flag_dir)
                for data_flag_file in data_flag_files:
                    dataset = pd.read_csv(data_flag_dir+"/"+data_flag_file)
                    x = dataset['Step'].tolist()
                    y = dataset['Flags_distance'].tolist()
                    n = int(data_flag_file.split("n_")[1].split(".csv")[0])
                    plt.plot(x, y, color=plot_colors[phase], label=f"best_ind_phase_{phase+1}" if n == 0 else "")

            plt.ylim(-0.1, 1) # 0 and 1 are respectively min and max values of flag distance
            plt.xlabel("Steps", fontsize=12)
            plt.ylabel("Flags distance", fontsize=12)
            plt.title(f"Flags distance related to the flag development over steps. Run {run}\n{setups_names[setup]}, {len(data_flag_files)} repetitions", fontsize=12)
            plt.legend()

            dir_name = f"{analysis_dir_plots}/{setups_names[setup]}"
            if not (os.path.exists(dir_name)):
                os.makedirs(dir_name, exist_ok=True)
            plt.savefig(f"{dir_name}/{setups_names[setup]}_merged_flag_fitnesses_run_{run:03}.png")

            plt.clf()
            plt.close()


    #---------------------------------------------------

    # @staticmethod
    # def plot_merged_multi_flag_fitnesses_from_file(data_flag_dirs, setups_names, run, analysis_dir_plots):

    #     plot_colors = ['#0173b2', '#de8f05', '#cc78bc'] # blue, orange
    #     labels = ["best_ind d=1.0", "best_ind d=0.9", "best_ind d=1.0 -> d=0.9"]
    #     for setup, data_flag_dirs_phase1_and_phase2 in enumerate(data_flag_dirs): # data_flag_dirs = [[data setup0 phase1, data setup0 phase2], [data setup1 phase1, data setup1 phase2]]
    #         for phase, data_flag_dir in enumerate(data_flag_dirs_phase1_and_phase2):
    #             data_flag_files = os.listdir(data_flag_dir)
    #             for data_flag_file in data_flag_files:
    #                 dataset = pd.read_csv(data_flag_dir+"/"+data_flag_file)
    #                 x = dataset['Step'].tolist()
    #                 y = dataset['Flags_distance'].tolist()
    #                 n = int(data_flag_file.split("n_")[1].split(".csv")[0])
    #                 plt.plot(x, y, color=plot_colors[phase], label=labels[phase] if n == 0 else "")
    #                 print(labels[phase], data_flag_dir)

    #         plt.ylim(-0.1, 1) # 0 and 1 are respectively min and max values of flag distance
    #         plt.xlabel("Steps", fontsize=12)
    #         plt.ylabel("Flags distance", fontsize=12)
    #         plt.title(f"Flags distance related to the flag development over steps. Run {run}\n{setups_names[setup]}, {len(data_flag_files)} repetitions", fontsize=12)
    #         plt.legend()

    #         dir_name = f"{analysis_dir_plots}/{setups_names[setup]}"
    #         if not (os.path.exists(dir_name)):
    #             os.makedirs(dir_name, exist_ok=True)
    #         plt.savefig(f"{dir_name}/{setups_names[setup]}_control_merged_flag_fitnesses_run_{run:03}.png")

    #         plt.clf()
    #         plt.close()

