import os

import random
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend instead of QtAgg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
#from flags_distance_methods import convert_flag_to_image, get_images_distance_MSE, get_images_distance_SSIM, get_images_distance_CLIP


###########################################################################
# Global variables
###########################################################################

verbose_debug = False
verbose_str = ""

env = None

###########################################################################
# Activation functions
###########################################################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def init_swarmGrid_env(grid_nb_rows, grid_nb_cols, learning_modes, flags_distance_mode, learning_with_noise_std, flag_pattern, flag_target, init_cell_state_value, nn_controller, agent_controller_weights, verbose_debug_bool, analysis_dir):
    
    global verbose_debug, env

    verbose_debug = verbose_debug_bool
    with open(analysis_dir['root']+"/verbose_debug.txt", 'w') as f: # to replace or erase eventual previous existing content in verbose_debug.txt
        f.write(verbose_str)

    if env is None:
        env = swarmGrid(grid_nb_rows=grid_nb_rows,
                        grid_nb_cols=grid_nb_cols,
                        learning_modes=learning_modes,
                        flags_distance_mode=flags_distance_mode,
                        learning_with_noise_std=learning_with_noise_std,
                        flag_pattern=flag_pattern,
                        flag_target=flag_target,
                        init_cell_state_value=init_cell_state_value,
                        nn_controller=nn_controller)
    
    env.write_flag_target_data(analysis_dir) # check emplacement
    
    # Some genomes could have 2 components, one dedicated to the controller NN and one for other purpose. Ex: in Devert 2011, 4 weights are required for the expression function
    agent_additional_weights = None
    nn_controller_weights_size = nn_controller.weights_biases_size
    ind_size = len(agent_controller_weights)
    if (ind_size > nn_controller_weights_size):
        agent_controller_weights = agent_controller_weights[:nn_controller_weights_size]
        agent_additional_weights = agent_controller_weights[nn_controller_weights_size:]
    
    # grid and agent re-initialization are required for each new individual (new agent_controller_weights) before evaluation
    env.set_agent_controller_weights(agent_controller_weights=agent_controller_weights, agent_additional_weights=agent_additional_weights)
    env.init_agents_state() # to execute at last in this function

    return env


###########################################################################
# Swarm agents
###########################################################################

class swarmAgent:
    def __init__(self, pos, size_state, init_cell_state_value=None) -> None:

        self.pos = pos
        self.size_state = size_state
        self.init_cell_state_value = init_cell_state_value
        self.neighbors_NWES = None
        self.state = None

    #---------------------------------------------------
    
    def init_state(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    #---------------------------------------------------

    def set_state(self, vector, with_noise_bool, noise_std):

        if with_noise_bool:
            state = vector.copy()
            noise = np.random.normal(0, noise_std, self.size_state) 
            vector = [self.clip(state[i]+noise[i]) for i in range(self.size_state)] # check  

        self.state = vector

    #---------------------------------------------------

    def clip(self, value):
        return max(0, min(value, 1))

    #---------------------------------------------------

    def get_state(self):
        return self.state
    
    #---------------------------------------------------

    def get_external_chemicals_to_spread(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    #---------------------------------------------------
    
    def get_phenotype(self):
        raise NotImplementedError("Subclasses must implement this method")


###########################################################################

class agent1Output(swarmAgent):
    def __init__(self, pos, init_cell_state_value, agent_additional_weights=None):
        self.size_state = 1
        self.size_chemicals_to_spread = 1
        super().__init__(pos=pos, size_state=self.size_state, init_cell_state_value=init_cell_state_value)
        self.init_state()

    #---------------------------------------------------

    def init_state(self, random_init_bool=False):
        
        if self.init_cell_state_value is None or random_init_bool:
            self.state = np.random.uniform(0, 1, self.size_state).tolist()
        else:
            self.state = [self.init_cell_state_value] * self.size_state

    #---------------------------------------------------

    def set_state(self, vector, with_noise_bool, noise_std):
        super().set_state(vector, with_noise_bool, noise_std)
    
    #---------------------------------------------------

    def get_external_chemicals_to_spread(self):
        state = super().get_state()
        return state # list of floats
    
    #---------------------------------------------------

    def get_phenotype(self):
        state = super().get_state()
        return (state[0] + 1) / 2 # (x+1)/2 to rescale (-1,1) in (0,1) keeping the scale. Issue with sigmoid(state[0]): sigmoid(x) with x in (-1, 1) returned inconvenient bounded result in [0.27, 0.73] and we need phenotype in [0,1]

###########################################################################

class agent2Outputs(swarmAgent):
    def __init__(self, pos, init_cell_state_value, agent_additional_weights=None):
        self.size_state = 2
        self.size_chemicals_to_spread = 1
        super().__init__(pos=pos, size_state=self.size_state, init_cell_state_value=init_cell_state_value)
        self.init_state()

    #---------------------------------------------------

    def init_state(self, random_init_bool=False):

        if self.init_cell_state_value is None or random_init_bool:
            self.state = np.random.uniform(-1, 1, self.size_state).tolist()
            self.state[1] = np.abs(self.state[1])
        else:
            self.state = [self.init_cell_state_value] * self.size_state

    #---------------------------------------------------

    def set_state(self, vector, with_noise_bool, noise_std):
        super().set_state(vector, with_noise_bool, noise_std)
    
    #---------------------------------------------------

    def get_external_chemicals_to_spread(self):
        state = super().get_state()
        return [state[0]] # list of floats
    
    #---------------------------------------------------

    def get_phenotype(self):
        state = super().get_state()
        return (state[1] + 1) / 2 # (x+1)/2 to rescale (-1,1) in (0,1) keeping the scale. Issue with sigmoid(state[1]): sigmoid(x) with x in (-1, 1) returned inconvenient bounded result in [0.27, 0.73] and we need phenotype in [0,1]
    

###########################################################################

class agent3Outputs_Devert2011(swarmAgent):
    def __init__(self, pos, init_cell_state_value, agent_additional_weights=None):
        self.size_state = 3
        self.size_chemicals_to_spread = 2
        self.agent_additional_weights = agent_additional_weights
        super().__init__(pos=pos, size_state=self.size_state, init_cell_state_value=init_cell_state_value)
        self.init_state()

    #---------------------------------------------------

    def init_state(self, random_init_bool=False):
        if self.init_cell_state_value is None or random_init_bool:
            self.state = np.random.uniform(-1, 1, self.size_state).tolist() # Devert, 2011. State in [-1,1] because phenotype is not included in the state
        else:
            self.state = [self.init_cell_state_value] * self.size_state

    #---------------------------------------------------

    def set_state(self, vector, with_noise_bool, noise_std):
        super().set_state(vector, with_noise_bool, noise_std)
    
    #---------------------------------------------------

    def get_external_chemicals_to_spread(self): # Devert, 2011. A state is 2 external chemicals + 1 internal chemicals
        state = super().get_state()
        return state[:self.size_chemicals_to_spread]

    #---------------------------------------------------

    def get_internal_chemicals(self): # Devert, 2011. A state is 2 external chemicals + 1 internal chemicals
        state = super().get_state()
        return state[self.size_chemicals_to_spread:] # list of floats
    
    #---------------------------------------------------

    def get_phenotype(self): # Devert, 2011. Expression function
        state = super().get_state()
        
        val = 0
        for i in range(len(state)):
            val += self.agent_additional_weights[i] * state[i]
        val += self.agent_additional_weights[-1] # bias neuron

        return (0.5 * (1 + np.tanh(val))) # float
    
    #---------------------------------------------------

    def get_agent_energy(self): # Devert, 2011. Expression function: energy is the square root of the sum of the squared values of chemicals u and v
        state = super().get_state()
        return (state[0]**2 + state[1]**2 + state[2]**2)**0.5
                

###########################################################################
# Learning evaluation function
###########################################################################

def flag_automata(env_eval_function_params, analysis_dir, run, gen, nb_eval, nb_ind, best_fit, weights, sliding_puzzle_nb_deletions, sliding_puzzle_proba_move):
    time_steps = env_eval_function_params['time_steps']
    time_window_start = env_eval_function_params['time_window_start']
    time_window_end = env_eval_function_params['time_window_end']

    flags_distances = []
    in_t_window_zone_bools = []
    flags = []

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
                             nn_controller=env_eval_function_params['controller'],
                             agent_controller_weights=weights,
                             verbose_debug_bool=env_eval_function_params['verbose_debug'],
                             analysis_dir=env_eval_function_params['analysis_dir'])

    flags_distance = 0.0
    sum_flags_distances = 0.0
    for step in range(time_steps):

        in_t_window_zone_bool = False
        flag = env.get_flag_from_grid()
        flags.append(env.convert_flag_to_list(flag))
        flags_distance = env.eval_flags_distance(flag) #check
        
        if step >= time_window_start and step <= time_window_end:
            in_t_window_zone_bool = True
            sum_flags_distances += flags_distance

        flags_distances.append(flags_distance)
        in_t_window_zone_bools.append(in_t_window_zone_bool)

        env.step(random_async_update_bool=random_async_update_bool,
                 with_noise_bool=with_noise_bool,
                 noise_std=noise_std)
        
    mean_tw_flags_distances = sum_flags_distances/(time_window_end - time_window_start)
    if mean_tw_flags_distances < best_fit:
        env.write_flag_data_learning(run=run, gen=gen, nb_eval=nb_eval, nb_ind=nb_ind, time_steps=time_steps, flags_distances=flags_distances, in_t_window_zone_bools=in_t_window_zone_bools, flags=flags, weights=weights, deleted_agents_per_step=None, nb_moves_per_step=None, analysis_dir=analysis_dir)
        env.write_controller_data_for_pogobots(run=run, gen=gen, nb_eval=nb_eval, nb_ind=nb_ind, analysis_file=analysis_dir['data']+ f"/data_env_run_{run:03}_individual_controller_pogobots.txt") # overwrite previous saved files, to keep the best ind controller

    return (mean_tw_flags_distances,) # it is important to return a tuple (deap framework)

#---------------------------------------------------

def sliding_puzzle_incremental(env_eval_function_params, analysis_dir, run, gen, nb_eval, nb_ind, best_fit, weights, sliding_puzzle_nb_deletions, sliding_puzzle_proba_move):
    global verbose_str
    
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
                             nn_controller=env_eval_function_params['controller'],
                             agent_controller_weights=weights,
                             verbose_debug_bool=env_eval_function_params['verbose_debug'],
                             analysis_dir=env_eval_function_params['analysis_dir'])

    #self.init_agents_state() # fait at init env

    in_t_window_zone_bools = []
    flags = []

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
        flags.append(env.convert_flag_to_list(flag))
        flags_distance = env.eval_flags_distance(flag)
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
        env.write_flag_data_learning(run=run, gen=gen, nb_eval=nb_eval, nb_ind=nb_ind, time_steps=time_steps, flags_distances=flags_distances, in_t_window_zone_bools=in_t_window_zone_bools, flags=flags, weights=weights, deleted_agents_per_step=deleted_agents_per_step, nb_moves_per_step=nb_moves_per_step, analysis_dir=analysis_dir)
        env.write_controller_data_for_pogobots(run=run, gen=gen, nb_eval=nb_eval, nb_ind=nb_ind, analysis_file=analysis_dir['data']+ f"/data_env_run_{run:03}_individual_controller_pogobots.txt") # overwrite previous saved files, to keep the best ind controller

    # Restore original grid
    env.restore_deleted_agents(deleted_map_pos_agent=deleted_map_pos_agent) # i have a new grid each time?

    with open(analysis_dir['root']+"/verbose_debug.txt", 'a') as f:
        f.write(verbose_str)
        verbose_str = ""

    return (mean_tw_flags_distances,) # it is important to return a tuple (deap framework)


###########################################################################
# Environment swarmGrid
###########################################################################

class swarmGrid:
    def __init__(self, grid_nb_rows, grid_nb_cols, learning_modes, flags_distance_mode, learning_with_noise_std, flag_pattern, flag_target, init_cell_state_value, nn_controller) -> None:
      
        self.grid_nb_rows = grid_nb_rows
        self.grid_nb_cols = grid_nb_cols
        self.grid_size = grid_nb_rows * grid_nb_cols

        self.agent_controller_weights = None
        self.agent_additional_weights = None
        self.agent_controller = nn_controller

        if nn_controller.output_size == 1:
            self.agent_type = agent1Output
        elif nn_controller.output_size == 2:
            self.agent_type = agent2Outputs
        elif nn_controller.output_size == 3:
            self.agent_type = agent3Outputs_Devert2011

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
        # self.agent_controller.setWeightsFromList(agent_controller_weights)
        self.agent_controller.set_weights_biases_vectors_from_list(agent_controller_weights)

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
            horizontal_threshold_upper = int(self.grid_nb_rows/3)
            horizontal_threshold_lower = int(self.grid_nb_rows/3*2)

            for cell in self.grid_map_pos_agent.keys():
                if cell[0] < horizontal_threshold_upper:
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
                    radius = np.floor(min((self.grid_nb_rows/2), (self.grid_nb_cols/2)))
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

        return self.convert_flag_to_list(flag_target)

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

        if random_async_update_bool:
            np.random.shuffle(agents) # random update

            for agent in agents:
                state = self.compute_agent_state(agent=agent)
                agent.set_state(state, with_noise_bool, noise_std)

        else: # sync_update
            grid_map_agent_state_tmp = {}
            for agent in agents:
                state = self.compute_agent_state(agent=agent)
                grid_map_agent_state_tmp[agent] = state

            if verbose_debug:
                verbose_str += f"\n<step> sync update. Grid states at t-1: {self.get_flag_from_grid()}"
                verbose_str += f"\n<step> sync update. Temporary grid states at t: {grid_map_agent_state_tmp.values()}"

            for agent in agents:
                agent.set_state(grid_map_agent_state_tmp[agent], with_noise_bool, noise_std)

            if verbose_debug:
                verbose_str += f"\n<step> sync update. Grid states at t+1: {self.get_flag_from_grid()}"

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
        
        if agent is None: # check
            return

        neighbors_states = []
        for neighbor in agent.neighbors_NWES: # est il à jour? calcolarlo ora?
            if neighbor is not None: # si il a un id, il y a l'agent? tjr?
                neighbors_states += neighbor.get_external_chemicals_to_spread()
            else:
                neighbors_states += [self.default_missing_neighbor_state] * agent.size_chemicals_to_spread
        
        if (self.agent_type == agent3Outputs_Devert2011):
            neighbors_states += agent.get_internal_chemicals()
        
        # print("final neighbors states", neighbors_states, "len neighbors states", len(neighbors_states))

        # print("compute_agent_state: agent.pos:", agent.pos, ", its neighbors:", agent.neighbors_NWES, "neighbors states:", neighbors_states)


        # print("prima di set_state - state:", agent.state, "agent.get_external_chemicals_to_spread():", agent.get_external_chemicals_to_spread(), "agent.get_phenotype():", agent.get_phenotype())
        state = self.agent_controller.predict(neighbors_states) # forwardPropagation, stableSigmoid on the last layer       
        # print("compute_agent_state: agent.pos:", agent.pos, "neighbors states:", neighbors_states, "state:", state)
        # print(self.agent_controller.getWeightsList())
        
        return state
        # print("dopo di set_state - state:", agent.state, "agent.get_external_chemicals_to_spread():", agent.get_external_chemicals_to_spread(), "agent.get_phenotype():", agent.get_phenotype())

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
                                             nn_controller=nn_controller,
                                             agent_controller_weights=best_ind_per_phase,
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
                                            nn_controller=nn_controller,
                                            agent_controller_weights=agent_controller_weights,
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
        for p, pos in enumerate(positions):
            # if self.grid_map_pos_agent[pos] is not None:
            if flag[pos] is not None:
                sum_states += (self.flag_target[p] - flag[pos])**2
                # print("eval_flags_distance:", "pos=", pos, self.flag_target[p], flag[pos], "-->",  ((self.flag_target[p] - flag[pos])**2))
                nb_agents += 1

        # flags_distance = sum_states/self.grid_size
        if nb_agents == 0:
            return 0.0

        flags_distance = sum_states/nb_agents
        # print("eval_flags_distance:", sum_states, "/", nb_agents, "= flags distance", flags_distance) 
        return flags_distance
    
    #---------------------------------------------------

    def convert_flag_to_list(self, flag):
        flag_tmp = {k: (v if v is not None else self.default_missing_neighbor_state) for k, v in flag.items()}
        return list(flag_tmp.values())

    #---------------------------------------------------
    
    def write_flag_target_data(self, analysis_dir):

        from learning_initializations import save_data_to_csv

        if not (os.path.exists(analysis_dir['root']+"/data_all_runs/data_env_flag_target.csv")):
            save_data_to_csv(analysis_dir['root']+"/data_all_runs/data_env_flag_target.csv", [[0, 0, 0, 0,  str(self.flag_target).strip(), 0]], header = ["Generation", "Step", "Flags_distance", "Time_window_zone", "Flag", "Individual"])

    #---------------------------------------------------
    
    def write_flag_data_learning(self, run, gen, nb_eval, nb_ind, time_steps, flags_distances, in_t_window_zone_bools, flags, weights, deleted_agents_per_step, nb_moves_per_step, analysis_dir):

        from learning_initializations import save_data_to_csv

        file_path = analysis_dir['data']+ f"/data_env_flag/data_env_flag_run_{run:03}_gen_{gen:05}_eval_{nb_eval:07}.csv"
        if not (os.path.exists(file_path)):
            os.makedirs(analysis_dir['data']+"/data_env_flag/", exist_ok=True)
            save_data_to_csv(file_path, [], header = ["Generation", "Nb_eval", "Nb_ind", "Step", "Flags_distance", "Time_window_zone", "Flag", "Individual", "Deleted_agents_positions", "Nb_moves"])
        
        if deleted_agents_per_step is None:
            deleted_agents_per_step = [[] for _ in range(time_steps)]
            nb_moves_per_step = [[] for _ in range(time_steps)]
        
        data_env_flag = []
        for step in range(time_steps):
            data_env_flag.append([str(gen), str(nb_eval), str(nb_ind), str(step), str(flags_distances[step]).strip(), str(in_t_window_zone_bools[step]).strip(), str(flags[step]).strip(), str(weights).strip(), str(deleted_agents_per_step[step]).strip(), str(nb_moves_per_step[step]).strip()])

        save_data_to_csv(file_path, data_env_flag)

    #---------------------------------------------------
        
    def write_controller_data_for_pogobots(self, run, gen, nb_eval, nb_ind, analysis_file):
        
        with open (analysis_file, 'w') as f:
            s = f"// flagAutomata individual controller run {run:03}, gen {gen:05}, eval {nb_eval:07}, nb_ind {nb_ind:03}\n"
            s += self.agent_controller.get_weights_biases_for_pogobots()
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
                data_env_flag.append([str(run), setup_name, str(n), str(step), str(flags_distance).strip(), str(self.convert_flag_to_list(flags[step])).strip(), str(permutated_agents_per_step[step]).strip()])
                header = ["Run", "Setup", "N", "Step", "Flags_distance", "Flag", "Permutated_agents_positions"]
            elif deleted_agents_per_step:
                data_env_flag.append([str(run), setup_name, str(n), str(step), str(flags_distance).strip(), str(self.convert_flag_to_list(flags[step])).strip(), str(deleted_agents_per_step[step]).strip(), str(nb_moves_per_step[step]).strip()])
                header = ["Run", "Setup", "N", "Step", "Flags_distance", "Flag", "Deleted_agents_positions", "Nb_moves"]
            else:
                data_env_flag.append([str(run), setup_name, str(n), str(step), str(flags_distance).strip(), str(self.convert_flag_to_list(flags[step])).strip()])
                header = ["Run", "Setup", "N", "Step", "Flags_distance", "Flag"]

        save_data_to_csv(dir_name+file_name, data_env_flag, header=header)

    #---------------------------------------------------

    @staticmethod
    def plot_flag(grid_nb_rows, grid_nb_cols, setup_name, run, nb_ind, gen, nb_eval, n, step, flag, fitness, permutated_pos=[], deleted_pos=[], nb_moves_per_step=0, analysis_dir_plots=None):
        
        fig, ax = plt.subplots()

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
                if grey_value > 0.9: # close to white
                    circle = patches.Circle((pos[1], -pos[0]), circle_radius, edgecolor='black', facecolor='white', linestyle='--', linewidth=1.0, zorder=2)    
                else:
                    circle = patches.Circle((pos[1], -pos[0]), circle_radius, edgecolor=str(grey_value), facecolor='white', linewidth=6.0, zorder=2)

                if pos in permutated_pos:
                    circle = patches.Circle((pos[1], -pos[0]), circle_radius, edgecolor='tab:green', facecolor='white', linestyle='--', linewidth=2.0, zorder=2)    

                if pos in deleted_pos:
                    circle = patches.Circle((pos[1], -pos[0]), circle_radius, edgecolor='tab:red', facecolor='white', linestyle='--', linewidth=2.0, zorder=2)    
                
                ax.add_patch(circle)

                if grid_nb_rows < 6 and grid_nb_cols < 6:
                    ax.text(pos[1], -pos[0], "(" + str(pos[0]) +"," + str(pos[1]) + ")\n"+ str(round(grey_value,2)), color='black', va='center', ha='center')
                
                # Hide axis lines and ticks, but still show labels
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)


        if grid_nb_rows > 10 or grid_nb_cols > 10:
            colors = [(0.0, 0.0, 0.0), (0.7, 0.9, 1.0)]  # black to light blue
            cmap = ListedColormap(np.linspace(colors[0], colors[1], 100))
            plt.scatter(x, y, c=grey_values, cmap=cmap) # cmap='grey'

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
            dir_name = analysis_dir_plots+"/"+setup_name+"/flag"
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            plt.savefig(f"{dir_name}/{setup_name}_flag_run_{run:03}_best_ind_{nb_ind:03}_n_{n:03}_step_{step:03}.png")
        else:
            if nb_ind is not None:
                file_name = f"run_{run:03}_gen_{gen:05}_eval_{nb_eval:07}_individual_{nb_ind:03}"
                plt.title(f"Flag states - learning.\nRun {run}, gen {gen}, nb_eval {nb_eval}, individual {nb_ind}, step {step}.\nFlags distance = {fitness}", fontsize=12)       
                dir_name = analysis_dir_plots+ f"/{file_name}/flag"
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name, exist_ok=True)
                plt.savefig(f"{dir_name}/plot_env_flag_{file_name}_step_{step:03}.png")
            else:
                plt.suptitle(f"Flag target {grid_nb_rows}x{grid_nb_cols}", fontsize=12)
                plt.savefig(f"{analysis_dir_plots}/plot_env_flag_target.png")

        plt.clf()
        plt.close()

    #---------------------------------------------------

    @staticmethod
    def plot_flag_fitnesses_from_file(data_flag_file, setup_name, time_window_start, time_window_length, run, nb_ind, ind, n, gen, nb_eval, switch_step, analysis_dir_plots):

        df = pd.read_csv(data_flag_file)

        dataset_gen = df.loc[(df.Generation==gen)]
        dataset = dataset_gen.loc[(dataset_gen.Individual==str(ind))]

        x = dataset['Step'].tolist()
        y = dataset['Flags_distance'].tolist()

        _, ax = plt.subplots()

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

