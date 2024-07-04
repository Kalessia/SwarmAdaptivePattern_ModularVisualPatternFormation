import os

import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap


###########################################################################
# Global variables
###########################################################################

global env
env = None


def init_swarmGrid_env(grid_nb_rows, grid_nb_cols, flag_pattern, flag_target, init_cell_state_value, nn_controller, agent_controller_weights):
    global env
    if env is None:
        env = swarmGrid(grid_nb_rows=grid_nb_rows,
                        grid_nb_cols=grid_nb_cols,
                        flag_pattern=flag_pattern,
                        flag_target=flag_target,
                        init_cell_state_value=init_cell_state_value,
                        nn_controller=nn_controller)
    
    env.set_agent_controller_weights(agent_controller_weights=agent_controller_weights)
    env.init_agents_state()

    return env


###########################################################################
# Swarm agents
###########################################################################

class swarmAgent:
    def __init__(self, pos, len_state, init_cell_state_value=None) -> None:

        self.pos = pos
        self.len_state = len_state
        self.init_cell_state_value = init_cell_state_value
        self.neighbors_NWES = None
        self.state = None
        self.init_state()

    #---------------------------------------------------
    
    def init_state(self, random_init_bool=False):

        if self.init_cell_state_value is None or random_init_bool:
            value = np.random.uniform(0, 1, 1)[0]
        else:
            value = self.init_cell_state_value
        
        self.state = [value] * self.len_state
    
    #---------------------------------------------------

    def set_state(self, vector, with_noise_bool, noise_std):

        if with_noise_bool:
            state = vector.copy()
            noise = np.random.normal(0, noise_std, self.len_state) 
            vector = [self.clip(state[i]+noise[i]) for i in range(self.len_state)] # check  

        self.state = vector

    #---------------------------------------------------

    def clip(self, value):
        return max(0, min(value, 1))

    #---------------------------------------------------

    def get_state(self):
        return self.state
    
    #---------------------------------------------------

    def get_chemical_species(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_phenotype(self):
        raise NotImplementedError("Subclasses must implement this method")


###########################################################################

class agent1Output(swarmAgent):
    def __init__(self, pos, init_cell_state_value):
        self.len_state = 1
        super().__init__(pos=pos, len_state=self.len_state, init_cell_state_value=init_cell_state_value)

    def init_state(self, random_init_bool=False):
        super().init_state(random_init_bool)

    def set_state(self, vector, with_noise_bool, noise_std):
        super().set_state(vector, with_noise_bool, noise_std)

    def get_chemical_species(self):
        state = super().get_state()
        return state[0]

    def get_phenotype(self):
        state = super().get_state()
        return state[0]

###########################################################################

class agent2Outputs(swarmAgent):
    def __init__(self, pos, init_cell_state_value):
        self.len_state = 2
        super().__init__(pos=pos, len_state=self.len_state, init_cell_state_value=init_cell_state_value)

    def init_state(self, random_init_bool=False):
        super().init_state(random_init_bool)

    def set_state(self, vector, with_noise_bool, noise_std):
        super().set_state(vector, with_noise_bool, noise_std)

    def get_chemical_species(self):
        state = super().get_state()
        return state[0]

    def get_phenotype(self):
        state = super().get_state()
        return state[1]


###########################################################################
# Evaluation functions
###########################################################################

def flag_automata(env_eval_function_params, analysis_dir, run, gen, best_fit, weights):
    time_steps = env_eval_function_params['time_steps']
    time_window_start = env_eval_function_params['time_window_start']
    time_window_end = env_eval_function_params['time_window_end']

    flags_distances = []
    in_t_window_zone_bools = []
    flags = []

    init_cell_state_value = env_eval_function_params['init_cell_state_value']
    if "learning_random_init_states_bool" in env_eval_function_params['learning_mode']:
        init_cell_state_value = None

    random_update_bool = False
    if "learning_random_update_states_bool" in env_eval_function_params['learning_mode']:
        random_update_bool = True

    with_noise_bool = False
    noise_std = None
    if "learning_with_noise_bool" in env_eval_function_params['learning_mode']:
        with_noise_bool = True
        noise_std = env_eval_function_params['noise_std']

    env = init_swarmGrid_env(grid_nb_rows=env_eval_function_params['grid_nb_rows'],
                             grid_nb_cols=env_eval_function_params['grid_nb_cols'],
                             flag_pattern=env_eval_function_params['flag_pattern'],
                             flag_target=env_eval_function_params['flag_target'],
                             init_cell_state_value=init_cell_state_value,
                             nn_controller=env_eval_function_params['controller'],
                             agent_controller_weights=weights)

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

        env.step(random_update_bool=random_update_bool,
                 with_noise_bool=with_noise_bool,
                 noise_std=noise_std)
        
    mean_tw_flags_distances = sum_flags_distances/(time_window_end - time_window_start)
    if mean_tw_flags_distances < best_fit:
        env.write_flag_data_learning(run=run, gen=gen, time_steps=time_steps, flags_distances=flags_distances, in_t_window_zone_bools=in_t_window_zone_bools, flags=flags, weights=weights, analysis_dir=analysis_dir)

    return (mean_tw_flags_distances,) # it is important to return a tuple


###########################################################################
# Environment swarmGrid
###########################################################################

class swarmGrid:
    def __init__(self, grid_nb_rows, grid_nb_cols, flag_pattern, flag_target, init_cell_state_value, nn_controller) -> None:
      
        self.grid_nb_rows = grid_nb_rows
        self.grid_nb_cols = grid_nb_cols
        self.grid_size = grid_nb_rows * grid_nb_cols

        self.agent_controller = nn_controller
        if nn_controller.n_neuronsPerOutputs == 1:
            self.agent_type = agent1Output
        elif nn_controller.n_neuronsPerOutputs == 2:
            self.agent_type = agent2Outputs

        self.default_missing_neighbor_state = 0.0
        self.grid_map_pos_agent = None
        self.init_grid(init_cell_state_value)

        if flag_target is not None:
            self.flag_target = flag_target
        else:
            self.flag_target = self.build_flag(flag_pattern) # flag_target is a dict pos:phenotype

    #---------------------------------------------------

    def set_agent_controller_weights(self, agent_controller_weights):
        self.agent_controller.setWeightsFromList(agent_controller_weights)

    #---------------------------------------------------

    def init_grid(self, init_cell_state_value):

        grid_map_pos_agent = {}
        for row in range(self.grid_nb_rows):
            for col in range(self.grid_nb_cols):
                agent = self.agent_type(pos=tuple((row, col)), init_cell_state_value=init_cell_state_value) # agent creation
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

        if flag_pattern == "2_stripes":
            vertical_threshold = np.floor(self.grid_nb_cols*2/5)

            for cell in self.grid_map_pos_agent.keys():
                if cell[1] < vertical_threshold:
                    flag_target[cell] = 0.0 # black
                else:
                    flag_target[cell] = 1.0 # white


        elif flag_pattern == "3_stripes":
            horizontal_threshold_upper = int(self.grid_nb_rows/3)
            horizontal_threshold_lower = int(self.grid_nb_rows/3*2)

            for cell in self.grid_map_pos_agent.keys():
                if cell[0] < horizontal_threshold_upper:
                    flag_target[cell] = 0.0 # black
                elif cell[0] >= horizontal_threshold_lower:
                    flag_target[cell] = 1.0 # white
                else:
                    flag_target[cell] = 0.5 # grey


        elif flag_pattern == "centered_circle" or flag_pattern == "not_centered_circle":
            center = (np.floor(self.grid_nb_rows/2), np.floor(self.grid_nb_cols/2))
            if flag_pattern == "not_centered_circle":
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


        elif flag_pattern == "half_circle":
            center = (np.floor(self.grid_nb_rows/2), np.floor(self.grid_nb_cols/2))
            radius = np.floor(min(self.grid_nb_rows/2, self.grid_nb_cols/2))

            for cell in self.grid_map_pos_agent.keys():
                if ((cell[0] - center[0])**2 + (cell[1] - center[1])**2 <= radius**2): # the cell is inside the circle area
                    if cell[1] <= center[1]:
                        flag_target[cell] = 0.0 # black
                    else:
                        flag_target[cell] = 1.0 # white
                else:  # the cell is outside the circle area
                    if cell[1] < center[1]:
                        flag_target[cell] = 1.0 # white
                    else:
                        flag_target[cell] = 0.0 # black


        elif flag_pattern == "centered_half_circle" or flag_pattern == "not_centered_half_circle":
            center = (np.floor(self.grid_nb_rows/2), np.floor(self.grid_nb_cols/2))
            if flag_pattern == "not_centered_half_circle":
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
                    else:  # the cell is outside the circle area
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
                    else:  # the cell is outside the circle area
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
    
    @staticmethod
    def build_map_cell_neighbors(grid_nb_rows, grid_nb_cols):

        map_cell_neighbors_NWES = {}
        for row in range(grid_nb_rows):
            for col in range(grid_nb_cols):
                l_tmp = []
                for pos in [(row-1, col), (row, col-1), (row, col+1), (row+1, col)]:
                    if swarmGrid.is_cell_pos_valid(pos, grid_nb_rows, grid_nb_cols):
                        l_tmp.append(tuple(pos))
                    else:
                        l_tmp.append(None)
                map_cell_neighbors_NWES[tuple((row, col))] = l_tmp

        return map_cell_neighbors_NWES

    #---------------------------------------------------

    def step(self, random_update_bool=False, with_noise_bool=False, noise_std=None):

        agents = self.get_agents() # ordered update

        if random_update_bool:
            np.random.shuffle(agents) # random update

        for agent in agents:
            state = self.compute_agent_state(agent=agent)
            agent.set_state(state, with_noise_bool, noise_std)

    #---------------------------------------------------

    def compute_agent_state(self, agent):
        
        if agent is None: # check
            return

        neighbors_states = []
        for neighbor in agent.neighbors_NWES: # est il à jour?

            if neighbor is not None: # si il a un id, il y a l'agent? tjr?
                neighbors_states.append(neighbor.get_chemical_species())
            else:
                neighbors_states.append(self.default_missing_neighbor_state)
        
        # print("compute_agent_state: agent.pos:", agent.pos, ", its neighbors:", agent.neighbors_NWES, "neighbors states:", neighbors_states)
        # print("prima di set_state - state:", agent.state, "agent.get_chemical_species():", agent.get_chemical_species(), "agent.get_phenotype():", agent.get_phenotype())
        state = self.agent_controller.predict(neighbors_states) # forwardPropagation, stableSigmoid on the last layer               
        return state
        # print("dopo di set_state - state:", agent.state, "agent.get_chemical_species():", agent.get_chemical_species(), "agent.get_phenotype():", agent.get_phenotype())

    #---------------------------------------------------

    def setup_ind_consistency(self, run, setup_name, nb_repetitions, time_steps, switch_step, switch_step_with_reset_env_bool, analysis_dir):
        
        for n in range(nb_repetitions):
            switch_step_with_random_update_bool = False
            self.init_agents_state()
            flags = []
            for step in range(time_steps):
                flag = self.get_flag_from_grid()
                flags.append(flag)

                if step == switch_step-1:
                
                    if setup_name == "setup_ind_consistency_random_init_states":
                        self.init_agents_state(random_init_bool=True)
                        continue

                    if setup_name == "setup_ind_consistency_random_update_states":
                        switch_step_with_random_update_bool = True
                        if switch_step_with_reset_env_bool:
                            self.init_agents_state()
                            continue

                self.step(random_update_bool=switch_step_with_random_update_bool)
            
            # Save flags for this run
            self.write_flag_data(setup_name=setup_name, run=run, n=n, time_steps=time_steps, flags=flags, deleted_agents=[], analysis_dir=analysis_dir)

    #---------------------------------------------------

    def setup_noise(self, run, setup_name, nb_repetitions, setup_noise_std_ticks, time_steps, switch_step, switch_step_with_reset_env_bool, switch_step_with_random_update_bool, analysis_dir):
        
        for n in range(nb_repetitions):
            for tick in setup_noise_std_ticks:
                setup_name_tick = setup_name+"_"+str(tick)
                with_noise_bool=False
                self.init_agents_state()
                flags = []

                for step in range(time_steps):
                    flag = self.get_flag_from_grid()
                    flags.append(flag)

                    if step == switch_step-1:
                        with_noise_bool = True
                        if switch_step_with_reset_env_bool:
                            self.init_agents_state()
                            continue

                    self.step(random_update_bool=switch_step_with_random_update_bool,
                              with_noise_bool=with_noise_bool,
                              noise_std=tick)
                
                # Save flags for this run
                self.write_flag_data(setup_name=setup_name_tick, run=run, n=n, time_steps=time_steps, flags=flags, deleted_agents=[], analysis_dir=analysis_dir)

    #---------------------------------------------------

    def setup_deletion(self, run, setup_name, nb_repetitions, deletion_ticks, time_steps, switch_step, switch_step_with_reset_env_bool, switch_step_with_random_update_bool, analysis_dir):
        
        for n in range(nb_repetitions):
            deleted_map_pos_agent = {}

            for tick in deletion_ticks:
                setup_name_tick = setup_name+"_"+str(tick)
                self.init_agents_state()
                flags = []
                deleted_agents = []
                agents_to_delete = []

                for step in range(time_steps):
                    flag = self.get_flag_from_grid()
                    flags.append(flag)
                    deleted_agents.append(agents_to_delete)

                    if step == switch_step-1:
                        agents = self.get_agents()
                        nb_deletions = tick - len(deleted_map_pos_agent)
                        agents_to_delete = list(deleted_map_pos_agent.values()) + random.sample(agents, nb_deletions) # old + new chosen agents
                        deleted_map_pos_agent = self.delete_agent(agents_to_delete=agents_to_delete)

                        if switch_step_with_reset_env_bool:
                            self.init_agents_state()
                            deleted_agents.append(agents_to_delete)
                            continue

                    self.step(random_update_bool=switch_step_with_random_update_bool)
                
                # Save flags for this run
                self.write_flag_data(setup_name=setup_name_tick, run=run, n=n, time_steps=time_steps, flags=flags, deleted_agents=deleted_agents, analysis_dir=analysis_dir)
            
                # Restore original grid
                self.restore_deleted_agents(deleted_map_pos_agent=deleted_map_pos_agent)

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
        for pos in deleted_map_pos_agent:
            self.grid_map_pos_agent[pos] = deleted_map_pos_agent[pos]

        # Update the neighbors list of each agent
        agents = self.get_agents()
        assert len(agents) == self.grid_size, f"\nrestore_deleted_agents {len(agents)} != {self.grid_size}"
        self.update_agent_neighbors()

    #---------------------------------------------------

    def get_flag_from_grid(self):
        flag = {}
        for pos in self.grid_map_pos_agent.keys():
            if self.grid_map_pos_agent[pos]:
                flag[pos] = self.grid_map_pos_agent[pos].get_phenotype()
            else:
                flag[pos] = self.default_missing_neighbor_state

        return flag
    
    #---------------------------------------------------

    def eval_flags_distance(self, flag): # quoi faire si les 2 flags ont taille differente?
        sum_states = 0.0
        
        for p, pos in enumerate(self.grid_map_pos_agent.keys()):
            sum_states += (self.flag_target[p] - flag[pos])**2 # check if ordre cells est coherent

        nb_agents = len([1 for a in self.grid_map_pos_agent.values() if a != None]) # check print(nb_agents)
        flags_distance = sum_states/nb_agents
        return flags_distance
    
    #---------------------------------------------------

    def convert_flag_to_list(self, flag):
        return list(flag.values())

    #---------------------------------------------------
    
    def write_flag_data_learning(self, run, gen, time_steps, flags_distances, in_t_window_zone_bools, flags, weights, analysis_dir):

        from learning_initializations import save_data_to_csv

        if run == 0 and gen == 0 and not (os.path.exists(analysis_dir['root']+"/data_all_runs/data_env_flag_target.csv")): # os.path.exists test is not enough with parallelization
            save_data_to_csv(analysis_dir['root']+"/data_all_runs/data_env_flag_target.csv", [[0, 0, 0, 0,  str(self.flag_target).strip(), 0]], header = ["Generation", "Step", "Flags_distance", "Time_window_zone", "Flag", "Individual"])
        
        if not (os.path.exists(analysis_dir['data']+"/data_env_flag/data_env_flag_run_"+str(run)+"_gen_"+str(gen)+".csv")):
            os.makedirs(analysis_dir['data']+"/data_env_flag/", exist_ok=True)
            save_data_to_csv(analysis_dir['data']+"/data_env_flag/data_env_flag_run_"+str(run)+"_gen_"+str(gen)+".csv", [], header = ["Generation", "Step", "Flags_distance", "Time_window_zone", "Flag", "Individual"])    

        data_env_flag = []
        for step in range(time_steps):
            data_env_flag.append([str(gen), str(step), str(flags_distances[step]).strip(), str(in_t_window_zone_bools[step]).strip(), str(flags[step]).strip(), str(weights).strip()])

        save_data_to_csv(analysis_dir['data']+"/data_env_flag/data_env_flag_run_"+str(run)+"_gen_"+str(gen)+".csv", data_env_flag)

    #---------------------------------------------------

    def write_flag_data(self, setup_name, run, n, time_steps, flags, deleted_agents, analysis_dir):

        from learning_initializations import save_data_to_csv

        dir_name = analysis_dir['data']+"/"+str(setup_name)
        file_name = "/data_"+setup_name+"_flag_run_"+str(run)+"_n_"+str(n)+".csv"
        if os.path.exists(dir_name+file_name):
            return
        if not (os.path.exists(dir_name)):
            os.makedirs(dir_name, exist_ok=True)

        data_env_flag = []
        for step in range(time_steps):
            flags_distance = self.eval_flags_distance(flags[step])

            if deleted_agents:
                deleted_agents_positions = [agent.pos for agent in deleted_agents[step]]
                data_env_flag.append([str(run), setup_name, str(n), str(step), str(flags_distance).strip(), str(self.convert_flag_to_list(flags[step])).strip(), str(deleted_agents_positions).strip()])
                header = ["Run", "Setup", "N", "Step", "Flags_distance", "Flag", "Deleted_agents_positions"]
            else:
                data_env_flag.append([str(run), setup_name, str(n), str(step), str(flags_distance).strip(), str(self.convert_flag_to_list(flags[step])).strip()])
                header = ["Run", "Setup", "N", "Step", "Flags_distance", "Flag"]

        save_data_to_csv(dir_name+file_name, data_env_flag, header=header)

    #---------------------------------------------------

    @staticmethod
    def plot_flag(grid_nb_rows, grid_nb_cols, setup_name, run, nb_ind, gen, n, step, flag, fitness, deleted_pos=[], analysis_dir_plots=None):
        
        fig, ax = plt.subplots()

        grid_pos = []
        for row in range(grid_nb_rows):
            for col in range(grid_nb_cols):
                grid_pos.append(tuple((row, col)))

        if deleted_pos:
            x_deleted = [pos[1] for pos in deleted_pos]
            y_deleted = [-pos[0] for pos in deleted_pos]

        if grid_nb_rows <= 10 and grid_nb_cols <= 10:
            for row, col in [pos for pos in grid_pos if pos not in deleted_pos]: # works?
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

                if pos in deleted_pos: # works?
                    circle = patches.Circle((pos[1], -pos[0]), circle_radius, edgecolor='tab:red', facecolor='white', linestyle='--', linewidth=2.0, zorder=2)    
                
                ax.add_patch(circle)

                if grid_nb_rows < 6 and grid_nb_cols < 6:
                    ax.text(pos[1], -pos[0], "(" + str(pos[0]) +"," + str(pos[1]) + ")\n"+ str(round(grey_value,2)), color='black', va='center', ha='center')
                            
                plt.axis('off')

        if grid_nb_rows > 10 or grid_nb_cols > 10:
            colors = [(0.0, 0.0, 0.0), (0.7, 0.9, 1.0)]  # Black to light blue
            cmap = ListedColormap(np.linspace(colors[0], colors[1], 100))
            plt.scatter(x, y, c=grey_values, cmap=cmap) # cmap='grey'

            if deleted_pos:
                plt.scatter(x_deleted, y_deleted, c='tab:red') # deleted agents
                                      
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

        ax.set_aspect('equal')
        plt.xlim(-0.5, grid_nb_cols-0.5)
        plt.ylim(-grid_nb_rows+0.5, 0.5)
        
        if setup_name:
            plt.title(f"Flag states - {setup_name}\nRun {run}, best individual {nb_ind}, step {step}.\nFitness (distance to flag target) = {fitness}", fontsize=12)
            dir_name = analysis_dir_plots+"/"+setup_name+"/flag"
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            plt.savefig(f"{dir_name}/{setup_name}_flag_run_{run}_best_ind_{nb_ind}_n_{n}_step_{step}.png")
        else:
            if nb_ind is not None:
                plt.title(f"Flag states - learning.\nRun {run}, gen {gen}, individual {nb_ind}, step {step}.\nFitness (distance to flag target) = {fitness}", fontsize=12)       
                file_name = "run_"+str(run)+"_gen_"+str(gen)+"_individual_"+str(nb_ind)
                dir_name = analysis_dir_plots+"/"+file_name+"/flag"
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name, exist_ok=True)
                plt.savefig(f"{dir_name}/plot_env_flag_{file_name}_step_{step}.png")
            else:
                plt.suptitle(f"Flag target {grid_nb_rows}x{grid_nb_cols}", fontsize=12)
                plt.savefig(f"{analysis_dir_plots}/plot_env_flag_target.png")

        plt.clf()
        plt.close()

    #---------------------------------------------------

    @staticmethod
    def plot_flag_fitnesses_from_file(data_flag_file, setup_name, time_window_start, time_window_length, run, nb_ind, ind, n, gen, switch_step, analysis_dir_plots):

        df = pd.read_csv(data_flag_file)

        dataset_gen = df.loc[(df.Generation==gen)]
        dataset = dataset_gen.loc[(dataset_gen.Individual==str(ind))]

        x = dataset['Step'].tolist()
        y = dataset['Flags_distance'].tolist()

        fig, ax = plt.subplots()

        plt.plot(x, y)

        if setup_name is not None:
            plt.axvline(x=switch_step, color='r', linestyle='--')
        else:
            rectangle = patches.Rectangle((time_window_start, 0), time_window_length, 1, linewidth=1, edgecolor=None, facecolor='lemonchiffon', alpha=0.5)
            ax.add_patch(rectangle)

        plt.ylim(-0.1, 1) # 0 and 1 are respectively min and max values of flag distance
        plt.xlabel("Steps", fontsize=12)
        plt.ylabel("Fitness (distance to flag target)", fontsize=12)

        if setup_name:
            plt.title(f"Fitness related to the flag evolution over steps\n{setup_name}, {n} repetitions", fontsize=12)
            dir_name = analysis_dir_plots+"/"+setup_name
            if not (os.path.exists(dir_name)):
                os.makedirs(dir_name, exist_ok=True)
            plt.savefig(f"{dir_name}/{setup_name}_flag_fitnesses_run_{run}_n_{n}.png")
        else:
            plt.title(f"Fitness related to the flag evolution over steps. Gen {gen}, individual {nb_ind}\nTime window zone from step {time_window_start} to step {time_window_start+time_window_length-1} (included).", fontsize=12)
            plt.savefig(f"{analysis_dir_plots}/run_{run}_gen_{gen}_individual_{nb_ind}/flag_fitnesses_run_{run}_gen_{gen}_individual_{nb_ind}.png")

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
            plt.plot(x, y)
            plt.axvline(x=switch_step, color='r', linestyle='--')

        plt.ylim(-0.1, 1) # 0 and 1 are respectively min and max values of flag distance
        plt.xlabel("Steps", fontsize=12)
        plt.ylabel("Fitness (distance to flag target)", fontsize=12)
        plt.title(f"Fitness related to the flag evolution over steps. Run {run}\n{setup_name}, {len(data_flag_files)} repetitions", fontsize=12)
        
        dir_name = analysis_dir_plots+"/"+setup_name
        if not (os.path.exists(dir_name)):
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(f"{dir_name}/{setup_name}_flag_fitnesses_run_{run}.png")

        plt.clf()
        plt.close()