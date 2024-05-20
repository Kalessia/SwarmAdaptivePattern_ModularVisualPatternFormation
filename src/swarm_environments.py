import os

import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

from swarm_initializations import save_data_to_csv



###########################################################################
# Global variables
###########################################################################

global env
env = None


def init_swarmGrid_env(grid_nb_rows, grid_nb_cols, init_cell_state_value, nn_controller, flag_target, agent_controller_weights):
    global env
    if env is None:
        env = swarmGrid(grid_nb_rows=grid_nb_rows,
                        grid_nb_cols=grid_nb_cols,
                        init_cell_state_value=init_cell_state_value,
                        nn_controller=nn_controller,
                        flag_target=flag_target)
        
    env.set_agent_controller_weights(agent_controller_weights=agent_controller_weights)
    # quelque chose pour reset
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

        self.state = self.init_state()

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
# Environment swarm grid
###########################################################################

class swarmGrid:
    def __init__(self, grid_nb_rows, grid_nb_cols, init_cell_state_value, nn_controller, flag_target) -> None:

        self.grid_nb_rows = grid_nb_rows
        self.grid_nb_cols = grid_nb_cols
        self.grid_size = grid_nb_rows * grid_nb_cols
        self.grid_map_pos_agent = None

        self.agent_controller = nn_controller
        if nn_controller.n_neuronsPerOutputs == 1:
            self.agent_type = agent1Output
        elif nn_controller.n_neuronsPerOutputs == 2:
            self.agent_type = agent2Outputs

        self.default_missing_neighbor_state = 0.0
        self.flag_target = flag_target
        self.init_grid(init_cell_state_value)

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
        self.update_grid() # check

    #---------------------------------------------------

    def get_agents(self):
        return [a for a in self.grid_map_pos_agent.values() if a != None]

    #---------------------------------------------------

    def init_agents_state(self, random_init_bool=False):

        agents = self.get_agents()
        for agent in agents:
            agent.init_state(random_init_bool)

    #---------------------------------------------------

    def step(self, switch_step_with_random_update_bool=False, with_noise_bool=False, noise_std=None):

        agents = self.get_agents() # ordered update

        if switch_step_with_random_update_bool:
            np.random.shuffle(agents) # random update

        for agent in agents:
            state = self.compute_agent_state(agent=agent)
            agent.set_state(state, with_noise_bool, noise_std)

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
                        switch_step_with_random_update_bool=True
                        if switch_step_with_reset_env_bool:
                            self.init_agents_state()
                            continue

                self.step(switch_step_with_random_update_bool=switch_step_with_random_update_bool)
            
            # Save flags for this run
            self.write_flag_data(setup_name=setup_name, run=run, n=n, time_steps=time_steps, flags=flags, analysis_dir=analysis_dir)

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

                    self.step(switch_step_with_random_update_bool=switch_step_with_random_update_bool,
                              with_noise_bool=with_noise_bool,
                              noise_std=tick)
                
                # Save flags for this run
                self.write_flag_data(setup_name=setup_name_tick, run=run, n=n, time_steps=time_steps, flags=flags, analysis_dir=analysis_dir)

    #---------------------------------------------------

    def setup_deletion(self, run, setup_name, nb_repetitions, deletion_ticks, time_steps, switch_step, switch_step_with_reset_env_bool, switch_step_with_random_update_bool, analysis_dir):
        
        for n in range(nb_repetitions):
            deleted_map_pos_agent = {}

            for tick in deletion_ticks:
                setup_name_tick = setup_name+"_"+str(tick)
                self.init_agents_state()
                flags = []

                for step in range(time_steps):
                    flag = self.get_flag_from_grid()
                    flags.append(flag)

                    if step == switch_step-1:
                        agents = self.get_agents()
                        nb_deletions = tick - len(deleted_map_pos_agent)
                        agents_to_delete = list(deleted_map_pos_agent.values()) + random.sample(agents, nb_deletions) # old + new chosen agents
                        deleted_map_pos_agent = self.delete_agent(agents_to_delete=agents_to_delete)

                        if switch_step_with_reset_env_bool:
                            self.init_agents_state()
                            continue

                    self.step(switch_step_with_random_update_bool=switch_step_with_random_update_bool)
                
                # Save flags for this run
                self.write_flag_data(setup_name=setup_name_tick, run=run, n=n, time_steps=time_steps, flags=flags, analysis_dir=analysis_dir)
            
                # Restore original grid
                self.restore_deleted_agents(deleted_map_pos_agent=deleted_map_pos_agent)

    #---------------------------------------------------

    def delete_agent(self, agents_to_delete=None):
        
        deleted_map_pos_agent = {}
        for agent in agents_to_delete:
            deleted_map_pos_agent[agent.pos] = agent # save agent
            self.grid_map_pos_agent[agent.pos] = None # eliminate agent from grid

        # Update the neighbors list of each agent
        agents = self.get_agents()
        for agent in agents:
            l_tmp = self.get_neighbors(agent=agent)
            agent.neighbors_NWES = l_tmp

        return deleted_map_pos_agent

    #---------------------------------------------------

    def restore_deleted_agents(self, deleted_map_pos_agent=None):
        for pos in deleted_map_pos_agent:
            self.grid_map_pos_agent[pos] = deleted_map_pos_agent[pos]

        # Update the neighbors list of each agent
        agents = self.get_agents()
        assert len(agents) == self.grid_size, f"\nrestore_deleted_agents {len(agents)} != {self.grid_size}"

        for agent in agents:
            l_tmp = self.get_neighbors(agent=agent)
            agent.neighbors_NWES = l_tmp

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

    def compute_agent_state(self, agent):
        
        if agent is None: # check
            return

        neighbors_states = []
        for neighbor in agent.neighbors_NWES:

            if neighbor is not None: # si il a un id, il y a l'agent? tjr?
                neighbors_states.append(neighbor.get_chemical_species())
            else:
                neighbors_states.append(self.default_missing_neighbor_state)
        
        # print("compute_agent_state: agent.pos:", agent.pos, ", its neighbors:", agent.neighbors_NWES, "neighbors states:", neighbors_states)
        # print("prima di set_state - state:", agent.state, "agent.get_chemical_species():", agent.get_chemical_species(), "agent.get_phenotype():", agent.get_phenotype())
        state = self.agent_controller.predict(neighbors_states) # forwardPropagation, stableSigmoid on the last layer               
        return state
        
        # print("dopo di set_state - state:", agent.state, "agent.get_chemical_species():", agent.get_chemical_species(), "agent.get_phenotype():", agent.get_phenotype())

    # #---------------------------------------------------    
    
    def update_grid(self):

        agents = self.grid_map_pos_agent.values() # exclure none?
        # print([agent.id for agent in agents])
        # agents_ids = list(self.grid_map_id_pos.keys())

        # Collect the neighbors of each agent (if we call 'refresh', they have likely changed)
        for agent in agents:
            l_tmp = self.get_neighbors(agent=agent)
            agent.neighbors_NWES = l_tmp
        # print("2", self.grid_map_pos_agent)

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

        nb_agents = len([1 for a in self.grid_map_pos_agent.values() if a != None])
        flags_distance = sum_states/nb_agents
        return flags_distance
    
    #---------------------------------------------------

    def convert_flag_to_list(self, flag):
        return list(flag.values())

    #---------------------------------------------------

    def write_flag_data(self, setup_name, run, n, time_steps, flags, analysis_dir):

        dir_name = analysis_dir['data']+"/"+str(setup_name)
        file_name = "/data_"+setup_name+"_flag_run_"+str(run)+"_n_"+str(n)+".csv"
        if not (os.path.exists(dir_name)):
            os.makedirs(dir_name, exist_ok=True)

        data_env_flag = []
        for step in range(time_steps):
            flags_distance = self.eval_flags_distance(flags[step])
            data_env_flag.append([str(run), setup_name, str(n), str(step), str(flags_distance).strip(), str(self.convert_flag_to_list(flags[step])).strip()])

        save_data_to_csv(dir_name+file_name, data_env_flag, header = ["Run", "Setup", "N", "Step", "Flags_distance", "Flag"])

    #---------------------------------------------------

    @staticmethod
    def plot_flag(grid_nb_rows, grid_nb_cols, setup_name, run, best_ind_run, n, step, flag, fitness, deleted_pos=[], analysis_dir_plots=None):

        fig, ax = plt.subplots()

        grid_pos = []
        for row in range(grid_nb_rows):
            for col in range(grid_nb_cols):
                grid_pos.append(tuple((row, col)))

        if deleted_pos: # works?
            x_deleted = [pos[0] for pos in deleted_pos]
            y_deleted = [pos[1] for pos in deleted_pos]

        if grid_nb_rows <= 10 and grid_nb_cols <= 10:
            for row, col in [pos for pos in grid_pos if pos not in deleted_pos]: # works?
                for neighbor_pos in [(row-1, col), (row, col-1), (row, col+1), (row+1, col)]:
                    if swarmGrid.is_pos_valid(grid_nb_rows, grid_nb_cols, pos):
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

        plt.title(f"{setup_name} Flag evolution states. Run {run}, setup name {setup_name}, best individual {best_ind_run}, step {step}.\nFitness (distance to flag target) = {fitness}", pad=20, fontsize=9)
        
        dir_name = analysis_dir_plots+"/"+setup_name+"/flag"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(dir_name+"/"+setup_name+"_flag_run_"+str(run)+"_best_ind_"+str(best_ind_run)+"_n_"+str(n)+"_step_"+str(step)+".png")
        
        plt.clf()
        plt.close()

    #---------------------------------------------------

    @staticmethod
    def plot_flag_fitnesses_from_file(data_flag_file, setup_name, run, n, switch_step, analysis_dir_plots):

        dataset = pd.read_csv(data_flag_file)

        x = dataset['Step'].tolist()
        y = dataset['Flags_distance'].tolist()

        plt.plot(x, y)
        plt.axvline(x=switch_step, color='r', linestyle='--')

        plt.ylim(0, 1) # 0 and 1 are respectively min and max values of flag distance
        plt.xlabel("Steps", fontsize=12)
        plt.ylabel("Fitness (distance to flag target)", fontsize=12)
        plt.suptitle(f"Fitness related to the flag evolution over steps run="+str(run), fontsize=14)
        plt.title(setup_name)
        # plt.title(f"Generation {gen}, individual {nb_ind}, {env_eval_function_params['time_steps']} steps. Time window zone from step {time_window_start} to step {time_window_end} (included).", fontsize=9)
        
        dir_name = analysis_dir_plots+"/"+setup_name
        if not (os.path.exists(dir_name)):
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(dir_name+"/"+setup_name+"_flag_fitnesses_run_"+str(run)+"_n_"+str(n)+".png")
        
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

        plt.ylim(0, 1) # 0 and 1 are respectively min and max values of flag distance
        plt.xlabel("Steps", fontsize=12)
        plt.ylabel("Fitness (distance to flag target)", fontsize=12)
        plt.suptitle(f"Fitness related to the flag evolution over steps run={run}", fontsize=14)
        plt.title(f"{setup_name} Nb repetitions {len(data_flag_files)}", fontsize=14)
        # plt.title(f"Generation {gen}, individual {nb_ind}, {env_eval_function_params['time_steps']} steps. Time window zone from step {time_window_start} to step {time_window_end} (included).", fontsize=9)
        
        dir_name = analysis_dir_plots+"/"+setup_name
        if not (os.path.exists(dir_name)):
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(f"{dir_name}/{setup_name}_flag_fitnesses_run_{run}.png")

        plt.clf()
        plt.close()

    #---------------------------------------------------