import os

import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

from nn import NeuralNetwork

from simulation_analysis import save_data_to_csv



global count_id
count_id = -1

###########################################################################
# Environment swarm grid
###########################################################################

class swarmAgent:
    def __init__(self, pos, init_cell_state_value=None) -> None:

        global count_id
        count_id += 1
        self.id = count_id
        self.pos = pos

        #state
        # self.state = None
        self.init_cell_state_value = init_cell_state_value
        self.chemical_species = None
        self.phenotype = None
        self.init_state()

        self.neighbors_ids_NWES = None

    #---------------------------------------------------
    
    def init_state(self):

        if self.init_cell_state_value is None:
            init_cell_state_value = np.random.uniform(0, 1, 1)[0]

        self.chemical_species = init_cell_state_value
        self.phenotype = init_cell_state_value


###########################################################################
# Environment swarm grid
###########################################################################

class swarmGrid:
    def __init__(self, grid_nb_rows, grid_nb_cols, init_cell_state_value, nb_neuronsPerInputs, nb_hiddenLayers, nb_neuronsPerHidden, nb_neuronsPerOutputs, agent_controller_weights, flag_target) -> None:

        self.grid_nb_rows = grid_nb_rows
        self.grid_nb_cols = grid_nb_cols
        self.grid_size = grid_nb_rows * grid_nb_cols
        self.grid_map_pos_agent = None
        self.grid_map_id_pos = None

        self.agent_controller = NeuralNetwork(nb_neuronsPerInputs=nb_neuronsPerInputs, nb_hiddenLayers=nb_hiddenLayers, nb_neuronsPerHidden=nb_neuronsPerHidden, nb_neuronsPerOutputs=nb_neuronsPerOutputs)
        self.agent_controller.setWeightsFromList(agent_controller_weights)
        
        self.default_missing_neighbor_state = 0.0

        self.flag_target = flag_target
        self.init_grid()

    #---------------------------------------------------

    def init_grid(self): # agent creation

        grid_map_pos_agent = {}
        grid_map_id_pos = {}
        for row in range(self.grid_nb_rows):
            for col in range(self.grid_nb_cols):
                agent = swarmAgent(pos=tuple((row, col)))
                grid_map_pos_agent[tuple((row, col))] = agent
                grid_map_id_pos[agent.id] = tuple((row, col))
        
        # print("grid_map_pos_agent", grid_map_pos_agent)
        # print("grid_map_id_pos", grid_map_id_pos)

        self.grid_map_pos_agent = grid_map_pos_agent
        self.grid_map_id_pos = grid_map_id_pos
        self.refresh_grid()

    #---------------------------------------------------

    def init_agents_state(self):
        # print(self.grid_map_pos_agent)
        agents = [a for a in self.grid_map_pos_agent.values() if a != None] #add assert?
        # agents = self.grid_map_pos_agent.values() # exclure none?
        for agent in agents:
            agent.init_state()

    #---------------------------------------------------

    def step(self):

        agents_ids = list(self.grid_map_id_pos.keys()) # ordered update
        np.random.shuffle(agents_ids) # random update

        for agent_id in agents_ids:
            self.compute_agent_state(agent=self.grid_map_pos_agent[self.grid_map_id_pos[agent_id]])










        # print("grid_map_pos_agent", self.grid_map_pos_agent)
        # print("grid_map_id_pos", self.grid_map_id_pos)

            # if self.automata_mode == 2:

            #     chemical_species, phenotype = self.compute_cell_chemical_species_and_phenotype(cell)
            #     self.flag[cell] = phenotype
            #     self.chemical_species[cell] = chemical_species

            # else:
            #     # print("step: The value in ", cell, "was", self.flag[cell])
            #     self.flag[cell] = self.compute_cell_state(cell)
            #     # print("step: After compute_cell_state, the value in", cell, "is", self.flag[cell], "\n")

      #---------------------------------------------------

    def setup_test_ind(self, run, n, time_steps, switch_step, best_ind_run, best_ind, simulation_params):
        setup_name = "setup_test_ind"
        
        self.init_agents_state()

        flags = []

        for _ in range(time_steps):

            flag = self.get_flag_from_grid()
            flags.append(flag)

            self.step()

        # Save flags for this run
        self.write_flag_data(setup_name, run, best_ind_run, best_ind, n, time_steps, flags, analysis_dir=simulation_params['analysis_dir'])
        data_flag_file = simulation_params['analysis_dir']['data']+"/best_ind_"+str(best_ind_run)+"/"+setup_name+"/data_"+setup_name+"_flag_run_"+str(run)+"_n_"+str(n)+".csv"
        self.plot_flag_fitnesses_from_file(data_flag_file=data_flag_file, setup_name=setup_name, run=run, best_ind_run=best_ind_run, n=n, switch_step=switch_step, analysis_dir_plots=simulation_params['analysis_dir']['plots'])

    #---------------------------------------------------

    def setup_noise1(self, run, n, tick, time_steps, switch_step, best_ind_run, best_ind, simulation_params):
        setup_name = "setup_noise1_"+str(tick)
        
        self.init_agents_state()
        agents_ids = list(self.grid_map_id_pos.keys()) # ordered update

        flags = []

        for step in range(time_steps):

            flag = self.get_flag_from_grid()
            flags.append(flag)
            if step in [0, switch_step, time_steps-1]:
                self.plot_flag(setup_name, run, best_ind_run, n, step, flag, analysis_dir_plots=simulation_params['analysis_dir']['plots'])
            
            if step >= switch_step:

                np.random.shuffle(agents_ids) # random update
                for agent_id in agents_ids:
                    self.compute_agent_state_with_noise_1(agent=self.grid_map_pos_agent[self.grid_map_id_pos[agent_id]], noise_std=tick)

            else:
                self.step()

        # Save flags for this run
        self.write_flag_data(setup_name, run, best_ind_run, best_ind, n, time_steps, flags, analysis_dir=simulation_params['analysis_dir'])
        data_flag_file = simulation_params['analysis_dir']['data']+"/best_ind_"+str(best_ind_run)+"/"+setup_name+"/data_"+setup_name+"_flag_run_"+str(run)+"_n_"+str(n)+".csv"
        self.plot_flag_fitnesses_from_file(data_flag_file=data_flag_file, setup_name=setup_name, run=run, best_ind_run=best_ind_run, n=n, switch_step=switch_step, analysis_dir_plots=simulation_params['analysis_dir']['plots'])

    #---------------------------------------------------

    def setup_noise2(self, run, n, tick, time_steps, switch_step, best_ind_run, best_ind, simulation_params):
        setup_name = "setup_noise2_"+str(tick)
        
        self.init_agents_state()
        agents_ids = list(self.grid_map_id_pos.keys()) # ordered update

        flags = []

        for step in range(time_steps):

            flag = self.get_flag_from_grid()
            flags.append(flag)
            if step in [0, switch_step, time_steps-1]:
                self.plot_flag(setup_name, run, best_ind_run, n, step, flag, analysis_dir_plots=simulation_params['analysis_dir']['plots'])
            
            if step >= switch_step:

                if True: #Kale parametrable bool
                    np.random.shuffle(agents_ids) # random update

                for agent_id in agents_ids:
                    self.compute_agent_state_with_noise_2(agent=self.grid_map_pos_agent[self.grid_map_id_pos[agent_id]], noise_std=tick)

            else:
                self.step()

        # Save flags for this run
        self.write_flag_data(setup_name, run, best_ind_run, best_ind, n, time_steps, flags, analysis_dir=simulation_params['analysis_dir'])
        data_flag_file = simulation_params['analysis_dir']['data']+"/best_ind_"+str(best_ind_run)+"/"+setup_name+"/data_"+setup_name+"_flag_run_"+str(run)+"_n_"+str(n)+".csv"
        self.plot_flag_fitnesses_from_file(data_flag_file=data_flag_file, setup_name=setup_name, run=run, best_ind_run=best_ind_run, n=n, switch_step=switch_step, analysis_dir_plots=simulation_params['analysis_dir']['plots'])

    #---------------------------------------------------

    def setup_deletion(self, run, n, tick, time_steps, switch_step, best_ind_run, best_ind, deleted_map_pos_agent, simulation_params):

        setup_name = "setup_deletion_"+str(tick)
        self.init_agents_state()

        flags = []
        for step in range(time_steps):

            flag = self.get_flag_from_grid()
            flags.append(flag)
            if step in [0, switch_step, time_steps-1]:
                self.plot_flag(setup_name, run, best_ind_run, n, step, flag, deleted_map_pos_agent, analysis_dir_plots=simulation_params['analysis_dir']['plots'])
            
            # elimination progressive kale
            if step == switch_step:
                nb_deletions = tick - len(deleted_map_pos_agent)
                deleted_map_pos_agent = self.delete_agent(nb_deletions=nb_deletions, deleted_map_pos_agent=deleted_map_pos_agent)

            self.step()

        # Save flags for this run
        self.write_flag_data(setup_name, run, best_ind_run, best_ind, n, time_steps, flags, analysis_dir=simulation_params['analysis_dir'])
        data_flag_file = simulation_params['analysis_dir']['data']+"/best_ind_"+str(best_ind_run)+"/"+setup_name+"/data_"+setup_name+"_flag_run_"+str(run)+"_n_"+str(n)+".csv"
        self.plot_flag_fitnesses_from_file(data_flag_file=data_flag_file, setup_name=setup_name, run=run, best_ind_run=best_ind_run, n=n, switch_step=switch_step, analysis_dir_plots=simulation_params['analysis_dir']['plots'])
        
        return deleted_map_pos_agent
    
    #---------------------------------------------------

    def is_pos_valid(self, pos):
        return pos[0]>=0 and pos[1]>=0 and pos[0]<self.grid_nb_rows and pos[1]<self.grid_nb_cols

    #---------------------------------------------------

    def get_neighbors_ids(self, agent):

        l_tmp = []
        row, col = agent.pos
        for pos in [(row-1, col), (row, col-1), (row, col+1), (row+1, col)]:
            if self.is_pos_valid(pos) and self.grid_map_pos_agent[tuple(pos)] != None:
                l_tmp.append(self.grid_map_pos_agent[tuple(pos)].id)
            else:
                l_tmp.append(None)
        
        return l_tmp

    #---------------------------------------------------

    def compute_agent_state_with_noise_1(self, agent, noise_std): # dividere in get neighbors states? + predict?
        
        # print("id", agent.id, "neighbors_ids", agent.neighbors_ids_NWES)

        # Gaussian noise distribution
        noise_mean = 0.5
        
        neighbors_states = []
        for i, neighbor_id in enumerate(agent.neighbors_ids_NWES):
            noise = np.random.normal(noise_mean, noise_std, len(agent.neighbors_ids_NWES))
            if neighbor_id: # si il a un id, il y a l'agent? tjr?
                # neighbors_states.append(self.flag[neighbor])
                neighbor = self.grid_map_pos_agent[self.grid_map_id_pos[neighbor_id]]
                neighbors_states.append(max(0, neighbor.chemical_species + noise[i])) # check: in che parte di codice di env usiamo chemical species
                #print originals and noisy to check, check if valeur bruité>0 toujours
            else:
                neighbors_states.append(self.default_missing_neighbor_state)
        
        # print("compute_cell_state: The neighbors of ", cell, "are", self.map_cell_neighbors_NWES[cell])
        # print("compute_cell_state: Their neighbors states are", neighbors_states)
        if True:
            chemical_species, phenotype = self.agent_controller.predict(neighbors_states) # forwardPropagation, stableSigmoid on the last layer
            agent.chemical_species = chemical_species
            agent.phenotype = phenotype
        else:
            agent_state = self.agent_controller.predict(neighbors_states)[0] # forwardPropagation, stableSigmoid on the last layer
            
        # print("compute_cell_state: The predicted state for", cell, "is", cell_state)

    #---------------------------------------------------

    def compute_agent_state_with_noise_2(self, agent, noise_std): # dividere in get neighbors states? + predict?
        
        # print("id", agent.id, "neighbors_ids", agent.neighbors_ids_NWES)

       # Gaussian noise distribution
        noise_mean = 0.5

        neighbors_states = []
        for neighbor_id in agent.neighbors_ids_NWES:
            if neighbor_id: # si il a un id, il y a l'agent? tjr?
                # neighbors_states.append(self.flag[neighbor])
                neighbor = self.grid_map_pos_agent[self.grid_map_id_pos[neighbor_id]]
                neighbors_states.append(neighbor.chemical_species) # check: in che parte di codice di env usiamo chemical species
            else:
                neighbors_states.append(self.default_missing_neighbor_state)
        
        # print("compute_cell_state: The neighbors of ", cell, "are", self.map_cell_neighbors_NWES[cell])
        # print("compute_cell_state: Their neighbors states are", neighbors_states)
        if True:
            noise = np.random.normal(noise_mean, noise_std, 2)
            chemical_species, phenotype = self.agent_controller.predict(neighbors_states) # forwardPropagation, stableSigmoid on the last layer
            agent.chemical_species = chemical_species + noise[0]
            agent.phenotype = phenotype + noise[1]
        else:
            agent_state = self.agent_controller.predict(neighbors_states)[0] # forwardPropagation, stableSigmoid on the last layer
            
        # print("compute_cell_state: The predicted state for", cell, "is", cell_state)

    #---------------------------------------------------    

    def compute_agent_state(self, agent): # dividere in get neighbors states? + predict?
        
        # print("id", agent.id, "neighbors_ids", agent.neighbors_ids_NWES)

        if agent == None: # check
            return

        neighbors_states = []

        for neighbor_id in agent.neighbors_ids_NWES:
            if neighbor_id: # si il a un id, il y a l'agent? tjr?
                # neighbors_states.append(self.flag[neighbor])
                neighbor = self.grid_map_pos_agent[self.grid_map_id_pos[neighbor_id]]
                neighbors_states.append(neighbor.chemical_species) # check: in che parte di codice di env usiamo chemical species
            else:
                neighbors_states.append(self.default_missing_neighbor_state)
        
        # print("compute_cell_state: The neighbors of ", cell, "are", self.map_cell_neighbors_NWES[cell])
        # print("compute_cell_state: Their neighbors states are", neighbors_states)
        if True:
            chemical_species, phenotype = self.agent_controller.predict(neighbors_states) # forwardPropagation, stableSigmoid on the last layer
            agent.chemical_species = chemical_species
            agent.phenotype = phenotype
        else:
            agent_state = self.agent_controller.predict(neighbors_states)[0] # forwardPropagation, stableSigmoid on the last layer
            
        # print("compute_cell_state: The predicted state for", cell, "is", cell_state)

    #---------------------------------------------------

    def delete_agent(self, nb_deletions=None, deleted_map_pos_agent=None): # a faire kale
        
        agents = [a for a in self.grid_map_pos_agent.values() if a != None]
        agents_to_delete = random.sample(agents, nb_deletions)

        for agent in agents_to_delete:
            deleted_map_pos_agent[agent.pos] = agent #save
            self.grid_map_pos_agent[agent.pos] = None #elimination

        # Collect the neighbors of each agent (if we call 'refresh', they have likely changed)
        for agent in agents:
            l_tmp = self.get_neighbors_ids(agent=agent)
            agent.neighbors_ids_NWES = l_tmp

        # check all
        return deleted_map_pos_agent

    #---------------------------------------------------

    def restore_deleted_agents(self, deleted_map_pos_agent=None):
        for pos in deleted_map_pos_agent:
            self.grid_map_pos_agent[pos] = deleted_map_pos_agent[pos]

        # Collect the neighbors of each agent (if we call 'refresh', they have likely changed)
        agents = [a for a in self.grid_map_pos_agent.values() if a != None]
        assert len(agents) == self.grid_size, f"{len(agents)} != {self.grid_size}"

        for agent in agents:  
            l_tmp = self.get_neighbors_ids(agent=agent)
            agent.neighbors_ids_NWES = l_tmp

    #---------------------------------------------------
    
    def refresh_grid(self):

        agents = self.grid_map_pos_agent.values() # exclure none?
        # print([agent.id for agent in agents])
        # agents_ids = list(self.grid_map_id_pos.keys())

        # print("1", self.grid_map_pos_agent)
        for agent in agents:
            # print("working on ", agent, agent.id, agent.pos)
            # print("self.grid_map_id_pos[agent.id] per", agent.id, "era", self.grid_map_id_pos[agent.id], "ora é", agent.pos)
            self.grid_map_id_pos[agent.id] = agent.pos

        # Collect the neighbors of each agent (if we call 'refresh', they have likely changed)
        for agent in agents:
            l_tmp = self.get_neighbors_ids(agent=agent)
            agent.neighbors_ids_NWES = l_tmp
        # print("2", self.grid_map_pos_agent)

    #---------------------------------------------------

    def get_flag_from_grid(self):

        flag = {}
        for pos in self.grid_map_pos_agent.keys():
            if self.grid_map_pos_agent[pos]:
                flag[pos] = self.grid_map_pos_agent[pos].phenotype
            else:
                flag[pos] = self.default_missing_neighbor_state # questa variabile? O direttamente 0?

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

    def write_flag_data(self, setup_name, run, best_ind_run, best_ind, n, time_steps, flags, analysis_dir):

        dir_name = analysis_dir['data']+"/best_ind_"+str(best_ind_run)+"/"+str(setup_name)
        file_name = "/data_"+setup_name+"_flag_run_"+str(run)+"_n_"+str(n)+".csv"
        if not (os.path.exists(dir_name)):
            os.makedirs(dir_name, exist_ok=True)
        # save_data_to_csv(dir_name+file_name, [], header = ["Run", "Setup", "N", "Step", "Flags_distance", "Flag"])

        file_path = analysis_dir['data']+"/best_ind_"+str(best_ind_run)+"/flag_individual.txt"
        if not os.path.exists(file_path):
            with open (file_path, 'w') as f:
                f.write(str(best_ind))

        data_env_flag = []
        for step in range(time_steps):
            flags_distance = self.eval_flags_distance(flags[step])
            data_env_flag.append([str(run), setup_name, str(n), str(step), str(flags_distance).strip(), str(self.convert_flag_to_list(flags[step])).strip()])

        save_data_to_csv(dir_name+file_name, data_env_flag, header = ["Run", "Setup", "N", "Step", "Flags_distance", "Flag"])

    #---------------------------------------------------

    # def get_map_cell_neighbors_NWES(self):

    #     map_cell_neighbors_NWES = {}
    #     for row in range(self.grid_nb_rows):
    #         for col in range(self.grid_nb_cols):
    #             l_tmp = []
    #             agent =  self.grid_map_pos_agent[(row, col)]
    #             if agent is not None: #check
    #                 for neighbor_id in agent.neighbors_NWES:
    #                     l_tmp.append(self.grid_map_id_pos[neighbor_id])
    #             map_cell_neighbors_NWES[tuple((row, col))] = l_tmp

        # return map_cell_neighbors_NWES

    #---------------------------------------------------

    def plot_flag(self, setup_name, run, best_ind_run, n, step, flag, deleted_map_pos_agent=None, analysis_dir_plots=None):

        dir_name = analysis_dir_plots+"/best_ind_"+str(best_ind_run)+"/"+setup_name+"/flag"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        fig, ax = plt.subplots()

        agents = [a for a in self.grid_map_pos_agent.values() if a != None] #add assert?

        if self.grid_nb_rows <= 10 and self.grid_nb_cols <= 10:
            for agent in agents:
                for neighbor_id in agent.neighbors_ids_NWES: # neighbors_ids_NWES = list de ids ou None car ils seront entrées NN kale
                    if neighbor_id: # this can be none? kale
                        neighbor_pos = self.grid_map_id_pos[neighbor_id]
                        ax.plot([agent.pos[1], neighbor_pos[1]], [-agent.pos[0], -neighbor_pos[0]], color='black', linestyle=':', zorder=1)
                    
            circle_radius = 0.4

            x = []
            y = []
            greys = []

            if deleted_map_pos_agent is not None:
                x_deleted = [pos[0] for pos in deleted_map_pos_agent.keys()]
                y_deleted = [pos[1] for pos in deleted_map_pos_agent.keys()]

            for pos in flag:
                grey_value = flag[pos]

                if self.grid_nb_rows > 10 or self.grid_nb_cols > 10:
                    x.append(pos[1])
                    y.append(-pos[0])
                    greys.append(grey_value)

                else:
                    if grey_value > 0.9: # close to white
                        circle = patches.Circle((pos[1], -pos[0]), circle_radius, edgecolor='black', facecolor='white', linestyle='--', linewidth=1.0, zorder=2)    
                    else:
                        circle = patches.Circle((pos[1], -pos[0]), circle_radius, edgecolor=str(grey_value), facecolor='white', linewidth=6.0, zorder=2)

                    if deleted_map_pos_agent is not None and pos in deleted_map_pos_agent.keys():
                        circle = patches.Circle((pos[1], -pos[0]), circle_radius, edgecolor='tab:red', facecolor='white', linestyle='--', linewidth=2.0, zorder=2)    
                    
                    ax.add_patch(circle)

                    if self.grid_nb_rows < 6 and self.grid_nb_cols < 6:
                        ax.text(pos[1], -pos[0], "(" + str(pos[0]) +"," + str(pos[1]) + ")\n"+ str(round(grey_value,2)), color='black', va='center', ha='center')
                                
                    plt.axis('off')

            if self.grid_nb_rows > 10 or self.grid_nb_cols > 10:
                colors = [(0, 0, 0), (0.7, 0.9, 1.0)]  # Black to light blue
                positions = [0.0, 1.0]
                cmap = LinearSegmentedColormap.from_list('CustomBlackToLightBlue', list(zip(positions, colors)))
                plt.scatter(x, y, c=greys, cmap=cmap) # cmap='grey'
                
                if deleted_map_pos_agent is not None:
                    plt.scatter(x_deleted, y_deleted, c='tab:red') # deleted agents
                
                plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

            ax.set_aspect('equal')
            plt.xlim(-0.5, self.grid_nb_cols-0.5)
            plt.ylim(-self.grid_nb_rows+0.5, 0.5)

            plt.title(f"Flag")     
            # plt.title(f"Flag evolution states. Run {run}, individual {nb_ind}, step {step}.\nFitness (distance to flag target) = {round(dataset.loc[(dataset.Step==step),['Flags_distance']].values.tolist()[0][0], 2)}", pad=20, fontsize=9)
            plt.savefig(dir_name+"/"+setup_name+"_flag_run_"+str(run)+"_best_ind_"+str(best_ind_run)+"_n_"+str(n)+"_step_"+str(step)+".png")

            plt.clf()
            plt.close()

    #---------------------------------------------------

    @staticmethod
    def plot_flag_fitnesses_from_file(data_flag_file, setup_name, run, best_ind_run, n, switch_step, analysis_dir_plots):

        dir_name = analysis_dir_plots+"/best_ind_"+str(best_ind_run)+"/"+setup_name+"/fitness"
        if not (os.path.exists(dir_name)):
            os.makedirs(dir_name, exist_ok=True)

        dataset = pd.read_csv(data_flag_file)

        fig, ax = plt.subplots()

        x = dataset['Step'].tolist()
        y = dataset['Flags_distance'].tolist()
        plt.plot(x, y)
        plt.axvline(x=switch_step, color='r', linestyle='--')

        plt.ylim(-0.2, 1) # 0 and 1 are respectively min and max values of flag distance
        plt.xlabel("Steps", fontsize=12)
        plt.ylabel("Fitness (distance to flag target)", fontsize=12)
        plt.suptitle(f"Fitness related to the flag evolution over steps run="+str(run), fontsize=14)
        # plt.title(f"Generation {gen}, individual {nb_ind}, {env_eval_function_params['time_steps']} steps. Time window zone from step {time_window_start} to step {time_window_end} (included).", fontsize=9)
        plt.savefig(dir_name+"/"+setup_name+"_flag_fitnesses_run_"+str(run)+"_n_"+str(n)+".png")
        dir_name+"/"+setup_name+"_fitness_run_"+str(run)+"_best_ind_"+str(best_ind_run)+"_n_"+str(n)+".png"
        plt.clf()
        plt.close()

    #---------------------------------------------------

    # def exchange_agents(self, agent1, agent2):
    #     import copy
    #     # print("-1", self.grid_map_pos_agent)

    #     # print("agent1_pos", agent1.pos)
    #     # print("agent2_pos", agent2.pos)
    #     agent1_pos = agent1.pos
    #     agent2_pos = agent2.pos

    #     agent1.pos = agent2.pos
    #     agent2.pos = agent1_pos
    #     # print("agen!t1_pos", agent1.pos)
    #     # print("agent2_pos", agent2.pos)

    #     # a = copy.copy(agent1)
    #     self.grid_map_pos_agent[agent1_pos] = agent2
    #     self.grid_map_pos_agent[agent2_pos] = agent1

    #     self.refresh_grid()