import os

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
    def __init__(self, pos) -> None:

        global count_id
        count_id += 1
        self.id = count_id

        self.pos = pos
        # self.state = None
        # init_cell_state_value = np.random.uniform(0, 1, len(self.map_cell_neighbors_NWES))
        init_cell_state_value = np.random.uniform(0, 1, 1)[0]
        # print("init_cell_state_value", init_cell_state_value)
        self.chemical_species = init_cell_state_value
        self.phenotype = init_cell_state_value # meme valeur? ou differente?
        self.neighbors_ids_NWES = None


###########################################################################
# Environment swarm grid
###########################################################################

class swarmGrid:
    def __init__(self, grid_nb_rows, grid_nb_cols, agent_controller_weights, flag_target) -> None:

        self.grid_nb_rows = grid_nb_rows
        self.grid_nb_cols = grid_nb_cols
        self.size = grid_nb_rows * grid_nb_cols
        self.init_grid()

        self.agent_controller = NeuralNetwork(nb_neuronsPerInputs=4, nb_hiddenLayers=1, nb_neuronsPerHidden=2, nb_neuronsPerOutputs=2)
        self.agent_controller.setWeightsFromList(agent_controller_weights)

        self.default_missing_neighbor_state = 0.0

        self.flag_target = flag_target

    #---------------------------------------------------

    def init_grid(self):

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

    def step(self):

        agents_ids = list(self.grid_map_id_pos.keys()) # ordered update

        if True: #Kale parametrable bool
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

    def is_pos_valid(self, pos):
        return pos[0]>=0 and pos[1]>=0 and pos[0]<self.grid_nb_rows and pos[1]<self.grid_nb_cols

    #---------------------------------------------------

    def get_agent_neighbors_ids(self, agent):

        l_tmp = []
        row, col = agent.pos
        for pos in [(row-1, col), (row, col-1), (row, col+1), (row+1, col)]:
            if self.is_pos_valid(pos) and self.grid_map_pos_agent[tuple(pos)] != None:
                l_tmp.append(self.grid_map_pos_agent[tuple(pos)].id)
            else:
                l_tmp.append(None)
        agent.neighbors_ids_NWES = l_tmp

    #---------------------------------------------------

    def compute_agent_state(self, agent):
        
        # print("id", agent.id, "neighbors_ids", agent.neighbors_ids_NWES)

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
    
    def refresh_grid(self):

        agents_ids = list(self.grid_map_id_pos.keys())

        # Collect the neighbors of each agent (if we call 'refresh', they have likely changed)
        for agent_id in agents_ids:
            self.get_agent_neighbors_ids(agent=self.grid_map_pos_agent[self.grid_map_id_pos[agent_id]])

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

        flags_distance = sum_states/len(self.grid_map_pos_agent.keys())
        return flags_distance
    
    #---------------------------------------------------

    def convert_flag_to_list(self, flag):
        return list(flag.values())

    #---------------------------------------------------
    
    def write_flag_data(self, run, time_steps, flags, analysis_dir):

        if not (os.path.exists(analysis_dir['data']+"/data_env_flag/data_env_flag_run_"+str(run)+".csv")):
            os.makedirs(analysis_dir['data']+"/data_env_flag/", exist_ok=True)
            save_data_to_csv(analysis_dir['data']+"/data_env_flag/data_env_flag_run_"+str(run)+".csv", [], header = ["Run", "Step", "Flags_distance", "Flag", "Individual"])

        data_env_flag = []
        for step in range(time_steps):
            flags_distance = self.eval_flags_distance(flags[step])
            data_env_flag.append([str(run), str(step), str(flags_distance).strip(), str(self.convert_flag_to_list(flags[step])).strip()])

        save_data_to_csv(analysis_dir['data']+"/data_env_flag/data_env_flag_run_"+str(run)+".csv", data_env_flag)

    #---------------------------------------------------

    @staticmethod
    def plot_flag_from_file(env_eval_function_params=None, data_flag_file=None, run=None, gen=None, ind=None, steps=None, analysis_dir_plots=None):

        automata_nb_rows = env_eval_function_params['automata_nb_rows']
        automata_nb_cols = env_eval_function_params['automata_nb_cols']
        map_cell_neighbors_NWES = flagAutomata.build_map_cell_neighbors(automata_nb_rows, automata_nb_cols)

        dataset = pd.read_csv(data_flag_file)

        if gen is not None:
            dataset_gen = dataset.loc[(dataset.Generation==gen)]
            dataset = dataset_gen.loc[(dataset_gen.Individual==str(ind))]

        if steps is None:
            steps = dataset['Step'].unique()

        for step in steps:
            flag_list = dataset.loc[(dataset.Step==step),['Flag']].values.tolist()[0][0]
            flag_list = str(flag_list).replace('[', '').replace(']', '').strip()
            flag_list = np.asarray(flag_list.split(','), dtype=np.float32)

            fig, ax = plt.subplots()

            if automata_nb_rows <= 10 and automata_nb_rows <= 10:
                for cell, neighbors in map_cell_neighbors_NWES.items():
                    for neighbor in neighbors:
                        if neighbor:
                            ax.plot([cell[1], neighbor[1]], [-cell[0], -neighbor[0]], color='black', linestyle=':', zorder=1)

                circle_radius = 0.4

            x = []
            y = []
            greys = []

            for p, pos in enumerate(map_cell_neighbors_NWES.keys()):
                grey_value = flag_list[p]

                if automata_nb_rows > 10 or automata_nb_rows > 10:
                    x.append(pos[1])
                    y.append(-pos[0])
                    greys.append(grey_value)

                else:
                    if grey_value > 0.9: # close to white
                        circle = patches.Circle((pos[1], -pos[0]), circle_radius, edgecolor='black', facecolor='white', linestyle='--', linewidth=1.0, zorder=2)    
                    else:
                        circle = patches.Circle((pos[1], -pos[0]), circle_radius, edgecolor=str(grey_value), facecolor='white', linewidth=6.0, zorder=2)
                    ax.add_patch(circle)

                    if automata_nb_rows < 6 and automata_nb_rows < 6:
                        ax.text(pos[1], -pos[0], "(" + str(pos[0]) +"," + str(pos[1]) + ")\n"+ str(round(flag_list[p],2)), color='black', va='center', ha='center')
                                
                    plt.axis('off')

            if automata_nb_rows > 10 or automata_nb_rows > 10:
                colors = [(0, 0, 0), (0.7, 0.9, 1.0)]  # Black to light blue
                positions = [0.0, 1.0]
                cmap = LinearSegmentedColormap.from_list('CustomBlackToLightBlue', list(zip(positions, colors)))
                plt.scatter(x, y, c=greys, cmap=cmap) # cmap=''grey
                plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)


            ax.set_aspect('equal')
            plt.xlim(-0.5, automata_nb_cols-0.5)
            plt.ylim(-automata_nb_rows+0.5, 0.5)


            if gen is None: # target flag
                plt.title(f"Flag target, {automata_nb_rows} rows x {automata_nb_cols} columns", fontsize=14)
                os.makedirs(analysis_dir_plots, exist_ok=True)
                plt.savefig(analysis_dir_plots+"/plot_env_flag_target.png")

            else:
                individuals_gen = dataset_gen['Individual'].unique()
                nb_ind = np.where(individuals_gen==str(ind))[0][0]
                file_name = "run_"+str(run)+"_gen_"+str(gen)+"_individual_"+str(nb_ind)
                if int(step) == steps[0]:
                    os.makedirs(analysis_dir_plots+"/"+file_name+"/flag", exist_ok=True)
                    file_path = analysis_dir_plots+"/"+file_name+"/flag_individual.txt"
                    if not os.path.exists(file_path):
                        with open (file_path, 'w') as f:
                            f.write(str(ind))
                plt.title(f"Flag evolution states. Run {run}, generation {gen}, individual {nb_ind}, step {step}.\nFitness (distance to flag target) = {round(dataset.loc[(dataset.Step==step),['Flags_distance']].values.tolist()[0][0], 2)}", pad=20, fontsize=9)
                plt.savefig(analysis_dir_plots+"/"+file_name+"/flag/plot_env_flag_"+file_name+"_step_"+str(step)+".png")

            plt.clf()
            plt.close()

    #---------------------------------------------------

    @staticmethod
    def plot_flag_fitnesses_from_file(env_eval_function_params=None, data_flag_file=None, run=None, gen=None, ind=None, analysis_dir_plots=None):

        time_window_start = env_eval_function_params['time_window_start']
        time_window_end = env_eval_function_params['time_window_end']
        time_window_length = time_window_end - time_window_start + 1

        df = pd.read_csv(data_flag_file)

        dataset_gen = df.loc[(df.Generation==gen)]
        individuals_gen = dataset_gen['Individual'].unique()
        dataset = dataset_gen.loc[(dataset_gen.Individual==str(ind))]

        fig, ax = plt.subplots()

        x = dataset['Step'].tolist()
        y = dataset['Flags_distance'].tolist()

        plt.plot(x, y)

        rectangle = patches.Rectangle((time_window_start, 0), time_window_length, dataset['Flags_distance'].max(), linewidth=1, edgecolor=None, facecolor='lemonchiffon', alpha=0.5)
        ax.add_patch(rectangle)


        nb_ind = np.where(individuals_gen==str(ind))[0][0]
        file_name = "run_"+str(run)+"_gen_"+str(gen)+"_individual_"+str(nb_ind)
        if not os.path.exists(analysis_dir_plots+"/"+file_name):
            os.makedirs(analysis_dir_plots+"/"+file_name, exist_ok=True)
            file_path = analysis_dir_plots+"/"+file_name+"/flag_individual.txt"
            if not os.path.exists(file_path):
                with open (file_path, 'w') as f:
                    f.write(str(ind))

        plt.xlabel("Steps", fontsize=12)
        plt.ylabel("Fitness (distance to flag target)", fontsize=12)
        plt.suptitle(f"Fitness related to the flag evolution over steps", fontsize=14)
        plt.title(f"Generation {gen}, individual {nb_ind}, {env_eval_function_params['time_steps']} steps. Time window zone from step {time_window_start} to step {time_window_end} (included).", fontsize=9)
        plt.savefig(analysis_dir_plots+"/run_"+str(run)+"_gen_"+str(gen)+"_individual_"+str(nb_ind)+"/flag_fitnesses_run_"+str(run)+"_gen_"+str(gen)+"_individual_"+str(nb_ind)+".png")
        plt.clf()
        plt.close()