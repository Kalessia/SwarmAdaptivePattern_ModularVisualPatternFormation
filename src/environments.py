import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

from nn import NeuralNetwork
from analysis import save_data_to_csv




###########################################################################
# Global variables
###########################################################################

global env
env = None


###########################################################################
# Evaluation functions
###########################################################################

def sphere_function(a):
    """
    a: ArrayLike
    """
    x = np.array(a)
    result = np.sum(x**2)
    # print(f"Sphere function value for {a}: {result}")
    return (result,)

#---------------------------------------------------

def rastrigin_function(a):
    """
    a: ArrayLike
    """
    x = np.array(a)
    A = 10
    n = len(x)
    result = A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    # print(f"Rastrigin function value for {a}: {result}")
    return (result,)

#---------------------------------------------------

def flag_automata(env_eval_function_params, analysis_dir, run, gen, weights):
    # time_window_end check validity

    automata_nb_rows = env_eval_function_params['automata_nb_rows']
    automata_nb_cols = env_eval_function_params['automata_nb_cols']
    flag_pattern = env_eval_function_params['flag_pattern']
    init_cell_state_value = env_eval_function_params['init_cell_state_value']
    time_steps = env_eval_function_params['time_steps']
    time_window_start = env_eval_function_params['time_window_start']
    time_window_end = env_eval_function_params['time_window_end']

    flags_distances = []
    in_t_window_zone_bools = []
    flags = []

    global env
    if not env:
        env = flagAutomata(automata_nb_rows, automata_nb_cols, flag_pattern, init_cell_state_value)
    
    env.set_cell_controller(weights)

    flags_distance = None
    sum_flags_distances = 0.0
    for step in range(time_steps):

        in_t_window_zone_bool = False
        flags_distance = env.eval_flags_distance()

        if step >= time_window_start and step <= time_window_end:   
            sum_flags_distances += flags_distance
            in_t_window_zone_bool = True

        flags_distances.append(flags_distance)
        in_t_window_zone_bools.append(in_t_window_zone_bool)
        flags.append(env.flag)
        env.step()

    env.write_flag_data(run, gen, time_steps, flags_distances, in_t_window_zone_bools, flags, weights, analysis_dir=analysis_dir)
    mean_tw_flags_distances = sum_flags_distances/(time_window_end - time_window_start)

    return (mean_tw_flags_distances,)


###########################################################################
# Environment flagAutomata
###########################################################################

class flagAutomata:
    def __init__(self, automata_nb_rows, automata_nb_cols, flag_pattern, init_cell_state_value=None) -> None:

        self.automata_nb_rows = automata_nb_rows
        self.automata_nb_cols = automata_nb_cols
        self.map_cell_neighbors_NWES = flagAutomata.build_map_cell_neighbors(self.automata_nb_rows, self.automata_nb_cols)

        self.flag_target = self.build_flag(flag_pattern)
        self.flag = self.init_flag(init_cell_state_value) # values in [0.0, 1.0]. 0.0 represents black, 1.0 represents white. NB: high value is excluded
        
        self.automata_mode = 1
        if self.automata_mode == 1:
            self.cell_controller = NeuralNetwork(nb_neuronsPerInputs=4, nb_hiddenLayers=1, nb_neuronsPerHidden=2, nb_neuronsPerOutputs=1)
        elif self.automata_mode == 2: # nb: weights ind doit etre adaptée 
            self.cell_controller = NeuralNetwork(nb_neuronsPerInputs=4, nb_hiddenLayers=1, nb_neuronsPerHidden=2, nb_neuronsPerOutputs=2)
            self.chemical_species = self.init_flag(init_cell_state_value) # change default value kale

        self.default_missing_neighbor_state = 0.0
            
    #---------------------------------------------------

    def init_flag(self, init_cell_state_value=None):

        if init_cell_state_value is None:
            # print("Automata flag cells are initialized with random uniform distribution of values in [0,1[")
            init_cell_state_value = np.random.uniform(0, 1, len(self.map_cell_neighbors_NWES))
        else:
            init_cell_state_value = [init_cell_state_value] * len(self.map_cell_neighbors_NWES)

        flag = {}
        for i, cell in enumerate(self.map_cell_neighbors_NWES.keys()):
            flag[cell] = init_cell_state_value[i]

        return flag

    #---------------------------------------------------
            
    def build_flag(self, flag_pattern):

        flag_target = {}

        if flag_pattern == "2_stripes":

            vertical_threshold = np.floor(self.automata_nb_cols*2/5)

            for cell in self.map_cell_neighbors_NWES.keys():
                if cell[1] < vertical_threshold:
                    flag_target[cell] = 0.0 # black
                else:
                    flag_target[cell] = 1.0 # white

        elif flag_pattern == "3_stripes":

            horizontal_threshold_upper = int(self.automata_nb_rows/3)
            horizontal_threshold_lower = int(self.automata_nb_rows/3*2)

            for cell in self.map_cell_neighbors_NWES.keys():
                if cell[0] < horizontal_threshold_upper:
                    flag_target[cell] = 0.0 # black
                elif cell[0] >= horizontal_threshold_lower:
                    flag_target[cell] = 1.0 # white
                else:
                    flag_target[cell] = 0.5 # grey

        elif flag_pattern == "circle":

            center = (np.floor(self.automata_nb_rows/2)-1, np.floor(self.automata_nb_cols/2)-1)
            radius = min(self.automata_nb_rows/2, self.automata_nb_cols/2) - 1

            for cell in self.map_cell_neighbors_NWES.keys():
                if ((cell[0] - center[0])**2 + (cell[1] - center[1])**2 <= radius**2):
                    flag_target[cell] = 0.0 # black
                else:
                    flag_target[cell] = 1.0 # white

        elif flag_pattern == "half_circle":

            center = (np.floor(self.automata_nb_rows/2)-1, np.floor(self.automata_nb_cols/2)-1)
            radius = min(self.automata_nb_rows/2, self.automata_nb_cols/2) - 1

            for cell in self.map_cell_neighbors_NWES.keys():

                if ((cell[0] - center[0])**2 + (cell[1] - center[1])**2 <= radius**2): # the cell is inside the circle area
                    if cell[1] < center[1]:
                        flag_target[cell] = 0.0 # black
                    else:
                        flag_target[cell] = 1.0 # white

                else:  # the cell is outside the circle area
                    if cell[1] < center[1]:
                        flag_target[cell] = 1.0 # white
                    else:
                        flag_target[cell] = 0.0 # black

        else:
            print("gneeeeeeeeee")
            exit()

        return flag_target

    #---------------------------------------------------

    def step(self):

        cells = list(self.flag.keys()) # ordered update

        if True: #Kale parametrable bool
            np.random.shuffle(cells) # random update

        for cell in cells:

            if self.automata_mode == 2:

                chemical_species, phenotype = self.compute_cell_chemical_species_and_phenotype(cell)
                self.flag[cell] = phenotype
                self.chemical_species[cell] = chemical_species

            else:    
                # print("step: The value in ", cell, "was", self.flag[cell])
                self.flag[cell] = self.compute_cell_state(cell)
                # print("step: After compute_cell_state, the value in", cell, "is", self.flag[cell], "\n")
        

    #---------------------------------------------------

    def set_cell_controller(self, weights):
        self.cell_controller.setWeightsFromList(weights)

    #---------------------------------------------------

    @staticmethod
    def is_cell_pos_valid(pos, automata_nb_rows, automata_nb_cols):
        return pos[0]>=0 and pos[1]>=0 and pos[0]<automata_nb_rows and pos[1]<automata_nb_cols

    #---------------------------------------------------
    
    @staticmethod
    def build_map_cell_neighbors(automata_nb_rows, automata_nb_cols):

        map_cell_neighbors_NWES = {}
        for row in range(automata_nb_rows):
            for col in range(automata_nb_cols):
                l_tmp = []
                for pos in [(row-1, col), (row, col-1), (row, col+1), (row+1, col)]:
                    if flagAutomata.is_cell_pos_valid(pos, automata_nb_rows, automata_nb_cols):
                        l_tmp.append(tuple(pos))
                    else:
                        l_tmp.append(None)
                map_cell_neighbors_NWES[tuple((row, col))] = l_tmp

        return map_cell_neighbors_NWES

    #---------------------------------------------------

    def compute_cell_state(self, cell):
        
        neighbors_states = []
        for neighbor in self.map_cell_neighbors_NWES[cell]:
            if neighbor:
                neighbors_states.append(self.flag[neighbor])
            else:
                neighbors_states.append(self.default_missing_neighbor_state)
        # print("compute_cell_state: The neighbors of ", cell, "are", self.map_cell_neighbors_NWES[cell])
        # print("compute_cell_state: Their neighbors states are", neighbors_states)
        cell_state = self.cell_controller.predict(neighbors_states)[0] # forwardPropagation, stableSigmoid on the last layer
        # print("compute_cell_state: The predicted state for", cell, "is", cell_state)
        return cell_state
        
    #---------------------------------------------------

    def compute_cell_chemical_species_and_phenotype(self, cell):
        
        neighbors_states = []
        for neighbor in self.map_cell_neighbors_NWES[cell]:
            if neighbor:
                neighbors_states.append(self.chemical_species[neighbor])
            else:
                neighbors_states.append(self.default_missing_neighbor_state)
        # print("compute_cell_state: The neighbors of ", cell, "are", self.map_cell_neighbors_NWES[cell])
        # print("compute_cell_state: Their neighbors states are", neighbors_states)
        chemical_species, phenotype = self.cell_controller.predict(neighbors_states) # forwardPropagation, stableSigmoid on the last layer
        # print("compute_cell_state: The predicted state for", cell, "is", cell_state)
        return chemical_species, phenotype
        
    #---------------------------------------------------

    def eval_flags_distance(self):
        sum_states = 0.0
        for cell in self.map_cell_neighbors_NWES.keys():
            sum_states += (self.flag_target[cell] - self.flag[cell])**2

        flags_distance = sum_states/len(self.map_cell_neighbors_NWES)
        return flags_distance
    
    #---------------------------------------------------

    def convert_flag_to_list(self, flag):
        return list(flag.values())

    #---------------------------------------------------
    
    def write_flag_data(self, run, gen, time_steps, flags_distances, in_t_window_zone_bools, flags, weights, analysis_dir):

        if not (os.path.exists(analysis_dir['root']+"/data_all_runs/data_env_flag_target.csv")):
            save_data_to_csv(analysis_dir['root']+"/data_all_runs/data_env_flag_target.csv", [[0, 0, 0, 0,  str(self.convert_flag_to_list(self.flag_target)).strip(), 0]], header = ["Generation", "Step", "Flags_distance", "Time_window_zone", "Flag", "Individual"])
        
        if not (os.path.exists(analysis_dir['data']+"/data_env_flag/data_env_flag_run_"+str(run)+"_gen_"+str(gen)+".csv")):
            os.makedirs(analysis_dir['data']+"/data_env_flag/", exist_ok=True)
            save_data_to_csv(analysis_dir['data']+"/data_env_flag/data_env_flag_run_"+str(run)+"_gen_"+str(gen)+".csv", [], header = ["Generation", "Step", "Flags_distance", "Time_window_zone", "Flag", "Individual"])    

        data_env_flag = []
        for step in range(time_steps):
            data_env_flag.append([str(gen), str(step), str(flags_distances[step]).strip(), str(in_t_window_zone_bools[step]).strip(), str(self.convert_flag_to_list(flags[step])).strip(), str(weights).strip()])

        save_data_to_csv(analysis_dir['data']+"/data_env_flag/data_env_flag_run_"+str(run)+"_gen_"+str(gen)+".csv", data_env_flag)

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
        file_path = analysis_dir_plots+"/run_"+str(run)+"_gen_"+str(gen)+"_individual_"+str(nb_ind)+"/flag_individual.txt"
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
