import numpy as np
from nn import NeuralNetwork


import matplotlib.pyplot as plt
import matplotlib.patches as patches

from analysis import *



first_call_bool = True


def sphere_function(a):
    """
    a: ArrayLike
    """
    x = np.array(a)
    result = np.sum(x**2)
    # print(f"Sphere function value for {a}: {result}")
    return (result,)

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


def flag_automata(time_steps, time_window_start, time_window_end, analysis_dir, weights):
    time_steps = 10
    time_window_start = 7
    time_window_end = 10
    # self.time_steps = 200 # evolution time, i.e. total life of the automata
    # self.time_window_start = 140 # time_window_start is in [0, time_window_end]
    # self.time_window_end = 200 # time_window_end is in [time_window_start, time_steps]

    fa = flagAutomata(automata_nb_rows=3, automata_nb_cols=5)
    fa.set_cell_controller(weights)

    flags_distance = None
    sum_flags_distances = 0.0
    for t in range(time_steps):

        if t >= time_window_start and t <= time_window_end:
            flags_distance = fa.eval_flags_distance()
            sum_flags_distances += flags_distance

        fa.write_flagAutomata_data(t, flags_distance, weights, analysis_dir=analysis_dir + "/data")
        fa.step()

    mean_tw_flags_distances = sum_flags_distances/(time_window_end - time_window_start)





    # plot flag target
    # flag_target_list = fa.convert_flag_to_list(fa.flag_target)
    # fa.plot_flag_target(flag_target_list, title = "Flag target", analysis_dir=analysis_dir + "/plots/flags", )

    # plot flag
    fa.plot_flag_target2(data_flag_file=analysis_dir+"/data/data_flag_target.csv", title="Flaggggggghhhhh target", analysis_dir=analysis_dir+"/plots/flags")
    fa.plot_flag_target2(data_flag_file=analysis_dir+"/data/data_flag.csv", title="Flaggggggghhhhh", analysis_dir=analysis_dir+"/plots/flags")


    exit()
    return (mean_tw_flags_distances,)


class flagAutomata:
    def __init__(self, automata_nb_rows, automata_nb_cols) -> None:

        self.automata_nb_rows = automata_nb_rows
        self.automata_nb_cols = automata_nb_cols
        self.connectivity_type = "vonNeumann" # moore
        self.map_cell_neighbors = self.build_map_cell_neighbors()

        self.cell_controller = NeuralNetwork(nb_neuronsPerInputs=4, nb_hiddenLayers=1, nb_neuronsPerHidden=2, nb_neuronsPerOutputs=1)
        self.default_missing_neighbor_state = -1

        self.flag = self.init_flag(0.0) # value in [0.0, 1.0]. 0.0 represents white, 1.0 represents black
        self.flag_target = self.build_flag()

    #---------------------------------------------------

    def init_flag(self, init_cell_state_value):

        flag = {}
        for cell in self.map_cell_neighbors.keys():
            flag[cell] = init_cell_state_value

        return flag

    #---------------------------------------------------
            
    def build_flag(self):

        flag_target = {}
        nb_cols = np.floor(self.automata_nb_cols*2/5)

        for cell in self.map_cell_neighbors.keys():
            if cell[1] < nb_cols:
                flag_target[cell] = 0.0 # black
            else:
                flag_target[cell] = 1.0 # white

        return flag_target

    #---------------------------------------------------

    def step(self):

        cells = self.flag.keys() # ordered update

        if False: #Kale parametrable bool
            cells = np.random.shuffle(cells) # random update

        for cell in cells:
            self.flag[cell] = self.compute_cell_state(cell)

    #---------------------------------------------------
            
    # def convert_flagStates_greyScale(flag):

    #     for k, v in flag.items():

    #         updated_dict = {key: value for key, value in zip(dictionary.keys(), result_array)}
    #     return cellState * 255
    #     return

    #---------------------------------------------------

    def set_cell_controller(self, weights):
        self.cell_controller.setWeightsFromList(weights)

    #---------------------------------------------------

    def is_cell_pos_valid(self, pos):
        return pos[0]>=0 and pos[1]>=0 and pos[0]<self.automata_nb_rows and pos[1]<self.automata_nb_cols

    #---------------------------------------------------
    
    def build_map_cell_neighbors(self):

        map_cell_neighbors = {}
        if self.connectivity_type == "vonNeumann":
            for row in range(self.automata_nb_rows):
                for col in range(self.automata_nb_cols):
                    l_tmp = []
                    for pos in [(row-1, col), (row, col-1), (row, col+1), (row+1, col)]:
                        if self.is_cell_pos_valid(pos):
                            l_tmp.append(tuple(pos))
                        else:
                            l_tmp.append(None)
                    map_cell_neighbors[tuple((row, col))] = l_tmp

        return map_cell_neighbors

    #---------------------------------------------------

    def compute_cell_state(self, cell):
        
        neighbors_states = []
        for neighbor in self.map_cell_neighbors[cell]:
            if neighbor:
                neighbors_states.append(self.flag[neighbor])
            else:
                neighbors_states.append(self.default_missing_neighbor_state)
        cell_state = self.cell_controller.forwardPropagation(neighbors_states)[0] # check
        return cell_state
        
    #---------------------------------------------------

    def eval_flags_distance(self):
        sum_states = 0.0
        for cell in self.map_cell_neighbors.keys():
            sum_states += (self.flag_target[cell] - self.flag[cell])**2

        flags_distance = sum_states/len(self.map_cell_neighbors)
        return flags_distance
    
    #---------------------------------------------------

    def convert_flag_to_list(self, flag):
        return list(flag.values())

    #---------------------------------------------------
    
    def write_flagAutomata_data(self, t, flags_distance, weights, analysis_dir):
        global first_call_bool

        if first_call_bool:
            save_data_to_csv(analysis_dir + "/data_flag_target.csv", [], header = ["Step", "Flags_distance", "Flag", "Individual"])
            save_data_to_csv(analysis_dir + "/data_flag_target.csv", [["None", "None", str(self.convert_flag_to_list(self.flag_target)).strip(), "None"]])
            save_data_to_csv(analysis_dir + "/data_flag.csv", [], header = ["Step", "Flags_distance", "Flag", "Individual"])    
            first_call_bool = False

        save_data_to_csv(analysis_dir + "/data_flag.csv", [[str(t), str(flags_distance).strip(), str(self.convert_flag_to_list(self.flag)).strip(), str(weights).strip()]])

    #---------------------------------------------------

    def plot_flag_target(self, flag_target_list, title, analysis_dir):
        circle_radius = 0.4

        fig, ax = plt.subplots()

        for cell, neighbors in self.map_cell_neighbors.items():
            for neighbor in neighbors:
                if neighbor:
                    ax.plot([cell[1], neighbor[1]], [-cell[0], -neighbor[0]], color='black', linestyle=':', zorder=1)

        for p, pos in enumerate(self.map_cell_neighbors.keys()):
            c = flag_target_list[p]
        
            if c > 0.9:
                circle = patches.Circle((pos[1], -pos[0]), circle_radius, edgecolor='black', facecolor='white', linestyle='-', linewidth=1.0, zorder=2)    
            else:
                circle = patches.Circle((pos[1], -pos[0]), circle_radius, edgecolor=str(c), facecolor='white', linewidth=6.0, zorder=2)
            ax.add_patch(circle)
            ax.text(pos[1], -pos[0], "(" + str(pos[0]) +"," + str(pos[1]) + ")\n"+ str(round(flag_target_list[p],2)), color='black', va='center', ha='center')

        ax.set_aspect('equal')
        plt.xlim(-0.5, self.automata_nb_cols-0.5)
        plt.ylim(-self.automata_nb_rows+0.5, 0.5)
        plt.axis('off')
        plt.title(title)

        os.makedirs(analysis_dir, exist_ok=True)
        plt.savefig(analysis_dir + "/" + title)

    #---------------------------------------------------

    def plot_flag_target2(self, data_flag_file, title, analysis_dir):

        dataset = pd.read_csv(data_flag_file)

        individuals = dataset['Individual'].unique()
        for i, ind in enumerate(individuals):
            dataset_ind = dataset.loc[dataset.Individual==ind]

            steps = dataset_ind['Step']
            for step in steps:
                
                flag_list = dataset_ind.loc[(dataset.Step==step),['Flag']].values.tolist()[0][0]
                flag_list = str(flag_list).replace('[', '').replace(']', '').strip()
                flag_list = np.asarray(flag_list.split(','), dtype=np.float32)



                circle_radius = 0.4

                fig, ax = plt.subplots()

                for cell, neighbors in self.map_cell_neighbors.items():
                    for neighbor in neighbors:
                        if neighbor:
                            ax.plot([cell[1], neighbor[1]], [-cell[0], -neighbor[0]], color='black', linestyle=':', zorder=1)

                for p, pos in enumerate(self.map_cell_neighbors.keys()):
                    c = flag_list[p]
                
                    if c > 0.9:
                        circle = patches.Circle((pos[1], -pos[0]), circle_radius, edgecolor='black', facecolor='white', linestyle='-', linewidth=1.0, zorder=2)    
                    else:
                        circle = patches.Circle((pos[1], -pos[0]), circle_radius, edgecolor=str(c), facecolor='white', linewidth=6.0, zorder=2)
                    ax.add_patch(circle)
                    ax.text(pos[1], -pos[0], "(" + str(pos[0]) +"," + str(pos[1]) + ")\n"+ str(round(flag_list[p],2)), color='black', va='center', ha='center')

                ax.set_aspect('equal')
                plt.xlim(-0.5, self.automata_nb_cols-0.5)
                plt.ylim(-self.automata_nb_rows+0.5, 0.5)
                plt.axis('off')
                plt.title(title)

                if ind != "None":
                    if int(step) == 0:
                        os.makedirs(analysis_dir + "/flag_individual_"+str(i), exist_ok=True)
                        with open (analysis_dir + "/flag_individual_"+str(i)+"/flag_individual.txt", 'w') as f:
                            f.write(ind + "\n")
                
                    plt.savefig(analysis_dir + "/flag_individual_"+str(i)+"/flag_step_"+str(step)+".png")
                else: # target
                    os.makedirs(analysis_dir, exist_ok=True)
                    plt.savefig(analysis_dir + "/" + title)





















    def plot_flag(self, flag_list):


                fig, ax = plt.subplots()



                # Draw connections
                for cell, neighbors in self.map_cell_neighbors.items():
                    for neighbor in neighbors:
                        if neighbor:
                            ax.plot([cell[1], neighbor[1]], [-cell[0], -neighbor[0]], color='black', linestyle='--')

                for p, pos in enumerate(self.map_cell_neighbors.keys()):
                    c = str((1 - flag_list[p])/2)
                    circle_radius = 0.4
                    circle = patches.Circle((pos[1], -pos[0]), circle_radius, edgecolor=c, facecolor='white', linewidth=3.0)
                    ax.add_patch(circle)
                    # ax.add_artist(circle)
                    ax.text(pos[1], -pos[0], "(" + str(pos[1]) +"," + str(-pos[0]) + ")\n"+ str(round(flag_list[p],2)), color='black', va='center', ha='center')
                plt.title(str(step))

                ax.set_aspect('equal', adjustable='box')
                plt.xlim(-circle_radius, self.automata_nb_cols + circle_radius)
                plt.ylim(-circle_radius, self.automata_nb_rows + circle_radius)
                plt.axis('off')

                if int(step) == 0:
                    os.makedirs(analysis_dir + "/flag_individual_"+str(i), exist_ok=True)
                    with open (analysis_dir + "/flag_individual_"+str(i)+"/flag_individual.txt", 'w') as f:
                        f.write(ind + "\n")
                
                plt.savefig(analysis_dir + "/flag_individual_"+str(i)+"/flag_step_"+str(step)+".png")

                plt.clf()