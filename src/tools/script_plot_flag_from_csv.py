import json
import pandas as pd

from environments import swarmGrid


def plot_flag_from_dir(flag_csv_files, params):
      
    dataset = pd.read_csv(flag_csv_files)

    run, gen, nb_ind = get_gen_ind_from_file_name(flag_csv_files)
    # dataset_gen = dataset.loc[(dataset.Generation==gen)]
    # dataset = dataset_gen.loc[(dataset_gen.Individual==str(ind))]
    # nb_ind = dataset['Nb_ind'].unique()[0]

    steps = dataset['Step'].unique()

    for step in range(steps):
        flag_list = dataset.loc[(dataset.Step==step),['Flag']].values.tolist()[0][0]
        flag_list = eval(flag_list)
        fitness = dataset.loc[(dataset.Step==step),['Flags_distance']].values.tolist()[0][0]            
        deleted_pos = dataset.loc[(dataset.Step==step),['Deleted_agents_positions']].values.tolist()[0][0]
        deleted_pos = eval(deleted_pos)
        nb_moves_per_step = dataset.loc[(dataset.Step==step),['Nb_moves']].values.tolist()[0][0]

        swarmGrid.plot_flag(grid_nb_rows=params['grid']['grid_nb_rows'],
                            grid_nb_cols=params['grid']['grid_nb_cols'],
                            setup_name=None,
                            run=run,
                            nb_ind=nb_ind,
                            gen=gen,
                            nb_eval=nb_eval,
                            n="",
                            step=step,
                            flag_list=flag_list,
                            fitness=fitness,
                            deleted_pos=deleted_pos,
                            nb_moves_per_step=nb_moves_per_step,
                            analysis_dir_plots=params['analysis_dir']['root']+ f"/run_{run:03}/plots/env")

def get_gen_ind_from_file_name(file_name):
    file_name_chunks = file_name.split('/')[-1]
    file_name_chunks = file_name_chunks.split('.')[0]
    file_name_chunks = file_name_chunks.split('_')
    print(file_name_chunks)
    run = int(file_name_chunks[4])
    gen = int(file_name_chunks[6])
    nb_ind = int(file_name_chunks[8])
    return run, gen, nb_ind



flag_csv_files = [

"/home/kalessia/flagAutomata/data_plots/simulationAnalysis/sliding_puzzle_coordinates_2025-03-12_00-44-39_two-bands_16x16/learning_coordinates_halfdiskfail2/run_007/data/data_env_flag/data_env_flag_run_007_gen_01397_eval_0013979.csv"

]

# Get parameters from the learning simulation
params_file = "/home/kalessia/flagAutomata/data_plots/simulationAnalysis/sliding_puzzle_coordinates_2025-03-12_00-44-39_two-bands_16x16/learning_coordinates_halfdiskfail2/coordinates_learning_params.json"
with open(params_file, "r") as f:
    params = json.load(f)

for flag_csv_file in flag_csv_files:
    plot_flag_from_dir(flag_csv_file, params)