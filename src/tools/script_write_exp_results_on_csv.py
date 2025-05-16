import os
import json
import pandas as pd
from learning_initializations import save_data_to_csv




if os.path.exists("data.csv"):
    os.remove("data.csv")

dirs = os.listdir("../data_plots/simulationAnalysis")
simulationAnalysis_dirs = [dirs[i] for i in range(len(dirs)) if dirs[i].startswith("flag_")]
simulationAnalysis_dirs.sort()

save_data_to_csv("data.csv", None, header = ["SimuName", "FlagTarget", "SizeX", "SizeY", "Run", "FitnessBestInd", "BestIndGen", "Mean", "Std", "MaxGen", "NNStructure", "async_update_states", "random_init_states", "with_noise", "Visible", "Time"])


for i in range(len(simulationAnalysis_dirs)):
    print(simulationAnalysis_dirs[i])

    with open("../data_plots/simulationAnalysis/"+ simulationAnalysis_dirs[i] +"/learning/learning_params.json", "r") as f:
        params = json.load(f)
    
    d1 = pd.read_csv("../data_plots/simulationAnalysis/"+ simulationAnalysis_dirs[i] + "/learning/data_all_runs/data_evo_all_runs_best_ind_per_run.csv")
    
    mean = None
    std = None
    if os.path.exists("../data_plots/simulationAnalysis/"+ simulationAnalysis_dirs[i] + "/learning/data_all_runs/data_evo_all_runs_mean_std.csv"):    
        try:
            d2 = pd.read_csv("simulationAnalysis/"+ simulationAnalysis_dirs[i] + "/learning/data_all_runs/data_evo_all_runs_mean_std.csv")  
            mean = round(d2.loc[0]['Mean fitnesses'], 3)
            std = round(d2.loc[0]['Std fitnesses'], 3)
        except:
            mean = None
            std = None


    t = None
    if os.path.exists("../data_plots/simulationAnalysis/"+ simulationAnalysis_dirs[i] + "/learning/data_all_runs/data_evo_all_runs_time.csv"):
        d3 = pd.read_csv("../data_plots/simulationAnalysis/"+ simulationAnalysis_dirs[i] + "/learning/data_all_runs/data_evo_all_runs_time.csv")

    
    for nb_run in range(int(params['nb_runs'])):
        
        if os.path.exists("../data_plots/simulationAnalysis/"+ simulationAnalysis_dirs[i] + "/learning/data_all_runs/data_evo_all_runs_time.csv"):
            try:
                t = int(d3.loc[(d1.Run==nb_run),['Time(s)']].values.tolist()[0][0])
            except:
                t = int(d3.loc[(d1.Run==nb_run),['Time']].values.tolist()[0][0])

        save_data_to_csv("data.csv",
                    [[str(simulationAnalysis_dirs[i]),
                    str(params['flag_pattern']),
                    str(params['grid_nb_rows']),
                    str(params['grid_nb_cols']),
                    str(nb_run),
                    str(d1.loc[(d1.Run==nb_run),['Fitness']].values.tolist()[0][0]),
                    str(d1.loc[(d1.Run==nb_run),['Generation']].values.tolist()[0][0]),
                    str(mean),
                    str(std),
                    str(params['nb_generations']),
                    str(f"{params['nb_neuronsPerInputs']}-{params['nb_hiddenLayers']}-{params['nb_neuronsPerHidden']}-{params['nb_neuronsPerOutputs']}"),
                    str(params['learning_random_async_update_states_bool']),
                    str(params['learning_random_init_states_bool']),
                    str(params['learning_with_noise_bool']),
                    str(None),
                    str(t)
                    ]])
