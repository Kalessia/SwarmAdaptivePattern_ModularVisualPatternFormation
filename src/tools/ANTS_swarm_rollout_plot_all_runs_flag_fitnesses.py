import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


def plot_all_runs_flag_fitnesses():

    # swarm_rollout_path = "/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_models-A-C-D-GECCO/sliding_puzzle_coordinates_2025-05-22_14-07-36_centered-half-discs_16x16/learning_coordinates_centered-half-discs_16x16_modelC_4-[]-3_6-[5,5]-1_swarm_rollout_2025-11-02_20-54-54" # centered-half-disc_singleEnvXY280K_pixelC55-16x16_newEnv16x16
    # swarm_rollout_path = "/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_models-A-C-D-GECCO/sliding_puzzle_coordinates_2025-05-22_14-07-36_centered-half-discs_16x16/learning_coordinates_centered-half-discs_16x16_modelC_4-[]-3_6-[5,5]-1_swarm_rollout_2025-11-02_23-30-58" # centered-half-disc_multiEnvXY280K_pixelC55-16x16_newEnv16x16



    swarm_rollout_path = "/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_models-A-C-D-GECCO/sliding_puzzle_coordinates_2025-05-22_14-07-36_centered-half-discs_16x16/learning_coordinates_rgb-italian-flag_16x16_modelA_4-[]-3_2-[5,5]-3_2025-10-21_01-56-46_swarm_rollout_2025-11-04_05-53-24"

    plt.style.use('dark_background')
    sns.set_style("darkgrid", {"grid.color": "white"})

    with open(swarm_rollout_path+"/swarm_rollout_params.json", "r") as f:
        swarm_rollout_params = json.load(f)

    run_dirs = sorted([d for d in os.listdir(swarm_rollout_path) if d.startswith("run_")]) # list all runs in swarm_rollout_path
    best_inds = sorted([d for d in os.listdir(f"{swarm_rollout_path}/{run_dirs[0]}") if d.startswith("best_ind_")]) # list all best_inds in run_000

    twenty_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#00429d", "#96ffea", "#ff66b3", "#00cc44", "#ffcc00",
        "#a6761d", "#009e73", "#f781bf", "#56b4e9", "#e41a1c",
    ]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=twenty_colors)
    time_steps = swarm_rollout_params['environment']['time_steps']

    env_dims_list = [[swarm_rollout_params['grid']['grid_nb_rows'], swarm_rollout_params['grid']['grid_nb_cols']]]
    if swarm_rollout_params['swarm_rollout_settings']['env_name'] in ['sliding_puzzle_multiEnvs', 'sliding_puzzle_multiEnvs_coordinates']:
        env_dims_list = swarm_rollout_params['swarm_rollout_settings']['sliding_puzzle_multiEnvs']['env_dims_list']

    for env_id, env_dims in enumerate(env_dims_list):
        for best_ind in best_inds:
            _, ax = plt.subplots(figsize=(11, 7), dpi=300, constrained_layout=True)

            time_window_start = swarm_rollout_params['environment']['time_window_start']
            time_window_length = swarm_rollout_params['environment']['time_window_end'] - swarm_rollout_params['environment']['time_window_start'] + 1
            rectangle = patches.Rectangle((time_window_start, 0), time_window_length, 1, linewidth=1, edgecolor=None, facecolor='lemonchiffon', alpha=0.5)
            ax.add_patch(rectangle)

            for run_dir in run_dirs:
                run_path = os.path.join(swarm_rollout_path, run_dir)
                data_flag_dir = os.path.join(run_path, best_ind, "data", f"data_env{env_id}_flag")
                
                if not os.path.exists(data_flag_dir):
                    print(f"Error in ANTS_swarm_rollout_plot_all_flags_fitnesses: {data_flag_dir} not found.")
                    exit()
                
                data_flag_files = sorted([f for f in os.listdir(data_flag_dir) if f.endswith(".csv")]) # list of data_env_flag_run_000_gen_00000_eval_0000000.csv

                for data_flag_file in data_flag_files:
                    csv_path = os.path.join(data_flag_dir, data_flag_file)
                    dataset = pd.read_csv(csv_path)

                    x = dataset['Step'].tolist()
                    y = dataset['Flags_distance'].tolist()
                    run = int(run_dir.split('_')[-1])
                    plt.plot(x, y, label=f"run {run}")


            plt.ylim(0, 1) # 0 and 1 are respectively min and max values of flag distance
            # plt.xlim(0, time_steps-1)
            plt.yticks([0, 0.5, 1.0], fontsize=18)
            plt.xticks([0, time_steps/2, time_steps-1], fontsize=18)
            plt.xlabel("Steps", fontsize=18)
            plt.ylabel("Distance", fontsize=18)

            plt.title(f"Swarm rollout of {best_ind}, {len(run_dirs)} repetitions\n{swarm_rollout_params['grid']['flag_pattern']} {swarm_rollout_params['grid']['grid_nb_rows']}x{swarm_rollout_params['grid']['grid_nb_cols']}", fontsize=18)
            plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=12, ncol=1, frameon=False)

            plt.savefig(f"{swarm_rollout_path}/plots_all_runs/flag_fitnesses_{best_ind}_env{env_id}.png")
            plt.clf()
            plt.close()


plot_all_runs_flag_fitnesses()