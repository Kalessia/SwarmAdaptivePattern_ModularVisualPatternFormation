#!/bin/bash

# LAUNCH OF THIS SCRIPT FOR ALL ALGORITHMS
# From the src directory, launch this script on the terminal: 'bash launch.sh'
#
# If it occurs a 'permission denied' bug, please give permissions to access
# the file_name.py file: 'chmod u+x file_name.py'


###########################################################################
# Learning and swarm plot parameters
###########################################################################

# Plot parameters, used in learning_analysis.py and swarm_analysis.py. NB: the parallelization parameters for simulations are in the learning_params.json and swarm_params.json files
with_parallelization_bool=True
with_parallelization_nb_free_cores=0
plot_with_animation_bool=False

# Specify a precise learning path to launch a coordinates_learning from an existing coordinate system, or to plot learning figures for a specific learning simulation
# Used in learning_analysis.py and coordinates_learning_main.py only
# example path: "/home/kalessia/flagAutomata/data_plots/simulationAnalysis/sliding_puzzle_coordinates_2025-03-10_15-39-37_two-bands_8x5"
# example path on cluster: "/scratch/sliding_puzzle_coordinates_2025-10-15_00-35-40_bn-SU_12x12"
learning_analysis_dir="/home/loi/Documents/flagAutomata/data_plots/simulationAnalysis/sliding_puzzle_coordinates_2025-11-12_03-13-01_rgb-rainbow-full_1x4"

# Specify a precise coordinate_learning path to plot coordinate_learning figures for a specific coordinate_learning simulation or to launch a related swarm_rollout simulation
# Used in coordinates_learning_analysis.py only
# example path: "/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_models-A-C-D-GECCO/sliding_puzzle_coordinates_2025-05-22_14-07-36_centered-half-discs_16x16/learning_coordinates_bn-smile_16x16_modelA_4-[]-3_2-[3]-1_2025-10-21_02-07-40"
coordinates_learning_analysis_dir=""

# Specify a swarm_rollout path to plot swarm_rollout figures
# Used in swarm_rollout_analysis.py only
# example path: "/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_models-A-C-D-GECCO/sliding_puzzle_coordinates_2025-05-22_14-07-36_centered-half-discs_16x16/learning_coordinates_bn-smile_16x16_modelC_4-[]-3_6-[5,5]-1_2025-10-21_01-34-38_swarm_rollout_2025-10-27_04-33-54"
swarm_rollout_analysis_dir=""


###########################################################################
# Launch learning simulation (build a flag)
# The flag "coordinates" creates a coordinate system
###########################################################################

# COMMENT the following line if you DON'T want to launch a new learning simulation. NB: this line modifies the "learning_analysis_dir" parameter
# output=$(mktemp); python3 -u learning_main.py | tee ${output}; learning_analysis_dir=$(tail -n 1 ${output}); rm ${output}


###########################################################################
# Launch coordinates learning simulation (build a flag from an existing coordinate system)
###########################################################################

# COMMENT the following line if you DON'T want to launch a new learning simulation. NB: this line modifies the "learning_analysis_dir" parameter
output=$(mktemp); python3 -u coordinates_learning_main.py --learning_analysis_dir ${learning_analysis_dir} | tee ${output}; coordinates_learning_analysis_dir=$(tail -n 1 ${output}); rm ${output}


###########################################################################
# Launch swarm rollout
###########################################################################

# COMMENT the following lines if you DON'T want to launch a new swarm rollout simulation. NB: this line modifies the "swarm_rollout_analysis_dir" parameter
output=$(mktemp); python3 -u swarm_rollout_main.py | tee ${output}; swarm_rollout_analysis_dir=$(tail -n 1 ${output}); rm ${output}


###########################################################################
# Launch learning plots
###########################################################################

# COMMENT the following lines if you DON'T want to plot the learning data (flag or gradient)
# python3 -u learning_analysis.py --learning_analysis_dir ${learning_analysis_dir} --with_parallelization_bool ${with_parallelization_bool} \
#                                 --with_parallelization_nb_free_cores ${with_parallelization_nb_free_cores} \
#                                 --plot_with_animation_bool ${plot_with_animation_bool}


###########################################################################
# Launch coordinates learning plots
###########################################################################

# # COMMENT the following lines if you DON'T want to plot the learning data (map_xy_flag)
python3 -u coordinates_learning_analysis.py --coordinates_learning_analysis_dir ${coordinates_learning_analysis_dir} --with_parallelization_bool ${with_parallelization_bool} \
                                            --with_parallelization_nb_free_cores ${with_parallelization_nb_free_cores} \
                                            --plot_with_animation_bool ${plot_with_animation_bool}


###########################################################################
# Launch swarm rollout plots
###########################################################################

# COMMENT the following lines if you DON'T want to plot the swarm rollout data
# python3 -u swarm_rollout_analysis.py --swarm_rollout_analysis_dir ${swarm_rollout_analysis_dir} --with_parallelization_bool ${with_parallelization_bool} \
#                                      --with_parallelization_nb_free_cores ${with_parallelization_nb_free_cores} \
#                                      --plot_with_animation_bool ${plot_with_animation_bool}