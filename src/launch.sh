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

# If you want to launch a swarm simulation or to plot learning figures for a specific learning simulation, write the corresponding path in the following line
# exemple path: "/home/kalessia/flagAutomata/data_plots/simulationAnalysis/sliding_puzzle_coordinates_2025-03-10_15-39-37_two-bands_8x5"
learning_analysis_dir="/home/kalessia/flagAutomata/data_plots/simulationAnalysis/sliding_puzzle_coordinates_2025-03-12_00-44-39_two-bands_16x16"


###########################################################################
# Launch learning simulation (build a flag)
# The flag "coordinates" creates a coordinate system
###########################################################################

# COMMENT the following line if you DON'T want to launch a new learning simulation. NB: this line modifies the "learning_analysis_dir" parameter
output=$(mktemp); python3 -u learning_main.py | tee ${output}; learning_analysis_dir=$(tail -n 1 ${output}); rm ${output}


###########################################################################
# Launch coordinates learning simulation (build a flag from an existing coordinate system)
###########################################################################

# COMMENT the following line if you DON'T want to launch a new learning simulation. NB: this line modifies the "learning_analysis_dir" parameter
# output=$(mktemp); python3 -u coordinates_learning_main.py --learning_analysis_dir ${learning_analysis_dir} | tee ${output}; learning_analysis_dir=$(tail -n 1 ${output}); rm ${output}


###########################################################################
# Launch swarm generalization
###########################################################################

# COMMENT the following line if you DON'T want to launch a new swarm simulation. NB: this line modifies the "swarm_analysis_dir" parameter
# output=$(mktemp); python3 -u swarm_main.py --learning_analysis_dir ${learning_analysis_dir} | tee ${output}; swarm_analysis_dir=$(tail -n 1 ${output}); rm ${output}


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
# python3 -u coordinates_learning_analysis.py --learning_analysis_dir ${learning_analysis_dir} --with_parallelization_bool ${with_parallelization_bool} \
#                                             --with_parallelization_nb_free_cores ${with_parallelization_nb_free_cores} \
#                                             --plot_with_animation_bool ${plot_with_animation_bool}


###########################################################################
# Launch swarm generalization plots
###########################################################################

# COMMENT the following lines if you DON'T want to plot the swarm data
# python3 -u swarm_analysis.py --swarm_analysis_dir ${swarm_analysis_dir} --with_parallelization_bool ${with_parallelization_bool} \
#                              --with_parallelization_nb_free_cores ${with_parallelization_nb_free_cores} \
#                              --plot_with_animation_bool ${plot_with_animation_bool}