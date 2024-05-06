#!/bin/bash


###########################################################################
# PARAMETERS TO SET
# Set parameters values and/or comment (#) the lines for which you prefer 
# to leave the default parameters.
###########################################################################

# RUN SWARM SIMULATION. In order to run the swarm simulation related to a specific learning simulation, write the corresponding learning directory path in the following line
# example: "simulationAnalysis/flag_automata_2024-05-02_05-45-47_circle_65x65/learning"
learning_analysis_dir="simulationAnalysis/flag_automata_2024-05-06_11-02-38_3_stripes_15x32/learning"

# PLOTS ONLY, FROM SWARM SIMULATION DATA. If you want to plot figures for a specific swarm simulation, write the corresponding path in the following line
# example: "simulationAnalysis/flag_automata_2024-05-02_05-45-47_circle_65x65/swarm"
swarm_analysis_dir="simulationAnalysis/flag_automata_2024-05-02_05-45-47_circle_65x65/swarm"

# Plot parameters
with_parallelization_bool=True
with_parallelization_nb_free_cores=0
plot_with_animation_bool=False


###########################################################################
# LAUNCH OF THIS SCRIPT FOR ALL ALGORITHMS
# From the src directory, launch this script on the terminal: 'bash swarm_launch.sh'
#
# If it occurs a 'permission denied' bug, please give permissions to access
# the analysis.py file: 'chmod u+x swarm_analysis.py'
###########################################################################

# Comment the following line if you DON'T want to launch a new swarm simulation
output=$(python3 swarm_main.py --learning_analysis_dir ${learning_analysis_dir}); echo "$output"| head -n -1; swarm_analysis_dir=$(echo "$output" | tail -n 1)

# python3 swarm_analysis.py   --analysis_dir ${swarm_analysis_dir} --with_parallelization_bool ${with_parallelization_bool} \
#                             --with_parallelization_nb_free_cores ${with_parallelization_nb_free_cores} \
#                             --plot_with_animation_bool ${plot_with_animation_bool}