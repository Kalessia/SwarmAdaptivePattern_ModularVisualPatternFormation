#!/bin/bash


# LAUNCH OF THIS SCRIPT FOR ALL ALGORITHMS
# From the src directory, launch this script on the terminal: 'bash launch.sh'
#
# If it occurs a 'permission denied' bug, please give permissions to access
# the file_name.py file: 'chmod u+x file_name.py'


###########################################################################
# Learning plots parameters
###########################################################################

# If you want to plot figures for a specific learning simulation, write the corresponding path in the following line
# exemple: "simulationAnalysis/flag_automata_2024-04-29_21-44-08_circle_9x9/learning"
learning_analysis_dir="simulationAnalysis/flag_automata_2024-05-24_20-15-47_circle_5x5/learning"

# Plot parameters
with_parallelization_bool=True
with_parallelization_nb_free_cores=0
plot_with_animation_bool=False


###########################################################################
# Launch learning simulation
###########################################################################

# Comment this line if you DON'T want to launch a new learning simulation
output=$(python3 learning_main.py); echo "$output"| head -n -1; learning_analysis_dir=$(echo "$output" | tail -n 1)


###########################################################################
# Launch learning plots
###########################################################################

python3 learning_analysis.py    --learning_analysis_dir ${learning_analysis_dir} --with_parallelization_bool ${with_parallelization_bool} \
                                --with_parallelization_nb_free_cores ${with_parallelization_nb_free_cores} \
                                --plot_with_animation_bool ${plot_with_animation_bool}


###########################################################################
# Swarm plot parameters
###########################################################################

# If you want to plot figures for a specific swarm simulation, write the corresponding path in the following line
# example: "simulationAnalysis/flag_automata_2024-05-02_05-45-47_circle_65x65/swarm"
swarm_analysis_dir="simulationAnalysis/flag_automata_2024-05-22_17-43-26_circle_11x11/swarm"

# Plot parameters
with_parallelization_bool=True
with_parallelization_nb_free_cores=0
plot_with_animation_bool=False


###########################################################################
# Launch swarm simulation
###########################################################################

# Comment the following line if you DON'T want to launch a new swarm simulation
# python3 swarm_main.py --learning_analysis_dir ${learning_analysis_dir} > simu.txt
# python3 swarm_main.py --learning_analysis_dir ${learning_analysis_dir}
output=$(python3 swarm_main.py --learning_analysis_dir ${learning_analysis_dir}); echo "$output"| head -n -1; swarm_analysis_dir=$(echo "$output" | tail -n 1)


###########################################################################
# Launch swarm plots
###########################################################################

python3 swarm_analysis.py    --swarm_analysis_dir ${swarm_analysis_dir} --with_parallelization_bool ${with_parallelization_bool} \
                             --with_parallelization_nb_free_cores ${with_parallelization_nb_free_cores} \
                             --plot_with_animation_bool ${plot_with_animation_bool}
