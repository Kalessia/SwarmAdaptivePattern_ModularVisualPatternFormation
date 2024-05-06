#!/bin/bash


###########################################################################
# PARAMETERS TO SET
# Set parameters values and/or comment (#) the lines for which you prefer 
# to leave the default parameters.
###########################################################################

# If you want to plot figures for a specific learning simulation, write the corresponding path in the following line
# exemple: "simulationAnalysis/flag_automata_2024-04-29_21-44-08_circle_9x9/learning"
analysis_dir="simulationAnalysis/flag_automata_2024-05-02_05-45-47_circle_65x65/learning"

# Parallelization parameters
with_parallelization_bool=True
with_parallelization_nb_free_cores=0

# Plot parameters
plot_with_animation_bool=False


###########################################################################
# LAUNCH OF THIS SCRIPT FOR ALL ALGORITHMS
# From the src directory, launch this script on the terminal: 'bash learning_launch.sh'
#
# If it occurs a 'permission denied' bug, please give permissions to access
# the analysis.py file: 'chmod u+x learning_analysis.py'
###########################################################################

# Comment this line if you DON'T want to launch a new learning simulation
output=$(python3 learning_main.py); echo "$output"| head -n -1; analysis_dir=$(echo "$output" | tail -n 1)

python3 learning_analysis.py    --analysis_dir ${analysis_dir} --with_parallelization_bool ${with_parallelization_bool} \
                                --with_parallelization_nb_free_cores ${with_parallelization_nb_free_cores} \
                                --plot_with_animation_bool ${plot_with_animation_bool}