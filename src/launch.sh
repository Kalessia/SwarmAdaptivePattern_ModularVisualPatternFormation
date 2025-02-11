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
# exemple path: "simulationAnalysis/flag_automata_2024-04-29_21-44-08_disc_9x9/learning"
learning_analysis_dir="simulationAnalysis/sliding_puzzle_incremental_2025-01-25_13-05-11_two-bands_16x16/learning"

# If you want to plot figures for a specific swarm simulation, write the corresponding path in the following line
# exemple path: "simulationAnalysis/flag_automata_2024-05-02_05-45-47_disc_65x65/swarm"
swarm_analysis_dir="simulationAnalysis/sliding_puzzle_incremental_2025-01-25_13-05-11_two-bands_16x16/swarm"


###########################################################################
# Launch learning simulation
###########################################################################

# COMMENT the following line if you DON'T want to launch a new learning simulation. NB: this line modifies the "learning_analysis_dir" parameter
# output=$(mktemp); python3 -u learning_main.py | tee ${output}; learning_analysis_dir=$(tail -n 1 ${output}); rm ${output}


###########################################################################
# Launch swarm simulation
###########################################################################

# COMMENT the following line if you DON'T want to launch a new swarm simulation. NB: this line modifies the "swarm_analysis_dir" parameter
# output=$(mktemp); python3 -u swarm_main.py --learning_analysis_dir ${learning_analysis_dir} | tee ${output}; swarm_analysis_dir=$(tail -n 1 ${output}); rm ${output}


###########################################################################
# Launch learning plots
###########################################################################

# COMMENT the following lines if you DON'T want to plot the learning data
# python3 -u learning_analysis.py --learning_analysis_dir ${learning_analysis_dir} --with_parallelization_bool ${with_parallelization_bool} \
#                                 --with_parallelization_nb_free_cores ${with_parallelization_nb_free_cores} \
#                                 --plot_with_animation_bool ${plot_with_animation_bool}


###########################################################################
# Launch swarm plots
###########################################################################

# COMMENT the following lines if you DON'T want to plot the swarm data
python3 -u swarm_analysis.py --swarm_analysis_dir ${swarm_analysis_dir} --with_parallelization_bool ${with_parallelization_bool} \
                             --with_parallelization_nb_free_cores ${with_parallelization_nb_free_cores} \
                             --plot_with_animation_bool ${plot_with_animation_bool}
