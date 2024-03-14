#!/bin/bash


###########################################################################
# PARAMETERS TO SET
# Set parameters values and/or comment (#) on the lines for which you prefer 
# to leave the default parameters.
###########################################################################

analysis_dir="simulationAnalysis/flag_automata_simulation_2024-03-14_16-53-48" # relative path, ex: "simulationAnalysis/flag_automata_simulation_2024-03-14_07-32-14"


###########################################################################
# LAUNCH OF THIS SCRIPT FOR ALL ALGORITHMS (DO NOT MODIFY)
# From the src directory, launch this script on the terminal: 'bash launch_plots.sh'
#
# If it occurs a 'permission denied' bug, please give permissions to access
# the analysis.py file: 'chmod u+x analysis.py'
###########################################################################

python3 src/analysis.py ${analysis_dir}