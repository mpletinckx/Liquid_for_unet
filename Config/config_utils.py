#
# Liquid neural network for brain tumor segmentation on volume
# 
# Config: generate config.txt files for experiments
#

import os

import numpy as np


def generateConfigFile(dir_name, experiment_name, parameter_name, parameter_values):
    n = len(parameter_values)
    arrays_id = np.arange(1, n + 1)

    file = open(dir_name + "/config_" + experiment_name + ".txt", "w")
    file.write("ArrayTaskID \t" + parameter_name + "\n")
    for i in range(n):
        file.write(str(arrays_id[i]) + " \t" + str(parameter_values[i]) + " \n")
    file.close()


def generateSHFile1(dir_name, experiment_name, slurm_argument):
    file = open(dir_name + "/run_" + experiment_name + ".sh", "w", newline='\n')

    file.write('#!/bin/bash' + '\n')
    file.write('## Extract SLURM information' + '\n')
    file.write('' + '\n')
    file.write('source ~/.bash_profile' + '\n')
    file.write('' + '\n')
    file.write('config=./config_' + experiment_name + '.txt' + '\n')
    file.write('n_lines=$(< "$config" wc -l)' + '\n')
    file.write('n_arrays=$(expr $n_lines - 1)' + '\n')
    file.write('' + '\n')
    file.write('sbatch ' + slurm_argument + ' run_' + experiment_name + '_slurm.sh' + '\n')

    file.close()


def generateSHFile2(dir_name, experiment_name, parameter_name, arguments, slurm_arguments):
    file = open(dir_name + "/run_" + experiment_name + "_slurm.sh", "w", newline='\n')

    file.write('#!/bin/bash' + '\n')
    file.write('## Submission script for Manneback' + '\n')
    file.write('' + '\n')
    file.write('#SBATCH --job-name=r_u_net_' + experiment_name + '\n')
    file.write('#SBATCH --ntasks=1' + '\n')
    file.write('#SBATCH --gres="gpu:TeslaV100:1"' + '\n')
    file.write('#SBATCH --partition=gpu' + '\n')
    for arg in slurm_arguments:
        file.write('#SBATCH ' + arg + '\n')
    file.write('' + '\n')
    file.write('source ~/.bash_profile' + '\n')
    file.write('' + '\n')
    file.write('config=./config_' + experiment_name + '.txt' + '\n')
    file.write(
        parameter_name + "=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)" + '\n')
    file.write('file=experiment_$' + parameter_name + '.out' + '\n')
    file.write('' + '\n')
    file.write('echo $file' + '\n')
    file.write('echo "" >> $file' + '\n')
    file.write(
        'echo -e "Files:\\t run_' + experiment_name + '.sh, run_' + experiment_name + '_slurm.sh, main.py" >> $file' + '\n')
    file.write('echo -e "Node:\\t" $SLURMD_NODENAME >> $file' + '\n')
    file.write('echo -e "Job ID:\\t" $SLURM_JOB_ID >> $file' + '\n')
    file.write('echo "" >> $file' + '\n')
    file.write('' + '\n')
    file.write(
        'srun --output=$file --open-mode=append python /auto/home/users/m/p/mpletin/Liquid_for_Unet/model/main.py ')
    for arg in arguments:
        file.write(arg + ' ')
    file.write('\n')

    file.close()


def generateNewExperiment(experiment_name, parameter_name, parameter_values, arguments, slurm_arguments):
    dir_name = '../experiment_' + experiment_name
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        print("created")

    generateConfigFile(dir_name, experiment_name, parameter_name, parameter_values)
    generateSHFile1(dir_name, experiment_name, slurm_arguments[-1])
    generateSHFile2(dir_name, experiment_name, parameter_name, arguments, slurm_arguments[:-1])
