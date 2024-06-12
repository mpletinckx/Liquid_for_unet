from config_utils import *

experiment_name = "benchmark"  # number of initial filter in the decoder
parameter_name = "None"
parameter_values = []
arguments = ['run_neptune=True',
             'run_local=False']
slurm_arguments = ['--time=48:00:00',
                   '--mem-per-cpu=131072 # 128GB',
                   '--array=1-$n_arrays%4']

generateNewExperiment(experiment_name, parameter_name, parameter_values, arguments, slurm_arguments)
print("done")
