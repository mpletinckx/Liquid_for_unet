from config_utils import *

experiment_name = "nf_new_unet"  # number of initial filter in the decoder
parameter_name = "nf_init"
parameter_values = [4, 8, 16, 32]
arguments = ['--nf_init=$' + parameter_name,
             '--run_neptune=True',
             '--run_local=False',
             '--crop_start=60',
             '--im_size=128',
             '--im_res=128',
             '--batch_size=64',]

slurm_arguments = ['--time=48:00:00',
                   '--mem-per-cpu=262154 # 128GB',
                   '--array=1-$n_arrays%4']

generateNewExperiment(experiment_name, parameter_name, parameter_values, arguments, slurm_arguments)
print("done")