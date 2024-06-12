from config_utils import *

experiment_name = "seq_len"  # length of the sequence during training
parameter_name = "seq_length"
parameter_values = [1, 4, 16, 32, 64, 155]
arguments = ['--seq_length=$' + parameter_name,
             '--nf_init=8',
             '--run_neptune=True',
             '--run_local=False',
             '--crop_start=60',
             '--im_size=128',
             '--im_res=128',
             '--batch_size=64',
             '--num_epochs=1000']

slurm_arguments = ['--time=48:00:00',
                   '--mem-per-cpu=262154 # 128GB',
                   '--array=1-$n_arrays%4']

generateNewExperiment(experiment_name, parameter_name, parameter_values, arguments, slurm_arguments)
print("done")
