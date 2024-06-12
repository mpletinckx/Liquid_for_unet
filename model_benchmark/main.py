import neptune
from sklearn.model_selection import train_test_split

from parse_args import *
from utils import *
from metrics import *
from model_builder import U_net

args = parse_arg()

if args.run_local:
    args.dataset_path = ('C:/Users/pletinckxm/Desktop/PhD/Codes/Liquid_for_Unet/Data/archive/BraTS2020_TrainingData'
                         '/MICCAI_BraTS2020_TrainingData/')

train_and_val_directories = [f.path for f in os.scandir(args.dataset_path) if f.is_dir()]

train_and_test_ids = pathListIntoIds(train_and_val_directories)
train_test_ids, val_ids = train_test_split(train_and_test_ids, test_size=0.2)
train_ids, test_ids = train_test_split(train_test_ids, test_size=0.15)

training_generator = DataGenerator(train_ids, args)
valid_generator = DataGenerator(val_ids, args)

input_shape = (args.im_res, args.im_res, 4)

model = U_net(args, input_shape, 4)
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy',
                       dice_coef,
                       precision,
                       sensitivity,
                       specificity,
                       dice_coef_necrotic,
                       dice_coef_edema,
                       dice_coef_enhancing],
              run_eagerly=True,
              )

if args.run_neptune:
    run = neptune.init_run(project='mpletinckx/U-net-benchmark',
                           api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjOThmMTYxMi02ZWI1LTRkOGMtOWYyMy1jOTNhMzRkZjVmODUifQ==')

    for i in range(4):
        data_sample, target = get_sample(args, (2 * i) + 1)
        make_gif(target, data_sample, i, original=True)

    run["algorithm"] = "R_UNet"

    params_model = {
        "activation": args.activation,
        "dropout": args.dropout,
        "nf_init": args.nf_init,
        "depth": args.depth,
        "normalization": args.batch_normalization,
        "flatten bottleneck": args.flatten_bottleneck,
    }

    params_opti = {
        "learning_rate": args.lr,
        "n_epochs": args.num_epochs,
        "loss function": args.loss,
        "if weighted, weigths": args.loss_weigths,
        "batch_size": args.batch_size,
    }

    run["parameters/model"] = params_model
    run["parameters/optimization"] = params_opti

    model.fit(training_generator,
              epochs=args.num_epochs,
              callbacks=[GIFCallback(args, run)],
              validation_data=valid_generator,
              )

    run.stop()

else:
    model.fit(training_generator,
              epochs=10,
              validation_data=valid_generator,
              )
