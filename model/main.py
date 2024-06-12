import neptune
from sklearn.model_selection import train_test_split

from parse_args import *
from utils import *
from metrics import *
from loss import *

# imports arguments and setting arguments for local tests
args = parse_arg()
args.run_neptune = True
args.run_local = True
args.batch_size = 5
args.num_epochs = 10

if args.run_local:
    args.dataset_path = ('C:/Users/pletinckxm/Desktop/PhD/Codes/Liquid_for_Unet/Data/archive/BraTS2020_TrainingData'
                         '/MICCAI_BraTS2020_TrainingData/')

# dataset preparation
train_and_val_directories = [f.path for f in os.scandir(args.dataset_path) if f.is_dir()]

train_and_test_ids = pathListIntoIds(train_and_val_directories)
train_test_ids, val_ids = train_test_split(train_and_test_ids, test_size=0.2)
train_ids, test_ids = train_test_split(train_test_ids, test_size=0.15)

training_generator = DataGenerator(train_ids, args)
valid_generator = DataGenerator(val_ids, args)

# model preparation
model = get_model(args)
model.compile(loss=get_loss(args),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy',
                       dice_coef,
                       precision,
                       dice_coef_necrotic,
                       dice_coef_edema,
                       dice_coef_enhancing],
              )

if args.run_neptune:
    # initiate neptune centralisation of results
    run = neptune.init_run(project='mpletinckx/LTC-R-Unet-BraTS',
                           api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjOThmMTYxMi02ZWI1LTRkOGMtOWYyMy1jOTNhMzRkZjVmODUifQ==')

    # creating a number (4) of gif to follow the behaviour of the model during training
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
        "backbone layers": args.backbone_layers,
        "backbone_dropout": args.backbone_layers,
    }

    params_opti = {
        "learning_rate": args.lr,
        "n_epochs": args.num_epochs,
        "loss function": args.loss,
        "if weighted, weigths": args.loss_weigths,
        "seq_len": args.seq_length,
        "batch_size": args.batch_size,
    }

    run["parameters/model"] = params_model
    run["parameters/optimization"] = params_opti

    # model training
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
