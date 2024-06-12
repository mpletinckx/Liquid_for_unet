import os
import argparse


def parse_arg():
    parser = argparse.ArgumentParser(description='train script')

    # seed
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')

    # processing unit (default: GPU)
    parser.add_argument('--device', default='cuda', type=str, help='device for the run')
    parser.add_argument('--num_workers', default=2, type=int, help='number of data loader worker')
    parser.add_argument('--run_neptune', default=False, type=str2bool, help='does it compile the results on neptune app')
    parser.add_argument('--run_local', default=True, type=str2bool, help='boolean value that serve at choosing the '
                                                                          'correct path for dataset')

    # data setter
    parser.add_argument('--dataset_path', default='/auto/home/users/m/p/mpletin/Liquid_for_Unet/Data/archive'
                                                  '/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/',
                        type=str, help='path to the dataset')
    parser.add_argument('--crop_start', default=64, type=int, help='the index from when the image dimension begin')
    parser.add_argument('--vol_tot', default=155, type=int, help='total volume of the brain')
    parser.add_argument('--im_size', default=128, type=int, help='image height, image are square so it is also '
                                                                 'image width')
    parser.add_argument('--im_res', default=128, type=int, help='final image size after resampling')

    # model
    parser.add_argument('--nf_init', default=8, type=int, help='number of initial filters for the encoder')
    parser.add_argument('--depth', default=4, type=int, help='depth of the neural network')
    parser.add_argument('--checkpoint_file', default=None, type=str,
                        help='file with checkpoint to resume from, can be imposed')
    parser.add_argument('--checkpoint_id', default=None, type=str, help='identification string of model')
    parser.add_argument('--activation', default="Relu", type=str, help='activation function used between convolution')
    parser.add_argument('--batch_normalization', default=False, type=str2bool,
                        help='does the model uses batch_normalization ?')
    parser.add_argument('--dropout', default=0.3, type=float, help='dropout ratio used')
    parser.add_argument('--backbone_dropout', default=0.1, type=float,
                        help='dropout ratio used inside the backbone layers')
    parser.add_argument('--backbone_layers', default=1, type=int,
                        help='numbers of layers used inside the backbone block')
    parser.add_argument('--flatten_bottleneck', default=True, type=str2bool,
                        help='does the bottleneck of the model flatten his inputs ?')

    # optimization
    parser.add_argument('--loss', default='cross-entropy', type=str, choices=['cross-entropy', 'weithed cross-entropy'],
                        help='loss function')
    parser.add_argument('--loss_weigths', default=[0.1, 3, 3, 3])
    parser.add_argument('--optim', default='adam', type=str, help='optimization function', choices=['sgd', 'adam'])
    parser.add_argument('--num_epochs', default=20, type=int, help='number of epochs to train')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=5, type=int, help='batch size')
    parser.add_argument('--seq_length', default=32, type=int, help='sequence length during training')

    return parser.parse_args()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in 'true':
        return True
    elif v.lower() in 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
