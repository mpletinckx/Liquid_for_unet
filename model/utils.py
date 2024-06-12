import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# neural imaging
import nibabel as nib

# ml libs
import tensorflow as tf

from r_unet_cell import RUnetCell

# neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback
from neptune.types import File


def get_sample(args, ind):
    """
    load a sample of the data
    :param args: args is a parser with the appropriate parameter for the loading
    :param ind: the indice of the sample
    :return: an array of the four input channels and the targeted segmentation
    """
    train_dataset_path = args.dataset_path
    brain_path_for_tracking_flair = train_dataset_path + f"BraTS20_Training_00{ind}/BraTS20_Training_00{ind}_flair.nii"
    brain_path_for_tracking_t1 = train_dataset_path + f"BraTS20_Training_00{ind}/BraTS20_Training_00{ind}_t1.nii"
    brain_path_for_tracking_t1ce = train_dataset_path + f"BraTS20_Training_00{ind}/BraTS20_Training_00{ind}_t1ce.nii"
    brain_path_for_tracking_t2 = train_dataset_path + f"BraTS20_Training_00{ind}/BraTS20_Training_00{ind}_t2.nii"

    seg_path_for_tracking = train_dataset_path + f"BraTS20_Training_00{ind}/BraTS20_Training_00{ind}_seg.nii"

    crop_start_at = args.crop_start
    im_size = args.im_size
    vol_tot = args.vol_tot
    data_sample = np.zeros((1, vol_tot, im_size, im_size, 4))
    data_sample[:, :, :, :, 0] = np.moveaxis(
        nib.load(brain_path_for_tracking_t1).get_fdata()[crop_start_at: crop_start_at + im_size,
        crop_start_at: crop_start_at + im_size, :], -1, 0)
    data_sample[:, :, :, :, 1] = np.moveaxis(
        nib.load(brain_path_for_tracking_t2).get_fdata()[crop_start_at: crop_start_at + im_size,
        crop_start_at: crop_start_at + im_size, :], -1, 0)
    data_sample[:, :, :, :, 2] = np.moveaxis(
        nib.load(brain_path_for_tracking_flair).get_fdata()[crop_start_at: crop_start_at + im_size,
        crop_start_at: crop_start_at + im_size, :], -1, 0)
    data_sample[:, :, :, :, 3] = np.moveaxis(
        nib.load(brain_path_for_tracking_t1ce).get_fdata()[crop_start_at: crop_start_at + im_size,
        crop_start_at: crop_start_at + im_size, :], -1, 0)

    target = np.moveaxis(nib.load(seg_path_for_tracking).get_fdata()[crop_start_at: crop_start_at + im_size,
                         crop_start_at: crop_start_at + im_size, :], -1, 0)
    target[target == 4] = 3
    target = tf.expand_dims(tf.one_hot(target, 4), 0)

    if np.max(data_sample) == 0:
        div = 1
    else:
        div = np.max(data_sample)

    data_sample = data_sample / div

    data_sample, target = (resample(data_sample, (args.im_res, args.im_res)),
                           resample(target, (args.im_res, args.im_res)))

    return data_sample, tf.squeeze(tf.one_hot(np.argmax(target, axis=-1), 4), axis=0)


def make_gif(frames_pred, data_sample, ind, original=False):
    """
    function that creates a gif and save it
    :param frames_pred: segmentation volume
    :param data_sample: the four input channels
    :param ind: the indice of the sample (in order to differentiate during saving
    :param original: boolean to see if it is the ground truth or a prediction
    :return: /
    """
    sample_data_gif = ImageToGIF()
    if original:
        filename = f"ground truth_{ind}.gif"

    else:
        filename = f"prediction_{ind}.gif"
    for i in range(frames_pred.shape[0]):
        image = np.rot90(data_sample[0, i, :, :, 2])
        mask = np.clip(np.rot90(frames_pred[i]), 0, 1)
        sample_data_gif.add(image, mask)

    sample_data_gif.save(filename, fps=10)


def resample(data, dim):
    """
    same as resize
    """
    new_data = np.zeros((data.shape[0], data.shape[1], *dim, data.shape[-1]))
    for i in range(data.shape[1]):
        new_data[:, i, :, :, :] = tf.image.resize(data[:, i, :, :, :], dim)
    return new_data


class ImageToGIF:
    """Create GIF without saving image files."""

    def __init__(self,
                 size=(600, 400),
                 xy_text=(80, 10),
                 dpi=100,
                 cmap=None):
        if cmap is None:
            cmap = ['gray', 'Paired', 'Blues', 'Greens', 'Oranges']
        self.fig = plt.figure()
        self.fig.set_size_inches(size[0] / dpi, size[1] / dpi)
        self.xy_text = xy_text
        self.cmap = cmap

        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.images = []

    def add(self, image, mask, with_mask=True):
        to_plot = [self.ax.imshow(image, animated=True, cmap=self.cmap[0])]
        if with_mask:
            for i in range(3):
                _mask = mask[:, :, i + 1]
                to_plot.append(self.ax.imshow(np.ma.masked_where(_mask == 0, _mask),
                                              alpha=0.8,
                                              animated=True,
                                              cmap=self.cmap[i + 1], vmin=0, vmax=1))
        self.images.append(to_plot)

    def save(self, filename, fps):
        animation = anim.ArtistAnimation(self.fig, self.images)
        animation.save(filename, fps=fps)


class GIFCallback(NeptuneCallback):
    """
    neptune callbacks in order to centralise all the validation result to the neptune app and during which we can see
    the evolution of training through gif also
    """
    def __init__(self, args, run, base_namespace: str = "training", log_model_diagram: bool = False,
                 log_on_batch: bool = False, log_model_summary: bool = True):
        self.args = args
        self.sample = []
        for i in range(1, 5):
            self.sample.append(get_sample(self.args, (2 * i)-1)[0])
        super().__init__(run, base_namespace, log_model_diagram, log_on_batch, log_model_summary)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            for i in range(4):
                data_sample = self.sample[i]
                y_pred = self.model.predict(data_sample)
                y_pred = np.squeeze(np.argmax(y_pred, axis=-1), axis=0)
                y_pred = tf.one_hot(y_pred, 4)
                make_gif(y_pred, data_sample, i)
                self._run["train/prediction_example"].append(File(f"prediction_{i}.gif"))
                self._run["train/ground_truth"].append(File(f"ground truth_{i}.gif"))


def get_model(args):
    cell = RUnetCell(args.nf_init, args.depth, (args.im_res, args.im_res), 4)
    inputs = tf.keras.Input((None, args.im_res, args.im_res, 4))
    r_unet = tf.keras.layers.RNN(cell, return_sequences=True)(inputs)
    my_model = tf.keras.Model(inputs, r_unet)
    return my_model


def pathListIntoIds(dirList):
    x = []
    for i in range(0, len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/') + 1:])
    return x


class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, list_IDs, args, n_channels=4, shuffle=True):
        """Initialization"""
        self.indexes = None
        self.final_dim = (args.im_res, args.im_res)
        self.dim = (args.im_size, args.im_size)
        self.batch_size = args.batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.seq_len = args.seq_length
        self.path = args.dataset_path
        self.crop = args.crop_start
        self.volume = args.vol_tot
        self.vol_at = (self.volume % self.seq_len) // 2
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        Batch_ids = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(Batch_ids)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        """Generates data containing batch_size samples"""  # X : (n_samples, seq_len ( or volume), *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size * (self.volume // self.seq_len), self.seq_len, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size * (self.volume // self.seq_len), self.seq_len, *self.dim))
        # Generate data
        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(self.path, i)

            data_path = os.path.join(case_path, f'{i}_t1.nii')
            t_1 = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, f'{i}_t2.nii')
            t_2 = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, f'{i}_flair.nii')
            flair = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, f'{i}_t1ce.nii')
            ce = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, f'{i}_seg.nii')
            seg = nib.load(data_path).get_fdata()
            for j in range(self.volume // self.seq_len):
                X[c * j + j, :, :, :, 0] = np.moveaxis(
                    t_1[self.crop: self.crop + self.dim[0], self.crop: self.crop + self.dim[0],
                    self.vol_at + (self.seq_len * j): self.vol_at + self.seq_len * (j + 1)], -1, 0)
                X[c * j + j, :, :, :, 1] = np.moveaxis(
                    t_2[self.crop: self.crop + self.dim[0], self.crop: self.crop + self.dim[0],
                    self.vol_at + (self.seq_len * j): self.vol_at + self.seq_len * (j + 1)], -1, 0)
                X[c * j + j, :, :, :, 2] = np.moveaxis(
                    flair[self.crop: self.crop + self.dim[0], self.crop: self.crop + self.dim[0],
                    self.vol_at + (self.seq_len * j): self.vol_at + self.seq_len * (j + 1)], -1, 0)
                X[c * j + j, :, :, :, 3] = np.moveaxis(
                    ce[self.crop: self.crop + self.dim[0], self.crop: self.crop + self.dim[0],
                    self.vol_at + (self.seq_len * j): self.vol_at + self.seq_len * (j + 1)], -1, 0)
                y[c * j + j, :, :, :] = np.moveaxis(
                    seg[self.crop: self.crop + self.dim[0], self.crop: self.crop + self.dim[0],
                    self.vol_at + (self.seq_len * j): self.vol_at + self.seq_len * (j + 1)], -1, 0)

        # Generate masks
        y[y == 4] = 3
        Y = tf.one_hot(y, 4)
        if np.max(X) == 0:
            div = 1
        else:
            div = np.max(X)

        X = X / div

        X, Y = resample(X, self.final_dim), resample(Y, self.final_dim)
        return X, tf.one_hot(np.argmax(Y, axis=-1), 4)
