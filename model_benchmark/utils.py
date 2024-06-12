import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim


import nibabel as nib

# ml libs
import tensorflow as tf

# neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback
from neptune.types import File


def resample(data, dim):
    new_data = np.zeros((data.shape[0], data.shape[1], *dim, data.shape[-1]))
    for i in range(data.shape[1]):
        new_data[:, i, :, :, :] = tf.image.resize(data[:, i, :, :, :], dim)
    return new_data


def pathListIntoIds(dirList):
    x = []
    for i in range(0, len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/') + 1:])
    return x


class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, list_IDs, args, shuffle=True):
        """Initialization"""
        self.final_dim = (args.im_res, args.im_res)
        self.indexes = None
        self.dim = (args.im_size, args.im_size)
        self.batch_size = args.batch_size
        self.list_IDs = list_IDs
        self.n_channels = args.n_channels
        self.shuffle = shuffle
        self.vol_tot = args.vol_tot
        self.vol_start = args.vol_start
        self.crop = args.crop_start
        self.dataset_path = args.dataset_path
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
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size * self.vol_tot, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size * self.vol_tot, *self.dim))

        # Generate data
        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(self.dataset_path, i)

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
            for j in range(self.vol_tot):
                X[j + self.vol_tot * c, :, :, 0] = t_1[self.crop: self.crop + self.dim[0],
                                                   self.crop: self.crop + self.dim[0], j + self.vol_start]
                X[j + self.vol_tot * c, :, :, 1] = t_2[self.crop: self.crop + self.dim[0],
                                                   self.crop: self.crop + self.dim[0], j + self.vol_start]
                X[j + self.vol_tot * c, :, :, 2] = flair[self.crop: self.crop + self.dim[0],
                                                   self.crop: self.crop + self.dim[0], j + self.vol_start]
                X[j + self.vol_tot * c, :, :, 3] = ce[self.crop: self.crop + self.dim[0],
                                                   self.crop: self.crop + self.dim[0], j + self.vol_start]

                y[j + self.vol_tot * c, :, :] = seg[self.crop: self.crop + self.dim[0],
                                                self.crop: self.crop + self.dim[0], j + self.vol_start]

        y[y == 4] = 3
        Y = tf.one_hot(y, 4)
        if np.max(X) == 0:
            div = 1
        else:
            div = np.max(X)

        X = X / div
        X, Y = tf.image.resize(X, self.final_dim), tf.image.resize(Y, self.final_dim)

        return X, tf.one_hot(np.argmax(Y, axis=-1), 4)


def get_sample(args, ind):
    train_dataset_path = args.dataset_path
    brain_path_for_tracking_flair = train_dataset_path + f"BraTS20_Training_00{ind}/BraTS20_Training_00{ind}_flair.nii"
    brain_path_for_tracking_t1 = train_dataset_path + f"BraTS20_Training_00{ind}/BraTS20_Training_00{ind}_t1.nii"
    brain_path_for_tracking_t1ce = train_dataset_path + f"BraTS20_Training_00{ind}/BraTS20_Training_00{ind}_t1ce.nii"
    brain_path_for_tracking_t2 = train_dataset_path + f"BraTS20_Training_00{ind}/BraTS20_Training_00{ind}_t2.nii"

    seg_path_for_tracking = train_dataset_path + f"BraTS20_Training_00{ind}/BraTS20_Training_00{ind}_seg.nii"

    crop_start_at = args.crop_start
    im_size = args.im_size
    vol_tot = args.vol_tot
    vol_start = args.vol_start
    data_sample = np.zeros((1, vol_tot, im_size, im_size, 4))
    data_sample[:, :, :, :, 0] = np.moveaxis(
        nib.load(brain_path_for_tracking_t1).get_fdata()[crop_start_at: crop_start_at + im_size,
        crop_start_at: crop_start_at + im_size, vol_start:vol_start+vol_tot], -1, 0)
    data_sample[:, :, :, :, 1] = np.moveaxis(
        nib.load(brain_path_for_tracking_t2).get_fdata()[crop_start_at: crop_start_at + im_size,
        crop_start_at: crop_start_at + im_size, vol_start:vol_start+vol_tot], -1, 0)
    data_sample[:, :, :, :, 2] = np.moveaxis(
        nib.load(brain_path_for_tracking_flair).get_fdata()[crop_start_at: crop_start_at + im_size,
        crop_start_at: crop_start_at + im_size, vol_start:vol_start+vol_tot], -1, 0)
    data_sample[:, :, :, :, 3] = np.moveaxis(
        nib.load(brain_path_for_tracking_t1ce).get_fdata()[crop_start_at: crop_start_at + im_size,
        crop_start_at: crop_start_at + im_size, vol_start:vol_start+vol_tot], -1, 0)

    target = np.moveaxis(nib.load(seg_path_for_tracking).get_fdata()[crop_start_at: crop_start_at + im_size,
                         crop_start_at: crop_start_at + im_size, vol_start:vol_start+vol_tot], -1, 0)
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
    def __init__(self, args, run, base_namespace: str = "training", log_model_diagram: bool = False,
                 log_on_batch: bool = False, log_model_summary: bool = True):
        self.args = args
        self.sample = []
        for i in range(1, 5):
            self.sample.append(get_sample(self.args, (2 * i) - 1)[0])
        super().__init__(run, base_namespace, log_model_diagram, log_on_batch, log_model_summary)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            for i in range(4):
                data_sample = self.sample[i]
                gif_list = []
                for j in range(np.shape(data_sample)[1]):
                    y_pred = self.model.predict(data_sample[:, j, :, :, :])
                    y_pred = np.squeeze(np.argmax(y_pred, axis=-1), axis=0)
                    gif_list.append(tf.one_hot(y_pred, 4))

                make_gif(np.array(gif_list), data_sample, i)
                self._run["train/prediction_example"].append(File(f"prediction_{i}.gif"))
                self._run["train/ground_truth"].append(File(f"ground truth_{i}.gif"))
