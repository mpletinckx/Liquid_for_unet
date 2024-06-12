import tensorflow as tf
import numpy as np
from model_utils import Encoder, Bottleneck, Decoder, Classifier


def print_is_there_nan(tensor, name, after=False, before=False):
    if before:
        print(name + ' : ' + str(tf.get_static_value(tf.math.reduce_any(tf.math.is_nan(tensor)))))
    elif after:
        print(name + ' : ' + str(tf.get_static_value(tf.math.reduce_any(tf.math.is_nan(tensor)))))
    else:
        print(' ')
        print(f"There is NaN value in this part ({name}) :" + str(
            tf.get_static_value(tf.math.reduce_any(tf.math.is_nan(tensor)))))
    return


def print_nan_in_weight(tensor, name, after=False, before=False):
    weights = tensor.get_weights()
    _nan = False
    for i in range(len(weights)):
        if np.isnan(weights[i]).any():
            _nan = True
            break
    print(' ')
    print(f"There is NaN value in the weights of ({name}) :" + str(_nan))

    return


class RUnetCell(tf.keras.layers.AbstractRNNCell):
    """
    This class act as a recurrent cell that performs the complete Unet with the liquid equations
    it keeps in memory the previous states at the bottleneck and at the output
    """
    def __init__(self,
                 nf_init,
                 depth,
                 im_size,
                 n_channels,
                 activation='relu',
                 kernel_size=3,
                 dropout=0.3,
                 flatten_bottleneck=True,
                 normalization=True,
                 backbone_units=None,
                 backbone_layers=1,
                 backbone_dropout=0.1,
                 ):
        self._seq_num = 0
        self._nf_init = nf_init
        self._depth = depth
        self._im_size = im_size
        self._activation = activation
        self._kernel_size = kernel_size
        self._dropout = dropout
        self._flatten_bottleneck = flatten_bottleneck
        self._normalization = normalization
        self.n_channels = n_channels
        self._encoder = Encoder(self._nf_init,
                                self._depth,
                                self._activation,
                                self._kernel_size,
                                self._dropout,
                                self._normalization)
        self._decoder = Decoder(self._nf_init,
                                self._depth,
                                self._activation,
                                self._kernel_size,
                                self._dropout,
                                self._normalization)

        super(RUnetCell, self).__init__()

        nf_down = self._nf_init * 2 ** (self._depth - 1)
        im_size_down = self._im_size[0] / (2 ** self._depth)
        self._bottleneck_shape = (int(im_size_down), int(im_size_down), nf_down)
        if self._flatten_bottleneck:
            self._bottleneck_units = int(nf_down * im_size_down * im_size_down)
        else:
            self._bottleneck_units = (int(im_size_down), int(im_size_down), int(nf_down))

        self._classifier_units = (int(self._im_size[0]), int(self._im_size[1]), int(self.n_channels))

        self._bottleneck = Bottleneck(self._bottleneck_units,
                                      self._flatten_bottleneck)

        self._classifier = Classifier(self._classifier_units, backbone_units=backbone_units, backbone_layers=backbone_layers, backbone_dropout=backbone_dropout)
        self._counter = 0

    @property
    def state_size(self):
        return [self._bottleneck_units, tf.TensorShape(self._classifier_units)]

    def call(self, inputs, states, **kwargs):
        states_bottleneck, states_classifier = states
        new_states = []
        inputs_bottleneck, skip_connections = self._encoder(inputs)
        if self._flatten_bottleneck:
            inputs_bottleneck = tf.keras.layers.Flatten()(inputs_bottleneck)
            outputs_bottleneck, new_states_bottleneck = self._bottleneck(inputs_bottleneck, states_bottleneck)
            outputs_bottleneck = tf.keras.layers.Reshape(self._bottleneck_shape)(outputs_bottleneck)
        else:
            outputs_bottleneck, new_states_bottleneck = self._bottleneck(inputs_bottleneck, states_bottleneck)

        new_states.append(new_states_bottleneck)
        inputs_decoder = (outputs_bottleneck, skip_connections)
        inputs_classifier = self._decoder(inputs_decoder)
        outputs, new_states_classifier = self._classifier(inputs_classifier, states_classifier)
        new_states.append(new_states_classifier)
        return outputs, new_states

