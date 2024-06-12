import numpy as np
import tensorflow as tf


def lecun_tanh(x):
    return 1.7159 * tf.nn.tanh(0.666 * x)


def relu_tanh(x):
    return tf.keras.activations.tanh(tf.keras.activations.relu(x))


def get_activation(activation):
    """Helper function to get non-linearity module, choose from relu/softplus/swish/lrelu"""
    if activation == 'relu':
        return tf.nn.relu
    elif activation == 'silu':
        return tf.nn.silu
    elif activation == "tanh":
        return tf.nn.tanh
    elif activation == 'gelu':
        return tf.nn.gelu
    elif activation == 'softplus':
        return tf.nn.softplus()
    elif activation == 'lecun':
        return lecun_tanh
    else:
        raise ValueError("Unknown backbone activation")


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 nf,
                 depth,
                 activation='relu',
                 kernel_size=3,
                 dropout=0.3,
                 normalization=False):
        self.nf = nf
        self.depth = depth
        self.activation = activation
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.first_conv = []
        self.second_conv = []
        self.normalization = normalization
        super(Encoder, self).__init__()

        n_f = self.nf

        if self.normalization:
            self.norm = []
            for i in range(self.depth):
                self.first_conv.append(tf.keras.layers.Conv2D(n_f,
                                                              self.kernel_size,
                                                              activation=self.activation,
                                                              padding='same',
                                                              kernel_initializer='HeNormal'))
                self.second_conv.append(tf.keras.layers.Conv2D(n_f,
                                                               self.kernel_size,
                                                               activation=self.activation,
                                                               padding='same',
                                                               kernel_initializer='HeNormal'))
                self.norm.append(tf.keras.layers.BatchNormalization())

                n_f = n_f * 2

        else:
            for i in range(self.depth):
                self.first_conv.append(tf.keras.layers.Conv2D(n_f,
                                                              self.kernel_size,
                                                              activation=self.activation,
                                                              padding='same',
                                                              kernel_initializer='HeNormal'))
                self.second_conv.append(tf.keras.layers.Conv2D(n_f,
                                                               self.kernel_size,
                                                               activation=self.activation,
                                                               padding='same',
                                                               kernel_initializer='HeNormal'))

    def __call__(self, inputs, *args, **kwargs):
        next_layer = inputs
        skip_connection = []
        if self.normalization:
            for i in range(self.depth):
                conv = self.norm[i](next_layer)
                conv = self.first_conv[i](conv)
                conv = self.second_conv[i](conv)
                if self.dropout > 0:
                    conv = tf.keras.layers.Dropout(self.dropout)(conv)
                next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                skip_connection.append(conv)
        else:
            for i in range(self.depth):
                conv = self.first_conv[i](next_layer)
                conv = self.second_conv[i](conv)
                if self.dropout > 0:
                    conv = tf.keras.layers.Dropout(self.dropout)(conv)
                next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
                skip_connection.append(conv)

        return next_layer, skip_connection


class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 nf,
                 depth,
                 activation='relu',
                 kernel_size=3,
                 dropout=0.3,
                 normalization=False):
        self.nf = nf
        self.depth = depth
        self.activation = get_activation(activation)
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.normalization = normalization
        self.first_conv = []
        self.second_conv = []
        self.up = []
        super(Decoder, self).__init__()
        self.nf_down = 0

        n_f = self.nf
        if self.normalization:
            self.norm = []
            for i in range(self.depth):
                self.first_conv.append(tf.keras.layers.Conv2D(n_f,
                                                              self.kernel_size,
                                                              activation=self.activation,
                                                              padding='same',
                                                              kernel_initializer='HeNormal'))
                self.second_conv.append(tf.keras.layers.Conv2D(n_f,
                                                               self.kernel_size,
                                                               activation=self.activation,
                                                               padding='same',
                                                               kernel_initializer='HeNormal'))
                self.norm.append(tf.keras.layers.BatchNormalization())
                self.up.append(tf.keras.layers.Conv2DTranspose(n_f,
                                                               (3, 3),
                                                               strides=(2, 2),
                                                               padding='same'))

                n_f = n_f * 2
            self.norm.append(tf.keras.layers.BatchNormalization())
        else:
            for i in range(self.depth):
                self.first_conv.append(tf.keras.layers.Conv2D(n_f,
                                                              self.kernel_size,
                                                              activation=self.activation,
                                                              padding='same',
                                                              kernel_initializer='HeNormal'))
                self.second_conv.append(tf.keras.layers.Conv2D(n_f,
                                                               self.kernel_size,
                                                               activation=self.activation,
                                                               padding='same',
                                                               kernel_initializer='HeNormal'))
                self.up.append(tf.keras.layers.Conv2DTranspose(n_f,
                                                               (3, 3),
                                                               strides=(2, 2),
                                                               padding='same'))

                n_f = n_f * 2
        self.up.append(tf.keras.layers.Conv2DTranspose(n_f,
                                                       (3, 3),
                                                       strides=(2, 2),
                                                       padding='same'))

    def __call__(self, inputs, *args, **kwargs):
        prev_layer, skip_layer = inputs
        for i in range(self.depth):
            ind = -(i + 1)
            skip_layer_input = skip_layer.pop()
            up = self.up[ind](prev_layer)
            merge = tf.keras.layers.Concatenate()([up, skip_layer_input])
            if self.normalization:
                merge = self.norm[ind](merge)
            conv = self.first_conv[ind](merge)
            conv = self.second_conv[ind](conv)

            if self.dropout > 0:
                conv = tf.keras.layers.Dropout(self.dropout)(conv)

            prev_layer = conv

        return prev_layer


class Bottleneck(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 flatten=True,
                 activation='relu',
                 batch_norm=False,
                 backbone_dropout=0.1,
                 **kwargs):

        self._norm2 = None
        self._norm1 = None
        self._conv2 = None
        self._conv1 = None
        self._dense_layer = None

        self._units = units  # if flatten = False, units should be in format HWC, else it should be an int
        self._flatten = flatten
        self._activation = activation
        self._batch_norm = batch_norm
        self._backbone_dropout = backbone_dropout
        super(Bottleneck, self).__init__(**kwargs)

    def build(self, __input_shape):
        if self._flatten:
            self._dense_layer = tf.keras.layers.Dense(self._units)
        else:
            self._conv1 = tf.keras.layers.Conv2D(self._units[-1], 3, activation=self._activation)
            self._conv2 = tf.keras.layers.Conv2D(self._units[-1], 3, activation=self._activation)
            if self._batch_norm:
                self._norm1 = tf.keras.layers.BatchNormalization()
                self._norm2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        if self._flatten:
            outputs = self._dense_layer(inputs)
        else:
            outputs = self._norm1(inputs)
            outputs = self._conv1(outputs)
            outputs = self._norm2(outputs)
            outputs = self._conv2(outputs)

        return outputs


def U_net(args, input_shape, output_channels):
    inputs = tf.keras.Input(input_shape)
    encoded, skip_connections = Encoder(args.nf_init, args.depth)(inputs)
    shape = (
        int(args.im_res / (2 ** args.depth)), int(args.im_res / (2 ** args.depth)),
        args.nf_init * 2 ** (args.depth - 1))
    if args.flatten_bottleneck:
        inputs_bottleneck = tf.keras.layers.Flatten()(encoded)
        units = shape[0] * shape[1] * shape[2]
        outputs_bottleneck = Bottleneck(units)(inputs_bottleneck)
        outputs_bottleneck = tf.keras.layers.Reshape(shape)(outputs_bottleneck)
    else:
        outputs_bottleneck = Bottleneck(shape)(encoded)

    outputs = Decoder(shape[-1], args.depth)((outputs_bottleneck, skip_connections))
    outputs = tf.keras.layers.Conv2D(output_channels, (1, 1), activation='softmax')(outputs)
    return tf.keras.Model(inputs, outputs)
