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
    """
    This is the encoder class that will do what a typical U-net encoder does
    need the number of filters (nf) at the first stage of the encoding as an arguments,
    after that the number of filters are doubled at each stage of the encoding.
    the encoder also need the depth or the number of stage it goes throught.

    Those are the two mandatory arguments, others arguments can also be modified at will
    """
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
        self.second_conv = []  # we have two convolutions at each stage
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
        """
        call function to run the encoder given certain inputs
        :param inputs: should be a number of images of the same size and the size of these inputs should be > 2^^depth
        :param args: -
        :param kwargs: -
        :return: two things: first the final results of the encoding,
                             second a list of all the results at each stage in order to be usable for skip connection in a u-net
        """
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
    """
    This is the decoder class that will do what a typical U-net decoder does except for the final output/ classifier
    need the number of filters (nf) at the final stage (same as for encoder),
    the nummber of filter are the same as the encoder at each level.
    the decoder also need the depth or the number of stage it goes throught (-1 because of the bottleneck).

    Those are the two mandatory arguments, others arguments can also be modified at will
    """
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
            for i in range(self.depth - 1):
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
            for i in range(self.depth - 1):
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
        """
        call function to run the decoder for a u-net
        :param inputs: should be a tuple of first the output of the bottleneck and a list of the skip connections
        :param args: -
        :param kwargs: -
        :return: an array of nf channels of image of size of the input image size
        """
        prev_layer, skip_layer = inputs
        for i in range(self.depth-1):
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

        skip_layer_input = skip_layer.pop()
        up = self.up[0](prev_layer)
        merge = tf.keras.layers.Concatenate()([up, skip_layer_input])
        if self.normalization:
            merge = self.norm[0](merge)

        return merge


class Classifier(tf.keras.layers.AbstractRNNCell):
    """
    Liquid classifier block where the neural network f is a fully connected convolutional network
    need units which is a tuple of the final size of the image plus the number of classes
    it gives the probability of each pixel to belong to each class
    """
    def __init__(self,
                 units,
                 activation='relu',
                 epsilon=0.01,
                 **kwargs):
        self._A = None
        self._w_tau = None
        self._units = units
        self._activation = activation
        self._final_conv = tf.keras.layers.Conv2D(self._units[-1], (1, 1), activation='softmax')
        self._epsilon = epsilon
        super(Classifier, self).__init__()

    def state_size(self):
        return tf.TensorShape(self._units)

    def build(self, input_shape):
        self._w_tau = self.add_weight(
            shape=self._units,
            initializer=tf.keras.initializers.Zeros(),
            name="w_tau",
        )
        self._A = self.add_weight(
            shape=self._units,
            initializer=tf.keras.initializers.Ones(),
            name="A",
        )

        self._conv1 = tf.keras.layers.Conv2D(self._units[-1],
                                             kernel_size=3,
                                             padding='same',
                                             activation=self._activation,
                                             kernel_initializer="he_normal")
        self._conv2 = tf.keras.layers.Conv2D(self._units[-1],
                                             kernel_size=3,
                                             padding='same',
                                             activation=self._activation,
                                             kernel_initializer="he_normal")
        self.built = True

    def call(self, inputs, states):
        """
        perform the closed form solution of the liquid equation
        :param inputs: are the ouputs of the decoder
        :param states: the internal states of each pixel fro each class (parameter of the liquid equation)
        :return: For each classe, the probability for each pixel to belong to it
        """
        t = 1.0
        x = tf.keras.layers.Concatenate()([inputs, states])
        ff1 = self._conv1(x)
        ff1 = self._conv2(ff1)

        new_hidden = (-self._A
                      * tf.math.exp(-t * (tf.math.abs(self._w_tau) + tf.math.abs(ff1)))
                      * ff1
                      + self._A
                      )
        outputs = self._final_conv(new_hidden)
        return outputs, [new_hidden]


class Bottleneck(tf.keras.layers.AbstractRNNCell):
    """
    This is the class that described what is happening on the deepest stage of the u-net
    which is being described by a liquid equation,
    needs units to be initiated, units being either an int or an array of HWC format (height, width, channels)
    depending on whether it is flattened or not
    """
    def __init__(self,
                 units,
                 flatten=True,
                 activation='relu',
                 backbone_units=None,
                 backbone_layers=1,
                 backbone_dropout=0.1,
                 **kwargs):
        self._A = None
        self._w_tau = None
        self._recurrent_kernel = None
        self._kernel = None
        self._units = units  # if flatten = False, units should be in format HWC, else it should be an int
        self._flatten = flatten
        self._counter = 0
        self._activation = activation
        self._backbone_units = backbone_units
        self._backbone_layers = backbone_layers
        self._backbone_dropout = backbone_dropout
        super(Bottleneck, self).__init__(**kwargs)

    @property
    def state_size(self):
        if self._flatten:
            return self._units
        else:
            return tf.TensorShape(self._units)

    def build(self, input_shape):
        self._w_tau = self.add_weight(
            shape=self._units,
            initializer=tf.keras.initializers.Zeros(),
            name="w_tau",
        )
        self._A = self.add_weight(
            shape=self._units,
            initializer=tf.keras.initializers.Ones(),
            name="A",
        )

        input_dim = input_shape[-1] + self._units

        if self._flatten:
            if self._backbone_units is None:
                self._backbone_units = input_dim * 1.3

            backbone_layers = []
            for i in range(self._backbone_layers):
                backbone_layers.append(
                    tf.keras.layers.Dense(
                        self._backbone_units, self._activation, name=f"backbone{i}"
                    )
                )
                backbone_layers.append(tf.keras.layers.Dropout(self._backbone_dropout))

            self._backbone_fn = tf.keras.models.Sequential(backbone_layers)
            cat_shape = int(input_dim
                            if self._backbone_layers == 0
                            else self._backbone_units
                            )
            self._ff1_kernel = self.add_weight(
                shape=(cat_shape, self._units),
                initializer="glorot_uniform",
                name="ff1_weight",
            )
            self._ff1_bias = self.add_weight(
                shape=(self._units,),
                initializer="zeros",
                name="ff1_bias",
            )
        else:
            self._norm1 = tf.keras.layers.BatchNormalization()
            self._conv1 = tf.keras.layers.Conv2D(self._units[-1],
                                                 kernel_size=3,
                                                 padding='same',
                                                 activation=self._activation,
                                                 kernel_initializer="he_normal")
            self._norm2 = tf.keras.layers.BatchNormalization()
            self._conv2 = tf.keras.layers.Conv2D(self._units[-1],
                                                 kernel_size=3,
                                                 padding='same',
                                                 activation=self._activation,
                                                 kernel_initializer="he_normal")
        self.built = True

    def get_weights(self):
        weights = self.weights
        output_weights = []
        for weight in weights:
            output_weights.append(weight)
        return tf.keras.backend.batch_get_value(output_weights)

    def call(self, inputs, states):
        """
        describe the liquid equation
        :param inputs: either an array of float or an array of images
        :param states: previous states
        :return: outputs as well as the new states which in this case is the same
        """
        t = 1.0
        x = tf.keras.layers.Concatenate()([inputs, states])
        if self._flatten:
            x = self._backbone_fn(x)
            ff1 = tf.matmul(x, self._ff1_kernel) + self._ff1_bias

        else:
            ff1 = self._norm1(x)
            ff1 = self._conv1(ff1)
            ff1 = self._norm2(ff1)
            ff1 = self._conv2(ff1)

        new_hidden = (
                -self._A
                * tf.math.exp(-t * (tf.math.abs(self._w_tau) + tf.math.abs(ff1)))
                * ff1
                + self._A
        )
        return new_hidden, new_hidden
