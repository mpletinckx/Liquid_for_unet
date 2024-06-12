import tensorflow as tf


def get_loss(args):
    loss = args.loss
    if loss == 'cross-entropy':
        return tf.keras.losses.CategoricalCrossentropy()
    elif loss == 'weighted cross-entropy':
        weigths = tf.constant(args.loss_weigths)
        return Weighted_Cross_Entropy(weigths)
    else:
        raise Exception("loss should be either 'cross-entropy' or ' 'weighted cross-entropy")


class Weighted_Cross_Entropy(tf.keras.losses.CategoricalCrossentropy):
    def __init__(self, weigths):
        self.weights = weigths
        super().__init__()

    def __call__(self, y_true, y_pred):
        return super().__call__(y_true, y_pred, self.weights)
