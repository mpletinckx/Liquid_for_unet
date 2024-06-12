import tensorflow as tf
from keras.src.engine import data_adapter


def print_info(a, name):
    print(' ')
    print(f"There is NaN value in this part ({name}) :" + str(
        tf.get_static_value(tf.math.reduce_any(tf.math.is_nan(a)))))
    return


class Model_debug(tf.keras.Model):

    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, y_pred, sample_weight)
