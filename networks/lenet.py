import typing
# ----------------------------------------
import tensorflow as tf
# --------------------------------------------------


class LeNet5(object):

    def __init__(self, *,
                 input_size: typing.Sequence[int],
                 num_classes: int):
        self._input_size = input_size
        self._num_classes = num_classes
        self._initialize_parameters()
        # ----------------------------------------
        self.inputs = tf.compat.v1.placeholder(    # tf.placeholder
            name='inputs',
            shape=(None, *self._input_size), dtype=tf.float32
        )
        self._features_sub_network()
        self._classification_sub_network()

    def _initialize_parameters(self):
        # ---------------------------------------- Layer01 (Conv + ReLU + MaxPool)
        num_channels = self._input_size[-1]
        self._weights_conv01 = tf.compat.v1.get_variable(    # tf.get_variable
            name='conv01_weights',
            shape=(5, 5, num_channels, 6), dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        self._bias_conv01 = tf.compat.v1.get_variable(    # tf.get_variable
            name='conv01_bias',
            shape=(6,), dtype=tf.float32,
            initializer=tf.constant_initializer(0.1))
        # ---------------------------------------- Layer02 (Conv + ReLU + MaxPool)
        self._weights_conv02 = tf.compat.v1.get_variable(    # tf.get_variable
            name='conv02_weights',
            shape=(5, 5, 6, 16), dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        self._bias_conv02 = tf.compat.v1.get_variable(    # tf.get_variable
            name='conv02_bias',
            shape=(16,), dtype=tf.float32,
            initializer=tf.constant_initializer(0.1))
        # ---------------------------------------- Layer03 (Dense)
        self._weights_fc01 = tf.compat.v1.get_variable(    # tf.get_variable
            name='fc01_weights',
            shape=(7 * 7 * 16, 120), dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        self._bias_fc01 = tf.compat.v1.get_variable(    # tf.get_variable
            name='fc01_bias',
            shape=(120,), dtype=tf.float32,
            initializer=tf.constant_initializer(0.1)
        )
        # ---------------------------------------- Layer04 (Dense)
        self._weights_fc02 = tf.compat.v1.get_variable(    # tf.get_variable
            name='fc02_weights',
            shape=(120, 84), dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        self._bias_fc02 = tf.compat.v1.get_variable(    # tf.get_variable
            name='fc02_bias',
            shape=(84,), dtype=tf.float32,
            initializer=tf.constant_initializer(0.1)
        )
        # ---------------------------------------- OutputLayer (Dense)
        self._weights_fc_output = tf.compat.v1.get_variable(    # tf.get_variable
            name='output_fc_weights',
            shape=(84, self._num_classes), dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        self._bias_fc_output = tf.compat.v1.get_variable(    # tf.get_variable
            name='output_fc_bias',
            shape=(self._num_classes,), dtype=tf.float32,
            initializer=tf.constant_initializer(0.1)
        )

    def _features_sub_network(self):
        # ---------------------------------------- Layer01
        outputs_conv01 = tf.nn.relu(
            tf.nn.conv2d(self.inputs, self._weights_conv01,
                         strides=(1, 1, 1, 1), padding='SAME')
            + self._bias_conv01
        )
        outputs_pool01 = tf.nn.max_pool2d(
            outputs_conv01,
            ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME'
        )
        # ---------------------------------------- Conv02
        outputs_conv02 = tf.nn.relu(
            tf.nn.conv2d(outputs_pool01, self._weights_conv02,
                         strides=(1, 1, 1, 1), padding='SAME')
            + self._bias_conv02
        )
        outputs_pool02 = tf.nn.max_pool2d(
            outputs_conv02,
            ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME'
        )
        # ----------------------------------------
        self.feature_maps = outputs_pool02

    def _classification_sub_network(self):
        flattened_feature_maps = tf.reshape(self.feature_maps, (-1, 7*7*16))
        # ---------------------------------------- Conv02
        outputs_fc01 = tf.nn.relu(
            flattened_feature_maps @ self._weights_fc01 + self._bias_fc01
        )
        outputs_fc02 = tf.nn.relu(
            outputs_fc01 @ self._weights_fc02 + self._bias_fc02
        )
        outputs = tf.nn.softmax(
            outputs_fc02 @ self._weights_fc_output + self._bias_fc_output
        )
        self.logits = outputs


if __name__ == '__main__':
    lenet5 = LeNet5(input_size=(28, 28, 1), num_classes=10)


