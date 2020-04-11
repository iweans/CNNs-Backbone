import typing
# ----------------------------------------
import tensorflow as tf
# --------------------------------------------------


class AlexNet(object):

    def __init__(self, *,
                 input_size: typing.Sequence[int],
                 num_classes: int):
        self._input_size = input_size
        self._num_classes = num_classes
        self._initialize_parameters()
        # ----------------------------------------
        self.inputs = tf.placeholder(
            name='inputs',
            shape=(None, *self._input_size), dtype=tf.float32,
        )
        self._features_sub_network()
        self._classification_sub_network()

    def _initialize_parameters(self):
        with tf.variable_scope('conv01') as scope:
            self._weights_conv01 = tf.get_variable(
                name='weights',
                shape=(11, 11, 3, 64), dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1)
            )
            self._biases_conv01 = tf.get_variable(
                name='biases',
                shape=(64,), dtype=tf.float32,
                initializer=tf.constant_initializer(0.1)
            )
        # ----------------------------------------
        with tf.variable_scope('conv02') as scope:
            self._weights_conv02 = tf.get_variable(
                name='weights',
                shape=(5, 5, 64, 192), dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1)
            )
            self._biases_conv02 = tf.get_variable(
                name='biases',
                shape=(192,), dtype=tf.float32,
                initializer=tf.constant_initializer(0.1)
            )
        # ----------------------------------------
        with tf.variable_scope('conv03') as scope:
            self._weights_conv03 = tf.get_variable(
                name='weights',
                shape=(3, 3, 192, 384), dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1)
            )
            self._biases_conv03 = tf.get_variable(
                name='biases',
                shape=(384,), dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1)
            )
        # ----------------------------------------
        with tf.variable_scope('conv04') as scope:
            self._weights_conv04 = tf.get_variable(
                name='weights',
                shape=(3, 3, 384, 256), dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1)
            )
            self._biases_conv04 = tf.get_variable(
                name='biases',
                shape=(256,), dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1)
            )
        # ----------------------------------------
        with tf.variable_scope('conv05') as scope:
            self._weights_conv05 = tf.get_variable(
                name='weights',
                shape=(3, 3, 256, 256), dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1)
            )
            self._biases_conv05 = tf.get_variable(
                name='biases',
                shape=(256,), dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1)
            )
        # ----------------------------------------
        with tf.variable_scope('fc01') as scope:
            self._weights_fc01 = tf.get_variable(
                name='weights',
                shape=(6*6*256, 4096), dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1)
            )
            self._biases_fc01 = tf.get_variable(
                name='biases',
                shape=(4096,), dtype=tf.float32,
                initializer=tf.constant_initializer(0.1),
            )
        # ----------------------------------------
        with tf.variable_scope('fc02') as scope:
            self._weights_fc02 = tf.get_variable(
                name='weights',
                shape=(4096, 4096), dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1)
            )
            self._biases_fc02 = tf.get_variable(
                name='biases',
                shape=(4096,), dtype=tf.float32,
                initializer=tf.constant_initializer(0.1),
            )
        # ----------------------------------------
        with tf.variable_scope('fc03') as scope:
            self._weights_fc03 = tf.get_variable(
                name='weights',
                shape=(4096, 1000), dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1)
            )
            self._biases_fc03 = tf.get_variable(
                name='biases',
                shape=(1000,), dtype=tf.float32,
                initializer=tf.constant_initializer(0.1),
            )

    def _features_sub_network(self):
        with tf.variable_scope('conv01') as scope:
            output_conv01 = tf.nn.relu(
                tf.nn.bias_add(
                    tf.nn.conv2d(self.inputs, self._weights_conv01,
                                 strides=(1, 4, 4, 1), padding='SAME'),
                    self._biases_conv01)
            )
            output_lrn01 = tf.nn.lrn(
                output_conv01, depth_radius=4, bias=1,
                alpha=1e-3/9, beta=0.75
            )
            output_pool01 = tf.nn.max_pool2d(
                output_lrn01,
                ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID'
            )
        with tf.variable_scope('conv02') as scope:
            output_conv02 = tf.nn.relu(
                tf.nn.bias_add(
                    tf.nn.conv2d(output_pool01, self._weights_conv02,
                                 strides=(1, 1, 1, 1), padding='SAME'),
                    self._biases_conv02)
            )
            output_lrn02 = tf.nn.lrn(
                output_conv02, depth_radius=4, bias=1,
                alpha=1e-3/9, beta=0.75
            )
            output_pool02 = tf.nn.max_pool2d(
                output_lrn02,
                ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID'
            )
        with tf.variable_scope('conv03') as scope:
            output_conv03 = tf.nn.relu(
                tf.nn.bias_add(
                    tf.nn.conv2d(output_pool02, self._weights_conv03,
                                 strides=(1, 1, 1, 1), padding='SAME'),
                    self._biases_conv03)
            )
        with tf.variable_scope('conv04') as scope:
            output_conv04 = tf.nn.relu(
                tf.nn.bias_add(
                    tf.nn.conv2d(output_conv03, self._weights_conv04,
                                 strides=(1, 1, 1, 1), padding='SAME'),
                    self._biases_conv04)
            )
        with tf.variable_scope('conv05') as scope:
            output_conv05 = tf.nn.relu(
                tf.nn.bias_add(
                    tf.nn.conv2d(output_conv04, self._weights_conv05,
                                 strides=(1, 1, 1, 1), padding='SAME'),
                    self._biases_conv05)
            )
            output_pool05 = tf.nn.max_pool2d(
                output_conv05,
                ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID'
            )
        # ----------------------------------------
        self.feature_maps = output_pool05

    def _classification_sub_network(self):
        flattened_feature_maps = tf.reshape(self.feature_maps, (-1, 6*6*256))
        with tf.variable_scope('fc01') as scope:
            output_fc01 = tf.nn.relu(
                flattened_feature_maps @ self._weights_fc01 + self._biases_fc01
            )
        with tf.variable_scope('fc02') as scope:
            output_fc02 = tf.nn.relu(
                output_fc01 @ self._weights_fc02 + self._biases_fc02
            )
        with tf.variable_scope('fc03') as scope:
            output_fc03 = tf.nn.relu(
                output_fc02 @ self._weights_fc03 + self._biases_fc03
            )
        # ----------------------------------------
        self.logits = output_fc03


if __name__ == '__main__':
    alexnet = AlexNet(input_size=(224, 224, 3), num_classes=100)
