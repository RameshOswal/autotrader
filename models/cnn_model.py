import tensorflow as tf
import functools
import os
import numpy as np

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class CNNModel:

    def __init__(self, num_hid=20, clip_norm=0.25,
                 num_features=3, num_assets=9,
                 bptt=5, lr=0.001, num_dense_units=64):
        tf.reset_default_graph()
        self._num_hid = num_hid
        self._clip_norm = clip_norm
        self._num_features = num_features
        self._num_assets = num_assets
        self._bptt = bptt
        self._lr = lr
        self._num_dense_units = num_dense_units
        self.tf_init = tf.global_variables_initializer
        with tf.variable_scope("inputs"):
            self.data = tf.placeholder(tf.float32, [None, self._bptt, self._num_features, self._num_assets])
            self.target =  tf.placeholder(tf.float32, [None, self._bptt, self._num_assets + 1])
            self._is_training = tf.placeholder(tf.bool, None)
        self.n_channels = [20, 20]
        self.kernel_sizes = [8, 8]
        self.strides = [1, 1]
        self.__logits_op
        self.__loss_op
        self.__optimize_op
        self.__predict_portfolio_op

    @lazy_property
    def __logits_op(self):
        net = self.data
        shape = net.get_shape().as_list()
        with tf.variable_scope("ConvLayers"):
            # bsz X bptt X (num_feats * num_assets)
            net = tf.reshape(net, [-1, shape[1], shape[2] * shape[3]])
            net = tf.expand_dims(net, axis = 1)
            net = tf.layers.conv2d(net, filters=self.n_channels[0], strides=(1, self.strides[0]),
                                   kernel_size=(1, self.kernel_sizes[0]), padding="SAME")
            net = self.batch_norm(net, self.n_channels[0], self._is_training)
            net = tf.layers.conv2d(net, filters=self.n_channels[1], strides=(1, self.strides[1]),
                                   kernel_size=(1, self.kernel_sizes[1]), padding="SAME")
            net = self.batch_norm(net, self.n_channels[1], self._is_training)
            net = tf.squeeze(net, axis=1)

        with tf.variable_scope("Asset_Projection"):
            net = tf.reshape(net, [-1, self.n_channels[-1]])
            net = tf.layers.dense(net, self._num_dense_units)
            net = tf.layers.dense(net, self._num_dense_units)
            net = tf.layers.dense(net, self._num_assets + 1)
            net = tf.reshape(net, [-1, self._bptt, self._num_assets + 1])
        return net

    @lazy_property
    def __loss_op(self):
        with tf.variable_scope("loss_op"):
            optimal_action = tf.argmax(self.target, axis = 2)
            predicted_action = self.__logits_op
            log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predicted_action,
                                                                       labels=optimal_action)
            loss = tf.reduce_mean(tf.reduce_sum(log_probs, axis = 1))
            return loss

    @lazy_property
    def __optimize_op(self):
        with tf.variable_scope("optimize_op"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
            params = tf.trainable_variables()
            grads = tf.gradients(self.__loss_op, params)
            grads, grad_norm = tf.clip_by_global_norm(grads, self._clip_norm)
            return optimizer.apply_gradients(zip(grads, params))

    @lazy_property
    def __predict_portfolio_op(self):
        with tf.variable_scope("portfolio_wt_op"):
            return tf.nn.softmax(self.__logits_op[:, -1, :])

    @staticmethod
    def batch_norm(x, n_out, phase_train, scope='bn'):
        """
        Batch normalization on convolutional maps.
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        with tf.variable_scope(scope):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                               name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

    def optimize(self, sess, market_state, optimal_action):
        loss, _ = sess.run([self.__loss_op, self.__optimize_op], {
            self.data: market_state,
            self.target: optimal_action,
            self._is_training: True
        })
        return loss

    def predict_allocation(self, sess, market_state):
        return sess.run(self.__predict_portfolio_op, {
            self.data: market_state,
            self._is_training: False
        })