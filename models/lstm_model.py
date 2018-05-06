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


class LSTMModel:

    def __init__(self, num_hid=20, clip_norm=0.25,
                 num_features=3, num_assets=9,
                 bptt=5, lr=0.001, scope_prefix=""):
        tf.reset_default_graph()
        self._num_hid = num_hid
        self._clip_norm = clip_norm
        self._num_features = num_features
        self._num_assets = num_assets
        self._bptt = bptt
        self._is_training = False
        self._lr = lr
        self._scope_prefix = scope_prefix
        self._gs = tf.train.create_global_step()
        self.tf_init = tf.global_variables_initializer
        with tf.variable_scope(self._scope_prefix+"inputs", reuse=tf.AUTO_REUSE):
            self.data = tf.placeholder(tf.float32, [None, self._bptt, self._num_features, self._num_assets])
            self.target =  tf.placeholder(tf.float32, [None, self._bptt, self._num_assets + 1])

        self.build_model()

        self.logits
        self.loss
        self.optimize
        self.predict_portfolio_allocation

        self.train_vars = tf.trainable_variables()


    def build_model(self):
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
        self._cell = tf.contrib.rnn.LSTMBlockCell(self._num_hid)
        self._asset_wt_projection = [tf.layers.Dense(64),
                                     tf.layers.Dense(64),
                                     tf.layers.Dense(self._num_assets + 1)]
    @lazy_property
    def logits(self):
        net = self.data
        shape = net.get_shape().as_list()
        with tf.variable_scope(self._scope_prefix+"LSTM_Cell", reuse=tf.AUTO_REUSE):
            # bsz X bptt X (num_feats * num_assets)
            net = tf.reshape(net, [-1, shape[1], shape[2] * shape[3]])
            net, _ = tf.nn.bidirectional_dynamic_rnn(self._cell, self._cell, net, dtype=tf.float32)
            net = tf.concat(net, axis=2)
        with tf.variable_scope(self._scope_prefix+"Asset_Projection", reuse=tf.AUTO_REUSE):
            net = tf.reshape(net, [-1, 2 * self._num_hid])
            net = self._asset_wt_projection[0](net)
            net = self._asset_wt_projection[1](net)
            net = self._asset_wt_projection[2](net)
            net = tf.reshape(net, [-1, self._bptt, self._num_assets + 1])
        return net

    @lazy_property
    def loss(self):
        with tf.variable_scope(self._scope_prefix+"loss_op", reuse=tf.AUTO_REUSE):
            optimal_action = tf.argmax(self.target, axis = 2)
            predicted_action = self.logits
            log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predicted_action,
                                                                       labels=optimal_action)
            loss = tf.reduce_mean(tf.reduce_sum(log_probs, axis = 1), axis = 0)
            return loss

    @lazy_property
    def optimize(self):
        with tf.variable_scope(self._scope_prefix+"optimize_op", reuse=tf.AUTO_REUSE):
            params = tf.trainable_variables()
            grads = tf.gradients(self.loss, params)
            grads, grad_norm = tf.clip_by_global_norm(grads, self._clip_norm)
            return self._optimizer.apply_gradients(zip(grads, params))

    @lazy_property
    def predict_portfolio_allocation(self):
        with tf.variable_scope(self._scope_prefix+"portfolio_wt_op", reuse=tf.AUTO_REUSE):
            return tf.nn.softmax(self.logits[:, -1, :])