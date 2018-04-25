import tensorflow as tf
import functools
import os
import numpy as np

"""
References for LSTM Model:
https://github.com/ZhengyaoJiang/PGPortfolio/blob/a9c148cc93e7d4654ff8bc241c432f208244d2f9/pgportfolio/learn/network.py#L125
https://github.com/ZhengyaoJiang/PGPortfolio/blob/a9c148cc93e7d4654ff8bc241c432f208244d2f9/pgportfolio/learn/nnagent.py#L12
https://github.com/ZhengyaoJiang/PGPortfolio/blob/master/pgportfolio/net_config.json
Architecture Description:
Input: [Bsz, feats, NUM_ASSETS, History]
Input -> Transpose([Bsz, History, feats, NUM_ASSETS]) -> Feed each asset to different LSTM(20 units)
-> Get output as list: len #assets [bsz, 1, 20] -> Concatenate/Stack output -> Reshape -> ??
"""

# def lazy_property(function):
#     attribute = '_cache_' + function.__name__
#
#     @property
#     @functools.wraps(function)
#     def decorator(self):
#         if not hasattr(self, attribute):
#             setattr(self, attribute, function(self))
#         return getattr(self, attribute)
#     return decorator


class LSTMModel:

    def __init__(
            self,
            num_hid=20,
            clip_norm=0.25,
            num_features=3,
            num_assets=9,
            bptt=5):
        self._num_hid = num_hid
        self._clip_norm = clip_norm
        self._num_features = num_features
        self._num_assets = num_assets
        self._bptt = bptt
        self._is_training = False
        self.optimizer = tf.train.AdamOptimizer()

    def lstm_cell(self, dropout, is_training):
        return tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.LSTMCell(self._num_hid),
            output_keep_prob=(dropout if is_training else 1.0))

    def logits(self, data, is_training=False):
        net = tf.constant(data, dtype=tf.float32)
        shape = net.get_shape().as_list()
        net = tf.reshape(net, [-1, shape[1], shape[2]*shape[3]])
        with tf.name_scope("LSTM_Cell"):
            # Stacked LSTM cell
            net, _ = tf.nn.dynamic_rnn(self.lstm_cell(0.5, is_training),
                                       net, dtype=tf.float32)
        with tf.name_scope("Output_dense"):
            net = tf.reshape(net, [-1, self._num_hid])
            net = tf.layers.dense(net, self._num_assets + 1)
            net = tf.reshape(net, [-1, self._bptt, self._num_assets + 1])
        return net

    def predict_portfolio_allocation(self, data):
        last_logit = self.logits(data)[:, -1, :]
        return tf.nn.softmax(last_logit).numpy()


    def loss(self, data, target, is_training=False):
        prediction = tf.nn.softmax(self.logits(data, is_training),dim=2)
        portfolio_ts = tf.multiply(prediction, target)
        portfolio_comb = tf.reduce_sum(portfolio_ts, axis=2)
        apv_batch = tf.reduce_prod(portfolio_comb,axis=1)
        apv_mean = tf.reduce_mean(apv_batch)
        return -apv_mean

    def optimize(self, data, target):
        with tf.name_scope("backprop_fn"):
            # params = tf.trainable_variables()
            # gradients = tf.gradients(self.loss(data, target), params)
            # clipped_gradients, _ = tf.clip_by_global_norm(
            #     gradients, clip_norm=self._clip_norm)
            self.optimizer.minimize(lambda :self.loss(data, target, True))

    def __repr__(self):
        return "Seq2SeqLSTM"
