# import tensorflow as tf
# import functools
# import os
# import numpy as np
#
# def lazy_property(function):
#     attribute = '_cache_' + function.__name__
#
#     @property
#     @functools.wraps(function)
#     def decorator(self):
#         if not hasattr(self, attribute):
#             setattr(self, attribute, function(self))
#         return getattr(self, attribute)
#
#     return decorator
#
#
# class DDPGActor:
#
#     def __init__(self, num_hid=20, clip_norm=0.25,
#                  num_features=3, num_assets=9,
#                  bptt=5, lr=0.001, bsz=12):
#         self._num_hid = num_hid
#         self._clip_norm = clip_norm
#         self._num_features = num_features
#         self._num_assets = num_assets
#         self._bptt = bptt
#         self._is_training = False
#         self._bsz = bsz
#         self._lr = lr
#         with tf.name_scope("inputs"):
#             self.data = tf.placeholder(tf.float32, [None, self._bptt, self._num_features, self._num_assets])
#             self.target =  tf.placeholder(tf.float32, [None, self._bptt, self._num_assets + 1])
#         self.build_model()
#
#
#     def build_model(self):
#         self._optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
#         self._cell = tf.keras.layers.LSTM(self._num_hid, return_sequences=True, unroll=True)
#         self._asset_wt_projection = tf.keras.Sequential([
#             tf.keras.layers.Dense(64, input_shape=[self._num_hid]),
#             tf.keras.layers.Dense(64),
#             tf.keras.layers.Dense(self._num_assets)
#         ])
#
#     def logits(self):
#         net = self.data
#         shape = net.get_shape().as_list()
#         with tf.variable_scope("LSTM_Cell", reuse=tf.AUTO_REUSE):
#             # bsz X bptt X (num_feats * num_assets)
#             net = tf.reshape(net, [-1, shape[1], shape[2] * shape[3]])
#             net = self._cell(net)
#         with tf.variable_scope("DenseOutput", reuse=tf.AUTO_REUSE):
#             net = tf.reshape(net, [-1, self._num_hid])
#             net = self._asset_wt_projection(net)
#         return net
#
#     def optimize(self):
#
#
#
#
#
#
#
