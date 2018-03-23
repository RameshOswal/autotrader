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

    def __init__(
            self,
            num_hid=20,
            clip_norm=0.25,
            num_features=3,
            num_assets=9,
            bptt=50):
        tf.reset_default_graph()
        with tf.name_scope('inputs'):
            self.data = tf.placeholder(tf.float32, [None, bptt, num_features, num_assets], name="data_")
            self.target = tf.sparse_placeholder(tf.int32, name="target_")
            self._is_training = tf.placeholder(tf.bool)
        self._num_hid = num_hid
        self._clip_norm = clip_norm
        self.tf_init = tf.global_variables_initializer
        self._num_features = num_features
        self._num_assets = num_assets
        self.logits
        self.loss
        self.optimize

    def lstm_cell(self, dropout):
        return tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.LSTMCell(self._num_hid),
            output_keep_prob=tf.cond(
                self._is_training,
                lambda: tf.constant(dropout),
                lambda: tf.constant(1.0)))

    @lazy_property
    def logits(self):
        with tf.name_scope("network"):
            net = self.data
            with tf.name_scope("LSTM_Cell"):
                # Stacked LSTM cell
                net_outs = []
                for i in range(self._num_assets):
                    net_i, _ = tf.nn.dynamic_rnn(self.lstm_cell(0.5),
                                                 net[:,:,:,i], dtype=tf.float32)
                    net_outs.append(net_i[:,-1,:])
                net = tf.stack(net_outs)
                net = tf.reshape(net, [-1, 1, self._num_hid, self._num_assets])
        return net

    @lazy_property
    def loss(self):
        pass
        # ctc_logits = tf.transpose(self.logits, (1, 0, 2))
        # loss = tf.nn.ctc_loss(self.target, ctc_logits, self.seqlen)
        # loss = tf.reduce_mean(loss)
        # return loss

    @lazy_property
    def optimize(self):
        with tf.name_scope("backprop_fn"):
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, clip_norm=self._clip_norm)
            optimizer = tf.train.AdamOptimizer()
            return optimizer.apply_gradients(zip(clipped_gradients,
                                                 params))

    def store(self, sess_var, model_path):
        if model_path is not None:
            saver = tf.train.Saver(var_list=tf.trainable_variables())
            save_path = saver.save(sess_var, model_path)
            print("Model saved in path: %s" % save_path)
        else:
            print("Model path is None - Nothing to store")

    def restore(self, sess_var, model_path):
        if model_path is not None:
            if os.path.exists("{}.index".format(model_path)):
                saver = tf.train.Saver(var_list=tf.trainable_variables())
                saver.restore(sess_var, model_path)
                print("Model at %s restored" % model_path)
            else:
                print("Model path does not exist, skipping...")
        else:
            print("Model path is None - Nothing to restore")

    def __repr__(self):
        return "Seq2SeqLSTM"
