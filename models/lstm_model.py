import tensorflow as tf

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

def lstm(x, prev_c, prev_h, w_lstm):
    ifog = tf.matmul(tf.concat([x, prev_h], axis=1), w_lstm)
    i, f, o, g = tf.split(ifog, 4, axis=1)
    i, f, o, g = tf.sigmoid(i), tf.sigmoid(f), tf.sigmoid(o), tf.tanh(g)
    next_c = f * prev_c + i * g
    next_h = o * tf.tanh(next_c)
    return next_c, next_h


class LSTMModel:

    def __init__(
            self,
            num_hid=20,
            clip_norm=5.0,
            num_features=3,
            num_assets=9,
            bptt=5,
            lr=0.001,
            bsz = 12):
        self._num_hid = num_hid
        self._clip_norm = clip_norm
        self._num_features = num_features
        self._num_assets = num_assets
        self._bptt = bptt
        self._is_training = False
        self._optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self._asset_wt_projection = tf.layers.Dense(self._num_assets + 1, activation=None)
        self.prev_h = tf.get_variable("prev_h", shape=[1, self._num_hid],
                                  initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        self.prev_c = tf.get_variable("prev_c", shape=[1, self._num_hid],
                                      initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        self.w_lstm = tf.get_variable("w_lstm", shape=[(num_assets * num_features) + self._num_hid, 4 * self._num_hid],
                                      initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        # self._cell = tf.keras.layers.LSTM(self._num_hid, activation=None, return_sequences=True,recurrent_activation=tf.nn.leaky_relu)

    def logits(self, data, is_training=False):
        net = tf.constant(data, dtype=tf.float32)
        shape = net.get_shape().as_list()
        net = tf.reshape(net, [-1, shape[1], shape[2]*shape[3]])
        # print(net.shape)
        # exit()
        with tf.variable_scope("LSTM_Cell", reuse=tf.AUTO_REUSE):
            hidden_states = tf.TensorArray(dtype=tf.float32, size=self._bptt, dynamic_size=True)
            def condition(i, *args): return tf.less(i, self._bptt)
            def body(i, prev_h, prev_c, hidden_states):
                x = net[:, i, :]
                next_c, next_h = lstm(x, prev_c, prev_h, self.w_lstm)
                hidden_states = hidden_states.write(i, next_h)
                return i + 1, next_h, next_c, hidden_states
            tiled_h, tiled_c = tf.tile(self.prev_h, [shape[0], 1]), tf.tile(self.prev_c, [shape[0], 1])
            loop_vars = [tf.constant(0, dtype=tf.int32), tiled_h, tiled_c, hidden_states]
            loop_out = tf.while_loop(condition, body, loop_vars)
            hs = loop_out[-1].stack()
            net = tf.transpose(hs, [1, 0, 2])


        with tf.name_scope("Output_dense"):
            net = tf.reshape(net, [-1, self._num_hid])
            net = self._asset_wt_projection(net)
            net = tf.reshape(net, [-1, self._bptt, self._num_assets + 1])
        return net

    # EIIE Logic
    # def logits(self, data, is_training=False):
    #     net = tf.constant(data, dtype=tf.float32)
    #     with tf.variable_scope("LSTM_Cell", reuse=tf.AUTO_REUSE):
    #         # (bsz, bptt, num_feats, num_assets)
    #         shape = net.get_shape().as_list()
    #         bsz = shape[0]
    #         bptt = shape[1]
    #         lstm_outs = []
    #         for i in range(self._num_assets):
    #             # Stacked LSTM cell
    #             net_asset = net[:,:,:,i]
    #             cell = tf.nn.rnn_cell.LSTMCell(self._num_hid)
    #             net_asset, _ = tf.nn.dynamic_rnn(cell, net_asset, dtype=tf.float32)
    #             lstm_outs.append(net_asset)
    #         net = tf.reshape(tf.stack(lstm_outs, axis=3), [bsz, bptt, -1])
    #     with tf.name_scope("Output_dense"):
    #         feat_len = tf.shape(net)[2]
    #         net = tf.reshape(net, [-1, feat_len])
    #         net = tf.layers.dense(net, self._num_assets + 1)
    #         net = tf.reshape(net, [-1, self._bptt, self._num_assets + 1])
    #     return net

    def predict_portfolio_allocation(self, data):
        last_logit = self.logits(data)[:, -1, :]
        return tf.nn.softmax(last_logit).numpy()


    def loss(self, data, target, is_training=False, one_hot_target = True):
        if one_hot_target:
            prediction = self.logits(data, is_training)
            # num_assets = tf.shape(target)[-1]
            idxs = tf.argmax(target, axis=2)
            log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=idxs)
            return tf.reduce_mean(tf.reduce_sum(log_probs))
            # target = tf.one_hot(idxs, depth = num_assets)
            # portfolio_ts = tf.multiply(prediction, target)
        else:
            prediction = tf.nn.softmax(self.logits(data, is_training), axis=2)
            portfolio_ts = tf.multiply(prediction, target)
            portfolio_comb = tf.reduce_sum(portfolio_ts, axis=2)
            apv_batch = tf.reduce_prod(portfolio_comb,axis=1)
            apv_mean = tf.reduce_mean(apv_batch)
            return -apv_mean

    def optimize(self, data, target):
        with tf.name_scope("backprop_fn"):
            # params = tf.trainable_variables()
            # grads = tf.gradients(self.loss(data, target, True), params)
            # grads, grad_norm = tf.clip_by_global_norm(grads, self._clip_norm)
            # self._optimizer.apply_gradients(zip(grads, params))

            gvs = self._optimizer.compute_gradients(lambda : self.loss(data, target, is_training=True))
            capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
            self._optimizer.apply_gradients(capped_gvs)

    def weighted_one_hot(self, target):
        pass


    def __repr__(self):
        return "Seq2SeqLSTM"
