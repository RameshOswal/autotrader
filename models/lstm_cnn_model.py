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


class LSTMCNNModel:

    def __init__(
            self,
            num_hid=20,
            clip_norm=0.25,
            num_features=3,
            num_assets=9,
            bptt=5,
            lr=0.001):
        self._num_hid = num_hid
        self._clip_norm = clip_norm
        self._num_features = num_features
        self._num_assets = num_assets
        self._bptt = bptt
        self._is_training = False
        self._optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self._asset_wt_projection = tf.layers.Dense(self._num_assets + 1, activation=tf.nn.softmax)
        self._cell = tf.contrib.rnn.LSTMBlockCell(self._num_hid)

        # Convolution Parameters
        self.kernel_sizes = [4, 4]
        self.strides = [1, 1]
        self.n_channels = [6, 6]

        self.conv1 = tf.layers.Conv2D(filters=self.n_channels[0], strides=(1, self.strides[0]),
                                      kernel_size=(1, self.kernel_sizes[0]), padding="SAME")

        self.conv2 = tf.layers.Conv2D(filters=self.n_channels[1], strides=(1, self.strides[1]),
                                      kernel_size=(1, self.kernel_sizes[1]), padding="SAME")

    def logits(self, data, is_training=False):
        net = tf.constant(data, dtype=tf.float32)
        shape = net.get_shape().as_list()
        net = tf.reshape(net, [-1, shape[1], shape[2]*shape[3]]) # bsz X bptt X (num_feats * num_assets)

        net = tf.expand_dims(net, axis = 1) # net => bsz X 1 X bptt X (num_feats * num_assets)

        with tf.variable_scope("CNN_LSTM", reuse=tf.AUTO_REUSE):
            conv1 = self.conv1(net)
            conv2 = self.conv2(conv1)

            net = tf.squeeze(conv2, axis = 1)

            cell = self._cell
            net, _ = tf.nn.dynamic_rnn(cell, net, dtype=tf.float32)
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


    def loss(self, data, target, is_training=False):
        prediction = tf.nn.softmax(self.logits(data, is_training),dim=2)
        portfolio_ts = tf.multiply(prediction, target)
        portfolio_comb = tf.reduce_sum(portfolio_ts, axis=2)
        apv_batch = tf.reduce_prod(portfolio_comb,axis=1)
        apv_mean = tf.reduce_mean(apv_batch)
        return -apv_mean

    def optimize(self, data, target):
        with tf.name_scope("backprop_fn"):
            gvs = self._optimizer.compute_gradients(lambda :self.loss(data, target, True))
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            self._optimizer.apply_gradients(capped_gvs)

    def __repr__(self):
        return "Seq2SeqLSTM"
