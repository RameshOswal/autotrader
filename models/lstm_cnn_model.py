import tensorflow as tf
import tensorflow.contrib.eager as tfe
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

# def lstm(x, prev_c, prev_h, w_lstm):
#     ifog = tf.matmul(tf.concat([x, prev_h], axis=1), w_lstm)
#     i, f, o, g = tf.split(ifog, 4, axis=1)
#     i, f, o, g = tf.sigmoid(i), tf.sigmoid(f), tf.sigmoid(o), tf.tanh(g)
#     next_c = f * prev_c + i * g
#     next_h = o * tf.tanh(next_c)
#     return next_c, next_h


class LSTMCNNModel:

    def __init__(
            self,
            num_hid=20,
            clip_norm=0.25,
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
        self._bsz = bsz

        self.dense1 = tf.layers.Dense(64, activation=None)
        self.dense2 = tf.layers.Dense(64, activation=None)

        self._asset_wt_projection = tf.layers.Dense(self._num_assets + 1, activation=None)



        self._cell = tf.contrib.rnn.LSTMBlockCell(self._num_hid)

        # Convolution Parameters
        self.kernel_sizes = [8, 8]
        self.strides = [1, 1]
        # self.n_channels = [20, 20]
        #
        # self.conv1 = tf.layers.Conv2D(filters=self.n_channels[0], strides=(1, self.strides[0]),
        #                               kernel_size=(1, self.kernel_sizes[0]), padding="SAME")
        #
        # self.conv2 = tf.layers.Conv2D(filters=self.n_channels[1], strides=(1, self.strides[1]),
        #                               kernel_size=(1, self.kernel_sizes[1]), padding="SAME")
        #
        # self.prev_h = tf.get_variable("prev_h", shape=[1, self._num_hid],
        #                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        # self.prev_c = tf.get_variable("prev_c", shape=[1, self._num_hid],
        #                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        # self.w_lstm = tf.get_variable("w_lstm", shape=[(num_assets * num_features) + self._num_hid, 4 * self._num_hid],
        #                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

    def logits(self, is_training=False):
        net = tf.placeholder(dtype = tf.float32, shape = [None, self._bptt, self._num_features, self._num_assets])
        target =  tf.placeholder(dtype = tf.float32, shape = [None, self._bptt, self._num_assets + 1])

        shape = net.get_shape().as_list()
        X_net = tf.reshape(net, [-1, shape[1], shape[2]*shape[3]]) # bsz X bptt X (num_feats * num_assets)

        # X_net = tf.expand_dims(X_net, axis = 1) # net => bsz X 1 X bptt X (num_feats * num_assets)
        # net = tf.keras.utils.normalize(net)

        with tf.variable_scope("LSTM_Cell", reuse=tf.AUTO_REUSE):
            X_net, _ = tf.nn.dynamic_rnn(self._cell, X_net, dtype=tf.float32)

        # with tf.variable_scope("CNN", reuse=tf.AUTO_REUSE):
        #     conv1 = self.conv1(X_net)
        #     conv1 = tf.nn.relu(conv1)
        #
        #     conv2 = self.conv2(conv1)
        #     conv2 = tf.nn.relu(conv2)
        #
        #     X_net = tf.squeeze(conv2, axis = 1)

        with tf.variable_scope("Output_dense", reuse=tf.AUTO_REUSE):
            X_net = tf.reshape(X_net, [-1, self._num_hid])
            d1 = self.dense1(X_net)
            d2 = self.dense2(d1)

            d2  = self._asset_wt_projection(d2)
            logits = tf.reshape(d2, [-1, self._bptt, self._num_assets + 1])

        with tf.variable_scope("loss_fn", reuse=tf.AUTO_REUSE):
            idxs = tf.argmax(target, axis = 2)
            log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=idxs)
            loss = tf.reduce_mean(tf.reduce_sum(log_probs, axis = 1))

            global_step = tf.train.get_or_create_global_step()

            params = tf.trainable_variables()
            grads = tf.gradients(loss, params)
            grads, grad_norm = tf.clip_by_global_norm(grads, self._clip_norm)
            train_op = self._optimizer.apply_gradients(zip(grads, params))

            # gvs = self._optimizer.compute_gradients(loss)
            # capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gvs]
            # train_op = self._optimizer.apply_gradients(capped_gvs)

        with tf.variable_scope("predict_portfolio_allocation", reuse=tf.AUTO_REUSE):
            alloc_weights = tf.nn.softmax(logits[:, -1, :])

        return {
            "net": net,
            "target": target,
            "train_op": train_op,
            "alloc_weights": alloc_weights,
            "loss": loss,
            "logits": logits,
            "gs": global_step
        }


    # def predict_portfolio_allocation(self, data):
    #     last_logit = self.logits(data)[:, -1, :]
    #     return tf.nn.softmax(last_logit).numpy()
    #
    #
    # def loss(self, data, target, is_training=False, one_hot_target = True):
    #     if one_hot_target:
    #         prediction = self.logits(data, is_training)
    #         # num_assets = tf.shape(target)[-1]
    #         idxs = tf.argmax(target, axis=2)
    #         log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=idxs)
    #         return tf.reduce_mean(tf.reduce_sum(log_probs, axis = 1))
    #         # target = tf.one_hot(idxs, depth = num_assets)
    #         # portfolio_ts = tf.multiply(prediction, target)
    #     else:
    #         prediction = tf.nn.softmax(self.logits(data, is_training), axis=2)
    #         portfolio_ts = tf.multiply(prediction, target)
    #         portfolio_comb = tf.reduce_sum(portfolio_ts, axis=2)
    #         apv_batch = tf.reduce_prod(portfolio_comb,axis=1)
    #         apv_mean = tf.reduce_mean(apv_batch)
    #         return -apv_mean
    #
    # def optimize(self, data, target):
    #     with tf.name_scope("backprop_fn"):
    #         # val_grad_fn = tfe.implicit_value_and_gradients(lambda x, y, z :self.loss(x, y, z))
    #         # val, grads = val_grad_fn(data, target, True)
    #         # self._optimizer.apply_gradients(grads)
    #         gvs = self._optimizer.compute_gradients(lambda :self.loss(data, target, True))
    #         capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gvs]
    #         self._optimizer.apply_gradients(capped_gvs)

    def __repr__(self):
        return "Seq2SeqLSTM"
