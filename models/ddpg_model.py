import tensorflow as tf
import functools

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class DDPGActor:
    """
    'State': Current Market State
    'Action': Allocation weights
    """

    def __init__(self, num_hid=20, clip_norm=0.25,
                 num_features=3, num_assets=9,
                 bptt=5, lr=0.001, bsz=12,
                 num_dense_units=64, mix_factor=0.75):
        self.BASE_SCOPE = "Actor"
        self.TRAIN_SCOPE = "{}Network".format(self.BASE_SCOPE)
        self.TARGET_SCOPE = "{}TargetNetwork".format(self.BASE_SCOPE)
        self._num_hid = num_hid
        self._clip_norm = clip_norm
        self._num_features = num_features
        self._num_assets = num_assets
        self._bptt = bptt
        self._is_training = False
        self._bsz = bsz
        self._lr = lr
        self._num_dense_units = num_dense_units
        self._mix_factor = mix_factor
        self._wt_init = tf.initializers.random_uniform(minval=-3e-3, maxval=3e-3)
        with tf.variable_scope("graph_inputs_".format(self.BASE_SCOPE)):
            self.__market_state = tf.placeholder(tf.float32, [None, self._bptt,
                                                            self._num_features, self._num_assets])
            self.__allocation_gradients = tf.placeholder(tf.float32, [None, self._num_assets + 1])
            self.__true_reward = tf.placeholder(tf.float32, [None])
        self.__pred_allocation_op = self.build_network(self.TRAIN_SCOPE)
        self.__pred_allocation_target_op = self.build_network(self.TARGET_SCOPE)
        self.__optimize_op
        self.__update_target_network_op

    def build_network(self, scope_name):
        # Head scope - used to differentiate between Actor/ActorTarget
        with tf.variable_scope(scope_name):
            net = self.__market_state
            with tf.variable_scope("LSTM_Cell"):
                shape = net.get_shape().as_list()
                net = tf.reshape(net, [-1, shape[1], shape[2] * shape[3]])
                cell = tf.contrib.rnn.LSTMBlockCell(self._num_hid)
                net, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, net, dtype=tf.float32)
                net = tf.concat(net, axis=2)
            with tf.variable_scope("Dense_Layers"):
                shape = net.get_shape().as_list()
                # [bsz, bptt, feats] -> [bsz, bptt*num_feats]
                net = tf.reshape(net, [-1, shape[1]*shape[2]])
                net = tf.layers.dense(net, self._num_dense_units, activation=tf.nn.leaky_relu)
                net = tf.layers.dense(net, self._num_dense_units, activation=tf.nn.leaky_relu)
            with tf.variable_scope("Asset_Projection"):
                net = tf.layers.dense(net, self._num_assets + 1, kernel_initializer=self._wt_init)
            return tf.nn.softmax(net, axis=-1)

    @lazy_property
    def __optimize_op(self):
        with tf.variable_scope("optimize_{}".format(self.BASE_SCOPE)):
            optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
            train_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.TRAIN_SCOPE)
            predicted_allocation = self.__pred_allocation_op
            bsz = tf.cast(tf.shape(predicted_allocation)[0], tf.float32)
            # TODO: Why is this normalization step needed?
            grads = [tf.div(x, bsz) for x in tf.gradients(predicted_allocation,
                                                          train_params,
                                                          -self.__allocation_gradients)]
            grads, grad_norm = tf.clip_by_global_norm(grads, self._clip_norm)
            return optimizer.apply_gradients(zip(grads, train_params))

    @lazy_property
    def __update_target_network_op(self):
        with tf.variable_scope("update_target_{}".format(self.BASE_SCOPE)):
            train_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.TRAIN_SCOPE)
            target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.TARGET_SCOPE)
            return [target_params[i].assign(tf.multiply(train_params[i], self._mix_factor) +
                                            tf.multiply(target_params[i], self._mix_factor))
                    for i in range(len(target_params))]

    def predict_allocation(self, sess, market_state):
        return sess.run(self.__pred_allocation_op, {
            self.__market_state: market_state
        })

    def target_predict_allocation(self, sess, market_state):
        return sess.run(self.__pred_allocation_target_op, {
            self.__market_state: market_state
        })

    def optimize(self, sess, market_state, allocation_grad):
        sess.run(self.__optimize_op, {
            self.__market_state: market_state,
            self.__allocation_gradients: allocation_grad
        })
        return self

    def update_target_network(self, sess):
        sess.run(self.__update_target_network_op)


class DDPGCritic:
    """
    'State': Current Market State
    'Action': Allocation weights
    """

    def __init__(self, num_hid=20, clip_norm=0.25,
                 num_features=3, num_assets=9,
                 bptt=5, lr=0.001, bsz=12,
                 mix_factor=0.75, num_dense_units=64):
        self.BASE_SCOPE = "Critic"
        self.TRAIN_SCOPE = "{}Network".format(self.BASE_SCOPE)
        self.TARGET_SCOPE = "{}TargetNetwork".format(self.BASE_SCOPE)
        self._num_hid = num_hid
        self._clip_norm = clip_norm
        self._num_features = num_features
        self._num_assets = num_assets
        self._bptt = bptt
        self._is_training = False
        self._bsz = bsz
        self._lr = lr
        self._mix_factor = mix_factor
        self._num_dense_units = num_dense_units
        self._wt_init = tf.initializers.random_uniform(minval=-3e-3, maxval=3e-3)
        with tf.variable_scope("graph_inputs_{}".format(self.BASE_SCOPE)):
            self.__market_state = tf.placeholder(tf.float32, [None, self._bptt,
                                                            self._num_features, self._num_assets])
            self.__predicted_allocation = tf.placeholder(tf.float32, [None, self._num_assets + 1])
            self.__reward = tf.placeholder(tf.float32, [None, 1])
        self.__pred_q_op = self.build_network(self.TRAIN_SCOPE)
        self.__pred_q_target_op = self.build_network(self.TARGET_SCOPE)
        self.__loss_op
        self.__optimize_op
        self.__update_target_network_op
        self.__allocation_gradients

    def build_network(self, scope_name):
        # Head scope - used to differentiate between Critic/CriticTarget
        with tf.variable_scope(scope_name):
            state_net = self.__market_state
            action_net = self.__predicted_allocation
            # TODO: Experiment with LSTM here later(maybe)
            with tf.variable_scope("StateAction_Dense"):
                # (bsz, bptt, num_feats, num_assets) -> (bsz, bptt*num_feats*num_assets)
                state_net = tf.reshape(state_net, [-1, self._bptt*self._num_features*self._num_assets])
                state_net = tf.layers.dense(state_net, self._num_dense_units*2, activation=tf.nn.leaky_relu)
                # TODO: Add BatchNorm later
                state_net = tf.layers.dense(state_net, self._num_dense_units, activation=tf.nn.leaky_relu)
                action_net = tf.layers.dense(action_net, self._num_dense_units, activation=tf.nn.leaky_relu)
            with tf.variable_scope("StateAction_Mix"):
                net = state_net + action_net
            with tf.variable_scope("QValue_Projection"):
                qval = tf.layers.dense(net, 1, kernel_initializer=self._wt_init)
            return qval

    @lazy_property
    def __loss_op(self):
        with tf.variable_scope("loss_{}".format(self.BASE_SCOPE)):
            predicted_reward = self.__pred_q_op
            true_reward = self.__reward
            return tf.losses.mean_squared_error(labels=true_reward,
                                                predictions=predicted_reward)

    @lazy_property
    def __optimize_op(self):
        with tf.variable_scope("optimize_{}".format(self.BASE_SCOPE)):
            optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
            return optimizer.minimize(self.__loss_op)

    @lazy_property
    def __allocation_gradients(self):
        with tf.variable_scope("allocation_grads_{}".format(self.BASE_SCOPE)):
            return tf.gradients(self.__pred_q_op, self.__predicted_allocation)


    @lazy_property
    def __update_target_network_op(self):
        with tf.variable_scope("update_{}_target".format(self.BASE_SCOPE)):
            train_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.TRAIN_SCOPE)
            target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.TARGET_SCOPE)
            return [target_params[i].assign(tf.multiply(train_params[i], self._mix_factor) +
                                            tf.multiply(target_params[i], self._mix_factor))
                    for i in range(len(target_params))]

    def optimize(self, sess, market_state, allocation, reward):
        loss, _ = sess.run([self.__loss_op, self.__optimize_op], {
            self.__market_state: market_state,
            self.__predicted_allocation: allocation,
            self.__reward: reward
        })
        return loss

    def predict_q(self, sess, market_state, allocation):
        return sess.run(self.__pred_q_op, {
            self.__market_state: market_state,
            self.__predicted_allocation: allocation,
        })

    def target_predict_q(self, sess, market_state, allocation):
        return sess.run(self.__pred_q_target_op, {
            self.__market_state: market_state,
            self.__predicted_allocation: allocation,
        })

    def allocation_grad(self, sess, market_state, allocation):
        return sess.run(self.__allocation_gradients, {
            self.__market_state: market_state,
            self.__predicted_allocation: allocation,
        })

    def update_target_network(self, sess):
        sess.run(self.__update_target_network_op)