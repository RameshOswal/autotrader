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

class Network:
    def __init__(self,num_hid=20, clip_norm=0.25,
                 num_features=3, num_assets=9,
                 bptt=5, lr=3e-4, bsz=12,
                 num_dense_units=64, num_layers=2, delta=1e-5):
        self._num_hid = num_hid
        self._clip_norm = clip_norm
        self._num_features = num_features
        self._num_assets = num_assets
        self._bptt = bptt
        self._is_training = False
        self._bsz = bsz
        self._lr = lr
        self._shared_layers = [num_dense_units] * num_layers
        self._num_dense_units = num_dense_units
        self._delta = delta
        self._wt_init = tf.initializers.random_uniform(minval=-3e-3, maxval=3e-3)

        with tf.variable_scope("Network_Input"):
            self.__market_state = tf.placeholder(tf.float32, [None, self._bptt, self._num_features, self._num_assets])
            # self.__allocation_gradients = tf.placeholder(tf.float32, [None, self._num_assets + 1])
            self.__true_reward = tf.placeholder(tf.float32, [None])

        self._new_scp_name = "NEW"
        self._old_scp_name = "OLD"

        self.__new_policy_op, self.__new_value_op, self.__new_dist_op = self.build_ac_network(self._new_scp_name, self._shared_layers)
        self.__old_policy_op, self.__old_value_op, self.__old_dist_op = self.build_ac_network(self._old_scp_name, self._shared_layers)

    @lazy_property
    def build_ac_network(self, pre_scope, hidden_layers):
        shapes = self.__market_state.get_shape().as_list()
        net = tf.reshape(self.__market_state, [-1, shapes[1] , shapes[2] * shapes[3]])
        with tf.variable_scope(pre_scope + "_shared"):
            cell = tf.contrib.rnn.LSTMBlockCell(self._num_hid)
            net, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, net, dtype=tf.float32)
            net = tf.concat(net, axis=2)

            net = tf.reshape(net, [-1, 2 * self._num_hid * self._bptt])
            for idx, hidden in enumerate(hidden_layers):
                net = tf.layers.dense(net, units=hidden,
                                      activation=tf.nn.tanh, kernel_initializer=self._wt_init,
                                      name = "hid_{}".format(idx))

        with tf.variable_scope(pre_scope + "_actor"):
            # actor_net = tf.layers.dense(net, units=self._num_assets + 1, activation=None)
            # policy = tf.nn.softmax(actor_net, axis = -1)

            # Produce 2 networks - Sample from them
            actor_net1 = tf.layers.dense(net, units=self._num_assets + 1, activation=None)
            actor_net1 = tf.nn.tanh(actor_net1 + self._delta)

            actor_net2 = tf.layers.dense(net, units=self._num_assets + 1, activation=None)
            actor_net2 = tf.nn.softplus(actor_net2 + self._delta)

            # Skeptical!
            dist = tf.distributions.Normal(actor_net1, actor_net2)
            policy = tf.squeeze(dist.sample(self._bsz), axis=1)

        with tf.variable_scope(pre_scope + "_critic"):
            value = tf.layers.dense(net, units=1, activation=None)

        return policy, value, dist

    @lazy_property
    def __get_loss_op(self, obs_dim,
                    num_actions, gamma=1.0, epsilon=0.2, beta=0.01, scope='ppo',
                    reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            state_input_t = tf.placeholder(tf.float32, [None, self._bptt, self._num_features, self._num_assets])
            actor_t = tf.placeholder(tf.float32, [None, self._num_assets + 1])
            value_t = tf.placeholder(tf.float32, [None])
            advantage_t = tf.placeholder(tf.float32, [None])

            cur_policy = self.__new_dist_op.log_prob(actor_t + self._delta)
            old_policy = self.__old_dist_op.log_prob(actor_t + self._delta)
            ratio = tf.exp(cur_policy - old_policy)
            clipped_ratio = tf.clip_by_value(ratio, 1.0 - epsilon, 1.0 + epsilon)

            with tf.variable_scope("loss"):
                clip_loss = -tf.reduce_mean(tf.minimum(ratio, clipped_ratio) * advantage_t)
                value_loss = tf.reduce_mean(tf.square(self.__new_value_op - value_t))
                entropy_loss = tf.reduce_mean(self.__new_dist_op.entropy())
                penalty = - beta * entropy_loss

            return clip_loss + value_loss + penalty

    @lazy_property
    def __optimize_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
        return optimizer.minimize(self.__get_loss_op, var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self._new_scp_name
        ))

    @lazy_property
    def __update_old_network_op(self):
        new_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._new_scp_name)
        old_parms = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._old_scp_name)
        return [new_params[idx].assign(old) for idx, old in enumerate(old_parms)]
            











Network()




