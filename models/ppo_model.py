import tensorflow as tf
import functools
import numpy as np
from collections import deque


def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class PPONetwork:
    def __init__(self,num_hid=20, clip_norm=0.25,
                 num_features=3, num_assets=9,
                 bptt=5, lr=3e-4, bsz=12,
                 num_dense_units=64, num_layers=2, delta=1e-5, tag=""):
        self._num_hid = num_hid
        self._clip_norm = clip_norm
        self._num_features = num_features
        self._num_assets = num_assets
        self._bptt = bptt

        self._bsz = bsz
        self._lr = lr
        self._shared_layers = [num_dense_units] * num_layers
        self._num_dense_units = num_dense_units
        self._delta = delta
        self._wt_init = tf.initializers.random_uniform(minval=-3e-3, maxval=3e-3)

        self.n_channels = [32, 32]
        self.kernel_sizes = [3, 3]
        self.strides = [1, 1]

        with tf.variable_scope("Network_Input"):
            self._market_state = tf.placeholder(tf.float32, [None, self._bptt, self._num_features, self._num_assets],name="INPUT")
            self._is_training = tf.placeholder(tf.bool, None)
            # self.__allocation_gradients = tf.placeholder(tf.float32, [None, self._num_assets + 1])
            # self._true_reward = tf.placeholder(tf.float32, [None,1])

        self.actor_t = tf.placeholder(tf.float32, [None, self._num_assets + 1],name="ACTOR_T")
        self.value_t = tf.placeholder(tf.float32, [None,1],name="VALUE_T")
        self.advantage_t = tf.placeholder(tf.float32, [None,1],name="ADVANTAGE_T")

        self._new_scp_name = "{}_NEW".format(tag)
        self._old_scp_name = "{}_OLD".format(tag)
        self._tmp_scp_name = "{}_TEMP".format(tag)

        self._new_policy_op, self._new_value_op, self._new_dist_op = self.build_ac_network(self._new_scp_name, self._shared_layers)
        self._old_policy_op, self._old_value_op, self._old_dist_op = self.build_ac_network(self._old_scp_name, self._shared_layers)
        self._tmp_policy_op, self._tmp_value_op, self._tmp_dist_op = self.build_ac_network(self._tmp_scp_name, self._shared_layers)


        self.__get_loss_op
        self.__optimize_op
        self.__backup_new_params_op
        self.__update_old_params_op


    def build_ac_network(self, pre_scope, hidden_layers):
        shapes = self._market_state.get_shape().as_list()
        net = tf.reshape(self._market_state, [-1, shapes[1] , shapes[2] * shapes[3]])
        with tf.variable_scope(pre_scope + "_shared"):
            cell = tf.contrib.rnn.LSTMBlockCell(self._num_hid)
            net, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, net, dtype=tf.float32)
            net = tf.concat(net, axis=2)
            # net = tf.expand_dims(net, axis=1)
            # net = tf.layers.conv2d(net, filters=self.n_channels[0], strides=(1, self.strides[0]),
            #                        kernel_size=(1, self.kernel_sizes[0]), padding="SAME")
            # net = self.batch_norm(net, self.n_channels[0], self._is_training)
            # net = tf.layers.conv2d(net, filters=self.n_channels[1], strides=(1, self.strides[1]),
            #                        kernel_size=(1, self.kernel_sizes[1]), padding="SAME")
            # net = self.batch_norm(net, self.n_channels[1], self._is_training)
            # net = tf.squeeze(net, axis=1)

            # shape = net.get_shape().as_list()
            net = tf.reshape(net, [-1, 2 * self._num_hid * self._bptt])
            for idx, hidden in enumerate(hidden_layers):
                net = tf.layers.dense(net, units=hidden,
                                      activation=tf.nn.tanh, kernel_initializer=self._wt_init,
                                      name = "hid_{}".format(idx))

        with tf.variable_scope(pre_scope + "_actor"):
            # actor_net = tf.layers.dense(net, units=self._num_dense_units, activation=None)
            # policy = tf.layers.dense(actor_net, units = self._num_assets + 1, activation=None)
            # policy = tf.nn.softmax(policy, axis = -1)
            # actor_net = tf.layers.dense(actor_net, units=self._num_assets + 1, activation=None)

            # policy = tf.squeeze(tf.one_hot(tf.multinomial(actor_net, num_samples = 1), depth=self._num_assets + 1), axis = 1)
            # policy = tf.to_float(tf.multinomial(actor_net, num_samples = 1))

            # Produce 2 networks - Sample from them
            actor_net1 = tf.layers.dense(net, units=self._num_assets + 1, activation=None)
            actor_net1 = tf.nn.tanh(actor_net1 + self._delta)

            actor_net2 = tf.layers.dense(net, units=self._num_assets + 1, activation=None)
            actor_net2 = tf.nn.softplus(actor_net2 + self._delta)
            #
            # # Skeptical!
            dist = tf.distributions.Normal(actor_net1, actor_net2)
            policy = dist.sample()

        with tf.variable_scope(pre_scope + "_critic"):
            value = tf.layers.dense(net, units=1, activation=None)

        return policy, value, dist

    @lazy_property
    def __get_loss_op(self, gamma=1.0, epsilon=0.2, beta=0.01, scope='ppo',
                    reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            # cur_policy = self._new_policy_op
            # old_policy = self._old_policy_op
            # ratio = cur_policy / old_policy
            cur_policy = self._new_dist_op.log_prob(self.actor_t + self._delta)
            old_policy = self._old_dist_op.log_prob(self.actor_t + self._delta)
            ratio = tf.exp( cur_policy - old_policy)
            clipped_ratio = tf.clip_by_value(ratio, 1.0 - epsilon, 1.0 + epsilon)

            with tf.variable_scope("loss"):
                clip_loss = -tf.reduce_mean(tf.minimum(ratio, clipped_ratio) * self.advantage_t)
                value_loss = tf.reduce_mean(tf.square(self._new_value_op - self.value_t))
                entropy_loss = tf.reduce_mean(self._new_dist_op.entropy())
                penalty = - beta * entropy_loss

            return clip_loss + value_loss + penalty, tf.reduce_mean(clipped_ratio)

    @lazy_property
    def __optimize_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
        return optimizer.minimize(self.__get_loss_op[0], var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self._new_scp_name
        ))


    @lazy_property
    def __backup_new_params_op(self):
        new_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._new_scp_name)
        tmp_parms = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._tmp_scp_name)
        return [tmp_parms[idx].assign(old) for idx, old in enumerate(new_params)]

    @lazy_property
    def __update_old_params_op(self):
        tmp_parms = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._tmp_scp_name)
        old_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._old_scp_name)
        return [old_params[idx].assign(old) for idx, old in enumerate(tmp_parms)]

    @lazy_property
    def __update_test_model_op(self):
        trn_model = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "train")
        tst_model = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "test")
        return [tst_model[idx].assign(old) for idx, old in enumerate(trn_model)]

    @lazy_property
    def __backup_train_model_op(self):
        trn_model = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "train")
        tmp_model = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "tmp")
        return [tmp_model[idx].assign(old) for idx, old in enumerate(trn_model)]

    @lazy_property
    def __update_train_model_op(self):
        tmp_model = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "tmp")
        trn_model = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "train")
        return [trn_model[idx].assign(old) for idx, old in enumerate(tmp_model)]


    def update_test_model(self, sess):
        return sess.run(self.__update_test_model_op)

    def backup_train_model(self, sess):
        return sess.run(self.__backup_train_model_op)

    def update_train_model(self, sess):
        return sess.run(self.__update_train_model_op)



    def backup_new_params(self, sess):
        return sess.run(self.__backup_new_params_op)

    def update_old_params(self, sess):
        return sess.run(self.__update_old_params_op)

    def optimize(self, sess, state, action, value, advantage, is_train):
        return sess.run([self.__optimize_op, self.__get_loss_op], {
            self._market_state: state, self.actor_t: action,
            self.value_t: value, self.advantage_t: advantage, self._is_training: is_train
        })

    def get_softmax_policy(self, sess, state, is_train):
        return sess.run([tf.nn.softmax(self._new_policy_op, axis=-1), self._new_value_op], {
            self._market_state: state, self._is_training: is_train
        })



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

class PPOAgent:
    def __init__(self, bsz, gamma=0.9, lam=0.95, reuse=None,
                   num_hid=20, clip_norm=0.25,
                   num_features=3, num_assets=9,
                   bptt=5, lr=3e-4,
                   num_dense_units=64, num_layers=2, delta=1e-5, tag=0):
        self.gamma = gamma
        self.lam = lam
        self.t = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.next_values = []

        self._network = PPONetwork(bsz=bsz,num_hid=num_hid, clip_norm=clip_norm,
                                   num_features=num_features, num_assets=num_assets,
                                   bptt=bptt, lr=lr,
                                   num_dense_units=num_dense_units, num_layers=num_layers, delta=delta, tag="train"
                                   )
        self._test_network = PPONetwork(bsz=bsz,num_hid=num_hid, clip_norm=clip_norm,
                                   num_features=num_features, num_assets=num_assets,
                                   bptt=bptt, lr=lr,
                                   num_dense_units=num_dense_units, num_layers=num_layers, delta=delta, tag="test"
                                   )

        self._tmp_network = PPONetwork(bsz=bsz, num_hid=num_hid, clip_norm=clip_norm,
                                        num_features=num_features, num_assets=num_assets,
                                        bptt=bptt, lr=lr,
                                        num_dense_units=num_dense_units, num_layers=num_layers, delta=delta, tag="tmp"
                                        )

    def act_and_fetch(self, sess, network, last_state, last_action, last_value, reward, cur_state, is_train, idx, mode = "train"):
        if mode == "train":
            action, value = sess.run([network._new_policy_op, network._new_value_op],{
                network._market_state : cur_state, network._is_training: is_train
            })
        else:
            action, value = network.get_softmax_policy(sess, cur_state, is_train)

        # action, value = action[0], value[0]
        if idx != 0:
            self.states.append(last_state)
            self.actions.append(last_action)
            self.rewards.append(reward)
            self.values.append(last_value)
            self.next_values.append(value)
        self.t += 1
        return action, value



    def train(self, sess, network, state, actions, values, advantage, is_train):
        network.backup_new_params(sess)
        _, loss_ratio = network.optimize(sess, state, actions, values, advantage, is_train)
        network.update_old_params(sess)
        return loss_ratio[0], loss_ratio[1]


    def _reset_trajectories(self):
        self.states = []
        self.rewards = []
        self.actions = []
        self.values = []
        self.next_values = []

    def backup_train_update_test_model(self, sess):
        self._network.backup_train_model(sess)
        self._network.update_test_model(sess)

    def update_train_from_tmp_model(self, sess):
        self._network.update_train_model(sess)


    def get_training_data(self, reset=True):
        obss = np.asarray(self.states)
        actions = np.asarray(self.actions)
        deltas = []
        returns = []
        V = 0

        for i in reversed(range(len(self.states))):
            reward = self.rewards[i]
            value = self.values[i]
            next_value = self.next_values[i]
            delta = reward + self.gamma * next_value - value
            V = delta + self.lam * self.gamma * V
            deltas.append(V)
            returns.append(V + value)
        delta_t = np.array(list(reversed(deltas)), dtype=np.float32)
        value_t = np.array(list(reversed(returns)), dtype=np.float32)

        # standardize advantages
        delta_t = (delta_t - delta_t.mean()) / (delta_t.std() + 1e-5)
        if reset==True: self._reset_trajectories()


        return np.squeeze(obss, axis=1), np.squeeze(actions, axis=1), np.squeeze(value_t, axis=1), np.squeeze(delta_t, axis=1)
