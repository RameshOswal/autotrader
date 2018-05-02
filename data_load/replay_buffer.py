import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        # Pool is a list of tuples (states, reward, action)
        self.pool = []

    def add(self, state, reward, action):
        """
        :param state (X): bsz X bptt X num_feats * num_assets
        :param reward (y): bsz
        :param action (weights): bsz X num_assets
        :return: None
        """
        assert len(state) == len(reward) and len(reward) == len(action), "Dimension Mismatch (REPLAY BUFFER)"

        for batch in range(state.shape[0]):
            if len(self.pool) < self.buffer_size:
                self.pool.append((state[batch, :, :], reward[batch], action[batch, :]))
            else:
                self.pool.remove(np.random.randint(0, self.buffer_size))
                self.pool.append((state[batch, :, :], reward[batch], action[batch, :]))

    def clear(self): self.pool = []

    def get_batch(self, bsz = 0):
        """
        :param bsz: batch size
        :return: states, rewards, actions
        """
        if bsz > len(self.pool): return np.random.shuffle(self.pool)

        ids = np.arange(start=0, stop=len(self.pool))
        np.random.shuffle(ids)

        states, rewards, actions = [], [], []
        for batch in ids:
            tup = self.pool[batch]
            states.append(tup[0])
            rewards.append(tup[1])
            actions.append(tup[2])
        return np.stack(states, axis = 0), np.stack(rewards, axis = 0), np.stack(actions, axis = 0)
