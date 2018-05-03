import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.max_size = buffer_size
        # Pool is a list of tuples (states, reward, action)
        self.pool = []
        self.size = 0

    def add(self, state, reward, action):
        """
        :param state (X): bsz X bptt X num_feats * num_assets
        :param reward (y): bsz
        :param action (weights): bsz X num_assets
        """
        assert len(state) == len(reward) and len(reward) == len(action), "Dimension Mismatch (REPLAY BUFFER)"

        for batch in range(state.shape[0]):
            if len(self.pool) < self.max_size:
                self.pool.append((state[batch, :, :], reward[batch], action[batch, :]))
            else:
                self.pool.remove(np.random.randint(0, self.max_size))
                self.pool.append((state[batch, :, :], reward[batch], action[batch, :]))
            self.size += 1

        assert len(self.pool) == self.size, "Error in Adding Elements to Buffer!"

    def clear(self):
        self.pool = []
        self.size = 0

    def get_batch(self, bsz = 0):
        """
        :param bsz: batch size
        :return: states, rewards, actions
        """
        ids = np.arange(start=0, stop=self.size)
        np.random.shuffle(ids)

        states, rewards, actions = [], [], []
        for batch in ids[:bsz]:
            tup = self.pool[batch]
            states.append(tup[0])
            rewards.append(tup[1])
            actions.append(tup[2])

        # Pop out the elements
        for batch in ids: self.pool.remove(batch)
        self.size -= bsz

        assert len(self.pool) == self.size, "Error in Replay Buffer Pool Size!"

        return np.stack(states, axis = 0), np.stack(rewards, axis = 0), np.stack(actions, axis = 0)
