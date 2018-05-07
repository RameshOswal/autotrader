import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size, rewards=False):
        self.max_size = buffer_size
        self.rewards = rewards
        # Pool is a list of tuples (states, reward, action)
        self.pool = deque()
        self.size = 0


    def add(self, state=None, reward=None, action=None):
        """
        :param state (X): bsz X bptt X num_feats * num_assets
        :param reward (y): bsz
        :param action (weights): bsz X num_assets
        """
        # assert len(state) == len(reward) and len(reward) == len(action), "Dimension Mismatch (REPLAY BUFFER)"

        if self.size < self.max_size:
            self.pool.append((state, reward, action))
            self.size += 1
        else:
            self.pool.popleft()
            self.pool.append((state, reward, action))

    def clear(self):
        self.pool = deque()
        self.size = 0

    def get_batch(self, bsz = 0):
        """
        :param bsz: batch size
        :return: list of vars [states, rewards, actions, ...]
        """
        ids = np.arange(0, self.size)
        np.random.shuffle(ids)

        elems = [self.pool[x] for x in ids[:bsz]]

        # print(elems[0])
        states = np.concatenate([x[0] for x in elems], axis=0)
        rewards = None if self.rewards == False else np.array([x[1] for x in elems])
        # (bsz, ) -> (bsz, 1)
        rewards = rewards.reshape((len(rewards), 1))
        actions = np.concatenate([x[2] for x in elems], axis=0)

        # print(states.shape, rewards, actions.shape)
        return states, rewards, actions
