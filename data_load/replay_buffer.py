import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.max_size = buffer_size
        # Pool is a list of tuples (states, reward, action)
        self.pool = deque()
        self.size = 0
        self.num_elem = 0

    def add(self, vars, bsz):
        """
        :param vars: list of elements whose first dimensions are EQUAL. [states, rewards, actions, ...]
            :param state (X): bsz X bptt X num_feats * num_assets
            :param reward (y): bsz
            :param action (weights): bsz X num_assets
        """
        # assert len(state) == len(reward) and len(reward) == len(action), "Dimension Mismatch (REPLAY BUFFER)"
        self.num_elem = len(vars)

        for batch in range(bsz):
            if self.size < self.max_size:
                self.pool.append(tuple([elem[batch] for elem in vars]))
                self.size += 1
            else:
                self.pool.popleft()
                self.pool.append(tuple([elem[batch] for elem in vars]))

        assert len(self.pool) == self.size, "Error in Adding Elements to Buffer! Sizes = {} and {}".format(len(self.pool), self.size)

    def clear(self):
        self.pool = deque()
        self.size = 0

    def get_batch(self, bsz = 0):
        """
        :param bsz: batch size
        :return: list of vars [states, rewards, actions, ...]
        """
        ids = np.arange(start=0, stop=self.size)
        np.random.shuffle(ids)

        out_vars = {_ : [] for _ in range(self.num_elem)}
        for batch in ids[:bsz]:
            tup = self.pool[batch]
            for idx in range(len(tup)):
                out_vars[idx].append(tup[idx])

        # Pop out the elements
        for _ in range(bsz): self.pool.popleft()

        self.size -= bsz

        assert len(self.pool) == self.size, "Error in Replay Buffer Pool Size!"

        return [np.stack(out_vars[x], axis = 0) for x in out_vars]
