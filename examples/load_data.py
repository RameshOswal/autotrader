__author__ = "deeptrader"

import tensorflow as tf
import numpy as np
from data_load.load_crypto import DataPreprocess
from literals import ASSET_LIST

class batchify:
    def __init__(self):
        self.dp = DataPreprocess()
        self.train_dates = self.dp.train_dates
        self.test_dates = self.dp.test_dates

    def create_batches(self, X, X_c):
        return X.divide(X_c.values)

    def load_train(self, bsz = 100, bptt = 50, asset_list = ASSET_LIST):
        for vals in zip(self.loader("high", asset_list=asset_list),
                        self.loader("low", asset_list=asset_list),
                        self.loader("close", asset_list=asset_list)):
            high, low, close = vals[0][0], vals[1][0], vals[-1][0]
            shuffle_ids = np.arange(start=0, stop=len(high) - bptt)
            np.random.shuffle(shuffle_ids)
            num_batches = len(high) // bsz

            for idx in range(num_batches):
                s_ids = shuffle_ids[idx * bsz: (idx + 1) * bsz]
                batch = np.zeros(shape=(bptt, 3, len(asset_list)))
                batch = batch[np.newaxis, :]
                for s_idx in s_ids:
                    h_batch = self.create_batches(high.iloc[s_idx: s_idx + bptt, :], close.iloc[s_idx + bptt - 1, :])
                    l_batch = self.create_batches(low.iloc[s_idx: s_idx + bptt, :], close.iloc[s_idx + bptt - 1, :])
                    c_batch = self.create_batches(close.iloc[s_idx: s_idx + bptt, :], close.iloc[s_idx + bptt - 1, :])
                    out = np.array([h_batch.as_matrix(), l_batch.as_matrix(), c_batch.as_matrix()]).transpose([1, 0, 2])
                    batch = np.vstack((batch, np.reshape(out, [1, bptt, 3, len(asset_list)])))

                yield batch[1:]

    def load_test(self, bsz = 100, bptt = 50, asset_list = ASSET_LIST):
        for vals in zip(self.loader("high", asset_list = asset_list), self.loader("low", asset_list = asset_list), self.loader("close", asset_list = asset_list)):
            high, low, close = vals[0][1], vals[1][1], vals[-1][1]
            shuffle_ids = np.arange(start = 0, stop = len(high) - bptt)
            np.random.shuffle(shuffle_ids)
            num_batches = len(high) // bsz

            for idx in range(num_batches):
                s_ids = shuffle_ids[idx * bsz : (idx + 1) * bsz]
                batch = np.zeros(shape = (bptt, 3, len(asset_list)))
                batch = batch[np.newaxis, :]
                for s_idx in s_ids:
                    h_batch = self.create_batches(high.iloc[s_idx : s_idx + bptt, :], close.iloc[s_idx + bptt - 1, :])
                    l_batch = self.create_batches(low.iloc[s_idx : s_idx + bptt, :], close.iloc[s_idx + bptt - 1, :])
                    c_batch = self.create_batches(close.iloc[s_idx : s_idx + bptt, :], close.iloc[s_idx + bptt - 1, :])
                    out = np.array([h_batch.as_matrix(), l_batch.as_matrix(), c_batch.as_matrix()]).transpose([1, 0, 2])
                    batch = np.vstack((batch, np.reshape(out, [1, bptt, 3, len(asset_list)])))

                yield batch[1:]


    def loader(self, name, asset_list = ASSET_LIST):
        return self.dp.load_train_test(asset_name = asset_list, feature_type = name)

if __name__ == '__main__':
    a = batchify()
    a.load_train()


