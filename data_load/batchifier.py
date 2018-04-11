__author__ = "deeptrader"

import numpy as np
from data_load.load_crypto import DataPreprocess
from literals import ASSET_LIST

class batchify:
    def __init__(self, data_path="."):
        self.dp = DataPreprocess()
        self.train_dates = self.dp.train_dates
        self.test_dates = self.dp.test_dates
        self.data_path = data_path

    def create_batches(self, X, X_c):
        """Divide the prices by the closing price to get Price Relative Vector"""
        return X.divide(X_c.values)


    def load_train(self, bsz = 16, bptt = 50, asset_list = ASSET_LIST, normalize = False, idx = 0):
        """
        :param bsz: Batch size
        :param bptt: History
        :param asset_list: Number of assets to keep
        :param idx: Date range (0-3)
        :return: batches of dataset
        """
        vals = zip(self.loader("high", asset_list=asset_list, idx=idx), self.loader("low", asset_list=asset_list, idx=idx), self.loader("close", asset_list=asset_list, idx=idx))
        H, L, C = zip(*vals)

        high, low, close = H[0], L[0], C[0]

        # Batch IDs, Multiplying with bptt parameter to ensure non-overlapping batches
        shuffle_ids = np.arange(start=0, stop=len(high) - bptt )
        # shuffle_ids *= bptt

        # np.random.shuffle(shuffle_ids)
        num_batches = len(shuffle_ids) // bsz + 1
        for idx in range(num_batches):
            # s_ids are the Batch IDs for this batch, of the form [  0  50 100 150 200 250 300 350 400 450 500 550 600 650 700 750] if bptt == 50
            s_ids = shuffle_ids[idx * bsz: (idx + 1) * bsz]

            # X.shape => (bptt X num_features X num_assets), y.shape => (bptt X num_assets)
            X = np.zeros(shape = (bptt, 3, len(asset_list)))
            y = np.zeros(shape = (bptt, len(asset_list)))

            # X.shape => (bsz X bptt X num_assets), y.shape => (bsz X bptt X num_assets)
            X = X[np.newaxis, :]

            for s_idx in s_ids:
                if normalize:
                    h_batch = self.create_batches(high.iloc[s_idx: s_idx + bptt, :], close.iloc[s_idx + bptt - 1, :])
                    l_batch = self.create_batches(low.iloc[s_idx: s_idx + bptt, :], close.iloc[s_idx + bptt - 1, :])
                    c_batch = self.create_batches(close.iloc[s_idx: s_idx + bptt, :], close.iloc[s_idx + bptt - 1, :])
                else:
                    h_batch = high.iloc[s_idx: s_idx + bptt, :]
                    l_batch = low.iloc[s_idx: s_idx + bptt, :]
                    c_batch = close.iloc[s_idx: s_idx + bptt, :]

                # Convert all three to matrix form, out.shape => (num_features X bptt X num_assets)
                out = np.array([h_batch.as_matrix(), l_batch.as_matrix(), c_batch.as_matrix()])

                if len(h_batch) != bptt: continue

                y = close.iloc[s_idx + 1: s_idx + bptt + 1, :].as_matrix() / c_batch
                y = np.pad(y , [(0, 0), (0,1)], constant_values=1, mode="constant")
                y = y[np.newaxis, :]
                # print(y)
                X = np.vstack((X, np.reshape(out.transpose([1, 0, 2]), [1, len(h_batch), 3, len(asset_list)])))
                y = np.vstack((y, np.reshape(y, [1, len(h_batch), len(asset_list) + 1])))

            yield X[1:, :, :, :], y[1:, :, :]


    def load_test(self, bsz = 16, bptt = 50, asset_list = ASSET_LIST, normalize = False, idx = 0):
        vals = zip(self.loader("high", asset_list = asset_list, idx=idx), self.loader("low", asset_list = asset_list, idx=idx), self.loader("close", asset_list = asset_list, idx=idx))
        H, L, C = zip(*vals)

        high, low, close = H[1], L[1], C[1]

        # Batch IDs, Multiplying with bptt parameter to ensure non-overlapping batches
        shuffle_ids = np.arange(start=0, stop=len(high) // bptt + 1)
        shuffle_ids *= bptt

        # np.random.shuffle(shuffle_ids)
        num_batches = len(shuffle_ids) // bsz + 1
        for idx in range(num_batches):
            # s_ids are the Batch IDs for this batch, of the form [  0  50 100 150 200 250 300 350 400 450 500 550 600 650 700 750] if bptt == 50
            s_ids = shuffle_ids[idx * bsz: (idx + 1) * bsz]

            # X.shape => (bptt X num_features X num_assets)
            X = np.zeros(shape=(bptt, 3, len(asset_list)))

            # X.shape => (bsz X bptt X num_assets)
            X = X[np.newaxis, :]

            for s_idx in s_ids:
                if normalize:
                    h_batch = self.create_batches(high.iloc[s_idx: s_idx + bptt, :], close.iloc[s_idx + bptt - 1, :])
                    l_batch = self.create_batches(low.iloc[s_idx: s_idx + bptt, :], close.iloc[s_idx + bptt - 1, :])
                    c_batch = self.create_batches(close.iloc[s_idx: s_idx + bptt, :], close.iloc[s_idx + bptt - 1, :])
                else:
                    h_batch = high.iloc[s_idx: s_idx + bptt, :]
                    l_batch = low.iloc[s_idx: s_idx + bptt, :]
                    c_batch = close.iloc[s_idx: s_idx + bptt, :]

                # Convert all three to matrix form, out.shape => (num_features X bptt X num_assets)
                out = np.array([h_batch.as_matrix(), l_batch.as_matrix(), c_batch.as_matrix()])

                if len(h_batch) != bptt: continue

                X = np.vstack((X, np.reshape(out.transpose([1, 0, 2]), [1, len(h_batch), 3, len(asset_list)])))

            yield X[1:, :, :, :]


    def loader(self, name, asset_list = ASSET_LIST, idx = 0):
        """
        :param name: feature name. Can take values like "open"/"close", "high", "low"
        :param asset_list: Number of assets to keep
        :return: (Generator iterates for all the date ranges) Dataframe where each column is an asset. Number of records is for the entire date range.
        """
        return self.dp.load_train_test(asset_name = asset_list, feature_type = name, idx=idx, path=self.data_path)

