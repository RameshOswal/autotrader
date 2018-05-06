__author__ = "deeptrader"

import numpy as np
# from data_load.load_crypto import DataPreprocess
import data_load.load_crypto  as crypto
import data_load.load_stocks as stocks
from literals import ASSET_LIST

class Batchifier:
    def __init__(self, data_path=".", bsz=16, bptt=50, idx=0, asset_list=ASSET_LIST,
                 randomize_train=True, overlapping_train=False, data_preprocess=crypto.DataPreprocess()):
        """
        :param data_path: Path to 'Poloneix_Preprocessed'
        :param bsz: Batch size
        :param bptt: History
        :param idx: Date range (0-3)
        :param asset_list: Number of assets to keep
        :param randomize_train: Randomize train batches
        :param overlapping_train: Should train batches be overlapping?
        Note that test input should be overlapping and sequential since at each timestep we
        will look at history to determine portfolio allocation for next timestep
        TODO(saatvik): I dont think 'normalize' is needed - Data seems to always be normalized
        """
        # self.dp = DataPreprocess()
        self.dp = data_preprocess
        self.train_dates = self.dp.train_dates
        self.test_dates = self.dp.test_dates
        self.data_path = data_path
        self.bsz = bsz
        self.bptt = bptt
        self.idx = idx
        self.asset_list = asset_list
        self.randomize_train = randomize_train
        self.overlapping_train = overlapping_train


    def load_train(self):
        """
        :return:
        X: (bsz, bptt, num_feats, num_assets) - Features
        y: (bsz, bptt, num_assets) - Relative change price
        """
        return self.load_batch(randomize_batches=self.randomize_train,
                              overlapping_batches=self.overlapping_train, is_test=False)


    def load_test(self):
        """
        :return:
        X: features[timestamp-bptt:timestamp],(bsz, bptt, num_feats, num_assets)
        y: price_of_assets[timestamp]/price_of_assets[timestamp-1] [bsz, num_assets]
        Note that X will start from the bptt'th timestep and have overlapping samples in the batch
        i.e. X[t] => X[0:t], X[t+1] = X[1:t+1] onwards.
        """
        return self.load_batch(randomize_batches=False, overlapping_batches=True, is_test=True)

    def load_batch(self, randomize_batches, overlapping_batches, is_test):
        vals = zip(self.loader("high", asset_list=self.asset_list, idx=self.idx),
                   self.loader("low", asset_list=self.asset_list, idx=self.idx),
                   self.loader("close", asset_list=self.asset_list, idx=self.idx))
        H, L, C = zip(*vals)

        high, low, close = H[int(is_test)], L[int(is_test)], C[int(is_test)]


        if is_test or overlapping_batches:
            # Overlapping batches
            shuffle_ids = np.arange(start=0, stop=(len(high) - self.bptt))
            if not is_test and randomize_batches:
                np.random.shuffle(shuffle_ids)

        else:
            # Non-Overlapping batches
            shuffle_ids = np.arange(start=0, stop=(len(high) - self.bptt) // (self.bptt + 1))
            shuffle_ids *= self.bptt
            if not is_test and randomize_batches:
                np.random.shuffle(shuffle_ids)

        num_batches = len(shuffle_ids) // self.bsz
        num_batches = (num_batches if len(shuffle_ids) % self.bsz == 0 else num_batches + 1)

        for batch_idx in range(num_batches):
            s_ids = shuffle_ids[batch_idx * self.bsz: (batch_idx + 1) * self.bsz]

            # X.shape => (bptt X num_features X num_assets), y.shape => (bptt X num_assets + 1)
            X = []
            y = []

            for s_idx in s_ids:
                h_batch = high.iloc[s_idx: s_idx + self.bptt, :]
                l_batch = low.iloc[s_idx: s_idx + self.bptt, :]
                c_batch = close.iloc[s_idx: s_idx + self.bptt, :]

                # Convert all three to matrix form, out.shape => (num_features X bptt X num_assets)
                x_out = np.array([h_batch.as_matrix(), l_batch.as_matrix(), c_batch.as_matrix()])

                #(saatvik): Why will this happen??
                if len(h_batch) != self.bptt:
                    continue

                # Relative price change(num_assets + 1(indicating change in cash==constant))
                # Note that last column indicates cash
                y_out = close.iloc[s_idx + 1: s_idx + self.bptt + 1, :].as_matrix() / c_batch
                y_out = np.pad(y_out , [(0, 0), (0,1)], constant_values=1, mode="constant")
                x_out = x_out.transpose([1, 0, 2])
                X.append(x_out)
                y.append(y_out)
            # X[0], y[0] is a zero pad meant for vstack convenience
            X = np.array(X)
            y = np.array(y)

            assert len(X.shape) == 4, "X shape: {}".format(X.shape)
            assert X.shape[1] == self.bptt and X.shape[2] == 3 and X.shape[3] == len(self.asset_list), "X shape: {}".format(X.shape)
            assert y.shape[1] == self.bptt and y.shape[2] == len(self.asset_list) + 1
            assert len(X) == len(y)
            yield X, y


    def loader(self, name, asset_list = ASSET_LIST, idx = 0):
        """
        :param name: feature name. Can take values like "open"/"close", "high", "low"
        :param asset_list: Number of assets to keep
        :return: (Generator iterates for all the date ranges) Dataframe where each column is an asset. Number of records is for the entire date range.
        """
        if isinstance(self.dp , crypto.DataPreprocess):
            return self.dp.load_train_test(asset_name = asset_list, feature_type = name, idx=idx, path=self.data_path)
        else: # is stock loader
            # asset_list = set(list(map(str.upper,asset_list)))
            # print(asset_list)
            name = str.upper(name)
            return self.dp.load_train_test(asset_name = asset_list,
                                           feature_type = name, path=self.data_path,
                                           train_test_ratio=0.8)


# def test_load_stocks():
#     """
#     Function used to test the load_Stocks
#     :return:
#     """
#     dp = stocks.DataPreprocess()
#     # dp.load_preprocessed('../../dataset/stock_data_Preprocessed/preprocessed_Stock_data.csv')
#     dp.load_preprocessed('../../dataset/5yrs_preprocessed/all_5_yrs_processed.csv')
#     stocks_name = dp.asset_names()
#     print(stocks_name)
#     batch_obj = Batchifier(data_path='../../dataset/5yrs_preprocessed/all_5_yrs_processed.csv',
#                            asset_list=stocks_name, data_preprocess=dp,
#                            idx=0,
#                            )
#
# def test_load_crypto():
#     batch_obj = Batchifier(data_path='../../dataset/Poloneix_Preprocessednew',
#                            asset_list=ASSET_LIST,
#                            idx=0,
#                            )
#     return batch_obj
#
# if __name__ == "__main__":
#     # batch_obj = test_load_stocks()
#     batch_obj = test_load_crypto()
#     for x, y in batch_obj.load_batch(overlapping_batches=True, randomize_batches=False, is_test=False):
#         print(x.shape, y.shape)
