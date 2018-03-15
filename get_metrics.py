__author__ = "deeptrader"

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

class get_metrics:
    def __init__(self): pass

    def graph_output(self, y, name):
        fig = plt.figure()
        ts = pd.Series(y,name=name)
        ts.plot()
        fig.savefig(name)
        # plt.show()

    def apv_single_asset(self, y_true, weights, pv_0 = 1, get_graph = False):
        """
        :param y_true: true closing/opening value of the stock/crypto
        :param weights: denotes decision taken at each time step
        :param lbl_dict: mappings required for dictionary [lower value when purchasing and higher when selling]
        :return: final portfolio value
        """
        assert len(y_true) == len(weights), "Dimension Mismatch!, Length of True labels {} != Decision length {}".format(len(y_true), len(weights))

        rp_vector = np.array([1] + [np.divide(y , x) for (x, y) in zip(y_true, y_true[1:])])


        portfolio_val = pv_0 * np.product([np.multiply(r , w) for (r, w) in zip(rp_vector, weights)], axis = 0)

        # Finding change in portfolio values for sharpe ratio
        del_portfolio = np.ndarray(shape = (rp_vector.shape[0] + 1,))
        del_portfolio[0] = 1.0
        for idx in range(1, del_portfolio.shape[0]):
            del_portfolio[idx] = del_portfolio[idx - 1] * (rp_vector[idx - 1] * weights[idx - 1])

        if get_graph:
            self.graph_output(del_portfolio, "ratios")
            self.graph_output(y_true, "actual_price")

        mdd = self.mdd_vals(del_portfolio)

        print("MDD = {}, fAPV = {}".format(mdd, portfolio_val))

        return mdd, portfolio_val

    def mdd_vals(self, del_pv):
        """
        :param del_pv: changes in portfolio value
        :return: MDD
        """
        pv_vals = [del_pv[0]]
        drawdown_lst = [0]
        max_val = 0
        for idx in range(1, del_pv.shape[0]):
            pv_vals.append(pv_vals[idx - 1] * del_pv[idx])
            if pv_vals[idx] > max_val:
                max_val = pv_vals[idx]
            else: drawdown_lst.append( (max_val - pv_vals[idx]) / max_val)
        return max(drawdown_lst)

    def apv_multiple_asset(self, y_true, weights, pv_0 = 1, get_graph = False):
        """
        :param y_true: true closing/opening value all the stocks/crypto
        :param weights: denotes decision taken at each time step for all stock/crypto
        :return: MDD, final portfolio value, SR
        """
        assert len(y_true) == len(weights) - 1 and len(y_true[0]) == len(weights[0]) - 1, "Dimension Mismatch!, Length of True labels {} != Decision length {}".format(len(y_true), len(weights))

        y_mod = np.ndarray(shape = weights.shape)
        y_mod[:,0] = 1
        y_mod[:-1, 1:] = y_true
        y_mod[-1, :] = 1

        rp_vector = np.array([[1] * weights.shape[1]] + [np.divide(y , x) for (x, y) in zip(y_mod[:, :], y_mod[1:, :])])

        portfolio_val = pv_0 * np.product([np.dot(r , w) for (r, w) in zip(rp_vector, weights)])

        # Finding change in portfolio values for sharpe ratio
        del_portfolio = np.ndarray(shape = (rp_vector.shape[0] + 1,))
        del_portfolio[0] = 1.0
        for idx in range(1, del_portfolio.shape[0]):
            del_portfolio[idx] = np.dot(del_portfolio[idx - 1] , np.dot(rp_vector[idx - 1, :] , weights[idx - 1, :]))


        sharpe_ratio = self.sr_vals_multiple_asset(rp_vector, weights)
        mdd = self.mdd_vals_multiple_asset(del_portfolio)

        if get_graph:
            self.graph_output(del_portfolio, "ratios")
            self.graph_output(sharpe_ratio, "sharpe_ratio")

        print("MDD = {}, fAPV = {}".format(mdd, portfolio_val))

        return mdd, portfolio_val

    def sr_vals_multiple_asset(self, rp, weights):
        """
        :param rp: relative price vector
        :param weights: weight vector
        :return: sharpe ratio at each time step
        """
        out = np.ndarray(shape = (rp.shape[0],))
        for sr in range(out.shape[0]):
            rho_val = np.multiply(rp[sr, :] , weights[sr, :]) - 1
            out[sr] = np.mean(rho_val) / np.std(rho_val)
        return out

    def mdd_vals_multiple_asset(self, del_pv):
        """
        :param del_pv: changes in portfolio value
        :return: MDD
        """
        drawdowns = []
        trough = peak = del_pv[0]
        for idx in range(1, del_pv.shape[0]):
            if del_pv[idx] <= peak:
                trough = min(trough, del_pv[idx])
            else:
                drawdowns.append((trough - peak) / peak)
                peak = del_pv[idx]
        return max(drawdowns[1:])

"""
if __name__ == "__main__":
    test = get_metrics()
    test.apv_single_asset(np.random.uniform(low = 2, high = 10, size = 125), np.random.randint(low = 1, high=4, size=125), lbl_dict={1 : 0.99, 2: 1, 3: 1.01}, get_graph=True)
"""
if __name__ == "__main__":
    test = get_metrics()
    test.apv_multiple_asset(np.random.uniform(low = 0, high = 1, size = [99, 3]), np.random.uniform(low = 0, high=1, size=[100, 4]), get_graph=True)