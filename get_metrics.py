import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

class get_metrics:
    def __init__(self, dt_range):
        self.dt_range = dt_range

    def graph_output(self, y, name):
        fig = plt.figure()
        ts = pd.Series(y,name=name)
        ts.plot(x = "Time", y = name, title = "{} for Test Data from {} to {}".format(name, self.dt_range.split("_")[0], self.dt_range.split("_")[1]))
        fig.savefig("{}_{}".format(name, self.dt_range))
        plt.close()

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

    def apv_multiple_asset(self, rp_vector, weights, pv_0 = 1, get_graph = False, tag=""):
        """
        :param rp_vector: time_steps X num_assets
        :param weights: time_steps X num_assets
        :return: MDD, final portfolio value, SR
        """
        assert rp_vector.shape == weights.shape, "Dimension Mismatch!, True labels {} != Weights {}".format(rp_vector.shape, weights.shape)


        # final portfolio value => scalar. At any time t, fAPV = p_initial * _prod { rp_{t} * w_{t - 1}}
        portfolio_val = pv_0 * np.product([np.dot(r , w) for (r, w) in zip(rp_vector, weights)])


        # time_steps = total_time_steps
        time_steps = rp_vector.shape[0] + 1

        del_portfolio = np.ndarray(shape = (time_steps,))
        # Initial value of portfolio = pv_0
        del_portfolio[0] = pv_0

        for idx in range(1, time_steps):
            del_portfolio[idx] = del_portfolio[idx - 1] * np.dot(rp_vector[idx - 1, :] , weights[idx - 1, :])


        sharpe_ratio = self.sr_vals_multiple_asset(rp_vector, weights)
        mdd = self.mdd_vals_multiple_asset(del_portfolio)

        if get_graph:
            self.graph_output(del_portfolio, "APV")
            self.graph_output(sharpe_ratio, "Sharpe Ratio")

        print("MDD = {}, fAPV = {}".format(mdd, portfolio_val))
        print("Saving allocation weights....")
        np.save("allocation_wts_{}_{}.npy".format(portfolio_val, tag), weights)
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
        return max(drawdowns[1:]) if len(drawdowns) > 0 else 0.0

