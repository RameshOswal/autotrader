## Metrics for evaluating performance

We are starting with three metrics that are widely used to analyze the performance of a portfolio management system.

### Accumulated Portfolio Value
Accumulated Portfolio Value (APV) or the final Accumulated Portfolio Value (fAPV) at any time **t** is the ratio of the portfolio value at time **t** to the initial portfolio value, ie. APV = $p_{t}/p_{0}$. Similarly, fAPV = $p_{f}/p_{0}$. For our experiments, we assume our initial portfolio value to be 1. 
Assuming we have **m** assets, to calculate portfolio value at ant time **t**, we first calculate the relative change in opening prices ($v_{t}$) at time $t$ and $t - 1$. This we represent as $y_{t}$.
$$
y_{t} = v_{t}/v_{t - 1} = \bigg[ \frac{v_{1,t}}{v_{1,t-1}}, \frac{v_{2,t}}{v_{2,t-1}}, ..., \frac{v_{m,t}}{v_{m,t-1}} \bigg]
$$  
To calculate $p_{t}$ and $p_f$, we use the portfolio value at the previous time step ($p_{t-1}$) along with the weights vector $w_t$.
$$
p_t = p_{t-1} ( y_t . w_t )
$$
$$
fAPV = p_f = p_0 \prod_{i = 1}^{i = t_f} y_i . w_i
$$


### Sharpe Ratio
A major disadvantage of APV is that it does not measure the risk factors, since it merely sums up all the periodic returns without considering fluctuation in these returns. A second metric, the Sharpe ratio (SR) is used to take risk into account. The ratio is a risk adjusted mean return, defined as the average of the risk-free return by its deviation,
$$
SR = \frac{E[p_T- p_F] }{std(p_T - p_F)}
$$
where $p_t$ are the portfolio values at time **t**. 

### Maximum Drawdown
Maximum drawdown is the maximum value among all the losses. Here, a loss is defined as the difference between a peak and a valley until a next peak is attained.
$$
MDD = \max\frac{p_P- p_V }{p_P}
$$

