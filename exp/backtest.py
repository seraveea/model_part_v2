import qlib
import pandas as pd
from qlib.utils.time import Freq
from qlib.utils import flatten_dict
from qlib.backtest import backtest, executor
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy


def backtest_loop(data, model_name, EXECUTOR_CONFIG, backtest_config):
    data = data[[model_name]]
    data.columns = [['score']]
    # init qlib
    # Benchmark is for calculating the excess return of your strategy.
    # Its data format will be like **ONE normal instrument**.
    # For example, you can query its data with the code below
    # `D.features(["SH000300"], ["$close"], start_time='2010-01-01', end_time='2017-12-31', freq='day')`
    # It is different from the argument `market`, which indicates a universe of stocks (e.g. **A SET** of stocks like csi300)
    # For example, you can query all data from a stock market with the code below.
    # ` D.features(D.instruments(market='csi300'), ["$close"], start_time='2010-01-01', end_time='2017-12-31', freq='day')`

    FREQ = "day"
    STRATEGY_CONFIG = {
    "topk": 30,
    "n_drop": 5,
    # pred_score, pd.Series
    "signal": data,
    }
    # strategy object
    strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
    # executor object
    executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)
    # backtest
    portfolio_metric_dict, indicator_dict = backtest(executor=executor_obj, strategy=strategy_obj, **backtest_config)
    analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))
    # backtest info
    report_normal, positions_normal = portfolio_metric_dict.get(analysis_freq)

    # analysis
    analysis = dict()
    analysis["excess_return_without_cost"] = risk_analysis(
    report_normal["return"] - report_normal["bench"], freq=analysis_freq
    )
    analysis["excess_return_with_cost"] = risk_analysis(
    report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=analysis_freq
    )

    analysis_df = pd.concat(analysis)  # type: pd.DataFrame
    # log metrics
    analysis_dict = flatten_dict(analysis_df["risk"].unstack().T.to_dict())
    # print out results
    benchmark_p = risk_analysis(report_normal["bench"], freq=analysis_freq)
    excess_return_wo_cost = analysis["excess_return_without_cost"]
    excess_return_w_cost = analysis["excess_return_with_cost"]
    return benchmark_p, excess_return_wo_cost, excess_return_w_cost


if __name__ == "__main__":
    data = pd.read_pickle('pred_output/all_in_one_incre.pkl')
    qlib.init(provider_uri="../qlib_data/cn_data")
    data = data.dropna()
    CSI300_BENCH = "SH000300"
    EXECUTOR_CONFIG = {
    "time_per_step": "day",
    "generate_portfolio_metrics": True,
    }
    FREQ = 'day'
    backtest_config = {
    "start_time": "2022-06-01",
    "end_time": "2023-06-30",
    "account": 100000000,
    "benchmark": CSI300_BENCH,  # "benchmark": NASDAQ_BENCH,
    "exchange_kwargs": {
    "freq": FREQ,
    "limit_threshold": 0.095,
    "deal_price": "close",
    "open_cost": 0.00005,
    "close_cost": 0.00015,
    "min_cost": 5,
    },}
    # model_pool = ['GRU','LSTM','GATs','MLP','ALSTM','HIST','ensemble_retrain','RSR_hidy_is','KEnhance','SFM',
    #               'ensemble_no_retrain', 'Perfomance_based_ensemble', 'average', 'blend', 'dynamic_ensemble']
    model_pool = ['GRU', 'LSTM', 'GATs', 'MLP', 'ALSTM', 'SFM']
    pd_pool = []
    for model in model_pool:
        symbol = model+'_score'
        benchmark, er_wo_cost, er_w_cost = backtest_loop(data, symbol, EXECUTOR_CONFIG, backtest_config)
        er_w_cost.columns = [[model+'_with_cost']]
        er_wo_cost.columns = [[model+'_without_cost']]
        benchmark.columns = [['benchmarks']]
        if len(pd_pool) == 0:
            pd_pool.extend([benchmark, er_w_cost, er_wo_cost])
        else:
            pd_pool.extend([er_w_cost, er_wo_cost])
    df = pd.concat(pd_pool, axis=1)
    df = df.T
    df.to_pickle('pred_output/backtest_incre_12.pkl')