import math

import pandas as pd


ATR_BENCHMARK = {
    "high": [48.70, 48.72, 48.90],
    "low": [47.79, 48.14, 48.39],
    "close": [48.16, 48.61, 48.75],
    "true_range": [0.91, 0.58, 0.51],
    "atr_sma_3": [None, None, (0.91 + 0.58 + 0.51) / 3],
    "atr_ema_3": [0.91, 0.745, 0.6275],
}


BLACK_SCHOLES_BENCHMARKS = [
    {
        "option_type": "call",
        "spot": 100.0,
        "strike": 100.0,
        "time": 1.0,
        "rate": 0.05,
        "volatility": 0.2,
        "price": 10.450583572185565,
    },
    {
        "option_type": "put",
        "spot": 100.0,
        "strike": 100.0,
        "time": 1.0,
        "rate": 0.05,
        "volatility": 0.2,
        "price": 5.573526022256971,
    },
    {
        "option_type": "call",
        "spot": 100.0,
        "strike": 105.0,
        "time": 0.5,
        "rate": 0.01,
        "volatility": 0.2,
        "price": 3.7988068633033976,
    },
]


REALIZED_VOLATILITY_BENCHMARK = {
    "index": pd.DatetimeIndex(
        [
            "2024-01-01 09:30",
            "2024-01-01 09:31",
            "2024-01-01 09:32",
            "2024-01-02 09:30",
            "2024-01-02 09:31",
        ]
    ),
    "log_returns": [0.1, 0.2, 0.3, 0.4],
    "expected_daily": pd.Series(
        [math.sqrt(0.1**2 + 0.2**2), math.sqrt(0.3**2 + 0.4**2)],
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    ),
}
