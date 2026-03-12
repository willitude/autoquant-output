"""성과 지표 계산.

system_operator에서 import하여 사용한다.
"""

import numpy as np
import pandas as pd


def calculate_metrics(returns: pd.Series) -> dict:
    """수익률 시리즈에서 주요 성과 지표를 계산한다."""
    if len(returns) == 0 or returns.std() == 0:
        return {
            "sharpe": 0.0,
            "total_return": 0.0,
            "annual_return": 0.0,
            "max_dd": 0.0,
            "volatility": 0.0,
            "win_rate": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
        }

    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0

    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()

    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.0

    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 0.0
    sortino = returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0.0
    calmar = annual_return / abs(max_dd) if max_dd < 0 else 0.0

    return {
        "sharpe": sharpe,
        "total_return": total_return * 100,
        "annual_return": annual_return * 100,
        "max_dd": max_dd * 100,
        "volatility": volatility * 100,
        "win_rate": win_rate * 100,
        "sortino": sortino,
        "calmar": calmar,
    }
