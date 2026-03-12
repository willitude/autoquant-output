"""기술적 지표 라이브러리.

에이전트가 새로운 지표를 추가하며 확장한다.
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def bollinger_bands(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, window)
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    return returns.rolling(window).std()


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    return volume / sma(volume, window)


def engineer_base_features(close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """price + volume 시리즈로부터 기본 피처 테이블을 생성한다."""
    returns = close.pct_change()
    feat = pd.DataFrame(index=close.index)
    feat["close"] = close
    feat["volume"] = volume
    feat["returns"] = returns
    feat["log_returns"] = np.log(close / close.shift(1))
    feat["volatility"] = rolling_volatility(returns)
    feat["sma_20"] = sma(close, 20)
    feat["sma_50"] = sma(close, 50)
    feat["sma_200"] = sma(close, 200)
    feat["rsi"] = rsi(close)
    bb_upper, bb_mid, bb_lower = bollinger_bands(close)
    feat["bb_upper"] = bb_upper
    feat["bb_mid"] = bb_mid
    feat["bb_lower"] = bb_lower
    feat["bb_width"] = (bb_upper - bb_lower) / bb_mid
    feat["volume_sma"] = sma(volume, 20)
    feat["volume_ratio"] = volume_ratio(volume)
    return feat
