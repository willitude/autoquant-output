"""
[Portfolio Manager]
포지션 비중 DataFrame을 받아 레버리지 제약을 적용한다.

gross exposure = sum(|w_i|) 이 max_leverage 를 초과하면 균등 스케일링으로 조정한다.
"""
import pandas as pd
import numpy as np


def allocate_capital(
    signals: pd.DataFrame,
    max_leverage: float,
    params: dict,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    signals      : pd.DataFrame  일별 포지션 비중 (index=날짜, columns=심볼)
    max_leverage : float         허용 최대 gross 레버리지
    params       : dict          예비 파라미터 (현재 미사용)

    Returns
    -------
    pd.DataFrame  레버리지 제약이 적용된 최종 비중
    """
    if signals is None or (isinstance(signals, pd.DataFrame) and signals.empty):
        return pd.DataFrame()

    weights = signals.copy()
    gross = weights.abs().sum(axis=1)

    # gross 가 max_leverage 초과하는 행만 스케일다운
    excess = gross > max_leverage
    if excess.any():
        scale = (max_leverage / gross[excess]).clip(upper=1.0)
        weights.loc[excess] = weights.loc[excess].multiply(scale, axis=0)

    return weights
