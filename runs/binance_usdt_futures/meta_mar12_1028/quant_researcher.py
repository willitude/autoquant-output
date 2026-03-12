"""
[Quant Researcher]
바이낸스 USDT 무기한 선물 단면 모멘텀 알파 (v1 baseline 복원).

가설 1 (검증 완료): 20일 위험조정 단면 모멘텀
  val_sharpe = 0.416 (2023-01-01 ~ 2025-03-11)

Look-ahead bias 방지:
  - weights[t] = 종가[t] 기준 신호 → backtest 엔진에서 shift(1) 처리
"""
import pandas as pd
import numpy as np

N_LONG  = 3
N_SHORT = 3
TARGET_DAILY_PORT_VOL = 0.05   # 포트폴리오 일별 변동성 목표


def research_alpha(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    20일 위험조정 모멘텀 단면 신호를 반환합니다.

    Returns:
        pd.DataFrame  shape=(날짜, 티커)
            +1/N_LONG  : 롱 포지션
            -1/N_SHORT : 숏 포지션
        ※ backtest 엔진에서 shift(1)하여 익일 적용
    """
    if not features:
        return pd.DataFrame()

    tickers = sorted(features.keys())

    signal_df = pd.DataFrame(
        {t: features[t]['mom_risk_adj_20d'] for t in tickers}
    ).sort_index()

    avg_vol_df = pd.DataFrame(
        {t: features[t]['vol_20d'] for t in tickers}
    ).sort_index().mean(axis=1)

    weights = pd.DataFrame(0.0, index=signal_df.index, columns=tickers)

    for date, row in signal_df.iterrows():
        valid = row.dropna()
        if len(valid) < N_LONG + N_SHORT:
            continue

        ranked = valid.sort_values()
        shorts = ranked.index[:N_SHORT].tolist()
        longs  = ranked.index[-N_LONG:].tolist()

        for s in longs:
            weights.loc[date, s] = 1.0 / N_LONG
        for s in shorts:
            weights.loc[date, s] = -1.0 / N_SHORT

        # 포트폴리오 변동성 스케일링
        if date in avg_vol_df.index:
            avg_vol = avg_vol_df.loc[date]
            if avg_vol > 1e-6:
                gross    = weights.loc[date].abs().sum()
                port_vol = avg_vol * gross
                if port_vol > TARGET_DAILY_PORT_VOL:
                    scale = TARGET_DAILY_PORT_VOL / port_vol
                    weights.loc[date] = weights.loc[date] * scale

    return weights
