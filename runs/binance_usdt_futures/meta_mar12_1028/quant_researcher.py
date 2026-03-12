"""
[Quant Researcher]
바이낸스 USDT 무기한 선물 단면 모멘텀(Cross-Sectional Momentum) 알파 모델.

가설: 과거 20일 위험조정 수익률 상위 종목을 롱, 하위 종목을 숏.
     순수 단면(market-neutral) 구조로 시장 전반의 베타 제거.

포지션 사이징:
  - 등가중: 각 포지션 ±(1/N_LONG 또는 1/N_SHORT)
  - total gross exposure = 2.0x (롱 1.0x + 숏 1.0x)
  - 개별 변동성 급등 시 gross 비례 축소로 리스크 관리

Look-ahead bias 방지:
  - weights[t] = 종가[t] 기준 신호 → backtest 엔진에서 shift(1) 처리
"""
import pandas as pd
import numpy as np

N_LONG  = 3     # 롱 종목 수
N_SHORT = 3     # 숏 종목 수
# 포트폴리오 일별 변동성 목표: 5%/day (연율화 약 79%)
# → 개별 포지션 사이즈는 1/N으로 고정, 포트폴리오 전체 크기로 조절
TARGET_DAILY_PORT_VOL = 0.05


def research_alpha(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    피처 딕셔너리를 입력받아 타겟 비중을 반환합니다.

    Returns:
        pd.DataFrame  shape=(날짜, 티커)
            +1/N_LONG  : 롱 포지션
            -1/N_SHORT : 숏 포지션
            0          : 보유 없음
        ※ backtest 엔진에서 shift(1)하여 익일 적용
    """
    if not features:
        return pd.DataFrame()

    tickers = sorted(features.keys())

    # 20일 위험조정 모멘텀 행렬 (신호 방향 결정)
    signal_df = pd.DataFrame(
        {t: features[t]['mom_risk_adj_20d'] for t in tickers}
    ).sort_index()

    # 포트폴리오 20일 실현 변동성 추정 (비중 크기 조절용)
    # 단면 포트폴리오 변동성 ≈ 개별 변동성 평균 (대략적 근사)
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

        # 기본 등가중 배분
        for s in longs:
            weights.loc[date, s] = 1.0 / N_LONG
        for s in shorts:
            weights.loc[date, s] = -1.0 / N_SHORT

        # 포트폴리오 변동성 스케일링
        # 포트폴리오 예상 일별 변동성 ≈ avg_vol × gross
        if date in avg_vol_df.index:
            avg_vol = avg_vol_df.loc[date]
            if avg_vol > 1e-6:
                gross = weights.loc[date].abs().sum()   # 현재 2.0
                port_vol = avg_vol * gross
                if port_vol > TARGET_DAILY_PORT_VOL:
                    scale = TARGET_DAILY_PORT_VOL / port_vol
                    weights.loc[date] = weights.loc[date] * scale

    return weights
