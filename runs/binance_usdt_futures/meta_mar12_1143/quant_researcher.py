"""
[Quant Researcher]
가설 #2: 멀티-윈도우 크로스-섹셔널 모멘텀 + 역변동성 + 시장 레짐 필터.

로직:
  1. 5/10/20일 모멘텀의 평균 크로스-섹셔널 랭킹으로 신호 안정성 향상.
  2. 레짐 필터: 심볼 평균 BTC+ETH 가격이 200일 SMA 위이면 full exposure,
     아래이면 gross를 REGIME_SCALE 비율로 축소 (하락장 손실 억제).
  3. 각 포지션 크기는 20일 변동성의 역수에 비례 (Inverse-Vol weighting).
  4. 롱 합계 = LONG_WEIGHT, 숏 합계 = -SHORT_WEIGHT (gross ≈ 1.5x 기준).
  5. 신호 없는 날(NaN 과다 등)은 0 포지션.

Look-ahead bias 방지: 오늘 종가로 계산한 시그널을 내일 수익률에 적용.
"""
import pandas as pd
import numpy as np

# --- 하이퍼파라미터 ---
MOM_WINDOWS  = [5, 10, 20]  # 멀티-윈도우 모멘텀
VOL_WINDOW   = 20            # 변동성 계산 기간 (일)
TOP_N        = 3             # 롱 심볼 수
BOTTOM_N     = 3             # 숏 심볼 수
LONG_WEIGHT  = 0.75          # 롱 사이드 합계 (gross 절반)
SHORT_WEIGHT = 0.75          # 숏 사이드 합계 (gross 절반)
REGIME_SCALE = 0.5           # 하락 레짐 시 노출 비율 (0=완전 회피, 1=동일)
REGIME_SMA   = 200           # 레짐 판단 SMA 기간 (일)
REGIME_SYMS  = ["BTCUSDT", "ETHUSDT"]  # 레짐 판단에 사용할 심볼


def _regime_scale(close_df: pd.DataFrame) -> pd.Series:
    """
    시장 레짐 스케일 시리즈 반환.
    BTCUSDT + ETHUSDT 가격이 200일 SMA 위: 1.0, 아래: REGIME_SCALE.
    두 심볼 모두 사용 불가이면 항상 1.0 (안전 기본값).
    """
    available = [s for s in REGIME_SYMS if s in close_df.columns]
    if not available:
        return pd.Series(1.0, index=close_df.index)

    # 각 심볼이 200일 SMA 위에 있으면 1, 아래면 0
    above = pd.DataFrame(index=close_df.index)
    for sym in available:
        sma200 = close_df[sym].rolling(REGIME_SMA).mean()
        above[sym] = (close_df[sym] >= sma200).astype(float)

    # 둘 다 SMA 위이면 1.0, 하나라도 아래면 REGIME_SCALE
    consensus = above.mean(axis=1)  # 0 ~ 1
    scale = consensus.apply(lambda x: 1.0 if x >= 0.5 else REGIME_SCALE)
    return scale


def research_alpha(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    멀티-윈도우 크로스-섹셔널 모멘텀 + 레짐 필터 시그널.

    Returns
    -------
    pd.DataFrame  shape=(T, N), values in [-SHORT_WEIGHT, +LONG_WEIGHT]
        index  : 날짜
        columns: 심볼
    """
    if not features:
        return pd.DataFrame()

    symbols = list(features.keys())

    # 공통 날짜 인덱스 구성
    common_idx = features[symbols[0]].index
    for s in symbols[1:]:
        common_idx = common_idx.intersection(features[s].index)
    common_idx = common_idx.sort_values()

    # 종가 행렬
    close_df = pd.DataFrame(
        {s: features[s]["close"] for s in symbols}
    ).reindex(common_idx)

    # 일별 수익률 행렬
    ret_df = close_df.pct_change()

    # ---- 멀티-윈도우 모멘텀 크로스-섹셔널 합산 랭킹 ----
    rank_sum = pd.DataFrame(0.0, index=common_idx, columns=symbols)
    valid_count = 0
    for w in MOM_WINDOWS:
        mom_w = close_df.pct_change(w)
        r_w   = mom_w.rank(axis=1, ascending=True, na_option="keep")
        # NaN이 있는 행은 해당 윈도우를 건너뜀
        mask_valid = ~mom_w.isna().any(axis=1)
        rank_sum.loc[mask_valid] += r_w.loc[mask_valid].fillna(0)
        valid_count += 1

    avg_rank = rank_sum / valid_count  # NaN 행에서는 0이 되지만 이후 has_nan 처리

    # 20일 변동성 (rolling std of daily returns)
    vol     = ret_df.rolling(VOL_WINDOW).std().clip(lower=1e-6)
    inv_vol = 1.0 / vol

    # 크로스-섹셔널 랭킹
    final_ranks = avg_rank.rank(axis=1, ascending=True, na_option="keep")
    n_sym       = len(symbols)
    long_mask   = final_ranks >= (n_sym - TOP_N  + 1)
    short_mask  = final_ranks <= BOTTOM_N

    # 역변동성 가중치
    long_raw  = long_mask.astype(float) * inv_vol
    short_raw = short_mask.astype(float) * inv_vol

    # 롱/숏 합계로 정규화 후 스케일 적용
    long_sum  = long_raw.sum(axis=1).replace(0, np.nan)
    short_sum = short_raw.sum(axis=1).replace(0, np.nan)

    long_w  = long_raw.divide(long_sum,  axis=0).fillna(0.0) * LONG_WEIGHT
    short_w = short_raw.divide(short_sum, axis=0).fillna(0.0) * SHORT_WEIGHT

    # 숏 시그널: 하위 BOTTOM_N 심볼에 음수 비중 부여 (양방향 매매 구현)
    short_signal = -short_w   # 하위 모멘텀 심볼 숏 포지션 (값 < 0)
    weights = long_w + short_signal

    # ---- 시장 레짐 필터 적용 ----
    regime = _regime_scale(close_df)
    weights = weights.multiply(regime, axis=0)

    # NaN 이 하나라도 있는 행(워밍업 구간)은 0으로
    warmup_mask = (
        close_df.pct_change(max(MOM_WINDOWS)).isna().any(axis=1)
        | vol.isna().any(axis=1)
        | regime.isna()
    )
    weights.loc[warmup_mask] = 0.0

    return weights.fillna(0.0)
