"""
[Quant Researcher]
크로스-섹셔널 20일 모멘텀 + 역변동성(Inverse-Vol) 스케일링 전략.

로직:
  1. 각 날짜별로 심볼들의 20일 수익률을 계산한다.
  2. 크로스-섹셔널로 랭킹: 상위 TOP_N 롱, 하위 BOTTOM_N 숏.
  3. 각 포지션 크기는 20일 변동성의 역수에 비례 (Inverse-Vol weighting).
  4. 롱 합계 = LONG_WEIGHT, 숏 합계 = -SHORT_WEIGHT (gross = 1.5x).
  5. 신호 없는 날(NaN 과다 등)은 0 포지션.

Look-ahead bias 방지: 오늘 종가로 계산한 시그널을 내일 수익률에 적용.
"""
import pandas as pd
import numpy as np

# --- 하이퍼파라미터 ---
MOM_WINDOW  = 20    # 모멘텀 계산 기간 (일) — 그리드 서치 후 20일이 최적
VOL_WINDOW  = 20    # 변동성 계산 기간 (일)
TOP_N       = 4     # 롱 심볼 수 (10개 중 4개)
BOTTOM_N    = 4     # 숏 심볼 수 (10개 중 4개)
LONG_WEIGHT  = 0.75  # 롱 사이드 합계 (gross 절반, 총 gross 1.5x)
SHORT_WEIGHT = 0.75  # 숏 사이드 합계


def research_alpha(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    크로스-섹셔널 모멘텀 시그널을 계산해 일별 포지션 비중 DataFrame을 반환한다.

    Returns
    -------
    pd.DataFrame  shape=(T, N), values in [-1, +1] (정규화 완료)
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

    # ---- 시그널 계산 (벡터화) ----
    # 20일 모멘텀: close[t] / close[t-20] - 1
    mom = close_df.pct_change(MOM_WINDOW)

    # 20일 변동성 (rolling std of daily returns)
    vol = ret_df.rolling(VOL_WINDOW).std().clip(lower=1e-6)
    inv_vol = 1.0 / vol

    # 크로스-섹셔널 랭킹 (매일, axis=1)
    ranks = mom.rank(axis=1, ascending=True, na_option="keep")

    n_sym = len(symbols)
    long_mask  = ranks >= (n_sym - TOP_N  + 1)
    short_mask = ranks <= BOTTOM_N

    # 역변동성 가중치
    long_raw  = long_mask.astype(float) * inv_vol   # NaN → 0
    short_raw = short_mask.astype(float) * inv_vol

    # 롱/숏 합계로 정규화 후 스케일 적용
    long_sum  = long_raw.sum(axis=1).replace(0, np.nan)
    short_sum = short_raw.sum(axis=1).replace(0, np.nan)

    long_w  = long_raw.divide(long_sum,  axis=0).fillna(0.0) * LONG_WEIGHT
    short_w = short_raw.divide(short_sum, axis=0).fillna(0.0) * SHORT_WEIGHT

    # 숏 시그널: 하위 BOTTOM_N 심볼에 음수 비중 부여 (양방향 매매 구현)
    short_signal = -short_w   # 하위 모멘텀 심볼 숏 포지션 (값 < 0)
    weights = long_w + short_signal

    # NaN 이 하나라도 있는 행(워밍업 구간)은 0으로
    has_nan = mom.isna().any(axis=1) | vol.isna().any(axis=1)
    weights.loc[has_nan] = 0.0

    return weights.fillna(0.0)
