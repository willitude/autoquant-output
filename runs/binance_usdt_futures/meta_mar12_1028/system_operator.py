"""
[System Operator]
전체 파이프라인(Data → Research → Portfolio → Backtest)을 실행합니다.

사용법: uv run system_operator.py

Train:      2021-01-01 ~ 2022-12-31 (24개월)
Validation: 2023-01-01 ~ 2025-03-11 (약 26개월)

비용 모델 (constraints.json 기준):
  수수료    4 bps/편도
  슬리피지  2 bps/편도  → 편도 합계 6 bps
  펀딩      1 bps/8시간 = 3 bps/일 (gross exposure 기준)
"""
import os
import json
import time
import pandas as pd
import numpy as np

from data_engineer   import collect_and_engineer
from quant_researcher import research_alpha
from portfolio_manager import allocate_capital
from risk_manager    import check_risk_limits

# ---------------------------------------------------------------------------
# 경로 및 제약 조건 로드
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_HERE, 'constraints.json'), 'r') as f:
    CONSTRAINTS = json.load(f)

C = CONSTRAINTS['constraints']
COMMISSION_BPS  = C['commission_bps']    # 4
SLIPPAGE_BPS    = C['slippage_bps']      # 2
FUNDING_BPS_8H  = C['funding_rate_bps_per_8h']  # 1
MAX_LEVERAGE    = C['max_leverage']      # 3

TRAIN_START = '2021-01-01'
TRAIN_END   = '2022-12-31'
VAL_START   = '2023-01-01'
VAL_END     = '2025-03-11'


# ---------------------------------------------------------------------------
# 백테스트 엔진
# ---------------------------------------------------------------------------
def run_backtest(
    weights: pd.DataFrame,
    features: dict[str, pd.DataFrame],
    period_start: str,
    period_end: str,
) -> pd.Series:
    """
    일별 순수익률(net PnL) 시계열을 계산합니다.

    비용 처리:
      - 거래비용 = |Δweight| × (수수료 + 슬리피지) 편도 → 왕복 2배
        (Δweight는 하루 한 번 리밸런싱 기준)
      - 펀딩비용 = |weight| × 3 × FUNDING_BPS_8H / 10000  (하루 3회 정산)
      - look-ahead bias 방지: weights[t-1] 를 ret[t] 에 적용 (shift 1)
    """
    tickers = [c for c in weights.columns if c in features]

    ret_df = pd.concat(
        {t: features[t]['ret_1d'] for t in tickers}, axis=1
    ).sort_index()

    # 기간 필터
    idx = weights.index
    idx = idx[(idx >= period_start) & (idx <= period_end)]
    idx = idx.intersection(ret_df.index)

    if len(idx) < 2:
        return pd.Series(dtype=float)

    w   = weights.loc[idx].fillna(0.0)
    ret = ret_df.loc[idx].fillna(0.0)

    # ── 주간 리밸런싱: 매 5거래일마다만 비중 변경 ────────────────────────
    # 날짜 인덱스 기준으로 5일마다 신호 업데이트, 그 사이는 이전 비중 유지
    REBAL_FREQ = 5  # 리밸런싱 주기 (영업일 기준)
    rebal_mask = pd.Series(False, index=w.index)
    rebal_mask.iloc[::REBAL_FREQ] = True
    # 리밸런싱 날이 아니면 이전 신호 유지
    w_rebal = w.copy()
    for i in range(1, len(w_rebal)):
        if not rebal_mask.iloc[i]:
            w_rebal.iloc[i] = w_rebal.iloc[i - 1]

    # ── PnL (look-ahead bias 없음: 전일 비중 × 당일 수익률) ──────────────
    raw_pnl = (w_rebal.shift(1) * ret).sum(axis=1)

    # ── 거래비용 (편도 × 2 = 왕복, 리밸런싱 시점 기준) ──────────────────
    turnover  = w_rebal.diff().abs().sum(axis=1)
    tx_cost   = turnover * (COMMISSION_BPS + SLIPPAGE_BPS) / 10000  # 6 bps × |Δw|

    # ── 펀딩비용 (1 bps/8h × 3회/일 = 3 bps/일, gross exposure 기준) ────
    gross         = w_rebal.abs().sum(axis=1)
    funding_cost  = gross * (3 * FUNDING_BPS_8H) / 10000

    net_pnl = raw_pnl - tx_cost - funding_cost
    net_pnl.iloc[0] = 0.0  # 첫날은 포지션 없음

    # 일별 PnL 하한: -100% (마진콜 현실 반영 — 자본금 이상 손실 불가)
    net_pnl = net_pnl.clip(lower=-1.0)

    return net_pnl


# ---------------------------------------------------------------------------
# 성과 지표 계산
# ---------------------------------------------------------------------------
def sharpe_ratio(pnl: pd.Series) -> float:
    """연율화 샤프 비율 (무위험이자율 0)."""
    if pnl.std() < 1e-10:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(252))


def sortino_ratio(pnl: pd.Series) -> float:
    """연율화 소르티노 비율."""
    downside = pnl[pnl < 0].std()
    if downside < 1e-10:
        return 0.0
    return float(pnl.mean() / downside * np.sqrt(252))


def max_drawdown(pnl: pd.Series) -> float:
    """최대 낙폭(MDD) — 음수 반환."""
    cum = (1 + pnl).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    return float(dd.min())


def calmar_ratio(pnl: pd.Series) -> float:
    """칼마 비율."""
    ann_ret = float((1 + pnl).prod() ** (252 / max(len(pnl), 1)) - 1)
    mdd = abs(max_drawdown(pnl))
    if mdd < 1e-10:
        return 0.0
    return ann_ret / mdd


def win_rate(pnl: pd.Series) -> float:
    if len(pnl) == 0:
        return 0.0
    return float((pnl > 0).sum() / len(pnl) * 100)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    t_start = time.time()

    # 1. 데이터 수집 및 피처 엔지니어링
    print('=' * 60)
    print('[1] 데이터 수집 및 피처 엔지니어링')
    features = collect_and_engineer({}, _HERE)
    if not features:
        print('[ERROR] 데이터 로드 실패')
        raise SystemExit(1)
    print(f'  유니버스: {sorted(features.keys())}')

    # 2. 알파 시그널 생성
    print('[2] 알파 시그널 생성 (단면 모멘텀)')
    raw_weights = research_alpha(features)
    print(f'  신호 행렬: {raw_weights.shape[0]}일 × {raw_weights.shape[1]}종목')

    # 3. 포트폴리오 관리 (레버리지 조정)
    print('[3] 포트폴리오 비중 결정')
    weights = allocate_capital(raw_weights, 1.0, {}, max_leverage=MAX_LEVERAGE)
    print(f'  최대 gross exposure: {weights.abs().sum(axis=1).max():.2f}x')

    # 4. 백테스트 — 훈련 기간
    print('[4] 백테스트 — 훈련 기간')
    train_pnl = run_backtest(weights, features, TRAIN_START, TRAIN_END)
    train_sharpe     = sharpe_ratio(train_pnl)
    train_return_pct = float((1 + train_pnl).prod() - 1) * 100
    print(f'  Train Sharpe: {train_sharpe:.4f}  Return: {train_return_pct:.1f}%')

    # 5. 백테스트 — 검증 기간 (보고 대상)
    print('[5] 백테스트 — 검증 기간')
    val_pnl = run_backtest(weights, features, VAL_START, VAL_END)
    if len(val_pnl) == 0:
        print('[ERROR] 검증 기간 수익률 계산 실패')
        raise SystemExit(1)

    val_sharpe      = sharpe_ratio(val_pnl)
    val_return_pct  = float((1 + val_pnl).prod() - 1) * 100
    val_ann_ret_pct = float((1 + val_pnl).prod() ** (252 / max(len(val_pnl), 1)) - 1) * 100
    val_max_dd_pct  = max_drawdown(val_pnl) * 100
    val_vol_pct     = val_pnl.std() * np.sqrt(252) * 100
    val_win_rate    = win_rate(val_pnl)
    val_sortino     = sortino_ratio(val_pnl)
    val_calmar      = calmar_ratio(val_pnl)

    # 평균 일별 회전율 & 거래 횟수 (실제 리밸런싱 기준)
    val_idx = val_pnl.index

    # 검증 기간의 실제 리밸런싱 회전율 계산
    val_w_idx = weights.index
    val_w_idx = val_w_idx[(val_w_idx >= VAL_START) & (val_w_idx <= VAL_END)]
    val_w_idx = val_w_idx.intersection(
        pd.concat({t: features[t]['ret_1d'] for t in list(weights.columns) if t in features}, axis=1).index
    )

    if len(val_w_idx) >= 2:
        w_val = weights.loc[val_w_idx].fillna(0.0)
        # 주간 리밸런싱 비중 복원
        rebal_mask_v = pd.Series(False, index=w_val.index)
        rebal_mask_v.iloc[::5] = True
        w_rebal_v = w_val.copy()
        for i in range(1, len(w_rebal_v)):
            if not rebal_mask_v.iloc[i]:
                w_rebal_v.iloc[i] = w_rebal_v.iloc[i - 1]
        rebal_turnover = w_rebal_v.diff().abs().sum(axis=1)
        avg_turnover = float(rebal_turnover.mean())
        num_trades   = int((rebal_turnover > 0.001).sum())
    else:
        avg_turnover = 0.0
        num_trades   = 0

    # 벤치마크: BTC 단순 보유 (검증 기간)
    if 'BTCUSDT' in features:
        btc_ret = features['BTCUSDT']['ret_1d'].reindex(val_idx).fillna(0)
        benchmark_ret_pct = float((1 + btc_ret).prod() - 1) * 100
    else:
        benchmark_ret_pct = 0.0

    t_end = time.time()

    # ---------------------------------------------------------------------------
    # 표준 출력 (system_operator 규약)
    # ---------------------------------------------------------------------------
    print('---')
    print(f'val_sharpe:         {val_sharpe:.6f}')
    print(f'val_return_pct:     {val_return_pct:.2f}')
    print(f'val_annual_ret_pct: {val_ann_ret_pct:.2f}')
    print(f'val_max_dd_pct:     {val_max_dd_pct:.2f}')
    print(f'val_volatility_pct: {val_vol_pct:.2f}')
    print(f'val_win_rate_pct:   {val_win_rate:.1f}')
    print(f'val_sortino:        {val_sortino:.6f}')
    print(f'val_calmar:         {val_calmar:.6f}')
    print(f'train_sharpe:       {train_sharpe:.6f}')
    print(f'train_return_pct:   {train_return_pct:.2f}')
    print(f'benchmark_ret_pct:  {benchmark_ret_pct:.2f}')
    print(f'num_trades:         {num_trades}')
    print(f'avg_turnover:       {avg_turnover:.4f}')
    print(f'total_seconds:      {t_end - t_start:.1f}')
