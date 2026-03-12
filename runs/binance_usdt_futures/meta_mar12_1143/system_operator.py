"""
[System Operator]
데이터 엔지니어링 → 알파 시그널 → 포트폴리오 배분 → 백테스트 성과 계산.

비용 모델 (constraints.json 준수):
  - 거래 비용: 편도 (commission 4bps + slippage 2bps) = 6bps
    → 일별 turnover × 6bps 차감
  - Funding 비용: 1bps/8h = 3bps/일 (보유 gross 노출 × 3bps/day)

Train / Val 분리:
  - Train : DATA_START ~ VAL_START 전날
  - Val   : VAL_START ~ DATA_END

Usage: uv run system_operator.py
"""
import os
import sys
import json
import time
import pandas as pd
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_ROOT = os.environ.get(
    "AUTOQUANT_OUTPUT_ROOT",
    os.path.join(_HERE, "..", "..", ".."),
)
if _OUTPUT_ROOT not in sys.path:
    sys.path.insert(0, _OUTPUT_ROOT)

from commons.metrics import calculate_metrics  # noqa: E402
from data_engineer import collect_and_engineer  # noqa: E402
from quant_researcher import research_alpha     # noqa: E402
from portfolio_manager import allocate_capital  # noqa: E402

with open(os.path.join(_HERE, "constraints.json"), "r") as f:
    CONSTRAINTS = json.load(f)

_C = CONSTRAINTS["constraints"]
COMMISSION_BPS      = _C["commission_bps"]            # 4
SLIPPAGE_BPS        = _C["slippage_bps"]              # 2
FUNDING_BPS_PER_8H  = _C["funding_rate_bps_per_8h"]  # 1
MAX_LEVERAGE        = float(_C["max_leverage"])        # 3.0

ONEWAY_COST_BPS   = COMMISSION_BPS + SLIPPAGE_BPS  # 6bps (편도)
FUNDING_DAILY_BPS = FUNDING_BPS_PER_8H * 3          # 3bps/일

VAL_START    = "2024-01-01"
REBAL_FREQ   = 5   # 리밸런싱 주기 (거래일 기준); 5 = 주 1회


def run_backtest(
    weights: pd.DataFrame,
    features: dict,
) -> tuple:
    """
    포지션 비중(weights)으로 일별 포트폴리오 수익률을 계산한다.

    Returns
    -------
    (portfolio_returns, turnover_series, gross_exposure_series)
    """
    symbols = list(weights.columns)

    # 전향(forward) 수익률: 오늘 비중 → 내일 실현
    close_df = pd.DataFrame({s: features[s]["close"] for s in symbols})
    fwd_ret  = close_df.pct_change().shift(-1)   # 오늘 종가 대비 내일 종가 수익률

    # 공통 인덱스
    common = weights.index.intersection(fwd_ret.index)
    w = weights.loc[common]
    r = fwd_ret.loc[common]

    # --- P&L ---
    gross_pnl = (w * r).sum(axis=1)

    # 거래 비용: 일별 포지션 변화 × 6bps (편도)
    delta_w  = w.diff().fillna(w)  # 첫 날은 전체 비중을 신규 진입으로 처리
    turnover = delta_w.abs().sum(axis=1)
    tx_cost  = turnover * ONEWAY_COST_BPS / 10_000

    # Funding 비용: gross exposure × 3bps/일
    gross_exp    = w.abs().sum(axis=1)
    funding_cost = gross_exp * FUNDING_DAILY_BPS / 10_000

    pnl = gross_pnl - tx_cost - funding_cost

    # 마지막 날(forward return = NaN)은 제거
    pnl = pnl.dropna()
    return pnl, turnover.loc[pnl.index], gross_exp.loc[pnl.index]


if __name__ == "__main__":
    t_start = time.time()

    # 1. Data Engineering
    print("Loading and engineering data...")
    features = collect_and_engineer({}, _HERE)
    if not features:
        print("ERROR: 데이터 수집 실패 — 인터넷 연결 또는 심볼을 확인하세요.")
        sys.exit(1)
    print(f"  로드된 심볼 수: {len(features)}")

    # 2. Alpha Signals
    print("Generating alpha signals...")
    signals = research_alpha(features)
    if signals.empty:
        print("ERROR: 시그널 생성 실패")
        sys.exit(1)

    # 3. Portfolio Allocation (레버리지 제약)
    print("Allocating capital...")
    weights_daily = allocate_capital(signals, MAX_LEVERAGE, {})

    # 일별 리밸런싱: 매일 신호 갱신
    weights = weights_daily

    active_days = int((weights.abs().sum(axis=1) > 1e-6).sum())
    print(f"  비중 행렬: {weights.shape[0]}일 × {weights.shape[1]}심볼 "
          f"(활성 일수: {active_days})")

    # 4. Backtest
    print("Running backtest...")
    pnl, turnover, gross_exp = run_backtest(weights, features)

    # 5. Train / Val 분리
    val_start = pd.Timestamp(VAL_START)
    train_pnl = pnl[pnl.index < val_start]
    val_pnl   = pnl[pnl.index >= val_start]
    val_turn  = turnover[turnover.index >= val_start]

    if len(train_pnl) < 10 or len(val_pnl) < 10:
        print("WARNING: 데이터가 너무 짧아 신뢰할 수 없는 결과입니다.")

    train_m = calculate_metrics(train_pnl)
    val_m   = calculate_metrics(val_pnl)

    # 6. 벤치마크: BTC 매수보유 (선물, funding 차감)
    btc_ret = features["BTCUSDT"]["close"].pct_change().shift(-1).dropna()
    btc_val = btc_ret[btc_ret.index >= val_start] - FUNDING_DAILY_BPS / 10_000
    btc_total_ret = float((1 + btc_val).prod() - 1) * 100

    # 7. 거래 통계
    num_trades   = int((val_turn > 1e-6).sum())
    avg_turnover = float(val_turn.mean()) if len(val_turn) > 0 else 0.0

    t_end = time.time()

    # ---- 출력 (system_operator 표준 포맷) ----
    print("---")
    print(f"val_sharpe:         {val_m['sharpe']:.6f}")
    print(f"val_return_pct:     {val_m['total_return']:.2f}")
    print(f"val_annual_ret_pct: {val_m['annual_return']:.2f}")
    print(f"val_max_dd_pct:     {val_m['max_dd']:.2f}")
    print(f"val_volatility_pct: {val_m['volatility']:.2f}")
    print(f"val_win_rate_pct:   {val_m['win_rate']:.1f}")
    print(f"val_sortino:        {val_m['sortino']:.6f}")
    print(f"val_calmar:         {val_m['calmar']:.6f}")
    print(f"train_sharpe:       {train_m['sharpe']:.6f}")
    print(f"train_return_pct:   {train_m['total_return']:.2f}")
    print(f"benchmark_ret_pct:  {btc_total_ret:.2f}")
    print(f"num_trades:         {num_trades}")
    print(f"avg_turnover:       {avg_turnover:.4f}")
    print(f"total_seconds:      {t_end - t_start:.1f}")
