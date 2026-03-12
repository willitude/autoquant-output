"""
[Data Engineer]
바이낸스 USDT 무기한 선물 공개 REST API에서 일봉 OHLCV를 수집하고,
기술적 피처를 생성합니다. API 키 불필요 (공개 엔드포인트).

캐시: {cache_dir}/data_cache/{symbol}.csv
심볼: 유동성 상위 10개 USDT 무기한 선물
"""
import os
import sys
import time
import requests
import pandas as pd
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_ROOT = os.environ.get(
    "AUTOQUANT_OUTPUT_ROOT",
    os.path.join(_HERE, "..", "..", ".."),
)
if _OUTPUT_ROOT not in sys.path:
    sys.path.insert(0, _OUTPUT_ROOT)

from commons.indicators import engineer_base_features  # noqa: E402

# 유동성 충분한 USDT 무기한 선물 심볼 (2022년 이전부터 상장)
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
]

DATA_START = "2022-01-01"
DATA_END   = "2025-12-31"
FAPI_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"


def _fetch_symbol_klines(symbol: str, start: str, end: str) -> pd.DataFrame:
    """바이낸스 선물 klines 엔드포인트에서 일봉 OHLCV를 다운로드한다."""
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms   = int(pd.Timestamp(end).timestamp() * 1000)

    all_rows: list[list] = []
    while start_ms < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  "1d",
            "startTime": start_ms,
            "endTime":   end_ms,
            "limit":     1000,
        }
        try:
            resp = requests.get(FAPI_KLINES_URL, params=params, timeout=20)
            resp.raise_for_status()
            batch = resp.json()
        except Exception as exc:
            print(f"  WARNING: {symbol} API 오류 — {exc}")
            break
        if not batch:
            break
        all_rows.extend(batch)
        # 다음 페이지 시작점: 마지막 캔들 close_time + 1ms
        start_ms = int(batch[-1][6]) + 1
        if len(batch) < 1000:
            break
        time.sleep(0.05)   # rate-limit 방지

    if not all_rows:
        return pd.DataFrame()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "_ignore",
    ]
    df = pd.DataFrame(all_rows, columns=cols)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.set_index("date").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    return df[["open", "high", "low", "close", "volume"]]


def collect_and_engineer(
    base_data: dict[str, pd.DataFrame],
    cache_dir: str,
) -> dict[str, pd.DataFrame]:
    """
    각 심볼에 대해:
      1. 캐시(CSV)가 있으면 로드, 없으면 바이낸스 API로 다운로드 후 저장
      2. commons.indicators.engineer_base_features 로 기본 피처 계산
    Returns: {symbol: feature_df}
    """
    cache_path = os.path.join(cache_dir, "data_cache")
    os.makedirs(cache_path, exist_ok=True)

    result: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        csv_file = os.path.join(cache_path, f"{symbol}.csv")

        if os.path.exists(csv_file):
            df_raw = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        else:
            print(f"  Fetching {symbol} from Binance futures API...")
            df_raw = _fetch_symbol_klines(symbol, DATA_START, DATA_END)
            if df_raw.empty:
                print(f"  WARNING: {symbol} 데이터 없음, 스킵")
                continue
            df_raw.to_csv(csv_file)
            print(f"  {symbol}: {len(df_raw)}개 캔들 저장 완료")

        if df_raw.empty or len(df_raw) < 60:
            print(f"  WARNING: {symbol} 데이터 부족({len(df_raw)}행), 스킵")
            continue

        feat = engineer_base_features(df_raw["close"], df_raw["volume"])
        feat["open"]  = df_raw["open"]
        feat["high"]  = df_raw["high"]
        feat["low"]   = df_raw["low"]
        result[symbol] = feat

    return result
