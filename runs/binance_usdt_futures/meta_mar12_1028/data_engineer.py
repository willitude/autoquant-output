"""
[Data Engineer]
바이낸스 USDT 무기한 선물 공개 API에서 일봉 OHLCV 데이터를 수집하고,
알파 시그널 생성에 필요한 피처(Feature)를 만듭니다.

데이터 출처: https://fapi.binance.com/fapi/v1/klines  (인증 불필요)
"""
import os
import json
import time
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime

# 유동성 충분한 USDT 무기한 선물 유니버스
SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LTCUSDT', 'MATICUSDT',
]

DATA_START = '2020-06-01'   # 20일 워밍업 + 훈련 기간 포함
DATA_END   = '2025-03-11'   # 검증 종료


def _date_to_ms(date_str: str) -> int:
    """날짜 문자열(YYYY-MM-DD)을 Unix ms로 변환."""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return int(dt.timestamp() * 1000)


def fetch_futures_klines(symbol: str, start_str: str, end_str: str) -> pd.DataFrame:
    """
    Binance USDT 무기한 선물 공개 REST API로 일봉 OHLCV를 가져옵니다.
    인증 불필요 (공개 klines 엔드포인트).
    """
    url_base = 'https://fapi.binance.com/fapi/v1/klines'
    start_ms = _date_to_ms(start_str)
    end_ms   = _date_to_ms(end_str)

    rows: list = []
    cur = start_ms

    while cur < end_ms:
        url = (
            f'{url_base}?symbol={symbol}&interval=1d'
            f'&startTime={cur}&endTime={end_ms}&limit=1500'
        )
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                batch = json.loads(r.read())
        except Exception as e:
            print(f"  [Warning] {symbol} 요청 실패: {e}")
            break

        if not batch:
            break

        rows.extend(batch)
        cur = batch[-1][6] + 1      # 마지막 close_time + 1 ms
        if len(batch) < 1500:
            break
        time.sleep(0.05)            # API rate-limit 준수

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_vol', 'n_trades', 'taker_base', 'taker_quote', '_',
    ])
    df.index = pd.to_datetime(df['open_time'], unit='ms').dt.normalize()
    df.index.name = 'date'
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c])

    return df[['open', 'high', 'low', 'close', 'volume']].sort_index()


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """종가 기반 피처 엔지니어링. look-ahead bias 없음."""
    close = df['close']
    df = df.copy()

    df['ret_1d']  = close.pct_change()
    df['ret_5d']  = close.pct_change(5)
    df['ret_20d'] = close.pct_change(20)

    df['vol_20d'] = df['ret_1d'].rolling(20).std()
    df['vol_5d']  = df['ret_1d'].rolling(5).std()

    # 위험조정 모멘텀 (ret / vol)
    df['mom_risk_adj_20d'] = df['ret_20d'] / (df['vol_20d'] + 1e-8)
    df['mom_risk_adj_5d']  = df['ret_5d']  / (df['vol_5d']  + 1e-8)

    return df


def collect_and_engineer(base_data: dict, cache_dir: str) -> dict[str, pd.DataFrame]:
    """
    1. 바이낸스 선물 API에서 일봉 OHLCV 수집 (로컬 캐시 활용)
    2. 피처(수익률·변동성·모멘텀) 생성
    3. 심볼 → DataFrame 딕셔너리 반환
    """
    cache_path = os.path.join(cache_dir, 'data_cache')
    os.makedirs(cache_path, exist_ok=True)

    result: dict[str, pd.DataFrame] = {}

    for symbol in SYMBOLS:
        csv_path = os.path.join(cache_path, f'{symbol}_1d.csv')

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            print(f"  [Cache] {symbol}: {len(df)}행")
        else:
            print(f"  [Download] {symbol} 다운로드 중 ...")
            df = fetch_futures_klines(symbol, DATA_START, DATA_END)
            if df.empty:
                print(f"  [Skip] {symbol}: 데이터 없음")
                continue
            df.to_csv(csv_path)
            print(f"  [Saved] {symbol}: {len(df)}행")

        if len(df) < 30:
            continue

        result[symbol] = _engineer(df)

    return result
