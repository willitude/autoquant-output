"""공통 실행 로직.

Nautilus Trader ExecutionAlgorithm으로 발전할 기반.
에이전트가 슬리피지 모델, TWAP/VWAP 로직 등을 여기에 추가한다.
"""

import pandas as pd


def execute_trades(
    target_weights: pd.DataFrame,
    current_positions: dict,
    market_data: dict,
) -> list:
    """타겟 비중과 현재 포지션을 비교하여 주문 리스트를 생성한다."""
    orders = []
    return orders
