"""
[Portfolio Manager]
리서처 시그널을 받아 레버리지 한도 내에서 최종 비중을 결정합니다.
"""
import pandas as pd


def allocate_capital(
    signals: pd.DataFrame,
    current_capital: float,
    current_positions: dict,
    max_leverage: float = 3.0,
) -> pd.DataFrame:
    """
    타겟 비중을 최대 레버리지(gross exposure) 범위 내로 조정합니다.

    Args:
        signals: pd.DataFrame — 리서처가 생성한 목표 비중 (날짜 × 티커)
        current_capital: 현재 자본 (백테스트에서는 1.0 정규화)
        current_positions: 현재 포지션 (라이브용, 백테스트에서는 빈 dict)
        max_leverage: 최대 gross exposure 배수 (기본 3.0)

    Returns:
        pd.DataFrame — 조정된 최종 비중
    """
    if not isinstance(signals, pd.DataFrame) or signals.empty:
        return signals

    # 총 gross exposure가 max_leverage를 초과하면 비례 축소
    gross = signals.abs().sum(axis=1)
    scale = gross.clip(lower=max_leverage) / max_leverage   # ≥ 1
    adjusted = signals.div(scale, axis=0)

    return adjusted
