"""
[Risk Manager]
개별 주문이나 전체 포트폴리오가 위험 한도를 넘지 않는지 감시하고,
위험할 경우 주문을 거절하거나 강제 청산(Kill Switch)을 발동합니다.
"""

def check_risk_limits(proposed_orders: list, portfolio_state: dict) -> list:
    """
    제안된 주문들을 검사하여 리스크 한도(예: 최대 노출, MDD)를 초과하는 주문은 필터링합니다.
    """
    # 에이전트: 여기에 서킷 브레이커, 자산군별 노출 한도 관리 로직을 작성하세요.
    approved_orders = proposed_orders
    return approved_orders
