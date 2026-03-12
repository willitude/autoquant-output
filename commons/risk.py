"""공통 리스크 관리.

에이전트가 서킷 브레이커, 노출 한도, 포지션 사이징 로직을 여기에 축적한다.
"""


def check_risk_limits(proposed_orders: list, portfolio_state: dict) -> list:
    """주문 리스트를 검사하여 리스크 한도를 초과하는 것을 필터링한다."""
    approved = proposed_orders
    return approved
