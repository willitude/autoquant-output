"""
[Live Operator]
실제 거래소 API 키를 물고 라이브 노드(LiveNode)를 띄우는 역할을 합니다.
모의투자(Paper Trading) 또는 실전 투자(Live Trading)에 사용됩니다.

Usage: uv run live_operator.py
"""
import os
import json
import time

from data_engineer import collect_and_engineer
from quant_researcher import research_alpha
from portfolio_manager import allocate_capital
from risk_manager import check_risk_limits
from execution_trader import execute_trades

# ---------------------------------------------------------------------------
# Load constraints
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "constraints.json"), "r") as f:
    CONSTRAINTS = json.load(f)

if __name__ == "__main__":
    print("Initializing Live Trading Node...")
    print(f"Goal: {CONSTRAINTS['goal']}")
    
    # 에이전트: 여기에 거래소 API 키 로드, 웹소켓 연결, 실시간 데이터 구독 로직을 작성하세요.
    # (나중에 Nautilus Trader의 LiveNode 설정으로 교체될 예정입니다.)
    
    print("Live Node is running. (Mock)")
    
    try:
        while True:
            # 실시간 이벤트 루프 시뮬레이션
            time.sleep(1)
    except KeyboardInterrupt:
        print("Live Node stopped by user.")
