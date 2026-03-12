"""GPT 수비 에이전트가 관리하는 추가 정적 규칙.

이 파일은 GPT defense agent가 자동으로 편집합니다.
수동으로 규칙을 추가할 수도 있습니다.

check() 함수가 반환하는 각 dict는 다음 키를 가져야 합니다:
  - source: "dynamic"
  - severity: "info" | "warning" | "error"
  - title: 규칙 이름 (짧게)
  - detail: 상세 설명
  - blocking: True이면 push를 차단합니다
"""

from __future__ import annotations


def check(
    repo_root,
    run_dir,
    revision: str,
    spec: dict,
    load_file,
) -> list[dict]:
    """추가 정적 검증을 수행하고 findings 리스트를 반환한다.

    Args:
        repo_root: git 저장소 루트 경로
        run_dir: 현재 run 디렉터리 경로
        revision: 검증 대상 git revision (보통 HEAD)
        spec: constraints.json 내용
        load_file: load_file("data_engineer.py") 로 파일 내용을 읽는 헬퍼
    """
    findings: list[dict] = []
    return findings
