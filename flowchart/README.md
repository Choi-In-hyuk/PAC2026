# 검증 피드백 기반 자율 표면 가공 경로 최적화 시스템

로봇 표면 도포(coverage painting) 공정의 전체 파이프라인을 **Streamlit** 데모로 시각화합니다.
입력(car_door CAD) → 곡률 분석 → 영역 분할 → 경로 생성(6D TCP) → 시뮬레이션 검증 → 실행 → 검사까지 성공 시나리오를 단계별로 보여줍니다.

- 대상물: **car_door** (차량 도어 패널)
- 로봇: **Franka Panda** (reach ≈ 0.855 m)
- 솔버: Hierarchical GTSP

상위 `arm_coverage/` 레포의 서브폴더로 들어 있으며, 데모가 참조하는 이미지·CSV는 `../visualizations/`, `../motions/`, `../performance/` 에서 상대참조로 로드합니다.

## 실행 방법

이 `flowchart/` 폴더 안에서:

```bash
pip install -r requirements.txt
streamlit run app.py
```

브라우저가 자동으로 열립니다. 안 열리면 http://localhost:8501 로 접속하세요.

하단의 **← 이전 / 처음으로 / 다음 단계 →** 버튼으로 16단계 흐름을 넘겨가며 발표할 수 있습니다.

## 구성

- `app.py` — Streamlit 앱 (단계 정의 + 플로우차트 하이라이트 + 상세 패널)
- `flowchart.jpg` — 전체 플로우차트 이미지 (좌측 패널 배경)
- `../visualizations/` — 곡률·영역 분할·경로 시각화 이미지 (`.png`, `.gif`) · 상위 arm_coverage에서 참조
- `../motions/` — 실행 조인트 궤적 CSV (7-DOF Panda) · 상위 arm_coverage에서 참조
- `../performance/` — 솔버 비교 성능 지표 CSV · 상위 arm_coverage에서 참조

## 플로우

```
작업 의뢰 → CAD 있음?(예) → CAD 로드·곡률 분석 → 실물-CAD 정합
→ 도달 범위 초과?(아니오) → 단일 베이스 → 영역 분할
→ 웨이포인트·경로 생성 (6D TCP) → IK·도달성 OK?(예)
→ 시뮬레이션 검증 → 품질 충족?(예) → 실행
→ 실행 중 이상?(아니오) → 도포 검사 → 재작업?(아니오) → 완료
```
