from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

def _bbox(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


_W_PROC = 0.085
_W_PROC_WIDE = 0.108
_H_PROC = 0.095
_W_DEC = 0.095
_H_DEC = 0.135

NODE_BBOX = {
    "intake":        _bbox(0.078, 0.095, 0.115, 0.125),
    "cad_check":     _bbox(0.248, 0.108, _W_DEC, _H_DEC),
    "cad_load":      _bbox(0.420, 0.100, _W_PROC_WIDE, _H_PROC),
    "align":         _bbox(0.583, 0.100, _W_PROC_WIDE, _H_PROC),
    "reach_check":   _bbox(0.745, 0.108, _W_DEC, _H_DEC),
    "single_base":   _bbox(0.912, 0.108, _W_PROC, _H_PROC),
    "regions":       _bbox(0.912, 0.258, _W_PROC, _H_PROC),
    "waypoints":     _bbox(0.912, 0.420, _W_PROC_WIDE, 0.125),
    "ik_check":      _bbox(0.912, 0.593, _W_DEC, _H_DEC),
    "sim":           _bbox(0.755, 0.585, _W_PROC_WIDE, 0.110),
    "quality_check": _bbox(0.595, 0.585, _W_DEC, _H_DEC),
    "execute":       _bbox(0.420, 0.590, _W_PROC_WIDE, 0.110),
    "exec_check":    _bbox(0.242, 0.585, _W_DEC, _H_DEC),
    "inspect":       _bbox(0.078, 0.588, _W_PROC, _H_PROC),
    "rework_check":  _bbox(0.078, 0.720, _W_DEC, _H_DEC),
    "done":          _bbox(0.078, 0.898, 0.115, _H_PROC),
}


ROOT = Path(__file__).parent
PAC_ROOT = ROOT.parent
VIZ = PAC_ROOT / "visualizations"
MOTIONS = PAC_ROOT / "motions"
PERF = PAC_ROOT / "performance"
FLOWCHART_IMG = ROOT / "flowchart.jpg"


STEPS = [
    {
        "id": "intake",
        "kind": "process",
        "title": "작업 의뢰 접수",
        "subtitle": "대상물 + 공정 조건",
        "detail": (
            "- 대상물: **car_door** (차량 도어 패널)\n"
            "- 공정: 표면 도포 (coverage painting)\n"
            "- 로봇: Franka Panda (reach ≈ 0.855 m)\n"
            "- CAD: 제공됨 (`meshes/car_door.obj`)\n"
            "- 공정 방향: dir0"
        ),
    },
    {
        "id": "cad_check",
        "kind": "decision",
        "title": "판단 — CAD 있음?",
        "verdict": "예",
        "reason": "입력에 CAD 메쉬 파일이 포함되어 있어 3D 스캔 단계를 건너뜁니다.",
        "next_label": "→ CAD 로드·곡률 분석",
    },
    {
        "id": "cad_load",
        "kind": "process",
        "title": "CAD 로드 · 곡률/법선 분석",
        "subtitle": "표면 곡률 추정으로 영역 분할의 근거를 만듭니다.",
        "images": [
            (VIZ / "curvature_car_door_dir0_persp.png", "곡률 (perspective)"),
            (VIZ / "curvature_car_door_dir0_hist.png", "곡률 분포 (histogram)"),
        ],
        "detail": "메쉬에 대해 주곡률을 계산 → 평탄 영역과 고곡률 영역을 식별합니다.",
    },
    {
        "id": "align",
        "kind": "process",
        "title": "실물 ↔ CAD 정합 · 오차 보정",
        "subtitle": "실측 좌표계와 CAD 좌표계를 정합합니다.",
        "detail": (
            "- 정합 방식: ICP 기반 rigid 정합\n"
            "- 정합 오차: < 1.0 mm (허용 범위 내)\n"
            "- 보정된 TCP 타겟 좌표를 다음 단계로 전달"
        ),
    },
    {
        "id": "reach_check",
        "kind": "decision",
        "title": "판단 — 도달 범위보다 큰 대상물?",
        "verdict": "아니오",
        "reason": "대상물 bounding box ≈ 0.9 × 0.4 m, Panda reach 0.855 m 안에서 단일 베이스로 커버 가능합니다.",
        "next_label": "→ 단일 베이스",
    },
    {
        "id": "single_base",
        "kind": "process",
        "title": "단일 베이스 선택",
        "subtitle": "로봇 베이스를 한 위치로 고정합니다.",
        "detail": "스테이션 분할 없이 단일 베이스 좌표 기준으로 이후 계획을 수행합니다.",
    },
    {
        "id": "regions",
        "kind": "process",
        "title": "영역 분할",
        "subtitle": "곡률 기반으로 표면을 커버리지 영역들로 분할합니다.",
        "images": [
            (VIZ / "regions_car_door_dir0_side.png", "영역 분할 (side)"),
        ],
        "detail": "각 영역은 한 번의 연속 도포 스트로크로 커버되는 단위입니다.",
    },
    {
        "id": "waypoints",
        "kind": "process",
        "title": "웨이포인트 · 경로 생성 (6D TCP 자세)",
        "subtitle": "영역별 커버리지 경로를 TSP/GTSP로 전역 최적화합니다.",
        "images": [
            (VIZ / "panda_full_dir0_v2_withIK.gif", "경로 실행 (IK 포함)"),
            (VIZ / "panda_full_dir0_v2_withIK.png", "경로 스냅샷"),
        ],
        "detail": (
            "- 전역 순서화: **Hierarchical GTSP** (영역 간 + 영역 내)\n"
            "- 각 웨이포인트에 대해 6D TCP (position + orientation) 생성\n"
            "- 곡률을 반영해 공구 자세·접촉각을 유지"
        ),
    },
    {
        "id": "ik_check",
        "kind": "decision",
        "title": "판단 — IK · 도달성 · 충돌 OK?",
        "verdict": "예",
        "reason": "전 웨이포인트에서 IK 해가 존재하고 자기충돌·환경충돌이 없습니다.",
        "next_label": "→ 시뮬레이션 검증",
    },
    {
        "id": "sim",
        "kind": "process",
        "title": "시뮬레이션 검증 — 도막 · 접촉력 예측",
        "subtitle": "실행 전에 커버리지와 효율을 수치로 확인합니다.",
        "performance_csv": PERF / "performance_20261006-142541.csv",
        "detail": (
            "- `hrchy_gtsp` 솔버가 다른 베이스라인(cartesian_tsp, joint_gtsp) 대비\n"
            "  **총 조인트 이동량**과 **재구성(reconfig) 횟수**에서 유리함을 확인."
        ),
    },
    {
        "id": "quality_check",
        "kind": "decision",
        "title": "판단 — 품질 기준 충족?",
        "verdict": "예",
        "reason": (
            "**품질 기준 평가 (hrchy_gtsp 솔버 결과)**\n\n"
            "- **커버리지율**: 100% (목표 ≥ 99%) — 표면 전 영역 샘플 포인트가 모두 도달 가능 TCP로 커버됨\n"
            "- **재구성(reconfig) 횟수**: 0회 (목표 ≤ 2회) — 중간 자세 재설정 없이 연속 실행 가능\n"
            "- **총 조인트 이동량**: baseline(cartesian_tsp) 대비 **-10% 수준**으로 단축 → 사이클 타임·마모 유리\n"
            "- **예상 도막 균일도**: 곡률 기반 공구 자세 유지로 접촉각 편차 임계 범위 이내\n\n"
            "→ 네 지표 모두 기준 충족, 되돌림 없이 실행 단계로 진행."
        ),
        "next_label": "→ 실행",
    },
    {
        "id": "execute",
        "kind": "process",
        "title": "실행 — 힘 · 거리 피드백 제어",
        "subtitle": "계획된 조인트 궤적을 로봇에서 실행합니다.",
        "motion_csv": MOTIONS / "panda_full_dir0_v2_withIK.csv",
        "detail": "실제 실행 중 수집된 조인트 궤적입니다 (7-DOF Panda).",
    },
    {
        "id": "exec_check",
        "kind": "decision",
        "title": "판단 — 실행 중 이상?",
        "verdict": "아니오",
        "reason": "힘·토크 피드백 모두 임계 범위 이내, 중단 이벤트 없음.",
        "next_label": "→ 도포 검사",
    },
    {
        "id": "inspect",
        "kind": "process",
        "title": "도포 검사 · 결함 검출",
        "subtitle": "실행 결과를 검사하여 결함 여부를 판정합니다.",
        "detail": (
            "- 커버리지 검사: 전 영역 목표 두께 달성\n"
            "- 결함: 없음"
        ),
    },
    {
        "id": "rework_check",
        "kind": "decision",
        "title": "판단 — 재작업 필요?",
        "verdict": "아니오",
        "reason": "검사 통과, 수동 터치업 대상 없음.",
        "next_label": "→ 종료",
    },
    {
        "id": "done",
        "kind": "terminal",
        "title": "종료 + 수동 터치업 목록 보고",
        "subtitle": "공정 완료.",
        "detail": "수동 터치업 목록: **없음**. 작업 종료.",
    },
]


@st.cache_data
def _load_flowchart_base() -> Image.Image:
    return Image.open(FLOWCHART_IMG).convert("RGBA")


def render_flowchart_with_highlight(current_step_id: str) -> Image.Image:
    base = _load_flowchart_base().copy()
    bbox_norm = NODE_BBOX.get(current_step_id)
    if bbox_norm is None:
        return base
    w, h = base.size
    x0, y0, x1, y1 = bbox_norm
    box = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(box, fill=(255, 0, 0, 60))
    draw.rectangle(box, outline=(220, 0, 0, 255), width=8)
    return Image.alpha_composite(base, overlay)


IMG_WIDTH = 380


def render_step(step: dict) -> None:
    kind = step["kind"]

    if kind == "decision":
        st.markdown(f"#### {step['title']}")
        verdict_color = "#22c55e" if step["verdict"] == "예" else "#ef4444"
        st.markdown(
            f"<div style='font-size:1.4rem;margin:0.3rem 0;'>"
            f"판정: <b style='color:{verdict_color};'>{step['verdict']}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.info(step["reason"])
        st.caption(step["next_label"])
        return

    st.markdown(f"#### {step['title']}")
    if step.get("subtitle"):
        st.caption(step["subtitle"])

    if step.get("detail"):
        st.markdown(step["detail"])

    imgs = step.get("images", [])
    if imgs:
        cols = st.columns(min(len(imgs), 2))
        for i, (img_path, caption) in enumerate(imgs):
            with cols[i % len(cols)]:
                if img_path.exists():
                    st.image(str(img_path), caption=caption, width=IMG_WIDTH)
                else:
                    st.warning(f"이미지 없음: {img_path}")

    perf_csv = step.get("performance_csv")
    if perf_csv and perf_csv.exists():
        df = pd.read_csv(perf_csv)
        st.caption("솔버 비교 (성능)")
        st.dataframe(df, width="stretch", height=240)

    motion_csv = step.get("motion_csv")
    if motion_csv and motion_csv.exists():
        df = pd.read_csv(motion_csv)
        joint_cols = [c for c in df.columns if c.startswith("panda-panda_joint")]
        if joint_cols and "time" in df.columns:
            st.caption("조인트 궤적 (7-DOF)")
            st.line_chart(df.set_index("time")[joint_cols], height=220)


def main() -> None:
    st.set_page_config(
        page_title="검증 피드백 기반 자율 표면 가공 경로 최적화 시스템",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 100%; }
        header[data-testid="stHeader"] { height: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "step_idx" not in st.session_state:
        st.session_state.step_idx = 0

    step = STEPS[st.session_state.step_idx]

    header_l, header_r = st.columns([3, 1])
    with header_l:
        st.markdown(
            "**검증 피드백 기반 자율 표면 가공 경로 최적화 시스템** — car_door · Franka Panda"
        )
    with header_r:
        st.markdown(
            f"<div style='text-align:right;'>단계 "
            f"<b>{st.session_state.step_idx + 1} / {len(STEPS)}</b></div>",
            unsafe_allow_html=True,
        )

    left, right = st.columns([5, 4], gap="small")

    with left:
        if FLOWCHART_IMG.exists():
            st.image(
                render_flowchart_with_highlight(step["id"]),
                width="stretch",
            )

    with right:
        render_step(step)

    st.divider()
    btn_prev, btn_reset, btn_next = st.columns([1, 1, 2])
    with btn_prev:
        if st.button(
            "← 이전 단계",
            width="stretch",
            disabled=st.session_state.step_idx == 0,
        ):
            st.session_state.step_idx -= 1
            st.rerun()
    with btn_reset:
        if st.button("처음으로", width="stretch"):
            st.session_state.step_idx = 0
            st.rerun()
    with btn_next:
        if st.button(
            "다음 단계 →",
            width="stretch",
            disabled=st.session_state.step_idx >= len(STEPS) - 1,
            type="primary",
        ):
            st.session_state.step_idx += 1
            st.rerun()


if __name__ == "__main__":
    main()
