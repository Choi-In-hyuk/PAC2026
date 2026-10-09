"""Build an experiment-results Word doc summarizing car_door coverage work."""
import os
from datetime import datetime
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH


OUT = "docs/cardoor_experiment_report.docx"


def h1(doc, t): doc.add_heading(t, level=1)
def h2(doc, t): doc.add_heading(t, level=2)
def h3(doc, t): doc.add_heading(t, level=3)


def para(doc, text, bold=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    if bold:
        r.bold = True
    return p


def bullet(doc, text):
    doc.add_paragraph(text, style="List Bullet")


def table(doc, header, rows, widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Light Grid Accent 1"
    hdr = t.rows[0].cells
    for i, h in enumerate(header):
        hdr[i].text = h
        for run in hdr[i].paragraphs[0].runs:
            run.bold = True
    for r, row in enumerate(rows, 1):
        for i, v in enumerate(row):
            t.rows[r].cells[i].text = str(v)
    if widths:
        for col, w in enumerate(widths):
            for row in t.rows:
                row.cells[col].width = Cm(w)
    return t


def add_image(doc, path, width_cm=14):
    if os.path.exists(path):
        doc.add_picture(path, width=Cm(width_cm))
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.CENTER


def main():
    os.makedirs("docs", exist_ok=True)
    doc = Document()

    # Title
    title = doc.add_heading("car_door 폴리싱 경로 계획 실험 결과", level=0)
    sub = doc.add_paragraph()
    sub.add_run(f"작성일: {datetime.now().strftime('%Y-%m-%d')}   |   데이터: car_door v2.6").italic = True

    # 1. 과제 개요
    h1(doc, "1. 과제 개요")
    para(doc, "표면 가공(폴리싱) 로봇 경로 생성에서 두 가지 공정 지표를 최적화하는 것이 목표.")
    bullet(doc, "커버리지 최대화: 작업 가능 면적을 최대한 많이 처리")
    bullet(doc, "중복 최소화: 같은 자리를 두 번 이상 지나가는 비율을 줄임")
    para(doc, "추가 요구사항: 공구 규격·자세·간섭 등 제약 조건 반영, 곡면 대응, 디지털 트윈 검증.")

    # 2. 입력 데이터
    h1(doc, "2. 입력 데이터 — car_door v2.6")
    para(doc, "동료(김재연)가 전처리한 car_door 메시 데이터. 공구 접촉 모델(Schneyer 2023) 기반.")

    h2(doc, "2.1 메시 면적 분포")
    table(doc, ["상태", "face 수", "면적 (cm²)", "비율"],
          [["FEASIBLE (1) — 폴리싱 가능", 5195, "1362.98", "94.8%"],
           ["로봇 불가 (3)", 250, "47.82", "3.3%"],
           ["공구 불가 (2)", 79, "12.98", "0.9%"],
           ["미검사 (0)", 46, "14.41", "1.0%"],
           ["차체 한 면 합계", 5570, "1438.19", "100%"]])

    h2(doc, "2.2 Waypoint 데이터")
    bullet(doc, "전체 샘플점: 1,469개")
    bullet(doc, "방문가능 (tool_pose_valid & accessibility): 1,220개 (중심 접촉 7, 가장자리 접촉 1,213)")
    bullet(doc, "영역(region): 88개 (접촉 모드·가공 방향 기준)")
    bullet(doc, "라인(line): 324개 (영역 안 쓸고 지나가는 단위, 영역당 평균 4.1개)")
    bullet(doc, "라인당 waypoint: 평균 3.8개, 최대 55개")

    h2(doc, "2.3 공구·공정 가정")
    bullet(doc, "공구 패드 반경: 5 cm (평판 디스크)")
    bullet(doc, "패드 압입량 (PAD_INDENT): 0.5 mm (가정값, 공정 스펙 확정 시 재계산)")
    bullet(doc, "로봇: Franka Panda (RelaxedIK)")
    bullet(doc, "로봇 베이스: 작업면 중심 + (0.7, 0, 0) m")

    # 3. 접촉 영역 모델
    h1(doc, "3. 접촉 영역 모델 (Schneyer 2023)")
    para(doc, "waypoint 하나가 커버하는 메시 face 집합을 두 조건으로 판정:")
    bullet(doc, "조건 1: face 중심이 공구 패드 디스크 안 (반경 ≤ 5 cm)")
    bullet(doc, "조건 2: 패드 평면과 face 중심 사이 수직 거리 ≤ 0.5 mm (PAD_INDENT)")
    para(doc, "곡률에 따라 waypoint마다 접촉 면적이 다름. 평평한 면은 넓게, 곡면은 좁게 접촉.")

    # 4. 평가 방식 두 가지
    h1(doc, "4. 평가 방식")

    h2(doc, "4.1 도장 모델 (단순)")
    para(doc, "waypoint 각각의 접촉 영역만 합집합. 사이 공간은 '안 폴리싱'으로 봄. 보수적.")

    h2(doc, "4.2 영역 내 쓸고 지나감 모델 (공정 가정 반영)")
    para(doc, "같은 영역 안에서 연속된 waypoint 사이는 공구가 쓸고 지나간다고 가정하고 보간(2mm 서브스텝). "
              "영역 사이 전환은 공구가 들어 올려지므로 보간 없음.")

    # 5. 알고리즘 비교
    h1(doc, "5. 알고리즘 비교")

    h2(doc, "5.1 비교 대상")
    bullet(doc, "line_order: 동료 line 순서를 유지하고 각 waypoint에 Panda IK만 적용 (베이스라인)")
    bullet(doc, "hrchy_gtsp: ICRA'25 Wang & Gleicher, joint 이동 최소화 (기존 알고리즘)")
    bullet(doc, "region_cover: 영역별 greedy set cover (본 작업에서 구현)")

    h2(doc, "5.2 결과 — 도장 모델 평가")
    table(doc, ["알고리즘", "waypoint", "커버 %", "중복 %", "평균 중복", "최대"],
          [["line_order", 1206, "83.40", "42.44", "1.52x", 6],
           ["hrchy_gtsp", 1217, "83.48", "42.54", "1.52x", 6],
           ["region_cover", 916, "80.81", "32.61", "1.39x", 5]])
    para(doc, "도장 모델에서는 waypoint 수가 많아도 PAD_INDENT 0.5mm band 제한 때문에 커버가 83% 수준에서 포화. "
              "region_cover는 waypoint 25% 감소에도 중복이 10%p 낮음.", bold=False)

    h2(doc, "5.3 결과 — 영역 내 쓸고 지나감 모델 평가")
    table(doc, ["알고리즘", "waypoint", "sweep 수", "커버 %", "중복 %", "평균 중복"],
          [["line_order", 1206, 133, "98.81", "26.16", "1.34x"],
           ["hrchy_gtsp", 1217, 199, "95.68", "26.37", "1.28x"],
           ["region_cover", 916, 88, "96.39", "22.53", "1.27x"]])
    para(doc, "쓸고 지나감 가정 하에서 region_cover는 커버 96.4%(line_order 상한 98.8% 대비 -2.4%p)를 유지하면서 "
              "중복을 22.5%로 가장 낮게 달성. sweep 수(공구 들어 올림 횟수)도 영역 수와 동일한 88로 최소.", bold=False)

    h2(doc, "5.4 joint 효율성 (참고)")
    table(doc, ["알고리즘", "joint 이동", "reconfigs"],
          [["line_order", "219.5", 107],
           ["hrchy_gtsp", "148.9", 79],
           ["region_cover", "163.8", 80]])
    para(doc, "hrchy_gtsp는 joint 이동 최소화에 특화되어 가장 낮음. region_cover는 공정 지표 최적화 과정에서 "
              "중간 수준의 joint 효율성 확보.")

    # 6. 알고리즘 상세
    h1(doc, "6. 제안 알고리즘 — Region-wise Greedy Set Cover")

    h2(doc, "6.1 핵심 아이디어")
    para(doc, "동료가 깐 88개 영역을 그대로 받고, 각 영역 안에서 독립적으로 set cover 문제를 품.")
    para(doc, "각 영역 안에서:")
    bullet(doc, "waypoint들의 접촉 face 집합을 계산")
    bullet(doc, "'아직 안 덮인 face를 가장 많이 덮는 waypoint'를 반복 선택")
    bullet(doc, "더 이상 새 face를 덮는 waypoint가 없으면 종료")

    h2(doc, "6.2 결과: 1,220 → 921 waypoint (24.5% 감소)")

    h2(doc, "6.3 왜 영역별로 분리하나")
    bullet(doc, "영역 경계는 접촉 모드/가공 방향이 바뀌는 지점 → 공구를 재자세해야 함")
    bullet(doc, "영역 내부에서만 '쓸고 지나감' 공정 가정이 성립")
    bullet(doc, "영역을 가로지르는 waypoint 선택은 공정상 비현실적")

    # 7. 발견한 데이터 한계
    h1(doc, "7. 현재 데이터의 한계")

    h2(doc, "7.1 도장 모델 상한이 83.5%")
    para(doc, "1,220 waypoint 전부를 써도 도장 모델에선 커버 83.5%가 상한. "
              "안 덮인 911개 face 전부가 'waypoint가 5cm 이내 있지만 PAD_INDENT 0.5mm 밖'이라는 특성.")
    para(doc, "원인: PAD_INDENT가 매우 빡빡함 + 곡면 face는 수직거리가 band 밖으로 쉽게 밀려남.")

    h2(doc, "7.2 쓸고 지나감 가정에 커버 의존")
    para(doc, "커버 83% → 98%로 끌어올리는 것은 '같은 영역 안 waypoint 사이가 공구가 연속으로 쓸고 지나간다'는 공정 가정에 의존. "
              "알고리즘이 waypoint 순서/구성을 바꾸면 이 가정이 깨지면서 커버가 하락할 수 있음.")

    h2(doc, "7.3 알고리즘 자유도 제약")
    para(doc, "동료 샘플링이 쓸고 지나감을 전제로 깐 결과, 알고리즘의 선택 자유도가 제한됨. "
              "waypoint를 임의로 재배치/제거하면 커버 손실.")

    # 8. 제안 로드맵
    h1(doc, "8. 제안 로드맵")

    para(doc, "역할 분리된 3단계 플로우:")
    table(doc, ["단계", "담당", "목표", "지표"],
          [["1. 완전 샘플링", "동료", "도장 모델로 커버 100% 달성", "도장 커버 ≥ 99%"],
           ["2. 중복 제거", "본 알고리즘", "waypoint 수 축소, 중복 최소화", "커버 유지, 중복 ↓"],
           ["3. 보완 샘플링", "알고리즘-데이터 공동", "남은 틈새 메움", "최종 커버 100% + 중복 최저"]])

    h2(doc, "8.1 1단계 요청 사항 (동료에게)")
    bullet(doc, "도장 모델(waypoint 각각 접촉 영역 합집합)만으로 FEASIBLE 100% 커버되는 샘플링")
    bullet(doc, "쓸고 지나감 가정 없이 평가")
    bullet(doc, "중복 많아도 괜찮음 (알고리즘이 제거)")
    bullet(doc, "IK/충돌 통과 waypoint만")
    bullet(doc, "접촉 모드, 공구 자세, PAD_INDENT 0.5mm 가정 그대로")

    h2(doc, "8.2 2단계 — 알고리즘 개선 아이디어")
    bullet(doc, "순서 기반 set cover: '선택 후 line 정렬' 대신 '순서 가정 하에서 중복 제거'")
    bullet(doc, "line 묶음 단위 선택: 공정 연속성 보존 + 중복 추가 감소")
    bullet(doc, "보완 waypoint 추가 패스: set cover 후 미커버 face 근처 waypoint 재추가")

    # 9. 분석 시각화
    h1(doc, "9. 결과 분석 시각화 (region_cover)")

    h2(doc, "9.1 미커버 영역 + 중복 히트맵")
    add_image(doc, "visualizations/region_cover_analysis.png", width_cm=16)
    para(doc, "왼쪽: 안 덮인 face(빨강, 49.15 cm²). 오른쪽: 방문 횟수 히트맵 "
              "(초록 1회, 노랑 2회, 주황 3회, 빨강 4+회).")

    h2(doc, "9.2 네 각도 뷰 — 중복 히트맵")
    add_image(doc, "visualizations/region_cover_multiview.png", width_cm=16)

    h2(doc, "9.3 영역별 커버/중복 분포")
    add_image(doc, "visualizations/region_cover_per_region.png", width_cm=16)

    h2(doc, "9.4 영역별 패턴 분석")
    para(doc, "커버 낮은 영역 top 10:")
    table(doc, ["region", "cov %", "ov %", "면적 (cm²)"],
          [[24, "0.0", "0.0", "0.0"],
           [22, "0.0", "0.0", "0.2"],
           [74, "0.0", "0.0", "0.6"],
           [48, "0.0", "0.0", "0.3"],
           [127, "0.0", "0.0", "0.3"],
           [138, "0.0", "0.0", "0.6"],
           [120, "0.0", "0.0", "0.3"],
           [27, "17.3", "0.0", "1.1"],
           [49, "29.6", "100.0", "0.6"],
           [103, "31.1", "6.1", "0.3"]])
    para(doc, "중복 심한 영역 top 10:")
    table(doc, ["region", "ov %", "cov %", "면적 (cm²)"],
          [[132, "100.0", "100.0", "1.6"],
           [135, "100.0", "100.0", "0.4"],
           [49, "100.0", "29.6", "0.6"],
           [70, "89.3", "100.0", "1.6"],
           [97, "88.2", "89.0", "3.7"],
           [63, "83.5", "100.0", "0.3"],
           [35, "81.7", "94.4", "0.6"],
           [130, "81.6", "95.0", "4.2"],
           [76, "74.9", "96.4", "1.7"],
           [32, "72.6", "94.1", "1.1"]])

    h2(doc, "9.5 패턴 요약")
    bullet(doc, "큰 영역은 대체로 커버·중복 모두 양호")
    bullet(doc, "작은 영역(<1 cm²)이 양극단으로 쏠림: 커버 0% 또는 중복 100%")
    bullet(doc, "원인: greedy set cover는 작은 영역에서 세밀한 조정이 어렵고, "
                "waypoint 하나가 영역 전체를 덮거나 아예 못 덮는 극단으로 치우침")
    bullet(doc, "개선 여지: region 97(3.7 cm², 중복 88%)와 region 130(4.2 cm², 중복 81%)처럼 "
                "면적이 있으면서 중복이 큰 영역은 추가로 waypoint를 더 지울 수 있음")

    # 10. 산출물
    h1(doc, "10. 산출물")

    h2(doc, "10.1 스크립트")
    bullet(doc, "scripts/car_door_to_surface.py — v2.6 npz → surface CSV 변환")
    bullet(doc, "scripts/run_cardoor_line_order.py — line 순서 유지 IK 솔버")
    bullet(doc, "scripts/run_cardoor_hrchy_only.py — hrchy_gtsp 솔버 러너")
    bullet(doc, "scripts/region_set_cover.py — 제안 알고리즘")
    bullet(doc, "scripts/eval_contact_area.py — 도장 모델 평가")
    bullet(doc, "scripts/eval_region_sweep.py — 영역 내 쓸고 지나감 평가")
    bullet(doc, "scripts/make_coverage_gif.py — 커버 진행 GIF")
    bullet(doc, "scripts/make_robot_gif.py — 로봇 동작 GIF")
    bullet(doc, "scripts/analyze_region_cover.py — 미커버/중복 분석 시각화")

    h2(doc, "10.2 모션 CSV")
    bullet(doc, "motions/panda_line_order_20261008_155556.csv")
    bullet(doc, "motions/panda_hrchy_gtsp_20261008_152555.csv")
    bullet(doc, "motions/panda_region_cover_20261008_162216.csv")

    h2(doc, "10.3 시각화")
    bullet(doc, "visualizations/coverage_region_cover.gif — 커버 상태 색변화")
    bullet(doc, "visualizations/robot_region_cover.gif — 공구 디스크 이동")
    bullet(doc, "visualizations/region_cover_analysis.png — 미커버 + 중복 히트맵")
    bullet(doc, "visualizations/region_cover_multiview.png — 4각도 뷰")
    bullet(doc, "visualizations/region_cover_per_region.png — 영역별 분포")

    # 11. 결론
    h1(doc, "11. 결론")
    para(doc, "현재 데이터(v2.6) 하에서 영역별 greedy set cover가 세 지표 모두에서 유리:")
    bullet(doc, "waypoint 25% 감소 (공정 시간 단축)")
    bullet(doc, "중복 10%p 감소 (26.4% → 22.5%)")
    bullet(doc, "커버는 상한에 2.4%p 아래 유지 (96.4%)")
    bullet(doc, "sweep 수 56% 감소 (199 → 88, 공구 들어 올림 횟수 축소)")

    para(doc, "다만 알고리즘의 진정한 성능 비교를 위해서는 '도장 모델로 100% 가능한' 샘플링 데이터가 필요. "
              "현재 데이터는 쓸고 지나감 가정에 커버가 묶여 있어 자유도가 제한됨.")

    doc.save(OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
