# Car Door 커버리지 파이프라인 수정 및 결과

작성일: 2026-10-09

## 1. 목표

- **높은 커버율**: feasible face 를 가능한 많이 덮기
- **낮은 중복(overlap)**: 공구가 같은 자리 반복해서 자원 낭비 줄이기
- **관절 움직임 절약**: hrchy_gtsp 로 영역 내 waypoint 순서 최적화

## 2. 입력 데이터

**데이터셋**: `meshes/car_door_dir0/` (v2.7, 2026-10-08, 김재연 전달)

- 작업물: 자동차 도어 (mesh.npz)
- **방문 waypoint 2,917개** (tool_pose_valid & accessibility)
- **영역 144개** (region_ids)
- **feasible face 5,239개**, 작업면 총 **1,377.94 cm²**
- 중심 접촉 32개, **가장자리 접촉 2,885개**
- 데이터셋 점접촉 커버 상한: **99.41%**
- 좌표계: W(STL) → B(로봇 베이스), T_B_W / T_L_C 는 metadata.json

### 공구 조건
- 패드 디스크 반지름 R_TOOL = **5 cm**
- 패드 압입량 (PAD_INDENT):
  - 중심 접촉 (contact_mode=1): **0.5 mm**
  - 가장자리 접촉 (contact_mode=2): **1.0 mm**

## 3. 파이프라인 설계

[scripts/run_full_pipeline.py](scripts/run_full_pipeline.py)

입력: 사전 선별된 waypoint 세트 (`region_set_cover.py` 로 2917 → **1179** 로 축소)

### 영역별 처리
각 영역 rid 에 대해:
1. **선별본에서 rid 소속 waypoint 수집** (K개)
2. **Sweep**: K≥15 이면 hrchy_gtsp 로 waypoint 순서 결정, 아니면 NN
3. **Sweep 커버 측정**: 공구 디스크 ∩ 패드밴드 footprint + waypoint 간 2mm 보간
4. **Stamp 보완**: 그 영역 face 중 미커버분을 **2917 중 unused** 에서 greedy pick
5. **영역 간 lift-off**: 다음 영역으로 이동 시 공중 이동 (코드상 명시 안 됨, 영역간 거리로 자연 분리)

### 전역 cleanup stamp (수정 후 추가)
모든 영역 처리 후, 전체 feasible 중 미커버 face 가 남으면 **2917 중 unused 로 추가 greedy stamp**

## 4. 발견한 버그 및 수정

### 버그 1: PAD_INDENT 하드코딩 (0.5mm)
**증상**: 영역 49, 66, 104, 74 등이 0% 커버. 원인은 **가장자리 접촉 waypoint 를 0.5mm 밴드로 평가**하여 데이터셋 가정(1mm) 이탈.

**수정**: waypoint 별 `contact_modes` 기반 가변 PAD_INDENT
```python
h_all = np.where(cm_all == 1, 0.0005, 0.0010)
```
`footprint_mask`, `sweep_visit` 모두 waypoint 별 h 사용. 보간 구간은 양 끝 h 중 큰 값.

### 버그 2: 영역 간 커버 공유 안 됨 (리셋 버그)
**증상**: 영역 72 (n_src=1, nFace=4) 는 자기 영역 waypoint 로 1mm 밴드로도 커버 0%. 그러나 **다른 영역(23, 71, 76) waypoint 가 그 face 를 덮을 수 있음**이 확인됨.

**원인**: 영역 처리 때마다 `swept_r` 를 새로 생성 → 이전 영역이 지나가며 덮은 face 가 반영 안 됨 → 중복 stamp 낭비 or 영원히 미커버.

**수정**: 전역 face 커버 마스크 `global_face_covered` 추가, 영역별로 공유
```python
# 초기화
global_face_covered = np.zeros(len(F), dtype=bool)

# 영역 sweep 평가 때 반영
swept_r |= global_face_covered[f_in]

# 영역 처리 끝에 업데이트
sw_mask_all = sweep_visit(..., centers_B_all_f, h_all[sweep_abs])
global_face_covered |= (sw_mask_all & feas)
for wp in stamp_pick:
    fp = footprint_mask(..., centers_B_all_f, h_all[wp])
    global_face_covered |= (fp & feas)
```

### 추가 보완: 전역 cleanup stamp
영역 for-loop 끝에, 전체 feasible 중 아직 미커버 face 를 2917 중 unused waypoint 로 greedy 로 메꿈.

## 5. 결과

### 5.1 메인 비교 (mode-aware 평가, PAD_INDENT 가변)

| 모션 | T | coverage | overlap/cov | overlap/feas | 평균 중복 | max |
|---|---|---|---|---|---|---|
| **2917 전수 (upper bound)** | 2917 | **99.41%** | 94.45% | 93.90% | 9.57× | 69 |
| baseline_dir0 | 2897 | 99.40% | 94.37% | 93.81% | 9.55× | 69 |
| region_cover_dir0 (선별만) | 1179 | 98.26% | 87.13% | 85.61% | 4.09× | 18 |
| full_dir0 (수정 전, 리셋 버그) | 1307 | 98.63% | 90.85% | 89.61% | 4.85× | 21 |
| **full_dir0_fix (버그 수정)** | **1236** | 98.57% | 89.24% | 87.96% | 4.38× | 19 |
| **full_dir0_v2 (+ cleanup)** | **1241** | **98.69%** | 89.07% | 87.90% | 4.32× | 19 |

### 5.2 중복 분포 (면 중심 평가)

| 중복 횟수 | 2917 전수 | **v2 (1241)** |
|---|---|---|
| 0× (미커버) | 0.59% | 1.31% |
| 1× | 5.51% | 10.78% |
| 2× | 11.83% | 21.17% |
| 3× | 14.12% | 20.57% |
| 4-5× | 20.30% | 23.14% |
| 6-8× | 15.62% | 12.06% |
| 9-15× | 13.66% | 10.50% |
| 16+× | **18.37%** | **0.46%** |

**핵심**: 과도 중복(16+×)이 18.37% → 0.46% 로 거의 사라짐. **1~5× 집중 분포로 균일성 개선**.

### 5.3 2mm 격자 샘플링 평가 (더 정확)

데이터셋 README 권장. feasible face 를 2mm 격자로 쪼개 32,245 샘플로 평가.

| 모션 | coverage | overlap/cov | 평균 중복 |
|---|---|---|---|
| 2917 전수 | 99.37% | 94.81% | 9.67× |
| **v2 (1241)** | **97.60%** | 89.16% | 4.35× |

면 중심 평가 (98.69%) 대비 **1.09%p 하향** — 큰 face 들이 공구 디스크 가장자리에 걸릴 때 과대평가 분을 제거한 결과.

## 6. 핵심 성과

### vs baseline (2897 waypoint)
- waypoint **-57% (2897 → 1241)**
- coverage **-0.71%p** (99.40% → 98.69%)
- **중복 반 이하** (9.55× → 4.32×)
- **과도 중복 (16+×) 거의 소멸** (18.4% → 0.5%)

### vs region_cover (선별만, 1179)
- waypoint **+62 (1179 → 1241)**
- coverage **+0.43%p** (98.26% → 98.69%)
- 중복 비슷 (4.09× → 4.32×)

### 상한 (99.41%) 대비 미커버
- v2 미커버: 1.31% (18 cm²)
  - 그 중 **2917 전수로도 못 덮는 face**: 8.14 cm² (0.59% = 데이터셋 한계)
  - 2917 중 unused 로 복구 가능하지만 못 가져온 face: 9.94 cm² (0.72%)

## 7. 현업 지표 관점 평가

### 산업계/논문의 overlap 정의
- **스프레이 50% overlap**: 공구 폭 대비 패스 간격 설계값. 평균 중복 2× 수준.
- **Overlap Ratio (OR, 논문/산업 평가)**: `(≥2회 커버 면적) / (전체 작업 면적)` → 사용자의 **overlap/feasible = 87.90%** 에 해당
- `overlap/covered` 는 보조 지표 (분모가 커버된 면적만)

### 현업 핵심 지표 (현재 미반영)
1. **CV%** (표면 거칠기 변동계수): 폴리싱 품질 등급 (A: ≤5%, D: >8%) — 실측 필요
2. **Force distribution rate** (접촉 압력 균일성)
3. **Path accuracy / cornering deviation / speed fluctuation** (ISO 9283)
4. **커버 횟수의 표준편차** (현재 평균만, 분산은 분포로 간접)

### 현재 알고리즘 평가
**강점**:
- Coverage 98.69% ≒ complete coverage (상한 99.41% 대비 거의 최대)
- 과도 중복 소멸로 작업 시간 효율
- 자유곡면 자동 처리 (티칭 대비 셋업 시간 큰 이점)

**아쉬움**:
- **1× 영역 10.78%** → 폴리싱 품질 리스크 (최소 2~3× 필요)
- 공구 접촉 압력 모델 없음
- waypoint 속도/가속도 균일성 미평가

## 8. 다음 액션 후보

- **1× 영역 줄이기**: region_set_cover 선별 기준 완화 (여유분 waypoint 추가) 또는 영역 경계 반복 패스
- **격자 샘플링 평가 정식 도입**: 데이터셋 저자 권장. 지금 결과도 포함.
- **중복 변동계수 지표 추가**: `std(visit_count)` 또는 CV%
- **dir90 데이터셋 동일 처리**: 양방향 결과 비교 (현재는 dir0만)
- **cleanup `used_global` 제약 완화**: v3 시도했으나 `unused.remove(best)` 때문에 효과 없음 — 중복 waypoint 허용 로직으로 재시도 가능

## 9. 코드 변경 요약

**[scripts/run_full_pipeline.py](scripts/run_full_pipeline.py)**:
- `PAD_INDENT_CENTER/EDGE` 상수, `h_all` 배열 추가
- `footprint_mask(..., h)` 로 가변 h 지원
- `sweep_visit(..., h_arr)` waypoint 별 h
- `global_face_covered` 전역 커버 마스크 추가 (영역 간 공유)
- 각 영역 sweep 평가 시 `swept_r |= global_face_covered[f_in]`
- 영역 처리 끝에 `global_face_covered` 업데이트 (sweep 보간 + stamp)
- Save 직전 **전역 cleanup stamp pass** 추가

**신규 평가/분석 스크립트**:
- [scripts/analyze_motion_per_region.py](scripts/analyze_motion_per_region.py) — 영역별 커버 분석
- [scripts/debug_region_coverage.py](scripts/debug_region_coverage.py) — 특정 영역 디버깅
- [scripts/verify_pad_indent_hypothesis.py](scripts/verify_pad_indent_hypothesis.py) — 0.5/1.0/mix 비교
- [scripts/find_cross_region_covers.py](scripts/find_cross_region_covers.py) — cross-region 커버 가능성
- [scripts/check_stuck_regions_in_motion.py](scripts/check_stuck_regions_in_motion.py) — 리셋 버그 수정 검증
- [scripts/trace_uncoverable_face.py](scripts/trace_uncoverable_face.py) — 미커버 face 원인 추적
- [scripts/eval_motion_mode_aware.py](scripts/eval_motion_mode_aware.py) — mode-aware 커버 평가
- [scripts/eval_grid_sampling.py](scripts/eval_grid_sampling.py) — 2mm 격자 샘플링 평가

## 10. 산출 모션 파일

| 파일 | 설명 |
|---|---|
| `motions/panda_full_dir0_20261009_011001.csv` | 수정 전 (리셋 버그 상태) |
| `motions/panda_full_dir0_fix_20261009_083611.csv` | PAD_INDENT + 리셋 버그 수정 |
| `motions/panda_full_dir0_v2_20261009_090106.csv` | **최종 (+ cleanup stamp)** |
| `motions/panda_full_dir0_v3_20261009_092857.csv` | cleanup 블락 완화 시도 (효과 미미) |

---

# 실행 가이드 (처음 받은 사람용)

## A. 환경 설정

### A.1 OS / 의존성
- Ubuntu 24.04 (24.04 외는 미검증)
- Python 3 (numpy, scipy)
- Rust (relaxed_ik 바이너리 빌드용)
- make, gcc (GLKH 빌드용)

### A.2 Rust 빌드 (최초 1회)
```bash
cd arm_coverage
cargo build --release
```
빌드 산출물은 `target/` 아래 (이미 커밋돼 있으면 스킵 가능).

### A.3 GLKH 빌드 (최초 1회)
```bash
cd GLKH
make
cd ..
```

### A.4 데이터셋 확인
작업물 데이터는 `meshes/car_door_dir0/`, `meshes/car_door_dir90/` 에 이미 포함:
- `surface_points.npz` — 2917 waypoint 세트
- `mesh.npz` — 메시 (vertices, faces, face_region_ids, face_status)
- `metadata.json` — T_B_W, T_L_C 등 좌표 변환
- `regions.json` — 영역별 waypoint 인덱스

**로봇 설정**: `configs/example_settings/panda.yaml` 사용 (URDF: `configs/urdfs/panda.urdf`)

## B. 파이프라인 전체 실행 (3단계)

### Step 1 — 영역별 greedy 선별 (2917 → ~1180)

```bash
DATA_DIR=meshes/car_door_dir0 TAG=region_cover_dir0 \
python scripts/region_set_cover.py
```

**출력**: `motions/panda_region_cover_dir0_<timestamp>.csv` (약 1,179개 waypoint)

**내용**: 각 영역별로 greedy set cover 로 "그 영역 face 를 덮기에 충분한" waypoint 를 뽑음. 공구 접촉 조건 (R=5cm 디스크, 가장자리 접촉 1mm 밴드) 반영됨.

### Step 2 — Full pipeline (영역별 sweep + stamp + 전역 cleanup)

Step 1 산출 CSV 경로를 `MOTION_IN` 으로 넘김:

```bash
DATA_DIR=meshes/car_door_dir0 \
MOTION_IN=motions/panda_region_cover_dir0_<timestamp>.csv \
TAG=full_dir0 \
N_MIN=15 TIME_LIMIT=20 NUM_SAMPLES=20 \
python -u scripts/run_full_pipeline.py 2>&1 | tee pipeline.log
```

**환경변수 설명**:
- `DATA_DIR` — 데이터셋 폴더 (dir0 또는 dir90)
- `MOTION_IN` — Step 1 산출 CSV (선별본)
- `TAG` — 출력 파일 식별자
- `N_MIN` — 영역당 waypoint 수 임계값 (≥15 면 hrchy_gtsp, 미만이면 NN)
- `TIME_LIMIT` — hrchy_gtsp 영역당 시간 제한 (초)
- `NUM_SAMPLES` — hrchy_gtsp 샘플 수

**실행 시간**: 약 10분 (영역 ~155개, dir0 기준)

**출력**:
- `motions/panda_full_dir0_<timestamp>.csv` — 최종 모션 (약 1240 waypoint)
- 콘솔 로그에 영역별 "cov sweep% → final%" 와 cleanup stamp 수 출력

### Step 3 — 평가

#### 3-A. mode-aware 커버/중복 평가 (면 중심 기준, 기본값)

```bash
MOTIONS=motions/panda_full_dir0_<timestamp>.csv \
python scripts/eval_motion_mode_aware.py
```

여러 모션 비교하려면 쉼표로:
```bash
MOTIONS="motions/panda_region_cover_dir0_...csv,motions/panda_full_dir0_...csv" \
python scripts/eval_motion_mode_aware.py
```

**출력 항목**:
- `coverage` — 커버된 면적 / 전체 feasible
- `overlap/covered` — ≥2회 커버 면적 / 커버 면적
- `overlap/feasible` — ≥2회 커버 면적 / 전체 feasible (**논문/산업 표준 지표**)
- `mean multiplicity` — 평균 중복 횟수
- `max` — 최대 중복 횟수

**해석**:
- coverage: 데이터셋 상한 99.41% 와 비교. 98% 중반이면 OK.
- overlap/feasible: 50% overlap 스프레이 설계 ≈ 평균 2× ≈ overlap/feas 50% 수준. 폴리싱은 보수적으로 더 높아도 됨.

#### 3-B. 영역별 커버 (어느 영역이 미커버인지 보고 싶을 때)

```bash
MOTION=motions/panda_full_dir0_<timestamp>.csv \
python scripts/analyze_motion_per_region.py
```

워스트 30개 영역 테이블 + 전체 집계. 영역별 K, nFace, cov%, gap_cm2 출력.

#### 3-C. 2mm 격자 샘플링 평가 (더 정확, 선택)

```bash
SAMPLE_SPACING=0.002 \
MOTIONS=motions/panda_full_dir0_<timestamp>.csv \
python scripts/eval_grid_sampling.py
```

면 중심 평가보다 1~2%p 낮은 수치 나옴 (= 더 정확한 접촉 면적 기반).

#### 3-D. 중복 분포 상세 (optional)

`scripts/eval_motion_mode_aware.py` 안의 bin breakdown 로직 참고. 또는 격자 평가 스크립트가 1x/2x/3x/4-5x/6-8x/9-15x/16+x 로 자동 분포 출력.

## C. 빠른 재현 (모든 결과물 한 번에)

Shell 스크립트로 전체 재현:

```bash
cd arm_coverage

# 1. 선별
DATA_DIR=meshes/car_door_dir0 TAG=region_cover_dir0 \
  python scripts/region_set_cover.py
SEL=$(ls -t motions/panda_region_cover_dir0_*.csv | head -1)

# 2. Full pipeline
DATA_DIR=meshes/car_door_dir0 MOTION_IN=$SEL TAG=full_dir0 \
  python -u scripts/run_full_pipeline.py 2>&1 | tee pipeline.log
FULL=$(ls -t motions/panda_full_dir0_*.csv | head -1)

# 3. 평가
MOTIONS="$SEL,$FULL" python scripts/eval_motion_mode_aware.py
MOTION=$FULL python scripts/analyze_motion_per_region.py
```

## D. dir90 데이터셋도 돌리려면

위 커맨드에서 모든 `dir0` → `dir90` 로 바꿈. 데이터셋 (`meshes/car_door_dir90/`) 과 선별본 (`motions/panda_region_cover_dir90_...csv`) 둘 다 교체.

## E. 트러블슈팅

- **"ModuleNotFoundError: region_set_cover"** → `cd arm_coverage` 안에서 실행했는지 확인. 스크립트는 `sys.path.insert(0, "scripts")` 로 상대 경로 의존.
- **hrchy_gtsp timeout** → `TIME_LIMIT` 늘리기. 기본 20초가 큰 영역에선 부족할 수 있음.
- **커버 수치가 예상보다 낮음** → `contact_modes` 값 확인. 평가 코드가 waypoint 별 가변 PAD_INDENT 를 쓰는지 확인.
- **IK 실패 로그** → `region_set_cover.py` 는 waypoint 별로 IK 풀고 실패 시 스킵. 로그에 `IK fail=X` 로 표시.
- **GLKH 산출물이 남음** (`GLKH/*.tour`, `GLKH/*.par` 등) → `run_full_pipeline.py` 가 영역별로 지우긴 하지만 중단 시 남을 수 있음. 수동으로 `rm GLKH/joint_gtsp*.tour` 로 정리.

## F. 주요 파라미터

| 파라미터 | 위치 | 기본값 | 설명 |
|---|---|---|---|
| `R_TOOL` | `eval_contact_area.py` | 0.05 m | 공구 디스크 반지름 |
| `PAD_INDENT_CENTER` | `run_full_pipeline.py` | 0.0005 m | 중심 접촉 밴드 반두께 |
| `PAD_INDENT_EDGE` | `run_full_pipeline.py` | 0.0010 m | 가장자리 접촉 밴드 반두께 |
| `INTERP_STEP` | `run_full_pipeline.py` | 0.002 m | sweep 보간 간격 |
| `N_MIN` | 환경변수 | 15 | hrchy_gtsp vs NN 분기점 |
| `TIME_LIMIT` | 환경변수 | 20 s | hrchy_gtsp 영역당 시간 |
| `NUM_SAMPLES` | 환경변수 | 20 | hrchy_gtsp 샘플 수 |
| `SAMPLE_SPACING` | 환경변수 | 0.002 m | 격자 평가 간격 |
