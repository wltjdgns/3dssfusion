---
title: Azure Kinect DK 실시간 객체 탐지 시스템
date: 2026-05-21
status: approved
---

# Azure Kinect DK 실시간 객체 탐지 시스템 설계

## 개요
Azure Kinect DK를 이용한 실시간 객체 탐지/분류 환경. 범용 탐지 기반 → 원거리 고속 소형 물체(드론) 탐지로 확장 가능한 모듈형 구조.

## 아키텍처

### 3가지 동작 모드
- **rgb**: YOLOv11 + RT-DETRv2 병렬 추론
- **depth**: PointPillars (Point Cloud 기반 3D 탐지)
- **fusion**: RGB + Depth Late Fusion (Weighted Box Fusion)

### 디렉토리 구조
```
C:\jsh\lig\
├── main.py                         # 실행 진입점
├── config.yaml                     # 전체 시스템 설정
├── setup.py
├── environment.yml
├── kinect/                         # Azure Kinect 캡처 모듈
│   └── capture.py
├── models/                         # 모델 로더/래퍼
│   ├── yolo_detector.py
│   ├── rtdetrv2_detector.py
│   ├── grounding_dino_detector.py
│   └── pointpillars_detector.py
├── pipeline/                       # 실행 파이프라인
│   ├── rgb_pipeline.py
│   ├── depth_pipeline.py
│   └── fusion_pipeline.py
├── fusion/                         # WBF 퓨전 로직
│   └── weighted_box_fusion.py
├── postprocess/                    # 추적, 시각화, 로깅
│   ├── tracker.py
│   ├── visualizer.py
│   └── logger.py
├── utils/                          # 공통 유틸리티
│   ├── config_loader.py
│   └── point_cloud_utils.py
├── weights/                        # 모델 가중치 (git-tracked dir, files excluded)
│   ├── yolo/
│   ├── rtdetrv2/
│   ├── grounding_dino/
│   └── pointpillars/
├── data/results/                   # 출력 영상/로그
├── third_party/
│   ├── OpenPCDet/
│   └── GroundingDINO/
├── docs/superpowers/specs/         # 설계 문서
└── tests/
```

### 모델 선정
| 용도 | 모델 | 비고 |
|------|------|------|
| RGB 1차 | YOLOv11-x | 속도/정확도 균형, Ultralytics API |
| RGB 2차 | RT-DETRv2-r50vd | Transformer 기반, 높은 mAP |
| RGB 개방형 | Grounding DINO SwinT | 텍스트 프롬프트 기반, 필요시 활성화 |
| Depth | PointPillars via OpenPCDet | Point Cloud → 3D BBox |

### Fusion 전략
**Late Fusion (Detection-level)** 채택 이유:
- Kinect RGB(1080p) vs Depth(640×576) 해상도 불일치로 Early Fusion 부적합
- 각 모달리티 독립 추론 → 지연 없는 RGB 처리 가능
- WBF(Weighted Box Fusion): **RGB 0.65 : Depth 0.35**

```
RGB Stream  ──→ [YOLOv11] ──┐
                             ├──→ [WBF Fusion] ──→ [ByteTrack] ──→ Display
            ──→ [RT-DETRv2] ─┤
Depth Stream ──→ [PointPillars] ─┘
```

### 성능 목표 (RTX 3090 기준)
| 모드 | 목표 FPS | VRAM |
|------|----------|------|
| RGB 단독 | 35~50 FPS | ~4 GB |
| Fusion 전체 | 15~20 FPS | ~8 GB / 24 GB |

### 드론 탐지 확장 경로
| Phase | 기간 | 작업 |
|-------|------|------|
| Phase 1 | 즉시 | `config.yaml` 프롬프트 변경 (`gdino_prompt`) |
| Phase 2 | 1~2주 | YOLOv11 드론 데이터셋 fine-tune 후 가중치 교체 |
| Phase 3 | 2~4주 | SAHI 타일링 통합, `drone_detector.py` 추가 |
| Phase 4 | 4~8주 | 멀티카메라 동기화, 궤적 예측 모듈 추가 |

## 핵심 설계 결정

### Point Cloud 전처리
- 유효 범위: 0.3m ~ 5.5m (Kinect NFOV 특성)
- Voxel Downsample: 0.02m (과밀 포인트 제거)
- Ground Removal: z < -0.3m 필터링

### Depth → RGB 좌표 변환
Kinect SDK `k4a_calibration_3d_to_2d` API를 통해 3D bbox → 2D 투영 후 WBF 입력으로 사용.

### 메모리 관리
- Frame Queue: 최대 3프레임 (지연 최소화)
- Half Precision (FP16): RGB 모델 전체 적용
- `skip_depth_every_n`: Depth 추론 프레임 스킵으로 GPU 부하 조절
