# Azure Kinect DK 실시간 객체 탐지 시스템

Azure Kinect DK의 RGB 카메라와 Depth(ToF) 센서를 동시에 활용하여  
**YOLOv11 / RT-DETR-r50vd / Grounding DINO (RGB)** + **PointPillars (Depth)** 를  
실시간으로 구동하고, Weighted Box Fusion으로 두 스트림을 통합하는 모듈형 파이프라인입니다.

---

## 목차

1. [시스템 요구사항](#시스템-요구사항)
2. [설치 가이드](#설치-가이드)
3. [빠른 시작](#빠른-시작)
4. [동작 모드 상세](#동작-모드-상세)
5. [모델별 설정 가이드](#모델별-설정-가이드)
6. [성능 튜닝](#성능-튜닝)
7. [드론 탐지 확장](#드론-탐지-확장)
8. [프로젝트 구조](#프로젝트-구조)
9. [모듈 설명](#모듈-설명)
10. [트러블슈팅](#트러블슈팅)

---

## 시스템 요구사항

### 하드웨어

| 항목 | 최소 | 권장 (메인) |
|---|---|---|
| **GPU** | GTX 1080 (8 GB VRAM) | RTX 3090 (24 GB) |
| RAM | 16 GB | 32 GB |
| USB | **USB 3.0 필수** | USB 3.1 Gen2 |
| OS | Windows 10 64-bit | Windows 10/11 64-bit |

> Azure Kinect DK는 반드시 **USB 3.0** 이상에 연결해야 합니다. USB 2.0 포트에서는 인식되지 않습니다.

### 소프트웨어

| 항목 | 버전 | 비고 |
|---|---|---|
| Python | 3.10.x | Conda 환경 자동 설치 |
| CUDA | 11.8 | nvcc 확인: `nvcc --version` |
| cuDNN | 8.x | PyTorch 설치 시 자동 |
| Azure Kinect SDK | **v1.4.1** | 수동 설치 필요 |
| [Visual Studio 2022 Build Tools](https://aka.ms/vs/17/release/vs_BuildTools.exe) | **2022 필수** | OpenPCDet CUDA 빌드 전용 (VS 2023+ 불가) |
| Miniconda / Anaconda | 최신 | 환경 관리 |

---

## 설치 가이드

### 0단계. Miniconda 설치

[공식 다운로드 페이지](https://docs.conda.io/en/latest/miniconda.html)에서 **Windows 64-bit 최신 버전**을 다운로드하여 설치합니다.

설치 후 터미널(CMD 또는 bash)에서 다음을 확인합니다:

```bash
conda --version
```

> 설치 경로 기본값: `C:\Users\<사용자명>\miniconda3`  
> PATH 미등록 시 설치 관리자에서 **"Add Miniconda3 to my PATH"** 옵션을 체크하거나, 이후 `conda init` 명령으로 초기화합니다.
>
> ```bash
> conda init bash   # Git Bash 사용 시
> conda init cmd.exe
> ```

### 1단계. Azure Kinect SDK 설치

[공식 다운로드 페이지](https://learn.microsoft.com/ko-kr/azure/kinect-dk/sensor-sdk-download)에서 **Azure Kinect SDK 1.4.1** Windows 패키지를 다운로드합니다.

설치 후 DLL 경로가 시스템 PATH에 자동 등록됩니다. 미등록 시 수동 추가:
```
C:\Program Files\Azure Kinect SDK v1.4.1\sdk\windows-desktop\amd64\release\bin
```

### 2단계. Conda 환경 구성

최신 Miniconda(conda 26+)는 첫 실행 시 채널 Terms of Service 동의가 필요합니다. `conda env create` 전에 먼저 실행합니다:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

그 후 환경을 생성합니다:

```bash
conda env create -f environment.yml
conda activate kinect_det
```

`environment.yml`에 정의된 주요 패키지:

| 패키지 | 버전 | 설치 방법 | 역할 |
|---|---|---|---|
| python | 3.10.14 | conda | 런타임 |
| torch | 2.1.2+cu118 | pip (pytorch.org whl) | 딥러닝 프레임워크 |
| torchvision | 0.16.2+cu118 | pip (pytorch.org whl) | 이미지 변환 |
| torchaudio | 2.1.2+cu118 | pip (pytorch.org whl) | 오디오 처리 |
| opencv-python-headless | 4.9.0.80 | pip | 이미지 처리 |
| ultralytics | 8.3.40 | pip | YOLOv11 |
| transformers | 4.47.1 | pip | RT-DETR |
| spconv-cu118 | 2.3.6 | pip | OpenPCDet 의존성 |
| open3d | 0.18.0 | pip | 포인트 클라우드 시각화 |
| supervision | 0.23.0 | pip | ByteTrack 추적 |
| ensemble-boxes | 1.0.9 | pip | Weighted Box Fusion |
| onnxruntime-gpu | 1.18.1 | pip | ONNX 추론 |

> **참고**: PyTorch는 conda 채널 대신 pytorch.org 공식 pip wheel을 사용합니다.  
> 이는 Windows에서 conda 빌드의 `libiomp5md.dll` 버전 불일치로 인한 import 오류를 방지하기 위함입니다.

### 3단계. 추가 패키지 설치 (Post-install)

아래 3개 패키지는 의존성 충돌로 인해 `conda env create` 후 별도로 설치해야 합니다.

#### sahi (소형 물체 분할 추론)

```powershell
# sahi는 opencv-python(non-headless)을 요구하나, 런타임에는 headless로 동작 가능
pip install sahi==0.11.19 --no-deps
```

#### groundingdino-py (Grounding DINO)

```powershell
# supervision==0.6.0 충돌 우회 + Windows cp949 인코딩 오류 방지
$env:PYTHONUTF8 = "1"
pip install groundingdino-py==0.4.0 --no-deps
```

#### pyk4a (Azure Kinect Python 래퍼)

pyk4a는 C 확장을 컴파일해야 하므로 **VS Build Tools + Azure Kinect SDK** 설치 후 진행합니다.

```cmd
:: x64 Native Tools Command Prompt 또는 vcvars64.bat 환경에서 실행
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
set MSSdk=1
pip install git+https://github.com/etiennedub/pyk4a.git@1.5.0
```

> 시스템에 Azure Kinect SDK v1.4.1이 설치되어 있어야 빌드 시 자동 감지됩니다.

### 4단계. 설치 검증

```bash
python verify_env.py
```

모든 핵심 패키지(torch, torchvision, cv2, ultralytics, supervision, transformers, open3d, spconv, onnxruntime, sahi, groundingdino, pyk4a)를 임포트하고 결과를 출력합니다.  
CUDA available: True가 표시되면 GPU 추론 준비 완료입니다.

### 5단계. 가중치 다운로드

```bash
python scripts/download_weights.py --model all      # 전체
python scripts/download_weights.py --model yolov11  # YOLOv11만
python scripts/download_weights.py --model grounding_dino
```

> PointPillars 가중치는 자동 다운로드가 불가합니다. [OpenPCDet 모델 동물원](https://github.com/open-mmlab/OpenPCDet)에서 KITTI 학습 가중치를 수동 다운로드 후 `weights/pointpillars/` 에 배치하세요.

### 6단계. OpenPCDet 설치 (Depth 모드 전용)

Depth / Fusion 모드에서 PointPillars를 사용하려면 OpenPCDet 빌드가 필요합니다.  
RGB 단독 모드만 사용한다면 이 단계는 건너뛸 수 있습니다.

**전제 조건: Visual Studio 2022 Build Tools 설치 필수**

> CUDA 11.8은 MSVC 14.39 이하 (VS 2022)까지만 지원합니다.  
> VS 2023 이상은 CUDA 11.8과 **근본적으로 호환되지 않습니다** (STL1002 빌드 오류).  
> VS 2026 등 최신 버전이 이미 설치되어 있어도 VS 2022 Build Tools를 **추가 설치**하면 공존 가능합니다.
>
> **VS 2022 Build Tools 설치 방법:**
> 1. 직접 다운로드: **[vs_BuildTools.exe (VS 2022)](https://aka.ms/vs/17/release/vs_BuildTools.exe)**
> 2. 설치 관리자 실행 후 **"C++를 사용한 데스크톱 개발"** 워크로드 선택
> 3. 설치 완료 확인: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\` 폴더 존재 여부

**반드시 Windows CMD에서 실행합니다 (PowerShell 불가).**

```cmd
:: 1. VS 2022 MSVC 환경 활성화
::    VS 2022 외 다른 버전(2023+)이 함께 설치된 경우 VCToolsVersion으로 14.37 고정 필수
set VCToolsVersion=14.37.32822
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
set MSSdk=1

:: 2. OpenPCDet 소스 fetch (레포 클론 시 패치 파일이 이미 존재하므로 git clone 대신 사용)
git -C third_party\OpenPCDet init
git -C third_party\OpenPCDet remote add origin https://github.com/open-mmlab/OpenPCDet.git
git -C third_party\OpenPCDet fetch --depth=1 origin master
git -C third_party\OpenPCDet checkout FETCH_HEAD -- .
:: 패치 파일 복원 (fetch로 덮어씌워진 파일을 우리 레포 버전으로 되돌림)
git checkout -- third_party/OpenPCDet/setup.py
git checkout -- third_party/OpenPCDet/pcdet/ops/iou3d_nms/src/
git checkout -- third_party/OpenPCDet/pcdet/ops/ingroup_inds/src/

:: 3. PyTorch 2.1.2 호환 setuptools 설치 (pkg_resources 오류 방지)
pip install "setuptools==69.5.1"

:: 4. OpenPCDet 의존성 설치 (SharedArray는 Windows 미지원 — 무시)
pip install tensorboardX easydict scikit-image pyquaternion

:: 5. OpenPCDet 빌드
cd third_party\OpenPCDet
python setup.py develop
cd ..\..
```

> **빌드 완료 후 반드시 실행:** setup.py develop 과정에서 `pcdet/version.py`에 따옴표가 제대로 닫히지 않는 버그가 있습니다.  
> 아래 명령으로 수정합니다:
>
> ```bash
> # Git Bash 또는 PowerShell
> python -c "
> p = 'third_party/OpenPCDet/pcdet/version.py'
> open(p, 'w').write('__version__ = \"0.6.0+HEAD\"\n')
> "
> ```
>
> 또는 `third_party/OpenPCDet/pcdet/version.py` 파일을 직접 열어 아래와 같이 수정합니다:
>
> ```python
> __version__ = "0.6.0+HEAD"
> ```

> **VCToolsVersion 확인:** `dir "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\" /b` 로  
> 설치된 툴셋 버전을 확인합니다. `14.37.xxxxx` 항목이 없으면 VS Installer → 수정 → 개별 구성 요소 →  
> `MSVC v143 - VS 2022 C++ x64/x86 빌드 도구 (v14.37-17.7)` 설치 필요.

> **SharedArray 빌드 실패는 무시합니다.** `sys/mman.h` POSIX 헤더를 사용하는 Linux 전용 패키지로,  
> PointPillars 추론(inference)에는 사용되지 않습니다.

---

## 빠른 시작

```bash
# 기본 실행 (config.yaml의 mode 따름, 기본값: fusion)
python main.py

# 모드 지정 실행
python main.py --mode rgb       # RGB 단독 (카메라 없이도 동작 확인 가능)
python main.py --mode depth     # Depth 단독 (OpenPCDet 필요)
python main.py --mode fusion    # RGB + Depth Fusion

# 커스텀 설정 파일
python main.py --config my_config.yaml

# GPU 지정
python main.py --device cuda:0
```

실행 시 콘솔에 다음 정보가 표시됩니다:
- 로드된 모델 목록 및 VRAM 사용량
- 실시간 FPS (rolling average 30프레임)
- 탐지된 객체 수 및 클래스

`ESC` 키로 종료합니다.

---

## 동작 모드 상세

`config.yaml` 최상단의 `mode` 값 하나로 전체 파이프라인이 전환됩니다.

```yaml
mode: "fusion"   # "rgb" | "depth" | "fusion"
```

### RGB 모드 (`mode: "rgb"`)

```
Azure Kinect (Color 1920×1080)
        │
        ├─→ YOLOv11-x (FP16, imgsz=640)  ─┐
        └─→ RT-DETR-r50vd (FP16)          ─┤ → NMS → ByteTracker → Visualizer
            [또는 Grounding DINO]           ─┘
```

- 두 모델을 `ThreadPoolExecutor`로 **병렬 추론**
- `rgb_models.active` 리스트로 활성 모델 선택
- 결과 합산 후 `nms_2d`로 중복 제거

**예상 FPS (RTX 3090):**
- YOLOv11-x 단독: 35~50 FPS
- YOLOv11-x + RT-DETR-r50vd 병렬: 20~30 FPS

---

### Depth 모드 (`mode: "depth"`)

```
Azure Kinect (Depth 640×576, NFOV_UNBINNED, max 5.46m)
        │
        → PointCloudConverter
          (Depth mm → XYZ m, 좌표계 변환: Kinect → LiDAR 표준)
          (지면 제거, 범위 필터링, 다운샘플링)
        │
        → PointPillars (OpenPCDet)
          (BEV voxelization → 3D Bounding Box 예측)
        │
        → project_3d_to_2d (color 이미지에 3D bbox 투영)
        → ByteTracker → Visualizer
```

- 출력: 3D 바운딩 박스 `[cx, cy, cz, l, w, h, yaw]` + 2D 투영 박스
- Point Cloud 범위: X [-5, 5]m, Y [-5, 5]m, Z [-1, 3]m (실내 최적화)

---

### Fusion 모드 (`mode: "fusion"`) — 기본값

```
Color (1920×1080) ──→ [RGB 모델 병렬] ──→ RGB_Detections[]
                                                  │
Depth (640×576) ──→ PointCloudConverter          │
                          │                       │
                          ├──→ PointPillars ──→ 3D_Detections[]
                          │                       │
                          └──→ FrustumProjector ──┘
                                (RGB bbox 영역의 포인트 추출 → 거리 보완)
                                          │
                              FusionDetector (WBF)
                              RGB 0.65 : Depth 0.35
                                          │
                              depth_m 필드가 채워진 통합 결과
                                          │
                              ByteTracker → Visualizer → Logger
```

- **Weighted Box Fusion (WBF)**: 겹치는 박스를 제거하지 않고 가중 평균으로 합산 → NMS보다 정확한 위치
- **depth_m 자동 보완**: 각 탐지 객체의 bbox 내 포인트 클라우드 중앙값 거리를 레이블에 표시
- `depth_confirmation_only: true` 설정 시 Depth는 거리 보완만 수행 (3D 탐지 비활성)

---

## 모델별 설정 가이드

### YOLOv11

```yaml
rgb_models:
  yolov11:
    enabled: true
    weights: "weights/yolo/yolo11x.pt"
    model_size: "x"      # n | s | m | l | x (클수록 정확, 느림)
    imgsz: 640           # 입력 해상도. 소형 물체는 1280 권장
    half: true           # FP16 추론 (VRAM 절약, RTX 시리즈 권장)
    conf: 0.25           # 낮을수록 더 많이 탐지 (오탐 증가)
    iou: 0.45            # NMS IoU threshold
    classes: null        # null=전체, [0,1,2]=특정 클래스만
    augment: false       # TTA (정확도↑, 속도↓)
    stream: true         # 메모리 최적화 스트리밍 모드
```

**모델 크기별 성능 (RTX 3090, imgsz=640):**

| model_size | COCO mAP | 추론 속도 | VRAM |
|---|---|---|---|
| n (nano) | 39.5 | ~120 FPS | ~0.6 GB |
| s (small) | 47.0 | ~80 FPS | ~0.8 GB |
| m (medium) | 51.5 | ~50 FPS | ~1.2 GB |
| l (large) | 53.4 | ~35 FPS | ~1.5 GB |
| **x (xlarge)** | **54.7** | **~25 FPS** | **~1.5 GB** |

**TensorRT 내보내기 (최대 속도):**
```python
from models.rgb.yolov11 import YOLOv11Detector
det = YOLOv11Detector(config["rgb_models"]["yolov11"])
det.load_model()
det.export_tensorrt("weights/yolo/yolo11x.engine", fp16=True)
```
내보내기 후 `weights` 경로를 `.engine` 파일로 변경하면 자동 로드됩니다.

---

### RT-DETR-r50vd

```yaml
rgb_models:
  rtdetrv2:
    enabled: true
    weights: "PekingU/rtdetr_r50vd"  # HuggingFace 모델 ID (자동 캐시)
    imgsz: 640
    half: true
    conf: 0.3
    iou: 0.5
```

- 첫 실행 시 HuggingFace에서 자동 다운로드 (`~/.cache/huggingface/`)
- NMS-free Transformer 구조 → 겹친 물체 탐지에 강점
- **모델**: `PekingU/rtdetr_r50vd` (RT-DETR v1, transformers 4.47.1 호환)

> **참고**: RT-DETRv2(`RTDetrV2ForObjectDetection`)는 `transformers>=4.49.0`부터 추가됐으나  
> `torch 2.1.2`와 호환되지 않아 RT-DETR v1을 사용합니다. mAP 차이는 약 1~2 포인트입니다.

---

### Grounding DINO (오픈 보캐뷸러리)

```yaml
rgb_models:
  grounding_dino:
    enabled: true
    config_file: "weights/grounding_dino/GroundingDINO_SwinT_OGC.py"
    weights: "weights/grounding_dino/groundingdino_swint_ogc.pth"
    text_prompt: "person . car . drone . uav ."  # ". "으로 클래스 구분
    box_threshold: 0.35
    text_threshold: 0.25
    half: false          # FP32 고정 (FP16 미지원)
```

**런타임 프롬프트 변경** (재로드 없이):
```python
detector.set_prompt("drone . uav . quadcopter . bird .")
```

> COCO 80 클래스에 없는 커스텀 객체(드론, 특수 장비 등)를 텍스트만으로 탐지 가능.  
> 속도는 상대적으로 느림 (약 8~12 FPS). 정확도 우선 또는 드론 탐지 실험에 활용.

---

### PointPillars (Depth 기반 3D 탐지)

```yaml
depth_model:
  pointpillars:
    enabled: true
    cfg_file: "third_party/OpenPCDet/tools/cfgs/custom_models/pointpillar_kinect.yaml"
    weights: "weights/pointpillars/pointpillar_7728.pth"
    point_cloud_range: [-5.0, -5.0, -1.0, 5.0, 5.0, 3.0]  # [xmin,ymin,zmin,xmax,ymax,zmax]
    voxel_size: [0.05, 0.05, 0.1]    # 복셀 크기 (m)
    max_points_per_voxel: 32
    max_voxels: 16000
    score_threshold: 0.3
    nms_threshold: 0.1
```

**Point Cloud 변환 설정:**
```yaml
pointcloud:
  min_range_m: 0.3      # 근거리 노이즈 제거
  max_range_m: 5.5      # NFOV_UNBINNED 최대 유효 범위
  remove_ground: true   # 지면 포인트 제거 (z < -0.3m)
  max_points: 200000    # 랜덤 다운샘플링 상한
```

---

## Sensor Fusion 설정

```yaml
fusion:
  strategy: "weighted_box_fusion"  # nms | soft_nms | weighted_box_fusion
  rgb_weight: 0.65                 # RGB 결과 가중치
  depth_weight: 0.35               # Depth 결과 가중치
  iou_threshold: 0.5               # WBF IoU 임계값
  depth_confirmation_only: false   # true: Depth는 거리 보완만
  depth_augment_rgb: true          # 탐지 결과에 depth_m 자동 보완
  min_depth_points: 10             # frustum 내 최소 유효 포인트 수
```

**Fusion 전략 비교:**

| 전략 | 특징 | 권장 상황 |
|---|---|---|
| `nms` | 겹친 박스 중 최고 점수만 유지 | 빠른 처리, 단순 환경 |
| `soft_nms` | 겹친 박스 점수를 Gaussian 감쇠 | 밀집 객체 환경 |
| `weighted_box_fusion` | 가중 평균으로 박스 위치 정밀화 | **기본 권장** |

---

## 추적 및 시각화 설정

```yaml
postprocess:
  tracker:
    enabled: true
    type: "bytetrack"
    track_thresh: 0.5      # 추적 시작 confidence
    track_buffer: 30       # 프레임 단위 객체 유지 (30프레임 = 1초)
    match_thresh: 0.8      # 매칭 IoU 임계값
    min_box_area: 100      # 최소 bbox 면적 (픽셀²). 드론: 16

  visualizer:
    enabled: true
    show_window: true
    show_depth_colormap: true   # 우측 상단 depth 히트맵 오버레이
    show_fps: true
    show_gpu_usage: true
    save_video: false           # true 시 data/results/ 에 영상 저장
    font_scale: 0.6
    line_thickness: 2

  logger:
    enabled: true
    log_format: "jsonl"         # jsonl | csv
    log_every_n_frames: 1
```

**시각화 색상 코딩:**
- 초록 박스: RGB 모델 탐지 결과
- 파랑 박스: Depth 모델 탐지 결과
- 빨강 박스: Fusion 통합 결과

---

## 성능 튜닝

### VRAM별 권장 설정

| VRAM | 대표 GPU | 권장 모드 | 활성 모델 |
|---|---|---|---|
| 4 GB | GTX 1650, RTX 3050 | RGB | YOLOv11-s 단독 |
| 6 GB | RTX 2060, RTX 3060 | RGB | YOLOv11-x + RT-DETR |
| 8 GB | RTX 3060 Ti, RTX 3070 | Fusion | YOLOv11-x + RT-DETR + PointPillars |
| 12 GB | RTX 3080 12G, RTX 4070 | Fusion | YOLOv11-x + RT-DETR + PointPillars (+ GDINO 주의) |
| 16 GB | RTX 4080, A4000 | Fusion | 전체 모델 |
| 24 GB | RTX 3090, RTX 4090 | Fusion | 전체 모델 + imgsz 1280 |

#### 4 GB — GTX 1650 / RTX 3050

```yaml
mode: "rgb"
rgb_models:
  active: ["yolov11"]
  yolov11:
    model_size: "s"   # s 또는 n
    imgsz: 512
    half: true
    conf: 0.3
depth_model:
  pointpillars:
    enabled: false
performance:
  rgb_inference_parallel: false
  skip_depth_every_n: 99
```

#### 6 GB — RTX 2060 / RTX 3060

```yaml
mode: "rgb"
rgb_models:
  active: ["yolov11", "rtdetrv2"]
  yolov11:
    model_size: "x"
    imgsz: 640
    half: true
  rtdetrv2:
    half: true
depth_model:
  pointpillars:
    enabled: false
performance:
  rgb_inference_parallel: true
```

#### 8 GB — RTX 3060 Ti / RTX 3070 / RTX 2080

```yaml
mode: "fusion"
rgb_models:
  active: ["yolov11", "rtdetrv2"]   # grounding_dino는 비활성 (3 GB 추가 소요)
  yolov11:
    model_size: "x"
    imgsz: 640
    half: true
  rtdetrv2:
    half: true
  grounding_dino:
    enabled: false
depth_model:
  pointpillars:
    enabled: true
performance:
  rgb_inference_parallel: true
  skip_depth_every_n: 2
```

#### 12 GB — RTX 3080 12G / RTX 4070

```yaml
mode: "fusion"
rgb_models:
  active: ["yolov11", "rtdetrv2"]
  yolov11:
    model_size: "x"
    imgsz: 640
    half: true
  rtdetrv2:
    half: true
  grounding_dino:
    enabled: false   # 활성화 시 VRAM 여유 확인 필요
depth_model:
  pointpillars:
    enabled: true
performance:
  rgb_inference_parallel: true
  skip_depth_every_n: 1
```

#### 16 GB — RTX 4080 / A4000

```yaml
mode: "fusion"
rgb_models:
  active: ["yolov11", "rtdetrv2", "grounding_dino"]
  yolov11:
    model_size: "x"
    imgsz: 640
    half: true
  rtdetrv2:
    half: true
  grounding_dino:
    enabled: true
    half: false   # FP32 고정
depth_model:
  pointpillars:
    enabled: true
performance:
  rgb_inference_parallel: true
  skip_depth_every_n: 1
```

#### 24 GB — RTX 3090 / RTX 4090

```yaml
mode: "fusion"
rgb_models:
  active: ["yolov11", "rtdetrv2", "grounding_dino"]
  yolov11:
    model_size: "x"
    imgsz: 1280
    half: true
    augment: true
  rtdetrv2:
    half: true
  grounding_dino:
    enabled: true
    half: false
depth_model:
  pointpillars:
    enabled: true
fusion:
  strategy: "weighted_box_fusion"
  rgb_weight: 0.6
  depth_weight: 0.4
performance:
  rgb_inference_parallel: true
  skip_depth_every_n: 1
```

> **Grounding DINO는 FP32 전용**으로 약 3 GB를 추가로 소비합니다.  
> 8 GB 이하에서는 비활성화를 권장하며, 12 GB 환경에서는 다른 모델 실행 후 여유 VRAM을 확인하고 활성화하세요.

---

### FPS 우선 (RTX 2060 / GTX 1080 환경)

```yaml
mode: "rgb"
rgb_models:
  active: ["yolov11"]
  yolov11:
    model_size: "s"      # nano(n) 또는 small(s)
    imgsz: 512
    half: true
depth_model:
  pointpillars:
    enabled: false       # Depth 비활성화
performance:
  rgb_inference_parallel: false
  skip_depth_every_n: 2
```

### 정확도 우선 (RTX 3090 최고 설정)

```yaml
mode: "fusion"
rgb_models:
  active: ["yolov11", "rtdetrv2", "grounding_dino"]
  yolov11:
    model_size: "x"
    imgsz: 1280
    augment: true
  grounding_dino:
    enabled: true
fusion:
  strategy: "weighted_box_fusion"
  rgb_weight: 0.6
  depth_weight: 0.4
```

### Depth 추론 부하 조절

```yaml
performance:
  skip_depth_every_n: 2   # Depth는 15fps, RGB는 30fps로 실행
  frame_queue_size: 3     # 프레임 버퍼 크기
```

---

## 드론 탐지 확장

### Phase 1: 즉시 적용 (config.yaml만 수정, 0분)

```yaml
drone_extension:
  enabled: true                               # false → true
  gdino_prompt: "drone . uav . quadcopter . multirotor ."
  imgsz: 1280
  min_box_area: 16     # 원거리 소형 드론은 픽셀 몇 개
  conf: 0.15           # 낮은 threshold (원거리 = 낮은 confidence)
```

Grounding DINO가 자동 활성화되어 텍스트 프롬프트 기반 드론 탐지를 즉시 수행합니다.

### Phase 2: YOLOv11 Fine-tuning (1~2주)

```bash
# 권장 드론 데이터셋
# - VisDrone: https://github.com/VisDrone/VisDrone-Dataset
# - anti-UAV: ICCV 2021 챌린지 데이터셋
# - DUT Anti-UAV: 4K 고해상도 드론 영상

yolo train model=yolo11x.pt data=drone_dataset.yaml epochs=100 imgsz=1280 batch=8 device=0
```

학습 완료 후 `config.yaml`에서 가중치 경로만 교체:
```yaml
drone_extension:
  yolov11_weights: "weights/yolo/yolo11x_drone_ft.pt"
```

### Phase 3: SAHI 소형 물체 추론 (2~4주)

`models/rgb/yolov11.py`의 `detect_with_sahi()` 메서드 활성화:
```python
# 슬라이딩 윈도우 분할 추론으로 원거리 소형 드론 감도 극대화
result = detector.detect_with_sahi(
    frame, slice_height=320, slice_width=320, overlap_ratio=0.2
)
```

### Phase 4: 3D 드론 추적 (4~8주)

`models/depth/` 에 `drone_3d_tracker.py` 추가:
- Kinect Depth에서 드론 고도(z) 실시간 추출
- Kalman Filter 기반 궤적 예측
- 광류(Optical Flow)로 속도 벡터 추정

---

## 프로젝트 구조

```
lig/
├── main.py                          # 실행 진입점 (argparse, pipeline 조립)
├── config.yaml                      # 전체 설정 (mode 한 줄로 전환)
├── environment.yml                  # kinect_det Conda 환경
├── setup.py                         # 패키지 설치 진입점
│
├── kinect/                          # Azure Kinect 센서 레이어
│   ├── __init__.py
│   ├── capture.py                   # KinectCapture: 프레임 취득, context manager
│   ├── calibration.py               # KinectCalibration: 캘리브레이션, 좌표 변환
│   └── pointcloud.py                # PointCloudConverter: Depth → Point Cloud
│
├── models/                          # 탐지 모델 레이어
│   ├── __init__.py
│   ├── base.py                      # Detection, DetectionResult dataclass + BaseDetector ABC
│   ├── rgb/
│   │   ├── __init__.py
│   │   ├── yolov11.py               # YOLOv11Detector (ultralytics)
│   │   ├── rtdetrv2.py              # RTDETRDetector (HuggingFace, PekingU/rtdetr_r50vd)
│   │   └── grounding_dino.py        # GroundingDINODetector (오픈 보캐뷸러리)
│   └── depth/
│       ├── __init__.py
│       └── pointpillars.py          # PointPillarsDetector (OpenPCDet 래퍼)
│
├── fusion/                          # Sensor Fusion 레이어
│   ├── __init__.py
│   ├── frustum_projector.py         # FrustumProjector: RGB bbox → 포인트 추출, 거리 통계
│   ├── fusion_detector.py           # FusionDetector: WBF Late Fusion 앙상블
│   └── nms.py                       # nms_2d / soft_nms_2d / weighted_box_fusion
│
├── pipeline/                        # 실행 파이프라인
│   ├── __init__.py
│   ├── rgb_pipeline.py              # RGBPipeline: RGB 단독 모드
│   ├── depth_pipeline.py            # DepthPipeline: Depth 단독 모드
│   └── fusion_pipeline.py           # FusionPipeline: Fusion 모드 (ThreadPoolExecutor 병렬)
│
├── postprocess/                     # 후처리 레이어
│   ├── __init__.py
│   ├── tracker.py                   # ByteTracker (supervision 기반, IoU 매칭)
│   ├── visualizer.py                # ResultVisualizer (bbox, depth colormap, FPS 오버레이)
│   └── logger.py                    # DetectionLogger (JSONL / CSV 저장)
│
├── utils/                           # 공통 유틸리티
│   ├── __init__.py
│   ├── config_loader.py             # ConfigLoader: YAML 로드 + drone_extension override
│   ├── fps_counter.py               # FPSCounter: deque 기반 rolling average
│   ├── gpu_monitor.py               # GPUMonitor: VRAM / 온도 / 사용률
│   └── coordinate.py                # kinect_to_lidar / xyxy_to_xywh 등 벡터화 변환
│
├── scripts/                         # 유틸리티 스크립트
│   ├── verify_install.py            # 10개 항목 설치 검증 (rich 테이블 출력)
│   ├── download_weights.py          # 모델 가중치 자동 다운로드
│   ├── benchmark.py                 # FPS / VRAM 벤치마크
│   └── calibration_check.py        # Kinect 캘리브레이션 확인 및 JSON 저장
│
├── third_party/                     # 외부 소스 (git submodule)
│   └── OpenPCDet/
│       └── tools/cfgs/custom_models/
│           └── pointpillar_kinect.yaml   # Azure Kinect 전용 PointPillars 설정
│
├── weights/                         # 모델 가중치 (git-ignored, .gitkeep만 추적)
│   ├── yolo/          ← yolo11x.pt
│   ├── rtdetrv2/      ← (HuggingFace 자동 캐시, PekingU/rtdetr_r50vd)
│   ├── grounding_dino/ ← groundingdino_swint_ogc.pth
│   └── pointpillars/  ← pointpillar_7728.pth
│
├── data/
│   ├── recordings/    ← .mkv 녹화 파일
│   └── results/       ← 탐지 결과 로그 및 영상
│
└── docs/
    └── superpowers/specs/
        └── 2026-05-21-azure-kinect-detection-design.md   # 설계 문서
```

---

## 모듈 설명

### `kinect/capture.py` — KinectCapture

Azure Kinect 장치로부터 동기화된 RGB+Depth 프레임을 취득합니다.

```python
from kinect import KinectCapture

with KinectCapture(config["kinect"]) as cap:
    frame = cap.get_frame()
    # frame.color  : (H, W, 3) uint8 BGR
    # frame.depth  : (H, W)    uint16 mm
    # frame.ir     : (H, W)    uint16
    # frame.timestamp_usec : int
    # frame.device_temp    : float (°C)
```

---

### `kinect/calibration.py` — KinectCalibration

내부 파라미터 추출 및 좌표계 변환을 담당합니다.

```python
from kinect import KinectCalibration

cal = KinectCalibration(device)
print(cal.color_intrinsics)   # fx, fy, cx, cy, dist_coeffs
print(cal.depth_intrinsics)

# Depth 픽셀 → 3D XYZ (m 단위, 벡터화)
pixels = np.array([[320, 240]])   # (N, 2)
depths = np.array([1500])         # (N,) mm
xyz = cal.depth_pixel_to_3d(pixels, depths)   # (N, 3)

# 캘리브레이션 저장/로드
cal.save("calibration.json")
cal2 = KinectCalibration.load("calibration.json")
```

---

### `kinect/pointcloud.py` — PointCloudConverter

Depth 이미지를 PointPillars 입력 형식의 포인트 클라우드로 변환합니다.

```python
from kinect import PointCloudConverter

converter = PointCloudConverter(calibration, config["pointcloud"])
points = converter.depth_to_pointcloud(frame.depth, frame.color)
# points: (N, 4) float32 [x, y, z, intensity]
# 좌표계: LiDAR 표준 (x-forward, y-left, z-up), 단위 m
```

**좌표계 변환 (Kinect → LiDAR):**
```
X_lidar =  Z_kinect   (앞쪽)
Y_lidar = -X_kinect   (왼쪽)
Z_lidar = -Y_kinect   (위쪽)
```

---

### `models/base.py` — Detection / DetectionResult

모든 모델이 공통으로 사용하는 탐지 결과 자료구조입니다.

```python
from models.base import Detection, DetectionResult

# Detection 필드
det = Detection(
    bbox_2d=np.array([x1, y1, x2, y2]),  # 픽셀 좌표
    bbox_3d=np.array([cx, cy, cz, l, w, h, yaw]),  # 미터 (Depth 모드)
    score=0.87,
    class_id=0,
    class_name="person",
    depth_m=2.34,       # Fusion 시 자동 보완
    source="fusion",    # "rgb" | "depth" | "fusion"
    track_id=5
)
```

---

### `models/rgb/yolov11.py` — YOLOv11Detector

```python
from models.rgb import YOLOv11Detector

det = YOLOv11Detector(config["rgb_models"]["yolov11"], device="cuda:0")
det.load_model()
det.warmup(n_iter=3)

result = det.detect(frame_bgr, conf_threshold=0.25)
for d in result.detections:
    print(d.class_name, d.score, d.bbox_2d)
```

---

### `models/rgb/grounding_dino.py` — GroundingDINODetector

```python
from models.rgb import GroundingDINODetector

det = GroundingDINODetector(config["rgb_models"]["grounding_dino"])
det.load_model()

# 런타임 프롬프트 변경
det.set_prompt("drone . uav . bird .")

result = det.detect(frame_bgr)
```

---

### `fusion/frustum_projector.py` — FrustumProjector

RGB 2D 바운딩 박스에 해당하는 포인트 클라우드를 추출하여 깊이 통계를 반환합니다.

```python
from fusion import FrustumProjector

projector = FrustumProjector(calibration)

# RGB bbox 영역의 포인트 추출
frustum_pts = projector.bbox2d_to_frustum_points(
    bbox_2d=np.array([100, 50, 400, 300]),
    points_3d=points,    # (N, 4)
    margin_px=5
)

stats = projector.get_depth_stats(frustum_pts)
# {'mean_depth': 2.1, 'median_depth': 2.0, 'min_depth': 1.8, 'point_count': 243}
```

---

### `pipeline/fusion_pipeline.py` — FusionPipeline

전체 파이프라인을 조립하고 실행합니다.

```python
from pipeline import FusionPipeline

pipe = FusionPipeline(config)
pipe.setup()       # 모델 로드, 카메라 오픈, 워밍업

# 단일 프레임
result = pipe.run_once()

# 스트리밍
for result in pipe.run_stream():
    print(f"FPS: {pipe.fps:.1f}, 탐지: {len(result.detections)}개")
    if should_stop:
        break

pipe.shutdown()
```

---

### `utils/coordinate.py` — 좌표 변환 유틸리티

```python
from utils.coordinate import kinect_to_lidar, xyxy_to_xywh, normalize_boxes

# Kinect → LiDAR 좌표계 변환
pts_lidar = kinect_to_lidar(pts_kinect)   # (N,3) → (N,3)

# bbox 형식 변환
xywh = xyxy_to_xywh(xyxy_boxes)          # [x1,y1,x2,y2] → [cx,cy,w,h]

# 픽셀 → 정규화 좌표
norm = normalize_boxes(boxes, image_shape=(1080, 1920))
```

---

## 트러블슈팅

### Azure Kinect 연결 안됨

```
ConnectionError: Azure Kinect 장치를 열 수 없습니다
```

1. USB **3.0** 포트 연결 확인 (2.0 불가)
2. DLL 경로를 시스템 PATH에 추가:
   ```
   C:\Program Files\Azure Kinect SDK v1.4.1\sdk\windows-desktop\amd64\release\bin
   ```
3. Azure Kinect Viewer 등 다른 프로그램이 장치를 점유하고 있지 않은지 확인
4. 전원 공급 확인 (USB 허브 사용 시 유전원 허브 필요)

---

### pyk4a 빌드 실패

```
error: Unable to find a compatible Visual Studio installation.
```

**원인**: pyk4a는 C 확장 빌드 시 MSVC 컴파일러를 필요로 합니다.  
**해결**: VS Build Tools 환경을 먼저 설정한 후 설치합니다.

```cmd
:: VS Build Tools의 vcvars64.bat 경로 확인 후 실행
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
set MSSdk=1
pip install git+https://github.com/etiennedub/pyk4a.git@1.5.0
```

> PyPI에는 `pyk4a==1.4.1`이 존재하지 않습니다. GitHub 소스(v1.5.0)에서 직접 빌드합니다.  
> Azure Kinect SDK v1.4.1이 `C:\Program Files\Azure Kinect SDK v1.4.1\`에 설치되어 있어야 합니다.

---

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

```yaml
# config.yaml에서 순서대로 시도
rgb_models:
  yolov11:
    half: true           # FP16 활성화
    imgsz: 512           # 640 → 512
  grounding_dino:
    enabled: false       # Grounding DINO 비활성화 (~3 GB 절약)
performance:
  rgb_inference_parallel: false   # 순차 추론으로 전환
```

---

### OpenPCDet 설치 오류

**`error: command 'cl.exe' failed`**

→ Windows CMD에서 vcvars64.bat을 먼저 실행해야 합니다. PowerShell에서는 `call`이 동작하지 않습니다.

```cmd
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
set MSSdk=1
python setup.py develop
```

**`ModuleNotFoundError: No module named 'pkg_resources'`**

→ PyTorch 2.1.2의 `cpp_extension.py`가 구버전 setuptools API를 사용합니다.

```cmd
pip install "setuptools==69.5.1"
python setup.py develop
```

**`error STL1002: Unexpected compiler version, expected CUDA 12.x or newer`**

→ CUDA 11.8은 MSVC **14.37(VS 2022 17.7) 이하**까지만 지원합니다.  
VS 2022라도 최신 업데이트(MSVC 14.38+)가 설치되면 동일 오류 발생합니다.

해결 순서:
1. VS Installer → 수정 → 개별 구성 요소 → `MSVC v143 - VS 2022 C++ x64/x86 빌드 도구 (v14.37-17.7)` 설치
2. 빌드 전 VCToolsVersion으로 14.37 고정 후 vcvars64.bat 호출

```cmd
set VCToolsVersion=14.37.32822
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

**`SharedArray 빌드 실패`**

→ `sys/mman.h` POSIX 헤더 의존으로 Windows에서는 빌드 불가입니다. 추론에 사용되지 않으므로 무시하고 진행합니다.

---

### FPS가 target_fps에 미달할 때

```yaml
performance:
  skip_depth_every_n: 2    # Depth 추론 2프레임마다 1회
  frame_queue_size: 5      # 버퍼 크기 증가
rgb_models:
  active: ["yolov11"]      # 단일 모델로 축소
  yolov11:
    imgsz: 512
```

---

## 라이선스 및 의존성

| 컴포넌트 | 라이선스 |
|---|---|
| YOLOv11 (ultralytics) | AGPL-3.0 |
| RT-DETR-r50vd (HuggingFace, PekingU/rtdetr_r50vd) | Apache-2.0 |
| Grounding DINO | Apache-2.0 |
| OpenPCDet | Apache-2.0 |
| pyk4a | MIT |
| Azure Kinect SDK | Microsoft 상업 라이선스 |
