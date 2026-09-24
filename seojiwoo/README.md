# seojiwoo/ — 랜드마크 탐지 · 위치 판정 (YOLOv8n)

시각장애인 실내 이동 보조 시스템의 탐지·판정 파트. 폰 카메라 → 노트북 → 탐지 → 거리 → 다수결 → 안내.

## 5분 안에 돌리기

```powershell
cd seojiwoo\core
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # GPU 없으면 이 줄 생략
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

녹화 영상으로 확인:
```powershell
.\.venv\Scripts\python.exe demo_live.py --weights v1_best.pt --logo_weights final_best.pt --source ..\..\sample\test.mp4 --device 0 --nav --ocr --tts --save
```
폰 카메라(DroidCam 가상 웹캠)로:
```powershell
.\.venv\Scripts\python.exe demo_live.py --weights v1_best.pt --logo_weights final_best.pt --source 1 --device 0 --nav --ocr --tts --save
```
`--weights` 는 표지판·문 탐지용 5클래스(v1_best.pt), `--logo_weights` 는 로고 전용 6클래스(final_best.pt) — 클래스별 탐지기 분리 이유는 아래 "자주 막히는 곳" 참고.
`--device 0` 은 GPU, 없으면 `--device cpu` (v8n 은 CPU 에서도 20 FPS).

## 자주 막히는 곳 (오늘 실제로 겪은 것)

| 증상 | 원인 | 해결 |
|---|---|---|
| `Activate.ps1 ... 실행할 수 없으므로` | PowerShell 실행 정책 | 활성화 대신 `.\.venv\Scripts\python.exe 스크립트.py` 로 직접 실행 |
| `cv2.imshow ... not implemented` | easyocr 설치 시 `opencv-python-headless` 가 덮어씀 | `pip uninstall opencv-python-headless opencv-python -y && pip install "opencv-python<5"` |
| 창은 뜨는데 초록/검정 화면 | DroidCam 폰 연결 끊김 | 폰 앱 화면 켜둔 채 클라이언트 "retry", 미리보기에 영상 뜬 뒤 실행 |
| 폰 영상 대신 노트북 카메라 | 소스 번호 | `--source 0` ↔ `--source 1` 바꿔보기 |
| `RuntimeError: freeze_support()` | Windows 멀티프로세스 | 스크립트 본문이 `if __name__ == "__main__":` 안에 있어야 함 (core 는 이미 적용) |
| 거리가 전부 20~30% 짧거나 김 | 폰마다 화각이 다름 | 아래 "카메라 보정" 1회 |
| 로고 클래스 추가 재학습 후 표지판 원거리 탐지 4/7→1/7로 급락 | logo 클래스를 합쳐 재학습하면서 표지판(2_class/4_class) 민감도가 떨어짐 | 클래스별 탐지기 분리로 해결: 표지판·문은 5클래스 v1_best.pt, 로고만 6클래스 final_best.pt (`--weights`/`--logo_weights`) |

## 카메라 보정 (폰 바꾸면 1회)

`f_norm` 은 카메라 화각 상수. 4강의실 표지판 정면 **2.0 m** 에 서서 화면에 찍힌 거리 `d` 를 읽고
`f_norm_new = 0.85 × 2.0 / d` 를 `--f_norm` 으로 넘기면 됨. 문은 박스에 문틀이 들어와 실제보다 크게 잡히므로
`locator.py` 의 `REAL_SIZE` 높이를 "박스가 잡는 높이" 기준으로 조정.

## 판정 로직 요약 (locator.py)

- 탐지: YOLOv8n, 640, conf 0.5 (표지판은 0.65)
- 거리: `d = f_norm × 실제높이 × 이미지너비 / bbox높이` (핀홀), 위아래 잘리면 너비 기준
- 투표: 최근 5프레임(초당 5회 추론 = 1초) 중 같은 랜드마크 4표 이상
- 안내: 거리 ≤ 3.0 m 이고 접근 중일 때 "앞쪽에 ○○이 있습니다". 같은 문장은 5초에 한 번
- 표지판 숫자: `--ocr` 켜면 근접 시 crop 을 EasyOCR 로 읽어 숫자 확인. 회의실/다른 숫자면 안내 취소
- 시나리오(`--nav`): 앞문 → 로고 찾기(좌/우/직진) → 벽(≤1.2 m) 좌회전 → 4강의실 → 3강의실(YOLO 클래스 없이 OCR만으로 통과 확인) → 2강의실 → 좌회전 → 뒷문 도착
- 표지판 단계 전환: 4강의실/3강의실에서 도달 안내 후에도, 해당 표지판이 화면에서 완전히 사라져야(연속 3프레임 미탐지) 다음 표지판 단계로 넘어감 — 같은 표지판을 다음 단계로 착각해 재판독하는 것을 방지

클래스: `2_class, 4_class, front_door, rear_door, water_dispenser(미사용), logo`

## 다른 모듈에서 쓰기

```python
from locator import Locator
loc = Locator('final_best.pt', device='0')
out = loc.process(frame_bgr, t_sec)
# out: {landmark, landmark_kr, distance_m, state('none'|'far'|'near'), confidence, text, announce, detections, ocr}
```
FastAPI 에서는 프레임마다 `process()` 호출 후 `out` 을 JSON 으로 내려주면 됨. OCR 검증기를 바꾸려면
`OcrVerifier` 와 같은 `verify(frame, box, expected_class) -> ('confirm'|'reject'|'unknown', text)` 인터페이스만 맞추면 됨.

## 결과 요약 (experiments/ 참고)

- 모델 비교(동일 조건, valid mAP50): v8n 0.968 · v8s 0.962 · 11n 0.959 · 11s 0.960 · 26n 0.971 → 정확도 포화
- 외부 테스트셋 100장(이미지 단위, conf 0.5): v8n 커버리지 80% · 응답 정확도 87.5% · 표지판 커버리지 73.5%
- 지연(중앙값): GPU 8.9 ms(112 FPS) / CPU 50 ms(20 FPS). 시스템은 초당 5프레임만 처리
- 거리 추정(핀홀): 평균 오차 6.9%, 표지판 탐지 한계 ≈ 3.6 m. Depth Anything 은 2배 과대추정 + 0.5 s/frame 으로 제외
- 증강 강화 모델은 내부 valid 0.995 → 외부셋 표지판 커버리지 42.9%(base 73.5%) 로 탈락. valid 14% 가 train 과 근접 중복

## 데이터 · 가중치

리포에는 `final_best.pt/onnx` 만 포함. 데이터셋·학습 run 전체는 Drive: `<링크>`
