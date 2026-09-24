"""
모델별 프레임 지연시간 벤치마크 — 실시간성 근거용
같은 영상의 앞 N프레임을 각 모델로 추론, 워밍업 제외 후 프레임당 ms(중앙값/95퍼센타일)와 FPS를 GPU·CPU 각각 측정.

  python bench_latency.py --video test.mp4
  python bench_latency.py --video test.mp4 --frames 150 --devices 0 cpu
출력: latency_bench.csv, 콘솔 표
"""
import argparse, time, platform
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('--video', required=True)
ap.add_argument('--frames', type=int, default=120)
ap.add_argument('--imgsz', type=int, default=640)
ap.add_argument('--devices', nargs='+', default=['0', 'cpu'])
ap.add_argument('--project', default='runs_local')
ap.add_argument('--extra', nargs='*', default=[], help='추가 가중치 경로 (예: teammate best.pt)')
args = ap.parse_args()

ROOT = Path(__file__).resolve().parent
runs = {p.parent.parent.name: p for p in sorted((ROOT / args.project).glob('*_base/weights/best.pt'))}
for e in args.extra: runs[Path(e).stem] = Path(e)

# 영상 앞 N프레임을 메모리에 (디코딩 시간 제외하고 순수 추론만 측정)
cap = cv2.VideoCapture(args.video); frames = []
while len(frames) < args.frames:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
cap.release()
print(f'{len(frames)} 프레임 로드, imgsz={args.imgsz}')

rows = []
for dev in args.devices:
    if dev != 'cpu' and not torch.cuda.is_available(): print('CUDA 없음, GPU 생략'); continue
    dev_name = torch.cuda.get_device_name(0) if dev != 'cpu' else platform.processor()
    for run, w in runs.items():
        m = YOLO(str(w)); n_params = sum(p.numel() for p in m.model.parameters()) / 1e6
        for f in frames[:10]: m.predict(f, imgsz=args.imgsz, device=dev, verbose=False)   # 워밍업
        t_inf, t_tot = [], []
        for f in frames:
            t0 = time.perf_counter()
            r = m.predict(f, imgsz=args.imgsz, device=dev, verbose=False)[0]
            if dev != 'cpu': torch.cuda.synchronize()
            t_tot.append((time.perf_counter() - t0) * 1000)
            t_inf.append(r.speed['inference'])
        t_tot = np.array(t_tot)
        rows.append({'device': 'GPU' if dev != 'cpu' else 'CPU', 'device_name': dev_name, 'run': run,
                     'params_M': round(n_params, 2), 'infer_ms_med': round(float(np.median(t_inf)), 2),
                     'total_ms_med': round(float(np.median(t_tot)), 2), 'total_ms_p95': round(float(np.percentile(t_tot, 95)), 2),
                     'fps': round(1000 / float(np.median(t_tot)), 1)})
        print(rows[-1])

df = pd.DataFrame(rows)
df.to_csv(ROOT / 'latency_bench.csv', index=False)
print('\n=== 프레임당 지연 (전처리+추론+후처리, 중앙값) ===')
print(df.pivot(index='run', columns='device', values='total_ms_med').to_string())
print('\n=== FPS ===')
print(df.pivot(index='run', columns='device', values='fps').to_string())
print('\n저장:', ROOT / 'latency_bench.csv')
