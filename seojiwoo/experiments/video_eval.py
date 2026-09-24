"""
video_eval.py — 영상 기반 위치 판정 평가 (locator.py 와 동일 로직)
  3) 처리 속도  프레임당 추론 ms / FPS (실행 머신 기준)
  4) 커버리지   안내(near)가 나간 프레임 / 전체 프레임, 그리고 랜드마크 구간 기준
  5) 응답 정확도 안내가 나간 프레임 중 정답과 일치한 비율
  2) 전환 지연  정답 구간이 바뀐 시점 → 시스템이 새 정답을 처음 안내하기까지(초)
  +) 오경보율   정답이 none 인 구간에서 안내가 나간 비율

  python video_eval.py --weights final_best.pt --video walk.mp4 --gt gt_segments.csv
  python video_eval.py ... --windows 1 3 5 --near_m 2.5 3.0 --device cpu

gt_segments.csv (초 단위): start,end,label   label ∈ {2_class, 4_class, front_door, rear_door, none}
  → "label" 은 그 구간에서 시스템이 '앞쪽에 ○○이 있습니다' 라고 안내해야 하는 랜드마크 (2.5m 이내로 접근한 구간)
"""
import argparse, time, platform
from pathlib import Path
import cv2, numpy as np, pandas as pd
from locator import Locator

ap = argparse.ArgumentParser()
ap.add_argument('--weights', required=True)
ap.add_argument('--video', required=True)
ap.add_argument('--gt', required=True)
ap.add_argument('--stride', type=int, default=6, help='N프레임마다 1회 추론 (30fps 영상, 6 → 5회/초)')
ap.add_argument('--windows', type=int, nargs='+', default=[1, 3, 5])
ap.add_argument('--near_m', type=float, nargs='+', default=[2.5])
ap.add_argument('--f_norm', type=float, default=0.85)
ap.add_argument('--conf', type=float, default=0.5)
ap.add_argument('--imgsz', type=int, default=640)
ap.add_argument('--device', default=None)
ap.add_argument('--out', default='video_eval_results.csv')
args = ap.parse_args()

gt = pd.read_csv(args.gt)
def gt_at(t):
    h = gt[(gt['start'] <= t) & (t < gt['end'])]
    return h['label'].iloc[0] if len(h) else 'none'

# ---------- 1패스: stride 간격으로 탐지만 수행하고 캐시 ----------
loc = Locator(args.weights, f_norm=args.f_norm, near_m=args.near_m[0], conf=args.conf, imgsz=args.imgsz, device=args.device)
cap = cv2.VideoCapture(args.video); fps = cap.get(cv2.CAP_PROP_FPS) or 30
W, H = int(cap.get(3)), int(cap.get(4))
ok, f0 = cap.read(); loc.detect(f0); cap.set(cv2.CAP_PROP_POS_FRAMES, 0)      # 워밍업
cached, times, i = [], [], 0
while True:
    ok, frame = cap.read()
    if not ok: break
    if i % args.stride == 0:
        t0 = time.perf_counter(); dets = loc.detect(frame); ms = (time.perf_counter() - t0) * 1000
        cached.append((i / fps, dets)); times.append(ms)
    i += 1
cap.release()
times = np.array(times)
print(f'영상 {i}프레임 @ {fps:.0f}fps, {W}x{H} → {len(cached)}프레임 추론 (stride {args.stride})')
print(f'[속도] 프레임당 {np.median(times):.1f} ms (p95 {np.percentile(times,95):.1f}) → 최대 {1000/np.median(times):.0f} FPS 가능, '
      f'시스템은 {fps/args.stride:.0f} FPS로 처리  | device={args.device or "auto"} {platform.processor() or platform.machine()}')

# ---------- 2패스: 윈도우·임계값별로 판정 재생 + 지표 ----------
def replay(window, near_m):
    l = Locator.__new__(Locator)             # 모델 재로딩 없이 판정 상태만 새로
    l.__dict__.update(loc.__dict__); l.window, l.near_m = window, near_m
    from collections import deque
    l.buf = deque(maxlen=window); l.last_text, l.last_announce_t = None, -1e9
    preds = []
    for t, dets in cached:
        o = l.update(dets, t)
        preds.append(o['landmark'] if o['state'] == 'near' else 'none')
    return preds

def metrics(preds):
    ts = [t for t, _ in cached]; truth = [gt_at(t) for t in ts]
    ans = [p != 'none' for p in preds]; lm = [g != 'none' for g in truth]
    m = dict(
        coverage_all=sum(ans) / len(preds),
        coverage_on_landmark=sum(a for a, l in zip(ans, lm) if l) / max(sum(lm), 1),
        acc_when_answered=sum(p == g for p, g, a in zip(preds, truth, ans) if a) / max(sum(ans), 1),
        false_alarm_rate=sum(a for a, l in zip(ans, lm) if not l) / max(len(lm) - sum(lm), 1))
    lat = []
    for k in range(1, len(truth)):
        if truth[k] != truth[k-1] and truth[k] != 'none':
            found = next((ts[j] for j in range(k, len(preds)) if preds[j] == truth[k]), None)
            lat.append(found - ts[k] if found is not None else np.nan)
    lat = np.array(lat, float)
    m.update(transitions=len(lat), missed=int(np.isnan(lat).sum()),
             latency_mean_s=float(np.nanmean(lat)) if len(lat) and not np.all(np.isnan(lat)) else np.nan,
             latency_max_s=float(np.nanmax(lat)) if len(lat) and not np.all(np.isnan(lat)) else np.nan)
    return m

rows = []
for nm in args.near_m:
    for w in args.windows:
        m = metrics(replay(w, nm))
        rows.append({'weights': Path(args.weights).name, 'device': args.device or 'auto', 'stride': args.stride,
                     'proc_fps': round(fps / args.stride, 1), 'infer_ms': round(float(np.median(times)), 1),
                     'near_m': nm, 'window': w, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}})
res = pd.DataFrame(rows)
print('\n[윈도우 · 임계값별 지표]'); print(res.drop(columns=['weights', 'device']).to_string(index=False))
old = pd.read_csv(args.out) if Path(args.out).exists() else pd.DataFrame()
pd.concat([old, res], ignore_index=True).to_csv(args.out, index=False)
print('\n저장:', args.out)
