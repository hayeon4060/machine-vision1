"""
거리 추정 비교 — 방식 A (bbox 크기 + 핀홀 모델) vs 방식 B (Depth Anything V2 Metric Indoor)
줄자로 잰 정답 거리 사진(distance_data/)으로 두 방식의 오차와 처리 시간을 비교.

파일명 규칙: front_4.05.jpg / rear_2.7.jpg / sign_0.9.jpg (sign3.6 처럼 언더스코어 없어도 인식)

사용:
  python distance_eval.py --weights final_best.pt --photos distance_data
  python distance_eval.py --weights final_best.pt --photos distance_data --no-depth   # 방식 A 만 (빠름)
  python distance_eval.py ... --imgsz 960                                          # 원거리 표지판 대응

출력: distance_results.csv (사진별), distance_summary.csv (클래스별 MAE/MAPE), distance_plot.png
"""
import argparse, re, time
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageOps
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('--weights', required=True)
ap.add_argument('--photos', default='distance_data')
ap.add_argument('--imgsz', type=int, default=640)
ap.add_argument('--conf', type=float, default=0.3)
ap.add_argument('--device', default=None)
ap.add_argument('--no-depth', action='store_true', help='Depth Anything 비교 생략')
ap.add_argument('--depth_model', default='depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf')
args = ap.parse_args()

# ---------- 실측 크기 (m) ----------
REAL = {  # 파일명 접두어 → (모델 클래스들, 실제 높이, 실제 너비)
    'front': (['front_door'], 2.5, 4.4),
    'rear':  (['rear_door'],  2.1, 1.8),
    'sign':  (['2_class', '4_class'], 0.20, 0.13),
}
PAT = re.compile(r'^(front|rear|sign)_?(\d+(?:\.\d+)?)', re.I)

# ---------- 사진 목록 ----------
items = []
for p in sorted(Path(args.photos).iterdir()):
    m = PAT.match(p.stem)
    if p.suffix.lower() in {'.jpg', '.jpeg', '.png'} and m:
        items.append((p, m.group(1).lower(), float(m.group(2))))
print(f'{len(items)}장 로드')

# ---------- 방식 A: 탐지 + bbox 크기 ----------
model = YOLO(args.weights)
rows = []
for p, kind, gt in items:
    img = ImageOps.exif_transpose(Image.open(p)).convert('RGB')   # EXIF 회전 반영 (라벨 기준과 동일)
    W, H = img.size
    classes, h_real, w_real = REAL[kind]
    t0 = time.perf_counter()
    r = model.predict(img, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
    t_det = (time.perf_counter() - t0) * 1000
    cand = [(float(c), b) for b, cl, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())
            if r.names[int(cl)] in classes]
    row = {'file': p.name, 'kind': kind, 'gt_m': gt, 'img_w': W, 'img_h': H, 'det_ms': round(t_det, 1)}
    if cand:
        conf, (x1, y1, x2, y2) = max(cand, key=lambda x: x[0])
        bh, bw = y2 - y1, x2 - x1
        clipped = y1 < 2 or y2 > H - 2            # 위아래가 잘리면 높이 대신 너비 사용
        row.update(conf=round(conf, 3), bbox_h_px=round(bh, 1), bbox_w_px=round(bw, 1), clipped=clipped,
                   x1=x1, y1=y1, x2=x2, y2=y2,
                   # 정규화 초점거리(해상도 무관): f/W = d * px / (real * W)
                   f_norm=(gt * (bw if clipped else bh)) / ((w_real if clipped else h_real) * W))
    else:
        row.update(conf=np.nan, bbox_h_px=np.nan, bbox_w_px=np.nan, clipped=False, f_norm=np.nan)
    rows.append(row)
df = pd.DataFrame(rows)

# 초점거리 보정: leave-one-out (자기 자신 제외한 나머지 사진들의 중앙값) → 과적합 없이 평가
valid = df['f_norm'].notna()
for i in df.index[valid]:
    others = df.loc[valid & (df.index != i), 'f_norm']
    f = others.median() if len(others) else df.loc[i, 'f_norm']
    _, h_real, w_real = REAL[df.loc[i, 'kind']]
    px = df.loc[i, 'bbox_w_px'] if df.loc[i, 'clipped'] else df.loc[i, 'bbox_h_px']
    real = w_real if df.loc[i, 'clipped'] else h_real
    df.loc[i, 'distA_m'] = f * real * df.loc[i, 'img_w'] / px
F_GLOBAL = df.loc[valid, 'f_norm'].median()
print(f'\n보정된 정규화 초점거리 f/W = {F_GLOBAL:.3f}  (사진별 편차 {df.loc[valid,"f_norm"].std():.3f})')
print('  → 실전 코드에서는  거리 = %.3f × 실제높이(m) × 이미지너비(px) / bbox높이(px)' % F_GLOBAL)

# ---------- 방식 B: Depth Anything V2 (metric, indoor) ----------
if not args.no_depth:
    import torch
    from transformers import pipeline
    dev = 0 if torch.cuda.is_available() else -1
    print(f'\nDepth Anything 로드: {args.depth_model} (device={"cuda" if dev == 0 else "cpu"})')
    depth_pipe = pipeline('depth-estimation', model=args.depth_model, device=dev)
    for i, (p, kind, gt) in enumerate(items):
        img = ImageOps.exif_transpose(Image.open(p)).convert('RGB')
        s = 640 / max(img.size); small = img.resize((round(img.width * s), round(img.height * s)))  # 속도 위해 축소
        t0 = time.perf_counter()
        out = depth_pipe(small)
        t_dep = (time.perf_counter() - t0) * 1000
        depth = out['predicted_depth']
        depth = depth.squeeze().cpu().numpy() if hasattr(depth, 'cpu') else np.array(depth)
        # 깊이맵을 원본 크기로 매핑해 bbox 영역 중앙값 추출
        dh, dw = depth.shape
        row = df.loc[i]
        if not np.isnan(row['bbox_h_px']):
            sx, sy = dw / row['img_w'], dh / row['img_h']
            x1, x2 = int(row['x1'] * sx), int(row['x2'] * sx); y1, y2 = int(row['y1'] * sy), int(row['y2'] * sy)
            # 가장자리 제외한 중앙 60% 영역 (문틀·배경 섞임 방지)
            mx, my = int((x2 - x1) * 0.2), int((y2 - y1) * 0.2)
            patch = depth[y1 + my:y2 - my, x1 + mx:x2 - mx]
            df.loc[i, 'distB_m'] = float(np.median(patch)) if patch.size else np.nan
        else:
            df.loc[i, 'distB_m'] = np.nan
        df.loc[i, 'depth_ms'] = round(t_dep, 1)

# ---------- 오차 집계 ----------
for k in ['A', 'B']:
    if f'dist{k}_m' in df:
        df[f'err{k}_m'] = (df[f'dist{k}_m'] - df['gt_m']).abs()
        df[f'err{k}_pct'] = df[f'err{k}_m'] / df['gt_m'] * 100
cols = ['file', 'kind', 'gt_m', 'conf', 'bbox_h_px', 'clipped', 'distA_m', 'errA_m', 'errA_pct'] + \
       (['distB_m', 'errB_m', 'errB_pct', 'depth_ms'] if 'distB_m' in df else []) + ['det_ms']
print('\n=== 사진별 결과 ===')
print(df[cols].round(2).to_string(index=False))

summ = df.groupby('kind').agg(n=('gt_m', 'size'), detected=('distA_m', lambda s: s.notna().sum()),
                              MAE_A=('errA_m', 'mean'), MAPE_A=('errA_pct', 'mean'),
                              **({'MAE_B': ('errB_m', 'mean'), 'MAPE_B': ('errB_pct', 'mean')} if 'distB_m' in df else {}))
summ.loc['ALL'] = summ.mean(numeric_only=True); summ.loc['ALL', 'n'] = len(df)
print('\n=== 클래스별 요약 (MAE: m, MAPE: %) ===')
print(summ.round(2).to_string())
print(f"\n처리 시간: 탐지 평균 {df['det_ms'].mean():.1f} ms" + (f", Depth 평균 {df['depth_ms'].mean():.1f} ms" if 'depth_ms' in df else ''))

df.drop(columns=['x1', 'y1', 'x2', 'y2'], errors='ignore').round(3).to_csv('distance_results.csv', index=False)
summ.round(3).to_csv('distance_summary.csv')

# ---------- 그래프 ----------
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    mk = {'front': 's', 'rear': 'o', 'sign': '^'}
    for kind, g in df.groupby('kind'):
        ax.scatter(g['gt_m'], g['distA_m'], marker=mk[kind], s=70, label=f'A bbox - {kind}')
        if 'distB_m' in g: ax.scatter(g['gt_m'], g['distB_m'], marker=mk[kind], s=70, facecolors='none', edgecolors='k', label=f'B depth - {kind}')
    lim = [0, df['gt_m'].max() + 1]; ax.plot(lim, lim, 'k--', lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('measured distance (m)'); ax.set_ylabel('estimated distance (m)'); ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig('distance_plot.png', dpi=150); print('저장: distance_results.csv, distance_summary.csv, distance_plot.png')
except Exception as e:
    print('plot skip:', e)
