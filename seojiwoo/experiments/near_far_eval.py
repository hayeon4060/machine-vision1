"""
near_far_eval.py — 근접 판정 정확도 (사진 단위, locator.py 동일 로직)
  data_correct/ : 가까이서 찍은 사진 → "앞쪽에 ○○이 있습니다" 가 나와야 정답
  data_wrong/   : 멀리서 찍은 사진   → 안내가 나오면 오경보 (화면 표시는 OK)
  파일명 앞 접두어로 클래스: 2_… 4_… front_… rear_…  (뒤의 숫자·괄호는 무시)

  python near_far_eval.py --weights final_best.pt --photos near_far
  python near_far_eval.py --weights final_best.pt --photos near_far --near_m 2.5 3.0 3.5
출력: near_far_results.csv (사진별), near_far_summary.csv, 콘솔 요약
"""
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageOps
from locator import Locator, DISPLAY

ap = argparse.ArgumentParser()
ap.add_argument('--weights', required=True)
ap.add_argument('--photos', default='near_far')
ap.add_argument('--near_m', type=float, nargs='+', default=[3.0])
ap.add_argument('--f_norm', type=float, default=0.85)
ap.add_argument('--conf', type=float, default=0.5)
ap.add_argument('--device', default=None)
args = ap.parse_args()

PAT = re.compile(r'^(2_class|4_class|front_door|rear_door|2|4|front|rear)(?=[_ (.])', re.I)
PREFIX = {'2': '2_class', '4': '4_class', 'front': 'front_door', 'rear': 'rear_door'}
FOLDERS = {'near': ['near', 'data_correct'], 'far': ['far', 'data_wrong']}
items = []
for split, names in FOLDERS.items():
    for d in (Path(args.photos) / n for n in names):
        if not d.exists(): continue
        for p in sorted(d.rglob('*')):
            m = PAT.match(p.name)
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.jfif', '.webp', '.bmp'} and m:
                g = m.group(1).lower(); items.append((p, split, PREFIX.get(g, g)))
print(f'{len(items)}장 로드 (near {sum(s=="near" for _,s,_ in items)}, far {sum(s=="far" for _,s,_ in items)})  [v2: data_correct/data_wrong 지원]')
if not items: raise SystemExit('사진을 못 찾음: near_far/data_correct, near_far/data_wrong 폴더와 파일명 접두어(2_, 4_, front_, rear_)를 확인하세요')

loc = Locator(args.weights, f_norm=args.f_norm, window=1, conf=args.conf, device=args.device)
rows = []
for p, split, gt_cls in items:
    img = np.array(ImageOps.exif_transpose(Image.open(p)).convert('RGB'))[:, :, ::-1].copy()   # EXIF 회전 반영, BGR
    dets = loc.detect(img)
    top = min(dets, key=lambda d: d['distance_m']) if dets else None
    rows.append({'file': p.name, 'split': split, 'gt_class': gt_cls,
                 'pred_class': top['name'] if top else None, 'distance_m': round(top['distance_m'], 2) if top else np.nan,
                 'conf': round(top['conf'], 3) if top else np.nan})
df = pd.DataFrame(rows)

print('\n=== 사진별 ==='); print(df.to_string(index=False))
summary = []
for nm in args.near_m:
    d = df.copy()
    d['announced'] = d['distance_m'].notna() & (d['distance_m'] <= nm)
    d['announced_correct'] = d['announced'] & (d['pred_class'] == d['gt_class'])
    near, far = d[d.split == 'near'], d[d.split == 'far']
    s = {'near_m': nm,
         'near_n': len(near), 'near_안내율': near['announced'].mean() if len(near) else np.nan,           # 커버리지
         'near_정답안내율': near['announced_correct'].mean() if len(near) else np.nan,                   # 안내가 맞은 비율(전체 기준)
         'near_응답정확도': near['announced_correct'].sum() / max(near['announced'].sum(), 1),          # 안내한 것 중 맞은 비율
         'near_탐지율': near['pred_class'].notna().mean() if len(near) else np.nan,
         'far_n': len(far), 'far_오경보율': far['announced'].mean() if len(far) else np.nan,              # 밖인데 안내함
         'far_탐지율': far['pred_class'].notna().mean() if len(far) else np.nan}
    summary.append(s)
    # 클래스별 (near)
    if nm == args.near_m[0]:
        print(f'\n=== 클래스별 (near, {nm}m) ===')
        print(near.groupby('gt_class').agg(n=('file', 'size'), 탐지=('pred_class', lambda s: s.notna().sum()),
                                           안내=('announced', 'sum'), 정답안내=('announced_correct', 'sum'),
                                           거리중앙값=('distance_m', 'median')).to_string())
sm = pd.DataFrame(summary)
print('\n=== 임계값별 요약 ===')
print(sm.round(3).to_string(index=False))
df.to_csv('near_far_results.csv', index=False); sm.round(4).to_csv('near_far_summary.csv', index=False)
print('\n저장: near_far_results.csv, near_far_summary.csv')
