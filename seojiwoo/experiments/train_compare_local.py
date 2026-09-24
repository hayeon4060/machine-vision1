"""
Landmark Vision — 로컬(MSI RTX 4050) 모델 비교 파이프라인
Colab 노트북과 동일한 실험을 스크립트 하나로 실행. landmark_vision 폴더 안에 두고 실행.

  python train_compare_local.py --stage 1        # YOLOv8n/v8s/11n/11s 동일 조건 비교
  python train_compare_local.py --stage 2        # 최종 후보(기본 yolov8n) 보완 실험: img960 / aug / 둘 다
  python train_compare_local.py --stage f1       # 클래스별 F1 + 혼동행렬 png
  python train_compare_local.py --stage export --final yolov8n_base   # best.pt/ONNX/예측 샘플
  python train_compare_local.py --stage table    # 비교표만 출력

결과: runs_local/comparison.csv, f1_per_class.csv, cm_*.png, comparison.png
이미 학습된 run 은 자동으로 건너뜀(평가만).
"""
import argparse, os, time, shutil, yaml
from pathlib import Path
from collections import Counter
import pandas as pd
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('--stage', required=True, choices=['1', '2', 'f1', 'export', 'table'])
ap.add_argument('--data', default='dataset_small', help='train/valid/test 와 data.yaml 이 있는 폴더')
ap.add_argument('--project', default='runs_local')
ap.add_argument('--epochs', type=int, default=80)
ap.add_argument('--imgsz', type=int, default=640)
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--best', default='yolov8n.pt', help='stage 2 에서 보완 실험할 모델')
ap.add_argument('--final', default='yolov8n_base', help='stage export 에서 내보낼 run 이름')
ap.add_argument('--device', default='0')
args = ap.parse_args()

ROOT = Path(__file__).resolve().parent
DATASET_DIR = (ROOT / args.data).resolve()
PROJECT = (ROOT / args.project).resolve(); PROJECT.mkdir(parents=True, exist_ok=True)
DATA_YAML = PROJECT / 'data.yaml'
RESULTS_CSV = PROJECT / 'comparison.csv'
MODELS = ['yolov8n.pt', 'yolov8s.pt', 'yolo11n.pt', 'yolo11s.pt', 'yolo26n.pt']
SEED, PATIENCE = 0, 30
MAIN_COLS = ['run', 'note', 'params_M', 'size_MB', 'train_min',
             'val_P', 'val_R', 'val_mAP50', 'val_mAP50-95', 'val_infer_ms',
             'test_P', 'test_R', 'test_mAP50', 'test_mAP50-95']

# ---------- data.yaml 절대경로화 ----------
src = yaml.safe_load(open(DATASET_DIR / 'data.yaml', encoding='utf-8'))
cfg = {'path': str(DATASET_DIR), 'train': 'train/images', 'val': 'valid/images', 'test': 'test/images',
       'nc': src['nc'], 'names': src['names']}
yaml.safe_dump(cfg, open(DATA_YAML, 'w', encoding='utf-8'), allow_unicode=True)
NAMES = cfg['names']

def class_counts():
    for split in ['train', 'valid', 'test']:
        c = Counter()
        for f in (DATASET_DIR / split / 'labels').glob('*.txt'):
            for line in open(f):
                if line.strip(): c[int(line.split()[0])] += 1
        print(split, {NAMES[k]: v for k, v in sorted(c.items())})

# ---------- 공통 ----------
def evaluate(weights, run_name, train_time_s, note=''):
    m = YOLO(str(weights))
    row = {'run': run_name, 'note': note,
           'params_M': round(sum(p.numel() for p in m.model.parameters()) / 1e6, 2),
           'size_MB': round(os.path.getsize(weights) / 1e6, 1), 'train_min': round(train_time_s / 60, 1)}
    for split in ['val', 'test']:
        r = m.val(data=str(DATA_YAML), split=split, imgsz=args.imgsz, batch=16, device=args.device,
                  plots=False, verbose=False, project=str(PROJECT / '_val_tmp'), name=f'{run_name}_{split}', exist_ok=True)
        row[f'{split}_P'] = round(float(r.box.mp), 3); row[f'{split}_R'] = round(float(r.box.mr), 3)
        row[f'{split}_mAP50'] = round(float(r.box.map50), 3); row[f'{split}_mAP50-95'] = round(float(r.box.map), 3)
        row[f'{split}_infer_ms'] = round(float(r.speed['inference']), 2)
        for ci, ap50 in zip(r.box.ap_class_index, r.box.ap50):
            row[f'{split}_AP50_{r.names[int(ci)]}'] = round(float(ap50), 3)
    return row

def save_row(row):
    df = pd.read_csv(RESULTS_CSV) if RESULTS_CSV.exists() else pd.DataFrame()
    if len(df) and row['run'] in df['run'].values and row['train_min'] == 0:
        row['train_min'] = float(df.loc[df['run'] == row['run'], 'train_min'].iloc[0])
    if len(df): df = df[df['run'] != row['run']]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(pd.DataFrame([row]).T.to_string())

def train_and_eval(base_weights, run_name, note='', **overrides):
    best = PROJECT / run_name / 'weights' / 'best.pt'
    if best.exists():
        print(f'[skip] {run_name} 이미 학습됨 → 평가만'); t = 0
    else:
        t0 = time.time()
        train_kwargs = dict(data=str(DATA_YAML), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                            seed=SEED, deterministic=True, patience=PATIENCE, workers=4, cache='ram',
                            device=args.device, project=str(PROJECT), name=run_name, exist_ok=True, plots=True)
        train_kwargs.update(overrides)
        YOLO(base_weights).train(**train_kwargs)
        t = time.time() - t0
    save_row(evaluate(best, run_name, t, note))

def show_table():
    df = pd.read_csv(RESULTS_CSV)
    print('\n=== 비교표 (valid mAP50 순) ===')
    print(df[[c for c in MAIN_COLS if c in df.columns]].sort_values('val_mAP50', ascending=False).to_string(index=False))
    print('\n=== 클래스별 AP50 (valid) ===')
    print(df[['run'] + [c for c in df.columns if c.startswith('val_AP50_')]].to_string(index=False))
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        d = df[df['note'] == 'baseline'].sort_values('params_M')
        if len(d):
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].bar(d['run'], d['val_mAP50'], label='valid mAP50'); ax[0].bar(d['run'], d['val_mAP50-95'], alpha=.6, label='valid mAP50-95')
            ax[0].set_ylim(0, 1); ax[0].legend(); ax[0].set_title('accuracy'); ax[0].tick_params(axis='x', rotation=20)
            ax[1].scatter(d['val_infer_ms'], d['val_mAP50'], s=d['params_M'] * 40)
            for _, r in d.iterrows(): ax[1].annotate(r['run'], (r['val_infer_ms'], r['val_mAP50']))
            ax[1].set_xlabel('inference ms/img'); ax[1].set_ylabel('valid mAP50'); ax[1].set_title('speed vs accuracy (size=params)')
            plt.tight_layout(); plt.savefig(PROJECT / 'comparison.png', dpi=150)
            print('\n저장:', PROJECT / 'comparison.png')
    except Exception as e:
        print('plot skip:', e)

# ---------- 단계 ----------
if __name__ == '__main__':
    if args.stage == '1':
        class_counts()
        for w in MODELS:
            train_and_eval(w, w.replace('.pt', '') + '_base', note='baseline')
        show_table()

    elif args.stage == '2':
        b = args.best.replace('.pt', '')
        AUG = dict(hsv_h=0.02, hsv_s=0.8, hsv_v=0.6, degrees=5, translate=0.2, scale=0.7, mosaic=1.0, mixup=0.1, close_mosaic=15)
        big = dict(imgsz=960, batch=8)                      # 6GB VRAM 고려
        train_and_eval(args.best, f'{b}_img960', note='imgsz 960', **big)
        train_and_eval(args.best, f'{b}_aug', note='aug 강화', **AUG)
        train_and_eval(args.best, f'{b}_img960_aug', note='960+aug', **big, **AUG)
        show_table()

    elif args.stage == 'f1':
        df = pd.read_csv(RESULTS_CSV); rows = []
        for run in df['run']:
            w = PROJECT / run / 'weights' / 'best.pt'
            if not w.exists(): continue
            r = YOLO(str(w)).val(data=str(DATA_YAML), split='val', imgsz=args.imgsz, device=args.device, plots=True,
                                 verbose=False, project=str(PROJECT / '_val_tmp'), name=f'{run}_cm', exist_ok=True)
            for ci, p, rc in zip(r.box.ap_class_index, r.box.p, r.box.r):
                rows.append({'run': run, 'class': r.names[int(ci)], 'P': round(float(p), 3), 'R': round(float(rc), 3),
                             'F1': round(float(2 * p * rc / (p + rc + 1e-9)), 3)})
            shutil.copy(PROJECT / '_val_tmp' / f'{run}_cm' / 'confusion_matrix.png', PROJECT / f'cm_{run}.png')
        f1 = pd.DataFrame(rows).pivot(index='class', columns='run', values='F1')
        print(f1.to_string()); f1.to_csv(PROJECT / 'f1_per_class.csv')
        print('혼동행렬:', sorted(p.name for p in PROJECT.glob('cm_*.png')))

    elif args.stage == 'export':
        w = PROJECT / args.final / 'weights' / 'best.pt'
        m = YOLO(str(w))
        m.export(format='onnx', imgsz=args.imgsz, simplify=True, opset=12)
        shutil.copy(w, ROOT / 'final_best.pt'); shutil.copy(w.with_suffix('.onnx'), ROOT / 'final_best.onnx')
        m.predict(source=str(DATASET_DIR / 'test' / 'images'), imgsz=args.imgsz, conf=0.4, device=args.device,
                  save=True, project=str(ROOT), name='pred_samples', exist_ok=True)
        print('저장:', ROOT / 'final_best.pt', ROOT / 'final_best.onnx', ROOT / 'pred_samples')

    elif args.stage == 'table':
        show_table()
