"""
외부 테스트셋(팀원 제작, 100장 / 4클래스) 평가 — runs_local 의 모든 학습 결과를 같은 셋에 평가
  python eval_external.py                       # test_ext/ 폴더, runs_local/
  python eval_external.py --ext test_ext --project runs_local --imgsz 640
출력: runs_local/ext_comparison.csv  (기존 comparison.csv 와 run 이름으로 합쳐 볼 수 있음)
"""
import argparse, yaml
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('--ext', default='test_ext')
ap.add_argument('--project', default='runs_local')
ap.add_argument('--imgsz', type=int, default=640)
ap.add_argument('--device', default='0')
args = ap.parse_args()

if __name__ == '__main__':
    ROOT = Path(__file__).resolve().parent
    EXT = (ROOT / args.ext).resolve(); PROJECT = (ROOT / args.project).resolve()
    img_dir = next(p for p in [EXT / 'train' / 'images', EXT / 'images', EXT / 'test' / 'images'] if p.exists())

    # 모델은 5클래스, 외부셋 라벨은 0~3 만 사용 → nc=5 로 yaml 작성 (정수기는 GT 0개로 자동 제외)
    NAMES = ['2_class', '4_class', 'front_door', 'rear_door', 'water_dispenser']
    yml = PROJECT / 'data_ext.yaml'
    yaml.safe_dump({'path': str(EXT), 'train': str(img_dir), 'val': str(img_dir), 'nc': 5, 'names': NAMES},
                   open(yml, 'w', encoding='utf-8'), allow_unicode=True)

    rows = []
    for w in sorted(PROJECT.glob('*/weights/best.pt')):
        run = w.parent.parent.name
        r = YOLO(str(w)).val(data=str(yml), split='val', imgsz=args.imgsz, device=args.device, plots=True, verbose=False,
                             project=str(PROJECT / '_val_tmp'), name=f'{run}_ext', exist_ok=True)
        row = {'run': run, 'ext_P': round(float(r.box.mp), 3), 'ext_R': round(float(r.box.mr), 3),
               'ext_mAP50': round(float(r.box.map50), 3), 'ext_mAP50-95': round(float(r.box.map), 3),
               'ext_infer_ms': round(float(r.speed['inference']), 2)}
        for ci, ap50 in zip(r.box.ap_class_index, r.box.ap50):
            row[f'ext_AP50_{r.names[int(ci)]}'] = round(float(ap50), 3)
        rows.append(row); print(run, row['ext_mAP50'], row['ext_mAP50-95'])

    df = pd.DataFrame(rows).sort_values('ext_mAP50', ascending=False)
    df.to_csv(PROJECT / 'ext_comparison.csv', index=False)
    print('\n=== 외부 테스트셋 (100장, 4클래스) ===')
    print(df.to_string(index=False))

    # ---------- 이미지 단위 혼동행렬 (OCR 조원 결과와 같은 형식: 4클래스 + 무응답) ----------
    # 판정: 이미지에서 conf >= CONF 인 탐지 중 최고 신뢰도 클래스 = 응답, 없으면 무응답
    # 정답: 라벨 파일의 첫 클래스 (이미지당 대표 랜드마크 1개 기준)
    import numpy as np
    CONF = 0.5
    lbl_dir = img_dir.parent / 'labels'
    DISP = {'2_class': 'room2', '4_class': 'room4', 'front_door': 'front_door', 'rear_door': 'rear_door'}
    order = ['front_door', 'rear_door', 'room2', 'room4', '(무응답)']
    for w in sorted(PROJECT.glob('*/weights/best.pt')):
        run = w.parent.parent.name
        m = YOLO(str(w)); cm = pd.DataFrame(0, index=order[:4], columns=order)
        for img in sorted(img_dir.iterdir()):
            lf = lbl_dir / (img.stem + '.txt')
            if not lf.exists(): continue
            L = [l.split() for l in open(lf) if l.strip()]
            if not L: continue
            gt = DISP[NAMES[int(L[0][0])]]
            r = m.predict(img, imgsz=args.imgsz, conf=CONF, device=args.device, verbose=False)[0]
            if len(r.boxes):
                k = int(r.boxes.conf.argmax()); pred = DISP.get(r.names[int(r.boxes.cls[k])], '(무응답)')
            else:
                pred = '(무응답)'
            cm.loc[gt, pred] += 1
        n = cm.values.sum(); answered = n - cm['(무응답)'].sum()
        correct = sum(cm.loc[c, c] for c in order[:4])
        sign_n = cm.loc[['room2', 'room4']].values.sum(); sign_ans = sign_n - cm.loc[['room2', 'room4'], '(무응답)'].sum()
        print(f'\n[{run}] 이미지 단위 혼동행렬 (conf>={CONF})'); print(cm.to_string())
        print(f'  커버리지 {answered/n*100:.1f}%  |  응답 정확도 {correct/max(answered,1)*100:.1f}%  |  표지판 커버리지 {sign_ans/max(sign_n,1)*100:.1f}%')
        cm.to_csv(PROJECT / f'ext_cm_{run}.csv', encoding='utf-8-sig')
        try:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            from matplotlib import font_manager
            fig, ax = plt.subplots(figsize=(6, 4.5)); im = ax.imshow(cm.values, cmap='Blues')
            ax.set_xticks(range(5)); ax.set_xticklabels(order, rotation=30); ax.set_yticks(range(4)); ax.set_yticklabels(order[:4])
            for i in range(4):
                for j in range(5): ax.text(j, i, cm.values[i, j], ha='center', va='center', color='white' if cm.values[i, j] > cm.values.max()/2 else 'black')
            ax.set_title(f'{run} - external test (per image)'); plt.colorbar(im); plt.tight_layout()
            plt.savefig(PROJECT / f'ext_cm_{run}.png', dpi=150); plt.close()
        except Exception as e:
            print('plot skip:', e)
    print('\n저장:', PROJECT / 'ext_comparison.csv', '+ ext_cm_<run>.csv/.png')
