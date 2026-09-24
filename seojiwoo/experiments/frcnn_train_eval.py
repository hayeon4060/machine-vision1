# ============================================================
# Faster R-CNN / SSDLite (torchvision) — YOLO 라벨 데이터셋으로 학습·평가
# 사용: YOLO 노트북과 같은 세션에서 새 셀에 붙여넣고 실행
#       (DATASET_DIR, PROJECT, DATA_YAML 이 이미 정의돼 있어야 함)
# 결과는 YOLO와 같은 comparison.csv 에 같은 열로 추가됨
# ============================================================
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torchmetrics', 'pycocotools'], check=True)

import os, time, yaml, random
from pathlib import Path
import numpy as np, pandas as pd, torch, torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchmetrics.detection import MeanAveragePrecision

# ---------- 설정 ----------
MODEL_TYPE = 'frcnn'          # 'frcnn' (ResNet50-FPN v2) 또는 'ssdlite' (MobileNetV3)
EPOCHS_FR  = 30               # 소규모 데이터라 30이면 충분 (T4 약 20~30분)
BATCH_FR   = 4
IMGSZ_FR   = 640
LR         = 0.01 if MODEL_TYPE == 'frcnn' else 0.002
SEED       = 0
DEV        = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

cfg   = yaml.safe_load(open(DATA_YAML))
NAMES = cfg['names']; NC = len(NAMES)
RESULTS_CSV = PROJECT / 'comparison.csv'
RUN_NAME = {'frcnn': 'fasterrcnn_r50fpn_base', 'ssdlite': 'ssdlite_mnv3_base'}[MODEL_TYPE]
SAVE_DIR = PROJECT / RUN_NAME; SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- YOLO txt → 박스 (polygon 라인도 bbox로 변환) ----------
def read_yolo_labels(txt, W, H):
    boxes, labels = [], []
    if not txt.exists(): return boxes, labels
    for line in open(txt):
        p = line.split()
        if not p: continue
        c = int(p[0]); v = list(map(float, p[1:]))
        if len(v) == 4:                                  # cx cy w h
            cx, cy, w, h = v
            x1, y1, x2, y2 = (cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H
        else:                                            # polygon x1 y1 x2 y2 ...
            xs, ys = v[0::2], v[1::2]
            x1, y1, x2, y2 = min(xs)*W, min(ys)*H, max(xs)*W, max(ys)*H
        if x2 - x1 < 1 or y2 - y1 < 1: continue
        boxes.append([x1, y1, x2, y2]); labels.append(c + 1)   # 0은 배경
    return boxes, labels

class YoloDetDataset(Dataset):
    def __init__(self, split, train=False):
        self.img_dir = Path(DATASET_DIR) / split / 'images'
        self.lbl_dir = Path(DATASET_DIR) / split / 'labels'
        self.files = sorted(p for p in self.img_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        self.train = train
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        p = self.files[i]
        img = Image.open(p).convert('RGB')
        # 긴 변을 IMGSZ_FR 로 리사이즈 (속도 + YOLO와 동일 조건)
        s = IMGSZ_FR / max(img.size); img = img.resize((round(img.width*s), round(img.height*s)))
        W, H = img.size
        boxes, labels = read_yolo_labels(self.lbl_dir / (p.stem + '.txt'), W, H)
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.int64)
        img = TF.to_tensor(img)
        if self.train and random.random() < 0.5:         # 수평 뒤집기 증강
            img = img.flip(-1)
            if len(boxes): boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
        return img, {'boxes': boxes, 'labels': labels}

collate = lambda b: tuple(zip(*b))
dl_train = DataLoader(YoloDetDataset('train', True), batch_size=BATCH_FR, shuffle=True, num_workers=0, collate_fn=collate)
dl_val   = DataLoader(YoloDetDataset('valid'), batch_size=4, num_workers=0, collate_fn=collate)
dl_test  = DataLoader(YoloDetDataset('test'),  batch_size=4, num_workers=0, collate_fn=collate)

# ---------- 모델 ----------
def build_model():
    if MODEL_TYPE == 'frcnn':
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        m = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                                       min_size=IMGSZ_FR, max_size=IMGSZ_FR)
        m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, NC + 1)
    else:
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
        from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
        from functools import partial
        m = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        in_ch = [c.in_channels for c in m.head.classification_head.module_list]  # 각 피처맵 채널
        anchors = m.anchor_generator.num_anchors_per_location()
        m.head.classification_head = SSDLiteClassificationHead(in_ch, anchors, NC + 1,
                                                                partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03))
    return m.to(DEV)

model = build_model()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(EPOCHS_FR*0.65), int(EPOCHS_FR*0.9)], gamma=0.1)
scaler = torch.cuda.amp.GradScaler(enabled=(DEV == 'cuda'))

# ---------- 평가 ----------
def box_iou(a, b): return torchvision.ops.box_iou(a, b)

@torch.no_grad()
def evaluate(loader, conf_thr=0.5, iou_thr=0.5):
    model.eval()
    metric = MeanAveragePrecision(iou_type='bbox', class_metrics=True)
    tp = fp = fn = 0; times = []; n_img = 0
    for imgs, tgts in loader:
        imgs = [i.to(DEV) for i in imgs]
        if DEV == 'cuda': torch.cuda.synchronize()
        t0 = time.time(); outs = model(imgs)
        if DEV == 'cuda': torch.cuda.synchronize()
        times.append((time.time() - t0) / len(imgs)); n_img += len(imgs)
        outs = [{k: v.cpu() for k, v in o.items()} for o in outs]
        metric.update(outs, tgts)
        # P / R @ conf>=0.5, IoU>=0.5 (클래스 일치 기준 그리디 매칭)
        for o, t in zip(outs, tgts):
            keep = o['scores'] >= conf_thr
            pb, pl = o['boxes'][keep], o['labels'][keep]
            gb, gl = t['boxes'], t['labels']
            matched = torch.zeros(len(gb), dtype=torch.bool)
            for j in pb.argsort(descending=False) if False else range(len(pb)):
                if len(gb) == 0: fp += 1; continue
                ious = box_iou(pb[j:j+1], gb)[0]
                ious[gl != pl[j]] = 0; ious[matched] = 0
                k = int(ious.argmax())
                if ious[k] >= iou_thr: tp += 1; matched[k] = True
                else: fp += 1
            fn += int((~matched).sum())
    r = metric.compute()
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    per_cls = {}
    if 'map_per_class' in r and r['map_per_class'].ndim:
        # torchmetrics 는 클래스별 mAP50-95 만 직접 제공 → AP50 은 IoU 0.5 단독 metric 으로 재계산
        pass
    return dict(P=P, R=R, mAP50=float(r['map_50']), mAP50_95=float(r['map']),
                infer_ms=float(np.mean(times) * 1000)), r

@torch.no_grad()
def per_class_ap50(loader):
    model.eval()
    m = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.5], class_metrics=True)
    for imgs, tgts in loader:
        outs = model([i.to(DEV) for i in imgs])
        m.update([{k: v.cpu() for k, v in o.items()} for o in outs], tgts)
    r = m.compute()
    out = {}
    for c, ap in zip(r['classes'].tolist(), r['map_per_class'].tolist()):
        out[NAMES[c - 1]] = round(ap, 3)
    return out

# ---------- 학습 ----------
best_path = SAVE_DIR / 'best.pt'
if best_path.exists():
    print(f'[skip] {RUN_NAME} 이미 학습됨'); model.load_state_dict(torch.load(best_path, map_location=DEV)); train_time = 0
else:
    best_map, t0, log = -1, time.time(), []
    for ep in range(1, EPOCHS_FR + 1):
        model.train(); tot = 0
        for imgs, tgts in dl_train:
            imgs = [i.to(DEV) for i in imgs]; tgts = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            with torch.autocast(device_type='cuda', enabled=(DEV == 'cuda')):
                loss = sum(model(imgs, tgts).values())
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += float(loss)
        sched.step()
        v, _ = evaluate(dl_val)
        log.append({'epoch': ep, 'loss': tot / len(dl_train), **{f'val_{k}': round(x, 4) for k, x in v.items()}})
        print(f"ep {ep:3d}  loss {tot/len(dl_train):.3f}  val mAP50 {v['mAP50']:.3f}  mAP50-95 {v['mAP50_95']:.3f}")
        if v['mAP50_95'] > best_map:
            best_map = v['mAP50_95']; torch.save(model.state_dict(), best_path)
    train_time = time.time() - t0
    pd.DataFrame(log).to_csv(SAVE_DIR / 'results.csv', index=False)
    model.load_state_dict(torch.load(best_path, map_location=DEV))

# ---------- 최종 평가 & CSV 기록 ----------
row = {'run': RUN_NAME, 'note': 'baseline',
       'params_M': round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
       'size_MB': round(os.path.getsize(best_path) / 1e6, 1),
       'train_min': round(train_time / 60, 1)}
for split, dl in [('val', dl_val), ('test', dl_test)]:
    v, _ = evaluate(dl)
    row[f'{split}_P'] = round(v['P'], 3); row[f'{split}_R'] = round(v['R'], 3)
    row[f'{split}_mAP50'] = round(v['mAP50'], 3); row[f'{split}_mAP50-95'] = round(v['mAP50_95'], 3)
    row[f'{split}_infer_ms'] = round(v['infer_ms'], 2)
    for n, ap in per_class_ap50(dl).items(): row[f'{split}_AP50_{n}'] = ap

df = pd.read_csv(RESULTS_CSV) if RESULTS_CSV.exists() else pd.DataFrame()
if len(df): df = df[df['run'] != RUN_NAME]
df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
df.to_csv(RESULTS_CSV, index=False)
print(pd.DataFrame([row]).T)
