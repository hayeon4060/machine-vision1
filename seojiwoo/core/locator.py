"""
locator.py — 위치 판정 모듈 (시연·평가·FastAPI 공용)

프레임 → YOLO 탐지 → bbox 크기로 거리 추정(핀홀) → 최근 N프레임 투표 → 상태/안내문
  상태: 'none'(탐지 없음) / 'far'(탐지, 거리 > near_m) / 'near'(거리 <= near_m → 안내)
  오경보 억제(3겹):
    - sign_conf: 표지판(2_class/4_class)만 더 높은 conf 요구 (문은 conf 그대로)
    - min_votes: window 안에서 같은 랜드마크가 min_votes 표 이상일 때만 판정
    - approach_only: 거리가 줄어드는 추세(접근 중)일 때만 안내, 옆으로 스쳐가는 건 무시
    - route: 동선 순서가 정해진 경우 "다음 랜드마크" 또는 "현재 랜드마크"만 안내 대상
  안내: 마지막으로 말한 문장과 다르면 즉시, 같으면 announce_every 초 후에만 (깜빡임 반복 방지)

사용:
    from locator import Locator
    loc = Locator('final_best.pt', window=5, min_votes=4, sign_conf=0.65, approach_only=True,
                  route=['front_door', '4_class', '2_class', 'rear_door'])
    out = loc.process(frame, t_sec)
"""
import time
from collections import deque, Counter
import numpy as np
from ultralytics import YOLO

REAL_SIZE = {'2_class': (0.20, 0.13), '4_class': (0.20, 0.13), 'front_door': (2.5, 4.4), 'rear_door': (2.1, 1.8),
             'logo': (1.00, 1.07)}                    # 실측: 세로 100cm × 가로 107cm
DISPLAY = {'2_class': '2강의실', '4_class': '4강의실', 'front_door': '앞문', 'rear_door': '뒷문', 'logo': '정면 로고 벽', 'sign': '강의실 표지판'}
REAL_SIZE['sign'] = REAL_SIZE['2_class']
SIGNS = {'2_class', '4_class'}


class Locator:
    def __init__(self, weights, f_norm=0.85, near_m=3.0, window=5, min_votes=4, conf=0.5, sign_conf=0.65,
                 approach_only=True, route=None, imgsz=640, device=None, announce_every=5.0, targets=tuple(REAL_SIZE),
                 verifier=None, merge_signs=False, logo_weights=None):
        self.model = YOLO(str(weights))
        # logo_weights: 로고 전용 모델(6클래스). 주 모델은 표지판·문, 로고는 이 모델에서만 가져옴
        #   (6클래스 재학습 시 소객체 표지판 탐지가 무너져 두 모델을 분리)
        self.logo_model = YOLO(str(logo_weights)) if logo_weights else None
        self.f_norm, self.near_m, self.window, self.min_votes = f_norm, near_m, window, max(1, min(min_votes, window))
        self.conf, self.sign_conf, self.approach_only = conf, sign_conf, approach_only
        self.route, self.route_idx = list(route) if route else None, -1     # -1: 아직 아무 랜드마크도 안 지남
        self.imgsz, self.device = imgsz, device
        self.announce_every, self.targets = announce_every, set(targets)
        self.buf = deque(maxlen=window)
        self.last_spoken, self.last_spoken_t = None, -1e9
        self.verifier, self.last_ocr = verifier, ''        # OCR 검증기(선택): 표지판 near 시 글자로 확인
        # merge_signs: 2_class/4_class 를 'sign' 하나로 합쳐 투표 → 숫자는 OCR 이 expected_sign 과 대조
        self.merge_signs, self.expected_sign = merge_signs, None
        self.ocr_hits, self.ocr_confirm_frames = 0, 2     # 연속 confirm 횟수 (2회 맞아야 안내)

    def _distance(self, name, x1, y1, x2, y2, W, H):
        h_real, w_real = REAL_SIZE[name]
        clipped = y1 < 2 or y2 > H - 2
        px = (x2 - x1) if clipped else (y2 - y1)
        real = w_real if clipped else h_real
        return float(self.f_norm * real * W / max(px, 1.0))

    def detect(self, frame):
        H, W = frame.shape[:2]
        results = [(self.model, False)] + ([(self.logo_model, True)] if self.logo_model else [])
        dets = []
        for model, logo_only in results:
            r = model.predict(frame, imgsz=self.imgsz, conf=min(self.conf, self.sign_conf), device=self.device, verbose=False)[0]
            for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                name = r.names[int(c)]
                if name not in self.targets: continue
                if self.logo_model and ((name == 'logo') != logo_only): continue   # 로고는 로고 모델에서만
                dets.append(self._make_det(name, b, s, W, H))
        return [d for d in dets if d is not None]

    def _make_det(self, name, b, s, W, H):
        if True:
            if float(s) < (self.sign_conf if name in SIGNS else self.conf): return None
            x1, y1, x2, y2 = map(float, b)
            yolo_name, name = name, ('sign' if (self.merge_signs and name in SIGNS) else name)
            return {'name': name, 'yolo_class': yolo_name, 'conf': float(s), 'box': (x1, y1, x2, y2),
                    'distance_m': self._distance(yolo_name, x1, y1, x2, y2, W, H)}

    def _allowed(self, name):
        """route 가 있으면 현재 또는 다음 랜드마크만 허용"""
        if not self.route: return True
        cur = self.route[self.route_idx] if self.route_idx >= 0 else None
        nxt = self.route[self.route_idx + 1] if self.route_idx + 1 < len(self.route) else None
        return name in (cur, nxt)

    def process(self, frame, t=None):
        return self.update(self.detect(frame), t, frame)

    def update(self, dets, t=None, frame=None):
        t = time.time() if t is None else t
        cand = [d for d in dets if self._allowed(d['name'])]
        top = min(cand, key=lambda d: d['distance_m']) if cand else None
        self.buf.append((top['name'], top['distance_m'], top['conf']) if top else (None, None, 0.0))

        landmark, distance, confidence, state, reason = None, None, 0.0, 'none', ''
        votes = Counter(n for n, _, _ in self.buf if n)
        if votes:
            name, n_votes = votes.most_common(1)[0]
            ds = [d for n, d, _ in self.buf if n == name]; cs = [c for n, _, c in self.buf if n == name]
            landmark, distance, confidence = name, float(np.median(ds)), float(np.mean(cs))
            if n_votes < self.min_votes:
                state, reason = 'far', f'votes {n_votes}/{self.window}'
            elif distance > self.near_m:
                state = 'far'
            elif self.approach_only and len(ds) >= 3 and ds[-1] > ds[0] * 1.15:
                state, reason = 'far', 'not approaching'          # 거리가 늘고 있음 → 지나가는 중
            else:
                state = 'near'
        # ---- OCR 검증: 표지판이 near 로 판정된 순간, 글자로 확인 (회의실/다른 숫자면 취소) ----
        if state == 'near' and (landmark in SIGNS or landmark == 'sign') and self.verifier is not None and frame is not None:
            box = next((d['box'] for d in dets if d['name'] == landmark), None)
            expected = self.expected_sign or (landmark if landmark in SIGNS else None)
            if box is not None and expected is not None:
                verdict, txt = self.verifier.verify(frame, box, expected); self.last_ocr = f'{verdict}:{txt}'
                if verdict == 'confirm': self.ocr_hits += 1
                elif verdict == 'reject': self.ocr_hits = 0
                if verdict == 'reject': state, reason = 'far', f'OCR reject "{txt}"'
                elif verdict == 'unknown' and self.merge_signs: state, reason = 'far', 'OCR unread'   # 합친 모드에선 숫자 확인 필수
                elif verdict == 'confirm' and self.ocr_hits < self.ocr_confirm_frames: state, reason = 'far', f'OCR {self.ocr_hits}/{self.ocr_confirm_frames}'

        text = f'앞쪽에 {DISPLAY[landmark]}이 있습니다' if state == 'near' else None
        announce = False
        if text is not None and (text != self.last_spoken or t - self.last_spoken_t >= self.announce_every):
            announce, self.last_spoken, self.last_spoken_t = True, text, t
            if self.route and landmark in self.route:            # 동선 진행: 안내한 랜드마크까지 전진
                self.route_idx = max(self.route_idx, self.route.index(landmark))

        return {'t': t, 'landmark': landmark, 'landmark_kr': DISPLAY.get(landmark), 'distance_m': distance,
                'state': state, 'reason': reason, 'confidence': confidence, 'text': text, 'announce': announce,
                'detections': dets, 'ocr': self.last_ocr, 'route_next': (self.route[self.route_idx + 1] if self.route and self.route_idx + 1 < len(self.route) else None)}

    def reset(self):
        self.buf.clear(); self.last_spoken, self.last_spoken_t, self.route_idx = None, -1e9, -1
