"""
navigator.py — 시나리오 상태 머신 (locator 위에 얹는 "지금 뭘 해야 하나" 층)

  0 find_logo   : 뒤돌아 로고 찾기 → 보이면 좌/우/직진 방향 안내
  1 reach_wall  : 로고 접근 → 거리 ≤ wall_m 이면 "벽, 왼쪽으로 돌아 벽 따라 이동"
  2 room4       : 4강의실 near → "4강의실 앞, 계속 직진"
  3 room2       : 2강의실 near (+OCR 확인) → "2강의실 앞, 왼쪽으로 돌아 뒷문 찾기"
  4 rear_door   : 뒷문 보이면 좌/우/직진 → near → "목적지 도착"

방향: bbox 중심 x / 화면 너비.  < 0.35 → 왼쪽,  > 0.65 → 오른쪽,  그 사이 → 직진
안내: 문장이 바뀌면 즉시, 같으면 repeat_s 마다. 방향 안내는 steer_s 간격.
"""
import time
from locator import Locator, DISPLAY

STEPS = [
    dict(key='find_logo',  target='logo',       near_m=None),
    dict(key='reach_wall', target='logo',       near_m=1.2),
    dict(key='room4',      target='sign',       near_m=3.0, expect='4_class'),   # YOLO=표지판, 숫자는 OCR
    dict(key='room3',      target='sign',       near_m=3.0, expect='3_class'),   # 중간 확인 (YOLO 클래스 없음, OCR 만)
    dict(key='room2',      target='sign',       near_m=3.0, expect='2_class'),
    dict(key='rear_door',  target='rear_door',  near_m=3.0),
]
SEARCH_HINT = {'find_logo': '뒤로 천천히 도세요. 로고를 찾습니다',
               'reach_wall': '로고를 향해 직진하세요',
               'room4': '왼쪽 벽을 따라 이동하세요. 4강의실을 찾습니다',
               'room3': '계속 벽을 따라 이동하세요. 3강의실을 지납니다',
               'room2': '계속 벽을 따라 이동하세요. 2강의실을 찾습니다',
               'rear_door': '왼쪽으로 돌아 뒷문을 찾으세요'}
ON_REACH = {'reach_wall': '벽입니다. 왼쪽으로 돌아 벽을 따라 이동하세요',
            'room4': '4강의실 앞입니다. 계속 직진하세요',
            'room3': '3강의실 앞입니다. 계속 직진하세요',
            'room2': '2강의실 앞입니다. 왼쪽으로 돌아 뒷문을 찾으세요',
            'rear_door': '뒷문입니다. 목적지에 도착했습니다'}


class Navigator:
    def __init__(self, locator: Locator, left=0.35, right=0.65, repeat_s=6.0, steer_s=2.5, wall_m=1.2):
        self.loc, self.left, self.right = locator, left, right
        self.repeat_s, self.steer_s = repeat_s, steer_s
        STEPS[1]['near_m'] = wall_m
        self.step = 0; self.last_text = None; self.last_t = -1e9; self.done = False
        self.clear_needed, self.clear_count = False, 0        # 표지판 확인 후 시야에서 사라질 때까지 대기
        self.loc.route = None                                  # 순서는 여기서 관리
        self.loc.merge_signs = True                            # 표지판 두 클래스 합쳐서 투표

    def _say(self, text, t, min_gap):
        if text != self.last_text or t - self.last_t >= min_gap:
            self.last_text, self.last_t = text, t; return True
        return False

    def _steer(self, box, W):
        cx = (box[0] + box[2]) / 2 / W
        return 'left' if cx < self.left else 'right' if cx > self.right else 'center'

    def update(self, frame, t=None):
        t = time.time() if t is None else t
        step = STEPS[self.step]; key, target = step['key'], step['target']
        if self.loc.expected_sign != step.get('expect'): self.loc.ocr_hits = 0
        self.loc.expected_sign = step.get('expect')            # OCR 이 대조할 숫자
        out = self.loc.process(frame, t)                       # 탐지·거리·투표·OCR 은 locator 그대로
        H, W = frame.shape[:2]
        # 직전 표지판이 아직 보이면 다음 표지판 단계를 시작하지 않음 (같은 표지판 재판독 방지)
        if self.clear_needed and target == 'sign':
            sign_visible = any(d['name'] == 'sign' for d in out['detections'])
            self.clear_count = 0 if sign_visible else self.clear_count + 1
            if self.clear_count < 3:
                text = SEARCH_HINT[key]; announce = self._say(text, t, self.repeat_s)
                return {**out, 'step': key, 'step_idx': self.step, 'target': target, 'steer': None,
                        'nav_text': text, 'nav_announce': announce, 'done': self.done}
            self.clear_needed = False
        text, announce, steer = None, False, None

        if self.done:
            text = '목적지에 도착했습니다'
        else:
            seen = out['landmark'] == target and out['state'] in ('far', 'near') and not str(out.get('reason', '')).startswith('votes')
            box = next((d['box'] for d in out['detections'] if d['name'] == target), None)
            if seen and box is not None:
                steer = self._steer(box, W)
                near = step['near_m'] is not None and out['distance_m'] is not None and out['distance_m'] <= step['near_m'] \
                       and not str(out.get('reason', '')).startswith('OCR')
                if near:
                    text = ON_REACH[key]
                    if self._say(text, t, 1e9):                 # 도달 안내는 한 번만
                        announce = True
                        if key == 'rear_door': self.done = True
                        elif key == 'find_logo': pass
                        self.step = min(self.step + 1, len(STEPS) - 1)
                        if key in ('reach_wall', 'room4', 'room3', 'room2'): self.loc.buf.clear(); self.loc.ocr_hits = 0
                        if key in ('room4', 'room3'): self.clear_needed, self.clear_count = True, 0
                elif key == 'find_logo' and steer == 'center':
                    text = '로고가 정면입니다. 직진하세요'
                    if self._say(text, t, self.repeat_s): announce = True; self.step = 1
                else:
                    d = f" {out['distance_m']:.0f}미터" if out['distance_m'] else ''
                    text = {'left': f'{DISPLAY[target]}이 왼쪽에 있습니다. 조금 왼쪽으로',
                            'right': f'{DISPLAY[target]}이 오른쪽에 있습니다. 조금 오른쪽으로',
                            'center': f'{DISPLAY[target]}이 정면{d}. 직진하세요'}[steer]
                    if self._say(text, t, self.steer_s): announce = True
            else:
                text = SEARCH_HINT[key]
                if self._say(text, t, self.repeat_s): announce = True

        return {**out, 'step': key, 'step_idx': self.step, 'target': target, 'steer': steer,
                'nav_text': text, 'nav_announce': announce, 'done': self.done}
