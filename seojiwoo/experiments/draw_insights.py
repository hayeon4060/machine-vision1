"""두 장의 인사이트 그래프 — landmark_vision 폴더에서 실행
  .\.venv_gpu\Scripts\python.exe draw_insights.py
출력: fig1_model_selection.png, fig2_aug_paradox.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
NAVY, BLUE, GRAY, RED = '#1A3C8F', '#4C7DFF', '#9CA3AF', '#C43D3D'

# ---------- 그림 1: 왜 YOLOv8n인가 ----------
models = ['YOLOv8n', 'YOLOv8s', 'YOLO11n', 'YOLO11s']
val_map50 = [0.968, 0.962, 0.959, 0.960]
f1_sign   = [0.886, 0.743, 0.723, 0.794]      # 2_class F1 (valid)
ext_map50 = [0.765, 0.718, 0.728, 0.705]      # 외부셋 mAP50
size_mb   = [6.2, 22.5, 5.5, 19.2]

fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150)
x = np.arange(4)
def bars(a, vals, title, ylim, fmt='{:.3f}', highlight=0):
    cols = [NAVY if i == highlight else GRAY for i in range(4)]
    b = a.bar(x, vals, color=cols, width=0.62)
    for r, v in zip(b, vals): a.text(r.get_x() + r.get_width()/2, r.get_height(), fmt.format(v), ha='center', va='bottom', fontsize=10)
    a.set_xlim(-0.6, 3.6)
    a.set_xticks(x); a.set_xticklabels(models, fontsize=10); a.set_ylim(*ylim); a.set_title(title, fontsize=11.5, pad=8)
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
bars(ax[0], val_map50, 'valid mAP50 — 차이 0.01 이내 (포화)', (0.90, 1.0))
bars(ax[1], f1_sign, '2_class 표지판 F1 — v8n만 0.85 이상', (0.5, 1.0))
bars(ax[2], ext_map50, '외부 테스트셋 mAP50 — v8n 1위', (0.6, 0.85))
for a, s in zip(ax, size_mb): pass
ax[0].text(0.5, -0.22, '모델 크기: v8n 6.2MB · v8s 22.5MB · 11n 5.5MB · 11s 19.2MB   |   동일 조건: seed 0 · 80 epoch · imgsz 640 · batch 16',
           transform=ax[0].transAxes, fontsize=9.5, color='#4B5563')
fig.suptitle('왜 YOLOv8n인가 — 정확도는 포화, 약한 클래스와 외부 데이터에서 차이가 난다', fontsize=13, y=1.02)
plt.tight_layout(); plt.savefig('fig1_model_selection.png', bbox_inches='tight'); plt.close()

# ---------- 그림 2: 증강이 왜 독이었나 ----------
runs = ['base', 'img960', 'aug', '960+aug']
val_sign_ap = [0.862, 0.928, 0.995, 0.995]          # valid 2_class AP50 (5장)
ext_sign_cov = [73.5, 69.4, 42.9, 65.3]             # 외부셋 표지판 커버리지 %
ext_map = [0.765, 0.760, 0.733, 0.715]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
x = np.arange(4); w = 0.62
b = ax[0].bar(x, val_sign_ap, color=[GRAY, GRAY, BLUE, BLUE], width=w)
for r, v in zip(b, val_sign_ap): ax[0].text(r.get_x()+w/2, r.get_height(), f'{v:.3f}', ha='center', va='bottom', fontsize=10)
ax[0].set_xticks(x); ax[0].set_xticklabels(runs); ax[0].set_ylim(0.7, 1.05)
ax[0].set_title('내부 valid — 2_class AP50 (표지판 5장)\n증강이 "완벽"해 보임', fontsize=11.5, pad=8)
b = ax[1].bar(x, ext_sign_cov, color=[NAVY, GRAY, RED, GRAY], width=w)
for r, v in zip(b, ext_sign_cov): ax[1].text(r.get_x()+w/2, r.get_height(), f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
ax[1].set_xticks(x); ax[1].set_xticklabels(runs); ax[1].set_ylim(0, 100)
ax[1].set_title('외부 테스트셋 100장 — 표지판 커버리지\n증강 모델이 base의 절반', fontsize=11.5, pad=8)
for a in ax: a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
ax[1].text(0.5, -0.2, '외부셋 mAP50: base 0.765 · img960 0.760 · aug 0.733 · 960+aug 0.715', transform=ax[1].transAxes,
           ha='center', fontsize=9.5, color='#4B5563')
fig.suptitle('증강이 왜 독이었나 — 작은 검증셋(표지판 5장)에서의 개선은 착시였다', fontsize=13, y=1.03)
plt.tight_layout(); plt.savefig('fig2_aug_paradox.png', bbox_inches='tight'); plt.close()
print('저장: fig1_model_selection.png, fig2_aug_paradox.png')
