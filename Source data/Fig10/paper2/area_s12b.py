# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 20:54:02 2025

@author: Tiezhu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------------------
# 1. Crossing 1 新数据（5 个点，去掉 0.2 Hz）
# ---------------------------
freq = np.array([1.00e-03, 5.00e-03, 1.00e-02,
                 5.00e-02, 1.00e-01])

B = np.array([
    0.483474877,
    0.86196791,
    1.0,
    0.585470481,
    0.495664985
])

C = np.array([
    0.06895477516257782,
    0.13518418753349498,
    0.16212168624355672,
    0.08701025918005206,
    0.06958190455438071
])

# 乘 100 变成百分比
B_pct = B * 100.0
C_pct = C * 100.0

# ---------------------------
# 2. log-normal 拟合函数
# ---------------------------
def lognormal(f, A, mu, sigma, y0):
    x = np.log10(f)
    return y0 + A * np.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2))

# ---------------------------
# 3. 拟合 B_pct 和 C_pct（各自在自己 scale 上拟合）
# ---------------------------
p0_B = [60.0, -2.0, 0.7, 40.0]
params_B, _ = curve_fit(lognormal, freq, B_pct, p0=p0_B, maxfev=10000)

p0_C = [15.0, -1.5, 0.8, 3.0]
params_C, _ = curve_fit(lognormal, freq, C_pct, p0=p0_C, maxfev=10000)

# 频率范围
fmin_ext = 1.0e-4
fmax_ext = 4.0e-1
freq_fit = np.logspace(np.log10(fmin_ext), np.log10(fmax_ext), 400)

B_fit_pct = lognormal(freq_fit, *params_B)
C_fit_pct = lognormal(freq_fit, *params_C)

# ---------------------------
# 4. 画图：左右 y 轴独立 scale
# ---------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

fig, ax1 = plt.subplots(figsize=(6, 4.5))

ax1.set_xscale('log')
ax1.set_xlim(fmin_ext, fmax_ext)

gui_color = 'tab:blue'
paper_color = 'red'

# ===== 左轴：GUI =====
ax1.scatter(freq, B_pct, marker='s', color=gui_color, zorder=3)
line_gui, = ax1.plot(freq_fit, B_fit_pct, color=gui_color, linewidth=1.5,
                     label='Device #2 (GUI)')

ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Loop Area Normalization (%)')

# 左轴范围 & 刻度（你可以随时改这两行）
left_ylim = (30, 105)
left_yticks = [40, 50, 60, 100]

ax1.set_ylim(*left_ylim)
ax1.set_yticks(left_yticks)

ax1.tick_params(axis='y', which='both', colors=gui_color)
ax1.spines['left'].set_color(gui_color)
ax1.spines['right'].set_visible(False)

#ax1.grid(True, which='both', axis='both',linestyle=':', linewidth=0.6, alpha=0.7)

# ===== 右轴：paper =====
ax2 = ax1.twinx()

ax2.scatter(freq, C_pct, marker='o', facecolors='none',
            edgecolors=paper_color, zorder=3)
line_paper, = ax2.plot(freq_fit, C_fit_pct, color=paper_color, linewidth=1.5,
                       linestyle='--', label='Device #2 (paper)')

# 右轴范围 & 刻度（这里也可以自己调）
right_ylim = (4, 16)           # 覆盖 ~6–16% 和拟合的峰
right_yticks = [6, 10, 14,18]

ax2.set_ylim(right_ylim)
ax2.set_yticks(right_yticks)

ax2.set_ylabel('')
ax2.tick_params(axis='y', which='both', colors=paper_color)
ax2.spines['right'].set_color(paper_color)
ax2.spines['left'].set_visible(False)

# legend
handles = [line_gui, line_paper]
labels = [h.get_label() for h in handles]
ax1.legend(handles, labels, loc='upper right', frameon=False)

plt.tight_layout()
plt.show()
