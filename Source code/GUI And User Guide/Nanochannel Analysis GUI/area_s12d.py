# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 20:09:23 2025

@author: Tiezhu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------------------
# 1. Crossing 1 新数据（5 个点）
# ---------------------------
freq = np.array([0.001, 0.004, 0.01, 0.093, 0.112])
B = np.array([0.491992994, 0.694929944, 1.0, 0.715122148, 0.475296839])
C = np.array([0.053976494, 0.076036941, 0.116938885, 0.069860073, 0.067105703])

# 乘 100 变成百分比
B_pct = B * 100.0
C_pct = C * 100.0

# ---------------------------
# 2. log-normal 拟合函数（只在 x 上取 log）
# ---------------------------
def lognormal(f, A, mu, sigma, y0):
    x = np.log10(f)
    return y0 + A * np.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2))

# ---------------------------
# 3. 用真实数据点拟合 B_pct 和 C_pct
# ---------------------------
p0_B = [60.0, -2.0, 0.7, 40.0]    # 按百分比尺度给初值
params_B, _ = curve_fit(lognormal, freq, B_pct, p0=p0_B, maxfev=10000)

p0_C = [7.0, -2.0, 0.7, 5.0]
params_C, _ = curve_fit(lognormal, freq, C_pct, p0=p0_C, maxfev=10000)

# 拟合曲线用的频率范围
fmin_ext = 4e-4
fmax_ext = 2e-1
freq_fit = np.logspace(np.log10(fmin_ext), np.log10(fmax_ext), 400)

B_fit_pct = lognormal(freq_fit, *params_B)
C_fit_pct = lognormal(freq_fit, *params_C)

# ---------------------------
# 4. 画图：x 轴 log，y 轴线性（百分比）
# ---------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

fig, ax1 = plt.subplots(figsize=(6, 4.5))

# x 轴 log
ax1.set_xscale('log')
ax1.set_xlim(fmin_ext, fmax_ext)

gui_color = 'tab:blue'
paper_color = 'red'

# 左轴：GUI 数据 B_pct
ax1.scatter(freq, B_pct, marker='s', color=gui_color, zorder=3)
line_gui, = ax1.plot(freq_fit, B_fit_pct, color=gui_color, linewidth=1.5,
                     label='Device #4 (GUI)')

ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Loop Area Normalization (%)')   # 如果想显示百分比可改成 'Crossing 1 (GUI, %)' 等

# 左轴 y 范围和刻度（线性，百分比）
ax1.set_ylim(30, 121)
yticks_left = [40, 60, 80, 100]
ax1.set_yticks(yticks_left)

ax1.tick_params(axis='y', which='both', colors=gui_color)
ax1.spines['left'].set_color(gui_color)
ax1.spines['right'].set_visible(False)

#ax1.grid(True, which='both', axis='both',linestyle=':', linewidth=0.6, alpha=0.7)

# 右轴：paper 数据 C_pct
ax2 = ax1.twinx()

ax2.scatter(freq, C_pct, marker='o', facecolors='none',
            edgecolors=paper_color, zorder=3)
line_paper, = ax2.plot(freq_fit, C_fit_pct, color=paper_color, linewidth=1.5,
                       linestyle='--', label='Device #4 (paper)')

# 右轴 y 范围和刻度（线性，百分比）
ax2.set_ylim(4, 15.8)
yticks_right = [5, 8, 11]
ax2.set_yticks(yticks_right)

ax2.set_ylabel('')  # 需要的话可以改成 'Crossing 1 (paper, %)'
ax2.tick_params(axis='y', which='both', colors=paper_color)
ax2.spines['right'].set_color(paper_color)
ax2.spines['left'].set_visible(False)

# legend 只放两条曲线
handles = [line_gui, line_paper]
labels = [h.get_label() for h in handles]
ax1.legend(handles, labels, loc='upper right', frameon=False)

plt.tight_layout()
plt.show()
