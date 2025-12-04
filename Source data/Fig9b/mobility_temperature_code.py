# -*- coding: utf-8 -*-
"""
Power density vs temperature:
GUI (this work) vs Paper (reported) + discrepancy
"""

import numpy as np
import matplotlib.pyplot as plt

# 全局字体风格保持不变
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
})

# ---------------------------
# 数据：温度 T, 发表值 P_pub, 程序计算 P_calc
# ---------------------------
T = np.array([296, 298, 303, 308, 313, 318, 323, 328, 333])

P_pub = np.array([
    9.6, 10.3, 13.3, 17.6, 24.7, 30.9, 40.6, 53.4, 65.1
])

P_calc = np.array([
    9.647050401,
    10.66919192,
    13.34935256,
    17.68728956,
    24.83612723,
    31.16353755,
    40.80029014,
    53.63733489,
    65.74155012
])

# x 轴直接用温度
x = T.astype(float)

# discrepancy：GUI - paper，和原来 B - C_paper 一样的逻辑
discrepancy = P_calc - P_pub

fig, (ax, ax2) = plt.subplots(
    nrows=2, ncols=1,
    figsize=(6.4, 4.6),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

# -------- 上：Power density --------
# GUI (this work) — 蓝色实心方块
ax.scatter(
    x, P_calc,
    marker="s", s=50,
    color="tab:blue",
    label="GUI (this work)",
    zorder=2
)

# Paper (reported) — 橙色空心圆
ax.scatter(
    x, P_pub,
    facecolors="none",
    edgecolors="tab:orange",
    marker="o", s=55,
    linewidths=1.4,
    label="Paper (reported)",
    zorder=3
)

ax.set_ylabel("Power density")
ax.set_xlim(T.min() - 1, T.max() + 1)

# y 轴范围稍微留一点边
y_min = min(P_pub.min(), P_calc.min())
y_max = max(P_pub.max(), P_calc.max())
ax.set_ylim(y_min * 0.9, y_max * 1.05)

ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.legend(loc="upper left", frameon=False)

# -------- 下：Discrepancy --------
ax2.axhline(0.0, color="0.7", linewidth=0.8)
ax2.scatter(
    x, discrepancy,
    color="tab:blue", s=40
)

ax2.set_ylabel("Discrepancy")
ax2.set_xlabel("Temperature (K)")

ax2.set_xticks(T)
ax2.set_xticklabels(T)

# discrepancy 范围也留一点 margin
d_min = discrepancy.min()
d_max = discrepancy.max()
margin = max(abs(d_min), abs(d_max)) * 0.2
ax2.set_ylim(d_min - margin, d_max + margin)

ax2.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)

plt.subplots_adjust(hspace=0.08)
plt.tight_layout()
plt.show()
