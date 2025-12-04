# -*- coding: utf-8 -*-
"""
Crossing-style IV plot
Blue solid: I_sc
Black dashed: I_os
"""

import numpy as np
import matplotlib.pyplot as plt

# 原始数据
V = np.array([
    -250.41473786794836,
    -220.86203459655812,
    -200.35991937805935,
    -99.79789142616664,
    1.1844116148752164,
    101.40368557443134
])

I = np.array([
    -2.543467186427165,
    -0.6274779064874068,
    1.6337017431172356,
    14.460231677335045,
    29.311723404726568,
    46.02039912290415
])

# 虚线通过的两点（I_os）
V_extra = np.array([-61.80203326762495, 0.9130877760304656])
I_extra = np.array([-0.25537664178608566, 8.636846884759358])

# 实线 I_sc 拟合 I = aV + b
a, b = np.polyfit(V, I, 1)

# 虚线 I_os 拟合 I = mV + c
m_extra, c_extra = np.polyfit(V_extra, I_extra, 1)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

fig, ax = plt.subplots(figsize=(5.0, 4.0))

# 颜色（Nature 常见蓝 + 黑）
color_data = "#0072B2"   # 蓝色
color_dashed = "orange"   # 虚线黑色

# 数据点（不进 legend）
ax.plot(
    V, I,
    's',
    color=color_data,
    markersize=6,
    label='_nolegend_'
)

# 统一的 x 范围，用来画两条线（长度一样，覆盖整个坐标范围）
V_range = np.linspace(-270, 110, 400)

# I_sc 实线
I_sc_range = a * V_range + b
ax.plot(
    V_range, I_sc_range,
    color=color_data,
    linewidth=2.0,
    label=r"$I_{\mathrm{sc}}$"
)

# I_os 虚线
I_os_range = m_extra * V_range + c_extra
ax.plot(
    V_range, I_os_range,
    linestyle='--',
    color=color_dashed,
    linewidth=1.4,
    label=r"$I_{\mathrm{os}}$"
)

# 固定坐标范围
ax.set_xlim(-270, 110)
ax.set_ylim(-20, 50)

# 把坐标轴移到 0 点
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')

ax.spines['bottom'].set_color('k')
ax.spines['left'].set_color('k')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['left'].set_linewidth(1.2)

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.tick_params(direction='out', length=3.5, width=0.8)

# ===== 轴箭头：x 在 150，y 在 50 =====
ax.annotate(
    '',
    xy=(150, 0),          # 箭头头在 x=150
    xytext=(0.8 * 150, 0),
    arrowprops=dict(arrowstyle='->', color='k', lw=1.2),
    clip_on=False
)
ax.annotate(
    '',
    xy=(0, 50),           # 箭头头在 y=50
    xytext=(0, 0.8 * 50),
    arrowprops=dict(arrowstyle='->', color='k', lw=1.2),
    clip_on=False
)

ax.set_xlabel("Potential (V)")
ax.set_ylabel("Current (nA)")

ax.xaxis.set_label_coords(0.9, 0.43)
ax.yaxis.set_label_coords(0.55, 0.9)

ax.grid(False)

# —— 隐藏 x 轴上 -300, -250, 0, 150 这几个刻度标签 ——
xticks = ax.get_xticks()
xlabels = []
for t in xticks:
    if any(np.isclose(t, val, atol=1e-6) for val in (-300, -250, 0, 150)):
        xlabels.append("")  # 不显示这些刻度
    else:
        if abs(t - int(t)) < 1e-6:
            xlabels.append(f"{int(t)}")
        else:
            xlabels.append(f"{t:g}")
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)

# —— 隐藏 y 轴上 50 这个刻度标签 —— 
yticks = ax.get_yticks()
ylabels = []
for y in yticks:
    if np.isclose(y, 50, atol=1e-6):
        ylabels.append("")  # 顶部 50 不显示
    else:
        if abs(y - int(y)) < 1e-6:
            ylabels.append(f"{int(y)}")
        else:
            ylabels.append(f"{y:g}")
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)

# —— 把 y 轴上 “0” 往上挪一点，避免遮挡原点 ——
for label in ax.get_yticklabels():
    if label.get_text() == "0":
        x0, y0 = label.get_position()
        label.set_position((x0, y0 + 2))  # 往上移 2 个 nA

ax.legend(loc='upper left', frameon=False)

plt.tight_layout()
plt.show()
