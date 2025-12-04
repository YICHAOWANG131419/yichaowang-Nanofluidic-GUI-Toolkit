# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 22:08:58 2025

@author: Tiezhu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取 Excel
path = r"C:\Users\Tiezhu\OneDrive\Desktop\Disseartation\Disertation 文件verify\2023 science memeristor\Figure_s12\S12_d\Figure supplementary D, 7 nm, 3M-.xlsx"

df = pd.read_excel(path, sheet_name="Sheet1")


# 第 0 行是单位, 第 1 行是频率标签, 第 2 行开始是数值
freq_row = df.iloc[1]       # 含有 "1 mHz" 这些字
data_numeric = df.iloc[2:]  # 真正的数字部分

# 2. 把每个频率的 (V, I) 提取出来, 转成 nA
iv_data = {}

for i_col in [c for c in df.columns if c.startswith("I")]:
    # 对应的电压列名: I -> E, I.1 -> E.1, ...
    e_col = "E" if i_col == "I" else "E" + i_col[1:]

    # 频率标签, 比如 "1 mHz"
    freq_label = str(freq_row[i_col])

    # 这一条曲线的数据, 去掉 NaN
    sub = data_numeric[[e_col, i_col]].dropna()

    V = sub[e_col].astype(float).to_numpy()           # Volt
    I = sub[i_col].astype(float).to_numpy() * 1e9     # A -> nA

    iv_data[freq_label] = (V, I)

# 3. 画 I-V 曲线

color_map = {
    "1 mHz": "#006400",        # 深灰
    "4 mHz": "tab:red",
    "10 mHz": "#CC79A7",
    "93 mHz": "tab:green",
    "112 mHz": "tab:purple",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

fig, ax = plt.subplots(figsize=(5.5, 4.5))

all_V = []
all_I = []

# 固定顺序和论文 legend 一致
plot_order = ["1 mHz", "4 mHz", "10 mHz", "93 mHz", "112 mHz"]

for label in plot_order:
    V, I = iv_data[label]
    all_V.append(V)
    all_I.append(I)

    ax.plot(
        V,
        I,
        color=color_map.get(label, "k"),
        linewidth=2.0,
        label=label,
    )

# 4. 设置对称坐标范围
all_V = np.concatenate(all_V)
all_I = np.concatenate(all_I)

xmax = np.max(np.abs(all_V))
ymax = np.max(np.abs(all_I))

ax.set_xlim(-xmax * 1.05, xmax * 1.05)
ax.set_ylim(-ymax * 1.05, ymax * 1.05)

# ========== 修改开始：把坐标轴移动到中心，形成十字 ==========
# 把下边框移到 y=0，把左边框移到 x=0
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')

# 只保留左/下两个 spine
ax.spines['bottom'].set_color('k')
ax.spines['left'].set_color('k')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['left'].set_linewidth(1.2)

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# 让刻度出现在 bottom / left
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.tick_params(direction='out', length=3.5, width=0.8)

# （可选）在正方向加个小箭头，模仿示例图
ax.annotate('', xy=(xmax * 1.05, 0), xytext=(xmax * 0.8, 0),
            arrowprops=dict(arrowstyle='->', color='k', lw=1.2),
            clip_on=False)
ax.annotate('', xy=(0, ymax * 1.05), xytext=(0, ymax * 0.8),
            arrowprops=dict(arrowstyle='->', color='k', lw=1.2),
            clip_on=False)
# ========== 修改结束 ========================================

ax.set_xlabel("Potential (V)")
ax.set_ylabel("Current (nA)")

# 把 x 轴标签挪到右侧一点（坐标是轴坐标系 0~1）
ax.xaxis.set_label_coords(0.9, 0.43)   # (x=1.02, y=0.52) 接近正 x 端点

# 把 y 轴标签挪到上方一点
ax.yaxis.set_label_coords(0.55, 0.9)   # (x=0.48, y=1.02) 接近正 y 端点

ax.grid(False)
ax.legend(loc="lower right", frameon=False)

plt.tight_layout()
plt.show()
