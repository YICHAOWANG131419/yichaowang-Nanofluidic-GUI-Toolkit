import numpy as np
import matplotlib.pyplot as plt

# 数据
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

plt.figure(figsize=(6, 4.5))

# 蓝色 Published 线加粗、方块更大
plt.plot(
    T, P_pub,
    marker='s', linestyle='-',
    color='tab:blue',
    markersize=9,
    markeredgewidth=2.0,
    linewidth=3.0,          # 线加粗
    label='Published'
)

# 黄色 Calculated 线保持之前设置
plt.plot(
    T, P_calc,
    marker='o', linestyle='--',
    color='gold',
    markersize=5,
    markeredgewidth=1.5,
    linewidth=2.0,
    label='Calculated'
)

plt.xlabel('Temperature (K)')
plt.ylabel('Power density (W/m$^2$)')
plt.legend()

# 不画 grid

plt.tight_layout()
plt.show()
