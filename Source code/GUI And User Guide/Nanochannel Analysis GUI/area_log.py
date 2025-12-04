import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------------------
# 1. Crossing 1 数据（5 个点）
# ---------------------------
freq = np.array([0.0032, 0.008, 0.016, 0.0325, 0.0735])
B = np.array([
    0.937486599,
    1.0,
    0.52414541,
    0.299723564,
    0.245149525
])
C = np.array([0.03757, 0.03938, 0.02856, 0.02396, 0.02077])

# 左轴：把 B 当成百分比（Crossing 1 memory retention, %）
B_pct = B * 100.0

# ---------------------------
# 2. log-normal 拟合函数
# ---------------------------
def lognormal(f, A, mu, sigma, y0):
    x = np.log10(f)
    return y0 + A * np.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2))

# ---------------------------
# 3. 用真实数据点拟合 B_pct 和 C
# ---------------------------
p0_B = [60.0, -2.0, 0.7, 40.0]
params_B, _ = curve_fit(lognormal, freq, B_pct, p0=p0_B, maxfev=10000)

p0_C = [0.03, -2.0, 0.7, 0.02]
params_C, _ = curve_fit(lognormal, freq, C, p0=p0_C, maxfev=10000)

# 扩展频率范围：还是 1e-3 ~ 1e-1（你刚刚用的是 9.8e-4，我也保留）
fmin_ext = 9.8e-4
fmax_ext = 1e-1
freq_fit = np.logspace(np.log10(fmin_ext), np.log10(fmax_ext), 400)

B_fit = lognormal(freq_fit, *params_B)
C_fit = lognormal(freq_fit, *params_C)

# ---- 关键：对 y 数据自己取 log10，用线性轴画 ----
B_log = np.log10(B_pct)
B_fit_log = np.log10(B_fit)

C_pct = C * 100.0          # 右轴以百分数画（2~4）
C_log = np.log10(C_pct)
C_fit_log = np.log10(C_fit * 100.0)

# ---------------------------
# 4. 主图：x 轴 log，不动；y 轴用 log10(数据) + 手动画刻度
# ---------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

fig, ax1 = plt.subplots(figsize=(6, 4.5))

# x 轴：跟原来一样
ax1.set_xscale('log')
ax1.set_xlim(fmin_ext, fmax_ext)

gui_color = 'tab:blue'
paper_color = 'red'

# 左轴：GUI（蓝色实线 + 方块），画的是 log10(B_pct)
ax1.scatter(freq, B_log, marker='s', color=gui_color, zorder=3)
line_gui, = ax1.plot(freq_fit, B_fit_log, color=gui_color, linewidth=1.5,
                     label='Crossing 1 (GUI)')

ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Loop Area Normalization (%)')

# 左轴 y 取值范围：log10(20) ~ log10(120)
ax1.set_ylim(np.log10(20), np.log10(120.1))

# 左轴 y 刻度（显示成 20, 30, 40, 60, 80, 100）
yticks_left = [20, 30, 40, 60, 80, 100]
ax1.set_yticks(np.log10(yticks_left))
ax1.set_yticklabels([f"{y:.0f}" for y in yticks_left])

# 左侧：刻度和轴线全蓝色
ax1.tick_params(axis='y', which='both', colors=gui_color)
ax1.spines['left'].set_color(gui_color)
ax1.spines['right'].set_visible(False)

#ax1.grid(True, which='both', axis='both',linestyle=':', linewidth=0.6, alpha=0.7)

# 右轴：paper（红色空心圆 + 虚线），画的是 log10(C_pct)
ax2 = ax1.twinx()

ax2.scatter(freq, C_log, marker='o', facecolors='none',
            edgecolors=paper_color, zorder=3)
line_paper, = ax2.plot(freq_fit, C_fit_log, color=paper_color, linewidth=1.5,
                       linestyle='--', label='Crossing 1 (paper)')

# 右轴 y 范围：log10(2) ~ log10(4.3)
ax2.set_ylim(np.log10(2.7), np.log10(4.39))

# 右轴刻度（2.00, 2.50, 3.00, 3.50, 4.00），不带百分号
yticks_right = [2.0, 2.5, 3.0, 3.5, 4.0]
ax2.set_yticks(np.log10(yticks_right))
ax2.set_yticklabels([f"{y:.2f}" for y in yticks_right])

# 右侧：不要 label，只保留刻度 & 轴线为红色
ax2.set_ylabel('')
ax2.tick_params(axis='y', which='both', colors=paper_color)
ax2.spines['right'].set_color(paper_color)
ax2.spines['left'].set_visible(False)

# legend 只放两条曲线
handles = [line_gui, line_paper]
labels = [h.get_label() for h in handles]
ax1.legend(handles, labels, loc='upper right', frameon=False)

plt.tight_layout()
plt.show()
