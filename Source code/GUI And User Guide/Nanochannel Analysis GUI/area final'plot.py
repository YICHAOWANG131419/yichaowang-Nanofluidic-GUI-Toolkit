import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import PercentFormatter   # 新增

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

# 把 B 当成百分比（Crossing 1 memory retention, %）
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

# 扩展频率范围：1e-3 ~ 1e-1
fmin_ext = 1e-3
fmax_ext = 1e-1
freq_fit = np.logspace(np.log10(fmin_ext), np.log10(fmax_ext), 400)

B_fit = lognormal(freq_fit, *params_B)
C_fit = lognormal(freq_fit, *params_C)

# ---------------------------
# 4. 主图：双 y 轴 + log x
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

gui_color = 'tab:blue'
paper_color = 'red'

# 左轴：GUI（蓝色实线 + 方块）
ax1.scatter(freq, B_pct, marker='s', color=gui_color, zorder=3)
line_gui, = ax1.plot(freq_fit, B_fit, color=gui_color, linewidth=1.5,
                     label='Crossing 1 (GUI)')

ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Crossing 1 memory retention (%)')   # label 保持默认黑色
ax1.set_ylim(0, 120.1)
ax1.set_xlim(fmin_ext, fmax_ext)

# 左侧：刻度和轴线蓝色
ax1.tick_params(axis='y', colors=gui_color)
ax1.spines['left'].set_color(gui_color)
ax1.spines['right'].set_visible(False)      # 关闭 ax1 的右轴，避免和 ax2 重叠

ax1.grid(True, which='both', axis='both',
         linestyle=':', linewidth=0.6, alpha=0.7)

# 右轴：paper（红色空心圆 + 虚线）
ax2 = ax1.twinx()
ax2.scatter(freq, C, marker='o', facecolors='none',
            edgecolors=paper_color, zorder=3)
line_paper, = ax2.plot(freq_fit, C_fit, color=paper_color, linewidth=1.5,
                       linestyle='--', label='Crossing 1 (paper)')

ax2.set_ylim(0.0167, 0.043)
ax2.yaxis.set_major_formatter(PercentFormatter(1.0))   # 把右侧刻度显示为百分比

# 右侧：不要 label，只保留刻度 & 轴线为红色
ax2.set_ylabel('')                     # 不要 y 轴标题
ax2.tick_params(axis='y', colors=paper_color)
ax2.spines['right'].set_color(paper_color)
ax2.spines['left'].set_visible(False)  # 把 ax2 的左轴关掉，不盖住蓝色那条

# legend 只放两条曲线
handles = [line_gui, line_paper]
labels = [h.get_label() for h in handles]
ax1.legend(handles, labels, loc='upper right', frameon=False)

plt.tight_layout()
plt.show()
