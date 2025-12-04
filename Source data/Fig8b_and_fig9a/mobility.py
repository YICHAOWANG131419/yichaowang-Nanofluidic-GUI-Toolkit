import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
})

labels = [
    "1 mM / 10 mM",
    "10 mM / 100 mM",
    "10 mM / 1 M",
    "1 mM / 100 mM",
    "100 mM / 1 M",
]
x = np.arange(len(labels), dtype=float)

# 真实数据
B = np.array([0.90177, 0.85476, 0.38707, 0.69353, 0.22840])    # GUI
C_paper = np.array([0.92,    0.86,    0.78,    0.70,    0.23]) # paper
D = np.array([np.nan, np.nan, 0.77416, np.nan, np.nan])         # corrected

discrepancy =  C_paper - B

fig, (ax, ax2) = plt.subplots(
    nrows=2, ncols=1,
    figsize=(6.4, 4.6),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

# -------- 上：Selectivity --------
# GUI
ax.scatter(
    x, B,
    marker="s", s=50,
    color="tab:blue",
    label="GUI (this work)",
    zorder=2
)

# paper，橙色空心圆
ax.scatter(
    x, C_paper,
    facecolors="none",
    edgecolors="tab:orange",
    marker="o", s=55,
    linewidths=1.4,
    label="Paper (reported)",
    zorder=3
)

# corrected
ax.scatter(
    x, D,
    marker="^", s=60,
    color="tab:red",
    label="Paper (error reproduced)",
    zorder=4
)

ax.set_ylabel(r"Selectivity $S$")
ax.set_ylim(0.2, 1.0)
ax.set_xlim(-0.5, len(labels) - 0.5)

ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.legend(loc="upper right", frameon=False)

# -------- 下：Discrepancy --------
ax2.axhline(0.0, color="0.7", linewidth=0.8)
ax2.scatter(
    x, discrepancy,
    color="tab:blue", s=40
)

ax2.set_ylabel("Discrepancy")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=28, ha="right")
ax2.set_ylim(-0.05, 0.45)

ax2.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)

plt.subplots_adjust(hspace=0.08)
plt.tight_layout()
plt.show()
