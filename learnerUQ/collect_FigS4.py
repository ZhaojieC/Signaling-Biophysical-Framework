#!/usr/bin/env python3
"""Collect the Monte-Carlo trials and draw Fig S4 (parameter scatter matrix).

Reads k_out/k_*.npz written by mc_trial.py, writes:
    all_k.npy        (N_trials x 8)  learned parameters in absolute units, i.e.
                     k_list * k_norm, so they are directly comparable with the
                     nominal values and units quoted in the parameter table
    FigS4.png/.pdf   scatter matrix: marginal histogram on the diagonal,
                     pairwise scatter off-diagonal, true value marked in red
    FigS4_corr.txt   Pearson correlation matrix of the 8 parameters
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from example_rbcUQ3 import K_INDEX, K_LIST, K_NORM

# Plot absolute units (k_list * k_norm). The trials are stored both ways, and
# the normalised k_rel values disagree with the parameter table for six of the
# eight parameters (K9 reads 2 there but 0.02 in the table, K23 reads 0.6 but
# 6e4), which is a trap for anyone checking the figure against the table.
# Rescaling by a constant shifts log10(K) by a constant, so it leaves every
# correlation, spread and decade count in the figure and the text untouched.

# --- label convention -------------------------------------------------------
# The code indexes the rate constants from 0 (K_INDEX above -> K7, K8, K10 ...).
# The published Fig S4 labels them from 1 (K8, K9, K11 ...). Set to 0 to fall
# back to the code's own convention.
LABEL_OFFSET = 1
LABELS = [f"K{i + LABEL_OFFSET}" for i in K_INDEX]

TRUE = np.array([K_LIST[i] * K_NORM[i] for i in K_INDEX])   # absolute units

# Every trial is kept. An earlier version dropped runs deviating by >100x, but
# the deviations form a smooth continuum with no gap (95th pct = 2.06 decades,
# 90th = 1.90), so any such cut splits near-identical trials arbitrarily. The
# runs where K15/K11 collapse toward zero are the actual result - those
# parameters are not identifiable - and must not be filtered out of the figure.
# One run does diverge outright and is excluded: its worst parameter sits 3.5
# decades from truth while every other run stays within 2.2, so unlike the
# K11/K15 collapses this one is genuinely isolated. It is reported by seed.
DROP_NONCONVERGED = False
DIVERGED_DECADES = 3.0

# The network parameterises K = exp(theta), so the parameters are multiplicative
# and both the axes and the correlations belong in log space.
LOG_SCALE = True

# --- collect ----------------------------------------------------------------
files = sorted(glob.glob("k_out/k_*.npz"))
if not files:
    raise SystemExit("no k_out/k_*.npz found - has the array job finished?")
K, seeds = [], []
for f in files:
    d = np.load(f)
    K.append(d["k_abs"])
    seeds.append(int(d["seed"]))
K = np.array(K)
print(f"collected {len(K)} trials from {len(files)} files")

n_total = len(K)
dev = np.abs(np.log10(K / TRUE[None, :])).max(axis=1)
good = dev <= DIVERGED_DECADES
for i in np.where(~good)[0]:
    j = int(np.argmax(np.abs(np.log10(K[i] / TRUE))))
    print(f"excluding diverged run seed {seeds[i]}: {LABELS[j]} = {K[i,j]:.3g} "
          f"vs true {TRUE[j]:g} ({dev[i]:.2f} decades; "
          f"worst surviving run {dev[good].max():.2f})")
K, seeds = K[good], list(np.array(seeds)[good])

np.save("all_k.npy", K)
n = len(K)
print(f"all_k.npy written: {K.shape}")
# Median and the 16-84 percentile range: the distributions are heavy-tailed and
# span decades, so a mean +- sd would be dominated by the collapsed runs.
print(f"{'param':>6} {'true':>9} {'median':>9} {'16th':>9} {'84th':>9} "
      f"{'decades':>8} {'<0.1x':>6}")
for j, lab in enumerate(LABELS):
    lo, md, hi = np.percentile(K[:, j], [16, 50, 84])
    frac_lo = 100.0 * np.mean(K[:, j] < 0.1 * TRUE[j])
    print(f"{lab:>6} {TRUE[j]:9.4g} {md:9.4g} {lo:9.4g} {hi:9.4g} "
          f"{np.log10(hi/max(lo,1e-12)):8.2f} {frac_lo:5.0f}%")

# --- correlation ------------------------------------------------------------
# Correlations are computed on log10(k): the network optimises log-parameters,
# and in linear space a handful of collapsed runs would dominate every r.
C = np.corrcoef(np.log10(K).T) if LOG_SCALE else np.corrcoef(K.T)
np.savetxt("FigS4_corr.txt", C, fmt="%8.4f",
           header="Pearson correlation, order: " + " ".join(LABELS))

# --- figure -----------------------------------------------------------------
# Full symmetric matrix, scatters in both triangles, as in the first version of
# this figure. The r values are not annotated in the panels: the text quotes the
# four that matter and FigS4_corr.txt carries the complete matrix, so printing
# all 56 of them on top of the scatters bought nothing.
p = len(LABELS)
SPAN = np.log10(K.max(axis=0) / K.min(axis=0))   # decades covered per parameter
# The figure prints at 0.9\textwidth = 6.30 in against a 7.5 in canvas, so text
# shrinks by 0.84 on the page: 8 pt here lands at 6.7 pt, clear of the ~6 pt floor
# that 7 pt (5.9 pt printed) sat just under. bottom=0.140 keeps the taller rotated
# x labels off the canvas edge.
# The figure is placed at 0.9\textwidth, i.e. reduced to about 0.8x. Keeping the
# canvas near its printed size is what keeps the tick labels above the ~6 pt
# floor; at the previous 12.5 in they landed near 3 pt on the page.
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.linewidth": 0.7})
fig, axes = plt.subplots(p, p, figsize=(7.5, 7.5))
# bottom holds the rotated tick labels, right holds row 0's y ticks
fig.subplots_adjust(wspace=0.10, hspace=0.10, left=0.105, right=0.945,
                    top=0.985, bottom=0.140)

BLUE, RED = "#2b6cb0", "#d1495b"


def _tick_label(v, _pos):
    """Compact tick text: plain digits nearby, mantissa x power far out.

    In absolute units the parameters range from 1e-5 (K11, K15) to 1e5 (K23),
    and a plain ScalarFormatter would write the latter as 60000/70000.
    """
    if v == 0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    if -3 <= e < 4:
        return f"{v:g}"
    m = v / 10.0 ** e
    return (rf"$10^{{{e}}}$" if abs(m - 1) < 1e-9
            else rf"${m:g}\times10^{{{e}}}$")


def set_log_ticks(axis, lo, hi, span):
    """Three labelled ticks, no labelled minors.

    A plain LogLocator leaves the tightly-determined parameters (well under a
    decade) with no labelled tick at all; the previous fix was to label the
    minor ticks, which is what produced the dense rotated number strings. Place
    three round values across the range instead.
    """
    if span >= 1.0:
        axis.set_major_locator(matplotlib.ticker.LogLocator(numticks=3))
        axis.set_minor_locator(
            matplotlib.ticker.LogLocator(subs="all", numticks=12))
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    else:
        # Under a decade the log axis is near enough to linear that a linear
        # locator gives round, evenly spaced values; geometric spacing rounded
        # to two digits collides (0.51 / 0.52 sit on top of each other).
        axis.set_major_locator(matplotlib.ticker.MaxNLocator(
            nbins=2, steps=[1, 2, 2.5, 5, 10]))
        axis.set_minor_locator(matplotlib.ticker.NullLocator())
        axis.set_major_formatter(matplotlib.ticker.FuncFormatter(_tick_label))


for i in range(p):
    for j in range(p):
        ax = axes[i, j]

        if i == j:
            bins = (np.logspace(np.log10(K[:, j].min()),
                                np.log10(K[:, j].max()),
                                max(9, int(np.sqrt(n)) + 1))
                    if LOG_SCALE else max(8, int(np.sqrt(n))))
            ax.hist(K[:, j], bins=bins, color=BLUE,
                    alpha=0.75, edgecolor="white", linewidth=0.4)
            ax.axvline(TRUE[j], color=RED, lw=1.6)
            ax.set_yticks([])
            if LOG_SCALE:
                ax.set_xscale("log")
        else:
            ax.scatter(K[:, j], K[:, i], s=6, color=BLUE, alpha=0.55,
                       edgecolors="none")
            if LOG_SCALE:
                ax.set_xscale("log"); ax.set_yscale("log")
            ax.axvline(TRUE[j], color=RED, lw=0.8, alpha=0.6)
            ax.axhline(TRUE[i], color=RED, lw=0.8, alpha=0.6)

        if i == p - 1:
            ax.set_xlabel(LABELS[j], fontsize=10, labelpad=2)
        if j == 0 and i > 0:
            ax.set_ylabel(LABELS[i], fontsize=10, labelpad=2)
        if LOG_SCALE:
            set_log_ticks(ax.xaxis, *ax.get_xlim(), SPAN[j])
            if i != j:
                set_log_ticks(ax.yaxis, *ax.get_ylim(), SPAN[i])
        else:
            ax.locator_params(nbins=3)
        # Hide interior tick labels. Must come AFTER the locators above, which
        # would otherwise switch them back on.
        ax.tick_params(which="both",
                       labelbottom=(i == p - 1),
                       labelleft=(j == 0 and i > 0),
                       labelsize=8, length=2.2, pad=1.2)
        if i == p - 1:
            # In absolute units the labels are wide ($6\times10^4$, 0.0065) and
            # three of them do not fit across a panel this narrow. Rotating is
            # what buys the width; the y labels are unaffected and stay level.
            for lab in ax.get_xticklabels():
                lab.set_rotation(45)
                lab.set_ha("right")
                lab.set_rotation_mode("anchor")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

# Every other row is named by the y-label of its left-most panel, but row 0's
# left-most panel is the K8 histogram, which the loop skips because its vertical
# axis is a count. Name it anyway, or K8 is the one parameter missing from the
# left-hand edge. The count axis stays unticked; K8's own scale is read from the
# foot of column 0, where every column's is.
axes[0, 0].set_ylabel(LABELS[0], fontsize=10, labelpad=2)

# Row 0 does have K8 on the y-axis of its seven scatters, but its left-most
# panel is the histogram, so the row is the one that ends up with no numeric
# scale. Hang its tick labels on the right-hand edge instead.
ax0 = axes[0, p - 1]
ax0.yaxis.set_tick_params(which="both", labelright=True, labelleft=False,
                          labelsize=8, length=2.2, pad=1.2)
ax0.spines["right"].set_visible(True)

# Tick labels differ in width ("0.03" vs "$7\times10^4$") and, once rotated, in
# height, so matplotlib parks each axis label at a different offset. Pull the
# K8...K23 labels onto a common line on each edge.
fig.align_ylabels(axes[:, 0])
fig.align_xlabels(axes[p - 1, :])

fig.savefig("FigS4.png", dpi=600, facecolor="white")
fig.savefig("FigS4.pdf", facecolor="white")
print("wrote FigS4.png / FigS4.pdf / FigS4_corr.txt")

strong = [(LABELS[i], LABELS[j], C[i, j])
          for i in range(p) for j in range(i + 1, p) if abs(C[i, j]) > 0.5]
if strong:
    print("\nstrongly correlated pairs (|r| > 0.5) - these are the "
          "non-identifiable directions:")
    for a, b, r in sorted(strong, key=lambda z: -abs(z[2])):
        print(f"  {a:>5} - {b:<5} r = {r:+.3f}")
else:
    print("\nno pair exceeds |r| = 0.5")
