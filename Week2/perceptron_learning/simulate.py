"""
Interactive Perceptron Learning Algorithm Simulation
-----------------------------------------------------
Controls:
  [Next Step]  — apply one weight update
  [Run All]    — run all remaining updates to convergence
  [Reset]      — restart from w = 0

The plot shows:
  • Blue circles  : positive class (+1)
  • Red  squares  : negative class (-1)
  • Orange star   : currently misclassified point being corrected
  • Dashed line   : decision boundary BEFORE update
  • Solid line    : decision boundary AFTER update (current w)
  • Info panel    : epoch, step, w vector
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from data import make_linearly_separable
from perceptron import perceptron_train, predict

# ── Data ────────────────────────────────────────────────────────────────────
X_aug, y, X_raw = make_linearly_separable()
N = len(y)
pos_mask = y == 1
neg_mask = y == -1

# Pre-compute full history
_, full_history = perceptron_train(X_aug, y)

# ── State ────────────────────────────────────────────────────────────────────
state = {"step": 0, "w": np.zeros(3)}  # step index into full_history


def w_to_boundary(w, x_range):
    """Convert weight vector to decision boundary x2 values for plotting."""
    w0, w1, w2 = w
    if abs(w2) < 1e-9:
        return None, None
    x2 = -(w0 + w1 * x_range) / w2
    return x_range, x2


# ── Figure setup ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
plt.subplots_adjust(bottom=0.22, right=0.75)

margin = 0.5
x_lim = (X_raw[:, 0].min() - margin, X_raw[:, 0].max() + margin)
x_range = np.linspace(x_lim[0], x_lim[1], 300)
y_lim = (X_raw[:, 1].min() - margin, X_raw[:, 1].max() + margin)

# Scatter
scat_pos = ax.scatter(X_raw[pos_mask, 0], X_raw[pos_mask, 1],
                      c="steelblue", marker="o", s=80, zorder=3, label="+1 (positive)")
scat_neg = ax.scatter(X_raw[neg_mask, 0], X_raw[neg_mask, 1],
                      c="tomato", marker="s", s=80, zorder=3, label="-1 (negative)")
highlight, = ax.plot([], [], "*", ms=18, color="orange", zorder=5, label="current point")
bound_after,  = ax.plot([], [], "-",  lw=2.5, color="green",  label="boundary (after)")
bound_before, = ax.plot([], [], "--", lw=1.5, color="gray",   label="boundary (before)", alpha=0.6)

ax.set_xlim(*x_lim)
ax.set_ylim(*y_lim)
ax.axhline(0, color="k", lw=0.4, alpha=0.3)
ax.axvline(0, color="k", lw=0.4, alpha=0.3)
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.legend(loc="upper left", fontsize=8)
ax.set_title("Perceptron Learning Algorithm — Interactive Simulation", fontsize=11)

# Info panel (right side)
info_ax = fig.add_axes([0.76, 0.25, 0.22, 0.65])
info_ax.axis("off")
info_text = info_ax.text(0.05, 0.95, "", transform=info_ax.transAxes,
                         fontsize=9, va="top", family="monospace",
                         bbox=dict(boxstyle="round", fc="lightyellow", ec="gray"))


def build_info(step_idx, w, h=None):
    total = len(full_history)
    lines = [
        f"Total updates: {total}",
        f"Current step : {step_idx}/{total}",
        "─" * 22,
    ]
    if h:
        lines += [
            f"Epoch        : {h['epoch']}",
            f"Point idx    : {h['point_idx']}",
            f"Label (y)    : {h['y']:+d}",
            "─" * 22,
            "w_before:",
            f"  bias={h['w_before'][0]:+.3f}",
            f"  w1  ={h['w_before'][1]:+.3f}",
            f"  w2  ={h['w_before'][2]:+.3f}",
            "w_after:",
            f"  bias={h['w_after'][0]:+.3f}",
            f"  w1  ={h['w_after'][1]:+.3f}",
            f"  w2  ={h['w_after'][2]:+.3f}",
        ]
    else:
        lines += [
            "w (current):",
            f"  bias={w[0]:+.3f}",
            f"  w1  ={w[1]:+.3f}",
            f"  w2  ={w[2]:+.3f}",
        ]
    if step_idx == total:
        lines += ["─" * 22, "✓ CONVERGED"]
    return "\n".join(lines)


def redraw(h=None):
    w = state["w"]
    step_idx = state["step"]

    # Decision boundary after update (current w)
    xb, yb = w_to_boundary(w, x_range)
    if xb is not None:
        bound_after.set_data(xb, yb)
    else:
        bound_after.set_data([], [])

    # Decision boundary before update (dashed)
    if h is not None:
        xb2, yb2 = w_to_boundary(h["w_before"], x_range)
        if xb2 is not None:
            bound_before.set_data(xb2, yb2)
        else:
            bound_before.set_data([], [])
        # Highlight the corrected point
        pt = X_raw[h["point_idx"]]
        highlight.set_data([pt[0]], [pt[1]])
    else:
        bound_before.set_data([], [])
        highlight.set_data([], [])

    info_text.set_text(build_info(step_idx, w, h))
    fig.canvas.draw_idle()


# Initial draw
redraw()


# ── Button callbacks ──────────────────────────────────────────────────────────
def next_step(event):
    idx = state["step"]
    if idx >= len(full_history):
        return
    h = full_history[idx]
    state["w"] = h["w_after"].copy()
    state["step"] += 1
    redraw(h)


def run_all(event):
    remaining = full_history[state["step"]:]
    if not remaining:
        return
    last_h = remaining[-1]
    state["w"] = last_h["w_after"].copy()
    state["step"] = len(full_history)
    redraw(last_h)


def reset(event):
    state["step"] = 0
    state["w"] = np.zeros(3)
    redraw()


ax_next  = fig.add_axes([0.10, 0.08, 0.22, 0.07])
ax_run   = fig.add_axes([0.38, 0.08, 0.22, 0.07])
ax_reset = fig.add_axes([0.66, 0.08, 0.22, 0.07])

btn_next  = Button(ax_next,  "Next Step",  color="lightcyan",   hovercolor="cyan")
btn_run   = Button(ax_run,   "Run All",    color="lightgreen",  hovercolor="lime")
btn_reset = Button(ax_reset, "Reset",      color="lightyellow", hovercolor="yellow")

btn_next.on_clicked(next_step)
btn_run.on_clicked(run_all)
btn_reset.on_clicked(reset)

plt.show()
