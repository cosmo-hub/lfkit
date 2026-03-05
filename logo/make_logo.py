"""Script to generate the LFKit logo."""

from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Colors
cmap = "cmr.guppy"
blue = cmr.take_cmap_colors(cmap, 3, cmap_range=(0.8, 1.0), return_fmt="hex")[1]
red = cmr.take_cmap_colors(cmap, 3, cmap_range=(0.0, 0.2), return_fmt="hex")[1]

# Figure setup
fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
ax.set_aspect("equal")
ax.axis("off")

# Geometry params (data units)
stroke = 50
L_height = 1.5
L_base = 0.8
phi_radius = 0.55

# --- Convert half linewidth to data units (for padding + centering) ---
fig.canvas.draw()
lw_px = (stroke * fig.dpi) / 72.0  # points -> pixels
p0 = ax.transData.transform((0.0, 0.0))
dx_data = ax.transData.inverted().transform((p0[0] + 0.5 * lw_px, p0[1]))[0] - 0.0
dy_data = ax.transData.inverted().transform((p0[0], p0[1] + 0.5 * lw_px))[1] - 0.0
half_lw_data = float(max(abs(dx_data), abs(dy_data)))

# Base uses projecting => extends downward by ~half linewidth
bottom_extension = half_lw_data

# Center ring on *visible* vertical span: y in [-bottom_extension, L_height]
phi_center = (0.0, (L_height - bottom_extension) / 2.0)
xc, yc = phi_center
r = phi_radius

# Small angular overlap to hide seams where arcs meet (degrees)
# Increase to 8–12 if you still see a join.
eps_deg = 8.0

# ---------------------------------------------------
# Draw order (THIS is the point):
# 1) bottom arc (behind L)
# 2) full L (in front of bottom arc)
# 3) top arc (in front of L vertical)
# ---------------------------------------------------

# 1) Bottom arc BEHIND the L  (angles: 180 -> 360)
bottom_arc = patches.Arc(
    phi_center,
    width=2 * r,
    height=2 * r,
    angle=0.0,
    theta1=180.0 - eps_deg,
    theta2=360.0 + eps_deg,
    linewidth=stroke,
    color=red,
    zorder=3,
    capstyle="butt",
)
ax.add_patch(bottom_arc)

# 2) Full L IN FRONT (vertical + base)
ax.plot(
    [0, 0], [0, L_height],
    color=blue, linewidth=stroke,
    solid_capstyle="butt",
    zorder=5,
)

ax.plot(
    [0, L_base], [0, 0],
    color=blue, linewidth=stroke,
    solid_capstyle="projecting",
    zorder=5,
)

# 3) Top arc IN FRONT of the vertical (angles: 0 -> 180)
top_arc = patches.Arc(
    phi_center,
    width=2 * r,
    height=2 * r,
    angle=0.0,
    theta1=0.0 - eps_deg,
    theta2=180.0 + eps_deg,
    linewidth=stroke,
    color=red,
    zorder=7,
    capstyle="butt",
)
ax.add_patch(top_arc)

# ---------------------------------------------------
# Robust limits (avoid cropping, include projecting caps)
# ---------------------------------------------------
pad = 2.4 * half_lw_data  # safety

x_left_cap = -half_lw_data
x_right_cap = L_base + half_lw_data

x_min = min(x_left_cap, xc - r) - pad
x_max = max(x_right_cap, xc + r) + pad

y_min = min(-bottom_extension, yc - r) - pad
y_max = max(L_height, yc + r) + pad

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
fig.tight_layout(pad=0)

# Paths
script_dir = Path(__file__).resolve().parent
docs_static = script_dir.parent / "docs" / "_static" / "logos"
for d in (script_dir, docs_static):
    d.mkdir(parents=True, exist_ok=True)

print("Saving to:", script_dir)
print("Saving to:", docs_static)


def save_all(out_dir: Path):
    fig.savefig(out_dir / "lfkit_logo.svg", transparent=True)
    fig.savefig(out_dir / "lfkit_logo.pdf", transparent=True)
    fig.savefig(out_dir / "lfkit_logo.png", transparent=True, dpi=300)

    # icon: more margin
    ax.set_xlim(x_min - 0.6 * pad, x_max + 0.6 * pad)
    ax.set_ylim(y_min - 0.6 * pad, y_max + 0.6 * pad)
    fig.savefig(out_dir / "lfkit_logo-icon.png", transparent=True, dpi=300)


save_all(script_dir)
save_all(docs_static)

plt.close(fig)
print("Done.")