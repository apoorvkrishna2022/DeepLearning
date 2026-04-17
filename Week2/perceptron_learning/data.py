import numpy as np


# True decision boundary: x2 = 0.9*x1 - 1.5  (all in first quadrant)
# Equivalently: -1.5 - 0.9*x1 + x2 = 0  →  w* = [-1.5, -0.9, 1.0]
# Positive class: above the line (x2 > 0.9*x1 - 1.5)
# Negative class: below the line

def _above_boundary(x1, x2):
    return x2 > (0.9 * x1 - 1.5)


def make_linearly_separable(seed=42):
    """
    Generate challenging but linearly separable 2D data entirely in the
    first quadrant (x1 > 0, x2 > 0).

    True decision boundary: x2 = 0.9*x1 - 1.5  (diagonal cut)

    Complexity added:
      - 4 sub-clusters per class spread across the quadrant
      - Points intentionally close to the boundary (margin ~0.4)
      - Mix of tight and spread clusters
      - 50 points total

    Labels: +1 (above boundary), -1 (below boundary).
    Returns X_aug (bias prepended), y, X_raw (for plotting).
    """
    rng = np.random.default_rng(seed)

    # --- Positive sub-clusters (above boundary) ---
    # Each: (center_x1, center_x2, std, n_points)
    pos_specs = [
        (2.0, 5.5, 0.35, 8),   # top-left region
        (5.0, 6.5, 0.50, 7),   # top-right region
        (7.5, 5.0, 0.40, 6),   # far right, high
        (3.5, 3.8, 0.30, 6),   # close to boundary, upper side
    ]

    # --- Negative sub-clusters (below boundary) ---
    neg_specs = [
        (1.5, 0.8, 0.35, 8),   # bottom-left
        (4.5, 2.0, 0.50, 7),   # middle-bottom
        (7.0, 2.5, 0.40, 6),   # bottom-right
        (5.5, 3.5, 0.30, 6),   # close to boundary, lower side
    ]

    pos_points, neg_points = [], []

    for (cx, cy, std, n) in pos_specs:
        for _ in range(n * 10):          # oversample, keep valid ones
            p = rng.normal([cx, cy], std)
            if p[0] > 0.2 and p[1] > 0.2 and _above_boundary(*p):
                pos_points.append(p)
                if len(pos_points) == sum(s[3] for s in pos_specs[:pos_specs.index((cx,cy,std,n))+1]):
                    break

    # simpler: just generate and filter
    pos_points = []
    for (cx, cy, std, n) in pos_specs:
        collected = 0
        attempts = 0
        while collected < n and attempts < 5000:
            p = rng.normal([cx, cy], std)
            if p[0] > 0.1 and p[1] > 0.1 and _above_boundary(*p):
                pos_points.append(p)
                collected += 1
            attempts += 1

    neg_points = []
    for (cx, cy, std, n) in neg_specs:
        collected = 0
        attempts = 0
        while collected < n and attempts < 5000:
            p = rng.normal([cx, cy], std)
            if p[0] > 0.1 and p[1] > 0.1 and not _above_boundary(*p):
                neg_points.append(p)
                collected += 1
            attempts += 1

    pos = np.array(pos_points)
    neg = np.array(neg_points)

    X = np.vstack([pos, neg])
    y = np.array([1] * len(pos) + [-1] * len(neg))

    X_aug = np.hstack([np.ones((len(X), 1)), X])

    return X_aug, y, X
