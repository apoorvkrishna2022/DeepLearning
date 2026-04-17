"""
Perceptron Learning Algorithm (as described in CS7015 Lec 2.5)

Setup:
  - Labels: +1 (positive) and -1 (negative)
  - Input x is augmented with x0=1 for the bias weight w0
  - w initialized to zeros (as instructed)

Update rule (equivalent to lecture's formulation):
  - If y * (w · x) <= 0  →  misclassified  →  w = w + y * x
  - This handles both cases from the lecture:
      * x in P and w·x < 0  → w = w + x   (y=+1)
      * x in N and w·x >= 0 → w = w - x   (y=-1)

Convergence: one full pass over data with zero misclassifications.
"""

import numpy as np
from data import make_linearly_separable


def perceptron_train(X_aug, y, max_epochs=1000):
    """
    Train perceptron. Returns final w and history of (w, mistake_index) per update.
    history entry: dict with w_before, w_after, point_index, epoch, step
    """
    N, D = X_aug.shape
    w = np.zeros(D)       # initialize w to zero (bias + 2 weights)
    history = []
    step = 0

    for epoch in range(1, max_epochs + 1):
        mistakes = 0
        for i in range(N):
            x_i, y_i = X_aug[i], y[i]
            if y_i * np.dot(w, x_i) <= 0:          # misclassified
                w_before = w.copy()
                w = w + y_i * x_i
                step += 1
                mistakes += 1
                history.append({
                    "step": step,
                    "epoch": epoch,
                    "point_idx": i,
                    "y": y_i,
                    "w_before": w_before,
                    "w_after": w.copy(),
                })
        if mistakes == 0:
            print(f"Converged in {epoch} epoch(s), {step} total weight update(s).")
            break
    else:
        print(f"Did not converge within {max_epochs} epochs.")

    return w, history


def predict(w, X_aug):
    return np.sign(X_aug @ w)


if __name__ == "__main__":
    X_aug, y, X_raw = make_linearly_separable()

    print(f"Dataset: {len(y)} points  |  pos={sum(y==1)}  neg={sum(y==-1)}")
    print(f"w initialized to: {np.zeros(X_aug.shape[1])}\n")

    w, history = perceptron_train(X_aug, y)

    print(f"\nFinal w (bias, w1, w2): {w}")
    preds = predict(w, X_aug)
    acc = np.mean(preds == y) * 100
    print(f"Training accuracy: {acc:.1f}%")

    print("\n--- Update History ---")
    for h in history:
        print(
            f"Step {h['step']:2d} | Epoch {h['epoch']} | "
            f"Point {h['point_idx']:2d} (y={h['y']:+d}) | "
            f"w: {np.round(h['w_before'], 2)} → {np.round(h['w_after'], 2)}"
        )
