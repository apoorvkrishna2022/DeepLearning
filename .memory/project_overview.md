---
name: Project Overview
description: Full context for the DeepLearning repo — structure, completed work, tools used
type: project
---

This is a deep learning self-study project at `/Users/apoorvkrishna/Desktop/My_Repos/DeepLearning`, organized by week. The course is **CS7015 (IIT Madras, NPTEL)** taught on YouTube.

**Playlist:** https://www.youtube.com/playlist?list=PLyqSpQzTE6M9gCgajvQbc68Hk_JKGBAYT

## Completed Work

### Week2/perceptron_learning/
Built from scratch based on Lec 2.5 and 2.6 transcripts.

**Files:**
- `data.py` — generates 54 linearly separable 2D points (4 sub-clusters per class, all in first quadrant, true boundary: x2 = 0.9·x1 − 1.5)
- `perceptron.py` — core PLA with w initialized to zeros, logs full update history; converges in 33 updates / 12 epochs
- `simulate.py` — interactive matplotlib visualization with Next Step / Run All / Reset buttons; shows decision boundary shifting per update, highlights misclassified point in orange

**Run:**
```bash
cd Week2/perceptron_learning
uv run python perceptron.py    # algorithm output
uv run python simulate.py      # interactive sim
```

**Package manager:** `uv` (pyproject.toml present, deps: numpy, matplotlib)

## Concepts Covered
- Perceptron Learning Algorithm (Lec 2.5): update rule w = w + y·x on misclassification
- Proof of Convergence (Lec 2.6): finite updates bounded by k ≤ 1/δ², only holds for linearly separable data
- Epoch = one full pass over all training data
- Non-linearly separable data → PLA never converges → motivation for multi-layer networks

**Why:** Learning deep learning week by week, building code alongside lectures.
**How to apply:** Follow week-based structure, always tie code to the specific lecture concepts.
