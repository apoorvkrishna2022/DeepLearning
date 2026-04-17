# DeepLearning Project — Claude Instructions

## Project
Self-study of deep learning using the CS7015 course by IIT Madras (NPTEL).
YouTube playlist: https://www.youtube.com/playlist?list=PLyqSpQzTE6M9gCgajvQbc68Hk_JKGBAYT
Structure: organized by week (Week2/, Week3/, ...). Code is built alongside lectures.
Package manager: `uv` (every sub-project has its own pyproject.toml).

## Completed Work
- **Week2/perceptron_learning/** — Perceptron Learning Algorithm (Lec 2.5 & 2.6)
  - `data.py` — 54 linearly separable 2D points, first quadrant, 4 sub-clusters per class
  - `perceptron.py` — core PLA, w initialized to zeros, full update history
  - `simulate.py` — interactive matplotlib simulation (Next Step / Run All / Reset)
  - Run: `uv run python perceptron.py` or `uv run python simulate.py`

## Behavioral Instructions (must follow every session)

### 1. Sub-agent orchestration
**Always fire sub-agents to get tasks done. Never perform tasks directly.**
- Writing code → fire a sub-agent
- Running shell commands → fire a sub-agent
- Researching/fetching content → fire a sub-agent
- Main agent role: plan, delegate, synthesize, communicate

### 2. Memory hygiene
**Always update memory after meaningful work.**
- After completing any task → update `project_overview.md` with what was built
- After user gives feedback → save a `feedback_*.md` entry immediately
- After any new concept is covered → add it to the concepts list in project memory
- At end of session → check: is anything new missing from memory?
- Always keep `MEMORY.md` index in sync when adding/updating memory files

### 3. Memory location
All memory files live at:
`~/.claude/projects/-Users-apoorvkrishna-Desktop-My-Repos-DeepLearning/memory/`
Read `MEMORY.md` there at the start of each session for full context.
