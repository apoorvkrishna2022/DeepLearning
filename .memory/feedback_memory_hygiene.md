---
name: Memory Update Discipline
description: Claude must keep memory up to date after every session — new work, decisions, and learnings must be persisted
type: feedback
---

Always update memory at the end of a session or whenever meaningful work is done. Do not let memory go stale.

**Why:** User explicitly requires memory to be maintained properly so future conversations have full context without re-explanation.

**How to apply:**
- After completing any coding task → update `project_overview.md` with what was built, file paths, how to run
- After any new concept is covered → note it under concepts in the project memory
- After user gives behavioral feedback → save a `feedback_*.md` entry immediately
- After any structural/architectural decision → record it so it isn't re-debated next session
- At end of session → do a quick check: is anything new that happened this session missing from memory?
- Always update `MEMORY.md` index when adding or renaming a memory file
- Never leave memory in an outdated state (e.g. saying "Week2 is empty" when it now has files)
