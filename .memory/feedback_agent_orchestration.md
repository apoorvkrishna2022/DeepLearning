---
name: Agent Orchestration Style
description: Main agent must always delegate tasks to sub-agents, never perform tasks itself
type: feedback
---

Always fire sub-agents to get the job done. The main agent's role is to **orchestrate and oversee** — not to execute tasks directly.

**Why:** User explicitly prefers this working style. The main agent should plan, delegate, and synthesize results — not run commands, write files, or do implementation work on its own.

**How to apply:**
- Writing code → fire a sub-agent
- Running shell commands (uv, git, etc.) → fire a sub-agent
- Fetching/researching content → fire a sub-agent
- Main agent only: reads results, makes decisions, coordinates next steps, communicates to user
