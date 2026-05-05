# Experiment Execution Policy - 2026-05-04

User instruction:

- Do not run experiments locally.
- Allowed experiment routes:
  1. Kaggle MCP directly in Kaggle, preferred.
  2. Write code and commands for the user to run on the cloud server, then analyze returned logs/results.

Current session status:

- Kaggle MCP resources/templates are not available in this Codex session.
- Therefore, use route 2 until Kaggle MCP becomes available.

Operational rule:

- Local workspace may be used for code editing, notes, and static file inspection.
- Do not run local dry-runs, training, inference, or leaderboard simulation.
- Any command that consumes audio/model data to produce predictions should be given to the user to run remotely.
