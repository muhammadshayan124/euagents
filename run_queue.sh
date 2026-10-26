#!/usr/bin/env bash
# Sequential evaluation queue: one configuration at a time so the machine is never overloaded.
# Resumable: finished tasks are read back from results/<config>.jsonl, so re-running continues.
cd "$(dirname "$0")"
for cfg in full single_agent no_planner no_verifier; do
  echo "=== $cfg $(date +%H:%M:%S)"
  .venv/Scripts/euagents eval "$cfg" --n 30
done
echo "=== all done $(date +%H:%M:%S)"
.venv/Scripts/euagents report
