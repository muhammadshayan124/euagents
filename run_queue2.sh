#!/usr/bin/env bash
# Runs after the first queue has finished its last configuration (waits for 30 finished no_verifier tasks), then the directive baseline.
cd "$(dirname "$0")"
until [ "$(wc -l < results/no_verifier.jsonl 2>/dev/null || echo 0)" -ge 30 ]; do sleep 45; done
sleep 20   # let the first queue's final report finish and release the model
echo "=== single_agent_directive $(date +%H:%M:%S)"
.venv/Scripts/euagents eval single_agent_directive --n 30
.venv/Scripts/euagents report
echo "=== all done $(date +%H:%M:%S)"
