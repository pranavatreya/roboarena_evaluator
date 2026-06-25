#!/bin/bash
# Convenience wrapper for evaluation_client/main.py (the central A/B eval).
#
# Flags used below:
#   -n / --num-runs N    Run N back-to-back A/B sessions in one process.
#                        DROID init + camera-check happen ONCE; sessions 2..N
#                        skip both and jump straight to the rollouts. Big win
#                        when an evaluator wants to grind through many sessions.
#
#   --prompt "..."       Optional. Pre-fills the language command sent to both
#                        policies, so the script doesn't pause to ask each time.
#                        Omit to be asked interactively per session (the default
#                        and what you usually want when tasks vary).
#
# Usage:
#   ./run_roboarena.sh                              # single session, default config
#   ./run_roboarena.sh configs/my_institution.yaml  # single session, custom config
#   ./run_roboarena.sh configs/my_institution.yaml 10                   # 10 sessions, each asks for a prompt
#   ./run_roboarena.sh configs/my_institution.yaml 10 "pick up the red block"  # 10 sessions, same prompt

CONFIG=${1:-configs/berkeley.yaml}
NUM_RUNS=${2:-1}
PROMPT=${3:-}

ARGS=("$CONFIG" --num-runs "$NUM_RUNS")
if [[ -n "$PROMPT" ]]; then
    ARGS+=(--prompt "$PROMPT")
fi

python3 evaluation_client/main.py "${ARGS[@]}"
