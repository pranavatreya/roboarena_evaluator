#!/bin/bash
# Convenience wrapper for evaluation_client/test_single_policy.py.
# Usage: ./run_single_policy.sh [config.yaml] <policy-host> [port]

CONFIG=${1:-configs/berkeley.yaml}
HOST=${2:?Pass your policy server hostname as the 2nd arg (no ws:// prefix)}
PORT=${3:-443}

python3 evaluation_client/test_single_policy.py "$CONFIG" \
    --host "$HOST" --port "$PORT"
