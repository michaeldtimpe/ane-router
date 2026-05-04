#!/usr/bin/env bash
# verify_ane.sh — the gate.
#
# Runs a tight-loop CoreML model (Linear+softmax) targeting CPU_AND_NE and
# samples powermetrics for ANE rail activity. Exits 0 if the rail showed
# non-zero power for >= MIN_HITS samples; exits non-zero otherwise.
#
# Requires sudo for powermetrics. Pre-authenticate with `sudo -v` first.
#
# Usage:
#   sudo -v
#   ./scripts/verify_ane.sh

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv-312/bin/python}"
HIDDEN="${HIDDEN:-4096}"
PULSE_SECONDS="${PULSE_SECONDS:-8}"
SAMPLE_INTERVAL_MS="${SAMPLE_INTERVAL_MS:-100}"
SAMPLE_COUNT="${SAMPLE_COUNT:-50}"   # 50 * 100ms = 5s of samples
MIN_HITS="${MIN_HITS:-5}"

echo "--- ANE rail verification ---"
echo "  python:            $PYTHON"
echo "  hidden:            $HIDDEN"
echo "  pulse_seconds:     $PULSE_SECONDS"
echo "  sample interval:   ${SAMPLE_INTERVAL_MS}ms"
echo "  sample count:      $SAMPLE_COUNT"
echo "  min hits required: $MIN_HITS"

# Test the specific sudo command we'll use; the passwordless rule may be
# scoped to powermetrics only.
if ! sudo -n /usr/bin/powermetrics --samplers ane_power -i 100 -n 1 >/dev/null 2>&1; then
    echo "ERROR: cannot run 'sudo -n powermetrics' without password." >&2
    echo "Either run 'sudo -v' first, or grant passwordless access to" >&2
    echo "  /usr/bin/powermetrics in /etc/sudoers.d/." >&2
    exit 2
fi

# Start the pulse in the background. It will exit on its own after PULSE_SECONDS.
"$PYTHON" -m bench._ane_pulse "$HIDDEN" "$PULSE_SECONDS" &
PULSE_PID=$!

# Give the pulse a moment to load the model and reach steady state.
sleep 1

# Sample powermetrics. On macOS 26.x, the ANE Power line is only emitted
# when cpu_power+gpu_power+ane_power are all enabled (subset doesn't work).
PM_OUTPUT="$(sudo -n powermetrics --samplers cpu_power,gpu_power,ane_power \
                -i "$SAMPLE_INTERVAL_MS" \
                -n "$SAMPLE_COUNT" 2>/dev/null || true)"

# Wait for pulse to finish.
wait "$PULSE_PID" 2>/dev/null || true

# Parse: count ANE Power lines with value > 0.
# powermetrics text output line: "ANE Power: <number> mW"
HITS=$(echo "$PM_OUTPUT" \
       | grep -E "^ANE Power: [0-9]+(\.[0-9]+)? mW" \
       | awk '{print $3}' \
       | awk '$1+0 > 0 {n++} END {print n+0}')

TOTAL=$(echo "$PM_OUTPUT" | grep -cE "^ANE Power: [0-9]+(\.[0-9]+)? mW" || true)

echo
echo "  ANE Power lines:    $TOTAL"
echo "  non-zero samples:   $HITS"

if [[ "$HITS" -ge "$MIN_HITS" ]]; then
    echo "PASS: ANE rail showed activity ($HITS / $TOTAL samples > 0 mW)."
    exit 0
else
    echo "FAIL: ANE rail stayed cold ($HITS < $MIN_HITS hits). Aborting." >&2
    echo "Possible causes: model fell back to CPU/GPU, ANE driver issue." >&2
    exit 1
fi
