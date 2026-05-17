#!/bin/sh
set -eu

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <seconds> <ish> [args...]" >&2
    exit 2
fi

timeout_secs=$1
shift

ish_pid=
timed_out=0

cleanup() {
    if [ -n "${watchdog_pid:-}" ]; then
        kill "$watchdog_pid" 2>/dev/null || true
        wait "$watchdog_pid" 2>/dev/null || true
    fi
    if [ -n "${ish_pid}" ]; then
        kill "$ish_pid" 2>/dev/null || true
        wait "$ish_pid" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

"$@" &
ish_pid=$!

(
    sleep "$timeout_secs"
    timed_out=1
    kill -TERM "$ish_pid" 2>/dev/null || true
    sleep 1
    kill -KILL "$ish_pid" 2>/dev/null || true
) &
watchdog_pid=$!

wait "$ish_pid"
status=$?

kill "$watchdog_pid" 2>/dev/null || true
wait "$watchdog_pid" 2>/dev/null || true
watchdog_pid=
ish_pid=

if [ "$timed_out" -ne 0 ]; then
    echo "run_ish_timeout: timed out after ${timeout_secs}s" >&2
    exit 124
fi

exit "$status"
