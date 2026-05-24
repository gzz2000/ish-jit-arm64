# AGENTS.md

This file is the working operator guide for this repo. Keep it current as the
actual build, run, dump, and verifier workflows evolve.

## Scope

These notes are for the ARM64 JIT work under:

- [/Users/zizhengguo/scratch/ish-arm64/jit/guest-arm64](/Users/zizhengguo/scratch/ish-arm64/jit/guest-arm64)
- [/Users/zizhengguo/scratch/ish-arm64/build-arm64-release](/Users/zizhengguo/scratch/ish-arm64/build-arm64-release)

## Non-Negotiable Process Rules

- Always use a bounded timeout for every `ish` run. Never launch unbounded guest
  runs during development. Hung guest runs will accumulate and clog the host.
- Do not treat `exit 0` by itself as proof that a run succeeded. For small
  smoke tests, explicitly inspect stdout.
- Prefer small repros first:
  - `/bin/echo hello`
  - `/root/cbench_lite_arm64`
- Use verifier aggressively to debug correctness. Do not roll back a feature
  just because it is broken; instead, find the first mismatch frontier and fix
  it.
- When changing helper ABI or memory-helper shape, update this file and
  [/Users/zizhengguo/scratch/ish-arm64/jit/guest-arm64/helpers.S](/Users/zizhengguo/scratch/ish-arm64/jit/guest-arm64/helpers.S)
  comments together.

## Build

The repo root does not have a top-level makefile. Use Meson/Ninja for the
standalone ARM64 host binary.

### Configure

```bash
meson setup build-arm64-release -Dguest_arch=arm64 --buildtype=release
```

If the build directory is already configured, do not rerun setup unless Meson
options changed.

### Build

```bash
ninja -C build-arm64-release
```

If the build directory already exists and only code changed, usually this is
enough:

```bash
ninja -C build-arm64-release
```

## Actual Build Commands Used In Current Work

Use these exact commands during JIT development:

### Rebuild after source edits

```bash
ninja -C build-arm64-release
```

### First-time setup, only if `build-arm64-release` does not exist yet

```bash
meson setup build-arm64-release -Dguest_arch=arm64 --buildtype=release
ninja -C build-arm64-release
```

Do not use a top-level `make` in repo root. It will fail because there is no
root makefile.
Do not use `make` inside `build-arm64-release` either; this build directory is
managed by Ninja, so use `ninja -C build-arm64-release`.

## Fakefs / Rootfs

The common rootfs used in local runs is:

- [/Users/zizhengguo/scratch/ish-arm64/build/alpine-arm64-fakefs](/Users/zizhengguo/scratch/ish-arm64/build/alpine-arm64-fakefs)

Interactive shell:

```bash
./build-arm64-release/ish -f ./build/alpine-arm64-fakefs /bin/sh
```

## Bounded Run Pattern

Always run guest programs with a timeout.

### Plain JIT smoke

```bash
timeout 20s env ISH_ARM64_BACKEND=arm64_jit \
  ./build-arm64-release/ish -f ./build/alpine-arm64-fakefs /bin/echo hello \
  > /private/tmp/echo.stdout 2> /private/tmp/echo.stderr
```

Success criterion for this smoke is:

- process exits before timeout
- `/private/tmp/echo.stdout` contains exactly `hello`

`exit 0` alone is not enough.

### Plain JIT benchmark

```bash
timeout 30s env ISH_ARM64_BACKEND=arm64_jit \
  ./build-arm64-release/ish -f ./build/alpine-arm64-fakefs /root/cbench_lite_arm64
```

### Redirect capture

```bash
timeout 30s env ISH_ARM64_BACKEND=arm64_jit \
  ./build-arm64-release/ish -f ./build/alpine-arm64-fakefs /root/cbench_lite_arm64 \
  > /private/tmp/cbench.stdout 2> /private/tmp/cbench.stderr
```

If redirected output disagrees with tty output, suspect guest stdio/path
compatibility rather than assuming a JIT semantic bug immediately.

## Verifier Launch

Verifier is for correctness, not throughput. Keep the target small first.

### Small verifier repro

```bash
timeout 30s env ISH_ARM64_BACKEND=arm64_jit ISH_ARM64_JIT_VERIFY=1 \
  ./build-arm64-release/ish -f ./build/alpine-arm64-fakefs /bin/echo hello \
  > /private/tmp/echo_verify.stdout 2> /private/tmp/echo_verify.stderr
```

### Benchmark verifier repro

```bash
timeout 20s env ISH_ARM64_BACKEND=arm64_jit ISH_ARM64_JIT_VERIFY=1 \
  ./build-arm64-release/ish -f ./build/alpine-arm64-fakefs /root/cbench_lite_arm64 \
  > /private/tmp/cbench_verify.stdout 2> /private/tmp/cbench_verify.stderr
```

### Verifier logging notes

- Verifier step logs include timestamps and deltas.
- Every verifier log line should carry a wall-time prefix. Use that to tell
  the difference between “actively making progress” and “reached the last
  visible frontier immediately, then stopped”.
- When a run appears “stuck”, check whether the last step was reached almost
  immediately and then stopped progressing.
- The first mismatch frontier is the only thing that matters; do not chase
  downstream symptoms first.
- For `/bin/echo hello`, “verifier cleared” means both:
  - no `mismatch at` line in stderr
  - stdout actually contains `hello`

## Trace Launch

Use trace only with timeout and only when narrowing a specific region.

### Plain JIT trace

```bash
timeout 20s env ISH_ARM64_BACKEND=arm64_jit ISH_ARM64_JIT_TRACE=1 \
  ./build-arm64-release/ish -f ./build/alpine-arm64-fakefs /bin/echo hello \
  > /private/tmp/echo_trace.stdout 2> /private/tmp/echo_trace.stderr
```

### Verifier + trace

```bash
timeout 20s env ISH_ARM64_BACKEND=arm64_jit ISH_ARM64_JIT_VERIFY=1 ISH_ARM64_JIT_TRACE=1 \
  ./build-arm64-release/ish -f ./build/alpine-arm64-fakefs /bin/echo hello \
  > /private/tmp/echo_verify_trace.stdout 2> /private/tmp/echo_verify_trace.stderr
```

Do not leave trace enabled on broad benchmark runs unless the scope is tightly
filtered.

## Dump Mode

Dump mode is independent of verifier mode and works in plain JIT mode.

### Plain JIT dump

```bash
timeout 20s env ISH_ARM64_BACKEND=arm64_jit ISH_ARM64_JIT_DUMP=1 \
  ISH_ARM64_JIT_DUMP_FILTER=0xeffddc \
  ./build-arm64-release/ish -f ./build/alpine-arm64-fakefs /root/cbench_lite_arm64 \
  > /private/tmp/jit_dump.stdout 2> /private/tmp/jit_dump.stderr
```

### Render a dumped fragment

```bash
python3 -u tools/render_arm64_jit_dump.py \
  /private/tmp/fragment.log \
  /Users/zizhengguo/scratch/ish-arm64/benchmark/assets/cbench_lite_arm64 \
  0xeffdd000 \
  > /private/tmp/fragment.rendered
```

### Extract a specific fragment by guest start PC

```bash
python3 -u - <<'PY'
import json
from pathlib import Path
src = Path('/private/tmp/jit_dump.stderr')
want = '0xeffddc94'
for line in src.read_text(errors='replace').splitlines():
    marker = '[arm64-jit-dump] '
    if marker not in line:
        continue
    obj = json.loads(line.split(marker, 1)[1])
    if obj.get('start_pc') == want:
        Path('/private/tmp/fragment.log').write_text(line + '\n')
        break
PY
```

## Common Targets

### `bench_integer`

The current `cbench_lite` `bench_integer` fragment has commonly shown up as:

- `0xeffddc94..0xeffddd78`

Useful dump filter:

```bash
ISH_ARM64_JIT_DUMP_FILTER=0xeffddc
```

### `bench_call`

Useful dump filter:

```bash
ISH_ARM64_JIT_DUMP_FILTER=0xeffde0
```

## JIT vs Gadget Backend

This repo has two separate execution engines relevant to ARM64 guest work:

- threaded-code gadget backend in:
  - [/Users/zizhengguo/scratch/ish-arm64/asbestos/guest-arm64](/Users/zizhengguo/scratch/ish-arm64/asbestos/guest-arm64)
- runtime codegen JIT backend in:
  - [/Users/zizhengguo/scratch/ish-arm64/jit/guest-arm64](/Users/zizhengguo/scratch/ish-arm64/jit/guest-arm64)

When discussing “JIT” in current debugging, that usually means the runtime
codegen backend selected by:

```bash
ISH_ARM64_BACKEND=arm64_jit
```

If work later needs explicit gadget backend commands, add them here together
with the exact selector env var or launch mechanism once confirmed from code.

## Current Memory-Helper Direction

The intended memory-helper architecture is:

- emitted code calls family helper as a success-only operation
- family helper computes effective address and performs typed access
- shared page helper owns:
  - TLB hit/miss handling
  - spill/reload around miss
  - permission failure / page fault publication
  - fragment exit on fault

The ABI is documented in:

- [/Users/zizhengguo/scratch/ish-arm64/jit/guest-arm64/helpers.S](/Users/zizhengguo/scratch/ish-arm64/jit/guest-arm64/helpers.S)

When changing this architecture, update both that file and this one.

## Best Practices

- Start with `/bin/echo hello` under verifier before using `cbench_lite`.
- Use one bounded reproducer at a time.
- If a helper becomes non-leaf, verify `x30` preservation immediately.
- If a helper starts reusing `x4-x15`, save/restore them explicitly or redesign
  the register use.
- If a dump shows helper-heavy code shape, inspect both:
  - emitted fragment
  - family helper
  - shared page helper
- Prefer exact frontier logs over intuition.

## Maintenance Rule

Update this file whenever any of these change:

- build command
- build directory
- fakefs path
- JIT backend selector
- verifier env vars
- dump workflow
- trace workflow
- current best-practice bounded timeout values
