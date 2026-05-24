#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
import tempfile


PREFIX = "[arm64-jit-dump] "


def run_cmd(cmd, cwd=None):
    return subprocess.check_output(cmd, cwd=cwd, text=True)


def parse_dump_lines(path):
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith(PREFIX):
                continue
            records.append(json.loads(line[len(PREFIX):]))
    return records


def disassemble_guest(binary_path, start_pc, end_pc, guest_base):
    start = start_pc - guest_base
    stop = end_pc
    out = run_cmd([
        "objdump", "-d",
        f"--start-address=0x{start:x}",
        f"--stop-address=0x{stop - guest_base:x}",
        binary_path,
    ])
    mapping = {}
    for line in out.splitlines():
        m = re.match(r"^\s*([0-9a-f]+):\s+([0-9a-f]{8})\s+(.*)$", line)
        if not m:
            continue
        off = int(m.group(1), 16)
        pc = guest_base + off
        mapping[pc] = f"0x{pc:016x}: {m.group(3).strip()}"
    return mapping


def disassemble_host_words(words):
    with tempfile.TemporaryDirectory(prefix="jitdump-host-") as td:
        asm = os.path.join(td, "frag.s")
        obj = os.path.join(td, "frag.o")
        with open(asm, "w", encoding="utf-8") as f:
            f.write(".text\n.globl _frag\n.p2align 2\n_frag:\n")
            for w in words:
                val = int(w, 16)
                b0 = val & 0xff
                b1 = (val >> 8) & 0xff
                b2 = (val >> 16) & 0xff
                b3 = (val >> 24) & 0xff
                f.write(f"  .byte 0x{b0:02x},0x{b1:02x},0x{b2:02x},0x{b3:02x}\n")
        subprocess.check_call(["clang", "-c", "-arch", "arm64", asm, "-o", obj])
        out = run_cmd(["otool", "-tvV", obj])
    host = {}
    for line in out.splitlines():
        m = re.match(r"^([0-9a-f]{16})\t(.*)$", line.strip())
        if not m:
            continue
        off = int(m.group(1), 16)
        host[off] = m.group(2).strip()
    return host


def render_record(rec, guest_map):
    print(f"# fragment {rec['start_pc']}..{rec['end_pc']}")
    if rec["gpr_map"]:
        regs = " ".join(
            f"{m['guest']}->{m['host']}({m['use_count']})" for m in rec["gpr_map"]
        )
    else:
        regs = "(none)"
    print(f"# register relation {regs}")
    print(f"# guest code segment at time of execution {rec['start_pc']}-{rec['end_pc']}")
    print(f"# host body bytes {rec['body_code_size']} total bytes {rec['code_size']}")
    host_map = disassemble_host_words(rec["host_words"])

    host_offs = [insn["host_off"] for insn in rec["guest_insns"]]
    body_end = rec["body_code_size"]
    for idx, insn in enumerate(rec["guest_insns"]):
        pc = int(insn["pc"], 16)
        print()
        print(f"# {guest_map.get(pc, insn['pc'])}")
        start = insn["host_off"]
        if start == 0xFFFFFFFF:
            continue
        next_start = body_end
        for later in host_offs[idx + 1:]:
            if later != 0xFFFFFFFF and later > start:
                next_start = later
                break
        for off in range(start, min(next_start, body_end), 4):
            asm = host_map.get(off, "<undisassembled>")
            print(f"  0x{off:04x}: {asm}")

    if rec["entry_thunks_offset"] < rec["code_size"]:
        print()
        print("# entry thunks")
        for insn in rec["guest_insns"]:
            if insn["entry_off"] == 0xFFFFFFFF:
                continue
            pc = int(insn["pc"], 16)
            print(f"# entry for {guest_map.get(pc, insn['pc'])}")
            off = insn["entry_off"]
            for x in range(off, min(off + 18 * 4 + 8, rec["code_size"]), 4):
                asm = host_map.get(x, "<undisassembled>")
                print(f"  0x{x:04x}: {asm}")


def main():
    if len(sys.argv) != 4:
        print("usage: render_arm64_jit_dump.py <dump.log> <guest-binary> <guest-base-hex>", file=sys.stderr)
        return 1
    dump_path = sys.argv[1]
    guest_binary = sys.argv[2]
    guest_base = int(sys.argv[3], 16)
    records = parse_dump_lines(dump_path)
    if not records:
        print("no dump records found", file=sys.stderr)
        return 1
    low = min(int(r["start_pc"], 16) for r in records)
    high = max(int(r["end_pc"], 16) for r in records)
    guest_map = disassemble_guest(guest_binary, low, high, guest_base)
    for rec in records:
        render_record(rec, guest_map)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
