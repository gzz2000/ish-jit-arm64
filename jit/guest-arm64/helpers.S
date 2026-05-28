#if defined(GUEST_ARM64) && defined(__aarch64__)

.text
.align 2

// JIT-private memory/helper ABI.
//
// Stable across these helpers:
//   x4-x15, x19-x29, v0-v31, NZCV, FPCR, FPSR
// Clobberable:
//   x0-x3, x16-x17, x30
//
// Fragment register model:
//   x19 = cpu*
//   x20 = tlb*
//   x21 = rt*
//   x22 = cached guest sp
//
// Emitted code -> family helper:
//   Inputs:
//     x0 = rt*
//     x1 = packed metadata; high 32 bits always carry guest_pc
//     x2 = guest effective address
//     x3+ = family-specific operands/results contract
//   Return:
//     success only
//     w0 = INT_NONE (-1)
//     x1/x2 may carry load results if that family requested register return
//   Fault / interrupt:
//     must not be exposed back to emitted code; helper exits internally
//
// Family helper -> shared page helper:
//   Inputs:
//     x0 = rt*
//     x1 = packed metadata; high 32 bits carry guest_pc
//     x2 = guest effective address
//     x4 = access_kind (0=read, 1=write, 2=exec)
//   Return on success:
//     x0 = host page base (page aligned)
//     x2 = guest effective address (preserved)
//     w1 = INT_NONE (-1)
//   Fault / interrupt:
//     shared helper owns spill/reload, fault publication, and fragment exit
//     and does not return control to the family helper
//
// Rules:
//   - Any helper that executes BL is non-leaf and must preserve x30.
//   - Helpers must not reuse x4-x15 as scratch unless they save/restore them.
//   - The shared page helper always returns host_page_base, never data_minus_addr.
//   - Callers derive host_ptr as host_page_base + (guest_addr & 0xfff).
//
// For now the JITABI wrappers tail into existing C semantic helpers while
// preserving the tighter JIT-private register contract. This gives the JIT a
// stable helper ABI now, and later the same entry points can inline more logic
// without changing the emitter.

// Offsets used below:
// arm64_jit_runtime:
//   cpu = 0
//   tlb = 8
//   block = 16
//   spill_state_fn = 24
//   reload_state_fn = 32
// arm64_jit_block:
//   start_pc = 0
//   end_pc = 8
//   insn_count = 24
//   insn_pcs = 32
//   code_rx = 37176
//   insn_host_offsets = 37220
// cpu_state:
//   regs = 16
//   sp = 264
//   pc = 272
//   segfault_addr = 832
//   segfault_was_write = 840
// tlb:
//   dirty_page = 8
//   mem_changes = 16
//   segfault_addr = 24
//   entries = 32
// tlb_entry (32 bytes on arm64 guest builds):
//   page = 0
//   page_if_writable = 8
//   data_minus_addr = 16
//
// arm64_jit_tlb_profile:
//   bench_integer_lookups = 0
//   bench_integer_hits = 8
//   bench_integer_misses = 16

.extern _g_arm64_jit_tlb_profile

.macro JIT_TLB_LOOKUP_READ_HIT guest_addr, out_reg, scratch0, scratch1, miss_label
    eor \scratch0, \guest_addr, \guest_addr, lsr #13
    lsr \scratch0, \scratch0, #12
    and \scratch0, \scratch0, #0x1fff
    add \out_reg, x20, #32
    add \out_reg, \out_reg, \scratch0, lsl #5
    and \scratch1, \guest_addr, #0xfffffffffffff000
    ldr \scratch0, [\out_reg]
    eor \scratch0, \scratch0, \scratch1
    cbnz \scratch0, \miss_label
    ldr \scratch0, [\out_reg, #16]
    add \out_reg, \scratch0, \scratch1
.endm

.macro JIT_TLB_LOOKUP_WRITE_HIT guest_addr, out_reg, scratch0, scratch1, miss_label
    eor \scratch0, \guest_addr, \guest_addr, lsr #13
    lsr \scratch0, \scratch0, #12
    and \scratch0, \scratch0, #0x1fff
    add \out_reg, x20, #32
    add \out_reg, \out_reg, \scratch0, lsl #5
    and \scratch1, \guest_addr, #0xfffffffffffff000
    ldr \scratch0, [\out_reg, #8]
    eor \scratch0, \scratch0, \scratch1
    cbnz \scratch0, \miss_label
    str \scratch1, [x20, #8]
    ldr \scratch0, [\out_reg, #16]
    add \out_reg, \scratch0, \scratch1
.endm

.macro JIT_SIMD_SYNC_CPU_TO_Q rt, ptr
    cmp \rt, #0
    b.eq .Ljit_simd_sync_q0_\@
    cmp \rt, #1
    b.eq .Ljit_simd_sync_q1_\@
    cmp \rt, #2
    b.eq .Ljit_simd_sync_q2_\@
    cmp \rt, #3
    b.eq .Ljit_simd_sync_q3_\@
    cmp \rt, #4
    b.eq .Ljit_simd_sync_q4_\@
    cmp \rt, #5
    b.eq .Ljit_simd_sync_q5_\@
    cmp \rt, #6
    b.eq .Ljit_simd_sync_q6_\@
    cmp \rt, #7
    b.eq .Ljit_simd_sync_q7_\@
    cmp \rt, #8
    b.eq .Ljit_simd_sync_q8_\@
    cmp \rt, #9
    b.eq .Ljit_simd_sync_q9_\@
    cmp \rt, #10
    b.eq .Ljit_simd_sync_q10_\@
    cmp \rt, #11
    b.eq .Ljit_simd_sync_q11_\@
    cmp \rt, #12
    b.eq .Ljit_simd_sync_q12_\@
    cmp \rt, #13
    b.eq .Ljit_simd_sync_q13_\@
    cmp \rt, #14
    b.eq .Ljit_simd_sync_q14_\@
    cmp \rt, #15
    b.eq .Ljit_simd_sync_q15_\@
    cmp \rt, #16
    b.eq .Ljit_simd_sync_q16_\@
    cmp \rt, #17
    b.eq .Ljit_simd_sync_q17_\@
    cmp \rt, #18
    b.eq .Ljit_simd_sync_q18_\@
    cmp \rt, #19
    b.eq .Ljit_simd_sync_q19_\@
    cmp \rt, #20
    b.eq .Ljit_simd_sync_q20_\@
    cmp \rt, #21
    b.eq .Ljit_simd_sync_q21_\@
    cmp \rt, #22
    b.eq .Ljit_simd_sync_q22_\@
    cmp \rt, #23
    b.eq .Ljit_simd_sync_q23_\@
    cmp \rt, #24
    b.eq .Ljit_simd_sync_q24_\@
    cmp \rt, #25
    b.eq .Ljit_simd_sync_q25_\@
    cmp \rt, #26
    b.eq .Ljit_simd_sync_q26_\@
    cmp \rt, #27
    b.eq .Ljit_simd_sync_q27_\@
    cmp \rt, #28
    b.eq .Ljit_simd_sync_q28_\@
    cmp \rt, #29
    b.eq .Ljit_simd_sync_q29_\@
    cmp \rt, #30
    b.eq .Ljit_simd_sync_q30_\@
    ldr q31, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q0_\@:  ldr q0, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q1_\@:  ldr q1, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q2_\@:  ldr q2, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q3_\@:  ldr q3, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q4_\@:  ldr q4, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q5_\@:  ldr q5, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q6_\@:  ldr q6, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q7_\@:  ldr q7, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q8_\@:  ldr q8, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q9_\@:  ldr q9, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q10_\@: ldr q10, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q11_\@: ldr q11, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q12_\@: ldr q12, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q13_\@: ldr q13, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q14_\@: ldr q14, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q15_\@: ldr q15, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q16_\@: ldr q16, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q17_\@: ldr q17, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q18_\@: ldr q18, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q19_\@: ldr q19, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q20_\@: ldr q20, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q21_\@: ldr q21, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q22_\@: ldr q22, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q23_\@: ldr q23, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q24_\@: ldr q24, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q25_\@: ldr q25, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q26_\@: ldr q26, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q27_\@: ldr q27, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q28_\@: ldr q28, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q29_\@: ldr q29, [\ptr]
    b .Ljit_simd_sync_done_\@
.Ljit_simd_sync_q30_\@: ldr q30, [\ptr]
.Ljit_simd_sync_done_\@:
.endm

.macro JITABI_WRAP1 name, target
.globl \name
\name:
    stp x29, x30, [sp, #-64]!
    stp x16, x17, [sp, #16]
    stp x0, x1, [sp, #32]
    stp x2, x3, [sp, #48]
    mov x16, x0
    ldr x17, [x16, #24]
    cbz x17, 1f
    blr x17
1:
    ldp x0, x1, [sp, #32]
    bl \target
    add w17, w0, #1
    cbnz w17, 2f
    ldr x16, [sp, #32]
    ldr x17, [x16, #32]
    cbz x17, 2f
    str x0, [sp, #56]
    mov x0, x16
    blr x17
    ldr x0, [sp, #56]
2:
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #64
    ret
.endm

.macro JITABI_WRAP2 name, target
.globl \name
\name:
    stp x29, x30, [sp, #-64]!
    stp x16, x17, [sp, #16]
    stp x0, x1, [sp, #32]
    stp x2, x3, [sp, #48]
    mov x16, x0
    ldr x17, [x16, #24]
    cbz x17, 1f
    blr x17
1:
    ldp x0, x1, [sp, #32]
    ldr x2, [sp, #48]
    bl \target
    add w17, w0, #1
    cbnz w17, 2f
    ldr x16, [sp, #32]
    ldr x17, [x16, #32]
    cbz x17, 2f
    str x0, [sp, #56]
    mov x0, x16
    blr x17
    ldr x0, [sp, #56]
2:
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #64
    ret
.endm

.macro JITABI_WRAP3 name, target
.globl \name
\name:
    stp x29, x30, [sp, #-64]!
    stp x16, x17, [sp, #16]
    stp x0, x1, [sp, #32]
    stp x2, x3, [sp, #48]
    mov x16, x0
    ldr x17, [x16, #24]
    cbz x17, 1f
    blr x17
1:
    ldp x0, x1, [sp, #32]
    ldp x2, x3, [sp, #48]
    bl \target
    add w17, w0, #1
    cbnz w17, 2f
    ldr x16, [sp, #32]
    ldr x17, [x16, #32]
    cbz x17, 2f
    str x0, [sp, #56]
    mov x0, x16
    blr x17
    ldr x0, [sp, #56]
2:
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #64
    ret
.endm

.macro JITABI_WRAP1_SUCCESS name, target
.globl \name
\name:
    stp x29, x30, [sp, #-64]!
    stp x16, x17, [sp, #16]
    stp x0, x1, [sp, #32]
    stp x2, x3, [sp, #48]
    mov x16, x0
    ldr x17, [x16, #24]
    cbz x17, 1f
    blr x17
1:
    ldp x0, x1, [sp, #32]
    bl \target
    add w17, w0, #1
    cbnz w17, 3f
    ldr x16, [sp, #32]
    ldr x17, [x16, #32]
    cbz x17, 2f
    str x0, [sp, #56]
    mov x0, x16
    blr x17
    ldr x0, [sp, #56]
2:
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #64
    ret
3:
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #64
    JIT_FRAGMENT_EPILOGUE_AND_RET
.endm

.macro JITABI_WRAP2_SUCCESS name, target
.globl \name
\name:
    stp x29, x30, [sp, #-64]!
    stp x16, x17, [sp, #16]
    stp x0, x1, [sp, #32]
    stp x2, x3, [sp, #48]
    mov x16, x0
    ldr x17, [x16, #24]
    cbz x17, 1f
    blr x17
1:
    ldp x0, x1, [sp, #32]
    ldr x2, [sp, #48]
    bl \target
    add w17, w0, #1
    cbnz w17, 3f
    ldr x16, [sp, #32]
    ldr x17, [x16, #32]
    cbz x17, 2f
    str x0, [sp, #56]
    mov x0, x16
    blr x17
    ldr x0, [sp, #56]
2:
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #64
    ret
3:
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #64
    JIT_FRAGMENT_EPILOGUE_AND_RET
.endm

.macro JITABI_WRAP3_SUCCESS name, target
.globl \name
\name:
    stp x29, x30, [sp, #-64]!
    stp x16, x17, [sp, #16]
    stp x0, x1, [sp, #32]
    stp x2, x3, [sp, #48]
    mov x16, x0
    ldr x17, [x16, #24]
    cbz x17, 1f
    blr x17
1:
    ldp x0, x1, [sp, #32]
    ldp x2, x3, [sp, #48]
    bl \target
    add w17, w0, #1
    cbnz w17, 3f
    ldr x16, [sp, #32]
    ldr x17, [x16, #32]
    cbz x17, 2f
    str x0, [sp, #56]
    mov x0, x16
    blr x17
    ldr x0, [sp, #56]
2:
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #64
    ret
3:
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #64
    JIT_FRAGMENT_EPILOGUE_AND_RET
.endm

.macro JIT_FRAGMENT_EPILOGUE_AND_RET
    ldr q8, [sp]
    ldr q9, [sp, #16]
    ldr q10, [sp, #32]
    ldr q11, [sp, #48]
    ldr q12, [sp, #64]
    ldr q13, [sp, #80]
    ldr q14, [sp, #96]
    ldr q15, [sp, #112]
    add sp, sp, #128
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #96
    ret
.endm

// Common page-aligned TLB lookup for scalar live memory helpers.
// Inputs:
//   x0 = rt*
//   x1 = packed guest_pc/metadata; guest_pc is in the high 32 bits
//   x2 = guest_addr
//   x4 = access_kind (MEM_READ=0, MEM_WRITE=1)
// Outputs on success:
//   x0 = host_page_base
//   w1 = -1 (INT_NONE)
//   x2 = guest_addr
// Outputs on failure:
//   x0 = interrupt
//   w1 = interrupt
//   x2 = guest_addr
// On miss, this helper performs the TLB refill internally. On fault, it also
// spills cached guest state and publishes the architectural fault state so the
// caller can return the interrupt directly without an extra C helper round-trip.
.globl _arm64_jit_mem_lookup_page_live_dead_draft
_arm64_jit_mem_lookup_page_live_dead_draft:
    b _arm64_jit_mem_lookup_page_live

    // Legacy draft body kept temporarily to minimize churn while the shared
    // page-lookup prototype is stabilized. It is intentionally unreachable.
    stp x29, x30, [sp, #-112]!
    stp x16, x17, [sp, #16]
    stp x4, x5, [sp, #32]
    stp x6, x7, [sp, #48]
    stp x0, x1, [sp, #64]
    stp x2, x4, [sp, #80]

    mov x5, x2
    ldr x6, [x0, #8]               // tlb*
    and x7, x5, #0xfffffffffffff000
    eor x16, x5, x5, lsr #13
    lsr x16, x16, #12
    and x16, x16, #0x1fff
    add x16, x6, #32
    add x16, x16, x16, lsl #0      // keep assembler from reusing scratch
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x16, x16, lsl #0
    add x16, x6, #32
    add x16, x16, x16, lsl #0
    add x16, x6, x16, lsl #5      // tlb entry*

    ldr x4, [sp, #88]              // access_kind
    cbnz x4, 0f
    ldr x17, [x16, #0]             // entry.page
    cmp x17, x7
    b.ne 1f
    ldr x17, [x16, #16]            // data_minus_addr
    add x0, x17, x7                // host_page_base
    mov w1, #-1
    b 2f
0:
    ldr x17, [x16, #8]             // entry.page_if_writable
    cmp x17, x7
    b.ne 1f
    str x7, [x6, #8]               // tlb->dirty_page
    ldr x17, [x16, #16]            // data_minus_addr
    add x0, x17, x7                // host_page_base
    mov w1, #-1
    b 2f
1:
    mov x0, x6                     // tlb*
    mov x1, x5                     // guest_addr
    mov x2, x4                     // access_kind
    bl _tlb_handle_miss
    cbz x0, 3f
    ldr x2, [sp, #80]              // guest_addr
    and x16, x2, #0xfff
    sub x0, x0, x16                // host_page_base
    mov w1, #-1
    b 2f
3:
    ldr x0, [sp, #64]              // rt*
    ldr x16, [x0, #24]             // spill_state_fn
    cbz x16, 4f
    blr x16
4:
    ldr x0, [sp, #64]              // rt*
    ldr x5, [x0, #0]               // cpu*
    ldr x6, [x0, #8]               // tlb*
    ldr x6, [x6, #24]              // tlb->segfault_addr
    str x6, [x5, #832]             // cpu->segfault_addr
    ldr x6, [sp, #88]              // access_kind
    cmp x6, #0
    cset w6, ne
    strb w6, [x5, #840]            // cpu->segfault_was_write
    ldr x6, [sp, #72]              // packed guest_pc/metadata
    lsr x6, x6, #32
    str x6, [x5, #272]             // cpu->pc
    str x6, [x0, #40]              // rt->resume_pc
    str x6, [x0, #48]              // rt->fault_pc
    mov w6, #13                    // INT_GPF
    str w6, [x0, #56]              // rt->exit_interrupt
    mov x0, #13
    mov w1, #13
2:
    ldr x2, [sp, #80]              // guest_addr
    ldp x6, x7, [sp, #48]
    ldp x4, x5, [sp, #32]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #112
    ret

// Common page-aligned TLB lookup for scalar live memory helpers.
// Inputs:
//   x0 = rt*
//   x1 = packed guest_pc/metadata; guest_pc is in the high 32 bits
//   x2 = guest_addr
//   x4 = access_kind (MEM_READ=0, MEM_WRITE=1)
// Outputs on success:
//   x0 = host_page_base
//   x1 = original packed guest_pc/metadata
//   x2 = guest_addr
// On failure:
//   does not return to the family helper; exits the fragment directly.
.globl _arm64_jit_mem_lookup_page_live
_arm64_jit_mem_lookup_page_live:
    stp x29, x30, [sp, #-112]!
    stp x16, x17, [sp, #16]
    stp x4, x5, [sp, #32]
    stp x6, x7, [sp, #48]
    stp x0, x1, [sp, #80]          // rt*, guest_pc
    str x2, [sp, #96]              // guest_addr
    str x4, [sp, #104]             // access_kind

    mov x3, x2                     // guest_addr scratch
    ldr x0, [x0, #8]               // tlb*
    and x17, x3, #0xfffffffffffff000
    eor x1, x3, x3, lsr #13
    lsr x1, x1, #12
    and x1, x1, #0x1fff
    add x16, x0, #32
    add x16, x16, x1, lsl #5       // tlb entry*

    ldr x3, [sp, #104]             // access_kind
    mov x5, #0                     // profile base or 0 when not tracking
    ldr x6, [sp, #88]              // packed guest_pc/metadata
    lsr x6, x6, #32                // guest_pc
    mov x7, #0xdc94
    movk x7, #0xeffd, lsl #16
    cmp x6, x7
    b.lo 90f
    mov x7, #0xdd78
    movk x7, #0xeffd, lsl #16
    cmp x6, x7
    b.hs 90f
    adrp x5, _g_arm64_jit_tlb_profile@PAGE
    add x5, x5, _g_arm64_jit_tlb_profile@PAGEOFF
    ldr x7, [x5]
    add x7, x7, #1
    str x7, [x5]
90:
    cbnz x3, 1f
    ldr x1, [x16, #0]              // entry.page
    cmp x1, x17
    b.ne 2f
    ldr x1, [x16, #16]             // data_minus_addr
    add x0, x1, x17                // host_page_base
    cmp x5, #0
    b.eq 91f
    add x7, x5, #8
    ldr x16, [x7]
    add x16, x16, #1
    str x16, [x7]
91:
    ldr x16, [sp, #80]             // rt*
    mov x1, #0x11
    str x1, [x16, #72]             // rt->debug1 = read-hit
    str x0, [x16, #80]             // rt->debug2 = host_page_base
    b 5f
1:
    ldr x1, [x16, #8]              // entry.page_if_writable
    cmp x1, x17
    b.ne 2f
    str x17, [x0, #8]              // tlb->dirty_page
    ldr x1, [x16, #16]             // data_minus_addr
    add x0, x1, x17                // host_page_base
    cmp x5, #0
    b.eq 92f
    add x7, x5, #8
    ldr x16, [x7]
    add x16, x16, #1
    str x16, [x7]
92:
    ldr x16, [sp, #80]             // rt*
    mov x1, #0x12
    str x1, [x16, #72]             // rt->debug1 = write-hit
    str x0, [x16, #80]             // rt->debug2 = host_page_base
    b 5f
2:
    ldr x0, [sp, #80]              // rt*
    ldr x16, [x0, #24]             // spill_state_fn
    cbz x16, 6f
    blr x16
6:
    ldr x0, [sp, #80]              // rt*
    ldr x6, [x0, #8]               // tlb*
    ldr x1, [sp, #96]              // guest_addr
    ldr x2, [sp, #104]             // access_kind
    str x5, [sp, #72]              // profile base across C miss handler
    mov x0, x6                     // tlb*
    bl _tlb_handle_miss
    ldr x5, [sp, #72]              // profile base
    cbz x0, 3f
    cmp x5, #0
    b.eq 93f
    add x7, x5, #16
    ldr x16, [x7]
    add x16, x16, #1
    str x16, [x7]
93:
    str x0, [sp, #64]              // host_ptr from tlb_handle_miss
    ldr x17, [sp, #80]             // rt*
    ldr x16, [x17, #32]            // reload_state_fn
    cbz x16, 7f
    mov x0, x17
    blr x16
7:
    ldr x0, [sp, #64]              // host_ptr from tlb_handle_miss
    ldr x2, [sp, #96]              // guest_addr
    and x16, x2, #0xfff
    sub x0, x0, x16                // host_page_base
    ldr x16, [sp, #80]             // rt*
    mov x17, #0x21
    str x17, [x16, #72]            // rt->debug1 = miss-success
    str x7, [x16, #64]             // rt->debug0 = cached host x7 on miss-success return
    str x0, [x16, #80]             // rt->debug2 = host_page_base
    b 5f
3:
    ldr x0, [sp, #80]              // rt*
    ldr x16, [x0, #24]             // spill_state_fn
    cbz x16, 4f
    blr x16
4:
    ldr x0, [sp, #80]              // rt*
    ldr x5, [x0, #0]               // cpu*
    ldr x6, [x0, #8]               // tlb*
    ldr x6, [x6, #24]              // tlb->segfault_addr
    str x6, [x5, #832]             // cpu->segfault_addr
    ldr x6, [sp, #104]             // access_kind
    cmp x6, #0
    cset w6, ne
    strb w6, [x5, #840]            // cpu->segfault_was_write
    ldr x6, [sp, #88]              // packed guest_pc/metadata
    lsr x6, x6, #32
    str x6, [x5, #272]             // cpu->pc
    str x6, [x0, #40]              // rt->resume_pc
    str x6, [x0, #48]              // rt->fault_pc
    mov w6, #13                    // INT_GPF
    str w6, [x0, #56]              // rt->exit_interrupt
    mov x6, #0x31
    str x6, [x0, #72]              // rt->debug1 = miss-fault
    mov x0, #13
    ldp x6, x7, [sp, #48]
    ldp x4, x5, [sp, #32]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #112
    JIT_FRAGMENT_EPILOGUE_AND_RET
5:
    ldr x1, [sp, #88]              // original packed guest_pc/metadata
    ldr x2, [sp, #96]              // guest_addr
    ldp x6, x7, [sp, #48]
    ldp x4, x5, [sp, #32]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #112
    ret
6:
    ldr x1, [sp, #88]              // original packed guest_pc/metadata
    ldr x2, [sp, #96]              // guest_addr
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #112
    ret

.globl _arm64_jit_helper_ldr_imm_unsigned
_arm64_jit_helper_ldr_imm_unsigned:
    // x0 = rt*, x1 = guest_pc, x2 = packed1
    // Fast path only for plain LDR{B,H,W,X} unsigned-immediate, single-page.
    stp x29, x30, [sp, #-64]!
    stp x16, x17, [sp, #16]
    stp x0, x1, [sp, #32]
    str x2, [sp, #48]

    ubfx x16, x2, #24, #2          // load_mode
    cbnz x16, 9f                   // special/sign-extending modes -> fallback

    ldr x16, [sp, #32]             // runtime*
    ldr x17, [x16, #0]             // cpu*
    ubfx x1, x2, #5, #5            // rn
    cmp x1, #31
    b.ne 0f
    ldr x1, [x17, #264]            // sp
    b 1f
0:
    add x16, x17, #16              // &cpu->regs[0]
    ldr x1, [x16, x1, lsl #3]
1:
    ubfx x16, x2, #10, #2          // size_shift
    ubfx x3, x2, #12, #12          // imm12
    lsl x3, x3, x16
    add x1, x1, x3                 // guest addr

    and x3, x1, #0xfff
    mov x2, #8
    lsl x2, x2, x16
    sub x2, x2, #1
    add x3, x3, x2
    cmp x3, #0xfff
    bhi 9f                         // cross-page -> fallback

    ldr x2, [sp, #32]              // runtime*
    ldr x2, [x2, #8]               // tlb*
    eor x3, x1, x1, lsr #13
    lsr x3, x3, #12
    and x3, x3, #0x1fff
    add x3, x2, #32
    add x3, x3, x3, lsl #5         // overwritten below
    eor x16, x1, x1, lsr #13
    lsr x16, x16, #12
    and x16, x16, #0x1fff
    add x3, x2, #32
    add x3, x3, x16, lsl #5        // tlb entry*
    and x16, x1, #0xfffffffffffff000
    ldr x2, [x3, #0]               // entry.page
    cmp x2, x16
    b.ne 9f
    ldr x2, [x3, #16]              // data_minus_addr
    add x2, x2, x1                 // host ptr

    ldr x1, [sp, #48]              // packed1
    ubfx x16, x1, #10, #2          // size_shift
    cmp x16, #3
    b.eq 2f
    cmp x16, #2
    b.eq 3f
    cmp x16, #1
    b.eq 4f
    ldrb w2, [x2]
    b 5f
2:
    ldr x2, [x2]
    b 5f
3:
    ldr w2, [x2]
    b 5f
4:
    ldrh w2, [x2]
5:
    ubfx x3, x1, #0, #5           // rt
    cmp x3, #31
    b.eq 6f
    ldr x17, [sp, #32]
    ldr x17, [x17, #0]            // cpu*
    add x17, x17, #16
    str x2, [x17, x3, lsl #3]
6:
    ldr x1, [sp, #40]             // guest_pc
    add x1, x1, #4
    ldr x17, [sp, #32]
    str x1, [x17, #40]            // rt->resume_pc
    ldr x17, [x17, #0]            // cpu*
    str x1, [x17, #272]           // cpu->pc
    mov x0, #-1                   // INT_NONE
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #64
    ret
9:
    ldr x0, [sp, #32]
    ldr x1, [sp, #40]
    ldr x2, [sp, #48]
    bl _arm64_jit_c_ldr_imm_unsigned
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #64
    ret

.globl _arm64_jit_helper_ldr_imm_unsigned_live
_arm64_jit_helper_ldr_imm_unsigned_live:
    // x0 = rt*, x1 = packed1, x2 = guest_addr, x3 = store_in_x1
    ubfx x17, x1, #24, #2          // load_mode
    mov x16, #3
    eor x16, x17, x16
    cbz x16, .Lldr_uimm_cold
    mov x16, #2
    eor x16, x17, x16
    cbnz x16, 0f
    ubfx x16, x1, #10, #2          // size_shift
    mov x17, #2
    eor x17, x16, x17
    cbnz x17, .Lldr_uimm_cold
0:
    ubfx x16, x1, #10, #2          // size_shift
    and x17, x2, #0xfff
    mov x0, #1
    lsl x0, x0, x16
    sub x0, x0, #1
    add x17, x17, x0
    lsr x0, x17, #12
    cbnz x0, .Lldr_uimm_cold

    JIT_TLB_LOOKUP_READ_HIT x2, x0, x16, x17, .Lldr_uimm_cold
    and x16, x2, #0xfff
    add x16, x0, x16               // host ptr

    ubfx x17, x1, #24, #2          // load_mode
    mov x0, #2
    eor x0, x17, x0
    cbz x0, 8f                     // LDRSW
    ubfx x17, x1, #10, #2          // size_shift
    mov x0, #3
    eor x0, x17, x0
    cbz x0, 1f
    mov x0, #2
    eor x0, x17, x0
    cbz x0, 2f
    mov x0, #1
    eor x0, x17, x0
    cbz x0, 3f
    ldrb w0, [x16]
    b 4f
1:
    ldr x0, [x16]
    b 4f
2:
    ldr w0, [x16]
    b 4f
3:
    ldrh w0, [x16]
    b 4f
8:
    ldr w0, [x16]
    sxtw x0, w0
4:
    lsr x16, x1, #32               // guest_pc
    cbnz x3, 5f
    ubfx x17, x1, #0, #5           // rt
    mov x1, #31
    eor x1, x17, x1
    cbz x1, 6f
    ldr x1, [x21, #0]              // cpu*
    add x1, x1, #16
    str x0, [x1, x17, lsl #3]
    b 6f
5:
    mov x1, x0
6:
    add x16, x16, #4
    str x16, [x21, #40]            // rt->resume_pc
    ldr x17, [x21, #0]             // cpu*
    str x16, [x17, #272]           // cpu->pc
    mov x0, #-1                    // INT_NONE
    ret

.Lldr_uimm_cold:
    stp x29, x30, [sp, #-80]!
    stp x16, x17, [sp, #16]
    stp x1, x3, [sp, #48]          // packed1, store_in_x1
    str x4, [sp, #72]
    mrs x17, nzcv
    str x17, [sp, #64]

    ubfx x17, x1, #24, #2          // load_mode
    ubfx x16, x1, #10, #2          // size_shift
    cmp x17, #3
    b.eq 90f                       // sign-extending byte/half loads -> fallback
    cmp x17, #2
    b.ne 0f
    cmp x16, #2
    b.ne 90f                       // only LDRSW is valid for load_mode==2
0:
    and x17, x2, #0xfff
    mov x3, #1
    lsl x3, x3, x16
    sub x3, x3, #1
    add x17, x17, x3
    cmp x17, #0xfff
    bhi 90f

    mov x0, x21
    mov x4, #0
    bl _arm64_jit_mem_lookup_page_live
    ldr x1, [sp, #48]              // packed1
    ldr x3, [sp, #56]              // store_in_x1
    and x16, x2, #0xfff
    add x16, x0, x16               // host ptr

    ubfx x17, x1, #10, #2          // size_shift
    ubfx x0, x1, #24, #2           // load_mode
    cmp x0, #2
    b.eq 98f
    cmp x17, #3
    b.eq 91f
    cmp x17, #2
    b.eq 92f
    cmp x17, #1
    b.eq 93f
    ldrb w0, [x16]
    b 94f
91:
    ldr x0, [x16]
    b 94f
92:
    ldr w0, [x16]
    b 94f
98:
    ldr w0, [x16]
    sxtw x0, w0
    b 94f
93:
    ldrh w0, [x16]
94:
    ldr x17, [sp, #48]             // packed1_live
    lsr x17, x17, #32              // guest_pc
    mov w16, #0x37c0
    cmp w17, w16
    b.ne 99f
    str x1, [x21, #64]             // debug0 = packed1
    str x2, [x21, #72]             // debug1 = guest_addr
    str x0, [x21, #80]             // debug2 = loaded value
    str x16, [x21, #88]            // debug3 = host ptr
99:
    cbnz x3, 95f
    ubfx x17, x1, #0, #5           // rt
    cmp x17, #31
    b.eq 96f
    ldr x16, [x21, #0]             // cpu*
    add x16, x16, #16
    str x0, [x16, x17, lsl #3]
    b 96f
95:
    mov x1, x0
96:
    ldr x17, [sp, #48]             // packed1
    lsr x17, x17, #32              // guest_pc
    add x17, x17, #4
    str x17, [x21, #40]            // rt->resume_pc
    ldr x16, [x21, #0]             // cpu*
    str x17, [x16, #272]           // cpu->pc
    mov x0, #-1                    // INT_NONE
    ldr x17, [sp, #64]
    msr nzcv, x17
    ldr x4, [sp, #72]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #80
    ret
90:
    ldr x17, [x21, #24]            // spill_state_fn
    cbz x17, .Lldr_uimm_fallback_spilled
    blr x17
.Lldr_uimm_fallback_spilled:
    mov x0, x21                    // runtime*
    ldr x1, [sp, #48]              // packed1
    mov w2, w1                     // low 32 = packed1
    lsr x1, x1, #32                // guest_pc
    bl _arm64_jit_c_ldr_imm_unsigned
    cmp w0, #-1
    b.ne 98f
    ldr x16, [x21, #32]            // reload_state_fn
    cbz x16, .Lldr_uimm_fallback_reloaded
    mov x0, x21
    blr x16
.Lldr_uimm_fallback_reloaded:
    ldr x3, [sp, #56]              // store_in_x1
    cbz x3, 97f
    ldr x2, [sp, #48]              // packed1
    ubfx x17, x2, #0, #5           // rt
    cmp x17, #31
    b.eq 97f
    ldr x16, [x21, #0]             // cpu*
    add x16, x16, #16
    ldr x1, [x16, x17, lsl #3]
97:
    mov x0, #-1
98:
    cmp w0, #-1
    b.ne 99f
    ldr x17, [sp, #64]
    msr nzcv, x17
    ldr x4, [sp, #72]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #80
    ret
99:
    ldr x17, [sp, #64]
    msr nzcv, x17
    ldr x4, [sp, #72]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #80
    JIT_FRAGMENT_EPILOGUE_AND_RET
    ret

.globl _arm64_jit_helper_str_imm_unsigned
_arm64_jit_helper_str_imm_unsigned:
    // x0 = rt*, x1 = packed
    // Fast path only for plain STR{B,H,W,X} unsigned-immediate, single-page.
    stp x29, x30, [sp, #-48]!
    stp x16, x17, [sp, #16]
    stp x0, x1, [sp, #32]

    // packed layout:
    // guest_pc: bits 31:0
    // rt_reg:   bits 36:32
    // rn_reg:   bits 41:37
    // size:     bits 43:42
    // imm12:    bits 55:44
    ubfx x16, x1, #37, #5          // rn
    ldr x17, [x0, #0]              // cpu*
    cmp x16, #31
    b.ne 0f
    ldr x2, [x17, #264]            // sp
    b 1f
0:
    add x3, x17, #16               // &cpu->regs[0]
    ldr x2, [x3, x16, lsl #3]
1:
    ubfx x16, x1, #42, #2          // size_shift
    ubfx x3, x1, #44, #12          // imm12
    lsl x3, x3, x16
    add x2, x2, x3                 // guest addr

    // Reject cross-page accesses on fast path.
    and x3, x2, #0xfff
    mov x4, #8
    lsl x4, x4, x16
    sub x4, x4, #1
    add x3, x3, x4
    cmp x3, #0xfff
    bhi 9f

    ldr x3, [x0, #8]               // tlb*
    eor x4, x2, x2, lsr #13
    lsr x4, x4, #12
    and x4, x4, #0x1fff
    add x4, x3, #32
    add x4, x4, x4, lsl #5         // overwritten below
    eor x16, x2, x2, lsr #13
    lsr x16, x16, #12
    and x16, x16, #0x1fff
    add x4, x3, #32
    add x4, x4, x16, lsl #5        // tlb entry*
    and x16, x2, #0xfffffffffffff000
    ldr x5, [x4, #8]               // entry.page_if_writable
    cmp x5, x16
    b.ne 9f

    // Mirror __tlb_write_ptr side effect on successful writable hit.
    str x16, [x3, #8]              // tlb->dirty_page
    ldr x5, [x4, #16]              // data_minus_addr
    add x5, x5, x2                 // host ptr

    ldr x1, [sp, #40]              // packed
    ubfx x4, x1, #32, #5           // rt
    cmp x4, #31
    b.eq 2f
    add x17, x17, #16
    ldr x6, [x17, x4, lsl #3]
    b 3f
2:
    mov x6, xzr
3:
    ubfx x16, x1, #42, #2          // size_shift
    cmp x16, #3
    b.eq 4f
    cmp x16, #2
    b.eq 5f
    cmp x16, #1
    b.eq 6f
    strb w6, [x5]
    b 7f
4:
    str x6, [x5]
    b 7f
5:
    str w6, [x5]
    b 7f
6:
    strh w6, [x5]
7:
    ldr w1, [sp, #40]              // guest_pc low 32
    add x1, x1, #4
    ldr x17, [sp, #32]             // runtime*
    str x1, [x17, #40]             // rt->resume_pc
    ldr x17, [x17, #0]             // cpu*
    str x1, [x17, #272]            // cpu->pc
    mov x0, #-1                    // INT_NONE
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #48
    ret
9:
    ldr x0, [sp, #32]
    ldr x1, [sp, #40]
    bl _arm64_jit_c_str_imm_unsigned
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

.globl _arm64_jit_helper_str_imm_unsigned_live
_arm64_jit_helper_str_imm_unsigned_live:
    // x0 = rt*, x1 = packed, x2 = guest_addr, x3 = value
    // Fast path mirrors _arm64_jit_helper_str_imm_unsigned, but uses live
    // register values directly and only materializes through ctx-provided
    // snippets on fallback.
    ubfx x16, x1, #42, #2          // size_shift
    and x17, x2, #0xfff
    mov x0, #1
    lsl x0, x0, x16
    sub x0, x0, #1
    add x17, x17, x0
    lsr x0, x17, #12
    cbnz x0, .Lstr_uimm_cold

    JIT_TLB_LOOKUP_WRITE_HIT x2, x0, x16, x17, .Lstr_uimm_cold
    and x16, x2, #0xfff
    add x16, x0, x16               // host ptr

    ubfx x17, x1, #42, #2          // size_shift
    mov x0, #3
    eor x0, x17, x0
    cbz x0, 11f
    mov x0, #2
    eor x0, x17, x0
    cbz x0, 12f
    mov x0, #1
    eor x0, x17, x0
    cbz x0, 13f
    strb w3, [x16]
    b 14f
11:
    str x3, [x16]
    b 14f
12:
    str w3, [x16]
    b 14f
13:
    strh w3, [x16]
14:
    uxtw x16, w1                   // guest_pc low 32
    add x16, x16, #4
    str x16, [x21, #40]            // rt->resume_pc
    ldr x17, [x21, #0]             // cpu*
    str x16, [x17, #272]           // cpu->pc
    mov x0, #-1                    // INT_NONE
    ret

.Lstr_uimm_cold:
    stp x29, x30, [sp, #-80]!
    stp x16, x17, [sp, #16]
    stp x1, x3, [sp, #48]          // packed, store value
    str x4, [sp, #72]
    mrs x17, nzcv
    str x17, [sp, #64]

    ubfx x16, x1, #42, #2          // size_shift
    and x17, x2, #0xfff
    mov x0, #1
    lsl x0, x0, x16
    sub x0, x0, #1
    add x17, x17, x0
    cmp x17, #0xfff
    bhi 90f

    mov x0, x21
    mov x4, #1
    bl _arm64_jit_mem_lookup_page_live
    ldr x1, [sp, #48]              // packed
    ldr x3, [sp, #56]              // store value
    and x16, x2, #0xfff
    add x16, x0, x16               // host ptr

    ubfx x17, x1, #42, #2          // size_shift
    cmp x17, #3
    b.eq 91f
    cmp x17, #2
    b.eq 92f
    cmp x17, #1
    b.eq 93f
    strb w3, [x16]
    b 94f
91:
    str x3, [x16]
    b 94f
92:
    str w3, [x16]
    b 94f
93:
    strh w3, [x16]
94:
    ldr w17, [sp, #48]             // guest_pc low 32
    add x17, x17, #4
    str x17, [x21, #40]            // rt->resume_pc
    ldr x16, [x21, #0]             // cpu*
    str x17, [x16, #272]           // cpu->pc
    mov x0, #-1                    // INT_NONE
    ldr x17, [sp, #64]
    msr nzcv, x17
    ldr x4, [sp, #72]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #80
    ret
90:
    ldr x17, [x21, #24]            // spill_state_fn
    cbz x17, .Lstr_uimm_fallback_spilled
    blr x17
.Lstr_uimm_fallback_spilled:
    mov x0, x21
    ldr x1, [sp, #48]
    bl _arm64_jit_c_str_imm_unsigned
    cmp w0, #-1
    b.ne 97f
    ldr x16, [x21, #32]            // reload_state_fn
    cbz x16, .Lstr_uimm_fallback_reloaded
    mov x0, x21
    blr x16
.Lstr_uimm_fallback_reloaded:
    mov x0, #-1
97:
    cmp w0, #-1
    b.ne 98f
    ldr x17, [sp, #64]
    msr nzcv, x17
    ldr x4, [sp, #72]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #80
    ret
98:
    ldr x17, [sp, #64]
    msr nzcv, x17
    ldr x4, [sp, #72]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #80
    JIT_FRAGMENT_EPILOGUE_AND_RET
    ret

.globl _arm64_jit_helper_simd_ldst_imm_unsigned
_arm64_jit_helper_simd_ldst_imm_unsigned:
    // x0 = rt*, x1 = guest_pc, x2 = insn
    // Fast path for plain SIMD unsigned-immediate load/store:
    //   Bt/Ht/St/Dt and Qt
    // same-page only, fallback for anything awkward.
    stp x29, x30, [sp, #-64]!
    stp x16, x17, [sp, #16]
    stp x0, x1, [sp, #32]
    str x2, [sp, #48]

    ubfx x16, x2, #30, #2          // size
    ubfx x17, x2, #22, #2          // opc
    cmp x16, #0
    b.ne 0f
    tst x17, #0x2
    b.eq 0f                        // size==0 and opc bit1 clear -> Bt, not Qt
    // Qt requires opc bit1 set, allow both STR/LDR via bit0
    mov x3, #16
    b 1f
0:
    cmp x17, #0x2
    b.hs 9f                        // other awkward SIMD encodings stay in C
    mov x3, #1
    lsl x3, x3, x16               // width bytes = 1/2/4/8
1:
    ldr x16, [sp, #32]             // runtime*
    ldr x17, [x16, #0]             // cpu*
    ubfx x1, x2, #5, #5            // rn
    cmp x1, #31
    b.ne 2f
    ldr x1, [x17, #264]            // sp
    b 3f
2:
    add x16, x17, #16
    ldr x1, [x16, x1, lsl #3]
3:
    ldr x2, [sp, #48]              // insn
    ubfx x16, x2, #30, #2          // size
    ubfx x17, x2, #22, #2          // opc
    cmp x16, #0
    b.ne 4f
    tst x17, #0x2
    b.eq 4f
    mov x4, #4                     // Qt scale
    b 5f
4:
    mov x4, x16                    // scalar SIMD element scale
5:
    ubfx x5, x2, #10, #12          // imm12
    lsl x5, x5, x4
    add x1, x1, x5                 // guest addr

    and x5, x1, #0xfff
    sub x6, x3, #1
    add x5, x5, x6
    cmp x5, #0xfff
    bhi 9f

    ldr x5, [sp, #32]              // runtime*
    ldr x5, [x5, #8]               // tlb*
    eor x6, x1, x1, lsr #13
    lsr x6, x6, #12
    and x6, x6, #0x1fff
    add x6, x5, #32
    add x6, x6, x6, lsl #5         // overwritten below
    eor x16, x1, x1, lsr #13
    lsr x16, x16, #12
    and x16, x16, #0x1fff
    add x6, x5, #32
    add x6, x6, x16, lsl #5        // tlb entry*
    and x16, x1, #0xfffffffffffff000

    ldr x2, [sp, #48]              // insn
    ubfx x17, x2, #22, #2          // opc
    tst x17, #0x1                  // L
    b.ne 6f
    ldr x17, [x6, #8]              // entry.page_if_writable
    cmp x17, x16
    b.ne 9f
    str x16, [x5, #8]              // tlb->dirty_page
    b 7f
6:
    ldr x17, [x6, #0]              // entry.page
    cmp x17, x16
    b.ne 9f
7:
    ldr x6, [x6, #16]              // data_minus_addr
    add x6, x6, x1                 // host ptr

    ldr x2, [sp, #48]              // insn
    ubfx x7, x2, #0, #5            // rt
    ldr x17, [sp, #32]
    ldr x17, [x17, #0]             // cpu*
    add x17, x17, #288             // &cpu->fp[0]
    add x7, x17, x7, lsl #4        // &cpu->fp[rt]

    ubfx x16, x2, #30, #2          // size
    ubfx x17, x2, #22, #2          // opc
    cmp x16, #0
    b.ne 8f
    tst x17, #0x2
    b.eq 8f
    // Qt
    tst x17, #0x1
    b.ne 20f
    ldr x8, [x7]
    ldr x9, [x7, #8]
    str x8, [x6]
    str x9, [x6, #8]
    b 30f
20:
    ldr x8, [x6]
    ldr x9, [x6, #8]
    str x8, [x7]
    str x9, [x7, #8]
    b 30f

8:
    // Bt/Ht/St/Dt
    tst x17, #0x1
    b.ne 21f                       // load
    cmp x16, #3
    b.eq 22f
    cmp x16, #2
    b.eq 23f
    cmp x16, #1
    b.eq 24f
    ldrb w8, [x7]
    strb w8, [x6]
    b 30f
22:
    ldr x8, [x7]
    str x8, [x6]
    b 30f
23:
    ldr w8, [x7]
    str w8, [x6]
    b 30f
24:
    ldrh w8, [x7]
    strh w8, [x6]
    b 30f

21:
    cmp x16, #3
    b.eq 25f
    cmp x16, #2
    b.eq 26f
    cmp x16, #1
    b.eq 27f
    ldrb w8, [x6]
    strb w8, [x7]
    mov x8, #0
    str x8, [x7, #8]
    b 30f
25:
    ldr x8, [x6]
    str x8, [x7]
    mov x8, #0
    str x8, [x7, #8]
    b 30f
26:
    ldr w8, [x6]
    str w8, [x7]
    mov x8, #0
    str x8, [x7, #8]
    b 30f
27:
    ldrh w8, [x6]
    strh w8, [x7]
    mov x8, #0
    str x8, [x7, #8]
    b 30f

30:
    ldr x17, [sp, #40]             // guest_pc
    add x1, x17, #4
    ldr x17, [sp, #32]
    str x1, [x17, #40]             // rt->resume_pc
    ldr x17, [x17, #0]
    str x1, [x17, #272]
    mov x0, #-1                    // INT_NONE
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #64
    ret
9:
    ldr x0, [sp, #32]
    ldr x1, [sp, #40]
    ldr x2, [sp, #48]
    bl _arm64_jit_c_simd_ldst_imm_unsigned
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #64
    ret

.globl _arm64_jit_helper_simd_ldst_imm_unsigned_live
_arm64_jit_helper_simd_ldst_imm_unsigned_live:
    // x0 = rt*, x1 = packed(guest_pc<<32 | insn), x2 = guest_addr
    // Live-cache path for plain SIMD unsigned-immediate load/store:
    // uses live guest address, keeps GPR cache intact on fast success,
    // still uses cpu->fp[] as the vector backing store.
    stp x29, x30, [sp, #-176]!
    stp x16, x17, [sp, #16]
    stp x4, x5, [sp, #32]
    stp x6, x7, [sp, #48]
    stp x8, x9, [sp, #64]
    stp x10, x11, [sp, #80]
    stp x12, x13, [sp, #96]
    stp x14, x15, [sp, #112]
    stp x0, x1, [sp, #128]
    mrs x16, nzcv
    str x16, [sp, #144]

    ldr w16, [sp, #136]            // insn low 32
    ubfx x17, x16, #30, #2         // size
    ubfx x0, x16, #22, #2          // opc
    cmp x17, #0
    b.ne 0f
    tst x0, #0x2
    b.eq 0f                        // Bt
    mov x3, #16                    // Qt width
    b 1f
0:
    cmp x0, #0x2
    b.hs 90f
    mov x3, #1
    lsl x3, x3, x17               // 1/2/4/8
1:
    and x0, x2, #0xfff
    sub x1, x3, #1
    add x0, x0, x1
    cmp x0, #0xfff
    bhi 90f

    ldr w17, [sp, #136]            // insn low 32
    ubfx x0, x17, #22, #2         // opc
    tst x0, #0x1                  // L
    b.ne 2f                       // load
    JIT_TLB_LOOKUP_WRITE_HIT x2, x1, x0, x16, 90f
    b 3f
2:
    JIT_TLB_LOOKUP_READ_HIT x2, x1, x0, x16, 90f
3:
    and x16, x2, #0xfff
    add x1, x1, x16

    ubfx x0, x17, #0, #5          // rt
    ldr x16, [sp, #128]
    ldr x16, [x16, #0]            // cpu*
    add x0, x16, x0, lsl #4
    add x0, x0, #288              // &cpu->fp[rt]

    ubfx x16, x17, #30, #2        // size
    ubfx x17, x17, #22, #2        // opc
    cmp x16, #0
    b.ne 4f
    tst x17, #0x2
    b.eq 4f
    tst x17, #0x1
    b.ne 10f
    ldr x2, [x0]
    ldr x3, [x0, #8]
    str x2, [x1]
    str x3, [x1, #8]
    b 20f
10:
    ldr x2, [x1]
    ldr x3, [x1, #8]
    str x2, [x0]
    str x3, [x0, #8]
    b 20f

4:
    tst x17, #0x1
    b.ne 11f
    cmp x16, #3
    b.eq 12f
    cmp x16, #2
    b.eq 13f
    cmp x16, #1
    b.eq 14f
    ldrb w2, [x0]
    strb w2, [x1]
    b 20f
12:
    ldr x2, [x0]
    str x2, [x1]
    b 20f
13:
    ldr w2, [x0]
    str w2, [x1]
    b 20f
14:
    ldrh w2, [x0]
    strh w2, [x1]
    b 20f

11:
    cmp x16, #3
    b.eq 15f
    cmp x16, #2
    b.eq 16f
    cmp x16, #1
    b.eq 17f
    ldrb w2, [x1]
    strb w2, [x0]
    mov x2, #0
    str x2, [x0, #8]
    b 20f
15:
    ldr x2, [x1]
    str x2, [x0]
    mov x2, #0
    str x2, [x0, #8]
    b 20f
16:
    ldr w2, [x1]
    str w2, [x0]
    mov x2, #0
    str x2, [x0, #8]
    b 20f
17:
    ldrh w2, [x1]
    strh w2, [x0]
    mov x2, #0
    str x2, [x0, #8]

20:
    ldr w16, [sp, #136]
    ubfx x17, x16, #22, #2
    tst x17, #0x1
    b.eq 21f
    ubfx x17, x16, #0, #5
    JIT_SIMD_SYNC_CPU_TO_Q x17, x0
21:
    ldr w1, [sp, #140]
    add x1, x1, #4
    ldr x16, [sp, #128]
    str x1, [x16, #40]             // rt->resume_pc
    ldr x16, [x16, #0]
    str x1, [x16, #272]
    mov x0, #-1
    ldr x16, [sp, #144]
    msr nzcv, x16
    ldp x14, x15, [sp, #112]
    ldp x12, x13, [sp, #96]
    ldp x10, x11, [sp, #80]
    ldp x8, x9, [sp, #64]
    ldp x6, x7, [sp, #48]
    ldp x4, x5, [sp, #32]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #176
    ret
90:
    ldr x0, [sp, #128]
    ldr x16, [x0, #24]
    cbz x16, 91f
    blr x16
91:
    ldr x0, [sp, #128]
    ldr w1, [sp, #140]
    ldr w2, [sp, #136]
    bl _arm64_jit_c_simd_ldst_imm_unsigned
    cmp w0, #-1
    b.ne 92f
    ldr x0, [sp, #128]
    ldr x16, [x0, #32]
    cbz x16, 92f
    mov x0, x21
    blr x16
    mov x0, #-1
92:
    cmp w0, #-1
    b.ne 93f
    ldr x16, [sp, #144]
    msr nzcv, x16
    ldp x14, x15, [sp, #112]
    ldp x12, x13, [sp, #96]
    ldp x10, x11, [sp, #80]
    ldp x8, x9, [sp, #64]
    ldp x6, x7, [sp, #48]
    ldp x4, x5, [sp, #32]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #176
    ret
93:
    ldr x16, [sp, #144]
    msr nzcv, x16
    ldp x14, x15, [sp, #112]
    ldp x12, x13, [sp, #96]
    ldp x10, x11, [sp, #80]
    ldp x8, x9, [sp, #64]
    ldp x6, x7, [sp, #48]
    ldp x4, x5, [sp, #32]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #176
    JIT_FRAGMENT_EPILOGUE_AND_RET

.globl _arm64_jit_helper_ldst_imm9
_arm64_jit_helper_ldst_imm9:
    // x0 = rt*, x1 = guest_pc, x2 = insn
    // Fast path for scalar, non-sign-extending, non-vector imm9 forms only.
    stp x29, x30, [sp, #-64]!
    stp x16, x17, [sp, #16]
    stp x0, x1, [sp, #32]
    str x2, [sp, #48]

    ubfx x16, x2, #26, #1          // V
    cbnz x16, 9f
    ubfx x16, x2, #10, #2          // mode
    cmp x16, #2
    b.eq 9f                        // unsupported mode
    ubfx x16, x2, #22, #2          // opc
    cmp x16, #2
    b.hs 9f                        // sign-extending forms stay in C path

    ldr x16, [sp, #32]             // runtime*
    ldr x17, [x16, #0]             // cpu*
    ubfx x1, x2, #5, #5            // rn
    cmp x1, #31
    b.ne 0f
    ldr x1, [x17, #264]            // sp
    b 1f
0:
    add x16, x17, #16
    ldr x1, [x16, x1, lsl #3]
1:
    sbfx x3, x2, #12, #9           // imm9
    ubfx x16, x2, #10, #2          // mode
    cmp x16, #1
    b.ne 2f
    mov x4, x1                     // post-index uses base as access addr
    b 3f
2:
    add x4, x1, x3                 // unscaled / pre-index use base+imm9
3:
    ubfx x16, x2, #30, #2          // size

    and x5, x4, #0xfff
    mov x6, #8
    lsl x6, x6, x16
    sub x6, x6, #1
    add x5, x5, x6
    cmp x5, #0xfff
    bhi 9f

    ldr x5, [sp, #32]              // runtime*
    ldr x5, [x5, #8]               // tlb*
    eor x6, x4, x4, lsr #13
    lsr x6, x6, #12
    and x6, x6, #0x1fff
    add x6, x5, #32
    add x6, x6, x6, lsl #5         // overwritten below
    eor x16, x4, x4, lsr #13
    lsr x16, x16, #12
    and x16, x16, #0x1fff
    add x6, x5, #32
    add x6, x6, x16, lsl #5

    and x16, x4, #0xfffffffffffff000
    ldr x7, [sp, #48]              // insn
    ubfx x7, x7, #22, #2           // opc
    cmp x7, #1
    b.eq 4f                        // load
    ldr x7, [x6, #8]               // entry.page_if_writable
    cmp x7, x16
    b.ne 9f
    str x16, [x5, #8]              // tlb->dirty_page
    b 5f
4:
    ldr x7, [x6, #0]               // entry.page
    cmp x7, x16
    b.ne 9f
5:
    ldr x7, [x6, #16]              // data_minus_addr
    add x7, x7, x4                 // host ptr

    ldr x2, [sp, #48]              // insn
    ubfx x16, x2, #22, #2          // opc
    cmp x16, #1
    b.eq 6f                        // load

    // store fast path
    ubfx x5, x2, #0, #5            // rt
    cmp x5, #31
    b.eq 10f
    ldr x17, [sp, #32]
    ldr x17, [x17, #0]
    add x17, x17, #16
    ldr x6, [x17, x5, lsl #3]
    b 11f
10:
    mov x6, xzr
11:
    ubfx x16, x2, #30, #2
    cmp x16, #3
    b.eq 12f
    cmp x16, #2
    b.eq 13f
    cmp x16, #1
    b.eq 14f
    strb w6, [x7]
    b 15f
12:
    str x6, [x7]
    b 15f
13:
    str w6, [x7]
    b 15f
14:
    strh w6, [x7]
15:
    b 20f

6:
    // load fast path
    ubfx x16, x2, #30, #2
    cmp x16, #3
    b.eq 16f
    cmp x16, #2
    b.eq 17f
    cmp x16, #1
    b.eq 18f
    ldrb w6, [x7]
    b 19f
16:
    ldr x6, [x7]
    b 19f
17:
    ldr w6, [x7]
    b 19f
18:
    ldrh w6, [x7]
19:
    ubfx x5, x2, #0, #5            // rt
    cmp x5, #31
    b.eq 20f
    ldr x17, [sp, #32]
    ldr x17, [x17, #0]
    add x17, x17, #16
    str x6, [x17, x5, lsl #3]

20:
    // writeback for post/pre-indexed forms
    ldr x2, [sp, #48]              // insn
    ubfx x16, x2, #10, #2          // mode
    cbz x16, 21f                   // unscaled: no writeback
    cmp x16, #1
    b.eq 22f                       // post-index
    // pre-index
    mov x6, x4
    b 23f
22:
    add x6, x1, x3
23:
    ubfx x5, x2, #5, #5            // rn
    ldr x17, [sp, #32]
    ldr x17, [x17, #0]
    cmp x5, #31
    b.ne 24f
    str x6, [x17, #264]
    b 21f
24:
    add x17, x17, #16
    str x6, [x17, x5, lsl #3]

21:
    ldr x1, [sp, #40]
    add x1, x1, #4
    ldr x17, [sp, #32]
    str x1, [x17, #40]             // rt->resume_pc
    ldr x17, [x17, #0]
    str x1, [x17, #272]
    mov x0, #-1
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #64
    ret
9:
    ldr x0, [sp, #32]
    ldr x1, [sp, #40]
    ldr x2, [sp, #48]
    bl _arm64_jit_c_ldst_imm9
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #64
    ret

// Slow fallback for imm9 live store helpers.
// Inputs:
//   x1 = packed(guest_pc<<32 | insn)
//   x21 = rt*
// Returns x0 = INT_NONE on success.
// On fault/interrupt, does not return to the family helper; exits the fragment directly.
.globl _arm64_jit_helper_ldst_imm9_store_slow
_arm64_jit_helper_ldst_imm9_store_slow:
    stp x29, x30, [sp, #-16]!
    ldr x17, [x21, #24]            // spill_state_fn
    cbz x17, 1f
    blr x17
1:
    mov x0, x21
    mov w2, w1                     // insn low 32
    lsr x1, x1, #32                // guest_pc
    bl _arm64_jit_c_ldst_imm9
    cmp w0, #-1
    b.eq 2f
    JIT_FRAGMENT_EPILOGUE_AND_RET
2:
    mov x0, #-1
    ldp x29, x30, [sp], #16
    ret

// Slow fallback for imm9 live load helpers.
// Inputs:
//   x1 = packed(guest_pc<<32 | insn)
//   x3 = store_in_x1 flag
//   x21 = rt*
// Returns x0 = INT_NONE on success, and x1 = loaded value when store_in_x1 != 0.
// On fault/interrupt, does not return to the family helper; exits the fragment directly.
.globl _arm64_jit_helper_ldst_imm9_load_slow
_arm64_jit_helper_ldst_imm9_load_slow:
    stp x29, x30, [sp, #-48]!
    str x17, [sp, #16]
    stp x1, x3, [sp, #24]          // packed, store_in_x1
    ldr x17, [x21, #24]            // spill_state_fn
    cbz x17, 3f
    blr x17
3:
    mov x0, x21
    ldr x1, [sp, #24]              // packed
    mov w2, w1                     // insn low 32
    lsr x1, x1, #32                // guest_pc
    bl _arm64_jit_c_ldst_imm9
    cmp w0, #-1
    b.ne 4f
    ldr x3, [sp, #32]              // store_in_x1
    cbz x3, 5f
    ldr x2, [sp, #24]              // packed
    ubfx x17, x2, #0, #5           // rt
    cmp x17, #31
    b.eq 5f
    ldr x0, [x21, #0]              // cpu*
    add x0, x0, #16
    ldr x1, [x0, x17, lsl #3]
5:
    mov x0, #-1
    ldr x17, [sp, #16]
    ldp x29, x30, [sp], #48
    ret
4:
    ldr x17, [sp, #16]
    ldp x29, x30, [sp], #48
    JIT_FRAGMENT_EPILOGUE_AND_RET

.globl _arm64_jit_helper_ldst_imm9_live
_arm64_jit_helper_ldst_imm9_live:
    // x0 = rt*, x1 = packed(guest_pc<<32 | insn), x2 = guest_addr, x3 = value
    // Clean live-cache path only for unscaled scalar store (mode==0, opc==0).
    mov w17, w1                    // insn low 32
    ubfx x0, x17, #26, #1          // V
    cbnz x0, .Limm9_store_cold
    ubfx x0, x17, #10, #2          // mode
    cbnz x0, .Limm9_store_cold
    ubfx x0, x17, #22, #2          // opc
    cbnz x0, .Limm9_store_cold
    ubfx x17, x17, #30, #2         // size

    and x0, x2, #0xfff
    mov x16, #1
    lsl x16, x16, x17
    sub x16, x16, #1
    add x0, x0, x16
    lsr x0, x0, #12
    cbnz x0, .Limm9_store_cold

    JIT_TLB_LOOKUP_WRITE_HIT x2, x0, x16, x17, .Limm9_store_cold
    and x16, x2, #0xfff
    add x16, x0, x16               // host ptr

    ubfx x17, x1, #30, #2          // size
    mov x0, #3
    eor x0, x17, x0
    cbz x0, 11f
    mov x0, #2
    eor x0, x17, x0
    cbz x0, 12f
    mov x0, #1
    eor x0, x17, x0
    cbz x0, 13f
    strb w3, [x16]
    b 14f
11:
    str x3, [x16]
    b 14f
12:
    str w3, [x16]
    b 14f
13:
    strh w3, [x16]
14:
    lsr x16, x1, #32               // guest_pc
    add x16, x16, #4
    str x16, [x21, #40]            // rt->resume_pc
    ldr x17, [x21, #0]             // cpu*
    str x16, [x17, #272]
    mov x0, #-1
    ret

.Limm9_store_cold:
    stp x29, x30, [sp, #-48]!
    stp x1, x3, [sp, #16]
    str x4, [sp, #32]
    str x1, [x21, #64]             // rt->debug0 = packed on helper entry
    mrs x16, nzcv
    mov w17, w1                    // insn low 32
    ubfx x0, x17, #26, #1          // V
    cbnz x0, 90f
    ubfx x0, x17, #10, #2          // mode
    cbnz x0, 90f
    ubfx x0, x17, #22, #2          // opc
    cbnz x0, 90f
    ubfx x17, x17, #30, #2         // size

    and x0, x2, #0xfff
    mov x4, #1
    lsl x4, x4, x17
    sub x4, x4, #1
    add x0, x0, x4
    cmp x0, #0xfff
    bhi 90f

    mov x0, x21
    mov x4, #1
    bl _arm64_jit_mem_lookup_page_live
    ldp x1, x3, [sp, #16]
    and x17, x2, #0xfff
    add x17, x0, x17
    str x1, [x21, #88]             // rt->debug3 = restored packed
    mov w0, w1                     // insn low 32
    ubfx x0, x0, #30, #2          // size

    cmp x0, #3
    b.eq 91f
    cmp x0, #2
    b.eq 92f
    cmp x0, #1
    b.eq 93f
    strb w3, [x17]
    b 94f
91:
    str x3, [x17]
    b 94f
92:
    str w3, [x17]
    b 94f
93:
    strh w3, [x17]
94:
    lsr x17, x1, #32               // guest_pc
    add x17, x17, #4
    str x17, [x21, #40]            // rt->resume_pc
    ldr x0, [x21, #0]              // cpu*
    str x17, [x0, #272]
    mov x0, #-1
    msr nzcv, x16
    ldr x4, [sp, #32]
    ldp x29, x30, [sp], #48
    ret
90:
    ldp x1, x3, [sp, #16]
    bl _arm64_jit_helper_ldst_imm9_store_slow
    msr nzcv, x16
    ldr x4, [sp, #32]
    ldp x29, x30, [sp], #48
    ret

.globl _arm64_jit_helper_ldst_imm9_load_live
_arm64_jit_helper_ldst_imm9_load_live:
    // x0 = rt*, x1 = packed(guest_pc<<32 | insn), x2 = guest_addr, x3 = store_in_x1
    // Clean live-cache path only for unscaled scalar load (mode==0, opc==1).
    mov w17, w1                    // insn low 32
    ubfx x0, x17, #26, #1          // V
    cbnz x0, .Limm9_load_cold
    ubfx x0, x17, #10, #2          // mode
    cbnz x0, .Limm9_load_cold
    ubfx x0, x17, #22, #2          // opc
    mov x16, #1
    eor x0, x0, x16
    cbnz x0, .Limm9_load_cold
    ubfx x17, x17, #30, #2         // size

    and x0, x2, #0xfff
    mov x16, #1
    lsl x16, x16, x17
    sub x16, x16, #1
    add x0, x0, x16
    lsr x0, x0, #12
    cbnz x0, .Limm9_load_cold

    JIT_TLB_LOOKUP_READ_HIT x2, x0, x16, x17, .Limm9_load_cold
    and x16, x2, #0xfff
    add x16, x0, x16               // host ptr

    ubfx x17, x1, #30, #2          // size
    mov x0, #3
    eor x0, x17, x0
    cbz x0, 21f
    mov x0, #2
    eor x0, x17, x0
    cbz x0, 22f
    mov x0, #1
    eor x0, x17, x0
    cbz x0, 23f
    ldrb w0, [x16]
    b 24f
21:
    ldr x0, [x16]
    b 24f
22:
    ldr w0, [x16]
    b 24f
23:
    ldrh w0, [x16]
24:
    lsr x16, x1, #32               // guest_pc
    cbnz x3, 25f
    ubfx x17, x1, #0, #5          // rt
    mov x4, #31
    eor x4, x17, x4
    cbz x4, 26f
    ldr x4, [x21, #0]
    add x4, x4, #16
    str x0, [x4, x17, lsl #3]
    b 26f
25:
    mov x1, x0
26:
    add x16, x16, #4
    str x16, [x21, #40]
    ldr x17, [x21, #0]
    str x16, [x17, #272]
    mov x0, #-1
    ret

.Limm9_load_cold:
    stp x29, x30, [sp, #-48]!
    stp x1, x3, [sp, #16]
    str x4, [sp, #32]
    mrs x16, nzcv
    mov w17, w1                    // insn low 32
    ubfx x0, x17, #26, #1          // V
    cbnz x0, 90f
    ubfx x0, x17, #10, #2          // mode
    cbnz x0, 90f
    ubfx x0, x17, #22, #2          // opc
    cmp x0, #1
    b.ne 90f
    ubfx x17, x17, #30, #2         // size

    and x0, x2, #0xfff
    mov x4, #1
    lsl x4, x4, x17
    sub x4, x4, #1
    add x0, x0, x4
    cmp x0, #0xfff
    bhi 90f

    mov x0, x21
    mov x4, #0
    bl _arm64_jit_mem_lookup_page_live
    ldp x1, x3, [sp, #16]
    and x17, x2, #0xfff
    add x17, x0, x17
    mov w0, w1                     // insn low 32
    ubfx x0, x0, #30, #2          // size

    cmp x0, #3
    b.eq 91f
    cmp x0, #2
    b.eq 92f
    cmp x0, #1
    b.eq 93f
    ldrb w0, [x17]
    b 94f
91:
    ldr x0, [x17]
    b 94f
92:
    ldr w0, [x17]
    b 94f
93:
    ldrh w0, [x17]
94:
    cbnz x3, 95f
    mov w17, w1                    // insn low 32
    ubfx x17, x17, #0, #5          // rt
    cmp x17, #31
    b.eq 96f
    ldr x2, [x21, #0]              // cpu*
    add x2, x2, #16
    str x0, [x2, x17, lsl #3]
    b 96f
95:
    mov x1, x0
96:
    ldr x17, [sp, #16]             // packed
    lsr x17, x17, #32              // guest_pc
    add x17, x17, #4
    str x17, [x21, #40]            // rt->resume_pc
    ldr x0, [x21, #0]
    str x17, [x0, #272]
    mov x0, #-1
    msr nzcv, x16
    ldr x4, [sp, #32]
    ldp x29, x30, [sp], #48
    ret
90:
    ldp x1, x3, [sp, #16]
    bl _arm64_jit_helper_ldst_imm9_load_slow
    msr nzcv, x16
    ldr x4, [sp, #32]
    ldp x29, x30, [sp], #48
    ret

.globl _arm64_jit_helper_ldst_regoff
_arm64_jit_helper_ldst_regoff:
    // x0 = rt*, x1 = guest_pc, x2 = insn
    // Fast path for scalar, option==3, non-sign-extending register-offset forms only.
    stp x29, x30, [sp, #-64]!
    stp x16, x17, [sp, #16]
    stp x0, x1, [sp, #32]
    str x2, [sp, #48]

    ubfx x16, x2, #26, #1          // V
    cbnz x16, 9f
    ubfx x16, x2, #13, #3          // option
    cmp x16, #3
    b.ne 9f
    ubfx x16, x2, #22, #2          // opc
    cmp x16, #2
    b.hs 9f                        // sign-extending forms stay in C path

    ldr x16, [sp, #32]             // runtime*
    ldr x17, [x16, #0]             // cpu*
    ubfx x1, x2, #5, #5            // rn
    cmp x1, #31
    b.ne 0f
    ldr x1, [x17, #264]            // sp
    b 1f
0:
    add x16, x17, #16
    ldr x1, [x16, x1, lsl #3]
1:
    ubfx x3, x2, #16, #5           // rm
    cmp x3, #31
    b.eq 9f                        // keep rm==31 corner cases on C path
    add x16, x17, #16
    ldr x3, [x16, x3, lsl #3]      // offset
    ubfx x16, x2, #12, #1          // S
    cbz x16, 2f
    ubfx x16, x2, #30, #2          // size
    lsl x3, x3, x16
2:
    add x4, x1, x3                 // guest addr
    ubfx x16, x2, #30, #2          // size

    and x5, x4, #0xfff
    mov x6, #8
    lsl x6, x6, x16
    sub x6, x6, #1
    add x5, x5, x6
    cmp x5, #0xfff
    bhi 9f

    ldr x5, [sp, #32]              // runtime*
    ldr x5, [x5, #8]               // tlb*
    eor x6, x4, x4, lsr #13
    lsr x6, x6, #12
    and x6, x6, #0x1fff
    add x6, x5, #32
    add x6, x6, x6, lsl #5         // overwritten below
    eor x16, x4, x4, lsr #13
    lsr x16, x16, #12
    and x16, x16, #0x1fff
    add x6, x5, #32
    add x6, x6, x16, lsl #5

    and x16, x4, #0xfffffffffffff000
    ldr x7, [sp, #48]              // insn
    ubfx x7, x7, #22, #2           // opc
    cmp x7, #1
    b.eq 3f                        // load
    ldr x7, [x6, #8]               // entry.page_if_writable
    cmp x7, x16
    b.ne 9f
    str x16, [x5, #8]              // tlb->dirty_page
    b 4f
3:
    ldr x7, [x6, #0]               // entry.page
    cmp x7, x16
    b.ne 9f
4:
    ldr x7, [x6, #16]
    add x7, x7, x4                 // host ptr

    ldr x2, [sp, #48]              // insn
    ubfx x16, x2, #22, #2          // opc
    cmp x16, #1
    b.eq 5f                        // load

    ubfx x5, x2, #0, #5            // rt
    cmp x5, #31
    b.eq 10f
    ldr x17, [sp, #32]
    ldr x17, [x17, #0]
    add x17, x17, #16
    ldr x6, [x17, x5, lsl #3]
    b 11f
10:
    mov x6, xzr
11:
    ubfx x16, x2, #30, #2
    cmp x16, #3
    b.eq 12f
    cmp x16, #2
    b.eq 13f
    cmp x16, #1
    b.eq 14f
    strb w6, [x7]
    b 15f
12:
    str x6, [x7]
    b 15f
13:
    str w6, [x7]
    b 15f
14:
    strh w6, [x7]
15:
    b 20f

5:
    ubfx x16, x2, #30, #2
    cmp x16, #3
    b.eq 16f
    cmp x16, #2
    b.eq 17f
    cmp x16, #1
    b.eq 18f
    ldrb w6, [x7]
    b 19f
16:
    ldr x6, [x7]
    b 19f
17:
    ldr w6, [x7]
    b 19f
18:
    ldrh w6, [x7]
19:
    ubfx x5, x2, #0, #5            // rt
    cmp x5, #31
    b.eq 20f
    ldr x17, [sp, #32]
    ldr x17, [x17, #0]
    add x17, x17, #16
    str x6, [x17, x5, lsl #3]

20:
    add x1, x1, #4
    ldr x17, [sp, #32]
    str x1, [x17, #40]             // rt->resume_pc
    ldr x17, [x17, #0]
    str x1, [x17, #272]
    mov x0, #-1
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #64
    ret
9:
    ldr x0, [sp, #32]
    ldr x1, [sp, #40]
    ldr x2, [sp, #48]
    bl _arm64_jit_c_ldst_regoff
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #64
    ret

.globl _arm64_jit_helper_ldst_regoff_live
_arm64_jit_helper_ldst_regoff_live:
    // x0 = rt*, x1 = packed(guest_pc<<32 | insn), x2 = guest_addr, x3 = value
    // Clean live-cache path only for scalar store, option==3.
    mov w17, w1                    // insn low 32
    ubfx x0, x17, #26, #1         // V
    cbnz x0, .Lregoff_store_cold
    ubfx x0, x17, #13, #3         // option
    mov x16, #3
    eor x0, x0, x16
    cbnz x0, .Lregoff_store_cold
    ubfx x0, x17, #22, #2         // opc
    cbnz x0, .Lregoff_store_cold
    ubfx x17, x17, #30, #2        // size

    and x0, x2, #0xfff
    mov x16, #1
    lsl x16, x16, x17
    sub x16, x16, #1
    add x0, x0, x16
    lsr x0, x0, #12
    cbnz x0, .Lregoff_store_cold

    JIT_TLB_LOOKUP_WRITE_HIT x2, x0, x16, x17, .Lregoff_store_cold
    and x16, x2, #0xfff
    add x16, x0, x16               // host ptr

    ubfx x17, x1, #30, #2
    mov x0, #3
    eor x0, x17, x0
    cbz x0, 31f
    mov x0, #2
    eor x0, x17, x0
    cbz x0, 32f
    mov x0, #1
    eor x0, x17, x0
    cbz x0, 33f
    strb w3, [x16]
    b 34f
31:
    str x3, [x16]
    b 34f
32:
    str w3, [x16]
    b 34f
33:
    strh w3, [x16]
34:
    lsr x16, x1, #32
    add x16, x16, #4
    str x16, [x21, #40]
    ldr x17, [x21, #0]
    str x16, [x17, #272]
    mov x0, #-1
    ret

.Lregoff_store_cold:
    stp x29, x30, [sp, #-176]!
    stp x16, x17, [sp, #16]
    mov x0, x21
    stp x0, x1, [sp, #32]
    stp x2, x3, [sp, #48]
    stp x4, x5, [sp, #64]
    stp x6, x7, [sp, #80]
    stp x8, x9, [sp, #96]
    stp x10, x11, [sp, #112]
    stp x12, x13, [sp, #128]
    stp x14, x15, [sp, #144]
    mrs x16, nzcv
    str x16, [sp, #160]

    ldr w16, [sp, #40]             // insn low 32
    ubfx x17, x16, #26, #1         // V
    cbnz x17, 90f
    ubfx x17, x16, #13, #3         // option
    cmp x17, #3
    b.ne 90f
    ubfx x17, x16, #22, #2         // opc
    cbnz x17, 90f
    ubfx x17, x16, #30, #2         // size

    and x16, x2, #0xfff
    mov x0, #1
    lsl x0, x0, x17
    sub x0, x0, #1
    add x16, x16, x0
    cmp x16, #0xfff
    bhi 90f

    ldr x1, [sp, #40]              // packed
    mov x0, x21
    mov x4, #1
    bl _arm64_jit_mem_lookup_page_live
    ldr x1, [sp, #40]              // packed
    ldr x3, [sp, #56]              // store value
    ldr w16, [sp, #40]             // insn low 32
    ubfx x17, x16, #30, #2         // size
    and x6, x2, #0xfff
    add x6, x0, x6                 // host ptr

    cmp x17, #3
    b.eq 91f
    cmp x17, #2
    b.eq 92f
    cmp x17, #1
    b.eq 93f
    strb w3, [x6]
    b 94f
91:
    str x3, [x6]
    b 94f
92:
    str w3, [x6]
    b 94f
93:
    strh w3, [x6]
94:
    ldr w1, [sp, #44]
    add x1, x1, #4
    ldr x17, [sp, #32]
    str x1, [x17, #40]             // rt->resume_pc
    ldr x17, [x17, #0]
    str x1, [x17, #272]
    ldr x16, [sp, #160]
    msr nzcv, x16
    mov x0, #-1
    ldp x14, x15, [sp, #144]
    ldp x12, x13, [sp, #128]
    ldp x10, x11, [sp, #112]
    ldp x8, x9, [sp, #96]
    ldp x6, x7, [sp, #80]
    ldp x4, x5, [sp, #64]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #176
    ret
90:
    ldr x0, [sp, #32]
    ldr x16, [x0, #24]
    cbz x16, 95f
    blr x16
95:
    ldr x0, [sp, #32]
    ldr w1, [sp, #44]
    ldr w2, [sp, #40]
    bl _arm64_jit_c_ldst_regoff
    cmp w0, #-1
    b.ne 96f
    ldr x0, [sp, #32]
    ldr x16, [x0, #32]
    cbz x16, 96f
    mov x0, x21
    blr x16
    mov x0, #-1
96:
    cmp w0, #-1
    b.ne 97f
    ldr x16, [sp, #160]
    msr nzcv, x16
    ldp x14, x15, [sp, #144]
    ldp x12, x13, [sp, #128]
    ldp x10, x11, [sp, #112]
    ldp x8, x9, [sp, #96]
    ldp x6, x7, [sp, #80]
    ldp x4, x5, [sp, #64]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #176
    ret
97:
    ldp x14, x15, [sp, #144]
    ldp x12, x13, [sp, #128]
    ldp x10, x11, [sp, #112]
    ldp x8, x9, [sp, #96]
    ldp x6, x7, [sp, #80]
    ldp x4, x5, [sp, #64]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #176
    JIT_FRAGMENT_EPILOGUE_AND_RET
    ret

.globl _arm64_jit_helper_ldst_regoff_load_live
_arm64_jit_helper_ldst_regoff_load_live:
    // x0 = rt*, x1 = packed(guest_pc<<32 | insn), x2 = guest_addr, x3 = store_in_x1
    // Clean live-cache path only for scalar load, option==3.
    mov w17, w1                    // insn low 32
    ubfx x0, x17, #26, #1         // V
    cbnz x0, .Lregoff_load_cold
    ubfx x0, x17, #13, #3         // option
    mov x16, #3
    eor x0, x0, x16
    cbnz x0, .Lregoff_load_cold
    ubfx x0, x17, #22, #2         // opc
    mov x16, #1
    eor x0, x0, x16
    cbnz x0, .Lregoff_load_cold
    ubfx x17, x17, #30, #2        // size

    and x0, x2, #0xfff
    mov x16, #1
    lsl x16, x16, x17
    sub x16, x16, #1
    add x0, x0, x16
    lsr x0, x0, #12
    cbnz x0, .Lregoff_load_cold

    JIT_TLB_LOOKUP_READ_HIT x2, x0, x16, x17, .Lregoff_load_cold
    and x16, x2, #0xfff
    add x16, x0, x16               // host ptr

    ubfx x17, x1, #30, #2
    mov x0, #3
    eor x0, x17, x0
    cbz x0, 41f
    mov x0, #2
    eor x0, x17, x0
    cbz x0, 42f
    mov x0, #1
    eor x0, x17, x0
    cbz x0, 43f
    ldrb w0, [x16]
    b 44f
41:
    ldr x0, [x16]
    b 44f
42:
    ldr w0, [x16]
    b 44f
43:
    ldrh w0, [x16]
44:
    lsr x16, x1, #32
    cbnz x3, 45f
    ubfx x17, x1, #0, #5
    mov x3, #31
    eor x3, x17, x3
    cbz x3, 46f
    ldr x3, [x21, #0]
    add x3, x3, #16
    str x0, [x3, x17, lsl #3]
    b 46f
45:
    mov x1, x0
46:
    add x16, x16, #4
    str x16, [x21, #40]
    ldr x17, [x21, #0]
    str x16, [x17, #272]
    mov x0, #-1
    ret

.Lregoff_load_cold:
    stp x29, x30, [sp, #-176]!
    stp x16, x17, [sp, #16]
    mov x0, x21
    stp x0, x1, [sp, #32]
    stp x2, x3, [sp, #48]
    stp x4, x5, [sp, #64]
    stp x6, x7, [sp, #80]
    stp x8, x9, [sp, #96]
    stp x10, x11, [sp, #112]
    stp x12, x13, [sp, #128]
    stp x14, x15, [sp, #144]
    mrs x16, nzcv
    str x16, [sp, #160]

    ldr w16, [sp, #40]             // insn low 32
    ubfx x17, x16, #26, #1         // V
    cbnz x17, 90f
    ubfx x17, x16, #13, #3         // option
    cmp x17, #3
    b.ne 90f
    ubfx x17, x16, #22, #2         // opc
    cmp x17, #1
    b.ne 90f
    ubfx x17, x16, #30, #2         // size

    and x16, x2, #0xfff
    mov x0, #1
    lsl x0, x0, x17
    sub x0, x0, #1
    add x16, x16, x0
    cmp x16, #0xfff
    bhi 90f

    ldr x1, [sp, #40]              // packed
    mov x0, x21
    mov x4, #0
    bl _arm64_jit_mem_lookup_page_live
    ldr x1, [sp, #40]              // packed
    ldr x3, [sp, #56]              // store_in_x1
    ldr w16, [sp, #40]             // insn low 32
    ubfx x17, x16, #30, #2         // size
    and x6, x2, #0xfff
    add x6, x0, x6                 // host ptr

    cmp x17, #3
    b.eq 91f
    cmp x17, #2
    b.eq 92f
    cmp x17, #1
    b.eq 93f
    ldrb w4, [x6]
    b 94f
91:
    ldr x4, [x6]
    b 94f
92:
    ldr w4, [x6]
    b 94f
93:
    ldrh w4, [x6]
94:
    ldr x17, [sp, #40]             // packed
    lsr x17, x17, #32              // guest_pc
    str x17, [x21, #72]            // debug1 = last regoff-load guest_pc
    str x6, [x21, #80]             // debug2 = host ptr
    str x4, [x21, #88]             // debug3 = loaded value
    cbnz x3, 95f
    ldr w16, [sp, #40]
    ubfx x5, x16, #0, #5           // rt
    cmp x5, #31
    b.eq 96f
    ldr x17, [sp, #32]
    ldr x17, [x17, #0]
    add x17, x17, #16
    str x4, [x17, x5, lsl #3]
    b 96f
95:
    mov x1, x4
96:
    ldr w4, [sp, #44]
    add x4, x4, #4
    ldr x17, [sp, #32]
    str x4, [x17, #40]             // rt->resume_pc
    ldr x17, [x17, #0]
    str x4, [x17, #272]
    ldr x16, [sp, #160]
    msr nzcv, x16
    mov x0, #-1
    ldp x14, x15, [sp, #144]
    ldp x12, x13, [sp, #128]
    ldp x10, x11, [sp, #112]
    ldp x8, x9, [sp, #96]
    ldp x6, x7, [sp, #80]
    ldp x4, x5, [sp, #64]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #176
    ret
90:
    ldr x0, [sp, #32]
    ldr x16, [x0, #24]
    cbz x16, 97f
    blr x16
97:
    ldr x0, [sp, #32]
    ldr w1, [sp, #44]
    ldr w2, [sp, #40]
    bl _arm64_jit_c_ldst_regoff
    cmp w0, #-1
    b.ne 98f
    ldr x0, [sp, #32]
    ldr x16, [x0, #32]
    cbz x16, 97f
    mov x0, x21
    blr x16
97:
    mov x0, #-1
    ldr x3, [sp, #56]              // store_in_x1
    cbz x3, 98f
    ldr w2, [sp, #40]              // insn low 32
    ubfx x2, x2, #0, #5            // rt
    cmp x2, #31
    b.eq 98f
    ldr x17, [sp, #32]             // runtime*
    ldr x17, [x17, #0]             // cpu*
    add x17, x17, #16
    ldr x1, [x17, x2, lsl #3]
98:
    cmp w0, #-1
    b.ne 99f
    ldr x16, [sp, #160]
    msr nzcv, x16
    ldp x14, x15, [sp, #144]
    ldp x12, x13, [sp, #128]
    ldp x10, x11, [sp, #112]
    ldp x8, x9, [sp, #96]
    ldp x6, x7, [sp, #80]
    ldp x4, x5, [sp, #64]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #176
    ret
99:
    ldp x14, x15, [sp, #144]
    ldp x12, x13, [sp, #128]
    ldp x10, x11, [sp, #112]
    ldp x8, x9, [sp, #96]
    ldp x6, x7, [sp, #80]
    ldp x4, x5, [sp, #64]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #176
    JIT_FRAGMENT_EPILOGUE_AND_RET
    ret

.globl _arm64_jit_helper_ldst_pair
_arm64_jit_helper_ldst_pair:
    // x0 = rt*, x1 = guest_pc, x2 = insn
    // Fast path for pair load/store:
    //   scalar: V == 0 and opc == 0/2
    //   vector: V == 1 and opc == 0/1/2
    //   mode == 1/2/3 (post/offset/pre)
    //   same-page only
    stp x29, x30, [sp, #-64]!
    stp x16, x17, [sp, #16]
    stp x0, x1, [sp, #32]
    str x2, [sp, #48]

    ubfx x16, x2, #26, #1          // V
    cbz x16, 0f
    ubfx x16, x2, #30, #2          // vector opc
    cmp x16, #2
    b.ls 1f
    b 9f
0:
    ubfx x16, x2, #30, #2          // scalar opc
    cmp x16, #0
    b.eq 1f
    cmp x16, #2
    b.eq 1f
    b 9f
1:
    ubfx x16, x2, #23, #3          // mode
    cmp x16, #1
    b.eq 2f
    cmp x16, #2
    b.eq 2f
    cmp x16, #3
    b.eq 2f
    b 9f
2:
    ubfx x17, x2, #26, #1          // V
    cbnz x17, 3f
    cmp x16, #2                    // scalar fast path only for offset mode
    b.ne 9f
3:
    ldr x16, [sp, #32]             // runtime*
    ldr x17, [x16, #0]             // cpu*
    ubfx x1, x2, #5, #5            // rn
    cmp x1, #31
    b.ne 3f
    ldr x1, [x17, #264]            // sp
    b 4f
3:
    add x16, x17, #16
    ldr x1, [x16, x1, lsl #3]
4:
    sbfx x3, x2, #15, #7           // imm7
    ubfx x16, x2, #26, #1          // V
    cbz x16, 5f
    ubfx x16, x2, #30, #2          // vector opc -> 4/8/16 bytes
    mov x4, #4
    lsl x4, x4, x16                // elem size = 4,8,16
    lsl x3, x3, x16
    lsl x3, x3, #2                 // offset = imm7 * elem size
    b 6f
5:
    ubfx x16, x2, #30, #2          // scalar opc
    cmp x16, #2
    cset x16, eq
    mov x4, #4
    add x4, x4, x16, lsl #2        // elem size = 4 or 8
    lsl x3, x3, x16                // *1 for 32-bit, *2 for 64-bit
    lsl x3, x3, #2                 // *4 total => 4 or 8 byte element scale
6:

    ubfx x16, x2, #23, #3          // mode
    cmp x16, #1
    b.ne 7f
    mov x5, x1                     // post-index uses base as access addr
    b 8f
7:
    add x5, x1, x3                 // pre/offset use base+offset
8:
    // Same-page requirement for full pair width.
    and x6, x5, #0xfff
    lsl x7, x4, #1                 // pair width = elem size * 2
    sub x7, x7, #1
    add x6, x6, x7
    cmp x6, #0xfff
    bhi 9f

    ldr x6, [sp, #32]              // runtime*
    ldr x6, [x6, #8]               // tlb*
    eor x7, x5, x5, lsr #13
    lsr x7, x7, #12
    and x7, x7, #0x1fff
    add x7, x6, #32
    add x7, x7, x7, lsl #5         // overwritten below
    eor x16, x5, x5, lsr #13
    lsr x16, x16, #12
    and x16, x16, #0x1fff
    add x7, x6, #32
    add x7, x7, x16, lsl #5

    and x16, x5, #0xfffffffffffff000
    ldr x2, [sp, #48]              // insn
    ubfx x2, x2, #22, #1           // L
    cbnz x2, 10f                   // load
    ldr x2, [x7, #8]               // entry.page_if_writable
    cmp x2, x16
    b.ne 9f
    str x16, [x6, #8]              // tlb->dirty_page
    b 11f
10:
    ldr x2, [x7, #0]               // entry.page
    cmp x2, x16
    b.ne 9f
11:
    ldr x7, [x7, #16]              // data_minus_addr
    add x7, x7, x5                 // host ptr

    ldr x2, [sp, #48]              // insn
    ubfx x16, x2, #26, #1          // V
    cbnz x16, 30f
    ubfx x16, x2, #22, #1          // L
    cbnz x16, 12f                  // load

    // store fast path
    ubfx x6, x2, #0, #5            // rt
    ubfx x8, x2, #10, #5           // rt2
    ldr x17, [sp, #32]
    ldr x17, [x17, #0]
    add x17, x17, #16
    cmp x6, #31
    b.eq 10f
    ldr x9, [x17, x6, lsl #3]
    b 11f
10:
    mov x9, xzr
11:
    cmp x8, #31
    b.eq 12f
    ldr x10, [x17, x8, lsl #3]
    b 13f
12:
    mov x10, xzr
13:
    ubfx x16, x2, #30, #2          // opc
    cmp x16, #2
    b.eq 13f
    str w9, [x7]
    str w10, [x7, #4]
    b 15f
13:
    str x9, [x7]
    str x10, [x7, #8]
    b 15f

12:
    // load fast path
    ubfx x6, x2, #0, #5            // rt
    ubfx x8, x2, #10, #5           // rt2
    ubfx x16, x2, #30, #2          // opc
    cmp x16, #2
    b.eq 14f
    ldr w9, [x7]
    ldr w10, [x7, #4]
    b 17f
14:
    ldr x9, [x7]
    ldr x10, [x7, #8]
17:
    ldr x17, [sp, #32]
    ldr x17, [x17, #0]
    add x17, x17, #16
    cmp x6, #31
    b.eq 18f
    str x9, [x17, x6, lsl #3]
18:
    cmp x8, #31
    b.eq 15f
    str x10, [x17, x8, lsl #3]
    b 15f

30:
    // vector pair fast path
    ubfx x6, x2, #0, #5            // rt
    ubfx x8, x2, #10, #5           // rt2
    ldr x17, [sp, #32]
    ldr x17, [x17, #0]
    add x9, x17, #288              // &cpu->fp[0]
    add x6, x9, x6, lsl #4         // &cpu->fp[rt]
    add x8, x9, x8, lsl #4         // &cpu->fp[rt2]
    ubfx x16, x2, #30, #2          // opc: 0/1/2 => 4/8/16 bytes
    ubfx x10, x2, #22, #1          // L
    cbnz x10, 31f                  // load
    cmp x16, #2
    b.eq 32f
    cmp x16, #1
    b.eq 33f
    ldr w10, [x6]
    ldr w11, [x8]
    str w10, [x7]
    str w11, [x7, #4]
    b 15f
32:
    ldr x10, [x6]
    ldr x11, [x6, #8]
    ldr x12, [x8]
    ldr x13, [x8, #8]
    str x10, [x7]
    str x11, [x7, #8]
    str x12, [x7, #16]
    str x13, [x7, #24]
    b 15f
33:
    ldr x10, [x6]
    ldr x11, [x8]
    str x10, [x7]
    str x11, [x7, #8]
    b 15f

31:
    cmp x16, #2
    b.eq 34f
    cmp x16, #1
    b.eq 35f
    ldr w10, [x7]
    ldr w11, [x7, #4]
    str w10, [x6]
    mov x10, #0
    str x10, [x6, #8]
    str w11, [x8]
    str x10, [x8, #8]
    b 15f
34:
    ldr x10, [x7]
    ldr x11, [x7, #8]
    ldr x12, [x7, #16]
    ldr x13, [x7, #24]
    str x10, [x6]
    str x11, [x6, #8]
    str x12, [x8]
    str x13, [x8, #8]
    b 15f
35:
    ldr x10, [x7]
    ldr x11, [x7, #8]
    str x10, [x6]
    mov x10, #0
    str x10, [x6, #8]
    str x11, [x8]
    str x10, [x8, #8]

15:
    // writeback for post/pre-indexed forms
    ldr x2, [sp, #48]
    ubfx x16, x2, #23, #3          // mode
    cmp x16, #2
    b.eq 19f                       // offset: no writeback
    cmp x16, #1
    b.eq 20f                       // post-index
    // pre-index
    mov x6, x5
    b 21f
20:
    add x6, x1, x3
21:
    ubfx x8, x2, #5, #5            // rn
    ldr x17, [sp, #32]
    ldr x17, [x17, #0]
    cmp x8, #31
    b.ne 22f
    str x6, [x17, #264]
    b 19f
22:
    add x17, x17, #16
    str x6, [x17, x8, lsl #3]

19:
    ldr x1, [sp, #40]              // guest_pc
    add x1, x1, #4
    ldr x17, [sp, #32]
    str x1, [x17, #40]             // rt->resume_pc
    ldr x17, [x17, #0]
    str x1, [x17, #272]
    mov x0, #-1                    // INT_NONE
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #64
    ret
9:
    ldr x0, [sp, #32]
    ldr x1, [sp, #40]
    ldr x2, [sp, #48]
    bl _arm64_jit_c_ldst_pair
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #64
    ret

.globl _arm64_jit_helper_ldst_pair_load_live
_arm64_jit_helper_ldst_pair_load_live:
    // x0 = rt*, x1 = packed(guest_pc<<32 | insn), x2 = guest_addr, x3 = return_in_x1_x2
    // Clean live-cache path only for scalar offset pair loads (V==0, L==1, mode==2, opc==0/2).
    stp x29, x30, [sp, #-176]!
    stp x16, x17, [sp, #16]
    stp x0, x1, [sp, #32]
    stp x2, x3, [sp, #48]
    stp x4, x5, [sp, #64]
    stp x6, x7, [sp, #80]
    stp x8, x9, [sp, #96]
    stp x10, x11, [sp, #112]
    stp x12, x13, [sp, #128]
    stp x14, x15, [sp, #144]
    mrs x16, nzcv
    str x16, [sp, #160]

    ldr w16, [sp, #40]             // insn low 32
    ubfx x17, x16, #26, #1         // V
    cbnz x17, 90f
    ubfx x17, x16, #22, #1         // L
    cbz x17, 90f
    ubfx x17, x16, #23, #3         // mode
    cmp x17, #2
    b.ne 90f
    ubfx x17, x16, #30, #2         // opc
    cmp x17, #0
    b.eq 1f
    cmp x17, #2
    b.ne 90f
1:
    mov x0, #4
    cmp x17, #2
    cinc x0, x0, eq                // x0 = 5 when 64-bit? no, replaced below
    cmp x17, #2
    b.ne 2f
    mov x0, #8
    b 3f
2:
    mov x0, #4
3:
    and x1, x2, #0xfff
    lsl x3, x0, #1                 // pair width
    sub x3, x3, #1
    add x1, x1, x3
    cmp x1, #0xfff
    bhi 90f

    JIT_TLB_LOOKUP_READ_HIT x2, x0, x1, x17, 90f
    ldr x1, [sp, #40]              // packed
    ldr x3, [sp, #56]              // return_in_x1_x2
    and x17, x2, #0xfff
    add x17, x0, x17               // host ptr

    ldr w16, [sp, #40]
    ubfx x16, x16, #30, #2         // opc
    cmp x16, #2
    b.eq 10f
    ldr w0, [x17]
    ldr w2, [x17, #4]
    b 11f
10:
    ldr x0, [x17]
    ldr x2, [x17, #8]
11:
    cbnz x3, 12f
    ldr w9, [sp, #40]              // insn low 32
    ubfx x1, x9, #0, #5            // rt
    ubfx x3, x9, #10, #5           // rt2
    ldr x17, [sp, #32]
    ldr x17, [x17, #0]
    add x17, x17, #16
    cmp x1, #31
    b.eq 13f
    str x0, [x17, x1, lsl #3]
13:
    cmp x3, #31
    b.eq 14f
    str x2, [x17, x3, lsl #3]
    b 14f
12:
    mov x1, x0
14:
    ldr w0, [sp, #44]
    add x0, x0, #4
    ldr x17, [sp, #32]
    str x0, [x17, #40]             // rt->resume_pc
    ldr x17, [x17, #0]
    str x0, [x17, #272]
    ldr x16, [sp, #160]
    msr nzcv, x16
    mov x0, #-1
    ldp x14, x15, [sp, #144]
    ldp x12, x13, [sp, #128]
    ldp x10, x11, [sp, #112]
    ldp x8, x9, [sp, #96]
    ldp x6, x7, [sp, #80]
    ldp x4, x5, [sp, #64]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #176
    ret
90:
    ldr x0, [sp, #32]
    ldr x16, [x0, #24]
    cbz x16, 97f
    blr x16
97:
    ldr x0, [sp, #32]
    ldr w1, [sp, #44]
    ldr w2, [sp, #40]
    bl _arm64_jit_c_ldst_pair
    cmp w0, #-1
    b.ne 98f
    ldr x0, [sp, #32]
    ldr x16, [x0, #32]
    cbz x16, 98f
    mov x0, x21
    blr x16
    mov x0, #-1
    ldr x3, [sp, #56]              // return_in_x1_x2
    cbz x3, 98f
    ldr w9, [sp, #40]              // insn low 32
    ubfx x5, x9, #0, #5            // rt
    ubfx x6, x9, #10, #5           // rt2
    ldr x17, [sp, #32]             // runtime*
    ldr x17, [x17, #0]             // cpu*
    add x17, x17, #16
    cmp x5, #31
    b.eq 101f
    ldr x1, [x17, x5, lsl #3]
    b 102f
101:
    mov x1, xzr
102:
    cmp x6, #31
    b.eq 103f
    ldr x2, [x17, x6, lsl #3]
    b 98f
103:
    mov x2, xzr
98:
    cmp w0, #-1
    b.ne 99f
    ldr x16, [sp, #160]
    msr nzcv, x16
99:
    ldp x14, x15, [sp, #144]
    ldp x12, x13, [sp, #128]
    ldp x10, x11, [sp, #112]
    ldp x8, x9, [sp, #96]
    ldp x6, x7, [sp, #80]
    ldp x4, x5, [sp, #64]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #176
    ret

.globl _arm64_jit_helper_ldst_pair_store_live
_arm64_jit_helper_ldst_pair_store_live:
    // x0 = rt*, x1 = packed(guest_pc<<32 | insn), x2 = guest_addr, x3 = val0, x16 = val1
    // Clean live-cache path only for scalar offset pair stores (V==0, L==0, mode==2, opc==0/2).
    stp x29, x30, [sp, #-176]!
    stp x16, x17, [sp, #16]
    stp x0, x1, [sp, #32]
    stp x2, x3, [sp, #48]
    stp x4, x5, [sp, #64]
    stp x6, x7, [sp, #80]
    stp x8, x9, [sp, #96]
    stp x10, x11, [sp, #112]
    stp x12, x13, [sp, #128]
    stp x14, x15, [sp, #144]
    mrs x16, nzcv
    str x16, [sp, #160]

    ldr w16, [sp, #40]             // insn low 32
    ubfx x17, x16, #26, #1         // V
    cbnz x17, 90f
    ubfx x17, x16, #22, #1         // L
    cbnz x17, 90f
    ubfx x17, x16, #23, #3         // mode
    cmp x17, #2
    b.ne 90f
    ubfx x17, x16, #30, #2         // opc
    cmp x17, #0
    b.eq 1f
    cmp x17, #2
    b.ne 90f
1:
    mov x0, #4
    cmp x17, #2
    b.ne 2f
    mov x0, #8
2:
    and x1, x2, #0xfff
    lsl x3, x0, #1                 // pair width
    sub x3, x3, #1
    add x1, x1, x3
    cmp x1, #0xfff
    bhi 90f

    JIT_TLB_LOOKUP_WRITE_HIT x2, x0, x1, x17, 90f
    ldr x1, [sp, #40]              // packed
    ldr x3, [sp, #56]              // val0
    and x17, x2, #0xfff
    add x17, x0, x17               // host ptr

    ldr w16, [sp, #40]
    ubfx x16, x16, #30, #2         // opc
    ldr x8, [sp, #16]              // saved incoming val1 from x16
    cmp x16, #2
    b.eq 10f
    str w3, [x17]
    str w8, [x17, #4]
    b 11f
10:
    str x3, [x17]
    str x8, [x17, #8]
11:
    ldr w1, [sp, #44]
    add x1, x1, #4
    ldr x17, [sp, #32]
    str x1, [x17, #40]             // rt->resume_pc
    ldr x17, [x17, #0]
    str x1, [x17, #272]
    ldr x16, [sp, #160]
    msr nzcv, x16
    mov x0, #-1
    ldp x14, x15, [sp, #144]
    ldp x12, x13, [sp, #128]
    ldp x10, x11, [sp, #112]
    ldp x8, x9, [sp, #96]
    ldp x6, x7, [sp, #80]
    ldp x4, x5, [sp, #64]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #176
    ret
90:
    ldr x0, [sp, #32]
    ldr x16, [x0, #24]
    cbz x16, 96f
    blr x16
96:
    ldr x0, [sp, #32]
    ldr w1, [sp, #44]
    ldr w2, [sp, #40]
    bl _arm64_jit_c_ldst_pair
    cmp w0, #-1
    b.ne 97f
    ldr x0, [sp, #32]
    ldr x16, [x0, #32]
    cbz x16, 97f
    mov x0, x21
    blr x16
    mov x0, #-1
97:
    cmp w0, #-1
    b.ne 98f
    ldr x16, [sp, #160]
    msr nzcv, x16
98:
    ldp x14, x15, [sp, #144]
    ldp x12, x13, [sp, #128]
    ldp x10, x11, [sp, #112]
    ldp x8, x9, [sp, #96]
    ldp x6, x7, [sp, #80]
    ldp x4, x5, [sp, #64]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #176
    ret

.globl _arm64_jit_helper_ldst_pair_vec_live
_arm64_jit_helper_ldst_pair_vec_live:
    // x0 = rt*, x1 = packed(guest_pc<<32 | insn), x2 = guest_addr
    // Live-cache path for vector offset pair load/store (V==1, mode==2, opc<=2).
    // Uses live guest address, keeps GPR cache intact on fast success, and
    // still uses cpu->fp[] as the vector backing store.
    stp x29, x30, [sp, #-128]!
    stp x16, x17, [sp, #16]
    stp x4, x5, [sp, #32]
    stp x6, x7, [sp, #48]
    stp x8, x9, [sp, #64]
    stp x10, x11, [sp, #80]
    stp x0, x1, [sp, #96]
    mrs x16, nzcv
    str x16, [sp, #112]

    ldr w16, [sp, #104]            // insn low 32
    ubfx x17, x16, #26, #1         // V
    cbz x17, 90f
    ubfx x17, x16, #23, #3         // mode
    cmp x17, #2
    b.ne 90f
    ubfx x17, x16, #30, #2         // opc
    cmp x17, #2
    b.hi 90f
    cmp x17, #2
    b.eq 90f                       // keep Q pair forms on the C path for now

    mov x3, #4
    lsl x3, x3, x17                // elem size = 4/8/16
    and x4, x2, #0xfff
    lsl x5, x3, #1                 // pair width
    sub x5, x5, #1
    add x4, x4, x5
    cmp x4, #0xfff
    bhi 90f

    ldr w17, [sp, #104]
    ubfx x4, x17, #22, #1          // L -> access_kind
    ldr x1, [sp, #104]             // packed
    mov x0, x21
    bl _arm64_jit_mem_lookup_page_live
    and x5, x2, #0xfff
    add x5, x0, x5                 // host ptr

    ldr w17, [sp, #104]            // insn low 32
    ubfx x6, x17, #0, #5           // rt
    ubfx x7, x17, #10, #5          // rt2
    ldr x16, [sp, #96]
    ldr x16, [x16, #0]             // cpu*
    add x8, x16, #288              // &cpu->fp[0]
    add x6, x8, x6, lsl #4
    add x7, x8, x7, lsl #4
    ubfx x16, x17, #30, #2         // opc
    ubfx x8, x17, #22, #1          // L
    cbnz x8, 10f
    cmp x16, #2
    b.eq 11f
    cmp x16, #1
    b.eq 12f
    ldr w8, [x6]
    ldr w9, [x7]
    str w8, [x5]
    str w9, [x5, #4]
    b 20f
11:
    ldr x8, [x6]
    ldr x9, [x6, #8]
    ldr x10, [x7]
    ldr x11, [x7, #8]
    str x8, [x5]
    str x9, [x5, #8]
    str x10, [x5, #16]
    str x11, [x5, #24]
    b 20f
12:
    ldr x8, [x6]
    ldr x9, [x7]
    str x8, [x5]
    str x9, [x5, #8]
    b 20f

10:
    cmp x16, #2
    b.eq 13f
    cmp x16, #1
    b.eq 14f
    ldr w8, [x5]
    ldr w9, [x5, #4]
    str w8, [x6]
    mov x8, #0
    str x8, [x6, #8]
    str w9, [x7]
    str x8, [x7, #8]
    b 20f
13:
    ldr x8, [x5]
    ldr x9, [x5, #8]
    ldr x10, [x5, #16]
    ldr x11, [x5, #24]
    str x8, [x6]
    str x9, [x6, #8]
    str x10, [x7]
    str x11, [x7, #8]
    b 20f
14:
    ldr x8, [x5]
    ldr x9, [x5, #8]
    str x8, [x6]
    mov x8, #0
    str x8, [x6, #8]
    str x9, [x7]
    str x8, [x7, #8]

20:
    ldr w1, [sp, #108]
    add x1, x1, #4
    ldr x16, [sp, #96]
    str x1, [x16, #40]             // rt->resume_pc
    ldr x16, [x16, #0]
    str x1, [x16, #272]
    mov x0, #-1
    ldr x16, [sp, #112]
    msr nzcv, x16
    ldp x10, x11, [sp, #80]
    ldp x8, x9, [sp, #64]
    ldp x6, x7, [sp, #48]
    ldp x4, x5, [sp, #32]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #128
    ret
90:
    ldr x16, [sp, #112]
    msr nzcv, x16
    ldp x10, x11, [sp, #80]
    ldp x8, x9, [sp, #64]
    ldp x6, x7, [sp, #48]
    ldp x4, x5, [sp, #32]
    ldr x0, [sp, #96]
    ldr x16, [x0, #24]
    cbz x16, 91f
    blr x16
91:
    ldr x0, [sp, #96]
    ldr w1, [sp, #108]
    ldr w2, [sp, #104]
    bl _arm64_jit_c_ldst_pair
    cmp w0, #-1
    b.ne 92f
    ldr x0, [sp, #96]
    ldr x16, [x0, #32]
    cbz x16, 93f
    mov x0, x21
    blr x16
93:
    mov x0, #-1
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #128
    ret
92:
    ldp x10, x11, [sp, #80]
    ldp x8, x9, [sp, #64]
    ldp x6, x7, [sp, #48]
    ldp x4, x5, [sp, #32]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #128
    JIT_FRAGMENT_EPILOGUE_AND_RET

JITABI_WRAP1 _arm64_jit_helper_unsupported_jitabi, _arm64_jit_helper_unsupported
JITABI_WRAP1 _arm64_jit_helper_syscall_jitabi, _arm64_jit_helper_syscall
.globl _arm64_jit_helper_verify_trap_jitabi
_arm64_jit_helper_verify_trap_jitabi:
    stp x29, x30, [sp, #-48]!
    stp x16, x17, [sp, #16]
    stp x0, x1, [sp, #32]
    mov x16, x0
    ldr x17, [x16, #24]
    cbz x17, 1f
    blr x17
1:
    ldp x0, x1, [sp, #32]
    bl _arm64_jit_helper_verify_trap
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #48
    ret
JITABI_WRAP1 _arm64_jit_helper_dispatch_jitabi, _arm64_jit_helper_dispatch
JITABI_WRAP1 _arm64_jit_helper_branch_link_jitabi, _arm64_jit_helper_branch_link
JITABI_WRAP2 _arm64_jit_helper_mrs_sysreg_jitabi, _arm64_jit_helper_mrs_sysreg
JITABI_WRAP2 _arm64_jit_helper_ldr_imm_unsigned_jitabi, _arm64_jit_helper_ldr_imm_unsigned
JITABI_WRAP1 _arm64_jit_helper_str_imm_unsigned_jitabi, _arm64_jit_helper_str_imm_unsigned
JITABI_WRAP2 _arm64_jit_helper_mrs_tpidr_jitabi, _arm64_jit_helper_mrs_tpidr
JITABI_WRAP2 _arm64_jit_helper_msr_tpidr_jitabi, _arm64_jit_helper_msr_tpidr
JITABI_WRAP2 _arm64_jit_helper_msr_daif_jitabi, _arm64_jit_helper_msr_daif
JITABI_WRAP2 _arm64_jit_helper_exec_dp_imm_jitabi, _arm64_jit_helper_exec_dp_imm
JITABI_WRAP2 _arm64_jit_helper_exec_dp_reg_jitabi, _arm64_jit_helper_exec_dp_reg
JITABI_WRAP2 _arm64_jit_helper_ldst_excl_jitabi, _arm64_jit_c_ldst_excl
JITABI_WRAP2_SUCCESS _arm64_jit_helper_ldst_excl_success_jitabi, _arm64_jit_c_ldst_excl
JITABI_WRAP2 _arm64_jit_helper_cbz_cbnz_jitabi, _arm64_jit_helper_cbz_cbnz
JITABI_WRAP2 _arm64_jit_helper_b_cond_jitabi, _arm64_jit_helper_b_cond
JITABI_WRAP2 _arm64_jit_helper_tbz_tbnz_jitabi, _arm64_jit_helper_tbz_tbnz
JITABI_WRAP2 _arm64_jit_helper_branch_reg_jitabi, _arm64_jit_helper_branch_reg
.globl _arm64_jit_helper_branch_reg_fast_jitabi
_arm64_jit_helper_branch_reg_fast_jitabi:
    stp x29, x30, [sp, #-80]!
    stp x16, x17, [sp, #16]
    stp x0, x1, [sp, #32]
    stp x2, x3, [sp, #48]
    str x4, [sp, #64]

    // Experimental opt-in branch-reg resolver. Falls back to the normal C
    // helper path on any miss; success tail-branches inside the same fragment.
    mrs x4, nzcv
    str x4, [sp, #72]
    ubfx x4, x2, #21, #4        // branch-register opc
    cmp x4, #2                  // RET only for now
    b.ne 2f
    ldr x16, [x0, #16]          // rt->block
    cbz x16, 2f
    ldr x17, [x16]              // block->start_pc
    cmp x3, x17
    b.lo 2f
    ldr x17, [x16, #8]          // block->end_pc
    cmp x3, x17
    b.hs 2f
    tst x3, #3
    b.ne 2f
    ldr w17, [x16, #24]         // block->insn_count
    add x4, x16, #32            // block->insn_pcs
0:
    cbz x17, 2f
    ldr x30, [x4], #8
    eor x30, x30, x3
    cbz x30, 1f
    sub x17, x17, #1
    b 0b
1:
    ldr w30, [x16, #24]         // block->insn_count
    sub x4, x30, x17            // matched index
    mov x17, #0x9164            // block->insn_host_offsets
    add x17, x16, x17
    ldr w17, [x17, x4, lsl #2]
    cmn w17, #1                 // UINT32_MAX is invalid
    b.eq 2f
    mov x4, #0x9138             // block->code_rx
    ldr x16, [x16, x4]
    add x3, x16, x17
    ldr x4, [sp, #72]
    msr nzcv, x4
    ldr x4, [sp, #64]
    ldr x2, [sp, #48]
    ldp x0, x1, [sp, #32]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #80
    br x3

2:
    // The miss path calls the fragment spill snippet. Restore any live JIT
    // state touched by this probe before spilling cached guest registers.
    ldr x4, [sp, #72]
    msr nzcv, x4
    ldr x4, [sp, #64]
    mov x16, x0
    ldr x17, [x16, #24]
    cbz x17, 3f
    blr x17
3:
    ldp x0, x1, [sp, #32]
    ldr x2, [sp, #48]
    bl _arm64_jit_helper_branch_reg
    add w17, w0, #1
    cbnz w17, 4f
    ldr x16, [sp, #32]
    ldr x17, [x16, #32]
    cbz x17, 4f
    str x0, [sp, #56]
    mov x0, x16
    blr x17
    ldr x0, [sp, #56]
4:
    ldr x4, [sp, #64]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #80
    ret
JITABI_WRAP2 _arm64_jit_helper_ldst_imm9_jitabi, _arm64_jit_helper_ldst_imm9
JITABI_WRAP2_SUCCESS _arm64_jit_helper_ldst_imm9_success_jitabi, _arm64_jit_helper_ldst_imm9
JITABI_WRAP2 _arm64_jit_helper_ldst_regoff_jitabi, _arm64_jit_helper_ldst_regoff
JITABI_WRAP2_SUCCESS _arm64_jit_helper_ldst_regoff_success_jitabi, _arm64_jit_helper_ldst_regoff
JITABI_WRAP2 _arm64_jit_helper_ldst_pair_jitabi, _arm64_jit_helper_ldst_pair
JITABI_WRAP2_SUCCESS _arm64_jit_helper_ldst_pair_success_jitabi, _arm64_jit_helper_ldst_pair
JITABI_WRAP2 _arm64_jit_helper_simd_ldst_imm_unsigned_jitabi, _arm64_jit_helper_simd_ldst_imm_unsigned
JITABI_WRAP2_SUCCESS _arm64_jit_helper_simd_ldst_multi_success_jitabi, _arm64_jit_c_simd_ldst_multi
JITABI_WRAP2_SUCCESS _arm64_jit_helper_simd_ldst_imm_unsigned_success_jitabi, _arm64_jit_helper_simd_ldst_imm_unsigned
JITABI_WRAP2_SUCCESS _arm64_jit_helper_ldr_imm_unsigned_success_jitabi, _arm64_jit_helper_ldr_imm_unsigned
JITABI_WRAP1_SUCCESS _arm64_jit_helper_str_imm_unsigned_success_jitabi, _arm64_jit_helper_str_imm_unsigned

#endif
