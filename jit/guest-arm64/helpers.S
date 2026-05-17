#if defined(GUEST_ARM64) && defined(__aarch64__)

.text
.align 2

// JIT-private fast-helper ABI wrappers.
// Stable across these helpers:
//   x4-x15, x19-x29, v0-v31, NZCV, FPCR, FPSR
// Clobberable:
//   x0-x3, x16-x17, x30
//
// For now these wrappers tail into the existing C semantic helpers while
// preserving the tighter JIT-private register contract. This gives the JIT a
// stable helper ABI now, and later the same entry points can inline TLB-hit
// logic directly in assembly without changing the emitter.

// Offsets used below:
// arm64_jit_runtime:
//   cpu = 0
//   tlb = 8
//   spill_state_fn = 24
//   reload_state_fn = 32
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

.macro JITABI_WRAP1 name, target
.globl \name
\name:
    stp x29, x30, [sp, #-80]!
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
    str x0, [sp, #64]
    mov x0, x16
    blr x17
    ldr x0, [sp, #64]
2:
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #80
    ret
.endm

.macro JITABI_WRAP2 name, target
.globl \name
\name:
    stp x29, x30, [sp, #-80]!
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
    str x0, [sp, #64]
    mov x0, x16
    blr x17
    ldr x0, [sp, #64]
2:
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #80
    ret
.endm

.macro JITABI_WRAP3 name, target
.globl \name
\name:
    stp x29, x30, [sp, #-80]!
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
    str x0, [sp, #64]
    mov x0, x16
    blr x17
    ldr x0, [sp, #64]
2:
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #80
    ret
.endm

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
    stp x29, x30, [sp, #-112]!
    stp x16, x17, [sp, #16]
    stp x4, x5, [sp, #32]
    stp x0, x1, [sp, #48]
    stp x2, x3, [sp, #64]
    mrs x17, nzcv
    str x17, [sp, #80]

    ubfx x17, x1, #24, #2          // load_mode
    ubfx x16, x1, #10, #2          // size_shift
    cmp x17, #3
    b.eq 90f                       // sign-extending byte/half loads -> fallback
    cmp x17, #2
    b.ne 0f
    cmp x16, #2
    b.ne 90f                       // only LDRSW is valid for load_mode==2
0:
    and x4, x2, #0xfff
    mov x5, #8
    lsl x5, x5, x16
    sub x5, x5, #1
    add x4, x4, x5
    cmp x4, #0xfff
    bhi 90f

    ldr x4, [x0, #8]               // tlb*
    eor x5, x2, x2, lsr #13
    lsr x5, x5, #12
    and x5, x5, #0x1fff
    add x5, x4, #32
    add x5, x5, x5, lsl #5         // overwritten below
    eor x16, x2, x2, lsr #13
    lsr x16, x16, #12
    and x16, x16, #0x1fff
    add x5, x4, #32
    add x5, x5, x16, lsl #5        // tlb entry*
    and x16, x2, #0xfffffffffffff000
    ldr x4, [x5, #0]               // entry.page
    cmp x4, x16
    b.ne 90f
    ldr x5, [x5, #16]              // data_minus_addr
    add x5, x5, x2                 // host ptr

    ldr x1, [sp, #56]              // packed1
    ubfx x16, x1, #10, #2          // size_shift
    ldr x17, [sp, #56]             // packed1
    ubfx x17, x17, #24, #2         // load_mode
    cmp x17, #2
    b.eq 98f
    cmp x16, #3
    b.eq 91f
    cmp x16, #2
    b.eq 92f
    cmp x16, #1
    b.eq 93f
    ldrb w4, [x5]
    b 94f
91:
    ldr x4, [x5]
    b 94f
92:
    ldr w4, [x5]
    b 94f
98:
    ldr w4, [x5]
    sxtw x4, w4
    b 94f
93:
    ldrh w4, [x5]
94:
    // For cached-destination continuations, return the architectural load
    // result in x1 so the emitted move into the mapped host reg can keep the
    // cache authoritative without round-tripping through cpu_state.
    // Zero-extend all non-sign-extending scalar loads here; these live fast
    // paths only cover the clean non-sign-extending subset.
    ldr x3, [sp, #72]              // store_in_x1
    cbnz x3, 95f
    ubfx x5, x1, #0, #5            // rt
    cmp x5, #31
    b.eq 96f
    ldr x17, [sp, #48]             // runtime*
    ldr x17, [x17, #0]             // cpu*
    add x17, x17, #16
    str x4, [x17, x5, lsl #3]
    b 96f
95:
    mov x1, x4
    // debug trace hook omitted in assembly; use runtime-side verifier signal path
96:
    ldr w4, [sp, #60]              // guest_pc
    add x4, x4, #4
    ldr x17, [sp, #48]
    str x4, [x17, #40]             // rt->resume_pc
    ldr x17, [x17, #0]             // cpu*
    str x4, [x17, #272]            // cpu->pc
    mov x0, #-1                    // INT_NONE
    ldr x17, [sp, #80]
    msr nzcv, x17
    ldp x2, x3, [sp, #64]
    ldp x4, x5, [sp, #32]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #112
    ret
90:
    ldr x17, [sp, #80]
    msr nzcv, x17
    ldp x4, x5, [sp, #32]
    ldr x0, [sp, #48]              // runtime*
    ldr x16, [x0, #24]             // spill_state_fn
    cbz x16, 96f
    blr x16
96:
    ldr x0, [sp, #48]
    ldr w1, [sp, #60]
    ldr x2, [sp, #56]
    bl _arm64_jit_c_ldr_imm_unsigned
    cmp w0, #-1
    b.ne 97f
    ldr x0, [sp, #48]
    ldr x16, [x0, #32]             // reload_state_fn
    cbz x16, 97f
    str x0, [sp, #88]
    blr x16
    ldr x0, [sp, #88]
    ldr x3, [sp, #72]              // store_in_x1
    cbz x3, 97f
    ldr x2, [sp, #56]              // packed1
    ubfx x2, x2, #0, #5            // rt
    cmp x2, #31
    b.eq 97f
    ldr x17, [x0, #0]              // cpu*
    add x17, x17, #16
    ldr x1, [x17, x2, lsl #3]
97:
    cmp w0, #-1
    b.ne 98f
    ldr x17, [sp, #80]
    msr nzcv, x17
98:
    ldp x4, x5, [sp, #32]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #112
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
    stp x29, x30, [sp, #-96]!
    stp x16, x17, [sp, #16]
    stp x4, x5, [sp, #32]
    stp x6, x7, [sp, #48]
    stp x0, x1, [sp, #64]
    mrs x17, nzcv
    str x17, [sp, #80]

    ubfx x16, x1, #42, #2          // size_shift
    and x4, x2, #0xfff
    mov x5, #8
    lsl x5, x5, x16
    sub x5, x5, #1
    add x4, x4, x5
    cmp x4, #0xfff
    bhi 90f

    ldr x4, [x0, #8]               // tlb*
    eor x5, x2, x2, lsr #13
    lsr x5, x5, #12
    and x5, x5, #0x1fff
    add x5, x4, #32
    add x5, x5, x5, lsl #5         // overwritten below
    eor x16, x2, x2, lsr #13
    lsr x16, x16, #12
    and x16, x16, #0x1fff
    add x5, x4, #32
    add x5, x5, x16, lsl #5        // tlb entry*
    and x16, x2, #0xfffffffffffff000
    ldr x6, [x5, #8]               // entry.page_if_writable
    cmp x6, x16
    b.ne 90f

    str x16, [x4, #8]              // tlb->dirty_page
    ldr x6, [x5, #16]              // data_minus_addr
    add x6, x6, x2                 // host ptr

    ldr x1, [sp, #72]              // packed
    ubfx x16, x1, #42, #2          // size_shift
    cmp x16, #3
    b.eq 91f
    cmp x16, #2
    b.eq 92f
    cmp x16, #1
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
    ldr w1, [sp, #72]              // guest_pc low 32
    add x1, x1, #4
    ldr x17, [sp, #64]             // runtime*
    str x1, [x17, #40]             // rt->resume_pc
    ldr x17, [x17, #0]             // cpu*
    str x1, [x17, #272]            // cpu->pc
    mov x0, #-1                    // INT_NONE
    ldr x17, [sp, #80]
    msr nzcv, x17
    ldp x6, x7, [sp, #48]
    ldp x4, x5, [sp, #32]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #96
    ret
90:
    ldr x17, [sp, #80]
    msr nzcv, x17
    ldp x6, x7, [sp, #48]
    ldp x4, x5, [sp, #32]
    ldr x0, [sp, #64]              // runtime*
    ldr x16, [x0, #24]             // spill_state_fn
    cbz x16, 95f
    blr x16
95:
    ldr x0, [sp, #64]
    ldr x1, [sp, #72]
    bl _arm64_jit_c_str_imm_unsigned
    cmp w0, #-1
    b.ne 96f
    ldr x0, [sp, #64]
    ldr x16, [x0, #32]             // reload_state_fn
    cbz x16, 96f
    str x0, [sp, #88]
    blr x16
    ldr x0, [sp, #88]
96:
    cmp w0, #-1
    b.ne 97f
    ldr x17, [sp, #80]
    msr nzcv, x17
97:
    ldp x6, x7, [sp, #48]
    ldp x4, x5, [sp, #32]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #96
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
    lsr x1, x1, #32               // guest_pc
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
    stp x29, x30, [sp, #-112]!
    stp x16, x17, [sp, #16]
    stp x4, x5, [sp, #32]
    stp x6, x7, [sp, #48]
    stp x8, x9, [sp, #64]
    stp x0, x1, [sp, #80]
    mrs x16, nzcv
    str x16, [sp, #96]

    ldr w16, [sp, #88]             // insn low 32
    ubfx x17, x16, #30, #2         // size
    ubfx x4, x16, #22, #2          // opc
    cmp x17, #0
    b.ne 0f
    tst x4, #0x2
    b.eq 0f                        // Bt
    mov x3, #16                    // Qt width
    b 1f
0:
    cmp x4, #0x2
    b.hs 90f
    mov x3, #1
    lsl x3, x3, x17               // 1/2/4/8
1:
    and x4, x2, #0xfff
    sub x5, x3, #1
    add x4, x4, x5
    cmp x4, #0xfff
    bhi 90f

    ldr x4, [x0, #8]               // tlb*
    eor x5, x2, x2, lsr #13
    lsr x5, x5, #12
    and x5, x5, #0x1fff
    add x5, x4, #32
    add x5, x5, x5, lsl #5         // overwritten below
    eor x16, x2, x2, lsr #13
    lsr x16, x16, #12
    and x16, x16, #0x1fff
    add x5, x4, #32
    add x5, x5, x16, lsl #5
    and x16, x2, #0xfffffffffffff000

    ldr w17, [sp, #88]             // insn low 32
    ubfx x6, x17, #22, #2         // opc
    tst x6, #0x1                  // L
    b.ne 2f
    ldr x6, [x5, #8]              // writable
    cmp x6, x16
    b.ne 90f
    str x16, [x4, #8]             // dirty_page
    b 3f
2:
    ldr x6, [x5, #0]              // readable
    cmp x6, x16
    b.ne 90f
3:
    ldr x5, [x5, #16]
    add x5, x5, x2

    ubfx x6, x17, #0, #5          // rt
    ldr x16, [sp, #80]
    ldr x16, [x16, #0]            // cpu*
    add x6, x16, x6, lsl #4
    add x6, x6, #288              // &cpu->fp[rt]

    ubfx x16, x17, #30, #2        // size
    ubfx x4, x17, #22, #2         // opc
    cmp x16, #0
    b.ne 4f
    tst x4, #0x2
    b.eq 4f
    tst x4, #0x1
    b.ne 10f
    ldr x7, [x6]
    ldr x8, [x6, #8]
    str x7, [x5]
    str x8, [x5, #8]
    b 20f
10:
    ldr x7, [x5]
    ldr x8, [x5, #8]
    str x7, [x6]
    str x8, [x6, #8]
    b 20f

4:
    tst x4, #0x1
    b.ne 11f
    cmp x16, #3
    b.eq 12f
    cmp x16, #2
    b.eq 13f
    cmp x16, #1
    b.eq 14f
    ldrb w7, [x6]
    strb w7, [x5]
    b 20f
12:
    ldr x7, [x6]
    str x7, [x5]
    b 20f
13:
    ldr w7, [x6]
    str w7, [x5]
    b 20f
14:
    ldrh w7, [x6]
    strh w7, [x5]
    b 20f

11:
    cmp x16, #3
    b.eq 15f
    cmp x16, #2
    b.eq 16f
    cmp x16, #1
    b.eq 17f
    ldrb w7, [x5]
    strb w7, [x6]
    mov x7, #0
    str x7, [x6, #8]
    b 20f
15:
    ldr x7, [x5]
    str x7, [x6]
    mov x7, #0
    str x7, [x6, #8]
    b 20f
16:
    ldr w7, [x5]
    str w7, [x6]
    mov x7, #0
    str x7, [x6, #8]
    b 20f
17:
    ldrh w7, [x5]
    strh w7, [x6]
    mov x7, #0
    str x7, [x6, #8]

20:
    ldr w1, [sp, #92]
    add x1, x1, #4
    ldr x16, [sp, #80]
    str x1, [x16, #40]             // rt->resume_pc
    ldr x16, [x16, #0]
    str x1, [x16, #272]
    mov x0, #-1
    ldr x16, [sp, #96]
    msr nzcv, x16
    ldp x8, x9, [sp, #64]
    ldp x6, x7, [sp, #48]
    ldp x4, x5, [sp, #32]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #112
    ret
90:
    ldr x0, [sp, #80]
    ldr x16, [sp, #96]
    msr nzcv, x16
    ldp x8, x9, [sp, #64]
    ldp x6, x7, [sp, #48]
    ldp x4, x5, [sp, #32]
    ldr x16, [x0, #24]
    cbz x16, 91f
    blr x16
91:
    ldr x0, [sp, #80]
    ldr w1, [sp, #92]
    ldr w2, [sp, #88]
    bl _arm64_jit_c_simd_ldst_imm_unsigned
    cmp w0, #-1
    b.ne 92f
    ldr x0, [sp, #80]
    ldr x16, [x0, #32]
    cbz x16, 92f
    str x0, [sp, #88]
    blr x16
    ldr x0, [sp, #88]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #112
    ret
92:
    ldp x8, x9, [sp, #64]
    ldp x6, x7, [sp, #48]
    ldp x4, x5, [sp, #32]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #112
    ret

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

.globl _arm64_jit_helper_ldst_imm9_live
_arm64_jit_helper_ldst_imm9_live:
    // x0 = rt*, x1 = packed(guest_pc<<32 | insn), x2 = guest_addr, x3 = value
    // Clean live-cache path only for unscaled scalar store (mode==0, opc==0).
    stp x29, x30, [sp, #-96]!
    stp x16, x17, [sp, #16]
    stp x4, x5, [sp, #32]
    stp x6, x7, [sp, #48]
    stp x0, x1, [sp, #64]
    mrs x17, nzcv
    str x17, [sp, #80]

    ldr w16, [sp, #72]             // insn low 32
    ubfx x17, x16, #26, #1         // V
    cbnz x17, 90f
    ubfx x17, x16, #10, #2         // mode
    cbnz x17, 90f
    ubfx x17, x16, #22, #2         // opc
    cbnz x17, 90f
    ubfx x17, x16, #30, #2         // size

    and x4, x2, #0xfff
    mov x5, #8
    lsl x5, x5, x17
    sub x5, x5, #1
    add x4, x4, x5
    cmp x4, #0xfff
    bhi 90f

    ldr x4, [x0, #8]               // tlb*
    eor x5, x2, x2, lsr #13
    lsr x5, x5, #12
    and x5, x5, #0x1fff
    add x5, x4, #32
    add x5, x5, x5, lsl #5         // overwritten below
    eor x16, x2, x2, lsr #13
    lsr x16, x16, #12
    and x16, x16, #0x1fff
    add x5, x4, #32
    add x5, x5, x16, lsl #5
    and x16, x2, #0xfffffffffffff000
    ldr x6, [x5, #8]               // entry.page_if_writable
    cmp x6, x16
    b.ne 90f
    str x16, [x4, #8]              // tlb->dirty_page
    ldr x6, [x5, #16]              // data_minus_addr
    add x6, x6, x2

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
    ldr w1, [sp, #76]              // guest_pc low 32
    add x1, x1, #4
    ldr x17, [sp, #64]
    str x1, [x17, #40]             // rt->resume_pc
    ldr x17, [x17, #0]
    str x1, [x17, #272]
    mov x0, #-1
    ldr x17, [sp, #80]
    msr nzcv, x17
    ldp x6, x7, [sp, #48]
    ldp x4, x5, [sp, #32]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #96
    ret

.globl _arm64_jit_helper_ldst_imm9_load_live
_arm64_jit_helper_ldst_imm9_load_live:
    // x0 = rt*, x1 = packed(guest_pc<<32 | insn), x2 = guest_addr, x3 = store_in_x1
    // Clean live-cache path only for unscaled scalar load (mode==0, opc==1).
    stp x29, x30, [sp, #-112]!
    stp x16, x17, [sp, #16]
    stp x4, x5, [sp, #32]
    stp x6, x7, [sp, #48]
    stp x0, x1, [sp, #64]
    stp x2, x3, [sp, #80]
    mrs x17, nzcv
    str x17, [sp, #96]

    ldr w16, [sp, #72]             // insn low 32
    ubfx x17, x16, #26, #1         // V
    cbnz x17, 90f
    ubfx x17, x16, #10, #2         // mode
    cbnz x17, 90f
    ubfx x17, x16, #22, #2         // opc
    cmp x17, #1
    b.ne 90f
    ubfx x17, x16, #30, #2         // size

    and x4, x2, #0xfff
    mov x5, #8
    lsl x5, x5, x17
    sub x5, x5, #1
    add x4, x4, x5
    cmp x4, #0xfff
    bhi 90f

    ldr x4, [x0, #8]               // tlb*
    eor x5, x2, x2, lsr #13
    lsr x5, x5, #12
    and x5, x5, #0x1fff
    add x5, x4, #32
    add x5, x5, x5, lsl #5         // overwritten below
    eor x16, x2, x2, lsr #13
    lsr x16, x16, #12
    and x16, x16, #0x1fff
    add x5, x4, #32
    add x5, x5, x16, lsl #5
    and x16, x2, #0xfffffffffffff000
    ldr x6, [x5, #0]               // entry.page
    cmp x6, x16
    b.ne 90f
    ldr x6, [x5, #16]
    add x6, x6, x2

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
    ldr x3, [sp, #88]              // store_in_x1
    cbnz x3, 95f
    ldr w16, [sp, #72]             // insn low 32
    ubfx x5, x16, #0, #5           // rt
    cmp x5, #31
    b.eq 96f
    ldr x17, [sp, #64]
    ldr x17, [x17, #0]
    add x17, x17, #16
    str x4, [x17, x5, lsl #3]
    b 96f
95:
    mov x1, x4
96:
    ldr w4, [sp, #76]
    add x4, x4, #4
    ldr x17, [sp, #64]
    str x4, [x17, #40]             // rt->resume_pc
    ldr x17, [x17, #0]
    str x4, [x17, #272]
    mov x0, #-1
    ldr x17, [sp, #96]
    msr nzcv, x17
    ldp x6, x7, [sp, #48]
    ldp x4, x5, [sp, #32]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #112
    ret
90:
    ldr x17, [sp, #96]
    msr nzcv, x17
    ldp x6, x7, [sp, #48]
    ldp x4, x5, [sp, #32]
    ldr x0, [sp, #64]
    ldr x16, [x0, #24]
    cbz x16, 97f
    blr x16
97:
    ldr x0, [sp, #64]
    ldr w1, [sp, #76]
    ldr w2, [sp, #72]
    bl _arm64_jit_c_ldst_imm9
    cmp w0, #-1
    b.ne 98f
    ldr x0, [sp, #64]
    ldr x16, [x0, #32]
    cbz x16, 98f
    str x0, [sp, #104]
    blr x16
    ldr x0, [sp, #104]
98:
    ldp x6, x7, [sp, #48]
    ldp x4, x5, [sp, #32]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #112
    ret
90:
    ldr x17, [sp, #80]
    msr nzcv, x17
    ldp x6, x7, [sp, #48]
    ldp x4, x5, [sp, #32]
    ldr x0, [sp, #64]
    ldr x16, [x0, #24]
    cbz x16, 95f
    blr x16
95:
    ldr x0, [sp, #64]
    ldr x1, [sp, #68]
    ldr w2, [sp, #72]
    bl _arm64_jit_c_ldst_imm9
    cmp w0, #-1
    b.ne 96f
    ldr x0, [sp, #64]
    ldr x16, [x0, #32]
    cbz x16, 96f
    str x0, [sp, #88]
    blr x16
    ldr x0, [sp, #88]
96:
    ldp x16, x17, [sp, #16]
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
    lsr x1, x1, #32
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
    stp x29, x30, [sp, #-112]!
    stp x16, x17, [sp, #16]
    stp x0, x1, [sp, #32]
    stp x2, x3, [sp, #48]
    stp x4, x5, [sp, #64]
    stp x6, x7, [sp, #80]
    mrs x16, nzcv
    str x16, [sp, #96]

    ldr w16, [sp, #40]             // insn low 32
    ubfx x17, x16, #26, #1         // V
    cbnz x17, 90f
    ubfx x17, x16, #13, #3         // option
    cmp x17, #3
    b.ne 90f
    ubfx x17, x16, #22, #2         // opc
    cbnz x17, 90f
    ubfx x17, x16, #30, #2         // size

    and x4, x2, #0xfff
    mov x5, #8
    lsl x5, x5, x17
    sub x5, x5, #1
    add x4, x4, x5
    cmp x4, #0xfff
    bhi 90f

    ldr x4, [x0, #8]               // tlb*
    eor x5, x2, x2, lsr #13
    lsr x5, x5, #12
    and x5, x5, #0x1fff
    add x5, x4, #32
    add x5, x5, x5, lsl #5         // overwritten below
    eor x16, x2, x2, lsr #13
    lsr x16, x16, #12
    and x16, x16, #0x1fff
    add x5, x4, #32
    add x5, x5, x16, lsl #5
    and x16, x2, #0xfffffffffffff000
    ldr x6, [x5, #8]               // entry.page_if_writable
    cmp x6, x16
    b.ne 90f
    str x16, [x4, #8]              // tlb->dirty_page
    ldr x6, [x5, #16]
    add x6, x6, x2

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
    ldr x16, [sp, #96]
    msr nzcv, x16
    mov x0, #-1
    ldp x6, x7, [sp, #80]
    ldp x4, x5, [sp, #64]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #112
    ret
90:
    ldr x0, [sp, #32]
    ldr x16, [sp, #96]
    msr nzcv, x16
    ldp x6, x7, [sp, #80]
    ldp x4, x5, [sp, #64]
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
    str x0, [sp, #104]
    blr x16
    ldr x0, [sp, #104]
96:
    ldp x6, x7, [sp, #80]
    ldp x4, x5, [sp, #64]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #112
    ret

.globl _arm64_jit_helper_ldst_regoff_load_live
_arm64_jit_helper_ldst_regoff_load_live:
    // x0 = rt*, x1 = packed(guest_pc<<32 | insn), x2 = guest_addr, x3 = store_in_x1
    // Clean live-cache path only for scalar load, option==3.
    stp x29, x30, [sp, #-112]!
    stp x16, x17, [sp, #16]
    stp x0, x1, [sp, #32]
    stp x2, x3, [sp, #48]
    stp x4, x5, [sp, #64]
    stp x6, x7, [sp, #80]
    mrs x16, nzcv
    str x16, [sp, #96]

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

    and x4, x2, #0xfff
    mov x5, #8
    lsl x5, x5, x17
    sub x5, x5, #1
    add x4, x4, x5
    cmp x4, #0xfff
    bhi 90f

    ldr x4, [x0, #8]               // tlb*
    eor x5, x2, x2, lsr #13
    lsr x5, x5, #12
    and x5, x5, #0x1fff
    add x5, x4, #32
    add x5, x5, x5, lsl #5         // overwritten below
    eor x16, x2, x2, lsr #13
    lsr x16, x16, #12
    and x16, x16, #0x1fff
    add x5, x4, #32
    add x5, x5, x16, lsl #5
    and x16, x2, #0xfffffffffffff000
    ldr x6, [x5, #0]               // entry.page
    cmp x6, x16
    b.ne 90f
    ldr x6, [x5, #16]
    add x6, x6, x2

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
    ldr x3, [sp, #56]              // store_in_x1
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
    ldr x16, [sp, #96]
    msr nzcv, x16
    mov x0, #-1
    ldp x6, x7, [sp, #80]
    ldp x4, x5, [sp, #64]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #112
    ret
90:
    ldr x0, [sp, #32]
    ldr x16, [sp, #96]
    msr nzcv, x16
    ldp x6, x7, [sp, #80]
    ldp x4, x5, [sp, #64]
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
    cbz x16, 98f
    str x0, [sp, #104]
    blr x16
    ldr x0, [sp, #104]
98:
    ldp x6, x7, [sp, #80]
    ldp x4, x5, [sp, #64]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #112
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
    stp x29, x30, [sp, #-128]!
    stp x16, x17, [sp, #16]
    stp x0, x1, [sp, #32]
    stp x2, x3, [sp, #48]
    stp x4, x5, [sp, #64]
    stp x6, x7, [sp, #80]
    stp x8, x9, [sp, #96]
    mrs x16, nzcv
    str x16, [sp, #112]

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
    mov x4, #4
    cmp x17, #2
    cinc x4, x4, eq                // x4 = 5 when 64-bit? no, replaced below
    cmp x17, #2
    b.ne 2f
    mov x4, #8
    b 3f
2:
    mov x4, #4
3:
    and x5, x2, #0xfff
    lsl x6, x4, #1                 // pair width
    sub x6, x6, #1
    add x5, x5, x6
    cmp x5, #0xfff
    bhi 90f

    ldr x5, [x0, #8]               // tlb*
    eor x6, x2, x2, lsr #13
    lsr x6, x6, #12
    and x6, x6, #0x1fff
    add x6, x5, #32
    add x6, x6, x6, lsl #5         // overwritten below
    eor x16, x2, x2, lsr #13
    lsr x16, x16, #12
    and x16, x16, #0x1fff
    add x6, x5, #32
    add x6, x6, x16, lsl #5
    and x16, x2, #0xfffffffffffff000
    ldr x7, [x6, #0]               // entry.page
    cmp x7, x16
    b.ne 90f
    ldr x6, [x6, #16]
    add x6, x6, x2

    ldr w16, [sp, #40]
    ubfx x17, x16, #30, #2         // opc
    cmp x17, #2
    b.eq 10f
    ldr w7, [x6]
    ldr w16, [x6, #4]
    b 11f
10:
    ldr x7, [x6]
    ldr x16, [x6, #8]
11:
    ldr x3, [sp, #56]              // return_in_x1_x2
    cbnz x3, 12f
    ubfx x5, x16, #0, #5           // rt
    ubfx x6, x16, #10, #5          // rt2
    ldr x17, [sp, #32]
    ldr x17, [x17, #0]
    add x17, x17, #16
    cmp x5, #31
    b.eq 13f
    str x7, [x17, x5, lsl #3]
13:
    cmp x6, #31
    b.eq 14f
    str x16, [x17, x6, lsl #3]
    b 14f
12:
    mov x1, x7
    mov x2, x16
14:
    ldr w4, [sp, #44]
    add x4, x4, #4
    ldr x17, [sp, #32]
    str x4, [x17, #40]             // rt->resume_pc
    ldr x17, [x17, #0]
    str x4, [x17, #272]
    ldr x16, [sp, #112]
    msr nzcv, x16
    mov x0, #-1
    ldp x8, x9, [sp, #96]
    ldp x6, x7, [sp, #80]
    ldp x4, x5, [sp, #64]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #128
    ret
90:
    ldr x0, [sp, #32]
    ldr x16, [sp, #112]
    msr nzcv, x16
    ldp x8, x9, [sp, #96]
    ldp x6, x7, [sp, #80]
    ldp x4, x5, [sp, #64]
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
    str x0, [sp, #48]
    blr x16
    ldr x0, [sp, #48]
98:
    ldp x8, x9, [sp, #96]
    ldp x6, x7, [sp, #80]
    ldp x4, x5, [sp, #64]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #128
    ret

.globl _arm64_jit_helper_ldst_pair_store_live
_arm64_jit_helper_ldst_pair_store_live:
    // x0 = rt*, x1 = packed(guest_pc<<32 | insn), x2 = guest_addr, x3 = val0, x16 = val1
    // Clean live-cache path only for scalar offset pair stores (V==0, L==0, mode==2, opc==0/2).
    stp x29, x30, [sp, #-128]!
    stp x16, x17, [sp, #16]
    stp x0, x1, [sp, #32]
    stp x2, x3, [sp, #48]
    stp x4, x5, [sp, #64]
    stp x6, x7, [sp, #80]
    stp x8, x9, [sp, #96]
    mrs x16, nzcv
    str x16, [sp, #112]

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
    mov x5, #4
    cmp x17, #2
    b.ne 2f
    mov x5, #8
2:
    and x6, x2, #0xfff
    lsl x7, x5, #1                 // pair width
    sub x7, x7, #1
    add x6, x6, x7
    cmp x6, #0xfff
    bhi 90f

    ldr x6, [x0, #8]               // tlb*
    eor x7, x2, x2, lsr #13
    lsr x7, x7, #12
    and x7, x7, #0x1fff
    add x7, x6, #32
    add x7, x7, x7, lsl #5         // overwritten below
    eor x16, x2, x2, lsr #13
    lsr x16, x16, #12
    and x16, x16, #0x1fff
    add x7, x6, #32
    add x7, x7, x16, lsl #5
    and x16, x2, #0xfffffffffffff000
    ldr x8, [x7, #8]               // entry.page_if_writable
    cmp x8, x16
    b.ne 90f
    str x16, [x6, #8]              // tlb->dirty_page
    ldr x7, [x7, #16]
    add x7, x7, x2

    ldr w16, [sp, #40]
    ubfx x17, x16, #30, #2         // opc
    ldr x8, [sp, #16]              // saved incoming val1 from x16
    cmp x17, #2
    b.eq 10f
    str w3, [x7]
    str w8, [x7, #4]
    b 11f
10:
    str x3, [x7]
    str x8, [x7, #8]
11:
    ldr w1, [sp, #44]
    add x1, x1, #4
    ldr x17, [sp, #32]
    str x1, [x17, #40]             // rt->resume_pc
    ldr x17, [x17, #0]
    str x1, [x17, #272]
    ldr x16, [sp, #112]
    msr nzcv, x16
    mov x0, #-1
    ldp x8, x9, [sp, #96]
    ldp x6, x7, [sp, #80]
    ldp x4, x5, [sp, #64]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #128
    ret
90:
    ldr x0, [sp, #32]
    ldr x16, [sp, #112]
    msr nzcv, x16
    ldp x8, x9, [sp, #96]
    ldp x6, x7, [sp, #80]
    ldp x4, x5, [sp, #64]
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
    str x0, [sp, #48]
    blr x16
    ldr x0, [sp, #48]
97:
    ldp x8, x9, [sp, #96]
    ldp x6, x7, [sp, #80]
    ldp x4, x5, [sp, #64]
    ldp x16, x17, [sp, #16]
    ldp x29, x30, [sp], #128
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

    ldr x4, [x0, #8]               // tlb*
    eor x5, x2, x2, lsr #13
    lsr x5, x5, #12
    and x5, x5, #0x1fff
    add x5, x4, #32
    add x5, x5, x5, lsl #5         // overwritten below
    eor x16, x2, x2, lsr #13
    lsr x16, x16, #12
    and x16, x16, #0x1fff
    add x5, x4, #32
    add x5, x5, x16, lsl #5
    and x16, x2, #0xfffffffffffff000

    ldr w17, [sp, #104]
    ubfx x6, x17, #22, #1          // L
    cbnz x6, 1f
    ldr x6, [x5, #8]               // page_if_writable
    cmp x6, x16
    b.ne 90f
    str x16, [x4, #8]              // dirty_page
    b 2f
1:
    ldr x6, [x5, #0]               // page
    cmp x6, x16
    b.ne 90f
2:
    ldr x5, [x5, #16]
    add x5, x5, x2

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
    cbz x16, 92f
    str x0, [sp, #104]
    blr x16
    ldr x0, [sp, #104]
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
    ret

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
JITABI_WRAP2 _arm64_jit_helper_cbz_cbnz_jitabi, _arm64_jit_helper_cbz_cbnz
JITABI_WRAP2 _arm64_jit_helper_b_cond_jitabi, _arm64_jit_helper_b_cond
JITABI_WRAP2 _arm64_jit_helper_tbz_tbnz_jitabi, _arm64_jit_helper_tbz_tbnz
JITABI_WRAP2 _arm64_jit_helper_branch_reg_jitabi, _arm64_jit_helper_branch_reg
JITABI_WRAP2 _arm64_jit_helper_ldst_imm9_jitabi, _arm64_jit_helper_ldst_imm9
JITABI_WRAP2 _arm64_jit_helper_ldst_regoff_jitabi, _arm64_jit_helper_ldst_regoff
JITABI_WRAP2 _arm64_jit_helper_ldst_pair_jitabi, _arm64_jit_helper_ldst_pair
JITABI_WRAP2 _arm64_jit_helper_simd_ldst_imm_unsigned_jitabi, _arm64_jit_helper_simd_ldst_imm_unsigned

#endif
