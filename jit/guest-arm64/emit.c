#if defined(GUEST_ARM64) && defined(__aarch64__)

#include <errno.h>
#include <sys/mman.h>

#include "jit/guest-arm64/jit.h"

uint32_t arm64_jit_enc_movz(unsigned rd, uint16_t imm16, unsigned shift) {
    return 0xd2800000u | ((uint32_t) shift << 21) | ((uint32_t) imm16 << 5) | rd;
}

uint32_t arm64_jit_enc_movk(unsigned rd, uint16_t imm16, unsigned shift) {
    return 0xf2800000u | ((uint32_t) shift << 21) | ((uint32_t) imm16 << 5) | rd;
}

uint32_t arm64_jit_enc_mov_reg(unsigned rd, unsigned rn) {
    return 0xaa0003e0u | (rn << 16) | rd;
}

uint32_t arm64_jit_enc_blr(unsigned rn) {
    return 0xd63f0000u | (rn << 5);
}

uint32_t arm64_jit_enc_ret(unsigned rn) {
    return 0xd65f0000u | (rn << 5);
}

uint32_t arm64_jit_enc_brk(unsigned imm16) {
    return 0xd4200000u | ((imm16 & 0xffff) << 5);
}

uint32_t arm64_jit_enc_ldr64_uimm(unsigned rt, unsigned rn, unsigned imm12) {
    return 0xf9400000u | ((imm12 & 0xfff) << 10) | ((rn & 0x1f) << 5) | (rt & 0x1f);
}

uint32_t arm64_jit_enc_str64_uimm(unsigned rt, unsigned rn, unsigned imm12) {
    return 0xf9000000u | ((imm12 & 0xfff) << 10) | ((rn & 0x1f) << 5) | (rt & 0x1f);
}

uint32_t arm64_jit_enc_ldr32_uimm(unsigned rt, unsigned rn, unsigned imm12) {
    return 0xb9400000u | ((imm12 & 0xfff) << 10) | ((rn & 0x1f) << 5) | (rt & 0x1f);
}

uint32_t arm64_jit_enc_str32_uimm(unsigned rt, unsigned rn, unsigned imm12) {
    return 0xb9000000u | ((imm12 & 0xfff) << 10) | ((rn & 0x1f) << 5) | (rt & 0x1f);
}

uint32_t arm64_jit_enc_ldr128_uimm(unsigned rt, unsigned rn, unsigned imm12) {
    return 0x3dc00000u | ((imm12 & 0xfff) << 10) | ((rn & 0x1f) << 5) | (rt & 0x1f);
}

uint32_t arm64_jit_enc_str128_uimm(unsigned rt, unsigned rn, unsigned imm12) {
    return 0x3d800000u | ((imm12 & 0xfff) << 10) | ((rn & 0x1f) << 5) | (rt & 0x1f);
}

uint32_t arm64_jit_enc_b_imm(int32_t imm26) {
    return 0x14000000u | ((uint32_t) imm26 & 0x03ffffffu);
}

uint32_t arm64_jit_enc_b_cond(enum arm64_cond cond, int32_t imm19) {
    return 0x54000000u | (((uint32_t) imm19 & 0x7ffffu) << 5) | (cond & 0xf);
}

uint32_t arm64_jit_enc_cbz_cbnz(bool sf, bool nonzero, unsigned rt, int32_t imm19) {
    return ((uint32_t) sf << 31) | 0x34000000u | ((uint32_t) nonzero << 24) |
           (((uint32_t) imm19 & 0x7ffffu) << 5) | (rt & 0x1f);
}

uint32_t arm64_jit_enc_tbz_tbnz(bool b5, bool nonzero, unsigned bit40, unsigned rt, int32_t imm14) {
    return ((uint32_t) b5 << 31) | 0x36000000u | ((uint32_t) nonzero << 24) |
           ((bit40 & 0x1f) << 19) | (((uint32_t) imm14 & 0x3fffu) << 5) | (rt & 0x1f);
}

void arm64_jit_emit32(struct arm64_jit_emitter *e, uint32_t insn) {
    if (e->size + 4 <= e->cap) {
        *(uint32_t *) (e->buf + e->size) = insn;
        e->size += 4;
    }
}

void arm64_jit_record_pc_map(struct arm64_jit_emitter *e, addr_t guest_pc) {
    if (e->block->pc_map_count < ARM64_JIT_MAX_PC_MAP) {
        struct arm64_jit_pc_map *m = &e->block->pc_map[e->block->pc_map_count++];
        m->host_offset = (uint32_t) e->size;
        m->guest_pc = guest_pc;
    }
}

void arm64_jit_record_verify_site(struct arm64_jit_emitter *e, addr_t guest_pc, uint32_t insn) {
    if (e->block->verify_site_count < ARM64_JIT_MAX_VERIFY_SITES) {
        struct arm64_jit_verify_site *site =
                &e->block->verify_sites[e->block->verify_site_count++];
        site->host_offset = (uint32_t) e->size;
        site->guest_pc = guest_pc;
        site->insn = insn;
    }
}

void arm64_jit_emit_verify_entry_brk(struct arm64_jit_emitter *e, addr_t guest_pc, uint32_t insn) {
    if (!arm64_jit_verify_mode())
        return;
    arm64_jit_record_verify_site(e, guest_pc, insn);
    arm64_jit_record_pc_map(e, guest_pc);
    arm64_jit_emit32(e, arm64_jit_enc_brk(0x4a1));
}

bool arm64_jit_block_has_pc(const struct arm64_jit_block *block, addr_t guest_pc) {
    for (uint32_t i = 0; i < block->insn_count; i++) {
        if (block->insn_pcs[i] == guest_pc)
            return true;
    }
    return false;
}

void arm64_jit_emit_local_fixup(struct arm64_jit_emitter *e, addr_t branch_pc, addr_t target_pc, uint32_t kind) {
    if (e->block->fixup_count >= ARM64_JIT_MAX_FIXUPS)
        return;
    struct arm64_jit_local_fixup *f = &e->block->fixups[e->block->fixup_count++];
    f->branch_offset = (uint32_t) e->size;
    f->branch_pc = branch_pc;
    f->target_pc = target_pc;
    f->kind = kind;
}

static bool arm64_jit_local_fixup_disabled(const struct arm64_jit_block *block, addr_t branch_pc) {
    for (uint32_t i = 0; i < block->disabled_local_fixup_count; i++) {
        if (block->disabled_local_fixup_pcs[i] == branch_pc)
            return true;
    }
    return false;
}

static uint32_t arm64_jit_find_host_offset_for_pc(const struct arm64_jit_block *block, addr_t guest_pc) {
    for (uint32_t i = 0; i < block->pc_map_count; i++) {
        if (block->pc_map[i].guest_pc == guest_pc)
            return block->pc_map[i].host_offset;
    }
    return UINT32_MAX;
}

bool arm64_jit_patch_local_fixups(struct arm64_jit_block *block, uint8_t *buf) {
    for (uint32_t i = 0; i < block->fixup_count; i++) {
        struct arm64_jit_local_fixup *f = &block->fixups[i];
        uint32_t target_off = arm64_jit_find_host_offset_for_pc(block, f->target_pc);
        if (target_off == UINT32_MAX) {
            if (arm64_jit_trace_mode()) {
                fprintf(stderr, "[arm64-jit] fixup unresolved idx=%u target_pc=0x%llx kind=%u branch_off=%u\n",
                        i, (unsigned long long) f->target_pc, f->kind, f->branch_offset);
            }
            if (block->disabled_local_fixup_count < ARM64_JIT_MAX_FIXUPS)
                block->disabled_local_fixup_pcs[block->disabled_local_fixup_count++] = f->branch_pc;
            return false;
        }
        int32_t byte_delta = (int32_t) target_off - (int32_t) f->branch_offset;
        uint32_t *slot = (uint32_t *) (buf + f->branch_offset);
        switch (f->kind) {
            case ARM64_JIT_FIXUP_B:
                *slot = arm64_jit_enc_b_imm(byte_delta >> 2);
                break;
            case ARM64_JIT_FIXUP_B_COND: {
                enum arm64_cond cond = (enum arm64_cond) (*slot & 0xf);
                *slot = arm64_jit_enc_b_cond(cond, byte_delta >> 2);
                break;
            }
            case ARM64_JIT_FIXUP_CBZ: {
                bool sf = ((*slot >> 31) & 1) != 0;
                bool nonzero = ((*slot >> 24) & 1) != 0;
                unsigned rt = *slot & 0x1f;
                *slot = arm64_jit_enc_cbz_cbnz(sf, nonzero, rt, byte_delta >> 2);
                break;
            }
            case ARM64_JIT_FIXUP_TBZ: {
                bool b5 = ((*slot >> 31) & 1) != 0;
                bool nonzero = ((*slot >> 24) & 1) != 0;
                unsigned bit40 = (*slot >> 19) & 0x1f;
                unsigned rt = *slot & 0x1f;
                *slot = arm64_jit_enc_tbz_tbnz(b5, nonzero, bit40, rt, byte_delta >> 2);
                break;
            }
            default:
                return false;
        }
    }
    return true;
}

void arm64_jit_emit_load_imm64(struct arm64_jit_emitter *e, unsigned rd, uint64_t value) {
    arm64_jit_emit32(e, arm64_jit_enc_movz(rd, value & 0xffff, 0));
    arm64_jit_emit32(e, arm64_jit_enc_movk(rd, (value >> 16) & 0xffff, 1));
    arm64_jit_emit32(e, arm64_jit_enc_movk(rd, (value >> 32) & 0xffff, 2));
    arm64_jit_emit32(e, arm64_jit_enc_movk(rd, (value >> 48) & 0xffff, 3));
}

void arm64_jit_emit_prologue(struct arm64_jit_emitter *e) {
    arm64_jit_emit32(e, 0xa9ba7bfd); // stp x29, x30, [sp, #-96]!
    arm64_jit_emit32(e, 0xa90153f3); // stp x19, x20, [sp, #16]
    arm64_jit_emit32(e, 0xa9025bf5); // stp x21, x22, [sp, #32]
    arm64_jit_emit32(e, 0xa90363f7); // stp x23, x24, [sp, #48]
    arm64_jit_emit32(e, 0xa9046bf9); // stp x25, x26, [sp, #64]
    arm64_jit_emit32(e, 0xa90573fb); // stp x27, x28, [sp, #80]
    arm64_jit_emit32(e, 0xd10203ff); // sub sp, sp, #128
    arm64_jit_emit32(e, arm64_jit_enc_str128_uimm(8, 31, 0));
    arm64_jit_emit32(e, arm64_jit_enc_str128_uimm(9, 31, 1));
    arm64_jit_emit32(e, arm64_jit_enc_str128_uimm(10, 31, 2));
    arm64_jit_emit32(e, arm64_jit_enc_str128_uimm(11, 31, 3));
    arm64_jit_emit32(e, arm64_jit_enc_str128_uimm(12, 31, 4));
    arm64_jit_emit32(e, arm64_jit_enc_str128_uimm(13, 31, 5));
    arm64_jit_emit32(e, arm64_jit_enc_str128_uimm(14, 31, 6));
    arm64_jit_emit32(e, arm64_jit_enc_str128_uimm(15, 31, 7));
    arm64_jit_emit32(e, 0xaa0003f5); // mov x21, x0 ; runtime*
    arm64_jit_emit32(e, arm64_jit_enc_ldr64_uimm(ARM64_JIT_HOST_CPU, ARM64_JIT_HOST_CTX, 0));
    arm64_jit_emit32(e, arm64_jit_enc_ldr64_uimm(ARM64_JIT_HOST_TLB, ARM64_JIT_HOST_CTX, 1));
}

void arm64_jit_emit_load_cached_state(struct arm64_jit_emitter *e) {
    for (uint32_t guest = 0; guest < 31; guest++) {
        int host = arm64_jit_host_reg_for_guest(e->block, guest);
        if (host < 0)
            continue;
        arm64_jit_emit32(e, arm64_jit_enc_ldr64_uimm((unsigned) host, ARM64_JIT_HOST_CPU,
                (CPU_OFFSET(regs[guest]) >> 3)));
    }
    arm64_jit_emit32(e, arm64_jit_enc_ldr64_uimm(ARM64_JIT_HOST_GUEST_SP, ARM64_JIT_HOST_CPU,
            (CPU_OFFSET(sp) >> 3)));
    arm64_jit_emit32(e, arm64_jit_enc_ldr32_uimm(ARM64_JIT_HOST_HELPER0, ARM64_JIT_HOST_CPU,
            (CPU_OFFSET(nzcv) >> 2)));
    arm64_jit_emit32(e, 0xd51b4210); // msr nzcv, x16
    arm64_jit_emit32(e, arm64_jit_enc_ldr32_uimm(ARM64_JIT_HOST_HELPER0, ARM64_JIT_HOST_CPU,
            (CPU_OFFSET(fpcr) >> 2)));
    arm64_jit_emit32(e, 0xd51b4410); // msr fpcr, x16
    arm64_jit_emit32(e, arm64_jit_enc_ldr32_uimm(ARM64_JIT_HOST_HELPER0, ARM64_JIT_HOST_CPU,
            (CPU_OFFSET(fpsr) >> 2)));
    arm64_jit_emit32(e, 0xd51b4430); // msr fpsr, x16
    for (uint32_t v = 0; v < 32; v++) {
        arm64_jit_emit32(e, arm64_jit_enc_ldr128_uimm(v, ARM64_JIT_HOST_CPU,
                ((CPU_OFFSET(fp[0]) >> 4) + v)));
    }
}

void arm64_jit_emit_spill_cached_state(struct arm64_jit_emitter *e) {
    for (uint32_t guest = 0; guest < 31; guest++) {
        int host = arm64_jit_host_reg_for_guest(e->block, guest);
        if (host < 0)
            continue;
        arm64_jit_emit32(e, arm64_jit_enc_str64_uimm((unsigned) host, ARM64_JIT_HOST_CPU,
                (CPU_OFFSET(regs[guest]) >> 3)));
    }
    arm64_jit_emit32(e, arm64_jit_enc_str64_uimm(ARM64_JIT_HOST_GUEST_SP, ARM64_JIT_HOST_CPU,
            (CPU_OFFSET(sp) >> 3)));
    arm64_jit_emit32(e, 0xd53b4210); // mrs x16, nzcv
    arm64_jit_emit32(e, arm64_jit_enc_str32_uimm(ARM64_JIT_HOST_HELPER0, ARM64_JIT_HOST_CPU,
            (CPU_OFFSET(nzcv) >> 2)));
    arm64_jit_emit32(e, 0xd53b4410); // mrs x16, fpcr
    arm64_jit_emit32(e, arm64_jit_enc_str32_uimm(ARM64_JIT_HOST_HELPER0, ARM64_JIT_HOST_CPU,
            (CPU_OFFSET(fpcr) >> 2)));
    arm64_jit_emit32(e, 0xd53b4430); // mrs x16, fpsr
    arm64_jit_emit32(e, arm64_jit_enc_str32_uimm(ARM64_JIT_HOST_HELPER0, ARM64_JIT_HOST_CPU,
            (CPU_OFFSET(fpsr) >> 2)));
    for (uint32_t v = 0; v < 32; v++) {
        arm64_jit_emit32(e, arm64_jit_enc_str128_uimm(v, ARM64_JIT_HOST_CPU,
                ((CPU_OFFSET(fp[0]) >> 4) + v)));
    }
}

void arm64_jit_emit_epilogue(struct arm64_jit_emitter *e) {
    arm64_jit_emit32(e, arm64_jit_enc_ldr128_uimm(8, 31, 0));
    arm64_jit_emit32(e, arm64_jit_enc_ldr128_uimm(9, 31, 1));
    arm64_jit_emit32(e, arm64_jit_enc_ldr128_uimm(10, 31, 2));
    arm64_jit_emit32(e, arm64_jit_enc_ldr128_uimm(11, 31, 3));
    arm64_jit_emit32(e, arm64_jit_enc_ldr128_uimm(12, 31, 4));
    arm64_jit_emit32(e, arm64_jit_enc_ldr128_uimm(13, 31, 5));
    arm64_jit_emit32(e, arm64_jit_enc_ldr128_uimm(14, 31, 6));
    arm64_jit_emit32(e, arm64_jit_enc_ldr128_uimm(15, 31, 7));
    arm64_jit_emit32(e, 0x910203ff); // add sp, sp, #128
    arm64_jit_emit32(e, 0xa94573fb); // ldp x27, x28, [sp, #80]
    arm64_jit_emit32(e, 0xa9446bf9); // ldp x25, x26, [sp, #64]
    arm64_jit_emit32(e, 0xa94363f7); // ldp x23, x24, [sp, #48]
    arm64_jit_emit32(e, 0xa9425bf5); // ldp x21, x22, [sp, #32]
    arm64_jit_emit32(e, 0xa94153f3); // ldp x19, x20, [sp, #16]
    arm64_jit_emit32(e, 0xa8c67bfd); // ldp x29, x30, [sp], #96
    arm64_jit_emit32(e, arm64_jit_enc_ret(30));
}

static void arm64_jit_emit_state_snippet_ret(struct arm64_jit_emitter *e) {
    arm64_jit_emit32(e, arm64_jit_enc_ret(30));
}

void arm64_jit_emit_helper_return(struct arm64_jit_emitter *e, void *helper, addr_t guest_pc) {
    arm64_jit_emit_load_imm64(e, 1, guest_pc);
    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(0, 21));
    arm64_jit_emit_load_imm64(e, 16, (uint64_t) helper);
    arm64_jit_emit32(e, arm64_jit_enc_blr(16));
    arm64_jit_emit_epilogue(e);
}

static void arm64_jit_emit_helper_return_regarg(struct arm64_jit_emitter *e, void *helper,
        addr_t guest_pc, unsigned x2_imm) {
    arm64_jit_emit_load_imm64(e, 1, guest_pc);
    arm64_jit_emit_load_imm64(e, 2, x2_imm);
    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(0, 21));
    arm64_jit_emit_load_imm64(e, 16, (uint64_t) helper);
    arm64_jit_emit32(e, arm64_jit_enc_blr(16));
    arm64_jit_emit_epilogue(e);
}

static void arm64_jit_emit_helper_continue_check(struct arm64_jit_emitter *e) {
    arm64_jit_emit32(e, 0x11000410); // add w16, w0, #1 ; w16 == 0 when return == INT_NONE (-1)
    // A64 conditional branch immediates are relative to the branch instruction
    // address. Our current epilogue is 16 instructions long, so a successful
    // helper return needs a 17-instruction displacement to land at the
    // continuation path after the epilogue.
    arm64_jit_emit32(e, 0x34000230); // cbz w16, +17 insns
}

static void arm64_jit_emit_helper_return_packed1(struct arm64_jit_emitter *e, void *helper,
        uint64_t x1_imm) {
    arm64_jit_emit_load_imm64(e, 1, x1_imm);
    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(0, 21));
    arm64_jit_emit_load_imm64(e, 16, (uint64_t) helper);
    arm64_jit_emit32(e, arm64_jit_enc_blr(16));
    arm64_jit_emit_epilogue(e);
}

static void arm64_jit_emit_helper_continue_packed1(struct arm64_jit_emitter *e, void *helper,
        uint64_t x1_imm) {
    arm64_jit_emit_load_imm64(e, 1, x1_imm);
    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(0, 21));
    arm64_jit_emit_load_imm64(e, 16, (uint64_t) helper);
    arm64_jit_emit32(e, arm64_jit_enc_blr(16));
    arm64_jit_emit_helper_continue_check(e);
    arm64_jit_emit_epilogue(e);
}

static void arm64_jit_emit_helper_continue_live_store(struct arm64_jit_emitter *e, void *helper,
        uint64_t packed, unsigned addr_reg, unsigned value_reg) {
    arm64_jit_emit_load_imm64(e, 1, packed);
    if (addr_reg != 2)
        arm64_jit_emit32(e, arm64_jit_enc_mov_reg(2, addr_reg));
    if (value_reg != 3)
        arm64_jit_emit32(e, arm64_jit_enc_mov_reg(3, value_reg));
    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(0, 21));
    arm64_jit_emit_load_imm64(e, 16, (uint64_t) helper);
    arm64_jit_emit32(e, arm64_jit_enc_blr(16));
    arm64_jit_emit_helper_continue_check(e);
    arm64_jit_emit_epilogue(e);
    // No reload here: live-cache helper contract preserves cached state on success.
}

static void arm64_jit_emit_helper_continue_live_store_pair(struct arm64_jit_emitter *e, void *helper,
        uint64_t packed, unsigned addr_reg, unsigned value0_reg, unsigned value1_reg) {
    arm64_jit_emit_load_imm64(e, 1, packed);
    if (addr_reg != 2)
        arm64_jit_emit32(e, arm64_jit_enc_mov_reg(2, addr_reg));
    if (value0_reg != 3)
        arm64_jit_emit32(e, arm64_jit_enc_mov_reg(3, value0_reg));
    if (value1_reg != 16)
        arm64_jit_emit32(e, arm64_jit_enc_mov_reg(16, value1_reg));
    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(0, 21));
    arm64_jit_emit_load_imm64(e, 17, (uint64_t) helper);
    arm64_jit_emit32(e, arm64_jit_enc_blr(17));
    arm64_jit_emit_helper_continue_check(e);
    arm64_jit_emit_epilogue(e);
}

static void arm64_jit_emit_helper_continue_live_load(struct arm64_jit_emitter *e, void *helper,
        addr_t guest_pc, uint64_t packed1, unsigned addr_reg, int dst_host, unsigned dst_guest) {
    (void) guest_pc;
    (void) dst_guest;
    arm64_jit_emit_load_imm64(e, 1, packed1);
    if (addr_reg != 2)
        arm64_jit_emit32(e, arm64_jit_enc_mov_reg(2, addr_reg));
    arm64_jit_emit_load_imm64(e, 3, dst_host >= 0 ? 1 : 0);
    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(0, 21));
    arm64_jit_emit_load_imm64(e, 16, (uint64_t) helper);
    arm64_jit_emit32(e, arm64_jit_enc_blr(16));
    arm64_jit_emit_helper_continue_check(e);
    arm64_jit_emit_epilogue(e);
    if (dst_host >= 0 && dst_host != 1)
        arm64_jit_emit32(e, arm64_jit_enc_mov_reg(dst_host, 1));
}

static void arm64_jit_emit_helper_continue_live_load_pair(struct arm64_jit_emitter *e, void *helper,
        uint64_t packed, unsigned addr_reg,
        unsigned dst0_host, unsigned dst0_guest,
        unsigned dst1_host, unsigned dst1_guest) {
    (void) dst0_guest;
    (void) dst1_guest;
    arm64_jit_emit_load_imm64(e, 1, packed);
    if (addr_reg != 2)
        arm64_jit_emit32(e, arm64_jit_enc_mov_reg(2, addr_reg));
    arm64_jit_emit_load_imm64(e, 3, 1);
    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(0, 21));
    arm64_jit_emit_load_imm64(e, 16, (uint64_t) helper);
    arm64_jit_emit32(e, arm64_jit_enc_blr(16));
    arm64_jit_emit_helper_continue_check(e);
    arm64_jit_emit_epilogue(e);
    if (dst0_host != 1)
        arm64_jit_emit32(e, arm64_jit_enc_mov_reg(dst0_host, 1));
    if (dst1_host != 2)
        arm64_jit_emit32(e, arm64_jit_enc_mov_reg(dst1_host, 2));
}

static void arm64_jit_emit_helper_continue_live_addronly(struct arm64_jit_emitter *e, void *helper,
        uint64_t packed, unsigned addr_reg) {
    arm64_jit_emit_load_imm64(e, 1, packed);
    if (addr_reg != 2)
        arm64_jit_emit32(e, arm64_jit_enc_mov_reg(2, addr_reg));
    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(0, 21));
    arm64_jit_emit_load_imm64(e, 16, (uint64_t) helper);
    arm64_jit_emit32(e, arm64_jit_enc_blr(16));
    arm64_jit_emit_helper_continue_check(e);
    arm64_jit_emit_epilogue(e);
}

static void arm64_jit_emit_helper_continue_packed2(struct arm64_jit_emitter *e, void *helper,
        uint64_t x1_imm, uint64_t x2_imm) {
    arm64_jit_emit_load_imm64(e, 1, x1_imm);
    arm64_jit_emit_load_imm64(e, 2, x2_imm);
    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(0, 21));
    arm64_jit_emit_load_imm64(e, 16, (uint64_t) helper);
    arm64_jit_emit32(e, arm64_jit_enc_blr(16));
    arm64_jit_emit_helper_continue_check(e);
    arm64_jit_emit_epilogue(e);
}

static void arm64_jit_emit_helper_return_packed2(struct arm64_jit_emitter *e, void *helper,
        uint64_t x1_imm, uint64_t x2_imm) {
    arm64_jit_emit_load_imm64(e, 1, x1_imm);
    arm64_jit_emit_load_imm64(e, 2, x2_imm);
    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(0, 21));
    arm64_jit_emit_load_imm64(e, 16, (uint64_t) helper);
    arm64_jit_emit32(e, arm64_jit_enc_blr(16));
    arm64_jit_emit_epilogue(e);
}

static void arm64_jit_emit_helper_continue_regarg(struct arm64_jit_emitter *e, void *helper,
        addr_t guest_pc, unsigned x2_imm) {
    arm64_jit_emit_load_imm64(e, 1, guest_pc);
    arm64_jit_emit_load_imm64(e, 2, x2_imm);
    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(0, 21));
    arm64_jit_emit_load_imm64(e, 16, (uint64_t) helper);
    arm64_jit_emit32(e, arm64_jit_enc_blr(16));
    arm64_jit_emit_helper_continue_check(e);
    arm64_jit_emit_epilogue(e);
}

static int arm64_jit_guest_src_host_reg(const struct arm64_jit_block *block, uint32_t guest_reg, bool sp_not_zr) {
    if (guest_reg == 31)
        return sp_not_zr ? ARM64_JIT_HOST_GUEST_SP : -1;
    return arm64_jit_host_reg_for_guest(block, guest_reg);
}

static int arm64_jit_guest_src_host_reg_or_zr(const struct arm64_jit_block *block, uint32_t guest_reg) {
    if (guest_reg == 31)
        return 31;
    return arm64_jit_host_reg_for_guest(block, guest_reg);
}

static uint64_t arm64_jit_rotate_right_u64(uint64_t value, unsigned amount, unsigned width) {
    amount %= width;
    if (amount == 0)
        return value;
    uint64_t mask = (width == 64) ? ~0ULL : ((1ULL << width) - 1);
    value &= mask;
    return ((value >> amount) | (value << (width - amount))) & mask;
}

static int arm64_jit_simd_elem_size_from_imm5_emit(uint32_t imm5) {
    if (imm5 & 0x1)
        return 0;
    if (imm5 & 0x2)
        return 1;
    if (imm5 & 0x4)
        return 2;
    if (imm5 & 0x8)
        return 3;
    return -1;
}

static uint64_t arm64_jit_decode_bitmask_u64(unsigned n, unsigned imms, unsigned immr, bool sf, bool *ok) {
    unsigned len = 0;
    unsigned levels, s, r, esize;
    uint64_t welem, wmask;

    if (!sf && n != 0) {
        *ok = false;
        return 0;
    }
    if (sf && n != 1) {
        *ok = false;
        return 0;
    }

    uint32_t val = (n << 6) | (~imms & 0x3f);
    while (val) {
        len++;
        val >>= 1;
    }
    len = len ? len - 1 : 0;
    if (len < 1) {
        *ok = false;
        return 0;
    }

    esize = 1u << len;
    levels = esize - 1;
    s = imms & levels;
    r = immr & levels;
    if (s == levels) {
        *ok = false;
        return 0;
    }

    welem = (1ULL << (s + 1)) - 1;
    welem = arm64_jit_rotate_right_u64(welem, r, esize);
    wmask = 0;
    for (unsigned i = 0; i < (sf ? 64u : 32u); i += esize)
        wmask |= welem << i;
    *ok = true;
    return wmask;
}

static bool arm64_jit_emit_move_wide_cached(struct arm64_jit_emitter *e, uint32_t insn) {
    if ((insn & 0x1f800000u) != 0x12800000u)
        return false;
    if (!ARM64_SF(insn))
        return false;
    uint32_t rd = ARM64_RD(insn);
    int dst = arm64_jit_host_reg_for_guest(e->block, rd);
    if (dst < 0)
        return false;
    uint32_t opc = (insn >> 29) & 0x3;
    uint32_t hw = (insn >> 21) & 0x3;
    uint16_t imm16 = (insn >> 5) & 0xffff;
    if (opc == 0) { // MOVN
        arm64_jit_emit32(e, 0x92800000u | (hw << 21) | ((uint32_t) imm16 << 5) | (dst & 0x1f));
        return true;
    }
    if (opc == 2) { // MOVZ
        arm64_jit_emit32(e, 0xd2800000u | (hw << 21) | ((uint32_t) imm16 << 5) | (dst & 0x1f));
        return true;
    }
    if (opc == 3) { // MOVK
        arm64_jit_emit32(e, 0xf2800000u | (hw << 21) | ((uint32_t) imm16 << 5) | (dst & 0x1f));
        return true;
    }
    return false;
}

static bool arm64_jit_emit_addsub_imm_cached(struct arm64_jit_emitter *e, uint32_t insn) {
    if (!((insn & 0x1f000000u) == 0x11000000u || (insn & 0x7f000000u) == 0x31000000u))
        return false;
    if (!ARM64_SF(insn))
        return false;
    bool setflags = (insn >> 29) & 1;
    uint32_t shift = (insn >> 22) & 0x3;
    if (shift > 1)
        return false;
    uint32_t rd = ARM64_RD(insn);
    uint32_t rn = ARM64_RN(insn);
    if (setflags) {
        int dst = arm64_jit_guest_src_host_reg_or_zr(e->block, rd);
        int src = arm64_jit_guest_src_host_reg(e->block, rn, true);
        if (dst < 0 || src < 0)
            return false;
        uint32_t emitted = insn;
        emitted &= ~((uint32_t) 0x1f);
        emitted &= ~((uint32_t) 0x1f << 5);
        emitted |= (uint32_t) dst;
        emitted |= (uint32_t) src << 5;
        arm64_jit_emit32(e, emitted);
        return true;
    }
    bool is_sub = (insn >> 30) & 1;
    int dst = (rd == 31) ? ARM64_JIT_HOST_GUEST_SP : arm64_jit_host_reg_for_guest(e->block, rd);
    int src = arm64_jit_guest_src_host_reg(e->block, rn, true);
    if (dst < 0 || src < 0)
        return false;
    uint32_t imm12 = (insn >> 10) & 0xfff;
    uint32_t sh = shift ? 1 : 0;
    uint32_t base = is_sub ? 0xd1000000u : 0x91000000u;
    arm64_jit_emit32(e, base | ((sh & 0x3) << 22) | ((imm12 & 0xfff) << 10) |
            ((src & 0x1f) << 5) | (dst & 0x1f));
    return true;
}

static bool arm64_jit_emit_simd_fp_cached(struct arm64_jit_emitter *e, uint32_t insn) {
    uint32_t simd_rrr = insn & ~(((uint32_t) 0x1f) | ((uint32_t) 0x1f << 5) |
            ((uint32_t) 0x1f << 16));

    switch (simd_rrr) {
        case 0x1e220800u: // FMUL Dd, Dn, Dm
        case 0x1e620800u: // FMUL Dd, Dn, Dm
        case 0x1e221800u: // FDIV Sd, Sn, Sm
        case 0x1e621800u: // FDIV Dd, Dn, Dm
        case 0x1e222800u: // FADD Sd, Sn, Sm
        case 0x1e622800u: // FADD Dd, Dn, Dm
        case 0x1e223800u: // FSUB Sd, Sn, Sm
        case 0x1e623800u: // FSUB Dd, Dn, Dm
        case 0x1e224800u: // FMAX Sd, Sn, Sm
        case 0x1e624800u: // FMAX Dd, Dn, Dm
        case 0x1e225800u: // FMIN Sd, Sn, Sm
        case 0x1e625800u: // FMIN Dd, Dn, Dm
        case 0x1e226800u: // FMAXNM Sd, Sn, Sm
        case 0x1e626800u: // FMAXNM Dd, Dn, Dm
        case 0x1e227800u: // FMINNM Sd, Sn, Sm
        case 0x1e627800u: // FMINNM Dd, Dn, Dm
        case 0x1e220c00u: // FCSEL Sd, Sn, Sm, cond
        case 0x1e620c00u: // FCSEL Dd, Dn, Dm, cond
            arm64_jit_emit32(e, insn);
            return true;
        default:
            break;
    }
    if ((insn & 0xfffffc00u) == 0x1e204000u) { // FMOV Sd, Sn
        arm64_jit_emit32(e, insn);
        return true;
    }
    if ((insn & 0xfffffc00u) == 0x1e604000u) { // FMOV Dd, Dn
        arm64_jit_emit32(e, insn);
        return true;
    }
    if ((insn & 0xfffffc00u) == 0x1e20c000u) { // FABS Sd, Sn
        arm64_jit_emit32(e, insn);
        return true;
    }
    if ((insn & 0xfffffc00u) == 0x1e60c000u) { // FABS Dd, Dn
        arm64_jit_emit32(e, insn);
        return true;
    }
    if ((insn & 0xfffffc00u) == 0x1e214000u) { // FNEG Sd, Sn
        arm64_jit_emit32(e, insn);
        return true;
    }
    if ((insn & 0xfffffc00u) == 0x1e614000u) { // FNEG Dd, Dn
        arm64_jit_emit32(e, insn);
        return true;
    }
    if ((insn & 0xfffffc00u) == 0x1e21c000u) { // FSQRT Sd, Sn
        arm64_jit_emit32(e, insn);
        return true;
    }
    if ((insn & 0xfffffc00u) == 0x1e61c000u) { // FSQRT Dd, Dn
        arm64_jit_emit32(e, insn);
        return true;
    }
    if ((insn & 0xfffffc1fu) == 0x1e212000u) { // FCMP Sn, Sm
        arm64_jit_emit32(e, insn);
        return true;
    }
    if ((insn & 0xfffffc1fu) == 0x1e612000u) { // FCMP Dn, Dm
        arm64_jit_emit32(e, insn);
        return true;
    }
    if ((insn & 0xfffffc1fu) == 0x1e222030u) { // FCMPE Sn, Sm
        arm64_jit_emit32(e, insn);
        return true;
    }
    if ((insn & 0xfffffc1fu) == 0x1e622030u) { // FCMPE Dn, Dm
        arm64_jit_emit32(e, insn);
        return true;
    }
    if ((insn & 0xff80fc00u) == 0x0f000400u || // MOVI .4H / .2S
            (insn & 0xff80fc00u) == 0x4f000400u || // MOVI .8H / .4S
            (insn & 0xff80fc00u) == 0x0f00e400u || // MOVI .8B
            (insn & 0xff80fc00u) == 0x4f00e400u) { // MOVI .16B
        arm64_jit_emit32(e, insn);
        return true;
    }
    if ((insn & 0xbfe0fc00u) == 0x0e000c00u) { // DUP (general) - GPR to vector
        uint32_t rn = ARM64_RN(insn);
        int src = arm64_jit_guest_src_host_reg_or_zr(e->block, rn);
        if (src < 0) {
            if (rn >= 31)
                return false;
            src = ARM64_JIT_HOST_HELPER0;
            arm64_jit_emit32(e, arm64_jit_enc_ldr64_uimm((unsigned) src, ARM64_JIT_HOST_CPU,
                    (CPU_OFFSET(regs[rn]) >> 3)));
        }
        uint32_t emitted = insn & ~((uint32_t) 0x1f << 5);
        emitted |= (uint32_t) src << 5;
        arm64_jit_emit32(e, emitted);
        return true;
    }
    if ((insn & 0xbfe0fc00u) == 0x0e000400u) { // DUP (element) - vector to vector
        arm64_jit_emit32(e, insn);
        return true;
    }
    if ((insn & 0xffe0fc00u) == 0x5e000400u) { // DUP (element, scalar)
        arm64_jit_emit32(e, insn);
        return true;
    }
    if ((insn & 0xffe08400u) == 0x6e000400u) { // INS (element) vector-to-vector
        arm64_jit_emit32(e, insn);
        return true;
    }
    if ((insn & 0xffe0fc00u) == 0x4e003c00u) { // MOV Xd, Vn.D[0]
        uint32_t rd = ARM64_RD(insn);
        int dst = arm64_jit_host_reg_for_guest(e->block, rd);
        bool storeback = false;
        if (dst < 0) {
            if (rd >= 31)
                return false;
            dst = ARM64_JIT_HOST_HELPER0;
            storeback = true;
        }
        uint32_t emitted = insn & ~((uint32_t) 0x1f);
        emitted |= (uint32_t) dst;
        arm64_jit_emit32(e, emitted);
        if (storeback) {
            arm64_jit_emit32(e, arm64_jit_enc_str64_uimm((unsigned) dst, ARM64_JIT_HOST_CPU,
                    (CPU_OFFSET(regs[rd]) >> 3)));
        }
        return true;
    }
    if ((insn & 0xbfe0fc00u) == 0x0e003c00u) { // UMOV/MOV scalar from vector to GPR
        uint32_t imm5 = (insn >> 16) & 0x1f;
        uint32_t rd = ARM64_RD(insn);
        int elem_size = arm64_jit_simd_elem_size_from_imm5_emit(imm5);
        if (elem_size < 0)
            return false;
        int dst = arm64_jit_host_reg_for_guest(e->block, rd);
        bool storeback = false;
        if (dst < 0) {
            if (rd >= 31)
                return false;
            dst = ARM64_JIT_HOST_HELPER0;
            storeback = true;
        }
        uint32_t emitted = insn & ~((uint32_t) 0x1f);
        emitted |= (uint32_t) dst;
        arm64_jit_emit32(e, emitted);
        if (storeback) {
            if (elem_size == 3) {
                arm64_jit_emit32(e, arm64_jit_enc_str64_uimm((unsigned) dst, ARM64_JIT_HOST_CPU,
                        (CPU_OFFSET(regs[rd]) >> 3)));
            } else {
                arm64_jit_emit32(e, arm64_jit_enc_str32_uimm((unsigned) dst, ARM64_JIT_HOST_CPU,
                        (CPU_OFFSET(regs[rd]) >> 3)));
            }
        }
        return true;
    }
    if ((insn & 0xffe0fc00u) == 0x4e001c00u && ((insn >> 16) & 0xf) == 0x8) { // MOV Vd.D[idx], Xn
        uint32_t rn = ARM64_RN(insn);
        int src = arm64_jit_guest_src_host_reg_or_zr(e->block, rn);
        if (src < 0) {
            if (rn >= 31)
                return false;
            src = ARM64_JIT_HOST_HELPER0;
            arm64_jit_emit32(e, arm64_jit_enc_ldr64_uimm((unsigned) src, ARM64_JIT_HOST_CPU,
                    (CPU_OFFSET(regs[rn]) >> 3)));
        }
        uint32_t emitted = insn & ~((uint32_t) 0x1f << 5);
        emitted |= (uint32_t) src << 5;
        arm64_jit_emit32(e, emitted);
        return true;
    }
    if ((insn & 0xffe7fc00u) == 0x4e041c00u) { // MOV Vd.S[idx], Wn
        uint32_t rn = ARM64_RN(insn);
        int src = arm64_jit_guest_src_host_reg_or_zr(e->block, rn);
        if (src < 0) {
            if (rn >= 31)
                return false;
            src = ARM64_JIT_HOST_HELPER0;
            arm64_jit_emit32(e, arm64_jit_enc_ldr64_uimm((unsigned) src, ARM64_JIT_HOST_CPU,
                    (CPU_OFFSET(regs[rn]) >> 3)));
        }
        uint32_t emitted = insn & ~((uint32_t) 0x1f << 5);
        emitted |= (uint32_t) src << 5;
        arm64_jit_emit32(e, emitted);
        return true;
    }
    if ((insn & 0xffe0fc00u) == 0x4e001c00u && ((insn >> 16) & 0x3) == 0x2) { // MOV Vd.H[idx], Wn
        uint32_t rn = ARM64_RN(insn);
        int src = arm64_jit_guest_src_host_reg_or_zr(e->block, rn);
        if (src < 0) {
            if (rn >= 31)
                return false;
            src = ARM64_JIT_HOST_HELPER0;
            arm64_jit_emit32(e, arm64_jit_enc_ldr64_uimm((unsigned) src, ARM64_JIT_HOST_CPU,
                    (CPU_OFFSET(regs[rn]) >> 3)));
        }
        uint32_t emitted = insn & ~((uint32_t) 0x1f << 5);
        emitted |= (uint32_t) src << 5;
        arm64_jit_emit32(e, emitted);
        return true;
    }
    if ((insn & 0xffe0fc00u) == 0x4e001c00u && ((insn >> 16) & 0x1) == 0x1) { // MOV Vd.B[idx], Wn
        uint32_t rn = ARM64_RN(insn);
        int src = arm64_jit_guest_src_host_reg_or_zr(e->block, rn);
        if (src < 0) {
            if (rn >= 31)
                return false;
            src = ARM64_JIT_HOST_HELPER0;
            arm64_jit_emit32(e, arm64_jit_enc_ldr64_uimm((unsigned) src, ARM64_JIT_HOST_CPU,
                    (CPU_OFFSET(regs[rn]) >> 3)));
        }
        uint32_t emitted = insn & ~((uint32_t) 0x1f << 5);
        emitted |= (uint32_t) src << 5;
        arm64_jit_emit32(e, emitted);
        return true;
    }
    switch (simd_rrr) {
        case 0x4e221c00u: // AND Vd.16B, Vn.16B, Vm.16B
        case 0x4ea21c00u: // ORR Vd.16B, Vn.16B, Vm.16B
        case 0x6e221c00u: // EOR Vd.16B, Vn.16B, Vm.16B
        case 0x4e621c00u: // BIC Vd.16B, Vn.16B, Vm.16B
        case 0x4e228400u: // ADD Vd.16B, Vn.16B, Vm.16B
        case 0x6e228400u: // SUB Vd.16B, Vn.16B, Vm.16B
        case 0x6e228c00u: // CMEQ Vd.16B, Vn.16B, Vm.16B
        case 0x6e223400u: // CMHI Vd.16B, Vn.16B, Vm.16B
        case 0x4e223400u: // CMGT Vd.16B, Vn.16B, Vm.16B
        case 0x6e226c00u: // UMIN Vd.16B, Vn.16B, Vm.16B
        case 0x4e226400u: // SMAX Vd.16B, Vn.16B, Vm.16B
        case 0x4e023800u: // ZIP1 Vd.16B, Vn.16B, Vm.16B
        case 0x4e027800u: // ZIP2 Vd.16B, Vn.16B, Vm.16B
        case 0x4e021800u: // UZP1 Vd.16B, Vn.16B, Vm.16B
        case 0x4e025800u: // UZP2 Vd.16B, Vn.16B, Vm.16B
        case 0x4e022800u: // TRN1 Vd.16B, Vn.16B, Vm.16B
        case 0x4e026800u: // TRN2 Vd.16B, Vn.16B, Vm.16B
        case 0x4e020000u: // TBL Vd.16B, {Vn.16B}, Vm.16B
        case 0x4e021000u: // TBX Vd.16B, {Vn.16B}, Vm.16B
            arm64_jit_emit32(e, insn);
            return true;
        default:
            break;
    }
    if ((insn & 0xff20fc00u) == 0x6e001800u) { // EXT Vd.16B, Vn.16B, Vm.16B, #imm
        arm64_jit_emit32(e, insn);
        return true;
    }
    if ((insn & 0xfffffc00u) == 0x6e205800u) { // MVN/NOT Vd.16B, Vn.16B
        arm64_jit_emit32(e, insn);
        return true;
    }
    if ((insn & 0xfffffc00u) == 0x4e205800u) { // CNT Vd.16B, Vn.16B
        arm64_jit_emit32(e, insn);
        return true;
    }
    if ((insn & 0xfffffc00u) == 0x4e200800u) { // REV64 Vd.16B, Vn.16B
        arm64_jit_emit32(e, insn);
        return true;
    }
    return false;
}

static bool arm64_jit_emit_logical_imm_cached(struct arm64_jit_emitter *e, uint32_t insn) {
    if ((insn & 0x1f800000u) != 0x12000000u)
        return false;

    uint32_t opc = (insn >> 29) & 0x3;
    uint32_t rd = ARM64_RD(insn);
    uint32_t rn = ARM64_RN(insn);
    int dst = (opc == 3) ? arm64_jit_guest_src_host_reg_or_zr(e->block, rd)
                         : arm64_jit_guest_src_host_reg(e->block, rd, true);
    int src = arm64_jit_guest_src_host_reg_or_zr(e->block, rn);
    if (dst < 0 || src < 0)
        return false;

    uint32_t emitted = insn;
    emitted &= ~((uint32_t) 0x1f);
    emitted &= ~((uint32_t) 0x1f << 5);
    emitted |= (uint32_t) dst;
    emitted |= (uint32_t) src << 5;
    arm64_jit_emit32(e, emitted);
    return true;
}

static bool arm64_jit_emit_adr_cached(struct arm64_jit_emitter *e, uint32_t insn, addr_t guest_pc) {
    if (!((insn & 0x1f000000u) == 0x10000000u || (insn & 0x9f000000u) == 0x90000000u))
        return false;

    uint32_t rd = ARM64_RD(insn);
    int dst = arm64_jit_host_reg_for_guest(e->block, rd);
    if (dst < 0)
        return false;

    int64_t imm = arm64_adr_imm(insn);
    uint64_t base = guest_pc;
    if (insn & 0x80000000u) {
        base &= ~0xfffULL;
        imm <<= 12;
    }
    arm64_jit_emit_load_imm64(e, (unsigned) dst, base + imm);
    return true;
}

static bool arm64_jit_emit_bitfield_cached(struct arm64_jit_emitter *e, uint32_t insn) {
    if ((insn & 0x1f800000u) != 0x13000000u)
        return false;

    uint32_t rd = ARM64_RD(insn);
    uint32_t rn = ARM64_RN(insn);
    int dst = arm64_jit_guest_src_host_reg(e->block, rd, false);
    int src = arm64_jit_guest_src_host_reg(e->block, rn, false);
    if (dst < 0 || src < 0)
        return false;

    uint32_t emitted = insn;
    emitted &= ~((uint32_t) 0x1f);
    emitted &= ~((uint32_t) 0x1f << 5);
    emitted |= (uint32_t) dst;
    emitted |= (uint32_t) src << 5;
    arm64_jit_emit32(e, emitted);
    return true;
}

static bool arm64_jit_emit_logical_shifted_cached(struct arm64_jit_emitter *e, uint32_t insn) {
    if (((insn >> 24) & 0x1f) != 0x0a)
        return false;

    uint32_t rd = ARM64_RD(insn);
    uint32_t rn = ARM64_RN(insn);
    uint32_t rm = ARM64_RM(insn);
    int dst = arm64_jit_guest_src_host_reg_or_zr(e->block, rd);
    int src1 = arm64_jit_guest_src_host_reg_or_zr(e->block, rn);
    int src2 = arm64_jit_guest_src_host_reg_or_zr(e->block, rm);
    if (dst < 0 || src1 < 0 || src2 < 0)
        return false;

    uint32_t emitted = insn;
    emitted &= ~((uint32_t) 0x1f);
    emitted &= ~((uint32_t) 0x1f << 5);
    emitted &= ~((uint32_t) 0x1f << 16);
    emitted |= (uint32_t) dst;
    emitted |= (uint32_t) src1 << 5;
    emitted |= (uint32_t) src2 << 16;
    arm64_jit_emit32(e, emitted);
    return true;
}

static bool arm64_jit_emit_addsub_shifted_cached(struct arm64_jit_emitter *e, uint32_t insn) {
    if (((insn >> 24) & 0x1f) != 0x0b)
        return false;
    if (((insn >> 22) & 0x3) == 3)
        return false;
    if (arm64_jit_trace_mode() && e->block->start_pc == 0xefeb36d0 && insn == 0xeb03005f)
        return false;

    uint32_t rd = ARM64_RD(insn);
    uint32_t rn = ARM64_RN(insn);
    uint32_t rm = ARM64_RM(insn);
    int dst = arm64_jit_guest_src_host_reg_or_zr(e->block, rd);
    int src1 = arm64_jit_guest_src_host_reg_or_zr(e->block, rn);
    int src2 = arm64_jit_guest_src_host_reg_or_zr(e->block, rm);
    if (dst < 0 || src1 < 0 || src2 < 0)
        return false;

    uint32_t emitted = insn;
    emitted &= ~((uint32_t) 0x1f);
    emitted &= ~((uint32_t) 0x1f << 5);
    emitted &= ~((uint32_t) 0x1f << 16);
    emitted |= (uint32_t) dst;
    emitted |= (uint32_t) src1 << 5;
    emitted |= (uint32_t) src2 << 16;
    arm64_jit_emit32(e, emitted);
    return true;
}

static bool arm64_jit_emit_dp_2src_cached(struct arm64_jit_emitter *e, uint32_t insn) {
    if ((insn & 0x5fe00000u) != 0x1ac00000u)
        return false;

    uint32_t opcode = (insn >> 10) & 0x3f;
    switch (opcode) {
        case 0x02: // UDIV
        case 0x03: // SDIV
        case 0x08: // LSLV
        case 0x09: // LSRV
        case 0x0a: // ASRV
        case 0x0b: // RORV
            break;
        default:
            return false;
    }

    uint32_t rd = ARM64_RD(insn);
    uint32_t rn = ARM64_RN(insn);
    uint32_t rm = ARM64_RM(insn);
    int dst = arm64_jit_guest_src_host_reg(e->block, rd, false);
    int src1 = arm64_jit_guest_src_host_reg(e->block, rn, false);
    int src2 = arm64_jit_guest_src_host_reg(e->block, rm, false);
    if (dst < 0 || src1 < 0 || src2 < 0)
        return false;

    uint32_t emitted = insn;
    emitted &= ~((uint32_t) 0x1f);
    emitted &= ~((uint32_t) 0x1f << 5);
    emitted &= ~((uint32_t) 0x1f << 16);
    emitted |= (uint32_t) dst;
    emitted |= (uint32_t) src1 << 5;
    emitted |= (uint32_t) src2 << 16;
    arm64_jit_emit32(e, emitted);
    return true;
}

static bool arm64_jit_emit_dp_1src_cached(struct arm64_jit_emitter *e, uint32_t insn) {
    if (((insn >> 30) & 1) != 1 || ((insn >> 28) & 1) != 1 || ((insn >> 21) & 0xf) != 0x6)
        return false;

    uint32_t S = (insn >> 29) & 1;
    uint32_t opcode2 = (insn >> 16) & 0x1f;
    uint32_t opcode = (insn >> 10) & 0x3f;
    if (S != 0 || opcode2 != 0)
        return false;

    switch (opcode) {
        case 0x00: // RBIT
        case 0x01: // REV16
        case 0x02: // REV / REV32
        case 0x03: // REV (64-bit only)
        case 0x04: // CLZ
        case 0x05: // CLS
            break;
        default:
            return false;
    }
    if (!ARM64_SF(insn) && opcode == 0x03)
        return false;

    uint32_t rd = ARM64_RD(insn);
    uint32_t rn = ARM64_RN(insn);
    int dst = arm64_jit_guest_src_host_reg(e->block, rd, false);
    int src = arm64_jit_guest_src_host_reg(e->block, rn, false);
    if (dst < 0 || src < 0)
        return false;

    uint32_t emitted = insn;
    emitted &= ~((uint32_t) 0x1f);
    emitted &= ~((uint32_t) 0x1f << 5);
    emitted |= (uint32_t) dst;
    emitted |= (uint32_t) src << 5;
    arm64_jit_emit32(e, emitted);
    return true;
}

static bool arm64_jit_emit_addsub_extended_cached(struct arm64_jit_emitter *e, uint32_t insn) {
    if ((insn & 0x1f200000u) != 0x0b200000u)
        return false;
    bool setflags = ((insn >> 29) & 1) != 0;
    uint32_t option = (insn >> 13) & 0x7;

    uint32_t rd = ARM64_RD(insn);
    uint32_t rn = ARM64_RN(insn);
    uint32_t rm = ARM64_RM(insn);
    int dst = setflags ? arm64_jit_guest_src_host_reg_or_zr(e->block, rd)
                       : ((rd == 31) ? ARM64_JIT_HOST_GUEST_SP : arm64_jit_host_reg_for_guest(e->block, rd));
    bool rn_is_sp = !setflags;
    if ((option & 0x3) == 3)
        rn_is_sp = true;
    int src1 = arm64_jit_guest_src_host_reg(e->block, rn, rn_is_sp);
    int src2 = arm64_jit_guest_src_host_reg_or_zr(e->block, rm);
    if (dst < 0 || src1 < 0 || src2 < 0)
        return false;

    uint32_t emitted = insn;
    emitted &= ~((uint32_t) 0x1f);
    emitted &= ~((uint32_t) 0x1f << 5);
    emitted &= ~((uint32_t) 0x1f << 16);
    emitted |= (uint32_t) dst;
    emitted |= (uint32_t) src1 << 5;
    emitted |= (uint32_t) src2 << 16;
    arm64_jit_emit32(e, emitted);
    return true;
}

static bool arm64_jit_emit_csel_cached(struct arm64_jit_emitter *e, uint32_t insn) {
    if ((insn & 0x1fe00000u) != 0x1a800000u)
        return false;
    if (((insn >> 29) & 1) != 0)
        return false;

    uint32_t rd = ARM64_RD(insn);
    uint32_t rn = ARM64_RN(insn);
    uint32_t rm = ARM64_RM(insn);
    int dst = arm64_jit_guest_src_host_reg_or_zr(e->block, rd);
    int src1 = arm64_jit_guest_src_host_reg_or_zr(e->block, rn);
    int src2 = arm64_jit_guest_src_host_reg_or_zr(e->block, rm);
    if (dst < 0 || src1 < 0 || src2 < 0)
        return false;

    uint32_t emitted = insn;
    emitted &= ~((uint32_t) 0x1f);
    emitted &= ~((uint32_t) 0x1f << 5);
    emitted &= ~((uint32_t) 0x1f << 16);
    emitted |= (uint32_t) dst;
    emitted |= (uint32_t) src1 << 5;
    emitted |= (uint32_t) src2 << 16;
    arm64_jit_emit32(e, emitted);
    return true;
}

static bool arm64_jit_emit_adc_cached(struct arm64_jit_emitter *e, uint32_t insn) {
    if ((insn & 0x1fe0fc00u) != 0x1a000000u)
        return false;

    uint32_t rd = ARM64_RD(insn);
    uint32_t rn = ARM64_RN(insn);
    uint32_t rm = ARM64_RM(insn);
    int dst = arm64_jit_guest_src_host_reg_or_zr(e->block, rd);
    int src1 = arm64_jit_guest_src_host_reg_or_zr(e->block, rn);
    int src2 = arm64_jit_guest_src_host_reg_or_zr(e->block, rm);
    if (dst < 0 || src1 < 0 || src2 < 0)
        return false;

    uint32_t emitted = insn;
    emitted &= ~((uint32_t) 0x1f);
    emitted &= ~((uint32_t) 0x1f << 5);
    emitted &= ~((uint32_t) 0x1f << 16);
    emitted |= (uint32_t) dst;
    emitted |= (uint32_t) src1 << 5;
    emitted |= (uint32_t) src2 << 16;
    arm64_jit_emit32(e, emitted);
    return true;
}

static bool arm64_jit_emit_ccmp_cached(struct arm64_jit_emitter *e, uint32_t insn) {
    if ((insn & 0x3fe00410u) != 0x3a400000u)
        return false;

    bool is_imm = ((insn >> 11) & 1) != 0;
    uint32_t rn = ARM64_RN(insn);
    int src1 = arm64_jit_guest_src_host_reg(e->block, rn, false);
    if (src1 < 0)
        return false;

    uint32_t emitted = insn;
    emitted &= ~((uint32_t) 0x1f << 5);
    emitted |= (uint32_t) src1 << 5;
    if (!is_imm) {
        uint32_t rm = (insn >> 16) & 0x1f;
        int src2 = arm64_jit_guest_src_host_reg(e->block, rm, false);
        if (src2 < 0)
            return false;
        emitted &= ~((uint32_t) 0x1f << 16);
        emitted |= (uint32_t) src2 << 16;
    }
    arm64_jit_emit32(e, emitted);
    return true;
}

static bool arm64_jit_emit_dp_3src_cached(struct arm64_jit_emitter *e, uint32_t insn) {
    if ((insn & 0x1f000000u) != 0x1b000000u)
        return false;

    uint32_t op54 = (insn >> 29) & 0x3;
    uint32_t op31 = (insn >> 21) & 0x7;
    uint32_t o0 = (insn >> 15) & 1;
    if (op54 != 0)
        return false;

    switch (op31) {
        case 0: // MADD/MSUB (32/64)
        case 1: // SMADDL/SMSUBL
        case 5: // UMADDL/UMSUBL
            break;
        case 2: // SMULH
        case 6: // UMULH
            if (o0 != 0)
                return false;
            break;
        default:
            return false;
    }

    uint32_t rd = ARM64_RD(insn);
    uint32_t rn = ARM64_RN(insn);
    uint32_t rm = ARM64_RM(insn);
    uint32_t ra = (insn >> 10) & 0x1f;
    int dst = arm64_jit_host_reg_for_guest(e->block, rd);
    int src1 = arm64_jit_guest_src_host_reg_or_zr(e->block, rn);
    int src2 = arm64_jit_guest_src_host_reg_or_zr(e->block, rm);
    int acc = arm64_jit_guest_src_host_reg_or_zr(e->block, ra);
    if (dst < 0 || src1 < 0 || src2 < 0 || acc < 0)
        return false;

    uint32_t emitted = insn;
    emitted &= ~((uint32_t) 0x1f);
    emitted &= ~((uint32_t) 0x1f << 5);
    emitted &= ~((uint32_t) 0x1f << 16);
    emitted &= ~((uint32_t) 0x1f << 10);
    emitted |= (uint32_t) dst;
    emitted |= (uint32_t) src1 << 5;
    emitted |= (uint32_t) src2 << 16;
    emitted |= (uint32_t) acc << 10;
    arm64_jit_emit32(e, emitted);
    return true;
}

static bool arm64_jit_emit_system_cached(struct arm64_jit_emitter *e, uint32_t insn) {
    if ((insn & 0xffffffe0U) == 0xd53bd040U) { // mrs xt, tpidr_el0
        uint32_t rd = ARM64_RD(insn);
        int dst = arm64_jit_host_reg_for_guest(e->block, rd);
        if (dst < 0)
            return false;
        arm64_jit_emit32(e, arm64_jit_enc_ldr64_uimm((unsigned) dst, ARM64_JIT_HOST_CPU,
                (CPU_OFFSET(tls_ptr) >> 3)));
        return true;
    }
    if ((insn & 0xffffffe0U) == 0xd51bd040U) { // msr tpidr_el0, xt
        uint32_t rt = ARM64_RT(insn);
        int src = arm64_jit_guest_src_host_reg(e->block, rt, false);
        if (src < 0)
            return false;
        arm64_jit_emit32(e, arm64_jit_enc_str64_uimm((unsigned) src, ARM64_JIT_HOST_CPU,
                (CPU_OFFSET(tls_ptr) >> 3)));
        return true;
    }
    if ((insn & 0xfffffff0U) == 0xd50340d0U || (insn & 0xfffffff0U) == 0xd50341d0U) {
        return true; // msr daifset/clr -> guest-local no-op
    }
    if ((insn & 0xfff00000U) == 0xd5300000U) { // generic MRS subset
        uint32_t op0 = (insn >> 19) & 0x3;
        uint32_t op1 = (insn >> 16) & 0x7;
        uint32_t crn = (insn >> 12) & 0xf;
        uint32_t crm = (insn >> 8) & 0xf;
        uint32_t op2 = (insn >> 5) & 0x7;
        uint32_t rd = ARM64_RD(insn);
        int dst = arm64_jit_host_reg_for_guest(e->block, rd);
        if (dst < 0)
            return false;

        if (op0 == 3 && op1 == 3 && crn == 4 && crm == 4 && op2 == 0) { // FPCR
            arm64_jit_emit32(e, arm64_jit_enc_ldr32_uimm((unsigned) dst, ARM64_JIT_HOST_CPU,
                    (CPU_OFFSET(fpcr) >> 2)));
            return true;
        }
        if (op0 == 3 && op1 == 3 && crn == 4 && crm == 4 && op2 == 1) { // FPSR
            arm64_jit_emit32(e, arm64_jit_enc_ldr32_uimm((unsigned) dst, ARM64_JIT_HOST_CPU,
                    (CPU_OFFSET(fpsr) >> 2)));
            return true;
        }

        uint64_t value = 0;
        bool known = true;
        if (op0 == 3 && op1 == 3 && crn == 0 && crm == 0 && op2 == 1)
            value = 0x8444c004ULL; // CTR_EL0
        else if (op0 == 3 && op1 == 3 && crn == 0 && crm == 0 && op2 == 7)
            value = 0x14ULL; // DCZID_EL0
        else if (op0 == 3 && op1 == 3 && crn == 2 && crm == 4 && op2 == 0)
            value = 0ULL; // RNDR
        else if (op0 == 3 && op1 == 3 && crn == 2 && crm == 4 && op2 == 1)
            value = 0ULL; // RNDRRS
        else if (op0 == 3 && op1 == 0 && crn == 0 && crm == 4 && op2 == 0)
            value = 0x1000220000000000ULL; // ID_AA64PFR0_EL1
        else if (op0 == 3 && op1 == 0 && crn == 0 && crm == 4 && op2 == 1)
            value = 0ULL; // ID_AA64PFR1_EL1
        else if (op0 == 3 && op1 == 0 && crn == 0 && crm == 6 && op2 == 0)
            value = 0x001011110121ULL; // ID_AA64ISAR0_EL1
        else if (op0 == 3 && op1 == 0 && crn == 0 && crm == 6 && op2 == 1)
            value = 0x1100001110211111ULL; // ID_AA64ISAR1_EL1
        else if (op0 == 3 && op1 == 0 && crn == 0 && crm == 4 && op2 == 4)
            value = 0ULL; // ID_AA64ZFR0_EL1
        else if (op0 == 3 && op1 == 3 && crn == 14 && crm == 0 && op2 == 2)
            value = 0ULL; // CNTVCT_EL0
        else if (op0 == 3 && op1 == 3 && crn == 14 && crm == 0 && op2 == 0)
            value = 24000000ULL; // CNTFRQ_EL0
        else
            known = false;

        if (!known)
            return false;
        arm64_jit_emit_load_imm64(e, (unsigned) dst, value);
        return true;
    }
    return false;
}

enum arm64_jit_emit_result arm64_jit_emit_dp_imm(struct arm64_jit_emitter *e, uint32_t insn, addr_t guest_pc) {
    if (arm64_jit_emit_move_wide_cached(e, insn))
        return ARM64_JIT_EMIT_CONTINUE;
    if (arm64_jit_emit_adr_cached(e, insn, guest_pc))
        return ARM64_JIT_EMIT_CONTINUE;
    if (arm64_jit_emit_addsub_imm_cached(e, insn))
        return ARM64_JIT_EMIT_CONTINUE;
    if (arm64_jit_emit_logical_imm_cached(e, insn))
        return ARM64_JIT_EMIT_CONTINUE;
    if (arm64_jit_emit_bitfield_cached(e, insn))
        return ARM64_JIT_EMIT_CONTINUE;
    arm64_jit_emit_helper_continue_regarg(e, arm64_jit_helper_exec_dp_imm_jitabi, guest_pc, insn);
    return ARM64_JIT_EMIT_CONTINUE;
}

enum arm64_jit_emit_result arm64_jit_emit_dp_reg(struct arm64_jit_emitter *e, uint32_t insn, addr_t guest_pc) {
    if (arm64_jit_emit_logical_shifted_cached(e, insn)) {
        return ARM64_JIT_EMIT_CONTINUE;
    }
    if (arm64_jit_emit_addsub_extended_cached(e, insn)) {
        return ARM64_JIT_EMIT_CONTINUE;
    }
    if (arm64_jit_emit_addsub_shifted_cached(e, insn)) {
        return ARM64_JIT_EMIT_CONTINUE;
    }
    if (arm64_jit_emit_dp_3src_cached(e, insn))
        return ARM64_JIT_EMIT_CONTINUE;
    if (arm64_jit_emit_csel_cached(e, insn))
        return ARM64_JIT_EMIT_CONTINUE;
    if (arm64_jit_emit_adc_cached(e, insn))
        return ARM64_JIT_EMIT_CONTINUE;
    if (arm64_jit_emit_ccmp_cached(e, insn))
        return ARM64_JIT_EMIT_CONTINUE;
    if (arm64_jit_emit_dp_2src_cached(e, insn))
        return ARM64_JIT_EMIT_CONTINUE;
    if (arm64_jit_emit_dp_1src_cached(e, insn))
        return ARM64_JIT_EMIT_CONTINUE;
    arm64_jit_emit_helper_continue_regarg(e, arm64_jit_helper_exec_dp_reg_jitabi, guest_pc, insn);
    return ARM64_JIT_EMIT_CONTINUE;
}

enum arm64_jit_emit_result arm64_jit_emit_branch(struct arm64_jit_emitter *e, uint32_t insn, addr_t guest_pc) {
    uint32_t op = (insn >> 26) & 0x3f;
    if (op == 0x05 || op == 0x25) {
        addr_t target = guest_pc + arm64_branch_imm26(insn);
        bool is_link = op == 0x25;
        if (!arm64_jit_local_fixup_disabled(e->block, guest_pc) &&
                !is_link && arm64_jit_block_has_pc(e->block, target)) {
            arm64_jit_emit_local_fixup(e, guest_pc, target, ARM64_JIT_FIXUP_B);
            arm64_jit_emit32(e, arm64_jit_enc_b_imm(0));
            return ARM64_JIT_EMIT_CONTINUE;
        }
        if (is_link) {
            uint64_t packed = ((uint64_t) target << 32) | (uint32_t) guest_pc;
            arm64_jit_emit_helper_return_packed1(e, arm64_jit_helper_branch_link_jitabi, packed);
            return ARM64_JIT_EMIT_TERMINATE;
        }
        arm64_jit_emit_helper_return(e, arm64_jit_helper_dispatch_jitabi, target);
        return ARM64_JIT_EMIT_TERMINATE;
    }
    if ((insn & 0x7e000000u) == 0x34000000u) {
        if (!arm64_jit_local_fixup_disabled(e->block, guest_pc)) {
            addr_t target = guest_pc + arm64_branch_imm19(insn);
            uint32_t rt = ARM64_RT(insn);
            int host_rt = arm64_jit_guest_src_host_reg(e->block, rt, false);
            if (host_rt >= 0 && arm64_jit_block_has_pc(e->block, target)) {
                bool sf = ((insn >> 31) & 1) != 0;
                bool nonzero = ((insn >> 24) & 1) != 0;
                arm64_jit_emit_local_fixup(e, guest_pc, target, ARM64_JIT_FIXUP_CBZ);
                arm64_jit_emit32(e, arm64_jit_enc_cbz_cbnz(sf, nonzero, (unsigned) host_rt, 0));
                return ARM64_JIT_EMIT_CONTINUE;
            }
        }
        arm64_jit_emit_helper_return_regarg(e, arm64_jit_helper_cbz_cbnz_jitabi, guest_pc, insn);
        return ARM64_JIT_EMIT_TERMINATE;
    }
    if ((insn & 0xff000010u) == 0x54000000u) {
        if (!arm64_jit_local_fixup_disabled(e->block, guest_pc)) {
            addr_t target = guest_pc + arm64_branch_imm19(insn);
            if (arm64_jit_block_has_pc(e->block, target)) {
                arm64_jit_emit_local_fixup(e, guest_pc, target, ARM64_JIT_FIXUP_B_COND);
                arm64_jit_emit32(e, insn);
                return ARM64_JIT_EMIT_CONTINUE;
            }
        }
        arm64_jit_emit_helper_return_regarg(e, arm64_jit_helper_b_cond_jitabi, guest_pc, insn);
        return ARM64_JIT_EMIT_TERMINATE;
    }
    if ((insn & 0x7e000000u) == 0x36000000u) {
        if (!arm64_jit_local_fixup_disabled(e->block, guest_pc)) {
            addr_t target = guest_pc + arm64_branch_imm14(insn);
            uint32_t rt = ARM64_RT(insn);
            int host_rt = arm64_jit_guest_src_host_reg(e->block, rt, false);
            if (host_rt >= 0 && arm64_jit_block_has_pc(e->block, target)) {
                bool b5 = ((insn >> 31) & 1) != 0;
                bool nonzero = ((insn >> 24) & 1) != 0;
                unsigned bit40 = (insn >> 19) & 0x1f;
                arm64_jit_emit_local_fixup(e, guest_pc, target, ARM64_JIT_FIXUP_TBZ);
                arm64_jit_emit32(e, arm64_jit_enc_tbz_tbnz(b5, nonzero, bit40,
                        (unsigned) host_rt, 0));
                return ARM64_JIT_EMIT_CONTINUE;
            }
        }
        arm64_jit_emit_helper_return_regarg(e, arm64_jit_helper_tbz_tbnz_jitabi, guest_pc, insn);
        return ARM64_JIT_EMIT_TERMINATE;
    }
    if ((insn & 0xfe000000u) == 0xd6000000u) {
        arm64_jit_emit_helper_return_regarg(e, arm64_jit_helper_branch_reg_jitabi, guest_pc, insn);
        return ARM64_JIT_EMIT_TERMINATE;
    }
    return ARM64_JIT_EMIT_UNSUPPORTED;
}

enum arm64_jit_emit_result arm64_jit_emit_exception(struct arm64_jit_emitter *e, uint32_t insn, addr_t guest_pc) {
    (void) insn;
    arm64_jit_emit_helper_return(e, arm64_jit_helper_syscall_jitabi, guest_pc);
    return ARM64_JIT_EMIT_TERMINATE;
}

enum arm64_jit_emit_result arm64_jit_emit_system(struct arm64_jit_emitter *e, uint32_t insn, addr_t guest_pc) {
    if (arm64_jit_emit_system_cached(e, insn)) {
        if (arm64_jit_trace_mode() && guest_pc == 0xefe62638) {
            uint32_t rd = ARM64_RD(insn);
            int dst = arm64_jit_host_reg_for_guest(e->block, rd);
            fprintf(stderr, "[arm64-jit] direct system pc=0x%llx rd=%u dst=x%d\n",
                    (unsigned long long) guest_pc, rd, dst);
        }
        return ARM64_JIT_EMIT_CONTINUE;
    }

    if ((insn & 0xffffffe0U) == 0xd53bd040U) { // mrs xt, tpidr_el0
        arm64_jit_emit_helper_continue_regarg(e, arm64_jit_helper_mrs_tpidr_jitabi, guest_pc, ARM64_RD(insn));
        return ARM64_JIT_EMIT_CONTINUE;
    }
    if ((insn & 0xffffffe0U) == 0xd51bd040U) { // msr tpidr_el0, xt
        arm64_jit_emit_helper_continue_regarg(e, arm64_jit_helper_msr_tpidr_jitabi, guest_pc, ARM64_RT(insn));
        return ARM64_JIT_EMIT_CONTINUE;
    }
    if ((insn & 0xfffffff0U) == 0xd50340d0U || (insn & 0xfffffff0U) == 0xd50341d0U) { // msr daifset/clr
        arm64_jit_emit_helper_continue_regarg(e, arm64_jit_helper_msr_daif_jitabi, guest_pc, 0);
        return ARM64_JIT_EMIT_CONTINUE;
    }
    if ((insn & 0xfff00000U) == 0xd5300000U) { // generic MRS
        uint32_t op0 = (insn >> 19) & 0x3;
        uint32_t op1 = (insn >> 16) & 0x7;
        uint32_t crn = (insn >> 12) & 0xf;
        uint32_t crm = (insn >> 8) & 0xf;
        uint32_t op2 = (insn >> 5) & 0x7;
        int sysreg_id = -1;
        if (op0 == 3 && op1 == 3 && crn == 0 && crm == 0 && op2 == 1)
            sysreg_id = 1; // CTR_EL0
        else if (op0 == 3 && op1 == 3 && crn == 0 && crm == 0 && op2 == 7)
            sysreg_id = 2; // DCZID_EL0
        else if (op0 == 3 && op1 == 3 && crn == 4 && crm == 4 && op2 == 0)
            sysreg_id = 3; // FPCR
        else if (op0 == 3 && op1 == 3 && crn == 4 && crm == 4 && op2 == 1)
            sysreg_id = 4; // FPSR
        else if (op0 == 3 && op1 == 3 && crn == 2 && crm == 4 && op2 == 0)
            sysreg_id = 5; // RNDR
        else if (op0 == 3 && op1 == 3 && crn == 2 && crm == 4 && op2 == 1)
            sysreg_id = 6; // RNDRRS
        else if (op0 == 3 && op1 == 0 && crn == 0 && crm == 4 && op2 == 0)
            sysreg_id = 7; // ID_AA64PFR0_EL1
        else if (op0 == 3 && op1 == 0 && crn == 0 && crm == 4 && op2 == 1)
            sysreg_id = 8; // ID_AA64PFR1_EL1
        else if (op0 == 3 && op1 == 0 && crn == 0 && crm == 6 && op2 == 0)
            sysreg_id = 9; // ID_AA64ISAR0_EL1
        else if (op0 == 3 && op1 == 0 && crn == 0 && crm == 6 && op2 == 1)
            sysreg_id = 10; // ID_AA64ISAR1_EL1
        else if (op0 == 3 && op1 == 0 && crn == 0 && crm == 4 && op2 == 4)
            sysreg_id = 11; // ID_AA64ZFR0_EL1
        else if (op0 == 3 && op1 == 3 && crn == 14 && crm == 0 && op2 == 2)
            sysreg_id = 12; // CNTVCT_EL0
        else if (op0 == 3 && op1 == 3 && crn == 14 && crm == 0 && op2 == 0)
            sysreg_id = 13; // CNTFRQ_EL0

        if (arm64_jit_trace_mode() && guest_pc == 0xefe62638) {
            fprintf(stderr,
                    "[arm64-jit] emit_system pc=0x%llx insn=0x%08x op0=%u op1=%u crn=%u crm=%u op2=%u sysreg_id=%d rd=%u\n",
                    (unsigned long long) guest_pc, insn, op0, op1, crn, crm, op2, sysreg_id, ARM64_RD(insn));
        }
        if (sysreg_id >= 0) {
            uint64_t packed = (uint32_t) guest_pc |
                    ((uint64_t) ARM64_RD(insn) << 32) |
                    ((uint64_t) (unsigned) sysreg_id << 40);
            arm64_jit_emit_helper_continue_packed1(e, arm64_jit_helper_mrs_sysreg_jitabi, packed);
            return ARM64_JIT_EMIT_CONTINUE;
        }
    }

    arm64_jit_emit_helper_return(e, arm64_jit_helper_unsupported_jitabi, guest_pc);
    return ARM64_JIT_EMIT_TERMINATE;
}

enum arm64_jit_emit_result arm64_jit_emit_ld_st(struct arm64_jit_emitter *e, uint32_t insn, addr_t guest_pc) {
    if (((insn >> 24) & 0x3f) == 0x08 &&
            ((insn >> 12) & 0x7) == 0x7 &&
            ((insn >> 10) & 0x3) == 0x3) {
        arm64_jit_emit_helper_return_regarg(e, arm64_jit_helper_ldst_excl_jitabi, guest_pc, insn);
        return ARM64_JIT_EMIT_TERMINATE;
    }

    if ((insn & 0x3a000000u) == 0x28000000u) {
        if (arm64_jit_trace_mode() && (guest_pc == 0xefe62688 || guest_pc == 0xefe6268c)) {
            fprintf(stderr, "[arm64-jit] emit_ld_st pair pc=0x%llx insn=0x%08x V=%u opc=%u mode=%u L=%u\n",
                    (unsigned long long) guest_pc, insn,
                    (insn >> 26) & 1, (insn >> 30) & 0x3, (insn >> 23) & 0x7, (insn >> 22) & 1);
        }
        uint32_t Vp = (insn >> 26) & 1;
        uint32_t opcp = (insn >> 30) & 0x3;
        uint32_t modep = (insn >> 23) & 0x7;
        uint32_t Lp = (insn >> 22) & 1;
        uint32_t rtp = ARM64_RT(insn);
        uint32_t rt2p = (insn >> 10) & 0x1f;
        uint32_t rnp = ARM64_RN(insn);
        if (Vp == 1 && modep == 2 && opcp <= 2) { // vector offset pair load/store
            int base_host = arm64_jit_guest_src_host_reg(e->block, rnp, true);
            if (base_host >= 0) {
                int32_t imm7 = (int32_t) ((insn >> 15) & 0x7f);
                if (imm7 & 0x40)
                    imm7 |= ~0x7f;
                int64_t offset = imm7 * ((int64_t) 4 << opcp);
                if (base_host != ARM64_JIT_HOST_TMP2)
                    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(ARM64_JIT_HOST_TMP2, (unsigned) base_host));
                if (offset != 0) {
                    arm64_jit_emit_load_imm64(e, ARM64_JIT_HOST_TMP1, (uint64_t) offset);
                    arm64_jit_emit32(e, 0x8b010042u); // add x2, x2, x1
                }
                arm64_jit_emit_helper_continue_live_addronly(e, arm64_jit_helper_ldst_pair_vec_live,
                        ((uint64_t) (uint32_t) guest_pc << 32) | insn, ARM64_JIT_HOST_TMP2);
                return ARM64_JIT_EMIT_CONTINUE;
            }
        }
        if (Vp == 0 && Lp == 0 && modep == 2 && (opcp == 0 || opcp == 2)) { // scalar offset pair store
            int base_host = arm64_jit_guest_src_host_reg(e->block, rnp, true);
            int val0_host = arm64_jit_guest_src_host_reg_or_zr(e->block, rtp);
            int val1_host = arm64_jit_guest_src_host_reg_or_zr(e->block, rt2p);
            if (base_host >= 0 && val0_host >= 0 && val1_host >= 0) {
                int32_t imm7 = (int32_t) ((insn >> 15) & 0x7f);
                if (imm7 & 0x40)
                    imm7 |= ~0x7f;
                int64_t scale = (opcp == 2) ? 8 : 4;
                int64_t offset = imm7 * scale;
                if (base_host != ARM64_JIT_HOST_TMP2)
                    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(ARM64_JIT_HOST_TMP2, (unsigned) base_host));
                if (offset != 0) {
                    arm64_jit_emit_load_imm64(e, ARM64_JIT_HOST_TMP1, (uint64_t) offset);
                    arm64_jit_emit32(e, 0x8b010042u); // add x2, x2, x1
                }
                if (val0_host != ARM64_JIT_HOST_TMP3)
                    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(ARM64_JIT_HOST_TMP3, (unsigned) val0_host));
                if (val1_host != 16)
                    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(16, (unsigned) val1_host));
                arm64_jit_emit_helper_continue_live_store_pair(e, arm64_jit_helper_ldst_pair_store_live,
                        ((uint64_t) (uint32_t) guest_pc << 32) | insn,
                        ARM64_JIT_HOST_TMP2, ARM64_JIT_HOST_TMP3, 16);
                return ARM64_JIT_EMIT_CONTINUE;
            }
        }
        if (Vp == 0 && Lp == 1 && modep == 2 && (opcp == 0 || opcp == 2)) { // scalar offset pair load
            int base_host = arm64_jit_guest_src_host_reg(e->block, rnp, true);
            int dst0_host = arm64_jit_host_reg_for_guest(e->block, rtp);
            int dst1_host = arm64_jit_host_reg_for_guest(e->block, rt2p);
            if (base_host >= 0) {
                int32_t imm7 = (int32_t) ((insn >> 15) & 0x7f);
                if (imm7 & 0x40)
                    imm7 |= ~0x7f;
                int64_t scale = (opcp == 2) ? 8 : 4;
                int64_t offset = imm7 * scale;
                if (base_host != ARM64_JIT_HOST_TMP2)
                    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(ARM64_JIT_HOST_TMP2, (unsigned) base_host));
                if (offset != 0) {
                    arm64_jit_emit_load_imm64(e, ARM64_JIT_HOST_TMP1, (uint64_t) offset);
                    arm64_jit_emit32(e, 0x8b010042u); // add x2, x2, x1
                }
                arm64_jit_emit_helper_continue_live_load_pair(e, arm64_jit_helper_ldst_pair_load_live,
                        ((uint64_t) (uint32_t) guest_pc << 32) | insn,
                        ARM64_JIT_HOST_TMP2,
                        (unsigned) dst0_host, rtp,
                        (unsigned) dst1_host, rt2p);
                return ARM64_JIT_EMIT_CONTINUE;
            }
        }
        arm64_jit_emit_helper_continue_regarg(e, arm64_jit_helper_ldst_pair_jitabi, guest_pc, insn);
        return ARM64_JIT_EMIT_CONTINUE;
    }
    if ((insn & 0x3b200000u) == 0x38000000u) {
        uint32_t opc9 = (insn >> 22) & 0x3;
        uint32_t mode9 = (insn >> 10) & 0x3;
        uint32_t rn9 = ARM64_RN(insn);
        uint32_t rt9 = ARM64_RT(insn);
        if (opc9 == 1 && mode9 == 0) { // unscaled load, no writeback
            int base_host = arm64_jit_guest_src_host_reg(e->block, rn9, true);
            int dst_host = arm64_jit_host_reg_for_guest(e->block, rt9);
            if (base_host >= 0) {
                int32_t imm9 = (int32_t) ((insn >> 12) & 0x1ff);
                if (imm9 & 0x100)
                    imm9 |= ~0x1ff;
                if (base_host != ARM64_JIT_HOST_TMP2)
                    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(ARM64_JIT_HOST_TMP2, (unsigned) base_host));
                if (imm9 != 0) {
                    arm64_jit_emit_load_imm64(e, ARM64_JIT_HOST_TMP1, (uint64_t) (int64_t) imm9);
                    arm64_jit_emit32(e, 0x8b010042u); // add x2, x2, x1
                }
                arm64_jit_emit_helper_continue_live_load(e, arm64_jit_helper_ldst_imm9_load_live,
                        guest_pc, ((uint64_t) (uint32_t) guest_pc << 32) | insn,
                        ARM64_JIT_HOST_TMP2, dst_host, rt9);
                return ARM64_JIT_EMIT_CONTINUE;
            }
        }
        if (opc9 == 0 && mode9 == 0) { // unscaled store, no writeback
            int base_host = arm64_jit_guest_src_host_reg(e->block, rn9, true);
            int value_host = arm64_jit_guest_src_host_reg_or_zr(e->block, rt9);
            if (base_host >= 0 && value_host >= 0) {
                int32_t imm9 = (int32_t) ((insn >> 12) & 0x1ff);
                if (imm9 & 0x100)
                    imm9 |= ~0x1ff;
                if (base_host != ARM64_JIT_HOST_TMP2)
                    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(ARM64_JIT_HOST_TMP2, (unsigned) base_host));
                if (imm9 != 0) {
                    arm64_jit_emit_load_imm64(e, ARM64_JIT_HOST_TMP1, (uint64_t) (int64_t) imm9);
                    arm64_jit_emit32(e, 0x8b010042u); // add x2, x2, x1
                }
                if (value_host != ARM64_JIT_HOST_TMP3)
                    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(ARM64_JIT_HOST_TMP3, (unsigned) value_host));
                arm64_jit_emit_helper_continue_live_store(e, arm64_jit_helper_ldst_imm9_live,
                        ((uint64_t) (uint32_t) guest_pc << 32) | insn,
                        ARM64_JIT_HOST_TMP2, ARM64_JIT_HOST_TMP3);
                return ARM64_JIT_EMIT_CONTINUE;
            }
        }
        arm64_jit_emit_helper_continue_regarg(e, arm64_jit_helper_ldst_imm9_jitabi, guest_pc, insn);
        return ARM64_JIT_EMIT_CONTINUE;
    }
    if ((insn & 0x3b200c00u) == 0x38200800u) {
        uint32_t opc_ro = (insn >> 22) & 0x3;
        uint32_t rn_ro = ARM64_RN(insn);
        uint32_t rm_ro = ARM64_RM(insn);
        uint32_t rt_ro = ARM64_RT(insn);
        uint32_t option_ro = (insn >> 13) & 0x7;
        uint32_t S_ro = (insn >> 12) & 1;
        if (opc_ro == 1 && option_ro == 3 && rm_ro != 31) { // load, clean no-writeback regoff
            int base_host = arm64_jit_guest_src_host_reg(e->block, rn_ro, true);
            int off_host = arm64_jit_guest_src_host_reg(e->block, rm_ro, false);
            int dst_host = arm64_jit_host_reg_for_guest(e->block, rt_ro);
            if (base_host >= 0 && off_host >= 0) {
                if (base_host != ARM64_JIT_HOST_TMP2)
                    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(ARM64_JIT_HOST_TMP2, (unsigned) base_host));
                if (off_host != ARM64_JIT_HOST_TMP1)
                    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(ARM64_JIT_HOST_TMP1, (unsigned) off_host));
                if (S_ro) {
                    uint32_t size_ro = (insn >> 30) & 0x3;
                    if (size_ro != 0) {
                        arm64_jit_emit_load_imm64(e, ARM64_JIT_HOST_TMP3, size_ro);
                        arm64_jit_emit32(e, 0x9ac32021u); // lslv x1, x1, x3
                    }
                }
                arm64_jit_emit32(e, 0x8b010042u); // add x2, x2, x1
                arm64_jit_emit_helper_continue_live_load(e, arm64_jit_helper_ldst_regoff_load_live,
                        guest_pc, ((uint64_t) (uint32_t) guest_pc << 32) | insn,
                        ARM64_JIT_HOST_TMP2, dst_host, rt_ro);
                return ARM64_JIT_EMIT_CONTINUE;
            }
        }
        if (opc_ro == 0 && option_ro == 3 && rm_ro != 31) { // store, clean no-writeback regoff
            int base_host = arm64_jit_guest_src_host_reg(e->block, rn_ro, true);
            int off_host = arm64_jit_guest_src_host_reg(e->block, rm_ro, false);
            int value_host = arm64_jit_guest_src_host_reg_or_zr(e->block, rt_ro);
            if (base_host >= 0 && off_host >= 0 && value_host >= 0) {
                if (base_host != ARM64_JIT_HOST_TMP2)
                    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(ARM64_JIT_HOST_TMP2, (unsigned) base_host));
                if (off_host != ARM64_JIT_HOST_TMP1)
                    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(ARM64_JIT_HOST_TMP1, (unsigned) off_host));
                if (S_ro) {
                    uint32_t size_ro = (insn >> 30) & 0x3;
                    if (size_ro != 0) {
                        arm64_jit_emit_load_imm64(e, ARM64_JIT_HOST_TMP3, size_ro);
                        arm64_jit_emit32(e, 0x9ac32021u); // lslv x1, x1, x3
                    }
                }
                arm64_jit_emit32(e, 0x8b010042u); // add x2, x2, x1
                if (value_host != ARM64_JIT_HOST_TMP3)
                    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(ARM64_JIT_HOST_TMP3, (unsigned) value_host));
                arm64_jit_emit_helper_continue_live_store(e, arm64_jit_helper_ldst_regoff_live,
                        ((uint64_t) (uint32_t) guest_pc << 32) | insn,
                        ARM64_JIT_HOST_TMP2, ARM64_JIT_HOST_TMP3);
                return ARM64_JIT_EMIT_CONTINUE;
            }
        }
        arm64_jit_emit_helper_continue_regarg(e, arm64_jit_helper_ldst_regoff_jitabi, guest_pc, insn);
        return ARM64_JIT_EMIT_CONTINUE;
    }

    if (((insn >> 26) & 1) == 1 && (insn & 0x3b200c00u) == 0x3c000800u) {
        arm64_jit_emit_helper_continue_regarg(e, arm64_jit_helper_ldst_imm9_jitabi, guest_pc, insn);
        return ARM64_JIT_EMIT_CONTINUE;
    }

    uint32_t size = (insn >> 30) & 0x3;
    uint32_t V = (insn >> 26) & 1;
    uint32_t opc = (insn >> 22) & 0x3;
    uint32_t rn = ARM64_RN(insn);
    uint32_t rt = ARM64_RT(insn);
    uint32_t imm12 = (insn >> 10) & 0xfff;

    if (V && ((insn & 0x3b000000u) == 0x39000000u ||
              (insn & 0x3b000000u) == 0x38000000u)) {
        if (arm64_jit_trace_mode() && (guest_pc == 0xefe62688 || guest_pc == 0xefe6268c)) {
            fprintf(stderr, "[arm64-jit] emit_ld_st simd_uimm pc=0x%llx insn=0x%08x\n",
                    (unsigned long long) guest_pc, insn);
        }
        arm64_jit_emit_helper_continue_regarg(e, arm64_jit_helper_simd_ldst_imm_unsigned_jitabi, guest_pc, insn);
        return ARM64_JIT_EMIT_CONTINUE;
    }

    // Unsigned-immediate scalar load/store examples for stage 1.
    if ((insn & 0x3b000000u) == 0x39000000u) {
        bool is_load = opc != 0;
        uint32_t load_mode = opc;
        if (is_load) {
            uint64_t packed1 = (rt & 0x1f) |
                    ((uint64_t) (rn & 0x1f) << 5) |
                    ((uint64_t) (size & 0x3) << 10) |
                    ((uint64_t) (imm12 & 0xfff) << 12) |
                    ((uint64_t) (load_mode & 0x3) << 24);
            uint64_t packed1_live = ((uint64_t) (uint32_t) guest_pc << 32) | packed1;
            int base_host = arm64_jit_guest_src_host_reg(e->block, rn, true);
            int dst_host = arm64_jit_host_reg_for_guest(e->block, rt);
            if (base_host >= 0) {
                if (base_host != ARM64_JIT_HOST_TMP3)
                    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(ARM64_JIT_HOST_TMP3, (unsigned) base_host));
                uint64_t offset = (uint64_t) imm12 << size;
                if (offset != 0) {
                    if (offset <= 4095) {
                        arm64_jit_emit32(e, 0x91000000u |
                                (((uint32_t) offset & 0xfff) << 10) |
                                ((ARM64_JIT_HOST_TMP3 & 0x1f) << 5) |
                                (ARM64_JIT_HOST_TMP3 & 0x1f));
                    } else {
                        arm64_jit_emit_load_imm64(e, ARM64_JIT_HOST_TMP2, offset);
                        arm64_jit_emit32(e, 0x8b020063u); // add x3, x3, x2
                    }
                }
                arm64_jit_emit_helper_continue_live_load(e, arm64_jit_helper_ldr_imm_unsigned_live,
                        guest_pc, packed1_live, ARM64_JIT_HOST_TMP3, dst_host, rt);
                return ARM64_JIT_EMIT_CONTINUE;
            }
            arm64_jit_emit_helper_continue_packed2(e, arm64_jit_helper_ldr_imm_unsigned_jitabi,
                    guest_pc, packed1);
            return ARM64_JIT_EMIT_CONTINUE;
        }
        if (opc == 0) {
            uint64_t packed = (uint32_t) guest_pc |
                    ((uint64_t) (rt & 0x1f) << 32) |
                    ((uint64_t) (rn & 0x1f) << 37) |
                    ((uint64_t) (size & 0x3) << 42) |
                    ((uint64_t) (imm12 & 0xfff) << 44);
            int base_host = arm64_jit_guest_src_host_reg(e->block, rn, true);
            int value_host = arm64_jit_guest_src_host_reg_or_zr(e->block, rt);
            if (base_host >= 0 && value_host >= 0) {
                // guest_addr = base + (imm12 << size)
                if (base_host != ARM64_JIT_HOST_TMP2)
                    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(ARM64_JIT_HOST_TMP2, (unsigned) base_host));
                uint64_t offset = (uint64_t) imm12 << size;
                if (offset != 0) {
                    if (offset <= 4095) {
                        arm64_jit_emit32(e, 0x91000000u |
                                (((uint32_t) offset & 0xfff) << 10) |
                                ((ARM64_JIT_HOST_TMP2 & 0x1f) << 5) |
                                (ARM64_JIT_HOST_TMP2 & 0x1f));
                    } else {
                        arm64_jit_emit_load_imm64(e, ARM64_JIT_HOST_TMP3, offset);
                        arm64_jit_emit32(e, 0x8b030042u); // add x2, x2, x3
                    }
                }
                if (value_host != ARM64_JIT_HOST_TMP3)
                    arm64_jit_emit32(e, arm64_jit_enc_mov_reg(ARM64_JIT_HOST_TMP3, (unsigned) value_host));
                arm64_jit_emit_helper_continue_live_store(e, arm64_jit_helper_str_imm_unsigned_live,
                        packed, ARM64_JIT_HOST_TMP2, ARM64_JIT_HOST_TMP3);
                return ARM64_JIT_EMIT_CONTINUE;
            }
            arm64_jit_emit_helper_continue_packed1(e, arm64_jit_helper_str_imm_unsigned_jitabi, packed);
            return ARM64_JIT_EMIT_CONTINUE;
        }
    }

    if (arm64_jit_trace_mode() && (guest_pc == 0xefe62688 || guest_pc == 0xefe6268c)) {
        fprintf(stderr, "[arm64-jit] emit_ld_st unsupported pc=0x%llx insn=0x%08x\n",
                (unsigned long long) guest_pc, insn);
    }
    arm64_jit_emit_helper_return(e, arm64_jit_helper_unsupported_jitabi, guest_pc);
    return ARM64_JIT_EMIT_TERMINATE;
}

enum arm64_jit_emit_result arm64_jit_emit_simd_fp(struct arm64_jit_emitter *e, uint32_t insn, addr_t guest_pc) {
    if (arm64_jit_emit_simd_fp_cached(e, insn))
        return ARM64_JIT_EMIT_CONTINUE;
    if (arm64_jit_trace_mode()) {
        fprintf(stderr, "[arm64-jit] unsupported simd_fp pc=0x%llx insn=0x%08x\n",
                (unsigned long long) guest_pc, insn);
    }
    arm64_jit_emit_helper_return(e, arm64_jit_helper_unsupported_jitabi, guest_pc);
    return ARM64_JIT_EMIT_TERMINATE;
}

static enum arm64_jit_emit_result arm64_jit_emit_one(struct arm64_jit_emitter *e,
        const struct arm64_jit_insn_info *info, uint32_t insn, addr_t guest_pc) {
    switch (info->type) {
        case INSN_DP_IMM:
            return arm64_jit_emit_dp_imm(e, insn, guest_pc);
        case INSN_DP_REG:
            return arm64_jit_emit_dp_reg(e, insn, guest_pc);
        case INSN_BRANCH:
            return arm64_jit_emit_branch(e, insn, guest_pc);
        case INSN_EXCEPTION:
            return arm64_jit_emit_exception(e, insn, guest_pc);
        case INSN_SYSTEM:
            return arm64_jit_emit_system(e, insn, guest_pc);
        case INSN_LD_ST:
            return arm64_jit_emit_ld_st(e, insn, guest_pc);
        case INSN_SIMD_FP:
            return arm64_jit_emit_simd_fp(e, insn, guest_pc);
        default:
            return ARM64_JIT_EMIT_UNSUPPORTED;
    }
}

static bool arm64_jit_branch_has_fallthrough(uint32_t insn) {
    if ((insn & 0x7e000000u) == 0x34000000u)
        return true;
    if ((insn & 0xff000010u) == 0x54000000u)
        return true;
    if ((insn & 0x7e000000u) == 0x36000000u)
        return true;
    return false;
}

static void arm64_jit_emit_internal_fallthrough(struct arm64_jit_emitter *e,
        addr_t branch_pc, addr_t target_pc) {
    if (arm64_jit_block_has_pc(e->block, target_pc)) {
        arm64_jit_emit_local_fixup(e, branch_pc, target_pc, ARM64_JIT_FIXUP_B);
        arm64_jit_emit32(e, arm64_jit_enc_b_imm(0));
        return;
    }
    arm64_jit_emit_helper_return(e, arm64_jit_helper_dispatch_jitabi, target_pc);
}

void arm64_jit_emit_block(struct arm64_jit_state *state, struct arm64_jit_block *block) {
retry_without_local_fixups:
    (void) state;
    size_t cap = block->insn_count * 256 + 1024;
    if (cap < 8192)
        cap = 8192;
    uint8_t *buf = mmap(NULL, cap, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (buf == MAP_FAILED)
        return;

    struct arm64_jit_emitter e = {
        .state = state,
        .block = block,
        .buf = buf,
        .cap = cap,
        .size = 0,
    };
    block->spill_state_fn = NULL;
    block->reload_state_fn = NULL;
    block->pc_map_count = 0;
    block->verify_site_count = 0;
    block->fixup_count = 0;

    arm64_jit_emit_prologue(&e);
    arm64_jit_emit_load_cached_state(&e);
    bool terminated = false;
    for (uint32_t i = 0; i < block->insn_count; i++) {
        addr_t guest_pc = block->insn_pcs[i];
        uint32_t insn = block->insns[i];
        arm64_jit_record_pc_map(&e, guest_pc);
        arm64_jit_emit_verify_entry_brk(&e, guest_pc, insn);
        const struct arm64_jit_insn_info *info = &block->infos[i];
        enum arm64_jit_emit_result res = arm64_jit_emit_one(&e, info, insn, guest_pc);
        if (res == ARM64_JIT_EMIT_UNSUPPORTED) {
            arm64_jit_emit_helper_return(&e, arm64_jit_helper_unsupported_jitabi, guest_pc);
            terminated = true;
            break;
        }
        if (res == ARM64_JIT_EMIT_TERMINATE) {
            terminated = true;
            break;
        }
        if (i + 1 < block->insn_count &&
                (info->type != INSN_BRANCH || arm64_jit_branch_has_fallthrough(insn))) {
            addr_t fallthrough_pc = guest_pc + 4;
            addr_t layout_next_pc = block->insn_pcs[i + 1];
            if (layout_next_pc != fallthrough_pc) {
                arm64_jit_emit_internal_fallthrough(&e, guest_pc, fallthrough_pc);
            }
        }
    }
    if (!terminated) {
        arm64_jit_emit_spill_cached_state(&e);
        arm64_jit_emit_load_imm64(&e, ARM64_JIT_HOST_HELPER0, block->end_pc);
        arm64_jit_emit32(&e, arm64_jit_enc_str64_uimm(ARM64_JIT_HOST_HELPER0, ARM64_JIT_HOST_CTX,
                (40 >> 3))); // rt->resume_pc
        arm64_jit_emit32(&e, arm64_jit_enc_str64_uimm(ARM64_JIT_HOST_HELPER0, ARM64_JIT_HOST_CPU,
                (CPU_OFFSET(pc) >> 3)));
        arm64_jit_emit32(&e, 0x12800000); // mov w0, #-1
        arm64_jit_emit_epilogue(&e);
    }

    block->spill_state_fn = buf + e.size;
    arm64_jit_emit_spill_cached_state(&e);
    arm64_jit_emit_state_snippet_ret(&e);

    block->reload_state_fn = buf + e.size;
    arm64_jit_emit_load_cached_state(&e);
    arm64_jit_emit_state_snippet_ret(&e);

    if (!arm64_jit_patch_local_fixups(block, buf)) {
        munmap(buf, cap);
        goto retry_without_local_fixups;
    }

    if (mprotect(buf, cap, PROT_READ | PROT_EXEC) != 0) {
        if (arm64_jit_trace_mode()) {
            fprintf(stderr, "[arm64-jit] emit abort: mprotect failed start=0x%llx size=%zu errno=%d\n",
                    (unsigned long long) block->start_pc, e.size, errno);
        }
        munmap(buf, cap);
        return;
    }
    if (arm64_jit_trace_mode()) {
        fprintf(stderr, "[arm64-jit] emitted start=0x%llx size=%zu\n",
                (unsigned long long) block->start_pc, e.size);
        for (size_t off = 0; off + 4 <= e.size; off += 4) {
            fprintf(stderr, "[arm64-jit]   +0x%02zx: 0x%08x\n",
                    off, *(uint32_t *) (buf + off));
        }
    }
    block->code_rw = buf;
    block->code_rx = buf;
    block->code_size = (uint32_t) e.size;
}

#endif
