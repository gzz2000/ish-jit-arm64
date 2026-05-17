#if defined(GUEST_ARM64) && defined(__aarch64__)

#include <stdatomic.h>
#include <sys/mman.h>

#include "jit/guest-arm64/jit.h"
#include "emu/tlb.h"
#include "kernel/calls.h"

__thread struct arm64_jit_runtime *g_arm64_jit_runtime;

struct mmu_arm64_jit_slot {
    struct mmu *mmu;
    struct arm64_jit_state *state;
};

static struct mmu_arm64_jit_slot g_arm64_jit_slots[64];
static lock_t g_arm64_jit_slots_lock = LOCK_INITIALIZER;

static uint64_t arm64_jit_read_gpr(struct cpu_state *cpu, uint32_t reg, bool sp_not_zr);

static struct arm64_jit_state *arm64_jit_state_new(struct mmu *mmu) {
    struct arm64_jit_state *state = calloc(1, sizeof(*state));
    if (state == NULL)
        return NULL;
    state->mmu = mmu;
    state->hash_size = ARM64_JIT_HASH_SIZE;
    state->hash = calloc(state->hash_size, sizeof(struct list));
    state->page_hash = calloc(ARM64_JIT_PAGE_HASH_SIZE, sizeof(*state->page_hash));
    if (state->hash == NULL || state->page_hash == NULL) {
        free(state->page_hash);
        free(state->hash);
        free(state);
        return NULL;
    }
    list_init(&state->jetsam);
    lock_init(&state->lock);
    wrlock_init(&state->jetsam_lock);
    return state;
}

struct arm64_jit_state *arm64_jit_state_for_mmu(struct mmu *mmu) {
    lock(&g_arm64_jit_slots_lock);
    for (size_t i = 0; i < sizeof(g_arm64_jit_slots) / sizeof(g_arm64_jit_slots[0]); i++) {
        if (g_arm64_jit_slots[i].mmu == mmu) {
            struct arm64_jit_state *state = g_arm64_jit_slots[i].state;
            unlock(&g_arm64_jit_slots_lock);
            return state;
        }
    }
    for (size_t i = 0; i < sizeof(g_arm64_jit_slots) / sizeof(g_arm64_jit_slots[0]); i++) {
        if (g_arm64_jit_slots[i].mmu == NULL) {
            struct arm64_jit_state *state = arm64_jit_state_new(mmu);
            if (state != NULL) {
                g_arm64_jit_slots[i].mmu = mmu;
                g_arm64_jit_slots[i].state = state;
            }
            unlock(&g_arm64_jit_slots_lock);
            return state;
        }
    }
    unlock(&g_arm64_jit_slots_lock);
    return NULL;
}

int arm64_jit_trace_mode(void) {
    static int trace_mode = -1;
    if (trace_mode == -1) {
        const char *env = getenv("ISH_ARM64_JIT_TRACE");
        trace_mode = (env != NULL && env[0] == '1') ? 1 : 0;
    }
    return trace_mode;
}

int arm64_jit_verify_mode(void) {
    static int verify_mode = -1;
    if (verify_mode == -1) {
        const char *env = getenv("ISH_ARM64_JIT_VERIFY");
        verify_mode = (env != NULL && env[0] == '1') ? 1 : 0;
    }
    return verify_mode;
}

void arm64_jit_set_saved_pc(addr_t pc) {
    extern __thread volatile addr_t jit_saved_pc;
    jit_saved_pc = pc;
}

void arm64_jit_record_fault_pc(void *host_pc) {
    if (g_arm64_jit_runtime == NULL || g_arm64_jit_runtime->block == NULL) {
        arm64_jit_set_saved_pc(0);
        return;
    }
    struct arm64_jit_block *block = g_arm64_jit_runtime->block;
    uintptr_t base = (uintptr_t) block->code_rx;
    uintptr_t pc = (uintptr_t) host_pc;
    addr_t guest_pc = block->start_pc;
    if (pc >= base && pc < base + block->code_size) {
        uint32_t off = (uint32_t) (pc - base);
        for (uint32_t i = 0; i < block->pc_map_count; i++) {
            if (block->pc_map[i].host_offset > off)
                break;
            guest_pc = block->pc_map[i].guest_pc;
        }
    }
    g_arm64_jit_runtime->fault_pc = guest_pc;
    arm64_jit_set_saved_pc(guest_pc);
}

int arm64_jit_helper_unsupported(struct arm64_jit_runtime *rt, addr_t guest_pc) {
    if (arm64_jit_trace_mode() && (guest_pc == 0xefeb4224 || guest_pc == 0xefeaa790)) {
        fprintf(stderr, "[arm64-jit] unsupported pc=0x%llx x0=0x%llx x1=0x%llx x2=0x%llx x30=0x%llx tls=0x%llx\n",
                (unsigned long long) guest_pc,
                (unsigned long long) rt->cpu->regs[0],
                (unsigned long long) rt->cpu->regs[1],
                (unsigned long long) rt->cpu->regs[2],
                (unsigned long long) rt->cpu->regs[30],
                (unsigned long long) rt->cpu->tls_ptr);
    }
    rt->resume_pc = guest_pc;
    rt->cpu->pc = guest_pc;
    rt->exit_interrupt = INT_DEBUG;
    return INT_DEBUG;
}

int arm64_jit_helper_syscall(struct arm64_jit_runtime *rt, addr_t guest_pc) {
    if (arm64_jit_trace_mode()) {
        fprintf(stderr,
                "[arm64-jit] syscall pc=0x%llx nr=%llu x0=0x%llx x1=0x%llx x2=0x%llx x3=0x%llx sp=0x%llx\n",
                (unsigned long long) guest_pc,
                (unsigned long long) rt->cpu->regs[8],
                (unsigned long long) rt->cpu->regs[0],
                (unsigned long long) rt->cpu->regs[1],
                (unsigned long long) rt->cpu->regs[2],
                (unsigned long long) rt->cpu->regs[3],
                (unsigned long long) rt->cpu->sp);
    }
    rt->resume_pc = guest_pc + 4;
    rt->cpu->pc = guest_pc + 4;
    rt->exit_interrupt = INT_SYSCALL;
    return INT_SYSCALL;
}

int arm64_jit_helper_timer(struct arm64_jit_runtime *rt, addr_t guest_pc) {
    rt->resume_pc = guest_pc;
    rt->cpu->pc = guest_pc;
    rt->exit_interrupt = INT_TIMER;
    return INT_TIMER;
}

int arm64_jit_helper_verify_trap(struct arm64_jit_runtime *rt, addr_t guest_pc) {
    if (arm64_jit_trace_mode() &&
            ((guest_pc >= 0xefeb39d4 && guest_pc <= 0xefeb39d8) ||
             guest_pc == 0xefeb3688 || guest_pc == 0xefeb6264 ||
             guest_pc == 0xefeb5038 || guest_pc == 0xefeb4064)) {
        fprintf(stderr, "[arm64-jit] verify_trap pc=0x%llx x0=0x%llx x3=0x%llx x29=0x%llx x30=0x%llx nzcv=0x%x resume=0x%llx\n",
                (unsigned long long) guest_pc,
                (unsigned long long) rt->cpu->regs[0],
                (unsigned long long) rt->cpu->regs[3],
                (unsigned long long) rt->cpu->regs[29],
                (unsigned long long) rt->cpu->regs[30],
                rt->cpu->nzcv,
                (unsigned long long) rt->resume_pc);
    }
    rt->resume_pc = guest_pc;
    rt->cpu->pc = guest_pc;
    rt->exit_interrupt = INT_BREAKPOINT;
    return INT_BREAKPOINT;
}

int arm64_jit_helper_dispatch(struct arm64_jit_runtime *rt, addr_t guest_pc) {
    if (guest_pc < 0x1000 || (rt->block != NULL && rt->block->start_pc == 0xefeb36d4)) {
        fprintf(stderr, "[arm64-jit] dispatch target=0x%llx block_start=0x%llx from resume=0x%llx x30=0x%llx\n",
                (unsigned long long) guest_pc,
                (unsigned long long) (rt->block ? rt->block->start_pc : 0),
                (unsigned long long) rt->resume_pc,
                (unsigned long long) rt->cpu->regs[30]);
    }
    rt->resume_pc = guest_pc;
    rt->cpu->pc = guest_pc;
    rt->exit_interrupt = INT_NONE;
    return INT_NONE;
}

int arm64_jit_helper_branch_link(struct arm64_jit_runtime *rt, uint64_t packed) {
    addr_t guest_pc = (addr_t) (packed & 0xffffffffu);
    addr_t target_pc = (addr_t) (packed >> 32);
    rt->cpu->regs[30] = guest_pc + 4;
    rt->resume_pc = target_pc;
    rt->cpu->pc = target_pc;
    rt->exit_interrupt = INT_NONE;
    return INT_NONE;
}

int arm64_jit_helper_branch_reg(struct arm64_jit_runtime *rt, addr_t guest_pc, uint32_t insn) {
    uint32_t opc = (insn >> 21) & 0xf;
    uint32_t rn = ARM64_RN(insn);
    uint64_t target = arm64_jit_read_gpr(rt->cpu, rn, false);

    switch (opc) {
        case 0: // BR
            if (arm64_jit_trace_mode() && target < 0x1000) {
                fprintf(stderr, "[arm64-jit] br suspicious pc=0x%llx rn=%u target=0x%llx x30=0x%llx\n",
                        (unsigned long long) guest_pc, rn,
                        (unsigned long long) target,
                        (unsigned long long) rt->cpu->regs[30]);
            }
            rt->resume_pc = target;
            rt->cpu->pc = target;
            rt->exit_interrupt = INT_NONE;
            return INT_NONE;
        case 1: // BLR
            if (arm64_jit_trace_mode() && target < 0x1000) {
                fprintf(stderr, "[arm64-jit] blr suspicious pc=0x%llx rn=%u target=0x%llx lr_before=0x%llx\n",
                        (unsigned long long) guest_pc, rn,
                        (unsigned long long) target,
                        (unsigned long long) rt->cpu->regs[30]);
            }
            rt->cpu->regs[30] = guest_pc + 4;
            rt->resume_pc = target;
            rt->cpu->pc = target;
            rt->exit_interrupt = INT_NONE;
            return INT_NONE;
        case 2: // RET
            if (arm64_jit_trace_mode() && target < 0x1000) {
                fprintf(stderr, "[arm64-jit] ret suspicious pc=0x%llx rn=%u target=0x%llx lr=0x%llx\n",
                        (unsigned long long) guest_pc, rn,
                        (unsigned long long) target,
                        (unsigned long long) rt->cpu->regs[30]);
            }
            rt->resume_pc = target;
            rt->cpu->pc = target;
            rt->exit_interrupt = INT_NONE;
            return INT_NONE;
        default:
            return arm64_jit_helper_unsupported(rt, guest_pc);
    }
}

int arm64_jit_helper_mrs_tpidr(struct arm64_jit_runtime *rt, addr_t guest_pc, uint32_t rd) {
    if (rd < 31)
        rt->cpu->regs[rd] = rt->cpu->tls_ptr;
    rt->resume_pc = guest_pc + 4;
    rt->cpu->pc = guest_pc + 4;
    rt->exit_interrupt = INT_NONE;
    return INT_NONE;
}

int arm64_jit_helper_mrs_sysreg(struct arm64_jit_runtime *rt, uint64_t packed) {
    addr_t guest_pc = (addr_t) (packed & 0xffffffffu);
    uint32_t rd = (packed >> 32) & 0x1f;
    uint32_t sysreg_id = (packed >> 40) & 0xff;
    uint64_t value = 0;
    if (arm64_jit_trace_mode() && guest_pc == 0xefe62638) {
        fprintf(stderr, "[arm64-jit] mrs_sysreg helper pc=0x%llx rd=%u sysreg_id=%u\n",
                (unsigned long long) guest_pc, rd, sysreg_id);
    }
    switch (sysreg_id) {
        case 0: // TPIDR_EL0
            value = rt->cpu->tls_ptr;
            break;
        case 1: // CTR_EL0
            value = 0x84448004ull;
            break;
        case 2: // DCZID_EL0
            value = 0x14;
            break;
        case 3: // FPCR
            value = rt->cpu->fpcr;
            break;
        case 4: // FPSR
            value = rt->cpu->fpsr;
            break;
        case 5: // RNDR
        case 6: // RNDRRS
            value = 0;
            rt->cpu->nf = 0;
            rt->cpu->zf = 1;
            rt->cpu->cf = 0;
            rt->cpu->vf = 0;
            arm64_sync_nzcv(rt->cpu);
            break;
        case 7: // ID_AA64PFR0_EL1
            value = 0x11;
            break;
        case 8: // ID_AA64PFR1_EL1
        case 9: // ID_AA64ISAR0_EL1
        case 10: // ID_AA64ISAR1_EL1
        case 11: // ID_AA64ZFR0_EL1
            value = 0;
            break;
#if defined(__aarch64__)
        case 12: { // CNTVCT_EL0
            uint64_t cntvct;
            asm volatile("mrs %0, cntvct_el0" : "=r"(cntvct));
            value = cntvct;
            break;
        }
        case 13: { // CNTFRQ_EL0
            uint64_t cntfrq;
            asm volatile("mrs %0, cntfrq_el0" : "=r"(cntfrq));
            value = cntfrq;
            break;
        }
#else
        case 12:
        case 13:
            value = 0;
            break;
#endif
        default:
            return arm64_jit_helper_unsupported(rt, guest_pc);
    }
    if (rd < 31)
        rt->cpu->regs[rd] = value;
    if (arm64_jit_trace_mode() && guest_pc == 0xefe62638) {
        fprintf(stderr, "[arm64-jit] mrs_sysreg done pc=0x%llx x%u=0x%llx\n",
                (unsigned long long) guest_pc, rd, (unsigned long long) value);
    }
    rt->resume_pc = guest_pc + 4;
    rt->cpu->pc = guest_pc + 4;
    rt->exit_interrupt = INT_NONE;
    return INT_NONE;
}

int arm64_jit_helper_msr_tpidr(struct arm64_jit_runtime *rt, addr_t guest_pc, uint32_t rn) {
    uint64_t value = (rn == 31) ? 0 : rt->cpu->regs[rn];
    if (arm64_jit_trace_mode() && guest_pc == 0xefeaa790) {
        fprintf(stderr, "[arm64-jit] msr_tpidr pc=0x%llx reg=%u value=0x%llx old_tls=0x%llx\n",
                (unsigned long long) guest_pc, rn,
                (unsigned long long) value,
                (unsigned long long) rt->cpu->tls_ptr);
    }
    rt->cpu->tls_ptr = value;
    rt->resume_pc = guest_pc + 4;
    rt->cpu->pc = guest_pc + 4;
    rt->exit_interrupt = INT_NONE;
    return INT_NONE;
}

int arm64_jit_helper_msr_daif(struct arm64_jit_runtime *rt, addr_t guest_pc) {
    rt->resume_pc = guest_pc + 4;
    rt->cpu->pc = guest_pc + 4;
    rt->exit_interrupt = INT_NONE;
    return INT_NONE;
}

static uint64_t arm64_jit_read_gpr(struct cpu_state *cpu, uint32_t reg, bool sp_not_zr) {
    if (reg == 31)
        return sp_not_zr ? cpu->sp : 0;
    return cpu->regs[reg];
}

static void arm64_jit_write_gpr(struct cpu_state *cpu, uint32_t reg, uint64_t value, bool sf) {
    if (reg == 31)
        return;
    cpu->regs[reg] = sf ? value : (uint32_t) value;
}

static void arm64_jit_nzcv_from_add(struct cpu_state *cpu, uint64_t lhs, uint64_t rhs, uint64_t res, bool sf) {
    uint64_t mask = sf ? UINT64_MAX : 0xffffffffu;
    uint64_t sign = sf ? (1ULL << 63) : (1ULL << 31);
    lhs &= mask;
    rhs &= mask;
    res &= mask;
    cpu->nf = (res & sign) != 0;
    cpu->zf = (res == 0);
    cpu->cf = res < lhs;
    bool lhs_sign = (lhs & sign) != 0;
    bool rhs_sign = (rhs & sign) != 0;
    bool res_sign = (res & sign) != 0;
    cpu->vf = (lhs_sign == rhs_sign) && (lhs_sign != res_sign);
    arm64_sync_nzcv(cpu);
}

static void arm64_jit_nzcv_from_sub(struct cpu_state *cpu, uint64_t lhs, uint64_t rhs, uint64_t res, bool sf) {
    uint64_t mask = sf ? UINT64_MAX : 0xffffffffu;
    uint64_t sign = sf ? (1ULL << 63) : (1ULL << 31);
    lhs &= mask;
    rhs &= mask;
    res &= mask;
    cpu->nf = (res & sign) != 0;
    cpu->zf = (res == 0);
    cpu->cf = lhs >= rhs;
    bool lhs_sign = (lhs & sign) != 0;
    bool rhs_sign = (rhs & sign) != 0;
    bool res_sign = (res & sign) != 0;
    cpu->vf = (lhs_sign != rhs_sign) && (lhs_sign != res_sign);
    arm64_sync_nzcv(cpu);
}

static uint64_t arm64_jit_mask_for_sf(bool sf) {
    return sf ? UINT64_MAX : 0xffffffffu;
}

static int arm64_jit_simd_elem_size_from_imm5(uint32_t imm5) {
    if (imm5 & 0x1)
        return 0; // B
    if (imm5 & 0x2)
        return 1; // H
    if (imm5 & 0x4)
        return 2; // S
    if (imm5 & 0x8)
        return 3; // D
    return -1;
}

static uint64_t arm64_jit_simd_extract_elem(const union arm64_vreg *v, int elem_size, int index) {
    switch (elem_size) {
        case 0: return v->b[index];
        case 1: return v->h[index];
        case 2: return v->s[index];
        case 3: return v->d[index];
        default: return 0;
    }
}

static void arm64_jit_simd_insert_elem(union arm64_vreg *v, int elem_size, int index, uint64_t value) {
    switch (elem_size) {
        case 0: v->b[index] = (uint8_t) value; break;
        case 1: v->h[index] = (uint16_t) value; break;
        case 2: v->s[index] = (uint32_t) value; break;
        case 3: v->d[index] = value; break;
        default: break;
    }
}

static void arm64_jit_simd_clear(union arm64_vreg *v) {
    v->q = 0;
}

static void arm64_jit_simd_dup_value(union arm64_vreg *v, uint64_t value, int elem_size, bool q) {
    arm64_jit_simd_clear(v);
    int count = q ? (16 >> elem_size) : (8 >> elem_size);
    for (int i = 0; i < count; i++)
        arm64_jit_simd_insert_elem(v, elem_size, i, value);
}

static void arm64_jit_write_nz_logical(struct cpu_state *cpu, uint64_t value, bool sf) {
    uint64_t mask = arm64_jit_mask_for_sf(sf);
    uint64_t sign = sf ? (1ULL << 63) : (1ULL << 31);
    value &= mask;
    cpu->nf = (value & sign) != 0;
    cpu->zf = (value == 0);
    cpu->cf = false;
    cpu->vf = false;
    arm64_sync_nzcv(cpu);
}

static uint64_t arm64_jit_add_with_carry(bool sf, uint64_t x, uint64_t y, uint64_t carry_in,
        bool *n, bool *z, bool *c, bool *v) {
    uint64_t mask = arm64_jit_mask_for_sf(sf);
    uint64_t sign = sf ? (1ULL << 63) : (1ULL << 31);
    __uint128_t ux = (__uint128_t) (x & mask);
    __uint128_t uy = (__uint128_t) (y & mask);
    __uint128_t usum = ux + uy + carry_in;
    uint64_t res = (uint64_t) usum & mask;
    if (n) *n = (res & sign) != 0;
    if (z) *z = (res == 0);
    if (c) *c = (usum >> (sf ? 64 : 32)) != 0;
    if (v) {
        bool sx = ((x & mask) & sign) != 0;
        bool sy = ((y & mask) & sign) != 0;
        bool sr = (res & sign) != 0;
        *v = (sx == sy) && (sx != sr);
    }
    return res;
}

static uint64_t arm64_jit_bit_reverse64(uint64_t x) {
    x = ((x >> 1) & 0x5555555555555555ull) | ((x & 0x5555555555555555ull) << 1);
    x = ((x >> 2) & 0x3333333333333333ull) | ((x & 0x3333333333333333ull) << 2);
    x = ((x >> 4) & 0x0f0f0f0f0f0f0f0full) | ((x & 0x0f0f0f0f0f0f0f0full) << 4);
    x = ((x >> 8) & 0x00ff00ff00ff00ffull) | ((x & 0x00ff00ff00ff00ffull) << 8);
    x = ((x >> 16) & 0x0000ffff0000ffffull) | ((x & 0x0000ffff0000ffffull) << 16);
    x = (x >> 32) | (x << 32);
    return x;
}

static uint64_t arm64_jit_rev16(uint64_t value, bool sf) {
    if (!sf) {
        uint32_t v = (uint32_t) value;
        uint32_t r = ((v & 0x00ff00ffu) << 8) | ((v & 0xff00ff00u) >> 8);
        return r;
    }
    uint64_t r = ((value & 0x00ff00ff00ff00ffull) << 8) |
                 ((value & 0xff00ff00ff00ff00ull) >> 8);
    return r;
}

static uint64_t arm64_jit_rev32_or_rev(bool sf, uint64_t value) {
    if (!sf) {
        uint32_t v = (uint32_t) value;
        return ((uint32_t) __builtin_bswap32(v));
    }
    uint64_t lo = (uint32_t) value;
    uint64_t hi = (uint32_t) (value >> 32);
    lo = __builtin_bswap32((uint32_t) lo);
    hi = __builtin_bswap32((uint32_t) hi);
    return lo | (hi << 32);
}

static uint64_t arm64_jit_rev64(uint64_t value) {
    return __builtin_bswap64(value);
}

static uint64_t arm64_jit_clz_value(uint64_t value, bool sf) {
    if (!sf) {
        uint32_t v = (uint32_t) value;
        return v ? (uint32_t) __builtin_clz(v) : 32;
    }
    return value ? (uint64_t) __builtin_clzll(value) : 64;
}

static uint64_t arm64_jit_cls_value(uint64_t value, bool sf) {
    if (!sf) {
        uint32_t v = (uint32_t) value;
        uint32_t sign = v >> 31;
        uint32_t x = sign ? ~v : v;
        return x ? ((uint32_t) __builtin_clz(x) - 1) : 31;
    }
    uint64_t sign = value >> 63;
    uint64_t x = sign ? ~value : value;
    return x ? ((uint64_t) __builtin_clzll(x) - 1) : 63;
}

static uint64_t arm64_jit_shift_reg_amount(uint64_t rhs, bool sf) {
    return sf ? (rhs & 63u) : (rhs & 31u);
}

static uint64_t arm64_jit_extend_reg_value(uint64_t value, uint32_t option) {
    bool sign = (option & 0x4) != 0;
    switch (option & 0x3) {
        case 0:
            return sign ? (uint64_t) (int8_t) value : (uint64_t) (uint8_t) value;
        case 1:
            return sign ? (uint64_t) (int16_t) value : (uint64_t) (uint16_t) value;
        case 2:
            return sign ? (uint64_t) (int32_t) value : (uint64_t) (uint32_t) value;
        case 3:
        default:
            return sign ? (uint64_t) (int64_t) value : value;
    }
}

static void arm64_jit_write_nzcv(struct cpu_state *cpu, bool n, bool z, bool c, bool v) {
    cpu->nf = n;
    cpu->zf = z;
    cpu->cf = c;
    cpu->vf = v;
    arm64_sync_nzcv(cpu);
}

static uint64_t arm64_jit_rotate_right(uint64_t value, unsigned amount, unsigned width) {
    amount %= width;
    uint64_t mask = width == 64 ? UINT64_MAX : ((1ULL << width) - 1);
    value &= mask;
    if (amount == 0)
        return value;
    return ((value >> amount) | (value << (width - amount))) & mask;
}

static uint64_t arm64_jit_decode_bitmask(unsigned n, unsigned imms, unsigned immr, bool sf, bool *ok) {
    unsigned width = sf ? 64 : 32;
    unsigned levels = sf ? 0x3f : 0x1f;
    if (!sf && n) {
        *ok = false;
        return 0;
    }
    unsigned s = imms & levels;
    unsigned r = immr & levels;
    unsigned len = -1;
    unsigned temp = ((n << 6) | (~imms & 0x3f)) & 0x7f;
    for (int i = 6; i >= 0; i--) {
        if (temp & (1u << i)) {
            len = i;
            break;
        }
    }
    if (len == (unsigned)-1) {
        *ok = false;
        return 0;
    }
    unsigned esize = 1u << len;
    unsigned levels_mask = esize - 1;
    s &= levels_mask;
    r &= levels_mask;
    if (s == levels_mask) {
        *ok = false;
        return 0;
    }
    uint64_t welem = (1ULL << (s + 1)) - 1;
    welem = arm64_jit_rotate_right(welem, r, esize);
    uint64_t mask = 0;
    for (unsigned pos = 0; pos < width; pos += esize)
        mask |= welem << pos;
    *ok = true;
    return mask;
}

int arm64_jit_helper_cbz_cbnz(struct arm64_jit_runtime *rt, addr_t guest_pc, uint32_t insn) {
    bool sf = (insn >> 31) & 1;
    bool is_nonzero = (insn >> 24) & 1;
    uint32_t rt_reg = ARM64_RT(insn);
    uint64_t value = arm64_jit_read_gpr(rt->cpu, rt_reg, false);
    if (!sf)
        value = (uint32_t) value;
    bool take = is_nonzero ? (value != 0) : (value == 0);
    addr_t target = guest_pc + (take ? arm64_branch_imm19(insn) : 4);
    if (guest_pc == 0xefeb36d8 || target < 0x1000) {
        fprintf(stderr,
                "[arm64-jit] cbz/cbnz pc=0x%llx insn=0x%08x sf=%d nz=%d rt=%u value=0x%llx take=%d target=0x%llx\n",
                (unsigned long long) guest_pc,
                insn,
                sf,
                is_nonzero,
                rt_reg,
                (unsigned long long) value,
                take,
                (unsigned long long) target);
    }
    rt->resume_pc = target;
    rt->cpu->pc = target;
    rt->exit_interrupt = INT_NONE;
    return INT_NONE;
}

int arm64_jit_helper_b_cond(struct arm64_jit_runtime *rt, addr_t guest_pc, uint32_t insn) {
    enum arm64_cond cond = (enum arm64_cond) (insn & 0xf);
    arm64_set_nzcv(rt->cpu, rt->cpu->nzcv);
    bool take = arm64_cond_check(rt->cpu, cond);
    addr_t target = guest_pc + (take ? arm64_branch_imm19(insn) : 4);
    rt->resume_pc = target;
    rt->cpu->pc = target;
    rt->exit_interrupt = INT_NONE;
    return INT_NONE;
}

int arm64_jit_helper_tbz_tbnz(struct arm64_jit_runtime *rt, addr_t guest_pc, uint32_t insn) {
    bool is_tbnz = (insn >> 24) & 1;
    uint32_t b5 = (insn >> 31) & 1;
    uint32_t b40 = (insn >> 19) & 0x1f;
    uint32_t bit_pos = (b5 << 5) | b40;
    uint32_t rt_reg = ARM64_RT(insn);
    uint64_t value = arm64_jit_read_gpr(rt->cpu, rt_reg, false);
    bool bit_set = ((value >> bit_pos) & 1) != 0;
    bool take = is_tbnz ? bit_set : !bit_set;
    addr_t target = guest_pc + (take ? arm64_branch_imm14(insn) : 4);
    rt->resume_pc = target;
    rt->cpu->pc = target;
    rt->exit_interrupt = INT_NONE;
    return INT_NONE;
}

int arm64_jit_helper_exec_dp_imm(struct arm64_jit_runtime *rt, addr_t guest_pc, uint32_t insn) {
    struct cpu_state *cpu = rt->cpu;
    if (arm64_jit_trace_mode() && guest_pc == 0xefeb3684) {
        fprintf(stderr,
                "[arm64-jit] dp_imm helper pc=0x%llx insn=0x%08x x0=0x%llx sp=0x%llx nzcv=0x%x\n",
                (unsigned long long) guest_pc,
                insn,
                (unsigned long long) cpu->regs[0],
                (unsigned long long) cpu->sp,
                cpu->nzcv);
    }
    bool sf = ARM64_SF(insn);

    // Logical immediate family.
    if ((insn & 0x1f800000u) == 0x12000000u) {
        uint32_t opc = (insn >> 29) & 0x3;
        uint32_t n = (insn >> 22) & 1;
        uint32_t immr = (insn >> 16) & 0x3f;
        uint32_t imms = (insn >> 10) & 0x3f;
        uint32_t rn = ARM64_RN(insn);
        uint32_t rd = ARM64_RD(insn);
        bool ok = false;
        uint64_t lhs = arm64_jit_read_gpr(cpu, rn, false);
        uint64_t mask = arm64_jit_decode_bitmask(n, imms, immr, sf, &ok);
        if (!ok)
            return arm64_jit_helper_unsupported(rt, guest_pc);
        uint64_t res;
        switch (opc) {
            case 0: res = lhs & mask; break;
            case 1: res = lhs | mask; break;
            case 2: res = lhs ^ mask; break;
            case 3:
                res = lhs & mask;
                cpu->nf = sf ? ((res >> 63) & 1) : ((res >> 31) & 1);
                cpu->zf = res == 0;
                cpu->cf = 0;
                cpu->vf = 0;
                arm64_sync_nzcv(cpu);
                break;
            default:
                return arm64_jit_helper_unsupported(rt, guest_pc);
        }
        if (rd == 31 && opc != 3)
            cpu->sp = sf ? res : (uint32_t) res;
        else
            arm64_jit_write_gpr(cpu, rd, res, sf);
        rt->resume_pc = guest_pc + 4;
        cpu->pc = guest_pc + 4;
        rt->exit_interrupt = INT_NONE;
        return INT_NONE;
    }

    // Move-wide immediate family.
    if ((insn & 0x1f800000u) == 0x12800000u) {
        uint32_t opc = (insn >> 29) & 0x3;
        uint32_t hw = (insn >> 21) & 0x3;
        uint64_t imm = ((uint64_t) ((insn >> 5) & 0xffff)) << (hw * 16);
        uint32_t rd = ARM64_RD(insn);
        uint64_t old = arm64_jit_read_gpr(cpu, rd, false);
        uint64_t value;
        switch (opc) {
            case 0: value = ~imm; break; // MOVN
            case 2: value = imm; break;  // MOVZ
            case 3: {
                uint64_t mask = 0xffffULL << (hw * 16);
                value = (old & ~mask) | imm;
                break;
            }
            default:
                return arm64_jit_helper_unsupported(rt, guest_pc);
        }
        arm64_jit_write_gpr(cpu, rd, value, sf);
        rt->resume_pc = guest_pc + 4;
        cpu->pc = guest_pc + 4;
        rt->exit_interrupt = INT_NONE;
        return INT_NONE;
    }

    // ADR / ADRP
    if ((insn & 0x1f000000u) == 0x10000000u || (insn & 0x9f000000u) == 0x90000000u) {
        int64_t imm = arm64_adr_imm(insn);
        uint64_t base = guest_pc;
        if (insn & 0x80000000u) { // ADRP
            base &= ~0xfffULL;
            imm <<= 12;
        }
        arm64_jit_write_gpr(cpu, ARM64_RD(insn), base + imm, true);
        rt->resume_pc = guest_pc + 4;
        cpu->pc = guest_pc + 4;
        rt->exit_interrupt = INT_NONE;
        return INT_NONE;
    }

    // ADD/SUB immediate
    if ((insn & 0x1f000000u) == 0x11000000u || (insn & 0x7f000000u) == 0x31000000u) {
        bool is_sub = (insn >> 30) & 1;
        bool setflags = (insn >> 29) & 1;
        uint32_t shift = (insn >> 22) & 0x3;
        uint64_t imm = (insn >> 10) & 0xfff;
        if (shift == 1)
            imm <<= 12;
        else if (shift != 0)
            return arm64_jit_helper_unsupported(rt, guest_pc);
        uint32_t rn = ARM64_RN(insn);
        uint32_t rd = ARM64_RD(insn);
        uint64_t lhs = arm64_jit_read_gpr(cpu, rn, true);
        uint64_t res = is_sub ? (lhs - imm) : (lhs + imm);
        if (setflags) {
            if (is_sub)
                arm64_jit_nzcv_from_sub(cpu, lhs, imm, res, sf);
            else
                arm64_jit_nzcv_from_add(cpu, lhs, imm, res, sf);
        }
        arm64_jit_write_gpr(cpu, rd, res, sf);
        rt->resume_pc = guest_pc + 4;
        cpu->pc = guest_pc + 4;
        rt->exit_interrupt = INT_NONE;
        return INT_NONE;
    }

    // Bitfield immediate: enough for common UBFM aliases.
    if ((insn & 0x1f800000u) == 0x13000000u) {
        uint32_t opc = (insn >> 29) & 0x3;
        uint32_t n = (insn >> 22) & 1;
        uint32_t immr = (insn >> 16) & 0x3f;
        uint32_t imms = (insn >> 10) & 0x3f;
        uint32_t rn = ARM64_RN(insn);
        uint32_t rd = ARM64_RD(insn);
        uint32_t size = sf ? 64 : 32;
        bool ok = false;
        uint64_t src = arm64_jit_read_gpr(cpu, rn, false);

        if (imms == size - 1) {
            uint64_t res;
            if (opc == 2) { // UBFM alias: LSR
                res = src >> immr;
                arm64_jit_write_gpr(cpu, rd, res, sf);
                rt->resume_pc = guest_pc + 4;
                cpu->pc = guest_pc + 4;
                rt->exit_interrupt = INT_NONE;
                return INT_NONE;
            }
            if (opc == 0) { // SBFM alias: ASR
                res = sf ? (uint64_t) ((int64_t) src >> immr)
                         : (uint64_t) ((int32_t) src >> immr);
                arm64_jit_write_gpr(cpu, rd, res, sf);
                rt->resume_pc = guest_pc + 4;
                cpu->pc = guest_pc + 4;
                rt->exit_interrupt = INT_NONE;
                return INT_NONE;
            }
        }
        if (opc == 2 && immr == imms + 1 && immr <= size - 1) { // UBFM alias: LSL
            uint32_t shift = size - immr;
            uint64_t res = src << shift;
            arm64_jit_write_gpr(cpu, rd, res, sf);
            rt->resume_pc = guest_pc + 4;
            cpu->pc = guest_pc + 4;
            rt->exit_interrupt = INT_NONE;
            return INT_NONE;
        }

        uint64_t mask = arm64_jit_decode_bitmask(n, imms, immr, sf, &ok);
        if (!ok)
            return arm64_jit_helper_unsupported(rt, guest_pc);
        uint64_t width = size;
        uint64_t rotated = arm64_jit_rotate_right(src, immr, width);
        uint64_t res = rotated & mask;
        if (opc == 0) { // SBFM
            unsigned topbit = sf ? 63 : 31;
            if (res & (1ULL << (imms & topbit)))
                res |= ~mask;
        } else if (opc == 1) {
            // BFM merges destination; not implemented yet.
            return arm64_jit_helper_unsupported(rt, guest_pc);
        }
        arm64_jit_write_gpr(cpu, rd, res, sf);
        rt->resume_pc = guest_pc + 4;
        cpu->pc = guest_pc + 4;
        rt->exit_interrupt = INT_NONE;
        return INT_NONE;
    }

    return arm64_jit_helper_unsupported(rt, guest_pc);
}

int arm64_jit_helper_exec_dp_reg(struct arm64_jit_runtime *rt, addr_t guest_pc, uint32_t insn) {
    struct cpu_state *cpu = rt->cpu;
    bool sf = ARM64_SF(insn);
    uint32_t top5 = (insn >> 24) & 0x1f;
    uint64_t mask = arm64_jit_mask_for_sf(sf);

    if ((insn & 0x1f000000u) == 0x1b000000u) { // Data-processing (3 source)
        uint32_t op54 = (insn >> 29) & 0x3;
        uint32_t op31 = (insn >> 21) & 0x7;
        uint32_t rm = ARM64_RM(insn);
        uint32_t o0 = (insn >> 15) & 1;
        uint32_t ra = (insn >> 10) & 0x1f;
        uint32_t rn = ARM64_RN(insn);
        uint32_t rd = ARM64_RD(insn);
        uint64_t lhs = arm64_jit_read_gpr(cpu, rn, false);
        uint64_t rhs = arm64_jit_read_gpr(cpu, rm, false);
        uint64_t acc = arm64_jit_read_gpr(cpu, ra, false);
        uint64_t res;

        if (arm64_jit_trace_mode() && guest_pc == 0xefeb4224) {
            fprintf(stderr,
                    "[arm64-jit] dp_reg_3src pc=0x%llx insn=0x%08x sf=%u op54=%u op31=%u o0=%u rd=%u rn=%u rm=%u ra=%u lhs=0x%llx rhs=0x%llx acc=0x%llx\n",
                    (unsigned long long) guest_pc, insn, sf, op54, op31, o0,
                    rd, rn, rm, ra,
                    (unsigned long long) lhs,
                    (unsigned long long) rhs,
                    (unsigned long long) acc);
        }

        if (op54 != 0)
            return arm64_jit_helper_unsupported(rt, guest_pc);

        if (sf == 1 && op31 == 0) { // MADD/MSUB 64-bit
            res = o0 ? (acc - (lhs * rhs)) : (acc + (lhs * rhs));
        } else if (sf == 0 && op31 == 0) { // MADD/MSUB 32-bit
            uint32_t lhs32 = (uint32_t) lhs;
            uint32_t rhs32 = (uint32_t) rhs;
            uint32_t acc32 = (uint32_t) acc;
            uint32_t res32 = o0 ? (acc32 - (lhs32 * rhs32)) : (acc32 + (lhs32 * rhs32));
            res = res32;
        } else if (sf == 1 && op31 == 1) { // SMADDL/SMSUBL
            int64_t lhs32 = (int32_t) lhs;
            int64_t rhs32 = (int32_t) rhs;
            int64_t acc64 = (int64_t) acc;
            int64_t res64 = o0 ? (acc64 - (lhs32 * rhs32)) : (acc64 + (lhs32 * rhs32));
            res = (uint64_t) res64;
        } else if (sf == 1 && op31 == 2 && o0 == 0) { // SMULH
            __int128 prod = (__int128) (int64_t) lhs * (__int128) (int64_t) rhs;
            res = (uint64_t) (prod >> 64);
        } else if (sf == 1 && op31 == 5) { // UMADDL/UMSUBL
            uint64_t lhs32 = (uint32_t) lhs;
            uint64_t rhs32 = (uint32_t) rhs;
            res = o0 ? (acc - (lhs32 * rhs32)) : (acc + (lhs32 * rhs32));
        } else if (sf == 1 && op31 == 6 && o0 == 0) { // UMULH
            unsigned __int128 prod = (unsigned __int128) lhs * (unsigned __int128) rhs;
            res = (uint64_t) (prod >> 64);
        } else {
            return arm64_jit_helper_unsupported(rt, guest_pc);
        }

        arm64_jit_write_gpr(cpu, rd, res, sf);
        rt->resume_pc = guest_pc + 4;
        cpu->pc = guest_pc + 4;
        rt->exit_interrupt = INT_NONE;
        return INT_NONE;
    }

    if (top5 == 0x0a) { // logical shifted-register subset
        uint32_t shift = (insn >> 22) & 0x3;
        uint32_t N = (insn >> 21) & 1;
        uint32_t rm = ARM64_RM(insn);
        uint32_t imm6 = (insn >> 10) & 0x3f;
        uint32_t rn = ARM64_RN(insn);
        uint32_t rd = ARM64_RD(insn);
        if (!sf && imm6 >= 32)
            return arm64_jit_helper_unsupported(rt, guest_pc);
        uint64_t lhs = arm64_jit_read_gpr(cpu, rn, false);
        uint64_t rhs = arm64_jit_read_gpr(cpu, rm, false);
        if (shift == 0)
            rhs <<= imm6;
        else if (shift == 1)
            rhs >>= imm6;
        else if (shift == 2)
            rhs = sf ? ((int64_t) rhs >> imm6) : ((int32_t) rhs >> imm6);
        else if (shift == 3)
            rhs = arm64_jit_rotate_right(rhs, imm6, sf ? 64 : 32);
        else
            return arm64_jit_helper_unsupported(rt, guest_pc);
        rhs &= mask;
        if (N)
            rhs = (~rhs) & mask;
        uint32_t opc = (insn >> 29) & 0x3;
        uint64_t res;
        switch (opc) {
            case 0: res = lhs & rhs; break;
            case 1: res = lhs | rhs; break;
            case 2: res = lhs ^ rhs; break;
            case 3: res = lhs & rhs; arm64_jit_write_nz_logical(cpu, res, sf); break;
            default: return arm64_jit_helper_unsupported(rt, guest_pc);
        }
        if (arm64_jit_trace_mode() && guest_pc >= 0xefea8878 && guest_pc <= 0xefea8888) {
            fprintf(stderr,
                    "[arm64-jit] dp_reg_logical pc=0x%llx insn=0x%08x opc=%u N=%u shift=%u imm6=%u rn=%u rm=%u rd=%u lhs=0x%llx rhs=0x%llx res=0x%llx x0=0x%llx x2=0x%llx x4=0x%llx x5=0x%llx x6=0x%llx\n",
                    (unsigned long long) guest_pc, insn, opc, N, shift, imm6, rn, rm, rd,
                    (unsigned long long) lhs, (unsigned long long) rhs,
                    (unsigned long long) res,
                    (unsigned long long) cpu->regs[0],
                    (unsigned long long) cpu->regs[2],
                    (unsigned long long) cpu->regs[4],
                    (unsigned long long) cpu->regs[5],
                    (unsigned long long) cpu->regs[6]);
        }
        arm64_jit_write_gpr(cpu, rd, res, sf);
        if (arm64_jit_trace_mode() && guest_pc >= 0xefea8878 && guest_pc <= 0xefea8888) {
            fprintf(stderr,
                    "[arm64-jit] dp_reg_logical_after pc=0x%llx rd=%u x0=0x%llx x2=0x%llx x4=0x%llx x5=0x%llx x6=0x%llx\n",
                    (unsigned long long) guest_pc, rd,
                    (unsigned long long) cpu->regs[0],
                    (unsigned long long) cpu->regs[2],
                    (unsigned long long) cpu->regs[4],
                    (unsigned long long) cpu->regs[5],
                    (unsigned long long) cpu->regs[6]);
        }
        rt->resume_pc = guest_pc + 4;
        cpu->pc = guest_pc + 4;
        rt->exit_interrupt = INT_NONE;
        return INT_NONE;
    }

    if ((insn & 0x1f200000u) == 0x0b200000u) { // add/sub extended-register
        bool is_sub = (insn >> 30) & 1;
        bool setflags = (insn >> 29) & 1;
        uint32_t rm = ARM64_RM(insn);
        uint32_t option = (insn >> 13) & 0x7;
        uint32_t imm3 = (insn >> 10) & 0x7;
        uint32_t rn = ARM64_RN(insn);
        uint32_t rd = ARM64_RD(insn);
        uint64_t lhs = arm64_jit_read_gpr(cpu, rn, !setflags);
        uint64_t rhs = arm64_jit_extend_reg_value(arm64_jit_read_gpr(cpu, rm, false), option);
        rhs <<= imm3;
        if (!sf)
            rhs &= 0xffffffffu;
        uint64_t res = is_sub ? (lhs - rhs) : (lhs + rhs);
        if (setflags) {
            if (is_sub)
                arm64_jit_nzcv_from_sub(cpu, lhs, rhs, res, sf);
            else
                arm64_jit_nzcv_from_add(cpu, lhs, rhs, res, sf);
        }
        if (rd == 31 && !setflags)
            cpu->sp = sf ? res : (uint32_t) res;
        else
            arm64_jit_write_gpr(cpu, rd, res, sf);
        rt->resume_pc = guest_pc + 4;
        cpu->pc = guest_pc + 4;
        rt->exit_interrupt = INT_NONE;
        return INT_NONE;
    }

    if (top5 == 0x0b) { // add/sub shifted-register subset
        bool is_sub = (insn >> 30) & 1;
        bool setflags = (insn >> 29) & 1;
        uint32_t shift = (insn >> 22) & 0x3;
        uint32_t rm = ARM64_RM(insn);
        uint32_t imm6 = (insn >> 10) & 0x3f;
        uint32_t rn = ARM64_RN(insn);
        uint32_t rd = ARM64_RD(insn);
        uint64_t lhs = arm64_jit_read_gpr(cpu, rn, false);
        uint64_t rhs = arm64_jit_read_gpr(cpu, rm, false);
        if (shift == 0)
            rhs <<= imm6;
        else if (shift == 1)
            rhs >>= imm6;
        else if (shift == 2)
            rhs = sf ? ((int64_t) rhs >> imm6) : ((int32_t) rhs >> imm6);
        else
            return arm64_jit_helper_unsupported(rt, guest_pc);
        uint64_t res = is_sub ? (lhs - rhs) : (lhs + rhs);
        if (arm64_jit_trace_mode() && (guest_pc == 0xefeb6384 || guest_pc == 0xefeb39e4)) {
            fprintf(stderr,
                    "[arm64-jit] dp_reg_shift pc=0x%llx rn=%u rd=%u rm=%u lhs=0x%llx rhs=0x%llx res=0x%llx setflags=%d sf=%d sp=0x%llx x2=0x%llx x6=0x%llx x9=0x%llx x10=0x%llx\n",
                    (unsigned long long) guest_pc, rn, rd, rm,
                    (unsigned long long) lhs, (unsigned long long) rhs,
                    (unsigned long long) res, setflags, sf,
                    (unsigned long long) cpu->sp, (unsigned long long) cpu->regs[2],
                    (unsigned long long) cpu->regs[6],
                    (unsigned long long) cpu->regs[9],
                    (unsigned long long) cpu->regs[10]);
        }
        if (setflags) {
            if (is_sub)
                arm64_jit_nzcv_from_sub(cpu, lhs, rhs, res, sf);
            else
                arm64_jit_nzcv_from_add(cpu, lhs, rhs, res, sf);
        }
        if (rd == 31 && !setflags)
            cpu->sp = sf ? res : (uint32_t) res;
        else
            arm64_jit_write_gpr(cpu, rd, res, sf);
        rt->resume_pc = guest_pc + 4;
        cpu->pc = guest_pc + 4;
        rt->exit_interrupt = INT_NONE;
        return INT_NONE;
    }

    if ((insn & 0x1fe00000u) == 0x1a800000u) { // CSEL/CSINC/CSINV/CSNEG
        uint32_t op = (insn >> 30) & 1;
        uint32_t S = (insn >> 29) & 1;
        uint32_t rm = ARM64_RM(insn);
        uint32_t cond = (insn >> 12) & 0xf;
        uint32_t op2 = (insn >> 10) & 0x3;
        uint32_t rn = ARM64_RN(insn);
        uint32_t rd = ARM64_RD(insn);
        if (S != 0)
            return arm64_jit_helper_unsupported(rt, guest_pc);
        uint64_t src_true = arm64_jit_read_gpr(cpu, rn, false);
        uint64_t src_false = arm64_jit_read_gpr(cpu, rm, false);
        uint64_t res = 0;
        bool take = arm64_cond_check(cpu, (enum arm64_cond) cond);
        switch ((op << 1) | op2) {
            case 0: // CSEL
                res = take ? src_true : src_false;
                break;
            case 1: // CSINC
                res = take ? src_true : (src_false + 1);
                break;
            case 2: // CSINV
                res = take ? src_true : ~src_false;
                break;
            case 3: // CSNEG
                res = take ? src_true : (~src_false + 1);
                break;
            default:
                return arm64_jit_helper_unsupported(rt, guest_pc);
        }
        arm64_jit_write_gpr(cpu, rd, res, sf);
        rt->resume_pc = guest_pc + 4;
        cpu->pc = guest_pc + 4;
        rt->exit_interrupt = INT_NONE;
        return INT_NONE;
    }

    if ((insn & 0x1fe0fc00u) == 0x1a000000u) { // ADC/ADCS/SBC/SBCS
        uint32_t op = (insn >> 30) & 1;
        uint32_t S = (insn >> 29) & 1;
        uint32_t rm = ARM64_RM(insn);
        uint32_t rn = ARM64_RN(insn);
        uint32_t rd = ARM64_RD(insn);
        uint64_t lhs = arm64_jit_read_gpr(cpu, rn, false);
        uint64_t rhs = arm64_jit_read_gpr(cpu, rm, false);
        uint64_t rhs_eff = op ? ~rhs : rhs;
        uint64_t carry_in = cpu->cf ? 1 : 0;
        bool n, z, c, v;
        uint64_t res = arm64_jit_add_with_carry(sf, lhs, rhs_eff, carry_in, &n, &z, &c, &v);
        if (S)
            arm64_jit_write_nzcv(cpu, n, z, c, v);
        arm64_jit_write_gpr(cpu, rd, res, sf);
        rt->resume_pc = guest_pc + 4;
        cpu->pc = guest_pc + 4;
        rt->exit_interrupt = INT_NONE;
        return INT_NONE;
    }

    if ((insn & 0x3fe00410u) == 0x3a400000u) { // CCMP/CCMN reg+imm
        uint32_t op = (insn >> 30) & 1; // 0=CCMN 1=CCMP
        uint32_t is_imm = (insn >> 11) & 1;
        uint32_t rm_or_imm5 = (insn >> 16) & 0x1f;
        uint32_t cond = (insn >> 12) & 0xf;
        uint32_t rn = ARM64_RN(insn);
        uint32_t nzcv = insn & 0xf;
        if (arm64_cond_check(cpu, (enum arm64_cond) cond)) {
            uint64_t lhs = arm64_jit_read_gpr(cpu, rn, false);
            uint64_t rhs = is_imm ? rm_or_imm5 : arm64_jit_read_gpr(cpu, rm_or_imm5, false);
            uint64_t res = op ? (lhs - rhs) : (lhs + rhs);
            if (op)
                arm64_jit_nzcv_from_sub(cpu, lhs, rhs, res, sf);
            else
                arm64_jit_nzcv_from_add(cpu, lhs, rhs, res, sf);
        } else {
            arm64_jit_write_nzcv(cpu, (nzcv >> 3) & 1, (nzcv >> 2) & 1, (nzcv >> 1) & 1, nzcv & 1);
        }
        rt->resume_pc = guest_pc + 4;
        cpu->pc = guest_pc + 4;
        rt->exit_interrupt = INT_NONE;
        return INT_NONE;
    }

    if ((insn & 0x5fe00000u) == 0x1ac00000u) { // Data-processing (2 source)
        uint32_t opcode = (insn >> 10) & 0x3f;
        uint32_t rm = ARM64_RM(insn);
        uint32_t rn = ARM64_RN(insn);
        uint32_t rd = ARM64_RD(insn);
        uint64_t lhs = arm64_jit_read_gpr(cpu, rn, false) & mask;
        uint64_t rhs = arm64_jit_read_gpr(cpu, rm, false) & mask;
        uint64_t res;
        switch (opcode) {
            case 0x02: // UDIV
                res = (rhs == 0) ? 0 : (lhs / rhs);
                break;
            case 0x03: { // SDIV
                if (rhs == 0) {
                    res = 0;
                } else if (!sf) {
                    int32_t a = (int32_t) lhs;
                    int32_t b = (int32_t) rhs;
                    res = (uint32_t) (a / b);
                } else {
                    int64_t a = (int64_t) lhs;
                    int64_t b = (int64_t) rhs;
                    res = (uint64_t) (a / b);
                }
                break;
            }
            case 0x08: // LSLV
                res = (lhs << arm64_jit_shift_reg_amount(rhs, sf)) & mask;
                break;
            case 0x09: // LSRV
                res = lhs >> arm64_jit_shift_reg_amount(rhs, sf);
                break;
            case 0x0a: // ASRV
                if (!sf)
                    res = (uint32_t) ((int32_t) lhs >> arm64_jit_shift_reg_amount(rhs, sf));
                else
                    res = (uint64_t) ((int64_t) lhs >> arm64_jit_shift_reg_amount(rhs, sf));
                break;
            case 0x0b: // RORV
                res = arm64_jit_rotate_right(lhs, arm64_jit_shift_reg_amount(rhs, sf), sf ? 64 : 32);
                break;
            default:
                return arm64_jit_helper_unsupported(rt, guest_pc);
        }
        arm64_jit_write_gpr(cpu, rd, res, sf);
        rt->resume_pc = guest_pc + 4;
        cpu->pc = guest_pc + 4;
        rt->exit_interrupt = INT_NONE;
        return INT_NONE;
    }

    if (((insn >> 30) & 1) == 1 && ((insn >> 28) & 1) == 1 && ((insn >> 21) & 0xf) == 0x6) {
        // Data-processing (1 source)
        uint32_t S = (insn >> 29) & 1;
        uint32_t opcode2 = (insn >> 16) & 0x1f;
        uint32_t opcode = (insn >> 10) & 0x3f;
        uint32_t rn = ARM64_RN(insn);
        uint32_t rd = ARM64_RD(insn);
        uint64_t src = arm64_jit_read_gpr(cpu, rn, false) & mask;
        uint64_t res;
        if (S != 0 || opcode2 != 0)
            return arm64_jit_helper_unsupported(rt, guest_pc);
        switch (opcode) {
            case 0x00: // RBIT
                res = arm64_jit_bit_reverse64(src);
                if (!sf)
                    res >>= 32;
                break;
            case 0x01: // REV16
                res = arm64_jit_rev16(src, sf);
                break;
            case 0x02: // REV / REV32
                res = arm64_jit_rev32_or_rev(sf, src);
                break;
            case 0x03: // REV (64-bit)
                if (!sf)
                    return arm64_jit_helper_unsupported(rt, guest_pc);
                res = arm64_jit_rev64(src);
                break;
            case 0x04: // CLZ
                res = arm64_jit_clz_value(src, sf);
                break;
            case 0x05: // CLS
                res = arm64_jit_cls_value(src, sf);
                break;
            default:
                return arm64_jit_helper_unsupported(rt, guest_pc);
        }
        arm64_jit_write_gpr(cpu, rd, res, sf);
        rt->resume_pc = guest_pc + 4;
        cpu->pc = guest_pc + 4;
        rt->exit_interrupt = INT_NONE;
        return INT_NONE;
    }

    return arm64_jit_helper_unsupported(rt, guest_pc);
}

int arm64_jit_c_ldr_imm_unsigned(struct arm64_jit_runtime *rt, uint64_t packed0, uint64_t packed1) {
    addr_t guest_pc = (addr_t) packed0;
    uint32_t rt_reg = packed1 & 0x1f;
    uint32_t rn_reg = (packed1 >> 5) & 0x1f;
    uint32_t size_shift = (packed1 >> 10) & 0x3;
    uint32_t imm12 = (packed1 >> 12) & 0xfff;
    uint32_t load_mode = (packed1 >> 24) & 0x3;
    addr_t addr = arm64_jit_read_gpr(rt->cpu, rn_reg, true) + ((addr_t) imm12 << size_shift);
    int rc = -1;
    if (load_mode == 2 && size_shift == 2) { // LDRSW
        int64_t value = 0;
        rc = c_load32_sx(rt->tlb, addr, &value);
        if (rc == 0)
            arm64_jit_write_gpr(rt->cpu, rt_reg, (uint64_t) value, true);
    } else if (load_mode == 3 && size_shift == 1) { // LDRSH X
        uint16_t value = 0;
        rc = c_load16(rt->tlb, addr, &value);
        if (rc == 0)
            arm64_jit_write_gpr(rt->cpu, rt_reg, (int16_t) value, true);
    } else if (load_mode == 3 && size_shift == 0) { // LDRSB X
        uint8_t value = 0;
        rc = c_load8(rt->tlb, addr, &value);
        if (rc == 0)
            arm64_jit_write_gpr(rt->cpu, rt_reg, (int8_t) value, true);
    } else if (load_mode == 1 && size_shift == 1) { // LDRH / LDRSH W
        uint16_t value = 0;
        rc = c_load16(rt->tlb, addr, &value);
        if (rc == 0)
            arm64_jit_write_gpr(rt->cpu, rt_reg, value, false);
    } else if (load_mode == 1 && size_shift == 0) { // LDRB / LDRSB W
        uint8_t value = 0;
        rc = c_load8(rt->tlb, addr, &value);
        if (rc == 0)
            arm64_jit_write_gpr(rt->cpu, rt_reg, value, false);
    } else if (size_shift == 3) {
        uint64_t value = 0;
        rc = c_load64(rt->tlb, addr, &value);
        if (rc == 0)
            arm64_jit_write_gpr(rt->cpu, rt_reg, value, true);
    } else if (size_shift == 2) {
        uint32_t value = 0;
        rc = c_load32(rt->tlb, addr, &value);
        if (rc == 0)
            arm64_jit_write_gpr(rt->cpu, rt_reg, value, false);
    } else if (size_shift == 1) {
        uint16_t value = 0;
        rc = c_load16(rt->tlb, addr, &value);
        if (rc == 0)
            arm64_jit_write_gpr(rt->cpu, rt_reg, value, false);
    } else {
        uint8_t value = 0;
        rc = c_load8(rt->tlb, addr, &value);
        if (rc == 0)
            arm64_jit_write_gpr(rt->cpu, rt_reg, value, false);
    }
    if (arm64_jit_trace_mode() && (guest_pc == 0xefeb39d4 || guest_pc == 0xefeb39d8 || guest_pc == 0xefeb39e0)) {
        fprintf(stderr, "[arm64-jit] ldr_imm pc=0x%llx rt=%u rn=%u size=%u imm12=%u mode=%u addr=0x%llx rc=%d x3=0x%llx x6=0x%llx x9=0x%llx x10=0x%llx sp=0x%llx\n",
                (unsigned long long) guest_pc, rt_reg, rn_reg, size_shift, imm12, load_mode,
                (unsigned long long) addr, rc,
                (unsigned long long) rt->cpu->regs[3],
                (unsigned long long) rt->cpu->regs[6],
                (unsigned long long) rt->cpu->regs[9],
                (unsigned long long) rt->cpu->regs[10],
                (unsigned long long) rt->cpu->sp);
    }
    if (rc != 0) {
        rt->cpu->segfault_addr = rt->tlb->segfault_addr;
        rt->cpu->segfault_was_write = false;
        rt->cpu->pc = guest_pc;
        rt->resume_pc = guest_pc;
        rt->exit_interrupt = INT_GPF;
        return INT_GPF;
    }
    rt->resume_pc = guest_pc + 4;
    rt->cpu->pc = guest_pc + 4;
    rt->exit_interrupt = INT_NONE;
    return INT_NONE;
}

int arm64_jit_c_str_imm_unsigned(struct arm64_jit_runtime *rt, uint64_t packed) {
    addr_t guest_pc = (addr_t) (packed & 0xffffffffu);
    uint32_t rt_reg = (packed >> 32) & 0x1f;
    uint32_t rn_reg = (packed >> 37) & 0x1f;
    uint32_t size_shift = (packed >> 42) & 0x3;
    uint32_t imm12 = (packed >> 44) & 0xfff;
    addr_t addr = arm64_jit_read_gpr(rt->cpu, rn_reg, true) + ((addr_t) imm12 << size_shift);
    uint64_t value = arm64_jit_read_gpr(rt->cpu, rt_reg, false);
    if (arm64_jit_trace_mode() && guest_pc == 0xefe73550) {
        fprintf(stderr,
                "[arm64-jit] str_uimm pc=0x%llx rt=%u rn=%u size=%u imm12=%u addr=0x%llx value=0x%llx tlb_changes=%u mmu_changes=%u\n",
                (unsigned long long) guest_pc, rt_reg, rn_reg, size_shift, imm12,
                (unsigned long long) addr, (unsigned long long) value,
                rt->tlb->mem_changes, __atomic_load_n(&rt->tlb->mmu->changes, __ATOMIC_ACQUIRE));
    }
    int rc = -1;
    if (size_shift == 3)
        rc = c_store64(rt->tlb, addr, value);
    else if (size_shift == 2)
        rc = c_store32(rt->tlb, addr, (uint32_t) value);
    else if (size_shift == 1)
        rc = c_store16(rt->tlb, addr, (uint16_t) value);
    else
        rc = c_store8(rt->tlb, addr, (uint8_t) value);
    if (rc != 0) {
        if (arm64_jit_trace_mode() && guest_pc == 0xefe73550) {
            fprintf(stderr,
                    "[arm64-jit] str_uimm fault pc=0x%llx segfault_addr=0x%llx was_write=%d\n",
                    (unsigned long long) guest_pc,
                    (unsigned long long) rt->tlb->segfault_addr, 1);
        }
        rt->cpu->segfault_addr = rt->tlb->segfault_addr;
        rt->cpu->segfault_was_write = true;
        rt->cpu->pc = guest_pc;
        rt->resume_pc = guest_pc;
        rt->exit_interrupt = INT_GPF;
        return INT_GPF;
    }
    rt->resume_pc = guest_pc + 4;
    rt->cpu->pc = guest_pc + 4;
    rt->exit_interrupt = INT_NONE;
    return INT_NONE;
}

int arm64_jit_c_simd_ldst_imm_unsigned(struct arm64_jit_runtime *rt, addr_t guest_pc,
        uint32_t insn) {
    struct cpu_state *cpu = rt->cpu;
    uint32_t size = (insn >> 30) & 0x3;
    uint32_t opc = (insn >> 22) & 0x3;
    uint32_t imm12 = (insn >> 10) & 0xfff;
    uint32_t rn = ARM64_RN(insn);
    uint32_t rt_reg = ARM64_RT(insn);

    uint32_t scale = (size == 0 && (opc & 2)) ? 4 : size;
    addr_t addr = arm64_jit_read_gpr(cpu, rn, true) + ((addr_t) imm12 << scale);
    bool is_load = (opc & 1) == 1;
    int rc = -1;

    if (arm64_jit_trace_mode() && guest_pc == 0xefe62628) {
        fprintf(stderr,
                "[arm64-jit] simd_ldst_uimm pc=0x%llx insn=0x%08x size=%u opc=%u imm12=%u rn=%u rt=%u addr=0x%llx is_load=%d x0=0x%llx v0=0x%016llx%016llx\n",
                (unsigned long long) guest_pc, insn, size, opc, imm12, rn, rt_reg,
                (unsigned long long) addr, is_load,
                (unsigned long long) cpu->regs[0],
                (unsigned long long) cpu->fp[0].d[1],
                (unsigned long long) cpu->fp[0].d[0]);
    }

    if (is_load) {
        if (size == 0 && (opc & 2)) { // LDR Qt
            uint64_t lo = 0, hi = 0;
            rc = c_load64(rt->tlb, addr, &lo);
            if (rc == 0)
                rc = c_load64(rt->tlb, addr + 8, &hi);
            if (rc == 0) {
                cpu->fp[rt_reg].d[0] = lo;
                cpu->fp[rt_reg].d[1] = hi;
            }
        } else {
            switch (size) {
                case 0: {
                    uint8_t value = 0;
                    rc = c_load8(rt->tlb, addr, &value);
                    if (rc == 0) {
                        cpu->fp[rt_reg].b[0] = value;
                        memset(&cpu->fp[rt_reg].b[1], 0, sizeof(cpu->fp[rt_reg].b) - 1);
                    }
                    break;
                }
                case 1: {
                    uint16_t value = 0;
                    rc = c_load16(rt->tlb, addr, &value);
                    if (rc == 0) {
                        cpu->fp[rt_reg].h[0] = value;
                        memset(&cpu->fp[rt_reg].b[2], 0, sizeof(cpu->fp[rt_reg].b) - 2);
                    }
                    break;
                }
                case 2: {
                    uint32_t value = 0;
                    rc = c_load32(rt->tlb, addr, &value);
                    if (rc == 0) {
                        cpu->fp[rt_reg].s[0] = value;
                        memset(&cpu->fp[rt_reg].b[4], 0, sizeof(cpu->fp[rt_reg].b) - 4);
                    }
                    break;
                }
                case 3: {
                    uint64_t value = 0;
                    rc = c_load64(rt->tlb, addr, &value);
                    if (rc == 0) {
                        cpu->fp[rt_reg].d[0] = value;
                        cpu->fp[rt_reg].d[1] = 0;
                    }
                    break;
                }
                default:
                    return arm64_jit_helper_unsupported(rt, guest_pc);
            }
        }
    } else {
        if (size == 0 && (opc & 2)) { // STR Qt
            rc = c_store64(rt->tlb, addr, cpu->fp[rt_reg].d[0]);
            if (rc == 0)
                rc = c_store64(rt->tlb, addr + 8, cpu->fp[rt_reg].d[1]);
        } else {
            switch (size) {
                case 0:
                    rc = c_store8(rt->tlb, addr, cpu->fp[rt_reg].b[0]);
                    break;
                case 1:
                    rc = c_store16(rt->tlb, addr, cpu->fp[rt_reg].h[0]);
                    break;
                case 2:
                    rc = c_store32(rt->tlb, addr, cpu->fp[rt_reg].s[0]);
                    break;
                case 3:
                    rc = c_store64(rt->tlb, addr, cpu->fp[rt_reg].d[0]);
                    break;
                default:
                    return arm64_jit_helper_unsupported(rt, guest_pc);
            }
        }
    }

    if (rc != 0) {
        cpu->segfault_addr = rt->tlb->segfault_addr;
        cpu->segfault_was_write = !is_load;
        cpu->pc = guest_pc;
        rt->resume_pc = guest_pc;
        rt->exit_interrupt = INT_GPF;
        return INT_GPF;
    }

    rt->resume_pc = guest_pc + 4;
    cpu->pc = guest_pc + 4;
    rt->exit_interrupt = INT_NONE;
    if (arm64_jit_trace_mode() && guest_pc == 0xefe62628) {
        fprintf(stderr,
                "[arm64-jit] simd_ldst_uimm done pc=0x%llx rc=%d x0=0x%llx pc=0x%llx\n",
                (unsigned long long) guest_pc, rc,
                (unsigned long long) cpu->regs[0],
                (unsigned long long) cpu->pc);
    }
    return INT_NONE;
}

int arm64_jit_c_ldst_imm9(struct arm64_jit_runtime *rt, addr_t guest_pc, uint32_t insn) {
    uint32_t size = (insn >> 30) & 0x3;
    uint32_t V = (insn >> 26) & 1;
    uint32_t opc = (insn >> 22) & 0x3;
    int32_t imm9 = (int32_t) ((insn >> 12) & 0x1ff);
    if (imm9 & 0x100)
        imm9 |= ~0x1ff;
    uint32_t mode = (insn >> 10) & 0x3;
    uint32_t rn = ARM64_RN(insn);
    uint32_t rt_reg = ARM64_RT(insn);

    if (mode == 2)
        return arm64_jit_helper_unsupported(rt, guest_pc);

    if (V) {
        bool is_load = (opc & 1) == 1;
        bool is_unscaled = (mode == 0);
        bool is_post = (mode == 1);
        bool is_pre = (mode == 3);
        if (!is_unscaled && !is_post && !is_pre)
            return arm64_jit_helper_unsupported(rt, guest_pc);

        uint64_t base = arm64_jit_read_gpr(rt->cpu, rn, true);
        addr_t addr = is_post ? base : (base + imm9);
        int rc = -1;

        if (is_load) {
            if (size == 0 && (opc & 2)) { // LDUR Qt
                uint64_t lo = 0, hi = 0;
                rc = c_load64(rt->tlb, addr, &lo);
                if (rc == 0)
                    rc = c_load64(rt->tlb, addr + 8, &hi);
                if (rc == 0) {
                    rt->cpu->fp[rt_reg].d[0] = lo;
                    rt->cpu->fp[rt_reg].d[1] = hi;
                }
            } else {
                switch (size) {
                    case 0: {
                        uint8_t value = 0;
                        rc = c_load8(rt->tlb, addr, &value);
                        if (rc == 0) {
                            rt->cpu->fp[rt_reg].b[0] = value;
                            memset(&rt->cpu->fp[rt_reg].b[1], 0, sizeof(rt->cpu->fp[rt_reg].b) - 1);
                        }
                        break;
                    }
                    case 1: {
                        uint16_t value = 0;
                        rc = c_load16(rt->tlb, addr, &value);
                        if (rc == 0) {
                            rt->cpu->fp[rt_reg].h[0] = value;
                            memset(&rt->cpu->fp[rt_reg].b[2], 0, sizeof(rt->cpu->fp[rt_reg].b) - 2);
                        }
                        break;
                    }
                    case 2: {
                        uint32_t value = 0;
                        rc = c_load32(rt->tlb, addr, &value);
                        if (rc == 0) {
                            rt->cpu->fp[rt_reg].s[0] = value;
                            memset(&rt->cpu->fp[rt_reg].b[4], 0, sizeof(rt->cpu->fp[rt_reg].b) - 4);
                        }
                        break;
                    }
                    case 3: {
                        uint64_t value = 0;
                        rc = c_load64(rt->tlb, addr, &value);
                        if (rc == 0) {
                            rt->cpu->fp[rt_reg].d[0] = value;
                            rt->cpu->fp[rt_reg].d[1] = 0;
                        }
                        break;
                    }
                    default:
                        return arm64_jit_helper_unsupported(rt, guest_pc);
                }
            }
        } else {
            if (size == 0 && (opc & 2)) { // STUR Qt
                rc = c_store64(rt->tlb, addr, rt->cpu->fp[rt_reg].d[0]);
                if (rc == 0)
                    rc = c_store64(rt->tlb, addr + 8, rt->cpu->fp[rt_reg].d[1]);
            } else {
                switch (size) {
                    case 0:
                        rc = c_store8(rt->tlb, addr, rt->cpu->fp[rt_reg].b[0]);
                        break;
                    case 1:
                        rc = c_store16(rt->tlb, addr, rt->cpu->fp[rt_reg].h[0]);
                        break;
                    case 2:
                        rc = c_store32(rt->tlb, addr, rt->cpu->fp[rt_reg].s[0]);
                        break;
                    case 3:
                        rc = c_store64(rt->tlb, addr, rt->cpu->fp[rt_reg].d[0]);
                        break;
                    default:
                        return arm64_jit_helper_unsupported(rt, guest_pc);
                }
            }
        }

        if (rc != 0) {
            rt->cpu->segfault_addr = rt->tlb->segfault_addr;
            rt->cpu->segfault_was_write = !is_load;
            rt->cpu->pc = guest_pc;
            rt->resume_pc = guest_pc;
            rt->exit_interrupt = INT_GPF;
            return INT_GPF;
        }

        if (!is_unscaled) {
            uint64_t writeback = is_post ? (base + imm9) : addr;
            if (rn == 31)
                rt->cpu->sp = writeback;
            else
                rt->cpu->regs[rn] = writeback;
        }

        rt->resume_pc = guest_pc + 4;
        rt->cpu->pc = guest_pc + 4;
        rt->exit_interrupt = INT_NONE;
        return INT_NONE;
    }

    bool is_load = (opc == 1) || (opc == 2) || (opc == 3);
    bool sign_extend = (opc >= 2);
    bool extend_to_64 = (opc == 2);
    bool is_unscaled = (mode == 0);
    bool is_post = (mode == 1);
    bool is_pre = (mode == 3);
    if (!is_unscaled && !is_post && !is_pre)
        return arm64_jit_helper_unsupported(rt, guest_pc);

    uint64_t base = arm64_jit_read_gpr(rt->cpu, rn, true);
    addr_t addr = is_post ? base : (base + imm9);
    int rc = -1;

    if (is_load) {
        if (sign_extend && size < 2)
            return arm64_jit_helper_unsupported(rt, guest_pc);
        if (size == 3) {
            uint64_t value = 0;
            rc = c_load64(rt->tlb, addr, &value);
            if (rc == 0)
                arm64_jit_write_gpr(rt->cpu, rt_reg, value, true);
        } else if (size == 2) {
            if (sign_extend && extend_to_64) {
                int64_t value = 0;
                rc = c_load32_sx(rt->tlb, addr, &value);
                if (rc == 0)
                    arm64_jit_write_gpr(rt->cpu, rt_reg, (uint64_t) value, true);
            } else if (!sign_extend) {
                uint32_t value = 0;
                rc = c_load32(rt->tlb, addr, &value);
                if (rc == 0)
                    arm64_jit_write_gpr(rt->cpu, rt_reg, value, false);
            } else {
                return arm64_jit_helper_unsupported(rt, guest_pc);
            }
        } else if (size == 1) {
            uint16_t value = 0;
            rc = c_load16(rt->tlb, addr, &value);
            if (rc == 0)
                arm64_jit_write_gpr(rt->cpu, rt_reg, value, false);
        } else {
            uint8_t value = 0;
            rc = c_load8(rt->tlb, addr, &value);
            if (rc == 0)
                arm64_jit_write_gpr(rt->cpu, rt_reg, value, false);
        }
    } else {
        uint64_t value = arm64_jit_read_gpr(rt->cpu, rt_reg, false);
        if (size == 3)
            rc = c_store64(rt->tlb, addr, value);
        else if (size == 2)
            rc = c_store32(rt->tlb, addr, (uint32_t) value);
        else if (size == 1)
            rc = c_store16(rt->tlb, addr, (uint16_t) value);
        else
            rc = c_store8(rt->tlb, addr, (uint8_t) value);
    }

    if (rc != 0) {
        rt->cpu->segfault_addr = rt->tlb->segfault_addr;
        rt->cpu->segfault_was_write = !is_load;
        rt->cpu->pc = guest_pc;
        rt->resume_pc = guest_pc;
        rt->exit_interrupt = INT_GPF;
        return INT_GPF;
    }

    if (!is_unscaled) {
        uint64_t writeback = is_post ? (base + imm9) : addr;
        if (rn == 31)
            rt->cpu->sp = writeback;
        else
            rt->cpu->regs[rn] = writeback;
    }

    rt->resume_pc = guest_pc + 4;
    rt->cpu->pc = guest_pc + 4;
    rt->exit_interrupt = INT_NONE;
    return INT_NONE;
}

int arm64_jit_c_ldst_regoff(struct arm64_jit_runtime *rt, addr_t guest_pc, uint32_t insn) {
    uint32_t size = (insn >> 30) & 0x3;
    uint32_t V = (insn >> 26) & 1;
    uint32_t opc = (insn >> 22) & 0x3;
    uint32_t rm = ARM64_RM(insn);
    uint32_t option = (insn >> 13) & 0x7;
    uint32_t S = (insn >> 12) & 1;
    uint32_t rn = ARM64_RN(insn);
    uint32_t rt_reg = ARM64_RT(insn);

    if (V)
        return arm64_jit_helper_unsupported(rt, guest_pc);

    bool is_load = (opc == 1) || (opc == 2) || (opc == 3);
    bool sign_extend = (opc >= 2);
    bool extend_to_64 = (opc == 2);
    if (is_load && sign_extend && size < 2)
        return arm64_jit_helper_unsupported(rt, guest_pc);

    uint64_t base = arm64_jit_read_gpr(rt->cpu, rn, true);
    uint64_t offset_raw = arm64_jit_read_gpr(rt->cpu, rm, false);
    uint64_t offset = 0;
    switch (option) {
        case 2: { // UXTW
            uint64_t v = (uint32_t) offset_raw;
            offset = S ? (v << size) : v;
            break;
        }
        case 3: { // LSL
            uint64_t v = offset_raw;
            offset = S ? (v << size) : v;
            break;
        }
        case 6: { // SXTW
            int64_t v = (int32_t) offset_raw;
            offset = (uint64_t) (S ? (v << size) : v);
            break;
        }
        case 7: { // SXTX
            int64_t v = (int64_t) offset_raw;
            offset = (uint64_t) (S ? (v << size) : v);
            break;
        }
        default:
            return arm64_jit_helper_unsupported(rt, guest_pc);
    }
    addr_t addr = base + offset;
    int rc = -1;

    if (is_load) {
        if (size == 3) {
            uint64_t value = 0;
            rc = c_load64(rt->tlb, addr, &value);
            if (rc == 0)
                arm64_jit_write_gpr(rt->cpu, rt_reg, value, true);
        } else if (size == 2) {
            if (sign_extend && extend_to_64) {
                int64_t value = 0;
                rc = c_load32_sx(rt->tlb, addr, &value);
                if (rc == 0)
                    arm64_jit_write_gpr(rt->cpu, rt_reg, (uint64_t) value, true);
            } else if (!sign_extend) {
                uint32_t value = 0;
                rc = c_load32(rt->tlb, addr, &value);
                if (rc == 0)
                    arm64_jit_write_gpr(rt->cpu, rt_reg, value, false);
            } else {
                return arm64_jit_helper_unsupported(rt, guest_pc);
            }
        } else if (size == 1) {
            uint16_t value = 0;
            rc = c_load16(rt->tlb, addr, &value);
            if (rc == 0)
                arm64_jit_write_gpr(rt->cpu, rt_reg, value, false);
        } else {
            uint8_t value = 0;
            rc = c_load8(rt->tlb, addr, &value);
            if (rc == 0)
                arm64_jit_write_gpr(rt->cpu, rt_reg, value, false);
        }
    } else {
        uint64_t value = arm64_jit_read_gpr(rt->cpu, rt_reg, false);
        if (size == 3)
            rc = c_store64(rt->tlb, addr, value);
        else if (size == 2)
            rc = c_store32(rt->tlb, addr, (uint32_t) value);
        else if (size == 1)
            rc = c_store16(rt->tlb, addr, (uint16_t) value);
        else
            rc = c_store8(rt->tlb, addr, (uint8_t) value);
    }

    if (rc != 0) {
        rt->cpu->segfault_addr = rt->tlb->segfault_addr;
        rt->cpu->segfault_was_write = !is_load;
        rt->cpu->pc = guest_pc;
        rt->resume_pc = guest_pc;
        rt->exit_interrupt = INT_GPF;
        return INT_GPF;
    }

    rt->resume_pc = guest_pc + 4;
    rt->cpu->pc = guest_pc + 4;
    rt->exit_interrupt = INT_NONE;
    return INT_NONE;
}

int arm64_jit_c_ldst_pair(struct arm64_jit_runtime *rt, addr_t guest_pc, uint32_t insn) {
    uint32_t opc = (insn >> 30) & 0x3;
    uint32_t V = (insn >> 26) & 1;
    uint32_t mode = (insn >> 23) & 0x7;
    uint32_t L = (insn >> 22) & 1;
    int32_t imm7 = (int32_t) ((insn >> 15) & 0x7f);
    if (imm7 & 0x40)
        imm7 |= ~0x7f;
    uint32_t rt2_reg = (insn >> 10) & 0x1f;
    uint32_t rn = ARM64_RN(insn);
    uint32_t rt_reg = ARM64_RT(insn);

    if (arm64_jit_trace_mode() &&
            (guest_pc == 0xefe62688 || guest_pc == 0xefe6268c || guest_pc == 0xefeb4060)) {
        fprintf(stderr,
                "[arm64-jit] ldst_pair pc=0x%llx insn=0x%08x opc=%u V=%u mode=%u L=%u imm7=%d rt=%u rt2=%u rn=%u sp=0x%llx x3=0x%llx\n",
                (unsigned long long) guest_pc, insn, opc, V, mode, L, imm7, rt_reg, rt2_reg, rn,
                (unsigned long long) rt->cpu->sp,
                (unsigned long long) rt->cpu->regs[3]);
    }

    if (V) {
        bool is_pre = (mode == 3);
        bool is_post = (mode == 1);
        bool is_offset = (mode == 2);
        if (!is_pre && !is_post && !is_offset)
            return arm64_jit_helper_unsupported(rt, guest_pc);

        uint32_t size_bytes;
        if (opc == 0)
            size_bytes = 4;
        else if (opc == 1)
            size_bytes = 8;
        else if (opc == 2)
            size_bytes = 16;
        else
            return arm64_jit_helper_unsupported(rt, guest_pc);

        int64_t offset = imm7 * (int64_t) size_bytes;
        uint64_t base = arm64_jit_read_gpr(rt->cpu, rn, true);
        addr_t addr = is_post ? base : (base + offset);
        int rc = -1;

        if (L) {
            if (size_bytes == 4) {
                uint32_t a = 0, b = 0;
                rc = c_load32(rt->tlb, addr, &a);
                if (rc == 0)
                    rc = c_load32(rt->tlb, addr + 4, &b);
                if (rc == 0) {
                    rt->cpu->fp[rt_reg].s[0] = a;
                    memset(&rt->cpu->fp[rt_reg].b[4], 0, sizeof(rt->cpu->fp[rt_reg].b) - 4);
                    rt->cpu->fp[rt2_reg].s[0] = b;
                    memset(&rt->cpu->fp[rt2_reg].b[4], 0, sizeof(rt->cpu->fp[rt2_reg].b) - 4);
                }
            } else if (size_bytes == 8) {
                uint64_t a = 0, b = 0;
                rc = c_load64(rt->tlb, addr, &a);
                if (rc == 0)
                    rc = c_load64(rt->tlb, addr + 8, &b);
                if (rc == 0) {
                    rt->cpu->fp[rt_reg].d[0] = a;
                    rt->cpu->fp[rt_reg].d[1] = 0;
                    rt->cpu->fp[rt2_reg].d[0] = b;
                    rt->cpu->fp[rt2_reg].d[1] = 0;
                }
            } else {
                uint64_t a_lo = 0, a_hi = 0, b_lo = 0, b_hi = 0;
                rc = c_load64(rt->tlb, addr, &a_lo);
                if (rc == 0)
                    rc = c_load64(rt->tlb, addr + 8, &a_hi);
                if (rc == 0)
                    rc = c_load64(rt->tlb, addr + 16, &b_lo);
                if (rc == 0)
                    rc = c_load64(rt->tlb, addr + 24, &b_hi);
                if (rc == 0) {
                    rt->cpu->fp[rt_reg].d[0] = a_lo;
                    rt->cpu->fp[rt_reg].d[1] = a_hi;
                    rt->cpu->fp[rt2_reg].d[0] = b_lo;
                    rt->cpu->fp[rt2_reg].d[1] = b_hi;
                }
            }
        } else {
            if (size_bytes == 4) {
                rc = c_store32(rt->tlb, addr, rt->cpu->fp[rt_reg].s[0]);
                if (rc == 0)
                    rc = c_store32(rt->tlb, addr + 4, rt->cpu->fp[rt2_reg].s[0]);
            } else if (size_bytes == 8) {
                rc = c_store64(rt->tlb, addr, rt->cpu->fp[rt_reg].d[0]);
                if (rc == 0)
                    rc = c_store64(rt->tlb, addr + 8, rt->cpu->fp[rt2_reg].d[0]);
            } else {
                rc = c_store64(rt->tlb, addr, rt->cpu->fp[rt_reg].d[0]);
                if (rc == 0)
                    rc = c_store64(rt->tlb, addr + 8, rt->cpu->fp[rt_reg].d[1]);
                if (rc == 0)
                    rc = c_store64(rt->tlb, addr + 16, rt->cpu->fp[rt2_reg].d[0]);
                if (rc == 0)
                    rc = c_store64(rt->tlb, addr + 24, rt->cpu->fp[rt2_reg].d[1]);
            }
        }

        if (rc != 0) {
            rt->cpu->segfault_addr = rt->tlb->segfault_addr;
            rt->cpu->segfault_was_write = !L;
            rt->cpu->pc = guest_pc;
            rt->resume_pc = guest_pc;
            rt->exit_interrupt = INT_GPF;
            return INT_GPF;
        }

        if (!is_offset) {
            uint64_t writeback = is_post ? (base + offset) : addr;
            if (rn == 31)
                rt->cpu->sp = writeback;
            else
                rt->cpu->regs[rn] = writeback;
        }

        rt->resume_pc = guest_pc + 4;
        rt->cpu->pc = guest_pc + 4;
        rt->exit_interrupt = INT_NONE;
        if (arm64_jit_trace_mode() && (guest_pc == 0xefe62688 || guest_pc == 0xefe6268c)) {
            fprintf(stderr,
                    "[arm64-jit] ldst_pair done pc=0x%llx rc=%d new_sp=0x%llx x3=0x%llx next_pc=0x%llx\n",
                    (unsigned long long) guest_pc, rc,
                    (unsigned long long) rt->cpu->sp,
                    (unsigned long long) rt->cpu->regs[3],
                    (unsigned long long) rt->cpu->pc);
        }
        return INT_NONE;
    }

    bool is64 = (opc == 2);
    if (!(opc == 0 || opc == 2))
        return arm64_jit_helper_unsupported(rt, guest_pc);

    bool is_pre = (mode == 3);
    bool is_post = (mode == 1);
    bool is_offset = (mode == 2);
    if (!is_pre && !is_post && !is_offset)
        return arm64_jit_helper_unsupported(rt, guest_pc);

    int64_t scale = is64 ? 8 : 4;
    int64_t offset = imm7 * scale;
    uint64_t base = arm64_jit_read_gpr(rt->cpu, rn, true);
    addr_t addr = is_post ? base : (base + offset);
    int rc = -1;

    if (L) {
        if (is64) {
            uint64_t a = 0, b = 0;
            rc = c_load64(rt->tlb, addr, &a);
            if (rc == 0)
                rc = c_load64(rt->tlb, addr + 8, &b);
            if (rc == 0) {
                arm64_jit_write_gpr(rt->cpu, rt_reg, a, true);
                arm64_jit_write_gpr(rt->cpu, rt2_reg, b, true);
            }
        } else {
            uint32_t a = 0, b = 0;
            rc = c_load32(rt->tlb, addr, &a);
            if (rc == 0)
                rc = c_load32(rt->tlb, addr + 4, &b);
            if (rc == 0) {
                arm64_jit_write_gpr(rt->cpu, rt_reg, a, false);
                arm64_jit_write_gpr(rt->cpu, rt2_reg, b, false);
            }
        }
    } else {
        uint64_t a = arm64_jit_read_gpr(rt->cpu, rt_reg, false);
        uint64_t b = arm64_jit_read_gpr(rt->cpu, rt2_reg, false);
        if (is64) {
            rc = c_store64(rt->tlb, addr, a);
            if (rc == 0)
                rc = c_store64(rt->tlb, addr + 8, b);
        } else {
            rc = c_store32(rt->tlb, addr, (uint32_t) a);
            if (rc == 0)
                rc = c_store32(rt->tlb, addr + 4, (uint32_t) b);
        }
    }

    if (rc != 0) {
        rt->cpu->segfault_addr = rt->tlb->segfault_addr;
        rt->cpu->segfault_was_write = !L;
        rt->cpu->pc = guest_pc;
        rt->resume_pc = guest_pc;
        rt->exit_interrupt = INT_GPF;
        return INT_GPF;
    }

    if (!is_offset) {
        uint64_t writeback = is_post ? (base + offset) : addr;
        if (rn == 31)
            rt->cpu->sp = writeback;
        else
            rt->cpu->regs[rn] = writeback;
    }

    rt->resume_pc = guest_pc + 4;
    rt->cpu->pc = guest_pc + 4;
    rt->exit_interrupt = INT_NONE;
    if (arm64_jit_trace_mode() && guest_pc == 0xefeb4060) {
        fprintf(stderr,
                "[arm64-jit] ldst_pair done pc=0x%llx x29=0x%llx x30=0x%llx sp=0x%llx next_pc=0x%llx\n",
                (unsigned long long) guest_pc,
                (unsigned long long) rt->cpu->regs[29],
                (unsigned long long) rt->cpu->regs[30],
                (unsigned long long) rt->cpu->sp,
                (unsigned long long) rt->cpu->pc);
    }
    return INT_NONE;
}

static bool arm64_jit_supported_insn(uint32_t insn) {
    switch (arm64_classify_insn(insn)) {
        case INSN_DP_IMM:
        case INSN_DP_REG:
        case INSN_BRANCH:
        case INSN_EXCEPTION:
        case INSN_SYSTEM:
        case INSN_LD_ST:
        case INSN_SIMD_FP:
            return true;
        default:
            return false;
    }
}

static void arm64_jit_record_gpr_use(struct arm64_jit_insn_info *info, uint32_t reg, uint8_t flags) {
    if (reg >= 31)
        return;
    for (uint32_t i = 0; i < info->gpr_use_count; i++) {
        if (info->gpr_uses[i].reg == reg) {
            info->gpr_uses[i].flags |= flags;
            return;
        }
    }
    if (info->gpr_use_count < ARM64_JIT_MAX_GPR_USES) {
        info->gpr_uses[info->gpr_use_count].reg = reg;
        info->gpr_uses[info->gpr_use_count].flags = flags;
        info->gpr_use_count++;
    }
}

bool arm64_jit_analyze_insn(uint32_t insn, struct arm64_jit_insn_info *info) {
    memset(info, 0, sizeof(*info));
    info->type = arm64_classify_insn(insn);
    switch (info->type) {
        case INSN_DP_IMM: {
            info->gpr_use_count = 0;
            if ((insn & 0x1f000000u) == 0x10000000u) { // ADR/ADRP
                arm64_jit_record_gpr_use(info, ARM64_RD(insn), ARM64_JIT_USE_WRITE);
                return true;
            }
            uint32_t rd = ARM64_RD(insn);
            uint32_t rn = ARM64_RN(insn);
            if (rn != 31)
                arm64_jit_record_gpr_use(info, rn, ARM64_JIT_USE_READ);
            else
                info->reads_flags |= 0;
            if (rd != 31)
                arm64_jit_record_gpr_use(info, rd, ARM64_JIT_USE_WRITE);
            info->writes_flags = ((insn >> 29) & 1) && (((insn >> 23) & 0x3f) != 0x24);
            return true;
        }
        case INSN_DP_REG: {
            uint32_t rd = ARM64_RD(insn);
            uint32_t rn = ARM64_RN(insn);
            uint32_t rm = ARM64_RM(insn);
            if (rn != 31)
                arm64_jit_record_gpr_use(info, rn, ARM64_JIT_USE_READ);
            if (rm != 31)
                arm64_jit_record_gpr_use(info, rm, ARM64_JIT_USE_READ);
            if (rd != 31)
                arm64_jit_record_gpr_use(info, rd, ARM64_JIT_USE_WRITE);
            return true;
        }
        case INSN_BRANCH: {
            info->terminates_fragment = 1;
            uint32_t op = (insn >> 26) & 0x3f;
            if ((insn & 0xfe000000u) == 0xd6000000u) {
                uint32_t rn = ARM64_RN(insn);
                if (rn != 31)
                    arm64_jit_record_gpr_use(info, rn, ARM64_JIT_USE_READ);
            } else if ((insn & 0x7e000000u) == 0x34000000u ||
                       (insn & 0x7e000000u) == 0x36000000u) {
                uint32_t rt = ARM64_RT(insn);
                if (rt != 31)
                    arm64_jit_record_gpr_use(info, rt, ARM64_JIT_USE_READ);
            }
            if (op == 0x25)
                arm64_jit_record_gpr_use(info, 30, ARM64_JIT_USE_WRITE);
            info->reads_flags = ((insn & 0xff000010u) == 0x54000000u);
            return true;
        }
        case INSN_EXCEPTION:
            info->terminates_fragment = 1;
            return true;
        case INSN_SYSTEM:
            if ((insn & 0xffffffe0U) == 0xd53bd040U)
                arm64_jit_record_gpr_use(info, ARM64_RD(insn), ARM64_JIT_USE_WRITE);
            else if ((insn & 0xffffffe0U) == 0xd51bd040U)
                arm64_jit_record_gpr_use(info, ARM64_RT(insn), ARM64_JIT_USE_READ);
            else if ((insn & 0xfff00000U) == 0xd5300000U)
                arm64_jit_record_gpr_use(info, ARM64_RD(insn), ARM64_JIT_USE_WRITE);
            return true;
        case INSN_LD_ST: {
            info->accesses_memory = 1;
            uint32_t rn = ARM64_RN(insn);
            uint32_t rt = ARM64_RT(insn);
            if (rn != 31)
                arm64_jit_record_gpr_use(info, rn, ARM64_JIT_USE_READ);
            if ((insn & 0x3a000000u) == 0x28000000u) {
                uint32_t rt2 = ARM64_RT2(insn);
                bool is_load = ((insn >> 22) & 1) != 0;
                if (is_load) {
                    if (rt != 31)
                        arm64_jit_record_gpr_use(info, rt, ARM64_JIT_USE_WRITE);
                    if (rt2 != 31)
                        arm64_jit_record_gpr_use(info, rt2, ARM64_JIT_USE_WRITE);
                } else {
                    if (((insn >> 26) & 1) == 0) {
                        if (rt != 31)
                            arm64_jit_record_gpr_use(info, rt, ARM64_JIT_USE_READ);
                        if (rt2 != 31)
                            arm64_jit_record_gpr_use(info, rt2, ARM64_JIT_USE_READ);
                    }
                }
                return true;
            }
            if ((insn & 0x3b200c00u) == 0x38200800u) {
                uint32_t rm = ARM64_RM(insn);
                if (rm != 31)
                    arm64_jit_record_gpr_use(info, rm, ARM64_JIT_USE_READ);
            }
            if (((insn >> 26) & 1) == 0) {
                bool is_load = ((insn >> 22) & 0x3) != 0;
                if (is_load) {
                    if (rt != 31)
                        arm64_jit_record_gpr_use(info, rt, ARM64_JIT_USE_WRITE);
                } else {
                    if (rt != 31)
                        arm64_jit_record_gpr_use(info, rt, ARM64_JIT_USE_READ);
                }
            }
            return true;
        }
        case INSN_SIMD_FP:
            return true;
        default:
            return false;
    }
}

void arm64_jit_build_fragment_gpr_map(struct arm64_jit_block *block) {
    static const int alloc_pool[ARM64_JIT_MAX_ALLOCATABLE_GPRS] = {
        4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        23, 24, 25, 26, 27, 28,
    };
    for (int i = 0; i < 31; i++) {
        block->gpr_map.host_reg[i] = -1;
        block->gpr_map.use_count[i] = 0;
    }
    for (int i = 0; i < 32; i++)
        block->gpr_map.guest_for_host[i] = -1;
    for (uint32_t i = 0; i < block->insn_count; i++) {
        const struct arm64_jit_insn_info *info = &block->infos[i];
        for (uint32_t j = 0; j < info->gpr_use_count; j++) {
            uint32_t reg = info->gpr_uses[j].reg;
            if (reg < 31)
                block->gpr_map.use_count[reg]++;
        }
    }
    bool used_pool[ARM64_JIT_MAX_ALLOCATABLE_GPRS];
    memset(used_pool, 0, sizeof(used_pool));
    for (int pass = 0; pass < 31; pass++) {
        int best_reg = -1;
        uint32_t best_count = 0;
        for (int reg = 0; reg < 31; reg++) {
            if (block->gpr_map.host_reg[reg] != -1)
                continue;
            if (block->gpr_map.use_count[reg] > best_count) {
                best_count = block->gpr_map.use_count[reg];
                best_reg = reg;
            }
        }
        if (best_reg < 0 || best_count == 0)
            break;
        int pool_idx = -1;
        for (int i = 0; i < ARM64_JIT_MAX_ALLOCATABLE_GPRS; i++) {
            if (!used_pool[i]) {
                pool_idx = i;
                break;
            }
        }
        if (pool_idx < 0)
            break;
        used_pool[pool_idx] = true;
        block->gpr_map.host_reg[best_reg] = alloc_pool[pool_idx];
        block->gpr_map.guest_for_host[alloc_pool[pool_idx]] = best_reg;
    }
}

bool arm64_jit_guest_reg_is_cached(const struct arm64_jit_block *block, uint32_t guest_reg) {
    return guest_reg < 31 && block->gpr_map.host_reg[guest_reg] >= 0;
}

int arm64_jit_host_reg_for_guest(const struct arm64_jit_block *block, uint32_t guest_reg) {
    if (guest_reg >= 31)
        return -1;
    return block->gpr_map.host_reg[guest_reg];
}

static bool arm64_jit_try_enqueue_pc(addr_t *queue, uint32_t *count, addr_t pc,
        addr_t start_pc, page_t base_page) {
    if (pc < start_pc)
        return false;
    if (PAGE(pc) != base_page)
        return false;
    if (((pc - start_pc) & 3) != 0)
        return false;
    for (uint32_t i = 0; i < *count; i++) {
        if (queue[i] == pc)
            return true;
    }
    if (*count >= ARM64_JIT_MAX_INSNS)
        return false;
    queue[(*count)++] = pc;
    return true;
}

static void arm64_jit_collect_local_branch_targets(uint32_t insn, addr_t guest_pc,
        addr_t *queue, uint32_t *count, addr_t start_pc, page_t base_page) {
    uint32_t op = (insn >> 26) & 0x3f;
    if (op == 0x05 || op == 0x25) {
        arm64_jit_try_enqueue_pc(queue, count, guest_pc + arm64_branch_imm26(insn), start_pc, base_page);
        return;
    }
    if ((insn & 0xff000010u) == 0x54000000u) {
        arm64_jit_try_enqueue_pc(queue, count, guest_pc + 4, start_pc, base_page);
        arm64_jit_try_enqueue_pc(queue, count, guest_pc + arm64_branch_imm19(insn), start_pc, base_page);
        return;
    }
    if ((insn & 0x7e000000u) == 0x34000000u) {
        arm64_jit_try_enqueue_pc(queue, count, guest_pc + 4, start_pc, base_page);
        arm64_jit_try_enqueue_pc(queue, count, guest_pc + arm64_branch_imm19(insn), start_pc, base_page);
        return;
    }
    if ((insn & 0x7e000000u) == 0x36000000u) {
        arm64_jit_try_enqueue_pc(queue, count, guest_pc + 4, start_pc, base_page);
        arm64_jit_try_enqueue_pc(queue, count, guest_pc + arm64_branch_imm14(insn), start_pc, base_page);
        return;
    }
    if ((insn & 0xfe000000u) == 0xd6000000u) {
        return;
    }
}

static struct arm64_jit_block *arm64_jit_lookup(struct arm64_jit_state *state, addr_t pc) {
    struct list *bucket = &state->hash[pc % state->hash_size];
    struct arm64_jit_block *block;
    if (list_null(bucket))
        return NULL;
    list_for_each_entry(bucket, block, hash_chain) {
        if (block->start_pc == pc)
            return block;
    }
    return NULL;
}

static void arm64_jit_insert(struct arm64_jit_state *state, struct arm64_jit_block *block) {
    list_init_add(&state->hash[block->start_pc % state->hash_size], &block->hash_chain);
    list_init_add(arm64_jit_blocks_list(state, PAGE(block->start_pc), 0), &block->page[0]);
    if (PAGE(block->start_pc) != PAGE(block->end_pc))
        list_init_add(arm64_jit_blocks_list(state, PAGE(block->end_pc), 1), &block->page[1]);
}

static void arm64_jit_disconnect(struct arm64_jit_block *block) {
    list_remove_safe(&block->hash_chain);
    list_remove_safe(&block->page[0]);
    list_remove_safe(&block->page[1]);
}

void arm64_jit_invalidate_page(struct mmu *mmu, page_t page) {
    struct arm64_jit_state *state = arm64_jit_state_for_mmu(mmu);
    if (state == NULL)
        return;
    lock(&state->lock);
    for (int i = 0; i <= 1; i++) {
        struct list *blocks = arm64_jit_blocks_list(state, page, i);
        struct arm64_jit_block *block, *tmp;
        if (list_null(blocks))
            continue;
        list_for_each_entry_safe(blocks, block, tmp, page[i]) {
            arm64_jit_disconnect(block);
            block->is_jetsam = true;
            list_add(&state->jetsam, &block->jetsam);
        }
    }
    state->invalidate_gen++;
    unlock(&state->lock);
}

static struct arm64_jit_block *arm64_jit_compile_block(addr_t start_pc, struct tlb *tlb, struct arm64_jit_state *state) {
    struct arm64_jit_block *block = calloc(1, sizeof(*block));
    if (block == NULL)
        return NULL;

    block->start_pc = start_pc;
    block->terminal_interrupt = INT_DEBUG;
    page_t base_page = PAGE(start_pc);
    addr_t queue[ARM64_JIT_MAX_INSNS];
    uint32_t queue_count = 1;
    queue[0] = start_pc;

    while (block->insn_count < queue_count && block->insn_count < ARM64_JIT_MAX_INSNS) {
        addr_t insn_pc = queue[block->insn_count];
        addr_t next_pc = insn_pc;
        uint32_t insn = 0;
        if (!arm64_read_insn(&next_pc, tlb, &insn)) {
            block->terminal_interrupt = INT_GPF;
            block->end_pc = insn_pc;
            break;
        }
        block->insn_pcs[block->insn_count] = insn_pc;
        block->insns[block->insn_count++] = insn;
        block->end_pc = next_pc;
        struct arm64_jit_insn_info *info = &block->infos[block->insn_count - 1];
        if (!arm64_jit_supported_insn(insn) || !arm64_jit_analyze_insn(insn, info)) {
            block->unsupported = true;
            break;
        }

        enum arm64_insn_type type = info->type;
        if (type == INSN_BRANCH && !arm64_jit_verify_mode() &&
                block->insn_count + 8 < ARM64_JIT_MAX_INSNS) {
            arm64_jit_collect_local_branch_targets(insn, insn_pc, queue, &queue_count, start_pc, base_page);
        }
        if (type != INSN_BRANCH) {
            if (PAGE(insn_pc) == PAGE(next_pc))
                arm64_jit_try_enqueue_pc(queue, &queue_count, next_pc, start_pc, base_page);
        }
    }

    arm64_jit_build_fragment_gpr_map(block);
    arm64_jit_emit_block(state, block);
    if (arm64_jit_trace_mode()) {
        fprintf(stderr, "[arm64-jit] block 0x%llx..0x%llx insns=%u unsupported=%d term=%d code=%u\n",
                (unsigned long long) block->start_pc,
                (unsigned long long) block->end_pc,
                block->insn_count,
                block->unsupported,
                block->terminal_interrupt,
                block->code_size);
        fprintf(stderr, "[arm64-jit]   gpr-map");
        for (uint32_t reg = 0; reg < 31; reg++) {
            if (block->gpr_map.host_reg[reg] >= 0) {
                fprintf(stderr, " x%u->x%d(%u)", reg, block->gpr_map.host_reg[reg],
                        block->gpr_map.use_count[reg]);
            }
        }
        fprintf(stderr, "\n");
        if (block->start_pc == 0xefe62638) {
            fprintf(stderr,
                    "[arm64-jit]   focus-map x5->x%d(%u) x0->x%d(%u) x1->x%d(%u) x2->x%d(%u) x3->x%d(%u)\n",
                    block->gpr_map.host_reg[5], block->gpr_map.use_count[5],
                    block->gpr_map.host_reg[0], block->gpr_map.use_count[0],
                    block->gpr_map.host_reg[1], block->gpr_map.use_count[1],
                    block->gpr_map.host_reg[2], block->gpr_map.use_count[2],
                    block->gpr_map.host_reg[3], block->gpr_map.use_count[3]);
        }
        for (uint32_t i = 0; i < block->insn_count; i++) {
            fprintf(stderr, "[arm64-jit]   pc=0x%llx insn=0x%08x type=%d\n",
                    (unsigned long long) block->insn_pcs[i],
                    block->insns[i],
                    block->infos[i].type);
        }
    }
    return block;
}

static int arm64_jit_should_exit_timer(struct cpu_state *cpu, struct tlb *tlb) {
    if (__atomic_exchange_n(cpu->poked_ptr, false, __ATOMIC_ACQUIRE))
        return 1;
    if ((++cpu->cycle & ((1 << 10) - 1)) == 0)
        return 1;
    if (tlb->mem_changes != __atomic_load_n(&tlb->mmu->changes, __ATOMIC_ACQUIRE))
        return 1;
    return 0;
}

typedef int (*arm64_jit_entry_fn_t)(struct arm64_jit_runtime *rt);

static bool arm64_jit_cpu_equal(const struct cpu_state *a, const struct cpu_state *b) {
    if (a->sp != b->sp || a->pc != b->pc || a->nzcv != b->nzcv || a->tls_ptr != b->tls_ptr)
        return false;
    if (a->segfault_addr != b->segfault_addr || a->segfault_was_write != b->segfault_was_write)
        return false;
    for (int i = 0; i < 31; i++) {
        if (a->regs[i] != b->regs[i])
            return false;
    }
    if (a->fpcr != b->fpcr || a->fpsr != b->fpsr)
        return false;
    for (int i = 0; i < 32; i++) {
        if (a->fp[i].q != b->fp[i].q)
            return false;
    }
    return true;
}

struct arm64_jit_verify_store_snapshot {
    bool valid;
    addr_t addr;
    uint32_t size;
    uint8_t before[8];
    uint8_t after[8];
};

static bool arm64_jit_is_exclusive_store_insn(uint32_t insn) {
    return (((insn >> 24) & 0x3f) == 0x08) &&
            (((insn >> 12) & 0x7) == 0x7) &&
            (((insn >> 10) & 0x3) == 0x3) &&
            (((insn >> 22) & 1) == 0);
}

static struct arm64_jit_verify_store_snapshot arm64_jit_verify_snapshot_store_before(
        struct cpu_state *cpu, struct tlb *tlb, uint32_t insn) {
    struct arm64_jit_verify_store_snapshot snap = {0};
    if (!arm64_jit_is_exclusive_store_insn(insn))
        return snap;
    uint32_t size = (insn >> 30) & 0x3;
    if (!(size == 2 || size == 3))
        return snap;
    snap.size = (size == 2) ? 4 : 8;
    snap.addr = arm64_jit_read_gpr(cpu, ARM64_RN(insn), true);
    if (tlb_read(tlb, snap.addr, snap.before, snap.size))
        snap.valid = true;
    return snap;
}

static void arm64_jit_verify_snapshot_capture_after(struct arm64_jit_verify_store_snapshot *snap,
        struct tlb *tlb) {
    if (!snap->valid)
        return;
    (void) tlb_read(tlb, snap->addr, snap->after, snap->size);
}

static void arm64_jit_verify_snapshot_restore_before(struct arm64_jit_verify_store_snapshot *snap,
        struct tlb *tlb) {
    if (!snap->valid)
        return;
    (void) tlb_write(tlb, snap->addr, snap->before, snap->size);
}

static void arm64_jit_verify_snapshot_restore_after(struct arm64_jit_verify_store_snapshot *snap,
        struct tlb *tlb) {
    if (!snap->valid)
        return;
    (void) tlb_write(tlb, snap->addr, snap->after, snap->size);
}

int arm64_jit_c_ldst_excl(struct arm64_jit_runtime *rt, addr_t guest_pc, uint32_t insn) {
    uint32_t size = (insn >> 30) & 0x3;
    uint32_t bits29_24 = (insn >> 24) & 0x3f;
    uint32_t R = (insn >> 22) & 1;
    uint32_t rs = (insn >> 16) & 0x1f;
    uint32_t opc = (insn >> 12) & 0x7;
    uint32_t mid = (insn >> 10) & 0x3;
    uint32_t rn = ARM64_RN(insn);
    uint32_t rt_reg = ARM64_RT(insn);

    if (bits29_24 != 0x08 || opc != 7 || mid != 3)
        return arm64_jit_helper_unsupported(rt, guest_pc);

    addr_t addr = arm64_jit_read_gpr(rt->cpu, rn, true);
    int rc = -1;

    if (R == 1 && rs == 31) {
        uint64_t value = 0;
        switch (size) {
            case 2: {
                uint32_t v32 = 0;
                rc = c_load32(rt->tlb, addr, &v32);
                value = v32;
                break;
            }
            case 3:
                rc = c_load64(rt->tlb, addr, &value);
                break;
            default:
                return arm64_jit_helper_unsupported(rt, guest_pc);
        }
        if (rc != 0) {
            rt->cpu->segfault_addr = rt->tlb->segfault_addr;
            rt->cpu->segfault_was_write = false;
            rt->cpu->pc = guest_pc;
            rt->resume_pc = guest_pc;
            rt->exit_interrupt = INT_GPF;
            return INT_GPF;
        }
        arm64_jit_write_gpr(rt->cpu, rt_reg, value, size == 3);
        rt->cpu->excl_addr = addr;
        rt->cpu->excl_val = value;
        if (arm64_jit_verify_mode() && guest_pc == 0xefead1b0) {
            fprintf(stderr,
                    "[arm64-jit] excl-load pc=0x%llx addr=0x%llx size=%u value=0x%llx rt=%u\n",
                    (unsigned long long) guest_pc,
                    (unsigned long long) addr, size,
                    (unsigned long long) value, rt_reg);
        }
        rt->resume_pc = guest_pc + 4;
        rt->cpu->pc = guest_pc + 4;
        rt->exit_interrupt = INT_NONE;
        return INT_NONE;
    }

    if (R == 0) {
        if (rs == 31)
            return arm64_jit_helper_unsupported(rt, guest_pc);

        if (rt->cpu->excl_addr != addr) {
            arm64_jit_write_gpr(rt->cpu, rs, 1, false);
            rt->cpu->excl_addr = UINT64_MAX;
            rt->resume_pc = guest_pc + 4;
            rt->cpu->pc = guest_pc + 4;
            rt->exit_interrupt = INT_NONE;
            return INT_NONE;
        }

        uint64_t expected_val = rt->cpu->excl_val;
        uint64_t new_val = arm64_jit_read_gpr(rt->cpu, rt_reg, false);
        uint64_t cur_mem = 0;
        if (size == 2) {
            uint32_t cur32 = 0;
            if (c_load32(rt->tlb, addr, &cur32) == 0)
                cur_mem = cur32;
        } else if (size == 3) {
            (void) c_load64(rt->tlb, addr, &cur_mem);
        }
        if (arm64_jit_verify_mode() && guest_pc == 0xefead1b8) {
            fprintf(stderr,
                    "[arm64-jit] excl-store pre pc=0x%llx addr=0x%llx excl_addr=0x%llx expected=0x%llx cur=0x%llx new=0x%llx rs=%u rt=%u size=%u\n",
                    (unsigned long long) guest_pc,
                    (unsigned long long) addr,
                    (unsigned long long) rt->cpu->excl_addr,
                    (unsigned long long) expected_val,
                    (unsigned long long) cur_mem,
                    (unsigned long long) new_val,
                    rs, rt_reg, size);
        }
        switch (size) {
            case 2:
            case 3:
                rc = c_stxr_cas(rt->tlb, addr, expected_val, new_val, size);
                break;
            default:
                return arm64_jit_helper_unsupported(rt, guest_pc);
        }

        rt->cpu->excl_addr = UINT64_MAX;
        if (rc < 0) {
            rt->cpu->segfault_addr = rt->tlb->segfault_addr;
            rt->cpu->segfault_was_write = true;
            rt->cpu->pc = guest_pc;
            rt->resume_pc = guest_pc;
            rt->exit_interrupt = INT_GPF;
            return INT_GPF;
        }

        arm64_jit_write_gpr(rt->cpu, rs, (uint64_t) rc, false);
        if (arm64_jit_verify_mode() && guest_pc == 0xefead1b8) {
            fprintf(stderr,
                    "[arm64-jit] excl-store post pc=0x%llx rc=%d result_x%u=0x%llx\n",
                    (unsigned long long) guest_pc, rc, rs,
                    (unsigned long long) arm64_jit_read_gpr(rt->cpu, rs, false));
        }
        rt->resume_pc = guest_pc + 4;
        rt->cpu->pc = guest_pc + 4;
        rt->exit_interrupt = INT_NONE;
        return INT_NONE;
    }

    return arm64_jit_helper_unsupported(rt, guest_pc);
}

static void arm64_jit_dump_cpu_diff(const struct cpu_state *expected, const struct cpu_state *actual,
        addr_t start_pc, int expected_interrupt, int actual_interrupt) {
    fprintf(stderr,
            "[arm64-jit-verify] mismatch at start_pc=0x%llx expected_int=%d actual_int=%d\n",
            (unsigned long long) start_pc, expected_interrupt, actual_interrupt);
    if (expected->pc != actual->pc) {
        fprintf(stderr, "  pc expected=0x%llx actual=0x%llx\n",
                (unsigned long long) expected->pc, (unsigned long long) actual->pc);
    }
    if (expected->sp != actual->sp) {
        fprintf(stderr, "  sp expected=0x%llx actual=0x%llx\n",
                (unsigned long long) expected->sp, (unsigned long long) actual->sp);
    }
    if (expected->nzcv != actual->nzcv) {
        fprintf(stderr, "  nzcv expected=0x%x actual=0x%x\n", expected->nzcv, actual->nzcv);
    }
    if (expected->tls_ptr != actual->tls_ptr) {
        fprintf(stderr, "  tls expected=0x%llx actual=0x%llx\n",
                (unsigned long long) expected->tls_ptr, (unsigned long long) actual->tls_ptr);
    }
    for (int i = 0; i < 31; i++) {
        if (expected->regs[i] != actual->regs[i]) {
            fprintf(stderr, "  x%d expected=0x%llx actual=0x%llx\n", i,
                    (unsigned long long) expected->regs[i],
                    (unsigned long long) actual->regs[i]);
        }
    }
    if (expected->fpcr != actual->fpcr || expected->fpsr != actual->fpsr) {
        fprintf(stderr, "  fpcr/fpsr expected=0x%x/0x%x actual=0x%x/0x%x\n",
                expected->fpcr, expected->fpsr, actual->fpcr, actual->fpsr);
    }
    for (int i = 0; i < 32; i++) {
        if (expected->fp[i].q != actual->fp[i].q) {
            fprintf(stderr, "  v%d expected=0x%016llx%016llx actual=0x%016llx%016llx\n", i,
                    (unsigned long long) expected->fp[i].d[1],
                    (unsigned long long) expected->fp[i].d[0],
                    (unsigned long long) actual->fp[i].d[1],
                    (unsigned long long) actual->fp[i].d[0]);
            break;
        }
    }
    if (expected->segfault_addr != actual->segfault_addr ||
            expected->segfault_was_write != actual->segfault_was_write) {
        fprintf(stderr, "  fault expected=0x%llx/%d actual=0x%llx/%d\n",
                (unsigned long long) expected->segfault_addr, expected->segfault_was_write,
                (unsigned long long) actual->segfault_addr, actual->segfault_was_write);
    }
}

static int arm64_jit_run_block(struct arm64_jit_block *block, struct cpu_state *cpu, struct tlb *tlb) {
    struct arm64_jit_runtime rt = {
        .cpu = cpu,
        .tlb = tlb,
        .block = block,
        .spill_state_fn = (void (*)(struct arm64_jit_runtime *)) block->spill_state_fn,
        .reload_state_fn = (void (*)(struct arm64_jit_runtime *)) block->reload_state_fn,
        .resume_pc = block->start_pc,
        .fault_pc = block->start_pc,
        .exit_interrupt = INT_NONE,
    };
    if (!arm64_jit_verify_mode() && arm64_jit_should_exit_timer(cpu, tlb)) {
        cpu->pc = block->start_pc;
        return INT_TIMER;
    }

    if (arm64_jit_trace_mode() &&
            block->start_pc >= 0xefeb3600 && block->start_pc <= 0xefeb3900) {
        fprintf(stderr,
                "[arm64-jit] exec-enter start=0x%llx end=0x%llx cpu_pc=0x%llx x0=0x%llx x1=0x%llx x2=0x%llx x30=0x%llx\n",
                (unsigned long long) block->start_pc,
                (unsigned long long) block->end_pc,
                (unsigned long long) cpu->pc,
                (unsigned long long) cpu->regs[0],
                (unsigned long long) cpu->regs[1],
                (unsigned long long) cpu->regs[2],
                (unsigned long long) cpu->regs[30]);
    }

    extern __thread volatile sig_atomic_t in_jit;
    arm64_jit_set_saved_pc(block->start_pc);
    g_arm64_jit_runtime = &rt;
    in_jit = 1;
    int interrupt = ((arm64_jit_entry_fn_t) block->code_rx)(&rt);
    in_jit = 0;
    g_arm64_jit_runtime = NULL;

    if (interrupt != INT_JIT_CRASH)
        interrupt = rt.exit_interrupt;

    if (arm64_jit_trace_mode() &&
            block->start_pc >= 0xefeb3600 && block->start_pc <= 0xefeb3900) {
        fprintf(stderr,
                "[arm64-jit] exec-leave start=0x%llx end=0x%llx interrupt=%d resume_pc=0x%llx cpu_pc=0x%llx x0=0x%llx x1=0x%llx x2=0x%llx x30=0x%llx\n",
                (unsigned long long) block->start_pc,
                (unsigned long long) block->end_pc,
                interrupt,
                (unsigned long long) rt.resume_pc,
                (unsigned long long) cpu->pc,
                (unsigned long long) cpu->regs[0],
                (unsigned long long) cpu->regs[1],
                (unsigned long long) cpu->regs[2],
                (unsigned long long) cpu->regs[30]);
    }

    if (!arm64_jit_verify_mode() && rt.resume_pc < 0x1000) {
        fprintf(stderr,
                "[arm64-jit] bad block return start=0x%llx end=0x%llx interrupt=%d resume_pc=0x%llx fault_pc=0x%llx cpu_pc=0x%llx x0=0x%llx x1=0x%llx x2=0x%llx x30=0x%llx\n",
                (unsigned long long) block->start_pc,
                (unsigned long long) block->end_pc,
                interrupt,
                (unsigned long long) rt.resume_pc,
                (unsigned long long) rt.fault_pc,
                (unsigned long long) cpu->pc,
                (unsigned long long) cpu->regs[0],
                (unsigned long long) cpu->regs[1],
                (unsigned long long) cpu->regs[2],
                (unsigned long long) cpu->regs[30]);
    }

    if (interrupt == INT_NONE) {
        cpu->pc = rt.resume_pc;
    } else if (cpu->pc == 0) {
        cpu->pc = rt.resume_pc;
    }
    return interrupt;
}

static void arm64_jit_dump_verify_step(uint64_t step, addr_t start_pc,
        const struct cpu_state *expected, const struct cpu_state *actual,
        int expected_interrupt, int actual_interrupt) {
    fprintf(stderr,
            "[arm64-jit-verify] step=%llu start_pc=0x%llx expected_pc=0x%llx actual_pc=0x%llx expected_int=%d actual_int=%d\n",
            (unsigned long long) step,
            (unsigned long long) start_pc,
            (unsigned long long) expected->pc,
            (unsigned long long) actual->pc,
            expected_interrupt, actual_interrupt);
}

int cpu_run_to_interrupt_arm64_jit(struct cpu_state *cpu, struct tlb *tlb) {
    if (cpu->poked_ptr == NULL)
        cpu->poked_ptr = &cpu->_poked;

    struct arm64_jit_state *state = arm64_jit_state_for_mmu(cpu->mmu);
    if (state == NULL)
        return cpu_run_to_interrupt_threaded(cpu, tlb);

    read_wrlock(&state->jetsam_lock);
    static uint64_t verify_steps;
    while (true) {
        if (arm64_jit_trace_mode() && cpu->pc < 0x1000) {
            fprintf(stderr, "[arm64-jit] low-pc loop entry pc=0x%llx x0=0x%llx x1=0x%llx x2=0x%llx x30=0x%llx\n",
                    (unsigned long long) cpu->pc,
                    (unsigned long long) cpu->regs[0],
                    (unsigned long long) cpu->regs[1],
                    (unsigned long long) cpu->regs[2],
                    (unsigned long long) cpu->regs[30]);
        }
        if (tlb->mem_changes != __atomic_load_n(&tlb->mmu->changes, __ATOMIC_ACQUIRE))
            tlb_flush(tlb);

        lock(&state->lock);
        struct arm64_jit_block *block = arm64_jit_lookup(state, cpu->pc);
        if (block == NULL) {
            block = arm64_jit_compile_block(cpu->pc, tlb, state);
            if (block != NULL)
                arm64_jit_insert(state, block);
        }
        unlock(&state->lock);

        if (block == NULL || block->code_rx == NULL) {
            if (arm64_jit_trace_mode()) {
                fprintf(stderr, "[arm64-jit] block missing/unemitted pc=0x%llx block=%p code=%p\n",
                        (unsigned long long) cpu->pc, (void *) block,
                        block ? block->code_rx : NULL);
            }
            read_wrunlock(&state->jetsam_lock);
            return INT_GPF;
        }

        struct cpu_state verify_expected_cpu;
        struct tlb verify_expected_tlb;
        addr_t verify_start_pc = cpu->pc;
        struct arm64_jit_verify_store_snapshot verify_store_snapshot = {0};
        if (arm64_jit_verify_mode()) {
            verify_expected_cpu = *cpu;
            verify_expected_tlb = *tlb;
            if (block->insn_count > 0)
                verify_store_snapshot = arm64_jit_verify_snapshot_store_before(cpu, tlb, block->insns[0]);
            fprintf(stderr,
                    "[arm64-jit-verify] enter step=%llu start_pc=0x%llx first_insn=0x%08x block_insns=%u\n",
                    (unsigned long long) (verify_steps + 1),
                    (unsigned long long) verify_start_pc,
                    block->insn_count ? block->insns[0] : 0,
                    block->insn_count);
        }

        int interrupt = arm64_jit_run_block(block, cpu, tlb);
        if (arm64_jit_verify_mode()) {
            arm64_jit_verify_snapshot_capture_after(&verify_store_snapshot, tlb);
            arm64_jit_verify_snapshot_restore_before(&verify_store_snapshot, &verify_expected_tlb);
            if (arm64_jit_trace_mode() && verify_start_pc == 0xefeb5034) {
                fprintf(stderr,
                        "[arm64-jit-verify] post-run start_pc=0x%llx x3=0x%llx nzcv=0x%x pc=0x%llx interrupt=%d\n",
                        (unsigned long long) verify_start_pc,
                        (unsigned long long) cpu->regs[3],
                        cpu->nzcv,
                        (unsigned long long) cpu->pc,
                        interrupt);
            }
            verify_steps++;
            int expected_interrupt = cpu_single_step_threaded_oracle(&verify_expected_cpu, &verify_expected_tlb);
            int compare_interrupt = interrupt;
            if (compare_interrupt == INT_BREAKPOINT || compare_interrupt == INT_NONE)
                compare_interrupt = INT_NONE;
            if (expected_interrupt == INT_DEBUG || expected_interrupt == INT_NONE)
                expected_interrupt = INT_NONE;
            arm64_jit_dump_verify_step(verify_steps, verify_start_pc,
                    &verify_expected_cpu, cpu, expected_interrupt, compare_interrupt);
            if (expected_interrupt != compare_interrupt ||
                    !arm64_jit_cpu_equal(&verify_expected_cpu, cpu)) {
                arm64_jit_verify_snapshot_restore_after(&verify_store_snapshot, tlb);
                arm64_jit_dump_cpu_diff(&verify_expected_cpu, cpu, verify_start_pc,
                        expected_interrupt, compare_interrupt);
                read_wrunlock(&state->jetsam_lock);
                return INT_DEBUG;
            }
            arm64_jit_verify_snapshot_restore_after(&verify_store_snapshot, tlb);
            if (expected_interrupt != INT_NONE) {
                read_wrunlock(&state->jetsam_lock);
                return interrupt;
            }
            read_wrunlock(&state->jetsam_lock);
            return INT_TIMER;
        }
        if (interrupt != INT_NONE) {
            cpu_log_interrupt_boundary("jit", cpu, interrupt);
            read_wrunlock(&state->jetsam_lock);
            return interrupt;
        }
    }
}

#endif
