#ifndef EMU_CPU_H
#define EMU_CPU_H

#include "misc.h"
#include "emu/mmu.h"

#ifdef __KERNEL__
#include <linux/stddef.h>
#else
#include <stddef.h>
#endif

// Include architecture-specific CPU state definition
#if defined(GUEST_X86)
#include "emu/arch/x86/cpu.h"
#elif defined(GUEST_ARM64)
#include "emu/arch/arm64/cpu.h"
#else
// Default to x86 for backward compatibility
#include "emu/arch/x86/cpu.h"
#endif

// Common CPU interface
struct cpu_state;
struct tlb;
enum cpu_backend_kind {
    CPU_BACKEND_UNSET = 0,
    CPU_BACKEND_THREADED = 1,
    CPU_BACKEND_ARM64_JIT = 2,
};
int cpu_run_to_interrupt(struct cpu_state *cpu, struct tlb *tlb);
int cpu_run_to_interrupt_threaded(struct cpu_state *cpu, struct tlb *tlb);
int cpu_single_step_threaded_oracle(struct cpu_state *cpu, struct tlb *tlb);
void cpu_log_interrupt_boundary(const char *who, const struct cpu_state *cpu, int interrupt);
#if defined(GUEST_ARM64) && defined(__aarch64__)
int cpu_run_to_interrupt_arm64_jit(struct cpu_state *cpu, struct tlb *tlb);
void arm64_jit_invalidate_page(struct mmu *mmu, page_t page);
#endif
void cpu_poke(struct cpu_state *cpu);

#define CPU_OFFSET(field) offsetof(struct cpu_state, field)

#endif
