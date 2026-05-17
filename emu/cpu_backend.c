#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "emu/cpu.h"
#include "emu/interrupt.h"
#include "kernel/task.h"

static int parse_backend_env(void) {
#if defined(GUEST_ARM64) && defined(__aarch64__)
    const char *value = getenv("ISH_ARM64_BACKEND");
    if (value != NULL && strcmp(value, "arm64_jit") == 0)
        return CPU_BACKEND_ARM64_JIT;
#endif
    return CPU_BACKEND_THREADED;
}

int cpu_run_to_interrupt(struct cpu_state *cpu, struct tlb *tlb) {
    if (current != NULL && current->exec_backend == CPU_BACKEND_UNSET)
        current->exec_backend = parse_backend_env();

    int backend = current != NULL ? current->exec_backend : CPU_BACKEND_THREADED;
#if defined(GUEST_ARM64) && defined(__aarch64__)
    if (backend == CPU_BACKEND_ARM64_JIT)
        return cpu_run_to_interrupt_arm64_jit(cpu, tlb);
#endif
    (void) backend;
    return cpu_run_to_interrupt_threaded(cpu, tlb);
}

void cpu_log_interrupt_boundary(const char *who, const struct cpu_state *cpu, int interrupt) {
    static int boundary_mode = -1;
    if (boundary_mode == -1) {
        const char *env = getenv("ISH_ARM64_BOUNDARY_TRACE");
        boundary_mode = (env != NULL && env[0] == '1') ? 1 : 0;
    }
    if (!boundary_mode || interrupt == INT_NONE)
        return;
    fprintf(stderr,
            "[interrupt-boundary] %s pc=0x%llx interrupt=%d fault=0x%llx write=%d x0=0x%llx x1=0x%llx x2=0x%llx x30=0x%llx nzcv=0x%x\n",
            who,
            (unsigned long long) cpu->pc,
            interrupt,
            (unsigned long long) cpu->segfault_addr,
            cpu->segfault_was_write,
            (unsigned long long) cpu->regs[0],
            (unsigned long long) cpu->regs[1],
            (unsigned long long) cpu->regs[2],
            (unsigned long long) cpu->regs[30],
            cpu->nzcv);
}
