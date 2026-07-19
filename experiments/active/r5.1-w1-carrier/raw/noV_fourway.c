// Four-way physical no-V machine-check for the 超锐/scalar board.
// Compile: clang-18 -march=rv64gc -mabi=lp64d -O2 noV_fourway.c -o noV_fourway
// Ways 1 & 2 (kernel /proc/cpuinfo ISA, firmware device-tree ISA) are captured by
// the harness shell; ways 3 & 4 are here (independent from the reported metadata).
#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <setjmp.h>
#include <sys/auxv.h>

#ifndef AT_HWCAP
#define AT_HWCAP 16
#endif
// RISC-V AT_HWCAP: single-letter ISA bits, bit ('V'-'A')=21 for the V extension.
#define RISCV_HWCAP_V (1UL << ('V' - 'A'))

static sigjmp_buf jb;
static volatile sig_atomic_t trapped;
static void onill(int s) { (void)s; trapped = 1; siglongjmp(jb, 1); }

int main(void) {
  // Way 3: ELF auxiliary vector AT_HWCAP V bit (libc/kernel handshake, independent
  // of the /proc text and the device-tree blob).
  unsigned long hw = getauxval(AT_HWCAP);
  int hwcap_v = (hw & RISCV_HWCAP_V) ? 1 : 0;
  printf("WAY3_AT_HWCAP hwcap=0x%lx V_bit21=%d verdict=%s\n",
         hw, hwcap_v, hwcap_v ? "V_PRESENT" : "NO_V");

  // Way 4: dynamic hardware trap. The raw 32-bit word 0x0C007057 is a V-major
  // opcode (OP-V, funct3=111) = `vsetvli x0, x0, e8, m1, ta, ma`. Emitted as raw
  // bytes so it does NOT depend on the assembler's -march (this is a HARDWARE
  // probe, not a toolchain probe): on a hart with no V unit the instruction is
  // unimplemented and traps SIGILL.
  struct sigaction sa; memset(&sa, 0, sizeof sa);
  sa.sa_handler = onill; sigaction(SIGILL, &sa, 0);
  trapped = 0;
  int exec_ok = 0;
  if (sigsetjmp(jb, 1) == 0) {
    __asm__ volatile(".word 0x0C007057" ::: "memory");  // vsetvli x0,x0,e8,m1,ta,ma
    exec_ok = 1;
  }
  printf("WAY4_VOPCODE_TRAP word=0x0C007057 exec_ok=%d sigill_trapped=%d verdict=%s\n",
         exec_ok, (int)trapped,
         trapped ? "PHYSICAL_NO_V_SIGILL" : (exec_ok ? "V_PRESENT" : "UNKNOWN"));

  int no_v = (!hwcap_v) && (int)trapped && !exec_ok;
  printf("DYNAMIC_NO_V_SUMMARY hwcap_no_v=%d hw_trap_no_v=%d => %s\n",
         !hwcap_v, ((int)trapped && !exec_ok),
         no_v ? "PHYSICAL_NO_V_CONFIRMED" : "INCONSISTENT");
  return no_v ? 0 : 1;
}
