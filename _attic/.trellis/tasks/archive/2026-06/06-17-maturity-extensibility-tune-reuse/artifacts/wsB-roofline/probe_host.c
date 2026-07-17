#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <riscv_vector.h>
static inline uint64_t rdcycle(void){ uint64_t c; asm volatile("rdcycle %0":"=r"(c)); return c; }
int main(void){
  // VLEN via vlenb CSR (bytes per vector reg)
  unsigned long vlenb=0; int csr_ok=1;
  // vlenb is CSR 0xC22; read may trap on some boards -> guard via separate test
  // Use vsetvl probe instead (always legal):
  size_t maxe8m1 = __riscv_vsetvlmax_e8m1();
  size_t maxe8m8 = __riscv_vsetvlmax_e8m8();
  printf("vsetvlmax_e8m1=%zu  vsetvlmax_e8m8=%zu  => VLEN_bits=%zu\n", maxe8m1, maxe8m8, maxe8m1*8);
  // rdcycle test
  uint64_t a=rdcycle(); for(volatile int i=0;i<1000;i++); uint64_t b=rdcycle();
  printf("rdcycle delta over 1000 nops = %llu (0 or huge => traps/emulated)\n",(unsigned long long)(b-a));
  // clock + estimate freq via busy loop timed by MONOTONIC
  struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
  uint64_t c0=rdcycle(); volatile uint64_t x=0; for(uint64_t i=0;i<200000000ULL;i++) x+=i;
  uint64_t c1=rdcycle(); clock_gettime(CLOCK_MONOTONIC,&t1);
  double ns=(t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec);
  printf("busyloop: cycles=%llu  ns=%.0f  => est_freq_GHz=%.3f (cyc/ns)\n",(unsigned long long)(c1-c0),ns,(double)(c1-c0)/ns);
  (void)csr_ok;(void)x;
  return 0;
}
