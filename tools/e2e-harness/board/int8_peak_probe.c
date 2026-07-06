/* int8_peak_probe.c -- register-resident sustained int8 vector-MAC throughput.
 *
 * This is the "peak int8 compute" leg of the roofline (the membw_probe.c is the
 * read-bandwidth leg). It measures the ACHIEVABLE (not paper) int8 multiply-
 * accumulate rate the vector unit sustains with zero memory traffic: 8 fully
 * independent widening-MAC chains (vint8m1 * vint8m1 -> vint16m2 accumulate)
 * kept live in the vector register file, so the only bottleneck is vwmacc issue.
 *
 * One vwmacc.vv over vl int8 lanes = vl int8 MACs = vl multiplies + vl adds.
 * Reported:  MACs = iters * NACC * vl ;  GMACs = MACs/secs/1e9 ;  GOPS = 2*GMACs.
 *
 * The 8 chains each read/write their OWN accumulator (loop-carried self-dep) so
 * clang cannot strength-reduce them to a closed form (verify with objdump: the
 * hot loop must contain NACC vwmacc.vv). va/vb are register-resident constants;
 * accumulators feed a final reduction into `sink` to defeat DCE. i16 wrap is
 * intentional and harmless (we count ops, not the sum).
 *
 * argv: [iters]   default 400,000,000   prints:
 *   INT8PEAK GMACs=<x> GOPS=<y> vl=<vl> nacc=8 iters=<n> secs=<s> sink=<z>
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <riscv_vector.h>

#define NACC 8

int main(int argc, char** argv){
    long iters = (argc>1)? atol(argv[1]) : 400000000L;
    size_t vl = __riscv_vsetvl_e8m1(1024);          /* = VLEN/8 int8 lanes @LMUL1 */

    vint8m1_t va = __riscv_vmv_v_x_i8m1(3, vl);
    vint8m1_t vb = __riscv_vmv_v_x_i8m1(5, vl);
    vint16m2_t a0=__riscv_vmv_v_x_i16m2(0,vl), a1=__riscv_vmv_v_x_i16m2(0,vl),
               a2=__riscv_vmv_v_x_i16m2(0,vl), a3=__riscv_vmv_v_x_i16m2(0,vl),
               a4=__riscv_vmv_v_x_i16m2(0,vl), a5=__riscv_vmv_v_x_i16m2(0,vl),
               a6=__riscv_vmv_v_x_i16m2(0,vl), a7=__riscv_vmv_v_x_i16m2(0,vl);

    struct timespec t0,t1;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    for(long it=0; it<iters; it++){
        a0=__riscv_vwmacc_vv_i16m2(a0,va,vb,vl);
        a1=__riscv_vwmacc_vv_i16m2(a1,va,vb,vl);
        a2=__riscv_vwmacc_vv_i16m2(a2,va,vb,vl);
        a3=__riscv_vwmacc_vv_i16m2(a3,va,vb,vl);
        a4=__riscv_vwmacc_vv_i16m2(a4,va,vb,vl);
        a5=__riscv_vwmacc_vv_i16m2(a5,va,vb,vl);
        a6=__riscv_vwmacc_vv_i16m2(a6,va,vb,vl);
        a7=__riscv_vwmacc_vv_i16m2(a7,va,vb,vl);
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);

    a0=__riscv_vadd_vv_i16m2(a0,a1,vl); a2=__riscv_vadd_vv_i16m2(a2,a3,vl);
    a4=__riscv_vadd_vv_i16m2(a4,a5,vl); a6=__riscv_vadd_vv_i16m2(a6,a7,vl);
    a0=__riscv_vadd_vv_i16m2(a0,a2,vl); a4=__riscv_vadd_vv_i16m2(a4,a6,vl);
    a0=__riscv_vadd_vv_i16m2(a0,a4,vl);
    vint16m1_t z = __riscv_vmv_v_x_i16m1(0,vl);
    vint16m1_t r = __riscv_vredsum_vs_i16m2_i16m1(a0,z,vl);
    long sink = (long)__riscv_vmv_x_s_i16m1_i16(r);

    double secs = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    double macs = (double)iters*(double)NACC*(double)vl;
    printf("INT8PEAK GMACs=%.3f GOPS=%.3f vl=%zu nacc=%d iters=%ld secs=%.4f sink=%ld\n",
           macs/secs/1e9, 2.0*macs/secs/1e9, vl, NACC, iters, secs, sink);
    return 0;
}
