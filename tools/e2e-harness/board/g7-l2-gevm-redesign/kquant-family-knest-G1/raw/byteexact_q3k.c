// q3_K hmask 3rd-bit reconstruction byte-exact model (G1 hard gate).
// Proves: OLD-style (vsll+vor+vsub) === NATIVE-mask (vmseq+vadd_mu) === stock scalar.
// q3_K value = base2 (0..3, from qs) with the hmask bit as the "high"/sign bit,
// biased by -4:  stock does q = base2; if (hbit==0) q -= 4.
// Enumerate base2 in [0,3] x hbit in {0,1}. 8 cases. 0 mismatch = PASS.
#include <stdio.h>
#include <stdint.h>

// stock scalar (ggml q3_K semantics: mask TRUE when hbit==0 -> subtract 4)
static int8_t ref_scalar(int base2, int hbit){
    int8_t q = (int8_t)base2;
    if (hbit == 0) q -= 4;
    return q;
}
// OLD-style vector lane algebra (what our current emitc emits per element):
//   base = qs & 0x3;  hb = hplane & 0x1;  v = base | (hb<<2);  q = (int8)v - 4;
static int8_t old_lane(int base2, int hbit){
    uint8_t base = (uint8_t)(base2 & 0x3);
    uint8_t hb   = (uint8_t)(hbit & 0x1);
    uint8_t v    = (uint8_t)(base | (hb << 2));
    return (int8_t)((int8_t)v - 4);
}
// NATIVE-mask lane algebra (KNEST): mask0 = (hplane==0); q = base_i8 + (mask0 ? -4 : 0)
//   via vmseq(hplane,0) + vadd_vx_i8mf2_mu(mask0, base, base, -4)
static int8_t native_lane(int base2, int hbit){
    int8_t base = (int8_t)(base2 & 0x3);
    int mask0 = ((hbit & 0x1) == 0);       // vmseq.vx (hplane, 0)
    return mask0 ? (int8_t)(base - 4) : base; // vadd_vx_i8mf2_mu under mask0
}
int main(void){
    int mism_old=0, mism_nat=0, n=0;
    for(int base2=0; base2<4; base2++) for(int hbit=0; hbit<2; hbit++){
        int8_t r = ref_scalar(base2,hbit);
        int8_t o = old_lane(base2,hbit);
        int8_t k = native_lane(base2,hbit);
        n++;
        if(o!=r) mism_old++;
        if(k!=r) mism_nat++;
    }
    printf("q3_K hmask recon byte-exact: cases=%d  OLD_vs_stock_mismatch=%d  NATIVE_vs_stock_mismatch=%d\n",
           n, mism_old, mism_nat);
    printf("VERDICT: %s\n", (mism_old==0 && mism_nat==0) ? "PASS (native-mask === OLD === stock, 0/8)" : "FAIL");
    return (mism_old==0 && mism_nat==0)?0:1;
}
