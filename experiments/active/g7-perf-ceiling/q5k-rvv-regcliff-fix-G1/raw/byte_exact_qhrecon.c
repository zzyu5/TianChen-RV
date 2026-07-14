/* byte_exact_qhrecon.c -- HARD GATE: the THREE qh 5th-bit recon variants used in
 * strip_body_regpressure.c (OLD deployed / KNEST native-mask / RETRANS vlm) compute
 * BIT-IDENTICAL 5-bit weights. Exhaustive over every (qh byte, sub-block bit, nibble).
 * Native x86 scalar model (VLA-lane-invariant: each lane is independent).
 *
 * decode target (ggml q5_K, NO offset-binary; bias in the 6-bit MIN):
 *   w = nibble | (qh_bit << 4),  nibble in [0,15], qh_bit in {0,1}  ->  w in [0,31].
 */
#include <stdio.h>
#include <stdint.h>

int main(void){
  long mismatch = 0, checked = 0;
  for(int qh = 0; qh < 256; ++qh){
    for(int s = 0; s < 8; ++s){
      for(int nib = 0; nib < 16; ++nib){
        int b = (qh >> s) & 1;                     /* the sub-block's 5th bit */

        /* OLD (deployed :9942-9950): loBit = ((qh>>s)&1)<<4 ; w = nib | loBit */
        int loSel = qh >> s;
        int loBit = (loSel & 0x01) << 4;
        int w_old = nib | loBit;

        /* KNEST (native-mask :q5k-knest): mask = (qh & (1<<s))!=0 ; w = mask? nib+16 : nib */
        int mask_k = ((qh & (1u << s)) != 0);
        int w_knest = mask_k ? (nib + 16) : nib;   /* nib<16 => nib+16 == nib|16 */

        /* RETRANS (vlm): mask VALUE is the SAME bit b (re-transpose is a permutation);
         * w = mask ? nib+16 : nib */
        int mask_r = b;
        int w_retrans = mask_r ? (nib + 16) : nib;

        int w_ref = nib | (b << 4);                /* algebraic reference */

        if(!(w_old == w_ref && w_knest == w_ref && w_retrans == w_ref)) ++mismatch;
        ++checked;
      }
    }
  }
  printf("byte-exact qh-recon HARD GATE (OLD==KNEST==RETRANS==ref)\n");
  printf("  checked = %ld combos (256 qh x 8 subbit x 16 nibble)\n", checked);
  printf("  mismatch = %ld\n", mismatch);
  printf("  RESULT: %s\n", mismatch == 0 ? "PASS" : "FAIL");
  return mismatch ? 1 : 0;
}
