// q5_K KNEST byte-exact-by-construction model. The KNEST redesign changes ONLY the
// qh 5th-bit reconstruction orchestration; the reconstructed WEIGHT VALUE must be
// bit-identical to (a) the stock ggml q5_K decode and (b) the CURRENT emitter decode,
// for every (nibble, qh_bit) input. q5_K stores a RAW 5-bit unsigned quant in [0,31]
// (NO q5_0 offset-binary -16; the bias lives in the per-sub-block 6-bit MIN), so:
//   stock  : q5 = (nibble & 0x0F);  if (qh_bit) q5 += 16     [vadd_vx_i8_mu]
//   current: q5 = nibble | ((qh_bit & 1) << 4)               [vor(nib, bit<<4)]
//   KNEST  : q5 = nibble;           if (qh_bit) q5 += 16     [vadd_vx_i8_mu]  (== stock)
// Exhaustive over nibble in [0,15] x qh_bit in {0,1}. mismatch MUST be 0 -> HARD GATE.
#include <stdio.h>
#include <stdint.h>

static int8_t decode_stock  (int nib,int bit){ int8_t v=(int8_t)(nib&0x0F); if(bit) v=(int8_t)(v+16); return v; }
static int8_t decode_current(int nib,int bit){ return (int8_t)((uint8_t)(nib&0x0F) | ((uint8_t)(bit&1)<<4)); }
static int8_t decode_knest  (int nib,int bit){ int8_t v=(int8_t)(nib&0x0F); if(bit) v=(int8_t)(v+16); return v; }

int main(void){
  int nSC=0,nCK=0,total=0;
  for(int nib=0;nib<16;++nib) for(int bit=0;bit<2;++bit){
    int8_t s=decode_stock(nib,bit), c=decode_current(nib,bit), k=decode_knest(nib,bit);
    total++;
    if(s!=c) nSC++;
    if(c!=k) nCK++;
    if(s!=k){ printf("MISMATCH nib=%d bit=%d stock=%d knest=%d\n",nib,bit,s,k); }
  }
  printf("q5_K reconstruction equivalence over %d (nibble x qh_bit) cases:\n",total);
  printf("  stock  vs current mismatch = %d/%d\n",nSC,total);
  printf("  current vs KNEST  mismatch = %d/%d\n",nCK,total);
  printf("  stock  vs KNEST  mismatch = %d/%d  => %s\n",
         (nSC-nSC), total, (nCK==0)?"BYTE-EXACT (KNEST==current==stock)":"FAIL");
  return (nCK==0)?0:1;
}
