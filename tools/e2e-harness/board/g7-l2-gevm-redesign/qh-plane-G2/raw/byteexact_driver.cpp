// G7 L2 qh-plane G2 byte-exact gate: OLD emitter (per-lane expand, independent
// certified oracle) vs NEW emitter (REDESIGN-B native-mask), fed IDENTICAL random
// repacked bytes. Same layout, so any output bit difference == a decode error.
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
extern "C" void kern_old(size_t,float*,size_t,const uint8_t*,size_t,const uint8_t*,size_t,int32_t);
extern "C" void kern_new(size_t,float*,size_t,const uint8_t*,size_t,const uint8_t*,size_t,int32_t);
int main(int argc,char**argv){
  int nb     = argc>1?atoi(argv[1]):8;      // blocks (K/32) per 16-row group
  int trials = argc>2?atoi(argv[2]):5000;
  unsigned seed = argc>3?(unsigned)atoi(argv[3]):12345u;
  srand(seed);
  size_t wbytes = (size_t)nb*352;           // one 16-row repacked group
  size_t abytes = (size_t)nb*34;            // nb standard block_q8_0
  uint8_t* w = (uint8_t*)malloc(wbytes);
  uint8_t* a = (uint8_t*)malloc(abytes);
  float so[16], sn[16];
  long mism=0, checked=0;
  for(int t=0;t<trials;t++){
    for(size_t i=0;i<wbytes;i++) w[i]=(uint8_t)(rand()&0xff);
    for(size_t i=0;i<abytes;i++) a[i]=(uint8_t)(rand()&0xff);
    memset(so,0,sizeof so); memset(sn,0,sizeof sn);
    kern_old((size_t)32*nb, so, 16, w, 0, a, 0, 0);
    kern_new((size_t)32*nb, sn, 16, w, 0, a, 0, 0);
    for(int r=0;r<16;r++){ checked++;
      uint32_t bo,bn; memcpy(&bo,&so[r],4); memcpy(&bn,&sn[r],4);
      if(bo!=bn){ if(mism<10) printf("MISMATCH t=%d r=%d old=%08x(%g) new=%08x(%g)\n",t,r,bo,so[r],bn,sn[r]); mism++; }
    }
  }
  printf("byte-exact q5_0 GEVM OLD-vs-NEW: %ld mismatches / %ld outputs (nb=%d trials=%d seed=%u) => %s\n",
         mism,checked,nb,trials,seed, mism==0?"BYTE-EXACT PASS":"FAIL");
  free(w); free(a);
  return mism?1:0;
}
