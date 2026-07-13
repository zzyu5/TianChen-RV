#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static inline void vmadot_mac_kloop(const int8_t*A,const int8_t*B,long kt,int32_t*frag){
  __asm__ volatile("vsetvli t0,zero,e8,m1,ta,ma\n\tvmv.v.i v2,0\n\tvmv.v.i v3,0\n\t"
    "mv t2,%[kt]\n\tmv t3,%[pa]\n\tmv t4,%[pb]\n\t1:\n\tvle8.v v0,(t3)\n\tvle8.v v1,(t4)\n\t"
    "vmadot v2,v0,v1\n\taddi t3,t3,32\n\taddi t4,t4,32\n\taddi t2,t2,-1\n\tbnez t2,1b\n\t"
    "vsetvli t0,zero,e32,m1,ta,ma\n\tvse32.v v2,(%[pf])\n\taddi t5,%[pf],32\n\tvse32.v v3,(t5)\n\t"
    ::[pa]"r"(A),[pb]"r"(B),[kt]"r"(kt),[pf]"r"(frag)
    :"t0","t2","t3","t4","t5","v0","v1","v2","v3","memory");
}
static inline void vmadot_mac_kloop_w2(const int8_t*A,const int8_t*B0,long bstride,long kt,int32_t*frag){
  const int8_t*B1=B0+bstride;
  __asm__ volatile("vsetvli t0,zero,e8,m1,ta,ma\n\tvmv.v.i v2,0\n\tvmv.v.i v3,0\n\tvmv.v.i v4,0\n\tvmv.v.i v5,0\n\t"
    "mv t2,%[kt]\n\tmv t3,%[pa]\n\tmv t4,%[pb0]\n\tmv t6,%[pb1]\n\t1:\n\t"
    "vle8.v v0,(t3)\n\tvle8.v v1,(t4)\n\tvle8.v v6,(t6)\n\tvmadot v2,v0,v1\n\tvmadot v4,v0,v6\n\t"
    "addi t3,t3,32\n\taddi t4,t4,32\n\taddi t6,t6,32\n\taddi t2,t2,-1\n\tbnez t2,1b\n\t"
    "vsetvli t0,zero,e32,m1,ta,ma\n\tvse32.v v2,(%[pf])\n\taddi t5,%[pf],32\n\tvse32.v v3,(t5)\n\t"
    "addi t5,%[pf],64\n\tvse32.v v4,(t5)\n\taddi t5,%[pf],96\n\tvse32.v v5,(t5)\n\t"
    ::[pa]"r"(A),[pb0]"r"(B0),[pb1]"r"(B1),[kt]"r"(kt),[pf]"r"(frag)
    :"t0","t2","t3","t4","t5","t6","v0","v1","v2","v3","v4","v5","v6","memory");
}
static inline void vmadot_mac_kloop_w4(const int8_t*A,const int8_t*B0,long bstride,long kt,int32_t*frag){
  const int8_t*B1=B0+bstride,*B2=B0+2*bstride,*B3=B0+3*bstride;
  __asm__ volatile("vsetvli t0,zero,e8,m1,ta,ma\n\tvmv.v.i v2,0\n\tvmv.v.i v3,0\n\tvmv.v.i v4,0\n\tvmv.v.i v5,0\n\t"
    "vmv.v.i v10,0\n\tvmv.v.i v11,0\n\tvmv.v.i v12,0\n\tvmv.v.i v13,0\n\t"
    "mv t2,%[kt]\n\tmv t1,%[pa]\n\tmv t3,%[pb0]\n\tmv t4,%[pb1]\n\tmv t5,%[pb2]\n\tmv t6,%[pb3]\n\t1:\n\t"
    "vle8.v v0,(t1)\n\tvle8.v v1,(t3)\n\tvle8.v v6,(t4)\n\tvle8.v v7,(t5)\n\tvle8.v v8,(t6)\n\t"
    "vmadot v2,v0,v1\n\tvmadot v4,v0,v6\n\tvmadot v10,v0,v7\n\tvmadot v12,v0,v8\n\t"
    "addi t1,t1,32\n\taddi t3,t3,32\n\taddi t4,t4,32\n\taddi t5,t5,32\n\taddi t6,t6,32\n\taddi t2,t2,-1\n\tbnez t2,1b\n\t"
    "vsetvli t0,zero,e32,m1,ta,ma\n\tmv t1,%[pf]\n\t"
    "vse32.v v2,(t1)\n\taddi t1,t1,32\n\tvse32.v v3,(t1)\n\taddi t1,t1,32\n\t"
    "vse32.v v4,(t1)\n\taddi t1,t1,32\n\tvse32.v v5,(t1)\n\taddi t1,t1,32\n\t"
    "vse32.v v10,(t1)\n\taddi t1,t1,32\n\tvse32.v v11,(t1)\n\taddi t1,t1,32\n\t"
    "vse32.v v12,(t1)\n\taddi t1,t1,32\n\tvse32.v v13,(t1)\n\t"
    ::[pa]"r"(A),[pb0]"r"(B0),[pb1]"r"(B1),[pb2]"r"(B2),[pb3]"r"(B3),[kt]"r"(kt),[pf]"r"(frag)
    :"t0","t1","t2","t3","t4","t5","t6","v0","v1","v2","v3","v4","v5","v6","v7","v8","v10","v11","v12","v13","memory");
}
int main(){
  srand(20260713); const long kt=4;
  int8_t A[128]; int8_t B[512];
  for(int i=0;i<128;i++)A[i]=(int8_t)(rand()%256-128);
  for(int i=0;i<512;i++)B[i]=(int8_t)(rand()%256-128);
  int32_t base[64]; // 4 tiles
  for(int w=0;w<4;w++) vmadot_mac_kloop(A,B+w*128,kt,base+w*16);
  int32_t w2[64];
  vmadot_mac_kloop_w2(A,B+0*128,128,kt,w2+0);   // tiles 0,1
  vmadot_mac_kloop_w2(A,B+2*128,128,kt,w2+32);  // tiles 2,3
  int32_t w4[64];
  vmadot_mac_kloop_w4(A,B+0,128,kt,w4);
  int d2=memcmp(base,w2,sizeof base), d4=memcmp(base,w4,sizeof base);
  printf("w2_vs_base memcmp=%d  w4_vs_base memcmp=%d\n",d2,d4);
  // show a couple values
  printf("base[0..3]=%d %d %d %d  sum|C|=", base[0],base[1],base[2],base[3]);
  long s=0; for(int i=0;i<64;i++) s+= base[i]<0?-base[i]:base[i]; printf("%ld\n",s);
  if(d2==0&&d4==0){printf("WIDE-LEAF BYTE-EXACT PASS (w2==w4==baseline int32)\n");return 0;}
  printf("WIDE-LEAF FAIL\n"); return 1;
}
