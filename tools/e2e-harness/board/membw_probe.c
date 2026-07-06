/* membw_probe.c -- minimal single-thread sustained READ-bandwidth probe.
 * Run 4 copies pinned to the measurement cores (taskset -c 8..11) concurrently
 * and SUM the reported GB/s to get the aggregate achievable read bandwidth for
 * the same 4-core set the decode e2e uses. Read-BW is the relevant ceiling for
 * weight-streaming decode. Array (default 256 MiB) >> LLC so it stays in DRAM.
 *
 * argv: [mib] [iters]   prints: "MEMBW_GBs=<x> bytes=<b> secs=<s>"
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

int main(int argc, char** argv){
    size_t mib   = (argc>1)? (size_t)atol(argv[1]) : 256;
    int    iters = (argc>2)? atoi(argv[2]) : 40;
    size_t n = mib*1024*1024/sizeof(uint64_t);
    uint64_t* a = (uint64_t*)malloc(n*sizeof(uint64_t));
    if(!a){ fprintf(stderr,"alloc fail\n"); return 1; }
    for(size_t i=0;i<n;i++) a[i]=i*2654435761u+1;   /* touch/fault-in */
    struct timespec t0,t1;
    volatile uint64_t sink=0;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    for(int it=0; it<iters; it++){
        uint64_t s0=0,s1=0,s2=0,s3=0;                /* 4-way unroll to expose BW not latency */
        for(size_t i=0;i+4<=n;i+=4){ s0+=a[i]; s1+=a[i+1]; s2+=a[i+2]; s3+=a[i+3]; }
        sink += s0+s1+s2+s3;
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double secs = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    double bytes = (double)n*sizeof(uint64_t)*iters;
    printf("MEMBW_GBs=%.3f bytes=%.0f secs=%.4f sink=%llu\n",
           bytes/secs/1e9, bytes, secs, (unsigned long long)sink);
    free(a);
    return 0;
}
