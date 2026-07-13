#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>

int main(){
  struct {const char*n; uint32_t t; uint64_t c;} evs[] = {
    {"HW_CACHE_MISSES", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES},
    {"HW_CACHE_REFERENCES", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES},
    {"HW_INSTRUCTIONS", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
    {"HW_CPU_CYCLES", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
    {"CACHE_LL_READ_MISS", PERF_TYPE_HW_CACHE,
       (PERF_COUNT_HW_CACHE_LL) | (PERF_COUNT_HW_CACHE_OP_READ<<8) | (PERF_COUNT_HW_CACHE_RESULT_MISS<<16)},
    {"CACHE_L1D_READ_MISS", PERF_TYPE_HW_CACHE,
       (PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_READ<<8) | (PERF_COUNT_HW_CACHE_RESULT_MISS<<16)},
  };
  int n=sizeof(evs)/sizeof(evs[0]);
  for(int i=0;i<n;i++){
    struct perf_event_attr a; memset(&a,0,sizeof(a));
    a.type=evs[i].t; a.size=sizeof(a); a.config=evs[i].c;
    a.disabled=1; a.exclude_kernel=1; a.exclude_hv=1;
    long fd=syscall(__NR_perf_event_open,&a,0,-1,-1,0);
    if(fd<0){ printf("%-22s UNSUPPORTED (open errno=%d %s)\n",evs[i].n,errno,strerror(errno)); continue; }
    ioctl(fd,PERF_EVENT_IOC_RESET,0);
    ioctl(fd,PERF_EVENT_IOC_ENABLE,0);
    size_t sz=256UL*1024*1024;
    volatile char* buf=malloc(sz);
    long acc=0;
    for(size_t k=0;k<sz;k+=64){ buf[k]=(char)(k^acc); acc+=buf[k]; }
    ioctl(fd,PERF_EVENT_IOC_DISABLE,0);
    uint64_t val=0; ssize_t r=read(fd,&val,sizeof(val));
    printf("%-22s val=%llu (read=%zd) acc=%ld\n",evs[i].n,(unsigned long long)val,r,acc);
    free((void*)buf);
    close(fd);
  }
  return 0;
}
