/* structcnt: run a child command under perf_event_open HW counters
 * (INSTRUCTIONS + CPU_CYCLES), inherit across threads. Prints totals.
 * Usage: structcnt -- <cmd> [args...]
 * Only these two HW events are used (k1 X60 PMU: cache-misses unsupported). */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>

static long pe_open(struct perf_event_attr *a, pid_t pid, int cpu, int grp, unsigned long fl){
  return syscall(__NR_perf_event_open, a, pid, cpu, grp, fl);
}
static int mkcnt(uint64_t cfg, pid_t pid){
  struct perf_event_attr a; memset(&a,0,sizeof(a));
  a.type=PERF_TYPE_HARDWARE; a.size=sizeof(a); a.config=cfg;
  a.disabled=1; a.inherit=1; a.exclude_kernel=1; a.exclude_hv=1;
  long fd=pe_open(&a,pid,-1,-1,0);
  if(fd<0){ fprintf(stderr,"perf_open cfg=%llu errno=%d %s\n",(unsigned long long)cfg,errno,strerror(errno)); }
  return (int)fd;
}
int main(int argc,char**argv){
  int ai=1;
  while(ai<argc && strcmp(argv[ai],"--")!=0) ai++;
  if(ai>=argc-0 || ai==argc){ fprintf(stderr,"usage: structcnt -- cmd args\n"); return 2; }
  ai++; /* first cmd arg */
  int p[2];
  if(pipe(p)){ perror("pipe"); return 3; }
  pid_t pid=fork();
  if(pid<0){ perror("fork"); return 3; }
  if(pid==0){
    close(p[1]); char b; read(p[0],&b,1); close(p[0]);
    execvp(argv[ai],&argv[ai]);
    perror("execvp"); _exit(127);
  }
  close(p[0]);
  int fi=mkcnt(PERF_COUNT_HW_INSTRUCTIONS,pid);
  int fc=mkcnt(PERF_COUNT_HW_CPU_CYCLES,pid);
  if(fi<0||fc<0){ fprintf(stderr,"counter open failed\n"); }
  if(fi>=0){ ioctl(fi,PERF_EVENT_IOC_RESET,0); ioctl(fi,PERF_EVENT_IOC_ENABLE,0); }
  if(fc>=0){ ioctl(fc,PERF_EVENT_IOC_RESET,0); ioctl(fc,PERF_EVENT_IOC_ENABLE,0); }
  write(p[1],"g",1); close(p[1]);
  int st; waitpid(pid,&st,0);
  if(fi>=0) ioctl(fi,PERF_EVENT_IOC_DISABLE,0);
  if(fc>=0) ioctl(fc,PERF_EVENT_IOC_DISABLE,0);
  uint64_t vi=0,vc=0;
  if(fi>=0) if(read(fi,&vi,8)!=8) vi=0;
  if(fc>=0) if(read(fc,&vc,8)!=8) vc=0;
  fprintf(stderr,"STRUCTCNT instructions=%llu cycles=%llu exit=%d\n",
          (unsigned long long)vi,(unsigned long long)vc,WEXITSTATUS(st));
  return WEXITSTATUS(st);
}
