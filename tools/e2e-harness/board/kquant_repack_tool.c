/* kquant_repack_tool.c -- [SEL-1 T4b / M1] standalone OFFLINE K-quant weight repacker CLI.
 * Thin driver over the general kquant_repacker.h: takes ANY q4_K/q5_K tensor shape and produces the
 * repacked block_q{4,5}_Kx16 buffer (+ optionally the interleaved q8_K activation), writes it to a
 * file, and prints the exact byte geometry + a deterministic FNV-1a fingerprint. This is the
 * generalization of the M0 one-shot pack: same verified encoding, arbitrary shape, no board kernel
 * or ggml needed (pure host repack -- runs anywhere, links nothing).
 *
 * Real gguf tensors are not on this board yet (M3 resource gap), so the tool synthesizes a
 * deterministic pseudo-random tensor of the requested shape (or, with --in, repacks a raw
 * [nc][nb]-block file). The numeric correctness of the produced layout is certified separately by
 * kquant_repack_cert.c against stock ggml. This tool proves the SHAPE generality + determinism.
 *
 * Build: clang-17/g++ -O2 -x c++ kquant_repack_tool.c -o kquant_repack_tool   (no libs)
 * Usage:  kquant_repack_tool <q4_K|q5_K> <nc(%16)> <K(%256)> [seed] [out.x16] [--in raw.blocks]
 *         kquant_repack_tool q8_K <nr(%4)> <K(%256)> [seed] [out.x4]     (activation interleave)
 */
#include "kquant_repacker.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint64_t RNG;
static uint32_t xr(void){ RNG^=RNG<<13; RNG^=RNG>>7; RNG^=RNG<<17; return (uint32_t)(RNG>>32); }
static uint64_t fnv1a(const uint8_t* p, size_t n){
    uint64_t h=1469598103934665603ull;
    for(size_t i=0;i<n;++i){ h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
static void randfill(uint8_t* p, size_t n){ for(size_t i=0;i<n;++i) p[i]=(uint8_t)(xr()&0xFF); }

int main(int argc, char** argv){
    if(argc<4){
        fprintf(stderr,"usage: %s <q4_K|q5_K|q8_K> <nc|nr> <K> [seed] [out] [--in raw.blocks]\n",argv[0]);
        return 2;
    }
    const char* fmt = argv[1];
    int dim = atoi(argv[2]);          /* nc for weights, nr for activation */
    int K   = atoi(argv[3]);
    unsigned seed = argc>4 ? (unsigned)strtoul(argv[4],0,0) : 12345u;
    const char* out = (argc>5 && strncmp(argv[5],"--",2)) ? argv[5] : nullptr;
    const char* inf = nullptr;
    for(int i=4;i<argc-1;++i) if(!strcmp(argv[i],"--in")) inf=argv[i+1];
    if(K%KQR_QK_K){ fprintf(stderr,"K must be %% %d\n",KQR_QK_K); return 2; }
    int nb=K/KQR_QK_K;
    RNG=((uint64_t)seed<<1)|1ull;

    size_t inBytes=0, outBytes=0, stride=0; int groups=0;
    std::vector<uint8_t> in, dst;

    auto load_or_gen = [&](size_t nblk, size_t blkSz){
        inBytes = nblk*blkSz;
        in.resize(inBytes);
        if(inf){ FILE* f=fopen(inf,"rb"); if(!f){perror("open --in");exit(3);}
                 if(fread(in.data(),1,inBytes,f)!=inBytes){fprintf(stderr,"short --in read\n");exit(3);} fclose(f); }
        else randfill(in.data(), inBytes);
    };

    if(!strcmp(fmt,"q4_K")){
        if(dim%16){ fprintf(stderr,"nc must be %% 16\n"); return 2; }
        load_or_gen((size_t)dim*nb, sizeof(kqr_block_q4_K));
        outBytes=kqr_bytes_q4_K(dim,nb); stride=KQR_STRIDE_Q4Kx16; groups=dim/16;
        dst.resize(outBytes);
        kqr_repack_q4_K(dst.data(), (const kqr_block_q4_K*)in.data(), dim, nb);
    } else if(!strcmp(fmt,"q5_K")){
        if(dim%16){ fprintf(stderr,"nc must be %% 16\n"); return 2; }
        load_or_gen((size_t)dim*nb, sizeof(kqr_block_q5_K));
        outBytes=kqr_bytes_q5_K(dim,nb); stride=KQR_STRIDE_Q5Kx16; groups=dim/16;
        dst.resize(outBytes);
        kqr_repack_q5_K(dst.data(), (const kqr_block_q5_K*)in.data(), dim, nb);
    } else if(!strcmp(fmt,"q8_K")){
        if(dim%4){ fprintf(stderr,"nr must be %% 4\n"); return 2; }
        load_or_gen((size_t)dim*nb, sizeof(kqr_block_q8_K));
        outBytes=kqr_bytes_q8_K(dim,nb); stride=KQR_STRIDE_Q8Kx4; groups=dim/4;
        dst.resize(outBytes);
        kqr_interleave_q8_K(dst.data(), (const kqr_block_q8_K*)in.data(), dim, nb);
    } else { fprintf(stderr,"unknown fmt %s\n",fmt); return 2; }

    /* determinism self-check: repack again into a fresh buffer, require identical bytes */
    std::vector<uint8_t> dst2(outBytes);
    if(!strcmp(fmt,"q4_K")) kqr_repack_q4_K(dst2.data(),(const kqr_block_q4_K*)in.data(),dim,nb);
    else if(!strcmp(fmt,"q5_K")) kqr_repack_q5_K(dst2.data(),(const kqr_block_q5_K*)in.data(),dim,nb);
    else kqr_interleave_q8_K(dst2.data(),(const kqr_block_q8_K*)in.data(),dim,nb);
    int deterministic = (memcmp(dst.data(),dst2.data(),outBytes)==0);

    printf("fmt=%s  dim=%d  K=%d  nb=%d  groups=%d  group_stride=%zu\n", fmt,dim,K,nb,groups,stride);
    printf("in_bytes=%zu  out_bytes=%zu  expand=%.4gx  fnv1a=0x%016llx  deterministic=%s\n",
           inBytes,outBytes,(double)outBytes/(double)inBytes,(unsigned long long)fnv1a(dst.data(),outBytes),
           deterministic?"YES":"NO");
    if(out){ FILE* f=fopen(out,"wb"); if(!f){perror("open out");return 4;}
             if(fwrite(dst.data(),1,outBytes,f)!=outBytes){fprintf(stderr,"short write\n");fclose(f);return 4;}
             fclose(f); printf("wrote %zu bytes -> %s\n",outBytes,out); }
    return deterministic?0:5;
}
