// option-2 stage-C1b: HOST byte-exact harness for the COMPILER-EMITTED q4_0
// -> q4_0x16 PACKER (make_block_q4_0x16 materialization).
//
// Drives the compiler-emitted packer (tcrv_emitc_..._pack_q4_0_to_q4_0x16,
// supplied via #include of the emitted .cpp with the dead setvl/riscv_vector.h
// stripped) against ggml's OWN inlined make_block_q4_0x16, over many (n, NC),
// random plain q4_0 weights. Verdict = EXACT byte equality (memcmp==0); this is
// a PURE byte gather + ^0x88 XOR, no arithmetic, so a single differing byte
// FAILS (NOT a norm/tolerance).
//
// HOST-ONLY: the emitted packer is pure scalar (no RVV intrinsics on the data
// path). The transform is identical on any ISA -- hardware adds nothing.
//
// e2e-REDUNDANT framing: ggml ALREADY packs plain->x16 at model-load. This
// harness proves the compiler CAN PRODUCE the x16 layout it DECLARES (an
// ISOLATED materialization-capability proof), NEVER a kernel/perf/e2e win.
//
// REFERENCE: inline of static make_block_q4_0x16(in, 1) from repack.cpp (the
// ^0x88 bake, blck_size_interleave==1 live branch). Copied verbatim from the
// archived verify_emitted_gemv.cpp oracle.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#define QK4_0 32
#define QK8_0 32

typedef uint16_t ggml_half;

static inline ggml_half float_to_half(float fv) {
    uint32_t x; memcpy(&x, &fv, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffff;
    if (exp <= 0) return (ggml_half)sign;
    if (exp >= 0x1f) return (ggml_half)(sign | 0x7c00);
    uint32_t h = sign | (exp << 10) | (man >> 13);
    if ((man & 0x1000) && ((man & 0x2fff) || (h & 1))) h++;
    return (ggml_half)h;
}

struct block_q4_0    { ggml_half d;     uint8_t qs[QK4_0/2]; };   // 18 bytes
struct block_q4_0x16 { ggml_half d[16]; int8_t  qs[QK8_0*8]; };   // 288 bytes
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 stride must be 18");
static_assert(sizeof(block_q4_0x16) == 288, "block_q4_0x16 stride must be 288");

extern "C" {
// COMPILER-EMITTED PACKER (via the #include'd emitted .cpp). Signature follows
// the runtime_abi_value definition order in input-pack-q4_0.mlir:
//   (size_t nblocks, const uint8_t* src, uint8_t* dst).
void tcrv_emitc_ggml_pack_q4_0_to_q4_0x16_kernel_ggml_pack_q4_0_to_q4_0x16(
    size_t nblocks, const uint8_t* src, uint8_t* dst);
}
#include "emitted-packer.host.cpp"

// Thin adapter: stable (src, dst, nblocks) calling form for the harness.
static void tcrv_pack_q4_0_to_q4_0x16(const block_q4_0* src,
                                      block_q4_0x16* dst, int nblocks) {
    tcrv_emitc_ggml_pack_q4_0_to_q4_0x16_kernel_ggml_pack_q4_0_to_q4_0x16(
        (size_t)nblocks, (const uint8_t*)src, (uint8_t*)dst);
}

// REFERENCE: inline of static make_block_q4_0x16(in, 1) from repack.cpp.
static block_q4_0x16 make_block(const block_q4_0* in) {
    block_q4_0x16 out;
    for (int i = 0; i < 16; i++) out.d[i] = in[i].d;
    const int end = QK4_0 * 8 / 1; // 256
    const uint8_t xor_mask = 0x88;
    for (int i = 0; i < end; ++i) {
        int src_id = i % 16, src_offset = i / 16;
        out.qs[i] = (int8_t)(in[src_id].qs[src_offset] ^ xor_mask);
    }
    return out;
}

static void make_random_q4_0_row(block_q4_0* row, int nblocks, std::mt19937& rng) {
    std::uniform_int_distribution<int> nib(0, 15);
    std::uniform_real_distribution<float> sc(0.005f, 0.05f);
    for (int b = 0; b < nblocks; b++) {
        row[b].d = float_to_half(sc(rng));
        for (int k = 0; k < QK4_0/2; k++) row[b].qs[k] = (uint8_t)(nib(rng) | (nib(rng) << 4));
    }
}

int main(int argc, char** argv) {
    int ns[] = {64, 256, 4096, 11008, 14336};
    const int NN = sizeof(ns)/sizeof(ns[0]);
    int trials = (argc > 1) ? atoi(argv[1]) : 200;
    std::mt19937 rng(20260622);

    int ncs[] = {16, 32, 64, 336};
    const int NNC = sizeof(ncs)/sizeof(ncs[0]);
    bool all_ok = true;
    printf("VERIFY EMITTED q4_0 -> q4_0x16 PACKER (HOST, byte-exact memcmp)  "
           "trials=%d  n in {64,256,4096,11008,14336}\n", trials);
    for (int ci = 0; ci < NNC; ci++) {
        const int NC = ncs[ci];
        const int ngrp = NC / 16;
        long blocks_checked = 0, mismatches = 0, first_n_fail = -1;
        for (int ni = 0; ni < NN; ni++) {
            int n = ns[ni];
            int nb = n / QK4_0;
            for (int t = 0; t < trials; t++) {
                // NC rows x nb blocks of random plain q4_0.
                std::vector<block_q4_0> w(NC * nb);
                for (int r = 0; r < NC; r++) make_random_q4_0_row(&w[r * nb], nb, rng);

                // REFERENCE: group-major repack via ggml's make_block. Output
                // block (g, x) lives at ref[g*nb + x], built from the 16 sources
                // tmp[i] = w[(g*16+i)*nb + x].
                std::vector<block_q4_0x16> ref(ngrp * nb);
                // EMITTED input: the SAME 16-source-block windows laid out
                // contiguously so emitted output block b=(g*nb+x) consumes
                // src[b*16 + 0 .. b*16 + 15].
                std::vector<block_q4_0> src(ngrp * nb * 16);
                for (int g = 0; g < ngrp; g++) {
                    for (int x = 0; x < nb; x++) {
                        int b = g * nb + x;
                        block_q4_0 tmp[16];
                        for (int i = 0; i < 16; i++) {
                            tmp[i] = w[(g * 16 + i) * nb + x];
                            src[b * 16 + i] = tmp[i];
                        }
                        ref[b] = make_block(tmp);
                    }
                }
                std::vector<block_q4_0x16> emitted(ngrp * nb);
                tcrv_pack_q4_0_to_q4_0x16(src.data(), emitted.data(), ngrp * nb);

                size_t bytes = (size_t)ngrp * nb * sizeof(block_q4_0x16);
                blocks_checked += (long)ngrp * nb;
                if (memcmp(emitted.data(), ref.data(), bytes) != 0) {
                    mismatches++;
                    if (first_n_fail < 0) first_n_fail = n;
                }
            }
        }
        bool ok = (mismatches == 0);
        all_ok = all_ok && ok;
        printf("  NC=%-4d (%2d grp)  blocks=%-9ld  mismatch_trials=%-4ld  %s%s\n",
               NC, ngrp, blocks_checked, mismatches, ok ? "PASS" : "FAIL",
               ok ? "" : "  (first failing n shown next)");
        if (!ok) printf("       first failing n = %ld\n", first_n_fail);
    }
    printf("  VERDICT: %s\n", all_ok ?
        "PASS (compiler-emitted PACKER byte-identical to ggml make_block_q4_0x16 "
        "across NC=16/32/64/336, memcmp==0) -- ISOLATED materialization proof, "
        "e2e-REDUNDANT, NOT a kernel/perf/e2e win" : "FAIL");
    return all_ok ? 0 : 1;
}
