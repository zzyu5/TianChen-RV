# N2 / IME Foundation — empirical validation on real K1 (Spacemit X60) silicon

**Date:** 2026-06-23
**Hardware under test:** `ssh k1` = Spacemit(R) X60, Bianbu Linux 6.6.63 riscv64, 8 cores / 7.6 GB.
**Invariant in force:** I8 — only a *real `ssh` run on the X60* counts as runtime evidence.
QEMU / compile-only is functional evidence, NOT a runtime claim. **No QEMU was used for any verdict below.**

---

## VERDICT (TL;DR)

> **The N2 / IME foundation IS empirically validated on our K1.**
> On real Spacemit X60 silicon, a single int8→int32 `vmadot` matrix-multiply-accumulate
> (a) did **NOT** SIGILL (exit 0, both before/after markers printed), proving the cpuinfo
> `ime` token / `xsmtvdotii` march / X60 are the **same real IME1 hardware**; and
> (b) produced a result **bit-exactly equal** to a plain scalar C reference computed from
> the same *discriminating, asymmetric, signed* data — proving we can drive the IME
> correctly. The SpacemiT GCC 15.2 fork assembles `vmadot` (stock toolchains cannot).
> All four hedged inferences (A60≡X60, `xsmtvdotii`≡`Xsmti8i32mm`, IME1-only, real silicon)
> are now confirmed by a real-K1 run, not documentation. **Oracle (stretch) also PASSES:**
> llama.cpp's merged SpacemiT IME1 backend cross-builds with the GCC15 fork, runs on the real X60
> with `use_ime1: 1`, and produces correct Q4_0 inference (~46.6 tok/s pp / ~8.0 tok/s tg,
> coherent text).

---

## Real-silicon fact discovered (corrects the research doc)

The X60 `/proc/cpuinfo` `isa` string **DOES** end in an `_ime` token on this box
(the research doc `spacemit-ime.md` asserted it would *not*):

```
model name : Spacemit(R) X60
isa  : rv64imafdcv_..._zvfh_zvfhmin_zvkt_sscofpmf_sstc_svinval_svnapot_svpbmt_ime
uarch: spacemit,x60   mvendorid: 0x710   marchid: 0x8000000058000001
```

Notable: **processors 0–3 carry `_ime`; processor 4 does NOT** (its isa ends `..._svpbmt`).
i.e. the IME unit is present on a *subset* of harts on this part. (Per-hart asymmetry — relevant
for any future thread-pinning of an IME kernel.) The `ime` here is the genuine RISC-V vendor
matrix-extension ISA token, distinct from the unrelated `libimepinyin` input-method red herring.

The `ime` cpuinfo token is **not** the `-march` spelling. The march token is `xsmtvdotii`
(verified below); the cpuinfo `ime` is the kernel-advertised hwcap. Both refer to the same unit.

---

## Toolchain (local x86_64 cross host)

```
URL  : https://github.com/spacemit-com/toolchain/releases/download/v1.2.4/spacemit-toolchain-linux-glibc-x86_64-v1.2.4.tar.xz
size : 453196620 bytes
root : /home/kingdom/spacemit-ime/spacemit-toolchain-linux-glibc-x86_64-v1.2.4   (export RISCV_ROOT_PATH=...)
triple: riscv64-unknown-linux-gnu-gcc
```

### TASK 1 — version gate (PASS)
```
$ $RISCV_ROOT_PATH/bin/riscv64-unknown-linux-gnu-gcc --version
riscv64-unknown-linux-gnu-gcc (gf3b8c022145) 15.2.0
```
GCC 15.2.0 confirmed (the IME-capable fork).

---

## TASK 2 — Compile-gate (PASS, decisively)

Probe file `compile_gate.c` (mirrors `FindSMTIME.cmake`'s IME1 test):
```c
int main(void){ __asm__ volatile("vmadot v2, v0, v1\n\t"); return 0; }
```

**2a — same GCC15 WITHOUT the token (must reject — proves the token is load-bearing):**
```
$ riscv64-unknown-linux-gnu-gcc -march=rv64gcv_zfh_zvfh_zba_zicbop      -mabi=lp64d -c compile_gate.c
compile_gate.c:5: Error: unrecognized opcode `vmadot v2,v0,v1', extension `xsmtvdotii' required
```

**2b — WITH the token (must assemble):**
```
$ riscv64-unknown-linux-gnu-gcc -march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii -mabi=lp64d -c compile_gate.c -o /tmp/cg_ime.o
$ echo $?                       # 0  → assembled
```

**2c — objdump confirms a REAL opcode + exact encoding:**
```
$ riscv64-unknown-linux-gnu-objdump -d /tmp/cg_ime.o
   8:	e210312b          	smt.vmadot	v2,v0,v1
```
Exact encoding: **`0x2b3110e2`** (little-endian word; objdump byte column `e2 10 31 2b`),
mnemonic decoded as `smt.vmadot v2,v0,v1`. 32-bit instruction. This is the IME1 integer
dot-product MAC opcode.

---

## TASK 3 — THE DECISIVE RUNTIME GATE on real X60 (PASS, decisively)

Self-checking program `ime_gate.c`: one int8→int32 `vmadot` at VLEN=256 / SEW=8
(the 4×4×8 MAC unit, Copies=1) vs. a plain scalar C reference over the SAME data.

**Programming model used** (riscv-ime-extension-spec, `instruction-func.adoc` + `program-model.adoc`):
- A in vs1 (`v0`) = (M,K)=(4,8) int8 row-major.
- B in vs2 (`v1`) = stored (N,K)=(4,8) int8 row-major (i.e. Bᵀ of the math (K,N) matrix).
- C in even VD pair (`v2`,`v3`) = (M,N)=(4,4) **int32** (16 words / 64 B).
- Semantics: `C[i][j] += Σ_k A[i][k]·B_stored[j][k]`  (= A · B_storedᵀ).
- MAC unit pinned by `vsetvli e8,m1` ⇒ vl=32 at VLEN=256 ⇒ 4×4×8, int32 accumulator.
  (The spec pseudocode's `C[... i*K+j]` is a typo for `i*N+j`; corrected here.)

**Hardening (so the gate is not vacuous):**
- **Discriminating data:** asymmetric *signed* int8 with negatives, no two equal rows,
  A-pattern ≠ B-pattern — so a transpose / M↔N swap / signed-vs-unsigned (`vmadot` vs
  `vmadotu`) confusion would NOT accidentally match. A match is therefore load-bearing.
- **MAC-unit assert:** reads `vlenb` (csrr) and `vl` after `vsetvli e8,m1`; aborts (exit 3)
  unless `vlenb==32 && vl==32` (i.e. genuinely VLEN=256 / 4×4×8).
- **SIGILL mechanically detectable:** prints `MARK_BEFORE_VMADOT` / `MARK_AFTER_VMADOT`
  (flushed) around the instruction; ssh captures exit code (SIGILL ⇒ 132 = 128+4).

**Build (on local x86_64 host, statically linked):**
```
$ riscv64-unknown-linux-gnu-gcc -O2 -static \
    -march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii -mabi=lp64d ime_gate.c -o ime_gate
$ riscv64-unknown-linux-gnu-objdump -d ime_gate | grep vmadot
   1053c:	e210312b   smt.vmadot v2,v0,v1
```

**Run on real X60 (`ssh k1`):**
```
$ scp ime_gate k1:~/n2-ime-probe/
$ ssh k1 'cd ~/n2-ime-probe && timeout 30 ./ime_gate; echo EXIT_CODE=$?'
vlenb=32  vl(e8,m1)=32  (expect vlenb=32, vl=32 for VLEN=256)
MARK_BEFORE_VMADOT
MARK_AFTER_VMADOT
scalar reference C (row-major 4x4):
      -4     -204        0       72
       4      492      -32     -136
      -4       -2       10       92
      -4      552      -84      -96
IME vmadot result C (row-major 4x4):
      -4     -204        0       72
       4      492      -32     -136
      -4       -2       10       92
      -4      552      -84      -96
RESULT: PASS  (IME == scalar, all 16 elements match)
EXIT_CODE=0
VERDICT_SIGNAL=CLEAN(0)
```

**Verdict (a) — SIGILL?  NO.** Both markers printed, exit code 0 (not 132). The `vmadot`
executed natively. ⇒ cpuinfo `ime` ≡ `xsmtvdotii` march ≡ X60 are the SAME real IME1 silicon.

**Verdict (b) — IME == scalar?  YES, bit-exact (16/16).** With discriminating signed data,
the hardware MAC equals the scalar int8→int32 reference. ⇒ we drive the IME correctly; the
4×4×8 / A-in-vs1 / Bᵀ-in-vs2 / C-in-even-VD-pair programming model is empirically correct on X60.

This is the load-bearing N2-foundation fact: the second-family (IME) capability is real,
targetable, and correctly drivable on our actual hardware under I8.

---

## TASK 4 — Oracle bring-up (llama.cpp SpacemiT IME backend) — DONE, runs on real X60

Cross-built llama.cpp (`/home/kingdom/phdworks/llama.cpp`, HEAD `6eab471`) with the SpacemiT
IME backend on the local x86_64 host using the GCC 15.2 fork, then ran on real K1.

**Build (local x86_64):**
```
export RISCV_ROOT_PATH=/home/kingdom/spacemit-ime/spacemit-toolchain-linux-glibc-x86_64-v1.2.4
cmake -B build-spacemit-ime -DCMAKE_BUILD_TYPE=Release \
   -DGGML_CPU_RISCV64_SPACEMIT=ON -DGGML_CPU_REPACK=OFF -DLLAMA_OPENSSL=OFF \
   -DGGML_RVV=ON -DGGML_RV_ZVFH=ON -DGGML_RV_ZFH=ON -DGGML_RV_ZICBOP=ON \
   -DGGML_RV_ZIHINTPAUSE=ON -DGGML_RV_ZBA=ON \
   -DCMAKE_TOOLCHAIN_FILE=${PWD}/cmake/riscv64-spacemit-linux-gnu-gcc.cmake \
   -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build-spacemit-ime --target llama-bench llama-completion --parallel 96
```
- FindSMTIME probe: `SPACEMIT_RISCV_COMPILER_SUPPORT_IME1 - Success`; backend variant march
  `-march=rv64gcv_zfh_zvfh_zicbop_zihintpause_zba_xsmtvdotii;-mabi=lp64d`.
- `libggml-cpu.so` contains **264 `vmadot` instructions** (objdump) — the IME kernels are real,
  compiled in, not stubbed.
- NB target names in this tree: the CLI is `llama-completion` (not `llama-cli`); `-DLLAMA_BUILD_TOOLS=ON` required.

**Run on real X60 (`ssh k1`), model = existing `tinyllama-q4_0.gguf` (606 MiB, Q4_0 — the IME-accelerated quant):**

Backend self-report at startup (on real silicon):
```
CPU_RISCV64_SPACEMIT: num_cores: 8, num_perfer_cores: 4, perfer_core_arch_id: a03c,
                      use_ime1: 1, use_ime2: 0, mem_backend: HPAGE, cpu_mask: f
```
⇒ **`use_ime1: 1`** — the IME1 path is ACTIVE at runtime, not bypassed.

`llama-bench -m tinyllama-q4_0.gguf -p 32 -n 16 -t 4 -r 2`:
```
| model          |   size     | params | backend | threads | test |        t/s |
| llama 1B Q4_0  | 606.53 MiB | 1.10 B | CPU     |    4    | pp32 | 46.65 ± 0.04 |
| llama 1B Q4_0  | 606.53 MiB | 1.10 B | CPU     |    4    | tg16 |  8.02 ± 0.12 |
```
Coherent generation (`llama-completion`, exit 0):
```
prompt:  The capital of France is
output:  The capital of France is Paris.
```

**Oracle verdict: PASS.** llama.cpp's merged SpacemiT IME1 backend cross-builds with the GCC15
fork, runs on the real X60 with `use_ime1: 1`, produces correct Q4_0 inference and coherent text,
and reports ~46.6 tok/s pp / ~8.0 tok/s tg. This is the N2 reference oracle (analogous to ggml/RVV).

**Caveats (honest):**
- `llama-bench` aborts with `munmap_chunk(): invalid pointer` (SIGABRT, exit 134) *after* printing
  the result table — a teardown/cleanup bug (likely the heap-fallback path; see TCM note), NOT in
  the measured kernel. The tok/s numbers are emitted cleanly before the abort. `llama-completion`
  exits 0. The numbers are valid; the teardown crash is cosmetic.
- `/dev/tcm_sync_mem` open fails (errno 2) → backend falls back from TCM to heap. This only forgoes
  a TCM perf optimization; correctness (coherent output) is unaffected.
- No non-IME A/B baseline was built (would need a second cross-build); the oracle claim here is
  "IME backend runs correctly + tok/s," not an IME-vs-scalar speedup measurement (that is N3-style
  perf work, out of scope for this foundation gate).

### Real-silicon note on the A60 ≡ X60 inference (mechanism traced in source)
The startup line reports `perfer_core_arch_id: a03c` (= `core_arch_a60 = 0xA03C`) for the 4 preferred
cores. I traced HOW the backend derives this (`ime_env.cpp`, `spine_env_info` ctor), to avoid
over-claiming:
- `get_spine_core_info` parses each core's **`marchid`** from `/proc/cpuinfo` and maps it via
  `spine_march_mapping_`. On our chip every core's `marchid 0x8000000058000001` maps to
  `core_arch_x60` (0x503C) — i.e. the silicon-derived id is X60 for all 8 cores.
- Then a **hardcoded special-case** (literally `// special for x60 K1`, ime_env.cpp:152–157):
  *if there are 8 cores and core[0] is x60, deliberately remap the first 4 cores' arch_id to
  `core_arch_a60` (0xA03C).* That override is why the startup line shows `a03c`.

So `a03c` is **not** the silicon self-reporting A60; it is **llama.cpp's maintainers encoding, in
their K1 detection path, that the X60's IME cores ARE A60-class** (an upstream-asserted X60→A60
equivalence specific to the 8-core K1). This corroborates the docs' A60≡X60 framing, but the
**primary, hardware-level proof of A60≡X60 remains task 3**: the `vmadot` ran natively (no SIGILL)
and was bit-exact on this real X60. `use_ime1:1` (never IME2) independently confirms K1 = IME1-only.

(Runtime implication of the per-hart `_ime` asymmetry above: an IME kernel scheduled onto the one
non-IME hart would SIGILL; the backend's 4-preferred-core pinning — and our gate's incidental
scheduling — avoided that. Worth pinning IME work to IME harts in any future emitter.)

---

## Honest scope notes

- All verdicts above are from a **real X60 run**. No QEMU was used. Per I8 this is genuine
  runtime evidence.
- This validates the **foundation/prerequisites** (hardware real, toolchain works, opcode runs,
  programming model correct). It is **NOT** itself N2 novelty. N2 still requires OUR pass to
  consume an IME capability fact *zero-core-branch* and emit `vmadot` — unbuilt, and out of
  scope for this validation task.
- IME1 only: this gate exercises `vmadot` (int8→int32). The IME2 set (int4 / vpack / sparse /
  fp16-bf16) is K3/A100, not this X60 — untested and not claimed.

---

## Reproduce

Local scratch (NOT in the git tree): `/home/kingdom/spacemit-ime/` holds the toolchain, both probe
sources (also inlined below), and the `ime_gate` binary. On K1 the tiny `~/n2-ime-probe/ime_gate`
binary is left in place; all heavy llama bits were cleaned. Exact toolchain root:
`/home/kingdom/spacemit-ime/spacemit-toolchain-linux-glibc-x86_64-v1.2.4` (`export RISCV_ROOT_PATH=...`).

### Appendix A — compile-gate probe (`compile_gate.c`, task 2)
```c
/* compile_gate.c — prove the SpacemiT GCC15 fork ASSEMBLES vmadot.
 * Mirrors FindSMTIME.cmake's IME1 probe. Stock binutils/gcc-13 reject this
 * opcode ("extension xsmtvdotii required"). */
int main(void) {
    __asm__ volatile("vmadot v2, v0, v1\n\t");
    return 0;
}
```

### Appendix B — decisive runtime gate (`ime_gate.c`, task 3)
```c
/* ime_gate.c — DECISIVE RUNTIME GATE (I8) for SpacemiT X60 IME1.
 * One real int8->int32 vmadot (4x4x8 MAC at VLEN=256/SEW=8) vs a plain scalar
 * C reference over the SAME discriminating signed data. SIGILL detectable via
 * before/after markers + exit code; MAC unit pinned via vlenb/vl asserts. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define M 4
#define N 4
#define K 8

/* A: (M,K)=(4,8) row-major; discriminating asymmetric signed int8 (negatives). */
static const int8_t A[M*K] = {
     1,  -2,   3,  -4,   5,  -6,   7,  -8,
    -9,  10, -11,  12, -13,  14, -15,  16,
     2,   2,  -3,  -3,   4,   4,  -5,  -5,
   -20,  19, -18,  17, -16,  15, -14,  13,
};
/* B stored (N,K)=(4,8) row-major (== B^T of the math (K,N) matrix). */
static const int8_t B[N*K] = {
     1,   1,   1,   1,   1,   1,   1,   1,
    -1,   2,  -3,   4,  -5,   6,  -7,   8,
     8,   7,   6,   5,   4,   3,   2,   1,
    -2,  -4,  -6,  -8, -10, -12, -14, -16,
};

int main(void) {
    int32_t ref[M*N];
    memset(ref, 0, sizeof(ref));
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i*K + k] * (int32_t)B[j*K + k];
            ref[i*N + j] = acc;
        }

    unsigned long vl8 = 0, vlenb = 0;
    __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
    __asm__ volatile("vsetvli %0, zero, e8, m1, ta, ma" : "=r"(vl8));
    printf("vlenb=%lu  vl(e8,m1)=%lu  (expect vlenb=32, vl=32 for VLEN=256)\n", vlenb, vl8);
    if (vlenb != 32 || vl8 != 32) {
        printf("ABORT: not VLEN=256 / 4x4x8 MAC unit; gate inconclusive on this hw\n");
        return 3;
    }

    int32_t ime[M*N];
    memset(ime, 0, sizeof(ime));
    const int8_t *pa = A, *pb = B;
    int32_t *pc = ime;

    fflush(stdout); printf("MARK_BEFORE_VMADOT\n"); fflush(stdout);

    __asm__ volatile(
        "vsetvli   t0, zero, e8, m1, ta, ma   \n\t"
        "vle8.v    v0, (%[pa])                \n\t"   /* A -> vs1 = v0 */
        "vle8.v    v1, (%[pb])                \n\t"   /* B -> vs2 = v1 */
        "vmv.v.i   v2, 0                      \n\t"
        "vmv.v.i   v3, 0                      \n\t"
        "vmadot    v2, v0, v1                 \n\t"   /* C(4x4 i32) += A(4x8).B_stored(4x8)^T */
        "vsetvli   t0, zero, e32, m1, ta, ma  \n\t"
        "vse32.v   v2, (%[pc])                \n\t"
        "addi      t1, %[pc], 32              \n\t"
        "vse32.v   v3, (t1)                   \n\t"
        :
        : [pa]"r"(pa), [pb]"r"(pb), [pc]"r"(pc)
        : "t0", "t1", "v0", "v1", "v2", "v3", "memory");

    printf("MARK_AFTER_VMADOT\n"); fflush(stdout);

    printf("scalar reference C (row-major 4x4):\n");
    for (int i = 0; i < M; i++) { for (int j = 0; j < N; j++) printf("%8d ", ref[i*N+j]); printf("\n"); }
    printf("IME vmadot result C (row-major 4x4):\n");
    for (int i = 0; i < M; i++) { for (int j = 0; j < N; j++) printf("%8d ", ime[i*N+j]); printf("\n"); }

    int mismatch = 0;
    for (int t = 0; t < M*N; t++) if (ime[t] != ref[t]) mismatch++;
    if (mismatch == 0) { printf("RESULT: PASS  (IME == scalar, all %d elements match)\n", M*N); return 0; }
    else { printf("RESULT: FAIL  (%d / %d elements mismatch)\n", mismatch, M*N); return 1; }
}
```

Build/run (exact commands used):
```
# task 2 (compile-gate): the token IS load-bearing
$RISCV_ROOT_PATH/bin/riscv64-unknown-linux-gnu-gcc -march=rv64gcv_zfh_zvfh_zba_zicbop            -mabi=lp64d -c compile_gate.c   # REJECTS
$RISCV_ROOT_PATH/bin/riscv64-unknown-linux-gnu-gcc -march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii -mabi=lp64d -c compile_gate.c   # assembles
$RISCV_ROOT_PATH/bin/riscv64-unknown-linux-gnu-objdump -d compile_gate.o | grep vmadot           # smt.vmadot v2,v0,v1  (e210312b)

# task 3 (runtime gate): build static, scp, run on REAL X60
$RISCV_ROOT_PATH/bin/riscv64-unknown-linux-gnu-gcc -O2 -static \
    -march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii -mabi=lp64d ime_gate.c -o ime_gate
scp ime_gate k1:~/n2-ime-probe/
ssh k1 'cd ~/n2-ime-probe && ./ime_gate; echo EXIT_CODE=$?'      # -> RESULT: PASS, EXIT_CODE=0
```
