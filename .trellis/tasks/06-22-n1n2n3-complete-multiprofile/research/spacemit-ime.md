# Research: Targeting the SpacemiT IME (Integer Matrix Extension)

- **Query**: How to TARGET the SpacemiT X60 IME matrix extension — toolchain, spec, llama.cpp backend, intrinsics/encodings, and correct `-march` spelling.
- **Scope**: external (web/GitHub) + internal cross-reference (N2 IME family)
- **Date**: 2026-06-23

## Verdict (TL;DR)

**IME is targetable TODAY** given the right toolchain. The pieces all exist publicly:

1. **Toolchain**: SpacemiT ships a GCC fork (currently **GCC 15.2**, release **v1.2.4**) mirrored at
   `github.com/spacemit-com/toolchain`. The IME `vmadot` family is gated behind a **custom
   `-march` sub-extension `xsmtvdotii`** that *only this fork's GCC ≥ 15* understands. Stock
   Bianbu clang-18 / gcc-13 reject it (they only know SiFive `xsf*` matrix exts), which is exactly
   the symptom in the task.
2. **Spec**: TWO public docs — an AsciiDoc IME-extension proposal
   (`github.com/spacemit-com/riscv-ime-extension-spec`) and a newer, more complete
   **"SpacemiT AI Matrix Extension" / `Zvvm_spacemit` profile v0.6** markdown
   (`github.com/spacemit-com/docs-ai`, 46 instructions, full encodings).
3. **Reference kernels**: **llama.cpp already has a merged SpacemiT IME backend** (PR #15288,
   merged 2025-09-29) with `vmadot` inline-asm kernels for the X60, plus a merged **IME2** path
   (PR #22863). This is a ready-made hand-emit reference (no intrinsics header needed — they use
   `__asm__ volatile("vmadot ...")`).

**Minimum artifacts needed to build IME code**: (a) the `spacemit-com/toolchain` GCC ≥ 15 release,
(b) `-march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii -mabi=lp64d`, (c) the spec for semantics. All free/public.

**Honesty caveat tied to our spec**: our `ime-plugin.md` says IME "needs real hardware/toolchain
evidence." The *toolchain + spec + reference kernels* gap is now CLOSED (public). What remains for
an N2 runtime/perf claim is **real X60/K1 (or K3) hardware** to run on — compile-only / QEMU is not
runtime evidence under I8. QEMU emulation IS available (`vlen=256` recipe below) for functional bring-up.

---

## Findings

### Q1 — Toolchain: does SpacemiT ship a compiler with IME, and how to install?

**YES — a GCC fork.** (No public clang/LLVM fork with IME found; the only LLVM matrix support in
stock toolchains is SiFive `xsf*`, which is a *different* vendor extension, not SpacemiT IME.)

- **Repo (mirror of official downloads)**: https://github.com/spacemit-com/toolchain
  README: "This repository mirrors the cross-compilation toolchains from SpacemiT
  (https://www.spacemit.com/community/resources-download/Tools/Cross-compilation%20toolchain)".
- **Latest release v1.2.4 (GCC 15, published 2026-05-14)** — direct asset URLs:
  - Linux glibc (what you want for native Linux binaries):
    `https://github.com/spacemit-com/toolchain/releases/download/v1.2.4/spacemit-toolchain-linux-glibc-x86_64-v1.2.4.tar.xz`
  - Bare-metal newlib (ELF):
    `https://github.com/spacemit-com/toolchain/releases/download/v1.2.4/spacemit-toolchain-elf-newlib-x86_64-v1.2.4.tar.xz`
  - Older tags also published: v1.2.2, v1.1.2, v1.0.5, v1.0.1, v1.0.0.
- **Compiler triple**: `riscv64-unknown-linux-gnu-gcc` / `-g++` (from llama.cpp's
  `cmake/riscv64-spacemit-linux-gnu-gcc.cmake`).
- **Install (cross-compile host)**:
  ```bash
  wget https://github.com/spacemit-com/toolchain/releases/download/v1.2.4/spacemit-toolchain-linux-glibc-x86_64-v1.2.4.tar.xz
  tar xf spacemit-toolchain-linux-glibc-x86_64-v1.2.4.tar.xz
  export RISCV_ROOT_PATH=$PWD/spacemit-toolchain-linux-glibc-x86_64-v1.2.4   # exact dir name after untar
  $RISCV_ROOT_PATH/bin/riscv64-unknown-linux-gnu-gcc --version
  ```
  Source for the `wget` line and `RISCV_ROOT_PATH` convention:
  https://github.com/ggml-org/llama.cpp/blob/master/docs/build-riscv64-spacemit.md
- **Packaging alternatives**:
  - **RuyiSDK** is the broader RISC-V SDK/package manager (`github.com/ruyisdk/ruyi`,
    `ruyisdk/packages-index`); it packages SpacemiT toolchains/boards (could not enumerate the
    exact package id — `packages-index` code search requires GitHub auth, which was unavailable).
    Worth a `ruyi list | grep -i spacemit` on a configured host.
  - **Bianbu** is SpacemiT's Ubuntu fork; its docs live at `github.com/spacemit-com/bianbu-docs`
    and the AI matrix docs at `github.com/spacemit-com/docs-ai`. The stock Bianbu apt clang-18 /
    gcc-13 do **not** support IME — you must use the `spacemit-com/toolchain` GCC ≥ 15.

### Q2 — Is there a PUBLIC IME spec (instructions / encodings / programming model)?

**YES — two public sources, both on GitHub:**

**(a) `riscv-ime-extension-spec` (the original IME proposal, AsciiDoc):**
https://github.com/spacemit-com/riscv-ime-extension-spec (branch `master`; releases page for PDFs).
Acknowledges T-Head's `riscv-matrix-extension-spec` as a reference (related but a *different* vendor
ext — do not conflate with `xtheadmatrix`). Key spec facts:
- **Programming model** (`src/program-model.adoc`): reuses standard RVV vector registers + CSRs,
  **no new matrix registers or state CSRs**. Uses `vsetvli`/`vsetivli`/`vsetvl`. The MAC unit is
  selected from the configured `vl` and `SEW`. **VLEN 128–4096; SEW only 4/8/16; LMUL ≤ 1.**
  A `Copies` variable (1 or 2): `Copies = (sqrt(VLEN/64)==floor(sqrt(VLEN/64)) ? 1 : 2)`.
  MAC-unit table (M×N×K[×Copies]) by `vl*SEW`, e.g. VLEN=256/SEW=8 → **4×4×8**, accumulator int32.
  Illegal-instruction trap if hardware lacks the selected MAC unit.
- **Instruction list** (`src/instruction-list.adoc`): integer `vmadot / vmadot1 / vmadot2 /
  vmadot3 / vmadotn` (funct7 `111000`/`111001`, signedness variants uu/ss/us/su) and float
  `vfmadot / vfmadot1..n` (funct7 `111010`).
- **Instruction semantics** (`src/instruction-func.adoc`): A in VS1 = `Copies*(M,K)`, B in VS2 =
  `Copies*(K,N)`, C in two sequential regs (VD index must be even) = `Copies*(M,N)`. Integer
  variants `vmadot / vmadotu / vmadotsu / vmadotus` over int4/int8/int16 → int32 (fp32 for int16).
  C ← C + A·Bᵀ accumulate.
- **Instruction format**: `src/instruction-format.adoc` (wavedrom `ma-format.adoc`).

**(b) `docs-ai` — "SpacemiT AI Matrix Extension", `Zvvm_spacemit` profile v0.6 (newer, fuller):**
https://github.com/spacemit-com/docs-ai/blob/main/en/architecture/ime_extension.md
(status "Public Release", updated 2026-04-13). This is the IME2-era spec. Highlights:
- **46 custom AI instructions**, 7 classes: integer MM, float MM, integer/float sliding-window MM
  (convolution), block-quantization MM, 4:2 structured-sparse MM, and data-layout-transform
  (`vpack`/`vnspack`) instructions.
- Operation form `C ← C + A·Bᵀ`, e.g. `smt.vmadot vd, vs1, vs2, i8`.
- Aligns with community `Zvvm` / `Zvzip` IME drafts; reuses the 32 V-registers as 2-D tiles, LMUL
  encodings constrained (others reserved). Chapter 8 = full instruction-encoding summary.
- **Sub-extension map (SpacemiT `Xsmt*` ↔ community `Zvvm`)** — the names that matter for `-march`:
  `Xsmti4i32mm`(↔`Zvvi4i8mm`), `Xsmti8i32mm`(↔`Zvvi8i32mm`), `Xsmti8i32mm_slide`,
  `Xsmti4i32mm_42sp`, `Xsmti8i32mm_42sp`, `Xsmti4fp16mm_scl16f`, `Xsmti4bf16mm_scl16f`,
  `Xsmti8fp16mm_scl16f`, `Xsmti8bf16mm_scl16f`, `Xsmtfp16fp32mm`(↔`Zvvfp16fp32mm`),
  `Xsmtbf16fp32mm`(↔`Zvvbf16fp32mm`), `Xsmtfp16fp32mm_slide`, `Xsmtbf16fp32mm_slide`.
- **Per-chip support (important for X60 vs K3):**
  - **K1 core** supports the *IME1* subset: `Xsmti8i32mm`, `Xsmti8i32mm_slide`.
  - **K3 (A100 core)** supports the full *IME2* set (int4/sparse/fp16/bf16 variants above).
  - **Naming reconciliation (unverified equivalence)**: `docs-ai` §1.7 labels the K1 core **A60**;
    the cpuinfo and llama.cpp PR #15288 label the BPI-F3 / Muse / Bianbu chip **X60**. Both are
    paired against the A100/K3 across the sources, so they are treated here as the same K1-class
    part (A60 ≡ X60) — but no single source states this equivalence, so it is an inference.
  - **`xsmtvdotii` ↔ `Xsmti8i32mm` is a functional bridge, not a pinned equivalence**: the IME1
    march token confirmed by evidence is `xsmtvdotii` (llama.cpp naming); `Xsmti8i32mm(_slide)` is
    the `docs-ai` formal sub-extension naming. No march token for any `Xsmt*mm` sub-extension was
    found, so equating GCC15's `xsmtvdotii` with the K1's `Xsmti8i32mm` capability is inferred from
    "both = int8→int32 MM on the K1-class core," not directly sourced.

### Q3 — Does llama.cpp / ggml already have a SpacemiT IME backend?

**YES — merged, in tree, with X60-specific IME kernels.**

- **PR #15288 "ggml: riscv: add riscv spacemit backend"** (MERGED 2025-09-29):
  https://github.com/ggml-org/llama.cpp/pull/15288 — "Specific optimizations have been made for the
  SpacemiT X60 CPU. The SpacemiT IME extended instructions are used to accelerate matrix
  calculations for Q4_0/Q4_1/Q4_K, along with general RVV optimizations." Body includes the X60
  `/proc/cpuinfo` ISA string and pp512/tg128 numbers for Qwen2.5.
- **PR #22863 "ggml-cpu: Add IME2 Instruction Support for the SpacemiT Backend"** (MERGED) —
  https://github.com/ggml-org/llama.cpp/pull/22863 — adds the IME2 path; links the `docs-ai`
  `ime_extension.md` spec.
- **PR #22317 "cmake: append xsmtvdotii march for SpacemiT IME"** (MERGED) —
  https://github.com/ggml-org/llama.cpp/pull/22317 — the `-march` fix (see Q5).
- **PR #16629** fix SpaceMit IME array OOB; **PR #23642** bump CI toolchain to v1.2.4 (GCC 15).
- **Backend source tree** `ggml/src/ggml-cpu/spacemit/`:
  `ime.cpp` (77 KB), `ime.h`, `ime1_kernels.cpp` (50 KB, IME1/X60), `ime2_kernels.cpp` (292 KB),
  `ime_kernels.h`, `ime_env.{cpp,h}`, `repack.{cpp,h}`, `rvv_kernels.{cpp,h}`, plus
  `spine_*` (mem pool / TCM / barrier) helpers.
- **Build doc** (canonical install + flags + QEMU + supported quants Q4_0/Q4_1/Q4_K on X60):
  https://github.com/ggml-org/llama.cpp/blob/master/docs/build-riscv64-spacemit.md
- **CMake detect** `ggml/src/ggml-cpu/cmake/FindSMTIME.cmake` + toolchain file
  `cmake/riscv64-spacemit-linux-gnu-gcc.cmake`. CI action `.github/actions/linux-setup-spacemit`.

### Q4 — Intrinsics header or `.insn` encoding reference for hand-emitting IME

- **No `spacemit_ime.h` intrinsics header found** in the public toolchain or ggml. The reference
  implementation hand-emits via **GCC inline assembly with the mnemonic directly** (the GCC ≥ 15
  fork's assembler recognizes the opcodes once `_xsmtvdotii` is on `-march`). From
  `ggml/src/ggml-cpu/spacemit/ime1_kernels.cpp`:
  ```c
  __asm__ volatile(
    "vmadot v16, v14, v0 \n\t"
    "vmadot v18, v14, v1 \n\t"
    ... );
  ```
- **`FindSMTIME.cmake` is itself a compile-probe encoding reference** — it tests each opcode the
  fork must accept:
  `vmadot v2,v0,v1` (IME1), `vmadot v2,v0,v1,i4`, `vmadot v2,v0,v1,i8`,
  `vfwmadot v2,v0,v1,fp16`, `vmadot.hp v2,v0,v1,v0,0,i4|i8`, `vmadot1 v2,v0,v1`,
  `vpack.vv v2,v0,v1,2`, `vnspack.vv v2,v0,v1,2`.
  IME1 is selected if `vmadot v2,v0,v1` compiles; IME2 requires the `i4` + `vpack` + `vnspack`
  forms to compile.
- **Raw `.insn` encodings** (if you ever need to emit IME on a toolchain that lacks the mnemonics):
  use the funct7 values from `riscv-ime-extension-spec/src/instruction-list.adoc`
  (integer `vmadot`=`111000`, `vmadot1..n`=`111001`; float `vfmadot*`=`111010`) and Chapter 8 of
  the `docs-ai` `ime_extension.md` for the full encoding table. Preferred path is the mnemonics +
  GCC ≥ 15 fork, not raw `.insn`.

### Q5 — Correct `-march` spelling for IME

**`xsmtvdotii`**, appended to a base ISA, on **SpacemiT/RuyiSDK GCC ≥ 15** (stock clang-18/gcc-13 reject it).

- Canonical base + ext (from `FindSMTIME.cmake` and the toolchain cmake file):
  ```
  -march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii  -mabi=lp64d
  ```
  `FindSMTIME.cmake` only appends `_xsmtvdotii` when `CMAKE_C_COMPILER_ID == GNU` and
  `VERSION >= 15` — i.e. the token is GCC-15-fork-specific.
- **Proof it's required** (PR #22317 body, tested on SpacemiT GCC 15.2): without it the assembler
  fails:
  ```
  Error: unrecognized opcode `vmadot v16,v14,v0', extension `xsmtvdotii' required
  ```
- **The X60 `/proc/cpuinfo` `isa` string does NOT contain an `ime` token** — it is the long
  `rv64imafdcv_...zvfh...` RVV string (see below). IME is a *vendor custom extension* exposed to the
  compiler as `xsmtvdotii` (and, in the IME2/`Zvvm_spacemit` spec, as the `Xsmt*mm` sub-extension
  family), not as a kernel-advertised `ime` hwcap. So "spell IME in `-march`" = `xsmtvdotii`
  (IME1; the `Xsmt*mm` sub-extension naming is the IME2 spec's formal scheme — see the naming
  reconciliation note above), **not** `rv64gcv_ime`.
- Why stock toolchains fail with `rv64gcv_ime`: neither upstream GCC/binutils nor LLVM define an
  `ime` extension; LLVM's only matrix exts are SiFive `xsf*` (`xsfvfwmaccqqq`, etc.) — a different
  vendor. Hence stock Bianbu clang-18/gcc-13 reject both `rv64gcv_ime` and `xsmtvdotii`.

---

## SpacemiT X60 reference facts (from llama.cpp build doc / PR #15288)

```
model name : Spacemit(R) X60
isa        : rv64imafdcv_zicbom_zicboz_zicntr_zicond_zicsr_zifencei_zihintpause_zihpm
             _zfh_zfhmin_zca_zcd_zba_zbb_zbc_zbs_zkt_zve32f_zve32x_zve64d_zve64f
             _zve64x_zvfh_zvfhmin_zvkt_sscofpmf_sstc_svinval_svnapot_svpbmt
uarch      : spacemit,x60      mvendorid : 0x710     marchid : 0x8000000058000001
```
Note: VLEN reported / emulated as **256** (QEMU recipe uses `-cpu max,vlen=256,elen=64,vext_spec=v1.0`).
IME1 MAC unit at VLEN=256/SEW=8 → **4×4×8** int32. X60/K1 supports IME1 only
(`Xsmti8i32mm`, `Xsmti8i32mm_slide`); K3/A100 adds the full IME2 set.

### QEMU functional bring-up (no hardware)
```bash
wget https://archive.spacemit.com/spacemit-ai/qemu/jdsk-qemu-v0.0.14.tar.gz   # SpacemiT QEMU w/ IME
# build llama.cpp per build-riscv64-spacemit.md, then:
$QEMU_ROOT_PATH/bin/qemu-riscv64 -L $RISCV_ROOT_PATH/sysroot \
  -cpu max,vlen=256,elen=64,vext_spec=v1.0 ./build/bin/llama-cli -m model-Q4_0.gguf -t 1
```

## Related internal specs (N2 IME family)

- `.trellis/spec/extension-plugins/ime-plugin.md` — IME plugin = N2 second-family target; lists
  expected ops (`tcrv.ime.mma/dot/...`), `vlen_dependent=true`, `register_model=rvv-vector-register-backed`.
  The spec's "needs real hardware/toolchain evidence" caveat: toolchain+spec gap now closed; only
  real X60/K1/K3 *runtime* evidence remains.
- `.trellis/spec/capability-model/profiles.md` (§ `k3-ime`) — capability ids `spacemit.ime`,
  vector-register-backed matrix capability, vendor-intrinsic / inline-asm path. **Note**: the spec
  named hardware "SpacemiT/K3-class"; per `docs-ai`, **K1/X60 supports IME1, K3/A100 supports IME2** —
  pick the profile to the actual board (our `ssh rvv` is the X60/K1-class → IME1 `xsmtvdotii`).
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`, `architecture/system-positioning.md` — IME as
  TCRV extension family on the common path (linalg.matmul → tcrv.exec → @ime variant).

## Caveats / Not Found

- **No public SpacemiT LLVM/clang fork with IME confirmed** (not proven absent). Evidence it's
  GCC-only: `FindSMTIME.cmake` appends `_xsmtvdotii` only when `CMAKE_C_COMPILER_ID STREQUAL "GNU"`
  and version ≥ 15 — i.e. llama.cpp's own maintainers gate the IME march to GCC. The SpacemiT
  toolchain releases are GCC. I did **not** grep `ruyisdk/llvm-project` (it exists in the org list),
  so a clang IME path is "not confirmed," not "confirmed absent." IME hand-emit on clang would
  otherwise need raw `.insn` (funct7 from the spec); GCC ≥ 15 fork (mnemonic + `xsmtvdotii`) is the
  confirmed supported path.
- **No `spacemit_ime.h` intrinsics header** — emission is inline-asm mnemonics, not C intrinsics.
- **Exact RuyiSDK package id for the SpacemiT IME toolchain not confirmed** — `ruyisdk/packages-index`
  code search needed GitHub auth (unavailable this session). Verify on a host with `ruyi list`.
- **`gh` CLI and exa MCP were unavailable**; all data above is from GitHub REST API (unauthenticated)
  + raw.githubusercontent reads of primary repos. Every claim cites a repo path or PR URL.
- **Hardware/runtime untested here** — all evidence is documentary. An N2 performance claim still
  requires real X60/K1 (IME1) or K3 (IME2) `ssh` runtime evidence per invariant I8 (compile-only /
  QEMU is functional, not runtime, evidence).
