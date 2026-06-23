# K1 Spacemit X60 IME — N2 second-family feasibility probe (2026-06-23)

**Status: hardware candidate FOUND, but N2 remains effort+toolchain-blocked. NO codegen attempted (correctly).**

## What was found (verified)
The K1 box (Spacemit X60, Bianbu) advertises an `ime` token at the tail of its `/proc/cpuinfo` ISA string:
`rv64imafdcv_…_zvfh_zvfhmin_zvkt_…_svpbmt_ime`. `model name : Spacemit(R) X60`. This is a genuine RISC-V
ISA-extension token — Spacemit's **IME (Integer Matrix Extension)** — a non-RVV matrix extension hanging off
the RISC-V core. (Note: the `libimepinyin`/`libimecore` files on the box are an unrelated input-method
red herring; the `ime` here is the cpuinfo ISA token.)

## Why this matters for N2 (the boundary judgement)
Per the N2 family-entry criterion (`spec/architecture/design-boundaries.md` + memory `n2-family-entry-boundary`):
*a family enters iff its capability is expressible as a RISC-V capability FACT and consumed zero-core-branch.*
IME is an extension **on the RISC-V core**, exposed as a capability fact in cpuinfo → it is exactly the
legitimate **Case A** second-family candidate the campaign said N2 was "hardware-blocked" pending. **The
hardware now exists.** So N2 is no longer *hardware*-blocked in the weakest sense (silicon present).

## Why N2 is NOT unblocked (brake — honest ceiling, no over-claim)
A cpuinfo bit is not N2 progress. The probe (read-only, no codegen) found:
1. **No toolchain support on the box.** Bianbu `clang-18.1.8` rejects `-march=rv64gcv_ime` (and `gcc-13.2.0`
   `-march=help` lists no `ime`). The only matrix extensions clang knows are **SiFive's** (`xsfvfwmaccqqq`,
   `xsfvqmaccdod`, `xsfvqmaccqoq`) — a DIFFERENT vendor; those encodings will not map to X60's IME. There is
   no march token, no intrinsics header, and no Spacemit IME programming-model doc on the box.
2. **No register/programming model in hand.** The user's own bar was "拿到编程/寄存器模型后再按判据重判."
   We do not have it. Without it we cannot even author hand-encoded IME instructions, let alone judge the
   capability shape.
3. **The deepest trap (does not count as N2):** even with a toolchain that emits IME, calling a vendor lib
   or a ggml IME path is NOT N2. N2 requires OUR pass to consume an IME-capability fact zero-core-branch and
   emit IME — entirely unbuilt.

## Honest status line (for the ledger / paper)
> **N2 hardware candidate found (K1 Spacemit X60 advertises the IME ISA extension); programming/register
> model, toolchain march support, and our codegen are all absent → N2 is no longer hardware-blocked, now
> toolchain+effort-blocked.** Next step (deferred, not this session): obtain Spacemit's IME spec / register
> model → judge the capability shape against the N2 criterion → only then scope an emitter. No build started.

## Probe artifacts
clang `--print-supported-extensions` (no `ime`, SiFive `xsf*` only); `gcc -march=rv64gcv_ime` rejected;
`clang --target=riscv64 -march=rv64gcv_ime` rejected; `rv64gcv_xsfvfwmaccqqq` accepted (SiFive, wrong vendor).

## CORRECTION (2026-06-23, web research `research/spacemit-ime.md`): IME IS TARGETABLE TODAY — N2 is effort-bounded, not blocked
The "no toolchain support / no programming model" framing above is true ONLY of the **stock Bianbu**
clang-18/gcc-13. It is the wrong conclusion about IME overall. The real, public, installable path:
1. **march spelling was wrong here.** `rv64gcv_ime` is NOT the token; the cpuinfo `ime` string is not a march
   token at all. IME = SpacemiT **vendor custom extension `xsmtvdotii`**. Correct:
   `-march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii -mabi=lp64d` (source: llama.cpp `FindSMTIME.cmake` +
   assembler-error in PR #22317).
2. **Toolchain EXISTS** — SpacemiT GCC ≥15 fork `github.com/spacemit-com/toolchain` **v1.2.4 (GCC 15.2)** —
   exactly the "XuanTie-for-RVV0.7" vendor-fork pattern. Install: wget v1.2.4 glibc tarball → untar →
   `export RISCV_ROOT_PATH=…` → triple `riscv64-unknown-linux-gnu-gcc`. (clang/ruyisdk-llvm IME path NOT
   confirmed this session — not grepped.)
3. **Programming/register model IS public** — `spacemit-com/riscv-ime-extension-spec` (AsciiDoc encodings) +
   `spacemit-com/docs-ai` `ime_extension.md` / `Zvvm_spacemit` v0.6. The user's "拿到编程模型再判" precondition
   is now SATISFIABLE.
4. **Reference oracle EXISTS** — llama.cpp has a MERGED SpacemiT X60 IME backend (PR #15288, 2025-09-29,
   `ggml/src/ggml-cpu/spacemit/`), hand-emitted `vmadot` inline-asm (IME1); IME2 via PR #22863. Use as the
   N2 oracle like ggml is for RVV. (No `spacemit_ime.h` intrinsics — emission is inline-asm mnemonics.)

**Corrected status line:** the four N2-via-IME prerequisites are all in hand — **hardware** (K1 = X60 = IME1,
satisfies I8 real-silicon), **toolchain** (SpacemiT GCC 15.2 fork), **spec** (public), **oracle** (llama.cpp
spacemit backend). **N2 is no longer hardware- OR toolchain-blocked; it is now effort-bounded:** the
remaining work is OUR novelty — model `xsmtvdotii` as a capability fact and have our pass consume it
**zero-core-branch** to emit `vmadot`. Hedged inferences to confirm on real K1 before claiming: A60(docs)≡
X60(cpuinfo); `xsmtvdotii`≡`Xsmti8i32mm` sub-ext; runtime/perf N2 claim needs a real-K1 run (QEMU = functional
evidence only, not I8 runtime).
