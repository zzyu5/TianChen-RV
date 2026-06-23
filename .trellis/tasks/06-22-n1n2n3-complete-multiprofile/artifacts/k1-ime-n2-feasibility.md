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
