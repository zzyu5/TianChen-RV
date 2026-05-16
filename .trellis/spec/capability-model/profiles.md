# Target Profiles

Profiles are capability fixtures, not task milestones. They describe target facts that can be queried, verified, and used by passes.

## RVV Main Profile

```text
name: rvv-main
access: ssh rvv
hardware: RISC-V CPU, 64 cores
vector: RVV 1.0
permission: sudo available
role: primary development, primary performance, primary correctness
```

Capability examples:

```text
rv64
rvv
zvl128b or actual probed minimum VLEN / vlenb-derived capacity
fp32/fp64 depending on hardware
thread-runtime: OpenMP or pthread
native compile support
```

Rules:

- This is the current real hardware mainline.
- RVV correctness, runtime, or performance claims require real `ssh rvv` evidence.
- RVV evidence should include probe output for relevant facts such as host identity, RISC-V architecture, core count, RVV/toolchain availability, CMake availability, and sudo behavior when relevant.
- Probe artifacts may expose a sanitized `capability_facts` section as bounded
  evidence-tool output. Python probe tooling must not translate that section
  into `tcrv.exec` MLIR capability, target, kernel, selected route, or scalar
  fallback modeling.
- The authoritative transition from RVV probe evidence to compiler-visible
  capabilities is plugin-local C++ RVV capability profile validation followed
  by `TargetCapabilitySet` population. Python must not implement capability
  relations, RVV legality, variant selection, lowering, emission, runtime ABI,
  target profile generation, or fallback selection.
- Remote probe output must not fabricate compiler-route config facts such as
  first-slice SEW, LMUL, tail policy, or mask policy. Such finite config facts
  are plugin-selected compiler facts, not raw hardware/toolchain evidence.
- Required positive facts for the RVV probe-derived profile include `riscv64` architecture, `hart_count > 0`, bounded RVV ISA/vector hints, clang and CMake availability/version facts, minimal hand-written RVV compile/run success, and selected march/mabi facts when emitted by the probe.
- Stable profile capability identities must remain plugin-local and generic,
  such as `rv64`, `rvv`, `rvv.hart_count`, `riscv.toolchain.march`,
  `riscv.toolchain.mabi`, `rvv.toolchain.clang`, `rvv.toolchain.cmake`,
  `rvv.probe.compile_run`, `rvv.toolchain.march`, and `rvv.toolchain.mabi`;
  provider identity, benchmark names, logs, and performance measurements must
  not become capability IDs.
- The RVV profile's `rvv.hart_count` capability may provide the generic
  relation id `target.hart_count` while retaining its plugin-local owning id.
  The `count` property is a hardware/uarch target fact for capability-aware
  planning; it is not a runtime thread count, dispatch guard, tensor shape,
  descriptor-local fixture size, AVL, or VL.
- Local compile-only, local smoke-only, or unproven docs/spec changes must not be described as RVV runtime evidence.
- RVV runtime/performance/correctness claims must name this profile or a derived probed profile.
- VLEN, vlenb-derived vector capacity, and dtype support should be probed or
  declared with provenance, not guessed. They are target capability facts, not
  runtime SSA/control values and not per-variant constants.

## K3/IME Later Profile

```text
name: k3-ime
hardware: SpacemiT/K3-class RISC-V system
role: IME extension plugin validation
status: later environment
```

Capability examples:

```text
rvv
spacemit.ime
vector-register-backed matrix capability
vendor intrinsic or inline asm path
```

Rules:

- Treat K3/IME as a later plugin validation path until actual hardware/toolchain facts exist.
- IME capability depends on RVV/vector-register-backed resource constraints and vendor emission path.
- Do not make IME the current primary hardware route unless the environment is verified.
- IME runtime/performance claims require real K3/IME hardware and toolchain evidence.

## RISC-V Sophgo Offload Profile

```text
name: riscv-sophgo-offload
hardware: RISC-V host + Sophgo accelerator path
role: runtime-offload capability case
```

Capability examples:

```text
rvv or scalar CPU fallback
sophgo runtime available
C ABI call path
PCIe or SoC mode
async call if available
```

Rules:

- Model this as `kind = "runtime-offload"`.
- Do not classify this as custom RISC-V ISA.
- Cost and dispatch must include runtime launch, transfer, sync, and fallback behavior.
