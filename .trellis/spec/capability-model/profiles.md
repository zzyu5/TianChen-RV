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
zvl128b or actual probed minimum VLEN
fp32/fp64 depending on hardware
thread-runtime: OpenMP or pthread
native compile support
```

Rules:

- This is the current real hardware mainline.
- RVV correctness, runtime, or performance claims require real `ssh rvv` evidence.
- RVV evidence should include probe output for relevant facts such as host identity, RISC-V architecture, core count, RVV/toolchain availability, CMake availability, and sudo behavior when relevant.
- Local compile-only, local smoke-only, or unproven docs/spec changes must not be described as RVV runtime evidence.
- RVV runtime/performance/correctness claims must name this profile or a derived probed profile.
- VLEN and dtype support should be probed or declared with provenance, not guessed.

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
