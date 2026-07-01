# IME prefill-regime net-contribution re-test — HALTED (perf is 收尾)

**Status:** **HALTED / premature.** evidence-status = **halted** (NOT measured-and-sealed).
**Date:** 2026-06-30
**Board:** `ssh k1` = SpacemiT X60, RVV1.0 **VLEN256**, **IME1**, harts 0-3 (`taskset -c 0-3`),
march `xsmtvdotii`. (Non-conflicting with the rvv board — the q4_K judge runs on rvv.)
**Target cell:** 注7 `ime-prefill-corrigendum` open follow-up in `doc/KERNEL-优化自查表.md`
(does the SpacemiT IME matrix unit net a contribution in **prefill** vs ≈0 in **decode**).

---

## Halt directive (governs this file)

Mid-setup, the plan was changed: **performance testing is premature** — it is the **收尾 /
wrap-up** step, done AFTER the implementation is mature, and a lot of implementation is still
undone. Directive: do not start further bench rounds; finish only the single in-flight command
(don't kill mid-run → would orphan the remote `llama-bench`); restore k1 pristine; mark this
re-test **HALTED / premature**; **do NOT present partial numbers as results**.

⇒ **No net-IME prefill verdict is computed or sealed here.** The question stays **OPEN**, deferred
to the wrap-up phase. The standing (un-sealed) reference measurement remains the prior
`IME-E2E-SPACEMIT-TOGGLE-FINDING.md` (06-25), which is unchanged by this halt.

---

## What WAS verified (durable infra facts — NOT perf claims, reusable when perf resumes)

The existing IME A/B build is intact and the engagement gate passes — this is plumbing
verification, not a performance result:

- **Build found (no rebuild):** `~/tcrv-k1-llama/build-ime` (SPACEMIT=ON) vs
  `~/tcrv-k1-llama/build-off` (SPACEMIT=OFF) — same clang-18 `-fno-integrated-as` toolchain, only
  `GGML_CPU_RISCV64_SPACEMIT` flipped (the can't-confound one-toolchain A/B 注7 used). Model
  `~/tcrv-k1-llama/models/tinyllama-q4_0.gguf` (637 MB, tinyllama-1B Q4_0).
- **Lib sizes:** ON `libggml-cpu.so.0.15.1` = 1222928 B / OFF = 920776 B (match the 06-25 finding).
- **Engagement gate PASS (objdump/strings/symbols):**
  - `objdump -d | grep -c vmadot`: **ON 32 / OFF 0** (real IME1 4x4x8 int8 MAC present ON-only).
  - `strings | grep -ci spacemit`: **ON 624 / OFF 0**.
  - `objdump -T | grep gemm_kernel_i8i4`: **ON 1 (defined T) / OFF 0**.
  - `ldd` each `llama-bench`: RUNPATH → its OWN `bin/` lib (ON→ON lib, OFF→OFF lib), no
    `LD_LIBRARY_PATH` juggling.
  - Runtime print (ON arm smoke): `CPU_RISCV64_SPACEMIT: ... use_ime1: 1` — IME engaged at runtime.

The IME instruction is physically present ON / absent OFF — engagement proven by the instruction,
not assumed. This gate is the reusable part; it carries no perf claim.

---

## Why the question is genuinely open (context, not a result)

注7 measured IME's e2e net contribution ≈ 0 in the **decode** regime (whole-llama 1.65× but M=1
decode also 1.47× → the 1.65× was a whole-kernel-family swap, not the matrix unit). The
corrigendum flagged that **prefill (large-M, where a matrix unit SHOULD help) was the open cell**.
The prior clean toggle (`IME-E2E-SPACEMIT-TOGGLE-FINDING.md`, 06-25) had already probed prefill and
landed on: a real ~+0.18 prefill-only increment above the 1.47× decode floor, **but confounded** —
`GGML_CPU_RISCV64_SPACEMIT` gates a **227-symbol kernel FAMILY** (RVV + quant + IME together, incl.
its own M=1 GEVM branch in `ime.cpp`), so the `vmadot` unit cannot be cleanly isolated by SPACEMIT
ON/OFF alone. That isolation gap — not a fresh number — is what a mature-implementation wrap-up
re-test would need to close. Re-running benches now does not advance it; hence the halt is correct.

**Honest framing (unchanged):** the GEMM is **ggml-spacemit's**, NOT our compiler-emitted kernel.
This cell answers "does the matrix UNIT net-contribute in prefill," not "our IME GEMM wins."

---

## k1 state left behind — PRISTINE

- `pgrep -a llama` → **NONE**; no bench procs; CPU **93.1% idle**; load avg **2.57** (the known
  D-state virtio floor ~2.0).
- Builds/libs **untouched** (only existing binaries were invoked; no rebuild, no `lib/` change, no
  `LD_LIBRARY_PATH`).
- Incident handled: an earlier exploratory round was `timeout`-wrapped on the ssh client; the
  client timeout killed only the local ssh, **orphaning the remote OFF `llama-bench`** (~200% CPU
  contention). It was identified (PID 63094, mine, user `bianbu`, my exact command) and
  `pkill`-ed — **no other users' processes touched**. The box then settled to idle and the one
  in-flight sweep was allowed to finish naturally (which is what left k1 clean).

---

## Raw capture — out-of-band, NOT presented here as results

One interleaved ON/OFF sweep had to finish naturally (killing it mid-run would have orphaned the
remote `llama-bench`). Per the halt directive its numbers are **not presented as results** — no
ratio is derived, no net-IME verdict is computed, the cell stays OPEN. The raw t/s are parked
out-of-band in `ime-prefill-raw-capture-HALTED.txt` (explicitly NOT a finding, do-not-cite,
cold-start hint only). The prior 06-25 toggle finding remains the standing un-sealed reference.

---

## Verdict

**HALTED — re-test deferred to the 收尾/wrap-up phase (after implementation matures).** No net-IME
prefill verdict is asserted. Engagement gate verified (durable). k1 restored pristine. The open
question (clean IME-unit isolation in prefill) is unchanged and remains owned by the prior 06-25
toggle finding as the standing un-sealed reference.

evidence-status = **halted** (not measured; not presumed; explicitly un-sealed).
