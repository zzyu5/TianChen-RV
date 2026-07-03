# Research: E4 — [D-2a] loading-time resolution record + declared_instance_hash

- **Query**: format, attach point, hash computation, per-process-once semantics, [NG-3] compliance.
- **Scope**: internal (design synthesis over HEAD facts)
- **Date**: 2026-07-02

Contract: `docs/TianChen-RV_科研目标总纲v2.md:87`,
`.trellis/spec/variant-pipeline/generation-selection-tuning.md:57-59`,
`.trellis/spec/architecture/core-invariants.md:94`.
> 消费显式 schema 事实实例（profile 先展开成规范化事实集）→ 对展开后的规范化事实集计算
> **declared-instance-hash** → 落**每进程一条**解析记录。热路径**零逐次**；per-dispatch
> 强制永久禁止（[NG-3]）。解析记录是缓存事实/镜像，不是 route/dtype/schedule authority (I4)。

---

## Current loading-time state (what E4 refactors)

`lib/Transforms/DispatchRuntimeGuard.cpp`:
- `ensureDispatchAvailabilityGuardParam` (`:210-223`) materializes a **single**
  `dispatch_available` runtime ABI `RuntimeParamOp` (spec from
  `support::getDispatchAvailabilityGuardParamSpec`, `:213-214`). Caller provides
  the switch **once**; guarded `DispatchCaseOp`s link to it via `runtime_guard`
  (`attachRuntimeGuardLink`, `:225-248`). Hot path reads the caller-provided
  value — **zero per-dispatch capability checks** ([NG-3] already respected).
- **Missing:** `declared-instance-hash` (grep = 0) + any dropped resolution record
  (`docs/TianChen-RV_执行总纲v2.md:61`).

---

## D-2a MVP scope (M1 = "重构现有"; runtime chain is D-2b/M2)

**⚠ Genuine escalation — draw the line before implementing.** The full runtime
chain (hwprobe → facts → instance-hash keyed dispatch table) is [D-2b]/[D-3],
**M2/M3**. So the M1 D-2a MVP should be **compile-time + documented shape**, NOT a
runtime code path:

1. **Compute `declared_instance_hash` at compile time** over the expanded
   normalized fact set (`TargetCapabilitySet.getCapabilities()`), and
2. **Stamp it** where the loader can read it (see attach point), and
3. **Emit / document the single per-process resolution-record shape** (one line:
   `{declared_instance_hash, resolved_dispatch_switch, ts}`).

**Open decision to escalate:** does D-2a-in-E4 include *any* runtime code that
actually drops the record at process load, or is it purely a compile-time stamp +
a documented one-liner shape (record actually dropped later with D-2b)? Given M1 =
重构现有 and the hot path is already NG-3-clean, the conservative MVP is
**compile-time stamp + documented shape**, leaving the live drop to D-2b. Confirm.

---

## declared_instance_hash — computation

- **Input** = `TargetCapabilitySet::getCapabilities()`
  (`include/TianChenRV/Support/CapabilityModel.h:97-99`), the **already-expanded**
  fact set (`buildFromKernel` expands profiles → descriptors; `:90-93`). This
  satisfies canon 7d781994: hash over the **expanded** set ⇒ "profile ≡ explicit
  list ⇒ same hash" (`capability-contract.md:65`).
- **Per-descriptor fields to serialize** (`CapabilityModel.h:39-61`):
  `id`, `kind`, `status`, `availability` (stringified), `properties` (the
  `std::map<std::string,std::string>` — already ordered), and the relation lists
  `provides` / `implies` / `conflicts` (from `getProvidedIDs/ImpliedIDs/
  ConflictingIDs`). **Exclude `symbolName`** (a local IR name, not a portable fact —
  two IRs with the same facts under different symbol names must hash equal;
  confirm during implement).
- **Canonicalization (MUST):**
  - **Sort descriptors by `id`** before hashing. Profile expansion may emit a
    different descriptor order than an equivalent explicit list; without sorting
    the "profile ≡ explicit list ⇒ same hash" property fails. This is the single
    most important correctness step.
  - Within each descriptor, serialize keys in a fixed order (or JSON with sorted
    keys); sort the relation-id lists; no incidental whitespace.
  - Serialize to canonical JSON mirroring the **spirit** of the E1 idiom
    (`check_schema_gate.py:59-61`: sorted keys, `separators=(",",":")`), then
    **SHA256** it.
- **Hash primitive**: LLVM `llvm/Support/SHA256.h` is in-tree — no new dependency.
  Emit lowercase hex, like the E1 `hashlib.sha256().hexdigest()` output.
- **spirit ≠ byte-identical to Python.** E1's hash is over the schema **shape**;
  `declared_instance_hash` is over a fact **instance** — different objects. Do NOT
  over-constrain a cross-language digest match unless a lit test recomputes the hash
  in Python (not required for E4). What matters is C++-internal determinism +
  profile-equivalence.

**Reuse:** the SAME hash + the SAME hex string are what `JSONL.declared_instance_hash`
(the ①-attribution field) should carry, so one C++ helper serves both ① and D-2a.
Suggested home: a new `support`-level free function (e.g.
`support::computeDeclaredInstanceHash(const TargetCapabilitySet&) -> std::string`)
next to `CapabilityModel`, callable from both `VariantSelection.cpp` and the
DispatchRuntimeGuard path.

---

## Resolution record — format + attach point

- **Format** (per-process, one line):
  `{"declared_instance_hash": "<hex>", "dispatch_switch": "<dispatch_available resolution>", "ts": "<...>"}`.
  Keep it a cached-fact **mirror** — it is NOT a route/dtype/schedule authority (I4).
- **Attach point (compile-time stamp, MVP):** stamp `declared_instance_hash` as an
  attribute on either
  - the `dispatch_available` `RuntimeParamOp` created in
    `ensureDispatchAvailabilityGuardParam` (`DispatchRuntimeGuard.cpp:210-223`), or
  - the `KernelOp` (module-level fact),
  so the loader reads it to drop its one line. Stamping on the runtime_param keeps
  it adjacent to the switch it resolves.
- **Per-process-once semantics:** the record is dropped **once at load**, not per
  dispatch. The compile-time artifact is the hash + the documented shape; the
  actual drop (if in scope for E4 — see escalation) is guarded so it fires once per
  process. Hot dispatch path is unchanged → **[NG-3] preserved**.

---

## [NG-3] compliance check

- E4 adds **no** per-dispatch capability check. The existing hot path
  (`dispatch_available` switch read) is untouched. declared_instance_hash is
  computed **once at compile time** (and, if dropped, **once at load**). This is
  exactly the "热路径零逐次 / per-dispatch 永禁" contract
  (`core-invariants.md:94`).

## Caveats / Not found

- Whether `symbolName` is included in the hash input is a correctness call
  (portability of the hash across IRs). Recommend **exclude**; confirm against the
  canon-7d781994 intent during implement.
- No existing C++ canonical serializer to mirror (`instance-hash`/`shape-hash`/
  serializer grep = 0, `执行总纲v2.md:33`). Net-new, but small and bounded.
- The live "drop one record per process" runtime mechanism may be deferred to
  D-2b; escalate the exact E4 cut (compile-time stamp only vs stamp + live drop).
