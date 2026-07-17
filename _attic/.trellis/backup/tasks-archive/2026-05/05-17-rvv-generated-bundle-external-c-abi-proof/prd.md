# RVV generated-bundle external C ABI correctness proof

## Goal

Close the bounded RVV generated-bundle external C ABI proof for the existing
public `tcrv-translate --tcrv-source-artifact-bundle-front-door` route. The
claim is that supported RVV i32m1 add/sub/mul source-front-door inputs produce
a coherent generated object/header bundle that an external C harness can
compile, link, and run on the real RVV machine through `ssh rvv`.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial real state for this round: HEAD
  `55ce5c9 target: erase binary self-check expectation API`; worktree clean.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.
- The previous deletion round removed the public binary self-check expectation
  API and rewrote testing guidance to reject the old self-check/direct
  executable bridge as deleted behavior.
- Current HEAD already contains the public source-artifact bundle front door,
  the RVV materialized EmitC object/header bundle path, and
  `scripts/rvv_generated_bundle_abi_e2e.py`.
- Archived task
  `.trellis/tasks/archive/2026-05/05-17-05-17-rvv-source-bundle-runtime-abi-proof/prd.md`
  shows that the evidence tool was previously moved from the manual
  `tcrv-opt | tcrv-translate` chain to the one-command source-artifact bundle
  front door and had real `ssh rvv` add/sub/mul evidence.
- The new Direction Brief asks for a fresh proof after the binary self-check
  deletion round, with explicit fail-closed coverage for bundle index/header/
  object coherence, forbidden descriptor/direct-C/source-export residue, and
  sanitizer redaction issues.

## Boundaries

- The module owner is the generated RVV bundle external C ABI boundary.
- Use the existing public source-artifact bundle front door and existing
  materialized EmitC artifact routes.
- Python remains evidence tooling only. It may invoke compiler tools, parse
  bundle metadata, generate an external harness, and run `ssh rvv`; it must not
  implement compiler IR semantics, lowering, route selection, fallback compute,
  or generated compute bodies.
- The runtime proof is bounded to the existing i32m1 add/sub/mul vector-source
  front-door fixtures and runtime counts that exercise the AVL/VL loop.
- If the compiler route is wrong, repair C++/MLIR route code. Do not add
  script-side compatibility compute or restore deleted self-check/direct-C
  behavior.

## Requirements

- Keep bundle generation on:

```text
build/bin/tcrv-translate --tcrv-source-artifact-bundle-front-door \
  --tcrv-target-artifact-bundle-output-dir=<bundle-dir> <source.mlir>
```

- Verify each add/sub/mul bundle ties the object and header to the same
  selected variant, dispatch role, route family, component group, external ABI
  name, runtime ABI kind/name, ordered ABI parameters, materialized EmitC
  provenance, arithmetic op, and RVV runtime AVL/VL metadata.
- Verify generated headers are declaration-only external C ABI surfaces:
  C++ `extern "C"` guards, expected prototype, no callable body, no RVV
  intrinsic body, no hidden `main`, no self-check helper, no raw log or
  credential text, and no descriptor/direct-C/source-export residue.
- Verify generated objects are RISC-V relocatable ELF objects exposing the
  unmangled selected C symbol and not only a C++-mangled symbol.
- Generate an external C harness that includes only the generated header, links
  only the generated object plus normal target toolchain libraries, executes
  bounded add/sub/mul runtime cases on `ssh rvv`, and records explicit PASS
  markers.
- Local negative coverage must fail closed for missing/mismatched bundle index,
  missing header, missing object, header/object symbol mismatch, missing or
  misordered runtime ABI parameters, stale selected variant, stale route,
  stale runtime ABI identity, stale arithmetic metadata, forbidden descriptor/
  direct-C/source-export/self-check metadata, and sanitizer redaction issues.
- Preserve focused evidence under `artifacts/tmp` only.

## Acceptance Criteria

- [x] The public translate command generates coherent add/sub/mul bundles for
  the existing RVV source-front-door fixtures.
- [x] Local bundle verification checks object/header/index coherence, runtime
  ABI identity, ordered ABI parameters, materialized EmitC provenance, RVV
  AVL/VL metadata, and unmangled object symbols.
- [x] Header verification rejects callable-body residue, RVV intrinsic residue,
  hidden `main`, self-check helper text, raw logs/credentials, descriptor,
  direct-C, and source-export residue.
- [x] Negative self-tests fail closed for missing/mismatched bundle metadata,
  stale route/runtime ABI/variant/arithmetic data, forbidden metadata residue,
  and sanitizer redaction issues.
- [x] Real `ssh rvv` evidence compiles, links, and runs external C harnesses
  for add/sub/mul and prints explicit PASS markers for bounded `n` cases.
- [x] Focused local lit/self-test coverage passes for the changed evidence
  surfaces.
- [x] Focused scans over touched files show no restored BinarySelfCheck,
  self-check route authority, descriptor-driven computation, direct-C semantic
  exporter, source-export route, scalar fallback compute, or Python
  compiler-core behavior.
- [x] Trellis task status, journal, archive, and final report truthfully
  distinguish fresh current-HEAD proof from older archived evidence.

## Completion Evidence

- Strengthened `scripts/rvv_generated_bundle_abi_e2e.py` so bundle index and
  artifact metadata verification rejects forbidden descriptor, direct-C,
  source-export, self-check, raw-log, and credential-like residue.
- Strengthened generated header verification with case-insensitive public
  residue rejection, including credential-like text such as `TOKEN=...`.
- Strengthened remote runtime verification so each `ssh rvv` run must contain
  both the operation-specific marker and explicit `PASS op=<kind>` output.
- Added self-test negative coverage for descriptor metadata residue,
  direct-C/source-export metadata residue, header self-check residue, header
  credential residue, and sanitizer redaction of private key, authorization
  bearer, environment token, and password-like text.
- Dry-run evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-rvv-generated-bundle-external-c-abi-dry`.
- Real RVV evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-rvv-generated-bundle-external-c-abi-ssh`.
- `ssh rvv` compiled/linked with `/usr/bin/clang`, reported
  `remote_arch=riscv64`, `clang_version=Ubuntu clang version 18.1.3
  (1ubuntu1)`, and ran add/sub/mul for `n=1,7,16,17,257`.
- Remote PASS markers observed:
  `tcrv_rvv_generated_bundle_abi_add_ok`,
  `tcrv_rvv_generated_bundle_abi_sub_ok`,
  `tcrv_rvv_generated_bundle_abi_mul_ok`,
  `PASS op=add counts=1,7,16,17,257`,
  `PASS op=sub counts=1,7,16,17,257`, and
  `PASS op=mul counts=1,7,16,17,257`.

## Checks

- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [x] `cmake --build build --target tcrv-translate -j2`
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --run-id codex-rvv-generated-bundle-external-c-abi-dry --overwrite --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --run-id codex-rvv-generated-bundle-external-c-abi-ssh --overwrite --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10`
- [x] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Scripts/rvv-generated-bundle-abi-e2e-self-test.test Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-rvv.mlir Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-fail-closed.mlir`
- [x] Focused residue scan:
  `rg -n "BinarySelfCheck|runtime-callable-c-source|source-export|source_export|direct-C|direct_c|descriptor|self-check|self_check" scripts/rvv_generated_bundle_abi_e2e.py include/TianChenRV/Target lib/Target tools/tcrv-translate test/Target`
  -> remaining matches are intentional fail-closed rejection logic, negative
  self-test fixtures, and existing `implicit-check-not` / C++ rejection
  coverage; no restored route authority was added.
- [x] `git diff --check`
- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-17-rvv-generated-bundle-external-c-abi-proof`

## Out of Scope

- No new RVV SEW/LMUL families.
- No new arithmetic beyond add/sub/mul.
- No performance claims or generic RVV backend claims.
- No broad smoke matrix.
- No new front-door command, descriptor adapter, direct C semantic exporter,
  source-export route, scalar fallback runtime compute, compatibility alias,
  self-check harness, hidden executable bridge, or Python compiler-core logic.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/extension-plugins/rvv-plugin.md`.
- Previous PRD read:
  `.trellis/tasks/archive/2026-05/05-17-binary-self-check-artifact-residue-erasure/prd.md`.
- Prior related proof PRD read:
  `.trellis/tasks/archive/2026-05/05-17-05-17-rvv-source-bundle-runtime-abi-proof/prd.md`.
- Implementation/evidence files inspected:
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-self-test.test`,
  `test/Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-rvv.mlir`,
  `test/Target/RVV/vector-materialized-target-artifact-exporters.mlir`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `lib/Target/TargetArtifactExport.cpp`, and
  `tools/tcrv-translate/tcrv-translate.cpp`.

## Definition of Done

- Focused implementation is complete or the exact blocker is recorded.
- Focused local checks and `ssh rvv` proof are recorded with artifact paths.
- Trellis context validates.
- Task is finished/archived if acceptance criteria are met.
- One coherent commit is created if the round completes.
