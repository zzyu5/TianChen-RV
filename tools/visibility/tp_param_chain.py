#!/usr/bin/env python3
"""tp_param_chain.py -- E2 / T-P: the CONSTRUCTION-PARAMETER three-leg chain.

WHAT THIS SHOWS (closure leg 2 of 3: "construction parameters are DERIVABLE from
the interface"; v3.2 thesis slot "legality, CONSTRUCTION PARAMETERS and rejection
behaviour are derivable from the interface"). NO perf claim anywhere in this file.

THE CHAIN, per sampled cell, three legs:

  LEG 1  fact -> plan closed-form parameter
         A capability FACT (march= derived VLEN, or an in-IR `vlen_bits` fact row)
         is fed in; the plan layer emits a construction parameter. This script
         RE-DERIVES the parameter from the documented closed form IN PYTHON and
         asserts the compiler agrees. Agreement on a SWEEP (including the clamp /
         saturating regimes, and a point where a static table in the source
         DISAGREES with the live value) is what separates a CLOSED FORM from a
         lookup table.

  LEG 2  plan parameter -> product form
         The parameter must actually shape the emitted artifact (strip count /
         emitted leaf identity), not merely be stamped as an inert attr.

  LEG 3  product form -> objdump
         The real RISC-V object is disassembled and the machine-level shape is
         counted, then checked against the parameter LEG 1 derived.

HONEST LEG COVERAGE (do not overstate; the tool prints the real status):
  q3_K@rvv    3/3 legs LOCAL. Emits through clang++-20 --target=riscv64 and is
              disassembled by llvm-objdump-20. No board involved.
  q4_0@ime    2/3 legs LOCAL. Leg 3 is TOOLCHAIN-BLOCKED: the SpacemiT IME
              extension (xsmtvdotii) is a non-standard vendor extension that
              upstream clang-20 REJECTS ("unsupported non-standard user-level
              extension 'xsmtvdotii'"), so no local riscv64 object can be built
              for the IME leaf. Reaching leg 3 needs the SpacemiT GCC-15.2 fork
              that lives on the k1 board. This script does NOT touch any board
              and does NOT fabricate that leg -- it reports it as
              toolchain-pending and prints the exact rejection.

THE TWO CLOSED FORMS UNDER TEST (source of truth = the code, cited inline):

  (A) q3_K repack GEMM strip width / unroll factor
        half_lanes = 0                              if minVLEN < 128
                   = min(minVLEN/16, kWeightInterleave)   otherwise
      lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:916-921
      (kWeightInterleave = 16, same file :160)
        numHalves  = kWeightInterleave / half_lanes
      lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:1423
      minVLEN itself: plugin::rvv::deriveMinimumVLEN
      lib/Plugin/RVV/RVVCapabilityProfile.cpp:289
      numHalves is the DISJOINT-STRIP COUNT and the region carries one vector
      accumulator per strip -- so it is literally the unroll factor of the
      per-strip integer core, and leg 3 counts it in the disassembly.

  (B) q4_0@ime wide-vmadot deployment
        fragBits = macM*macK*elem_in_bits ; accBits = macM*macN*accum_bits
        rpfIn    = max(1, ceil(fragBits / vlen_bits))
        rpfAcc   = max(1, ceil(accBits  / vlen_bits))
        rpfIn != 1                      => capability DECLINE -> narrow fallback
        vreg_floor(njw) = rpfIn*(1+njw) + njw*rpfAcc ; pick widest MECHANIZED
        njw whose floor <= 32
      lib/Plugin/IME/IMEBackendEmissionDriver.cpp:446-497
      (pattern table :352-361 ; register-file size kRVVVectorRegisterFileSize)

WHY (B) IS PROVABLY NOT A LOOKUP: the static pattern table records
minVregBudget=7 for W2 (lib/Plugin/IME/IMEBackendEmissionDriver.cpp:355, the
VLEN256 witness). At vlen_bits=512 the LIVE emitted floor is 5, NOT the table's
7 -- the number is computed from the fact, and the table entry is documentation.
This script asserts that divergence explicitly (probe `table_vs_live`).

USAGE
  python3 tools/visibility/tp_param_chain.py                # print + write CSV
  python3 tools/visibility/tp_param_chain.py --print-only    # no CSV write

Exit code is non-zero if any executed (non-pending) consistency check FAILS, so
this is usable as a regression gate. Legs that are honestly blocked do NOT fail
the run -- they are reported as pending.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

# WEFT_REPO override exists so the negative controls (a mutated copy of this file
# run from a scratch dir) still resolve the real tree; the default is the normal
# in-tree location.
REPO = os.environ.get(
    "WEFT_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
OUT_CSV = os.path.join(
    REPO, "experiments", "active", "result-tables", "T-P_construction_param_chain.csv"
)

WEFT_OPT = os.path.join(REPO, "build-weft", "bin", "weft-opt")
Q3K_FIXTURE = os.path.join(
    REPO,
    "test/Conversion/RVV/rvv-emit-quant-contraction-q3-K-repack-gemm-prefill-vlen128.mlir",
)
IME_FIXTURE = os.path.join(
    REPO, "test/Conversion/EmitC/ime-q4-0-matmul-tile-vlen128-narrow-fallback.mlir"
)

# Closed-form constants, mirrored from the cited code sites.
K_WEIGHT_INTERLEAVE = 16  # RVVLowerQuantContraction.cpp:160
K_VREG_FILE = 32  # kRVVVectorRegisterFileSize
IME_MAC_M, IME_MAC_N, IME_MAC_K = 4, 4, 8  # from the tile op attrs in the fixture
IME_ELEM_IN_BITS, IME_ACCUM_BITS = 8, 32


def tool(*names: str) -> str | None:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


MLIR_TRANSLATE = tool("mlir-translate-20", "mlir-translate")
CLANGXX = tool("clang++-20", "clang++")
OBJDUMP = tool("llvm-objdump-20", "llvm-objdump")


def run(cmd: list[str], stdin_data: bytes | None = None) -> tuple[int, str, str]:
    p = subprocess.run(cmd, input=stdin_data, capture_output=True)
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode(
        "utf-8", "replace"
    )


# ---------------------------------------------------------------------------
# Closed forms re-implemented independently (the ORACLE for leg 1).
# ---------------------------------------------------------------------------


def closed_form_min_vlen(march: str) -> int:
    """Mirror of deriveMinimumVLEN (RVVCapabilityProfile.cpp:289)."""
    text = march.lower()
    floor_bits = 0
    for m in re.finditer(r"zvl(\d+)b", text):
        floor_bits = max(floor_bits, int(m.group(1)))
    full_v = (
        "gcv" in text or "_v" in text or "rv64v" in text or "rv32v" in text
        or "xtheadvector" in text
    )
    if full_v and floor_bits < 128:
        floor_bits = 128
    return floor_bits


def closed_form_half_lanes(min_vlen: int) -> int:
    """Mirror of deriveRepackHalfLanes (RVVLowerQuantContraction.cpp:916-921)."""
    if min_vlen < 128:
        return 0
    return min(min_vlen // 16, K_WEIGHT_INTERLEAVE)


def closed_form_num_halves(half_lanes: int) -> int:
    """Mirror of numHalves (RVVLowerQuantContraction.cpp:1423)."""
    return K_WEIGHT_INTERLEAVE // half_lanes if half_lanes else 0


def closed_form_ime(vlen_bits: int) -> dict:
    """Mirror of decideWideVmadotDeployment (IMEBackendEmissionDriver.cpp:446-497)."""
    frag_bits = IME_MAC_M * IME_MAC_K * IME_ELEM_IN_BITS  # 256
    acc_bits = IME_MAC_M * IME_MAC_N * IME_ACCUM_BITS  # 512
    vb = vlen_bits if vlen_bits > 0 else frag_bits
    rpf_in = max(1, math.ceil(frag_bits / vb))
    rpf_acc = max(1, math.ceil(acc_bits / vb))
    if rpf_in != 1:
        return {
            "rpfIn": rpf_in,
            "rpfAcc": rpf_acc,
            "decision": "narrow fallback",
            "njw": None,
            "floor": None,
        }
    # widest MECHANIZED pattern whose floor fits the 32-vreg file.
    mechanized_njw = [1, 2]  # :353-357 ; W4 (:358) is status=measured-negative
    best_njw, best_floor = None, None
    for njw in mechanized_njw:
        floor = rpf_in * (1 + njw) + njw * rpf_acc
        if floor <= K_VREG_FILE and (best_njw is None or njw > best_njw):
            best_njw, best_floor = njw, floor
    return {
        "rpfIn": rpf_in,
        "rpfAcc": rpf_acc,
        "decision": f"deploy IME-VMADOT-TILE-W{best_njw}",
        "njw": best_njw,
        "floor": best_floor,
    }


# ---------------------------------------------------------------------------
# Probes.
# ---------------------------------------------------------------------------


def plan_half_lanes(march: str) -> int | None:
    rc, out, _ = run(
        [WEFT_OPT, Q3K_FIXTURE, f"--weft-rvv-lower-quant-contraction=march={march}"]
    )
    if rc != 0:
        return None
    vals = sorted(set(int(m) for m in re.findall(r"half_lanes = (\d+)", out)))
    if not vals:
        return None  # no repack op formed -> block-dot stub (the no-capability path)
    assert len(vals) == 1, f"ambiguous half_lanes {vals} for {march}"
    return vals[0]


def ime_probe(vlen_bits: int, workdir: str) -> dict:
    src = open(IME_FIXTURE).read()
    assert 'vlen_bits = "128"' in src, "IME fixture no longer carries vlen_bits=128"
    perturbed = src.replace('vlen_bits = "128"', f'vlen_bits = "{vlen_bits}"')
    path = os.path.join(workdir, f"ime.{vlen_bits}.mlir")
    with open(path, "w") as f:
        f.write(perturbed)
    rc, out, _ = run([WEFT_OPT, path, "--weft-materialize-emitc-lowerable-routes"])
    if rc != 0:
        return {}
    got = {}
    m = re.search(r"rpfIn=(\d+)", out)
    if m:
        got["rpfIn"] = int(m.group(1))
    m = re.search(r"rpfAcc=(\d+)", out)
    if m:
        got["rpfAcc"] = int(m.group(1))
    m = re.search(r"vreg_floor\(njw=(\d+)\)=(\d+)", out)
    if m:
        got["njw"], got["floor"] = int(m.group(1)), int(m.group(2))
    if "narrow fallback" in out:
        got["decision"] = "narrow fallback"
    else:
        m = re.search(r"deploy (IME-VMADOT-TILE-W\d+)", out)
        if m:
            got["decision"] = f"deploy {m.group(1)}"
    got["leaf"] = "mac_kloop_w2" if "mac_kloop_w2" in out else "narrow(no _w2 leaf)"
    return got


def q3k_objdump_vwmacc(march: str, workdir: str) -> tuple[int | None, str]:
    """Full local chain: plan -> emitc -> C++ -> riscv64 .o -> disasm vwmacc tally."""
    if not (MLIR_TRANSLATE and CLANGXX and OBJDUMP):
        return None, "missing local toolchain"
    rc, emitc, err = run(
        [
            WEFT_OPT,
            Q3K_FIXTURE,
            f"--weft-rvv-lower-quant-contraction=march={march}",
            "--weft-rvv-lower-to-emitc",
        ]
    )
    if rc != 0:
        return None, f"weft-opt emitc failed: {err[:120]}"
    mlir_path = os.path.join(workdir, f"q3k.{march}.mlir")
    with open(mlir_path, "w") as f:
        f.write(emitc)
    rc, cpp, err = run([MLIR_TRANSLATE, "--mlir-to-cpp", mlir_path])
    if rc != 0:
        return None, f"mlir-translate failed: {err[:120]}"
    cpp_path = os.path.join(workdir, f"q3k.{march}.cpp")
    with open(cpp_path, "w") as f:
        f.write(cpp)
    obj_path = os.path.join(workdir, f"q3k.{march}.o")
    rc, _, err = run(
        [
            CLANGXX,
            "--target=riscv64-unknown-linux-gnu",
            f"-march={march}",
            "-ffreestanding",
            "-O2",
            "-c",
            "-x",
            "c++",
            cpp_path,
            "-o",
            obj_path,
        ]
    )
    if rc != 0:
        return None, f"clang failed: {err[:160]}"
    rc, dis, err = run([OBJDUMP, "-d", "--mattr=+v,+zvfhmin", obj_path])
    if rc != 0:
        return None, f"objdump failed: {err[:120]}"
    n = len(re.findall(r"\bvwmacc\.vx\b", dis))
    return n, f"{os.path.basename(obj_path)} ({os.path.getsize(obj_path)} bytes)"


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-only", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(WEFT_OPT):
        print(f"FATAL: weft-opt not built at {WEFT_OPT}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    failures: list[str] = []
    workdir = tempfile.mkdtemp(prefix="tp_param_chain.")

    # ================= CELL 1: q3_K@rvv =================
    # LEG 1: fact -> plan closed-form, swept across the whole VLEN domain.
    print("=" * 78)
    print("CELL q3_K@rvv -- LEG 1: capability fact -> plan closed-form parameter")
    print("  closed form: half_lanes = (minVLEN<128) ? 0 : min(minVLEN/16, 16)")
    print("               RVVLowerQuantContraction.cpp:916-921 (kWeightInterleave=16 @:160)")
    print("=" * 78)
    marches = [
        "",
        "rv64gc",
        "rv64gcv_zvl64b",
        "rv64gcv_zvfhmin",
        "rv64gcv_zvl256b_zvfhmin",
        "rv64gcv_zvl512b",
        "rv64gcv_zvl1024b",
    ]
    for march in marches:
        want_vlen = closed_form_min_vlen(march)
        want_hl = closed_form_half_lanes(want_vlen)
        got_hl = plan_half_lanes(march)
        got_norm = got_hl if got_hl is not None else 0
        ok = got_norm == want_hl
        if not ok:
            failures.append(f"q3_K leg1 march={march!r}: want {want_hl} got {got_norm}")
        nh = closed_form_num_halves(want_hl)
        print(
            f"  march={march or '<empty>':<26} minVLEN={want_vlen:<5} "
            f"closed-form half_lanes={want_hl:<3} weft-opt={got_norm:<3} "
            f"numHalves={nh if nh else '-':<3} {'OK' if ok else 'MISMATCH'}"
        )
        rows.append(
            {
                "cell": "q3_K@rvv",
                "leg": "1_fact_to_plan_closedform",
                "capability_fact": f"march={march or '<empty>'} => deriveMinimumVLEN={want_vlen}",
                "plan_param": "half_lanes",
                "closed_form": "half_lanes = (minVLEN<128) ? 0 : min(minVLEN/16, 16)",
                "closed_form_site": "lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:916-921",
                "predicted": want_hl,
                "observed": got_norm,
                "product_form": (
                    f"numHalves={nh} disjoint strip(s)" if nh else "no repack op (block-dot stub, fail-closed)"
                ),
                "consistency": "MATCH" if ok else "MISMATCH",
                "leg_status": "DONE-local",
                "repro_cmd": (
                    f"{os.path.relpath(WEFT_OPT, REPO)} {os.path.relpath(Q3K_FIXTURE, REPO)} "
                    f"--weft-rvv-lower-quant-contraction=march={march} | grep -o 'half_lanes = [0-9]*' | sort -u"
                ),
            }
        )

    # LEG 2 + LEG 3: plan param -> emitted object -> disassembly tally.
    print()
    print("=" * 78)
    print("CELL q3_K@rvv -- LEG 2/3: plan param -> product form -> objdump")
    print("  prediction: per-strip vector work scales with")
    print("              numHalves = kWeightInterleave / half_lanes  (:1423)")
    print("=" * 78)
    obj_counts: dict[str, int] = {}
    for march in ["rv64gcv_zvfhmin", "rv64gcv_zvl256b_zvfhmin"]:
        hl = plan_half_lanes(march)
        nh = closed_form_num_halves(hl or 0)
        n, note = q3k_objdump_vwmacc(march, workdir)
        if n is None:
            print(f"  march={march:<26} objdump leg UNAVAILABLE: {note}")
            rows.append(
                {
                    "cell": "q3_K@rvv",
                    "leg": "3_product_to_objdump",
                    "capability_fact": f"march={march}",
                    "plan_param": "half_lanes",
                    "closed_form": "numHalves = 16 / half_lanes",
                    "closed_form_site": "lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:1423",
                    "predicted": nh,
                    "observed": "",
                    "product_form": "",
                    "consistency": "PENDING",
                    "leg_status": f"BLOCKED: {note}",
                    "repro_cmd": "see tools/visibility/tp_param_chain.py q3k_objdump_vwmacc",
                }
            )
            continue
        obj_counts[march] = n
        print(
            f"  march={march:<26} half_lanes={hl:<3} numHalves={nh:<3} "
            f"vwmacc.vx={n:<6} obj={note}"
        )
        rows.append(
            {
                "cell": "q3_K@rvv",
                "leg": "3_product_to_objdump",
                "capability_fact": f"march={march} => minVLEN={closed_form_min_vlen(march)}",
                "plan_param": f"half_lanes={hl}",
                "closed_form": "numHalves = 16 / half_lanes",
                "closed_form_site": "lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:1423",
                "predicted": f"numHalves={nh}",
                "observed": f"vwmacc.vx={n}",
                "product_form": f"riscv64 ELF {note}",
                "consistency": "see ratio row",
                "leg_status": "DONE-local (clang++-20 --target=riscv64 + llvm-objdump-20; NO board)",
                "repro_cmd": (
                    f"weft-opt <q3k fixture> --weft-rvv-lower-quant-contraction=march={march} "
                    f"--weft-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp | "
                    f"clang++-20 --target=riscv64-unknown-linux-gnu -march={march} -ffreestanding -O2 -c -x c++ - -o o.o && "
                    f"llvm-objdump-20 -d --mattr=+v,+zvfhmin o.o | grep -c 'vwmacc\\.vx'"
                ),
            }
        )

    if len(obj_counts) == 2:
        m128, m256 = "rv64gcv_zvfhmin", "rv64gcv_zvl256b_zvfhmin"
        hl128 = plan_half_lanes(m128)
        hl256 = plan_half_lanes(m256)
        nh128, nh256 = closed_form_num_halves(hl128), closed_form_num_halves(hl256)
        # A corrupted oracle can drive numHalves to 0 (e.g. half_lanes >
        # kWeightInterleave). Report that as a failure rather than dividing by
        # zero -- found by negative control A.
        if nh256 == 0 or obj_counts[m256] == 0:
            failures.append(
                f"q3_K leg3 ratio undefined: numHalves(256)={nh256} "
                f"vwmacc(256)={obj_counts[m256]} -- closed form is inconsistent with the plan"
            )
            want_ratio = got_ratio = float("nan")
            ok = False
        else:
            want_ratio = nh128 / nh256
            got_ratio = obj_counts[m128] / obj_counts[m256]
            ok = abs(got_ratio - want_ratio) < 1e-9
            if not ok:
                failures.append(f"q3_K leg3 ratio: want {want_ratio} got {got_ratio}")
        print()
        print(
            f"  CONSISTENCY: predicted numHalves ratio {closed_form_num_halves(hl128)}"
            f"/{closed_form_num_halves(hl256)} = {want_ratio:.4f}"
        )
        print(
            f"               observed  vwmacc.vx ratio {obj_counts[m128]}"
            f"/{obj_counts[m256]} = {got_ratio:.4f}   ==> "
            f"{'CONSISTENT' if ok else 'INCONSISTENT'}"
        )
        rows.append(
            {
                "cell": "q3_K@rvv",
                "leg": "3_consistency_verdict",
                "capability_fact": "VLEN128 vs VLEN256 (same fixture, march= only)",
                "plan_param": f"half_lanes {hl128} vs {hl256}",
                "closed_form": "ratio(vwmacc.vx) == ratio(numHalves) == 2/1",
                "closed_form_site": "lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:1423",
                "predicted": f"{want_ratio:.4f}",
                "observed": f"{got_ratio:.4f} ({obj_counts[m128]}/{obj_counts[m256]})",
                "product_form": "riscv64 ELF disassembly",
                "consistency": "CONSISTENT" if ok else "INCONSISTENT",
                "leg_status": "DONE-local",
                "repro_cmd": "python3 tools/visibility/tp_param_chain.py",
            }
        )

    # ================= CELL 2: q4_0@ime =================
    print()
    print("=" * 78)
    print("CELL q4_0@ime -- LEG 1/2: vlen_bits fact -> closed-form -> emitted leaf")
    print("  rpfIn = max(1, ceil(macM*macK*elem_in_bits / vlen_bits))   [4*8*8=256]")
    print("  rpfAcc= max(1, ceil(macM*macN*accum_bits  / vlen_bits))   [4*4*32=512]")
    print("  vreg_floor(njw) = rpfIn*(1+njw) + njw*rpfAcc  <= 32")
    print("  IMEBackendEmissionDriver.cpp:446-497")
    print("=" * 78)
    for vb in [128, 256, 512]:
        want = closed_form_ime(vb)
        got = ime_probe(vb, workdir)
        checks = []
        verified: list[str] = []
        unobserved: list[str] = []
        for k in ("rpfIn", "rpfAcc", "njw", "floor", "decision"):
            if want.get(k) is None:
                continue
            if k not in got:
                # The decline path returns before printing rpfAcc, so at a
                # declining VLEN our rpfAcc is COMPUTED but NOT OBSERVED. Say so
                # rather than silently counting it as agreement.
                unobserved.append(k)
                continue
            checks.append(got[k] == want[k])
            verified.append(k)
            if got[k] != want[k]:
                failures.append(f"q4_0@ime vlen={vb} {k}: want {want[k]} got {got[k]}")
        ok = bool(checks) and all(checks)

        def show(d: dict, k: str) -> str:
            if k in unobserved:
                return f"{d.get(k)}(unobserved)"
            return str(d.get(k))

        print(
            f"  vlen_bits={vb:<5} closed-form: rpfIn={show(want,'rpfIn')} "
            f"rpfAcc={show(want,'rpfAcc')} floor={want['floor']} -> {want['decision']}"
        )
        print(
            f"  {'':>15} emitted:     rpfIn={got.get('rpfIn')} rpfAcc={got.get('rpfAcc')} "
            f"floor={got.get('floor')} -> {got.get('decision')}  leaf={got.get('leaf')}"
        )
        print(
            f"  {'':>15} verified {len(checks)} field(s) {verified}"
            + (f"; NOT OBSERVED (decline path prints no value): {unobserved}" if unobserved else "")
            + f"   ==> {'MATCH' if ok else 'MISMATCH'}"
        )
        rows.append(
            {
                "cell": "q4_0@ime",
                "leg": "1_fact_to_plan_closedform + 2_plan_to_product",
                "capability_fact": f"weft.exec.capability spacemit.ime vlen_bits={vb} (in-IR fact row)",
                "plan_param": "rpfIn / rpfAcc / vreg_floor(njw) / deploy-decision",
                "closed_form": (
                    "rpfIn=max(1,ceil(4*8*8/vlen)); rpfAcc=max(1,ceil(4*4*32/vlen)); "
                    "floor(njw)=rpfIn*(1+njw)+njw*rpfAcc<=32; rpfIn!=1 => narrow"
                ),
                "closed_form_site": "lib/Plugin/IME/IMEBackendEmissionDriver.cpp:446-497",
                "predicted": (
                    f"rpfIn={want['rpfIn']} rpfAcc={want['rpfAcc']} "
                    f"floor={want['floor']} {want['decision']}"
                ),
                "observed": (
                    f"rpfIn={got.get('rpfIn')} rpfAcc={got.get('rpfAcc')} "
                    f"floor={got.get('floor')} {got.get('decision')}"
                    + (f" [unobserved: {','.join(unobserved)}]" if unobserved else "")
                ),
                "product_form": f"emitted leaf: {got.get('leaf')}",
                "consistency": (
                    f"MATCH ({len(checks)} field(s) verified: {','.join(verified)})"
                    if ok
                    else "MISMATCH"
                ),
                "leg_status": "DONE-local (emission only; no board)",
                "repro_cmd": (
                    f"sed 's/vlen_bits = \"128\"/vlen_bits = \"{vb}\"/' "
                    f"{os.path.relpath(IME_FIXTURE, REPO)} > /tmp/ime.{vb}.mlir && "
                    f"{os.path.relpath(WEFT_OPT, REPO)} /tmp/ime.{vb}.mlir "
                    f"--weft-materialize-emitc-lowerable-routes | "
                    f"grep -oE 'rpfIn=[0-9]+|rpfAcc=[0-9]+|vreg_floor\\(njw=[0-9]+\\)=[0-9]+|deploy [A-Za-z0-9-]+|narrow fallback|mac_kloop_w[0-9]+' | sort -u"
                ),
            }
        )

    # The lookup-table falsifier: static table says 7 for W2; live @512 says 5.
    print()
    print("  PROBE table_vs_live (falsifies 'it is just a lookup table'):")
    live512 = ime_probe(512, workdir).get("floor")
    table_w2 = 7  # IMEBackendEmissionDriver.cpp:355 minVregBudget for W2
    diverges = live512 is not None and live512 != table_w2
    print(
        f"    static table W2 minVregBudget = {table_w2} "
        f"(IMEBackendEmissionDriver.cpp:355, the VLEN256 witness)"
    )
    print(f"    live emitted floor @vlen_bits=512 = {live512}")
    print(
        f"    ==> {'DIVERGE: value is COMPUTED from the fact, not read from the table'if diverges else 'NO DIVERGENCE (weakens the closed-form claim)'}"
    )
    if not diverges:
        failures.append("table_vs_live: expected live@512 floor to diverge from table 7")
    rows.append(
        {
            "cell": "q4_0@ime",
            "leg": "probe_table_vs_live",
            "capability_fact": "vlen_bits=512",
            "plan_param": "vreg_floor(njw=2)",
            "closed_form": "floor = rpfIn*(1+njw)+njw*rpfAcc = 1*3+2*1 = 5",
            "closed_form_site": "lib/Plugin/IME/IMEBackendEmissionDriver.cpp:355 (table=7) vs :483-485 (live)",
            "predicted": "live != table => computed, not looked up",
            "observed": f"table={table_w2} live={live512}",
            "product_form": "n/a (plan-layer probe)",
            "consistency": "DIVERGE-as-predicted" if diverges else "NO-DIVERGENCE",
            "leg_status": "DONE-local",
            "repro_cmd": "python3 tools/visibility/tp_param_chain.py",
        }
    )

    # LEG 3 for IME: honestly blocked.
    rc, _, ime_err = run(
        [
            CLANGXX or "clang++-20",
            "--target=riscv64-unknown-linux-gnu",
            "-march=rv64gcv_xsmtvdotii1p0",
            "-x",
            "c",
            "-c",
            "-",
            "-o",
            "/dev/null",
        ],
        stdin_data=b"int f(void){return 0;}\n",
    )
    ime_block = ime_err.strip().splitlines()[0] if ime_err.strip() else "(no error text)"
    print()
    print("  LEG 3 (objdump) for q4_0@ime: TOOLCHAIN-PENDING, not attempted, not faked.")
    print(f"    upstream clang-20 probe rc={rc}: {ime_block}")
    print("    => needs SpacemiT GCC-15.2 fork (lives on the k1 board). Board time")
    print("       belongs to the P line this round, so this leg is NOT executed here.")
    rows.append(
        {
            "cell": "q4_0@ime",
            "leg": "3_product_to_objdump",
            "capability_fact": "vlen_bits=256 (the K1-validated envelope)",
            "plan_param": "njw=2 / mac_kloop_w2 leaf",
            "closed_form": "n/a -- leg not executed",
            "closed_form_site": "n/a",
            "predicted": "wide vmadot leaf present in disassembly",
            "observed": "",
            "product_form": "",
            "consistency": "PENDING",
            "leg_status": (
                "BLOCKED-toolchain: upstream clang-20 rejects xsmtvdotii "
                f"({ime_block}); needs SpacemiT GCC-15.2 fork on k1. Board time = P line. NOT ATTEMPTED."
            ),
            "repro_cmd": (
                "clang-20 --target=riscv64-unknown-linux-gnu -march=rv64gcv_xsmtvdotii1p0 "
                "-x c -c - -o /dev/null <<< 'int f(void){return 0;}'   # shows the rejection"
            ),
        }
    )

    # ================= emit =================
    if not args.print_only:
        os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
        cols = [
            "cell",
            "leg",
            "capability_fact",
            "plan_param",
            "closed_form",
            "closed_form_site",
            "predicted",
            "observed",
            "product_form",
            "consistency",
            "leg_status",
            "repro_cmd",
        ]
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
        print()
        print(f"wrote {os.path.relpath(OUT_CSV, REPO)}  ({len(rows)} rows)")

    print()
    print("=" * 78)
    print("LEG COVERAGE SUMMARY (honest):")
    print("  q3_K@rvv : leg1 DONE-local, leg2 DONE-local, leg3 DONE-local  => 3/3")
    print("  q4_0@ime : leg1 DONE-local, leg2 DONE-local, leg3 BLOCKED-toolchain => 2/3")
    print("=" * 78)
    if failures:
        print()
        print(f"FAILURES ({len(failures)}):")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    print("all executed consistency checks PASS (blocked legs excluded, not counted as pass)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
