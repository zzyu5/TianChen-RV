#include "Weft/Plugin/IME/IMEBackendEmissionDriver.h"

#include "Weft/Conversion/EmitC/BackendEmissionRegistry.h"
#include "Weft/Conversion/EmitC/TypedBackendEmissionDriver.h"
#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/IME/IR/IMEDialect.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>

namespace weft {
namespace plugin {
namespace ime {

namespace {

namespace emitc = ::mlir::emitc;
namespace weftemitc = ::weft::conversion::emitc;

constexpr llvm::StringLiteral kOpInterface = "WEFTEmitCLowerableOpInterface";
// The self-contained asm-leaf helper name. ALL dataflow is structured emitc;
// this helper holds the SINGLE justified instruction leaf (the `vmadot` asm) —
// there is no IME intrinsic header, so per the design note we claim ONE leaf,
// not raw()==0. Reached only by a structured emitc.call_opaque on the A/B/C
// pointer block-args of the wrapping emitc.func (no SSA name is interpolated
// into the asm text; the helper has its own fixed parameter names).
constexpr llvm::StringLiteral kVmadotHelperName("weft_ime_vmadot_mma_4x4x8");
// The SECOND (unsigned) asm-leaf helper — same structure, but the single
// justified instruction leaf is `vmadotu` (unsigned*unsigned int8 MAC), a
// GENUINELY different instruction (encoding 0xe210012b vs vmadot's 0xe210312b)
// with unsigned numeric semantics. There is no IME intrinsic header, so this is
// likewise ONE justified verbatim leaf reached by a structured call_opaque.
constexpr llvm::StringLiteral kVmadotuHelperName("weft_ime_vmadotu_mma_4x4x8");
// The FOURTH (mixed-sign) asm-leaf helper — same structure, but the single
// justified instruction leaf is `vmadotsu` (signed*unsigned int8 MAC), a
// GENUINELY different instruction (encoding 0xe210212b vs vmadot's 0xe210312b
// and vmadotu's 0xe210012b) with mixed-sign numeric semantics (signed A,
// unsigned B). There is no IME intrinsic header, so this is likewise ONE
// justified verbatim leaf reached by a structured call_opaque.
constexpr llvm::StringLiteral kVmadotsuHelperName(
    "weft_ime_vmadotsu_mma_4x4x8");
// The SIXTH IME op: the reversed-order mixed-sign asm-leaf helper — same
// structure, but the single justified instruction leaf is `vmadotus` (unsigned A
// * signed B int8 MAC), a GENUINELY different instruction (encoding 0xe210112b
// vs vmadotsu's 0xe210212b, vmadot's 0xe210312b and vmadotu's 0xe210012b) with
// the OTHER mixed-sign numeric semantics. There is no IME intrinsic header, so
// this is likewise ONE justified verbatim leaf reached by a structured
// call_opaque. This completes the signedness family.
constexpr llvm::StringLiteral kVmadotusHelperName(
    "weft_ime_vmadotus_mma_4x4x8");
// The FIFTH IME op: the sliding-window MAC. The justified instruction leaf is
// `vmadot1`/`vmadot2`/`vmadot3` (funct7 111001, e6..., DISTINCT from the
// non-slide 111000/e2...). A genuinely different kernel SHAPE: A is an EVEN
// VS1:VS1+1 PAIR (v0:v1 = 8x8 int8, 64B), the slide window shifts the A read
// down by `slide` rows. vmadot1 v4,v0,v2 = 0xe620322b (vs vmadot v4,v0,v2 =
// 0xe220322b). Confirmed bit-exact on real K1 by a 4-way window discriminator.
constexpr llvm::StringLiteral kVmadot1SlideHelperName(
    "weft_ime_vmadot1_mma_slide_4x4x8");
constexpr llvm::StringLiteral kVmadot2SlideHelperName(
    "weft_ime_vmadot2_mma_slide_4x4x8");
constexpr llvm::StringLiteral kVmadot3SlideHelperName(
    "weft_ime_vmadot3_mma_slide_4x4x8");

std::string routeSourceComment(llvm::StringRef opName, llvm::StringRef role) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// weft_emitc.route_source_op=" << opName << " role=" << role
     << " op_interface=" << kOpInterface;
  os.flush();
  return text;
}

std::string stepComment(llvm::StringRef opName, llvm::StringRef role,
                        llvm::StringRef callee) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// weft_emitc.source_op=" << opName << " role=" << role
     << " op_interface=" << kOpInterface << " callee=" << callee;
  os.flush();
  return text;
}

/// The FOUNDATION-validated (real-K1 bit-exact, FOUNDATION.md task 3) int8->int32
/// IME MAC kernel, emitted verbatim as ONE self-contained `static inline` helper.
/// This is the single justified asm leaf: there is no IME intrinsic header, so
/// the instruction leaf is unavoidably one asm block. The helper signature is
/// fixed (its own parameter names), so the asm never references any
/// translator-generated SSA name — the dataflow into it is structured.
///
///   A: (M,K)=(4,8) int8 row-major   -> vs1 = v0
///   B: stored (N,K)=(4,8) int8       -> vs2 = v1  (== B^T of the math matrix)
///   C: (M,N)=(4,4) int32 (even VD pair v2/v3), C += A . B_stored^T
///
/// `helperName` / `mnemonic` parameterize the ONLY two things that differ
/// between the signed (vmadot) and unsigned (vmadotu) MAC: the helper name and
/// the single instruction leaf. Everything structural (load/store/clear) is
/// identical, so the signed vs unsigned divergence is exactly the instruction.
std::string macHelperBody(llvm::StringRef helperName, llvm::StringRef mnemonic) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// weft_ime.asm_leaf=" << helperName
     << " mac=4x4x8 elem_in=int8 accum=int32 ime_op=" << mnemonic << "\n";
  os << "static inline void " << helperName
     << "(const int8_t *A, const int8_t *B, int32_t *C) {\n";
  os << "  __asm__ volatile(\n";
  os << "      \"vsetvli   t0, zero, e8, m1, ta, ma   \\n\\t\"\n";
  os << "      \"vle8.v    v0, (%[pa])                \\n\\t\"\n";
  os << "      \"vle8.v    v1, (%[pb])                \\n\\t\"\n";
  os << "      \"vmv.v.i   v2, 0                      \\n\\t\"\n";
  os << "      \"vmv.v.i   v3, 0                      \\n\\t\"\n";
  os << "      \"" << mnemonic << "    v2, v0, v1                 \\n\\t\"\n";
  os << "      \"vsetvli   t0, zero, e32, m1, ta, ma  \\n\\t\"\n";
  os << "      \"vse32.v   v2, (%[pc])                \\n\\t\"\n";
  os << "      \"addi      t1, %[pc], 32              \\n\\t\"\n";
  os << "      \"vse32.v   v3, (t1)                   \\n\\t\"\n";
  os << "      :\n";
  os << "      : [pa] \"r\"(A), [pb] \"r\"(B), [pc] \"r\"(C)\n";
  os << "      : \"t0\", \"t1\", \"v0\", \"v1\", \"v2\", \"v3\", \"memory\");\n";
  os << "}";
  os.flush();
  return text;
}

std::string vmadotHelperBody() {
  return macHelperBody(kVmadotHelperName, "vmadot");
}

std::string vmadotuHelperBody() {
  return macHelperBody(kVmadotuHelperName, "vmadotu");
}

std::string vmadotsuHelperBody() {
  return macHelperBody(kVmadotsuHelperName, "vmadotsu");
}

std::string vmadotusHelperBody() {
  return macHelperBody(kVmadotusHelperName, "vmadotus");
}

// The BATCHED register-resident MAC leaf name. Same single justified `vmadot`
// instruction leaf as macHelperBody, but driven over a K/8 FRAGMENT LOOP with the
// 4x4 int32 accumulator (v2/v3) kept RESIDENT across the whole loop: ONE
// `vsetvli e8` at entry, ZERO the accumulator ONCE, `vmadot` MACs into v2/v3 per
// fragment (the instruction accumulates C += A.B^T in-register), then ONE
// `vsetvli e32` + a SINGLE store at the end. This drops the per-fragment
// vsetvli e8<->e32 toggle + zeroing + store + scalar acc[] that the un-batched
// leaf incurred once per fragment ([GAP-IME-LEAF-PIPELINE]). The int32 result is
// bit-identical to the per-fragment form (same vmadot reductions, summed in the
// same kf order); only the accumulate/store SCHEDULE changes.
constexpr llvm::StringLiteral kVmadotMacKloopHelperName(
    "weft_ime_vmadot_mac_kloop");

/// The batched register-resident int8->int32 MAC leaf, emitted as ONE
/// self-contained `static inline` helper. It reduces `kt` contiguous 4x8 A/B
/// fragments (32B each) into a single 4x4 int32 result, accumulating in the
/// v2/v3 VD pair across the WHOLE fragment loop:
///   A: kt fragments of (4,8) int8 row-major, contiguous -> vs1 = v0 per iter
///   B: kt fragments of stored (4,8) int8, contiguous     -> vs2 = v1 per iter
///   frag: (M,N)=(4,4) int32 (v2/v3), frag = Sum_kf A_kf . B_kf^T
/// This is the SAME 0xe210312b `vmadot` leaf as macHelperBody; the only structural
/// change vs the un-batched leaf is that the accumulator stays register-resident
/// (single vsetvli, one store) instead of being cleared/stored per fragment. The
/// per-fragment form's `acc[r] += vmadot(A_kf,B_kf)[r]` becomes an in-register
/// `v2 += vmadot(A_kf,B_kf)` over the same kf order => int32 bit-identical. The
/// helper has fixed parameter names, so no translator-generated SSA name is
/// interpolated into the asm.
std::string macKloopHelperBody(llvm::StringRef helperName,
                               llvm::StringRef mnemonic) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// weft_ime.asm_leaf=" << helperName
     << " batched_kloop mac=4x4x8 elem_in=int8 accum=int32 ime_op=" << mnemonic
     << " register_resident_accumulate=1 single_vsetvli=1 store_once=1\n";
  os << "static inline void " << helperName
     << "(const int8_t *A, const int8_t *B, long kt, int32_t *frag) {\n";
  os << "  __asm__ volatile(\n";
  // ONE vsetvli e8 + clear the 4x4 int32 accumulator (v2/v3) ONCE for the tile.
  os << "      \"vsetvli   t0, zero, e8, m1, ta, ma   \\n\\t\"\n";
  os << "      \"vmv.v.i   v2, 0                       \\n\\t\"\n";
  os << "      \"vmv.v.i   v3, 0                       \\n\\t\"\n";
  os << "      \"mv        t2, %[kt]                   \\n\\t\"\n";
  os << "      \"mv        t3, %[pa]                   \\n\\t\"\n";
  os << "      \"mv        t4, %[pb]                   \\n\\t\"\n";
  // K/8 fragment loop: each iter MACs one 4x8 A + 4x8 B fragment into v2/v3
  // (the instruction accumulates), advancing both pointers by 32 bytes.
  os << "      \"1:                                    \\n\\t\"\n";
  os << "      \"vle8.v    v0, (t3)                    \\n\\t\"\n";
  os << "      \"vle8.v    v1, (t4)                    \\n\\t\"\n";
  os << "      \"" << mnemonic << "    v2, v0, v1                 \\n\\t\"\n";
  os << "      \"addi      t3, t3, 32                  \\n\\t\"\n";
  os << "      \"addi      t4, t4, 32                  \\n\\t\"\n";
  os << "      \"addi      t2, t2, -1                  \\n\\t\"\n";
  os << "      \"bnez      t2, 1b                      \\n\\t\"\n";
  // ONE vsetvli e32 + a SINGLE store of the 4x4 int32 result (v2 rows 0,1;
  // v3 rows 2,3 -- the FOUNDATION store shape).
  os << "      \"vsetvli   t0, zero, e32, m1, ta, ma   \\n\\t\"\n";
  os << "      \"vse32.v   v2, (%[pf])                 \\n\\t\"\n";
  os << "      \"addi      t5, %[pf], 32               \\n\\t\"\n";
  os << "      \"vse32.v   v3, (t5)                    \\n\\t\"\n";
  os << "      :\n";
  os << "      : [pa] \"r\"(A), [pb] \"r\"(B), [kt] \"r\"(kt), [pf] \"r\"(frag)\n";
  os << "      : \"t0\", \"t2\", \"t3\", \"t4\", \"t5\", \"v0\", \"v1\", \"v2\", "
        "\"v3\", \"memory\");\n";
  os << "}";
  os.flush();
  return text;
}

std::string vmadotMacKloopHelperBody() {
  return macKloopHelperBody(kVmadotMacKloopHelperName, "vmadot");
}

//===----------------------------------------------------------------------===//
// G6-A M7: the WIDE (output-tiled) vmadot MAC leaf + its [PAT-1] registration.
//
// A LAYER-4 array-UTILIZATION optimization on the already-mechanized batched vmadot MAC leaf: the
// WIDE leaf reuses the 4x8 A fragment IN-REGISTER across NJW adjacent column-tiles, so ONE `vle8`
// of A feeds NJW independent `vmadot` chains (into NJW distinct 4x4 int32 accumulator pairs). This
// (a) hides the vmadot array latency (NJW MACs in flight instead of a single serialized accumulator
// chain) and (b) amortizes the A-load + per-tile vsetvli/clear/store entry-exit over NJW tiles. Each
// 4x4 sub-tile still accumulates its kt fragments in the SAME kf order into its OWN accumulator ->
// the int32 result is BIT-IDENTICAL to NJW separate weft_ime_vmadot_mac_kloop calls (byte-exact by
// construction: the 0xe210312b vmadot and its integer accumulation order are UNTOUCHED; only the
// loop/reuse SCHEDULE changes). This is an optimization on the front-door construction, NOT a new
// brick -- it NEVER moves C_construct. Board-sealed byte-exact on real K1 (md5 f5e77482, the M1..M6
// q4_0 bridge lineage) + array-util measured (vmadot compute 1.740s@w1 -> 1.095s@w2 (1.59x) ->
// 0.890s@w4 (1.955x, ~= the vendor's same-silicon headroom)); the full-matmul perf verdict is w2
// (matmul 1.159x), w4 is measured-negative (compute win real, but its 16-accumulator f32 epilogue
// spills -> full-matmul NULL).
//===----------------------------------------------------------------------===//

/// [PAT-1] IME vmadot output-tiling registry ROW (registration-as-data, the C3' capability-keyed
/// pattern library object). NOT open-coded selection: a data table the emitter consults. Mirrors the
/// schema/pattern-registry.v1.json row shape {pattern_id, requires(capability), status, metrics_hook}.
struct IMEVmadotTilingPattern {
  llvm::StringRef pattern_id;
  int njw;                    ///< output-tile width in column-tiles (= A-fragment reuse factor)
  int minVregBudget;          ///< register floor: 1 A + njw B + 2*njw int32 accumulators
  llvm::StringRef status;     ///< [PAT-1] status: mechanized | measured-negative | planned
  llvm::StringRef metricsHook;
};

/// The DISCRIMINANT capability fact is the vector-register budget (VLEN256 => 32 vregs; each 4x8
/// int8 fragment fills one m1 register, so NJW=w consumes 1 A + w B + 2w accumulator regs). The
/// widths whose register floor fits the budget AND that are `mechanized` are selectable; a
/// `measured-negative` row is a design-space asset (records the falsified w4-epilogue-spill boundary)
/// and is NOT selectable. Per-format board-measured (perf discipline: no projection).
static constexpr IMEVmadotTilingPattern kIMEVmadotTilingPatterns[] = {
    {"IME-VMADOT-TILE-W1-baseline", 1, 3, "mechanized",
     "test/Conversion/EmitC/ime-q4-0-matmul-tile-materialization.mlir"},
    {"IME-VMADOT-TILE-W2-Areuse", 2, 7, "mechanized",
     "test/Target/IME/q4-0-vmadot-tile-wide-int32-oracle.c + "
     "experiments/active/g6-a-ime-perf-bridge/M7-vmadot-tiling/evidence.md"},
    {"IME-VMADOT-TILE-W4-Areuse-epilogue-spill", 4, 13, "measured-negative",
     "experiments/active/g6-a-ime-perf-bridge/M7-vmadot-tiling/evidence.md "
     "(w4 compute 1.955x but full-matmul NULL: 16-accumulator f32 epilogue spills)"},
};

/// The RISC-V architectural vector register-file size. This is the ONE genuinely
/// VLEN-INVARIANT fact in the wide-vmadot accounting: 32 architectural vregs exist
/// at every VLEN. (The per-fragment vreg COST is NOT VLEN-invariant -- that is the
/// bug the hardcoded `32` budget hid; see decideWideVmadotDeployment below.)
constexpr int kRVVVectorRegisterFileSize = 32;

/// G8 wide-vmadot deployment decision (the parametric capability-key result). njw>1
/// => deploy the WIDE (output-tiled) leaf; njw==1 => narrow-only (auto fallback /
/// legal decline). The reason string records WHICH predicate leg fired (for the
/// emitted provenance comment: honest decline is materialized, never silent).
struct IMEWideDeployDecision {
  int njw = 1;
  llvm::StringRef patternId;
  llvm::StringRef statusStr;
  llvm::StringRef metricsHook;
  std::string reason;
};

/// The VLEN-PARAMETRIC capability-keyed wide-vmadot predicate (replaces the
/// hardcoded `32` + q4_0-only hand-wiring). Three legs, all derived from FACTS on
/// the tile op + the capability provider -- none hardcoded per-format:
///   (1) CAPABILITY FACT: vlen_bits -> the per-fragment vreg cost. The int8 4x8 A/B
///       fragment costs rpfIn = ceil(macM*macK*elem_in_bits / VLEN) vregs; the 4x4
///       int32 accumulator tile costs rpfAcc = ceil(macM*macN*accum_bits / VLEN).
///       (=1 / =2 @ VLEN256; =2 / =4 @ VLEN128 -- so the budget floor is NOT
///       VLEN-invariant.) The emitted leaf loads the fragment with ONE `vle8 e8,m1`,
///       which is only valid when the fragment fits ONE vreg (rpfIn==1); otherwise
///       fail-closed to narrow.
///   (2) FORMAT FEATURE: the fragment kt/bits (macM/macN/macK, elem/accum bits) come
///       from the tile op -- the same predicate serves q4_0/q8_0/q4_K, no format name.
///   (3) BOTTLENECK SHAPE: MAC-tiling payoff vs. epilogue weight. `epilogueBound`
///       formats (q4_K: the two-level 6-bit sc/m + S_scale/S_min + fp16 fold that
///       dominates the tile) get a MAC-tiling NULL -> LEGAL DECLINE (the M7
///       q4_K@ime 0.909x measured-null boundary is made an explicit predicate leg,
///       not a silent hand-omission).
/// The vreg floor for a width-njw tiling is rpfIn (the reused A) + njw*rpfIn (the
/// njw B col-tiles) + njw*rpfAcc (the njw int32 accumulator tiles); the widest
/// MECHANIZED tiling whose floor fits the 32-register file is selected.
IMEWideDeployDecision
decideWideVmadotDeployment(int64_t vlenBits, int64_t macM, int64_t macN,
                          int64_t macK, int64_t elemInBits, int64_t accumBits,
                          bool epilogueBound) {
  IMEWideDeployDecision d;
  const int64_t fragBits = macM * macK * elemInBits;
  const int64_t accBits = macM * macN * accumBits;
  // Back-compat default when no provider property is reachable: the validated X60
  // shape has macM*macK*elem_in_bits == VLEN == one m1 vreg, so rpfIn==1.
  const int64_t vb = vlenBits > 0 ? vlenBits : fragBits;
  auto ceilDiv = [](int64_t a, int64_t b) -> int64_t {
    return b > 0 ? (a + b - 1) / b : 1;
  };
  const int64_t rpfIn = std::max<int64_t>(1, ceilDiv(fragBits, vb));
  const int64_t rpfAcc = std::max<int64_t>(1, ceilDiv(accBits, vb));

  // (3) bottleneck-shape leg: epilogue-bound formats decline MAC-tiling.
  if (epilogueBound) {
    d.reason =
        "bottleneck-shape=epilogue-bound decline (MAC-tiling NULL): the two-level "
        "6-bit sc/m + S_scale/S_min + fp16 fold dominates the tile; array-util wins "
        "wash out [PAT-1 format-keyed boundary; M7 q4_K@ime 0.909x measured-null]";
    return d;
  }
  // (1) capability leg: the single-`vle8 e8,m1` leaf requires the fragment to fit
  // ONE vreg. At a VLEN where it does not (rpfIn>1) the emitted leaf is invalid ->
  // fail-closed to narrow.
  if (rpfIn != 1) {
    d.reason = "capability=vlen-fragment-mismatch decline: macM*macK*elem_in_bits=" +
               std::to_string(fragBits) + " does not fit one " + std::to_string(vb) +
               "-bit vreg (rpfIn=" + std::to_string(rpfIn) +
               "); single-vle8 e8,m1 leaf invalid at this VLEN -> narrow fallback";
    return d;
  }
  // (1)+(2): widest MECHANIZED tiling whose VLEN-parametric vreg floor fits 32.
  const IMEVmadotTilingPattern *best = &kIMEVmadotTilingPatterns[0];
  int64_t bestFloor = rpfIn * (1 + best->njw) + rpfAcc * best->njw;
  for (const IMEVmadotTilingPattern &p : kIMEVmadotTilingPatterns) {
    const int64_t floorNjw = rpfIn * (1 + p.njw) + rpfAcc * (int64_t)p.njw;
    if (p.status == "mechanized" && floorNjw <= kRVVVectorRegisterFileSize &&
        p.njw > best->njw) {
      best = &p;
      bestFloor = floorNjw;
    }
  }
  d.njw = best->njw;
  d.patternId = best->pattern_id;
  d.statusStr = best->status;
  d.metricsHook = best->metricsHook;
  d.reason = "capability=vlen_bits=" + std::to_string(vb) + " rpfIn=1 rpfAcc=" +
             std::to_string(rpfAcc) + " vreg_floor(njw=" + std::to_string(best->njw) +
             ")=" + std::to_string(bestFloor) + "<=32 => deploy " +
             best->pattern_id.str();
  return d;
}

/// Reads the DEPLOYED VLEN capability FACT (bits) reachable from the tile op's
/// module. The IME capability provider (weft.exec.capability, id "spacemit.ime")
/// carries vlen_bits as a string property; emission runs on a whole-module clone so
/// the provider is still a sibling of the tile op's kernel. Returns 0 when no
/// provider/property is found (caller falls back to the fragment-fits-one-vreg
/// default -- back-compat with standalone tile-op fixtures that carry no provider).
int64_t readDeployedVlenBits(mlir::Operation *tile) {
  auto module = tile->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return 0;
  int64_t vlenBits = 0;
  module.walk([&](mlir::Operation *op) {
    if (!op->getName().getStringRef().ends_with("exec.capability"))
      return mlir::WalkResult::advance();
    auto idAttr = op->getAttrOfType<mlir::StringAttr>("id");
    if (!idAttr || !idAttr.getValue().contains("spacemit.ime"))
      return mlir::WalkResult::advance();
    if (auto vb = op->getAttrOfType<mlir::StringAttr>("vlen_bits")) {
      long long parsed = 0;
      if (!vb.getValue().getAsInteger(10, parsed) && parsed > 0)
        vlenBits = parsed;
    }
    return mlir::WalkResult::interrupt();
  });
  return vlenBits;
}

/// The WIDE (NJW-tiled) register-resident int8->int32 MAC leaf, emitted as ONE self-contained
/// `static inline` helper. ONE `vle8` of the 4x8 A fragment feeds NJW independent `vmadot`s into NJW
/// distinct 4x4 int32 accumulator pairs, over the whole kt fragment loop (single entry vsetvli, one
/// clear per accumulator, NJW stores at the end). `B0` is the first col-tile's fragment stream;
/// col-tile w is `B0 + w*bstride`. `frag` holds NJW*16 int32 (tile w = frag[w*16 .. w*16+15]). The
/// per-sub-tile int32 is BIT-IDENTICAL to `macKloopHelperName` run NJW times (same vmadot, same kf
/// order). Register map matches the K1-sealed board mirror: A=v0; B=v1,v6,v7,v8; acc pairs
/// v2:v3,v4:v5,v10:v11,v12:v13. Fixed parameter names -> no translator SSA name in the asm.
std::string macKloopHelperBodyWide(llvm::StringRef helperName,
                                   llvm::StringRef mnemonic, int njw) {
  assert((njw == 2 || njw == 4) && "IME wide vmadot leaf supports NJW in {2,4}");
  static const char *const bReg[4] = {"v1", "v6", "v7", "v8"};
  static const char *const accLo[4] = {"v2", "v4", "v10", "v12"};
  static const char *const accHi[4] = {"v3", "v5", "v11", "v13"};
  static const char *const bPtr[4] = {"t3", "t4", "t5", "t6"};
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// weft_ime.asm_leaf=" << helperName
     << " wide_batched_kloop mac=4x4x8 elem_in=int8 accum=int32 ime_op=" << mnemonic
     << " tile_width_njw=" << njw
     << " a_fragment_reuse=1 register_resident_accumulate=1 single_vsetvli=1\n";
  os << "static inline void " << helperName
     << "(const int8_t *A, const int8_t *B0, long bstride, long kt, int32_t *frag) {\n";
  for (int w = 1; w < njw; ++w)
    os << "  const int8_t *B" << w << " = B0 + (long)" << w << " * bstride;\n";
  os << "  __asm__ volatile(\n";
  os << "      \"vsetvli   t0, zero, e8, m1, ta, ma   \\n\\t\"\n";
  for (int w = 0; w < njw; ++w) {
    os << "      \"vmv.v.i   " << accLo[w] << ", 0                       \\n\\t\"\n";
    os << "      \"vmv.v.i   " << accHi[w] << ", 0                       \\n\\t\"\n";
  }
  os << "      \"mv        t2, %[kt]                   \\n\\t\"\n";
  os << "      \"mv        t1, %[pa]                   \\n\\t\"\n";
  for (int w = 0; w < njw; ++w)
    os << "      \"mv        " << bPtr[w] << ", %[pb" << w << "]                  \\n\\t\"\n";
  os << "      \"1:                                    \\n\\t\"\n";
  os << "      \"vle8.v    v0, (t1)                    \\n\\t\"\n";
  for (int w = 0; w < njw; ++w)
    os << "      \"vle8.v    " << bReg[w] << ", (" << bPtr[w] << ")                    \\n\\t\"\n";
  for (int w = 0; w < njw; ++w)
    os << "      \"" << mnemonic << "    " << accLo[w] << ", v0, " << bReg[w]
       << "                 \\n\\t\"\n";
  os << "      \"addi      t1, t1, 32                  \\n\\t\"\n";
  for (int w = 0; w < njw; ++w)
    os << "      \"addi      " << bPtr[w] << ", " << bPtr[w] << ", 32                  \\n\\t\"\n";
  os << "      \"addi      t2, t2, -1                  \\n\\t\"\n";
  os << "      \"bnez      t2, 1b                      \\n\\t\"\n";
  os << "      \"vsetvli   t0, zero, e32, m1, ta, ma   \\n\\t\"\n";
  os << "      \"mv        t1, %[pf]                   \\n\\t\"\n";
  for (int w = 0; w < njw; ++w) {
    os << "      \"vse32.v   " << accLo[w] << ", (t1)                   \\n\\t\"\n";
    os << "      \"addi      t1, t1, 32                  \\n\\t\"\n";
    os << "      \"vse32.v   " << accHi[w] << ", (t1)                   \\n\\t\"\n";
    if (w + 1 < njw)
      os << "      \"addi      t1, t1, 32                  \\n\\t\"\n";
  }
  os << "      :\n";
  os << "      : [pa] \"r\"(A), [kt] \"r\"(kt), [pf] \"r\"(frag)";
  for (int w = 0; w < njw; ++w)
    os << ", [pb" << w << "] \"r\"(B" << (w == 0 ? std::string("0") : std::to_string(w)) << ")";
  os << "\n";
  os << "      : \"t0\", \"t1\", \"t2\", \"t3\", \"t4\", \"t5\", \"t6\", \"v0\", \"v1\"";
  for (int w = 0; w < njw; ++w)
    os << ", \"" << accLo[w] << "\", \"" << accHi[w] << "\"";
  for (int w = 1; w < njw; ++w)
    os << ", \"" << bReg[w] << "\"";
  os << ", \"memory\");\n";
  os << "}";
  os.flush();
  return text;
}

/// G8 applied=>deployed: emits (at the CURRENT insertion point = module prologue,
/// BEFORE the format matmul helper that CALLS it) the [PAT-1] registry provenance
/// comment + the selected WIDE vmadot MAC leaf body, and returns its helper name so
/// the matmul body can be wired to it. Returns "" when the decision declines wide
/// (njw<=1): the narrow leaf stays the deployed leaf (auto fallback). This closes
/// the "_w2 emitted but never wired -> dead" gap: the leaf is now emitted ONLY when
/// it is deployed, and always at prologue scope (declared-before-use).
std::string
emitDeployedWideVmadotLeaf(mlir::ConversionPatternRewriter &rewriter,
                           mlir::Location loc,
                           const IMEWideDeployDecision &decision) {
  if (decision.njw <= 1)
    return {};
  std::string wideName =
      (kVmadotMacKloopHelperName + "_w" + std::to_string(decision.njw)).str();
  rewriter.create<emitc::VerbatimOp>(
      loc, std::string("// weft_ime.pat1_tiling=") + decision.patternId.str() +
               " status=" + decision.statusStr.str() +
               " njw=" + std::to_string(decision.njw) +
               " discriminant=vreg_budget deployed=1 metrics_hook=" +
               decision.metricsHook.str() + " reason=" + decision.reason);
  rewriter.create<emitc::VerbatimOp>(
      loc, macKloopHelperBodyWide(wideName, "vmadot", decision.njw));
  return wideName;
}

/// The sliding-window MAC helper. UNLIKE macHelperBody (single 32B A fragment),
/// the slide kernel loads A as an EVEN VS1:VS1+1 PAIR holding an 8x8 int8 block
/// (v0 = A rows 0..3 at A+0, v1 = A rows 4..7 at A+32; 64B total), B as one 4x8
/// fragment into v2, and C as the EVEN VD PAIR v4:v5 (4x4 int32). The single
/// justified asm leaf is `vmadot{1,2,3} v4, v0, v2` — the A read-window shifts
/// DOWN by `slide` rows (vmadot1=rows1..4). funct7 111001 (e6...) is DISTINCT
/// from the non-slide MAC. This layout was verified bit-exact on real K1 (the
/// STEP-0 4-way window discriminator). `helperName`/`mnemonic` parameterize the
/// only divergence across slide 1/2/3; everything structural is shared.
std::string macSlideHelperBody(llvm::StringRef helperName,
                               llvm::StringRef mnemonic) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// weft_ime.asm_leaf=" << helperName
     << " slide_mac=4x4x8 a_pair=8x8 elem_in=int8 accum=int32 ime_op="
     << mnemonic << " funct7=111001\n";
  os << "static inline void " << helperName
     << "(const int8_t *A, const int8_t *B, int32_t *C) {\n";
  os << "  __asm__ volatile(\n";
  os << "      \"vsetvli   t0, zero, e8, m1, ta, ma   \\n\\t\"\n";
  // A even pair v0:v1 = 8x8 int8 block (v0 rows 0..3, v1 rows 4..7).
  os << "      \"vle8.v    v0, (%[pa])                \\n\\t\"\n";
  os << "      \"addi      t1, %[pa], 32              \\n\\t\"\n";
  os << "      \"vle8.v    v1, (t1)                   \\n\\t\"\n";
  // B one 4x8 fragment into v2.
  os << "      \"vle8.v    v2, (%[pb])                \\n\\t\"\n";
  // Clear the 4x4 int32 accumulator (even VD pair v4:v5).
  os << "      \"vsetvli   t0, zero, e32, m1, ta, ma  \\n\\t\"\n";
  os << "      \"vmv.v.i   v4, 0                      \\n\\t\"\n";
  os << "      \"vmv.v.i   v5, 0                      \\n\\t\"\n";
  os << "      \"vsetvli   t0, zero, e8, m1, ta, ma   \\n\\t\"\n";
  // The slide MAC leaf: A window slides DOWN by `slide` rows.
  os << "      \"" << mnemonic << "   v4, v0, v2                 \\n\\t\"\n";
  os << "      \"vsetvli   t0, zero, e32, m1, ta, ma  \\n\\t\"\n";
  os << "      \"vse32.v   v4, (%[pc])                \\n\\t\"\n";
  os << "      \"addi      t1, %[pc], 32              \\n\\t\"\n";
  os << "      \"vse32.v   v5, (t1)                   \\n\\t\"\n";
  os << "      :\n";
  os << "      : [pa] \"r\"(A), [pb] \"r\"(B), [pc] \"r\"(C)\n";
  os << "      : \"t0\", \"t1\", \"v0\", \"v1\", \"v2\", \"v4\", \"v5\", "
        "\"memory\");\n";
  os << "}";
  os.flush();
  return text;
}

std::string vmadot1SlideHelperBody() {
  return macSlideHelperBody(kVmadot1SlideHelperName, "vmadot1");
}

std::string vmadot2SlideHelperBody() {
  return macSlideHelperBody(kVmadot2SlideHelperName, "vmadot2");
}

std::string vmadot3SlideHelperBody() {
  return macSlideHelperBody(kVmadot3SlideHelperName, "vmadot3");
}

// The tiled whole-matrix micro-kernel helper names. The single justified asm
// leaf is the same `vmadot`/`vmadotu` instruction, here driven IN-REGISTER over
// the whole K reduction so the per-output reduce (RVV's vredsum) is
// structurally absent — that is exactly the IME advantage the bench measures.
constexpr llvm::StringLiteral kMatmulHelperName("weft_ime_vmadot_matmul");
constexpr llvm::StringLiteral kMatmulUHelperName("weft_ime_vmadotu_matmul");

/// The tiled int8->int32 whole-matrix kernel, emitted as ONE self-contained
/// `static inline` helper. It computes C[M,N] += A[M,K] . B[K,N] by looping the
/// FOUNDATION-validated 4x4x8 `vmadot` fragment over the (M/4)x(N/4)x(K/8) tile
/// grid. The inputs are PRE-PACKED into FRAGMENT-MAJOR tile-contiguous layout
/// (the same repack policy applied to BOTH the IME and the RVV baseline; weight
/// repack is amortized offline in e2e), so each 4x8 fragment of A and B is a
/// contiguous 32-byte `vle8` — exactly the FOUNDATION load (`vle8 v0,(A)` over a
/// 4x8 row-major tile loads [r0(8),r1(8),r2(8),r3(8)] = one fragment):
///   Apack: (M/4) row-tiles. Tile mi = K/8 fragments concatenated; fragment f =
///          [Arow0[8f..8f+7], Arow1[..], Arow2[..], Arow3[..]] (32B).
///   Bpack: (N/4) col-tiles, same fragment-major layout over B^T's 4 rows.
///   C:     (M,N) int32 row-major.
/// Per output tile (mi,nj), the 4x4 int32 accumulator lives in the v2/v3 VD
/// pair across the WHOLE K loop (vmadot accumulates C += A.B^T in-register), so
/// the per-output reduce RVV needs (vredsum) is structurally absent. The result
/// is stored ONCE to a contiguous 16-int32 scratch (v2 = rows 0,1; v3 = rows
/// 2,3 — exactly FOUNDATION's validated store), then the structured C wrapper
/// scatters it into the N-strided C tile. The signed/unsigned divergence is
/// exactly the instruction mnemonic; everything structural is identical. The
/// helper has fixed parameter names, so no translator-generated SSA name is
/// interpolated into the asm.
std::string matmulHelperBody(llvm::StringRef helperName,
                             llvm::StringRef mnemonic) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// weft_ime.asm_leaf=" << helperName
     << " tiled_matmul mac=4x4x8 elem_in=int8 accum=int32 ime_op=" << mnemonic
     << " in_register_K_accumulate=1\n";
  os << "static inline void " << helperName
     << "(const int8_t *Apack, const int8_t *Bpack, int32_t *C,\n";
  os << "    long M, long N, long K) {\n";
  os << "  const long mt = M / 4, nt = N / 4, kt = K / 8;\n";
  os << "  for (long mi = 0; mi < mt; ++mi) {\n";
  // Apack row-tile (mi): K/8 fragments of 32B, contiguous = mi*4*K bytes in.
  os << "    const int8_t *Atile = Apack + (long)mi * 4 * K;\n";
  os << "    for (long nj = 0; nj < nt; ++nj) {\n";
  // Bpack col-tile (nj): K/8 fragments of 32B (B^T col-tile), contiguous.
  os << "      const int8_t *Btile = Bpack + (long)nj * 4 * K;\n";
  // 16-int32 scratch for the single per-tile store (rows 0,1 in v2; 2,3 in v3),
  // exactly the FOUNDATION store shape. The strided scatter into C is below, in
  // STRUCTURED C (not asm), keeping the asm leaf the validated store.
  os << "      int32_t Cs[16];\n";
  os << "      __asm__ volatile(\n";
  // Clear the 4x4 int32 accumulator (v2/v3) once for this output tile.
  os << "          \"vsetvli   t0, zero, e32, m1, ta, ma  \\n\\t\"\n";
  os << "          \"vmv.v.i   v2, 0                      \\n\\t\"\n";
  os << "          \"vmv.v.i   v3, 0                      \\n\\t\"\n";
  os << "          \"vsetvli   t0, zero, e8, m1, ta, ma   \\n\\t\"\n";
  os << "          \"mv        t1, %[ka]                  \\n\\t\"\n";
  os << "          \"mv        t2, %[kb]                  \\n\\t\"\n";
  os << "          \"mv        t3, %[kt]                  \\n\\t\"\n";
  // K loop: each iter consumes one 4x8 A fragment + one 4x8 B fragment (32B
  // each), accumulating into v2/v3. Pointers advance by 32 bytes per fragment.
  os << "          \"1:                                  \\n\\t\"\n";
  os << "          \"vle8.v    v0, (t1)                   \\n\\t\"\n";
  os << "          \"vle8.v    v1, (t2)                   \\n\\t\"\n";
  os << "          \"" << mnemonic << "    v2, v0, v1                 \\n\\t\"\n";
  os << "          \"addi      t1, t1, 32                 \\n\\t\"\n";
  os << "          \"addi      t2, t2, 32                 \\n\\t\"\n";
  os << "          \"addi      t3, t3, -1                 \\n\\t\"\n";
  os << "          \"bnez      t3, 1b                     \\n\\t\"\n";
  // Store the 4x4 int32 result ONCE to the contiguous scratch (FOUNDATION
  // store: v2 -> Cs[0..7] rows 0,1; v3 -> Cs[8..15] rows 2,3).
  os << "          \"vsetvli   t0, zero, e32, m1, ta, ma  \\n\\t\"\n";
  os << "          \"vse32.v   v2, (%[pcs])               \\n\\t\"\n";
  os << "          \"addi      t4, %[pcs], 32             \\n\\t\"\n";
  os << "          \"vse32.v   v3, (t4)                   \\n\\t\"\n";
  os << "          :\n";
  os << "          : [ka] \"r\"(Atile), [kb] \"r\"(Btile), [kt] \"r\"(kt),\n";
  os << "            [pcs] \"r\"(Cs)\n";
  os << "          : \"t0\", \"t1\", \"t2\", \"t3\", \"t4\",\n";
  os << "            \"v0\", \"v1\", \"v2\", \"v3\", \"memory\");\n";
  // Structured strided scatter: Cs[r*4+c] -> C[(mi*4+r)*N + nj*4+c].
  os << "      for (long r = 0; r < 4; ++r)\n";
  os << "        for (long c = 0; c < 4; ++c)\n";
  os << "          C[(long)(mi * 4 + r) * N + (nj * 4 + c)] += Cs[r * 4 + c];\n";
  os << "    }\n";
  os << "  }\n";
  os << "}";
  os.flush();
  return text;
}

std::string matmulHelperBodySigned() {
  return matmulHelperBody(kMatmulHelperName, "vmadot");
}

std::string matmulHelperBodyUnsigned() {
  return matmulHelperBody(kMatmulUHelperName, "vmadotu");
}

//===----------------------------------------------------------------------===//
// G4 M1a: the FORMAT-KEYED q4_0 IME GEMM tile emitter.
//
// The typed region (weft.ime.q4_0_matmul_tile) carries two decomposed bricks:
//   * weft.ime.q4_0_dequant_core -> the q4_0 offset-binary nibble DECODE, emitted
//     as ONE structured C helper (no asm): out[j]=(qs[j]&0xF)-8, out[j+16]=
//     (qs[j]>>4)-8, the ggml q4_0 quant in [-8,7] that fits int8 exactly;
//   * weft.ime.vmadot_mac_leaf -> the FOUNDATION-validated single-fragment vmadot
//     MAC (the SAME justified asm leaf as weft.ime.mma). The K reduction
//     accumulates in int32 in a C scratch (int32-EXACT; the in-register-accumulate
//     perf form is a board/perf concern, not this correctness construction).
// The output is the int32-EXACT MAC accumulator (what the M1b K1 seal validates
// bit-exact); the per-block fp16 scale fold to float is a SEPARATE downstream
// epilogue (it introduces fp16 rounding, not int32-exact) and is deliberately not
// part of this int32-exact construction.
//===----------------------------------------------------------------------===//
constexpr llvm::StringLiteral kQ40DequantHelperName(
    "weft_ime_q4_0_dequant_fragment");
constexpr llvm::StringLiteral kQ40MatmulHelperName(
    "weft_ime_q4_0_vmadot_matmul");
// G5-M3: the q4_0 forward-bridge SCALE-FOLD epilogue kernel. This is the
// deferred per-block fp16 scale fold that turns the int32-EXACT MAC core into the
// ggml q4_0 x q8_0 f32 mul_mat result the forward path consumes. It REUSES the
// int32-EXACT decode + batched vmadot MAC verbatim (the M1b K1-sealed core); the
// ONLY new arithmetic is the per-32-block d_a*d_w*partial float fold (this
// introduces fp16 rounding, so it is the forward-facing epilogue, NOT the
// int32-exact seal object). Host + K1-silicon validated against a canonical
// q4_0 x q8_0 ZERO-MODEL reference (test/Target/IME/q4-0-matmul-tile-scalefold-*.c).
constexpr llvm::StringLiteral kQ40ScaleFoldMatmulHelperName(
    "weft_ime_q4_0_vmadot_matmul_f32");

/// The q4_0 offset-binary nibble DECODE helper (structured C, no asm). Decodes
/// one 18-byte ggml q4_0 block ([fp16 d][16 nibble bytes]) into a 32-int8 4x8 MAC
/// fragment. The fp16 scale `d` is NOT applied (a separate float fold), so this is
/// a PURE integer transform and the fragment feeds the int8->int32 vmadot exactly.
std::string q40DequantHelperBody() {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// weft_ime.decode_core=" << kQ40DequantHelperName
     << " decode_model=q4_0_offset_binary_nibble qk=32 weight_block_stride=18 "
        "weight_quant_byte_offset=2\n";
  os << "static inline void " << kQ40DequantHelperName
     << "(const uint8_t *blk, int8_t *out) {\n";
  os << "  const uint8_t *qs = blk + 2; // past the 2-byte fp16 d\n";
  os << "  for (int j = 0; j < 16; ++j) {\n";
  os << "    out[j]      = (int8_t)((int)(qs[j] & 0x0F) - 8);\n";
  os << "    out[j + 16] = (int8_t)((int)(qs[j] >> 4)  - 8);\n";
  os << "  }\n";
  os << "}";
  os.flush();
  return text;
}

/// The tiled q4_0 int8->int32 GEMM helper. Per 4x4 output tile (mi,nj) it reduces
/// over the K/8 contraction blocks: DECODE the q4_0 weight block into an int8 4x8
/// fragment (the decode core), run the FOUNDATION-validated `vmadot` MAC on it and
/// the pre-packed int8 activation fragment (the MAC leaf), and accumulate the 4x4
/// int32 fragment result into the tile's int32 accumulator (int32-EXACT). The
/// weight is pre-packed FRAGMENT-MAJOR (one 18-byte q4_0 block per 4x8 MAC
/// fragment): Bq4 col-tile nj is kt contiguous q4_0 blocks. The activation is the
/// SAME fragment-major int8 pack as weft.ime.matmul. `vmadotHelperName` is the
/// FOUNDATION single-fragment MAC helper this reuses.
/// G8 shared FLAT-format (q4_0/q8_0) tiled int32 GEMM loop body. Emits, into `os`,
/// the (M/4)x(N/4) output-tile grid over a flat per-fragment decode. When a wide
/// leaf is DEPLOYED (njw>1, wideName non-empty) the column-tile loop runs in NJW
/// steps through `wideName` -- decoding NJW adjacent col-tiles into ONE contiguous
/// buffer (col-tile w at w*kt*32, bstride kt*32) and reusing ONE `vle8` of A across
/// NJW `vmadot` chains -- then a NARROW remainder loop covers the leftover (<njw)
/// col-tiles. Each wide sub-tile w's int32 is BIT-IDENTICAL to `macKloopName` on
/// that col-tile (K1-sealed f5e77482): same vmadot, same kf order, same scatter;
/// only the A-reuse/register schedule differs. njw<=1 => narrow-only (byte-exact
/// auto fallback), identical to the pre-G8 body.
void emitFlatTiledMatmulLoop(llvm::raw_string_ostream &os,
                             llvm::StringRef weightParam,
                             llvm::StringRef blockBytesConst,
                             llvm::StringRef dequantName,
                             llvm::StringRef macKloopName,
                             llvm::StringRef wideName, int njw) {
  os << "  const long mt = M / 4, nt = N / 4, kt = K / 8;\n";
  os << "  for (long mi = 0; mi < mt; ++mi) {\n";
  os << "    const int8_t *Arow = Apack + (long)mi * 4 * K;\n";
  os << "    long nj = 0;\n";
  if (njw > 1 && !wideName.empty()) {
    os << "    for (; nj + " << njw << " <= nt; nj += " << njw << ") {\n";
    os << "      int8_t Bdec[" << njw << " * kt * 32];\n";
    os << "      for (long w = 0; w < " << njw << "; ++w) {\n";
    os << "        const uint8_t *Bcol = " << weightParam << " + (nj + w) * kt * "
       << blockBytesConst << ";\n";
    os << "        for (long kf = 0; kf < kt; ++kf)\n";
    os << "          " << dequantName << "(Bcol + kf * " << blockBytesConst
       << ", Bdec + (w * kt + kf) * 32);\n";
    os << "      }\n";
    os << "      int32_t frag[" << njw << " * 16];\n";
    os << "      " << wideName << "(Arow, Bdec, kt * 32, kt, frag);\n";
    os << "      for (long w = 0; w < " << njw << "; ++w)\n";
    os << "        for (long r = 0; r < 4; ++r)\n";
    os << "          for (long c = 0; c < 4; ++c)\n";
    os << "            C[(long)(mi * 4 + r) * N + ((nj + w) * 4 + c)] += "
          "frag[w * 16 + r * 4 + c];\n";
    os << "    }\n";
  }
  os << "    for (; nj < nt; ++nj) {\n";
  os << "      const uint8_t *Bcol = " << weightParam << " + (long)nj * kt * "
     << blockBytesConst << ";\n";
  os << "      int8_t Bdec[kt * 32];\n";
  os << "      for (long kf = 0; kf < kt; ++kf)\n";
  os << "        " << dequantName << "(Bcol + kf * " << blockBytesConst
     << ", Bdec + kf * 32);\n";
  os << "      int32_t frag[16];\n";
  os << "      " << macKloopName << "(Arow, Bdec, kt, frag);\n";
  os << "      for (long r = 0; r < 4; ++r)\n";
  os << "        for (long c = 0; c < 4; ++c)\n";
  os << "          C[(long)(mi * 4 + r) * N + (nj * 4 + c)] += frag[r * 4 + c];\n";
  os << "    }\n";
  os << "  }\n";
}

std::string q40MatmulHelperBody(llvm::StringRef macKloopHelperName,
                                llvm::StringRef wideName, int njw) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// weft_ime.asm_leaf=" << macKloopHelperName
     << " tiled_q4_0_matmul mac=4x4x8 elem_in=int8 accum=int32 ime_op=vmadot "
        "weight_format=q4_0 int32_exact=1 register_resident_accumulate=1"
     << (njw > 1 ? " wide_deployed_njw=" + std::to_string(njw) : std::string())
     << "\n";
  os << "static void " << kQ40MatmulHelperName
     << "(const int8_t *Apack, const uint8_t *Bq4, int32_t *C,\n";
  os << "    long M, long N, long K) {\n";
  os << "  const long q40_block_bytes = 18; // fp16 d + 16 nibble bytes\n";
  emitFlatTiledMatmulLoop(os, "Bq4", "q40_block_bytes", kQ40DequantHelperName,
                          macKloopHelperName, wideName, njw);
  os << "}";
  os.flush();
  return text;
}

/// The q4_0 forward-bridge SCALE-FOLD f32 GEMM helper (G5-M3). Per 4x4 output tile
/// it walks the K/32 contraction blocks; per 32-block it decodes the block's 4
/// fragment-major q4_0 weight blocks, runs ONE register-resident batched vmadot MAC
/// over the 4 fragments -> the int32-EXACT partial Sum(qa*qw) for that block (the
/// M1b-sealed core, decode + MAC REUSED verbatim), then folds d_a*d_w*partial into
/// the f32 accumulator Cf. The weight nibble pack (Bnib) is the SAME fragment-major
/// 18-byte-block layout the int32 seal uses; the per-(column,block) fp16 weight
/// scale is carried in the parallel dW array, the per-(row,block) activation scale
/// in dA. M/N/K are RUNTIME parameters (the fixed micro-tile generalized to the
/// tensor's real shape). This is the forward-facing epilogue (introduces fp16
/// rounding), distinct from the int32-exact weft_ime_q4_0_vmadot_matmul seal object.
/// `macKloopHelperName` is the FOUNDATION batched MAC helper this reuses.
std::string q40ScaleFoldMatmulHelperBody(llvm::StringRef macKloopHelperName,
                                         llvm::StringRef wideName, int njw) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// weft_ime.scale_fold_epilogue=" << kQ40ScaleFoldMatmulHelperName
     << " tiled_q4_0_matmul_f32 fold_model=per_block_da_dw weight_format=q4_0 "
        "int32_core_exact=1 fp16_scale_fold=deferred "
        "register_resident_accumulate=1"
     << (njw > 1 ? " wide_deployed_njw=" + std::to_string(njw) : std::string())
     << "\n";
  os << "static void " << kQ40ScaleFoldMatmulHelperName
     << "(const int8_t *Apack, const float *dA, const uint8_t *Bnib,\n";
  os << "    const float *dW, float *Cf, long M, long N, long K) {\n";
  os << "  const long mt = M / 4, nt = N / 4, nb = K / 32;\n";
  os << "  const long q40_block_bytes = 18;\n";
  os << "  const long frags_per_block = 4;\n";
  os << "  const long kt = K / 8;\n";
  os << "  for (long mi = 0; mi < mt; ++mi) {\n";
  os << "    const int8_t *Arow = Apack + (long)mi * 4 * K;\n";
  os << "    long nj = 0;\n";
  if (njw > 1 && !wideName.empty()) {
    // G8 applied=>deployed on the ggml-called FORWARD path: the wide leaf tiles the
    // per-32-block MAC (frags_per_block fragments) across NJW adjacent col-tiles.
    // The f32 fold's per-(m,n) accumulation over b keeps the SAME order as narrow
    // (b-loop inside the nj-group), so Cf is byte-exact (the int32 core is the
    // K1-sealed wide==narrow leaf; the fp16 fold is per-element, order-preserved).
    os << "    for (; nj + " << njw << " <= nt; nj += " << njw << ") {\n";
    os << "      for (long b = 0; b < nb; ++b) {\n";
    os << "        int8_t Bdec[" << njw << " * 128];\n";
    os << "        for (long w = 0; w < " << njw << "; ++w) {\n";
    os << "          const uint8_t *Bcol = Bnib + (nj + w) * kt * q40_block_bytes;\n";
    os << "          for (long f = 0; f < frags_per_block; ++f)\n";
    os << "            " << kQ40DequantHelperName
       << "(Bcol + (b * frags_per_block + f) * q40_block_bytes, "
          "Bdec + (w * frags_per_block + f) * 32);\n";
    os << "        }\n";
    os << "        int32_t frag[" << njw << " * 16];\n";
    os << "        " << wideName
       << "(Arow + b * frags_per_block * 32, Bdec, frags_per_block * 32, "
          "frags_per_block, frag);\n";
    os << "        for (long w = 0; w < " << njw << "; ++w)\n";
    os << "          for (long r = 0; r < 4; ++r)\n";
    os << "            for (long c = 0; c < 4; ++c) {\n";
    os << "              long m = mi * 4 + r, n = (nj + w) * 4 + c;\n";
    os << "              Cf[m * N + n] += dA[m * nb + b] * dW[n * nb + b] * "
          "(float)frag[w * 16 + r * 4 + c];\n";
    os << "            }\n";
    os << "      }\n";
    os << "    }\n";
  }
  os << "    for (; nj < nt; ++nj) {\n";
  os << "      const uint8_t *Bcol = Bnib + (long)nj * kt * q40_block_bytes;\n";
  os << "      for (long b = 0; b < nb; ++b) {\n";
  os << "        int8_t Bdec[128];\n";
  os << "        for (long f = 0; f < frags_per_block; ++f)\n";
  os << "          " << kQ40DequantHelperName
     << "(Bcol + (b * frags_per_block + f) * q40_block_bytes, Bdec + f * 32);\n";
  os << "        int32_t frag[16];\n";
  os << "        " << macKloopHelperName
     << "(Arow + b * frags_per_block * 32, Bdec, frags_per_block, frag);\n";
  os << "        for (long r = 0; r < 4; ++r)\n";
  os << "          for (long c = 0; c < 4; ++c) {\n";
  os << "            long m = mi * 4 + r, n = nj * 4 + c;\n";
  os << "            Cf[m * N + n] += dA[m * nb + b] * dW[n * nb + b] * "
        "(float)frag[r * 4 + c];\n";
  os << "          }\n";
  os << "      }\n";
  os << "    }\n";
  os << "  }\n";
  os << "}";
  os.flush();
  return text;
}

//===----------------------------------------------------------------------===//
// G4 M2: the FORMAT-KEYED q8_0 IME GEMM tile emitter (the FLAT-int8 copy-adapt
// sibling of the q4_0 emitter above).
//
// The typed region (weft.ime.q8_0_matmul_tile) carries two decomposed bricks:
//   * weft.ime.q8_0_dequant_core -> the q8_0 DIRECT int8 read, emitted as ONE
//     structured C helper (no asm, no nibble unpack, no offset): out[j]=qs[j],
//     the ggml q8_0 int8 quant in [-128,127] copied straight into the fragment;
//   * weft.ime.vmadot_mac_leaf -> the SAME FOUNDATION-validated single-fragment
//     vmadot MAC as q4_0 (reused verbatim). The K reduction accumulates in int32.
// The output is the int32-EXACT MAC accumulator (what the K1 seal validates
// bit-exact); the per-block fp16 scale fold is a SEPARATE downstream epilogue.
//===----------------------------------------------------------------------===//
constexpr llvm::StringLiteral kQ80DequantHelperName(
    "weft_ime_q8_0_dequant_fragment");
constexpr llvm::StringLiteral kQ80MatmulHelperName(
    "weft_ime_q8_0_vmadot_matmul");

/// The q8_0 DIRECT int8 DECODE helper (structured C, no asm). Copies one 34-byte
/// ggml q8_0 block ([fp16 d][32 int8 quants]) into a 32-int8 4x8 MAC fragment. The
/// fp16 scale `d` is NOT applied (a separate float fold), so this is a PURE
/// integer transform and the fragment feeds the int8->int32 vmadot exactly.
std::string q80DequantHelperBody() {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// weft_ime.decode_core=" << kQ80DequantHelperName
     << " decode_model=q8_0_direct_int8 qk=32 weight_block_stride=34 "
        "weight_quant_byte_offset=2\n";
  os << "static inline void " << kQ80DequantHelperName
     << "(const uint8_t *blk, int8_t *out) {\n";
  os << "  const int8_t *qs = (const int8_t *)(blk + 2); // past the 2-byte fp16 d\n";
  os << "  for (int j = 0; j < 32; ++j) {\n";
  os << "    out[j] = qs[j];\n";
  os << "  }\n";
  os << "}";
  os.flush();
  return text;
}

/// The tiled q8_0 int8->int32 GEMM helper (the copy-adapt sibling of
/// q40MatmulHelperBody). Per 4x4 output tile (mi,nj) it reduces over the K/8
/// contraction blocks: DECODE the q8_0 weight block (direct int8) into an int8 4x8
/// fragment, run the FOUNDATION-validated `vmadot` MAC on it and the pre-packed
/// int8 activation fragment, and accumulate the 4x4 int32 fragment into the tile's
/// int32 accumulator (int32-EXACT). The weight is pre-packed FRAGMENT-MAJOR (one
/// 34-byte q8_0 block per 4x8 MAC fragment). `vmadotHelperName` is the FOUNDATION
/// single-fragment MAC helper this reuses.
std::string q80MatmulHelperBody(llvm::StringRef macKloopHelperName,
                                llvm::StringRef wideName, int njw) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// weft_ime.asm_leaf=" << macKloopHelperName
     << " tiled_q8_0_matmul mac=4x4x8 elem_in=int8 accum=int32 ime_op=vmadot "
        "weight_format=q8_0 int32_exact=1 register_resident_accumulate=1"
     << (njw > 1 ? " wide_deployed_njw=" + std::to_string(njw) : std::string())
     << "\n";
  os << "static void " << kQ80MatmulHelperName
     << "(const int8_t *Apack, const uint8_t *Bq8, int32_t *C,\n";
  os << "    long M, long N, long K) {\n";
  os << "  const long q80_block_bytes = 34; // fp16 d + 32 int8 quant bytes\n";
  emitFlatTiledMatmulLoop(os, "Bq8", "q80_block_bytes", kQ80DequantHelperName,
                          macKloopHelperName, wideName, njw);
  os << "}";
  os.flush();
  return text;
}

//===----------------------------------------------------------------------===//
// G4 M2b: the FORMAT-KEYED q4_K IME GEMM tile emitter (the SUPER-BLOCK K-quant
// sibling; the DEDICATED effort with the TWO-LEVEL 6-bit scale/min fold).
//
// The typed region (weft.ime.q4_K_matmul_tile) carries SIX decomposed bricks:
//   * weft.ime.q4_K_dequant_core -> the q4_K RAW-nibble decode (unsigned [0,15],
//     NOT q4_0 offset-binary), emitted as weft_ime_q4_K_dequant_fragment;
//   * weft.ime.q4_K_scale_min_unpack_core -> the canonical 6-bit get_scale_min_k4
//     bit-unpack, emitted as weft_ime_q4_K_get_scale_min;
//   * weft.ime.vmadot_mac_leaf -> the SAME FOUNDATION vmadot MAC (reused verbatim),
//     producing the per-sub-block sumi_b = Sum A_i*q_i;
//   * weft.ime.q4_K_scale_weighted_accum -> S_scale += sc_b*sumi_b (int32-exact);
//   * weft.ime.q4_K_min_bias_accum -> S_min += m_b*asum_b (int32-exact; asum_b is
//     the pure activation sub-block sum vmadot cannot express);
//   * weft.ime.q4_K_matmul_tile_yield -> the two-tile (S_scale, S_min) terminator.
// The int32-EXACT core (S_scale, S_min) is what the K1 seal validates bit-exact;
// the d/dmin fp16 fold C = d*S_scale - dmin*S_min is the SOLE deferred float
// epilogue (the two-level kquant_dmin_bsums_min fold, mirroring the RVV precedent).
//===----------------------------------------------------------------------===//
constexpr llvm::StringLiteral kQ4KDequantHelperName(
    "weft_ime_q4_K_dequant_fragment");
constexpr llvm::StringLiteral kQ4KScaleMinHelperName(
    "weft_ime_q4_K_get_scale_min");
constexpr llvm::StringLiteral kQ4KFp16HelperName("weft_ime_fp16_to_f32");
constexpr llvm::StringLiteral kQ4KMatmulHelperName(
    "weft_ime_q4_K_vmadot_matmul");

/// The q4_K RAW-nibble DECODE helper (structured C, no asm). Decodes 8 nibbles of
/// one column's super-block (sub-block b, fragment kf) into an 8-int8 vmadot lane
/// group. q4_K nibbles are UNSIGNED [0,15] (NO offset-binary centering); the min
/// bias is applied separately downstream, so this is a PURE integer transform.
std::string q4KDequantHelperBody() {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// weft_ime.decode_core=" << kQ4KDequantHelperName
     << " decode_model=q4_K_raw_nibble qk=256 weight_block_stride=144 "
        "weight_quant_byte_offset=16\n";
  os << "static inline void " << kQ4KDequantHelperName
     << "(const uint8_t *blk, int b, int kf, int8_t *out8) {\n";
  os << "  const uint8_t *qs = blk + 16; // past fp16 d/dmin + 12-byte 6-bit "
        "scales\n";
  os << "  for (int kl = 0; kl < 8; ++kl) {\n";
  os << "    int pl = kf * 8 + kl;\n";
  os << "    uint8_t byte = qs[(b / 2) * 32 + pl];\n";
  os << "    out8[kl] = (int8_t)((b & 1) ? (byte >> 4) : (byte & 0x0F));\n";
  os << "  }\n";
  os << "}";
  os.flush();
  return text;
}

/// The q4_K 6-bit scale/min UNPACK helper (canonical ggml get_scale_min_k4). For
/// sub-block j in [0,8) it unpacks the 6-bit scale `sc` and min `m` from the
/// 12-byte packed scales region.
std::string q4KScaleMinHelperBody() {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// weft_ime.scale_min_core=" << kQ4KScaleMinHelperName
     << " scale_min_model=get_scale_min_k4 num_sub_blocks=8 scale_bits=6 "
        "k_scale_size=12\n";
  os << "static inline void " << kQ4KScaleMinHelperName
     << "(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {\n";
  os << "  if (j < 4) {\n";
  os << "    *sc = q[j] & 63;\n";
  os << "    *m = q[j + 4] & 63;\n";
  os << "  } else {\n";
  os << "    *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);\n";
  os << "    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);\n";
  os << "  }\n";
  os << "}";
  os.flush();
  return text;
}

/// The q4_K deferred fp16 epilogue helpers (deterministic IEEE half->float). Only
/// the d/dmin float fold uses these; the int32 core (S_scale/S_min) is fp16-free.
std::string q4KFp16HelperBody() {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// weft_ime.fp16_epilogue=" << kQ4KFp16HelperName
     << " (deterministic IEEE half->float; the deferred d/dmin float fold)\n";
  os << "static inline unsigned short weft_ime_load_fp16(const uint8_t *p) {\n";
  os << "  return (unsigned short)((unsigned)p[0] | ((unsigned)p[1] << 8));\n";
  os << "}\n";
  os << "static inline float " << kQ4KFp16HelperName
     << "(unsigned short h) {\n";
  os << "  unsigned int sign = (unsigned int)(h & 0x8000u) << 16;\n";
  os << "  unsigned int exp = (h >> 10) & 0x1Fu;\n";
  os << "  unsigned int mant = h & 0x3FFu;\n";
  os << "  unsigned int bits;\n";
  os << "  if (exp == 0u) {\n";
  os << "    if (mant == 0u) {\n";
  os << "      bits = sign;\n";
  os << "    } else {\n";
  os << "      exp = 127u - 15u + 1u;\n";
  os << "      while ((mant & 0x400u) == 0u) { mant <<= 1; exp--; }\n";
  os << "      mant &= 0x3FFu;\n";
  os << "      bits = sign | (exp << 23) | (mant << 13);\n";
  os << "    }\n";
  os << "  } else if (exp == 0x1Fu) {\n";
  os << "    bits = sign | 0x7F800000u | (mant << 13);\n";
  os << "  } else {\n";
  os << "    bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);\n";
  os << "  }\n";
  os << "  float f;\n";
  os << "  __builtin_memcpy(&f, &bits, 4);\n";
  os << "  return f;\n";
  os << "}";
  os.flush();
  return text;
}

/// The tiled q4_K int8->int32 GEMM helper (the SUPER-BLOCK K-quant kernel). Per
/// (mi,nj) output tile it walks the K/256 super-blocks; per super-block it unpacks
/// the 8 per-column 6-bit sc/m, runs the 8 sub-blocks (each = 4 vmadot fragments
/// -> sumi_b + the pure activation sub-block sum asum_b), and folds the TWO-LEVEL
/// int32-EXACT core S_scale = Sum_b sc_b*sumi_b + S_min = Sum_b m_b*asum_b (the
/// board-seal object), then the deferred fp16 epilogue C = d*S_scale - dmin*S_min.
/// The weight is pre-packed as 4 native block_q4_K per (col-tile, super-block).
/// `vmadotHelperName` is the FOUNDATION single-fragment MAC helper this reuses.
std::string q4KMatmulHelperBody(llvm::StringRef macKloopHelperName) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// weft_ime.asm_leaf=" << macKloopHelperName
     << " tiled_q4_K_matmul mac=4x4x8 elem_in=int8 accum=int32 ime_op=vmadot "
        "weight_format=q4_K int32_exact=1 two_level_fold=kquant_dmin_bsums_min "
        "register_resident_accumulate=1\n";
  os << "static void " << kQ4KMatmulHelperName
     << "(const int8_t *Apack, const uint8_t *Bq4k, int32_t *Sscale,\n";
  os << "    int32_t *Smin, float *Cf, long M, long N, long K) {\n";
  os << "  const long mt = M / 4, nt = N / 4, nsb = K / 256;\n";
  os << "  const long q4k_block_bytes = 144; // fp16 d+dmin + 12B scales + 128B "
        "nibbles\n";
  os << "  for (long mi = 0; mi < mt; ++mi) {\n";
  os << "    const int8_t *Arow = Apack + (long)mi * 4 * K;\n";
  os << "    for (long nj = 0; nj < nt; ++nj) {\n";
  os << "      for (long sb = 0; sb < nsb; ++sb) {\n";
  os << "        const uint8_t *blk[4];\n";
  os << "        uint8_t sc[8][4], mm[8][4];\n";
  os << "        for (int nl = 0; nl < 4; ++nl) {\n";
  os << "          blk[nl] = Bq4k + ((((nj * nsb) + sb) * 4) + nl) * "
        "q4k_block_bytes;\n";
  os << "          for (int b = 0; b < 8; ++b)\n";
  os << "            " << kQ4KScaleMinHelperName
     << "(b, blk[nl] + 4, &sc[b][nl], &mm[b][nl]);\n";
  os << "        }\n";
  os << "        int32_t Sc[16], Sm[16];\n";
  os << "        for (int r = 0; r < 16; ++r) { Sc[r] = 0; Sm[r] = 0; }\n";
  os << "        for (int b = 0; b < 8; ++b) {\n";
  os << "          int32_t sumi[16];\n";
  os << "          int32_t asum[4] = {0, 0, 0, 0};\n";
  // Decode the sub-block's 4 MAC fragments into a contiguous int8 buffer (and
  // sum the activation sub-block asum_b alongside -- vmadot cannot express it),
  // then run ONE register-resident batched MAC over the 4 fragments (v2/v3
  // accumulate, single vsetvli, single store) => sumi_b = Sum_kf A_kf . B_kf^T.
  // The 4 A fragments (kf=0..3) are contiguous at Arow + (sb*32+b*4)*32.
  os << "          int8_t Bdec[128];\n";
  os << "          for (int kf = 0; kf < 4; ++kf) {\n";
  os << "            long gf = sb * 32 + b * 4 + kf;\n";
  os << "            const int8_t *Aframe = Arow + gf * 32;\n";
  os << "            for (int nl = 0; nl < 4; ++nl)\n";
  os << "              " << kQ4KDequantHelperName
     << "(blk[nl], b, kf, Bdec + kf * 32 + nl * 8);\n";
  os << "            for (int ml = 0; ml < 4; ++ml)\n";
  os << "              for (int kl = 0; kl < 8; ++kl)\n";
  os << "                asum[ml] += (int32_t)Aframe[ml * 8 + kl];\n";
  os << "          }\n";
  os << "          const int8_t *Ablk = Arow + (long)(sb * 32 + b * 4) * 32;\n";
  os << "          " << macKloopHelperName << "(Ablk, Bdec, 4, sumi);\n";
  os << "          for (int ml = 0; ml < 4; ++ml)\n";
  os << "            for (int nl = 0; nl < 4; ++nl) {\n";
  os << "              Sc[ml * 4 + nl] += (int32_t)sc[b][nl] * sumi[ml * 4 + "
        "nl];\n";
  os << "              Sm[ml * 4 + nl] += (int32_t)mm[b][nl] * asum[ml];\n";
  os << "            }\n";
  os << "        }\n";
  os << "        for (int ml = 0; ml < 4; ++ml)\n";
  os << "          for (int nl = 0; nl < 4; ++nl) {\n";
  os << "            long mo = mi * 4 + ml, no = nj * 4 + nl;\n";
  os << "            long oidx = sb * M * N + mo * N + no;\n";
  os << "            Sscale[oidx] = Sc[ml * 4 + nl];\n";
  os << "            Smin[oidx] = Sm[ml * 4 + nl];\n";
  os << "            float d = " << kQ4KFp16HelperName
     << "(weft_ime_load_fp16(blk[nl] + 0));\n";
  os << "            float dmin = " << kQ4KFp16HelperName
     << "(weft_ime_load_fp16(blk[nl] + 2));\n";
  os << "            Cf[mo * N + no] += d * (float)Sc[ml * 4 + nl] - dmin * "
        "(float)Sm[ml * 4 + nl];\n";
  os << "          }\n";
  os << "      }\n";
  os << "    }\n";
  os << "  }\n";
  os << "}";
  os.flush();
  return text;
}

/// Lowers a selected IME MAC boundary (`weft.ime.mma` signed / `weft.ime.mma_u`
/// unsigned) into a standalone EmitC module:
///   #include <stdint.h>
///   static inline void weft_ime_vmadot[u]_mma_4x4x8(const int8_t*,
///                              const int8_t*, int32_t*) { __asm__(...vmadot[u]...) }
///   extern "C" void weft_emitc_<kernel>_<variant>(const int8_t *A,
///                                                  const int8_t *B,
///                                                  int32_t *C) {
///     // route_source_op + source_op provenance comments
///     weft_ime_vmadot[u]_mma_4x4x8(A, B, C);
///   }
/// The wrapper is structured emitc (emitc.func + emitc.call_opaque on the A/B/C
/// block args); the single asm leaf is confined to the helper. The pattern is
/// templated over the op type; the static traits pin the helper name + body, so
/// the signed/unsigned divergence is ONLY the emitted instruction leaf.
template <typename OpT>
class IMEMACToEmitCFunc final : public mlir::OpConversionPattern<OpT> {
public:
  using mlir::OpConversionPattern<OpT>::OpConversionPattern;
  using OpAdaptor = typename mlir::OpConversionPattern<OpT>::OpAdaptor;

  mlir::LogicalResult
  matchAndRewrite(OpT mma, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MLIRContext *context = mma.getContext();
    mlir::Location loc = mma.getLoc();

    llvm::StringRef helperName = IMEMACToEmitCFunc::helperName();
    std::string helperBody = IMEMACToEmitCFunc::helperBody();

    auto variant =
        mma->template getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
    auto sourceKernel =
        mma->template getAttrOfType<mlir::StringAttr>("source_kernel");
    if (!variant || !sourceKernel)
      return rewriter.notifyMatchFailure(
          mma, "IME MAC boundary requires selected_variant and source_kernel "
               "attributes");
    std::string functionName =
        ("weft_emitc_" + sourceKernel.getValue() + "_" + variant.getValue())
            .str();

    llvm::StringRef sourceOpName = mma.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef sourceRole = mma.getWEFTEmitCLowerableSourceRole();

    auto module = mma->template getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(mma, "IME MAC boundary has no module");

    auto i8PtrType = emitc::PointerType::get(
        context, emitc::OpaqueType::get(context, "const int8_t"));
    auto i32PtrType = emitc::PointerType::get(
        context, emitc::OpaqueType::get(context, "int32_t"));

    // Module-scope prologue: include + the self-contained asm-leaf helper.
    {
      mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<emitc::IncludeOp>(loc, "stdint.h",
                                        /*is_standard_include=*/true);
      rewriter.create<emitc::VerbatimOp>(loc, helperBody);
    }

    mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());

    // Exported wrapper: extern "C" void <name>(const int8_t*, const int8_t*,
    // int32_t*).
    llvm::SmallVector<mlir::Type, 3> paramTypes{i8PtrType, i8PtrType,
                                                i32PtrType};
    mlir::FunctionType functionType =
        rewriter.getFunctionType(paramTypes, /*results=*/{});
    llvm::SmallVector<mlir::NamedAttribute, 1> funcAttrs;
    funcAttrs.push_back(rewriter.getNamedAttr(
        "specifiers", rewriter.getStrArrayAttr({"extern", "\"C\""})));
    auto func = rewriter.create<emitc::FuncOp>(loc, functionName, functionType,
                                               funcAttrs);
    mlir::Block *entry = func.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);

    rewriter.create<emitc::VerbatimOp>(
        loc, routeSourceComment(sourceOpName, sourceRole));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(sourceOpName, sourceRole, helperName));

    // Structured dataflow into the leaf: call_opaque on the A/B/C block args.
    llvm::SmallVector<mlir::Value, 3> callOperands;
    for (mlir::BlockArgument arg : entry->getArguments())
      callOperands.push_back(arg);
    rewriter.create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{}, helperName,
                                         callOperands);

    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());

    rewriter.eraseOp(mma);
    return mlir::success();
  }

private:
  static llvm::StringRef helperName();
  static std::string helperBody();
};

// Signed surface: weft.ime.mma -> the vmadot asm leaf (the first slice).
template <>
llvm::StringRef IMEMACToEmitCFunc<weft::ime::MMAOp>::helperName() {
  return kVmadotHelperName;
}
template <> std::string IMEMACToEmitCFunc<weft::ime::MMAOp>::helperBody() {
  return vmadotHelperBody();
}

// Unsigned surface: weft.ime.mma_u -> the vmadotu asm leaf (the second op).
template <>
llvm::StringRef IMEMACToEmitCFunc<weft::ime::MMAUOp>::helperName() {
  return kVmadotuHelperName;
}
template <> std::string IMEMACToEmitCFunc<weft::ime::MMAUOp>::helperBody() {
  return vmadotuHelperBody();
}

// Mixed-sign surface: weft.ime.mma_su -> the vmadotsu asm leaf (the fourth op).
template <>
llvm::StringRef IMEMACToEmitCFunc<weft::ime::MMASUOp>::helperName() {
  return kVmadotsuHelperName;
}
template <> std::string IMEMACToEmitCFunc<weft::ime::MMASUOp>::helperBody() {
  return vmadotsuHelperBody();
}

// Reversed-order mixed-sign surface: weft.ime.mma_us -> the vmadotus asm leaf
// (the signedness-family-completing op: unsigned A * signed B).
template <>
llvm::StringRef IMEMACToEmitCFunc<weft::ime::MMAUSOp>::helperName() {
  return kVmadotusHelperName;
}
template <> std::string IMEMACToEmitCFunc<weft::ime::MMAUSOp>::helperBody() {
  return vmadotusHelperBody();
}

/// Lowers the sliding-window IME boundary (`weft.ime.mma_slide`) into a
/// standalone EmitC module. UNLIKE IMEMACToEmitCFunc, the emitted leaf depends
/// on the `slide` window FACT carried on the op (1=>vmadot1, 2=>vmadot2,
/// 3=>vmadot3) — a pure data flow of the capability-derived fact, NOT a
/// family-name branch. The helper loads A as an even VS1:VS1+1 pair (8x8 int8)
/// and runs the slide MAC leaf. The structured EmitC wrapper (func signature +
/// A/B/C ptr args + the call) is identical to the non-slide MAC; only the helper
/// body/name differ.
class IMEMACSlideToEmitCFunc final
    : public mlir::OpConversionPattern<weft::ime::MMASlideOp> {
public:
  using mlir::OpConversionPattern<weft::ime::MMASlideOp>::OpConversionPattern;
  using OpAdaptor =
      typename mlir::OpConversionPattern<weft::ime::MMASlideOp>::OpAdaptor;

  mlir::LogicalResult
  matchAndRewrite(weft::ime::MMASlideOp mma, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MLIRContext *context = mma.getContext();
    mlir::Location loc = mma.getLoc();

    // The slide window FACT selects the emitted leaf (a data flow of the
    // capability-derived fact stamped on the op, not a family-name branch).
    int64_t slide = mma.getSlide();
    llvm::StringRef helperName = slide == 1   ? kVmadot1SlideHelperName
                                 : slide == 2 ? kVmadot2SlideHelperName
                                              : kVmadot3SlideHelperName;
    std::string helperBody = slide == 1   ? vmadot1SlideHelperBody()
                             : slide == 2 ? vmadot2SlideHelperBody()
                                          : vmadot3SlideHelperBody();

    auto variant =
        mma->getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
    auto sourceKernel = mma->getAttrOfType<mlir::StringAttr>("source_kernel");
    if (!variant || !sourceKernel)
      return rewriter.notifyMatchFailure(
          mma, "IME slide MAC boundary requires selected_variant and "
               "source_kernel attributes");
    std::string functionName =
        ("weft_emitc_" + sourceKernel.getValue() + "_" + variant.getValue())
            .str();

    llvm::StringRef sourceOpName = mma.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef sourceRole = mma.getWEFTEmitCLowerableSourceRole();

    auto module = mma->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(mma,
                                         "IME slide MAC boundary has no module");

    auto i8PtrType = emitc::PointerType::get(
        context, emitc::OpaqueType::get(context, "const int8_t"));
    auto i32PtrType = emitc::PointerType::get(
        context, emitc::OpaqueType::get(context, "int32_t"));

    // Module-scope prologue: include + the self-contained asm-leaf helper.
    {
      mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<emitc::IncludeOp>(loc, "stdint.h",
                                        /*is_standard_include=*/true);
      rewriter.create<emitc::VerbatimOp>(loc, helperBody);
    }

    mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());

    llvm::SmallVector<mlir::Type, 3> paramTypes{i8PtrType, i8PtrType,
                                                i32PtrType};
    mlir::FunctionType functionType =
        rewriter.getFunctionType(paramTypes, /*results=*/{});
    llvm::SmallVector<mlir::NamedAttribute, 1> funcAttrs;
    funcAttrs.push_back(rewriter.getNamedAttr(
        "specifiers", rewriter.getStrArrayAttr({"extern", "\"C\""})));
    auto func = rewriter.create<emitc::FuncOp>(loc, functionName, functionType,
                                               funcAttrs);
    mlir::Block *entry = func.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);

    rewriter.create<emitc::VerbatimOp>(
        loc, routeSourceComment(sourceOpName, sourceRole));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(sourceOpName, sourceRole, helperName));

    llvm::SmallVector<mlir::Value, 3> callOperands;
    for (mlir::BlockArgument arg : entry->getArguments())
      callOperands.push_back(arg);
    rewriter.create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{}, helperName,
                                         callOperands);

    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());

    rewriter.eraseOp(mma);
    return mlir::success();
  }
};

/// Lowers the tiled whole-matrix IME boundary (`weft.ime.matmul`) into a
/// standalone EmitC module:
///   #include <stdint.h>
///   static inline void weft_ime_vmadot[u]_matmul(const int8_t*, const int8_t*,
///                                 int32_t*, long M, long N, long K) { ...tiled... }
///   extern "C" void weft_emitc_<kernel>_<variant>(const int8_t *Apack,
///                                                  const int8_t *Bpack,
///                                                  int32_t *C) {
///     weft_ime_vmadot[u]_matmul(Apack, Bpack, C, <M>, <N>, <K>);
///   }
/// M/N/K are the op's capability-bound problem-dim FACTS (mat_m/mat_n/mat_k),
/// baked into the wrapper as constants (they are compile-time facts of the
/// selected variant). The wrapper is structured emitc (emitc.func +
/// emitc.call_opaque on the Apack/Bpack/C block args + emitc.constant M/N/K);
/// the verbatim helper holds the validated vmadot/vmadotu plus its
/// load/store/K-loop scaffold. Signed vs unsigned is the `ime_op` fact (NOT a
/// family-name branch): vmadot => signed helper, vmadotu => unsigned helper.
class IMEMatMulToEmitCFunc final
    : public mlir::OpConversionPattern<weft::ime::MatMulOp> {
public:
  using mlir::OpConversionPattern<weft::ime::MatMulOp>::OpConversionPattern;
  using OpAdaptor =
      typename mlir::OpConversionPattern<weft::ime::MatMulOp>::OpAdaptor;

  mlir::LogicalResult
  matchAndRewrite(weft::ime::MatMulOp matmul, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MLIRContext *context = matmul.getContext();
    mlir::Location loc = matmul.getLoc();

    // Signed/unsigned is a capability-derived FACT carried on the op, not a
    // family-name branch: ime_op == "vmadotu" => unsigned helper, else signed.
    bool isUnsigned = matmul.getImeOp() == "vmadotu";
    llvm::StringRef helperName =
        isUnsigned ? kMatmulUHelperName : kMatmulHelperName;
    std::string helperBody =
        isUnsigned ? matmulHelperBodyUnsigned() : matmulHelperBodySigned();

    auto variant =
        matmul->getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
    auto sourceKernel =
        matmul->getAttrOfType<mlir::StringAttr>("source_kernel");
    if (!variant || !sourceKernel)
      return rewriter.notifyMatchFailure(
          matmul, "IME matmul boundary requires selected_variant and "
                  "source_kernel attributes");
    std::string functionName =
        ("weft_emitc_" + sourceKernel.getValue() + "_" + variant.getValue())
            .str();

    int64_t matM = matmul.getMatM();
    int64_t matN = matmul.getMatN();
    int64_t matK = matmul.getMatK();

    llvm::StringRef sourceOpName = matmul.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef sourceRole = matmul.getWEFTEmitCLowerableSourceRole();

    auto module = matmul->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(matmul,
                                         "IME matmul boundary has no module");

    auto i8PtrType = emitc::PointerType::get(
        context, emitc::OpaqueType::get(context, "const int8_t"));
    auto i32PtrType = emitc::PointerType::get(
        context, emitc::OpaqueType::get(context, "int32_t"));
    auto longType = emitc::OpaqueType::get(context, "long");

    // Module-scope prologue: include + the self-contained tiled-kernel helper.
    {
      mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<emitc::IncludeOp>(loc, "stdint.h",
                                        /*is_standard_include=*/true);
      rewriter.create<emitc::VerbatimOp>(loc, helperBody);
    }

    mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());

    // Exported wrapper: extern "C" void <name>(const int8_t* Apack,
    // const int8_t* Bpack, int32_t* C).
    llvm::SmallVector<mlir::Type, 3> paramTypes{i8PtrType, i8PtrType,
                                                i32PtrType};
    mlir::FunctionType functionType =
        rewriter.getFunctionType(paramTypes, /*results=*/{});
    llvm::SmallVector<mlir::NamedAttribute, 1> funcAttrs;
    funcAttrs.push_back(rewriter.getNamedAttr(
        "specifiers", rewriter.getStrArrayAttr({"extern", "\"C\""})));
    auto func = rewriter.create<emitc::FuncOp>(loc, functionName, functionType,
                                               funcAttrs);
    mlir::Block *entry = func.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);

    rewriter.create<emitc::VerbatimOp>(
        loc, routeSourceComment(sourceOpName, sourceRole));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(sourceOpName, sourceRole, helperName));

    // Structured dataflow into the leaf: call_opaque on the Apack/Bpack/C block
    // args plus the M/N/K problem-dim constants (compile-time variant facts).
    llvm::SmallVector<mlir::Value, 6> callOperands;
    for (mlir::BlockArgument arg : entry->getArguments())
      callOperands.push_back(arg);
    for (int64_t dim : {matM, matN, matK}) {
      auto dimAttr = emitc::OpaqueAttr::get(context, std::to_string(dim));
      auto constOp = rewriter.create<emitc::ConstantOp>(loc, longType, dimAttr);
      callOperands.push_back(constOp.getResult());
    }
    rewriter.create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{}, helperName,
                                         callOperands);

    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());

    rewriter.eraseOp(matmul);
    return mlir::success();
  }
};

/// Lowers the FORMAT-KEYED q4_0 IME GEMM tile (`weft.ime.q4_0_matmul_tile`) into a
/// standalone EmitC module. Unlike IMEMatMulToEmitCFunc (format-agnostic pre-packed
/// int8), this op OWNS a typed region carrying the DECOMPOSED q4_0-decode + vmadot
/// MAC bricks; emission reads the region by OP-IDENTITY (I5) -- the presence of the
/// weft.ime.q4_0_dequant_core decode brick keys the q4_0 dequant helper, the
/// weft.ime.vmadot_mac_leaf brick keys the validated vmadot MAC leaf -- never a q4/q8
/// name string. The emitted module:
///   #include <stdint.h>
///   static inline void weft_ime_vmadot_mma_4x4x8(...) { __asm__(...vmadot...) }
///   static inline void weft_ime_q4_0_dequant_fragment(const uint8_t*, int8_t*) {...}
///   static inline void weft_ime_q4_0_vmadot_matmul(const int8_t* Apack,
///                          const uint8_t* Bq4, int32_t* C, long M,N,K) {...}
///   extern "C" void weft_emitc_<kernel>_<variant>(const int8_t* Apack,
///                          const uint8_t* Bq4, int32_t* C) { ...matmul(...M,N,K); }
/// M/N/K are the tile op's compile-time problem-dim FACTS (mat_m/mat_n/mat_k). The
/// int32 output is the int32-EXACT MAC accumulator (the M1b K1-seal bit-exact
/// contract); the per-block fp16 scale fold is a separate downstream epilogue.
class IMEQ40MatMulTileToEmitCFunc final
    : public mlir::OpConversionPattern<weft::ime::Q40MatMulTileOp> {
public:
  using mlir::OpConversionPattern<weft::ime::Q40MatMulTileOp>::OpConversionPattern;
  using OpAdaptor =
      typename mlir::OpConversionPattern<weft::ime::Q40MatMulTileOp>::OpAdaptor;

  mlir::LogicalResult
  matchAndRewrite(weft::ime::Q40MatMulTileOp tile, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MLIRContext *context = tile.getContext();
    mlir::Location loc = tile.getLoc();

    // Read the typed region by OP-IDENTITY (I5): the decode core keys the q4_0
    // dequant helper, the vmadot leaf keys the validated MAC leaf. Absence of
    // either brick is a fail-closed match failure (never emit an opaque body).
    mlir::Block &body = tile.getBody().front();
    auto dequantCores = body.getOps<weft::ime::Q40DequantCoreOp>();
    auto macLeaves = body.getOps<weft::ime::VmadotMacLeafOp>();
    if (dequantCores.empty() || macLeaves.empty())
      return rewriter.notifyMatchFailure(
          tile, "q4_0 tile region must carry the q4_0_dequant_core + "
                "vmadot_mac_leaf bricks");
    // The MAC leaf's ime_op fact keys the emitted MAC instruction (vmadot); the
    // leaf is the register-resident BATCHED K/8-loop form ([GAP-IME-LEAF-PIPELINE]).
    llvm::StringRef macKloopHelperName = kVmadotMacKloopHelperName;

    auto variant =
        tile->getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
    auto sourceKernel =
        tile->getAttrOfType<mlir::StringAttr>("source_kernel");
    if (!variant || !sourceKernel)
      return rewriter.notifyMatchFailure(
          tile, "IME q4_0 tile requires selected_variant and source_kernel "
                "attributes");
    std::string functionName =
        ("weft_emitc_" + sourceKernel.getValue() + "_" + variant.getValue())
            .str();

    int64_t matM = tile.getMatM();
    int64_t matN = tile.getMatN();
    int64_t matK = tile.getMatK();

    llvm::StringRef sourceOpName = tile.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef sourceRole = tile.getWEFTEmitCLowerableSourceRole();

    auto module = tile->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(tile, "IME q4_0 tile has no module");

    auto i8PtrType = emitc::PointerType::get(
        context, emitc::OpaqueType::get(context, "const int8_t"));
    auto u8PtrType = emitc::PointerType::get(
        context, emitc::OpaqueType::get(context, "const uint8_t"));
    auto i32PtrType = emitc::PointerType::get(
        context, emitc::OpaqueType::get(context, "int32_t"));
    auto longType = emitc::OpaqueType::get(context, "long");

    // G8 wide-vmadot key: the VLEN-parametric capability predicate (q4_0 is a flat
    // MAC-BOUND format -> the bottleneck-shape leg admits the wide tiling). The
    // selected wide leaf is DEPLOYED into both the int32 seal kernel and the f32
    // forward kernel below (closing applied!=deployed); narrow is the auto fallback.
    IMEWideDeployDecision wide = decideWideVmadotDeployment(
        readDeployedVlenBits(tile), tile.getMacM(), tile.getMacN(),
        tile.getMacK(), tile.getElemInBits(), tile.getAccumBits(),
        /*epilogueBound=*/false);
    std::string wideName;

    // Module-scope prologue: include + the validated vmadot MAC leaf + the DEPLOYED
    // wide leaf (declared-before-use, only when selected) + the q4_0 decode helper +
    // the tiled q4_0 kernel.
    {
      mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<emitc::IncludeOp>(loc, "stdint.h",
                                        /*is_standard_include=*/true);
      rewriter.create<emitc::VerbatimOp>(loc, vmadotMacKloopHelperBody());
      wideName = emitDeployedWideVmadotLeaf(rewriter, loc, wide);
      rewriter.create<emitc::VerbatimOp>(loc, q40DequantHelperBody());
      rewriter.create<emitc::VerbatimOp>(
          loc, q40MatmulHelperBody(macKloopHelperName, wideName, wide.njw));
    }

    mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());

    // Exported wrapper: extern "C" void <name>(const int8_t* Apack,
    // const uint8_t* Bq4, int32_t* C).
    llvm::SmallVector<mlir::Type, 3> paramTypes{i8PtrType, u8PtrType,
                                                i32PtrType};
    mlir::FunctionType functionType =
        rewriter.getFunctionType(paramTypes, /*results=*/{});
    llvm::SmallVector<mlir::NamedAttribute, 1> funcAttrs;
    funcAttrs.push_back(rewriter.getNamedAttr(
        "specifiers", rewriter.getStrArrayAttr({"extern", "\"C\""})));
    auto func = rewriter.create<emitc::FuncOp>(loc, functionName, functionType,
                                               funcAttrs);
    mlir::Block *entry = func.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);

    rewriter.create<emitc::VerbatimOp>(
        loc, routeSourceComment(sourceOpName, sourceRole));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(sourceOpName, sourceRole, kQ40MatmulHelperName));

    // Structured dataflow into the leaf: call_opaque on the Apack/Bq4/C block
    // args plus the M/N/K problem-dim constants (compile-time variant facts).
    llvm::SmallVector<mlir::Value, 6> callOperands;
    for (mlir::BlockArgument arg : entry->getArguments())
      callOperands.push_back(arg);
    for (int64_t dim : {matM, matN, matK}) {
      auto dimAttr = emitc::OpaqueAttr::get(context, std::to_string(dim));
      auto constOp = rewriter.create<emitc::ConstantOp>(loc, longType, dimAttr);
      callOperands.push_back(constOp.getResult());
    }
    rewriter.create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{},
                                         kQ40MatmulHelperName, callOperands);

    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());

    // G5-M3: a SECOND exported wrapper for the forward-bridge scale-fold epilogue.
    // extern "C" void <name>_f32(const int8_t* Apack, const float* dA,
    //   const uint8_t* Bnib, const float* dW, float* Cf) -> the ggml q4_0 x q8_0
    // f32 mul_mat result (M/N/K baked problem-dim facts). This is what the ggml
    // forward hook calls; the int32 wrapper above stays the seal object.
    {
      mlir::OpBuilder::InsertionGuard f32Guard(rewriter);
      rewriter.setInsertionPointToEnd(module.getBody());
      // Emit the f32 scale-fold helper here (after the int32 wrapper, before the
      // f32 wrapper that uses it): declared-before-use, and a deterministic
      // module order (int32 kernel + wrapper, then f32 kernel + wrapper).
      rewriter.create<emitc::VerbatimOp>(
          loc, q40ScaleFoldMatmulHelperBody(macKloopHelperName, wideName,
                                            wide.njw));
      auto cf32PtrType = emitc::PointerType::get(
          context, emitc::OpaqueType::get(context, "const float"));
      auto f32PtrType = emitc::PointerType::get(
          context, emitc::OpaqueType::get(context, "float"));
      llvm::SmallVector<mlir::Type, 5> f32ParamTypes{
          i8PtrType, cf32PtrType, u8PtrType, cf32PtrType, f32PtrType};
      mlir::FunctionType f32FnType =
          rewriter.getFunctionType(f32ParamTypes, /*results=*/{});
      llvm::SmallVector<mlir::NamedAttribute, 1> f32FuncAttrs;
      f32FuncAttrs.push_back(rewriter.getNamedAttr(
          "specifiers", rewriter.getStrArrayAttr({"extern", "\"C\""})));
      auto f32Func = rewriter.create<emitc::FuncOp>(loc, functionName + "_f32",
                                                    f32FnType, f32FuncAttrs);
      mlir::Block *f32Entry = f32Func.addEntryBlock();
      rewriter.setInsertionPointToStart(f32Entry);
      rewriter.create<emitc::VerbatimOp>(loc,
                                         routeSourceComment(sourceOpName, sourceRole));
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(sourceOpName, sourceRole,
                           kQ40ScaleFoldMatmulHelperName));
      llvm::SmallVector<mlir::Value, 8> f32Operands;
      for (mlir::BlockArgument arg : f32Entry->getArguments())
        f32Operands.push_back(arg);
      for (int64_t dim : {matM, matN, matK}) {
        auto dimAttr = emitc::OpaqueAttr::get(context, std::to_string(dim));
        auto constOp =
            rewriter.create<emitc::ConstantOp>(loc, longType, dimAttr);
        f32Operands.push_back(constOp.getResult());
      }
      rewriter.create<emitc::CallOpaqueOp>(
          loc, mlir::TypeRange{}, kQ40ScaleFoldMatmulHelperName, f32Operands);
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
    }

    // G8: the WIDE leaf is now emitted in the PROLOGUE and DEPLOYED into both the
    // int32 seal kernel (weft_ime_q4_0_vmadot_matmul) and the f32 forward kernel
    // (..._f32) above -- no longer a dead module-END primitive (applied!=deployed
    // closed). Its perf-vs-narrow verdict is a STAGE-3 board (kernel-axis) remeasure;
    // this stage seals only the byte-exact int32 keying (wide sub-tile == narrow,
    // K1-sealed f5e77482).
    rewriter.eraseOp(tile);
    return mlir::success();
  }
};

/// Lowers the FORMAT-KEYED q8_0 IME GEMM tile (`weft.ime.q8_0_matmul_tile`) into a
/// standalone EmitC module (the FLAT-int8 copy-adapt sibling of
/// IMEQ40MatMulTileToEmitCFunc). This op OWNS a typed region carrying the
/// DECOMPOSED q8_0-decode (DIRECT int8 read) + vmadot MAC bricks; emission reads
/// the region by OP-IDENTITY (I5) -- the weft.ime.q8_0_dequant_core brick keys the
/// q8_0 direct-read helper, the weft.ime.vmadot_mac_leaf brick keys the validated
/// vmadot MAC leaf -- never a q4/q8 name string. The int32 output is the
/// int32-EXACT MAC accumulator (the K1-seal bit-exact contract).
class IMEQ80MatMulTileToEmitCFunc final
    : public mlir::OpConversionPattern<weft::ime::Q80MatMulTileOp> {
public:
  using mlir::OpConversionPattern<weft::ime::Q80MatMulTileOp>::OpConversionPattern;
  using OpAdaptor =
      typename mlir::OpConversionPattern<weft::ime::Q80MatMulTileOp>::OpAdaptor;

  mlir::LogicalResult
  matchAndRewrite(weft::ime::Q80MatMulTileOp tile, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MLIRContext *context = tile.getContext();
    mlir::Location loc = tile.getLoc();

    // Read the typed region by OP-IDENTITY (I5): the decode core keys the q8_0
    // direct-read helper, the vmadot leaf keys the validated MAC leaf. Absence of
    // either brick is a fail-closed match failure (never emit an opaque body).
    mlir::Block &body = tile.getBody().front();
    auto dequantCores = body.getOps<weft::ime::Q80DequantCoreOp>();
    auto macLeaves = body.getOps<weft::ime::VmadotMacLeafOp>();
    if (dequantCores.empty() || macLeaves.empty())
      return rewriter.notifyMatchFailure(
          tile, "q8_0 tile region must carry the q8_0_dequant_core + "
                "vmadot_mac_leaf bricks");
    // The MAC leaf's ime_op fact keys the emitted MAC instruction (vmadot); the
    // leaf is the register-resident BATCHED K/8-loop form ([GAP-IME-LEAF-PIPELINE]).
    llvm::StringRef macKloopHelperName = kVmadotMacKloopHelperName;

    auto variant =
        tile->getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
    auto sourceKernel =
        tile->getAttrOfType<mlir::StringAttr>("source_kernel");
    if (!variant || !sourceKernel)
      return rewriter.notifyMatchFailure(
          tile, "IME q8_0 tile requires selected_variant and source_kernel "
                "attributes");
    std::string functionName =
        ("weft_emitc_" + sourceKernel.getValue() + "_" + variant.getValue())
            .str();

    int64_t matM = tile.getMatM();
    int64_t matN = tile.getMatN();
    int64_t matK = tile.getMatK();

    llvm::StringRef sourceOpName = tile.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef sourceRole = tile.getWEFTEmitCLowerableSourceRole();

    auto module = tile->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(tile, "IME q8_0 tile has no module");

    auto i8PtrType = emitc::PointerType::get(
        context, emitc::OpaqueType::get(context, "const int8_t"));
    auto u8PtrType = emitc::PointerType::get(
        context, emitc::OpaqueType::get(context, "const uint8_t"));
    auto i32PtrType = emitc::PointerType::get(
        context, emitc::OpaqueType::get(context, "int32_t"));
    auto longType = emitc::OpaqueType::get(context, "long");

    // G8 wide-vmadot key: q8_0 is a flat MAC-BOUND format (direct int8 decode, light
    // int32 scatter epilogue) -> the bottleneck-shape leg admits the wide tiling, so
    // the selected wide leaf is DEPLOYED into the int32 kernel below (closing
    // applied!=deployed for q8_0). narrow is the auto fallback.
    IMEWideDeployDecision wide = decideWideVmadotDeployment(
        readDeployedVlenBits(tile), tile.getMacM(), tile.getMacN(),
        tile.getMacK(), tile.getElemInBits(), tile.getAccumBits(),
        /*epilogueBound=*/false);
    std::string wideName;

    // Module-scope prologue: include + the validated vmadot MAC leaf + the DEPLOYED
    // wide leaf (declared-before-use, only when selected) + the q8_0 decode helper +
    // the tiled q8_0 kernel.
    {
      mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<emitc::IncludeOp>(loc, "stdint.h",
                                        /*is_standard_include=*/true);
      rewriter.create<emitc::VerbatimOp>(loc, vmadotMacKloopHelperBody());
      wideName = emitDeployedWideVmadotLeaf(rewriter, loc, wide);
      rewriter.create<emitc::VerbatimOp>(loc, q80DequantHelperBody());
      rewriter.create<emitc::VerbatimOp>(
          loc, q80MatmulHelperBody(macKloopHelperName, wideName, wide.njw));
    }

    mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());

    // Exported wrapper: extern "C" void <name>(const int8_t* Apack,
    // const uint8_t* Bq8, int32_t* C).
    llvm::SmallVector<mlir::Type, 3> paramTypes{i8PtrType, u8PtrType,
                                                i32PtrType};
    mlir::FunctionType functionType =
        rewriter.getFunctionType(paramTypes, /*results=*/{});
    llvm::SmallVector<mlir::NamedAttribute, 1> funcAttrs;
    funcAttrs.push_back(rewriter.getNamedAttr(
        "specifiers", rewriter.getStrArrayAttr({"extern", "\"C\""})));
    auto func = rewriter.create<emitc::FuncOp>(loc, functionName, functionType,
                                               funcAttrs);
    mlir::Block *entry = func.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);

    rewriter.create<emitc::VerbatimOp>(
        loc, routeSourceComment(sourceOpName, sourceRole));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(sourceOpName, sourceRole, kQ80MatmulHelperName));

    // Structured dataflow into the leaf: call_opaque on the Apack/Bq8/C block
    // args plus the M/N/K problem-dim constants (compile-time variant facts).
    llvm::SmallVector<mlir::Value, 6> callOperands;
    for (mlir::BlockArgument arg : entry->getArguments())
      callOperands.push_back(arg);
    for (int64_t dim : {matM, matN, matK}) {
      auto dimAttr = emitc::OpaqueAttr::get(context, std::to_string(dim));
      auto constOp = rewriter.create<emitc::ConstantOp>(loc, longType, dimAttr);
      callOperands.push_back(constOp.getResult());
    }
    rewriter.create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{},
                                         kQ80MatmulHelperName, callOperands);

    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());

    rewriter.eraseOp(tile);
    return mlir::success();
  }
};

/// Lowers the FORMAT-KEYED q4_K IME GEMM tile (`weft.ime.q4_K_matmul_tile`) into a
/// standalone EmitC module (the SUPER-BLOCK K-quant sibling; the DEDICATED effort
/// beyond the q4_0/q8_0 single-decode tiles). This op OWNS a typed region carrying
/// the SIX DECOMPOSED bricks -- the raw-nibble decode, the 6-bit scale/min unpack,
/// the reused vmadot MAC, the scale-weighted accum (S_scale), the min-bias accum
/// (S_min), and the two-tile yield; emission reads the region by OP-IDENTITY (I5).
/// The int32 S_scale/S_min core is the K1-seal bit-exact contract; the d/dmin fp16
/// fold is the sole deferred float epilogue.
class IMEQ4KMatMulTileToEmitCFunc final
    : public mlir::OpConversionPattern<weft::ime::Q4KMatMulTileOp> {
public:
  using mlir::OpConversionPattern<weft::ime::Q4KMatMulTileOp>::OpConversionPattern;
  using OpAdaptor =
      typename mlir::OpConversionPattern<weft::ime::Q4KMatMulTileOp>::OpAdaptor;

  mlir::LogicalResult
  matchAndRewrite(weft::ime::Q4KMatMulTileOp tile, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MLIRContext *context = tile.getContext();
    mlir::Location loc = tile.getLoc();

    // Read the typed region by OP-IDENTITY (I5). The two-level q4_K fold requires
    // ALL of: the raw-nibble decode, the 6-bit scale/min unpack, the vmadot MAC,
    // the scale-weighted accum (S_scale), and the min-bias accum (S_min). Absence
    // of ANY brick -- ESPECIALLY the scale-weighted or min-bias accum (the HOLLOW
    // bare-MAC shape) -- is a fail-closed match failure.
    mlir::Block &body = tile.getBody().front();
    auto dequantCores = body.getOps<weft::ime::Q4KDequantCoreOp>();
    auto scaleMinCores = body.getOps<weft::ime::Q4KScaleMinUnpackCoreOp>();
    auto macLeaves = body.getOps<weft::ime::VmadotMacLeafOp>();
    auto scaleAccums = body.getOps<weft::ime::Q4KScaleWeightedAccumOp>();
    auto minBiasAccums = body.getOps<weft::ime::Q4KMinBiasAccumOp>();
    if (dequantCores.empty() || scaleMinCores.empty() || macLeaves.empty() ||
        scaleAccums.empty() || minBiasAccums.empty())
      return rewriter.notifyMatchFailure(
          tile, "q4_K tile region must carry the q4_K_dequant_core + "
                "q4_K_scale_min_unpack_core + vmadot_mac_leaf + "
                "q4_K_scale_weighted_accum + q4_K_min_bias_accum bricks (the "
                "hollow bare-MAC shape is rejected)");
    // The register-resident BATCHED K-loop MAC leaf ([GAP-IME-LEAF-PIPELINE]);
    // for q4_K it batches the 4 fragments of each sub-block into one sumi_b.
    llvm::StringRef macKloopHelperName = kVmadotMacKloopHelperName;

    auto variant =
        tile->getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
    auto sourceKernel = tile->getAttrOfType<mlir::StringAttr>("source_kernel");
    if (!variant || !sourceKernel)
      return rewriter.notifyMatchFailure(
          tile, "IME q4_K tile requires selected_variant and source_kernel "
                "attributes");
    std::string functionName =
        ("weft_emitc_" + sourceKernel.getValue() + "_" + variant.getValue())
            .str();

    int64_t matM = tile.getMatM();
    int64_t matN = tile.getMatN();
    int64_t matK = tile.getMatK();

    llvm::StringRef sourceOpName = tile.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef sourceRole = tile.getWEFTEmitCLowerableSourceRole();

    auto module = tile->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(tile, "IME q4_K tile has no module");

    auto i8PtrType = emitc::PointerType::get(
        context, emitc::OpaqueType::get(context, "const int8_t"));
    auto u8PtrType = emitc::PointerType::get(
        context, emitc::OpaqueType::get(context, "const uint8_t"));
    auto i32PtrType = emitc::PointerType::get(
        context, emitc::OpaqueType::get(context, "int32_t"));
    auto f32PtrType = emitc::PointerType::get(
        context, emitc::OpaqueType::get(context, "float"));
    auto longType = emitc::OpaqueType::get(context, "long");

    // G8 wide-vmadot key: q4_K is EPILOGUE-BOUND (the two-level 6-bit sc/m +
    // S_scale/S_min + fp16 fold dominates the tile), so the bottleneck-shape leg
    // DECLINES the wide MAC tiling (measured NULL: M7 q4_K@ime 0.909x). The decline
    // is MATERIALIZED as a provenance comment (honest, not a silent hand-omission);
    // no wide leaf is emitted and the narrow leaf stays deployed.
    IMEWideDeployDecision wide = decideWideVmadotDeployment(
        readDeployedVlenBits(tile), tile.getMacM(), tile.getMacN(),
        tile.getMacK(), tile.getElemInBits(), tile.getAccumBits(),
        /*epilogueBound=*/true);

    // Module-scope prologue: include + the validated vmadot MAC leaf + the q4_K
    // fp16 epilogue helpers + the raw-nibble decode + the 6-bit scale/min unpack +
    // the tiled q4_K two-level-fold kernel (declared-before-use ordering).
    {
      mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<emitc::IncludeOp>(loc, "stdint.h",
                                        /*is_standard_include=*/true);
      rewriter.create<emitc::VerbatimOp>(loc, vmadotMacKloopHelperBody());
      rewriter.create<emitc::VerbatimOp>(
          loc, std::string("// weft_ime.pat1_tiling=decline njw=1 ") + wide.reason);
      rewriter.create<emitc::VerbatimOp>(loc, q4KFp16HelperBody());
      rewriter.create<emitc::VerbatimOp>(loc, q4KDequantHelperBody());
      rewriter.create<emitc::VerbatimOp>(loc, q4KScaleMinHelperBody());
      rewriter.create<emitc::VerbatimOp>(loc,
                                         q4KMatmulHelperBody(macKloopHelperName));
    }

    mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());

    // Exported wrapper: extern "C" void <name>(const int8_t* Apack,
    // const uint8_t* Bq4k, int32_t* Sscale, int32_t* Smin, float* Cf).
    llvm::SmallVector<mlir::Type, 5> paramTypes{i8PtrType, u8PtrType, i32PtrType,
                                                i32PtrType, f32PtrType};
    mlir::FunctionType functionType =
        rewriter.getFunctionType(paramTypes, /*results=*/{});
    llvm::SmallVector<mlir::NamedAttribute, 1> funcAttrs;
    funcAttrs.push_back(rewriter.getNamedAttr(
        "specifiers", rewriter.getStrArrayAttr({"extern", "\"C\""})));
    auto func = rewriter.create<emitc::FuncOp>(loc, functionName, functionType,
                                               funcAttrs);
    mlir::Block *entry = func.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);

    rewriter.create<emitc::VerbatimOp>(
        loc, routeSourceComment(sourceOpName, sourceRole));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(sourceOpName, sourceRole, kQ4KMatmulHelperName));

    // Structured dataflow into the kernel: call_opaque on the pointer block args
    // plus the M/N/K problem-dim constants (compile-time variant facts).
    llvm::SmallVector<mlir::Value, 8> callOperands;
    for (mlir::BlockArgument arg : entry->getArguments())
      callOperands.push_back(arg);
    for (int64_t dim : {matM, matN, matK}) {
      auto dimAttr = emitc::OpaqueAttr::get(context, std::to_string(dim));
      auto constOp = rewriter.create<emitc::ConstantOp>(loc, longType, dimAttr);
      callOperands.push_back(constOp.getResult());
    }
    rewriter.create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{},
                                         kQ4KMatmulHelperName, callOperands);

    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());

    rewriter.eraseOp(tile);
    return mlir::success();
  }
};

class IMEBackendEmissionDriver final
    : public weftemitc::TypedBackendEmissionDriver {
public:
  llvm::StringRef getBackendName() const override { return "ime"; }

  void populateTypeConversions(
      mlir::TypeConverter & /*typeConverter*/) const override {}

  void
  configureConversionTarget(mlir::ConversionTarget &target) const override {
    target.addIllegalOp<weft::ime::MMAOp, weft::ime::MMAUOp,
                        weft::ime::MMASUOp, weft::ime::MMAUSOp,
                        weft::ime::MMASlideOp, weft::ime::MatMulOp,
                        weft::ime::Q40MatMulTileOp,
                        weft::ime::Q80MatMulTileOp,
                        weft::ime::Q4KMatMulTileOp>();
    target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
  }

  void
  populateLoweringPatterns(mlir::TypeConverter &typeConverter,
                           mlir::RewritePatternSet &patterns) const override {
    patterns.add<IMEMACToEmitCFunc<weft::ime::MMAOp>,
                 IMEMACToEmitCFunc<weft::ime::MMAUOp>,
                 IMEMACToEmitCFunc<weft::ime::MMASUOp>,
                 IMEMACToEmitCFunc<weft::ime::MMAUSOp>, IMEMACSlideToEmitCFunc,
                 IMEMatMulToEmitCFunc, IMEQ40MatMulTileToEmitCFunc,
                 IMEQ80MatMulTileToEmitCFunc, IMEQ4KMatMulTileToEmitCFunc>(
        typeConverter, patterns.getContext());
  }

  llvm::LogicalResult
  postConversionCleanup(mlir::ModuleOp module) const override;

  bool moduleHasBackendBody(mlir::ModuleOp module) const override {
    bool hasIME = false;
    module.walk([&](mlir::Operation *op) {
      if (op->getName().getDialectNamespace() ==
          weft::ime::WEFTIMEDialect::getDialectNamespace()) {
        hasIME = true;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    return hasIME;
  }
};

llvm::LogicalResult
IMEBackendEmissionDriver::postConversionCleanup(mlir::ModuleOp module) const {
  bool producedFunc = false;
  module.walk([&](emitc::FuncOp) { producedFunc = true; });
  if (!producedFunc)
    return llvm::success();

  llvm::SmallVector<mlir::Operation *, 2> drainedTopLevel;
  for (mlir::Operation &op : module.getBody()->getOperations()) {
    llvm::StringRef dialect = op.getName().getDialectNamespace();
    if (dialect != emitc::EmitCDialect::getDialectNamespace())
      drainedTopLevel.push_back(&op);
  }
  for (mlir::Operation *op : drainedTopLevel)
    op->erase();
  return llvm::success();
}

} // namespace

void registerIMEBackendEmitter(weftemitc::BackendEmissionRegistry &registry) {
  static const IMEBackendEmissionDriver driver;
  registry.registerBackend(driver);
}

} // namespace ime
} // namespace plugin
} // namespace weft
