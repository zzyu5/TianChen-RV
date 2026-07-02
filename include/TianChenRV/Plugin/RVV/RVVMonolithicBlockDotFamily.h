//===- RVVMonolithicBlockDotFamily.h ----------------------------*- C++ -*-===//
//
// P2-b chunk3: the SHARED monolithic ggml block-dot family mechanism.
//
// Chunk 1 (RVVExtensionPlugin.cpp) wired the emission-plans stage and chunk 2
// (RVVTargetSupportBundle.cpp) wired the target-artifact export -- both keyed to
// the ONE q4_K op (tcrv_rvv.q4_k_q8_k_block_dot) with q4_K-specific constants
// duplicated in each file. The wall those chunks climbed was always block-dot
// GENERIC, not q4_K-specific: EVERY monolithic ggml block-dot body is ONE
// plugin-owned typed op that lowers DIRECTLY through the RVV->EmitC
// DialectConversion with no decomposed route slice for the slice-based
// describeRVVSelectedBodyEmitCRoute to walk, so it hits the same fail-closed
// rejection.
//
// This header is the single source of truth both chunks now key off. It carries:
//   (1) The STRUCTURAL recognition -- "is a with_vl body exactly one monolithic
//       block-dot op" -- keyed off a small per-op family TABLE, not any one op
//       type. This is the "trait" at the C++ level (the task allows "a common
//       base -- a small per-op table"): unwired block-dot ops are deliberately
//       NOT in the table, so they stay fail-closed at the slice describe until
//       they get front doors (their honest state).
//   (2) The SUPER-BLOCK vs FLAT route-family split. The K-quants (q4_K) are
//       genuine super-blocks; the flat 32-element block-dots (q4_0/iq4_nl) are
//       NOT. The route id, runtime-ABI name, archetype, and the two op-derived
//       metadata keys are parameterized by route family; everything else
//       (artifact kind, EmitC-lowerable op interface, construction protocol, the
//       fixed metadata keys) is family-invariant.
//   (3) The per-op DATA (op identity, route family, kind, ordered runtime-ABI
//       expectation) as a table. The STRUCTURAL wiring is generic (family-keyed);
//       the per-op data parameterizes it -- the same generic-vs-data split as the
//       P1 N-operand descriptor refactor. q4_K/iq4_nl carry the 4-role ggml
//       vec_dot ABI (n, s, vx, vy); q4_0 carries the full 8-role strided ggml
//       vec_dot ABI (n, s, bs, vx, bx, vy, by, nrc) -- the block-dot op consumes
//       only vx/vy/s/n but the exported C signature mirrors ggml's prototype.
//
//===----------------------------------------------------------------------===//

#ifndef TIANCHENRV_PLUGIN_RVV_RVVMONOLITHICBLOCKDOTFAMILY_H
#define TIANCHENRV_PLUGIN_RVV_RVVMONOLITHICBLOCKDOTFAMILY_H

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace tianchenrv::plugin::rvv {

// The two monolithic ggml block-dot route families. K-quants (q4_K) are genuine
// super-blocks (QK_K == 256, 8 sub-blocks, per-sub-block scale/min bit-dance);
// flat block-dots (q4_0, iq4_nl) are single 32-element AoS blocks.
enum class MonolithicBlockDotRouteFamily { SuperBlock, Flat };

// Family-INVARIANT constants -- identical for the super-block AND flat monolithic
// block-dot routes (they describe the shared EmitC-lowerable typed-body mechanism
// and the RISC-V object artifact, neither of which differs by block structure).
namespace monolithic_block_dot {
inline constexpr llvm::StringLiteral kArtifactKind(
    "riscv-elf-relocatable-object");
inline constexpr llvm::StringLiteral kSourceOpInterfaceName(
    "TCRVEmitCLowerableOpInterface");
inline constexpr llvm::StringLiteral kConstructionProtocol(
    "extension-family-construction-protocol.v1");
// The fixed-value artifact-metadata keys the monolithic emission plan attaches
// (family-invariant; only the *values* under kKindKey/kScaleModelKey are
// op-derived).
inline constexpr llvm::StringLiteral kRouteMetadataKey(
    "rvv_emitc_lowerable_route");
inline constexpr llvm::StringLiteral kSourceOpInterfaceKey(
    "rvv_source_op_interface");
inline constexpr llvm::StringLiteral kArchetypeKey("rvv_extension_archetype");
inline constexpr llvm::StringLiteral kTargetArtifactKindKey(
    "rvv_target_artifact_kind");
inline constexpr llvm::StringLiteral kConstructionProtocolKey(
    "rvv_construction_protocol");
} // namespace monolithic_block_dot

// Family-PARAMETERIZED constants. The super-block set is byte-identical to the
// values chunks 1+2 hardcoded for q4_K (the q4_K full-pipeline e2e pins the route
// id + metadata keys), so keying q4_K off the super-block family stays byte-exact.
struct MonolithicBlockDotFamilyConstants {
  llvm::StringRef routeID;
  llvm::StringRef headerRouteID;
  llvm::StringRef runtimeABIName;
  llvm::StringRef archetype;
  llvm::StringRef kindMetadataKey;
  llvm::StringRef scaleModelMetadataKey;
};

inline const MonolithicBlockDotFamilyConstants &
getMonolithicBlockDotFamilyConstants(MonolithicBlockDotRouteFamily family) {
  static const MonolithicBlockDotFamilyConstants kSuperBlock{
      "rvv-ggml-super-block-block-dot-monolithic-emitc-route-family",
      "rvv-ggml-super-block-block-dot-monolithic-emitc-route-family.header",
      "rvv-ggml-super-block-block-dot-callable-c-abi.v1",
      "rvv-ggml-super-block-monolithic-typed-body",
      "rvv_ggml_super_block_block_dot_kind",
      "rvv_ggml_super_block_scale_model"};
  static const MonolithicBlockDotFamilyConstants kFlat{
      "rvv-ggml-flat-block-dot-monolithic-emitc-route-family",
      "rvv-ggml-flat-block-dot-monolithic-emitc-route-family.header",
      "rvv-ggml-flat-block-dot-callable-c-abi.v1",
      "rvv-ggml-flat-monolithic-typed-body",
      "rvv_ggml_flat_block_dot_kind",
      "rvv_ggml_flat_scale_model"};
  return family == MonolithicBlockDotRouteFamily::SuperBlock ? kSuperBlock
                                                             : kFlat;
}

// The long human-readable plan description (NOT rendered in the coherence
// diagnostic the e2e pins; kept per-family for honesty). The super-block text is
// byte-identical to chunk 1's.
inline llvm::StringRef
getMonolithicBlockDotPlanDescription(MonolithicBlockDotRouteFamily family) {
  static const llvm::StringRef kSuperBlock =
      "RVV selected monolithic ggml super-block block-dot typed body "
      "materializes a verified EmitC module through the common RVV->EmitC "
      "DialectConversion (the super-block loop, the 6-bit scale/min bit-dance, "
      "the aux32 accumulation, and the deferred fp32 fold/min are first-class "
      "op structure), then uses the MLIR EmitC C/C++ emitter before RISC-V "
      "object packaging";
  static const llvm::StringRef kFlat =
      "RVV selected monolithic ggml flat block-dot typed body materializes a "
      "verified EmitC module through the common RVV->EmitC DialectConversion "
      "(the AoS block loop, the per-block fp16 scale model, the integer decode/"
      "product core, and the fp32 fold are first-class op structure), then uses "
      "the MLIR EmitC C/C++ emitter before RISC-V object packaging";
  return family == MonolithicBlockDotRouteFamily::SuperBlock ? kSuperBlock
                                                             : kFlat;
}

// One expected ordered runtime-ABI parameter (c parameter name + role) of a
// monolithic block-dot's exported ggml vec_dot C signature.
struct MonolithicBlockDotABIRole {
  llvm::StringRef cName;
  support::RuntimeABIParameterRole role;
};

// The 4-role ggml vec_dot ABI (n, s, vx, vy) -- q4_K and iq4_nl.
inline llvm::ArrayRef<MonolithicBlockDotABIRole> monolithicBlockDotABI4() {
  static const MonolithicBlockDotABIRole kRoles[] = {
      {"n", support::RuntimeABIParameterRole::RuntimeElementCount},
      {"s", support::RuntimeABIParameterRole::OutputBuffer},
      {"vx", support::RuntimeABIParameterRole::LHSInputBuffer},
      {"vy", support::RuntimeABIParameterRole::RHSInputBuffer}};
  return kRoles;
}

// The full 8-role strided ggml vec_dot ABI (n, s, bs, vx, bx, vy, by, nrc) --
// q4_0. The block-dot op consumes only vx/vy/s/n; the stride/scalar params
// (bs/bx/by/nrc) are present so the exported C signature matches ggml's vec_dot
// prototype (they are dropped by the block-dot lowering).
inline llvm::ArrayRef<MonolithicBlockDotABIRole>
monolithicBlockDotABI8Strided() {
  static const MonolithicBlockDotABIRole kRoles[] = {
      {"n", support::RuntimeABIParameterRole::RuntimeElementCount},
      {"s", support::RuntimeABIParameterRole::OutputBuffer},
      {"bs", support::RuntimeABIParameterRole::OutputStride},
      {"vx", support::RuntimeABIParameterRole::LHSInputBuffer},
      {"bx", support::RuntimeABIParameterRole::LHSInputStride},
      {"vy", support::RuntimeABIParameterRole::RHSInputBuffer},
      {"by", support::RuntimeABIParameterRole::RHSInputStride},
      {"nrc", support::RuntimeABIParameterRole::RHSScalarValue}};
  return kRoles;
}

// The per-op family table (the "small per-op table"): every monolithic block-dot
// op that is production-reachable (front door + brick witness + byte-exact CORE
// emit). Its route family selects the route id; its kind keys the target-side ABI
// lookup; its abiRoles pins the exported C signature. Adding a new op = one row
// here + its front door (if missing).
struct MonolithicBlockDotOpEntry {
  llvm::StringRef opName; // the tcrv_rvv.* op mnemonic (full name)
  MonolithicBlockDotRouteFamily routeFamily;
  llvm::StringRef kind; // the op's bounded `kind` attribute value
  llvm::ArrayRef<MonolithicBlockDotABIRole> (*abiRoles)();
};

inline llvm::ArrayRef<MonolithicBlockDotOpEntry> monolithicBlockDotOpTable() {
  static const MonolithicBlockDotOpEntry kTable[] = {
      {tcrv::rvv::GgmlBlockDotQ4KQ8KOp::getOperationName(),
       MonolithicBlockDotRouteFamily::SuperBlock, "ggml_q4_k_q8_k_block_dot",
       &monolithicBlockDotABI4},
      // iq4_xs is the SUPER-BLOCK-CODEBOOK rung: it takes the SAME super-block
      // route family as q4_K (the codebook is an OP attr consumed by the emitter,
      // NOT a route-family concern -- the emission plan + target-export validator
      // key only off op name -> family + kind/scale_model + ABI roles). Its 4-role
      // ggml vec_dot ABI (n, s, vx, vy) matches q4_K's.
      {tcrv::rvv::GgmlBlockDotIQ4XSQ8KOp::getOperationName(),
       MonolithicBlockDotRouteFamily::SuperBlock, "ggml_iq4_xs_q8_k_block_dot",
       &monolithicBlockDotABI4},
      // The four common super-block-PLAIN K-quants (P2-e): q2_K/q3_K/q5_K/q6_K.
      // All are genuine super-blocks (QK_K == 256) so they take the SAME super-block
      // route family as q4_K, and all carry the SAME 4-role ggml vec_dot ABI
      // (n, s, vx, vy) as q4_K -- the block-format delta (2/3/5/6-bit weights, the
      // qh/hmask high-bit planes, the scale/min hierarchy, the scalar-vs-deferred
      // fold) is op structure the emitter consumes, NOT a route-family or ABI concern.
      {tcrv::rvv::GgmlBlockDotQ2KQ8KOp::getOperationName(),
       MonolithicBlockDotRouteFamily::SuperBlock, "ggml_q2_k_q8_k_block_dot",
       &monolithicBlockDotABI4},
      {tcrv::rvv::GgmlBlockDotQ3KQ8KOp::getOperationName(),
       MonolithicBlockDotRouteFamily::SuperBlock, "ggml_q3_k_q8_k_block_dot",
       &monolithicBlockDotABI4},
      {tcrv::rvv::GgmlBlockDotQ5KQ8KOp::getOperationName(),
       MonolithicBlockDotRouteFamily::SuperBlock, "ggml_q5_k_q8_k_block_dot",
       &monolithicBlockDotABI4},
      {tcrv::rvv::GgmlBlockDotQ6KQ8KOp::getOperationName(),
       MonolithicBlockDotRouteFamily::SuperBlock, "ggml_q6_k_q8_k_block_dot",
       &monolithicBlockDotABI4},
      {tcrv::rvv::GgmlBlockDotQ40Q80Op::getOperationName(),
       MonolithicBlockDotRouteFamily::Flat, "ggml_q4_0_q8_0_block_dot",
       &monolithicBlockDotABI8Strided},
      {tcrv::rvv::GgmlBlockDotQ80Q80Op::getOperationName(),
       MonolithicBlockDotRouteFamily::Flat, "ggml_q8_0_q8_0_block_dot",
       &monolithicBlockDotABI8Strided},
      {tcrv::rvv::GgmlBlockDotIQ4NLQ80Op::getOperationName(),
       MonolithicBlockDotRouteFamily::Flat, "ggml_iq4_nl_q8_0_block_dot",
       &monolithicBlockDotABI4},
      // The three common legacy FLAT block-dot formats (P2-d): q4_1 (Family-B
      // scale+MIN), q5_0 (Family-A 5-bit weight), q5_1 (Family-B 5-bit scale+MIN).
      // All three carry the SAME 4-role ggml vec_dot ABI (n, s, vx, vy) as iq4_nl --
      // the block-format delta (min/qh planes) is op structure the emitter consumes,
      // NOT a route-family or ABI concern.
      {tcrv::rvv::GgmlBlockDotQ41Q81Op::getOperationName(),
       MonolithicBlockDotRouteFamily::Flat, "ggml_q4_1_q8_1_block_dot",
       &monolithicBlockDotABI4},
      {tcrv::rvv::GgmlBlockDotQ50Q80Op::getOperationName(),
       MonolithicBlockDotRouteFamily::Flat, "ggml_q5_0_q8_0_block_dot",
       &monolithicBlockDotABI4},
      {tcrv::rvv::GgmlBlockDotQ51Q81Op::getOperationName(),
       MonolithicBlockDotRouteFamily::Flat, "ggml_q5_1_q8_1_block_dot",
       &monolithicBlockDotABI4}};
  return kTable;
}

// The op-identity lookup (plugin side: from the recognized body op).
inline const MonolithicBlockDotOpEntry *
findMonolithicBlockDotOpEntry(mlir::Operation *op) {
  if (!op)
    return nullptr;
  llvm::StringRef name = op->getName().getStringRef();
  for (const MonolithicBlockDotOpEntry &entry : monolithicBlockDotOpTable())
    if (entry.opName == name)
      return &entry;
  return nullptr;
}

// The kind lookup (target side: from the candidate's op-derived kind metadata,
// where the op itself is no longer in hand).
inline const MonolithicBlockDotOpEntry *
findMonolithicBlockDotOpEntryByKind(llvm::StringRef kind) {
  if (kind.empty())
    return nullptr;
  for (const MonolithicBlockDotOpEntry &entry : monolithicBlockDotOpTable())
    if (entry.kind == kind)
      return &entry;
  return nullptr;
}

// Recognize a selected with_vl scope whose ENTIRE compute body is exactly one
// supported monolithic ggml block-dot op (any family). Returns the op for that
// single-op body; nullptr otherwise -- so decomposed route bodies (and unwired
// block-dot ops) fall through to the slice-based route path unchanged.
inline mlir::Operation *
findSelectedMonolithicBlockDotBody(tcrv::rvv::WithVLOp withVL) {
  if (withVL.getBody().empty())
    return nullptr;
  mlir::Block &block = withVL.getBody().front();
  mlir::Operation *found = nullptr;
  for (mlir::Operation &op : block) {
    if (!findMonolithicBlockDotOpEntry(&op))
      return nullptr; // any non-family op => not the monolithic single-op shape
    if (found)
      return nullptr; // more than one op => not the monolithic single-op shape
    found = &op;
  }
  return found;
}

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVMONOLITHICBLOCKDOTFAMILY_H
