//===- RVVContractionRouteIdentity.h - N-operand contraction route ID -----===//
//
// P1a/P1b FOUNDATION (subtask 07-01-p1b). One descriptor per contraction
// product-head route, keyed by (product-head op mnemonic, signedness). The
// eventual goal (1c/1d) is that BOTH the route-family validator (R1) and the
// construction-protocol conformance validator (R2) derive their per-source
// arity + decoration from THIS single source, guaranteeing lockstep across the
// mirror validators (see DESIGN-noperand-route-identity.md §1/§3).
//
// This header is a PURELY ADDITIVE foundation: as of 1b NOTHING reads the
// registry, so emitted bytes are unchanged (trivially byte-exact). It is
// deliberately dependency-light (llvm ADT only) because 1c will invert the
// dependency -- the EmitC contraction validators will come to depend on THIS,
// so this must not depend back on the EmitC internal seam.
//
// It is parallel to RVVSelectedBodyConstructionRoute (RVVConstructionProtocol.h)
// -- NOT merged into that flat 6-StringRef struct, and NOT a named-slot slice
// (that named-slot layering is the antipattern the earlier arc discarded).
//
//===----------------------------------------------------------------------===//

#ifndef TIANCHENRV_PLUGIN_RVV_RVVCONTRACTIONROUTEIDENTITY_H
#define TIANCHENRV_PLUGIN_RVV_RVVCONTRACTIONROUTEIDENTITY_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace tianchenrv::plugin::rvv {

/// How a contraction source materializes in the constructed body (DESIGN §1,
/// critic fix #3). Only the ordered `sources` of a route carry a meaningful
/// SourceKind; the fixed acc/out/n tail specs do not classify as loads.
enum class SourceKind {
  /// A per-iteration unit/strided LOAD backed by a runtime-ABI input-buffer
  /// parameter (lhs / rhs / weight / qlo / qhi). Every one of these contributes
  /// exactly one input-buffer ABI param -- #input-buffer-ABI-params ==
  /// count(PerIterInputBufferLoad).
  PerIterInputBufferLoad,
  /// A vle8 LOAD of a compile-time constant table (the C4 codebook broadcast).
  /// It IS a load but has NO ABI param and is NOT a multiplicand factor. Defined
  /// here for completeness; first USED by the C4 route in 1f, not by 1b.
  ConstantTableLoad,
};

/// One ordered contraction source (a product factor or a decode/table source).
///
/// TWO ORTHOGONAL order axes (the byte gate bites on both -- DESIGN §1):
///   - headOperandIndex (axis-A): drives the multiplicand-roles join order, the
///     routeOperandBindingSummary, and the ABI c-name order.
///   - bodyStepPosition (axis-B): drives the R2 role-step list and the
///     getRVVCanonicalRoleOrder integer. For all N=2 routes the two axes agree;
///     they diverge only at C4 (the constant table, added in 1f).
struct ContractionSourceSpec {
  SourceKind kind;
  /// true  = a genuine multiplicand factor (participates in the vwmul/vwmacc
  ///         chain + LMUL-width inference: lhs / rhs / weight / qlo / qhi);
  /// false = a decode source (constant table) or a fixed-tail spec.
  /// (critic fix #4: product-factor count is separate from total slot count.)
  bool isMultiplicandFactor;

  /// The generic slot name -- the token BEFORE '=' in the multiplicand-roles
  /// join (e.g. "lhs" / "rhs").
  llvm::StringRef slotName;
  /// The product role token -- the middle token of the roles join (e.g.
  /// "wprod-lhs"). Empty for fixed-tail specs (acc/out/n carry no product role).
  llvm::StringRef roleName;
  /// The runtime-ABI role -- the RuntimeABIValueOp "role" attr (e.g.
  /// "lhs-input-buffer", "accumulator-input-buffer", "output-buffer").
  llvm::StringRef abiRole;
  /// The runtime-ABI C parameter name -- RuntimeABIValueOp "c_name" (e.g.
  /// "lhs" / "rhs" / "acc" / "out" / "n").
  llvm::StringRef abiCName;
  /// The runtime-ABI C parameter type -- RuntimeABIValueOp "c_type" (e.g.
  /// "const int8_t *", "const int32_t *", "int32_t *", "float *", "size_t").
  llvm::StringRef abiCType;
  /// The source strip label -- the trailing token of the roles join (e.g.
  /// "src-i8mf4" / "src-u8mf4"). Empty for fixed-tail specs.
  llvm::StringRef srcStripLabel;

  /// axis-A: roles-join / routeOperandBindingSummary / ABI c-name order.
  unsigned headOperandIndex;
  /// axis-B: role-step list / getRVVCanonicalRoleOrder order.
  unsigned bodyStepPosition;
};

/// A predicate-keyed conditional insert into the role-step list (DESIGN §1).
/// getRVVCanonicalRoleOrder already carries structural shifts
/// (dequantDeferredWideExtra +1, dequantTwoScopeExtra +2, dequant-scale +1);
/// the 1d generator = BASE step list + these predicate-keyed inserts spliced in.
///
/// MINIMAL for 1b: this only DEFINES the type so a ContractionRouteIdentity can
/// carry it. The generator that consumes it is 1d; the N=2 base routes populated
/// here carry no conditional inserts (they are the un-shifted base).
struct ConditionalStep {
  /// The structural predicate that gates the insert (e.g. the presence of a
  /// wideningAccumulateOp / gearboxCrossRegionHandoffOp / dequantize op).
  llvm::StringRef predicateKey;
  /// The role-step name spliced in when the predicate holds.
  llvm::StringRef stepRoleName;
  /// The canonical-order shift the insert contributes (e.g. +1 / +2).
  int orderDelta;
};

/// One descriptor per (product-head op mnemonic, signedness). R1 and R2 both
/// derive arity + decoration from this SINGLE source (1c/1d). Fields whose
/// provenance is not yet fully pinned for a given route are documented at their
/// registration site so 1c can promote them from inferred -> derived.
struct ContractionRouteIdentity {
  /// The product-head typed op mnemonic (e.g. "tcrv_rvv.widening_product",
  /// "tcrv_rvv.packed_i4_nibble_unpack_product").
  llvm::StringRef headOpName;
  bool isSigned;

  /// Ordered sources: product factors first, then any aux (constant table).
  /// The N=2 routes registered in 1b have exactly two PerIterInputBufferLoad
  /// product factors.
  llvm::SmallVector<ContractionSourceSpec, 4> sources;

  /// Fixed ABI tail. accSpec is optional: the widening_product head keys
  /// multiple ABI tails ("lhs,rhs,out,n" has NO acc; "lhs,rhs,acc,out,n" and the
  /// dequant form do). See DESIGN §6 open Q1 (per-VL-config vs per-route keying)
  /// -- 1c resolves whether the tail is descriptor-owned or route-form-owned.
  std::optional<ContractionSourceSpec> accSpec;
  ContractionSourceSpec outSpec;
  ContractionSourceSpec nSpec;

  /// The standalone-reduce head that follows the product (e.g.
  /// "tcrv_rvv.standalone_reduce").
  llvm::StringRef reduceOpName;

  /// The narrow product-relation constant (the op's "product_relation" attr,
  /// e.g. "signed-i8mf4xi8mf4-to-i16mf2-widening-product.v1").
  llvm::StringRef productRelation;
  /// The leaf-profile constant, if the route has one. No contraction
  /// leaf-profile constant exists in-tree today (only the elementwise family
  /// carries one), so this is an empty placeholder for the N=2 routes; 1c/1d
  /// bind it if/when a contraction leaf-profile constant is introduced.
  llvm::StringRef leafProfile;

  /// Predicate-keyed canonical-order inserts. Empty for the N=2 base routes;
  /// the deferred-wide / two-scope-handoff / dequant-scale shifts are attached
  /// by 1c/1d when the dequant-family routes are migrated.
  llvm::SmallVector<ConditionalStep, 4> conditionalInserts;
};

/// Look up the descriptor for (mnemonic, isSigned). Returns nullptr when no
/// route is registered.
///
/// 1b NOTE: this registry is NOT read by any validator/consumer yet. It is a
/// purely additive foundation (zero binary diff). 1c/1d migrate the R1/R2
/// validators to derive their strings from this registry.
const ContractionRouteIdentity *
getContractionRouteIdentity(llvm::StringRef mnemonic, bool isSigned);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVCONTRACTIONROUTEIDENTITY_H
