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

#ifndef WEFT_PLUGIN_RVV_RVVCONTRACTIONROUTEIDENTITY_H
#define WEFT_PLUGIN_RVV_RVVCONTRACTIONROUTEIDENTITY_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace weft::plugin::rvv {

/// How a contraction source materializes in the constructed body (DESIGN §1,
/// critic fix #3). Every entry in a route's ordered `sources` carries a
/// SourceKind (a product-factor load or a constant-table load).
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
  /// "wprod-lhs"). Empty for non-multiplicand decode sources (e.g. the C4
  /// constant table carries no product role).
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
  /// "src-i8mf4" / "src-u8mf4"). Empty for non-multiplicand decode sources.
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
/// derive arity + decoration from this SINGLE source (1c/1d).
///
/// This descriptor owns ONLY: the (headOpName, isSigned) key, the per-source
/// multiplicand/arity decoration (`sources`), and the empty-for-now 1d
/// conditional-insert mechanism (`conditionalInserts`). Nothing else lives here.
///
/// Two things deliberately do NOT belong on this descriptor because they are
/// form-owned, not head-owned (a head-keyed descriptor cannot represent them):
///   - The ABI tail (acc/out/n and the dequant scale/clamp) -- keyed by
///     RVVSelectedBodyOperationKind via getContractionRuntimeABIOrder. >=4 body-op
///     FORMS share the head "weft_rvv.widening_product" signed, each with a
///     different tail (see DESIGN-noperand-route-identity.md REVISION v2).
///   - product_relation -- candidate/form-owned, sourced from
///     selectedResourceCandidate->primitiveWideningProductRelation at realization
///     (RVVContractionSelectedBodyRealizationOwner.cpp:2371-2376), so it varies
///     per selected candidate and is not provably head-owned; never on this
///     descriptor.
struct ContractionRouteIdentity {
  /// The product-head typed op mnemonic (e.g. "weft_rvv.widening_product",
  /// "weft_rvv.packed_i4_nibble_unpack_product").
  llvm::StringRef headOpName;
  bool isSigned;

  /// Ordered sources: product factors first, then any aux (constant table).
  /// The N=2 routes registered in 1b have exactly two PerIterInputBufferLoad
  /// product factors.
  llvm::SmallVector<ContractionSourceSpec, 4> sources;

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

/// Return the multiplicand-roles summary string for (mnemonic, isSigned),
/// derived from the route's ordered `sources` (axis-A / headOperandIndex order)
/// via the join template  {slotName}={abiRole}:{roleName}:{srcStripLabel}
/// joined by ';'. Only entries with isMultiplicandFactor==true participate.
///
/// The returned StringRef is backed by a process-lifetime cache (one joined
/// string per registry entry), so it is stable and safe to store in a StringRef
/// field. Returns an empty StringRef when no route is registered.
///
/// 1c: this is the SINGLE source the R1 route-family producers/validators derive
/// the multiplicand-roles fact from, replacing the per-site hardcoded
/// kRVVLowPrecision{Signed,Unsigned}WideningProductMultiplicandRoles constants.
llvm::StringRef getContractionMultiplicandRoleSummary(llvm::StringRef mnemonic,
                                                      bool isSigned);

/// The number of ordered product-factor sources of a route -- the arity of its
/// productSources[] vector (count of isMultiplicandFactor PerIterInputBufferLoad
/// sources; the ConstantTableLoad aux of C4 does NOT count). For every N=2 route
/// registered as of 2b this returns 2.
///
/// 2b: the load-binding derives the productSources[] SIZE from this, replacing
/// the hardcoded 2.
unsigned getContractionProductFactorCount(const ContractionRouteIdentity &route);

/// Resolve the ORDERED product-factor slot index k of a bound input-buffer
/// parameter identified by (abiRole, abiCName), among the route's product-factor
/// PerIterInputBufferLoad sources. This is the index at which the bound load is
/// placed into productSources[] (productSources[k]). Returns std::nullopt when no
/// product-factor source matches BOTH abiRole and abiCName (a route-provider
/// error: the bound ABI role/c-name is not a product factor of this route).
///
/// For every N=2 route this maps ("lhs-input-buffer","lhs") -> 0 and
/// ("rhs-input-buffer","rhs") -> 1, exactly reproducing the 2a hardcoded 0/1
/// aliasing; it cannot return nullopt for those valid bindings. (For all N=2
/// routes the product-factor ordinal equals the source's index in the full
/// `sources` vector, so k is ALSO the RVVProductSource.sourceIndex; the two
/// diverge only when the C4 constant table lands, at which point the caller that
/// needs the distinction will introduce it -- P1f.)
///
/// 2b: the load-binding derives the productSources[] SLOT from this, replacing
/// the hardcoded 0 (LHS branch) / 1 (RHS branch).
std::optional<unsigned>
getContractionProductFactorSlotIndex(const ContractionRouteIdentity &route,
                                     llvm::StringRef abiRole,
                                     llvm::StringRef abiCName);

/// The role-step callee labels of ONE product factor: its runtime_abi callee
/// (the factor's abiCName, e.g. "qhi") and its per-iteration load callee (the
/// abiCName + "_load", e.g. "qhi_load"). Both are process-lifetime-stable
/// StringRefs safe to store in an ExecutableRoleStep::callee field.
struct ContractionProductFactorRoleLabels {
  llvm::StringRef runtimeABICallee;
  llvm::StringRef loadCallee;
  /// The operand-binding product role token spliced into the route operand
  /// binding plan for this factor: "wprod-" + abiCName (e.g. "wprod-qhi"). Backed
  /// by the same process-lifetime cache.
  llvm::StringRef productRoleToken;
};

/// The ordered role-step callee labels for the route's product factors BEYOND
/// the abstract lhs/rhs pair -- product-factor ordinal k >= 2 (the "extra" N=3+
/// sources, e.g. the offset-binary/codebook qhi). The role-step builder + the
/// canonical-order derivation splice one runtime_abi + one load step per entry
/// into the otherwise-abstract 12-step widening-product-reduce spec, so a route
/// of arity N contributes (N-2) extra middle sources with NO consumer edit --
/// only its registry `sources[]` decoration. Empty for every N=2 route (so those
/// keep the byte-identical 12-step spec). The returned StringRefs are backed by a
/// process-lifetime cache.
llvm::SmallVector<ContractionProductFactorRoleLabels, 2>
getContractionExtraProductFactorRoleLabels(const ContractionRouteIdentity &route);

/// Derive the runtime-ABI parameter ORDER string for a product-reduction route
/// GENERICALLY from its descriptor: the ordered product-factor c-names (the
/// headOperandIndex / axis-A order, e.g. "lhs,rhs" or "w,qlo,qhi") followed by
/// the form-owned tail (acc[,scale][,bounds],out,n). The tail is recovered from
/// `abstractOpKindABIOrder` -- the op-kind-keyed abstract order string, which
/// lists the abstract binary form's TWO multiplicand slots (lhs,rhs) then the
/// form tail -- by stripping that abstract multiplicand prefix and re-attaching
/// the descriptor's concrete c-name prefix. This reproduces the N=2
/// "lhs,rhs,acc,out,n" (IDEMPOTENT -- descriptor c-names ARE lhs,rhs) and the N=3
/// "w,qlo,qhi,acc,out,n" from ONE path, retiring the per-route offset-binary /
/// codebook literal overrides. The returned StringRef is process-lifetime cached.
llvm::StringRef getContractionProductReductionRuntimeABIOrder(
    const ContractionRouteIdentity &route,
    llvm::StringRef abstractOpKindABIOrder);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVCONTRACTIONROUTEIDENTITY_H
