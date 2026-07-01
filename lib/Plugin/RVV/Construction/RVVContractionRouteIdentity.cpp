//===- RVVContractionRouteIdentity.cpp - contraction route ID registry ----===//
//
// P1b FOUNDATION. The static registry of ContractionRouteIdentity descriptors +
// the (mnemonic, isSigned) lookup. Only the existing N=2 routes are populated:
//   - tcrv_rvv.widening_product              (signed)
//   - tcrv_rvv.widening_product              (unsigned)
//   - tcrv_rvv.packed_i4_nibble_unpack_product (signed)
//
// ZERO BINARY DIFF: nothing reads this registry as of 1b. Data accuracy matters
// only so 1c can derive byte-identical strings from it; every field is
// cross-checked against the existing hardcoded decoration and its provenance is
// recorded inline (VERIFIED vs the front-door / roles constant, or INFERRED
// where no front-door anchor exists).
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVContractionRouteIdentity.h"

#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <string>
#include <vector>

namespace tianchenrv::plugin::rvv {

namespace {

// Build the immutable registry once. Kept as a function-local static so there is
// no global constructor (and, since nothing calls this in 1b, it is never even
// initialized -- reinforcing the zero-diff property).
const std::vector<ContractionRouteIdentity> &contractionRouteRegistry() {
  static const std::vector<ContractionRouteIdentity> registry = [] {
    std::vector<ContractionRouteIdentity> table;

    // ======================================================================
    // Route 1: tcrv_rvv.widening_product, SIGNED
    // ----------------------------------------------------------------------
    // Source decoration VERIFIED byte-for-byte vs
    // kRVVLowPrecisionSignedWideningProductMultiplicandRoles
    // (RVVEmitCContractionRouteFamilyInternal.h:167-170):
    //   "lhs=lhs-input-buffer:wprod-lhs:src-i8mf4;
    //    rhs=rhs-input-buffer:wprod-rhs:src-i8mf4"
    // c-name/c-type VERIFIED vs RVVReductionSourceFrontDoor.cpp:706-711 and the
    // dequant front door RVVDequantDotSourceFrontDoor.cpp:910-915
    // (role "lhs-input-buffer"/"rhs-input-buffer", c-name "lhs"/"rhs",
    //  c-type "const int8_t *").
    {
      ContractionRouteIdentity r;
      r.headOpName = "tcrv_rvv.widening_product";
      r.isSigned = true;
      r.sources.push_back(ContractionSourceSpec{
          SourceKind::PerIterInputBufferLoad, /*isMultiplicandFactor=*/true,
          /*slotName=*/"lhs", /*roleName=*/"wprod-lhs",
          /*abiRole=*/"lhs-input-buffer", /*abiCName=*/"lhs",
          /*abiCType=*/"const int8_t *", /*srcStripLabel=*/"src-i8mf4",
          /*headOperandIndex=*/0, /*bodyStepPosition=*/0});
      r.sources.push_back(ContractionSourceSpec{
          SourceKind::PerIterInputBufferLoad, /*isMultiplicandFactor=*/true,
          /*slotName=*/"rhs", /*roleName=*/"wprod-rhs",
          /*abiRole=*/"rhs-input-buffer", /*abiCName=*/"rhs",
          /*abiCType=*/"const int8_t *", /*srcStripLabel=*/"src-i8mf4",
          /*headOperandIndex=*/1, /*bodyStepPosition=*/1});
      table.push_back(std::move(r));
    }

    // ======================================================================
    // Route 2: tcrv_rvv.widening_product, UNSIGNED
    // ----------------------------------------------------------------------
    // Source roles VERIFIED byte-for-byte vs
    // kRVVLowPrecisionUnsignedWideningProductMultiplicandRoles
    // (RVVEmitCContractionRouteFamilyInternal.h:172-174):
    //   "lhs=lhs-input-buffer:wprod-lhs:src-u8mf4;
    //    rhs=rhs-input-buffer:wprod-rhs:src-u8mf4"
    // INFERRED (no front-door anchor found for the unsigned route): source
    // abiCType "const uint8_t *" as the natural analog of "const int8_t *"
    // matching src-u8mf4. 1c must confirm the unsigned route is front-door
    // reachable (or resolve it as metadata-only) before deriving.
    {
      ContractionRouteIdentity r;
      r.headOpName = "tcrv_rvv.widening_product";
      r.isSigned = false;
      r.sources.push_back(ContractionSourceSpec{
          SourceKind::PerIterInputBufferLoad, /*isMultiplicandFactor=*/true,
          /*slotName=*/"lhs", /*roleName=*/"wprod-lhs",
          /*abiRole=*/"lhs-input-buffer", /*abiCName=*/"lhs",
          /*abiCType=*/"const uint8_t *", /*srcStripLabel=*/"src-u8mf4",
          /*headOperandIndex=*/0, /*bodyStepPosition=*/0});
      r.sources.push_back(ContractionSourceSpec{
          SourceKind::PerIterInputBufferLoad, /*isMultiplicandFactor=*/true,
          /*slotName=*/"rhs", /*roleName=*/"wprod-rhs",
          /*abiRole=*/"rhs-input-buffer", /*abiCName=*/"rhs",
          /*abiCType=*/"const uint8_t *", /*srcStripLabel=*/"src-u8mf4",
          /*headOperandIndex=*/1, /*bodyStepPosition=*/1});
      table.push_back(std::move(r));
    }

    // ======================================================================
    // Route 3: tcrv_rvv.packed_i4_nibble_unpack_product, SIGNED
    // ----------------------------------------------------------------------
    // The nibble-unpack head is emitted by the Track-B single-scope packed-i4
    // FLIP at RVVContractionSelectedBodyRealizationOwner.cpp:2390-2396. That
    // flip only swaps the product HEAD op: it reuses lhsValue/rhsValue from
    // realizeContractionSourceLoad(plan.lhs/plan.rhs) (:2352-2355), which come
    // from the packed-i4 dequant front door RVVDequantDotSourceFrontDoor.cpp
    // :910-930. So the two SOURCES mirror the signed widening-product lhs/rhs
    // decoration (packed-i4 nibble core pinned at i8mf4-i16mf2-i32m1, so
    // srcStripLabel "src-i8mf4"). The ABI tail (the dequant f32 out + the
    // intervening dequant-scale param) is FORM-owned and NOT on this descriptor
    // -- it is keyed by body-op form via getContractionRuntimeABIOrder (see the
    // header note + DESIGN REVISION v2); this route illustrates exactly why the
    // tail cannot be head-keyed (same head as Route 1, different tail).
    // 1c will confirm the flip's emitted metadata string once the nibble route
    // is derived from this descriptor (its role names are assumed to be the
    // shared wprod-lhs/wprod-rhs generic slots -- no distinct nibble role string
    // exists in-tree).
    {
      ContractionRouteIdentity r;
      r.headOpName = "tcrv_rvv.packed_i4_nibble_unpack_product";
      r.isSigned = true;
      r.sources.push_back(ContractionSourceSpec{
          SourceKind::PerIterInputBufferLoad, /*isMultiplicandFactor=*/true,
          /*slotName=*/"lhs", /*roleName=*/"wprod-lhs",
          /*abiRole=*/"lhs-input-buffer", /*abiCName=*/"lhs",
          /*abiCType=*/"const int8_t *", /*srcStripLabel=*/"src-i8mf4",
          /*headOperandIndex=*/0, /*bodyStepPosition=*/0});
      r.sources.push_back(ContractionSourceSpec{
          SourceKind::PerIterInputBufferLoad, /*isMultiplicandFactor=*/true,
          /*slotName=*/"rhs", /*roleName=*/"wprod-rhs",
          /*abiRole=*/"rhs-input-buffer", /*abiCName=*/"rhs",
          /*abiCType=*/"const int8_t *", /*srcStripLabel=*/"src-i8mf4",
          /*headOperandIndex=*/1, /*bodyStepPosition=*/1});
      table.push_back(std::move(r));
    }

    // ======================================================================
    // Route 4: tcrv_rvv.packed_i4_offset_binary_x_i8_product, SIGNED (C3, N=3)
    // ----------------------------------------------------------------------
    // The first N=3 contraction route: the ggml Q4_0 x Q8_0 integer core, where
    // ONLY the packed-i4 weight is nibble-decoded and the TWO plain-i8 q8
    // activation halves (low/high) stay plain. The offset-binary op is
    // PackedI4OffsetBinaryXI8ProductOp (three multiplicand operands: weight,
    // activation_low, activation_high), lowered structurally by RVVToEmitC (no
    // multiplicand-roles string read), so no existing hardcoded roles constant
    // anchors it -- this descriptor is the sole registry source for the C3 route.
    //
    // ABI trio (abiRole / abiCName / abiCType) VERIFIED byte-for-byte vs the
    // offset-binary front door RVVPackedI4DotSourceFrontDoor.cpp:571-579:
    //   weight -> role "lhs-input-buffer", c-name "w",   c-type "const int8_t *"
    //   qlo    -> role "rhs-input-buffer", c-name "qlo", c-type "const int8_t *"
    //   qhi    -> role "rhs-input-buffer", c-name "qhi", c-type "const int8_t *"
    // The TWO same-role rhs-input-buffer sources (qlo/qhi) are exactly why
    // getContractionProductFactorSlotIndex disambiguates on BOTH (role, c-name):
    // qlo -> slot 1, qhi -> slot 2 (see contractionProductSourceBindingC3SelfCheck).
    //
    // Roles-join trio (slotName / roleName / srcStripLabel) INFERRED (no
    // front-door anchor exists -- the op carries no multiplicand-roles metadata
    // string and W1 consumes none): structurally-descriptive tokens mirroring the
    // weight/activation-low/activation-high operand names, src-i8mf4 matching the
    // pinned i8mf4-i16mf2-i32m1 nibble integer core. NOT read on any W1 path; W4
    // (the R2 construction-protocol role-step derive) is where the role-step
    // ORDER is derived from these sources and confirmed -- not W1.
    {
      ContractionRouteIdentity r;
      r.headOpName = "tcrv_rvv.packed_i4_offset_binary_x_i8_product";
      r.isSigned = true;
      r.sources.push_back(ContractionSourceSpec{
          SourceKind::PerIterInputBufferLoad, /*isMultiplicandFactor=*/true,
          /*slotName=*/"weight", /*roleName=*/"wprod-weight",
          /*abiRole=*/"lhs-input-buffer", /*abiCName=*/"w",
          /*abiCType=*/"const int8_t *", /*srcStripLabel=*/"src-i8mf4",
          /*headOperandIndex=*/0, /*bodyStepPosition=*/0});
      r.sources.push_back(ContractionSourceSpec{
          SourceKind::PerIterInputBufferLoad, /*isMultiplicandFactor=*/true,
          /*slotName=*/"activation-low", /*roleName=*/"wprod-activation-low",
          /*abiRole=*/"rhs-input-buffer", /*abiCName=*/"qlo",
          /*abiCType=*/"const int8_t *", /*srcStripLabel=*/"src-i8mf4",
          /*headOperandIndex=*/1, /*bodyStepPosition=*/1});
      r.sources.push_back(ContractionSourceSpec{
          SourceKind::PerIterInputBufferLoad, /*isMultiplicandFactor=*/true,
          /*slotName=*/"activation-high", /*roleName=*/"wprod-activation-high",
          /*abiRole=*/"rhs-input-buffer", /*abiCName=*/"qhi",
          /*abiCType=*/"const int8_t *", /*srcStripLabel=*/"src-i8mf4",
          /*headOperandIndex=*/2, /*bodyStepPosition=*/2});
      table.push_back(std::move(r));
    }

    return table;
  }();
  return registry;
}

} // namespace

const ContractionRouteIdentity *
getContractionRouteIdentity(llvm::StringRef mnemonic, bool isSigned) {
  for (const ContractionRouteIdentity &route : contractionRouteRegistry()) {
    if (route.headOpName == mnemonic && route.isSigned == isSigned)
      return &route;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// OPTIONAL self-check (NOT wired into any emit path).
//
// Proves BOTH the signed and the unsigned widening-product routes' source data
// reproduce kRVVLowPrecisionSignedWideningProductMultiplicandRoles /
// kRVVLowPrecisionUnsignedWideningProductMultiplicandRoles VERBATIM -- i.e. the
// registry is "1c-ready" for the roles-join derivation. Never called from any
// consumer; exists so the join logic is compiled + the invariant is executable
// (e.g. from a future unit test). We compare against a LOCAL copy of the string
// rather than #including the EmitC internal header, because 1c inverts that
// dependency (the internal constant will come to DERIVE from this registry).
//===----------------------------------------------------------------------===//
namespace {

/// join order = axis-A (headOperandIndex). Per-source token:
///   slotName '=' abiRole ':' roleName ':' srcStripLabel
/// joined by ';'. Reproduces the multiplicand-roles metadata string.
std::string joinMultiplicandRoles(const ContractionRouteIdentity &route) {
  std::string out;
  bool first = true;
  for (const ContractionSourceSpec &s : route.sources) {
    if (!s.isMultiplicandFactor)
      continue;
    if (!first)
      out += ";";
    first = false;
    out += s.slotName.str();
    out += "=";
    out += s.abiRole.str();
    out += ":";
    out += s.roleName.str();
    out += ":";
    out += s.srcStripLabel.str();
  }
  return out;
}

// Process-lifetime cache of the joined multiplicand-roles summary strings, one
// per registry entry, in the same order as contractionRouteRegistry(). Gives
// getContractionMultiplicandRoleSummary a stable StringRef to return (the join
// itself produces a temporary std::string, which a StringRef field could not
// safely alias). Function-local static -> thread-safe one-time init.
const std::vector<std::string> &contractionMultiplicandRoleSummaryTable() {
  static const std::vector<std::string> table = [] {
    std::vector<std::string> t;
    for (const ContractionRouteIdentity &route : contractionRouteRegistry())
      t.push_back(joinMultiplicandRoles(route));
    return t;
  }();
  return table;
}

} // namespace

unsigned
getContractionProductFactorCount(const ContractionRouteIdentity &route) {
  unsigned count = 0;
  for (const ContractionSourceSpec &source : route.sources)
    if (source.kind == SourceKind::PerIterInputBufferLoad &&
        source.isMultiplicandFactor)
      ++count;
  return count;
}

std::optional<unsigned>
getContractionProductFactorSlotIndex(const ContractionRouteIdentity &route,
                                     llvm::StringRef abiRole,
                                     llvm::StringRef abiCName) {
  // k = ordinal position among the product-factor PerIterInputBufferLoad
  // sources (the productSources[] slot). The non-product-factor aux sources
  // (e.g. the C4 constant table) are skipped and do NOT advance k, so the slot
  // stays contiguous with the productSources[] arity.
  unsigned k = 0;
  for (const ContractionSourceSpec &source : route.sources) {
    if (source.kind != SourceKind::PerIterInputBufferLoad ||
        !source.isMultiplicandFactor)
      continue;
    // Match on BOTH the ABI role and the ABI c-name -- the 2b validation that
    // the bound parameter agrees with the descriptor's sources[k] decoration.
    if (source.abiRole == abiRole && source.abiCName == abiCName)
      return k;
    ++k;
  }
  return std::nullopt;
}

llvm::StringRef getContractionMultiplicandRoleSummary(llvm::StringRef mnemonic,
                                                      bool isSigned) {
  const std::vector<ContractionRouteIdentity> &registry =
      contractionRouteRegistry();
  const std::vector<std::string> &summaries =
      contractionMultiplicandRoleSummaryTable();
  for (std::size_t i = 0; i < registry.size(); ++i) {
    if (registry[i].headOpName == mnemonic && registry[i].isSigned == isSigned)
      return summaries[i];
  }
  return llvm::StringRef();
}

bool contractionRouteIdentityRegistrySelfCheck() {
  // EXACT copy of kRVVLowPrecisionSignedWideningProductMultiplicandRoles
  // (RVVEmitCContractionRouteFamilyInternal.h:167-170).
  static constexpr llvm::StringLiteral kExpectedSignedWprodRoles(
      "lhs=lhs-input-buffer:wprod-lhs:src-i8mf4;"
      "rhs=rhs-input-buffer:wprod-rhs:src-i8mf4");
  // EXACT copy of kRVVLowPrecisionUnsignedWideningProductMultiplicandRoles
  // (RVVEmitCContractionRouteFamilyInternal.h:172-174).
  static constexpr llvm::StringLiteral kExpectedUnsignedWprodRoles(
      "lhs=lhs-input-buffer:wprod-lhs:src-u8mf4;"
      "rhs=rhs-input-buffer:wprod-rhs:src-u8mf4");

  const ContractionRouteIdentity *signedWprod =
      getContractionRouteIdentity("tcrv_rvv.widening_product",
                                  /*isSigned=*/true);
  if (!signedWprod)
    return false;
  if (joinMultiplicandRoles(*signedWprod) != kExpectedSignedWprodRoles)
    return false;

  const ContractionRouteIdentity *unsignedWprod =
      getContractionRouteIdentity("tcrv_rvv.widening_product",
                                  /*isSigned=*/false);
  if (!unsignedWprod)
    return false;
  if (joinMultiplicandRoles(*unsignedWprod) != kExpectedUnsignedWprodRoles)
    return false;

  // The packed-i4 nibble-unpack route mirrors the SIGNED widening-product
  // multiplicand decoration (same lhs/rhs slots, src-i8mf4): the Track-B flip
  // only swaps the product HEAD op, reusing lhsValue/rhsValue. Prove its join
  // reproduces the signed roles VERBATIM. (No live call site keys on the nibble
  // head yet -- that is 1e/C3 activation; this self-check is the only
  // reproduction proof for Route 3 in the roles-derive pass.)
  const ContractionRouteIdentity *nibble = getContractionRouteIdentity(
      "tcrv_rvv.packed_i4_nibble_unpack_product", /*isSigned=*/true);
  if (!nibble)
    return false;
  if (joinMultiplicandRoles(*nibble) != kExpectedSignedWprodRoles)
    return false;

  // The public cache-backed accessor must agree with the direct join for every
  // migrated route (this is the surface R1 producers/validators consume).
  return getContractionMultiplicandRoleSummary("tcrv_rvv.widening_product",
                                               /*isSigned=*/true) ==
             kExpectedSignedWprodRoles &&
         getContractionMultiplicandRoleSummary("tcrv_rvv.widening_product",
                                               /*isSigned=*/false) ==
             kExpectedUnsignedWprodRoles &&
         getContractionMultiplicandRoleSummary(
             "tcrv_rvv.packed_i4_nibble_unpack_product", /*isSigned=*/true) ==
             kExpectedSignedWprodRoles;
}

//===----------------------------------------------------------------------===//
// OPTIONAL 2b self-check (NOT wired into any emit path).
//
// Proves the identity-driven productSources[] SLOT derivation
// (getContractionProductFactorSlotIndex + getContractionProductFactorCount)
// reproduces the 2a hardcoded aliasing for EVERY registered N=2 product route:
// the lhs-input-buffer/"lhs" binding lands at productSources[0] and the
// rhs-input-buffer/"rhs" binding lands at productSources[1], with arity 2. This
// is the correctness proof of the 2b load-binding change, which the byte gate
// CANNOT observe because productSources[] has no reader on any emit path yet
// (2c brings the reader online). Same pattern as
// contractionRouteIdentityRegistrySelfCheck: defined + compiled + runnable off
// any emit path, never called from a consumer.
//===----------------------------------------------------------------------===//
bool contractionProductSourceBindingSelfCheck() {
  struct Route {
    llvm::StringRef mnemonic;
    bool isSigned;
  };
  const Route routes[] = {
      {"tcrv_rvv.widening_product", /*isSigned=*/true},
      {"tcrv_rvv.widening_product", /*isSigned=*/false},
      {"tcrv_rvv.packed_i4_nibble_unpack_product", /*isSigned=*/true},
  };

  for (const Route &r : routes) {
    const ContractionRouteIdentity *route =
        getContractionRouteIdentity(r.mnemonic, r.isSigned);
    if (!route)
      return false;
    // Arity 2 = the productSources[] size the 2b load-binding resizes to.
    if (getContractionProductFactorCount(*route) != 2)
      return false;
    // lhs binding -> slot 0 (reproduces 2a's productSources[0] alias of lhs).
    std::optional<unsigned> lhsSlot =
        getContractionProductFactorSlotIndex(*route, "lhs-input-buffer", "lhs");
    if (!lhsSlot || *lhsSlot != 0)
      return false;
    // rhs binding -> slot 1 (reproduces 2a's productSources[1] alias of rhs).
    std::optional<unsigned> rhsSlot =
        getContractionProductFactorSlotIndex(*route, "rhs-input-buffer", "rhs");
    if (!rhsSlot || *rhsSlot != 1)
      return false;
    // A non-product-factor role/c-name is NOT a product-factor source: the
    // route-provider error path (nullopt). This can only fire on a genuinely
    // malformed binding, never for the valid N=2 lhs/rhs above.
    if (getContractionProductFactorSlotIndex(*route, "accumulator-input-buffer",
                                             "acc"))
      return false;
    // A right role with the wrong c-name must ALSO miss (the abiCName half of
    // the 2b validation).
    if (getContractionProductFactorSlotIndex(*route, "rhs-input-buffer",
                                             "wrong-c-name"))
      return false;
  }

  return true;
}

//===----------------------------------------------------------------------===//
// OPTIONAL P1e/W1 C3 self-check (NOT wired into any emit path).
//
// The N=2 sibling above cannot cover the C3 route: it asserts arity==2 and a
// single lhs/rhs slot pair. C3 is the first N=3 route with TWO same-abiRole
// (rhs-input-buffer) product-factor sources (qlo/qhi), so it is the first route
// whose disambiguation genuinely EXERCISES the (abiRole, abiCName) join in
// getContractionProductFactorSlotIndex -- role alone is ambiguous. This proves:
// arity==3; weight/"w" -> slot 0; qlo/"qlo" -> slot 1; qhi/"qhi" -> slot 2 (the
// two rhs-input-buffer sources split ONLY by c-name); and that a right-role /
// wrong-c-name and a wrong-role / right-c-name binding both miss. Defined +
// compiled + runnable off any emit path, never called from a consumer (same
// pattern as contractionProductSourceBindingSelfCheck).
//===----------------------------------------------------------------------===//
bool contractionProductSourceBindingC3SelfCheck() {
  const ContractionRouteIdentity *route = getContractionRouteIdentity(
      "tcrv_rvv.packed_i4_offset_binary_x_i8_product", /*isSigned=*/true);
  if (!route)
    return false;
  // Arity 3 = the productSources[] size the N=3 load-binding resizes to (W2/W3).
  if (getContractionProductFactorCount(*route) != 3)
    return false;
  // weight binding -> slot 0.
  std::optional<unsigned> weightSlot =
      getContractionProductFactorSlotIndex(*route, "lhs-input-buffer", "w");
  if (!weightSlot || *weightSlot != 0)
    return false;
  // qlo binding (rhs-input-buffer / "qlo") -> slot 1.
  std::optional<unsigned> qloSlot =
      getContractionProductFactorSlotIndex(*route, "rhs-input-buffer", "qlo");
  if (!qloSlot || *qloSlot != 1)
    return false;
  // qhi binding (SAME rhs-input-buffer role, distinct c-name "qhi") -> slot 2.
  // This is the (abiRole, abiCName) disambiguation the whole C3 route rests on.
  std::optional<unsigned> qhiSlot =
      getContractionProductFactorSlotIndex(*route, "rhs-input-buffer", "qhi");
  if (!qhiSlot || *qhiSlot != 2)
    return false;
  // Right role, wrong c-name must miss (the c-name half of the disambiguation).
  if (getContractionProductFactorSlotIndex(*route, "rhs-input-buffer",
                                           "wrong-c-name"))
    return false;
  // Right c-name, wrong role must ALSO miss (the role half): "w" is bound to
  // lhs-input-buffer, so it is not an rhs-input-buffer product factor.
  if (getContractionProductFactorSlotIndex(*route, "rhs-input-buffer", "w"))
    return false;
  return true;
}

} // namespace tianchenrv::plugin::rvv
