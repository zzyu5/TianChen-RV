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

} // namespace tianchenrv::plugin::rvv
