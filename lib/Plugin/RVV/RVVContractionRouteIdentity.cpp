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

    // ---- Shared generic fixed-tail specs (route-independent decoration) ----
    // outputRole / runtimeCountRole are the shared generic ABI roles
    // (RVVEmitCContractionRouteFamilyPlanOwners.cpp:60-61). The signed
    // dot-reduce tail c-types are VERIFIED vs RVVReductionSourceFrontDoor.cpp
    // :712-720 (acc "const int32_t *", out "int32_t *", n "size_t").
    auto signedAccSpec = ContractionSourceSpec{
        SourceKind::PerIterInputBufferLoad, /*isMultiplicandFactor=*/false,
        /*slotName=*/"acc", /*roleName=*/"",
        /*abiRole=*/"accumulator-input-buffer", /*abiCName=*/"acc",
        /*abiCType=*/"const int32_t *", /*srcStripLabel=*/"",
        /*headOperandIndex=*/2, /*bodyStepPosition=*/0};
    auto signedI32OutSpec = ContractionSourceSpec{
        SourceKind::PerIterInputBufferLoad, /*isMultiplicandFactor=*/false,
        /*slotName=*/"out", /*roleName=*/"", /*abiRole=*/"output-buffer",
        /*abiCName=*/"out", /*abiCType=*/"int32_t *", /*srcStripLabel=*/"",
        /*headOperandIndex=*/3, /*bodyStepPosition=*/0};
    auto nSpec = ContractionSourceSpec{
        SourceKind::PerIterInputBufferLoad, /*isMultiplicandFactor=*/false,
        /*slotName=*/"n", /*roleName=*/"", /*abiRole=*/"runtime-element-count",
        /*abiCName=*/"n", /*abiCType=*/"size_t", /*srcStripLabel=*/"",
        /*headOperandIndex=*/4, /*bodyStepPosition=*/0};

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
      r.accSpec = signedAccSpec;
      r.outSpec = signedI32OutSpec;
      r.nSpec = nSpec;
      r.reduceOpName = "tcrv_rvv.standalone_reduce";
      // "product_relation" widening-product constant, signed
      // (kRVVLowPrecisionPrimitiveSignedProductKind,
      //  RVVEmitCContractionRouteFamilyInternal.h:120-121).
      r.productRelation = "signed-i8mf4xi8mf4-to-i16mf2-widening-product.v1";
      r.leafProfile = ""; // no in-tree contraction leaf-profile constant (see header)
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
    // INFERRED (no front-door anchor found for the unsigned route): abiCType
    // "const uint8_t *" as the natural analog of "const int8_t *" matching
    // src-u8mf4; likewise the u32 accumulator/out tail follows the unsigned
    // reduction primitive kind (u32m1). 1c must confirm the unsigned route is
    // front-door reachable (or resolve it as metadata-only) before deriving.
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
      // INFERRED unsigned tail (u32m1); 1c to confirm.
      r.accSpec = ContractionSourceSpec{
          SourceKind::PerIterInputBufferLoad, /*isMultiplicandFactor=*/false,
          /*slotName=*/"acc", /*roleName=*/"",
          /*abiRole=*/"accumulator-input-buffer", /*abiCName=*/"acc",
          /*abiCType=*/"const uint32_t *", /*srcStripLabel=*/"",
          /*headOperandIndex=*/2, /*bodyStepPosition=*/0};
      r.outSpec = ContractionSourceSpec{
          SourceKind::PerIterInputBufferLoad, /*isMultiplicandFactor=*/false,
          /*slotName=*/"out", /*roleName=*/"", /*abiRole=*/"output-buffer",
          /*abiCName=*/"out", /*abiCType=*/"uint32_t *", /*srcStripLabel=*/"",
          /*headOperandIndex=*/3, /*bodyStepPosition=*/0};
      r.nSpec = nSpec;
      r.reduceOpName = "tcrv_rvv.standalone_reduce";
      // kRVVLowPrecisionPrimitiveUnsignedProductKind
      // (RVVEmitCContractionRouteFamilyInternal.h:122-123).
      r.productRelation = "unsigned-u8mf4xu8mf4-to-u16mf2-widening-product.v1";
      r.leafProfile = "";
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
    // srcStripLabel "src-i8mf4"). The tail is the dequant tail VERIFIED vs
    // RVVDequantDotSourceFrontDoor.cpp:916-930 (acc "const int32_t *",
    // out "float *", n "size_t"); the intervening f32 dequant-scale param is
    // modeled by 1c/1d via conditionalInserts, not a fixed-tail field.
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
      r.accSpec = signedAccSpec;
      // Dequant path: f32 output (RVVDequantDotSourceFrontDoor.cpp:925-927).
      r.outSpec = ContractionSourceSpec{
          SourceKind::PerIterInputBufferLoad, /*isMultiplicandFactor=*/false,
          /*slotName=*/"out", /*roleName=*/"", /*abiRole=*/"output-buffer",
          /*abiCName=*/"out", /*abiCType=*/"float *", /*srcStripLabel=*/"",
          /*headOperandIndex=*/3, /*bodyStepPosition=*/0};
      r.nSpec = nSpec;
      r.reduceOpName = "tcrv_rvv.standalone_reduce";
      // product_relation is candidate-driven at realization
      // (selectedResourceCandidate->primitiveWideningProductRelation,
      //  RVVContractionSelectedBodyRealizationOwner.cpp:2371-2376); the signed
      // widening-product kind is the representative value.
      r.productRelation = "signed-i8mf4xi8mf4-to-i16mf2-widening-product.v1";
      r.leafProfile = "";
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
// Proves the signed widening-product route's source data reproduces
// kRVVLowPrecisionSignedWideningProductMultiplicandRoles VERBATIM -- i.e. the
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

} // namespace

bool contractionRouteIdentityRegistrySelfCheck() {
  // EXACT copy of kRVVLowPrecisionSignedWideningProductMultiplicandRoles
  // (RVVEmitCContractionRouteFamilyInternal.h:167-170).
  static constexpr llvm::StringLiteral kExpectedSignedWprodRoles(
      "lhs=lhs-input-buffer:wprod-lhs:src-i8mf4;"
      "rhs=rhs-input-buffer:wprod-rhs:src-i8mf4");

  const ContractionRouteIdentity *signedWprod =
      getContractionRouteIdentity("tcrv_rvv.widening_product",
                                  /*isSigned=*/true);
  if (!signedWprod)
    return false;
  return joinMultiplicandRoles(*signedWprod) == kExpectedSignedWprodRoles;
}

} // namespace tianchenrv::plugin::rvv
