#include "TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <utility>

namespace tianchenrv::plugin::rvv {

llvm::Error makeRVVEmitCRouteProviderError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine(
          "TianChen-RV RVV plugin-owned EmitC route provider failed: ") +
          message,
      llvm::errc::invalid_argument);
}

namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral
    kRVVSelectedBodyEmissionKind("materialized-emitc-cpp-rvv-intrinsic-object");
constexpr llvm::StringLiteral
    kRVVSelectedBodyLoweringBoundaryOpName("tcrv_rvv.with_vl");
constexpr llvm::StringLiteral
    kRVVSelectedBodyRuntimeABIKind("plugin-owned-runtime-abi");
constexpr llvm::StringLiteral
    kRVVSelectedBodyRuntimeGlueRole("emitc-cpp-rvv-intrinsic-runtime-glue");
constexpr llvm::StringLiteral
    kRVVReductionAccumulatorLayout("rhs-vector-seed-lane0-per-vl-chunk");
constexpr llvm::StringLiteral
    kRVVReductionResultLayout("store-reduction-lane0-to-output-chunk-base");
constexpr llvm::StringLiteral kRVVReductionStoreVL("1");
constexpr llvm::StringLiteral
    kRVVMaskedCompareMaskSource("compare-produced-mask-same-vl-scope");
constexpr llvm::StringLiteral
    kRVVMaskedPredicateMaskRole("predicate-mask-produced-by-compare");
constexpr llvm::StringLiteral kRVVMaskedInactiveLaneContract(
    "masked-off-lanes-preserve-passthrough-vector");
constexpr llvm::StringLiteral kRVVMaskedPassthroughLayout(
    "passthrough-vector-preserves-inactive-lanes");
constexpr llvm::StringLiteral
    kRVVMAccAccumulatorLayout("output-buffer-vector-accumulator-input");
constexpr llvm::StringLiteral
    kRVVMAccResultLayout("store-multiply-accumulate-result-to-output-buffer");
constexpr llvm::StringLiteral kRVVStridedRuntimeABIOrder(
    "lhs,rhs,out,n,lhs_stride,rhs_stride,out_stride");
constexpr llvm::StringLiteral kRVVStridedLoadUnitStoreRuntimeABIOrder(
    "src,out,n,src_stride");
constexpr llvm::StringLiteral kRVVIndexedGatherRuntimeABIOrder(
    "data,index,out,n");
constexpr llvm::StringLiteral kRVVIndexedScatterRuntimeABIOrder(
    "src,index,dst,n");
constexpr llvm::StringLiteral kRVVMaskedMemoryRuntimeABIOrder(
    "src,mask,dst,n");
constexpr llvm::StringLiteral kRVVScalarBroadcastRuntimeABIOrder(
    "lhs,rhs_scalar,out,n");
constexpr llvm::StringLiteral kRVVWideningConversionRuntimeABIOrder(
    "lhs,out,n");
constexpr llvm::StringLiteral kRVVStridedMemoryLayout(
    "element-strided-lhs-rhs-output-runtime-abi");
constexpr llvm::StringLiteral kRVVStridedLoadUnitStoreMemoryLayout(
    "element-strided-source-unit-stride-output-runtime-abi");
constexpr llvm::StringLiteral kRVVIndexedGatherMemoryLayout(
    "element-indexed-data-index-unit-stride-output-runtime-abi");
constexpr llvm::StringLiteral kRVVIndexedScatterMemoryLayout(
    "unit-stride-source-indexed-destination-index-runtime-abi");
constexpr llvm::StringLiteral kRVVMaskedMemoryLayout(
    "unit-stride-source-mask-old-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVLHSStrideSource("runtime_abi:lhs_stride");
constexpr llvm::StringLiteral kRVVRHSStrideSource("runtime_abi:rhs_stride");
constexpr llvm::StringLiteral kRVVOutStrideSource("runtime_abi:out_stride");
constexpr llvm::StringLiteral kRVVSourceStrideSource(
    "runtime_abi:src_stride");
constexpr llvm::StringLiteral kRVVSourceMemoryForm("strided-load");
constexpr llvm::StringLiteral kRVVUnitStrideSourceMemoryForm(
    "unit-stride-load");
constexpr llvm::StringLiteral kRVVIndexedDataMemoryForm("indexed-load");
constexpr llvm::StringLiteral kRVVIndexedDestinationMemoryForm(
    "indexed-store");
constexpr llvm::StringLiteral kRVVIndexSource("runtime_abi:index");
constexpr llvm::StringLiteral kRVVMaskSource("runtime_abi:mask");
constexpr llvm::StringLiteral kRVVMaskRole("predicate-mask-input-buffer");
constexpr llvm::StringLiteral kRVVMaskMemoryForm("unit-stride-mask-load");
constexpr llvm::StringLiteral kRVVMaskedMemoryInactiveLaneContract(
    "masked-off-lanes-preserve-old-destination");
constexpr llvm::StringLiteral kRVVMaskedMemoryPassthroughLayout(
    "old-destination-vector-preserves-inactive-lanes");
constexpr llvm::StringLiteral kRVVIndexedGatherOffsetUnit("element");
constexpr llvm::StringLiteral kRVVIndexedScatterIndexUniqueness("unique");
constexpr llvm::StringLiteral kRVVDestinationMemoryForm("unit-stride-store");
constexpr llvm::StringLiteral kRVVWideningConversionRelation(
    "signed-i32m1-to-i64m2");

struct RVVSelectedBodyOperationProfile {
  RVVSelectedBodyOperationKind operation;
  llvm::StringRef operationMnemonic;
  llvm::StringRef resultName;
  llvm::StringRef maskName;
  bool isCompareSelect;
  bool isReduction;
  bool isMaskedArithmetic;
  bool isMultiplyAccumulate;
  bool isStridedMemory;
  bool isMemoryMovement;
  bool isIndexedMemoryMovement;
  bool isMaskedMemoryMovement;
  bool isWideningConversion;
};

struct RVVSelectedBodyConfigProfile {
  std::int64_t sew;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  const tcrv::rvv::RVVSelectedBodyConfigVLContract *configContract = nullptr;
  llvm::StringRef vlCType;
  llvm::StringRef vectorTypeName;
  llvm::StringRef indexVectorTypeName;
  llvm::StringRef maskTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef indexVectorCType;
  llvm::StringRef maskCType;
  llvm::StringRef scalarCType;
  llvm::StringRef constInputPointerCType;
  llvm::StringRef outputPointerCType;
  llvm::StringRef elementByteSize;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef indexLoadIntrinsic;
  llvm::StringRef indexScaleIntrinsic;
  llvm::StringRef indexedLoadIntrinsic;
  llvm::StringRef indexedStoreIntrinsic;
  llvm::StringRef stridedLoadIntrinsic;
  llvm::StringRef rhsBroadcastIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef stridedStoreIntrinsic;
};

struct RVVSelectedBodyTargetLeafProfile {
  llvm::StringRef intrinsic;
  llvm::StringRef compareIntrinsic;
  llvm::StringRef maskedMergeIntrinsic;
  llvm::StringRef rhsBroadcastIntrinsic;
};

struct RVVSelectedBodyRouteProfile {
  RVVSelectedBodyOperationProfile operation;
  RVVSelectedBodyConfigProfile config;
  RVVSelectedBodyTargetLeafProfile targetLeaves;
};

constexpr RVVSelectedBodyOperationKind kRVVSelectedBodyOperationKinds[] = {
    RVVSelectedBodyOperationKind::Add, RVVSelectedBodyOperationKind::Sub,
    RVVSelectedBodyOperationKind::Mul,
    RVVSelectedBodyOperationKind::CmpSelect,
    RVVSelectedBodyOperationKind::ReduceAdd,
    RVVSelectedBodyOperationKind::MaskedAdd,
    RVVSelectedBodyOperationKind::MAccAdd,
    RVVSelectedBodyOperationKind::StridedAdd,
    RVVSelectedBodyOperationKind::StridedLoadUnitStore,
    RVVSelectedBodyOperationKind::IndexedGatherUnitStore,
    RVVSelectedBodyOperationKind::IndexedScatterUnitLoad,
    RVVSelectedBodyOperationKind::MaskedUnitLoadStore,
    RVVSelectedBodyOperationKind::ScalarBroadcastAdd,
    RVVSelectedBodyOperationKind::WidenI32ToI64};

const RVVSelectedBodyOperationProfile &
getRVVSelectedBodyOperationProfile(RVVSelectedBodyOperationKind op) {
  static const RVVSelectedBodyOperationProfile kAdd = {
      RVVSelectedBodyOperationKind::Add, "add", "sum_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kSub = {
      RVVSelectedBodyOperationKind::Sub, "sub", "difference_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kMul = {
      RVVSelectedBodyOperationKind::Mul, "mul", "product_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kCmpSelect = {
      RVVSelectedBodyOperationKind::CmpSelect, "cmp_select", "selected_vec",
      "cmp_mask", /*isCompareSelect=*/true, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kReduceAdd = {
      RVVSelectedBodyOperationKind::ReduceAdd, "reduce_add", "reduced_vec",
      "", /*isCompareSelect=*/false, /*isReduction=*/true,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kMaskedAdd = {
      RVVSelectedBodyOperationKind::MaskedAdd, "masked_add",
      "masked_sum_vec", "add_mask", /*isCompareSelect=*/false,
      /*isReduction=*/false, /*isMaskedArithmetic=*/true,
      /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
      /*isMemoryMovement=*/false, /*isIndexedMemoryMovement=*/false,
      /*isMaskedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kMAccAdd = {
      RVVSelectedBodyOperationKind::MAccAdd, "macc_add", "macc_sum_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/true,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kStridedAdd = {
      RVVSelectedBodyOperationKind::StridedAdd, "strided_add",
      "strided_sum_vec", "", /*isCompareSelect=*/false,
      /*isReduction=*/false, /*isMaskedArithmetic=*/false,
      /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/true,
      /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kStridedLoadUnitStore = {
      RVVSelectedBodyOperationKind::StridedLoadUnitStore,
      "strided_load_unit_store", "loaded_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/true, /*isMemoryMovement=*/true,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kIndexedGatherUnitStore = {
      RVVSelectedBodyOperationKind::IndexedGatherUnitStore,
      "indexed_gather_unit_store", "loaded_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/true, /*isMaskedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kIndexedScatterUnitLoad = {
      RVVSelectedBodyOperationKind::IndexedScatterUnitLoad,
      "indexed_scatter_unit_load", "loaded_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/true, /*isMaskedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kMaskedUnitLoadStore = {
      RVVSelectedBodyOperationKind::MaskedUnitLoadStore,
      "masked_unit_load_store", "masked_loaded_vec", "memory_mask",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/true,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kScalarBroadcastAdd = {
      RVVSelectedBodyOperationKind::ScalarBroadcastAdd,
      "scalar_broadcast_add", "sum_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kWidenI32ToI64 = {
      RVVSelectedBodyOperationKind::WidenI32ToI64, "widen_i32_to_i64",
      "widened_vec", "", /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isWideningConversion=*/true};

  switch (op) {
  case RVVSelectedBodyOperationKind::Add:
    return kAdd;
  case RVVSelectedBodyOperationKind::Sub:
    return kSub;
  case RVVSelectedBodyOperationKind::Mul:
    return kMul;
  case RVVSelectedBodyOperationKind::CmpSelect:
    return kCmpSelect;
  case RVVSelectedBodyOperationKind::ReduceAdd:
    return kReduceAdd;
  case RVVSelectedBodyOperationKind::MaskedAdd:
    return kMaskedAdd;
  case RVVSelectedBodyOperationKind::MAccAdd:
    return kMAccAdd;
  case RVVSelectedBodyOperationKind::StridedAdd:
    return kStridedAdd;
  case RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    return kStridedLoadUnitStore;
  case RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
    return kIndexedGatherUnitStore;
  case RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    return kIndexedScatterUnitLoad;
  case RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    return kMaskedUnitLoadStore;
  case RVVSelectedBodyOperationKind::ScalarBroadcastAdd:
    return kScalarBroadcastAdd;
  case RVVSelectedBodyOperationKind::WidenI32ToI64:
    return kWidenI32ToI64;
  }
  llvm_unreachable("unknown RVV selected-body operation");
}

llvm::Error makeUnsupportedRVVSelectedBodyRouteProfileError(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("unsupported RVV selected-body route profile: operation=") +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      ", memory_form=" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) +
      ", SEW=" + llvm::Twine(description.sew) + ", LMUL=" + description.lmul +
      ", tail_policy=" + description.tailPolicy +
      ", mask_policy=" + description.maskPolicy);
}

llvm::Expected<RVVSelectedBodyConfigProfile>
deriveRVVSelectedBodyConfigProfile(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.tailPolicy != "agnostic" ||
      description.maskPolicy != "agnostic")
    return makeUnsupportedRVVSelectedBodyRouteProfileError(description);

  if (description.sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      description.lmul == tcrv::rvv::getRVVLMULM1())
    return RVVSelectedBodyConfigProfile{
        32,
        "m1",
        "agnostic",
        "agnostic",
        &tcrv::rvv::getRVVSelectedBodyConfigVLContract("m1"),
        "size_t",
        "!tcrv_rvv.vector<i32, \"m1\">",
        "!tcrv_rvv.index_vector<i32, \"m1\">",
        "!tcrv_rvv.mask<i32, \"m1\">",
        "vint32m1_t",
        "vuint32m1_t",
        "vbool32_t",
        "int32_t",
        "const int32_t *",
        "int32_t *",
        "4",
        "__riscv_vsetvl_e32m1",
        "__riscv_vle32_v_i32m1",
        "__riscv_vle32_v_u32m1",
        "__riscv_vmul_vx_u32m1",
        "__riscv_vloxei32_v_i32m1",
        "__riscv_vsoxei32_v_i32m1",
        "__riscv_vlse32_v_i32m1",
        "__riscv_vmv_v_x_i32m1",
        "__riscv_vse32_v_i32m1",
        "__riscv_vsse32_v_i32m1"};

  if (description.sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      description.lmul == tcrv::rvv::getRVVLMULM2())
    return RVVSelectedBodyConfigProfile{
        32,
        "m2",
        "agnostic",
        "agnostic",
        &tcrv::rvv::getRVVSelectedBodyConfigVLContract("m2"),
        "size_t",
        "!tcrv_rvv.vector<i32, \"m2\">",
        "",
        "!tcrv_rvv.mask<i32, \"m2\">",
        "vint32m2_t",
        "",
        "vbool16_t",
        "int32_t",
        "const int32_t *",
        "int32_t *",
        "4",
        "__riscv_vsetvl_e32m2",
        "__riscv_vle32_v_i32m2",
        "",
        "",
        "",
        "",
        "__riscv_vlse32_v_i32m2",
        "__riscv_vmv_v_x_i32m2",
        "__riscv_vse32_v_i32m2",
        "__riscv_vsse32_v_i32m2"};

  if (description.sew == tcrv::rvv::getRVVSEW64Bits() &&
      description.lmul == tcrv::rvv::getRVVLMULM1() &&
      description.tailPolicy == "agnostic" &&
      description.maskPolicy == "agnostic")
    return RVVSelectedBodyConfigProfile{
        64,
        "m1",
        "agnostic",
        "agnostic",
        &tcrv::rvv::getRVVSelectedBodyConfigVLContract(64, "m1"),
        "size_t",
        "!tcrv_rvv.vector<i64, \"m1\">",
        "",
        "!tcrv_rvv.mask<i64, \"m1\">",
        "vint64m1_t",
        "",
        "vbool64_t",
        "int64_t",
        "const int64_t *",
        "int64_t *",
        "8",
        "__riscv_vsetvl_e64m1",
        "__riscv_vle64_v_i64m1",
        "",
        "",
        "",
        "",
        "__riscv_vlse64_v_i64m1",
        "__riscv_vmv_v_x_i64m1",
        "__riscv_vse64_v_i64m1",
        "__riscv_vsse64_v_i64m1"};

  if (description.sew == tcrv::rvv::getRVVSEW64Bits() &&
      description.lmul == tcrv::rvv::getRVVLMULM2() &&
      description.tailPolicy == "agnostic" &&
      description.maskPolicy == "agnostic")
    return RVVSelectedBodyConfigProfile{
        64,
        "m2",
        "agnostic",
        "agnostic",
        &tcrv::rvv::getRVVSelectedBodyConfigVLContract(64, "m2"),
        "size_t",
        "!tcrv_rvv.vector<i64, \"m2\">",
        "",
        "!tcrv_rvv.mask<i64, \"m2\">",
        "vint64m2_t",
        "",
        "vbool32_t",
        "int64_t",
        "const int64_t *",
        "int64_t *",
        "8",
        "__riscv_vsetvl_e64m2",
        "__riscv_vle64_v_i64m2",
        "",
        "",
        "",
        "",
        "__riscv_vlse64_v_i64m2",
        "__riscv_vmv_v_x_i64m2",
        "__riscv_vse64_v_i64m2",
        "__riscv_vsse64_v_i64m2"};

  return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
}

llvm::StringRef getRVVSelectedBodyArithmeticIntrinsic(
    RVVSelectedBodyOperationKind operation,
    const RVVSelectedBodyConfigProfile &config) {
  if (config.sew == tcrv::rvv::getRVVSEW64Bits()) {
    if (operation == RVVSelectedBodyOperationKind::Add &&
        config.lmul == tcrv::rvv::getRVVLMULM1())
      return "__riscv_vadd_vv_i64m1";
    return {};
  }

  switch (operation) {
  case RVVSelectedBodyOperationKind::Add:
  case RVVSelectedBodyOperationKind::StridedAdd:
  case RVVSelectedBodyOperationKind::ScalarBroadcastAdd:
    return config.lmul == tcrv::rvv::getRVVLMULM2()
               ? "__riscv_vadd_vv_i32m2"
               : "__riscv_vadd_vv_i32m1";
  case RVVSelectedBodyOperationKind::Sub:
    return config.lmul == tcrv::rvv::getRVVLMULM2()
               ? "__riscv_vsub_vv_i32m2"
               : "__riscv_vsub_vv_i32m1";
  case RVVSelectedBodyOperationKind::Mul:
    return config.lmul == tcrv::rvv::getRVVLMULM2()
               ? "__riscv_vmul_vv_i32m2"
               : "__riscv_vmul_vv_i32m1";
  case RVVSelectedBodyOperationKind::CmpSelect:
    llvm_unreachable("compare/select uses dedicated compare and merge leaves");
  case RVVSelectedBodyOperationKind::ReduceAdd:
    llvm_unreachable("reduction uses dedicated reduction intrinsic leaf");
  case RVVSelectedBodyOperationKind::MaskedAdd:
    llvm_unreachable("masked arithmetic uses dedicated masked intrinsic leaf");
  case RVVSelectedBodyOperationKind::MAccAdd:
    llvm_unreachable("multiply-accumulate uses dedicated macc intrinsic leaf");
  case RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    llvm_unreachable("strided memory movement uses load/store leaves only");
  case RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
    llvm_unreachable("indexed memory movement uses load/store leaves only");
  case RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    llvm_unreachable("indexed memory movement uses load/store leaves only");
  case RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    llvm_unreachable("masked memory movement uses mask/load/store leaves only");
  case RVVSelectedBodyOperationKind::WidenI32ToI64:
    llvm_unreachable("widening conversion uses dedicated conversion leaf");
  }
  llvm_unreachable("unknown RVV selected-body operation");
}

llvm::StringRef getRVVSelectedBodyWideningConversionIntrinsic(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.sourceSEW == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      description.sourceLMUL == tcrv::rvv::getRVVLMULM1() &&
      description.sew == tcrv::rvv::getRVVSEW64Bits() &&
      description.lmul == tcrv::rvv::getRVVLMULM2() &&
      description.conversionRelation == kRVVWideningConversionRelation)
    return "__riscv_vwcvt_x_x_v_i64m2";
  return {};
}

llvm::StringRef
getRVVSelectedBodyMAccIntrinsic(llvm::StringRef lmul) {
  return lmul == tcrv::rvv::getRVVLMULM2()
             ? "__riscv_vmacc_vv_i32m2"
             : "__riscv_vmacc_vv_i32m1";
}

llvm::StringRef
getRVVSelectedBodyReductionIntrinsic(llvm::StringRef lmul) {
  if (lmul == tcrv::rvv::getRVVLMULM1())
    return "__riscv_vredsum_vs_i32m1_i32m1";
  return {};
}

llvm::StringRef
getRVVSelectedBodyCompareIntrinsic(llvm::StringRef lmul) {
  return lmul == tcrv::rvv::getRVVLMULM2()
             ? "__riscv_vmseq_vv_i32m2_b16"
             : "__riscv_vmseq_vv_i32m1_b32";
}

llvm::StringRef getRVVSelectedBodyMaskFromI32Intrinsic(llvm::StringRef lmul) {
  return lmul == tcrv::rvv::getRVVLMULM2()
             ? "__riscv_vmsne_vx_i32m2_b16"
             : "__riscv_vmsne_vx_i32m1_b32";
}

llvm::StringRef
getRVVSelectedBodySelectIntrinsic(llvm::StringRef lmul) {
  return lmul == tcrv::rvv::getRVVLMULM2()
             ? "__riscv_vmerge_vvm_i32m2"
             : "__riscv_vmerge_vvm_i32m1";
}

llvm::Expected<RVVSelectedBodyTargetLeafProfile>
deriveRVVSelectedBodyTargetLeafProfile(
    const RVVSelectedBodyEmitCRouteDescription &description,
    const RVVSelectedBodyOperationProfile &operationProfile,
    const RVVSelectedBodyConfigProfile &configProfile) {
  if (configProfile.sew == tcrv::rvv::getRVVSEW64Bits() &&
      description.operation != RVVSelectedBodyOperationKind::WidenI32ToI64) {
    if (description.operation != RVVSelectedBodyOperationKind::Add ||
        description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad ||
        configProfile.lmul != tcrv::rvv::getRVVLMULM1())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
  }

  if (operationProfile.isWideningConversion) {
    if (description.memoryForm !=
        RVVSelectedBodyMemoryForm::UnitStrideConversion)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    llvm::StringRef intrinsic =
        getRVVSelectedBodyWideningConversionIntrinsic(description);
    if (intrinsic.empty())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{intrinsic, "", "", ""};
  }

  if (operationProfile.isCompareSelect) {
    if (description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{
        getRVVSelectedBodySelectIntrinsic(configProfile.lmul),
        getRVVSelectedBodyCompareIntrinsic(configProfile.lmul), "", ""};
  }

  if (operationProfile.isReduction) {
    if (description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    llvm::StringRef reductionIntrinsic =
        getRVVSelectedBodyReductionIntrinsic(configProfile.lmul);
    if (reductionIntrinsic.empty())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{reductionIntrinsic, "", "", ""};
  }

  if (operationProfile.isMaskedArithmetic) {
    if (description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{
        getRVVSelectedBodyArithmeticIntrinsic(
            RVVSelectedBodyOperationKind::Add, configProfile),
        getRVVSelectedBodyCompareIntrinsic(configProfile.lmul),
        getRVVSelectedBodySelectIntrinsic(configProfile.lmul), ""};
  }

  if (operationProfile.isMultiplyAccumulate) {
    if (description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{
        getRVVSelectedBodyMAccIntrinsic(configProfile.lmul), "", "", ""};
  }

  if (operationProfile.isMemoryMovement) {
    if (description.memoryForm !=
        RVVSelectedBodyMemoryForm::StridedLoadUnitStore)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    if (configProfile.lmul != tcrv::rvv::getRVVLMULM1())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{"", "", "", ""};
  }

  if (operationProfile.isIndexedMemoryMovement) {
    const bool isGather =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::IndexedGatherUnitStore;
    const bool isScatter =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::IndexedScatterUnitLoad;
    if ((isGather &&
         description.memoryForm !=
             RVVSelectedBodyMemoryForm::IndexedLoadUnitStore) ||
        (isScatter &&
         description.memoryForm !=
             RVVSelectedBodyMemoryForm::UnitLoadIndexedStore))
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    if (configProfile.lmul != tcrv::rvv::getRVVLMULM1() ||
        configProfile.indexLoadIntrinsic.empty() ||
        configProfile.indexScaleIntrinsic.empty() ||
        (isGather && configProfile.indexedLoadIntrinsic.empty()) ||
        (isScatter && configProfile.indexedStoreIntrinsic.empty()))
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{"", "", "", ""};
  }

  if (operationProfile.isMaskedMemoryMovement) {
    if (description.memoryForm !=
        RVVSelectedBodyMemoryForm::MaskedUnitLoadStore)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    if (configProfile.lmul != tcrv::rvv::getRVVLMULM1())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{
        "", getRVVSelectedBodyMaskFromI32Intrinsic(configProfile.lmul),
        getRVVSelectedBodySelectIntrinsic(configProfile.lmul), ""};
  }

  if (operationProfile.isStridedMemory) {
    if (description.memoryForm != RVVSelectedBodyMemoryForm::StridedLoadStore)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    if (configProfile.lmul != tcrv::rvv::getRVVLMULM1())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{
        getRVVSelectedBodyArithmeticIntrinsic(
            RVVSelectedBodyOperationKind::StridedAdd, configProfile),
        "", "", ""};
  }

  if (description.memoryForm == RVVSelectedBodyMemoryForm::RHSBroadcastLoad ||
      description.memoryForm == RVVSelectedBodyMemoryForm::RHSScalarBroadcast)
    return RVVSelectedBodyTargetLeafProfile{
        getRVVSelectedBodyArithmeticIntrinsic(description.operation,
                                             configProfile),
        "", "", configProfile.rhsBroadcastIntrinsic};

  if (description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad)
    return makeUnsupportedRVVSelectedBodyRouteProfileError(description);

  return RVVSelectedBodyTargetLeafProfile{
      getRVVSelectedBodyArithmeticIntrinsic(description.operation,
                                           configProfile),
      "", "", ""};
}

llvm::Expected<RVVSelectedBodyRouteProfile>
deriveRVVSelectedBodyRouteProfile(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  RVVSelectedBodyRouteProfile profile;
  profile.operation = getRVVSelectedBodyOperationProfile(description.operation);

  llvm::Expected<RVVSelectedBodyConfigProfile> config =
      deriveRVVSelectedBodyConfigProfile(description);
  if (!config)
    return config.takeError();
  profile.config = *config;

  llvm::Expected<RVVSelectedBodyTargetLeafProfile> targetLeaves =
      deriveRVVSelectedBodyTargetLeafProfile(description, profile.operation,
                                            profile.config);
  if (!targetLeaves)
    return targetLeaves.takeError();
  profile.targetLeaves = *targetLeaves;
  return profile;
}

llvm::Error requireRouteDescriptionText(llvm::StringRef context,
                                        llvm::StringRef field,
                                        llvm::StringRef value) {
  if (!value.trim().empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(llvm::Twine(context) + " " + field +
                                        " must be provider-derived and "
                                        "non-empty");
}

llvm::Error requireRouteDescriptionField(llvm::StringRef context,
                                         llvm::StringRef field,
                                         llvm::StringRef actual,
                                         llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) + " " + field +
      " must mirror selected-body route profile fact '" + expected +
      "' but was '" + actual + "'");
}

llvm::StringRef stringifyRVVTailPolicy(tcrv::rvv::TailPolicy policy) {
  switch (policy) {
  case tcrv::rvv::TailPolicy::Agnostic:
    return "agnostic";
  case tcrv::rvv::TailPolicy::Undisturbed:
    return "undisturbed";
  }
  llvm_unreachable("unknown RVV tail policy");
}

llvm::StringRef stringifyRVVMaskPolicy(tcrv::rvv::MaskPolicy policy) {
  switch (policy) {
  case tcrv::rvv::MaskPolicy::Agnostic:
    return "agnostic";
  case tcrv::rvv::MaskPolicy::Undisturbed:
    return "undisturbed";
  }
  llvm_unreachable("unknown RVV mask policy");
}

const RVVSelectedBodyConstructionRoute &
getRVVSelectedBodyConstructionRouteOrDie(RVVSelectedBodyOperationKind op) {
  const RVVSelectedBodyOperationProfile &profile =
      getRVVSelectedBodyOperationProfile(op);
  llvm::Expected<const RVVSelectedBodyConstructionRoute *> route =
      lookupRVVSelectedBodyConstructionRouteByOperationMnemonic(
          profile.operationMnemonic);
  if (!route) {
    std::string message = llvm::toString(route.takeError());
    llvm::report_fatal_error(llvm::StringRef(message));
  }
  return **route;
}

bool variantContainsExplicitTypedRVVBody(tcrv::exec::VariantOp variant) {
  if (!variant || variant.getBody().empty())
    return false;

  bool found = false;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (found || op == variant.getOperation())
      return;
    if (op->getName().getDialectNamespace() == "tcrv_rvv")
      found = true;
  });
  return found;
}

llvm::Error requireRVVVariantLegality(tcrv::exec::VariantOp variant) {
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "requires a materialized tcrv.exec.variant");

  auto originAttr = variant->getAttrOfType<mlir::StringAttr>("origin");
  if (!originAttr || originAttr.getValue() != kRVVPluginName)
    return makeRVVEmitCRouteProviderError(
        "materialized RVV variant must be owned by origin 'rvv-plugin'");

  if (variantContainsExplicitTypedRVVBody(variant))
    return llvm::Error::success();

  return makeRVVEmitCRouteProviderError(
      "materialized RVV variant requires explicit typed RVV "
      "extension-family body");
}

llvm::Error validateRVVSelectedBodyVectorTypeAgainstConfig(
    mlir::Value value, llvm::StringRef role,
    const tcrv::rvv::RVVCompileTimeConfig &config) {
  auto vectorType = llvm::dyn_cast<tcrv::rvv::VectorType>(value.getType());
  if (!vectorType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " to be a generic !tcrv_rvv.vector value before route construction");

  auto integerElementType =
      llvm::dyn_cast<mlir::IntegerType>(vectorType.getElementType());
  if (!integerElementType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " element type to be an integer type");
  if (integerElementType.getWidth() != config.sew)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " element width " + llvm::Twine(integerElementType.getWidth()) +
        " to match selected config SEW " + llvm::Twine(config.sew));
  if (vectorType.getLmul() != config.lmul)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " LMUL '" + vectorType.getLmul() +
        "' to match selected config LMUL '" + config.lmul + "'");

  return llvm::Error::success();
}

llvm::Error validateRVVSelectedBodyMaskTypeAgainstConfig(
    mlir::Value value, llvm::StringRef role,
    const tcrv::rvv::RVVCompileTimeConfig &config) {
  auto maskType = llvm::dyn_cast<tcrv::rvv::MaskType>(value.getType());
  if (!maskType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " to be a generic !tcrv_rvv.mask value before route construction");

  auto integerElementType =
      llvm::dyn_cast<mlir::IntegerType>(maskType.getElementType());
  if (!integerElementType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " element type to be an integer type");
  if (integerElementType.getWidth() != config.sew)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " element width " + llvm::Twine(integerElementType.getWidth()) +
        " to match selected config SEW " + llvm::Twine(config.sew));
  if (maskType.getLmul() != config.lmul)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " LMUL '" + maskType.getLmul() +
        "' to match selected config LMUL '" + config.lmul + "'");

  return llvm::Error::success();
}

llvm::Error validateRVVSelectedBodyIndexVectorTypeAgainstConfig(
    mlir::Value value, llvm::StringRef role,
    const tcrv::rvv::RVVCompileTimeConfig &config) {
  auto vectorType =
      llvm::dyn_cast<tcrv::rvv::IndexVectorType>(value.getType());
  if (!vectorType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " to be a generic !tcrv_rvv.index_vector value before route "
        "construction");

  auto integerElementType =
      llvm::dyn_cast<mlir::IntegerType>(vectorType.getElementType());
  if (!integerElementType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " element type to be an integer type");
  if (integerElementType.getWidth() != 32)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " index element width 32 for the bounded indexed gather route");
  if (vectorType.getLmul() != config.lmul ||
      vectorType.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " LMUL '" + vectorType.getLmul() +
        "' to match selected config LMUL m1 for indexed gather");

  return llvm::Error::success();
}

llvm::Error validateRVVSelectedBodyTypedConfigFacts(
    const RVVSelectedBodyRouteSlice &slice,
    const tcrv::rvv::RVVCompileTimeConfig &config) {
  if (slice.arithmeticKind == RVVSelectedBodyOperationKind::WidenI32ToI64) {
    tcrv::rvv::RVVCompileTimeConfig sourceConfig;
    sourceConfig.sew = tcrv::rvv::getRVVFirstSliceSEWBits();
    sourceConfig.lmul = tcrv::rvv::getRVVLMULM1();
    sourceConfig.policy = config.policy;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "conversion source vector", sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "conversion result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "conversion stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::StridedLoadUnitStore) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "strided source vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "movement result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "unit-stride stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::IndexedGatherUnitStore) {
    if (llvm::Error error = validateRVVSelectedBodyIndexVectorTypeAgainstConfig(
            slice.indexValue, "indexed gather index vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "indexed gather loaded data vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "movement result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "unit-stride stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::IndexedScatterUnitLoad) {
    if (llvm::Error error = validateRVVSelectedBodyIndexVectorTypeAgainstConfig(
            slice.indexValue, "indexed scatter index vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "indexed scatter source vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "movement result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "indexed-store stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::MaskedUnitLoadStore) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "masked memory source vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.accumulatorValue, "masked memory old destination vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.maskValue, "masked memory predicate mask", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "masked movement result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "masked unit-stride stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
          slice.lhsValue, "lhs vector", config))
    return error;
  if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
          slice.rhsValue, "rhs vector", config))
    return error;
  if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
          slice.arithmeticResult, "compute result vector", config))
    return error;
  if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
          slice.storeValue, "stored vector", config))
    return error;
  if (slice.compareOp)
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.compareMask, "compare mask", config))
      return error;
  if (slice.maskedBinaryOp)
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.maskedPassthrough, "masked passthrough vector", config))
      return error;
  if (slice.maccOp)
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.accumulatorValue, "multiply-accumulate accumulator vector",
            config))
      return error;
  return llvm::Error::success();
}

std::string formatRuntimeABIExpectedRoles(
    llvm::ArrayRef<support::RuntimeABIParameterRole> expectedRoles) {
  std::string expected;
  llvm::raw_string_ostream stream(expected);
  llvm::interleave(
      expectedRoles,
      [&](support::RuntimeABIParameterRole role) {
        stream << "'" << support::stringifyRuntimeABIParameterRole(role) << "'";
      },
      [&] { stream << " or "; });
  stream.flush();
  return expected;
}

llvm::Expected<support::RuntimeABIParameter>
getRuntimeABIParameterBindingFromValue(
    mlir::Value value, llvm::StringRef context,
    llvm::ArrayRef<support::RuntimeABIParameterRole> expectedRoles) {
  auto binding = value.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (!binding)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " must be defined by explicit tcrv_rvv.runtime_abi_value before "
        "RVV EmitC route construction");
  if (llvm::Error error =
          verifyRVVRuntimeABIValueRoleOpInterface(binding.getOperation()))
    return std::move(error);

  std::optional<support::RuntimeABIParameterRole> role =
      support::symbolizeRuntimeABIParameterRole(binding.getRole());
  if (!role)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " carries unsupported runtime ABI role '" +
        binding.getRole() + "'");
  if (!llvm::is_contained(expectedRoles, *role)) {
    std::string expectedRolesText =
        formatRuntimeABIExpectedRoles(expectedRoles);
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " must bind runtime ABI role " +
        expectedRolesText + "; got '" + binding.getRole() + "'");
  }

  std::optional<support::RuntimeABIParameterOwnership> ownership =
      support::symbolizeRuntimeABIParameterOwnership(binding.getOwnership());
  if (!ownership)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " carries unsupported runtime ABI ownership '" +
        binding.getOwnership() + "'");

  return support::RuntimeABIParameter(binding.getCName(), binding.getCType(),
                                      *role, *ownership);
}

llvm::Error
assignRVVGenericLoadBinding(RVVSelectedBodyRouteSlice &slice,
                            tcrv::rvv::LoadOp load,
                            const support::RuntimeABIParameter &parameter) {
  if (parameter.role == support::RuntimeABIParameterRole::LHSInputBuffer) {
    if (slice.lhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV EmitC route requires a unique lhs-input-buffer load");
    slice.lhsGenericLoad = load;
    slice.lhsLoadOperation = load.getOperation();
    slice.lhsBuffer = load.getBuffer();
    slice.lhsValue = load.getLoaded();
    slice.lhsABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role == support::RuntimeABIParameterRole::RHSInputBuffer) {
    if (slice.rhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV EmitC route requires a unique rhs-input-buffer load");
    slice.rhsGenericLoad = load;
    slice.rhsLoadOperation = load.getOperation();
    slice.rhsBuffer = load.getBuffer();
    slice.rhsValue = load.getLoaded();
    slice.rhsABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role == support::RuntimeABIParameterRole::OutputBuffer) {
    if (slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV EmitC route requires a unique output-buffer "
          "accumulator load");
    slice.accumulatorLoadOperation = load.getOperation();
    slice.accumulatorBuffer = load.getBuffer();
    slice.accumulatorValue = load.getLoaded();
    slice.accumulatorABI = parameter;
    return llvm::Error::success();
  }

  return makeRVVEmitCRouteProviderError(
      llvm::Twine("unsupported RVV generic load runtime ABI role '") +
      support::stringifyRuntimeABIParameterRole(parameter.role) +
      "' for bounded EmitC route");
}

llvm::Error assignRVVGenericBroadcastBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::BroadcastLoadOp load,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role != support::RuntimeABIParameterRole::RHSInputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV generic broadcast runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) +
        "' for bounded EmitC route");
  if (slice.rhsLoadOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires a unique RHS source load or "
        "broadcast");
  slice.rhsBroadcastLoad = load;
  slice.rhsLoadOperation = load.getOperation();
  slice.rhsBuffer = load.getBuffer();
  slice.rhsValue = load.getBroadcast();
  slice.rhsABI = parameter;
  slice.memoryForm = RVVSelectedBodyMemoryForm::RHSBroadcastLoad;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericScalarSplatBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::SplatOp splat,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role != support::RuntimeABIParameterRole::RHSScalarValue)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV generic scalar splat runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) +
        "' for bounded EmitC route");
  if (slice.rhsLoadOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires a unique RHS source load, "
        "broadcast, or scalar splat");
  slice.rhsScalarSplat = splat;
  slice.rhsLoadOperation = splat.getOperation();
  slice.rhsBuffer = splat.getScalar();
  slice.rhsValue = splat.getBroadcast();
  slice.rhsABI = parameter;
  slice.memoryForm = RVVSelectedBodyMemoryForm::RHSScalarBroadcast;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericStridedLoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::StridedLoadOp load,
    const support::RuntimeABIParameter &bufferParameter,
    const support::RuntimeABIParameter &strideParameter) {
  if (bufferParameter.role == support::RuntimeABIParameterRole::LHSInputBuffer) {
    if (strideParameter.role !=
        support::RuntimeABIParameterRole::LHSInputStride)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided lhs load requires "
          "lhs-input-stride runtime ABI value");
    if (slice.lhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided route requires a unique lhs load");
    slice.lhsStridedLoad = load;
    slice.lhsLoadOperation = load.getOperation();
    slice.lhsBuffer = load.getBuffer();
    slice.lhsStride = load.getStride();
    slice.lhsValue = load.getLoaded();
    slice.lhsABI = bufferParameter;
    slice.lhsStrideABI = strideParameter;
    return llvm::Error::success();
  }

  if (bufferParameter.role == support::RuntimeABIParameterRole::RHSInputBuffer) {
    if (strideParameter.role !=
        support::RuntimeABIParameterRole::RHSInputStride)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided rhs load requires "
          "rhs-input-stride runtime ABI value");
    if (slice.rhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided route requires a unique rhs load");
    slice.rhsStridedLoad = load;
    slice.rhsLoadOperation = load.getOperation();
    slice.rhsBuffer = load.getBuffer();
    slice.rhsStride = load.getStride();
    slice.rhsValue = load.getLoaded();
    slice.rhsABI = bufferParameter;
    slice.rhsStrideABI = strideParameter;
    slice.memoryForm = RVVSelectedBodyMemoryForm::StridedLoadStore;
    return llvm::Error::success();
  }

  return makeRVVEmitCRouteProviderError(
      llvm::Twine("unsupported RVV strided load buffer runtime ABI role '") +
      support::stringifyRuntimeABIParameterRole(bufferParameter.role) + "'");
}

llvm::Error assignRVVGenericStridedStoreBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::StridedStoreOp store,
    const support::RuntimeABIParameter &bufferParameter,
    const support::RuntimeABIParameter &strideParameter) {
  if (bufferParameter.role != support::RuntimeABIParameterRole::OutputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV strided store buffer runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(bufferParameter.role) + "'");
  if (strideParameter.role != support::RuntimeABIParameterRole::OutputStride)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided store requires output-stride runtime ABI "
        "value");
  slice.stridedStore = store;
  slice.storeOperation = store.getOperation();
  slice.outBuffer = store.getBuffer();
  slice.storeValue = store.getValue();
  slice.outStride = store.getStride();
  slice.outABI = bufferParameter;
  slice.outStrideABI = strideParameter;
  slice.memoryForm = RVVSelectedBodyMemoryForm::StridedLoadStore;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericIndexLoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::IndexLoadOp load,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role != support::RuntimeABIParameterRole::IndexInputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV index load runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) + "'");
  if (slice.indexLoadOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route requires a unique "
        "tcrv_rvv.index_load op");
  if (static_cast<std::int64_t>(load.getIndexEew()) != 32)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route supports only index_eew 32");

  slice.indexLoad = load;
  slice.indexLoadOperation = load.getOperation();
  slice.indexBuffer = load.getIndex();
  slice.indexValue = load.getLoaded();
  slice.indexABI = parameter;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericIndexedLoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::IndexedLoadOp load,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role != support::RuntimeABIParameterRole::LHSInputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV indexed data load runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) + "'");
  if (slice.lhsLoadOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route requires a unique data "
        "source tcrv_rvv.indexed_load op");
  if (static_cast<std::int64_t>(load.getIndexEew()) != 32)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route supports only index_eew 32");
  if (load.getOffsetUnit() != kRVVIndexedGatherOffsetUnit)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route supports only offset_unit "
        "'element'");

  slice.indexedLoad = load;
  slice.indexedLoadOperation = load.getOperation();
  slice.lhsLoadOperation = load.getOperation();
  slice.lhsBuffer = load.getData();
  slice.indexedDataBuffer = load.getData();
  slice.lhsValue = load.getLoaded();
  slice.lhsABI = parameter;
  slice.memoryForm = RVVSelectedBodyMemoryForm::IndexedLoadUnitStore;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericIndexedStoreBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::IndexedStoreOp store,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role != support::RuntimeABIParameterRole::OutputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV indexed data store runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) + "'");
  if (slice.indexedStoreOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed scatter route requires a unique "
        "tcrv_rvv.indexed_store op");
  if (static_cast<std::int64_t>(store.getIndexEew()) != 32)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed scatter route supports only index_eew "
        "32");
  if (store.getOffsetUnit() != kRVVIndexedGatherOffsetUnit)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed scatter route supports only offset_unit "
        "'element'");

  slice.indexedStore = store;
  slice.indexedStoreOperation = store.getOperation();
  slice.storeOperation = store.getOperation();
  slice.outBuffer = store.getDestination();
  slice.indexedDestinationBuffer = store.getDestination();
  slice.storeValue = store.getValue();
  slice.outABI = parameter;
  slice.memoryForm = RVVSelectedBodyMemoryForm::UnitLoadIndexedStore;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericMaskLoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::MaskLoadOp load,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role != support::RuntimeABIParameterRole::MaskInputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV mask load runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) + "'");
  if (slice.maskLoadOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked memory route requires a unique "
        "tcrv_rvv.mask_load op");
  if (load.getMaskRole() != kRVVMaskRole)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked memory route requires mask_role "
        "'predicate-mask-input-buffer'");
  if (load.getMaskMemoryForm() != kRVVMaskMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked memory route requires mask_memory_form "
        "'unit-stride-mask-load'");

  slice.maskLoad = load;
  slice.maskLoadOperation = load.getOperation();
  slice.maskBuffer = load.getMask();
  slice.maskValue = load.getLoaded();
  slice.maskABI = parameter;
  return llvm::Error::success();
}

llvm::Error validateRVVSelectedBodyRuntimeABIParameters(
    RVVSelectedBodyRouteSlice &slice,
    const RVVSelectedBodyConstructionRoute &constructionRoute,
    const RVVSelectedBodyConfigProfile &configProfile,
    const support::RuntimeABIParameter &runtimeElementCountABI,
    const support::RuntimeABIParameter &outABI) {
  slice.runtimeElementCountABI = runtimeElementCountABI;
  slice.outABI = outABI;

  llvm::SmallVector<support::RuntimeABIParameter, 7> ordered;
  ordered.push_back(slice.lhsABI);
  if (slice.memoryForm == RVVSelectedBodyMemoryForm::IndexedLoadUnitStore ||
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitLoadIndexedStore)
    ordered.push_back(slice.indexABI);
  if (slice.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitLoadStore)
    ordered.push_back(slice.maskABI);
  if (slice.memoryForm != RVVSelectedBodyMemoryForm::UnitStrideConversion &&
      slice.memoryForm != RVVSelectedBodyMemoryForm::StridedLoadUnitStore &&
      slice.memoryForm != RVVSelectedBodyMemoryForm::IndexedLoadUnitStore &&
      slice.memoryForm != RVVSelectedBodyMemoryForm::UnitLoadIndexedStore &&
      slice.memoryForm != RVVSelectedBodyMemoryForm::MaskedUnitLoadStore)
    ordered.push_back(slice.rhsABI);
  ordered.push_back(slice.outABI);
  ordered.push_back(slice.runtimeElementCountABI);
  if (slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore) {
    ordered.push_back(slice.lhsStrideABI);
    ordered.push_back(slice.rhsStrideABI);
    ordered.push_back(slice.outStrideABI);
  } else if (slice.memoryForm ==
             RVVSelectedBodyMemoryForm::StridedLoadUnitStore) {
    ordered.push_back(slice.lhsStrideABI);
  } else if (slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::RHSScalarBroadcast &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::UnitStrideConversion &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::StridedLoadUnitStore &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::IndexedLoadUnitStore &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::UnitLoadIndexedStore &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::MaskedUnitLoadStore) {
    support::FiniteBinaryRuntimeABIContract contract(
        support::FiniteBinaryRuntimeABIContractSpec{
            constructionRoute.runtimeABIContractName,
            configProfile.constInputPointerCType,
            configProfile.outputPointerCType});
    llvm::Expected<support::FiniteBinaryCallableRuntimeABIParameterBindings>
        bindings = support::bindFiniteBinaryCallableRuntimeABIParametersByRole(
            ordered, "RVV selected-body explicit runtime ABI values",
            contract);
    if (!bindings)
      return bindings.takeError();
  }

  if (llvm::Error error = tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
          ordered, "selected RVV EmitC route explicit runtime ABI values"))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));

  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyOperation(RVVSelectedBodyRouteSlice &slice,
                                           mlir::Operation *op,
                                           RVVSelectedBodyOperationKind kind,
                                           mlir::Value lhs, mlir::Value rhs,
                                           mlir::Value result) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one supported "
        "generic tcrv_rvv.binary op");
  slice.arithmeticOp = op;
  slice.arithmeticKind = kind;
  slice.arithmeticLhs = lhs;
  slice.arithmeticRhs = rhs;
  slice.arithmeticResult = result;
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyCompare(RVVSelectedBodyRouteSlice &slice,
                                         tcrv::rvv::CompareOp compare) {
  if (slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one generic "
        "tcrv_rvv.compare op for mask-producing routes");
  if (compare.getKind() != "eq")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.compare kind '") +
        compare.getKind() + "' for bounded RVV EmitC route");
  slice.compareOp = compare;
  slice.compareLhs = compare.getLhs();
  slice.compareRhs = compare.getRhs();
  slice.compareMask = compare.getMask();
  return llvm::Error::success();
}

llvm::Error
recordRVVSelectedBodyMaskedBinary(RVVSelectedBodyRouteSlice &slice,
                                  tcrv::rvv::MaskedBinaryOp maskedBinary) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  if (maskedBinary.getKind() != "add")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_binary kind '") +
        maskedBinary.getKind() + "' for bounded RVV masked route");
  slice.maskedBinaryOp = maskedBinary;
  slice.arithmeticOp = maskedBinary.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::MaskedAdd;
  slice.compareMask = maskedBinary.getMask();
  slice.maskedPassthrough = maskedBinary.getPassthrough();
  slice.arithmeticLhs = maskedBinary.getLhs();
  slice.arithmeticRhs = maskedBinary.getRhs();
  slice.arithmeticResult = maskedBinary.getResult();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodySelect(RVVSelectedBodyRouteSlice &slice,
                                        tcrv::rvv::SelectOp select) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  slice.selectOp = select;
  slice.arithmeticOp = select.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::CmpSelect;
  slice.arithmeticLhs = select.getTrueValue();
  slice.arithmeticRhs = select.getFalseValue();
  slice.arithmeticResult = select.getSelected();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyReduction(RVVSelectedBodyRouteSlice &slice,
                                           tcrv::rvv::ReduceOp reduce) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  if (reduce.getKind() != "add")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.reduce kind '") +
        reduce.getKind() + "' for bounded RVV reduction route");
  std::optional<llvm::StringRef> accumulatorLayout =
      reduce.getAccumulatorLayout();
  if (!accumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV reduction route requires tcrv_rvv.reduce to carry "
        "accumulator_layout 'rhs-vector-seed-lane0-per-vl-chunk'");
  if (*accumulatorLayout != kRVVReductionAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.reduce accumulator_layout '") +
        *accumulatorLayout + "' for bounded RVV reduction route");
  std::optional<llvm::StringRef> resultLayout = reduce.getResultLayout();
  if (!resultLayout)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV reduction route requires tcrv_rvv.reduce to carry "
        "result_layout 'store-reduction-lane0-to-output-chunk-base'");
  if (*resultLayout != kRVVReductionResultLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.reduce result_layout '") +
        *resultLayout + "' for bounded RVV reduction route");
  slice.reduceOp = reduce;
  slice.arithmeticOp = reduce.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::ReduceAdd;
  slice.arithmeticLhs = reduce.getInput();
  slice.arithmeticRhs = reduce.getAccumulator();
  slice.arithmeticResult = reduce.getResult();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyMAcc(RVVSelectedBodyRouteSlice &slice,
                                      tcrv::rvv::MAccOp macc) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  if (macc.getKind() != "add")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.macc kind '") +
        macc.getKind() + "' for bounded RVV multiply-accumulate route");
  std::optional<llvm::StringRef> accumulatorLayout =
      macc.getAccumulatorLayout();
  if (!accumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV multiply-accumulate route requires tcrv_rvv.macc to "
        "carry accumulator_layout 'output-buffer-vector-accumulator-input'");
  if (*accumulatorLayout != kRVVMAccAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(
            "unsupported generic tcrv_rvv.macc accumulator_layout '") +
        *accumulatorLayout + "' for bounded RVV multiply-accumulate route");
  std::optional<llvm::StringRef> resultLayout = macc.getResultLayout();
  if (!resultLayout)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV multiply-accumulate route requires tcrv_rvv.macc to "
        "carry result_layout "
        "'store-multiply-accumulate-result-to-output-buffer'");
  if (*resultLayout != kRVVMAccResultLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.macc result_layout '") +
        *resultLayout + "' for bounded RVV multiply-accumulate route");
  slice.maccOp = macc;
  slice.arithmeticOp = macc.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::MAccAdd;
  slice.arithmeticLhs = macc.getLhs();
  slice.arithmeticRhs = macc.getRhs();
  slice.arithmeticAccumulator = macc.getAccumulator();
  slice.arithmeticResult = macc.getResult();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyWideningConvert(
    RVVSelectedBodyRouteSlice &slice,
    tcrv::rvv::WideningConvertOp conversion) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  if (conversion.getKind() != "widen_i32_to_i64")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.widening_convert kind '") +
        conversion.getKind() + "' for bounded RVV widening conversion route");
  slice.wideningConvertOp = conversion;
  slice.arithmeticOp = conversion.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::WidenI32ToI64;
  slice.conversionSource = conversion.getSource();
  slice.arithmeticLhs = conversion.getSource();
  slice.arithmeticResult = conversion.getResult();
  slice.memoryForm = RVVSelectedBodyMemoryForm::UnitStrideConversion;
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyMove(RVVSelectedBodyRouteSlice &slice,
                                      tcrv::rvv::MoveOp move) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute or "
        "movement op");
  if (move.getKind() != "copy")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.move kind '") +
        move.getKind() + "' for bounded RVV strided memory movement route");
  slice.moveOp = move;
  slice.arithmeticOp = move.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::StridedLoadUnitStore;
  slice.arithmeticLhs = move.getSource();
  slice.arithmeticResult = move.getResult();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyMaskedMove(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::MaskedMoveOp move) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute or "
        "movement op");
  if (move.getKind() != "active-source-preserve-old-destination")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_move kind '") +
        move.getKind() + "' for bounded RVV masked memory route");
  slice.maskedMoveOp = move;
  slice.arithmeticOp = move.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::MaskedUnitLoadStore;
  slice.maskValue = move.getMask();
  slice.maskedActiveValue = move.getActiveValue();
  slice.maskedInactivePassthrough = move.getInactivePassthrough();
  slice.arithmeticLhs = move.getActiveValue();
  slice.arithmeticRhs = move.getInactivePassthrough();
  slice.arithmeticResult = move.getResult();
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyOperationKind>
parseRVVSelectedBodyBinaryKind(llvm::StringRef kind) {
  if (kind == "add")
    return RVVSelectedBodyOperationKind::Add;
  if (kind == "sub")
    return RVVSelectedBodyOperationKind::Sub;
  if (kind == "mul")
    return RVVSelectedBodyOperationKind::Mul;
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("unsupported generic tcrv_rvv.binary kind '") + kind +
      "' for bounded RVV EmitC route");
}

llvm::Expected<RVVSelectedBodyRouteSlice>
collectRVVSelectedBodyRouteSlice(tcrv::exec::VariantOp variant) {
  llvm::SmallVector<tcrv::rvv::SetVLOp, 2> setvls;
  llvm::SmallVector<tcrv::rvv::WithVLOp, 2> withVLs;
  unsigned rvvOpCount = 0;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    ++rvvOpCount;
    if (auto setvl = llvm::dyn_cast<tcrv::rvv::SetVLOp>(op))
      setvls.push_back(setvl);
    if (auto withVL = llvm::dyn_cast<tcrv::rvv::WithVLOp>(op))
      withVLs.push_back(withVL);
  });

  if (setvls.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one tcrv_rvv.setvl op");
  if (withVLs.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one tcrv_rvv.with_vl op");

  RVVSelectedBodyRouteSlice slice;
  slice.setvl = setvls.front();
  slice.withVL = withVLs.front();

  tcrv::rvv::RVVConfigContractDiagnostic configDiagnostic =
      tcrv::rvv::validateRVVSelectedBodyConfigVLStructure(slice.setvl,
                                                          slice.withVL);
  if (!configDiagnostic.ok)
    return makeRVVEmitCRouteProviderError(configDiagnostic.message);

  llvm::Expected<support::RuntimeABIParameter> runtimeElementCountABI =
      getRuntimeABIParameterBindingFromValue(
          slice.setvl.getAvl(), "tcrv_rvv.setvl AVL operand",
          {support::RuntimeABIParameterRole::RuntimeElementCount});
  if (!runtimeElementCountABI)
    return runtimeElementCountABI.takeError();

  llvm::SmallVector<tcrv::rvv::LoadOp, 2> genericLoads;
  llvm::SmallVector<tcrv::rvv::StridedLoadOp, 2> genericStridedLoads;
  llvm::SmallVector<tcrv::rvv::IndexLoadOp, 1> genericIndexLoads;
  llvm::SmallVector<tcrv::rvv::IndexedLoadOp, 1> genericIndexedLoads;
  llvm::SmallVector<tcrv::rvv::IndexedStoreOp, 1> genericIndexedStores;
  llvm::SmallVector<tcrv::rvv::MaskLoadOp, 1> genericMaskLoads;
  llvm::SmallVector<tcrv::rvv::BroadcastLoadOp, 1> genericBroadcastLoads;
  llvm::SmallVector<tcrv::rvv::SplatOp, 1> genericScalarSplats;
  unsigned storeCount = 0;
  unsigned stridedStoreCount = 0;
  for (mlir::Operation &op : slice.withVL.getBody().front()) {
    if (auto load = llvm::dyn_cast<tcrv::rvv::LoadOp>(op)) {
      genericLoads.push_back(load);
      continue;
    }
    if (auto stridedLoad = llvm::dyn_cast<tcrv::rvv::StridedLoadOp>(op)) {
      genericStridedLoads.push_back(stridedLoad);
      continue;
    }
    if (auto indexLoad = llvm::dyn_cast<tcrv::rvv::IndexLoadOp>(op)) {
      genericIndexLoads.push_back(indexLoad);
      continue;
    }
    if (auto indexedLoad = llvm::dyn_cast<tcrv::rvv::IndexedLoadOp>(op)) {
      genericIndexedLoads.push_back(indexedLoad);
      continue;
    }
    if (auto indexedStore = llvm::dyn_cast<tcrv::rvv::IndexedStoreOp>(op)) {
      genericIndexedStores.push_back(indexedStore);
      continue;
    }
    if (auto maskLoad = llvm::dyn_cast<tcrv::rvv::MaskLoadOp>(op)) {
      genericMaskLoads.push_back(maskLoad);
      continue;
    }
    if (auto broadcast = llvm::dyn_cast<tcrv::rvv::BroadcastLoadOp>(op)) {
      genericBroadcastLoads.push_back(broadcast);
      continue;
    }
    if (auto splat = llvm::dyn_cast<tcrv::rvv::SplatOp>(op)) {
      genericScalarSplats.push_back(splat);
      continue;
    }
    if (auto binary = llvm::dyn_cast<tcrv::rvv::BinaryOp>(op)) {
      llvm::Expected<RVVSelectedBodyOperationKind> kind =
          parseRVVSelectedBodyBinaryKind(binary.getKind());
      if (!kind)
        return kind.takeError();
      if (llvm::Error error = recordRVVSelectedBodyOperation(
              slice, binary.getOperation(), *kind, binary.getLhs(),
              binary.getRhs(), binary.getResult()))
        return std::move(error);
      continue;
    }
    if (auto compare = llvm::dyn_cast<tcrv::rvv::CompareOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodyCompare(slice, compare))
        return std::move(error);
      continue;
    }
    if (auto maskedBinary = llvm::dyn_cast<tcrv::rvv::MaskedBinaryOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyMaskedBinary(slice, maskedBinary))
        return std::move(error);
      continue;
    }
    if (auto select = llvm::dyn_cast<tcrv::rvv::SelectOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodySelect(slice, select))
        return std::move(error);
      continue;
    }
    if (auto reduce = llvm::dyn_cast<tcrv::rvv::ReduceOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodyReduction(slice, reduce))
        return std::move(error);
      continue;
    }
    if (auto macc = llvm::dyn_cast<tcrv::rvv::MAccOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodyMAcc(slice, macc))
        return std::move(error);
      continue;
    }
    if (auto conversion = llvm::dyn_cast<tcrv::rvv::WideningConvertOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyWideningConvert(slice, conversion))
        return std::move(error);
      continue;
    }
    if (auto move = llvm::dyn_cast<tcrv::rvv::MoveOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodyMove(slice, move))
        return std::move(error);
      continue;
    }
    if (auto maskedMove = llvm::dyn_cast<tcrv::rvv::MaskedMoveOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyMaskedMove(slice, maskedMove))
        return std::move(error);
      continue;
    }
    if (auto store = llvm::dyn_cast<tcrv::rvv::StoreOp>(op)) {
      slice.genericStore = store;
      ++storeCount;
      continue;
    }
    if (auto stridedStore = llvm::dyn_cast<tcrv::rvv::StridedStoreOp>(op)) {
      slice.stridedStore = stridedStore;
      ++stridedStoreCount;
      continue;
    }
    if (op.getName().getStringRef().starts_with("tcrv_rvv.i32_"))
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("legacy selected-body op '") +
          op.getName().getStringRef() +
          "' is fail-closed during RVV Stage1; Stage2 routes must use generic "
          "tcrv_rvv.load, tcrv_rvv.broadcast_load, "
          "tcrv_rvv.splat, tcrv_rvv.strided_load, tcrv_rvv.binary, "
          "tcrv_rvv.index_load, tcrv_rvv.indexed_load, "
          "tcrv_rvv.indexed_store, tcrv_rvv.mask_load, tcrv_rvv.compare, "
          "tcrv_rvv.masked_binary, tcrv_rvv.select, tcrv_rvv.reduce, "
          "tcrv_rvv.macc, tcrv_rvv.widening_convert, tcrv_rvv.move, "
          "tcrv_rvv.masked_move, tcrv_rvv.store, and "
          "tcrv_rvv.strided_store body structure");
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("bounded RVV EmitC route does not support op '") +
        op.getName().getStringRef() +
        "' inside tcrv_rvv.with_vl; expected generic load, broadcast_load, "
        "splat, strided_load, index_load, indexed_load, indexed_store, "
        "mask_load, binary, compare, masked_binary, select, reduce, macc, "
        "widening_convert, move, masked_move, store, and strided_store only");
  }

  const bool hasIndexedMemory =
      !genericIndexLoads.empty() || !genericIndexedLoads.empty() ||
      !genericIndexedStores.empty();
  const bool hasMaskedMemory =
      !genericMaskLoads.empty() || static_cast<bool>(slice.maskedMoveOp);
  if (hasIndexedMemory) {
    if (!slice.moveOp)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed memory route requires exactly one "
          "tcrv_rvv.move {kind = \"copy\"} movement op");
    slice.arithmeticKind =
        genericIndexedStores.empty()
            ? RVVSelectedBodyOperationKind::IndexedGatherUnitStore
            : RVVSelectedBodyOperationKind::IndexedScatterUnitLoad;
  }

  const bool isCompareSelect =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::CmpSelect;
  const bool isReduction =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::ReduceAdd;
  const bool isMaskedAdd =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::MaskedAdd;
  const bool isMAccAdd =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::MAccAdd;
  const bool isStridedLoadUnitStore =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::StridedLoadUnitStore;
  const bool isIndexedGatherUnitStore =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::IndexedGatherUnitStore;
  const bool isIndexedScatterUnitLoad =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::IndexedScatterUnitLoad;
  const bool isMaskedUnitLoadStore =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::MaskedUnitLoadStore;
  const bool isWidenI32ToI64 =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::WidenI32ToI64;
  const bool hasStridedMemory =
      !genericStridedLoads.empty() || static_cast<bool>(slice.stridedStore);
  const bool hasScalarBroadcast = !genericScalarSplats.empty();
  if (hasIndexedMemory && isIndexedGatherUnitStore &&
      (!genericStridedLoads.empty() || stridedStoreCount != 0 ||
       !genericLoads.empty() || !genericBroadcastLoads.empty() ||
       hasScalarBroadcast))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route cannot mix indexed memory "
        "ops with unit-stride input loads, broadcast, scalar splat, strided "
        "loads, or strided stores");
  if (hasIndexedMemory && isIndexedScatterUnitLoad &&
      (!genericStridedLoads.empty() || stridedStoreCount != 0 ||
       !genericBroadcastLoads.empty() || hasScalarBroadcast || storeCount != 0))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed scatter route cannot mix indexed store "
        "memory ops with broadcast, scalar splat, unit-stride stores, "
        "strided loads, or strided stores");
  if (genericScalarSplats.size() > 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires at most one "
        "tcrv_rvv.splat op");
  if (genericBroadcastLoads.size() > 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires at most one "
        "tcrv_rvv.broadcast_load op");
  const bool hasRHSBroadcastLike =
      !genericBroadcastLoads.empty() || hasScalarBroadcast;
  if (!genericBroadcastLoads.empty() && hasScalarBroadcast)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route cannot mix RHS buffer broadcast and "
        "RHS scalar splat in one slice");
  if (hasStridedMemory && !isStridedLoadUnitStore &&
      (!genericBroadcastLoads.empty() || hasScalarBroadcast ||
       !genericLoads.empty() || slice.genericStore))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided route cannot mix strided memory ops with "
        "unit-stride load/store, broadcast, or scalar-splat memory forms");
  if (hasStridedMemory && isStridedLoadUnitStore &&
      (!genericBroadcastLoads.empty() || hasScalarBroadcast ||
       !genericLoads.empty()))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided-load to unit-stride-store route cannot "
        "mix strided source memory with unit-stride input loads, broadcast, or "
        "scalar-splat memory forms");
  if (hasStridedMemory && isMAccAdd)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided route supports only add in this slice, "
        "not multiply-accumulate");
  if (hasStridedMemory && isWidenI32ToI64)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV widening conversion route requires unit-stride "
        "source load and destination store");
  if (hasStridedMemory && !isStridedLoadUnitStore &&
      (!slice.arithmeticOp ||
                           slice.arithmeticKind !=
                               RVVSelectedBodyOperationKind::Add))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided route requires exactly one "
        "tcrv_rvv.binary {kind = \"add\"} compute op");
  if (hasStridedMemory && !isStridedLoadUnitStore &&
      genericStridedLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided route requires exactly two "
        "tcrv_rvv.strided_load ops for lhs and rhs");
  if (hasStridedMemory && isStridedLoadUnitStore &&
      genericStridedLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided-load to unit-stride-store route requires "
        "exactly one tcrv_rvv.strided_load source op");
  if (hasStridedMemory && !isStridedLoadUnitStore && stridedStoreCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided route requires exactly one "
        "tcrv_rvv.strided_store op");
  if (hasStridedMemory && isStridedLoadUnitStore && stridedStoreCount != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided-load to unit-stride-store route must use "
        "unit-stride tcrv_rvv.store, not tcrv_rvv.strided_store");
  if (hasStridedMemory && isStridedLoadUnitStore && storeCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided-load to unit-stride-store route requires "
        "exactly one unit-stride tcrv_rvv.store op");
  if (hasIndexedMemory && genericIndexLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed memory route requires exactly one "
        "tcrv_rvv.index_load op");
  if (isIndexedGatherUnitStore && genericIndexedLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route requires exactly one "
        "tcrv_rvv.indexed_load op");
  if (isIndexedGatherUnitStore && genericIndexedStores.size() != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route must use unit-stride "
        "tcrv_rvv.store, not tcrv_rvv.indexed_store");
  if (isIndexedGatherUnitStore && storeCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route requires exactly one "
        "unit-stride tcrv_rvv.store op");
  if (isIndexedScatterUnitLoad && genericLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed scatter route requires exactly one "
        "unit-stride tcrv_rvv.load source op");
  if (isIndexedScatterUnitLoad && genericIndexedLoads.size() != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed scatter route must use unit-stride "
        "tcrv_rvv.load source, not tcrv_rvv.indexed_load");
  if (isIndexedScatterUnitLoad && genericIndexedStores.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed scatter route requires exactly one "
        "tcrv_rvv.indexed_store op");
  if (hasIndexedMemory && (isMAccAdd || isWidenI32ToI64 || isCompareSelect ||
                           isMaskedAdd || isReduction))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed memory route supports only "
        "index_load/indexed_load_or_unit_load/move/store_or_indexed_store "
        "memory movement in this slice");
  if (hasMaskedMemory &&
      (!isMaskedUnitLoadStore || hasIndexedMemory || hasStridedMemory ||
       !genericBroadcastLoads.empty() || hasScalarBroadcast ||
       slice.compareOp || slice.maskedBinaryOp || slice.selectOp ||
       slice.reduceOp || slice.maccOp || isWidenI32ToI64))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked memory route supports only "
        "unit-stride load/mask_load/old-destination-load/masked_move/store "
        "memory movement in this slice");
  if (isMaskedUnitLoadStore && genericMaskLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked memory route requires exactly one "
        "tcrv_rvv.mask_load op");
  if (isMaskedUnitLoadStore && genericLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked memory route requires exactly two "
        "tcrv_rvv.load ops for source and old destination");
  if (isMaskedUnitLoadStore && storeCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked memory route requires exactly one "
        "unit-stride tcrv_rvv.store op");
  if (isMAccAdd && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV multiply-accumulate route requires explicit "
        "vector lhs, rhs, and accumulator loads; broadcast/splat macc is not "
        "in this bounded slice");
  if (isWidenI32ToI64 && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV widening conversion route does not consume an "
        "RHS broadcast or scalar splat");
  if (isMAccAdd && genericLoads.size() != 3)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV multiply-accumulate route requires exactly three "
        "tcrv_rvv.load ops for lhs, rhs, and output-buffer accumulator");
  if (hasScalarBroadcast && (!slice.arithmeticOp ||
                             slice.arithmeticKind !=
                                 RVVSelectedBodyOperationKind::Add))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV scalar-broadcast route currently requires "
        "exactly one tcrv_rvv.binary {kind = \"add\"} compute op");
  if (isWidenI32ToI64 && genericLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV widening conversion route requires exactly one "
        "tcrv_rvv.load source op");
  if (isWidenI32ToI64 && (slice.compareOp || slice.maskedBinaryOp ||
                          slice.selectOp || slice.reduceOp || slice.maccOp))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV widening conversion route cannot mix "
        "compare/select/masked/reduce/macc compute ops");
  if (!hasStridedMemory && !hasIndexedMemory && !isMAccAdd &&
      !isMaskedUnitLoadStore &&
      !hasRHSBroadcastLike &&
      !isWidenI32ToI64 &&
      genericLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV vector-load route requires exactly two "
        "tcrv_rvv.load ops");
  if (!hasStridedMemory && !hasIndexedMemory && !isMAccAdd &&
      !isMaskedUnitLoadStore &&
      !genericBroadcastLoads.empty() && genericLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV broadcast route requires exactly one "
        "tcrv_rvv.load op and one tcrv_rvv.broadcast_load op");
  if (!hasStridedMemory && !hasIndexedMemory && !isMAccAdd &&
      !isMaskedUnitLoadStore &&
      hasScalarBroadcast && genericLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV scalar-broadcast route requires exactly one "
        "tcrv_rvv.load op and one tcrv_rvv.splat op");
  if (!hasStridedMemory && !hasIndexedMemory && !slice.genericStore)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires exactly one "
        "tcrv_rvv.store op");
  if (!slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires exactly one supported "
        "tcrv_rvv.binary, tcrv_rvv.select, tcrv_rvv.reduce, tcrv_rvv.macc, "
        "tcrv_rvv.widening_convert, tcrv_rvv.move, or "
        "tcrv_rvv.masked_move op");
  if (!hasStridedMemory && !hasIndexedMemory && storeCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires exactly one tcrv_rvv.store "
        "op");
  if (hasStridedMemory && !isStridedLoadUnitStore && storeCount != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided route must use tcrv_rvv.strided_store "
        "instead of tcrv_rvv.store");
  if ((isCompareSelect || isMaskedAdd) && !slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV mask-consuming route requires one "
        "tcrv_rvv.compare op before the mask-consuming compute op");
  if (!isCompareSelect && !isMaskedAdd && slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV non-mask route does not support a standalone "
        "tcrv_rvv.compare op");
  if (isCompareSelect && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV compare/select route requires an explicit RHS "
        "vector load; broadcast/splat compare/select is not in this bounded "
        "slice");
  if (isMaskedAdd && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked add route requires an explicit RHS vector "
        "load; broadcast/splat masked add is not in this bounded slice");
  if (isReduction && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV reduction route requires explicit vector input "
        "and accumulator loads; broadcast/splat reduction is not in this "
        "bounded slice");
  const unsigned expectedRVVOps =
      isWidenI32ToI64
          ? 8
      : isMaskedUnitLoadStore
          ? 11
          : (isIndexedGatherUnitStore || isIndexedScatterUnitLoad
                 ? 10
                 : (isStridedLoadUnitStore
                        ? 9
          : (hasStridedMemory
                 ? 13
                 : ((isCompareSelect || isMaskedAdd || isMAccAdd) ? 11 : 10))));
  if (rvvOpCount != expectedRVVOps)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route supports only runtime_abi_value/"
        "runtime_abi_value/runtime_abi_value plus optional runtime_abi_value "
        "and optional strided runtime_abi_value/runtime_abi_value/"
        "runtime_abi_value, "
        "setvl/with_vl, and generic load/broadcast_load/splat/strided_load/"
        "index_load/indexed_load/mask_load/binary/compare/select/"
        "masked_binary/reduce/macc/widening_convert/move/masked_move/store/"
        "strided_store body structure");

  for (tcrv::rvv::LoadOp load : genericLoads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            load.getBuffer(), "tcrv_rvv.load buffer operand",
            {support::RuntimeABIParameterRole::LHSInputBuffer,
             support::RuntimeABIParameterRole::RHSInputBuffer,
             support::RuntimeABIParameterRole::OutputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error = assignRVVGenericLoadBinding(slice, load, *parameter))
      return error;
  }
  for (tcrv::rvv::BroadcastLoadOp broadcast : genericBroadcastLoads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            broadcast.getBuffer(), "tcrv_rvv.broadcast_load buffer operand",
            {support::RuntimeABIParameterRole::RHSInputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericBroadcastBinding(slice, broadcast, *parameter))
      return error;
  }
  for (tcrv::rvv::SplatOp splat : genericScalarSplats) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            splat.getScalar(), "tcrv_rvv.splat scalar operand",
            {support::RuntimeABIParameterRole::RHSScalarValue});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericScalarSplatBinding(slice, splat, *parameter))
      return error;
  }
  for (tcrv::rvv::StridedLoadOp load : genericStridedLoads) {
    llvm::Expected<support::RuntimeABIParameter> bufferParameter =
        getRuntimeABIParameterBindingFromValue(
            load.getBuffer(), "tcrv_rvv.strided_load buffer operand",
            {support::RuntimeABIParameterRole::LHSInputBuffer,
             support::RuntimeABIParameterRole::RHSInputBuffer});
    if (!bufferParameter)
      return bufferParameter.takeError();
    llvm::Expected<support::RuntimeABIParameter> strideParameter =
        getRuntimeABIParameterBindingFromValue(
            load.getStride(), "tcrv_rvv.strided_load stride operand",
            {support::RuntimeABIParameterRole::LHSInputStride,
             support::RuntimeABIParameterRole::RHSInputStride});
    if (!strideParameter)
      return strideParameter.takeError();
    if (llvm::Error error = assignRVVGenericStridedLoadBinding(
            slice, load, *bufferParameter, *strideParameter))
      return error;
  }
  for (tcrv::rvv::IndexLoadOp load : genericIndexLoads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            load.getIndex(), "tcrv_rvv.index_load index operand",
            {support::RuntimeABIParameterRole::IndexInputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericIndexLoadBinding(slice, load, *parameter))
      return error;
  }
  for (tcrv::rvv::IndexedLoadOp load : genericIndexedLoads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            load.getData(), "tcrv_rvv.indexed_load data operand",
            {support::RuntimeABIParameterRole::LHSInputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericIndexedLoadBinding(slice, load, *parameter))
      return error;
  }
  for (tcrv::rvv::IndexedStoreOp store : genericIndexedStores) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            store.getDestination(), "tcrv_rvv.indexed_store destination operand",
            {support::RuntimeABIParameterRole::OutputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericIndexedStoreBinding(slice, store, *parameter))
      return error;
  }
  for (tcrv::rvv::MaskLoadOp load : genericMaskLoads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            load.getMask(), "tcrv_rvv.mask_load mask operand",
            {support::RuntimeABIParameterRole::MaskInputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericMaskLoadBinding(slice, load, *parameter))
      return error;
  }
  if (isIndexedGatherUnitStore) {
    if (slice.indexedLoad.getIndices() != slice.indexValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed gather route requires "
          "tcrv_rvv.indexed_load to consume the index vector produced by "
          "tcrv_rvv.index_load");
    if (slice.indexLoad.getVl() != slice.setvl.getVl() ||
        slice.indexedLoad.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed gather route requires index_load and "
          "indexed_load to consume the selected !tcrv_rvv.vl token");
  }
  if (isIndexedScatterUnitLoad) {
    if (slice.indexedStore.getIndices() != slice.indexValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed scatter route requires "
          "tcrv_rvv.indexed_store to consume the index vector produced by "
          "tcrv_rvv.index_load");
    if (slice.indexLoad.getVl() != slice.setvl.getVl() ||
        slice.indexedStore.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed scatter route requires index_load and "
          "indexed_store to consume the selected !tcrv_rvv.vl token");
  }
  if (isMaskedUnitLoadStore) {
    if (slice.maskedMoveOp.getMask() != slice.maskValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked memory route requires "
          "tcrv_rvv.masked_move to consume the mask produced by "
          "tcrv_rvv.mask_load");
    if (slice.maskLoad.getVl() != slice.setvl.getVl() ||
        slice.maskedMoveOp.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked memory route requires mask_load and "
          "masked_move to consume the selected !tcrv_rvv.vl token");
  }

  if (!slice.lhsLoadOperation ||
      (!isWidenI32ToI64 && !isStridedLoadUnitStore &&
       !isIndexedGatherUnitStore && !isIndexedScatterUnitLoad &&
       !isMaskedUnitLoadStore &&
       !slice.rhsLoadOperation))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires lhs-input-buffer and "
        "rhs-input-buffer or rhs-scalar-value generic load, broadcast, or "
        "scalar-splat dataflow");
  llvm::Expected<support::RuntimeABIParameter> outABI =
      hasStridedMemory && !isStridedLoadUnitStore
          ? getRuntimeABIParameterBindingFromValue(
                slice.stridedStore.getBuffer(),
                "tcrv_rvv.strided_store buffer operand",
                {support::RuntimeABIParameterRole::OutputBuffer})
      : isIndexedScatterUnitLoad
          ? getRuntimeABIParameterBindingFromValue(
                slice.indexedStore.getDestination(),
                "tcrv_rvv.indexed_store destination operand",
                {support::RuntimeABIParameterRole::OutputBuffer})
          : getRuntimeABIParameterBindingFromValue(
                slice.genericStore.getBuffer(), "tcrv_rvv.store buffer operand",
                {support::RuntimeABIParameterRole::OutputBuffer});
  if (!outABI)
    return outABI.takeError();
  if (hasStridedMemory && !isStridedLoadUnitStore) {
    llvm::Expected<support::RuntimeABIParameter> outStrideABI =
        getRuntimeABIParameterBindingFromValue(
            slice.stridedStore.getStride(),
            "tcrv_rvv.strided_store stride operand",
            {support::RuntimeABIParameterRole::OutputStride});
    if (!outStrideABI)
      return outStrideABI.takeError();
    if (llvm::Error error = assignRVVGenericStridedStoreBinding(
            slice, slice.stridedStore, *outABI, *outStrideABI))
      return error;
    slice.arithmeticKind = RVVSelectedBodyOperationKind::StridedAdd;
  } else if (isStridedLoadUnitStore) {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    slice.memoryForm = RVVSelectedBodyMemoryForm::StridedLoadUnitStore;
  } else if (isIndexedGatherUnitStore) {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    slice.memoryForm = RVVSelectedBodyMemoryForm::IndexedLoadUnitStore;
  } else if (isIndexedScatterUnitLoad) {
    slice.storeOperation = slice.indexedStore.getOperation();
    slice.outBuffer = slice.indexedStore.getDestination();
    slice.storeValue = slice.indexedStore.getValue();
    slice.memoryForm = RVVSelectedBodyMemoryForm::UnitLoadIndexedStore;
  } else if (isMaskedUnitLoadStore) {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    slice.memoryForm = RVVSelectedBodyMemoryForm::MaskedUnitLoadStore;
  } else {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    if (hasScalarBroadcast)
      slice.arithmeticKind =
          RVVSelectedBodyOperationKind::ScalarBroadcastAdd;
  }
  llvm::StringRef operationMnemonic =
      stringifyRVVSelectedBodyOperationKind(slice.arithmeticKind);
  slice.runtimeElementCountABI = *runtimeElementCountABI;
  slice.outABI = *outABI;
  if (isCompareSelect) {
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV compare/select route requires tcrv_rvv.compare "
          "to consume lhs/rhs generic load results");
    if (slice.selectOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV compare/select route requires tcrv_rvv.select "
          "to consume the mask produced by tcrv_rvv.compare");
    if (slice.arithmeticLhs != slice.lhsValue ||
        slice.arithmeticRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV compare/select route requires tcrv_rvv.select "
          "to consume lhs as true value and rhs as false value");
  } else if (isMaskedAdd) {
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked add route requires tcrv_rvv.compare to "
          "consume lhs/rhs generic load results");
    if (slice.maskedBinaryOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked add route requires "
          "tcrv_rvv.masked_binary to consume the mask produced by "
          "tcrv_rvv.compare");
    if (slice.maskedPassthrough != slice.lhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked add route requires "
          "tcrv_rvv.masked_binary passthrough to consume the lhs load result");
    if (slice.arithmeticLhs != slice.lhsValue ||
        slice.arithmeticRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked add route requires "
          "tcrv_rvv.masked_binary to consume lhs/rhs generic load results");
  } else if (isMAccAdd) {
    if (!slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV multiply-accumulate route requires one "
          "output-buffer accumulator tcrv_rvv.load");
    if (slice.accumulatorBuffer != slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV multiply-accumulate route requires the "
          "accumulator load to consume the same output buffer as "
          "tcrv_rvv.store");
    if (slice.arithmeticLhs != slice.lhsValue ||
        slice.arithmeticRhs != slice.rhsValue ||
        slice.arithmeticAccumulator != slice.accumulatorValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV multiply-accumulate route requires "
          "tcrv_rvv.macc to consume lhs/rhs generic load results and the "
          "output-buffer accumulator load result");
  } else if (isWidenI32ToI64) {
    if (slice.conversionSource != slice.lhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV widening conversion route requires "
          "tcrv_rvv.widening_convert to consume the lhs source load result");
    if (slice.accumulatorLoadOperation || slice.rhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV widening conversion route does not support RHS "
          "or accumulator loads");
  } else if (isStridedLoadUnitStore) {
    if (slice.arithmeticLhs != slice.lhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided-load to unit-stride-store route "
          "requires tcrv_rvv.move to consume the strided source load result");
    if (slice.rhsLoadOperation || slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided-load to unit-stride-store route does "
          "not support RHS or accumulator loads");
  } else if (isIndexedGatherUnitStore) {
    if (slice.arithmeticLhs != slice.lhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed gather route requires tcrv_rvv.move "
          "to consume the indexed data load result");
    if (slice.rhsLoadOperation || slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed gather route does not support RHS or "
          "accumulator loads");
  } else if (isIndexedScatterUnitLoad) {
    if (slice.arithmeticLhs != slice.lhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed scatter route requires tcrv_rvv.move "
          "to consume the unit-stride source load result");
    if (slice.rhsLoadOperation || slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed scatter route does not support RHS or "
          "accumulator loads");
  } else if (isMaskedUnitLoadStore) {
    if (!slice.maskLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked memory route requires one "
          "tcrv_rvv.mask_load");
    if (!slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked memory route requires an "
          "output-buffer tcrv_rvv.load for old-destination preservation");
    if (slice.accumulatorBuffer != slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked memory route requires the "
          "old-destination load to consume the same output buffer as "
          "tcrv_rvv.store");
    if (slice.maskedActiveValue != slice.lhsValue ||
        slice.maskedInactivePassthrough != slice.accumulatorValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked memory route requires "
          "tcrv_rvv.masked_move to consume source load as active value and "
          "old-destination load as inactive passthrough");
    if (slice.rhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked memory route does not support an RHS "
          "data load");
  } else if (slice.arithmeticLhs != slice.lhsValue ||
             slice.arithmeticRhs != slice.rhsValue) {
    if (slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV non-multiply-accumulate route does not support "
          "an output-buffer accumulator load");
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("bounded generic RVV EmitC route requires selected-body ") +
        operationMnemonic +
        " to consume lhs/rhs generic load or broadcast results");
  }
  if (slice.storeValue != slice.arithmeticResult)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires selected-body store to "
        "consume the selected compute result");

  return slice;
}

struct RVVOrderedRoleOperations {
  llvm::SmallVector<mlir::Operation *, 12> operations;
  llvm::SmallVector<unsigned, 12> constructionOrders;
};

unsigned getRVVCanonicalRoleOrder(RVVSelectedBodyRouteSlice &slice,
                                  mlir::Operation *op) {
  auto getRuntimeABI =
      [](mlir::Value value) -> tcrv::rvv::RuntimeABIValueOp {
    if (!value)
      return nullptr;
    return value.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  };
  auto lhsABI =
      getRuntimeABI(slice.lhsBuffer);
  auto rhsABI =
      getRuntimeABI(slice.rhsBuffer);
  auto indexABI =
      getRuntimeABI(slice.indexBuffer);
  auto maskABI =
      getRuntimeABI(slice.maskBuffer);
  auto outABI =
      getRuntimeABI(slice.outBuffer);
  auto nABI =
      getRuntimeABI(slice.setvl ? slice.setvl.getAvl() : mlir::Value());
  auto lhsStrideABI =
      slice.lhsStride
          ? slice.lhsStride.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>()
          : nullptr;
  auto rhsStrideABI =
      slice.rhsStride
          ? slice.rhsStride.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>()
          : nullptr;
  auto outStrideABI =
      slice.outStride
          ? slice.outStride.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>()
          : nullptr;
  const bool isStrided =
      slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore;
  const bool isStridedLoadUnitStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadUnitStore;
  const bool isIndexedGatherUnitStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::IndexedLoadUnitStore;
  const bool isIndexedScatterUnitLoad =
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitLoadIndexedStore;
  const bool isMaskedUnitLoadStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitLoadStore;
  const bool isConversion =
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitStrideConversion;
  if (lhsABI && op == lhsABI.getOperation())
    return 0;
  if (isMaskedUnitLoadStore) {
    if (maskABI && op == maskABI.getOperation())
      return 1;
    if (outABI && op == outABI.getOperation())
      return 2;
    if (nABI && op == nABI.getOperation())
      return 3;
    if (op == slice.setvl.getOperation())
      return 4;
    if (op == slice.withVL.getOperation())
      return 5;
    if (op == slice.lhsLoadOperation)
      return 6;
    if (op == slice.maskLoadOperation)
      return 7;
    if (op == slice.accumulatorLoadOperation)
      return 8;
    if (op == slice.arithmeticOp)
      return 9;
    if (op == slice.storeOperation)
      return 10;
    return 11;
  }
  if (isIndexedGatherUnitStore) {
    if (indexABI && op == indexABI.getOperation())
      return 1;
    if (outABI && op == outABI.getOperation())
      return 2;
    if (nABI && op == nABI.getOperation())
      return 3;
    if (op == slice.setvl.getOperation())
      return 4;
    if (op == slice.withVL.getOperation())
      return 5;
    if (op == slice.indexLoadOperation)
      return 6;
    if (op == slice.indexedLoadOperation)
      return 7;
    if (op == slice.arithmeticOp)
      return 8;
    if (op == slice.storeOperation)
      return 9;
    return 10;
  }
  if (isIndexedScatterUnitLoad) {
    if (indexABI && op == indexABI.getOperation())
      return 1;
    if (outABI && op == outABI.getOperation())
      return 2;
    if (nABI && op == nABI.getOperation())
      return 3;
    if (op == slice.setvl.getOperation())
      return 4;
    if (op == slice.withVL.getOperation())
      return 5;
    if (op == slice.lhsLoadOperation)
      return 6;
    if (op == slice.indexLoadOperation)
      return 7;
    if (op == slice.arithmeticOp)
      return 8;
    if (op == slice.indexedStoreOperation)
      return 9;
    return 10;
  }
  if (isStridedLoadUnitStore) {
    if (outABI && op == outABI.getOperation())
      return 1;
    if (nABI && op == nABI.getOperation())
      return 2;
    if (lhsStrideABI && op == lhsStrideABI.getOperation())
      return 3;
    if (op == slice.setvl.getOperation())
      return 4;
    if (op == slice.withVL.getOperation())
      return 5;
    if (op == slice.lhsLoadOperation)
      return 6;
    if (op == slice.arithmeticOp)
      return 7;
    if (op == slice.storeOperation)
      return 8;
    return 9;
  }
  if (isConversion) {
    if (outABI && op == outABI.getOperation())
      return 1;
    if (nABI && op == nABI.getOperation())
      return 2;
    if (op == slice.setvl.getOperation())
      return 3;
    if (op == slice.withVL.getOperation())
      return 4;
    if (op == slice.lhsLoadOperation)
      return 5;
    if (op == slice.arithmeticOp)
      return 6;
    if (op == slice.storeOperation)
      return 7;
    return 8;
  }
  if (rhsABI && op == rhsABI.getOperation())
    return 1;
  if (outABI && op == outABI.getOperation())
    return 2;
  if (nABI && op == nABI.getOperation())
    return 3;
  if (isStrided) {
    if (lhsStrideABI && op == lhsStrideABI.getOperation())
      return 4;
    if (rhsStrideABI && op == rhsStrideABI.getOperation())
      return 5;
    if (outStrideABI && op == outStrideABI.getOperation())
      return 6;
  }
  if (op == slice.setvl.getOperation())
    return isStrided ? 7 : 4;
  if (op == slice.withVL.getOperation())
    return isStrided ? 8 : 5;
  if (op == slice.lhsLoadOperation)
    return isStrided ? 9 : 6;
  if (op == slice.rhsLoadOperation)
    return isStrided ? 10 : 7;
  if (isStrided && op == slice.arithmeticOp)
    return 11;
  if (isStrided && op == slice.storeOperation)
    return 12;
  if (slice.accumulatorLoadOperation && op == slice.accumulatorLoadOperation)
    return 8;
  if (slice.compareOp && op == slice.compareOp.getOperation())
    return 8;
  if (op == slice.arithmeticOp)
    return (slice.compareOp || slice.accumulatorLoadOperation) ? 9 : 8;
  if (op == slice.storeOperation)
    return (slice.compareOp || slice.accumulatorLoadOperation) ? 10 : 9;
  return (slice.compareOp || slice.accumulatorLoadOperation) ? 11 : 10;
}

RVVOrderedRoleOperations
collectRVVRoleOperationsInBodyOrder(tcrv::exec::VariantOp variant,
                                    RVVSelectedBodyRouteSlice &slice) {
  RVVOrderedRoleOperations ordered;
  if (!variant || variant.getBody().empty())
    return ordered;

  for (mlir::Operation &op : variant.getBody().front()) {
    if (op.getName().getDialectNamespace() != "tcrv_rvv")
      continue;
    ordered.operations.push_back(&op);
    ordered.constructionOrders.push_back(getRVVCanonicalRoleOrder(slice, &op));
    if (auto withVL = llvm::dyn_cast<tcrv::rvv::WithVLOp>(op))
      for (mlir::Operation &nested : withVL.getBody().front())
        if (nested.getName().getDialectNamespace() == "tcrv_rvv") {
          ordered.operations.push_back(&nested);
          ordered.constructionOrders.push_back(
              getRVVCanonicalRoleOrder(slice, &nested));
        }
  }
  return ordered;
}

llvm::Error verifySelectedRVVRoleSequence(
    RVVSelectedBodyRouteSlice &slice,
    const VariantEmitCLowerableRequest &request,
    const RVVSelectedBodyConstructionRoute &constructionRoute) {
  auto getRuntimeABI =
      [](mlir::Value value) -> tcrv::rvv::RuntimeABIValueOp {
    if (!value)
      return nullptr;
    return value.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  };
  auto lhsABI =
      getRuntimeABI(slice.lhsBuffer);
  auto rhsABI =
      getRuntimeABI(slice.rhsBuffer);
  auto indexABI =
      getRuntimeABI(slice.indexBuffer);
  auto maskABI =
      getRuntimeABI(slice.maskBuffer);
  auto outABI =
      getRuntimeABI(slice.outBuffer);
  auto nABI =
      getRuntimeABI(slice.setvl ? slice.setvl.getAvl() : mlir::Value());
  auto lhsStrideABI =
      slice.lhsStride
          ? slice.lhsStride.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>()
          : nullptr;
  auto rhsStrideABI =
      slice.rhsStride
          ? slice.rhsStride.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>()
          : nullptr;
  auto outStrideABI =
      slice.outStride
          ? slice.outStride.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>()
          : nullptr;
  const bool isStrided =
      slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore;
  const bool isStridedLoadUnitStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadUnitStore;
  const bool isIndexedGatherUnitStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::IndexedLoadUnitStore;
  const bool isIndexedScatterUnitLoad =
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitLoadIndexedStore;
  const bool isMaskedUnitLoadStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitLoadStore;
  const bool isConversion =
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitStrideConversion;
  if (!lhsABI ||
      (!isConversion && !isStridedLoadUnitStore &&
       !isIndexedGatherUnitStore && !isIndexedScatterUnitLoad &&
       !isMaskedUnitLoadStore && !rhsABI) ||
      !outABI ||
      !nABI || (isStrided && (!lhsStrideABI || !rhsStrideABI || !outStrideABI)) ||
      (isStridedLoadUnitStore && !lhsStrideABI) ||
      ((isIndexedGatherUnitStore || isIndexedScatterUnitLoad) && !indexABI) ||
      (isMaskedUnitLoadStore && !maskABI))
    return makeRVVEmitCRouteProviderError(
        "selected RVV construction role sequence requires runtime ABI values "
        "to be explicit tcrv_rvv.runtime_abi_value ops");

  mlir::ArrayAttr requires =
      request.getVariant()->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!requires || requires.empty())
    return makeRVVEmitCRouteProviderError(
        "selected RVV construction role sequence requires non-empty selected "
        "variant requires metadata");

  RVVOrderedRoleOperations ordered =
      collectRVVRoleOperationsInBodyOrder(request.getVariant(), slice);
  return verifyRVVSelectedBodySelectedRoleSequence(
      ordered.operations, ordered.constructionOrders,
      request.getVariant().getSymName(),
      stringifyVariantEmissionRole(request.getRole()),
      constructionRoute.operationMnemonic,
      slice.arithmeticOp->getName().getStringRef(),
      isIndexedGatherUnitStore
          ? slice.indexedLoadOperation->getName().getStringRef()
      : isIndexedScatterUnitLoad
          ? slice.indexedStoreOperation->getName().getStringRef()
      : isMaskedUnitLoadStore
          ? slice.maskLoadOperation->getName().getStringRef()
      : isStridedLoadUnitStore
          ? slice.lhsLoadOperation->getName().getStringRef()
          : (slice.rhsLoadOperation
                 ? slice.rhsLoadOperation->getName().getStringRef()
                 : llvm::StringRef()),
      "selected RVV EmitC route");
}

} // namespace

llvm::Expected<RVVSelectedBodyRouteAnalysis>
analyzeRVVSelectedBodyRoute(const VariantEmitCLowerableRequest &request) {
  if (!request.getVariant())
    return makeRVVEmitCRouteProviderError(
        "EmitC route construction requires a materialized tcrv.exec.variant");
  if (!request.getKernel())
    return makeRVVEmitCRouteProviderError(
        "EmitC route construction requires an enclosing tcrv.exec.kernel");

  if (llvm::Error error = requireRVVVariantLegality(request.getVariant()))
    return std::move(error);
  if (llvm::Error error = verifyRVVConstructionProtocolReady())
    return std::move(error);

  llvm::Expected<RVVSelectedBodyRouteSlice> slice =
      collectRVVSelectedBodyRouteSlice(request.getVariant());
  if (!slice)
    return slice.takeError();

  tcrv::rvv::RVVCompileTimeConfig config =
      tcrv::rvv::getRVVSetVLCompileTimeConfig(slice->setvl);
  if (llvm::Error error =
          validateRVVSelectedBodyTypedConfigFacts(*slice, config))
    return std::move(error);
  const auto &configContract =
      tcrv::rvv::getRVVSelectedBodyConfigVLContract(config.sew, config.lmul);

  RVVSelectedBodyRouteAnalysis analysis;
  analysis.slice = std::move(*slice);
  analysis.description.operation = analysis.slice.arithmeticKind;
  analysis.description.memoryForm = analysis.slice.memoryForm;
  analysis.description.sew = config.sew;
  analysis.description.lmul = config.lmul;
  analysis.description.tailPolicy =
      stringifyRVVTailPolicy(config.policy.getTail());
  analysis.description.maskPolicy =
      stringifyRVVMaskPolicy(config.policy.getMask());
  analysis.description.configContractID = configContract.configContractID;
  analysis.description.runtimeVLContractID = configContract.runtimeVLContractID;
  analysis.description.runtimeAVLASource = configContract.runtimeAVLASource;
  analysis.description.runtimeABIOrder =
      analysis.slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore
      ? kRVVStridedRuntimeABIOrder
      : (analysis.slice.memoryForm ==
                 RVVSelectedBodyMemoryForm::StridedLoadUnitStore
             ? kRVVStridedLoadUnitStoreRuntimeABIOrder
      : (analysis.slice.memoryForm ==
                 RVVSelectedBodyMemoryForm::IndexedLoadUnitStore
             ? kRVVIndexedGatherRuntimeABIOrder
      : (analysis.slice.memoryForm ==
                 RVVSelectedBodyMemoryForm::UnitLoadIndexedStore
             ? kRVVIndexedScatterRuntimeABIOrder
      : (analysis.slice.memoryForm ==
                 RVVSelectedBodyMemoryForm::MaskedUnitLoadStore
             ? kRVVMaskedMemoryRuntimeABIOrder
      : (analysis.slice.memoryForm ==
                 RVVSelectedBodyMemoryForm::UnitStrideConversion
             ? kRVVWideningConversionRuntimeABIOrder
             : (analysis.slice.memoryForm ==
                 RVVSelectedBodyMemoryForm::RHSScalarBroadcast
             ? kRVVScalarBroadcastRuntimeABIOrder
             : configContract.runtimeABIOrder))))));
  analysis.description.vlDefOpName = configContract.vlDefOpName;
  analysis.description.vlScopeOpName = configContract.vlScopeOpName;
  analysis.description.vlUses = configContract.vlUses;
  analysis.description.emitCLoopKind = configContract.emitCLoopKind;
  analysis.description.emitCLoopInductionName =
      configContract.emitCLoopInductionName;
  analysis.description.emitCFullChunkVLName =
      configContract.emitCFullChunkVLName;
  analysis.description.emitCLoopVLName =
      tcrv::rvv::getRVVSelectedBodyEmitCLoopVLName();
  analysis.description.remainingAVLMetadata =
      configContract.remainingAVLMetadata;
  analysis.description.pointerAdvanceMetadata =
      configContract.pointerAdvanceMetadata;
  analysis.description.boundedSlice = configContract.boundedSlice;
  analysis.description.multiVL = configContract.multiVL;
  if (analysis.slice.memoryForm ==
      RVVSelectedBodyMemoryForm::UnitStrideConversion) {
    analysis.description.sourceSEW = tcrv::rvv::getRVVFirstSliceSEWBits();
    analysis.description.sourceLMUL = tcrv::rvv::getRVVLMULM1();
    analysis.description.sourceVectorTypeName = "!tcrv_rvv.vector<i32, \"m1\">";
    analysis.description.sourceVectorCType = "vint32m1_t";
    analysis.description.sourceVectorLoadIntrinsic = "__riscv_vle32_v_i32m1";
    analysis.description.conversionRelation = kRVVWideningConversionRelation;
  }
  analysis.description.boundaryOpName = kRVVSelectedBodyLoweringBoundaryOpName;
  analysis.description.targetArtifactRouteID =
      getRVVSelectedBodyTargetArtifactRouteID();
  analysis.description.targetArtifactKind =
      getRVVSelectedBodyTargetArtifactKind();
  analysis.description.runtimeABIParameters.push_back(analysis.slice.lhsABI);
  if (analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::IndexedLoadUnitStore ||
      analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::UnitLoadIndexedStore)
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.indexABI);
  if (analysis.slice.memoryForm ==
      RVVSelectedBodyMemoryForm::MaskedUnitLoadStore)
    analysis.description.runtimeABIParameters.push_back(analysis.slice.maskABI);
  if (analysis.slice.memoryForm !=
          RVVSelectedBodyMemoryForm::UnitStrideConversion &&
      analysis.slice.memoryForm !=
          RVVSelectedBodyMemoryForm::StridedLoadUnitStore &&
      analysis.slice.memoryForm !=
          RVVSelectedBodyMemoryForm::IndexedLoadUnitStore &&
      analysis.slice.memoryForm !=
          RVVSelectedBodyMemoryForm::UnitLoadIndexedStore &&
      analysis.slice.memoryForm !=
          RVVSelectedBodyMemoryForm::MaskedUnitLoadStore)
    analysis.description.runtimeABIParameters.push_back(analysis.slice.rhsABI);
  analysis.description.runtimeABIParameters.push_back(analysis.slice.outABI);
  analysis.description.runtimeABIParameters.push_back(
      analysis.slice.runtimeElementCountABI);
  if (analysis.slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore) {
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.lhsStrideABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.rhsStrideABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.outStrideABI);
  } else if (analysis.slice.memoryForm ==
             RVVSelectedBodyMemoryForm::StridedLoadUnitStore) {
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.lhsStrideABI);
  }

  llvm::Expected<RVVSelectedBodyRouteProfile> routeProfile =
      deriveRVVSelectedBodyRouteProfile(analysis.description);
  if (!routeProfile)
    return routeProfile.takeError();
  llvm::Expected<const RVVSelectedBodyConstructionRoute *> constructionRoute =
      lookupRVVSelectedBodyConstructionRouteByOperationMnemonic(
          routeProfile->operation.operationMnemonic);
  if (!constructionRoute)
    return constructionRoute.takeError();
  analysis.constructionRoute = *constructionRoute;

  analysis.description.typedComputeOpName =
      analysis.slice.arithmeticOp->getName().getStringRef();
  analysis.description.emitCRouteID = analysis.constructionRoute->emitCRouteID;
  analysis.description.runtimeABIName =
      analysis.constructionRoute->runtimeABIName;
  analysis.description.runtimeABIContractName =
      analysis.constructionRoute->runtimeABIContractName;
  analysis.description.vlCType = routeProfile->config.vlCType;
  analysis.description.vectorTypeName =
      routeProfile->config.vectorTypeName;
  analysis.description.indexVectorTypeName =
      routeProfile->operation.isIndexedMemoryMovement
          ? routeProfile->config.indexVectorTypeName
          : "";
  analysis.description.maskTypeName =
      (routeProfile->operation.isCompareSelect ||
       routeProfile->operation.isMaskedArithmetic ||
       routeProfile->operation.isMaskedMemoryMovement)
          ? routeProfile->config.maskTypeName
          : "";
  analysis.description.vectorCType = routeProfile->config.vectorCType;
  analysis.description.indexVectorCType =
      routeProfile->operation.isIndexedMemoryMovement
          ? routeProfile->config.indexVectorCType
          : "";
  analysis.description.maskCType =
      (routeProfile->operation.isCompareSelect ||
       routeProfile->operation.isMaskedArithmetic ||
       routeProfile->operation.isMaskedMemoryMovement)
          ? routeProfile->config.maskCType
          : "";
  analysis.description.setVLIntrinsic =
      routeProfile->config.setVLIntrinsic;
  analysis.description.vectorLoadIntrinsic =
      routeProfile->config.vectorLoadIntrinsic;
  analysis.description.indexLoadIntrinsic =
      routeProfile->operation.isIndexedMemoryMovement
          ? routeProfile->config.indexLoadIntrinsic
          : "";
  analysis.description.indexScaleIntrinsic =
      routeProfile->operation.isIndexedMemoryMovement
          ? routeProfile->config.indexScaleIntrinsic
          : "";
  analysis.description.indexedLoadIntrinsic =
      routeProfile->operation.operation ==
              RVVSelectedBodyOperationKind::IndexedGatherUnitStore
          ? routeProfile->config.indexedLoadIntrinsic
          : "";
  analysis.description.indexedStoreIntrinsic =
      routeProfile->operation.operation ==
              RVVSelectedBodyOperationKind::IndexedScatterUnitLoad
          ? routeProfile->config.indexedStoreIntrinsic
          : "";
  analysis.description.stridedLoadIntrinsic =
      (analysis.slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore ||
       analysis.slice.memoryForm ==
           RVVSelectedBodyMemoryForm::StridedLoadUnitStore)
          ? routeProfile->config.stridedLoadIntrinsic
          : "";
  analysis.description.rhsBroadcastIntrinsic =
      routeProfile->targetLeaves.rhsBroadcastIntrinsic;
  analysis.description.storeIntrinsic =
      routeProfile->config.storeIntrinsic;
  analysis.description.stridedStoreIntrinsic =
      analysis.slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore
          ? routeProfile->config.stridedStoreIntrinsic
          : "";
  analysis.description.intrinsic = routeProfile->targetLeaves.intrinsic;
  analysis.description.compareIntrinsic =
      routeProfile->targetLeaves.compareIntrinsic;
  analysis.description.maskedMergeIntrinsic =
      routeProfile->targetLeaves.maskedMergeIntrinsic;
  analysis.description.resultName = routeProfile->operation.resultName;
  analysis.description.maskName = routeProfile->operation.maskName;
  if (routeProfile->operation.isMaskedArithmetic) {
    analysis.description.maskRole = kRVVMaskedPredicateMaskRole;
    analysis.description.maskSource = kRVVMaskedCompareMaskSource;
    analysis.description.inactiveLaneContract =
        kRVVMaskedInactiveLaneContract;
    analysis.description.maskedPassthroughLayout =
        kRVVMaskedPassthroughLayout;
  }
  if (routeProfile->operation.isMaskedMemoryMovement) {
    analysis.description.maskRole = kRVVMaskRole;
    analysis.description.maskSource = kRVVMaskSource;
    analysis.description.maskMemoryForm = kRVVMaskMemoryForm;
    analysis.description.inactiveLaneContract =
        kRVVMaskedMemoryInactiveLaneContract;
    analysis.description.maskedPassthroughLayout =
        kRVVMaskedMemoryPassthroughLayout;
  }
  if (routeProfile->operation.isReduction) {
    analysis.description.reductionAccumulatorLayout =
        *analysis.slice.reduceOp.getAccumulatorLayout();
    analysis.description.reductionResultLayout =
        *analysis.slice.reduceOp.getResultLayout();
    analysis.description.reductionStoreVL = kRVVReductionStoreVL;
  }
  if (routeProfile->operation.isMultiplyAccumulate) {
    analysis.description.maccAccumulatorLayout =
        *analysis.slice.maccOp.getAccumulatorLayout();
    analysis.description.maccResultLayout =
        *analysis.slice.maccOp.getResultLayout();
  }
  if (routeProfile->operation.operation ==
      RVVSelectedBodyOperationKind::StridedAdd) {
    analysis.description.stridedMemoryLayout = kRVVStridedMemoryLayout;
    analysis.description.lhsStrideSource = kRVVLHSStrideSource;
    analysis.description.rhsStrideSource = kRVVRHSStrideSource;
    analysis.description.outStrideSource = kRVVOutStrideSource;
  }
  if (routeProfile->operation.isMemoryMovement) {
    analysis.description.stridedMemoryLayout =
        kRVVStridedLoadUnitStoreMemoryLayout;
    analysis.description.sourceStrideSource = kRVVSourceStrideSource;
    analysis.description.sourceMemoryForm = kRVVSourceMemoryForm;
    analysis.description.destinationMemoryForm = kRVVDestinationMemoryForm;
  }
  if (routeProfile->operation.isIndexedMemoryMovement) {
    const bool isScatter =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::IndexedScatterUnitLoad;
    analysis.description.indexedMemoryLayout =
        isScatter ? kRVVIndexedScatterMemoryLayout
                  : kRVVIndexedGatherMemoryLayout;
    analysis.description.indexEEW =
        static_cast<std::int64_t>(analysis.slice.indexLoad.getIndexEew());
    analysis.description.offsetUnit =
        isScatter ? analysis.slice.indexedStore.getOffsetUnit()
                  : analysis.slice.indexedLoad.getOffsetUnit();
    analysis.description.indexSource = kRVVIndexSource;
    if (isScatter) {
      analysis.description.indexUniqueness =
          analysis.slice.indexedStore.getIndexUniqueness();
      analysis.description.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
      analysis.description.indexedDestinationMemoryForm =
          kRVVIndexedDestinationMemoryForm;
      analysis.description.destinationMemoryForm =
          kRVVIndexedDestinationMemoryForm;
    } else {
      analysis.description.indexedDataMemoryForm = kRVVIndexedDataMemoryForm;
      analysis.description.destinationMemoryForm = kRVVDestinationMemoryForm;
    }
  }
  if (routeProfile->operation.isMaskedMemoryMovement) {
    analysis.description.indexedMemoryLayout = kRVVMaskedMemoryLayout;
    analysis.description.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
    analysis.description.destinationMemoryForm = kRVVDestinationMemoryForm;
  }

  if (llvm::Error error = validateRVVSelectedBodyRuntimeABIParameters(
          analysis.slice, *analysis.constructionRoute,
          routeProfile->config, analysis.slice.runtimeElementCountABI,
          analysis.slice.outABI))
    return error;
  if (llvm::Error error = verifyRVVSelectedBodyConstructionRouteMapping(
          routeProfile->operation.operationMnemonic,
          analysis.description.typedComputeOpName,
          analysis.constructionRoute->emitCRouteID,
          analysis.constructionRoute->runtimeABIName))
    return std::move(error);
  if (llvm::Error error = verifySelectedRVVRoleSequence(
          analysis.slice, request, *analysis.constructionRoute))
    return std::move(error);
  if (llvm::Error error = verifyRVVSelectedBodyEmitCRouteDescription(
          analysis.description, "selected RVV EmitC route description"))
    return std::move(error);
  return analysis;
}

llvm::ArrayRef<RVVSelectedBodyOperationKind>
getRVVSelectedBodyOperationKinds() {
  return kRVVSelectedBodyOperationKinds;
}

llvm::StringRef
stringifyRVVSelectedBodyOperationKind(RVVSelectedBodyOperationKind op) {
  return getRVVSelectedBodyOperationProfile(op).operationMnemonic;
}

llvm::StringRef
stringifyRVVSelectedBodyMemoryForm(RVVSelectedBodyMemoryForm form) {
  switch (form) {
  case RVVSelectedBodyMemoryForm::VectorRHSLoad:
    return "vector-rhs-load";
  case RVVSelectedBodyMemoryForm::RHSBroadcastLoad:
    return "rhs-broadcast-load";
  case RVVSelectedBodyMemoryForm::RHSScalarBroadcast:
    return "rhs-scalar-broadcast";
  case RVVSelectedBodyMemoryForm::StridedLoadStore:
    return "strided-load-store";
  case RVVSelectedBodyMemoryForm::StridedLoadUnitStore:
    return "strided-load-unit-store";
  case RVVSelectedBodyMemoryForm::IndexedLoadUnitStore:
    return "indexed-load-unit-store";
  case RVVSelectedBodyMemoryForm::UnitLoadIndexedStore:
    return "unit-load-indexed-store";
  case RVVSelectedBodyMemoryForm::MaskedUnitLoadStore:
    return "masked-unit-load-store";
  case RVVSelectedBodyMemoryForm::UnitStrideConversion:
    return "unit-stride-conversion";
  }
  llvm_unreachable("unknown RVV selected-body memory form");
}

llvm::StringRef
getRVVSelectedBodyEmitCRouteID(RVVSelectedBodyOperationKind op) {
  return getRVVSelectedBodyConstructionRouteOrDie(op).emitCRouteID;
}

llvm::StringRef getRVVSelectedBodyEmissionKind() {
  return kRVVSelectedBodyEmissionKind;
}

llvm::StringRef getRVVSelectedBodyLoweringBoundaryOpName() {
  return kRVVSelectedBodyLoweringBoundaryOpName;
}

llvm::StringRef getRVVSelectedBodyRuntimeABIKind() {
  return kRVVSelectedBodyRuntimeABIKind;
}

llvm::StringRef
getRVVSelectedBodyRuntimeABIName(RVVSelectedBodyOperationKind op) {
  return getRVVSelectedBodyConstructionRouteOrDie(op).runtimeABIName;
}

llvm::StringRef getRVVSelectedBodyRuntimeGlueRole() {
  return kRVVSelectedBodyRuntimeGlueRole;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVSelectedBodyRuntimeABIParameters() {
  return tcrv::rvv::getRVVSelectedBodyRuntimeABIParameters();
}

RVVSelectedBodyConstructionMetadataFacts
getRVVSelectedBodyConstructionMetadataFacts(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  RVVSelectedBodyConstructionMetadataFacts facts;
  facts.operationMnemonic =
      stringifyRVVSelectedBodyOperationKind(description.operation);
  facts.typedComputeOpName = description.typedComputeOpName;
  facts.emitCRouteID = description.emitCRouteID;
  facts.targetArtifactRouteID = description.targetArtifactRouteID;
  facts.targetArtifactKind = description.targetArtifactKind;
  facts.runtimeABIName = description.runtimeABIName;
  facts.runtimeABIContractName = description.runtimeABIContractName;
  facts.runtimeABIParameters = description.runtimeABIParameters;
  return facts;
}

llvm::Error verifyRVVSelectedBodyEmitCRouteDescription(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (context.trim().empty())
    return makeRVVEmitCRouteProviderError(
        "selected-body route description verification requires a non-empty "
        "context");

  llvm::Expected<RVVSelectedBodyRouteProfile> profile =
      deriveRVVSelectedBodyRouteProfile(description);
  if (!profile)
    return profile.takeError();
  const RVVSelectedBodyOperationProfile &operationProfile =
      profile->operation;
  const RVVSelectedBodyConfigProfile &configProfile = profile->config;
  const RVVSelectedBodyTargetLeafProfile &targetLeaves =
      profile->targetLeaves;
  const tcrv::rvv::RVVSelectedBodyConfigVLContract &configContract =
      *configProfile.configContract;

  llvm::Expected<const RVVSelectedBodyConstructionRoute *> route =
      lookupRVVSelectedBodyConstructionRouteByOperationMnemonic(
          operationProfile.operationMnemonic);
  if (!route)
    return route.takeError();
  const RVVSelectedBodyConstructionRoute &constructionRoute = **route;

  const bool usesGenericBinary =
      description.typedComputeOpName == "tcrv_rvv.binary";
  if (usesGenericBinary && operationProfile.isCompareSelect)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " compare/select cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary && operationProfile.isReduction)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " reduction cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary && operationProfile.isMaskedArithmetic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " masked arithmetic cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary && operationProfile.isMultiplyAccumulate)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " multiply-accumulate cannot use generic tcrv_rvv.binary");
  if (!usesGenericBinary)
    if (llvm::Error error = requireRouteDescriptionField(
            context, "typed compute op", description.typedComputeOpName,
            constructionRoute.typedComputeOpName))
      return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "EmitC route id", description.emitCRouteID,
          constructionRoute.emitCRouteID))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "target artifact route id",
          description.targetArtifactRouteID,
          getRVVSelectedBodyTargetArtifactRouteID()))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "target artifact kind", description.targetArtifactKind,
          getRVVSelectedBodyTargetArtifactKind()))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "runtime ABI name", description.runtimeABIName,
          constructionRoute.runtimeABIName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "runtime ABI contract", description.runtimeABIContractName,
          constructionRoute.runtimeABIContractName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "lowering boundary op", description.boundaryOpName,
          kRVVSelectedBodyLoweringBoundaryOpName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "config contract", description.configContractID,
          configContract.configContractID))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "runtime VL contract", description.runtimeVLContractID,
          configContract.runtimeVLContractID))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "runtime AVL source", description.runtimeAVLASource,
          configContract.runtimeAVLASource))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "runtime ABI order", description.runtimeABIOrder,
          operationProfile.operation == RVVSelectedBodyOperationKind::StridedAdd
              ? kRVVStridedRuntimeABIOrder
          : (operationProfile.isMemoryMovement
              ? kRVVStridedLoadUnitStoreRuntimeABIOrder
          : (operationProfile.isMaskedMemoryMovement
              ? kRVVMaskedMemoryRuntimeABIOrder
          : (operationProfile.isIndexedMemoryMovement
              ? (operationProfile.operation ==
                         RVVSelectedBodyOperationKind::IndexedScatterUnitLoad
                     ? kRVVIndexedScatterRuntimeABIOrder
                     : kRVVIndexedGatherRuntimeABIOrder)
              : (operationProfile.isWideningConversion
                     ? kRVVWideningConversionRuntimeABIOrder
                     : (description.memoryForm ==
                         RVVSelectedBodyMemoryForm::RHSScalarBroadcast
                     ? kRVVScalarBroadcastRuntimeABIOrder
                     : configContract.runtimeABIOrder)))))))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "VL def op", description.vlDefOpName,
          configContract.vlDefOpName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "VL scope op", description.vlScopeOpName,
          configContract.vlScopeOpName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "VL uses", description.vlUses, configContract.vlUses))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "EmitC loop kind", description.emitCLoopKind,
          configContract.emitCLoopKind))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "EmitC loop induction", description.emitCLoopInductionName,
          configContract.emitCLoopInductionName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "EmitC full-chunk VL", description.emitCFullChunkVLName,
          configContract.emitCFullChunkVLName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "EmitC loop VL", description.emitCLoopVLName,
          tcrv::rvv::getRVVSelectedBodyEmitCLoopVLName()))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "remaining AVL metadata",
          description.remainingAVLMetadata, configContract.remainingAVLMetadata))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "pointer advance metadata",
          description.pointerAdvanceMetadata,
          configContract.pointerAdvanceMetadata))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "bounded slice", description.boundedSlice,
          configContract.boundedSlice))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "multi-VL support", description.multiVL,
          configContract.multiVL))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "VL C type", description.vlCType, configProfile.vlCType))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "vector type", description.vectorTypeName,
          configProfile.vectorTypeName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "vector C type", description.vectorCType,
          configProfile.vectorCType))
    return error;
  if (operationProfile.isIndexedMemoryMovement) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index vector type", description.indexVectorTypeName,
            configProfile.indexVectorTypeName))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index vector C type", description.indexVectorCType,
            configProfile.indexVectorCType))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index vector type", description.indexVectorTypeName, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index vector C type", description.indexVectorCType, ""))
      return error;
  }
  if (operationProfile.isWideningConversion) {
    if (description.sourceSEW != tcrv::rvv::getRVVFirstSliceSEWBits())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " source SEW must be provider-derived as 32 for i32-to-i64 "
          "widening conversion");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source LMUL", description.sourceLMUL,
            tcrv::rvv::getRVVLMULM1()))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector type", description.sourceVectorTypeName,
            "!tcrv_rvv.vector<i32, \"m1\">"))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector C type", description.sourceVectorCType,
            "vint32m1_t"))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector-load intrinsic",
            description.sourceVectorLoadIntrinsic,
            "__riscv_vle32_v_i32m1"))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "conversion relation", description.conversionRelation,
            kRVVWideningConversionRelation))
      return error;
  } else {
    if (description.sourceSEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " source SEW must be empty for non-conversion routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source LMUL", description.sourceLMUL, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector type", description.sourceVectorTypeName,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector C type", description.sourceVectorCType,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector-load intrinsic",
            description.sourceVectorLoadIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "conversion relation", description.conversionRelation,
            ""))
      return error;
  }
  if (operationProfile.isCompareSelect || operationProfile.isMaskedArithmetic ||
      operationProfile.isMaskedMemoryMovement) {
    if (llvm::Error error =
            requireRouteDescriptionField(context, "mask type",
                                         description.maskTypeName,
                                         configProfile.maskTypeName))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "mask C type",
                                         description.maskCType,
                                         configProfile.maskCType))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask type", description.maskTypeName, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask C type", description.maskCType, ""))
      return error;
  }
  if (llvm::Error error = requireRouteDescriptionField(
          context, "setvl intrinsic", description.setVLIntrinsic,
          configProfile.setVLIntrinsic))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "vector-load intrinsic", description.vectorLoadIntrinsic,
          configProfile.vectorLoadIntrinsic))
    return error;
  if (operationProfile.isIndexedMemoryMovement) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index-load intrinsic", description.indexLoadIntrinsic,
            configProfile.indexLoadIntrinsic))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index-scale intrinsic", description.indexScaleIntrinsic,
            configProfile.indexScaleIntrinsic))
      return error;
    if (operationProfile.operation ==
        RVVSelectedBodyOperationKind::IndexedGatherUnitStore) {
      if (llvm::Error error = requireRouteDescriptionField(
              context, "indexed-load intrinsic",
              description.indexedLoadIntrinsic,
              configProfile.indexedLoadIntrinsic))
        return error;
      if (llvm::Error error = requireRouteDescriptionField(
              context, "indexed-store intrinsic",
              description.indexedStoreIntrinsic, ""))
        return error;
    } else {
      if (llvm::Error error = requireRouteDescriptionField(
              context, "indexed-load intrinsic",
              description.indexedLoadIntrinsic, ""))
        return error;
      if (llvm::Error error = requireRouteDescriptionField(
              context, "indexed-store intrinsic",
              description.indexedStoreIntrinsic,
              configProfile.indexedStoreIntrinsic))
        return error;
    }
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index-load intrinsic", description.indexLoadIntrinsic,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index-scale intrinsic", description.indexScaleIntrinsic,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed-load intrinsic",
            description.indexedLoadIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed-store intrinsic",
            description.indexedStoreIntrinsic, ""))
      return error;
  }
  if (operationProfile.isStridedMemory) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided-load intrinsic",
            description.stridedLoadIntrinsic,
            configProfile.stridedLoadIntrinsic))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided-load intrinsic",
            description.stridedLoadIntrinsic, ""))
      return error;
  }
  if (llvm::Error error = requireRouteDescriptionField(
          context, "store intrinsic", description.storeIntrinsic,
          configProfile.storeIntrinsic))
    return error;
  if (operationProfile.operation == RVVSelectedBodyOperationKind::StridedAdd) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided-store intrinsic",
            description.stridedStoreIntrinsic,
            configProfile.stridedStoreIntrinsic))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided-store intrinsic",
            description.stridedStoreIntrinsic, ""))
      return error;
  }
  if (llvm::Error error = requireRouteDescriptionField(
          context, "compute intrinsic", description.intrinsic,
          targetLeaves.intrinsic))
    return error;
  if (operationProfile.isCompareSelect) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "compare intrinsic", description.compareIntrinsic,
            targetLeaves.compareIntrinsic))
      return error;
  } else if (operationProfile.isMaskedArithmetic ||
             operationProfile.isMaskedMemoryMovement) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "compare intrinsic", description.compareIntrinsic,
            targetLeaves.compareIntrinsic))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "compare intrinsic", description.compareIntrinsic, ""))
      return error;
  }
  if (operationProfile.isMaskedArithmetic ||
      operationProfile.isMaskedMemoryMovement) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked merge intrinsic",
            description.maskedMergeIntrinsic,
            targetLeaves.maskedMergeIntrinsic))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked merge intrinsic",
            description.maskedMergeIntrinsic, ""))
      return error;
  }
  if (llvm::Error error = requireRouteDescriptionField(
          context, "result value name", description.resultName,
          operationProfile.resultName))
    return error;
  if (operationProfile.isCompareSelect || operationProfile.isMaskedArithmetic ||
      operationProfile.isMaskedMemoryMovement) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask value name", description.maskName,
            operationProfile.maskName))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask value name", description.maskName, ""))
      return error;
  }
  if (operationProfile.isMaskedArithmetic) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask role", description.maskRole,
            kRVVMaskedPredicateMaskRole))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask source", description.maskSource,
            kRVVMaskedCompareMaskSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "inactive lane contract",
            description.inactiveLaneContract,
            kRVVMaskedInactiveLaneContract))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked passthrough layout",
            description.maskedPassthroughLayout,
            kRVVMaskedPassthroughLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask memory form", description.maskMemoryForm, ""))
      return error;
  } else if (operationProfile.isMaskedMemoryMovement) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask role", description.maskRole, kRVVMaskRole))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask source", description.maskSource, kRVVMaskSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask memory form", description.maskMemoryForm,
            kRVVMaskMemoryForm))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "inactive lane contract",
            description.inactiveLaneContract,
            kRVVMaskedMemoryInactiveLaneContract))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked passthrough layout",
            description.maskedPassthroughLayout,
            kRVVMaskedMemoryPassthroughLayout))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask role", description.maskRole, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask source", description.maskSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask memory form", description.maskMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "inactive lane contract",
            description.inactiveLaneContract, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked passthrough layout",
            description.maskedPassthroughLayout, ""))
      return error;
  }
  if (operationProfile.isReduction) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction accumulator layout",
            description.reductionAccumulatorLayout,
            kRVVReductionAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction result layout",
            description.reductionResultLayout, kRVVReductionResultLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction store VL", description.reductionStoreVL,
            kRVVReductionStoreVL))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction accumulator layout",
            description.reductionAccumulatorLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction result layout",
            description.reductionResultLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction store VL", description.reductionStoreVL, ""))
      return error;
  }
  if (operationProfile.isMultiplyAccumulate) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "multiply-accumulate accumulator layout",
            description.maccAccumulatorLayout, kRVVMAccAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "multiply-accumulate result layout",
            description.maccResultLayout, kRVVMAccResultLayout))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "multiply-accumulate accumulator layout",
            description.maccAccumulatorLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "multiply-accumulate result layout",
            description.maccResultLayout, ""))
      return error;
  }
  if (operationProfile.operation == RVVSelectedBodyOperationKind::StridedAdd) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout",
            description.stridedMemoryLayout, kRVVStridedMemoryLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource,
            kRVVLHSStrideSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource,
            kRVVRHSStrideSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource,
            kRVVOutStrideSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm, ""))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm, ""))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for non-indexed routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (operationProfile.isMemoryMovement) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout",
            description.stridedMemoryLayout,
            kRVVStridedLoadUnitStoreMemoryLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            kRVVSourceStrideSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVSourceMemoryForm))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "destination memory form",
            description.destinationMemoryForm, kRVVDestinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for non-indexed routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (operationProfile.isMaskedMemoryMovement) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout", description.stridedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            kRVVMaskedMemoryLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm,
                                         kRVVDestinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for masked memory routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (operationProfile.isIndexedMemoryMovement) {
    const bool isScatter =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::IndexedScatterUnitLoad;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout",
            description.stridedMemoryLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            isScatter ? kRVVIndexedScatterMemoryLayout
                      : kRVVIndexedGatherMemoryLayout))
      return error;
    if (description.indexEEW != 32)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be provider-derived as 32 for indexed memory");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit,
            kRVVIndexedGatherOffsetUnit))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource,
            kRVVIndexSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness,
            isScatter ? kRVVIndexedScatterIndexUniqueness : ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm,
            isScatter ? "" : kRVVIndexedDataMemoryForm))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm,
            isScatter ? kRVVIndexedDestinationMemoryForm : ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            isScatter ? kRVVUnitStrideSourceMemoryForm : ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "destination memory form",
            description.destinationMemoryForm,
            isScatter ? kRVVIndexedDestinationMemoryForm
                      : kRVVDestinationMemoryForm))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout",
            description.stridedMemoryLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm, ""))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm, ""))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for non-indexed routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  }
  if (description.memoryForm == RVVSelectedBodyMemoryForm::RHSBroadcastLoad ||
      description.memoryForm == RVVSelectedBodyMemoryForm::RHSScalarBroadcast)
    if (llvm::Error error = requireRouteDescriptionText(
            context, "RHS broadcast intrinsic",
            description.rhsBroadcastIntrinsic))
      return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "RHS broadcast intrinsic",
          description.rhsBroadcastIntrinsic,
          targetLeaves.rhsBroadcastIntrinsic))
    return error;

  if (llvm::Error error = verifyRVVSelectedBodyConstructionRuntimeABIParameters(
          description.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));

  return llvm::Error::success();
}

llvm::SmallVector<support::ArtifactMetadataEntry, 16>
getRVVSelectedBodyConfigArtifactMetadata(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  llvm::SmallVector<support::ArtifactMetadataEntry, 16> metadata;
  metadata.push_back(
      {"tcrv_rvv.config_contract", description.configContractID});
  metadata.push_back({"tcrv_rvv.sew", llvm::Twine(description.sew).str()});
  metadata.push_back({"tcrv_rvv.lmul", description.lmul});
  metadata.push_back({"tcrv_rvv.tail_policy", description.tailPolicy});
  metadata.push_back({"tcrv_rvv.mask_policy", description.maskPolicy});
  metadata.push_back({"tcrv_rvv.memory_form",
                      stringifyRVVSelectedBodyMemoryForm(
                          description.memoryForm)});
  metadata.push_back(
      {"tcrv_rvv.runtime_vl_contract", description.runtimeVLContractID});
  metadata.push_back(
      {"tcrv_rvv.runtime_avl_source", description.runtimeAVLASource});
  metadata.push_back({"tcrv_rvv.vl_def", description.vlDefOpName});
  metadata.push_back({"tcrv_rvv.vl_scope", description.vlScopeOpName});
  metadata.push_back({"tcrv_rvv.vl_uses", description.vlUses});
  metadata.push_back(
      {"tcrv_rvv.runtime_abi_order", description.runtimeABIOrder});
  metadata.push_back({"tcrv_rvv.runtime_avl_abi_parameter",
                      tcrv::rvv::getRVVSelectedBodyRuntimeAVLParameterName()});
  metadata.push_back({"tcrv_rvv.emitc_loop", description.emitCLoopKind});
  metadata.push_back(
      {"tcrv_rvv.loop_induction", description.emitCLoopInductionName});
  metadata.push_back({"tcrv_rvv.loop_step", description.emitCFullChunkVLName});
  metadata.push_back(
      {"tcrv_rvv.remaining_avl", description.remainingAVLMetadata});
  metadata.push_back(
      {"tcrv_rvv.pointer_advance", description.pointerAdvanceMetadata});
  metadata.push_back({"tcrv_rvv.bounded_slice", description.boundedSlice});
  metadata.push_back({"tcrv_rvv.multi_vl", description.multiVL});
  if (description.operation == RVVSelectedBodyOperationKind::MaskedAdd) {
    metadata.push_back({"tcrv_rvv.mask_role", description.maskRole});
    metadata.push_back({"tcrv_rvv.mask_source", description.maskSource});
    metadata.push_back({"tcrv_rvv.inactive_lane_contract",
                        description.inactiveLaneContract});
    metadata.push_back({"tcrv_rvv.masked_passthrough_layout",
                        description.maskedPassthroughLayout});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::MaskedUnitLoadStore) {
    metadata.push_back(
        {"tcrv_rvv.masked_memory_layout", description.indexedMemoryLayout});
    metadata.push_back({"tcrv_rvv.mask_role", description.maskRole});
    metadata.push_back({"tcrv_rvv.mask_source", description.maskSource});
    metadata.push_back(
        {"tcrv_rvv.mask_memory_form", description.maskMemoryForm});
    metadata.push_back({"tcrv_rvv.inactive_lane_contract",
                        description.inactiveLaneContract});
    metadata.push_back({"tcrv_rvv.masked_passthrough_layout",
                        description.maskedPassthroughLayout});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  if (description.operation == RVVSelectedBodyOperationKind::ReduceAdd) {
    metadata.push_back({"tcrv_rvv.reduction_accumulator_layout",
                        description.reductionAccumulatorLayout});
    metadata.push_back({"tcrv_rvv.reduction_result_layout",
                        description.reductionResultLayout});
    metadata.push_back(
        {"tcrv_rvv.reduction_store_vl", description.reductionStoreVL});
  }
  if (description.operation == RVVSelectedBodyOperationKind::MAccAdd) {
    metadata.push_back({"tcrv_rvv.macc_accumulator_layout",
                        description.maccAccumulatorLayout});
    metadata.push_back(
        {"tcrv_rvv.macc_result_layout", description.maccResultLayout});
  }
  if (description.operation == RVVSelectedBodyOperationKind::StridedAdd) {
    metadata.push_back({"tcrv_rvv.strided_memory_layout",
                        description.stridedMemoryLayout});
    metadata.push_back(
        {"tcrv_rvv.lhs_stride_source", description.lhsStrideSource});
    metadata.push_back(
        {"tcrv_rvv.rhs_stride_source", description.rhsStrideSource});
    metadata.push_back(
        {"tcrv_rvv.out_stride_source", description.outStrideSource});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::StridedLoadUnitStore) {
    metadata.push_back({"tcrv_rvv.strided_memory_layout",
                        description.stridedMemoryLayout});
    metadata.push_back(
        {"tcrv_rvv.source_stride_source", description.sourceStrideSource});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::IndexedGatherUnitStore) {
    metadata.push_back({"tcrv_rvv.indexed_memory_layout",
                        description.indexedMemoryLayout});
    metadata.push_back(
        {"tcrv_rvv.index_source", description.indexSource});
    metadata.push_back({"tcrv_rvv.index_eew",
                        llvm::Twine(description.indexEEW).str()});
    metadata.push_back({"tcrv_rvv.offset_unit", description.offsetUnit});
    metadata.push_back({"tcrv_rvv.indexed_data_memory_form",
                        description.indexedDataMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::IndexedScatterUnitLoad) {
    metadata.push_back({"tcrv_rvv.indexed_memory_layout",
                        description.indexedMemoryLayout});
    metadata.push_back(
        {"tcrv_rvv.index_source", description.indexSource});
    metadata.push_back({"tcrv_rvv.index_eew",
                        llvm::Twine(description.indexEEW).str()});
    metadata.push_back({"tcrv_rvv.offset_unit", description.offsetUnit});
    metadata.push_back({"tcrv_rvv.index_uniqueness",
                        description.indexUniqueness});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.indexed_destination_memory_form",
                        description.indexedDestinationMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  if (description.operation == RVVSelectedBodyOperationKind::WidenI32ToI64) {
    metadata.push_back(
        {"tcrv_rvv.source_sew", llvm::Twine(description.sourceSEW).str()});
    metadata.push_back({"tcrv_rvv.source_lmul", description.sourceLMUL});
    metadata.push_back(
        {"tcrv_rvv.dest_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.dest_lmul", description.lmul});
    metadata.push_back(
        {"tcrv_rvv.conversion_relation", description.conversionRelation});
  }
  return metadata;
}


} // namespace tianchenrv::plugin::rvv
