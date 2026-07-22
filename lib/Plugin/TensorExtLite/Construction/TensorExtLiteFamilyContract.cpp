#include "Weft/Plugin/TensorExtLite/TensorExtLiteFamilyContract.h"

namespace weft::plugin::tensorext_lite {
namespace {

const TensorExtLiteConstructionStep kConstructionSteps[] = {
    {"weft_tensorext_lite.config_skeleton"},
    {"weft_tensorext_lite.load_frag_skeleton"},
    {"weft_tensorext_lite.tile_mma_skeleton"},
    {"weft_tensorext_lite.store_frag_skeleton"},
};

constexpr llvm::StringLiteral kRouteID(
    "tensorext-lite-fragment-mma-emitc-route");
constexpr llvm::StringLiteral kEmissionKind(
    "materialized-emitc-cpp-tensorext-lite-fragment-mma-module");
constexpr llvm::StringLiteral kArtifactKind("riscv-elf-relocatable-object");
constexpr llvm::StringLiteral kRuntimeABI(
    "tensorext-lite-fragment-mma-runtime-c-abi.v1");
constexpr llvm::StringLiteral kRuntimeABIKind("plugin-owned-runtime-abi");
constexpr llvm::StringLiteral kRuntimeGlueRole(
    "emitc-cpp-tensorext-lite-fragment-runtime-glue");
constexpr llvm::StringLiteral kBodyRootOpName(
    "weft_tensorext_lite.config_skeleton");
constexpr llvm::StringLiteral kHeaderRouteID(
    "tensorext-lite-fragment-mma-emitc-route.header");
constexpr llvm::StringLiteral kHeaderArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kComponentGroup(
    "tensorext-lite-fragment-mma-materialized-emitc-bundle.v1");
constexpr llvm::StringLiteral kObjectHandoffKind(
    "materialized-emitc-cpp-tensorext-lite-fragment-object");
constexpr llvm::StringLiteral kTranslateRouteID(
    "weft-tensorext-lite-emitc-to-cpp");
constexpr llvm::StringLiteral kConfigCallee("weft_tensorext_lite_config");
constexpr llvm::StringLiteral kLoadCallee("weft_tensorext_lite_load_frag");
constexpr llvm::StringLiteral kComputeCallee("weft_tensorext_lite_tile_mma");
constexpr llvm::StringLiteral kStoreCallee("weft_tensorext_lite_store_frag");

constexpr llvm::StringLiteral kRouteMetadataName(
    "tensorext_lite_emitc_route");
constexpr llvm::StringLiteral kSourceOpsMetadataName(
    "tensorext_lite_source_ops");
constexpr llvm::StringLiteral kSourceRolesMetadataName(
    "tensorext_lite_source_roles");
constexpr llvm::StringLiteral kSourceInterfaceMetadataName(
    "tensorext_lite_source_op_interface");

const TensorExtLiteArtifactRoute kRoute = {
    kRouteID,          kEmissionKind,     kArtifactKind,
    kRuntimeABI,       kRuntimeABIKind,   kRuntimeABI,
    kRuntimeGlueRole,  kBodyRootOpName,   kHeaderRouteID,
    kHeaderArtifactKind, kComponentGroup, kObjectHandoffKind,
    kTranslateRouteID, kConfigCallee,     kLoadCallee,
    kComputeCallee,    kStoreCallee};

} // namespace

llvm::ArrayRef<TensorExtLiteConstructionStep>
getTensorExtLiteConstructionSteps() {
  return kConstructionSteps;
}

const TensorExtLiteArtifactRoute &getTensorExtLiteArtifactRoute() {
  return kRoute;
}

llvm::ArrayRef<support::RuntimeABIParameter>
getTensorExtLiteRuntimeABIParameters() {
  return {};
}

llvm::StringRef getTensorExtLiteArtifactRouteMetadataName() {
  return kRouteMetadataName;
}

llvm::StringRef getTensorExtLiteSourceOpsMetadataName() {
  return kSourceOpsMetadataName;
}

llvm::StringRef getTensorExtLiteSourceRolesMetadataName() {
  return kSourceRolesMetadataName;
}

llvm::StringRef getTensorExtLiteSourceOpInterfaceMetadataName() {
  return kSourceInterfaceMetadataName;
}

} // namespace weft::plugin::tensorext_lite
