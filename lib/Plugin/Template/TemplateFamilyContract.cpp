#include "Weft/Plugin/Template/TemplateFamilyContract.h"

namespace weft::plugin::template_ext {
namespace {

constexpr llvm::StringLiteral kRouteID(
    "template-extension-compute-skeleton-emitc-route");
constexpr llvm::StringLiteral kEmissionKind(
    "materialized-emitc-cpp-template-compute-skeleton-module");
constexpr llvm::StringLiteral kArtifactKind("riscv-elf-relocatable-object");
constexpr llvm::StringLiteral kBodyOpName("weft_template.compute_skeleton");
constexpr llvm::StringLiteral kRuntimeABI(
    "template-extension-compute-skeleton-runtime-c-abi.v1");
constexpr llvm::StringLiteral kRuntimeABIKind("plugin-owned-runtime-abi");
constexpr llvm::StringLiteral kRuntimeGlueRole(
    "emitc-cpp-template-compute-skeleton-runtime-glue");
constexpr llvm::StringLiteral kHeaderRouteID(
    "template-extension-compute-skeleton-emitc-route.header");
constexpr llvm::StringLiteral kHeaderArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kComponentGroup(
    "template-compute-skeleton-materialized-emitc-bundle.v1");
constexpr llvm::StringLiteral kObjectHandoffKind(
    "materialized-emitc-cpp-template-object");
constexpr llvm::StringLiteral kCallee("weft_template_compute_skeleton");
constexpr llvm::StringLiteral kResultName("template_compute_sentinel");
constexpr llvm::StringLiteral kResultCType("int32_t");
constexpr llvm::StringLiteral kTranslateRouteID("weft-template-emitc-to-cpp");

constexpr llvm::StringLiteral kRouteMetadataName("template_emitc_route");
constexpr llvm::StringLiteral kSourceOpMetadataName("template_source_op");
constexpr llvm::StringLiteral kSourceRoleMetadataName("template_source_role");
constexpr llvm::StringLiteral kSourceInterfaceMetadataName(
    "template_source_op_interface");

const TemplateArtifactRoute kRoute = {
    kRouteID,          kEmissionKind,      kArtifactKind,
    kBodyOpName,       kRuntimeABI,        kRuntimeABIKind,
    kRuntimeABI,       kRuntimeGlueRole,   kHeaderRouteID,
    kHeaderArtifactKind, kComponentGroup, kObjectHandoffKind,
    kCallee,           kResultName,        kResultCType,
    kTranslateRouteID};

} // namespace

const TemplateArtifactRoute &getTemplateArtifactRoute() { return kRoute; }

llvm::ArrayRef<support::RuntimeABIParameter>
getTemplateRuntimeABIParameters() {
  return {};
}

llvm::StringRef getTemplateArtifactRouteMetadataName() {
  return kRouteMetadataName;
}

llvm::StringRef getTemplateSourceOpMetadataName() {
  return kSourceOpMetadataName;
}

llvm::StringRef getTemplateSourceRoleMetadataName() {
  return kSourceRoleMetadataName;
}

llvm::StringRef getTemplateSourceOpInterfaceMetadataName() {
  return kSourceInterfaceMetadataName;
}

} // namespace weft::plugin::template_ext
