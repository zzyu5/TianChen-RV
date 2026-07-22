#include "Weft/Plugin/Demo/DemoFamilyContract.h"

namespace weft::plugin::demo_ext {
namespace {

constexpr llvm::StringLiteral kRouteID(
    "demo-extension-compute-skeleton-emitc-route");
constexpr llvm::StringLiteral kEmissionKind(
    "materialized-emitc-cpp-demo-compute-skeleton-module");
constexpr llvm::StringLiteral kArtifactKind("riscv-elf-relocatable-object");
constexpr llvm::StringLiteral kRuntimeABI(
    "demo-extension-compute-skeleton-runtime-c-abi.v1");
constexpr llvm::StringLiteral kRuntimeABIKind("plugin-owned-runtime-abi");
constexpr llvm::StringLiteral kRuntimeGlueRole(
    "emitc-cpp-demo-compute-skeleton-runtime-glue");
constexpr llvm::StringLiteral kBodyOpName("weft_demo.compute_skeleton");
constexpr llvm::StringLiteral kHeaderRouteID(
    "demo-extension-compute-skeleton-emitc-route.header");
constexpr llvm::StringLiteral kHeaderArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kComponentGroup(
    "demo-compute-skeleton-materialized-emitc-bundle.v1");
constexpr llvm::StringLiteral kObjectHandoffKind(
    "materialized-emitc-cpp-demo-object");
constexpr llvm::StringLiteral kCallee("weft_demo_compute_skeleton");
constexpr llvm::StringLiteral kResultName("demo_compute_sentinel");
constexpr llvm::StringLiteral kResultCType("int32_t");
constexpr llvm::StringLiteral kTranslateRouteID("weft-demo-emitc-to-cpp");

const DemoArtifactRoute kRoute = {
    kRouteID,          kEmissionKind,      kArtifactKind,
    kBodyOpName,       kRuntimeABI,        kRuntimeABIKind,
    kRuntimeABI,       kRuntimeGlueRole,   kHeaderRouteID,
    kHeaderArtifactKind, kComponentGroup, kObjectHandoffKind,
    kCallee,           kResultName,        kResultCType,
    kTranslateRouteID};

} // namespace

const DemoArtifactRoute &getDemoArtifactRoute() { return kRoute; }

llvm::ArrayRef<support::RuntimeABIParameter> getDemoRuntimeABIParameters() {
  return {};
}

} // namespace weft::plugin::demo_ext
