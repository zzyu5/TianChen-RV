#include "Weft/Plugin/Toy/ToyFamilyContract.h"

namespace weft::plugin::toy {
namespace {

constexpr llvm::StringLiteral kRouteID("toy-template-compute-emitc-route");
constexpr llvm::StringLiteral kEmissionKind(
    "materialized-emitc-cpp-toy-template-module");
constexpr llvm::StringLiteral kArtifactKind("riscv-elf-relocatable-object");
constexpr llvm::StringLiteral kBodyOpName("weft_toy.compute_skeleton");
constexpr llvm::StringLiteral kRuntimeABI(
    "toy-template-compute-runtime-c-abi.v1");
constexpr llvm::StringLiteral kRuntimeABIKind("plugin-owned-runtime-abi");
constexpr llvm::StringLiteral kRuntimeGlueRole(
    "emitc-cpp-toy-template-runtime-glue");
constexpr llvm::StringLiteral kHeaderRouteID(
    "toy-template-compute-emitc-route.header");
constexpr llvm::StringLiteral kHeaderArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kComponentGroup(
    "toy-template-compute-materialized-emitc-bundle.v1");
constexpr llvm::StringLiteral kObjectHandoffKind(
    "materialized-emitc-cpp-toy-template-object");
constexpr llvm::StringLiteral kCallee("weft_toy_template_compute");
constexpr llvm::StringLiteral kResultName("toy_value");
constexpr llvm::StringLiteral kResultCType("int32_t");

const ToyArtifactRoute kRoute = {
    kRouteID,          kEmissionKind,    kArtifactKind,
    kBodyOpName,       kRuntimeABI,      kRuntimeABIKind,
    kRuntimeABI,       kRuntimeGlueRole, kHeaderRouteID,
    kHeaderArtifactKind, kComponentGroup, kObjectHandoffKind,
    kCallee,           kResultName,      kResultCType};

const support::RuntimeABIParameter kRuntimeABIParameters[] = {
    support::makeTargetExportABIParameter(
        "toy_value_count", "size_t",
        support::RuntimeABIParameterRole::RuntimeElementCount)};

} // namespace

const ToyArtifactRoute &getToyArtifactRoute() { return kRoute; }

llvm::ArrayRef<support::RuntimeABIParameter> getToyRuntimeABIParameters() {
  return kRuntimeABIParameters;
}

} // namespace weft::plugin::toy
