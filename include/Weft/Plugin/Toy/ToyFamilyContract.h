#ifndef WEFT_PLUGIN_TOY_TOYFAMILYCONTRACT_H
#define WEFT_PLUGIN_TOY_TOYFAMILYCONTRACT_H

#include "Weft/Support/RuntimeABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace weft::exec {
class KernelOp;
class VariantOp;
} // namespace weft::exec

namespace weft::support {
class TargetCapabilitySet;
} // namespace weft::support

namespace weft::plugin::toy {

llvm::Error verifyToySelectedVariantLegality(
    weft::exec::VariantOp variant, weft::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities);

struct ToyArtifactRoute {
  llvm::StringRef routeID;
  llvm::StringRef emissionKind;
  llvm::StringRef artifactKind;
  llvm::StringRef loweringBoundaryOpName;
  llvm::StringRef runtimeABI;
  llvm::StringRef runtimeABIKind;
  llvm::StringRef runtimeABIName;
  llvm::StringRef runtimeGlueRole;
  llvm::StringRef headerRouteID;
  llvm::StringRef headerArtifactKind;
  llvm::StringRef bundleComponentGroup;
  llvm::StringRef objectHandoffKind;
  llvm::StringRef callee;
  llvm::StringRef resultName;
  llvm::StringRef resultCType;
};

const ToyArtifactRoute &getToyArtifactRoute();
llvm::ArrayRef<support::RuntimeABIParameter> getToyRuntimeABIParameters();

} // namespace weft::plugin::toy

#endif // WEFT_PLUGIN_TOY_TOYFAMILYCONTRACT_H
