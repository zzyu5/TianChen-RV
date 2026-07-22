#ifndef WEFT_PLUGIN_DEMO_DEMOFAMILYCONTRACT_H
#define WEFT_PLUGIN_DEMO_DEMOFAMILYCONTRACT_H

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

namespace weft::plugin::demo_ext {

/// Family legality consumes only the selected variant and its bound target
/// capability set.  Artifact routes and evidence metadata are not inputs.
llvm::Error verifyDemoSelectedVariantLegality(
    weft::exec::VariantOp variant, weft::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities);

/// Pure artifact projection constants for the already constructed Demo body.
/// They contain no candidate, legality, completion, or compute decision.
struct DemoArtifactRoute {
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
  llvm::StringRef emitCToCppTranslateRouteID;
};

const DemoArtifactRoute &getDemoArtifactRoute();
llvm::ArrayRef<support::RuntimeABIParameter> getDemoRuntimeABIParameters();

} // namespace weft::plugin::demo_ext

#endif // WEFT_PLUGIN_DEMO_DEMOFAMILYCONTRACT_H
