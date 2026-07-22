#ifndef WEFT_PLUGIN_TENSOREXTLITE_TENSOREXTLITEFAMILYCONTRACT_H
#define WEFT_PLUGIN_TENSOREXTLITE_TENSOREXTLITEFAMILYCONTRACT_H

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

namespace weft::plugin::tensorext_lite {

llvm::Error verifyTensorExtLiteSelectedVariantLegality(
    weft::exec::VariantOp variant, weft::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities);

/// The family-local typed construction order.  It deliberately contains no
/// artifact callee, interface spelling, status, evidence, or completion stamp.
struct TensorExtLiteConstructionStep {
  llvm::StringRef operationName;
};

llvm::ArrayRef<TensorExtLiteConstructionStep>
getTensorExtLiteConstructionSteps();

struct TensorExtLiteArtifactRoute {
  llvm::StringRef routeID;
  llvm::StringRef emissionKind;
  llvm::StringRef artifactKind;
  llvm::StringRef runtimeABI;
  llvm::StringRef runtimeABIKind;
  llvm::StringRef runtimeABIName;
  llvm::StringRef runtimeGlueRole;
  llvm::StringRef loweringBoundaryOpName;
  llvm::StringRef headerRouteID;
  llvm::StringRef headerArtifactKind;
  llvm::StringRef bundleComponentGroup;
  llvm::StringRef objectHandoffKind;
  llvm::StringRef emitCToCppTranslateRouteID;
  llvm::StringRef configCallee;
  llvm::StringRef loadFragCallee;
  llvm::StringRef tileMmaCallee;
  llvm::StringRef storeFragCallee;
};

const TensorExtLiteArtifactRoute &getTensorExtLiteArtifactRoute();
llvm::ArrayRef<support::RuntimeABIParameter>
getTensorExtLiteRuntimeABIParameters();

} // namespace weft::plugin::tensorext_lite

#endif // WEFT_PLUGIN_TENSOREXTLITE_TENSOREXTLITEFAMILYCONTRACT_H
