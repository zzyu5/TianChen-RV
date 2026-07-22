#ifndef WEFT_PLUGIN_TEMPLATE_TEMPLATEFAMILYCONTRACT_H
#define WEFT_PLUGIN_TEMPLATE_TEMPLATEFAMILYCONTRACT_H

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

namespace weft::plugin::template_ext {

llvm::Error verifyTemplateSelectedVariantLegality(
    weft::exec::VariantOp variant, weft::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities);

struct TemplateArtifactRoute {
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

const TemplateArtifactRoute &getTemplateArtifactRoute();
llvm::ArrayRef<support::RuntimeABIParameter> getTemplateRuntimeABIParameters();

} // namespace weft::plugin::template_ext

#endif // WEFT_PLUGIN_TEMPLATE_TEMPLATEFAMILYCONTRACT_H
