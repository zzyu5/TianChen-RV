#ifndef WEFT_PLUGIN_RVV_RVVARTIFACTCONTRACT_H
#define WEFT_PLUGIN_RVV_RVVARTIFACTCONTRACT_H

#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Support/RuntimeABI.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace weft::plugin::rvv {

/// Artifact-only contract derived from one exact, construction-qualified RVV
/// typed body. The ordered ABI parameters are the RuntimeABIValueOp leaves in
/// the body's actual SSA dependency closure. No kind/format/capability or route
/// provider is consulted, and no compute decision is reconstructed here.
struct RVVArtifactContract {
  llvm::SmallVector<weft::support::RuntimeABIParameter, 8> runtimeABIParameters;
};

llvm::StringRef getRVVExactBodyArtifactRouteID();
llvm::StringRef getRVVExactBodyArtifactKind();
llvm::StringRef getRVVExactBodyEmissionKind();
llvm::StringRef getRVVExactBodyLoweringBoundaryOpName();
llvm::StringRef getRVVExactBodyRuntimeABIKind();
llvm::StringRef getRVVExactBodyRuntimeABIName();
llvm::StringRef getRVVExactBodyRuntimeGlueRole();
llvm::StringRef getRVVExactBodyHeaderRouteID();
llvm::StringRef getRVVExactBodyHeaderArtifactKind();
llvm::StringRef getRVVExactBodyBundleComponentGroup();
llvm::StringRef getRVVExactBodyObjectHandoffKind();
llvm::StringRef getRVVExactBodyTranslateRouteID();

llvm::Expected<RVVArtifactContract>
deriveRVVArtifactContract(weft::rvv::WithVLOp body);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVARTIFACTCONTRACT_H
