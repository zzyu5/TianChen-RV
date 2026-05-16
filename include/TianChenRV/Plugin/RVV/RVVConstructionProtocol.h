#ifndef TIANCHENRV_PLUGIN_RVV_RVVCONSTRUCTIONPROTOCOL_H
#define TIANCHENRV_PLUGIN_RVV_RVVCONSTRUCTIONPROTOCOL_H

#include "TianChenRV/Plugin/ConstructionProtocol.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace tianchenrv::plugin::rvv {

using RVVConstructionSemanticRole =
    tianchenrv::plugin::construction::SemanticRole;
using RVVConstructionFamilyDeclaration =
    tianchenrv::plugin::construction::FamilyDeclaration;
using RVVConstructionEmitCMapping =
    tianchenrv::plugin::construction::EmitCMapping;
using RVVConstructionManifest = tianchenrv::plugin::construction::Manifest;
using RVVTypedRoleInterfaceRealization =
    tianchenrv::plugin::construction::TypedRoleInterfaceRealization;
using RVVTypedRoleGraphRealization =
    tianchenrv::plugin::construction::TypedRoleGraphRealization;

struct RVVI32M1ArithmeticConstructionRoute {
  llvm::StringRef mnemonic;
  llvm::StringRef operationName;
  llvm::StringRef typedRoleID;
  llvm::StringRef emitCRouteID;
  llvm::StringRef runtimeABIName;
  llvm::StringRef runtimeABIContractName;
};

llvm::StringRef getRVVConstructionProtocolVersion();
llvm::StringRef getRVVConstructionArchetype();
llvm::StringRef getRVVConstructionSemanticRoleGraph();
llvm::StringRef getRVVConstructionInterfaceRealization();
llvm::StringRef getRVVTypedRoleRealizationSummary();
llvm::StringRef getRVVConstructionEvidenceProfile();

const RVVConstructionManifest &getRVVConstructionManifest();
const RVVTypedRoleGraphRealization &getRVVTypedRoleGraphRealization();
llvm::ArrayRef<RVVI32M1ArithmeticConstructionRoute>
getRVVI32M1ArithmeticConstructionRoutes();

llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVI32M1ArithmeticConstructionRuntimeABIParameters();

llvm::Error verifyRVVConstructionManifest(
    const RVVConstructionManifest &manifest);
llvm::Error verifyRVVTypedRoleGraphRealization(
    const RVVConstructionManifest &manifest,
    const RVVTypedRoleGraphRealization &realization);
llvm::Error verifyRVVConstructionProtocolReady();
llvm::Error verifyRVVI32M1ArithmeticConstructionRuntimeABIParameters(
    llvm::ArrayRef<tianchenrv::support::RuntimeABIParameter> parameters);

llvm::Expected<const RVVI32M1ArithmeticConstructionRoute *>
lookupRVVI32M1ArithmeticConstructionRouteByMnemonic(llvm::StringRef mnemonic);
llvm::Expected<const RVVI32M1ArithmeticConstructionRoute *>
lookupRVVI32M1ArithmeticConstructionRouteByOperationName(
    llvm::StringRef operationName);
llvm::Expected<const RVVI32M1ArithmeticConstructionRoute *>
lookupRVVI32M1ArithmeticConstructionRouteByEmitCRouteID(
    llvm::StringRef emitCRouteID);

llvm::Error verifyRVVRoleOperationInterface(mlir::Operation *roleOp,
                                            llvm::StringRef role);
llvm::Error verifyRVVRuntimeABIValueRoleOpInterface(mlir::Operation *roleOp);
llvm::Error verifyRVVSetVLRoleOpInterface(mlir::Operation *roleOp);
llvm::Error verifyRVVWithVLRoleOpInterface(mlir::Operation *roleOp);
llvm::Error verifyRVVLoadRoleOpInterface(mlir::Operation *roleOp);
llvm::Error verifyRVVArithmeticRoleOpInterface(mlir::Operation *roleOp);
llvm::Error verifyRVVStoreRoleOpInterface(mlir::Operation *roleOp);

llvm::Error verifyRVVI32M1ArithmeticConstructionRouteMapping(
    llvm::StringRef mnemonic, llvm::StringRef operationName,
    llvm::StringRef emitCRouteID, llvm::StringRef runtimeABIName);
llvm::Error verifyRVVI32M1ArithmeticConstructionPlanMapping(
    llvm::StringRef emitCRouteID, llvm::StringRef runtimeABIName,
    llvm::StringRef emissionKind,
    llvm::StringRef loweringBoundaryOpName, llvm::StringRef runtimeABIKind,
    llvm::StringRef runtimeGlueRole);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVCONSTRUCTIONPROTOCOL_H
