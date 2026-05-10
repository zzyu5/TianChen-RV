#ifndef TIANCHENRV_PLUGIN_RVV_RVVBINARYPLANNING_H
#define TIANCHENRV_PLUGIN_RVV_RVVBINARYPLANNING_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Target/RVV/RVVBinaryDescriptor.h"
#include "TianChenRV/Target/RVV/RVVVectorShape.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>

namespace tianchenrv::plugin::rvv {

struct RVVSelectedVectorShapeMetadataNames {
  llvm::StringRef shape;
  llvm::StringRef sew;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef vectorType;
  llvm::StringRef vectorSuffix;
  llvm::StringRef setvlSuffix;
};

struct RVVBinarySelectedPlan {
  target::rvv::RVVBinaryIntrinsicDescriptor descriptor;
  const target::rvv::RVVBinaryFamilyDescriptor *family = nullptr;
  const target::rvv::RVVVectorShapeConfig *shape = nullptr;
  std::int64_t elementCount = 0;
  std::string requiredMarch;
  std::optional<std::string> selectedMABI;
  std::string emissionPath;

  llvm::StringRef getFamilyID() const;
  llvm::StringRef getDTypeID() const;
  llvm::StringRef getLoweringDescriptor() const;
  llvm::StringRef getMicrokernelOpName() const;
  llvm::StringRef getArithmeticOpName() const;
  llvm::StringRef getEmissionKind() const;
  llvm::StringRef getEmissionPath() const;
  llvm::StringRef getRouteID() const;
  llvm::StringRef getRuntimeABI() const;
  llvm::StringRef getRuntimeABIKind() const;
  llvm::StringRef getRuntimeABIName() const;
  llvm::StringRef getRuntimeGlueRole() const;
  std::string getSetVLIntrinsicName() const;
  std::string getLoadIntrinsicName() const;
  std::string getArithmeticIntrinsicName() const;
  std::string getStoreIntrinsicName() const;
};

const RVVSelectedVectorShapeMetadataNames &
getRVVVariantSelectedVectorShapeMetadataNames();

const RVVSelectedVectorShapeMetadataNames &
getRVVBoundarySelectedVectorShapeMetadataNames();

llvm::Expected<RVVBinarySelectedPlan> buildRVVBinarySelectedPlan(
    const target::rvv::RVVBinaryFamilyDescriptor &family,
    const target::rvv::RVVVectorShapeConfig &shape,
    std::int64_t elementCount, llvm::StringRef requiredMarch,
    std::optional<std::string> selectedMABI = std::nullopt);

llvm::Expected<std::optional<RVVBinarySelectedPlan>>
buildRVVBinarySelectedPlanFromVariant(
    tcrv::exec::VariantOp variant,
    const target::rvv::RVVVectorShapeConfig &shape,
    llvm::StringRef expectedDTypeID = llvm::StringRef(),
    std::optional<std::string> selectedMABI = std::nullopt);

void addRVVSelectedVectorShapeMetadataToProposal(
    VariantProposal &proposal, mlir::MLIRContext *context,
    const target::rvv::RVVVectorShapeConfig &shape);

void addRVVSelectedVectorShapeMetadataToOperationState(
    mlir::OperationState &state, mlir::MLIRContext *context,
    const target::rvv::RVVVectorShapeConfig &shape);

llvm::Error addRVVSelectedVectorShapeMetadataToPlan(
    VariantEmissionPlan &plan, tcrv::exec::VariantOp variant,
    const target::rvv::RVVVectorShapeConfig &shape);

llvm::Error validateRVVSelectedVectorShapeMetadata(
    mlir::Operation *op, llvm::StringRef context,
    const target::rvv::RVVVectorShapeConfig &shape,
    const RVVSelectedVectorShapeMetadataNames &names);

bool hasAnyRVVSelectedVectorShapeMetadata(
    mlir::Operation *op, const RVVSelectedVectorShapeMetadataNames &names);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVBINARYPLANNING_H
