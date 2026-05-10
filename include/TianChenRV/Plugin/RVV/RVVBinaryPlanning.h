#ifndef TIANCHENRV_PLUGIN_RVV_RVVBINARYPLANNING_H
#define TIANCHENRV_PLUGIN_RVV_RVVBINARYPLANNING_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Target/RVV/RVVBinaryDescriptor.h"
#include "TianChenRV/Target/RVV/RVVVectorShape.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
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

struct RVVBinaryEmissionIdentity {
  const target::rvv::RVVBinaryFamilyDescriptor *family = nullptr;
  std::string emissionPath;
  std::string supportedMessage;

  llvm::StringRef getFamilyID() const;
  llvm::StringRef getEmissionKind() const;
  llvm::StringRef getEmissionPath() const;
  llvm::StringRef getRouteID() const;
  llvm::StringRef getArtifactKind() const;
  llvm::StringRef getRuntimeABI() const;
  llvm::StringRef getRuntimeABIKind() const;
  llvm::StringRef getRuntimeABIName() const;
  llvm::StringRef getRuntimeGlueRole() const;
  llvm::StringRef getSupportedMessage() const;
};

struct RVVBinaryCapabilityPropertyView {
  std::string architecture;
  std::string isaVectorHints;
  std::string selectedMarch;
  std::uint64_t hartCount = 0;
  std::optional<std::uint64_t> vlenbBytes;
  std::optional<std::uint64_t> i32M1LaneCount;
  const target::rvv::RVVVectorShapeConfig *selectedShape = nullptr;
};

struct RVVBinarySelectedConfig {
  const target::rvv::RVVVectorShapeConfig *shape = nullptr;

  bool isValid() const { return shape != nullptr; }
  const target::rvv::RVVVectorShapeConfig &getShape() const {
    return *shape;
  }
  llvm::StringRef getShapeID() const { return shape ? shape->shapeID : ""; }
  llvm::StringRef getDTypeID() const { return shape ? shape->dtypeID : ""; }
  std::int64_t getSEWBits() const { return shape ? shape->sewBits : 0; }
  llvm::StringRef getLMUL() const { return shape ? shape->lmul : ""; }
  llvm::StringRef getTailPolicy() const {
    return shape ? shape->tailPolicy : "";
  }
  llvm::StringRef getMaskPolicy() const {
    return shape ? shape->maskPolicy : "";
  }
  llvm::StringRef getVectorType() const {
    return shape ? shape->vectorType : "";
  }
  llvm::StringRef getVectorSuffix() const {
    return shape ? shape->vectorSuffix : "";
  }
  llvm::StringRef getSetVLSuffix() const {
    return shape ? shape->setvlSuffix : "";
  }
  llvm::SmallVector<llvm::StringRef, 4> getCapabilityIDs() const {
    llvm::SmallVector<llvm::StringRef, 4> ids;
    if (!shape)
      return ids;
    ids.push_back(shape->sewCapabilityID);
    ids.push_back(shape->lmulCapabilityID);
    ids.push_back(shape->tailPolicyCapabilityID);
    ids.push_back(shape->maskPolicyCapabilityID);
    return ids;
  }
};

struct RVVBinarySelectedPlan {
  target::rvv::RVVBinaryIntrinsicDescriptor descriptor;
  const target::rvv::RVVBinaryFamilyDescriptor *family = nullptr;
  RVVBinarySelectedConfig selectedConfig;
  std::int64_t elementCount = 0;
  std::string requiredMarch;
  std::optional<std::string> selectedMABI;
  std::string emissionPath;
  std::string supportedMessage;

  llvm::StringRef getFamilyID() const;
  llvm::StringRef getDTypeID() const;
  llvm::StringRef getLoweringDescriptor() const;
  llvm::StringRef getMicrokernelOpName() const;
  llvm::StringRef getArithmeticOpName() const;
  llvm::StringRef getEmissionKind() const;
  llvm::StringRef getEmissionPath() const;
  llvm::StringRef getRouteID() const;
  llvm::StringRef getArtifactKind() const;
  llvm::StringRef getRuntimeABI() const;
  llvm::StringRef getRuntimeABIKind() const;
  llvm::StringRef getRuntimeABIName() const;
  llvm::StringRef getRuntimeGlueRole() const;
  llvm::StringRef getSupportedMessage() const;
  const RVVBinarySelectedConfig &getSelectedConfig() const {
    return selectedConfig;
  }
  const target::rvv::RVVVectorShapeConfig &getShape() const {
    return selectedConfig.getShape();
  }
  std::string getSetVLIntrinsicName() const;
  std::string getLoadIntrinsicName() const;
  std::string getArithmeticIntrinsicName() const;
  std::string getStoreIntrinsicName() const;
};

struct RVVBinaryProposalPlan {
  RVVBinarySelectedPlan selectedPlan;
  RVVBinaryCapabilityPropertyView capabilityView;
  llvm::SmallVector<std::string, 5> requiredCapabilityIDs;
  std::string condition;
  std::string guard;
  std::string policy;

  llvm::StringRef getFamilyID() const;
  llvm::StringRef getDTypeID() const;
  llvm::StringRef getLoweringDescriptor() const;
  const target::rvv::RVVBinaryFamilyDescriptor &getFamily() const;
  const target::rvv::RVVVectorShapeConfig &getSelectedShape() const;
  llvm::ArrayRef<std::string> getRequiredCapabilityIDs() const;
  llvm::StringRef getCondition() const;
  llvm::StringRef getGuard() const;
  llvm::StringRef getPolicy() const;
  bool hasCapacityMetadata() const;
};

const RVVSelectedVectorShapeMetadataNames &
getRVVVariantSelectedVectorShapeMetadataNames();

const RVVSelectedVectorShapeMetadataNames &
getRVVBoundarySelectedVectorShapeMetadataNames();

llvm::StringRef getRVVBinaryRuntimeCallableCSourceArtifactKind();

llvm::Expected<RVVBinaryEmissionIdentity> buildRVVBinaryEmissionIdentity(
    const target::rvv::RVVBinaryFamilyDescriptor &family);

std::string formatRVVBinaryFamilyFrontendLoweringList();

llvm::Expected<RVVBinaryCapabilityPropertyView>
buildRVVBinaryCapabilityPropertyView(
    const support::TargetCapabilitySet &capabilities,
    const target::rvv::RVVVectorShapeConfig *requiredShape = nullptr);

llvm::Expected<const target::rvv::RVVVectorShapeConfig *>
getRVVBinaryVariantRequiredShapeConfig(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities);

llvm::Error verifyRVVBinaryVariantRequiresCapabilityID(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities, llvm::StringRef id);

llvm::Expected<RVVBinarySelectedPlan> buildRVVBinarySelectedPlan(
    const target::rvv::RVVBinaryFamilyDescriptor &family,
    const target::rvv::RVVVectorShapeConfig &shape,
    std::int64_t elementCount, llvm::StringRef requiredMarch,
    std::optional<std::string> selectedMABI = std::nullopt);

llvm::Expected<RVVBinaryProposalPlan> buildRVVBinaryProposalPlan(
    const support::TargetCapabilitySet &capabilities,
    llvm::StringRef frontendLowering = llvm::StringRef(),
    llvm::StringRef diagnosticContext = "RVV binary proposal");

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
