#ifndef TIANCHENRV_PLUGIN_RVV_RVVBINARYPLANNING_H
#define TIANCHENRV_PLUGIN_RVV_RVVBINARYPLANNING_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Target/RVV/RVVBinaryDescriptor.h"
#include "TianChenRV/Target/RVV/RVVSelectedConfigContract.h"
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
  target::rvv::RVVBinarySelectedConfigContract contract;

  bool isValid() const { return contract.isValid(); }
  const target::rvv::RVVVectorShapeConfig &getShape() const {
    return contract.getShape();
  }
  const target::rvv::RVVBinarySelectedConfigContract &getContract() const {
    return contract;
  }
  target::rvv::RVVBinarySelectedConfigContract &getContract() {
    return contract;
  }
  llvm::StringRef getShapeID() const { return contract.getShapeID(); }
  llvm::StringRef getDTypeID() const { return contract.getDTypeID(); }
  std::int64_t getSEWBits() const { return contract.getSEWBits(); }
  llvm::StringRef getLMUL() const { return contract.getLMUL(); }
  llvm::StringRef getTailPolicy() const {
    return contract.getTailPolicy();
  }
  llvm::StringRef getMaskPolicy() const {
    return contract.getMaskPolicy();
  }
  llvm::StringRef getVectorType() const {
    return contract.getVectorType();
  }
  llvm::StringRef getVectorSuffix() const {
    return contract.getVectorSuffix();
  }
  llvm::StringRef getSetVLSuffix() const {
    return contract.getSetVLSuffix();
  }
  llvm::SmallVector<llvm::StringRef, 4> getCapabilityIDs() const {
    return contract.getSelectedShapeCapabilityIDs();
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
  bool attachLoweringDescriptorAttr = true;

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
  bool shouldAttachLoweringDescriptorAttr() const {
    return attachLoweringDescriptorAttr;
  }
};

struct RVVBinaryFamilyPlanningResolution {
  const target::rvv::RVVBinaryFamilyDescriptor *family = nullptr;
  const target::rvv::RVVVectorShapeConfig *directSelectedShape = nullptr;
  std::string sourceKind;

  bool isValid() const { return family != nullptr; }
  llvm::StringRef getFamilyID() const {
    return family ? family->familyID : llvm::StringRef();
  }
  llvm::StringRef getFrontendLowering() const {
    return family ? family->frontendLowering : llvm::StringRef();
  }
  llvm::StringRef getLoweringDescriptor() const {
    return family ? family->loweringDescriptor : llvm::StringRef();
  }
  llvm::StringRef getSourceKind() const { return sourceKind; }
  llvm::StringRef getDirectSelectedShapeID() const {
    return directSelectedShape ? directSelectedShape->shapeID
                               : llvm::StringRef();
  }
  llvm::SmallVector<llvm::StringRef, 4>
  getDirectSelectedCapabilityIDs() const {
    llvm::SmallVector<llvm::StringRef, 4> ids;
    if (!directSelectedShape)
      return ids;
    ids.push_back(directSelectedShape->sewCapabilityID);
    ids.push_back(directSelectedShape->lmulCapabilityID);
    ids.push_back(directSelectedShape->tailPolicyCapabilityID);
    ids.push_back(directSelectedShape->maskPolicyCapabilityID);
    return ids;
  }
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

llvm::Expected<RVVBinaryFamilyPlanningResolution>
resolveRVVBinaryFamilyForProposal(
    tcrv::exec::KernelOp kernel,
    llvm::StringRef diagnosticContext = "RVV binary proposal");

llvm::Expected<RVVBinaryProposalPlan> buildRVVBinaryProposalPlan(
    const support::TargetCapabilitySet &capabilities,
    const target::rvv::RVVBinaryFamilyDescriptor &family,
    llvm::StringRef diagnosticContext = "RVV binary proposal");

llvm::Expected<RVVBinaryProposalPlan> buildRVVBinaryProposalPlan(
    const support::TargetCapabilitySet &capabilities,
    llvm::StringRef frontendLowering = llvm::StringRef(),
    llvm::StringRef diagnosticContext = "RVV binary proposal");

llvm::Expected<RVVBinaryProposalPlan> buildRVVBinaryProposalPlan(
    const support::TargetCapabilitySet &capabilities,
    tcrv::exec::KernelOp kernel,
    llvm::StringRef diagnosticContext = "RVV binary proposal");

llvm::Expected<std::optional<RVVBinarySelectedPlan>>
buildRVVBinarySelectedPlanFromVariant(
    tcrv::exec::VariantOp variant,
    const target::rvv::RVVVectorShapeConfig &shape,
    llvm::StringRef expectedDTypeID = llvm::StringRef(),
    std::optional<std::string> selectedMABI = std::nullopt);

llvm::Expected<RVVBinarySelectedPlan>
buildRVVBinarySelectedPlanFromTypedFamilyVariant(
    tcrv::exec::VariantOp variant,
    const target::rvv::RVVBinaryFamilyDescriptor &family,
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
