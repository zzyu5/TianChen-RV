#include "TianChenRV/Target/RVV/RVVTargetArtifactRouteFamilyValidation.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCElementwiseRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace tianchenrv::target::rvv {
namespace {

constexpr llvm::StringLiteral kRVVProductReductionOutCarryBoundary(
    "scalar-result-out0-seeded-before-loop-and-carried-across-runtime-vl-chunks.v1");
constexpr llvm::StringLiteral
    kRVVProductReductionDequantVectorCarryBoundary(
        "vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1");

llvm::Error makeRVVTargetRouteError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV materialized EmitC target artifact bridge "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::StringRef lookupCandidateMetadataValue(
    const TargetArtifactCandidate &candidate, llvm::StringRef key) {
  for (const support::ArtifactMetadataEntry &entry :
       candidate.artifactMetadata)
    if (entry.key == key)
      return entry.value;
  return {};
}

llvm::Error validateRVVRuntimeAVLVLSelectedBoundaryContract(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVRuntimeAVLVLSelectedBoundaryContract &contract);

llvm::Error requireRVVRouteLocalRuntimeAVLVLMirror(
    llvm::StringRef consumerLabel, llvm::StringRef label,
    llvm::StringRef mirror,
    llvm::StringRef selectedBoundaryAuthority) {
  if (mirror == selectedBoundaryAuthority)
    return llvm::Error::success();
  if (selectedBoundaryAuthority.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-owned runtime AVL/VL selected-boundary " + label +
        " before checking route-local mirrors");
  if (mirror.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires route-local runtime AVL/VL mirror " + label + " '" +
        selectedBoundaryAuthority +
        "' after selected-boundary validation");
  return makeRVVTargetRouteError(
      llvm::Twine(consumerLabel) +
      " rejects stale route-local runtime AVL/VL mirror " + label + " '" +
      mirror + "'; selected-boundary contract requires '" +
      selectedBoundaryAuthority + "'");
}

llvm::Error validateRVVRouteLocalRuntimeAVLVLMirrors(
    llvm::StringRef consumerLabel,
    const plugin::rvv::RVVRuntimeAVLVLSelectedBoundaryContract &authority,
    llvm::StringRef runtimeControlPlanIDMirror,
    llvm::StringRef runtimeABIOrderMirror,
    llvm::StringRef setVLIntrinsicMirror, llvm::StringRef vlCTypeMirror,
    llvm::StringRef emitCFullChunkVLNameMirror,
    llvm::StringRef emitCLoopVLNameMirror,
    llvm::StringRef emitCLoopInductionNameMirror) {
  if (llvm::Error error = requireRVVRouteLocalRuntimeAVLVLMirror(
          consumerLabel, "runtime control plan", runtimeControlPlanIDMirror,
          authority.runtimeControlPlanID))
    return error;
  if (llvm::Error error = requireRVVRouteLocalRuntimeAVLVLMirror(
          consumerLabel, "runtime ABI order", runtimeABIOrderMirror,
          authority.runtimeABIOrder))
    return error;
  if (llvm::Error error = requireRVVRouteLocalRuntimeAVLVLMirror(
          consumerLabel, "setvl callee", setVLIntrinsicMirror,
          authority.setVLIntrinsic))
    return error;
  if (llvm::Error error = requireRVVRouteLocalRuntimeAVLVLMirror(
          consumerLabel, "VL C type", vlCTypeMirror, authority.vlCType))
    return error;
  if (llvm::Error error = requireRVVRouteLocalRuntimeAVLVLMirror(
          consumerLabel, "EmitC full-chunk VL", emitCFullChunkVLNameMirror,
          authority.emitCFullChunkVLName))
    return error;
  if (llvm::Error error = requireRVVRouteLocalRuntimeAVLVLMirror(
          consumerLabel, "EmitC loop VL", emitCLoopVLNameMirror,
          authority.emitCLoopVLName))
    return error;
  if (llvm::Error error = requireRVVRouteLocalRuntimeAVLVLMirror(
          consumerLabel, "EmitC loop induction",
          emitCLoopInductionNameMirror, authority.emitCLoopInductionName))
    return error;
  return llvm::Error::success();
}

llvm::Error requireCandidateMetadataMirror(
    const TargetArtifactCandidate &candidate, llvm::StringRef key,
    llvm::StringRef expected, llvm::StringRef label) {
  llvm::StringRef actual = lookupCandidateMetadataValue(candidate, key);
  if (!expected.empty()) {
    if (actual.empty())
      return makeRVVTargetRouteError(llvm::Twine("candidate metadata must "
                                                 "carry ") +
                                     key + " provenance for " + label);
    if (actual != expected)
      return makeRVVTargetRouteError(
          llvm::Twine("candidate ") + key + " provenance must mirror " +
          label + " '" + expected + "' but was '" + actual + "'");
  } else if (!actual.empty()) {
    return makeRVVTargetRouteError(
        llvm::Twine("candidate metadata must not carry ") + key +
        " mirrors for a selected typed RVV body route without " + label);
  }

  return llvm::Error::success();
}

llvm::Error validateRVVMemoryRouteMetadataMirrorContract(
    const TargetArtifactCandidate &candidate,
    llvm::ArrayRef<plugin::rvv::RVVMemoryRouteMetadataMirrorContract>
        mirrors) {
  for (const plugin::rvv::RVVMemoryRouteMetadataMirrorContract &mirror :
       mirrors)
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, mirror.key, mirror.expected, mirror.label))
      return error;
  return llvm::Error::success();
}

llvm::Error validateRVVMemoryRouteEmptyMetadataMirrors(
    const TargetArtifactCandidate &candidate,
    llvm::ArrayRef<llvm::StringRef> staleMirrorKeys, llvm::StringRef label) {
  for (llvm::StringRef key : staleMirrorKeys)
    if (llvm::Error error =
            requireCandidateMetadataMirror(candidate, key, "", label))
      return error;
  return llvm::Error::success();
}

llvm::Error validateRVVProviderMemoryRouteMetadataMirrorContract(
    const TargetArtifactCandidate &candidate,
    const plugin::rvv::RVVMemoryRouteMetadataMirrorContractSet &contract) {
  if (llvm::Error error = validateRVVMemoryRouteMetadataMirrorContract(
          candidate, contract.mirrors))
    return error;
  return validateRVVMemoryRouteEmptyMetadataMirrors(
      candidate, contract.staleMirrorKeys, contract.staleMirrorLabel);
}

llvm::Error validateRVVMAccRouteMetadataMirrorContract(
    const TargetArtifactCandidate &candidate,
    llvm::ArrayRef<plugin::rvv::RVVMAccRouteMetadataMirrorContract> mirrors) {
  for (const plugin::rvv::RVVMAccRouteMetadataMirrorContract &mirror :
       mirrors)
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, mirror.key, mirror.expected, mirror.label))
      return error;
  return llvm::Error::success();
}

llvm::Error validateRVVMAccRouteEmptyMetadataMirrors(
    const TargetArtifactCandidate &candidate,
    llvm::ArrayRef<llvm::StringRef> staleMirrorKeys, llvm::StringRef label) {
  for (llvm::StringRef key : staleMirrorKeys)
    if (llvm::Error error =
            requireCandidateMetadataMirror(candidate, key, "", label))
      return error;
  return llvm::Error::success();
}

llvm::Error validateRVVProviderMAccRouteMetadataMirrorContract(
    const TargetArtifactCandidate &candidate,
    const plugin::rvv::RVVMAccRouteMetadataMirrorContractSet &contract) {
  if (llvm::Error error =
          validateRVVMAccRouteMetadataMirrorContract(candidate,
                                                     contract.mirrors))
    return error;
  return validateRVVMAccRouteEmptyMetadataMirrors(
      candidate, contract.staleMirrorKeys, contract.staleMirrorLabel);
}

llvm::Error validateRVVStandaloneReductionRouteMetadataMirrorContract(
    const TargetArtifactCandidate &candidate,
    llvm::ArrayRef<
        plugin::rvv::RVVStandaloneReductionRouteMetadataMirrorContract>
        mirrors) {
  for (const plugin::rvv::RVVStandaloneReductionRouteMetadataMirrorContract
           &mirror : mirrors)
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, mirror.key, mirror.expected, mirror.label))
      return error;
  return llvm::Error::success();
}

llvm::Error validateRVVStandaloneReductionRouteEmptyMetadataMirrors(
    const TargetArtifactCandidate &candidate,
    llvm::ArrayRef<llvm::StringRef> staleMirrorKeys, llvm::StringRef label) {
  for (llvm::StringRef key : staleMirrorKeys)
    if (llvm::Error error =
            requireCandidateMetadataMirror(candidate, key, "", label))
      return error;
  return llvm::Error::success();
}

llvm::Error validateRVVProviderStandaloneReductionRouteMetadataMirrorContract(
    const TargetArtifactCandidate &candidate,
    const plugin::rvv::RVVStandaloneReductionRouteMetadataMirrorContractSet
        &contract) {
  if (llvm::Error error =
          validateRVVStandaloneReductionRouteMetadataMirrorContract(
              candidate, contract.mirrors))
    return error;
  return validateRVVStandaloneReductionRouteEmptyMetadataMirrors(
      candidate, contract.staleMirrorKeys, contract.staleMirrorLabel);
}

llvm::Error validateRVVCompareSelectRouteMetadataMirrorContract(
    const TargetArtifactCandidate &candidate,
    llvm::ArrayRef<plugin::rvv::RVVCompareSelectRouteMetadataMirrorContract>
        mirrors) {
  for (const plugin::rvv::RVVCompareSelectRouteMetadataMirrorContract &mirror :
       mirrors)
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, mirror.key, mirror.expected, mirror.label))
      return error;
  return llvm::Error::success();
}

llvm::Error validateRVVCompareSelectRouteEmptyMetadataMirrors(
    const TargetArtifactCandidate &candidate,
    llvm::ArrayRef<llvm::StringRef> staleMirrorKeys, llvm::StringRef label) {
  for (llvm::StringRef key : staleMirrorKeys)
    if (llvm::Error error =
            requireCandidateMetadataMirror(candidate, key, "", label))
      return error;
  return llvm::Error::success();
}

llvm::Error validateRVVProviderCompareSelectRouteMetadataMirrorContract(
    const TargetArtifactCandidate &candidate,
    const plugin::rvv::RVVCompareSelectRouteMetadataMirrorContractSet
        &contract) {
  if (llvm::Error error =
          validateRVVCompareSelectRouteMetadataMirrorContract(
              candidate, contract.mirrors))
    return error;
  return validateRVVCompareSelectRouteEmptyMetadataMirrors(
      candidate, contract.staleMirrorKeys, contract.staleMirrorLabel);
}

llvm::Error validateRVVConversionDtypePolicyRouteMetadataMirrorContract(
    const TargetArtifactCandidate &candidate,
    llvm::ArrayRef<
        plugin::rvv::RVVConversionDtypePolicyRouteMetadataMirrorContract>
        mirrors) {
  for (const plugin::rvv::RVVConversionDtypePolicyRouteMetadataMirrorContract
           &mirror : mirrors)
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, mirror.key, mirror.expected, mirror.label))
      return error;
  return llvm::Error::success();
}

llvm::Error validateRVVConversionDtypePolicyRouteEmptyMetadataMirrors(
    const TargetArtifactCandidate &candidate,
    llvm::ArrayRef<llvm::StringRef> staleMirrorKeys, llvm::StringRef label) {
  for (llvm::StringRef key : staleMirrorKeys)
    if (llvm::Error error =
            requireCandidateMetadataMirror(candidate, key, "", label))
      return error;
  return llvm::Error::success();
}

llvm::Error
validateRVVProviderConversionDtypePolicyRouteMetadataMirrorContract(
    const TargetArtifactCandidate &candidate,
    const plugin::rvv::RVVConversionDtypePolicyRouteMetadataMirrorContractSet
        &contract) {
  if (llvm::Error error =
          validateRVVConversionDtypePolicyRouteMetadataMirrorContract(
              candidate, contract.mirrors))
    return error;
  return validateRVVConversionDtypePolicyRouteEmptyMetadataMirrors(
      candidate, contract.staleMirrorKeys, contract.staleMirrorLabel);
}

llvm::Error requireRVVProviderDerivedField(llvm::StringRef consumerLabel,
                                           llvm::StringRef fieldLabel,
                                           llvm::StringRef value) {
  if (!value.empty())
    return llvm::Error::success();
  return makeRVVTargetRouteError(llvm::Twine(consumerLabel) +
                                 " requires provider-derived " + fieldLabel +
                                 " before artifact export");
}

llvm::Error requireRVVProviderCanonicalField(llvm::StringRef consumerLabel,
                                             llvm::StringRef fieldLabel,
                                             llvm::StringRef actual,
                                             llvm::StringRef expected) {
  if (llvm::Error error =
          requireRVVProviderDerivedField(consumerLabel, fieldLabel, actual))
    return error;
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVTargetRouteError(llvm::Twine(consumerLabel) +
                                 " requires provider-derived " + fieldLabel +
                                 " '" + expected + "' but was '" + actual +
                                 "'");
}

bool isRVVWideningDotReductionRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::WideningProductReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      WideningProductReduceDequantizeF32:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      WideningProductReduceDequantClampF32:
  case plugin::rvv::RVVSelectedBodyOperationKind::WideningDotReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      StridedInputWideningDotReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskWideningDotReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    return true;
  default:
    return false;
  }
}

bool isRVVComputedMaskWideningDotReductionRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation == plugin::rvv::RVVSelectedBodyOperationKind::
                          ComputedMaskWideningDotReduceAdd ||
         operation == plugin::rvv::RVVSelectedBodyOperationKind::
                          ComputedMaskStridedInputWideningDotReduceAdd;
}

bool isRVVNonComputedMaskWideningDotReductionRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 WideningProductReduceAdd ||
         operation == plugin::rvv::RVVSelectedBodyOperationKind::
                          WideningProductReduceDequantizeF32 ||
         operation == plugin::rvv::RVVSelectedBodyOperationKind::
                          WideningProductReduceDequantClampF32 ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::WideningDotReduceAdd ||
         operation == plugin::rvv::RVVSelectedBodyOperationKind::
                          StridedInputWideningDotReduceAdd;
}

bool isRVVMAccRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::MAccAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskedMAccAdd:
    return true;
  default:
    return false;
  }
}

bool isRVVStandaloneReductionAccumulationRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::StandaloneReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::StandaloneReduceMin:
  case plugin::rvv::RVVSelectedBodyOperationKind::StandaloneReduceMax:
  case plugin::rvv::RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMin:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMax:
    return true;
  default:
    return false;
  }
}

bool isRVVComputedMaskStandaloneReductionRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMin:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMax:
    return true;
  default:
    return false;
  }
}

bool isRVVVectorComputedMaskStandaloneReductionRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax:
    return true;
  default:
    return false;
  }
}

bool isRVVRuntimeScalarComputedMaskStandaloneReductionRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMin:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMax:
    return true;
  default:
    return false;
  }
}

bool isRVVPlainStandaloneReductionRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::StandaloneReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::StandaloneReduceMin:
  case plugin::rvv::RVVSelectedBodyOperationKind::StandaloneReduceMax:
  case plugin::rvv::RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd:
    return true;
  default:
    return false;
  }
}

constexpr llvm::StringLiteral kRVVMaskTailPolicyRouteFamilyPlanID(
    "rvv-mask-tail-policy-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVComputedMaskSelectMaskTailPolicyOwner(
    "computed-mask select mask/tail policy");
constexpr llvm::StringLiteral kRVVComputedMaskMemoryMaskTailPolicyOwner(
    "computed-mask memory mask/tail policy");

bool isRVVCompareSelectMaskProducerRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::CmpSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarDualCompareMaskAndSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::F32ClampSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::DequantClampF32Epilogue:
    return true;
  default:
    return false;
  }
}

bool isRVVPlainCompareSelectMaskRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation == plugin::rvv::RVVSelectedBodyOperationKind::CmpSelect;
}

bool isRVVComputedMaskSelectRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarDualCompareMaskAndSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::F32ClampSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::DequantClampF32Epilogue:
    return true;
  default:
    return false;
  }
}

bool isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedGatherLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskLoadStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return true;
  default:
    return false;
  }
}

bool isRVVCompareSelectMaskRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return isRVVCompareSelectMaskProducerRouteFamilyOperation(operation) ||
         isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(operation);
}

bool isRVVScalarBroadcastElementwiseRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::ScalarBroadcastAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::ScalarBroadcastSub:
  case plugin::rvv::RVVSelectedBodyOperationKind::ScalarBroadcastMul:
    return true;
  default:
    return false;
  }
}

bool isRVVElementwiseArithmeticRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::Add:
  case plugin::rvv::RVVSelectedBodyOperationKind::Sub:
  case plugin::rvv::RVVSelectedBodyOperationKind::Mul:
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedSub:
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedMul:
  case plugin::rvv::RVVSelectedBodyOperationKind::StridedAdd:
    return true;
  default:
    return isRVVScalarBroadcastElementwiseRouteFamilyOperation(operation);
  }
}

bool isRVVMaskedElementwiseArithmeticRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedSub:
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedMul:
    return true;
  default:
    return false;
  }
}

bool isRVVStridedElementwiseArithmeticRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation == plugin::rvv::RVVSelectedBodyOperationKind::StridedAdd;
}

bool isRVVVectorReductionRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation == plugin::rvv::RVVSelectedBodyOperationKind::ReduceAdd;
}

bool isRVVWideningMAccContractionRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
         plugin::rvv::RVVSelectedBodyOperationKind::WideningMAccAdd;
}

bool isRVVRHSBroadcastLoadElementwiseArithmeticRouteFamilyDescription(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  switch (description.operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::Add:
  case plugin::rvv::RVVSelectedBodyOperationKind::Sub:
  case plugin::rvv::RVVSelectedBodyOperationKind::Mul:
    return description.memoryForm ==
           plugin::rvv::RVVSelectedBodyMemoryForm::RHSBroadcastLoad;
  default:
    return false;
  }
}

bool isRVVElementwiseArithmeticRouteFamilyDescription(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (!isRVVElementwiseArithmeticRouteFamilyOperation(description.operation))
    return false;
  if (isRVVStridedElementwiseArithmeticRouteFamilyOperation(
          description.operation))
    return description.memoryForm ==
           plugin::rvv::RVVSelectedBodyMemoryForm::StridedLoadStore;
  if (isRVVRHSBroadcastLoadElementwiseArithmeticRouteFamilyDescription(
          description))
    return true;
  if (isRVVScalarBroadcastElementwiseRouteFamilyOperation(
          description.operation))
    return description.memoryForm ==
           plugin::rvv::RVVSelectedBodyMemoryForm::RHSScalarBroadcast;
  return description.memoryForm ==
         plugin::rvv::RVVSelectedBodyMemoryForm::VectorRHSLoad;
}

bool isRVVConversionDtypePolicyWideningRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::WidenI32ToI64:
  case plugin::rvv::RVVSelectedBodyOperationKind::WidenI16ToI32:
    return true;
  default:
    return false;
  }
}

bool isRVVConversionDtypePolicyRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return isRVVConversionDtypePolicyWideningRouteFamilyOperation(operation) ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::DequantizeI32ToF32;
}

bool isRVVSegment2MemoryRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2LoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2LoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2StoreUnitLoad:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2UpdateUnitLoad:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      Segment2DeinterleaveUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      Segment2InterleaveUnitLoad:
    return true;
  default:
    return false;
  }
}

llvm::Error validateRVVUnitStrideMaskedMemoryCanonicalProviderFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description);

bool isRVVBaseMemoryMovementRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::StridedLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::UnitLoadStridedStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitStore:
    return true;
  default:
    return false;
  }
}

bool isRVVUnitStrideMaskedMemoryRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskLoadStore:
    return true;
  default:
    return false;
  }
}

llvm::Error requireRVVBaseMemoryMovementProviderField(
    llvm::StringRef label, llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  if (expected.empty())
    return makeRVVTargetRouteError(
        llvm::Twine("base-memory-movement target artifact consumer rejects "
                    "stale provider-derived ") +
        label + " fact '" + actual + "'");
  if (actual.empty())
    return makeRVVTargetRouteError(
        llvm::Twine("base-memory-movement target artifact consumer requires "
                    "provider-derived ") +
        label + " '" + expected + "' before artifact export");
  return makeRVVTargetRouteError(
      llvm::Twine("base-memory-movement target artifact consumer requires "
                  "provider-derived ") +
      label + " '" + expected + "' but was '" + actual + "'");
}

bool isRVVRuntimeScalarSplatStoreRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
         plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarSplatStore;
}

bool routeHasHeader(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    llvm::StringRef expectedHeader) {
  for (const conversion::emitc::TCRVEmitCHeaderRequirement &header :
       route.getHeaders())
    if (header.header == expectedHeader)
      return true;
  return false;
}

bool routeHasTypeMapping(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    llvm::StringRef sourceType, llvm::StringRef cType) {
  for (const conversion::emitc::TCRVEmitCTypeMapping &mapping :
       route.getTypeMappings())
    if (mapping.sourceType == sourceType && mapping.cType == cType)
      return true;
  return false;
}

bool runtimeABIParameterEquals(const support::RuntimeABIParameter &lhs,
                               const support::RuntimeABIParameter &rhs) {
  return lhs.cName == rhs.cName && lhs.cType == rhs.cType &&
         lhs.role == rhs.role && lhs.ownership == rhs.ownership;
}

bool stepHasResult(const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
                   llvm::StringRef resultName, llvm::StringRef cType) {
  return step.result && step.result->name == resultName &&
         step.result->cType == cType;
}

bool routeStepSourceIsSelectedRVVBody(
    const conversion::emitc::TCRVEmitCCallOpaqueStep &step) {
  return step.sourceOp.opInterface ==
             plugin::rvv::getRVVEmitCLowerableOpInterfaceName() &&
         !step.sourceOp.opName.empty() && !step.sourceOp.role.empty();
}

bool routeLocalVariableSourceIsSelectedRVVBody(
    const conversion::emitc::TCRVEmitCLocalVariable &variable) {
  return variable.sourceOp.opInterface ==
             plugin::rvv::getRVVEmitCLowerableOpInterfaceName() &&
         !variable.sourceOp.opName.empty() && !variable.sourceOp.role.empty();
}

bool routeAssignSourceIsSelectedRVVBody(
    const conversion::emitc::TCRVEmitCAssignStep &step) {
  return step.sourceOp.opInterface ==
             plugin::rvv::getRVVEmitCLowerableOpInterfaceName() &&
         !step.sourceOp.opName.empty() && !step.sourceOp.role.empty();
}

llvm::Error validateRVVRuntimeScalarSplatStoreRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVRuntimeScalarSplatStoreRouteValidationContract
        &contract) {
  if (contract.requiredHeaderDeclarations.empty() ||
      contract.requiredHeaders.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) + " requires "
        "provider-derived required_header_declarations before accepting the "
        "route artifact");

  for (llvm::StringRef header : contract.requiredHeaders) {
    if (header.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " saw an empty provider route header declaration");
    if (!routeHasHeader(route, header))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route header '" + header +
          "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVRuntimeScalarSplatStoreRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVRuntimeScalarSplatStoreRouteValidationContract
        &contract) {
  if (contract.vlCType.empty() || contract.vectorTypeName.empty() ||
      contract.vectorCType.empty() || contract.cTypeMappingSummary.empty() ||
      contract.typeMappings.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) + " requires "
        "provider-derived VL, vector, and C type mapping facts before "
        "artifact export");

  for (const plugin::rvv::RVVRuntimeScalarSplatStoreRouteTypeMappingContract
           &mapping : contract.typeMappings) {
    if (mapping.sourceType.empty() || mapping.cType.empty())
      return makeRVVTargetRouteError(llvm::Twine(contract.consumerLabel) +
                                     " requires provider-derived " +
                                     mapping.label +
                                     " facts before artifact export");
    if (!routeHasTypeMapping(route, mapping.sourceType, mapping.cType))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route type mapping '" +
          mapping.sourceType + "' -> '" + mapping.cType + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVRuntimeScalarSplatStoreRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVRuntimeScalarSplatStoreRouteValidationContract
        &contract) {
  if (contract.runtimeABIParameterRoles.size() !=
      contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned runtime ABI role order with one role per "
        "ABI parameter before artifact export");
  if (contract.runtimeABIOrder.empty() || contract.runtimeABIParameters.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) + " requires "
        "provider-derived runtime ABI order and ABI parameters before "
        "artifact export");
  if (route.getABIMappings().size() != contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires rebuilt provider route ABI mapping count " +
        llvm::Twine(contract.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < contract.runtimeABIParameters.size();
       ++index) {
    support::RuntimeABIParameterRole expectedRole =
        contract.runtimeABIParameterRoles[index];
    support::RuntimeABIParameterRole actualRole =
        contract.runtimeABIParameters[index].role;
    if (actualRole != expectedRole)
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-owned runtime ABI parameter[" +
          llvm::Twine(index) + "] role '" +
          support::stringifyRuntimeABIParameterRole(expectedRole) +
          "' but saw '" +
          support::stringifyRuntimeABIParameterRole(actualRole) +
          "' for parameter '" + contract.runtimeABIParameters[index].cName +
          "'");
  }

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        contract.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVRuntimeScalarSplatStoreRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVRuntimeScalarSplatStoreRouteValidationContract
        &contract) {
  const plugin::rvv::RVVRuntimeAVLVLSelectedBoundaryContract &runtimeContract =
      contract.runtimeAVLVLContract;
  if (contract.runtimeABIParameters.size() != 3 ||
      contract.logicalOperands.size() != 3)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) + " requires "
        "provider-derived ABI order rhs_scalar,out,n before artifact export");
  const support::RuntimeABIParameter &rhsScalar =
      contract.runtimeABIParameters[0];
  const support::RuntimeABIParameter &out = contract.runtimeABIParameters[1];
  const support::RuntimeABIParameter &runtimeN =
      runtimeContract.runtimeAVLParameter;

  if (contract.expectedPreLoopStepCount != 1 ||
      contract.expectedLoopBodyStepCount != 3)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned statement-plan expectations before artifact "
        "export");
  if (route.getCallOpaqueSteps().size() != contract.expectedPreLoopStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires exact provider-built pre-loop statement count " +
        llvm::Twine(contract.expectedPreLoopStepCount) +
        " before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != runtimeContract.setVLIntrinsic ||
      preLoopSetVL.operands.size() != 1 ||
      preLoopSetVL.operands.front().expression != runtimeN.cName ||
      preLoopSetVL.operands.front().cType != runtimeN.cType ||
      !stepHasResult(preLoopSetVL, runtimeContract.emitCFullChunkVLName,
                     runtimeContract.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires rebuilt provider route pre-loop setvl statement to use "
        "runtime n/AVL and define the full-chunk VL");
  if (!routeStepSourceIsSelectedRVVBody(preLoopSetVL))
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) + " requires "
        "pre-loop setvl to carry selected typed RVV source provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) + " requires exactly "
        "one provider-built runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != runtimeContract.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != runtimeContract.vlCType ||
      loop.upperBound.expression != runtimeN.cName ||
      loop.upperBound.cType != runtimeN.cType ||
      loop.step.expression != runtimeContract.emitCFullChunkVLName ||
      loop.step.cType != runtimeContract.vlCType)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) + " requires "
        "provider-built loop bounds and step to mirror runtime n/AVL/VL facts");
  if (loop.bodySteps.size() != contract.expectedLoopBodyStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires exact provider-built loop statement count " +
        llvm::Twine(contract.expectedLoopBodyStepCount) +
        " before artifact export");

  const conversion::emitc::TCRVEmitCCallOpaqueStep &loopSetVL =
      loop.bodySteps[0];
  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeN.cName) + " - " +
       runtimeContract.emitCLoopInductionName)
          .str();
  if (loopSetVL.callee != runtimeContract.setVLIntrinsic ||
      loopSetVL.operands.size() != 1 ||
      loopSetVL.operands.front().expression != expectedRemainingAVL ||
      loopSetVL.operands.front().cType != runtimeContract.vlCType ||
      !stepHasResult(loopSetVL, runtimeContract.emitCLoopVLName,
                     runtimeContract.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) + " requires "
        "provider-built loop setvl to derive per-iteration VL from remaining "
        "runtime AVL");

  const conversion::emitc::TCRVEmitCCallOpaqueStep &splat = loop.bodySteps[1];
  if (splat.callee != contract.rhsScalarSplatIntrinsic ||
      splat.operands.size() != 2 ||
      splat.operands[0].expression != rhsScalar.cName ||
      splat.operands[0].cType != rhsScalar.cType ||
      splat.operands[1].expression != runtimeContract.emitCLoopVLName ||
      splat.operands[1].cType != runtimeContract.vlCType ||
      !stepHasResult(splat, contract.resultName, contract.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) + " requires "
        "provider-built runtime scalar splat statement to consume rhs_scalar "
        "and per-iteration VL");

  const conversion::emitc::TCRVEmitCCallOpaqueStep &store = loop.bodySteps[2];
  const std::string expectedOutPointer =
      (llvm::StringRef(out.cName) + " + " +
       runtimeContract.emitCLoopInductionName)
          .str();
  if (store.callee != contract.storeIntrinsic ||
      store.operands.size() != 3 ||
      store.operands[0].expression != expectedOutPointer ||
      store.operands[0].cType != out.cType ||
      store.operands[1].expression != contract.resultName ||
      store.operands[1].cType != contract.vectorCType ||
      store.operands[2].expression != runtimeContract.emitCLoopVLName ||
      store.operands[2].cType != runtimeContract.vlCType)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) + " requires "
        "provider-built store statement to consume out+offset, splat vector, "
        "and per-iteration VL");

  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) + " requires loop "
          "statements to carry selected typed RVV source provenance");

  return llvm::Error::success();
}

llvm::Error requireRVVRuntimeScalarSplatStoreProviderField(
    llvm::StringRef fieldLabel, llvm::StringRef actual,
    llvm::StringRef expected) {
  const llvm::StringRef consumerLabel =
      "runtime scalar splat-store target artifact consumer";
  if (actual == expected)
    return llvm::Error::success();
  if (expected.empty())
    return makeRVVTargetRouteError(llvm::Twine(consumerLabel) +
                                   " rejects stale provider-derived " +
                                   fieldLabel + " fact '" + actual + "'");
  if (actual.empty())
    return makeRVVTargetRouteError(llvm::Twine(consumerLabel) +
                                   " requires provider-derived " +
                                   fieldLabel + " '" + expected +
                                   "' before artifact export");
  return makeRVVTargetRouteError(llvm::Twine(consumerLabel) +
                                 " requires provider-derived " + fieldLabel +
                                 " '" + expected + "' but was '" + actual +
                                 "'");
}

llvm::Error validateRVVRuntimeScalarSplatStoreRuntimeABIFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVRuntimeScalarSplatStoreRouteValidationContract
        &contract) {
  // This runs after runtimeAVLVLContract acceptance and checks only ABI binding
  // list consistency; runtime ABI order authority stays in the selected boundary.
  if (description.runtimeABIParameters.size() !=
      contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived runtime ABI parameter count " +
        llvm::Twine(contract.runtimeABIParameters.size()) +
        " but saw " + llvm::Twine(description.runtimeABIParameters.size()));

  for (std::size_t index = 0; index < contract.runtimeABIParameters.size();
       ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    const support::RuntimeABIParameter &expected =
        contract.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(actual, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-derived runtime ABI parameter[" +
          llvm::Twine(index) + "] to bind '" + expected.cName +
          "' as " +
          support::stringifyRuntimeABIParameterRole(expected.role) +
          " before validating route statements");
  }

  if (description.routeOperandBindingPlanID !=
      contract.routeOperandBindingPlanID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires route operand binding plan '" +
        contract.routeOperandBindingPlanID + "' before artifact export");
  if (description.routeOperandBindingSummary.empty() ||
      !llvm::StringRef(description.routeOperandBindingSummary)
           .starts_with(contract.routeOperandBindingPlanID))
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires route operand binding summary to start with provider plan '" +
        contract.routeOperandBindingPlanID + "' before artifact export");
  if (description.routeOperandBindingSummary !=
      contract.routeOperandBindingSummary)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider route operand binding summary '" +
        contract.routeOperandBindingSummary + "' but was '" +
        description.routeOperandBindingSummary + "'");

  return llvm::Error::success();
}

llvm::Error validateRVVRuntimeScalarSplatStoreRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVRuntimeScalarSplatStoreRouteValidationContract
        &contract) {
  if (contract.providerSupportedMirror.empty() ||
      contract.routeOperandBindingPlanID.empty() ||
      contract.routeOperandBindingSummary.empty() ||
      contract.runtimeScalarSplatStoreRouteFamilyPlanID.empty() ||
      contract.elementTypeName.empty() || contract.sew == 0 ||
      contract.lmul.empty() || contract.tailPolicy.empty() ||
      contract.maskPolicy.empty() || contract.configContractID.empty() ||
      contract.requiredHeaderDeclarations.empty() ||
      contract.cTypeMappingSummary.empty() ||
      contract.typedComputeOpName.empty() || contract.sourceMemoryForm.empty() ||
      contract.destinationMemoryForm.empty() || contract.scalarCType.empty() ||
      contract.vectorTypeName.empty() ||
      contract.vectorCType.empty() ||
      contract.rhsScalarSplatIntrinsic.empty() ||
      contract.storeIntrinsic.empty() || contract.resultName.empty() ||
      contract.runtimeABIParameters.size() != 3 ||
      contract.runtimeABIParameterRoles.size() != 3 ||
      contract.logicalOperands.size() != 3)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires complete provider-owned route payload, dtype/config, "
        "runtime, binding, header/type, intrinsic, and statement-plan contract "
        "facts before artifact export");

  if (route.getRouteID() != contract.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires rebuilt provider route id '" + contract.emitCRouteID +
        "' but route carried '" +
        route.getRouteID() + "'");
  if (description.operation != contract.operation)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned operation '" +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            contract.operation) +
        "' before artifact export");
  if (description.emitCRouteID != contract.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned route id '" + contract.emitCRouteID +
        "' but description carried '" + description.emitCRouteID + "'");
  if (description.memoryForm != contract.memoryForm)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires selected typed RVV memory form '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(contract.memoryForm) +
        "' before artifact export but saw '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
            description.memoryForm) +
        "'");

  if (llvm::Error error = validateRVVRuntimeAVLVLSelectedBoundaryContract(
          description, contract.runtimeAVLVLContract))
    return error;
  if (llvm::Error error = validateRVVRouteLocalRuntimeAVLVLMirrors(
          contract.consumerLabel, contract.runtimeAVLVLContract,
          contract.runtimeControlPlanID, contract.runtimeABIOrder,
          contract.setVLIntrinsic, contract.vlCType,
          contract.emitCFullChunkVLName, contract.emitCLoopVLName,
          contract.emitCLoopInductionName))
    return error;

  if (llvm::Error error = requireRVVRuntimeScalarSplatStoreProviderField(
          "typed compute op", description.typedComputeOpName,
          contract.typedComputeOpName))
    return error;
  if (llvm::Error error = requireRVVRuntimeScalarSplatStoreProviderField(
          "provider-supported mirror", description.providerSupportedMirror,
          contract.providerSupportedMirror))
    return error;
  if (llvm::Error error = requireRVVRuntimeScalarSplatStoreProviderField(
          "runtime-scalar splat-store route-family plan",
          description.runtimeScalarSplatStoreRouteFamilyPlanID,
          contract.runtimeScalarSplatStoreRouteFamilyPlanID))
    return error;
  if (description.elementTypeName != contract.elementTypeName ||
      description.sew != contract.sew)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived element '" + contract.elementTypeName +
        "' and SEW " + llvm::Twine(contract.sew) + " but description carried "
        "element '" +
        description.elementTypeName + "' and SEW " +
        llvm::Twine(description.sew));
  if (llvm::Error error = requireRVVRuntimeScalarSplatStoreProviderField(
          "LMUL", description.lmul, contract.lmul))
    return error;
  if (llvm::Error error = requireRVVRuntimeScalarSplatStoreProviderField(
          "tail policy", description.tailPolicy, contract.tailPolicy))
    return error;
  if (llvm::Error error = requireRVVRuntimeScalarSplatStoreProviderField(
          "mask policy", description.maskPolicy, contract.maskPolicy))
    return error;
  if (llvm::Error error = requireRVVRuntimeScalarSplatStoreProviderField(
          "config contract", description.configContractID,
          contract.configContractID))
    return error;
  if (llvm::Error error = requireRVVRuntimeScalarSplatStoreProviderField(
          "target leaf profile", description.targetLeafProfile,
          contract.targetLeafProfile))
    return error;
  if (llvm::Error error = requireRVVRuntimeScalarSplatStoreProviderField(
          "required header declarations", description.requiredHeaderDeclarations,
          contract.requiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireRVVRuntimeScalarSplatStoreProviderField(
          "C type mapping summary", description.cTypeMappingSummary,
          contract.cTypeMappingSummary))
    return error;
  if (llvm::Error error = requireRVVRuntimeScalarSplatStoreProviderField(
          "scalar C type", contract.runtimeABIParameters[0].cType,
          contract.scalarCType))
    return error;
  if (llvm::Error error = requireRVVRuntimeScalarSplatStoreProviderField(
          "vector type", description.vectorTypeName, contract.vectorTypeName))
    return error;
  if (llvm::Error error = requireRVVRuntimeScalarSplatStoreProviderField(
          "vector C type", description.vectorCType, contract.vectorCType))
    return error;
  if (llvm::Error error = requireRVVRuntimeScalarSplatStoreProviderField(
          "runtime scalar splat intrinsic",
          description.rhsBroadcastIntrinsic,
          contract.rhsScalarSplatIntrinsic))
    return error;
  if (llvm::Error error = requireRVVRuntimeScalarSplatStoreProviderField(
          "store intrinsic", description.storeIntrinsic,
          contract.storeIntrinsic))
    return error;
  if (llvm::Error error = requireRVVRuntimeScalarSplatStoreProviderField(
          "result", description.resultName, contract.resultName))
    return error;
  if (!description.sourceMemoryForm.empty() &&
      description.sourceMemoryForm != contract.sourceMemoryForm)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " rejects stale source memory form fact '" +
        description.sourceMemoryForm + "'");
  if (!description.destinationMemoryForm.empty() &&
      description.destinationMemoryForm != contract.destinationMemoryForm)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " rejects stale destination memory form fact '" +
        description.destinationMemoryForm + "'");
  if (!description.intrinsic.empty() ||
      !description.sourceVectorTypeName.empty() ||
      !description.sourceVectorCType.empty() ||
      !description.scalarBroadcastElementwiseRouteFamilyPlanID.empty() ||
      !description.elementwiseArithmeticRouteFamilyPlanID.empty() ||
      !description.wideningConversionRouteFamilyPlanID.empty() ||
      !description.plainMAccRouteFamilyPlanID.empty() ||
      !description.scalarBroadcastMAccRouteFamilyPlanID.empty() ||
      !description.accumulationRouteFamilyPlanID.empty() ||
      !description.wideningMAccRelation.empty() ||
      !description.wideningDotProductRelation.empty() ||
      !description.plainCompareSelectRouteFamilyPlanID.empty() ||
      !description.computedMaskSelectRouteFamilyPlanID.empty() ||
      !description.computedMaskMemoryRouteFamilyPlanID.empty() ||
      !description.segment2MemoryRouteFamilyPlanID.empty() ||
      !description.standaloneReductionRouteFamilyPlanID.empty() ||
      !description.contractionRouteFamilyPlanID.empty() ||
      !description.baseMemoryMovementRouteFamilyPlanID.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) + " must reject stale "
        "non-splat-store route-family facts");

  if (llvm::Error error =
          validateRVVRuntimeScalarSplatStoreRuntimeABIFacts(description,
                                                            contract))
    return error;
  if (llvm::Error error =
          validateRVVRuntimeScalarSplatStoreRouteHeaders(route, contract))
    return error;
  if (llvm::Error error =
          validateRVVRuntimeScalarSplatStoreRouteTypeMappings(route,
                                                              contract))
    return error;
  if (llvm::Error error =
          validateRVVRuntimeScalarSplatStoreRouteABIMappings(route, contract))
    return error;
  return validateRVVRuntimeScalarSplatStoreRouteStatementPlan(route, contract);
}

llvm::Error validateRVVRuntimeScalarSplatStoreTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  std::optional<
      plugin::rvv::RVVRuntimeScalarSplatStoreRouteValidationContract>
      contract =
          plugin::rvv::getRVVRuntimeScalarSplatStoreRouteValidationContract(
              context.description);
  if (!contract)
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-owned runtime scalar splat-store route validation contract "
        "before artifact export");
  return validateRVVRuntimeScalarSplatStoreRoutePayloadFacts(
      context.route, context.description, *contract);
}

llvm::Error validateRVVRuntimeScalarSplatStoreTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  std::optional<
      plugin::rvv::RVVRuntimeScalarSplatStoreRouteValidationContract>
      contract =
          plugin::rvv::getRVVRuntimeScalarSplatStoreRouteValidationContract(
              context.description);
  if (!contract)
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-owned runtime scalar splat-store metadata mirror contract "
        "before checking candidate mirrors");

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_plan",
          contract->routeOperandBindingPlanID,
          "selected typed RVV runtime scalar splat-store binding plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_operands",
          contract->routeOperandBindingSummary,
          "selected typed RVV runtime scalar splat-store binding summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.provider_supported_mirror",
          contract->providerSupportedMirror,
          "selected typed RVV runtime scalar splat-store provider support"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_scalar_splat_store_route_family_plan",
          contract->runtimeScalarSplatStoreRouteFamilyPlanID,
          "selected typed RVV runtime scalar splat-store route-family plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.memory_form",
          plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
              contract->memoryForm),
          "selected typed RVV runtime scalar splat-store memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "rvv_selected_body_typed_compute_op",
          contract->typedComputeOpName,
          "selected typed RVV runtime scalar splat-store typed compute op"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.target_leaf_profile",
          contract->targetLeafProfile,
          "selected typed RVV runtime scalar splat-store target leaf profile"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_control_plan",
          contract->runtimeControlPlanID,
          "route-local runtime AVL/VL control plan mirror"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order", contract->runtimeABIOrder,
          "route-local runtime AVL/VL ABI order mirror"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.required_header_declarations",
          contract->requiredHeaderDeclarations,
          "selected typed RVV runtime scalar splat-store route header "
          "requirements"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.c_type_mapping",
          contract->cTypeMappingSummary,
          "selected typed RVV runtime scalar splat-store route type mapping "
          "summary"))
    return error;

  constexpr llvm::StringLiteral staleRouteFamilies[] = {
      "tcrv_rvv.elementwise_arithmetic_route_family_plan",
      "tcrv_rvv.scalar_broadcast_elementwise_route_family_plan",
      "tcrv_rvv.widening_conversion_route_family_plan",
      "tcrv_rvv.plain_compare_select_route_family_plan",
      "tcrv_rvv.computed_mask_select_route_family_plan",
      "tcrv_rvv.computed_mask_memory_route_family_plan",
      "tcrv_rvv.plain_macc_route_family_plan",
      "tcrv_rvv.scalar_broadcast_macc_route_family_plan",
      "tcrv_rvv.accumulation_route_family_plan",
      "tcrv_rvv.segment2_memory_route_family_plan",
      "tcrv_rvv.standalone_reduction_route_family_plan",
      "tcrv_rvv.contraction_route_family_plan",
      "tcrv_rvv.base_memory_movement_route_family_plan"};
  for (llvm::StringRef key : staleRouteFamilies)
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, key, "",
            "selected typed RVV non-splat-store route-family plan"))
      return error;

  return llvm::Error::success();
}

bool routeStepsContainCallee(
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueStep> steps,
    llvm::StringRef callee) {
  if (callee.empty())
    return true;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : steps)
    if (step.callee == callee)
      return true;
  return false;
}

bool routeLoopContainsCallee(
    const conversion::emitc::TCRVEmitCForLoop &loop,
    llvm::StringRef callee) {
  return routeStepsContainCallee(loop.bodySteps, callee);
}

struct RVVExpectedRouteStepOperand {
  llvm::StringRef expression;
  llvm::StringRef cType;
};

llvm::Error validateRVVProviderBuiltRouteStep(
    const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
    llvm::StringRef consumerLabel, llvm::StringRef stepLabel,
    llvm::StringRef expectedCallee,
    llvm::ArrayRef<RVVExpectedRouteStepOperand> expectedOperands,
    llvm::StringRef expectedResultName = {},
    llvm::StringRef expectedResultCType = {}) {
  if (step.callee != expectedCallee)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) + " requires provider-built " + stepLabel +
        " statement to use callee '" + expectedCallee + "' but saw '" +
        step.callee + "'");
  if (step.operands.size() != expectedOperands.size())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) + " requires provider-built " + stepLabel +
        " statement to carry " + llvm::Twine(expectedOperands.size()) +
        " operands but saw " + llvm::Twine(step.operands.size()));
  for (std::size_t index = 0; index < expectedOperands.size(); ++index) {
    if (step.operands[index].expression != expectedOperands[index].expression ||
        step.operands[index].cType != expectedOperands[index].cType)
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) + " requires provider-built " +
          stepLabel + " operand[" + llvm::Twine(index) + "] to be '" +
          expectedOperands[index].expression + "' with C type '" +
          expectedOperands[index].cType + "' but saw '" +
          step.operands[index].expression + "' with C type '" +
          step.operands[index].cType + "'");
  }
  if (!expectedResultName.empty()) {
    if (!step.result)
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) + " requires provider-built " +
          stepLabel + " result '" + expectedResultName + "' with C type '" +
          expectedResultCType + "'");
    if (step.result->name != expectedResultName ||
        step.result->cType != expectedResultCType)
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) + " requires provider-built " +
          stepLabel + " result '" + expectedResultName + "' with C type '" +
          expectedResultCType + "' but saw '" + step.result->name +
          "' with C type '" + step.result->cType + "'");
  } else if (step.result) {
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) + " requires provider-built " + stepLabel +
        " statement not to define a result");
  }

  return llvm::Error::success();
}

const support::RuntimeABIParameter *findRuntimeElementCountABIParameter(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  for (const support::RuntimeABIParameter &parameter :
       description.runtimeABIParameters)
    if (parameter.role ==
        support::RuntimeABIParameterRole::RuntimeElementCount)
      return &parameter;
  return nullptr;
}

const support::RuntimeABIParameter *findRuntimeABIParameterByRole(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    support::RuntimeABIParameterRole role) {
  for (const support::RuntimeABIParameter &parameter :
       description.runtimeABIParameters)
    if (parameter.role == role)
      return &parameter;
  return nullptr;
}

bool bindingSummaryUseListContains(llvm::StringRef useList,
                                   llvm::StringRef expectedUse) {
  llvm::SmallVector<llvm::StringRef, 8> uses;
  useList.split(uses, '|', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (llvm::StringRef use : uses)
    if (use == expectedUse)
      return true;
  return false;
}

llvm::Error requireRuntimeByteStrideContract(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef consumerLabel, llvm::StringRef label,
    llvm::StringRef strideSource, llvm::StringRef strideCType,
    llvm::StringRef strideUnit, support::RuntimeABIParameterRole expectedRole,
    llvm::StringRef expectedBindingUse) {
  const support::RuntimeABIParameter *parameter =
      findRuntimeABIParameterByRole(description, expectedRole);
  if (strideSource.empty()) {
    if (!strideCType.empty() || !strideUnit.empty() || parameter)
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) + " rejects stale provider-derived " +
          label + " byte-stride contract facts");
    return llvm::Error::success();
  }

  if (strideCType.empty() || strideUnit.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) + " requires provider-derived " + label +
        " stride C type and stride unit before artifact export");
  if (!parameter)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) + " requires runtime ABI parameter role '" +
        support::stringifyRuntimeABIParameterRole(expectedRole) + "' for " +
        label + " byte-stride contract before artifact export");
  if (parameter->cType != strideCType)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) + " requires " + label +
        " runtime byte-stride C type '" + strideCType + "' but saw '" +
        parameter->cType + "'");
  if (strideUnit != "byte")
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) + " requires provider-derived " + label +
        " stride unit 'byte' but saw '" + strideUnit + "'");

  const std::string expectedSource =
      (llvm::Twine("runtime_abi:") + parameter->cName).str();
  if (strideSource != expectedSource)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) + " requires provider-derived " + label +
        " stride source '" + expectedSource + "' but saw '" + strideSource +
        "'");

  llvm::SmallVector<llvm::StringRef, 8> entries;
  llvm::StringRef(description.routeOperandBindingSummary)
      .split(entries, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  const std::string expectedPrefix =
      (llvm::Twine(parameter->cName) + "=").str();
  for (llvm::StringRef entry : entries) {
    if (!entry.starts_with(expectedPrefix))
      continue;
    llvm::SmallVector<llvm::StringRef, 4> fields;
    entry.drop_front(expectedPrefix.size())
        .split(fields, ':', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    if (fields.size() != 3)
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) + " requires " + label +
          " route operand binding entry to carry role, C name, and uses");
    if (fields[0] != support::stringifyRuntimeABIParameterRole(expectedRole) ||
        fields[1] != parameter->cName)
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) + " requires " + label +
          " route operand binding entry to mirror byte-stride ABI role and C "
          "name before artifact export");
    if (!bindingSummaryUseListContains(fields[2], expectedBindingUse))
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) + " requires " + label +
          " route operand binding entry to carry byte-stride use '" +
          expectedBindingUse + "'");
    return llvm::Error::success();
  }

  return makeRVVTargetRouteError(
      llvm::Twine(consumerLabel) + " requires " + label +
      " route operand binding entry for runtime ABI parameter '" +
      parameter->cName + "' before artifact export");
}

llvm::Error requireIndexedBaseMemoryHeaderBindingSummaryEntry(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef operationMnemonic,
    llvm::StringRef logicalOperand,
    const support::RuntimeABIParameter &parameter) {
  llvm::SmallVector<llvm::StringRef, 8> entries;
  llvm::StringRef(description.routeOperandBindingSummary)
      .split(entries, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  const std::string expectedPrefix = (llvm::Twine(logicalOperand) + "=").str();
  for (llvm::StringRef entry : entries) {
    if (!entry.starts_with(expectedPrefix))
      continue;

    llvm::StringRef rest = entry.drop_front(expectedPrefix.size());
    llvm::SmallVector<llvm::StringRef, 4> fields;
    rest.split(fields, ':', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    if (fields.size() != 3)
      return makeRVVTargetRouteError(
          llvm::Twine(operationMnemonic) +
          " route operand binding summary requires logical operand '" +
          logicalOperand + "' to record role, C name, and materialized uses "
                           "before artifact export");

    llvm::StringRef expectedRole =
        support::stringifyRuntimeABIParameterRole(parameter.role);
    if (fields[0] != expectedRole || fields[1] != parameter.cName)
      return makeRVVTargetRouteError(
          llvm::Twine(operationMnemonic) +
          " route operand binding summary requires logical operand '" +
          logicalOperand + "' to bind runtime ABI role '" + expectedRole +
          "' and C name '" + parameter.cName + "' before artifact export");

    if (!bindingSummaryUseListContains(fields[2], "abi") ||
        !bindingSummaryUseListContains(fields[2], "hdr"))
      return makeRVVTargetRouteError(
          llvm::Twine(operationMnemonic) +
          " route operand binding summary requires logical operand '" +
          logicalOperand +
          "' to carry provider ABI marker 'abi' and header/prototype marker "
          "'hdr' before artifact export");
    return llvm::Error::success();
  }

  return makeRVVTargetRouteError(
      llvm::Twine(operationMnemonic) +
      " route operand binding summary requires logical operand '" +
      logicalOperand + "' before artifact export");
}

llvm::Error validateIndexedBaseMemoryHeaderBindingSummary(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVBaseMemoryMovementRouteValidationContract &contract) {
  using OperationKind = plugin::rvv::RVVSelectedBodyOperationKind;
  if (contract.operation != OperationKind::IndexedGatherUnitStore &&
      contract.operation != OperationKind::IndexedScatterUnitLoad)
    return llvm::Error::success();

  const llvm::StringRef operationMnemonic =
      plugin::rvv::stringifyRVVSelectedBodyOperationKind(contract.operation);
  if (description.routeOperandBindingPlanID !=
      contract.routeOperandBindingPlanID)
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary requires provider plan '" +
        contract.routeOperandBindingPlanID + "' before artifact export");
  if (description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary is required before artifact export");
  if (!llvm::StringRef(description.routeOperandBindingSummary)
           .starts_with(contract.routeOperandBindingPlanID))
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary must start with provider plan '" +
        contract.routeOperandBindingPlanID + "' before artifact export");
  if (description.runtimeABIParameters.size() !=
      contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary requires the provider runtime ABI "
        "order before artifact export");
  if (description.routeOperandBindingSummary !=
      contract.routeOperandBindingSummary)
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary must mirror provider-owned "
        "base-memory binding summary '" + contract.routeOperandBindingSummary +
        "' before artifact export");
  return llvm::Error::success();
}

llvm::Error validateComputedMaskIndexedGatherHeaderBindingSummary(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  using OperationKind = plugin::rvv::RVVSelectedBodyOperationKind;
  if (description.operation !=
      OperationKind::ComputedMaskIndexedGatherLoadUnitStore)
    return llvm::Error::success();

  std::optional<
      plugin::rvv::RVVComputedMaskIndexedMemoryRouteValidationContract>
      contract =
          plugin::rvv::getRVVComputedMaskIndexedMemoryRouteValidationContract(
              description);
  if (!contract)
    return makeRVVTargetRouteError(
        "computed-mask indexed memory target artifact consumer requires "
        "provider-owned indexed route validation contract before artifact "
        "export");

  const llvm::StringRef operationMnemonic =
      plugin::rvv::stringifyRVVSelectedBodyOperationKind(description.operation);
  if (llvm::StringRef(description.routeOperandBindingPlanID) !=
      contract->routeOperandBindingPlanID)
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary requires provider plan '" +
        contract->routeOperandBindingPlanID + "' before artifact export");
  if (description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary is required before artifact export");
  if (!llvm::StringRef(description.routeOperandBindingSummary)
           .starts_with(contract->routeOperandBindingPlanID))
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary must start with provider plan '" +
        contract->routeOperandBindingPlanID + "' before artifact export");

  if (description.runtimeABIParameters.size() !=
          contract->runtimeABIParameters.size() ||
      contract->logicalOperands.size() !=
          contract->runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary requires the provider runtime ABI "
        "order before artifact export");

  for (std::size_t index = 0; index < contract->logicalOperands.size();
       ++index)
    if (llvm::Error error = requireIndexedBaseMemoryHeaderBindingSummaryEntry(
            description, operationMnemonic, contract->logicalOperands[index],
            description.runtimeABIParameters[index]))
      return error;
  if (description.routeOperandBindingSummary !=
      contract->routeOperandBindingSummary)
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary must mirror provider-owned "
        "computed-mask indexed memory binding summary '" +
        contract->routeOperandBindingSummary + "' before artifact export");
  return llvm::Error::success();
}

llvm::Error validateComputedMaskIndexedScatterHeaderBindingSummary(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  using OperationKind = plugin::rvv::RVVSelectedBodyOperationKind;
  if (description.operation !=
          OperationKind::ComputedMaskIndexedScatterStoreUnitLoad &&
      description.operation !=
          OperationKind::RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad)
    return llvm::Error::success();

  std::optional<
      plugin::rvv::RVVComputedMaskIndexedMemoryRouteValidationContract>
      contract =
          plugin::rvv::getRVVComputedMaskIndexedMemoryRouteValidationContract(
              description);
  if (!contract)
    return makeRVVTargetRouteError(
        "computed-mask indexed memory target artifact consumer requires "
        "provider-owned indexed route validation contract before artifact "
        "export");

  const llvm::StringRef operationMnemonic =
      plugin::rvv::stringifyRVVSelectedBodyOperationKind(description.operation);
  if (llvm::StringRef(description.routeOperandBindingPlanID) !=
      contract->routeOperandBindingPlanID)
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary requires provider plan '" +
        contract->routeOperandBindingPlanID + "' before artifact export");
  if (description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary is required before artifact export");
  if (!llvm::StringRef(description.routeOperandBindingSummary)
           .starts_with(contract->routeOperandBindingPlanID))
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary must start with provider plan '" +
        contract->routeOperandBindingPlanID + "' before artifact export");

  if (description.runtimeABIParameters.size() !=
          contract->runtimeABIParameters.size() ||
      contract->logicalOperands.size() !=
          contract->runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary requires the provider runtime ABI "
        "order before artifact export");

  for (std::size_t index = 0; index < contract->logicalOperands.size();
       ++index)
    if (llvm::Error error = requireIndexedBaseMemoryHeaderBindingSummaryEntry(
            description, operationMnemonic, contract->logicalOperands[index],
            description.runtimeABIParameters[index]))
      return error;
  if (description.routeOperandBindingSummary !=
      contract->routeOperandBindingSummary)
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary must mirror provider-owned "
        "computed-mask indexed memory binding summary '" +
        contract->routeOperandBindingSummary + "' before artifact export");
  return llvm::Error::success();
}

llvm::Error validateRVVSegment2HeaderBindingSummary(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVSegment2MemoryRouteValidationContract &contract) {
  const llvm::StringRef operationMnemonic =
      plugin::rvv::stringifyRVVSelectedBodyOperationKind(contract.operation);
  if (llvm::StringRef(description.routeOperandBindingPlanID) !=
      contract.routeOperandBindingPlanID)
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary requires provider plan '" +
        contract.routeOperandBindingPlanID + "' before artifact export");
  if (description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary is required before artifact export");
  if (!llvm::StringRef(description.routeOperandBindingSummary)
           .starts_with(contract.routeOperandBindingPlanID))
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary must start with provider plan '" +
        contract.routeOperandBindingPlanID + "' before artifact export");

  if (description.runtimeABIParameters.size() !=
          contract.logicalOperands.size() ||
      contract.runtimeABIParameters.size() != contract.logicalOperands.size())
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary requires the provider runtime ABI "
        "order before artifact export");

  for (std::size_t index = 0; index < contract.logicalOperands.size(); ++index)
    if (llvm::Error error = requireIndexedBaseMemoryHeaderBindingSummaryEntry(
            description, operationMnemonic, contract.logicalOperands[index],
            contract.runtimeABIParameters[index]))
      return error;
  if (description.routeOperandBindingSummary !=
      contract.routeOperandBindingSummary)
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary must mirror provider-owned "
        "segment2-memory binding summary '" +
        contract.routeOperandBindingSummary + "' before artifact export");
  return llvm::Error::success();
}

llvm::Error validateRVVBaseMemoryMovementRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVBaseMemoryMovementRouteValidationContract &contract) {
  if (contract.requiredHeaderDeclarations.empty() ||
      contract.requiredHeaders.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned required_header_declarations before "
        "accepting the route artifact");

  for (llvm::StringRef header : contract.requiredHeaders) {
    if (header.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " saw an empty provider route header declaration");
    if (!routeHasHeader(route, header))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route header '" + header +
          "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVBaseMemoryMovementRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVBaseMemoryMovementRouteValidationContract &contract) {
  if (contract.vlCType.empty() || contract.vectorTypeName.empty() ||
      contract.vectorCType.empty() || contract.cTypeMappingSummary.empty() ||
      contract.typeMappings.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires "
        "provider-derived VL, vector, and C type mapping facts before "
        "artifact export");

  for (const plugin::rvv::RVVBaseMemoryMovementRouteTypeMappingContract
           &mapping : contract.typeMappings) {
    if (mapping.sourceType.empty() || mapping.cType.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires non-empty provider type mapping contract entries");
    if (!routeHasTypeMapping(route, mapping.sourceType, mapping.cType))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route type mapping '" +
          mapping.sourceType + "' -> '" + mapping.cType + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVBaseMemoryMovementRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVBaseMemoryMovementRouteValidationContract &contract) {
  if (contract.runtimeABIParameterRoles.size() !=
      contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned runtime ABI role order with one role per "
        "ABI parameter before artifact export");
  if (contract.runtimeABIOrder.empty() || contract.runtimeABIParameters.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires "
        "provider-derived runtime ABI order and ABI parameters before "
        "artifact export");
  if (route.getABIMappings().size() != contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires rebuilt provider route ABI mapping count " +
        llvm::Twine(contract.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0, count = contract.runtimeABIParameters.size();
       index < count; ++index) {
    support::RuntimeABIParameterRole expectedRole =
        contract.runtimeABIParameterRoles[index];
    support::RuntimeABIParameterRole actualRole =
        contract.runtimeABIParameters[index].role;
    if (actualRole != expectedRole)
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-owned runtime ABI parameter[" +
          llvm::Twine(index) + "] role '" +
          support::stringifyRuntimeABIParameterRole(expectedRole) +
          "' but saw '" +
          support::stringifyRuntimeABIParameterRole(actualRole) +
          "' for parameter '" + contract.runtimeABIParameters[index].cName +
          "'");
  }

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        contract.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVBaseMemoryMovementRuntimeABIFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVBaseMemoryMovementRouteValidationContract &contract) {
  if (description.runtimeABIParameters.size() !=
      contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived runtime ABI parameter count " +
        llvm::Twine(contract.runtimeABIParameters.size()) +
        " for operation '" +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            contract.operation) +
        "' but saw " + llvm::Twine(description.runtimeABIParameters.size()));

  for (std::size_t index = 0; index < contract.runtimeABIParameters.size();
       ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    const support::RuntimeABIParameter &expected =
        contract.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(actual, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-derived runtime ABI parameter[" +
          llvm::Twine(index) + "] to bind '" + expected.cName +
          "' as " +
          support::stringifyRuntimeABIParameterRole(expected.role) +
          " before validating route statements");
  }

  return validateIndexedBaseMemoryHeaderBindingSummary(description, contract);
}

llvm::Error validateRVVBaseMemoryMovementRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVBaseMemoryMovementRouteValidationContract &contract) {
  llvm::StringRef consumerLabel = contract.consumerLabel;
  const plugin::rvv::RVVRuntimeAVLVLSelectedBoundaryContract &runtimeContract =
      contract.runtimeAVLVLContract;

  if (llvm::Error error =
          validateRVVBaseMemoryMovementRuntimeABIFacts(description, contract))
    return error;
  if (runtimeContract.vlCType.empty() || contract.vectorCType.empty() ||
      contract.resultName.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived result, vector C type, and VL C type "
        "facts before validating route statements");

  using OperationKind = plugin::rvv::RVVSelectedBodyOperationKind;
  const OperationKind operation = contract.operation;
  const support::RuntimeABIParameter *sourceABI =
      &contract.runtimeABIParameters[0];
  const support::RuntimeABIParameter *indexABI = nullptr;
  const support::RuntimeABIParameter *maskABI = nullptr;
  const support::RuntimeABIParameter *destinationABI = nullptr;
  const support::RuntimeABIParameter *runtimeNABI = nullptr;
  const support::RuntimeABIParameter *sourceStrideABI = nullptr;
  const support::RuntimeABIParameter *destinationStrideABI = nullptr;

  switch (operation) {
  case OperationKind::StridedLoadUnitStore:
    destinationABI = &contract.runtimeABIParameters[1];
    runtimeNABI = &contract.runtimeABIParameters[2];
    sourceStrideABI = &contract.runtimeABIParameters[3];
    break;
  case OperationKind::UnitLoadStridedStore:
    destinationABI = &contract.runtimeABIParameters[1];
    runtimeNABI = &contract.runtimeABIParameters[2];
    destinationStrideABI = &contract.runtimeABIParameters[3];
    break;
  case OperationKind::IndexedGatherUnitStore:
    indexABI = &contract.runtimeABIParameters[1];
    destinationABI = &contract.runtimeABIParameters[2];
    runtimeNABI = &contract.runtimeABIParameters[3];
    break;
  case OperationKind::IndexedScatterUnitLoad:
    indexABI = &contract.runtimeABIParameters[1];
    destinationABI = &contract.runtimeABIParameters[2];
    runtimeNABI = &contract.runtimeABIParameters[3];
    break;
  case OperationKind::MaskedUnitLoadStore:
  case OperationKind::MaskedUnitStore:
    maskABI = &contract.runtimeABIParameters[1];
    destinationABI = &contract.runtimeABIParameters[2];
    runtimeNABI = &contract.runtimeABIParameters[3];
    break;
  default:
    llvm_unreachable("validated non-base-memory operation as base memory");
  }

  const support::RuntimeABIParameter *runtimeElementCount = nullptr;
  for (const support::RuntimeABIParameter &parameter :
       contract.runtimeABIParameters)
    if (parameter.role ==
        support::RuntimeABIParameterRole::RuntimeElementCount) {
      runtimeElementCount = &parameter;
      break;
    }
  if (!runtimeElementCount || runtimeElementCount != runtimeNABI)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires runtime n/AVL ABI role to match the selected base-memory "
        "ABI order before validating route statements");
  if (!runtimeABIParameterEquals(*runtimeNABI,
                                 runtimeContract.runtimeAVLParameter))
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires runtime n/AVL ABI parameter to match the provider-owned "
        "runtime AVL/VL selected-boundary contract before validating route "
        "statements");

  if (route.getCallOpaqueSteps().size() != contract.expectedPreLoopStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires exact provider-built base-memory pre-loop statement count " +
        llvm::Twine(contract.expectedPreLoopStepCount) +
        " before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          preLoopSetVL, consumerLabel, "pre-loop setvl",
          runtimeContract.setVLIntrinsic,
          {{runtimeContract.runtimeAVLParameter.cName,
            runtimeContract.runtimeAVLParameter.cType}},
          runtimeContract.emitCFullChunkVLName, runtimeContract.vlCType))
    return error;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "base-memory-movement target artifact consumer requires pre-loop "
          "statements to carry selected typed RVV source provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "base-memory-movement target artifact consumer requires exactly one "
        "provider-built runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != runtimeContract.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != runtimeContract.vlCType ||
      loop.upperBound.expression !=
          runtimeContract.runtimeAVLParameter.cName ||
      loop.upperBound.cType != runtimeContract.runtimeAVLParameter.cType ||
      loop.step.expression != runtimeContract.emitCFullChunkVLName ||
      loop.step.cType != runtimeContract.vlCType)
    return makeRVVTargetRouteError(
        "base-memory-movement target artifact consumer requires "
        "provider-built loop bounds and step to mirror runtime n/AVL/VL facts");
  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeContract.runtimeAVLParameter.cName) + " - " +
       runtimeContract.emitCLoopInductionName)
          .str();

  if (contract.expectedPreLoopStepCount == 0 ||
      contract.expectedLoopBodyStepCount == 0)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned statement-plan expectations before artifact "
        "export");
  if (loop.bodySteps.size() != contract.expectedLoopBodyStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires exact provider-built base-memory loop statement count " +
        llvm::Twine(contract.expectedLoopBodyStepCount) +
        " before artifact export");

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[0], consumerLabel, "loop setvl",
          runtimeContract.setVLIntrinsic,
          {{expectedRemainingAVL, runtimeContract.vlCType}},
          runtimeContract.emitCLoopVLName, runtimeContract.vlCType))
    return error;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "base-memory-movement target artifact consumer requires loop "
          "statements to carry selected typed RVV source provenance");

  const std::string sourcePointer =
      (llvm::StringRef(sourceABI->cName) + " + " +
       runtimeContract.emitCLoopInductionName)
          .str();
  const std::string destinationPointer =
      (llvm::StringRef(destinationABI->cName) + " + " +
       runtimeContract.emitCLoopInductionName)
          .str();
  const std::string stridedSourcePointer =
      sourceStrideABI
          ? ("(const int32_t *)((const uint8_t *)" +
             llvm::StringRef(sourceABI->cName) + " + (" +
             runtimeContract.emitCLoopInductionName + " * " +
             sourceStrideABI->cName + "))")
                .str()
          : std::string();
  const std::string stridedDestinationPointer =
      destinationStrideABI
          ? ("(int32_t *)((uint8_t *)" +
             llvm::StringRef(destinationABI->cName) + " + (" +
             runtimeContract.emitCLoopInductionName + " * " +
             destinationStrideABI->cName + "))")
                .str()
          : std::string();
  const std::string indexPointer =
      indexABI ? (llvm::StringRef(indexABI->cName) + " + " +
                  runtimeContract.emitCLoopInductionName)
                     .str()
               : std::string();
  const std::string maskPointer =
      maskABI ? (llvm::StringRef(maskABI->cName) + " + " +
                 runtimeContract.emitCLoopInductionName)
                    .str()
              : std::string();

  auto validateUnitLoad =
      [&](std::size_t stepIndex, llvm::StringRef resultName,
          llvm::StringRef stepLabel) -> llvm::Error {
    return validateRVVProviderBuiltRouteStep(
        loop.bodySteps[stepIndex], consumerLabel, stepLabel,
        contract.vectorLoadIntrinsic,
        {{sourcePointer, sourceABI->cType},
         {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
        resultName, contract.vectorCType);
  };
  auto validateUnitStore = [&](std::size_t stepIndex,
                               llvm::StringRef valueName,
                               llvm::StringRef stepLabel) -> llvm::Error {
    return validateRVVProviderBuiltRouteStep(
        loop.bodySteps[stepIndex], consumerLabel, stepLabel,
        contract.storeIntrinsic,
        {{destinationPointer, destinationABI->cType},
         {valueName, contract.vectorCType},
         {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}});
  };

  switch (operation) {
  case OperationKind::StridedLoadUnitStore:
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[1], consumerLabel, "strided load",
            contract.stridedLoadIntrinsic,
            {{stridedSourcePointer, sourceABI->cType},
             {sourceStrideABI->cName, "ptrdiff_t"},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            contract.resultName, contract.vectorCType))
      return error;
    return validateUnitStore(/*stepIndex=*/2, contract.resultName,
                             "unit store");
  case OperationKind::UnitLoadStridedStore:
    if (llvm::Error error =
            validateUnitLoad(/*stepIndex=*/1, contract.resultName,
                             "unit load"))
      return error;
    return validateRVVProviderBuiltRouteStep(
        loop.bodySteps[2], consumerLabel, "strided store",
        contract.stridedStoreIntrinsic,
        {{stridedDestinationPointer, destinationABI->cType},
         {destinationStrideABI->cName, "ptrdiff_t"},
         {contract.resultName, contract.vectorCType},
         {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}});
  case OperationKind::IndexedGatherUnitStore:
    if (contract.indexVectorCType.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires provider-derived index vector C type before validating "
          "indexed gather statements");
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[1], consumerLabel, "index load",
            contract.indexLoadIntrinsic,
            {{indexPointer, indexABI->cType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            "index_vec", contract.indexVectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[2], consumerLabel, "index scale",
            contract.indexScaleIntrinsic,
            {{"index_vec", contract.indexVectorCType},
             {"4", "uint32_t"},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            "byte_offsets", contract.indexVectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[3], consumerLabel, "indexed gather load",
            contract.indexedLoadIntrinsic,
            {{sourceABI->cName, sourceABI->cType},
             {"byte_offsets", contract.indexVectorCType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            contract.resultName, contract.vectorCType))
      return error;
    return validateUnitStore(/*stepIndex=*/4, contract.resultName,
                             "unit store");
  case OperationKind::IndexedScatterUnitLoad:
    if (contract.indexVectorCType.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires provider-derived index vector C type before validating "
          "indexed scatter statements");
    if (llvm::Error error =
            validateUnitLoad(/*stepIndex=*/1, contract.resultName,
                             "unit load"))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[2], consumerLabel, "index load",
            contract.indexLoadIntrinsic,
            {{indexPointer, indexABI->cType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            "index_vec", contract.indexVectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[3], consumerLabel, "index scale",
            contract.indexScaleIntrinsic,
            {{"index_vec", contract.indexVectorCType},
             {"4", "uint32_t"},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            "byte_offsets", contract.indexVectorCType))
      return error;
    return validateRVVProviderBuiltRouteStep(
        loop.bodySteps[4], consumerLabel, "indexed scatter store",
        contract.indexedStoreIntrinsic,
        {{destinationABI->cName, destinationABI->cType},
         {"byte_offsets", contract.indexVectorCType},
         {contract.resultName, contract.vectorCType},
         {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}});
  case OperationKind::MaskedUnitLoadStore:
    if (contract.maskName.empty() || contract.maskCType.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires provider-derived mask result and mask C type before "
          "validating masked load/store statements");
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[1], consumerLabel, "mask vector load",
            contract.vectorLoadIntrinsic,
            {{maskPointer, maskABI->cType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            "mask_i32_vec", contract.vectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[2], consumerLabel, "mask predicate",
            contract.compareIntrinsic,
            {{"mask_i32_vec", contract.vectorCType},
             {"0", "int32_t"},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            contract.maskName, contract.maskCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[3], consumerLabel, "passthrough load",
            contract.vectorLoadIntrinsic,
            {{destinationPointer, destinationABI->cType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            "old_dst_vec", contract.vectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[4], consumerLabel, "masked load",
            contract.maskedLoadIntrinsic,
            {{contract.maskName, contract.maskCType},
             {"old_dst_vec", contract.vectorCType},
             {sourcePointer, sourceABI->cType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            contract.resultName, contract.vectorCType))
      return error;
    return validateUnitStore(/*stepIndex=*/5, contract.resultName,
                             "unit store");
  case OperationKind::MaskedUnitStore:
    if (contract.maskName.empty() || contract.maskCType.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires provider-derived mask result and mask C type before "
          "validating masked store statements");
    if (llvm::Error error =
            validateUnitLoad(/*stepIndex=*/1, "lhs_vec", "source load"))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[2], consumerLabel, "mask vector load",
            contract.vectorLoadIntrinsic,
            {{maskPointer, maskABI->cType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            "mask_i32_vec", contract.vectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[3], consumerLabel, "mask predicate",
            contract.compareIntrinsic,
            {{"mask_i32_vec", contract.vectorCType},
             {"0", "int32_t"},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            contract.maskName, contract.maskCType))
      return error;
    return validateRVVProviderBuiltRouteStep(
        loop.bodySteps[4], consumerLabel, "masked store",
        contract.storeIntrinsic,
        {{contract.maskName, contract.maskCType},
         {destinationPointer, destinationABI->cType},
         {"lhs_vec", contract.vectorCType},
         {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}});
  default:
    llvm_unreachable("validated non-base-memory operation as base memory");
  }
}

llvm::Error validateRVVBaseMemoryMovementRouteLayoutFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVBaseMemoryMovementRouteValidationContract &contract) {
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "strided memory layout", description.stridedMemoryLayout,
          contract.stridedMemoryLayout))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "indexed or masked memory layout", description.indexedMemoryLayout,
          contract.indexedMemoryLayout))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "lhs stride source", description.lhsStrideSource, ""))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "rhs stride source", description.rhsStrideSource, ""))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "source stride binding", description.sourceStrideSource,
          contract.sourceStrideSource))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "destination stride binding", description.outStrideSource,
          contract.destinationStrideSource))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "source memory form", description.sourceMemoryForm,
          contract.sourceMemoryForm))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "destination memory form", description.destinationMemoryForm,
          contract.destinationMemoryForm))
    return error;

  if (description.indexEEW != contract.indexEEW)
    return makeRVVTargetRouteError(
        llvm::Twine("base-memory-movement target artifact consumer requires "
                    "provider-derived index EEW ") +
        llvm::Twine(contract.indexEEW) + " but was " +
        llvm::Twine(description.indexEEW));
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "index source", description.indexSource, contract.indexSource))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "offset unit", description.offsetUnit, contract.offsetUnit))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "index uniqueness", description.indexUniqueness,
          contract.indexUniqueness))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "indexed data memory form", description.indexedDataMemoryForm,
          contract.indexedDataMemoryForm))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "indexed destination memory form",
          description.indexedDestinationMemoryForm,
          contract.indexedDestinationMemoryForm))
    return error;

  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "mask role", description.maskRole, contract.maskRole))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "mask source", description.maskSource, contract.maskSource))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "mask memory form", description.maskMemoryForm,
          contract.maskMemoryForm))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "inactive lane contract", description.inactiveLaneContract,
          contract.inactiveLaneContract))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "masked passthrough layout", description.maskedPassthroughLayout,
          contract.maskedPassthroughLayout))
    return error;
  return llvm::Error::success();
}

llvm::Error validateRVVBaseMemoryMovementRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  std::optional<plugin::rvv::RVVBaseMemoryMovementRouteValidationContract>
      contract =
          plugin::rvv::getRVVBaseMemoryMovementRouteValidationContract(
              description);
  if (!contract)
    return makeRVVTargetRouteError(
        "base-memory-movement target artifact consumer requires "
        "provider-owned base-memory movement route validation contract before "
        "artifact export");
  if (contract->providerSupportedMirror.empty() ||
      contract->routeOperandBindingPlanID.empty() ||
      contract->routeOperandBindingSummary.empty() ||
      contract->baseMemoryMovementRouteFamilyPlanID.empty() ||
      contract->elementTypeName.empty() || contract->sew == 0 ||
      contract->lmul.empty() || contract->tailPolicy.empty() ||
      contract->maskPolicy.empty() || contract->configContractID.empty() ||
      contract->requiredHeaderDeclarations.empty() ||
      contract->cTypeMappingSummary.empty() ||
      contract->vectorCType.empty() || contract->resultName.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract->consumerLabel) +
        " requires complete provider-owned route payload, dtype/config, "
        "binding, header/type, intrinsic, and result contract facts "
        "before artifact export");

  if (llvm::Error error = validateRVVRuntimeAVLVLSelectedBoundaryContract(
          description, contract->runtimeAVLVLContract))
    return error;
  if (llvm::Error error = validateRVVRouteLocalRuntimeAVLVLMirrors(
          contract->consumerLabel, contract->runtimeAVLVLContract,
          contract->runtimeControlPlanID, contract->runtimeABIOrder,
          contract->setVLIntrinsic, contract->vlCType,
          contract->emitCFullChunkVLName, contract->emitCLoopVLName,
          contract->emitCLoopInductionName))
    return error;

  if (route.getRouteID() != contract->emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract->consumerLabel) +
        " requires rebuilt provider route id '" + contract->emitCRouteID +
        "' but route carried '" +
        route.getRouteID() + "'");
  if (description.operation != contract->operation)
    return makeRVVTargetRouteError(
        llvm::Twine(contract->consumerLabel) +
        " requires provider-owned operation '" +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            contract->operation) +
        "' before artifact export");
  if (description.emitCRouteID != contract->emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract->consumerLabel) +
        " requires provider-owned route id '" + contract->emitCRouteID +
        "' but description carried '" + description.emitCRouteID + "'");
  if (description.memoryForm != contract->memoryForm)
    return makeRVVTargetRouteError(
        llvm::Twine(contract->consumerLabel) +
        " requires selected typed RVV memory form '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
            contract->memoryForm) +
        "' before artifact export but saw '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
            description.memoryForm) +
        "'");
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "typed compute op", description.typedComputeOpName,
          contract->typedComputeOpName))
    return error;
  if (description.routeOperandBindingSummary.empty() ||
      description.targetLeafProfile.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract->consumerLabel) +
        " requires "
        "provider-derived binding summary and target leaf profile facts before "
        "artifact export");

  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "provider-supported mirror", description.providerSupportedMirror,
          contract->providerSupportedMirror))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "target leaf profile", description.targetLeafProfile,
          contract->targetLeafProfile))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "route-family plan",
          description.baseMemoryMovementRouteFamilyPlanID,
          contract->baseMemoryMovementRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "route operand binding plan", description.routeOperandBindingPlanID,
          contract->routeOperandBindingPlanID))
    return error;
  if (!llvm::StringRef(description.routeOperandBindingSummary)
           .starts_with(description.routeOperandBindingPlanID))
    return makeRVVTargetRouteError(
        llvm::Twine("base-memory-movement target artifact consumer requires "
                    "route operand binding summary to mirror provider binding "
                    "plan '") +
        description.routeOperandBindingPlanID + "' but was '" +
        description.routeOperandBindingSummary + "'");
  if (description.routeOperandBindingSummary !=
      contract->routeOperandBindingSummary)
    return makeRVVTargetRouteError(
        llvm::Twine(contract->consumerLabel) +
        " requires provider route operand binding summary '" +
        contract->routeOperandBindingSummary + "' but was '" +
        description.routeOperandBindingSummary + "'");
  if (description.elementTypeName != contract->elementTypeName ||
      description.sew != contract->sew)
    return makeRVVTargetRouteError(
        llvm::Twine(contract->consumerLabel) +
        " requires provider-derived element '" + contract->elementTypeName +
        "' and SEW " + llvm::Twine(contract->sew) + " but description carried "
        "element '" +
        description.elementTypeName + "' and SEW " +
        llvm::Twine(description.sew));
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "LMUL", description.lmul, contract->lmul))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "tail policy", description.tailPolicy, contract->tailPolicy))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "mask policy", description.maskPolicy, contract->maskPolicy))
    return error;
  if (description.configContractID != contract->configContractID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract->consumerLabel) +
        " requires provider-owned config contract '" +
        contract->configContractID + "' but description carried '" +
        description.configContractID + "'");
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "vector type", description.vectorTypeName, contract->vectorTypeName))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "vector C type", description.vectorCType, contract->vectorCType))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "index vector type", description.indexVectorTypeName,
          contract->indexVectorTypeName))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "index vector C type", description.indexVectorCType,
          contract->indexVectorCType))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "mask type", description.maskTypeName, contract->maskTypeName))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "mask C type", description.maskCType, contract->maskCType))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "vector load callee", description.vectorLoadIntrinsic,
          contract->vectorLoadIntrinsic))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "index load callee", description.indexLoadIntrinsic,
          contract->indexLoadIntrinsic))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "index scale callee", description.indexScaleIntrinsic,
          contract->indexScaleIntrinsic))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "indexed load callee", description.indexedLoadIntrinsic,
          contract->indexedLoadIntrinsic))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "indexed store callee", description.indexedStoreIntrinsic,
          contract->indexedStoreIntrinsic))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "strided load callee", description.stridedLoadIntrinsic,
          contract->stridedLoadIntrinsic))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "masked load callee", description.maskedLoadIntrinsic,
          contract->maskedLoadIntrinsic))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "compare callee", description.compareIntrinsic,
          contract->compareIntrinsic))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "store callee", description.storeIntrinsic,
          contract->storeIntrinsic))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "strided store callee", description.stridedStoreIntrinsic,
          contract->stridedStoreIntrinsic))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "result", description.resultName, contract->resultName))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "mask result", description.maskName, contract->maskName))
    return error;
  if (llvm::Error error = requireRuntimeByteStrideContract(
          description, "base-memory-movement target artifact consumer",
          "source", contract->sourceStrideSource,
          contract->sourceStrideCType, contract->sourceStrideUnit,
          support::RuntimeABIParameterRole::SourceByteStride,
          "materialized-byte-address"))
    return error;
  if (llvm::Error error = requireRuntimeByteStrideContract(
          description, "base-memory-movement target artifact consumer",
          "destination", contract->destinationStrideSource,
          contract->destinationStrideCType,
          contract->destinationStrideUnit,
          support::RuntimeABIParameterRole::DestinationByteStride,
          "materialized-byte-address"))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "required_header_declarations",
          description.requiredHeaderDeclarations,
          contract->requiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "C type mapping", description.cTypeMappingSummary,
          contract->cTypeMappingSummary))
    return error;
  if (!description.runtimeScalarSplatStoreRouteFamilyPlanID.empty() ||
      !description.scalarBroadcastElementwiseRouteFamilyPlanID.empty() ||
      !description.elementwiseArithmeticRouteFamilyPlanID.empty() ||
      !description.wideningConversionRouteFamilyPlanID.empty() ||
      !description.plainMAccRouteFamilyPlanID.empty() ||
      !description.scalarBroadcastMAccRouteFamilyPlanID.empty() ||
      !description.accumulationRouteFamilyPlanID.empty() ||
      !description.plainCompareSelectRouteFamilyPlanID.empty() ||
      !description.computedMaskSelectRouteFamilyPlanID.empty() ||
      !description.computedMaskMemoryRouteFamilyPlanID.empty() ||
      !description.segment2MemoryRouteFamilyPlanID.empty() ||
      !description.standaloneReductionRouteFamilyPlanID.empty() ||
      !description.contractionRouteFamilyPlanID.empty() ||
      !description.wideningMAccRelation.empty() ||
      !description.wideningDotProductRelation.empty())
    return makeRVVTargetRouteError(
        "base-memory-movement target artifact consumer rejects stale non-base "
        "route-family facts");

  if (description.operation ==
          plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitLoadStore ||
      description.operation ==
          plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitStore)
    if (llvm::Error error =
            validateRVVUnitStrideMaskedMemoryCanonicalProviderFacts(
                description))
      return error;

  if (llvm::Error error =
          validateRVVBaseMemoryMovementRouteLayoutFacts(description,
                                                       *contract))
    return error;
  if (llvm::Error error =
          validateRVVBaseMemoryMovementRouteHeaders(route, *contract))
    return error;
  if (llvm::Error error =
          validateRVVBaseMemoryMovementRouteTypeMappings(route, *contract))
    return error;
  if (llvm::Error error =
          validateRVVBaseMemoryMovementRouteABIMappings(route, *contract))
    return error;
  return validateRVVBaseMemoryMovementRouteStatementPlan(route, description,
                                                        *contract);
}

llvm::Error validateRVVBaseMemoryMovementTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVBaseMemoryMovementRoutePayloadFacts(context.route,
                                                        context.description);
}

llvm::Error validateRVVBaseMemoryMovementTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description =
      context.description;
  std::optional<plugin::rvv::RVVMemoryRouteMetadataMirrorContractSet>
      contract =
          plugin::rvv::getRVVBaseMemoryRouteMetadataMirrorContract(description);
  if (!contract)
    return makeRVVTargetRouteError(
        "base-memory-movement target artifact consumer requires "
        "provider-owned canonical memory metadata mirror contract before "
        "checking candidate mirrors");
  return validateRVVProviderMemoryRouteMetadataMirrorContract(candidate,
                                                              *contract);
}

std::optional<plugin::rvv::RVVWideningDotReduceRouteValidationContract>
getRVVWideningDotReductionTargetRouteValidationContract(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return plugin::rvv::getRVVWideningDotReduceRouteValidationContract(
      description);
}

std::optional<plugin::rvv::RVVWideningProductRouteValidationContract>
getRVVWideningProductTargetRouteValidationContract(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return plugin::rvv::getRVVWideningProductRouteValidationContract(
      description);
}

llvm::Error validateRVVWideningProductNoStaleNonFamilyProviderFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (!description.elementwiseArithmeticRouteFamilyPlanID.empty() ||
      !description.scalarBroadcastElementwiseRouteFamilyPlanID.empty() ||
      !description.runtimeScalarSplatStoreRouteFamilyPlanID.empty() ||
      !description.wideningConversionRouteFamilyPlanID.empty() ||
      !description.plainMAccRouteFamilyPlanID.empty() ||
      !description.scalarBroadcastMAccRouteFamilyPlanID.empty() ||
      !description.accumulationRouteFamilyPlanID.empty() ||
      !description.standaloneReductionRouteFamilyPlanID.empty() ||
      !description.plainCompareSelectRouteFamilyPlanID.empty() ||
      !description.computedMaskSelectRouteFamilyPlanID.empty() ||
      !description.computedMaskMemoryRouteFamilyPlanID.empty() ||
      !description.segment2MemoryRouteFamilyPlanID.empty() ||
      !description.baseMemoryMovementRouteFamilyPlanID.empty() ||
      !description.maccAccumulatorLayout.empty() ||
      !description.maccResultLayout.empty() ||
      !description.wideningMAccAccumulatorLayout.empty() ||
      !description.wideningMAccResultLayout.empty() ||
      !description.wideningMAccRelation.empty() ||
      !description.wideningDotProductAccumulatorLayout.empty() ||
      !description.wideningDotProductResultLayout.empty() ||
      !description.wideningDotProductRelation.empty() ||
      !description.scalarSeedSplatIntrinsic.empty() ||
      !description.reductionStoreVL.empty() ||
      !description.maskedWideningProductIntrinsic.empty())
    return makeRVVTargetRouteError(
        "low-precision widening-product target artifact consumer rejects "
        "stale non-product route-family facts");
  return llvm::Error::success();
}

llvm::Error requireRVVWideningProductContractStringField(
    llvm::StringRef consumerLabel, llvm::StringRef fieldLabel,
    llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  if (expected.empty())
    return makeRVVTargetRouteError(llvm::Twine(consumerLabel) +
                                   " rejects stale " + fieldLabel +
                                   " facts before artifact export");
  return makeRVVTargetRouteError(llvm::Twine(consumerLabel) +
                                 " requires provider-derived " + fieldLabel +
                                 " '" + expected + "' but was '" + actual +
                                 "'");
}

llvm::Error requireRVVWideningProductContractIntField(
    llvm::StringRef consumerLabel, llvm::StringRef fieldLabel,
    std::int64_t actual, std::int64_t expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVTargetRouteError(llvm::Twine(consumerLabel) +
                                 " requires provider-derived " + fieldLabel +
                                 " " + llvm::Twine(expected) + " but was " +
                                 llvm::Twine(actual));
}

llvm::Error validateRVVWideningProductDescriptionAgainstContract(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVWideningProductRouteValidationContract &contract) {
  if (description.memoryForm != contract.core.memoryForm)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires selected typed RVV memory form '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
            contract.core.memoryForm) +
        "' but was '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
            description.memoryForm) +
        "'");

  if (llvm::Error error = requireRVVWideningProductContractIntField(
          contract.consumerLabel, "source SEW", description.sourceSEW,
          contract.sourceSEW))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "source LMUL", description.sourceLMUL,
          contract.sourceLMUL))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractIntField(
          contract.consumerLabel, "result SEW", description.sew,
          contract.resultSEW))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "result LMUL", description.lmul,
          contract.resultLMUL))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "tail policy", description.tailPolicy,
          contract.tailPolicy))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "mask policy", description.maskPolicy,
          contract.maskPolicy))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "config contract",
          description.configContractID, contract.core.configContractID))
    return error;
  if (llvm::Error error = validateRVVRuntimeAVLVLSelectedBoundaryContract(
          description, contract.runtimeAVLVLContract))
    return error;
  if (llvm::Error error = validateRVVRouteLocalRuntimeAVLVLMirrors(
          contract.consumerLabel, contract.runtimeAVLVLContract,
          contract.core.runtimeControlPlanID, contract.core.runtimeABIOrder,
          contract.setVLIntrinsic, contract.core.vlCType,
          contract.emitCFullChunkVLName, contract.emitCLoopVLName,
          contract.emitCLoopInductionName))
    return error;
  if (description.runtimeABIParameters.size() !=
      contract.core.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned runtime ABI parameter count " +
        llvm::Twine(contract.core.runtimeABIParameters.size()) +
        " before artifact export");
  for (std::size_t index = 0; index < contract.core.runtimeABIParameters.size();
       ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    const support::RuntimeABIParameter &expected =
        contract.core.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(actual, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-derived runtime ABI parameter " +
          std::to_string(index) + " to mirror provider-owned parameter '" +
          expected.cName + "' as " +
          support::stringifyRuntimeABIParameterRole(expected.role) +
          " with C type '" + expected.cType + "' before artifact export");
  }

  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "route operand binding plan",
          description.routeOperandBindingPlanID,
          contract.core.routeOperandBindingPlanID))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "route operand binding facts",
          description.routeOperandBindingSummary,
          contract.core.routeOperandBindingSummary))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "contraction route-family plan",
          description.contractionRouteFamilyPlanID,
          contract.contractionRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "target leaf profile",
          description.targetLeafProfile, contract.core.targetLeafProfile))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "provider-supported mirror",
          description.providerSupportedMirror,
          contract.core.providerSupportedMirror))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "required header declarations",
          description.requiredHeaderDeclarations,
          contract.core.requiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "C type mapping summary",
          description.cTypeMappingSummary, contract.core.cTypeMappingSummary))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "typed compute op",
          description.typedComputeOpName, contract.core.typedComputeOpName))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "source memory form",
          description.sourceMemoryForm, contract.sourceMemoryForm))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "destination memory form",
          description.destinationMemoryForm, contract.destinationMemoryForm))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "widening product relation",
          description.wideningProductRelation,
          contract.wideningProductRelation))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "source vector load intrinsic",
          description.sourceVectorLoadIntrinsic,
          contract.sourceVectorLoadIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "widening product intrinsic",
          description.wideningProductIntrinsic,
          contract.wideningProductIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "low-precision primitive contract",
          description.lowPrecisionPrimitiveContractID,
          contract.lowPrecisionPrimitiveContractID))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "low-precision primitive kind",
          description.lowPrecisionPrimitiveKind,
          contract.lowPrecisionPrimitiveKind))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "low-precision primitive source dtype",
          description.lowPrecisionPrimitiveSourceElementTypeName,
          contract.lowPrecisionPrimitiveSourceElementTypeName))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "low-precision primitive product dtype",
          description.lowPrecisionPrimitiveProductElementTypeName,
          contract.lowPrecisionPrimitiveProductElementTypeName))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "low-precision primitive accumulator dtype",
          description.lowPrecisionPrimitiveAccumulatorElementTypeName,
          contract.lowPrecisionPrimitiveAccumulatorElementTypeName))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "low-precision primitive result dtype",
          description.lowPrecisionPrimitiveResultElementTypeName,
          contract.lowPrecisionPrimitiveResultElementTypeName))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "compute intrinsic", description.intrinsic,
          contract.intrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "store intrinsic",
          description.storeIntrinsic, contract.storeIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "setvl intrinsic",
          description.setVLIntrinsic, contract.setVLIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "VL C type", description.vlCType,
          contract.core.vlCType))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "source vector type",
          description.sourceVectorTypeName, contract.core.sourceVectorTypeName))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "source vector C type",
          description.sourceVectorCType, contract.core.sourceVectorCType))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "result vector type",
          description.vectorTypeName, contract.core.resultVectorTypeName))
    return error;
  if (llvm::Error error = requireRVVWideningProductContractStringField(
          contract.consumerLabel, "result vector C type",
          description.vectorCType, contract.core.resultVectorCType))
    return error;

  return validateRVVWideningProductNoStaleNonFamilyProviderFacts(description);
}

llvm::Error validateRVVWideningProductRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVWideningProductRouteValidationContract &contract) {
  if (contract.core.requiredHeaderDeclarations.empty() ||
      contract.requiredHeaders.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived required_header_declarations before "
        "accepting the route artifact");
  for (llvm::StringRef header : contract.requiredHeaders) {
    if (header.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " saw an empty provider route header declaration");
    if (!routeHasHeader(route, header))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route header '" + header +
          "' before artifact export");
  }
  return llvm::Error::success();
}

llvm::Error validateRVVWideningProductRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVWideningProductRouteValidationContract &contract) {
  if (contract.core.cTypeMappingSummary.empty() || contract.typeMappings.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived route type mapping facts before artifact "
        "export");
  for (const plugin::rvv::RVVWideningProductRouteTypeMappingContract &mapping :
       contract.typeMappings) {
    if (mapping.sourceType.empty() || mapping.cType.empty())
      return makeRVVTargetRouteError(llvm::Twine(contract.consumerLabel) +
                                     " requires provider-derived " +
                                     mapping.label +
                                     " facts before artifact export");
    if (!routeHasTypeMapping(route, mapping.sourceType, mapping.cType))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route type mapping '" +
          mapping.sourceType + "' -> '" + mapping.cType + "'");
  }
  return llvm::Error::success();
}

llvm::Error validateRVVWideningProductRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVWideningProductRouteValidationContract &contract) {
  if (contract.core.runtimeABIOrder.empty() ||
      contract.core.runtimeABIParameters.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived runtime ABI order and ABI parameters "
        "before artifact export");
  if (route.getABIMappings().size() !=
      contract.core.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires rebuilt provider route ABI mapping count " +
        llvm::Twine(contract.core.runtimeABIParameters.size()) +
        " but route has " +
        llvm::Twine(route.getABIMappings().size()));
  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        contract.core.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }
  return llvm::Error::success();
}

llvm::Error validateRVVWideningProductRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVWideningProductRouteValidationContract &contract) {
  const auto &description = contract;
  const plugin::rvv::RVVContractionArtifactContractCore &core = contract.core;
  const llvm::StringRef consumerLabel = contract.consumerLabel;
  const plugin::rvv::RVVRuntimeAVLVLSelectedBoundaryContract &runtimeContract =
      contract.runtimeAVLVLContract;
  if (core.runtimeABIParameters.size() != 4)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived widening-product ABI parameters before "
        "validating route statements");
  if (description.resultName.empty() || core.sourceVectorCType.empty() ||
      core.resultVectorCType.empty() || runtimeContract.vlCType.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived result, source/result vector C type, and "
        "VL C type facts before validating route statements");

  const support::RuntimeABIParameter &lhsABI =
      core.runtimeABIParameters[0];
  const support::RuntimeABIParameter &rhsABI =
      core.runtimeABIParameters[1];
  const support::RuntimeABIParameter &outABI =
      core.runtimeABIParameters[2];
  const support::RuntimeABIParameter &runtimeNABI =
      core.runtimeABIParameters[3];
  if (!runtimeABIParameterEquals(runtimeNABI,
                                 runtimeContract.runtimeAVLParameter))
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires runtime n/AVL ABI role to match the selected widening "
        "product ABI order before validating route statements");

  if (route.getCallOpaqueSteps().size() != contract.expectedPreLoopStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires exact provider-built pre-loop statement count " +
        llvm::Twine(contract.expectedPreLoopStepCount) +
        " before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps()[0];
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          preLoopSetVL, consumerLabel, "pre-loop setvl",
          runtimeContract.setVLIntrinsic,
          {{runtimeContract.runtimeAVLParameter.cName,
            runtimeContract.runtimeAVLParameter.cType}},
          runtimeContract.emitCFullChunkVLName, runtimeContract.vlCType))
    return error;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "low-precision widening-product target artifact consumer requires "
          "pre-loop statements to carry selected typed RVV source provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "low-precision widening-product target artifact consumer requires "
        "exactly one provider-built runtime AVL/VL loop before artifact "
        "export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != runtimeContract.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != runtimeContract.vlCType ||
      loop.upperBound.expression != runtimeContract.runtimeAVLParameter.cName ||
      loop.upperBound.cType != runtimeContract.runtimeAVLParameter.cType ||
      loop.step.expression != runtimeContract.emitCFullChunkVLName ||
      loop.step.cType != runtimeContract.vlCType)
    return makeRVVTargetRouteError(
        "low-precision widening-product target artifact consumer requires "
        "provider-built loop bounds and step to mirror runtime AVL/VL route "
        "facts");
  if (loop.bodySteps.size() != contract.expectedLoopBodyStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires exact provider-built widening-product loop statement "
        "count " +
        llvm::Twine(contract.expectedLoopBodyStepCount) +
        " before artifact export");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "low-precision widening-product target artifact consumer requires "
          "loop statements to carry selected typed RVV source provenance");

  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeContract.runtimeAVLParameter.cName) + " - " +
       runtimeContract.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[0], consumerLabel, "loop setvl",
          runtimeContract.setVLIntrinsic,
          {{expectedRemainingAVL, runtimeContract.vlCType}},
          runtimeContract.emitCLoopVLName, runtimeContract.vlCType))
    return error;
  auto validateUnitSourceLoad =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          const support::RuntimeABIParameter &abi, llvm::StringRef resultName,
          llvm::StringRef stepLabel) -> llvm::Error {
    const std::string expectedPointer =
        (llvm::StringRef(abi.cName) + " + " +
         runtimeContract.emitCLoopInductionName)
            .str();
    return validateRVVProviderBuiltRouteStep(
        step, consumerLabel, stepLabel, description.sourceVectorLoadIntrinsic,
        {{expectedPointer, abi.cType},
         {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
        resultName, core.sourceVectorCType);
  };
  if (llvm::Error error =
          validateUnitSourceLoad(loop.bodySteps[1], lhsABI, "lhs_vec",
                                 "lhs source load"))
    return error;
  if (llvm::Error error =
          validateUnitSourceLoad(loop.bodySteps[2], rhsABI, "rhs_vec",
                                 "rhs source load"))
    return error;
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[3], consumerLabel, "widening product",
          description.wideningProductIntrinsic,
          {{"lhs_vec", core.sourceVectorCType},
           {"rhs_vec", core.sourceVectorCType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
          description.resultName, core.resultVectorCType))
    return error;
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[4], consumerLabel, "output store",
          description.storeIntrinsic,
          {{(llvm::StringRef(outABI.cName) + " + " +
             runtimeContract.emitCLoopInductionName)
                .str(),
            outABI.cType},
           {description.resultName, core.resultVectorCType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}}))
    return error;
  return llvm::Error::success();
}

llvm::Error validateRVVWideningProductRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  const std::optional<
      plugin::rvv::RVVWideningProductRouteValidationContract>
      contract = getRVVWideningProductTargetRouteValidationContract(
          description);
  if (!contract)
    return makeRVVTargetRouteError(
        "low-precision widening-product target artifact consumer requires "
        "provider-owned route validation contract before artifact export");
  if (route.getRouteID() != contract->core.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract->consumerLabel) +
        " requires rebuilt provider route id '" + contract->core.emitCRouteID +
        "' but route carried '" + route.getRouteID() + "'");
  if (llvm::Error error =
          validateRVVWideningProductDescriptionAgainstContract(description,
                                                               *contract))
    return error;
  if (llvm::Error error =
          validateRVVWideningProductRouteHeaders(route, *contract))
    return error;
  if (llvm::Error error =
          validateRVVWideningProductRouteTypeMappings(route, *contract))
    return error;
  if (llvm::Error error =
          validateRVVWideningProductRouteABIMappings(route, *contract))
    return error;
  return validateRVVWideningProductRouteStatementPlan(route, *contract);
}

llvm::Error validateRVVWideningProductTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVWideningProductRoutePayloadFacts(context.route,
                                                     context.description);
}

llvm::Error requireEmptyWideningProductStaleMirror(
    const TargetArtifactCandidate &candidate, llvm::StringRef key,
    llvm::StringRef label) {
  return requireCandidateMetadataMirror(candidate, key, "", label);
}

llvm::Error validateRVVWideningProductTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description =
      context.description;
  const std::optional<
      plugin::rvv::RVVWideningProductRouteValidationContract>
      contract = getRVVWideningProductTargetRouteValidationContract(
          description);
  if (!contract)
    return makeRVVTargetRouteError(
        "low-precision widening-product target artifact consumer requires "
        "provider-owned route validation contract before validating candidate "
        "mirrors");
  const std::string sourceSEW = llvm::Twine(contract->sourceSEW).str();
  const std::string resultSEW = llvm::Twine(contract->resultSEW).str();

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_plan",
          contract->core.routeOperandBindingPlanID,
          "selected typed RVV widening product binding plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_operands",
          contract->core.routeOperandBindingSummary,
          "selected typed RVV widening product binding summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.provider_supported_mirror",
          contract->core.providerSupportedMirror,
          "selected typed RVV widening product provider support"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.contraction_route_family_plan",
          contract->contractionRouteFamilyPlanID,
          "selected typed RVV widening product contraction route-family plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.memory_form",
          plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
              contract->core.memoryForm),
          "selected typed RVV widening product memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_control_plan",
          contract->core.runtimeControlPlanID,
          "route-local runtime AVL/VL control plan mirror"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order",
          contract->core.runtimeABIOrder,
          "route-local runtime AVL/VL ABI order mirror"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.required_header_declarations",
          contract->core.requiredHeaderDeclarations,
          "selected typed RVV widening product route header requirements"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.c_type_mapping",
          contract->core.cTypeMappingSummary,
          "selected typed RVV widening product route type mapping summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.target_leaf_profile",
          contract->core.targetLeafProfile,
          "selected typed RVV widening product target leaf profile"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_sew", sourceSEW,
          "selected typed RVV widening product i8 source SEW"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_lmul", contract->sourceLMUL,
          "selected typed RVV widening product source LMUL"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.result_sew", resultSEW,
          "selected typed RVV widening product i16 result SEW"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.result_lmul", contract->resultLMUL,
          "selected typed RVV widening product result LMUL"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_memory_form",
          contract->sourceMemoryForm,
          "selected typed RVV widening product source memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.destination_memory_form",
          contract->destinationMemoryForm,
          "selected typed RVV widening product destination memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.widening_product_relation",
          contract->wideningProductRelation,
          "selected typed RVV widening product relation"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.widening_product_intrinsic",
          contract->wideningProductIntrinsic,
          "selected typed RVV widening product intrinsic"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.low_precision_primitive.contract",
          contract->lowPrecisionPrimitiveContractID,
          "selected typed RVV widening product low-precision primitive "
          "contract"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.low_precision_primitive.kind",
          contract->lowPrecisionPrimitiveKind,
          "selected typed RVV widening product low-precision primitive kind"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.low_precision_primitive.source_dtype",
          contract->lowPrecisionPrimitiveSourceElementTypeName,
          "selected typed RVV widening product low-precision primitive source "
          "dtype"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.low_precision_primitive.product_dtype",
          contract->lowPrecisionPrimitiveProductElementTypeName,
          "selected typed RVV widening product low-precision primitive product "
          "dtype"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.low_precision_primitive.accumulator_dtype",
          contract->lowPrecisionPrimitiveAccumulatorElementTypeName,
          "selected typed RVV widening product low-precision primitive "
          "accumulator dtype"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.low_precision_primitive.result_dtype",
          contract->lowPrecisionPrimitiveResultElementTypeName,
          "selected typed RVV widening product low-precision primitive result "
          "dtype"))
    return error;

  constexpr llvm::StringLiteral staleRouteFamilyMirrors[] = {
      "tcrv_rvv.elementwise_arithmetic_route_family_plan",
      "tcrv_rvv.scalar_broadcast_elementwise_route_family_plan",
      "tcrv_rvv.runtime_scalar_splat_store_route_family_plan",
      "tcrv_rvv.widening_conversion_route_family_plan",
      "tcrv_rvv.plain_compare_select_route_family_plan",
      "tcrv_rvv.computed_mask_select_route_family_plan",
      "tcrv_rvv.computed_mask_memory_route_family_plan",
      "tcrv_rvv.segment2_memory_route_family_plan",
      "tcrv_rvv.plain_macc_route_family_plan",
      "tcrv_rvv.scalar_broadcast_macc_route_family_plan",
      "tcrv_rvv.accumulation_route_family_plan",
      "tcrv_rvv.standalone_reduction_route_family_plan",
      "tcrv_rvv.base_memory_movement_route_family_plan",
      "tcrv_rvv.mask_tail_policy_route_family_plan",
      "tcrv_rvv.mask_tail_policy_owner",
      "tcrv_rvv.accumulator_sew",
      "tcrv_rvv.accumulator_lmul",
      "tcrv_rvv.widening_macc_relation",
      "tcrv_rvv.widening_dot_accumulator_layout",
      "tcrv_rvv.widening_dot_result_layout",
      "tcrv_rvv.widening_dot_relation",
      "tcrv_rvv.widening_dot_source_accumulator_result_contract",
      "tcrv_rvv.widening_dot_reduction_store_vl",
      "tcrv_rvv.masked_widening_product_intrinsic"};
  for (llvm::StringRef key : staleRouteFamilyMirrors)
    if (llvm::Error error = requireEmptyWideningProductStaleMirror(
            candidate, key,
            "selected typed RVV non-widening-product route-family mirror"))
      return error;

  return llvm::Error::success();
}

llvm::Error validateRVVWideningDotReductionNoStaleNonFamilyProviderFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (!description.elementwiseArithmeticRouteFamilyPlanID.empty() ||
      !description.scalarBroadcastElementwiseRouteFamilyPlanID.empty() ||
      !description.runtimeScalarSplatStoreRouteFamilyPlanID.empty() ||
      !description.wideningConversionRouteFamilyPlanID.empty() ||
      !description.plainMAccRouteFamilyPlanID.empty() ||
      !description.scalarBroadcastMAccRouteFamilyPlanID.empty() ||
      !description.accumulationRouteFamilyPlanID.empty() ||
      !description.standaloneReductionRouteFamilyPlanID.empty() ||
      !description.plainCompareSelectRouteFamilyPlanID.empty() ||
      !description.computedMaskSelectRouteFamilyPlanID.empty() ||
      !description.computedMaskMemoryRouteFamilyPlanID.empty() ||
      !description.segment2MemoryRouteFamilyPlanID.empty() ||
      !description.baseMemoryMovementRouteFamilyPlanID.empty() ||
      !description.wideningMAccRelation.empty() ||
      !description.maccAccumulatorLayout.empty() ||
      !description.maccResultLayout.empty() ||
      !description.wideningMAccAccumulatorLayout.empty() ||
      !description.wideningMAccResultLayout.empty())
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer rejects stale "
        "non-widening-dot route-family facts");
  return llvm::Error::success();
}

llvm::Error requireRVVWideningDotContractStringField(
    llvm::StringRef consumerLabel, llvm::StringRef fieldLabel,
    llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  if (expected.empty())
    return makeRVVTargetRouteError(llvm::Twine(consumerLabel) +
                                   " rejects stale " + fieldLabel +
                                   " facts before artifact export");
  return makeRVVTargetRouteError(llvm::Twine(consumerLabel) +
                                 " requires provider-derived " + fieldLabel +
                                 " '" + expected + "' but was '" + actual +
                                 "'");
}

llvm::Error requireRVVWideningDotContractIntField(
    llvm::StringRef consumerLabel, llvm::StringRef fieldLabel,
    std::int64_t actual, std::int64_t expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVTargetRouteError(llvm::Twine(consumerLabel) +
                                 " requires provider-derived " + fieldLabel +
                                 " " + llvm::Twine(expected) + " but was " +
                                 llvm::Twine(actual));
}

llvm::Error validateRVVLowPrecisionWideningReductionPrimitiveProviderFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVWideningDotReduceRouteValidationContract &contract) {
  const plugin::rvv::RVVLowPrecisionWideningReductionPrimitiveFacts
      &primitive = contract.lowPrecisionWideningReductionPrimitiveFacts;
  if (!primitive.hasFacts)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned low-precision widening-reduction primitive "
        "facts before artifact export");
  if (primitive.contractID.empty() || primitive.kind.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires non-empty low-precision widening-reduction primitive "
        "contract and kind before artifact export");
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel,
          "low-precision widening-reduction primitive source dtype",
          description.lowPrecisionPrimitiveSourceElementTypeName,
          primitive.sourceElementTypeName))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel,
          "low-precision widening-reduction primitive product dtype",
          description.lowPrecisionPrimitiveProductElementTypeName,
          primitive.productElementTypeName))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel,
          "low-precision widening-reduction primitive accumulator dtype",
          description.lowPrecisionPrimitiveAccumulatorElementTypeName,
          primitive.accumulatorElementTypeName))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel,
          "low-precision widening-reduction primitive final result dtype",
          description.lowPrecisionPrimitiveResultElementTypeName,
          primitive.finalResultElementTypeName))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel,
          "low-precision widening-reduction primitive source SEW",
          description.sourceSEW, primitive.sourceSEW))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel,
          "low-precision widening-reduction primitive source LMUL",
          description.sourceLMUL, primitive.sourceLMUL))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel,
          "low-precision widening-reduction primitive product SEW",
          description.productSEW, primitive.productSEW))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel,
          "low-precision widening-reduction primitive product LMUL",
          description.productLMUL, primitive.productLMUL))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel,
          "low-precision widening-reduction primitive accumulator SEW",
          description.sew, primitive.accumulatorSEW))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel,
          "low-precision widening-reduction primitive accumulator LMUL",
          description.lmul, primitive.accumulatorLMUL))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel,
          "low-precision widening-reduction primitive result SEW",
          description.sew, primitive.reductionResultSEW))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel,
          "low-precision widening-reduction primitive result LMUL",
          description.lmul, primitive.reductionResultLMUL))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel,
          "low-precision widening-reduction primitive product relation",
          description.wideningProductRelation,
          primitive.wideningProductRelation))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel,
          "low-precision widening-reduction primitive reduction relation",
          description.productReductionChainRelation,
          primitive.productReductionChainRelation))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel,
          "low-precision widening-reduction primitive product intrinsic",
          description.wideningProductIntrinsic,
          primitive.wideningProductIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel,
          "low-precision widening-reduction primitive reduction intrinsic",
          description.intrinsic, primitive.reductionIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel,
          "low-precision widening-reduction primitive scalar seed splat",
          description.scalarSeedSplatIntrinsic,
          primitive.scalarSeedSplatIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel,
          "low-precision widening-reduction primitive accumulator layout",
          description.reductionAccumulatorLayout,
          primitive.accumulatorLayout))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel,
          "low-precision widening-reduction primitive result layout",
          description.reductionResultLayout, primitive.resultLayout))
    return error;
  return requireRVVWideningDotContractStringField(
      contract.consumerLabel,
      "low-precision widening-reduction primitive store VL",
      description.reductionStoreVL, primitive.reductionStoreVL);
}

llvm::Error validateRVVPackedI4LowPrecisionResourceProviderFacts(
    const plugin::rvv::RVVWideningDotReduceRouteValidationContract &contract) {
  const plugin::rvv::RVVLowPrecisionContractionResourceSelection &selection =
      contract.lowPrecisionResourceSelection;
  if (!selection.hasSelection ||
      !plugin::rvv::isRVVLowPrecisionResourcePackedI4CandidateID(
          selection.selectedCandidateID))
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-selected packed-i4 low-precision resource facts "
        "before artifact export");
  if (!selection.isLegal)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-selected packed-i4 low-precision resource facts "
        "to be legal before artifact export");
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "packed-i4 operand form",
          selection.operandForm,
          plugin::rvv::kRVVLowPrecisionResourceOperandFormPackedI4Nibbles))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "packed-i4 source signedness",
          selection.sourceSignedness,
          plugin::rvv::kRVVLowPrecisionResourceSourceSignednessSigned))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel, "packed-i4 storage element width",
          selection.storageElementWidth,
          plugin::rvv::kRVVLowPrecisionResourcePackedI4StorageElementWidth))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel, "packed-i4 effective element width",
          selection.effectiveElementWidth,
          plugin::rvv::kRVVLowPrecisionResourcePackedI4EffectiveElementWidth))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "packed-i4 packing layout",
          selection.packingLayout,
          plugin::rvv::kRVVLowPrecisionResourcePackingLayoutPackedI4Nibbles))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "packed-i4 unpack intent",
          selection.unpackIntent,
          plugin::rvv::kRVVLowPrecisionResourceUnpackIntentPackedI4Nibbles))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel, "packed-i4 unroll factor",
          selection.unrollFactor,
          plugin::rvv::kRVVLowPrecisionResourcePackedI4Unroll))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel, "packed-i4 accumulator count",
          selection.accumulatorCount,
          plugin::rvv::kRVVLowPrecisionResourcePackedI4AccumulatorCount))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel, "packed-i4 vsetvl region count",
          selection.vsetvlRegionCount,
          plugin::rvv::kRVVLowPrecisionResourcePackedI4VSetVLRegions))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel, "packed-i4 peak live vector groups",
          selection.peakLiveVectorGroups,
          plugin::rvv::kRVVLowPrecisionResourcePackedI4PeakLiveVectorGroups))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "packed-i4 realization producer",
          selection.realizationProducer,
          plugin::rvv::kRVVLowPrecisionResourceRealizationProducer))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "packed-i4 realization decision",
          selection.realizationDecision,
          plugin::rvv::kRVVLowPrecisionResourcePackedI4RealizationDecision))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel, "packed-i4 realized unroll factor",
          selection.realizedUnrollFactor,
          plugin::rvv::kRVVLowPrecisionResourcePackedI4Unroll))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel, "packed-i4 realized vsetvl region count",
          selection.realizedVSetVLRegionCount,
          plugin::rvv::kRVVLowPrecisionResourcePackedI4VSetVLRegions))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel,
          "packed-i4 realized peak live vector groups",
          selection.realizedPeakLiveVectorGroups,
          plugin::rvv::kRVVLowPrecisionResourcePackedI4PeakLiveVectorGroups))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel, "packed-i4 product region index",
          selection.productRegionIndex,
          plugin::rvv::
              getRVVLowPrecisionResourceProductRegionIndexForRealizationDecision(
                  plugin::rvv::kRVVLowPrecisionResourcePackedI4RealizationDecision)))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel, "packed-i4 dequant region index",
          selection.dequantRegionIndex,
          plugin::rvv::
              getRVVLowPrecisionResourceDequantRegionIndexForRealizationDecision(
                  plugin::rvv::kRVVLowPrecisionResourcePackedI4RealizationDecision)))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "packed-i4 product phase",
          selection.productPhase,
          plugin::rvv::
              getRVVLowPrecisionResourceProductPhaseForRealizationDecision(
                  plugin::rvv::kRVVLowPrecisionResourcePackedI4RealizationDecision)))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "packed-i4 dequant phase",
          selection.dequantPhase, "dequant-store"))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "packed-i4 performance feedback",
          selection.performanceFeedback,
          plugin::rvv::kRVVLowPrecisionResourcePackedI4PerformanceFeedback))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "packed-i4 performance baseline",
          selection.performanceBaseline,
          plugin::rvv::kRVVLowPrecisionResourcePackedI4PerformanceBaseline))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "packed-i4 performance best-speedup range",
          selection.performanceBestSpeedupRange,
          plugin::rvv::
              kRVVLowPrecisionResourcePackedI4PerformanceBestSpeedupRange))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "packed-i4 performance action",
          selection.performanceAction,
          plugin::rvv::kRVVLowPrecisionResourcePackedI4PerformanceAction))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "packed-i4 performance maturity",
          selection.performanceMaturity,
          plugin::rvv::kRVVLowPrecisionResourcePackedI4PerformanceMaturity))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "packed-i4 performance maturity evidence",
          selection.performanceMaturityEvidence,
          plugin::rvv::
              kRVVLowPrecisionResourcePackedI4PerformanceMaturityEvidence))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "packed-i4 performance maturity outcome",
          selection.performanceMaturityOutcome,
          plugin::rvv::
              kRVVLowPrecisionResourcePackedI4PerformanceMaturityOutcome))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "packed-i4 performance selection eligibility",
          selection.performanceSelectionEligible,
          plugin::rvv::
              kRVVLowPrecisionResourcePackedI4PerformanceSelectionEligible))
    return error;
  return requireRVVWideningDotContractStringField(
      contract.consumerLabel, "packed-i4 dispatch preference",
      selection.dispatchPreference,
      plugin::rvv::kRVVLowPrecisionResourcePackedI4DispatchPreference);
}

llvm::Error validateRVVLowPrecisionPrimitiveChainResourceProviderFacts(
    const plugin::rvv::RVVWideningDotReduceRouteValidationContract &contract) {
  const plugin::rvv::RVVLowPrecisionContractionResourceSelection &selection =
      contract.lowPrecisionResourceSelection;
  const plugin::rvv::RVVLowPrecisionWideningReductionPrimitiveFacts
      &primitive = contract.lowPrecisionWideningReductionPrimitiveFacts;
  if (!selection.hasSelection || !primitive.hasFacts)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned low-precision resource and "
        "widening-reduction primitive facts before artifact export");
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "resource primitive contract",
          selection.primitiveContractID,
          primitive.lowPrecisionPrimitiveContractID))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "resource primitive kind",
          selection.primitiveKind, primitive.lowPrecisionPrimitiveKind))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "resource primitive chain contract",
          selection.primitiveChainContractID, primitive.contractID))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "resource primitive chain kind",
          selection.primitiveChainKind, primitive.kind))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel,
          "resource primitive widening product relation",
          selection.primitiveWideningProductRelation,
          primitive.wideningProductRelation))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel,
          "resource primitive product-reduction chain relation",
          selection.primitiveProductReductionChainRelation,
          primitive.productReductionChainRelation))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel,
          "resource primitive widening product intrinsic",
          selection.primitiveWideningProductIntrinsic,
          primitive.wideningProductIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "resource primitive reduction intrinsic",
          selection.primitiveReductionIntrinsic, primitive.reductionIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel,
          "resource primitive scalar seed splat intrinsic",
          selection.primitiveScalarSeedSplatIntrinsic,
          primitive.scalarSeedSplatIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "resource primitive accumulator layout",
          selection.primitiveAccumulatorLayout, primitive.accumulatorLayout))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "resource primitive result layout",
          selection.primitiveResultLayout, primitive.resultLayout))
    return error;
  return requireRVVWideningDotContractStringField(
      contract.consumerLabel, "resource primitive reduction store VL",
      selection.primitiveReductionStoreVL, primitive.reductionStoreVL);
}

llvm::Error validateRVVLowPrecisionProductReductionRealizationProviderFacts(
    const plugin::rvv::RVVWideningDotReduceRouteValidationContract &contract) {
  const plugin::rvv::RVVLowPrecisionContractionResourceSelection &selection =
      contract.lowPrecisionResourceSelection;
  if (!selection.hasSelection)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned low-precision product-reduction "
        "resource/realization facts before artifact export");
  if (!selection.isLegal)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned low-precision product-reduction "
        "resource/realization facts to be legal before artifact export");
  if (selection.rejectionReason !=
      plugin::rvv::kRVVLowPrecisionResourceNoRejectionReason)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " rejects stale low-precision product-reduction resource rejection "
        "reason '" +
        selection.rejectionReason + "' before artifact export");

  const llvm::StringRef expectedRealizationDecision =
      plugin::rvv::getRVVLowPrecisionContractionResourceRealizationDecision(
          selection.selectedCandidateID);
  if (expectedRealizationDecision.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " cannot derive provider-owned low-precision product-reduction "
        "realization decision for selected candidate '" +
        selection.selectedCandidateID + "' before artifact export");
  if (selection.peakLiveVectorGroups > selection.vectorRegisterBudget ||
      selection.realizedPeakLiveVectorGroups > selection.vectorRegisterBudget)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned low-precision product-reduction resource "
        "budget facts before artifact export");
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel, "low-precision resource unroll factor",
          selection.unrollFactor,
          plugin::rvv::getRVVLowPrecisionResourceExpectedUnrollFactor(
              selection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel, "low-precision resource accumulator count",
          selection.accumulatorCount,
          plugin::rvv::getRVVLowPrecisionResourceExpectedAccumulatorCount(
              selection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel,
          "low-precision resource vsetvl region count",
          selection.vsetvlRegionCount,
          plugin::rvv::getRVVLowPrecisionResourceExpectedVSetVLRegionCount(
              selection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel,
          "low-precision resource peak live vector groups",
          selection.peakLiveVectorGroups,
          plugin::rvv::getRVVLowPrecisionResourceExpectedPeakLiveVectorGroups(
              selection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "low-precision realization producer",
          selection.realizationProducer,
          plugin::rvv::kRVVLowPrecisionResourceRealizationProducer))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "low-precision realization decision",
          selection.realizationDecision, expectedRealizationDecision))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel, "low-precision realized unroll factor",
          selection.realizedUnrollFactor,
          plugin::rvv::getRVVLowPrecisionResourceExpectedUnrollFactor(
              selection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel,
          "low-precision realized vsetvl region count",
          selection.realizedVSetVLRegionCount,
          plugin::rvv::
              getRVVLowPrecisionResourceExpectedVSetVLRegionCountForRealizationDecision(
                  expectedRealizationDecision)))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel,
          "low-precision realized peak live vector groups",
          selection.realizedPeakLiveVectorGroups,
          plugin::rvv::getRVVLowPrecisionResourceExpectedPeakLiveVectorGroups(
              selection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel, "low-precision product region index",
          selection.productRegionIndex,
          plugin::rvv::
              getRVVLowPrecisionResourceProductRegionIndexForRealizationDecision(
                  expectedRealizationDecision)))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel, "low-precision dequant region index",
          selection.dequantRegionIndex,
          plugin::rvv::
              getRVVLowPrecisionResourceDequantRegionIndexForRealizationDecision(
                  expectedRealizationDecision)))
    return error;
  if (selection.productRegionIndex <= 0 || selection.dequantRegionIndex <= 0 ||
      selection.productRegionIndex >= selection.dequantRegionIndex ||
      selection.dequantRegionIndex > selection.realizedVSetVLRegionCount)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires ordered provider-owned low-precision product/dequant "
        "realization region facts before artifact export");
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "low-precision product phase",
          selection.productPhase,
          plugin::rvv::
              getRVVLowPrecisionResourceProductPhaseForRealizationDecision(
                  expectedRealizationDecision)))
    return error;
  return requireRVVWideningDotContractStringField(
      contract.consumerLabel, "low-precision dequant phase",
      selection.dequantPhase, "dequant-store");
}

llvm::Error validateRVVWideningDotReductionDescriptionAgainstContract(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVWideningDotReduceRouteValidationContract &contract) {
  const bool isProductReductionDequantClamp =
      contract.kind == plugin::rvv::RVVWideningDotReduceRouteValidationKind::
                           ProductReductionDequantClampF32;
  const bool isProductReductionDequantization =
      contract.kind == plugin::rvv::RVVWideningDotReduceRouteValidationKind::
                           ProductReductionDequantization ||
      isProductReductionDequantClamp;
  const bool isProductReductionChain =
      contract.kind == plugin::rvv::RVVWideningDotReduceRouteValidationKind::
                           ProductReductionChain ||
      isProductReductionDequantization;
  const bool usesPackedI4LowPrecisionProductReduction =
      isProductReductionDequantization &&
      plugin::rvv::isRVVLowPrecisionResourcePackedI4CandidateID(
          contract.lowPrecisionResourceSelection.selectedCandidateID);
  if (isProductReductionChain &&
      contract.lowPrecisionResourceSelection.hasSelection)
    if (llvm::Error error =
            validateRVVLowPrecisionPrimitiveChainResourceProviderFacts(
                contract))
      return error;
  if (isProductReductionDequantization)
    if (llvm::Error error =
            validateRVVLowPrecisionProductReductionRealizationProviderFacts(
                contract))
      return error;
  if (usesPackedI4LowPrecisionProductReduction)
    if (llvm::Error error =
            validateRVVPackedI4LowPrecisionResourceProviderFacts(contract))
      return error;
  if (description.memoryForm != contract.memoryForm)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires selected typed RVV memory form '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(contract.memoryForm) +
        "' but was '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
            description.memoryForm) +
        "'");

  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel, "source SEW", description.sourceSEW,
          contract.sourceSEW))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "source LMUL", description.sourceLMUL,
          contract.sourceLMUL))
    return error;
  if (isProductReductionChain) {
    if (llvm::Error error = requireRVVWideningDotContractIntField(
            contract.consumerLabel, "product SEW", description.productSEW,
            contract.productSEW))
      return error;
    if (llvm::Error error = requireRVVWideningDotContractStringField(
            contract.consumerLabel, "product LMUL", description.productLMUL,
            contract.productLMUL))
      return error;
  }
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel, "accumulator SEW", description.sew,
          contract.accumulatorSEW))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "accumulator LMUL", description.lmul,
          contract.accumulatorLMUL))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractIntField(
          contract.consumerLabel, "result SEW", description.sew,
          contract.resultSEW))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "result LMUL", description.lmul,
          contract.resultLMUL))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "tail policy", description.tailPolicy,
          contract.tailPolicy))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "mask policy", description.maskPolicy,
          contract.maskPolicy))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "config contract",
          description.configContractID, contract.core.configContractID))
    return error;
  if (llvm::Error error = validateRVVRuntimeAVLVLSelectedBoundaryContract(
          description, contract.runtimeAVLVLContract))
    return error;
  if (llvm::Error error = validateRVVRouteLocalRuntimeAVLVLMirrors(
          contract.consumerLabel, contract.runtimeAVLVLContract,
          contract.core.runtimeControlPlanID, contract.core.runtimeABIOrder,
          contract.setVLIntrinsic, contract.core.vlCType,
          contract.emitCFullChunkVLName, contract.emitCLoopVLName,
          contract.emitCLoopInductionName))
    return error;
  if (description.runtimeABIParameters.size() !=
      contract.core.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned runtime ABI parameter count " +
        llvm::Twine(contract.core.runtimeABIParameters.size()) +
        " before artifact export");
  for (std::size_t index = 0; index < contract.core.runtimeABIParameters.size();
       ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    const support::RuntimeABIParameter &expected =
        contract.core.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(actual, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-derived runtime ABI parameter " +
          std::to_string(index) + " to mirror provider-owned parameter '" +
          expected.cName + "' as " +
          support::stringifyRuntimeABIParameterRole(expected.role) +
          " with C type '" + expected.cType + "' before artifact export");
  }

  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "route operand binding plan",
          description.routeOperandBindingPlanID,
          contract.core.routeOperandBindingPlanID))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "route operand binding facts",
          description.routeOperandBindingSummary,
          contract.core.routeOperandBindingSummary))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "contraction route-family plan",
          description.contractionRouteFamilyPlanID,
          contract.contractionRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "target leaf profile",
          description.targetLeafProfile, contract.core.targetLeafProfile))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "provider-supported mirror",
          description.providerSupportedMirror,
          contract.core.providerSupportedMirror))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "required header declarations",
          description.requiredHeaderDeclarations,
          contract.requiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "C type mapping summary",
          description.cTypeMappingSummary, contract.cTypeMappingSummary))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "typed compute op",
          description.typedComputeOpName, contract.typedComputeOpName))
    return error;

  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "compare predicate",
          description.comparePredicateKind, contract.comparePredicateKind))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "mask role", description.maskRole,
          contract.maskRole))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "mask source", description.maskSource,
          contract.maskSource))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "mask memory form",
          description.maskMemoryForm, contract.maskMemoryForm))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "source memory form",
          description.sourceMemoryForm, contract.sourceMemoryForm))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "destination memory form",
          description.destinationMemoryForm, contract.destinationMemoryForm))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "strided memory layout",
          description.stridedMemoryLayout, contract.stridedMemoryLayout))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "lhs stride source",
          description.lhsStrideSource, contract.lhsStrideSource))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "rhs stride source",
          description.rhsStrideSource, contract.rhsStrideSource))
    return error;
  if (isProductReductionChain) {
    if (llvm::Error error = requireRVVWideningDotContractStringField(
            contract.consumerLabel, "product-reduction accumulator layout",
            description.reductionAccumulatorLayout,
            contract.wideningDotProductAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRVVWideningDotContractStringField(
            contract.consumerLabel, "product-reduction result layout",
            description.reductionResultLayout,
            contract.wideningDotProductResultLayout))
      return error;
    if (llvm::Error error = requireRVVWideningDotContractStringField(
            contract.consumerLabel, "product-reduction relation",
            description.productReductionChainRelation,
            contract.productReductionChainRelation))
      return error;
    if (llvm::Error error = requireRVVWideningDotContractStringField(
            contract.consumerLabel, "scalar result runtime boundary",
            description.standaloneReductionScalarResultRuntimeBoundary,
            isProductReductionDequantization
                ? kRVVProductReductionDequantVectorCarryBoundary
                : kRVVProductReductionOutCarryBoundary))
      return error;
    if (llvm::Error error =
            validateRVVLowPrecisionWideningReductionPrimitiveProviderFacts(
                description, contract))
      return error;
  } else {
    if (llvm::Error error = requireRVVWideningDotContractStringField(
            contract.consumerLabel, "widening dot accumulator layout",
            description.wideningDotProductAccumulatorLayout,
            contract.wideningDotProductAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRVVWideningDotContractStringField(
            contract.consumerLabel, "widening dot result layout",
            description.wideningDotProductResultLayout,
            contract.wideningDotProductResultLayout))
      return error;
    if (llvm::Error error = requireRVVWideningDotContractStringField(
            contract.consumerLabel, "widening dot relation",
            description.wideningDotProductRelation,
            contract.wideningDotProductRelation))
      return error;
    if (llvm::Error error = requireRVVWideningDotContractStringField(
            contract.consumerLabel,
            "widening dot source/accumulator/result contract",
            description.wideningDotSourceAccumulatorResultContract,
            contract.wideningDotSourceAccumulatorResultContract))
      return error;
  }
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "widening product intrinsic",
          description.wideningProductIntrinsic,
          contract.wideningProductIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "dequantization relation",
          description.dequantizationRelation, contract.dequantizationRelation))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "dequantize convert intrinsic",
          description.dequantizeConvertIntrinsic,
          contract.dequantizeConvertIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "dequantize scale intrinsic",
          description.dequantizeScaleIntrinsic,
          contract.dequantizeScaleIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "dequant scale role",
          description.dequantScaleRole, contract.dequantScaleRole))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "dequant scale C type",
          description.dequantScaleCType, contract.dequantScaleCType))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "dequant scale name",
          description.dequantScaleName, contract.dequantScaleName))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "lower bound role", description.lowerBoundRole,
          contract.lowerBoundRole))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "upper bound role", description.upperBoundRole,
          contract.upperBoundRole))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "lower bound C type",
          description.lowerBoundCType, contract.lowerBoundCType))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "upper bound C type",
          description.upperBoundCType, contract.upperBoundCType))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "bound order", description.boundOrder,
          contract.boundOrder))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "clamp relation", description.clampRelation,
          contract.clampRelation))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "select layout", description.selectLayout,
          contract.selectLayout))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "secondary compare predicate",
          description.secondaryComparePredicateKind,
          contract.secondaryComparePredicateKind))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "secondary compare intrinsic",
          description.secondaryCompareIntrinsic,
          contract.secondaryCompareIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "masked widening product intrinsic",
          description.maskedWideningProductIntrinsic,
          contract.maskedWideningProductIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "scalar seed splat intrinsic",
          description.scalarSeedSplatIntrinsic,
          contract.scalarSeedSplatIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "strided load intrinsic",
          description.stridedLoadIntrinsic, contract.stridedLoadIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "source vector load intrinsic",
          description.sourceVectorLoadIntrinsic,
          contract.sourceVectorLoadIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "compare vector load intrinsic",
          description.vectorLoadIntrinsic, contract.compareVectorLoadIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "reduction intrinsic",
          description.intrinsic, contract.reductionIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "store intrinsic",
          description.storeIntrinsic, contract.storeIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "setvl intrinsic",
          description.setVLIntrinsic, contract.setVLIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "compare intrinsic",
          description.compareIntrinsic, contract.compareIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "masked merge intrinsic",
          description.maskedMergeIntrinsic, contract.maskedMergeIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "rhs broadcast intrinsic",
          description.rhsBroadcastIntrinsic, contract.rhsBroadcastIntrinsic))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "reduction store VL",
          description.reductionStoreVL, contract.reductionStoreVL))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "inactive-lane zeroing",
          description.inactiveLaneZeroingRequirement,
          contract.inactiveLaneZeroingRequirement))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "VL C type", description.vlCType,
          contract.vlCType))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "source vector type",
          description.sourceVectorTypeName, contract.sourceVectorTypeName))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "source vector C type",
          description.sourceVectorCType, contract.sourceVectorCType))
    return error;
  if (isProductReductionChain) {
    if (llvm::Error error = requireRVVWideningDotContractStringField(
            contract.consumerLabel, "product vector type",
            description.productVectorTypeName, contract.productVectorTypeName))
      return error;
    if (llvm::Error error = requireRVVWideningDotContractStringField(
            contract.consumerLabel, "product vector C type",
            description.productVectorCType, contract.productVectorCType))
      return error;
  }
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "result vector type",
          description.vectorTypeName, contract.resultVectorTypeName))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "result vector C type",
          description.vectorCType, contract.resultVectorCType))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "mask type", description.maskTypeName,
          contract.maskTypeName))
    return error;
  if (llvm::Error error = requireRVVWideningDotContractStringField(
          contract.consumerLabel, "mask C type", description.maskCType,
          contract.maskCType))
    return error;

  return validateRVVWideningDotReductionNoStaleNonFamilyProviderFacts(
      description);
}

llvm::Error validateRVVNonComputedMaskWideningDotReductionRoutePayloadFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  const std::optional<
      plugin::rvv::RVVWideningDotReduceRouteValidationContract>
      contract =
          getRVVWideningDotReductionTargetRouteValidationContract(description);
  if (!contract ||
      !isRVVNonComputedMaskWideningDotReductionRouteFamilyOperation(
          description.operation))
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-owned route validation contract before artifact export");
  return validateRVVWideningDotReductionDescriptionAgainstContract(
      description, *contract);
}

llvm::Error validateRVVComputedMaskWideningDotReductionRoutePayloadFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  const std::optional<
      plugin::rvv::RVVWideningDotReduceRouteValidationContract>
      contract =
          getRVVWideningDotReductionTargetRouteValidationContract(description);
  if (!contract ||
      !isRVVComputedMaskWideningDotReductionRouteFamilyOperation(
          description.operation))
    return makeRVVTargetRouteError(
        "computed-mask widening dot-reduction target artifact consumer "
        "requires provider-owned route validation contract before artifact "
        "export");
  return validateRVVWideningDotReductionDescriptionAgainstContract(
      description, *contract);
}

llvm::Error validateRVVWideningDotReductionRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVWideningDotReduceRouteValidationContract &contract) {
  if (contract.core.requiredHeaderDeclarations.empty() ||
      contract.requiredHeaders.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires "
        "provider-derived required_header_declarations before accepting the "
        "route artifact");

  for (llvm::StringRef header : contract.requiredHeaders) {
    if (header.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " saw an empty provider route header declaration");
    if (!routeHasHeader(route, header))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route header '" + header +
          "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVWideningDotReductionRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVWideningDotReduceRouteValidationContract &contract) {
  if (contract.core.cTypeMappingSummary.empty() ||
      contract.typeMappings.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived route type mapping facts before artifact "
        "export");

  for (const plugin::rvv::RVVWideningDotReduceRouteTypeMappingContract
           &mapping : contract.typeMappings) {
    if (mapping.sourceType.empty() || mapping.cType.empty())
      return makeRVVTargetRouteError(llvm::Twine(contract.consumerLabel) +
                                     " requires provider-derived " +
                                     mapping.label +
                                     " facts before artifact export");
    if (!routeHasTypeMapping(route, mapping.sourceType, mapping.cType))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route type mapping '" +
          mapping.sourceType + "' -> '" + mapping.cType + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVWideningDotReductionRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVWideningDotReduceRouteValidationContract &contract) {
  if (contract.core.runtimeABIOrder.empty() ||
      contract.core.runtimeABIParameters.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires "
        "provider-derived runtime ABI order and ABI parameters before "
        "artifact export");
  if (route.getABIMappings().size() !=
      contract.core.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires rebuilt provider route ABI mapping count " +
        llvm::Twine(contract.core.runtimeABIParameters.size()) +
        " but route has " + llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        contract.core.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVWideningDotReductionRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVWideningDotReduceRouteValidationContract &contract) {
  const auto &description = contract;
  const llvm::StringRef consumerLabel = contract.consumerLabel;
  const plugin::rvv::RVVRuntimeAVLVLSelectedBoundaryContract &runtimeContract =
      contract.runtimeAVLVLContract;
  const bool isComputedMask =
      contract.kind ==
          plugin::rvv::RVVWideningDotReduceRouteValidationKind::ComputedMask ||
      contract.kind == plugin::rvv::
                           RVVWideningDotReduceRouteValidationKind::
                               ComputedMaskStridedInput;
  const bool isProductReductionChain =
      contract.kind == plugin::rvv::RVVWideningDotReduceRouteValidationKind::
                           ProductReductionChain ||
      contract.kind == plugin::rvv::RVVWideningDotReduceRouteValidationKind::
                           ProductReductionDequantization ||
      contract.kind == plugin::rvv::RVVWideningDotReduceRouteValidationKind::
                           ProductReductionDequantClampF32;
  const bool isProductReductionDequantClamp =
      contract.kind == plugin::rvv::RVVWideningDotReduceRouteValidationKind::
                           ProductReductionDequantClampF32;
  const bool isProductReductionDequantization =
      contract.kind == plugin::rvv::RVVWideningDotReduceRouteValidationKind::
                           ProductReductionDequantization ||
      isProductReductionDequantClamp;
  const bool isStrided =
      contract.kind ==
          plugin::rvv::RVVWideningDotReduceRouteValidationKind::StridedInput ||
      contract.kind == plugin::rvv::
                           RVVWideningDotReduceRouteValidationKind::
                               ComputedMaskStridedInput;
  const bool usesGroupedLowPrecisionProductReduction =
      isProductReductionDequantization &&
      plugin::rvv::isRVVLowPrecisionResourceGroupedCandidateID(
          contract.lowPrecisionResourceSelection.selectedCandidateID);
  const bool usesPackedI4LowPrecisionProductReduction =
      isProductReductionDequantization &&
      plugin::rvv::isRVVLowPrecisionResourcePackedI4CandidateID(
          contract.lowPrecisionResourceSelection.selectedCandidateID);
  if (description.runtimeABIParameters.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived widening dot ABI parameters before "
        "validating route statements");
  if (description.resultName.empty() || description.sourceVectorCType.empty() ||
      description.vectorCType.empty() || runtimeContract.vlCType.empty() ||
      (isProductReductionChain && description.productVectorCType.empty()) ||
      (isProductReductionDequantization &&
       (description.rhsBroadcastIntrinsic.empty() ||
        description.dequantScaleRole.empty() ||
        description.dequantScaleCType.empty() ||
        description.dequantScaleName.empty())))
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived result, source/result vector C type, "
        "optional product vector C type, optional scalar-dequant splat/scale "
        "facts, and VL C type facts before validating route statements");
  if (isComputedMask &&
      (description.maskName.empty() || description.maskCType.empty()))
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived computed-mask name and C type before "
        "validating route statements");

  const support::RuntimeABIParameter *cmpLHSABI = nullptr;
  const support::RuntimeABIParameter *cmpRHSABI = nullptr;
  const support::RuntimeABIParameter *lhsABI = nullptr;
  const support::RuntimeABIParameter *rhsABI = nullptr;
  const support::RuntimeABIParameter *dotLHSABI = nullptr;
  const support::RuntimeABIParameter *dotRHSABI = nullptr;
  const support::RuntimeABIParameter *accumulatorABI = nullptr;
  const support::RuntimeABIParameter *dequantScaleABI = nullptr;
  const support::RuntimeABIParameter *lowerBoundABI = nullptr;
  const support::RuntimeABIParameter *upperBoundABI = nullptr;
  const support::RuntimeABIParameter *outABI = nullptr;
  const support::RuntimeABIParameter *runtimeNABI = nullptr;
  const support::RuntimeABIParameter *lhsStrideABI = nullptr;
  const support::RuntimeABIParameter *rhsStrideABI = nullptr;

  if (isComputedMask) {
    cmpLHSABI = &description.runtimeABIParameters[0];
    cmpRHSABI = &description.runtimeABIParameters[1];
    dotLHSABI = &description.runtimeABIParameters[2];
    dotRHSABI = &description.runtimeABIParameters[3];
    accumulatorABI = &description.runtimeABIParameters[4];
    outABI = &description.runtimeABIParameters[5];
    runtimeNABI = &description.runtimeABIParameters[6];
    if (isStrided) {
      lhsStrideABI = &description.runtimeABIParameters[7];
      rhsStrideABI = &description.runtimeABIParameters[8];
    }
  } else {
    lhsABI = &description.runtimeABIParameters[0];
    rhsABI = &description.runtimeABIParameters[1];
    accumulatorABI = &description.runtimeABIParameters[2];
    if (isProductReductionDequantClamp) {
      dequantScaleABI = &description.runtimeABIParameters[3];
      lowerBoundABI = &description.runtimeABIParameters[4];
      upperBoundABI = &description.runtimeABIParameters[5];
      outABI = &description.runtimeABIParameters[6];
      runtimeNABI = &description.runtimeABIParameters[7];
    } else if (isProductReductionDequantization) {
      dequantScaleABI = &description.runtimeABIParameters[3];
      outABI = &description.runtimeABIParameters[4];
      runtimeNABI = &description.runtimeABIParameters[5];
    } else {
      outABI = &description.runtimeABIParameters[3];
      runtimeNABI = &description.runtimeABIParameters[4];
    }
    if (isStrided) {
      lhsStrideABI = &description.runtimeABIParameters[5];
      rhsStrideABI = &description.runtimeABIParameters[6];
    }
  }

  const support::RuntimeABIParameter *runtimeElementCount = nullptr;
  for (const support::RuntimeABIParameter &parameter :
       description.runtimeABIParameters)
    if (parameter.role ==
        support::RuntimeABIParameterRole::RuntimeElementCount) {
      runtimeElementCount = &parameter;
      break;
    }
  if (!runtimeElementCount || runtimeElementCount != runtimeNABI ||
      !runtimeABIParameterEquals(*runtimeNABI,
                                 runtimeContract.runtimeAVLParameter))
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires runtime n/AVL ABI role to match the selected widening dot "
        "ABI order before validating route statements");

  const llvm::StringRef scalarI32CType = "int32_t";
  const llvm::StringRef packedI4ShiftAmount = "4";
  const llvm::StringRef packedI4ShiftAmountCType = "uint8_t";
  const llvm::StringRef packedI4ShiftLeftIntrinsic = "__riscv_vsll_vx_i8mf4";
  const llvm::StringRef packedI4ArithmeticShiftRightIntrinsic =
      "__riscv_vsra_vx_i8mf4";
  const llvm::StringRef packedI4ProductPairAddIntrinsic =
      "__riscv_vadd_vv_i16mf2";
  const llvm::StringRef accumulatorVectorCType =
      isProductReductionDequantization ? llvm::StringRef("vint32m1_t")
                                       : description.vectorCType;
  const llvm::StringRef finalResultVectorCType =
      isProductReductionDequantization ? description.resultVectorCType
                                       : description.vectorCType;
  const llvm::StringRef finalResultScalarCType = "float";
  if (route.getCallOpaqueSteps().size() != contract.expectedPreLoopStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires exact provider-built pre-loop statement count " +
        llvm::Twine(contract.expectedPreLoopStepCount) +
        " before artifact export");

  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps()[0];
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          preLoopSetVL, consumerLabel, "pre-loop setvl",
          runtimeContract.setVLIntrinsic,
          {{runtimeContract.runtimeAVLParameter.cName,
            runtimeContract.runtimeAVLParameter.cType}},
          runtimeContract.emitCFullChunkVLName, runtimeContract.vlCType))
    return error;

  const std::string expectedInitialAccumulatorLane =
      (llvm::StringRef(accumulatorABI->cName) + "[0]").str();
  if (isProductReductionDequantization) {
    if (!dequantScaleABI ||
        (isProductReductionDequantClamp &&
         (!lowerBoundABI || !upperBoundABI)))
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires dequant scale and optional lower/upper bound ABI before "
          "validating product-reduction dequantization carry/dequant statements");
    const std::size_t expectedLocalVariableCount =
        usesGroupedLowPrecisionProductReduction ? 2 : 1;
    if (route.getLocalVariables().size() != expectedLocalVariableCount)
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires exact provider-built local vector carry and grouped "
          "tail-start variable count " +
          llvm::Twine(expectedLocalVariableCount) +
          " before artifact export");
    const conversion::emitc::TCRVEmitCLocalVariable &carry =
        route.getLocalVariables()[0];
    if (!routeLocalVariableSourceIsSelectedRVVBody(carry) ||
        carry.name != "dot_acc_vec" ||
        carry.cType != accumulatorVectorCType ||
        carry.declarationInitializer != "__riscv_vmv_v_x_i32m1(0, 1)" ||
        !carry.initialValue.expression.empty() ||
        !carry.initialValue.cType.empty())
      return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-built local i32m1 vector carry to initialize "
        "from a safe declaration value before pre-loop acc[0] seeding");
    if (usesGroupedLowPrecisionProductReduction) {
      const conversion::emitc::TCRVEmitCLocalVariable &tailStart =
          route.getLocalVariables()[1];
      if (!routeLocalVariableSourceIsSelectedRVVBody(tailStart) ||
          tailStart.name != "grouped_tail_start" ||
          tailStart.cType != runtimeContract.vlCType ||
          tailStart.declarationInitializer != "0" ||
          !tailStart.initialValue.expression.empty() ||
          !tailStart.initialValue.cType.empty())
        return makeRVVTargetRouteError(
            llvm::Twine(consumerLabel) +
            " requires provider-built grouped tail-start local to initialize "
            "from zero before deriving the grouped main-loop bound");
    }
    const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSeed =
        route.getCallOpaqueSteps()[1];
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            preLoopSeed, consumerLabel, "pre-loop vector carry seed splat",
            description.scalarSeedSplatIntrinsic,
            {{expectedInitialAccumulatorLane, scalarI32CType},
             {description.reductionStoreVL, runtimeContract.vlCType}},
            "dot_acc_vec_seed", accumulatorVectorCType))
      return error;
    const std::size_t expectedPreLoopAssignmentCount =
        usesGroupedLowPrecisionProductReduction ? 2 : 1;
    if (route.getPreLoopAssignments().size() !=
        expectedPreLoopAssignmentCount)
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires exact provider-built pre-loop assignment count " +
          llvm::Twine(expectedPreLoopAssignmentCount) +
          " for vector carry and grouped tail-start facts before export");
    const conversion::emitc::TCRVEmitCAssignStep &seedAssign =
        route.getPreLoopAssignments()[0];
    if (!routeAssignSourceIsSelectedRVVBody(seedAssign) ||
        seedAssign.targetName != "dot_acc_vec" ||
        seedAssign.value.expression != "dot_acc_vec_seed" ||
        seedAssign.value.cType != accumulatorVectorCType)
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires pre-loop assignment to seed the local i32m1 vector carry "
          "from acc[0] before the runtime VL loop");
    if (usesGroupedLowPrecisionProductReduction) {
      const std::string expectedGroupedTailStartExpression =
          (llvm::Twine("((") + runtimeContract.runtimeAVLParameter.cName +
           " / (" + runtimeContract.emitCFullChunkVLName + " * 2)) * (" +
           runtimeContract.emitCFullChunkVLName + " * 2))")
              .str();
      const conversion::emitc::TCRVEmitCAssignStep &tailStartAssign =
          route.getPreLoopAssignments()[1];
      if (!routeAssignSourceIsSelectedRVVBody(tailStartAssign) ||
          tailStartAssign.targetName != "grouped_tail_start" ||
          tailStartAssign.value.expression != expectedGroupedTailStartExpression ||
          tailStartAssign.value.cType != runtimeContract.vlCType)
        return makeRVVTargetRouteError(
            llvm::Twine(consumerLabel) +
            " requires grouped tail-start pre-loop assignment to derive the "
            "largest tail-safe u2 main-loop bound from runtime AVL and "
            "full_chunk_vl");
    }
  } else {
    if (!route.getPreLoopAssignments().empty())
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " rejects pre-loop local carry assignments for non-dequant "
          "widening dot-reduction artifacts");
    if (!route.getLocalVariables().empty())
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " rejects local carry variables for non-dequant widening "
          "dot-reduction artifacts");
    const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSeed =
        route.getCallOpaqueSteps()[1];
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            preLoopSeed, consumerLabel, "pre-loop scalar seed splat",
            description.scalarSeedSplatIntrinsic,
            {{expectedInitialAccumulatorLane, scalarI32CType},
             {description.reductionStoreVL, runtimeContract.vlCType}},
            "dot_initial_acc_vec", accumulatorVectorCType))
      return error;
    const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopStore =
        route.getCallOpaqueSteps()[2];
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            preLoopStore, consumerLabel, "pre-loop initial output store",
            description.storeIntrinsic,
            {{outABI->cName, outABI->cType},
             {"dot_initial_acc_vec", description.vectorCType},
             {description.reductionStoreVL, runtimeContract.vlCType}}))
      return error;
  }

  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "widening dot-reduction target artifact consumer requires pre-loop "
          "statements to carry selected typed RVV source provenance");

  const std::size_t expectedLoopCount =
      usesGroupedLowPrecisionProductReduction ? 2 : 1;
  if (route.getForLoops().size() != expectedLoopCount)
    return makeRVVTargetRouteError(
        llvm::Twine("widening dot-reduction target artifact consumer requires "
                    "exactly ") +
        llvm::Twine(expectedLoopCount) +
        " provider-built runtime AVL/VL loop(s) before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  const std::string expectedMainLoopUpperBound =
      usesGroupedLowPrecisionProductReduction
          ? std::string("grouped_tail_start")
          : runtimeContract.runtimeAVLParameter.cName;
  const std::string expectedMainLoopUpperBoundCType =
      usesGroupedLowPrecisionProductReduction
          ? runtimeContract.vlCType
          : runtimeContract.runtimeAVLParameter.cType;
  const std::string expectedMainLoopStep =
      usesGroupedLowPrecisionProductReduction
          ? (llvm::Twine(runtimeContract.emitCFullChunkVLName) + " * 2").str()
          : runtimeContract.emitCFullChunkVLName;
  if (loop.inductionVarName != runtimeContract.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != runtimeContract.vlCType ||
      loop.upperBound.expression != expectedMainLoopUpperBound ||
      loop.upperBound.cType != expectedMainLoopUpperBoundCType ||
      loop.step.expression != expectedMainLoopStep ||
      loop.step.cType != runtimeContract.vlCType)
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-built loop bounds and step to mirror runtime AVL/VL route "
        "facts");

  const std::size_t expectedMainLoopBodyStepCount =
      usesGroupedLowPrecisionProductReduction
          ? contract.expectedLoopBodyStepCount + 5
          : contract.expectedLoopBodyStepCount;
  if (loop.bodySteps.size() != expectedMainLoopBodyStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires exact provider-built widening dot loop statement count " +
        llvm::Twine(expectedMainLoopBodyStepCount) +
        " before artifact export");

  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeContract.runtimeAVLParameter.cName) + " - " +
       runtimeContract.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[0], consumerLabel, "loop setvl",
          runtimeContract.setVLIntrinsic,
          {{expectedRemainingAVL, runtimeContract.vlCType}},
          runtimeContract.emitCLoopVLName, runtimeContract.vlCType))
    return error;

  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "widening dot-reduction target artifact consumer requires loop "
          "statements to carry selected typed RVV source provenance");

  auto validateUnitSourceLoadAt =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          const support::RuntimeABIParameter &abi,
          llvm::StringRef expectedPointer, llvm::StringRef vlName,
          llvm::StringRef resultName, llvm::StringRef stepLabel) -> llvm::Error {
    return validateRVVProviderBuiltRouteStep(
        step, consumerLabel, stepLabel, description.sourceVectorLoadIntrinsic,
        {{expectedPointer, abi.cType},
         {vlName, runtimeContract.vlCType}},
        resultName, description.sourceVectorCType);
  };
  auto validateUnitSourceLoad =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          const support::RuntimeABIParameter &abi, llvm::StringRef resultName,
          llvm::StringRef stepLabel) -> llvm::Error {
    const std::string expectedPointer =
        (llvm::StringRef(abi.cName) + " + " +
         runtimeContract.emitCLoopInductionName)
            .str();
    return validateUnitSourceLoadAt(step, abi, expectedPointer,
                                    runtimeContract.emitCLoopVLName,
                                    resultName, stepLabel);
  };

  auto validateStridedSourceLoad =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          const support::RuntimeABIParameter &abi,
          const support::RuntimeABIParameter &strideABI,
          llvm::StringRef resultName, llvm::StringRef stepLabel) -> llvm::Error {
    const std::string expectedPointer =
        (llvm::StringRef(abi.cName) + " + (" +
         runtimeContract.emitCLoopInductionName + " * " + strideABI.cName +
         ")")
            .str();
    const std::string expectedStrideBytes =
        (llvm::StringRef(strideABI.cName) + " * 2").str();
    return validateRVVProviderBuiltRouteStep(
        step, consumerLabel, stepLabel, description.stridedLoadIntrinsic,
        {{expectedPointer, abi.cType},
         {expectedStrideBytes, "ptrdiff_t"},
         {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
        resultName, description.sourceVectorCType);
  };

  auto validateDotSourceLoad =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          const support::RuntimeABIParameter &abi,
          const support::RuntimeABIParameter *strideABI,
          llvm::StringRef resultName, llvm::StringRef stepLabel) -> llvm::Error {
    if (isStrided) {
      if (!strideABI)
        return makeRVVTargetRouteError(
            llvm::Twine(consumerLabel) +
            " requires provider-derived stride ABI facts before validating " +
            stepLabel);
      return validateStridedSourceLoad(step, abi, *strideABI, resultName,
                                       stepLabel);
    }
    return validateUnitSourceLoad(step, abi, resultName, stepLabel);
  };

  auto validatePackedI4SignExtend =
      [&](std::size_t shiftLeftIndex, llvm::StringRef packedVecName,
          llvm::StringRef lowShiftedVecName, llvm::StringRef lowVecName,
          llvm::StringRef highVecName, llvm::StringRef shiftLeftLabel,
          llvm::StringRef lowShiftRightLabel,
          llvm::StringRef highShiftRightLabel) -> llvm::Error {
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[shiftLeftIndex], consumerLabel, shiftLeftLabel,
            packedI4ShiftLeftIntrinsic,
            {{packedVecName, description.sourceVectorCType},
             {packedI4ShiftAmount, packedI4ShiftAmountCType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            lowShiftedVecName, description.sourceVectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[shiftLeftIndex + 1], consumerLabel,
            lowShiftRightLabel, packedI4ArithmeticShiftRightIntrinsic,
            {{lowShiftedVecName, description.sourceVectorCType},
             {packedI4ShiftAmount, packedI4ShiftAmountCType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            lowVecName, description.sourceVectorCType))
      return error;
    return validateRVVProviderBuiltRouteStep(
        loop.bodySteps[shiftLeftIndex + 2], consumerLabel, highShiftRightLabel,
        packedI4ArithmeticShiftRightIntrinsic,
        {{packedVecName, description.sourceVectorCType},
         {packedI4ShiftAmount, packedI4ShiftAmountCType},
         {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
        highVecName, description.sourceVectorCType);
  };

  if (isComputedMask) {
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[1], consumerLabel, "compare lhs vector load",
            description.vectorLoadIntrinsic,
            {{(llvm::StringRef(cmpLHSABI->cName) + " + " +
               runtimeContract.emitCLoopInductionName)
                  .str(),
              cmpLHSABI->cType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            "cmp_lhs_vec", description.vectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[2], consumerLabel, "compare rhs vector load",
            description.vectorLoadIntrinsic,
            {{(llvm::StringRef(cmpRHSABI->cName) + " + " +
               runtimeContract.emitCLoopInductionName)
                  .str(),
              cmpRHSABI->cType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            "cmp_rhs_vec", description.vectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[3], consumerLabel, "compare predicate",
            description.compareIntrinsic,
            {{"cmp_lhs_vec", description.vectorCType},
             {"cmp_rhs_vec", description.vectorCType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            description.maskName, description.maskCType))
      return error;
    if (llvm::Error error =
            validateDotSourceLoad(loop.bodySteps[4], *dotLHSABI, lhsStrideABI,
                                  "dot_lhs_vec", "dot lhs source load"))
      return error;
    if (llvm::Error error =
            validateDotSourceLoad(loop.bodySteps[5], *dotRHSABI, rhsStrideABI,
                                  "dot_rhs_vec", "dot rhs source load"))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[6], consumerLabel, "inactive zero scalar splat",
            description.scalarSeedSplatIntrinsic,
            {{"0", scalarI32CType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            "dot_zero_vec", description.vectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[7], consumerLabel, "masked widening product",
            description.maskedWideningProductIntrinsic,
            {{description.maskName, description.maskCType},
             {"dot_lhs_vec", description.sourceVectorCType},
             {"dot_rhs_vec", description.sourceVectorCType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            "active_dot_product_vec", description.vectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[8], consumerLabel, "inactive-lane merge",
            description.maskedMergeIntrinsic,
            {{"dot_zero_vec", description.vectorCType},
             {"active_dot_product_vec", description.vectorCType},
             {description.maskName, description.maskCType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            "dot_product_vec", description.vectorCType))
      return error;
  } else {
    if (usesPackedI4LowPrecisionProductReduction) {
      if (llvm::Error error = validateUnitSourceLoad(
              loop.bodySteps[1], *lhsABI, "lhs_packed_i4_vec",
              "lhs packed-i4 source load"))
        return error;
      if (llvm::Error error = validateUnitSourceLoad(
              loop.bodySteps[2], *rhsABI, "rhs_packed_i4_vec",
              "rhs packed-i4 source load"))
        return error;
      if (llvm::Error error = validatePackedI4SignExtend(
              3, "lhs_packed_i4_vec", "lhs_low_i4_shifted_vec",
              "lhs_low_i4_vec", "lhs_high_i4_vec",
              "lhs packed-i4 low-nibble shift-left",
              "lhs packed-i4 low-nibble arithmetic shift-right",
              "lhs packed-i4 high-nibble arithmetic shift-right"))
        return error;
      if (llvm::Error error = validatePackedI4SignExtend(
              6, "rhs_packed_i4_vec", "rhs_low_i4_shifted_vec",
              "rhs_low_i4_vec", "rhs_high_i4_vec",
              "rhs packed-i4 low-nibble shift-left",
              "rhs packed-i4 low-nibble arithmetic shift-right",
              "rhs packed-i4 high-nibble arithmetic shift-right"))
        return error;
      if (llvm::Error error = validateRVVProviderBuiltRouteStep(
              loop.bodySteps[9], consumerLabel,
              "packed-i4 low-nibble widening product",
              description.wideningProductIntrinsic,
              {{"lhs_low_i4_vec", description.sourceVectorCType},
               {"rhs_low_i4_vec", description.sourceVectorCType},
               {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
              "product_vec", description.productVectorCType))
        return error;
      if (llvm::Error error = validateRVVProviderBuiltRouteStep(
              loop.bodySteps[10], consumerLabel,
              "packed-i4 high-nibble widening product",
              description.wideningProductIntrinsic,
              {{"lhs_high_i4_vec", description.sourceVectorCType},
               {"rhs_high_i4_vec", description.sourceVectorCType},
               {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
              "product_vec_i4_high", description.productVectorCType))
        return error;
      if (llvm::Error error = validateRVVProviderBuiltRouteStep(
              loop.bodySteps[11], consumerLabel,
              "packed-i4 low/high product-pair sum",
              packedI4ProductPairAddIntrinsic,
              {{"product_vec", description.productVectorCType},
               {"product_vec_i4_high", description.productVectorCType},
               {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
              "product_vec_i4_pair_sum", description.productVectorCType))
        return error;
    } else {
      if (llvm::Error error =
              validateDotSourceLoad(loop.bodySteps[1], *lhsABI, lhsStrideABI,
                                    "lhs_vec", "lhs source load"))
        return error;
      if (llvm::Error error =
              validateDotSourceLoad(loop.bodySteps[2], *rhsABI, rhsStrideABI,
                                    "rhs_vec", "rhs source load"))
        return error;
      const llvm::StringRef productResultName =
          isProductReductionChain ? "product_vec" : "dot_product_vec";
      const llvm::StringRef productResultCType =
          isProductReductionChain ? description.productVectorCType
                                  : description.vectorCType;
      if (llvm::Error error = validateRVVProviderBuiltRouteStep(
              loop.bodySteps[3], consumerLabel, "widening product",
              description.wideningProductIntrinsic,
              {{"lhs_vec", description.sourceVectorCType},
               {"rhs_vec", description.sourceVectorCType},
               {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
              productResultName, productResultCType))
        return error;
    }
  }

  const std::size_t seedIndex =
      isComputedMask ? 9 : usesPackedI4LowPrecisionProductReduction ? 12 : 4;
  const std::size_t reductionIndex =
      isProductReductionDequantization ? seedIndex : seedIndex + 1;
  const std::size_t storeIndex = reductionIndex + 1;
  const llvm::StringRef reductionInputName =
      usesPackedI4LowPrecisionProductReduction
          ? llvm::StringRef("product_vec_i4_pair_sum")
          : isProductReductionChain ? llvm::StringRef("product_vec")
                                    : llvm::StringRef("dot_product_vec");
  const llvm::StringRef reductionInputCType =
      isProductReductionChain ? description.productVectorCType
                              : description.vectorCType;
  const llvm::StringRef reductionResultName =
      isProductReductionChain ? llvm::StringRef("reduced_i32_vec")
                              : description.resultName;
  if (!isProductReductionDequantization) {
    const std::string expectedLoopSeed =
        (llvm::StringRef(outABI->cName) + "[0]").str();
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[seedIndex], consumerLabel, "loop scalar seed splat",
            description.scalarSeedSplatIntrinsic,
            {{expectedLoopSeed, scalarI32CType},
             {description.reductionStoreVL, runtimeContract.vlCType}},
            "dot_acc_vec", accumulatorVectorCType))
      return error;
  }
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[reductionIndex], consumerLabel,
          "widening dot reduction", description.intrinsic,
          {{reductionInputName, reductionInputCType},
           {"dot_acc_vec", accumulatorVectorCType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
          reductionResultName, accumulatorVectorCType))
    return error;
  if (isProductReductionDequantization) {
    if (!dequantScaleABI)
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires dequant scale ABI before validating post-loop dequant "
          "statements");
    const std::size_t expectedLoopAssignmentCount =
        usesGroupedLowPrecisionProductReduction ? 2 : 1;
    if (loop.bodyAssignments.size() != expectedLoopAssignmentCount)
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires exact provider-built loop assignment count " +
          llvm::Twine(expectedLoopAssignmentCount) +
          " from reduced vectors to the local vector carry before artifact "
          "export");
    auto validateCarryAssignment =
        [&](const conversion::emitc::TCRVEmitCAssignStep &assign,
            llvm::StringRef resultName, llvm::StringRef label) -> llvm::Error {
      if (!routeAssignSourceIsSelectedRVVBody(assign) ||
          assign.targetName != "dot_acc_vec" ||
          assign.value.expression != resultName ||
          assign.value.cType != accumulatorVectorCType)
        return makeRVVTargetRouteError(
            llvm::Twine(consumerLabel) + " requires " + label +
            " assignment to carry the reduced i32m1 vector across runtime VL "
            "chunks before artifact export");
      return llvm::Error::success();
    };
    const llvm::StringRef finalCarryResultName =
        usesPackedI4LowPrecisionProductReduction
            ? llvm::StringRef("reduced_i32_vec")
            : reductionResultName;
    if (llvm::Error error = validateCarryAssignment(
            loop.bodyAssignments[0], finalCarryResultName,
            "first product-reduction"))
      return error;
    if (usesGroupedLowPrecisionProductReduction) {
      const std::string expectedGroupedSecondRemainingAVL =
          (llvm::Twine(runtimeContract.runtimeAVLParameter.cName) + " - " +
           runtimeContract.emitCLoopInductionName + " - " +
           runtimeContract.emitCLoopVLName)
              .str();
      if (llvm::Error error = validateRVVProviderBuiltRouteStep(
              loop.bodySteps[5], consumerLabel, "grouped second loop setvl",
              runtimeContract.setVLIntrinsic,
              {{expectedGroupedSecondRemainingAVL, runtimeContract.vlCType}},
              "grouped_loop_vl_u1", runtimeContract.vlCType))
        return error;
      const std::string expectedGroupedLHSPointer =
          (llvm::Twine(lhsABI->cName) + " + " +
           runtimeContract.emitCLoopInductionName + " + " +
           runtimeContract.emitCLoopVLName)
              .str();
      const std::string expectedGroupedRHSPointer =
          (llvm::Twine(rhsABI->cName) + " + " +
           runtimeContract.emitCLoopInductionName + " + " +
           runtimeContract.emitCLoopVLName)
              .str();
      if (llvm::Error error = validateUnitSourceLoadAt(
              loop.bodySteps[6], *lhsABI, expectedGroupedLHSPointer,
              "grouped_loop_vl_u1", "lhs_vec_u1",
              "grouped second lhs source load"))
        return error;
      if (llvm::Error error = validateUnitSourceLoadAt(
              loop.bodySteps[7], *rhsABI, expectedGroupedRHSPointer,
              "grouped_loop_vl_u1", "rhs_vec_u1",
              "grouped second rhs source load"))
        return error;
      if (llvm::Error error = validateRVVProviderBuiltRouteStep(
              loop.bodySteps[8], consumerLabel,
              "grouped second widening product",
              description.wideningProductIntrinsic,
              {{"lhs_vec_u1", description.sourceVectorCType},
               {"rhs_vec_u1", description.sourceVectorCType},
               {"grouped_loop_vl_u1", runtimeContract.vlCType}},
              "product_vec_u1", description.productVectorCType))
        return error;
      if (llvm::Error error = validateRVVProviderBuiltRouteStep(
              loop.bodySteps[9], consumerLabel,
              "grouped second widening dot reduction", description.intrinsic,
              {{"product_vec_u1", description.productVectorCType},
               {"reduced_i32_vec", accumulatorVectorCType},
               {"grouped_loop_vl_u1", runtimeContract.vlCType}},
              "reduced_i32_vec_u1", accumulatorVectorCType))
        return error;
      if (llvm::Error error = validateCarryAssignment(
              loop.bodyAssignments[1], "reduced_i32_vec_u1",
              "grouped second product-reduction"))
        return error;
      const conversion::emitc::TCRVEmitCForLoop &tailLoop =
          route.getForLoops()[1];
      if (tailLoop.inductionVarName !=
              runtimeContract.emitCLoopInductionName ||
          tailLoop.lowerBound.expression != "grouped_tail_start" ||
          tailLoop.lowerBound.cType != runtimeContract.vlCType ||
          tailLoop.upperBound.expression !=
              runtimeContract.runtimeAVLParameter.cName ||
          tailLoop.upperBound.cType !=
              runtimeContract.runtimeAVLParameter.cType ||
          tailLoop.step.expression !=
              runtimeContract.emitCFullChunkVLName ||
          tailLoop.step.cType != runtimeContract.vlCType)
        return makeRVVTargetRouteError(
            llvm::Twine(consumerLabel) +
            " requires grouped u2 tail loop bounds to cover "
            "grouped_tail_start..runtime AVL with single-chunk VL steps");
      if (tailLoop.bodySteps.size() != contract.expectedLoopBodyStepCount)
        return makeRVVTargetRouteError(
            llvm::Twine(consumerLabel) +
            " requires exact provider-built grouped tail loop statement "
            "count " +
            llvm::Twine(contract.expectedLoopBodyStepCount) +
            " before artifact export");
      for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
           tailLoop.bodySteps)
        if (!routeStepSourceIsSelectedRVVBody(step))
          return makeRVVTargetRouteError(
              "widening dot-reduction target artifact consumer requires "
              "grouped tail loop statements to carry selected typed RVV "
              "source provenance");
      if (llvm::Error error = validateRVVProviderBuiltRouteStep(
              tailLoop.bodySteps[0], consumerLabel,
              "grouped tail loop setvl", runtimeContract.setVLIntrinsic,
              {{expectedRemainingAVL, runtimeContract.vlCType}},
              runtimeContract.emitCLoopVLName, runtimeContract.vlCType))
        return error;
      if (llvm::Error error =
              validateDotSourceLoad(tailLoop.bodySteps[1], *lhsABI,
                                    lhsStrideABI, "lhs_vec",
                                    "grouped tail lhs source load"))
        return error;
      if (llvm::Error error =
              validateDotSourceLoad(tailLoop.bodySteps[2], *rhsABI,
                                    rhsStrideABI, "rhs_vec",
                                    "grouped tail rhs source load"))
        return error;
      if (llvm::Error error = validateRVVProviderBuiltRouteStep(
              tailLoop.bodySteps[3], consumerLabel,
              "grouped tail widening product",
              description.wideningProductIntrinsic,
              {{"lhs_vec", description.sourceVectorCType},
               {"rhs_vec", description.sourceVectorCType},
               {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
              "product_vec", description.productVectorCType))
        return error;
      if (llvm::Error error = validateRVVProviderBuiltRouteStep(
              tailLoop.bodySteps[4], consumerLabel,
              "grouped tail widening dot reduction", description.intrinsic,
              {{"product_vec", description.productVectorCType},
               {"dot_acc_vec", accumulatorVectorCType},
               {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
              "reduced_i32_vec", accumulatorVectorCType))
        return error;
      if (tailLoop.bodyAssignments.size() != 1)
        return makeRVVTargetRouteError(
            llvm::Twine(consumerLabel) +
            " requires exactly one provider-built grouped tail assignment "
            "from the reduced vector to the local vector carry");
      if (llvm::Error error = validateCarryAssignment(
              tailLoop.bodyAssignments[0], "reduced_i32_vec",
              "grouped tail product-reduction"))
        return error;
    }
    const std::size_t expectedPostLoopStepCount =
        isProductReductionDequantClamp ? 9 : 3;
    if (route.getPostLoopSteps().size() != expectedPostLoopStepCount)
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires exactly " + llvm::Twine(expectedPostLoopStepCount) +
          " provider-built post-loop dequant/clamp/store statements before "
          "artifact export");
    for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
         route.getPostLoopSteps())
      if (!routeStepSourceIsSelectedRVVBody(step))
        return makeRVVTargetRouteError(
            "widening dot-reduction target artifact consumer requires "
            "post-loop statements to carry selected typed RVV source "
            "provenance");
    const llvm::StringRef scaledResultName =
        isProductReductionDequantClamp ? llvm::StringRef("dequantized_vec")
                                       : description.resultName;
    const std::string expectedScalarDequantExpression =
        (llvm::Twine("dot_acc_scalar * ") + dequantScaleABI->cName).str();
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            route.getPostLoopSteps()[0], consumerLabel,
            "post-loop vector carry scalar extract",
            "__riscv_vmv_x_s_i32m1_i32",
            {{"dot_acc_vec", accumulatorVectorCType}},
            "dot_acc_scalar", scalarI32CType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            route.getPostLoopSteps()[1], consumerLabel,
            "post-loop scalar dequant splat", description.rhsBroadcastIntrinsic,
            {{expectedScalarDequantExpression, finalResultScalarCType},
             {description.reductionStoreVL, runtimeContract.vlCType}},
            scaledResultName, finalResultVectorCType))
      return error;
    if (isProductReductionDequantClamp) {
      if (llvm::Error error = validateRVVProviderBuiltRouteStep(
              route.getPostLoopSteps()[2], consumerLabel,
              "post-loop lower bound splat", description.rhsBroadcastIntrinsic,
              {{lowerBoundABI->cName, lowerBoundABI->cType},
               {description.reductionStoreVL, runtimeContract.vlCType}},
              "lower_bound_vec", finalResultVectorCType))
        return error;
      if (llvm::Error error = validateRVVProviderBuiltRouteStep(
              route.getPostLoopSteps()[3], consumerLabel,
              "post-loop lower clamp compare", description.compareIntrinsic,
              {{scaledResultName, finalResultVectorCType},
               {"lower_bound_vec", finalResultVectorCType},
               {description.reductionStoreVL, runtimeContract.vlCType}},
              "lower_clamp_mask", description.maskCType))
        return error;
      if (llvm::Error error = validateRVVProviderBuiltRouteStep(
              route.getPostLoopSteps()[4], consumerLabel,
              "post-loop lower clamp select", description.maskedMergeIntrinsic,
              {{scaledResultName, finalResultVectorCType},
               {"lower_bound_vec", finalResultVectorCType},
               {"lower_clamp_mask", description.maskCType},
               {description.reductionStoreVL, runtimeContract.vlCType}},
              "lower_clamped_vec", finalResultVectorCType))
        return error;
      if (llvm::Error error = validateRVVProviderBuiltRouteStep(
              route.getPostLoopSteps()[5], consumerLabel,
              "post-loop upper bound splat", description.rhsBroadcastIntrinsic,
              {{upperBoundABI->cName, upperBoundABI->cType},
               {description.reductionStoreVL, runtimeContract.vlCType}},
              "upper_bound_vec", finalResultVectorCType))
        return error;
      if (llvm::Error error = validateRVVProviderBuiltRouteStep(
              route.getPostLoopSteps()[6], consumerLabel,
              "post-loop upper clamp compare",
              description.secondaryCompareIntrinsic,
              {{"upper_bound_vec", finalResultVectorCType},
               {"lower_clamped_vec", finalResultVectorCType},
               {description.reductionStoreVL, runtimeContract.vlCType}},
              "upper_clamp_mask", description.maskCType))
        return error;
      if (llvm::Error error = validateRVVProviderBuiltRouteStep(
              route.getPostLoopSteps()[7], consumerLabel,
              "post-loop upper clamp select", description.maskedMergeIntrinsic,
              {{"lower_clamped_vec", finalResultVectorCType},
               {"upper_bound_vec", finalResultVectorCType},
               {"upper_clamp_mask", description.maskCType},
               {description.reductionStoreVL, runtimeContract.vlCType}},
              description.resultName, finalResultVectorCType))
        return error;
      return validateRVVProviderBuiltRouteStep(
          route.getPostLoopSteps()[8], consumerLabel,
          "post-loop output f32 store", description.storeIntrinsic,
          {{outABI->cName, outABI->cType},
           {description.resultName, finalResultVectorCType},
           {description.reductionStoreVL, runtimeContract.vlCType}});
    }
    return validateRVVProviderBuiltRouteStep(
        route.getPostLoopSteps()[2], consumerLabel,
        "post-loop output f32 store", description.storeIntrinsic,
        {{outABI->cName, outABI->cType},
         {description.resultName, finalResultVectorCType},
         {description.reductionStoreVL, runtimeContract.vlCType}});
  } else if (llvm::Error error = validateRVVProviderBuiltRouteStep(
                 loop.bodySteps[storeIndex], consumerLabel, "output store",
                 description.storeIntrinsic,
                 {{outABI->cName, outABI->cType},
                  {isProductReductionChain ? reductionResultName
                                           : description.resultName,
                   accumulatorVectorCType},
                  {description.reductionStoreVL, runtimeContract.vlCType}}))
    return error;

  if (!loop.bodyAssignments.empty() || !route.getPostLoopSteps().empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " rejects local carry assignments or post-loop statements for "
        "non-dequant widening dot-reduction artifacts");

  return llvm::Error::success();
}

llvm::Error validateRVVWideningDotReductionRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  const std::optional<
      plugin::rvv::RVVWideningDotReduceRouteValidationContract>
      contract =
          getRVVWideningDotReductionTargetRouteValidationContract(description);
  if (!contract)
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-owned route validation contract before artifact export");

  if (route.getRouteID() != contract->core.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract->consumerLabel) +
        " requires rebuilt provider route token '" +
        contract->core.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");

  if (isRVVNonComputedMaskWideningDotReductionRouteFamilyOperation(
          description.operation))
    if (llvm::Error error =
            validateRVVNonComputedMaskWideningDotReductionRoutePayloadFacts(
                description))
      return error;
  if (isRVVComputedMaskWideningDotReductionRouteFamilyOperation(
          description.operation))
    if (llvm::Error error =
            validateRVVComputedMaskWideningDotReductionRoutePayloadFacts(
                description))
      return error;

  if (llvm::Error error =
          validateRVVWideningDotReductionRouteHeaders(route, *contract))
    return error;
  if (llvm::Error error =
          validateRVVWideningDotReductionRouteTypeMappings(route, *contract))
    return error;
  if (llvm::Error error =
          validateRVVWideningDotReductionRouteABIMappings(route, *contract))
    return error;
  return validateRVVWideningDotReductionRouteStatementPlan(route, *contract);
}

llvm::Error validateRVVWideningDotReductionTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVWideningDotReductionRoutePayloadFacts(
      context.route, context.description);
}

llvm::Error requireEmptyWideningDotReductionStaleMirror(
    const TargetArtifactCandidate &candidate, llvm::StringRef key,
    llvm::StringRef label) {
  return requireCandidateMetadataMirror(candidate, key, "", label);
}

llvm::Error validateRVVLowPrecisionResourceCandidateMirrors(
    const TargetArtifactCandidate &candidate,
    const plugin::rvv::RVVLowPrecisionContractionResourceSelection
        &selection) {
  if (!selection.hasSelection)
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-selected low-precision direct-contraction resource facts "
        "before validating resource candidate mirrors");
  auto requireResourceMirror =
      [&](llvm::StringRef key, llvm::StringRef expected,
          llvm::StringRef label) -> llvm::Error {
    std::string fullLabel =
        (llvm::Twine("provider-selected low-precision direct-contraction "
                     "resource ")
             + label)
            .str();
    return requireCandidateMetadataMirror(
        candidate, key, expected, fullLabel);
  };
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.candidate_set",
          selection.candidateSetID, "candidate set"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.selected_candidate",
          selection.selectedCandidateID, "selected candidate"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.selection_reason",
          selection.selectionReason, "selection reason"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.legality_scope",
          selection.legalityScope, "legality scope"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.source_dtype",
          selection.sourceElementTypeName, "source dtype"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.source_sew",
          llvm::Twine(selection.sourceSEW).str(), "source SEW"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.source_lmul",
          selection.sourceLMUL, "source LMUL"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.operand_form",
          selection.operandForm, "operand form"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.source_signedness",
          selection.sourceSignedness, "source signedness"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.storage_element_width",
          llvm::Twine(selection.storageElementWidth).str(),
          "storage element width"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.effective_element_width",
          llvm::Twine(selection.effectiveElementWidth).str(),
          "effective element width"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.packing_layout",
          selection.packingLayout, "packing layout"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.unpack_intent",
          selection.unpackIntent, "unpack intent"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.product_dtype",
          selection.productElementTypeName, "product dtype"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.product_sew",
          llvm::Twine(selection.productSEW).str(), "product SEW"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.product_lmul",
          selection.productLMUL, "product LMUL"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.product_emul",
          selection.productEMUL, "product EMUL"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.accumulator_dtype",
          selection.accumulatorElementTypeName, "accumulator dtype"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.accumulator_sew",
          llvm::Twine(selection.accumulatorSEW).str(), "accumulator SEW"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.accumulator_lmul",
          selection.accumulatorLMUL, "accumulator LMUL"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.accumulator_emul",
          selection.accumulatorEMUL, "accumulator EMUL"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.result_dtype",
          selection.resultElementTypeName, "result dtype"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.result_sew",
          llvm::Twine(selection.resultSEW).str(), "result SEW"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.result_lmul",
          selection.resultLMUL, "result LMUL"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.memory_form",
          selection.memoryForm, "memory form"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.tail_policy",
          selection.tailPolicy, "tail policy"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.mask_policy",
          selection.maskPolicy, "mask policy"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.unroll_factor",
          llvm::Twine(selection.unrollFactor).str(), "unroll factor"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.accumulator_count",
          llvm::Twine(selection.accumulatorCount).str(),
          "accumulator count"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.reduction_layout",
          selection.reductionLayout, "reduction layout"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.vsetvl_region_count",
          llvm::Twine(selection.vsetvlRegionCount).str(),
          "vsetvl region count"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.peak_live_vector_groups",
          llvm::Twine(selection.peakLiveVectorGroups).str(),
          "peak live vector-group estimate"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.vector_register_budget",
          llvm::Twine(selection.vectorRegisterBudget).str(),
          "vector register budget"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.runtime_avl_source",
          selection.runtimeAVLSource, "runtime AVL source"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.gearbox.producer_scope", selection.producerScope,
          "Gearbox producer scope"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.gearbox.consumer_scope", selection.consumerScope,
          "Gearbox consumer scope"))
    return error;
  if (selection.producerScope == selection.consumerScope)
    return makeRVVTargetRouteError(
        "target artifact candidate validation requires distinct Gearbox "
        "producer and consumer scopes");
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.runtime_abi_order",
          selection.runtimeABIOrder, "runtime ABI order"))
    return error;
  if (!selection.primitiveContractID.empty()) {
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.primitive_contract",
            selection.primitiveContractID, "primitive contract"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.primitive_kind",
            selection.primitiveKind, "primitive kind"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.primitive_chain_contract",
            selection.primitiveChainContractID, "primitive chain contract"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.primitive_chain_kind",
            selection.primitiveChainKind, "primitive chain kind"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource."
            "primitive_widening_product_relation",
            selection.primitiveWideningProductRelation,
            "primitive widening product relation"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource."
            "primitive_product_reduction_chain_relation",
            selection.primitiveProductReductionChainRelation,
            "primitive product-reduction chain relation"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource."
            "primitive_widening_product_intrinsic",
            selection.primitiveWideningProductIntrinsic,
            "primitive widening product intrinsic"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.primitive_reduction_intrinsic",
            selection.primitiveReductionIntrinsic,
            "primitive reduction intrinsic"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource."
            "primitive_scalar_seed_splat_intrinsic",
            selection.primitiveScalarSeedSplatIntrinsic,
            "primitive scalar seed splat intrinsic"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.primitive_accumulator_layout",
            selection.primitiveAccumulatorLayout,
            "primitive accumulator layout"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.primitive_result_layout",
            selection.primitiveResultLayout, "primitive result layout"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.primitive_reduction_store_vl",
            selection.primitiveReductionStoreVL,
            "primitive reduction store VL"))
      return error;
  }
  if (!selection.realizationDecision.empty()) {
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.realization_producer",
            selection.realizationProducer, "realization producer"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.realization_decision",
            selection.realizationDecision, "realization decision"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.realized_unroll_factor",
            llvm::Twine(selection.realizedUnrollFactor).str(),
            "realized unroll factor"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.realized_vsetvl_region_count",
            llvm::Twine(selection.realizedVSetVLRegionCount).str(),
            "realized vsetvl region count"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.realized_peak_live_vector_groups",
            llvm::Twine(selection.realizedPeakLiveVectorGroups).str(),
            "realized peak live vector-group estimate"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.product_region_index",
            llvm::Twine(selection.productRegionIndex).str(),
            "product region index"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.dequant_region_index",
            llvm::Twine(selection.dequantRegionIndex).str(),
            "dequant region index"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.product_phase",
            selection.productPhase, "product phase"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.dequant_phase",
            selection.dequantPhase, "dequant phase"))
      return error;
  }
  if (!selection.performanceFeedback.empty()) {
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.performance_feedback",
            selection.performanceFeedback, "performance feedback"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.performance_baseline",
            selection.performanceBaseline, "performance baseline"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.performance_best_speedup_range",
            selection.performanceBestSpeedupRange,
            "performance best-speedup range"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.performance_action",
            selection.performanceAction, "performance action"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.performance_maturity",
            selection.performanceMaturity, "performance maturity"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.performance_maturity_evidence",
            selection.performanceMaturityEvidence,
            "performance maturity evidence"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.performance_maturity_outcome",
            selection.performanceMaturityOutcome, "performance maturity outcome"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.performance_selection_eligible",
            selection.performanceSelectionEligible,
            "performance selection eligibility"))
      return error;
    if (llvm::Error error = requireResourceMirror(
            "tcrv_rvv.low_precision_resource.dispatch_preference",
            selection.dispatchPreference, "dispatch preference"))
      return error;
  }
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.target_capability_provider_mirror",
          selection.targetCapabilityProviderMirror,
          "target capability provider mirror"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.target_capability_legality_mirror",
          selection.targetCapabilityLegalityMirror,
          "target capability legality mirror"))
    return error;
  if (llvm::Error error = requireResourceMirror(
          "tcrv_rvv.low_precision_resource.legality",
          selection.isLegal ? "legal" : "rejected", "legality decision"))
    return error;
  return requireResourceMirror(
      "tcrv_rvv.low_precision_resource.rejection_reason",
      selection.rejectionReason, "rejection reason");
}

llvm::Error validateRVVLowPrecisionWideningReductionPrimitiveCandidateMirrors(
    const TargetArtifactCandidate &candidate,
    const plugin::rvv::RVVWideningDotReduceRouteValidationContract &contract) {
  const plugin::rvv::RVVLowPrecisionWideningReductionPrimitiveFacts
      &primitive = contract.lowPrecisionWideningReductionPrimitiveFacts;
  if (!primitive.hasFacts)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned low-precision widening-reduction primitive "
        "facts before validating artifact mirrors");

  auto requirePrimitiveMirror =
      [&](llvm::StringRef key, llvm::StringRef expected,
          llvm::StringRef label) -> llvm::Error {
    std::string fullLabel =
        (llvm::Twine("selected typed RVV product-reduction low-precision "
                     "widening-reduction primitive ")
             + label)
            .str();
    return requireCandidateMetadataMirror(candidate, key, expected, fullLabel);
  };

  if (llvm::Error error = requirePrimitiveMirror(
          "tcrv_rvv.low_precision_primitive.contract",
          primitive.lowPrecisionPrimitiveContractID, "contract"))
    return error;
  if (llvm::Error error = requirePrimitiveMirror(
          "tcrv_rvv.low_precision_primitive.kind",
          primitive.lowPrecisionPrimitiveKind, "kind"))
    return error;
  if (llvm::Error error = requirePrimitiveMirror(
          "tcrv_rvv.low_precision_primitive.source_dtype",
          primitive.sourceElementTypeName, "source dtype"))
    return error;
  if (llvm::Error error = requirePrimitiveMirror(
          "tcrv_rvv.low_precision_primitive.product_dtype",
          primitive.productElementTypeName, "product dtype"))
    return error;
  if (llvm::Error error = requirePrimitiveMirror(
          "tcrv_rvv.low_precision_primitive.accumulator_dtype",
          primitive.accumulatorElementTypeName, "accumulator dtype"))
    return error;
  if (llvm::Error error = requirePrimitiveMirror(
          "tcrv_rvv.low_precision_primitive.result_dtype",
          primitive.finalResultElementTypeName, "final result dtype"))
    return error;
  if (llvm::Error error = requirePrimitiveMirror(
          "tcrv_rvv.source_sew", llvm::Twine(primitive.sourceSEW).str(),
          "source SEW"))
    return error;
  if (llvm::Error error = requirePrimitiveMirror(
          "tcrv_rvv.source_lmul", primitive.sourceLMUL, "source LMUL"))
    return error;
  if (llvm::Error error = requirePrimitiveMirror(
          "tcrv_rvv.product_sew", llvm::Twine(primitive.productSEW).str(),
          "product SEW"))
    return error;
  if (llvm::Error error = requirePrimitiveMirror(
          "tcrv_rvv.product_lmul", primitive.productLMUL, "product LMUL"))
    return error;
  if (llvm::Error error = requirePrimitiveMirror(
          "tcrv_rvv.accumulator_sew",
          llvm::Twine(primitive.accumulatorSEW).str(), "accumulator SEW"))
    return error;
  if (llvm::Error error = requirePrimitiveMirror(
          "tcrv_rvv.accumulator_lmul", primitive.accumulatorLMUL,
          "accumulator LMUL"))
    return error;
  if (llvm::Error error = requirePrimitiveMirror(
          "tcrv_rvv.result_sew",
          llvm::Twine(primitive.reductionResultSEW).str(), "result SEW"))
    return error;
  if (llvm::Error error = requirePrimitiveMirror(
          "tcrv_rvv.result_lmul", primitive.reductionResultLMUL,
          "result LMUL"))
    return error;
  if (llvm::Error error = requirePrimitiveMirror(
          "tcrv_rvv.widening_product_relation",
          primitive.wideningProductRelation, "widening product relation"))
    return error;
  if (llvm::Error error = requirePrimitiveMirror(
          "tcrv_rvv.product_reduction_chain_relation",
          primitive.productReductionChainRelation,
          "product-reduction relation"))
    return error;
  if (llvm::Error error = requirePrimitiveMirror(
          "tcrv_rvv.widening_product_intrinsic",
          primitive.wideningProductIntrinsic, "widening product intrinsic"))
    return error;
  if (llvm::Error error = requirePrimitiveMirror(
          "tcrv_rvv.widening_reduction_intrinsic",
          primitive.reductionIntrinsic, "widening reduction intrinsic"))
    return error;
  if (llvm::Error error = requirePrimitiveMirror(
          "tcrv_rvv.scalar_seed_splat_intrinsic",
          primitive.scalarSeedSplatIntrinsic, "scalar seed splat intrinsic"))
    return error;
  if (llvm::Error error = requirePrimitiveMirror(
          "tcrv_rvv.reduction_accumulator_layout",
          primitive.accumulatorLayout, "accumulator layout"))
    return error;
  if (llvm::Error error = requirePrimitiveMirror(
          "tcrv_rvv.reduction_result_layout", primitive.resultLayout,
          "result layout"))
    return error;
  return requirePrimitiveMirror("tcrv_rvv.reduction_store_vl",
                                primitive.reductionStoreVL, "store VL");
}

llvm::Error validateRVVWideningDotReductionTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description =
      context.description;
  const std::optional<
      plugin::rvv::RVVWideningDotReduceRouteValidationContract>
      contract =
          getRVVWideningDotReductionTargetRouteValidationContract(description);
  if (!contract)
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-owned route validation contract before validating candidate "
        "mirrors");
  const std::string sourceSEW = llvm::Twine(contract->sourceSEW).str();
  const bool isProductReductionDequantClamp =
      contract->kind == plugin::rvv::RVVWideningDotReduceRouteValidationKind::
                            ProductReductionDequantClampF32;
  const bool isProductReductionDequantization =
      contract->kind == plugin::rvv::RVVWideningDotReduceRouteValidationKind::
                            ProductReductionDequantization ||
      isProductReductionDequantClamp;
  const bool isProductReductionChain =
      contract->kind == plugin::rvv::RVVWideningDotReduceRouteValidationKind::
                            ProductReductionChain ||
      isProductReductionDequantization;
  const std::string productSEW = llvm::Twine(contract->productSEW).str();
  const std::string accumulatorSEW =
      llvm::Twine(contract->accumulatorSEW).str();
  const std::string resultSEW = llvm::Twine(contract->resultSEW).str();

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_plan",
          contract->routeOperandBindingPlanID,
          "selected typed RVV widening dot binding plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_operands",
          contract->routeOperandBindingSummary,
          "selected typed RVV widening dot binding summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.provider_supported_mirror",
          contract->providerSupportedMirror,
          "selected typed RVV widening dot provider support"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.contraction_route_family_plan",
          contract->contractionRouteFamilyPlanID,
          "selected typed RVV widening dot contraction route-family plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.memory_form",
          plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
              contract->memoryForm),
          "selected typed RVV widening dot memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_control_plan",
          contract->runtimeControlPlanID,
          "route-local runtime AVL/VL control plan mirror"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order", contract->runtimeABIOrder,
          "route-local runtime AVL/VL ABI order mirror"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.required_header_declarations",
          contract->requiredHeaderDeclarations,
          "selected typed RVV widening dot route header requirements"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.c_type_mapping",
          contract->cTypeMappingSummary,
          "selected typed RVV widening dot route type mapping summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.target_leaf_profile",
          contract->targetLeafProfile,
          "selected typed RVV widening dot target leaf profile"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_sew", sourceSEW,
          "selected typed RVV widening dot i16 source SEW"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_lmul", contract->sourceLMUL,
          "selected typed RVV widening dot source LMUL"))
    return error;
  if (isProductReductionChain) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.product_sew", productSEW,
            "selected typed RVV product-reduction intermediate SEW"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.product_lmul", contract->productLMUL,
            "selected typed RVV product-reduction intermediate LMUL"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.product_vector_type",
            contract->productVectorTypeName,
            "selected typed RVV product-reduction intermediate vector type"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.product_vector_c_type",
            contract->productVectorCType,
            "selected typed RVV product-reduction intermediate vector C type"))
      return error;
  }
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.accumulator_sew", accumulatorSEW,
          "selected typed RVV widening dot accumulator SEW"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.accumulator_lmul",
          contract->accumulatorLMUL,
          "selected typed RVV widening dot accumulator LMUL"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.result_sew", resultSEW,
          "selected typed RVV widening dot result SEW"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.result_lmul", contract->resultLMUL,
          "selected typed RVV widening dot result LMUL"))
    return error;
  if (contract->lowPrecisionResourceSelection.hasSelection)
    if (llvm::Error error = validateRVVLowPrecisionResourceCandidateMirrors(
            candidate, contract->lowPrecisionResourceSelection))
      return error;
  if (isProductReductionChain) {
    if (llvm::Error error =
            validateRVVLowPrecisionWideningReductionPrimitiveCandidateMirrors(
                candidate, *contract))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.reduction_accumulator_layout",
            contract->wideningDotProductAccumulatorLayout,
            "selected typed RVV product-reduction accumulator layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.reduction_result_layout",
            contract->wideningDotProductResultLayout,
            "selected typed RVV product-reduction result layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.product_reduction_chain_relation",
            contract->productReductionChainRelation,
            "selected typed RVV product-reduction chain relation"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.widening_dot_accumulator_layout", "",
            "selected typed RVV product-reduction route without dot layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.widening_dot_result_layout", "",
            "selected typed RVV product-reduction route without dot layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.widening_dot_relation", "",
            "selected typed RVV product-reduction route without dot relation"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate,
            "tcrv_rvv.widening_dot_source_accumulator_result_contract", "",
            "selected typed RVV product-reduction route without widening dot "
            "source/accumulator/result contract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.widening_reduction_intrinsic",
            contract->intrinsic,
            "selected typed RVV product-reduction widening reduction intrinsic"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.scalar_seed_splat_intrinsic",
            contract->scalarSeedSplatIntrinsic,
            "selected typed RVV product-reduction scalar seed splat intrinsic"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.reduction_store_vl",
            contract->reductionStoreVL,
            "selected typed RVV product-reduction reduction store VL"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.scalar_result_runtime_boundary",
            isProductReductionDequantization
                ? kRVVProductReductionDequantVectorCarryBoundary
                : kRVVProductReductionOutCarryBoundary,
            "selected typed RVV product-reduction scalar result boundary"))
      return error;
    if (isProductReductionDequantization) {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.dequantization_relation",
              contract->dequantizationRelation,
              "selected typed RVV product-reduction dequantization relation"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.dequantize_convert_intrinsic", "",
              "selected typed RVV product-reduction scalar dequant route "
              "without standalone vector dequant convert"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.dequantize_scale_intrinsic", "",
              "selected typed RVV product-reduction scalar dequant route "
              "without standalone vector dequant scale"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.dequant_scale_role",
              contract->dequantScaleRole,
              "selected typed RVV product-reduction dequant scale role"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.dequant_scale_c_type",
              contract->dequantScaleCType,
              "selected typed RVV product-reduction dequant scale C type"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.dequant_scale_name",
              contract->dequantScaleName,
              "selected typed RVV product-reduction dequant scale name"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.rhs_broadcast_intrinsic",
              contract->rhsBroadcastIntrinsic,
              "selected typed RVV product-reduction post-loop scalar dequant "
              "splat intrinsic"))
        return error;
      if (isProductReductionDequantClamp) {
        if (llvm::Error error = requireCandidateMetadataMirror(
                candidate, "tcrv_rvv.lower_bound_role",
                contract->lowerBoundRole,
                "selected typed RVV product-reduction lower bound role"))
          return error;
        if (llvm::Error error = requireCandidateMetadataMirror(
                candidate, "tcrv_rvv.upper_bound_role",
                contract->upperBoundRole,
                "selected typed RVV product-reduction upper bound role"))
          return error;
        if (llvm::Error error = requireCandidateMetadataMirror(
                candidate, "tcrv_rvv.lower_bound_c_type",
                contract->lowerBoundCType,
                "selected typed RVV product-reduction lower bound C type"))
          return error;
        if (llvm::Error error = requireCandidateMetadataMirror(
                candidate, "tcrv_rvv.upper_bound_c_type",
                contract->upperBoundCType,
                "selected typed RVV product-reduction upper bound C type"))
          return error;
        if (llvm::Error error = requireCandidateMetadataMirror(
                candidate, "tcrv_rvv.bound_order", contract->boundOrder,
                "selected typed RVV product-reduction clamp bound order"))
          return error;
        if (llvm::Error error = requireCandidateMetadataMirror(
                candidate, "tcrv_rvv.clamp_relation", contract->clampRelation,
                "selected typed RVV product-reduction clamp relation"))
          return error;
        if (llvm::Error error = requireCandidateMetadataMirror(
                candidate, "tcrv_rvv.select_layout", contract->selectLayout,
                "selected typed RVV product-reduction clamp select layout"))
          return error;
        if (llvm::Error error = requireCandidateMetadataMirror(
                candidate, "tcrv_rvv.compare_predicate_kind",
                contract->comparePredicateKind,
                "selected typed RVV product-reduction lower compare "
                "predicate"))
          return error;
        if (llvm::Error error = requireCandidateMetadataMirror(
                candidate, "tcrv_rvv.secondary_compare_predicate_kind",
                contract->secondaryComparePredicateKind,
                "selected typed RVV product-reduction secondary compare "
                "predicate"))
          return error;
        if (llvm::Error error = requireCandidateMetadataMirror(
                candidate, "tcrv_rvv.compare_intrinsic",
                contract->compareIntrinsic,
                "selected typed RVV product-reduction lower compare "
                "intrinsic"))
          return error;
        if (llvm::Error error = requireCandidateMetadataMirror(
                candidate, "tcrv_rvv.secondary_compare_intrinsic",
                contract->secondaryCompareIntrinsic,
                "selected typed RVV product-reduction secondary compare "
                "intrinsic"))
          return error;
        if (llvm::Error error = requireCandidateMetadataMirror(
                candidate, "tcrv_rvv.masked_merge_intrinsic",
                contract->maskedMergeIntrinsic,
                "selected typed RVV product-reduction clamp select intrinsic"))
          return error;
      }
    } else {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.dequantization_relation", "",
              "selected typed RVV product-reduction without dequant relation"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.dequantize_convert_intrinsic", "",
              "selected typed RVV product-reduction without dequant convert"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.dequantize_scale_intrinsic", "",
              "selected typed RVV product-reduction without dequant scale"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.dequant_scale_role", "",
              "selected typed RVV product-reduction without dequant scale role"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.dequant_scale_c_type", "",
              "selected typed RVV product-reduction without dequant scale type"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.dequant_scale_name", "",
              "selected typed RVV product-reduction without dequant scale name"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.rhs_broadcast_intrinsic", "",
              "selected typed RVV product-reduction without scalar dequant "
              "splat"))
        return error;
    }
  } else {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.widening_dot_accumulator_layout",
            contract->wideningDotProductAccumulatorLayout,
            "selected typed RVV widening dot accumulator layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.widening_dot_result_layout",
            contract->wideningDotProductResultLayout,
            "selected typed RVV widening dot result layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.widening_dot_relation",
            contract->wideningDotProductRelation,
            "selected typed RVV widening dot relation"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate,
            "tcrv_rvv.widening_dot_source_accumulator_result_contract",
            contract->wideningDotSourceAccumulatorResultContract,
            "selected typed RVV widening dot source/accumulator/result "
            "contract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.widening_dot_reduction_store_vl",
            contract->reductionStoreVL,
            "selected typed RVV widening dot reduction store VL"))
      return error;
  }
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.widening_product_intrinsic",
          contract->wideningProductIntrinsic,
          "selected typed RVV widening dot product intrinsic"))
    return error;

  if (!contract->maskRole.empty()) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_role", contract->maskRole,
            "selected typed RVV computed-mask widening dot mask role"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_source", contract->maskSource,
            "selected typed RVV computed-mask widening dot mask source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_memory_form",
            contract->maskMemoryForm,
            "selected typed RVV computed-mask widening dot mask memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.inactive_lane_zeroing_requirement",
            contract->inactiveLaneZeroingRequirement,
            "selected typed RVV computed-mask widening dot inactive lane "
            "zeroing requirement"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.compare_predicate_kind",
            contract->comparePredicateKind,
            "selected typed RVV computed-mask widening dot compare predicate"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.masked_widening_product_intrinsic",
            contract->maskedWideningProductIntrinsic,
            "selected typed RVV computed-mask widening dot product intrinsic"))
      return error;
  } else {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_role", "",
            "selected typed RVV computed-mask widening dot mask role"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_source", "",
            "selected typed RVV computed-mask widening dot mask source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_memory_form", "",
            "selected typed RVV computed-mask widening dot mask memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.inactive_lane_zeroing_requirement", "",
            "selected typed RVV computed-mask widening dot inactive lane "
            "zeroing requirement"))
      return error;
    if (!isProductReductionDequantClamp) {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.compare_predicate_kind", "",
              "selected typed RVV computed-mask widening dot compare "
              "predicate"))
        return error;
    }
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.masked_widening_product_intrinsic", "",
            "selected typed RVV computed-mask widening dot product intrinsic"))
      return error;
  }

  if (!contract->stridedMemoryLayout.empty()) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.strided_memory_layout",
            contract->stridedMemoryLayout,
            "selected typed RVV strided widening dot memory layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.lhs_stride_source",
            contract->lhsStrideSource,
            "selected typed RVV strided widening dot lhs stride source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.rhs_stride_source",
            contract->rhsStrideSource,
            "selected typed RVV strided widening dot rhs stride source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.source_memory_form",
            contract->sourceMemoryForm,
            "selected typed RVV strided widening dot source memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.destination_memory_form",
            contract->destinationMemoryForm,
            "selected typed RVV strided widening dot destination memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.strided_load_intrinsic",
            contract->stridedLoadIntrinsic,
            "selected typed RVV strided widening dot source load intrinsic"))
      return error;
  } else {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.strided_memory_layout", "",
            "selected typed RVV strided widening dot memory layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.lhs_stride_source", "",
            "selected typed RVV strided widening dot lhs stride source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.rhs_stride_source", "",
            "selected typed RVV strided widening dot rhs stride source"))
      return error;
    if (isProductReductionChain) {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.source_memory_form",
              contract->sourceMemoryForm,
              "selected typed RVV product-reduction source memory form"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.destination_memory_form",
              contract->destinationMemoryForm,
              "selected typed RVV product-reduction destination memory form"))
        return error;
    } else {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.source_memory_form", "",
              "selected typed RVV strided widening dot source memory form"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.destination_memory_form", "",
              "selected typed RVV strided widening dot destination memory form"))
        return error;
    }
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.strided_load_intrinsic", "",
            "selected typed RVV strided widening dot source load intrinsic"))
      return error;
  }

  constexpr llvm::StringLiteral staleRouteFamilyMirrors[] = {
      "tcrv_rvv.elementwise_arithmetic_route_family_plan",
      "tcrv_rvv.scalar_broadcast_elementwise_route_family_plan",
      "tcrv_rvv.runtime_scalar_splat_store_route_family_plan",
      "tcrv_rvv.widening_conversion_route_family_plan",
      "tcrv_rvv.plain_compare_select_route_family_plan",
      "tcrv_rvv.computed_mask_select_route_family_plan",
      "tcrv_rvv.computed_mask_memory_route_family_plan",
      "tcrv_rvv.segment2_memory_route_family_plan",
      "tcrv_rvv.plain_macc_route_family_plan",
      "tcrv_rvv.scalar_broadcast_macc_route_family_plan",
      "tcrv_rvv.accumulation_route_family_plan",
      "tcrv_rvv.standalone_reduction_route_family_plan",
      "tcrv_rvv.base_memory_movement_route_family_plan",
      "tcrv_rvv.mask_tail_policy_route_family_plan",
      "tcrv_rvv.mask_tail_policy_owner",
      "tcrv_rvv.widening_macc_relation"};
  for (llvm::StringRef key : staleRouteFamilyMirrors)
    if (llvm::Error error = requireEmptyWideningDotReductionStaleMirror(
            candidate, key,
            "selected typed RVV non-widening-dot route-family mirror"))
      return error;

  return llvm::Error::success();
}

bool isRVVMAccContractComputedMask(
    plugin::rvv::RVVMAccRouteValidationKind kind) {
  return kind == plugin::rvv::RVVMAccRouteValidationKind::ComputedMask ||
         kind ==
             plugin::rvv::RVVMAccRouteValidationKind::RuntimeScalarComputedMask;
}

bool isRVVMAccContractRuntimeScalarComputedMask(
    plugin::rvv::RVVMAccRouteValidationKind kind) {
  return kind ==
         plugin::rvv::RVVMAccRouteValidationKind::RuntimeScalarComputedMask;
}

bool isRVVMAccContractWidening(
    plugin::rvv::RVVMAccRouteValidationKind kind) {
  return kind == plugin::rvv::RVVMAccRouteValidationKind::Widening;
}

std::size_t getRVVMAccContractRuntimeABIParameterCount(
    plugin::rvv::RVVMAccRouteValidationKind kind) {
  return isRVVMAccContractComputedMask(kind) ? 7 : 5;
}

llvm::Error requireRVVMAccContractStringField(llvm::StringRef consumerLabel,
                                              llvm::StringRef fieldLabel,
                                              llvm::StringRef actual,
                                              llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  if (expected.empty())
    return makeRVVTargetRouteError(llvm::Twine(consumerLabel) +
                                   " rejects stale " + fieldLabel +
                                   " facts before artifact export");
  return makeRVVTargetRouteError(llvm::Twine(consumerLabel) +
                                 " requires provider-derived " + fieldLabel +
                                 " '" + expected + "' but was '" + actual +
                                 "'");
}

llvm::Error requireRVVMAccContractIntField(llvm::StringRef consumerLabel,
                                           llvm::StringRef fieldLabel,
                                           std::int64_t actual,
                                           std::int64_t expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVTargetRouteError(llvm::Twine(consumerLabel) +
                                 " requires provider-derived " + fieldLabel +
                                 " " + llvm::Twine(expected) + " but was " +
                                 llvm::Twine(actual));
}

llvm::Error rejectRVVMAccStaleField(llvm::StringRef consumerLabel,
                                    llvm::StringRef fieldLabel,
                                    llvm::StringRef actual) {
  if (actual.empty())
    return llvm::Error::success();
  return makeRVVTargetRouteError(llvm::Twine(consumerLabel) +
                                 " rejects stale " + fieldLabel +
                                 " facts before artifact export");
}

const support::RuntimeABIParameter *findRuntimeElementCountABIParameter(
    llvm::ArrayRef<support::RuntimeABIParameter> parameters) {
  for (const support::RuntimeABIParameter &parameter : parameters)
    if (parameter.role ==
        support::RuntimeABIParameterRole::RuntimeElementCount)
      return &parameter;
  return nullptr;
}

llvm::Error validateRVVMAccRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVMAccRouteValidationContract &contract) {
  if (contract.core.requiredHeaderDeclarations.empty() ||
      contract.requiredHeaders.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived required_header_declarations before "
        "accepting the route artifact");

  for (llvm::StringRef header : contract.requiredHeaders) {
    if (header.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " saw an empty provider route header declaration");
    if (!routeHasHeader(route, header))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route header '" + header +
          "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVMAccRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVMAccRouteValidationContract &contract) {
  if (contract.core.cTypeMappingSummary.empty() || contract.typeMappings.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived route type mapping facts before artifact "
        "export");

  for (const plugin::rvv::RVVMAccRouteTypeMappingContract &mapping :
       contract.typeMappings) {
    if (mapping.sourceType.empty() || mapping.cType.empty())
      return makeRVVTargetRouteError(llvm::Twine(contract.consumerLabel) +
                                     " requires provider-derived " +
                                     mapping.label +
                                     " facts before artifact export");
    if (!routeHasTypeMapping(route, mapping.sourceType, mapping.cType))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route type mapping '" +
          mapping.sourceType + "' -> '" + mapping.cType + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVMAccRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVMAccRouteValidationContract &contract) {
  if (contract.core.runtimeABIOrder.empty() ||
      contract.core.runtimeABIParameters.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived runtime ABI order and ABI parameters "
        "before artifact export");
  const std::size_t expectedParameterCount =
      getRVVMAccContractRuntimeABIParameterCount(contract.kind);
  if (contract.core.runtimeABIParameters.size() != expectedParameterCount)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived runtime ABI parameter count " +
        llvm::Twine(expectedParameterCount) + " but contract has " +
        llvm::Twine(contract.core.runtimeABIParameters.size()));
  if (route.getABIMappings().size() != contract.core.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires rebuilt provider route ABI mapping count " +
        llvm::Twine(contract.core.runtimeABIParameters.size()) +
        " but route has " + llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        contract.core.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVMAccDescriptionAgainstContract(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVMAccRouteValidationContract &contract) {
  if (description.memoryForm != contract.memoryForm)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires selected typed RVV memory form '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(contract.memoryForm) +
        "' but was '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
            description.memoryForm) +
        "'");

  if (llvm::Error error = requireRVVMAccContractIntField(
          contract.consumerLabel, "SEW", description.sew, contract.sew))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "LMUL", description.lmul, contract.lmul))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "tail policy", description.tailPolicy,
          contract.tailPolicy))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "mask policy", description.maskPolicy,
          contract.maskPolicy))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "config contract",
          description.configContractID, contract.core.configContractID))
    return error;
  if (llvm::Error error = validateRVVRuntimeAVLVLSelectedBoundaryContract(
          description, contract.runtimeAVLVLContract))
    return error;
  if (llvm::Error error = validateRVVRouteLocalRuntimeAVLVLMirrors(
          contract.consumerLabel, contract.runtimeAVLVLContract,
          contract.core.runtimeControlPlanID, contract.core.runtimeABIOrder,
          contract.setVLIntrinsic, contract.core.vlCType,
          contract.emitCFullChunkVLName, contract.emitCLoopVLName,
          contract.emitCLoopInductionName))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "route operand binding plan",
          description.routeOperandBindingPlanID,
          contract.core.routeOperandBindingPlanID))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "route operand binding facts",
          description.routeOperandBindingSummary,
          contract.core.routeOperandBindingSummary))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "target leaf profile",
          description.targetLeafProfile, contract.core.targetLeafProfile))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "provider-supported mirror",
          description.providerSupportedMirror,
          contract.core.providerSupportedMirror))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "required header declarations",
          description.requiredHeaderDeclarations,
          contract.core.requiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "C type mapping summary",
          description.cTypeMappingSummary, contract.core.cTypeMappingSummary))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "typed compute op",
          description.typedComputeOpName, contract.core.typedComputeOpName))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel,
          isRVVMAccContractWidening(contract.kind)
              ? llvm::StringRef("widening arithmetic kind")
              : llvm::StringRef("multiply-add arithmetic kind"),
          description.maccArithmeticKind, contract.arithmeticKind))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "MAcc accumulator layout",
          description.maccAccumulatorLayout, contract.maccAccumulatorLayout))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "MAcc result layout",
          description.maccResultLayout, contract.maccResultLayout))
    return error;
  if (description.runtimeABIParameters.size() !=
      contract.core.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned runtime ABI parameter count " +
        llvm::Twine(contract.core.runtimeABIParameters.size()) +
        " before artifact export");
  for (std::size_t index = 0; index < contract.core.runtimeABIParameters.size();
       ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    const support::RuntimeABIParameter &expected =
        contract.core.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(actual, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-derived runtime ABI parameter " +
          std::to_string(index) + " to mirror provider-owned parameter '" +
          expected.cName + "' as " +
          support::stringifyRuntimeABIParameterRole(expected.role) +
          " with C type '" + expected.cType + "' before artifact export");
  }

  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "plain MAcc route-family plan",
          description.plainMAccRouteFamilyPlanID,
          contract.plainMAccRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "scalar-broadcast MAcc route-family plan",
          description.scalarBroadcastMAccRouteFamilyPlanID,
          contract.scalarBroadcastMAccRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "computed-mask MAcc route-family plan",
          description.accumulationRouteFamilyPlanID,
          contract.accumulationRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "contraction route-family plan",
          description.contractionRouteFamilyPlanID,
          contract.contractionRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "compare predicate",
          description.comparePredicateKind, contract.comparePredicateKind))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "accumulation compute suffix",
          description.accumulationComputeSuffix,
          contract.accumulationComputeSuffix))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "accumulation mask producer",
          description.accumulationMaskProducerSource,
          contract.accumulationMaskProducerSource))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "accumulation accumulator contract",
          description.accumulationAccumulatorContract,
          contract.accumulationAccumulatorContract))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "accumulation result contract",
          description.accumulationResultContract,
          contract.accumulationResultContract))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "mask role", description.maskRole,
          contract.maskRole))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "mask source", description.maskSource,
          contract.maskSource))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "mask memory form",
          description.maskMemoryForm, contract.maskMemoryForm))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "inactive-lane contract",
          description.inactiveLaneContract, contract.inactiveLaneContract))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "masked passthrough layout",
          description.maskedPassthroughLayout,
          contract.maskedPassthroughLayout))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "source memory form",
          description.sourceMemoryForm, contract.sourceMemoryForm))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "destination memory form",
          description.destinationMemoryForm, contract.destinationMemoryForm))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "indexed memory layout",
          description.indexedMemoryLayout, contract.indexedMemoryLayout))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "widening MAcc accumulator layout",
          description.wideningMAccAccumulatorLayout,
          contract.wideningMAccAccumulatorLayout))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "widening MAcc result layout",
          description.wideningMAccResultLayout,
          contract.wideningMAccResultLayout))
    return error;
  if (llvm::Error error = requireRVVMAccContractStringField(
          contract.consumerLabel, "widening MAcc relation",
          description.wideningMAccRelation, contract.wideningMAccRelation))
    return error;

  if (isRVVMAccContractWidening(contract.kind)) {
    if (llvm::Error error = requireRVVMAccContractIntField(
            contract.consumerLabel, "source SEW", description.sourceSEW,
            contract.sourceSEW))
      return error;
    if (llvm::Error error = requireRVVMAccContractStringField(
            contract.consumerLabel, "source LMUL", description.sourceLMUL,
            contract.sourceLMUL))
      return error;
    if (llvm::Error error = requireRVVMAccContractIntField(
            contract.consumerLabel, "result SEW", description.sew,
            contract.resultSEW))
      return error;
    if (llvm::Error error = requireRVVMAccContractStringField(
            contract.consumerLabel, "result LMUL", description.lmul,
            contract.resultLMUL))
      return error;
  }

  if (llvm::Error error = rejectRVVMAccStaleField(
          contract.consumerLabel, "standalone-reduction route-family",
          description.standaloneReductionRouteFamilyPlanID))
    return error;
  if (llvm::Error error = rejectRVVMAccStaleField(
          contract.consumerLabel, "standalone-reduction accumulator layout",
          description.reductionAccumulatorLayout))
    return error;
  if (llvm::Error error = rejectRVVMAccStaleField(
          contract.consumerLabel, "standalone-reduction result layout",
          description.reductionResultLayout))
    return error;
  if (llvm::Error error = rejectRVVMAccStaleField(
          contract.consumerLabel, "standalone-reduction store VL",
          description.reductionStoreVL))
    return error;
  if (llvm::Error error = rejectRVVMAccStaleField(
          contract.consumerLabel, "standalone-reduction scalar-result boundary",
          description.standaloneReductionScalarResultRuntimeBoundary))
    return error;
  if (llvm::Error error = rejectRVVMAccStaleField(
          contract.consumerLabel, "accumulation scalar carry",
          description.accumulationScalarCarryContract))
    return error;
  if (llvm::Error error = rejectRVVMAccStaleField(
          contract.consumerLabel, "base-memory route-family",
          description.baseMemoryMovementRouteFamilyPlanID))
    return error;
  if (llvm::Error error = rejectRVVMAccStaleField(
          contract.consumerLabel, "segment2 route-family",
          description.segment2MemoryRouteFamilyPlanID))
    return error;
  if (isRVVMAccContractWidening(contract.kind)) {
    if (llvm::Error error = rejectRVVMAccStaleField(
            contract.consumerLabel, "elementwise route-family",
            description.elementwiseArithmeticRouteFamilyPlanID))
      return error;
    if (llvm::Error error = rejectRVVMAccStaleField(
            contract.consumerLabel, "scalar-broadcast elementwise route-family",
            description.scalarBroadcastElementwiseRouteFamilyPlanID))
      return error;
    if (llvm::Error error = rejectRVVMAccStaleField(
            contract.consumerLabel, "runtime scalar splat-store route-family",
            description.runtimeScalarSplatStoreRouteFamilyPlanID))
      return error;
    if (llvm::Error error = rejectRVVMAccStaleField(
            contract.consumerLabel, "widening conversion route-family",
            description.wideningConversionRouteFamilyPlanID))
      return error;
    if (llvm::Error error = rejectRVVMAccStaleField(
            contract.consumerLabel, "computed-mask memory route-family",
            description.computedMaskMemoryRouteFamilyPlanID))
      return error;
    if (llvm::Error error = rejectRVVMAccStaleField(
            contract.consumerLabel, "widening dot relation",
            description.wideningDotProductRelation))
      return error;
    if (llvm::Error error = rejectRVVMAccStaleField(
            contract.consumerLabel, "widening product intrinsic",
            description.wideningProductIntrinsic))
      return error;
    if (llvm::Error error = rejectRVVMAccStaleField(
            contract.consumerLabel, "scalar seed splat intrinsic",
            description.scalarSeedSplatIntrinsic))
      return error;
  }

  return llvm::Error::success();
}

llvm::Error validateRVVMAccRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVMAccRouteValidationContract &contract) {
  const plugin::rvv::RVVRuntimeAVLVLSelectedBoundaryContract &runtimeContract =
      contract.runtimeAVLVLContract;
  if (contract.runtimeABIParameters.empty() || contract.resultName.empty() ||
      contract.vectorCType.empty() || runtimeContract.vlCType.empty() ||
      runtimeContract.setVLIntrinsic.empty() ||
      runtimeContract.emitCFullChunkVLName.empty() ||
      runtimeContract.emitCLoopVLName.empty() ||
      runtimeContract.emitCLoopInductionName.empty() ||
      runtimeContract.runtimeAVLParameter.cName.empty() ||
      runtimeContract.runtimeAVLParameter.cType.empty() ||
      contract.intrinsic.empty() ||
      contract.storeIntrinsic.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider contract result, vector C type, VL C type, setvl, "
        "compute, and store facts before validating route statements");
  if (!isRVVMAccContractWidening(contract.kind) &&
      contract.vectorLoadIntrinsic.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider contract vector load facts before validating "
        "route statements");
  if (isRVVMAccContractWidening(contract.kind) &&
      (contract.sourceVectorLoadIntrinsic.empty() ||
       contract.sourceVectorCType.empty() || contract.vectorLoadIntrinsic.empty()))
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider contract widening source/result load facts before "
        "validating route statements");
  if ((contract.kind ==
       plugin::rvv::RVVMAccRouteValidationKind::ScalarBroadcast ||
       isRVVMAccContractRuntimeScalarComputedMask(contract.kind)) &&
      contract.rhsBroadcastIntrinsic.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived RHS scalar splat facts before validating "
        "route statements");
  if (isRVVMAccContractComputedMask(contract.kind) &&
      (contract.maskName.empty() || contract.maskCType.empty() ||
       contract.compareIntrinsic.empty() ||
       contract.maskedMergeIntrinsic.empty()))
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived mask, compare, and masked-merge facts "
        "before validating route statements");

  const support::RuntimeABIParameter &runtimeNABI =
      runtimeContract.runtimeAVLParameter;
  const support::RuntimeABIParameter *orderedRuntimeElementCount =
      contract.runtimeABIParameters.empty()
          ? nullptr
          : &contract.runtimeABIParameters.back();
  if (!orderedRuntimeElementCount ||
      !runtimeABIParameterEquals(*orderedRuntimeElementCount, runtimeNABI))
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires runtime n/AVL ABI role to match provider runtime AVL/VL "
        "selected-boundary contract and provider runtime ABI order "
        "before validating route statements");

  if (route.getCallOpaqueSteps().size() != contract.expectedPreLoopStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires exact provider-built pre-loop statement count " +
        llvm::Twine(contract.expectedPreLoopStepCount) +
        " before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          preLoopSetVL, contract.consumerLabel, "pre-loop setvl",
          runtimeContract.setVLIntrinsic,
          {{runtimeNABI.cName, runtimeNABI.cType}},
          runtimeContract.emitCFullChunkVLName, runtimeContract.vlCType))
    return error;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires pre-loop statements to carry selected typed RVV source "
          "provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires exactly one provider-built runtime AVL/VL loop before "
        "artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != runtimeContract.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != runtimeContract.vlCType ||
      loop.upperBound.expression != runtimeNABI.cName ||
      loop.upperBound.cType != runtimeNABI.cType ||
      loop.step.expression != runtimeContract.emitCFullChunkVLName ||
      loop.step.cType != runtimeContract.vlCType)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-built loop bounds and step to mirror runtime "
        "n/AVL/VL route facts");
  if (loop.bodySteps.size() != contract.expectedLoopBodyStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires exact provider-built loop statement count " +
        llvm::Twine(contract.expectedLoopBodyStepCount) +
        " before artifact export");

  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeNABI.cName) + " - " +
       runtimeContract.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[0], contract.consumerLabel, "loop setvl",
          runtimeContract.setVLIntrinsic,
          {{expectedRemainingAVL, runtimeContract.vlCType}},
          runtimeContract.emitCLoopVLName, runtimeContract.vlCType))
    return error;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires loop statements to carry selected typed RVV source "
          "provenance");

  auto advancedPointer = [&](const support::RuntimeABIParameter &abi) {
    return (llvm::StringRef(abi.cName) + " + " +
            runtimeContract.emitCLoopInductionName)
        .str();
  };
  auto validateVectorLoad =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          const support::RuntimeABIParameter &abi, llvm::StringRef resultName,
          llvm::StringRef stepLabel, llvm::StringRef loadIntrinsic,
          llvm::StringRef resultCType) -> llvm::Error {
    const std::string pointer = advancedPointer(abi);
    return validateRVVProviderBuiltRouteStep(
        step, contract.consumerLabel, stepLabel, loadIntrinsic,
        {{pointer, abi.cType},
         {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
        resultName, resultCType);
  };
  auto validateOutputStore =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          const support::RuntimeABIParameter &outABI) -> llvm::Error {
    const std::string pointer = advancedPointer(outABI);
    return validateRVVProviderBuiltRouteStep(
        step, contract.consumerLabel, "output store", contract.storeIntrinsic,
        {{pointer, outABI.cType},
         {contract.resultName, contract.vectorCType},
         {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}});
  };

  if (!isRVVMAccContractComputedMask(contract.kind)) {
    const support::RuntimeABIParameter &lhsABI =
        contract.runtimeABIParameters[0];
    const support::RuntimeABIParameter &rhsOrScalarABI =
        contract.runtimeABIParameters[1];
    const support::RuntimeABIParameter &accumulatorABI =
        contract.runtimeABIParameters[2];
    const support::RuntimeABIParameter &outABI =
        contract.runtimeABIParameters[3];
    const llvm::StringRef sourceLoadIntrinsic =
        isRVVMAccContractWidening(contract.kind)
            ? llvm::StringRef(contract.sourceVectorLoadIntrinsic)
            : llvm::StringRef(contract.vectorLoadIntrinsic);
    const llvm::StringRef sourceVectorCType =
        isRVVMAccContractWidening(contract.kind)
            ? llvm::StringRef(contract.sourceVectorCType)
            : llvm::StringRef(contract.vectorCType);
    const llvm::StringRef lhsLoadLabel =
        isRVVMAccContractWidening(contract.kind)
            ? llvm::StringRef("lhs source load")
            : llvm::StringRef("lhs load");
    const llvm::StringRef rhsLoadLabel =
        isRVVMAccContractWidening(contract.kind)
            ? llvm::StringRef("rhs source load")
            : llvm::StringRef("rhs load");

    if (llvm::Error error = validateVectorLoad(
            loop.bodySteps[1], lhsABI, contract.lhsVectorName, lhsLoadLabel,
            sourceLoadIntrinsic, sourceVectorCType))
      return error;
    if (contract.kind ==
        plugin::rvv::RVVMAccRouteValidationKind::ScalarBroadcast) {
      if (llvm::Error error = validateRVVProviderBuiltRouteStep(
              loop.bodySteps[2], contract.consumerLabel, "RHS scalar splat",
              contract.rhsBroadcastIntrinsic,
              {{rhsOrScalarABI.cName, rhsOrScalarABI.cType},
               {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
              contract.rhsVectorName, contract.vectorCType))
        return error;
    } else if (llvm::Error error = validateVectorLoad(
                   loop.bodySteps[2], rhsOrScalarABI, contract.rhsVectorName,
                   rhsLoadLabel, sourceLoadIntrinsic, sourceVectorCType)) {
      return error;
    }
    if (llvm::Error error = validateVectorLoad(
            loop.bodySteps[3], accumulatorABI, contract.accumulatorVectorName,
            "accumulator load", contract.vectorLoadIntrinsic,
            contract.vectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[4], contract.consumerLabel,
            isRVVMAccContractWidening(contract.kind) ? "widening MAcc"
                                                     : "MAcc compute",
            contract.intrinsic,
            {{contract.accumulatorVectorName, contract.vectorCType},
             {contract.lhsVectorName, sourceVectorCType},
             {contract.rhsVectorName, sourceVectorCType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            contract.resultName, contract.vectorCType))
      return error;
    return validateOutputStore(loop.bodySteps[5], outABI);
  }

  const support::RuntimeABIParameter &compareLhsABI =
      contract.runtimeABIParameters[0];
  const support::RuntimeABIParameter &compareRhsOrScalarABI =
      contract.runtimeABIParameters[1];
  const support::RuntimeABIParameter &lhsABI =
      contract.runtimeABIParameters[2];
  const support::RuntimeABIParameter &rhsABI =
      contract.runtimeABIParameters[3];
  const support::RuntimeABIParameter &accumulatorABI =
      contract.runtimeABIParameters[4];
  const support::RuntimeABIParameter &outABI =
      contract.runtimeABIParameters[5];

  if (llvm::Error error = validateVectorLoad(
          loop.bodySteps[1], compareLhsABI, contract.lhsVectorName,
          "compare lhs load", contract.vectorLoadIntrinsic,
          contract.vectorCType))
    return error;
  if (isRVVMAccContractRuntimeScalarComputedMask(contract.kind)) {
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[2], contract.consumerLabel, "RHS scalar splat",
            contract.rhsBroadcastIntrinsic,
            {{compareRhsOrScalarABI.cName, compareRhsOrScalarABI.cType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            contract.rhsVectorName, contract.vectorCType))
      return error;
  } else if (llvm::Error error = validateVectorLoad(
                 loop.bodySteps[2], compareRhsOrScalarABI,
                 contract.rhsVectorName, "compare rhs load",
                 contract.vectorLoadIntrinsic, contract.vectorCType)) {
    return error;
  }
  if (llvm::Error error = validateVectorLoad(
          loop.bodySteps[3], lhsABI, contract.maccLHSVectorName,
          "MAcc lhs load", contract.vectorLoadIntrinsic, contract.vectorCType))
    return error;
  if (llvm::Error error = validateVectorLoad(
          loop.bodySteps[4], rhsABI, contract.maccRHSVectorName,
          "MAcc rhs load", contract.vectorLoadIntrinsic, contract.vectorCType))
    return error;
  if (llvm::Error error = validateVectorLoad(
          loop.bodySteps[5], accumulatorABI, contract.accumulatorVectorName,
          "accumulator load", contract.vectorLoadIntrinsic,
          contract.vectorCType))
    return error;
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[6], contract.consumerLabel, "compare mask",
          contract.compareIntrinsic,
          {{contract.lhsVectorName, contract.vectorCType},
           {contract.rhsVectorName, contract.vectorCType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
          contract.maskName, contract.maskCType))
    return error;
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[7], contract.consumerLabel, "active MAcc",
          contract.intrinsic,
          {{contract.accumulatorVectorName, contract.vectorCType},
           {contract.maccLHSVectorName, contract.vectorCType},
           {contract.maccRHSVectorName, contract.vectorCType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
          contract.activeMAccVectorName, contract.vectorCType))
    return error;
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[8], contract.consumerLabel, "masked merge",
          contract.maskedMergeIntrinsic,
          {{contract.accumulatorVectorName, contract.vectorCType},
           {contract.activeMAccVectorName, contract.vectorCType},
           {contract.maskName, contract.maskCType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
          contract.resultName, contract.vectorCType))
    return error;
  return validateOutputStore(loop.bodySteps[9], outABI);
}

llvm::Error validateRVVMAccRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVMAccRouteValidationContract &contract) {
  if (route.getRouteID() != contract.core.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires rebuilt provider route token '" +
        contract.core.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");

  if (llvm::Error error =
          validateRVVMAccDescriptionAgainstContract(description, contract))
    return error;
  if (llvm::Error error = validateRVVMAccRouteHeaders(route, contract))
    return error;
  if (llvm::Error error = validateRVVMAccRouteTypeMappings(route, contract))
    return error;
  if (llvm::Error error = validateRVVMAccRouteABIMappings(route, contract))
    return error;
  return validateRVVMAccRouteStatementPlan(route, contract);
}

llvm::Error validateRVVMAccTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  std::optional<plugin::rvv::RVVMAccRouteValidationContract> contract =
      plugin::rvv::getRVVMAccRouteValidationContract(context.description);
  if (!contract)
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-owned MAcc route "
        "validation contract before validating provider facts");
  return validateRVVMAccRoutePayloadFacts(context.route, context.description,
                                          *contract);
}

llvm::Error validateRVVMAccTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  std::optional<plugin::rvv::RVVMAccRouteMetadataMirrorContractSet> contract =
      plugin::rvv::getRVVMAccRouteMetadataMirrorContract(context.description);
  if (!contract)
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-owned MAcc metadata "
        "mirror contract before validating candidate mirrors");
  return validateRVVProviderMAccRouteMetadataMirrorContract(context.candidate,
                                                            *contract);
}

bool isRVVStandaloneReductionAccumulationTargetArtifactRouteFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVStandaloneReductionAccumulationRouteFamilyOperation(
      description.operation);
}

std::optional<plugin::rvv::RVVStandaloneReductionRouteValidationContract>
getRVVStandaloneReductionTargetRouteValidationContract(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return plugin::rvv::getRVVStandaloneReductionRouteValidationContract(
      description);
}

llvm::Error requireRVVStandaloneReductionContractStringField(
    llvm::StringRef consumerLabel, llvm::StringRef fieldLabel,
    llvm::StringRef actual, llvm::StringRef expected) {
  return requireRVVProviderCanonicalField(consumerLabel, fieldLabel, actual,
                                          expected);
}

llvm::Error requireRVVStandaloneReductionContractIntField(
    llvm::StringRef consumerLabel, llvm::StringRef fieldLabel,
    std::int64_t actual, std::int64_t expected) {
  if (actual == expected && actual != 0)
    return llvm::Error::success();
  return makeRVVTargetRouteError(
      llvm::Twine(consumerLabel) + " requires provider-derived " +
      fieldLabel + " " + llvm::Twine(expected) + " but was " +
      llvm::Twine(actual));
}

llvm::Error requireRVVStandaloneReductionEmptyResidue(
    llvm::StringRef consumerLabel, llvm::StringRef fieldLabel,
    llvm::StringRef actual) {
  if (actual.empty())
    return llvm::Error::success();
  return makeRVVTargetRouteError(llvm::Twine(consumerLabel) +
                                 " rejects stale " + fieldLabel);
}

llvm::Error validateRVVStandaloneReductionRuntimeABIFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVStandaloneReductionRouteValidationContract
        &contract) {
  if (description.runtimeABIParameters.size() !=
      contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived runtime ABI parameter count " +
        llvm::Twine(contract.runtimeABIParameters.size()) +
        " before artifact export");

  for (size_t index = 0; index < contract.runtimeABIParameters.size();
       ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    const support::RuntimeABIParameter &expected =
        contract.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(actual, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-derived runtime ABI parameter " +
          std::to_string(index) + " to bind " + expected.cName + " as " +
          support::stringifyRuntimeABIParameterRole(expected.role) +
          " with C type '" + expected.cType + "' before artifact export");
  }
  return llvm::Error::success();
}

llvm::Error validateRVVStandaloneReductionDescriptionAgainstContract(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVStandaloneReductionRouteValidationContract
        &contract) {
  const llvm::StringRef consumerLabel = contract.consumerLabel;
  if (description.memoryForm != contract.memoryForm)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires selected typed RVV memory form '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(contract.memoryForm) +
        "' but was '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
            description.memoryForm) +
        "'");
  if (llvm::Error error = requireRVVStandaloneReductionContractStringField(
          consumerLabel, "typed compute op", description.typedComputeOpName,
          contract.typedComputeOpName))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "element type", description.elementTypeName))
    return error;
  if (llvm::Error error = requireRVVStandaloneReductionContractIntField(
          consumerLabel, "SEW", description.sew, contract.sew))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "LMUL", description.lmul))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "tail policy", description.tailPolicy))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "mask policy", description.maskPolicy))
    return error;
  if (llvm::Error error = requireRVVStandaloneReductionContractStringField(
          consumerLabel, "provider-supported mirror",
          description.providerSupportedMirror, contract.providerSupportedMirror))
    return error;
  if (llvm::Error error = requireRVVStandaloneReductionContractStringField(
          consumerLabel, "target leaf profile", description.targetLeafProfile,
          contract.targetLeafProfile))
    return error;
  if (llvm::Error error = requireRVVStandaloneReductionContractStringField(
          consumerLabel, "standalone reduction route-family plan",
          description.standaloneReductionRouteFamilyPlanID,
          contract.standaloneReductionRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVStandaloneReductionContractStringField(
          consumerLabel, "scalar-result runtime boundary",
          description.standaloneReductionScalarResultRuntimeBoundary,
          contract.scalarResultRuntimeBoundary))
    return error;
  if (llvm::Error error = requireRVVStandaloneReductionContractStringField(
          consumerLabel, "required header declarations",
          description.requiredHeaderDeclarations,
          contract.requiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireRVVStandaloneReductionContractStringField(
          consumerLabel, "C type mapping summary",
          description.cTypeMappingSummary, contract.cTypeMappingSummary))
    return error;
  if (llvm::Error error = requireRVVStandaloneReductionContractStringField(
          consumerLabel, "route operand binding plan",
          description.routeOperandBindingPlanID,
          contract.routeOperandBindingPlanID))
    return error;
  if (llvm::Error error = requireRVVStandaloneReductionContractStringField(
          consumerLabel, "route operand binding summary",
          description.routeOperandBindingSummary,
          contract.routeOperandBindingSummary))
    return error;

  if (llvm::Error error = validateRVVStandaloneReductionRuntimeABIFacts(
          description, contract))
    return error;

  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "vector type", description.vectorTypeName))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "vector C type", description.vectorCType))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "standalone reduction source vector type",
          description.standaloneReductionSourceVectorTypeName))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "standalone reduction source vector C type",
          description.standaloneReductionSourceVectorCType))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "standalone reduction scalar C type",
          description.standaloneReductionScalarCType))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "standalone reduction scalar-result vector type",
          description.standaloneReductionScalarResultVectorTypeName))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "standalone reduction scalar-result vector C type",
          description.standaloneReductionScalarResultVectorCType))
    return error;
  if (llvm::Error error = requireRVVStandaloneReductionContractStringField(
          consumerLabel, "reduction accumulator layout",
          description.reductionAccumulatorLayout,
          contract.reductionAccumulatorLayout))
    return error;
  if (llvm::Error error = requireRVVStandaloneReductionContractStringField(
          consumerLabel, "reduction result layout",
          description.reductionResultLayout, contract.reductionResultLayout))
    return error;
  if (llvm::Error error = requireRVVStandaloneReductionContractStringField(
          consumerLabel, "reduction kind", description.reductionKind,
          contract.reductionKind))
    return error;
  if (llvm::Error error = requireRVVStandaloneReductionContractStringField(
          consumerLabel, "reduction store VL", description.reductionStoreVL,
          contract.reductionStoreVL))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "vector load intrinsic",
          description.vectorLoadIntrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "source splat intrinsic",
          description.sourceSplatIntrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "scalar seed splat intrinsic",
          description.scalarSeedSplatIntrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "scalar result store intrinsic",
          description.storeIntrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "reduction intrinsic", description.intrinsic))
    return error;

  const bool isComputedMask =
      contract.kind ==
          plugin::rvv::RVVStandaloneReductionRouteValidationKind::
              ComputedMask ||
      contract.kind ==
          plugin::rvv::RVVStandaloneReductionRouteValidationKind::
              RuntimeScalarComputedMask;
  if (isComputedMask) {
    if (llvm::Error error = requireRVVProviderDerivedField(
            consumerLabel, "mask type", description.maskTypeName))
      return error;
    if (llvm::Error error = requireRVVProviderDerivedField(
            consumerLabel, "mask C type", description.maskCType))
      return error;
    if (llvm::Error error =
            requireRVVStandaloneReductionContractStringField(
                consumerLabel, "mask role", description.maskRole,
                contract.maskRole))
      return error;
    if (llvm::Error error =
            requireRVVStandaloneReductionContractStringField(
                consumerLabel, "mask source", description.maskSource,
                contract.maskSource))
      return error;
    if (llvm::Error error =
            requireRVVStandaloneReductionContractStringField(
                consumerLabel, "mask memory form", description.maskMemoryForm,
                contract.maskMemoryForm))
      return error;
    if (llvm::Error error =
            requireRVVStandaloneReductionContractStringField(
                consumerLabel, "inactive-lane requirement",
                description.inactiveLaneZeroingRequirement,
                contract.inactiveLaneRequirement))
      return error;
    if (llvm::Error error =
            requireRVVStandaloneReductionContractStringField(
                consumerLabel, "compare predicate kind",
                description.comparePredicateKind,
                contract.comparePredicateKind))
      return error;
    if (llvm::Error error = requireRVVProviderDerivedField(
            consumerLabel, "compare intrinsic", description.compareIntrinsic))
      return error;
    if (llvm::Error error =
            requireRVVProviderDerivedField(consumerLabel,
                                           "masked merge intrinsic",
                                           description.maskedMergeIntrinsic))
      return error;
    if (llvm::Error error =
            requireRVVStandaloneReductionContractStringField(
                consumerLabel,
                "computed-mask accumulation route-family plan",
                description.accumulationRouteFamilyPlanID,
                contract.accumulationRouteFamilyPlanID))
      return error;
    if (llvm::Error error =
            requireRVVStandaloneReductionContractStringField(
                consumerLabel,
                "computed-mask accumulation compute suffix",
                description.accumulationComputeSuffix,
                contract.accumulationComputeSuffix))
      return error;
    if (llvm::Error error =
            requireRVVStandaloneReductionContractStringField(
                consumerLabel,
                "computed-mask accumulation mask producer source",
                description.accumulationMaskProducerSource,
                contract.accumulationMaskProducerSource))
      return error;
    if (llvm::Error error =
            requireRVVStandaloneReductionContractStringField(
                consumerLabel,
                "computed-mask accumulation accumulator contract",
                description.accumulationAccumulatorContract,
                contract.accumulationAccumulatorContract))
      return error;
    if (llvm::Error error =
            requireRVVStandaloneReductionContractStringField(
                consumerLabel, "computed-mask accumulation result contract",
                description.accumulationResultContract,
                contract.accumulationResultContract))
      return error;
    if (llvm::Error error =
            requireRVVStandaloneReductionContractStringField(
                consumerLabel,
                "computed-mask accumulation scalar-carry contract",
                description.accumulationScalarCarryContract,
                contract.accumulationScalarCarryContract))
      return error;
    if (contract.kind == plugin::rvv::
                             RVVStandaloneReductionRouteValidationKind::
                                 RuntimeScalarComputedMask) {
      if (llvm::Error error = requireRVVProviderDerivedField(
              consumerLabel, "RHS scalar splat intrinsic",
              description.rhsBroadcastIntrinsic))
        return error;
    } else if (!description.rhsBroadcastIntrinsic.empty()) {
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " rejects stale runtime-scalar RHS broadcast facts");
    }
  } else {
    if (llvm::Error error = requireRVVStandaloneReductionEmptyResidue(
            consumerLabel, "computed-mask mask role", description.maskRole))
      return error;
    if (llvm::Error error = requireRVVStandaloneReductionEmptyResidue(
            consumerLabel, "computed-mask mask source", description.maskSource))
      return error;
    if (llvm::Error error = requireRVVStandaloneReductionEmptyResidue(
            consumerLabel, "computed-mask mask memory form",
            description.maskMemoryForm))
      return error;
    if (llvm::Error error = requireRVVStandaloneReductionEmptyResidue(
            consumerLabel, "computed-mask inactive-lane facts",
            description.inactiveLaneZeroingRequirement))
      return error;
    if (llvm::Error error = requireRVVStandaloneReductionEmptyResidue(
            consumerLabel, "computed-mask compare predicate",
            description.comparePredicateKind))
      return error;
    if (llvm::Error error = requireRVVStandaloneReductionEmptyResidue(
            consumerLabel, "computed-mask compare intrinsic",
            description.compareIntrinsic))
      return error;
    if (llvm::Error error = requireRVVStandaloneReductionEmptyResidue(
            consumerLabel, "computed-mask masked merge intrinsic",
            description.maskedMergeIntrinsic))
      return error;
    if (llvm::Error error = requireRVVStandaloneReductionEmptyResidue(
            consumerLabel, "computed-mask accumulation route-family plan",
            description.accumulationRouteFamilyPlanID))
      return error;
    if (llvm::Error error = requireRVVStandaloneReductionEmptyResidue(
            consumerLabel, "computed-mask accumulation compute suffix",
            description.accumulationComputeSuffix))
      return error;
    if (llvm::Error error = requireRVVStandaloneReductionEmptyResidue(
            consumerLabel, "computed-mask accumulation mask producer source",
            description.accumulationMaskProducerSource))
      return error;
    if (llvm::Error error = requireRVVStandaloneReductionEmptyResidue(
            consumerLabel, "computed-mask accumulation accumulator contract",
            description.accumulationAccumulatorContract))
      return error;
    if (llvm::Error error = requireRVVStandaloneReductionEmptyResidue(
            consumerLabel, "computed-mask accumulation result contract",
            description.accumulationResultContract))
      return error;
    if (llvm::Error error = requireRVVStandaloneReductionEmptyResidue(
            consumerLabel, "computed-mask accumulation scalar-carry contract",
            description.accumulationScalarCarryContract))
      return error;
    if (llvm::Error error = requireRVVStandaloneReductionEmptyResidue(
            consumerLabel, "runtime-scalar RHS broadcast facts",
            description.rhsBroadcastIntrinsic))
      return error;
  }

  if (!description.scalarBroadcastElementwiseRouteFamilyPlanID.empty() ||
      !description.elementwiseArithmeticRouteFamilyPlanID.empty() ||
      !description.wideningConversionRouteFamilyPlanID.empty() ||
      !description.runtimeScalarSplatStoreRouteFamilyPlanID.empty() ||
      !description.plainMAccRouteFamilyPlanID.empty() ||
      !description.scalarBroadcastMAccRouteFamilyPlanID.empty() ||
      !description.wideningMAccRelation.empty() ||
      !description.wideningDotProductRelation.empty() ||
      !description.plainCompareSelectRouteFamilyPlanID.empty() ||
      !description.computedMaskSelectRouteFamilyPlanID.empty() ||
      !description.computedMaskMemoryRouteFamilyPlanID.empty() ||
      !description.segment2MemoryRouteFamilyPlanID.empty() ||
      !description.contractionRouteFamilyPlanID.empty() ||
      !description.baseMemoryMovementRouteFamilyPlanID.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " rejects stale non-standalone route-family facts");

  return llvm::Error::success();
}

llvm::Error validateRVVStandaloneReductionAccumulationRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVStandaloneReductionRouteValidationContract
        &contract) {
  if (contract.requiredHeaderDeclarations.empty() ||
      contract.requiredHeaders.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived required_header_declarations before "
        "accepting the route artifact");

  for (llvm::StringRef header : contract.requiredHeaders) {
    if (header.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " saw an empty provider route header declaration");
    if (!routeHasHeader(route, header))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route header '" + header +
          "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVStandaloneReductionAccumulationRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVStandaloneReductionRouteValidationContract
        &contract) {
  if (contract.cTypeMappingSummary.empty() || contract.typeMappings.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived route type mapping facts before artifact "
        "export");

  for (const plugin::rvv::RVVStandaloneReductionRouteTypeMappingContract
           &mapping : contract.typeMappings) {
    if (mapping.sourceType.empty() || mapping.cType.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-derived " + mapping.label +
          " facts before artifact export");
    if (!routeHasTypeMapping(route, mapping.sourceType, mapping.cType))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route type mapping '" +
          mapping.sourceType + "' -> '" + mapping.cType + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVStandaloneReductionAccumulationRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVStandaloneReductionRouteValidationContract
        &contract) {
  if (contract.runtimeABIOrder.empty() || contract.runtimeABIParameters.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived runtime ABI order and ABI parameters "
        "before artifact export");
  if (route.getABIMappings().size() != contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires rebuilt provider route ABI mapping count " +
        llvm::Twine(contract.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        contract.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVStandaloneReductionPreLoopStatements(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVStandaloneReductionRouteValidationContract
        &description,
    const support::RuntimeABIParameter &accumulator,
    const support::RuntimeABIParameter &out,
    const support::RuntimeABIParameter &runtimeN) {
  constexpr llvm::StringLiteral consumerLabel(
      "standalone reduction/accumulation target artifact consumer");
  const plugin::rvv::RVVRuntimeAVLVLSelectedBoundaryContract &runtimeContract =
      description.runtimeAVLVLContract;
  if (!runtimeABIParameterEquals(runtimeN,
                                 runtimeContract.runtimeAVLParameter))
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires statement-plan runtime n/AVL ABI parameter to come from "
        "the embedded runtime AVL/VL selected-boundary contract");
  if (route.getCallOpaqueSteps().size() != 3)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-built pre-loop setvl, scalar seed splat, and "
        "initial scalar-result store statements before artifact export");

  const llvm::StringRef scalarCType =
      description.standaloneReductionScalarCType;
  if (scalarCType.empty() ||
      description.standaloneReductionScalarResultVectorCType.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived scalar and scalar-result vector C types "
        "before validating pre-loop statements");

  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps()[0];
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          preLoopSetVL, consumerLabel, "pre-loop setvl",
          runtimeContract.setVLIntrinsic,
          {{runtimeContract.runtimeAVLParameter.cName,
            runtimeContract.runtimeAVLParameter.cType}},
          runtimeContract.emitCFullChunkVLName, runtimeContract.vlCType))
    return error;

  const conversion::emitc::TCRVEmitCCallOpaqueStep &initialSplat =
      route.getCallOpaqueSteps()[1];
  const std::string expectedAccumulatorLane0 =
      (llvm::StringRef(accumulator.cName) + "[0]").str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          initialSplat, consumerLabel, "pre-loop scalar seed splat",
          description.scalarSeedSplatIntrinsic,
          {{expectedAccumulatorLane0, scalarCType},
           {description.reductionStoreVL, runtimeContract.vlCType}},
          "standalone_initial_acc_vec",
          description.standaloneReductionScalarResultVectorCType))
    return error;

  const conversion::emitc::TCRVEmitCCallOpaqueStep &initialStore =
      route.getCallOpaqueSteps()[2];
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          initialStore, consumerLabel,
          "pre-loop initial scalar-result store", description.storeIntrinsic,
          {{out.cName, out.cType},
           {"standalone_initial_acc_vec",
            description.standaloneReductionScalarResultVectorCType},
           {description.reductionStoreVL, runtimeContract.vlCType}}))
    return error;

  return llvm::Error::success();
}

llvm::Error validateRVVPlainStandaloneReductionLoopStatements(
    const conversion::emitc::TCRVEmitCForLoop &loop,
    const plugin::rvv::RVVStandaloneReductionRouteValidationContract
        &description,
    const support::RuntimeABIParameter &lhs,
    const support::RuntimeABIParameter &out,
    const support::RuntimeABIParameter &runtimeN) {
  constexpr llvm::StringLiteral consumerLabel(
      "plain standalone reduction target artifact consumer");
  const plugin::rvv::RVVRuntimeAVLVLSelectedBoundaryContract &runtimeContract =
      description.runtimeAVLVLContract;
  if (!runtimeABIParameterEquals(runtimeN,
                                 runtimeContract.runtimeAVLParameter))
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires statement-plan runtime n/AVL ABI parameter to come from "
        "the embedded runtime AVL/VL selected-boundary contract");
  if (loop.bodySteps.size() != 5)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-built loop setvl, source load, scalar seed splat, "
        "reduction, and scalar-result store statements before artifact export");

  const llvm::StringRef scalarCType =
      description.standaloneReductionScalarCType;
  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeContract.runtimeAVLParameter.cName) + " - " +
       runtimeContract.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[0], consumerLabel, "loop setvl",
          runtimeContract.setVLIntrinsic,
          {{expectedRemainingAVL, runtimeContract.vlCType}},
          runtimeContract.emitCLoopVLName, runtimeContract.vlCType))
    return error;

  const std::string expectedLhsPointer =
      (llvm::StringRef(lhs.cName) + " + " +
       runtimeContract.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[1], consumerLabel, "source vector load",
          description.vectorLoadIntrinsic,
          {{expectedLhsPointer, lhs.cType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
          "lhs_vec", description.standaloneReductionSourceVectorCType))
    return error;

  const std::string expectedOutLane0 =
      (llvm::StringRef(out.cName) + "[0]").str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[2], consumerLabel, "loop scalar seed splat",
          description.scalarSeedSplatIntrinsic,
          {{expectedOutLane0, scalarCType},
           {description.reductionStoreVL, runtimeContract.vlCType}},
          "standalone_acc_vec",
          description.standaloneReductionScalarResultVectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[3], consumerLabel, "signed min/max/add reduction",
          description.intrinsic,
          {{"lhs_vec", description.standaloneReductionSourceVectorCType},
           {"standalone_acc_vec",
            description.standaloneReductionScalarResultVectorCType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
          description.resultName,
          description.standaloneReductionScalarResultVectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[4], consumerLabel, "scalar-result store",
          description.storeIntrinsic,
          {{out.cName, out.cType},
           {description.resultName,
            description.standaloneReductionScalarResultVectorCType},
           {description.reductionStoreVL, runtimeContract.vlCType}}))
    return error;

  return llvm::Error::success();
}

llvm::Error validateRVVComputedMaskStandaloneReductionLoopStatements(
    const conversion::emitc::TCRVEmitCForLoop &loop,
    const plugin::rvv::RVVStandaloneReductionRouteValidationContract
        &description,
    const support::RuntimeABIParameter &cmpLHS,
    const support::RuntimeABIParameter &cmpRHS,
    const support::RuntimeABIParameter &source,
    const support::RuntimeABIParameter &out,
    const support::RuntimeABIParameter &runtimeN) {
  constexpr llvm::StringLiteral consumerLabel(
      "computed-mask standalone reduction target artifact consumer");
  const plugin::rvv::RVVRuntimeAVLVLSelectedBoundaryContract &runtimeContract =
      description.runtimeAVLVLContract;
  if (!runtimeABIParameterEquals(runtimeN,
                                 runtimeContract.runtimeAVLParameter))
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires statement-plan runtime n/AVL ABI parameter to come from "
        "the embedded runtime AVL/VL selected-boundary contract");
  if (loop.bodySteps.size() != 10)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-built loop setvl, compare/source loads, compare, "
        "inactive neutral splat, merge, scalar seed, reduction, and "
        "scalar-result store statements before artifact export");

  const llvm::StringRef scalarCType =
      description.standaloneReductionScalarCType;
  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeContract.runtimeAVLParameter.cName) + " - " +
       runtimeContract.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[0], consumerLabel, "loop setvl",
          runtimeContract.setVLIntrinsic,
          {{expectedRemainingAVL, runtimeContract.vlCType}},
          runtimeContract.emitCLoopVLName, runtimeContract.vlCType))
    return error;

  const std::string expectedCmpLHSPointer =
      (llvm::StringRef(cmpLHS.cName) + " + " +
       runtimeContract.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[1], consumerLabel, "compare lhs vector load",
          description.vectorLoadIntrinsic,
          {{expectedCmpLHSPointer, cmpLHS.cType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
          "lhs_vec", description.standaloneReductionSourceVectorCType))
    return error;

  const std::string expectedCmpRHSPointer =
      (llvm::StringRef(cmpRHS.cName) + " + " +
       runtimeContract.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[2], consumerLabel, "compare rhs vector load",
          description.vectorLoadIntrinsic,
          {{expectedCmpRHSPointer, cmpRHS.cType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
          "rhs_vec", description.standaloneReductionSourceVectorCType))
    return error;

  const std::string expectedSourcePointer =
      (llvm::StringRef(source.cName) + " + " +
       runtimeContract.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[3], consumerLabel, "payload source vector load",
          description.vectorLoadIntrinsic,
          {{expectedSourcePointer, source.cType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
          "source_vec", description.standaloneReductionSourceVectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[4], consumerLabel, "compare predicate",
          description.compareIntrinsic,
          {{"lhs_vec", description.standaloneReductionSourceVectorCType},
           {"rhs_vec", description.standaloneReductionSourceVectorCType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
          description.maskName, description.maskCType))
    return error;

  const llvm::StringRef inactiveNeutral = description.inactiveNeutralLiteral;
  if (inactiveNeutral.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived operation-specific inactive neutral "
        "literal before artifact export");
  if (description.sourceSplatIntrinsic.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived source-vector splat intrinsic before "
        "artifact export");
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[5], consumerLabel, "inactive neutral splat",
          description.sourceSplatIntrinsic,
          {{inactiveNeutral, scalarCType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
          "standalone_inactive_neutral_vec",
          description.standaloneReductionSourceVectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[6], consumerLabel, "inactive-lane merge",
          description.maskedMergeIntrinsic,
          {{"standalone_inactive_neutral_vec",
           description.standaloneReductionSourceVectorCType},
           {"source_vec", description.standaloneReductionSourceVectorCType},
           {description.maskName, description.maskCType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
          "standalone_masked_source_vec",
          description.standaloneReductionSourceVectorCType))
    return error;

  const std::string expectedOutLane0 =
      (llvm::StringRef(out.cName) + "[0]").str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[7], consumerLabel, "loop scalar seed splat",
          description.scalarSeedSplatIntrinsic,
          {{expectedOutLane0, scalarCType},
           {description.reductionStoreVL, runtimeContract.vlCType}},
          "standalone_acc_vec",
          description.standaloneReductionScalarResultVectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[8], consumerLabel, "signed min/max/add reduction",
          description.intrinsic,
          {{"standalone_masked_source_vec",
            description.standaloneReductionSourceVectorCType},
           {"standalone_acc_vec",
            description.standaloneReductionScalarResultVectorCType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
          description.resultName,
          description.standaloneReductionScalarResultVectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[9], consumerLabel, "scalar-result store",
          description.storeIntrinsic,
          {{out.cName, out.cType},
           {description.resultName,
            description.standaloneReductionScalarResultVectorCType},
           {description.reductionStoreVL, runtimeContract.vlCType}}))
    return error;

  return llvm::Error::success();
}

llvm::Error
validateRVVRuntimeScalarComputedMaskStandaloneReductionLoopStatements(
    const conversion::emitc::TCRVEmitCForLoop &loop,
    const plugin::rvv::RVVStandaloneReductionRouteValidationContract
        &description,
    const support::RuntimeABIParameter &cmpLHS,
    const support::RuntimeABIParameter &rhsScalar,
    const support::RuntimeABIParameter &source,
    const support::RuntimeABIParameter &out,
    const support::RuntimeABIParameter &runtimeN) {
  constexpr llvm::StringLiteral consumerLabel(
      "runtime-scalar computed-mask standalone reduction target artifact "
      "consumer");
  const plugin::rvv::RVVRuntimeAVLVLSelectedBoundaryContract &runtimeContract =
      description.runtimeAVLVLContract;
  if (!runtimeABIParameterEquals(runtimeN,
                                 runtimeContract.runtimeAVLParameter))
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires statement-plan runtime n/AVL ABI parameter to come from "
        "the embedded runtime AVL/VL selected-boundary contract");
  if (loop.bodySteps.size() != 10)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-built loop setvl, compare lhs load, RHS scalar "
        "splat, source load, compare, inactive neutral splat, merge, scalar "
        "seed, reduction, and scalar-result store statements before artifact "
        "export");

  const llvm::StringRef scalarCType =
      description.standaloneReductionScalarCType;
  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeContract.runtimeAVLParameter.cName) + " - " +
       runtimeContract.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[0], consumerLabel, "loop setvl",
          runtimeContract.setVLIntrinsic,
          {{expectedRemainingAVL, runtimeContract.vlCType}},
          runtimeContract.emitCLoopVLName, runtimeContract.vlCType))
    return error;

  const std::string expectedCmpLHSPointer =
      (llvm::StringRef(cmpLHS.cName) + " + " +
       runtimeContract.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[1], consumerLabel, "compare lhs vector load",
          description.vectorLoadIntrinsic,
          {{expectedCmpLHSPointer, cmpLHS.cType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
          "lhs_vec", description.standaloneReductionSourceVectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[2], consumerLabel, "RHS scalar splat",
          description.rhsBroadcastIntrinsic,
          {{rhsScalar.cName, rhsScalar.cType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
          "rhs_vec", description.standaloneReductionSourceVectorCType))
    return error;

  const std::string expectedSourcePointer =
      (llvm::StringRef(source.cName) + " + " +
       runtimeContract.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[3], consumerLabel, "payload source vector load",
          description.vectorLoadIntrinsic,
          {{expectedSourcePointer, source.cType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
          "source_vec", description.standaloneReductionSourceVectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[4], consumerLabel, "compare predicate",
          description.compareIntrinsic,
          {{"lhs_vec", description.standaloneReductionSourceVectorCType},
           {"rhs_vec", description.standaloneReductionSourceVectorCType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
          description.maskName, description.maskCType))
    return error;

  const llvm::StringRef inactiveNeutral = description.inactiveNeutralLiteral;
  if (inactiveNeutral.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived operation-specific inactive neutral "
        "literal before artifact export");
  if (description.sourceSplatIntrinsic.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived source-vector splat intrinsic before "
        "artifact export");
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[5], consumerLabel, "inactive neutral splat",
          description.sourceSplatIntrinsic,
          {{inactiveNeutral, scalarCType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
          "standalone_inactive_neutral_vec",
          description.standaloneReductionSourceVectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[6], consumerLabel, "inactive-lane merge",
          description.maskedMergeIntrinsic,
          {{"standalone_inactive_neutral_vec",
           description.standaloneReductionSourceVectorCType},
           {"source_vec", description.standaloneReductionSourceVectorCType},
           {description.maskName, description.maskCType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
          "standalone_masked_source_vec",
          description.standaloneReductionSourceVectorCType))
    return error;

  const std::string expectedOutLane0 =
      (llvm::StringRef(out.cName) + "[0]").str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[7], consumerLabel, "loop scalar seed splat",
          description.scalarSeedSplatIntrinsic,
          {{expectedOutLane0, scalarCType},
           {description.reductionStoreVL, runtimeContract.vlCType}},
          "standalone_acc_vec",
          description.standaloneReductionScalarResultVectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[8], consumerLabel, "signed min/max/add reduction",
          description.intrinsic,
          {{"standalone_masked_source_vec",
            description.standaloneReductionSourceVectorCType},
           {"standalone_acc_vec",
            description.standaloneReductionScalarResultVectorCType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
          description.resultName,
          description.standaloneReductionScalarResultVectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[9], consumerLabel, "scalar-result store",
          description.storeIntrinsic,
          {{out.cName, out.cType},
           {description.resultName,
            description.standaloneReductionScalarResultVectorCType},
           {description.reductionStoreVL, runtimeContract.vlCType}}))
    return error;

  return llvm::Error::success();
}

llvm::Error validateRVVStandaloneReductionAccumulationRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVStandaloneReductionRouteValidationContract
        &description) {
  const bool isPlainStandalone =
      isRVVPlainStandaloneReductionRouteFamilyOperation(description.operation);
  const bool isVectorComputedMaskStandalone =
      isRVVVectorComputedMaskStandaloneReductionRouteFamilyOperation(
          description.operation);
  const bool isComputedMaskStandalone =
      isRVVComputedMaskStandaloneReductionRouteFamilyOperation(
          description.operation);
  const bool isRuntimeScalarComputedMaskStandalone =
      isRVVRuntimeScalarComputedMaskStandaloneReductionRouteFamilyOperation(
          description.operation);
  const plugin::rvv::RVVRuntimeAVLVLSelectedBoundaryContract &runtimeContract =
      description.runtimeAVLVLContract;

  const support::RuntimeABIParameter *accumulatorABI = nullptr;
  const support::RuntimeABIParameter *outABI = nullptr;
  const support::RuntimeABIParameter *runtimeN = nullptr;
  const support::RuntimeABIParameter *plainLHSABI = nullptr;
  const support::RuntimeABIParameter *cmpLHSABI = nullptr;
  const support::RuntimeABIParameter *cmpRHSABI = nullptr;
  const support::RuntimeABIParameter *sourceABI = nullptr;
  if (isPlainStandalone) {
    if (description.runtimeABIParameters.size() < 4)
      return makeRVVTargetRouteError(
          "plain standalone reduction target artifact consumer requires "
          "provider-derived lhs, acc, out, and n ABI parameters before "
          "validating route statements");
    plainLHSABI = &description.runtimeABIParameters[0];
    accumulatorABI = &description.runtimeABIParameters[1];
    outABI = &description.runtimeABIParameters[2];
    runtimeN = &description.runtimeABIParameters[3];
  } else if (isComputedMaskStandalone) {
    if (description.runtimeABIParameters.size() < 6)
      return makeRVVTargetRouteError(
          "computed-mask standalone reduction target artifact consumer "
          "requires provider-derived cmp_lhs, cmp_rhs or rhs scalar, src, "
          "acc, out, and n ABI parameters before validating route statements");
    cmpLHSABI = &description.runtimeABIParameters[0];
    cmpRHSABI = &description.runtimeABIParameters[1];
    sourceABI = &description.runtimeABIParameters[2];
    accumulatorABI = &description.runtimeABIParameters[3];
    outABI = &description.runtimeABIParameters[4];
    runtimeN = &description.runtimeABIParameters[5];
  }

  if (!accumulatorABI || !outABI || !runtimeN)
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-derived scalar seed, scalar-result output, and runtime n "
        "ABI facts before validating route statements");

  if (route.getCallOpaqueSteps().empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-built pre-loop setvl statement facts before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != runtimeContract.setVLIntrinsic ||
      !stepHasResult(preLoopSetVL, runtimeContract.emitCFullChunkVLName,
                     runtimeContract.vlCType))
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "rebuilt provider route pre-loop setvl statement to define the "
        "full-chunk VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "standalone reduction/accumulation target artifact consumer "
          "requires pre-loop statements to carry selected typed RVV source "
          "provenance");
  if (llvm::Error error =
          validateRVVStandaloneReductionPreLoopStatements(
              route, description, *accumulatorABI, *outABI, *runtimeN))
    return error;

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "exactly one provider-built runtime AVL/VL loop before artifact "
        "export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != runtimeContract.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != runtimeContract.vlCType ||
      loop.step.expression != runtimeContract.emitCFullChunkVLName ||
      loop.step.cType != runtimeContract.vlCType)
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-built loop bounds and step to mirror runtime AVL/VL route "
        "facts");
  if (loop.upperBound.expression !=
          runtimeContract.runtimeAVLParameter.cName ||
      loop.upperBound.cType != runtimeContract.runtimeAVLParameter.cType)
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-built loop upper bound to use the runtime n/AVL ABI "
        "parameter");
  if (loop.bodySteps.empty() ||
      loop.bodySteps.front().callee != runtimeContract.setVLIntrinsic ||
      !stepHasResult(loop.bodySteps.front(), runtimeContract.emitCLoopVLName,
                     runtimeContract.vlCType))
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-built loop setvl statement to define per-iteration VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "standalone reduction/accumulation target artifact consumer "
          "requires loop statements to carry selected typed RVV source "
          "provenance");

  if (isPlainStandalone) {
    if (llvm::Error error =
            validateRVVPlainStandaloneReductionLoopStatements(
                loop, description, *plainLHSABI, *outABI, *runtimeN))
      return error;
  } else if (isVectorComputedMaskStandalone) {
    if (llvm::Error error =
            validateRVVComputedMaskStandaloneReductionLoopStatements(
                loop, description, *cmpLHSABI, *cmpRHSABI, *sourceABI,
                *outABI, *runtimeN))
      return error;
  } else if (isRuntimeScalarComputedMaskStandalone) {
    if (llvm::Error error =
            validateRVVRuntimeScalarComputedMaskStandaloneReductionLoopStatements(
                loop, description, *cmpLHSABI, *cmpRHSABI, *sourceABI,
                *outABI, *runtimeN))
      return error;
  }

  return llvm::Error::success();
}

llvm::Error validateRVVStandaloneReductionAccumulationRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  std::optional<plugin::rvv::RVVStandaloneReductionRouteValidationContract>
      contract =
          getRVVStandaloneReductionTargetRouteValidationContract(description);
  if (!contract)
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-owned route validation contract before artifact export");

  if (llvm::Error error = validateRVVRuntimeAVLVLSelectedBoundaryContract(
          description, contract->runtimeAVLVLContract))
    return error;
  if (llvm::Error error = validateRVVRouteLocalRuntimeAVLVLMirrors(
          contract->consumerLabel, contract->runtimeAVLVLContract,
          contract->runtimeControlPlanID, contract->runtimeABIOrder,
          contract->setVLIntrinsic, contract->vlCType,
          contract->emitCFullChunkVLName, contract->emitCLoopVLName,
          contract->emitCLoopInductionName))
    return error;

  if (route.getRouteID() != contract->emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("standalone reduction/accumulation target artifact "
                    "consumer requires rebuilt provider route id '") +
        contract->emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");

  if (llvm::Error error =
          validateRVVStandaloneReductionDescriptionAgainstContract(
              description, *contract))
    return error;

  if (llvm::Error error =
          validateRVVStandaloneReductionAccumulationRouteHeaders(route,
                                                                *contract))
    return error;
  if (llvm::Error error =
          validateRVVStandaloneReductionAccumulationRouteTypeMappings(
              route, *contract))
    return error;
  if (llvm::Error error =
          validateRVVStandaloneReductionAccumulationRouteABIMappings(
              route, *contract))
    return error;
  return validateRVVStandaloneReductionAccumulationRouteStatementPlan(
      route, *contract);
}

llvm::Error
validateRVVStandaloneReductionAccumulationTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVStandaloneReductionAccumulationRoutePayloadFacts(
      context.route, context.description);
}


llvm::Error
validateRVVStandaloneReductionAccumulationTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  std::optional<
      plugin::rvv::RVVStandaloneReductionRouteMetadataMirrorContractSet>
      contract =
          plugin::rvv::getRVVStandaloneReductionRouteMetadataMirrorContract(
              context.description);
  if (!contract)
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-owned standalone reduction metadata mirror contract before "
        "validating candidate mirrors");
  return validateRVVProviderStandaloneReductionRouteMetadataMirrorContract(
      context.candidate, *contract);
}

llvm::Error validateRVVCompareSelectMaskRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires "
        "provider-derived required_header_declarations before accepting the "
        "route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires at least one "
        "provider route header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "compare/select mask target artifact consumer saw an empty "
          "provider route header declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("compare/select mask target artifact consumer requires "
                      "rebuilt provider route header '") +
          trimmed + "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVCompareSelectMaskRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.vlCType.empty() || description.vectorTypeName.empty() ||
      description.vectorCType.empty() || description.maskTypeName.empty() ||
      description.maskCType.empty() || description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires "
        "provider-derived VL, vector, mask, and C type mapping facts before "
        "artifact export");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("compare/select mask target artifact consumer requires "
                    "rebuilt provider route type mapping '!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("compare/select mask target artifact consumer requires "
                    "rebuilt provider route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");
  if (!routeHasTypeMapping(route, description.maskTypeName,
                           description.maskCType))
    return makeRVVTargetRouteError(
        llvm::Twine("compare/select mask target artifact consumer requires "
                    "rebuilt provider route type mapping '") +
        description.maskTypeName + "' -> '" + description.maskCType + "'");
  if (!description.indexVectorTypeName.empty()) {
    if (description.indexVectorCType.empty())
      return makeRVVTargetRouteError(
          "compare/select mask target artifact consumer requires index vector "
          "C type facts when index vector type facts are present");
    if (!routeHasTypeMapping(route, description.indexVectorTypeName,
                             description.indexVectorCType))
      return makeRVVTargetRouteError(
          llvm::Twine("compare/select mask target artifact consumer requires "
                      "rebuilt provider route type mapping '") +
          description.indexVectorTypeName + "' -> '" +
          description.indexVectorCType + "'");
  }

  return llvm::Error::success();
}

bool isRVVCompareSelectMaskIndexedMemoryOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 ComputedMaskIndexedGatherLoadUnitStore ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 RuntimeScalarComputedMaskIndexedGatherLoadUnitStore ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 ComputedMaskIndexedScatterStoreUnitLoad ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad;
}

bool isRVVCompareSelectMaskStridedMemoryOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 ComputedMaskStridedStore ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 ComputedMaskStridedLoadUnitStore;
}

llvm::Error validateRVVCompareSelectMaskRuntimeABIFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  std::optional<plugin::rvv::RVVCompareSelectRouteValidationContract>
      compareSelectContract;
  if (isRVVCompareSelectMaskProducerRouteFamilyOperation(description.operation))
    compareSelectContract =
        plugin::rvv::getRVVCompareSelectRouteValidationContract(description);
  std::optional<
      plugin::rvv::RVVComputedMaskStridedMemoryRouteValidationContract>
      stridedContract;
  if (isRVVCompareSelectMaskStridedMemoryOperation(description.operation))
    stridedContract =
        plugin::rvv::getRVVComputedMaskStridedMemoryRouteValidationContract(
            description);
  std::optional<
      plugin::rvv::RVVComputedMaskIndexedMemoryRouteValidationContract>
      indexedContract;
  if (isRVVCompareSelectMaskIndexedMemoryOperation(description.operation))
    indexedContract =
        plugin::rvv::getRVVComputedMaskIndexedMemoryRouteValidationContract(
            description);
  std::optional<plugin::rvv::RVVUnitStrideMaskedMemoryRouteValidationContract>
      unitStrideMaskedContract;
  if (isRVVUnitStrideMaskedMemoryRouteFamilyOperation(description.operation))
    unitStrideMaskedContract =
        plugin::rvv::getRVVUnitStrideMaskedMemoryRouteValidationContract(
            description);
  llvm::StringRef expectedOrder =
      compareSelectContract
          ? llvm::StringRef(compareSelectContract->runtimeABIOrder)
      : stridedContract
          ? llvm::StringRef(stridedContract->runtimeABIOrder)
      : indexedContract
          ? llvm::StringRef(indexedContract->runtimeABIOrder)
      : unitStrideMaskedContract
          ? llvm::StringRef(unitStrideMaskedContract->runtimeABIOrder)
          : llvm::StringRef();
  if (expectedOrder.empty())
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires a known "
        "compare/select mask runtime ABI order owner before artifact export");
  if (description.runtimeABIOrder != expectedOrder)
    return makeRVVTargetRouteError(
        llvm::Twine("compare/select mask target artifact consumer requires "
                    "provider-derived runtime ABI order '") +
        expectedOrder + "' but was '" + description.runtimeABIOrder + "'");

  llvm::ArrayRef<support::RuntimeABIParameter> expectedParameters;
  if (compareSelectContract)
    expectedParameters = compareSelectContract->runtimeABIParameters;
  else if (stridedContract)
    expectedParameters = stridedContract->runtimeABIParameters;
  else if (indexedContract)
    expectedParameters = indexedContract->runtimeABIParameters;
  else if (unitStrideMaskedContract)
    expectedParameters = unitStrideMaskedContract->runtimeABIParameters;

  if (description.runtimeABIParameters.size() != expectedParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("compare/select mask target artifact consumer requires ") +
        llvm::Twine(expectedParameters.size()) +
        " provider-derived runtime ABI parameters before artifact export but "
        "saw " +
        llvm::Twine(description.runtimeABIParameters.size()));

  for (std::size_t index = 0; index < expectedParameters.size(); ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    const support::RuntimeABIParameter &expected =
        expectedParameters[index];
    if (!runtimeABIParameterEquals(actual, expected))
      return makeRVVTargetRouteError(
          llvm::Twine("compare/select mask target artifact consumer requires "
                      "provider-derived runtime ABI parameter[") +
          llvm::Twine(index) + "] to bind '" + expected.cName + "' as " +
          support::stringifyRuntimeABIParameterRole(expected.role) +
          " with C type '" + expected.cType + "' and ownership '" +
          support::stringifyRuntimeABIParameterOwnership(expected.ownership) +
          "' before artifact export but saw '" + actual.cName + "' as " +
          support::stringifyRuntimeABIParameterRole(actual.role) +
          " with C type '" + actual.cType + "' and ownership '" +
          support::stringifyRuntimeABIParameterOwnership(actual.ownership) +
          "'");
  }

  return llvm::Error::success();
}

llvm::StringRef getRVVCompareSelectMaskExpectedProviderSupportedMirror(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
    return "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-store-plan-validated";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskLoadStore:
    return "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-load-store-plan-validated";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return "provider_supported_mirror:rvv-computed-mask-unit-load-store-plan-validated";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
    return {};
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedGatherLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return {};
  default:
    return {};
  }
}

llvm::StringRef getRVVCompareSelectMaskExpectedTargetLeafProfile(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
    return "rvv-v1-typed-runtime-scalar-cmp-masked-store-leaf-profile.v1";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskLoadStore:
    return "rvv-v1-typed-runtime-scalar-cmp-masked-load-store-leaf-profile.v1";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return "rvv-v1-e32m1-computed-mask-unit-load-store-leaf-profile.v1";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
    return {};
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedGatherLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return {};
  default:
    return {};
  }
}

llvm::StringRef getRVVCompareSelectMaskExpectedRequiredHeaderDeclarations(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (isRVVCompareSelectMaskIndexedMemoryOperation(operation) ||
      isRVVCompareSelectMaskStridedMemoryOperation(operation))
    return {};
  if (isRVVUnitStrideMaskedMemoryRouteFamilyOperation(operation))
    return "stddef.h,stdint.h,riscv_vector.h";
  return {};
}

llvm::StringRef getRVVCompareSelectMaskExpectedCTypeMappingSummary(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStore:
    return "vl:size_t,lhs_payload:typed-vector,rhs_scalar:typed-scalar,mask:typed-mask,dst:masked-store";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskLoadStore:
    return "vl:size_t,lhs/source/passthrough:typed-vector,rhs_scalar:typed-scalar,mask:typed-mask,result:masked-load-store";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return "vl:size_t,compare/source/passthrough:signed-e32m1,mask:b32,result:masked-load-store";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
    return {};
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedGatherLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return {};
  default:
    return {};
  }
}

llvm::StringRef getRVVCompareSelectMaskExpectedMaskProducerSource(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskLoadStore:
    return "runtime-scalar-splat-compare-rhs";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return "vector-compare-rhs-load";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
    return {};
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedGatherLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return {};
  default:
    return {};
  }
}

llvm::StringRef getRVVCompareSelectMaskExpectedSourceMemoryForm(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskLoadStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return "unit-stride-load";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedGatherLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return {};
  default:
    return {};
  }
}

llvm::StringRef getRVVCompareSelectMaskExpectedDestinationMemoryForm(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskLoadStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return "unit-stride-store";
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
    return "masked-unit-store";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedGatherLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return {};
  default:
    return {};
  }
}

llvm::StringRef getRVVCompareSelectMaskExpectedInactiveLaneContract(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::CmpSelect:
    return "masked-off-lanes-preserve-passthrough-vector";
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
    return "masked-store-false-lanes-preserve-output-buffer";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskLoadStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return "masked-off-lanes-preserve-old-destination";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedGatherLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return {};
  default:
    return {};
  }
}

llvm::StringRef getRVVCompareSelectMaskExpectedPassthroughLayout(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::CmpSelect:
    return "passthrough-vector-preserves-inactive-lanes";
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
    return "masked-store-has-no-passthrough-load";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskLoadStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return "old-destination-vector-preserves-inactive-lanes";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedGatherLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return {};
  default:
    return {};
  }
}

llvm::StringRef getRVVCompareSelectMaskExpectedMaskedMemoryLayout(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
    return "unit-stride-lhs-runtime-scalar-threshold-source-masked-destination-runtime-abi";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskLoadStore:
    return "unit-stride-lhs-runtime-scalar-threshold-source-old-destination-runtime-abi";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return "unit-stride-compare-source-old-destination-runtime-abi";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedGatherLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return {};
  default:
    return {};
  }
}

llvm::StringRef getRVVCompareSelectMaskExpectedStridedMemoryLayout(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
    return {};
  default:
    return {};
  }
}

llvm::StringRef getRVVCompareSelectMaskExpectedSourceStrideSource(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (isRVVCompareSelectMaskStridedMemoryOperation(operation))
    return {};
  return {};
}

llvm::StringRef getRVVCompareSelectMaskExpectedDestinationStrideSource(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (isRVVCompareSelectMaskStridedMemoryOperation(operation))
    return {};
  return {};
}

llvm::Error requireRVVCompareSelectMaskProviderField(llvm::StringRef label,
                                                    llvm::StringRef actual,
                                                    llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  if (expected.empty())
    return makeRVVTargetRouteError(
        llvm::Twine("compare/select mask target artifact consumer rejects "
                    "stale provider-derived ") +
        label + " fact '" + actual + "'");
  if (actual.empty())
    return makeRVVTargetRouteError(
        llvm::Twine("compare/select mask target artifact consumer requires "
                    "provider-derived ") +
        label + " '" + expected + "' before artifact export");
  return makeRVVTargetRouteError(
      llvm::Twine("compare/select mask target artifact consumer requires "
                  "provider-derived ") +
      label + " '" + expected + "' but was '" + actual + "'");
}

llvm::Error validateRVVCompareSelectRouteValidationContract(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVCompareSelectRouteValidationContract &contract) {
  if (description.operation != contract.operation)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned operation '" +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            contract.operation) +
        "' but was '" +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            description.operation) +
        "'");
  if (description.memoryForm != contract.memoryForm)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned memory form '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(contract.memoryForm) +
        "' but was '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
            description.memoryForm) +
        "'");
  if (description.sew != contract.sew)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived SEW '" + llvm::Twine(contract.sew) +
        "' but was '" + llvm::Twine(description.sew) + "'");

  auto require = [&](llvm::StringRef label, llvm::StringRef actual,
                     llvm::StringRef expected) -> llvm::Error {
    return requireRVVCompareSelectMaskProviderField(label, actual, expected);
  };

  if (llvm::Error error =
          require("element type", description.elementTypeName,
                  contract.elementTypeName))
    return error;
  if (description.configContractID != contract.configContractID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned config contract '" +
        contract.configContractID + "' but description carried '" +
        description.configContractID + "'");
  if (llvm::Error error = validateRVVRuntimeAVLVLSelectedBoundaryContract(
          description, contract.runtimeAVLVLContract))
    return error;
  if (llvm::Error error = validateRVVRouteLocalRuntimeAVLVLMirrors(
          contract.consumerLabel, contract.runtimeAVLVLContract,
          contract.runtimeControlPlanID, contract.runtimeABIOrder,
          contract.setVLIntrinsic, contract.vlCType,
          contract.emitCFullChunkVLName, contract.emitCLoopVLName,
          contract.emitCLoopInductionName))
    return error;
  if (llvm::Error error = require("LMUL", description.lmul, contract.lmul))
    return error;
  if (llvm::Error error =
          require("tail policy", description.tailPolicy,
                  contract.tailPolicy))
    return error;
  if (llvm::Error error =
          require("mask policy", description.maskPolicy,
                  contract.maskPolicy))
    return error;
  if (llvm::Error error =
          require("provider_supported_mirror",
                  description.providerSupportedMirror,
                  contract.providerSupportedMirror))
    return error;
  if (llvm::Error error = require("target_leaf_profile",
                                  description.targetLeafProfile,
                                  contract.targetLeafProfile))
    return error;
  if (llvm::Error error =
          require("required header declarations",
                  description.requiredHeaderDeclarations,
                  contract.requiredHeaderDeclarations))
    return error;
  if (llvm::Error error =
          require("C type mapping summary", description.cTypeMappingSummary,
                  contract.cTypeMappingSummary))
    return error;
  if (llvm::Error error =
          require("route operand binding plan",
                  description.routeOperandBindingPlanID,
                  contract.routeOperandBindingPlanID))
    return error;
  if (llvm::Error error =
          require("route operand binding facts",
                  description.routeOperandBindingSummary,
                  contract.routeOperandBindingSummary))
    return error;
  if (llvm::Error error = require("typed compute op",
                                  description.typedComputeOpName,
                                  contract.typedComputeOpName))
    return error;
  if (llvm::Error error =
          require("vector type", description.vectorTypeName,
                  contract.vectorTypeName))
    return error;
  if (llvm::Error error =
          require("vector C type", description.vectorCType,
                  contract.vectorCType))
    return error;
  if (llvm::Error error =
          require("mask type", description.maskTypeName,
                  contract.maskTypeName))
    return error;
  if (llvm::Error error =
          require("mask C type", description.maskCType, contract.maskCType))
    return error;
  if (llvm::Error error =
          require("vector load callee", description.vectorLoadIntrinsic,
                  contract.vectorLoadIntrinsic))
    return error;
  if (llvm::Error error =
          require("source vector type", description.sourceVectorTypeName,
                  contract.sourceVectorTypeName))
    return error;
  if (llvm::Error error =
          require("source vector C type", description.sourceVectorCType,
                  contract.sourceVectorCType))
    return error;
  if (llvm::Error error =
          require("source vector load callee",
                  description.sourceVectorLoadIntrinsic,
                  contract.sourceVectorLoadIntrinsic))
    return error;
  if (llvm::Error error =
          require("dequantize convert callee",
                  description.dequantizeConvertIntrinsic,
                  contract.dequantizeConvertIntrinsic))
    return error;
  if (llvm::Error error =
          require("dequantize scale callee",
                  description.dequantizeScaleIntrinsic,
                  contract.dequantizeScaleIntrinsic))
    return error;
  if (llvm::Error error =
          require("primary compare predicate",
                  description.comparePredicateKind,
                  contract.comparePredicateKind))
    return error;
  if (llvm::Error error =
          require("secondary compare predicate",
                  description.secondaryComparePredicateKind,
                  contract.secondaryComparePredicateKind))
    return error;
  if (llvm::Error error =
          require("runtime scalar splat callee",
                  description.rhsBroadcastIntrinsic,
                  contract.rhsScalarSplatIntrinsic))
    return error;
  if (llvm::Error error =
          require("primary compare callee", description.compareIntrinsic,
                  contract.compareIntrinsic))
    return error;
  if (llvm::Error error =
          require("secondary compare callee",
                  description.secondaryCompareIntrinsic,
                  contract.secondaryCompareIntrinsic))
    return error;
  if (llvm::Error error =
          require("mask-and callee", description.maskAndIntrinsic,
                  contract.maskAndIntrinsic))
    return error;
  if (llvm::Error error = require("select callee", description.intrinsic,
                                  contract.selectIntrinsic))
    return error;
  if (llvm::Error error = require("store callee", description.storeIntrinsic,
                                  contract.storeIntrinsic))
    return error;
  if (llvm::Error error =
          require("result name", description.resultName,
                  contract.resultName))
    return error;
  if (llvm::Error error =
          require("mask name", description.maskName, contract.maskName))
    return error;
  if (llvm::Error error =
          require("plain compare-select route-family plan",
                  description.plainCompareSelectRouteFamilyPlanID,
                  contract.plainCompareSelectRouteFamilyPlanID))
    return error;
  if (llvm::Error error =
          require("computed-mask select route-family plan",
                  description.computedMaskSelectRouteFamilyPlanID,
                  contract.computedMaskSelectRouteFamilyPlanID))
    return error;
  if (llvm::Error error =
          require("computed-mask select producer source",
                  description.computedMaskSelectMaskProducerSource,
                  contract.computedMaskSelectMaskProducerSource))
    return error;
  if (llvm::Error error =
          require("mask/tail policy route-family plan",
                  description.maskTailPolicyRouteFamilyPlanID,
                  contract.maskTailPolicyRouteFamilyPlanID))
    return error;
  if (llvm::Error error =
          require("mask/tail policy owner", description.maskTailPolicyOwner,
                  contract.maskTailPolicyOwner))
    return error;
  if (llvm::Error error =
          require("mask role", description.maskRole, contract.maskRole))
    return error;
  if (llvm::Error error =
          require("mask source", description.maskSource, contract.maskSource))
    return error;
  if (llvm::Error error = require("mask memory form",
                                  description.maskMemoryForm,
                                  contract.maskMemoryForm))
    return error;
  if (contract.maskComposition.empty() && !description.maskComposition.empty())
    return makeRVVTargetRouteError(
        llvm::Twine("compare/select mask target artifact consumer rejects "
                    "stale provider-derived dual compare/select mask "
                    "composition fact '") +
        description.maskComposition + "' for non-dual route");
  if (llvm::Error error =
          require("dual compare/select mask composition",
                  description.maskComposition, contract.maskComposition))
    return error;
  if (llvm::Error error =
          require("select layout", description.selectLayout,
                  contract.selectLayout))
    return error;
  if (llvm::Error error =
          require("inactive-lane contract", description.inactiveLaneContract,
                  contract.inactiveLaneContract))
    return error;
  if (llvm::Error error =
          require("masked passthrough layout",
                  description.maskedPassthroughLayout,
                  contract.maskedPassthroughLayout))
    return error;
  if (llvm::Error error = require("source memory form",
                                  description.sourceMemoryForm,
                                  contract.sourceMemoryForm))
    return error;
  if (llvm::Error error =
          require("destination memory form", description.destinationMemoryForm,
                  contract.destinationMemoryForm))
    return error;
  if (llvm::Error error =
          require("indexed memory layout", description.indexedMemoryLayout,
                  contract.indexedMemoryLayout))
    return error;
  if (llvm::Error error =
          require("lower bound role", description.lowerBoundRole,
                  contract.lowerBoundRole))
    return error;
  if (llvm::Error error =
          require("upper bound role", description.upperBoundRole,
                  contract.upperBoundRole))
    return error;
  if (llvm::Error error =
          require("lower bound C type", description.lowerBoundCType,
                  contract.lowerBoundCType))
    return error;
  if (llvm::Error error =
          require("upper bound C type", description.upperBoundCType,
                  contract.upperBoundCType))
    return error;
  if (llvm::Error error =
          require("bound order", description.boundOrder, contract.boundOrder))
    return error;
  if (llvm::Error error = require("clamp relation",
                                  description.clampRelation,
                                  contract.clampRelation))
    return error;

  auto rejectStaleRouteFamilyFact =
      [&](llvm::StringRef label, llvm::StringRef actual) -> llvm::Error {
    if (actual.empty())
      return llvm::Error::success();
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " rejects stale non-compare/select route-family " + label + " fact '" +
        actual + "'");
  };
  if (llvm::Error error = rejectStaleRouteFamilyFact(
          "elementwise arithmetic route-family plan",
          description.elementwiseArithmeticRouteFamilyPlanID))
    return error;
  if (llvm::Error error = rejectStaleRouteFamilyFact(
          "scalar-broadcast elementwise route-family plan",
          description.scalarBroadcastElementwiseRouteFamilyPlanID))
    return error;
  if (llvm::Error error = rejectStaleRouteFamilyFact(
          "runtime scalar splat-store route-family plan",
          description.runtimeScalarSplatStoreRouteFamilyPlanID))
    return error;
  if (llvm::Error error = rejectStaleRouteFamilyFact(
          "widening conversion route-family plan",
          description.wideningConversionRouteFamilyPlanID))
    return error;
  if (llvm::Error error = rejectStaleRouteFamilyFact(
          "dequantization route-family plan",
          description.dequantizationRouteFamilyPlanID))
    return error;
  if (llvm::Error error = rejectStaleRouteFamilyFact(
          "computed-mask memory route-family plan",
          description.computedMaskMemoryRouteFamilyPlanID))
    return error;
  if (llvm::Error error = rejectStaleRouteFamilyFact(
          "computed-mask memory producer source",
          description.computedMaskMemoryMaskProducerSource))
    return error;
  if (llvm::Error error = rejectStaleRouteFamilyFact(
          "plain MAcc route-family plan",
          description.plainMAccRouteFamilyPlanID))
    return error;
  if (llvm::Error error = rejectStaleRouteFamilyFact(
          "scalar-broadcast MAcc route-family plan",
          description.scalarBroadcastMAccRouteFamilyPlanID))
    return error;
  if (llvm::Error error = rejectStaleRouteFamilyFact(
          "accumulation route-family plan",
          description.accumulationRouteFamilyPlanID))
    return error;
  if (llvm::Error error = rejectStaleRouteFamilyFact(
          "standalone reduction route-family plan",
          description.standaloneReductionRouteFamilyPlanID))
    return error;
  if (llvm::Error error = rejectStaleRouteFamilyFact(
          "contraction route-family plan",
          description.contractionRouteFamilyPlanID))
    return error;
  if (llvm::Error error = rejectStaleRouteFamilyFact(
          "base-memory route-family plan",
          description.baseMemoryMovementRouteFamilyPlanID))
    return error;
  if (llvm::Error error = rejectStaleRouteFamilyFact(
          "segment2 route-family plan",
          description.segment2MemoryRouteFamilyPlanID))
    return error;
  if (llvm::Error error = rejectStaleRouteFamilyFact(
          "widening MAcc relation", description.wideningMAccRelation))
    return error;
  if (llvm::Error error = rejectStaleRouteFamilyFact(
          "widening dot relation", description.wideningDotProductRelation))
    return error;

  if (description.runtimeABIParameters.size() !=
      contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) + " requires " +
        llvm::Twine(contract.runtimeABIParameters.size()) +
        " provider-owned runtime ABI parameters before artifact export but "
        "saw " +
        llvm::Twine(description.runtimeABIParameters.size()));
  for (std::size_t index = 0; index < contract.runtimeABIParameters.size();
       ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    const support::RuntimeABIParameter &expected =
        contract.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(actual, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires runtime ABI parameter[" + llvm::Twine(index) +
          "] to mirror provider-owned parameter '" + expected.cName + "' as " +
          support::stringifyRuntimeABIParameterRole(expected.role) +
          " with C type '" + expected.cType + "' and ownership '" +
          support::stringifyRuntimeABIParameterOwnership(expected.ownership) +
          "' before artifact export but saw '" + actual.cName + "' as " +
          support::stringifyRuntimeABIParameterRole(actual.role) +
          " with C type '" + actual.cType + "' and ownership '" +
          support::stringifyRuntimeABIParameterOwnership(actual.ownership) +
          "'");
  }

  if (contract.expectedPreLoopStepCount == 0 ||
      contract.expectedLoopBodyStepCount == 0)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned compare/select statement-plan expectations "
        "before artifact export");

  return llvm::Error::success();
}

llvm::Error validateRVVComputedMaskIndexedMemoryDescriptionAgainstContract(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVComputedMaskIndexedMemoryRouteValidationContract
        &contract) {
  auto require = [&](llvm::StringRef label, llvm::StringRef actual,
                     llvm::StringRef expected) -> llvm::Error {
    return requireRVVCompareSelectMaskProviderField(label, actual, expected);
  };

  if (description.operation != contract.operation)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned operation '" +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            contract.operation) +
        "' but was '" +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            description.operation) +
        "'");
  if (description.emitCRouteID != contract.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned route id '" + contract.emitCRouteID +
        "' but description carried '" + description.emitCRouteID + "'");
  if (description.memoryForm != contract.memoryForm)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned memory form '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
            contract.memoryForm) +
        "' but was '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
            description.memoryForm) +
        "'");
  if (description.sew != contract.sew)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived SEW '" + llvm::Twine(contract.sew) +
        "' but was '" + llvm::Twine(description.sew) + "'");

  if (llvm::Error error =
          require("element type", description.elementTypeName,
                  contract.elementTypeName))
    return error;
  if (llvm::Error error = require("LMUL", description.lmul, contract.lmul))
    return error;
  if (llvm::Error error =
          require("tail policy", description.tailPolicy,
                  contract.tailPolicy))
    return error;
  if (llvm::Error error =
          require("mask policy", description.maskPolicy,
                  contract.maskPolicy))
    return error;
  if (description.configContractID != contract.configContractID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned config contract '" +
        contract.configContractID + "' but description carried '" +
        description.configContractID + "'");
  if (llvm::Error error = validateRVVRuntimeAVLVLSelectedBoundaryContract(
          description, contract.runtimeAVLVLContract))
    return error;
  if (llvm::Error error = validateRVVRouteLocalRuntimeAVLVLMirrors(
          contract.consumerLabel, contract.runtimeAVLVLContract,
          contract.runtimeControlPlanID, contract.runtimeABIOrder,
          contract.setVLIntrinsic, contract.vlCType,
          contract.emitCFullChunkVLName, contract.emitCLoopVLName,
          contract.emitCLoopInductionName))
    return error;
  if (llvm::Error error =
          require("provider_supported_mirror",
                  description.providerSupportedMirror,
                  contract.providerSupportedMirror))
    return error;
  if (llvm::Error error = require("target_leaf_profile",
                                  description.targetLeafProfile,
                                  contract.targetLeafProfile))
    return error;
  if (llvm::Error error =
          require("required header declarations",
                  description.requiredHeaderDeclarations,
                  contract.requiredHeaderDeclarations))
    return error;
  if (llvm::Error error =
          require("C type mapping summary", description.cTypeMappingSummary,
                  contract.cTypeMappingSummary))
    return error;
  if (llvm::Error error =
          require("route operand binding plan",
                  description.routeOperandBindingPlanID,
                  contract.routeOperandBindingPlanID))
    return error;
  if (llvm::Error error =
          require("route operand binding facts",
                  description.routeOperandBindingSummary,
                  contract.routeOperandBindingSummary))
    return error;
  if (llvm::Error error = require("typed compute op",
                                  description.typedComputeOpName,
                                  contract.typedComputeOpName))
    return error;
  if (llvm::Error error =
          require("vector type", description.vectorTypeName,
                  contract.vectorTypeName))
    return error;
  if (llvm::Error error =
          require("vector C type", description.vectorCType,
                  contract.vectorCType))
    return error;
  if (llvm::Error error =
          require("index vector type", description.indexVectorTypeName,
                  contract.indexVectorTypeName))
    return error;
  if (llvm::Error error =
          require("index vector C type", description.indexVectorCType,
                  contract.indexVectorCType))
    return error;
  if (llvm::Error error =
          require("mask type", description.maskTypeName,
                  contract.maskTypeName))
    return error;
  if (llvm::Error error =
          require("mask C type", description.maskCType, contract.maskCType))
    return error;
  if (llvm::Error error =
          require("vector load callee", description.vectorLoadIntrinsic,
                  contract.vectorLoadIntrinsic))
    return error;
  if (llvm::Error error =
          require("index load callee", description.indexLoadIntrinsic,
                  contract.indexLoadIntrinsic))
    return error;
  if (llvm::Error error =
          require("index scale callee", description.indexScaleIntrinsic,
                  contract.indexScaleIntrinsic))
    return error;
  if (llvm::Error error =
          require("masked indexed load callee",
                  description.maskedLoadIntrinsic,
                  contract.maskedIndexedLoadIntrinsic))
    return error;
  if (llvm::Error error =
          require("masked indexed store callee",
                  description.indexedStoreIntrinsic,
                  contract.maskedIndexedStoreIntrinsic))
    return error;
  if (llvm::Error error = require("store callee",
                                  description.storeIntrinsic,
                                  contract.maskedStoreIntrinsic))
    return error;
  if (llvm::Error error =
          require("compare callee", description.compareIntrinsic,
                  contract.compareIntrinsic))
    return error;
  if (llvm::Error error =
          require("compare predicate", description.comparePredicateKind,
                  contract.comparePredicateKind))
    return error;
  if (llvm::Error error =
          require("computed-mask memory route-family plan",
                  description.computedMaskMemoryRouteFamilyPlanID,
                  contract.computedMaskMemoryRouteFamilyPlanID))
    return error;
  if (llvm::Error error =
          require("computed-mask memory producer source",
                  description.computedMaskMemoryMaskProducerSource,
                  contract.computedMaskMemoryMaskProducerSource))
    return error;
  if (contract.kind ==
      plugin::rvv::RVVComputedMaskIndexedMemoryRouteValidationKind::
          RuntimeScalarIndexedGatherMAccScatter) {
    if (llvm::Error error = require(
            "composite gather-MAcc-scatter route-family plan",
            description.compositeGatherMAccScatterRouteFamilyPlanID,
            contract.compositeGatherMAccScatterRouteFamilyPlanID))
      return error;
    if (llvm::Error error = require(
            "composite gather-MAcc-scatter typed compute chain",
            description.compositeGatherMAccScatterTypedComputeChain,
            contract.compositeGatherMAccScatterTypedComputeChain))
      return error;
  } else {
    if (llvm::Error error =
            require("composite gather-MAcc-scatter route-family plan",
                    description.compositeGatherMAccScatterRouteFamilyPlanID, ""))
      return error;
    if (llvm::Error error =
            require("composite gather-MAcc-scatter typed compute chain",
                    description.compositeGatherMAccScatterTypedComputeChain, ""))
      return error;
  }
  if (llvm::Error error =
          require("mask/tail policy route-family plan",
                  description.maskTailPolicyRouteFamilyPlanID,
                  contract.maskTailPolicyRouteFamilyPlanID))
    return error;
  if (llvm::Error error =
          require("mask/tail policy owner", description.maskTailPolicyOwner,
                  contract.maskTailPolicyOwner))
    return error;
  if (llvm::Error error =
          require("mask role", description.maskRole, contract.maskRole))
    return error;
  if (llvm::Error error =
          require("mask source", description.maskSource, contract.maskSource))
    return error;
  if (llvm::Error error = require("mask memory form",
                                  description.maskMemoryForm,
                                  contract.maskMemoryForm))
    return error;
  if (llvm::Error error =
          require("inactive-lane contract", description.inactiveLaneContract,
                  contract.inactiveLaneContract))
    return error;
  if (llvm::Error error =
          require("masked passthrough layout",
                  description.maskedPassthroughLayout,
                  contract.maskedPassthroughLayout))
    return error;
  if (llvm::Error error =
          require("masked memory layout", description.indexedMemoryLayout,
                  contract.indexedMemoryLayout))
    return error;
  if (llvm::Error error =
          require("indexed write-side contract",
                  description.indexedWriteSideContract,
                  contract.indexedWriteSideContract))
    return error;
  if (llvm::Error error = require("source memory form",
                                  description.sourceMemoryForm,
                                  contract.sourceMemoryForm))
    return error;
  if (llvm::Error error =
          require("destination memory form", description.destinationMemoryForm,
                  contract.destinationMemoryForm))
    return error;
  if (description.indexEEW != contract.indexEEW)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived index EEW '" +
        llvm::Twine(contract.indexEEW) + "' but was '" +
        llvm::Twine(description.indexEEW) + "'");
  if (llvm::Error error =
          require("offset unit", description.offsetUnit, contract.offsetUnit))
    return error;
  if (llvm::Error error =
          require("index source", description.indexSource,
                  contract.indexSource))
    return error;
  if (llvm::Error error =
          require("index uniqueness", description.indexUniqueness,
                  contract.indexUniqueness))
    return error;
  if (llvm::Error error =
          require("indexed data memory form",
                  description.indexedDataMemoryForm,
                  contract.indexedDataMemoryForm))
    return error;
  if (llvm::Error error =
          require("indexed destination memory form",
                  description.indexedDestinationMemoryForm,
                  contract.indexedDestinationMemoryForm))
    return error;
  if (llvm::Error error =
          require("plain compare-select route-family plan",
                  description.plainCompareSelectRouteFamilyPlanID, ""))
    return error;
  if (llvm::Error error =
          require("computed-mask select route-family plan",
                  description.computedMaskSelectRouteFamilyPlanID, ""))
    return error;
  if (!description.baseMemoryMovementRouteFamilyPlanID.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " rejects stale plain base-memory route-family facts");
  if (llvm::Error error =
          require("segment2 route-family plan",
                  description.segment2MemoryRouteFamilyPlanID, ""))
    return error;
  if (llvm::Error error =
          require("strided memory layout", description.stridedMemoryLayout, ""))
    return error;
  if (llvm::Error error =
          require("source stride source", description.sourceStrideSource, ""))
    return error;
  if (llvm::Error error =
          require("destination stride source", description.outStrideSource, ""))
    return error;

  if (description.runtimeABIParameters.size() !=
      contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) + " requires " +
        llvm::Twine(contract.runtimeABIParameters.size()) +
        " provider-owned runtime ABI parameters before artifact export but "
        "saw " +
        llvm::Twine(description.runtimeABIParameters.size()));
  for (std::size_t index = 0; index < contract.runtimeABIParameters.size();
       ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    const support::RuntimeABIParameter &expected =
        contract.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(actual, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires runtime ABI parameter[" + llvm::Twine(index) +
          "] to mirror provider-owned parameter '" + expected.cName +
          "' as " + support::stringifyRuntimeABIParameterRole(expected.role) +
          " with C type '" + expected.cType +
          "' before artifact export but saw '" + actual.cName + "' as " +
          support::stringifyRuntimeABIParameterRole(actual.role) +
          " with C type '" + actual.cType + "'");
  }

  if (contract.expectedPreLoopStepCount == 0 ||
      contract.expectedLoopBodyStepCount == 0)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned statement-plan expectations before artifact "
        "export");

  return llvm::Error::success();
}

llvm::Error validateRVVComputedMaskIndexedMemoryCanonicalProviderFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (!isRVVCompareSelectMaskIndexedMemoryOperation(description.operation))
    return llvm::Error::success();

  std::optional<
      plugin::rvv::RVVComputedMaskIndexedMemoryRouteValidationContract>
      contract =
          plugin::rvv::getRVVComputedMaskIndexedMemoryRouteValidationContract(
              description);
  if (!contract)
    return makeRVVTargetRouteError(
        "computed-mask indexed memory target artifact consumer requires "
        "provider-owned route validation contract before artifact export");
  return validateRVVComputedMaskIndexedMemoryDescriptionAgainstContract(
      description, *contract);
}

llvm::Error validateRVVComputedMaskIndexedMemoryRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVComputedMaskIndexedMemoryRouteValidationContract
        &contract) {
  if (contract.requiredHeaderDeclarations.empty() ||
      contract.requiredHeaders.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived required_header_declarations before "
        "accepting the route artifact");
  for (llvm::StringRef header : contract.requiredHeaders)
    if (!routeHasHeader(route, header))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route header '" + header + "'");
  return llvm::Error::success();
}

llvm::Error validateRVVComputedMaskIndexedMemoryRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVComputedMaskIndexedMemoryRouteValidationContract
        &contract) {
  if (contract.cTypeMappingSummary.empty() || contract.typeMappings.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived c_type_mapping facts before accepting "
        "the route artifact");
  for (const auto &mapping : contract.typeMappings)
    if (!routeHasTypeMapping(route, mapping.sourceType, mapping.cType))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route type mapping '" +
          mapping.sourceType + "' -> '" + mapping.cType + "' for " +
          mapping.label);
  return llvm::Error::success();
}

llvm::Error validateRVVComputedMaskIndexedMemoryRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVComputedMaskIndexedMemoryRouteValidationContract
        &contract) {
  if (route.getABIMappings().size() != contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires rebuilt provider route ABI mapping count " +
        llvm::Twine(contract.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        contract.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) +
          "] to mirror provider runtime ABI parameter '" + expected.cName +
          "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVComputedMaskIndexedMemoryRouteStatementPlanShape(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVComputedMaskIndexedMemoryRouteValidationContract
        &contract) {
  if (route.getCallOpaqueSteps().size() != contract.expectedPreLoopStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-built pre-loop to carry exactly " +
        llvm::Twine(contract.expectedPreLoopStepCount) + " statement(s)");
  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires exactly one provider-built runtime AVL/VL loop");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.bodySteps.size() != contract.expectedLoopBodyStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-built loop body to carry exactly " +
        llvm::Twine(contract.expectedLoopBodyStepCount) +
        " statements but saw " + llvm::Twine(loop.bodySteps.size()));
  return llvm::Error::success();
}

llvm::Error validateRVVComputedMaskStridedMemoryDescriptionAgainstContract(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVComputedMaskStridedMemoryRouteValidationContract
        &contract) {
  auto require = [&](llvm::StringRef label, llvm::StringRef actual,
                     llvm::StringRef expected) -> llvm::Error {
    return requireRVVCompareSelectMaskProviderField(label, actual, expected);
  };

  if (description.operation != contract.operation)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned operation '" +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            contract.operation) +
        "' but was '" +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            description.operation) +
        "'");
  if (description.emitCRouteID != contract.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned route id '" + contract.emitCRouteID +
        "' but description carried '" + description.emitCRouteID + "'");
  if (description.memoryForm != contract.memoryForm)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned memory form '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
            contract.memoryForm) +
        "' but was '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
            description.memoryForm) +
        "'");
  if (description.sew != contract.sew)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived SEW '" + llvm::Twine(contract.sew) +
        "' but was '" + llvm::Twine(description.sew) + "'");

  if (llvm::Error error =
          require("element type", description.elementTypeName,
                  contract.elementTypeName))
    return error;
  if (llvm::Error error = require("LMUL", description.lmul, contract.lmul))
    return error;
  if (llvm::Error error =
          require("tail policy", description.tailPolicy,
                  contract.tailPolicy))
    return error;
  if (llvm::Error error =
          require("mask policy", description.maskPolicy,
                  contract.maskPolicy))
    return error;
  if (description.configContractID != contract.configContractID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned config contract '" +
        contract.configContractID + "' but description carried '" +
        description.configContractID + "'");
  if (llvm::Error error = validateRVVRuntimeAVLVLSelectedBoundaryContract(
          description, contract.runtimeAVLVLContract))
    return error;
  if (llvm::Error error = validateRVVRouteLocalRuntimeAVLVLMirrors(
          contract.consumerLabel, contract.runtimeAVLVLContract,
          contract.runtimeControlPlanID, contract.runtimeABIOrder,
          contract.setVLIntrinsic, contract.vlCType,
          contract.emitCFullChunkVLName, contract.emitCLoopVLName,
          contract.emitCLoopInductionName))
    return error;
  if (llvm::Error error =
          require("provider_supported_mirror",
                  description.providerSupportedMirror,
                  contract.providerSupportedMirror))
    return error;
  if (llvm::Error error = require("target_leaf_profile",
                                  description.targetLeafProfile,
                                  contract.targetLeafProfile))
    return error;
  if (llvm::Error error =
          require("required header declarations",
                  description.requiredHeaderDeclarations,
                  contract.requiredHeaderDeclarations))
    return error;
  if (llvm::Error error =
          require("C type mapping summary", description.cTypeMappingSummary,
                  contract.cTypeMappingSummary))
    return error;
  if (llvm::Error error =
          require("route operand binding plan",
                  description.routeOperandBindingPlanID,
                  contract.routeOperandBindingPlanID))
    return error;
  if (llvm::Error error =
          require("route operand binding facts",
                  description.routeOperandBindingSummary,
                  contract.routeOperandBindingSummary))
    return error;
  if (llvm::Error error = require("typed compute op",
                                  description.typedComputeOpName,
                                  contract.typedComputeOpName))
    return error;
  if (llvm::Error error =
          require("vector type", description.vectorTypeName,
                  contract.vectorTypeName))
    return error;
  if (llvm::Error error =
          require("vector C type", description.vectorCType,
                  contract.vectorCType))
    return error;
  if (llvm::Error error =
          require("mask type", description.maskTypeName,
                  contract.maskTypeName))
    return error;
  if (llvm::Error error =
          require("mask C type", description.maskCType, contract.maskCType))
    return error;
  if (llvm::Error error =
          require("vector load callee", description.vectorLoadIntrinsic,
                  contract.vectorLoadIntrinsic))
    return error;
  if (llvm::Error error =
          require("masked strided load callee",
                  description.maskedLoadIntrinsic,
                  contract.maskedLoadIntrinsic))
    return error;
  if (llvm::Error error =
          require("store callee", description.storeIntrinsic,
                  contract.storeIntrinsic))
    return error;
  if (llvm::Error error =
          require("masked strided store callee",
                  description.stridedStoreIntrinsic,
                  contract.stridedStoreIntrinsic))
    return error;
  if (llvm::Error error =
          require("compare callee", description.compareIntrinsic,
                  contract.compareIntrinsic))
    return error;
  if (llvm::Error error =
          require("compare predicate", description.comparePredicateKind,
                  contract.comparePredicateKind))
    return error;
  if (llvm::Error error =
          require("computed-mask memory route-family plan",
                  description.computedMaskMemoryRouteFamilyPlanID,
                  contract.computedMaskMemoryRouteFamilyPlanID))
    return error;
  if (llvm::Error error =
          require("computed-mask memory producer source",
                  description.computedMaskMemoryMaskProducerSource,
                  contract.computedMaskMemoryMaskProducerSource))
    return error;
  if (llvm::Error error =
          require("mask/tail policy route-family plan",
                  description.maskTailPolicyRouteFamilyPlanID,
                  contract.maskTailPolicyRouteFamilyPlanID))
    return error;
  if (llvm::Error error =
          require("mask/tail policy owner", description.maskTailPolicyOwner,
                  contract.maskTailPolicyOwner))
    return error;
  if (llvm::Error error =
          require("mask role", description.maskRole, contract.maskRole))
    return error;
  if (llvm::Error error =
          require("mask source", description.maskSource, contract.maskSource))
    return error;
  if (llvm::Error error = require("mask memory form",
                                  description.maskMemoryForm,
                                  contract.maskMemoryForm))
    return error;
  if (llvm::Error error =
          require("inactive-lane contract", description.inactiveLaneContract,
                  contract.inactiveLaneContract))
    return error;
  if (llvm::Error error =
          require("masked passthrough layout",
                  description.maskedPassthroughLayout,
                  contract.maskedPassthroughLayout))
    return error;
  if (llvm::Error error =
          require("masked memory layout", description.indexedMemoryLayout,
                  contract.maskedMemoryLayout))
    return error;
  if (llvm::Error error =
          require("strided memory layout", description.stridedMemoryLayout,
                  contract.stridedMemoryLayout))
    return error;
  if (llvm::Error error = require("source memory form",
                                  description.sourceMemoryForm,
                                  contract.sourceMemoryForm))
    return error;
  if (llvm::Error error = require("destination memory form",
                                  description.destinationMemoryForm,
                                  contract.destinationMemoryForm))
    return error;
  if (llvm::Error error = require("source stride source",
                                  description.sourceStrideSource,
                                  contract.sourceStrideSource))
    return error;
  if (llvm::Error error =
          require("destination stride source", description.outStrideSource,
                  contract.destinationStrideSource))
    return error;

  if (description.indexEEW != 0)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " rejects stale provider-derived index EEW '" +
        llvm::Twine(description.indexEEW) + "'");
  if (llvm::Error error = require("offset unit", description.offsetUnit, ""))
    return error;
  if (llvm::Error error = require("index source", description.indexSource, ""))
    return error;
  if (llvm::Error error =
          require("index uniqueness", description.indexUniqueness, ""))
    return error;
  if (llvm::Error error =
          require("indexed data memory form",
                  description.indexedDataMemoryForm, ""))
    return error;
  if (llvm::Error error =
          require("indexed destination memory form",
                  description.indexedDestinationMemoryForm, ""))
    return error;

  if (description.runtimeABIParameters.size() !=
      contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) + " requires " +
        llvm::Twine(contract.runtimeABIParameters.size()) +
        " provider-owned runtime ABI parameters before artifact export but "
        "saw " +
        llvm::Twine(description.runtimeABIParameters.size()));
  for (std::size_t index = 0; index < contract.runtimeABIParameters.size();
       ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    const support::RuntimeABIParameter &expected =
        contract.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(actual, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires runtime ABI parameter[" + llvm::Twine(index) +
          "] to mirror provider-owned parameter '" + expected.cName +
          "' as " + support::stringifyRuntimeABIParameterRole(expected.role) +
          " with C type '" + expected.cType +
          "' before artifact export but saw '" + actual.cName + "' as " +
          support::stringifyRuntimeABIParameterRole(actual.role) +
          " with C type '" + actual.cType + "'");
  }

  if (llvm::Error error = requireRuntimeByteStrideContract(
          description, contract.consumerLabel, "source",
          contract.sourceStrideSource, contract.sourceStrideCType,
          contract.sourceStrideUnit,
          support::RuntimeABIParameterRole::SourceByteStride, "byte"))
    return error;
  if (llvm::Error error = requireRuntimeByteStrideContract(
          description, contract.consumerLabel, "destination",
          contract.destinationStrideSource, contract.destinationStrideCType,
          contract.destinationStrideUnit,
          support::RuntimeABIParameterRole::DestinationByteStride, "byte"))
    return error;

  if (contract.expectedPreLoopStepCount == 0 ||
      contract.expectedLoopBodyStepCount == 0)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned statement-plan expectations before artifact "
        "export");

  return llvm::Error::success();
}

llvm::Error validateRVVComputedMaskStridedMemoryRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVComputedMaskStridedMemoryRouteValidationContract
        &contract) {
  if (contract.requiredHeaderDeclarations.empty() ||
      contract.requiredHeaders.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived required_header_declarations before "
        "accepting the route artifact");
  for (llvm::StringRef header : contract.requiredHeaders)
    if (!routeHasHeader(route, header))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route header '" + header + "'");
  return llvm::Error::success();
}

llvm::Error validateRVVComputedMaskStridedMemoryRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVComputedMaskStridedMemoryRouteValidationContract
        &contract) {
  if (contract.cTypeMappingSummary.empty() || contract.typeMappings.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived c_type_mapping facts before accepting "
        "the route artifact");
  for (const auto &mapping : contract.typeMappings)
    if (!routeHasTypeMapping(route, mapping.sourceType, mapping.cType))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route type mapping '" +
          mapping.sourceType + "' -> '" + mapping.cType + "' for " +
          mapping.label);
  return llvm::Error::success();
}

llvm::Error validateRVVComputedMaskStridedMemoryRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVComputedMaskStridedMemoryRouteValidationContract
        &contract) {
  if (route.getABIMappings().size() != contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires rebuilt provider route ABI mapping count " +
        llvm::Twine(contract.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        contract.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) +
          "] to mirror provider runtime ABI parameter '" + expected.cName +
          "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVComputedMaskStridedMemoryRouteStatementPlanShape(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVComputedMaskStridedMemoryRouteValidationContract
        &contract) {
  if (route.getCallOpaqueSteps().size() != contract.expectedPreLoopStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-built pre-loop to carry exactly " +
        llvm::Twine(contract.expectedPreLoopStepCount) + " statement(s)");
  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires exactly one provider-built runtime AVL/VL loop");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.bodySteps.size() != contract.expectedLoopBodyStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-built loop body to carry exactly " +
        llvm::Twine(contract.expectedLoopBodyStepCount) +
        " statements but saw " + llvm::Twine(loop.bodySteps.size()));
  return llvm::Error::success();
}

llvm::Error validateRVVComputedMaskStridedMemoryCanonicalProviderFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (!isRVVCompareSelectMaskStridedMemoryOperation(description.operation))
    return llvm::Error::success();

  std::optional<
      plugin::rvv::RVVComputedMaskStridedMemoryRouteValidationContract>
      contract =
          plugin::rvv::getRVVComputedMaskStridedMemoryRouteValidationContract(
              description);
  if (!contract)
    return makeRVVTargetRouteError(
        "computed-mask strided memory target artifact consumer requires "
        "provider-owned route validation contract before artifact export");
  return validateRVVComputedMaskStridedMemoryDescriptionAgainstContract(
      description, *contract);
}

llvm::Error validateRVVUnitStrideMaskedMemoryDescriptionAgainstContract(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVUnitStrideMaskedMemoryRouteValidationContract
        &contract) {
  auto require = [&](llvm::StringRef label, llvm::StringRef actual,
                     llvm::StringRef expected) -> llvm::Error {
    return requireRVVCompareSelectMaskProviderField(label, actual, expected);
  };

  if (description.operation != contract.operation)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned operation '" +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            contract.operation) +
        "' but was '" +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            description.operation) +
        "'");
  if (description.emitCRouteID != contract.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned route id '" + contract.emitCRouteID +
        "' but description carried '" + description.emitCRouteID + "'");
  if (description.memoryForm != contract.memoryForm)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned memory form '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
            contract.memoryForm) +
        "' but was '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
            description.memoryForm) +
        "'");
  if (description.sew != contract.sew)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived SEW '" + llvm::Twine(contract.sew) +
        "' but was '" + llvm::Twine(description.sew) + "'");

  if (llvm::Error error =
          require("element type", description.elementTypeName,
                  contract.elementTypeName))
    return error;
  if (llvm::Error error = require("LMUL", description.lmul, contract.lmul))
    return error;
  if (llvm::Error error =
          require("tail policy", description.tailPolicy,
                  contract.tailPolicy))
    return error;
  if (llvm::Error error =
          require("mask policy", description.maskPolicy,
                  contract.maskPolicy))
    return error;
  if (description.configContractID != contract.configContractID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned config contract '" +
        contract.configContractID + "' but description carried '" +
        description.configContractID + "'");
  if (llvm::Error error = validateRVVRuntimeAVLVLSelectedBoundaryContract(
          description, contract.runtimeAVLVLContract))
    return error;
  if (llvm::Error error = validateRVVRouteLocalRuntimeAVLVLMirrors(
          contract.consumerLabel, contract.runtimeAVLVLContract,
          contract.runtimeControlPlanID, contract.runtimeABIOrder,
          contract.setVLIntrinsic, contract.vlCType,
          contract.emitCFullChunkVLName, contract.emitCLoopVLName,
          contract.emitCLoopInductionName))
    return error;
  if (llvm::Error error =
          require("provider_supported_mirror",
                  description.providerSupportedMirror,
                  contract.providerSupportedMirror))
    return error;
  if (llvm::Error error = require("target_leaf_profile",
                                  description.targetLeafProfile,
                                  contract.targetLeafProfile))
    return error;
  if (llvm::Error error =
          require("required header declarations",
                  description.requiredHeaderDeclarations,
                  contract.requiredHeaderDeclarations))
    return error;
  if (llvm::Error error =
          require("C type mapping summary", description.cTypeMappingSummary,
                  contract.cTypeMappingSummary))
    return error;
  if (llvm::Error error =
          require("route operand binding plan",
                  description.routeOperandBindingPlanID,
                  contract.routeOperandBindingPlanID))
    return error;
  if (llvm::Error error =
          require("route operand binding facts",
                  description.routeOperandBindingSummary,
                  contract.routeOperandBindingSummary))
    return error;
  if (llvm::Error error = require("typed compute op",
                                  description.typedComputeOpName,
                                  contract.typedComputeOpName))
    return error;
  if (llvm::Error error =
          require("vector type", description.vectorTypeName,
                  contract.vectorTypeName))
    return error;
  if (llvm::Error error =
          require("vector C type", description.vectorCType,
                  contract.vectorCType))
    return error;
  if (llvm::Error error =
          require("mask type", description.maskTypeName,
                  contract.maskTypeName))
    return error;
  if (llvm::Error error =
          require("mask C type", description.maskCType, contract.maskCType))
    return error;
  if (llvm::Error error =
          require("vector load callee", description.vectorLoadIntrinsic,
                  contract.vectorLoadIntrinsic))
    return error;
  if (llvm::Error error =
          require("masked load callee", description.maskedLoadIntrinsic,
                  contract.maskedLoadIntrinsic))
    return error;
  if (llvm::Error error = require("store callee",
                                  description.storeIntrinsic,
                                  contract.storeIntrinsic))
    return error;
  if (llvm::Error error =
          require("compare callee", description.compareIntrinsic,
                  contract.compareIntrinsic))
    return error;
  if (llvm::Error error =
          require("compare predicate", description.comparePredicateKind,
                  contract.comparePredicateKind))
    return error;
  if (llvm::Error error =
          require("runtime scalar splat callee",
                  description.rhsBroadcastIntrinsic,
                  contract.rhsScalarSplatIntrinsic))
    return error;
  if (llvm::Error error =
          require("mask role", description.maskRole, contract.maskRole))
    return error;
  if (llvm::Error error =
          require("mask source", description.maskSource, contract.maskSource))
    return error;
  if (llvm::Error error = require("mask memory form",
                                  description.maskMemoryForm,
                                  contract.maskMemoryForm))
    return error;
  if (llvm::Error error =
          require("inactive-lane contract", description.inactiveLaneContract,
                  contract.inactiveLaneContract))
    return error;
  if (llvm::Error error =
          require("masked passthrough layout",
                  description.maskedPassthroughLayout,
                  contract.maskedPassthroughLayout))
    return error;
  if (llvm::Error error =
          require("masked memory layout", description.indexedMemoryLayout,
                  contract.maskedMemoryLayout))
    return error;
  if (llvm::Error error = require("source memory form",
                                  description.sourceMemoryForm,
                                  contract.sourceMemoryForm))
    return error;
  if (llvm::Error error =
          require("destination memory form", description.destinationMemoryForm,
                  contract.destinationMemoryForm))
    return error;

  const bool isStaticMask =
      contract.kind ==
          plugin::rvv::RVVUnitStrideMaskedMemoryRouteValidationKind::
              MaskedUnitLoadStore ||
      contract.kind ==
          plugin::rvv::RVVUnitStrideMaskedMemoryRouteValidationKind::
              MaskedUnitStore;
  if (isStaticMask) {
    if (llvm::Error error =
            require("base-memory route-family plan",
                    description.baseMemoryMovementRouteFamilyPlanID,
                    contract.baseMemoryMovementRouteFamilyPlanID))
      return error;
    if (llvm::Error error =
            require("computed-mask memory route-family plan",
                    description.computedMaskMemoryRouteFamilyPlanID, ""))
      return error;
    if (llvm::Error error =
            require("computed-mask memory producer source",
                    description.computedMaskMemoryMaskProducerSource, ""))
      return error;
    if (llvm::Error error =
            require("mask/tail policy route-family plan",
                    description.maskTailPolicyRouteFamilyPlanID, ""))
      return error;
    if (llvm::Error error =
            require("mask/tail policy owner", description.maskTailPolicyOwner,
                    ""))
      return error;
  } else {
    if (llvm::Error error =
            require("computed-mask memory route-family plan",
                    description.computedMaskMemoryRouteFamilyPlanID,
                    contract.computedMaskMemoryRouteFamilyPlanID))
      return error;
    if (llvm::Error error =
            require("computed-mask memory producer source",
                    description.computedMaskMemoryMaskProducerSource,
                    contract.computedMaskMemoryMaskProducerSource))
      return error;
    if (llvm::Error error =
            require("mask/tail policy route-family plan",
                    description.maskTailPolicyRouteFamilyPlanID,
                    contract.maskTailPolicyRouteFamilyPlanID))
      return error;
    if (llvm::Error error =
            require("mask/tail policy owner", description.maskTailPolicyOwner,
                    contract.maskTailPolicyOwner))
      return error;
    if (llvm::Error error =
            require("base-memory route-family plan",
                    description.baseMemoryMovementRouteFamilyPlanID, ""))
      return error;
  }

  if (description.indexEEW != 0)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " rejects stale provider-derived index EEW '" +
        llvm::Twine(description.indexEEW) + "'");
  if (llvm::Error error = require("offset unit", description.offsetUnit, ""))
    return error;
  if (llvm::Error error = require("index source", description.indexSource, ""))
    return error;
  if (llvm::Error error =
          require("index uniqueness", description.indexUniqueness, ""))
    return error;
  if (llvm::Error error =
          require("indexed data memory form",
                  description.indexedDataMemoryForm, ""))
    return error;
  if (llvm::Error error =
          require("indexed destination memory form",
                  description.indexedDestinationMemoryForm, ""))
    return error;
  if (llvm::Error error =
          require("strided memory layout", description.stridedMemoryLayout, ""))
    return error;
  if (llvm::Error error =
          require("source stride source", description.sourceStrideSource, ""))
    return error;
  if (llvm::Error error =
          require("destination stride source", description.outStrideSource, ""))
    return error;

  if (description.runtimeABIParameters.size() !=
      contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) + " requires " +
        llvm::Twine(contract.runtimeABIParameters.size()) +
        " provider-owned runtime ABI parameters before artifact export but "
        "saw " +
        llvm::Twine(description.runtimeABIParameters.size()));
  for (std::size_t index = 0; index < contract.runtimeABIParameters.size();
       ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    const support::RuntimeABIParameter &expected =
        contract.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(actual, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires runtime ABI parameter[" + llvm::Twine(index) +
          "] to mirror provider-owned parameter '" + expected.cName +
          "' as " + support::stringifyRuntimeABIParameterRole(expected.role) +
          " with C type '" + expected.cType +
          "' before artifact export but saw '" + actual.cName + "' as " +
          support::stringifyRuntimeABIParameterRole(actual.role) +
          " with C type '" + actual.cType + "'");
  }

  if (contract.expectedPreLoopStepCount == 0 ||
      contract.expectedLoopBodyStepCount == 0)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned statement-plan expectations before artifact "
        "export");

  return llvm::Error::success();
}

llvm::Error validateRVVUnitStrideMaskedMemoryCanonicalProviderFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (!isRVVUnitStrideMaskedMemoryRouteFamilyOperation(description.operation))
    return llvm::Error::success();

  std::optional<plugin::rvv::RVVUnitStrideMaskedMemoryRouteValidationContract>
      contract =
          plugin::rvv::getRVVUnitStrideMaskedMemoryRouteValidationContract(
              description);
  if (!contract)
    return makeRVVTargetRouteError(
        "unit-stride masked memory target artifact consumer requires "
        "provider-owned route validation contract from typed "
        "body/config/runtime facts before artifact export");
  return validateRVVUnitStrideMaskedMemoryDescriptionAgainstContract(
      description, *contract);
}

llvm::Error validateRVVUnitStrideMaskedMemoryRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVUnitStrideMaskedMemoryRouteValidationContract
        &contract) {
  if (contract.requiredHeaderDeclarations.empty() ||
      contract.requiredHeaders.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived required_header_declarations before "
        "accepting the route artifact");
  for (llvm::StringRef header : contract.requiredHeaders)
    if (!routeHasHeader(route, header))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route header '" + header + "'");
  return llvm::Error::success();
}

llvm::Error validateRVVUnitStrideMaskedMemoryRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVUnitStrideMaskedMemoryRouteValidationContract
        &contract) {
  if (contract.cTypeMappingSummary.empty() || contract.typeMappings.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived c_type_mapping facts before accepting "
        "the route artifact");
  for (const auto &mapping : contract.typeMappings)
    if (!routeHasTypeMapping(route, mapping.sourceType, mapping.cType))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route type mapping '" +
          mapping.sourceType + "' -> '" + mapping.cType + "' for " +
          mapping.label);
  return llvm::Error::success();
}

llvm::Error validateRVVUnitStrideMaskedMemoryRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVUnitStrideMaskedMemoryRouteValidationContract
        &contract) {
  if (route.getABIMappings().size() != contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires rebuilt provider route ABI mapping count " +
        llvm::Twine(contract.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        contract.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) +
          "] to mirror provider runtime ABI parameter '" + expected.cName +
          "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVCompareSelectMaskDualProviderFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  const bool isRuntimeScalarDualCompareSelect =
      description.operation ==
      plugin::rvv::RVVSelectedBodyOperationKind::
          RuntimeScalarDualCompareMaskAndSelect;

  auto rejectStaleDualFact = [&](llvm::StringRef label,
                                 llvm::StringRef actual) -> llvm::Error {
    if (actual.empty())
      return llvm::Error::success();
    return makeRVVTargetRouteError(
        llvm::Twine("compare/select mask target artifact consumer rejects "
                    "stale provider-derived dual compare/select ") +
        label + " fact '" + actual + "' for non-dual route");
  };

  if (!isRuntimeScalarDualCompareSelect) {
    if (llvm::Error error = rejectStaleDualFact(
            "secondary compare predicate",
            description.secondaryComparePredicateKind))
      return error;
    if (llvm::Error error = rejectStaleDualFact(
            "secondary compare callee", description.secondaryCompareIntrinsic))
      return error;
    if (llvm::Error error = rejectStaleDualFact("mask-and callee",
                                                description.maskAndIntrinsic))
      return error;
    return rejectStaleDualFact("mask composition",
                               description.maskComposition);
  }

  if (description.secondaryComparePredicateKind.empty() ||
      description.secondaryCompareIntrinsic.empty() ||
      description.maskAndIntrinsic.empty() || description.maskComposition.empty())
    return makeRVVTargetRouteError(
        "dual compare/select target artifact consumer requires "
        "provider-derived secondary compare predicate, secondary compare "
        "callee, mask-and callee, and mask composition facts before artifact "
        "export");

  return llvm::Error::success();
}

std::size_t getRVVCompareSelectMaskExpectedLoopBodyStepCount(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return 6;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskLoadStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
    return 7;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedGatherLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherLoadUnitStore:
    return 9;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return 8;
  default:
    return 0;
  }
}

llvm::Error validateRVVCompareSelectMaskRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("compare/select mask target artifact consumer requires "
                    "rebuilt provider route ABI mapping count ") +
        llvm::Twine(description.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        description.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine("compare/select mask target artifact consumer requires "
                      "rebuilt provider route ABI mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("compare/select mask target artifact consumer requires "
                      "rebuilt provider route ABI mapping[") +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVCompareSelectMaskRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    std::size_t providerExpectedPreLoopStepCount = 0,
    std::size_t providerExpectedLoopBodyStepCount = 0,
    const plugin::rvv::RVVRuntimeAVLVLSelectedBoundaryContract
        *providerRuntimeAVLVLContract = nullptr) {
  constexpr llvm::StringLiteral consumerLabel(
      "compare/select mask target artifact consumer");
  if (llvm::Error error =
          validateRVVCompareSelectMaskRuntimeABIFacts(description))
    return error;
  const llvm::StringRef setVLIntrinsic =
      providerRuntimeAVLVLContract
          ? llvm::StringRef(providerRuntimeAVLVLContract->setVLIntrinsic)
          : description.setVLIntrinsic;
  const llvm::StringRef vlCType =
      providerRuntimeAVLVLContract
          ? llvm::StringRef(providerRuntimeAVLVLContract->vlCType)
          : description.vlCType;
  const llvm::StringRef emitCFullChunkVLName =
      providerRuntimeAVLVLContract
          ? llvm::StringRef(providerRuntimeAVLVLContract->emitCFullChunkVLName)
          : description.emitCFullChunkVLName;
  const llvm::StringRef emitCLoopVLName =
      providerRuntimeAVLVLContract
          ? llvm::StringRef(providerRuntimeAVLVLContract->emitCLoopVLName)
          : description.emitCLoopVLName;
  const llvm::StringRef emitCLoopInductionName =
      providerRuntimeAVLVLContract
          ? llvm::StringRef(
                providerRuntimeAVLVLContract->emitCLoopInductionName)
          : description.emitCLoopInductionName;

  if (setVLIntrinsic.empty() || description.vectorLoadIntrinsic.empty() ||
      description.compareIntrinsic.empty() || description.maskName.empty() ||
      description.resultName.empty() || description.maskCType.empty() ||
      description.vectorCType.empty() || vlCType.empty() ||
      emitCFullChunkVLName.empty() || emitCLoopVLName.empty() ||
      emitCLoopInductionName.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived setvl, compare, mask/result, vector/VL, "
        "and loop facts before validating route statements");

  const std::size_t expectedPreLoopStepCount =
      providerExpectedPreLoopStepCount != 0 ? providerExpectedPreLoopStepCount
                                            : 1;
  if (route.getCallOpaqueSteps().size() != expectedPreLoopStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires exactly " + llvm::Twine(expectedPreLoopStepCount) +
        " provider-built pre-loop setvl statement(s) before artifact export");
  const support::RuntimeABIParameter *runtimeN =
      providerRuntimeAVLVLContract
          ? &providerRuntimeAVLVLContract->runtimeAVLParameter
          : findRuntimeElementCountABIParameter(description);
  if (!runtimeN)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires a provider-derived runtime element count ABI parameter");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          preLoopSetVL, consumerLabel, "pre-loop setvl", setVLIntrinsic,
          {{runtimeN->cName, runtimeN->cType}}, emitCFullChunkVLName, vlCType))
    return error;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires pre-loop statements to carry selected typed RVV source "
          "provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires exactly one provider-built runtime AVL/VL loop before "
        "artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != emitCLoopInductionName ||
      loop.lowerBound.expression != "0" || loop.lowerBound.cType != vlCType ||
      loop.upperBound.expression != runtimeN->cName ||
      loop.upperBound.cType != runtimeN->cType ||
      loop.step.expression != emitCFullChunkVLName ||
      loop.step.cType != vlCType)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-built loop bounds and step to mirror runtime "
        "n/AVL/VL facts");
  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeN->cName) + " - " + emitCLoopInductionName)
          .str();
  const std::size_t expectedBodyStepCount =
      providerExpectedLoopBodyStepCount != 0
          ? providerExpectedLoopBodyStepCount
          : getRVVCompareSelectMaskExpectedLoopBodyStepCount(
                description.operation);
  if (expectedBodyStepCount == 0)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires a known compare/select mask statement plan owner before "
        "artifact export");
  if (loop.bodySteps.size() != expectedBodyStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) + " requires provider-built loop body to "
        "carry exactly " + llvm::Twine(expectedBodyStepCount) +
        " statements for operation '" +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            description.operation) +
        "' but saw " + llvm::Twine(loop.bodySteps.size()));
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps.front(), consumerLabel, "loop setvl",
          setVLIntrinsic, {{expectedRemainingAVL, vlCType}}, emitCLoopVLName,
          vlCType))
    return error;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires loop statements to carry selected typed RVV source "
          "provenance");

  auto pointerAtInduction =
      [&](const support::RuntimeABIParameter &abi) -> std::string {
    return (llvm::StringRef(abi.cName) + " + " + emitCLoopInductionName)
        .str();
  };
  auto stridedSourcePointerAtInduction =
      [&](const support::RuntimeABIParameter &sourceABI,
          const support::RuntimeABIParameter &strideABI) -> std::string {
    return ("(const int32_t *)((const uint8_t *)" +
            llvm::StringRef(sourceABI.cName) + " + (" +
            emitCLoopInductionName + " * " + strideABI.cName + "))")
        .str();
  };
  auto stridedDestinationPointerAtInduction =
      [&](const support::RuntimeABIParameter &destinationABI,
          const support::RuntimeABIParameter &strideABI) -> std::string {
    return ("(int32_t *)((uint8_t *)" + llvm::StringRef(destinationABI.cName) +
            " + (" + emitCLoopInductionName + " * " + strideABI.cName +
            "))")
        .str();
  };
  auto validateVectorLoad =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          llvm::StringRef stepLabel, const support::RuntimeABIParameter &abi,
          llvm::StringRef resultName) -> llvm::Error {
    return validateRVVProviderBuiltRouteStep(
        step, consumerLabel, stepLabel, description.vectorLoadIntrinsic,
        {{pointerAtInduction(abi), abi.cType}, {emitCLoopVLName, vlCType}},
        resultName, description.vectorCType);
  };
  auto validateSourceVectorLoad =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          llvm::StringRef stepLabel, const support::RuntimeABIParameter &abi,
          llvm::StringRef resultName) -> llvm::Error {
    return validateRVVProviderBuiltRouteStep(
        step, consumerLabel, stepLabel, description.sourceVectorLoadIntrinsic,
        {{pointerAtInduction(abi), abi.cType}, {emitCLoopVLName, vlCType}},
        resultName, description.sourceVectorCType);
  };
  auto validateSplat =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          llvm::StringRef stepLabel, const support::RuntimeABIParameter &abi,
          llvm::StringRef resultName) -> llvm::Error {
    return validateRVVProviderBuiltRouteStep(
        step, consumerLabel, stepLabel, description.rhsBroadcastIntrinsic,
        {{abi.cName, abi.cType}, {emitCLoopVLName, vlCType}},
        resultName, description.vectorCType);
  };
  auto validateCompare =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          llvm::StringRef stepLabel, llvm::StringRef callee,
          llvm::StringRef lhsName, llvm::StringRef rhsName,
          llvm::StringRef resultName) -> llvm::Error {
    return validateRVVProviderBuiltRouteStep(
        step, consumerLabel, stepLabel, callee,
        {{lhsName, description.vectorCType},
         {rhsName, description.vectorCType},
         {emitCLoopVLName, vlCType}},
        resultName, description.maskCType);
  };
  auto validateSelect =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          llvm::StringRef falseOperand,
          llvm::StringRef trueOperand) -> llvm::Error {
    return validateRVVProviderBuiltRouteStep(
        step, consumerLabel, "select/merge", description.intrinsic,
        {{falseOperand, description.vectorCType},
         {trueOperand, description.vectorCType},
         {description.maskName, description.maskCType},
         {emitCLoopVLName, vlCType}},
        description.resultName, description.vectorCType);
  };
  auto validateUnitStore =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          llvm::StringRef stepLabel, const support::RuntimeABIParameter &abi,
          llvm::StringRef valueName) -> llvm::Error {
    return validateRVVProviderBuiltRouteStep(
        step, consumerLabel, stepLabel, description.storeIntrinsic,
        {{pointerAtInduction(abi), abi.cType},
         {valueName, description.vectorCType},
         {emitCLoopVLName, vlCType}});
  };
  auto validateMaskedLoad =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          llvm::StringRef stepLabel,
          llvm::ArrayRef<RVVExpectedRouteStepOperand> operands) -> llvm::Error {
    return validateRVVProviderBuiltRouteStep(
        step, consumerLabel, stepLabel, description.maskedLoadIntrinsic,
        operands, description.resultName, description.vectorCType);
  };

  using OperationKind = plugin::rvv::RVVSelectedBodyOperationKind;
  switch (description.operation) {
  case OperationKind::CmpSelect: {
    if (description.intrinsic.empty() || description.storeIntrinsic.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires provider-derived select and store leaves before "
          "validating cmp_select statements");
    const support::RuntimeABIParameter &lhsABI =
        description.runtimeABIParameters[0];
    const support::RuntimeABIParameter &rhsABI =
        description.runtimeABIParameters[1];
    const support::RuntimeABIParameter &outABI =
        description.runtimeABIParameters[2];
    if (llvm::Error error =
            validateVectorLoad(loop.bodySteps[1], "lhs vector load", lhsABI,
                               "lhs_vec"))
      return error;
    if (llvm::Error error =
            validateVectorLoad(loop.bodySteps[2], "rhs vector load", rhsABI,
                               "rhs_vec"))
      return error;
    if (llvm::Error error = validateCompare(
            loop.bodySteps[3], "compare", description.compareIntrinsic,
            "lhs_vec", "rhs_vec", description.maskName))
      return error;
    if (llvm::Error error =
            validateSelect(loop.bodySteps[4], "rhs_vec", "lhs_vec"))
      return error;
    return validateUnitStore(loop.bodySteps[5], "output store", outABI,
                             description.resultName);
  }
  case OperationKind::ComputedMaskSelect: {
    if (description.intrinsic.empty() || description.storeIntrinsic.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires provider-derived select and store leaves before "
          "validating computed_mask_select statements");
    const support::RuntimeABIParameter &cmpLhsABI =
        description.runtimeABIParameters[0];
    const support::RuntimeABIParameter &cmpRhsABI =
        description.runtimeABIParameters[1];
    const support::RuntimeABIParameter &trueValueABI =
        description.runtimeABIParameters[2];
    const support::RuntimeABIParameter &falseValueABI =
        description.runtimeABIParameters[3];
    const support::RuntimeABIParameter &outABI =
        description.runtimeABIParameters[4];
    if (llvm::Error error = validateVectorLoad(
            loop.bodySteps[1], "compare lhs vector load", cmpLhsABI, "lhs_vec"))
      return error;
    if (llvm::Error error = validateVectorLoad(
            loop.bodySteps[2], "compare rhs vector load", cmpRhsABI, "rhs_vec"))
      return error;
    if (llvm::Error error =
            validateVectorLoad(loop.bodySteps[3], "true-value vector load",
                               trueValueABI, "true_value_vec"))
      return error;
    if (llvm::Error error =
            validateVectorLoad(loop.bodySteps[4], "false-value vector load",
                               falseValueABI, "false_value_vec"))
      return error;
    if (llvm::Error error = validateCompare(
            loop.bodySteps[5], "compare", description.compareIntrinsic,
            "lhs_vec", "rhs_vec", description.maskName))
      return error;
    if (llvm::Error error = validateSelect(loop.bodySteps[6],
                                           "false_value_vec", "true_value_vec"))
      return error;
    return validateUnitStore(loop.bodySteps[7], "output store", outABI,
                             description.resultName);
  }
  case OperationKind::RuntimeScalarCompareSelect: {
    if (description.intrinsic.empty() || description.storeIntrinsic.empty() ||
        description.rhsBroadcastIntrinsic.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires provider-derived splat, select, and store leaves before "
          "validating runtime scalar compare/select statements");
    const support::RuntimeABIParameter &lhsABI =
        description.runtimeABIParameters[0];
    const support::RuntimeABIParameter &rhsScalarABI =
        description.runtimeABIParameters[1];
    const support::RuntimeABIParameter &trueValueABI =
        description.runtimeABIParameters[2];
    const support::RuntimeABIParameter &falseValueABI =
        description.runtimeABIParameters[3];
    const support::RuntimeABIParameter &outABI =
        description.runtimeABIParameters[4];
    if (llvm::Error error =
            validateVectorLoad(loop.bodySteps[1], "lhs vector load", lhsABI,
                               "lhs_vec"))
      return error;
    if (llvm::Error error =
            validateSplat(loop.bodySteps[2], "rhs scalar splat", rhsScalarABI,
                          "rhs_vec"))
      return error;
    if (llvm::Error error =
            validateVectorLoad(loop.bodySteps[3], "true-value vector load",
                               trueValueABI, "true_value_vec"))
      return error;
    if (llvm::Error error =
            validateVectorLoad(loop.bodySteps[4], "false-value vector load",
                               falseValueABI, "false_value_vec"))
      return error;
    if (llvm::Error error = validateCompare(
            loop.bodySteps[5], "compare", description.compareIntrinsic,
            "lhs_vec", "rhs_vec", description.maskName))
      return error;
    if (llvm::Error error = validateSelect(loop.bodySteps[6],
                                           "false_value_vec", "true_value_vec"))
      return error;
    return validateUnitStore(loop.bodySteps[7], "output store", outABI,
                             description.resultName);
  }
  case OperationKind::F32ClampSelect: {
    if (description.intrinsic.empty() || description.storeIntrinsic.empty() ||
        description.rhsBroadcastIntrinsic.empty() ||
        description.secondaryCompareIntrinsic.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires provider-derived splat, dual compare, select, and store "
          "leaves before validating f32 clamp/select statements");
    const support::RuntimeABIParameter &inputABI =
        description.runtimeABIParameters[0];
    const support::RuntimeABIParameter &lowerBoundABI =
        description.runtimeABIParameters[1];
    const support::RuntimeABIParameter &upperBoundABI =
        description.runtimeABIParameters[2];
    const support::RuntimeABIParameter &outABI =
        description.runtimeABIParameters[3];
    if (llvm::Error error =
            validateVectorLoad(loop.bodySteps[1], "input vector load", inputABI,
                               "lhs_vec"))
      return error;
    if (llvm::Error error =
            validateSplat(loop.bodySteps[2], "lower bound scalar splat",
                          lowerBoundABI, "lower_bound_vec"))
      return error;
    if (llvm::Error error =
            validateSplat(loop.bodySteps[3], "upper bound scalar splat",
                          upperBoundABI, "upper_bound_vec"))
      return error;
    if (llvm::Error error =
            validateCompare(loop.bodySteps[4], "lower-bound compare",
                            description.compareIntrinsic, "lhs_vec",
                            "lower_bound_vec", "lower_clamp_mask"))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[5], consumerLabel, "lower-bound select",
            description.intrinsic,
            {{"lhs_vec", description.vectorCType},
             {"lower_bound_vec", description.vectorCType},
             {"lower_clamp_mask", description.maskCType},
             {emitCLoopVLName, vlCType}},
            "lower_clamped_vec", description.vectorCType))
      return error;
    if (llvm::Error error =
            validateCompare(loop.bodySteps[6], "upper-bound compare",
                            description.secondaryCompareIntrinsic,
                            "upper_bound_vec", "lower_clamped_vec",
                            "upper_clamp_mask"))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[7], consumerLabel, "upper-bound select",
            description.intrinsic,
            {{"lower_clamped_vec", description.vectorCType},
             {"upper_bound_vec", description.vectorCType},
             {"upper_clamp_mask", description.maskCType},
             {emitCLoopVLName, vlCType}},
            description.resultName, description.vectorCType))
      return error;
    return validateUnitStore(loop.bodySteps[8], "output store", outABI,
                             description.resultName);
  }
  case OperationKind::DequantClampF32Epilogue: {
    if (description.intrinsic.empty() || description.storeIntrinsic.empty() ||
        description.rhsBroadcastIntrinsic.empty() ||
        description.secondaryCompareIntrinsic.empty() ||
        description.sourceVectorLoadIntrinsic.empty() ||
        description.sourceVectorCType.empty() ||
        description.dequantizeConvertIntrinsic.empty() ||
        description.dequantizeScaleIntrinsic.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires provider-derived source load, dequant convert/scale, "
          "splat, dual compare, select, and store leaves before validating "
          "dequant-clamp epilogue statements");
    const support::RuntimeABIParameter &sourceABI =
        description.runtimeABIParameters[0];
    const support::RuntimeABIParameter &scaleABI =
        description.runtimeABIParameters[1];
    const support::RuntimeABIParameter &lowerBoundABI =
        description.runtimeABIParameters[2];
    const support::RuntimeABIParameter &upperBoundABI =
        description.runtimeABIParameters[3];
    const support::RuntimeABIParameter &outABI =
        description.runtimeABIParameters[4];
    if (llvm::Error error =
            validateSourceVectorLoad(loop.bodySteps[1], "source i32 load",
                                     sourceABI, "source_i32_vec"))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[2], consumerLabel, "dequant convert",
            description.dequantizeConvertIntrinsic,
            {{"source_i32_vec", description.sourceVectorCType},
             {emitCLoopVLName, vlCType}},
            "converted_f32_vec", description.vectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[3], consumerLabel, "dequant scale",
            description.dequantizeScaleIntrinsic,
            {{"converted_f32_vec", description.vectorCType},
             {scaleABI.cName, scaleABI.cType},
             {emitCLoopVLName, vlCType}},
            "lhs_vec", description.vectorCType))
      return error;
    if (llvm::Error error =
            validateSplat(loop.bodySteps[4], "lower bound scalar splat",
                          lowerBoundABI, "lower_bound_vec"))
      return error;
    if (llvm::Error error =
            validateSplat(loop.bodySteps[5], "upper bound scalar splat",
                          upperBoundABI, "upper_bound_vec"))
      return error;
    if (llvm::Error error =
            validateCompare(loop.bodySteps[6], "lower-bound compare",
                            description.compareIntrinsic, "lhs_vec",
                            "lower_bound_vec", "lower_clamp_mask"))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[7], consumerLabel, "lower-bound select",
            description.intrinsic,
            {{"lhs_vec", description.vectorCType},
             {"lower_bound_vec", description.vectorCType},
             {"lower_clamp_mask", description.maskCType},
             {emitCLoopVLName, vlCType}},
            "lower_clamped_vec", description.vectorCType))
      return error;
    if (llvm::Error error =
            validateCompare(loop.bodySteps[8], "upper-bound compare",
                            description.secondaryCompareIntrinsic,
                            "upper_bound_vec", "lower_clamped_vec",
                            "upper_clamp_mask"))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[9], consumerLabel, "upper-bound select",
            description.intrinsic,
            {{"lower_clamped_vec", description.vectorCType},
             {"upper_bound_vec", description.vectorCType},
             {"upper_clamp_mask", description.maskCType},
             {emitCLoopVLName, vlCType}},
            description.resultName, description.vectorCType))
      return error;
    return validateUnitStore(loop.bodySteps[10], "output store", outABI,
                             description.resultName);
  }
  case OperationKind::RuntimeScalarDualCompareMaskAndSelect: {
    if (description.intrinsic.empty() || description.storeIntrinsic.empty() ||
        description.rhsBroadcastIntrinsic.empty() ||
        description.secondaryCompareIntrinsic.empty() ||
        description.maskAndIntrinsic.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires provider-derived splat, compare, mask-and, select, and "
          "store leaves before validating dual compare/select statements");
    const support::RuntimeABIParameter &cmpLhsAABI =
        description.runtimeABIParameters[0];
    const support::RuntimeABIParameter &rhsScalarAABI =
        description.runtimeABIParameters[1];
    const support::RuntimeABIParameter &cmpLhsBABI =
        description.runtimeABIParameters[2];
    const support::RuntimeABIParameter &rhsScalarBABI =
        description.runtimeABIParameters[3];
    const support::RuntimeABIParameter &trueValueABI =
        description.runtimeABIParameters[4];
    const support::RuntimeABIParameter &falseValueABI =
        description.runtimeABIParameters[5];
    const support::RuntimeABIParameter &outABI =
        description.runtimeABIParameters[6];
    if (llvm::Error error =
            validateVectorLoad(loop.bodySteps[1], "compare A lhs vector load",
                               cmpLhsAABI, "lhs_vec"))
      return error;
    if (llvm::Error error =
            validateSplat(loop.bodySteps[2], "compare A rhs scalar splat",
                          rhsScalarAABI, "rhs_vec"))
      return error;
    if (llvm::Error error =
            validateVectorLoad(loop.bodySteps[3], "compare B lhs vector load",
                               cmpLhsBABI, "cmp_lhs_b_vec"))
      return error;
    if (llvm::Error error =
            validateSplat(loop.bodySteps[4], "compare B rhs scalar splat",
                          rhsScalarBABI, "rhs_b_vec"))
      return error;
    if (llvm::Error error =
            validateVectorLoad(loop.bodySteps[5], "true-value vector load",
                               trueValueABI, "true_value_vec"))
      return error;
    if (llvm::Error error =
            validateVectorLoad(loop.bodySteps[6], "false-value vector load",
                               falseValueABI, "false_value_vec"))
      return error;
    if (llvm::Error error =
            validateCompare(loop.bodySteps[7], "compare A",
                            description.compareIntrinsic, "lhs_vec",
                            "rhs_vec", "mask_a"))
      return error;
    if (llvm::Error error =
            validateCompare(loop.bodySteps[8], "compare B",
                            description.secondaryCompareIntrinsic,
                            "cmp_lhs_b_vec", "rhs_b_vec", "mask_b"))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[9], consumerLabel, "mask composition",
            description.maskAndIntrinsic,
            {{"mask_a", description.maskCType},
             {"mask_b", description.maskCType},
             {emitCLoopVLName, vlCType}},
            description.maskName, description.maskCType))
      return error;
    if (llvm::Error error = validateSelect(loop.bodySteps[10],
                                           "false_value_vec", "true_value_vec"))
      return error;
    return validateUnitStore(loop.bodySteps[11], "output store", outABI,
                             description.resultName);
  }
  default:
    break;
  }

  const bool isRuntimeScalarStore =
      description.operation == OperationKind::RuntimeScalarComputedMaskStore;
  const bool isRuntimeScalarLoadStore =
      description.operation == OperationKind::RuntimeScalarComputedMaskLoadStore;
  const bool isRuntimeScalarIndexedGather =
      description.operation ==
      OperationKind::RuntimeScalarComputedMaskIndexedGatherLoadUnitStore;
  const bool isRuntimeScalarIndexedScatter =
      description.operation ==
      OperationKind::RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad;
  const bool isRuntimeScalarMemory =
      isRuntimeScalarStore || isRuntimeScalarLoadStore ||
      isRuntimeScalarIndexedGather || isRuntimeScalarIndexedScatter;
  const bool isStridedStore =
      description.operation == OperationKind::ComputedMaskStridedStore;
  const bool isStridedLoad =
      description.operation == OperationKind::ComputedMaskStridedLoadUnitStore;
  const bool isIndexedGather =
      description.operation ==
          OperationKind::ComputedMaskIndexedGatherLoadUnitStore ||
      isRuntimeScalarIndexedGather;
  const bool isIndexedScatter =
      description.operation ==
          OperationKind::ComputedMaskIndexedScatterStoreUnitLoad ||
      isRuntimeScalarIndexedScatter;
  const bool isIndexed = isIndexedGather || isIndexedScatter;
  const bool isLoadMerge =
      isRuntimeScalarLoadStore || isRuntimeScalarIndexedGather ||
      description.operation == OperationKind::ComputedMaskUnitLoadStore ||
      isStridedLoad || isIndexedGather;
  const bool isStoreOnly =
      isRuntimeScalarStore || isRuntimeScalarIndexedScatter ||
      isStridedStore || isIndexedScatter;

  if (!isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(
          description.operation))
    llvm_unreachable(
        "validated non compare/select mask operation as compare/select mask");
  if ((isRuntimeScalarMemory && description.rhsBroadcastIntrinsic.empty()) ||
      ((isRuntimeScalarStore || isLoadMerge) &&
       description.storeIntrinsic.empty()))
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived computed-mask memory splat/store leaves "
        "before validating route statements");
  if (isLoadMerge && description.maskedLoadIntrinsic.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived masked-load leaf before validating "
        "computed-mask memory statements");
  if (isStridedStore && description.stridedStoreIntrinsic.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived strided-store leaf before validating "
        "computed-mask memory statements");
  if (isIndexed && (description.indexLoadIntrinsic.empty() ||
                    description.indexScaleIntrinsic.empty() ||
                    description.indexVectorCType.empty()))
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived index load/scale leaves and index vector "
        "C type before validating indexed computed-mask memory statements");
  if (isIndexedScatter && description.indexedStoreIntrinsic.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived indexed-store leaf before validating "
        "indexed scatter statements");

  const support::RuntimeABIParameter &compareLhsABI =
      description.runtimeABIParameters[0];
  const support::RuntimeABIParameter *compareRhsABI =
      isRuntimeScalarMemory ? nullptr : &description.runtimeABIParameters[1];
  const support::RuntimeABIParameter *rhsScalarABI =
      isRuntimeScalarMemory ? &description.runtimeABIParameters[1] : nullptr;
  const support::RuntimeABIParameter &sourceABI =
      description.runtimeABIParameters[2];
  const support::RuntimeABIParameter *indexABI =
      isIndexed ? &description.runtimeABIParameters[3] : nullptr;
  const support::RuntimeABIParameter &destinationABI =
      isIndexed ? description.runtimeABIParameters[4]
                : description.runtimeABIParameters[3];
  const support::RuntimeABIParameter *sourceStrideABI =
      isStridedLoad ? &description.runtimeABIParameters[5] : nullptr;
  const support::RuntimeABIParameter *destinationStrideABI =
      isStridedStore ? &description.runtimeABIParameters[5] : nullptr;

  std::size_t stepIndex = 1;
  if (llvm::Error error =
          validateVectorLoad(loop.bodySteps[stepIndex++],
                             "compare lhs vector load", compareLhsABI,
                             "lhs_vec"))
    return error;
  if (isIndexed) {
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[stepIndex++], consumerLabel, "index vector load",
            description.indexLoadIntrinsic,
            {{pointerAtInduction(*indexABI), indexABI->cType},
             {emitCLoopVLName, vlCType}},
            "index_vec", description.indexVectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[stepIndex++], consumerLabel, "index scale",
            description.indexScaleIntrinsic,
            {{"index_vec", description.indexVectorCType},
             {"4", "uint32_t"},
             {emitCLoopVLName, vlCType}},
            "byte_offsets", description.indexVectorCType))
      return error;
  }
  if (isRuntimeScalarMemory) {
    if (llvm::Error error =
            validateSplat(loop.bodySteps[stepIndex++], "rhs scalar splat",
                          *rhsScalarABI, "rhs_vec"))
      return error;
  } else if (llvm::Error error =
                 validateVectorLoad(loop.bodySteps[stepIndex++],
                                    "compare rhs vector load", *compareRhsABI,
                                    "rhs_vec")) {
    return error;
  }
  if (isStoreOnly) {
    if (llvm::Error error =
            validateVectorLoad(loop.bodySteps[stepIndex++],
                               "source payload vector load", sourceABI,
                               "source_vec"))
      return error;
  } else {
    if (llvm::Error error =
            validateVectorLoad(loop.bodySteps[stepIndex++],
                               "old destination vector load", destinationABI,
                               "old_dst_vec"))
      return error;
  }
  if (llvm::Error error =
          validateCompare(loop.bodySteps[stepIndex++], "compare",
                          description.compareIntrinsic, "lhs_vec", "rhs_vec",
                          description.maskName))
    return error;

  if (isIndexedGather) {
    if (llvm::Error error = validateMaskedLoad(
            loop.bodySteps[stepIndex++], "masked indexed load",
            {{description.maskName, description.maskCType},
             {"old_dst_vec", description.vectorCType},
             {sourceABI.cName, sourceABI.cType},
             {"byte_offsets", description.indexVectorCType},
             {emitCLoopVLName, vlCType}}))
      return error;
    return validateUnitStore(loop.bodySteps[stepIndex++], "output store",
                             destinationABI, description.resultName);
  }
  if (isStridedLoad) {
    if (llvm::Error error = validateMaskedLoad(
            loop.bodySteps[stepIndex++], "masked strided load",
            {{description.maskName, description.maskCType},
             {"old_dst_vec", description.vectorCType},
             {stridedSourcePointerAtInduction(sourceABI, *sourceStrideABI),
              sourceABI.cType},
             {sourceStrideABI->cName, "ptrdiff_t"},
             {emitCLoopVLName, vlCType}}))
      return error;
    return validateUnitStore(loop.bodySteps[stepIndex++], "output store",
                             destinationABI, description.resultName);
  }
  if (isLoadMerge) {
    if (llvm::Error error = validateMaskedLoad(
            loop.bodySteps[stepIndex++], "masked unit load",
            {{description.maskName, description.maskCType},
             {"old_dst_vec", description.vectorCType},
             {pointerAtInduction(sourceABI), sourceABI.cType},
             {emitCLoopVLName, vlCType}}))
      return error;
    return validateUnitStore(loop.bodySteps[stepIndex++], "output store",
                             destinationABI, description.resultName);
  }
  if (isStridedStore) {
    return validateRVVProviderBuiltRouteStep(
        loop.bodySteps[stepIndex++], consumerLabel, "masked strided store",
        description.stridedStoreIntrinsic,
        {{description.maskName, description.maskCType},
         {stridedDestinationPointerAtInduction(destinationABI,
                                              *destinationStrideABI),
          destinationABI.cType},
         {destinationStrideABI->cName, "ptrdiff_t"},
         {"source_vec", description.vectorCType},
         {emitCLoopVLName, vlCType}});
  }
  if (isIndexedScatter) {
    return validateRVVProviderBuiltRouteStep(
        loop.bodySteps[stepIndex++], consumerLabel, "masked indexed store",
        description.indexedStoreIntrinsic,
        {{description.maskName, description.maskCType},
         {destinationABI.cName, destinationABI.cType},
         {"byte_offsets", description.indexVectorCType},
         {"source_vec", description.vectorCType},
         {emitCLoopVLName, vlCType}});
  }
  if (isRuntimeScalarStore) {
    return validateRVVProviderBuiltRouteStep(
        loop.bodySteps[stepIndex++], consumerLabel, "masked unit store",
        description.storeIntrinsic,
        {{description.maskName, description.maskCType},
         {pointerAtInduction(destinationABI), destinationABI.cType},
         {"source_vec", description.vectorCType},
         {emitCLoopVLName, vlCType}});
  }

  return llvm::Error::success();
}

llvm::Error validateRVVCompareSelectMaskRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (!isRVVCompareSelectMaskRouteFamilyOperation(description.operation))
    return llvm::Error::success();

  const bool isCompareSelectProducer =
      isRVVCompareSelectMaskProducerRouteFamilyOperation(
          description.operation);
  const bool isComputedMaskMemory =
      isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(
          description.operation);

  if (isCompareSelectProducer) {
    std::optional<plugin::rvv::RVVCompareSelectRouteValidationContract>
        contract =
            plugin::rvv::getRVVCompareSelectRouteValidationContract(
                description);
    if (!contract)
      return makeRVVTargetRouteError(
          "compare/select mask target artifact consumer requires "
          "provider-owned compare/select facts and route validation contract "
          "from typed body/config/runtime facts before artifact export");
    if (route.getRouteID() != contract->emitCRouteID)
      return makeRVVTargetRouteError(
          llvm::Twine(contract->consumerLabel) +
          " requires rebuilt provider route id '" + contract->emitCRouteID +
          "' but route carried '" + route.getRouteID() + "'");
    if (llvm::Error error =
            validateRVVCompareSelectRouteValidationContract(description,
                                                            *contract))
      return error;
    if (llvm::Error error =
            validateRVVCompareSelectMaskRouteHeaders(route, description))
      return error;
    if (llvm::Error error =
            validateRVVCompareSelectMaskRouteTypeMappings(route, description))
      return error;
    if (llvm::Error error =
            validateRVVCompareSelectMaskRouteABIMappings(route, description))
      return error;
    return validateRVVCompareSelectMaskRouteStatementPlan(
        route, description, contract->expectedPreLoopStepCount,
        contract->expectedLoopBodyStepCount, &contract->runtimeAVLVLContract);
  }

  if (isRVVCompareSelectMaskIndexedMemoryOperation(description.operation)) {
    std::optional<
        plugin::rvv::RVVComputedMaskIndexedMemoryRouteValidationContract>
        contract =
            plugin::rvv::
                getRVVComputedMaskIndexedMemoryRouteValidationContract(
                    description);
    if (!contract)
      return makeRVVTargetRouteError(
          "computed-mask indexed memory target artifact consumer requires "
          "provider-owned route validation contract from typed "
          "body/config/runtime facts before artifact export");
    if (route.getRouteID() != contract->emitCRouteID)
      return makeRVVTargetRouteError(
          llvm::Twine(contract->consumerLabel) +
          " requires rebuilt provider route id '" + contract->emitCRouteID +
          "' but route carried '" + route.getRouteID() + "'");
    if (llvm::Error error =
            validateRVVComputedMaskIndexedMemoryDescriptionAgainstContract(
                description, *contract))
      return error;
    if (llvm::Error error =
            validateRVVComputedMaskIndexedMemoryRouteHeaders(route, *contract))
      return error;
    if (llvm::Error error =
            validateRVVComputedMaskIndexedMemoryRouteTypeMappings(route,
                                                                  *contract))
      return error;
    if (llvm::Error error =
            validateRVVComputedMaskIndexedMemoryRouteABIMappings(route,
                                                                 *contract))
      return error;
    if (llvm::Error error =
            validateRVVComputedMaskIndexedMemoryRouteStatementPlanShape(
                route, *contract))
      return error;
    return validateRVVCompareSelectMaskRouteStatementPlan(
        route, description, contract->expectedPreLoopStepCount,
        contract->expectedLoopBodyStepCount, &contract->runtimeAVLVLContract);
  }

  if (isRVVCompareSelectMaskStridedMemoryOperation(description.operation)) {
    std::optional<
        plugin::rvv::RVVComputedMaskStridedMemoryRouteValidationContract>
        contract =
            plugin::rvv::
                getRVVComputedMaskStridedMemoryRouteValidationContract(
                    description);
    if (!contract)
      return makeRVVTargetRouteError(
          "computed-mask strided memory target artifact consumer requires "
          "provider-owned route validation contract from typed "
          "body/config/runtime facts before artifact export");
    if (route.getRouteID() != contract->emitCRouteID)
      return makeRVVTargetRouteError(
          llvm::Twine(contract->consumerLabel) +
          " requires rebuilt provider route id '" + contract->emitCRouteID +
          "' but route carried '" + route.getRouteID() + "'");
    if (llvm::Error error =
            validateRVVComputedMaskStridedMemoryDescriptionAgainstContract(
                description, *contract))
      return error;
    if (llvm::Error error =
            validateRVVComputedMaskStridedMemoryRouteHeaders(route, *contract))
      return error;
    if (llvm::Error error =
            validateRVVComputedMaskStridedMemoryRouteTypeMappings(route,
                                                                  *contract))
      return error;
    if (llvm::Error error =
            validateRVVComputedMaskStridedMemoryRouteABIMappings(route,
                                                                 *contract))
      return error;
    if (llvm::Error error =
            validateRVVComputedMaskStridedMemoryRouteStatementPlanShape(
                route, *contract))
      return error;
    return validateRVVCompareSelectMaskRouteStatementPlan(
        route, description, contract->expectedPreLoopStepCount,
        contract->expectedLoopBodyStepCount, &contract->runtimeAVLVLContract);
  }

  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("compare/select mask target artifact consumer requires "
                    "rebuilt provider route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");
  if (description.providerSupportedMirror.empty())
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires a "
        "provider-supported mirror label after route construction");
  if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
          "provider_supported_mirror", description.providerSupportedMirror,
          getRVVCompareSelectMaskExpectedProviderSupportedMirror(
              description.operation)))
    return error;
  if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
          "target_leaf_profile", description.targetLeafProfile,
          getRVVCompareSelectMaskExpectedTargetLeafProfile(
              description.operation)))
    return error;
  if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
          "required header declarations",
          description.requiredHeaderDeclarations,
          getRVVCompareSelectMaskExpectedRequiredHeaderDeclarations(
              description.operation)))
    return error;
  if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
          "C type mapping summary", description.cTypeMappingSummary,
          getRVVCompareSelectMaskExpectedCTypeMappingSummary(
              description.operation)))
    return error;
  if (llvm::Error error =
          validateRVVCompareSelectMaskDualProviderFacts(description))
    return error;
  if (description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires provider route "
        "operand binding facts before artifact export");
  if (llvm::Error error =
          validateRVVComputedMaskIndexedMemoryCanonicalProviderFacts(
              description))
    return error;
  if (llvm::Error error =
          validateRVVComputedMaskStridedMemoryCanonicalProviderFacts(
              description))
    return error;
  if (llvm::Error error =
          validateRVVUnitStrideMaskedMemoryCanonicalProviderFacts(description))
    return error;
  if (description.runtimeControlPlanID.empty() ||
      description.runtimeABIOrder.empty() || description.setVLIntrinsic.empty())
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires "
        "provider-derived runtime AVL/VL facts before artifact export");
  if (llvm::Error error =
          validateRVVCompareSelectMaskRuntimeABIFacts(description))
    return error;
  if (llvm::Error error =
          validateComputedMaskIndexedGatherHeaderBindingSummary(description))
    return error;
  if (llvm::Error error =
          validateComputedMaskIndexedScatterHeaderBindingSummary(description))
    return error;
  if (description.comparePredicateKind.empty() ||
      description.compareIntrinsic.empty() ||
      (isCompareSelectProducer && description.intrinsic.empty()) ||
      description.maskName.empty() || description.maskRole.empty() ||
      description.maskSource.empty() || description.maskMemoryForm.empty())
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires "
      "provider-derived compare predicate, compare/select or masked-memory "
      "callee, mask result, mask role, mask source, and mask memory-form "
      "facts before artifact export");
  if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
          "source memory form", description.sourceMemoryForm,
          getRVVCompareSelectMaskExpectedSourceMemoryForm(
              description.operation)))
    return error;
  if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
          "destination memory form", description.destinationMemoryForm,
          getRVVCompareSelectMaskExpectedDestinationMemoryForm(
              description.operation)))
    return error;
  if (isComputedMaskMemory && description.maskedLoadIntrinsic.empty() &&
      description.storeIntrinsic.empty() &&
      description.stridedStoreIntrinsic.empty() &&
      description.indexedStoreIntrinsic.empty())
    return makeRVVTargetRouteError(
        "compare-produced computed-mask memory target artifact consumer "
        "requires provider-derived masked load or store callee facts before "
        "artifact export");

  if (isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(
          description.operation)) {
    if (description.computedMaskMemoryRouteFamilyPlanID.empty() ||
        description.computedMaskMemoryMaskProducerSource.empty() ||
        description.inactiveLaneContract.empty() ||
        description.maskedPassthroughLayout.empty() ||
        description.indexedMemoryLayout.empty())
      return makeRVVTargetRouteError(
          "compare-produced computed-mask memory target artifact consumer "
          "requires provider-derived memory plan, mask producer, inactive-lane, "
          "passthrough, and masked-memory layout facts before artifact export");
    if (!description.plainCompareSelectRouteFamilyPlanID.empty() ||
        !description.computedMaskSelectRouteFamilyPlanID.empty() ||
        !description.computedMaskSelectMaskProducerSource.empty())
      return makeRVVTargetRouteError(
          "compare-produced computed-mask memory target artifact consumer "
          "rejects stale compare/select route-family facts");
    if (!description.baseMemoryMovementRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          "compare-produced computed-mask memory target artifact consumer "
          "rejects stale plain base-memory route-family facts");
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "computed-mask memory route-family plan",
            description.computedMaskMemoryRouteFamilyPlanID,
            "rvv-computed-mask-memory-route-family-plan.v1"))
      return error;
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "computed-mask memory producer source",
            description.computedMaskMemoryMaskProducerSource,
            getRVVCompareSelectMaskExpectedMaskProducerSource(
                description.operation)))
      return error;
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "mask role", description.maskRole,
            "predicate-mask-produced-by-compare"))
      return error;
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "mask source", description.maskSource,
            "compare-produced-mask-same-vl-scope"))
      return error;
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "mask memory form", description.maskMemoryForm,
            "compare-produced-mask"))
      return error;
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "inactive-lane contract", description.inactiveLaneContract,
            getRVVCompareSelectMaskExpectedInactiveLaneContract(
                description.operation)))
      return error;
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "masked passthrough layout", description.maskedPassthroughLayout,
            getRVVCompareSelectMaskExpectedPassthroughLayout(
                description.operation)))
      return error;
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "masked memory layout", description.indexedMemoryLayout,
            getRVVCompareSelectMaskExpectedMaskedMemoryLayout(
                description.operation)))
      return error;
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "strided memory layout", description.stridedMemoryLayout,
            getRVVCompareSelectMaskExpectedStridedMemoryLayout(
                description.operation)))
      return error;
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "source stride source", description.sourceStrideSource,
            getRVVCompareSelectMaskExpectedSourceStrideSource(
                description.operation)))
      return error;
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "destination stride source", description.outStrideSource,
            getRVVCompareSelectMaskExpectedDestinationStrideSource(
                description.operation)))
      return error;
    if (description.indexEEW != 0)
      return makeRVVTargetRouteError(
          llvm::Twine("compare-produced computed-mask memory target artifact "
                      "consumer requires provider-derived index EEW '") +
          llvm::Twine(0) + "' but was '" +
          llvm::Twine(description.indexEEW) + "'");
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "index uniqueness", description.indexUniqueness, ""))
      return error;
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "indexed data memory form", description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  }

  if (isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(
          description.operation)) {
    llvm::StringRef expectedOwner(kRVVComputedMaskMemoryMaskTailPolicyOwner);
    llvm::StringRef expectedPlan(kRVVMaskTailPolicyRouteFamilyPlanID);
    if (description.maskTailPolicyRouteFamilyPlanID != expectedPlan ||
        description.maskTailPolicyOwner != expectedOwner)
      return makeRVVTargetRouteError(
          llvm::Twine("compare/select mask target artifact consumer requires "
                      "provider-derived mask/tail policy owner '") +
          expectedOwner + "' and route-family plan '" +
          expectedPlan + "' before artifact export");
  } else if (!description.maskTailPolicyRouteFamilyPlanID.empty() ||
             !description.maskTailPolicyOwner.empty()) {
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer rejects mask/tail policy "
        "mirrors for routes that are not computed-mask select or "
        "compare-produced computed-mask memory");
  }

  if (isRVVUnitStrideMaskedMemoryRouteFamilyOperation(
          description.operation)) {
    std::optional<plugin::rvv::RVVUnitStrideMaskedMemoryRouteValidationContract>
        contract =
            plugin::rvv::getRVVUnitStrideMaskedMemoryRouteValidationContract(
                description);
    if (!contract)
      return makeRVVTargetRouteError(
          "unit-stride masked memory target artifact consumer requires "
          "provider-owned route validation contract from typed "
          "body/config/runtime facts before validating route statements");
    if (llvm::Error error =
            validateRVVUnitStrideMaskedMemoryRouteHeaders(route, *contract))
      return error;
    if (llvm::Error error =
            validateRVVUnitStrideMaskedMemoryRouteTypeMappings(route,
                                                               *contract))
      return error;
    if (llvm::Error error =
            validateRVVUnitStrideMaskedMemoryRouteABIMappings(route,
                                                              *contract))
      return error;
    return validateRVVCompareSelectMaskRouteStatementPlan(
        route, description, contract->expectedPreLoopStepCount,
        contract->expectedLoopBodyStepCount,
        &contract->runtimeAVLVLContract);
  }
  if (llvm::Error error =
          validateRVVCompareSelectMaskRouteHeaders(route, description))
    return error;
  if (llvm::Error error =
          validateRVVCompareSelectMaskRouteTypeMappings(route, description))
    return error;
  if (llvm::Error error =
          validateRVVCompareSelectMaskRouteABIMappings(route, description))
    return error;
  return validateRVVCompareSelectMaskRouteStatementPlan(route, description);
}

llvm::Error validateRVVCompareSelectMaskTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVCompareSelectMaskRoutePayloadFacts(context.route,
                                                       context.description);
}

llvm::Error validateRVVCompareSelectMaskTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description =
      context.description;

  if (isRVVCompareSelectMaskProducerRouteFamilyOperation(
          description.operation)) {
    std::optional<
        plugin::rvv::RVVCompareSelectRouteMetadataMirrorContractSet>
        contract =
            plugin::rvv::getRVVCompareSelectRouteMetadataMirrorContract(
                description);
    if (!contract)
      return makeRVVTargetRouteError(
          "compare/select mask target artifact consumer requires "
          "provider-owned compare/select metadata mirror contract before "
          "validating candidate mirrors");
    return validateRVVProviderCompareSelectRouteMetadataMirrorContract(
        candidate, *contract);
  }

  if (isRVVCompareSelectMaskIndexedMemoryOperation(description.operation)) {
    std::optional<plugin::rvv::RVVMemoryRouteMetadataMirrorContractSet>
        contract =
            plugin::rvv::
                getRVVComputedMaskIndexedMemoryRouteMetadataMirrorContract(
                    description);
    if (!contract)
      return makeRVVTargetRouteError(
          "computed-mask indexed memory target artifact consumer requires "
          "provider-owned metadata mirror contract before validating "
          "candidate mirrors");
    return validateRVVProviderMemoryRouteMetadataMirrorContract(candidate,
                                                               *contract);
  }

  if (isRVVCompareSelectMaskStridedMemoryOperation(description.operation)) {
    std::optional<plugin::rvv::RVVMemoryRouteMetadataMirrorContractSet>
        contract =
            plugin::rvv::
                getRVVComputedMaskStridedMemoryRouteMetadataMirrorContract(
                    description);
    if (!contract)
      return makeRVVTargetRouteError(
          "computed-mask strided memory target artifact consumer requires "
          "provider-owned metadata mirror contract before validating "
          "candidate mirrors");
    return validateRVVProviderMemoryRouteMetadataMirrorContract(candidate,
                                                               *contract);
  }

  if (isRVVUnitStrideMaskedMemoryRouteFamilyOperation(
          description.operation)) {
    if (llvm::Error error =
            validateRVVUnitStrideMaskedMemoryCanonicalProviderFacts(
                description))
      return error;
    std::optional<plugin::rvv::RVVMemoryRouteMetadataMirrorContractSet>
        contract =
            plugin::rvv::
                getRVVUnitStrideMaskedMemoryRouteMetadataMirrorContract(
                    description);
    if (!contract)
      return makeRVVTargetRouteError(
          "unit-stride masked memory target artifact consumer requires "
          "provider-owned metadata mirror contract before validating "
          "candidate mirrors");
    return validateRVVProviderMemoryRouteMetadataMirrorContract(candidate,
                                                               *contract);
  }

  if (llvm::Error error =
          validateRVVComputedMaskIndexedMemoryCanonicalProviderFacts(
              description))
    return error;
  if (llvm::Error error =
          validateRVVComputedMaskStridedMemoryCanonicalProviderFacts(
              description))
    return error;
  if (llvm::Error error =
          validateRVVUnitStrideMaskedMemoryCanonicalProviderFacts(description))
    return error;

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.compare_predicate_kind",
          description.comparePredicateKind,
          "selected typed RVV compare/select mask predicate"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_plan",
          description.routeOperandBindingPlanID,
          "selected typed RVV compare/select mask binding plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_operands",
          description.routeOperandBindingSummary,
          "selected typed RVV compare/select mask binding summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.provider_supported_mirror",
          description.providerSupportedMirror,
          "selected typed RVV compare/select mask provider support"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.target_leaf_profile",
          description.targetLeafProfile,
          "selected typed RVV compare/select mask target leaf profile"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "rvv_selected_body_typed_compute_op",
          description.typedComputeOpName,
          "selected typed RVV compare/select mask typed compute op"))
    return error;

  if (isRVVPlainCompareSelectMaskRouteFamilyOperation(description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.plain_compare_select_route_family_plan",
            description.plainCompareSelectRouteFamilyPlanID,
            "selected typed RVV plain compare-select route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_select_route_family_plan", "",
            "selected typed RVV computed-mask select route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_memory_route_family_plan", "",
            "selected typed RVV computed-mask memory route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_tail_policy_route_family_plan", "",
            "selected typed RVV mask/tail policy route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_tail_policy_owner", "",
            "selected typed RVV mask/tail policy owner"))
      return error;
  } else if (isRVVComputedMaskSelectRouteFamilyOperation(
                 description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_select_route_family_plan",
            description.computedMaskSelectRouteFamilyPlanID,
            "selected typed RVV computed-mask select route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_select_mask_producer_source",
            description.computedMaskSelectMaskProducerSource,
            "selected typed RVV computed-mask select producer source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.plain_compare_select_route_family_plan", "",
            "selected typed RVV plain compare-select route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_memory_route_family_plan", "",
            "selected typed RVV computed-mask memory route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_tail_policy_route_family_plan",
            description.maskTailPolicyRouteFamilyPlanID,
            "selected typed RVV mask/tail policy route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_tail_policy_owner",
            description.maskTailPolicyOwner,
            "selected typed RVV mask/tail policy owner"))
      return error;
  } else if (isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(
                 description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_memory_route_family_plan",
            description.computedMaskMemoryRouteFamilyPlanID,
            "selected typed RVV computed-mask memory route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_memory_mask_producer_source",
            description.computedMaskMemoryMaskProducerSource,
            "selected typed RVV computed-mask memory producer source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.plain_compare_select_route_family_plan", "",
            "selected typed RVV plain compare-select route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_select_route_family_plan", "",
            "selected typed RVV computed-mask select route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.base_memory_movement_route_family_plan", "",
            "selected typed RVV plain base-memory route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_tail_policy_route_family_plan",
            description.maskTailPolicyRouteFamilyPlanID,
            "selected typed RVV mask/tail policy route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_tail_policy_owner",
            description.maskTailPolicyOwner,
            "selected typed RVV mask/tail policy owner"))
      return error;
  }

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.memory_form",
          plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
              description.memoryForm),
          "selected typed RVV compare/select mask memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_control_plan",
          description.runtimeControlPlanID,
          "route-local runtime AVL/VL control plan mirror"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order", description.runtimeABIOrder,
          "route-local runtime AVL/VL ABI order mirror"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.tail_policy", description.tailPolicy,
          "selected typed RVV compare/select mask tail policy"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.mask_policy", description.maskPolicy,
          "selected typed RVV compare/select mask policy"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.required_header_declarations",
          description.requiredHeaderDeclarations,
          "selected typed RVV compare/select mask route header requirements"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.c_type_mapping",
          description.cTypeMappingSummary,
          "selected typed RVV compare/select mask route type mapping summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.mask_role", description.maskRole,
          "selected typed RVV compare/select mask role"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.mask_source", description.maskSource,
          "selected typed RVV compare/select mask source"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.mask_memory_form", description.maskMemoryForm,
          "selected typed RVV compare/select mask memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_memory_form",
          description.sourceMemoryForm,
          "selected typed RVV compare/select mask source memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.destination_memory_form",
          description.destinationMemoryForm,
          "selected typed RVV compare/select mask destination memory form"))
    return error;
  if (isRVVCompareSelectMaskProducerRouteFamilyOperation(
          description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.select_layout", description.selectLayout,
            "selected typed RVV compare/select selected-value layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.lower_bound_role",
            description.lowerBoundRole,
            "selected typed RVV compare/select lower bound role"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.upper_bound_role",
            description.upperBoundRole,
            "selected typed RVV compare/select upper bound role"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.lower_bound_c_type",
            description.lowerBoundCType,
            "selected typed RVV compare/select lower bound C type"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.upper_bound_c_type",
            description.upperBoundCType,
            "selected typed RVV compare/select upper bound C type"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.bound_order", description.boundOrder,
            "selected typed RVV compare/select bound order"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.clamp_relation",
            description.clampRelation,
            "selected typed RVV compare/select clamp relation"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.secondary_compare_predicate_kind",
            description.secondaryComparePredicateKind,
            "selected typed RVV dual compare secondary predicate"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_composition",
            description.maskComposition,
            "selected typed RVV dual compare mask composition"))
      return error;
  }
  if (isRVVPlainCompareSelectMaskRouteFamilyOperation(description.operation) ||
      isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(
          description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.inactive_lane_contract",
            description.inactiveLaneContract,
            "selected typed RVV compare/select mask inactive lane contract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.masked_passthrough_layout",
            description.maskedPassthroughLayout,
            "selected typed RVV compare/select mask passthrough layout"))
      return error;
  }
  if (isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(
          description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.masked_memory_layout",
            description.indexedMemoryLayout,
            "selected typed RVV computed-mask memory layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.strided_memory_layout",
            description.stridedMemoryLayout,
            "selected typed RVV computed-mask strided memory layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.source_stride_source",
            description.sourceStrideSource,
            "selected typed RVV computed-mask source stride binding"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.destination_stride_source",
            description.outStrideSource,
            "selected typed RVV computed-mask destination stride binding"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.indexed_memory_layout",
            isRVVCompareSelectMaskIndexedMemoryOperation(
                description.operation)
                ? description.indexedMemoryLayout
                : llvm::StringRef(),
            "selected typed RVV computed-mask indexed memory layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.index_source", description.indexSource,
            "selected typed RVV computed-mask index source"))
      return error;
    const std::string indexEEWMirror =
        isRVVCompareSelectMaskIndexedMemoryOperation(description.operation)
            ? llvm::Twine(description.indexEEW).str()
            : std::string();
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.index_eew", indexEEWMirror,
            "selected typed RVV computed-mask index EEW"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.offset_unit", description.offsetUnit,
            "selected typed RVV computed-mask index offset unit"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.index_uniqueness",
            description.indexUniqueness,
            "selected typed RVV computed-mask index uniqueness"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.indexed_data_memory_form",
            description.indexedDataMemoryForm,
            "selected typed RVV computed-mask indexed data memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.indexed_destination_memory_form",
            description.indexedDestinationMemoryForm,
            "selected typed RVV computed-mask indexed destination memory form"))
      return error;
  }

  return llvm::Error::success();
}

llvm::Error validateRVVConversionDtypePolicyRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVConversionDtypePolicyRouteValidationContract
        &contract) {
  if (contract.requiredHeaderDeclarations.empty() ||
      contract.requiredHeaders.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived required_header_declarations before "
        "accepting the route artifact");

  for (llvm::StringRef header : contract.requiredHeaders) {
    if (header.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " saw an empty provider route header declaration");
    if (!routeHasHeader(route, header))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route header '" + header +
          "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVConversionDtypePolicyRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVConversionDtypePolicyRouteValidationContract
        &contract) {
  if (contract.cTypeMappingSummary.empty() || contract.typeMappings.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived route type mapping facts before artifact "
        "export");

  for (const plugin::rvv::RVVConversionDtypePolicyRouteTypeMappingContract
           &mapping : contract.typeMappings) {
    if (mapping.sourceType.empty() || mapping.cType.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-derived " + mapping.label +
          " facts before artifact export");
    if (!routeHasTypeMapping(route, mapping.sourceType, mapping.cType))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route type mapping '" +
          mapping.sourceType + "' -> '" + mapping.cType + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVConversionDtypePolicyRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVConversionDtypePolicyRouteValidationContract
        &contract) {
  if (contract.runtimeABIOrder.empty() || contract.runtimeABIParameters.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived runtime ABI order and ABI parameters "
        "before artifact export");
  if (route.getABIMappings().size() != contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires rebuilt provider route ABI mapping count " +
        llvm::Twine(contract.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        contract.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

std::string formatRVVRuntimeABIParameterNames(
    llvm::ArrayRef<support::RuntimeABIParameter> parameters) {
  std::string names;
  for (const auto &indexed : llvm::enumerate(parameters)) {
    if (indexed.index() != 0)
      names += ", ";
    names += indexed.value().cName;
  }
  return names;
}

llvm::Error validateRVVConversionDtypePolicyRuntimeABIFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVConversionDtypePolicyRouteValidationContract
        &contract) {
  if (description.runtimeABIParameters.size() !=
      contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires "
        "provider-derived runtime ABI parameters for " +
        formatRVVRuntimeABIParameterNames(contract.runtimeABIParameters) +
        " before artifact export; route description carried " +
        llvm::Twine(description.runtimeABIParameters.size()) +
        " parameter(s)");

  for (size_t index = 0, count = contract.runtimeABIParameters.size();
       index < count; ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    const support::RuntimeABIParameter &expectedParameter =
        contract.runtimeABIParameters[index];
    if (actual.cName != expectedParameter.cName ||
        actual.cType != expectedParameter.cType ||
        actual.role != expectedParameter.role ||
        actual.ownership != expectedParameter.ownership)
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-derived runtime ABI parameter " +
          std::to_string(index) + " to bind " + expectedParameter.cName +
          " as " +
          support::stringifyRuntimeABIParameterRole(expectedParameter.role) +
          " with C type '" + expectedParameter.cType +
          "' before artifact export");
  }
  return llvm::Error::success();
}

llvm::Error validateRVVConversionDtypePolicyTypedFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVConversionDtypePolicyRouteValidationContract
        &contract) {
  if (description.operation != contract.operation)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned operation '" +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            contract.operation) +
        "' but was '" +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            description.operation) +
        "'");
  if (description.emitCRouteID != contract.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned route id '" + contract.emitCRouteID +
        "' but description carried '" + description.emitCRouteID + "'");
  const bool isWidening =
      contract.kind ==
      plugin::rvv::RVVConversionDtypePolicyRouteValidationKind::
          WideningConversion;
  const bool isDequantization =
      contract.kind ==
      plugin::rvv::RVVConversionDtypePolicyRouteValidationKind::
          Dequantization;
  if (!isWidening && !isDequantization)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned conversion dtype-policy contract kind before "
        "artifact export");
  if (description.memoryForm != contract.memoryForm)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned memory form '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(contract.memoryForm) +
        "' but was '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
            description.memoryForm) +
        "'");
  if (isWidening) {
    if (description.wideningConversionRouteFamilyPlanID !=
        contract.wideningConversionRouteFamilyPlanID)
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-owned widening conversion route-family plan '" +
          contract.wideningConversionRouteFamilyPlanID + "' but was '" +
          description.wideningConversionRouteFamilyPlanID + "'");
  } else {
    if (description.dequantizationRouteFamilyPlanID !=
        contract.dequantizationRouteFamilyPlanID)
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-owned dequantization route-family plan '" +
          contract.dequantizationRouteFamilyPlanID + "' but was '" +
          description.dequantizationRouteFamilyPlanID + "'");
  }
  if (description.typedComputeOpName != contract.typedComputeOpName)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires selected typed " + contract.typedComputeOpName +
        " body before artifact export");
  if (description.conversionKind != contract.conversionKind)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned conversion kind '" +
        contract.conversionKind + "' but was '" + description.conversionKind +
        "'");

  if (description.sourceElementTypeName != contract.sourceElementTypeName ||
      description.resultElementTypeName != contract.resultElementTypeName ||
      description.sourceSEW != contract.sourceSEW ||
      description.sourceLMUL != contract.sourceLMUL ||
      description.sourceVectorTypeName != contract.sourceVectorTypeName ||
      description.sourceVectorCType != contract.sourceVectorCType ||
      description.sew != contract.resultSEW ||
      description.lmul != contract.resultLMUL ||
      description.vectorTypeName != contract.resultVectorTypeName ||
      description.vectorCType != contract.resultVectorCType)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived source/result dtype policy for '" +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            contract.operation) +
        "' but saw source element '" + description.sourceElementTypeName +
        "', result element '" + description.resultElementTypeName +
        "', source SEW " + llvm::Twine(description.sourceSEW) +
        ", source LMUL '" + description.sourceLMUL + "', source type '" +
        description.sourceVectorTypeName + "'/'" +
        description.sourceVectorCType + "', result SEW " +
        llvm::Twine(description.sew) + ", result LMUL '" + description.lmul +
        "', result type '" + description.vectorTypeName + "'/'" +
        description.vectorCType + "'");
  if (isWidening &&
      description.conversionRelation != contract.conversionRelation)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived widening conversion relation '" +
        contract.conversionRelation + "' but saw '" +
        description.conversionRelation + "'");
  if (isDequantization &&
      (description.dequantizationRelation != contract.dequantizationRelation ||
       description.dequantizeConvertIntrinsic !=
           contract.dequantizeConvertIntrinsic ||
       description.dequantizeScaleIntrinsic !=
           contract.dequantizeScaleIntrinsic ||
       description.dequantScaleRole != contract.dequantScaleRole ||
       description.dequantScaleCType != contract.dequantScaleCType ||
       description.dequantScaleName != contract.dequantScaleName))
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived dequantization relation, convert/scale "
        "intrinsics, and runtime scale role/type/name facts before artifact "
        "export");
  if (isDequantization &&
      (description.gearboxCandidateSet != contract.gearboxCandidateSet ||
       description.gearboxSelectedCandidate !=
           contract.gearboxSelectedCandidate ||
       description.gearboxSelectionReason !=
           contract.gearboxSelectionReason ||
       description.gearboxLegalityScope != contract.gearboxLegalityScope ||
       description.gearboxScheduleID != contract.gearboxScheduleID ||
       description.gearboxSelector != contract.gearboxSelector ||
       description.gearboxSource != contract.gearboxSource ||
       description.gearboxOperation != contract.gearboxOperation ||
       description.gearboxUnroll != contract.gearboxUnroll ||
       description.gearboxVLPolicy != contract.gearboxVLPolicy ||
       description.gearboxSourceSEW != contract.gearboxSourceSEW ||
       description.gearboxSourceLMUL != contract.gearboxSourceLMUL ||
       description.gearboxDestSEW != contract.gearboxDestSEW ||
       description.gearboxDestLMUL != contract.gearboxDestLMUL ||
       description.gearboxRuntimeAVLSource !=
           contract.gearboxRuntimeAVLSource ||
       description.gearboxProducerScope != contract.gearboxProducerScope ||
       description.gearboxConsumerScope != contract.gearboxConsumerScope ||
       description.gearboxProducerScope == description.gearboxConsumerScope))
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-consumed RVV Gearbox schedule facts before "
        "artifact export");
  if (description.tailPolicy != contract.tailPolicy ||
      description.maskPolicy != contract.maskPolicy ||
      description.sourceMemoryForm != contract.sourceMemoryForm ||
      description.destinationMemoryForm != contract.destinationMemoryForm)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned policy and memory-form facts for '" +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            contract.operation) +
        "' but saw tail '" + description.tailPolicy + "', mask '" +
        description.maskPolicy + "', source memory form '" +
        description.sourceMemoryForm + "', and destination memory form '" +
        description.destinationMemoryForm + "'");

  if (description.cTypeMappingSummary != contract.cTypeMappingSummary ||
      description.providerSupportedMirror != contract.providerSupportedMirror ||
      description.targetLeafProfile != contract.targetLeafProfile ||
      description.requiredHeaderDeclarations !=
          contract.requiredHeaderDeclarations ||
      description.routeOperandBindingPlanID !=
          contract.routeOperandBindingPlanID ||
      description.routeOperandBindingSummary !=
          contract.routeOperandBindingSummary)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned support, header, C type, binding, and "
        "target leaf facts for '" +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            contract.operation) +
        "' but saw support '" + description.providerSupportedMirror +
        "', header declarations '" + description.requiredHeaderDeclarations +
        "', C type mapping '" + description.cTypeMappingSummary +
        "', binding plan '" + description.routeOperandBindingPlanID +
        "', binding summary '" + description.routeOperandBindingSummary +
        "', and target leaf profile '" + description.targetLeafProfile + "'");

  const bool intrinsicMatches =
      isWidening ? description.intrinsic == contract.conversionIntrinsic
                 : description.dequantizeConvertIntrinsic ==
                       contract.dequantizeConvertIntrinsic;
  if (description.sourceVectorLoadIntrinsic !=
          contract.sourceVectorLoadIntrinsic ||
      !intrinsicMatches || description.storeIntrinsic != contract.storeIntrinsic ||
      description.resultName != contract.resultName)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned source-load, conversion, store, and "
        "result-name facts for '" +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            contract.operation) +
        "' before artifact export");

  return llvm::Error::success();
}

llvm::Error validateRVVConversionDtypePolicyRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVConversionDtypePolicyRouteValidationContract
        &contract) {
  const plugin::rvv::RVVRuntimeAVLVLSelectedBoundaryContract &runtimeContract =
      contract.runtimeAVLVLContract;
  const bool isDequantization =
      contract.kind ==
      plugin::rvv::RVVConversionDtypePolicyRouteValidationKind::
          Dequantization;
  const std::size_t expectedRuntimeABIParameterCount =
      isDequantization ? 4 : 3;
  if (contract.runtimeABIParameters.size() != expectedRuntimeABIParameterCount)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived conversion/dequantization ABI parameters before "
        "validating route statements");
  if (contract.resultName.empty() || contract.sourceVectorCType.empty() ||
      contract.vectorCType.empty() || runtimeContract.vlCType.empty() ||
      runtimeContract.setVLIntrinsic.empty() ||
      runtimeContract.emitCFullChunkVLName.empty() ||
      runtimeContract.emitCLoopVLName.empty() ||
      runtimeContract.emitCLoopInductionName.empty() ||
      runtimeContract.runtimeAVLParameter.cName.empty() ||
      runtimeContract.runtimeAVLParameter.cType.empty() ||
      contract.sourceVectorLoadIntrinsic.empty() ||
      contract.intrinsic.empty() || contract.storeIntrinsic.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived source/result vector C types, result "
        "name, VL C type, and conversion callees before validating route "
        "statements");
  if (contract.expectedPreLoopStepCount == 0 ||
      contract.expectedLoopBodyStepCount == 0)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned conversion statement-plan expectations "
        "before artifact export");

  const support::RuntimeABIParameter &sourceABI =
      contract.runtimeABIParameters[0];
  const support::RuntimeABIParameter *scaleABI =
      isDequantization ? &contract.runtimeABIParameters[1] : nullptr;
  const support::RuntimeABIParameter &outABI =
      contract.runtimeABIParameters[isDequantization ? 2 : 1];
  const support::RuntimeABIParameter &runtimeNABI =
      runtimeContract.runtimeAVLParameter;
  const support::RuntimeABIParameter &orderedRuntimeNABI =
      contract.runtimeABIParameters[isDequantization ? 3 : 2];
  if (!runtimeABIParameterEquals(orderedRuntimeNABI, runtimeNABI))
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires runtime n/AVL ABI role to match provider runtime AVL/VL "
        "selected-boundary contract and selected conversion ABI order before "
        "validating route statements");

  if (route.getCallOpaqueSteps().size() != contract.expectedPreLoopStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires exact provider-built pre-loop statement count " +
        llvm::Twine(contract.expectedPreLoopStepCount) +
        " before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          preLoopSetVL, contract.consumerLabel, "pre-loop setvl",
          runtimeContract.setVLIntrinsic,
          {{runtimeNABI.cName, runtimeNABI.cType}},
          runtimeContract.emitCFullChunkVLName, runtimeContract.vlCType))
    return error;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires pre-loop statements to carry selected typed RVV source "
          "provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires exactly one provider-built runtime AVL/VL loop before "
        "artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  const std::string expectedLoopStep =
      contract.gearboxLoopStepExpression.empty()
          ? runtimeContract.emitCFullChunkVLName
          : contract.gearboxLoopStepExpression;
  if (loop.inductionVarName != runtimeContract.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != runtimeContract.vlCType ||
      loop.step.expression != expectedLoopStep ||
      loop.step.cType != runtimeContract.vlCType)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-built loop bounds and step to mirror runtime "
        "AVL/VL facts");
  if (loop.upperBound.expression != runtimeNABI.cName ||
      loop.upperBound.cType != runtimeNABI.cType)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-built loop upper bound to use the runtime n/AVL ABI "
        "parameter");
  if (loop.bodySteps.size() != contract.expectedLoopBodyStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires exact provider-built loop statement count " +
        llvm::Twine(contract.expectedLoopBodyStepCount) +
        " before artifact export");

  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeNABI.cName) + " - " +
       runtimeContract.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[0], contract.consumerLabel, "loop setvl",
          runtimeContract.setVLIntrinsic,
          {{expectedRemainingAVL, runtimeContract.vlCType}},
          runtimeContract.emitCLoopVLName, runtimeContract.vlCType))
    return error;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires loop statements to carry selected typed RVV source "
          "provenance");

  const std::string expectedSourcePointer =
      (llvm::StringRef(sourceABI.cName) + " + " +
       runtimeContract.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[1], contract.consumerLabel, "source vector load",
          contract.sourceVectorLoadIntrinsic,
          {{expectedSourcePointer, sourceABI.cType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
          "lhs_vec", contract.sourceVectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[2], contract.consumerLabel, "widening conversion",
          isDequantization ? contract.dequantizeConvertIntrinsic
                           : contract.intrinsic,
          {{"lhs_vec", contract.sourceVectorCType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
          isDequantization ? "converted_f32_vec" : contract.resultName,
          contract.vectorCType))
    return error;

  const std::size_t storeStepIndex = isDequantization ? 4 : 3;
  if (isDequantization) {
    if (!scaleABI || contract.dequantizeScaleIntrinsic.empty() ||
        contract.dequantScaleRole.empty() || contract.dequantScaleCType.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-derived runtime scale ABI and scale intrinsic "
          "facts before validating dequantization route statements");
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[3], contract.consumerLabel, "dequantization scale",
            contract.dequantizeScaleIntrinsic,
            {{"converted_f32_vec", contract.vectorCType},
             {scaleABI->cName, scaleABI->cType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            contract.resultName, contract.vectorCType))
      return error;
  }

  const std::string expectedOutPointer =
      (llvm::StringRef(outABI.cName) + " + " +
       runtimeContract.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[storeStepIndex], contract.consumerLabel,
          "output store",
          contract.storeIntrinsic,
          {{expectedOutPointer, outABI.cType},
           {contract.resultName, contract.vectorCType},
           {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}}))
    return error;

  const bool expectsGearboxTwoSlice =
      isDequantization && contract.gearboxUnroll == 2;
  if (!expectsGearboxTwoSlice)
    return llvm::Error::success();

  if (contract.gearboxSecondRemainingAVLExpression.empty() ||
      contract.gearboxSecondLoopVLName.empty() ||
      contract.gearboxSecondSourcePointerExpression.empty() ||
      contract.gearboxSecondOutPointerExpression.empty() ||
      contract.gearboxSecondSourceName.empty() ||
      contract.gearboxSecondConvertedName.empty() ||
      contract.gearboxSecondResultName.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived Gearbox u2 second-slice route-plan facts "
        "before validating route statements");

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[5], contract.consumerLabel,
          "Gearbox u2 second-slice setvl", runtimeContract.setVLIntrinsic,
          {{contract.gearboxSecondRemainingAVLExpression,
            runtimeContract.vlCType}},
          contract.gearboxSecondLoopVLName, runtimeContract.vlCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[6], contract.consumerLabel,
          "Gearbox u2 second-slice source vector load",
          contract.sourceVectorLoadIntrinsic,
          {{contract.gearboxSecondSourcePointerExpression, sourceABI.cType},
           {contract.gearboxSecondLoopVLName, runtimeContract.vlCType}},
          contract.gearboxSecondSourceName, contract.sourceVectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[7], contract.consumerLabel,
          "Gearbox u2 second-slice dequant convert",
          contract.dequantizeConvertIntrinsic,
          {{contract.gearboxSecondSourceName, contract.sourceVectorCType},
           {contract.gearboxSecondLoopVLName, runtimeContract.vlCType}},
          contract.gearboxSecondConvertedName, contract.vectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[8], contract.consumerLabel,
          "Gearbox u2 second-slice dequant scale",
          contract.dequantizeScaleIntrinsic,
          {{contract.gearboxSecondConvertedName, contract.vectorCType},
           {scaleABI->cName, scaleABI->cType},
           {contract.gearboxSecondLoopVLName, runtimeContract.vlCType}},
          contract.gearboxSecondResultName, contract.vectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[9], contract.consumerLabel,
          "Gearbox u2 second-slice output store", contract.storeIntrinsic,
          {{contract.gearboxSecondOutPointerExpression, outABI.cType},
           {contract.gearboxSecondResultName, contract.vectorCType},
           {contract.gearboxSecondLoopVLName, runtimeContract.vlCType}}))
    return error;

  return llvm::Error::success();
}

llvm::Error validateRVVConversionDtypePolicyRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVConversionDtypePolicyRouteValidationContract
        &contract) {
  if (route.getRouteID() != contract.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires rebuilt provider route id '" + contract.emitCRouteID +
        "' but route carried '" + route.getRouteID() + "'");
  if (contract.providerSupportedMirror.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires a provider-supported mirror label after route construction");
  const bool isWidening =
      contract.kind ==
      plugin::rvv::RVVConversionDtypePolicyRouteValidationKind::
          WideningConversion;
  const bool isDequantization =
      contract.kind ==
      plugin::rvv::RVVConversionDtypePolicyRouteValidationKind::
          Dequantization;
  if (!isWidening && !isDequantization)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires a provider-owned conversion dtype-policy kind before "
        "artifact export");
  if (contract.routeOperandBindingPlanID.empty() ||
      contract.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider route operand binding facts before artifact export");
  if (contract.configContractID.empty() || contract.resultSEW == 0 ||
      contract.resultLMUL.empty() || contract.tailPolicy.empty() ||
      contract.maskPolicy.empty() || contract.intrinsic.empty() ||
      contract.storeIntrinsic.empty() ||
      contract.resultName.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived dtype, policy, intrinsic, and "
        "result facts before artifact export");

  const plugin::rvv::RVVSelectedBodyMemoryForm expectedMemoryForm =
      isDequantization
          ? plugin::rvv::RVVSelectedBodyMemoryForm::UnitStrideDequantization
          : plugin::rvv::RVVSelectedBodyMemoryForm::UnitStrideConversion;
  if (contract.memoryForm != expectedMemoryForm)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires unit-stride conversion/dequantization memory form from the "
        "selected typed RVV body");
  if (description.configContractID != contract.configContractID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned config contract '" +
        contract.configContractID + "' but description carried '" +
        description.configContractID + "'");
  if (llvm::Error error = validateRVVRuntimeAVLVLSelectedBoundaryContract(
          description, contract.runtimeAVLVLContract))
    return error;
  if (llvm::Error error = validateRVVRouteLocalRuntimeAVLVLMirrors(
          contract.consumerLabel, contract.runtimeAVLVLContract,
          contract.runtimeControlPlanID, contract.runtimeABIOrder,
          contract.setVLIntrinsic, contract.vlCType,
          contract.emitCFullChunkVLName, contract.emitCLoopVLName,
          contract.emitCLoopInductionName))
    return error;
  if (llvm::Error error =
          validateRVVConversionDtypePolicyRuntimeABIFacts(description,
                                                          contract))
    return error;

  if (contract.sourceElementTypeName.empty() ||
      contract.sourceSEW == 0 || contract.sourceLMUL.empty() ||
      contract.sourceVectorTypeName.empty() ||
      contract.sourceVectorCType.empty() ||
      contract.resultElementTypeName.empty() ||
      contract.conversionKind.empty() ||
      contract.sourceVectorLoadIntrinsic.empty() ||
      contract.sourceMemoryForm.empty() ||
      contract.destinationMemoryForm.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived source dtype, source vector type, "
        "source load, result dtype, conversion kind, memory forms, "
        "and route-family plan facts before artifact export");
  if (isWidening && (contract.wideningConversionRouteFamilyPlanID.empty() ||
                     contract.conversionRelation.empty()))
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived widening conversion route-family plan and "
        "conversion relation before artifact export");
  if (isDequantization &&
      (contract.dequantizationRouteFamilyPlanID.empty() ||
       contract.dequantizationRelation.empty() ||
       contract.dequantizeConvertIntrinsic.empty() ||
       contract.dequantizeScaleIntrinsic.empty() ||
       contract.dequantScaleRole.empty() || contract.dequantScaleCType.empty() ||
       contract.dequantScaleName.empty()))
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived dequantization plan, relation, "
        "convert/scale intrinsics, and runtime scale facts before artifact "
        "export");
  if (isWidening && contract.sourceSEW >= contract.resultSEW)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires source SEW to be smaller than result SEW");
  if (llvm::Error error =
          validateRVVConversionDtypePolicyTypedFacts(description, contract))
    return error;
  if (!description.scalarBroadcastElementwiseRouteFamilyPlanID.empty() ||
      !description.elementwiseArithmeticRouteFamilyPlanID.empty() ||
      !description.rhsBroadcastIntrinsic.empty() ||
      !description.runtimeScalarSplatStoreRouteFamilyPlanID.empty() ||
      !description.plainMAccRouteFamilyPlanID.empty() ||
      !description.scalarBroadcastMAccRouteFamilyPlanID.empty() ||
      !description.accumulationRouteFamilyPlanID.empty() ||
      !description.wideningMAccRelation.empty() ||
      !description.wideningDotProductRelation.empty() ||
      !description.plainCompareSelectRouteFamilyPlanID.empty() ||
      !description.computedMaskSelectRouteFamilyPlanID.empty() ||
      !description.computedMaskMemoryRouteFamilyPlanID.empty() ||
      !description.segment2MemoryRouteFamilyPlanID.empty() ||
      !description.standaloneReductionRouteFamilyPlanID.empty() ||
      !description.contractionRouteFamilyPlanID.empty() ||
      !description.baseMemoryMovementRouteFamilyPlanID.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " rejects stale non-conversion route-family facts");
  if (isWidening &&
      (!description.dequantizationRouteFamilyPlanID.empty() ||
       !description.dequantizationRelation.empty() ||
       !description.dequantizeConvertIntrinsic.empty() ||
       !description.dequantizeScaleIntrinsic.empty() ||
       !description.dequantScaleRole.empty() ||
       !description.dequantScaleCType.empty() ||
       !description.dequantScaleName.empty()))
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " rejects stale dequantization route-family facts on widening "
        "conversion routes");
  if (isDequantization &&
      (!description.wideningConversionRouteFamilyPlanID.empty() ||
       !description.conversionRelation.empty()))
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " rejects stale widening conversion facts on dequantization routes");

  if (llvm::Error error =
          validateRVVConversionDtypePolicyRouteHeaders(route, contract))
    return error;
  if (llvm::Error error =
          validateRVVConversionDtypePolicyRouteTypeMappings(route, contract))
    return error;
  if (llvm::Error error =
          validateRVVConversionDtypePolicyRouteABIMappings(route, contract))
    return error;
  return validateRVVConversionDtypePolicyRouteStatementPlan(route, contract);
}

llvm::Error validateRVVConversionDtypePolicyTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  std::optional<plugin::rvv::RVVConversionDtypePolicyRouteValidationContract>
      contract = plugin::rvv::getRVVConversionDtypePolicyRouteValidationContract(
          context.description);
  if (!contract)
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-owned conversion dtype-policy route validation contract "
        "before validating provider facts");
  return validateRVVConversionDtypePolicyRoutePayloadFacts(
      context.route, context.description, *contract);
}

llvm::Error validateRVVConversionDtypePolicyTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  std::optional<
      plugin::rvv::RVVConversionDtypePolicyRouteMetadataMirrorContractSet>
      contract =
          plugin::rvv::getRVVConversionDtypePolicyRouteMetadataMirrorContract(
              context.description);
  if (!contract)
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-owned conversion dtype-policy metadata mirror contract "
        "before validating candidate mirrors");
  return validateRVVProviderConversionDtypePolicyRouteMetadataMirrorContract(
      context.candidate, *contract);
}

llvm::Error validateRVVSegment2MemoryRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSegment2MemoryRouteValidationContract &contract) {
  if (contract.requiredHeaderDeclarations.empty() ||
      contract.requiredHeaders.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned required_header_declarations before "
        "accepting the route artifact");

  for (llvm::StringRef header : contract.requiredHeaders) {
    if (header.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " saw an empty provider route header declaration");
    if (!routeHasHeader(route, header))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route header '" + header +
          "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVSegment2MemoryRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSegment2MemoryRouteValidationContract &contract) {
  if (contract.vlCType.empty() || contract.vectorTypeName.empty() ||
      contract.vectorCType.empty() || contract.cTypeMappingSummary.empty() ||
      contract.typeMappings.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived VL, vector, and C type mapping facts "
        "before artifact export");

  for (const plugin::rvv::RVVSegment2MemoryRouteTypeMappingContract &mapping :
       contract.typeMappings) {
    if (mapping.sourceType.empty() || mapping.cType.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires non-empty provider type mapping contract entries");
    if (!routeHasTypeMapping(route, mapping.sourceType, mapping.cType))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route type mapping '" +
          mapping.sourceType + "' -> '" + mapping.cType + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVSegment2MemoryRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSegment2MemoryRouteValidationContract &contract) {
  if (contract.runtimeABIParameterRoles.size() !=
      contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned runtime ABI role order with one role per "
        "ABI parameter before artifact export");
  if (contract.runtimeABIOrder.empty() ||
      contract.runtimeABIParameters.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived runtime ABI order and ABI parameters "
        "before artifact export");
  if (route.getABIMappings().size() != contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires rebuilt provider route ABI mapping count " +
        llvm::Twine(contract.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        contract.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error requireRVVSegment2MemoryProviderField(
    llvm::StringRef label, llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  if (expected.empty())
    return makeRVVTargetRouteError(
        llvm::Twine("segment2-memory target artifact consumer rejects stale "
                    "provider-derived ") +
        label + " fact '" + actual + "'");
  if (actual.empty())
    return makeRVVTargetRouteError(
        llvm::Twine("segment2-memory target artifact consumer requires "
                    "provider-derived ") +
        label + " '" + expected + "' before artifact export");
  return makeRVVTargetRouteError(
      llvm::Twine("segment2-memory target artifact consumer requires "
                  "provider-derived ") +
      label + " '" + expected + "' but was '" + actual + "'");
}

llvm::Error requireRVVRuntimeAVLVLSelectedBoundaryField(
    llvm::StringRef consumerLabel, llvm::StringRef label,
    llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  if (expected.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " rejects stale runtime AVL/VL selected-boundary " + label +
        " fact '" + actual + "'");
  if (actual.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-owned runtime AVL/VL selected-boundary " + label +
        " '" + expected + "' before artifact export");
  return makeRVVTargetRouteError(
      llvm::Twine(consumerLabel) +
      " requires provider-owned runtime AVL/VL selected-boundary " + label +
      " '" + expected + "' but was '" + actual + "'");
}

llvm::Error validateRVVRuntimeAVLVLSelectedBoundaryContract(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVRuntimeAVLVLSelectedBoundaryContract &contract) {
  const llvm::StringRef consumerLabel = contract.consumerLabel.empty()
                                            ? "RVV target artifact consumer"
                                            : contract.consumerLabel;
  if (contract.sew == 0 || contract.lmul.empty() ||
      contract.tailPolicy.empty() || contract.maskPolicy.empty() ||
      contract.configContractID.empty() ||
      contract.runtimeControlPlanID.empty() ||
      contract.runtimeVLContractID.empty() ||
      contract.runtimeAVLABIParameterName.empty() ||
      contract.runtimeAVLASource.empty() || contract.runtimeABIOrder.empty() ||
      contract.selectedBoundaryOpName.empty() ||
      contract.selectedBodyProvenance.empty() || contract.vlDefOpName.empty() ||
      contract.vlScopeOpName.empty() || contract.vlUses.empty() ||
      contract.setVLIntrinsic.empty() || contract.vlCType.empty() ||
      contract.emitCLoopKind.empty() ||
      contract.emitCLoopInductionName.empty() ||
      contract.emitCFullChunkVLName.empty() ||
      contract.emitCLoopVLName.empty() ||
      contract.remainingAVLMetadata.empty() ||
      contract.pointerAdvanceMetadata.empty() ||
      contract.boundedSlice.empty() || contract.multiVL.empty() ||
      contract.runtimeAVLParameter.cName.empty() ||
      contract.runtimeAVLParameter.cType.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires complete provider-owned runtime AVL/VL "
        "selected-boundary contract before artifact export");

  if (description.sew != contract.sew)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-owned runtime AVL/VL selected-boundary SEW '" +
        llvm::Twine(contract.sew) + "' but was '" +
        llvm::Twine(description.sew) + "'");

  if (llvm::Error error = requireRVVRuntimeAVLVLSelectedBoundaryField(
          consumerLabel, "LMUL", description.lmul, contract.lmul))
    return error;
  if (llvm::Error error = requireRVVRuntimeAVLVLSelectedBoundaryField(
          consumerLabel, "tail policy", description.tailPolicy,
          contract.tailPolicy))
    return error;
  if (llvm::Error error = requireRVVRuntimeAVLVLSelectedBoundaryField(
          consumerLabel, "mask policy", description.maskPolicy,
          contract.maskPolicy))
    return error;
  if (llvm::Error error = requireRVVRuntimeAVLVLSelectedBoundaryField(
          consumerLabel, "config contract", description.configContractID,
          contract.configContractID))
    return error;
  if (llvm::Error error = requireRVVRuntimeAVLVLSelectedBoundaryField(
          consumerLabel, "runtime control plan",
          description.runtimeControlPlanID, contract.runtimeControlPlanID))
    return error;
  if (llvm::Error error = requireRVVRuntimeAVLVLSelectedBoundaryField(
          consumerLabel, "runtime VL contract",
          description.runtimeVLContractID, contract.runtimeVLContractID))
    return error;
  if (llvm::Error error = requireRVVRuntimeAVLVLSelectedBoundaryField(
          consumerLabel, "runtime AVL source", description.runtimeAVLASource,
          contract.runtimeAVLASource))
    return error;
  if (llvm::Error error = requireRVVRuntimeAVLVLSelectedBoundaryField(
          consumerLabel, "runtime ABI order", description.runtimeABIOrder,
          contract.runtimeABIOrder))
    return error;
  if (llvm::Error error = requireRVVRuntimeAVLVLSelectedBoundaryField(
          consumerLabel, "selected boundary op", description.boundaryOpName,
          contract.selectedBoundaryOpName))
    return error;
  if (llvm::Error error = requireRVVRuntimeAVLVLSelectedBoundaryField(
          consumerLabel, "VL def op", description.vlDefOpName,
          contract.vlDefOpName))
    return error;
  if (llvm::Error error = requireRVVRuntimeAVLVLSelectedBoundaryField(
          consumerLabel, "VL scope op", description.vlScopeOpName,
          contract.vlScopeOpName))
    return error;
  if (llvm::Error error = requireRVVRuntimeAVLVLSelectedBoundaryField(
          consumerLabel, "VL uses", description.vlUses, contract.vlUses))
    return error;
  if (llvm::Error error = requireRVVRuntimeAVLVLSelectedBoundaryField(
          consumerLabel, "setvl callee", description.setVLIntrinsic,
          contract.setVLIntrinsic))
    return error;
  if (llvm::Error error = requireRVVRuntimeAVLVLSelectedBoundaryField(
          consumerLabel, "VL C type", description.vlCType, contract.vlCType))
    return error;
  if (llvm::Error error = requireRVVRuntimeAVLVLSelectedBoundaryField(
          consumerLabel, "EmitC loop kind", description.emitCLoopKind,
          contract.emitCLoopKind))
    return error;
  if (llvm::Error error = requireRVVRuntimeAVLVLSelectedBoundaryField(
          consumerLabel, "EmitC loop induction",
          description.emitCLoopInductionName,
          contract.emitCLoopInductionName))
    return error;
  if (llvm::Error error = requireRVVRuntimeAVLVLSelectedBoundaryField(
          consumerLabel, "EmitC full-chunk VL",
          description.emitCFullChunkVLName,
          contract.emitCFullChunkVLName))
    return error;
  if (llvm::Error error = requireRVVRuntimeAVLVLSelectedBoundaryField(
          consumerLabel, "EmitC loop VL", description.emitCLoopVLName,
          contract.emitCLoopVLName))
    return error;
  if (llvm::Error error = requireRVVRuntimeAVLVLSelectedBoundaryField(
          consumerLabel, "remaining AVL metadata",
          description.remainingAVLMetadata, contract.remainingAVLMetadata))
    return error;
  if (llvm::Error error = requireRVVRuntimeAVLVLSelectedBoundaryField(
          consumerLabel, "pointer advancement metadata",
          description.pointerAdvanceMetadata,
          contract.pointerAdvanceMetadata))
    return error;
  if (llvm::Error error = requireRVVRuntimeAVLVLSelectedBoundaryField(
          consumerLabel, "bounded slice", description.boundedSlice,
          contract.boundedSlice))
    return error;
  if (llvm::Error error = requireRVVRuntimeAVLVLSelectedBoundaryField(
          consumerLabel, "multi-VL support", description.multiVL,
          contract.multiVL))
    return error;

  const support::RuntimeABIParameter *runtimeAVLParameter = nullptr;
  for (const support::RuntimeABIParameter &parameter :
       description.runtimeABIParameters) {
    if (parameter.role != support::RuntimeABIParameterRole::RuntimeElementCount)
      continue;
    if (runtimeAVLParameter)
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires exactly one provider-owned runtime n/AVL ABI parameter "
          "before artifact export");
    runtimeAVLParameter = &parameter;
  }
  if (!runtimeAVLParameter)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires a provider-owned runtime n/AVL ABI parameter before "
        "artifact export");
  if (!runtimeABIParameterEquals(*runtimeAVLParameter,
                                 contract.runtimeAVLParameter))
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires runtime n/AVL ABI parameter '" +
        contract.runtimeAVLParameter.cName + "' as " +
        support::stringifyRuntimeABIParameterRole(
            contract.runtimeAVLParameter.role) +
        " with C type '" + contract.runtimeAVLParameter.cType +
        "' before artifact export");
  if (contract.runtimeAVLParameter.cName !=
      contract.runtimeAVLABIParameterName)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires runtime AVL ABI parameter name '" +
        contract.runtimeAVLABIParameterName +
        "' to match the provider-owned runtime AVL parameter");

  return llvm::Error::success();
}

llvm::Error validateRVVSegment2MemoryProviderFactsFromContract(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVSegment2MemoryRouteValidationContract &contract) {
  if (description.operation != contract.operation)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned operation '" +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            contract.operation) +
        "' before artifact export");

  if (description.memoryForm != contract.memoryForm)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires selected typed RVV memory form '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(contract.memoryForm) +
        "' before artifact export but saw '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
            description.memoryForm) +
        "'");
  if (description.sew != contract.sew)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived SEW '" + llvm::Twine(contract.sew) +
        "' but was '" + llvm::Twine(description.sew) + "'");
  if (description.elementTypeName != contract.elementTypeName)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived element type '" +
        contract.elementTypeName + "' but was '" +
        description.elementTypeName + "'");
  if (description.configContractID != contract.configContractID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned config contract '" +
        contract.configContractID + "' but description carried '" +
        description.configContractID + "'");
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "LMUL", description.lmul, contract.lmul))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "tail policy", description.tailPolicy, contract.tailPolicy))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "mask policy", description.maskPolicy, contract.maskPolicy))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "typed compute op", description.typedComputeOpName,
          contract.typedComputeOpName))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "route operand binding plan", description.routeOperandBindingPlanID,
          contract.routeOperandBindingPlanID))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "route operand binding summary",
          description.routeOperandBindingSummary,
          contract.routeOperandBindingSummary))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "target leaf profile", description.targetLeafProfile,
          contract.targetLeafProfile))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "provider-supported mirror", description.providerSupportedMirror,
          contract.providerSupportedMirror))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "required header declarations",
          description.requiredHeaderDeclarations,
          contract.requiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "C type mapping", description.cTypeMappingSummary,
          contract.cTypeMappingSummary))
    return error;

  if (contract.usesPlainSegment2) {
    const bool expectsDeinterleave =
        contract.operation == plugin::rvv::RVVSelectedBodyOperationKind::
                                  Segment2DeinterleaveUnitStore;
    const bool expectsInterleave =
        contract.operation == plugin::rvv::RVVSelectedBodyOperationKind::
                                  Segment2InterleaveUnitLoad;
    if (contract.usesDeinterleaveLoad != expectsDeinterleave ||
        contract.usesInterleaveStore != expectsInterleave ||
        contract.usesDeinterleaveLoad == contract.usesInterleaveStore)
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-owned segment2 direction facts to match the "
          "selected interleave/deinterleave route before artifact export");
  }

  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "plain segment2 route-family plan",
          description.segment2MemoryRouteFamilyPlanID,
          contract.segment2MemoryRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "computed-mask route-family plan",
          description.computedMaskMemoryRouteFamilyPlanID,
          contract.computedMaskMemoryRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "computed-mask producer source",
          description.computedMaskMemoryMaskProducerSource,
          contract.computedMaskMemoryMaskProducerSource))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "mask/tail route-family plan",
          description.maskTailPolicyRouteFamilyPlanID,
          contract.maskTailPolicyRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "mask/tail route-family owner", description.maskTailPolicyOwner,
          contract.maskTailPolicyOwner))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "compare predicate", description.comparePredicateKind,
          contract.comparePredicateKind))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "mask role", description.maskRole, contract.maskRole))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "mask source", description.maskSource, contract.maskSource))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "mask memory form", description.maskMemoryForm,
          contract.maskMemoryForm))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "inactive lane contract", description.inactiveLaneContract,
          contract.inactiveLaneContract))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "masked passthrough layout", description.maskedPassthroughLayout,
          contract.maskedPassthroughLayout))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "compare callee", description.compareIntrinsic,
          contract.compareIntrinsic))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "mask result", description.maskName, contract.maskName))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "mask type", description.maskTypeName, contract.maskTypeName))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "mask C type", description.maskCType, contract.maskCType))
    return error;

  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "vector type", description.vectorTypeName, contract.vectorTypeName))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "vector C type", description.vectorCType, contract.vectorCType))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "vector load callee", description.vectorLoadIntrinsic,
          contract.vectorLoadIntrinsic))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "store callee", description.storeIntrinsic,
          contract.storeIntrinsic))
    return error;

  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "segment memory layout", description.segmentMemoryLayout,
          contract.segmentMemoryLayout))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "source memory form", description.sourceMemoryForm,
          contract.sourceMemoryForm))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "destination memory form", description.destinationMemoryForm,
          contract.destinationMemoryForm))
    return error;
  if (description.segmentCount != contract.segmentCount)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived segment-count facts '" +
        llvm::Twine(contract.segmentCount) + "' but was '" +
        llvm::Twine(description.segmentCount) + "'");
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "segment tuple C type", description.segmentTupleCType,
          contract.segmentTupleCType))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "segment load callee", description.segmentLoadIntrinsic,
          contract.segmentLoadIntrinsic))
    return error;

  llvm::StringRef actualSegmentStoreIntrinsic =
      contract.usesComputedMaskLoad ? llvm::StringRef()
                                    : description.segmentStoreIntrinsic;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "segment store callee", actualSegmentStoreIntrinsic,
          contract.usesComputedMaskLoad ? llvm::StringRef()
                                        : contract.segmentStoreIntrinsic))
    return error;
  llvm::StringRef actualTupleCreateIntrinsic =
      contract.usesComputedMaskLoad ? description.segmentStoreIntrinsic
                                    : description.segmentFieldExtractIntrinsic;
  if (contract.usesPlainSegment2 && contract.usesDeinterleaveLoad)
    actualTupleCreateIntrinsic = llvm::StringRef();
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "segment tuple create callee", actualTupleCreateIntrinsic,
          contract.segmentTupleCreateIntrinsic))
    return error;
  llvm::StringRef actualRHSScalarSplatIntrinsic =
      (contract.kind == plugin::rvv::RVVSegment2MemoryRouteValidationKind::
                            RuntimeScalarComputedMaskLoadUnitStore ||
       contract.kind == plugin::rvv::RVVSegment2MemoryRouteValidationKind::
                            RuntimeScalarComputedMaskStoreUnitLoad)
          ? description.rhsBroadcastIntrinsic
          : llvm::StringRef();
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "runtime scalar splat callee", actualRHSScalarSplatIntrinsic,
          contract.rhsScalarSplatIntrinsic))
    return error;
  llvm::StringRef actualFieldExtractIntrinsic =
      (contract.usesDeinterleaveLoad || contract.usesComputedMaskLoad)
          ? description.segmentFieldExtractIntrinsic
          : llvm::StringRef();
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "segment field extract callee", actualFieldExtractIntrinsic,
          contract.segmentFieldExtractIntrinsic))
    return error;

  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field0 role", description.field0Role, contract.field0Role))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field1 role", description.field1Role, contract.field1Role))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field0 name", description.field0Name, contract.field0Name))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field1 name", description.field1Name, contract.field1Name))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field0 source memory form", description.field0SourceMemoryForm,
          contract.field0SourceMemoryForm))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field1 source memory form", description.field1SourceMemoryForm,
          contract.field1SourceMemoryForm))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field0 destination memory form",
          description.field0DestinationMemoryForm,
          contract.field0DestinationMemoryForm))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field1 destination memory form",
          description.field1DestinationMemoryForm,
          contract.field1DestinationMemoryForm))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "segment2 update arithmetic kind",
          description.segment2UpdateArithmeticKind,
          contract.segment2UpdateArithmeticKind))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "segment2 update arithmetic callee",
          description.segment2UpdateArithmeticIntrinsic,
          contract.segment2UpdateArithmeticIntrinsic))
    return error;

  return llvm::Error::success();
}

llvm::Error validateRVVSegment2MemoryRuntimeABIFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVSegment2MemoryRouteValidationContract &contract) {
  if (description.runtimeABIParameters.size() !=
      contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("segment2-memory target artifact consumer requires ") +
        llvm::Twine(contract.runtimeABIParameters.size()) +
        " provider-derived runtime ABI parameters before artifact export but "
        "saw " +
        llvm::Twine(description.runtimeABIParameters.size()));

  for (std::size_t index = 0; index < contract.runtimeABIParameters.size();
       ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    const support::RuntimeABIParameter &expected =
        contract.runtimeABIParameters[index];
    if (actual.cName != expected.cName || actual.cType != expected.cType ||
        actual.role != expected.role ||
        actual.ownership !=
            support::RuntimeABIParameterOwnership::TargetExportABIOwned)
      return makeRVVTargetRouteError(
          llvm::Twine("segment2-memory target artifact consumer requires "
                      "provider-derived runtime ABI parameter[") +
          llvm::Twine(index) + "] to bind '" + expected.cName + "' as " +
          support::stringifyRuntimeABIParameterRole(expected.role) +
          " with C type '" + expected.cType +
          "' before artifact export but saw '" + actual.cName + "' as " +
          support::stringifyRuntimeABIParameterRole(actual.role) +
          " with C type '" + actual.cType + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVSegment2MemoryRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVSegment2MemoryRouteValidationContract &contract) {
  const llvm::StringRef consumerLabel = contract.consumerLabel;
  const plugin::rvv::RVVRuntimeAVLVLSelectedBoundaryContract &runtimeContract =
      contract.runtimeAVLVLContract;
  if (llvm::Error error =
          validateRVVSegment2MemoryRuntimeABIFacts(description, contract))
    return error;
  if (contract.setVLIntrinsic.empty() || contract.vectorCType.empty() ||
      contract.vlCType.empty() || contract.emitCFullChunkVLName.empty() ||
      contract.emitCLoopVLName.empty() ||
      contract.emitCLoopInductionName.empty() ||
      contract.field0Name.empty() || contract.field1Name.empty() ||
      contract.segmentTupleCType.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived setvl, vector, VL, field, tuple, and loop "
        "facts before validating route statements");

  const support::RuntimeABIParameter &runtimeNABI =
      contract.runtimeABIParameters.back();
  const support::RuntimeABIParameter *runtimeElementCount = nullptr;
  for (const support::RuntimeABIParameter &parameter :
       contract.runtimeABIParameters)
    if (parameter.role ==
        support::RuntimeABIParameterRole::RuntimeElementCount) {
      runtimeElementCount = &parameter;
      break;
    }
  if (!runtimeElementCount || runtimeElementCount != &runtimeNABI)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires runtime n/AVL ABI role to match the selected segment2 ABI "
        "order before validating route statements");

  if (route.getCallOpaqueSteps().size() != contract.expectedPreLoopStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires exact provider-built pre-loop statement count " +
        llvm::Twine(contract.expectedPreLoopStepCount) +
        " before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          preLoopSetVL, consumerLabel, "pre-loop setvl",
          contract.setVLIntrinsic, {{runtimeNABI.cName, runtimeNABI.cType}},
          contract.emitCFullChunkVLName, contract.vlCType))
    return error;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "segment2-memory target artifact consumer requires pre-loop "
          "statements to carry selected typed RVV source provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires exactly one "
        "provider-built runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != contract.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != contract.vlCType ||
      loop.step.expression != runtimeContract.emitCFullChunkVLName ||
      loop.step.cType != contract.vlCType)
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires provider-built "
        "loop bounds and step to mirror runtime AVL/VL route facts");
  if (loop.upperBound.expression != runtimeNABI.cName ||
      loop.upperBound.cType != runtimeNABI.cType)
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires provider-built "
        "loop upper bound to use the runtime n/AVL ABI parameter");

  if (contract.expectedPreLoopStepCount == 0 ||
      contract.expectedLoopBodyStepCount == 0)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-owned statement-plan expectations before artifact "
        "export");
  if (loop.bodySteps.size() != contract.expectedLoopBodyStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires exact provider-built loop statement count " +
        llvm::Twine(contract.expectedLoopBodyStepCount) +
        " for the selected segment2-memory family before artifact export");

  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeNABI.cName) + " - " +
       runtimeContract.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[0], consumerLabel, "loop setvl",
          contract.setVLIntrinsic,
          {{expectedRemainingAVL, contract.vlCType}},
          runtimeContract.emitCLoopVLName, runtimeContract.vlCType))
    return error;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "segment2-memory target artifact consumer requires loop statements "
          "to carry selected typed RVV source provenance");

  auto pointerAtInduction = [&](const support::RuntimeABIParameter &abi) {
    return (llvm::StringRef(abi.cName) + " + " +
            runtimeContract.emitCLoopInductionName)
        .str();
  };
  auto interleavedPointerAtInduction =
      [&](const support::RuntimeABIParameter &abi) {
        return (llvm::StringRef(abi.cName) + " + (" +
                runtimeContract.emitCLoopInductionName + " * 2)")
            .str();
      };
  auto validateVectorLoad =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          llvm::StringRef stepLabel, const support::RuntimeABIParameter &abi,
          llvm::StringRef resultName) -> llvm::Error {
    return validateRVVProviderBuiltRouteStep(
        step, consumerLabel, stepLabel, description.vectorLoadIntrinsic,
        {{pointerAtInduction(abi), abi.cType},
         {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
        resultName, description.vectorCType);
  };
  auto validateFieldStore =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          llvm::StringRef stepLabel, const support::RuntimeABIParameter &abi,
          llvm::StringRef valueName) -> llvm::Error {
    return validateRVVProviderBuiltRouteStep(
        step, consumerLabel, stepLabel, description.storeIntrinsic,
        {{pointerAtInduction(abi), abi.cType},
         {valueName, description.vectorCType},
         {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}});
  };
  auto validateTupleFieldExtract =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          llvm::StringRef stepLabel, llvm::StringRef fieldIndex,
          llvm::StringRef resultName) -> llvm::Error {
    return validateRVVProviderBuiltRouteStep(
        step, consumerLabel, stepLabel, description.segmentFieldExtractIntrinsic,
        {{"segment2_tuple", description.segmentTupleCType},
         {fieldIndex, "size_t"}},
        resultName, description.vectorCType);
  };
  auto validateTupleCreate =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          llvm::StringRef stepLabel, llvm::StringRef field0Name,
          llvm::StringRef field1Name,
          llvm::StringRef resultName) -> llvm::Error {
    return validateRVVProviderBuiltRouteStep(
        step, consumerLabel, stepLabel, description.segmentFieldExtractIntrinsic,
        {{field0Name, description.vectorCType},
         {field1Name, description.vectorCType}},
        resultName, description.segmentTupleCType);
  };
  auto validateComputedMaskPrefix =
      [&](const support::RuntimeABIParameter &compareLhsABI,
          const support::RuntimeABIParameter &compareRhsABI,
          const support::RuntimeABIParameter &field0ABI,
          const support::RuntimeABIParameter &field1ABI,
          llvm::StringRef field0ResultName,
          llvm::StringRef field1ResultName) -> llvm::Error {
    if (description.compareIntrinsic.empty() || description.maskName.empty() ||
        description.maskCType.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires provider-derived computed-mask compare, mask name, and "
          "mask C type facts before validating route statements");
    if (llvm::Error error = validateVectorLoad(
            loop.bodySteps[1], "compare lhs vector load", compareLhsABI,
            "cmp_lhs_vec"))
      return error;
    if (llvm::Error error = validateVectorLoad(
            loop.bodySteps[2], "compare rhs vector load", compareRhsABI,
            "cmp_rhs_vec"))
      return error;
    if (llvm::Error error = validateVectorLoad(
            loop.bodySteps[3], "field0 payload vector load", field0ABI,
            field0ResultName))
      return error;
    if (llvm::Error error = validateVectorLoad(
            loop.bodySteps[4], "field1 payload vector load", field1ABI,
            field1ResultName))
      return error;
    return validateRVVProviderBuiltRouteStep(
        loop.bodySteps[5], consumerLabel, "compare/mask",
        description.compareIntrinsic,
        {{"cmp_lhs_vec", description.vectorCType},
         {"cmp_rhs_vec", description.vectorCType},
         {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
        description.maskName, description.maskCType);
  };
  auto validateRuntimeScalarComputedMaskPrefix =
      [&](const support::RuntimeABIParameter &compareLhsABI,
          const support::RuntimeABIParameter &rhsScalarABI,
          const support::RuntimeABIParameter &field0ABI,
          const support::RuntimeABIParameter &field1ABI,
          llvm::StringRef field0ResultName,
          llvm::StringRef field1ResultName) -> llvm::Error {
    if (description.compareIntrinsic.empty() || description.maskName.empty() ||
        description.maskCType.empty() ||
        contract.rhsScalarSplatIntrinsic.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires provider-derived runtime scalar splat, computed-mask "
          "compare, mask name, and mask C type facts before validating route "
          "statements");
    if (llvm::Error error = validateVectorLoad(
            loop.bodySteps[1], "compare lhs vector load", compareLhsABI,
            "cmp_lhs_vec"))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[2], consumerLabel, "rhs scalar splat",
            contract.rhsScalarSplatIntrinsic,
            {{rhsScalarABI.cName, rhsScalarABI.cType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            "cmp_rhs_vec", description.vectorCType))
      return error;
    if (llvm::Error error = validateVectorLoad(
            loop.bodySteps[3], "field0 payload vector load", field0ABI,
            field0ResultName))
      return error;
    if (llvm::Error error = validateVectorLoad(
            loop.bodySteps[4], "field1 payload vector load", field1ABI,
            field1ResultName))
      return error;
    return validateRVVProviderBuiltRouteStep(
        loop.bodySteps[5], consumerLabel, "compare/mask",
        description.compareIntrinsic,
        {{"cmp_lhs_vec", description.vectorCType},
         {"cmp_rhs_vec", description.vectorCType},
         {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
        description.maskName, description.maskCType);
  };

  switch (description.operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2LoadUnitStore: {
    const support::RuntimeABIParameter &compareLhsABI =
        description.runtimeABIParameters[0];
    const support::RuntimeABIParameter &compareRhsABI =
        description.runtimeABIParameters[1];
    const support::RuntimeABIParameter &sourceABI =
        description.runtimeABIParameters[2];
    const support::RuntimeABIParameter &field0ABI =
        description.runtimeABIParameters[3];
    const support::RuntimeABIParameter &field1ABI =
        description.runtimeABIParameters[4];
    if (llvm::Error error =
            validateComputedMaskPrefix(compareLhsABI, compareRhsABI, field0ABI,
                                       field1ABI, "field0_old_vec",
                                       "field1_old_vec"))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[6], consumerLabel, "masked passthrough tuple",
            description.segmentStoreIntrinsic,
            {{"field0_old_vec", description.vectorCType},
             {"field1_old_vec", description.vectorCType}},
            "segment2_passthrough_tuple", description.segmentTupleCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[7], consumerLabel, "masked segment2 load",
            description.segmentLoadIntrinsic,
            {{description.maskName, description.maskCType},
             {"segment2_passthrough_tuple", description.segmentTupleCType},
             {interleavedPointerAtInduction(sourceABI), sourceABI.cType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            "segment2_tuple", description.segmentTupleCType))
      return error;
    if (llvm::Error error = validateTupleFieldExtract(
            loop.bodySteps[8], "segment2 field0 extract", "0",
            description.field0Name))
      return error;
    if (llvm::Error error = validateTupleFieldExtract(
            loop.bodySteps[9], "segment2 field1 extract", "1",
            description.field1Name))
      return error;
    if (llvm::Error error = validateFieldStore(
            loop.bodySteps[10], "field0 output store", field0ABI,
            description.field0Name))
      return error;
    if (llvm::Error error = validateFieldStore(
            loop.bodySteps[11], "field1 output store", field1ABI,
            description.field1Name))
      return error;
    break;
  }
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2LoadUnitStore: {
    const support::RuntimeABIParameter &compareLhsABI =
        description.runtimeABIParameters[0];
    const support::RuntimeABIParameter &rhsScalarABI =
        description.runtimeABIParameters[1];
    const support::RuntimeABIParameter &sourceABI =
        description.runtimeABIParameters[2];
    const support::RuntimeABIParameter &field0ABI =
        description.runtimeABIParameters[3];
    const support::RuntimeABIParameter &field1ABI =
        description.runtimeABIParameters[4];
    if (llvm::Error error = validateRuntimeScalarComputedMaskPrefix(
            compareLhsABI, rhsScalarABI, field0ABI, field1ABI,
            "field0_old_vec", "field1_old_vec"))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[6], consumerLabel, "masked passthrough tuple",
            description.segmentStoreIntrinsic,
            {{"field0_old_vec", description.vectorCType},
             {"field1_old_vec", description.vectorCType}},
            "segment2_passthrough_tuple", description.segmentTupleCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[7], consumerLabel, "masked segment2 load",
            description.segmentLoadIntrinsic,
            {{description.maskName, description.maskCType},
             {"segment2_passthrough_tuple", description.segmentTupleCType},
             {interleavedPointerAtInduction(sourceABI), sourceABI.cType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            "segment2_tuple", description.segmentTupleCType))
      return error;
    if (llvm::Error error = validateTupleFieldExtract(
            loop.bodySteps[8], "segment2 field0 extract", "0",
            description.field0Name))
      return error;
    if (llvm::Error error = validateTupleFieldExtract(
            loop.bodySteps[9], "segment2 field1 extract", "1",
            description.field1Name))
      return error;
    if (llvm::Error error = validateFieldStore(
            loop.bodySteps[10], "field0 output store", field0ABI,
            description.field0Name))
      return error;
    if (llvm::Error error = validateFieldStore(
            loop.bodySteps[11], "field1 output store", field1ABI,
            description.field1Name))
      return error;
    break;
  }
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2StoreUnitLoad: {
    const support::RuntimeABIParameter &compareLhsABI =
        description.runtimeABIParameters[0];
    const support::RuntimeABIParameter &compareRhsABI =
        description.runtimeABIParameters[1];
    const support::RuntimeABIParameter &field0ABI =
        description.runtimeABIParameters[2];
    const support::RuntimeABIParameter &field1ABI =
        description.runtimeABIParameters[3];
    const support::RuntimeABIParameter &destinationABI =
        description.runtimeABIParameters[4];
    if (llvm::Error error =
            validateComputedMaskPrefix(compareLhsABI, compareRhsABI, field0ABI,
                                       field1ABI, description.field0Name,
                                       description.field1Name))
      return error;
    if (llvm::Error error =
            validateTupleCreate(loop.bodySteps[6], "segment2 tuple create",
                                description.field0Name, description.field1Name,
                                "segment2_tuple"))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[7], consumerLabel, "masked segment2 store",
            description.segmentStoreIntrinsic,
            {{description.maskName, description.maskCType},
             {interleavedPointerAtInduction(destinationABI),
              destinationABI.cType},
             {"segment2_tuple", description.segmentTupleCType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}}))
      return error;
    break;
  }
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad: {
    const support::RuntimeABIParameter &compareLhsABI =
        description.runtimeABIParameters[0];
    const support::RuntimeABIParameter &rhsScalarABI =
        description.runtimeABIParameters[1];
    const support::RuntimeABIParameter &field0ABI =
        description.runtimeABIParameters[2];
    const support::RuntimeABIParameter &field1ABI =
        description.runtimeABIParameters[3];
    const support::RuntimeABIParameter &destinationABI =
        description.runtimeABIParameters[4];
    if (llvm::Error error = validateRuntimeScalarComputedMaskPrefix(
            compareLhsABI, rhsScalarABI, field0ABI, field1ABI,
            description.field0Name, description.field1Name))
      return error;
    if (llvm::Error error =
            validateTupleCreate(loop.bodySteps[6], "segment2 tuple create",
                                description.field0Name, description.field1Name,
                                "segment2_tuple"))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[7], consumerLabel, "masked segment2 store",
            description.segmentStoreIntrinsic,
            {{description.maskName, description.maskCType},
             {interleavedPointerAtInduction(destinationABI),
              destinationABI.cType},
             {"segment2_tuple", description.segmentTupleCType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}}))
      return error;
    break;
  }
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2UpdateUnitLoad: {
    if (description.segment2UpdateArithmeticIntrinsic.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires provider-derived segment2 update arithmetic intrinsic "
          "before validating route statements");
    const support::RuntimeABIParameter &compareLhsABI =
        description.runtimeABIParameters[0];
    const support::RuntimeABIParameter &compareRhsABI =
        description.runtimeABIParameters[1];
    const support::RuntimeABIParameter &field0ABI =
        description.runtimeABIParameters[2];
    const support::RuntimeABIParameter &field1ABI =
        description.runtimeABIParameters[3];
    const support::RuntimeABIParameter &destinationABI =
        description.runtimeABIParameters[4];
    if (llvm::Error error = validateComputedMaskPrefix(
            compareLhsABI, compareRhsABI, field0ABI, field1ABI,
            "segment2_update_field0_src_vec", description.field1Name))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[6], consumerLabel, "segment2 update arithmetic",
            description.segment2UpdateArithmeticIntrinsic,
            {{"segment2_update_field0_src_vec", description.vectorCType},
             {description.field1Name, description.vectorCType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            description.field0Name, description.vectorCType))
      return error;
    if (llvm::Error error =
            validateTupleCreate(loop.bodySteps[7], "segment2 tuple create",
                                description.field0Name, description.field1Name,
                                "segment2_tuple"))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[8], consumerLabel, "masked segment2 store",
            description.segmentStoreIntrinsic,
            {{description.maskName, description.maskCType},
             {interleavedPointerAtInduction(destinationABI),
              destinationABI.cType},
             {"segment2_tuple", description.segmentTupleCType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}}))
      return error;
    break;
  }
  case plugin::rvv::RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore: {
    const support::RuntimeABIParameter &sourceABI =
        description.runtimeABIParameters[0];
    const support::RuntimeABIParameter &field0ABI =
        description.runtimeABIParameters[1];
    const support::RuntimeABIParameter &field1ABI =
        description.runtimeABIParameters[2];
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[1], consumerLabel, "segment2 load",
            description.segmentLoadIntrinsic,
            {{interleavedPointerAtInduction(sourceABI), sourceABI.cType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}},
            "segment2_tuple", description.segmentTupleCType))
      return error;
    if (llvm::Error error = validateTupleFieldExtract(
            loop.bodySteps[2], "segment2 field0 extract", "0",
            description.field0Name))
      return error;
    if (llvm::Error error = validateTupleFieldExtract(
            loop.bodySteps[3], "segment2 field1 extract", "1",
            description.field1Name))
      return error;
    if (llvm::Error error = validateFieldStore(
            loop.bodySteps[4], "field0 output store", field0ABI,
            description.field0Name))
      return error;
    if (llvm::Error error = validateFieldStore(
            loop.bodySteps[5], "field1 output store", field1ABI,
            description.field1Name))
      return error;
    break;
  }
  case plugin::rvv::RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad: {
    const support::RuntimeABIParameter &field0ABI =
        description.runtimeABIParameters[0];
    const support::RuntimeABIParameter &field1ABI =
        description.runtimeABIParameters[1];
    const support::RuntimeABIParameter &destinationABI =
        description.runtimeABIParameters[2];
    if (llvm::Error error = validateVectorLoad(
            loop.bodySteps[1], "field0 input vector load", field0ABI,
            description.field0Name))
      return error;
    if (llvm::Error error = validateVectorLoad(
            loop.bodySteps[2], "field1 input vector load", field1ABI,
            description.field1Name))
      return error;
    if (llvm::Error error =
            validateTupleCreate(loop.bodySteps[3], "segment2 tuple create",
                                description.field0Name, description.field1Name,
                                "segment2_tuple"))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[4], consumerLabel, "segment2 store",
            description.segmentStoreIntrinsic,
            {{interleavedPointerAtInduction(destinationABI),
              destinationABI.cType},
             {"segment2_tuple", description.segmentTupleCType},
             {runtimeContract.emitCLoopVLName, runtimeContract.vlCType}}))
      return error;
    break;
  }
  default:
    llvm_unreachable("validated non-segment2 operation as segment2-memory");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVSegment2MemoryRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  std::optional<plugin::rvv::RVVSegment2MemoryRouteValidationContract>
      contract =
          plugin::rvv::getRVVSegment2MemoryRouteValidationContract(description);
  if (!contract)
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires provider-owned "
        "segment2 memory route validation contract before artifact export");

  if (route.getRouteID() != contract->emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract->consumerLabel) +
        " requires rebuilt provider route id '" + contract->emitCRouteID +
        "' but route carried '" +
        route.getRouteID() + "'");
  if (description.emitCRouteID != contract->emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract->consumerLabel) +
        " requires provider-owned route id '" + contract->emitCRouteID +
        "' but description carried '" + description.emitCRouteID + "'");
  if (contract->providerSupportedMirror.empty() ||
      contract->routeOperandBindingPlanID.empty() ||
      contract->routeOperandBindingSummary.empty() ||
      contract->elementTypeName.empty() || contract->sew == 0 ||
      contract->lmul.empty() || contract->tailPolicy.empty() ||
      contract->maskPolicy.empty() || contract->configContractID.empty() ||
      contract->requiredHeaderDeclarations.empty() ||
      contract->cTypeMappingSummary.empty() ||
      contract->segmentMemoryLayout.empty() ||
      contract->sourceMemoryForm.empty() ||
      contract->destinationMemoryForm.empty() ||
      contract->segmentCount != 2 || contract->vectorCType.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract->consumerLabel) +
        " requires complete provider-owned route payload, dtype/config, "
        "binding, header/type, intrinsic, and segment layout contract "
        "facts before artifact export");

  if (llvm::Error error = validateRVVRuntimeAVLVLSelectedBoundaryContract(
          description, contract->runtimeAVLVLContract))
    return error;
  if (llvm::Error error = validateRVVRouteLocalRuntimeAVLVLMirrors(
          contract->consumerLabel, contract->runtimeAVLVLContract,
          contract->runtimeControlPlanID, contract->runtimeABIOrder,
          contract->setVLIntrinsic, contract->vlCType,
          contract->emitCFullChunkVLName, contract->emitCLoopVLName,
          contract->emitCLoopInductionName))
    return error;

  if (contract->usesComputedMaskSegment2) {
    if (contract->computedMaskMemoryRouteFamilyPlanID.empty() ||
        contract->computedMaskMemoryMaskProducerSource.empty() ||
        contract->maskRole.empty() || contract->maskSource.empty() ||
        contract->maskMemoryForm.empty() ||
        contract->compareIntrinsic.empty())
      return makeRVVTargetRouteError(
          "computed-mask segment2-memory target artifact consumer requires "
          "provider-derived computed-mask family, producer, role, source, "
          "memory-form, and compare facts before artifact export");
    if (llvm::Error error =
            validateRVVSegment2HeaderBindingSummary(description, *contract))
      return error;
  } else if (contract->usesPlainSegment2) {
    if (contract->segment2MemoryRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          "plain segment2-memory target artifact consumer requires a "
          "provider-derived segment2 route-family plan mirror before artifact "
          "export");
    if (llvm::Error error =
            validateRVVSegment2HeaderBindingSummary(description, *contract))
      return error;
  } else {
    llvm_unreachable("validated non-segment2 operation as segment2-memory");
  }
  if (llvm::Error error =
          validateRVVSegment2MemoryProviderFactsFromContract(description,
                                                            *contract))
    return error;
  if (llvm::Error error =
          validateRVVSegment2MemoryRuntimeABIFacts(description, *contract))
    return error;

  if (!description.scalarBroadcastElementwiseRouteFamilyPlanID.empty() ||
      !description.elementwiseArithmeticRouteFamilyPlanID.empty() ||
      !description.wideningConversionRouteFamilyPlanID.empty() ||
      !description.runtimeScalarSplatStoreRouteFamilyPlanID.empty() ||
      !description.plainMAccRouteFamilyPlanID.empty() ||
      !description.scalarBroadcastMAccRouteFamilyPlanID.empty() ||
      !description.accumulationRouteFamilyPlanID.empty() ||
      !description.wideningMAccRelation.empty() ||
      !description.wideningDotProductRelation.empty() ||
      !description.plainCompareSelectRouteFamilyPlanID.empty() ||
      !description.computedMaskSelectRouteFamilyPlanID.empty() ||
      !description.standaloneReductionRouteFamilyPlanID.empty() ||
      !description.contractionRouteFamilyPlanID.empty() ||
      !description.baseMemoryMovementRouteFamilyPlanID.empty())
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer rejects stale non-segment2 "
        "route-family facts");

  if (llvm::Error error =
          validateRVVSegment2MemoryRouteHeaders(route, *contract))
    return error;
  if (llvm::Error error =
          validateRVVSegment2MemoryRouteTypeMappings(route, *contract))
    return error;
  if (llvm::Error error =
          validateRVVSegment2MemoryRouteABIMappings(route, *contract))
    return error;
  return validateRVVSegment2MemoryRouteStatementPlan(route, description,
                                                    *contract);
}

llvm::Error validateRVVSegment2MemoryTargetArtifactProviderFactsImpl(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVSegment2MemoryRoutePayloadFacts(context.route,
                                                    context.description);
}

llvm::Error validateRVVSegment2MemoryTargetArtifactCandidateMirrorsImpl(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description =
      context.description;

  std::optional<plugin::rvv::RVVMemoryRouteMetadataMirrorContractSet>
      contract = plugin::rvv::getRVVSegment2MemoryRouteMetadataMirrorContract(
          description);
  if (!contract)
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires provider-owned "
        "canonical memory metadata mirror contract before checking candidate "
        "mirrors");
  return validateRVVProviderMemoryRouteMetadataMirrorContract(candidate,
                                                              *contract);
}

bool isRVVSegment2LoadTargetArtifactFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return description.operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 ComputedMaskSegment2LoadUnitStore ||
         description.operation == plugin::rvv::RVVSelectedBodyOperationKind::
                                      RuntimeScalarComputedMaskSegment2LoadUnitStore;
}

bool isRVVSegment2StoreTargetArtifactFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return description.operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 ComputedMaskSegment2StoreUnitLoad ||
         description.operation == plugin::rvv::RVVSelectedBodyOperationKind::
                                      RuntimeScalarComputedMaskSegment2StoreUnitLoad;
}

bool isRVVSegment2UpdateTargetArtifactFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return description.operation == plugin::rvv::RVVSelectedBodyOperationKind::
                                      ComputedMaskSegment2UpdateUnitLoad;
}

bool isRVVSegment2DeinterleaveTargetArtifactFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return description.operation == plugin::rvv::RVVSelectedBodyOperationKind::
                                      Segment2DeinterleaveUnitStore;
}

bool isRVVSegment2InterleaveTargetArtifactFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return description.operation == plugin::rvv::RVVSelectedBodyOperationKind::
                                      Segment2InterleaveUnitLoad;
}

struct RVVSegment2MemoryTargetArtifactFamilyValidator {
  llvm::StringLiteral familyName;
  bool (*isConsumer)(
      const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description);
  llvm::Error (*validateProviderFacts)(
      const RVVTargetArtifactRouteFamilyValidationContext &context);
  llvm::Error (*validateCandidateMirrors)(
      const RVVTargetArtifactRouteFamilyValidationContext &context);
};

llvm::Error validateRVVSegment2MemoryTargetArtifactFamilyProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVSegment2MemoryTargetArtifactProviderFactsImpl(context);
}

llvm::Error validateRVVSegment2MemoryTargetArtifactFamilyCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVSegment2MemoryTargetArtifactCandidateMirrorsImpl(context);
}

llvm::ArrayRef<RVVSegment2MemoryTargetArtifactFamilyValidator>
getRVVSegment2MemoryTargetArtifactFamilyValidators() {
  static const RVVSegment2MemoryTargetArtifactFamilyValidator validators[] = {
      {llvm::StringLiteral("computed-mask-segment2-load"),
       isRVVSegment2LoadTargetArtifactFamilyConsumer,
       validateRVVSegment2MemoryTargetArtifactFamilyProviderFacts,
       validateRVVSegment2MemoryTargetArtifactFamilyCandidateMirrors},
      {llvm::StringLiteral("computed-mask-segment2-store"),
       isRVVSegment2StoreTargetArtifactFamilyConsumer,
       validateRVVSegment2MemoryTargetArtifactFamilyProviderFacts,
       validateRVVSegment2MemoryTargetArtifactFamilyCandidateMirrors},
      {llvm::StringLiteral("computed-mask-segment2-update"),
       isRVVSegment2UpdateTargetArtifactFamilyConsumer,
       validateRVVSegment2MemoryTargetArtifactFamilyProviderFacts,
       validateRVVSegment2MemoryTargetArtifactFamilyCandidateMirrors},
      {llvm::StringLiteral("plain-segment2-deinterleave"),
       isRVVSegment2DeinterleaveTargetArtifactFamilyConsumer,
       validateRVVSegment2MemoryTargetArtifactFamilyProviderFacts,
       validateRVVSegment2MemoryTargetArtifactFamilyCandidateMirrors},
      {llvm::StringLiteral("plain-segment2-interleave"),
       isRVVSegment2InterleaveTargetArtifactFamilyConsumer,
       validateRVVSegment2MemoryTargetArtifactFamilyProviderFacts,
       validateRVVSegment2MemoryTargetArtifactFamilyCandidateMirrors},
  };
  return validators;
}

llvm::Error selectRVVSegment2MemoryTargetArtifactFamilyValidator(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef validationKind,
    const RVVSegment2MemoryTargetArtifactFamilyValidator *&selected) {
  selected = nullptr;
  for (const RVVSegment2MemoryTargetArtifactFamilyValidator &validator :
       getRVVSegment2MemoryTargetArtifactFamilyValidators()) {
    if (!validator.isConsumer(description))
      continue;
    if (selected)
      return makeRVVTargetRouteError(
          llvm::Twine("segment2-memory target artifact family registry matched "
                      "both '") +
          selected->familyName + "' and '" + validator.familyName +
          "' while checking " + validationKind);
    selected = &validator;
  }
  return llvm::Error::success();
}

llvm::Error makeMissingRVVSegment2MemoryTargetArtifactFamilyValidatorError(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef validationKind) {
  return makeRVVTargetRouteError(
      llvm::Twine("no segment2-memory target artifact family validator owns "
                  "selected typed RVV route operation '") +
      plugin::rvv::stringifyRVVSelectedBodyOperationKind(
          description.operation) +
      "' with memory form '" +
      plugin::rvv::stringifyRVVSelectedBodyMemoryForm(description.memoryForm) +
      "' while checking " + validationKind);
}

llvm::Error validateRVVSegment2MemoryTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const RVVSegment2MemoryTargetArtifactFamilyValidator *validator = nullptr;
  if (llvm::Error error = selectRVVSegment2MemoryTargetArtifactFamilyValidator(
          context.description, "rebuilt provider facts", validator))
    return error;
  if (!validator)
    return makeMissingRVVSegment2MemoryTargetArtifactFamilyValidatorError(
        context.description, "rebuilt provider facts");
  if (!validator->validateProviderFacts)
    return makeRVVTargetRouteError(
        llvm::Twine("segment2-memory target artifact family validator '") +
        validator->familyName + "' has no provider-fact validator");
  return validator->validateProviderFacts(context);
}

llvm::Error validateRVVSegment2MemoryTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const RVVSegment2MemoryTargetArtifactFamilyValidator *validator = nullptr;
  if (llvm::Error error = selectRVVSegment2MemoryTargetArtifactFamilyValidator(
          context.description, "candidate metadata mirrors", validator))
    return error;
  if (!validator)
    return makeMissingRVVSegment2MemoryTargetArtifactFamilyValidatorError(
        context.description, "candidate metadata mirrors");
  if (!validator->validateCandidateMirrors)
    return makeRVVTargetRouteError(
        llvm::Twine("segment2-memory target artifact family validator '") +
        validator->familyName + "' has no candidate-mirror validator");
  return validator->validateCandidateMirrors(context);
}

llvm::StringRef getRVVElementwiseArithmeticSignedCType(
    const plugin::rvv::RVVElementwiseArithmeticRouteValidationContract
        &contract) {
  if (contract.elementTypeName == "i16")
    return "int16_t";
  if (contract.elementTypeName == "i32")
    return "int32_t";
  if (contract.elementTypeName == "i64")
    return "int64_t";
  return {};
}

llvm::Error validateRVVElementwiseArithmeticRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVElementwiseArithmeticRouteValidationContract
        &contract) {
  if (contract.requiredHeaderDeclarations.empty() ||
      contract.requiredHeaders.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned required_header_declarations before "
        "accepting the route artifact");

  for (llvm::StringRef header : contract.requiredHeaders) {
    if (header.empty())
      return makeRVVTargetRouteError(llvm::Twine(contract.consumerLabel) +
                                     " saw an empty provider route header "
                                     "declaration");
    if (!routeHasHeader(route, header))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route header '" + header +
          "' before artifact export");
  }
  return llvm::Error::success();
}

llvm::Error validateRVVElementwiseArithmeticRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVElementwiseArithmeticRouteValidationContract
        &contract) {
  if (contract.elementTypeName.empty() || contract.sew == 0 ||
      contract.lmul.empty() || contract.vlCType.empty() ||
      contract.vectorTypeName.empty() || contract.vectorCType.empty() ||
      contract.cTypeMappingSummary.empty() || contract.typeMappings.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived dtype, SEW, LMUL, VL, vector, and C type "
        "mapping facts before artifact export");

  if (getRVVElementwiseArithmeticSignedCType(contract).empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires a provider-derived signed C type for the typed RVV element "
        "type");

  for (const plugin::rvv::RVVElementwiseArithmeticRouteTypeMappingContract
           &mapping : contract.typeMappings) {
    if (mapping.sourceType.empty() || mapping.cType.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires non-empty provider type mapping contract entries");
    if (!routeHasTypeMapping(route, mapping.sourceType, mapping.cType))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route type mapping '" +
          mapping.sourceType + "' -> '" + mapping.cType + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVElementwiseArithmeticRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVElementwiseArithmeticRouteValidationContract
        &contract) {
  if (contract.runtimeABIParameterRoles.size() !=
      contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned runtime ABI role order with one role per "
        "ABI parameter before artifact export");
  for (std::size_t index = 0, count = contract.runtimeABIParameters.size();
       index < count; ++index) {
    support::RuntimeABIParameterRole expectedRole =
        contract.runtimeABIParameterRoles[index];
    support::RuntimeABIParameterRole actualRole =
        contract.runtimeABIParameters[index].role;
    if (actualRole != expectedRole)
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-owned runtime ABI parameter[" +
          llvm::Twine(index) + "] role '" +
          support::stringifyRuntimeABIParameterRole(expectedRole) +
          "' but saw '" +
          support::stringifyRuntimeABIParameterRole(actualRole) +
          "' for parameter '" + contract.runtimeABIParameters[index].cName +
          "'");
  }
  if (route.getABIMappings().size() != contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires rebuilt provider route ABI mapping count " +
        llvm::Twine(contract.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  llvm::StringRef signedCType =
      getRVVElementwiseArithmeticSignedCType(contract);
  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        contract.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");

    switch (expected.role) {
    case support::RuntimeABIParameterRole::LHSInputBuffer:
    case support::RuntimeABIParameterRole::RHSInputBuffer:
    case support::RuntimeABIParameterRole::OutputBuffer:
      if (!llvm::StringRef(expected.cType).contains(signedCType))
        return makeRVVTargetRouteError(
            llvm::Twine("elementwise arithmetic target artifact consumer "
                        "requires ABI parameter '") +
            expected.cName + "' C type '" + expected.cType +
            "' to carry signed element C type '" + signedCType + "'");
      break;
    case support::RuntimeABIParameterRole::RHSScalarValue:
      if (expected.cType != signedCType)
        return makeRVVTargetRouteError(
            "scalar-broadcast elementwise target artifact consumer requires "
            "rhs scalar ABI C type to mirror the typed RVV element C type");
      break;
    case support::RuntimeABIParameterRole::RuntimeElementCount:
      if (expected.cType != contract.vlCType)
        return makeRVVTargetRouteError(
            llvm::Twine(contract.consumerLabel) +
            " requires runtime element-count ABI C type to mirror provider VL "
            "C type");
      break;
    case support::RuntimeABIParameterRole::LHSInputStride:
    case support::RuntimeABIParameterRole::RHSInputStride:
    case support::RuntimeABIParameterRole::OutputStride:
      if (expected.cType != "size_t" && expected.cType != "ptrdiff_t")
        return makeRVVTargetRouteError(
            "strided elementwise arithmetic target artifact consumer requires "
            "stride ABI C types to be size_t or ptrdiff_t integer stride "
            "parameters");
      break;
    default:
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " rejects ABI roles outside lhs, rhs, rhs scalar, out, runtime n, "
          "and optional stride roles");
    }
  }

  return llvm::Error::success();
}

llvm::Error validateRVVElementwiseArithmeticRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVElementwiseArithmeticRouteValidationContract
        &contract) {
  const plugin::rvv::RVVRuntimeAVLVLSelectedBoundaryContract &runtimeContract =
      contract.runtimeAVLVLContract;
  const support::RuntimeABIParameter &runtimeN =
      runtimeContract.runtimeAVLParameter;

  if (contract.expectedPreLoopStepCount == 0 ||
      contract.expectedLoopBodyStepCount == 0)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned statement-plan expectations before artifact "
        "export");
  if (route.getCallOpaqueSteps().size() != contract.expectedPreLoopStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires exact provider-built pre-loop statement count " +
        llvm::Twine(contract.expectedPreLoopStepCount) +
        " before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != runtimeContract.setVLIntrinsic ||
      preLoopSetVL.operands.size() != 1 ||
      preLoopSetVL.operands.front().expression != runtimeN.cName ||
      preLoopSetVL.operands.front().cType != runtimeN.cType ||
      !stepHasResult(preLoopSetVL, runtimeContract.emitCFullChunkVLName,
                     runtimeContract.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires rebuilt provider route pre-loop setvl statement to use "
        "runtime n/AVL and define the full-chunk VL");
  if (!routeStepSourceIsSelectedRVVBody(preLoopSetVL))
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires pre-loop setvl provenance from the selected typed RVV body");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires exactly one provider-built runtime AVL/VL loop before "
        "artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != runtimeContract.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != runtimeContract.vlCType ||
      loop.upperBound.expression != runtimeN.cName ||
      loop.upperBound.cType != runtimeN.cType ||
      loop.step.expression != runtimeContract.emitCFullChunkVLName ||
      loop.step.cType != runtimeContract.vlCType)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-built loop bounds and step to mirror runtime "
        "n/AVL/VL facts");
  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeN.cName) + " - " +
       runtimeContract.emitCLoopInductionName)
          .str();
  if (loop.bodySteps.size() != contract.expectedLoopBodyStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires exact provider-built loop statement count " +
        llvm::Twine(contract.expectedLoopBodyStepCount) +
        " before artifact export");
  if (loop.bodySteps.empty() ||
      loop.bodySteps.front().callee != runtimeContract.setVLIntrinsic ||
      loop.bodySteps.front().operands.size() != 1 ||
      loop.bodySteps.front().operands.front().expression !=
          expectedRemainingAVL ||
      loop.bodySteps.front().operands.front().cType !=
          runtimeContract.vlCType ||
      !stepHasResult(loop.bodySteps.front(), runtimeContract.emitCLoopVLName,
                     runtimeContract.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-built loop setvl to derive per-iteration VL from "
        "remaining runtime AVL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires loop statements to carry selected typed RVV source "
          "provenance");

  const bool isMasked =
      contract.kind ==
      plugin::rvv::RVVElementwiseArithmeticRouteValidationKind::Masked;
  const bool isStrided =
      contract.kind ==
      plugin::rvv::RVVElementwiseArithmeticRouteValidationKind::Strided;
  const bool isScalarBroadcast =
      contract.kind == plugin::rvv::
                           RVVElementwiseArithmeticRouteValidationKind::
                               ScalarBroadcast;
  const bool isRHSBroadcastLoad =
      contract.memoryForm ==
      plugin::rvv::RVVSelectedBodyMemoryForm::RHSBroadcastLoad;

  if (isStrided) {
    if (!routeLoopContainsCallee(loop, contract.stridedLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, contract.intrinsic) ||
        !routeLoopContainsCallee(loop, contract.stridedStoreIntrinsic))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-built strided loads, elementwise compute, and "
          "strided store statements before artifact export");
  } else if (isScalarBroadcast || isRHSBroadcastLoad) {
    if (!routeLoopContainsCallee(loop, contract.vectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, contract.rhsBroadcastIntrinsic) ||
        !routeLoopContainsCallee(loop, contract.intrinsic) ||
        !routeLoopContainsCallee(loop, contract.storeIntrinsic))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-built vector load, RHS broadcast, elementwise "
          "compute, and store statements before artifact export");
  } else {
    if (!routeLoopContainsCallee(loop, contract.vectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, contract.intrinsic) ||
        !routeLoopContainsCallee(loop, contract.storeIntrinsic))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-built vector loads, elementwise compute, and "
          "store statements before artifact export");
  }
  if (isMasked) {
    if (!routeLoopContainsCallee(loop, contract.compareIntrinsic) ||
        !routeLoopContainsCallee(loop, contract.maskedMergeIntrinsic))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-built compare and masked merge statements before "
          "artifact export");
  } else if (!contract.compareIntrinsic.empty() ||
             !contract.maskedMergeIntrinsic.empty()) {
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " rejects stale masked compare/merge leaves for non-masked "
        "elementwise routes");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVElementwiseArithmeticDescriptionAgainstContract(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVElementwiseArithmeticRouteValidationContract
        &contract) {
  if (description.operation != contract.operation)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned operation '" +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            contract.operation) +
        "' but was '" +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            description.operation) +
        "'");
  if (description.emitCRouteID != contract.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned route id '" + contract.emitCRouteID +
        "' but description carried '" + description.emitCRouteID + "'");
  if (description.memoryForm != contract.memoryForm)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned memory form '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(contract.memoryForm) +
        "' but was '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
            description.memoryForm) +
        "'");
  if (description.typedComputeOpName != contract.typedComputeOpName)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires selected typed " + contract.typedComputeOpName +
        " body before artifact export");
  if (description.elementTypeName != contract.elementTypeName ||
      description.sew != contract.sew || description.lmul != contract.lmul ||
      description.tailPolicy != contract.tailPolicy ||
      description.maskPolicy != contract.maskPolicy ||
      description.configContractID != contract.configContractID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned dtype/config facts element '" +
        contract.elementTypeName + "', SEW " + llvm::Twine(contract.sew) +
        ", LMUL '" + contract.lmul + "', tail '" + contract.tailPolicy +
        "', mask '" + contract.maskPolicy + "', config '" +
        contract.configContractID + "' but description carried element '" +
        description.elementTypeName + "', SEW " + llvm::Twine(description.sew) +
        ", LMUL '" + description.lmul + "', tail '" +
        description.tailPolicy + "', mask '" + description.maskPolicy +
        "', config '" + description.configContractID + "'");
  if (description.runtimeABIParameters.size() !=
      contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned runtime ABI parameter count " +
        llvm::Twine(contract.runtimeABIParameters.size()) +
        " before artifact export");
  for (std::size_t index = 0, count = contract.runtimeABIParameters.size();
       index < count; ++index)
    if (!runtimeABIParameterEquals(description.runtimeABIParameters[index],
                                   contract.runtimeABIParameters[index]))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-owned runtime ABI parameter[" +
          llvm::Twine(index) + "] '" +
          contract.runtimeABIParameters[index].cName +
          "' before artifact export");

  if (description.providerSupportedMirror !=
          contract.providerSupportedMirror ||
      description.targetLeafProfile != contract.targetLeafProfile ||
      description.requiredHeaderDeclarations !=
          contract.requiredHeaderDeclarations ||
      description.cTypeMappingSummary != contract.cTypeMappingSummary ||
      description.routeOperandBindingPlanID !=
          contract.routeOperandBindingPlanID ||
      description.routeOperandBindingSummary !=
          contract.routeOperandBindingSummary)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned support, target profile, header, type, and "
        "binding facts but saw support '" +
        description.providerSupportedMirror + "', profile '" +
        description.targetLeafProfile + "', headers '" +
        description.requiredHeaderDeclarations + "', type mapping '" +
        description.cTypeMappingSummary + "', binding plan '" +
        description.routeOperandBindingPlanID + "', binding summary '" +
        description.routeOperandBindingSummary + "'");

  if (description.elementwiseArithmeticRouteFamilyPlanID !=
          contract.elementwiseArithmeticRouteFamilyPlanID ||
      description.scalarBroadcastElementwiseRouteFamilyPlanID !=
          contract.scalarBroadcastElementwiseRouteFamilyPlanID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned elementwise route-family plan mirrors");
  if (description.sourceMemoryForm != contract.sourceMemoryForm ||
      description.destinationMemoryForm != contract.destinationMemoryForm ||
      description.stridedMemoryLayout != contract.stridedMemoryLayout ||
      description.lhsStrideSource != contract.lhsStrideSource ||
      description.rhsStrideSource != contract.rhsStrideSource ||
      description.outStrideSource != contract.outStrideSource)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned source/destination and strided memory facts "
        "before artifact export");
  if (description.maskRole != contract.maskRole ||
      description.maskSource != contract.maskSource ||
      description.maskMemoryForm != contract.maskMemoryForm ||
      description.inactiveLaneContract != contract.inactiveLaneContract ||
      description.maskedPassthroughLayout !=
          contract.maskedPassthroughLayout)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned mask, inactive-lane, and passthrough facts "
        "before artifact export");
  if (description.vectorTypeName != contract.vectorTypeName ||
      description.vectorCType != contract.vectorCType ||
      description.maskTypeName != contract.maskTypeName ||
      description.maskCType != contract.maskCType ||
      description.vectorLoadIntrinsic != contract.vectorLoadIntrinsic ||
      description.stridedLoadIntrinsic != contract.stridedLoadIntrinsic ||
      description.rhsBroadcastIntrinsic != contract.rhsBroadcastIntrinsic ||
      description.intrinsic != contract.intrinsic ||
      description.compareIntrinsic != contract.compareIntrinsic ||
      description.maskedMergeIntrinsic != contract.maskedMergeIntrinsic ||
      description.storeIntrinsic != contract.storeIntrinsic ||
      description.stridedStoreIntrinsic != contract.stridedStoreIntrinsic ||
      description.resultName != contract.resultName ||
      description.maskName != contract.maskName)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned type, intrinsic, result, and mask leaf facts "
        "before artifact export");
  return llvm::Error::success();
}

llvm::Error validateRVVElementwiseArithmeticRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVElementwiseArithmeticRouteValidationContract
        &contract) {
  if (contract.providerSupportedMirror.empty() ||
      contract.routeOperandBindingPlanID.empty() ||
      contract.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-supported and operand-binding contract facts "
        "before artifact export");

  const bool isScalarBroadcast =
      contract.kind == plugin::rvv::
                           RVVElementwiseArithmeticRouteValidationKind::
                               ScalarBroadcast;
  const bool isStrided = isRVVStridedElementwiseArithmeticRouteFamilyOperation(
      contract.operation);
  const bool isRHSBroadcastLoad =
      contract.memoryForm ==
      plugin::rvv::RVVSelectedBodyMemoryForm::RHSBroadcastLoad;
  if (contract.elementTypeName.empty() ||
      contract.sew == 0 || contract.lmul.empty() ||
      contract.tailPolicy.empty() || contract.maskPolicy.empty() ||
      contract.configContractID.empty() ||
      contract.intrinsic.empty() || contract.storeIntrinsic.empty() ||
      contract.resultName.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived runtime AVL/VL, dtype, SEW, LMUL, policy, "
        "config, intrinsic, and result facts before artifact export");
  if (llvm::Error error = validateRVVRuntimeAVLVLSelectedBoundaryContract(
          description, contract.runtimeAVLVLContract))
    return error;
  if (llvm::Error error = validateRVVRouteLocalRuntimeAVLVLMirrors(
          contract.consumerLabel, contract.runtimeAVLVLContract,
          contract.runtimeControlPlanID, contract.runtimeABIOrder,
          contract.setVLIntrinsic, contract.vlCType,
          contract.emitCFullChunkVLName, contract.emitCLoopVLName,
          contract.emitCLoopInductionName))
    return error;
  if (route.getRouteID() != contract.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires rebuilt provider route id '" + contract.emitCRouteID +
        "' but route carried '" + route.getRouteID() + "'");
  if (llvm::Error error = validateRVVElementwiseArithmeticDescriptionAgainstContract(
          description, contract))
    return error;
  if (contract.vectorLoadIntrinsic.empty() && !isStrided)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived vector load facts before artifact export");
  if (isScalarBroadcast) {
    if (contract.scalarBroadcastElementwiseRouteFamilyPlanID.empty() ||
        contract.rhsBroadcastIntrinsic.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-derived scalar-broadcast plan and RHS broadcast "
          "facts before artifact export");
    if (!contract.elementwiseArithmeticRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " rejects stale plain elementwise arithmetic route-family facts");
  } else {
    if (contract.elementwiseArithmeticRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-derived elementwise route-family facts before "
          "artifact export");
    if (!contract.scalarBroadcastElementwiseRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " rejects stale scalar-broadcast route facts");
    if (isRHSBroadcastLoad) {
      if (contract.rhsBroadcastIntrinsic.empty())
        return makeRVVTargetRouteError(
            llvm::Twine(contract.consumerLabel) +
            " requires provider-derived RHS broadcast facts before artifact "
            "export");
    } else if (!contract.rhsBroadcastIntrinsic.empty()) {
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " rejects stale RHS broadcast route facts");
    }
  }
  if (!description.wideningConversionRouteFamilyPlanID.empty() ||
      !description.sourceVectorTypeName.empty() ||
      !description.sourceVectorCType.empty() ||
      !description.sourceVectorLoadIntrinsic.empty() ||
      !description.conversionRelation.empty())
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer rejects stale "
        "conversion/source route facts");

  const bool isMasked = isRVVMaskedElementwiseArithmeticRouteFamilyOperation(
      contract.operation);
  if (isMasked) {
    if (contract.compareIntrinsic.empty() ||
        contract.maskedMergeIntrinsic.empty() ||
        contract.maskName.empty() || contract.maskRole.empty() ||
        contract.maskSource.empty() ||
        contract.inactiveLaneContract.empty() ||
        contract.maskedPassthroughLayout.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-derived mask, compare, merge, inactive-lane, "
          "and passthrough facts before artifact export");
  } else if (!contract.maskName.empty() || !contract.maskRole.empty() ||
             !contract.maskSource.empty() ||
             !contract.inactiveLaneContract.empty() ||
             !contract.maskedPassthroughLayout.empty()) {
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " rejects stale mask metadata for non-masked elementwise routes");
  }
  if (isStrided) {
    if (contract.stridedLoadIntrinsic.empty() ||
        contract.stridedStoreIntrinsic.empty() ||
        contract.stridedMemoryLayout.empty() ||
        contract.lhsStrideSource.empty() ||
        contract.rhsStrideSource.empty() ||
        contract.outStrideSource.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-derived strided load/store and stride-source "
          "facts before artifact export");
  } else if (!contract.stridedLoadIntrinsic.empty() ||
             !contract.stridedStoreIntrinsic.empty() ||
             !contract.stridedMemoryLayout.empty() ||
             !contract.lhsStrideSource.empty() ||
             !contract.rhsStrideSource.empty() ||
             !contract.outStrideSource.empty()) {
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " rejects stale strided metadata for non-strided elementwise routes");
  }

  if (llvm::Error error =
          validateRVVElementwiseArithmeticRouteHeaders(route, contract))
    return error;
  if (llvm::Error error =
          validateRVVElementwiseArithmeticRouteTypeMappings(route, contract))
    return error;
  if (llvm::Error error =
          validateRVVElementwiseArithmeticRouteABIMappings(route, contract))
    return error;
  return validateRVVElementwiseArithmeticRouteStatementPlan(route, contract);
}

llvm::Error validateRVVElementwiseArithmeticTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  std::optional<plugin::rvv::RVVElementwiseArithmeticRouteValidationContract>
      contract =
          plugin::rvv::getRVVElementwiseArithmeticRouteValidationContract(
              context.description);
  if (!contract)
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires "
        "provider-owned elementwise arithmetic route validation contract "
        "before validating provider facts");
  return validateRVVElementwiseArithmeticRoutePayloadFacts(
      context.route, context.description, *contract);
}

llvm::Error validateRVVElementwiseArithmeticRouteMetadataMirrorContract(
    const TargetArtifactCandidate &candidate,
    llvm::ArrayRef<
        plugin::rvv::RVVElementwiseArithmeticRouteMetadataMirrorContract>
        mirrors) {
  for (const plugin::rvv::RVVElementwiseArithmeticRouteMetadataMirrorContract
           &mirror : mirrors)
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, mirror.key, mirror.expected, mirror.label))
      return error;
  return llvm::Error::success();
}

llvm::Error validateRVVElementwiseArithmeticRouteEmptyMetadataMirrors(
    const TargetArtifactCandidate &candidate,
    llvm::ArrayRef<llvm::StringRef> staleMirrorKeys, llvm::StringRef label) {
  for (llvm::StringRef key : staleMirrorKeys)
    if (llvm::Error error =
            requireCandidateMetadataMirror(candidate, key, "", label))
      return error;
  return llvm::Error::success();
}

llvm::Error validateRVVProviderElementwiseArithmeticRouteMetadataMirrorContract(
    const TargetArtifactCandidate &candidate,
    const plugin::rvv::RVVElementwiseArithmeticRouteMetadataMirrorContractSet
        &contract) {
  if (llvm::Error error =
          validateRVVElementwiseArithmeticRouteMetadataMirrorContract(
              candidate, contract.mirrors))
    return error;
  return validateRVVElementwiseArithmeticRouteEmptyMetadataMirrors(
      candidate, contract.staleMirrorKeys, contract.staleMirrorLabel);
}

llvm::Error validateRVVElementwiseArithmeticTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  std::optional<
      plugin::rvv::RVVElementwiseArithmeticRouteMetadataMirrorContractSet>
      contract =
          plugin::rvv::getRVVElementwiseArithmeticRouteMetadataMirrorContract(
              context.description);
  if (!contract)
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires "
        "provider-owned elementwise arithmetic metadata mirror contract "
        "before validating candidate mirrors");
  return validateRVVProviderElementwiseArithmeticRouteMetadataMirrorContract(
      context.candidate, *contract);
}

llvm::Error validateRVVTargetOwnedRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef familyLabel) {
  if (description.runtimeABIOrder.empty() ||
      description.runtimeABIParameters.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(familyLabel) +
        " target artifact consumer requires provider-derived runtime ABI "
        "order and ABI parameters before artifact export");
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(familyLabel) +
        " target artifact consumer requires rebuilt provider route ABI "
        "mapping count to match provider runtime ABI parameters");
  for (size_t index = 0, count = description.runtimeABIParameters.size();
       index < count; ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        description.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(familyLabel) +
          " target artifact consumer requires rebuilt provider route ABI "
          "mapping to mirror provider runtime ABI parameters");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine(familyLabel) +
          " target artifact consumer requires rebuilt provider route ABI "
          "mapping value names to use provider runtime ABI parameter names");
  }
  return llvm::Error::success();
}

bool isRVVVectorReductionTargetArtifactRouteFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVVectorReductionRouteFamilyOperation(description.operation) &&
         description.memoryForm ==
             plugin::rvv::RVVSelectedBodyMemoryForm::VectorRHSLoad;
}

llvm::Error validateRVVVectorReductionRuntimeABIFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVVectorReductionRouteValidationContract &contract) {
  if (description.runtimeABIParameters.size() !=
      contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived runtime ABI parameters for lhs, rhs "
        "seed/accumulator, out, and n before artifact export");

  for (size_t index = 0, count = contract.runtimeABIParameters.size();
       index < count; ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    const support::RuntimeABIParameter &expected =
        contract.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(actual, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-derived runtime ABI parameter " +
          std::to_string(index) + " to bind " + expected.cName + " as " +
          support::stringifyRuntimeABIParameterRole(expected.role) +
          " before artifact export");
  }
  return llvm::Error::success();
}

llvm::Error validateRVVVectorReductionRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVVectorReductionRouteValidationContract &contract) {
  if (contract.requiredHeaderDeclarations.empty() ||
      contract.requiredHeaders.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived required_header_declarations before "
        "artifact export");
  for (llvm::StringRef header : contract.requiredHeaders) {
    if (header.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " saw an empty provider route header declaration");
    if (!routeHasHeader(route, header))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route header '" + header +
          "' before artifact export");
  }
  return llvm::Error::success();
}

llvm::Error validateRVVVectorReductionRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVVectorReductionRouteValidationContract &contract) {
  if (contract.cTypeMappingSummary.empty() || contract.typeMappings.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived route type mapping facts before artifact "
        "export");
  for (const plugin::rvv::RVVVectorReductionRouteTypeMappingContract &mapping :
       contract.typeMappings) {
    if (mapping.sourceType.empty() || mapping.cType.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-derived " + mapping.label +
          " facts before artifact export");
    if (!routeHasTypeMapping(route, mapping.sourceType, mapping.cType))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route type mapping '" +
          mapping.sourceType + "' -> '" + mapping.cType + "'");
  }
  return llvm::Error::success();
}

llvm::Error validateRVVVectorReductionRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVVectorReductionRouteValidationContract &contract) {
  if (contract.runtimeABIParameterRoles.size() !=
      contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned runtime ABI role order with one role per "
        "ABI parameter before artifact export");
  if (contract.runtimeABIOrder.empty() || contract.runtimeABIParameters.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived runtime ABI order and ABI parameters "
        "before artifact export");
  if (route.getABIMappings().size() != contract.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires rebuilt provider route ABI mapping count " +
        llvm::Twine(contract.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < contract.runtimeABIParameters.size();
       ++index) {
    support::RuntimeABIParameterRole expectedRole =
        contract.runtimeABIParameterRoles[index];
    support::RuntimeABIParameterRole actualRole =
        contract.runtimeABIParameters[index].role;
    if (actualRole != expectedRole)
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires provider-owned runtime ABI parameter[" +
          llvm::Twine(index) + "] role '" +
          support::stringifyRuntimeABIParameterRole(expectedRole) +
          "' but saw '" +
          support::stringifyRuntimeABIParameterRole(actualRole) +
          "' for parameter '" + contract.runtimeABIParameters[index].cName +
          "'");
  }

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        contract.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine(contract.consumerLabel) +
          " requires rebuilt provider route ABI mapping[" +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVVectorReductionRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVVectorReductionRouteValidationContract &contract) {
  llvm::StringRef consumerLabel = contract.consumerLabel;
  if (contract.runtimeABIParameters.size() < 4)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived lhs, rhs seed/accumulator, out, and n "
        "ABI parameters before validating route statements");
  const support::RuntimeABIParameter &lhsABI =
      contract.runtimeABIParameters[0];
  const support::RuntimeABIParameter &rhsABI =
      contract.runtimeABIParameters[1];
  const support::RuntimeABIParameter &outABI =
      contract.runtimeABIParameters[2];
  const support::RuntimeABIParameter &runtimeNABI =
      contract.runtimeABIParameters[3];
  if (contract.resultName.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived reduction result name before validating "
        "route statements");

  if (route.getCallOpaqueSteps().size() != contract.expectedPreLoopStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires exact provider-built pre-loop statement count " +
        llvm::Twine(contract.expectedPreLoopStepCount) +
        " before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          preLoopSetVL, consumerLabel, "pre-loop setvl",
          contract.setVLIntrinsic, {{runtimeNABI.cName, runtimeNABI.cType}},
          contract.emitCFullChunkVLName, contract.vlCType))
    return error;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "vector reduction target artifact consumer requires pre-loop "
          "statements to carry selected typed RVV source provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "vector reduction target artifact consumer requires exactly one "
        "provider-built runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  const support::RuntimeABIParameter *runtimeN = nullptr;
  for (const support::RuntimeABIParameter &parameter :
       contract.runtimeABIParameters)
    if (parameter.role ==
        support::RuntimeABIParameterRole::RuntimeElementCount) {
      runtimeN = &parameter;
      break;
    }
  if (!runtimeN)
    return makeRVVTargetRouteError(
        "vector reduction target artifact consumer requires a "
        "provider-derived runtime element count ABI parameter");
  if (loop.inductionVarName != contract.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != contract.vlCType ||
      loop.upperBound.expression != runtimeN->cName ||
      loop.upperBound.cType != runtimeN->cType ||
      loop.step.expression != contract.emitCFullChunkVLName ||
      loop.step.cType != contract.vlCType)
    return makeRVVTargetRouteError(
        "vector reduction target artifact consumer requires provider-built "
        "loop bounds and step to mirror runtime n/AVL/VL facts");
  if (runtimeN != &runtimeNABI)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires runtime n/AVL ABI role to match the selected ABI order "
        "before validating route statements");
  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeN->cName) + " - " +
       contract.emitCLoopInductionName)
          .str();
  if (loop.bodySteps.size() != contract.expectedLoopBodyStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires exact provider-built vector reduction loop statement count " +
        llvm::Twine(contract.expectedLoopBodyStepCount) +
        " before artifact export");
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[0], consumerLabel, "loop setvl",
          contract.setVLIntrinsic, {{expectedRemainingAVL, contract.vlCType}},
          contract.emitCLoopVLName, contract.vlCType))
    return error;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "vector reduction target artifact consumer requires loop statements "
          "to carry selected typed RVV source provenance");

  const std::string expectedLhsPointer =
      (llvm::StringRef(lhsABI.cName) + " + " +
       contract.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[1], consumerLabel, "lhs source vector load",
          contract.vectorLoadIntrinsic,
          {{expectedLhsPointer, lhsABI.cType},
           {contract.emitCLoopVLName, contract.vlCType}},
          "lhs_vec", contract.vectorCType))
    return error;

  const std::string expectedRhsPointer =
      (llvm::StringRef(rhsABI.cName) + " + " +
       contract.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[2], consumerLabel,
          "RHS seed/accumulator vector load", contract.vectorLoadIntrinsic,
          {{expectedRhsPointer, rhsABI.cType},
           {contract.emitCLoopVLName, contract.vlCType}},
          "rhs_vec", contract.vectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[3], consumerLabel, "reduce_add intrinsic",
          contract.intrinsic,
          {{"lhs_vec", contract.vectorCType},
           {"rhs_vec", contract.vectorCType},
           {contract.emitCLoopVLName, contract.vlCType}},
          contract.resultName, contract.vectorCType))
    return error;

  const std::string expectedOutPointer =
      (llvm::StringRef(outABI.cName) + " + " +
       contract.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[4], consumerLabel, "output store",
          contract.storeIntrinsic,
          {{expectedOutPointer, outABI.cType},
           {contract.resultName, contract.vectorCType},
           {contract.reductionStoreVL, contract.vlCType}}))
    return error;
  return llvm::Error::success();
}

llvm::Error validateRVVVectorReductionTypedRouteFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const plugin::rvv::RVVVectorReductionRouteValidationContract &contract) {
  if (description.operation != contract.operation)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned reduce_add operation before artifact export");
  if (description.emitCRouteID != contract.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned route id '" + contract.emitCRouteID +
        "' but description carried '" + description.emitCRouteID + "'");
  if (description.memoryForm != contract.memoryForm)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires vector RHS-load memory form before artifact export");
  if (description.typedComputeOpName != contract.typedComputeOpName)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires selected typed " + contract.typedComputeOpName +
        " body before artifact export");
  if (description.elementTypeName != contract.elementTypeName ||
      description.sew != contract.sew || description.lmul != contract.lmul ||
      description.tailPolicy != contract.tailPolicy ||
      description.maskPolicy != contract.maskPolicy ||
      description.configContractID != contract.configContractID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned dtype/config facts element '" +
        contract.elementTypeName + "', SEW " + llvm::Twine(contract.sew) +
        ", LMUL '" + contract.lmul + "', tail '" + contract.tailPolicy +
        "', mask '" + contract.maskPolicy + "', and config '" +
        contract.configContractID + "' before artifact export");
  if (description.providerSupportedMirror != contract.providerSupportedMirror ||
      description.targetLeafProfile != contract.targetLeafProfile ||
      description.requiredHeaderDeclarations !=
          contract.requiredHeaderDeclarations ||
      description.cTypeMappingSummary != contract.cTypeMappingSummary)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned support, header, C type, and "
        "target leaf facts before artifact export but saw support '" +
        description.providerSupportedMirror + "', header declarations '" +
        description.requiredHeaderDeclarations + "', C type mapping '" +
        description.cTypeMappingSummary + "', and target leaf profile '" +
        description.targetLeafProfile + "'");
  if (description.routeOperandBindingPlanID !=
      contract.routeOperandBindingPlanID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned operand binding plan '" +
        contract.routeOperandBindingPlanID + "' but saw '" +
        description.routeOperandBindingPlanID + "'");
  if (description.routeOperandBindingSummary !=
      contract.routeOperandBindingSummary)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-owned operand binding summary for lhs input, rhs "
        "seed/accumulator, scalar output slot, and runtime n but saw '" +
        description.routeOperandBindingSummary + "'");
  if (description.reductionAccumulatorLayout !=
          contract.reductionAccumulatorLayout ||
      description.reductionResultLayout != contract.reductionResultLayout ||
      description.reductionStoreVL != contract.reductionStoreVL)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived reduce_add layout facts accumulator '" +
        contract.reductionAccumulatorLayout + "', result '" +
        contract.reductionResultLayout + "', and store VL '" +
        contract.reductionStoreVL + "' before artifact export but provider "
        "carried accumulator '" +
        description.reductionAccumulatorLayout + "', result '" +
        description.reductionResultLayout + "', and store VL '" +
        description.reductionStoreVL + "'");
  if (description.vlCType != contract.vlCType ||
      description.vectorTypeName != contract.vectorTypeName ||
      description.vectorCType != contract.vectorCType)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived vector type facts VL '" +
        contract.vlCType + "', vector type '" + contract.vectorTypeName +
        "', and vector C type '" + contract.vectorCType +
        "' before artifact export");
  if (description.setVLIntrinsic != contract.setVLIntrinsic ||
      description.vectorLoadIntrinsic != contract.vectorLoadIntrinsic)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived setvl/load intrinsics '" +
        contract.setVLIntrinsic + "' and '" + contract.vectorLoadIntrinsic +
        "' before artifact export");
  if (description.intrinsic != contract.intrinsic)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived reduce_add intrinsic '" +
        contract.intrinsic + "' but saw '" + description.intrinsic + "'");
  if (description.storeIntrinsic != contract.storeIntrinsic)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived store intrinsic '" +
        contract.storeIntrinsic + "' but saw '" +
        description.storeIntrinsic + "'");
  if (description.resultName != contract.resultName)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived reduce_add result '" +
        contract.resultName + "' but saw '" + description.resultName + "'");
  if (description.emitCFullChunkVLName != contract.emitCFullChunkVLName ||
      description.emitCLoopVLName != contract.emitCLoopVLName ||
      description.emitCLoopInductionName != contract.emitCLoopInductionName)
    return makeRVVTargetRouteError(
        llvm::Twine(contract.consumerLabel) +
        " requires provider-derived AVL/VL statement-plan names full chunk '" +
        contract.emitCFullChunkVLName + "', loop VL '" +
        contract.emitCLoopVLName + "', and induction '" +
        contract.emitCLoopInductionName + "' before artifact export");
  return llvm::Error::success();
}

llvm::Error validateRVVVectorReductionRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  const std::optional<plugin::rvv::RVVVectorReductionRouteValidationContract>
      contract = plugin::rvv::getRVVVectorReductionRouteValidationContract(
          description);
  if (!contract)
    return makeRVVTargetRouteError(
        "vector reduction target artifact consumer requires provider-owned "
        "route validation contract before artifact export");

  if (llvm::Error error = validateRVVRuntimeAVLVLSelectedBoundaryContract(
          description, contract->runtimeAVLVLContract))
    return error;
  if (llvm::Error error = validateRVVRouteLocalRuntimeAVLVLMirrors(
          contract->consumerLabel, contract->runtimeAVLVLContract,
          contract->runtimeControlPlanID, contract->runtimeABIOrder,
          contract->setVLIntrinsic, contract->vlCType,
          contract->emitCFullChunkVLName, contract->emitCLoopVLName,
          contract->emitCLoopInductionName))
    return error;

  if (route.getRouteID() != contract->emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine(contract->consumerLabel) +
        " requires rebuilt provider route id '" + contract->emitCRouteID +
        "' but route carried '" +
        route.getRouteID() + "'");
  if (contract->routeOperandBindingPlanID.empty() ||
      contract->routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract->consumerLabel) +
        " requires provider route operand binding facts before artifact export");
  llvm::StringRef routeOperandBindingSummary(
      contract->routeOperandBindingSummary);
  if (!routeOperandBindingSummary.contains("lhs=lhs-input-buffer") ||
      !routeOperandBindingSummary.contains("rhs=rhs-input-buffer") ||
      !routeOperandBindingSummary.contains("out=output-buffer") ||
      !routeOperandBindingSummary.contains("n=runtime-element-count"))
    return makeRVVTargetRouteError(
        llvm::Twine(contract->consumerLabel) +
        " requires provider route operand binding facts for lhs input, rhs "
        "seed/accumulator, scalar output slot, and runtime n before artifact "
        "export");
  if (contract->runtimeABIOrder.empty() || contract->setVLIntrinsic.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(contract->consumerLabel) + " requires "
        "provider-derived runtime AVL/VL facts before artifact export");
  if (llvm::Error error =
          validateRVVVectorReductionTypedRouteFacts(description, *contract))
    return error;
  if (llvm::Error error =
          validateRVVVectorReductionRuntimeABIFacts(description, *contract))
    return error;
  if (!description.elementwiseArithmeticRouteFamilyPlanID.empty() ||
      !description.scalarBroadcastElementwiseRouteFamilyPlanID.empty() ||
      !description.runtimeScalarSplatStoreRouteFamilyPlanID.empty() ||
      !description.wideningConversionRouteFamilyPlanID.empty() ||
      !description.plainMAccRouteFamilyPlanID.empty() ||
      !description.scalarBroadcastMAccRouteFamilyPlanID.empty() ||
      !description.accumulationRouteFamilyPlanID.empty() ||
      !description.standaloneReductionRouteFamilyPlanID.empty() ||
      !description.contractionRouteFamilyPlanID.empty() ||
      !description.plainCompareSelectRouteFamilyPlanID.empty() ||
      !description.computedMaskSelectRouteFamilyPlanID.empty() ||
      !description.computedMaskMemoryRouteFamilyPlanID.empty() ||
      !description.segment2MemoryRouteFamilyPlanID.empty() ||
      !description.baseMemoryMovementRouteFamilyPlanID.empty() ||
      !description.wideningMAccRelation.empty() ||
      !description.wideningDotProductRelation.empty() ||
      !description.scalarSeedSplatIntrinsic.empty())
    return makeRVVTargetRouteError(
        "vector reduction target artifact consumer rejects stale non-vector "
        "reduction route-family facts");

  if (llvm::Error error =
          validateRVVVectorReductionRouteHeaders(route, *contract))
    return error;
  if (llvm::Error error =
          validateRVVVectorReductionRouteTypeMappings(route, *contract))
    return error;
  if (llvm::Error error =
          validateRVVVectorReductionRouteABIMappings(route, *contract))
    return error;
  return validateRVVVectorReductionRouteStatementPlan(route, *contract);
}

llvm::Error validateRVVVectorReductionTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVVectorReductionRoutePayloadFacts(context.route,
                                                    context.description);
}

llvm::Error requireEmptyVectorReductionStaleMirror(
    const TargetArtifactCandidate &candidate, llvm::StringRef key,
    llvm::StringRef label) {
  return requireCandidateMetadataMirror(candidate, key, "", label);
}

llvm::Error validateRVVVectorReductionTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description =
      context.description;
  const std::optional<plugin::rvv::RVVVectorReductionRouteValidationContract>
      contract = plugin::rvv::getRVVVectorReductionRouteValidationContract(
          description);
  if (!contract)
    return makeRVVTargetRouteError(
        "vector reduction target artifact consumer requires provider-owned "
        "route validation contract before validating candidate mirrors");

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "rvv_selected_body_typed_compute_op",
          contract->typedComputeOpName,
          "selected typed RVV vector reduction typed compute op"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_plan",
          contract->routeOperandBindingPlanID,
          "selected typed RVV vector reduction binding plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_operands",
          contract->routeOperandBindingSummary,
          "selected typed RVV vector reduction binding summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.provider_supported_mirror",
          contract->providerSupportedMirror,
          "selected typed RVV vector reduction provider support"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.target_leaf_profile",
          contract->targetLeafProfile,
          "selected typed RVV vector reduction target leaf profile"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.memory_form",
          plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
              contract->memoryForm),
          "selected typed RVV vector reduction memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_control_plan",
          contract->runtimeControlPlanID,
          "route-local runtime AVL/VL control plan mirror"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order",
          contract->runtimeABIOrder,
          "route-local runtime AVL/VL ABI order mirror"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.required_header_declarations",
          contract->requiredHeaderDeclarations,
          "selected typed RVV vector reduction route header requirements"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.c_type_mapping",
          contract->cTypeMappingSummary,
          "selected typed RVV vector reduction route type mapping summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.reduction_accumulator_layout",
          contract->reductionAccumulatorLayout,
          "selected typed RVV vector reduction accumulator layout"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.reduction_result_layout",
          contract->reductionResultLayout,
          "selected typed RVV vector reduction result layout"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.reduction_store_vl",
          contract->reductionStoreVL,
          "selected typed RVV vector reduction store VL"))
    return error;

  constexpr llvm::StringLiteral staleRouteFamilyMirrors[] = {
      "tcrv_rvv.elementwise_arithmetic_route_family_plan",
      "tcrv_rvv.scalar_broadcast_elementwise_route_family_plan",
      "tcrv_rvv.runtime_scalar_splat_store_route_family_plan",
      "tcrv_rvv.widening_conversion_route_family_plan",
      "tcrv_rvv.plain_compare_select_route_family_plan",
      "tcrv_rvv.computed_mask_select_route_family_plan",
      "tcrv_rvv.computed_mask_memory_route_family_plan",
      "tcrv_rvv.segment2_memory_route_family_plan",
      "tcrv_rvv.plain_macc_route_family_plan",
      "tcrv_rvv.scalar_broadcast_macc_route_family_plan",
      "tcrv_rvv.accumulation_route_family_plan",
      "tcrv_rvv.standalone_reduction_route_family_plan",
      "tcrv_rvv.contraction_route_family_plan",
      "tcrv_rvv.base_memory_movement_route_family_plan"};
  for (llvm::StringRef key : staleRouteFamilyMirrors)
    if (llvm::Error error = requireEmptyVectorReductionStaleMirror(
            candidate, key,
            "selected typed RVV non-vector-reduction route-family mirror"))
      return error;
  return llvm::Error::success();
}

bool isRVVWideningMAccContractionTargetArtifactRouteFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVWideningMAccContractionRouteFamilyOperation(
             description.operation) &&
         description.memoryForm ==
             plugin::rvv::RVVSelectedBodyMemoryForm::VectorRHSLoad;
}

llvm::Error validateRVVWideningMAccContractionTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  std::optional<plugin::rvv::RVVMAccRouteValidationContract> contract =
      plugin::rvv::getRVVMAccRouteValidationContract(context.description);
  if (!contract)
    return makeRVVTargetRouteError(
        "widening MAcc contraction target artifact consumer requires "
        "provider-owned MAcc route validation contract before validating "
        "provider facts");
  return validateRVVMAccRoutePayloadFacts(context.route, context.description,
                                          *contract);
}

llvm::Error validateRVVWideningMAccContractionTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  std::optional<plugin::rvv::RVVMAccRouteMetadataMirrorContractSet> contract =
      plugin::rvv::getRVVMAccRouteMetadataMirrorContract(context.description);
  if (!contract)
    return makeRVVTargetRouteError(
        "widening MAcc contraction target artifact consumer requires "
        "provider-owned MAcc metadata mirror contract before validating "
        "candidate mirrors");
  return validateRVVProviderMAccRouteMetadataMirrorContract(context.candidate,
                                                            *contract);
}

bool isRVVWideningDotReductionTargetArtifactRouteFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVWideningDotReductionRouteFamilyOperation(description.operation);
}

bool isRVVWideningProductTargetArtifactRouteFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return description.operation ==
         plugin::rvv::RVVSelectedBodyOperationKind::WideningProduct;
}

bool
isRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterTargetArtifactRouteFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return description.operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 RuntimeScalarComputedMaskIndexedGatherMAccScatter &&
         description.memoryForm ==
             plugin::rvv::RVVSelectedBodyMemoryForm::
                 RuntimeScalarComputedMaskIndexedGatherMAccScatter;
}

llvm::Error
validateRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  std::optional<
      plugin::rvv::RVVCompositeGatherMAccScatterRouteValidationContract>
      contract =
          plugin::rvv::getRVVCompositeGatherMAccScatterRouteValidationContract(
              context.description);
  if (!contract)
    return makeRVVTargetRouteError(
        "runtime-scalar indexed gather-MAcc-scatter target artifact consumer "
        "requires provider-owned composite route-family validation contract "
        "before artifact export");
  const plugin::rvv::RVVComputedMaskIndexedMemoryRouteValidationContract
      &indexedContract = contract->indexedMemoryContract;
  const plugin::rvv::RVVCompositeGatherMAccScatterResourceSelection
      &resourceSelection = contract->resourceSelection;
  if (context.description.compositeGatherMAccScatterRouteFamilyPlanID !=
          contract->routeFamilyPlanID ||
      context.description.compositeGatherMAccScatterTypedComputeChain !=
          contract->typedComputeChain)
    return makeRVVTargetRouteError(
        "runtime-scalar indexed gather-MAcc-scatter target artifact consumer "
        "requires provider-owned composite route-family plan and typed compute "
        "chain before artifact export");
  if (!resourceSelection.hasSelection || !resourceSelection.isLegal ||
      resourceSelection.selectedCandidateID.empty())
    return makeRVVTargetRouteError(
        "runtime-scalar indexed gather-MAcc-scatter target artifact consumer "
        "requires provider-consumed legal composite resource selection facts "
        "before artifact export");
  if (resourceSelection.peakLiveVectorGroups >
      resourceSelection.vectorRegisterBudget)
    return makeRVVTargetRouteError(
        llvm::Twine("runtime-scalar indexed gather-MAcc-scatter target "
                    "artifact consumer rejects composite resource peak live "
                    "vector-group estimate ") +
        llvm::Twine(resourceSelection.peakLiveVectorGroups) +
        " above vector register budget " +
        llvm::Twine(resourceSelection.vectorRegisterBudget));
  if (resourceSelection.operation !=
      "runtime_scalar_cmp_masked_indexed_gather_macc_scatter")
    return makeRVVTargetRouteError(
        "runtime-scalar indexed gather-MAcc-scatter target artifact consumer "
        "rejects composite resource facts whose operation does not match the "
        "provider-owned composite route-family contract");
  if (resourceSelection.memoryForm !=
      "runtime-scalar-computed-mask-indexed-gather-macc-scatter")
    return makeRVVTargetRouteError(
        "runtime-scalar indexed gather-MAcc-scatter target artifact consumer "
        "rejects composite resource facts whose memory form does not match "
        "the provider-owned composite route-family contract");
  if (resourceSelection.sew != indexedContract.sew ||
      resourceSelection.lmul != indexedContract.lmul ||
      resourceSelection.tailPolicy != indexedContract.tailPolicy ||
      resourceSelection.maskPolicy != indexedContract.maskPolicy)
    return makeRVVTargetRouteError(
        "runtime-scalar indexed gather-MAcc-scatter target artifact consumer "
        "requires composite resource dtype/config/policy facts to match the "
        "provider route-family contract");
  if (resourceSelection.runtimeABIOrder != indexedContract.runtimeABIOrder)
    return makeRVVTargetRouteError(
        "runtime-scalar indexed gather-MAcc-scatter target artifact consumer "
        "requires composite resource runtime ABI order to match the provider "
        "route-family contract");
  if (context.route.getRouteID() != indexedContract.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine(indexedContract.consumerLabel) +
        " requires rebuilt provider route id '" + indexedContract.emitCRouteID +
        "' but route carried '" + context.route.getRouteID() + "'");
  if (llvm::Error error =
          validateRVVComputedMaskIndexedMemoryDescriptionAgainstContract(
              context.description, indexedContract))
    return error;
  if (llvm::Error error =
          validateRVVComputedMaskIndexedMemoryRouteHeaders(context.route,
                                                          indexedContract))
    return error;
  if (llvm::Error error =
          validateRVVComputedMaskIndexedMemoryRouteTypeMappings(
              context.route, indexedContract))
    return error;
  if (llvm::Error error =
          validateRVVComputedMaskIndexedMemoryRouteABIMappings(context.route,
                                                              indexedContract))
    return error;
  return validateRVVComputedMaskIndexedMemoryRouteStatementPlanShape(
      context.route, indexedContract);
}

llvm::Error
validateRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  std::optional<plugin::rvv::RVVMemoryRouteMetadataMirrorContractSet>
      contract =
          plugin::rvv::
              getRVVComputedMaskIndexedMemoryRouteMetadataMirrorContract(
                  context.description);
  if (!contract)
    return makeRVVTargetRouteError(
        "runtime-scalar indexed gather-MAcc-scatter target artifact consumer "
        "requires provider-owned metadata mirror contract before validating "
        "candidate mirrors");
  return validateRVVProviderMemoryRouteMetadataMirrorContract(
      context.candidate, *contract);
}

bool isRVVMAccTargetArtifactRouteFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVMAccRouteFamilyOperation(description.operation);
}

bool isRVVCompareSelectMaskTargetArtifactRouteFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVCompareSelectMaskRouteFamilyOperation(description.operation);
}

bool isRVVConversionDtypePolicyTargetArtifactRouteFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVConversionDtypePolicyRouteFamilyOperation(description.operation);
}

bool isRVVSegment2MemoryTargetArtifactRouteFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVSegment2MemoryRouteFamilyOperation(description.operation);
}

bool isRVVBaseMemoryMovementTargetArtifactRouteFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVBaseMemoryMovementRouteFamilyOperation(description.operation);
}

bool isRVVElementwiseArithmeticTargetArtifactRouteFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVElementwiseArithmeticRouteFamilyDescription(description);
}

bool isRVVRuntimeScalarSplatStoreTargetArtifactRouteFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVRuntimeScalarSplatStoreRouteFamilyOperation(
      description.operation);
}

struct RVVTargetArtifactRouteFamilyValidator {
  llvm::StringLiteral familyName;
  bool (*isConsumer)(
      const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description);
  llvm::Error (*validateProviderFacts)(
      const RVVTargetArtifactRouteFamilyValidationContext &context);
  llvm::Error (*validateCandidateMirrors)(
      const RVVTargetArtifactRouteFamilyValidationContext &context);
};

llvm::ArrayRef<RVVTargetArtifactRouteFamilyValidator>
getRVVTargetArtifactRouteFamilyValidators() {
  static const RVVTargetArtifactRouteFamilyValidator validators[] = {
      {llvm::StringLiteral("compare-select-mask"),
       isRVVCompareSelectMaskTargetArtifactRouteFamilyConsumer,
       validateRVVCompareSelectMaskTargetArtifactProviderFacts,
       validateRVVCompareSelectMaskTargetArtifactCandidateMirrors},
      {llvm::StringLiteral("conversion-dtype-policy"),
       isRVVConversionDtypePolicyTargetArtifactRouteFamilyConsumer,
       validateRVVConversionDtypePolicyTargetArtifactProviderFacts,
       validateRVVConversionDtypePolicyTargetArtifactCandidateMirrors},
      {llvm::StringLiteral("segment2-memory"),
       isRVVSegment2MemoryTargetArtifactRouteFamilyConsumer,
       validateRVVSegment2MemoryTargetArtifactProviderFacts,
       validateRVVSegment2MemoryTargetArtifactCandidateMirrors},
      {llvm::StringLiteral("base-memory-movement"),
       isRVVBaseMemoryMovementTargetArtifactRouteFamilyConsumer,
       validateRVVBaseMemoryMovementTargetArtifactProviderFacts,
       validateRVVBaseMemoryMovementTargetArtifactCandidateMirrors},
      {llvm::StringLiteral("standalone-reduction-accumulation"),
       isRVVStandaloneReductionAccumulationTargetArtifactRouteFamilyConsumer,
       validateRVVStandaloneReductionAccumulationTargetArtifactProviderFacts,
       validateRVVStandaloneReductionAccumulationTargetArtifactCandidateMirrors},
      {llvm::StringLiteral("vector-reduction"),
       isRVVVectorReductionTargetArtifactRouteFamilyConsumer,
       validateRVVVectorReductionTargetArtifactProviderFacts,
       validateRVVVectorReductionTargetArtifactCandidateMirrors},
      {llvm::StringLiteral("elementwise-arithmetic"),
       isRVVElementwiseArithmeticTargetArtifactRouteFamilyConsumer,
       validateRVVElementwiseArithmeticTargetArtifactProviderFacts,
       validateRVVElementwiseArithmeticTargetArtifactCandidateMirrors},
      {llvm::StringLiteral("runtime-scalar-splat-store"),
       isRVVRuntimeScalarSplatStoreTargetArtifactRouteFamilyConsumer,
       validateRVVRuntimeScalarSplatStoreTargetArtifactProviderFacts,
       validateRVVRuntimeScalarSplatStoreTargetArtifactCandidateMirrors},
      {llvm::StringLiteral(
           "runtime-scalar-indexed-gather-macc-scatter"),
       isRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterTargetArtifactRouteFamilyConsumer,
       validateRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterTargetArtifactProviderFacts,
       validateRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterTargetArtifactCandidateMirrors},
      {llvm::StringLiteral("macc"),
       isRVVMAccTargetArtifactRouteFamilyConsumer,
       validateRVVMAccTargetArtifactProviderFacts,
       validateRVVMAccTargetArtifactCandidateMirrors},
      {llvm::StringLiteral("widening-macc-contraction"),
       isRVVWideningMAccContractionTargetArtifactRouteFamilyConsumer,
       validateRVVWideningMAccContractionTargetArtifactProviderFacts,
       validateRVVWideningMAccContractionTargetArtifactCandidateMirrors},
      {llvm::StringLiteral("low-precision-widening-product"),
       isRVVWideningProductTargetArtifactRouteFamilyConsumer,
       validateRVVWideningProductTargetArtifactProviderFacts,
       validateRVVWideningProductTargetArtifactCandidateMirrors},
      {llvm::StringLiteral("widening-dot-reduction"),
       isRVVWideningDotReductionTargetArtifactRouteFamilyConsumer,
       validateRVVWideningDotReductionTargetArtifactProviderFacts,
       validateRVVWideningDotReductionTargetArtifactCandidateMirrors},
  };
  return validators;
}

llvm::Error selectRVVTargetArtifactRouteFamilyValidator(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef validationKind,
    const RVVTargetArtifactRouteFamilyValidator *&selected) {
  selected = nullptr;
  for (const RVVTargetArtifactRouteFamilyValidator &validator :
       getRVVTargetArtifactRouteFamilyValidators()) {
    if (!validator.isConsumer(description))
      continue;
    if (selected)
      return makeRVVTargetRouteError(
          llvm::Twine("target artifact route-family validator registry matched "
                      "both '") +
          selected->familyName + "' and '" + validator.familyName +
          "' while checking " + validationKind);
    selected = &validator;
  }
  return llvm::Error::success();
}

llvm::Error makeMissingRVVTargetArtifactRouteFamilyValidatorError(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef validationKind) {
  return makeRVVTargetRouteError(
      llvm::Twine("no target artifact route-family validator owns selected "
                  "typed RVV route operation '") +
      plugin::rvv::stringifyRVVSelectedBodyOperationKind(
          description.operation) +
      "' with memory form '" +
      plugin::rvv::stringifyRVVSelectedBodyMemoryForm(description.memoryForm) +
      "' while checking " + validationKind);
}

} // namespace

llvm::Error validateRVVTargetArtifactRouteFamilyProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const RVVTargetArtifactRouteFamilyValidator *validator = nullptr;
  if (llvm::Error error = selectRVVTargetArtifactRouteFamilyValidator(
          context.description, "rebuilt provider facts", validator))
    return error;
  if (!validator)
    return makeMissingRVVTargetArtifactRouteFamilyValidatorError(
        context.description, "rebuilt provider facts");
  if (!validator->validateProviderFacts)
    return makeRVVTargetRouteError(
        llvm::Twine("target artifact route-family validator '") +
        validator->familyName + "' has no provider-fact validator");
  return validator->validateProviderFacts(context);
}

llvm::Error validateRVVTargetArtifactRouteFamilyCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const RVVTargetArtifactRouteFamilyValidator *validator = nullptr;
  if (llvm::Error error = selectRVVTargetArtifactRouteFamilyValidator(
          context.description, "candidate metadata mirrors", validator))
    return error;
  if (!validator)
    return makeMissingRVVTargetArtifactRouteFamilyValidatorError(
        context.description, "candidate metadata mirrors");
  if (!validator->validateCandidateMirrors)
    return makeRVVTargetRouteError(
        llvm::Twine("target artifact route-family validator '") +
        validator->familyName + "' has no candidate-mirror validator");
  return validator->validateCandidateMirrors(context);
}

} // namespace tianchenrv::target::rvv
