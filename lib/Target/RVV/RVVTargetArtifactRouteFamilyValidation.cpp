#include "TianChenRV/Target/RVV/RVVTargetArtifactRouteFamilyValidation.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "llvm/ADT/ArrayRef.h"
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

llvm::Error requireCandidateMetadataMirror(
    const TargetArtifactCandidate &candidate, llvm::StringRef key,
    llvm::StringRef expected, llvm::StringRef label) {
  llvm::StringRef actual = lookupCandidateMetadataValue(candidate, key);
  if (!expected.empty()) {
    if (actual.empty())
      return makeRVVTargetRouteError(llvm::Twine("candidate metadata must "
                                                 "carry ") +
                                     key + " provenance");
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

llvm::Error requireRVVProviderDerivedCount(llvm::StringRef consumerLabel,
                                           llvm::StringRef fieldLabel,
                                           std::int64_t value) {
  if (value != 0)
    return llvm::Error::success();
  return makeRVVTargetRouteError(llvm::Twine(consumerLabel) +
                                 " requires provider-derived " + fieldLabel +
                                 " before artifact export");
}

bool isRVVWideningDotReductionRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
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
             plugin::rvv::RVVSelectedBodyOperationKind::WideningDotReduceAdd ||
         operation == plugin::rvv::RVVSelectedBodyOperationKind::
                          StridedInputWideningDotReduceAdd;
}

bool isRVVStridedInputWideningDotReductionRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation == plugin::rvv::RVVSelectedBodyOperationKind::
                          StridedInputWideningDotReduceAdd ||
         operation == plugin::rvv::RVVSelectedBodyOperationKind::
                          ComputedMaskStridedInputWideningDotReduceAdd;
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

bool isRVVScalarBroadcastMAccRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
         plugin::rvv::RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd;
}

bool isRVVComputedMaskMAccRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd ||
         operation == plugin::rvv::RVVSelectedBodyOperationKind::
                          RuntimeScalarComputedMaskedMAccAdd;
}

bool isRVVRuntimeScalarComputedMaskMAccRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation == plugin::rvv::RVVSelectedBodyOperationKind::
                          RuntimeScalarComputedMaskedMAccAdd;
}

bool isRVVStandaloneReductionAccumulationRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::StandaloneReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::StandaloneReduceMin:
  case plugin::rvv::RVVSelectedBodyOperationKind::StandaloneReduceMax:
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
  return isRVVConversionDtypePolicyWideningRouteFamilyOperation(operation);
}

bool isRVVSegment2MemoryRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2LoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2StoreUnitLoad:
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

bool isRVVComputedMaskSegment2MemoryRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 ComputedMaskSegment2LoadUnitStore ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 ComputedMaskSegment2StoreUnitLoad ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 ComputedMaskSegment2UpdateUnitLoad;
}

bool isRVVPlainSegment2MemoryRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 Segment2DeinterleaveUnitStore ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 Segment2InterleaveUnitLoad;
}

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

bool isRVVIndexedBaseMemoryMovementRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 IndexedGatherUnitStore ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::IndexedScatterUnitLoad;
}

bool isRVVMaskedBaseMemoryMovementRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitLoadStore ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitStore;
}

plugin::rvv::RVVSelectedBodyMemoryForm
getRVVBaseMemoryMovementExpectedMemoryForm(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    return plugin::rvv::RVVSelectedBodyMemoryForm::StridedLoadUnitStore;
  case plugin::rvv::RVVSelectedBodyOperationKind::UnitLoadStridedStore:
    return plugin::rvv::RVVSelectedBodyMemoryForm::UnitLoadStridedStore;
  case plugin::rvv::RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
    return plugin::rvv::RVVSelectedBodyMemoryForm::IndexedLoadUnitStore;
  case plugin::rvv::RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    return plugin::rvv::RVVSelectedBodyMemoryForm::UnitLoadIndexedStore;
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    return plugin::rvv::RVVSelectedBodyMemoryForm::MaskedUnitLoadStore;
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitStore:
    return plugin::rvv::RVVSelectedBodyMemoryForm::MaskedUnitStore;
  default:
    llvm_unreachable("queried base memory movement memory form for non-base op");
  }
}

llvm::StringRef getRVVBaseMemoryMovementExpectedRuntimeABIOrder(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    return "src,out,n,stride_bytes";
  case plugin::rvv::RVVSelectedBodyOperationKind::UnitLoadStridedStore:
    return "src,dst,n,dst_stride_bytes";
  case plugin::rvv::RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
    return "data,index,out,n";
  case plugin::rvv::RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    return "src,index,dst,n";
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitStore:
    return "src,mask,dst,n";
  default:
    return {};
  }
}

llvm::StringRef getRVVBaseMemoryMovementOperationMnemonic(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return plugin::rvv::stringifyRVVSelectedBodyOperationKind(operation);
}

std::string getRVVBaseMemoryMovementExpectedRouteOperandBindingPlan(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return (llvm::Twine("rvv-route-operand-binding:") +
          getRVVBaseMemoryMovementOperationMnemonic(operation) + ".v1")
      .str();
}

std::string getRVVBaseMemoryMovementExpectedProviderSupportedMirror(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  llvm::StringRef mnemonic = getRVVBaseMemoryMovementOperationMnemonic(operation);
  std::string hyphenated = mnemonic.str();
  for (char &character : hyphenated)
    if (character == '_')
      character = '-';
  return (llvm::Twine("provider_supported_mirror:rvv-") + hyphenated +
          "-plan-validated")
      .str();
}

llvm::StringRef getRVVBaseMemoryMovementExpectedStridedLayout(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    return "byte-strided-source-unit-stride-output-runtime-abi";
  case plugin::rvv::RVVSelectedBodyOperationKind::UnitLoadStridedStore:
    return "unit-stride-source-byte-strided-destination-runtime-abi";
  default:
    return {};
  }
}

llvm::StringRef getRVVBaseMemoryMovementExpectedIndexedOrMaskedLayout(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
    return "element-indexed-data-index-unit-stride-output-runtime-abi";
  case plugin::rvv::RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    return "unit-stride-source-indexed-destination-index-runtime-abi";
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    return "unit-stride-source-mask-old-destination-runtime-abi";
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitStore:
    return "unit-stride-source-mask-destination-masked-store-runtime-abi";
  default:
    return {};
  }
}

llvm::StringRef getRVVBaseMemoryMovementExpectedSourceStrideSource(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation ==
      plugin::rvv::RVVSelectedBodyOperationKind::StridedLoadUnitStore)
    return "runtime_abi:stride_bytes";
  return {};
}

llvm::StringRef getRVVBaseMemoryMovementExpectedDestinationStrideSource(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation ==
      plugin::rvv::RVVSelectedBodyOperationKind::UnitLoadStridedStore)
    return "runtime_abi:dst_stride_bytes";
  return {};
}

llvm::StringRef getRVVBaseMemoryMovementExpectedSourceMemoryForm(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    return "strided-load";
  case plugin::rvv::RVVSelectedBodyOperationKind::UnitLoadStridedStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitStore:
    return "unit-stride-load";
  case plugin::rvv::RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
  default:
    return {};
  }
}

llvm::StringRef getRVVBaseMemoryMovementExpectedDestinationMemoryForm(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    return "indexed-store";
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitStore:
    return "masked-unit-store";
  case plugin::rvv::RVVSelectedBodyOperationKind::UnitLoadStridedStore:
    return "strided-store";
  default:
    return "unit-stride-store";
  }
}

int64_t getRVVBaseMemoryMovementExpectedIndexEEW(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return isRVVIndexedBaseMemoryMovementRouteFamilyOperation(operation) ? 32 : 0;
}

llvm::StringRef getRVVBaseMemoryMovementExpectedIndexSource(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return isRVVIndexedBaseMemoryMovementRouteFamilyOperation(operation)
             ? llvm::StringRef("runtime_abi:index")
             : llvm::StringRef();
}

llvm::StringRef getRVVBaseMemoryMovementExpectedOffsetUnit(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return isRVVIndexedBaseMemoryMovementRouteFamilyOperation(operation)
             ? llvm::StringRef("element")
             : llvm::StringRef();
}

llvm::StringRef getRVVBaseMemoryMovementExpectedIndexUniqueness(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation ==
      plugin::rvv::RVVSelectedBodyOperationKind::IndexedScatterUnitLoad)
    return "unique";
  return {};
}

llvm::StringRef getRVVBaseMemoryMovementExpectedIndexedDataMemoryForm(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation ==
      plugin::rvv::RVVSelectedBodyOperationKind::IndexedGatherUnitStore)
    return "indexed-load";
  return {};
}

llvm::StringRef getRVVBaseMemoryMovementExpectedIndexedDestinationMemoryForm(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation ==
      plugin::rvv::RVVSelectedBodyOperationKind::IndexedScatterUnitLoad)
    return "indexed-store";
  return {};
}

llvm::StringRef getRVVBaseMemoryMovementExpectedMaskRole(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return isRVVMaskedBaseMemoryMovementRouteFamilyOperation(operation)
             ? llvm::StringRef("predicate-mask-input-buffer")
             : llvm::StringRef();
}

llvm::StringRef getRVVBaseMemoryMovementExpectedMaskSource(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return isRVVMaskedBaseMemoryMovementRouteFamilyOperation(operation)
             ? llvm::StringRef("runtime_abi:mask")
             : llvm::StringRef();
}

llvm::StringRef getRVVBaseMemoryMovementExpectedMaskMemoryForm(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return isRVVMaskedBaseMemoryMovementRouteFamilyOperation(operation)
             ? llvm::StringRef("unit-stride-mask-load")
             : llvm::StringRef();
}

llvm::StringRef getRVVBaseMemoryMovementExpectedInactiveLaneContract(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    return "masked-off-lanes-preserve-old-destination";
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitStore:
    return "masked-store-false-lanes-preserve-output-buffer";
  default:
    return {};
  }
}

llvm::StringRef getRVVBaseMemoryMovementExpectedMaskedPassthroughLayout(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    return "old-destination-vector-preserves-inactive-lanes";
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitStore:
    return "masked-store-has-no-passthrough-load";
  default:
    return {};
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

llvm::Error validateRVVRuntimeScalarSplatStoreRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-derived required_header_declarations before accepting the "
        "route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires at "
        "least one provider route header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "runtime scalar splat-store target artifact consumer saw an empty "
          "provider route header declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("runtime scalar splat-store target artifact consumer "
                      "requires rebuilt provider route header '") +
          trimmed + "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVRuntimeScalarSplatStoreRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.vlCType.empty() || description.vectorTypeName.empty() ||
      description.vectorCType.empty() || description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-derived VL, vector, and C type mapping facts before "
        "artifact export");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("runtime scalar splat-store target artifact consumer "
                    "requires rebuilt provider route type mapping "
                    "'!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("runtime scalar splat-store target artifact consumer "
                    "requires rebuilt provider route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");

  return llvm::Error::success();
}

llvm::Error validateRVVRuntimeScalarSplatStoreRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("runtime scalar splat-store target artifact consumer "
                    "requires rebuilt provider route ABI mapping count ") +
        llvm::Twine(description.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        description.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine("runtime scalar splat-store target artifact consumer "
                      "requires rebuilt provider route ABI mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("runtime scalar splat-store target artifact consumer "
                      "requires rebuilt provider route ABI mapping[") +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVRuntimeScalarSplatStoreRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.runtimeABIParameters.size() != 3)
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-derived ABI order rhs_scalar,out,n before artifact export");
  const support::RuntimeABIParameter &rhsScalar =
      description.runtimeABIParameters[0];
  const support::RuntimeABIParameter &out = description.runtimeABIParameters[1];
  const support::RuntimeABIParameter &runtimeN =
      description.runtimeABIParameters[2];

  if (route.getCallOpaqueSteps().size() != 1)
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires exactly "
        "one provider-built pre-loop setvl statement before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      preLoopSetVL.operands.size() != 1 ||
      preLoopSetVL.operands.front().expression != runtimeN.cName ||
      preLoopSetVL.operands.front().cType != runtimeN.cType ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires rebuilt "
        "provider route pre-loop setvl statement to use runtime n/AVL and "
        "define the full-chunk VL");
  if (!routeStepSourceIsSelectedRVVBody(preLoopSetVL))
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "pre-loop setvl to carry selected typed RVV source provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires exactly "
        "one provider-built runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.upperBound.expression != runtimeN.cName ||
      loop.upperBound.cType != runtimeN.cType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-built loop bounds and step to mirror runtime n/AVL/VL facts");
  if (loop.bodySteps.size() != 3)
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-built loop setvl, runtime scalar splat, and store "
        "statements before artifact export");

  const conversion::emitc::TCRVEmitCCallOpaqueStep &loopSetVL =
      loop.bodySteps[0];
  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeN.cName) + " - " +
       description.emitCLoopInductionName)
          .str();
  if (loopSetVL.callee != description.setVLIntrinsic ||
      loopSetVL.operands.size() != 1 ||
      loopSetVL.operands.front().expression != expectedRemainingAVL ||
      loopSetVL.operands.front().cType != description.vlCType ||
      !stepHasResult(loopSetVL, description.emitCLoopVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-built loop setvl to derive per-iteration VL from remaining "
        "runtime AVL");

  const conversion::emitc::TCRVEmitCCallOpaqueStep &splat = loop.bodySteps[1];
  if (splat.callee != description.rhsBroadcastIntrinsic ||
      splat.operands.size() != 2 ||
      splat.operands[0].expression != rhsScalar.cName ||
      splat.operands[0].cType != rhsScalar.cType ||
      splat.operands[1].expression != description.emitCLoopVLName ||
      splat.operands[1].cType != description.vlCType ||
      !stepHasResult(splat, description.resultName, description.vectorCType))
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-built runtime scalar splat statement to consume rhs_scalar "
        "and per-iteration VL");

  const conversion::emitc::TCRVEmitCCallOpaqueStep &store = loop.bodySteps[2];
  const std::string expectedOutPointer =
      (llvm::StringRef(out.cName) + " + " +
       description.emitCLoopInductionName)
          .str();
  if (store.callee != description.storeIntrinsic ||
      store.operands.size() != 3 ||
      store.operands[0].expression != expectedOutPointer ||
      store.operands[0].cType != out.cType ||
      store.operands[1].expression != description.resultName ||
      store.operands[1].cType != description.vectorCType ||
      store.operands[2].expression != description.emitCLoopVLName ||
      store.operands[2].cType != description.vlCType)
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-built store statement to consume out+offset, splat vector, "
        "and per-iteration VL");

  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "runtime scalar splat-store target artifact consumer requires loop "
          "statements to carry selected typed RVV source provenance");

  return llvm::Error::success();
}

llvm::Error validateRVVRuntimeScalarSplatStoreRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("runtime scalar splat-store target artifact consumer "
                    "requires rebuilt provider route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");
  if (description.memoryForm !=
      plugin::rvv::RVVSelectedBodyMemoryForm::RuntimeScalarSplatStore)
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "selected typed RVV memory form runtime-scalar-splat-store before "
        "artifact export");
  if (description.providerSupportedMirror.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-supported mirror facts before artifact export");
  if (description.runtimeScalarSplatStoreRouteFamilyPlanID.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "runtime-scalar splat-store route-family plan facts before artifact "
        "export");
  if (description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "route operand-binding mirror facts before artifact export");
  if (description.runtimeControlPlanID.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-derived runtime AVL/VL control plan facts before artifact "
        "export");
  if (description.runtimeABIOrder != "rhs_scalar,out,n")
    return makeRVVTargetRouteError(
        llvm::Twine("runtime scalar splat-store target artifact consumer "
                    "requires provider-derived runtime ABI order "
                    "rhs_scalar,out,n but was '") +
        description.runtimeABIOrder + "'");
  if (description.sew == 0 || description.lmul.empty() ||
      description.tailPolicy.empty() || description.maskPolicy.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-derived dtype and policy facts before artifact export");
  if (description.setVLIntrinsic.empty() ||
      description.rhsBroadcastIntrinsic.empty() ||
      description.storeIntrinsic.empty() || description.resultName.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-derived setvl, runtime scalar splat, store, and result facts "
        "before artifact export");
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
        "runtime scalar splat-store target artifact consumer must reject stale "
        "non-splat-store route-family facts");

  if (llvm::Error error =
          validateRVVRuntimeScalarSplatStoreRouteHeaders(route, description))
    return error;
  if (llvm::Error error =
          validateRVVRuntimeScalarSplatStoreRouteTypeMappings(route,
                                                              description))
    return error;
  if (llvm::Error error =
          validateRVVRuntimeScalarSplatStoreRouteABIMappings(route,
                                                             description))
    return error;
  return validateRVVRuntimeScalarSplatStoreRouteStatementPlan(route,
                                                              description);
}

llvm::Error validateRVVRuntimeScalarSplatStoreTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVRuntimeScalarSplatStoreRoutePayloadFacts(
      context.route, context.description);
}

llvm::Error validateRVVRuntimeScalarSplatStoreTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description =
      context.description;

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_plan",
          description.routeOperandBindingPlanID,
          "selected typed RVV runtime scalar splat-store binding plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_operands",
          description.routeOperandBindingSummary,
          "selected typed RVV runtime scalar splat-store binding summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.provider_supported_mirror",
          description.providerSupportedMirror,
          "selected typed RVV runtime scalar splat-store provider support"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_scalar_splat_store_route_family_plan",
          description.runtimeScalarSplatStoreRouteFamilyPlanID,
          "selected typed RVV runtime scalar splat-store route-family plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.memory_form",
          plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
              description.memoryForm),
          "selected typed RVV runtime scalar splat-store memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.target_leaf_profile",
          description.targetLeafProfile,
          "selected typed RVV runtime scalar splat-store target leaf profile"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_control_plan",
          description.runtimeControlPlanID,
          "selected typed RVV runtime scalar splat-store runtime AVL/VL "
          "control plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order", description.runtimeABIOrder,
          "selected typed RVV runtime scalar splat-store runtime ABI order"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.required_header_declarations",
          description.requiredHeaderDeclarations,
          "selected typed RVV runtime scalar splat-store route header "
          "requirements"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.c_type_mapping",
          description.cTypeMappingSummary,
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

bool bindingSummaryUseListContains(llvm::StringRef useList,
                                   llvm::StringRef expectedUse) {
  llvm::SmallVector<llvm::StringRef, 8> uses;
  useList.split(uses, '|', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (llvm::StringRef use : uses)
    if (use == expectedUse)
      return true;
  return false;
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
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  using OperationKind = plugin::rvv::RVVSelectedBodyOperationKind;
  if (description.operation != OperationKind::IndexedGatherUnitStore &&
      description.operation != OperationKind::IndexedScatterUnitLoad)
    return llvm::Error::success();
  const llvm::StringRef operationMnemonic =
      plugin::rvv::stringifyRVVSelectedBodyOperationKind(description.operation);
  if (description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary is required before artifact export");
  constexpr llvm::StringLiteral gatherLogicalOperands[] = {"data", "index",
                                                           "out", "n"};
  constexpr llvm::StringLiteral scatterLogicalOperands[] = {"src", "index",
                                                            "dst", "n"};
  llvm::ArrayRef<llvm::StringLiteral> logicalOperands =
      description.operation == OperationKind::IndexedGatherUnitStore
          ? llvm::ArrayRef<llvm::StringLiteral>(gatherLogicalOperands)
          : llvm::ArrayRef<llvm::StringLiteral>(scatterLogicalOperands);
  constexpr std::size_t logicalOperandCount =
      sizeof(gatherLogicalOperands) / sizeof(gatherLogicalOperands[0]);
  if (description.runtimeABIParameters.size() != logicalOperandCount)
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary requires the provider runtime ABI "
        "order before artifact export");
  for (std::size_t index = 0; index < logicalOperandCount; ++index)
    if (llvm::Error error = requireIndexedBaseMemoryHeaderBindingSummaryEntry(
            description, operationMnemonic, logicalOperands[index],
            description.runtimeABIParameters[index]))
      return error;
  return llvm::Error::success();
}

llvm::Error validateComputedMaskIndexedGatherHeaderBindingSummary(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  using OperationKind = plugin::rvv::RVVSelectedBodyOperationKind;
  if (description.operation !=
      OperationKind::ComputedMaskIndexedGatherLoadUnitStore)
    return llvm::Error::success();

  constexpr llvm::StringLiteral kExpectedPlan(
      "rvv-route-operand-binding:computed_masked_indexed_gather_load_unit_store.v1");
  const llvm::StringRef operationMnemonic =
      plugin::rvv::stringifyRVVSelectedBodyOperationKind(description.operation);
  if (llvm::StringRef(description.routeOperandBindingPlanID) != kExpectedPlan)
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary requires provider plan '" +
        kExpectedPlan + "' before artifact export");
  if (description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary is required before artifact export");
  if (!llvm::StringRef(description.routeOperandBindingSummary)
           .starts_with(kExpectedPlan))
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary must start with provider plan '" +
        kExpectedPlan + "' before artifact export");

  constexpr llvm::StringLiteral logicalOperands[] = {
      "cmp_lhs", "cmp_rhs", "src", "index", "dst", "n"};
  constexpr std::size_t logicalOperandCount =
      sizeof(logicalOperands) / sizeof(logicalOperands[0]);
  if (description.runtimeABIParameters.size() != logicalOperandCount)
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary requires the provider runtime ABI "
        "order for cmp_lhs/cmp_rhs/src/index/dst/n before artifact export");

  for (std::size_t index = 0; index < logicalOperandCount; ++index)
    if (llvm::Error error = requireIndexedBaseMemoryHeaderBindingSummaryEntry(
            description, operationMnemonic, logicalOperands[index],
            description.runtimeABIParameters[index]))
      return error;
  return llvm::Error::success();
}

llvm::Error validateComputedMaskIndexedScatterHeaderBindingSummary(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  using OperationKind = plugin::rvv::RVVSelectedBodyOperationKind;
  if (description.operation !=
      OperationKind::ComputedMaskIndexedScatterStoreUnitLoad)
    return llvm::Error::success();

  constexpr llvm::StringLiteral kExpectedPlan(
      "rvv-route-operand-binding:computed_masked_indexed_scatter_store_unit_load.v1");
  const llvm::StringRef operationMnemonic =
      plugin::rvv::stringifyRVVSelectedBodyOperationKind(description.operation);
  if (llvm::StringRef(description.routeOperandBindingPlanID) != kExpectedPlan)
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary requires provider plan '" +
        kExpectedPlan + "' before artifact export");
  if (description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary is required before artifact export");
  if (!llvm::StringRef(description.routeOperandBindingSummary)
           .starts_with(kExpectedPlan))
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary must start with provider plan '" +
        kExpectedPlan + "' before artifact export");

  constexpr llvm::StringLiteral logicalOperands[] = {
      "cmp_lhs", "cmp_rhs", "src", "index", "dst", "n"};
  constexpr std::size_t logicalOperandCount =
      sizeof(logicalOperands) / sizeof(logicalOperands[0]);
  if (description.runtimeABIParameters.size() != logicalOperandCount)
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary requires the provider runtime ABI "
        "order for cmp_lhs/cmp_rhs/src/index/dst/n before artifact export");

  for (std::size_t index = 0; index < logicalOperandCount; ++index)
    if (llvm::Error error = requireIndexedBaseMemoryHeaderBindingSummaryEntry(
            description, operationMnemonic, logicalOperands[index],
            description.runtimeABIParameters[index]))
      return error;
  return llvm::Error::success();
}

llvm::Error validateComputedMaskSegment2HeaderBindingSummary(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  using OperationKind = plugin::rvv::RVVSelectedBodyOperationKind;

  constexpr llvm::StringLiteral loadPlan(
      "rvv-route-operand-binding:computed_masked_segment2_load_unit_store.v1");
  constexpr llvm::StringLiteral storePlan(
      "rvv-route-operand-binding:computed_masked_segment2_store_unit_load.v1");
  constexpr llvm::StringLiteral updatePlan(
      "rvv-route-operand-binding:cmseg2_update_unit_load.v1");
  constexpr llvm::StringLiteral loadLogicalOperands[] = {
      "cmp_lhs", "cmp_rhs", "src", "out0", "out1", "n"};
  constexpr llvm::StringLiteral storeLogicalOperands[] = {
      "cmp_lhs", "cmp_rhs", "src0", "src1", "dst", "n"};

  llvm::StringRef expectedPlan;
  llvm::ArrayRef<llvm::StringLiteral> logicalOperands;
  switch (description.operation) {
  case OperationKind::ComputedMaskSegment2LoadUnitStore:
    expectedPlan = loadPlan;
    logicalOperands = llvm::ArrayRef<llvm::StringLiteral>(loadLogicalOperands);
    break;
  case OperationKind::ComputedMaskSegment2StoreUnitLoad:
    expectedPlan = storePlan;
    logicalOperands =
        llvm::ArrayRef<llvm::StringLiteral>(storeLogicalOperands);
    break;
  case OperationKind::ComputedMaskSegment2UpdateUnitLoad:
    expectedPlan = updatePlan;
    logicalOperands =
        llvm::ArrayRef<llvm::StringLiteral>(storeLogicalOperands);
    break;
  default:
    return llvm::Error::success();
  }

  const llvm::StringRef operationMnemonic =
      plugin::rvv::stringifyRVVSelectedBodyOperationKind(description.operation);
  if (llvm::StringRef(description.routeOperandBindingPlanID) != expectedPlan)
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary requires provider plan '" +
        expectedPlan + "' before artifact export");
  if (description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary is required before artifact export");
  if (!llvm::StringRef(description.routeOperandBindingSummary)
           .starts_with(expectedPlan))
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary must start with provider plan '" +
        expectedPlan + "' before artifact export");

  if (description.runtimeABIParameters.size() != logicalOperands.size())
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary requires the provider runtime ABI "
        "order before artifact export");

  for (std::size_t index = 0; index < logicalOperands.size(); ++index)
    if (llvm::Error error = requireIndexedBaseMemoryHeaderBindingSummaryEntry(
            description, operationMnemonic, logicalOperands[index],
            description.runtimeABIParameters[index]))
      return error;
  return llvm::Error::success();
}

llvm::Error validatePlainSegment2DeinterleaveHeaderBindingSummary(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  using OperationKind = plugin::rvv::RVVSelectedBodyOperationKind;
  if (description.operation != OperationKind::Segment2DeinterleaveUnitStore)
    return llvm::Error::success();

  constexpr llvm::StringLiteral kExpectedPlan(
      "rvv-route-operand-binding:segment2_deinterleave_unit_store.v1");
  const llvm::StringRef operationMnemonic =
      plugin::rvv::stringifyRVVSelectedBodyOperationKind(description.operation);
  if (llvm::StringRef(description.routeOperandBindingPlanID) != kExpectedPlan)
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary requires provider plan '" +
        kExpectedPlan + "' before artifact export");
  if (description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary is required before artifact export");
  if (!llvm::StringRef(description.routeOperandBindingSummary)
           .starts_with(kExpectedPlan))
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary must start with provider plan '" +
        kExpectedPlan + "' before artifact export");

  constexpr llvm::StringLiteral logicalOperands[] = {"src", "out0", "out1",
                                                     "n"};
  constexpr std::size_t logicalOperandCount =
      sizeof(logicalOperands) / sizeof(logicalOperands[0]);
  if (description.runtimeABIParameters.size() != logicalOperandCount)
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary requires the provider runtime ABI "
        "order for src/out0/out1/n before artifact export");

  for (std::size_t index = 0; index < logicalOperandCount; ++index)
    if (llvm::Error error = requireIndexedBaseMemoryHeaderBindingSummaryEntry(
            description, operationMnemonic, logicalOperands[index],
            description.runtimeABIParameters[index]))
      return error;
  return llvm::Error::success();
}

llvm::Error validatePlainSegment2InterleaveHeaderBindingSummary(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  using OperationKind = plugin::rvv::RVVSelectedBodyOperationKind;
  if (description.operation != OperationKind::Segment2InterleaveUnitLoad)
    return llvm::Error::success();

  constexpr llvm::StringLiteral kExpectedPlan(
      "rvv-route-operand-binding:segment2_interleave_unit_load.v1");
  const llvm::StringRef operationMnemonic =
      plugin::rvv::stringifyRVVSelectedBodyOperationKind(description.operation);
  if (llvm::StringRef(description.routeOperandBindingPlanID) != kExpectedPlan)
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary requires provider plan '" +
        kExpectedPlan + "' before artifact export");
  if (description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary is required before artifact export");
  if (!llvm::StringRef(description.routeOperandBindingSummary)
           .starts_with(kExpectedPlan))
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary must start with provider plan '" +
        kExpectedPlan + "' before artifact export");

  constexpr llvm::StringLiteral logicalOperands[] = {"src0", "src1", "dst",
                                                     "n"};
  constexpr std::size_t logicalOperandCount =
      sizeof(logicalOperands) / sizeof(logicalOperands[0]);
  if (description.runtimeABIParameters.size() != logicalOperandCount)
    return makeRVVTargetRouteError(
        llvm::Twine(operationMnemonic) +
        " route operand binding summary requires the provider runtime ABI "
        "order for src0/src1/dst/n before artifact export");

  for (std::size_t index = 0; index < logicalOperandCount; ++index)
    if (llvm::Error error = requireIndexedBaseMemoryHeaderBindingSummaryEntry(
            description, operationMnemonic, logicalOperands[index],
            description.runtimeABIParameters[index]))
      return error;
  return llvm::Error::success();
}

llvm::Error validateRVVBaseMemoryMovementRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "base-memory-movement target artifact consumer requires "
        "provider-derived required_header_declarations before accepting the "
        "route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "base-memory-movement target artifact consumer requires at least one "
        "provider route header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "base-memory-movement target artifact consumer saw an empty "
          "provider route header declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("base-memory-movement target artifact consumer requires "
                      "rebuilt provider route header '") +
          trimmed + "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVBaseMemoryMovementRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.vlCType.empty() || description.vectorTypeName.empty() ||
      description.vectorCType.empty() || description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "base-memory-movement target artifact consumer requires "
        "provider-derived VL, vector, and C type mapping facts before "
        "artifact export");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("base-memory-movement target artifact consumer requires "
                    "rebuilt provider route type mapping '!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("base-memory-movement target artifact consumer requires "
                    "rebuilt provider route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");

  if (isRVVIndexedBaseMemoryMovementRouteFamilyOperation(
          description.operation)) {
    if (description.indexVectorTypeName.empty() ||
        description.indexVectorCType.empty())
      return makeRVVTargetRouteError(
          "indexed base-memory-movement target artifact consumer requires "
          "provider-derived index vector type mapping facts before artifact "
          "export");
    if (!routeHasTypeMapping(route, description.indexVectorTypeName,
                             description.indexVectorCType))
      return makeRVVTargetRouteError(
          llvm::Twine("indexed base-memory-movement target artifact consumer "
                      "requires rebuilt provider route type mapping '") +
          description.indexVectorTypeName + "' -> '" +
          description.indexVectorCType + "'");
  } else if (!description.indexVectorTypeName.empty() ||
             !description.indexVectorCType.empty()) {
    return makeRVVTargetRouteError(
        "non-indexed base-memory-movement target artifact consumer rejects "
        "stale index vector type mapping facts");
  }

  if (isRVVMaskedBaseMemoryMovementRouteFamilyOperation(
          description.operation)) {
    if (description.maskTypeName.empty() || description.maskCType.empty())
      return makeRVVTargetRouteError(
          "masked base-memory-movement target artifact consumer requires "
          "provider-derived mask type mapping facts before artifact export");
    if (!routeHasTypeMapping(route, description.maskTypeName,
                             description.maskCType))
      return makeRVVTargetRouteError(
          llvm::Twine("masked base-memory-movement target artifact consumer "
                      "requires rebuilt provider route type mapping '") +
          description.maskTypeName + "' -> '" + description.maskCType + "'");
  } else if (!description.maskTypeName.empty() ||
             !description.maskCType.empty()) {
    return makeRVVTargetRouteError(
        "non-masked base-memory-movement target artifact consumer rejects "
        "stale mask type mapping facts");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVBaseMemoryMovementRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.runtimeABIOrder.empty() ||
      description.runtimeABIParameters.empty())
    return makeRVVTargetRouteError(
        "base-memory-movement target artifact consumer requires "
        "provider-derived runtime ABI order and ABI parameters before "
        "artifact export");
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("base-memory-movement target artifact consumer requires "
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
          llvm::Twine("base-memory-movement target artifact consumer requires "
                      "rebuilt provider route ABI mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("base-memory-movement target artifact consumer requires "
                      "rebuilt provider route ABI mapping[") +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVBaseMemoryMovementRuntimeABIFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  struct ExpectedRuntimeABIParameterRole {
    llvm::StringRef cName;
    support::RuntimeABIParameterRole role;
  };

  using OperationKind = plugin::rvv::RVVSelectedBodyOperationKind;
  const ExpectedRuntimeABIParameterRole stridedLoadRoles[] = {
      {"src", support::RuntimeABIParameterRole::SourceInputBuffer},
      {"out", support::RuntimeABIParameterRole::OutputBuffer},
      {"n", support::RuntimeABIParameterRole::RuntimeElementCount},
      {"stride_bytes", support::RuntimeABIParameterRole::SourceByteStride},
  };
  const ExpectedRuntimeABIParameterRole unitLoadStridedStoreRoles[] = {
      {"src", support::RuntimeABIParameterRole::LHSInputBuffer},
      {"dst", support::RuntimeABIParameterRole::OutputBuffer},
      {"n", support::RuntimeABIParameterRole::RuntimeElementCount},
      {"dst_stride_bytes",
       support::RuntimeABIParameterRole::DestinationByteStride},
  };
  const ExpectedRuntimeABIParameterRole indexedRoles[] = {
      {"data", support::RuntimeABIParameterRole::LHSInputBuffer},
      {"index", support::RuntimeABIParameterRole::IndexInputBuffer},
      {"out", support::RuntimeABIParameterRole::OutputBuffer},
      {"n", support::RuntimeABIParameterRole::RuntimeElementCount},
  };
  const ExpectedRuntimeABIParameterRole indexedScatterRoles[] = {
      {"src", support::RuntimeABIParameterRole::LHSInputBuffer},
      {"index", support::RuntimeABIParameterRole::IndexInputBuffer},
      {"dst", support::RuntimeABIParameterRole::OutputBuffer},
      {"n", support::RuntimeABIParameterRole::RuntimeElementCount},
  };
  const ExpectedRuntimeABIParameterRole maskedRoles[] = {
      {"src", support::RuntimeABIParameterRole::LHSInputBuffer},
      {"mask", support::RuntimeABIParameterRole::MaskInputBuffer},
      {"dst", support::RuntimeABIParameterRole::OutputBuffer},
      {"n", support::RuntimeABIParameterRole::RuntimeElementCount},
  };

  llvm::ArrayRef<ExpectedRuntimeABIParameterRole> expectedRoles;
  switch (description.operation) {
  case OperationKind::StridedLoadUnitStore:
    expectedRoles = stridedLoadRoles;
    break;
  case OperationKind::UnitLoadStridedStore:
    expectedRoles = unitLoadStridedStoreRoles;
    break;
  case OperationKind::IndexedGatherUnitStore:
    expectedRoles = indexedRoles;
    break;
  case OperationKind::IndexedScatterUnitLoad:
    expectedRoles = indexedScatterRoles;
    break;
  case OperationKind::MaskedUnitLoadStore:
  case OperationKind::MaskedUnitStore:
    expectedRoles = maskedRoles;
    break;
  default:
    llvm_unreachable("validated non-base-memory operation as base memory");
  }

  if (description.runtimeABIParameters.size() != expectedRoles.size())
    return makeRVVTargetRouteError(
        llvm::Twine("base-memory-movement target artifact consumer requires "
                    "provider-derived runtime ABI parameter count ") +
        llvm::Twine(expectedRoles.size()) + " for operation '" +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            description.operation) +
        "' but saw " + llvm::Twine(description.runtimeABIParameters.size()));

  for (std::size_t index = 0; index < expectedRoles.size(); ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    if (actual.cName != expectedRoles[index].cName ||
        actual.role != expectedRoles[index].role)
      return makeRVVTargetRouteError(
          llvm::Twine("base-memory-movement target artifact consumer requires "
                      "provider-derived runtime ABI parameter[") +
          llvm::Twine(index) + "] to bind '" + expectedRoles[index].cName +
          "' as " +
          support::stringifyRuntimeABIParameterRole(expectedRoles[index].role) +
          " before validating route statements");
  }

  return validateIndexedBaseMemoryHeaderBindingSummary(description);
}

llvm::Error validateRVVBaseMemoryMovementRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  constexpr llvm::StringLiteral consumerLabel(
      "base-memory-movement target artifact consumer");

  if (llvm::Error error =
          validateRVVBaseMemoryMovementRuntimeABIFacts(description))
    return error;
  if (description.vlCType.empty() || description.vectorCType.empty() ||
      description.resultName.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived result, vector C type, and VL C type "
        "facts before validating route statements");

  using OperationKind = plugin::rvv::RVVSelectedBodyOperationKind;
  const OperationKind operation = description.operation;
  const support::RuntimeABIParameter *sourceABI =
      &description.runtimeABIParameters[0];
  const support::RuntimeABIParameter *indexABI = nullptr;
  const support::RuntimeABIParameter *maskABI = nullptr;
  const support::RuntimeABIParameter *destinationABI = nullptr;
  const support::RuntimeABIParameter *runtimeNABI = nullptr;
  const support::RuntimeABIParameter *sourceStrideABI = nullptr;
  const support::RuntimeABIParameter *destinationStrideABI = nullptr;

  switch (operation) {
  case OperationKind::StridedLoadUnitStore:
    destinationABI = &description.runtimeABIParameters[1];
    runtimeNABI = &description.runtimeABIParameters[2];
    sourceStrideABI = &description.runtimeABIParameters[3];
    break;
  case OperationKind::UnitLoadStridedStore:
    destinationABI = &description.runtimeABIParameters[1];
    runtimeNABI = &description.runtimeABIParameters[2];
    destinationStrideABI = &description.runtimeABIParameters[3];
    break;
  case OperationKind::IndexedGatherUnitStore:
    indexABI = &description.runtimeABIParameters[1];
    destinationABI = &description.runtimeABIParameters[2];
    runtimeNABI = &description.runtimeABIParameters[3];
    break;
  case OperationKind::IndexedScatterUnitLoad:
    indexABI = &description.runtimeABIParameters[1];
    destinationABI = &description.runtimeABIParameters[2];
    runtimeNABI = &description.runtimeABIParameters[3];
    break;
  case OperationKind::MaskedUnitLoadStore:
  case OperationKind::MaskedUnitStore:
    maskABI = &description.runtimeABIParameters[1];
    destinationABI = &description.runtimeABIParameters[2];
    runtimeNABI = &description.runtimeABIParameters[3];
    break;
  default:
    llvm_unreachable("validated non-base-memory operation as base memory");
  }

  const support::RuntimeABIParameter *runtimeElementCount =
      findRuntimeElementCountABIParameter(description);
  if (!runtimeElementCount || runtimeElementCount != runtimeNABI)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires runtime n/AVL ABI role to match the selected base-memory "
        "ABI order before validating route statements");

  if (route.getCallOpaqueSteps().size() != 1)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires exactly one provider-built pre-loop setvl statement before "
        "artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          preLoopSetVL, consumerLabel, "pre-loop setvl",
          description.setVLIntrinsic,
          {{runtimeNABI->cName, runtimeNABI->cType}},
          description.emitCFullChunkVLName, description.vlCType))
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
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.upperBound.expression != runtimeNABI->cName ||
      loop.upperBound.cType != runtimeNABI->cType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "base-memory-movement target artifact consumer requires "
        "provider-built loop bounds and step to mirror runtime n/AVL/VL facts");
  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeNABI->cName) + " - " +
       description.emitCLoopInductionName)
          .str();

  std::size_t expectedLoopStepCount = 0;
  switch (operation) {
  case OperationKind::StridedLoadUnitStore:
  case OperationKind::UnitLoadStridedStore:
    expectedLoopStepCount = 3;
    break;
  case OperationKind::IndexedGatherUnitStore:
  case OperationKind::IndexedScatterUnitLoad:
  case OperationKind::MaskedUnitStore:
    expectedLoopStepCount = 5;
    break;
  case OperationKind::MaskedUnitLoadStore:
    expectedLoopStepCount = 6;
    break;
  default:
    llvm_unreachable("validated non-base-memory operation as base memory");
  }
  if (loop.bodySteps.size() != expectedLoopStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires exact provider-built base-memory loop statement count " +
        llvm::Twine(expectedLoopStepCount) + " before artifact export");

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[0], consumerLabel, "loop setvl",
          description.setVLIntrinsic,
          {{expectedRemainingAVL, description.vlCType}},
          description.emitCLoopVLName, description.vlCType))
    return error;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "base-memory-movement target artifact consumer requires loop "
          "statements to carry selected typed RVV source provenance");

  const std::string sourcePointer =
      (llvm::StringRef(sourceABI->cName) + " + " +
       description.emitCLoopInductionName)
          .str();
  const std::string destinationPointer =
      (llvm::StringRef(destinationABI->cName) + " + " +
       description.emitCLoopInductionName)
          .str();
  const std::string stridedSourcePointer =
      sourceStrideABI
          ? ("(const int32_t *)((const uint8_t *)" +
             llvm::StringRef(sourceABI->cName) + " + (" +
             description.emitCLoopInductionName + " * " +
             sourceStrideABI->cName + "))")
                .str()
          : std::string();
  const std::string stridedDestinationPointer =
      destinationStrideABI
          ? ("(int32_t *)((uint8_t *)" +
             llvm::StringRef(destinationABI->cName) + " + (" +
             description.emitCLoopInductionName + " * " +
             destinationStrideABI->cName + "))")
                .str()
          : std::string();
  const std::string indexPointer =
      indexABI ? (llvm::StringRef(indexABI->cName) + " + " +
                  description.emitCLoopInductionName)
                     .str()
               : std::string();
  const std::string maskPointer =
      maskABI ? (llvm::StringRef(maskABI->cName) + " + " +
                 description.emitCLoopInductionName)
                    .str()
              : std::string();

  auto validateUnitLoad =
      [&](std::size_t stepIndex, llvm::StringRef resultName,
          llvm::StringRef stepLabel) -> llvm::Error {
    return validateRVVProviderBuiltRouteStep(
        loop.bodySteps[stepIndex], consumerLabel, stepLabel,
        description.vectorLoadIntrinsic,
        {{sourcePointer, sourceABI->cType},
         {description.emitCLoopVLName, description.vlCType}},
        resultName, description.vectorCType);
  };
  auto validateUnitStore = [&](std::size_t stepIndex,
                               llvm::StringRef valueName,
                               llvm::StringRef stepLabel) -> llvm::Error {
    return validateRVVProviderBuiltRouteStep(
        loop.bodySteps[stepIndex], consumerLabel, stepLabel,
        description.storeIntrinsic,
        {{destinationPointer, destinationABI->cType},
         {valueName, description.vectorCType},
         {description.emitCLoopVLName, description.vlCType}});
  };

  switch (operation) {
  case OperationKind::StridedLoadUnitStore:
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[1], consumerLabel, "strided load",
            description.stridedLoadIntrinsic,
            {{stridedSourcePointer, sourceABI->cType},
             {sourceStrideABI->cName, "ptrdiff_t"},
             {description.emitCLoopVLName, description.vlCType}},
            description.resultName, description.vectorCType))
      return error;
    return validateUnitStore(/*stepIndex=*/2, description.resultName,
                             "unit store");
  case OperationKind::UnitLoadStridedStore:
    if (llvm::Error error =
            validateUnitLoad(/*stepIndex=*/1, description.resultName,
                             "unit load"))
      return error;
    return validateRVVProviderBuiltRouteStep(
        loop.bodySteps[2], consumerLabel, "strided store",
        description.stridedStoreIntrinsic,
        {{stridedDestinationPointer, destinationABI->cType},
         {destinationStrideABI->cName, "ptrdiff_t"},
         {description.resultName, description.vectorCType},
         {description.emitCLoopVLName, description.vlCType}});
  case OperationKind::IndexedGatherUnitStore:
    if (description.indexVectorCType.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires provider-derived index vector C type before validating "
          "indexed gather statements");
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[1], consumerLabel, "index load",
            description.indexLoadIntrinsic,
            {{indexPointer, indexABI->cType},
             {description.emitCLoopVLName, description.vlCType}},
            "index_vec", description.indexVectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[2], consumerLabel, "index scale",
            description.indexScaleIntrinsic,
            {{"index_vec", description.indexVectorCType},
             {"4", "uint32_t"},
             {description.emitCLoopVLName, description.vlCType}},
            "byte_offsets", description.indexVectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[3], consumerLabel, "indexed gather load",
            description.indexedLoadIntrinsic,
            {{sourceABI->cName, sourceABI->cType},
             {"byte_offsets", description.indexVectorCType},
             {description.emitCLoopVLName, description.vlCType}},
            description.resultName, description.vectorCType))
      return error;
    return validateUnitStore(/*stepIndex=*/4, description.resultName,
                             "unit store");
  case OperationKind::IndexedScatterUnitLoad:
    if (description.indexVectorCType.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires provider-derived index vector C type before validating "
          "indexed scatter statements");
    if (llvm::Error error =
            validateUnitLoad(/*stepIndex=*/1, description.resultName,
                             "unit load"))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[2], consumerLabel, "index load",
            description.indexLoadIntrinsic,
            {{indexPointer, indexABI->cType},
             {description.emitCLoopVLName, description.vlCType}},
            "index_vec", description.indexVectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[3], consumerLabel, "index scale",
            description.indexScaleIntrinsic,
            {{"index_vec", description.indexVectorCType},
             {"4", "uint32_t"},
             {description.emitCLoopVLName, description.vlCType}},
            "byte_offsets", description.indexVectorCType))
      return error;
    return validateRVVProviderBuiltRouteStep(
        loop.bodySteps[4], consumerLabel, "indexed scatter store",
        description.indexedStoreIntrinsic,
        {{destinationABI->cName, destinationABI->cType},
         {"byte_offsets", description.indexVectorCType},
         {description.resultName, description.vectorCType},
         {description.emitCLoopVLName, description.vlCType}});
  case OperationKind::MaskedUnitLoadStore:
    if (description.maskName.empty() || description.maskCType.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires provider-derived mask result and mask C type before "
          "validating masked load/store statements");
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[1], consumerLabel, "mask vector load",
            description.vectorLoadIntrinsic,
            {{maskPointer, maskABI->cType},
             {description.emitCLoopVLName, description.vlCType}},
            "mask_i32_vec", description.vectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[2], consumerLabel, "mask predicate",
            description.compareIntrinsic,
            {{"mask_i32_vec", description.vectorCType},
             {"0", "int32_t"},
             {description.emitCLoopVLName, description.vlCType}},
            description.maskName, description.maskCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[3], consumerLabel, "passthrough load",
            description.vectorLoadIntrinsic,
            {{destinationPointer, destinationABI->cType},
             {description.emitCLoopVLName, description.vlCType}},
            "old_dst_vec", description.vectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[4], consumerLabel, "masked load",
            description.maskedLoadIntrinsic,
            {{description.maskName, description.maskCType},
             {"old_dst_vec", description.vectorCType},
             {sourcePointer, sourceABI->cType},
             {description.emitCLoopVLName, description.vlCType}},
            description.resultName, description.vectorCType))
      return error;
    return validateUnitStore(/*stepIndex=*/5, description.resultName,
                             "unit store");
  case OperationKind::MaskedUnitStore:
    if (description.maskName.empty() || description.maskCType.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires provider-derived mask result and mask C type before "
          "validating masked store statements");
    if (llvm::Error error =
            validateUnitLoad(/*stepIndex=*/1, "lhs_vec", "source load"))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[2], consumerLabel, "mask vector load",
            description.vectorLoadIntrinsic,
            {{maskPointer, maskABI->cType},
             {description.emitCLoopVLName, description.vlCType}},
            "mask_i32_vec", description.vectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[3], consumerLabel, "mask predicate",
            description.compareIntrinsic,
            {{"mask_i32_vec", description.vectorCType},
             {"0", "int32_t"},
             {description.emitCLoopVLName, description.vlCType}},
            description.maskName, description.maskCType))
      return error;
    return validateRVVProviderBuiltRouteStep(
        loop.bodySteps[4], consumerLabel, "masked store",
        description.storeIntrinsic,
        {{description.maskName, description.maskCType},
         {destinationPointer, destinationABI->cType},
         {"lhs_vec", description.vectorCType},
         {description.emitCLoopVLName, description.vlCType}});
  default:
    llvm_unreachable("validated non-base-memory operation as base memory");
  }
}

llvm::Error validateRVVBaseMemoryMovementRouteLayoutFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  const plugin::rvv::RVVSelectedBodyOperationKind operation =
      description.operation;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "strided memory layout", description.stridedMemoryLayout,
          getRVVBaseMemoryMovementExpectedStridedLayout(operation)))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "indexed or masked memory layout", description.indexedMemoryLayout,
          getRVVBaseMemoryMovementExpectedIndexedOrMaskedLayout(operation)))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "lhs stride source", description.lhsStrideSource, ""))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "rhs stride source", description.rhsStrideSource, ""))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "source stride binding", description.sourceStrideSource,
          getRVVBaseMemoryMovementExpectedSourceStrideSource(operation)))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "destination stride binding", description.outStrideSource,
          getRVVBaseMemoryMovementExpectedDestinationStrideSource(operation)))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "source memory form", description.sourceMemoryForm,
          getRVVBaseMemoryMovementExpectedSourceMemoryForm(operation)))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "destination memory form", description.destinationMemoryForm,
          getRVVBaseMemoryMovementExpectedDestinationMemoryForm(operation)))
    return error;

  const int64_t expectedIndexEEW =
      getRVVBaseMemoryMovementExpectedIndexEEW(operation);
  if (description.indexEEW != expectedIndexEEW)
    return makeRVVTargetRouteError(
        llvm::Twine("base-memory-movement target artifact consumer requires "
                    "provider-derived index EEW ") +
        llvm::Twine(expectedIndexEEW) + " but was " +
        llvm::Twine(description.indexEEW));
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "index source", description.indexSource,
          getRVVBaseMemoryMovementExpectedIndexSource(operation)))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "offset unit", description.offsetUnit,
          getRVVBaseMemoryMovementExpectedOffsetUnit(operation)))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "index uniqueness", description.indexUniqueness,
          getRVVBaseMemoryMovementExpectedIndexUniqueness(operation)))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "indexed data memory form", description.indexedDataMemoryForm,
          getRVVBaseMemoryMovementExpectedIndexedDataMemoryForm(operation)))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "indexed destination memory form",
          description.indexedDestinationMemoryForm,
          getRVVBaseMemoryMovementExpectedIndexedDestinationMemoryForm(
              operation)))
    return error;

  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "mask role", description.maskRole,
          getRVVBaseMemoryMovementExpectedMaskRole(operation)))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "mask source", description.maskSource,
          getRVVBaseMemoryMovementExpectedMaskSource(operation)))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "mask memory form", description.maskMemoryForm,
          getRVVBaseMemoryMovementExpectedMaskMemoryForm(operation)))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "inactive lane contract", description.inactiveLaneContract,
          getRVVBaseMemoryMovementExpectedInactiveLaneContract(operation)))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "masked passthrough layout", description.maskedPassthroughLayout,
          getRVVBaseMemoryMovementExpectedMaskedPassthroughLayout(operation)))
    return error;
  return llvm::Error::success();
}

llvm::Error validateRVVBaseMemoryMovementRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("base-memory-movement target artifact consumer requires "
                    "rebuilt provider route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");
  const plugin::rvv::RVVSelectedBodyMemoryForm expectedMemoryForm =
      getRVVBaseMemoryMovementExpectedMemoryForm(description.operation);
  if (description.memoryForm != expectedMemoryForm)
    return makeRVVTargetRouteError(
        llvm::Twine("base-memory-movement target artifact consumer requires "
                    "selected typed RVV memory form '") +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(expectedMemoryForm) +
        "' before artifact export but saw '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
            description.memoryForm) +
        "'");
  if (description.routeOperandBindingSummary.empty() ||
      description.targetLeafProfile.empty())
    return makeRVVTargetRouteError(
        "base-memory-movement target artifact consumer requires "
        "provider-derived binding summary and target leaf profile facts before "
        "artifact export");

  const std::string expectedProviderSupportedMirror =
      getRVVBaseMemoryMovementExpectedProviderSupportedMirror(
          description.operation);
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "provider-supported mirror", description.providerSupportedMirror,
          expectedProviderSupportedMirror))
    return error;
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "route-family plan",
          description.baseMemoryMovementRouteFamilyPlanID,
          "rvv-base-memory-movement-route-family-plan.v1"))
    return error;
  const std::string expectedRouteOperandBindingPlan =
      getRVVBaseMemoryMovementExpectedRouteOperandBindingPlan(
          description.operation);
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "route operand binding plan", description.routeOperandBindingPlanID,
          expectedRouteOperandBindingPlan))
    return error;
  if (!llvm::StringRef(description.routeOperandBindingSummary)
           .starts_with(description.routeOperandBindingPlanID))
    return makeRVVTargetRouteError(
        llvm::Twine("base-memory-movement target artifact consumer requires "
                    "route operand binding summary to mirror provider binding "
                    "plan '") +
        description.routeOperandBindingPlanID + "' but was '" +
        description.routeOperandBindingSummary + "'");
  if (llvm::Error error = requireRVVBaseMemoryMovementProviderField(
          "runtime AVL/VL control plan", description.runtimeControlPlanID,
          "rvv-runtime-avl-vl-control-plan.v1"))
    return error;

  llvm::StringRef expectedRuntimeABIOrder =
      getRVVBaseMemoryMovementExpectedRuntimeABIOrder(description.operation);
  if (description.runtimeABIOrder != expectedRuntimeABIOrder)
    return makeRVVTargetRouteError(
        llvm::Twine("base-memory-movement target artifact consumer requires "
                    "provider-derived runtime ABI order '") +
        expectedRuntimeABIOrder + "' but was '" + description.runtimeABIOrder +
        "'");
  if (description.sew == 0 || description.lmul.empty() ||
      description.tailPolicy.empty() || description.maskPolicy.empty() ||
      description.setVLIntrinsic.empty() || description.resultName.empty())
    return makeRVVTargetRouteError(
        "base-memory-movement target artifact consumer requires "
        "provider-derived dtype, policy, setvl, and result facts before "
        "artifact export");
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

  if (llvm::Error error =
          validateRVVBaseMemoryMovementRouteLayoutFacts(description))
    return error;
  if (llvm::Error error =
          validateRVVBaseMemoryMovementRouteHeaders(route, description))
    return error;
  if (llvm::Error error =
          validateRVVBaseMemoryMovementRouteTypeMappings(route, description))
    return error;
  if (llvm::Error error =
          validateRVVBaseMemoryMovementRouteABIMappings(route, description))
    return error;
  return validateRVVBaseMemoryMovementRouteStatementPlan(route, description);
}

llvm::Error validateRVVBaseMemoryMovementTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVBaseMemoryMovementRoutePayloadFacts(context.route,
                                                        context.description);
}

llvm::Error requireEmptyBaseMemoryMovementStaleMirror(
    const TargetArtifactCandidate &candidate, llvm::StringRef key,
    llvm::StringRef label) {
  return requireCandidateMetadataMirror(candidate, key, "", label);
}

llvm::Error validateRVVBaseMemoryMovementTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description =
      context.description;

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_plan",
          description.routeOperandBindingPlanID,
          "selected typed RVV base-memory binding plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_operands",
          description.routeOperandBindingSummary,
          "selected typed RVV base-memory binding summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.provider_supported_mirror",
          description.providerSupportedMirror,
          "selected typed RVV base-memory provider support"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.base_memory_movement_route_family_plan",
          description.baseMemoryMovementRouteFamilyPlanID,
          "selected typed RVV base-memory route-family plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.memory_form",
          plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
              description.memoryForm),
          "selected typed RVV base-memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.target_leaf_profile",
          description.targetLeafProfile,
          "selected typed RVV base-memory target leaf profile"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_control_plan",
          description.runtimeControlPlanID,
          "selected typed RVV base-memory runtime AVL/VL control plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order",
          description.runtimeABIOrder,
          "selected typed RVV base-memory runtime ABI order"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.required_header_declarations",
          description.requiredHeaderDeclarations,
          "selected typed RVV base-memory route header requirements"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.c_type_mapping",
          description.cTypeMappingSummary,
          "selected typed RVV base-memory route type mapping summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_memory_form",
          description.sourceMemoryForm,
          "selected typed RVV base-memory source memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.destination_memory_form",
          description.destinationMemoryForm,
          "selected typed RVV base-memory destination memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.strided_memory_layout",
          description.stridedMemoryLayout,
          "selected typed RVV base-memory strided layout"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_stride_source",
          description.sourceStrideSource,
          "selected typed RVV base-memory source stride binding"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.destination_stride_source",
          description.outStrideSource,
          "selected typed RVV base-memory destination stride binding"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.indexed_memory_layout",
          isRVVIndexedBaseMemoryMovementRouteFamilyOperation(
              description.operation)
              ? description.indexedMemoryLayout
              : llvm::StringRef(),
          "selected typed RVV base-memory indexed layout"))
    return error;

  std::string indexEEWMirror =
      description.indexEEW == 0 ? "" : llvm::Twine(description.indexEEW).str();
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.index_source", description.indexSource,
          "selected typed RVV base-memory index source"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.index_eew", indexEEWMirror,
          "selected typed RVV base-memory index EEW"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.offset_unit", description.offsetUnit,
          "selected typed RVV base-memory offset unit"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.index_uniqueness",
          description.indexUniqueness,
          "selected typed RVV base-memory index uniqueness"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.indexed_data_memory_form",
          description.indexedDataMemoryForm,
          "selected typed RVV base-memory indexed data memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.indexed_destination_memory_form",
          description.indexedDestinationMemoryForm,
          "selected typed RVV base-memory indexed destination memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.masked_memory_layout",
          isRVVMaskedBaseMemoryMovementRouteFamilyOperation(
              description.operation)
              ? description.indexedMemoryLayout
              : llvm::StringRef(),
          "selected typed RVV base-memory masked layout"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.mask_role", description.maskRole,
          "selected typed RVV base-memory mask role"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.mask_source", description.maskSource,
          "selected typed RVV base-memory mask source"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.mask_memory_form",
          description.maskMemoryForm,
          "selected typed RVV base-memory mask memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.inactive_lane_contract",
          description.inactiveLaneContract,
          "selected typed RVV base-memory inactive lane contract"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.masked_passthrough_layout",
          description.maskedPassthroughLayout,
          "selected typed RVV base-memory masked passthrough layout"))
    return error;

  constexpr llvm::StringLiteral staleRouteFamilyMirrors[] = {
      "tcrv_rvv.elementwise_arithmetic_route_family_plan",
      "tcrv_rvv.scalar_broadcast_elementwise_route_family_plan",
      "tcrv_rvv.widening_conversion_route_family_plan",
      "tcrv_rvv.runtime_scalar_splat_store_route_family_plan",
      "tcrv_rvv.plain_compare_select_route_family_plan",
      "tcrv_rvv.computed_mask_select_route_family_plan",
      "tcrv_rvv.computed_mask_memory_route_family_plan",
      "tcrv_rvv.segment2_memory_route_family_plan",
      "tcrv_rvv.plain_macc_route_family_plan",
      "tcrv_rvv.scalar_broadcast_macc_route_family_plan",
      "tcrv_rvv.accumulation_route_family_plan",
      "tcrv_rvv.standalone_reduction_route_family_plan",
      "tcrv_rvv.contraction_route_family_plan",
      "tcrv_rvv.mask_tail_policy_route_family_plan",
      "tcrv_rvv.mask_tail_policy_owner",
      "tcrv_rvv.widening_macc_relation",
      "tcrv_rvv.widening_dot_relation"};
  for (llvm::StringRef key : staleRouteFamilyMirrors)
    if (llvm::Error error = requireEmptyBaseMemoryMovementStaleMirror(
            candidate, key,
            "selected typed RVV non-base-memory route-family mirror"))
      return error;

  return llvm::Error::success();
}

constexpr llvm::StringLiteral kRVVWideningDotRuntimeABIOrder("lhs,rhs,acc,out,n");
constexpr llvm::StringLiteral kRVVStridedInputWideningDotRuntimeABIOrder(
    "lhs,rhs,acc,out,n,lhs_stride,rhs_stride");
constexpr llvm::StringLiteral kRVVComputedMaskWideningDotRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n");
constexpr llvm::StringLiteral
    kRVVComputedMaskStridedInputWideningDotRuntimeABIOrder(
        "cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride");
constexpr llvm::StringLiteral kRVVWideningDotRouteOperandBindingPlan(
    "rvv-route-operand-binding:widening_dot_reduce.v1");
constexpr llvm::StringLiteral kRVVStridedInputWideningDotRouteOperandBindingPlan(
    "rvv-route-operand-binding:strided_widening_dot_reduce.v1");
constexpr llvm::StringLiteral kRVVComputedMaskWideningDotRouteOperandBindingPlan(
    "rvv-route-operand-binding:masked_widening_dot_reduce.v1");
constexpr llvm::StringLiteral
    kRVVComputedMaskStridedInputWideningDotRouteOperandBindingPlan(
        "rvv-route-operand-binding:masked_strided_wdot.v1");
constexpr llvm::StringLiteral kRVVWideningDotRouteOperandBindingSummary(
    "rvv-route-operand-binding:widening_dot_reduce.v1;"
    "lhs=lhs-input-buffer:lhs:abi|ld|dot-lhs|i16|hdr;"
    "rhs=rhs-input-buffer:rhs:abi|ld|dot-rhs|i16|hdr;"
    "acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;"
    "out=output-buffer:out:abi|store|i32|hdr;"
    "n=runtime-element-count:n:abi|setvl-avl|loop|hdr");
constexpr llvm::StringLiteral kRVVStridedInputWideningDotRouteOperandBindingSummary(
    "rvv-route-operand-binding:strided_widening_dot_reduce.v1;"
    "lhs=lhs-input-buffer:lhs:abi|sld|dot-lhs|i16|hdr;"
    "rhs=rhs-input-buffer:rhs:abi|sld|dot-rhs|i16|hdr;"
    "acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;"
    "out=output-buffer:out:abi|store|i32|hdr;"
    "n=runtime-element-count:n:abi|setvl-avl|loop|hdr;"
    "lhs_stride=lhs-input-stride:lhs_stride:abi|str|addr|hdr;"
    "rhs_stride=rhs-input-stride:rhs_stride:abi|str|addr|hdr");
constexpr llvm::StringLiteral kRVVComputedMaskWideningDotRouteOperandBindingSummary(
    "rvv-route-operand-binding:masked_widening_dot_reduce.v1;"
    "cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp|mask|hdr;"
    "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp|mask|hdr;"
    "dot_lhs=dot-lhs-input-buffer:lhs:abi|ld|mlhs|i16|hdr;"
    "dot_rhs=dot-rhs-input-buffer:rhs:abi|ld|mrhs|i16|hdr;"
    "acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;"
    "out=output-buffer:out:abi|store|i32|hdr;"
    "n=runtime-element-count:n:abi|setvl-avl|loop|hdr");
constexpr llvm::StringLiteral
    kRVVComputedMaskStridedInputWideningDotRouteOperandBindingSummary(
        "rvv-route-operand-binding:masked_strided_wdot.v1;"
        "cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp|mask|hdr;"
        "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp|mask|hdr;"
        "dot_lhs=dot-lhs-input-buffer:lhs:abi|sld|mlhs|i16|hdr;"
        "dot_rhs=dot-rhs-input-buffer:rhs:abi|sld|mrhs|i16|hdr;"
        "acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;"
        "out=output-buffer:out:abi|store|i32|hdr;"
        "n=runtime-element-count:n:abi|setvl-avl|loop|hdr;"
        "lhs_stride=lhs-input-stride:lhs_stride:abi|str|addr|hdr;"
        "rhs_stride=rhs-input-stride:rhs_stride:abi|str|addr|hdr");
constexpr llvm::StringLiteral kRVVWideningDotContractionRouteFamilyPlan(
    "rvv-contraction-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVWideningDotProviderSupportedMirror(
    "provider_supported_mirror:rvv-contraction-family-plan-validated");
constexpr llvm::StringLiteral kRVVComputedMaskRole(
    "predicate-mask-produced-by-compare");
constexpr llvm::StringLiteral kRVVComputedMaskSource(
    "compare-produced-mask-same-vl-scope");
constexpr llvm::StringLiteral kRVVComputedMaskMemoryForm(
    "compare-produced-mask");
constexpr llvm::StringLiteral kRVVComputedMaskComparePredicateKind("slt");

llvm::Error validateRVVNonComputedMaskWideningDotReductionRuntimeABIFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  const bool isStrided =
      isRVVStridedInputWideningDotReductionRouteFamilyOperation(
          description.operation);
  const llvm::StringRef expectedABIOrder =
      isStrided ? kRVVStridedInputWideningDotRuntimeABIOrder
                : kRVVWideningDotRuntimeABIOrder;
  if (description.runtimeABIOrder != expectedABIOrder)
    return makeRVVTargetRouteError(
        llvm::Twine("widening dot-reduction target artifact consumer "
                    "requires provider-derived runtime ABI order '") +
        expectedABIOrder + "' but was '" + description.runtimeABIOrder + "'");
  const size_t expectedCount = isStrided ? 7 : 5;
  if (description.runtimeABIParameters.size() != expectedCount)
    return makeRVVTargetRouteError(
        llvm::Twine("widening dot-reduction target artifact consumer "
                    "requires provider-derived runtime ABI parameter count ") +
        llvm::Twine(expectedCount) + " before artifact export");

  struct ExpectedRuntimeABIParameterRole {
    llvm::StringRef cName;
    support::RuntimeABIParameterRole role;
  };
  const ExpectedRuntimeABIParameterRole expectedPlainRoles[] = {
      {"lhs", support::RuntimeABIParameterRole::LHSInputBuffer},
      {"rhs", support::RuntimeABIParameterRole::RHSInputBuffer},
      {"acc", support::RuntimeABIParameterRole::AccumulatorInputBuffer},
      {"out", support::RuntimeABIParameterRole::OutputBuffer},
      {"n", support::RuntimeABIParameterRole::RuntimeElementCount},
  };
  const ExpectedRuntimeABIParameterRole expectedStridedRoles[] = {
      {"lhs", support::RuntimeABIParameterRole::LHSInputBuffer},
      {"rhs", support::RuntimeABIParameterRole::RHSInputBuffer},
      {"acc", support::RuntimeABIParameterRole::AccumulatorInputBuffer},
      {"out", support::RuntimeABIParameterRole::OutputBuffer},
      {"n", support::RuntimeABIParameterRole::RuntimeElementCount},
      {"lhs_stride", support::RuntimeABIParameterRole::LHSInputStride},
      {"rhs_stride", support::RuntimeABIParameterRole::RHSInputStride},
  };
  llvm::ArrayRef<ExpectedRuntimeABIParameterRole> expectedRoles =
      isStrided ? llvm::ArrayRef(expectedStridedRoles)
                : llvm::ArrayRef(expectedPlainRoles);
  for (size_t index = 0; index < expectedRoles.size(); ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    if (actual.cName != expectedRoles[index].cName ||
        actual.role != expectedRoles[index].role)
      return makeRVVTargetRouteError(
          llvm::Twine("widening dot-reduction target artifact consumer "
                      "requires provider-derived runtime ABI parameter ") +
          std::to_string(index) + " to bind " + expectedRoles[index].cName +
          " as " +
          support::stringifyRuntimeABIParameterRole(expectedRoles[index].role) +
          " before artifact export");
  }
  return llvm::Error::success();
}

llvm::Error validateRVVComputedMaskWideningDotReductionRuntimeABIFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  const bool isStrided =
      isRVVStridedInputWideningDotReductionRouteFamilyOperation(
          description.operation);
  const std::optional<
      plugin::rvv::RVVComputedMaskStridedInputWideningDotReduceRouteFacts>
      computedMaskStridedFacts =
          plugin::rvv::getRVVComputedMaskStridedInputWideningDotReduceRouteFacts(
              description.operation);
  if (isStrided && !computedMaskStridedFacts)
    return makeRVVTargetRouteError(
        "computed-mask strided widening dot-reduction target artifact "
        "consumer requires provider-owned canonical route facts before "
        "validating runtime ABI order");
  const llvm::StringRef expectedABIOrder =
      isStrided ? computedMaskStridedFacts->runtimeABIOrder
                : kRVVComputedMaskWideningDotRuntimeABIOrder;
  if (description.runtimeABIOrder != expectedABIOrder)
    return makeRVVTargetRouteError(
        llvm::Twine("computed-mask widening dot-reduction target artifact "
                    "consumer requires provider-derived runtime ABI order '") +
        expectedABIOrder + "' but was '" + description.runtimeABIOrder + "'");
  const size_t expectedCount = isStrided ? 9 : 7;
  if (description.runtimeABIParameters.size() != expectedCount)
    return makeRVVTargetRouteError(
        llvm::Twine("computed-mask widening dot-reduction target artifact "
                    "consumer requires provider-derived runtime ABI "
                    "parameter count ") +
        llvm::Twine(expectedCount) + " before artifact export");

  struct ExpectedRuntimeABIParameterRole {
    llvm::StringRef cName;
    support::RuntimeABIParameterRole role;
  };
  const ExpectedRuntimeABIParameterRole expectedUnitRoles[] = {
      {"cmp_lhs", support::RuntimeABIParameterRole::LHSInputBuffer},
      {"cmp_rhs", support::RuntimeABIParameterRole::RHSInputBuffer},
      {"lhs", support::RuntimeABIParameterRole::DotLHSInputBuffer},
      {"rhs", support::RuntimeABIParameterRole::DotRHSInputBuffer},
      {"acc", support::RuntimeABIParameterRole::AccumulatorInputBuffer},
      {"out", support::RuntimeABIParameterRole::OutputBuffer},
      {"n", support::RuntimeABIParameterRole::RuntimeElementCount},
  };
  const ExpectedRuntimeABIParameterRole expectedStridedRoles[] = {
      {"cmp_lhs", support::RuntimeABIParameterRole::LHSInputBuffer},
      {"cmp_rhs", support::RuntimeABIParameterRole::RHSInputBuffer},
      {"lhs", support::RuntimeABIParameterRole::DotLHSInputBuffer},
      {"rhs", support::RuntimeABIParameterRole::DotRHSInputBuffer},
      {"acc", support::RuntimeABIParameterRole::AccumulatorInputBuffer},
      {"out", support::RuntimeABIParameterRole::OutputBuffer},
      {"n", support::RuntimeABIParameterRole::RuntimeElementCount},
      {"lhs_stride", support::RuntimeABIParameterRole::LHSInputStride},
      {"rhs_stride", support::RuntimeABIParameterRole::RHSInputStride},
  };
  llvm::ArrayRef<ExpectedRuntimeABIParameterRole> expectedRoles =
      isStrided ? llvm::ArrayRef(expectedStridedRoles)
                : llvm::ArrayRef(expectedUnitRoles);
  for (size_t index = 0; index < expectedRoles.size(); ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    if (actual.cName != expectedRoles[index].cName ||
        actual.role != expectedRoles[index].role)
      return makeRVVTargetRouteError(
          llvm::Twine("computed-mask widening dot-reduction target artifact "
                      "consumer requires provider-derived runtime ABI "
                      "parameter ") +
          std::to_string(index) + " to bind " + expectedRoles[index].cName +
          " as " +
          support::stringifyRuntimeABIParameterRole(expectedRoles[index].role) +
          " before artifact export");
  }
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

llvm::Error validateRVVNonComputedMaskWideningDotReductionRoutePayloadFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  const bool isStrided =
      isRVVStridedInputWideningDotReductionRouteFamilyOperation(
          description.operation);
  const llvm::StringRef expectedBindingPlan =
      isStrided ? kRVVStridedInputWideningDotRouteOperandBindingPlan
                : kRVVWideningDotRouteOperandBindingPlan;
  const llvm::StringRef expectedBindingSummary =
      isStrided ? kRVVStridedInputWideningDotRouteOperandBindingSummary
                : kRVVWideningDotRouteOperandBindingSummary;

  constexpr llvm::StringLiteral consumerLabel(
      "widening dot-reduction target artifact consumer");
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "provider-supported mirror",
          description.providerSupportedMirror))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "contraction route-family plan",
          description.contractionRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "target leaf profile", description.targetLeafProfile))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "required header declarations",
          description.requiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "C type mapping summary",
          description.cTypeMappingSummary))
    return error;
  if (description.routeOperandBindingPlanID != expectedBindingPlan ||
      description.routeOperandBindingSummary != expectedBindingSummary)
    return makeRVVTargetRouteError(
        llvm::Twine("widening dot-reduction target artifact consumer requires "
                    "provider route operand binding plan '") +
        expectedBindingPlan + "' and exact operand binding summary before "
        "artifact export");
  if (description.memoryForm !=
          (isStrided
               ? plugin::rvv::RVVSelectedBodyMemoryForm::
                     StridedInputWideningDotReduce
               : plugin::rvv::RVVSelectedBodyMemoryForm::VectorRHSLoad) ||
      description.typedComputeOpName != "tcrv_rvv.widening_dot_reduce")
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires a selected "
        "tcrv_rvv.widening_dot_reduce body with the provider-derived source "
        "memory form");
  if (llvm::Error error =
          validateRVVNonComputedMaskWideningDotReductionRuntimeABIFacts(
              description))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "dot accumulator layout",
          description.wideningDotProductAccumulatorLayout))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "dot result layout",
          description.wideningDotProductResultLayout))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "dot relation",
          description.wideningDotProductRelation))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "reduction store VL", description.reductionStoreVL))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "source load intrinsic",
          description.sourceVectorLoadIntrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "widening product intrinsic",
          description.wideningProductIntrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "scalar seed splat intrinsic",
          description.scalarSeedSplatIntrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "reduction intrinsic", description.intrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "store intrinsic", description.storeIntrinsic))
    return error;
  if (isStrided) {
    if (llvm::Error error = requireRVVProviderDerivedField(
            "strided-input widening dot-reduction target artifact consumer",
            "strided load intrinsic", description.stridedLoadIntrinsic))
      return error;
    if (llvm::Error error = requireRVVProviderDerivedField(
            "strided-input widening dot-reduction target artifact consumer",
            "strided memory layout", description.stridedMemoryLayout))
      return error;
    if (llvm::Error error = requireRVVProviderDerivedField(
            "strided-input widening dot-reduction target artifact consumer",
            "lhs stride source", description.lhsStrideSource))
      return error;
    if (llvm::Error error = requireRVVProviderDerivedField(
            "strided-input widening dot-reduction target artifact consumer",
            "rhs stride source", description.rhsStrideSource))
      return error;
    if (llvm::Error error = requireRVVProviderDerivedField(
            "strided-input widening dot-reduction target artifact consumer",
            "source memory form", description.sourceMemoryForm))
      return error;
    if (llvm::Error error = requireRVVProviderDerivedField(
            "strided-input widening dot-reduction target artifact consumer",
            "destination memory form", description.destinationMemoryForm))
      return error;
  } else if (!description.stridedLoadIntrinsic.empty() ||
             !description.stridedMemoryLayout.empty() ||
             !description.lhsStrideSource.empty() ||
             !description.rhsStrideSource.empty() ||
             !description.sourceMemoryForm.empty() ||
             !description.destinationMemoryForm.empty()) {
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer rejects stale "
        "strided-input facts on unit-stride widening dot routes");
  }
  if (llvm::Error error =
          validateRVVWideningDotReductionNoStaleNonFamilyProviderFacts(
              description))
    return error;
  return llvm::Error::success();
}

llvm::Error validateRVVComputedMaskWideningDotReductionRoutePayloadFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  const bool isStrided =
      isRVVStridedInputWideningDotReductionRouteFamilyOperation(
          description.operation);
  const std::optional<
      plugin::rvv::RVVComputedMaskStridedInputWideningDotReduceRouteFacts>
      computedMaskStridedFacts =
          plugin::rvv::getRVVComputedMaskStridedInputWideningDotReduceRouteFacts(
              description.operation);
  if (isStrided && !computedMaskStridedFacts)
    return makeRVVTargetRouteError(
        "computed-mask strided widening dot-reduction target artifact "
        "consumer requires provider-owned canonical route facts before "
        "artifact export");
  const llvm::StringRef expectedBindingPlan =
      isStrided ? computedMaskStridedFacts->routeOperandBindingPlanID
                : kRVVComputedMaskWideningDotRouteOperandBindingPlan;
  const llvm::StringRef expectedBindingSummary =
      isStrided
          ? llvm::StringRef(
                computedMaskStridedFacts->routeOperandBindingSummary)
          : llvm::StringRef(
                kRVVComputedMaskWideningDotRouteOperandBindingSummary);
  const plugin::rvv::RVVSelectedBodyMemoryForm expectedMemoryForm =
      isStrided
          ? computedMaskStridedFacts->memoryForm
          : plugin::rvv::RVVSelectedBodyMemoryForm::
                ComputedMaskUnitStrideWideningDotReduce;

  constexpr llvm::StringLiteral consumerLabel(
      "computed-mask widening dot-reduction target artifact consumer");
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "provider-supported mirror",
          description.providerSupportedMirror))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "contraction route-family plan",
          description.contractionRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "target leaf profile", description.targetLeafProfile))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "required header declarations",
          description.requiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "C type mapping summary",
          description.cTypeMappingSummary))
    return error;
  if (description.maskRole != kRVVComputedMaskRole ||
      description.maskSource != kRVVComputedMaskSource ||
      description.maskMemoryForm != kRVVComputedMaskMemoryForm)
    return makeRVVTargetRouteError(
        "computed-mask widening dot-reduction target artifact consumer "
        "requires provider-derived computed-mask role/source/form facts "
        "before artifact export");
  if (computedMaskStridedFacts) {
    if (description.runtimeABIOrder !=
            computedMaskStridedFacts->runtimeABIOrder ||
        description.targetLeafProfile !=
            computedMaskStridedFacts->targetLeafProfile ||
        description.providerSupportedMirror !=
            computedMaskStridedFacts->providerSupportedMirror ||
        description.requiredHeaderDeclarations !=
            computedMaskStridedFacts->requiredHeaderDeclarations ||
        description.cTypeMappingSummary !=
            computedMaskStridedFacts->cTypeMappingSummary ||
        description.contractionRouteFamilyPlanID !=
            computedMaskStridedFacts->contractionRouteFamilyPlanID ||
        description.typedComputeOpName !=
            computedMaskStridedFacts->typedComputeOpName ||
        description.comparePredicateKind !=
            computedMaskStridedFacts->comparePredicateKind ||
        description.maskRole != computedMaskStridedFacts->maskRole ||
        description.maskSource != computedMaskStridedFacts->maskSource ||
        description.maskMemoryForm != computedMaskStridedFacts->maskMemoryForm ||
        description.sourceSEW != computedMaskStridedFacts->sourceSEW ||
        description.sourceLMUL != computedMaskStridedFacts->sourceLMUL ||
        description.sew != computedMaskStridedFacts->resultSEW ||
        description.lmul != computedMaskStridedFacts->resultLMUL ||
        description.sourceMemoryForm !=
            computedMaskStridedFacts->sourceMemoryForm ||
        description.destinationMemoryForm !=
            computedMaskStridedFacts->destinationMemoryForm ||
        description.stridedMemoryLayout !=
            computedMaskStridedFacts->stridedMemoryLayout ||
        description.lhsStrideSource !=
            computedMaskStridedFacts->lhsStrideSource ||
        description.rhsStrideSource !=
            computedMaskStridedFacts->rhsStrideSource ||
        description.wideningDotProductAccumulatorLayout !=
            computedMaskStridedFacts->wideningDotProductAccumulatorLayout ||
        description.wideningDotProductResultLayout !=
            computedMaskStridedFacts->wideningDotProductResultLayout ||
        description.wideningDotProductRelation !=
            computedMaskStridedFacts->wideningDotProductRelation ||
        description.wideningProductIntrinsic !=
            computedMaskStridedFacts->wideningProductIntrinsic ||
        description.maskedWideningProductIntrinsic !=
            computedMaskStridedFacts->maskedWideningProductIntrinsic ||
        description.scalarSeedSplatIntrinsic !=
            computedMaskStridedFacts->scalarSeedSplatIntrinsic ||
        description.stridedLoadIntrinsic !=
            computedMaskStridedFacts->stridedLoadIntrinsic ||
        description.sourceVectorLoadIntrinsic !=
            computedMaskStridedFacts->sourceVectorLoadIntrinsic ||
        description.vectorLoadIntrinsic !=
            computedMaskStridedFacts->compareVectorLoadIntrinsic ||
        description.intrinsic !=
            computedMaskStridedFacts->reductionIntrinsic ||
        description.storeIntrinsic != computedMaskStridedFacts->storeIntrinsic ||
        description.setVLIntrinsic != computedMaskStridedFacts->setVLIntrinsic ||
        description.compareIntrinsic !=
            computedMaskStridedFacts->compareIntrinsic ||
        description.maskedMergeIntrinsic !=
            computedMaskStridedFacts->maskedMergeIntrinsic ||
        description.reductionStoreVL !=
            computedMaskStridedFacts->reductionStoreVL ||
        description.inactiveLaneZeroingRequirement !=
            computedMaskStridedFacts->inactiveLaneZeroingRequirement ||
        description.vlCType != computedMaskStridedFacts->vlCType ||
        description.sourceVectorTypeName !=
            computedMaskStridedFacts->sourceVectorTypeName ||
        description.sourceVectorCType !=
            computedMaskStridedFacts->sourceVectorCType ||
        description.vectorTypeName !=
            computedMaskStridedFacts->resultVectorTypeName ||
        description.vectorCType !=
            computedMaskStridedFacts->resultVectorCType ||
        description.maskTypeName != computedMaskStridedFacts->maskTypeName ||
        description.maskCType != computedMaskStridedFacts->maskCType)
      return makeRVVTargetRouteError(
          "computed-mask strided widening dot-reduction target artifact "
          "consumer requires provider-owned canonical route facts before "
          "artifact export");
  }
  if (description.routeOperandBindingPlanID != expectedBindingPlan ||
      description.routeOperandBindingSummary != expectedBindingSummary)
    return makeRVVTargetRouteError(
        llvm::Twine("computed-mask widening dot-reduction target artifact "
                    "consumer requires provider route operand binding plan '") +
        expectedBindingPlan + "' and exact operand binding summary before "
        "artifact export");
  if (description.memoryForm != expectedMemoryForm ||
      description.typedComputeOpName != "tcrv_rvv.masked_widening_dot_reduce")
    return makeRVVTargetRouteError(
        "computed-mask widening dot-reduction target artifact consumer "
        "requires a selected tcrv_rvv.masked_widening_dot_reduce body with "
        "the provider-derived computed-mask memory form");
  if (llvm::Error error =
          validateRVVComputedMaskWideningDotReductionRuntimeABIFacts(
              description))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "tail policy", description.tailPolicy))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "mask policy", description.maskPolicy))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "inactive-lane zeroing requirement",
          description.inactiveLaneZeroingRequirement))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "compare predicate", description.comparePredicateKind))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "compare intrinsic", description.compareIntrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "compare/source load intrinsic",
          description.vectorLoadIntrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "masked widening product intrinsic",
          description.maskedWideningProductIntrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "masked merge intrinsic",
          description.maskedMergeIntrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "mask type", description.maskTypeName))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "mask C type", description.maskCType))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "dot accumulator layout",
          description.wideningDotProductAccumulatorLayout))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "dot result layout",
          description.wideningDotProductResultLayout))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "dot relation",
          description.wideningDotProductRelation))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "reduction store VL", description.reductionStoreVL))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "source load intrinsic",
          description.sourceVectorLoadIntrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "widening product intrinsic",
          description.wideningProductIntrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "scalar seed splat intrinsic",
          description.scalarSeedSplatIntrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "reduction intrinsic", description.intrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "store intrinsic", description.storeIntrinsic))
    return error;
  if (isStrided) {
    if (llvm::Error error = requireRVVProviderDerivedField(
            "computed-mask strided-input widening dot-reduction target "
            "artifact consumer",
            "strided load intrinsic", description.stridedLoadIntrinsic))
      return error;
    if (llvm::Error error = requireRVVProviderDerivedField(
            "computed-mask strided-input widening dot-reduction target "
            "artifact consumer",
            "strided memory layout", description.stridedMemoryLayout))
      return error;
    if (llvm::Error error = requireRVVProviderDerivedField(
            "computed-mask strided-input widening dot-reduction target "
            "artifact consumer",
            "lhs stride source", description.lhsStrideSource))
      return error;
    if (llvm::Error error = requireRVVProviderDerivedField(
            "computed-mask strided-input widening dot-reduction target "
            "artifact consumer",
            "rhs stride source", description.rhsStrideSource))
      return error;
    if (llvm::Error error = requireRVVProviderDerivedField(
            "computed-mask strided-input widening dot-reduction target "
            "artifact consumer",
            "source memory form", description.sourceMemoryForm))
      return error;
    if (llvm::Error error = requireRVVProviderDerivedField(
            "computed-mask strided-input widening dot-reduction target "
            "artifact consumer",
            "destination memory form", description.destinationMemoryForm))
      return error;
  } else if (!description.stridedLoadIntrinsic.empty() ||
             !description.stridedMemoryLayout.empty() ||
             !description.lhsStrideSource.empty() ||
             !description.rhsStrideSource.empty() ||
             !description.sourceMemoryForm.empty() ||
             !description.destinationMemoryForm.empty()) {
    return makeRVVTargetRouteError(
        "computed-mask widening dot-reduction target artifact consumer "
        "rejects stale strided-input facts on unit-stride computed-mask "
        "widening dot routes");
  }
  if (llvm::Error error =
          validateRVVWideningDotReductionNoStaleNonFamilyProviderFacts(
              description))
    return error;
  return llvm::Error::success();
}

llvm::Error validateRVVWideningDotReductionRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-derived required_header_declarations before accepting the "
        "route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires at least "
        "one provider route header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "widening dot-reduction target artifact consumer saw an empty "
          "provider route header declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("widening dot-reduction target artifact consumer "
                      "requires rebuilt provider route header '") +
          trimmed + "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVWideningDotReductionRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.vlCType.empty() || description.vectorTypeName.empty() ||
      description.vectorCType.empty() ||
      description.sourceVectorTypeName.empty() ||
      description.sourceVectorCType.empty() ||
      description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-derived VL, result vector, source vector, and C type "
        "mapping facts before artifact export");

  if (description.sourceSEW == 0 || description.sourceLMUL.empty() ||
      description.sew == 0 || description.lmul.empty())
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-derived source/result dtype and LMUL facts before artifact "
        "export");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("widening dot-reduction target artifact consumer requires "
                    "rebuilt provider route type mapping '!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("widening dot-reduction target artifact consumer requires "
                    "rebuilt provider route result type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");
  if (!routeHasTypeMapping(route, description.sourceVectorTypeName,
                           description.sourceVectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("widening dot-reduction target artifact consumer requires "
                    "rebuilt provider route source type mapping '") +
        description.sourceVectorTypeName + "' -> '" +
        description.sourceVectorCType + "'");

  if (isRVVComputedMaskWideningDotReductionRouteFamilyOperation(
          description.operation)) {
    if (description.maskTypeName.empty() || description.maskCType.empty())
      return makeRVVTargetRouteError(
          "computed-mask widening dot-reduction target artifact consumer "
          "requires provider-derived mask type mapping facts before artifact "
          "export");
    if (!routeHasTypeMapping(route, description.maskTypeName,
                             description.maskCType))
      return makeRVVTargetRouteError(
          llvm::Twine("computed-mask widening dot-reduction target artifact "
                      "consumer requires rebuilt provider route mask type "
                      "mapping '") +
          description.maskTypeName + "' -> '" + description.maskCType + "'");
  } else if (!description.maskTypeName.empty() ||
             !description.maskCType.empty()) {
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer rejects stale mask "
        "type mapping facts for non-computed-mask widening dot routes");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVWideningDotReductionRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (isRVVNonComputedMaskWideningDotReductionRouteFamilyOperation(
          description.operation))
    if (llvm::Error error =
            validateRVVNonComputedMaskWideningDotReductionRuntimeABIFacts(
                description))
      return error;
  if (isRVVComputedMaskWideningDotReductionRouteFamilyOperation(
          description.operation))
    if (llvm::Error error =
            validateRVVComputedMaskWideningDotReductionRuntimeABIFacts(
                description))
      return error;
  if (description.runtimeABIOrder.empty() ||
      description.runtimeABIParameters.empty())
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-derived runtime ABI order and ABI parameters before "
        "artifact export");
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("widening dot-reduction target artifact consumer requires "
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
          llvm::Twine("widening dot-reduction target artifact consumer "
                      "requires rebuilt provider route ABI mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("widening dot-reduction target artifact consumer "
                      "requires rebuilt provider route ABI mapping[") +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVWideningDotReductionRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  constexpr llvm::StringLiteral consumerLabel(
      "widening dot-reduction target artifact consumer");
  const bool isComputedMask =
      isRVVComputedMaskWideningDotReductionRouteFamilyOperation(
          description.operation);
  const bool isStrided =
      isRVVStridedInputWideningDotReductionRouteFamilyOperation(
          description.operation);
  const std::size_t expectedABIParameterCount =
      isComputedMask ? (isStrided ? 9 : 7) : (isStrided ? 7 : 5);
  if (description.runtimeABIParameters.size() < expectedABIParameterCount)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived widening dot ABI parameters before "
        "validating route statements");
  if (description.resultName.empty() || description.sourceVectorCType.empty() ||
      description.vectorCType.empty() || description.vlCType.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived result, source/result vector C type, and "
        "VL C type facts before validating route statements");
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
    outABI = &description.runtimeABIParameters[3];
    runtimeNABI = &description.runtimeABIParameters[4];
    if (isStrided) {
      lhsStrideABI = &description.runtimeABIParameters[5];
      rhsStrideABI = &description.runtimeABIParameters[6];
    }
  }

  const support::RuntimeABIParameter *runtimeElementCount =
      findRuntimeElementCountABIParameter(description);
  if (!runtimeElementCount || runtimeElementCount != runtimeNABI)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires runtime n/AVL ABI role to match the selected widening dot "
        "ABI order before validating route statements");

  const llvm::StringRef scalarI32CType = "int32_t";
  if (route.getCallOpaqueSteps().size() != 3)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-built pre-loop setvl, scalar seed splat, and "
        "initial output store statements before artifact export");

  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps()[0];
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          preLoopSetVL, consumerLabel, "pre-loop setvl",
          description.setVLIntrinsic, {{runtimeNABI->cName, runtimeNABI->cType}},
          description.emitCFullChunkVLName, description.vlCType))
    return error;

  const std::string expectedInitialAccumulatorLane =
      (llvm::StringRef(accumulatorABI->cName) + "[0]").str();
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSeed =
      route.getCallOpaqueSteps()[1];
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          preLoopSeed, consumerLabel, "pre-loop scalar seed splat",
          description.scalarSeedSplatIntrinsic,
          {{expectedInitialAccumulatorLane, scalarI32CType},
           {description.reductionStoreVL, description.vlCType}},
          "dot_initial_acc_vec", description.vectorCType))
    return error;

  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopStore =
      route.getCallOpaqueSteps()[2];
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          preLoopStore, consumerLabel, "pre-loop initial output store",
          description.storeIntrinsic,
          {{outABI->cName, outABI->cType},
           {"dot_initial_acc_vec", description.vectorCType},
           {description.reductionStoreVL, description.vlCType}}))
    return error;

  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "widening dot-reduction target artifact consumer requires pre-loop "
          "statements to carry selected typed RVV source provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires exactly one "
        "provider-built runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.upperBound.expression != runtimeNABI->cName ||
      loop.upperBound.cType != runtimeNABI->cType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-built loop bounds and step to mirror runtime AVL/VL route "
        "facts");

  const std::size_t expectedLoopStepCount = isComputedMask ? 12 : 7;
  if (loop.bodySteps.size() != expectedLoopStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires exact provider-built widening dot loop statement count " +
        llvm::Twine(expectedLoopStepCount) + " before artifact export");

  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeNABI->cName) + " - " +
       description.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[0], consumerLabel, "loop setvl",
          description.setVLIntrinsic,
          {{expectedRemainingAVL, description.vlCType}},
          description.emitCLoopVLName, description.vlCType))
    return error;

  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "widening dot-reduction target artifact consumer requires loop "
          "statements to carry selected typed RVV source provenance");

  auto validateUnitSourceLoad =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          const support::RuntimeABIParameter &abi, llvm::StringRef resultName,
          llvm::StringRef stepLabel) -> llvm::Error {
    const std::string expectedPointer =
        (llvm::StringRef(abi.cName) + " + " +
         description.emitCLoopInductionName)
            .str();
    return validateRVVProviderBuiltRouteStep(
        step, consumerLabel, stepLabel, description.sourceVectorLoadIntrinsic,
        {{expectedPointer, abi.cType},
         {description.emitCLoopVLName, description.vlCType}},
        resultName, description.sourceVectorCType);
  };

  auto validateStridedSourceLoad =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          const support::RuntimeABIParameter &abi,
          const support::RuntimeABIParameter &strideABI,
          llvm::StringRef resultName, llvm::StringRef stepLabel) -> llvm::Error {
    const std::string expectedPointer =
        (llvm::StringRef(abi.cName) + " + (" +
         description.emitCLoopInductionName + " * " + strideABI.cName + ")")
            .str();
    const std::string expectedStrideBytes =
        (llvm::StringRef(strideABI.cName) + " * 2").str();
    return validateRVVProviderBuiltRouteStep(
        step, consumerLabel, stepLabel, description.stridedLoadIntrinsic,
        {{expectedPointer, abi.cType},
         {expectedStrideBytes, "ptrdiff_t"},
         {description.emitCLoopVLName, description.vlCType}},
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

  if (isComputedMask) {
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[1], consumerLabel, "compare lhs vector load",
            description.vectorLoadIntrinsic,
            {{(llvm::StringRef(cmpLHSABI->cName) + " + " +
               description.emitCLoopInductionName)
                  .str(),
              cmpLHSABI->cType},
             {description.emitCLoopVLName, description.vlCType}},
            "cmp_lhs_vec", description.vectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[2], consumerLabel, "compare rhs vector load",
            description.vectorLoadIntrinsic,
            {{(llvm::StringRef(cmpRHSABI->cName) + " + " +
               description.emitCLoopInductionName)
                  .str(),
              cmpRHSABI->cType},
             {description.emitCLoopVLName, description.vlCType}},
            "cmp_rhs_vec", description.vectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[3], consumerLabel, "compare predicate",
            description.compareIntrinsic,
            {{"cmp_lhs_vec", description.vectorCType},
             {"cmp_rhs_vec", description.vectorCType},
             {description.emitCLoopVLName, description.vlCType}},
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
             {description.emitCLoopVLName, description.vlCType}},
            "dot_zero_vec", description.vectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[7], consumerLabel, "masked widening product",
            description.maskedWideningProductIntrinsic,
            {{description.maskName, description.maskCType},
             {"dot_lhs_vec", description.sourceVectorCType},
             {"dot_rhs_vec", description.sourceVectorCType},
             {description.emitCLoopVLName, description.vlCType}},
            "active_dot_product_vec", description.vectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[8], consumerLabel, "inactive-lane merge",
            description.maskedMergeIntrinsic,
            {{"dot_zero_vec", description.vectorCType},
             {"active_dot_product_vec", description.vectorCType},
             {description.maskName, description.maskCType},
             {description.emitCLoopVLName, description.vlCType}},
            "dot_product_vec", description.vectorCType))
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
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[3], consumerLabel, "widening product",
            description.wideningProductIntrinsic,
            {{"lhs_vec", description.sourceVectorCType},
             {"rhs_vec", description.sourceVectorCType},
             {description.emitCLoopVLName, description.vlCType}},
            "dot_product_vec", description.vectorCType))
      return error;
  }

  const std::size_t seedIndex = isComputedMask ? 9 : 4;
  const std::size_t reductionIndex = isComputedMask ? 10 : 5;
  const std::size_t storeIndex = isComputedMask ? 11 : 6;
  const std::string expectedOutLane =
      (llvm::StringRef(outABI->cName) + "[0]").str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[seedIndex], consumerLabel, "loop scalar seed splat",
          description.scalarSeedSplatIntrinsic,
          {{expectedOutLane, scalarI32CType},
           {description.reductionStoreVL, description.vlCType}},
          "dot_acc_vec", description.vectorCType))
    return error;
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[reductionIndex], consumerLabel,
          "widening dot reduction", description.intrinsic,
          {{"dot_product_vec", description.vectorCType},
           {"dot_acc_vec", description.vectorCType},
           {description.emitCLoopVLName, description.vlCType}},
          description.resultName, description.vectorCType))
    return error;
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[storeIndex], consumerLabel, "output store",
          description.storeIntrinsic,
          {{outABI->cName, outABI->cType},
           {description.resultName, description.vectorCType},
           {description.reductionStoreVL, description.vlCType}}))
    return error;

  return llvm::Error::success();
}

llvm::Error validateRVVWideningDotReductionRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("widening dot-reduction target artifact consumer requires "
                    "rebuilt provider route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");

  if (description.providerSupportedMirror.empty())
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires a "
        "provider-supported mirror label after route construction");
  if (description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires provider "
        "route operand binding facts before artifact export");
  if (description.contractionRouteFamilyPlanID.empty())
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires a "
        "provider-derived contraction route-family plan before artifact "
        "export");
  if (description.typedComputeOpName.empty())
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires a "
        "provider-derived typed compute op name before artifact export");
  if (isRVVComputedMaskWideningDotReductionRouteFamilyOperation(
          description.operation)) {
    if (description.typedComputeOpName !=
        "tcrv_rvv.masked_widening_dot_reduce")
      return makeRVVTargetRouteError(
          "computed-mask widening dot-reduction target artifact consumer "
          "requires the provider-derived masked widening dot typed body op");
  } else if (description.typedComputeOpName != "tcrv_rvv.widening_dot_reduce") {
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires the "
        "provider-derived widening dot typed body op");
  }

  if (description.wideningDotProductAccumulatorLayout.empty() ||
      description.wideningDotProductResultLayout.empty() ||
      description.wideningDotProductRelation.empty() ||
      description.reductionStoreVL.empty() ||
      description.wideningProductIntrinsic.empty() ||
      description.scalarSeedSplatIntrinsic.empty() ||
      description.intrinsic.empty() || description.storeIntrinsic.empty() ||
      description.sourceVectorLoadIntrinsic.empty())
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-derived dot relation, accumulator/result layout, store-VL, "
        "source load, product, seed, reduction, and store facts before "
        "artifact export");

  if (isRVVComputedMaskWideningDotReductionRouteFamilyOperation(
          description.operation)) {
    if (description.maskRole.empty() || description.maskSource.empty() ||
        description.maskMemoryForm.empty() ||
        description.inactiveLaneZeroingRequirement.empty() ||
        description.comparePredicateKind.empty() ||
        description.compareIntrinsic.empty() ||
        description.maskedWideningProductIntrinsic.empty() ||
        description.maskedMergeIntrinsic.empty() || description.maskName.empty())
      return makeRVVTargetRouteError(
          "computed-mask widening dot-reduction target artifact consumer "
          "requires provider-derived mask, compare, masked product, merge, "
          "and inactive-lane facts before artifact export");
  } else if (!description.maskRole.empty() || !description.maskSource.empty() ||
             !description.maskMemoryForm.empty() ||
             !description.inactiveLaneZeroingRequirement.empty() ||
             !description.maskedWideningProductIntrinsic.empty()) {
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer rejects stale "
        "computed-mask widening dot facts on non-computed-mask routes");
  }

  if (isRVVStridedInputWideningDotReductionRouteFamilyOperation(
          description.operation)) {
    if (description.stridedLoadIntrinsic.empty() ||
        description.stridedMemoryLayout.empty() ||
        description.lhsStrideSource.empty() ||
        description.rhsStrideSource.empty() ||
        description.sourceMemoryForm.empty() ||
        description.destinationMemoryForm.empty())
      return makeRVVTargetRouteError(
          "strided-input widening dot-reduction target artifact consumer "
          "requires provider-derived strided load, stride ABI, source/result "
          "memory form, and layout facts before artifact export");
  } else if (!description.stridedLoadIntrinsic.empty() ||
             !description.stridedMemoryLayout.empty() ||
             !description.lhsStrideSource.empty() ||
             !description.rhsStrideSource.empty() ||
             !description.sourceMemoryForm.empty() ||
             !description.destinationMemoryForm.empty()) {
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer rejects stale "
        "strided-input facts on unit-stride widening dot routes");
  }

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
          validateRVVWideningDotReductionRouteHeaders(route, description))
    return error;
  if (llvm::Error error =
          validateRVVWideningDotReductionRouteTypeMappings(route, description))
    return error;
  if (llvm::Error error =
          validateRVVWideningDotReductionRouteABIMappings(route, description))
    return error;
  return validateRVVWideningDotReductionRouteStatementPlan(route, description);
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

llvm::Error validateRVVWideningDotReductionTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description =
      context.description;
  const std::string sourceSEW = llvm::Twine(description.sourceSEW).str();
  const std::string resultSEW = llvm::Twine(description.sew).str();

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_plan",
          description.routeOperandBindingPlanID,
          "selected typed RVV widening dot binding plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_operands",
          description.routeOperandBindingSummary,
          "selected typed RVV widening dot binding summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.provider_supported_mirror",
          description.providerSupportedMirror,
          "selected typed RVV widening dot provider support"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.contraction_route_family_plan",
          description.contractionRouteFamilyPlanID,
          "selected typed RVV widening dot contraction route-family plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.memory_form",
          plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
              description.memoryForm),
          "selected typed RVV widening dot memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_control_plan",
          description.runtimeControlPlanID,
          "selected typed RVV widening dot runtime AVL/VL control plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order", description.runtimeABIOrder,
          "selected typed RVV widening dot runtime ABI order"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.required_header_declarations",
          description.requiredHeaderDeclarations,
          "selected typed RVV widening dot route header requirements"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.c_type_mapping",
          description.cTypeMappingSummary,
          "selected typed RVV widening dot route type mapping summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.target_leaf_profile",
          description.targetLeafProfile,
          "selected typed RVV widening dot target leaf profile"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_sew", sourceSEW,
          "selected typed RVV widening dot i16 source SEW"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_lmul", description.sourceLMUL,
          "selected typed RVV widening dot source LMUL"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.accumulator_sew", resultSEW,
          "selected typed RVV widening dot accumulator SEW"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.accumulator_lmul", description.lmul,
          "selected typed RVV widening dot accumulator LMUL"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.result_sew", resultSEW,
          "selected typed RVV widening dot result SEW"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.result_lmul", description.lmul,
          "selected typed RVV widening dot result LMUL"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.widening_dot_accumulator_layout",
          description.wideningDotProductAccumulatorLayout,
          "selected typed RVV widening dot accumulator layout"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.widening_dot_result_layout",
          description.wideningDotProductResultLayout,
          "selected typed RVV widening dot result layout"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.widening_dot_relation",
          description.wideningDotProductRelation,
          "selected typed RVV widening dot relation"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.widening_dot_reduction_store_vl",
          description.reductionStoreVL,
          "selected typed RVV widening dot reduction store VL"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.widening_product_intrinsic",
          description.wideningProductIntrinsic,
          "selected typed RVV widening dot product intrinsic"))
    return error;

  if (isRVVComputedMaskWideningDotReductionRouteFamilyOperation(
          description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_role", description.maskRole,
            "selected typed RVV computed-mask widening dot mask role"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_source", description.maskSource,
            "selected typed RVV computed-mask widening dot mask source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_memory_form",
            description.maskMemoryForm,
            "selected typed RVV computed-mask widening dot mask memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.inactive_lane_zeroing_requirement",
            description.inactiveLaneZeroingRequirement,
            "selected typed RVV computed-mask widening dot inactive lane "
            "zeroing requirement"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.compare_predicate_kind",
            description.comparePredicateKind,
            "selected typed RVV computed-mask widening dot compare predicate"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.masked_widening_product_intrinsic",
            description.maskedWideningProductIntrinsic,
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
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.compare_predicate_kind", "",
            "selected typed RVV computed-mask widening dot compare predicate"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.masked_widening_product_intrinsic", "",
            "selected typed RVV computed-mask widening dot product intrinsic"))
      return error;
  }

  if (isRVVStridedInputWideningDotReductionRouteFamilyOperation(
          description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.strided_memory_layout",
            description.stridedMemoryLayout,
            "selected typed RVV strided widening dot memory layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.lhs_stride_source",
            description.lhsStrideSource,
            "selected typed RVV strided widening dot lhs stride source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.rhs_stride_source",
            description.rhsStrideSource,
            "selected typed RVV strided widening dot rhs stride source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.source_memory_form",
            description.sourceMemoryForm,
            "selected typed RVV strided widening dot source memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.destination_memory_form",
            description.destinationMemoryForm,
            "selected typed RVV strided widening dot destination memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.strided_load_intrinsic",
            description.stridedLoadIntrinsic,
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
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.source_memory_form", "",
            "selected typed RVV strided widening dot source memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.destination_memory_form", "",
            "selected typed RVV strided widening dot destination memory form"))
      return error;
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

llvm::Error validateRVVMAccRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-derived "
        "required_header_declarations before accepting the route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires at least one provider route "
        "header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "MAcc target artifact consumer saw an empty provider route header "
          "declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("MAcc target artifact consumer requires rebuilt "
                      "provider route header '") +
          trimmed + "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVMAccRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.vlCType.empty() || description.vectorTypeName.empty() ||
      description.vectorCType.empty() || description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-derived VL, vector, "
        "and C type mapping facts before artifact export");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("MAcc target artifact consumer requires rebuilt provider "
                    "route type mapping '!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("MAcc target artifact consumer requires rebuilt provider "
                    "route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");

  if (isRVVComputedMaskMAccRouteFamilyOperation(description.operation)) {
    if (description.maskTypeName.empty() || description.maskCType.empty())
      return makeRVVTargetRouteError(
          "computed-mask MAcc target artifact consumer requires "
          "provider-derived mask type mapping facts before artifact export");
    if (!routeHasTypeMapping(route, description.maskTypeName,
                             description.maskCType))
      return makeRVVTargetRouteError(
          llvm::Twine("computed-mask MAcc target artifact consumer requires "
                      "rebuilt provider route type mapping '") +
          description.maskTypeName + "' -> '" + description.maskCType + "'");
  } else if (!description.maskTypeName.empty() ||
             !description.maskCType.empty()) {
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer rejects stale mask type mapping facts "
        "for non-computed-mask MAcc routes");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVMAccRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.runtimeABIOrder.empty() ||
      description.runtimeABIParameters.empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-derived runtime ABI "
        "order and ABI parameters before artifact export");
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("MAcc target artifact consumer requires rebuilt provider "
                    "route ABI mapping count ") +
        llvm::Twine(description.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        description.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine("MAcc target artifact consumer requires rebuilt provider "
                      "route ABI mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("MAcc target artifact consumer requires rebuilt provider "
                      "route ABI mapping[") +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

constexpr llvm::StringLiteral kRVVMAccRuntimeABIOrder("lhs,rhs,acc,out,n");
constexpr llvm::StringLiteral kRVVScalarBroadcastMAccRuntimeABIOrder(
    "lhs,rhs_scalar,acc,out,n");
constexpr llvm::StringLiteral kRVVMAccOperandBindingPlanID(
    "rvv-route-operand-binding:macc_add.v1");
constexpr llvm::StringLiteral kRVVPlainMAccOperandBindingSummary(
    "rvv-route-operand-binding:macc_add.v1;"
    "lhs=lhs-input-buffer:lhs:abi|lhs-load|macc-lhs|hdr;"
    "rhs=rhs-input-buffer:rhs:abi|rhs-load|macc-rhs|hdr;"
    "acc=accumulator-input-buffer:acc:abi|acc-load|macc-acc|macc-pass|hdr;"
    "out=output-buffer:out:abi|store|hdr;"
    "n=runtime-element-count:n:abi|setvl-avl|loop|hdr");
constexpr llvm::StringLiteral kRVVScalarBroadcastMAccOperandBindingPlanID(
    "rvv-route-operand-binding:scalar_broadcast_macc_add.v1");
constexpr llvm::StringLiteral kRVVScalarBroadcastMAccOperandBindingSummary(
    "rvv-route-operand-binding:scalar_broadcast_macc_add.v1;"
    "lhs=lhs-input-buffer:lhs:abi|lhs-load|macc-lhs|hdr;"
    "rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|macc-rhs|hdr;"
    "acc=accumulator-input-buffer:acc:abi|acc-load|macc-acc|macc-pass|hdr;"
    "out=output-buffer:out:abi|store|hdr;"
    "n=runtime-element-count:n:abi|setvl-avl|loop|hdr");
constexpr llvm::StringLiteral kRVVPlainMAccRouteFamilyPlanID(
    "rvv-plain-macc-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVScalarBroadcastMAccRouteFamilyPlanID(
    "rvv-scalar-broadcast-macc-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVMAccComputedMaskAccumulationRouteFamilyPlanID(
    "rvv-computed-mask-accumulation-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVPlainMAccTargetLeafProfile(
    "rvv-v1-typed-plain-macc-add-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVScalarBroadcastMAccTargetLeafProfile(
    "rvv-v1-typed-scalar-broadcast-macc-add-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVPlainMAccProviderSupportedMirror(
    "provider_supported_mirror:rvv-plain-macc-add-plan-validated");
constexpr llvm::StringLiteral kRVVScalarBroadcastMAccProviderSupportedMirror(
    "provider_supported_mirror:rvv-scalar-broadcast-macc-add-composition-plan-validated");
constexpr llvm::StringLiteral kRVVMAccRequiredHeaderDeclarations(
    "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVPlainMAccCTypeMappingSummary(
    "vl:size_t,lhs/rhs/acc:typed-vector,result:typed-vector");
constexpr llvm::StringLiteral kRVVScalarBroadcastMAccCTypeMappingSummary(
    "vl:size_t,lhs/acc:typed-vector,rhs_scalar:typed-scalar,result:typed-vector");
constexpr llvm::StringLiteral kRVVMAccAccumulatorLayout(
    "separate-i32-vector-accumulator-input");
constexpr llvm::StringLiteral kRVVMAccResultLayout(
    "store-multiply-accumulate-result-to-output-buffer");
constexpr llvm::StringLiteral kRVVComputedMaskedMAccComputeSuffix(
    "vector-masked-macc-add");
constexpr llvm::StringLiteral kRVVComputedMaskedMAccMaskProducerSource(
    "vector-compare-rhs-load");
constexpr llvm::StringLiteral kRVVComputedMaskedMAccAccumulatorContract(
    "vector-accumulator-input-preserves-inactive-lanes");
constexpr llvm::StringLiteral kRVVComputedMaskedMAccResultContract(
    "vector-macc-result-stored-to-output-buffer");
constexpr llvm::StringLiteral kRVVComputedMaskedMAccMaskRole(
    "predicate-mask-produced-by-compare");
constexpr llvm::StringLiteral kRVVComputedMaskedMAccMaskSource(
    "compare-produced-mask-same-vl-scope");
constexpr llvm::StringLiteral kRVVComputedMaskedMAccMaskMemoryForm(
    "compare-produced-mask");
constexpr llvm::StringLiteral kRVVComputedMaskedMAccInactiveLaneContract(
    "masked-macc-false-lanes-preserve-accumulator");
constexpr llvm::StringLiteral kRVVComputedMaskedMAccPassthroughLayout(
    "accumulator-vector-preserves-inactive-lanes");
constexpr llvm::StringLiteral kRVVPlainMAccTypedComputeOp("tcrv_rvv.macc");
constexpr llvm::StringLiteral kRVVMAccSourceMemoryForm("unit-stride-load");
constexpr llvm::StringLiteral kRVVMAccDestinationMemoryForm(
    "unit-stride-store");
constexpr llvm::StringLiteral kRVVComputedMaskedMAccMemoryLayout(
    "unit-stride-compare-lhs-rhs-accumulator-masked-macc-output-runtime-abi");
constexpr llvm::StringLiteral kRVVRuntimeScalarComputedMaskedMAccMemoryLayout(
    "unit-stride-compare-lhs-runtime-scalar-threshold-lhs-rhs-accumulator-masked-macc-output-runtime-abi");

llvm::StringRef getRVVMAccExpectedTypedComputeOp(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::MAccAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd:
    return kRVVPlainMAccTypedComputeOp;
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd: {
    std::optional<plugin::rvv::RVVComputedMaskMAccRouteFacts> routeFacts =
        plugin::rvv::getRVVComputedMaskMAccRouteFacts(operation);
    return routeFacts ? routeFacts->typedComputeOpName : llvm::StringRef();
  }
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskedMAccAdd: {
    std::optional<plugin::rvv::RVVRuntimeScalarComputedMaskMAccRouteFacts>
        routeFacts =
            plugin::rvv::getRVVRuntimeScalarComputedMaskMAccRouteFacts(
                operation);
    return routeFacts ? routeFacts->typedComputeOpName : llvm::StringRef();
  }
  default:
    return {};
  }
}

llvm::StringRef getRVVMAccExpectedComparePredicateKind(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd: {
    std::optional<plugin::rvv::RVVComputedMaskMAccRouteFacts> routeFacts =
        plugin::rvv::getRVVComputedMaskMAccRouteFacts(operation);
    return routeFacts ? routeFacts->comparePredicateKind : llvm::StringRef();
  }
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskedMAccAdd: {
    std::optional<plugin::rvv::RVVRuntimeScalarComputedMaskMAccRouteFacts>
        routeFacts =
            plugin::rvv::getRVVRuntimeScalarComputedMaskMAccRouteFacts(
                operation);
    return routeFacts ? routeFacts->comparePredicateKind : llvm::StringRef();
  }
  default:
    return {};
  }
}

llvm::StringRef getRVVMAccExpectedRuntimeABIOrder(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::MAccAdd:
    return kRVVMAccRuntimeABIOrder;
  case plugin::rvv::RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd:
    return kRVVScalarBroadcastMAccRuntimeABIOrder;
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd: {
    std::optional<plugin::rvv::RVVComputedMaskMAccRouteFacts> routeFacts =
        plugin::rvv::getRVVComputedMaskMAccRouteFacts(operation);
    return routeFacts ? routeFacts->runtimeABIOrder : llvm::StringRef();
  }
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskedMAccAdd: {
    std::optional<plugin::rvv::RVVRuntimeScalarComputedMaskMAccRouteFacts>
        routeFacts =
            plugin::rvv::getRVVRuntimeScalarComputedMaskMAccRouteFacts(
                operation);
    return routeFacts ? routeFacts->runtimeABIOrder : llvm::StringRef();
  }
  default:
    return {};
  }
}

llvm::StringRef getRVVMAccExpectedOperandBindingPlanID(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::MAccAdd:
    return kRVVMAccOperandBindingPlanID;
  case plugin::rvv::RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd:
    return kRVVScalarBroadcastMAccOperandBindingPlanID;
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd: {
    std::optional<plugin::rvv::RVVComputedMaskMAccRouteFacts> routeFacts =
        plugin::rvv::getRVVComputedMaskMAccRouteFacts(operation);
    return routeFacts ? routeFacts->routeOperandBindingPlanID
                      : llvm::StringRef();
  }
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskedMAccAdd: {
    std::optional<plugin::rvv::RVVRuntimeScalarComputedMaskMAccRouteFacts>
        routeFacts =
            plugin::rvv::getRVVRuntimeScalarComputedMaskMAccRouteFacts(
                operation);
    return routeFacts ? routeFacts->routeOperandBindingPlanID
                      : llvm::StringRef();
  }
  default:
    return {};
  }
}

llvm::StringRef getRVVMAccExpectedTargetLeafProfile(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::MAccAdd:
    return kRVVPlainMAccTargetLeafProfile;
  case plugin::rvv::RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd:
    return kRVVScalarBroadcastMAccTargetLeafProfile;
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd: {
    std::optional<plugin::rvv::RVVComputedMaskMAccRouteFacts> routeFacts =
        plugin::rvv::getRVVComputedMaskMAccRouteFacts(operation);
    return routeFacts ? routeFacts->targetLeafProfile : llvm::StringRef();
  }
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskedMAccAdd: {
    std::optional<plugin::rvv::RVVRuntimeScalarComputedMaskMAccRouteFacts>
        routeFacts =
            plugin::rvv::getRVVRuntimeScalarComputedMaskMAccRouteFacts(
                operation);
    return routeFacts ? routeFacts->targetLeafProfile : llvm::StringRef();
  }
  default:
    return {};
  }
}

llvm::StringRef getRVVMAccExpectedProviderSupportedMirror(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::MAccAdd:
    return kRVVPlainMAccProviderSupportedMirror;
  case plugin::rvv::RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd:
    return kRVVScalarBroadcastMAccProviderSupportedMirror;
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd: {
    std::optional<plugin::rvv::RVVComputedMaskMAccRouteFacts> routeFacts =
        plugin::rvv::getRVVComputedMaskMAccRouteFacts(operation);
    return routeFacts ? routeFacts->providerSupportedMirror
                      : llvm::StringRef();
  }
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskedMAccAdd: {
    std::optional<plugin::rvv::RVVRuntimeScalarComputedMaskMAccRouteFacts>
        routeFacts =
            plugin::rvv::getRVVRuntimeScalarComputedMaskMAccRouteFacts(
                operation);
    return routeFacts ? routeFacts->providerSupportedMirror
                      : llvm::StringRef();
  }
  default:
    return {};
  }
}

llvm::StringRef getRVVMAccExpectedCTypeMappingSummary(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::MAccAdd:
    return kRVVPlainMAccCTypeMappingSummary;
  case plugin::rvv::RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd:
    return kRVVScalarBroadcastMAccCTypeMappingSummary;
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd: {
    std::optional<plugin::rvv::RVVComputedMaskMAccRouteFacts> routeFacts =
        plugin::rvv::getRVVComputedMaskMAccRouteFacts(operation);
    return routeFacts ? routeFacts->cTypeMappingSummary : llvm::StringRef();
  }
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskedMAccAdd: {
    std::optional<plugin::rvv::RVVRuntimeScalarComputedMaskMAccRouteFacts>
        routeFacts =
            plugin::rvv::getRVVRuntimeScalarComputedMaskMAccRouteFacts(
                operation);
    return routeFacts ? routeFacts->cTypeMappingSummary : llvm::StringRef();
  }
  default:
    return {};
  }
}

llvm::Error validateRVVMAccRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  constexpr llvm::StringLiteral consumerLabel("MAcc target artifact consumer");
  const bool isScalarBroadcast =
      isRVVScalarBroadcastMAccRouteFamilyOperation(description.operation);
  const bool isComputedMask =
      isRVVComputedMaskMAccRouteFamilyOperation(description.operation);
  const bool isRuntimeScalarComputedMask =
      isRVVRuntimeScalarComputedMaskMAccRouteFamilyOperation(
          description.operation);
  const size_t expectedABIParameterCount = isComputedMask ? 7 : 5;
  if (description.runtimeABIParameters.size() != expectedABIParameterCount)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived runtime ABI parameters before validating "
        "route statements");
  if (description.resultName.empty() || description.vectorCType.empty() ||
      description.vlCType.empty() || description.setVLIntrinsic.empty() ||
      description.vectorLoadIntrinsic.empty() || description.intrinsic.empty() ||
      description.storeIntrinsic.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived result, vector C type, VL C type, setvl, "
        "load, MAcc, and store facts before validating route statements");
  if (isScalarBroadcast && description.rhsBroadcastIntrinsic.empty())
    return makeRVVTargetRouteError(
        "scalar-broadcast MAcc target artifact consumer requires "
        "provider-derived RHS scalar broadcast facts before validating route "
        "statements");
  if (isComputedMask &&
      (description.maskName.empty() || description.maskCType.empty() ||
       description.compareIntrinsic.empty() ||
       description.maskedMergeIntrinsic.empty()))
    return makeRVVTargetRouteError(
        "computed-mask MAcc target artifact consumer requires provider-derived "
        "mask, compare, and masked-merge facts before validating route "
        "statements");
  if (isRuntimeScalarComputedMask && description.rhsBroadcastIntrinsic.empty())
    return makeRVVTargetRouteError(
        "runtime-scalar computed-mask MAcc target artifact consumer requires "
        "provider-derived RHS scalar splat facts before validating route "
        "statements");

  const support::RuntimeABIParameter *runtimeElementCount =
      findRuntimeElementCountABIParameter(description);
  if (!runtimeElementCount)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires a provider-derived runtime n/AVL ABI parameter before "
        "artifact export");

  const support::RuntimeABIParameter &runtimeNABI =
      description.runtimeABIParameters.back();
  if (runtimeElementCount != &runtimeNABI)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires runtime n/AVL ABI role to match provider runtime ABI order "
        "before validating route statements");

  if (route.getCallOpaqueSteps().size() != 1)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires exactly one provider-built pre-loop setvl statement before "
        "artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          preLoopSetVL, consumerLabel, "pre-loop setvl",
          description.setVLIntrinsic, {{runtimeNABI.cName, runtimeNABI.cType}},
          description.emitCFullChunkVLName, description.vlCType))
    return error;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "MAcc target artifact consumer requires pre-loop statements to "
          "carry selected typed RVV source provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires exactly one provider-built "
        "runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.upperBound.expression != runtimeNABI.cName ||
      loop.upperBound.cType != runtimeNABI.cType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-built loop bounds "
        "and step to mirror runtime AVL/VL route facts");
  const size_t expectedLoopBodyStepCount = isComputedMask ? 10 : 6;
  if (loop.bodySteps.size() != expectedLoopBodyStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires exact provider-built MAcc loop statement count " +
        llvm::Twine(expectedLoopBodyStepCount) + " before artifact export");

  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeNABI.cName) + " - " +
       description.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[0], consumerLabel, "loop setvl",
          description.setVLIntrinsic, {{expectedRemainingAVL, description.vlCType}},
          description.emitCLoopVLName, description.vlCType))
    return error;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "MAcc target artifact consumer requires loop statements to carry "
          "selected typed RVV source provenance");

  auto advancedPointer = [&](const support::RuntimeABIParameter &abi) {
    return (llvm::StringRef(abi.cName) + " + " +
            description.emitCLoopInductionName)
        .str();
  };
  auto validateVectorLoad =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          const support::RuntimeABIParameter &abi, llvm::StringRef resultName,
          llvm::StringRef stepLabel) -> llvm::Error {
    const std::string pointer = advancedPointer(abi);
    return validateRVVProviderBuiltRouteStep(
        step, consumerLabel, stepLabel, description.vectorLoadIntrinsic,
        {{pointer, abi.cType}, {description.emitCLoopVLName, description.vlCType}},
        resultName, description.vectorCType);
  };
  auto validateOutputStore =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          const support::RuntimeABIParameter &outABI) -> llvm::Error {
    const std::string pointer = advancedPointer(outABI);
    return validateRVVProviderBuiltRouteStep(
        step, consumerLabel, "output store", description.storeIntrinsic,
        {{pointer, outABI.cType},
         {description.resultName, description.vectorCType},
         {description.emitCLoopVLName, description.vlCType}});
  };

  if (!isComputedMask) {
    const support::RuntimeABIParameter &lhsABI =
        description.runtimeABIParameters[0];
    const support::RuntimeABIParameter &rhsOrScalarABI =
        description.runtimeABIParameters[1];
    const support::RuntimeABIParameter &accumulatorABI =
        description.runtimeABIParameters[2];
    const support::RuntimeABIParameter &outABI =
        description.runtimeABIParameters[3];

    if (llvm::Error error =
            validateVectorLoad(loop.bodySteps[1], lhsABI, "lhs_vec",
                               "lhs load"))
      return error;
    if (isScalarBroadcast) {
      if (llvm::Error error = validateRVVProviderBuiltRouteStep(
              loop.bodySteps[2], consumerLabel, "RHS scalar splat",
              description.rhsBroadcastIntrinsic,
              {{rhsOrScalarABI.cName, rhsOrScalarABI.cType},
               {description.emitCLoopVLName, description.vlCType}},
              "rhs_vec", description.vectorCType))
        return error;
    } else if (llvm::Error error =
                   validateVectorLoad(loop.bodySteps[2], rhsOrScalarABI,
                                      "rhs_vec", "rhs load")) {
      return error;
    }
    if (llvm::Error error = validateVectorLoad(
            loop.bodySteps[3], accumulatorABI, "acc_vec", "accumulator load"))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[4], consumerLabel, "MAcc compute",
            description.intrinsic,
            {{"acc_vec", description.vectorCType},
             {"lhs_vec", description.vectorCType},
             {"rhs_vec", description.vectorCType},
             {description.emitCLoopVLName, description.vlCType}},
            description.resultName, description.vectorCType))
      return error;
    return validateOutputStore(loop.bodySteps[5], outABI);
  }

  const support::RuntimeABIParameter &compareLhsABI =
      description.runtimeABIParameters[0];
  const support::RuntimeABIParameter &compareRhsOrScalarABI =
      description.runtimeABIParameters[1];
  const support::RuntimeABIParameter &lhsABI =
      description.runtimeABIParameters[2];
  const support::RuntimeABIParameter &rhsABI =
      description.runtimeABIParameters[3];
  const support::RuntimeABIParameter &accumulatorABI =
      description.runtimeABIParameters[4];
  const support::RuntimeABIParameter &outABI =
      description.runtimeABIParameters[5];

  if (llvm::Error error = validateVectorLoad(
          loop.bodySteps[1], compareLhsABI, "lhs_vec", "compare lhs load"))
    return error;
  if (isRuntimeScalarComputedMask) {
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[2], consumerLabel, "RHS scalar splat",
            description.rhsBroadcastIntrinsic,
            {{compareRhsOrScalarABI.cName, compareRhsOrScalarABI.cType},
             {description.emitCLoopVLName, description.vlCType}},
            "rhs_vec", description.vectorCType))
      return error;
  } else if (llvm::Error error = validateVectorLoad(
                 loop.bodySteps[2], compareRhsOrScalarABI, "rhs_vec",
                 "compare rhs load")) {
    return error;
  }
  if (llvm::Error error =
          validateVectorLoad(loop.bodySteps[3], lhsABI, "macc_lhs_vec",
                             "MAcc lhs load"))
    return error;
  if (llvm::Error error =
          validateVectorLoad(loop.bodySteps[4], rhsABI, "macc_rhs_vec",
                             "MAcc rhs load"))
    return error;
  if (llvm::Error error = validateVectorLoad(
          loop.bodySteps[5], accumulatorABI, "acc_vec", "accumulator load"))
    return error;
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[6], consumerLabel, "compare mask",
          description.compareIntrinsic,
          {{"lhs_vec", description.vectorCType},
           {"rhs_vec", description.vectorCType},
           {description.emitCLoopVLName, description.vlCType}},
          description.maskName, description.maskCType))
    return error;
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[7], consumerLabel, "active MAcc",
          description.intrinsic,
          {{"acc_vec", description.vectorCType},
           {"macc_lhs_vec", description.vectorCType},
           {"macc_rhs_vec", description.vectorCType},
           {description.emitCLoopVLName, description.vlCType}},
          "active_macc_vec", description.vectorCType))
    return error;
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[8], consumerLabel, "masked merge",
          description.maskedMergeIntrinsic,
          {{"acc_vec", description.vectorCType},
           {"active_macc_vec", description.vectorCType},
           {description.maskName, description.maskCType},
           {description.emitCLoopVLName, description.vlCType}},
          description.resultName, description.vectorCType))
    return error;
  if (llvm::Error error = validateOutputStore(loop.bodySteps[9], outABI))
    return error;

  return llvm::Error::success();
}

llvm::Error validateRVVMAccRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  const std::optional<plugin::rvv::RVVComputedMaskMAccRouteFacts>
      computedMaskMAccFacts =
          plugin::rvv::getRVVComputedMaskMAccRouteFacts(
              description.operation);
  const std::optional<
      plugin::rvv::RVVRuntimeScalarComputedMaskMAccRouteFacts>
      runtimeScalarMAccFacts =
          plugin::rvv::getRVVRuntimeScalarComputedMaskMAccRouteFacts(
              description.operation);
  const llvm::StringRef expectedComputedMaskMAccBindingSummary =
      computedMaskMAccFacts
          ? llvm::StringRef(computedMaskMAccFacts->routeOperandBindingSummary)
          : llvm::StringRef();
  const llvm::StringRef expectedRuntimeScalarMAccBindingSummary =
      runtimeScalarMAccFacts
          ? llvm::StringRef(runtimeScalarMAccFacts->routeOperandBindingSummary)
          : llvm::StringRef();
  if (description.operation ==
          plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd &&
      !computedMaskMAccFacts)
    return makeRVVTargetRouteError(
        "computed-mask MAcc target artifact consumer requires "
        "provider-owned computed-mask MAcc route facts before artifact export");

  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("MAcc target artifact consumer requires rebuilt provider "
                    "route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");

  if (description.providerSupportedMirror.empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires a provider-supported mirror "
        "label after route construction");
  if (description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider route operand "
        "binding facts before artifact export");
  if (description.maccAccumulatorLayout.empty() ||
      description.maccResultLayout.empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-derived accumulator "
        "and result layout facts before artifact export");

  const llvm::StringRef expectedRuntimeABIOrder =
      getRVVMAccExpectedRuntimeABIOrder(description.operation);
  const llvm::StringRef expectedOperandBindingPlanID =
      getRVVMAccExpectedOperandBindingPlanID(description.operation);
  const llvm::StringRef expectedTargetLeafProfile =
      getRVVMAccExpectedTargetLeafProfile(description.operation);
  const llvm::StringRef expectedProviderSupportedMirror =
      getRVVMAccExpectedProviderSupportedMirror(description.operation);
  const llvm::StringRef expectedCTypeMappingSummary =
      getRVVMAccExpectedCTypeMappingSummary(description.operation);
  const llvm::StringRef expectedTypedComputeOp =
      getRVVMAccExpectedTypedComputeOp(description.operation);
  const llvm::StringRef expectedComparePredicateKind =
      getRVVMAccExpectedComparePredicateKind(description.operation);
  if (expectedRuntimeABIOrder.empty() || expectedOperandBindingPlanID.empty() ||
      expectedTargetLeafProfile.empty() ||
      expectedProviderSupportedMirror.empty() ||
      expectedCTypeMappingSummary.empty() || expectedTypedComputeOp.empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires a known provider MAcc route "
        "family before artifact export");
  if (description.typedComputeOpName != expectedTypedComputeOp)
    return makeRVVTargetRouteError(
        llvm::Twine("MAcc target artifact consumer requires typed compute op '") +
        expectedTypedComputeOp + "' but was '" +
        description.typedComputeOpName + "'");
  if (!expectedComparePredicateKind.empty() &&
      description.comparePredicateKind != expectedComparePredicateKind)
    return makeRVVTargetRouteError(
        llvm::Twine("computed-mask MAcc target artifact consumer requires "
                    "provider-derived compare predicate '") +
        expectedComparePredicateKind + "' but was '" +
        description.comparePredicateKind + "'");
  if (!description.accumulationScalarCarryContract.empty() ||
      !description.standaloneReductionRouteFamilyPlanID.empty() ||
      !description.reductionAccumulatorLayout.empty() ||
      !description.reductionResultLayout.empty() ||
      !description.reductionStoreVL.empty() ||
      !description.standaloneReductionScalarResultRuntimeBoundary.empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer rejects stale standalone-reduction "
        "or scalar-carry facts before artifact export");
  if (description.runtimeABIOrder != expectedRuntimeABIOrder)
    return makeRVVTargetRouteError(
        llvm::Twine("MAcc target artifact consumer requires provider-derived "
                    "runtime ABI order '") +
        expectedRuntimeABIOrder + "' but was '" + description.runtimeABIOrder +
        "'");
  if (description.routeOperandBindingPlanID != expectedOperandBindingPlanID)
    return makeRVVTargetRouteError(
        llvm::Twine("MAcc target artifact consumer requires provider route "
                    "operand binding plan '") +
        expectedOperandBindingPlanID + "' but was '" +
        description.routeOperandBindingPlanID + "'");
  if (description.operation ==
          plugin::rvv::RVVSelectedBodyOperationKind::MAccAdd &&
      description.routeOperandBindingSummary != kRVVPlainMAccOperandBindingSummary)
    return makeRVVTargetRouteError(
        llvm::Twine("plain MAcc target artifact consumer requires provider "
                    "route operand binding summary '") +
        kRVVPlainMAccOperandBindingSummary + "' but was '" +
        description.routeOperandBindingSummary + "'");
  if (isRVVScalarBroadcastMAccRouteFamilyOperation(description.operation) &&
      description.routeOperandBindingSummary !=
          kRVVScalarBroadcastMAccOperandBindingSummary)
    return makeRVVTargetRouteError(
        llvm::Twine("scalar-broadcast MAcc target artifact consumer requires "
                    "provider route operand binding summary '") +
        kRVVScalarBroadcastMAccOperandBindingSummary + "' but was '" +
        description.routeOperandBindingSummary + "'");
  if (description.operation ==
          plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd &&
      (!computedMaskMAccFacts ||
       description.routeOperandBindingSummary !=
           expectedComputedMaskMAccBindingSummary))
    return makeRVVTargetRouteError(
        llvm::Twine("computed-mask MAcc target artifact consumer requires "
                    "provider route operand binding summary '") +
        (computedMaskMAccFacts
             ? expectedComputedMaskMAccBindingSummary
             : llvm::StringRef("<missing-canonical-route-facts>")) +
        "' but was '" +
        description.routeOperandBindingSummary + "'");
  if (description.operation == plugin::rvv::RVVSelectedBodyOperationKind::
                                   RuntimeScalarComputedMaskedMAccAdd &&
      (!runtimeScalarMAccFacts ||
       description.routeOperandBindingSummary !=
           expectedRuntimeScalarMAccBindingSummary))
    return makeRVVTargetRouteError(
        llvm::Twine("runtime-scalar computed-mask MAcc target artifact "
                    "consumer requires provider route operand binding summary '") +
        (runtimeScalarMAccFacts
             ? expectedRuntimeScalarMAccBindingSummary
             : llvm::StringRef("<missing-canonical-route-facts>")) +
        "' but was '" + description.routeOperandBindingSummary + "'");
  if (description.targetLeafProfile != expectedTargetLeafProfile)
    return makeRVVTargetRouteError(
        llvm::Twine("MAcc target artifact consumer requires target leaf profile "
                    "'") +
        expectedTargetLeafProfile + "' but was '" +
        description.targetLeafProfile + "'");
  if (description.providerSupportedMirror != expectedProviderSupportedMirror)
    return makeRVVTargetRouteError(
        llvm::Twine("MAcc target artifact consumer requires provider-supported "
                    "mirror '") +
        expectedProviderSupportedMirror + "' but was '" +
        description.providerSupportedMirror + "'");
  if (description.requiredHeaderDeclarations != kRVVMAccRequiredHeaderDeclarations)
    return makeRVVTargetRouteError(
        llvm::Twine("MAcc target artifact consumer requires required header "
                    "declarations '") +
        kRVVMAccRequiredHeaderDeclarations + "' but was '" +
        description.requiredHeaderDeclarations + "'");
  if (description.cTypeMappingSummary != expectedCTypeMappingSummary)
    return makeRVVTargetRouteError(
        llvm::Twine("MAcc target artifact consumer requires C type mapping "
                    "summary '") +
        expectedCTypeMappingSummary + "' but was '" +
        description.cTypeMappingSummary + "'");
  const llvm::StringRef expectedMAccAccumulatorLayout =
      computedMaskMAccFacts ? computedMaskMAccFacts->maccAccumulatorLayout
      : runtimeScalarMAccFacts ? runtimeScalarMAccFacts->maccAccumulatorLayout
                             : llvm::StringRef(kRVVMAccAccumulatorLayout);
  const llvm::StringRef expectedMAccResultLayout =
      computedMaskMAccFacts ? computedMaskMAccFacts->maccResultLayout
      : runtimeScalarMAccFacts ? runtimeScalarMAccFacts->maccResultLayout
                             : llvm::StringRef(kRVVMAccResultLayout);
  if (description.maccAccumulatorLayout != expectedMAccAccumulatorLayout ||
      description.maccResultLayout != expectedMAccResultLayout)
    return makeRVVTargetRouteError(
        llvm::Twine("MAcc target artifact consumer requires accumulator layout "
                    "'") +
        expectedMAccAccumulatorLayout + "' and result layout '" +
        expectedMAccResultLayout + "' but saw accumulator '" +
        description.maccAccumulatorLayout + "' and result '" +
        description.maccResultLayout + "'");

  if (description.operation ==
      plugin::rvv::RVVSelectedBodyOperationKind::MAccAdd) {
    if (description.plainMAccRouteFamilyPlanID != kRVVPlainMAccRouteFamilyPlanID)
      return makeRVVTargetRouteError(
          llvm::Twine("plain MAcc target artifact consumer requires provider "
                      "route-family plan '") +
          kRVVPlainMAccRouteFamilyPlanID + "' but was '" +
          description.plainMAccRouteFamilyPlanID + "'");
  } else if (isRVVScalarBroadcastMAccRouteFamilyOperation(
                 description.operation)) {
    if (description.scalarBroadcastMAccRouteFamilyPlanID !=
        kRVVScalarBroadcastMAccRouteFamilyPlanID)
      return makeRVVTargetRouteError(
          llvm::Twine("scalar-broadcast MAcc target artifact consumer requires "
                      "provider route-family plan '") +
          kRVVScalarBroadcastMAccRouteFamilyPlanID + "' but was '" +
          description.scalarBroadcastMAccRouteFamilyPlanID + "'");
    if (description.rhsBroadcastIntrinsic.empty())
      return makeRVVTargetRouteError(
          "scalar-broadcast MAcc target artifact consumer requires a "
          "provider-derived RHS scalar broadcast intrinsic before artifact "
          "export");
  } else if (isRVVComputedMaskMAccRouteFamilyOperation(
                 description.operation)) {
    const llvm::StringRef expectedAccumulationRouteFamilyPlanID =
        computedMaskMAccFacts
            ? computedMaskMAccFacts->accumulationRouteFamilyPlanID
        : runtimeScalarMAccFacts
            ? runtimeScalarMAccFacts->accumulationRouteFamilyPlanID
            : llvm::StringRef(kRVVMAccComputedMaskAccumulationRouteFamilyPlanID);
    if (description.accumulationRouteFamilyPlanID !=
        expectedAccumulationRouteFamilyPlanID)
      return makeRVVTargetRouteError(
          llvm::Twine("computed-mask MAcc target artifact consumer requires "
                      "provider accumulation route-family plan '") +
          expectedAccumulationRouteFamilyPlanID + "' but was '" +
          description.accumulationRouteFamilyPlanID + "'");
    const llvm::StringRef expectedAccumulationComputeSuffix =
        computedMaskMAccFacts
            ? computedMaskMAccFacts->accumulationComputeSuffix
        : runtimeScalarMAccFacts
            ? runtimeScalarMAccFacts->accumulationComputeSuffix
            : llvm::StringRef(kRVVComputedMaskedMAccComputeSuffix);
    const llvm::StringRef expectedMaskProducer =
        computedMaskMAccFacts
            ? computedMaskMAccFacts->accumulationMaskProducerSource
        : runtimeScalarMAccFacts
            ? runtimeScalarMAccFacts->accumulationMaskProducerSource
            : llvm::StringRef(kRVVComputedMaskedMAccMaskProducerSource);
    const llvm::StringRef expectedAccumulationAccumulatorContract =
        computedMaskMAccFacts
            ? computedMaskMAccFacts->accumulationAccumulatorContract
        : runtimeScalarMAccFacts
            ? runtimeScalarMAccFacts->accumulationAccumulatorContract
            : llvm::StringRef(kRVVComputedMaskedMAccAccumulatorContract);
    const llvm::StringRef expectedAccumulationResultContract =
        computedMaskMAccFacts
            ? computedMaskMAccFacts->accumulationResultContract
        : runtimeScalarMAccFacts
            ? runtimeScalarMAccFacts->accumulationResultContract
            : llvm::StringRef(kRVVComputedMaskedMAccResultContract);
    const llvm::StringRef expectedMaskRole =
        computedMaskMAccFacts ? computedMaskMAccFacts->maskRole
        : runtimeScalarMAccFacts ? runtimeScalarMAccFacts->maskRole
                               : llvm::StringRef(kRVVComputedMaskedMAccMaskRole);
    const llvm::StringRef expectedMaskSource =
        computedMaskMAccFacts
            ? computedMaskMAccFacts->maskSource
        : runtimeScalarMAccFacts
            ? runtimeScalarMAccFacts->maskSource
            : llvm::StringRef(kRVVComputedMaskedMAccMaskSource);
    const llvm::StringRef expectedMaskMemoryForm =
        computedMaskMAccFacts
            ? computedMaskMAccFacts->maskMemoryForm
        : runtimeScalarMAccFacts
            ? runtimeScalarMAccFacts->maskMemoryForm
            : llvm::StringRef(kRVVComputedMaskedMAccMaskMemoryForm);
    const llvm::StringRef expectedInactiveLaneContract =
        computedMaskMAccFacts
            ? computedMaskMAccFacts->inactiveLaneContract
        : runtimeScalarMAccFacts
            ? runtimeScalarMAccFacts->inactiveLaneContract
            : llvm::StringRef(kRVVComputedMaskedMAccInactiveLaneContract);
    const llvm::StringRef expectedMaskedPassthroughLayout =
        computedMaskMAccFacts
            ? computedMaskMAccFacts->maskedPassthroughLayout
        : runtimeScalarMAccFacts
            ? runtimeScalarMAccFacts->maskedPassthroughLayout
            : llvm::StringRef(kRVVComputedMaskedMAccPassthroughLayout);
    const llvm::StringRef expectedSourceMemoryForm =
        computedMaskMAccFacts ? computedMaskMAccFacts->sourceMemoryForm
        : runtimeScalarMAccFacts ? runtimeScalarMAccFacts->sourceMemoryForm
                               : llvm::StringRef(kRVVMAccSourceMemoryForm);
    const llvm::StringRef expectedDestinationMemoryForm =
        computedMaskMAccFacts
            ? computedMaskMAccFacts->destinationMemoryForm
        : runtimeScalarMAccFacts
            ? runtimeScalarMAccFacts->destinationMemoryForm
            : llvm::StringRef(kRVVMAccDestinationMemoryForm);
    const llvm::StringRef expectedIndexedMemoryLayout =
        computedMaskMAccFacts
            ? computedMaskMAccFacts->indexedMemoryLayout
        : runtimeScalarMAccFacts
            ? runtimeScalarMAccFacts->indexedMemoryLayout
            : llvm::StringRef(kRVVComputedMaskedMAccMemoryLayout);
    if (description.accumulationComputeSuffix !=
            expectedAccumulationComputeSuffix ||
        description.accumulationMaskProducerSource != expectedMaskProducer ||
        description.accumulationAccumulatorContract !=
            expectedAccumulationAccumulatorContract ||
        description.accumulationResultContract !=
            expectedAccumulationResultContract ||
        description.maskRole != expectedMaskRole ||
        description.maskSource != expectedMaskSource ||
        description.maskMemoryForm != expectedMaskMemoryForm ||
        description.inactiveLaneContract != expectedInactiveLaneContract ||
        description.maskedPassthroughLayout != expectedMaskedPassthroughLayout ||
        description.sourceMemoryForm != expectedSourceMemoryForm ||
        description.destinationMemoryForm != expectedDestinationMemoryForm ||
        description.indexedMemoryLayout != expectedIndexedMemoryLayout ||
        description.compareIntrinsic.empty() ||
        description.maskedMergeIntrinsic.empty())
      return makeRVVTargetRouteError(
          "computed-mask MAcc target artifact consumer requires "
          "provider-derived exact mask, passthrough, compare predicate, "
          "source/destination memory, accumulator, and result facts before "
          "artifact export");
    if (isRVVRuntimeScalarComputedMaskMAccRouteFamilyOperation(
            description.operation) &&
        description.rhsBroadcastIntrinsic.empty())
      return makeRVVTargetRouteError(
          "runtime-scalar computed-mask MAcc target artifact consumer requires "
          "provider-derived RHS scalar splat facts before artifact export");
  }

  if (llvm::Error error = validateRVVMAccRouteHeaders(route, description))
    return error;
  if (llvm::Error error = validateRVVMAccRouteTypeMappings(route, description))
    return error;
  if (llvm::Error error = validateRVVMAccRouteABIMappings(route, description))
    return error;
  return validateRVVMAccRouteStatementPlan(route, description);
}

llvm::Error validateRVVMAccTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVMAccRoutePayloadFacts(context.route, context.description);
}

llvm::Error validateRVVMAccTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description =
      context.description;
  const std::string sew = llvm::Twine(description.sew).str();

  if (description.operation ==
      plugin::rvv::RVVSelectedBodyOperationKind::MAccAdd) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.plain_macc_route_family_plan",
            description.plainMAccRouteFamilyPlanID,
            "selected typed RVV plain MAcc route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.scalar_broadcast_macc_route_family_plan", "",
            "selected typed RVV scalar-broadcast MAcc route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_route_family_plan", "",
            "selected typed RVV accumulation route-family plan"))
      return error;
  } else if (isRVVScalarBroadcastMAccRouteFamilyOperation(
                 description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.scalar_broadcast_macc_route_family_plan",
            description.scalarBroadcastMAccRouteFamilyPlanID,
            "selected typed RVV scalar-broadcast MAcc route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.plain_macc_route_family_plan", "",
            "selected typed RVV plain MAcc route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_route_family_plan", "",
            "selected typed RVV accumulation route-family plan"))
      return error;
  } else if (isRVVComputedMaskMAccRouteFamilyOperation(description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.compare_predicate_kind",
            description.comparePredicateKind,
            "selected typed RVV computed-mask MAcc compare predicate"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_route_family_plan",
            description.accumulationRouteFamilyPlanID,
            "selected typed RVV computed-mask MAcc route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_compute_suffix",
            description.accumulationComputeSuffix,
            "selected typed RVV computed-mask MAcc compute suffix"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_mask_producer_source",
            description.accumulationMaskProducerSource,
            "selected typed RVV computed-mask MAcc mask producer"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_accumulator_contract",
            description.accumulationAccumulatorContract,
            "selected typed RVV computed-mask MAcc accumulator contract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_result_contract",
            description.accumulationResultContract,
            "selected typed RVV computed-mask MAcc result contract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_scalar_carry_contract",
            description.accumulationScalarCarryContract,
            "selected typed RVV computed-mask MAcc scalar carry contract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.plain_macc_route_family_plan", "",
            "selected typed RVV plain MAcc route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.scalar_broadcast_macc_route_family_plan", "",
            "selected typed RVV scalar-broadcast MAcc route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_role", description.maskRole,
            "selected typed RVV computed-mask MAcc mask role"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_source", description.maskSource,
            "selected typed RVV computed-mask MAcc mask source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_memory_form",
            description.maskMemoryForm,
            "selected typed RVV computed-mask MAcc mask memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.inactive_lane_contract",
            description.inactiveLaneContract,
            "selected typed RVV computed-mask MAcc inactive lane contract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.masked_passthrough_layout",
            description.maskedPassthroughLayout,
            "selected typed RVV computed-mask MAcc passthrough layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.source_memory_form",
            description.sourceMemoryForm,
            "selected typed RVV computed-mask MAcc source memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.destination_memory_form",
            description.destinationMemoryForm,
            "selected typed RVV computed-mask MAcc destination memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.indexed_memory_layout",
            description.indexedMemoryLayout,
            "selected typed RVV computed-mask MAcc memory layout"))
      return error;
  }

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "rvv_selected_body_typed_compute_op",
          description.typedComputeOpName,
          "selected typed RVV MAcc compute operation"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.config_contract",
          description.configContractID,
          "selected typed RVV MAcc config contract"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.element_type", description.elementTypeName,
          "selected typed RVV MAcc element type"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.sew", sew, "selected typed RVV MAcc SEW"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.lmul", description.lmul,
          "selected typed RVV MAcc LMUL"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.bounded_slice", description.boundedSlice,
          "selected typed RVV MAcc bounded slice"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.memory_form",
          plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
              description.memoryForm),
          "selected typed RVV MAcc memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_control_plan",
          description.runtimeControlPlanID,
          "selected typed RVV MAcc runtime AVL/VL control plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order", description.runtimeABIOrder,
          "selected typed RVV MAcc runtime ABI order"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.required_header_declarations",
          description.requiredHeaderDeclarations,
          "selected typed RVV MAcc route header requirements"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.c_type_mapping",
          description.cTypeMappingSummary,
          "selected typed RVV MAcc route type mapping summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.macc_accumulator_layout",
          description.maccAccumulatorLayout,
          "selected typed RVV MAcc accumulator layout"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.macc_result_layout",
          description.maccResultLayout,
          "selected typed RVV MAcc result layout"))
    return error;

  return llvm::Error::success();
}

bool isRVVStandaloneReductionAccumulationTargetArtifactRouteFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVStandaloneReductionAccumulationRouteFamilyOperation(
      description.operation);
}

constexpr llvm::StringLiteral kRVVStandaloneReductionTypedComputeOp(
    "tcrv_rvv.standalone_reduce");
constexpr llvm::StringLiteral kRVVStandaloneReductionRuntimeABIOrder(
    "lhs,acc,out,n");
constexpr llvm::StringLiteral kRVVComputedMaskStandaloneReductionRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,src,acc,out,n");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskStandaloneReductionRuntimeABIOrder(
        "cmp_lhs,rhs_scalar,src,acc,out,n");
constexpr llvm::StringLiteral kRVVStandaloneReductionI32AccumulatorLayout(
    "scalar-i32-seed-lane0-from-accumulator-input");
constexpr llvm::StringLiteral kRVVStandaloneReductionI64AccumulatorLayout(
    "scalar-i64-seed-lane0-from-accumulator-input");
constexpr llvm::StringLiteral kRVVStandaloneReductionResultLayout(
    "store-standalone-reduction-lane0-to-output-scalar");
constexpr llvm::StringLiteral kRVVStandaloneReductionStoreVL("1");

llvm::StringRef getRVVPlainStandaloneReductionExpectedBindingPlanID(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::StandaloneReduceAdd:
    return "rvv-route-operand-binding:standalone_reduce_add.v1";
  case plugin::rvv::RVVSelectedBodyOperationKind::StandaloneReduceMin:
    return "rvv-route-operand-binding:standalone_reduce_min.v1";
  case plugin::rvv::RVVSelectedBodyOperationKind::StandaloneReduceMax:
    return "rvv-route-operand-binding:standalone_reduce_max.v1";
  default:
    return {};
  }
}

llvm::StringRef getRVVStandaloneReductionExpectedAccumulatorLayoutForSEW(
    std::int64_t sew) {
  if (sew == 64)
    return kRVVStandaloneReductionI64AccumulatorLayout;
  if (sew == 32)
    return kRVVStandaloneReductionI32AccumulatorLayout;
  return {};
}

std::string getRVVPlainStandaloneReductionExpectedBindingSummary(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  llvm::StringRef planID =
      getRVVPlainStandaloneReductionExpectedBindingPlanID(operation);
  if (planID.empty())
    return {};
  return (llvm::Twine(planID) +
          ";lhs=lhs-input-buffer:lhs:abi|load|reduce-input|hdr;"
          "acc=accumulator-input-buffer:acc:abi|seed|acc-state|hdr;"
          "out=output-buffer:out:abi|acc-state|store|hdr;"
          "n=runtime-element-count:n:abi|setvl-avl|loop|hdr")
      .str();
}

llvm::StringRef getRVVComputedMaskStandaloneReductionExpectedBindingPlanID(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd:
    return "rvv-route-operand-binding:computed_mask_standalone_reduce_add.v1";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin:
    return "rvv-route-operand-binding:computed_mask_standalone_reduce_min.v1";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax:
    return "rvv-route-operand-binding:computed_mask_standalone_reduce_max.v1";
  default:
    return {};
  }
}

std::string getRVVComputedMaskStandaloneReductionExpectedBindingSummary(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  llvm::StringRef planID =
      getRVVComputedMaskStandaloneReductionExpectedBindingPlanID(operation);
  if (planID.empty())
    return {};
  const llvm::StringRef inactiveUse =
      operation ==
              plugin::rvv::RVVSelectedBodyOperationKind::
                  ComputedMaskStandaloneReduceAdd
          ? llvm::StringRef("zero-inactive")
          : llvm::StringRef("neutral-inactive");
  return (llvm::Twine(planID) +
          ";cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|cmp-lhs-call|hdr;"
          "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|cmp-rhs-call|hdr;"
          "src=source-input-buffer:src:abi|src-load|masked-reduce-input|" +
          inactiveUse +
          "|hdr;"
          "acc=accumulator-input-buffer:acc:abi|initial-seed|acc-state|masked-reduce-acc|hdr;"
          "out=output-buffer:out:abi|acc-state|store-base|hdr;"
          "n=runtime-element-count:n:abi|setvl-avl|loop|hdr")
      .str();
}

llvm::Error validateRVVPlainStandaloneReductionRuntimeABIFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.runtimeABIOrder != kRVVStandaloneReductionRuntimeABIOrder)
    return makeRVVTargetRouteError(
        llvm::Twine("plain standalone reduction target artifact consumer "
                    "requires provider-derived runtime ABI order '") +
        kRVVStandaloneReductionRuntimeABIOrder + "' but was '" +
        description.runtimeABIOrder + "'");
  if (description.runtimeABIParameters.size() != 4)
    return makeRVVTargetRouteError(
        "plain standalone reduction target artifact consumer requires "
        "provider-derived runtime ABI parameters for lhs source, acc scalar "
        "seed, out scalar result, and n before artifact export");

  struct ExpectedRuntimeABIParameter {
    llvm::StringRef cName;
    support::RuntimeABIParameterRole role;
  };
  const ExpectedRuntimeABIParameter expected[] = {
      {"lhs", support::RuntimeABIParameterRole::LHSInputBuffer},
      {"acc", support::RuntimeABIParameterRole::AccumulatorInputBuffer},
      {"out", support::RuntimeABIParameterRole::OutputBuffer},
      {"n", support::RuntimeABIParameterRole::RuntimeElementCount},
  };
  constexpr size_t expectedCount = sizeof(expected) / sizeof(expected[0]);
  for (size_t index = 0; index < expectedCount; ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    if (actual.cName != expected[index].cName ||
        actual.role != expected[index].role || actual.cType.empty())
      return makeRVVTargetRouteError(
          llvm::Twine("plain standalone reduction target artifact consumer "
                      "requires provider-derived runtime ABI parameter ") +
          std::to_string(index) + " to bind " + expected[index].cName +
          " as " +
          support::stringifyRuntimeABIParameterRole(expected[index].role) +
          " with a provider-derived C type before artifact export");
  }
  return llvm::Error::success();
}

llvm::Error validateRVVComputedMaskStandaloneReductionRuntimeABIFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.runtimeABIOrder !=
      kRVVComputedMaskStandaloneReductionRuntimeABIOrder)
    return makeRVVTargetRouteError(
        llvm::Twine("computed-mask standalone reduction target artifact "
                    "consumer requires provider-derived runtime ABI order '") +
        kRVVComputedMaskStandaloneReductionRuntimeABIOrder + "' but was '" +
        description.runtimeABIOrder + "'");
  if (description.runtimeABIParameters.size() != 6)
    return makeRVVTargetRouteError(
        "computed-mask standalone reduction target artifact consumer requires "
        "provider-derived runtime ABI parameters for cmp_lhs, cmp_rhs, src, "
        "acc scalar seed, out scalar result, and n before artifact export");

  struct ExpectedRuntimeABIParameter {
    llvm::StringRef cName;
    support::RuntimeABIParameterRole role;
  };
  const ExpectedRuntimeABIParameter expected[] = {
      {"cmp_lhs", support::RuntimeABIParameterRole::LHSInputBuffer},
      {"cmp_rhs", support::RuntimeABIParameterRole::RHSInputBuffer},
      {"src", support::RuntimeABIParameterRole::SourceInputBuffer},
      {"acc", support::RuntimeABIParameterRole::AccumulatorInputBuffer},
      {"out", support::RuntimeABIParameterRole::OutputBuffer},
      {"n", support::RuntimeABIParameterRole::RuntimeElementCount},
  };
  constexpr size_t expectedCount = sizeof(expected) / sizeof(expected[0]);
  for (size_t index = 0; index < expectedCount; ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    if (actual.cName != expected[index].cName ||
        actual.role != expected[index].role || actual.cType.empty())
      return makeRVVTargetRouteError(
          llvm::Twine("computed-mask standalone reduction target artifact "
                      "consumer requires provider-derived runtime ABI "
                      "parameter ") +
          std::to_string(index) + " to bind " + expected[index].cName +
          " as " +
          support::stringifyRuntimeABIParameterRole(expected[index].role) +
          " with a provider-derived C type before artifact export");
  }
  return llvm::Error::success();
}

llvm::Error
validateRVVRuntimeScalarComputedMaskStandaloneReductionRuntimeABIFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  std::optional<
      plugin::rvv::RVVRuntimeScalarComputedMaskStandaloneReductionRouteFacts>
      routeFacts =
          plugin::rvv::
              getRVVRuntimeScalarComputedMaskStandaloneReductionRouteFacts(
                  description.operation);
  if (!routeFacts)
    return makeRVVTargetRouteError(
        "runtime-scalar computed-mask standalone reduction target artifact "
        "consumer requires add/min/max canonical route facts before artifact "
        "export");
  if (description.runtimeABIOrder != routeFacts->runtimeABIOrder)
    return makeRVVTargetRouteError(
        llvm::Twine(
            "runtime-scalar computed-mask standalone reduction target artifact "
            "consumer requires provider-derived runtime ABI order '") +
        routeFacts->runtimeABIOrder + "' but was '" +
        description.runtimeABIOrder + "'");
  if (description.runtimeABIParameters.size() != 6)
    return makeRVVTargetRouteError(
        "runtime-scalar computed-mask standalone reduction target artifact "
        "consumer requires provider-derived runtime ABI parameters for "
        "cmp_lhs, rhs_scalar, src, acc scalar seed, out scalar result, and n "
        "before artifact export");

  struct ExpectedRuntimeABIParameter {
    llvm::StringRef cName;
    support::RuntimeABIParameterRole role;
  };
  const ExpectedRuntimeABIParameter expected[] = {
      {"cmp_lhs", support::RuntimeABIParameterRole::LHSInputBuffer},
      {"rhs_scalar", support::RuntimeABIParameterRole::RHSScalarValue},
      {"src", support::RuntimeABIParameterRole::SourceInputBuffer},
      {"acc", support::RuntimeABIParameterRole::AccumulatorInputBuffer},
      {"out", support::RuntimeABIParameterRole::OutputBuffer},
      {"n", support::RuntimeABIParameterRole::RuntimeElementCount},
  };
  constexpr size_t expectedCount = sizeof(expected) / sizeof(expected[0]);
  for (size_t index = 0; index < expectedCount; ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    if (actual.cName != expected[index].cName ||
        actual.role != expected[index].role || actual.cType.empty())
      return makeRVVTargetRouteError(
          llvm::Twine(
              "runtime-scalar computed-mask standalone reduction target "
              "artifact consumer requires provider-derived runtime ABI "
              "parameter ") +
          std::to_string(index) + " to bind " + expected[index].cName +
          " as " +
          support::stringifyRuntimeABIParameterRole(expected[index].role) +
          " with a provider-derived C type before artifact export");
  }
  return llvm::Error::success();
}

llvm::Error validateRVVPlainStandaloneReductionRoutePayloadFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (!isRVVPlainStandaloneReductionRouteFamilyOperation(
          description.operation))
    return llvm::Error::success();

  if (description.typedComputeOpName != kRVVStandaloneReductionTypedComputeOp ||
      description.memoryForm != plugin::rvv::RVVSelectedBodyMemoryForm::
                                    UnitStrideStandaloneReduction)
    return makeRVVTargetRouteError(
        "plain standalone reduction target artifact consumer requires a "
        "selected tcrv_rvv.standalone_reduce body with unit-stride standalone "
        "reduction memory form");
  constexpr llvm::StringLiteral consumerLabel(
      "plain standalone reduction target artifact consumer");
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "element type", description.elementTypeName))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedCount(
          consumerLabel, "SEW", description.sew))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "LMUL", description.lmul))
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
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "provider-supported mirror",
          description.providerSupportedMirror))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "target leaf profile", description.targetLeafProfile))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "standalone reduction route-family plan",
          description.standaloneReductionRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "scalar-result runtime boundary",
          description.standaloneReductionScalarResultRuntimeBoundary))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "required header declarations",
          description.requiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "C type mapping summary",
          description.cTypeMappingSummary))
    return error;

  llvm::StringRef expectedBindingPlan =
      getRVVPlainStandaloneReductionExpectedBindingPlanID(
          description.operation);
  std::string expectedBindingSummary =
      getRVVPlainStandaloneReductionExpectedBindingSummary(
          description.operation);
  if (description.routeOperandBindingPlanID != expectedBindingPlan ||
      description.routeOperandBindingSummary != expectedBindingSummary)
    return makeRVVTargetRouteError(
        llvm::Twine("plain standalone reduction target artifact consumer "
                    "requires provider-derived route operand binding plan '") +
        expectedBindingPlan +
        "' with lhs source, acc seed, out scalar result, and n bindings but "
        "provider carried plan '" +
        description.routeOperandBindingPlanID + "'");

  if (llvm::Error error =
          validateRVVPlainStandaloneReductionRuntimeABIFacts(description))
    return error;

  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "reduction accumulator layout",
          description.reductionAccumulatorLayout))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "reduction result layout",
          description.reductionResultLayout))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "reduction store VL", description.reductionStoreVL))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "vector load intrinsic",
          description.vectorLoadIntrinsic))
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

  return llvm::Error::success();
}

llvm::Error validateRVVComputedMaskStandaloneReductionRoutePayloadFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (!isRVVVectorComputedMaskStandaloneReductionRouteFamilyOperation(
          description.operation))
    return llvm::Error::success();

  if (description.typedComputeOpName != "tcrv_rvv.masked_standalone_reduce" ||
      description.memoryForm !=
          plugin::rvv::RVVSelectedBodyMemoryForm::
              ComputedMaskUnitStrideStandaloneReduction)
    return makeRVVTargetRouteError(
        "computed-mask standalone reduction target artifact consumer requires "
        "a selected tcrv_rvv.masked_standalone_reduce body with "
        "computed-mask unit-stride standalone reduction memory form");
  constexpr llvm::StringLiteral consumerLabel(
      "computed-mask standalone reduction target artifact consumer");
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "element type", description.elementTypeName))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedCount(
          consumerLabel, "SEW", description.sew))
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
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "mask type", description.maskTypeName))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "mask C type", description.maskCType))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "provider-supported mirror",
          description.providerSupportedMirror))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "target leaf profile", description.targetLeafProfile))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "standalone reduction route-family plan",
          description.standaloneReductionRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "scalar-result runtime boundary",
          description.standaloneReductionScalarResultRuntimeBoundary))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "required header declarations",
          description.requiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "C type mapping summary",
          description.cTypeMappingSummary))
    return error;

  llvm::StringRef expectedBindingPlan =
      getRVVComputedMaskStandaloneReductionExpectedBindingPlanID(
          description.operation);
  std::string expectedBindingSummary =
      getRVVComputedMaskStandaloneReductionExpectedBindingSummary(
          description.operation);
  if (description.routeOperandBindingPlanID != expectedBindingPlan ||
      description.routeOperandBindingSummary != expectedBindingSummary)
    return makeRVVTargetRouteError(
        llvm::Twine("computed-mask standalone reduction target artifact "
                    "consumer requires provider-derived route operand binding "
                    "plan '") +
        expectedBindingPlan +
        "' with cmp_lhs, cmp_rhs, src, acc seed, out scalar result, and n "
        "bindings but provider carried plan '" +
        description.routeOperandBindingPlanID + "'");

  if (llvm::Error error =
          validateRVVComputedMaskStandaloneReductionRuntimeABIFacts(
              description))
    return error;

  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "reduction accumulator layout",
          description.reductionAccumulatorLayout))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "reduction result layout",
          description.reductionResultLayout))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "reduction store VL", description.reductionStoreVL))
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
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "compare intrinsic", description.compareIntrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "masked merge intrinsic",
          description.maskedMergeIntrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "mask role", description.maskRole))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "mask source", description.maskSource))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "mask memory form", description.maskMemoryForm))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "inactive-lane requirement",
          description.inactiveLaneZeroingRequirement))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "compare predicate", description.comparePredicateKind))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "computed-mask accumulation route-family plan",
          description.accumulationRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "computed-mask accumulation compute suffix",
          description.accumulationComputeSuffix))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "computed-mask accumulation producer source",
          description.accumulationMaskProducerSource))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "computed-mask accumulation accumulator contract",
          description.accumulationAccumulatorContract))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "computed-mask accumulation result contract",
          description.accumulationResultContract))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "computed-mask accumulation scalar-carry contract",
          description.accumulationScalarCarryContract))
    return error;

  return llvm::Error::success();
}

llvm::Error
validateRVVRuntimeScalarComputedMaskStandaloneReductionRoutePayloadFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.operation != plugin::rvv::RVVSelectedBodyOperationKind::
                                   RuntimeScalarComputedMaskStandaloneReduceAdd &&
      description.operation != plugin::rvv::RVVSelectedBodyOperationKind::
                                   RuntimeScalarComputedMaskStandaloneReduceMin &&
      description.operation != plugin::rvv::RVVSelectedBodyOperationKind::
                                   RuntimeScalarComputedMaskStandaloneReduceMax)
    return llvm::Error::success();

  if (description.typedComputeOpName != "tcrv_rvv.masked_standalone_reduce" ||
      description.memoryForm !=
          plugin::rvv::RVVSelectedBodyMemoryForm::
              RuntimeScalarComputedMaskUnitStrideStandaloneReduction)
    return makeRVVTargetRouteError(
        "runtime-scalar computed-mask standalone reduction target artifact "
        "consumer requires a selected tcrv_rvv.masked_standalone_reduce body "
        "with runtime-scalar computed-mask unit-stride standalone reduction "
        "memory form");
  constexpr llvm::StringLiteral consumerLabel(
      "runtime-scalar computed-mask standalone reduction target artifact "
      "consumer");
  std::optional<
      plugin::rvv::RVVRuntimeScalarComputedMaskStandaloneReductionRouteFacts>
      routeFacts =
          plugin::rvv::
              getRVVRuntimeScalarComputedMaskStandaloneReductionRouteFacts(
                  description.operation);
  if (!routeFacts)
    return makeRVVTargetRouteError(
        "runtime-scalar computed-mask standalone reduction target artifact "
        "consumer requires add/min/max canonical route facts before artifact "
        "export");
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "element type", description.elementTypeName))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedCount(
          consumerLabel, "SEW", description.sew))
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
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "mask type", description.maskTypeName))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "mask C type", description.maskCType))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "provider-supported mirror",
          description.providerSupportedMirror))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "target leaf profile", description.targetLeafProfile))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "standalone reduction route-family plan",
          description.standaloneReductionRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "scalar-result runtime boundary",
          description.standaloneReductionScalarResultRuntimeBoundary))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "required header declarations",
          description.requiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "C type mapping summary",
          description.cTypeMappingSummary))
    return error;
  if (description.providerSupportedMirror !=
          routeFacts->providerSupportedMirror ||
      description.targetLeafProfile != routeFacts->targetLeafProfile ||
      description.requiredHeaderDeclarations !=
          routeFacts->requiredHeaderDeclarations ||
      description.cTypeMappingSummary != routeFacts->cTypeMappingSummary)
    return makeRVVTargetRouteError(
        llvm::Twine("runtime-scalar computed-mask standalone reduction target "
                    "artifact consumer requires canonical route facts for ") +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            description.operation) +
        " before artifact export");
  if (llvm::Error error = requireRVVProviderCanonicalField(
          consumerLabel, "scalar-result runtime boundary",
          description.standaloneReductionScalarResultRuntimeBoundary,
          routeFacts->scalarResultRuntimeBoundary))
    return error;

  if (description.routeOperandBindingPlanID !=
          routeFacts->routeOperandBindingPlanID ||
      description.routeOperandBindingSummary !=
          routeFacts->routeOperandBindingSummary)
    return makeRVVTargetRouteError(
        llvm::Twine("runtime-scalar computed-mask standalone reduction target "
                    "artifact consumer requires provider-derived route operand "
                    "binding plan '") +
        routeFacts->routeOperandBindingPlanID +
        "' with cmp_lhs, rhs_scalar, src, acc seed, out scalar result, and n "
        "bindings but provider carried plan '" +
        description.routeOperandBindingPlanID + "'");

  if (llvm::Error error =
          validateRVVRuntimeScalarComputedMaskStandaloneReductionRuntimeABIFacts(
              description))
    return error;

  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "reduction accumulator layout",
          description.reductionAccumulatorLayout))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "reduction result layout",
          description.reductionResultLayout))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "reduction store VL", description.reductionStoreVL))
    return error;
  llvm::StringRef expectedAccumulatorLayout =
      getRVVStandaloneReductionExpectedAccumulatorLayoutForSEW(description.sew);
  if (expectedAccumulatorLayout.empty())
    return makeRVVTargetRouteError(
        llvm::Twine("runtime-scalar computed-mask standalone reduction target "
                    "artifact consumer requires canonical standalone "
                    "reduction accumulator layout for SEW ") +
        llvm::Twine(description.sew) + " before artifact export");
  if (llvm::Error error = requireRVVProviderCanonicalField(
          consumerLabel, "reduction accumulator layout",
          description.reductionAccumulatorLayout, expectedAccumulatorLayout))
    return error;
  if (llvm::Error error = requireRVVProviderCanonicalField(
          consumerLabel, "reduction result layout",
          description.reductionResultLayout, routeFacts->reductionResultLayout))
    return error;
  if (llvm::Error error = requireRVVProviderCanonicalField(
          consumerLabel, "reduction store VL", description.reductionStoreVL,
          routeFacts->reductionStoreVL))
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
          consumerLabel, "RHS scalar splat intrinsic",
          description.rhsBroadcastIntrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "scalar result store intrinsic",
          description.storeIntrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "reduction intrinsic", description.intrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "compare intrinsic", description.compareIntrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "masked merge intrinsic",
          description.maskedMergeIntrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "mask role", description.maskRole))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "mask source", description.maskSource))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "mask memory form", description.maskMemoryForm))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "inactive-lane requirement",
          description.inactiveLaneZeroingRequirement))
    return error;
  if (description.inactiveLaneZeroingRequirement !=
      routeFacts->inactiveLaneRequirement)
    return makeRVVTargetRouteError(
        llvm::Twine("runtime-scalar computed-mask standalone reduction target "
                    "artifact consumer requires canonical inactive-lane "
                    "contract '") +
        routeFacts->inactiveLaneRequirement + "' but provider carried '" +
        description.inactiveLaneZeroingRequirement + "'");
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "compare predicate", description.comparePredicateKind))
    return error;
  if (llvm::Error error = requireRVVProviderCanonicalField(
          consumerLabel, "compare predicate", description.comparePredicateKind,
          routeFacts->comparePredicateKind))
    return error;
  if (llvm::Error error = requireRVVProviderCanonicalField(
          consumerLabel, "mask role", description.maskRole,
          routeFacts->maskRole))
    return error;
  if (llvm::Error error = requireRVVProviderCanonicalField(
          consumerLabel, "mask source", description.maskSource,
          routeFacts->maskSource))
    return error;
  if (llvm::Error error = requireRVVProviderCanonicalField(
          consumerLabel, "mask memory form", description.maskMemoryForm,
          routeFacts->maskMemoryForm))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "computed-mask accumulation route-family plan",
          description.accumulationRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "computed-mask accumulation compute suffix",
          description.accumulationComputeSuffix))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "runtime-scalar accumulation producer source",
          description.accumulationMaskProducerSource))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "computed-mask accumulation accumulator contract",
          description.accumulationAccumulatorContract))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "computed-mask accumulation result contract",
          description.accumulationResultContract))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          consumerLabel, "computed-mask accumulation scalar-carry contract",
          description.accumulationScalarCarryContract))
    return error;
  if (llvm::Error error = requireRVVProviderCanonicalField(
          consumerLabel, "computed-mask accumulation route-family plan",
          description.accumulationRouteFamilyPlanID,
          routeFacts->accumulationRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVProviderCanonicalField(
          consumerLabel, "computed-mask accumulation compute suffix",
          description.accumulationComputeSuffix,
          routeFacts->accumulationComputeSuffix))
    return error;
  if (llvm::Error error = requireRVVProviderCanonicalField(
          consumerLabel, "runtime-scalar accumulation producer source",
          description.accumulationMaskProducerSource,
          routeFacts->accumulationMaskProducerSource))
    return error;
  if (llvm::Error error = requireRVVProviderCanonicalField(
          consumerLabel, "computed-mask accumulation accumulator contract",
          description.accumulationAccumulatorContract,
          routeFacts->accumulationAccumulatorContract))
    return error;
  if (llvm::Error error = requireRVVProviderCanonicalField(
          consumerLabel, "computed-mask accumulation result contract",
          description.accumulationResultContract,
          routeFacts->accumulationResultContract))
    return error;
  if (llvm::Error error = requireRVVProviderCanonicalField(
          consumerLabel, "computed-mask accumulation scalar-carry contract",
          description.accumulationScalarCarryContract,
          routeFacts->accumulationScalarCarryContract))
    return error;

  return llvm::Error::success();
}

llvm::Error validateRVVStandaloneReductionAccumulationRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-derived required_header_declarations before accepting the "
        "route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "at least one provider route header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "standalone reduction/accumulation target artifact consumer saw an "
          "empty provider route header declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("standalone reduction/accumulation target artifact "
                      "consumer requires rebuilt provider route header '") +
          trimmed + "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVStandaloneReductionAccumulationRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.vlCType.empty() || description.vectorTypeName.empty() ||
      description.vectorCType.empty() || description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-derived VL, vector, and C type mapping facts before "
        "artifact export");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("standalone reduction/accumulation target artifact "
                    "consumer requires rebuilt provider route type mapping "
                    "'!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("standalone reduction/accumulation target artifact "
                    "consumer requires rebuilt provider route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");

  if (description.standaloneReductionSourceVectorTypeName.empty() ||
      description.standaloneReductionSourceVectorCType.empty() ||
      description.standaloneReductionScalarResultVectorTypeName.empty() ||
      description.standaloneReductionScalarResultVectorCType.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-derived source and scalar-result vector type mapping facts "
        "before artifact export");
  if (!routeHasTypeMapping(
          route, description.standaloneReductionSourceVectorTypeName,
          description.standaloneReductionSourceVectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("standalone reduction/accumulation target artifact "
                    "consumer requires rebuilt provider route source type "
                    "mapping '") +
        description.standaloneReductionSourceVectorTypeName + "' -> '" +
        description.standaloneReductionSourceVectorCType + "'");
  if (!routeHasTypeMapping(
          route, description.standaloneReductionScalarResultVectorTypeName,
          description.standaloneReductionScalarResultVectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("standalone reduction/accumulation target artifact "
                    "consumer requires rebuilt provider route scalar-result "
                    "type mapping '") +
        description.standaloneReductionScalarResultVectorTypeName + "' -> '" +
        description.standaloneReductionScalarResultVectorCType + "'");

  if (isRVVComputedMaskStandaloneReductionRouteFamilyOperation(
          description.operation)) {
    if (description.maskTypeName.empty() || description.maskCType.empty())
      return makeRVVTargetRouteError(
          "computed-mask standalone reduction/accumulation target artifact "
          "consumer requires provider-derived mask type mapping facts before "
          "artifact export");
    if (!routeHasTypeMapping(route, description.maskTypeName,
                             description.maskCType))
      return makeRVVTargetRouteError(
          llvm::Twine("computed-mask standalone reduction/accumulation target "
                      "artifact consumer requires rebuilt provider route type "
                      "mapping '") +
          description.maskTypeName + "' -> '" + description.maskCType + "'");
  } else if (!description.maskTypeName.empty() ||
             !description.maskCType.empty()) {
    return makeRVVTargetRouteError(
        "plain standalone reduction/accumulation target artifact consumer "
        "rejects stale mask type mapping facts");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVStandaloneReductionAccumulationRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.runtimeABIOrder.empty() ||
      description.runtimeABIParameters.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-derived runtime ABI order and ABI parameters before "
        "artifact export");
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("standalone reduction/accumulation target artifact "
                    "consumer requires rebuilt provider route ABI mapping "
                    "count ") +
        llvm::Twine(description.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        description.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine("standalone reduction/accumulation target artifact "
                      "consumer requires rebuilt provider route ABI "
                      "mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("standalone reduction/accumulation target artifact "
                      "consumer requires rebuilt provider route ABI "
                      "mapping[") +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVStandaloneReductionPreLoopStatements(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const support::RuntimeABIParameter &accumulator,
    const support::RuntimeABIParameter &out,
    const support::RuntimeABIParameter &runtimeN) {
  constexpr llvm::StringLiteral consumerLabel(
      "standalone reduction/accumulation target artifact consumer");
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
          description.setVLIntrinsic,
          {{runtimeN.cName, runtimeN.cType}},
          description.emitCFullChunkVLName, description.vlCType))
    return error;

  const conversion::emitc::TCRVEmitCCallOpaqueStep &initialSplat =
      route.getCallOpaqueSteps()[1];
  const std::string expectedAccumulatorLane0 =
      (llvm::StringRef(accumulator.cName) + "[0]").str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          initialSplat, consumerLabel, "pre-loop scalar seed splat",
          description.scalarSeedSplatIntrinsic,
          {{expectedAccumulatorLane0, scalarCType},
           {description.reductionStoreVL, description.vlCType}},
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
           {description.reductionStoreVL, description.vlCType}}))
    return error;

  return llvm::Error::success();
}

llvm::Error validateRVVPlainStandaloneReductionLoopStatements(
    const conversion::emitc::TCRVEmitCForLoop &loop,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const support::RuntimeABIParameter &lhs,
    const support::RuntimeABIParameter &out,
    const support::RuntimeABIParameter &runtimeN) {
  constexpr llvm::StringLiteral consumerLabel(
      "plain standalone reduction target artifact consumer");
  if (loop.bodySteps.size() != 5)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-built loop setvl, source load, scalar seed splat, "
        "reduction, and scalar-result store statements before artifact export");

  const llvm::StringRef scalarCType =
      description.standaloneReductionScalarCType;
  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeN.cName) + " - " +
       description.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[0], consumerLabel, "loop setvl",
          description.setVLIntrinsic,
          {{expectedRemainingAVL, description.vlCType}},
          description.emitCLoopVLName, description.vlCType))
    return error;

  const std::string expectedLhsPointer =
      (llvm::StringRef(lhs.cName) + " + " +
       description.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[1], consumerLabel, "source vector load",
          description.vectorLoadIntrinsic,
          {{expectedLhsPointer, lhs.cType},
           {description.emitCLoopVLName, description.vlCType}},
          "lhs_vec", description.standaloneReductionSourceVectorCType))
    return error;

  const std::string expectedOutLane0 =
      (llvm::StringRef(out.cName) + "[0]").str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[2], consumerLabel, "loop scalar seed splat",
          description.scalarSeedSplatIntrinsic,
          {{expectedOutLane0, scalarCType},
           {description.reductionStoreVL, description.vlCType}},
          "standalone_acc_vec",
          description.standaloneReductionScalarResultVectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[3], consumerLabel, "signed min/max/add reduction",
          description.intrinsic,
          {{"lhs_vec", description.standaloneReductionSourceVectorCType},
           {"standalone_acc_vec",
            description.standaloneReductionScalarResultVectorCType},
           {description.emitCLoopVLName, description.vlCType}},
          description.resultName,
          description.standaloneReductionScalarResultVectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[4], consumerLabel, "scalar-result store",
          description.storeIntrinsic,
          {{out.cName, out.cType},
           {description.resultName,
            description.standaloneReductionScalarResultVectorCType},
           {description.reductionStoreVL, description.vlCType}}))
    return error;

  return llvm::Error::success();
}

llvm::Error validateRVVComputedMaskStandaloneReductionLoopStatements(
    const conversion::emitc::TCRVEmitCForLoop &loop,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const support::RuntimeABIParameter &cmpLHS,
    const support::RuntimeABIParameter &cmpRHS,
    const support::RuntimeABIParameter &source,
    const support::RuntimeABIParameter &out,
    const support::RuntimeABIParameter &runtimeN) {
  constexpr llvm::StringLiteral consumerLabel(
      "computed-mask standalone reduction target artifact consumer");
  if (loop.bodySteps.size() != 10)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-built loop setvl, compare/source loads, compare, "
        "inactive neutral splat, merge, scalar seed, reduction, and "
        "scalar-result store statements before artifact export");

  const llvm::StringRef scalarCType =
      description.standaloneReductionScalarCType;
  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeN.cName) + " - " +
       description.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[0], consumerLabel, "loop setvl",
          description.setVLIntrinsic,
          {{expectedRemainingAVL, description.vlCType}},
          description.emitCLoopVLName, description.vlCType))
    return error;

  const std::string expectedCmpLHSPointer =
      (llvm::StringRef(cmpLHS.cName) + " + " +
       description.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[1], consumerLabel, "compare lhs vector load",
          description.vectorLoadIntrinsic,
          {{expectedCmpLHSPointer, cmpLHS.cType},
           {description.emitCLoopVLName, description.vlCType}},
          "lhs_vec", description.standaloneReductionSourceVectorCType))
    return error;

  const std::string expectedCmpRHSPointer =
      (llvm::StringRef(cmpRHS.cName) + " + " +
       description.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[2], consumerLabel, "compare rhs vector load",
          description.vectorLoadIntrinsic,
          {{expectedCmpRHSPointer, cmpRHS.cType},
           {description.emitCLoopVLName, description.vlCType}},
          "rhs_vec", description.standaloneReductionSourceVectorCType))
    return error;

  const std::string expectedSourcePointer =
      (llvm::StringRef(source.cName) + " + " +
       description.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[3], consumerLabel, "payload source vector load",
          description.vectorLoadIntrinsic,
          {{expectedSourcePointer, source.cType},
           {description.emitCLoopVLName, description.vlCType}},
          "source_vec", description.standaloneReductionSourceVectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[4], consumerLabel, "compare predicate",
          description.compareIntrinsic,
          {{"lhs_vec", description.standaloneReductionSourceVectorCType},
           {"rhs_vec", description.standaloneReductionSourceVectorCType},
           {description.emitCLoopVLName, description.vlCType}},
          description.maskName, description.maskCType))
    return error;

  const llvm::StringRef inactiveNeutral =
      plugin::rvv::getRVVSelectedBodyStandaloneReductionInactiveNeutralLiteral(
          description.operation, description.sew);
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
           {description.emitCLoopVLName, description.vlCType}},
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
           {description.emitCLoopVLName, description.vlCType}},
          "standalone_masked_source_vec",
          description.standaloneReductionSourceVectorCType))
    return error;

  const std::string expectedOutLane0 =
      (llvm::StringRef(out.cName) + "[0]").str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[7], consumerLabel, "loop scalar seed splat",
          description.scalarSeedSplatIntrinsic,
          {{expectedOutLane0, scalarCType},
           {description.reductionStoreVL, description.vlCType}},
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
           {description.emitCLoopVLName, description.vlCType}},
          description.resultName,
          description.standaloneReductionScalarResultVectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[9], consumerLabel, "scalar-result store",
          description.storeIntrinsic,
          {{out.cName, out.cType},
           {description.resultName,
            description.standaloneReductionScalarResultVectorCType},
           {description.reductionStoreVL, description.vlCType}}))
    return error;

  return llvm::Error::success();
}

llvm::Error
validateRVVRuntimeScalarComputedMaskStandaloneReductionLoopStatements(
    const conversion::emitc::TCRVEmitCForLoop &loop,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    const support::RuntimeABIParameter &cmpLHS,
    const support::RuntimeABIParameter &rhsScalar,
    const support::RuntimeABIParameter &source,
    const support::RuntimeABIParameter &out,
    const support::RuntimeABIParameter &runtimeN) {
  constexpr llvm::StringLiteral consumerLabel(
      "runtime-scalar computed-mask standalone reduction target artifact "
      "consumer");
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
      (llvm::StringRef(runtimeN.cName) + " - " +
       description.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[0], consumerLabel, "loop setvl",
          description.setVLIntrinsic,
          {{expectedRemainingAVL, description.vlCType}},
          description.emitCLoopVLName, description.vlCType))
    return error;

  const std::string expectedCmpLHSPointer =
      (llvm::StringRef(cmpLHS.cName) + " + " +
       description.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[1], consumerLabel, "compare lhs vector load",
          description.vectorLoadIntrinsic,
          {{expectedCmpLHSPointer, cmpLHS.cType},
           {description.emitCLoopVLName, description.vlCType}},
          "lhs_vec", description.standaloneReductionSourceVectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[2], consumerLabel, "RHS scalar splat",
          description.rhsBroadcastIntrinsic,
          {{rhsScalar.cName, rhsScalar.cType},
           {description.emitCLoopVLName, description.vlCType}},
          "rhs_vec", description.standaloneReductionSourceVectorCType))
    return error;

  const std::string expectedSourcePointer =
      (llvm::StringRef(source.cName) + " + " +
       description.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[3], consumerLabel, "payload source vector load",
          description.vectorLoadIntrinsic,
          {{expectedSourcePointer, source.cType},
           {description.emitCLoopVLName, description.vlCType}},
          "source_vec", description.standaloneReductionSourceVectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[4], consumerLabel, "compare predicate",
          description.compareIntrinsic,
          {{"lhs_vec", description.standaloneReductionSourceVectorCType},
           {"rhs_vec", description.standaloneReductionSourceVectorCType},
           {description.emitCLoopVLName, description.vlCType}},
          description.maskName, description.maskCType))
    return error;

  std::optional<
      plugin::rvv::RVVRuntimeScalarComputedMaskStandaloneReductionRouteFacts>
      routeFacts =
          plugin::rvv::
              getRVVRuntimeScalarComputedMaskStandaloneReductionRouteFacts(
                  description.operation);
  if (!routeFacts)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires runtime-scalar computed-mask standalone reduction "
        "canonical route facts before artifact export");
  llvm::StringRef inactiveNeutral;
  if (description.sew == 64)
    inactiveNeutral = routeFacts->inactiveNeutralLiteralSEW64;
  else if (description.sew == 32)
    inactiveNeutral = routeFacts->inactiveNeutralLiteralSEW32;
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
           {description.emitCLoopVLName, description.vlCType}},
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
           {description.emitCLoopVLName, description.vlCType}},
          "standalone_masked_source_vec",
          description.standaloneReductionSourceVectorCType))
    return error;

  const std::string expectedOutLane0 =
      (llvm::StringRef(out.cName) + "[0]").str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[7], consumerLabel, "loop scalar seed splat",
          description.scalarSeedSplatIntrinsic,
          {{expectedOutLane0, scalarCType},
           {description.reductionStoreVL, description.vlCType}},
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
           {description.emitCLoopVLName, description.vlCType}},
          description.resultName,
          description.standaloneReductionScalarResultVectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[9], consumerLabel, "scalar-result store",
          description.storeIntrinsic,
          {{out.cName, out.cType},
           {description.resultName,
            description.standaloneReductionScalarResultVectorCType},
           {description.reductionStoreVL, description.vlCType}}))
    return error;

  return llvm::Error::success();
}

llvm::Error validateRVVStandaloneReductionAccumulationRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
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
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
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
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-built loop bounds and step to mirror runtime AVL/VL route "
        "facts");
  if (description.runtimeABIParameters.empty() ||
      loop.upperBound.expression !=
          description.runtimeABIParameters.back().cName ||
      loop.upperBound.cType != description.runtimeABIParameters.back().cType)
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-built loop upper bound to use the runtime n/AVL ABI "
        "parameter");
  if (loop.bodySteps.empty() ||
      loop.bodySteps.front().callee != description.setVLIntrinsic ||
      !stepHasResult(loop.bodySteps.front(), description.emitCLoopVLName,
                     description.vlCType))
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
  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("standalone reduction/accumulation target artifact "
                    "consumer requires rebuilt provider route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");

  if (description.providerSupportedMirror.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "a provider-supported mirror label after route construction");
  if (description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider route operand binding facts before artifact export");
  if (description.standaloneReductionRouteFamilyPlanID.empty() ||
      description.standaloneReductionScalarResultRuntimeBoundary.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-derived standalone reduction family and scalar-result "
        "boundary mirrors before artifact export");
  if (description.reductionAccumulatorLayout.empty() ||
      description.reductionResultLayout.empty() ||
      description.reductionStoreVL.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-derived reduction accumulator, result, and store-VL facts "
        "before artifact export");
  if (description.sourceSplatIntrinsic.empty() ||
      description.scalarSeedSplatIntrinsic.empty() ||
      description.intrinsic.empty() || description.storeIntrinsic.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-derived source splat, scalar seed, reduction intrinsic, and "
        "store facts before artifact export");
  if (llvm::Error error =
          validateRVVPlainStandaloneReductionRoutePayloadFacts(description))
    return error;
  if (llvm::Error error =
          validateRVVComputedMaskStandaloneReductionRoutePayloadFacts(
              description))
    return error;
  if (llvm::Error error =
          validateRVVRuntimeScalarComputedMaskStandaloneReductionRoutePayloadFacts(
              description))
    return error;

  if (isRVVComputedMaskStandaloneReductionRouteFamilyOperation(
          description.operation)) {
    if (description.maskRole.empty() || description.maskSource.empty() ||
        description.maskMemoryForm.empty() ||
        description.inactiveLaneZeroingRequirement.empty() ||
        description.comparePredicateKind.empty() ||
        description.compareIntrinsic.empty() ||
        description.maskedMergeIntrinsic.empty())
      return makeRVVTargetRouteError(
          "computed-mask standalone reduction/accumulation target artifact "
          "consumer requires provider-derived mask, compare, and "
          "inactive-lane facts before artifact export");
  } else if (!description.maskRole.empty() || !description.maskSource.empty() ||
             !description.maskMemoryForm.empty() ||
             !description.inactiveLaneZeroingRequirement.empty() ||
             !description.comparePredicateKind.empty() ||
             !description.compareIntrinsic.empty() ||
             !description.maskedMergeIntrinsic.empty()) {
    return makeRVVTargetRouteError(
        "plain standalone reduction/accumulation target artifact consumer "
        "rejects stale computed-mask facts");
  }

  if (isRVVComputedMaskStandaloneReductionRouteFamilyOperation(
          description.operation)) {
    if (description.accumulationRouteFamilyPlanID.empty() ||
        description.accumulationComputeSuffix.empty() ||
        description.accumulationMaskProducerSource.empty() ||
        description.accumulationAccumulatorContract.empty() ||
        description.accumulationResultContract.empty() ||
        description.accumulationScalarCarryContract.empty())
      return makeRVVTargetRouteError(
          "computed-mask standalone accumulation target artifact consumer "
          "requires provider-derived accumulation and scalar-carry facts "
          "before artifact export");
    if (isRVVRuntimeScalarComputedMaskStandaloneReductionRouteFamilyOperation(
            description.operation) &&
        description.rhsBroadcastIntrinsic.empty())
      return makeRVVTargetRouteError(
          "runtime-scalar computed-mask standalone reduction/accumulation "
          "target artifact consumer requires a provider-derived RHS scalar "
          "splat intrinsic before artifact export");
  } else if (!description.accumulationRouteFamilyPlanID.empty() ||
             !description.accumulationComputeSuffix.empty() ||
             !description.accumulationMaskProducerSource.empty() ||
             !description.accumulationAccumulatorContract.empty() ||
             !description.accumulationResultContract.empty() ||
             !description.accumulationScalarCarryContract.empty() ||
             !description.rhsBroadcastIntrinsic.empty()) {
    return makeRVVTargetRouteError(
        "plain standalone reduction/accumulation target artifact consumer "
        "rejects stale accumulation, scalar-carry, or runtime-scalar facts");
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
        "standalone reduction/accumulation target artifact consumer rejects "
        "stale non-standalone route-family facts");

  if (llvm::Error error =
          validateRVVStandaloneReductionAccumulationRouteHeaders(route,
                                                                description))
    return error;
  if (llvm::Error error =
          validateRVVStandaloneReductionAccumulationRouteTypeMappings(
              route, description))
    return error;
  if (llvm::Error error =
          validateRVVStandaloneReductionAccumulationRouteABIMappings(
              route, description))
    return error;
  return validateRVVStandaloneReductionAccumulationRouteStatementPlan(
      route, description);
}

llvm::Error
validateRVVStandaloneReductionAccumulationTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVStandaloneReductionAccumulationRoutePayloadFacts(
      context.route, context.description);
}

llvm::Error requireEmptyStandaloneReductionAccumulationStaleMirror(
    const TargetArtifactCandidate &candidate, llvm::StringRef key,
    llvm::StringRef label) {
  return requireCandidateMetadataMirror(candidate, key, "", label);
}

llvm::Error
validateRVVStandaloneReductionAccumulationTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description =
      context.description;

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "rvv_selected_body_operation",
          plugin::rvv::stringifyRVVSelectedBodyOperationKind(
              description.operation),
          "selected typed RVV standalone reduction operation kind"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "rvv_selected_body_typed_compute_op",
          description.typedComputeOpName,
          "selected typed RVV standalone reduction typed compute op"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_plan",
          description.routeOperandBindingPlanID,
          "selected typed RVV standalone reduction binding plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_operands",
          description.routeOperandBindingSummary,
          "selected typed RVV standalone reduction binding summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.provider_supported_mirror",
          description.providerSupportedMirror,
          "selected typed RVV standalone reduction provider support"))
    return error;

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.standalone_reduction_route_family_plan",
          description.standaloneReductionRouteFamilyPlanID,
          "selected typed RVV standalone reduction route-family plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.standalone_reduction_source_vector_type",
          description.standaloneReductionSourceVectorTypeName,
          "selected typed RVV standalone reduction source vector type"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.standalone_reduction_source_vector_c_type",
          description.standaloneReductionSourceVectorCType,
          "selected typed RVV standalone reduction source vector C type"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.standalone_reduction_scalar_c_type",
          description.standaloneReductionScalarCType,
          "selected typed RVV standalone reduction scalar C type"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.standalone_reduction_scalar_result_vector_type",
          description.standaloneReductionScalarResultVectorTypeName,
          "selected typed RVV standalone reduction scalar-result vector type"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate,
          "tcrv_rvv.standalone_reduction_scalar_result_vector_c_type",
          description.standaloneReductionScalarResultVectorCType,
          "selected typed RVV standalone reduction scalar-result vector C type"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate,
          "tcrv_rvv.standalone_reduction_scalar_result_runtime_boundary",
          description.standaloneReductionScalarResultRuntimeBoundary,
          "selected typed RVV standalone reduction scalar result boundary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.memory_form",
          plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
              description.memoryForm),
          "selected typed RVV standalone reduction memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_control_plan",
          description.runtimeControlPlanID,
          "selected typed RVV standalone reduction runtime AVL/VL control plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order", description.runtimeABIOrder,
          "selected typed RVV standalone reduction runtime ABI order"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.required_header_declarations",
          description.requiredHeaderDeclarations,
          "selected typed RVV standalone reduction route header requirements"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.c_type_mapping",
          description.cTypeMappingSummary,
          "selected typed RVV standalone reduction route type mapping summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.reduction_accumulator_layout",
          description.reductionAccumulatorLayout,
          "selected typed RVV standalone reduction accumulator layout"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.reduction_result_layout",
          description.reductionResultLayout,
          "selected typed RVV standalone reduction result layout"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.reduction_store_vl",
          description.reductionStoreVL,
          "selected typed RVV standalone reduction store VL"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.vector_load_intrinsic",
          description.vectorLoadIntrinsic,
          "selected typed RVV standalone reduction vector load leaf"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_splat_intrinsic",
          description.sourceSplatIntrinsic,
          "selected typed RVV standalone reduction source splat leaf"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.scalar_seed_splat_intrinsic",
          description.scalarSeedSplatIntrinsic,
          "selected typed RVV standalone reduction scalar seed splat leaf"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.reduction_intrinsic", description.intrinsic,
          "selected typed RVV standalone reduction intrinsic leaf"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.scalar_result_store_intrinsic",
          description.storeIntrinsic,
          "selected typed RVV standalone reduction scalar-result store leaf"))
    return error;

  if (isRVVComputedMaskStandaloneReductionRouteFamilyOperation(
          description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_route_family_plan",
            description.accumulationRouteFamilyPlanID,
            "selected typed RVV computed-mask standalone accumulation "
            "route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_compute_suffix",
            description.accumulationComputeSuffix,
            "selected typed RVV computed-mask standalone accumulation compute "
            "suffix"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_mask_producer_source",
            description.accumulationMaskProducerSource,
            "selected typed RVV computed-mask standalone accumulation mask "
            "producer"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_accumulator_contract",
            description.accumulationAccumulatorContract,
            "selected typed RVV computed-mask standalone accumulation "
            "accumulator contract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_result_contract",
            description.accumulationResultContract,
            "selected typed RVV computed-mask standalone accumulation result "
            "contract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_scalar_carry_contract",
            description.accumulationScalarCarryContract,
            "selected typed RVV computed-mask standalone accumulation scalar "
            "carry contract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_role", description.maskRole,
            "selected typed RVV computed-mask standalone reduction mask role"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_source", description.maskSource,
            "selected typed RVV computed-mask standalone reduction mask source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_memory_form", description.maskMemoryForm,
            "selected typed RVV computed-mask standalone reduction mask memory "
            "form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.inactive_lane_zeroing_requirement",
            description.inactiveLaneZeroingRequirement,
            "selected typed RVV computed-mask standalone reduction inactive "
            "lane neutral requirement"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.compare_predicate_kind",
            description.comparePredicateKind,
            "selected typed RVV computed-mask standalone reduction compare "
            "predicate"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.compare_intrinsic",
            description.compareIntrinsic,
            "selected typed RVV computed-mask standalone reduction compare leaf"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.masked_merge_intrinsic",
            description.maskedMergeIntrinsic,
            "selected typed RVV computed-mask standalone reduction merge leaf"))
      return error;
    const llvm::StringRef expectedRHSBroadcast =
        isRVVRuntimeScalarComputedMaskStandaloneReductionRouteFamilyOperation(
            description.operation)
            ? description.rhsBroadcastIntrinsic
            : llvm::StringRef();
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.rhs_broadcast_intrinsic",
            expectedRHSBroadcast,
            "selected typed RVV runtime-scalar computed-mask standalone "
            "reduction RHS broadcast leaf"))
      return error;
  } else {
    constexpr llvm::StringLiteral staleComputedMaskMirrors[] = {
        "tcrv_rvv.accumulation_route_family_plan",
        "tcrv_rvv.accumulation_compute_suffix",
        "tcrv_rvv.accumulation_mask_producer_source",
        "tcrv_rvv.accumulation_accumulator_contract",
        "tcrv_rvv.accumulation_result_contract",
        "tcrv_rvv.accumulation_scalar_carry_contract",
        "tcrv_rvv.mask_role",
        "tcrv_rvv.mask_source",
        "tcrv_rvv.mask_memory_form",
        "tcrv_rvv.inactive_lane_zeroing_requirement",
        "tcrv_rvv.compare_predicate_kind",
        "tcrv_rvv.compare_intrinsic",
        "tcrv_rvv.masked_merge_intrinsic",
        "tcrv_rvv.rhs_broadcast_intrinsic"};
    for (llvm::StringRef key : staleComputedMaskMirrors)
      if (llvm::Error error =
              requireEmptyStandaloneReductionAccumulationStaleMirror(
                  candidate, key,
                  "selected typed RVV computed-mask standalone reduction "
                  "mirror"))
        return error;
  }

  constexpr llvm::StringLiteral staleRouteFamilyMirrors[] = {
      "tcrv_rvv.elementwise_arithmetic_route_family_plan",
      "tcrv_rvv.scalar_broadcast_elementwise_route_family_plan",
      "tcrv_rvv.widening_conversion_route_family_plan",
      "tcrv_rvv.plain_compare_select_route_family_plan",
      "tcrv_rvv.computed_mask_select_route_family_plan",
      "tcrv_rvv.computed_mask_memory_route_family_plan",
      "tcrv_rvv.segment2_memory_route_family_plan",
      "tcrv_rvv.plain_macc_route_family_plan",
      "tcrv_rvv.scalar_broadcast_macc_route_family_plan",
      "tcrv_rvv.contraction_route_family_plan",
      "tcrv_rvv.base_memory_movement_route_family_plan",
      "tcrv_rvv.widening_macc_relation",
      "tcrv_rvv.widening_dot_relation",
      "tcrv_rvv.widening_dot_reduction_store_vl"};
  for (llvm::StringRef key : staleRouteFamilyMirrors)
    if (llvm::Error error =
            requireEmptyStandaloneReductionAccumulationStaleMirror(
                candidate, key,
                "selected typed RVV non-standalone route-family mirror"))
      return error;

  return llvm::Error::success();
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

struct RVVExpectedCompareSelectMaskRuntimeABIParameter {
  std::string cName;
  std::string cType;
  support::RuntimeABIParameterRole role;
};

llvm::StringRef getRVVCompareSelectMaskExpectedRuntimeABIOrder(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::CmpSelect:
    return "lhs,rhs,out,n";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskSelect:
    return "cmp_lhs,cmp_rhs,true_value,false_value,out,n";
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
    return "lhs,rhs_scalar,true_value,false_value,out,n";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarDualCompareMaskAndSelect:
    return "cmp_lhs_a,rhs_scalar_a,cmp_lhs_b,rhs_scalar_b,true_value,false_value,out,n";
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskLoadStore:
    return "lhs,rhs_scalar,src,dst,n";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return "cmp_lhs,cmp_rhs,src,dst,n";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return "cmp_lhs,cmp_rhs,src,dst,n,dst_stride_bytes";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
    return "cmp_lhs,cmp_rhs,src,dst,n,src_stride_bytes";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedGatherLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
    return "cmp_lhs,cmp_rhs,src,index,dst,n";
  default:
    return {};
  }
}

std::string getRVVCompareSelectMaskElementCType(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.elementTypeName == "i64" || description.sew == 64)
    return "int64_t";
  if (description.elementTypeName == "i16" || description.sew == 16)
    return "int16_t";
  return "int32_t";
}

std::string makeRVVCompareSelectMaskConstPointerType(llvm::StringRef cType) {
  return (llvm::Twine("const ") + cType + " *").str();
}

std::string makeRVVCompareSelectMaskMutablePointerType(llvm::StringRef cType) {
  return (llvm::Twine(cType) + " *").str();
}

llvm::SmallVector<RVVExpectedCompareSelectMaskRuntimeABIParameter, 8>
getRVVCompareSelectMaskExpectedRuntimeABIParameters(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  using OperationKind = plugin::rvv::RVVSelectedBodyOperationKind;
  using Role = support::RuntimeABIParameterRole;
  llvm::SmallVector<RVVExpectedCompareSelectMaskRuntimeABIParameter, 8>
      expected;
  const std::string elementCType =
      getRVVCompareSelectMaskElementCType(description);
  const std::string constElementPointer =
      makeRVVCompareSelectMaskConstPointerType(elementCType);
  const std::string mutableElementPointer =
      makeRVVCompareSelectMaskMutablePointerType(elementCType);

  auto add = [&](llvm::StringRef cName, llvm::StringRef cType, Role role) {
    expected.push_back(
        RVVExpectedCompareSelectMaskRuntimeABIParameter{cName.str(),
                                                        cType.str(), role});
  };

  switch (description.operation) {
  case OperationKind::CmpSelect:
    add("lhs", constElementPointer, Role::LHSInputBuffer);
    add("rhs", constElementPointer, Role::RHSInputBuffer);
    add("out", mutableElementPointer, Role::OutputBuffer);
    add("n", "size_t", Role::RuntimeElementCount);
    break;
  case OperationKind::ComputedMaskSelect:
    add("cmp_lhs", constElementPointer, Role::LHSInputBuffer);
    add("cmp_rhs", constElementPointer, Role::RHSInputBuffer);
    add("true_value", constElementPointer, Role::TrueValueInputBuffer);
    add("false_value", constElementPointer, Role::FalseValueInputBuffer);
    add("out", mutableElementPointer, Role::OutputBuffer);
    add("n", "size_t", Role::RuntimeElementCount);
    break;
  case OperationKind::RuntimeScalarCompareSelect:
    add("lhs", constElementPointer, Role::LHSInputBuffer);
    add("rhs_scalar", elementCType, Role::RHSScalarValue);
    add("true_value", constElementPointer, Role::TrueValueInputBuffer);
    add("false_value", constElementPointer, Role::FalseValueInputBuffer);
    add("out", mutableElementPointer, Role::OutputBuffer);
    add("n", "size_t", Role::RuntimeElementCount);
    break;
  case OperationKind::RuntimeScalarDualCompareMaskAndSelect:
    add("cmp_lhs_a", constElementPointer, Role::LHSInputBuffer);
    add("rhs_scalar_a", elementCType, Role::RHSScalarValue);
    add("cmp_lhs_b", constElementPointer, Role::RHSInputBuffer);
    add("rhs_scalar_b", elementCType, Role::RHSSecondaryScalarValue);
    add("true_value", constElementPointer, Role::TrueValueInputBuffer);
    add("false_value", constElementPointer, Role::FalseValueInputBuffer);
    add("out", mutableElementPointer, Role::OutputBuffer);
    add("n", "size_t", Role::RuntimeElementCount);
    break;
  case OperationKind::RuntimeScalarComputedMaskStore:
  case OperationKind::RuntimeScalarComputedMaskLoadStore:
    add("lhs", constElementPointer, Role::LHSInputBuffer);
    add("rhs_scalar", elementCType, Role::RHSScalarValue);
    add("src", constElementPointer, Role::SourceInputBuffer);
    add("dst", mutableElementPointer, Role::OutputBuffer);
    add("n", "size_t", Role::RuntimeElementCount);
    break;
  case OperationKind::ComputedMaskUnitLoadStore:
    add("cmp_lhs", "const int32_t *", Role::LHSInputBuffer);
    add("cmp_rhs", "const int32_t *", Role::RHSInputBuffer);
    add("src", "const int32_t *", Role::SourceInputBuffer);
    add("dst", "int32_t *", Role::OutputBuffer);
    add("n", "size_t", Role::RuntimeElementCount);
    break;
  case OperationKind::ComputedMaskStridedStore:
    add("cmp_lhs", "const int32_t *", Role::LHSInputBuffer);
    add("cmp_rhs", "const int32_t *", Role::RHSInputBuffer);
    add("src", "const int32_t *", Role::SourceInputBuffer);
    add("dst", "int32_t *", Role::OutputBuffer);
    add("n", "size_t", Role::RuntimeElementCount);
    add("dst_stride_bytes", "size_t", Role::DestinationByteStride);
    break;
  case OperationKind::ComputedMaskStridedLoadUnitStore:
    add("cmp_lhs", "const int32_t *", Role::LHSInputBuffer);
    add("cmp_rhs", "const int32_t *", Role::RHSInputBuffer);
    add("src", "const int32_t *", Role::SourceInputBuffer);
    add("dst", "int32_t *", Role::OutputBuffer);
    add("n", "size_t", Role::RuntimeElementCount);
    add("src_stride_bytes", "size_t", Role::SourceByteStride);
    break;
  case OperationKind::ComputedMaskIndexedGatherLoadUnitStore:
  case OperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
    add("cmp_lhs", "const int32_t *", Role::LHSInputBuffer);
    add("cmp_rhs", "const int32_t *", Role::RHSInputBuffer);
    add("src", "const int32_t *", Role::SourceInputBuffer);
    add("index", "const uint32_t *", Role::IndexInputBuffer);
    add("dst", "int32_t *", Role::OutputBuffer);
    add("n", "size_t", Role::RuntimeElementCount);
    break;
  default:
    break;
  }

  return expected;
}

llvm::Error validateRVVCompareSelectMaskRuntimeABIFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  llvm::StringRef expectedOrder =
      getRVVCompareSelectMaskExpectedRuntimeABIOrder(description.operation);
  if (expectedOrder.empty())
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires a known "
        "compare/select mask runtime ABI order owner before artifact export");
  if (description.runtimeABIOrder != expectedOrder)
    return makeRVVTargetRouteError(
        llvm::Twine("compare/select mask target artifact consumer requires "
                    "provider-derived runtime ABI order '") +
        expectedOrder + "' but was '" + description.runtimeABIOrder + "'");

  llvm::SmallVector<RVVExpectedCompareSelectMaskRuntimeABIParameter, 8>
      expectedParameters =
          getRVVCompareSelectMaskExpectedRuntimeABIParameters(description);
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
    const RVVExpectedCompareSelectMaskRuntimeABIParameter &expected =
        expectedParameters[index];
    if (actual.cName != expected.cName || actual.cType != expected.cType ||
        actual.role != expected.role ||
        actual.ownership !=
            support::RuntimeABIParameterOwnership::TargetExportABIOwned)
      return makeRVVTargetRouteError(
          llvm::Twine("compare/select mask target artifact consumer requires "
                      "provider-derived runtime ABI parameter[") +
          llvm::Twine(index) + "] to bind '" + expected.cName + "' as " +
          support::stringifyRuntimeABIParameterRole(expected.role) +
          " with C type '" + expected.cType + "' and ownership '" +
          support::stringifyRuntimeABIParameterOwnership(
              support::RuntimeABIParameterOwnership::TargetExportABIOwned) +
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
  case plugin::rvv::RVVSelectedBodyOperationKind::CmpSelect:
    return "provider_supported_mirror:rvv-plain-compare-select-plan-validated";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskSelect:
    return "provider_supported_mirror:rvv-computed-mask-select-plan-validated";
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
    return "provider_supported_mirror:rvv-runtime-scalar-cmp-select-plan-validated";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarDualCompareMaskAndSelect:
    return "provider_supported_mirror:rvv-runtime-scalar-dual-cmp-mask-and-select-plan-validated";
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
    return "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-store-plan-validated";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskLoadStore:
    return "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-load-store-plan-validated";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return "provider_supported_mirror:rvv-computed-mask-unit-load-store-plan-validated";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return "provider_supported_mirror:rvv-computed-mask-strided-store-plan-validated";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
    return "provider_supported_mirror:rvv-computed-mask-strided-load-plan-validated";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedGatherLoadUnitStore:
    return "provider_supported_mirror:rvv-computed-mask-indexed-gather-load-plan-validated";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
    return "provider_supported_mirror:rvv-computed-mask-indexed-scatter-store-plan-validated";
  default:
    return {};
  }
}

llvm::StringRef getRVVCompareSelectMaskExpectedTargetLeafProfile(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::CmpSelect:
    return "rvv-v1-typed-plain-compare-select-leaf-profile.v1";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskSelect:
    return "rvv-v1-typed-computed-mask-select-leaf-profile.v1";
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
    return "rvv-v1-typed-runtime-scalar-cmp-select-leaf-profile.v1";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarDualCompareMaskAndSelect:
    return "rvv-v1-typed-runtime-scalar-dual-cmp-mask-and-select-leaf-profile.v1";
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
    return "rvv-v1-typed-runtime-scalar-cmp-masked-store-leaf-profile.v1";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskLoadStore:
    return "rvv-v1-typed-runtime-scalar-cmp-masked-load-store-leaf-profile.v1";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return "rvv-v1-e32m1-computed-mask-unit-load-store-leaf-profile.v1";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return "rvv-v1-e32m1-computed-mask-strided-store-leaf-profile.v1";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
    return "rvv-v1-e32m1-computed-mask-strided-load-leaf-profile.v1";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedGatherLoadUnitStore:
    return "rvv-v1-e32m1-computed-mask-indexed-gather-load-leaf-profile.v1";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
    return "rvv-v1-e32m1-computed-mask-indexed-scatter-store-leaf-profile.v1";
  default:
    return {};
  }
}

llvm::StringRef getRVVCompareSelectMaskExpectedRequiredHeaderDeclarations(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (isRVVCompareSelectMaskRouteFamilyOperation(operation))
    return "stddef.h,stdint.h,riscv_vector.h";
  return {};
}

llvm::StringRef getRVVCompareSelectMaskExpectedCTypeMappingSummary(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::CmpSelect:
    return "vl:size_t,lhs/rhs:typed-vector,mask:typed-mask,result:typed-vector";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskSelect:
    return "vl:size_t,compare:true_false:typed-vector,mask:typed-mask,result:typed-vector";
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
    return "vl:size_t,lhs:typed-vector,rhs_scalar:typed-scalar,mask:typed-mask,true_false:typed-vector,result:typed-vector";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarDualCompareMaskAndSelect:
    return "vl:size_t,cmp_lhs_a:typed-vector,rhs_scalar_a:typed-scalar,cmp_lhs_b:typed-vector,rhs_scalar_b:typed-scalar,mask_a:typed-mask,mask_b:typed-mask,mask_and:typed-mask,true_false:typed-vector,result:typed-vector";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStore:
    return "vl:size_t,lhs_payload:typed-vector,rhs_scalar:typed-scalar,mask:typed-mask,dst:masked-store";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskLoadStore:
    return "vl:size_t,lhs/source/passthrough:typed-vector,rhs_scalar:typed-scalar,mask:typed-mask,result:masked-load-store";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return "vl:size_t,compare/source/passthrough:signed-e32m1,mask:b32,result:masked-load-store";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return "vl:size_t,compare/source:signed-e32m1,mask:b32,dst:masked-strided-store";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
    return "vl:size_t,compare/source/passthrough:signed-e32m1,mask:b32,result:masked-strided-load-store";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedGatherLoadUnitStore:
    return "vl:size_t,compare/source/passthrough:signed-e32m1,index:u32m1,mask:b32,result:masked-indexed-load-store";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
    return "vl:size_t,compare/source:signed-e32m1,index:u32m1,mask:b32,dst:masked-indexed-store";
  default:
    return {};
  }
}

llvm::StringRef getRVVCompareSelectMaskExpectedMaskProducerSource(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskSelect:
    return "vector-compare-rhs-load";
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
    return "runtime-scalar-splat-compare-rhs";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarDualCompareMaskAndSelect:
    return "dual-runtime-scalar-splat-compare-rhs-mask-and";
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskLoadStore:
    return "runtime-scalar-splat-compare-rhs";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedGatherLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
    return "vector-compare-rhs-load";
  default:
    return {};
  }
}

llvm::StringRef getRVVCompareSelectMaskExpectedSelectLayout(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::CmpSelect:
    return "select-lhs-when-mask-else-rhs";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarDualCompareMaskAndSelect:
    return "select-true-value-when-mask-else-false-value";
  default:
    return {};
  }
}

llvm::StringRef getRVVCompareSelectMaskExpectedSourceMemoryForm(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::CmpSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarDualCompareMaskAndSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskLoadStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
    return "unit-stride-load";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
    return "masked-strided-load";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedGatherLoadUnitStore:
    return "masked-indexed-load";
  default:
    return {};
  }
}

llvm::StringRef getRVVCompareSelectMaskExpectedDestinationMemoryForm(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::CmpSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarDualCompareMaskAndSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskLoadStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedGatherLoadUnitStore:
    return "unit-stride-store";
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
    return "masked-unit-store";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return "masked-strided-store";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
    return "masked-indexed-store";
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
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedGatherLoadUnitStore:
    return "masked-off-lanes-preserve-old-destination";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return "masked-strided-store-false-lanes-preserve-output-buffer";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
    return "masked-indexed-store-false-lanes-preserve-output-buffer";
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
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedGatherLoadUnitStore:
    return "old-destination-vector-preserves-inactive-lanes";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return "masked-strided-store-has-no-passthrough-load";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
    return "masked-indexed-store-has-no-passthrough-load";
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
    return "unit-stride-compare-source-byte-strided-masked-destination-runtime-abi";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
    return "unit-stride-compare-byte-strided-masked-source-old-destination-runtime-abi";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedGatherLoadUnitStore:
    return "unit-stride-compare-indexed-masked-source-old-destination-runtime-abi";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
    return "unit-stride-compare-source-indexed-masked-destination-runtime-abi";
  default:
    return {};
  }
}

llvm::StringRef getRVVCompareSelectMaskExpectedStridedMemoryLayout(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return "unit-stride-compare-source-byte-strided-masked-destination-runtime-abi";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
    return "unit-stride-compare-byte-strided-masked-source-old-destination-runtime-abi";
  default:
    return {};
  }
}

llvm::StringRef getRVVCompareSelectMaskExpectedSourceStrideSource(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation == plugin::rvv::RVVSelectedBodyOperationKind::
                          ComputedMaskStridedLoadUnitStore
             ? llvm::StringRef("runtime_abi:src_stride_bytes")
             : llvm::StringRef();
}

llvm::StringRef getRVVCompareSelectMaskExpectedDestinationStrideSource(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
                 plugin::rvv::RVVSelectedBodyOperationKind::
                     ComputedMaskStridedStore
             ? llvm::StringRef("runtime_abi:dst_stride_bytes")
             : llvm::StringRef();
}

bool isRVVCompareSelectMaskIndexedMemoryOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 ComputedMaskIndexedGatherLoadUnitStore ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 ComputedMaskIndexedScatterStoreUnitLoad;
}

llvm::StringRef getRVVCompareSelectMaskExpectedIndexedDataMemoryForm(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
                 plugin::rvv::RVVSelectedBodyOperationKind::
                     ComputedMaskIndexedGatherLoadUnitStore
             ? llvm::StringRef("masked-indexed-load")
             : llvm::StringRef();
}

llvm::StringRef getRVVCompareSelectMaskExpectedIndexedDestinationMemoryForm(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
                 plugin::rvv::RVVSelectedBodyOperationKind::
                     ComputedMaskIndexedScatterStoreUnitLoad
             ? llvm::StringRef("masked-indexed-store")
             : llvm::StringRef();
}

llvm::StringRef getRVVCompareSelectMaskExpectedIndexUniqueness(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
                 plugin::rvv::RVVSelectedBodyOperationKind::
                     ComputedMaskIndexedScatterStoreUnitLoad
             ? llvm::StringRef("unique")
             : llvm::StringRef();
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
  case plugin::rvv::RVVSelectedBodyOperationKind::CmpSelect:
    return 6;
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
    return 8;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarDualCompareMaskAndSelect:
    return 12;
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
    return 9;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
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
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  constexpr llvm::StringLiteral consumerLabel(
      "compare/select mask target artifact consumer");
  if (llvm::Error error =
          validateRVVCompareSelectMaskRuntimeABIFacts(description))
    return error;
  if (description.setVLIntrinsic.empty() || description.vectorLoadIntrinsic.empty() ||
      description.compareIntrinsic.empty() || description.maskName.empty() ||
      description.resultName.empty() || description.maskCType.empty() ||
      description.vectorCType.empty() || description.vlCType.empty() ||
      description.emitCFullChunkVLName.empty() ||
      description.emitCLoopVLName.empty() ||
      description.emitCLoopInductionName.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived setvl, compare, mask/result, vector/VL, "
        "and loop facts before validating route statements");

  if (route.getCallOpaqueSteps().size() != 1)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires exactly one provider-built pre-loop setvl statement before "
        "artifact export");
  const support::RuntimeABIParameter *runtimeN =
      findRuntimeElementCountABIParameter(description);
  if (!runtimeN)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires a provider-derived runtime element count ABI parameter");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          preLoopSetVL, consumerLabel, "pre-loop setvl",
          description.setVLIntrinsic, {{runtimeN->cName, runtimeN->cType}},
          description.emitCFullChunkVLName, description.vlCType))
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
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.upperBound.expression != runtimeN->cName ||
      loop.upperBound.cType != runtimeN->cType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-built loop bounds and step to mirror runtime "
        "n/AVL/VL facts");
  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeN->cName) + " - " +
       description.emitCLoopInductionName)
          .str();
  const std::size_t expectedBodyStepCount =
      getRVVCompareSelectMaskExpectedLoopBodyStepCount(description.operation);
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
          description.setVLIntrinsic, {{expectedRemainingAVL, description.vlCType}},
          description.emitCLoopVLName, description.vlCType))
    return error;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          llvm::Twine(consumerLabel) +
          " requires loop statements to carry selected typed RVV source "
          "provenance");

  auto pointerAtInduction =
      [&](const support::RuntimeABIParameter &abi) -> std::string {
    return (llvm::StringRef(abi.cName) + " + " +
            description.emitCLoopInductionName)
        .str();
  };
  auto stridedSourcePointerAtInduction =
      [&](const support::RuntimeABIParameter &sourceABI,
          const support::RuntimeABIParameter &strideABI) -> std::string {
    return ("(const int32_t *)((const uint8_t *)" +
            llvm::StringRef(sourceABI.cName) + " + (" +
            description.emitCLoopInductionName + " * " + strideABI.cName +
            "))")
        .str();
  };
  auto stridedDestinationPointerAtInduction =
      [&](const support::RuntimeABIParameter &destinationABI,
          const support::RuntimeABIParameter &strideABI) -> std::string {
    return ("(int32_t *)((uint8_t *)" + llvm::StringRef(destinationABI.cName) +
            " + (" + description.emitCLoopInductionName + " * " +
            strideABI.cName + "))")
        .str();
  };
  auto validateVectorLoad =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          llvm::StringRef stepLabel, const support::RuntimeABIParameter &abi,
          llvm::StringRef resultName) -> llvm::Error {
    return validateRVVProviderBuiltRouteStep(
        step, consumerLabel, stepLabel, description.vectorLoadIntrinsic,
        {{pointerAtInduction(abi), abi.cType},
         {description.emitCLoopVLName, description.vlCType}},
        resultName, description.vectorCType);
  };
  auto validateSplat =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          llvm::StringRef stepLabel, const support::RuntimeABIParameter &abi,
          llvm::StringRef resultName) -> llvm::Error {
    return validateRVVProviderBuiltRouteStep(
        step, consumerLabel, stepLabel, description.rhsBroadcastIntrinsic,
        {{abi.cName, abi.cType},
         {description.emitCLoopVLName, description.vlCType}},
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
         {description.emitCLoopVLName, description.vlCType}},
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
         {description.emitCLoopVLName, description.vlCType}},
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
         {description.emitCLoopVLName, description.vlCType}});
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
             {description.emitCLoopVLName, description.vlCType}},
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
  const bool isRuntimeScalarMemory =
      isRuntimeScalarStore || isRuntimeScalarLoadStore;
  const bool isStridedStore =
      description.operation == OperationKind::ComputedMaskStridedStore;
  const bool isStridedLoad =
      description.operation == OperationKind::ComputedMaskStridedLoadUnitStore;
  const bool isIndexedGather =
      description.operation ==
      OperationKind::ComputedMaskIndexedGatherLoadUnitStore;
  const bool isIndexedScatter =
      description.operation ==
      OperationKind::ComputedMaskIndexedScatterStoreUnitLoad;
  const bool isIndexed = isIndexedGather || isIndexedScatter;
  const bool isLoadMerge =
      isRuntimeScalarLoadStore ||
      description.operation == OperationKind::ComputedMaskUnitLoadStore ||
      isStridedLoad || isIndexedGather;
  const bool isStoreOnly =
      isRuntimeScalarStore || isStridedStore || isIndexedScatter;

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
             {description.emitCLoopVLName, description.vlCType}},
            "index_vec", description.indexVectorCType))
      return error;
    if (llvm::Error error = validateRVVProviderBuiltRouteStep(
            loop.bodySteps[stepIndex++], consumerLabel, "index scale",
            description.indexScaleIntrinsic,
            {{"index_vec", description.indexVectorCType},
             {"4", "uint32_t"},
             {description.emitCLoopVLName, description.vlCType}},
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
             {description.emitCLoopVLName, description.vlCType}}))
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
             {description.emitCLoopVLName, description.vlCType}}))
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
             {description.emitCLoopVLName, description.vlCType}}))
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
         {description.emitCLoopVLName, description.vlCType}});
  }
  if (isIndexedScatter) {
    return validateRVVProviderBuiltRouteStep(
        loop.bodySteps[stepIndex++], consumerLabel, "masked indexed store",
        description.indexedStoreIntrinsic,
        {{description.maskName, description.maskCType},
         {destinationABI.cName, destinationABI.cType},
         {"byte_offsets", description.indexVectorCType},
         {"source_vec", description.vectorCType},
         {description.emitCLoopVLName, description.vlCType}});
  }
  if (isRuntimeScalarStore) {
    return validateRVVProviderBuiltRouteStep(
        loop.bodySteps[stepIndex++], consumerLabel, "masked unit store",
        description.storeIntrinsic,
        {{description.maskName, description.maskCType},
         {pointerAtInduction(destinationABI), destinationABI.cType},
         {"source_vec", description.vectorCType},
         {description.emitCLoopVLName, description.vlCType}});
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

  if (isRVVPlainCompareSelectMaskRouteFamilyOperation(description.operation)) {
    if (description.plainCompareSelectRouteFamilyPlanID.empty() ||
        description.selectLayout.empty() ||
        description.inactiveLaneContract.empty() ||
        description.maskedPassthroughLayout.empty())
      return makeRVVTargetRouteError(
          "plain compare/select target artifact consumer requires "
          "provider-derived plain compare-select plan, select layout, "
          "inactive-lane, and passthrough facts before artifact export");
    if (!description.computedMaskSelectRouteFamilyPlanID.empty() ||
        !description.computedMaskMemoryRouteFamilyPlanID.empty() ||
        !description.computedMaskSelectMaskProducerSource.empty() ||
        !description.computedMaskMemoryMaskProducerSource.empty())
      return makeRVVTargetRouteError(
          "plain compare/select target artifact consumer rejects stale "
          "computed-mask select or memory route-family facts");
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "plain compare-select route-family plan",
            description.plainCompareSelectRouteFamilyPlanID,
            "rvv-plain-compare-select-route-family-plan.v1"))
      return error;
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "select layout", description.selectLayout,
            getRVVCompareSelectMaskExpectedSelectLayout(description.operation)))
      return error;
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "inactive-lane contract", description.inactiveLaneContract,
            "masked-off-lanes-preserve-passthrough-vector"))
      return error;
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "masked passthrough layout", description.maskedPassthroughLayout,
            getRVVCompareSelectMaskExpectedPassthroughLayout(
                description.operation)))
      return error;
  } else if (isRVVComputedMaskSelectRouteFamilyOperation(
                 description.operation)) {
    if (description.computedMaskSelectRouteFamilyPlanID.empty() ||
        description.computedMaskSelectMaskProducerSource.empty() ||
        description.selectLayout.empty())
      return makeRVVTargetRouteError(
          "computed-mask select target artifact consumer requires "
          "provider-derived computed-mask select plan, mask producer source, "
          "and select layout before artifact export");
    if (!description.plainCompareSelectRouteFamilyPlanID.empty() ||
        !description.computedMaskMemoryRouteFamilyPlanID.empty() ||
        !description.computedMaskMemoryMaskProducerSource.empty())
      return makeRVVTargetRouteError(
          "computed-mask select target artifact consumer rejects stale plain "
          "compare-select or computed-mask memory route-family facts");
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "computed-mask select route-family plan",
            description.computedMaskSelectRouteFamilyPlanID,
            "rvv-computed-mask-select-route-family-plan.v1"))
      return error;
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "computed-mask select producer source",
            description.computedMaskSelectMaskProducerSource,
            getRVVCompareSelectMaskExpectedMaskProducerSource(
                description.operation)))
      return error;
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "select layout", description.selectLayout,
            getRVVCompareSelectMaskExpectedSelectLayout(description.operation)))
      return error;
    if (description.operation ==
            plugin::rvv::RVVSelectedBodyOperationKind::
                RuntimeScalarDualCompareMaskAndSelect &&
        description.maskComposition.empty())
      return makeRVVTargetRouteError(
          "dual compare/select target artifact consumer requires "
          "provider-derived mask composition facts before artifact export");
    if (description.operation ==
        plugin::rvv::RVVSelectedBodyOperationKind::
            RuntimeScalarDualCompareMaskAndSelect) {
      if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
              "mask composition", description.maskComposition, "and"))
        return error;
      if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
              "mask role", description.maskRole,
              "predicate-mask-produced-by-mask-and"))
        return error;
      if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
              "mask source", description.maskSource,
              "mask-and-of-two-runtime-scalar-compare-produced-masks"))
        return error;
      if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
              "mask memory form", description.maskMemoryForm,
              "composed-compare-produced-mask"))
        return error;
    } else {
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
    }
  } else if (isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(
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
    const bool isIndexed =
        isRVVCompareSelectMaskIndexedMemoryOperation(description.operation);
    if (description.indexEEW != (isIndexed ? 32 : 0))
      return makeRVVTargetRouteError(
          llvm::Twine("compare-produced computed-mask memory target artifact "
                      "consumer requires provider-derived index EEW '") +
          llvm::Twine(isIndexed ? 32 : 0) + "' but was '" +
          llvm::Twine(description.indexEEW) + "'");
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "offset unit", description.offsetUnit,
            isIndexed ? llvm::StringRef("element") : llvm::StringRef()))
      return error;
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "index source", description.indexSource,
            isIndexed ? llvm::StringRef("runtime_abi:index")
                      : llvm::StringRef()))
      return error;
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "index uniqueness", description.indexUniqueness,
            getRVVCompareSelectMaskExpectedIndexUniqueness(
                description.operation)))
      return error;
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "indexed data memory form", description.indexedDataMemoryForm,
            getRVVCompareSelectMaskExpectedIndexedDataMemoryForm(
                description.operation)))
      return error;
    if (llvm::Error error = requireRVVCompareSelectMaskProviderField(
            "indexed destination memory form",
            description.indexedDestinationMemoryForm,
            getRVVCompareSelectMaskExpectedIndexedDestinationMemoryForm(
                description.operation)))
      return error;
  }

  if (isRVVComputedMaskSelectRouteFamilyOperation(description.operation) ||
      isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(
          description.operation)) {
    llvm::StringRef expectedOwner =
        isRVVComputedMaskSelectRouteFamilyOperation(description.operation)
            ? llvm::StringRef(kRVVComputedMaskSelectMaskTailPolicyOwner)
            : llvm::StringRef(kRVVComputedMaskMemoryMaskTailPolicyOwner);
    if (description.maskTailPolicyRouteFamilyPlanID !=
            kRVVMaskTailPolicyRouteFamilyPlanID ||
        description.maskTailPolicyOwner != expectedOwner)
      return makeRVVTargetRouteError(
          llvm::Twine("compare/select mask target artifact consumer requires "
                      "provider-derived mask/tail policy owner '") +
          expectedOwner + "' and route-family plan '" +
          kRVVMaskTailPolicyRouteFamilyPlanID + "' before artifact export");
  } else if (!description.maskTailPolicyRouteFamilyPlanID.empty() ||
             !description.maskTailPolicyOwner.empty()) {
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer rejects mask/tail policy "
        "mirrors for routes that are not computed-mask select or "
        "compare-produced computed-mask memory");
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
          "selected typed RVV compare/select mask runtime AVL/VL control plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order", description.runtimeABIOrder,
          "selected typed RVV compare/select mask runtime ABI order"))
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
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-derived required_header_declarations before accepting the "
        "route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires at least "
        "one provider route header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "conversion dtype-policy target artifact consumer saw an empty "
          "provider route header declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("conversion dtype-policy target artifact consumer "
                      "requires rebuilt provider route header '") +
          trimmed + "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVConversionDtypePolicyRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.vlCType.empty() || description.vectorTypeName.empty() ||
      description.vectorCType.empty() || description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-derived VL and result vector C type mapping facts before "
        "artifact export");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("conversion dtype-policy target artifact consumer "
                    "requires rebuilt provider route type mapping "
                    "'!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("conversion dtype-policy target artifact consumer "
                    "requires rebuilt provider route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");

  if (isRVVConversionDtypePolicyWideningRouteFamilyOperation(
          description.operation)) {
    if (description.sourceVectorTypeName.empty() ||
        description.sourceVectorCType.empty())
      return makeRVVTargetRouteError(
          "widening conversion dtype-policy target artifact consumer "
          "requires provider-derived source vector type mapping facts before "
          "artifact export");
    if (!routeHasTypeMapping(route, description.sourceVectorTypeName,
                             description.sourceVectorCType))
      return makeRVVTargetRouteError(
          llvm::Twine("widening conversion dtype-policy target artifact "
                      "consumer requires rebuilt provider route type mapping "
                      "'") +
          description.sourceVectorTypeName + "' -> '" +
          description.sourceVectorCType + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVConversionDtypePolicyRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("conversion dtype-policy target artifact consumer "
                    "requires rebuilt provider route ABI mapping count ") +
        llvm::Twine(description.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        description.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine("conversion dtype-policy target artifact consumer "
                      "requires rebuilt provider route ABI mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("conversion dtype-policy target artifact consumer "
                      "requires rebuilt provider route ABI mapping[") +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVConversionDtypePolicyRuntimeABIFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.runtimeABIOrder != "lhs,out,n")
    return makeRVVTargetRouteError(
        llvm::Twine("conversion dtype-policy target artifact consumer "
                    "requires provider-derived runtime ABI order 'lhs,out,n' "
                    "but was '") +
        description.runtimeABIOrder + "'");
  if (description.runtimeABIParameters.size() != 3)
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-derived runtime ABI parameters for lhs, out, and n before "
        "artifact export");

  struct ExpectedRuntimeABIParameterRole {
    llvm::StringRef cName;
    support::RuntimeABIParameterRole role;
  };
  const ExpectedRuntimeABIParameterRole expectedRoles[] = {
      {"lhs", support::RuntimeABIParameterRole::LHSInputBuffer},
      {"out", support::RuntimeABIParameterRole::OutputBuffer},
      {"n", support::RuntimeABIParameterRole::RuntimeElementCount},
  };
  constexpr size_t expectedRoleCount =
      sizeof(expectedRoles) / sizeof(expectedRoles[0]);
  for (size_t index = 0; index < expectedRoleCount; ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    if (actual.cName != expectedRoles[index].cName ||
        actual.role != expectedRoles[index].role)
      return makeRVVTargetRouteError(
          llvm::Twine("conversion dtype-policy target artifact consumer "
                      "requires provider-derived runtime ABI parameter ") +
          std::to_string(index) + " to bind " + expectedRoles[index].cName +
          " as " +
          support::stringifyRuntimeABIParameterRole(expectedRoles[index].role) +
          " before artifact export");
  }
  return llvm::Error::success();
}

llvm::Error validateRVVConversionDtypePolicyTypedFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.wideningConversionRouteFamilyPlanID !=
      "rvv-widening-conversion-route-family-plan.v1")
    return makeRVVTargetRouteError(
        llvm::Twine("widening conversion dtype-policy target artifact "
                    "consumer requires provider-owned widening conversion "
                    "route-family plan "
                    "'rvv-widening-conversion-route-family-plan.v1' but was "
                    "'") +
        description.wideningConversionRouteFamilyPlanID + "'");
  if (description.typedComputeOpName != "tcrv_rvv.widening_convert")
    return makeRVVTargetRouteError(
        "widening conversion dtype-policy target artifact consumer requires "
        "selected typed tcrv_rvv.widening_convert body before artifact export");

  struct ExpectedConversionFacts {
    plugin::rvv::RVVSelectedBodyOperationKind operation;
    std::int64_t sourceSEW;
    llvm::StringRef sourceLMUL;
    llvm::StringRef sourceVectorTypeName;
    llvm::StringRef sourceVectorCType;
    std::int64_t resultSEW;
    llvm::StringRef resultLMUL;
    llvm::StringRef resultVectorTypeName;
    llvm::StringRef resultVectorCType;
    llvm::StringRef conversionRelation;
    llvm::StringRef cTypeMappingSummary;
    llvm::StringRef providerSupportedMirror;
    llvm::StringRef targetLeafProfile;
  };
  const ExpectedConversionFacts widenI32ToI64 = {
      plugin::rvv::RVVSelectedBodyOperationKind::WidenI32ToI64,
      32,
      "m1",
      "!tcrv_rvv.vector<i32, \"m1\">",
      "vint32m1_t",
      64,
      "m2",
      "!tcrv_rvv.vector<i64, \"m2\">",
      "vint64m2_t",
      "signed-i32m1-to-i64m2",
      "vl:size_t,source:signed-e32m1,result:signed-e64m2",
      "provider_supported_mirror:rvv-widen-i32-to-i64-plan-validated",
      "rvv-v1-i32m1-i64m2-widening-conversion-leaf-profile.v1"};
  const ExpectedConversionFacts widenI16ToI32 = {
      plugin::rvv::RVVSelectedBodyOperationKind::WidenI16ToI32,
      16,
      "mf2",
      "!tcrv_rvv.vector<i16, \"mf2\">",
      "vint16mf2_t",
      32,
      "m1",
      "!tcrv_rvv.vector<i32, \"m1\">",
      "vint32m1_t",
      "signed-i16mf2-to-i32m1",
      "vl:size_t,source:signed-e16mf2,result:signed-e32m1",
      "provider_supported_mirror:rvv-widen-i16-to-i32-plan-validated",
      "rvv-v1-i16mf2-i32m1-widening-conversion-leaf-profile.v1"};

  const ExpectedConversionFacts *expected = nullptr;
  switch (description.operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::WidenI32ToI64:
    expected = &widenI32ToI64;
    break;
  case plugin::rvv::RVVSelectedBodyOperationKind::WidenI16ToI32:
    expected = &widenI16ToI32;
    break;
  default:
    return makeRVVTargetRouteError(
        "widening conversion dtype-policy target artifact consumer rejects "
        "non-widening-conversion operation kinds");
  }

  if (description.sourceSEW != expected->sourceSEW ||
      description.sourceLMUL != expected->sourceLMUL ||
      description.sourceVectorTypeName != expected->sourceVectorTypeName ||
      description.sourceVectorCType != expected->sourceVectorCType ||
      description.sew != expected->resultSEW ||
      description.lmul != expected->resultLMUL ||
      description.vectorTypeName != expected->resultVectorTypeName ||
      description.vectorCType != expected->resultVectorCType ||
      description.conversionRelation != expected->conversionRelation)
    return makeRVVTargetRouteError(
        llvm::Twine("widening conversion dtype-policy target artifact "
                    "consumer requires provider-derived source/result dtype "
                    "policy and conversion relation for '") +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            expected->operation) +
        "' but saw source SEW " + llvm::Twine(description.sourceSEW) +
        ", source LMUL '" + description.sourceLMUL + "', source type '" +
        description.sourceVectorTypeName + "'/'" + description.sourceVectorCType +
        "', result SEW " + llvm::Twine(description.sew) + ", result LMUL '" +
        description.lmul + "', result type '" + description.vectorTypeName +
        "'/'" + description.vectorCType + "', and relation '" +
        description.conversionRelation + "'");

  if (description.cTypeMappingSummary != expected->cTypeMappingSummary ||
      description.providerSupportedMirror != expected->providerSupportedMirror ||
      description.targetLeafProfile != expected->targetLeafProfile)
    return makeRVVTargetRouteError(
        llvm::Twine("widening conversion dtype-policy target artifact "
                    "consumer requires provider-owned support, C type, and "
                    "target leaf facts for '") +
        plugin::rvv::stringifyRVVSelectedBodyOperationKind(
            expected->operation) +
        "' but saw support '" + description.providerSupportedMirror +
        "', C type mapping '" + description.cTypeMappingSummary +
        "', and target leaf profile '" + description.targetLeafProfile + "'");

  return llvm::Error::success();
}

llvm::Error validateRVVConversionDtypePolicyRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  constexpr llvm::StringLiteral consumerLabel(
      "conversion dtype-policy target artifact consumer");
  if (description.runtimeABIParameters.size() != 3)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived lhs, out, and n ABI parameters before "
        "validating route statements");
  if (description.resultName.empty() || description.sourceVectorCType.empty() ||
      description.vectorCType.empty() || description.vlCType.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived source/result vector C types, result "
        "name, and VL C type before validating route statements");

  const support::RuntimeABIParameter &sourceABI =
      description.runtimeABIParameters[0];
  const support::RuntimeABIParameter &outABI =
      description.runtimeABIParameters[1];
  const support::RuntimeABIParameter &runtimeNABI =
      description.runtimeABIParameters[2];
  const support::RuntimeABIParameter *runtimeElementCount =
      findRuntimeElementCountABIParameter(description);
  if (!runtimeElementCount || runtimeElementCount != &runtimeNABI)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires runtime n/AVL ABI role to match the selected conversion "
        "ABI order before validating route statements");

  if (route.getCallOpaqueSteps().size() != 1)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires exactly one provider-built pre-loop setvl statement before "
        "artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          preLoopSetVL, consumerLabel, "pre-loop setvl",
          description.setVLIntrinsic, {{runtimeNABI.cName, runtimeNABI.cType}},
          description.emitCFullChunkVLName, description.vlCType))
    return error;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "conversion dtype-policy target artifact consumer requires "
          "pre-loop statements to carry selected typed RVV source provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires exactly "
        "one provider-built runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-built loop bounds and step to mirror runtime AVL/VL facts");
  if (loop.upperBound.expression != runtimeNABI.cName ||
      loop.upperBound.cType != runtimeNABI.cType)
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-built loop upper bound to use the runtime n/AVL ABI "
        "parameter");
  if (loop.bodySteps.size() != 4)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires exact provider-built loop statement count 4 for setvl, "
        "source load, widening conversion, and output store before artifact "
        "export");

  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeNABI.cName) + " - " +
       description.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[0], consumerLabel, "loop setvl",
          description.setVLIntrinsic,
          {{expectedRemainingAVL, description.vlCType}},
          description.emitCLoopVLName, description.vlCType))
    return error;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "conversion dtype-policy target artifact consumer requires loop "
          "statements to carry selected typed RVV source provenance");

  const std::string expectedSourcePointer =
      (llvm::StringRef(sourceABI.cName) + " + " +
       description.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[1], consumerLabel, "source vector load",
          description.sourceVectorLoadIntrinsic,
          {{expectedSourcePointer, sourceABI.cType},
           {description.emitCLoopVLName, description.vlCType}},
          "lhs_vec", description.sourceVectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[2], consumerLabel, "widening conversion",
          description.intrinsic,
          {{"lhs_vec", description.sourceVectorCType},
           {description.emitCLoopVLName, description.vlCType}},
          description.resultName, description.vectorCType))
    return error;

  const std::string expectedOutPointer =
      (llvm::StringRef(outABI.cName) + " + " +
       description.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[3], consumerLabel, "output store",
          description.storeIntrinsic,
          {{expectedOutPointer, outABI.cType},
           {description.resultName, description.vectorCType},
           {description.emitCLoopVLName, description.vlCType}}))
    return error;

  return llvm::Error::success();
}

llvm::Error validateRVVConversionDtypePolicyRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("conversion dtype-policy target artifact consumer "
                    "requires rebuilt provider route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");
  if (description.providerSupportedMirror.empty())
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires a "
        "provider-supported mirror label after route construction");
  if (description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires provider "
        "route operand binding facts before artifact export");
  if (description.runtimeControlPlanID.empty() ||
      description.runtimeABIOrder.empty() || description.sew == 0 ||
      description.lmul.empty() || description.tailPolicy.empty() ||
      description.maskPolicy.empty() || description.setVLIntrinsic.empty() ||
      description.intrinsic.empty() || description.storeIntrinsic.empty() ||
      description.resultName.empty())
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-derived runtime, dtype, policy, intrinsic, and result facts "
        "before artifact export");

  if (description.memoryForm !=
      plugin::rvv::RVVSelectedBodyMemoryForm::UnitStrideConversion)
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "unit-stride conversion memory form from the selected typed RVV body");
  if (llvm::Error error =
          validateRVVConversionDtypePolicyRuntimeABIFacts(description))
    return error;

  if (description.wideningConversionRouteFamilyPlanID.empty() ||
      description.sourceSEW == 0 || description.sourceLMUL.empty() ||
      description.sourceVectorTypeName.empty() ||
      description.sourceVectorCType.empty() ||
      description.sourceVectorLoadIntrinsic.empty() ||
      description.conversionRelation.empty())
    return makeRVVTargetRouteError(
        "widening conversion dtype-policy target artifact consumer "
        "requires provider-derived source dtype, source vector type, "
        "source load, conversion relation, and widening plan facts before "
        "artifact export");
  if (description.sourceSEW >= description.sew)
    return makeRVVTargetRouteError(
        "widening conversion dtype-policy target artifact consumer requires "
        "source SEW to be smaller than result SEW");
  if (llvm::Error error =
          validateRVVConversionDtypePolicyTypedFacts(description))
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
        "conversion dtype-policy target artifact consumer rejects stale "
        "non-conversion route-family facts");

  if (llvm::Error error =
          validateRVVConversionDtypePolicyRouteHeaders(route, description))
    return error;
  if (llvm::Error error =
          validateRVVConversionDtypePolicyRouteTypeMappings(route, description))
    return error;
  if (llvm::Error error =
          validateRVVConversionDtypePolicyRouteABIMappings(route, description))
    return error;
  return validateRVVConversionDtypePolicyRouteStatementPlan(route, description);
}

llvm::Error validateRVVConversionDtypePolicyTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVConversionDtypePolicyRoutePayloadFacts(
      context.route, context.description);
}

llvm::Error requireEmptyConversionDtypePolicyStaleMirror(
    const TargetArtifactCandidate &candidate, llvm::StringRef key,
    llvm::StringRef label) {
  return requireCandidateMetadataMirror(candidate, key, "", label);
}

llvm::Error validateRVVConversionDtypePolicyTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description =
      context.description;

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_plan",
          description.routeOperandBindingPlanID,
          "selected typed RVV conversion dtype-policy binding plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_operands",
          description.routeOperandBindingSummary,
          "selected typed RVV conversion dtype-policy binding summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.provider_supported_mirror",
          description.providerSupportedMirror,
          "selected typed RVV conversion dtype-policy provider support"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.widening_conversion_route_family_plan",
          description.wideningConversionRouteFamilyPlanID,
          "selected typed RVV widening conversion route-family plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_sew",
          llvm::Twine(description.sourceSEW).str(),
          "selected typed RVV widening conversion source SEW"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_lmul", description.sourceLMUL,
          "selected typed RVV widening conversion source LMUL"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.dest_sew", llvm::Twine(description.sew).str(),
          "selected typed RVV widening conversion destination SEW"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.dest_lmul", description.lmul,
          "selected typed RVV widening conversion destination LMUL"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.conversion_relation",
          description.conversionRelation,
          "selected typed RVV widening conversion relation"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.memory_form",
          plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
              description.memoryForm),
          "selected typed RVV conversion dtype-policy memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_control_plan",
          description.runtimeControlPlanID,
          "selected typed RVV conversion dtype-policy runtime AVL/VL control "
          "plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order", description.runtimeABIOrder,
          "selected typed RVV conversion dtype-policy runtime ABI order"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.required_header_declarations",
          description.requiredHeaderDeclarations,
          "selected typed RVV conversion dtype-policy route header "
          "requirements"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.c_type_mapping",
          description.cTypeMappingSummary,
          "selected typed RVV conversion dtype-policy route type mapping "
          "summary"))
    return error;

  constexpr llvm::StringLiteral staleMirrors[] = {
      "tcrv_rvv.elementwise_arithmetic_route_family_plan",
      "tcrv_rvv.scalar_broadcast_elementwise_route_family_plan",
      "tcrv_rvv.plain_compare_select_route_family_plan",
      "tcrv_rvv.computed_mask_select_route_family_plan",
      "tcrv_rvv.computed_mask_memory_route_family_plan",
      "tcrv_rvv.plain_macc_route_family_plan",
      "tcrv_rvv.scalar_broadcast_macc_route_family_plan",
      "tcrv_rvv.accumulation_route_family_plan",
      "tcrv_rvv.segment2_memory_route_family_plan",
      "tcrv_rvv.standalone_reduction_route_family_plan",
      "tcrv_rvv.contraction_route_family_plan",
      "tcrv_rvv.base_memory_movement_route_family_plan",
      "tcrv_rvv.widening_macc_relation",
      "tcrv_rvv.widening_dot_relation"};
  for (llvm::StringRef key : staleMirrors)
    if (llvm::Error error = requireEmptyConversionDtypePolicyStaleMirror(
            candidate, key,
            "selected typed RVV non-conversion route-family mirror"))
      return error;

  return llvm::Error::success();
}

llvm::Error validateRVVSegment2MemoryRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires provider-derived "
        "required_header_declarations before accepting the route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires at least one "
        "provider route header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "segment2-memory target artifact consumer saw an empty provider "
          "route header declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("segment2-memory target artifact consumer requires "
                      "rebuilt provider route header '") +
          trimmed + "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVSegment2MemoryRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.vlCType.empty() || description.vectorTypeName.empty() ||
      description.vectorCType.empty() || description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires provider-derived "
        "VL, vector, and C type mapping facts before artifact export");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("segment2-memory target artifact consumer requires "
                    "rebuilt provider route type mapping '!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("segment2-memory target artifact consumer requires "
                    "rebuilt provider route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");

  if (isRVVComputedMaskSegment2MemoryRouteFamilyOperation(
          description.operation)) {
    if (description.maskTypeName.empty() || description.maskCType.empty())
      return makeRVVTargetRouteError(
          "computed-mask segment2-memory target artifact consumer requires "
          "provider-derived mask type mapping facts before artifact export");
    if (!routeHasTypeMapping(route, description.maskTypeName,
                             description.maskCType))
      return makeRVVTargetRouteError(
          llvm::Twine("computed-mask segment2-memory target artifact consumer "
                      "requires rebuilt provider route type mapping '") +
          description.maskTypeName + "' -> '" + description.maskCType + "'");
  } else if (!description.maskTypeName.empty() ||
             !description.maskCType.empty()) {
    return makeRVVTargetRouteError(
        "plain segment2-memory target artifact consumer rejects stale mask "
        "type mapping facts");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVSegment2MemoryRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.runtimeABIOrder.empty() ||
      description.runtimeABIParameters.empty())
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires provider-derived "
        "runtime ABI order and ABI parameters before artifact export");
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("segment2-memory target artifact consumer requires "
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
          llvm::Twine("segment2-memory target artifact consumer requires "
                      "rebuilt provider route ABI mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("segment2-memory target artifact consumer requires "
                      "rebuilt provider route ABI mapping[") +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

constexpr llvm::StringLiteral kRVVComputedMaskMemoryRouteFamilyPlanID(
    "rvv-computed-mask-memory-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVComputedMaskMemoryMaskProducerSource(
    "vector-compare-rhs-load");
constexpr llvm::StringLiteral kRVVSegment2RequiredHeaders(
    "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVSegment2TupleCType("vint32m1x2_t");
constexpr llvm::StringLiteral kRVVSegment2MemoryRouteFamilyPlanID(
    "rvv-segment2-memory-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVSegment2DeinterleaveRuntimeABIOrder(
    "src,out0,out1,n");
constexpr llvm::StringLiteral kRVVSegment2InterleaveRuntimeABIOrder(
    "src0,src1,dst,n");
constexpr llvm::StringLiteral kRVVSegment2DeinterleaveRouteOperandPlan(
    "rvv-route-operand-binding:segment2_deinterleave_unit_store.v1");
constexpr llvm::StringLiteral kRVVSegment2InterleaveRouteOperandPlan(
    "rvv-route-operand-binding:segment2_interleave_unit_load.v1");
constexpr llvm::StringLiteral kRVVSegment2DeinterleaveRouteOperandSummary(
    "rvv-route-operand-binding:segment2_deinterleave_unit_store.v1;"
    "src=lhs-input-buffer:src:abi|seg-load-base|src-mem|hdr;"
    "out0=segment-field0-output-buffer:out0:abi|field0-store-base|field0-role|dst-mem|hdr;"
    "out1=segment-field1-output-buffer:out1:abi|field1-store-base|field1-role|dst-mem|hdr;"
    "n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr");
constexpr llvm::StringLiteral kRVVSegment2InterleaveRouteOperandSummary(
    "rvv-route-operand-binding:segment2_interleave_unit_load.v1;"
    "src0=segment-field0-input-buffer:src0:abi|field0-load-base|field0-role|src0-mem|tuple-field0|hdr;"
    "src1=segment-field1-input-buffer:src1:abi|field1-load-base|field1-role|src1-mem|tuple-field1|hdr;"
    "dst=segment-interleaved-output-buffer:dst:abi|seg-store-base|dst-mem|hdr;"
    "n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr");
constexpr llvm::StringLiteral kRVVSegment2DeinterleaveTargetLeafProfile(
    "rvv-v1-e32m1-segment2-deinterleave-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVSegment2InterleaveTargetLeafProfile(
    "rvv-v1-e32m1-segment2-interleave-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVSegment2DeinterleaveProviderMirror(
    "provider_supported_mirror:rvv-segment2-deinterleave-plan-validated");
constexpr llvm::StringLiteral kRVVSegment2InterleaveProviderMirror(
    "provider_supported_mirror:rvv-segment2-interleave-plan-validated");
constexpr llvm::StringLiteral kRVVSegment2DeinterleaveCTypeMapping(
    "vl:size_t,segment2:vint32m1x2,field-outputs:signed-e32m1");
constexpr llvm::StringLiteral kRVVSegment2InterleaveCTypeMapping(
    "vl:size_t,field-inputs:signed-e32m1,segment2:vint32m1x2");
constexpr llvm::StringLiteral kRVVSegment2DeinterleaveMemoryLayout(
    "segment2-interleaved-source-dual-unit-stride-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVSegment2InterleaveMemoryLayout(
    "dual-unit-stride-source-segment2-interleaved-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVSegment2DeinterleaveTypedComputeOp(
    "tcrv_rvv.move");
constexpr llvm::StringLiteral kRVVSegment2InterleaveTypedComputeOp(
    "tcrv_rvv.segment2_store");
constexpr llvm::StringLiteral kRVVSegment2Field0Name("field0_vec");
constexpr llvm::StringLiteral kRVVSegment2Field1Name("field1_vec");
constexpr llvm::StringLiteral kRVVSegment2Field0OutputRole(
    "segment-field0-output-buffer");
constexpr llvm::StringLiteral kRVVSegment2Field1OutputRole(
    "segment-field1-output-buffer");
constexpr llvm::StringLiteral kRVVSegment2Field0InputRole(
    "segment-field0-input-buffer");
constexpr llvm::StringLiteral kRVVSegment2Field1InputRole(
    "segment-field1-input-buffer");
constexpr llvm::StringLiteral kRVVSegment2UnitStrideMemoryForm(
    "unit-stride-load");
constexpr llvm::StringLiteral kRVVSegment2UnitStrideStoreMemoryForm(
    "unit-stride-store");
constexpr llvm::StringLiteral kRVVSegment2InterleavedLoadMemoryForm(
    "segment2-interleaved-unit-stride-load");
constexpr llvm::StringLiteral kRVVSegment2InterleavedStoreMemoryForm(
    "segment2-interleaved-unit-stride-store");
constexpr llvm::StringLiteral kRVVSegment2MaskedMemoryInactiveLaneContract(
    "masked-off-lanes-preserve-old-destination");
constexpr llvm::StringLiteral kRVVSegment2MaskedMemoryPassthroughLayout(
    "old-destination-vector-preserves-inactive-lanes");
constexpr llvm::StringLiteral kRVVSegment2MaskedStoreInactiveLaneContract(
    "masked-store-false-lanes-preserve-output-buffer");
constexpr llvm::StringLiteral kRVVSegment2MaskedStorePassthroughLayout(
    "masked-store-has-no-passthrough-load");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2LoadMemoryLayout(
    "unit-stride-compare-segment2-masked-source-old-fields-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2StoreMemoryLayout(
    "unit-stride-compare-field-payloads-segment2-masked-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2UpdateMemoryLayout(
    "unit-stride-compare-field-payloads-arithmetic-segment2-masked-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2LoadRouteOperandPlan(
    "rvv-route-operand-binding:computed_masked_segment2_load_unit_store.v1");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2LoadRouteOperandSummary(
    "rvv-route-operand-binding:computed_masked_segment2_load_unit_store.v1;"
    "cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call|hdr;"
    "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call|hdr;"
    "src=source-input-buffer:src:abi|mseg-base|mseg-call|src-mem|hdr;"
    "out0=segment-field0-output-buffer:out0:abi|old0-load|f0-pass|"
    "f0-store|f0-role|dst-mem|hdr;"
    "out1=segment-field1-output-buffer:out1:abi|old1-load|f1-pass|"
    "f1-store|f1-role|dst-mem|hdr;"
    "n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2StoreRouteOperandPlan(
    "rvv-route-operand-binding:computed_masked_segment2_store_unit_load.v1");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2StoreRouteOperandSummary(
    "rvv-route-operand-binding:computed_masked_segment2_store_unit_load.v1;"
    "cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call|hdr;"
    "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call|hdr;"
    "src0=segment-field0-input-buffer:src0:abi|f0-load|f0-payload|tuple0|f0-role|src0-mem|hdr;"
    "src1=segment-field1-input-buffer:src1:abi|f1-load|f1-payload|tuple1|f1-role|src1-mem|hdr;"
    "dst=segment-interleaved-output-buffer:dst:abi|mseg-store|dst-mem|hdr;"
    "n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2UpdateRouteOperandPlan(
    "rvv-route-operand-binding:cmseg2_update_unit_load.v1");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2UpdateRouteOperandSummary(
    "rvv-route-operand-binding:cmseg2_update_unit_load.v1;"
    "cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call|hdr;"
    "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call|hdr;"
    "src0=segment-field0-input-buffer:src0:abi|f0-load|f0-payload|"
    "add-lhs|tuple0|f0-role|src0-mem|hdr;"
    "src1=segment-field1-input-buffer:src1:abi|f1-load|f1-payload|"
    "add-rhs|tuple1|f1-role|src1-mem|hdr;"
    "dst=segment-interleaved-output-buffer:dst:abi|mseg-store|dst-mem|hdr;"
    "n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2LoadTargetLeafProfile(
    "rvv-v1-e32m1-computed-mask-segment2-load-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2StoreTargetLeafProfile(
    "rvv-v1-e32m1-computed-mask-segment2-store-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2UpdateTargetLeafProfile(
    "rvv-v1-e32m1-computed-mask-segment2-update-add-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2LoadProviderMirror(
    "provider_supported_mirror:rvv-computed-mask-segment2-load-plan-validated");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2StoreProviderMirror(
    "provider_supported_mirror:rvv-computed-mask-segment2-store-plan-validated");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2UpdateProviderMirror(
    "provider_supported_mirror:rvv-computed-mask-segment2-update-add-plan-validated");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2LoadCTypeMapping(
    "vl:size_t,compare/source/passthrough-fields:signed-e32m1,mask:b32,"
    "segment2:vint32m1x2,result:masked-segment2-load-store");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2StoreCTypeMapping(
    "vl:size_t,compare/field-payloads:signed-e32m1,mask:b32,"
    "segment2:vint32m1x2,dst:masked-segment2-store");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2UpdateCTypeMapping(
    "vl:size_t,compare/field-payloads/update-add:signed-e32m1,mask:b32,"
    "segment2:vint32m1x2,dst:masked-segment2-update-store");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2LoadRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,src,out0,out1,n");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2StoreRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,src0,src1,dst,n");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2LoadField0Name(
    "masked_segment2_field0_vec");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2LoadField1Name(
    "masked_segment2_field1_vec");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2StoreField0Name(
    "masked_segment2_store_field0_vec");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2StoreField1Name(
    "masked_segment2_store_field1_vec");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2UpdateField0Name(
    "masked_segment2_update_field0_vec");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2UpdateField1Name(
    "masked_segment2_update_field1_vec");
constexpr llvm::StringLiteral kRVVSegment2UpdateArithmeticKind("add");
constexpr llvm::StringLiteral kRVVSegment2UpdateArithmeticIntrinsic(
    "__riscv_vadd_vv_i32m1");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2LoadIntrinsic(
    "__riscv_vlseg2e32_v_i32m1x2_tumu");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2MaskedStoreIntrinsic(
    "__riscv_vsseg2e32_v_i32m1x2_m");
constexpr llvm::StringLiteral kRVVSegment2TupleCreateIntrinsic(
    "__riscv_vcreate_v_i32m1x2");
constexpr llvm::StringLiteral kRVVSegment2FieldExtractIntrinsic(
    "__riscv_vget_v_i32m1x2_i32m1");

plugin::rvv::RVVSelectedBodyMemoryForm
getRVVSegment2MemoryExpectedMemoryForm(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2LoadUnitStore:
    return plugin::rvv::RVVSelectedBodyMemoryForm::
        ComputedMaskSegment2LoadUnitStore;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2StoreUnitLoad:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2UpdateUnitLoad:
    return plugin::rvv::RVVSelectedBodyMemoryForm::
        ComputedMaskUnitLoadSegment2Store;
  case plugin::rvv::RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore:
    return plugin::rvv::RVVSelectedBodyMemoryForm::Segment2LoadUnitStore;
  case plugin::rvv::RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad:
    return plugin::rvv::RVVSelectedBodyMemoryForm::UnitLoadSegment2Store;
  default:
    llvm_unreachable("queried segment2 memory form for non-segment2 operation");
  }
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedTypedComputeOp(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2LoadUnitStore:
    return "tcrv_rvv.masked_segment2_load";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2StoreUnitLoad:
    return "tcrv_rvv.masked_segment2_store";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2UpdateUnitLoad:
    return "tcrv_rvv.binary";
  default:
    return {};
  }
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedRuntimeABIOrder(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       ComputedMaskSegment2LoadUnitStore)
    return kRVVComputedMaskSegment2LoadRuntimeABIOrder;
  if (isRVVComputedMaskSegment2MemoryRouteFamilyOperation(operation))
    return kRVVComputedMaskSegment2StoreRuntimeABIOrder;
  return {};
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedRouteOperandPlan(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2LoadUnitStore:
    return kRVVComputedMaskSegment2LoadRouteOperandPlan;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2StoreUnitLoad:
    return kRVVComputedMaskSegment2StoreRouteOperandPlan;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2UpdateUnitLoad:
    return kRVVComputedMaskSegment2UpdateRouteOperandPlan;
  default:
    return {};
  }
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedRouteOperandSummary(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2LoadUnitStore:
    return kRVVComputedMaskSegment2LoadRouteOperandSummary;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2StoreUnitLoad:
    return kRVVComputedMaskSegment2StoreRouteOperandSummary;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2UpdateUnitLoad:
    return kRVVComputedMaskSegment2UpdateRouteOperandSummary;
  default:
    return {};
  }
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedTargetLeafProfile(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2LoadUnitStore:
    return kRVVComputedMaskSegment2LoadTargetLeafProfile;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2StoreUnitLoad:
    return kRVVComputedMaskSegment2StoreTargetLeafProfile;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2UpdateUnitLoad:
    return kRVVComputedMaskSegment2UpdateTargetLeafProfile;
  default:
    return {};
  }
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedProviderMirror(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2LoadUnitStore:
    return kRVVComputedMaskSegment2LoadProviderMirror;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2StoreUnitLoad:
    return kRVVComputedMaskSegment2StoreProviderMirror;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2UpdateUnitLoad:
    return kRVVComputedMaskSegment2UpdateProviderMirror;
  default:
    return {};
  }
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedCTypeMapping(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2LoadUnitStore:
    return kRVVComputedMaskSegment2LoadCTypeMapping;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2StoreUnitLoad:
    return kRVVComputedMaskSegment2StoreCTypeMapping;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2UpdateUnitLoad:
    return kRVVComputedMaskSegment2UpdateCTypeMapping;
  default:
    return {};
  }
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedSegmentMemoryLayout(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2LoadUnitStore:
    return kRVVComputedMaskSegment2LoadMemoryLayout;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2StoreUnitLoad:
    return kRVVComputedMaskSegment2StoreMemoryLayout;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2UpdateUnitLoad:
    return kRVVComputedMaskSegment2UpdateMemoryLayout;
  default:
    return {};
  }
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedInactiveLaneContract(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       ComputedMaskSegment2LoadUnitStore)
    return kRVVSegment2MaskedMemoryInactiveLaneContract;
  if (isRVVComputedMaskSegment2MemoryRouteFamilyOperation(operation))
    return kRVVSegment2MaskedStoreInactiveLaneContract;
  return {};
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedMaskedPassthroughLayout(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       ComputedMaskSegment2LoadUnitStore)
    return kRVVSegment2MaskedMemoryPassthroughLayout;
  if (isRVVComputedMaskSegment2MemoryRouteFamilyOperation(operation))
    return kRVVSegment2MaskedStorePassthroughLayout;
  return {};
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedSourceMemoryForm(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       ComputedMaskSegment2LoadUnitStore)
    return kRVVSegment2InterleavedLoadMemoryForm;
  if (isRVVComputedMaskSegment2MemoryRouteFamilyOperation(operation))
    return kRVVSegment2UnitStrideMemoryForm;
  return {};
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedDestinationMemoryForm(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       ComputedMaskSegment2LoadUnitStore)
    return kRVVSegment2UnitStrideStoreMemoryForm;
  if (isRVVComputedMaskSegment2MemoryRouteFamilyOperation(operation))
    return kRVVSegment2InterleavedStoreMemoryForm;
  return {};
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedField0Role(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       ComputedMaskSegment2LoadUnitStore)
    return kRVVSegment2Field0OutputRole;
  if (isRVVComputedMaskSegment2MemoryRouteFamilyOperation(operation))
    return kRVVSegment2Field0InputRole;
  return {};
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedField1Role(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       ComputedMaskSegment2LoadUnitStore)
    return kRVVSegment2Field1OutputRole;
  if (isRVVComputedMaskSegment2MemoryRouteFamilyOperation(operation))
    return kRVVSegment2Field1InputRole;
  return {};
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedField0Name(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2LoadUnitStore:
    return kRVVComputedMaskSegment2LoadField0Name;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2StoreUnitLoad:
    return kRVVComputedMaskSegment2StoreField0Name;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2UpdateUnitLoad:
    return kRVVComputedMaskSegment2UpdateField0Name;
  default:
    return {};
  }
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedField1Name(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2LoadUnitStore:
    return kRVVComputedMaskSegment2LoadField1Name;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2StoreUnitLoad:
    return kRVVComputedMaskSegment2StoreField1Name;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2UpdateUnitLoad:
    return kRVVComputedMaskSegment2UpdateField1Name;
  default:
    return {};
  }
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedFieldSourceMemoryForm(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       ComputedMaskSegment2StoreUnitLoad ||
      operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       ComputedMaskSegment2UpdateUnitLoad)
    return kRVVSegment2UnitStrideMemoryForm;
  return {};
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedFieldDestinationMemoryForm(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       ComputedMaskSegment2LoadUnitStore)
    return kRVVSegment2UnitStrideStoreMemoryForm;
  return {};
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedUpdateArithmeticKind(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       ComputedMaskSegment2UpdateUnitLoad)
    return kRVVSegment2UpdateArithmeticKind;
  return {};
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedUpdateArithmeticIntrinsic(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       ComputedMaskSegment2UpdateUnitLoad)
    return kRVVSegment2UpdateArithmeticIntrinsic;
  return {};
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedSegmentLoadIntrinsic(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       ComputedMaskSegment2LoadUnitStore)
    return kRVVComputedMaskSegment2LoadIntrinsic;
  return {};
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedSegmentStoreIntrinsic(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       ComputedMaskSegment2StoreUnitLoad ||
      operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       ComputedMaskSegment2UpdateUnitLoad)
    return kRVVComputedMaskSegment2MaskedStoreIntrinsic;
  return {};
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedTupleCreateIntrinsic(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (isRVVComputedMaskSegment2MemoryRouteFamilyOperation(operation))
    return kRVVSegment2TupleCreateIntrinsic;
  return {};
}

llvm::StringRef getRVVComputedMaskSegment2ExpectedFieldExtractIntrinsic(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       ComputedMaskSegment2LoadUnitStore)
    return kRVVSegment2FieldExtractIntrinsic;
  return {};
}

llvm::StringRef getRVVPlainSegment2ExpectedTypedComputeOp(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2DeinterleaveUnitStore)
    return kRVVSegment2DeinterleaveTypedComputeOp;
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2InterleaveUnitLoad)
    return kRVVSegment2InterleaveTypedComputeOp;
  return {};
}

llvm::StringRef getRVVPlainSegment2ExpectedRuntimeABIOrder(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2DeinterleaveUnitStore)
    return kRVVSegment2DeinterleaveRuntimeABIOrder;
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2InterleaveUnitLoad)
    return kRVVSegment2InterleaveRuntimeABIOrder;
  return {};
}

llvm::StringRef getRVVPlainSegment2ExpectedRouteOperandPlan(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2DeinterleaveUnitStore)
    return kRVVSegment2DeinterleaveRouteOperandPlan;
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2InterleaveUnitLoad)
    return kRVVSegment2InterleaveRouteOperandPlan;
  return {};
}

llvm::StringRef getRVVPlainSegment2ExpectedRouteOperandSummary(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2DeinterleaveUnitStore)
    return kRVVSegment2DeinterleaveRouteOperandSummary;
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2InterleaveUnitLoad)
    return kRVVSegment2InterleaveRouteOperandSummary;
  return {};
}

llvm::StringRef getRVVPlainSegment2ExpectedTargetLeafProfile(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2DeinterleaveUnitStore)
    return kRVVSegment2DeinterleaveTargetLeafProfile;
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2InterleaveUnitLoad)
    return kRVVSegment2InterleaveTargetLeafProfile;
  return {};
}

llvm::StringRef getRVVPlainSegment2ExpectedProviderMirror(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2DeinterleaveUnitStore)
    return kRVVSegment2DeinterleaveProviderMirror;
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2InterleaveUnitLoad)
    return kRVVSegment2InterleaveProviderMirror;
  return {};
}

llvm::StringRef getRVVPlainSegment2ExpectedCTypeMapping(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2DeinterleaveUnitStore)
    return kRVVSegment2DeinterleaveCTypeMapping;
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2InterleaveUnitLoad)
    return kRVVSegment2InterleaveCTypeMapping;
  return {};
}

llvm::StringRef getRVVPlainSegment2ExpectedSegmentMemoryLayout(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2DeinterleaveUnitStore)
    return kRVVSegment2DeinterleaveMemoryLayout;
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2InterleaveUnitLoad)
    return kRVVSegment2InterleaveMemoryLayout;
  return {};
}

llvm::StringRef getRVVPlainSegment2ExpectedSourceMemoryForm(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2DeinterleaveUnitStore)
    return kRVVSegment2InterleavedLoadMemoryForm;
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2InterleaveUnitLoad)
    return kRVVSegment2UnitStrideMemoryForm;
  return {};
}

llvm::StringRef getRVVPlainSegment2ExpectedDestinationMemoryForm(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2DeinterleaveUnitStore)
    return kRVVSegment2UnitStrideStoreMemoryForm;
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2InterleaveUnitLoad)
    return kRVVSegment2InterleavedStoreMemoryForm;
  return {};
}

llvm::StringRef getRVVPlainSegment2ExpectedField0Role(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2DeinterleaveUnitStore)
    return kRVVSegment2Field0OutputRole;
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2InterleaveUnitLoad)
    return kRVVSegment2Field0InputRole;
  return {};
}

llvm::StringRef getRVVPlainSegment2ExpectedField1Role(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2DeinterleaveUnitStore)
    return kRVVSegment2Field1OutputRole;
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2InterleaveUnitLoad)
    return kRVVSegment2Field1InputRole;
  return {};
}

llvm::StringRef getRVVPlainSegment2ExpectedFieldSourceMemoryForm(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2InterleaveUnitLoad)
    return kRVVSegment2UnitStrideMemoryForm;
  return {};
}

llvm::StringRef getRVVPlainSegment2ExpectedFieldDestinationMemoryForm(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       Segment2DeinterleaveUnitStore)
    return kRVVSegment2UnitStrideStoreMemoryForm;
  return {};
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

llvm::Error validateRVVComputedMaskSegment2MemoryProviderFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  const plugin::rvv::RVVSelectedBodyOperationKind operation =
      description.operation;
  const plugin::rvv::RVVSelectedBodyMemoryForm expectedMemoryForm =
      getRVVSegment2MemoryExpectedMemoryForm(operation);
  if (description.memoryForm != expectedMemoryForm)
    return makeRVVTargetRouteError(
        llvm::Twine("segment2-memory target artifact consumer requires "
                    "selected typed RVV memory form '") +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(expectedMemoryForm) +
        "' before artifact export but saw '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
            description.memoryForm) +
        "'");

  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "typed compute op", description.typedComputeOpName,
          getRVVComputedMaskSegment2ExpectedTypedComputeOp(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "runtime ABI order", description.runtimeABIOrder,
          getRVVComputedMaskSegment2ExpectedRuntimeABIOrder(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "route operand binding plan", description.routeOperandBindingPlanID,
          getRVVComputedMaskSegment2ExpectedRouteOperandPlan(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "route operand binding summary",
          description.routeOperandBindingSummary,
          getRVVComputedMaskSegment2ExpectedRouteOperandSummary(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "target leaf profile", description.targetLeafProfile,
          getRVVComputedMaskSegment2ExpectedTargetLeafProfile(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "provider-supported mirror", description.providerSupportedMirror,
          getRVVComputedMaskSegment2ExpectedProviderMirror(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "required header declarations",
          description.requiredHeaderDeclarations, kRVVSegment2RequiredHeaders))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "C type mapping", description.cTypeMappingSummary,
          getRVVComputedMaskSegment2ExpectedCTypeMapping(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "computed-mask route-family plan",
          description.computedMaskMemoryRouteFamilyPlanID,
          kRVVComputedMaskMemoryRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "computed-mask producer source",
          description.computedMaskMemoryMaskProducerSource,
          kRVVComputedMaskMemoryMaskProducerSource))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "mask role", description.maskRole, kRVVComputedMaskRole))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "mask source", description.maskSource, kRVVComputedMaskSource))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "mask memory form", description.maskMemoryForm,
          kRVVComputedMaskMemoryForm))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "compare predicate", description.comparePredicateKind,
          kRVVComputedMaskComparePredicateKind))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          "computed-mask segment2-memory target artifact consumer",
          "compare intrinsic", description.compareIntrinsic))
    return error;
  if (llvm::Error error = requireRVVProviderDerivedField(
          "computed-mask segment2-memory target artifact consumer",
          "compare/source load intrinsic", description.vectorLoadIntrinsic))
    return error;
  if (description.tailPolicy != "agnostic" || description.maskPolicy != "agnostic")
    return makeRVVTargetRouteError(
        "computed-mask segment2-memory target artifact consumer requires "
        "provider-derived agnostic tail/mask policy before artifact export");

  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "inactive lane contract", description.inactiveLaneContract,
          getRVVComputedMaskSegment2ExpectedInactiveLaneContract(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "masked passthrough layout", description.maskedPassthroughLayout,
          getRVVComputedMaskSegment2ExpectedMaskedPassthroughLayout(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "segment memory layout", description.segmentMemoryLayout,
          getRVVComputedMaskSegment2ExpectedSegmentMemoryLayout(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "source memory form", description.sourceMemoryForm,
          getRVVComputedMaskSegment2ExpectedSourceMemoryForm(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "destination memory form", description.destinationMemoryForm,
          getRVVComputedMaskSegment2ExpectedDestinationMemoryForm(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "segment tuple C type", description.segmentTupleCType,
          kRVVSegment2TupleCType))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "segment load callee", description.segmentLoadIntrinsic,
          getRVVComputedMaskSegment2ExpectedSegmentLoadIntrinsic(operation)))
    return error;
  if (operation != plugin::rvv::RVVSelectedBodyOperationKind::
                       ComputedMaskSegment2LoadUnitStore)
    if (llvm::Error error = requireRVVSegment2MemoryProviderField(
            "segment store callee", description.segmentStoreIntrinsic,
            getRVVComputedMaskSegment2ExpectedSegmentStoreIntrinsic(operation)))
      return error;
  llvm::StringRef tupleCreateIntrinsic =
      operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       ComputedMaskSegment2LoadUnitStore
          ? description.segmentStoreIntrinsic
          : description.segmentFieldExtractIntrinsic;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "segment tuple create callee", tupleCreateIntrinsic,
          getRVVComputedMaskSegment2ExpectedTupleCreateIntrinsic(operation)))
    return error;
  llvm::StringRef fieldExtractIntrinsic =
      operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       ComputedMaskSegment2LoadUnitStore
          ? description.segmentFieldExtractIntrinsic
          : llvm::StringRef();
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "segment field extract callee", fieldExtractIntrinsic,
          getRVVComputedMaskSegment2ExpectedFieldExtractIntrinsic(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field0 role", description.field0Role,
          getRVVComputedMaskSegment2ExpectedField0Role(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field1 role", description.field1Role,
          getRVVComputedMaskSegment2ExpectedField1Role(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field0 name", description.field0Name,
          getRVVComputedMaskSegment2ExpectedField0Name(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field1 name", description.field1Name,
          getRVVComputedMaskSegment2ExpectedField1Name(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field0 source memory form", description.field0SourceMemoryForm,
          getRVVComputedMaskSegment2ExpectedFieldSourceMemoryForm(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field1 source memory form", description.field1SourceMemoryForm,
          getRVVComputedMaskSegment2ExpectedFieldSourceMemoryForm(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field0 destination memory form",
          description.field0DestinationMemoryForm,
          getRVVComputedMaskSegment2ExpectedFieldDestinationMemoryForm(
              operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field1 destination memory form",
          description.field1DestinationMemoryForm,
          getRVVComputedMaskSegment2ExpectedFieldDestinationMemoryForm(
              operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "segment2 update arithmetic kind",
          description.segment2UpdateArithmeticKind,
          getRVVComputedMaskSegment2ExpectedUpdateArithmeticKind(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "segment2 update arithmetic callee",
          description.segment2UpdateArithmeticIntrinsic,
          getRVVComputedMaskSegment2ExpectedUpdateArithmeticIntrinsic(
              operation)))
    return error;

  return llvm::Error::success();
}

llvm::Error validateRVVPlainSegment2MemoryProviderFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  const plugin::rvv::RVVSelectedBodyOperationKind operation =
      description.operation;
  const plugin::rvv::RVVSelectedBodyMemoryForm expectedMemoryForm =
      getRVVSegment2MemoryExpectedMemoryForm(operation);
  if (description.memoryForm != expectedMemoryForm)
    return makeRVVTargetRouteError(
        llvm::Twine("plain segment2-memory target artifact consumer requires "
                    "selected typed RVV memory form '") +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(expectedMemoryForm) +
        "' before artifact export but saw '" +
        plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
            description.memoryForm) +
        "'");

  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "typed compute op", description.typedComputeOpName,
          getRVVPlainSegment2ExpectedTypedComputeOp(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "runtime ABI order", description.runtimeABIOrder,
          getRVVPlainSegment2ExpectedRuntimeABIOrder(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "route operand binding plan", description.routeOperandBindingPlanID,
          getRVVPlainSegment2ExpectedRouteOperandPlan(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "route operand binding summary", description.routeOperandBindingSummary,
          getRVVPlainSegment2ExpectedRouteOperandSummary(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "target leaf profile", description.targetLeafProfile,
          getRVVPlainSegment2ExpectedTargetLeafProfile(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "provider-supported mirror", description.providerSupportedMirror,
          getRVVPlainSegment2ExpectedProviderMirror(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "required header declarations", description.requiredHeaderDeclarations,
          kRVVSegment2RequiredHeaders))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "C type mapping", description.cTypeMappingSummary,
          getRVVPlainSegment2ExpectedCTypeMapping(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "plain segment2 route-family plan",
          description.segment2MemoryRouteFamilyPlanID,
          kRVVSegment2MemoryRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "segment memory layout", description.segmentMemoryLayout,
          getRVVPlainSegment2ExpectedSegmentMemoryLayout(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "source memory form", description.sourceMemoryForm,
          getRVVPlainSegment2ExpectedSourceMemoryForm(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "destination memory form", description.destinationMemoryForm,
          getRVVPlainSegment2ExpectedDestinationMemoryForm(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "segment tuple C type", description.segmentTupleCType,
          kRVVSegment2TupleCType))
    return error;
  if (description.segmentCount != 2)
    return makeRVVTargetRouteError(
        "plain segment2-memory target artifact consumer requires "
        "provider-derived segment_count 2 before artifact export");
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field0 role", description.field0Role,
          getRVVPlainSegment2ExpectedField0Role(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field1 role", description.field1Role,
          getRVVPlainSegment2ExpectedField1Role(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field0 name", description.field0Name, kRVVSegment2Field0Name))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field1 name", description.field1Name, kRVVSegment2Field1Name))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field0 source memory form", description.field0SourceMemoryForm,
          getRVVPlainSegment2ExpectedFieldSourceMemoryForm(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field1 source memory form", description.field1SourceMemoryForm,
          getRVVPlainSegment2ExpectedFieldSourceMemoryForm(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field0 destination memory form",
          description.field0DestinationMemoryForm,
          getRVVPlainSegment2ExpectedFieldDestinationMemoryForm(operation)))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "field1 destination memory form",
          description.field1DestinationMemoryForm,
          getRVVPlainSegment2ExpectedFieldDestinationMemoryForm(operation)))
    return error;

  return llvm::Error::success();
}

struct RVVExpectedSegment2RuntimeABIParameter {
  llvm::StringRef cName;
  llvm::StringRef cType;
  support::RuntimeABIParameterRole role;
};

llvm::StringRef getRVVSegment2MemoryExpectedRuntimeABIOrder(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  if (isRVVComputedMaskSegment2MemoryRouteFamilyOperation(operation))
    return getRVVComputedMaskSegment2ExpectedRuntimeABIOrder(operation);
  if (isRVVPlainSegment2MemoryRouteFamilyOperation(operation))
    return getRVVPlainSegment2ExpectedRuntimeABIOrder(operation);
  return {};
}

llvm::SmallVector<RVVExpectedSegment2RuntimeABIParameter, 6>
getRVVSegment2MemoryExpectedRuntimeABIParameters(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  using support::RuntimeABIParameterRole;
  llvm::SmallVector<RVVExpectedSegment2RuntimeABIParameter, 6> expected;
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2LoadUnitStore:
    expected.emplace_back(RVVExpectedSegment2RuntimeABIParameter{
        "cmp_lhs", "const int32_t *", RuntimeABIParameterRole::LHSInputBuffer});
    expected.emplace_back(RVVExpectedSegment2RuntimeABIParameter{
        "cmp_rhs", "const int32_t *", RuntimeABIParameterRole::RHSInputBuffer});
    expected.emplace_back(RVVExpectedSegment2RuntimeABIParameter{
        "src", "const int32_t *", RuntimeABIParameterRole::SourceInputBuffer});
    expected.emplace_back(RVVExpectedSegment2RuntimeABIParameter{
        "out0", "int32_t *",
        RuntimeABIParameterRole::SegmentField0OutputBuffer});
    expected.emplace_back(RVVExpectedSegment2RuntimeABIParameter{
        "out1", "int32_t *",
        RuntimeABIParameterRole::SegmentField1OutputBuffer});
    expected.emplace_back(RVVExpectedSegment2RuntimeABIParameter{
        "n", "size_t", RuntimeABIParameterRole::RuntimeElementCount});
    return expected;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2StoreUnitLoad:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2UpdateUnitLoad:
    expected.emplace_back(RVVExpectedSegment2RuntimeABIParameter{
        "cmp_lhs", "const int32_t *", RuntimeABIParameterRole::LHSInputBuffer});
    expected.emplace_back(RVVExpectedSegment2RuntimeABIParameter{
        "cmp_rhs", "const int32_t *", RuntimeABIParameterRole::RHSInputBuffer});
    expected.emplace_back(RVVExpectedSegment2RuntimeABIParameter{
        "src0", "const int32_t *",
        RuntimeABIParameterRole::SegmentField0InputBuffer});
    expected.emplace_back(RVVExpectedSegment2RuntimeABIParameter{
        "src1", "const int32_t *",
        RuntimeABIParameterRole::SegmentField1InputBuffer});
    expected.emplace_back(RVVExpectedSegment2RuntimeABIParameter{
        "dst", "int32_t *",
        RuntimeABIParameterRole::SegmentInterleavedOutputBuffer});
    expected.emplace_back(RVVExpectedSegment2RuntimeABIParameter{
        "n", "size_t", RuntimeABIParameterRole::RuntimeElementCount});
    return expected;
  case plugin::rvv::RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore:
    expected.emplace_back(RVVExpectedSegment2RuntimeABIParameter{
        "src", "const int32_t *", RuntimeABIParameterRole::LHSInputBuffer});
    expected.emplace_back(RVVExpectedSegment2RuntimeABIParameter{
        "out0", "int32_t *",
        RuntimeABIParameterRole::SegmentField0OutputBuffer});
    expected.emplace_back(RVVExpectedSegment2RuntimeABIParameter{
        "out1", "int32_t *",
        RuntimeABIParameterRole::SegmentField1OutputBuffer});
    expected.emplace_back(RVVExpectedSegment2RuntimeABIParameter{
        "n", "size_t", RuntimeABIParameterRole::RuntimeElementCount});
    return expected;
  case plugin::rvv::RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad:
    expected.emplace_back(RVVExpectedSegment2RuntimeABIParameter{
        "src0", "const int32_t *",
        RuntimeABIParameterRole::SegmentField0InputBuffer});
    expected.emplace_back(RVVExpectedSegment2RuntimeABIParameter{
        "src1", "const int32_t *",
        RuntimeABIParameterRole::SegmentField1InputBuffer});
    expected.emplace_back(RVVExpectedSegment2RuntimeABIParameter{
        "dst", "int32_t *",
        RuntimeABIParameterRole::SegmentInterleavedOutputBuffer});
    expected.emplace_back(RVVExpectedSegment2RuntimeABIParameter{
        "n", "size_t", RuntimeABIParameterRole::RuntimeElementCount});
    return expected;
  default:
    return expected;
  }
}

llvm::Error validateRVVSegment2MemoryRuntimeABIFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  llvm::StringRef expectedOrder =
      getRVVSegment2MemoryExpectedRuntimeABIOrder(description.operation);
  if (expectedOrder.empty())
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires a segment2 runtime "
        "ABI order owner before artifact export");
  if (description.runtimeABIOrder != expectedOrder)
    return makeRVVTargetRouteError(
        llvm::Twine("segment2-memory target artifact consumer requires "
                    "provider-derived runtime ABI order '") +
        expectedOrder + "' but was '" + description.runtimeABIOrder + "'");

  llvm::SmallVector<RVVExpectedSegment2RuntimeABIParameter, 6>
      expectedParameters =
          getRVVSegment2MemoryExpectedRuntimeABIParameters(
              description.operation);
  if (description.runtimeABIParameters.size() != expectedParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("segment2-memory target artifact consumer requires ") +
        llvm::Twine(expectedParameters.size()) +
        " provider-derived runtime ABI parameters before artifact export but "
        "saw " +
        llvm::Twine(description.runtimeABIParameters.size()));

  for (std::size_t index = 0; index < expectedParameters.size(); ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    const RVVExpectedSegment2RuntimeABIParameter &expected =
        expectedParameters[index];
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
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  constexpr llvm::StringLiteral consumerLabel(
      "segment2-memory target artifact consumer");
  if (llvm::Error error = validateRVVSegment2MemoryRuntimeABIFacts(description))
    return error;
  if (description.setVLIntrinsic.empty() || description.vectorCType.empty() ||
      description.vlCType.empty() || description.emitCFullChunkVLName.empty() ||
      description.emitCLoopVLName.empty() ||
      description.emitCLoopInductionName.empty() ||
      description.field0Name.empty() || description.field1Name.empty() ||
      description.segmentTupleCType.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived setvl, vector, VL, field, tuple, and loop "
        "facts before validating route statements");

  const support::RuntimeABIParameter &runtimeNABI =
      description.runtimeABIParameters.back();
  const support::RuntimeABIParameter *runtimeElementCount =
      findRuntimeElementCountABIParameter(description);
  if (!runtimeElementCount || runtimeElementCount != &runtimeNABI)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires runtime n/AVL ABI role to match the selected segment2 ABI "
        "order before validating route statements");

  if (route.getCallOpaqueSteps().size() != 1)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires exactly one provider-built pre-loop setvl statement before "
        "artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          preLoopSetVL, consumerLabel, "pre-loop setvl",
          description.setVLIntrinsic, {{runtimeNABI.cName, runtimeNABI.cType}},
          description.emitCFullChunkVLName, description.vlCType))
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
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires provider-built "
        "loop bounds and step to mirror runtime AVL/VL route facts");
  if (loop.upperBound.expression != runtimeNABI.cName ||
      loop.upperBound.cType != runtimeNABI.cType)
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires provider-built "
        "loop upper bound to use the runtime n/AVL ABI parameter");

  std::size_t expectedLoopBodyStepCount = 0;
  switch (description.operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2LoadUnitStore:
    expectedLoopBodyStepCount = 12;
    break;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2StoreUnitLoad:
    expectedLoopBodyStepCount = 8;
    break;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2UpdateUnitLoad:
    expectedLoopBodyStepCount = 9;
    break;
  case plugin::rvv::RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore:
    expectedLoopBodyStepCount = 6;
    break;
  case plugin::rvv::RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad:
    expectedLoopBodyStepCount = 5;
    break;
  default:
    llvm_unreachable("validated non-segment2 operation as segment2-memory");
  }
  if (loop.bodySteps.size() != expectedLoopBodyStepCount)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires exact provider-built loop statement count " +
        llvm::Twine(expectedLoopBodyStepCount) +
        " for the selected segment2-memory family before artifact export");

  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeNABI.cName) + " - " +
       description.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[0], consumerLabel, "loop setvl",
          description.setVLIntrinsic,
          {{expectedRemainingAVL, description.vlCType}},
          description.emitCLoopVLName, description.vlCType))
    return error;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "segment2-memory target artifact consumer requires loop statements "
          "to carry selected typed RVV source provenance");

  auto pointerAtInduction = [&](const support::RuntimeABIParameter &abi) {
    return (llvm::StringRef(abi.cName) + " + " +
            description.emitCLoopInductionName)
        .str();
  };
  auto interleavedPointerAtInduction =
      [&](const support::RuntimeABIParameter &abi) {
        return (llvm::StringRef(abi.cName) + " + (" +
                description.emitCLoopInductionName + " * 2)")
            .str();
      };
  auto validateVectorLoad =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          llvm::StringRef stepLabel, const support::RuntimeABIParameter &abi,
          llvm::StringRef resultName) -> llvm::Error {
    return validateRVVProviderBuiltRouteStep(
        step, consumerLabel, stepLabel, description.vectorLoadIntrinsic,
        {{pointerAtInduction(abi), abi.cType},
         {description.emitCLoopVLName, description.vlCType}},
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
         {description.emitCLoopVLName, description.vlCType}});
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
         {description.emitCLoopVLName, description.vlCType}},
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
             {description.emitCLoopVLName, description.vlCType}},
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
             {description.emitCLoopVLName, description.vlCType}}))
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
             {description.emitCLoopVLName, description.vlCType}},
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
             {description.emitCLoopVLName, description.vlCType}}))
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
             {description.emitCLoopVLName, description.vlCType}},
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
             {description.emitCLoopVLName, description.vlCType}}))
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
  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("segment2-memory target artifact consumer requires "
                    "rebuilt provider route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");
  if (description.providerSupportedMirror.empty() ||
      description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty() ||
      description.runtimeControlPlanID.empty() ||
      description.segmentMemoryLayout.empty() ||
      description.sourceMemoryForm.empty() ||
      description.destinationMemoryForm.empty() ||
      description.segmentCount != 2)
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires provider-derived "
        "support, binding, runtime control, segment layout, memory form, and "
        "segment-count facts before artifact export");
  if (llvm::Error error = validateRVVSegment2MemoryRuntimeABIFacts(description))
    return error;

  if (isRVVComputedMaskSegment2MemoryRouteFamilyOperation(
          description.operation)) {
    if (description.computedMaskMemoryRouteFamilyPlanID.empty() ||
        description.computedMaskMemoryMaskProducerSource.empty() ||
        description.maskRole.empty() || description.maskSource.empty() ||
        description.maskMemoryForm.empty() ||
        description.compareIntrinsic.empty())
      return makeRVVTargetRouteError(
          "computed-mask segment2-memory target artifact consumer requires "
          "provider-derived computed-mask family, producer, role, source, "
          "memory-form, and compare facts before artifact export");
    if (llvm::Error error =
            validateComputedMaskSegment2HeaderBindingSummary(description))
      return error;
    if (llvm::Error error =
            validateRVVComputedMaskSegment2MemoryProviderFacts(description))
      return error;
    if (!description.segment2MemoryRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          "computed-mask segment2-memory target artifact consumer rejects "
          "stale plain segment2 route-family facts");
  } else if (isRVVPlainSegment2MemoryRouteFamilyOperation(
                 description.operation)) {
    if (description.segment2MemoryRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          "plain segment2-memory target artifact consumer requires a "
          "provider-derived segment2 route-family plan mirror before artifact "
          "export");
    if (llvm::Error error =
            validatePlainSegment2DeinterleaveHeaderBindingSummary(description))
      return error;
    if (llvm::Error error =
            validatePlainSegment2InterleaveHeaderBindingSummary(description))
      return error;
    if (llvm::Error error =
            validateRVVPlainSegment2MemoryProviderFacts(description))
      return error;
    if (!description.computedMaskMemoryRouteFamilyPlanID.empty() ||
        !description.computedMaskMemoryMaskProducerSource.empty() ||
        !description.maskRole.empty() || !description.maskSource.empty() ||
        !description.maskMemoryForm.empty())
      return makeRVVTargetRouteError(
          "plain segment2-memory target artifact consumer rejects stale "
          "computed-mask route-family facts");
  } else {
    llvm_unreachable("validated non-segment2 operation as segment2-memory");
  }

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
          validateRVVSegment2MemoryRouteHeaders(route, description))
    return error;
  if (llvm::Error error =
          validateRVVSegment2MemoryRouteTypeMappings(route, description))
    return error;
  if (llvm::Error error =
          validateRVVSegment2MemoryRouteABIMappings(route, description))
    return error;
  return validateRVVSegment2MemoryRouteStatementPlan(route, description);
}

llvm::Error validateRVVSegment2MemoryTargetArtifactProviderFactsImpl(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVSegment2MemoryRoutePayloadFacts(context.route,
                                                    context.description);
}

llvm::Error requireEmptySegment2MemoryStaleMirror(
    const TargetArtifactCandidate &candidate, llvm::StringRef key,
    llvm::StringRef label) {
  return requireCandidateMetadataMirror(candidate, key, "", label);
}

llvm::Error validateRVVPlainSegment2MemoryTargetArtifactCandidateMirrors(
    const TargetArtifactCandidate &candidate,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.segment_tuple_c_type",
          description.segmentTupleCType,
          "selected typed RVV plain segment2 tuple C type"))
    return error;

  if (description.operation ==
      plugin::rvv::RVVSelectedBodyOperationKind::
          Segment2DeinterleaveUnitStore) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.segment_load_intrinsic",
            description.segmentLoadIntrinsic,
            "selected typed RVV plain segment2 load callee"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.segment_store_intrinsic", "",
            "selected typed RVV plain segment2 store callee"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.segment_field_extract_intrinsic",
            description.segmentFieldExtractIntrinsic,
            "selected typed RVV plain segment2 field extract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.segment_tuple_create_intrinsic", "",
            "selected typed RVV plain segment2 tuple create"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.field0_destination_memory_form",
            description.field0DestinationMemoryForm,
            "selected typed RVV plain segment2 field0 destination memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.field1_destination_memory_form",
            description.field1DestinationMemoryForm,
            "selected typed RVV plain segment2 field1 destination memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.field0_source_memory_form", "",
            "selected typed RVV plain segment2 field0 source memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.field1_source_memory_form", "",
            "selected typed RVV plain segment2 field1 source memory form"))
      return error;
  } else if (description.operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 Segment2InterleaveUnitLoad) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.segment_load_intrinsic", "",
            "selected typed RVV plain segment2 load callee"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.segment_store_intrinsic",
            description.segmentStoreIntrinsic,
            "selected typed RVV plain segment2 store callee"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.segment_field_extract_intrinsic", "",
            "selected typed RVV plain segment2 field extract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.segment_tuple_create_intrinsic",
            description.segmentFieldExtractIntrinsic,
            "selected typed RVV plain segment2 tuple create"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.field0_source_memory_form",
            description.field0SourceMemoryForm,
            "selected typed RVV plain segment2 field0 source memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.field1_source_memory_form",
            description.field1SourceMemoryForm,
            "selected typed RVV plain segment2 field1 source memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.field0_destination_memory_form", "",
            "selected typed RVV plain segment2 field0 destination memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.field1_destination_memory_form", "",
            "selected typed RVV plain segment2 field1 destination memory form"))
      return error;
  } else {
    llvm_unreachable("validated non-plain segment2 candidate as plain segment2");
  }

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.field0_role", description.field0Role,
          "selected typed RVV plain segment2 field0 role"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.field1_role", description.field1Role,
          "selected typed RVV plain segment2 field1 role"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.field0_name", description.field0Name,
          "selected typed RVV plain segment2 field0 binding"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.field1_name", description.field1Name,
          "selected typed RVV plain segment2 field1 binding"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.segment2_update_arithmetic_kind", "",
          "selected typed RVV computed-mask segment2 update arithmetic kind"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.segment2_update_arithmetic_intrinsic", "",
          "selected typed RVV computed-mask segment2 update arithmetic callee"))
    return error;

  return llvm::Error::success();
}

llvm::Error validateRVVSegment2MemoryTargetArtifactCandidateMirrorsImpl(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description =
      context.description;

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_plan",
          description.routeOperandBindingPlanID,
          "selected typed RVV segment2-memory binding plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_operands",
          description.routeOperandBindingSummary,
          "selected typed RVV segment2-memory binding summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.provider_supported_mirror",
          description.providerSupportedMirror,
          "selected typed RVV segment2-memory provider support"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.target_leaf_profile",
          description.targetLeafProfile,
          "selected typed RVV segment2-memory target leaf profile"))
    return error;

  if (isRVVPlainSegment2MemoryRouteFamilyOperation(description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.segment2_memory_route_family_plan",
            description.segment2MemoryRouteFamilyPlanID,
            "selected typed RVV segment2 route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_memory_route_family_plan", "",
            "selected typed RVV computed-mask segment2 route-family plan"))
      return error;
  } else if (isRVVComputedMaskSegment2MemoryRouteFamilyOperation(
                 description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.segment2_memory_route_family_plan", "",
            "selected typed RVV segment2 route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_memory_route_family_plan",
            description.computedMaskMemoryRouteFamilyPlanID,
            "selected typed RVV computed-mask segment2 route-family plan"))
      return error;
  }

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.memory_form",
          plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
              description.memoryForm),
          "selected typed RVV segment2-memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_control_plan",
          description.runtimeControlPlanID,
          "selected typed RVV segment2-memory runtime AVL/VL control plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order",
          description.runtimeABIOrder,
          "selected typed RVV segment2-memory runtime ABI order"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.required_header_declarations",
          description.requiredHeaderDeclarations,
          "selected typed RVV segment2-memory route header requirements"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.c_type_mapping",
          description.cTypeMappingSummary,
          "selected typed RVV segment2-memory route type mapping summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.segment_memory_layout",
          description.segmentMemoryLayout,
          "selected typed RVV segment2 memory layout"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_memory_form",
          description.sourceMemoryForm,
          "selected typed RVV segment2 source memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.destination_memory_form",
          description.destinationMemoryForm,
          "selected typed RVV segment2 destination memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.segment_count",
          llvm::Twine(description.segmentCount).str(),
          "selected typed RVV segment2 count"))
    return error;

  if (isRVVComputedMaskSegment2MemoryRouteFamilyOperation(
          description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.compare_predicate_kind",
            description.comparePredicateKind,
            "selected typed RVV computed-mask segment2 compare predicate"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.masked_memory_layout",
            description.segmentMemoryLayout,
            "selected typed RVV computed-mask segment2 masked memory layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_memory_mask_producer_source",
            description.computedMaskMemoryMaskProducerSource,
            "selected typed RVV computed-mask segment2 producer source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_role", description.maskRole,
            "selected typed RVV computed-mask segment2 mask role"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_source", description.maskSource,
            "selected typed RVV computed-mask segment2 mask source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_memory_form", description.maskMemoryForm,
            "selected typed RVV computed-mask segment2 mask memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.inactive_lane_contract",
            description.inactiveLaneContract,
            "selected typed RVV computed-mask segment2 inactive lane contract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.masked_passthrough_layout",
            description.maskedPassthroughLayout,
            "selected typed RVV computed-mask segment2 passthrough layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.segment_tuple_c_type",
            description.segmentTupleCType,
            "selected typed RVV computed-mask segment2 tuple C type"))
      return error;
    if (description.operation ==
        plugin::rvv::RVVSelectedBodyOperationKind::
            ComputedMaskSegment2LoadUnitStore) {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.segment_load_intrinsic",
              description.segmentLoadIntrinsic,
              "selected typed RVV computed-mask segment2 load callee"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.segment_store_intrinsic", "",
              "selected typed RVV computed-mask segment2 store callee"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.segment_field_extract_intrinsic",
              description.segmentFieldExtractIntrinsic,
              "selected typed RVV computed-mask segment2 field extract"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.segment_tuple_create_intrinsic",
              description.segmentStoreIntrinsic,
              "selected typed RVV computed-mask segment2 tuple create"))
        return error;
    } else {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.segment_load_intrinsic", "",
              "selected typed RVV computed-mask segment2 load callee"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.segment_store_intrinsic",
              description.segmentStoreIntrinsic,
              "selected typed RVV computed-mask segment2 masked store callee"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.segment_tuple_create_intrinsic",
              description.segmentFieldExtractIntrinsic,
              "selected typed RVV computed-mask segment2 tuple create"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.segment_field_extract_intrinsic", "",
              "selected typed RVV computed-mask segment2 field extract"))
        return error;
    }
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.field0_role", description.field0Role,
            "selected typed RVV computed-mask segment2 field0 role"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.field1_role", description.field1Role,
            "selected typed RVV computed-mask segment2 field1 role"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.field0_name", description.field0Name,
            "selected typed RVV computed-mask segment2 field0 binding"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.field1_name", description.field1Name,
            "selected typed RVV computed-mask segment2 field1 binding"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.field0_source_memory_form",
            description.field0SourceMemoryForm,
            "selected typed RVV computed-mask segment2 field0 source memory "
            "form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.field1_source_memory_form",
            description.field1SourceMemoryForm,
            "selected typed RVV computed-mask segment2 field1 source memory "
            "form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.field0_destination_memory_form",
            description.field0DestinationMemoryForm,
            "selected typed RVV computed-mask segment2 field0 destination "
            "memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.field1_destination_memory_form",
            description.field1DestinationMemoryForm,
            "selected typed RVV computed-mask segment2 field1 destination "
            "memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.segment2_update_arithmetic_kind",
            description.segment2UpdateArithmeticKind,
            "selected typed RVV computed-mask segment2 update arithmetic "
            "kind"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.segment2_update_arithmetic_intrinsic",
            description.segment2UpdateArithmeticIntrinsic,
            "selected typed RVV computed-mask segment2 update arithmetic "
            "callee"))
      return error;
  } else {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_memory_mask_producer_source",
            "", "selected typed RVV computed-mask segment2 producer source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_role", "",
            "selected typed RVV computed-mask segment2 mask role"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_source", "",
            "selected typed RVV computed-mask segment2 mask source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_memory_form", "",
            "selected typed RVV computed-mask segment2 mask memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.compare_predicate_kind", "",
            "selected typed RVV computed-mask segment2 compare predicate"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.masked_memory_layout", "",
            "selected typed RVV computed-mask segment2 masked memory layout"))
      return error;
    if (llvm::Error error =
            validateRVVPlainSegment2MemoryTargetArtifactCandidateMirrors(
                candidate, description))
      return error;
  }

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.mask_tail_policy_route_family_plan",
          description.maskTailPolicyRouteFamilyPlanID,
          "selected typed RVV segment2-memory mask/tail policy plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.mask_tail_policy_owner",
          description.maskTailPolicyOwner,
          "selected typed RVV segment2-memory mask/tail policy owner"))
    return error;

  constexpr llvm::StringLiteral staleMirrors[] = {
      "tcrv_rvv.elementwise_arithmetic_route_family_plan",
      "tcrv_rvv.scalar_broadcast_elementwise_route_family_plan",
      "tcrv_rvv.widening_conversion_route_family_plan",
      "tcrv_rvv.plain_compare_select_route_family_plan",
      "tcrv_rvv.computed_mask_select_route_family_plan",
      "tcrv_rvv.plain_macc_route_family_plan",
      "tcrv_rvv.scalar_broadcast_macc_route_family_plan",
      "tcrv_rvv.accumulation_route_family_plan",
      "tcrv_rvv.standalone_reduction_route_family_plan",
      "tcrv_rvv.contraction_route_family_plan",
      "tcrv_rvv.base_memory_movement_route_family_plan",
      "tcrv_rvv.widening_macc_relation",
      "tcrv_rvv.widening_dot_relation"};
  for (llvm::StringRef key : staleMirrors)
    if (llvm::Error error = requireEmptySegment2MemoryStaleMirror(
            candidate, key,
            "selected typed RVV non-segment2 route-family mirror"))
      return error;

  return llvm::Error::success();
}

bool isRVVSegment2LoadTargetArtifactFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return description.operation == plugin::rvv::RVVSelectedBodyOperationKind::
                                      ComputedMaskSegment2LoadUnitStore;
}

bool isRVVSegment2StoreTargetArtifactFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return description.operation == plugin::rvv::RVVSelectedBodyOperationKind::
                                      ComputedMaskSegment2StoreUnitLoad;
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
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.elementTypeName == "i16")
    return "int16_t";
  if (description.elementTypeName == "i32")
    return "int32_t";
  if (description.elementTypeName == "i64")
    return "int64_t";
  return {};
}

llvm::Error validateRVVElementwiseArithmeticRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires "
        "provider-derived required_header_declarations before accepting the "
        "route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires at least "
        "one provider route header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "elementwise arithmetic target artifact consumer saw an empty "
          "provider route header declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("elementwise arithmetic target artifact consumer "
                      "requires rebuilt provider route header '") +
          trimmed + "' before artifact export");
  }
  return llvm::Error::success();
}

llvm::Error validateRVVElementwiseArithmeticRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.elementTypeName.empty() || description.sew == 0 ||
      description.lmul.empty() || description.vlCType.empty() ||
      description.vectorTypeName.empty() || description.vectorCType.empty() ||
      description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires "
        "provider-derived dtype, SEW, LMUL, VL, vector, and C type mapping "
        "facts before artifact export");

  if (getRVVElementwiseArithmeticSignedCType(description).empty())
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires a "
        "provider-derived signed C type for the typed RVV element type");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("elementwise arithmetic target artifact consumer "
                    "requires rebuilt provider route type mapping "
                    "'!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("elementwise arithmetic target artifact consumer "
                    "requires rebuilt provider route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");

  if (isRVVMaskedElementwiseArithmeticRouteFamilyOperation(
          description.operation)) {
    if (description.maskTypeName.empty() || description.maskCType.empty())
      return makeRVVTargetRouteError(
          "masked elementwise arithmetic target artifact consumer requires "
          "provider-derived mask type mapping facts before artifact export");
    if (!routeHasTypeMapping(route, description.maskTypeName,
                             description.maskCType))
      return makeRVVTargetRouteError(
          llvm::Twine("masked elementwise arithmetic target artifact consumer "
                      "requires rebuilt provider route type mapping '") +
          description.maskTypeName + "' -> '" + description.maskCType + "'");
  } else if (!description.maskTypeName.empty() ||
             !description.maskCType.empty()) {
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer rejects stale mask "
        "type mapping facts for non-masked elementwise routes");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVElementwiseArithmeticRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("elementwise arithmetic target artifact consumer "
                    "requires rebuilt provider route ABI mapping count ") +
        llvm::Twine(description.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  llvm::StringRef signedCType =
      getRVVElementwiseArithmeticSignedCType(description);
  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        description.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine("elementwise arithmetic target artifact consumer "
                      "requires rebuilt provider route ABI mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("elementwise arithmetic target artifact consumer "
                      "requires rebuilt provider route ABI mapping[") +
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
      if (expected.cType != description.vlCType)
        return makeRVVTargetRouteError(
            "elementwise arithmetic target artifact consumer requires runtime "
            "element-count ABI C type to mirror provider VL C type");
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
          "elementwise arithmetic target artifact consumer rejects ABI roles "
          "outside lhs, rhs, rhs scalar, out, runtime n, and optional stride "
          "roles");
    }
  }

  return llvm::Error::success();
}

llvm::Error validateRVVElementwiseArithmeticRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  const support::RuntimeABIParameter *runtimeN =
      findRuntimeElementCountABIParameter(description);
  if (!runtimeN)
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires a runtime "
        "n/AVL ABI parameter before artifact export");

  if (route.getCallOpaqueSteps().size() != 1)
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires exactly "
        "one provider-built pre-loop setvl statement before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      preLoopSetVL.operands.size() != 1 ||
      preLoopSetVL.operands.front().expression != runtimeN->cName ||
      preLoopSetVL.operands.front().cType != runtimeN->cType ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires rebuilt "
        "provider route pre-loop setvl statement to use runtime n/AVL and "
        "define the full-chunk VL");
  if (!routeStepSourceIsSelectedRVVBody(preLoopSetVL))
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires pre-loop "
        "setvl provenance from the selected typed RVV body");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires exactly "
        "one provider-built runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.upperBound.expression != runtimeN->cName ||
      loop.upperBound.cType != runtimeN->cType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires "
        "provider-built loop bounds and step to mirror runtime n/AVL/VL facts");
  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeN->cName) + " - " +
       description.emitCLoopInductionName)
          .str();
  if (loop.bodySteps.empty() ||
      loop.bodySteps.front().callee != description.setVLIntrinsic ||
      loop.bodySteps.front().operands.size() != 1 ||
      loop.bodySteps.front().operands.front().expression !=
          expectedRemainingAVL ||
      loop.bodySteps.front().operands.front().cType != description.vlCType ||
      !stepHasResult(loop.bodySteps.front(), description.emitCLoopVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires "
        "provider-built loop setvl to derive per-iteration VL from remaining "
        "runtime AVL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "elementwise arithmetic target artifact consumer requires loop "
          "statements to carry selected typed RVV source provenance");

  const bool isMasked = isRVVMaskedElementwiseArithmeticRouteFamilyOperation(
      description.operation);
  const bool isStrided = isRVVStridedElementwiseArithmeticRouteFamilyOperation(
      description.operation);
  const bool isScalarBroadcast =
      isRVVScalarBroadcastElementwiseRouteFamilyOperation(
          description.operation);
  const bool isRHSBroadcastLoad =
      isRVVRHSBroadcastLoadElementwiseArithmeticRouteFamilyDescription(
          description);

  if (isStrided) {
    if (!routeLoopContainsCallee(loop, description.stridedLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.intrinsic) ||
        !routeLoopContainsCallee(loop, description.stridedStoreIntrinsic))
      return makeRVVTargetRouteError(
          "strided elementwise arithmetic target artifact consumer requires "
          "provider-built strided loads, elementwise compute, and strided "
          "store statements before artifact export");
  } else if (isScalarBroadcast || isRHSBroadcastLoad) {
    if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.rhsBroadcastIntrinsic) ||
        !routeLoopContainsCallee(loop, description.intrinsic) ||
        !routeLoopContainsCallee(loop, description.storeIntrinsic))
      return makeRVVTargetRouteError(
          "broadcast elementwise target artifact consumer requires "
          "provider-built vector load, RHS broadcast, elementwise compute, "
          "and store statements before artifact export");
  } else {
    if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.intrinsic) ||
        !routeLoopContainsCallee(loop, description.storeIntrinsic))
      return makeRVVTargetRouteError(
          "elementwise arithmetic target artifact consumer requires "
          "provider-built vector loads, elementwise compute, and store "
          "statements before artifact export");
  }
  if (isMasked) {
    if (!routeLoopContainsCallee(loop, description.compareIntrinsic) ||
        !routeLoopContainsCallee(loop, description.maskedMergeIntrinsic))
      return makeRVVTargetRouteError(
          "masked elementwise arithmetic target artifact consumer requires "
          "provider-built compare and masked merge statements before artifact "
          "export");
  } else if (!description.compareIntrinsic.empty() ||
             !description.maskedMergeIntrinsic.empty()) {
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer rejects stale "
        "masked compare/merge leaves for non-masked elementwise routes");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVElementwiseArithmeticRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("elementwise arithmetic target artifact consumer requires "
                    "rebuilt provider route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");
  if (description.providerSupportedMirror.empty() ||
      description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires "
        "provider-supported and operand-binding mirror facts before artifact "
        "export");

  const bool isScalarBroadcast =
      isRVVScalarBroadcastElementwiseRouteFamilyOperation(
          description.operation);
  const bool isStrided = isRVVStridedElementwiseArithmeticRouteFamilyOperation(
      description.operation);
  const bool isRHSBroadcastLoad =
      isRVVRHSBroadcastLoadElementwiseArithmeticRouteFamilyDescription(
          description);
  llvm::StringRef expectedRuntimeABIOrder =
      isStrided ? llvm::StringRef(
                      "lhs,rhs,out,n,lhs_stride,rhs_stride,out_stride")
      : isScalarBroadcast
          ? llvm::StringRef("lhs,rhs_scalar,out,n")
          : llvm::StringRef("lhs,rhs,out,n");
  if (description.runtimeControlPlanID.empty() ||
      description.runtimeABIOrder != expectedRuntimeABIOrder ||
      description.elementTypeName.empty() || description.sew == 0 ||
      description.lmul.empty() || description.tailPolicy.empty() ||
      description.maskPolicy.empty() || description.configContractID.empty() ||
      description.setVLIntrinsic.empty() || description.intrinsic.empty() ||
      description.storeIntrinsic.empty() || description.resultName.empty())
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires "
        "provider-derived runtime AVL/VL, dtype, SEW, LMUL, policy, config, "
        "intrinsic, and result facts before artifact export");
  if (description.vectorLoadIntrinsic.empty() && !isStrided)
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires "
        "provider-derived vector load facts before artifact export");
  if (isScalarBroadcast) {
    if (description.scalarBroadcastElementwiseRouteFamilyPlanID.empty() ||
        description.rhsBroadcastIntrinsic.empty())
      return makeRVVTargetRouteError(
          "scalar-broadcast elementwise target artifact consumer requires "
          "provider-derived scalar-broadcast plan and RHS broadcast facts "
          "before artifact export");
    if (!description.elementwiseArithmeticRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          "scalar-broadcast elementwise target artifact consumer rejects stale "
          "plain elementwise arithmetic route-family facts");
  } else {
    if (description.elementwiseArithmeticRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          "elementwise arithmetic target artifact consumer requires "
          "provider-derived elementwise route-family facts before artifact "
          "export");
    if (!description.scalarBroadcastElementwiseRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          "elementwise arithmetic target artifact consumer rejects stale "
          "scalar-broadcast route facts");
    if (isRHSBroadcastLoad) {
      if (description.rhsBroadcastIntrinsic.empty())
        return makeRVVTargetRouteError(
            "broadcast-load elementwise target artifact consumer requires "
            "provider-derived RHS broadcast facts before artifact export");
    } else if (!description.rhsBroadcastIntrinsic.empty()) {
      return makeRVVTargetRouteError(
          "elementwise arithmetic target artifact consumer rejects stale RHS "
          "broadcast route facts");
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
      description.operation);
  if (isMasked) {
    if (description.compareIntrinsic.empty() ||
        description.maskedMergeIntrinsic.empty() ||
        description.maskName.empty() || description.maskRole.empty() ||
        description.maskSource.empty() ||
        description.inactiveLaneContract.empty() ||
        description.maskedPassthroughLayout.empty())
      return makeRVVTargetRouteError(
          "masked elementwise arithmetic target artifact consumer requires "
          "provider-derived mask, compare, merge, inactive-lane, and "
          "passthrough facts before artifact export");
  } else if (!description.maskName.empty() || !description.maskRole.empty() ||
             !description.maskSource.empty() ||
             !description.inactiveLaneContract.empty() ||
             !description.maskedPassthroughLayout.empty()) {
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer rejects stale mask "
        "metadata for non-masked elementwise routes");
  }
  if (isStrided) {
    if (description.stridedLoadIntrinsic.empty() ||
        description.stridedStoreIntrinsic.empty() ||
        description.stridedMemoryLayout.empty() ||
        description.lhsStrideSource.empty() ||
        description.rhsStrideSource.empty() ||
        description.outStrideSource.empty())
      return makeRVVTargetRouteError(
          "strided elementwise arithmetic target artifact consumer requires "
          "provider-derived strided load/store and stride-source facts before "
          "artifact export");
  } else if (!description.stridedLoadIntrinsic.empty() ||
             !description.stridedStoreIntrinsic.empty() ||
             !description.stridedMemoryLayout.empty() ||
             !description.lhsStrideSource.empty() ||
             !description.rhsStrideSource.empty() ||
             !description.outStrideSource.empty()) {
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer rejects stale "
        "strided metadata for non-strided elementwise routes");
  }

  if (llvm::Error error =
          validateRVVElementwiseArithmeticRouteHeaders(route, description))
    return error;
  if (llvm::Error error =
          validateRVVElementwiseArithmeticRouteTypeMappings(route,
                                                            description))
    return error;
  if (llvm::Error error =
          validateRVVElementwiseArithmeticRouteABIMappings(route, description))
    return error;
  return validateRVVElementwiseArithmeticRouteStatementPlan(route, description);
}

llvm::Error validateRVVElementwiseArithmeticTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVElementwiseArithmeticRoutePayloadFacts(
      context.route, context.description);
}

llvm::Error requireEmptyElementwiseStaleRouteFamilyMirror(
    const TargetArtifactCandidate &candidate, llvm::StringRef key,
    llvm::StringRef label) {
  return requireCandidateMetadataMirror(candidate, key, "", label);
}

llvm::Error validateRVVElementwiseArithmeticTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description =
      context.description;
  const bool isScalarBroadcast =
      isRVVScalarBroadcastElementwiseRouteFamilyOperation(
          description.operation);
  const bool isMasked = isRVVMaskedElementwiseArithmeticRouteFamilyOperation(
      description.operation);
  const bool isStrided = isRVVStridedElementwiseArithmeticRouteFamilyOperation(
      description.operation);

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_plan",
          description.routeOperandBindingPlanID,
          "selected typed RVV elementwise binding plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_operands",
          description.routeOperandBindingSummary,
          "selected typed RVV elementwise binding summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.provider_supported_mirror",
          description.providerSupportedMirror,
          "selected typed RVV elementwise provider support"))
    return error;

  if (isScalarBroadcast) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.scalar_broadcast_elementwise_route_family_plan",
            description.scalarBroadcastElementwiseRouteFamilyPlanID,
            "selected typed RVV scalar-broadcast elementwise route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.elementwise_arithmetic_route_family_plan", "",
            "selected typed RVV elementwise arithmetic route-family plan"))
      return error;
  } else {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.elementwise_arithmetic_route_family_plan",
            description.elementwiseArithmeticRouteFamilyPlanID,
            "selected typed RVV elementwise arithmetic route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate,
            "tcrv_rvv.scalar_broadcast_elementwise_route_family_plan", "",
            "selected typed RVV scalar-broadcast elementwise route-family plan"))
      return error;
  }

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.memory_form",
          plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
              description.memoryForm),
          "selected typed RVV elementwise memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.target_leaf_profile",
          description.targetLeafProfile,
          "selected typed RVV elementwise target leaf profile"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_control_plan",
          description.runtimeControlPlanID,
          "selected typed RVV elementwise runtime AVL/VL control plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order", description.runtimeABIOrder,
          "selected typed RVV elementwise runtime ABI order"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.required_header_declarations",
          description.requiredHeaderDeclarations,
          "selected typed RVV elementwise route header requirements"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.c_type_mapping",
          description.cTypeMappingSummary,
          "selected typed RVV elementwise route type mapping summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_memory_form",
          description.sourceMemoryForm,
          "selected typed RVV elementwise source memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.destination_memory_form",
          description.destinationMemoryForm,
          "selected typed RVV elementwise destination memory form"))
    return error;

  if (isMasked) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_role", description.maskRole,
            "selected typed RVV masked elementwise mask role"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_source", description.maskSource,
            "selected typed RVV masked elementwise mask source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.inactive_lane_contract",
            description.inactiveLaneContract,
            "selected typed RVV masked elementwise inactive lane contract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.masked_passthrough_layout",
            description.maskedPassthroughLayout,
            "selected typed RVV masked elementwise passthrough layout"))
      return error;
  } else {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_role", "",
            "selected typed RVV masked elementwise mask role"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_source", "",
            "selected typed RVV masked elementwise mask source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.inactive_lane_contract", "",
            "selected typed RVV masked elementwise inactive lane contract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.masked_passthrough_layout", "",
            "selected typed RVV masked elementwise passthrough layout"))
      return error;
  }
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.mask_memory_form", "",
          "selected typed RVV masked elementwise mask memory form"))
    return error;

  if (isStrided) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.strided_memory_layout",
            description.stridedMemoryLayout,
            "selected typed RVV strided elementwise memory layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.lhs_stride_source",
            description.lhsStrideSource,
            "selected typed RVV strided elementwise lhs stride source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.rhs_stride_source",
            description.rhsStrideSource,
            "selected typed RVV strided elementwise rhs stride source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.out_stride_source",
            description.outStrideSource,
            "selected typed RVV strided elementwise output stride source"))
      return error;
  } else {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.strided_memory_layout", "",
            "selected typed RVV strided elementwise memory layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.lhs_stride_source", "",
            "selected typed RVV strided elementwise lhs stride source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.rhs_stride_source", "",
            "selected typed RVV strided elementwise rhs stride source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.out_stride_source", "",
            "selected typed RVV strided elementwise output stride source"))
      return error;
  }

  constexpr llvm::StringLiteral staleRouteFamilies[] = {
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
    if (llvm::Error error =
            requireEmptyElementwiseStaleRouteFamilyMirror(
                candidate, key,
                "selected typed RVV non-elementwise route-family plan"))
      return error;

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_sew", "",
          "selected typed RVV widening conversion source SEW"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_lmul", "",
          "selected typed RVV widening conversion source LMUL"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.dest_sew", "",
          "selected typed RVV widening conversion destination SEW"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.dest_lmul", "",
          "selected typed RVV widening conversion destination LMUL"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.conversion_relation", "",
          "selected typed RVV widening conversion relation"))
    return error;

  return llvm::Error::success();
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

constexpr llvm::StringLiteral kRVVVectorReductionRuntimeABIOrder(
    "lhs,rhs,out,n");
constexpr llvm::StringLiteral kRVVVectorReductionAccumulatorLayout(
    "rhs-vector-seed-lane0-per-vl-chunk");
constexpr llvm::StringLiteral kRVVVectorReductionResultLayout(
    "store-reduction-lane0-to-output-chunk-base");
constexpr llvm::StringLiteral kRVVVectorReductionStoreVL("1");

llvm::Error validateRVVVectorReductionRuntimeABIFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.runtimeABIOrder != kRVVVectorReductionRuntimeABIOrder)
    return makeRVVTargetRouteError(
        llvm::Twine("vector reduction target artifact consumer requires "
                    "provider-derived runtime ABI order '") +
        kRVVVectorReductionRuntimeABIOrder + "' but was '" +
        description.runtimeABIOrder + "'");
  if (description.runtimeABIParameters.size() != 4)
    return makeRVVTargetRouteError(
        "vector reduction target artifact consumer requires provider-derived "
        "runtime ABI parameters for lhs, rhs seed/accumulator, out, and n "
        "before artifact export");

  struct ExpectedRuntimeABIParameterRole {
    llvm::StringRef cName;
    support::RuntimeABIParameterRole role;
  };
  const ExpectedRuntimeABIParameterRole expectedRoles[] = {
      {"lhs", support::RuntimeABIParameterRole::LHSInputBuffer},
      {"rhs", support::RuntimeABIParameterRole::RHSInputBuffer},
      {"out", support::RuntimeABIParameterRole::OutputBuffer},
      {"n", support::RuntimeABIParameterRole::RuntimeElementCount},
  };
  constexpr size_t expectedRoleCount =
      sizeof(expectedRoles) / sizeof(expectedRoles[0]);
  for (size_t index = 0; index < expectedRoleCount; ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    if (actual.cName != expectedRoles[index].cName ||
        actual.role != expectedRoles[index].role)
      return makeRVVTargetRouteError(
          llvm::Twine("vector reduction target artifact consumer requires "
                      "provider-derived runtime ABI parameter ") +
          std::to_string(index) + " to bind " + expectedRoles[index].cName +
          " as " +
          support::stringifyRuntimeABIParameterRole(expectedRoles[index].role) +
          " before artifact export");
  }
  return llvm::Error::success();
}

llvm::Error validateRVVVectorReductionRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.vlCType.empty() || description.vectorTypeName.empty() ||
      description.vectorCType.empty())
    return makeRVVTargetRouteError(
        "vector reduction target artifact consumer requires provider-derived "
        "VL and vector type facts before artifact export");
  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("vector reduction target artifact consumer requires "
                    "rebuilt provider route type mapping '!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("vector reduction target artifact consumer requires "
                    "rebuilt provider route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");
  return llvm::Error::success();
}

llvm::Error validateRVVVectorReductionRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  constexpr llvm::StringLiteral consumerLabel(
      "vector reduction target artifact consumer");
  if (description.runtimeABIParameters.size() < 4)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived lhs, rhs seed/accumulator, out, and n "
        "ABI parameters before validating route statements");
  const support::RuntimeABIParameter &lhsABI =
      description.runtimeABIParameters[0];
  const support::RuntimeABIParameter &rhsABI =
      description.runtimeABIParameters[1];
  const support::RuntimeABIParameter &outABI =
      description.runtimeABIParameters[2];
  const support::RuntimeABIParameter &runtimeNABI =
      description.runtimeABIParameters[3];
  if (description.resultName.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived reduction result name before validating "
        "route statements");

  if (route.getCallOpaqueSteps().size() != 1)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires exactly one provider-built pre-loop setvl statement before "
        "artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          preLoopSetVL, consumerLabel, "pre-loop setvl",
          description.setVLIntrinsic, {{runtimeNABI.cName, runtimeNABI.cType}},
          description.emitCFullChunkVLName, description.vlCType))
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
  const support::RuntimeABIParameter *runtimeN =
      findRuntimeElementCountABIParameter(description);
  if (!runtimeN)
    return makeRVVTargetRouteError(
        "vector reduction target artifact consumer requires a "
        "provider-derived runtime element count ABI parameter");
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.upperBound.expression != runtimeN->cName ||
      loop.upperBound.cType != runtimeN->cType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
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
       description.emitCLoopInductionName)
          .str();
  if (loop.bodySteps.size() != 5)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-built loop setvl, lhs load, RHS seed/"
        "accumulator load, reduction, and output store statements before "
        "artifact export");
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[0], consumerLabel, "loop setvl",
          description.setVLIntrinsic,
          {{expectedRemainingAVL, description.vlCType}},
          description.emitCLoopVLName, description.vlCType))
    return error;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "vector reduction target artifact consumer requires loop statements "
          "to carry selected typed RVV source provenance");

  const std::string expectedLhsPointer =
      (llvm::StringRef(lhsABI.cName) + " + " +
       description.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[1], consumerLabel, "lhs source vector load",
          description.vectorLoadIntrinsic,
          {{expectedLhsPointer, lhsABI.cType},
           {description.emitCLoopVLName, description.vlCType}},
          "lhs_vec", description.vectorCType))
    return error;

  const std::string expectedRhsPointer =
      (llvm::StringRef(rhsABI.cName) + " + " +
       description.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[2], consumerLabel,
          "RHS seed/accumulator vector load", description.vectorLoadIntrinsic,
          {{expectedRhsPointer, rhsABI.cType},
           {description.emitCLoopVLName, description.vlCType}},
          "rhs_vec", description.vectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[3], consumerLabel, "reduce_add intrinsic",
          description.intrinsic,
          {{"lhs_vec", description.vectorCType},
           {"rhs_vec", description.vectorCType},
           {description.emitCLoopVLName, description.vlCType}},
          description.resultName, description.vectorCType))
    return error;

  const std::string expectedOutPointer =
      (llvm::StringRef(outABI.cName) + " + " +
       description.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[4], consumerLabel, "output store",
          description.storeIntrinsic,
          {{expectedOutPointer, outABI.cType},
           {description.resultName, description.vectorCType},
           {description.reductionStoreVL, description.vlCType}}))
    return error;
  return llvm::Error::success();
}

llvm::Error validateRVVVectorReductionRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("vector reduction target artifact consumer requires "
                    "rebuilt provider route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");
  if (description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        "vector reduction target artifact consumer requires provider route "
        "operand binding facts before artifact export");
  llvm::StringRef routeOperandBindingSummary(
      description.routeOperandBindingSummary);
  if (!routeOperandBindingSummary.contains("lhs=lhs-input-buffer") ||
      !routeOperandBindingSummary.contains("rhs=rhs-input-buffer") ||
      !routeOperandBindingSummary.contains("out=output-buffer") ||
      !routeOperandBindingSummary.contains("n=runtime-element-count"))
    return makeRVVTargetRouteError(
        "vector reduction target artifact consumer requires provider route "
        "operand binding facts for lhs input, rhs seed/accumulator, scalar "
        "output slot, and runtime n before artifact export");
  if (description.memoryForm !=
          plugin::rvv::RVVSelectedBodyMemoryForm::VectorRHSLoad ||
      description.typedComputeOpName != "tcrv_rvv.reduce")
    return makeRVVTargetRouteError(
        "vector reduction target artifact consumer requires a selected "
        "tcrv_rvv.reduce body with vector RHS-load memory form");
  if (description.runtimeABIOrder.empty() || description.setVLIntrinsic.empty())
    return makeRVVTargetRouteError(
        "vector reduction target artifact consumer requires "
        "provider-derived runtime AVL/VL facts before artifact export");
  if (llvm::Error error =
          validateRVVVectorReductionRuntimeABIFacts(description))
    return error;
  if (description.reductionAccumulatorLayout.empty() ||
      description.reductionResultLayout.empty() ||
      description.reductionStoreVL.empty() ||
      description.vectorLoadIntrinsic.empty() ||
      description.intrinsic.empty() || description.storeIntrinsic.empty())
    return makeRVVTargetRouteError(
        "vector reduction target artifact consumer requires "
        "provider-derived reduction layout, vector load, reduction, and store "
        "facts before artifact export");
  if (description.reductionAccumulatorLayout !=
          kRVVVectorReductionAccumulatorLayout ||
      description.reductionResultLayout != kRVVVectorReductionResultLayout ||
      description.reductionStoreVL != kRVVVectorReductionStoreVL)
    return makeRVVTargetRouteError(
        llvm::Twine("vector reduction target artifact consumer requires "
                    "provider-derived reduce_add layout facts accumulator '") +
        kRVVVectorReductionAccumulatorLayout + "', result '" +
        kRVVVectorReductionResultLayout + "', and store VL '" +
        kRVVVectorReductionStoreVL + "' before artifact export but provider "
        "carried accumulator '" +
        description.reductionAccumulatorLayout + "', result '" +
        description.reductionResultLayout + "', and store VL '" +
        description.reductionStoreVL + "'");
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
          validateRVVVectorReductionRouteTypeMappings(route, description))
    return error;
  if (llvm::Error error = validateRVVTargetOwnedRouteABIMappings(
          route, description, "vector reduction"))
    return error;
  return validateRVVVectorReductionRouteStatementPlan(route, description);
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

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "rvv_selected_body_typed_compute_op",
          description.typedComputeOpName,
          "selected typed RVV vector reduction typed compute op"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_plan",
          description.routeOperandBindingPlanID,
          "selected typed RVV vector reduction binding plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_operands",
          description.routeOperandBindingSummary,
          "selected typed RVV vector reduction binding summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.memory_form",
          plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
              description.memoryForm),
          "selected typed RVV vector reduction memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order",
          description.runtimeABIOrder,
          "selected typed RVV vector reduction runtime ABI order"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.reduction_accumulator_layout",
          description.reductionAccumulatorLayout,
          "selected typed RVV vector reduction accumulator layout"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.reduction_result_layout",
          description.reductionResultLayout,
          "selected typed RVV vector reduction result layout"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.reduction_store_vl",
          description.reductionStoreVL,
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

constexpr llvm::StringLiteral kRVVWideningMAccRuntimeABIOrder(
    "lhs,rhs,acc,out,n");
constexpr llvm::StringLiteral kRVVWideningMAccRouteOperandBindingPlan(
    "rvv-route-operand-binding:widening_macc_add.v1");
constexpr llvm::StringLiteral kRVVWideningMAccContractionRouteFamilyPlan(
    "rvv-contraction-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVWideningMAccProviderSupportedMirror(
    "provider_supported_mirror:rvv-contraction-family-plan-validated");
constexpr llvm::StringLiteral kRVVWideningMAccRequiredHeaders(
    "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVWideningMAccCTypeMapping(
    "vl:size_t,source:signed-e16mf2,result:signed-e32m1,mask:b32");
constexpr llvm::StringLiteral kRVVWideningMAccAccumulatorLayout(
    "separate-i32-vector-accumulator-input");
constexpr llvm::StringLiteral kRVVWideningMAccResultLayout(
    "store-widening-multiply-accumulate-result-to-output-buffer");
constexpr llvm::StringLiteral kRVVWideningMAccRelation(
    "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1");

llvm::Error validateRVVWideningMAccContractionRuntimeABIFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.runtimeABIOrder != kRVVWideningMAccRuntimeABIOrder)
    return makeRVVTargetRouteError(
        llvm::Twine("widening MAcc contraction target artifact consumer "
                    "requires provider-derived runtime ABI order '") +
        kRVVWideningMAccRuntimeABIOrder + "' but was '" +
        description.runtimeABIOrder + "'");
  if (description.runtimeABIParameters.size() != 5)
    return makeRVVTargetRouteError(
        "widening MAcc contraction target artifact consumer requires "
        "provider-derived runtime ABI parameters for lhs, rhs, accumulator, "
        "out, and n before artifact export");

  struct ExpectedRuntimeABIParameterRole {
    llvm::StringRef cName;
    support::RuntimeABIParameterRole role;
  };
  const ExpectedRuntimeABIParameterRole expectedRoles[] = {
      {"lhs", support::RuntimeABIParameterRole::LHSInputBuffer},
      {"rhs", support::RuntimeABIParameterRole::RHSInputBuffer},
      {"acc", support::RuntimeABIParameterRole::AccumulatorInputBuffer},
      {"out", support::RuntimeABIParameterRole::OutputBuffer},
      {"n", support::RuntimeABIParameterRole::RuntimeElementCount},
  };
  constexpr size_t expectedRoleCount =
      sizeof(expectedRoles) / sizeof(expectedRoles[0]);
  for (size_t index = 0; index < expectedRoleCount; ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    if (actual.cName != expectedRoles[index].cName ||
        actual.role != expectedRoles[index].role)
      return makeRVVTargetRouteError(
          llvm::Twine("widening MAcc contraction target artifact consumer "
                      "requires provider-derived runtime ABI parameter ") +
          std::to_string(index) + " to bind " + expectedRoles[index].cName +
          " as " +
          support::stringifyRuntimeABIParameterRole(expectedRoles[index].role) +
          " before artifact export");
  }
  return llvm::Error::success();
}

llvm::Error validateRVVWideningMAccContractionRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.vlCType.empty() || description.vectorTypeName.empty() ||
      description.vectorCType.empty() ||
      description.sourceVectorTypeName.empty() ||
      description.sourceVectorCType.empty())
    return makeRVVTargetRouteError(
        "widening MAcc contraction target artifact consumer requires "
        "provider-derived VL, source vector, and accumulator/result vector "
        "type facts before artifact export");
  if (description.sourceSEW != 16 || description.sourceLMUL != "mf2" ||
      description.sew != 32 || description.lmul != "m1")
    return makeRVVTargetRouteError(
        "widening MAcc contraction target artifact consumer requires "
        "provider-derived i16mf2 source and i32m1 accumulator/result facts "
        "before artifact export");
  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("widening MAcc contraction target artifact consumer "
                    "requires rebuilt provider route type mapping "
                    "'!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("widening MAcc contraction target artifact consumer "
                    "requires rebuilt provider route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");
  if (!routeHasTypeMapping(route, description.sourceVectorTypeName,
                           description.sourceVectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("widening MAcc contraction target artifact consumer "
                    "requires rebuilt provider route type mapping '") +
        description.sourceVectorTypeName + "' -> '" +
        description.sourceVectorCType + "'");
  return llvm::Error::success();
}

llvm::Error validateRVVWideningMAccContractionRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  constexpr llvm::StringLiteral consumerLabel(
      "widening MAcc contraction target artifact consumer");
  if (description.runtimeABIParameters.size() != 5)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived widening MAcc ABI parameters before "
        "validating route statements");
  if (description.resultName.empty() || description.sourceVectorCType.empty() ||
      description.vectorCType.empty() || description.vlCType.empty() ||
      description.vectorLoadIntrinsic.empty())
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires provider-derived result, source/result vector C type, "
        "accumulator load, and VL C type facts before validating route "
        "statements");

  const support::RuntimeABIParameter &lhsABI =
      description.runtimeABIParameters[0];
  const support::RuntimeABIParameter &rhsABI =
      description.runtimeABIParameters[1];
  const support::RuntimeABIParameter &accumulatorABI =
      description.runtimeABIParameters[2];
  const support::RuntimeABIParameter &outABI =
      description.runtimeABIParameters[3];
  const support::RuntimeABIParameter &runtimeNABI =
      description.runtimeABIParameters[4];
  const support::RuntimeABIParameter *runtimeElementCount =
      findRuntimeElementCountABIParameter(description);
  if (!runtimeElementCount || runtimeElementCount != &runtimeNABI)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires runtime n/AVL ABI role to match the selected widening "
        "MAcc ABI order before validating route statements");

  if (route.getCallOpaqueSteps().size() != 1)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires exactly one provider-built pre-loop setvl statement "
        "before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          preLoopSetVL, consumerLabel, "pre-loop setvl",
          description.setVLIntrinsic,
          {{runtimeNABI.cName, runtimeNABI.cType}},
          description.emitCFullChunkVLName, description.vlCType))
    return error;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "widening MAcc contraction target artifact consumer requires "
          "pre-loop statements to carry selected typed RVV source provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "widening MAcc contraction target artifact consumer requires exactly "
        "one provider-built runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.upperBound.expression != runtimeNABI.cName ||
      loop.upperBound.cType != runtimeNABI.cType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "widening MAcc contraction target artifact consumer requires "
        "provider-built loop bounds and step to mirror runtime n/AVL/VL facts");
  if (loop.bodySteps.size() != 6)
    return makeRVVTargetRouteError(
        llvm::Twine(consumerLabel) +
        " requires exact provider-built widening MAcc loop statement count 6 "
        "before artifact export");

  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeNABI.cName) + " - " +
       description.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[0], consumerLabel, "loop setvl",
          description.setVLIntrinsic,
          {{expectedRemainingAVL, description.vlCType}},
          description.emitCLoopVLName, description.vlCType))
    return error;

  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "widening MAcc contraction target artifact consumer requires loop "
          "statements to carry selected typed RVV source provenance");

  auto validateSourceLoad =
      [&](const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
          const support::RuntimeABIParameter &abi, llvm::StringRef resultName,
          llvm::StringRef stepLabel) -> llvm::Error {
    const std::string expectedPointer =
        (llvm::StringRef(abi.cName) + " + " +
         description.emitCLoopInductionName)
            .str();
    return validateRVVProviderBuiltRouteStep(
        step, consumerLabel, stepLabel, description.sourceVectorLoadIntrinsic,
        {{expectedPointer, abi.cType},
         {description.emitCLoopVLName, description.vlCType}},
        resultName, description.sourceVectorCType);
  };

  if (llvm::Error error =
          validateSourceLoad(loop.bodySteps[1], lhsABI, "lhs_vec",
                             "lhs source load"))
    return error;
  if (llvm::Error error =
          validateSourceLoad(loop.bodySteps[2], rhsABI, "rhs_vec",
                             "rhs source load"))
    return error;

  const std::string expectedAccumulatorPointer =
      (llvm::StringRef(accumulatorABI.cName) + " + " +
       description.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[3], consumerLabel, "accumulator load",
          description.vectorLoadIntrinsic,
          {{expectedAccumulatorPointer, accumulatorABI.cType},
           {description.emitCLoopVLName, description.vlCType}},
          "acc_vec", description.vectorCType))
    return error;

  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[4], consumerLabel, "widening MAcc",
          description.intrinsic,
          {{"acc_vec", description.vectorCType},
           {"lhs_vec", description.sourceVectorCType},
           {"rhs_vec", description.sourceVectorCType},
           {description.emitCLoopVLName, description.vlCType}},
          description.resultName, description.vectorCType))
    return error;

  const std::string expectedOutPointer =
      (llvm::StringRef(outABI.cName) + " + " +
       description.emitCLoopInductionName)
          .str();
  if (llvm::Error error = validateRVVProviderBuiltRouteStep(
          loop.bodySteps[5], consumerLabel, "output store",
          description.storeIntrinsic,
          {{expectedOutPointer, outABI.cType},
           {description.resultName, description.vectorCType},
           {description.emitCLoopVLName, description.vlCType}}))
    return error;

  return llvm::Error::success();
}

llvm::Error validateRVVWideningMAccContractionRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("widening MAcc contraction target artifact consumer "
                    "requires rebuilt provider route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");
  if (description.providerSupportedMirror.empty() ||
      description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty() ||
      description.contractionRouteFamilyPlanID.empty())
    return makeRVVTargetRouteError(
        "widening MAcc contraction target artifact consumer requires "
        "provider support, binding, and contraction route-family facts before "
        "artifact export");
  if (description.providerSupportedMirror !=
          kRVVWideningMAccProviderSupportedMirror ||
      description.contractionRouteFamilyPlanID !=
          kRVVWideningMAccContractionRouteFamilyPlan ||
      description.requiredHeaderDeclarations !=
          kRVVWideningMAccRequiredHeaders ||
      description.cTypeMappingSummary != kRVVWideningMAccCTypeMapping)
    return makeRVVTargetRouteError(
        llvm::Twine("widening MAcc contraction target artifact consumer "
                    "requires provider-owned contraction support, header, "
                    "and C type facts before artifact export but provider "
                    "carried support '") +
        description.providerSupportedMirror + "', plan '" +
        description.contractionRouteFamilyPlanID + "', headers '" +
        description.requiredHeaderDeclarations + "', and C type mapping '" +
        description.cTypeMappingSummary + "'");
  if (description.routeOperandBindingPlanID !=
      kRVVWideningMAccRouteOperandBindingPlan)
    return makeRVVTargetRouteError(
        llvm::Twine("widening MAcc contraction target artifact consumer "
                    "requires provider route operand binding plan '") +
        kRVVWideningMAccRouteOperandBindingPlan + "' but was '" +
        description.routeOperandBindingPlanID + "'");
  llvm::StringRef routeOperandBindingSummary(
      description.routeOperandBindingSummary);
  if (!routeOperandBindingSummary.starts_with(
          kRVVWideningMAccRouteOperandBindingPlan) ||
      !routeOperandBindingSummary.contains("lhs=lhs-input-buffer") ||
      !routeOperandBindingSummary.contains("rhs=rhs-input-buffer") ||
      !routeOperandBindingSummary.contains(
          "acc=accumulator-input-buffer") ||
      !routeOperandBindingSummary.contains("out=output-buffer") ||
      !routeOperandBindingSummary.contains("n=runtime-element-count") ||
      !routeOperandBindingSummary.contains("wmacc-lhs") ||
      !routeOperandBindingSummary.contains("wmacc-rhs") ||
      !routeOperandBindingSummary.contains("wmacc-acc") ||
      !routeOperandBindingSummary.contains("src-i16mf2") ||
      !routeOperandBindingSummary.contains("acc-i32m1") ||
      !routeOperandBindingSummary.contains("res-i32m1"))
    return makeRVVTargetRouteError(
        "widening MAcc contraction target artifact consumer requires "
        "provider route operand binding facts for lhs/rhs i16 sources, i32 "
        "accumulator, i32 output, and runtime n before artifact export");
  if (description.memoryForm !=
          plugin::rvv::RVVSelectedBodyMemoryForm::VectorRHSLoad ||
      description.typedComputeOpName != "tcrv_rvv.widening_macc")
    return makeRVVTargetRouteError(
        "widening MAcc contraction target artifact consumer requires a "
        "selected tcrv_rvv.widening_macc body with vector RHS-load memory "
        "form");
  if (description.runtimeControlPlanID.empty() ||
      description.runtimeABIOrder.empty() || description.setVLIntrinsic.empty())
    return makeRVVTargetRouteError(
        "widening MAcc contraction target artifact consumer requires "
        "provider-derived runtime AVL/VL facts before artifact export");
  if (llvm::Error error =
          validateRVVWideningMAccContractionRuntimeABIFacts(description))
    return error;
  if (description.wideningMAccAccumulatorLayout.empty() ||
      description.wideningMAccResultLayout.empty() ||
      description.wideningMAccRelation.empty() ||
      description.sourceVectorLoadIntrinsic.empty() ||
      description.intrinsic.empty() || description.storeIntrinsic.empty())
    return makeRVVTargetRouteError(
        "widening MAcc contraction target artifact consumer requires "
        "provider-derived accumulator/result layout, relation, source load, "
        "widening MAcc, and store facts before artifact export");
  if (description.wideningMAccAccumulatorLayout !=
          kRVVWideningMAccAccumulatorLayout ||
      description.wideningMAccResultLayout != kRVVWideningMAccResultLayout ||
      description.wideningMAccRelation != kRVVWideningMAccRelation)
    return makeRVVTargetRouteError(
        llvm::Twine("widening MAcc contraction target artifact consumer "
                    "requires provider-derived widening MAcc layout and "
                    "relation facts accumulator '") +
        kRVVWideningMAccAccumulatorLayout + "', result '" +
        kRVVWideningMAccResultLayout + "', relation '" +
        kRVVWideningMAccRelation +
        "' before artifact export but provider carried accumulator '" +
        description.wideningMAccAccumulatorLayout + "', result '" +
        description.wideningMAccResultLayout + "', and relation '" +
        description.wideningMAccRelation + "'");
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
      !description.wideningDotProductRelation.empty() ||
      !description.wideningProductIntrinsic.empty() ||
      !description.scalarSeedSplatIntrinsic.empty())
    return makeRVVTargetRouteError(
        "widening MAcc contraction target artifact consumer rejects stale "
        "non-widening-MAcc route-family facts");

  if (llvm::Error error =
          validateRVVWideningMAccContractionRouteTypeMappings(route,
                                                              description))
    return error;
  if (llvm::Error error = validateRVVTargetOwnedRouteABIMappings(
          route, description, "widening MAcc contraction"))
    return error;
  return validateRVVWideningMAccContractionRouteStatementPlan(route,
                                                              description);
}

llvm::Error validateRVVWideningMAccContractionTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVWideningMAccContractionRoutePayloadFacts(
      context.route, context.description);
}

llvm::Error requireEmptyWideningMAccContractionStaleMirror(
    const TargetArtifactCandidate &candidate, llvm::StringRef key,
    llvm::StringRef label) {
  return requireCandidateMetadataMirror(candidate, key, "", label);
}

llvm::Error validateRVVWideningMAccContractionTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description =
      context.description;
  const std::string sourceSEW = llvm::Twine(description.sourceSEW).str();
  const std::string resultSEW = llvm::Twine(description.sew).str();

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_plan",
          description.routeOperandBindingPlanID,
          "selected typed RVV widening MAcc binding plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_operands",
          description.routeOperandBindingSummary,
          "selected typed RVV widening MAcc binding summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.provider_supported_mirror",
          description.providerSupportedMirror,
          "selected typed RVV widening MAcc provider support"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.contraction_route_family_plan",
          description.contractionRouteFamilyPlanID,
          "selected typed RVV widening MAcc contraction route-family plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.memory_form",
          plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
              description.memoryForm),
          "selected typed RVV widening MAcc memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_control_plan",
          description.runtimeControlPlanID,
          "selected typed RVV widening MAcc runtime AVL/VL control plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order", description.runtimeABIOrder,
          "selected typed RVV widening MAcc runtime ABI order"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.required_header_declarations",
          description.requiredHeaderDeclarations,
          "selected typed RVV widening MAcc route header requirements"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.c_type_mapping",
          description.cTypeMappingSummary,
          "selected typed RVV widening MAcc route type mapping summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_sew", sourceSEW,
          "selected typed RVV widening MAcc source SEW"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_lmul", description.sourceLMUL,
          "selected typed RVV widening MAcc source LMUL"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.accumulator_sew", resultSEW,
          "selected typed RVV widening MAcc accumulator SEW"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.accumulator_lmul", description.lmul,
          "selected typed RVV widening MAcc accumulator LMUL"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.result_sew", resultSEW,
          "selected typed RVV widening MAcc result SEW"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.result_lmul", description.lmul,
          "selected typed RVV widening MAcc result LMUL"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.widening_macc_accumulator_layout",
          description.wideningMAccAccumulatorLayout,
          "selected typed RVV widening MAcc accumulator layout"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.widening_macc_result_layout",
          description.wideningMAccResultLayout,
          "selected typed RVV widening MAcc result layout"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.widening_macc_relation",
          description.wideningMAccRelation,
          "selected typed RVV widening MAcc relation"))
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
      "tcrv_rvv.widening_dot_relation"};
  for (llvm::StringRef key : staleRouteFamilyMirrors)
    if (llvm::Error error = requireEmptyWideningMAccContractionStaleMirror(
            candidate, key,
            "selected typed RVV non-widening-MAcc route-family mirror"))
      return error;
  return llvm::Error::success();
}

bool isRVVWideningDotReductionTargetArtifactRouteFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVWideningDotReductionRouteFamilyOperation(description.operation);
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
      {llvm::StringLiteral("macc"),
       isRVVMAccTargetArtifactRouteFamilyConsumer,
       validateRVVMAccTargetArtifactProviderFacts,
       validateRVVMAccTargetArtifactCandidateMirrors},
      {llvm::StringLiteral("widening-macc-contraction"),
       isRVVWideningMAccContractionTargetArtifactRouteFamilyConsumer,
       validateRVVWideningMAccContractionTargetArtifactProviderFacts,
       validateRVVWideningMAccContractionTargetArtifactCandidateMirrors},
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
