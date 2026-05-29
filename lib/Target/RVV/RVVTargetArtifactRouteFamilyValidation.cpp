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
         plugin::rvv::RVVSelectedBodyOperationKind::RuntimeI32SplatStore;
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

const support::RuntimeABIParameter *findRuntimeElementCountABIParameter(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  for (const support::RuntimeABIParameter &parameter :
       description.runtimeABIParameters)
    if (parameter.role ==
        support::RuntimeABIParameterRole::RuntimeElementCount)
      return &parameter;
  return nullptr;
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

llvm::Error validateRVVBaseMemoryMovementRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getCallOpaqueSteps().empty())
    return makeRVVTargetRouteError(
        "base-memory-movement target artifact consumer requires "
        "provider-built pre-loop setvl statement facts before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "base-memory-movement target artifact consumer requires rebuilt "
        "provider route pre-loop setvl statement to define the full-chunk VL");
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
  const support::RuntimeABIParameter *runtimeN =
      findRuntimeElementCountABIParameter(description);
  if (!runtimeN)
    return makeRVVTargetRouteError(
        "base-memory-movement target artifact consumer requires a "
        "provider-derived runtime element count ABI parameter");
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.upperBound.expression != runtimeN->cName ||
      loop.upperBound.cType != runtimeN->cType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "base-memory-movement target artifact consumer requires "
        "provider-built loop bounds and step to mirror runtime n/AVL/VL facts");
  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeN->cName) + " - " +
       description.emitCLoopInductionName)
          .str();
  if (loop.bodySteps.empty() ||
      loop.bodySteps.front().callee != description.setVLIntrinsic ||
      loop.bodySteps.front().operands.empty() ||
      loop.bodySteps.front().operands.front().expression !=
          expectedRemainingAVL ||
      loop.bodySteps.front().operands.front().cType != description.vlCType ||
      !stepHasResult(loop.bodySteps.front(), description.emitCLoopVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "base-memory-movement target artifact consumer requires "
        "provider-built loop setvl to derive per-iteration VL from remaining "
        "runtime AVL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "base-memory-movement target artifact consumer requires loop "
          "statements to carry selected typed RVV source provenance");

  auto requireLoopCallee = [&](llvm::StringRef callee,
                               llvm::StringRef factLabel) -> llvm::Error {
    if (callee.empty())
      return makeRVVTargetRouteError(
          llvm::Twine("base-memory-movement target artifact consumer requires "
                      "provider-derived ") +
          factLabel + " statement facts before artifact export");
    if (routeLoopContainsCallee(loop, callee))
      return llvm::Error::success();
    return makeRVVTargetRouteError(
        llvm::Twine("base-memory-movement target artifact consumer requires "
                    "provider-built ") +
        factLabel + " statement callee '" + callee +
        "' to mirror selected body route facts before artifact export");
  };

  using OperationKind = plugin::rvv::RVVSelectedBodyOperationKind;
  switch (description.operation) {
  case OperationKind::StridedLoadUnitStore:
    if (llvm::Error error =
            requireLoopCallee(description.stridedLoadIntrinsic,
                              "strided load"))
      return error;
    return requireLoopCallee(description.storeIntrinsic, "unit store");
  case OperationKind::UnitLoadStridedStore:
    if (llvm::Error error =
            requireLoopCallee(description.vectorLoadIntrinsic, "unit load"))
      return error;
    return requireLoopCallee(description.stridedStoreIntrinsic,
                             "strided store");
  case OperationKind::IndexedGatherUnitStore:
    if (llvm::Error error =
            requireLoopCallee(description.indexLoadIntrinsic, "index load"))
      return error;
    if (llvm::Error error =
            requireLoopCallee(description.indexScaleIntrinsic, "index scale"))
      return error;
    if (llvm::Error error = requireLoopCallee(
            description.indexedLoadIntrinsic, "indexed gather load"))
      return error;
    return requireLoopCallee(description.storeIntrinsic, "unit store");
  case OperationKind::IndexedScatterUnitLoad:
    if (llvm::Error error =
            requireLoopCallee(description.vectorLoadIntrinsic, "unit load"))
      return error;
    if (llvm::Error error =
            requireLoopCallee(description.indexLoadIntrinsic, "index load"))
      return error;
    if (llvm::Error error =
            requireLoopCallee(description.indexScaleIntrinsic, "index scale"))
      return error;
    return requireLoopCallee(description.indexedStoreIntrinsic,
                             "indexed scatter store");
  case OperationKind::MaskedUnitLoadStore:
    if (llvm::Error error =
            requireLoopCallee(description.vectorLoadIntrinsic,
                              "mask/passthrough load"))
      return error;
    if (llvm::Error error =
            requireLoopCallee(description.compareIntrinsic,
                              "mask predicate"))
      return error;
    if (llvm::Error error =
            requireLoopCallee(description.maskedLoadIntrinsic, "masked load"))
      return error;
    return requireLoopCallee(description.storeIntrinsic, "unit store");
  case OperationKind::MaskedUnitStore:
    if (llvm::Error error =
            requireLoopCallee(description.vectorLoadIntrinsic,
                              "source/mask load"))
      return error;
    if (llvm::Error error =
            requireLoopCallee(description.compareIntrinsic,
                              "mask predicate"))
      return error;
    return requireLoopCallee(description.storeIntrinsic, "masked store");
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
        "cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp|mask;"
        "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp|mask;"
        "dot_lhs=dot-lhs-input-buffer:lhs:abi|sld|mlhs|i16;"
        "dot_rhs=dot-rhs-input-buffer:rhs:abi|sld|mrhs|i16;"
        "acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;"
        "out=output-buffer:out:abi|store|i32|hdr;"
        "n=runtime-element-count:n:abi|setvl-avl|loop|hdr;"
        "lhs_stride=lhs-input-stride:lhs_stride:abi|str|addr;"
        "rhs_stride=rhs-input-stride:rhs_stride:abi|str|addr");
constexpr llvm::StringLiteral kRVVWideningDotContractionRouteFamilyPlan(
    "rvv-contraction-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVWideningDotProviderSupportedMirror(
    "provider_supported_mirror:rvv-contraction-family-plan-validated");
constexpr llvm::StringLiteral kRVVWideningDotTargetLeafProfile(
    "rvv-v1-i16mf2-i32m1-contraction-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVWideningDotRequiredHeaders(
    "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVWideningDotCTypeMapping(
    "vl:size_t,source:signed-e16mf2,result:signed-e32m1,mask:b32");
constexpr llvm::StringLiteral kRVVWideningDotAccumulatorLayout(
    "scalar-i32-seed-lane0-from-accumulator-input");
constexpr llvm::StringLiteral kRVVWideningDotResultLayout(
    "store-dot-reduction-lane0-to-output-scalar");
constexpr llvm::StringLiteral kRVVWideningDotRelation(
    "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32");
constexpr llvm::StringLiteral kRVVWideningDotReductionStoreVL("1");
constexpr llvm::StringLiteral kRVVWideningDotProductIntrinsic(
    "__riscv_vwmul_vv_i32m1");
constexpr llvm::StringLiteral kRVVWideningDotScalarSeedSplatIntrinsic(
    "__riscv_vmv_v_x_i32m1");
constexpr llvm::StringLiteral kRVVWideningDotReductionIntrinsic(
    "__riscv_vredsum_vs_i32m1_i32m1");
constexpr llvm::StringLiteral kRVVWideningDotSourceLoadIntrinsic(
    "__riscv_vle16_v_i16mf2");
constexpr llvm::StringLiteral kRVVStridedInputWideningDotSourceLoadIntrinsic(
    "__riscv_vlse16_v_i16mf2");
constexpr llvm::StringLiteral kRVVWideningDotStoreIntrinsic(
    "__riscv_vse32_v_i32m1");
constexpr llvm::StringLiteral kRVVStridedInputWideningDotMemoryLayout(
    "element-strided-lhs-rhs-dot-source-unit-stride-output-runtime-abi");
constexpr llvm::StringLiteral kRVVStridedInputWideningDotLHSStrideSource(
    "runtime_abi:lhs_stride");
constexpr llvm::StringLiteral kRVVStridedInputWideningDotRHSStrideSource(
    "runtime_abi:rhs_stride");
constexpr llvm::StringLiteral kRVVStridedInputWideningDotSourceMemoryForm(
    "strided-load");
constexpr llvm::StringLiteral kRVVStridedInputWideningDotDestinationMemoryForm(
    "unit-stride-store");
constexpr llvm::StringLiteral
    kRVVComputedMaskStridedInputWideningDotMemoryLayout(
        "unit-stride-compare-element-strided-lhs-rhs-dot-source-unit-stride-output-runtime-abi");
constexpr llvm::StringLiteral kRVVComputedMaskRole(
    "predicate-mask-produced-by-compare");
constexpr llvm::StringLiteral kRVVComputedMaskSource(
    "compare-produced-mask-same-vl-scope");
constexpr llvm::StringLiteral kRVVComputedMaskMemoryForm(
    "compare-produced-mask");
constexpr llvm::StringLiteral kRVVComputedMaskInactiveLaneZeroingRequirement(
    "masked-widening-products-zero-inactive-lanes-before-reduction");
constexpr llvm::StringLiteral kRVVComputedMaskComparePredicateKind("slt");
constexpr llvm::StringLiteral kRVVComputedMaskCompareIntrinsic(
    "__riscv_vmslt_vv_i32m1_b32");
constexpr llvm::StringLiteral kRVVComputedMaskCompareLoadIntrinsic(
    "__riscv_vle32_v_i32m1");
constexpr llvm::StringLiteral kRVVComputedMaskWideningDotMaskedProductIntrinsic(
    "__riscv_vwmul_vv_i32m1_m");
constexpr llvm::StringLiteral kRVVComputedMaskWideningDotMergeIntrinsic(
    "__riscv_vmerge_vvm_i32m1");
constexpr llvm::StringLiteral kRVVComputedMaskTypeName(
    "!tcrv_rvv.mask<i32, \"m1\">");
constexpr llvm::StringLiteral kRVVComputedMaskCType("vbool32_t");
constexpr llvm::StringLiteral kRVVWideningDotTailPolicy("agnostic");
constexpr llvm::StringLiteral kRVVWideningDotMaskPolicy("agnostic");

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
  const llvm::StringRef expectedABIOrder =
      isStrided ? kRVVComputedMaskStridedInputWideningDotRuntimeABIOrder
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

  if (description.providerSupportedMirror !=
          kRVVWideningDotProviderSupportedMirror ||
      description.contractionRouteFamilyPlanID !=
          kRVVWideningDotContractionRouteFamilyPlan ||
      description.targetLeafProfile != kRVVWideningDotTargetLeafProfile ||
      description.requiredHeaderDeclarations != kRVVWideningDotRequiredHeaders ||
      description.cTypeMappingSummary != kRVVWideningDotCTypeMapping)
    return makeRVVTargetRouteError(
        llvm::Twine("widening dot-reduction target artifact consumer requires "
                    "provider-owned contraction support, target leaf profile, "
                    "header, and C type facts before artifact export but "
                    "provider carried support '") +
        description.providerSupportedMirror + "', plan '" +
        description.contractionRouteFamilyPlanID + "', target profile '" +
        description.targetLeafProfile + "', headers '" +
        description.requiredHeaderDeclarations + "', and C type mapping '" +
        description.cTypeMappingSummary + "'");
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
  if (description.wideningDotProductAccumulatorLayout !=
          kRVVWideningDotAccumulatorLayout ||
      description.wideningDotProductResultLayout !=
          kRVVWideningDotResultLayout ||
      description.wideningDotProductRelation != kRVVWideningDotRelation ||
      description.reductionStoreVL != kRVVWideningDotReductionStoreVL)
    return makeRVVTargetRouteError(
        llvm::Twine("widening dot-reduction target artifact consumer requires "
                    "provider-derived dot layout/relation facts accumulator '") +
        kRVVWideningDotAccumulatorLayout + "', result '" +
        kRVVWideningDotResultLayout + "', relation '" +
        kRVVWideningDotRelation + "', and store VL '" +
        kRVVWideningDotReductionStoreVL + "' before artifact export but saw "
        "accumulator '" +
        description.wideningDotProductAccumulatorLayout + "', result '" +
        description.wideningDotProductResultLayout + "', relation '" +
        description.wideningDotProductRelation + "', and store VL '" +
        description.reductionStoreVL + "'");
  if (description.sourceVectorLoadIntrinsic !=
          kRVVWideningDotSourceLoadIntrinsic ||
      description.wideningProductIntrinsic != kRVVWideningDotProductIntrinsic ||
      description.scalarSeedSplatIntrinsic !=
          kRVVWideningDotScalarSeedSplatIntrinsic ||
      description.intrinsic != kRVVWideningDotReductionIntrinsic ||
      description.storeIntrinsic != kRVVWideningDotStoreIntrinsic)
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-derived source load, widening product, scalar seed, "
        "reduction, and store statement facts before artifact export");
  if (isStrided) {
    if (description.stridedLoadIntrinsic !=
            kRVVStridedInputWideningDotSourceLoadIntrinsic ||
        description.stridedMemoryLayout !=
            kRVVStridedInputWideningDotMemoryLayout ||
        description.lhsStrideSource !=
            kRVVStridedInputWideningDotLHSStrideSource ||
        description.rhsStrideSource !=
            kRVVStridedInputWideningDotRHSStrideSource ||
        description.sourceMemoryForm !=
            kRVVStridedInputWideningDotSourceMemoryForm ||
        description.destinationMemoryForm !=
            kRVVStridedInputWideningDotDestinationMemoryForm)
      return makeRVVTargetRouteError(
          "strided-input widening dot-reduction target artifact consumer "
          "requires exact provider-derived strided load, stride ABI, "
          "source/result memory form, and layout facts before artifact export");
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
  const llvm::StringRef expectedBindingPlan =
      isStrided ? kRVVComputedMaskStridedInputWideningDotRouteOperandBindingPlan
                : kRVVComputedMaskWideningDotRouteOperandBindingPlan;
  const llvm::StringRef expectedBindingSummary =
      isStrided
          ? kRVVComputedMaskStridedInputWideningDotRouteOperandBindingSummary
          : kRVVComputedMaskWideningDotRouteOperandBindingSummary;
  const plugin::rvv::RVVSelectedBodyMemoryForm expectedMemoryForm =
      isStrided
          ? plugin::rvv::RVVSelectedBodyMemoryForm::
                ComputedMaskStridedInputWideningDotReduce
          : plugin::rvv::RVVSelectedBodyMemoryForm::
                ComputedMaskUnitStrideWideningDotReduce;
  const llvm::StringRef expectedMemoryLayout =
      isStrided ? kRVVComputedMaskStridedInputWideningDotMemoryLayout : "";

  if (description.providerSupportedMirror !=
          kRVVWideningDotProviderSupportedMirror ||
      description.contractionRouteFamilyPlanID !=
          kRVVWideningDotContractionRouteFamilyPlan ||
      description.targetLeafProfile != kRVVWideningDotTargetLeafProfile ||
      description.requiredHeaderDeclarations != kRVVWideningDotRequiredHeaders ||
      description.cTypeMappingSummary != kRVVWideningDotCTypeMapping)
    return makeRVVTargetRouteError(
        llvm::Twine("computed-mask widening dot-reduction target artifact "
                    "consumer requires provider-owned contraction support, "
                    "target leaf profile, header, and C type facts before "
                    "artifact export but provider carried support '") +
        description.providerSupportedMirror + "', plan '" +
        description.contractionRouteFamilyPlanID + "', target profile '" +
        description.targetLeafProfile + "', headers '" +
        description.requiredHeaderDeclarations + "', and C type mapping '" +
        description.cTypeMappingSummary + "'");
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
  if (description.tailPolicy != kRVVWideningDotTailPolicy ||
      description.maskPolicy != kRVVWideningDotMaskPolicy)
    return makeRVVTargetRouteError(
        llvm::Twine("computed-mask widening dot-reduction target artifact "
                    "consumer requires provider-derived tail/mask policy '") +
        kRVVWideningDotTailPolicy + "/" + kRVVWideningDotMaskPolicy +
        "' before artifact export");
  if (description.maskRole != kRVVComputedMaskRole ||
      description.maskSource != kRVVComputedMaskSource ||
      description.maskMemoryForm != kRVVComputedMaskMemoryForm ||
      description.inactiveLaneZeroingRequirement !=
          kRVVComputedMaskInactiveLaneZeroingRequirement ||
      description.comparePredicateKind != kRVVComputedMaskComparePredicateKind ||
      description.compareIntrinsic != kRVVComputedMaskCompareIntrinsic ||
      description.vectorLoadIntrinsic != kRVVComputedMaskCompareLoadIntrinsic ||
      description.maskedWideningProductIntrinsic !=
          kRVVComputedMaskWideningDotMaskedProductIntrinsic ||
      description.maskedMergeIntrinsic !=
          kRVVComputedMaskWideningDotMergeIntrinsic ||
      description.maskTypeName != kRVVComputedMaskTypeName ||
      description.maskCType != kRVVComputedMaskCType)
    return makeRVVTargetRouteError(
        "computed-mask widening dot-reduction target artifact consumer "
        "requires exact provider-derived mask role/source/form, predicate, "
        "compare, masked product, merge, inactive-lane zeroing, and mask type "
        "facts before artifact export");
  if (description.wideningDotProductAccumulatorLayout !=
          kRVVWideningDotAccumulatorLayout ||
      description.wideningDotProductResultLayout !=
          kRVVWideningDotResultLayout ||
      description.wideningDotProductRelation != kRVVWideningDotRelation ||
      description.reductionStoreVL != kRVVWideningDotReductionStoreVL)
    return makeRVVTargetRouteError(
        llvm::Twine("computed-mask widening dot-reduction target artifact "
                    "consumer requires provider-derived dot layout/relation "
                    "facts accumulator '") +
        kRVVWideningDotAccumulatorLayout + "', result '" +
        kRVVWideningDotResultLayout + "', relation '" +
        kRVVWideningDotRelation + "', and store VL '" +
        kRVVWideningDotReductionStoreVL + "' before artifact export but saw "
        "accumulator '" +
        description.wideningDotProductAccumulatorLayout + "', result '" +
        description.wideningDotProductResultLayout + "', relation '" +
        description.wideningDotProductRelation + "', and store VL '" +
        description.reductionStoreVL + "'");
  if (description.sourceVectorLoadIntrinsic !=
          kRVVWideningDotSourceLoadIntrinsic ||
      description.wideningProductIntrinsic != kRVVWideningDotProductIntrinsic ||
      description.scalarSeedSplatIntrinsic !=
          kRVVWideningDotScalarSeedSplatIntrinsic ||
      description.intrinsic != kRVVWideningDotReductionIntrinsic ||
      description.storeIntrinsic != kRVVWideningDotStoreIntrinsic)
    return makeRVVTargetRouteError(
        "computed-mask widening dot-reduction target artifact consumer "
        "requires provider-derived source load, widening product, scalar "
        "seed, reduction, and store statement facts before artifact export");
  if (isStrided) {
    if (description.stridedLoadIntrinsic !=
            kRVVStridedInputWideningDotSourceLoadIntrinsic ||
        description.stridedMemoryLayout != expectedMemoryLayout ||
        description.lhsStrideSource !=
            kRVVStridedInputWideningDotLHSStrideSource ||
        description.rhsStrideSource !=
            kRVVStridedInputWideningDotRHSStrideSource ||
        description.sourceMemoryForm !=
            kRVVStridedInputWideningDotSourceMemoryForm ||
        description.destinationMemoryForm !=
            kRVVStridedInputWideningDotDestinationMemoryForm)
      return makeRVVTargetRouteError(
          "computed-mask strided-input widening dot-reduction target artifact "
          "consumer requires exact provider-derived strided load, stride ABI, "
          "source/result memory form, and layout facts before artifact export");
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
        "provider-derived VL, result vector, i16 source vector, and C type "
        "mapping facts before artifact export");

  if (description.sourceSEW != 16 || description.sourceLMUL != "mf2" ||
      description.sew != 32 || description.lmul != "m1" ||
      description.sourceVectorTypeName != "!tcrv_rvv.vector<i16, \"mf2\">" ||
      description.sourceVectorCType != "vint16mf2_t" ||
      description.vectorTypeName != "!tcrv_rvv.vector<i32, \"m1\">" ||
      description.vectorCType != "vint32m1_t")
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-derived i16mf2 source and i32m1 result type facts before "
        "artifact export");

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
                    "rebuilt provider route i16 source type mapping '") +
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
  if (route.getCallOpaqueSteps().empty())
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-built pre-loop setvl, seed, and initial store statement "
        "facts before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires rebuilt "
        "provider route pre-loop setvl statement to define the full-chunk VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "widening dot-reduction target artifact consumer requires pre-loop "
          "statements to carry selected typed RVV source provenance");
  if (!routeStepsContainCallee(route.getCallOpaqueSteps(),
                               description.scalarSeedSplatIntrinsic) ||
      !routeStepsContainCallee(route.getCallOpaqueSteps(),
                               description.storeIntrinsic))
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-built pre-loop scalar seed and initial store statements "
        "before artifact export");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires exactly one "
        "provider-built runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-built loop bounds and step to mirror runtime AVL/VL route "
        "facts");
  const support::RuntimeABIParameter *runtimeElementCount =
      findRuntimeElementCountABIParameter(description);
  if (!runtimeElementCount)
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires a "
        "provider-derived runtime n/AVL ABI parameter before artifact export");
  if (loop.upperBound.expression != runtimeElementCount->cName ||
      loop.upperBound.cType != runtimeElementCount->cType)
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-built loop upper bound to use the runtime n/AVL ABI "
        "parameter");
  if (loop.bodySteps.empty() ||
      loop.bodySteps.front().callee != description.setVLIntrinsic ||
      !stepHasResult(loop.bodySteps.front(), description.emitCLoopVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-built loop setvl statement to define per-iteration VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "widening dot-reduction target artifact consumer requires loop "
          "statements to carry selected typed RVV source provenance");

  llvm::StringRef sourceLoad =
      isRVVStridedInputWideningDotReductionRouteFamilyOperation(
          description.operation)
          ? description.stridedLoadIntrinsic
          : description.sourceVectorLoadIntrinsic;
  if (sourceLoad.empty() || !routeLoopContainsCallee(loop, sourceLoad) ||
      !routeLoopContainsCallee(loop, description.scalarSeedSplatIntrinsic) ||
      !routeLoopContainsCallee(loop, description.intrinsic) ||
      !routeLoopContainsCallee(loop, description.storeIntrinsic))
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-built source load, scalar seed, reduction, and store "
        "statements before artifact export");

  if (isRVVComputedMaskWideningDotReductionRouteFamilyOperation(
          description.operation)) {
    if (description.vectorLoadIntrinsic.empty() ||
        description.compareIntrinsic.empty() ||
        description.maskedWideningProductIntrinsic.empty() ||
        description.maskedMergeIntrinsic.empty())
      return makeRVVTargetRouteError(
          "computed-mask widening dot-reduction target artifact consumer "
          "requires provider-derived compare, masked product, and merge "
          "statement facts before artifact export");
    if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.compareIntrinsic) ||
        !routeLoopContainsCallee(loop,
                                 description.maskedWideningProductIntrinsic) ||
        !routeLoopContainsCallee(loop, description.maskedMergeIntrinsic))
      return makeRVVTargetRouteError(
          "computed-mask widening dot-reduction target artifact consumer "
          "requires provider-built compare, masked product, and merge "
          "statements before artifact export");
  } else {
    if (description.wideningProductIntrinsic.empty() ||
        !routeLoopContainsCallee(loop, description.wideningProductIntrinsic))
      return makeRVVTargetRouteError(
          "widening dot-reduction target artifact consumer requires "
          "provider-built widening product statement facts before artifact "
          "export");
  }

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

llvm::Error validateRVVMAccRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getCallOpaqueSteps().empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-built pre-loop setvl "
        "statement facts before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires rebuilt provider route "
        "pre-loop setvl statement to define the full-chunk VL");
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
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-built loop bounds "
        "and step to mirror runtime AVL/VL route facts");
  const support::RuntimeABIParameter *runtimeElementCount =
      findRuntimeElementCountABIParameter(description);
  if (!runtimeElementCount)
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires a provider-derived runtime "
        "n/AVL ABI parameter before artifact export");
  if (loop.upperBound.expression != runtimeElementCount->cName ||
      loop.upperBound.cType != runtimeElementCount->cType)
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-built loop upper "
        "bound to use the runtime n/AVL ABI parameter");
  if (loop.bodySteps.empty() ||
      loop.bodySteps.front().callee != description.setVLIntrinsic ||
      !stepHasResult(loop.bodySteps.front(), description.emitCLoopVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-built loop setvl "
        "statement to define per-iteration VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "MAcc target artifact consumer requires loop statements to carry "
          "selected typed RVV source provenance");

  if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
      !routeLoopContainsCallee(loop, description.intrinsic) ||
      !routeLoopContainsCallee(loop, description.storeIntrinsic))
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-built vector load, "
        "MAcc arithmetic, and store statements before artifact export");

  if (isRVVScalarBroadcastMAccRouteFamilyOperation(description.operation) &&
      !routeLoopContainsCallee(loop, description.rhsBroadcastIntrinsic))
    return makeRVVTargetRouteError(
        "scalar-broadcast MAcc target artifact consumer requires "
        "provider-built RHS scalar broadcast statement facts before artifact "
        "export");

  if (isRVVComputedMaskMAccRouteFamilyOperation(description.operation)) {
    if (!routeLoopContainsCallee(loop, description.compareIntrinsic))
      return makeRVVTargetRouteError(
          "computed-mask MAcc target artifact consumer requires "
          "provider-built compare statement facts before artifact export");
    if (description.maskedMergeIntrinsic.empty() ||
        !routeLoopContainsCallee(loop, description.maskedMergeIntrinsic))
      return makeRVVTargetRouteError(
          "computed-mask MAcc target artifact consumer requires "
          "provider-built masked-merge statement facts before artifact export");
    if (isRVVRuntimeScalarComputedMaskMAccRouteFamilyOperation(
            description.operation) &&
        !routeLoopContainsCallee(loop, description.rhsBroadcastIntrinsic))
      return makeRVVTargetRouteError(
          "runtime-scalar computed-mask MAcc target artifact consumer requires "
          "provider-built RHS scalar splat statement facts before artifact "
          "export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVMAccRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
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

  if (description.operation ==
      plugin::rvv::RVVSelectedBodyOperationKind::MAccAdd) {
    if (description.plainMAccRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          "plain MAcc target artifact consumer requires a provider-derived "
          "plain MAcc route-family plan mirror before artifact export");
  } else if (isRVVScalarBroadcastMAccRouteFamilyOperation(
                 description.operation)) {
    if (description.scalarBroadcastMAccRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          "scalar-broadcast MAcc target artifact consumer requires a "
          "provider-derived scalar-broadcast MAcc route-family plan mirror "
          "before artifact export");
    if (description.rhsBroadcastIntrinsic.empty())
      return makeRVVTargetRouteError(
          "scalar-broadcast MAcc target artifact consumer requires a "
          "provider-derived RHS scalar broadcast intrinsic before artifact "
          "export");
  } else if (isRVVComputedMaskMAccRouteFamilyOperation(
                 description.operation)) {
    if (description.accumulationRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          "computed-mask MAcc target artifact consumer requires a "
          "provider-derived accumulation route-family plan mirror before "
          "artifact export");
    if (description.accumulationComputeSuffix.empty() ||
        description.accumulationMaskProducerSource.empty() ||
        description.accumulationAccumulatorContract.empty() ||
        description.accumulationResultContract.empty() ||
        description.maskRole.empty() || description.maskSource.empty() ||
        description.maskMemoryForm.empty() ||
        description.inactiveLaneContract.empty() ||
        description.maskedPassthroughLayout.empty() ||
        description.comparePredicateKind.empty() ||
        description.compareIntrinsic.empty() ||
        description.maskedMergeIntrinsic.empty())
      return makeRVVTargetRouteError(
          "computed-mask MAcc target artifact consumer requires "
          "provider-derived mask, passthrough, compare, accumulator, "
          "and result facts before artifact export");
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
  }

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
constexpr llvm::StringLiteral kRVVStandaloneReductionRouteFamilyPlanID(
    "rvv-standalone-reduction-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVStandaloneReductionScalarResultBoundary(
    "scalar-result-out0-seeded-before-loop-and-carried-across-runtime-vl-chunks.v1");
constexpr llvm::StringLiteral kRVVStandaloneReductionTargetLeafProfile(
    "rvv-v1-typed-standalone-reduction-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVStandaloneReductionProviderSupportedMirror(
    "provider_supported_mirror:rvv-standalone-reduction-plan-validated");
constexpr llvm::StringLiteral
    kRVVComputedMaskStandaloneReductionProviderSupportedMirror(
        "provider_supported_mirror:rvv-computed-mask-standalone-reduction-plan-validated");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskStandaloneReductionProviderSupportedMirror(
        "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-standalone-reduction-plan-validated");
constexpr llvm::StringLiteral kRVVStandaloneReductionHeaderDeclarations(
    "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVStandaloneReductionCTypeMappingSummary(
    "vl:size_t,input:typed-source-vector,seed:typed-scalar,result:typed-scalar-reduction-vector");
constexpr llvm::StringLiteral kRVVComputedMaskStandaloneReductionCTypeMappingSummary(
    "vl:size_t,compare/source:typed-source-vector,mask:typed-mask,seed:typed-scalar,result:typed-scalar-reduction-vector");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskStandaloneReductionCTypeMappingSummary(
        "vl:size_t,cmp_lhs/source:typed-source-vector,rhs_scalar:typed-scalar,mask:typed-mask,seed:typed-scalar,result:typed-scalar-reduction-vector");
constexpr llvm::StringLiteral kRVVStandaloneReductionAccumulatorLayout(
    "scalar-i32-seed-lane0-from-accumulator-input");
constexpr llvm::StringLiteral kRVVStandaloneReductionResultLayout(
    "store-standalone-reduction-lane0-to-output-scalar");
constexpr llvm::StringLiteral kRVVStandaloneReductionStoreVL("1");
constexpr llvm::StringLiteral kRVVComputedMaskStandaloneReductionTargetLeafProfile(
    "rvv-v1-typed-computed-mask-standalone-reduction-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVComputedMaskStandaloneReductionRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,src,acc,out,n");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskStandaloneReductionTargetLeafProfile(
        "rvv-v1-typed-runtime-scalar-cmp-masked-standalone-reduction-leaf-profile.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskStandaloneReductionRuntimeABIOrder(
        "cmp_lhs,rhs_scalar,src,acc,out,n");
constexpr llvm::StringLiteral kRVVComputedMaskStandaloneReductionMaskRole(
    "predicate-mask-produced-by-compare");
constexpr llvm::StringLiteral kRVVComputedMaskStandaloneReductionMaskSource(
    "compare-produced-mask-same-vl-scope");
constexpr llvm::StringLiteral kRVVComputedMaskStandaloneReductionMaskMemoryForm(
    "compare-produced-mask");
constexpr llvm::StringLiteral
    kRVVComputedMaskStandaloneReductionInactiveLaneNeutralRequirement(
        "masked-standalone-reduction-neutral-inactive-lanes-before-reduction");
constexpr llvm::StringLiteral kRVVComputedMaskStandaloneReductionComparePredicate(
    "sle");
constexpr llvm::StringLiteral kRVVComputedMaskAccumulationRouteFamilyPlanID(
    "rvv-computed-mask-accumulation-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVComputedMaskStandaloneReductionComputeSuffix(
    "scalar-horizontal-masked-standalone-reduction");
constexpr llvm::StringLiteral kRVVComputedMaskStandaloneReductionProducerSource(
    "vector-compare-rhs-load");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskStandaloneReductionProducerSource(
        "runtime-scalar-splat-compare-rhs");
constexpr llvm::StringLiteral
    kRVVComputedMaskStandaloneReductionAccumulatorContract(
        "scalar-seed-input-feeds-masked-horizontal-reduction");
constexpr llvm::StringLiteral kRVVComputedMaskStandaloneReductionResultContract(
    "scalar-horizontal-reduction-lane0-stored-to-output");
constexpr llvm::StringLiteral
    kRVVComputedMaskStandaloneReductionScalarCarryContract(
        "scalar-result-carries-across-runtime-vl-chunks");

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

std::string getRVVPlainStandaloneReductionExpectedBindingSummary(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  llvm::StringRef planID =
      getRVVPlainStandaloneReductionExpectedBindingPlanID(operation);
  if (planID.empty())
    return {};
  return (llvm::Twine(planID) +
          ";lhs=lhs-input-buffer:lhs:runtime-abi-mirror|materialized-load-base|standalone-reduction-input-call;"
          "acc=accumulator-input-buffer:acc:runtime-abi-mirror|standalone-initial-accumulator-call;"
          "out=output-buffer:out:runtime-abi-mirror|standalone-accumulator-state-load|materialized-store-base|header-mirror;"
          "n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror")
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
          "acc=accumulator-input-buffer:acc:abi|initial-seed|acc-state|masked-reduce-acc;"
          "out=output-buffer:out:abi|acc-state|store-base|hdr;"
          "n=runtime-element-count:n:abi|setvl-avl|loop|hdr")
      .str();
}

llvm::StringRef
getRVVRuntimeScalarComputedMaskStandaloneReductionExpectedBindingPlanID(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceAdd:
    return "rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_add.v1";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMin:
    return "rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_min.v1";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMax:
    return "rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_max.v1";
  default:
    return {};
  }
}

std::string
getRVVRuntimeScalarComputedMaskStandaloneReductionExpectedBindingSummary(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  llvm::StringRef planID =
      getRVVRuntimeScalarComputedMaskStandaloneReductionExpectedBindingPlanID(
          operation);
  if (planID.empty())
    return {};
  const llvm::StringRef inactiveUse =
      operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       RuntimeScalarComputedMaskStandaloneReduceAdd
          ? llvm::StringRef("zero-inactive")
          : llvm::StringRef("neutral-inactive");
  return (llvm::Twine(planID) +
          ";cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|cmp-lhs-call|hdr;"
          "rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs-call|hdr;"
          "src=source-input-buffer:src:abi|src-load|masked-reduce-input|" +
          inactiveUse +
          "|hdr;"
          "acc=accumulator-input-buffer:acc:abi|initial-seed|acc-state|masked-reduce-acc;"
          "out=output-buffer:out:abi|acc-state|store-base|hdr;"
          "n=runtime-element-count:n:abi|setvl-avl|loop|hdr")
      .str();
}

llvm::StringRef getRVVPlainStandaloneReductionExpectedVectorTypeName(
    int64_t sew, llvm::StringRef lmul) {
  if (sew == 64 && lmul == "m1")
    return "!tcrv_rvv.vector<i64, \"m1\">";
  if (sew != 32)
    return {};
  if (lmul == "m1")
    return "!tcrv_rvv.vector<i32, \"m1\">";
  if (lmul == "m2")
    return "!tcrv_rvv.vector<i32, \"m2\">";
  return {};
}

llvm::StringRef getRVVPlainStandaloneReductionExpectedVectorCType(
    int64_t sew, llvm::StringRef lmul) {
  if (sew == 64 && lmul == "m1")
    return "vint64m1_t";
  if (sew != 32)
    return {};
  if (lmul == "m1")
    return "vint32m1_t";
  if (lmul == "m2")
    return "vint32m2_t";
  return {};
}

llvm::StringRef getRVVComputedMaskStandaloneReductionExpectedMaskTypeName(
    int64_t sew, llvm::StringRef lmul) {
  if (sew == 64 && lmul == "m1")
    return "!tcrv_rvv.mask<i64, \"m1\">";
  if (sew != 32)
    return {};
  if (lmul == "m1")
    return "!tcrv_rvv.mask<i32, \"m1\">";
  if (lmul == "m2")
    return "!tcrv_rvv.mask<i32, \"m2\">";
  return {};
}

llvm::StringRef getRVVComputedMaskStandaloneReductionExpectedMaskCType(
    int64_t sew, llvm::StringRef lmul) {
  if (sew == 64 && lmul == "m1")
    return "vbool64_t";
  if (sew != 32)
    return {};
  if (lmul == "m1")
    return "vbool32_t";
  if (lmul == "m2")
    return "vbool16_t";
  return {};
}

llvm::StringRef getRVVPlainStandaloneReductionExpectedScalarResultVectorTypeName(
    int64_t sew, llvm::StringRef lmul) {
  if (sew == 64 && lmul == "m1")
    return "!tcrv_rvv.vector<i64, \"m1\">";
  if (sew == 32 && (lmul == "m1" || lmul == "m2"))
    return "!tcrv_rvv.vector<i32, \"m1\">";
  return {};
}

llvm::StringRef getRVVPlainStandaloneReductionExpectedScalarResultVectorCType(
    int64_t sew, llvm::StringRef lmul) {
  if (sew == 64 && lmul == "m1")
    return "vint64m1_t";
  if (sew == 32 && (lmul == "m1" || lmul == "m2"))
    return "vint32m1_t";
  return {};
}

llvm::StringRef getRVVPlainStandaloneReductionExpectedSeedSplatIntrinsic(
    int64_t sew, llvm::StringRef lmul) {
  if (sew == 64 && lmul == "m1")
    return "__riscv_vmv_v_x_i64m1";
  if (sew == 32 && (lmul == "m1" || lmul == "m2"))
    return "__riscv_vmv_v_x_i32m1";
  return {};
}

llvm::StringRef getRVVPlainStandaloneReductionExpectedStoreIntrinsic(
    int64_t sew, llvm::StringRef lmul) {
  if (sew == 64 && lmul == "m1")
    return "__riscv_vse64_v_i64m1";
  if (sew == 32 && (lmul == "m1" || lmul == "m2"))
    return "__riscv_vse32_v_i32m1";
  return {};
}

llvm::StringRef
getRVVRuntimeScalarComputedMaskStandaloneReductionExpectedRHSBroadcastIntrinsic(
    int64_t sew, llvm::StringRef lmul) {
  if (sew == 64 && lmul == "m1")
    return "__riscv_vmv_v_x_i64m1";
  if (sew != 32)
    return {};
  if (lmul == "m1")
    return "__riscv_vmv_v_x_i32m1";
  if (lmul == "m2")
    return "__riscv_vmv_v_x_i32m2";
  return {};
}

llvm::StringRef getRVVComputedMaskStandaloneReductionExpectedCompareIntrinsic(
    int64_t sew, llvm::StringRef lmul) {
  if (sew == 64 && lmul == "m1")
    return "__riscv_vmsle_vv_i64m1_b64";
  if (sew != 32)
    return {};
  if (lmul == "m1")
    return "__riscv_vmsle_vv_i32m1_b32";
  if (lmul == "m2")
    return "__riscv_vmsle_vv_i32m2_b16";
  return {};
}

llvm::StringRef getRVVComputedMaskStandaloneReductionExpectedMergeIntrinsic(
    int64_t sew, llvm::StringRef lmul) {
  if (sew == 64 && lmul == "m1")
    return "__riscv_vmerge_vvm_i64m1";
  if (sew != 32)
    return {};
  if (lmul == "m1")
    return "__riscv_vmerge_vvm_i32m1";
  if (lmul == "m2")
    return "__riscv_vmerge_vvm_i32m2";
  return {};
}

llvm::StringRef
getRVVComputedMaskStandaloneReductionExpectedInactiveLaneRequirement(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
                 plugin::rvv::RVVSelectedBodyOperationKind::
                     ComputedMaskStandaloneReduceAdd
             ? llvm::StringRef(
                   "masked-standalone-reduction-zero-inactive-lanes-before-reduction")
             : llvm::StringRef(
                   kRVVComputedMaskStandaloneReductionInactiveLaneNeutralRequirement);
}

llvm::StringRef getRVVPlainStandaloneReductionExpectedIntrinsic(
    plugin::rvv::RVVSelectedBodyOperationKind operation, int64_t sew,
    llvm::StringRef lmul) {
  if (sew != 32)
    return {};
  const bool usesM2 = lmul == "m2";
  if (lmul != "m1" && !usesM2)
    return {};
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::StandaloneReduceAdd:
    return usesM2 ? "__riscv_vredsum_vs_i32m2_i32m1"
                  : "__riscv_vredsum_vs_i32m1_i32m1";
  case plugin::rvv::RVVSelectedBodyOperationKind::StandaloneReduceMin:
    return usesM2 ? "__riscv_vredmin_vs_i32m2_i32m1"
                  : "__riscv_vredmin_vs_i32m1_i32m1";
  case plugin::rvv::RVVSelectedBodyOperationKind::StandaloneReduceMax:
    return usesM2 ? "__riscv_vredmax_vs_i32m2_i32m1"
                  : "__riscv_vredmax_vs_i32m1_i32m1";
  default:
    return {};
  }
}

llvm::StringRef getRVVComputedMaskStandaloneReductionExpectedIntrinsic(
    plugin::rvv::RVVSelectedBodyOperationKind operation, int64_t sew,
    llvm::StringRef lmul) {
  if (sew != 32)
    return {};
  const bool usesM2 = lmul == "m2";
  if (lmul != "m1" && !usesM2)
    return {};
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd:
    return usesM2 ? "__riscv_vredsum_vs_i32m2_i32m1"
                  : "__riscv_vredsum_vs_i32m1_i32m1";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin:
    return usesM2 ? "__riscv_vredmin_vs_i32m2_i32m1"
                  : "__riscv_vredmin_vs_i32m1_i32m1";
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax:
    return usesM2 ? "__riscv_vredmax_vs_i32m2_i32m1"
                  : "__riscv_vredmax_vs_i32m1_i32m1";
  default:
    return {};
  }
}

llvm::StringRef
getRVVRuntimeScalarComputedMaskStandaloneReductionExpectedIntrinsic(
    plugin::rvv::RVVSelectedBodyOperationKind operation, int64_t sew,
    llvm::StringRef lmul) {
  if (sew == 64 && lmul == "m1") {
    if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                         RuntimeScalarComputedMaskStandaloneReduceAdd)
      return "__riscv_vredsum_vs_i64m1_i64m1";
    return {};
  }
  if (sew != 32)
    return {};
  const bool usesM2 = lmul == "m2";
  if (lmul != "m1" && !usesM2)
    return {};
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceAdd:
    return usesM2 ? "__riscv_vredsum_vs_i32m2_i32m1"
                  : "__riscv_vredsum_vs_i32m1_i32m1";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMin:
    return usesM2 ? "__riscv_vredmin_vs_i32m2_i32m1"
                  : "__riscv_vredmin_vs_i32m1_i32m1";
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMax:
    return usesM2 ? "__riscv_vredmax_vs_i32m2_i32m1"
                  : "__riscv_vredmax_vs_i32m1_i32m1";
  default:
    return {};
  }
}

llvm::StringRef getRVVRuntimeScalarComputedMaskStandaloneReductionExpectedTypeName(
    int64_t sew) {
  if (sew == 64)
    return "i64";
  if (sew == 32)
    return "i32";
  return {};
}

llvm::StringRef
getRVVRuntimeScalarComputedMaskStandaloneReductionExpectedElementCType(
    int64_t sew) {
  if (sew == 64)
    return "int64_t";
  if (sew == 32)
    return "int32_t";
  return {};
}

llvm::StringRef
getRVVRuntimeScalarComputedMaskStandaloneReductionExpectedAccumulatorLayout(
    int64_t sew) {
  if (sew == 64)
    return "scalar-i64-seed-lane0-from-accumulator-input";
  if (sew == 32)
    return kRVVStandaloneReductionAccumulatorLayout;
  return {};
}

bool isRVVRuntimeScalarComputedMaskStandaloneReductionSupportedConfig(
    plugin::rvv::RVVSelectedBodyOperationKind operation, int64_t sew,
    llvm::StringRef lmul) {
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       RuntimeScalarComputedMaskStandaloneReduceAdd) {
    return (sew == 32 && (lmul == "m1" || lmul == "m2")) ||
           (sew == 64 && lmul == "m1");
  }
  return sew == 32 && (lmul == "m1" || lmul == "m2");
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
    llvm::StringRef cType;
    support::RuntimeABIParameterRole role;
  };
  const ExpectedRuntimeABIParameter expected[] = {
      {"lhs", "const int32_t *",
       support::RuntimeABIParameterRole::LHSInputBuffer},
      {"acc", "const int32_t *",
       support::RuntimeABIParameterRole::AccumulatorInputBuffer},
      {"out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer},
      {"n", "size_t", support::RuntimeABIParameterRole::RuntimeElementCount},
  };
  constexpr size_t expectedCount = sizeof(expected) / sizeof(expected[0]);
  for (size_t index = 0; index < expectedCount; ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    if (actual.cName != expected[index].cName ||
        actual.cType != expected[index].cType ||
        actual.role != expected[index].role)
      return makeRVVTargetRouteError(
          llvm::Twine("plain standalone reduction target artifact consumer "
                      "requires provider-derived runtime ABI parameter ") +
          std::to_string(index) + " to bind " + expected[index].cName +
          " as " +
          support::stringifyRuntimeABIParameterRole(expected[index].role) +
          " with C type '" + expected[index].cType +
          "' before artifact export");
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
    llvm::StringRef cType;
    support::RuntimeABIParameterRole role;
  };
  const ExpectedRuntimeABIParameter expected[] = {
      {"cmp_lhs", "const int32_t *",
       support::RuntimeABIParameterRole::LHSInputBuffer},
      {"cmp_rhs", "const int32_t *",
       support::RuntimeABIParameterRole::RHSInputBuffer},
      {"src", "const int32_t *",
       support::RuntimeABIParameterRole::SourceInputBuffer},
      {"acc", "const int32_t *",
       support::RuntimeABIParameterRole::AccumulatorInputBuffer},
      {"out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer},
      {"n", "size_t", support::RuntimeABIParameterRole::RuntimeElementCount},
  };
  constexpr size_t expectedCount = sizeof(expected) / sizeof(expected[0]);
  for (size_t index = 0; index < expectedCount; ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    if (actual.cName != expected[index].cName ||
        actual.cType != expected[index].cType ||
        actual.role != expected[index].role)
      return makeRVVTargetRouteError(
          llvm::Twine("computed-mask standalone reduction target artifact "
                      "consumer requires provider-derived runtime ABI "
                      "parameter ") +
          std::to_string(index) + " to bind " + expected[index].cName +
          " as " +
          support::stringifyRuntimeABIParameterRole(expected[index].role) +
          " with C type '" + expected[index].cType +
          "' before artifact export");
  }
  return llvm::Error::success();
}

llvm::Error
validateRVVRuntimeScalarComputedMaskStandaloneReductionRuntimeABIFacts(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.runtimeABIOrder !=
      kRVVRuntimeScalarComputedMaskStandaloneReductionRuntimeABIOrder)
    return makeRVVTargetRouteError(
        llvm::Twine(
            "runtime-scalar computed-mask standalone reduction target artifact "
            "consumer requires provider-derived runtime ABI order '") +
        kRVVRuntimeScalarComputedMaskStandaloneReductionRuntimeABIOrder +
        "' but was '" + description.runtimeABIOrder + "'");
  if (description.runtimeABIParameters.size() != 6)
    return makeRVVTargetRouteError(
        "runtime-scalar computed-mask standalone reduction target artifact "
        "consumer requires provider-derived runtime ABI parameters for "
        "cmp_lhs, rhs_scalar, src, acc scalar seed, out scalar result, and n "
        "before artifact export");

  struct ExpectedRuntimeABIParameter {
    llvm::StringRef cName;
    llvm::StringRef cType;
    support::RuntimeABIParameterRole role;
  };
  const llvm::StringRef elementCType =
      getRVVRuntimeScalarComputedMaskStandaloneReductionExpectedElementCType(
          description.sew);
  const std::string constElementPointerCType =
      (llvm::Twine("const ") + elementCType + " *").str();
  const std::string elementPointerCType =
      (llvm::Twine(elementCType) + " *").str();
  const ExpectedRuntimeABIParameter expected[] = {
      {"cmp_lhs", constElementPointerCType,
       support::RuntimeABIParameterRole::LHSInputBuffer},
      {"rhs_scalar", elementCType,
       support::RuntimeABIParameterRole::RHSScalarValue},
      {"src", constElementPointerCType,
       support::RuntimeABIParameterRole::SourceInputBuffer},
      {"acc", constElementPointerCType,
       support::RuntimeABIParameterRole::AccumulatorInputBuffer},
      {"out", elementPointerCType,
       support::RuntimeABIParameterRole::OutputBuffer},
      {"n", "size_t", support::RuntimeABIParameterRole::RuntimeElementCount},
  };
  constexpr size_t expectedCount = sizeof(expected) / sizeof(expected[0]);
  for (size_t index = 0; index < expectedCount; ++index) {
    const support::RuntimeABIParameter &actual =
        description.runtimeABIParameters[index];
    if (actual.cName != expected[index].cName ||
        actual.cType != expected[index].cType ||
        actual.role != expected[index].role)
      return makeRVVTargetRouteError(
          llvm::Twine(
              "runtime-scalar computed-mask standalone reduction target "
              "artifact consumer requires provider-derived runtime ABI "
              "parameter ") +
          std::to_string(index) + " to bind " + expected[index].cName +
          " as " +
          support::stringifyRuntimeABIParameterRole(expected[index].role) +
          " with C type '" + expected[index].cType +
          "' before artifact export");
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
  if (description.elementTypeName != "i32" || description.sew != 32 ||
      (description.lmul != "m1" && description.lmul != "m2"))
    return makeRVVTargetRouteError(
        llvm::Twine("plain standalone reduction target artifact consumer "
                    "requires provider-derived signed i32 SEW32 LMUL m1/m2 "
                    "dtype/config facts but saw element '") +
        description.elementTypeName + "', SEW " +
        llvm::Twine(description.sew) + ", LMUL '" + description.lmul + "'");

  llvm::StringRef expectedVectorType =
      getRVVPlainStandaloneReductionExpectedVectorTypeName(description.sew,
                                                          description.lmul);
  llvm::StringRef expectedVectorCType =
      getRVVPlainStandaloneReductionExpectedVectorCType(description.sew,
                                                       description.lmul);
  llvm::StringRef expectedScalarResultVectorType =
      getRVVPlainStandaloneReductionExpectedScalarResultVectorTypeName(
          description.sew, description.lmul);
  llvm::StringRef expectedScalarResultVectorCType =
      getRVVPlainStandaloneReductionExpectedScalarResultVectorCType(
          description.sew, description.lmul);
  if (description.vectorTypeName != expectedVectorType ||
      description.vectorCType != expectedVectorCType ||
      description.standaloneReductionSourceVectorTypeName !=
          expectedVectorType ||
      description.standaloneReductionSourceVectorCType !=
          expectedVectorCType ||
      description.standaloneReductionScalarResultVectorTypeName !=
          expectedScalarResultVectorType ||
      description.standaloneReductionScalarResultVectorCType !=
          expectedScalarResultVectorCType)
    return makeRVVTargetRouteError(
        llvm::Twine("plain standalone reduction target artifact consumer "
                    "requires provider-derived source vector type '") +
        expectedVectorType + "' / '" + expectedVectorCType +
        "' and scalar-result vector type '" + expectedScalarResultVectorType +
        "' / '" + expectedScalarResultVectorCType +
        "' before artifact export");

  if (description.providerSupportedMirror !=
          kRVVStandaloneReductionProviderSupportedMirror ||
      description.targetLeafProfile != kRVVStandaloneReductionTargetLeafProfile ||
      description.standaloneReductionRouteFamilyPlanID !=
          kRVVStandaloneReductionRouteFamilyPlanID ||
      description.standaloneReductionScalarResultRuntimeBoundary !=
          kRVVStandaloneReductionScalarResultBoundary)
    return makeRVVTargetRouteError(
        llvm::Twine("plain standalone reduction target artifact consumer "
                    "requires provider mirror '") +
        kRVVStandaloneReductionProviderSupportedMirror + "', target leaf '" +
        kRVVStandaloneReductionTargetLeafProfile + "', route-family plan '" +
        kRVVStandaloneReductionRouteFamilyPlanID +
        "', and scalar-result boundary '" +
        kRVVStandaloneReductionScalarResultBoundary +
        "' before artifact export");

  if (description.requiredHeaderDeclarations !=
          kRVVStandaloneReductionHeaderDeclarations ||
      description.cTypeMappingSummary != kRVVStandaloneReductionCTypeMappingSummary)
    return makeRVVTargetRouteError(
        llvm::Twine("plain standalone reduction target artifact consumer "
                    "requires provider-derived header declarations '") +
        kRVVStandaloneReductionHeaderDeclarations + "' and C type mapping '" +
        kRVVStandaloneReductionCTypeMappingSummary +
        "' before artifact export");

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

  if (description.reductionAccumulatorLayout !=
          kRVVStandaloneReductionAccumulatorLayout ||
      description.reductionResultLayout != kRVVStandaloneReductionResultLayout ||
      description.reductionStoreVL != kRVVStandaloneReductionStoreVL)
    return makeRVVTargetRouteError(
        llvm::Twine("plain standalone reduction target artifact consumer "
                    "requires scalar seed accumulator layout '") +
        kRVVStandaloneReductionAccumulatorLayout + "', result layout '" +
        kRVVStandaloneReductionResultLayout + "', and store VL '" +
        kRVVStandaloneReductionStoreVL + "' before artifact export");

  llvm::StringRef expectedSeedSplat =
      getRVVPlainStandaloneReductionExpectedSeedSplatIntrinsic(description.sew,
                                                              description.lmul);
  llvm::StringRef expectedStore =
      getRVVPlainStandaloneReductionExpectedStoreIntrinsic(description.sew,
                                                          description.lmul);
  llvm::StringRef expectedReduction =
      getRVVPlainStandaloneReductionExpectedIntrinsic(
          description.operation, description.sew, description.lmul);
  if (description.scalarSeedSplatIntrinsic != expectedSeedSplat ||
      description.storeIntrinsic != expectedStore ||
      description.intrinsic != expectedReduction)
    return makeRVVTargetRouteError(
        llvm::Twine("plain standalone reduction target artifact consumer "
                    "requires provider-derived scalar seed splat '") +
        expectedSeedSplat + "', signed min/max/add reduction intrinsic '" +
        expectedReduction + "', and scalar result store '" + expectedStore +
        "' before artifact export");

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
  if (description.elementTypeName != "i32" || description.sew != 32 ||
      (description.lmul != "m1" && description.lmul != "m2"))
    return makeRVVTargetRouteError(
        llvm::Twine("computed-mask standalone reduction target artifact "
                    "consumer requires provider-derived signed i32 SEW32 "
                    "LMUL m1/m2 dtype/config facts but saw element '") +
        description.elementTypeName + "', SEW " +
        llvm::Twine(description.sew) + ", LMUL '" + description.lmul + "'");
  if (description.tailPolicy != "agnostic" ||
      description.maskPolicy != "agnostic")
    return makeRVVTargetRouteError(
        llvm::Twine("computed-mask standalone reduction target artifact "
                    "consumer requires provider-derived agnostic tail/mask "
                    "policy but saw tail '") +
        description.tailPolicy + "', mask '" + description.maskPolicy + "'");

  llvm::StringRef expectedVectorType =
      getRVVPlainStandaloneReductionExpectedVectorTypeName(description.sew,
                                                          description.lmul);
  llvm::StringRef expectedVectorCType =
      getRVVPlainStandaloneReductionExpectedVectorCType(description.sew,
                                                       description.lmul);
  llvm::StringRef expectedScalarResultVectorType =
      getRVVPlainStandaloneReductionExpectedScalarResultVectorTypeName(
          description.sew, description.lmul);
  llvm::StringRef expectedScalarResultVectorCType =
      getRVVPlainStandaloneReductionExpectedScalarResultVectorCType(
          description.sew, description.lmul);
  llvm::StringRef expectedMaskType =
      getRVVComputedMaskStandaloneReductionExpectedMaskTypeName(
          description.sew, description.lmul);
  llvm::StringRef expectedMaskCType =
      getRVVComputedMaskStandaloneReductionExpectedMaskCType(description.sew,
                                                            description.lmul);
  if (description.vectorTypeName != expectedVectorType ||
      description.vectorCType != expectedVectorCType ||
      description.standaloneReductionSourceVectorTypeName !=
          expectedVectorType ||
      description.standaloneReductionSourceVectorCType !=
          expectedVectorCType ||
      description.standaloneReductionScalarResultVectorTypeName !=
          expectedScalarResultVectorType ||
      description.standaloneReductionScalarResultVectorCType !=
          expectedScalarResultVectorCType ||
      description.maskTypeName != expectedMaskType ||
      description.maskCType != expectedMaskCType)
    return makeRVVTargetRouteError(
        llvm::Twine("computed-mask standalone reduction target artifact "
                    "consumer requires provider-derived source vector type '") +
        expectedVectorType + "' / '" + expectedVectorCType +
        "', scalar-result vector type '" + expectedScalarResultVectorType +
        "' / '" + expectedScalarResultVectorCType + "', and mask type '" +
        expectedMaskType + "' / '" + expectedMaskCType +
        "' before artifact export");

  if (description.providerSupportedMirror !=
          kRVVComputedMaskStandaloneReductionProviderSupportedMirror ||
      description.targetLeafProfile !=
          kRVVComputedMaskStandaloneReductionTargetLeafProfile ||
      description.standaloneReductionRouteFamilyPlanID !=
          kRVVStandaloneReductionRouteFamilyPlanID ||
      description.standaloneReductionScalarResultRuntimeBoundary !=
          kRVVStandaloneReductionScalarResultBoundary ||
      description.requiredHeaderDeclarations !=
          kRVVStandaloneReductionHeaderDeclarations ||
      description.cTypeMappingSummary !=
          kRVVComputedMaskStandaloneReductionCTypeMappingSummary)
    return makeRVVTargetRouteError(
        llvm::Twine("computed-mask standalone reduction target artifact "
                    "consumer requires provider mirror '") +
        kRVVComputedMaskStandaloneReductionProviderSupportedMirror +
        "', target leaf '" +
        kRVVComputedMaskStandaloneReductionTargetLeafProfile +
        "', standalone route-family plan '" +
        kRVVStandaloneReductionRouteFamilyPlanID +
        "', scalar-result boundary '" +
        kRVVStandaloneReductionScalarResultBoundary + "', headers '" +
        kRVVStandaloneReductionHeaderDeclarations + "', and C type mapping '" +
        kRVVComputedMaskStandaloneReductionCTypeMappingSummary +
        "' before artifact export");

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

  if (description.reductionAccumulatorLayout !=
          kRVVStandaloneReductionAccumulatorLayout ||
      description.reductionResultLayout != kRVVStandaloneReductionResultLayout ||
      description.reductionStoreVL != kRVVStandaloneReductionStoreVL)
    return makeRVVTargetRouteError(
        llvm::Twine("computed-mask standalone reduction target artifact "
                    "consumer requires scalar seed accumulator layout '") +
        kRVVStandaloneReductionAccumulatorLayout + "', result layout '" +
        kRVVStandaloneReductionResultLayout + "', and store VL '" +
        kRVVStandaloneReductionStoreVL + "' before artifact export");

  llvm::StringRef expectedSeedSplat =
      getRVVPlainStandaloneReductionExpectedSeedSplatIntrinsic(description.sew,
                                                              description.lmul);
  llvm::StringRef expectedStore =
      getRVVPlainStandaloneReductionExpectedStoreIntrinsic(description.sew,
                                                          description.lmul);
  llvm::StringRef expectedReduction =
      getRVVComputedMaskStandaloneReductionExpectedIntrinsic(
          description.operation, description.sew, description.lmul);
  llvm::StringRef expectedCompare =
      getRVVComputedMaskStandaloneReductionExpectedCompareIntrinsic(
          description.sew, description.lmul);
  llvm::StringRef expectedMerge =
      getRVVComputedMaskStandaloneReductionExpectedMergeIntrinsic(
          description.sew, description.lmul);
  if (description.scalarSeedSplatIntrinsic != expectedSeedSplat ||
      description.storeIntrinsic != expectedStore ||
      description.intrinsic != expectedReduction ||
      description.compareIntrinsic != expectedCompare ||
      description.maskedMergeIntrinsic != expectedMerge)
    return makeRVVTargetRouteError(
        llvm::Twine("computed-mask standalone reduction target artifact "
                    "consumer requires provider-derived scalar seed splat '") +
        expectedSeedSplat + "', signed min/max/add reduction intrinsic '" +
        expectedReduction + "', compare intrinsic '" + expectedCompare +
        "', inactive neutral merge '" + expectedMerge +
        "', and scalar result store '" + expectedStore +
        "' before artifact export");

  llvm::StringRef expectedInactiveRequirement =
      getRVVComputedMaskStandaloneReductionExpectedInactiveLaneRequirement(
          description.operation);
  if (description.maskRole != kRVVComputedMaskStandaloneReductionMaskRole ||
      description.maskSource != kRVVComputedMaskStandaloneReductionMaskSource ||
      description.maskMemoryForm !=
          kRVVComputedMaskStandaloneReductionMaskMemoryForm ||
      description.inactiveLaneZeroingRequirement != expectedInactiveRequirement ||
      description.comparePredicateKind !=
          kRVVComputedMaskStandaloneReductionComparePredicate)
    return makeRVVTargetRouteError(
        llvm::Twine("computed-mask standalone reduction target artifact "
                    "consumer requires provider-derived mask role/source/form, "
                    "operation-specific inactive-lane requirement '") +
        expectedInactiveRequirement +
        "', and compare predicate '" +
        kRVVComputedMaskStandaloneReductionComparePredicate +
        "' before artifact export");

  if (description.accumulationRouteFamilyPlanID !=
          kRVVComputedMaskAccumulationRouteFamilyPlanID ||
      description.accumulationComputeSuffix !=
          kRVVComputedMaskStandaloneReductionComputeSuffix ||
      description.accumulationMaskProducerSource !=
          kRVVComputedMaskStandaloneReductionProducerSource ||
      description.accumulationAccumulatorContract !=
          kRVVComputedMaskStandaloneReductionAccumulatorContract ||
      description.accumulationResultContract !=
          kRVVComputedMaskStandaloneReductionResultContract ||
      description.accumulationScalarCarryContract !=
          kRVVComputedMaskStandaloneReductionScalarCarryContract)
    return makeRVVTargetRouteError(
        "computed-mask standalone reduction target artifact consumer requires "
        "provider-derived computed-mask accumulation plan, scalar horizontal "
        "reduction suffix, vector compare producer, scalar seed/result "
        "contracts, and scalar carry contract before artifact export");

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
  const llvm::StringRef expectedElementType =
      getRVVRuntimeScalarComputedMaskStandaloneReductionExpectedTypeName(
          description.sew);
  if (description.elementTypeName != expectedElementType ||
      !isRVVRuntimeScalarComputedMaskStandaloneReductionSupportedConfig(
          description.operation, description.sew, description.lmul))
    return makeRVVTargetRouteError(
        llvm::Twine("runtime-scalar computed-mask standalone reduction target "
                    "artifact consumer requires provider-derived signed i32 "
                    "SEW32 LMUL m1/m2 facts, or signed i64 SEW64 LMUL m1 "
                    "facts for reduce_add, but saw element '") +
        description.elementTypeName + "', SEW " +
        llvm::Twine(description.sew) + ", LMUL '" + description.lmul + "'");
  if (description.tailPolicy != "agnostic" ||
      description.maskPolicy != "agnostic")
    return makeRVVTargetRouteError(
        llvm::Twine("runtime-scalar computed-mask standalone reduction target "
                    "artifact consumer requires provider-derived agnostic "
                    "tail/mask policy but saw tail '") +
        description.tailPolicy + "', mask '" + description.maskPolicy + "'");

  llvm::StringRef expectedVectorType =
      getRVVPlainStandaloneReductionExpectedVectorTypeName(description.sew,
                                                          description.lmul);
  llvm::StringRef expectedVectorCType =
      getRVVPlainStandaloneReductionExpectedVectorCType(description.sew,
                                                       description.lmul);
  llvm::StringRef expectedScalarResultVectorType =
      getRVVPlainStandaloneReductionExpectedScalarResultVectorTypeName(
          description.sew, description.lmul);
  llvm::StringRef expectedScalarResultVectorCType =
      getRVVPlainStandaloneReductionExpectedScalarResultVectorCType(
          description.sew, description.lmul);
  llvm::StringRef expectedMaskType =
      getRVVComputedMaskStandaloneReductionExpectedMaskTypeName(
          description.sew, description.lmul);
  llvm::StringRef expectedMaskCType =
      getRVVComputedMaskStandaloneReductionExpectedMaskCType(description.sew,
                                                            description.lmul);
  if (description.vectorTypeName != expectedVectorType ||
      description.vectorCType != expectedVectorCType ||
      description.standaloneReductionSourceVectorTypeName !=
          expectedVectorType ||
      description.standaloneReductionSourceVectorCType !=
          expectedVectorCType ||
      description.standaloneReductionScalarResultVectorTypeName !=
          expectedScalarResultVectorType ||
      description.standaloneReductionScalarResultVectorCType !=
          expectedScalarResultVectorCType ||
      description.maskTypeName != expectedMaskType ||
      description.maskCType != expectedMaskCType)
    return makeRVVTargetRouteError(
        llvm::Twine("runtime-scalar computed-mask standalone reduction target "
                    "artifact consumer requires provider-derived source vector "
                    "type '") +
        expectedVectorType + "' / '" + expectedVectorCType +
        "', scalar-result vector type '" + expectedScalarResultVectorType +
        "' / '" + expectedScalarResultVectorCType + "', and mask type '" +
        expectedMaskType + "' / '" + expectedMaskCType +
        "' before artifact export");

  if (description.providerSupportedMirror !=
          kRVVRuntimeScalarComputedMaskStandaloneReductionProviderSupportedMirror ||
      description.targetLeafProfile !=
          kRVVRuntimeScalarComputedMaskStandaloneReductionTargetLeafProfile ||
      description.standaloneReductionRouteFamilyPlanID !=
          kRVVStandaloneReductionRouteFamilyPlanID ||
      description.standaloneReductionScalarResultRuntimeBoundary !=
          kRVVStandaloneReductionScalarResultBoundary ||
      description.requiredHeaderDeclarations !=
          kRVVStandaloneReductionHeaderDeclarations ||
      description.cTypeMappingSummary !=
          kRVVRuntimeScalarComputedMaskStandaloneReductionCTypeMappingSummary)
    return makeRVVTargetRouteError(
        llvm::Twine("runtime-scalar computed-mask standalone reduction target "
                    "artifact consumer requires provider mirror '") +
        kRVVRuntimeScalarComputedMaskStandaloneReductionProviderSupportedMirror +
        "', target leaf '" +
        kRVVRuntimeScalarComputedMaskStandaloneReductionTargetLeafProfile +
        "', standalone route-family plan '" +
        kRVVStandaloneReductionRouteFamilyPlanID +
        "', scalar-result boundary '" +
        kRVVStandaloneReductionScalarResultBoundary + "', headers '" +
        kRVVStandaloneReductionHeaderDeclarations + "', and C type mapping '" +
        kRVVRuntimeScalarComputedMaskStandaloneReductionCTypeMappingSummary +
        "' before artifact export");

  llvm::StringRef expectedBindingPlan =
      getRVVRuntimeScalarComputedMaskStandaloneReductionExpectedBindingPlanID(
          description.operation);
  std::string expectedBindingSummary =
      getRVVRuntimeScalarComputedMaskStandaloneReductionExpectedBindingSummary(
          description.operation);
  if (description.routeOperandBindingPlanID != expectedBindingPlan ||
      description.routeOperandBindingSummary != expectedBindingSummary)
    return makeRVVTargetRouteError(
        llvm::Twine("runtime-scalar computed-mask standalone reduction target "
                    "artifact consumer requires provider-derived route operand "
                    "binding plan '") +
        expectedBindingPlan +
        "' with cmp_lhs, rhs_scalar, src, acc seed, out scalar result, and n "
        "bindings but provider carried plan '" +
        description.routeOperandBindingPlanID + "'");

  if (llvm::Error error =
          validateRVVRuntimeScalarComputedMaskStandaloneReductionRuntimeABIFacts(
              description))
    return error;

  const llvm::StringRef expectedAccumulatorLayout =
      getRVVRuntimeScalarComputedMaskStandaloneReductionExpectedAccumulatorLayout(
          description.sew);
  if (description.reductionAccumulatorLayout != expectedAccumulatorLayout ||
      description.reductionResultLayout != kRVVStandaloneReductionResultLayout ||
      description.reductionStoreVL != kRVVStandaloneReductionStoreVL)
    return makeRVVTargetRouteError(
        llvm::Twine("runtime-scalar computed-mask standalone reduction target "
                    "artifact consumer requires scalar seed accumulator "
                    "layout '") +
        expectedAccumulatorLayout + "', result layout '" +
        kRVVStandaloneReductionResultLayout + "', and store VL '" +
        kRVVStandaloneReductionStoreVL + "' before artifact export");

  llvm::StringRef expectedSeedSplat =
      getRVVPlainStandaloneReductionExpectedSeedSplatIntrinsic(description.sew,
                                                              description.lmul);
  llvm::StringRef expectedRHSBroadcast =
      getRVVRuntimeScalarComputedMaskStandaloneReductionExpectedRHSBroadcastIntrinsic(
          description.sew, description.lmul);
  llvm::StringRef expectedStore =
      getRVVPlainStandaloneReductionExpectedStoreIntrinsic(description.sew,
                                                          description.lmul);
  llvm::StringRef expectedReduction =
      getRVVRuntimeScalarComputedMaskStandaloneReductionExpectedIntrinsic(
          description.operation, description.sew, description.lmul);
  llvm::StringRef expectedCompare =
      getRVVComputedMaskStandaloneReductionExpectedCompareIntrinsic(
          description.sew, description.lmul);
  llvm::StringRef expectedMerge =
      getRVVComputedMaskStandaloneReductionExpectedMergeIntrinsic(
          description.sew, description.lmul);
  if (description.scalarSeedSplatIntrinsic != expectedSeedSplat ||
      description.rhsBroadcastIntrinsic != expectedRHSBroadcast ||
      description.storeIntrinsic != expectedStore ||
      description.intrinsic != expectedReduction ||
      description.compareIntrinsic != expectedCompare ||
      description.maskedMergeIntrinsic != expectedMerge)
    return makeRVVTargetRouteError(
        llvm::Twine("runtime-scalar computed-mask standalone reduction target "
                    "artifact consumer requires provider-derived scalar seed "
                    "splat '") +
        expectedSeedSplat + "', RHS scalar splat '" + expectedRHSBroadcast +
        "', signed min/max/add reduction intrinsic '" + expectedReduction +
        "', compare intrinsic '" + expectedCompare +
        "', inactive neutral merge '" + expectedMerge +
        "', and scalar result store '" + expectedStore +
        "' before artifact export but saw seed splat '" +
        description.scalarSeedSplatIntrinsic + "', RHS scalar splat '" +
        description.rhsBroadcastIntrinsic + "', reduction '" +
        description.intrinsic + "', compare '" + description.compareIntrinsic +
        "', merge '" + description.maskedMergeIntrinsic + "', and store '" +
        description.storeIntrinsic + "'");

  const llvm::StringRef expectedInactiveRequirement =
      description.operation == plugin::rvv::RVVSelectedBodyOperationKind::
                                   RuntimeScalarComputedMaskStandaloneReduceAdd
          ? llvm::StringRef(
                "masked-standalone-reduction-zero-inactive-lanes-before-reduction")
          : llvm::StringRef(
                kRVVComputedMaskStandaloneReductionInactiveLaneNeutralRequirement);
  if (description.maskRole != kRVVComputedMaskStandaloneReductionMaskRole ||
      description.maskSource != kRVVComputedMaskStandaloneReductionMaskSource ||
      description.maskMemoryForm !=
          kRVVComputedMaskStandaloneReductionMaskMemoryForm ||
      description.inactiveLaneZeroingRequirement != expectedInactiveRequirement ||
      description.comparePredicateKind !=
          kRVVComputedMaskStandaloneReductionComparePredicate)
    return makeRVVTargetRouteError(
        llvm::Twine("runtime-scalar computed-mask standalone reduction target "
                    "artifact consumer requires provider-derived mask "
                    "role/source/form, operation-specific inactive-lane "
                    "requirement '") +
        expectedInactiveRequirement + "', and compare predicate '" +
        kRVVComputedMaskStandaloneReductionComparePredicate +
        "' before artifact export");

  if (description.accumulationRouteFamilyPlanID !=
          kRVVComputedMaskAccumulationRouteFamilyPlanID ||
      description.accumulationComputeSuffix !=
          kRVVComputedMaskStandaloneReductionComputeSuffix ||
      description.accumulationMaskProducerSource !=
          kRVVRuntimeScalarComputedMaskStandaloneReductionProducerSource ||
      description.accumulationAccumulatorContract !=
          kRVVComputedMaskStandaloneReductionAccumulatorContract ||
      description.accumulationResultContract !=
          kRVVComputedMaskStandaloneReductionResultContract ||
      description.accumulationScalarCarryContract !=
          kRVVComputedMaskStandaloneReductionScalarCarryContract)
    return makeRVVTargetRouteError(
        llvm::Twine("runtime-scalar computed-mask standalone reduction target "
                    "artifact consumer requires provider-derived "
                    "computed-mask accumulation plan '") +
        kRVVComputedMaskAccumulationRouteFamilyPlanID +
        "', scalar horizontal reduction suffix '" +
        kRVVComputedMaskStandaloneReductionComputeSuffix +
        "', runtime-scalar producer plan '" +
        kRVVRuntimeScalarComputedMaskStandaloneReductionProducerSource +
        "', scalar seed/result contracts, and scalar carry contract before "
        "artifact export; saw producer '" +
        description.accumulationMaskProducerSource + "', accumulator '" +
        description.accumulationAccumulatorContract + "', result '" +
        description.accumulationResultContract + "', and scalar carry '" +
        description.accumulationScalarCarryContract + "'");

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

llvm::Error validateRVVStandaloneReductionAccumulationRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
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

  if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
      !routeLoopContainsCallee(loop, description.intrinsic) ||
      !routeLoopContainsCallee(loop, description.storeIntrinsic) ||
      !routeLoopContainsCallee(loop, description.scalarSeedSplatIntrinsic))
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-built vector load, reduction, seed splat, and store "
        "statements before artifact export");

  if (isRVVComputedMaskStandaloneReductionRouteFamilyOperation(
          description.operation)) {
    if (!routeLoopContainsCallee(loop, description.compareIntrinsic))
      return makeRVVTargetRouteError(
          "computed-mask standalone reduction/accumulation target artifact "
          "consumer requires provider-built compare statement facts before "
          "artifact export");
    if (!routeLoopContainsCallee(loop, description.maskedMergeIntrinsic))
      return makeRVVTargetRouteError(
          "computed-mask standalone reduction/accumulation target artifact "
          "consumer requires provider-built inactive-lane merge statement "
          "facts before artifact export");
    if (isRVVRuntimeScalarComputedMaskStandaloneReductionRouteFamilyOperation(
            description.operation) &&
        !routeLoopContainsCallee(loop, description.rhsBroadcastIntrinsic))
      return makeRVVTargetRouteError(
          "runtime-scalar computed-mask standalone reduction/accumulation "
          "target artifact consumer requires provider-built RHS scalar splat "
          "statement facts before artifact export");
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
  if (description.scalarSeedSplatIntrinsic.empty() ||
      description.intrinsic.empty() || description.storeIntrinsic.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-derived scalar seed, reduction intrinsic, and store facts "
        "before artifact export");
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
        "tcrv_rvv.compare_predicate_kind"};
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
  if (route.getCallOpaqueSteps().empty())
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires "
        "provider-built pre-loop setvl statement facts before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires rebuilt "
        "provider route pre-loop setvl statement to define the full-chunk VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "compare/select mask target artifact consumer requires pre-loop "
          "statements to carry selected typed RVV source provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires exactly one "
        "provider-built runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  const support::RuntimeABIParameter *runtimeN =
      findRuntimeElementCountABIParameter(description);
  if (!runtimeN)
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires a "
        "provider-derived runtime element count ABI parameter");
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.upperBound.expression != runtimeN->cName ||
      loop.upperBound.cType != runtimeN->cType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires "
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
        "compare/select mask target artifact consumer requires "
        "provider-built loop setvl to derive per-iteration VL from remaining "
        "runtime AVL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "compare/select mask target artifact consumer requires loop "
          "statements to carry selected typed RVV source provenance");

  auto requireLoopCallee = [&](llvm::StringRef callee,
                               llvm::StringRef factLabel) -> llvm::Error {
    if (callee.empty())
      return llvm::Error::success();
    if (routeLoopContainsCallee(loop, callee))
      return llvm::Error::success();
    return makeRVVTargetRouteError(
        llvm::Twine("compare/select mask target artifact consumer requires "
                    "provider-built ") +
        factLabel + " statement callee '" + callee +
        "' to mirror selected body route facts before artifact export");
  };

  if (llvm::Error error =
          requireLoopCallee(description.compareIntrinsic, "compare"))
    return error;
  if (llvm::Error error = requireLoopCallee(
          description.secondaryCompareIntrinsic, "secondary compare"))
    return error;
  if (llvm::Error error =
          requireLoopCallee(description.maskAndIntrinsic, "mask composition"))
    return error;

  using OperationKind = plugin::rvv::RVVSelectedBodyOperationKind;
  const bool isComputedMaskUnitLoadStore =
      description.operation == OperationKind::ComputedMaskUnitLoadStore;
  const bool isComputedMaskStridedStore =
      description.operation == OperationKind::ComputedMaskStridedStore;
  const bool isComputedMaskStridedLoadUnitStore =
      description.operation == OperationKind::ComputedMaskStridedLoadUnitStore;
  const bool isComputedMaskIndexedGatherLoadUnitStore =
      description.operation ==
      OperationKind::ComputedMaskIndexedGatherLoadUnitStore;
  const bool isComputedMaskIndexedScatterStoreUnitLoad =
      description.operation ==
      OperationKind::ComputedMaskIndexedScatterStoreUnitLoad;
  const bool isRuntimeScalarComputedMaskStore =
      description.operation == OperationKind::RuntimeScalarComputedMaskStore;
  const bool isRuntimeScalarComputedMaskLoadStore =
      description.operation ==
      OperationKind::RuntimeScalarComputedMaskLoadStore;

  if (isRVVCompareSelectMaskProducerRouteFamilyOperation(
          description.operation)) {
    if (llvm::Error error =
            requireLoopCallee(description.intrinsic, "primary compute"))
      return error;
  }

  if (llvm::Error error =
          requireLoopCallee(description.vectorLoadIntrinsic, "vector load"))
    return error;
  if (llvm::Error error =
          requireLoopCallee(description.rhsBroadcastIntrinsic, "rhs splat"))
    return error;
  if ((isComputedMaskIndexedGatherLoadUnitStore ||
       isComputedMaskIndexedScatterStoreUnitLoad)) {
    if (llvm::Error error =
            requireLoopCallee(description.indexLoadIntrinsic, "index load"))
      return error;
  }
  const bool requiresIndexScaleStatement =
      !description.indexScaleIntrinsic.empty() &&
      description.offsetUnit != "element";
  if (requiresIndexScaleStatement) {
    if (llvm::Error error =
            requireLoopCallee(description.indexScaleIntrinsic, "index scale"))
      return error;
  }
  if (isComputedMaskUnitLoadStore ||
      isComputedMaskStridedLoadUnitStore ||
      isComputedMaskIndexedGatherLoadUnitStore ||
      isRuntimeScalarComputedMaskLoadStore) {
    if (llvm::Error error =
            requireLoopCallee(description.maskedLoadIntrinsic, "masked load"))
      return error;
  }
  if (isComputedMaskStridedLoadUnitStore) {
    if (llvm::Error error =
            requireLoopCallee(description.stridedLoadIntrinsic, "strided load"))
      return error;
  }
  if (isComputedMaskIndexedGatherLoadUnitStore) {
    if (llvm::Error error =
            requireLoopCallee(description.indexedLoadIntrinsic, "indexed load"))
      return error;
  }
  if (isRVVCompareSelectMaskProducerRouteFamilyOperation(
          description.operation) ||
      isComputedMaskUnitLoadStore ||
      isComputedMaskStridedLoadUnitStore ||
      isComputedMaskIndexedGatherLoadUnitStore ||
      isRuntimeScalarComputedMaskStore ||
      isRuntimeScalarComputedMaskLoadStore) {
    if (llvm::Error error =
            requireLoopCallee(description.storeIntrinsic, "store"))
      return error;
  }
  if (isComputedMaskStridedStore) {
    if (llvm::Error error = requireLoopCallee(
            description.stridedStoreIntrinsic, "strided store"))
      return error;
  }
  if (isComputedMaskIndexedScatterStoreUnitLoad) {
    if (llvm::Error error = requireLoopCallee(
            description.indexedStoreIntrinsic, "indexed store"))
      return error;
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
    if (description.operation ==
            plugin::rvv::RVVSelectedBodyOperationKind::
                RuntimeScalarDualCompareMaskAndSelect &&
        description.maskComposition.empty())
      return makeRVVTargetRouteError(
          "dual compare/select target artifact consumer requires "
          "provider-derived mask composition facts before artifact export");
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

llvm::Error validateRVVConversionDtypePolicyRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getCallOpaqueSteps().empty())
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-built pre-loop setvl statement facts before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires rebuilt "
        "provider route pre-loop setvl statement to define the full-chunk VL");
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
  if (description.runtimeABIParameters.empty() ||
      loop.upperBound.expression !=
          description.runtimeABIParameters.back().cName ||
      loop.upperBound.cType != description.runtimeABIParameters.back().cType)
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-built loop upper bound to use the runtime n/AVL ABI "
        "parameter");
  if (loop.bodySteps.empty() ||
      loop.bodySteps.front().callee != description.setVLIntrinsic ||
      !stepHasResult(loop.bodySteps.front(), description.emitCLoopVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-built loop setvl statement to define per-iteration VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "conversion dtype-policy target artifact consumer requires loop "
          "statements to carry selected typed RVV source provenance");

  if (isRVVConversionDtypePolicyWideningRouteFamilyOperation(
          description.operation)) {
    if (!routeLoopContainsCallee(loop,
                                 description.sourceVectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.intrinsic) ||
        !routeLoopContainsCallee(loop, description.storeIntrinsic))
      return makeRVVTargetRouteError(
          "widening conversion dtype-policy target artifact consumer "
          "requires provider-built source load, widening conversion, and "
          "store statements before artifact export");
  }

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
    "src=lhs-input-buffer:src:runtime-abi-mirror|seg-load-base|src-mem|header;"
    "out0=segment-field0-output-buffer:out0:runtime-abi-mirror|field0-store-base|field0-role|dst-mem|header;"
    "out1=segment-field1-output-buffer:out1:runtime-abi-mirror|field1-store-base|field1-role|dst-mem|header;"
    "n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header");
constexpr llvm::StringLiteral kRVVSegment2InterleaveRouteOperandSummary(
    "rvv-route-operand-binding:segment2_interleave_unit_load.v1;"
    "src0=segment-field0-input-buffer:src0:runtime-abi-mirror|field0-load-base|field0-role|src0-mem|tuple-field0|header;"
    "src1=segment-field1-input-buffer:src1:runtime-abi-mirror|field1-load-base|field1-role|src1-mem|tuple-field1|header;"
    "dst=segment-interleaved-output-buffer:dst:runtime-abi-mirror|seg-store-base|dst-mem|header;"
    "n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header");
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
    "cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call;"
    "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call;"
    "src=source-input-buffer:src:abi|mseg-base|mseg-call|src-mem;"
    "out0=segment-field0-output-buffer:out0:abi|old0-load|f0-pass|"
    "f0-store|f0-role|dst-mem|hdr;"
    "out1=segment-field1-output-buffer:out1:abi|old1-load|f1-pass|"
    "f1-store|f1-role|dst-mem|hdr;"
    "n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2StoreRouteOperandPlan(
    "rvv-route-operand-binding:computed_masked_segment2_store_unit_load.v1");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2StoreRouteOperandSummary(
    "rvv-route-operand-binding:computed_masked_segment2_store_unit_load.v1;"
    "cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call;"
    "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call;"
    "src0=segment-field0-input-buffer:src0:abi|f0-load|f0-payload|tuple0|f0-role|src0-mem;"
    "src1=segment-field1-input-buffer:src1:abi|f1-load|f1-payload|tuple1|f1-role|src1-mem;"
    "dst=segment-interleaved-output-buffer:dst:abi|mseg-store|dst-mem|hdr;"
    "n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2UpdateRouteOperandPlan(
    "rvv-route-operand-binding:computed_masked_segment2_update_unit_load.v1");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2UpdateRouteOperandSummary(
    "rvv-route-operand-binding:computed_masked_segment2_update_unit_load.v1;"
    "cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call;"
    "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call;"
    "src0=segment-field0-input-buffer:src0:abi|f0-load|f0-payload|"
    "add-lhs|tuple0|f0-role|src0-mem;"
    "src1=segment-field1-input-buffer:src1:abi|f1-load|f1-payload|"
    "add-rhs|tuple1|f1-role|src1-mem;"
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
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "compare intrinsic", description.compareIntrinsic,
          kRVVComputedMaskCompareIntrinsic))
    return error;
  if (llvm::Error error = requireRVVSegment2MemoryProviderField(
          "compare/source load intrinsic", description.vectorLoadIntrinsic,
          kRVVComputedMaskCompareLoadIntrinsic))
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
  if (operation == plugin::rvv::RVVSelectedBodyOperationKind::
                       ComputedMaskSegment2UpdateUnitLoad) {
    if (description.segment2UpdateArithmeticIntrinsic.empty())
      return makeRVVTargetRouteError(
          "computed-mask segment2 update target artifact consumer requires "
          "provider-derived update arithmetic callee before artifact export");
  } else if (!description.segment2UpdateArithmeticIntrinsic.empty()) {
    return makeRVVTargetRouteError(
        llvm::Twine("segment2-memory target artifact consumer rejects stale "
                    "provider-derived segment2 update arithmetic callee fact '") +
        description.segment2UpdateArithmeticIntrinsic + "'");
  }

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

llvm::Error validateRVVSegment2MemoryRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getCallOpaqueSteps().empty())
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires provider-built "
        "pre-loop setvl statement facts before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires rebuilt provider "
        "route pre-loop setvl statement to define the full-chunk VL");
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
  if (description.runtimeABIParameters.empty() ||
      loop.upperBound.expression != description.runtimeABIParameters.back().cName ||
      loop.upperBound.cType != description.runtimeABIParameters.back().cType)
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires provider-built "
        "loop upper bound to use the runtime n/AVL ABI parameter");
  if (loop.bodySteps.empty() ||
      loop.bodySteps.front().callee != description.setVLIntrinsic ||
      !stepHasResult(loop.bodySteps.front(), description.emitCLoopVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires provider-built "
        "loop setvl statement to define per-iteration VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "segment2-memory target artifact consumer requires loop statements "
          "to carry selected typed RVV source provenance");

  if (isRVVComputedMaskSegment2MemoryRouteFamilyOperation(
          description.operation) &&
      !routeLoopContainsCallee(loop, description.compareIntrinsic))
    return makeRVVTargetRouteError(
        "computed-mask segment2-memory target artifact consumer requires "
        "provider-built compare/mask statement facts before artifact export");

  switch (description.operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2LoadUnitStore:
    if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.segmentLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.segmentStoreIntrinsic) ||
        !routeLoopContainsCallee(loop,
                                 description.segmentFieldExtractIntrinsic) ||
        !routeLoopContainsCallee(loop, description.storeIntrinsic))
      return makeRVVTargetRouteError(
          "computed-mask segment2 load target artifact consumer requires "
          "provider-built segment load, tuple, extract, and store statements");
    break;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2StoreUnitLoad:
    if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.segmentStoreIntrinsic) ||
        !routeLoopContainsCallee(loop,
                                 description.segmentFieldExtractIntrinsic))
      return makeRVVTargetRouteError(
          "computed-mask segment2 store target artifact consumer requires "
          "provider-built tuple and masked segment-store statements");
    break;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2UpdateUnitLoad:
    if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop,
                                 description.segment2UpdateArithmeticIntrinsic) ||
        !routeLoopContainsCallee(loop, description.segmentStoreIntrinsic) ||
        !routeLoopContainsCallee(loop,
                                 description.segmentFieldExtractIntrinsic))
      return makeRVVTargetRouteError(
          "computed-mask segment2 update target artifact consumer requires "
          "provider-built arithmetic, tuple, and masked segment-store "
          "statements");
    break;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      Segment2DeinterleaveUnitStore:
    if (!routeLoopContainsCallee(loop, description.segmentLoadIntrinsic) ||
        !routeLoopContainsCallee(loop,
                                 description.segmentFieldExtractIntrinsic) ||
        !routeLoopContainsCallee(loop, description.storeIntrinsic))
      return makeRVVTargetRouteError(
          "segment2 deinterleave target artifact consumer requires "
          "provider-built segment load, extract, and field-store statements");
    break;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      Segment2InterleaveUnitLoad:
    if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.segmentStoreIntrinsic) ||
        !routeLoopContainsCallee(loop,
                                 description.segmentFieldExtractIntrinsic))
      return makeRVVTargetRouteError(
          "segment2 interleave target artifact consumer requires "
          "provider-built tuple and segment-store statements");
    break;
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
              candidate, "tcrv_rvv.segment_store_intrinsic",
              description.segmentStoreIntrinsic,
              "selected typed RVV computed-mask segment2 tuple materializer"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.segment_field_extract_intrinsic",
              description.segmentFieldExtractIntrinsic,
              "selected typed RVV computed-mask segment2 field extract"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.segment_tuple_create_intrinsic", "",
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
  if (route.getCallOpaqueSteps().empty())
    return makeRVVTargetRouteError(
        "vector reduction target artifact consumer requires provider-built "
        "pre-loop setvl statement facts before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "vector reduction target artifact consumer requires rebuilt provider "
        "route pre-loop setvl statement to define the full-chunk VL");
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
        "vector reduction target artifact consumer requires provider-built "
        "loop setvl to derive per-iteration VL from remaining runtime AVL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "vector reduction target artifact consumer requires loop statements "
          "to carry selected typed RVV source provenance");

  if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
      !routeLoopContainsCallee(loop, description.intrinsic) ||
      !routeLoopContainsCallee(loop, description.storeIntrinsic))
    return makeRVVTargetRouteError(
        "vector reduction target artifact consumer requires provider-built "
        "vector load, reduction, and store statements before artifact export");
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
  if (route.getCallOpaqueSteps().empty())
    return makeRVVTargetRouteError(
        "widening MAcc contraction target artifact consumer requires "
        "provider-built pre-loop setvl statement facts before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "widening MAcc contraction target artifact consumer requires rebuilt "
        "provider route pre-loop setvl statement to define the full-chunk VL");
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
  const support::RuntimeABIParameter *runtimeN =
      findRuntimeElementCountABIParameter(description);
  if (!runtimeN)
    return makeRVVTargetRouteError(
        "widening MAcc contraction target artifact consumer requires a "
        "provider-derived runtime element count ABI parameter");
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.upperBound.expression != runtimeN->cName ||
      loop.upperBound.cType != runtimeN->cType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "widening MAcc contraction target artifact consumer requires "
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
        "widening MAcc contraction target artifact consumer requires "
        "provider-built loop setvl to derive per-iteration VL from remaining "
        "runtime AVL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "widening MAcc contraction target artifact consumer requires loop "
          "statements to carry selected typed RVV source provenance");

  if (!routeLoopContainsCallee(loop, description.sourceVectorLoadIntrinsic) ||
      !routeLoopContainsCallee(loop, description.intrinsic) ||
      !routeLoopContainsCallee(loop, description.storeIntrinsic))
    return makeRVVTargetRouteError(
        "widening MAcc contraction target artifact consumer requires "
        "provider-built source load, widening MAcc, and store statements "
        "before artifact export");
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
