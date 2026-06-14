//===- RVVEmitCRoutePlanningInternal.h - shared RVV route-planning seam ---===//
//
// Implementation-private header (NOT part of the public API under include/).
// It declares the RVV selected-body "primitive" builders -- the pure
// (operation / sew / lmul / count) -> StringRef intrinsic / type / CType
// constructors plundered out of the RVVEmitCRoutePlanning.cpp monolith into a
// co-located translation unit (RVVEmitCRoutePlanningPrimitives.cpp). These were
// file-local `static` helpers; they are now promoted to external linkage in the
// named namespace so the route-analysis / route-derivation half of the monolith
// (which still lives in RVVEmitCRoutePlanning.cpp) can call them across the TU
// boundary.
//
// This is a behavior-preserving relocation: the helper bodies are unchanged and
// the route-description content they synthesize is byte-identical.
//
//===----------------------------------------------------------------------===//

#ifndef TIANCHENRV_PLUGIN_RVV_EMITC_RVVEMITCROUTEPLANNINGINTERNAL_H
#define TIANCHENRV_PLUGIN_RVV_EMITC_RVVEMITCROUTEPLANNINGINTERNAL_H

#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace tianchenrv::plugin::rvv {

// The intrinsic / type / CType profile triple consumed by the route-family
// derivation. Plain POD aggregates depending only on RVVSelectedBodyOperationKind
// (public) + StringRef + the public RVVSelectedBodyConfigVLContract. Relocated
// out of the monolith so the primitive TU and the derivation TU share one
// definition.
struct RVVSelectedBodyOperationProfile {
  RVVSelectedBodyOperationKind operation;
  llvm::StringRef operationMnemonic;
  llvm::StringRef resultName;
  llvm::StringRef maskName;
  bool isCompareSelect;
  bool isReduction;
  bool isMaskedArithmetic;
  bool isMultiplyAccumulate;
  bool isStridedMemory;
  bool isMemoryMovement;
  bool isIndexedMemoryMovement;
  bool isMaskedMemoryMovement;
  bool isSegmentedMemoryMovement;
  bool isWideningConversion;
};

struct RVVSelectedBodyConfigProfile {
  std::int64_t sew;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  const tcrv::rvv::RVVSelectedBodyConfigVLContract *configContract = nullptr;
  llvm::StringRef vlCType;
  llvm::StringRef vectorTypeName;
  llvm::StringRef indexVectorTypeName;
  llvm::StringRef maskTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef indexVectorCType;
  llvm::StringRef maskCType;
  llvm::StringRef scalarCType;
  llvm::StringRef constInputPointerCType;
  llvm::StringRef outputPointerCType;
  llvm::StringRef elementByteSize;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef indexLoadIntrinsic;
  llvm::StringRef indexScaleIntrinsic;
  llvm::StringRef indexedLoadIntrinsic;
  llvm::StringRef indexedStoreIntrinsic;
  llvm::StringRef stridedLoadIntrinsic;
  llvm::StringRef rhsBroadcastIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef stridedStoreIntrinsic;
};

struct RVVSelectedBodyTargetLeafProfile {
  llvm::StringRef intrinsic;
  llvm::StringRef compareIntrinsic;
  llvm::StringRef maskedMergeIntrinsic;
  llvm::StringRef rhsBroadcastIntrinsic;
  llvm::StringRef maskAndIntrinsic;
};

// Derived-text interning pool: returns a StringRef into a process-stable string
// pool so the by-value route descriptions can hold the views. Single definition
// lives in RVVEmitCRoutePlanningPrimitives.cpp.
llvm::StringRef internRVVSelectedBodyDerivedText(std::string text);

// Element-type / scalar-CType / pointer-CType primitives.
llvm::StringRef getRVVSelectedBodyIntegerElementTypeName(std::int64_t sew);
llvm::StringRef
getRVVSelectedBodyUnsignedIntegerElementTypeName(std::int64_t sew);
llvm::StringRef
getRVVSelectedBodyUnsignedMLIRIntegerElementTypeName(std::int64_t sew);
llvm::StringRef getRVVSelectedBodySignedScalarCType(std::int64_t sew);
llvm::StringRef getRVVSelectedBodyUnsignedScalarCType(std::int64_t sew);
llvm::StringRef getRVVSelectedBodyConstInputPointerCType(std::int64_t sew);
llvm::StringRef
getRVVSelectedBodyUnsignedConstInputPointerCType(std::int64_t sew);
llvm::StringRef getRVVSelectedBodyOutputPointerCType(std::int64_t sew);
llvm::StringRef
getRVVSelectedBodyUnsignedOutputPointerCType(std::int64_t sew);
llvm::StringRef getRVVSelectedBodyFloatElementTypeName(std::int64_t sew);
llvm::StringRef getRVVSelectedBodyFloatOutputPointerCType(std::int64_t sew);
llvm::StringRef getRVVSelectedBodyFloatConstInputPointerCType(std::int64_t sew);
llvm::StringRef getRVVSelectedBodyElementByteSize(std::int64_t sew);

// Vector / mask type-name + CType primitives.
llvm::StringRef getRVVSelectedBodyVectorTypeName(std::int64_t sew,
                                                 llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyUnsignedVectorTypeName(std::int64_t sew,
                                                         llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyFloatVectorTypeName(std::int64_t sew,
                                                      llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyMaskTypeName(std::int64_t sew,
                                               llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyFloatMaskTypeName(std::int64_t sew,
                                                    llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodySignedVectorCType(std::int64_t sew,
                                                    llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyUnsignedVectorCType(std::int64_t sew,
                                                      llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyFloatVectorCType(std::int64_t sew,
                                                   llvm::StringRef lmul);
std::optional<std::int64_t>
getRVVSelectedBodyMaskBitWidth(std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyMaskCType(std::int64_t sew,
                                            llvm::StringRef lmul);

// setvl / load / splat / store intrinsic builders.
llvm::StringRef getRVVSelectedBodySetVLIntrinsic(std::int64_t sew,
                                                 llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyVectorLoadIntrinsic(std::int64_t sew,
                                                      llvm::StringRef lmul);
llvm::StringRef
getRVVSelectedBodyUnsignedVectorLoadIntrinsic(std::int64_t sew,
                                              llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyFloatVectorLoadIntrinsic(std::int64_t sew,
                                                           llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyStridedLoadIntrinsic(std::int64_t sew,
                                                       llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyScalarSplatIntrinsic(std::int64_t sew,
                                                       llvm::StringRef lmul);
llvm::StringRef
getRVVSelectedBodyUnsignedScalarSplatIntrinsic(std::int64_t sew,
                                               llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyFloatScalarSplatIntrinsic(std::int64_t sew,
                                                            llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyStoreIntrinsic(std::int64_t sew,
                                                 llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyUnsignedStoreIntrinsic(std::int64_t sew,
                                                         llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyFloatStoreIntrinsic(std::int64_t sew,
                                                      llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyI32ToF32DequantConvertIntrinsic();
llvm::StringRef getRVVSelectedBodyF32ScalarScaleIntrinsic();
llvm::StringRef getRVVSelectedBodyStridedStoreIntrinsic(std::int64_t sew,
                                                        llvm::StringRef lmul);

// Index / indexed / masked memory intrinsic builders.
bool hasRVVSelectedBodyI32M1IndexFacts(std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyIndexVectorTypeName(std::int64_t sew,
                                                      llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyIndexVectorCType(std::int64_t sew,
                                                   llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyIndexLoadIntrinsic(std::int64_t sew,
                                                     llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyIndexScaleIntrinsic(std::int64_t sew,
                                                      llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyIndexedLoadIntrinsic(std::int64_t sew,
                                                       llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyIndexedStoreIntrinsic(std::int64_t sew,
                                                        llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyMaskedLoadIntrinsic(std::int64_t sew,
                                                      llvm::StringRef lmul);
llvm::StringRef
getRVVSelectedBodyMaskedStridedLoadIntrinsic(std::int64_t sew,
                                             llvm::StringRef lmul);
llvm::StringRef
getRVVSelectedBodyMaskedIndexedLoadIntrinsic(std::int64_t sew,
                                             llvm::StringRef lmul);
llvm::StringRef
getRVVSelectedBodyMaskedIndexedStoreIntrinsic(std::int64_t sew,
                                              llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyMaskedStoreIntrinsic(std::int64_t sew,
                                                       llvm::StringRef lmul);
llvm::StringRef
getRVVSelectedBodyMaskedStridedStoreIntrinsic(std::int64_t sew,
                                              llvm::StringRef lmul);

// Segment (deinterleave/interleave) intrinsic builders.
llvm::StringRef getRVVSelectedBodySegmentTupleCType(std::int64_t sew,
                                                    llvm::StringRef lmul,
                                                    std::int64_t count);
llvm::StringRef getRVVSelectedBodySegmentLoadIntrinsic(std::int64_t sew,
                                                       llvm::StringRef lmul,
                                                       std::int64_t count);
llvm::StringRef
getRVVSelectedBodyMaskedSegmentLoadIntrinsic(std::int64_t sew,
                                             llvm::StringRef lmul,
                                             std::int64_t count);
llvm::StringRef getRVVSelectedBodySegmentStoreIntrinsic(std::int64_t sew,
                                                        llvm::StringRef lmul,
                                                        std::int64_t count);
llvm::StringRef
getRVVSelectedBodyMaskedSegmentStoreIntrinsic(std::int64_t sew,
                                              llvm::StringRef lmul,
                                              std::int64_t count);
llvm::StringRef getRVVSelectedBodySegmentFieldExtractIntrinsic(
    std::int64_t sew, llvm::StringRef lmul, std::int64_t count);
llvm::StringRef getRVVSelectedBodySegmentTupleCreateIntrinsic(
    std::int64_t sew, llvm::StringRef lmul, std::int64_t count);

// Arithmetic / compare / select / mask-and intrinsic builders.
llvm::StringRef getRVVSelectedBodyArithmeticIntrinsic(
    RVVSelectedBodyOperationKind operation, std::int64_t sew,
    llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyCompareIntrinsicForPredicate(
    llvm::StringRef predicateKind, std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyFloatCompareIntrinsicForPredicate(
    llvm::StringRef predicateKind, std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodySelectIntrinsic(std::int64_t sew,
                                                  llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyFloatSelectIntrinsic(std::int64_t sew,
                                                       llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyMaskAndIntrinsic(std::int64_t sew,
                                                   llvm::StringRef lmul);

// Runtime-scalar computed-mask config predicates + their scalar-channel builders.
bool isRVVSelectedBodyRuntimeScalarComputedMaskMemoryConfig(
    std::int64_t sew, llvm::StringRef lmul);
bool isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionConfig(
    std::int64_t sew, llvm::StringRef lmul);
bool isRVVSelectedBodyStandaloneReductionScalarChannelConfig(
    std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVStandaloneReductionScalarResultVectorTypeName(
    std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVStandaloneReductionScalarResultVectorCType(
    std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVStandaloneReductionScalarSeedSplatIntrinsic(
    std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVStandaloneReductionScalarResultStoreIntrinsic(
    std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyRuntimeScalarMaskedLoadIntrinsic(
    std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyRuntimeScalarMaskedLoadIntrinsic(
    const RVVSelectedBodyConfigProfile &config);
llvm::StringRef getRVVSelectedBodyRuntimeScalarMaskedStoreIntrinsic(
    std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVSelectedBodyRuntimeScalarMaskedStoreIntrinsic(
    const RVVSelectedBodyConfigProfile &config);
llvm::StringRef getRVVRuntimeScalarComputedMaskMemoryElementCType(
    std::int64_t sew, llvm::StringRef lmul);
llvm::StringRef getRVVRuntimeScalarComputedMaskMemoryUnitStoreIntrinsic(
    std::int64_t sew, llvm::StringRef lmul);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_EMITC_RVVEMITCROUTEPLANNINGINTERNAL_H
