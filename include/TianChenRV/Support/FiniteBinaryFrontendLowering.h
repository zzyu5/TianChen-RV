#ifndef TIANCHENRV_SUPPORT_FINITEBINARYFRONTENDLOWERING_H
#define TIANCHENRV_SUPPORT_FINITEBINARYFRONTENDLOWERING_H

#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace tianchenrv::support {

enum class FiniteBinaryElementKind {
  I32,
  I64,
};

// Neutral frontend marker/ABI contract for the bounded linalg binary slice.
// It names accepted route markers and reusable ABI spellings only after the
// source linalg body has already determined the finite family. RVV/scalar
// lowering route labels, arithmetic operators, selected vector shape, route
// ids, artifacts, and evidence remain plugin/target-owned.
struct FiniteBinaryFrontendContract {
  FiniteBinaryElementKind elementKind = FiniteBinaryElementKind::I32;
  llvm::StringRef dtypeID;
  unsigned elementBitWidth = 32;
  llvm::StringRef familyID;
  llvm::StringRef frontendLowering;
  llvm::StringRef constInputPointerCType;
  llvm::StringRef outputPointerCType;
};

// Shared source-frontdoor lowering contract for finite binary source adapters.
// Adapters own source-shape recognition; this contract owns the common
// tcrv.exec runtime ABI surface plus any source extent authority metadata that
// downstream artifact routes validate before output.
struct FiniteBinarySourceFrontendLoweringContract {
  const FiniteBinaryFrontendContract *contract = nullptr;
  llvm::SmallVector<RuntimeABIMemWindowSpec, 3> bufferMemWindowSpecs;
  llvm::SmallVector<RuntimeABIParamSpec, 1> runtimeElementCountSpecs;
  std::optional<std::int64_t> fixedSourceVectorExtent;
  bool dynamicRuntimeExtentFromSCFUpperBound = false;
};

inline constexpr llvm::StringLiteral kFrontendSourceKindAttrName(
    "tcrv_frontend_source_kind");
inline constexpr llvm::StringLiteral kFrontendSourceAuthorityAttrName(
    "tcrv_frontend_source_authority");
inline constexpr llvm::StringLiteral kFrontendSourceVectorExtentAttrName(
    "tcrv_frontend_source_vector_extent");
inline constexpr llvm::StringLiteral
    kFrontendRuntimeElementCountConstraintAttrName(
        "tcrv_frontend_runtime_element_count_constraint");
inline constexpr llvm::StringLiteral kFrontendRuntimeExtentArgAttrName(
    "tcrv_frontend_runtime_extent_arg");
inline constexpr llvm::StringLiteral kFrontendSourceLoopStepAttrName(
    "tcrv_frontend_source_loop_step");
inline constexpr llvm::StringLiteral kFrontendSourceVectorChunkExtentAttrName(
    "tcrv_frontend_source_vector_chunk_extent");
inline constexpr llvm::StringLiteral kFrontendActiveLaneAuthorityAttrName(
    "tcrv_frontend_active_lane_authority");
inline constexpr llvm::StringLiteral kFrontendSourceTailPolicyAttrName(
    "tcrv_frontend_source_tail_policy");

inline constexpr llvm::StringLiteral kFrontendFixedVectorI32VAddSourceKind(
    "mlir-vector-transfer-fixed-i32-vadd.v1");
inline constexpr llvm::StringLiteral kFrontendDynamicVectorI32VAddSourceKind(
    "mlir-vector-scf-runtime-i32-vadd.v1");
inline constexpr llvm::StringLiteral kFrontendDynamicVectorI32VSubSourceKind(
    "mlir-vector-scf-runtime-i32-vsub.v1");
inline constexpr llvm::StringLiteral kFrontendDynamicVectorI32VMulSourceKind(
    "mlir-vector-scf-runtime-i32-vmul.v1");
inline constexpr std::int64_t kFrontendFixedVectorI32VAddSourceExtent = 16;
inline constexpr std::int64_t kFrontendDynamicVectorI32VAddLoopStep = 16;
inline constexpr std::int64_t kFrontendDynamicVectorI32VAddChunkExtent = 16;
inline constexpr llvm::StringLiteral kFrontendFixedVectorSourceAuthority(
    "source-vector-transfer-read-write-fixed-extent");
inline constexpr llvm::StringLiteral kFrontendDynamicVectorSourceAuthority(
    "source-scf-for-runtime-upper-bound");
inline constexpr llvm::StringLiteral
    kFrontendDynamicVectorActiveLaneAuthority(
        "mlir-vector-transfer-tail-active-lanes");
inline constexpr llvm::StringLiteral kFrontendDynamicVectorSourceTailPolicy(
    "runtime-n-bounded-transfer-tail-padding-and-store");
inline constexpr llvm::StringLiteral
    kFrontendRuntimeElementCountMustEqualSourceExtent(
        "must-equal-source-vector-extent");
inline constexpr llvm::StringLiteral
    kFrontendRuntimeElementCountFromSourceRuntimeExtent(
        "source-runtime-extent");

inline llvm::StringRef getFrontendSourceKindMetadataName() {
  return "tcrv_frontend.source_kind";
}

inline llvm::StringRef getFrontendSourceAuthorityMetadataName() {
  return "tcrv_frontend.source_authority";
}

inline llvm::StringRef getFrontendSourceVectorExtentMetadataName() {
  return "tcrv_frontend.source_vector_extent";
}

inline llvm::StringRef getFrontendRuntimeExtentArgMetadataName() {
  return "tcrv_frontend.runtime_extent_arg";
}

inline llvm::StringRef getFrontendSourceLoopStepMetadataName() {
  return "tcrv_frontend.source_loop_step";
}

inline llvm::StringRef getFrontendSourceVectorChunkExtentMetadataName() {
  return "tcrv_frontend.source_vector_chunk_extent";
}

inline llvm::StringRef getFrontendActiveLaneAuthorityMetadataName() {
  return "tcrv_frontend.active_lane_authority";
}

inline llvm::StringRef getFrontendSourceTailPolicyMetadataName() {
  return "tcrv_frontend.source_tail_policy";
}

inline llvm::StringRef
getFrontendRuntimeElementCountConstraintMetadataName() {
  return "tcrv_frontend.runtime_element_count_constraint";
}

inline llvm::StringRef getFrontendSourceExtentMetadataRole() {
  return "source-frontdoor-extent-authority";
}

inline llvm::StringRef getFrontendSourceExtentMetadataNote() {
  return "fixed MLIR vector frontdoor source extent; runtime element-count "
         "must match this source authority; not selected RVV config, "
         "artifact-local component capacity, VL, correctness evidence, or performance "
         "evidence";
}

inline llvm::StringRef getFrontendRuntimeExtentMetadataRole() {
  return "source-frontdoor-runtime-avl-authority";
}

inline llvm::StringRef getFrontendRuntimeExtentMetadataNote() {
  return "dynamic MLIR vector/SCF frontdoor runtime extent; source scf.for "
         "upper bound maps to runtime element-count/AVL and MLIR transfer "
         "tail semantics define active lanes; not selected RVV tail/mask "
         "policy, selected vector config, artifact-local component capacity, "
         "correctness evidence, or performance evidence";
}

inline bool isFrontendFixedVectorSourceKind(llvm::StringRef sourceKind) {
  return sourceKind == kFrontendFixedVectorI32VAddSourceKind;
}

inline bool isFrontendDynamicVectorSourceKind(llvm::StringRef sourceKind) {
  return sourceKind == kFrontendDynamicVectorI32VAddSourceKind ||
         sourceKind == kFrontendDynamicVectorI32VSubSourceKind ||
         sourceKind == kFrontendDynamicVectorI32VMulSourceKind;
}

inline std::string formatFrontendDynamicVectorSourceKinds() {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "'" << kFrontendDynamicVectorI32VAddSourceKind << "', '"
         << kFrontendDynamicVectorI32VSubSourceKind << "', or '"
         << kFrontendDynamicVectorI32VMulSourceKind << "'";
  stream.flush();
  return text;
}

struct FixedVectorSourceExtentContract {
  std::string sourceKind;
  std::string sourceAuthority;
  std::int64_t sourceVectorExtent = 0;
  std::string runtimeElementCountConstraint;

  bool isValid() const {
    return isFrontendFixedVectorSourceKind(sourceKind) &&
           sourceAuthority == kFrontendFixedVectorSourceAuthority &&
           runtimeElementCountConstraint ==
               kFrontendRuntimeElementCountMustEqualSourceExtent &&
           sourceVectorExtent == kFrontendFixedVectorI32VAddSourceExtent;
  }

  std::string formatCommentBody() const {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << "source_frontend_extent_authority: source_kind=" << sourceKind
           << ", source_authority=" << sourceAuthority
           << ", source_vector_extent=" << sourceVectorExtent
           << ", runtime_element_count_constraint="
           << runtimeElementCountConstraint;
    stream.flush();
    return text;
  }
};

struct DynamicVectorRuntimeExtentContract {
  std::string sourceKind;
  std::string sourceAuthority;
  std::string runtimeExtentArg;
  std::int64_t sourceLoopStep = 0;
  std::int64_t sourceVectorChunkExtent = 0;
  std::string activeLaneAuthority;
  std::string sourceTailPolicy;
  std::string runtimeElementCountConstraint;

  bool isValid() const {
    return isFrontendDynamicVectorSourceKind(sourceKind) &&
           sourceAuthority == kFrontendDynamicVectorSourceAuthority &&
           runtimeExtentArg == "n" &&
           sourceLoopStep == kFrontendDynamicVectorI32VAddLoopStep &&
           sourceVectorChunkExtent ==
               kFrontendDynamicVectorI32VAddChunkExtent &&
           activeLaneAuthority ==
               kFrontendDynamicVectorActiveLaneAuthority &&
           sourceTailPolicy == kFrontendDynamicVectorSourceTailPolicy &&
           runtimeElementCountConstraint ==
               kFrontendRuntimeElementCountFromSourceRuntimeExtent;
  }

  std::string formatCommentBody() const {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << "source_frontend_runtime_avl_authority: source_kind="
           << sourceKind << ", source_authority=" << sourceAuthority
           << ", runtime_extent_arg=" << runtimeExtentArg
           << ", source_loop_step=" << sourceLoopStep
           << ", source_vector_chunk_extent=" << sourceVectorChunkExtent
           << ", active_lane_authority=" << activeLaneAuthority
           << ", source_tail_policy=" << sourceTailPolicy
           << ", runtime_element_count_constraint="
           << runtimeElementCountConstraint;
    stream.flush();
    return text;
  }
};

inline llvm::Error makeFixedVectorSourceExtentError(
    tcrv::exec::KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "fixed vector source extent authority validation failed";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

inline llvm::Error makeDynamicVectorRuntimeExtentError(
    tcrv::exec::KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "dynamic vector runtime extent authority validation failed";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

inline bool hasAnyFixedVectorSourceExtentAttr(mlir::Operation *op) {
  return op && (op->hasAttr(kFrontendSourceKindAttrName) ||
                op->hasAttr(kFrontendSourceAuthorityAttrName) ||
                op->hasAttr(kFrontendSourceVectorExtentAttrName) ||
                op->hasAttr(kFrontendRuntimeElementCountConstraintAttrName));
}

inline bool hasAnyDynamicVectorRuntimeExtentAttr(mlir::Operation *op) {
  return op && (op->hasAttr(kFrontendSourceKindAttrName) ||
                op->hasAttr(kFrontendSourceAuthorityAttrName) ||
                op->hasAttr(kFrontendRuntimeExtentArgAttrName) ||
                op->hasAttr(kFrontendSourceLoopStepAttrName) ||
                op->hasAttr(kFrontendSourceVectorChunkExtentAttrName) ||
                op->hasAttr(kFrontendActiveLaneAuthorityAttrName) ||
                op->hasAttr(kFrontendSourceTailPolicyAttrName) ||
                op->hasAttr(kFrontendRuntimeElementCountConstraintAttrName));
}

inline llvm::StringRef getFrontendSourceStringAttr(mlir::Operation *op,
                                                   llvm::StringRef attrName) {
  auto attr =
      op ? op->getAttrOfType<mlir::StringAttr>(attrName) : mlir::StringAttr();
  if (!attr)
    return {};
  return attr.getValue();
}

inline llvm::Expected<std::int64_t>
getFrontendPositiveI64Attr(tcrv::exec::KernelOp kernel, mlir::Operation *op,
                           llvm::StringRef owner, llvm::StringRef attrName,
                           llvm::Twine context) {
  auto attr = op ? op->getAttrOfType<mlir::IntegerAttr>(attrName)
                 : mlir::IntegerAttr();
  if (!attr)
    return makeDynamicVectorRuntimeExtentError(
        kernel, llvm::Twine(owner) + " requires integer attribute '" +
                    attrName + "' for " + context);
  if (attr.getInt() <= 0 || attr.getInt() > 64)
    return makeDynamicVectorRuntimeExtentError(
        kernel, llvm::Twine(owner) + " attribute '" + attrName +
                    "' must be in the bounded range [1, 64] for " +
                    context);
  return attr.getInt();
}

inline llvm::Expected<std::int64_t>
getFrontendSourceVectorExtentAttr(tcrv::exec::KernelOp kernel,
                                  mlir::Operation *op,
                                  llvm::StringRef owner) {
  auto attr = op ? op->getAttrOfType<mlir::IntegerAttr>(
                       kFrontendSourceVectorExtentAttrName)
                 : mlir::IntegerAttr();
  if (!attr)
    return makeFixedVectorSourceExtentError(
        kernel, llvm::Twine(owner) + " requires integer attribute '" +
                    kFrontendSourceVectorExtentAttrName + "'");
  if (attr.getInt() <= 0 || attr.getInt() > 64)
    return makeFixedVectorSourceExtentError(
        kernel, llvm::Twine(owner) + " attribute '" +
                    kFrontendSourceVectorExtentAttrName +
                    "' must be in the bounded range [1, 64]");
  return attr.getInt();
}

inline llvm::Expected<std::optional<FixedVectorSourceExtentContract>>
getFixedVectorSourceExtentContract(
    tcrv::exec::KernelOp kernel,
    tcrv::exec::RuntimeParamOp runtimeElementCountParam) {
  mlir::Operation *kernelOp = kernel.getOperation();
  mlir::Operation *paramOp = runtimeElementCountParam.getOperation();
  bool kernelHasAttrs = hasAnyFixedVectorSourceExtentAttr(kernelOp);
  bool paramHasAttrs = hasAnyFixedVectorSourceExtentAttr(paramOp);
  if (!kernelHasAttrs && !paramHasAttrs)
    return std::optional<FixedVectorSourceExtentContract>();
  llvm::StringRef kernelKind =
      getFrontendSourceStringAttr(kernelOp, kFrontendSourceKindAttrName);
  llvm::StringRef paramKind =
      getFrontendSourceStringAttr(paramOp, kFrontendSourceKindAttrName);
  if (isFrontendDynamicVectorSourceKind(kernelKind) ||
      isFrontendDynamicVectorSourceKind(paramKind))
    return std::optional<FixedVectorSourceExtentContract>();
  if (!kernelHasAttrs || !paramHasAttrs)
    return makeFixedVectorSourceExtentError(
        kernel,
        "fixed vector source extent authority requires matching metadata on "
        "both tcrv.exec.kernel and the runtime-element-count "
        "tcrv.exec.runtime_param");

  auto requireMatchingString =
      [&](llvm::StringRef attrName,
          llvm::StringRef expected) -> llvm::Expected<std::string> {
    llvm::StringRef kernelValue = getFrontendSourceStringAttr(kernelOp,
                                                              attrName);
    llvm::StringRef paramValue = getFrontendSourceStringAttr(paramOp,
                                                             attrName);
    if (kernelValue.empty() || paramValue.empty())
      return makeFixedVectorSourceExtentError(
          kernel,
          llvm::Twine("fixed vector source extent authority requires string "
                      "attribute '") +
              attrName + "' on both kernel and runtime_param");
    if (kernelValue != paramValue)
      return makeFixedVectorSourceExtentError(
          kernel,
          llvm::Twine("fixed vector source extent authority attribute '") +
              attrName + "' is stale between kernel value '" + kernelValue +
              "' and runtime_param value '" + paramValue + "'");
    if (kernelValue != expected)
      return makeFixedVectorSourceExtentError(
          kernel,
          llvm::Twine("fixed vector source extent authority attribute '") +
              attrName + "' must be '" + expected + "'");
    return kernelValue.str();
  };

  llvm::Expected<std::string> sourceKind =
      requireMatchingString(kFrontendSourceKindAttrName,
                            kFrontendFixedVectorI32VAddSourceKind);
  if (!sourceKind)
    return sourceKind.takeError();
  llvm::Expected<std::string> sourceAuthority =
      requireMatchingString(kFrontendSourceAuthorityAttrName,
                            kFrontendFixedVectorSourceAuthority);
  if (!sourceAuthority)
    return sourceAuthority.takeError();
  llvm::Expected<std::string> constraint = requireMatchingString(
      kFrontendRuntimeElementCountConstraintAttrName,
      kFrontendRuntimeElementCountMustEqualSourceExtent);
  if (!constraint)
    return constraint.takeError();

  llvm::Expected<std::int64_t> kernelExtent =
      getFrontendSourceVectorExtentAttr(kernel, kernelOp, "kernel");
  if (!kernelExtent)
    return kernelExtent.takeError();
  llvm::Expected<std::int64_t> paramExtent =
      getFrontendSourceVectorExtentAttr(kernel, paramOp, "runtime_param");
  if (!paramExtent)
    return paramExtent.takeError();
  if (*kernelExtent != *paramExtent)
    return makeFixedVectorSourceExtentError(
        kernel,
        llvm::Twine("source vector extent is stale between kernel value ") +
            llvm::Twine(*kernelExtent) + " and runtime_param value " +
            llvm::Twine(*paramExtent));

  FixedVectorSourceExtentContract contract;
  contract.sourceKind = std::move(*sourceKind);
  contract.sourceAuthority = std::move(*sourceAuthority);
  contract.sourceVectorExtent = *kernelExtent;
  contract.runtimeElementCountConstraint = std::move(*constraint);
  if (!contract.isValid())
    return makeFixedVectorSourceExtentError(
        kernel, "fixed vector source extent contract is incomplete");
  return std::optional<FixedVectorSourceExtentContract>(std::move(contract));
}

inline llvm::Expected<std::optional<DynamicVectorRuntimeExtentContract>>
getDynamicVectorRuntimeExtentContract(
    tcrv::exec::KernelOp kernel,
    tcrv::exec::RuntimeParamOp runtimeElementCountParam) {
  mlir::Operation *kernelOp = kernel.getOperation();
  mlir::Operation *paramOp = runtimeElementCountParam.getOperation();
  bool kernelHasAttrs = hasAnyDynamicVectorRuntimeExtentAttr(kernelOp);
  bool paramHasAttrs = hasAnyDynamicVectorRuntimeExtentAttr(paramOp);
  if (!kernelHasAttrs && !paramHasAttrs)
    return std::optional<DynamicVectorRuntimeExtentContract>();
  llvm::StringRef kernelKind =
      getFrontendSourceStringAttr(kernelOp, kFrontendSourceKindAttrName);
  llvm::StringRef paramKind =
      getFrontendSourceStringAttr(paramOp, kFrontendSourceKindAttrName);
  if (isFrontendFixedVectorSourceKind(kernelKind) ||
      isFrontendFixedVectorSourceKind(paramKind))
    return std::optional<DynamicVectorRuntimeExtentContract>();
  if (!kernelHasAttrs || !paramHasAttrs)
    return makeDynamicVectorRuntimeExtentError(
        kernel,
        "dynamic vector runtime extent authority requires matching metadata "
        "on both tcrv.exec.kernel and the runtime-element-count "
        "tcrv.exec.runtime_param");

  auto requireMatchingString =
      [&](llvm::StringRef attrName) -> llvm::Expected<std::string> {
    llvm::StringRef kernelValue = getFrontendSourceStringAttr(kernelOp,
                                                              attrName);
    llvm::StringRef paramValue = getFrontendSourceStringAttr(paramOp,
                                                             attrName);
    if (kernelValue.empty() || paramValue.empty())
      return makeDynamicVectorRuntimeExtentError(
          kernel,
          llvm::Twine("dynamic vector runtime extent authority requires "
                      "string attribute '") +
              attrName + "' on both kernel and runtime_param");
    if (kernelValue != paramValue)
      return makeDynamicVectorRuntimeExtentError(
          kernel,
          llvm::Twine("dynamic vector runtime extent authority attribute '") +
              attrName + "' is stale between kernel value '" + kernelValue +
              "' and runtime_param value '" + paramValue + "'");
    return kernelValue.str();
  };

  llvm::Expected<std::string> sourceKind =
      requireMatchingString(kFrontendSourceKindAttrName);
  if (!sourceKind)
    return sourceKind.takeError();
  if (!isFrontendDynamicVectorSourceKind(*sourceKind))
    return makeDynamicVectorRuntimeExtentError(
        kernel,
        llvm::Twine("dynamic vector runtime extent authority attribute '") +
            kFrontendSourceKindAttrName + "' must be " +
            formatFrontendDynamicVectorSourceKinds());
  llvm::Expected<std::string> sourceAuthority = requireMatchingString(
      kFrontendSourceAuthorityAttrName);
  if (!sourceAuthority)
    return sourceAuthority.takeError();
  if (*sourceAuthority != kFrontendDynamicVectorSourceAuthority)
    return makeDynamicVectorRuntimeExtentError(
        kernel,
        llvm::Twine("dynamic vector runtime extent authority attribute '") +
            kFrontendSourceAuthorityAttrName + "' must be '" +
            kFrontendDynamicVectorSourceAuthority + "'");
  llvm::Expected<std::string> runtimeExtentArg =
      requireMatchingString(kFrontendRuntimeExtentArgAttrName);
  if (!runtimeExtentArg)
    return runtimeExtentArg.takeError();
  if (*runtimeExtentArg != "n")
    return makeDynamicVectorRuntimeExtentError(
        kernel,
        llvm::Twine("dynamic vector runtime extent authority attribute '") +
            kFrontendRuntimeExtentArgAttrName + "' must be 'n'");
  llvm::Expected<std::string> constraint = requireMatchingString(
      kFrontendRuntimeElementCountConstraintAttrName);
  if (!constraint)
    return constraint.takeError();
  if (*constraint != kFrontendRuntimeElementCountFromSourceRuntimeExtent)
    return makeDynamicVectorRuntimeExtentError(
        kernel,
        llvm::Twine("dynamic vector runtime extent authority attribute '") +
            kFrontendRuntimeElementCountConstraintAttrName + "' must be '" +
            kFrontendRuntimeElementCountFromSourceRuntimeExtent + "'");
  llvm::Expected<std::string> activeLaneAuthority = requireMatchingString(
      kFrontendActiveLaneAuthorityAttrName);
  if (!activeLaneAuthority)
    return activeLaneAuthority.takeError();
  if (*activeLaneAuthority != kFrontendDynamicVectorActiveLaneAuthority)
    return makeDynamicVectorRuntimeExtentError(
        kernel,
        llvm::Twine("dynamic vector runtime extent authority attribute '") +
            kFrontendActiveLaneAuthorityAttrName + "' must be '" +
            kFrontendDynamicVectorActiveLaneAuthority + "'");
  llvm::Expected<std::string> sourceTailPolicy = requireMatchingString(
      kFrontendSourceTailPolicyAttrName);
  if (!sourceTailPolicy)
    return sourceTailPolicy.takeError();
  if (*sourceTailPolicy != kFrontendDynamicVectorSourceTailPolicy)
    return makeDynamicVectorRuntimeExtentError(
        kernel,
        llvm::Twine("dynamic vector runtime extent authority attribute '") +
            kFrontendSourceTailPolicyAttrName + "' must be '" +
            kFrontendDynamicVectorSourceTailPolicy + "'");

  llvm::Expected<std::int64_t> kernelLoopStep =
      getFrontendPositiveI64Attr(kernel, kernelOp, "kernel",
                                 kFrontendSourceLoopStepAttrName,
                                 "source scf.for step");
  if (!kernelLoopStep)
    return kernelLoopStep.takeError();
  llvm::Expected<std::int64_t> paramLoopStep =
      getFrontendPositiveI64Attr(kernel, paramOp, "runtime_param",
                                 kFrontendSourceLoopStepAttrName,
                                 "source scf.for step");
  if (!paramLoopStep)
    return paramLoopStep.takeError();
  if (*kernelLoopStep != *paramLoopStep)
    return makeDynamicVectorRuntimeExtentError(
        kernel,
        llvm::Twine("source loop step is stale between kernel value ") +
            llvm::Twine(*kernelLoopStep) + " and runtime_param value " +
            llvm::Twine(*paramLoopStep));

  llvm::Expected<std::int64_t> kernelChunk =
      getFrontendPositiveI64Attr(kernel, kernelOp, "kernel",
                                 kFrontendSourceVectorChunkExtentAttrName,
                                 "source vector chunk extent");
  if (!kernelChunk)
    return kernelChunk.takeError();
  llvm::Expected<std::int64_t> paramChunk =
      getFrontendPositiveI64Attr(kernel, paramOp, "runtime_param",
                                 kFrontendSourceVectorChunkExtentAttrName,
                                 "source vector chunk extent");
  if (!paramChunk)
    return paramChunk.takeError();
  if (*kernelChunk != *paramChunk)
    return makeDynamicVectorRuntimeExtentError(
        kernel,
        llvm::Twine("source vector chunk extent is stale between kernel "
                    "value ") +
            llvm::Twine(*kernelChunk) + " and runtime_param value " +
            llvm::Twine(*paramChunk));

  DynamicVectorRuntimeExtentContract contract;
  contract.sourceKind = std::move(*sourceKind);
  contract.sourceAuthority = std::move(*sourceAuthority);
  contract.runtimeExtentArg = std::move(*runtimeExtentArg);
  contract.sourceLoopStep = *kernelLoopStep;
  contract.sourceVectorChunkExtent = *kernelChunk;
  contract.activeLaneAuthority = std::move(*activeLaneAuthority);
  contract.sourceTailPolicy = std::move(*sourceTailPolicy);
  contract.runtimeElementCountConstraint = std::move(*constraint);
  if (!contract.isValid())
    return makeDynamicVectorRuntimeExtentError(
        kernel, "dynamic vector runtime extent contract is incomplete");
  return std::optional<DynamicVectorRuntimeExtentContract>(
      std::move(contract));
}

inline const FiniteBinaryFrontendContract &
getI32VAddFiniteBinaryFrontendContract() {
  static const FiniteBinaryFrontendContract contract{
      FiniteBinaryElementKind::I32,
      "i32",
      32,
      "i32-vadd",
      "i32-vadd",
      "const int32_t *",
      "int32_t *"};
  return contract;
}

inline const FiniteBinaryFrontendContract &
getI32VSubFiniteBinaryFrontendContract() {
  static const FiniteBinaryFrontendContract contract{
      FiniteBinaryElementKind::I32,
      "i32",
      32,
      "i32-vsub",
      "i32-vsub",
      "const int32_t *",
      "int32_t *"};
  return contract;
}

inline const FiniteBinaryFrontendContract &
getI32VMulFiniteBinaryFrontendContract() {
  static const FiniteBinaryFrontendContract contract{
      FiniteBinaryElementKind::I32,
      "i32",
      32,
      "i32-vmul",
      "i32-vmul",
      "const int32_t *",
      "int32_t *"};
  return contract;
}

inline const FiniteBinaryFrontendContract &
getI64VAddFiniteBinaryFrontendContract() {
  static const FiniteBinaryFrontendContract contract{
      FiniteBinaryElementKind::I64,
      "i64",
      64,
      "i64-vadd",
      "i64-vadd",
      "const int64_t *",
      "int64_t *"};
  return contract;
}

inline const FiniteBinaryFrontendContract &
getI64VSubFiniteBinaryFrontendContract() {
  static const FiniteBinaryFrontendContract contract{
      FiniteBinaryElementKind::I64,
      "i64",
      64,
      "i64-vsub",
      "i64-vsub",
      "const int64_t *",
      "int64_t *"};
  return contract;
}

inline const FiniteBinaryFrontendContract &
getI64VMulFiniteBinaryFrontendContract() {
  static const FiniteBinaryFrontendContract contract{
      FiniteBinaryElementKind::I64,
      "i64",
      64,
      "i64-vmul",
      "i64-vmul",
      "const int64_t *",
      "int64_t *"};
  return contract;
}

inline llvm::SmallVector<RuntimeABIMemWindowSpec, 3>
getFiniteBinaryFrontendBufferMemWindowSpecs(
    const FiniteBinaryFrontendContract &contract) {
  llvm::SmallVector<RuntimeABIMemWindowSpec, 3> specs;
  specs.push_back(RuntimeABIMemWindowSpec(
      "abi_lhs_input_buffer", RuntimeABIParameterRole::LHSInputBuffer,
      kRuntimeABIReadAccess,
      stringifyRuntimeABIParameterOwnership(
          RuntimeABIParameterOwnership::TargetExportABIOwned),
      contract.constInputPointerCType));
  specs.push_back(RuntimeABIMemWindowSpec(
      "abi_rhs_input_buffer", RuntimeABIParameterRole::RHSInputBuffer,
      kRuntimeABIReadAccess,
      stringifyRuntimeABIParameterOwnership(
          RuntimeABIParameterOwnership::TargetExportABIOwned),
      contract.constInputPointerCType));
  specs.push_back(RuntimeABIMemWindowSpec(
      "abi_output_buffer", RuntimeABIParameterRole::OutputBuffer,
      kRuntimeABIWriteAccess,
      stringifyRuntimeABIParameterOwnership(
          RuntimeABIParameterOwnership::TargetExportABIOwned),
      contract.outputPointerCType));
  return specs;
}

inline RuntimeABIParamSpec getFiniteBinaryFrontendRuntimeElementCountParamSpec(
    llvm::StringRef cName = "n") {
  return RuntimeABIParamSpec(
      "abi_runtime_element_count",
      RuntimeABIParameterRole::RuntimeElementCount, cName, "size_t",
      stringifyRuntimeABIParameterOwnership(
          RuntimeABIParameterOwnership::TargetExportABIOwned));
}

inline llvm::SmallVector<RuntimeABIParamSpec, 1>
getFiniteBinaryFrontendRuntimeElementCountParamSpecs(
    const FiniteBinaryFrontendContract &contract,
    llvm::StringRef cName = "n") {
  (void)contract;
  llvm::SmallVector<RuntimeABIParamSpec, 1> specs;
  specs.push_back(getFiniteBinaryFrontendRuntimeElementCountParamSpec(cName));
  return specs;
}

inline FiniteBinarySourceFrontendLoweringContract
makeFiniteBinarySourceFrontendLoweringContract(
    const FiniteBinaryFrontendContract &contract) {
  FiniteBinarySourceFrontendLoweringContract sourceContract;
  sourceContract.contract = &contract;
  sourceContract.bufferMemWindowSpecs =
      getFiniteBinaryFrontendBufferMemWindowSpecs(contract);
  sourceContract.runtimeElementCountSpecs =
      getFiniteBinaryFrontendRuntimeElementCountParamSpecs(contract);
  return sourceContract;
}

inline FiniteBinarySourceFrontendLoweringContract
makeFixedVectorI32VAddSourceFrontendLoweringContract() {
  FiniteBinarySourceFrontendLoweringContract sourceContract =
      makeFiniteBinarySourceFrontendLoweringContract(
          getI32VAddFiniteBinaryFrontendContract());
  sourceContract.fixedSourceVectorExtent =
      kFrontendFixedVectorI32VAddSourceExtent;
  return sourceContract;
}

inline FiniteBinarySourceFrontendLoweringContract
makeDynamicVectorI32VAddSourceFrontendLoweringContract() {
  FiniteBinarySourceFrontendLoweringContract sourceContract =
      makeFiniteBinarySourceFrontendLoweringContract(
          getI32VAddFiniteBinaryFrontendContract());
  sourceContract.dynamicRuntimeExtentFromSCFUpperBound = true;
  return sourceContract;
}

inline FiniteBinarySourceFrontendLoweringContract
makeDynamicVectorI32SourceFrontendLoweringContract(
    const FiniteBinaryFrontendContract &contract) {
  FiniteBinarySourceFrontendLoweringContract sourceContract =
      makeFiniteBinarySourceFrontendLoweringContract(contract);
  sourceContract.dynamicRuntimeExtentFromSCFUpperBound = true;
  return sourceContract;
}

} // namespace tianchenrv::support

#endif // TIANCHENRV_SUPPORT_FINITEBINARYFRONTENDLOWERING_H
