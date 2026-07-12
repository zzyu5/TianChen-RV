#include "Weft/Target/RVV/RVVTargetSupportBundle.h"

#include "Weft/Conversion/EmitC/WEFTEmitCLowerableInterface.h"
#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/RVV/IR/RVVConfigContract.h"
#include "Weft/Plugin/ExtensionBundle.h"
#include "Weft/Plugin/ExtensionPlugin.h"
#include "Weft/Plugin/RVV/RVVConstructionProtocol.h"
#include "Weft/Plugin/RVV/RVVEmitCRouteProvider.h"
#include "Weft/Plugin/RVV/RVVMonolithicBlockDotFamily.h"
#include "Weft/Support/CapabilityModel.h"
#include "Weft/Target/ConstructionTemplateArtifactAdapter.h"
#include "Weft/Target/TargetArtifactExport.h"
#include "Weft/Target/TargetTranslateRegistration.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

namespace weft::target::rvv {
namespace {

struct ScopedTempPath {
  llvm::SmallString<128> path;

  ~ScopedTempPath() {
    if (!path.empty())
      (void)llvm::sys::fs::remove(path);
  }
};

// A scoped temporary DIRECTORY (recursively removed on destruction). Used to host
// the freestanding libm declaration shim consumed by the RISC-V object-packaging
// clang -isystem search path (see compileRVVGeneratedSourceToObject).
struct ScopedTempDir {
  llvm::SmallString<128> path;

  ~ScopedTempDir() {
    if (!path.empty())
      (void)llvm::sys::fs::remove_directories(path);
  }
};

// A minimal freestanding declaration shim for the scalar libm functions the RVV
// EmitC bodies call. The local RISC-V cross clang has NO libc sysroot, so the
// hosted <math.h> the libm-calling bodies emit (nvfp4's ldexpf UE4M3 scale
// decode; rms_norm's 1/sqrtf; rope's cosf/sinf) is not found and the -c compile
// fatals. A compile-to-object (no link) needs only these declarations; the
// platform's own libm is linked on the target, so the object's libm symbols stay
// undefined externals either way. The emitted C/C++ source is left byte-identical
// -- only clang's header search is augmented, and only bodies that #include a
// hosted header ever consult the shim.
constexpr llvm::StringLiteral kFreestandingLibmShimHeader =
    "#ifndef WEFT_RVV_FREESTANDING_LIBM_SHIM\n"
    "#define WEFT_RVV_FREESTANDING_LIBM_SHIM\n"
    "#ifdef __cplusplus\n"
    "extern \"C\" {\n"
    "#endif\n"
    "float ldexpf(float, int);\n"
    "float sqrtf(float);\n"
    "float cosf(float);\n"
    "float sinf(float);\n"
    "float expf(float);\n"
    "double ldexp(double, int);\n"
    "double sqrt(double);\n"
    "#ifdef __cplusplus\n"
    "}\n"
    "#endif\n"
    "#endif\n";

llvm::Error makeRVVTargetRouteError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV RVV materialized EmitC target artifact bridge "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

const plugin::rvv::RVVConstructionManifest &getRVVManifest() {
  return plugin::rvv::getRVVConstructionManifest();
}

const plugin::rvv::RVVSelectedBodyTargetArtifactMapping &
getRVVTargetMapping() {
  return plugin::rvv::getRVVSelectedBodyTargetArtifactMapping();
}

llvm::Error requireCandidateField(llvm::StringRef fieldName,
                                  llvm::StringRef actual,
                                  llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVTargetRouteError(llvm::Twine("candidate ") + fieldName +
                                 " must be '" + expected + "' but was '" +
                                 actual + "'");
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

struct RVVSelectedVariantRouteValidation {
  plugin::rvv::RVVSelectedBodyEmitCRouteDescription description;
};

llvm::Error validateRVVRouteMetadataMirrorsSelectedBody(
    const TargetArtifactCandidate &candidate,
    llvm::StringRef selectedBodyRouteID,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  // Stage 3 换心 decouple (C2): `selectedBodyRouteID` is the selected body's
  // EmitC route id. For the legacy string path it is the rebuilt route's
  // `getRouteID()`; for the converted path it is `description.emitCRouteID` —
  // the SAME value the route is constructed from
  // (RVVEmitCRouteProvider.cpp builds the route id from
  // `analysis.description.emitCRouteID`), so this mirror check is identical
  // either way and no longer needs the owner-built string route.
  llvm::StringRef routeID = lookupCandidateMetadataValue(
      candidate, plugin::rvv::getRVVEmitCLowerableRouteMetadataName());
  llvm::StringRef selectedBodyOperation = lookupCandidateMetadataValue(
      candidate, plugin::rvv::getRVVSelectedBodyOperationMetadataName());
  if (routeID.empty())
    return makeRVVTargetRouteError(
        llvm::Twine("candidate metadata must carry ") +
        plugin::rvv::getRVVEmitCLowerableRouteMetadataName() +
        " provenance");
  if (selectedBodyRouteID != routeID)
    return makeRVVTargetRouteError(
        llvm::Twine("candidate ") +
        plugin::rvv::getRVVEmitCLowerableRouteMetadataName() +
        " provenance must mirror selected typed RVV body route '" +
        selectedBodyRouteID + "' but was '" + routeID + "'");

  if (selectedBodyOperation.empty())
    return makeRVVTargetRouteError(
        llvm::Twine("candidate metadata must carry ") +
        plugin::rvv::getRVVSelectedBodyOperationMetadataName() +
        " provenance");
  llvm::StringRef expectedOperation =
      plugin::rvv::stringifyRVVSelectedBodyOperationKind(
          description.operation);
  if (selectedBodyOperation != expectedOperation)
    return makeRVVTargetRouteError(
        llvm::Twine("candidate ") +
        plugin::rvv::getRVVSelectedBodyOperationMetadataName() +
        " provenance must mirror selected typed RVV body operation '" +
        expectedOperation + "' but was '" + selectedBodyOperation + "'");
  llvm::StringRef comparePredicateKind = lookupCandidateMetadataValue(
      candidate, "weft_rvv.compare_predicate_kind");
  if (!description.comparePredicateKind.empty()) {
    if (comparePredicateKind.empty())
      return makeRVVTargetRouteError(
          "candidate metadata must carry "
          "weft_rvv.compare_predicate_kind provenance");
    if (comparePredicateKind != description.comparePredicateKind)
      return makeRVVTargetRouteError(
          llvm::Twine("candidate weft_rvv.compare_predicate_kind provenance "
                      "must mirror selected typed RVV body predicate '") +
          description.comparePredicateKind + "' but was '" +
          comparePredicateKind + "'");
  } else if (!comparePredicateKind.empty()) {
    return makeRVVTargetRouteError(
        "candidate metadata must not carry compare predicate mirrors for a "
        "selected typed RVV body route without a compare predicate");
  }
  llvm::StringRef routeOperandBindingPlan = lookupCandidateMetadataValue(
      candidate, "weft_rvv.route_operand_binding_plan");
  llvm::StringRef routeOperandBindingOperands = lookupCandidateMetadataValue(
      candidate, "weft_rvv.route_operand_binding_operands");
  if (!description.routeOperandBindingPlanID.empty()) {
    if (routeOperandBindingPlan.empty())
      return makeRVVTargetRouteError(
          "candidate metadata must carry "
          "weft_rvv.route_operand_binding_plan provenance");
    if (routeOperandBindingPlan != description.routeOperandBindingPlanID)
      return makeRVVTargetRouteError(
          llvm::Twine("candidate weft_rvv.route_operand_binding_plan "
                      "provenance must mirror selected typed RVV body binding "
                      "plan '") +
          description.routeOperandBindingPlanID + "' but was '" +
          routeOperandBindingPlan + "'");
    if (routeOperandBindingOperands.empty())
      return makeRVVTargetRouteError(
          "candidate metadata must carry "
          "weft_rvv.route_operand_binding_operands provenance");
    if (routeOperandBindingOperands != description.routeOperandBindingSummary)
      return makeRVVTargetRouteError(
          llvm::Twine("candidate weft_rvv.route_operand_binding_operands "
                      "provenance must mirror selected typed RVV body binding "
                      "summary '") +
          description.routeOperandBindingSummary + "' but was '" +
          routeOperandBindingOperands + "'");
  } else if (!routeOperandBindingPlan.empty() ||
             !routeOperandBindingOperands.empty()) {
    return makeRVVTargetRouteError(
        "candidate metadata must not carry route operand binding mirrors for a "
        "selected typed RVV body route without a binding plan");
  }

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "weft_rvv.exec_abi_bindings",
          description.execABIBindingSummary,
          "selected weft.exec ABI binding summary"))
    return error;

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "weft_rvv.provider_supported_mirror",
          description.providerSupportedMirror,
          "selected typed RVV body provider support"))
    return error;

  llvm::StringRef targetCapabilityProviderMirror =
      lookupCandidateMetadataValue(
          candidate, "weft_rvv.target_capability_provider_mirror");
  llvm::StringRef targetCapabilityLegalityMirror =
      lookupCandidateMetadataValue(
          candidate, "weft_rvv.target_capability_legality_mirror");
  if (!description.targetCapabilityProviderMirror.empty()) {
    if (targetCapabilityProviderMirror.empty())
      return makeRVVTargetRouteError(
          "candidate metadata must carry "
          "weft_rvv.target_capability_provider_mirror provenance");
    if (targetCapabilityProviderMirror !=
        llvm::StringRef(description.targetCapabilityProviderMirror))
      return makeRVVTargetRouteError(
          llvm::Twine(
              "candidate weft_rvv.target_capability_provider_mirror "
              "provenance must mirror selected RVV target capability facts '") +
          description.targetCapabilityProviderMirror + "' but was '" +
          targetCapabilityProviderMirror + "'");
  } else if (!targetCapabilityProviderMirror.empty()) {
    return makeRVVTargetRouteError(
        "candidate metadata must not carry target capability provider mirrors "
        "for a selected typed RVV body route without provider facts");
  }
  if (!description.targetCapabilityLegalityMirror.empty()) {
    if (targetCapabilityLegalityMirror.empty())
      return makeRVVTargetRouteError(
          "candidate metadata must carry "
          "weft_rvv.target_capability_legality_mirror provenance");
    if (targetCapabilityLegalityMirror !=
        llvm::StringRef(description.targetCapabilityLegalityMirror))
      return makeRVVTargetRouteError(
          llvm::Twine(
              "candidate weft_rvv.target_capability_legality_mirror "
              "provenance must mirror selected RVV target capability facts '") +
          description.targetCapabilityLegalityMirror + "' but was '" +
          targetCapabilityLegalityMirror + "'");
  } else if (!targetCapabilityLegalityMirror.empty()) {
    return makeRVVTargetRouteError(
        "candidate metadata must not carry target capability legality mirrors "
        "for a selected typed RVV body route without provider facts");
  }
  llvm::StringRef selectedDispatchCaseMirror = lookupCandidateMetadataValue(
      candidate, "weft_rvv.selected_dispatch_case_mirror");
  llvm::StringRef selectedDispatchFallbackMirror = lookupCandidateMetadataValue(
      candidate, "weft_rvv.selected_dispatch_fallback_mirror");
  if (!description.selectedDispatchCaseMirror.empty()) {
    if (selectedDispatchCaseMirror.empty())
      return makeRVVTargetRouteError(
          "candidate metadata must carry "
          "weft_rvv.selected_dispatch_case_mirror provenance");
    if (selectedDispatchCaseMirror !=
        llvm::StringRef(description.selectedDispatchCaseMirror))
      return makeRVVTargetRouteError(
          llvm::Twine("candidate weft_rvv.selected_dispatch_case_mirror "
                      "provenance must mirror selected dispatch case facts '") +
          description.selectedDispatchCaseMirror + "' but was '" +
          selectedDispatchCaseMirror + "'");
  } else if (!selectedDispatchCaseMirror.empty()) {
    return makeRVVTargetRouteError(
        "candidate metadata must not carry selected dispatch case mirrors for "
        "a selected typed RVV body route without dispatch-case envelope facts");
  }
  if (!description.selectedDispatchFallbackMirror.empty()) {
    if (selectedDispatchFallbackMirror.empty())
      return makeRVVTargetRouteError(
          "candidate metadata must carry "
          "weft_rvv.selected_dispatch_fallback_mirror provenance");
    if (selectedDispatchFallbackMirror !=
        llvm::StringRef(description.selectedDispatchFallbackMirror))
      return makeRVVTargetRouteError(
          llvm::Twine(
              "candidate weft_rvv.selected_dispatch_fallback_mirror "
              "provenance must mirror selected dispatch fallback facts '") +
          description.selectedDispatchFallbackMirror + "' but was '" +
          selectedDispatchFallbackMirror + "'");
  } else if (!selectedDispatchFallbackMirror.empty()) {
    return makeRVVTargetRouteError(
        "candidate metadata must not carry selected dispatch fallback mirrors "
        "for a selected typed RVV body route without dispatch fallback facts");
  }
  return llvm::Error::success();
}

llvm::Expected<plugin::VariantEmissionRole>
parseCandidateEmissionRole(const TargetArtifactCandidate &candidate) {
  if (candidate.role ==
      plugin::stringifyVariantEmissionRole(
          plugin::VariantEmissionRole::DirectVariant))
    return plugin::VariantEmissionRole::DirectVariant;
  if (candidate.role ==
      plugin::stringifyVariantEmissionRole(
          plugin::VariantEmissionRole::DispatchCase))
    return plugin::VariantEmissionRole::DispatchCase;
  if (candidate.role ==
      plugin::stringifyVariantEmissionRole(
          plugin::VariantEmissionRole::DispatchFallback))
    return plugin::VariantEmissionRole::DispatchFallback;

  return makeRVVTargetRouteError(
      llvm::Twine("candidate selected path role '") + candidate.role +
      "' is not supported by the RVV materialized EmitC artifact bridge");
}

llvm::Expected<weft::exec::VariantOp>
resolveCandidateSelectedVariant(const TargetArtifactCandidate &candidate) {
  if (!candidate.kernel)
    return makeRVVTargetRouteError(
        "candidate selected variant cannot be cross-checked without an "
        "enclosing weft.exec.kernel");
  weft::exec::KernelOp kernel = candidate.kernel;
  if (kernel.getBody().empty())
    return makeRVVTargetRouteError(
        "candidate selected variant cannot be cross-checked because the "
        "enclosing weft.exec.kernel body is empty");
  if (candidate.selectedVariant.empty())
    return makeRVVTargetRouteError(
        "candidate selected variant must be non-empty before RVV artifact "
        "route cross-check");

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto variant = llvm::dyn_cast<weft::exec::VariantOp>(op);
    if (variant && variant.getSymName() == candidate.selectedVariant)
      return variant;
  }

  return makeRVVTargetRouteError(
      llvm::Twine("candidate selected variant @") +
      candidate.selectedVariant +
      " must resolve to a direct sibling weft.exec.variant before RVV "
      "artifact route cross-check");
}

llvm::Expected<RVVSelectedVariantRouteValidation>
validateRVVSelectedVariantRouteAgreesWithCandidate(
    const TargetArtifactCandidate &candidate) {
  if (!candidate.kernel)
    return makeRVVTargetRouteError(
        "selected RVV materialized EmitC candidate requires an enclosing "
        "weft.exec.kernel so target export can validate the selected typed "
        "weft_rvv body before consuming route metadata");

  llvm::Expected<weft::exec::VariantOp> selectedVariant =
      resolveCandidateSelectedVariant(candidate);
  if (!selectedVariant)
    return selectedVariant.takeError();

  llvm::Expected<support::TargetCapabilitySet> capabilities =
      support::TargetCapabilitySet::buildFromKernelChecked(candidate.kernel);
  if (!capabilities)
    return capabilities.takeError();

  llvm::Expected<plugin::VariantEmissionRole> role =
      parseCandidateEmissionRole(candidate);
  if (!role)
    return role.takeError();

  plugin::VariantEmitCLowerableRequest request(
      *selectedVariant, candidate.kernel, *capabilities, *role);

  // Stage 3 换心 decouple (C2, candidate validation — header AND object
  // export). For a converted family the real RVV->emitc DialectConversion is
  // the authority; rebuilding the legacy string route here (and dispatching
  // into its per-family statement-plan owner) is redundant. So gate on the
  // try-convert: a converted family validates from the route DESCRIPTION alone
  // (owner-free), a not-yet-converted family keeps rebuilding + verifying its
  // string route exactly as before.
  if (plugin::rvv::rvvSelectedBodyFullyConvertsToEmitC(request)) {
    llvm::Expected<plugin::rvv::RVVSelectedBodyEmitCRouteDescription>
        description = plugin::rvv::describeRVVSelectedBodyEmitCRoute(request);
    if (!description) {
      llvm::Error error = description.takeError();
      std::string message = llvm::toString(std::move(error));
      return makeRVVTargetRouteError(
          llvm::Twine("selected typed RVV body could not describe the "
                      "materialized EmitC route before candidate metadata "
                      "validation: ") +
          message);
    }

    // The route id mirror reads `description.emitCRouteID` — the exact value
    // the string route's getRouteID() carries (it is constructed from it), so
    // the mirror is byte-identical without building the route.
    if (llvm::Error error = validateRVVRouteMetadataMirrorsSelectedBody(
            candidate, description->emitCRouteID, *description))
      return std::move(error);

    // The two former route-only checks are subsumed for the converted path:
    //  - source provenance ("exactly one weft_rvv.with_vl scope") is
    //    guaranteed by convertRVVModuleToEmitC fully legalizing the body — the
    //    conversion target only illegalizes a variant that carries a with_vl
    //    boundary and the elementwise pattern lowers exactly that one scope, so
    //    a fully-converted family necessarily had the unique with_vl boundary.
    //  - ABI mappings == candidate.runtimeABIParameters is re-validated
    //    directly against the same source data by the caller, which already
    //    compares candidate.runtimeABIParameters against
    //    description.runtimeABIParameters (the route's ABI mappings are built
    //    1:1 from description.runtimeABIParameters), so dropping the route ABI
    //    mapping check loses no coverage.
    RVVSelectedVariantRouteValidation validation;
    validation.description = std::move(*description);
    return validation;
  }

  // Stage 1 (description-engine retirement): a selected RVV body that does NOT
  // fully convert through the RVV->emitc DialectConversion has no legal
  // materialized route. The legacy string statement-plan route is retired, so
  // refuse it fail-closed here (the same fail-closed decision the former route
  // build-and-verify enforced), preserving the original diagnostic framing.
  llvm::Error refusal =
      plugin::rvv::refuseRetiredRVVSelectedBodyStringRoute(request);
  std::string message = llvm::toString(std::move(refusal));
  return makeRVVTargetRouteError(
      llvm::Twine("selected typed RVV body could not build the "
                  "materialized EmitC route before candidate metadata "
                  "validation: ") +
      message);
}

llvm::Error rejectForbiddenRVVArtifactMetadata(
    const TargetArtifactCandidate &candidate) {
  for (const support::ArtifactMetadataEntry &entry :
       candidate.artifactMetadata) {
    llvm::StringRef key(entry.key);
    std::string lowerKeyStorage = key.lower();
    llvm::StringRef lowerKey(lowerKeyStorage);
    std::string lowerValueStorage = llvm::StringRef(entry.value).lower();
    llvm::StringRef lowerValue(lowerValueStorage);
    if (lowerKey.contains("element_count") ||
        lowerKey.contains("element-count") ||
        lowerKey.contains("descriptor") ||
        containsForbiddenDirectCMarker(lowerKey) ||
        lowerKey.contains("source_export") ||
        lowerKey.contains("source-export") ||
        lowerKey.contains("compute_body") ||
        lowerKey.contains("compute-body") ||
        lowerValue.contains("descriptor") ||
        containsForbiddenDirectCMarker(lowerValue) ||
        lowerValue.contains("source_export") ||
        lowerValue.contains("source-export") ||
        lowerValue.contains("compute_body") ||
        lowerValue.contains("compute-body"))
      return makeRVVTargetRouteError(
          llvm::Twine("candidate artifact metadata key '") + key +
          "' attempts to reintroduce descriptor-driven computation, "
          "direct C/source-export authority, compute-body metadata, or "
          "hardcoded element-count residue");
  }
  return llvm::Error::success();
}

llvm::Error validateRVVConstructionArtifactMetadata(
    const TargetArtifactCandidate &candidate,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  llvm::SmallVector<support::ArtifactMetadataEntry, 16> rvvMetadata;
  for (const support::ArtifactMetadataEntry &entry :
       candidate.artifactMetadata) {
    if (llvm::StringRef(entry.key).starts_with("rvv_"))
      rvvMetadata.push_back(entry);
  }
  plugin::rvv::RVVSelectedBodyConstructionMetadataFacts facts =
      plugin::rvv::getRVVSelectedBodyConstructionMetadataFacts(description);
  return plugin::rvv::
      verifyRVVSelectedBodyConstructionArtifactMetadata(
          rvvMetadata, facts, "selected RVV materialized EmitC candidate");
}

llvm::Error validateRVVConfigArtifactMetadataMirrorsSelectedBody(
    const TargetArtifactCandidate &candidate,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  llvm::SmallVector<support::ArtifactMetadataEntry, 16> actual;
  for (const support::ArtifactMetadataEntry &entry :
       candidate.artifactMetadata) {
    if (llvm::StringRef(entry.key).starts_with("weft_rvv."))
      actual.push_back(entry);
  }

  llvm::SmallVector<support::ArtifactMetadataEntry, 16> expected =
      plugin::rvv::getRVVSelectedBodyConfigArtifactMetadata(description);
  if (support::artifactMetadataEntriesEqual(actual, expected))
    return llvm::Error::success();

  if (actual.size() != expected.size())
    return makeRVVTargetRouteError(
        llvm::Twine("candidate metadata must carry exactly ") +
        llvm::Twine(expected.size()) +
        " weft_rvv selected-body config/runtime-VL artifact metadata entries "
        "derived from the provider route description");

  for (auto [index, pair] : llvm::enumerate(llvm::zip(actual, expected))) {
    const support::ArtifactMetadataEntry &got = std::get<0>(pair);
    const support::ArtifactMetadataEntry &want = std::get<1>(pair);
    if (got.key != want.key)
      return makeRVVTargetRouteError(
          llvm::Twine("candidate weft_rvv selected-body metadata[") +
          llvm::Twine(index) + "] key must mirror provider route description "
                               "key '" +
          want.key + "' but was '" + got.key + "'");
    if (got.value != want.value)
      return makeRVVTargetRouteError(
          llvm::Twine("candidate weft_rvv selected-body metadata key '") +
          want.key + "' must mirror provider route description value '" +
          want.value + "' but was '" + got.value + "'");
  }

  return makeRVVTargetRouteError(
      "candidate weft_rvv selected-body config/runtime-VL artifact metadata "
      "must mirror the provider route description");
}

llvm::Error validateRVVRuntimeAVLVLArtifactMetadata(
    const TargetArtifactCandidate &candidate,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (llvm::Error error = rejectForbiddenRVVArtifactMetadata(candidate))
    return error;
  if (llvm::Error error =
          validateRVVConstructionArtifactMetadata(candidate, description))
    return error;

  return validateRVVConfigArtifactMetadataMirrorsSelectedBody(candidate,
                                                             description);
}

llvm::Error validateRVVSelectedBodyTargetArtifactCandidate(
    const TargetArtifactCandidate &candidate) {
  if (llvm::Error error = plugin::rvv::verifyRVVConstructionProtocolReady())
    return error;

  if (candidate.role == "dispatch fallback")
    return makeRVVTargetRouteError(
        "selected RVV materialized EmitC candidate must not be fallback-only");
  if (llvm::Error error = requireCandidateField(
          "origin", candidate.origin,
          plugin::rvv::getRVVConstructionManifest().family.pluginName))
    return error;
  if (llvm::Error error =
          requireCandidateField("emission kind", candidate.emissionKind,
                                plugin::rvv::getRVVSelectedBodyEmissionKind()))
    return error;
  if (llvm::Error error = requireCandidateField(
          "lowering boundary", candidate.loweringBoundary,
          plugin::rvv::getRVVSelectedBodyLoweringBoundaryOpName()))
    return error;
  if (llvm::Error error = requireCandidateField(
          "runtime ABI kind", candidate.runtimeABIKind,
          plugin::rvv::getRVVSelectedBodyRuntimeABIKind()))
    return error;
  if (llvm::Error error = requireCandidateField(
          "runtime glue role", candidate.runtimeGlueRole,
          plugin::rvv::getRVVSelectedBodyRuntimeGlueRole()))
    return error;

  llvm::Expected<RVVSelectedVariantRouteValidation> selectedRoute =
      validateRVVSelectedVariantRouteAgreesWithCandidate(candidate);
  if (!selectedRoute)
    return selectedRoute.takeError();

  if (!support::runtimeABIParametersEqual(
          candidate.runtimeABIParameters,
          selectedRoute->description.runtimeABIParameters))
    return makeRVVTargetRouteError(
        "candidate runtime ABI parameters must mirror the provider-derived "
        "selected-body runtime ABI signature");

  if (llvm::Error error = requireCandidateField(
          "route id", candidate.routeID,
          selectedRoute->description.targetArtifactRouteID))
    return error;
  if (llvm::Error error = requireCandidateField(
          "artifact kind", candidate.artifactKind,
          selectedRoute->description.targetArtifactKind))
    return error;

  llvm::StringRef selectedBodyRuntimeABIName =
      selectedRoute->description.runtimeABIName;
  if (llvm::Error error = requireCandidateField(
          "runtime ABI", candidate.runtimeABI, selectedBodyRuntimeABIName))
    return error;
  if (llvm::Error error = requireCandidateField(
          "runtime ABI name", candidate.runtimeABIName,
          selectedBodyRuntimeABIName))
    return error;
  if (llvm::Error error = validateRVVRuntimeAVLVLArtifactMetadata(
          candidate, selectedRoute->description))
    return error;
  return llvm::Error::success();
}

void appendRVVConfigVLMetadataEvidence(
    llvm::SmallVectorImpl<MaterializedEmitCHeaderArtifactMetadataEvidence>
        &out) {
  constexpr llvm::StringLiteral kRVVMetadataPrefix("weft_rvv.");
  for (const support::ArtifactMetadataEntry &entry :
       weft::rvv::getRVVSelectedBodyConfigArtifactMetadata()) {
    llvm::StringRef key(entry.key);
    llvm::StringRef commentName = key;
    if (key.starts_with(kRVVMetadataPrefix))
      commentName = key.drop_front(kRVVMetadataPrefix.size());
    out.push_back({commentName, key, /*expectedValue=*/"",
                   /*allowDynamicValue=*/true});
  }
}

void appendRVVLowPrecisionMirrorSourceMetadataEvidence(
    llvm::SmallVectorImpl<MaterializedEmitCHeaderArtifactMetadataEvidence> &out,
    plugin::rvv::RVVLowPrecisionMirrorTransportContract contract) {
  out.push_back({contract.headerEvidenceName, contract.metadataKey,
                 contract.sourceValue, /*allowDynamicValue=*/false,
                 /*optional=*/true});
}

void appendRVVLowPrecisionPrimitivePayloadMirrorMetadataEvidence(
    llvm::SmallVectorImpl<MaterializedEmitCHeaderArtifactMetadataEvidence>
        &out) {
  appendRVVLowPrecisionMirrorSourceMetadataEvidence(
      out,
      plugin::rvv::getRVVLowPrecisionPrimitivePayloadMirrorTransportContract());
  constexpr llvm::StringLiteral kPayloadMirrorMetadata[][2] = {
      {"low_precision_primitive.payload_mirror.contract",
       "weft_rvv.low_precision_primitive.contract"},
      {"low_precision_primitive.payload_mirror.kind",
       "weft_rvv.low_precision_primitive.kind"},
      {"low_precision_primitive.payload_mirror.source_dtype",
       "weft_rvv.low_precision_primitive.source_dtype"},
      {"low_precision_primitive.payload_mirror.source_signedness",
       "weft_rvv.low_precision_primitive.source_signedness"},
      {"low_precision_primitive.payload_mirror.source_load",
       "weft_rvv.low_precision_primitive.source_load"},
      {"low_precision_primitive.payload_mirror.source_extension",
       "weft_rvv.low_precision_primitive.source_extension"},
      {"low_precision_primitive.payload_mirror.product_dtype",
       "weft_rvv.low_precision_primitive.product_dtype"},
      {"low_precision_primitive.payload_mirror.accumulator_dtype",
       "weft_rvv.low_precision_primitive.accumulator_dtype"},
      {"low_precision_primitive.payload_mirror.result_dtype",
       "weft_rvv.low_precision_primitive.result_dtype"},
      {"low_precision_primitive.payload_mirror.source_sew",
       "weft_rvv.low_precision_primitive.source_sew"},
      {"low_precision_primitive.payload_mirror.source_lmul",
       "weft_rvv.low_precision_primitive.source_lmul"},
      {"low_precision_primitive.payload_mirror.product_sew",
       "weft_rvv.low_precision_primitive.product_sew"},
      {"low_precision_primitive.payload_mirror.product_lmul",
       "weft_rvv.low_precision_primitive.product_lmul"},
      {"low_precision_primitive.payload_mirror.accumulator_sew",
       "weft_rvv.low_precision_primitive.accumulator_sew"},
      {"low_precision_primitive.payload_mirror.accumulator_lmul",
       "weft_rvv.low_precision_primitive.accumulator_lmul"},
      {"low_precision_primitive.payload_mirror.result_sew",
       "weft_rvv.low_precision_primitive.result_sew"},
      {"low_precision_primitive.payload_mirror.result_lmul",
       "weft_rvv.low_precision_primitive.result_lmul"},
      {"low_precision_primitive.payload_mirror.tail_policy",
       "weft_rvv.low_precision_primitive.tail_policy"},
      {"low_precision_primitive.payload_mirror.mask_policy",
       "weft_rvv.low_precision_primitive.mask_policy"},
      {"low_precision_primitive.payload_mirror.runtime_control_plan",
       "weft_rvv.low_precision_primitive.runtime_control_plan"},
      {"low_precision_primitive.payload_mirror.runtime_avl_source",
       "weft_rvv.low_precision_primitive.runtime_avl_source"},
  };
  for (const auto &entry : kPayloadMirrorMetadata)
    out.push_back({entry[0], entry[1], /*expectedValue=*/"",
                   /*allowDynamicValue=*/true, /*optional=*/true});
}

llvm::SmallVector<MaterializedEmitCHeaderArtifactMetadataEvidence, 32>
buildRVVSelectedBodyHeaderMetadataEvidence() {
  llvm::SmallVector<MaterializedEmitCHeaderArtifactMetadataEvidence, 32>
      evidence;
  evidence.append({
      {"source_ops", plugin::rvv::getRVVSourceOpsMetadataName(),
       plugin::rvv::getRVVSelectedBodySourceOps()},
      {"source_roles", plugin::rvv::getRVVSourceRolesMetadataName(),
       plugin::rvv::getRVVSelectedBodySourceRoles()},
      {"source_op_interface",
       plugin::rvv::getRVVSourceOpInterfaceMetadataName(),
       plugin::rvv::getRVVEmitCLowerableOpInterfaceName()},
      {"construction_protocol",
       plugin::rvv::getRVVConstructionProtocolMetadataName(),
       plugin::rvv::getRVVConstructionManifest().protocolVersion},
      {"extension_archetype",
       plugin::rvv::getRVVConstructionArchetypeMetadataName(),
       plugin::rvv::getRVVConstructionManifest().archetype},
      {"semantic_role_graph", plugin::rvv::getRVVSemanticRoleGraphMetadataName(),
       plugin::rvv::getRVVConstructionManifest().semanticRoleGraph},
      {"common_interface_realization",
       plugin::rvv::getRVVCommonInterfaceRealizationMetadataName(),
       plugin::rvv::getRVVConstructionArtifactInterfaceRealization()},
      {"typed_role_realization",
       plugin::rvv::getRVVTypedRoleRealizationMetadataName(),
       plugin::rvv::getRVVArtifactTypedRoleRealizationSummary()},
      {"emitc_route_mapping",
       plugin::rvv::getRVVEmitCRouteMappingMetadataName(),
       plugin::rvv::getRVVSelectedBodyTargetArtifactRouteID()},
      {"target_artifact_route",
       plugin::rvv::getRVVTargetArtifactRouteMetadataName(),
       plugin::rvv::getRVVSelectedBodyTargetArtifactRouteID()},
      {"target_artifact_kind",
       plugin::rvv::getRVVTargetArtifactKindMetadataName(),
       plugin::rvv::getRVVSelectedBodyTargetArtifactKind()},
      {"evidence_profile", plugin::rvv::getRVVEvidenceProfileMetadataName(),
       plugin::rvv::getRVVConstructionManifest().evidenceProfile},
      {"bundle_component_group",
       plugin::rvv::getRVVBundleComponentGroupMetadataName(),
       plugin::rvv::getRVVSelectedBodyTargetArtifactMapping()
           .bundleComponentGroup},
      {"object_handoff", plugin::rvv::getRVVObjectHandoffMetadataName(),
       plugin::rvv::getRVVSelectedBodyTargetArtifactMapping()
           .objectHandoffKind},
  });
  appendRVVConfigVLMetadataEvidence(evidence);
  evidence.append({
      {"compare_predicate_kind", "weft_rvv.compare_predicate_kind", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"memory_form", "weft_rvv.memory_form", "",
       /*allowDynamicValue=*/true},
      {"strided_memory_layout", "weft_rvv.strided_memory_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"lhs_stride_source", "weft_rvv.lhs_stride_source", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"rhs_stride_source", "weft_rvv.rhs_stride_source", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"source_stride_source", "weft_rvv.source_stride_source", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"destination_stride_source", "weft_rvv.destination_stride_source", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"source_memory_form", "weft_rvv.source_memory_form", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"destination_memory_form", "weft_rvv.destination_memory_form", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"mask_role", "weft_rvv.mask_role", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"mask_source", "weft_rvv.mask_source", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"mask_memory_form", "weft_rvv.mask_memory_form", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"mask_tail_policy_route_family_plan",
       "weft_rvv.mask_tail_policy_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"mask_tail_policy_owner", "weft_rvv.mask_tail_policy_owner", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"secondary_compare_predicate_kind",
       "weft_rvv.secondary_compare_predicate_kind", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"mask_composition", "weft_rvv.mask_composition", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"inactive_lane_contract", "weft_rvv.inactive_lane_contract", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"masked_passthrough_layout", "weft_rvv.masked_passthrough_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"masked_memory_layout", "weft_rvv.masked_memory_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"select_layout", "weft_rvv.select_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"indexed_memory_layout", "weft_rvv.indexed_memory_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"indexed_write_side_contract",
       "weft_rvv.indexed_write_side_contract", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"index_source", "weft_rvv.index_source", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"index_eew", "weft_rvv.index_eew", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"offset_unit", "weft_rvv.offset_unit", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"index_uniqueness", "weft_rvv.index_uniqueness", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"indexed_data_memory_form", "weft_rvv.indexed_data_memory_form", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"indexed_destination_memory_form",
       "weft_rvv.indexed_destination_memory_form", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"source_sew", "weft_rvv.source_sew", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"source_lmul", "weft_rvv.source_lmul", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"product_sew", "weft_rvv.product_sew", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"product_lmul", "weft_rvv.product_lmul", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"product_vector_type", "weft_rvv.product_vector_type", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"product_vector_c_type", "weft_rvv.product_vector_c_type", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"dest_sew", "weft_rvv.dest_sew", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"dest_lmul", "weft_rvv.dest_lmul", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"conversion_relation", "weft_rvv.conversion_relation", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"dequantization_relation", "weft_rvv.dequantization_relation", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"dequantize_convert_intrinsic",
       "weft_rvv.dequantize_convert_intrinsic", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"dequantize_scale_intrinsic",
       "weft_rvv.dequantize_scale_intrinsic", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"dequant_scale_role", "weft_rvv.dequant_scale_role", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"dequant_scale_c_type", "weft_rvv.dequant_scale_c_type", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"dequant_scale_name", "weft_rvv.dequant_scale_name", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
  });
  appendRVVLowPrecisionPrimitivePayloadMirrorMetadataEvidence(evidence);
  appendRVVLowPrecisionMirrorSourceMetadataEvidence(
      evidence,
      plugin::rvv::getRVVLowPrecisionResourceOwnerMirrorTransportContract());
  evidence.append({
      {"low_precision_resource.candidate_set",
       "weft_rvv.low_precision_resource.candidate_set", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.selected_candidate",
       "weft_rvv.low_precision_resource.selected_candidate", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.candidate_count",
       "weft_rvv.low_precision_resource.candidate_count", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.legal_candidate_count",
       "weft_rvv.low_precision_resource.legal_candidate_count", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.selected_candidate_index",
       "weft_rvv.low_precision_resource.selected_candidate_index", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.selection_reason",
       "weft_rvv.low_precision_resource.selection_reason", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.planning_contract",
       "weft_rvv.low_precision_resource.planning_contract", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.legality_scope",
       "weft_rvv.low_precision_resource.legality_scope", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.source_dtype",
       "weft_rvv.low_precision_resource.source_dtype", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.source_sew",
       "weft_rvv.low_precision_resource.source_sew", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.source_lmul",
       "weft_rvv.low_precision_resource.source_lmul", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.operand_form",
       "weft_rvv.low_precision_resource.operand_form", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.source_signedness",
       "weft_rvv.low_precision_resource.source_signedness", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.storage_element_width",
       "weft_rvv.low_precision_resource.storage_element_width", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.effective_element_width",
       "weft_rvv.low_precision_resource.effective_element_width", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.packing_layout",
       "weft_rvv.low_precision_resource.packing_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.unpack_intent",
       "weft_rvv.low_precision_resource.unpack_intent", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.packed_load_unpack_contract",
       "weft_rvv.low_precision_resource.packed_load_unpack_contract", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.packed_storage_load",
       "weft_rvv.low_precision_resource.packed_storage_load", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.packed_unpack_plan",
       "weft_rvv.low_precision_resource.packed_unpack_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.packed_unpacked_source",
       "weft_rvv.low_precision_resource.packed_unpacked_source", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.product_dtype",
       "weft_rvv.low_precision_resource.product_dtype", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.product_sew",
       "weft_rvv.low_precision_resource.product_sew", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.product_lmul",
       "weft_rvv.low_precision_resource.product_lmul", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.product_emul",
       "weft_rvv.low_precision_resource.product_emul", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.accumulator_dtype",
       "weft_rvv.low_precision_resource.accumulator_dtype", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.accumulator_sew",
       "weft_rvv.low_precision_resource.accumulator_sew", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.accumulator_lmul",
       "weft_rvv.low_precision_resource.accumulator_lmul", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.accumulator_emul",
       "weft_rvv.low_precision_resource.accumulator_emul", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.result_dtype",
       "weft_rvv.low_precision_resource.result_dtype", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.result_sew",
       "weft_rvv.low_precision_resource.result_sew", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.result_lmul",
       "weft_rvv.low_precision_resource.result_lmul", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.memory_form",
       "weft_rvv.low_precision_resource.memory_form", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.tail_policy",
       "weft_rvv.low_precision_resource.tail_policy", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.mask_policy",
       "weft_rvv.low_precision_resource.mask_policy", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.unroll_factor",
       "weft_rvv.low_precision_resource.unroll_factor", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.accumulator_count",
       "weft_rvv.low_precision_resource.accumulator_count", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.reduction_layout",
       "weft_rvv.low_precision_resource.reduction_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.vsetvl_region_count",
       "weft_rvv.low_precision_resource.vsetvl_region_count", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.peak_live_vector_groups",
       "weft_rvv.low_precision_resource.peak_live_vector_groups", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.vector_register_budget",
       "weft_rvv.low_precision_resource.vector_register_budget", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.runtime_avl_source",
       "weft_rvv.low_precision_resource.runtime_avl_source", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.runtime_abi_order",
       "weft_rvv.low_precision_resource.runtime_abi_order", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.route_family_plan",
       "weft_rvv.low_precision_resource.route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.provider_supported_mirror",
       "weft_rvv.low_precision_resource.provider_supported_mirror", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.primitive_contract",
       "weft_rvv.low_precision_resource.primitive_contract", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.primitive_kind",
       "weft_rvv.low_precision_resource.primitive_kind", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.primitive_chain_contract",
       "weft_rvv.low_precision_resource.primitive_chain_contract", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.primitive_chain_kind",
       "weft_rvv.low_precision_resource.primitive_chain_kind", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.widening_product_multiplicand_roles",
       "weft_rvv.low_precision_resource."
       "widening_product_multiplicand_roles",
       "", /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.widening_product_extension_policy",
       "weft_rvv.low_precision_resource."
       "widening_product_extension_policy",
       "", /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.widening_product_candidate_fact",
       "weft_rvv.low_precision_resource."
       "widening_product_candidate_fact",
       "", /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.reduction_candidate_fact",
       "weft_rvv.low_precision_resource.reduction_candidate_fact", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.primitive_source_load",
       "weft_rvv.low_precision_resource.primitive_source_load", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.primitive_source_extension",
       "weft_rvv.low_precision_resource.primitive_source_extension", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.primitive_widening_product_relation",
       "weft_rvv.low_precision_resource."
       "primitive_widening_product_relation",
       "", /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.primitive_product_reduction_chain_relation",
       "weft_rvv.low_precision_resource."
       "primitive_product_reduction_chain_relation",
       "", /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.primitive_widening_product_intrinsic",
       "weft_rvv.low_precision_resource."
       "primitive_widening_product_intrinsic",
       "", /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.primitive_reduction_intrinsic",
       "weft_rvv.low_precision_resource.primitive_reduction_intrinsic", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.primitive_scalar_seed_splat_intrinsic",
       "weft_rvv.low_precision_resource."
       "primitive_scalar_seed_splat_intrinsic",
       "", /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.primitive_accumulator_layout",
       "weft_rvv.low_precision_resource.primitive_accumulator_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.primitive_result_layout",
       "weft_rvv.low_precision_resource.primitive_result_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.primitive_reduction_store_vl",
       "weft_rvv.low_precision_resource.primitive_reduction_store_vl", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.realization_producer",
       "weft_rvv.low_precision_resource.realization_producer", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.realization_decision",
       "weft_rvv.low_precision_resource.realization_decision", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "realization_admission_contract",
       "weft_rvv.low_precision_resource.realization_admission_contract", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "realization_admission_decision",
       "weft_rvv.low_precision_resource.realization_admission_decision", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "realization_admission_evidence",
       "weft_rvv.low_precision_resource.realization_admission_evidence", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "realization_admission_dispatch_policy",
       "weft_rvv.low_precision_resource."
       "realization_admission_dispatch_policy",
       "", /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "realization_admission_schedule_decision_contract",
       "weft_rvv.low_precision_resource."
       "realization_admission_schedule_decision_contract",
       "", /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "realization_admission_schedule_decision",
       "weft_rvv.low_precision_resource."
       "realization_admission_schedule_decision",
       "", /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "realization_admission_schedule_decision_reason",
       "weft_rvv.low_precision_resource."
       "realization_admission_schedule_decision_reason",
       "", /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.realized_unroll_factor",
       "weft_rvv.low_precision_resource.realized_unroll_factor", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.realized_vsetvl_region_count",
       "weft_rvv.low_precision_resource.realized_vsetvl_region_count", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.realized_peak_live_vector_groups",
       "weft_rvv.low_precision_resource.realized_peak_live_vector_groups", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.product_region_index",
       "weft_rvv.low_precision_resource.product_region_index", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.dequant_region_index",
       "weft_rvv.low_precision_resource.dequant_region_index", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.product_phase",
       "weft_rvv.low_precision_resource.product_phase", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.dequant_phase",
       "weft_rvv.low_precision_resource.dequant_phase", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.clamp_region_index",
       "weft_rvv.low_precision_resource.clamp_region_index", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.clamp_phase",
       "weft_rvv.low_precision_resource.clamp_phase", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.clamp_compare_select_phase",
       "weft_rvv.low_precision_resource.clamp_compare_select_phase", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.clamp_select_layout",
       "weft_rvv.low_precision_resource.clamp_select_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "performance_feedback",
       "weft_rvv.low_precision_resource.performance_feedback", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "performance_baseline",
       "weft_rvv.low_precision_resource.performance_baseline", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "performance_best_speedup_range",
       "weft_rvv.low_precision_resource.performance_best_speedup_range", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "performance_action",
       "weft_rvv.low_precision_resource.performance_action", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "remediation_handoff_contract",
       "weft_rvv.low_precision_resource.remediation_handoff_contract", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "remediation_diagnosis",
       "weft_rvv.low_precision_resource.remediation_diagnosis", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "remediation_measurement_evidence",
       "weft_rvv.low_precision_resource.remediation_measurement_evidence", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "remediation_decision",
       "weft_rvv.low_precision_resource.remediation_decision", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "remediation_action",
       "weft_rvv.low_precision_resource.remediation_action", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "remediation_dispatch_preference",
       "weft_rvv.low_precision_resource.remediation_dispatch_preference", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "remediation_blocker",
       "weft_rvv.low_precision_resource.remediation_blocker", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "remediation_plan_contract",
       "weft_rvv.low_precision_resource.remediation_plan_contract", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "remediation_plan",
       "weft_rvv.low_precision_resource.remediation_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "remediation_statement_strategy",
       "weft_rvv.low_precision_resource.remediation_statement_strategy", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "remediation_vector_budget",
       "weft_rvv.low_precision_resource.remediation_vector_budget", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "remediation_schedule_contract",
       "weft_rvv.low_precision_resource.remediation_schedule_contract", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "remediation_unpack_plan",
       "weft_rvv.low_precision_resource.remediation_unpack_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "remediation_product_plan",
       "weft_rvv.low_precision_resource.remediation_product_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "remediation_reduction_plan",
       "weft_rvv.low_precision_resource.remediation_reduction_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "remediation_vl_plan",
       "weft_rvv.low_precision_resource.remediation_vl_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.schedule_decision_contract",
       "weft_rvv.low_precision_resource.schedule_decision_contract", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.schedule_decision",
       "weft_rvv.low_precision_resource.schedule_decision", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.schedule_decision_reason",
       "weft_rvv.low_precision_resource.schedule_decision_reason", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.resource_cost_contract",
       "weft_rvv.low_precision_resource.resource_cost_contract", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.resource_cost_model",
       "weft_rvv.low_precision_resource.resource_cost_model", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.resource_cost_loop_body_steps",
       "weft_rvv.low_precision_resource.resource_cost_loop_body_steps", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.resource_cost_blocker",
       "weft_rvv.low_precision_resource.resource_cost_blocker", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "performance_admission_decision",
       "weft_rvv.low_precision_resource.performance_admission_decision", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "performance_admission_closure",
       "weft_rvv.low_precision_resource.performance_admission_closure", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "performance_admission_reopen_requirement",
       "weft_rvv.low_precision_resource."
       "performance_admission_reopen_requirement",
       "", /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "beyond_local_repair_admission_contract",
       "weft_rvv.low_precision_resource."
       "beyond_local_repair_admission_contract",
       "", /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "beyond_local_repair_admission_decision",
       "weft_rvv.low_precision_resource."
       "beyond_local_repair_admission_decision",
       "", /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "beyond_local_repair_admission_blocker",
       "weft_rvv.low_precision_resource."
       "beyond_local_repair_admission_blocker",
       "", /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "beyond_local_repair_admission_reopen_requirement",
       "weft_rvv.low_precision_resource."
       "beyond_local_repair_admission_reopen_requirement",
       "", /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "performance_maturity",
       "weft_rvv.low_precision_resource.performance_maturity", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "performance_maturity_evidence",
       "weft_rvv.low_precision_resource.performance_maturity_evidence", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "performance_maturity_outcome",
       "weft_rvv.low_precision_resource.performance_maturity_outcome", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "performance_selection_eligible",
       "weft_rvv.low_precision_resource.performance_selection_eligible", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.selected_dispatch_policy_output_mirror."
       "selected_dispatch_policy_contract",
       "weft_rvv.low_precision_resource.selected_dispatch_policy_contract",
       "", /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.selected_dispatch_policy_output_mirror."
       "dispatch_policy_path",
       "weft_rvv.low_precision_resource.dispatch_policy_path", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.selected_dispatch_policy_output_mirror."
       "selected_dispatch_preference",
       "weft_rvv.low_precision_resource.selected_dispatch_preference", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.selected_dispatch_policy_output_mirror."
       "performance_preference_denial_reason",
       "weft_rvv.low_precision_resource."
       "performance_preference_denial_reason",
       "", /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.selected_dispatch_policy_output_mirror."
       "fallback_reason",
       "weft_rvv.low_precision_resource.fallback_reason", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.selected_dispatch_policy_output_mirror."
       "route_support_allowed",
       "weft_rvv.low_precision_resource.route_support_allowed", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.selected_dispatch_policy_output_mirror."
       "correctness_execution_allowed",
       "weft_rvv.low_precision_resource.correctness_execution_allowed", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.selected_dispatch_policy_output_mirror."
       "performance_selection_allowed",
       "weft_rvv.low_precision_resource.performance_selection_allowed", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.selected_dispatch_policy_output_mirror."
       "performance_win_claim_allowed",
       "weft_rvv.low_precision_resource.performance_win_claim_allowed", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.selected_dispatch_policy_output_mirror."
       "correctness_fallback_path_selected",
       "weft_rvv.low_precision_resource.correctness_fallback_path_selected",
       "", /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.selected_dispatch_policy_output_mirror."
       "performance_preferred_path_selected",
       "weft_rvv.low_precision_resource.performance_preferred_path_selected",
       "", /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.measurement_disposition_evidence_mirror."
       "dispatch_preference",
       "weft_rvv.low_precision_resource.dispatch_preference", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.target_capability_provider_mirror",
       "weft_rvv.low_precision_resource.target_capability_provider_mirror", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.target_capability_legality_mirror",
       "weft_rvv.low_precision_resource.target_capability_legality_mirror", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.legality",
       "weft_rvv.low_precision_resource.legality", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"low_precision_resource.rejection_reason",
       "weft_rvv.low_precision_resource.rejection_reason", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_route_family_plan",
       "weft_rvv.composite_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_typed_compute_chain",
       "weft_rvv.composite_typed_compute_chain", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.candidate_set",
       "weft_rvv.composite_resource.candidate_set", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.selected_candidate",
       "weft_rvv.composite_resource.selected_candidate", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.selection_reason",
       "weft_rvv.composite_resource.selection_reason", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.legality_scope",
       "weft_rvv.composite_resource.legality_scope", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.operation",
       "weft_rvv.composite_resource.operation", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.memory_form",
       "weft_rvv.composite_resource.memory_form", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.sew", "weft_rvv.composite_resource.sew", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.lmul", "weft_rvv.composite_resource.lmul", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.tail_policy",
       "weft_rvv.composite_resource.tail_policy", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.mask_policy",
       "weft_rvv.composite_resource.mask_policy", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.vl_policy",
       "weft_rvv.composite_resource.vl_policy", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.accumulator_layout",
       "weft_rvv.composite_resource.accumulator_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.unroll_factor",
       "weft_rvv.composite_resource.unroll_factor", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.pipeline_intent",
       "weft_rvv.composite_resource.pipeline_intent", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.prefetch_intent",
       "weft_rvv.composite_resource.prefetch_intent", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.vsetvl_region_count",
       "weft_rvv.composite_resource.vsetvl_region_count", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.peak_live_vector_groups",
       "weft_rvv.composite_resource.peak_live_vector_groups", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.vector_register_budget",
       "weft_rvv.composite_resource.vector_register_budget", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.runtime_avl_source",
       "weft_rvv.composite_resource.runtime_avl_source", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.runtime_abi_order",
       "weft_rvv.composite_resource.runtime_abi_order", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.target_capability_provider_mirror",
       "weft_rvv.composite_resource.target_capability_provider_mirror", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.target_capability_legality_mirror",
       "weft_rvv.composite_resource.target_capability_legality_mirror", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.legality",
       "weft_rvv.composite_resource.legality", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"composite_resource.rejection_reason",
       "weft_rvv.composite_resource.rejection_reason", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"gearbox_candidate_set", "weft_rvv.gearbox.candidate_set", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"gearbox_selected_candidate",
       "weft_rvv.gearbox.selected_candidate", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"gearbox_selection_reason", "weft_rvv.gearbox.selection_reason", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"gearbox_legality_scope", "weft_rvv.gearbox.legality_scope", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"gearbox_schedule_id", "weft_rvv.gearbox.schedule_id", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"gearbox_selector", "weft_rvv.gearbox.selector", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"gearbox_source", "weft_rvv.gearbox.source", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"gearbox_operation", "weft_rvv.gearbox.operation", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"gearbox_unroll", "weft_rvv.gearbox.unroll", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"gearbox_vl_policy", "weft_rvv.gearbox.vl_policy", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"gearbox_source_sew", "weft_rvv.gearbox.source_sew", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"gearbox_source_lmul", "weft_rvv.gearbox.source_lmul", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"gearbox_dest_sew", "weft_rvv.gearbox.dest_sew", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"gearbox_dest_lmul", "weft_rvv.gearbox.dest_lmul", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"gearbox_runtime_avl_source",
       "weft_rvv.gearbox.runtime_avl_source", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"gearbox_producer_scope", "weft_rvv.gearbox.producer_scope", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"gearbox_consumer_scope", "weft_rvv.gearbox.consumer_scope", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"accumulator_sew", "weft_rvv.accumulator_sew", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"accumulator_lmul", "weft_rvv.accumulator_lmul", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"result_sew", "weft_rvv.result_sew", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"result_lmul", "weft_rvv.result_lmul", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"macc_accumulator_layout", "weft_rvv.macc_accumulator_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"macc_result_layout", "weft_rvv.macc_result_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"macc_arithmetic_kind", "weft_rvv.macc_arithmetic_kind", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"widening_macc_accumulator_layout",
       "weft_rvv.widening_macc_accumulator_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"widening_macc_result_layout",
       "weft_rvv.widening_macc_result_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"widening_macc_relation", "weft_rvv.widening_macc_relation", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"widening_product_relation", "weft_rvv.widening_product_relation", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"widening_product_multiplicand_roles",
       "weft_rvv.widening_product_multiplicand_roles", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"widening_product_extension_policy",
       "weft_rvv.widening_product_extension_policy", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"widening_product_intrinsic",
       "weft_rvv.widening_product_intrinsic", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"product_reduction_chain_relation",
       "weft_rvv.product_reduction_chain_relation", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"widening_reduction_intrinsic",
       "weft_rvv.widening_reduction_intrinsic", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"scalar_seed_splat_intrinsic",
       "weft_rvv.scalar_seed_splat_intrinsic", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"widening_dot_accumulator_layout",
       "weft_rvv.widening_dot_accumulator_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"widening_dot_result_layout", "weft_rvv.widening_dot_result_layout",
       "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"widening_dot_relation", "weft_rvv.widening_dot_relation", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"widening_dot_source_accumulator_result_contract",
       "weft_rvv.widening_dot_source_accumulator_result_contract", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"widening_dot_reduction_store_vl",
       "weft_rvv.widening_dot_reduction_store_vl", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"reduction_accumulator_layout",
       "weft_rvv.reduction_accumulator_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"reduction_result_layout", "weft_rvv.reduction_result_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"reduction_kind", "weft_rvv.reduction_kind", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"reduction_store_vl", "weft_rvv.reduction_store_vl", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"target_leaf_profile", "weft_rvv.target_leaf_profile", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"runtime_control_plan", "weft_rvv.runtime_control_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"provider_supported_mirror", "weft_rvv.provider_supported_mirror", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"target_capability_provider_mirror",
       "weft_rvv.target_capability_provider_mirror", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"target_capability_legality_mirror",
       "weft_rvv.target_capability_legality_mirror", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"selected_dispatch_case_mirror",
       "weft_rvv.selected_dispatch_case_mirror", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"selected_dispatch_fallback_mirror",
       "weft_rvv.selected_dispatch_fallback_mirror", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"route_operand_binding_plan",
       "weft_rvv.route_operand_binding_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"route_operand_binding_operands",
       "weft_rvv.route_operand_binding_operands", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"exec_abi_bindings", "weft_rvv.exec_abi_bindings", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"contraction_route_family_plan",
       "weft_rvv.contraction_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"computed_mask_select_route_family_plan",
       "weft_rvv.computed_mask_select_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"computed_mask_select_mask_producer_source",
       "weft_rvv.computed_mask_select_mask_producer_source", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"computed_mask_memory_route_family_plan",
       "weft_rvv.computed_mask_memory_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"computed_mask_memory_mask_producer_source",
       "weft_rvv.computed_mask_memory_mask_producer_source", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"mask_tail_policy_route_family_plan",
       "weft_rvv.mask_tail_policy_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"mask_tail_policy_owner", "weft_rvv.mask_tail_policy_owner", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"secondary_compare_predicate_kind",
       "weft_rvv.secondary_compare_predicate_kind", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"lower_bound_role", "weft_rvv.lower_bound_role", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"upper_bound_role", "weft_rvv.upper_bound_role", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"lower_bound_c_type", "weft_rvv.lower_bound_c_type", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"upper_bound_c_type", "weft_rvv.upper_bound_c_type", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"bound_order", "weft_rvv.bound_order", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"clamp_relation", "weft_rvv.clamp_relation", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"source_memory_form", "weft_rvv.source_memory_form", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"destination_memory_form", "weft_rvv.destination_memory_form", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"select_layout", "weft_rvv.select_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"base_memory_movement_route_family_plan",
       "weft_rvv.base_memory_movement_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"elementwise_arithmetic_route_family_plan",
       "weft_rvv.elementwise_arithmetic_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"scalar_broadcast_elementwise_route_family_plan",
       "weft_rvv.scalar_broadcast_elementwise_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"scalar_broadcast_macc_route_family_plan",
       "weft_rvv.scalar_broadcast_macc_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"plain_macc_route_family_plan",
       "weft_rvv.plain_macc_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"plain_compare_select_route_family_plan",
       "weft_rvv.plain_compare_select_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"segment2_memory_route_family_plan",
       "weft_rvv.segment2_memory_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"widening_conversion_route_family_plan",
       "weft_rvv.widening_conversion_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"dequantization_route_family_plan",
       "weft_rvv.dequantization_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"standalone_reduction_route_family_plan",
       "weft_rvv.standalone_reduction_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"standalone_reduction_source_vector_type",
       "weft_rvv.standalone_reduction_source_vector_type", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"standalone_reduction_source_vector_c_type",
       "weft_rvv.standalone_reduction_source_vector_c_type", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"standalone_reduction_scalar_result_vector_type",
       "weft_rvv.standalone_reduction_scalar_result_vector_type", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"standalone_reduction_scalar_result_vector_c_type",
       "weft_rvv.standalone_reduction_scalar_result_vector_c_type", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"standalone_reduction_scalar_result_runtime_boundary",
       "weft_rvv.standalone_reduction_scalar_result_runtime_boundary", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"accumulation_route_family_plan",
       "weft_rvv.accumulation_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"accumulation_compute_suffix",
       "weft_rvv.accumulation_compute_suffix", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"accumulation_mask_producer_source",
       "weft_rvv.accumulation_mask_producer_source", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"accumulation_accumulator_contract",
       "weft_rvv.accumulation_accumulator_contract", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"accumulation_result_contract",
       "weft_rvv.accumulation_result_contract", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"accumulation_scalar_carry_contract",
       "weft_rvv.accumulation_scalar_carry_contract", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"required_header_declarations",
       "weft_rvv.required_header_declarations", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"c_type_mapping", "weft_rvv.c_type_mapping", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"segment2_update_arithmetic_kind",
       "weft_rvv.segment2_update_arithmetic_kind", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"inactive_lane_zeroing_requirement",
       "weft_rvv.inactive_lane_zeroing_requirement", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
  });
  return evidence;
}

ConstructionTemplateArtifactAdapterConfig
getRVVSelectedBodyArtifactAdapterConfig();

llvm::Error exportMaterializedRVVEmitCToCpp(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  return exportConstructionTemplateEmitCToCpp(
      module, os, getRVVSelectedBodyArtifactAdapterConfig());
}

// Package the generated EmitC C/C++ as a RISC-V RVV relocatable object under an
// explicit target -march. Every existing RVV route packages under the baseline
// rv64gcv (below); the repacked-GEVM route needs rv64gcv_zvfh because its dual-
// fp16 scale fold loads the per-strip fp16 scales as a VECTOR (vle16_v_f16m1 --
// the repack locality win), which is a Zvfh op the baseline march rejects. Only
// the march string differs; the source, headers, ABI, and object are otherwise
// packaged identically (so every non-repack route stays byte-identical under the
// unchanged rv64gcv default).
llvm::Error compileRVVGeneratedSourceToObjectWithMarch(llvm::StringRef source,
                                                       llvm::raw_ostream &os,
                                                       llvm::StringRef marchArg) {
  llvm::ErrorOr<std::string> clang = llvm::sys::findProgramByName("clang");
  if (!clang)
    clang = llvm::sys::findProgramByName(
        "clang", {"/usr/lib/llvm-20/bin", "/usr/local/bin", "/usr/bin"});
  if (!clang)
    return makeRVVTargetRouteError(
        llvm::Twine("requires clang on PATH or a standard LLVM tools path for "
                    "RISC-V object packaging: ") +
        clang.getError().message());

  int sourceFD = -1;
  ScopedTempPath sourcePath;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "weft-rvv-materialized-emitc", "cpp", sourceFD, sourcePath.path))
    return makeRVVTargetRouteError(
        llvm::Twine("failed to create temporary C++ source: ") +
        error.message());
  {
    llvm::raw_fd_ostream sourceOS(sourceFD, /*shouldClose=*/true);
    sourceOS << source;
    sourceOS.close();
    if (sourceOS.has_error())
      return makeRVVTargetRouteError(
          "failed to write generated MLIR EmitC C/C++ source before object "
          "packaging");
  }

  ScopedTempPath objectPath;
  objectPath.path = sourcePath.path;
  llvm::sys::path::replace_extension(objectPath.path, "o");

  int stderrFD = -1;
  ScopedTempPath stderrPath;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "weft-rvv-materialized-emitc-clang", "stderr", stderrFD,
          stderrPath.path))
    return makeRVVTargetRouteError(
        llvm::Twine("failed to create temporary clang stderr file: ") +
        error.message());
  {
    llvm::raw_fd_ostream stderrOS(stderrFD, /*shouldClose=*/true);
    stderrOS.close();
  }

  // Host the freestanding libm declaration shim on an -isystem path so the
  // hosted <math.h> the libm-calling bodies emit resolves under the sysroot-less
  // RISC-V cross clang. See kFreestandingLibmShimHeader: the emitted source is
  // byte-identical; only bodies that #include a hosted header consult the shim.
  ScopedTempDir libmShimDir;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          "weft-rvv-libm-shim", libmShimDir.path))
    return makeRVVTargetRouteError(
        llvm::Twine("failed to create freestanding libm shim directory: ") +
        error.message());
  llvm::SmallString<160> libmShimHeaderPath(libmShimDir.path);
  llvm::sys::path::append(libmShimHeaderPath, "math.h");
  {
    std::error_code error;
    llvm::raw_fd_ostream shimOS(libmShimHeaderPath, error);
    if (error)
      return makeRVVTargetRouteError(
          llvm::Twine("failed to write freestanding libm shim header: ") +
          error.message());
    shimOS << kFreestandingLibmShimHeader;
    shimOS.close();
    if (shimOS.has_error())
      return makeRVVTargetRouteError(
          "failed to flush freestanding libm shim header before object "
          "packaging");
  }

  llvm::SmallVector<llvm::StringRef, 14> args = {
      *clang,
      "-target",
      "riscv64",
      "-O2",
      marchArg,
      "-mabi=lp64d",
      "-isystem",
      libmShimDir.path,
      "-c",
      sourcePath.path,
      "-o",
      objectPath.path};
  llvm::SmallVector<std::optional<llvm::StringRef>, 3> redirects = {
      llvm::StringRef(), llvm::StringRef(), llvm::StringRef(stderrPath.path)};
  std::string executeError;
  bool executionFailed = false;
  int result = llvm::sys::ExecuteAndWait(
      *clang, args, std::nullopt, redirects, /*SecondsToWait=*/30,
      /*MemoryLimit=*/0, &executeError, &executionFailed);
  if (executionFailed || result != 0) {
    std::string stderrText;
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> stderrBuffer =
        llvm::MemoryBuffer::getFile(stderrPath.path);
    if (stderrBuffer)
      stderrText = (*stderrBuffer)->getBuffer().take_front(512).str();
    return makeRVVTargetRouteError(
        llvm::Twine("clang failed to package materialized EmitC C/C++ source "
                    "as a RISC-V RVV relocatable object; exit=") +
        llvm::Twine(result) + " execution_failed=" +
        (executionFailed ? "true" : "false") + " error='" + executeError +
        "' stderr='" + stderrText + "'");
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> objectBuffer =
      llvm::MemoryBuffer::getFile(objectPath.path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!objectBuffer)
    return makeRVVTargetRouteError(
        llvm::Twine("failed to read generated RISC-V object: ") +
        objectBuffer.getError().message());
  if ((*objectBuffer)->getBufferSize() == 0)
    return makeRVVTargetRouteError("generated RISC-V object is empty");
  os << (*objectBuffer)->getBuffer();
  return llvm::Error::success();
}

// The baseline RVV object packager (rv64gcv): the decomposed selected-body route
// uses this -- byte-identical to before the march was parameterized. (The
// monolithic block-dot / repack families now package under the Zvfh variant
// below, because every one of them folds per-block fp16 scales.)
llvm::Error compileRVVGeneratedSourceToObject(llvm::StringRef source,
                                              llvm::raw_ostream &os) {
  return compileRVVGeneratedSourceToObjectWithMarch(source, os,
                                                    "-march=rv64gcv");
}

// The Zvfh object packager (rv64gcv_zvfh): used by every monolithic block-dot /
// repack family. The repack dual-fp16 scale fold emits Zvfh fp16 vector loads
// (vle16_v_f16m1), and the block-dot families fold each block's scalar fp16 scale
// d; enabling Zvfh (which implies Zfhmin) lets clang hardware-ize the fp16->fp32
// conversion (fcvt.s.h) instead of emitting a __extendhfsf2 softfloat libcall.
llvm::Error compileRVVGeneratedSourceToObjectZvfh(llvm::StringRef source,
                                                  llvm::raw_ostream &os) {
  return compileRVVGeneratedSourceToObjectWithMarch(source, os,
                                                    "-march=rv64gcv_zvfh");
}

// P2-b chunk3: monolithic ggml block-dot (super-block + flat) target-artifact-export
// wiring, generalized from chunk2's q4_K-only mirror.
//
// A monolithic ggml block-dot selected body is ONE plugin-owned typed op
// (weft_rvv.q4_k_q8_k_block_dot / q4_0_q8_0_block_dot / iq4_nl_q8_0_block_dot /
// ...) that lowers DIRECTLY through the RVV->EmitC DialectConversion; there is no
// decomposed route slice for the slice-based describeRVVSelectedBodyEmitCRoute to
// walk (it fails fail-closed on the monolithic op). Chunk 1 wired the
// emission-plans stage by early-returning an HONEST monolithic-body emission plan
// in the plugin (RVVExtensionPlugin.cpp buildMonolithicBlockDotEmissionPlan),
// keyed off the shared block-dot family predicate. This is the mirror at the
// target-artifact-export chokepoint: HONEST peer OBJECT exporters (one per route
// family) registered under the monolithic route ids, with a purpose-built
// candidate validator that checks the emission plan's honest monolithic fields
// DIRECTLY (no describe-slice). They are registered as bare object exporters (not
// through the construction-template header/object bundle adapter) on purpose: a
// second RVV header/bundle COMPOSITE would select every rvv-origin candidate by
// origin and then reject the decomposed ones, breaking their header/bundle
// export. The bare object exporters are keyed by the monolithic route ids, so
// every decomposed route is byte-exact untouched (it never carries these route
// ids and never reaches this validator/exporter). Each monolithic plan produces
// only the object candidate (no header artifact), so the object exporter is all
// it needs.
//
// The route ids, runtime-ABI name, archetype, and the two op-derived metadata
// keys are parameterized by route family (super-block vs flat) from the shared
// RVVMonolithicBlockDotFamily.h mechanism; the per-op ordered ABI expectation is
// looked up there by the candidate's kind metadata. Keying q4_K off the
// super-block family reproduces chunk2's exact constants/checks (byte-exact).

// Determine the route family a candidate route id belongs to (error if the id is
// neither the super-block nor the flat monolithic route id).
llvm::Expected<plugin::rvv::MonolithicBlockDotRouteFamily>
monolithicBlockDotFamilyForRouteID(llvm::StringRef routeID) {
  using plugin::rvv::MonolithicBlockDotRouteFamily;
  if (routeID == plugin::rvv::getMonolithicBlockDotFamilyConstants(
                     MonolithicBlockDotRouteFamily::SuperBlock)
                     .routeID)
    return MonolithicBlockDotRouteFamily::SuperBlock;
  if (routeID == plugin::rvv::getMonolithicBlockDotFamilyConstants(
                     MonolithicBlockDotRouteFamily::Flat)
                     .routeID)
    return MonolithicBlockDotRouteFamily::Flat;
  if (routeID == plugin::rvv::getMonolithicBlockDotFamilyConstants(
                     MonolithicBlockDotRouteFamily::RepackGemv)
                     .routeID)
    return MonolithicBlockDotRouteFamily::RepackGemv;
  if (routeID == plugin::rvv::getMonolithicBlockDotFamilyConstants(
                     MonolithicBlockDotRouteFamily::RepackGemm)
                     .routeID)
    return MonolithicBlockDotRouteFamily::RepackGemm;
  return makeRVVTargetRouteError(
      llvm::Twine("candidate route id '") + routeID +
      "' is not a monolithic ggml block-dot route id");
}

llvm::Error validateRVVMonolithicBlockDotTargetArtifactCandidate(
    const TargetArtifactCandidate &candidate) {
  namespace mbd = plugin::rvv::monolithic_block_dot;
  if (llvm::Error error = plugin::rvv::verifyRVVConstructionProtocolReady())
    return error;

  llvm::Expected<plugin::rvv::MonolithicBlockDotRouteFamily> family =
      monolithicBlockDotFamilyForRouteID(candidate.routeID);
  if (!family)
    return family.takeError();
  const plugin::rvv::MonolithicBlockDotFamilyConstants &fc =
      plugin::rvv::getMonolithicBlockDotFamilyConstants(*family);

  if (candidate.role == "dispatch fallback")
    return makeRVVTargetRouteError(
        "monolithic ggml block-dot candidate must not be fallback-only");
  if (llvm::Error error = requireCandidateField(
          "origin", candidate.origin,
          plugin::rvv::getRVVConstructionManifest().family.pluginName))
    return error;
  if (llvm::Error error = requireCandidateField("route id", candidate.routeID,
                                                fc.routeID))
    return error;
  if (llvm::Error error = requireCandidateField(
          "artifact kind", candidate.artifactKind,
          plugin::rvv::getRVVSelectedBodyTargetArtifactKind()))
    return error;
  if (llvm::Error error =
          requireCandidateField("emission kind", candidate.emissionKind,
                                plugin::rvv::getRVVSelectedBodyEmissionKind()))
    return error;
  if (llvm::Error error = requireCandidateField(
          "lowering boundary", candidate.loweringBoundary,
          plugin::rvv::getRVVSelectedBodyLoweringBoundaryOpName()))
    return error;
  if (llvm::Error error = requireCandidateField(
          "runtime ABI kind", candidate.runtimeABIKind,
          plugin::rvv::getRVVSelectedBodyRuntimeABIKind()))
    return error;
  if (llvm::Error error = requireCandidateField(
          "runtime glue role", candidate.runtimeGlueRole,
          plugin::rvv::getRVVSelectedBodyRuntimeGlueRole()))
    return error;
  if (llvm::Error error = requireCandidateField("runtime ABI",
                                                candidate.runtimeABI,
                                                fc.runtimeABIName))
    return error;
  if (llvm::Error error = requireCandidateField(
          "runtime ABI name", candidate.runtimeABIName, fc.runtimeABIName))
    return error;

  // Per-op ABI: look up the expected ordered ggml vec_dot ABI by the candidate's
  // op-derived kind metadata (q4_0 carries the 8-role strided ABI; q4_K/iq4_nl
  // carry the 4-role n/s/vx/vy). The op kind must belong to this route family.
  llvm::StringRef kindValue =
      lookupCandidateMetadataValue(candidate, fc.kindMetadataKey);
  const plugin::rvv::MonolithicBlockDotOpEntry *opEntry =
      plugin::rvv::findMonolithicBlockDotOpEntryByKind(kindValue);
  if (!opEntry || opEntry->routeFamily != *family)
    return makeRVVTargetRouteError(
        llvm::Twine("monolithic ggml block-dot candidate carries an "
                    "unrecognized op kind '") +
        kindValue + "' for its route family");
  llvm::ArrayRef<plugin::rvv::MonolithicBlockDotABIRole> abiExpectation =
      opEntry->abiRoles();
  if (candidate.runtimeABIParameters.size() != abiExpectation.size())
    return makeRVVTargetRouteError(
        llvm::Twine("monolithic ggml block-dot candidate for op kind '") +
        kindValue + "' must carry its " +
        llvm::Twine(static_cast<unsigned>(abiExpectation.size())) +
        " ordered ggml vec_dot ABI parameters");
  for (auto [expected, actual] :
       llvm::zip(abiExpectation, candidate.runtimeABIParameters)) {
    if (actual.role != expected.role || actual.cName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("monolithic ggml block-dot candidate ABI parameter must "
                      "be c parameter '") +
          expected.cName + "' with role '" +
          support::stringifyRuntimeABIParameterRole(expected.role) +
          "' but was c parameter '" + actual.cName + "' with role '" +
          support::stringifyRuntimeABIParameterRole(actual.role) + "'");
  }

  // The honest monolithic metadata mirror: the fixed-value keys are pinned, the
  // op-derived keys (kind / scale_model) are required present.
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, mbd::kRouteMetadataKey, fc.routeID,
          "monolithic block-dot route id"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, mbd::kSourceOpInterfaceKey, mbd::kSourceOpInterfaceName,
          "EmitC-lowerable op interface"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, mbd::kArchetypeKey, fc.archetype,
          "monolithic typed-body archetype"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, mbd::kTargetArtifactKindKey,
          plugin::rvv::getRVVSelectedBodyTargetArtifactKind(),
          "target artifact kind"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, mbd::kConstructionProtocolKey, mbd::kConstructionProtocol,
          "construction protocol"))
    return error;
  if (lookupCandidateMetadataValue(candidate, fc.kindMetadataKey).empty())
    return makeRVVTargetRouteError(
        llvm::Twine("candidate metadata must carry ") + fc.kindMetadataKey +
        " provenance");
  if (lookupCandidateMetadataValue(candidate, fc.scaleModelMetadataKey).empty())
    return makeRVVTargetRouteError(
        llvm::Twine("candidate metadata must carry ") +
        fc.scaleModelMetadataKey + " provenance");

  // The monolithic route honestly carries NO decomposed-route weft_rvv.* config
  // metadata (the slice-based describe never ran); assert none leaked in, which
  // is what positively distinguishes it from a decomposed selected-body route.
  for (const support::ArtifactMetadataEntry &meta : candidate.artifactMetadata)
    if (llvm::StringRef(meta.key).starts_with("weft_rvv."))
      return makeRVVTargetRouteError(
          llvm::Twine("monolithic ggml block-dot candidate must not carry "
                      "decomposed-route config metadata '") +
          meta.key + "'");

  return rejectForbiddenRVVArtifactMetadata(candidate);
}

llvm::SmallVector<MaterializedEmitCHeaderArtifactMetadataEvidence, 8>
buildRVVMonolithicBlockDotHeaderMetadataEvidence(
    plugin::rvv::MonolithicBlockDotRouteFamily family) {
  namespace mbd = plugin::rvv::monolithic_block_dot;
  const plugin::rvv::MonolithicBlockDotFamilyConstants &fc =
      plugin::rvv::getMonolithicBlockDotFamilyConstants(family);
  llvm::SmallVector<MaterializedEmitCHeaderArtifactMetadataEvidence, 8> evidence;
  evidence.append({
      {"emitc_lowerable_route", mbd::kRouteMetadataKey, fc.routeID,
       /*allowDynamicValue=*/false},
      {"source_op_interface", mbd::kSourceOpInterfaceKey,
       mbd::kSourceOpInterfaceName, /*allowDynamicValue=*/false},
      {"extension_archetype", mbd::kArchetypeKey, fc.archetype,
       /*allowDynamicValue=*/false},
      {"ggml_block_dot_kind", fc.kindMetadataKey, /*expectedValue=*/"",
       /*allowDynamicValue=*/true},
      {"ggml_scale_model", fc.scaleModelMetadataKey, /*expectedValue=*/"",
       /*allowDynamicValue=*/true},
      {"target_artifact_kind", mbd::kTargetArtifactKindKey,
       plugin::rvv::getRVVSelectedBodyTargetArtifactKind(),
       /*allowDynamicValue=*/false},
      {"construction_protocol", mbd::kConstructionProtocolKey,
       mbd::kConstructionProtocol, /*allowDynamicValue=*/false},
  });
  return evidence;
}

SelectedEmitCArtifactRouteConfig getRVVSelectedBodyArtifactConfig() {
  const plugin::rvv::RVVConstructionManifest &manifest = getRVVManifest();

  SelectedEmitCArtifactRouteConfig config;
  config.routeID = plugin::rvv::getRVVSelectedBodyTargetArtifactRouteID();
  config.artifactKind = plugin::rvv::getRVVSelectedBodyTargetArtifactKind();
  config.originPlugin = manifest.family.pluginName;
  config.routeDescription = "RVV selected-body materialized EmitC target "
                            "artifact bridge for the bounded i32 arithmetic "
                            "selected-body slice";
  config.candidateValidationFn =
      validateRVVSelectedBodyTargetArtifactCandidate;
  return config;
}

ConstructionTemplateArtifactAdapterConfig
getRVVSelectedBodyArtifactAdapterConfig() {
  static const llvm::StringRef kHeaderIncludes[] = {"stddef.h", "stdint.h"};
  static const ConstructionTemplateSelectedBoundaryAttributeExpectation
      kBoundaryAttributeExpectations[] = {
          {plugin::rvv::getRVVConstructionProtocolMetadataName(),
           plugin::rvv::getRVVConstructionProtocolVersion(), {}},
      };
  static const llvm::SmallVector<MaterializedEmitCHeaderArtifactMetadataEvidence,
                                 32>
      kMetadataEvidence = buildRVVSelectedBodyHeaderMetadataEvidence();
  static const llvm::SmallVector<support::RuntimeABIParameter, 4>
      kRuntimeABIParameters =
          plugin::rvv::getRVVSelectedBodyConstructionRuntimeABIParameters();

  const plugin::rvv::RVVConstructionManifest &manifest = getRVVManifest();
  const plugin::rvv::RVVSelectedBodyTargetArtifactMapping &mapping =
      getRVVTargetMapping();

  ConstructionTemplateArtifactAdapterConfig config;
  config.selectedRoute = getRVVSelectedBodyArtifactConfig();
  config.selectedRoute.routeDescription =
      "RVV selected-body construction-template materialized EmitC artifact "
      "adapter for the bounded i32 arithmetic selected-body slice";
  config.headerRouteID = mapping.headerRouteID;
  config.headerArtifactKind = mapping.headerArtifactKind;
  config.ownerPlugin = manifest.family.pluginName;
  config.headerGuard = "WEFT_RVV_MATERIALIZED_EMITC_HEADER_H";
  config.evidencePrefix = "weft.rvv";
  config.includes = kHeaderIncludes;
  config.selectedVariant = "";
  config.emissionKind = plugin::rvv::getRVVSelectedBodyEmissionKind();
  config.loweringBoundary =
      plugin::rvv::getRVVSelectedBodyLoweringBoundaryOpName();
  config.runtimeABIKind =
      plugin::rvv::getRVVSelectedBodyRuntimeABIKind();
  config.runtimeGlueRole =
      plugin::rvv::getRVVSelectedBodyRuntimeGlueRole();
  config.allowDynamicRuntimeABIIdentity = true;
  config.runtimeABIParameters = kRuntimeABIParameters;
  config.metadataEvidence = kMetadataEvidence;
  config.componentGroup = mapping.bundleComponentGroup;
  config.externalABIName = "";
  config.handoffKind = mapping.objectHandoffKind;
  config.selectedObjectDescription = "RVV materialized EmitC candidate";
  config.selectedLoweringBoundary.required = true;
  config.selectedLoweringBoundary.boundaryDescription =
      "selected RVV construction-template artifact boundary";
  config.selectedLoweringBoundary.status =
      plugin::rvv::getRVVLoweringBoundaryStatus();
  config.selectedLoweringBoundary.sourceKernelAttrName =
      plugin::rvv::getRVVSourceKernelAttrName();
  config.selectedLoweringBoundary.selectedVariantAttrName =
      plugin::rvv::getRVVSelectedVariantAttrName();
  config.selectedLoweringBoundary.originAttrName =
      plugin::rvv::getRVVOriginAttrName();
  config.selectedLoweringBoundary.roleAttrName =
      plugin::rvv::getRVVSelectedPathRoleAttrName();
  config.selectedLoweringBoundary.statusAttrName =
      plugin::rvv::getRVVStatusAttrName();
  config.selectedLoweringBoundary.requiredCapabilitiesAttrName =
      plugin::rvv::getRVVRequiredCapabilitiesAttrName();
  config.selectedLoweringBoundary.extraStringAttributes =
      kBoundaryAttributeExpectations;
  config.selectedLoweringBoundary.searchSelectedVariantBody = true;
  config.selectedLoweringBoundary.countOnlyDirectVariantBodyBoundaries = true;
  config.objectPackagerFn = compileRVVGeneratedSourceToObject;
  return config;
}

llvm::Error exportRVVSelectedBodyTargetArtifact(mlir::ModuleOp module,
                                                   llvm::raw_ostream &os) {
  return exportConstructionTemplateObjectArtifact(
      module, os, getRVVSelectedBodyArtifactAdapterConfig());
}

llvm::Error exportRVVSelectedBodyHeaderArtifact(mlir::ModuleOp module,
                                                   llvm::raw_ostream &os) {
  return exportConstructionTemplateHeaderArtifact(
      module, os, getRVVSelectedBodyArtifactAdapterConfig());
}

// P2-b chunk3: the monolithic block-dot object-export adapter config, per route
// family. It clones the RVV selected-body adapter config and overrides ONLY the
// fields that make the export honestly monolithic: the (family) route id, the
// honest monolithic candidate validator, the (family) header route id, and the
// (family) metadata evidence. The object emit reuses the common RVV->EmitC
// DialectConversion (materializeSelectedEmitCArtifactModule ->
// tryConvertModuleWithRegisteredBackend), which chunk 1 already made lower the
// monolithic op, so the exported EmitC is byte-identical to the CORE
// --weft-rvv-lower-to-emitc emit. This config is used ONLY by the monolithic
// OBJECT export fns (below); it is never handed to the header/object bundle
// registration helper, so no monolithic composite is registered and decomposed
// routes are byte-exact untouched.
ConstructionTemplateArtifactAdapterConfig
getRVVMonolithicBlockDotArtifactAdapterConfig(
    plugin::rvv::MonolithicBlockDotRouteFamily family) {
  static const llvm::SmallVector<MaterializedEmitCHeaderArtifactMetadataEvidence,
                                 8>
      kSuperBlockEvidence = buildRVVMonolithicBlockDotHeaderMetadataEvidence(
          plugin::rvv::MonolithicBlockDotRouteFamily::SuperBlock);
  static const llvm::SmallVector<MaterializedEmitCHeaderArtifactMetadataEvidence,
                                 8>
      kFlatEvidence = buildRVVMonolithicBlockDotHeaderMetadataEvidence(
          plugin::rvv::MonolithicBlockDotRouteFamily::Flat);
  static const llvm::SmallVector<MaterializedEmitCHeaderArtifactMetadataEvidence,
                                 8>
      kRepackGemvEvidence = buildRVVMonolithicBlockDotHeaderMetadataEvidence(
          plugin::rvv::MonolithicBlockDotRouteFamily::RepackGemv);
  static const llvm::SmallVector<MaterializedEmitCHeaderArtifactMetadataEvidence,
                                 8>
      kRepackGemmEvidence = buildRVVMonolithicBlockDotHeaderMetadataEvidence(
          plugin::rvv::MonolithicBlockDotRouteFamily::RepackGemm);
  const plugin::rvv::MonolithicBlockDotFamilyConstants &fc =
      plugin::rvv::getMonolithicBlockDotFamilyConstants(family);

  llvm::StringRef routeDescription;
  llvm::StringRef objectDescription;
  const llvm::SmallVector<MaterializedEmitCHeaderArtifactMetadataEvidence, 8>
      *evidence = &kFlatEvidence;
  switch (family) {
  case plugin::rvv::MonolithicBlockDotRouteFamily::SuperBlock:
    routeDescription =
        "RVV monolithic ggml super-block block-dot materialized EmitC target "
        "artifact bridge (single plugin-owned typed body lowering directly "
        "through the common RVV->EmitC DialectConversion)";
    objectDescription = "RVV monolithic ggml super-block block-dot materialized "
                        "EmitC candidate";
    evidence = &kSuperBlockEvidence;
    break;
  case plugin::rvv::MonolithicBlockDotRouteFamily::Flat:
    routeDescription =
        "RVV monolithic ggml flat block-dot materialized EmitC target artifact "
        "bridge (single plugin-owned typed body lowering directly through the "
        "common RVV->EmitC DialectConversion)";
    objectDescription =
        "RVV monolithic ggml flat block-dot materialized EmitC candidate";
    evidence = &kFlatEvidence;
    break;
  case plugin::rvv::MonolithicBlockDotRouteFamily::RepackGemv:
    routeDescription =
        "RVV monolithic ggml repacked-GEVM materialized EmitC target artifact "
        "bridge (single plugin-owned typed body lowering directly through the "
        "common RVV->EmitC DialectConversion)";
    objectDescription =
        "RVV monolithic ggml repacked-GEVM materialized EmitC candidate";
    evidence = &kRepackGemvEvidence;
    break;
  case plugin::rvv::MonolithicBlockDotRouteFamily::RepackGemm:
    routeDescription =
        "RVV monolithic ggml repacked block-as-lane GEMM (prefill) materialized "
        "EmitC target artifact bridge (single plugin-owned typed body lowering "
        "directly through the common RVV->EmitC DialectConversion)";
    objectDescription = "RVV monolithic ggml repacked-GEMM (prefill) materialized "
                        "EmitC candidate";
    evidence = &kRepackGemmEvidence;
    break;
  }

  ConstructionTemplateArtifactAdapterConfig config =
      getRVVSelectedBodyArtifactAdapterConfig();
  config.selectedRoute.routeID = fc.routeID;
  config.selectedRoute.routeDescription = routeDescription;
  config.selectedRoute.candidateValidationFn =
      validateRVVMonolithicBlockDotTargetArtifactCandidate;
  config.headerRouteID = fc.headerRouteID;
  config.metadataEvidence = *evidence;
  config.selectedObjectDescription = objectDescription;
  // Every monolithic family carries per-block fp16 scales: the repacked GEVM/GEMM
  // emit Zvfh fp16 vector loads (vle16_v_f16m1) for their dual-fp16 scale fold,
  // and the block-dot families (super-block / flat) fold each block's scalar fp16
  // scale d (a (float)*(const _Float16 *) read). Packaging under rv64gcv_zvfh
  // (which implies Zfhmin) hardware-izes that fp16->fp32 conversion (fcvt.s.h)
  // instead of degrading each block-scale read to a __extendhfsf2 softfloat
  // libcall -- the libcall both blocks format-micro linking and one-sidedly
  // pollutes the perf ratio vs the factory (hardware fcvt.s.h) baseline. So all
  // four monolithic block-dot / repack families package under rv64gcv_zvfh; only
  // the decomposed selected-body route (getRVVSelectedBodyArtifactAdapterConfig,
  // above) keeps the baseline rv64gcv packager.
  config.objectPackagerFn = compileRVVGeneratedSourceToObjectZvfh;
  return config;
}

llvm::Error
exportRVVMonolithicSuperBlockBlockDotTargetArtifact(mlir::ModuleOp module,
                                                    llvm::raw_ostream &os) {
  return exportConstructionTemplateObjectArtifact(
      module, os,
      getRVVMonolithicBlockDotArtifactAdapterConfig(
          plugin::rvv::MonolithicBlockDotRouteFamily::SuperBlock));
}

llvm::Error
exportRVVMonolithicFlatBlockDotTargetArtifact(mlir::ModuleOp module,
                                              llvm::raw_ostream &os) {
  return exportConstructionTemplateObjectArtifact(
      module, os,
      getRVVMonolithicBlockDotArtifactAdapterConfig(
          plugin::rvv::MonolithicBlockDotRouteFamily::Flat));
}

llvm::Error
exportRVVMonolithicRepackGemvTargetArtifact(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  return exportConstructionTemplateObjectArtifact(
      module, os,
      getRVVMonolithicBlockDotArtifactAdapterConfig(
          plugin::rvv::MonolithicBlockDotRouteFamily::RepackGemv));
}

llvm::Error
exportRVVMonolithicRepackGemmTargetArtifact(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  return exportConstructionTemplateObjectArtifact(
      module, os,
      getRVVMonolithicBlockDotArtifactAdapterConfig(
          plugin::rvv::MonolithicBlockDotRouteFamily::RepackGemm));
}

// Register one bare peer OBJECT exporter for a monolithic block-dot route family.
llvm::Error registerRVVMonolithicBlockDotObjectExporter(
    TargetArtifactExporterRegistry &registry,
    plugin::rvv::MonolithicBlockDotRouteFamily family,
    TargetArtifactExportFn exportFn) {
  llvm::StringRef routeID =
      plugin::rvv::getMonolithicBlockDotFamilyConstants(family).routeID;
  if (registry.lookup(routeID))
    return llvm::Error::success();
  return registry.registerExporter(TargetArtifactExporter(
      routeID, plugin::rvv::getRVVSelectedBodyTargetArtifactKind(),
      getRVVManifest().family.pluginName,
      plugin::rvv::getRVVSelectedBodyEmissionKind(), exportFn,
      /*requiredRuntimeABIParameters=*/{},
      getRVVTargetMapping().objectHandoffKind,
      validateRVVMonolithicBlockDotTargetArtifactCandidate,
      getRVVTargetMapping().bundleComponentGroup,
      /*externalABIName=*/""));
}

llvm::Error registerRVVSelectedBodyTargetArtifactExporter(
    TargetArtifactExporterRegistry &registry) {
  if (llvm::Error error = plugin::rvv::verifyRVVConstructionProtocolReady())
    return error;

  if (llvm::Error error = registerConstructionTemplateArtifactAdapterExporters(
          registry, getRVVSelectedBodyArtifactAdapterConfig(),
          exportRVVSelectedBodyTargetArtifact,
          exportRVVSelectedBodyHeaderArtifact))
    return error;

  // P2-b chunk3: the honest monolithic ggml block-dot peer OBJECT routes, one per
  // route family (super-block for q4_K; flat for q4_0/iq4_nl). Registered as bare
  // object exporters (NOT header/object bundle composites) under their own route
  // ids so coherence + target-artifact export recognize the monolithic-body plan
  // without a composite that would shadow the decomposed routes. Idempotent-
  // guarded like the standard adapter.
  if (llvm::Error error = registerRVVMonolithicBlockDotObjectExporter(
          registry, plugin::rvv::MonolithicBlockDotRouteFamily::SuperBlock,
          exportRVVMonolithicSuperBlockBlockDotTargetArtifact))
    return error;
  if (llvm::Error error = registerRVVMonolithicBlockDotObjectExporter(
          registry, plugin::rvv::MonolithicBlockDotRouteFamily::Flat,
          exportRVVMonolithicFlatBlockDotTargetArtifact))
    return error;

  // The option-2 quant_contraction BRIDGE repacked-GEVM peer OBJECT route: the
  // q4_0 16x1-repacked GEVM the front door constructs as the typed
  // weft_rvv.typed_repack_gemv_loop_body region flows through the SAME monolithic
  // emission-plan + object-export mechanism as the flat/super-block block-dots,
  // on its own RepackGemv route id. Registered as a bare object exporter like its
  // block-dot siblings; idempotent-guarded.
  if (llvm::Error error = registerRVVMonolithicBlockDotObjectExporter(
          registry, plugin::rvv::MonolithicBlockDotRouteFamily::RepackGemv,
          exportRVVMonolithicRepackGemvTargetArtifact))
    return error;

  // The option-2 quant_contraction BRIDGE repacked-GEMM (prefill) peer OBJECT
  // route: the q4_0 16x1-repacked block-as-lane GEMM the front door constructs as
  // the typed weft_rvv.typed_repack_gemm_loop_body region flows through the SAME
  // monolithic emission-plan + object-export mechanism as the GEVM / block-dots, on
  // its own RepackGemm route id. Registered as a bare object exporter like its
  // siblings; idempotent-guarded.
  if (llvm::Error error = registerRVVMonolithicBlockDotObjectExporter(
          registry, plugin::rvv::MonolithicBlockDotRouteFamily::RepackGemm,
          exportRVVMonolithicRepackGemmTargetArtifact))
    return error;

  return llvm::Error::success();
}

} // namespace

llvm::Error registerRVVTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry) {
  llvm::StringRef pluginName = getRVVManifest().family.pluginName;
  if (const PluginTargetArtifactExporterBundle *existing =
          registry.lookup(pluginName)) {
    for (const PluginTargetArtifactExporterBundle &bundle :
         registry.lookupAll(pluginName))
      if (bundle.getRegistrationFn() ==
          registerRVVSelectedBodyTargetArtifactExporter)
        return llvm::Error::success();
    (void)existing;
  }
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      pluginName, registerRVVSelectedBodyTargetArtifactExporter));
}

llvm::Error
configureRVVTargetSupportExtensionBundle(plugin::ExtensionBundle &bundle) {
  bundle.addLoweringBoundaryOp(
      plugin::rvv::getRVVSelectedBodyLoweringBoundaryOpName());
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      registerRVVTargetSupportPluginTargetExporterBundles);
  return llvm::Error::success();
}

llvm::Error registerRVVTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry) {
  llvm::StringRef routeID = getRVVTargetMapping().emitCToCppTranslateRouteID;
  if (registry.lookup(routeID))
    return llvm::Error::success();

  return registry.registerRoute(TargetTranslateRoute(
      routeID,
      "export a materialized RVV EmitC module through the MLIR EmitC "
      "C/C++ emitter",
      exportMaterializedRVVEmitCToCpp));
}

} // namespace weft::target::rvv
