#include "TianChenRV/Target/RVV/RVVScalarDispatch.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVLowerToEmitC.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Support/FiniteBinaryFrontendLowering.h"
#include "TianChenRV/Support/RuntimeABICallablePlan.h"
#include "TianChenRV/Support/RuntimeABIContract.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"
#include "TianChenRV/Target/BinarySelfCheckExpectation.h"
#include "TianChenRV/Target/RVV/RVVBinaryDescriptor.h"
#include "TianChenRV/Target/RVV/RVVMicrokernel.h"
#include "TianChenRV/Target/RVV/RVVSelectedConfigContract.h"
#include "TianChenRV/Target/RVV/RVVVectorShape.h"
#include "TianChenRV/Target/RVVScalarBinaryFamily.h"
#include "TianChenRV/Target/Scalar/ScalarMicrokernel.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <system_error>

namespace tianchenrv::target::rvv_scalar {
namespace {

using tianchenrv::target::TargetArtifactCandidate;
using tianchenrv::target::TargetArtifactCompositeBundleMetadata;
using tianchenrv::target::TargetArtifactCompositeExporter;
using tianchenrv::target::TargetArtifactExporter;
using tianchenrv::target::TargetArtifactExporterRegistry;
using tianchenrv::conversion::emitc::TCRVEmitCCallOpaqueStep;
using tianchenrv::conversion::emitc::TCRVEmitCLowerableInterface;
using tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute;
using tianchenrv::conversion::emitc::TCRVLowerToEmitCSourceOptions;
using tianchenrv::conversion::emitc::lowerTCRVEmitCLowerableToEmitCSource;
using tianchenrv::support::CapabilityDescriptor;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::target::BinarySelfCheckArithmeticKind;
using tianchenrv::target::BinarySelfCheckExpectation;
using tianchenrv::tcrv::exec::DispatchOp;
using tianchenrv::tcrv::exec::DispatchCaseOp;
using tianchenrv::tcrv::exec::FallbackOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::MemWindowOp;
using tianchenrv::tcrv::exec::RuntimeParamOp;
using tianchenrv::tcrv::exec::VariantOp;

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kScalarPluginName("scalar-plugin");
constexpr llvm::StringLiteral kDispatchCaseRole("dispatch case");
constexpr llvm::StringLiteral kDispatchFallbackRole("dispatch fallback");
constexpr llvm::StringLiteral kRuntimeCallableCSourceArtifactKind(
    "runtime-callable-c-source");
constexpr llvm::StringLiteral kRuntimeCallableCHeaderArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kRiscvELFRelocatableObjectArtifactKind(
    "riscv-elf-relocatable-object");
constexpr llvm::StringLiteral kSelfCheckCSourceArtifactKind(
    "self-check-c-source");
constexpr llvm::StringLiteral kSelfCheckObjectArtifactKind(
    "self-check-riscv-elf-relocatable-object");
constexpr llvm::StringLiteral kDispatchRuntimeABIParametersAttrName(
    "tcrv_rvv_scalar.dispatch_runtime_abi_parameters");
constexpr llvm::StringLiteral kRuntimeGuardAttrName("runtime_guard");

constexpr llvm::StringLiteral kDispatchTargetOwner(
    "rvv-scalar-dispatch-target");
constexpr llvm::StringLiteral kDispatchSelectedConfigMetadataRole(
    "rvv-dispatch-selected-config-contract");
constexpr llvm::StringLiteral kDispatchSelectedConfigMetadataNote(
    "RVV dispatch component selected-config contract consumed from the "
    "validated direct RVV runtime AVL source authority; descriptor element "
    "count is bounded component capacity metadata only");
constexpr llvm::StringLiteral kRVVRequiredMarchAttrName(
    "tcrv_rvv.required_march");
constexpr llvm::StringLiteral kRVVCapabilityID("rvv");
constexpr llvm::StringLiteral kRVVProbeCompileRunCapabilityID(
    "rvv.probe.compile_run");
constexpr llvm::StringLiteral kRVVToolchainMarchCapabilityID(
    "rvv.toolchain.march");
constexpr llvm::StringLiteral kRVVToolchainMABICapabilityID(
    "rvv.toolchain.mabi");
constexpr llvm::StringLiteral kArchitecturePropertyName("architecture");
constexpr llvm::StringLiteral kSelectedMarchPropertyName("selected_march");
constexpr llvm::StringLiteral kSelectedMABIPropertyName("selected_mabi");
constexpr llvm::StringLiteral kValuePropertyName("value");
constexpr llvm::StringLiteral kSEWBitsPropertyName("sew_bits");
constexpr llvm::StringLiteral kLMULPropertyName("lmul");
constexpr llvm::StringLiteral kTailPolicyPropertyName("tail_policy");
constexpr llvm::StringLiteral kMaskPolicyPropertyName("mask_policy");

struct DispatchObjectCompileConfig {
  std::string targetTriple;
  std::string selectedMarch;
  std::optional<std::string> selectedMABI;
};

struct DispatchABIPlan {
  llvm::SmallVector<support::RuntimeABIParameter, 5> parameters;
  llvm::SmallVector<tcrv::exec::MemWindowOp, 3> bufferWindows;
  llvm::SmallVector<RuntimeParamOp, 2> runtimeParams;
  RuntimeParamOp runtimeGuardParam;
};

struct DispatchRuntimeABIParameterBindings {
  const support::RuntimeABIParameter *lhs = nullptr;
  const support::RuntimeABIParameter *rhs = nullptr;
  const support::RuntimeABIParameter *out = nullptr;
  const support::RuntimeABIParameter *runtimeElementCount = nullptr;
  const support::RuntimeABIParameter *dispatchAvailabilityGuard = nullptr;
};

struct CallableABIPlan {
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters;
  llvm::SmallVector<MemWindowOp, 3> bufferWindows;
  RuntimeParamOp runtimeElementCountParam;
};

struct DispatchIRLink {
  DispatchOp dispatch;
  DispatchCaseOp selectedRVVCase;
  FallbackOp fallback;
  VariantOp fallbackVariant;
  std::string fallbackTarget;
  std::string fallbackOrigin;
  std::string fallbackRole;
};

using DispatchI32FamilySpec =
    tianchenrv::target::rvv_scalar::DispatchBinaryFamilyDescriptor;
using DispatchRVVVectorShapeConfig =
    tianchenrv::target::rvv::RVVVectorShapeConfig;
using RVVI32BinaryIntrinsicDescriptor =
    tianchenrv::target::rvv::RVVBinaryIntrinsicDescriptor;

struct DispatchPair {
  const DispatchI32FamilySpec *family = nullptr;
  const DispatchRVVVectorShapeConfig *selectedShape = nullptr;
  tianchenrv::target::rvv::RVVBinarySelectedConfigContract selectedConfig;
  struct CompositeIdentity {
    std::string diagnosticName;
    std::string functionStem;
    std::string headerGuardStem;
    std::string runtimeABIKind;
    std::string runtimeABIName;
    std::string componentGroup;
    std::string externalABIName;
    std::string selfCheckSuccessMarker;
  } composite;
  TargetArtifactCandidate rvv;
  TargetArtifactCandidate scalar;
  DispatchIRLink irLink;
  DispatchABIPlan abiPlan;
};

llvm::Expected<const DispatchRVVVectorShapeConfig *>
resolveDispatchPairSelectedVectorShape(const DispatchPair &pair);

llvm::Expected<tianchenrv::target::rvv::RVVBinarySelectedConfigContract>
buildDispatchPairSelectedConfigContract(
    const DispatchPair &pair, const DispatchRVVVectorShapeConfig &shape);

llvm::Error validateDispatchSelectedConfigContractMetadata(
    const TargetArtifactCandidate &candidate,
    const tianchenrv::target::rvv::RVVBinarySelectedConfigContract &contract);

llvm::Expected<std::string>
requireSelectedComponentFamilyID(const TargetArtifactCandidate &candidate,
                                 llvm::StringRef metadataName,
                                 llvm::StringRef componentLabel);

llvm::Expected<DispatchPair::CompositeIdentity>
deriveDispatchCompositeIdentityFromSelectedComponents(
    const DispatchPair &pair);

llvm::Error validateEmbeddedRVVSourceSelectedShape(
    const DispatchPair &pair, llvm::StringRef rvvSource);

llvm::Error validateDispatchPairComponentAuthorities(const DispatchPair &pair);

llvm::Error makeModuleDispatchError(llvm::Twine message);

llvm::Expected<BinarySelfCheckExpectation>
buildDispatchSelfCheckExpectation(const DispatchPair &pair) {
  if (!pair.family || !pair.family->rvvFamily)
    return makeModuleDispatchError(
        "RVV+scalar dispatch harness requires a validated typed binary "
        "family before self-check expectation emission");

  BinarySelfCheckArithmeticKind arithmetic =
      BinarySelfCheckArithmeticKind::Add;
  switch (pair.family->rvvFamily->arithmetic) {
  case tianchenrv::target::rvv::RVVBinaryArithmeticKind::Add:
    arithmetic = BinarySelfCheckArithmeticKind::Add;
    break;
  case tianchenrv::target::rvv::RVVBinaryArithmeticKind::Sub:
    arithmetic = BinarySelfCheckArithmeticKind::Sub;
    break;
  case tianchenrv::target::rvv::RVVBinaryArithmeticKind::Mul:
    arithmetic = BinarySelfCheckArithmeticKind::Mul;
    break;
  }

  return tianchenrv::target::buildBinarySelfCheckExpectationFromRuntimeABI(
      pair.abiPlan.parameters, arithmetic,
      "validated RVV dispatch-case component + validated scalar fallback "
      "component + IR-backed dispatch ABI",
      "RVV+scalar dispatch harness");
}

struct TemporaryFile {
  llvm::SmallString<128> path;

  ~TemporaryFile() {
    if (!path.empty())
      llvm::sys::fs::remove(path);
  }

  llvm::StringRef get() const { return llvm::StringRef(path); }
};

llvm::Error makeDispatchError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV RVV+scalar binary dispatch C export failed";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

llvm::Error makeModuleDispatchError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV+scalar binary dispatch C export "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error makeDispatchObjectError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV RVV+scalar binary dispatch object export "
            "failed";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

llvm::Error makeModuleDispatchObjectError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV+scalar binary dispatch object export "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error makeModuleDispatchHeaderError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV+scalar binary dispatch header export "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

const DispatchI32FamilySpec &getI32VAddDispatchFamilySpec() {
  return tianchenrv::target::rvv_scalar::getI32VAddFamilyRegistrationRecord()
      .dispatch;
}

const DispatchI32FamilySpec &getI32VSubDispatchFamilySpec() {
  return tianchenrv::target::rvv_scalar::getI32VSubFamilyRegistrationRecord()
      .dispatch;
}

const DispatchI32FamilySpec &getI32VMulDispatchFamilySpec() {
  return tianchenrv::target::rvv_scalar::getI32VMulFamilyRegistrationRecord()
      .dispatch;
}

const DispatchI32FamilySpec &getI64VAddDispatchFamilySpec() {
  return tianchenrv::target::rvv_scalar::getI64VAddFamilyRegistrationRecord()
      .dispatch;
}

const DispatchI32FamilySpec &getI64VSubDispatchFamilySpec() {
  return tianchenrv::target::rvv_scalar::getI64VSubFamilyRegistrationRecord()
      .dispatch;
}

const DispatchI32FamilySpec &getI64VMulDispatchFamilySpec() {
  return tianchenrv::target::rvv_scalar::getI64VMulFamilyRegistrationRecord()
      .dispatch;
}

using DispatchRouteKind = RVVScalarDispatchRouteKind;
using DispatchRouteManifestEntry = RVVScalarDispatchRouteManifestEntry;

bool containsForbiddenText(llvm::StringRef value) {
  std::string lower = value.lower();
  llvm::StringRef normalized(lower);
  return normalized.contains("password") || normalized.contains("passwd") ||
         normalized.contains("token") || normalized.contains("secret") ||
         normalized.contains("private key") ||
         normalized.contains("authorization:") ||
         normalized.contains("api_key") || normalized.contains("access_key") ||
         normalized.contains("http://") || normalized.contains("https://") ||
         normalized.contains("://");
}

llvm::Error validateBoundedCompileText(KernelOp kernel,
                                       llvm::StringRef fieldName,
                                       llvm::StringRef value) {
  constexpr std::size_t kMaxTextLength = 128;
  if (value.empty() || value.size() > kMaxTextLength)
    return makeDispatchObjectError(
        kernel, llvm::Twine(fieldName) +
                    " must be bounded non-empty compile metadata");

  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (!std::isalnum(byte) && character != '_' && character != '.')
      return makeDispatchObjectError(
          kernel, llvm::Twine(fieldName) +
                      " must contain only bounded compile-flag characters");
  }

  if (containsForbiddenText(value))
    return makeDispatchObjectError(
        kernel, llvm::Twine(fieldName) +
                    " must not contain secret-like or raw credential text");
  return llvm::Error::success();
}

bool hasRVVVectorHint(llvm::StringRef hints) {
  std::string lower = hints.lower();
  llvm::StringRef normalized(lower);
  if (normalized.contains("zve") || normalized.contains("zvl") ||
      normalized.contains("zvfh") || normalized.contains("gcv"))
    return true;

  std::size_t position = lower.find("rv64");
  while (position != std::string::npos) {
    std::size_t end = position;
    while (end < lower.size()) {
      unsigned char byte = static_cast<unsigned char>(lower[end]);
      if (!std::isalnum(byte) && lower[end] != '_' && lower[end] != '-')
        break;
      ++end;
    }
    if (llvm::StringRef(lower).slice(position, end).drop_front(4).contains("v"))
      return true;
    position = lower.find("rv64", position + 4);
  }
  return false;
}

llvm::Error requireCallableCandidateField(
    const TargetArtifactCandidate &candidate, const DispatchI32FamilySpec &family,
    llvm::StringRef componentLabel, llvm::StringRef fieldName,
    llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeDispatchError(
      candidate.kernel,
      llvm::Twine("selected ") + componentLabel + " callable route '" +
          candidate.routeID + "' for " + family.diagnosticName +
          " has stale " + fieldName + " '" + actual + "'; expected '" +
          expected + "'");
}

llvm::Error validateRVVCallableCandidateShapeForFamily(
    const TargetArtifactCandidate &candidate,
    const DispatchI32FamilySpec &family) {
  if (llvm::Error error = requireCallableCandidateField(
          candidate, family, "RVV dispatch case", "origin", candidate.origin,
          kRVVPluginName))
    return error;
  if (llvm::Error error = requireCallableCandidateField(
          candidate, family, "RVV dispatch case", "role", candidate.role,
          kDispatchCaseRole))
    return error;
  if (llvm::Error error = requireCallableCandidateField(
          candidate, family, "RVV dispatch case", "route id",
          candidate.routeID, family.rvvRouteID))
    return error;
  if (llvm::Error error = requireCallableCandidateField(
          candidate, family, "RVV dispatch case", "emission_kind",
          candidate.emissionKind, family.rvvEmissionKind))
    return error;
  if (llvm::Error error = requireCallableCandidateField(
          candidate, family, "RVV dispatch case", "artifact_kind",
          candidate.artifactKind, kRuntimeCallableCSourceArtifactKind))
    return error;
  if (llvm::Error error = requireCallableCandidateField(
          candidate, family, "RVV dispatch case", "runtime_abi",
          candidate.runtimeABI, family.rvvRuntimeABI))
    return error;
  if (llvm::Error error = requireCallableCandidateField(
          candidate, family, "RVV dispatch case", "runtime_abi_kind",
          candidate.runtimeABIKind, family.rvvRuntimeABIKind))
    return error;
  if (llvm::Error error = requireCallableCandidateField(
          candidate, family, "RVV dispatch case", "runtime_abi_name",
          candidate.runtimeABIName, family.rvvRuntimeABIName))
    return error;
  if (llvm::Error error = requireCallableCandidateField(
          candidate, family, "RVV dispatch case", "runtime_glue_role",
          candidate.runtimeGlueRole, family.rvvRuntimeGlueRole))
    return error;
  return llvm::Error::success();
}

llvm::Error validateScalarCallableCandidateShapeForFamily(
    const TargetArtifactCandidate &candidate,
    const DispatchI32FamilySpec &family) {
  if (llvm::Error error = requireCallableCandidateField(
          candidate, family, "scalar dispatch fallback", "origin",
          candidate.origin, kScalarPluginName))
    return error;
  if (llvm::Error error = requireCallableCandidateField(
          candidate, family, "scalar dispatch fallback", "role",
          candidate.role, kDispatchFallbackRole))
    return error;
  if (llvm::Error error = requireCallableCandidateField(
          candidate, family, "scalar dispatch fallback", "route id",
          candidate.routeID, family.scalarRouteID))
    return error;
  if (llvm::Error error = requireCallableCandidateField(
          candidate, family, "scalar dispatch fallback", "emission_kind",
          candidate.emissionKind, family.scalarEmissionKind))
    return error;
  if (llvm::Error error = requireCallableCandidateField(
          candidate, family, "scalar dispatch fallback", "artifact_kind",
          candidate.artifactKind, kRuntimeCallableCSourceArtifactKind))
    return error;
  if (llvm::Error error = requireCallableCandidateField(
          candidate, family, "scalar dispatch fallback", "runtime_abi",
          candidate.runtimeABI, family.scalarRuntimeABI))
    return error;
  if (llvm::Error error = requireCallableCandidateField(
          candidate, family, "scalar dispatch fallback", "runtime_abi_kind",
          candidate.runtimeABIKind, family.scalarRuntimeABIKind))
    return error;
  if (llvm::Error error = requireCallableCandidateField(
          candidate, family, "scalar dispatch fallback", "runtime_abi_name",
          candidate.runtimeABIName, family.scalarRuntimeABIName))
    return error;
  if (llvm::Error error = requireCallableCandidateField(
          candidate, family, "scalar dispatch fallback", "runtime_glue_role",
          candidate.runtimeGlueRole, family.scalarRuntimeGlueRole))
    return error;
  return llvm::Error::success();
}

const DispatchI32FamilySpec *
lookupDispatchFamilyRegistrationByRVVRouteID(llvm::StringRef routeID) {
  if (const rvv::RVVMicrokernelDirectRouteManifestEntry *route =
          rvv::lookupRVVMicrokernelDirectRoute(routeID))
    if (route->routeKind ==
            rvv::RVVMicrokernelDirectRouteKind::Source &&
        route->family)
      for (const auto *descriptor :
           tianchenrv::target::rvv_scalar::
               getRVVScalarBinaryRegistrationRecords()) {
        const DispatchI32FamilySpec &family = descriptor->dispatch;
        if (family.rvvFamily == route->family)
          return &family;
      }
  return nullptr;
}

const DispatchI32FamilySpec *
lookupDispatchFamilyRegistrationByScalarRouteID(llvm::StringRef routeID) {
  for (const auto *descriptor :
       tianchenrv::target::rvv_scalar::getRVVScalarBinaryRegistrationRecords()) {
    const DispatchI32FamilySpec &family = descriptor->dispatch;
    if (family.scalarRouteID == routeID)
      return &family;
  }
  return nullptr;
}

const DispatchI32FamilySpec *
lookupDispatchFamilyRegistrationBySelectedFamilyID(
    llvm::StringRef selectedFamilyID) {
  if (const auto *descriptor =
          tianchenrv::target::rvv_scalar::
              lookupRVVScalarBinaryRegistrationByID(selectedFamilyID))
    return &descriptor->dispatch;
  return nullptr;
}

llvm::Expected<const DispatchI32FamilySpec *>
getRVVCallableCandidateFamilyFromSelectedPlan(
    const TargetArtifactCandidate &candidate) {
  // The selected-plan family metadata is the dispatch component authority.
  // The registration record below is only the finite route manifest used for
  // post-authority route/ABI consistency checks.
  llvm::Expected<std::string> selectedFamilyID =
      requireSelectedComponentFamilyID(
          candidate,
          tianchenrv::target::rvv::getRVVSelectedBinaryFamilyMetadataName(),
          "RVV dispatch case");
  if (!selectedFamilyID)
    return selectedFamilyID.takeError();

  const DispatchI32FamilySpec *family =
      lookupDispatchFamilyRegistrationBySelectedFamilyID(*selectedFamilyID);
  if (!family)
    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("selected RVV dispatch case candidate @") +
            candidate.selectedVariant +
            " names unsupported finite selected component family '" +
            *selectedFamilyID + "'");
  return family;
}

llvm::Expected<const DispatchI32FamilySpec *>
getScalarCallableCandidateFamilyFromSelectedPlan(
    const TargetArtifactCandidate &candidate) {
  // The scalar selected-plan metadata chooses the finite component family.
  // Descriptor-shaped records may only validate route registration mirrors.
  llvm::Expected<std::string> selectedFamilyID =
      requireSelectedComponentFamilyID(
          candidate,
          tianchenrv::target::rvv_scalar::
              getScalarSelectedBinaryFamilyMetadataName(),
          "scalar dispatch fallback");
  if (!selectedFamilyID)
    return selectedFamilyID.takeError();

  const DispatchI32FamilySpec *family =
      lookupDispatchFamilyRegistrationBySelectedFamilyID(*selectedFamilyID);
  if (!family)
    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("selected scalar dispatch fallback candidate @") +
            candidate.selectedVariant +
            " names unsupported finite selected component family '" +
            *selectedFamilyID + "'");
  return family;
}

llvm::Error makeFamilyMismatchError(
    const TargetArtifactCandidate &rvvCandidate,
    const DispatchI32FamilySpec &rvvFamily,
    const TargetArtifactCandidate &scalarCandidate,
    const DispatchI32FamilySpec &scalarFamily) {
  std::string message;
  llvm::raw_string_ostream stream(message);
  stream << "selected RVV dispatch case callable family '"
         << rvvFamily.diagnosticName
         << "' does not match selected scalar dispatch fallback callable "
            "family '"
         << scalarFamily.diagnosticName
         << "'; refusing to emit a mixed RVV/scalar dispatch artifact for @"
         << rvvCandidate.selectedVariant << " and @"
         << scalarCandidate.selectedVariant;
  stream.flush();
  return makeDispatchError(rvvCandidate.kernel, message);
}

llvm::Error validateRegisteredCallableRouteMetadata(
    const TargetArtifactCandidate &candidate,
    const TargetArtifactExporterRegistry &registry) {
  const TargetArtifactExporter *exporter = registry.lookup(candidate.routeID);
  if (!exporter)
    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("unknown selected callable artifact route id '") +
            candidate.routeID + "'");

  if (llvm::Error error =
          validateTargetArtifactCandidateAgainstExporter(candidate, *exporter)) {
    std::string message = llvm::toString(std::move(error));
    return makeDispatchError(candidate.kernel, message);
  }
  return llvm::Error::success();
}

bool isSameDispatchFamily(const DispatchI32FamilySpec &lhs,
                          const DispatchI32FamilySpec &rhs) {
  return lhs.rvvFamily && rhs.rvvFamily &&
         lhs.rvvFamily->familyID == rhs.rvvFamily->familyID &&
         lhs.diagnosticName == rhs.diagnosticName;
}

llvm::Error validateDispatchManifestRouteForFamily(
    KernelOp kernel, const DispatchI32FamilySpec &family,
    DispatchRouteKind routeKind, llvm::StringRef context) {
  const DispatchRouteManifestEntry *route =
      lookupRVVScalarDispatchRoute(family, routeKind);
  if (!route || !route->family)
    return makeDispatchError(
        kernel,
        llvm::Twine(context) +
            " requires a manifest-backed RVV+scalar dispatch route for " +
            family.diagnosticName);

  if (!isSameDispatchFamily(*route->family, family))
    return makeDispatchError(
        kernel,
        llvm::Twine(context) +
            " resolved a stale RVV+scalar dispatch manifest family for " +
            family.diagnosticName);
  return llvm::Error::success();
}

llvm::Error validateDispatchManifestRoutesForFamily(
    KernelOp kernel, const DispatchI32FamilySpec &family) {
  for (DispatchRouteKind routeKind : getRVVScalarDispatchRouteKinds())
    if (llvm::Error error = validateDispatchManifestRouteForFamily(
            kernel, family, routeKind, "selected dispatch pair"))
      return error;
  return llvm::Error::success();
}

bool isValidCParameterName(llvm::StringRef value) {
  if (value.empty())
    return false;
  unsigned char first = static_cast<unsigned char>(value.front());
  if (!std::isalpha(first) && value.front() != '_')
    return false;
  return llvm::all_of(value.drop_front(), [](char character) {
    unsigned char byte = static_cast<unsigned char>(character);
    return std::isalnum(byte) || character == '_';
  });
}

llvm::Error validateDispatchCParameterName(KernelOp kernel,
                                           llvm::StringRef value) {
  if (!isValidCParameterName(value))
    return makeDispatchError(
        kernel, llvm::Twine("runtime ABI parameter c_name '") + value +
                    "' must be a valid C identifier for RVV+scalar dispatch "
                    "source export");
  return llvm::Error::success();
}

llvm::Error validateDispatchRuntimeABIText(KernelOp kernel,
                                           llvm::StringRef fieldName,
                                           llvm::StringRef value) {
  constexpr std::size_t kMaxTextLength = 512;
  if (value.empty() || value.size() > kMaxTextLength)
    return makeDispatchError(
        kernel, llvm::Twine(fieldName) +
                    " must be bounded non-empty single-line metadata");

  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      return makeDispatchError(
          kernel, llvm::Twine(fieldName) +
                      " must be bounded non-empty single-line metadata");
    if (byte < 0x20 && character != '\t')
      return makeDispatchError(
          kernel, llvm::Twine(fieldName) +
                      " must be bounded non-empty single-line metadata");
  }

  if (containsForbiddenText(value))
    return makeDispatchError(
        kernel, llvm::Twine(fieldName) +
                    " must not contain secret-like or raw credential text");
  return llvm::Error::success();
}

llvm::StringRef getRuntimeParamStringAttr(RuntimeParamOp param,
                                          llvm::StringRef attrName) {
  auto attr = param->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr)
    return {};
  return attr.getValue();
}

llvm::Expected<support::RuntimeABIParameter>
makeRuntimeABIParameterFromRuntimeParam(KernelOp kernel,
                                        RuntimeParamOp param) {
  llvm::StringRef cName =
      getRuntimeParamStringAttr(param, support::kRuntimeParamCNameAttrName);
  llvm::StringRef cType =
      getRuntimeParamStringAttr(param, support::kRuntimeParamCTypeAttrName);
  llvm::StringRef role =
      getRuntimeParamStringAttr(param, support::kRuntimeParamABIRoleAttrName);
  llvm::StringRef ownership =
      getRuntimeParamStringAttr(param,
                                support::kRuntimeParamOwnershipAttrName);

  if (llvm::Error error = validateDispatchRuntimeABIText(
          kernel, "dispatch runtime_param c_name", cName))
    return std::move(error);
  if (llvm::Error error = validateDispatchRuntimeABIText(
          kernel, "dispatch runtime_param c_type", cType))
    return std::move(error);
  if (llvm::Error error = validateDispatchRuntimeABIText(
          kernel, "dispatch runtime_param abi_role", role))
    return std::move(error);
  if (llvm::Error error = validateDispatchRuntimeABIText(
          kernel, "dispatch runtime_param ownership", ownership))
    return std::move(error);
  if (llvm::Error error = validateDispatchCParameterName(kernel, cName))
    return std::move(error);

  std::optional<support::RuntimeABIParameterRole> parsedRole =
      support::symbolizeRuntimeABIParameterRole(role);
  if (!parsedRole)
    return makeDispatchError(
        kernel, llvm::Twine("unsupported tcrv.exec.runtime_param ABI role '") +
                    role + "'");

  std::optional<support::RuntimeABIParameterOwnership> parsedOwnership =
      support::symbolizeRuntimeABIParameterOwnership(ownership);
  if (!parsedOwnership)
    return makeDispatchError(
        kernel,
        llvm::Twine("unsupported tcrv.exec.runtime_param ownership '") +
            ownership + "'");

  return support::RuntimeABIParameter(cName, cType, *parsedRole,
                                      *parsedOwnership);
}

mlir::Operation *findDirectKernelSymbol(KernelOp kernel,
                                        llvm::StringRef symbolName) {
  if (!kernel || kernel.getBody().empty())
    return nullptr;

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto symbolAttr = op.getAttrOfType<mlir::StringAttr>(
        mlir::SymbolTable::getSymbolAttrName());
    if (symbolAttr && symbolAttr.getValue() == symbolName)
      return &op;
  }
  return nullptr;
}

llvm::Expected<const support::RuntimeABIParameter *>
findDispatchABIParameterByRole(
    KernelOp kernel, llvm::ArrayRef<support::RuntimeABIParameter> parameters,
    support::RuntimeABIParameterRole role, llvm::StringRef context) {
  llvm::Expected<const support::RuntimeABIParameter *> parameter =
      support::findUniqueRuntimeABIParameterByRole(parameters, role, context);
  if (!parameter) {
    std::string message = llvm::toString(parameter.takeError());
    return makeDispatchError(kernel, message);
  }
  return *parameter;
}

llvm::Error validateDispatchCallableABIParameterMirror(
    KernelOp kernel,
    llvm::ArrayRef<support::RuntimeABIParameter> metadataParameters,
    llvm::ArrayRef<support::RuntimeABIParameter> irBackedParameters,
    llvm::StringRef metadataSource, const DispatchI32FamilySpec &family) {
  if (!family.rvvFamily)
    return makeDispatchError(
        kernel, llvm::Twine("dispatch callable ABI mirror validation for ") +
                    family.diagnosticName +
                    " requires finite RVV binary family metadata");

  return support::validateFiniteBinaryCallableABIParameterMirror(
      kernel, metadataParameters, irBackedParameters, metadataSource,
      target::rvv::getRVVBinaryRuntimeABIContract(*family.rvvFamily));
}

llvm::Expected<CallableABIPlan>
buildDispatchCallableABIPlan(KernelOp kernel,
                             const DispatchI32FamilySpec &family) {
  if (!kernel || kernel.getBody().empty())
    return makeDispatchError(
        kernel, "requires a materialized tcrv.exec.kernel body");

  if (!family.rvvFamily)
    return makeDispatchError(
        kernel, llvm::Twine("dispatch callable ABI plan for ") +
                    family.diagnosticName +
                    " requires finite RVV binary family metadata");

  llvm::Expected<support::FiniteBinaryCallableABIPlan> finitePlan =
      support::buildFiniteBinaryCallableABIPlan(
          kernel,
          target::rvv::getRVVBinaryRuntimeABIContract(*family.rvvFamily));
  if (!finitePlan)
    return finitePlan.takeError();

  CallableABIPlan plan;
  plan.parameters = std::move(finitePlan->parameters);
  plan.bufferWindows = std::move(finitePlan->bufferWindows);
  plan.runtimeElementCountParam = finitePlan->runtimeElementCountParam;
  return plan;
}

llvm::Expected<DispatchRuntimeABIParameterBindings>
bindDispatchRuntimeABIParametersByRole(
    KernelOp kernel, llvm::ArrayRef<support::RuntimeABIParameter> parameters,
    llvm::StringRef context) {
  using Role = support::RuntimeABIParameterRole;
  DispatchRuntimeABIParameterBindings bindings;

  llvm::Expected<const support::RuntimeABIParameter *> lhs =
      findDispatchABIParameterByRole(kernel, parameters, Role::LHSInputBuffer,
                                     context);
  if (!lhs)
    return lhs.takeError();
  bindings.lhs = *lhs;

  llvm::Expected<const support::RuntimeABIParameter *> rhs =
      findDispatchABIParameterByRole(kernel, parameters, Role::RHSInputBuffer,
                                     context);
  if (!rhs)
    return rhs.takeError();
  bindings.rhs = *rhs;

  llvm::Expected<const support::RuntimeABIParameter *> out =
      findDispatchABIParameterByRole(kernel, parameters, Role::OutputBuffer,
                                     context);
  if (!out)
    return out.takeError();
  bindings.out = *out;

  llvm::Expected<const support::RuntimeABIParameter *> runtimeElementCount =
      findDispatchABIParameterByRole(kernel, parameters,
                                     Role::RuntimeElementCount, context);
  if (!runtimeElementCount)
    return runtimeElementCount.takeError();
  bindings.runtimeElementCount = *runtimeElementCount;

  llvm::Expected<const support::RuntimeABIParameter *>
      dispatchAvailabilityGuard = findDispatchABIParameterByRole(
          kernel, parameters, Role::DispatchAvailabilityGuard, context);
  if (!dispatchAvailabilityGuard)
    return dispatchAvailabilityGuard.takeError();
  bindings.dispatchAvailabilityGuard = *dispatchAvailabilityGuard;

  return bindings;
}

llvm::Expected<DispatchIRLink>
resolveDispatchIRLinkForPair(const DispatchPair &pair) {
  KernelOp kernel = pair.rvv.kernel;
  if (!kernel || kernel.getBody().empty())
    return makeDispatchError(
        kernel,
        "requires one direct tcrv.exec.dispatch to link the selected RVV "
        "dispatch case and selected scalar dispatch fallback callable route");

  DispatchOp dispatch;
  unsigned dispatchCount = 0;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto candidate = llvm::dyn_cast<DispatchOp>(op);
    if (!candidate)
      continue;
    dispatch = candidate;
    ++dispatchCount;
  }

  if (dispatchCount == 0)
    return makeDispatchError(
        kernel,
        "requires one direct tcrv.exec.dispatch to link the selected RVV "
        "dispatch case and selected scalar dispatch fallback callable route");
  if (dispatchCount > 1)
    return makeDispatchError(
        kernel,
        "requires exactly one direct tcrv.exec.dispatch to link the selected "
        "RVV dispatch case and selected scalar dispatch fallback callable "
        "route; found multiple");
  if (dispatch.getBody().empty())
    return makeDispatchError(
        kernel,
        "selected tcrv.exec.dispatch requires a materialized body block before "
        "RVV+scalar dispatch export");

  DispatchIRLink link;
  link.dispatch = dispatch;
  unsigned selectedCaseCount = 0;
  unsigned fallbackCount = 0;
  for (mlir::Operation &op : dispatch.getBody().front()) {
    if (auto dispatchCase = llvm::dyn_cast<DispatchCaseOp>(op)) {
      auto targetAttr =
          dispatchCase->getAttrOfType<mlir::FlatSymbolRefAttr>("target");
      if (targetAttr && targetAttr.getValue() == pair.rvv.selectedVariant) {
        link.selectedRVVCase = dispatchCase;
        ++selectedCaseCount;
      }
      continue;
    }

    if (auto fallback = llvm::dyn_cast<FallbackOp>(op)) {
      link.fallback = fallback;
      if (auto origin = fallback->getAttrOfType<mlir::StringAttr>("origin"))
        link.fallbackOrigin = origin.getValue().str();
      if (auto role =
              fallback->getAttrOfType<mlir::StringAttr>("fallback_role"))
        link.fallbackRole = role.getValue().str();
      ++fallbackCount;
    }
  }

  if (selectedCaseCount == 0)
    return makeDispatchError(
        kernel, llvm::Twine("selected RVV dispatch case @") +
                    pair.rvv.selectedVariant +
                    " must be present in tcrv.exec.dispatch before dispatch "
                    "C export");
  if (selectedCaseCount > 1)
    return makeDispatchError(
        kernel, llvm::Twine("selected RVV dispatch case @") +
                    pair.rvv.selectedVariant +
                    " has duplicate or ambiguous runtime_guard linkage");

  if (fallbackCount == 0)
    return makeDispatchError(
        kernel,
        llvm::Twine("selected scalar dispatch fallback callable route @") +
            pair.scalar.selectedVariant +
            " requires exactly one tcrv.exec.fallback target in the selected "
            "tcrv.exec.dispatch; found none");
  if (fallbackCount > 1)
    return makeDispatchError(
        kernel,
        llvm::Twine("selected scalar dispatch fallback callable route @") +
            pair.scalar.selectedVariant +
            " requires exactly one tcrv.exec.fallback target in the selected "
            "tcrv.exec.dispatch; found multiple");

  auto fallbackTarget =
      link.fallback->getAttrOfType<mlir::FlatSymbolRefAttr>("target");
  if (!fallbackTarget)
    return makeDispatchError(
        kernel,
        llvm::Twine("selected scalar dispatch fallback callable route @") +
            pair.scalar.selectedVariant +
            " requires tcrv.exec.fallback target symbol linkage");
  link.fallbackTarget = fallbackTarget.getValue().str();

  mlir::Operation *resolved =
      findDirectKernelSymbol(kernel, fallbackTarget.getValue());
  if (!resolved)
    return makeDispatchError(
        kernel,
        llvm::Twine("selected scalar dispatch fallback callable route @") +
            pair.scalar.selectedVariant + " references tcrv.exec.fallback "
            "target @" + fallbackTarget.getValue() +
            ", but that target is unknown in the enclosing tcrv.exec.kernel");

  link.fallbackVariant = llvm::dyn_cast<VariantOp>(resolved);
  if (!link.fallbackVariant)
    return makeDispatchError(
        kernel,
        llvm::Twine("selected scalar dispatch fallback callable route @") +
            pair.scalar.selectedVariant + " references tcrv.exec.fallback "
            "target @" + fallbackTarget.getValue() +
            ", but that target resolves to a direct sibling symbol that is "
            "not a tcrv.exec.variant");

  if (fallbackTarget.getValue() != pair.scalar.selectedVariant)
    return makeDispatchError(
        kernel,
        llvm::Twine("selected scalar dispatch fallback callable route @") +
            pair.scalar.selectedVariant +
            " does not match tcrv.exec.fallback target @" +
            fallbackTarget.getValue());

  return link;
}

llvm::Expected<RuntimeParamOp>
resolveRuntimeGuardParamFromSelectedCase(const DispatchPair &pair) {
  KernelOp kernel = pair.rvv.kernel;
  DispatchCaseOp selectedCase = pair.irLink.selectedRVVCase;
  if (!selectedCase)
    return makeDispatchError(
        kernel,
        "requires selected RVV tcrv.exec.case IR link before runtime_guard "
        "validation");

  auto runtimeGuard =
      selectedCase->getAttrOfType<mlir::FlatSymbolRefAttr>(
          kRuntimeGuardAttrName);
  if (!runtimeGuard)
    return makeDispatchError(
        kernel, llvm::Twine("selected RVV dispatch case @") +
                    pair.rvv.selectedVariant +
                    " requires runtime_guard symbol reference to a "
                    "dispatch-availability-guard tcrv.exec.runtime_param");

  mlir::Operation *resolved =
      findDirectKernelSymbol(kernel, runtimeGuard.getValue());
  if (!resolved)
    return makeDispatchError(
        kernel, llvm::Twine("selected RVV dispatch case @") +
                    pair.rvv.selectedVariant +
                    " runtime_guard references unknown symbol @" +
                    runtimeGuard.getValue());

  auto guardParam = llvm::dyn_cast<RuntimeParamOp>(resolved);
  if (!guardParam)
    return makeDispatchError(
        kernel, llvm::Twine("selected RVV dispatch case @") +
                    pair.rvv.selectedVariant + " runtime_guard @" +
                    runtimeGuard.getValue() +
                    " resolves to a direct sibling symbol that is not a "
                    "tcrv.exec.runtime_param");

  llvm::StringRef role = getRuntimeParamStringAttr(
      guardParam, support::kRuntimeParamABIRoleAttrName);
  if (role != support::stringifyRuntimeABIParameterRole(
                  support::RuntimeABIParameterRole::
                      DispatchAvailabilityGuard))
    return makeDispatchError(
        kernel, llvm::Twine("selected RVV dispatch case @") +
                    pair.rvv.selectedVariant + " runtime_guard @" +
                    runtimeGuard.getValue() +
                    " must reference a tcrv.exec.runtime_param with ABI role "
                    "'dispatch-availability-guard'");

  support::RuntimeABIParamSpec guardSpec =
      support::getDispatchAvailabilityGuardParamSpec(/*cName=*/"");
  llvm::SmallVector<support::RuntimeABIParamSpec, 1> guardSpecs;
  guardSpecs.push_back(guardSpec);
  llvm::SmallVector<RuntimeParamOp, 1> guardParamsByRole;
  if (llvm::Error error =
          support::collectRuntimeABIParams(kernel, guardSpecs,
                                           guardParamsByRole)) {
    std::string message = llvm::toString(std::move(error));
    return makeDispatchError(kernel, message);
  }
  if (guardParamsByRole.front().getOperation() != guardParam.getOperation())
    return makeDispatchError(
        kernel, llvm::Twine("selected RVV dispatch case @") +
                    pair.rvv.selectedVariant + " runtime_guard @" +
                    runtimeGuard.getValue() +
                    " does not match the unique kernel runtime_param with ABI "
                    "role 'dispatch-availability-guard'");

  return guardParam;
}

llvm::Expected<DispatchABIPlan> buildDispatchABIPlan(const DispatchPair &pair) {
  llvm::Expected<CallableABIPlan> callablePlan =
      buildDispatchCallableABIPlan(pair.rvv.kernel, *pair.family);
  if (!callablePlan) {
    std::string message = llvm::toString(callablePlan.takeError());
    return makeDispatchError(pair.rvv.kernel, message);
  }

  if (llvm::Error error = validateDispatchCallableABIParameterMirror(
          pair.rvv.kernel, pair.rvv.runtimeABIParameters,
          callablePlan->parameters, "selected RVV callable artifact route",
          *pair.family)) {
    std::string message = llvm::toString(std::move(error));
    return makeDispatchError(pair.rvv.kernel, message);
  }
  if (llvm::Error error = validateDispatchCallableABIParameterMirror(
          pair.scalar.kernel, pair.scalar.runtimeABIParameters,
          callablePlan->parameters,
          "selected scalar callable artifact route", *pair.family)) {
    std::string message = llvm::toString(std::move(error));
    return makeDispatchError(pair.scalar.kernel, message);
  }

  DispatchOp dispatch = pair.irLink.dispatch;
  if (!dispatch)
    return makeDispatchError(
        pair.rvv.kernel,
        "requires tcrv.exec.dispatch IR link to build dispatch ABI runtime "
        "parameter boundary");

  if (dispatch->hasAttr(kDispatchRuntimeABIParametersAttrName))
    return makeDispatchError(
        pair.rvv.kernel,
        llvm::Twine(kDispatchRuntimeABIParametersAttrName) +
            " is detached dispatch ABI metadata; use direct "
            "tcrv.exec.runtime_param IR for dispatch runtime ABI scalar "
            "values");

  llvm::Expected<const support::RuntimeABIParameter *> runtimeCount =
      findDispatchABIParameterByRole(
          pair.rvv.kernel, callablePlan->parameters,
          support::RuntimeABIParameterRole::RuntimeElementCount,
          "selected RVV callable ABI plan");
  if (!runtimeCount)
    return runtimeCount.takeError();

  DispatchABIPlan plan;
  llvm::SmallVector<support::RuntimeABIParamSpec, 1> runtimeParamSpecs;
  auto countSpecs =
      tianchenrv::target::rvv::getRVVBinaryRuntimeElementCountParamSpecs(
          *pair.family->rvvFamily, (*runtimeCount)->cName);
  runtimeParamSpecs.append(countSpecs.begin(), countSpecs.end());
  if (llvm::Error error = support::collectRuntimeABIParams(
          pair.rvv.kernel, runtimeParamSpecs, plan.runtimeParams)) {
    std::string message = llvm::toString(std::move(error));
    return makeDispatchError(pair.rvv.kernel, message);
  }

  RuntimeParamOp countParam = plan.runtimeParams.front();

  llvm::Expected<RuntimeParamOp> guardParam =
      resolveRuntimeGuardParamFromSelectedCase(pair);
  if (!guardParam)
    return guardParam.takeError();
  plan.runtimeParams.push_back(*guardParam);
  plan.runtimeGuardParam = *guardParam;

  llvm::Expected<support::RuntimeABIParameter> count =
      makeRuntimeABIParameterFromRuntimeParam(pair.rvv.kernel, countParam);
  if (!count)
    return count.takeError();
  if (count->cName != (*runtimeCount)->cName ||
      count->cType != (*runtimeCount)->cType ||
      count->ownership != (*runtimeCount)->ownership)
    return makeDispatchError(
        pair.rvv.kernel,
        "tcrv.exec.runtime_param runtime-element-count must agree with the "
        "selected RVV callable runtime ABI parameter c_name/c_type/ownership");

  llvm::Expected<support::RuntimeABIParameter> guard =
      makeRuntimeABIParameterFromRuntimeParam(pair.rvv.kernel, *guardParam);
  if (!guard)
    return guard.takeError();
  support::RuntimeABIParameter expectedGuard =
      tianchenrv::target::rvv_scalar::
          makeRVVScalarDispatchAvailabilityGuardParameter(guard->cName);
  if (guard->cType != expectedGuard.cType)
    return makeDispatchError(
        pair.rvv.kernel,
        "dispatch availability guard tcrv.exec.runtime_param must use c type "
        "'int'");
  if (guard->ownership != expectedGuard.ownership)
    return makeDispatchError(
        pair.rvv.kernel,
        llvm::Twine("dispatch availability guard tcrv.exec.runtime_param must "
                    "use ownership '") +
            support::stringifyRuntimeABIParameterOwnership(
                expectedGuard.ownership) +
            "'");

  plan.parameters = std::move(callablePlan->parameters);
  plan.parameters.push_back(*guard);
  plan.bufferWindows = std::move(callablePlan->bufferWindows);
  return plan;
}

llvm::Expected<DispatchPair> collectDispatchPairFromCandidates(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  TargetArtifactExporterRegistry registry;
  if (llvm::Error error =
          rvv::registerRVVMicrokernelTargetExporters(registry))
    return std::move(error);
  if (llvm::Error error =
          scalar::registerScalarMicrokernelTargetExporters(registry))
    return std::move(error);

  const TargetArtifactCandidate *rvvCandidate = nullptr;
  const TargetArtifactCandidate *scalarCandidate = nullptr;
  const DispatchI32FamilySpec *rvvFamily = nullptr;
  const DispatchI32FamilySpec *scalarFamily = nullptr;
  for (const TargetArtifactCandidate &candidate : candidates) {
    if (candidate.role == kDispatchCaseRole) {
      llvm::Expected<const DispatchI32FamilySpec *> family =
          getRVVCallableCandidateFamilyFromSelectedPlan(candidate);
      if (!family)
        return family.takeError();
      if (rvvCandidate)
        return makeDispatchError(candidate.kernel,
                                 "requires exactly one supported RVV dispatch "
                                 "case callable route; found duplicate");
      if (llvm::Error error =
              validateRVVCallableCandidateShapeForFamily(candidate, **family))
        return std::move(error);
      if (llvm::Error error =
              validateRegisteredCallableRouteMetadata(candidate, registry))
        return std::move(error);
      rvvCandidate = &candidate;
      rvvFamily = *family;
      continue;
    }

    if (candidate.role == kDispatchFallbackRole) {
      llvm::Expected<const DispatchI32FamilySpec *> family =
          getScalarCallableCandidateFamilyFromSelectedPlan(candidate);
      if (!family)
        return family.takeError();
      if (scalarCandidate)
        return makeDispatchError(
            candidate.kernel,
            "requires exactly one supported scalar dispatch fallback callable "
            "route; found duplicate");
      if (llvm::Error error =
              validateScalarCallableCandidateShapeForFamily(candidate,
                                                            **family))
        return std::move(error);
      if (llvm::Error error =
              validateRegisteredCallableRouteMetadata(candidate, registry))
        return std::move(error);
      scalarCandidate = &candidate;
      scalarFamily = *family;
      continue;
    }

    // Unknown-role candidates may still name a known compatibility route.
    // Use the registration record only to produce a stale-role diagnostic; it
    // must not admit the candidate into the default dispatch pair.
    if (const DispatchI32FamilySpec *family =
            lookupDispatchFamilyRegistrationByRVVRouteID(candidate.routeID))
      return validateRVVCallableCandidateShapeForFamily(candidate, *family);
    if (const DispatchI32FamilySpec *family =
            lookupDispatchFamilyRegistrationByScalarRouteID(candidate.routeID))
      return validateScalarCallableCandidateShapeForFamily(candidate, *family);

    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("unsupported supported artifact candidate route '") +
            candidate.routeID +
            "' for RVV+scalar binary dispatch export");
  }

  if (!rvvCandidate)
    return makeModuleDispatchError(
        "requires exactly one supported RVV dispatch case callable route; "
        "found none");
  if (!scalarCandidate)
    return makeModuleDispatchError(
        "requires exactly one supported scalar dispatch fallback callable "
        "route; found none");
  if (rvvCandidate->kernel != scalarCandidate->kernel)
    return makeModuleDispatchError(
        "requires RVV dispatch case and scalar fallback callable routes in "
        "the same tcrv.exec.kernel");
  if (rvvFamily != scalarFamily)
    return makeFamilyMismatchError(*rvvCandidate, *rvvFamily, *scalarCandidate,
                                   *scalarFamily);

  DispatchPair pair;
  pair.family = rvvFamily;
  pair.rvv = *rvvCandidate;
  pair.scalar = *scalarCandidate;
  llvm::Expected<DispatchPair::CompositeIdentity> composite =
      deriveDispatchCompositeIdentityFromSelectedComponents(pair);
  if (!composite)
    return composite.takeError();
  pair.composite = std::move(*composite);
  llvm::Expected<DispatchIRLink> irLink = resolveDispatchIRLinkForPair(pair);
  if (!irLink)
    return irLink.takeError();
  pair.irLink = std::move(*irLink);
  llvm::Expected<DispatchABIPlan> abiPlan = buildDispatchABIPlan(pair);
  if (!abiPlan)
    return abiPlan.takeError();
  pair.abiPlan = std::move(*abiPlan);
  llvm::Expected<const DispatchRVVVectorShapeConfig *> selectedShape =
      resolveDispatchPairSelectedVectorShape(pair);
  if (!selectedShape)
    return selectedShape.takeError();
  pair.selectedShape = *selectedShape;
  llvm::Expected<tianchenrv::target::rvv::RVVBinarySelectedConfigContract>
      selectedConfig =
          buildDispatchPairSelectedConfigContract(pair, **selectedShape);
  if (!selectedConfig)
    return selectedConfig.takeError();
  pair.selectedConfig = std::move(*selectedConfig);
  if (llvm::Error error = validateDispatchSelectedConfigContractMetadata(
          pair.rvv, pair.selectedConfig))
    return std::move(error);
  if (llvm::Error error =
          validateDispatchManifestRoutesForFamily(pair.rvv.kernel,
                                                  *pair.family))
    return std::move(error);
  return pair;
}

llvm::Expected<DispatchPair> collectDispatchPair(mlir::ModuleOp module) {
  llvm::SmallVector<TargetArtifactCandidate, 4> candidates;
  if (llvm::Error error = collectTargetArtifactCandidates(module, candidates))
    return std::move(error);
  return collectDispatchPairFromCandidates(candidates);
}

llvm::Expected<bool> matchRVVScalarDispatchCandidatesForFamily(
    llvm::ArrayRef<TargetArtifactCandidate> candidates,
    const DispatchI32FamilySpec &expectedFamily) {
  if (candidates.size() != 2)
    return false;

  const TargetArtifactCandidate *rvvCandidate = nullptr;
  const TargetArtifactCandidate *scalarCandidate = nullptr;
  const DispatchI32FamilySpec *rvvFamily = nullptr;
  const DispatchI32FamilySpec *scalarFamily = nullptr;
  for (const TargetArtifactCandidate &candidate : candidates) {
    if (candidate.role == kDispatchCaseRole) {
      if (!lookupDispatchFamilyRegistrationByRVVRouteID(candidate.routeID))
        return false;
      llvm::Expected<const DispatchI32FamilySpec *> selectedFamily =
          getRVVCallableCandidateFamilyFromSelectedPlan(candidate);
      if (!selectedFamily)
        return selectedFamily.takeError();
      if (llvm::Error error =
              validateRVVCallableCandidateShapeForFamily(candidate,
                                                         **selectedFamily))
        return std::move(error);
      if (rvvCandidate)
        return makeDispatchError(
            candidate.kernel,
            "requires exactly one supported RVV dispatch "
            "case callable route; found duplicate");
      rvvCandidate = &candidate;
      rvvFamily = *selectedFamily;
      continue;
    }

    if (candidate.role == kDispatchFallbackRole) {
      if (!lookupDispatchFamilyRegistrationByScalarRouteID(candidate.routeID))
        return false;
      llvm::Expected<const DispatchI32FamilySpec *> selectedFamily =
          getScalarCallableCandidateFamilyFromSelectedPlan(candidate);
      if (!selectedFamily)
        return selectedFamily.takeError();
      if (llvm::Error error =
              validateScalarCallableCandidateShapeForFamily(candidate,
                                                            **selectedFamily))
        return std::move(error);
      if (scalarCandidate)
        return makeDispatchError(
            candidate.kernel,
            "requires exactly one supported scalar dispatch fallback "
            "callable route; found duplicate");
      scalarCandidate = &candidate;
      scalarFamily = *selectedFamily;
      continue;
    }
    return false;
  }
  if (!rvvCandidate || !scalarCandidate)
    return false;
  if (rvvFamily != scalarFamily)
    return makeFamilyMismatchError(*rvvCandidate, *rvvFamily,
                                   *scalarCandidate, *scalarFamily);

  return rvvFamily == &expectedFamily;
}

llvm::Expected<bool> matchRVVScalarI32VAddDispatchCandidates(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  return matchRVVScalarDispatchCandidatesForFamily(
      candidates, getI32VAddDispatchFamilySpec());
}

llvm::Expected<bool> matchRVVScalarI32VSubDispatchCandidates(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  return matchRVVScalarDispatchCandidatesForFamily(
      candidates, getI32VSubDispatchFamilySpec());
}

llvm::Expected<bool> matchRVVScalarI32VMulDispatchCandidates(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  return matchRVVScalarDispatchCandidatesForFamily(
      candidates, getI32VMulDispatchFamilySpec());
}

llvm::Expected<bool> matchRVVScalarI64VAddDispatchCandidates(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  return matchRVVScalarDispatchCandidatesForFamily(
      candidates, getI64VAddDispatchFamilySpec());
}

llvm::Expected<bool> matchRVVScalarI64VSubDispatchCandidates(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  return matchRVVScalarDispatchCandidatesForFamily(
      candidates, getI64VSubDispatchFamilySpec());
}

llvm::Expected<bool> matchRVVScalarI64VMulDispatchCandidates(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  return matchRVVScalarDispatchCandidatesForFamily(
      candidates, getI64VMulDispatchFamilySpec());
}

TargetArtifactCompositeMatchFn
getDispatchCompositeMatchFn(const DispatchI32FamilySpec &family) {
  if (&family == &getI32VAddDispatchFamilySpec())
    return matchRVVScalarI32VAddDispatchCandidates;
  if (&family == &getI32VSubDispatchFamilySpec())
    return matchRVVScalarI32VSubDispatchCandidates;
  if (&family == &getI32VMulDispatchFamilySpec())
    return matchRVVScalarI32VMulDispatchCandidates;
  if (&family == &getI64VAddDispatchFamilySpec())
    return matchRVVScalarI64VAddDispatchCandidates;
  if (&family == &getI64VSubDispatchFamilySpec())
    return matchRVVScalarI64VSubDispatchCandidates;
  if (&family == &getI64VMulDispatchFamilySpec())
    return matchRVVScalarI64VMulDispatchCandidates;
  return nullptr;
}

llvm::Error validateRVVScalarDispatchCandidates(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  llvm::Expected<DispatchPair> pair =
      collectDispatchPairFromCandidates(candidates);
  if (!pair)
    return pair.takeError();
  return llvm::Error::success();
}

llvm::Expected<llvm::SmallVector<support::RuntimeABIParameter, 5>>
resolveRVVScalarDispatchRuntimeABIParameters(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  llvm::Expected<DispatchPair> pair =
      collectDispatchPairFromCandidates(candidates);
  if (!pair)
    return pair.takeError();
  return std::move(pair->abiPlan.parameters);
}

const SelectedPlanMetadataEntry *findFirstSelectedPlanMetadataEntry(
    const TargetArtifactCandidate &candidate, llvm::StringRef name) {
  for (const SelectedPlanMetadataEntry &metadata :
       candidate.selectedPlanMetadata)
    if (metadata.name == name)
      return &metadata;
  return nullptr;
}

void printDispatchSelectedEmitCBodyMappingSummary(llvm::raw_ostream &os,
                                                  const DispatchPair &pair) {
  const SelectedPlanMetadataEntry *routeKind =
      findFirstSelectedPlanMetadataEntry(
          pair.rvv, tianchenrv::target::rvv::getRVVEmitCRouteKindMetadataName());
  const SelectedPlanMetadataEntry *sourceAuthority =
      findFirstSelectedPlanMetadataEntry(
          pair.rvv,
          tianchenrv::target::rvv::getRVVEmitCSourceAuthorityMetadataName());
  const SelectedPlanMetadataEntry *requiredHeader =
      findFirstSelectedPlanMetadataEntry(
          pair.rvv,
          tianchenrv::target::rvv::getRVVEmitCRequiredHeaderMetadataName());
  const SelectedPlanMetadataEntry *arithmeticIntrinsic =
      findFirstSelectedPlanMetadataEntry(
          pair.rvv,
          tianchenrv::target::rvv::getRVVEmitCArithmeticIntrinsicMetadataName());
  if (!routeKind || !sourceAuthority || !requiredHeader || !arithmeticIntrinsic)
    return;

  os << "/* dispatch_rvv_emitc_body_mapping: route_kind=" << routeKind->value
     << ", source_authority=" << sourceAuthority->value
     << ", required_header_metadata=validated-selected-plan-entry"
     << ", arithmetic_intrinsic_metadata=validated-selected-plan-entry"
     << ", embedded_rvv_body=selected-rvv-source-artifact */\n";
}

TargetArtifactCompositeBundleMetadata
deriveRVVScalarDispatchBundleMetadataFromPair(const DispatchPair &pair) {
  TargetArtifactCompositeBundleMetadata metadata;
  metadata.runtimeABIKind = pair.composite.runtimeABIKind;
  metadata.runtimeABIName = pair.composite.runtimeABIName;
  metadata.componentGroup = pair.composite.componentGroup;
  metadata.externalABIName = pair.composite.externalABIName;
  const auto &contract = pair.selectedConfig;
  metadata.selectedPlanMetadata.push_back(
      {"tcrv_rvv.dispatch_contract_runtime_element_count_c_name",
       contract.getRuntimeElementCountCName().str(),
       kDispatchSelectedConfigMetadataRole.str(),
       kDispatchSelectedConfigMetadataNote.str()});
  metadata.selectedPlanMetadata.push_back(
      {"tcrv_rvv.dispatch_contract_selected_vector_config",
       contract.formatDispatchContractSelectedVectorConfigMetadataValue(),
       kDispatchSelectedConfigMetadataRole.str(),
       kDispatchSelectedConfigMetadataNote.str()});
  metadata.selectedPlanMetadata.push_back(
      {"tcrv_rvv.dispatch_contract_runtime_vl_boundary",
       contract.formatDispatchContractRuntimeVLBoundaryMetadataValue(),
       kDispatchSelectedConfigMetadataRole.str(),
       kDispatchSelectedConfigMetadataNote.str()});
  metadata.selectedPlanMetadata.push_back(
      {"tcrv_rvv.dispatch_contract_selected_role",
       contract.getSelectedRole().str(),
       kDispatchSelectedConfigMetadataRole.str(),
       kDispatchSelectedConfigMetadataNote.str()});
  metadata.selectedPlanMetadata.push_back(
      {"tcrv_rvv.dispatch_contract_descriptor_element_count",
       std::to_string(contract.getDescriptorElementCount()),
       kDispatchSelectedConfigMetadataRole.str(),
       kDispatchSelectedConfigMetadataNote.str()});
  metadata.selectedPlanMetadata.push_back(
      {"tcrv_rvv.dispatch_contract_selected_config_profile_hardware_facts",
       contract.formatSelectedConfigProfileHardwareFactsMetadataValue(),
       kDispatchSelectedConfigMetadataRole.str(),
       kDispatchSelectedConfigMetadataNote.str()});
  metadata.selectedPlanMetadata.push_back(
      {"tcrv_rvv.dispatch_contract_selected_config_profile_variant_config",
       contract.formatSelectedConfigProfileVariantConfigMetadataValue(),
       kDispatchSelectedConfigMetadataRole.str(),
       kDispatchSelectedConfigMetadataNote.str()});
  metadata.selectedPlanMetadata.push_back(
      {"tcrv_rvv.dispatch_contract_selected_config_profile_runtime_roles",
       contract.formatSelectedConfigProfileRuntimeRolesMetadataValue(),
       kDispatchSelectedConfigMetadataRole.str(),
       kDispatchSelectedConfigMetadataNote.str()});
  const SelectedPlanMetadataEntry *sourceKind =
      findFirstSelectedPlanMetadataEntry(
          pair.rvv,
          tianchenrv::target::rvv::
              getRVVSelectedBinarySourceKindMetadataName());
  const SelectedPlanMetadataEntry *microkernelOp =
      findFirstSelectedPlanMetadataEntry(
          pair.rvv,
          tianchenrv::target::rvv::
              getRVVSelectedBinaryMicrokernelOpMetadataName());
  if (sourceKind && microkernelOp) {
    metadata.selectedPlanMetadata.push_back(
        {"tcrv_rvv.dispatch_contract_selected_source_identity",
         contract.formatDispatchContractSelectedSourceIdentityMetadataValue(
             sourceKind->value, microkernelOp->value),
         kDispatchSelectedConfigMetadataRole.str(),
         kDispatchSelectedConfigMetadataNote.str()});
  }
  return metadata;
}

llvm::Expected<TargetArtifactCompositeBundleMetadata>
resolveRVVScalarDispatchBundleMetadata(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  llvm::Expected<DispatchPair> pair =
      collectDispatchPairFromCandidates(candidates);
  if (!pair)
    return pair.takeError();
  return deriveRVVScalarDispatchBundleMetadataFromPair(*pair);
}

std::string sanitizeCIdentifierComponent(llvm::StringRef value) {
  std::string result;
  result.reserve(std::min<std::size_t>(value.size(), 64));
  for (char character : value.take_front(64)) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (std::isalnum(byte))
      result.push_back(character);
    else
      result.push_back('_');
  }
  if (result.empty() ||
      std::isdigit(static_cast<unsigned char>(result.front())))
    result.insert(result.begin(), '_');
  return result;
}

llvm::Expected<std::string>
requireSelectedComponentFamilyID(const TargetArtifactCandidate &candidate,
                                 llvm::StringRef metadataName,
                                 llvm::StringRef componentLabel) {
  const SelectedPlanMetadataEntry *match = nullptr;
  unsigned count = 0;
  for (const SelectedPlanMetadataEntry &metadata :
       candidate.selectedPlanMetadata) {
    if (metadata.name == metadataName) {
      match = &metadata;
      ++count;
    }
  }

  if (count == 0)
    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("selected ") + componentLabel + " candidate @" +
            candidate.selectedVariant + " requires selected_plan_metadata '" +
            metadataName + "' before RVV+scalar dispatch identity export");
  if (count > 1)
    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("selected ") + componentLabel + " candidate @" +
            candidate.selectedVariant +
            " has duplicate selected_plan_metadata '" + metadataName + "'");
  std::string fieldName =
      (llvm::Twine(componentLabel) + " selected component family id").str();
  if (llvm::Error error = validateDispatchRuntimeABIText(
          candidate.kernel, fieldName, match->value))
    return std::move(error);
  return match->value;
}

std::string deriveDispatchFunctionStemFromSelectedFamily(
    llvm::StringRef selectedFamilyID) {
  return sanitizeCIdentifierComponent(selectedFamilyID);
}

std::string deriveDispatchHeaderGuardStemFromSelectedFamily(
    llvm::StringRef selectedFamilyID) {
  std::string guard = deriveDispatchFunctionStemFromSelectedFamily(
      selectedFamilyID);
  for (char &character : guard)
    character = static_cast<char>(
        std::toupper(static_cast<unsigned char>(character)));
  return guard;
}

llvm::Expected<DispatchPair::CompositeIdentity>
deriveDispatchCompositeIdentityFromSelectedComponents(
    const DispatchPair &pair) {
  llvm::Expected<std::string> rvvFamilyID =
      requireSelectedComponentFamilyID(
          pair.rvv,
          tianchenrv::target::rvv::getRVVSelectedBinaryFamilyMetadataName(),
          "RVV dispatch case");
  if (!rvvFamilyID)
    return rvvFamilyID.takeError();

  llvm::Expected<std::string> scalarFamilyID =
      requireSelectedComponentFamilyID(
          pair.scalar,
          tianchenrv::target::rvv_scalar::
              getScalarSelectedBinaryFamilyMetadataName(),
          "scalar dispatch fallback");
  if (!scalarFamilyID)
    return scalarFamilyID.takeError();

  if (*rvvFamilyID != *scalarFamilyID)
    return makeDispatchError(
        pair.rvv.kernel,
        llvm::Twine("selected RVV dispatch case family '") + *rvvFamilyID +
            "' does not match selected scalar dispatch fallback family '" +
            *scalarFamilyID +
            "'; selected component plans are the dispatch identity authority");

  if (pair.family && pair.family->rvvFamily &&
      *rvvFamilyID != pair.family->rvvFamily->familyID)
    return makeDispatchError(
        pair.rvv.kernel,
        llvm::Twine("selected component family '") + *rvvFamilyID +
            "' does not match finite dispatch route registration metadata '" +
            pair.family->rvvFamily->familyID +
            "'; descriptor-local metadata cannot alter RVV+scalar dispatch "
            "identity");

  DispatchPair::CompositeIdentity identity;
  identity.diagnosticName = *rvvFamilyID;
  identity.functionStem =
      deriveDispatchFunctionStemFromSelectedFamily(*rvvFamilyID);
  identity.headerGuardStem =
      deriveDispatchHeaderGuardStemFromSelectedFamily(*rvvFamilyID);
  identity.runtimeABIKind =
      "rvv-scalar-dispatch-runtime-callable-c-abi";
  identity.runtimeABIName =
      (llvm::Twine("rvv-scalar-") + *rvvFamilyID +
       "-dispatch-runtime-callable-c-function.v1")
          .str();
  identity.componentGroup =
      (llvm::Twine("rvv-scalar-") + *rvvFamilyID +
       "-dispatch-external-abi.v1")
          .str();
  identity.externalABIName = identity.runtimeABIName;
  identity.selfCheckSuccessMarker =
      (llvm::Twine("tcrv_rvv_scalar_") + identity.functionStem +
       "_dispatch_self_check_ok")
          .str();
  return identity;
}

std::string makeRVVFunctionName(const DispatchPair &pair) {
  const TargetArtifactCandidate &candidate = pair.rvv;
  KernelOp kernel = candidate.kernel;
  std::string name;
  llvm::raw_string_ostream stream(name);
  stream << "tcrv_rvv_" << pair.composite.functionStem << "_microkernel_"
         << sanitizeCIdentifierComponent(kernel.getSymName()) << "_"
         << sanitizeCIdentifierComponent(candidate.selectedVariant);
  stream.flush();
  return name;
}

std::string makeScalarFunctionName(const DispatchPair &pair) {
  const TargetArtifactCandidate &candidate = pair.scalar;
  KernelOp kernel = candidate.kernel;
  std::string name;
  llvm::raw_string_ostream stream(name);
  stream << "tcrv_scalar_" << pair.composite.functionStem << "_microkernel_"
         << sanitizeCIdentifierComponent(kernel.getSymName()) << "_"
         << sanitizeCIdentifierComponent(candidate.selectedVariant);
  stream.flush();
  return name;
}

std::string makeDispatcherFunctionName(const DispatchPair &pair) {
  KernelOp kernel = pair.rvv.kernel;
  std::string name;
  llvm::raw_string_ostream stream(name);
  stream << "tcrv_dispatch_" << pair.composite.functionStem << "_"
         << sanitizeCIdentifierComponent(kernel.getSymName());
  stream.flush();
  return name;
}

std::string makeDispatchHeaderIncludeGuard(const DispatchPair &pair) {
  KernelOp kernel = pair.rvv.kernel;
  std::string guard = "TIANCHENRV_RVV_SCALAR_";
  guard += pair.composite.headerGuardStem;
  guard += "_DISPATCH_";
  guard += sanitizeCIdentifierComponent(kernel.getSymName());
  guard += "_H";
  for (char &character : guard)
    character = static_cast<char>(
        std::toupper(static_cast<unsigned char>(character)));
  return guard;
}

VariantOp findDirectVariant(KernelOp kernel, llvm::StringRef symbol) {
  if (!kernel || kernel.getBody().empty())
    return VariantOp();
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto variant = llvm::dyn_cast<VariantOp>(op);
    if (variant && variant.getSymName() == symbol)
      return variant;
  }
  return VariantOp();
}

llvm::Expected<bool> variantRequiresCapabilityID(
    KernelOp kernel, VariantOp variant, const TargetCapabilitySet &capabilities,
    llvm::StringRef id) {
  auto requiresAttr = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!requiresAttr)
    return makeDispatchError(kernel, llvm::Twine("selected RVV dispatch case "
                                                 "variant @") +
                                         variant.getSymName() +
                                         " requires a 'requires' capability "
                                         "symbol list");

  for (mlir::Attribute attr : requiresAttr) {
    auto symbol = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
    if (!symbol || symbol.getValue().trim().empty())
      return makeDispatchError(
          kernel, llvm::Twine("selected RVV dispatch case variant @") +
                      variant.getSymName() +
                      " requires only non-empty capability symbol references");

    const CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbol.getValue());
    if (!capability)
      continue;
    if (capability->satisfiesID(id))
      return true;
  }

  return false;
}

llvm::Expected<bool> variantRequiresVectorShapeConfig(
    KernelOp kernel, VariantOp variant, const TargetCapabilitySet &capabilities,
    const DispatchRVVVectorShapeConfig &config) {
  llvm::Expected<bool> requiresSEW =
      variantRequiresCapabilityID(kernel, variant, capabilities,
                                  config.sewCapabilityID);
  if (!requiresSEW)
    return requiresSEW.takeError();
  llvm::Expected<bool> requiresLMUL =
      variantRequiresCapabilityID(kernel, variant, capabilities,
                                  config.lmulCapabilityID);
  if (!requiresLMUL)
    return requiresLMUL.takeError();
  llvm::Expected<bool> requiresTail =
      variantRequiresCapabilityID(kernel, variant, capabilities,
                                  config.tailPolicyCapabilityID);
  if (!requiresTail)
    return requiresTail.takeError();
  llvm::Expected<bool> requiresMask =
      variantRequiresCapabilityID(kernel, variant, capabilities,
                                  config.maskPolicyCapabilityID);
  if (!requiresMask)
    return requiresMask.takeError();
  return *requiresSEW && *requiresLMUL && *requiresTail && *requiresMask;
}

llvm::Error requireDispatchConfigCapabilityProperty(
    KernelOp kernel, const TargetCapabilitySet &capabilities,
    llvm::StringRef id, llvm::StringRef propertyName,
    llvm::StringRef expectedValue,
    const DispatchRVVVectorShapeConfig &config) {
  const CapabilityDescriptor *capability = capabilities.lookupProviderByID(id);
  if (!capability || !capability->isAvailable())
    return makeDispatchError(
        kernel, llvm::Twine("selected RVV dispatch shape '") +
                    config.diagnosticSpelling +
                    "' requires available capability id '" + id + "'");

  llvm::StringRef value = capability->getProperty(propertyName).trim();
  if (llvm::Error error =
          validateDispatchRuntimeABIText(kernel, propertyName, value))
    return error;
  if (value != expectedValue)
    return makeDispatchError(
        kernel, llvm::Twine("selected RVV dispatch shape '") +
                    config.diagnosticSpelling + "' capability id '" + id +
                    "' property '" + propertyName + "' must be '" +
                    expectedValue + "'");
  return llvm::Error::success();
}

llvm::Error validateDispatchConfigCapabilityProperties(
    KernelOp kernel, const TargetCapabilitySet &capabilities,
    const DispatchRVVVectorShapeConfig &config) {
  if (llvm::Error error = requireDispatchConfigCapabilityProperty(
          kernel, capabilities, config.sewCapabilityID, kSEWBitsPropertyName,
          llvm::Twine(config.sewBits).str(),
          config))
    return error;
  if (llvm::Error error = requireDispatchConfigCapabilityProperty(
          kernel, capabilities, config.lmulCapabilityID, kLMULPropertyName,
          config.lmul, config))
    return error;
  if (llvm::Error error = requireDispatchConfigCapabilityProperty(
          kernel, capabilities, config.tailPolicyCapabilityID,
          kTailPolicyPropertyName, config.tailPolicy, config))
    return error;
  if (llvm::Error error = requireDispatchConfigCapabilityProperty(
          kernel, capabilities, config.maskPolicyCapabilityID,
          kMaskPolicyPropertyName, config.maskPolicy, config))
    return error;
  return llvm::Error::success();
}

llvm::Expected<const SelectedPlanMetadataEntry *>
findUniqueSelectedPlanMetadataEntry(const TargetArtifactCandidate &candidate,
                                    llvm::StringRef name) {
  const SelectedPlanMetadataEntry *match = nullptr;
  unsigned count = 0;
  for (const SelectedPlanMetadataEntry &metadata :
       candidate.selectedPlanMetadata) {
    if (metadata.name == name) {
      match = &metadata;
      ++count;
    }
  }

  if (count == 0)
    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("selected RVV dispatch candidate @") +
            candidate.selectedVariant + " requires selected_plan_metadata '" +
            name + "'");
  if (count > 1)
    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("selected RVV dispatch candidate @") +
            candidate.selectedVariant +
            " has duplicate selected_plan_metadata '" + name + "'");
  return match;
}

llvm::Error validateDispatchSelectedPlanMetadataEntry(
    const TargetArtifactCandidate &candidate,
    const tianchenrv::target::rvv::
        RVVVectorShapeSelectedPlanMetadataDescriptor &expected) {
  llvm::Expected<const SelectedPlanMetadataEntry *> metadata =
      findUniqueSelectedPlanMetadataEntry(candidate, expected.name);
  if (!metadata)
    return metadata.takeError();

  if ((*metadata)->value != expected.value)
    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("selected RVV dispatch candidate @") +
            candidate.selectedVariant + " selected_plan_metadata '" +
            expected.name + "' " + expected.diagnosticSpelling +
            " must be '" + expected.value + "'");
  if ((*metadata)->role != expected.role)
    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("selected RVV dispatch candidate @") +
            candidate.selectedVariant + " selected_plan_metadata '" +
            expected.name + "' role must be '" + expected.role + "'");
  if ((*metadata)->note != expected.note)
    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("selected RVV dispatch candidate @") +
            candidate.selectedVariant + " selected_plan_metadata '" +
            expected.name + "' note must be '" + expected.note + "'");
  return llvm::Error::success();
}

llvm::Error validateDispatchSelectedPlanMetadata(
    const TargetArtifactCandidate &candidate,
    const DispatchRVVVectorShapeConfig &config) {
  llvm::SmallVector<
      tianchenrv::target::rvv::RVVVectorShapeSelectedPlanMetadataDescriptor, 16>
      expected;
  tianchenrv::target::rvv::appendRVVVectorShapeSelectedPlanMetadata(config,
                                                                   expected);
  tianchenrv::target::rvv::appendRVVRuntimeVLBoundarySelectedPlanMetadata(
      expected);
  for (const auto &entry : expected)
    if (llvm::Error error =
            validateDispatchSelectedPlanMetadataEntry(candidate, entry))
      return error;
  return llvm::Error::success();
}

bool isRecognizedDispatchSelectedBinarySourceKind(llvm::StringRef value) {
  return value == "frontend-lowering" ||
         value == "default-i32-vadd-typed-body-materialization" ||
         value == "direct-typed-microkernel-body";
}

llvm::Expected<std::optional<std::string>>
resolveDispatchVariantSelectedBinarySourceKind(
    const TargetArtifactCandidate &candidate) {
  VariantOp variant = findDirectVariant(candidate.kernel,
                                        candidate.selectedVariant);
  if (!variant)
    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("selected RVV dispatch case variant @") +
            candidate.selectedVariant +
            " must resolve before selected-source identity validation");

  auto attr = variant->getAttrOfType<mlir::StringAttr>(
      tianchenrv::target::rvv::getRVVSelectedBinarySourceKindMetadataName());
  if (!attr)
    return std::optional<std::string>();

  llvm::StringRef sourceKind = attr.getValue().trim();
  if (sourceKind.empty())
    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("selected RVV dispatch case variant @") +
            candidate.selectedVariant +
            " selected binary source kind must be non-empty before "
            "RVV+scalar dispatch selected-source identity consumption");
  if (llvm::Error error = validateDispatchRuntimeABIText(
          candidate.kernel, "selected RVV binary source kind", sourceKind))
    return std::move(error);
  if (!isRecognizedDispatchSelectedBinarySourceKind(sourceKind))
    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("selected RVV dispatch case variant @") +
            candidate.selectedVariant +
            " has unsupported selected binary source kind '" + sourceKind +
            "' for RVV+scalar dispatch selected-source identity consumption");
  return std::optional<std::string>(sourceKind.str());
}

llvm::Error validateDispatchSelectedSourceIdentityMetadata(
    const TargetArtifactCandidate &candidate,
    const tianchenrv::target::rvv::RVVBinarySelectedConfigContract &contract) {
  llvm::Expected<std::optional<std::string>> variantSourceKind =
      resolveDispatchVariantSelectedBinarySourceKind(candidate);
  if (!variantSourceKind)
    return variantSourceKind.takeError();

  const SelectedPlanMetadataEntry *planSourceKind =
      findFirstSelectedPlanMetadataEntry(
          candidate,
          tianchenrv::target::rvv::
              getRVVSelectedBinarySourceKindMetadataName());
  const SelectedPlanMetadataEntry *planMicrokernelOp =
      findFirstSelectedPlanMetadataEntry(
          candidate,
          tianchenrv::target::rvv::
              getRVVSelectedBinaryMicrokernelOpMetadataName());

  if (!*variantSourceKind && !planSourceKind && !planMicrokernelOp)
    return llvm::Error::success();

  std::string expectedSourceKind;
  if (*variantSourceKind) {
    expectedSourceKind = **variantSourceKind;
  } else {
    llvm::Expected<const SelectedPlanMetadataEntry *> sourceKind =
        findUniqueSelectedPlanMetadataEntry(
            candidate,
            tianchenrv::target::rvv::
                getRVVSelectedBinarySourceKindMetadataName());
    if (!sourceKind)
      return sourceKind.takeError();
    llvm::StringRef value = (*sourceKind)->value;
    if (llvm::Error error = validateDispatchRuntimeABIText(
            candidate.kernel, "selected RVV binary source kind", value))
      return std::move(error);
    if (!isRecognizedDispatchSelectedBinarySourceKind(value))
      return makeDispatchError(
          candidate.kernel,
          llvm::Twine("selected RVV dispatch candidate @") +
              candidate.selectedVariant +
              " selected_plan_metadata '" +
              tianchenrv::target::rvv::
                  getRVVSelectedBinarySourceKindMetadataName() +
              "' has unsupported selected binary source kind '" + value + "'");
    expectedSourceKind = value.str();
  }

  llvm::SmallVector<
      tianchenrv::target::rvv::RVVVectorShapeSelectedPlanMetadataDescriptor, 2>
      expected;
  tianchenrv::target::rvv::appendRVVBinarySelectedSourceIdentityMetadata(
      contract, expectedSourceKind, expected);
  for (const auto &entry : expected)
    if (llvm::Error error =
            validateDispatchSelectedPlanMetadataEntry(candidate, entry))
      return error;
  return llvm::Error::success();
}

llvm::Error validateDispatchDescriptorElementCountMetadata(
    const TargetArtifactCandidate &candidate,
    const tianchenrv::target::rvv::RVVBinarySelectedConfigContract &contract) {
  llvm::Expected<const SelectedPlanMetadataEntry *> metadata =
      findUniqueSelectedPlanMetadataEntry(
          candidate,
          tianchenrv::target::rvv::getRVVDescriptorElementCountMetadataName());
  if (!metadata)
    return metadata.takeError();
  std::string expectedCount =
      std::to_string(contract.getDescriptorElementCount());
  if ((*metadata)->value != expectedCount)
    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("selected RVV dispatch candidate @") +
            candidate.selectedVariant + " selected_plan_metadata '" +
            tianchenrv::target::rvv::getRVVDescriptorElementCountMetadataName() +
            "' descriptor element count must be '" + expectedCount + "'");
  if ((*metadata)->role !=
      tianchenrv::target::rvv::
          getRVVDescriptorElementCountCapacityMetadataRole())
    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("selected RVV dispatch candidate @") +
            candidate.selectedVariant + " selected_plan_metadata '" +
            tianchenrv::target::rvv::getRVVDescriptorElementCountMetadataName() +
            "' role must be '" +
            tianchenrv::target::rvv::
                getRVVDescriptorElementCountCapacityMetadataRole() +
            "'");
  if ((*metadata)->note !=
      tianchenrv::target::rvv::
          getRVVDescriptorElementCountCapacityMetadataNote())
    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("selected RVV dispatch candidate @") +
            candidate.selectedVariant + " selected_plan_metadata '" +
            tianchenrv::target::rvv::getRVVDescriptorElementCountMetadataName() +
            "' note must be '" +
            tianchenrv::target::rvv::
                getRVVDescriptorElementCountCapacityMetadataNote() +
            "'");
  return llvm::Error::success();
}

llvm::Error validateDispatchSelectedConfigContractMetadata(
    const TargetArtifactCandidate &candidate,
    const tianchenrv::target::rvv::RVVBinarySelectedConfigContract &contract) {
  llvm::SmallVector<
      tianchenrv::target::rvv::RVVVectorShapeSelectedPlanMetadataDescriptor, 32>
      expected;
  tianchenrv::target::rvv::appendRVVBinarySelectedVectorShapeMetadata(
      contract, expected);
  tianchenrv::target::rvv::appendRVVBinaryRuntimeVLBoundarySelectedPlanMetadata(
      contract, expected);
  tianchenrv::target::rvv::
      appendRVVBinaryFixedVectorSourceExtentSelectedPlanMetadata(contract,
                                                                expected);
  tianchenrv::target::rvv::
      appendRVVBinaryDynamicRuntimeExtentSelectedPlanMetadata(contract,
                                                             expected);
  if (contract.getFamily().dtype ==
          tianchenrv::target::rvv::RVVBinaryDTypeKind::I32 ||
      contract.getFamily().dtype ==
          tianchenrv::target::rvv::RVVBinaryDTypeKind::I64)
    tianchenrv::target::rvv::appendRVVBinarySelectedTypedSourceMetadata(
        contract, expected);
  else
    tianchenrv::target::rvv::appendRVVBinaryLegacyDescriptorMirrorMetadata(
        contract, expected);
  if (contract.getFamily().dtype ==
          tianchenrv::target::rvv::RVVBinaryDTypeKind::I32 ||
      contract.getFamily().dtype ==
          tianchenrv::target::rvv::RVVBinaryDTypeKind::I64)
    tianchenrv::target::rvv::appendRVVBinaryEmitCRouteMetadata(contract,
                                                              expected);
  tianchenrv::target::rvv::appendRVVBinarySelectedConfigProfileMetadata(
      contract, expected);

  for (const auto &entry : expected)
    if (llvm::Error error =
            validateDispatchSelectedPlanMetadataEntry(candidate, entry))
      return error;
  if (llvm::Error error =
          validateDispatchSelectedSourceIdentityMetadata(candidate, contract))
    return error;
  return validateDispatchDescriptorElementCountMetadata(candidate, contract);
}

llvm::Expected<std::string>
buildDispatchSelectedSourceIdentityContractSummary(const DispatchPair &pair) {
  if (!pair.selectedConfig.isValid())
    return makeDispatchError(
        pair.rvv.kernel,
        "RVV+scalar dispatch artifact export requires a validated selected "
        "config contract before selected-source identity emission");

  llvm::Expected<const SelectedPlanMetadataEntry *> sourceKind =
      findUniqueSelectedPlanMetadataEntry(
          pair.rvv,
          tianchenrv::target::rvv::
              getRVVSelectedBinarySourceKindMetadataName());
  if (!sourceKind)
    return sourceKind.takeError();

  llvm::Expected<const SelectedPlanMetadataEntry *> microkernelOp =
      findUniqueSelectedPlanMetadataEntry(
          pair.rvv,
          tianchenrv::target::rvv::
              getRVVSelectedBinaryMicrokernelOpMetadataName());
  if (!microkernelOp)
    return microkernelOp.takeError();

  if (llvm::Error error =
          validateDispatchSelectedSourceIdentityMetadata(pair.rvv,
                                                         pair.selectedConfig))
    return std::move(error);

  return pair.selectedConfig.formatDispatchContractSelectedSourceIdentityMetadataValue(
      (*sourceKind)->value, (*microkernelOp)->value);
}

llvm::Error printDispatchSelectedSourceIdentityContract(
    llvm::raw_ostream &os, const DispatchPair &pair) {
  llvm::Expected<std::string> summary =
      buildDispatchSelectedSourceIdentityContractSummary(pair);
  if (!summary)
    return summary.takeError();

  os << "/* dispatch_selected_source_identity: " << *summary << " */\n";
  os << "/* dispatch_selected_source_identity_authority: "
        "RVVScalarDispatch.cpp consumed selected RVV source identity from "
        "selected-plan metadata and the direct selected config before "
        "dispatch source/header/object artifact export. */\n";
  os << "/* dispatch_embedded_rvv_artifact_contract_consumed: "
        "selected_source_identity=rvv_microkernel_selected_source_identity, "
        "runtime_abi_invocation_contract=runtime_abi_invocation_contract, "
        "runtime_length=rvv_microkernel_runtime_length_contract, "
        "production_owner=rvv-target-export */\n";
  return llvm::Error::success();
}

llvm::Expected<const DispatchRVVVectorShapeConfig *>
resolveDispatchPairSelectedVectorShape(const DispatchPair &pair) {
  KernelOp kernel = pair.rvv.kernel;
  VariantOp rvvVariant = findDirectVariant(kernel, pair.rvv.selectedVariant);
  if (!rvvVariant)
    return makeDispatchError(
        kernel, llvm::Twine("selected RVV dispatch case variant @") +
                    pair.rvv.selectedVariant +
                    " must resolve before selected vector-shape validation");

  llvm::Expected<TargetCapabilitySet> capabilities =
      TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities)
    return capabilities.takeError();

  llvm::SmallVector<const DispatchRVVVectorShapeConfig *, 2> requiredConfigs;
  for (const DispatchRVVVectorShapeConfig *config :
       tianchenrv::target::rvv::getRVVBinaryFamilyShapeConfigs(
           *pair.family->rvvFamily)) {
    llvm::Expected<bool> required =
        variantRequiresVectorShapeConfig(kernel, rvvVariant, *capabilities,
                                         *config);
    if (!required)
      return required.takeError();
    if (*required)
      requiredConfigs.push_back(config);
  }

  if (requiredConfigs.empty())
    return makeDispatchError(
        kernel, llvm::Twine("selected RVV dispatch case variant @") +
                    pair.rvv.selectedVariant +
                    " must require exactly one finite RVV vector-shape "
                    "config capability set for family '" +
                    pair.family->diagnosticName + "'");
  if (requiredConfigs.size() > 1)
    return makeDispatchError(
        kernel, llvm::Twine("selected RVV dispatch case variant @") +
                    pair.rvv.selectedVariant +
                    " must require exactly one finite RVV vector-shape "
                    "config for family '" +
                    pair.family->diagnosticName + "'");

  const DispatchRVVVectorShapeConfig *selectedConfig = requiredConfigs.front();
  if (llvm::Error error = validateDispatchConfigCapabilityProperties(
          kernel, *capabilities, *selectedConfig))
    return std::move(error);

  llvm::Expected<const SelectedPlanMetadataEntry *> shapeMetadata =
      findUniqueSelectedPlanMetadataEntry(
          pair.rvv, tianchenrv::target::rvv::getRVVSelectedVectorShapeAttrName());
  if (!shapeMetadata)
    return shapeMetadata.takeError();
  const DispatchRVVVectorShapeConfig *metadataConfig =
      tianchenrv::target::rvv::lookupRVVBinaryFamilyShapeConfigByID(
          *pair.family->rvvFamily,
          (*shapeMetadata)->value);
  if (!metadataConfig)
    return makeDispatchError(
        kernel, llvm::Twine("selected RVV dispatch candidate @") +
                    pair.rvv.selectedVariant +
                    " selected_plan_metadata 'tcrv_rvv.selected_vector_shape' "
                    "has unsupported finite shape '" +
                    (*shapeMetadata)->value + "'");
  if (metadataConfig != selectedConfig)
    return makeDispatchError(
        kernel, llvm::Twine("selected RVV dispatch candidate @") +
                    pair.rvv.selectedVariant +
                    " selected_plan_metadata shape '" +
                    (*shapeMetadata)->value +
                    "' does not match selected variant capability shape '" +
                    selectedConfig->diagnosticSpelling + "'");

  if (llvm::Error error =
          validateDispatchSelectedPlanMetadata(pair.rvv, *selectedConfig))
    return std::move(error);
  return selectedConfig;
}

llvm::Expected<tianchenrv::target::rvv::RVVBinarySelectedConfigContract>
buildDispatchPairSelectedConfigContract(
    const DispatchPair &pair, const DispatchRVVVectorShapeConfig &shape) {
  KernelOp kernel = pair.rvv.kernel;
  mlir::ModuleOp module = kernel ? kernel->getParentOfType<mlir::ModuleOp>()
                                 : mlir::ModuleOp();
  if (!module)
    return makeDispatchError(
        kernel,
        "selected RVV dispatch case requires an enclosing module before "
        "direct RVV selected config contract consumption");

  llvm::StringRef runtimeElementCountCName;
  llvm::StringRef dispatchGuardCName;
  unsigned runtimeElementCountCount = 0;
  unsigned dispatchGuardCount = 0;
  for (const support::RuntimeABIParameter &parameter :
       pair.abiPlan.parameters) {
    if (parameter.role == support::RuntimeABIParameterRole::RuntimeElementCount) {
      runtimeElementCountCName = parameter.cName;
      ++runtimeElementCountCount;
    }
    if (parameter.role ==
        support::RuntimeABIParameterRole::DispatchAvailabilityGuard) {
      dispatchGuardCName = parameter.cName;
      ++dispatchGuardCount;
    }
  }
  if (runtimeElementCountCount != 1 || dispatchGuardCount != 1)
    return makeDispatchError(
        kernel,
        llvm::Twine("selected RVV dispatch case @") +
            pair.rvv.selectedVariant +
            " requires exactly one runtime-element-count and one "
            "dispatch-availability-guard parameter in the selected config "
            "contract");

  llvm::Expected<tianchenrv::target::rvv::RVVBinarySelectedConfigContract>
      directContract =
          tianchenrv::target::rvv::
              resolveRVVMicrokernelSelectedConfigContractAuthority(
                  kernel, *pair.family->rvvFamily, pair.rvv.selectedVariant,
                  pair.rvv.role, pair.family->rvvRouteID);
  if (!directContract)
    return directContract.takeError();

  if (directContract->getFamilyID() != pair.family->rvvFamily->familyID)
    return makeDispatchError(
        kernel, llvm::Twine("selected RVV dispatch case candidate @") +
                    pair.rvv.selectedVariant +
                    " direct selected config family '" +
                    directContract->getFamilyID() +
                    "' does not match dispatch family '" +
                    pair.family->rvvFamily->familyID + "'");
  if (directContract->getShapeID() != shape.shapeID)
    return makeDispatchError(
        kernel, llvm::Twine("selected RVV dispatch case candidate @") +
                    pair.rvv.selectedVariant +
                    " direct selected config shape '" +
                    directContract->getShapeID() +
                    "' does not match dispatch capability shape '" +
                    shape.shapeID + "'");
  if (directContract->getDescriptorElementCount() <= 0)
    return makeDispatchError(
        kernel, llvm::Twine("selected RVV dispatch case candidate @") +
                    pair.rvv.selectedVariant +
                    " direct selected config contract requires descriptor-local "
                    "component capacity before dispatch artifact export");
  if (directContract->getRuntimeElementCountCName() !=
      runtimeElementCountCName)
    return makeDispatchError(
        kernel, llvm::Twine("selected RVV dispatch case candidate @") +
                    pair.rvv.selectedVariant +
                    " direct selected config runtime_element_count_c_name '" +
                    directContract->getRuntimeElementCountCName() +
                    "' does not match dispatch runtime element-count ABI "
                    "parameter c_name '" +
                    runtimeElementCountCName + "'");

  return tianchenrv::target::rvv::buildRVVBinarySelectedConfigContract(
      *pair.family->rvvFamily, shape, pair.rvv.selectedVariant, pair.rvv.role,
      directContract->getDescriptorElementCount(), runtimeElementCountCName,
      dispatchGuardCName,
      directContract->getFixedVectorSourceExtentContract(),
      directContract->getDynamicVectorRuntimeExtentContract());
}

llvm::Error requireEmbeddedRVVSourceSnippet(const DispatchPair &pair,
                                            llvm::StringRef rvvSource,
                                            llvm::StringRef snippet,
                                            llvm::StringRef context) {
  if (rvvSource.contains(snippet))
    return llvm::Error::success();
  return makeDispatchError(
      pair.rvv.kernel,
      llvm::Twine("embedded selected RVV source for dispatch candidate @") +
          pair.rvv.selectedVariant + " is inconsistent with target-owned " +
          "selected vector-shape descriptor: missing " + context + " '" +
          snippet + "'");
}

llvm::Error validateEmbeddedRVVSourceArtifactContract(
    const DispatchPair &pair, llvm::StringRef rvvSource) {
  llvm::Expected<std::string> selectedSourceIdentity =
      buildDispatchSelectedSourceIdentityContractSummary(pair);
  if (!selectedSourceIdentity)
    return selectedSourceIdentity.takeError();

  std::string selectedSourceIdentitySnippet;
  {
    llvm::raw_string_ostream stream(selectedSourceIdentitySnippet);
    stream << "rvv_microkernel_selected_source_identity: "
           << *selectedSourceIdentity;
  }
  if (llvm::Error error = requireEmbeddedRVVSourceSnippet(
          pair, rvvSource, selectedSourceIdentitySnippet,
          "RVVMicrokernel exported selected source identity contract"))
    return error;

  if (!pair.family || !pair.family->rvvFamily)
    return makeDispatchError(
        pair.rvv.kernel,
        "embedded RVV artifact invocation contract validation requires an "
        "exact selected RVV binary family");
  llvm::Expected<support::RuntimeABIInvocationContract> invocationContract =
      support::buildRuntimeABIInvocationContract(
          pair.rvv.kernel, pair.family->rvvFamily->familyID,
          pair.rvv.runtimeABIParameters, "RVVMicrokernel.cpp",
          makeRVVFunctionName(pair), pair.rvv.runtimeABIKind,
          pair.rvv.runtimeABIName, pair.rvv.runtimeGlueRole,
          pair.selectedConfig.getRuntimeElementCountCName(),
          "rvv-target-export");
  if (!invocationContract)
    return invocationContract.takeError();
  std::string invocationContractSnippet =
      support::formatRuntimeABIInvocationContractCommentBody(
          "runtime_abi_invocation_contract", *invocationContract);
  if (llvm::Error error = requireEmbeddedRVVSourceSnippet(
          pair, rvvSource, invocationContractSnippet,
          "RVVMicrokernel exported runtime ABI invocation contract"))
    return error;

  if (auto sourceExtent =
          pair.selectedConfig.getFixedVectorSourceExtentContract()) {
    if (llvm::Error error = requireEmbeddedRVVSourceSnippet(
            pair, rvvSource, sourceExtent->formatCommentBody(),
            "RVVMicrokernel fixed vector source extent contract"))
      return error;
    std::string countConstraintSnippet;
    llvm::raw_string_ostream stream(countConstraintSnippet);
    stream << "runtime_element_count_constraint: "
           << pair.selectedConfig.getRuntimeElementCountCName()
           << " must equal fixed source vector extent "
           << sourceExtent->sourceVectorExtent
           << " before runtime AVL/VL execution";
    stream.flush();
    if (llvm::Error error = requireEmbeddedRVVSourceSnippet(
            pair, rvvSource, countConstraintSnippet,
            "RVVMicrokernel fixed runtime element-count contract"))
      return error;
  }

  if (auto runtimeExtent =
          pair.selectedConfig.getDynamicVectorRuntimeExtentContract()) {
    if (llvm::Error error = requireEmbeddedRVVSourceSnippet(
            pair, rvvSource, runtimeExtent->formatCommentBody(),
            "RVVMicrokernel dynamic runtime AVL authority contract"))
      return error;
    std::string countSourceSnippet;
    llvm::raw_string_ostream stream(countSourceSnippet);
    stream << "runtime_element_count_source: "
           << pair.selectedConfig.getRuntimeElementCountCName()
           << " is the source scf.for upper bound and runtime AVL; no fixed "
              "source-extent trap is emitted for this dynamic vector route";
    stream.flush();
    if (llvm::Error error = requireEmbeddedRVVSourceSnippet(
            pair, rvvSource, countSourceSnippet,
            "RVVMicrokernel dynamic runtime element-count authority"))
      return error;
  }

  return llvm::Error::success();
}

llvm::Error validateEmbeddedRVVSourceSelectedShape(
    const DispatchPair &pair, llvm::StringRef rvvSource) {
  if (!pair.selectedConfig.isValid())
    return makeDispatchError(
        pair.rvv.kernel,
        "selected RVV dispatch pair must resolve a target-owned selected "
        "config contract before embedded RVV source validation");

  const DispatchRVVVectorShapeConfig &shape = pair.selectedConfig.getShape();
  RVVI32BinaryIntrinsicDescriptor descriptor =
      pair.selectedConfig.getIntrinsicDescriptor();

  std::string shapeComment =
      descriptor.formatSelectedVectorShapeConfigCommentBody();
  std::string intrinsicConfig =
      descriptor.formatIntrinsicConfigCommentBody();
  std::string setvlIntrinsic = descriptor.getSetVLIntrinsicName();
  std::string loadIntrinsic = descriptor.getLoadIntrinsicName();
  std::string arithmeticIntrinsic = descriptor.getArithmeticIntrinsicName();
  std::string storeIntrinsic = descriptor.getStoreIntrinsicName();

  if (llvm::Error error = requireEmbeddedRVVSourceSnippet(
          pair, rvvSource, shapeComment, "selected shape config comment"))
    return error;
  if (llvm::Error error = requireEmbeddedRVVSourceSnippet(
          pair, rvvSource, shape.sewCapabilityID,
          "selected shape SEW capability id"))
    return error;
  if (llvm::Error error = requireEmbeddedRVVSourceSnippet(
          pair, rvvSource, shape.lmulCapabilityID,
          "selected shape LMUL capability id"))
    return error;
  if (llvm::Error error = requireEmbeddedRVVSourceSnippet(
          pair, rvvSource, shape.tailPolicyCapabilityID,
          "selected shape tail-policy capability id"))
    return error;
  if (llvm::Error error = requireEmbeddedRVVSourceSnippet(
          pair, rvvSource, shape.maskPolicyCapabilityID,
          "selected shape mask-policy capability id"))
    return error;
  if (llvm::Error error = requireEmbeddedRVVSourceSnippet(
          pair, rvvSource, intrinsicConfig, "intrinsic config comment"))
    return error;
  if (llvm::Error error = requireEmbeddedRVVSourceSnippet(
          pair, rvvSource, setvlIntrinsic, "vsetvl intrinsic"))
    return error;
  if (llvm::Error error = requireEmbeddedRVVSourceSnippet(
          pair, rvvSource, loadIntrinsic, "load intrinsic"))
    return error;
  if (llvm::Error error = requireEmbeddedRVVSourceSnippet(
          pair, rvvSource, arithmeticIntrinsic, "arithmetic intrinsic"))
    return error;
  if (llvm::Error error = requireEmbeddedRVVSourceSnippet(
          pair, rvvSource, storeIntrinsic, "store intrinsic"))
    return error;
  if (llvm::Error error =
          validateEmbeddedRVVSourceArtifactContract(pair, rvvSource))
    return error;
  return llvm::Error::success();
}

llvm::Error requireEmbeddedScalarSourceSnippet(const DispatchPair &pair,
                                               llvm::StringRef scalarSource,
                                               llvm::StringRef snippet,
                                               llvm::StringRef context) {
  if (scalarSource.contains(snippet))
    return llvm::Error::success();
  return makeDispatchError(
      pair.scalar.kernel,
      llvm::Twine("embedded selected scalar source for dispatch fallback @") +
          pair.scalar.selectedVariant +
          " is inconsistent with target-owned scalar component authority: "
          "missing " +
          context + " '" + snippet + "'");
}

const tianchenrv::target::rvv_scalar::RVVScalarBinaryFamilyDescriptor *
lookupDispatchPairFamilyRegistration(const DispatchI32FamilySpec &family) {
  for (const auto *descriptor :
       tianchenrv::target::rvv_scalar::getRVVScalarBinaryRegistrationRecords())
    if (isSameDispatchFamily(descriptor->dispatch, family))
      return descriptor;
  return nullptr;
}

llvm::Error validateEmbeddedScalarSourceAuthority(
    const DispatchPair &pair, llvm::StringRef scalarSource) {
  const auto *familyDescriptor =
      lookupDispatchPairFamilyRegistration(*pair.family);
  if (!familyDescriptor)
    return makeDispatchError(
        pair.scalar.kernel,
        llvm::Twine("selected scalar dispatch fallback @") +
            pair.scalar.selectedVariant +
            " requires a manifest-backed scalar component family descriptor");

  std::string expectedFunction = makeScalarFunctionName(pair);
  if (llvm::Error error = requireEmbeddedScalarSourceSnippet(
          pair, scalarSource, expectedFunction, "microkernel function"))
    return error;
  if (llvm::Error error = requireEmbeddedScalarSourceSnippet(
          pair, scalarSource, familyDescriptor->scalar.microkernelOpName,
          "typed scalar microkernel op"))
    return error;
  if (llvm::Error error = requireEmbeddedScalarSourceSnippet(
          pair, scalarSource, pair.scalar.selectedVariant,
          "selected scalar variant"))
    return error;
  if (llvm::Error error = requireEmbeddedScalarSourceSnippet(
          pair, scalarSource, pair.scalar.role, "selected scalar role"))
    return error;
  if (llvm::Error error = requireEmbeddedScalarSourceSnippet(
          pair, scalarSource, familyDescriptor->scalar.runtimeABIKind,
          "runtime ABI kind"))
    return error;
  if (llvm::Error error = requireEmbeddedScalarSourceSnippet(
          pair, scalarSource, familyDescriptor->scalar.runtimeABIName,
          "runtime ABI name"))
    return error;
  if (llvm::Error error = requireEmbeddedScalarSourceSnippet(
          pair, scalarSource, familyDescriptor->scalar.runtimeGlueRole,
          "runtime glue role"))
    return error;
  return llvm::Error::success();
}

llvm::Error wrapComponentAuthorityError(const DispatchPair &pair,
                                        llvm::StringRef component,
                                        llvm::Error error) {
  std::string message = llvm::toString(std::move(error));
  return makeDispatchError(
      pair.rvv.kernel,
      llvm::Twine("selected ") + component +
          " component body authority failed before RVV+scalar dispatch "
          "artifact emission: " +
          message);
}

llvm::Error validateDispatchPairComponentAuthorities(const DispatchPair &pair) {
  mlir::ModuleOp module = pair.rvv.kernel->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return makeDispatchError(
        pair.rvv.kernel,
        "requires enclosing builtin.module before component body authority "
        "validation");

  if (llvm::Error error =
          rvv::validateRVVMicrokernelSourceAuthority(
              module, *pair.family->rvvFamily, pair.rvv.selectedVariant,
              pair.rvv.role, pair.family->rvvRouteID))
    return wrapComponentAuthorityError(pair, "RVV dispatch case",
                                       std::move(error));

  const auto *familyDescriptor =
      lookupDispatchPairFamilyRegistration(*pair.family);
  if (!familyDescriptor)
    return makeDispatchError(
        pair.scalar.kernel,
        llvm::Twine("selected scalar dispatch fallback @") +
            pair.scalar.selectedVariant +
            " requires a manifest-backed scalar component family descriptor");

  if (llvm::Error error =
          scalar::validateScalarMicrokernelSourceAuthority(
              module, familyDescriptor->scalar, pair.scalar.selectedVariant,
              pair.scalar.role))
    return wrapComponentAuthorityError(pair, "scalar dispatch fallback",
                                       std::move(error));

  return llvm::Error::success();
}

llvm::Error requireSafeCompileStringAttr(KernelOp kernel, mlir::Operation *op,
                                         llvm::StringRef attrName,
                                         llvm::StringRef context,
                                         std::string &out) {
  auto attr = op ? op->getAttrOfType<mlir::StringAttr>(attrName)
                 : mlir::StringAttr();
  if (!attr || attr.getValue().trim().empty())
    return makeDispatchObjectError(
        kernel, llvm::Twine(context) + " requires non-empty string attribute '" +
                    attrName + "'");
  llvm::StringRef value = attr.getValue().trim();
  if (llvm::Error error = validateBoundedCompileText(kernel, attrName, value))
    return error;
  out = value.str();
  return llvm::Error::success();
}

llvm::Error mergeOptionalMABI(KernelOp kernel, llvm::StringRef fieldName,
                              llvm::StringRef value,
                              std::optional<std::string> &out) {
  value = value.trim();
  if (value.empty())
    return llvm::Error::success();
  if (llvm::Error error = validateBoundedCompileText(kernel, fieldName, value))
    return error;
  if (out && *out != value)
    return makeDispatchObjectError(
        kernel, "conflicting preserved selected_mabi capability metadata");
  out = value.str();
  return llvm::Error::success();
}

llvm::Expected<DispatchObjectCompileConfig>
buildDispatchObjectCompileConfig(const DispatchPair &pair) {
  KernelOp kernel = pair.rvv.kernel;
  VariantOp rvvVariant = findDirectVariant(kernel, pair.rvv.selectedVariant);
  if (!rvvVariant)
    return makeDispatchObjectError(
        kernel, llvm::Twine("selected RVV dispatch case variant @") +
                    pair.rvv.selectedVariant +
                    " must resolve before object compilation");

  llvm::Expected<TargetCapabilitySet> capabilities =
      TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities)
    return capabilities.takeError();

  const CapabilityDescriptor *rvvCapability =
      capabilities->lookupProviderByID(kRVVCapabilityID);
  if (!rvvCapability || !rvvCapability->isAvailable())
    return makeDispatchObjectError(
        kernel,
        "selected RVV dispatch case requires an available RVV capability "
        "provider before object compilation");
  llvm::StringRef architecture =
      rvvCapability->getProperty(kArchitecturePropertyName).trim();
  if (llvm::Error error = validateBoundedCompileText(
          kernel, kArchitecturePropertyName, architecture))
    return std::move(error);
  if (architecture != "riscv64")
    return makeDispatchObjectError(
        kernel, llvm::Twine("selected RVV dispatch case requires "
                            "architecture capability metadata 'riscv64', "
                            "got '") +
                    architecture + "'");

  std::string requiredMarch;
  if (llvm::Error error = requireSafeCompileStringAttr(
          kernel, rvvVariant.getOperation(), kRVVRequiredMarchAttrName,
          "selected RVV dispatch case variant", requiredMarch))
    return std::move(error);
  if (!hasRVVVectorHint(requiredMarch))
    return makeDispatchObjectError(
        kernel, llvm::Twine("selected RVV dispatch case variant @") +
                    pair.rvv.selectedVariant +
                    " attribute 'tcrv_rvv.required_march' must contain RVV "
                    "vector evidence");

  llvm::SmallVector<std::string, 2> preservedMarches;
  if (const CapabilityDescriptor *compileRun =
          capabilities->lookupProviderByID(kRVVProbeCompileRunCapabilityID)) {
    if (compileRun->isAvailable()) {
      llvm::StringRef value =
          compileRun->getProperty(kSelectedMarchPropertyName).trim();
      if (!value.empty()) {
        if (llvm::Error error = validateBoundedCompileText(
                kernel, kSelectedMarchPropertyName, value))
          return std::move(error);
        preservedMarches.push_back(value.str());
      }
    }
  }

  if (const CapabilityDescriptor *march =
          capabilities->lookupProviderByID(kRVVToolchainMarchCapabilityID)) {
    if (march->isAvailable()) {
      llvm::StringRef value = march->getProperty(kValuePropertyName).trim();
      if (!value.empty()) {
        if (llvm::Error error =
                validateBoundedCompileText(kernel, kValuePropertyName, value))
          return std::move(error);
        preservedMarches.push_back(value.str());
      }
    }
  }

  if (preservedMarches.empty())
    return makeDispatchObjectError(
        kernel, "selected RVV dispatch case requires available preserved "
                "selected_march metadata from capability id "
                "'rvv.probe.compile_run' or 'rvv.toolchain.march'");
  if (!llvm::is_contained(preservedMarches, requiredMarch))
    return makeDispatchObjectError(
        kernel, llvm::Twine("selected RVV dispatch case variant @") +
                    pair.rvv.selectedVariant +
                    " 'tcrv_rvv.required_march' metadata is not satisfied by "
                    "preserved selected_march capability metadata");

  std::optional<std::string> selectedMABI;
  if (const CapabilityDescriptor *compileRun =
          capabilities->lookupProviderByID(kRVVProbeCompileRunCapabilityID)) {
    if (compileRun->isAvailable())
      if (llvm::Error error =
              mergeOptionalMABI(kernel, kSelectedMABIPropertyName,
                                compileRun->getProperty(
                                    kSelectedMABIPropertyName),
                                selectedMABI))
        return std::move(error);
  }

  if (const CapabilityDescriptor *mabi =
          capabilities->lookupProviderByID(kRVVToolchainMABICapabilityID)) {
    if (mabi->isAvailable())
      if (llvm::Error error =
              mergeOptionalMABI(kernel, kValuePropertyName,
                                mabi->getProperty(kValuePropertyName),
                                selectedMABI))
        return std::move(error);
  }

  DispatchObjectCompileConfig config;
  config.targetTriple = architecture.str();
  config.selectedMarch = std::move(requiredMarch);
  config.selectedMABI = std::move(selectedMABI);
  return config;
}

void printRequiredCapabilitiesComment(llvm::raw_ostream &os,
                                      llvm::StringRef label,
                                      KernelOp kernel,
                                      llvm::StringRef variantSymbol) {
  os << "/* " << label << "_required_capabilities:";
  VariantOp variant = findDirectVariant(kernel, variantSymbol);
  if (variant) {
    if (auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires")) {
      for (mlir::Attribute attr : requires) {
        auto symbol = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
        if (symbol)
          os << " @" << symbol.getValue();
      }
    }
  }
  os << " */\n";
}

void printCandidateMetadata(llvm::raw_ostream &os, llvm::StringRef label,
                            const TargetArtifactCandidate &candidate) {
  os << "/* " << label << "_selected_variant: @"
     << candidate.selectedVariant << " */\n";
  os << "/* " << label << "_selected_role: " << candidate.role << " */\n";
  os << "/* " << label << "_origin: " << candidate.origin << " */\n";
  os << "/* " << label << "_lowering_boundary: "
     << candidate.loweringBoundary << " */\n";
  os << "/* " << label << "_emission_kind: " << candidate.emissionKind
     << " */\n";
  os << "/* " << label << "_artifact_kind: " << candidate.artifactKind
     << " */\n";
  os << "/* " << label << "_artifact_route_id: " << candidate.routeID
     << " */\n";
  os << "/* " << label << "_runtime_abi: " << candidate.runtimeABI
     << " */\n";
  os << "/* " << label << "_runtime_abi_kind: "
     << candidate.runtimeABIKind << " */\n";
  os << "/* " << label << "_runtime_abi_name: "
     << candidate.runtimeABIName << " */\n";
  os << "/* " << label << "_runtime_glue_role: "
     << candidate.runtimeGlueRole << " */\n";
  for (auto [index, parameter] :
       llvm::enumerate(candidate.runtimeABIParameters)) {
    os << "/* " << label << "_runtime_abi_parameter[" << index
       << "]: c_name=" << parameter.cName << ", c_type=" << parameter.cType
       << ", role="
       << support::stringifyRuntimeABIParameterRole(parameter.role)
       << ", ownership="
       << support::stringifyRuntimeABIParameterOwnership(parameter.ownership)
       << " */\n";
  }
  for (auto [index, metadata] :
       llvm::enumerate(candidate.selectedPlanMetadata)) {
    os << "/* " << label << "_selected_plan_metadata[" << index
       << "]: name=" << metadata.name << ", value=" << metadata.value
       << ", role=" << metadata.role << ", note=" << metadata.note
       << " */\n";
  }
  printRequiredCapabilitiesComment(os, label, candidate.kernel,
                                   candidate.selectedVariant);
}

llvm::StringRef getMemWindowStringAttr(tcrv::exec::MemWindowOp window,
                                       llvm::StringRef attrName) {
  auto attr = window->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr)
    return {};
  return attr.getValue();
}

void printDispatchMemWindowMetadata(
    llvm::raw_ostream &os,
    llvm::ArrayRef<tcrv::exec::MemWindowOp> bufferWindows) {
  for (auto [index, window] : llvm::enumerate(bufferWindows)) {
    os << "/* dispatch_mem_window[" << index << "]: symbol=@"
       << getMemWindowStringAttr(window, "sym_name") << ", abi_role="
       << getMemWindowStringAttr(window, support::kMemWindowABIRoleAttrName)
       << ", access="
       << getMemWindowStringAttr(window, support::kMemWindowAccessAttrName)
       << ", ownership="
       << getMemWindowStringAttr(window, support::kMemWindowOwnershipAttrName)
       << ", c_type="
       << getMemWindowStringAttr(window, support::kMemWindowCTypeAttrName)
       << ", purpose="
       << getMemWindowStringAttr(window, support::kMemWindowPurposeAttrName)
       << ", binding="
       << getMemWindowStringAttr(window, support::kMemWindowBindingAttrName)
       << ", memory_space="
       << getMemWindowStringAttr(window, support::kMemWindowMemorySpaceAttrName)
       << " */\n";
  }
}

void printDispatchRuntimeParamMetadata(
    llvm::raw_ostream &os, llvm::ArrayRef<RuntimeParamOp> runtimeParams) {
  for (auto [index, param] : llvm::enumerate(runtimeParams)) {
    os << "/* dispatch_runtime_param[" << index << "]: symbol=@"
       << getRuntimeParamStringAttr(param, "sym_name") << ", abi_role="
       << getRuntimeParamStringAttr(
              param, support::kRuntimeParamABIRoleAttrName)
       << ", c_name="
       << getRuntimeParamStringAttr(param, support::kRuntimeParamCNameAttrName)
       << ", c_type="
       << getRuntimeParamStringAttr(param, support::kRuntimeParamCTypeAttrName)
       << ", ownership="
       << getRuntimeParamStringAttr(
              param, support::kRuntimeParamOwnershipAttrName)
       << ", purpose="
       << getRuntimeParamStringAttr(
              param, support::kRuntimeParamPurposeAttrName)
       << " */\n";
  }
}

void printDispatchRuntimeCallableABI(
    llvm::raw_ostream &os, llvm::StringRef dispatcherFunctionName,
    llvm::ArrayRef<support::RuntimeABIParameter> parameters) {
  os << "/* dispatch_runtime_callable_abi: void " << dispatcherFunctionName
     << "(";
  for (auto [index, parameter] : llvm::enumerate(parameters)) {
    if (index != 0)
      os << ", ";
    support::printRuntimeABIParameterCDeclaration(os, parameter);
  }
  os << ") */\n";
}

llvm::Error printDispatchRuntimeABIInvocationContract(
    llvm::raw_ostream &os, const DispatchPair &pair,
    llvm::StringRef dispatcherFunctionName,
    llvm::ArrayRef<support::RuntimeABIParameter> parameters,
    llvm::StringRef runtimeElementCountCName,
    llvm::StringRef dispatchGuardCName) {
  if (!pair.family || !pair.family->rvvFamily)
    return makeDispatchError(
        pair.rvv.kernel,
        "dispatch runtime ABI invocation contract requires an exact selected "
        "RVV binary family");

  llvm::Expected<support::RuntimeABIInvocationContract> contract =
      support::buildRuntimeABIInvocationContract(
          pair.rvv.kernel, pair.family->rvvFamily->familyID, parameters,
          "RVVScalarDispatch.cpp", dispatcherFunctionName,
          pair.composite.runtimeABIKind, pair.composite.runtimeABIName,
          /*runtimeGlueRole=*/"", runtimeElementCountCName,
          kDispatchTargetOwner, dispatchGuardCName);
  if (!contract)
    return contract.takeError();

  os << "/* "
     << support::formatRuntimeABIInvocationContractCommentBody(
            "dispatch_runtime_abi_invocation_contract", *contract)
     << " */\n";
  return llvm::Error::success();
}

llvm::Error buildEmbeddedCallableSources(const DispatchPair &pair,
                                         std::string &rvvSource,
                                         std::string &scalarSource) {
  mlir::ModuleOp module = pair.rvv.kernel->getParentOfType<mlir::ModuleOp>();
  if (llvm::Error error = validateDispatchPairComponentAuthorities(pair))
    return error;

  if (!pair.family || !pair.family->rvvFamily)
    return makeModuleDispatchError("RVV dispatch embedding requires an exact "
                                   "selected RVV binary family");

  llvm::raw_string_ostream rvvStream(rvvSource);
  if (llvm::Error error = rvv::exportRVVMicrokernelCForBinaryFamily(
          module, *pair.family->rvvFamily, rvvStream))
    return error;
  rvvStream.flush();
  if (llvm::Error error =
          validateEmbeddedRVVSourceSelectedShape(pair, rvvSource))
    return error;

  llvm::raw_string_ostream scalarStream(scalarSource);
  if (llvm::Error error = scalar::exportScalarMicrokernelC(module, scalarStream))
    return error;
  scalarStream.flush();
  if (llvm::Error error =
          validateEmbeddedScalarSourceAuthority(pair, scalarSource))
    return error;
  return llvm::Error::success();
}

TCRVEmitCCallOpaqueStep buildDispatchComponentCallStep(
    llvm::StringRef sourceOpName, llvm::StringRef sourceRole,
    llvm::StringRef callee,
    const DispatchRuntimeABIParameterBindings &bindings) {
  TCRVEmitCCallOpaqueStep step;
  step.sourceOp.opName = sourceOpName.str();
  step.sourceOp.role = sourceRole.str();
  step.callee = callee.str();
  const support::RuntimeABIParameter *callableParameters[] = {
      bindings.lhs, bindings.rhs, bindings.out, bindings.runtimeElementCount};
  for (const support::RuntimeABIParameter *parameter : callableParameters)
    step.operands.push_back({parameter->cName, parameter->cType});
  return step;
}

llvm::Expected<TCRVEmitCLowerableRoute> buildDispatchControlEmitCRoute(
    const DispatchPair &pair, const DispatchRouteManifestEntry &route,
    llvm::StringRef rvvFunctionName, llvm::StringRef scalarFunctionName,
    const DispatchRuntimeABIParameterBindings &bindings) {
  TCRVEmitCLowerableRoute emitcRoute(
      route.routeID, "tcrv-exec-dispatch-control-to-emitc-call-opaque");
  emitcRoute.addHeader("stddef.h");
  emitcRoute.addHeader("stdint.h");
  for (const support::RuntimeABIParameter &parameter : pair.abiPlan.parameters)
    emitcRoute.addABIValueMapping(parameter, parameter.cName);

  emitcRoute.addCallOpaqueStep(buildDispatchComponentCallStep(
      "tcrv.exec.case", "dispatch-case-call", rvvFunctionName, bindings));
  emitcRoute.addCallOpaqueStep(buildDispatchComponentCallStep(
      "tcrv.exec.fallback", "dispatch-fallback-call", scalarFunctionName,
      bindings));
  if (llvm::Error error = emitcRoute.verify())
    return std::move(error);
  return emitcRoute;
}

class DispatchControlEmitCLowerable final : public TCRVEmitCLowerableInterface {
public:
  DispatchControlEmitCLowerable(
      const DispatchPair &pair, const DispatchRouteManifestEntry &route,
      llvm::StringRef rvvFunctionName, llvm::StringRef scalarFunctionName,
      const DispatchRuntimeABIParameterBindings &bindings)
      : pair(pair), route(route), rvvFunctionName(rvvFunctionName.str()),
        scalarFunctionName(scalarFunctionName.str()), bindings(bindings) {}

  llvm::Expected<TCRVEmitCLowerableRoute>
  buildEmitCLowerableRoute() const override {
    return buildDispatchControlEmitCRoute(pair, route, rvvFunctionName,
                                          scalarFunctionName, bindings);
  }

private:
  const DispatchPair &pair;
  const DispatchRouteManifestEntry &route;
  std::string rvvFunctionName;
  std::string scalarFunctionName;
  const DispatchRuntimeABIParameterBindings &bindings;
};

void printDispatchEmitCRouteMetadata(llvm::raw_ostream &os,
                                     const TCRVEmitCLowerableRoute &route,
                                     llvm::StringRef dispatcherFunctionName,
                                     llvm::StringRef guardValueName) {
  os << "/* dispatch_emitc_route: tcrv.exec.dispatch -> emitc.if -> "
        "selected callable artifacts */\n";
  os << "/* dispatch_emitc_lowerable_interface: "
        "TCRVEmitCLowerableInterface */\n";
  os << "/* dispatch_emitc_common_lower_to_emitc_boundary: "
        "TCRVLowerToEmitCSourceAuthority */\n";
  os << "/* dispatch_emitc_materialization_boundary: verified MLIR EmitC "
        "module with emitc.include, emitc.func, emitc.if, emitc.cmp, "
        "emitc.call_opaque, and emitc.return before MLIR Cpp emitter "
        "production source output */\n";
  os << "/* dispatch_emitc_materialization_function: @"
     << dispatcherFunctionName << " */\n";
  os << "/* dispatch_emitc_c_source_authority: MLIR EmitC module translated "
        "by mlir::emitc::translateToCpp */\n";
  os << "/* dispatch_emitc_runtime_guard_value: " << guardValueName
     << " */\n";
  os << "/* dispatch_emitc_route_id: " << route.getRouteID()
     << ", route_kind=" << route.getRouteKind() << " */\n";
  for (auto [index, step] : llvm::enumerate(route.getCallOpaqueSteps())) {
    os << "/* dispatch_emitc.call_opaque[" << index
       << "]: " << step.callee << " from " << step.sourceOp.opName
       << ", source_role=" << step.sourceOp.role
       << ", operands=" << step.operands.size() << " */\n";
  }
}

llvm::Error emitDispatchFunctionFromEmitC(
    const DispatchPair &pair, const DispatchRouteManifestEntry &route,
    llvm::StringRef dispatcherFunctionName, llvm::StringRef rvvFunctionName,
    llvm::StringRef scalarFunctionName,
    const DispatchRuntimeABIParameterBindings &bindings,
    std::string &dispatchFunctionSource, TCRVEmitCLowerableRoute &emitcRoute) {
  DispatchControlEmitCLowerable lowerable(pair, route, rvvFunctionName,
                                          scalarFunctionName, bindings);

  TCRVLowerToEmitCSourceOptions options;
  options.sourceAuthorityOptions.functionName = dispatcherFunctionName.str();
  options.sourceAuthorityOptions.dispatchGuardValueName =
      bindings.dispatchAvailabilityGuard->cName;
  if (pair.selectedConfig.getFixedVectorSourceExtentContract())
    options.sourceAuthorityOptions.fixedRuntimeElementCount =
        pair.selectedConfig.getFixedVectorSourceExtentContract()
            ->sourceVectorExtent;
  options.sourceAuthorityOptions.requireInterfaceBackedCompute = false;

  llvm::Expected<tianchenrv::conversion::emitc::TCRVLowerToEmitCSourceResult>
      loweredSource =
          lowerTCRVEmitCLowerableToEmitCSource(lowerable, options);
  if (!loweredSource)
    return loweredSource.takeError();

  emitcRoute = loweredSource->getRoute();
  dispatchFunctionSource = loweredSource->takeSource();
  return llvm::Error::success();
}

llvm::Error printDispatchHeader(const DispatchPair &pair,
                                llvm::raw_ostream &os) {
  const DispatchRouteManifestEntry *route =
      lookupRVVScalarDispatchRoute(*pair.family, DispatchRouteKind::Header);
  if (!route || !route->family)
    return makeDispatchError(
        pair.rvv.kernel,
        llvm::Twine("dispatch header export requires manifest route for ") +
            pair.family->diagnosticName);

  llvm::Expected<DispatchRuntimeABIParameterBindings> bindings =
      bindDispatchRuntimeABIParametersByRole(
          pair.rvv.kernel, pair.abiPlan.parameters,
          "RVV+scalar dispatch header ABI plan");
  if (!bindings)
    return bindings.takeError();

  std::string includeGuard = makeDispatchHeaderIncludeGuard(pair);
  std::string dispatcherFunctionName = makeDispatcherFunctionName(pair);

  os << "/* dispatch_manifest_route_id: " << route->routeID << " */\n";
  os << "/* dispatch_manifest_artifact_kind: " << route->artifactKind
     << " */\n";
  if (llvm::Error error = printDispatchSelectedSourceIdentityContract(os, pair))
    return error;
  printDispatchSelectedEmitCBodyMappingSummary(os, pair);
  if (pair.selectedConfig.getFixedVectorSourceExtentContract()) {
    const support::FixedVectorSourceExtentContract &sourceExtent =
        *pair.selectedConfig.getFixedVectorSourceExtentContract();
    os << "/* " << sourceExtent.formatCommentBody() << " */\n";
    os << "/* dispatch_runtime_element_count_constraint: "
       << pair.selectedConfig.getRuntimeElementCountCName()
       << " must equal fixed source vector extent "
       << sourceExtent.sourceVectorExtent
       << " before dispatching to RVV or scalar callable branches */\n";
  }
  if (pair.selectedConfig.getDynamicVectorRuntimeExtentContract()) {
    const support::DynamicVectorRuntimeExtentContract &runtimeExtent =
        *pair.selectedConfig.getDynamicVectorRuntimeExtentContract();
    os << "/* " << runtimeExtent.formatCommentBody() << " */\n";
    os << "/* dispatch_runtime_element_count_source: "
       << pair.selectedConfig.getRuntimeElementCountCName()
       << " is the source scf.for upper bound and runtime AVL; no fixed "
          "source-extent trap is emitted before dispatch */\n";
  }
  for (auto [index, parameter] : llvm::enumerate(pair.abiPlan.parameters)) {
    os << "/* dispatch_runtime_abi_parameter[" << index
       << "]: c_name=" << parameter.cName << ", c_type=" << parameter.cType
       << ", role="
       << support::stringifyRuntimeABIParameterRole(parameter.role)
       << ", ownership="
       << support::stringifyRuntimeABIParameterOwnership(parameter.ownership)
       << " */\n";
  }
  printDispatchRuntimeCallableABI(os, dispatcherFunctionName,
                                  pair.abiPlan.parameters);
  if (llvm::Error error = printDispatchRuntimeABIInvocationContract(
          os, pair, dispatcherFunctionName, pair.abiPlan.parameters,
          bindings->runtimeElementCount->cName,
          bindings->dispatchAvailabilityGuard->cName))
    return error;
  os << "#ifndef " << includeGuard << "\n";
  os << "#define " << includeGuard << "\n\n";
  os << "#include <stddef.h>\n";
  os << "#include <stdint.h>\n\n";
  os << "#ifdef __cplusplus\n";
  os << "extern \"C\" {\n";
  os << "#endif\n\n";

  os << "void " << dispatcherFunctionName << "(";
  for (auto [index, parameter] : llvm::enumerate(pair.abiPlan.parameters)) {
    if (index != 0)
      os << ", ";
    support::printRuntimeABIParameterCDeclaration(os, parameter);
  }
  os << ");\n\n";

  os << "#ifdef __cplusplus\n";
  os << "}\n";
  os << "#endif\n\n";
  os << "#endif /* " << includeGuard << " */\n";
  return llvm::Error::success();
}

void printDispatchSelfCheckHarness(llvm::raw_ostream &os,
                                   const BinarySelfCheckExpectation
                                       &expectation,
                                   llvm::StringRef dispatcherFunctionName,
                                   llvm::StringRef runtimeElementCountName,
                                   llvm::StringRef guardParameterName,
                                   llvm::StringRef successMarker,
                                   std::optional<std::int64_t>
                                       fixedSourceVectorExtent) {
  os << "\n/* Explicit bounded self-check harness for RVV+scalar dispatch "
        "runtime invocation evidence. */\n";
  if (fixedSourceVectorExtent) {
    os << "/* Harness scope: calls the generated dispatcher with fixed "
       << runtimeElementCountName << " value " << *fixedSourceVectorExtent
       << " for " << guardParameterName << " = 0 and " << guardParameterName
       << " = 1 because the vector source extent is fixed. */\n";
  } else {
    os << "/* Harness scope: calls the generated dispatcher with explicit "
       << runtimeElementCountName << " values 7 and 16 for "
       << guardParameterName << " = 0 and " << guardParameterName
       << " = 1. */\n";
  }
  os << "/* Runtime element count is a target/export-owned ABI parameter in "
        "this harness; "
        "descriptor-local element_count remains metadata only. */\n";
  os << "/* self_check_expectation_source: " << expectation.provenance
     << "; legacy descriptor mirrors cannot select expected arithmetic or "
        "scalar element type. */\n";
  os << "int puts(const char *);\n\n";
  os << "static int " << dispatcherFunctionName
     << "_self_check_one(size_t runtime_n, int " << guardParameterName
     << ") {\n";
  os << "  enum { kCapacity = 32 };\n";
  os << "  " << expectation.scalarElementCType << " lhs[kCapacity];\n";
  os << "  " << expectation.scalarElementCType << " rhs[kCapacity];\n";
  os << "  " << expectation.scalarElementCType << " out[kCapacity];\n";
  os << "  for (size_t index = 0; index < (size_t)kCapacity; ++index) {\n";
  os << "    lhs[index] = (" << expectation.scalarElementCType << ")index;\n";
  os << "    rhs[index] = (" << expectation.scalarElementCType
     << ")(31 - (int)index);\n";
  os << "    out[index] = (" << expectation.scalarElementCType
     << ")-12345;\n";
  os << "  }\n";
  os << "  " << dispatcherFunctionName
     << "(lhs, rhs, out, runtime_n, " << guardParameterName << ");\n";
  os << "  for (size_t index = 0; index < runtime_n; ++index) {\n";
  os << "    if (out[index] != "
     << expectation.formatExpression("lhs[index]", "rhs[index]")
     << ")\n";
  os << "      return 1;\n";
  os << "  }\n";
  os << "  for (size_t index = runtime_n; index < (size_t)kCapacity; "
        "++index) {\n";
  os << "    if (out[index] != (" << expectation.scalarElementCType
     << ")-12345)\n";
  os << "      return 2;\n";
  os << "  }\n";
  os << "  return 0;\n";
  os << "}\n\n";
  os << "int main(void) {\n";
  if (fixedSourceVectorExtent) {
    os << "  if (" << dispatcherFunctionName << "_self_check_one("
       << *fixedSourceVectorExtent << ", 0))\n";
    os << "    return 1;\n";
    os << "  if (" << dispatcherFunctionName << "_self_check_one("
       << *fixedSourceVectorExtent << ", 1))\n";
    os << "    return 2;\n";
    os << "  puts(\"" << successMarker
       << " runtime_counts=" << *fixedSourceVectorExtent
       << " branches=scalar_and_rvv\");\n";
    os << "  return 0;\n";
    os << "}\n";
    return;
  }
  os << "  if (" << dispatcherFunctionName << "_self_check_one(7, 0))\n";
  os << "    return 1;\n";
  os << "  if (" << dispatcherFunctionName << "_self_check_one(16, 0))\n";
  os << "    return 2;\n";
  os << "  if (" << dispatcherFunctionName << "_self_check_one(7, 1))\n";
  os << "    return 3;\n";
  os << "  if (" << dispatcherFunctionName << "_self_check_one(16, 1))\n";
  os << "    return 4;\n";
  os << "  puts(\"" << successMarker
     << " runtime_counts=7,16 branches=scalar_and_rvv\");\n";
  os << "  return 0;\n";
  os << "}\n";
}

llvm::Error printDispatchSource(const DispatchPair &pair,
                                llvm::StringRef rvvSource,
                                llvm::StringRef scalarSource,
                                bool includeSelfCheck, llvm::raw_ostream &os) {
  DispatchRouteKind manifestRouteKind =
      includeSelfCheck ? DispatchRouteKind::SelfCheckSource
                       : DispatchRouteKind::Source;
  const DispatchRouteManifestEntry *route =
      lookupRVVScalarDispatchRoute(*pair.family, manifestRouteKind);
  if (!route || !route->family)
    return makeDispatchError(
        pair.rvv.kernel,
        llvm::Twine("dispatch source export requires manifest route for ") +
            pair.family->diagnosticName);

  KernelOp kernel = pair.rvv.kernel;
  std::string rvvFunctionName = makeRVVFunctionName(pair);
  std::string scalarFunctionName = makeScalarFunctionName(pair);
  std::string dispatcherFunctionName = makeDispatcherFunctionName(pair);
  llvm::ArrayRef<support::RuntimeABIParameter> dispatchParameters =
      pair.abiPlan.parameters;
  llvm::Expected<DispatchRuntimeABIParameterBindings> bindings =
      bindDispatchRuntimeABIParametersByRole(
          kernel, dispatchParameters, "RVV+scalar dispatch C ABI plan");
  if (!bindings)
    return bindings.takeError();

  std::string dispatchFunctionSource;
  TCRVEmitCLowerableRoute dispatchEmitCRoute;
  if (llvm::Error error = emitDispatchFunctionFromEmitC(
          pair, *route, dispatcherFunctionName, rvvFunctionName,
          scalarFunctionName, *bindings, dispatchFunctionSource,
          dispatchEmitCRoute))
    return error;

  std::optional<BinarySelfCheckExpectation> selfCheckExpectation;
  if (includeSelfCheck) {
    llvm::Expected<BinarySelfCheckExpectation> expectation =
        buildDispatchSelfCheckExpectation(pair);
    if (!expectation)
      return expectation.takeError();
    selfCheckExpectation = std::move(*expectation);
  }

  os << "/* TianChen-RV RVV+scalar host runtime dispatch C export. */\n";
  os << "/* Scope: one selected RVV " << pair.composite.diagnosticName
     << " dispatch case plus one scalar " << pair.composite.diagnosticName
     << " dispatch fallback. */\n";
  os << "/* Runtime guard: explicit host-provided "
     << bindings->dispatchAvailabilityGuard->cName
     << " parameter; no automatic hardware probe is generated. */\n";
  os << "/* selected_kernel: @" << kernel.getSymName() << " */\n";
  os << "/* " << pair.selectedConfig.formatSummaryCommentBody() << " */\n";
  if (pair.selectedConfig.getFixedVectorSourceExtentContract()) {
    const support::FixedVectorSourceExtentContract &sourceExtent =
        *pair.selectedConfig.getFixedVectorSourceExtentContract();
    os << "/* " << sourceExtent.formatCommentBody() << " */\n";
    os << "/* dispatch_runtime_element_count_constraint: "
       << bindings->runtimeElementCount->cName
       << " must equal fixed source vector extent "
       << sourceExtent.sourceVectorExtent
       << " before dispatching to RVV or scalar callable branches */\n";
  }
  if (pair.selectedConfig.getDynamicVectorRuntimeExtentContract()) {
    const support::DynamicVectorRuntimeExtentContract &runtimeExtent =
        *pair.selectedConfig.getDynamicVectorRuntimeExtentContract();
    os << "/* " << runtimeExtent.formatCommentBody() << " */\n";
    os << "/* dispatch_runtime_element_count_source: "
       << bindings->runtimeElementCount->cName
       << " is the source scf.for upper bound and runtime AVL; no fixed "
          "source-extent trap is emitted before dispatch */\n";
  }
  os << "/* dispatch_manifest_route_id: " << route->routeID << " */\n";
  os << "/* dispatch_manifest_artifact_kind: " << route->artifactKind
     << " */\n";
  if (llvm::Error error = printDispatchSelectedSourceIdentityContract(os, pair))
    return error;
  printDispatchSelectedEmitCBodyMappingSummary(os, pair);
  printCandidateMetadata(os, "rvv", pair.rvv);
  printCandidateMetadata(os, "scalar", pair.scalar);
  printDispatchMemWindowMetadata(os, pair.abiPlan.bufferWindows);
  printDispatchRuntimeParamMetadata(os, pair.abiPlan.runtimeParams);
  os << "/* dispatch_runtime_guard_link: case=@"
     << pair.rvv.selectedVariant << ", runtime_guard=@"
     << getRuntimeParamStringAttr(pair.abiPlan.runtimeGuardParam, "sym_name")
     << " */\n";
  os << "/* dispatch_fallback_link: target=@"
     << pair.irLink.fallbackTarget << ", selected_scalar_callable=@"
     << pair.scalar.selectedVariant << " */\n";
  os << "/* dispatch_fallback_metadata: target=@"
     << pair.irLink.fallbackTarget
     << ", origin=" << pair.irLink.fallbackOrigin
     << ", fallback_role=" << pair.irLink.fallbackRole << " */\n";
  os << "/* rvv_callable_symbol: " << rvvFunctionName << " */\n";
  os << "/* scalar_callable_symbol: " << scalarFunctionName << " */\n";
  for (auto [index, parameter] : llvm::enumerate(dispatchParameters)) {
    os << "/* dispatch_runtime_abi_parameter[" << index
       << "]: c_name=" << parameter.cName << ", c_type=" << parameter.cType
       << ", role="
       << support::stringifyRuntimeABIParameterRole(parameter.role)
       << ", ownership="
       << support::stringifyRuntimeABIParameterOwnership(parameter.ownership)
       << " */\n";
  }
  printDispatchRuntimeCallableABI(os, dispatcherFunctionName,
                                  dispatchParameters);
  if (llvm::Error error = printDispatchRuntimeABIInvocationContract(
          os, pair, dispatcherFunctionName, dispatchParameters,
          bindings->runtimeElementCount->cName,
          bindings->dispatchAvailabilityGuard->cName))
    return error;
  os << "\n";

  os << "/* Embedded selected RVV runtime-callable source artifact. */\n";
  os << rvvSource;
  if (!rvvSource.ends_with("\n"))
    os << "\n";
  os << "\n/* Embedded selected scalar runtime-callable fallback source "
        "artifact. */\n";
  os << scalarSource;
  if (!scalarSource.ends_with("\n"))
    os << "\n";
  os << "\n/* Host dispatcher emitted through MLIR EmitC source authority over "
        "the two validated callable artifacts. */\n";
  printDispatchEmitCRouteMetadata(os, dispatchEmitCRoute,
                                  dispatcherFunctionName,
                                  bindings->dispatchAvailabilityGuard->cName);
  os << dispatchFunctionSource;
  if (!llvm::StringRef(dispatchFunctionSource).ends_with("\n"))
    os << "\n";
  if (includeSelfCheck) {
    printDispatchSelfCheckHarness(os, *selfCheckExpectation,
                                  dispatcherFunctionName,
                                  bindings->runtimeElementCount->cName,
                                  bindings->dispatchAvailabilityGuard->cName,
                                  pair.composite.selfCheckSuccessMarker,
                                  pair.selectedConfig
                                          .getFixedVectorSourceExtentContract()
                                      ? std::optional<std::int64_t>(
                                            pair.selectedConfig
                                                .getFixedVectorSourceExtentContract()
                                                ->sourceVectorExtent)
                                      : std::nullopt);
  }
  return llvm::Error::success();
}

llvm::Error createTempFile(llvm::StringRef prefix, llvm::StringRef suffix,
                           TemporaryFile &file) {
  std::error_code ec =
      llvm::sys::fs::createTemporaryFile(prefix, suffix, file.path);
  if (ec)
    return makeModuleDispatchObjectError(
        llvm::Twine("failed to create temporary ") + suffix +
        " file for object export: " + ec.message());
  return llvm::Error::success();
}

llvm::Error writeTempSource(llvm::StringRef source, TemporaryFile &sourceFile) {
  int fd = -1;
  std::error_code ec = llvm::sys::fs::createTemporaryFile(
      "tcrv-rvv-scalar-dispatch", "c", fd, sourceFile.path);
  if (ec)
    return makeModuleDispatchObjectError(
        llvm::Twine("failed to create temporary C source for object export: ") +
        ec.message());

  llvm::raw_fd_ostream stream(fd, /*shouldClose=*/true);
  stream << source;
  stream.close();
  if (stream.has_error())
    return makeModuleDispatchObjectError(
        "failed to write generated dispatch C source before object export");
  return llvm::Error::success();
}

std::string readBoundedStderr(llvm::StringRef stderrPath) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(stderrPath);
  if (!buffer)
    return "<stderr unavailable>";
  std::string text = (*buffer)->getBuffer().str();
  for (char &character : text) {
    if (character == '\n' || character == '\r' || character == '\t')
      character = ' ';
  }
  constexpr std::size_t kMaxDiagnosticBytes = 1024;
  if (text.size() > kMaxDiagnosticBytes) {
    text.resize(kMaxDiagnosticBytes);
    text += "...";
  }
  return text;
}

std::string formatCompileCommand(llvm::StringRef targetTriple,
                                 llvm::StringRef selectedMarch,
                                 const std::optional<std::string> &selectedMABI) {
  std::string command;
  llvm::raw_string_ostream stream(command);
  stream << "clang -target " << targetTriple << " -O2 -march="
         << selectedMarch;
  if (selectedMABI)
    stream << " -mabi=" << *selectedMABI;
  stream << " -c <generated-dispatch-source> -o <object-file>";
  stream.flush();
  return command;
}

llvm::Error compileGeneratedDispatchSourceToObject(
    KernelOp kernel, llvm::StringRef source,
    const DispatchObjectCompileConfig &compileConfig, llvm::raw_ostream &os) {
  llvm::ErrorOr<std::string> clangPath =
      llvm::sys::findProgramByName("clang");
  if (!clangPath)
    clangPath = llvm::sys::findProgramByName("clang-20");
  if (!clangPath)
    return makeDispatchObjectError(
        kernel,
        "requires clang or clang-20 on PATH to compile the bounded dispatch C "
        "source into an object file");

  TemporaryFile sourceFile;
  if (llvm::Error error = writeTempSource(source, sourceFile))
    return error;

  TemporaryFile objectFile;
  if (llvm::Error error =
          createTempFile("tcrv-rvv-scalar-dispatch", "o", objectFile))
    return error;

  TemporaryFile stderrFile;
  if (llvm::Error error =
          createTempFile("tcrv-rvv-scalar-dispatch", "stderr",
                         stderrFile))
    return error;

  std::string marchArg = "-march=" + compileConfig.selectedMarch;
  std::string mabiArg;
  llvm::SmallVector<llvm::StringRef, 8> args;
  args.push_back(*clangPath);
  args.push_back("-target");
  args.push_back(compileConfig.targetTriple);
  args.push_back("-O2");
  args.push_back(marchArg);
  if (compileConfig.selectedMABI) {
    mabiArg = "-mabi=" + *compileConfig.selectedMABI;
    args.push_back(mabiArg);
  }
  args.push_back("-c");
  args.push_back(sourceFile.get());
  args.push_back("-o");
  args.push_back(objectFile.get());

  llvm::SmallVector<std::optional<llvm::StringRef>, 3> redirects;
  redirects.emplace_back(llvm::StringRef());
  redirects.emplace_back(llvm::StringRef());
  redirects.emplace_back(stderrFile.get());

  std::string executionError;
  bool executionFailed = false;
  int exitCode = llvm::sys::ExecuteAndWait(
      *clangPath, args, std::nullopt, redirects, /*SecondsToWait=*/60,
      /*MemoryLimit=*/0, &executionError, &executionFailed);
  if (executionFailed || !executionError.empty() || exitCode != 0) {
    std::string stderrText = readBoundedStderr(stderrFile.get());
    return makeDispatchObjectError(
        kernel, llvm::Twine("clang failed while creating object file; ") +
                    "command: " +
                    formatCompileCommand(compileConfig.targetTriple,
                                         compileConfig.selectedMarch,
                                         compileConfig.selectedMABI) +
                    "; exit_code=" + llvm::Twine(exitCode) +
                    "; stderr: " + stderrText);
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> objectBuffer =
      llvm::MemoryBuffer::getFile(objectFile.get(), /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!objectBuffer)
    return makeDispatchObjectError(
        kernel, llvm::Twine("failed to read generated object file: ") +
                    objectBuffer.getError().message());

  llvm::StringRef bytes = (*objectBuffer)->getBuffer();
  if (bytes.empty())
    return makeDispatchObjectError(kernel,
                                   "generated object file must be non-empty");
  if (bytes.size() < 4 || !bytes.starts_with("\177ELF"))
    return makeDispatchObjectError(
        kernel, "generated object file must have an ELF object-file signature");

  os.write(bytes.data(), bytes.size());
  return llvm::Error::success();
}

llvm::Expected<DispatchPair> collectDispatchPairForExpectedFamily(
    mlir::ModuleOp module, const DispatchI32FamilySpec &expectedFamily,
    DispatchRouteKind routeKind, llvm::StringRef routeContext) {
  if (llvm::Error error = validateDispatchManifestRouteForFamily(
          KernelOp(), expectedFamily, routeKind, routeContext))
    return std::move(error);

  llvm::Expected<DispatchPair> pair = collectDispatchPair(module);
  if (!pair)
    return pair.takeError();
  if (pair->family != &expectedFamily)
    return makeDispatchError(
        pair->rvv.kernel,
        llvm::Twine(routeContext) + " expected " +
            expectedFamily.diagnosticName + " dispatch artifacts, got " +
            pair->composite.diagnosticName);
  return std::move(*pair);
}

llvm::Error exportDispatchSourceFromPair(const DispatchPair &pair,
                                         bool includeSelfCheck,
                                         llvm::raw_ostream &os) {
  std::string rvvSource;
  std::string scalarSource;
  if (llvm::Error error =
          buildEmbeddedCallableSources(pair, rvvSource, scalarSource))
    return error;

  std::string source;
  llvm::raw_string_ostream stream(source);
  if (llvm::Error error = printDispatchSource(pair, rvvSource, scalarSource,
                                              includeSelfCheck, stream))
    return error;
  stream.flush();
  os << source;
  return llvm::Error::success();
}

llvm::Error exportDispatchSourceImpl(mlir::ModuleOp module,
                                     llvm::raw_ostream &os) {
  llvm::Expected<DispatchPair> pair = collectDispatchPair(module);
  if (!pair)
    return pair.takeError();
  if (llvm::Error error = validateDispatchManifestRouteForFamily(
          pair->rvv.kernel, *pair->family, DispatchRouteKind::Source,
          "generic source export route"))
    return error;
  return exportDispatchSourceFromPair(*pair, /*includeSelfCheck=*/false, os);
}

llvm::Error exportDispatchSourceForFamily(
    mlir::ModuleOp module, const DispatchI32FamilySpec &expectedFamily,
    llvm::raw_ostream &os) {
  llvm::Expected<DispatchPair> pair = collectDispatchPairForExpectedFamily(
      module, expectedFamily, DispatchRouteKind::Source,
      "direct source export route");
  if (!pair)
    return pair.takeError();
  return exportDispatchSourceFromPair(*pair, /*includeSelfCheck=*/false, os);
}

llvm::Error exportDispatchHeaderFromPair(const DispatchPair &pair,
                                         llvm::raw_ostream &os) {
  std::string rvvSource;
  std::string scalarSource;
  if (llvm::Error error =
          buildEmbeddedCallableSources(pair, rvvSource, scalarSource)) {
    std::string message = llvm::toString(std::move(error));
    return makeModuleDispatchHeaderError(message);
  }

  llvm::Expected<DispatchObjectCompileConfig> compileConfig =
      buildDispatchObjectCompileConfig(pair);
  if (!compileConfig) {
    std::string message = llvm::toString(compileConfig.takeError());
    return makeModuleDispatchHeaderError(message);
  }

  if (llvm::Error error = printDispatchHeader(pair, os)) {
    std::string message = llvm::toString(std::move(error));
    return makeModuleDispatchHeaderError(message);
  }
  return llvm::Error::success();
}

llvm::Error exportDispatchHeaderImpl(mlir::ModuleOp module,
                                     llvm::raw_ostream &os) {
  llvm::Expected<DispatchPair> pair = collectDispatchPair(module);
  if (!pair) {
    std::string message = llvm::toString(pair.takeError());
    return makeModuleDispatchHeaderError(message);
  }
  if (llvm::Error error = validateDispatchManifestRouteForFamily(
          pair->rvv.kernel, *pair->family, DispatchRouteKind::Header,
          "generic header export route")) {
    std::string message = llvm::toString(std::move(error));
    return makeModuleDispatchHeaderError(message);
  }
  return exportDispatchHeaderFromPair(*pair, os);
}

llvm::Error exportDispatchHeaderForFamily(
    mlir::ModuleOp module, const DispatchI32FamilySpec &expectedFamily,
    llvm::raw_ostream &os) {
  llvm::Expected<DispatchPair> pair = collectDispatchPairForExpectedFamily(
      module, expectedFamily, DispatchRouteKind::Header,
      "direct header export route");
  if (!pair) {
    std::string message = llvm::toString(pair.takeError());
    return makeModuleDispatchHeaderError(message);
  }
  return exportDispatchHeaderFromPair(*pair, os);
}

llvm::Error exportDispatchSelfCheckSourceForFamily(
    mlir::ModuleOp module, const DispatchI32FamilySpec &expectedFamily,
    llvm::raw_ostream &os) {
  llvm::Expected<DispatchPair> pair = collectDispatchPairForExpectedFamily(
      module, expectedFamily, DispatchRouteKind::SelfCheckSource,
      "self-check export route");
  if (!pair)
    return pair.takeError();
  return exportDispatchSourceFromPair(*pair, /*includeSelfCheck=*/true, os);
}

llvm::Error exportDispatchObjectFromPair(const DispatchPair &pair,
                                         bool includeSelfCheck,
                                         llvm::raw_ostream &os) {
  llvm::Expected<DispatchObjectCompileConfig> compileConfig =
      buildDispatchObjectCompileConfig(pair);
  if (!compileConfig)
    return compileConfig.takeError();

  std::string source;
  llvm::raw_string_ostream stream(source);
  if (llvm::Error error =
          exportDispatchSourceFromPair(pair, includeSelfCheck, stream)) {
    std::string message = llvm::toString(std::move(error));
    return makeModuleDispatchObjectError(message);
  }
  stream.flush();
  if (source.empty())
    return makeDispatchObjectError(
        pair.rvv.kernel,
        includeSelfCheck
            ? "validated self-check C source must be non-empty before object "
              "export"
            : "validated library-style dispatch C source must be non-empty "
              "before object export");

  return compileGeneratedDispatchSourceToObject(pair.rvv.kernel, source,
                                                *compileConfig, os);
}

llvm::Error exportDispatchObjectImpl(mlir::ModuleOp module,
                                     llvm::raw_ostream &os) {
  llvm::Expected<DispatchPair> pair = collectDispatchPair(module);
  if (!pair) {
    std::string message = llvm::toString(pair.takeError());
    return makeModuleDispatchObjectError(message);
  }
  if (llvm::Error error = validateDispatchManifestRouteForFamily(
          pair->rvv.kernel, *pair->family, DispatchRouteKind::Object,
          "generic object export route")) {
    std::string message = llvm::toString(std::move(error));
    return makeModuleDispatchObjectError(message);
  }
  return exportDispatchObjectFromPair(*pair, /*includeSelfCheck=*/false, os);
}

llvm::Error exportDispatchObjectForFamily(
    mlir::ModuleOp module, const DispatchI32FamilySpec &expectedFamily,
    llvm::raw_ostream &os) {
  llvm::Expected<DispatchPair> pair = collectDispatchPairForExpectedFamily(
      module, expectedFamily, DispatchRouteKind::Object,
      "direct object export route");
  if (!pair) {
    std::string message = llvm::toString(pair.takeError());
    return makeModuleDispatchObjectError(message);
  }
  return exportDispatchObjectFromPair(*pair, /*includeSelfCheck=*/false, os);
}

llvm::Error exportDispatchSelfCheckObjectForFamily(
    mlir::ModuleOp module, const DispatchI32FamilySpec &expectedFamily,
    llvm::raw_ostream &os) {
  llvm::Expected<DispatchPair> pair = collectDispatchPairForExpectedFamily(
      module, expectedFamily, DispatchRouteKind::SelfCheckObject,
      "self-check object export route");
  if (!pair) {
    std::string message = llvm::toString(pair.takeError());
    return makeModuleDispatchObjectError(message);
  }
  return exportDispatchObjectFromPair(*pair, /*includeSelfCheck=*/true, os);
}

RVVScalarDispatchRouteManifestEntry
makeRVVScalarDispatchRouteManifestEntry(const DispatchI32FamilySpec &family,
                                        DispatchRouteKind routeKind) {
  switch (routeKind) {
  case DispatchRouteKind::Source:
    return {&family,
            DispatchRouteKind::Source,
            family.dispatchSourceRouteID,
            "export one host RVV+scalar binary dispatch C source",
            kRuntimeCallableCSourceArtifactKind,
            family.dispatchRuntimeABIKind,
            family.dispatchRuntimeABIName,
            family.dispatchExternalABIComponentGroup,
            family.dispatchRuntimeABIName,
            family.selfCheckSuccessMarker,
            /*requiresBinaryStdout=*/false};
  case DispatchRouteKind::Header:
    return {&family,
            DispatchRouteKind::Header,
            family.dispatchHeaderRouteID,
            "export one host RVV+scalar binary dispatch C ABI header",
            kRuntimeCallableCHeaderArtifactKind,
            family.dispatchRuntimeABIKind,
            family.dispatchRuntimeABIName,
            family.dispatchExternalABIComponentGroup,
            family.dispatchRuntimeABIName,
            family.selfCheckSuccessMarker,
            /*requiresBinaryStdout=*/false};
  case DispatchRouteKind::Object:
    return {&family,
            DispatchRouteKind::Object,
            family.dispatchObjectRouteID,
            "export one host RVV+scalar binary dispatch library object file",
            kRiscvELFRelocatableObjectArtifactKind,
            family.dispatchRuntimeABIKind,
            family.dispatchRuntimeABIName,
            family.dispatchExternalABIComponentGroup,
            family.dispatchRuntimeABIName,
            family.selfCheckSuccessMarker,
            /*requiresBinaryStdout=*/true};
  case DispatchRouteKind::SelfCheckSource:
    return {&family,
            DispatchRouteKind::SelfCheckSource,
            family.dispatchSelfCheckSourceRouteID,
            "export one host RVV+scalar binary dispatch C source with "
            "self-check harness",
            kSelfCheckCSourceArtifactKind,
            family.dispatchRuntimeABIKind,
            family.dispatchRuntimeABIName,
            family.dispatchExternalABIComponentGroup,
            family.dispatchRuntimeABIName,
            family.selfCheckSuccessMarker,
            /*requiresBinaryStdout=*/false};
  case DispatchRouteKind::SelfCheckObject:
    return {&family,
            DispatchRouteKind::SelfCheckObject,
            family.dispatchSelfCheckObjectRouteID,
            "export one host RVV+scalar binary dispatch self-check object "
            "file",
            kSelfCheckObjectArtifactKind,
            family.dispatchRuntimeABIKind,
            family.dispatchRuntimeABIName,
            family.dispatchExternalABIComponentGroup,
            family.dispatchRuntimeABIName,
            family.selfCheckSuccessMarker,
            /*requiresBinaryStdout=*/true};
  }
  llvm_unreachable("unknown RVV+scalar dispatch route kind");
}

TargetArtifactRouteMetadata buildRVVScalarDispatchRouteMetadata(
    const RVVScalarDispatchRouteManifestEntry &route) {
  (void)route;
  TargetArtifactRouteMetadata metadata;
  metadata.addClaimField("compile_export_claim", "compiler-artifact-only");
  metadata.addClaimField("runtime_correctness_claim", "none");
  metadata.addClaimField("hardware_execution_claim", "none");
  metadata.addClaimField("performance_claim", "none");
  metadata.addClaimField("descriptor_compute_authority",
                         "quarantined-by-selected-rvv-scalar-components");
  metadata.addClaimField("descriptor_config_authority",
                         "quarantined-by-dispatch-selected-config-contract");
  metadata.addClaimField("descriptor_runtime_authority",
                         "quarantined-runtime-avl-from-ir-backed-abi");
  return metadata;
}

} // namespace

llvm::ArrayRef<RVVScalarDispatchRouteKind>
getRVVScalarDispatchRouteKinds() {
  static const RVVScalarDispatchRouteKind routeKinds[] = {
      RVVScalarDispatchRouteKind::Source,
      RVVScalarDispatchRouteKind::Header,
      RVVScalarDispatchRouteKind::Object,
      RVVScalarDispatchRouteKind::SelfCheckSource,
      RVVScalarDispatchRouteKind::SelfCheckObject,
  };
  return routeKinds;
}

std::size_t getRVVScalarDispatchRouteCount() {
  return getRVVScalarBinaryRegistrationRecords().size() *
         getRVVScalarDispatchRouteKinds().size();
}

llvm::ArrayRef<RVVScalarDispatchRouteManifestEntry>
getRVVScalarDispatchRouteManifest() {
  static const llvm::SmallVector<RVVScalarDispatchRouteManifestEntry, 32>
      routes = [] {
        llvm::SmallVector<RVVScalarDispatchRouteManifestEntry, 32> result;
        result.reserve(getRVVScalarDispatchRouteCount());
        for (const RVVScalarBinaryFamilyDescriptor *descriptor :
             getRVVScalarBinaryRegistrationRecords()) {
          for (RVVScalarDispatchRouteKind routeKind :
               getRVVScalarDispatchRouteKinds())
            result.push_back(makeRVVScalarDispatchRouteManifestEntry(
                descriptor->dispatch, routeKind));
        }
        return result;
      }();
  return llvm::ArrayRef(routes);
}

const RVVScalarDispatchRouteManifestEntry *
lookupRVVScalarDispatchRoute(llvm::StringRef routeID) {
  routeID = routeID.trim();
  for (const RVVScalarDispatchRouteManifestEntry &route :
       getRVVScalarDispatchRouteManifest())
    if (route.routeID == routeID)
      return &route;
  return nullptr;
}

const RVVScalarDispatchRouteManifestEntry *
lookupRVVScalarDispatchRoute(const DispatchBinaryFamilyDescriptor &family,
                             RVVScalarDispatchRouteKind routeKind) {
  for (const RVVScalarDispatchRouteManifestEntry &route :
       getRVVScalarDispatchRouteManifest())
    if (route.family && route.routeKind == routeKind &&
        isSameDispatchFamily(*route.family, family))
      return &route;
  return nullptr;
}

llvm::Error exportRVVScalarDispatchRoute(
    mlir::ModuleOp module, const RVVScalarDispatchRouteManifestEntry &route,
    llvm::raw_ostream &os) {
  if (!route.family)
    return makeModuleDispatchError("manifest route is missing a dispatch family");

  switch (route.routeKind) {
  case DispatchRouteKind::Source:
    return exportDispatchSourceForFamily(module, *route.family, os);
  case DispatchRouteKind::Header:
    return exportDispatchHeaderForFamily(module, *route.family, os);
  case DispatchRouteKind::Object:
    return exportDispatchObjectForFamily(module, *route.family, os);
  case DispatchRouteKind::SelfCheckSource:
    return exportDispatchSelfCheckSourceForFamily(module, *route.family, os);
  case DispatchRouteKind::SelfCheckObject:
    return exportDispatchSelfCheckObjectForFamily(module, *route.family, os);
  }
  return makeModuleDispatchError("unsupported RVV+scalar dispatch manifest route");
}

llvm::Error exportRVVScalarI32VAddDispatchC(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  return exportDispatchSourceForFamily(module, getI32VAddDispatchFamilySpec(),
                                       os);
}

llvm::Error exportRVVScalarI32VSubDispatchC(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  return exportDispatchSourceForFamily(module, getI32VSubDispatchFamilySpec(),
                                       os);
}

llvm::Error exportRVVScalarI32VMulDispatchC(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  return exportDispatchSourceForFamily(module, getI32VMulDispatchFamilySpec(),
                                       os);
}

llvm::Error exportRVVScalarI64VAddDispatchC(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  return exportDispatchSourceForFamily(module, getI64VAddDispatchFamilySpec(),
                                       os);
}

llvm::Error exportRVVScalarI64VSubDispatchC(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  return exportDispatchSourceForFamily(module, getI64VSubDispatchFamilySpec(),
                                       os);
}

llvm::Error exportRVVScalarI64VMulDispatchC(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  return exportDispatchSourceForFamily(module, getI64VMulDispatchFamilySpec(),
                                       os);
}

llvm::Error exportRVVScalarI32VAddDispatchHeader(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os) {
  return exportDispatchHeaderForFamily(module, getI32VAddDispatchFamilySpec(),
                                       os);
}

llvm::Error exportRVVScalarI32VSubDispatchHeader(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os) {
  return exportDispatchHeaderForFamily(module, getI32VSubDispatchFamilySpec(),
                                       os);
}

llvm::Error exportRVVScalarI32VMulDispatchHeader(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os) {
  return exportDispatchHeaderForFamily(module, getI32VMulDispatchFamilySpec(),
                                       os);
}

llvm::Error exportRVVScalarI64VAddDispatchHeader(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os) {
  return exportDispatchHeaderForFamily(module, getI64VAddDispatchFamilySpec(),
                                       os);
}

llvm::Error exportRVVScalarI64VSubDispatchHeader(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os) {
  return exportDispatchHeaderForFamily(module, getI64VSubDispatchFamilySpec(),
                                       os);
}

llvm::Error exportRVVScalarI64VMulDispatchHeader(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os) {
  return exportDispatchHeaderForFamily(module, getI64VMulDispatchFamilySpec(),
                                       os);
}

llvm::Error exportRVVScalarI32VAddDispatchSelfCheckC(mlir::ModuleOp module,
                                                     llvm::raw_ostream &os) {
  return exportDispatchSelfCheckSourceForFamily(
      module, getI32VAddDispatchFamilySpec(), os);
}

llvm::Error exportRVVScalarI32VSubDispatchSelfCheckC(mlir::ModuleOp module,
                                                     llvm::raw_ostream &os) {
  return exportDispatchSelfCheckSourceForFamily(
      module, getI32VSubDispatchFamilySpec(), os);
}

llvm::Error exportRVVScalarI32VMulDispatchSelfCheckC(mlir::ModuleOp module,
                                                     llvm::raw_ostream &os) {
  return exportDispatchSelfCheckSourceForFamily(
      module, getI32VMulDispatchFamilySpec(), os);
}

llvm::Error exportRVVScalarI64VAddDispatchSelfCheckC(mlir::ModuleOp module,
                                                     llvm::raw_ostream &os) {
  return exportDispatchSelfCheckSourceForFamily(
      module, getI64VAddDispatchFamilySpec(), os);
}

llvm::Error exportRVVScalarI64VSubDispatchSelfCheckC(mlir::ModuleOp module,
                                                     llvm::raw_ostream &os) {
  return exportDispatchSelfCheckSourceForFamily(
      module, getI64VSubDispatchFamilySpec(), os);
}

llvm::Error exportRVVScalarI64VMulDispatchSelfCheckC(mlir::ModuleOp module,
                                                     llvm::raw_ostream &os) {
  return exportDispatchSelfCheckSourceForFamily(
      module, getI64VMulDispatchFamilySpec(), os);
}

llvm::Error exportRVVScalarI32VAddDispatchObject(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os) {
  return exportDispatchObjectForFamily(module, getI32VAddDispatchFamilySpec(),
                                       os);
}

llvm::Error exportRVVScalarI32VSubDispatchObject(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os) {
  return exportDispatchObjectForFamily(module, getI32VSubDispatchFamilySpec(),
                                       os);
}

llvm::Error exportRVVScalarI32VMulDispatchObject(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os) {
  return exportDispatchObjectForFamily(module, getI32VMulDispatchFamilySpec(),
                                       os);
}

llvm::Error exportRVVScalarI64VAddDispatchObject(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os) {
  return exportDispatchObjectForFamily(module, getI64VAddDispatchFamilySpec(),
                                       os);
}

llvm::Error exportRVVScalarI64VSubDispatchObject(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os) {
  return exportDispatchObjectForFamily(module, getI64VSubDispatchFamilySpec(),
                                       os);
}

llvm::Error exportRVVScalarI64VMulDispatchObject(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os) {
  return exportDispatchObjectForFamily(module, getI64VMulDispatchFamilySpec(),
                                       os);
}

llvm::Error
exportRVVScalarI32VAddDispatchSelfCheckObject(mlir::ModuleOp module,
                                              llvm::raw_ostream &os) {
  return exportDispatchSelfCheckObjectForFamily(
      module, getI32VAddDispatchFamilySpec(), os);
}

llvm::Error
exportRVVScalarI32VSubDispatchSelfCheckObject(mlir::ModuleOp module,
                                              llvm::raw_ostream &os) {
  return exportDispatchSelfCheckObjectForFamily(
      module, getI32VSubDispatchFamilySpec(), os);
}

llvm::Error
exportRVVScalarI32VMulDispatchSelfCheckObject(mlir::ModuleOp module,
                                              llvm::raw_ostream &os) {
  return exportDispatchSelfCheckObjectForFamily(
      module, getI32VMulDispatchFamilySpec(), os);
}

llvm::Error
exportRVVScalarI64VAddDispatchSelfCheckObject(mlir::ModuleOp module,
                                              llvm::raw_ostream &os) {
  return exportDispatchSelfCheckObjectForFamily(
      module, getI64VAddDispatchFamilySpec(), os);
}

llvm::Error
exportRVVScalarI64VSubDispatchSelfCheckObject(mlir::ModuleOp module,
                                              llvm::raw_ostream &os) {
  return exportDispatchSelfCheckObjectForFamily(
      module, getI64VSubDispatchFamilySpec(), os);
}

llvm::Error
exportRVVScalarI64VMulDispatchSelfCheckObject(mlir::ModuleOp module,
                                              llvm::raw_ostream &os) {
  return exportDispatchSelfCheckObjectForFamily(
      module, getI64VMulDispatchFamilySpec(), os);
}

llvm::Error registerRVVScalarDispatchRouteTargetExporter(
    TargetArtifactExporterRegistry &registry,
    const RVVScalarDispatchRouteManifestEntry &route) {
  TargetArtifactCompositeMatchFn matchFn =
      getDispatchCompositeMatchFn(*route.family);
  if (!matchFn)
    return llvm::make_error<llvm::StringError>(
        "missing RVV+scalar dispatch composite matcher for binary family",
        llvm::errc::invalid_argument);

  TargetArtifactExportFn exportFn = nullptr;
  switch (route.routeKind) {
  case DispatchRouteKind::Source:
    exportFn = exportDispatchSourceImpl;
    break;
  case DispatchRouteKind::Header:
    exportFn = exportDispatchHeaderImpl;
    break;
  case DispatchRouteKind::Object:
    exportFn = exportDispatchObjectImpl;
    break;
  case DispatchRouteKind::SelfCheckSource:
  case DispatchRouteKind::SelfCheckObject:
    return llvm::Error::success();
  }

  return registry.registerCompositeExporter(TargetArtifactCompositeExporter(
      route.routeID, route.artifactKind, matchFn, exportFn,
      kDispatchTargetOwner, route.runtimeABIKind, route.runtimeABIName,
      resolveRVVScalarDispatchRuntimeABIParameters,
      /*directHelperRoute=*/true, route.componentGroup, route.externalABIName,
      validateRVVScalarDispatchCandidates,
      buildRVVScalarDispatchRouteMetadata(route),
      resolveRVVScalarDispatchBundleMetadata));
}

llvm::Error registerRVVScalarDispatchTargetExporters(
    TargetArtifactExporterRegistry &registry) {
  for (const RVVScalarDispatchRouteManifestEntry &route :
       getRVVScalarDispatchRouteManifest()) {
    if (llvm::Error error =
            registerRVVScalarDispatchRouteTargetExporter(registry, route))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error registerRVVScalarDispatchPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry) {
  static const llvm::StringRef requiredPlugins[] = {kScalarPluginName};
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      kRVVPluginName, registerRVVScalarDispatchTargetExporters,
      requiredPlugins));
}

llvm::Error registerRVVScalarDispatchTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry) {
  for (const RVVScalarDispatchRouteManifestEntry &route :
       getRVVScalarDispatchRouteManifest()) {
    const RVVScalarDispatchRouteManifestEntry *routePtr = &route;
    llvm::StringRef targetArtifactRouteID;
    switch (route.routeKind) {
    case DispatchRouteKind::Source:
    case DispatchRouteKind::Header:
    case DispatchRouteKind::Object:
      targetArtifactRouteID = route.routeID;
      break;
    case DispatchRouteKind::SelfCheckSource:
    case DispatchRouteKind::SelfCheckObject:
      break;
    }
    if (llvm::Error error = registry.registerRoute(TargetTranslateRoute(
            route.routeID, route.description,
            [routePtr](mlir::ModuleOp module, llvm::raw_ostream &os) {
              return exportRVVScalarDispatchRoute(module, *routePtr, os);
            },
            route.requiresBinaryStdout, targetArtifactRouteID)))
      return error;
  }
  return llvm::Error::success();
}

} // namespace tianchenrv::target::rvv_scalar
