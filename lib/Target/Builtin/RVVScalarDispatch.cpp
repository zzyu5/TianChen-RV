#include "TianChenRV/Target/RVVScalarDispatch.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Target/RVV/RVVMicrokernel.h"
#include "TianChenRV/Target/Scalar/ScalarMicrokernel.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "mlir/IR/Attributes.h"
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
#include <optional>
#include <string>
#include <system_error>

namespace tianchenrv::target::rvv_scalar {
namespace {

using tianchenrv::target::TargetArtifactCandidate;
using tianchenrv::target::TargetArtifactExporter;
using tianchenrv::target::TargetArtifactExporterRegistry;
using tianchenrv::support::CapabilityDescriptor;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::DispatchOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kScalarPluginName("scalar-plugin");
constexpr llvm::StringLiteral kDispatchCaseRole("dispatch case");
constexpr llvm::StringLiteral kDispatchFallbackRole("dispatch fallback");
constexpr llvm::StringLiteral kRuntimeCallableCSourceArtifactKind(
    "runtime-callable-c-source");
constexpr llvm::StringLiteral kDispatchRuntimeABIParametersAttrName(
    "tcrv_rvv_scalar.dispatch_runtime_abi_parameters");

constexpr llvm::StringLiteral kRVVRouteID("tcrv-export-rvv-microkernel-c");
constexpr llvm::StringLiteral kRVVEmissionKind(
    "rvv-explicit-i32-vadd-microkernel-c-source");
constexpr llvm::StringLiteral kRVVRuntimeABI(
    "rvv-i32-vadd-runtime-callable-c-abi.v1");
constexpr llvm::StringLiteral kRVVRuntimeABIKind(
    "rvv-runtime-callable-c-abi");
constexpr llvm::StringLiteral kRVVRuntimeABIName(
    "rvv-i32-vadd-runtime-callable-c-function.v1");
constexpr llvm::StringLiteral kRVVRuntimeGlueRole(
    "runtime-callable-i32-vadd-function");

constexpr llvm::StringLiteral kScalarRouteID(
    "tcrv-export-scalar-microkernel-c");
constexpr llvm::StringLiteral kScalarEmissionKind(
    "scalar-explicit-i32-vadd-microkernel-c-source");
constexpr llvm::StringLiteral kScalarRuntimeABI(
    "scalar-i32-vadd-runtime-callable-c-abi.v1");
constexpr llvm::StringLiteral kScalarRuntimeABIKind(
    "scalar-runtime-callable-c-abi");
constexpr llvm::StringLiteral kScalarRuntimeABIName(
    "scalar-i32-vadd-runtime-callable-c-function.v1");
constexpr llvm::StringLiteral kScalarRuntimeGlueRole(
    "runtime-callable-i32-vadd-fallback-function");
constexpr llvm::StringLiteral kRVVRequiredMarchAttrName(
    "tcrv_rvv.required_march");
constexpr llvm::StringLiteral kRVVProbeCompileRunCapabilityID(
    "rvv.probe.compile_run");
constexpr llvm::StringLiteral kRVVToolchainMarchCapabilityID(
    "rvv.toolchain.march");
constexpr llvm::StringLiteral kRVVToolchainMABICapabilityID(
    "rvv.toolchain.mabi");
constexpr llvm::StringLiteral kSelectedMarchPropertyName("selected_march");
constexpr llvm::StringLiteral kSelectedMABIPropertyName("selected_mabi");
constexpr llvm::StringLiteral kValuePropertyName("value");

struct DispatchObjectCompileConfig {
  std::string selectedMarch;
  std::optional<std::string> selectedMABI;
};

struct DispatchABIPlan {
  llvm::SmallVector<support::RuntimeABIParameter, 5> parameters;
};

struct DispatchPair {
  TargetArtifactCandidate rvv;
  TargetArtifactCandidate scalar;
  DispatchABIPlan abiPlan;
};

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
  stream << "TianChen-RV RVV+scalar i32-vadd dispatch C export failed";
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
      llvm::Twine("TianChen-RV RVV+scalar i32-vadd dispatch C export failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error makeDispatchObjectError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV RVV+scalar i32-vadd dispatch self-check object "
            "export failed";
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
      llvm::Twine("TianChen-RV RVV+scalar i32-vadd dispatch self-check "
                  "object export failed: ") +
          message,
      llvm::errc::invalid_argument);
}

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

bool hasCandidateShape(const TargetArtifactCandidate &candidate,
                       llvm::StringRef origin, llvm::StringRef role,
                       llvm::StringRef routeID,
                       llvm::StringRef emissionKind,
                       llvm::StringRef runtimeABI,
                       llvm::StringRef runtimeABIKind,
                       llvm::StringRef runtimeABIName,
                       llvm::StringRef runtimeGlueRole) {
  return candidate.origin == origin && candidate.role == role &&
         candidate.routeID == routeID &&
         candidate.emissionKind == emissionKind &&
         candidate.artifactKind == kRuntimeCallableCSourceArtifactKind &&
         candidate.runtimeABI == runtimeABI &&
         candidate.runtimeABIKind == runtimeABIKind &&
         candidate.runtimeABIName == runtimeABIName &&
         candidate.runtimeGlueRole == runtimeGlueRole;
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

  if (candidate.artifactKind != exporter->getArtifactKind())
    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("selected callable artifact route '") + candidate.routeID +
            "' does not support artifact_kind '" + candidate.artifactKind +
            "'");
  if (!exporter->getOriginPlugin().empty() &&
      candidate.origin != exporter->getOriginPlugin())
    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("selected callable artifact route '") + candidate.routeID +
            "' is registered for origin '" + exporter->getOriginPlugin() +
            "' but selected emission-plan origin is '" + candidate.origin + "'");
  if (!exporter->getEmissionKind().empty() &&
      candidate.emissionKind != exporter->getEmissionKind())
    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("selected callable artifact route '") + candidate.routeID +
            "' is registered for emission_kind '" +
            exporter->getEmissionKind() +
            "' but selected emission-plan emission_kind is '" +
            candidate.emissionKind + "'");
  if (!exporter->getExportFn())
    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("selected callable artifact route '") + candidate.routeID +
            "' has no registered export callback");
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

llvm::Error resolveCallableI32VAddABIParameters(
    const TargetArtifactCandidate &candidate,
    llvm::SmallVectorImpl<support::RuntimeABIParameter> &out) {
  llvm::SmallVector<support::RuntimeABIParameter, 4> expectedParameters =
      support::getI32VAddRuntimeABIRoleRequirements();

  for (const support::RuntimeABIParameter &expected : expectedParameters) {
    const support::RuntimeABIParameter *actualParameter = nullptr;
    unsigned count = 0;
    for (const support::RuntimeABIParameter &actual :
         candidate.runtimeABIParameters) {
      if (actual.role != expected.role)
        continue;
      actualParameter = &actual;
      ++count;
    }

    if (count == 0)
      return makeDispatchError(
          candidate.kernel,
          llvm::Twine("selected callable artifact route '") +
              candidate.routeID + "' requires runtime ABI parameter role '" +
              support::stringifyRuntimeABIParameterRole(expected.role) + "'");
    if (count > 1)
      return makeDispatchError(
          candidate.kernel,
          llvm::Twine("selected callable artifact route '") +
              candidate.routeID +
              "' received duplicate runtime ABI parameter role '" +
              support::stringifyRuntimeABIParameterRole(expected.role) + "'");

    if (llvm::Error error =
            validateDispatchCParameterName(candidate.kernel,
                                           actualParameter->cName))
      return error;

    if (actualParameter->cType != expected.cType)
      return makeDispatchError(
          candidate.kernel,
          llvm::Twine("selected callable artifact route '") +
              candidate.routeID + "' runtime ABI parameter role '" +
              support::stringifyRuntimeABIParameterRole(expected.role) +
              "' must use c type '" + expected.cType + "'");

    if (actualParameter->ownership != expected.ownership)
      return makeDispatchError(
          candidate.kernel,
          llvm::Twine("selected callable artifact route '") +
              candidate.routeID + "' runtime ABI parameter role '" +
              support::stringifyRuntimeABIParameterRole(expected.role) +
              "' must use ownership '" +
              support::stringifyRuntimeABIParameterOwnership(
                  expected.ownership) +
              "'");

    out.push_back(*actualParameter);
  }

  for (const support::RuntimeABIParameter &actual :
       candidate.runtimeABIParameters) {
    bool expectedRole = llvm::any_of(
        expectedParameters, [&](const support::RuntimeABIParameter &expected) {
          return expected.role == actual.role;
        });
    if (!expectedRole)
      return makeDispatchError(
          candidate.kernel,
          llvm::Twine("selected callable artifact route '") +
              candidate.routeID +
              "' received unsupported runtime ABI parameter role '" +
              support::stringifyRuntimeABIParameterRole(actual.role) + "'");
  }

  return llvm::Error::success();
}

DispatchOp findDispatchOpForPair(const DispatchPair &pair) {
  KernelOp kernel = pair.rvv.kernel;
  if (!kernel || kernel.getBody().empty())
    return DispatchOp();

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto dispatch = llvm::dyn_cast<DispatchOp>(op);
    if (dispatch)
      return dispatch;
  }
  return DispatchOp();
}

llvm::Error parseDispatchGuardRuntimeABIParameter(
    KernelOp kernel, mlir::DictionaryAttr dict, std::size_t index,
    support::RuntimeABIParameter &out) {
  if (!dict)
    return makeDispatchError(
        kernel, llvm::Twine(kDispatchRuntimeABIParametersAttrName) + "[" +
                    llvm::Twine(index) + "] must be a dictionary attribute");

  auto cName =
      dict.getAs<mlir::StringAttr>(support::kRuntimeABIParameterCNameAttrName);
  auto cType =
      dict.getAs<mlir::StringAttr>(support::kRuntimeABIParameterCTypeAttrName);
  auto role =
      dict.getAs<mlir::StringAttr>(support::kRuntimeABIParameterRoleAttrName);
  auto ownership = dict.getAs<mlir::StringAttr>(
      support::kRuntimeABIParameterOwnershipAttrName);
  if (!cName || cName.getValue().trim().empty())
    return makeDispatchError(
        kernel, llvm::Twine(kDispatchRuntimeABIParametersAttrName) + "[" +
                    llvm::Twine(index) + "] requires non-empty c_name");
  if (!cType || cType.getValue().trim().empty())
    return makeDispatchError(
        kernel, llvm::Twine(kDispatchRuntimeABIParametersAttrName) + "[" +
                    llvm::Twine(index) + "] requires non-empty c_type");
  if (!role || role.getValue().trim().empty())
    return makeDispatchError(
        kernel, llvm::Twine(kDispatchRuntimeABIParametersAttrName) + "[" +
                    llvm::Twine(index) + "] requires non-empty role");
  if (!ownership || ownership.getValue().trim().empty())
    return makeDispatchError(
        kernel, llvm::Twine(kDispatchRuntimeABIParametersAttrName) + "[" +
                    llvm::Twine(index) + "] requires non-empty ownership");

  llvm::StringRef cNameValue = cName.getValue().trim();
  llvm::StringRef cTypeValue = cType.getValue().trim();
  llvm::StringRef roleValue = role.getValue().trim();
  llvm::StringRef ownershipValue = ownership.getValue().trim();

  if (llvm::Error error = validateDispatchRuntimeABIText(
          kernel, "dispatch runtime ABI parameter c_name", cNameValue))
    return error;
  if (llvm::Error error = validateDispatchRuntimeABIText(
          kernel, "dispatch runtime ABI parameter c_type", cTypeValue))
    return error;
  if (llvm::Error error = validateDispatchRuntimeABIText(
          kernel, "dispatch runtime ABI parameter role", roleValue))
    return error;
  if (llvm::Error error = validateDispatchRuntimeABIText(
          kernel, "dispatch runtime ABI parameter ownership", ownershipValue))
    return error;
  if (llvm::Error error = validateDispatchCParameterName(kernel, cNameValue))
    return error;

  std::optional<support::RuntimeABIParameterRole> parsedRole =
      support::symbolizeRuntimeABIParameterRole(roleValue);
  if (!parsedRole)
    return makeDispatchError(
        kernel,
        llvm::Twine("unsupported dispatch runtime ABI parameter role '") +
            roleValue + "'");
  if (*parsedRole != support::RuntimeABIParameterRole::DispatchAvailabilityGuard)
    return makeDispatchError(
        kernel,
        llvm::Twine(kDispatchRuntimeABIParametersAttrName) +
            " only accepts runtime ABI role 'dispatch-availability-guard'");

  std::optional<support::RuntimeABIParameterOwnership> parsedOwnership =
      support::symbolizeRuntimeABIParameterOwnership(ownershipValue);
  if (!parsedOwnership)
    return makeDispatchError(
        kernel,
        llvm::Twine("unsupported dispatch runtime ABI parameter ownership '") +
            ownershipValue + "'");

  out = support::RuntimeABIParameter(cNameValue, cTypeValue, *parsedRole,
                                     *parsedOwnership);
  return llvm::Error::success();
}

llvm::Expected<support::RuntimeABIParameter>
resolveDispatchAvailabilityGuardParameter(const DispatchPair &pair) {
  KernelOp kernel = pair.rvv.kernel;
  DispatchOp dispatch = findDispatchOpForPair(pair);
  if (!dispatch)
    return makeDispatchError(
        kernel, "requires one tcrv.exec.dispatch to build dispatch ABI "
                "availability guard metadata");

  auto guardParameters =
      dispatch->getAttrOfType<mlir::ArrayAttr>(
          kDispatchRuntimeABIParametersAttrName);
  if (!guardParameters)
    return support::makeI32VAddDispatchAvailabilityGuard();

  if (guardParameters.empty())
    return makeDispatchError(
        kernel,
        llvm::Twine(kDispatchRuntimeABIParametersAttrName) +
            " must contain exactly one dispatch-availability-guard parameter");
  if (guardParameters.size() != 1)
    return makeDispatchError(
        kernel,
        llvm::Twine(kDispatchRuntimeABIParametersAttrName) +
            " must contain exactly one dispatch-availability-guard parameter");

  support::RuntimeABIParameter guard;
  if (llvm::Error error = parseDispatchGuardRuntimeABIParameter(
          kernel, llvm::dyn_cast<mlir::DictionaryAttr>(guardParameters[0]), 0,
          guard))
    return std::move(error);

  support::RuntimeABIParameter expected =
      support::makeI32VAddDispatchAvailabilityGuard(guard.cName);
  if (guard.cType != expected.cType)
    return makeDispatchError(
        kernel, "dispatch availability guard runtime ABI parameter must use c "
                "type 'int'");
  if (guard.ownership != expected.ownership)
    return makeDispatchError(
        kernel,
        llvm::Twine("dispatch availability guard runtime ABI parameter must "
                    "use ownership '") +
            support::stringifyRuntimeABIParameterOwnership(expected.ownership) +
            "'");
  return guard;
}

llvm::Expected<DispatchABIPlan> buildDispatchABIPlan(const DispatchPair &pair) {
  llvm::SmallVector<support::RuntimeABIParameter, 4> rvvParameters;
  if (llvm::Error error =
          resolveCallableI32VAddABIParameters(pair.rvv, rvvParameters))
    return std::move(error);

  llvm::SmallVector<support::RuntimeABIParameter, 4> scalarParameters;
  if (llvm::Error error =
          resolveCallableI32VAddABIParameters(pair.scalar, scalarParameters))
    return std::move(error);

  for (auto [rvvParameter, scalarParameter] :
       llvm::zip(rvvParameters, scalarParameters)) {
    if (rvvParameter.role != scalarParameter.role ||
        rvvParameter.cType != scalarParameter.cType ||
        rvvParameter.ownership != scalarParameter.ownership)
      return makeDispatchError(
          pair.rvv.kernel,
          llvm::Twine("RVV and scalar callable runtime ABI role '") +
              support::stringifyRuntimeABIParameterRole(rvvParameter.role) +
              "' must agree on c type and ownership before dispatch export");
  }

  llvm::Expected<support::RuntimeABIParameter> guard =
      resolveDispatchAvailabilityGuardParameter(pair);
  if (!guard)
    return guard.takeError();

  DispatchABIPlan plan;
  support::appendI32VAddDispatchRuntimeABIParameters(plan.parameters,
                                                    rvvParameters, *guard);
  return plan;
}

llvm::Expected<DispatchPair> collectDispatchPair(mlir::ModuleOp module) {
  llvm::SmallVector<TargetArtifactCandidate, 4> candidates;
  if (llvm::Error error = collectTargetArtifactCandidates(module, candidates))
    return std::move(error);

  TargetArtifactExporterRegistry registry;
  if (llvm::Error error =
          rvv::registerRVVMicrokernelTargetExporters(registry))
    return std::move(error);
  if (llvm::Error error =
          scalar::registerScalarMicrokernelTargetExporters(registry))
    return std::move(error);

  const TargetArtifactCandidate *rvvCandidate = nullptr;
  const TargetArtifactCandidate *scalarCandidate = nullptr;
  for (const TargetArtifactCandidate &candidate : candidates) {
    if (hasCandidateShape(candidate, kRVVPluginName, kDispatchCaseRole,
                          kRVVRouteID, kRVVEmissionKind, kRVVRuntimeABI,
                          kRVVRuntimeABIKind, kRVVRuntimeABIName,
                          kRVVRuntimeGlueRole)) {
      if (rvvCandidate)
        return makeDispatchError(candidate.kernel,
                                 "requires exactly one supported RVV dispatch "
                                 "case callable route; found duplicate");
      if (llvm::Error error =
              validateRegisteredCallableRouteMetadata(candidate, registry))
        return std::move(error);
      rvvCandidate = &candidate;
      continue;
    }

    if (hasCandidateShape(candidate, kScalarPluginName, kDispatchFallbackRole,
                          kScalarRouteID, kScalarEmissionKind,
                          kScalarRuntimeABI, kScalarRuntimeABIKind,
                          kScalarRuntimeABIName, kScalarRuntimeGlueRole)) {
      if (scalarCandidate)
        return makeDispatchError(
            candidate.kernel,
            "requires exactly one supported scalar dispatch fallback callable "
            "route; found duplicate");
      if (llvm::Error error =
              validateRegisteredCallableRouteMetadata(candidate, registry))
        return std::move(error);
      scalarCandidate = &candidate;
      continue;
    }

    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("unsupported supported artifact candidate route '") +
            candidate.routeID + "' for RVV+scalar i32-vadd dispatch export");
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

  DispatchPair pair;
  pair.rvv = *rvvCandidate;
  pair.scalar = *scalarCandidate;
  llvm::Expected<DispatchABIPlan> abiPlan = buildDispatchABIPlan(pair);
  if (!abiPlan)
    return abiPlan.takeError();
  pair.abiPlan = std::move(*abiPlan);
  return pair;
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

std::string makeRVVFunctionName(const TargetArtifactCandidate &candidate) {
  KernelOp kernel = candidate.kernel;
  std::string name;
  llvm::raw_string_ostream stream(name);
  stream << "tcrv_rvv_i32_vadd_microkernel_"
         << sanitizeCIdentifierComponent(kernel.getSymName()) << "_"
         << sanitizeCIdentifierComponent(candidate.selectedVariant);
  stream.flush();
  return name;
}

std::string makeScalarFunctionName(const TargetArtifactCandidate &candidate) {
  KernelOp kernel = candidate.kernel;
  std::string name;
  llvm::raw_string_ostream stream(name);
  stream << "tcrv_scalar_i32_vadd_microkernel_"
         << sanitizeCIdentifierComponent(kernel.getSymName()) << "_"
         << sanitizeCIdentifierComponent(candidate.selectedVariant);
  stream.flush();
  return name;
}

std::string makeDispatcherFunctionName(const DispatchPair &pair) {
  KernelOp kernel = pair.rvv.kernel;
  std::string name;
  llvm::raw_string_ostream stream(name);
  stream << "tcrv_dispatch_i32_vadd_"
         << sanitizeCIdentifierComponent(kernel.getSymName());
  stream.flush();
  return name;
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

  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);

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
          capabilities.lookupByID(kRVVProbeCompileRunCapabilityID)) {
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
          capabilities.lookupByID(kRVVToolchainMarchCapabilityID)) {
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
          capabilities.lookupByID(kRVVProbeCompileRunCapabilityID)) {
    if (compileRun->isAvailable())
      if (llvm::Error error =
              mergeOptionalMABI(kernel, kSelectedMABIPropertyName,
                                compileRun->getProperty(
                                    kSelectedMABIPropertyName),
                                selectedMABI))
        return std::move(error);
  }

  if (const CapabilityDescriptor *mabi =
          capabilities.lookupByID(kRVVToolchainMABICapabilityID)) {
    if (mabi->isAvailable())
      if (llvm::Error error =
              mergeOptionalMABI(kernel, kValuePropertyName,
                                mabi->getProperty(kValuePropertyName),
                                selectedMABI))
        return std::move(error);
  }

  DispatchObjectCompileConfig config;
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
  printRequiredCapabilitiesComment(os, label, candidate.kernel,
                                   candidate.selectedVariant);
}

llvm::Error buildEmbeddedCallableSources(mlir::ModuleOp module,
                                         std::string &rvvSource,
                                         std::string &scalarSource) {
  llvm::raw_string_ostream rvvStream(rvvSource);
  if (llvm::Error error = rvv::exportRVVMicrokernelC(module, rvvStream))
    return error;
  rvvStream.flush();

  llvm::raw_string_ostream scalarStream(scalarSource);
  if (llvm::Error error = scalar::exportScalarMicrokernelC(module, scalarStream))
    return error;
  scalarStream.flush();
  return llvm::Error::success();
}

void printDispatcherFunction(llvm::raw_ostream &os,
                             llvm::StringRef dispatcherFunctionName,
                             llvm::StringRef rvvFunctionName,
                             llvm::StringRef scalarFunctionName,
                             llvm::ArrayRef<support::RuntimeABIParameter>
                                 parameters) {
  llvm::ArrayRef<support::RuntimeABIParameter> callableParameters =
      parameters.take_front(4);
  const support::RuntimeABIParameter &guardParameter = parameters[4];

  os << "void " << dispatcherFunctionName << "(";
  for (auto [index, parameter] : llvm::enumerate(parameters)) {
    if (index != 0)
      os << ", ";
    support::printRuntimeABIParameterCDeclaration(os, parameter);
  }
  os << ") {\n";
  os << "  if (" << guardParameter.cName << ") {\n";
  os << "    " << rvvFunctionName << "(";
  for (auto [index, parameter] : llvm::enumerate(callableParameters)) {
    if (index != 0)
      os << ", ";
    os << parameter.cName;
  }
  os << ");\n";
  os << "    return;\n";
  os << "  }\n";
  os << "  " << scalarFunctionName << "(";
  for (auto [index, parameter] : llvm::enumerate(callableParameters)) {
    if (index != 0)
      os << ", ";
    os << parameter.cName;
  }
  os << ");\n";
  os << "}\n";
}

void printDispatchSelfCheckHarness(llvm::raw_ostream &os,
                                   llvm::StringRef dispatcherFunctionName,
                                   llvm::StringRef guardParameterName) {
  os << "\n/* Explicit bounded self-check harness for RVV+scalar dispatch "
        "runtime invocation evidence. */\n";
  os << "/* Harness scope: calls the generated dispatcher once with "
     << guardParameterName << " = 0 and once with " << guardParameterName
     << " = 1. */\n";
  os << "/* Runtime n is a target/export-owned ABI parameter in this harness; "
        "descriptor-local element_count remains metadata only. */\n";
  os << "#include <stdio.h>\n\n";
  os << "static int " << dispatcherFunctionName
     << "_self_check_one(int " << guardParameterName << ") {\n";
  os << "  const int32_t lhs[16] = {0, 1, 2, 3, 4, 5, 6, 7, "
        "8, 9, 10, 11, 12, 13, 14, 15};\n";
  os << "  const int32_t rhs[16] = {31, 29, 23, 19, 17, 13, 11, 7, "
        "5, 3, 2, 1, -1, -3, -5, -7};\n";
  os << "  int32_t out[16] = {0};\n";
  os << "  " << dispatcherFunctionName
     << "(lhs, rhs, out, 16, " << guardParameterName << ");\n";
  os << "  for (size_t index = 0; index < 16; ++index) {\n";
  os << "    if (out[index] != lhs[index] + rhs[index])\n";
  os << "      return 1;\n";
  os << "  }\n";
  os << "  return 0;\n";
  os << "}\n\n";
  os << "int main(void) {\n";
  os << "  if (" << dispatcherFunctionName << "_self_check_one(0))\n";
  os << "    return 1;\n";
  os << "  if (" << dispatcherFunctionName << "_self_check_one(1))\n";
  os << "    return 2;\n";
  os << "  puts(\"tcrv_rvv_scalar_i32_vadd_dispatch_self_check_ok\");\n";
  os << "  return 0;\n";
  os << "}\n";
}

void printDispatchSource(const DispatchPair &pair, llvm::StringRef rvvSource,
                         llvm::StringRef scalarSource, bool includeSelfCheck,
                         llvm::raw_ostream &os) {
  KernelOp kernel = pair.rvv.kernel;
  std::string rvvFunctionName = makeRVVFunctionName(pair.rvv);
  std::string scalarFunctionName = makeScalarFunctionName(pair.scalar);
  std::string dispatcherFunctionName = makeDispatcherFunctionName(pair);
  llvm::ArrayRef<support::RuntimeABIParameter> dispatchParameters =
      pair.abiPlan.parameters;
  const support::RuntimeABIParameter &guardParameter =
      dispatchParameters.back();

  os << "/* TianChen-RV RVV+scalar host runtime dispatch C export. */\n";
  os << "/* Scope: one selected RVV i32-vadd dispatch case plus one scalar "
        "i32-vadd dispatch fallback. */\n";
  os << "/* Runtime guard: explicit host-provided " << guardParameter.cName
     << " parameter; no automatic hardware probe is generated. */\n";
  os << "/* selected_kernel: @" << kernel.getSymName() << " */\n";
  printCandidateMetadata(os, "rvv", pair.rvv);
  printCandidateMetadata(os, "scalar", pair.scalar);
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
  os << "/* dispatch_runtime_callable_abi: void " << dispatcherFunctionName
     << "(";
  for (auto [index, parameter] : llvm::enumerate(dispatchParameters)) {
    if (index != 0)
      os << ", ";
    support::printRuntimeABIParameterCDeclaration(os, parameter);
  }
  os << ") */\n\n";

  os << "/* Embedded selected RVV runtime-callable source artifact. */\n";
  os << rvvSource;
  if (!rvvSource.ends_with("\n"))
    os << "\n";
  os << "\n/* Embedded selected scalar runtime-callable fallback source "
        "artifact. */\n";
  os << scalarSource;
  if (!scalarSource.ends_with("\n"))
    os << "\n";
  os << "\n/* Host dispatcher over the two validated callable artifacts. */\n";
  printDispatcherFunction(os, dispatcherFunctionName, rvvFunctionName,
                          scalarFunctionName, dispatchParameters);
  if (includeSelfCheck)
    printDispatchSelfCheckHarness(os, dispatcherFunctionName,
                                  guardParameter.cName);
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
      "tcrv-rvv-scalar-dispatch-self-check", "c", fd, sourceFile.path);
  if (ec)
    return makeModuleDispatchObjectError(
        llvm::Twine("failed to create temporary C source for object export: ") +
        ec.message());

  llvm::raw_fd_ostream stream(fd, /*shouldClose=*/true);
  stream << source;
  stream.close();
  if (stream.has_error())
    return makeModuleDispatchObjectError(
        "failed to write generated self-check C source before object export");
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

std::string formatCompileCommand(llvm::StringRef selectedMarch,
                                 const std::optional<std::string> &selectedMABI) {
  std::string command;
  llvm::raw_string_ostream stream(command);
  stream << "clang -O2 -march=" << selectedMarch;
  if (selectedMABI)
    stream << " -mabi=" << *selectedMABI;
  stream << " -c <generated-self-check-source> -o <object-file>";
  stream.flush();
  return command;
}

llvm::Error compileSelfCheckSourceToObject(
    KernelOp kernel, llvm::StringRef source,
    const DispatchObjectCompileConfig &compileConfig, llvm::raw_ostream &os) {
  llvm::ErrorOr<std::string> clangPath =
      llvm::sys::findProgramByName("clang");
  if (!clangPath)
    return makeDispatchObjectError(
        kernel, "requires clang on PATH to compile the bounded self-check C "
                "source into an object file");

  TemporaryFile sourceFile;
  if (llvm::Error error = writeTempSource(source, sourceFile))
    return error;

  TemporaryFile objectFile;
  if (llvm::Error error =
          createTempFile("tcrv-rvv-scalar-dispatch-self-check", "o",
                         objectFile))
    return error;

  TemporaryFile stderrFile;
  if (llvm::Error error =
          createTempFile("tcrv-rvv-scalar-dispatch-self-check", "stderr",
                         stderrFile))
    return error;

  std::string marchArg = "-march=" + compileConfig.selectedMarch;
  std::string mabiArg;
  llvm::SmallVector<llvm::StringRef, 8> args;
  args.push_back(*clangPath);
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
                    formatCompileCommand(compileConfig.selectedMarch,
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

} // namespace

llvm::Error exportRVVScalarI32VAddDispatchC(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  llvm::Expected<DispatchPair> pair = collectDispatchPair(module);
  if (!pair)
    return pair.takeError();

  std::string rvvSource;
  std::string scalarSource;
  if (llvm::Error error =
          buildEmbeddedCallableSources(module, rvvSource, scalarSource))
    return error;

  std::string source;
  llvm::raw_string_ostream stream(source);
  printDispatchSource(*pair, rvvSource, scalarSource,
                      /*includeSelfCheck=*/false, stream);
  stream.flush();
  os << source;
  return llvm::Error::success();
}

llvm::Error exportRVVScalarI32VAddDispatchSelfCheckC(mlir::ModuleOp module,
                                                     llvm::raw_ostream &os) {
  llvm::Expected<DispatchPair> pair = collectDispatchPair(module);
  if (!pair)
    return pair.takeError();

  std::string rvvSource;
  std::string scalarSource;
  if (llvm::Error error =
          buildEmbeddedCallableSources(module, rvvSource, scalarSource))
    return error;

  std::string source;
  llvm::raw_string_ostream stream(source);
  printDispatchSource(*pair, rvvSource, scalarSource,
                      /*includeSelfCheck=*/true, stream);
  stream.flush();
  os << source;
  return llvm::Error::success();
}

llvm::Error
exportRVVScalarI32VAddDispatchSelfCheckObject(mlir::ModuleOp module,
                                              llvm::raw_ostream &os) {
  llvm::Expected<DispatchPair> pair = collectDispatchPair(module);
  if (!pair) {
    std::string message = llvm::toString(pair.takeError());
    return makeModuleDispatchObjectError(message);
  }

  llvm::Expected<DispatchObjectCompileConfig> compileConfig =
      buildDispatchObjectCompileConfig(*pair);
  if (!compileConfig)
    return compileConfig.takeError();

  std::string source;
  llvm::raw_string_ostream stream(source);
  if (llvm::Error error =
          exportRVVScalarI32VAddDispatchSelfCheckC(module, stream)) {
    std::string message = llvm::toString(std::move(error));
    return makeModuleDispatchObjectError(message);
  }
  stream.flush();
  if (source.empty())
    return makeDispatchObjectError(
        pair->rvv.kernel,
        "validated self-check C source must be non-empty before object export");

  return compileSelfCheckSourceToObject(pair->rvv.kernel, source,
                                        *compileConfig, os);
}

} // namespace tianchenrv::target::rvv_scalar
