#include "TianChenRV/Target/RVVScalarDispatch.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Support/RuntimeABICallablePlan.h"
#include "TianChenRV/Support/RuntimeABIContract.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"
#include "TianChenRV/Target/RVV/RVVMicrokernel.h"
#include "TianChenRV/Target/Scalar/ScalarMicrokernel.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

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
#include <optional>
#include <string>
#include <system_error>

namespace tianchenrv::target::rvv_scalar {
namespace {

using tianchenrv::target::TargetArtifactCandidate;
using tianchenrv::target::TargetArtifactCompositeExporter;
using tianchenrv::target::TargetArtifactExporter;
using tianchenrv::target::TargetArtifactExporterRegistry;
using tianchenrv::support::CapabilityDescriptor;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::DispatchOp;
using tianchenrv::tcrv::exec::DispatchCaseOp;
using tianchenrv::tcrv::exec::FallbackOp;
using tianchenrv::tcrv::exec::KernelOp;
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
constexpr llvm::StringLiteral kDispatchRuntimeABIParametersAttrName(
    "tcrv_rvv_scalar.dispatch_runtime_abi_parameters");
constexpr llvm::StringLiteral kRuntimeGuardAttrName("runtime_guard");

constexpr llvm::StringLiteral kI32VAddRVVRouteID(
    "tcrv-export-rvv-microkernel-c");
constexpr llvm::StringLiteral kI32VAddRVVEmissionKind(
    "rvv-explicit-i32-vadd-microkernel-c-source");
constexpr llvm::StringLiteral kI32VSubRVVRouteID(
    "tcrv-export-rvv-i32-vsub-microkernel-c");
constexpr llvm::StringLiteral kI32VSubRVVEmissionKind(
    "rvv-explicit-i32-vsub-microkernel-c-source");
constexpr llvm::StringLiteral kI32VAddRVVRuntimeABI(
    "rvv-i32-vadd-runtime-callable-c-abi.v1");
constexpr llvm::StringLiteral kI32VSubRVVRuntimeABI(
    "rvv-i32-vsub-runtime-callable-c-abi.v1");
constexpr llvm::StringLiteral kI32VAddRVVRuntimeABIName(
    "rvv-i32-vadd-runtime-callable-c-function.v1");
constexpr llvm::StringLiteral kI32VSubRVVRuntimeABIName(
    "rvv-i32-vsub-runtime-callable-c-function.v1");
constexpr llvm::StringLiteral kRVVRuntimeCallableABIKind(
    "rvv-runtime-callable-c-abi");
constexpr llvm::StringLiteral kI32VAddRVVRuntimeGlueRole(
    "runtime-callable-i32-vadd-function");
constexpr llvm::StringLiteral kI32VSubRVVRuntimeGlueRole(
    "runtime-callable-i32-vsub-function");

constexpr llvm::StringLiteral kI32VAddScalarRouteID(
    "tcrv-export-scalar-microkernel-c");
constexpr llvm::StringLiteral kI32VAddScalarEmissionKind(
    "scalar-explicit-i32-vadd-microkernel-c-source");
constexpr llvm::StringLiteral kI32VSubScalarRouteID(
    "tcrv-export-scalar-i32-vsub-microkernel-c");
constexpr llvm::StringLiteral kI32VSubScalarEmissionKind(
    "scalar-explicit-i32-vsub-microkernel-c-source");
constexpr llvm::StringLiteral kI32VAddScalarRuntimeABI(
    "scalar-i32-vadd-runtime-callable-c-abi.v1");
constexpr llvm::StringLiteral kI32VSubScalarRuntimeABI(
    "scalar-i32-vsub-runtime-callable-c-abi.v1");
constexpr llvm::StringLiteral kI32VAddScalarRuntimeABIName(
    "scalar-i32-vadd-runtime-callable-c-function.v1");
constexpr llvm::StringLiteral kI32VSubScalarRuntimeABIName(
    "scalar-i32-vsub-runtime-callable-c-function.v1");
constexpr llvm::StringLiteral kScalarRuntimeCallableABIKind(
    "scalar-runtime-callable-c-abi");
constexpr llvm::StringLiteral kI32VAddScalarRuntimeGlueRole(
    "runtime-callable-i32-vadd-fallback-function");
constexpr llvm::StringLiteral kI32VSubScalarRuntimeGlueRole(
    "runtime-callable-i32-vsub-fallback-function");

constexpr llvm::StringLiteral kDispatchTargetOwner(
    "rvv-scalar-dispatch-target");
constexpr llvm::StringLiteral kI32VAddDispatchSourceRouteID(
    "tcrv-export-rvv-scalar-i32-vadd-dispatch-c");
constexpr llvm::StringLiteral kI32VAddDispatchHeaderRouteID(
    "tcrv-export-rvv-scalar-i32-vadd-dispatch-header");
constexpr llvm::StringLiteral kI32VAddDispatchObjectRouteID(
    "tcrv-export-rvv-scalar-i32-vadd-dispatch-object");
constexpr llvm::StringLiteral kI32VSubDispatchSourceRouteID(
    "tcrv-export-rvv-scalar-i32-vsub-dispatch-c");
constexpr llvm::StringLiteral kI32VSubDispatchHeaderRouteID(
    "tcrv-export-rvv-scalar-i32-vsub-dispatch-header");
constexpr llvm::StringLiteral kI32VSubDispatchObjectRouteID(
    "tcrv-export-rvv-scalar-i32-vsub-dispatch-object");
constexpr llvm::StringLiteral kDispatchRuntimeABIKind(
    "rvv-scalar-dispatch-runtime-callable-c-abi");
constexpr llvm::StringLiteral kI32VAddDispatchRuntimeABIName(
    "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1");
constexpr llvm::StringLiteral kI32VSubDispatchRuntimeABIName(
    "rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1");
constexpr llvm::StringLiteral kI32VAddDispatchExternalABIComponentGroup(
    "rvv-scalar-i32-vadd-dispatch-external-abi.v1");
constexpr llvm::StringLiteral kI32VSubDispatchExternalABIComponentGroup(
    "rvv-scalar-i32-vsub-dispatch-external-abi.v1");
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

struct DispatchIRLink {
  DispatchOp dispatch;
  DispatchCaseOp selectedRVVCase;
  FallbackOp fallback;
  VariantOp fallbackVariant;
  std::string fallbackTarget;
};

enum class DispatchI32FamilyKind {
  Add,
  Sub,
};

struct DispatchI32FamilySpec {
  DispatchI32FamilyKind kind;
  llvm::StringRef diagnosticName;
  llvm::StringRef operationNoun;
  llvm::StringRef functionStem;
  llvm::StringRef headerGuardStem;
  llvm::StringRef cOperator;
  llvm::StringRef selfCheckSuccessMarker;
  llvm::StringRef rvvRouteID;
  llvm::StringRef rvvEmissionKind;
  llvm::StringRef rvvRuntimeABI;
  llvm::StringRef rvvRuntimeABIKind;
  llvm::StringRef rvvRuntimeABIName;
  llvm::StringRef rvvRuntimeGlueRole;
  llvm::StringRef scalarRouteID;
  llvm::StringRef scalarEmissionKind;
  llvm::StringRef scalarRuntimeABI;
  llvm::StringRef scalarRuntimeABIKind;
  llvm::StringRef scalarRuntimeABIName;
  llvm::StringRef scalarRuntimeGlueRole;
  llvm::StringRef dispatchSourceRouteID;
  llvm::StringRef dispatchHeaderRouteID;
  llvm::StringRef dispatchObjectRouteID;
  llvm::StringRef dispatchRuntimeABIKind;
  llvm::StringRef dispatchRuntimeABIName;
  llvm::StringRef dispatchExternalABIComponentGroup;
};

struct DispatchPair {
  const DispatchI32FamilySpec *family = nullptr;
  TargetArtifactCandidate rvv;
  TargetArtifactCandidate scalar;
  DispatchIRLink irLink;
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
  stream << "TianChen-RV RVV+scalar i32 add/sub dispatch C export failed";
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
      llvm::Twine("TianChen-RV RVV+scalar i32 add/sub dispatch C export "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error makeDispatchObjectError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV RVV+scalar i32 add/sub dispatch object export "
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
      llvm::Twine("TianChen-RV RVV+scalar i32 add/sub dispatch object export "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error makeModuleDispatchHeaderError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV+scalar i32 add/sub dispatch header export "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

const DispatchI32FamilySpec &getI32VAddDispatchFamilySpec() {
  static const DispatchI32FamilySpec spec{
      DispatchI32FamilyKind::Add,
      "i32-vadd",
      "i32 vector-add",
      "i32_vadd",
      "I32_VADD",
      "+",
      "tcrv_rvv_scalar_i32_vadd_dispatch_self_check_ok",
      kI32VAddRVVRouteID,
      kI32VAddRVVEmissionKind,
      kI32VAddRVVRuntimeABI,
      kRVVRuntimeCallableABIKind,
      kI32VAddRVVRuntimeABIName,
      kI32VAddRVVRuntimeGlueRole,
      kI32VAddScalarRouteID,
      kI32VAddScalarEmissionKind,
      kI32VAddScalarRuntimeABI,
      kScalarRuntimeCallableABIKind,
      kI32VAddScalarRuntimeABIName,
      kI32VAddScalarRuntimeGlueRole,
      kI32VAddDispatchSourceRouteID,
      kI32VAddDispatchHeaderRouteID,
      kI32VAddDispatchObjectRouteID,
      kDispatchRuntimeABIKind,
      kI32VAddDispatchRuntimeABIName,
      kI32VAddDispatchExternalABIComponentGroup};
  return spec;
}

const DispatchI32FamilySpec &getI32VSubDispatchFamilySpec() {
  static const DispatchI32FamilySpec spec{
      DispatchI32FamilyKind::Sub,
      "i32-vsub",
      "i32 vector-subtract",
      "i32_vsub",
      "I32_VSUB",
      "-",
      "tcrv_rvv_scalar_i32_vsub_dispatch_self_check_ok",
      kI32VSubRVVRouteID,
      kI32VSubRVVEmissionKind,
      kI32VSubRVVRuntimeABI,
      kRVVRuntimeCallableABIKind,
      kI32VSubRVVRuntimeABIName,
      kI32VSubRVVRuntimeGlueRole,
      kI32VSubScalarRouteID,
      kI32VSubScalarEmissionKind,
      kI32VSubScalarRuntimeABI,
      kScalarRuntimeCallableABIKind,
      kI32VSubScalarRuntimeABIName,
      kI32VSubScalarRuntimeGlueRole,
      kI32VSubDispatchSourceRouteID,
      kI32VSubDispatchHeaderRouteID,
      kI32VSubDispatchObjectRouteID,
      kDispatchRuntimeABIKind,
      kI32VSubDispatchRuntimeABIName,
      kI32VSubDispatchExternalABIComponentGroup};
  return spec;
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

bool isRVVCallableCandidateForFamily(const TargetArtifactCandidate &candidate,
                                     const DispatchI32FamilySpec &family) {
  return hasCandidateShape(candidate, kRVVPluginName, kDispatchCaseRole,
                           family.rvvRouteID, family.rvvEmissionKind,
                           family.rvvRuntimeABI, family.rvvRuntimeABIKind,
                           family.rvvRuntimeABIName,
                           family.rvvRuntimeGlueRole);
}

bool isScalarCallableCandidateForFamily(
    const TargetArtifactCandidate &candidate,
    const DispatchI32FamilySpec &family) {
  return hasCandidateShape(candidate, kScalarPluginName,
                           kDispatchFallbackRole, family.scalarRouteID,
                           family.scalarEmissionKind,
                           family.scalarRuntimeABI,
                           family.scalarRuntimeABIKind,
                           family.scalarRuntimeABIName,
                           family.scalarRuntimeGlueRole);
}

const DispatchI32FamilySpec *
getRVVCallableCandidateFamily(const TargetArtifactCandidate &candidate) {
  if (isRVVCallableCandidateForFamily(candidate,
                                      getI32VAddDispatchFamilySpec()))
    return &getI32VAddDispatchFamilySpec();
  if (isRVVCallableCandidateForFamily(candidate,
                                      getI32VSubDispatchFamilySpec()))
    return &getI32VSubDispatchFamilySpec();
  return nullptr;
}

const DispatchI32FamilySpec *
getScalarCallableCandidateFamily(const TargetArtifactCandidate &candidate) {
  if (isScalarCallableCandidateForFamily(candidate,
                                         getI32VAddDispatchFamilySpec()))
    return &getI32VAddDispatchFamilySpec();
  if (isScalarCallableCandidateForFamily(candidate,
                                         getI32VSubDispatchFamilySpec()))
    return &getI32VSubDispatchFamilySpec();
  return nullptr;
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

  return validateTargetArtifactCandidateAgainstExporter(candidate, *exporter);
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
      support::getI32VAddDispatchAvailabilityGuardParamSpec(/*cName=*/"");
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
  llvm::Expected<support::I32VAddCallableABIPlan> callablePlan =
      support::buildI32VAddCallableABIPlan(pair.rvv.kernel);
  if (!callablePlan) {
    std::string message = llvm::toString(callablePlan.takeError());
    return makeDispatchError(pair.rvv.kernel, message);
  }

  if (llvm::Error error = support::validateI32VAddCallableABIParameterMirror(
          pair.rvv.kernel, pair.rvv.runtimeABIParameters,
          callablePlan->parameters, "selected RVV callable artifact route")) {
    std::string message = llvm::toString(std::move(error));
    return makeDispatchError(pair.rvv.kernel, message);
  }
  if (llvm::Error error = support::validateI32VAddCallableABIParameterMirror(
          pair.scalar.kernel, pair.scalar.runtimeABIParameters,
          callablePlan->parameters,
          "selected scalar callable artifact route")) {
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
  runtimeParamSpecs.push_back(
      support::getI32VAddRuntimeElementCountParamSpec((*runtimeCount)->cName));
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
      support::makeI32VAddDispatchAvailabilityGuard(guard->cName);
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

  support::appendI32VAddDispatchRuntimeABIParameters(
      plan.parameters, callablePlan->parameters, *guard);
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
    if (const DispatchI32FamilySpec *family =
            getRVVCallableCandidateFamily(candidate)) {
      if (rvvCandidate)
        return makeDispatchError(candidate.kernel,
                                 "requires exactly one supported RVV dispatch "
                                 "case callable route; found duplicate");
      if (llvm::Error error =
              validateRegisteredCallableRouteMetadata(candidate, registry))
        return std::move(error);
      rvvCandidate = &candidate;
      rvvFamily = family;
      continue;
    }

    if (const DispatchI32FamilySpec *family =
            getScalarCallableCandidateFamily(candidate)) {
      if (scalarCandidate)
        return makeDispatchError(
            candidate.kernel,
            "requires exactly one supported scalar dispatch fallback callable "
            "route; found duplicate");
      if (llvm::Error error =
              validateRegisteredCallableRouteMetadata(candidate, registry))
        return std::move(error);
      scalarCandidate = &candidate;
      scalarFamily = family;
      continue;
    }

    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("unsupported supported artifact candidate route '") +
            candidate.routeID +
            "' for RVV+scalar i32 add/sub dispatch export");
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
  llvm::Expected<DispatchIRLink> irLink = resolveDispatchIRLinkForPair(pair);
  if (!irLink)
    return irLink.takeError();
  pair.irLink = std::move(*irLink);
  llvm::Expected<DispatchABIPlan> abiPlan = buildDispatchABIPlan(pair);
  if (!abiPlan)
    return abiPlan.takeError();
  pair.abiPlan = std::move(*abiPlan);
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
    if (const DispatchI32FamilySpec *family =
            getRVVCallableCandidateFamily(candidate)) {
      rvvCandidate = &candidate;
      rvvFamily = family;
      continue;
    }
    if (const DispatchI32FamilySpec *family =
            getScalarCallableCandidateFamily(candidate)) {
      scalarCandidate = &candidate;
      scalarFamily = family;
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

llvm::Error validateRVVScalarI32VAddDispatchCandidates(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  llvm::Expected<DispatchPair> pair =
      collectDispatchPairFromCandidates(candidates);
  if (!pair)
    return pair.takeError();
  return llvm::Error::success();
}

llvm::Expected<llvm::SmallVector<support::RuntimeABIParameter, 5>>
resolveRVVScalarI32VAddDispatchRuntimeABIParameters(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  llvm::Expected<DispatchPair> pair =
      collectDispatchPairFromCandidates(candidates);
  if (!pair)
    return pair.takeError();
  return std::move(pair->abiPlan.parameters);
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

std::string makeRVVFunctionName(const DispatchPair &pair) {
  const TargetArtifactCandidate &candidate = pair.rvv;
  KernelOp kernel = candidate.kernel;
  std::string name;
  llvm::raw_string_ostream stream(name);
  stream << "tcrv_rvv_" << pair.family->functionStem << "_microkernel_"
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
  stream << "tcrv_scalar_" << pair.family->functionStem << "_microkernel_"
         << sanitizeCIdentifierComponent(kernel.getSymName()) << "_"
         << sanitizeCIdentifierComponent(candidate.selectedVariant);
  stream.flush();
  return name;
}

std::string makeDispatcherFunctionName(const DispatchPair &pair) {
  KernelOp kernel = pair.rvv.kernel;
  std::string name;
  llvm::raw_string_ostream stream(name);
  stream << "tcrv_dispatch_" << pair.family->functionStem << "_"
         << sanitizeCIdentifierComponent(kernel.getSymName());
  stream.flush();
  return name;
}

std::string makeDispatchHeaderIncludeGuard(const DispatchPair &pair) {
  KernelOp kernel = pair.rvv.kernel;
  std::string guard = "TIANCHENRV_RVV_SCALAR_";
  guard += pair.family->headerGuardStem;
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
                                 parameters,
                             const DispatchRuntimeABIParameterBindings
                                 &bindings) {
  os << "void " << dispatcherFunctionName << "(";
  for (auto [index, parameter] : llvm::enumerate(parameters)) {
    if (index != 0)
      os << ", ";
    support::printRuntimeABIParameterCDeclaration(os, parameter);
  }
  os << ") {\n";
  os << "  if (" << bindings.dispatchAvailabilityGuard->cName << ") {\n";
  os << "    " << rvvFunctionName << "(";
  const support::RuntimeABIParameter *callableParameters[] = {
      bindings.lhs, bindings.rhs, bindings.out, bindings.runtimeElementCount};
  for (auto [index, parameter] : llvm::enumerate(callableParameters)) {
    if (index != 0)
      os << ", ";
    os << parameter->cName;
  }
  os << ");\n";
  os << "    return;\n";
  os << "  }\n";
  os << "  " << scalarFunctionName << "(";
  for (auto [index, parameter] : llvm::enumerate(callableParameters)) {
    if (index != 0)
      os << ", ";
    os << parameter->cName;
  }
  os << ");\n";
  os << "}\n";
}

llvm::Error printDispatchHeader(const DispatchPair &pair,
                                llvm::raw_ostream &os) {
  llvm::Expected<DispatchRuntimeABIParameterBindings> bindings =
      bindDispatchRuntimeABIParametersByRole(
          pair.rvv.kernel, pair.abiPlan.parameters,
          "RVV+scalar dispatch header ABI plan");
  if (!bindings)
    return bindings.takeError();

  std::string includeGuard = makeDispatchHeaderIncludeGuard(pair);
  std::string dispatcherFunctionName = makeDispatcherFunctionName(pair);

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
                                   const DispatchI32FamilySpec &family,
                                   llvm::StringRef dispatcherFunctionName,
                                   llvm::StringRef runtimeElementCountName,
                                   llvm::StringRef guardParameterName) {
  os << "\n/* Explicit bounded self-check harness for RVV+scalar dispatch "
        "runtime invocation evidence. */\n";
  os << "/* Harness scope: calls the generated dispatcher with explicit "
     << runtimeElementCountName << " values 7 and 16 for "
     << guardParameterName << " = 0 and " << guardParameterName
     << " = 1. */\n";
  os << "/* Runtime element count is a target/export-owned ABI parameter in "
        "this harness; "
        "descriptor-local element_count remains metadata only. */\n";
  os << "#include <stdio.h>\n\n";
  os << "static int " << dispatcherFunctionName
     << "_self_check_one(size_t runtime_n, int " << guardParameterName
     << ") {\n";
  os << "  enum { kCapacity = 32 };\n";
  os << "  int32_t lhs[kCapacity];\n";
  os << "  int32_t rhs[kCapacity];\n";
  os << "  int32_t out[kCapacity];\n";
  os << "  for (size_t index = 0; index < (size_t)kCapacity; ++index) {\n";
  os << "    lhs[index] = (int32_t)index;\n";
  os << "    rhs[index] = (int32_t)(31 - (int)index);\n";
  os << "    out[index] = -12345;\n";
  os << "  }\n";
  os << "  " << dispatcherFunctionName
     << "(lhs, rhs, out, runtime_n, " << guardParameterName << ");\n";
  os << "  for (size_t index = 0; index < runtime_n; ++index) {\n";
  os << "    if (out[index] != lhs[index] " << family.cOperator
     << " rhs[index])\n";
  os << "      return 1;\n";
  os << "  }\n";
  os << "  for (size_t index = runtime_n; index < (size_t)kCapacity; "
        "++index) {\n";
  os << "    if (out[index] != -12345)\n";
  os << "      return 2;\n";
  os << "  }\n";
  os << "  return 0;\n";
  os << "}\n\n";
  os << "int main(void) {\n";
  os << "  if (" << dispatcherFunctionName << "_self_check_one(7, 0))\n";
  os << "    return 1;\n";
  os << "  if (" << dispatcherFunctionName << "_self_check_one(16, 0))\n";
  os << "    return 2;\n";
  os << "  if (" << dispatcherFunctionName << "_self_check_one(7, 1))\n";
  os << "    return 3;\n";
  os << "  if (" << dispatcherFunctionName << "_self_check_one(16, 1))\n";
  os << "    return 4;\n";
  os << "  puts(\"" << family.selfCheckSuccessMarker
     << " runtime_counts=7,16 branches=scalar_and_rvv\");\n";
  os << "  return 0;\n";
  os << "}\n";
}

llvm::Error printDispatchSource(const DispatchPair &pair,
                                llvm::StringRef rvvSource,
                                llvm::StringRef scalarSource,
                                bool includeSelfCheck, llvm::raw_ostream &os) {
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

  os << "/* TianChen-RV RVV+scalar host runtime dispatch C export. */\n";
  os << "/* Scope: one selected RVV " << pair.family->diagnosticName
     << " dispatch case plus one scalar " << pair.family->diagnosticName
     << " dispatch fallback. */\n";
  os << "/* Runtime guard: explicit host-provided "
     << bindings->dispatchAvailabilityGuard->cName
     << " parameter; no automatic hardware probe is generated. */\n";
  os << "/* selected_kernel: @" << kernel.getSymName() << " */\n";
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
                          scalarFunctionName, dispatchParameters, *bindings);
  if (includeSelfCheck)
    printDispatchSelfCheckHarness(os, *pair.family, dispatcherFunctionName,
                                  bindings->runtimeElementCount->cName,
                                  bindings->dispatchAvailabilityGuard->cName);
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
    return makeDispatchObjectError(
        kernel, "requires clang on PATH to compile the bounded dispatch C "
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
  if (llvm::Error error = printDispatchSource(*pair, rvvSource, scalarSource,
                                              /*includeSelfCheck=*/false,
                                              stream))
    return error;
  stream.flush();
  os << source;
  return llvm::Error::success();
}

llvm::Error exportRVVScalarI32VAddDispatchHeader(mlir::ModuleOp module,
                                                 llvm::raw_ostream &os) {
  llvm::Expected<DispatchPair> pair = collectDispatchPair(module);
  if (!pair) {
    std::string message = llvm::toString(pair.takeError());
    return makeModuleDispatchHeaderError(message);
  }

  std::string rvvSource;
  std::string scalarSource;
  if (llvm::Error error =
          buildEmbeddedCallableSources(module, rvvSource, scalarSource)) {
    std::string message = llvm::toString(std::move(error));
    return makeModuleDispatchHeaderError(message);
  }

  llvm::Expected<DispatchObjectCompileConfig> compileConfig =
      buildDispatchObjectCompileConfig(*pair);
  if (!compileConfig) {
    std::string message = llvm::toString(compileConfig.takeError());
    return makeModuleDispatchHeaderError(message);
  }

  if (llvm::Error error = printDispatchHeader(*pair, os)) {
    std::string message = llvm::toString(std::move(error));
    return makeModuleDispatchHeaderError(message);
  }
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
  if (llvm::Error error = printDispatchSource(*pair, rvvSource, scalarSource,
                                              /*includeSelfCheck=*/true,
                                              stream))
    return error;
  stream.flush();
  os << source;
  return llvm::Error::success();
}

llvm::Error exportRVVScalarI32VAddDispatchObject(mlir::ModuleOp module,
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
  if (llvm::Error error = exportRVVScalarI32VAddDispatchC(module, stream)) {
    std::string message = llvm::toString(std::move(error));
    return makeModuleDispatchObjectError(message);
  }
  stream.flush();
  if (source.empty())
    return makeDispatchObjectError(
        pair->rvv.kernel,
        "validated library-style dispatch C source must be non-empty before "
        "object export");

  return compileGeneratedDispatchSourceToObject(pair->rvv.kernel, source,
                                                *compileConfig, os);
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

  return compileGeneratedDispatchSourceToObject(pair->rvv.kernel, source,
                                                *compileConfig, os);
}

llvm::Error registerRVVScalarDispatchFamilyTargetExporters(
    TargetArtifactExporterRegistry &registry,
    const DispatchI32FamilySpec &family) {
  bool directHelperRoute = family.kind == DispatchI32FamilyKind::Add;
  TargetArtifactCompositeMatchFn matchFn =
      family.kind == DispatchI32FamilyKind::Add
          ? matchRVVScalarI32VAddDispatchCandidates
          : matchRVVScalarI32VSubDispatchCandidates;
  if (llvm::Error error =
          registry.registerCompositeExporter(TargetArtifactCompositeExporter(
              family.dispatchSourceRouteID, kRuntimeCallableCSourceArtifactKind,
              matchFn, exportRVVScalarI32VAddDispatchC, kDispatchTargetOwner,
              family.dispatchRuntimeABIKind, family.dispatchRuntimeABIName,
              resolveRVVScalarI32VAddDispatchRuntimeABIParameters,
              directHelperRoute, family.dispatchExternalABIComponentGroup,
              family.dispatchRuntimeABIName,
              validateRVVScalarI32VAddDispatchCandidates)))
    return error;

  if (llvm::Error error =
          registry.registerCompositeExporter(TargetArtifactCompositeExporter(
              family.dispatchHeaderRouteID, kRuntimeCallableCHeaderArtifactKind,
              matchFn, exportRVVScalarI32VAddDispatchHeader,
              kDispatchTargetOwner,
              family.dispatchRuntimeABIKind, family.dispatchRuntimeABIName,
              resolveRVVScalarI32VAddDispatchRuntimeABIParameters,
              directHelperRoute, family.dispatchExternalABIComponentGroup,
              family.dispatchRuntimeABIName,
              validateRVVScalarI32VAddDispatchCandidates)))
    return error;

  return registry.registerCompositeExporter(TargetArtifactCompositeExporter(
      family.dispatchObjectRouteID, kRiscvELFRelocatableObjectArtifactKind,
      matchFn, exportRVVScalarI32VAddDispatchObject, kDispatchTargetOwner,
      family.dispatchRuntimeABIKind, family.dispatchRuntimeABIName,
      resolveRVVScalarI32VAddDispatchRuntimeABIParameters, directHelperRoute,
      family.dispatchExternalABIComponentGroup, family.dispatchRuntimeABIName,
      validateRVVScalarI32VAddDispatchCandidates));
}

llvm::Error registerRVVScalarDispatchTargetExporters(
    TargetArtifactExporterRegistry &registry) {
  if (llvm::Error error = registerRVVScalarDispatchFamilyTargetExporters(
          registry, getI32VAddDispatchFamilySpec()))
    return error;

  return registerRVVScalarDispatchFamilyTargetExporters(
      registry, getI32VSubDispatchFamilySpec());
}

} // namespace tianchenrv::target::rvv_scalar
