#include "TianChenRV/Target/Scalar/ScalarMicrokernel.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVLowerToEmitC.h"
#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/Scalar/IR/ScalarDialect.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Support/RuntimeABICallablePlan.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"
#include "TianChenRV/Target/RVVScalarBinaryFamily.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
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

namespace tianchenrv::target::scalar {
namespace {

namespace execDiagnostic = tianchenrv::tcrv::exec::diagnostic;

using tianchenrv::support::CapabilityDescriptor;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::conversion::emitc::TCRVEmitCCallOpaqueResult;
using tianchenrv::conversion::emitc::TCRVEmitCCallOpaqueStep;
using tianchenrv::conversion::emitc::TCRVEmitCLowerableInterface;
using tianchenrv::conversion::emitc::TCRVEmitCLowerableOpInterface;
using tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute;
using tianchenrv::conversion::emitc::
    TCRVLowerToEmitCSourceOptions;
using tianchenrv::conversion::emitc::
    TCRVLowerToEmitCSourceResult;
using tianchenrv::conversion::emitc::
    lowerTCRVEmitCLowerableToEmitCSource;
using tianchenrv::tcrv::exec::DiagnosticOp;
using tianchenrv::tcrv::exec::DispatchCaseOp;
using tianchenrv::tcrv::exec::DispatchOp;
using tianchenrv::tcrv::exec::FallbackOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::MemWindowOp;
using tianchenrv::tcrv::exec::RuntimeParamOp;
using tianchenrv::tcrv::exec::VariantOp;
using tianchenrv::tcrv::scalar::I32VAddMicrokernelOp;
using tianchenrv::tcrv::scalar::I32VMulMicrokernelOp;
using tianchenrv::tcrv::scalar::I32VSubMicrokernelOp;
using tianchenrv::tcrv::scalar::I64VAddMicrokernelOp;
using tianchenrv::tcrv::scalar::I64VMulMicrokernelOp;
using tianchenrv::tcrv::scalar::I64VSubMicrokernelOp;
using tianchenrv::tcrv::scalar::LoweringBoundaryOp;
using tianchenrv::target::rvv::RVVBinaryArithmeticKind;
using tianchenrv::target::rvv::RVVBinaryDTypeKind;

constexpr llvm::StringLiteral kScalarPluginName("scalar-plugin");
constexpr llvm::StringLiteral kScalarFallbackCapabilityID("scalar.fallback");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kElementCountAttrName("element_count");
constexpr llvm::StringLiteral kFallbackRoleAttrName("fallback_role");
constexpr llvm::StringLiteral kMetadataOnlyStatusValue("metadata-only");
constexpr llvm::StringLiteral kDirectVariantRole("direct variant");
constexpr llvm::StringLiteral kDispatchCaseRole("dispatch case");
constexpr llvm::StringLiteral kDispatchFallbackRole("dispatch fallback");
constexpr llvm::StringLiteral kRV64CapabilityID("rv64");
constexpr llvm::StringLiteral kArchitecturePropertyName("architecture");
constexpr llvm::StringLiteral kArchPropertyName("arch");
constexpr llvm::StringLiteral kRiscvToolchainMarchCapabilityID(
    "riscv.toolchain.march");
constexpr llvm::StringLiteral kRiscvToolchainMABICapabilityID(
    "riscv.toolchain.mabi");
constexpr llvm::StringLiteral kRVVProbeCompileRunCapabilityID(
    "rvv.probe.compile_run");
constexpr llvm::StringLiteral kRVVToolchainMarchCapabilityID(
    "rvv.toolchain.march");
constexpr llvm::StringLiteral kRVVToolchainMABICapabilityID(
    "rvv.toolchain.mabi");
constexpr llvm::StringLiteral kSelectedMarchPropertyName("selected_march");
constexpr llvm::StringLiteral kSelectedMABIPropertyName("selected_mabi");
constexpr llvm::StringLiteral kValuePropertyName("value");
constexpr llvm::StringLiteral kMicrokernelArtifactKind(
    "runtime-callable-c-source");
constexpr llvm::StringLiteral kMicrokernelHeaderArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kMicrokernelObjectArtifactKind(
    "riscv-elf-relocatable-object");
constexpr llvm::StringLiteral kScalarElementCountAttrName(
    "tcrv_scalar.element_count");
constexpr llvm::StringLiteral kDirectCSourceRouteDeletedReason(
    "scalar runtime-callable direct C semantic exporter was deleted; rebuild "
    "requires a materialized MLIR EmitC module source route");

using ScalarI32MicrokernelFamilySpec =
    tianchenrv::target::rvv_scalar::ScalarBinaryMicrokernelRecord;

const ScalarI32MicrokernelFamilySpec &getI32VAddFamilySpec() {
  return tianchenrv::target::rvv_scalar::getI32VAddFamilyRegistrationRecord().scalar;
}

const ScalarI32MicrokernelFamilySpec &getI32VSubFamilySpec() {
  return tianchenrv::target::rvv_scalar::getI32VSubFamilyRegistrationRecord().scalar;
}

const ScalarI32MicrokernelFamilySpec &getI32VMulFamilySpec() {
  return tianchenrv::target::rvv_scalar::getI32VMulFamilyRegistrationRecord().scalar;
}

const ScalarI32MicrokernelFamilySpec &getI64VAddFamilySpec() {
  return tianchenrv::target::rvv_scalar::getI64VAddFamilyRegistrationRecord().scalar;
}

const ScalarI32MicrokernelFamilySpec &getI64VSubFamilySpec() {
  return tianchenrv::target::rvv_scalar::getI64VSubFamilyRegistrationRecord().scalar;
}

const ScalarI32MicrokernelFamilySpec &getI64VMulFamilySpec() {
  return tianchenrv::target::rvv_scalar::getI64VMulFamilyRegistrationRecord().scalar;
}

llvm::ArrayRef<const ScalarI32MicrokernelFamilySpec *>
getScalarMicrokernelFamilySpecs() {
  static const ScalarI32MicrokernelFamilySpec *families[] = {
      &getI32VAddFamilySpec(), &getI32VSubFamilySpec(),
      &getI32VMulFamilySpec(), &getI64VAddFamilySpec(),
      &getI64VSubFamilySpec(), &getI64VMulFamilySpec()};
  return llvm::ArrayRef(families);
}

const ScalarI32MicrokernelFamilySpec *
getScalarI32MicrokernelFamilyForOp(mlir::Operation *op) {
  if (llvm::isa_and_nonnull<I32VAddMicrokernelOp>(op))
    return &getI32VAddFamilySpec();
  if (llvm::isa_and_nonnull<I32VSubMicrokernelOp>(op))
    return &getI32VSubFamilySpec();
  if (llvm::isa_and_nonnull<I32VMulMicrokernelOp>(op))
    return &getI32VMulFamilySpec();
  if (llvm::isa_and_nonnull<I64VAddMicrokernelOp>(op))
    return &getI64VAddFamilySpec();
  if (llvm::isa_and_nonnull<I64VSubMicrokernelOp>(op))
    return &getI64VSubFamilySpec();
  if (llvm::isa_and_nonnull<I64VMulMicrokernelOp>(op))
    return &getI64VMulFamilySpec();
  return nullptr;
}

const ScalarI32MicrokernelFamilySpec *
getScalarI32MicrokernelFamilyForSourceRoute(llvm::StringRef routeID) {
  if (routeID == getI32VAddFamilySpec().routeID)
    return &getI32VAddFamilySpec();
  if (routeID == getI32VSubFamilySpec().routeID)
    return &getI32VSubFamilySpec();
  if (routeID == getI32VMulFamilySpec().routeID)
    return &getI32VMulFamilySpec();
  if (routeID == getI64VAddFamilySpec().routeID)
    return &getI64VAddFamilySpec();
  if (routeID == getI64VSubFamilySpec().routeID)
    return &getI64VSubFamilySpec();
  if (routeID == getI64VMulFamilySpec().routeID)
    return &getI64VMulFamilySpec();
  return nullptr;
}

bool candidateMatchesScalarMicrokernelFamily(
    const TargetArtifactCandidate &candidate,
    const ScalarI32MicrokernelFamilySpec &family) {
  return candidate.origin == kScalarPluginName &&
         candidate.routeID == family.routeID &&
         candidate.emissionKind == family.emissionKind &&
         candidate.artifactKind == kMicrokernelArtifactKind &&
         candidate.runtimeABI == family.runtimeABI &&
         candidate.runtimeABIKind == family.runtimeABIKind &&
         candidate.runtimeABIName == family.runtimeABIName &&
         candidate.runtimeGlueRole == family.runtimeGlueRole;
}

struct SelectedPath {
  VariantOp variant;
  std::string role;
};

struct ScalarMicrokernelRecord {
  const ScalarI32MicrokernelFamilySpec *family = nullptr;
  std::string kernelSymbol;
  std::string variantSymbol;
  std::string role;
  std::string fallbackRole;
  llvm::SmallVector<std::string, 4> requiredCapabilities;
  llvm::SmallVector<support::RuntimeABIParameter, 4> runtimeABIParameters;
  llvm::SmallVector<MemWindowOp, 3> bufferWindows;
  RuntimeParamOp runtimeElementCountParam;
  std::int64_t elementCount = 0;
  std::string emitcSourceOpName;
  std::string emitcSourceOpRole;
  std::string emitcSourceOpInterface;
};

struct ScalarCallableABIPlan {
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters;
  llvm::SmallVector<MemWindowOp, 3> bufferWindows;
  RuntimeParamOp runtimeElementCountParam;
};

struct ScalarObjectCompileConfig {
  std::string targetTriple;
  std::string selectedMarch;
  std::optional<std::string> selectedMABI;
};

struct TemporaryFile {
  llvm::SmallString<128> path;

  ~TemporaryFile() {
    if (!path.empty())
      llvm::sys::fs::remove(path);
  }

  llvm::StringRef get() const { return llvm::StringRef(path); }
};

VariantOp getPathVariant(const SelectedPath &path) {
  return const_cast<SelectedPath &>(path).variant;
}

llvm::StringRef getPathVariantSymbol(const SelectedPath &path) {
  return getPathVariant(path).getSymName();
}

mlir::Operation *getPathVariantOperation(const SelectedPath &path) {
  return getPathVariant(path).getOperation();
}

llvm::Error makeMicrokernelError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV scalar microkernel C export failed";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

llvm::Error makeModuleMicrokernelError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV scalar microkernel C export failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error makeMicrokernelObjectError(llvm::StringRef kernelSymbol,
                                       llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV scalar microkernel object export failed";
  if (!kernelSymbol.empty())
    stream << " for kernel @" << kernelSymbol;
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

llvm::Error makeModuleMicrokernelObjectError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV scalar microkernel object export failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool hasKernelBody(KernelOp kernel) {
  return kernel && !kernel.getBody().empty();
}

mlir::StringAttr getStringAttr(mlir::Operation *op, llvm::StringRef name) {
  return op ? op->getAttrOfType<mlir::StringAttr>(name) : mlir::StringAttr();
}

mlir::StringAttr getDirectSymbolName(mlir::Operation &op) {
  return op.getAttrOfType<mlir::StringAttr>(
      mlir::SymbolTable::getSymbolAttrName());
}

bool isSelectedMarker(DiagnosticOp diagnostic) {
  auto reason =
      diagnostic->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kReasonAttrName);
  return reason && reason.getValue() == execDiagnostic::kSelectedReasonValue;
}

bool isEmissionPlanDiagnostic(DiagnosticOp diagnostic) {
  auto reason =
      diagnostic->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kReasonAttrName);
  return reason && execDiagnostic::isEmissionPlanReason(reason.getValue());
}

std::string makePathKey(llvm::StringRef variant, llvm::StringRef role) {
  std::string key;
  llvm::raw_string_ostream stream(key);
  stream << variant << "\n" << role;
  stream.flush();
  return key;
}

bool containsForbiddenText(llvm::StringRef value) {
  std::string lower = value.lower();
  llvm::StringRef normalized(lower);
  return normalized.contains("password") || normalized.contains("passwd") ||
         normalized.contains("token") || normalized.contains("secret") ||
         normalized.contains("private key") ||
         normalized.contains("authorization:") ||
         normalized.contains("api_key") || normalized.contains("access_key");
}

llvm::Error validateBoundedText(KernelOp kernel, llvm::StringRef fieldName,
                                llvm::StringRef value) {
  constexpr std::size_t kMaxTextLength = 512;
  if (value.empty() || value.size() > kMaxTextLength)
    return makeMicrokernelError(
        kernel, llvm::Twine(fieldName) +
                    " must be bounded non-empty single-line metadata");

  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      return makeMicrokernelError(
          kernel, llvm::Twine(fieldName) +
                      " must be bounded non-empty single-line metadata");
    if (byte < 0x20 && character != '\t')
      return makeMicrokernelError(
          kernel, llvm::Twine(fieldName) +
                      " must be bounded non-empty single-line metadata");
  }

  if (value.contains("/*") || value.contains("*/"))
    return makeMicrokernelError(
        kernel, llvm::Twine(fieldName) +
                    " must not contain C comment delimiter text");

  if (containsForbiddenText(value))
    return makeMicrokernelError(
        kernel, llvm::Twine(fieldName) +
                    " must not contain secret-like or raw credential text");
  return llvm::Error::success();
}

llvm::Error collectRuntimeABIParameters(
    KernelOp kernel, DiagnosticOp diagnostic,
    llvm::SmallVectorImpl<support::RuntimeABIParameter> &out) {
  auto parameters = diagnostic->getAttrOfType<mlir::ArrayAttr>(
      execDiagnostic::kRuntimeABIParametersAttrName);
  if (!parameters || parameters.empty())
    return makeMicrokernelError(
        kernel, "supported scalar microkernel emission-plan diagnostic "
                "requires non-empty runtime_abi_parameters metadata");

  llvm::StringSet<> seenNames;
  llvm::StringSet<> seenRoles;
  for (auto [index, attr] : llvm::enumerate(parameters)) {
    auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(attr);
    if (!dict)
      return makeMicrokernelError(
          kernel, llvm::Twine("runtime_abi_parameters[") +
                      llvm::Twine(index) + "] must be a dictionary attribute");

    auto cName = dict.getAs<mlir::StringAttr>(
        support::kRuntimeABIParameterCNameAttrName);
    auto cType = dict.getAs<mlir::StringAttr>(
        support::kRuntimeABIParameterCTypeAttrName);
    auto role = dict.getAs<mlir::StringAttr>(
        support::kRuntimeABIParameterRoleAttrName);
    auto ownership = dict.getAs<mlir::StringAttr>(
        support::kRuntimeABIParameterOwnershipAttrName);
    if (!cName || cName.getValue().trim().empty())
      return makeMicrokernelError(
          kernel, llvm::Twine("runtime_abi_parameters[") +
                      llvm::Twine(index) + "] requires non-empty c_name");
    if (!cType || cType.getValue().trim().empty())
      return makeMicrokernelError(
          kernel, llvm::Twine("runtime_abi_parameters[") +
                      llvm::Twine(index) + "] requires non-empty c_type");
    if (!role || role.getValue().trim().empty())
      return makeMicrokernelError(
          kernel, llvm::Twine("runtime_abi_parameters[") +
                      llvm::Twine(index) + "] requires non-empty role");
    if (!ownership || ownership.getValue().trim().empty())
      return makeMicrokernelError(
          kernel, llvm::Twine("runtime_abi_parameters[") +
                      llvm::Twine(index) + "] requires non-empty ownership");

    llvm::StringRef cNameValue = cName.getValue().trim();
    llvm::StringRef cTypeValue = cType.getValue().trim();
    llvm::StringRef roleValue = role.getValue().trim();
    llvm::StringRef ownershipValue = ownership.getValue().trim();
    if (llvm::Error error =
            validateBoundedText(kernel, "runtime ABI parameter c_name",
                                cNameValue))
      return error;
    if (llvm::Error error =
            validateBoundedText(kernel, "runtime ABI parameter c_type",
                                cTypeValue))
      return error;
    if (llvm::Error error =
            validateBoundedText(kernel, "runtime ABI parameter role",
                                roleValue))
      return error;
    if (llvm::Error error =
            validateBoundedText(kernel, "runtime ABI parameter ownership",
                                ownershipValue))
      return error;
    if (!seenNames.insert(cNameValue).second)
      return makeMicrokernelError(
          kernel, llvm::Twine("duplicate runtime ABI parameter c_name '") +
                      cNameValue + "'");
    if (!seenRoles.insert(roleValue).second)
      return makeMicrokernelError(
          kernel, llvm::Twine("duplicate runtime ABI parameter role '") +
                      roleValue + "'");

    std::optional<support::RuntimeABIParameterRole> parsedRole =
        support::symbolizeRuntimeABIParameterRole(roleValue);
    if (!parsedRole)
      return makeMicrokernelError(
          kernel, llvm::Twine("unsupported runtime ABI parameter role '") +
                      roleValue + "'");
    std::optional<support::RuntimeABIParameterOwnership> parsedOwnership =
        support::symbolizeRuntimeABIParameterOwnership(ownershipValue);
    if (!parsedOwnership)
      return makeMicrokernelError(
          kernel, llvm::Twine("unsupported runtime ABI parameter ownership '") +
                      ownershipValue + "'");

    out.push_back(support::RuntimeABIParameter(cNameValue, cTypeValue,
                                               *parsedRole, *parsedOwnership));
  }

  return llvm::Error::success();
}

llvm::Error requireSafeStringAttr(KernelOp kernel, mlir::Operation *op,
                                  llvm::StringRef attrName,
                                  llvm::StringRef owner,
                                  std::string &out);

llvm::Error validateScalarCallableABIParameterMirror(
    KernelOp kernel,
    llvm::ArrayRef<support::RuntimeABIParameter> metadataParameters,
    llvm::ArrayRef<support::RuntimeABIParameter> irBackedParameters,
    llvm::StringRef metadataSource,
    const ScalarI32MicrokernelFamilySpec &family) {
  if (!family.rvvFamily)
    return makeMicrokernelError(
        kernel, llvm::Twine("scalar callable ABI mirror validation for ") +
                    family.microkernelOpName +
                    " requires finite RVV binary family metadata");

  return support::validateFiniteBinaryCallableABIParameterMirror(
      kernel, metadataParameters, irBackedParameters, metadataSource,
      target::rvv::getRVVBinaryRuntimeABIContract(*family.rvvFamily));
}

llvm::Error validateEmissionPlanParameterMirror(
    KernelOp kernel, const SelectedPath &path,
    const ScalarI32MicrokernelFamilySpec &family,
    llvm::ArrayRef<support::RuntimeABIParameter> irBackedParameters) {
  llvm::SmallVector<DiagnosticOp, 2> matches;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto diagnostic = llvm::dyn_cast<DiagnosticOp>(op);
    if (!diagnostic || !isEmissionPlanDiagnostic(diagnostic))
      continue;

    auto target = diagnostic->getAttrOfType<mlir::FlatSymbolRefAttr>(
        execDiagnostic::kTargetAttrName);
    auto role = diagnostic->getAttrOfType<mlir::StringAttr>(
        execDiagnostic::kRoleAttrName);
    if (!target || !role)
      continue;
    if (target.getValue() == getPathVariantSymbol(path) &&
        role.getValue() == path.role)
      matches.push_back(diagnostic);
  }

  if (matches.empty())
    return llvm::Error::success();
  if (matches.size() > 1)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected scalar path @") +
                    getPathVariantSymbol(path) + " as " + path.role +
                    " has duplicate emission-plan diagnostics");

  DiagnosticOp diagnostic = matches.front();
  std::string origin;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kOriginAttrName,
                                "emission-plan diagnostic", origin))
    return error;
  if (origin != kScalarPluginName)
    return makeMicrokernelError(
        kernel, llvm::Twine("emission-plan origin '") + origin +
                    "' does not match scalar microkernel origin "
                    "'scalar-plugin'");

  std::string status;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kStatusAttrName,
                                "emission-plan diagnostic", status))
    return error;
  if (status != execDiagnostic::kEmissionPlanSupportedStatusValue)
    return makeMicrokernelError(
        kernel, "scalar microkernel export requires a supported "
                "emission-plan diagnostic when runtime ABI metadata is "
                "present");

  std::string routeID;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kLoweringPipelineAttrName,
                                "supported emission-plan route", routeID))
    return error;
  if (routeID != family.routeID)
    return makeMicrokernelError(
        kernel, llvm::Twine("supported emission-plan route '") + routeID +
                    "' does not match scalar microkernel route '" +
                    family.routeID + "'");

  std::string emissionKind;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kEmissionKindAttrName,
                                "supported emission-plan route", emissionKind))
    return error;
  if (emissionKind != family.emissionKind)
    return makeMicrokernelError(
        kernel, llvm::Twine("supported emission-plan emission_kind '") +
                    emissionKind +
                    "' does not match scalar microkernel emission_kind '" +
                    family.emissionKind + "'");

  std::string runtimeABI;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kRuntimeABIAttrName,
                                "supported emission-plan route", runtimeABI))
    return error;
  if (runtimeABI != family.runtimeABI)
    return makeMicrokernelError(
        kernel, llvm::Twine("supported emission-plan runtime_abi '") +
                    runtimeABI +
                    "' does not match scalar microkernel runtime_abi '" +
                    family.runtimeABI + "'");

  std::string runtimeABIKind;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kRuntimeABIKindAttrName,
                                "emission-plan diagnostic", runtimeABIKind))
    return error;
  if (runtimeABIKind != family.runtimeABIKind)
    return makeMicrokernelError(
        kernel, llvm::Twine("supported emission-plan runtime_abi_kind '") +
                    runtimeABIKind +
                    "' does not match scalar microkernel runtime_abi_kind '" +
                    family.runtimeABIKind + "'");

  std::string runtimeABIName;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kRuntimeABINameAttrName,
                                "emission-plan diagnostic", runtimeABIName))
    return error;
  if (runtimeABIName != family.runtimeABIName)
    return makeMicrokernelError(
        kernel, llvm::Twine("supported emission-plan runtime_abi_name '") +
                    runtimeABIName +
                    "' does not match scalar microkernel runtime_abi_name '" +
                    family.runtimeABIName + "'");

  std::string runtimeGlueRole;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kRuntimeGlueRoleAttrName,
                                "emission-plan diagnostic", runtimeGlueRole))
    return error;
  if (runtimeGlueRole != family.runtimeGlueRole)
    return makeMicrokernelError(
        kernel, llvm::Twine("supported emission-plan runtime_glue_role '") +
                    runtimeGlueRole +
                    "' does not match scalar microkernel runtime_glue_role '" +
                    family.runtimeGlueRole + "'");

  llvm::SmallVector<support::RuntimeABIParameter, 5> planParameters;
  if (llvm::Error error =
          collectRuntimeABIParameters(kernel, diagnostic, planParameters))
    return error;

  return validateScalarCallableABIParameterMirror(
      kernel, planParameters, irBackedParameters,
      "supported scalar microkernel emission-plan", family);
}

llvm::Error requireSafeStringAttr(KernelOp kernel, mlir::Operation *op,
                                  llvm::StringRef attrName,
                                  llvm::StringRef context,
                                  std::string &out) {
  auto attr = getStringAttr(op, attrName);
  if (!attr || attr.getValue().trim().empty())
    return makeMicrokernelError(kernel, llvm::Twine(context) +
                                            " requires non-empty string "
                                            "attribute '" +
                                            attrName + "'");
  llvm::StringRef value = attr.getValue().trim();
  if (llvm::Error error = validateBoundedText(kernel, attrName, value))
    return error;
  out = value.str();
  return llvm::Error::success();
}

void collectDirectKernelSymbols(
    KernelOp kernel, llvm::StringMap<VariantOp> &directVariants,
    llvm::StringMap<mlir::Operation *> &directSymbols) {
  if (!hasKernelBody(kernel))
    return;

  for (mlir::Operation &op : kernel.getBody().front()) {
    mlir::StringAttr symbolName = getDirectSymbolName(op);
    if (!symbolName)
      continue;

    directSymbols.try_emplace(symbolName.getValue(), &op);
    if (auto variant = llvm::dyn_cast<VariantOp>(op))
      directVariants.try_emplace(symbolName.getValue(), variant);
  }
}

llvm::Error resolveDirectVariant(
    KernelOp kernel, llvm::StringRef symbol, llvm::StringRef context,
    const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    VariantOp &variant) {
  if (symbol.trim().empty())
    return makeMicrokernelError(kernel, llvm::Twine(context) +
                                            " has an empty selected variant "
                                            "symbol reference");

  auto variantIt = directVariants.find(symbol);
  if (variantIt != directVariants.end()) {
    variant = variantIt->getValue();
    return llvm::Error::success();
  }

  if (directSymbols.count(symbol))
    return makeMicrokernelError(kernel, llvm::Twine(context) + " target @" +
                                            symbol +
                                            " resolves to a direct sibling "
                                            "symbol that is not a "
                                            "tcrv.exec.variant");

  return makeMicrokernelError(kernel, llvm::Twine(context) + " target @" +
                                          symbol +
                                          " does not resolve to a direct "
                                          "sibling tcrv.exec.variant");
}

llvm::Error collectDispatchSelectedPaths(
    KernelOp kernel, DispatchOp dispatch,
    const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    llvm::SmallVectorImpl<SelectedPath> &paths) {
  if (!dispatch || dispatch.getBody().empty())
    return makeMicrokernelError(
        kernel, "selected dispatch requires a materialized body block");

  bool sawCase = false;
  bool sawFallback = false;
  llvm::StringSet<> seenTargets;
  for (mlir::Operation &op : dispatch.getBody().front()) {
    if (auto dispatchCase = llvm::dyn_cast<DispatchCaseOp>(op)) {
      auto target =
          dispatchCase->getAttrOfType<mlir::FlatSymbolRefAttr>(
              execDiagnostic::kTargetAttrName);
      if (!target)
        return makeMicrokernelError(
            kernel, "dispatch case requires a selected variant target");

      VariantOp variant;
      if (llvm::Error error = resolveDirectVariant(
              kernel, target.getValue(), "dispatch case", directVariants,
              directSymbols, variant))
        return error;
      if (!seenTargets.insert(target.getValue()).second)
        return makeMicrokernelError(
            kernel, llvm::Twine("duplicate selected dispatch target @") +
                        target.getValue());

      sawCase = true;
      paths.push_back(SelectedPath{variant, kDispatchCaseRole.str()});
      continue;
    }

    if (auto fallbackOp = llvm::dyn_cast<FallbackOp>(op)) {
      auto target =
          fallbackOp->getAttrOfType<mlir::FlatSymbolRefAttr>(
              execDiagnostic::kTargetAttrName);
      if (!target)
        return makeMicrokernelError(
            kernel, "dispatch fallback requires a selected variant target");

      VariantOp variant;
      if (llvm::Error error = resolveDirectVariant(
              kernel, target.getValue(), "dispatch fallback", directVariants,
              directSymbols, variant))
        return error;
      if (!seenTargets.insert(target.getValue()).second)
        return makeMicrokernelError(
            kernel, llvm::Twine("duplicate selected dispatch target @") +
                        target.getValue());
      if (sawFallback)
        return makeMicrokernelError(
            kernel, "selected dispatch requires exactly one fallback target");

      sawFallback = true;
      paths.push_back(SelectedPath{variant, kDispatchFallbackRole.str()});
      continue;
    }

    return makeMicrokernelError(
        kernel, llvm::Twine("unexpected operation '") +
                    op.getName().getStringRef() +
                    "' in selected dispatch surface");
  }

  if (!sawCase)
    return makeMicrokernelError(kernel,
                                "selected dispatch requires at least one case");
  if (!sawFallback)
    return makeMicrokernelError(kernel,
                                "selected dispatch requires one fallback target");
  return llvm::Error::success();
}

llvm::Error collectSelectedPaths(
    KernelOp kernel, const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    llvm::SmallVectorImpl<SelectedPath> &paths) {
  if (!hasKernelBody(kernel))
    return makeMicrokernelError(kernel,
                                "requires kernel to have a materialized body "
                                "block");

  llvm::SmallVector<DispatchOp, 2> dispatches;
  llvm::SmallVector<DiagnosticOp, 2> markers;
  for (mlir::Operation &op : kernel.getBody().front()) {
    if (auto dispatch = llvm::dyn_cast<DispatchOp>(op))
      dispatches.push_back(dispatch);
    if (auto diagnostic = llvm::dyn_cast<DiagnosticOp>(op))
      if (isSelectedMarker(diagnostic))
        markers.push_back(diagnostic);
  }

  if (dispatches.size() > 1)
    return makeMicrokernelError(
        kernel, "requires exactly one selected dispatch surface; found "
                "multiple direct tcrv.exec.dispatch operations");
  if (!dispatches.empty() && !markers.empty())
    return makeMicrokernelError(
        kernel, "requires one selected path surface; found both dispatch and "
                "selected diagnostic marker");

  if (!dispatches.empty())
    return collectDispatchSelectedPaths(kernel, dispatches.front(),
                                        directVariants, directSymbols, paths);

  if (markers.size() > 1)
    return makeMicrokernelError(
        kernel, "requires at most one selected diagnostic marker when no "
                "dispatch is present");
  if (markers.empty())
    return makeMicrokernelError(
        kernel, "requires a selected path surface before exporting a scalar "
                "microkernel");

  DiagnosticOp marker = markers.front();
  auto selectionKind =
      marker->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kSelectionKindAttrName);
  if (!selectionKind || selectionKind.getValue().trim().empty())
    return makeMicrokernelError(
        kernel, "selected diagnostic marker requires non-empty "
                "selection_kind");
  if (selectionKind.getValue() != execDiagnostic::kStaticSelectionKindValue &&
      selectionKind.getValue() !=
          execDiagnostic::kFallbackOnlySelectionKindValue)
    return makeMicrokernelError(
        kernel, llvm::Twine("unsupported selected diagnostic marker "
                            "selection_kind '") +
                    selectionKind.getValue() + "'");

  auto target =
      marker->getAttrOfType<mlir::FlatSymbolRefAttr>(
          execDiagnostic::kTargetAttrName);
  if (!target)
    return makeMicrokernelError(
        kernel, "selected diagnostic marker requires a selected variant "
                "target");

  VariantOp variant;
  if (llvm::Error error = resolveDirectVariant(
          kernel, target.getValue(), "selected diagnostic marker",
          directVariants, directSymbols, variant))
    return error;

  paths.push_back(SelectedPath{variant, kDirectVariantRole.str()});
  return llvm::Error::success();
}

bool arrayAttrsEqual(mlir::ArrayAttr lhs, mlir::ArrayAttr rhs) {
  if (!lhs || !rhs || lhs.size() != rhs.size())
    return false;
  for (auto [lhsAttr, rhsAttr] : llvm::zip(lhs, rhs))
    if (lhsAttr != rhsAttr)
      return false;
  return true;
}

llvm::Error validateRequiredCapabilities(
    KernelOp kernel, VariantOp variant, const TargetCapabilitySet &capabilities,
    llvm::SmallVectorImpl<std::string> &out) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr || requiresAttr.empty())
    return makeMicrokernelError(
        kernel, llvm::Twine("selected scalar variant @") +
                    variant.getSymName() +
                    " requires non-empty structured 'requires' metadata");

  bool requiresScalarFallback = false;
  for (mlir::Attribute attr : requiresAttr) {
    auto symbol = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
    if (!symbol || symbol.getValue().trim().empty())
      return makeMicrokernelError(
          kernel, llvm::Twine("selected scalar variant @") +
                      variant.getSymName() +
                      " requires only non-empty capability symbol references");

    const CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbol.getValue());
    if (!capability)
      return makeMicrokernelError(
          kernel, llvm::Twine("selected scalar variant @") +
                      variant.getSymName() + " requires unknown capability @" +
                      symbol.getValue());
    if (!capability->isAvailable())
      return makeMicrokernelError(
          kernel, llvm::Twine("selected scalar variant @") +
                      variant.getSymName() +
                      " requires unavailable capability @" +
                      symbol.getValue());
    if (capability->satisfiesID(kScalarFallbackCapabilityID))
      requiresScalarFallback = true;
    out.push_back(symbol.getValue().str());
  }

  if (!requiresScalarFallback)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected scalar variant @") +
                    variant.getSymName() +
                    " must require capability id 'scalar.fallback'");

  return llvm::Error::success();
}

bool startsWithRV64March(llvm::StringRef march) {
  std::string lower = march.trim().lower();
  return llvm::StringRef(lower).starts_with("rv64");
}

llvm::Expected<std::string>
getRequiredScalarObjectTargetTriple(KernelOp kernel,
                                    const TargetCapabilitySet &capabilities) {
  const CapabilityDescriptor *rv64Capability =
      capabilities.lookupProviderByID(kRV64CapabilityID);
  if (!rv64Capability || !rv64Capability->isAvailable())
    return makeMicrokernelError(
        kernel,
        "scalar microkernel object export requires an available structured "
        "capability provider for capability id 'rv64'");

  llvm::SmallVector<llvm::StringRef, 2> architectureValues;
  llvm::StringRef architecture =
      rv64Capability->getProperty(kArchitecturePropertyName).trim();
  if (!architecture.empty())
    architectureValues.push_back(architecture);
  llvm::StringRef arch = rv64Capability->getProperty(kArchPropertyName).trim();
  if (!arch.empty())
    architectureValues.push_back(arch);

  if (architectureValues.empty())
    return makeMicrokernelError(
        kernel,
        "scalar microkernel object export requires rv64 capability metadata "
        "with architecture or arch property 'riscv64'");

  bool sawRiscv64 = false;
  for (llvm::StringRef value : architectureValues) {
    if (llvm::Error error =
            validateBoundedText(kernel, "rv64 architecture metadata", value))
      return std::move(error);
    if (value == "riscv64")
      sawRiscv64 = true;
  }

  if (!sawRiscv64)
    return makeMicrokernelError(
        kernel,
        "scalar microkernel object export requires rv64 architecture "
        "metadata 'riscv64'");
  return "riscv64";
}

llvm::Error mergeScalarObjectCompileText(KernelOp kernel,
                                         llvm::StringRef fieldName,
                                         llvm::StringRef value,
                                         std::optional<std::string> &out) {
  value = value.trim();
  if (value.empty())
    return llvm::Error::success();
  if (llvm::Error error = validateBoundedText(kernel, fieldName, value))
    return error;
  if (out && *out != value)
    return makeMicrokernelError(
        kernel, llvm::Twine("conflicting scalar object ") + fieldName +
                    " capability metadata");
  out = value.str();
  return llvm::Error::success();
}

llvm::Expected<std::string>
getRequiredScalarObjectSelectedMarch(
    KernelOp kernel, const TargetCapabilitySet &capabilities) {
  std::optional<std::string> selectedMarch;
  auto mergeMarchFromCapability =
      [&](llvm::StringRef capabilityID,
          llvm::StringRef propertyName) -> llvm::Error {
    if (selectedMarch)
      return llvm::Error::success();
    const CapabilityDescriptor *capability =
        capabilities.lookupProviderByID(capabilityID);
    if (!capability || !capability->isAvailable())
      return llvm::Error::success();
    return mergeScalarObjectCompileText(
        kernel, propertyName, capability->getProperty(propertyName),
        selectedMarch);
  };

  if (llvm::Error error = mergeMarchFromCapability(
          kRiscvToolchainMarchCapabilityID, kValuePropertyName))
    return std::move(error);
  if (llvm::Error error = mergeMarchFromCapability(
          kRVVProbeCompileRunCapabilityID, kSelectedMarchPropertyName))
    return std::move(error);
  if (llvm::Error error = mergeMarchFromCapability(
          kRVVToolchainMarchCapabilityID, kValuePropertyName))
    return std::move(error);

  if (!selectedMarch)
    return makeMicrokernelError(
        kernel, "scalar microkernel object export requires available "
                "selected_march metadata from capability id "
                "'riscv.toolchain.march', 'rvv.probe.compile_run', or "
                "'rvv.toolchain.march'");
  if (!startsWithRV64March(*selectedMarch))
    return makeMicrokernelError(
        kernel,
        llvm::Twine("scalar microkernel object export selected_march '") +
            *selectedMarch + "' must describe an rv64 RISC-V target");
  return *selectedMarch;
}

llvm::Error getOptionalScalarObjectSelectedMABI(
    KernelOp kernel, const TargetCapabilitySet &capabilities,
    std::optional<std::string> &out) {
  auto mergeMABIFromCapability =
      [&](llvm::StringRef capabilityID,
          llvm::StringRef propertyName) -> llvm::Error {
    if (out)
      return llvm::Error::success();
    const CapabilityDescriptor *capability =
        capabilities.lookupProviderByID(capabilityID);
    if (!capability || !capability->isAvailable())
      return llvm::Error::success();
    return mergeScalarObjectCompileText(kernel, propertyName,
                                        capability->getProperty(propertyName),
                                        out);
  };

  if (llvm::Error error = mergeMABIFromCapability(
          kRiscvToolchainMABICapabilityID, kValuePropertyName))
    return error;
  if (llvm::Error error = mergeMABIFromCapability(
          kRVVProbeCompileRunCapabilityID, kSelectedMABIPropertyName))
    return error;
  if (llvm::Error error = mergeMABIFromCapability(kRVVToolchainMABICapabilityID,
                                                  kValuePropertyName))
    return error;
  return llvm::Error::success();
}

llvm::Expected<ScalarObjectCompileConfig>
buildScalarObjectCompileConfig(KernelOp kernel) {
  llvm::Expected<TargetCapabilitySet> capabilities =
      TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities)
    return capabilities.takeError();

  llvm::Expected<std::string> targetTriple =
      getRequiredScalarObjectTargetTriple(kernel, *capabilities);
  if (!targetTriple)
    return targetTriple.takeError();

  llvm::Expected<std::string> selectedMarch =
      getRequiredScalarObjectSelectedMarch(kernel, *capabilities);
  if (!selectedMarch)
    return selectedMarch.takeError();

  std::optional<std::string> selectedMABI;
  if (llvm::Error error =
          getOptionalScalarObjectSelectedMABI(kernel, *capabilities,
                                             selectedMABI))
    return std::move(error);

  ScalarObjectCompileConfig config;
  config.targetTriple = std::move(*targetTriple);
  config.selectedMarch = std::move(*selectedMarch);
  config.selectedMABI = std::move(selectedMABI);
  return config;
}

llvm::Error validateBoundaryForPath(KernelOp kernel, const SelectedPath &path,
                                    LoweringBoundaryOp boundary) {
  if (!boundary)
    return makeMicrokernelError(
        kernel, "requires a matching tcrv_scalar.lowering_boundary");
  if (boundary->getParentOp() != kernel.getOperation())
    return makeMicrokernelError(
        kernel, "matching tcrv_scalar.lowering_boundary must be a direct "
                "child of the selected kernel");

  std::string sourceKernel;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, boundary.getOperation(),
                                kSourceKernelAttrName,
                                "tcrv_scalar.lowering_boundary",
                                sourceKernel))
    return error;
  if (sourceKernel != kernel.getSymName())
    return makeMicrokernelError(
        kernel, llvm::Twine("tcrv_scalar.lowering_boundary source_kernel '") +
                    sourceKernel + "' does not match selected kernel @" +
                    kernel.getSymName());

  std::string origin;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, boundary.getOperation(),
                                execDiagnostic::kOriginAttrName,
                                "tcrv_scalar.lowering_boundary", origin))
    return error;
  if (origin != kScalarPluginName)
    return makeMicrokernelError(
        kernel, "tcrv_scalar.lowering_boundary origin must be 'scalar-plugin'");

  std::string role;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, boundary.getOperation(),
                                execDiagnostic::kRoleAttrName,
                                "tcrv_scalar.lowering_boundary", role))
    return error;
  if (role != path.role)
    return makeMicrokernelError(
        kernel, llvm::Twine("tcrv_scalar.lowering_boundary role '") + role +
                    "' does not match selected scalar path role '" + path.role +
                    "'");

  std::string status;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, boundary.getOperation(),
                                execDiagnostic::kStatusAttrName,
                                "tcrv_scalar.lowering_boundary", status))
    return error;
  if (status != kMetadataOnlyStatusValue)
    return makeMicrokernelError(
        kernel, "tcrv_scalar.lowering_boundary status must remain "
                "'metadata-only' for this scalar source-export slice");

  auto selectedVariant =
      boundary->getAttrOfType<mlir::FlatSymbolRefAttr>(
          kSelectedVariantAttrName);
  if (!selectedVariant ||
      selectedVariant.getValue() != getPathVariantSymbol(path))
    return makeMicrokernelError(
        kernel, "tcrv_scalar.lowering_boundary selected_variant does not "
                "match the selected scalar path");

  auto boundaryCapabilities =
      boundary->getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName);
  auto variantRequires =
      path.variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!arrayAttrsEqual(boundaryCapabilities, variantRequires))
    return makeMicrokernelError(
        kernel, "tcrv_scalar.lowering_boundary required_capabilities must "
                "match selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error findAndValidateBoundary(
    KernelOp kernel, const SelectedPath &path,
    const llvm::StringSet<> &selectedScalarPathKeys,
    LoweringBoundaryOp &matchedBoundary) {
  unsigned matches = 0;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto boundary = llvm::dyn_cast<LoweringBoundaryOp>(op);
    if (!boundary)
      continue;

    auto selectedVariant =
        boundary->getAttrOfType<mlir::FlatSymbolRefAttr>(
            kSelectedVariantAttrName);
    auto role =
        boundary->getAttrOfType<mlir::StringAttr>(execDiagnostic::kRoleAttrName);
    if (!selectedVariant || !role)
      continue;

    std::string key = makePathKey(selectedVariant.getValue(), role.getValue());
    if (!selectedScalarPathKeys.count(key))
      return makeMicrokernelError(
          kernel, llvm::Twine("stale tcrv_scalar.lowering_boundary for @") +
                      selectedVariant.getValue() + " as " + role.getValue() +
                      " is not selected by the current scalar microkernel "
                      "surface");

    if (selectedVariant.getValue() == getPathVariantSymbol(path) &&
        role.getValue() == path.role) {
      ++matches;
      matchedBoundary = boundary;
    }
  }

  if (matches == 0)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected scalar path @") +
                    getPathVariantSymbol(path) + " as " + path.role +
                    " requires exactly one matching "
                    "tcrv_scalar.lowering_boundary");
  if (matches > 1)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected scalar path @") +
                    getPathVariantSymbol(path) + " as " + path.role +
                    " has duplicate tcrv_scalar.lowering_boundary metadata");

  return validateBoundaryForPath(kernel, path, matchedBoundary);
}

llvm::Error validateMicrokernelForPath(
    KernelOp kernel, const SelectedPath &path, mlir::Operation *microkernel,
    const ScalarI32MicrokernelFamilySpec &family,
    std::int64_t &elementCount) {
  if (!microkernel)
    return makeMicrokernelError(
        kernel, llvm::Twine("requires a matching ") +
                    family.microkernelOpName);
  if (microkernel->getParentOp() != kernel.getOperation())
    return makeMicrokernelError(
        kernel, llvm::Twine("matching ") + family.microkernelOpName +
                    " must be a direct child of the selected kernel");

  std::string sourceKernel;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, microkernel, kSourceKernelAttrName,
                                family.microkernelOpName, sourceKernel))
    return error;
  if (sourceKernel != kernel.getSymName())
    return makeMicrokernelError(
        kernel,
        llvm::Twine(family.microkernelOpName) + " source_kernel '" +
            sourceKernel + "' does not match selected kernel @" +
            kernel.getSymName());

  std::string origin;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, microkernel,
                                execDiagnostic::kOriginAttrName,
                                family.microkernelOpName, origin))
    return error;
  if (origin != kScalarPluginName)
    return makeMicrokernelError(
        kernel, llvm::Twine(family.microkernelOpName) +
                    " origin must be 'scalar-plugin'");

  std::string role;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, microkernel,
                                execDiagnostic::kRoleAttrName,
                                family.microkernelOpName, role))
    return error;
  if (role != path.role)
    return makeMicrokernelError(
        kernel, llvm::Twine(family.microkernelOpName) + " role '" + role +
                    "' does not match selected scalar path role '" + path.role +
                    "'");

  auto selectedVariant =
      microkernel->getAttrOfType<mlir::FlatSymbolRefAttr>(
          kSelectedVariantAttrName);
  if (!selectedVariant ||
      selectedVariant.getValue() != getPathVariantSymbol(path))
    return makeMicrokernelError(
        kernel, llvm::Twine(family.microkernelOpName) +
                    " selected_variant does not match the selected scalar "
                    "path");

  auto microkernelCapabilities =
      microkernel->getAttrOfType<mlir::ArrayAttr>(
          kRequiredCapabilitiesAttrName);
  auto variantRequires =
      path.variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!arrayAttrsEqual(microkernelCapabilities, variantRequires))
    return makeMicrokernelError(
        kernel, llvm::Twine(family.microkernelOpName) +
                    " required_capabilities must match selected variant "
                    "requires metadata");

  auto elementCountAttr =
      microkernel->getAttrOfType<mlir::IntegerAttr>(kElementCountAttrName);
  if (!elementCountAttr)
    return makeMicrokernelError(
        kernel, llvm::Twine(family.microkernelOpName) +
                    " requires integer element_count metadata");
  elementCount = elementCountAttr.getInt();
  if (elementCount <= 0 || elementCount > 64)
    return makeMicrokernelError(
        kernel, llvm::Twine(family.microkernelOpName) +
                    " element_count must be in the bounded smoke range "
                    "[1, 64]");

  return llvm::Error::success();
}

llvm::Error findAndValidateMicrokernel(
    KernelOp kernel, const SelectedPath &path,
    const llvm::StringSet<> &selectedScalarPathKeys,
    mlir::Operation *&matchedMicrokernel,
    const ScalarI32MicrokernelFamilySpec *&matchedFamily,
    std::int64_t &elementCount) {
  unsigned matches = 0;
  for (mlir::Operation &op : kernel.getBody().front()) {
    const ScalarI32MicrokernelFamilySpec *family =
        getScalarI32MicrokernelFamilyForOp(&op);
    if (!family)
      continue;

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    auto role =
        op.getAttrOfType<mlir::StringAttr>(execDiagnostic::kRoleAttrName);
    if (!selectedVariant || !role)
      continue;

    std::string key = makePathKey(selectedVariant.getValue(), role.getValue());
    if (!selectedScalarPathKeys.count(key))
      return makeMicrokernelError(
          kernel,
          llvm::Twine("stale ") + family->microkernelOpName + " for @" +
              selectedVariant.getValue() + " as " + role.getValue() +
              " is not selected by the current scalar microkernel surface");

    if (selectedVariant.getValue() == getPathVariantSymbol(path) &&
        role.getValue() == path.role) {
      ++matches;
      matchedMicrokernel = &op;
      matchedFamily = family;
    }
  }

  if (matches == 0)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected scalar path @") +
                    getPathVariantSymbol(path) + " as " + path.role +
                    " requires exactly one matching scalar microkernel");
  if (matches > 1)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected scalar path @") +
                    getPathVariantSymbol(path) + " as " + path.role +
                    " has duplicate scalar microkernel metadata");

  return validateMicrokernelForPath(kernel, path, matchedMicrokernel,
                                    *matchedFamily, elementCount);
}

llvm::Error validateVariantElementCountMatchesMicrokernel(
    KernelOp kernel, VariantOp variant,
    std::int64_t microkernelElementCount) {
  if (auto elementCountAttr =
          variant->getAttrOfType<mlir::IntegerAttr>(kScalarElementCountAttrName);
      elementCountAttr &&
      elementCountAttr.getInt() != microkernelElementCount)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected scalar variant @") +
                    variant.getSymName() + " attribute '" +
                    kScalarElementCountAttrName +
                    "' must match materialized scalar microkernel "
                    "element_count");

  return llvm::Error::success();
}

bool isScalarEmitCLowerableFamily(
    const ScalarI32MicrokernelFamilySpec &family) {
  return family.rvvFamily &&
         (family.rvvFamily->dtype == RVVBinaryDTypeKind::I32 ||
          family.rvvFamily->dtype == RVVBinaryDTypeKind::I64);
}

std::string getScalarEmitCCallee(
    const ScalarI32MicrokernelFamilySpec &family) {
  llvm::StringRef dtype = family.rvvFamily->dtypeID;
  switch (family.rvvFamily->arithmetic) {
  case RVVBinaryArithmeticKind::Add:
    return (llvm::Twine("tcrv_scalar_") + dtype + "_add").str();
  case RVVBinaryArithmeticKind::Sub:
    return (llvm::Twine("tcrv_scalar_") + dtype + "_sub").str();
  case RVVBinaryArithmeticKind::Mul:
    return (llvm::Twine("tcrv_scalar_") + dtype + "_mul").str();
  }
  llvm_unreachable("unknown scalar arithmetic kind");
}

std::string getScalarEmitCStoreCallee(
    const ScalarI32MicrokernelFamilySpec &family) {
  return (llvm::Twine("tcrv_scalar_") + family.rvvFamily->dtypeID +
          "_store")
      .str();
}

llvm::StringRef getScalarEmitCResultName(
    const ScalarI32MicrokernelFamilySpec &family) {
  switch (family.rvvFamily->arithmetic) {
  case RVVBinaryArithmeticKind::Add:
    return "sum";
  case RVVBinaryArithmeticKind::Sub:
    return "difference";
  case RVVBinaryArithmeticKind::Mul:
    return "product";
  }
  llvm_unreachable("unknown scalar arithmetic kind");
}

llvm::Error requireScalarEmitCLowerableInterface(
    KernelOp kernel, mlir::Operation *microkernel,
    const ScalarI32MicrokernelFamilySpec &family,
    std::string &sourceOpName, std::string &sourceOpRole,
    std::string &sourceOpInterface) {
  if (!isScalarEmitCLowerableFamily(family))
    return llvm::Error::success();

  auto lowerable = llvm::dyn_cast<TCRVEmitCLowerableOpInterface>(microkernel);
  if (!lowerable)
    return makeMicrokernelError(
        kernel, llvm::Twine("typed scalar source op '") +
                    microkernel->getName().getStringRef() +
                    "' must implement generated "
                    "TCRVEmitCLowerableOpInterface before the scalar "
                    "EmitC route is constructed");

  llvm::StringRef opName = lowerable.getTCRVEmitCLowerableSourceOpName();
  llvm::StringRef role = lowerable.getTCRVEmitCLowerableSourceRole();
  if (llvm::Error error =
          validateBoundedText(kernel, "EmitC lowerable source op", opName))
    return error;
  if (llvm::Error error =
          validateBoundedText(kernel, "EmitC lowerable source role", role))
    return error;
  if (opName != family.microkernelOpName)
    return makeMicrokernelError(
        kernel, llvm::Twine("generated TCRVEmitCLowerableOpInterface source "
                            "op '") +
                    opName + "' must match typed scalar family operation '" +
                    family.microkernelOpName + "'");
  if (role != "compute")
    return makeMicrokernelError(
        kernel, llvm::Twine("generated TCRVEmitCLowerableOpInterface source "
                            "role '") +
                    role + "' must be 'compute' for bounded scalar typed "
                    "microkernels");

  sourceOpName = opName.str();
  sourceOpRole = role.str();
  sourceOpInterface = "TCRVEmitCLowerableOpInterface";
  return llvm::Error::success();
}

llvm::Expected<ScalarCallableABIPlan>
buildScalarCallableABIPlan(KernelOp kernel,
                           const ScalarI32MicrokernelFamilySpec &family) {
  if (!kernel || kernel.getBody().empty())
    return makeMicrokernelError(
        kernel, "requires a materialized tcrv.exec.kernel body");

  if (!family.rvvFamily)
    return makeMicrokernelError(
        kernel, llvm::Twine("scalar callable ABI plan for ") +
                    family.microkernelOpName +
                    " requires finite RVV binary family metadata");

  llvm::Expected<support::FiniteBinaryCallableABIPlan> finitePlan =
      support::buildFiniteBinaryCallableABIPlan(
          kernel,
          target::rvv::getRVVBinaryRuntimeABIContract(*family.rvvFamily));
  if (!finitePlan)
    return finitePlan.takeError();

  ScalarCallableABIPlan plan;
  plan.parameters = std::move(finitePlan->parameters);
  plan.bufferWindows = std::move(finitePlan->bufferWindows);
  plan.runtimeElementCountParam = finitePlan->runtimeElementCountParam;
  return plan;
}

llvm::Expected<ScalarMicrokernelRecord>
buildMicrokernelRecord(KernelOp kernel, const SelectedPath &path,
                       const TargetCapabilitySet &capabilities,
                       const llvm::StringSet<> &selectedScalarPathKeys) {
  std::string origin;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, getPathVariantOperation(path),
                                execDiagnostic::kOriginAttrName,
                                "selected scalar variant", origin))
    return std::move(error);
  if (origin != kScalarPluginName)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected scalar microkernel path @") +
                    getPathVariantSymbol(path) +
                    " must be owned by origin 'scalar-plugin'");

  if (llvm::Error error =
          validateBoundedText(kernel, "kernel symbol", kernel.getSymName()))
    return std::move(error);
  if (llvm::Error error = validateBoundedText(
          kernel, "variant symbol", getPathVariantSymbol(path)))
    return std::move(error);

  std::string fallbackRole;
  if (auto attr = getStringAttr(getPathVariantOperation(path),
                                kFallbackRoleAttrName)) {
    llvm::StringRef value = attr.getValue().trim();
    if (!value.empty()) {
      if (llvm::Error error =
              validateBoundedText(kernel, kFallbackRoleAttrName, value))
        return std::move(error);
      fallbackRole = value.str();
    }
  }

  llvm::SmallVector<std::string, 4> requiredCapabilities;
  if (llvm::Error error = validateRequiredCapabilities(
          kernel, getPathVariant(path), capabilities, requiredCapabilities))
    return std::move(error);

  LoweringBoundaryOp boundary;
  if (llvm::Error error = findAndValidateBoundary(
          kernel, path, selectedScalarPathKeys, boundary))
    return std::move(error);

  mlir::Operation *microkernel = nullptr;
  const ScalarI32MicrokernelFamilySpec *microkernelFamily = nullptr;
  std::int64_t elementCount = 0;
  if (llvm::Error error =
          findAndValidateMicrokernel(kernel, path, selectedScalarPathKeys,
                                     microkernel, microkernelFamily,
                                     elementCount))
    return std::move(error);
  if (llvm::Error error = validateVariantElementCountMatchesMicrokernel(
          kernel, getPathVariant(path), elementCount))
    return std::move(error);

  std::string emitcSourceOpName;
  std::string emitcSourceOpRole;
  std::string emitcSourceOpInterface;
  if (llvm::Error error = requireScalarEmitCLowerableInterface(
          kernel, microkernel, *microkernelFamily, emitcSourceOpName,
          emitcSourceOpRole, emitcSourceOpInterface))
    return std::move(error);

  llvm::Expected<ScalarCallableABIPlan> callablePlan =
      buildScalarCallableABIPlan(kernel, *microkernelFamily);
  if (!callablePlan)
    return callablePlan.takeError();
  if (llvm::Error error = validateEmissionPlanParameterMirror(
          kernel, path, *microkernelFamily, callablePlan->parameters))
    return std::move(error);

  ScalarMicrokernelRecord record;
  record.family = microkernelFamily;
  record.kernelSymbol = kernel.getSymName().str();
  record.variantSymbol = getPathVariantSymbol(path).str();
  record.role = path.role;
  record.fallbackRole = std::move(fallbackRole);
  record.requiredCapabilities = std::move(requiredCapabilities);
  record.runtimeABIParameters = std::move(callablePlan->parameters);
  record.bufferWindows = std::move(callablePlan->bufferWindows);
  record.runtimeElementCountParam = callablePlan->runtimeElementCountParam;
  record.elementCount = elementCount;
  record.emitcSourceOpName = std::move(emitcSourceOpName);
  record.emitcSourceOpRole = std::move(emitcSourceOpRole);
  record.emitcSourceOpInterface = std::move(emitcSourceOpInterface);
  return record;
}

bool isScalarPluginSelectedPath(const SelectedPath &path) {
  auto origin =
      getPathVariantOperation(path)->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kOriginAttrName);
  return origin && origin.getValue() == kScalarPluginName;
}

bool hasScalarLikeOrigin(const SelectedPath &path) {
  auto origin =
      getPathVariantOperation(path)->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kOriginAttrName);
  if (!origin)
    return false;
  std::string lower = origin.getValue().lower();
  return llvm::StringRef(lower).contains("scalar");
}

llvm::Expected<ScalarMicrokernelRecord> buildModuleRecord(mlir::ModuleOp module) {
  if (!module)
    return makeModuleMicrokernelError("requires a builtin.module operation");

  llvm::SmallVector<KernelOp, 4> kernels;
  module->walk([&](KernelOp kernel) { kernels.push_back(kernel); });
  std::sort(kernels.begin(), kernels.end(),
            [](KernelOp lhs, KernelOp rhs) {
              return lhs.getSymName() < rhs.getSymName();
            });
  if (kernels.empty())
    return makeModuleMicrokernelError("requires at least one tcrv.exec.kernel");

  llvm::SmallVector<ScalarMicrokernelRecord, 2> records;
  for (KernelOp kernel : kernels) {
    llvm::StringMap<VariantOp> directVariants;
    llvm::StringMap<mlir::Operation *> directSymbols;
    collectDirectKernelSymbols(kernel, directVariants, directSymbols);

    llvm::SmallVector<SelectedPath, 4> selectedPaths;
    if (llvm::Error error =
            collectSelectedPaths(kernel, directVariants, directSymbols,
                                 selectedPaths))
      return std::move(error);

    llvm::SmallVector<SelectedPath, 2> selectedScalarPaths;
    for (const SelectedPath &path : selectedPaths) {
      if (isScalarPluginSelectedPath(path)) {
        selectedScalarPaths.push_back(path);
        continue;
      }
      if (hasScalarLikeOrigin(path))
        return makeMicrokernelError(
            kernel, llvm::Twine("selected scalar-like path @") +
                        getPathVariantSymbol(path) +
                        " uses unknown origin; scalar microkernel export only "
                        "accepts registered origin 'scalar-plugin'");
    }

    if (selectedScalarPaths.empty())
      return makeMicrokernelError(
          kernel, "requires one selected scalar-plugin path; non-scalar "
                  "selected paths are not scalar microkernel inputs");
    if (selectedScalarPaths.size() != 1)
      return makeMicrokernelError(
          kernel, "requires exactly one selected scalar-plugin path for this "
                  "bounded scalar microkernel export");

    llvm::StringSet<> selectedScalarPathKeys;
    for (const SelectedPath &path : selectedScalarPaths)
      selectedScalarPathKeys.insert(
          makePathKey(getPathVariantSymbol(path), path.role));

    llvm::Expected<TargetCapabilitySet> capabilities =
        TargetCapabilitySet::buildFromKernelChecked(kernel);
    if (!capabilities)
      return capabilities.takeError();
    llvm::Expected<ScalarMicrokernelRecord> record = buildMicrokernelRecord(
        kernel, selectedScalarPaths.front(), *capabilities,
        selectedScalarPathKeys);
    if (!record)
      return record.takeError();
    records.push_back(std::move(*record));
  }

  if (records.size() != 1)
    return makeModuleMicrokernelError(
        "requires exactly one valid scalar microkernel record in "
        "the module");
  return std::move(records.front());
}

bool isSameScalarMicrokernelFamily(
    const ScalarI32MicrokernelFamilySpec &lhs,
    const ScalarI32MicrokernelFamilySpec &rhs) {
  return lhs.rvvFamily && rhs.rvvFamily &&
         lhs.rvvFamily->familyID == rhs.rvvFamily->familyID &&
         lhs.microkernelOpName == rhs.microkernelOpName &&
         lhs.routeID == rhs.routeID &&
         lhs.emissionKind == rhs.emissionKind &&
         lhs.runtimeABI == rhs.runtimeABI &&
         lhs.runtimeABIKind == rhs.runtimeABIKind &&
         lhs.runtimeABIName == rhs.runtimeABIName &&
         lhs.runtimeGlueRole == rhs.runtimeGlueRole;
}

llvm::Error requireScalarSourceAuthorityField(llvm::StringRef fieldName,
                                              llvm::StringRef actual,
                                              llvm::StringRef expected,
                                              llvm::StringRef selectedVariant) {
  if (actual == expected)
    return llvm::Error::success();
  return makeModuleMicrokernelError(
      llvm::Twine("selected scalar component authority expected ") +
      fieldName + " '" + expected + "' for @" + selectedVariant +
      ", got '" + actual + "'");
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

std::string makeMicrokernelFunctionName(const ScalarMicrokernelRecord &record) {
  std::string name;
  llvm::raw_string_ostream stream(name);
  stream << "tcrv_scalar_" << record.family->functionStem << "_microkernel_"
         << sanitizeCIdentifierComponent(record.kernelSymbol) << "_"
         << sanitizeCIdentifierComponent(record.variantSymbol);
  stream.flush();
  return name;
}

const support::RuntimeABIParameter *lookupParameterByRole(
    llvm::ArrayRef<support::RuntimeABIParameter> parameters,
    support::RuntimeABIParameterRole role) {
  for (const support::RuntimeABIParameter &parameter : parameters)
    if (parameter.role == role)
      return &parameter;
  return nullptr;
}

llvm::Expected<std::string> deriveScalarElementCTypeFromPointerABI(
    const support::RuntimeABIParameter &parameter, llvm::StringRef context) {
  llvm::StringRef cType = llvm::StringRef(parameter.cType).trim();
  if (!cType.ends_with("*"))
    return makeModuleMicrokernelError(
        llvm::Twine(context) + " ABI parameter '" + parameter.cName +
        "' with role '" +
        support::stringifyRuntimeABIParameterRole(parameter.role) +
        "' must be a pointer type before scalar EmitC source rendering");

  llvm::StringRef pointee = cType.drop_back().rtrim();
  if (pointee.consume_front("const "))
    pointee = pointee.ltrim();
  if (pointee.empty())
    return makeModuleMicrokernelError(
        llvm::Twine(context) + " ABI parameter '" + parameter.cName +
        "' has no scalar pointee type before scalar EmitC source rendering");
  return pointee.str();
}

llvm::Expected<std::string> deriveScalarElementCTypeFromCallableABI(
    llvm::ArrayRef<support::RuntimeABIParameter> parameters,
    llvm::StringRef context) {
  const support::RuntimeABIParameter *lhs = lookupParameterByRole(
      parameters, support::RuntimeABIParameterRole::LHSInputBuffer);
  const support::RuntimeABIParameter *rhs = lookupParameterByRole(
      parameters, support::RuntimeABIParameterRole::RHSInputBuffer);
  const support::RuntimeABIParameter *out = lookupParameterByRole(
      parameters, support::RuntimeABIParameterRole::OutputBuffer);
  if (!lhs || !rhs || !out)
    return makeModuleMicrokernelError(
        llvm::Twine(context) +
        " requires lhs, rhs, and output ABI mappings from the IR-backed "
        "callable plan before scalar element type selection");

  llvm::Expected<std::string> lhsType =
      deriveScalarElementCTypeFromPointerABI(*lhs, context);
  if (!lhsType)
    return lhsType.takeError();
  llvm::Expected<std::string> rhsType =
      deriveScalarElementCTypeFromPointerABI(*rhs, context);
  if (!rhsType)
    return rhsType.takeError();
  llvm::Expected<std::string> outType =
      deriveScalarElementCTypeFromPointerABI(*out, context);
  if (!outType)
    return outType.takeError();

  if (*lhsType != *rhsType || *lhsType != *outType)
    return makeModuleMicrokernelError(
        llvm::Twine(context) +
        " requires lhs, rhs, and output ABI pointer scalar types to agree "
        "before scalar EmitC source rendering");
  return std::move(*lhsType);
}

class ScalarBinaryEmitCLowerable final : public TCRVEmitCLowerableInterface {
public:
  explicit ScalarBinaryEmitCLowerable(const ScalarMicrokernelRecord &record)
      : record(record) {}

  llvm::Expected<TCRVEmitCLowerableRoute>
  buildEmitCLowerableRoute() const override {
    if (!record.family || !isScalarEmitCLowerableFamily(*record.family))
      return makeModuleMicrokernelError(
          "scalar EmitC lowerable route is currently defined only for typed "
          "scalar binary microkernels");
    if (record.emitcSourceOpName.empty() || record.emitcSourceOpRole.empty() ||
        record.emitcSourceOpInterface.empty())
      return makeModuleMicrokernelError(
          "scalar binary EmitC route requires generated op-interface "
          "provenance from the typed scalar microkernel op");

    const support::RuntimeABIParameter *lhs = lookupParameterByRole(
        record.runtimeABIParameters,
        support::RuntimeABIParameterRole::LHSInputBuffer);
    const support::RuntimeABIParameter *rhs = lookupParameterByRole(
        record.runtimeABIParameters,
        support::RuntimeABIParameterRole::RHSInputBuffer);
    const support::RuntimeABIParameter *out = lookupParameterByRole(
        record.runtimeABIParameters,
        support::RuntimeABIParameterRole::OutputBuffer);
    if (!lhs || !rhs || !out)
      return makeModuleMicrokernelError(
          "scalar binary EmitC route requires lhs, rhs, and output ABI "
          "mappings from the IR-backed callable plan");

    TCRVEmitCLowerableRoute route(
        record.family->routeID,
        "typed-scalar-family-op-to-emitc-call-opaque");
    route.addHeader("stddef.h");
    route.addHeader("stdint.h");
    llvm::Expected<std::string> scalarElementCType =
        deriveScalarElementCTypeFromCallableABI(
            record.runtimeABIParameters, "scalar binary EmitC route");
    if (!scalarElementCType)
      return scalarElementCType.takeError();
    route.addTypeMapping(record.family->rvvFamily->dtypeID,
                         *scalarElementCType);
    for (const support::RuntimeABIParameter &parameter :
         record.runtimeABIParameters)
      route.addABIValueMapping(parameter, parameter.cName);

    TCRVEmitCCallOpaqueStep computeStep;
    computeStep.sourceOp.opName = record.emitcSourceOpName;
    computeStep.sourceOp.role = record.emitcSourceOpRole;
    computeStep.sourceOp.opInterface = record.emitcSourceOpInterface;
    computeStep.callee = getScalarEmitCCallee(*record.family);
    computeStep.operands.push_back(
        {(llvm::Twine(lhs->cName) + "[index]").str(), *scalarElementCType});
    computeStep.operands.push_back(
        {(llvm::Twine(rhs->cName) + "[index]").str(), *scalarElementCType});
    llvm::StringRef resultName = getScalarEmitCResultName(*record.family);
    computeStep.result =
        TCRVEmitCCallOpaqueResult{resultName.str(), *scalarElementCType};
    route.addCallOpaqueStep(std::move(computeStep));

    TCRVEmitCCallOpaqueStep storeStep;
    storeStep.sourceOp.opName = record.emitcSourceOpName;
    storeStep.sourceOp.role = "buffer-store";
    storeStep.sourceOp.opInterface = record.emitcSourceOpInterface;
    storeStep.callee = getScalarEmitCStoreCallee(*record.family);
    storeStep.operands.push_back(
        {(llvm::Twine("&") + out->cName + "[index]").str(), out->cType});
    storeStep.operands.push_back({resultName.str(), *scalarElementCType});
    route.addCallOpaqueStep(std::move(storeStep));

    return route;
  }

private:
  const ScalarMicrokernelRecord &record;
};

llvm::Expected<TCRVLowerToEmitCSourceResult>
lowerScalarBinaryToEmitCSource(const ScalarMicrokernelRecord &record,
                               llvm::StringRef functionName) {
  ScalarBinaryEmitCLowerable lowerable(record);
  TCRVLowerToEmitCSourceOptions options;
  options.sourceAuthorityOptions.functionName = functionName.str();
  options.sourceAuthorityOptions.loopIndexName = "index";
  options.sourceAuthorityOptions.requireInterfaceBackedCompute = true;
  return lowerTCRVEmitCLowerableToEmitCSource(lowerable, options);
}

std::string
makeMicrokernelHeaderIncludeGuard(const ScalarMicrokernelRecord &record) {
  std::string guard = "TIANCHENRV_SCALAR_";
  guard += record.family->headerGuardStem;
  guard += "_MICROKERNEL_";
  guard += sanitizeCIdentifierComponent(record.kernelSymbol);
  guard += "_";
  guard += sanitizeCIdentifierComponent(record.variantSymbol);
  guard += "_H";
  for (char &character : guard)
    character = static_cast<char>(
        std::toupper(static_cast<unsigned char>(character)));
  return guard;
}

llvm::StringRef getAttrValue(mlir::Operation *op, llvm::StringRef attrName) {
  auto attr = getStringAttr(op, attrName);
  if (!attr)
    return {};
  return attr.getValue();
}

void printCallableBoundaryMetadata(llvm::raw_ostream &os,
                                   const ScalarMicrokernelRecord &record) {
  os << "/* callable_abi_source: tcrv.exec.mem_window + "
        "tcrv.exec.runtime_param */\n";
  for (auto [index, windowRef] : llvm::enumerate(record.bufferWindows)) {
    MemWindowOp window = windowRef;
    os << "/* callable_mem_window[" << index << "]: symbol=@"
       << getAttrValue(window.getOperation(), "sym_name") << ", abi_role="
       << getAttrValue(window.getOperation(),
                       support::kMemWindowABIRoleAttrName)
       << ", access="
       << getAttrValue(window.getOperation(), support::kMemWindowAccessAttrName)
       << ", ownership="
       << getAttrValue(window.getOperation(),
                       support::kMemWindowOwnershipAttrName)
       << ", c_type="
       << getAttrValue(window.getOperation(), support::kMemWindowCTypeAttrName)
       << " */\n";
  }

  RuntimeParamOp runtimeParam = record.runtimeElementCountParam;
  os << "/* callable_runtime_param[0]: symbol=@"
     << getAttrValue(runtimeParam.getOperation(), "sym_name")
     << ", abi_role="
     << getAttrValue(runtimeParam.getOperation(),
                     support::kRuntimeParamABIRoleAttrName)
     << ", c_name="
     << getAttrValue(runtimeParam.getOperation(),
                     support::kRuntimeParamCNameAttrName)
     << ", c_type="
     << getAttrValue(runtimeParam.getOperation(),
                     support::kRuntimeParamCTypeAttrName)
     << ", ownership="
     << getAttrValue(runtimeParam.getOperation(),
                     support::kRuntimeParamOwnershipAttrName)
     << " */\n";
}

void printScalarEmitCRouteMetadata(llvm::raw_ostream &os,
                                   const TCRVEmitCLowerableRoute &route,
                                   llvm::StringRef functionName) {
  llvm::StringRef sourceOp =
      route.getCallOpaqueSteps().empty()
          ? llvm::StringRef("tcrv_scalar.<unknown_i32_binary_microkernel>")
          : route.getCallOpaqueSteps().front().sourceOp.opName;
  os << "/* emitc_route: " << sourceOp << " -> "
        "emitc.call_opaque -> scalar runtime C/C++ */\n";
  os << "/* emitc_lowerable_interface: TCRVEmitCLowerableInterface */\n";
  os << "/* emitc_common_lower_to_emitc_boundary: "
        "TCRVLowerToEmitCSourceAuthority */\n";
  os << "/* emitc_materialization_boundary: verified MLIR EmitC module with "
        "emitc.include, emitc.func, emitc.if, emitc.call_opaque, and "
        "emitc.call before MLIR Cpp emitter production source output */\n";
  os << "/* emitc_materialization_function: @" << functionName << " */\n";
  os << "/* deleted_direct_c_source_route: historical EmitC source text is not "
        "a registered scalar artifact authority */\n";
  for (const TCRVEmitCCallOpaqueStep &step : route.getCallOpaqueSteps()) {
    if (step.sourceOp.opInterface.empty())
      continue;
    os << "/* emitc_lowerable_op_interface: "
       << step.sourceOp.opInterface << " */\n";
    break;
  }
  os << "/* emitc_route_id: " << route.getRouteID()
     << ", route_kind=" << route.getRouteKind() << " */\n";
  os << "/* emitc_route_headers:";
  for (const auto &header : route.getHeaders())
    os << " <" << header.header << ">";
  os << " */\n";
  os << "/* emitc_route_source_ops:";
  for (const TCRVEmitCCallOpaqueStep &step : route.getCallOpaqueSteps())
    os << " " << step.sourceOp.opName;
  os << " */\n";
  for (auto [index, step] : llvm::enumerate(route.getCallOpaqueSteps())) {
    os << "/* emitc.call_opaque[" << index
       << "]: " << step.callee << " from " << step.sourceOp.opName
       << " */\n";
    os << "/* emitc.call_opaque_boundary[" << index << "]: source_role="
       << step.sourceOp.role << ", operands=" << step.operands.size();
    if (step.result)
      os << ", result=" << step.result->name << ":" << step.result->cType;
    else
      os << ", result=void";
    if (!step.sourceOp.opInterface.empty())
      os << ", op_interface=" << step.sourceOp.opInterface;
    os << " */\n";
  }
}

void printRecordComment(llvm::raw_ostream &os,
                        const ScalarMicrokernelRecord &record,
                        llvm::StringRef functionName,
                        const TCRVEmitCLowerableRoute *emitcRoute = nullptr) {
  os << "/* microkernel function: " << functionName << " */\n";
  os << "/* selected_kernel: @" << record.kernelSymbol << " */\n";
  os << "/* selected_variant: @" << record.variantSymbol << " */\n";
  os << "/* selected_role: " << record.role << " */\n";
  if (!record.fallbackRole.empty())
    os << "/* fallback_role: " << record.fallbackRole << " */\n";
  os << "/* lowering_boundary: tcrv_scalar.lowering_boundary */\n";
  os << "/* executable_microkernel: " << record.family->microkernelOpName
     << " */\n";
  os << "/* artifact_kind: " << kMicrokernelArtifactKind << " */\n";
  os << "/* element_count: " << record.elementCount << " */\n";
  os << "/* required_capabilities:";
  for (llvm::StringRef capability : record.requiredCapabilities)
    os << " @" << capability;
  os << " */\n";
  os << "/* runtime_abi_kind: " << record.family->runtimeABIKind << " */\n";
  os << "/* runtime_abi_name: " << record.family->runtimeABIName << " */\n";
  os << "/* runtime_glue_role: " << record.family->runtimeGlueRole << " */\n";
  printCallableBoundaryMetadata(os, record);
  for (auto [index, parameter] :
       llvm::enumerate(record.runtimeABIParameters)) {
    os << "/* runtime_abi_parameter[" << index << "]: c_name="
       << parameter.cName << ", c_type=" << parameter.cType
       << ", role="
       << support::stringifyRuntimeABIParameterRole(parameter.role)
       << ", ownership="
       << support::stringifyRuntimeABIParameterOwnership(parameter.ownership)
       << " */\n";
  }
  os << "/* runtime_callable_abi: void " << functionName
     << "(";
  for (auto [index, parameter] :
       llvm::enumerate(record.runtimeABIParameters)) {
    if (index != 0)
      os << ", ";
    support::printRuntimeABIParameterCDeclaration(os, parameter);
  }
  os << ") */\n";
  if (emitcRoute)
    printScalarEmitCRouteMetadata(os, *emitcRoute, functionName);
}

void printMicrokernelPrototype(
    llvm::raw_ostream &os, llvm::StringRef functionName,
    llvm::ArrayRef<support::RuntimeABIParameter> parameters) {
  os << "void " << functionName << "(";
  for (auto [index, parameter] : llvm::enumerate(parameters)) {
    if (index != 0)
      os << ", ";
    support::printRuntimeABIParameterCDeclaration(os, parameter);
  }
  os << ")";
}

void printMicrokernelHeader(const ScalarMicrokernelRecord &record,
                            llvm::raw_ostream &os) {
  std::string includeGuard = makeMicrokernelHeaderIncludeGuard(record);
  std::string functionName = makeMicrokernelFunctionName(record);

  os << "#ifndef " << includeGuard << "\n";
  os << "#define " << includeGuard << "\n\n";
  os << "#include <stddef.h>\n";
  os << "#include <stdint.h>\n\n";
  os << "#ifdef __cplusplus\n";
  os << "extern \"C\" {\n";
  os << "#endif\n\n";

  printMicrokernelPrototype(os, functionName, record.runtimeABIParameters);
  os << ";\n\n";

  os << "#ifdef __cplusplus\n";
  os << "}\n";
  os << "#endif\n\n";
  os << "#endif /* " << includeGuard << " */\n";
}

TargetArtifactRouteMetadata buildScalarMicrokernelSourceRouteMetadata(
    const ScalarI32MicrokernelFamilySpec &family) {
  TargetArtifactRouteMetadata metadata(family.runtimeABI,
                                       family.runtimeABIKind,
                                       family.runtimeABIName,
                                       family.runtimeGlueRole);
  if (family.rvvFamily->dtype == RVVBinaryDTypeKind::I32 ||
      family.rvvFamily->dtype == RVVBinaryDTypeKind::I64) {
    llvm::StringRef typedRole =
        tianchenrv::target::rvv_scalar::
            getScalarTypedBinarySourceMetadataRole();
    metadata.addSelectedPlanMetadataRequirement(
        tianchenrv::target::rvv_scalar::
            getScalarSelectedBinaryDTypeMetadataName(),
        family.rvvFamily->dtypeID, typedRole);
    metadata.addSelectedPlanMetadataRequirement(
        tianchenrv::target::rvv_scalar::
            getScalarSelectedBinaryFamilyMetadataName(),
        family.rvvFamily->familyID, typedRole);
    metadata.addSelectedPlanMetadataRequirement(
        tianchenrv::target::rvv_scalar::
            getScalarSelectedBinaryOperatorMetadataName(),
        family.rvvFamily->arithmeticVerb, typedRole);
    metadata.addSelectedPlanMetadataRequirement(
        tianchenrv::target::rvv_scalar::
            getScalarEmitCSourceOpMetadataName(),
        family.microkernelOpName,
        tianchenrv::target::rvv_scalar::
            getScalarEmitCSourceOpMetadataRole());
    metadata.addSelectedPlanMetadataRequirement(
        tianchenrv::target::rvv_scalar::
            getScalarEmitCLowerableOpInterfaceMetadataName(),
        "TCRVEmitCLowerableOpInterface",
        tianchenrv::target::rvv_scalar::
            getScalarEmitCSourceOpMetadataRole());
  }
  metadata.addSelectedPlanMetadataPresenceRequirement(
      tianchenrv::target::rvv_scalar::
          getScalarRuntimeElementCountCNameMetadataName(),
      tianchenrv::target::rvv_scalar::
          getScalarRuntimeControlNameMetadataRole());
  metadata.addClaimField("compile_export_claim", "compiler-artifact-only");
  metadata.addClaimField("runtime_correctness_claim", "none");
  metadata.addClaimField("hardware_execution_claim", "none");
  metadata.addClaimField("performance_claim", "none");
  return metadata;
}

llvm::Expected<const SelectedPlanMetadataEntry *>
findUniqueScalarSelectedPlanMetadataEntry(
    const TargetArtifactCandidate &candidate, llvm::StringRef name) {
  const SelectedPlanMetadataEntry *match = nullptr;
  for (const SelectedPlanMetadataEntry &metadata :
       candidate.selectedPlanMetadata) {
    if (metadata.name != name)
      continue;
    if (match)
      return makeModuleMicrokernelError(
          llvm::Twine("target artifact route '") + candidate.routeID +
          "' has duplicate selected_plan_metadata '" + name + "'");
    match = &metadata;
  }
  if (!match)
    return makeModuleMicrokernelError(
        llvm::Twine("target artifact route '") + candidate.routeID +
        "' requires selected_plan_metadata '" + name + "'");
  return match;
}

llvm::Expected<llvm::StringRef>
resolveScalarRuntimeElementCountCName(
    const TargetArtifactCandidate &candidate) {
  const support::RuntimeABIParameter *runtimeElementCount = nullptr;
  for (const support::RuntimeABIParameter &parameter :
       candidate.runtimeABIParameters) {
    if (parameter.role != support::RuntimeABIParameterRole::RuntimeElementCount)
      continue;
    if (runtimeElementCount)
      return makeModuleMicrokernelError(
          llvm::Twine("target artifact route '") + candidate.routeID +
          "' has duplicate runtime element-count ABI parameters");
    runtimeElementCount = &parameter;
  }
  if (!runtimeElementCount)
    return makeModuleMicrokernelError(
        llvm::Twine("target artifact route '") + candidate.routeID +
        "' requires one runtime element-count ABI parameter");
  return llvm::StringRef(runtimeElementCount->cName);
}

llvm::Error validateScalarSelectedPlanMetadata(
    const TargetArtifactCandidate &candidate,
    const ScalarI32MicrokernelFamilySpec &family) {
  llvm::Expected<llvm::StringRef> runtimeElementCountCName =
      resolveScalarRuntimeElementCountCName(candidate);
  if (!runtimeElementCountCName)
    return runtimeElementCountCName.takeError();
  llvm::Expected<const SelectedPlanMetadataEntry *> metadata =
      findUniqueScalarSelectedPlanMetadataEntry(
          candidate, tianchenrv::target::rvv_scalar::
                         getScalarRuntimeElementCountCNameMetadataName());
  if (!metadata)
    return metadata.takeError();
  if ((*metadata)->value != *runtimeElementCountCName)
    return makeModuleMicrokernelError(
        llvm::Twine("target artifact route '") + candidate.routeID +
        "' selected_plan_metadata '" +
        tianchenrv::target::rvv_scalar::
            getScalarRuntimeElementCountCNameMetadataName() +
        "' runtime element-count C name must be '" +
        *runtimeElementCountCName + "' for scalar family '" +
        family.rvvFamily->familyID + "'");
  return llvm::Error::success();
}

bool isScalarMicrokernelSourceCandidateForFamily(
    const tianchenrv::target::TargetArtifactCandidate &candidate,
    const ScalarI32MicrokernelFamilySpec &family) {
  return candidateMatchesScalarMicrokernelFamily(candidate, family);
}

llvm::Error validateScalarMicrokernelSourceCandidate(
    const tianchenrv::target::TargetArtifactCandidate &candidate) {
  const ScalarI32MicrokernelFamilySpec *family =
      getScalarI32MicrokernelFamilyForSourceRoute(candidate.routeID);
  if (!family)
    return makeModuleMicrokernelError(
        llvm::Twine("target artifact route '") + candidate.routeID +
        "' is not a supported scalar microkernel source route");

  if (!candidateMatchesScalarMicrokernelFamily(candidate, *family))
    return makeModuleMicrokernelError(
        llvm::Twine("target artifact route '") + candidate.routeID +
        "' does not match supported scalar microkernel family ABI "
        "metadata; expected emission_kind '" +
        family->emissionKind + "', artifact_kind '" +
        kMicrokernelArtifactKind + "', runtime_abi '" + family->runtimeABI +
        "', runtime_abi_kind '" + family->runtimeABIKind +
        "', runtime_abi_name '" + family->runtimeABIName +
        "', runtime_glue_role '" + family->runtimeGlueRole + "'");

  TargetArtifactExporter sourceExporter(
      family->routeID, kMicrokernelArtifactKind, kScalarPluginName,
      family->emissionKind, exportScalarMicrokernelC,
      tianchenrv::target::rvv_scalar::
          getRVVScalarBinaryCallableRuntimeABIRoleRequirements(
              *tianchenrv::target::rvv_scalar::
                  lookupRVVScalarBinaryRegistrationByScalarRouteID(
                      family->routeID)),
      /*directHelperRoute=*/false, /*handoffKind=*/{},
      /*candidateValidationFn=*/nullptr, /*componentGroup=*/{},
      /*externalABIName=*/{}, buildScalarMicrokernelSourceRouteMetadata(*family));
  if (llvm::Error error =
          validateTargetArtifactCandidateAgainstExporter(candidate,
                                                        sourceExporter))
    return error;
  return validateScalarSelectedPlanMetadata(candidate, *family);
}

llvm::Error validateScalarMicrokernelCallableCandidatePreflight(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  if (candidates.size() != 1)
    return makeModuleMicrokernelError(
        "scalar microkernel helper routes require exactly one callable "
        "artifact candidate for preflight");

  return validateScalarMicrokernelSourceCandidate(candidates.front());
}

llvm::Expected<bool> matchScalarMicrokernelObjectCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates,
    const ScalarI32MicrokernelFamilySpec &family) {
  if (candidates.size() != 1)
    return false;
  return isScalarMicrokernelSourceCandidateForFamily(candidates.front(),
                                                     family);
}

llvm::Expected<bool> matchScalarI32VAddMicrokernelCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  return matchScalarMicrokernelObjectCandidate(candidates,
                                               getI32VAddFamilySpec());
}

llvm::Expected<bool> matchScalarI32VSubMicrokernelCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  return matchScalarMicrokernelObjectCandidate(candidates,
                                               getI32VSubFamilySpec());
}

llvm::Expected<bool> matchScalarI32VMulMicrokernelCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  return matchScalarMicrokernelObjectCandidate(candidates,
                                               getI32VMulFamilySpec());
}

llvm::Expected<bool> matchScalarI64VAddMicrokernelCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  return matchScalarMicrokernelObjectCandidate(candidates,
                                               getI64VAddFamilySpec());
}

llvm::Expected<bool> matchScalarI64VSubMicrokernelCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  return matchScalarMicrokernelObjectCandidate(candidates,
                                               getI64VSubFamilySpec());
}

llvm::Expected<bool> matchScalarI64VMulMicrokernelCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  return matchScalarMicrokernelObjectCandidate(candidates,
                                               getI64VMulFamilySpec());
}

TargetArtifactCompositeMatchFn getScalarMicrokernelCompositeMatchFn(
    const ScalarI32MicrokernelFamilySpec &family) {
  if (family.routeID == getI32VAddFamilySpec().routeID)
    return matchScalarI32VAddMicrokernelCandidate;
  if (family.routeID == getI32VSubFamilySpec().routeID)
    return matchScalarI32VSubMicrokernelCandidate;
  if (family.routeID == getI32VMulFamilySpec().routeID)
    return matchScalarI32VMulMicrokernelCandidate;
  if (family.routeID == getI64VAddFamilySpec().routeID)
    return matchScalarI64VAddMicrokernelCandidate;
  if (family.routeID == getI64VSubFamilySpec().routeID)
    return matchScalarI64VSubMicrokernelCandidate;
  if (family.routeID == getI64VMulFamilySpec().routeID)
    return matchScalarI64VMulMicrokernelCandidate;
  return nullptr;
}

llvm::Error createTempFile(llvm::StringRef prefix, llvm::StringRef suffix,
                           TemporaryFile &file) {
  std::error_code ec =
      llvm::sys::fs::createTemporaryFile(prefix, suffix, file.path);
  if (ec)
    return makeModuleMicrokernelObjectError(
        llvm::Twine("failed to create temporary ") + suffix +
        " file for object export: " + ec.message());
  return llvm::Error::success();
}

llvm::Error writeTempSource(llvm::StringRef source, TemporaryFile &sourceFile) {
  int fd = -1;
  std::error_code ec = llvm::sys::fs::createTemporaryFile(
      "tcrv-scalar-microkernel", "c", fd, sourceFile.path);
  if (ec)
    return makeModuleMicrokernelObjectError(
        llvm::Twine("failed to create temporary C source for object export: ") +
        ec.message());

  llvm::raw_fd_ostream stream(fd, /*shouldClose=*/true);
  stream << source;
  stream.close();
  if (stream.has_error())
    return makeModuleMicrokernelObjectError(
        "failed to write generated scalar microkernel C source before object "
        "export");
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

std::string formatCompileCommand(const ScalarObjectCompileConfig &config) {
  std::string command;
  llvm::raw_string_ostream stream(command);
  stream << "clang -target " << config.targetTriple << " -O2 -march="
         << config.selectedMarch;
  if (config.selectedMABI)
    stream << " -mabi=" << *config.selectedMABI;
  stream << " -c <generated-scalar-microkernel-source> -o <object-file>";
  stream.flush();
  return command;
}

llvm::Error validateBoundedCompileText(llvm::StringRef kernelSymbol,
                                       llvm::StringRef fieldName,
                                       llvm::StringRef value) {
  if (value.empty() || value.size() > 512)
    return makeMicrokernelObjectError(
        kernelSymbol,
        llvm::Twine(fieldName) +
            " must be bounded non-empty single-line metadata");
  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0 ||
        (byte < 0x20 && character != '\t'))
      return makeMicrokernelObjectError(
          kernelSymbol,
          llvm::Twine(fieldName) +
              " must be bounded non-empty single-line metadata");
  }
  if (value.contains("/*") || value.contains("*/"))
    return makeMicrokernelObjectError(
        kernelSymbol,
        llvm::Twine(fieldName) + " must not contain C comment delimiter text");
  if (containsForbiddenText(value))
    return makeMicrokernelObjectError(
        kernelSymbol,
        llvm::Twine(fieldName) +
            " must not contain secret-like or raw credential text");
  return llvm::Error::success();
}

llvm::Error compileGeneratedMicrokernelSourceToObject(
    const ScalarMicrokernelRecord &record,
    const ScalarObjectCompileConfig &compileConfig, llvm::StringRef source,
    llvm::raw_ostream &os) {
  if (llvm::Error error =
          validateBoundedCompileText(record.kernelSymbol, "target triple",
                                     compileConfig.targetTriple))
    return error;
  if (llvm::Error error =
          validateBoundedCompileText(record.kernelSymbol, "selected_march",
                                     compileConfig.selectedMarch))
    return error;
  if (compileConfig.selectedMABI)
    if (llvm::Error error =
            validateBoundedCompileText(record.kernelSymbol, "selected_mabi",
                                       *compileConfig.selectedMABI))
      return error;

  llvm::ErrorOr<std::string> clangPath =
      llvm::sys::findProgramByName("clang");
  if (!clangPath)
    clangPath = llvm::sys::findProgramByName("clang-20");
  if (!clangPath)
    return makeMicrokernelObjectError(
        record.kernelSymbol,
        "requires clang or clang-20 on PATH to compile the bounded scalar "
        "microkernel C source into a RISC-V object file");

  TemporaryFile sourceFile;
  if (llvm::Error error = writeTempSource(source, sourceFile))
    return error;

  TemporaryFile objectFile;
  if (llvm::Error error =
          createTempFile("tcrv-scalar-microkernel", "o", objectFile))
    return error;

  TemporaryFile stderrFile;
  if (llvm::Error error =
          createTempFile("tcrv-scalar-microkernel", "stderr", stderrFile))
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
    return makeMicrokernelObjectError(
        record.kernelSymbol,
        llvm::Twine("clang failed while creating object file; command: ") +
            formatCompileCommand(compileConfig) + "; exit_code=" +
            llvm::Twine(exitCode) + "; stderr: " + stderrText);
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> objectBuffer =
      llvm::MemoryBuffer::getFile(objectFile.get(), /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!objectBuffer)
    return makeMicrokernelObjectError(
        record.kernelSymbol,
        llvm::Twine("failed to read generated object file: ") +
            objectBuffer.getError().message());

  llvm::StringRef bytes = (*objectBuffer)->getBuffer();
  if (bytes.empty())
    return makeMicrokernelObjectError(record.kernelSymbol,
                                      "generated object file must be non-empty");
  if (bytes.size() < 4 || !bytes.starts_with("\177ELF"))
    return makeMicrokernelObjectError(
        record.kernelSymbol,
        "generated object file must have an ELF object-file signature");

  os.write(bytes.data(), bytes.size());
  return llvm::Error::success();
}

} // namespace

llvm::Error exportScalarMicrokernelC(mlir::ModuleOp module,
                                     llvm::raw_ostream &os) {
  (void)module;
  (void)os;
  return makeModuleMicrokernelError(kDirectCSourceRouteDeletedReason);
}

llvm::Error validateScalarMicrokernelSourceAuthority(
    mlir::ModuleOp module,
    const rvv_scalar::ScalarBinaryMicrokernelRecord &family,
    llvm::StringRef selectedVariant, llvm::StringRef role) {
  llvm::Expected<ScalarMicrokernelRecord> record = buildModuleRecord(module);
  if (!record)
    return record.takeError();

  if (llvm::Error error = requireScalarSourceAuthorityField(
          "selected variant", record->variantSymbol, selectedVariant,
          selectedVariant))
    return error;
  if (llvm::Error error = requireScalarSourceAuthorityField(
          "role", record->role, role, selectedVariant))
    return error;

  if (!record->family ||
      !isSameScalarMicrokernelFamily(*record->family, family)) {
    llvm::StringRef actual =
        record->family ? llvm::StringRef(record->family->microkernelOpName)
                       : llvm::StringRef("<missing>");
    return makeModuleMicrokernelError(
        llvm::Twine("selected scalar component authority for @") +
        selectedVariant + " requires " + family.microkernelOpName +
        " but typed scalar record is " + actual);
  }

  return llvm::Error::success();
}

llvm::Error exportScalarMicrokernelHeader(mlir::ModuleOp module,
                                          llvm::raw_ostream &os) {
  llvm::Expected<ScalarMicrokernelRecord> record = buildModuleRecord(module);
  if (!record)
    return record.takeError();

  printMicrokernelHeader(*record, os);
  return llvm::Error::success();
}

llvm::Error exportScalarMicrokernelObject(mlir::ModuleOp module,
                                          llvm::raw_ostream &os) {
  (void)module;
  (void)os;
  return makeModuleMicrokernelObjectError(kDirectCSourceRouteDeletedReason);
}

llvm::Error registerScalarMicrokernelTargetExporters(
    TargetArtifactExporterRegistry &registry) {
  (void)registry;
  return llvm::Error::success();
}

llvm::Error registerScalarMicrokernelPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry) {
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      kScalarPluginName, registerScalarMicrokernelTargetExporters));
}

} // namespace tianchenrv::target::scalar
