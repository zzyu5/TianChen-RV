#include "TianChenRV/Target/RVV/RVVMicrokernel.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVLowerToEmitC.h"
#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Support/RuntimeABICallablePlan.h"
#include "TianChenRV/Support/RuntimeABIContract.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"
#include "TianChenRV/Target/BinarySelfCheckExpectation.h"
#include "TianChenRV/Target/I32BinaryFamilyRegistry.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"
#include "TianChenRV/Target/RVV/RVVBinaryMicrokernelBodyVerifier.h"
#include "TianChenRV/Target/RVV/RVVBinaryDescriptor.h"
#include "TianChenRV/Target/RVV/RVVVectorShape.h"
#include "TianChenRV/Target/RVV/RVVSelectedConfigContract.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
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
#include <utility>

namespace tianchenrv::target::rvv {
namespace {

namespace execDiagnostic = tianchenrv::tcrv::exec::diagnostic;

using tianchenrv::support::CapabilityDescriptor;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::target::BinarySelfCheckArithmeticKind;
using tianchenrv::target::BinarySelfCheckExpectation;
using tianchenrv::conversion::emitc::TCRVEmitCCallOpaqueResult;
using tianchenrv::conversion::emitc::TCRVEmitCCallOpaqueStep;
using tianchenrv::conversion::emitc::TCRVEmitCLowerableInterface;
using tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute;
using tianchenrv::conversion::emitc::TCRVLowerToEmitCSourceOptions;
using tianchenrv::conversion::emitc::TCRVLowerToEmitCSourceResult;
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
using tianchenrv::tcrv::rvv::I32VAddMicrokernelOp;
using tianchenrv::tcrv::rvv::I32VMulMicrokernelOp;
using tianchenrv::tcrv::rvv::I32VSubMicrokernelOp;
using tianchenrv::tcrv::rvv::LoweringBoundaryOp;
using tianchenrv::tcrv::rvv::PolicyAttr;

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kRVVCapabilityID("rvv");
constexpr llvm::StringLiteral kRVVRequiredMarchAttrName(
    "tcrv_rvv.required_march");
constexpr llvm::StringLiteral kRVVLoweringDescriptorAttrName(
    "tcrv_rvv.lowering_descriptor");
constexpr llvm::StringLiteral kRVVPolicyAttrName("tcrv_rvv.policy");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kArchitecturePropertyName("architecture");
constexpr llvm::StringLiteral kSEWBitsPropertyName("sew_bits");
constexpr llvm::StringLiteral kLMULPropertyName("lmul");
constexpr llvm::StringLiteral kTailPolicyPropertyName("tail_policy");
constexpr llvm::StringLiteral kMaskPolicyPropertyName("mask_policy");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kRequiredMarchAttrName("required_march");
constexpr llvm::StringLiteral kRVVElementCountAttrName(
    "tcrv_rvv.element_count");
constexpr llvm::StringLiteral kSelectedMABIAttrName("selected_mabi");
constexpr llvm::StringLiteral kRVVSelectedVectorShapeAttrName(
    "tcrv_rvv.selected_vector_shape");
constexpr llvm::StringLiteral kRVVSelectedVectorSEWAttrName(
    "tcrv_rvv.selected_vector_sew");
constexpr llvm::StringLiteral kRVVSelectedVectorLMULAttrName(
    "tcrv_rvv.selected_vector_lmul");
constexpr llvm::StringLiteral kRVVSelectedTailPolicyAttrName(
    "tcrv_rvv.selected_tail_policy");
constexpr llvm::StringLiteral kRVVSelectedMaskPolicyAttrName(
    "tcrv_rvv.selected_mask_policy");
constexpr llvm::StringLiteral kRVVSelectedVectorTypeAttrName(
    "tcrv_rvv.selected_vector_type");
constexpr llvm::StringLiteral kRVVSelectedVectorSuffixAttrName(
    "tcrv_rvv.selected_vector_suffix");
constexpr llvm::StringLiteral kRVVSelectedSetVLSuffixAttrName(
    "tcrv_rvv.selected_setvl_suffix");
constexpr llvm::StringLiteral kBoundarySelectedVectorShapeAttrName(
    "selected_vector_shape");
constexpr llvm::StringLiteral kBoundarySelectedVectorSEWAttrName(
    "selected_vector_sew");
constexpr llvm::StringLiteral kBoundarySelectedVectorLMULAttrName(
    "selected_vector_lmul");
constexpr llvm::StringLiteral kBoundarySelectedTailPolicyAttrName(
    "selected_tail_policy");
constexpr llvm::StringLiteral kBoundarySelectedMaskPolicyAttrName(
    "selected_mask_policy");
constexpr llvm::StringLiteral kBoundarySelectedVectorTypeAttrName(
    "selected_vector_type");
constexpr llvm::StringLiteral kBoundarySelectedVectorSuffixAttrName(
    "selected_vector_suffix");
constexpr llvm::StringLiteral kBoundarySelectedSetVLSuffixAttrName(
    "selected_setvl_suffix");
constexpr llvm::StringLiteral kUnsupportedStatusValue("unsupported");
constexpr llvm::StringLiteral kDirectVariantRole("direct variant");
constexpr llvm::StringLiteral kDispatchCaseRole("dispatch case");
constexpr llvm::StringLiteral kDispatchFallbackRole("dispatch fallback");
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
enum class RVVMicrokernelCExportMode {
  RuntimeCallableLibrary,
  SelfCheckHarness,
};

using RVVI32VAddDataflowStepKind = RVVBinaryDataflowStepKind;
using RVVI32VAddDataflowValue = RVVBinaryDataflowValue;
using RVVI32VAddDataflowStep = RVVBinaryDataflowStep;
using RVVI32VAddDataflowEmissionPlan = RVVBinaryDataflowEmissionPlan;

using RVVI32MicrokernelKind =
    tianchenrv::target::rvv::RVVBinaryArithmeticKind;
using RVVI32MicrokernelFamilySpec =
    tianchenrv::target::rvv::RVVBinaryFamilyDescriptor;

struct SelectedPath {
  VariantOp variant;
  mlir::Operation *selector = nullptr;
  std::string role;
};

struct RVVMicrokernelRecord {
  const RVVI32MicrokernelFamilySpec *family = nullptr;
  RVVBinaryIntrinsicDescriptor descriptor;
  std::string activeRouteID;
  std::string kernelSymbol;
  std::string variantSymbol;
  std::string role;
  std::string targetTriple;
  std::string selectedMarch;
  std::optional<std::string> selectedMABI;
  llvm::SmallVector<std::string, 4> requiredCapabilities;
  llvm::SmallVector<support::RuntimeABIParameter, 4> runtimeABIParameters;
  llvm::SmallVector<MemWindowOp, 3> bufferWindows;
  RuntimeParamOp runtimeElementCountParam;
  RVVBinarySelectedConfigContract selectedConfigContract;
  RVVI32VAddDataflowEmissionPlan dataflowPlan;
  RVVIntrinsicConfig intrinsicConfig;
  const RVVI32VectorShapeConfig *selectedShape = nullptr;
  std::int64_t elementCount = 0;
  std::int64_t controlPlaneSEW = 0;
  std::string controlPlaneLMUL;
};

struct TemporaryFile {
  llvm::SmallString<128> path;

  ~TemporaryFile() {
    if (!path.empty())
      llvm::sys::fs::remove(path);
  }

  llvm::StringRef get() const { return llvm::StringRef(path); }
};

const RVVI32MicrokernelFamilySpec &getI32VAddFamilySpec() {
  return tianchenrv::target::rvv::getI32VAddFamilyRegistrationRecord();
}

const RVVI32MicrokernelFamilySpec &getI32VSubFamilySpec() {
  return tianchenrv::target::rvv::getI32VSubFamilyRegistrationRecord();
}

const RVVI32MicrokernelFamilySpec &getI32VMulFamilySpec() {
  return tianchenrv::target::rvv::getI32VMulFamilyRegistrationRecord();
}

const RVVI32MicrokernelFamilySpec &
getI32MicrokernelFamilySpec(RVVI32MicrokernelKind kind) {
  switch (kind) {
  case RVVI32MicrokernelKind::Add:
    return getI32VAddFamilySpec();
  case RVVI32MicrokernelKind::Sub:
    return getI32VSubFamilySpec();
  case RVVI32MicrokernelKind::Mul:
    return getI32VMulFamilySpec();
  }
  llvm_unreachable("unknown RVV i32 microkernel family");
}

bool isSameRVVBinaryFamily(const RVVBinaryFamilyDescriptor &lhs,
                           const RVVBinaryFamilyDescriptor &rhs) {
  return lhs.dtype == rhs.dtype && lhs.arithmetic == rhs.arithmetic;
}

RVVI32MicrokernelKind
convertI32BinaryFamilyKind(i32_binary::I32BinaryFamilyKind kind) {
  using I32Kind = i32_binary::I32BinaryFamilyKind;
  switch (kind) {
  case I32Kind::Add:
    return RVVI32MicrokernelKind::Add;
  case I32Kind::Sub:
    return RVVI32MicrokernelKind::Sub;
  case I32Kind::Mul:
    return RVVI32MicrokernelKind::Mul;
  }
  llvm_unreachable("unknown legacy i32 binary family kind");
}

RVVBinaryIntrinsicDescriptor
getI32BinaryIntrinsicDescriptorForMicrokernel(
    const RVVI32MicrokernelFamilySpec &family,
    const RVVI32VectorShapeConfig &shape) {
  return getRVVBinaryIntrinsicDescriptor(family, shape);
}

const RVVI32VectorShapeConfig &getI32M1ConfigSpec() {
  return getI32M1VectorShapeConfig();
}

const RVVI32VectorShapeConfig &getI32M2ConfigSpec() {
  return getI32M2VectorShapeConfig();
}

const RVVI32MicrokernelFamilySpec *
getI32MicrokernelFamilyForOp(mlir::Operation *op) {
  if (llvm::isa_and_nonnull<I32VAddMicrokernelOp>(op))
    return &getI32VAddFamilySpec();
  if (llvm::isa_and_nonnull<I32VSubMicrokernelOp>(op))
    return &getI32VSubFamilySpec();
  if (llvm::isa_and_nonnull<I32VMulMicrokernelOp>(op))
    return &getI32VMulFamilySpec();
  return nullptr;
}

const RVVBinaryFamilyDescriptor *
getI64MicrokernelFamilyForOp(mlir::Operation *op) {
  if (!op)
    return nullptr;
  llvm::StringRef opName = op->getName().getStringRef();
  for (const RVVBinaryFamilyDescriptor *family :
       getRVVBinaryFamilyRegistrationRecords()) {
    if (family->dtype == RVVBinaryDTypeKind::I64 &&
        family->microkernelOpName == opName)
      return family;
  }
  return nullptr;
}

bool candidateMatchesRVVMicrokernelFamily(
    const TargetArtifactCandidate &candidate,
    const RVVI32MicrokernelFamilySpec &family) {
  return candidate.origin == kRVVPluginName &&
         candidate.routeID == family.routeID &&
         candidate.emissionKind == family.emissionKind &&
         candidate.artifactKind == kMicrokernelArtifactKind &&
         candidate.runtimeABI == family.runtimeABI &&
         candidate.runtimeABIKind == family.runtimeABIKind &&
         candidate.runtimeABIName == family.runtimeABIName &&
         candidate.runtimeGlueRole == family.runtimeGlueRole;
}

bool candidateMatchesRVVRouteRegistration(
    const TargetArtifactCandidate &candidate,
    const RVVBinaryIntrinsicDescriptor &descriptor) {
  return candidate.origin == kRVVPluginName &&
         candidate.routeID == descriptor.getRVVRouteID() &&
         candidate.emissionKind == descriptor.family.emissionKind &&
         candidate.artifactKind == kMicrokernelArtifactKind &&
         candidate.runtimeABI == descriptor.getRVVRuntimeABI() &&
         candidate.runtimeABIKind == descriptor.getRVVRuntimeABIKind() &&
         candidate.runtimeABIName == descriptor.getRVVRuntimeABIName() &&
         candidate.runtimeGlueRole == descriptor.getRVVRuntimeGlueRole();
}

VariantOp getPathVariant(const SelectedPath &path) {
  return const_cast<SelectedPath &>(path).variant;
}

llvm::StringRef getPathVariantSymbol(const SelectedPath &path) {
  return getPathVariant(path).getSymName();
}

mlir::Operation *getPathVariantOperation(const SelectedPath &path) {
  return getPathVariant(path).getOperation();
}

std::optional<std::int64_t>
getSelectedVariantElementCount(const SelectedPath &path) {
  auto attr =
      getPathVariantOperation(path)->getAttrOfType<mlir::IntegerAttr>(
          kRVVElementCountAttrName);
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

llvm::Error makeMicrokernelError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV RVV microkernel C export failed";
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
      llvm::Twine("TianChen-RV RVV microkernel C export failed: ") + message,
      llvm::errc::invalid_argument);
}

llvm::Error makeMicrokernelObjectError(llvm::StringRef kernelSymbol,
                                       llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV RVV microkernel object export failed";
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
      llvm::Twine("TianChen-RV RVV microkernel object export failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool hasKernelBody(KernelOp kernel) {
  return kernel && !kernel.getBody().empty();
}

mlir::StringAttr getStringAttr(mlir::Operation *op, llvm::StringRef name) {
  return op ? op->getAttrOfType<mlir::StringAttr>(name) : mlir::StringAttr();
}

llvm::StringRef getAttrValue(mlir::Operation *op, llvm::StringRef attrName) {
  auto attr = getStringAttr(op, attrName);
  if (!attr)
    return {};
  return attr.getValue();
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

llvm::Error validateBoundedText(KernelOp kernel, llvm::StringRef fieldName,
                                llvm::StringRef value) {
  constexpr std::size_t kMaxTextLength = 512;
  if (value.empty() || value.size() > kMaxTextLength)
    return makeMicrokernelError(kernel, llvm::Twine(fieldName) +
                                            " must be bounded non-empty "
                                            "single-line metadata");

  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      return makeMicrokernelError(kernel, llvm::Twine(fieldName) +
                                              " must be bounded non-empty "
                                              "single-line metadata");
    if (byte < 0x20 && character != '\t')
      return makeMicrokernelError(kernel, llvm::Twine(fieldName) +
                                              " must be bounded non-empty "
                                              "single-line metadata");
  }

  if (value.contains("/*") || value.contains("*/"))
    return makeMicrokernelError(kernel, llvm::Twine(fieldName) +
                                            " must not contain C comment "
                                            "delimiter text");

  if (containsForbiddenText(value))
    return makeMicrokernelError(kernel, llvm::Twine(fieldName) +
                                            " must not contain secret-like or "
                                            "raw credential text");
  return llvm::Error::success();
}

llvm::Error validateBoundedCompileText(llvm::StringRef kernelSymbol,
                                       llvm::StringRef fieldName,
                                       llvm::StringRef value) {
  constexpr std::size_t kMaxTextLength = 128;
  if (value.empty() || value.size() > kMaxTextLength)
    return makeMicrokernelObjectError(
        kernelSymbol, llvm::Twine(fieldName) +
                          " must be bounded non-empty compile metadata");

  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (!std::isalnum(byte) && character != '_' && character != '.')
      return makeMicrokernelObjectError(
          kernelSymbol, llvm::Twine(fieldName) +
                            " must contain only bounded compile-flag "
                            "characters");
  }

  if (containsForbiddenText(value))
    return makeMicrokernelObjectError(
        kernelSymbol, llvm::Twine(fieldName) +
                          " must not contain secret-like or raw credential "
                          "text");
  return llvm::Error::success();
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

llvm::Error validateCParameterName(KernelOp kernel, llvm::StringRef value) {
  if (!isValidCParameterName(value))
    return makeMicrokernelError(
        kernel, llvm::Twine("runtime ABI parameter c_name '") + value +
                    "' must be a valid C identifier for RVV source export");
  return llvm::Error::success();
}

std::string getEffectiveRouteID(
    llvm::StringRef activeRouteID,
    const RVVI32MicrokernelFamilySpec &family) {
  if (!activeRouteID.empty())
    return activeRouteID.str();
  return family.routeID.str();
}

std::string getEffectiveRouteID(
    llvm::StringRef activeRouteID,
    const RVVBinaryIntrinsicDescriptor &descriptor) {
  if (!activeRouteID.empty())
    return activeRouteID.str();
  return descriptor.getRVVRouteID().str();
}

llvm::Error collectRuntimeABIParameters(
    KernelOp kernel, DiagnosticOp diagnostic,
    llvm::SmallVectorImpl<support::RuntimeABIParameter> &out) {
  auto parameters = diagnostic->getAttrOfType<mlir::ArrayAttr>(
      execDiagnostic::kRuntimeABIParametersAttrName);
  if (!parameters || parameters.empty())
    return makeMicrokernelError(
        kernel, "supported RVV microkernel emission-plan diagnostic requires "
                "non-empty runtime_abi_parameters metadata");

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
    if (llvm::Error error = validateCParameterName(kernel, cNameValue))
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

llvm::Error validateRVVBinaryCallableABIParameterMirror(
    KernelOp kernel,
    llvm::ArrayRef<support::RuntimeABIParameter> metadataParameters,
    llvm::ArrayRef<support::RuntimeABIParameter> irBackedParameters,
    llvm::StringRef metadataSource,
    const RVVBinaryIntrinsicDescriptor &descriptor) {
  return support::validateFiniteBinaryCallableABIParameterMirror(
      kernel, metadataParameters, irBackedParameters, metadataSource,
      target::rvv::getRVVBinaryRuntimeABIContract(descriptor.family));
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
    return makeMicrokernelError(kernel, "selected dispatch requires a "
                                        "materialized body block");

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
      paths.push_back(SelectedPath{variant, dispatchCase.getOperation(),
                                   kDispatchCaseRole.str()});
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
      paths.push_back(SelectedPath{variant, fallbackOp.getOperation(),
                                   kDispatchFallbackRole.str()});
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
        kernel, "requires a selected path surface before exporting an RVV "
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

  paths.push_back(
      SelectedPath{variant, marker.getOperation(), kDirectVariantRole.str()});
  return llvm::Error::success();
}

llvm::Error validateRequiredCapabilities(
    KernelOp kernel, VariantOp variant, const TargetCapabilitySet &capabilities,
    llvm::SmallVectorImpl<std::string> &out) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr || requiresAttr.empty())
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV variant @") + variant.getSymName() +
                    " requires non-empty structured 'requires' metadata");

  bool requiresRVV = false;
  for (mlir::Attribute attr : requiresAttr) {
    auto symbol = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
    if (!symbol || symbol.getValue().trim().empty())
      return makeMicrokernelError(
          kernel, llvm::Twine("selected RVV variant @") + variant.getSymName() +
                      " requires only non-empty capability symbol references");

    const CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbol.getValue());
    if (!capability)
      return makeMicrokernelError(
          kernel, llvm::Twine("selected RVV variant @") + variant.getSymName() +
                      " requires unknown capability @" + symbol.getValue());
    if (!capability->isAvailable())
      return makeMicrokernelError(
          kernel, llvm::Twine("selected RVV variant @") + variant.getSymName() +
                      " requires unavailable capability @" +
                      symbol.getValue());
    if (capability->satisfiesID(kRVVCapabilityID))
      requiresRVV = true;
    out.push_back(symbol.getValue().str());
  }

  if (!requiresRVV)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV variant @") + variant.getSymName() +
                    " must require capability id 'rvv'");

  return llvm::Error::success();
}

llvm::Expected<bool>
variantRequiresCapabilityID(KernelOp kernel, VariantOp variant,
                            const TargetCapabilitySet &capabilities,
                            llvm::StringRef id) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr || requiresAttr.empty())
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV variant @") + variant.getSymName() +
                    " requires non-empty structured 'requires' metadata");

  for (mlir::Attribute attr : requiresAttr) {
    auto symbol = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
    if (!symbol || symbol.getValue().trim().empty())
      return makeMicrokernelError(
          kernel, llvm::Twine("selected RVV variant @") + variant.getSymName() +
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

llvm::Expected<bool> variantRequiresConfig(
    KernelOp kernel, VariantOp variant, const TargetCapabilitySet &capabilities,
    const RVVI32VectorShapeConfig &config) {
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

llvm::Error requireConfigCapabilityProperty(
    KernelOp kernel, const TargetCapabilitySet &capabilities,
    llvm::StringRef id, llvm::StringRef propertyName,
    llvm::StringRef expectedValue) {
  const CapabilityDescriptor *capability = capabilities.lookupProviderByID(id);
  if (!capability || !capability->isAvailable())
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV path requires available capability "
                            "id '") +
                    id + "'");

  llvm::StringRef value = capability->getProperty(propertyName).trim();
  if (llvm::Error error = validateBoundedText(kernel, propertyName, value))
    return error;
  if (value != expectedValue)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV path capability id '") + id +
                    "' property '" + propertyName + "' must be '" +
                    expectedValue + "'");
  return llvm::Error::success();
}

llvm::Error validateSelectedVectorShapeMetadata(
    KernelOp kernel, mlir::Operation *op, llvm::StringRef context,
    const RVVI32VectorShapeConfig &config, llvm::StringRef shapeAttrName,
    llvm::StringRef sewAttrName, llvm::StringRef lmulAttrName,
    llvm::StringRef tailPolicyAttrName, llvm::StringRef maskPolicyAttrName,
    llvm::StringRef vectorTypeAttrName, llvm::StringRef vectorSuffixAttrName,
    llvm::StringRef setvlSuffixAttrName) {
  bool hasAnyMetadata = op->hasAttr(shapeAttrName) ||
                        op->hasAttr(sewAttrName) ||
                        op->hasAttr(lmulAttrName) ||
                        op->hasAttr(tailPolicyAttrName) ||
                        op->hasAttr(maskPolicyAttrName) ||
                        op->hasAttr(vectorTypeAttrName) ||
                        op->hasAttr(vectorSuffixAttrName) ||
                        op->hasAttr(setvlSuffixAttrName);
  if (!hasAnyMetadata)
    return llvm::Error::success();

  auto shape = op->getAttrOfType<mlir::StringAttr>(shapeAttrName);
  auto sew = op->getAttrOfType<mlir::IntegerAttr>(sewAttrName);
  auto lmul = op->getAttrOfType<mlir::StringAttr>(lmulAttrName);
  auto tailPolicy =
      op->getAttrOfType<mlir::StringAttr>(tailPolicyAttrName);
  auto maskPolicy =
      op->getAttrOfType<mlir::StringAttr>(maskPolicyAttrName);
  auto vectorType =
      op->getAttrOfType<mlir::StringAttr>(vectorTypeAttrName);
  auto vectorSuffix =
      op->getAttrOfType<mlir::StringAttr>(vectorSuffixAttrName);
  auto setvlSuffix =
      op->getAttrOfType<mlir::StringAttr>(setvlSuffixAttrName);
  if (!shape || !sew || !lmul || !tailPolicy || !maskPolicy || !vectorType ||
      !vectorSuffix || !setvlSuffix)
    return makeMicrokernelError(
        kernel, llvm::Twine(context) +
                    " selected vector-shape metadata must be complete when "
                    "any selected-shape attribute is present");

  auto validateStringField = [&](llvm::StringRef fieldName,
                                 llvm::StringRef observed,
                                 llvm::StringRef expected) -> llvm::Error {
    if (llvm::Error error = validateBoundedText(kernel, fieldName, observed))
      return error;
    if (observed != expected)
      return makeMicrokernelError(
          kernel, llvm::Twine(context) + " selected vector-shape " +
                      fieldName + " must be '" + expected + "'");
    return llvm::Error::success();
  };

  if (llvm::Error error = validateStringField(
          "shape", shape.getValue().trim(), config.shapeID))
    return error;
  if (sew.getInt() != config.sewBits)
    return makeMicrokernelError(
        kernel, llvm::Twine(context) +
                    " selected vector-shape sew must be '" +
                    llvm::Twine(config.sewBits) + "'");
  if (llvm::Error error =
          validateStringField("lmul", lmul.getValue().trim(), config.lmul))
    return error;
  if (llvm::Error error = validateStringField(
          "tail policy", tailPolicy.getValue().trim(), config.tailPolicy))
    return error;
  if (llvm::Error error = validateStringField(
          "mask policy", maskPolicy.getValue().trim(), config.maskPolicy))
    return error;
  if (llvm::Error error = validateStringField(
          "vector type", vectorType.getValue().trim(), config.vectorType))
    return error;
  if (llvm::Error error = validateStringField(
          "vector suffix", vectorSuffix.getValue().trim(),
          config.vectorSuffix))
    return error;
  if (llvm::Error error = validateStringField(
          "setvl suffix", setvlSuffix.getValue().trim(), config.setvlSuffix))
    return error;
  return llvm::Error::success();
}

llvm::Error validateSelectedDescriptorMatchesMicrokernelFamily(
    KernelOp kernel, const SelectedPath &path,
    const RVVI32MicrokernelFamilySpec &family) {
  auto descriptorAttr =
      getPathVariant(path)->getAttrOfType<mlir::StringAttr>(
          kRVVLoweringDescriptorAttrName);
  if (!descriptorAttr)
    return llvm::Error::success();

  llvm::StringRef descriptor = descriptorAttr.getValue().trim();
  if (llvm::Error error =
          validateBoundedText(kernel, kRVVLoweringDescriptorAttrName,
                              descriptor))
    return error;

  const RVVBinaryFamilyDescriptor *selectedFamily =
      lookupRVVBinaryFamilyRegistrationByLegacyLoweringDescriptor(descriptor);
  if (!selectedFamily)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV variant @") +
                    getPathVariantSymbol(path) +
                    " has unsupported tcrv_rvv.lowering_descriptor '" +
                    descriptor + "' for RVV binary microkernel export");
  if (selectedFamily->dtype != RVVBinaryDTypeKind::I32)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV variant @") +
                    getPathVariantSymbol(path) +
                    " tcrv_rvv.lowering_descriptor '" + descriptor +
                    "' does not describe an i32 RVV microkernel body");
  if (!isSameRVVBinaryFamily(*selectedFamily, family))
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV variant @") +
                    getPathVariantSymbol(path) +
                    " tcrv_rvv.lowering_descriptor '" + descriptor +
                    "' requires " + selectedFamily->microkernelOpName +
                    " but typed microkernel body is " +
                    family.microkernelOpName +
                    "; descriptor and typed body family must agree before "
                    "artifact export");
  return llvm::Error::success();
}

llvm::Error validateSelectedI64DescriptorMirrorMatchesTypedBody(
    KernelOp kernel, const SelectedPath &path,
    const RVVBinaryFamilyDescriptor &typedFamily) {
  auto descriptorAttr =
      getPathVariant(path)->getAttrOfType<mlir::StringAttr>(
          kRVVLoweringDescriptorAttrName);
  if (!descriptorAttr)
    return llvm::Error::success();

  llvm::StringRef descriptor = descriptorAttr.getValue().trim();
  if (llvm::Error error =
          validateBoundedText(kernel, kRVVLoweringDescriptorAttrName,
                              descriptor))
    return error;

  const RVVBinaryFamilyDescriptor *mirrorFamily =
      lookupRVVBinaryFamilyRegistrationByLegacyLoweringDescriptor(descriptor);
  if (!mirrorFamily)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV variant @") +
                    getPathVariantSymbol(path) +
                    " has unsupported tcrv_rvv.lowering_descriptor '" +
                    descriptor +
                    "' for RVV i64 target artifact export; the selected typed "
                    "RVV i64 microkernel body is authoritative");

  if (!isSameRVVBinaryFamily(*mirrorFamily, typedFamily))
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV variant @") +
                    getPathVariantSymbol(path) +
                    " tcrv_rvv.lowering_descriptor '" + descriptor +
                    "' is non-authoritative legacy mirror metadata for " +
                    mirrorFamily->microkernelOpName +
                    " but the selected typed RVV i64 microkernel body is " +
                    typedFamily.microkernelOpName +
                    "; typed body is authoritative for RVV target artifact "
                    "export");
  return llvm::Error::success();
}

llvm::Expected<const RVVBinaryFamilyDescriptor *>
resolveSelectedI64FamilyForPath(KernelOp kernel, const SelectedPath &path) {
  const RVVBinaryFamilyDescriptor *matchedFamily = nullptr;
  unsigned matches = 0;
  for (mlir::Operation &op : kernel.getBody().front()) {
    const RVVBinaryFamilyDescriptor *candidateFamily =
        getI64MicrokernelFamilyForOp(&op);
    if (!candidateFamily)
      continue;

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    auto role = op.getAttrOfType<mlir::StringAttr>(
        execDiagnostic::kRoleAttrName);
    if (!selectedVariant || !role ||
        selectedVariant.getValue() != getPathVariantSymbol(path) ||
        role.getValue() != path.role)
      continue;

    ++matches;
    matchedFamily = candidateFamily;
  }

  if (matches > 1)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV path @") + getPathVariantSymbol(path) +
                    " as " + path.role +
                    " has duplicate selected typed RVV i64 microkernel bodies; "
                    "typed body is authoritative for RVV target artifact export");

  if (matchedFamily) {
    if (llvm::Error error =
            validateSelectedI64DescriptorMirrorMatchesTypedBody(
                kernel, path, *matchedFamily))
      return std::move(error);
    return matchedFamily;
  }

  if (auto descriptorAttr =
          getPathVariant(path)->getAttrOfType<mlir::StringAttr>(
              kRVVLoweringDescriptorAttrName)) {
    llvm::StringRef descriptor = descriptorAttr.getValue().trim();
    if (llvm::Error error =
            validateBoundedText(kernel, kRVVLoweringDescriptorAttrName,
                                descriptor))
      return std::move(error);

    const RVVBinaryFamilyDescriptor *descriptorFamily =
        lookupRVVBinaryFamilyRegistrationByLegacyLoweringDescriptor(descriptor);
    if (descriptorFamily && descriptorFamily->dtype == RVVBinaryDTypeKind::I64)
      return makeMicrokernelError(
          kernel, llvm::Twine("selected RVV variant @") +
                      getPathVariantSymbol(path) +
                      " carries tcrv_rvv.lowering_descriptor '" + descriptor +
                      "' for " + descriptorFamily->microkernelOpName +
                      " but has no selected typed RVV i64 microkernel body; "
                      "typed body is authoritative for RVV target artifact "
                      "export and descriptor-only i64 export is rejected");
  }
  return matchedFamily;
}

llvm::Expected<const SelectedPlanMetadataEntry *>
findUniqueRVVMicrokernelSelectedPlanMetadataEntry(
    const TargetArtifactCandidate &candidate, llvm::StringRef name) {
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
    return makeMicrokernelError(
        candidate.kernel,
        llvm::Twine("selected RVV target artifact candidate @") +
            candidate.selectedVariant + " requires selected_plan_metadata '" +
            name + "'");
  if (count > 1)
    return makeMicrokernelError(
        candidate.kernel,
        llvm::Twine("selected RVV target artifact candidate @") +
            candidate.selectedVariant +
            " has duplicate selected_plan_metadata '" + name + "'");
  return match;
}

llvm::Error validateRVVMicrokernelSelectedPlanMetadataEntry(
    const TargetArtifactCandidate &candidate,
    const RVVVectorShapeSelectedPlanMetadataDescriptor &expected) {
  llvm::Expected<const SelectedPlanMetadataEntry *> metadata =
      findUniqueRVVMicrokernelSelectedPlanMetadataEntry(candidate,
                                                       expected.name);
  if (!metadata)
    return metadata.takeError();

  if ((*metadata)->value != expected.value)
    return makeMicrokernelError(
        candidate.kernel,
        llvm::Twine("selected RVV target artifact candidate @") +
            candidate.selectedVariant + " selected_plan_metadata '" +
            expected.name + "' " + expected.diagnosticSpelling +
            " must be '" + expected.value + "'");
  if ((*metadata)->role != expected.role)
    return makeMicrokernelError(
        candidate.kernel,
        llvm::Twine("selected RVV target artifact candidate @") +
            candidate.selectedVariant + " selected_plan_metadata '" +
            expected.name + "' role must be '" + expected.role + "'");
  if ((*metadata)->note != expected.note)
    return makeMicrokernelError(
        candidate.kernel,
        llvm::Twine("selected RVV target artifact candidate @") +
            candidate.selectedVariant + " selected_plan_metadata '" +
            expected.name + "' note must be '" + expected.note + "'");
  return llvm::Error::success();
}

llvm::Expected<const RVVVectorShapeConfig *>
resolveRVVMicrokernelCandidateSelectedShape(
    const TargetArtifactCandidate &candidate,
    const RVVBinaryFamilyDescriptor &family) {
  llvm::Expected<const SelectedPlanMetadataEntry *> shapeMetadata =
      findUniqueRVVMicrokernelSelectedPlanMetadataEntry(
          candidate, getRVVSelectedVectorShapeAttrName());
  if (!shapeMetadata)
    return shapeMetadata.takeError();

  const RVVVectorShapeConfig *metadataConfig =
      lookupRVVBinaryFamilyShapeConfigByID(family, (*shapeMetadata)->value);
  if (!metadataConfig)
    return makeMicrokernelError(
        candidate.kernel,
        llvm::Twine("selected RVV target artifact candidate @") +
            candidate.selectedVariant +
            " selected_plan_metadata 'tcrv_rvv.selected_vector_shape' has "
            "unsupported finite shape '" +
            (*shapeMetadata)->value + "' for family '" + family.familyID + "'");
  return metadataConfig;
}

llvm::Expected<llvm::StringRef>
resolveRVVMicrokernelRuntimeElementCountCName(
    const TargetArtifactCandidate &candidate) {
  const support::RuntimeABIParameter *match = nullptr;
  unsigned count = 0;
  for (const support::RuntimeABIParameter &parameter :
       candidate.runtimeABIParameters) {
    if (parameter.role !=
        support::RuntimeABIParameterRole::RuntimeElementCount)
      continue;
    match = &parameter;
    ++count;
  }

  if (count == 0)
    return makeMicrokernelError(
        candidate.kernel,
        llvm::Twine("selected RVV target artifact candidate @") +
            candidate.selectedVariant +
            " requires one runtime-element-count ABI parameter before "
            "selected config metadata validation");
  if (count > 1)
    return makeMicrokernelError(
        candidate.kernel,
        llvm::Twine("selected RVV target artifact candidate @") +
            candidate.selectedVariant +
            " has duplicate runtime-element-count ABI parameters before "
            "selected config metadata validation");
  if (match->cName.empty())
    return makeMicrokernelError(
        candidate.kernel,
        llvm::Twine("selected RVV target artifact candidate @") +
            candidate.selectedVariant +
            " runtime-element-count ABI parameter requires a non-empty C "
            "name");
  return llvm::StringRef(match->cName);
}

llvm::Expected<std::int64_t>
resolveRVVMicrokernelDescriptorElementCountMetadata(
    const TargetArtifactCandidate &candidate) {
  llvm::Expected<const SelectedPlanMetadataEntry *> metadata =
      findUniqueRVVMicrokernelSelectedPlanMetadataEntry(
          candidate, getRVVDescriptorElementCountMetadataName());
  if (!metadata)
    return metadata.takeError();

  if ((*metadata)->role != getRVVLegacyDescriptorMirrorMetadataRole())
    return makeMicrokernelError(
        candidate.kernel,
        llvm::Twine("selected RVV target artifact candidate @") +
            candidate.selectedVariant + " selected_plan_metadata '" +
            getRVVDescriptorElementCountMetadataName() +
            "' role must be '" + getRVVLegacyDescriptorMirrorMetadataRole() +
            "'");
  if ((*metadata)->note != getRVVLegacyDescriptorMirrorMetadataNote())
    return makeMicrokernelError(
        candidate.kernel,
        llvm::Twine("selected RVV target artifact candidate @") +
            candidate.selectedVariant + " selected_plan_metadata '" +
            getRVVDescriptorElementCountMetadataName() +
            "' note must be '" + getRVVLegacyDescriptorMirrorMetadataNote() +
            "'");

  std::int64_t value = 0;
  if (llvm::StringRef((*metadata)->value).getAsInteger(10, value) ||
      value <= 0 || value > 64)
    return makeMicrokernelError(
        candidate.kernel,
        llvm::Twine("selected RVV target artifact candidate @") +
            candidate.selectedVariant + " selected_plan_metadata '" +
            getRVVDescriptorElementCountMetadataName() +
            "' descriptor-local element_count must be an integer in the "
            "bounded smoke range [1, 64]");
  return value;
}

llvm::Error validateRVVMicrokernelDescriptorElementCountMetadata(
    const TargetArtifactCandidate &candidate,
    const RVVBinarySelectedConfigContract &contract) {
  llvm::Expected<std::int64_t> metadataElementCount =
      resolveRVVMicrokernelDescriptorElementCountMetadata(candidate);
  if (!metadataElementCount)
    return metadataElementCount.takeError();

  if (*metadataElementCount != contract.getDescriptorElementCount())
    return makeMicrokernelError(
        candidate.kernel,
        llvm::Twine("selected RVV target artifact candidate @") +
            candidate.selectedVariant + " selected_plan_metadata '" +
            getRVVDescriptorElementCountMetadataName() +
            "' descriptor-local element_count layer is stale; expected " +
            llvm::Twine(contract.getDescriptorElementCount()) +
            " from the selected config/runtime AVL contract but found " +
            llvm::Twine(*metadataElementCount));
  return llvm::Error::success();
}

llvm::Error validateRVVMicrokernelSelectedPlanMetadata(
    const TargetArtifactCandidate &candidate,
    const RVVBinarySelectedConfigContract &contract) {
  llvm::SmallVector<RVVVectorShapeSelectedPlanMetadataDescriptor, 24> expected;
  appendRVVBinarySelectedVectorShapeMetadata(contract, expected);
  appendRVVBinaryRuntimeVLBoundarySelectedPlanMetadata(contract, expected);
  if (contract.getFamily().dtype == RVVBinaryDTypeKind::I32 ||
      contract.getFamily().dtype == RVVBinaryDTypeKind::I64)
    appendRVVBinarySelectedTypedSourceMetadata(contract, expected);
  else
    appendRVVBinaryLegacyDescriptorMirrorMetadata(contract, expected);

  for (const RVVVectorShapeSelectedPlanMetadataDescriptor &entry : expected)
    if (llvm::Error error =
            validateRVVMicrokernelSelectedPlanMetadataEntry(candidate, entry))
      return error;
  return validateRVVMicrokernelDescriptorElementCountMetadata(candidate,
                                                            contract);
}

llvm::Error validateRVVMicrokernelSelectedPlanMetadata(
    const TargetArtifactCandidate &candidate,
    const RVVBinaryIntrinsicDescriptor &descriptor) {
  llvm::Expected<llvm::StringRef> runtimeElementCountCName =
      resolveRVVMicrokernelRuntimeElementCountCName(candidate);
  if (!runtimeElementCountCName)
    return runtimeElementCountCName.takeError();

  llvm::Expected<std::int64_t> descriptorElementCount =
      resolveRVVMicrokernelDescriptorElementCountMetadata(candidate);
  if (!descriptorElementCount)
    return descriptorElementCount.takeError();

  llvm::Expected<RVVBinarySelectedConfigContract> selectedConfig =
      buildRVVBinarySelectedConfigContract(
          descriptor.family, *descriptor.shape, candidate.selectedVariant,
          candidate.role, *descriptorElementCount,
          *runtimeElementCountCName);
  if (!selectedConfig)
    return selectedConfig.takeError();
  return validateRVVMicrokernelSelectedPlanMetadata(candidate, *selectedConfig);
}

bool hasMatchingRVVMicrokernelAttachmentForCandidate(
    const TargetArtifactCandidate &candidate,
    const RVVBinaryFamilyDescriptor &family) {
  KernelOp kernel = candidate.kernel;
  if (!kernel || kernel.getBody().empty())
    return false;

  for (mlir::Operation &op : kernel.getBody().front()) {
    const RVVBinaryFamilyDescriptor *opFamily = getI32MicrokernelFamilyForOp(&op);
    if (!opFamily)
      opFamily = getI64MicrokernelFamilyForOp(&op);
    if (!opFamily || opFamily->familyID != family.familyID)
      continue;

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    auto role =
        op.getAttrOfType<mlir::StringAttr>(execDiagnostic::kRoleAttrName);
    if (selectedVariant && role &&
        selectedVariant.getValue() == candidate.selectedVariant &&
        role.getValue() == candidate.role)
      return true;
  }

  return false;
}

llvm::Expected<const RVVI32VectorShapeConfig *>
getSelectedRVVConfig(KernelOp kernel, VariantOp variant,
                     const TargetCapabilitySet &capabilities) {
  const RVVI32VectorShapeConfig *selectedConfig = nullptr;
  unsigned matchedConfigs = 0;
  for (const RVVI32VectorShapeConfig *config :
       {&getI32M1ConfigSpec(), &getI32M2ConfigSpec(),
        &getI64M1VectorShapeConfig()}) {
    llvm::Expected<bool> requiresConfig =
        variantRequiresConfig(kernel, variant, capabilities, *config);
    if (!requiresConfig)
      return requiresConfig.takeError();
    if (!*requiresConfig)
      continue;

    selectedConfig = config;
    ++matchedConfigs;
  }

  if (matchedConfigs > 1)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV variant @") + variant.getSymName() +
                    " must require exactly one finite RVV vector-shape config "
                    "shape, not multiple dtype/LMUL configs");
  if (!selectedConfig)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV variant @") + variant.getSymName() +
                    " must require one finite RVV vector-shape config "
                    "capability set: i32m1, i32m2, or i64m1");

  if (llvm::Error error = requireConfigCapabilityProperty(
          kernel, capabilities, selectedConfig->sewCapabilityID,
          kSEWBitsPropertyName, std::to_string(selectedConfig->sewBits)))
    return std::move(error);
  if (llvm::Error error = requireConfigCapabilityProperty(
          kernel, capabilities, selectedConfig->lmulCapabilityID,
          kLMULPropertyName, selectedConfig->lmul))
    return std::move(error);
  if (llvm::Error error = requireConfigCapabilityProperty(
          kernel, capabilities, selectedConfig->tailPolicyCapabilityID,
          kTailPolicyPropertyName, selectedConfig->tailPolicy))
    return std::move(error);
  if (llvm::Error error = requireConfigCapabilityProperty(
          kernel, capabilities, selectedConfig->maskPolicyCapabilityID,
          kMaskPolicyPropertyName, selectedConfig->maskPolicy))
    return std::move(error);
  if (llvm::Error error =
          validateSelectedVectorShapeMetadata(
              kernel, variant.getOperation(),
              (llvm::Twine("selected RVV variant @") + variant.getSymName())
                  .str(),
              *selectedConfig, kRVVSelectedVectorShapeAttrName,
              kRVVSelectedVectorSEWAttrName, kRVVSelectedVectorLMULAttrName,
              kRVVSelectedTailPolicyAttrName, kRVVSelectedMaskPolicyAttrName,
              kRVVSelectedVectorTypeAttrName,
              kRVVSelectedVectorSuffixAttrName,
              kRVVSelectedSetVLSuffixAttrName))
    return std::move(error);

  return selectedConfig;
}

llvm::Expected<std::string>
getRequiredTargetTriple(KernelOp kernel,
                        const TargetCapabilitySet &capabilities) {
  const CapabilityDescriptor *rvvCapability =
      capabilities.lookupProviderByID(kRVVCapabilityID);
  if (!rvvCapability || !rvvCapability->isAvailable())
    return makeMicrokernelError(
        kernel,
        "selected RVV path requires an available RVV capability provider "
        "before object export");

  llvm::StringRef architecture =
      rvvCapability->getProperty(kArchitecturePropertyName).trim();
  if (llvm::Error error =
          validateBoundedText(kernel, kArchitecturePropertyName, architecture))
    return std::move(error);
  if (architecture != "riscv64")
    return makeMicrokernelError(
        kernel,
        llvm::Twine("selected RVV path requires architecture capability "
                    "metadata 'riscv64', got '") +
            architecture + "'");

  return architecture.str();
}

llvm::Expected<std::string>
getRequiredSelectedMarch(KernelOp kernel, VariantOp variant,
                         const TargetCapabilitySet &capabilities) {
  std::string requiredMarch;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, variant.getOperation(),
                                kRVVRequiredMarchAttrName,
                                "selected RVV variant", requiredMarch))
    return std::move(error);
  if (!hasRVVVectorHint(requiredMarch))
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV variant @") + variant.getSymName() +
                    " attribute 'tcrv_rvv.required_march' must contain RVV "
                    "vector evidence");

  llvm::SmallVector<std::string, 2> preservedMarches;
  if (const CapabilityDescriptor *compileRun =
          capabilities.lookupProviderByID(kRVVProbeCompileRunCapabilityID)) {
    if (compileRun->isAvailable()) {
      llvm::StringRef value =
          compileRun->getProperty(kSelectedMarchPropertyName).trim();
      if (!value.empty()) {
        if (llvm::Error error =
                validateBoundedText(kernel, kSelectedMarchPropertyName, value))
          return std::move(error);
        preservedMarches.push_back(value.str());
      }
    }
  }

  if (const CapabilityDescriptor *march =
          capabilities.lookupProviderByID(kRVVToolchainMarchCapabilityID)) {
    if (march->isAvailable()) {
      llvm::StringRef value = march->getProperty(kValuePropertyName).trim();
      if (!value.empty()) {
        if (llvm::Error error =
                validateBoundedText(kernel, kValuePropertyName, value))
          return std::move(error);
        preservedMarches.push_back(value.str());
      }
    }
  }

  if (preservedMarches.empty())
    return makeMicrokernelError(
        kernel, "selected RVV path requires available preserved selected_march "
                "metadata from capability id 'rvv.probe.compile_run' or "
                "'rvv.toolchain.march'");

  if (!llvm::is_contained(preservedMarches, requiredMarch))
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV variant @") + variant.getSymName() +
                    " 'tcrv_rvv.required_march' metadata is not satisfied by "
                    "preserved selected_march capability metadata");

  return requiredMarch;
}

llvm::Error getOptionalSelectedMABI(KernelOp kernel,
                                    const TargetCapabilitySet &capabilities,
                                    std::optional<std::string> &out) {
  auto mergeMABI = [&](llvm::StringRef fieldName,
                       llvm::StringRef value) -> llvm::Error {
    value = value.trim();
    if (value.empty())
      return llvm::Error::success();
    if (llvm::Error error = validateBoundedText(kernel, fieldName, value))
      return error;
    if (out && *out != value)
      return makeMicrokernelError(
          kernel, "conflicting preserved selected_mabi capability metadata");
    out = value.str();
    return llvm::Error::success();
  };

  if (const CapabilityDescriptor *compileRun =
          capabilities.lookupProviderByID(kRVVProbeCompileRunCapabilityID)) {
    if (compileRun->isAvailable())
      if (llvm::Error error =
              mergeMABI(kSelectedMABIPropertyName,
                        compileRun->getProperty(kSelectedMABIPropertyName)))
        return error;
  }

  if (const CapabilityDescriptor *mabi =
          capabilities.lookupProviderByID(kRVVToolchainMABICapabilityID)) {
    if (mabi->isAvailable())
      if (llvm::Error error =
              mergeMABI(kValuePropertyName, mabi->getProperty(kValuePropertyName)))
        return error;
  }
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

llvm::Error validateBoundaryForPath(KernelOp kernel, const SelectedPath &path,
                                    LoweringBoundaryOp boundary,
                                    const RVVI32VectorShapeConfig &selectedConfig) {
  if (!boundary)
    return makeMicrokernelError(kernel, "requires a matching "
                                        "tcrv_rvv.lowering_boundary");
  if (boundary->getParentOp() != kernel.getOperation())
    return makeMicrokernelError(
        kernel, "matching tcrv_rvv.lowering_boundary must be a direct child "
                "of the selected kernel");

  std::string sourceKernel;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, boundary.getOperation(),
                                kSourceKernelAttrName,
                                "tcrv_rvv.lowering_boundary", sourceKernel))
    return error;
  if (sourceKernel != kernel.getSymName())
    return makeMicrokernelError(
        kernel, llvm::Twine("tcrv_rvv.lowering_boundary source_kernel '") +
                    sourceKernel + "' does not match selected kernel @" +
                    kernel.getSymName());

  std::string origin;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, boundary.getOperation(),
                                execDiagnostic::kOriginAttrName,
                                "tcrv_rvv.lowering_boundary", origin))
    return error;
  if (origin != kRVVPluginName)
    return makeMicrokernelError(
        kernel, "tcrv_rvv.lowering_boundary origin must be 'rvv-plugin'");

  std::string role;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, boundary.getOperation(),
                                execDiagnostic::kRoleAttrName,
                                "tcrv_rvv.lowering_boundary", role))
    return error;
  if (role != path.role)
    return makeMicrokernelError(
        kernel, llvm::Twine("tcrv_rvv.lowering_boundary role '") + role +
                    "' does not match selected RVV path role '" + path.role +
                    "'");

  std::string status;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, boundary.getOperation(),
                                execDiagnostic::kStatusAttrName,
                                "tcrv_rvv.lowering_boundary", status))
    return error;
  if (status != kUnsupportedStatusValue)
    return makeMicrokernelError(
        kernel, "tcrv_rvv.lowering_boundary status must remain 'unsupported' "
                "for this first executable microkernel export slice");

  auto selectedVariant =
      boundary->getAttrOfType<mlir::FlatSymbolRefAttr>(
          kSelectedVariantAttrName);
  if (!selectedVariant || selectedVariant.getValue() != getPathVariantSymbol(path))
    return makeMicrokernelError(
        kernel, "tcrv_rvv.lowering_boundary selected_variant does not match "
                "the selected RVV path");

  auto boundaryCapabilities =
      boundary->getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName);
  auto variantRequires =
      path.variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!arrayAttrsEqual(boundaryCapabilities, variantRequires))
    return makeMicrokernelError(
        kernel, "tcrv_rvv.lowering_boundary required_capabilities must match "
                "selected variant requires metadata");

  if (llvm::Error error =
          validateSelectedVectorShapeMetadata(
              kernel, boundary.getOperation(),
              (llvm::Twine("tcrv_rvv.lowering_boundary for @") +
               getPathVariantSymbol(path))
                  .str(),
              selectedConfig, kBoundarySelectedVectorShapeAttrName,
              kBoundarySelectedVectorSEWAttrName,
              kBoundarySelectedVectorLMULAttrName,
              kBoundarySelectedTailPolicyAttrName,
              kBoundarySelectedMaskPolicyAttrName,
              kBoundarySelectedVectorTypeAttrName,
              kBoundarySelectedVectorSuffixAttrName,
              kBoundarySelectedSetVLSuffixAttrName))
    return error;

  return llvm::Error::success();
}

llvm::Error findAndValidateBoundary(
    KernelOp kernel, const SelectedPath &path,
    const llvm::StringSet<> &selectedRVVPathKeys,
    const RVVI32VectorShapeConfig &selectedConfig,
    LoweringBoundaryOp &matchedBoundary) {
  unsigned matches = 0;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto boundary = llvm::dyn_cast<LoweringBoundaryOp>(op);
    if (!boundary)
      continue;

    auto selectedVariant =
        boundary->getAttrOfType<mlir::FlatSymbolRefAttr>(
            kSelectedVariantAttrName);
    auto role = boundary->getAttrOfType<mlir::StringAttr>(
        execDiagnostic::kRoleAttrName);
    if (!selectedVariant || !role)
      continue;

    std::string key = makePathKey(selectedVariant.getValue(), role.getValue());
    if (!selectedRVVPathKeys.count(key))
      return makeMicrokernelError(
          kernel, llvm::Twine("stale tcrv_rvv.lowering_boundary for @") +
                      selectedVariant.getValue() + " as " + role.getValue() +
                      " is not selected by the current RVV microkernel "
                      "surface");

    if (selectedVariant.getValue() == getPathVariantSymbol(path) &&
        role.getValue() == path.role) {
      ++matches;
      matchedBoundary = boundary;
    }
  }

  if (matches == 0)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV path @") + getPathVariantSymbol(path) +
                    " as " + path.role +
                    " requires exactly one matching "
                    "tcrv_rvv.lowering_boundary");
  if (matches > 1)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV path @") + getPathVariantSymbol(path) +
                    " as " + path.role +
                    " has duplicate tcrv_rvv.lowering_boundary metadata");

  return validateBoundaryForPath(kernel, path, matchedBoundary, selectedConfig);
}

llvm::Error validateMicrokernelForPath(
    KernelOp kernel, const SelectedPath &path, llvm::StringRef selectedMarch,
    const std::optional<std::string> &selectedMABI,
    PolicyAttr expectedPolicy, mlir::Operation *microkernel,
    const RVVI32MicrokernelFamilySpec &family,
    const RVVI32VectorShapeConfig &selectedConfig,
    llvm::StringRef activeRouteID,
    std::int64_t &elementCount, std::int64_t &controlPlaneSEW,
    std::string &controlPlaneLMUL, RVVIntrinsicConfig &intrinsicConfig,
    RVVI32VAddDataflowEmissionPlan &dataflowPlan) {
  if (!microkernel)
    return makeMicrokernelError(kernel, llvm::Twine("requires a matching ") +
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
        kernel, llvm::Twine(family.microkernelOpName) + " source_kernel '" +
                    sourceKernel + "' does not match selected kernel @" +
                    kernel.getSymName());

  std::string origin;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, microkernel,
                                execDiagnostic::kOriginAttrName,
                                family.microkernelOpName, origin))
    return error;
  if (origin != kRVVPluginName)
    return makeMicrokernelError(
        kernel, llvm::Twine(family.microkernelOpName) +
                    " origin must be 'rvv-plugin'");

  std::string role;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, microkernel,
                                execDiagnostic::kRoleAttrName,
                                family.microkernelOpName, role))
    return error;
  if (role != path.role)
    return makeMicrokernelError(
        kernel, llvm::Twine(family.microkernelOpName) + " role '" + role +
                    "' does not match selected RVV path role '" + path.role +
                    "'");

  auto selectedVariant =
      microkernel->getAttrOfType<mlir::FlatSymbolRefAttr>(
          kSelectedVariantAttrName);
  if (!selectedVariant || selectedVariant.getValue() != getPathVariantSymbol(path))
    return makeMicrokernelError(
        kernel, llvm::Twine(family.microkernelOpName) +
                    " selected_variant does not match the selected RVV path");

  auto microkernelCapabilities =
      microkernel->getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName);
  auto variantRequires =
      path.variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!arrayAttrsEqual(microkernelCapabilities, variantRequires))
    return makeMicrokernelError(
        kernel, llvm::Twine(family.microkernelOpName) +
                    " required_capabilities must match selected variant "
                    "requires metadata");

  if (llvm::Error error =
          validateSelectedVectorShapeMetadata(
              kernel, microkernel, family.microkernelOpName, selectedConfig,
              kBoundarySelectedVectorShapeAttrName,
              kBoundarySelectedVectorSEWAttrName,
              kBoundarySelectedVectorLMULAttrName,
              kBoundarySelectedTailPolicyAttrName,
              kBoundarySelectedMaskPolicyAttrName,
              kBoundarySelectedVectorTypeAttrName,
              kBoundarySelectedVectorSuffixAttrName,
              kBoundarySelectedSetVLSuffixAttrName))
    return error;

  if (llvm::Error error =
          validateSelectedDescriptorMatchesMicrokernelFamily(kernel, path,
                                                            family))
    return error;

  std::string microkernelMarch;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, microkernel, kRequiredMarchAttrName,
                                family.microkernelOpName,
                                microkernelMarch))
    return error;
  if (microkernelMarch != selectedMarch)
    return makeMicrokernelError(
        kernel, llvm::Twine(family.microkernelOpName) +
                    " required_march must match selected RVV march metadata");

  if (auto mabi =
          microkernel->getAttrOfType<mlir::StringAttr>(kSelectedMABIAttrName)) {
    llvm::StringRef value = mabi.getValue().trim();
    if (llvm::Error error =
            validateBoundedText(kernel, kSelectedMABIAttrName, value))
      return error;
    if (!selectedMABI || *selectedMABI != value)
      return makeMicrokernelError(
          kernel, llvm::Twine(family.microkernelOpName) +
                      " selected_mabi must match preserved selected_mabi "
                      "capability metadata");
  }

  RVVBinaryIntrinsicDescriptor descriptor =
      getI32BinaryIntrinsicDescriptorForMicrokernel(family, selectedConfig);
  llvm::SmallVector<support::RuntimeABIParameter, 4> callableABIParameters =
      descriptor.getCallableRuntimeABIParameters();
  RVVBinaryMicrokernelBodyValidationRequest request;
  request.kernel = kernel;
  request.microkernel = microkernel;
  request.descriptor = descriptor;
  request.selectedPolicy = expectedPolicy;
  request.activeRouteID = activeRouteID;
  request.callableABIParameters = callableABIParameters;
  request.expectedDescriptorElementCount = getSelectedVariantElementCount(path);
  llvm::Expected<RVVBinaryMicrokernelBodyValidationResult> validation =
      validateRVVBinaryMicrokernelBody(request);
  if (!validation)
    return validation.takeError();

  elementCount = validation->elementCount;
  controlPlaneSEW = validation->controlPlaneSEW;
  controlPlaneLMUL = std::move(validation->controlPlaneLMUL);
  intrinsicConfig = std::move(validation->intrinsicConfig);
  dataflowPlan = std::move(validation->dataflowPlan);
  return llvm::Error::success();
}

llvm::Error findAndValidateMicrokernel(
    KernelOp kernel, const SelectedPath &path,
    const llvm::StringSet<> &selectedRVVPathKeys, llvm::StringRef selectedMarch,
    const std::optional<std::string> &selectedMABI,
    PolicyAttr expectedPolicy, mlir::Operation *&matchedMicrokernel,
    const RVVI32MicrokernelFamilySpec *&matchedFamily,
    const RVVI32VectorShapeConfig &selectedConfig,
    llvm::StringRef activeRouteID,
    std::int64_t &elementCount, std::int64_t &controlPlaneSEW,
    std::string &controlPlaneLMUL, RVVIntrinsicConfig &intrinsicConfig,
    RVVI32VAddDataflowEmissionPlan &dataflowPlan) {
  unsigned matches = 0;
  for (mlir::Operation &op : kernel.getBody().front()) {
    const RVVI32MicrokernelFamilySpec *family =
        getI32MicrokernelFamilyForOp(&op);
    if (!family)
      continue;

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(
            kSelectedVariantAttrName);
    auto role = op.getAttrOfType<mlir::StringAttr>(
        execDiagnostic::kRoleAttrName);
    if (!selectedVariant || !role)
      continue;

    std::string key = makePathKey(selectedVariant.getValue(), role.getValue());
    if (!selectedRVVPathKeys.count(key))
      return makeMicrokernelError(
          kernel, llvm::Twine("stale ") + family->microkernelOpName + " for @" +
                      selectedVariant.getValue() + " as " + role.getValue() +
                      " is not selected by the current RVV microkernel "
                      "surface");

    if (selectedVariant.getValue() == getPathVariantSymbol(path) &&
        role.getValue() == path.role) {
      ++matches;
      matchedMicrokernel = &op;
      matchedFamily = family;
    }
  }

  if (matches == 0)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV path @") + getPathVariantSymbol(path) +
                    " as " + path.role +
                    " requires exactly one matching "
                    "RVV i32 microkernel");
  if (matches > 1)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV path @") + getPathVariantSymbol(path) +
                    " as " + path.role +
                    " has duplicate RVV i32 microkernel metadata");

  return validateMicrokernelForPath(kernel, path, selectedMarch, selectedMABI,
                                    expectedPolicy, matchedMicrokernel,
                                    *matchedFamily, selectedConfig,
                                    activeRouteID,
                                    elementCount, controlPlaneSEW,
                                    controlPlaneLMUL, intrinsicConfig,
                                    dataflowPlan);
}

llvm::Error validateI64MicrokernelForPath(
    KernelOp kernel, const SelectedPath &path, llvm::StringRef selectedMarch,
    const std::optional<std::string> &selectedMABI,
    PolicyAttr expectedPolicy, mlir::Operation *microkernel,
    const RVVBinaryIntrinsicDescriptor &descriptor,
    const RVVI32VectorShapeConfig &selectedConfig,
    llvm::StringRef activeRouteID,
    std::int64_t &elementCount, std::int64_t &controlPlaneSEW,
    std::string &controlPlaneLMUL, RVVIntrinsicConfig &intrinsicConfig,
    RVVI32VAddDataflowEmissionPlan &dataflowPlan) {
  if (!microkernel)
    return makeMicrokernelError(kernel, llvm::Twine("requires a matching ") +
                                            descriptor.getRVVMicrokernelOpName());
  if (microkernel->getParentOp() != kernel.getOperation())
    return makeMicrokernelError(
        kernel, llvm::Twine("matching ") +
                    descriptor.getRVVMicrokernelOpName() +
                    " must be a direct child of the selected kernel");

  std::string sourceKernel;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, microkernel,
                                kSourceKernelAttrName,
                                descriptor.getRVVMicrokernelOpName(),
                                sourceKernel))
    return error;
  if (sourceKernel != kernel.getSymName())
    return makeMicrokernelError(
        kernel, llvm::Twine(descriptor.getRVVMicrokernelOpName()) +
                    " source_kernel '" + sourceKernel +
                    "' does not match selected kernel @" +
                    kernel.getSymName());

  std::string origin;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, microkernel,
                                execDiagnostic::kOriginAttrName,
                                descriptor.getRVVMicrokernelOpName(), origin))
    return error;
  if (origin != kRVVPluginName)
    return makeMicrokernelError(
        kernel, llvm::Twine(descriptor.getRVVMicrokernelOpName()) +
                    " origin must be 'rvv-plugin'");

  std::string role;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, microkernel,
                                execDiagnostic::kRoleAttrName,
                                descriptor.getRVVMicrokernelOpName(), role))
    return error;
  if (role != path.role)
    return makeMicrokernelError(
        kernel, llvm::Twine(descriptor.getRVVMicrokernelOpName()) +
                    " role '" + role +
                    "' does not match selected RVV path role '" + path.role +
                    "'");

  auto selectedVariant =
      microkernel->getAttrOfType<mlir::FlatSymbolRefAttr>(
          kSelectedVariantAttrName);
  if (!selectedVariant ||
      selectedVariant.getValue() != getPathVariantSymbol(path))
    return makeMicrokernelError(
        kernel, llvm::Twine(descriptor.getRVVMicrokernelOpName()) +
                    " selected_variant does not match the selected RVV path");

  auto microkernelCapabilities =
      microkernel->getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName);
  auto variantRequires =
      path.variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!arrayAttrsEqual(microkernelCapabilities, variantRequires))
    return makeMicrokernelError(
        kernel, llvm::Twine(descriptor.getRVVMicrokernelOpName()) +
                    " required_capabilities must match selected variant "
                    "requires metadata");

  if (llvm::Error error =
          validateSelectedVectorShapeMetadata(
              kernel, microkernel,
              descriptor.getRVVMicrokernelOpName(), selectedConfig,
              kBoundarySelectedVectorShapeAttrName,
              kBoundarySelectedVectorSEWAttrName,
              kBoundarySelectedVectorLMULAttrName,
              kBoundarySelectedTailPolicyAttrName,
              kBoundarySelectedMaskPolicyAttrName,
              kBoundarySelectedVectorTypeAttrName,
              kBoundarySelectedVectorSuffixAttrName,
              kBoundarySelectedSetVLSuffixAttrName))
    return error;

  std::string microkernelMarch;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, microkernel,
                                kRequiredMarchAttrName,
                                descriptor.getRVVMicrokernelOpName(),
                                microkernelMarch))
    return error;
  if (microkernelMarch != selectedMarch)
    return makeMicrokernelError(
        kernel, llvm::Twine(descriptor.getRVVMicrokernelOpName()) +
                    " required_march must match selected RVV march metadata");

  if (auto mabi = microkernel->getAttrOfType<mlir::StringAttr>(
          kSelectedMABIAttrName)) {
    llvm::StringRef value = mabi.getValue().trim();
    if (llvm::Error error =
            validateBoundedText(kernel, kSelectedMABIAttrName, value))
      return error;
    if (!selectedMABI || *selectedMABI != value)
      return makeMicrokernelError(
          kernel, llvm::Twine(descriptor.getRVVMicrokernelOpName()) +
                      " selected_mabi must match preserved selected_mabi "
                      "capability metadata");
  }

  llvm::SmallVector<support::RuntimeABIParameter, 4> callableABIParameters =
      descriptor.getCallableRuntimeABIParameters();
  RVVBinaryMicrokernelBodyValidationRequest request;
  request.kernel = kernel;
  request.microkernel = microkernel;
  request.descriptor = descriptor;
  request.selectedPolicy = expectedPolicy;
  request.activeRouteID = activeRouteID;
  request.callableABIParameters = callableABIParameters;
  request.expectedDescriptorElementCount = getSelectedVariantElementCount(path);
  llvm::Expected<RVVBinaryMicrokernelBodyValidationResult> validation =
      validateRVVBinaryMicrokernelBody(request);
  if (!validation)
    return validation.takeError();

  elementCount = validation->elementCount;
  controlPlaneSEW = validation->controlPlaneSEW;
  controlPlaneLMUL = std::move(validation->controlPlaneLMUL);
  intrinsicConfig = std::move(validation->intrinsicConfig);
  dataflowPlan = std::move(validation->dataflowPlan);
  return llvm::Error::success();
}

llvm::Error findAndValidateI64Microkernel(
    KernelOp kernel, const SelectedPath &path,
    const llvm::StringSet<> &selectedRVVPathKeys, llvm::StringRef selectedMarch,
    const std::optional<std::string> &selectedMABI,
    PolicyAttr expectedPolicy, mlir::Operation *&matchedMicrokernel,
    const RVVBinaryIntrinsicDescriptor &descriptor,
    const RVVI32VectorShapeConfig &selectedConfig,
    llvm::StringRef activeRouteID,
    std::int64_t &elementCount, std::int64_t &controlPlaneSEW,
    std::string &controlPlaneLMUL, RVVIntrinsicConfig &intrinsicConfig,
    RVVI32VAddDataflowEmissionPlan &dataflowPlan) {
  unsigned matches = 0;
  for (mlir::Operation &op : kernel.getBody().front()) {
    const RVVBinaryFamilyDescriptor *microkernelFamily =
        getI64MicrokernelFamilyForOp(&op);
    if (!microkernelFamily)
      continue;

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    auto role =
        op.getAttrOfType<mlir::StringAttr>(execDiagnostic::kRoleAttrName);
    if (!selectedVariant || !role)
      continue;

    std::string key = makePathKey(selectedVariant.getValue(), role.getValue());
    if (!selectedRVVPathKeys.count(key))
      return makeMicrokernelError(
          kernel, llvm::Twine("stale ") +
                      microkernelFamily->microkernelOpName + " for @" +
                      selectedVariant.getValue() + " as " + role.getValue() +
                      " is not selected by the current RVV microkernel "
                      "surface");

    if (selectedVariant.getValue() == getPathVariantSymbol(path) &&
        role.getValue() == path.role) {
      if (microkernelFamily->familyID != descriptor.family.familyID)
        return makeMicrokernelError(
            kernel, llvm::Twine("selected RVV path @") +
                        getPathVariantSymbol(path) + " as " + path.role +
                        " requires " + descriptor.getRVVMicrokernelOpName() +
                        " but found " + microkernelFamily->microkernelOpName);
      ++matches;
      matchedMicrokernel = &op;
    }
  }

  if (matches == 0)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV path @") + getPathVariantSymbol(path) +
                    " as " + path.role +
                    " requires exactly one matching RVV i64 microkernel");
  if (matches > 1)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV path @") + getPathVariantSymbol(path) +
                    " as " + path.role +
                    " has duplicate RVV i64 microkernel metadata");

  return validateI64MicrokernelForPath(
      kernel, path, selectedMarch, selectedMABI, expectedPolicy,
      matchedMicrokernel, descriptor, selectedConfig, activeRouteID,
      elementCount, controlPlaneSEW, controlPlaneLMUL, intrinsicConfig,
      dataflowPlan);
}

llvm::Error buildRVVBinaryCallableABIPlanFromIR(
    KernelOp kernel, const RVVBinaryIntrinsicDescriptor &descriptor,
    llvm::SmallVectorImpl<support::RuntimeABIParameter> &parameters,
    llvm::SmallVectorImpl<MemWindowOp> &bufferWindows,
    RuntimeParamOp &runtimeElementCountParam) {
  llvm::Expected<support::FiniteBinaryCallableABIPlan> callablePlan =
      support::buildFiniteBinaryCallableABIPlan(
          kernel,
          target::rvv::getRVVBinaryRuntimeABIContract(descriptor.family));
  if (!callablePlan)
    return callablePlan.takeError();

  parameters.clear();
  parameters.append(callablePlan->parameters.begin(),
                    callablePlan->parameters.end());
  bufferWindows.clear();
  bufferWindows.append(callablePlan->bufferWindows.begin(),
                       callablePlan->bufferWindows.end());
  runtimeElementCountParam = callablePlan->runtimeElementCountParam;
  return llvm::Error::success();
}

llvm::Error resolveRVVBinaryRuntimeABIParametersForPath(
    KernelOp kernel, const SelectedPath &path,
    const RVVBinaryIntrinsicDescriptor &descriptor,
    llvm::SmallVectorImpl<support::RuntimeABIParameter> &parameters,
    llvm::SmallVectorImpl<MemWindowOp> &bufferWindows,
    RuntimeParamOp &runtimeElementCountParam) {
  if (llvm::Error error = buildRVVBinaryCallableABIPlanFromIR(
          kernel, descriptor, parameters, bufferWindows,
          runtimeElementCountParam))
    return error;

  llvm::SmallVector<DiagnosticOp, 2> matches;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto diagnostic = llvm::dyn_cast<DiagnosticOp>(op);
    if (!diagnostic || !isEmissionPlanDiagnostic(diagnostic))
      continue;

    auto target = diagnostic->getAttrOfType<mlir::FlatSymbolRefAttr>(
        execDiagnostic::kTargetAttrName);
    auto role =
        diagnostic->getAttrOfType<mlir::StringAttr>(
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
        kernel, llvm::Twine("selected RVV path @") +
                    getPathVariantSymbol(path) + " as " + path.role +
                    " has duplicate emission-plan diagnostics");

  DiagnosticOp diagnostic = matches.front();
  std::string origin;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kOriginAttrName,
                                "emission-plan diagnostic", origin))
    return error;
  if (origin != kRVVPluginName)
    return makeMicrokernelError(
        kernel, llvm::Twine("emission-plan origin '") + origin +
                    "' does not match RVV microkernel origin 'rvv-plugin'");

  std::string status;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, diagnostic.getOperation(),
                                execDiagnostic::kStatusAttrName,
                                "emission-plan diagnostic", status))
    return error;
  if (status != execDiagnostic::kEmissionPlanSupportedStatusValue)
    return makeMicrokernelError(
        kernel, "RVV microkernel export requires a supported emission-plan "
                "diagnostic when runtime ABI metadata is present");

  auto validateDiagnosticText =
      [&](llvm::StringRef attrName, llvm::StringRef expected,
          llvm::StringRef context) -> llvm::Error {
    std::string actual;
    if (llvm::Error error = requireSafeStringAttr(
            kernel, diagnostic.getOperation(), attrName, context, actual))
      return error;
    if (actual != expected)
      return makeMicrokernelError(
          kernel, llvm::Twine(context) + " '" + actual +
                      "' does not match RVV binary microkernel descriptor '" +
                      expected + "'");
    return llvm::Error::success();
  };

  if (llvm::Error error = validateDiagnosticText(
          execDiagnostic::kLoweringPipelineAttrName, descriptor.getRVVRouteID(),
          "supported emission-plan route"))
    return error;
  if (llvm::Error error = validateDiagnosticText(
          execDiagnostic::kEmissionKindAttrName,
          descriptor.family.emissionKind, "supported emission-plan route"))
    return error;
  if (llvm::Error error = validateDiagnosticText(
          execDiagnostic::kArtifactKindAttrName, kMicrokernelArtifactKind,
          "supported emission-plan route"))
    return error;
  if (llvm::Error error = validateDiagnosticText(
          execDiagnostic::kRuntimeABIAttrName, descriptor.getRVVRuntimeABI(),
          "supported emission-plan route"))
    return error;
  if (llvm::Error error = validateDiagnosticText(
          execDiagnostic::kRuntimeABIKindAttrName,
          descriptor.getRVVRuntimeABIKind(), "supported emission-plan route"))
    return error;
  if (llvm::Error error = validateDiagnosticText(
          execDiagnostic::kRuntimeABINameAttrName,
          descriptor.getRVVRuntimeABIName(), "supported emission-plan route"))
    return error;
  if (llvm::Error error = validateDiagnosticText(
          execDiagnostic::kRuntimeGlueRoleAttrName,
          descriptor.getRVVRuntimeGlueRole(), "supported emission-plan route"))
    return error;

  llvm::SmallVector<support::RuntimeABIParameter, 5> planParameters;
  if (llvm::Error error =
          collectRuntimeABIParameters(kernel, diagnostic, planParameters))
    return error;
  if (llvm::Error error = validateRVVBinaryCallableABIParameterMirror(
          kernel, planParameters, parameters,
          "supported RVV microkernel emission-plan", descriptor))
    return error;

  return llvm::Error::success();
}

llvm::Error validateRVVBinaryCandidateRuntimeABIMirrorsIR(
    const TargetArtifactCandidate &candidate,
    const RVVBinaryIntrinsicDescriptor &descriptor) {
  if (!candidate.kernel)
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 4> irBackedParameters;
  llvm::SmallVector<MemWindowOp, 3> ignoredWindows;
  RuntimeParamOp ignoredRuntimeElementCount;
  if (llvm::Error error = buildRVVBinaryCallableABIPlanFromIR(
          candidate.kernel, descriptor, irBackedParameters, ignoredWindows,
          ignoredRuntimeElementCount))
    return error;

  return validateRVVBinaryCallableABIParameterMirror(
      candidate.kernel, candidate.runtimeABIParameters, irBackedParameters,
      "selected RVV target artifact candidate", descriptor);
}

llvm::Expected<RVVBinarySelectedConfigContract>
buildRVVMicrokernelSelectedConfigContract(KernelOp kernel,
                                          const RVVMicrokernelRecord &record) {
  if (!record.selectedShape)
    return makeMicrokernelError(
        kernel,
        "selected RVV microkernel record requires selected vector-shape "
        "metadata before runtime AVL/VL authority validation");

  const RVVBinaryFamilyDescriptor *registeredFamily =
      lookupRVVBinaryFamilyRegistrationByID(
          record.descriptor.getArithmeticFamilyID());
  if (!registeredFamily)
    return makeMicrokernelError(
        kernel,
        llvm::Twine("selected RVV microkernel record references unknown "
                    "finite binary family '") +
            record.descriptor.getArithmeticFamilyID() + "'");

  llvm::Expected<const support::RuntimeABIParameter *> runtimeElementCount =
      support::findUniqueRuntimeABIParameterByRole(
          record.runtimeABIParameters,
          support::RuntimeABIParameterRole::RuntimeElementCount,
          "RVV direct microkernel selected config contract");
  if (!runtimeElementCount) {
    std::string message = llvm::toString(runtimeElementCount.takeError());
    return makeMicrokernelError(kernel, message);
  }

  RuntimeParamOp runtimeElementCountParam = record.runtimeElementCountParam;
  llvm::StringRef runtimeParamCName =
      getAttrValue(runtimeElementCountParam.getOperation(),
                   support::kRuntimeParamCNameAttrName);
  if (runtimeParamCName != (*runtimeElementCount)->cName)
    return makeMicrokernelError(
        kernel,
        llvm::Twine("IR-backed runtime-element-count tcrv.exec.runtime_param "
                    "c_name '") +
            runtimeParamCName +
            "' does not match the selected callable ABI parameter c_name '" +
            (*runtimeElementCount)->cName + "'");

  return buildRVVBinarySelectedConfigContract(
      *registeredFamily, *record.selectedShape, record.variantSymbol,
      record.role, record.elementCount, (*runtimeElementCount)->cName);
}

llvm::Expected<RVVMicrokernelRecord>
buildMicrokernelRecord(KernelOp kernel, const SelectedPath &path,
                       const TargetCapabilitySet &capabilities,
                       const llvm::StringSet<> &selectedRVVPathKeys,
                       llvm::StringRef activeRouteID) {
  if (path.role == kDispatchFallbackRole)
    return makeMicrokernelError(
        kernel, "RVV microkernel export does not accept RVV dispatch fallback "
                "paths in this first slice");

  std::string origin;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, getPathVariantOperation(path),
                                execDiagnostic::kOriginAttrName,
                                "selected RVV variant", origin))
    return std::move(error);
  if (origin != kRVVPluginName)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV microkernel path @") +
                    getPathVariantSymbol(path) +
                    " must be owned by origin 'rvv-plugin'");

  if (llvm::Error error =
          validateBoundedText(kernel, "kernel symbol", kernel.getSymName()))
    return std::move(error);
  if (llvm::Error error = validateBoundedText(
          kernel, "variant symbol", getPathVariantSymbol(path)))
    return std::move(error);

  llvm::SmallVector<std::string, 4> requiredCapabilities;
  if (llvm::Error error = validateRequiredCapabilities(
          kernel, getPathVariant(path), capabilities, requiredCapabilities))
    return std::move(error);
  llvm::Expected<const RVVI32VectorShapeConfig *> selectedConfig =
      getSelectedRVVConfig(kernel, getPathVariant(path), capabilities);
  if (!selectedConfig)
    return selectedConfig.takeError();

  llvm::Expected<std::string> targetTriple =
      getRequiredTargetTriple(kernel, capabilities);
  if (!targetTriple)
    return targetTriple.takeError();

  mlir::Attribute rawPolicy =
      getPathVariant(path)->getAttr(kRVVPolicyAttrName);
  auto policy =
      rawPolicy ? llvm::dyn_cast<tianchenrv::tcrv::rvv::PolicyAttr>(rawPolicy)
                : tianchenrv::tcrv::rvv::PolicyAttr();
  if (!policy)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV variant @") +
                    getPathVariantSymbol(path) +
                    " requires typed 'tcrv_rvv.policy' metadata before "
                    "microkernel export");
  llvm::Expected<std::string> selectedMarch =
      getRequiredSelectedMarch(kernel, getPathVariant(path), capabilities);
  if (!selectedMarch)
    return selectedMarch.takeError();

  std::optional<std::string> selectedMABI;
  if (llvm::Error error =
          getOptionalSelectedMABI(kernel, capabilities, selectedMABI))
    return std::move(error);

  LoweringBoundaryOp boundary;
  if (llvm::Error error =
          findAndValidateBoundary(kernel, path, selectedRVVPathKeys,
                                  **selectedConfig, boundary))
    return std::move(error);

  llvm::Expected<const RVVBinaryFamilyDescriptor *> rvvBinaryFamily =
      resolveSelectedI64FamilyForPath(kernel, path);
  if (!rvvBinaryFamily)
    return rvvBinaryFamily.takeError();

  if (*rvvBinaryFamily) {
    RVVBinaryIntrinsicDescriptor descriptor =
        getRVVBinaryIntrinsicDescriptor(**rvvBinaryFamily, **selectedConfig);
    if (descriptor.getDTypeID() != (*selectedConfig)->dtypeID)
      return makeMicrokernelError(
          kernel, llvm::Twine("selected RVV variant @") +
                      getPathVariantSymbol(path) +
                      " descriptor dtype '" + descriptor.getDTypeID() +
                      "' does not match selected vector-shape dtype '" +
                      (*selectedConfig)->dtypeID + "'");

    mlir::Operation *i64Microkernel = nullptr;
    std::int64_t elementCount = 0;
    std::int64_t controlPlaneSEW = 0;
    std::string controlPlaneLMUL;
    RVVIntrinsicConfig intrinsicConfig;
    RVVI32VAddDataflowEmissionPlan dataflowPlan;
    if (llvm::Error error = findAndValidateI64Microkernel(
            kernel, path, selectedRVVPathKeys, *selectedMarch, selectedMABI,
            policy, i64Microkernel, descriptor, **selectedConfig,
            activeRouteID, elementCount, controlPlaneSEW, controlPlaneLMUL,
            intrinsicConfig, dataflowPlan))
      return std::move(error);

    llvm::SmallVector<support::RuntimeABIParameter, 4> runtimeABIParameters;
    llvm::SmallVector<MemWindowOp, 3> bufferWindows;
    RuntimeParamOp runtimeElementCountParam;
    if (llvm::Error error = resolveRVVBinaryRuntimeABIParametersForPath(
            kernel, path, descriptor, runtimeABIParameters, bufferWindows,
            runtimeElementCountParam))
      return std::move(error);

    RVVMicrokernelRecord record;
    record.descriptor = descriptor;
    record.activeRouteID = getEffectiveRouteID(activeRouteID, descriptor);
    record.kernelSymbol = kernel.getSymName().str();
    record.variantSymbol = getPathVariantSymbol(path).str();
    record.role = path.role;
    record.targetTriple = std::move(*targetTriple);
    record.selectedMarch = std::move(*selectedMarch);
    record.selectedMABI = std::move(selectedMABI);
    record.requiredCapabilities = std::move(requiredCapabilities);
    record.runtimeABIParameters = std::move(runtimeABIParameters);
    record.bufferWindows = std::move(bufferWindows);
    record.runtimeElementCountParam = runtimeElementCountParam;
    record.dataflowPlan = std::move(dataflowPlan);
    record.intrinsicConfig = std::move(intrinsicConfig);
    record.selectedShape = *selectedConfig;
    record.elementCount = elementCount;
    record.controlPlaneSEW = controlPlaneSEW;
    record.controlPlaneLMUL = std::move(controlPlaneLMUL);
    llvm::Expected<RVVBinarySelectedConfigContract> selectedConfigContract =
        buildRVVMicrokernelSelectedConfigContract(kernel, record);
    if (!selectedConfigContract)
      return selectedConfigContract.takeError();
    record.selectedConfigContract = std::move(*selectedConfigContract);
    return record;
  }

  mlir::Operation *microkernel = nullptr;
  const RVVI32MicrokernelFamilySpec *microkernelFamily = nullptr;
  std::int64_t elementCount = 0;
  std::int64_t controlPlaneSEW = 0;
  std::string controlPlaneLMUL;
  RVVIntrinsicConfig intrinsicConfig;
  RVVI32VAddDataflowEmissionPlan dataflowPlan;
  if (llvm::Error error = findAndValidateMicrokernel(
          kernel, path, selectedRVVPathKeys, *selectedMarch, selectedMABI,
          policy, microkernel, microkernelFamily, **selectedConfig,
          activeRouteID, elementCount, controlPlaneSEW, controlPlaneLMUL,
          intrinsicConfig, dataflowPlan))
    return std::move(error);

  RVVMicrokernelRecord record;
  record.family = microkernelFamily;
  record.descriptor =
      getI32BinaryIntrinsicDescriptorForMicrokernel(*microkernelFamily,
                                                   **selectedConfig);
  llvm::SmallVector<support::RuntimeABIParameter, 4> runtimeABIParameters;
  llvm::SmallVector<MemWindowOp, 3> bufferWindows;
  RuntimeParamOp runtimeElementCountParam;
  if (llvm::Error error = resolveRVVBinaryRuntimeABIParametersForPath(
          kernel, path, record.descriptor, runtimeABIParameters, bufferWindows,
          runtimeElementCountParam))
    return std::move(error);
  record.activeRouteID =
      getEffectiveRouteID(activeRouteID, *microkernelFamily);
  record.kernelSymbol = kernel.getSymName().str();
  record.variantSymbol = getPathVariantSymbol(path).str();
  record.role = path.role;
  record.targetTriple = std::move(*targetTriple);
  record.selectedMarch = std::move(*selectedMarch);
  record.selectedMABI = std::move(selectedMABI);
  record.requiredCapabilities = std::move(requiredCapabilities);
  record.runtimeABIParameters = std::move(runtimeABIParameters);
  record.bufferWindows = std::move(bufferWindows);
  record.runtimeElementCountParam = runtimeElementCountParam;
  record.dataflowPlan = std::move(dataflowPlan);
  record.intrinsicConfig = std::move(intrinsicConfig);
  record.selectedShape = *selectedConfig;
  record.elementCount = elementCount;
  record.controlPlaneSEW = controlPlaneSEW;
  record.controlPlaneLMUL = std::move(controlPlaneLMUL);
  llvm::Expected<RVVBinarySelectedConfigContract> selectedConfigContract =
      buildRVVMicrokernelSelectedConfigContract(kernel, record);
  if (!selectedConfigContract)
    return selectedConfigContract.takeError();
  record.selectedConfigContract = std::move(*selectedConfigContract);
  return record;
}

bool isRVVPluginSelectedPath(const SelectedPath &path) {
  auto origin =
      getPathVariantOperation(path)->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kOriginAttrName);
  return origin && origin.getValue() == kRVVPluginName;
}

bool hasRVVLikeOrigin(const SelectedPath &path) {
  auto origin =
      getPathVariantOperation(path)->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kOriginAttrName);
  if (!origin)
    return false;
  std::string lower = origin.getValue().lower();
  return llvm::StringRef(lower).contains("rvv");
}

llvm::Expected<RVVMicrokernelRecord>
buildModuleRecord(mlir::ModuleOp module, llvm::StringRef activeRouteID) {
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

  llvm::SmallVector<RVVMicrokernelRecord, 2> records;
  for (KernelOp kernel : kernels) {
    llvm::StringMap<VariantOp> directVariants;
    llvm::StringMap<mlir::Operation *> directSymbols;
    collectDirectKernelSymbols(kernel, directVariants, directSymbols);

    llvm::SmallVector<SelectedPath, 4> selectedPaths;
    if (llvm::Error error =
            collectSelectedPaths(kernel, directVariants, directSymbols,
                                 selectedPaths))
      return std::move(error);

    llvm::SmallVector<SelectedPath, 2> selectedRVVPaths;
    for (const SelectedPath &path : selectedPaths) {
      if (isRVVPluginSelectedPath(path)) {
        selectedRVVPaths.push_back(path);
        continue;
      }
      if (hasRVVLikeOrigin(path))
        return makeMicrokernelError(
            kernel, llvm::Twine("selected RVV-like path @") +
                        getPathVariantSymbol(path) +
                        " uses unknown origin; RVV microkernel export only "
                        "accepts registered origin 'rvv-plugin'");
    }

    if (selectedRVVPaths.empty())
      return makeMicrokernelError(
          kernel, "requires one selected rvv-plugin path; scalar, offload, "
                  "or fallback-only selected paths are not RVV microkernel "
                  "inputs");
    if (selectedRVVPaths.size() != 1)
      return makeMicrokernelError(
          kernel, "requires exactly one selected rvv-plugin path for this "
                  "bounded RVV microkernel export");

    llvm::StringSet<> selectedRVVPathKeys;
    for (const SelectedPath &path : selectedRVVPaths)
      selectedRVVPathKeys.insert(
          makePathKey(getPathVariantSymbol(path), path.role));

    llvm::Expected<TargetCapabilitySet> capabilities =
        TargetCapabilitySet::buildFromKernelChecked(kernel);
    if (!capabilities)
      return capabilities.takeError();
    llvm::Expected<RVVMicrokernelRecord> record = buildMicrokernelRecord(
        kernel, selectedRVVPaths.front(), *capabilities,
        selectedRVVPathKeys, activeRouteID);
    if (!record)
      return record.takeError();
    records.push_back(std::move(*record));
  }

  if (records.size() != 1)
    return makeModuleMicrokernelError(
        "requires exactly one valid RVV binary microkernel record in "
        "the module");
  return std::move(records.front());
}

llvm::Expected<RVVMicrokernelRecord> buildModuleRecord(mlir::ModuleOp module) {
  return buildModuleRecord(module, llvm::StringRef());
}

llvm::Expected<RVVMicrokernelRecord> buildModuleRecordForRVVBinaryFamily(
    mlir::ModuleOp module, const RVVBinaryFamilyDescriptor &expectedFamily,
    llvm::StringRef routeID) {
  llvm::Expected<RVVMicrokernelRecord> record =
      buildModuleRecord(module, routeID);
  if (!record)
    return record.takeError();

  if (!isSameRVVBinaryFamily(record->descriptor.family, expectedFamily)) {
    return makeModuleMicrokernelError(
        llvm::Twine("route '") + routeID + "' requires " +
        expectedFamily.microkernelOpName + " but the selected RVV record is " +
        record->descriptor.getRVVMicrokernelOpName());
  }

  return std::move(*record);
}

llvm::Error requireRVVSourceAuthorityField(llvm::StringRef fieldName,
                                           llvm::StringRef actual,
                                           llvm::StringRef expected,
                                           llvm::StringRef routeID,
                                           llvm::StringRef selectedVariant) {
  if (actual == expected)
    return llvm::Error::success();
  return makeModuleMicrokernelError(
      llvm::Twine("selected RVV component authority for route '") + routeID +
      "' expected " + fieldName + " '" + expected + "' for @" +
      selectedVariant + ", got '" + actual + "'");
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

std::string makeMicrokernelFunctionName(const RVVMicrokernelRecord &record) {
  std::string name;
  llvm::raw_string_ostream stream(name);
  stream << "tcrv_rvv_" << record.descriptor.family.functionStem
         << "_microkernel_"
         << sanitizeCIdentifierComponent(record.kernelSymbol) << "_"
         << sanitizeCIdentifierComponent(record.variantSymbol);
  stream.flush();
  return name;
}

std::string makeMicrokernelHeaderIncludeGuard(
    const RVVMicrokernelRecord &record) {
  std::string guard = "TIANCHENRV_RVV_";
  guard += record.descriptor.family.headerGuardStem;
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

std::string
getDataflowStepOpName(const RVVI32VAddDataflowStep &step,
                      const RVVBinaryIntrinsicDescriptor &descriptor) {
  if (!step.sourceOpName.empty())
    return step.sourceOpName;
  switch (step.kind) {
  case RVVI32VAddDataflowStepKind::Load:
    return (llvm::Twine("tcrv_rvv.") + descriptor.getDTypeID() + "_load")
        .str();
  case RVVI32VAddDataflowStepKind::Add:
    return descriptor.getRVVOperationName().str();
  case RVVI32VAddDataflowStepKind::Sub:
    return descriptor.getRVVOperationName().str();
  case RVVI32VAddDataflowStepKind::Mul:
    return descriptor.getRVVOperationName().str();
  case RVVI32VAddDataflowStepKind::Store:
    return (llvm::Twine("tcrv_rvv.") + descriptor.getDTypeID() + "_store")
        .str();
  }
  return "unknown";
}

llvm::StringRef
getDataflowValueCName(RVVI32VAddDataflowValue value,
                      const RVVBinaryIntrinsicDescriptor &descriptor) {
  switch (value) {
  case RVVI32VAddDataflowValue::LHSVector:
    return "lhs_vec";
  case RVVI32VAddDataflowValue::RHSVector:
    return "rhs_vec";
  case RVVI32VAddDataflowValue::ResultVector:
    return descriptor.family.resultCName;
  case RVVI32VAddDataflowValue::None:
    return "";
  }
  return "";
}

std::string getEmitCCallOpaqueCalleeForStep(
    const RVVI32VAddDataflowStep &step,
    const RVVIntrinsicConfig &intrinsicConfig) {
  switch (step.kind) {
  case RVVI32VAddDataflowStepKind::Load:
    return intrinsicConfig.loadIntrinsicName;
  case RVVI32VAddDataflowStepKind::Add:
  case RVVI32VAddDataflowStepKind::Sub:
  case RVVI32VAddDataflowStepKind::Mul:
    return intrinsicConfig.arithmeticIntrinsicName;
  case RVVI32VAddDataflowStepKind::Store:
    return intrinsicConfig.storeIntrinsicName;
  }
  return "";
}

llvm::Expected<BinarySelfCheckArithmeticKind>
getSelfCheckArithmeticFromDataflowPlan(
    const RVVI32VAddDataflowEmissionPlan &dataflowPlan,
    llvm::StringRef context) {
  std::optional<BinarySelfCheckArithmeticKind> arithmetic;
  for (const RVVI32VAddDataflowStep &step : dataflowPlan.steps) {
    std::optional<BinarySelfCheckArithmeticKind> candidate;
    switch (step.kind) {
    case RVVI32VAddDataflowStepKind::Load:
    case RVVI32VAddDataflowStepKind::Store:
      continue;
    case RVVI32VAddDataflowStepKind::Add:
      candidate = BinarySelfCheckArithmeticKind::Add;
      break;
    case RVVI32VAddDataflowStepKind::Sub:
      candidate = BinarySelfCheckArithmeticKind::Sub;
      break;
    case RVVI32VAddDataflowStepKind::Mul:
      candidate = BinarySelfCheckArithmeticKind::Mul;
      break;
    }

    if (arithmetic)
      return makeModuleMicrokernelError(
          llvm::Twine(context) +
          " requires exactly one typed compute dataflow step before "
          "self-check expectation emission");
    if (step.sourceOpRole != "compute" || step.sourceOpInterface.empty())
      return makeModuleMicrokernelError(
          llvm::Twine(context) +
          " requires the typed compute dataflow step to carry generated "
          "EmitC lowerable interface provenance before self-check expectation "
          "emission");
    arithmetic = *candidate;
  }

  if (!arithmetic)
    return makeModuleMicrokernelError(
        llvm::Twine(context) +
        " requires one typed compute dataflow step before self-check "
        "expectation emission");
  return *arithmetic;
}

llvm::Expected<BinarySelfCheckExpectation>
buildMicrokernelSelfCheckExpectation(const RVVMicrokernelRecord &record) {
  llvm::Expected<BinarySelfCheckArithmeticKind> arithmetic =
      getSelfCheckArithmeticFromDataflowPlan(
          record.dataflowPlan,
          "RVV direct microkernel harness");
  if (!arithmetic)
    return arithmetic.takeError();

  return tianchenrv::target::buildBinarySelfCheckExpectationFromRuntimeABI(
      record.runtimeABIParameters, *arithmetic,
      "verified RVV dataflow body + generated EmitC route + IR-backed "
      "callable ABI",
      "RVV direct microkernel harness");
}

const support::RuntimeABIParameter *lookupParameterByRole(
    llvm::ArrayRef<support::RuntimeABIParameter> parameters,
    support::RuntimeABIParameterRole role) {
  for (const support::RuntimeABIParameter &parameter : parameters)
    if (parameter.role == role)
      return &parameter;
  return nullptr;
}

class RVVBinaryEmitCLowerable final : public TCRVEmitCLowerableInterface {
public:
  RVVBinaryEmitCLowerable(
      const RVVBinaryIntrinsicDescriptor &descriptor,
      const RVVIntrinsicConfig &intrinsicConfig,
      const RVVI32VAddDataflowEmissionPlan &dataflowPlan,
      llvm::ArrayRef<support::RuntimeABIParameter> runtimeABIParameters)
      : descriptor(descriptor), intrinsicConfig(intrinsicConfig),
        dataflowPlan(dataflowPlan) {
    this->runtimeABIParameters.append(runtimeABIParameters.begin(),
                                      runtimeABIParameters.end());
  }

  llvm::Expected<TCRVEmitCLowerableRoute>
  buildEmitCLowerableRoute() const override {
    if (dataflowPlan.steps.size() != 4)
      return makeModuleMicrokernelError(
          "RVV family-op to EmitC route requires exactly four verified "
          "load/load/arithmetic/store dataflow steps");

    const support::RuntimeABIParameter *runtimeN = lookupParameterByRole(
        runtimeABIParameters,
        support::RuntimeABIParameterRole::RuntimeElementCount);
    if (!runtimeN)
      return makeModuleMicrokernelError(
          "RVV family-op to EmitC route requires runtime-element-count ABI "
          "mapping from the IR-backed callable plan");

    TCRVEmitCLowerableRoute route(
        descriptor.getRVVRouteID(),
        "extension-family-ops-to-emitc-call-opaque");
    route.addHeader("stddef.h");
    route.addHeader("stdint.h");
    route.addHeader("riscv_vector.h");
    route.addTypeMapping("!tcrv_rvv.vl", "size_t");
    route.addTypeMapping(
        (llvm::Twine("!tcrv_rvv.") + descriptor.getShapeID()).str(),
        intrinsicConfig.vectorType);
    for (const support::RuntimeABIParameter &parameter : runtimeABIParameters)
      route.addABIValueMapping(parameter, parameter.cName);

    TCRVEmitCCallOpaqueStep setvlStep;
    setvlStep.sourceOp.opName = "tcrv_rvv.setvl";
    setvlStep.sourceOp.role = "runtime-avl-to-vl";
    setvlStep.callee = intrinsicConfig.setvlIntrinsicName;
    setvlStep.operands.push_back(
        {(llvm::Twine(runtimeN->cName) + " - offset").str(), runtimeN->cType});
    setvlStep.result = TCRVEmitCCallOpaqueResult{"vl", "size_t"};
    route.addCallOpaqueStep(std::move(setvlStep));

    for (const RVVI32VAddDataflowStep &step : dataflowPlan.steps) {
      TCRVEmitCCallOpaqueStep emitcStep;
      emitcStep.sourceOp.opName = getDataflowStepOpName(step, descriptor);
      emitcStep.sourceOp.role = getRouteSourceRole(step).str();
      emitcStep.sourceOp.opInterface = step.sourceOpInterface;
      emitcStep.callee =
          getEmitCCallOpaqueCalleeForStep(step, intrinsicConfig);
      if (emitcStep.sourceOp.opName.empty() || emitcStep.callee.empty())
        return makeModuleMicrokernelError(
            "RVV family-op to EmitC route requires every dataflow step to map "
            "to a bounded emitc.call_opaque callee");

      switch (step.kind) {
      case RVVI32VAddDataflowStepKind::Load: {
        const support::RuntimeABIParameter *parameter =
            lookupParameterByRole(runtimeABIParameters, step.bufferRole);
        if (!parameter)
          return makeModuleMicrokernelError(
              "RVV EmitC route load step references a missing buffer ABI "
              "mapping");
        emitcStep.operands.push_back(
            {(llvm::Twine("&") + parameter->cName + "[offset]").str(),
             parameter->cType});
        emitcStep.operands.push_back({"vl", "size_t"});
        emitcStep.result = TCRVEmitCCallOpaqueResult{
            getDataflowValueCName(step.result, descriptor).str(),
            intrinsicConfig.vectorType};
        break;
      }
      case RVVI32VAddDataflowStepKind::Add:
      case RVVI32VAddDataflowStepKind::Sub:
      case RVVI32VAddDataflowStepKind::Mul:
        emitcStep.operands.push_back(
            {getDataflowValueCName(step.lhs, descriptor).str(),
             intrinsicConfig.vectorType});
        emitcStep.operands.push_back(
            {getDataflowValueCName(step.rhs, descriptor).str(),
             intrinsicConfig.vectorType});
        emitcStep.operands.push_back({"vl", "size_t"});
        emitcStep.result = TCRVEmitCCallOpaqueResult{
            getDataflowValueCName(step.result, descriptor).str(),
            intrinsicConfig.vectorType};
        break;
      case RVVI32VAddDataflowStepKind::Store: {
        const support::RuntimeABIParameter *parameter =
            lookupParameterByRole(runtimeABIParameters, step.bufferRole);
        if (!parameter)
          return makeModuleMicrokernelError(
              "RVV EmitC route store step references a missing buffer ABI "
              "mapping");
        emitcStep.operands.push_back(
            {(llvm::Twine("&") + parameter->cName + "[offset]").str(),
             parameter->cType});
        emitcStep.operands.push_back(
            {getDataflowValueCName(step.value, descriptor).str(),
             intrinsicConfig.vectorType});
        emitcStep.operands.push_back({"vl", "size_t"});
        break;
      }
      }
      route.addCallOpaqueStep(std::move(emitcStep));
    }

    return route;
  }

private:
  static llvm::StringRef getRouteSourceRole(const RVVI32VAddDataflowStep &step) {
    if (!step.sourceOpRole.empty())
      return step.sourceOpRole;
    switch (step.kind) {
    case RVVI32VAddDataflowStepKind::Load:
      return "buffer-load";
    case RVVI32VAddDataflowStepKind::Add:
    case RVVI32VAddDataflowStepKind::Sub:
    case RVVI32VAddDataflowStepKind::Mul:
      return "compute";
    case RVVI32VAddDataflowStepKind::Store:
      return "buffer-store";
    }
    return "unknown";
  }

  RVVBinaryIntrinsicDescriptor descriptor;
  RVVIntrinsicConfig intrinsicConfig;
  RVVI32VAddDataflowEmissionPlan dataflowPlan;
  llvm::SmallVector<support::RuntimeABIParameter, 4> runtimeABIParameters;
};

llvm::Expected<TCRVLowerToEmitCSourceResult> lowerRVVBinaryToEmitCSource(
    const RVVBinaryIntrinsicDescriptor &descriptor,
    const RVVIntrinsicConfig &intrinsicConfig,
    const RVVI32VAddDataflowEmissionPlan &dataflowPlan,
    llvm::ArrayRef<support::RuntimeABIParameter> runtimeABIParameters,
    llvm::StringRef functionName) {
  RVVBinaryEmitCLowerable lowerable(descriptor, intrinsicConfig, dataflowPlan,
                                    runtimeABIParameters);
  TCRVLowerToEmitCSourceOptions options;
  options.sourceAuthorityOptions.functionName = functionName.str();
  options.sourceAuthorityOptions.loopIndexName = "offset";
  options.sourceAuthorityOptions.requireInterfaceBackedCompute = true;
  return lowerTCRVEmitCLowerableToEmitCSource(lowerable, options);
}

void printDataflowPlanMetadata(
    llvm::raw_ostream &os,
    const RVVI32VAddDataflowEmissionPlan &dataflowPlan,
    const RVVBinaryIntrinsicDescriptor &descriptor) {
  for (auto [index, step] : llvm::enumerate(dataflowPlan.steps)) {
    os << "/* dataflow_emission_step[" << index
       << "]: op=" << getDataflowStepOpName(step, descriptor);
    switch (step.kind) {
    case RVVI32VAddDataflowStepKind::Load:
      os << ", role="
         << support::stringifyRuntimeABIParameterRole(step.bufferRole)
         << ", result=" << getDataflowValueCName(step.result, descriptor);
      break;
    case RVVI32VAddDataflowStepKind::Add:
    case RVVI32VAddDataflowStepKind::Sub:
    case RVVI32VAddDataflowStepKind::Mul:
      os << ", lhs=" << getDataflowValueCName(step.lhs, descriptor)
         << ", rhs=" << getDataflowValueCName(step.rhs, descriptor)
         << ", result=" << getDataflowValueCName(step.result, descriptor);
      if (!step.sourceOpInterface.empty())
        os << ", interface=" << step.sourceOpInterface
           << ", source_role=" << step.sourceOpRole;
      break;
    case RVVI32VAddDataflowStepKind::Store:
      os << ", role="
         << support::stringifyRuntimeABIParameterRole(step.bufferRole)
         << ", value=" << getDataflowValueCName(step.value, descriptor);
      break;
    }
    os << " */\n";
  }
}

void printEmitCRouteMetadata(llvm::raw_ostream &os,
                             const TCRVEmitCLowerableRoute &route,
                             llvm::StringRef functionName) {
  os << "/* emitc_route: tcrv_rvv.family_ops -> emitc.call_opaque -> RVV "
        "intrinsic C/C++ */\n";
  os << "/* emitc_lowerable_interface: TCRVEmitCLowerableInterface */\n";
  os << "/* emitc_common_lower_to_emitc_boundary: "
        "TCRVLowerToEmitCSourceAuthority */\n";
  os << "/* emitc_materialization_boundary: verified MLIR EmitC module with "
        "emitc.include, emitc.func, emitc.if, emitc.call_opaque, and "
        "emitc.call before MLIR Cpp emitter production source output */\n";
  os << "/* emitc_materialization_function: @" << functionName << " */\n";
  os << "/* emitc_c_source_authority: MLIR EmitC module translated by "
        "mlir::emitc::translateToCpp */\n";
  bool hasGeneratedOpInterface = llvm::any_of(
      route.getCallOpaqueSteps(), [](const TCRVEmitCCallOpaqueStep &step) {
        return !step.sourceOp.opInterface.empty();
      });
  if (hasGeneratedOpInterface) {
    for (const TCRVEmitCCallOpaqueStep &step : route.getCallOpaqueSteps()) {
      if (step.sourceOp.opInterface.empty())
        continue;
      os << "/* emitc_lowerable_op_interface: "
         << step.sourceOp.opInterface << " */\n";
      break;
    }
  }
  os << "/* emitc_route_id: " << route.getRouteID()
     << ", route_kind=" << route.getRouteKind() << " */\n";
  os << "/* emitc_route_headers:";
  for (const auto &header : route.getHeaders())
    os << " <" << header.header << ">";
  os << " */\n";
  os << "/* emitc_route_source_ops:";
  llvm::ArrayRef<TCRVEmitCCallOpaqueStep> callSteps =
      route.getCallOpaqueSteps();
  if (!callSteps.empty())
    os << " " << callSteps.front().sourceOp.opName;
  os << " tcrv_rvv.with_vl";
  for (const TCRVEmitCCallOpaqueStep &step : callSteps.drop_front())
    os << " " << step.sourceOp.opName;
  os << " */\n";
  for (auto [index, step] : llvm::enumerate(route.getCallOpaqueSteps())) {
    os << "/* emitc.call_opaque[" << index
       << "]: " << step.callee << " from " << step.sourceOp.opName << " */\n";
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

void printCallableBoundaryMetadata(llvm::raw_ostream &os,
                                   const RVVMicrokernelRecord &record) {
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

void printRecordComment(llvm::raw_ostream &os,
                        const RVVMicrokernelRecord &record,
                        llvm::StringRef functionName,
                        const TCRVEmitCLowerableRoute &emitcRoute) {
  os << "/* microkernel function: " << functionName << " */\n";
  os << "/* selected_kernel: @" << record.kernelSymbol << " */\n";
  os << "/* selected_variant: @" << record.variantSymbol << " */\n";
  os << "/* selected_role: " << record.role << " */\n";
  os << "/* selected_march: " << record.selectedMarch << " */\n";
  if (record.selectedMABI)
    os << "/* selected_mabi: " << *record.selectedMABI << " */\n";
  os << "/* lowering_boundary: tcrv_rvv.lowering_boundary */\n";
  os << "/* executable_microkernel: "
     << record.descriptor.getRVVMicrokernelOpName()
     << " */\n";
  os << "/* arithmetic_family: "
     << record.descriptor.getArithmeticFamilyID() << " */\n";
  os << "/* dtype: " << record.descriptor.getDTypeID() << " */\n";
  os << "/* " << record.selectedConfigContract.formatSummaryCommentBody()
     << " */\n";
  os << "/* "
     << record.selectedConfigContract.formatRuntimeVLBoundaryCommentBody()
     << " */\n";
  os << "/* arithmetic_source: typed op "
     << record.descriptor.getRVVOperationName()
     << " via generated EmitC route and IR-backed callable ABI */\n";
  os << "/* active_route: " << record.activeRouteID << " */\n";
  os << "/* control_plane_body: tcrv_rvv.setvl -> tcrv_rvv.with_vl */\n";
  os << "/* control_plane_runtime_avl: body index argument maps to "
        "target/export-owned runtime "
     << record.selectedConfigContract.getRuntimeElementCountCName()
     << " ABI parameter */\n";
  os << "/* control_plane_vl: !tcrv_rvv.vl value consumed by "
        "tcrv_rvv.with_vl */\n";
  os << "/* dataflow_body: tcrv_rvv." << record.descriptor.getDTypeID()
     << "_load -> tcrv_rvv." << record.descriptor.getDTypeID()
     << "_load -> " << record.descriptor.getRVVOperationName()
     << " -> tcrv_rvv." << record.descriptor.getDTypeID()
     << "_store */\n";
  os << "/* dataflow_emission_source: derived from verified "
        "tcrv_rvv.with_vl body order, SSA chain, and buffer_role "
        "attributes */\n";
  os << "/* dataflow_abi_roles: lhs_load.buffer_role=lhs-input-buffer, "
        "rhs_load.buffer_role=rhs-input-buffer, "
        "store.buffer_role=output-buffer; runtime "
     << record.selectedConfigContract.getRuntimeElementCountCName()
     << " remains the target/export-owned runtime element-count ABI "
        "parameter */\n";
  printDataflowPlanMetadata(os, record.dataflowPlan, record.descriptor);
  printEmitCRouteMetadata(os, emitcRoute, functionName);
  if (record.selectedShape) {
    os << "/* "
       << record.descriptor.formatSelectedVectorShapeConfigCommentBody()
       << " */\n";
    os << "/* "
       << record.descriptor.formatSelectedVectorShapeCapabilitiesCommentBody()
       << " */\n";
  }
  os << "/* control_plane_config: sew=" << record.controlPlaneSEW
     << ", lmul=" << record.controlPlaneLMUL
     << ", policy=#tcrv_rvv.policy<tail = "
     << record.intrinsicConfig.tailPolicy
     << ", mask = " << record.intrinsicConfig.maskPolicy << "> */\n";
  os << "/* intrinsic_config_source: validated tcrv_rvv.setvl and "
        "tcrv_rvv.with_vl SEW/LMUL/policy metadata */\n";
  os << "/* intrinsic_config: vector_type="
     << record.intrinsicConfig.vectorType
     << ", vector_suffix=" << record.intrinsicConfig.vectorSuffix
     << ", setvl_suffix=" << record.intrinsicConfig.setvlSuffix
     << ", tail_policy=" << record.intrinsicConfig.tailPolicy
     << ", mask_policy=" << record.intrinsicConfig.maskPolicy << " */\n";
  os << "/* artifact_kind: runtime-callable-c-source */\n";
  os << "/* element_count: " << record.elementCount << " */\n";
  os << "/* required_capabilities:";
  for (llvm::StringRef capability : record.requiredCapabilities)
    os << " @" << capability;
  os << " */\n";
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
}

void printMicrokernelPrototype(llvm::raw_ostream &os,
                               llvm::StringRef functionName,
                               llvm::ArrayRef<support::RuntimeABIParameter>
                                   parameters) {
  os << "void " << functionName << "(";
  for (auto [index, parameter] : llvm::enumerate(parameters)) {
    if (index != 0)
      os << ", ";
    support::printRuntimeABIParameterCDeclaration(os, parameter);
  }
  os << ")";
}

void printMicrokernelSelfCheckHarness(llvm::raw_ostream &os,
                                      llvm::StringRef functionName,
                                      const BinarySelfCheckExpectation
                                          &expectation,
                                      const RVVIntrinsicConfig &intrinsicConfig,
                                      std::int64_t elementCount) {
  os << "/* Harness capacity comes from descriptor-local element_count; each "
        "call still supplies runtime n through the generated C ABI. */\n";
  os << "/* self_check_expectation_source: " << expectation.provenance
     << "; legacy descriptor mirrors cannot select expected arithmetic or "
        "scalar element type. */\n";
  os << "static int " << functionName
     << "_self_check_one(size_t runtime_n) {\n";
  os << "  enum { kTCRVMicrokernelCapacity = " << elementCount << " };\n";
  os << "  " << expectation.scalarElementCType
     << " lhs[kTCRVMicrokernelCapacity];\n";
  os << "  " << expectation.scalarElementCType
     << " rhs[kTCRVMicrokernelCapacity];\n";
  os << "  " << expectation.scalarElementCType
     << " out[kTCRVMicrokernelCapacity];\n\n";
  os << "  if (runtime_n == 0 || runtime_n > (size_t)kTCRVMicrokernelCapacity) "
        "{\n";
  os << "    fprintf(stderr, \"invalid rvv microkernel runtime_n=%zu\\n\", "
        "runtime_n);\n";
  os << "    return 1;\n";
  os << "  }\n\n";
  os << "  for (size_t index = 0; index < (size_t)kTCRVMicrokernelCapacity; "
        "++index) {\n";
  os << "    lhs[index] = (" << expectation.scalarElementCType
     << ")(index + 1);\n";
  os << "    rhs[index] = (" << expectation.scalarElementCType
     << ")(100 - (int)index);\n";
  os << "    out[index] = (" << expectation.scalarElementCType
     << ")-12345;\n";
  os << "  }\n\n";
  os << "  size_t first_vl = " << intrinsicConfig.setvlIntrinsicName
     << "(runtime_n);\n";
  os << "  if (first_vl == 0 || first_vl > runtime_n) {\n";
  os << "    fprintf(stderr, \"invalid rvv microkernel vl=%zu\\n\", first_vl);\n";
  os << "    return 2;\n";
  os << "  }\n\n";
  os << "  " << functionName << "(lhs, rhs, out, runtime_n);\n\n";
  os << "  for (size_t index = 0; index < runtime_n; ++index) {\n";
  os << "    " << expectation.scalarElementCType << " expected = "
     << expectation.formatExpression("lhs[index]", "rhs[index]")
     << ";\n";
  os << "    if (out[index] != expected) {\n";
  os << "      fprintf(stderr, \"rvv microkernel mismatch at %zu\\n\", index);\n";
  os << "      return 3;\n";
  os << "    }\n";
  os << "  }\n";
  os << "  for (size_t index = runtime_n; "
        "index < (size_t)kTCRVMicrokernelCapacity; ++index) {\n";
  os << "    if (out[index] != (" << expectation.scalarElementCType
     << ")-12345) {\n";
  os << "      fprintf(stderr, \"rvv microkernel wrote past runtime_n at "
        "%zu\\n\", index);\n";
  os << "      return 4;\n";
  os << "    }\n";
  os << "  }\n";
  os << "  return 0;\n";
  os << "}\n\n";

  os << "int main(void) {\n";
  os << "  enum { kTCRVMicrokernelCapacity = " << elementCount << " };\n";
  os << "  enum { kTCRVMicrokernelShortRuntimeN = "
        "kTCRVMicrokernelCapacity >= 7 ? 7 : kTCRVMicrokernelCapacity };\n";
  os << "  int status = " << functionName
     << "_self_check_one((size_t)kTCRVMicrokernelShortRuntimeN);\n";
  os << "  if (status != 0)\n";
  os << "    return status;\n";
  os << "  status = " << functionName
     << "_self_check_one((size_t)kTCRVMicrokernelCapacity);\n";
  os << "  if (status != 0)\n";
  os << "    return 10 + status;\n";
  os << "  printf(\"tcrv_rvv_microkernel_ok runtime_counts=%zu,%zu\\n\", "
        "(size_t)kTCRVMicrokernelShortRuntimeN, "
        "(size_t)kTCRVMicrokernelCapacity);\n";
  os << "  return 0;\n";
  os << "}\n";
}

void printMicrokernelHeader(const RVVMicrokernelRecord &record,
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

llvm::Error printMicrokernelSource(const RVVMicrokernelRecord &record,
                                   llvm::raw_ostream &os,
                                   RVVMicrokernelCExportMode mode) {
  std::string functionName = makeMicrokernelFunctionName(record);
  bool includeHarness = mode == RVVMicrokernelCExportMode::SelfCheckHarness;
  llvm::Expected<TCRVLowerToEmitCSourceResult> loweredSource =
      lowerRVVBinaryToEmitCSource(record.descriptor, record.intrinsicConfig,
                                  record.dataflowPlan,
                                  record.runtimeABIParameters, functionName);
  if (!loweredSource)
    return loweredSource.takeError();
  const TCRVEmitCLowerableRoute &emitcRoute = loweredSource->getRoute();

  os << "/* TianChen-RV RVV runtime-callable microkernel C export. */\n";
  os << "/* Scope: library-style C source for exactly one "
     << record.descriptor.getRVVMicrokernelOpName() << ". */\n";
  os << "/* Route: verified RVV family ops build the common EmitC lowerable "
        "route emitted by the common lower-to-EmitC source-authority "
        "boundary. */\n";
  os << "/* Default artifact shape: runtime-callable C ABI function with no "
        "embedded main or self-check harness. */\n";
  if (includeHarness)
    os << "/* Harness mode: adds a bounded self-check main for explicit ssh rvv "
          "evidence only. */\n";
  os << "/* Correctness claims require the explicit self-check harness and ssh "
        "rvv evidence; this source is not generic TianChen-RV lowering or "
        "performance evidence. */\n\n";

  printRecordComment(os, record, functionName, emitcRoute);
  os << loweredSource->getSource();
  if (includeHarness) {
    os << "#include <stdio.h>\n\n";
    if (record.descriptor.family.dtype != RVVBinaryDTypeKind::I32)
      return makeModuleMicrokernelError(
          "RVV self-check harness export is currently bounded to i32 "
          "microkernel records");
    llvm::Expected<BinarySelfCheckExpectation> expectation =
        buildMicrokernelSelfCheckExpectation(record);
    if (!expectation)
      return expectation.takeError();
    printMicrokernelSelfCheckHarness(os, functionName, *expectation,
                                     record.intrinsicConfig,
                                     record.elementCount);
  }
  return llvm::Error::success();
}

TargetArtifactRouteMetadata
buildRVVMicrokernelSourceRouteMetadata(
    const RVVBinaryFamilyDescriptor &family);

void addRVVMicrokernelConservativeRouteClaims(
    TargetArtifactRouteMetadata &metadata);

TargetArtifactRouteMetadata buildRVVMicrokernelArtifactRouteMetadata(
    const RVVMicrokernelDirectRouteManifestEntry &route);

tianchenrv::target::TargetArtifactExportFn
getRVVMicrokernelExactExportFn(const RVVBinaryFamilyDescriptor &family,
                               RVVMicrokernelDirectRouteKind routeKind);

llvm::Error validateRVVMicrokernelSourceCandidate(
    const TargetArtifactCandidate &candidate);

TargetArtifactExporter buildRVVMicrokernelSourceTargetArtifactExporter(
    const RVVMicrokernelDirectRouteManifestEntry &route,
    bool enableCandidateValidation) {
  return TargetArtifactExporter(
      route.getRouteID(), route.getArtifactKind(), route.getOwner(),
      route.getEmissionKind(),
      getRVVMicrokernelExactExportFn(*route.family, route.routeKind),
      getRVVBinaryCallableRuntimeABIRoleRequirements(*route.family),
      route.isDirectHelperCompatibilityRoute(),
      /*handoffKind=*/{},
      enableCandidateValidation ? validateRVVMicrokernelSourceCandidate
                                : nullptr,
      route.getComponentGroup(), route.getExternalABIName(),
      buildRVVMicrokernelArtifactRouteMetadata(route));
}

llvm::Error validateRVVMicrokernelSourceCandidate(
    const TargetArtifactCandidate &candidate) {
  const RVVMicrokernelDirectRouteManifestEntry *route =
      lookupRVVMicrokernelDirectRoute(candidate.routeID);
  if (!route || route->routeKind != RVVMicrokernelDirectRouteKind::Source ||
      !route->family)
    return makeModuleMicrokernelError(
        llvm::Twine("target artifact route '") + candidate.routeID +
        "' is not a manifest-backed RVV microkernel source route");

  const RVVBinaryFamilyDescriptor &family = *route->family;
  llvm::Expected<const RVVVectorShapeConfig *> selectedShape =
      resolveRVVMicrokernelCandidateSelectedShape(candidate, family);
  if (!selectedShape)
    return selectedShape.takeError();

  RVVBinaryIntrinsicDescriptor descriptor =
      getRVVBinaryIntrinsicDescriptor(family, **selectedShape);
  if (!candidateMatchesRVVRouteRegistration(candidate, descriptor)) {
    llvm::StringRef expectedDescription =
        family.dtype == RVVBinaryDTypeKind::I64
            ? "supported RVV i64 microkernel ABI metadata"
            : "supported RVV i32 microkernel family ABI metadata";
    return makeModuleMicrokernelError(
        llvm::Twine("target artifact route '") + candidate.routeID +
        "' does not match " + expectedDescription + "; "
        "expected emission_kind '" +
        route->getEmissionKind() + "', artifact_kind '" +
        route->getArtifactKind() + "', runtime_abi '" +
        route->getRuntimeABI() + "', runtime_abi_kind '" +
        route->getRuntimeABIKind() + "', runtime_abi_name '" +
        route->getRuntimeABIName() + "', runtime_glue_role '" +
        route->getRuntimeGlueRole() + "'");
  }

  TargetArtifactExporter sourceExporter =
      buildRVVMicrokernelSourceTargetArtifactExporter(
          *route, /*enableCandidateValidation=*/false);
  if (llvm::Error error =
          validateTargetArtifactCandidateAgainstExporter(candidate,
                                                        sourceExporter))
    return error;
  if (llvm::Error error =
          validateRVVMicrokernelSelectedPlanMetadata(candidate, descriptor))
    return error;
  if (hasMatchingRVVMicrokernelAttachmentForCandidate(candidate, family)) {
    mlir::ModuleOp module =
        candidate.kernel->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return makeMicrokernelError(
          candidate.kernel,
          "selected RVV target artifact candidate requires an enclosing "
          "builtin.module for selected config/runtime AVL contract "
          "validation");
    llvm::Expected<RVVBinarySelectedConfigContract> selectedContract =
        resolveRVVMicrokernelSelectedConfigContractAuthority(
            module, family, candidate.selectedVariant, candidate.role,
            candidate.routeID);
    if (!selectedContract)
      return selectedContract.takeError();
    if (llvm::Error error = validateRVVMicrokernelSelectedPlanMetadata(
            candidate, *selectedContract))
      return error;
  }
  return validateRVVBinaryCandidateRuntimeABIMirrorsIR(candidate, descriptor);
}

TargetArtifactRouteMetadata
buildRVVMicrokernelSourceRouteMetadata(
    const RVVBinaryFamilyDescriptor &family) {
  TargetArtifactRouteMetadata metadata(
      family.runtimeABI, family.runtimeABIKind, family.runtimeABIName,
      family.runtimeGlueRole);

  if (family.dtype == RVVBinaryDTypeKind::I32 ||
      family.dtype == RVVBinaryDTypeKind::I64) {
    llvm::StringRef typedRole = getRVVTypedBinarySourceMetadataRole();
    metadata.addSelectedPlanMetadataRequirement(
        getRVVSelectedBinaryDTypeMetadataName(), family.dtypeID, typedRole);
    metadata.addSelectedPlanMetadataRequirement(
        getRVVSelectedBinaryFamilyMetadataName(), family.familyID, typedRole);
    metadata.addSelectedPlanMetadataRequirement(
        getRVVSelectedBinaryOperatorMetadataName(), family.arithmeticVerb,
        typedRole);
    metadata.addSelectedPlanMetadataRequirement(
        getRVVEmitCSourceOpMetadataName(), family.arithmeticOpName,
        getRVVEmitCSourceOpMetadataRole());
    metadata.addSelectedPlanMetadataRequirement(
        getRVVEmitCLowerableOpInterfaceMetadataName(),
        "TCRVEmitCLowerableOpInterface", getRVVEmitCSourceOpMetadataRole());
  } else {
    llvm::StringRef descriptorRole =
        getRVVLegacyDescriptorMirrorMetadataRole();
    metadata.addSelectedPlanMetadataRequirement(
        getRVVSelectedBinaryDTypeMetadataName(), family.dtypeID,
        descriptorRole);
    metadata.addSelectedPlanMetadataRequirement(
        getRVVSelectedBinaryFamilyMetadataName(), family.familyID,
        descriptorRole);
    metadata.addSelectedPlanMetadataRequirement(
        getRVVSelectedBinaryOperatorMetadataName(), family.arithmeticVerb,
        descriptorRole);
    metadata.addSelectedPlanMetadataRequirement(
        getRVVSelectedLoweringDescriptorMetadataName(),
        family.loweringDescriptor, descriptorRole);
  }
  metadata.addSelectedPlanMetadataPresenceRequirement(
      getRVVRuntimeElementCountCNameMetadataName(),
      getRVVRuntimeControlNameMetadataRole());
  metadata.addSelectedPlanMetadataPresenceRequirement(
      getRVVDescriptorElementCountMetadataName(),
      getRVVLegacyDescriptorMirrorMetadataRole());

  llvm::StringRef shapeRole = getSelectedRVVVectorShapeMetadataRole();
  metadata.addSelectedPlanMetadataPresenceRequirement(
      getRVVSelectedVectorShapeAttrName(), shapeRole);
  metadata.addSelectedPlanMetadataPresenceRequirement(
      getRVVSelectedVectorSEWAttrName(), shapeRole);
  metadata.addSelectedPlanMetadataPresenceRequirement(
      getRVVSelectedVectorLMULAttrName(), shapeRole);
  metadata.addSelectedPlanMetadataPresenceRequirement(
      getRVVSelectedTailPolicyAttrName(), shapeRole);
  metadata.addSelectedPlanMetadataPresenceRequirement(
      getRVVSelectedMaskPolicyAttrName(), shapeRole);
  metadata.addSelectedPlanMetadataPresenceRequirement(
      getRVVSelectedVectorTypeAttrName(), shapeRole);
  metadata.addSelectedPlanMetadataPresenceRequirement(
      getRVVSelectedVectorSuffixAttrName(), shapeRole);
  metadata.addSelectedPlanMetadataPresenceRequirement(
      getRVVSelectedSetVLSuffixAttrName(), shapeRole);

  llvm::StringRef capabilityRole =
      getSelectedRVVVectorShapeCapabilityMetadataRole();
  metadata.addSelectedPlanMetadataPresenceRequirement(
      getRVVSelectedVectorSEWCapabilityAttrName(), capabilityRole);
  metadata.addSelectedPlanMetadataPresenceRequirement(
      getRVVSelectedVectorLMULCapabilityAttrName(), capabilityRole);
  metadata.addSelectedPlanMetadataPresenceRequirement(
      getRVVSelectedTailPolicyCapabilityAttrName(), capabilityRole);
  metadata.addSelectedPlanMetadataPresenceRequirement(
      getRVVSelectedMaskPolicyCapabilityAttrName(), capabilityRole);

  llvm::StringRef runtimeVLRole = getRVVRuntimeVLBoundaryMetadataRole();
  metadata.addSelectedPlanMetadataRequirement(
      getRVVRuntimeAVLSourceMetadataName(),
      getRVVRuntimeAVLSourceMetadataValue(), runtimeVLRole);
  metadata.addSelectedPlanMetadataRequirement(
      getRVVRuntimeAVLRoleMetadataName(), getRVVRuntimeAVLRoleMetadataValue(),
      runtimeVLRole);
  metadata.addSelectedPlanMetadataRequirement(
      getRVVRuntimeVLSourceMetadataName(), getRVVRuntimeVLSourceMetadataValue(),
      runtimeVLRole);
  metadata.addSelectedPlanMetadataRequirement(
      getRVVRuntimeVLScopeMetadataName(), getRVVRuntimeVLScopeMetadataValue(),
      runtimeVLRole);

  addRVVMicrokernelConservativeRouteClaims(metadata);
  return metadata;
}

void addRVVMicrokernelConservativeRouteClaims(
    TargetArtifactRouteMetadata &metadata) {
  metadata.addClaimField("compile_export_claim", "compiler-artifact-only");
  metadata.addClaimField("runtime_correctness_claim", "none");
  metadata.addClaimField("hardware_execution_claim", "none");
  metadata.addClaimField("performance_claim", "none");
}

TargetArtifactRouteMetadata buildRVVMicrokernelArtifactRouteMetadata(
    const RVVMicrokernelDirectRouteManifestEntry &route) {
  if (route.routeKind == RVVMicrokernelDirectRouteKind::Source)
    return buildRVVMicrokernelSourceRouteMetadata(*route.family);

  TargetArtifactRouteMetadata metadata(
      route.getRuntimeABI(), route.getRuntimeABIKind(),
      route.getRuntimeABIName(), route.getRuntimeGlueRole());
  addRVVMicrokernelConservativeRouteClaims(metadata);
  return metadata;
}

llvm::Error validateRVVMicrokernelCallableCandidatePreflight(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  if (candidates.size() != 1)
    return makeModuleMicrokernelError(
        "RVV microkernel helper routes require exactly one callable artifact "
        "candidate for preflight");
  return validateRVVMicrokernelSourceCandidate(candidates.front());
}

llvm::Expected<llvm::SmallVector<support::RuntimeABIParameter, 5>>
resolveRVVMicrokernelRuntimeABIParameters(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  if (llvm::Error error =
          validateRVVMicrokernelCallableCandidatePreflight(candidates))
    return std::move(error);

  llvm::SmallVector<support::RuntimeABIParameter, 5> parameters;
  parameters.append(candidates.front().runtimeABIParameters.begin(),
                    candidates.front().runtimeABIParameters.end());
  return parameters;
}

llvm::Expected<bool> matchRVVMicrokernelAddObjectCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  if (candidates.size() != 1)
    return false;
  return candidateMatchesRVVMicrokernelFamily(candidates.front(),
                                             getI32VAddFamilySpec());
}

llvm::Expected<bool> matchRVVMicrokernelSubObjectCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  if (candidates.size() != 1)
    return false;
  return candidateMatchesRVVMicrokernelFamily(candidates.front(),
                                             getI32VSubFamilySpec());
}

llvm::Expected<bool> matchRVVMicrokernelMulObjectCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  if (candidates.size() != 1)
    return false;
  return candidateMatchesRVVMicrokernelFamily(candidates.front(),
                                             getI32VMulFamilySpec());
}

llvm::Expected<bool> matchRVVMicrokernelI64FamilyCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates,
    const RVVBinaryIntrinsicDescriptor &descriptor) {
  if (candidates.size() != 1)
    return false;
  return candidateMatchesRVVRouteRegistration(candidates.front(), descriptor);
}

llvm::Expected<bool> matchRVVMicrokernelI64VAddObjectCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  return matchRVVMicrokernelI64FamilyCandidate(
      candidates, getI64VAddIntrinsicDescriptor());
}

llvm::Expected<bool> matchRVVMicrokernelI64VSubObjectCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  return matchRVVMicrokernelI64FamilyCandidate(
      candidates, getI64VSubIntrinsicDescriptor());
}

llvm::Expected<bool> matchRVVMicrokernelI64VMulObjectCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  return matchRVVMicrokernelI64FamilyCandidate(
      candidates, getI64VMulIntrinsicDescriptor());
}

llvm::Expected<bool> matchRVVMicrokernelAddHeaderCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  if (candidates.size() != 1)
    return false;
  return candidateMatchesRVVMicrokernelFamily(candidates.front(),
                                             getI32VAddFamilySpec());
}

llvm::Expected<bool> matchRVVMicrokernelSubHeaderCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  if (candidates.size() != 1)
    return false;
  return candidateMatchesRVVMicrokernelFamily(candidates.front(),
                                             getI32VSubFamilySpec());
}

llvm::Expected<bool> matchRVVMicrokernelMulHeaderCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  if (candidates.size() != 1)
    return false;
  return candidateMatchesRVVMicrokernelFamily(candidates.front(),
                                             getI32VMulFamilySpec());
}

llvm::Expected<bool> matchRVVMicrokernelI64VAddHeaderCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  return matchRVVMicrokernelI64FamilyCandidate(
      candidates, getI64VAddIntrinsicDescriptor());
}

llvm::Expected<bool> matchRVVMicrokernelI64VSubHeaderCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  return matchRVVMicrokernelI64FamilyCandidate(
      candidates, getI64VSubIntrinsicDescriptor());
}

llvm::Expected<bool> matchRVVMicrokernelI64VMulHeaderCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  return matchRVVMicrokernelI64FamilyCandidate(
      candidates, getI64VMulIntrinsicDescriptor());
}

TargetArtifactCompositeMatchFn
getRVVMicrokernelHeaderMatchFn(const RVVBinaryFamilyDescriptor &family) {
  switch (family.dtype) {
  case RVVBinaryDTypeKind::I32:
    switch (family.arithmetic) {
    case RVVBinaryArithmeticKind::Add:
      return matchRVVMicrokernelAddHeaderCandidate;
    case RVVBinaryArithmeticKind::Sub:
      return matchRVVMicrokernelSubHeaderCandidate;
    case RVVBinaryArithmeticKind::Mul:
      return matchRVVMicrokernelMulHeaderCandidate;
    }
    break;
  case RVVBinaryDTypeKind::I64:
    switch (family.arithmetic) {
    case RVVBinaryArithmeticKind::Add:
      return matchRVVMicrokernelI64VAddHeaderCandidate;
    case RVVBinaryArithmeticKind::Sub:
      return matchRVVMicrokernelI64VSubHeaderCandidate;
    case RVVBinaryArithmeticKind::Mul:
      return matchRVVMicrokernelI64VMulHeaderCandidate;
    }
    break;
  }
  llvm_unreachable("unknown RVV binary family for header route");
}

TargetArtifactCompositeMatchFn
getRVVMicrokernelObjectMatchFn(const RVVBinaryFamilyDescriptor &family) {
  switch (family.dtype) {
  case RVVBinaryDTypeKind::I32:
    switch (family.arithmetic) {
    case RVVBinaryArithmeticKind::Add:
      return matchRVVMicrokernelAddObjectCandidate;
    case RVVBinaryArithmeticKind::Sub:
      return matchRVVMicrokernelSubObjectCandidate;
    case RVVBinaryArithmeticKind::Mul:
      return matchRVVMicrokernelMulObjectCandidate;
    }
    break;
  case RVVBinaryDTypeKind::I64:
    switch (family.arithmetic) {
    case RVVBinaryArithmeticKind::Add:
      return matchRVVMicrokernelI64VAddObjectCandidate;
    case RVVBinaryArithmeticKind::Sub:
      return matchRVVMicrokernelI64VSubObjectCandidate;
    case RVVBinaryArithmeticKind::Mul:
      return matchRVVMicrokernelI64VMulObjectCandidate;
    }
    break;
  }
  llvm_unreachable("unknown RVV binary family for object route");
}

TargetArtifactCompositeExporter buildRVVMicrokernelCompositeTargetArtifactExporter(
    const RVVMicrokernelDirectRouteManifestEntry &route) {
  TargetArtifactCompositeMatchFn matchFn = nullptr;
  switch (route.routeKind) {
  case RVVMicrokernelDirectRouteKind::Source:
    llvm_unreachable("RVV source routes are standalone target exporters");
  case RVVMicrokernelDirectRouteKind::Header:
    matchFn = getRVVMicrokernelHeaderMatchFn(*route.family);
    break;
  case RVVMicrokernelDirectRouteKind::Object:
    matchFn = getRVVMicrokernelObjectMatchFn(*route.family);
    break;
  }

  return TargetArtifactCompositeExporter(
      route.getRouteID(), route.getArtifactKind(), matchFn,
      getRVVMicrokernelExactExportFn(*route.family, route.routeKind),
      route.getOwner(), route.getRuntimeABIKind(), route.getRuntimeABIName(),
      resolveRVVMicrokernelRuntimeABIParameters,
      route.isDirectHelperCompatibilityRoute(), route.getComponentGroup(),
      route.getExternalABIName(), validateRVVMicrokernelCallableCandidatePreflight,
      buildRVVMicrokernelArtifactRouteMetadata(route));
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
      "tcrv-rvv-microkernel", "c", fd, sourceFile.path);
  if (ec)
    return makeModuleMicrokernelObjectError(
        llvm::Twine("failed to create temporary C source for object export: ") +
        ec.message());

  llvm::raw_fd_ostream stream(fd, /*shouldClose=*/true);
  stream << source;
  stream.close();
  if (stream.has_error())
    return makeModuleMicrokernelObjectError(
        "failed to write generated RVV microkernel C source before object "
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

std::string formatCompileCommand(const RVVMicrokernelRecord &record,
                                 llvm::StringRef compilerPath) {
  std::string command;
  llvm::raw_string_ostream stream(command);
  stream << compilerPath << " -target " << record.targetTriple
         << " -O2 -march=" << record.selectedMarch;
  if (record.selectedMABI)
    stream << " -mabi=" << *record.selectedMABI;
  stream << " -c <generated-rvv-microkernel-source> -o <object-file>";
  stream.flush();
  return command;
}

llvm::Error compileGeneratedMicrokernelSourceToObject(
    const RVVMicrokernelRecord &record, llvm::StringRef source,
    llvm::raw_ostream &os) {
  if (llvm::Error error =
          validateBoundedCompileText(record.kernelSymbol, "target triple",
                                     record.targetTriple))
    return error;
  if (llvm::Error error =
          validateBoundedCompileText(record.kernelSymbol, "selected_march",
                                     record.selectedMarch))
    return error;
  if (record.selectedMABI)
    if (llvm::Error error =
            validateBoundedCompileText(record.kernelSymbol, "selected_mabi",
                                       *record.selectedMABI))
      return error;

  llvm::ErrorOr<std::string> clangPath =
      llvm::sys::findProgramByName("clang");
  if (!clangPath)
    clangPath = llvm::sys::findProgramByName("clang-20");
  if (!clangPath)
    return makeMicrokernelObjectError(
        record.kernelSymbol,
        "requires clang or clang-20 on PATH to compile the bounded RVV "
        "microkernel C source into an object file");

  TemporaryFile sourceFile;
  if (llvm::Error error = writeTempSource(source, sourceFile))
    return error;

  TemporaryFile objectFile;
  if (llvm::Error error =
          createTempFile("tcrv-rvv-microkernel", "o", objectFile))
    return error;

  TemporaryFile stderrFile;
  if (llvm::Error error =
          createTempFile("tcrv-rvv-microkernel", "stderr", stderrFile))
    return error;

  std::string marchArg = "-march=" + record.selectedMarch;
  std::string mabiArg;
  llvm::SmallVector<llvm::StringRef, 8> args;
  args.push_back(*clangPath);
  args.push_back("-target");
  args.push_back(record.targetTriple);
  args.push_back("-O2");
  args.push_back(marchArg);
  if (record.selectedMABI) {
    mabiArg = "-mabi=" + *record.selectedMABI;
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
        llvm::Twine("RVV object compiler failed while creating object file; "
                    "command: ") +
            formatCompileCommand(record, *clangPath) + "; exit_code=" +
            llvm::Twine(exitCode) + "; stderr: " + stderrText);
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> objectBuffer =
      llvm::MemoryBuffer::getFile(objectFile.get(), /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!objectBuffer)
    return makeMicrokernelObjectError(
        record.kernelSymbol, llvm::Twine("failed to read generated object "
                                         "file: ") +
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

llvm::Error exportI32VAddMicrokernelC(mlir::ModuleOp module,
                                      llvm::raw_ostream &os) {
  return exportRVVMicrokernelCForBinaryFamily(
      module, getI32VAddFamilyRegistrationRecord(), os);
}

llvm::Error exportI32VAddMicrokernelHeader(mlir::ModuleOp module,
                                           llvm::raw_ostream &os) {
  return exportRVVMicrokernelHeaderForBinaryFamily(
      module, getI32VAddFamilyRegistrationRecord(), os);
}

llvm::Error exportI32VAddMicrokernelObject(mlir::ModuleOp module,
                                           llvm::raw_ostream &os) {
  return exportRVVMicrokernelObjectForBinaryFamily(
      module, getI32VAddFamilyRegistrationRecord(), os);
}

llvm::Error exportI32VSubMicrokernelC(mlir::ModuleOp module,
                                      llvm::raw_ostream &os) {
  return exportRVVMicrokernelCForBinaryFamily(
      module, getI32VSubFamilyRegistrationRecord(), os);
}

llvm::Error exportI32VSubMicrokernelHeader(mlir::ModuleOp module,
                                           llvm::raw_ostream &os) {
  return exportRVVMicrokernelHeaderForBinaryFamily(
      module, getI32VSubFamilyRegistrationRecord(), os);
}

llvm::Error exportI32VSubMicrokernelObject(mlir::ModuleOp module,
                                           llvm::raw_ostream &os) {
  return exportRVVMicrokernelObjectForBinaryFamily(
      module, getI32VSubFamilyRegistrationRecord(), os);
}

llvm::Error exportI32VMulMicrokernelC(mlir::ModuleOp module,
                                      llvm::raw_ostream &os) {
  return exportRVVMicrokernelCForBinaryFamily(
      module, getI32VMulFamilyRegistrationRecord(), os);
}

llvm::Error exportI32VMulMicrokernelHeader(mlir::ModuleOp module,
                                           llvm::raw_ostream &os) {
  return exportRVVMicrokernelHeaderForBinaryFamily(
      module, getI32VMulFamilyRegistrationRecord(), os);
}

llvm::Error exportI32VMulMicrokernelObject(mlir::ModuleOp module,
                                           llvm::raw_ostream &os) {
  return exportRVVMicrokernelObjectForBinaryFamily(
      module, getI32VMulFamilyRegistrationRecord(), os);
}

llvm::Error exportI64VAddMicrokernelC(mlir::ModuleOp module,
                                      llvm::raw_ostream &os) {
  return exportRVVMicrokernelCForBinaryFamily(
      module, getI64VAddFamilyRegistrationRecord(), os);
}

llvm::Error exportI64VAddMicrokernelHeader(mlir::ModuleOp module,
                                           llvm::raw_ostream &os) {
  return exportRVVMicrokernelHeaderForBinaryFamily(
      module, getI64VAddFamilyRegistrationRecord(), os);
}

llvm::Error exportI64VAddMicrokernelObject(mlir::ModuleOp module,
                                           llvm::raw_ostream &os) {
  return exportRVVMicrokernelObjectForBinaryFamily(
      module, getI64VAddFamilyRegistrationRecord(), os);
}

llvm::Error exportI64VSubMicrokernelC(mlir::ModuleOp module,
                                      llvm::raw_ostream &os) {
  return exportRVVMicrokernelCForBinaryFamily(
      module, getI64VSubFamilyRegistrationRecord(), os);
}

llvm::Error exportI64VSubMicrokernelHeader(mlir::ModuleOp module,
                                           llvm::raw_ostream &os) {
  return exportRVVMicrokernelHeaderForBinaryFamily(
      module, getI64VSubFamilyRegistrationRecord(), os);
}

llvm::Error exportI64VSubMicrokernelObject(mlir::ModuleOp module,
                                           llvm::raw_ostream &os) {
  return exportRVVMicrokernelObjectForBinaryFamily(
      module, getI64VSubFamilyRegistrationRecord(), os);
}

llvm::Error exportI64VMulMicrokernelC(mlir::ModuleOp module,
                                      llvm::raw_ostream &os) {
  return exportRVVMicrokernelCForBinaryFamily(
      module, getI64VMulFamilyRegistrationRecord(), os);
}

llvm::Error exportI64VMulMicrokernelHeader(mlir::ModuleOp module,
                                           llvm::raw_ostream &os) {
  return exportRVVMicrokernelHeaderForBinaryFamily(
      module, getI64VMulFamilyRegistrationRecord(), os);
}

llvm::Error exportI64VMulMicrokernelObject(mlir::ModuleOp module,
                                           llvm::raw_ostream &os) {
  return exportRVVMicrokernelObjectForBinaryFamily(
      module, getI64VMulFamilyRegistrationRecord(), os);
}

tianchenrv::target::TargetArtifactExportFn getExportFnForRouteKind(
    RVVMicrokernelDirectRouteKind routeKind,
    tianchenrv::target::TargetArtifactExportFn sourceFn,
    tianchenrv::target::TargetArtifactExportFn headerFn,
    tianchenrv::target::TargetArtifactExportFn objectFn) {
  switch (routeKind) {
  case RVVMicrokernelDirectRouteKind::Source:
    return sourceFn;
  case RVVMicrokernelDirectRouteKind::Header:
    return headerFn;
  case RVVMicrokernelDirectRouteKind::Object:
    return objectFn;
  }
  llvm_unreachable("unknown RVV microkernel direct route kind");
}

tianchenrv::target::TargetArtifactExportFn
getRVVMicrokernelExactExportFn(const RVVBinaryFamilyDescriptor &family,
                               RVVMicrokernelDirectRouteKind routeKind) {
  switch (family.dtype) {
  case RVVBinaryDTypeKind::I32:
    switch (family.arithmetic) {
    case RVVBinaryArithmeticKind::Add:
      return getExportFnForRouteKind(routeKind, exportI32VAddMicrokernelC,
                                     exportI32VAddMicrokernelHeader,
                                     exportI32VAddMicrokernelObject);
    case RVVBinaryArithmeticKind::Sub:
      return getExportFnForRouteKind(routeKind, exportI32VSubMicrokernelC,
                                     exportI32VSubMicrokernelHeader,
                                     exportI32VSubMicrokernelObject);
    case RVVBinaryArithmeticKind::Mul:
      return getExportFnForRouteKind(routeKind, exportI32VMulMicrokernelC,
                                     exportI32VMulMicrokernelHeader,
                                     exportI32VMulMicrokernelObject);
    }
    break;
  case RVVBinaryDTypeKind::I64:
    switch (family.arithmetic) {
    case RVVBinaryArithmeticKind::Add:
      return getExportFnForRouteKind(routeKind, exportI64VAddMicrokernelC,
                                     exportI64VAddMicrokernelHeader,
                                     exportI64VAddMicrokernelObject);
    case RVVBinaryArithmeticKind::Sub:
      return getExportFnForRouteKind(routeKind, exportI64VSubMicrokernelC,
                                     exportI64VSubMicrokernelHeader,
                                     exportI64VSubMicrokernelObject);
    case RVVBinaryArithmeticKind::Mul:
      return getExportFnForRouteKind(routeKind, exportI64VMulMicrokernelC,
                                     exportI64VMulMicrokernelHeader,
                                     exportI64VMulMicrokernelObject);
    }
    break;
  }
  llvm_unreachable("unknown RVV binary family");
}

} // namespace

llvm::StringRef RVVMicrokernelDirectRouteManifestEntry::getRouteID() const {
  switch (routeKind) {
  case RVVMicrokernelDirectRouteKind::Source:
    return family->routeID;
  case RVVMicrokernelDirectRouteKind::Header:
    return family->headerRouteID;
  case RVVMicrokernelDirectRouteKind::Object:
    return family->objectRouteID;
  }
  llvm_unreachable("unknown RVV microkernel direct route kind");
}

llvm::StringRef
RVVMicrokernelDirectRouteManifestEntry::getArtifactKind() const {
  switch (routeKind) {
  case RVVMicrokernelDirectRouteKind::Source:
    return kMicrokernelArtifactKind;
  case RVVMicrokernelDirectRouteKind::Header:
    return kMicrokernelHeaderArtifactKind;
  case RVVMicrokernelDirectRouteKind::Object:
    return kMicrokernelObjectArtifactKind;
  }
  llvm_unreachable("unknown RVV microkernel direct route kind");
}

llvm::StringRef RVVMicrokernelDirectRouteManifestEntry::getOwner() const {
  return kRVVPluginName;
}

llvm::StringRef
RVVMicrokernelDirectRouteManifestEntry::getEmissionKind() const {
  return family->emissionKind;
}

llvm::StringRef RVVMicrokernelDirectRouteManifestEntry::getRuntimeABI() const {
  return family->runtimeABI;
}

llvm::StringRef
RVVMicrokernelDirectRouteManifestEntry::getRuntimeABIKind() const {
  return family->runtimeABIKind;
}

llvm::StringRef
RVVMicrokernelDirectRouteManifestEntry::getRuntimeABIName() const {
  return family->runtimeABIName;
}

llvm::StringRef
RVVMicrokernelDirectRouteManifestEntry::getRuntimeGlueRole() const {
  return family->runtimeGlueRole;
}

llvm::StringRef
RVVMicrokernelDirectRouteManifestEntry::getComponentGroup() const {
  return family->externalABIComponentGroup;
}

llvm::StringRef
RVVMicrokernelDirectRouteManifestEntry::getExternalABIName() const {
  return family->runtimeABIName;
}

llvm::StringRef
RVVMicrokernelDirectRouteManifestEntry::getComponentRole() const {
  switch (routeKind) {
  case RVVMicrokernelDirectRouteKind::Source:
    return "source";
  case RVVMicrokernelDirectRouteKind::Header:
    return "header";
  case RVVMicrokernelDirectRouteKind::Object:
    return "object";
  }
  llvm_unreachable("unknown RVV microkernel direct route kind");
}

std::string RVVMicrokernelDirectRouteManifestEntry::getDescription() const {
  switch (routeKind) {
  case RVVMicrokernelDirectRouteKind::Source:
    return (llvm::Twine("export one runtime-callable RVV ") +
            family->familyID + " microkernel C source")
        .str();
  case RVVMicrokernelDirectRouteKind::Header:
    return (llvm::Twine("export one RVV ") + family->familyID +
            " microkernel runtime-callable C ABI header")
        .str();
  case RVVMicrokernelDirectRouteKind::Object:
    return (llvm::Twine("export one RVV ") + family->familyID +
            " microkernel library object file")
        .str();
  }
  llvm_unreachable("unknown RVV microkernel direct route kind");
}

bool RVVMicrokernelDirectRouteManifestEntry::requiresBinaryStdout() const {
  return routeKind == RVVMicrokernelDirectRouteKind::Object;
}

bool RVVMicrokernelDirectRouteManifestEntry::
    isDirectHelperCompatibilityRoute() const {
  return true;
}

llvm::ArrayRef<RVVMicrokernelDirectRouteKind>
getRVVMicrokernelDirectRouteKinds() {
  static const RVVMicrokernelDirectRouteKind routeKinds[] = {
      RVVMicrokernelDirectRouteKind::Source,
      RVVMicrokernelDirectRouteKind::Header,
      RVVMicrokernelDirectRouteKind::Object,
  };
  return routeKinds;
}

std::size_t getRVVMicrokernelDirectRouteCount() {
  return getRVVBinaryFamilyRegistrationRecords().size() *
         getRVVMicrokernelDirectRouteKinds().size();
}

llvm::ArrayRef<RVVMicrokernelArtifactRouteDescriptor>
getRVVMicrokernelArtifactRouteAuthority() {
  static const llvm::SmallVector<RVVMicrokernelDirectRouteManifestEntry, 32>
      routes = [] {
        llvm::SmallVector<RVVMicrokernelDirectRouteManifestEntry, 32> result;
        result.reserve(getRVVMicrokernelDirectRouteCount());
        for (const RVVBinaryFamilyDescriptor *family :
             getRVVBinaryFamilyRegistrationRecords()) {
          for (RVVMicrokernelDirectRouteKind routeKind :
               getRVVMicrokernelDirectRouteKinds())
            result.push_back({family, routeKind});
        }
        return result;
      }();
  return llvm::ArrayRef(routes);
}

llvm::ArrayRef<RVVMicrokernelDirectRouteManifestEntry>
getRVVMicrokernelDirectRouteManifest() {
  return getRVVMicrokernelArtifactRouteAuthority();
}

const RVVMicrokernelDirectRouteManifestEntry *
lookupRVVMicrokernelDirectRoute(llvm::StringRef routeID) {
  routeID = routeID.trim();
  for (const RVVMicrokernelDirectRouteManifestEntry &route :
       getRVVMicrokernelDirectRouteManifest())
    if (route.family && route.getRouteID() == routeID)
      return &route;
  return nullptr;
}

const RVVMicrokernelDirectRouteManifestEntry *
lookupRVVMicrokernelDirectRoute(
    const RVVBinaryFamilyDescriptor &family,
    RVVMicrokernelDirectRouteKind routeKind) {
  for (const RVVMicrokernelDirectRouteManifestEntry &route :
       getRVVMicrokernelDirectRouteManifest())
    if (route.family && route.routeKind == routeKind &&
        isSameRVVBinaryFamily(*route.family, family))
      return &route;
  return nullptr;
}

llvm::Error exportRVVMicrokernelDirectRoute(
    mlir::ModuleOp module, const RVVMicrokernelDirectRouteManifestEntry &route,
    llvm::raw_ostream &os) {
  if (!route.family)
    return makeModuleMicrokernelError(
        "RVV microkernel direct artifact export requires an exact typed "
        "binary family route");

  llvm::Expected<RVVMicrokernelRecord> record =
      buildModuleRecordForRVVBinaryFamily(module, *route.family,
                                          route.getRouteID());
  if (!record) {
    if (route.routeKind == RVVMicrokernelDirectRouteKind::Object) {
      std::string message = llvm::toString(record.takeError());
      return makeModuleMicrokernelObjectError(message);
    }
    return record.takeError();
  }

  switch (route.routeKind) {
  case RVVMicrokernelDirectRouteKind::Source: {
    std::string source;
    llvm::raw_string_ostream stream(source);
    if (llvm::Error error = printMicrokernelSource(
            *record, stream, RVVMicrokernelCExportMode::RuntimeCallableLibrary))
      return error;
    stream.flush();
    os << source;
    return llvm::Error::success();
  }
  case RVVMicrokernelDirectRouteKind::Header:
    printMicrokernelHeader(*record, os);
    return llvm::Error::success();
  case RVVMicrokernelDirectRouteKind::Object: {
    std::string source;
    llvm::raw_string_ostream stream(source);
    if (llvm::Error error = printMicrokernelSource(
            *record, stream, RVVMicrokernelCExportMode::RuntimeCallableLibrary))
      return error;
    stream.flush();
    if (source.empty())
      return makeMicrokernelObjectError(
          record->kernelSymbol,
          "validated RVV microkernel C source must be non-empty before object "
          "export");
    return compileGeneratedMicrokernelSourceToObject(*record, source, os);
  }
  }
  llvm_unreachable("unknown RVV microkernel direct route kind");
}

llvm::Error exportRVVMicrokernelCForBinaryFamily(
    mlir::ModuleOp module, const RVVBinaryFamilyDescriptor &family,
    llvm::raw_ostream &os) {
  const RVVMicrokernelDirectRouteManifestEntry *route =
      lookupRVVMicrokernelDirectRoute(
          family, RVVMicrokernelDirectRouteKind::Source);
  if (!route)
    return makeModuleMicrokernelError(
        llvm::Twine("no manifest-backed RVV source export route for ") +
        family.familyID);
  return exportRVVMicrokernelDirectRoute(module, *route, os);
}

llvm::Error exportRVVMicrokernelCForFamily(
    mlir::ModuleOp module, i32_binary::I32BinaryFamilyKind family,
    llvm::raw_ostream &os) {
  RVVI32MicrokernelKind descriptorKind = convertI32BinaryFamilyKind(family);
  const RVVI32MicrokernelFamilySpec &expected =
      getI32MicrokernelFamilySpec(descriptorKind);
  return exportRVVMicrokernelCForBinaryFamily(module, expected, os);
}

llvm::Error validateRVVMicrokernelSourceAuthority(
    mlir::ModuleOp module, const RVVBinaryFamilyDescriptor &family,
    llvm::StringRef selectedVariant, llvm::StringRef role,
    llvm::StringRef routeID) {
  llvm::Expected<RVVBinarySelectedConfigContract> contract =
      resolveRVVMicrokernelSelectedConfigContractAuthority(
          module, family, selectedVariant, role, routeID);
  if (!contract)
    return contract.takeError();
  return llvm::Error::success();
}

llvm::Expected<RVVBinarySelectedConfigContract>
resolveRVVMicrokernelSelectedConfigContractAuthority(
    mlir::ModuleOp module, const RVVBinaryFamilyDescriptor &family,
    llvm::StringRef selectedVariant, llvm::StringRef role,
    llvm::StringRef routeID) {
  llvm::Expected<RVVMicrokernelRecord> record =
      buildModuleRecordForRVVBinaryFamily(module, family, routeID);
  if (!record)
    return record.takeError();

  if (llvm::Error error = requireRVVSourceAuthorityField(
          "selected variant", record->variantSymbol, selectedVariant, routeID,
          selectedVariant))
    return error;
  if (llvm::Error error = requireRVVSourceAuthorityField(
          "role", record->role, role, routeID, selectedVariant))
    return error;
  if (llvm::Error error = requireRVVSourceAuthorityField(
          "active route", record->activeRouteID, routeID, routeID,
          selectedVariant))
    return error;

  if (record->descriptor.getRVVMicrokernelOpName() !=
      family.microkernelOpName)
    return makeModuleMicrokernelError(
        llvm::Twine("selected RVV component authority for route '") +
        routeID + "' requires " + family.microkernelOpName +
        " but typed RVV record is " +
        record->descriptor.getRVVMicrokernelOpName());
  return record->selectedConfigContract;
}

llvm::Error exportRVVMicrokernelSelfCheckC(mlir::ModuleOp module,
                                           llvm::raw_ostream &os) {
  llvm::Expected<RVVMicrokernelRecord> record = buildModuleRecord(module);
  if (!record)
    return record.takeError();

  std::string source;
  llvm::raw_string_ostream stream(source);
  if (llvm::Error error = printMicrokernelSource(
          *record, stream, RVVMicrokernelCExportMode::SelfCheckHarness))
    return error;
  stream.flush();
  os << source;
  return llvm::Error::success();
}

llvm::Error exportRVVMicrokernelHeaderForBinaryFamily(
    mlir::ModuleOp module, const RVVBinaryFamilyDescriptor &family,
    llvm::raw_ostream &os) {
  const RVVMicrokernelDirectRouteManifestEntry *route =
      lookupRVVMicrokernelDirectRoute(
          family, RVVMicrokernelDirectRouteKind::Header);
  if (!route)
    return makeModuleMicrokernelError(
        llvm::Twine("no manifest-backed RVV header export route for ") +
        family.familyID);
  return exportRVVMicrokernelDirectRoute(module, *route, os);
}

llvm::Error exportRVVMicrokernelHeaderForFamily(
    mlir::ModuleOp module, i32_binary::I32BinaryFamilyKind family,
    llvm::raw_ostream &os) {
  RVVI32MicrokernelKind descriptorKind = convertI32BinaryFamilyKind(family);
  const RVVI32MicrokernelFamilySpec &expected =
      getI32MicrokernelFamilySpec(descriptorKind);
  return exportRVVMicrokernelHeaderForBinaryFamily(module, expected, os);
}

llvm::Error exportRVVMicrokernelObjectForBinaryFamily(
    mlir::ModuleOp module, const RVVBinaryFamilyDescriptor &family,
    llvm::raw_ostream &os) {
  const RVVMicrokernelDirectRouteManifestEntry *route =
      lookupRVVMicrokernelDirectRoute(
          family, RVVMicrokernelDirectRouteKind::Object);
  if (!route)
    return makeModuleMicrokernelObjectError(
        (llvm::Twine("no manifest-backed RVV object export route for ") +
         family.familyID)
            .str());
  return exportRVVMicrokernelDirectRoute(module, *route, os);
}

llvm::Error exportRVVMicrokernelObjectForFamily(
    mlir::ModuleOp module, i32_binary::I32BinaryFamilyKind family,
    llvm::raw_ostream &os) {
  RVVI32MicrokernelKind descriptorKind = convertI32BinaryFamilyKind(family);
  const RVVI32MicrokernelFamilySpec &expected =
      getI32MicrokernelFamilySpec(descriptorKind);
  return exportRVVMicrokernelObjectForBinaryFamily(module, expected, os);
}

llvm::Error registerRVVMicrokernelTargetExporters(
    TargetArtifactExporterRegistry &registry) {
  for (const RVVMicrokernelDirectRouteManifestEntry &route :
       getRVVMicrokernelArtifactRouteAuthority()) {
    switch (route.routeKind) {
    case RVVMicrokernelDirectRouteKind::Source: {
      if (llvm::Error error = registry.registerExporter(TargetArtifactExporter(
              buildRVVMicrokernelSourceTargetArtifactExporter(
                  route, /*enableCandidateValidation=*/true))))
        return error;
      break;
    }
    case RVVMicrokernelDirectRouteKind::Header:
      if (llvm::Error error = registry.registerCompositeExporter(
              buildRVVMicrokernelCompositeTargetArtifactExporter(route)))
        return error;
      break;
    case RVVMicrokernelDirectRouteKind::Object:
      if (llvm::Error error = registry.registerCompositeExporter(
              buildRVVMicrokernelCompositeTargetArtifactExporter(route)))
        return error;
      break;
    }
  }

  return llvm::Error::success();
}

llvm::Error registerRVVMicrokernelPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry) {
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      kRVVPluginName, registerRVVMicrokernelTargetExporters));
}

llvm::Error registerRVVMicrokernelTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry) {
  for (const RVVMicrokernelDirectRouteManifestEntry &route :
       getRVVMicrokernelArtifactRouteAuthority()) {
    const RVVMicrokernelDirectRouteManifestEntry *routePtr = &route;
    if (llvm::Error error = registry.registerRoute(TargetTranslateRoute(
            route.getRouteID(), route.getDescription(),
            [routePtr](mlir::ModuleOp module, llvm::raw_ostream &os) {
              return exportRVVMicrokernelDirectRoute(module, *routePtr, os);
            },
            route.requiresBinaryStdout(), route.getRouteID())))
      return error;
  }
  return llvm::Error::success();
}

} // namespace tianchenrv::target::rvv
