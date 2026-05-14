#include "TianChenRV/Target/RVV/RVVMicrokernel.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVLowerToEmitC.h"
#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Support/FiniteBinaryFrontendLowering.h"
#include "TianChenRV/Support/RuntimeABICallablePlan.h"
#include "TianChenRV/Support/RuntimeABIContract.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"
#include "TianChenRV/Target/BinarySelfCheckExpectation.h"
#include "TianChenRV/Target/I32BinaryFamilyRegistry.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"
#include "TianChenRV/Target/RVV/RVVBinaryMicrokernelBodyVerifier.h"
#include "TianChenRV/Target/RVV/RVVBinaryRoute.h"
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
constexpr llvm::StringLiteral kBoundarySelectedBinarySourceKindAttrName(
    "selected_binary_source_kind");
constexpr llvm::StringLiteral kBoundarySelectedBinaryDTypeAttrName(
    "selected_binary_dtype");
constexpr llvm::StringLiteral kBoundarySelectedBinaryFamilyAttrName(
    "selected_binary_family");
constexpr llvm::StringLiteral kBoundarySelectedBinaryOperatorAttrName(
    "selected_binary_operator");
constexpr llvm::StringLiteral kBoundarySelectedBinaryMicrokernelOpAttrName(
    "selected_binary_microkernel_op");
constexpr llvm::StringLiteral kBoundaryEmitCSourceOpAttrName(
    "emitc_source_op");
constexpr llvm::StringLiteral kBoundaryEmitCLowerableOpInterfaceAttrName(
    "emitc_lowerable_op_interface");
constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "TCRVEmitCLowerableOpInterface");
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
constexpr llvm::StringLiteral kDirectCSourceRouteDeletedReason(
    "RVV runtime-callable direct C semantic exporter was deleted; rebuild "
    "requires a materialized MLIR EmitC module source route");
using RVVI32MicrokernelKind =
    tianchenrv::target::rvv::RVVBinaryArithmeticKind;
using RVVI32MicrokernelFamilySpec =
    tianchenrv::target::rvv::RVVBinaryFamilyRecord;

struct SelectedPath {
  VariantOp variant;
  mlir::Operation *selector = nullptr;
  std::string role;
};

struct RVVMicrokernelRecord {
  const RVVI32MicrokernelFamilySpec *family = nullptr;
  RVVBinaryIntrinsicRoute descriptor;
  std::string activeRouteID;
  std::string kernelSymbol;
  std::string variantSymbol;
  std::string role;
  std::string targetTriple;
  std::string selectedMarch;
  std::optional<std::string> selectedMABI;
  llvm::SmallVector<std::string, 4> requiredCapabilities;
  llvm::SmallVector<support::RuntimeABIParameter, 4> runtimeABIParameters;
  llvm::SmallVector<SelectedPlanMetadataEntry, 24> selectedPlanMetadata;
  llvm::SmallVector<MemWindowOp, 3> bufferWindows;
  RuntimeParamOp runtimeElementCountParam;
  RVVBinarySelectedConfigContract selectedConfigContract;
  RVVBinarySelectedConfigEmissionView selectedConfigEmission;
  RVVBinaryEmitCBodyMapping emitcBodyMapping;
  std::string emitcBodyMappingSource;
  RVVBinaryDataflowEmissionPlan dataflowPlan;
  RVVIntrinsicConfig intrinsicConfig;
  const RVVI32VectorShapeConfig *selectedShape = nullptr;
  std::string selectedBinarySourceKind;
  std::int64_t elementCount = 0;
  std::int64_t controlPlaneSEW = 0;
  std::string controlPlaneLMUL;
  std::optional<support::FixedVectorSourceExtentContract> fixedSourceExtent;
  std::optional<support::DynamicVectorRuntimeExtentContract>
      dynamicRuntimeExtent;
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

bool isSameRVVBinaryFamily(const RVVBinaryFamilyRecord &lhs,
                           const RVVBinaryFamilyRecord &rhs) {
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

RVVBinaryIntrinsicRoute
getI32BinaryIntrinsicRouteForMicrokernel(
    const RVVI32MicrokernelFamilySpec &family,
    const RVVI32VectorShapeConfig &shape) {
  return getRVVBinaryIntrinsicRoute(family, shape);
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

const RVVBinaryFamilyRecord *
getI64MicrokernelFamilyForOp(mlir::Operation *op) {
  if (!op)
    return nullptr;
  llvm::StringRef opName = op->getName().getStringRef();
  for (const RVVBinaryFamilyRecord *family :
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
    const RVVBinaryIntrinsicRoute &descriptor) {
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
    const RVVBinaryIntrinsicRoute &descriptor) {
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

llvm::Error collectSelectedPlanMetadataEntries(
    KernelOp kernel, DiagnosticOp diagnostic,
    llvm::SmallVectorImpl<SelectedPlanMetadataEntry> &out) {
  auto metadata = diagnostic->getAttrOfType<mlir::ArrayAttr>(
      execDiagnostic::kSelectedPlanMetadataAttrName);
  if (!metadata)
    return llvm::Error::success();

  llvm::StringSet<> seenNames;
  for (auto [index, attr] : llvm::enumerate(metadata)) {
    auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(attr);
    if (!dict)
      return makeMicrokernelError(
          kernel, llvm::Twine("selected_plan_metadata[") +
                      llvm::Twine(index) + "] must be a dictionary attribute");

    auto name = dict.getAs<mlir::StringAttr>(
        execDiagnostic::kSelectedPlanMetadataNameAttrName);
    auto value = dict.getAs<mlir::StringAttr>(
        execDiagnostic::kSelectedPlanMetadataValueAttrName);
    auto role = dict.getAs<mlir::StringAttr>(
        execDiagnostic::kSelectedPlanMetadataRoleAttrName);
    auto note = dict.getAs<mlir::StringAttr>(
        execDiagnostic::kSelectedPlanMetadataNoteAttrName);
    if (!name || name.getValue().trim().empty())
      return makeMicrokernelError(
          kernel, llvm::Twine("selected_plan_metadata[") +
                      llvm::Twine(index) + "] requires non-empty name");
    if (!value || value.getValue().trim().empty())
      return makeMicrokernelError(
          kernel, llvm::Twine("selected_plan_metadata[") +
                      llvm::Twine(index) + "] requires non-empty value");
    if (!role || role.getValue().trim().empty())
      return makeMicrokernelError(
          kernel, llvm::Twine("selected_plan_metadata[") +
                      llvm::Twine(index) + "] requires non-empty role");
    if (!note || note.getValue().trim().empty())
      return makeMicrokernelError(
          kernel, llvm::Twine("selected_plan_metadata[") +
                      llvm::Twine(index) + "] requires non-empty note");

    llvm::StringRef nameValue = name.getValue().trim();
    llvm::StringRef metadataValue = value.getValue().trim();
    llvm::StringRef roleValue = role.getValue().trim();
    llvm::StringRef noteValue = note.getValue().trim();
    if (!seenNames.insert(nameValue).second)
      return makeMicrokernelError(
          kernel, llvm::Twine("duplicate selected_plan_metadata name '") +
                      nameValue + "'");
    if (llvm::Error error =
            validateBoundedText(kernel, "selected plan metadata name",
                                nameValue))
      return error;
    if (llvm::Error error = validateBoundedText(
            kernel, "selected plan metadata value", metadataValue))
      return error;
    if (llvm::Error error =
            validateBoundedText(kernel, "selected plan metadata role",
                                roleValue))
      return error;
    if (llvm::Error error =
            validateBoundedText(kernel, "selected plan metadata note",
                                noteValue))
      return error;

    out.push_back({nameValue.str(), metadataValue.str(), roleValue.str(),
                   noteValue.str()});
  }

  return llvm::Error::success();
}

llvm::Error validateRVVBinaryCallableABIParameterMirror(
    KernelOp kernel,
    llvm::ArrayRef<support::RuntimeABIParameter> metadataParameters,
    llvm::ArrayRef<support::RuntimeABIParameter> irBackedParameters,
    llvm::StringRef metadataSource,
    const RVVBinaryIntrinsicRoute &descriptor) {
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

llvm::Expected<const RVVBinaryFamilyRecord *>
resolveSelectedI64FamilyForPath(KernelOp kernel, const SelectedPath &path) {
  const RVVBinaryFamilyRecord *matchedFamily = nullptr;
  unsigned matches = 0;
  for (mlir::Operation &op : kernel.getBody().front()) {
    const RVVBinaryFamilyRecord *candidateFamily =
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

  if (matchedFamily)
    return matchedFamily;
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
    const RVVBinaryFamilyRecord &family) {
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
resolveRVVMicrokernelComponentCapacityElementCountMetadata(
    const TargetArtifactCandidate &candidate) {
  llvm::Expected<const SelectedPlanMetadataEntry *> metadata =
      findUniqueRVVMicrokernelSelectedPlanMetadataEntry(
          candidate, getRVVComponentCapacityElementCountMetadataName());
  if (!metadata)
    return metadata.takeError();

  if ((*metadata)->role != getRVVComponentCapacityElementCountMetadataRole())
    return makeMicrokernelError(
        candidate.kernel,
        llvm::Twine("selected RVV target artifact candidate @") +
            candidate.selectedVariant + " selected_plan_metadata '" +
            getRVVComponentCapacityElementCountMetadataName() +
            "' role must be '" +
            getRVVComponentCapacityElementCountMetadataRole() +
            "'");
  if ((*metadata)->note != getRVVComponentCapacityElementCountMetadataNote())
    return makeMicrokernelError(
        candidate.kernel,
        llvm::Twine("selected RVV target artifact candidate @") +
            candidate.selectedVariant + " selected_plan_metadata '" +
            getRVVComponentCapacityElementCountMetadataName() +
            "' note must be '" +
            getRVVComponentCapacityElementCountMetadataNote() +
            "'");

  std::int64_t value = 0;
  if (llvm::StringRef((*metadata)->value).getAsInteger(10, value) ||
      value <= 0 || value > 64)
    return makeMicrokernelError(
        candidate.kernel,
        llvm::Twine("selected RVV target artifact candidate @") +
            candidate.selectedVariant + " selected_plan_metadata '" +
            getRVVComponentCapacityElementCountMetadataName() +
            "' artifact-local component capacity must be an integer in the "
            "bounded smoke range [1, 64]");
  return value;
}

llvm::Error validateRVVMicrokernelComponentCapacityElementCountMetadata(
    const TargetArtifactCandidate &candidate,
    const RVVBinarySelectedConfigContract &contract) {
  llvm::Expected<std::int64_t> metadataElementCount =
      resolveRVVMicrokernelComponentCapacityElementCountMetadata(candidate);
  if (!metadataElementCount)
    return metadataElementCount.takeError();

  if (*metadataElementCount != contract.getComponentCapacityElementCount())
    return makeMicrokernelError(
        candidate.kernel,
        llvm::Twine("selected RVV target artifact candidate @") +
            candidate.selectedVariant + " selected_plan_metadata '" +
            getRVVComponentCapacityElementCountMetadataName() +
            "' artifact-local component capacity layer is stale; expected " +
            llvm::Twine(contract.getComponentCapacityElementCount()) +
            " from the selected config/runtime AVL contract but found " +
            llvm::Twine(*metadataElementCount));
  return llvm::Error::success();
}

llvm::Error validateRVVMicrokernelSelectedPlanMetadata(
    const TargetArtifactCandidate &candidate,
    const RVVBinarySelectedConfigContract &contract,
    bool includeSelectedConfigProfileMetadata = true) {
  llvm::SmallVector<RVVVectorShapeSelectedPlanMetadataDescriptor, 24> expected;
  appendRVVBinarySelectedVectorShapeMetadata(contract, expected);
  appendRVVBinaryRuntimeVLBoundarySelectedPlanMetadata(contract, expected);
  appendRVVBinaryFixedVectorSourceExtentSelectedPlanMetadata(contract,
                                                            expected);
  appendRVVBinaryDynamicRuntimeExtentSelectedPlanMetadata(contract, expected);
  if (contract.getFamily().dtype == RVVBinaryDTypeKind::I32 ||
      contract.getFamily().dtype == RVVBinaryDTypeKind::I64)
    appendRVVBinarySelectedTypedSourceMetadata(contract, expected);
  if (contract.getFamily().dtype == RVVBinaryDTypeKind::I32 ||
      contract.getFamily().dtype == RVVBinaryDTypeKind::I64)
    appendRVVBinaryEmitCRouteMetadata(contract, expected);
  if (includeSelectedConfigProfileMetadata)
    appendRVVBinarySelectedConfigProfileMetadata(contract, expected);

  for (const RVVVectorShapeSelectedPlanMetadataDescriptor &entry : expected)
    if (llvm::Error error =
            validateRVVMicrokernelSelectedPlanMetadataEntry(candidate, entry))
      return error;
  return validateRVVMicrokernelComponentCapacityElementCountMetadata(candidate,
                                                            contract);
}

llvm::Error validateRVVMicrokernelSelectedPlanMetadata(
    const TargetArtifactCandidate &candidate,
    const RVVBinaryIntrinsicRoute &descriptor) {
  llvm::Expected<llvm::StringRef> runtimeElementCountCName =
      resolveRVVMicrokernelRuntimeElementCountCName(candidate);
  if (!runtimeElementCountCName)
    return runtimeElementCountCName.takeError();

  llvm::Expected<std::int64_t> componentCapacityElementCount =
      resolveRVVMicrokernelComponentCapacityElementCountMetadata(candidate);
  if (!componentCapacityElementCount)
    return componentCapacityElementCount.takeError();

  llvm::Expected<RVVBinarySelectedConfigContract> selectedConfig =
      buildRVVBinarySelectedConfigContract(
          descriptor.family, *descriptor.shape, candidate.selectedVariant,
          candidate.role, *componentCapacityElementCount,
          *runtimeElementCountCName);
  if (!selectedConfig)
    return selectedConfig.takeError();
  return validateRVVMicrokernelSelectedPlanMetadata(
      candidate, *selectedConfig,
      /*includeSelectedConfigProfileMetadata=*/!candidate.kernel);
}

llvm::Error validateRVVMicrokernelSelectedSourceIdentityMetadata(
    const TargetArtifactCandidate &candidate,
    const RVVMicrokernelRecord &sourceAuthority) {
  if (sourceAuthority.selectedBinarySourceKind.empty())
    return llvm::Error::success();

  llvm::SmallVector<RVVVectorShapeSelectedPlanMetadataDescriptor, 2> expected;
  appendRVVBinarySelectedSourceIdentityMetadata(
      sourceAuthority.selectedConfigContract,
      sourceAuthority.selectedBinarySourceKind, expected);
  for (const RVVVectorShapeSelectedPlanMetadataDescriptor &entry : expected)
    if (llvm::Error error =
            validateRVVMicrokernelSelectedPlanMetadataEntry(candidate, entry))
      return error;
  return llvm::Error::success();
}

llvm::Expected<const SelectedPlanMetadataEntry *>
findUniqueSelectedPlanMetadataEntryForBodyMapping(
    KernelOp kernel, llvm::ArrayRef<SelectedPlanMetadataEntry> metadata,
    llvm::StringRef selectedVariant, llvm::StringRef name) {
  const SelectedPlanMetadataEntry *match = nullptr;
  unsigned count = 0;
  for (const SelectedPlanMetadataEntry &entry : metadata) {
    if (entry.name == name) {
      match = &entry;
      ++count;
    }
  }

  if (count == 0)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV EmitC body mapping for @") +
                    selectedVariant + " requires selected_plan_metadata '" +
                    name + "'");
  if (count > 1)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV EmitC body mapping for @") +
                    selectedVariant +
                    " has duplicate selected_plan_metadata '" + name + "'");
  return match;
}

llvm::Expected<std::string> requireSelectedPlanMetadataValueForBodyMapping(
    KernelOp kernel, llvm::ArrayRef<SelectedPlanMetadataEntry> metadata,
    llvm::StringRef selectedVariant, llvm::StringRef name,
    llvm::StringRef expectedRole, llvm::StringRef expectedNote,
    std::optional<llvm::StringRef> expectedValue,
    llvm::StringRef diagnosticSpelling) {
  llvm::Expected<const SelectedPlanMetadataEntry *> entry =
      findUniqueSelectedPlanMetadataEntryForBodyMapping(
          kernel, metadata, selectedVariant, name);
  if (!entry)
    return entry.takeError();

  if ((*entry)->role != expectedRole)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV EmitC body mapping for @") +
                    selectedVariant + " selected_plan_metadata '" + name +
                    "' role must be '" + expectedRole + "'");
  if ((*entry)->note != expectedNote)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV EmitC body mapping for @") +
                    selectedVariant + " selected_plan_metadata '" + name +
                    "' note must be '" + expectedNote + "'");
  if (expectedValue && (*entry)->value != *expectedValue)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV EmitC body mapping for @") +
                    selectedVariant + " selected_plan_metadata '" + name +
                    "' " + diagnosticSpelling + " must be '" +
                    *expectedValue + "'");
  return (*entry)->value;
}

llvm::Expected<RVVBinaryEmitCBodyMapping>
buildRVVEmitCBodyMappingFromSelectedPlanMetadata(
    KernelOp kernel, llvm::StringRef selectedVariant,
    const RVVBinarySelectedConfigContract &contract,
    llvm::ArrayRef<SelectedPlanMetadataEntry> selectedPlanMetadata) {
  if (selectedPlanMetadata.empty())
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV EmitC body mapping for @") +
                    selectedVariant +
                    " requires selected_plan_metadata from the plugin-owned "
                    "emission plan before production body emission");

  RVVBinaryEmitCBodyMapping mapping;
  llvm::StringRef routeRole = getRVVEmitCRouteMetadataRole();
  llvm::StringRef routeNote = getRVVEmitCRouteMetadataNote();

  llvm::Expected<std::string> routeKind =
      requireSelectedPlanMetadataValueForBodyMapping(
          kernel, selectedPlanMetadata, selectedVariant,
          getRVVEmitCRouteKindMetadataName(), routeRole, routeNote,
          getRVVEmitCRouteKindMetadataValue(), "EmitC route kind");
  if (!routeKind)
    return routeKind.takeError();
  mapping.routeKind = std::move(*routeKind);

  llvm::Expected<std::string> sourceAuthority =
      requireSelectedPlanMetadataValueForBodyMapping(
          kernel, selectedPlanMetadata, selectedVariant,
          getRVVEmitCSourceAuthorityMetadataName(), routeRole, routeNote,
          getRVVEmitCSourceAuthorityMetadataValue(),
          "EmitC source authority");
  if (!sourceAuthority)
    return sourceAuthority.takeError();
  mapping.sourceAuthority = std::move(*sourceAuthority);

  llvm::Expected<std::string> requiredHeader =
      requireSelectedPlanMetadataValueForBodyMapping(
          kernel, selectedPlanMetadata, selectedVariant,
          getRVVEmitCRequiredHeaderMetadataName(), routeRole, routeNote,
          getRVVEmitCRequiredHeaderMetadataValue(), "EmitC required header");
  if (!requiredHeader)
    return requiredHeader.takeError();
  mapping.requiredHeader = std::move(*requiredHeader);

  llvm::Expected<std::string> arithmeticIntrinsic =
      requireSelectedPlanMetadataValueForBodyMapping(
          kernel, selectedPlanMetadata, selectedVariant,
          getRVVEmitCArithmeticIntrinsicMetadataName(), routeRole,
          getRVVEmitCArithmeticIntrinsicMetadataNote(),
          contract.getArithmeticIntrinsicName(), "EmitC arithmetic intrinsic");
  if (!arithmeticIntrinsic)
    return arithmeticIntrinsic.takeError();
  mapping.arithmeticIntrinsicName = std::move(*arithmeticIntrinsic);

  if (!mapping.isValid())
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV EmitC body mapping for @") +
                    selectedVariant +
                    " must contain route kind, source authority, required "
                    "header, and arithmetic intrinsic");
  return mapping;
}

llvm::Expected<RVVBinaryEmitCBodyMapping>
buildRVVEmitCBodyMappingForRecord(KernelOp kernel,
                                  const RVVMicrokernelRecord &record,
                                  std::string &mappingSource) {
  if (!record.selectedPlanMetadata.empty() ||
      record.selectedBinarySourceKind == "frontend-lowering") {
    llvm::Expected<RVVBinaryEmitCBodyMapping> mapping =
        buildRVVEmitCBodyMappingFromSelectedPlanMetadata(
            kernel, record.variantSymbol, record.selectedConfigContract,
            record.selectedPlanMetadata);
    if (!mapping)
      return mapping.takeError();
    mappingSource = "selected_plan_metadata";
    return std::move(*mapping);
  }

  llvm::Expected<RVVBinaryEmitCBodyMapping> mapping =
      buildRVVBinaryEmitCBodyMappingFromSelectedConfig(
          record.selectedConfigContract);
  if (!mapping)
    return mapping.takeError();
  mappingSource = "selected_config_contract_compatibility";
  return std::move(*mapping);
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

bool hasAnyBoundaryBinarySourceIdentity(LoweringBoundaryOp boundary) {
  return boundary &&
         (boundary->hasAttr(kBoundarySelectedBinarySourceKindAttrName) ||
          boundary->hasAttr(kBoundarySelectedBinaryDTypeAttrName) ||
          boundary->hasAttr(kBoundarySelectedBinaryFamilyAttrName) ||
          boundary->hasAttr(kBoundarySelectedBinaryOperatorAttrName) ||
          boundary->hasAttr(kBoundarySelectedBinaryMicrokernelOpAttrName) ||
          boundary->hasAttr(kBoundaryEmitCSourceOpAttrName) ||
          boundary->hasAttr(kBoundaryEmitCLowerableOpInterfaceAttrName));
}

llvm::Error requireBoundarySourceIdentityAttr(KernelOp kernel,
                                             LoweringBoundaryOp boundary,
                                             llvm::StringRef attrName,
                                             llvm::StringRef expectedValue,
                                             llvm::StringRef description) {
  std::string value;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, boundary.getOperation(), attrName,
                                "tcrv_rvv.lowering_boundary", value))
    return error;
  if (value != expectedValue)
    return makeMicrokernelError(
        kernel, llvm::Twine("tcrv_rvv.lowering_boundary ") + attrName +
                    " '" + value + "' does not match selected RVV " +
                    description + " '" + expectedValue + "'");
  return llvm::Error::success();
}

llvm::Error validateBoundarySourceIdentityForRecord(
    KernelOp kernel, const SelectedPath &path, LoweringBoundaryOp boundary,
    const RVVBinaryIntrinsicRoute &descriptor,
    bool requireBoundarySourceIdentity,
    std::string *selectedBinarySourceKind = nullptr) {
  if (!hasAnyBoundaryBinarySourceIdentity(boundary)) {
    if (requireBoundarySourceIdentity)
      return makeMicrokernelError(
          kernel,
          llvm::Twine("tcrv_rvv.lowering_boundary for @") +
              getPathVariantSymbol(path) +
              " requires selected RVV binary source identity before target "
              "artifact export");
    return llvm::Error::success();
  }

  if (llvm::Error error = requireBoundarySourceIdentityAttr(
          kernel, boundary, kBoundarySelectedBinaryDTypeAttrName,
          descriptor.getDTypeID(), "typed source dtype"))
    return error;
  if (llvm::Error error = requireBoundarySourceIdentityAttr(
          kernel, boundary, kBoundarySelectedBinaryFamilyAttrName,
          descriptor.getArithmeticFamilyID(), "typed source family"))
    return error;
  if (llvm::Error error = requireBoundarySourceIdentityAttr(
          kernel, boundary, kBoundarySelectedBinaryOperatorAttrName,
          descriptor.family.arithmeticVerb, "typed source operator"))
    return error;
  if (llvm::Error error = requireBoundarySourceIdentityAttr(
          kernel, boundary, kBoundarySelectedBinaryMicrokernelOpAttrName,
          descriptor.getRVVMicrokernelOpName(), "typed microkernel op"))
    return error;
  if (llvm::Error error = requireBoundarySourceIdentityAttr(
          kernel, boundary, kBoundaryEmitCSourceOpAttrName,
          descriptor.getRVVOperationName(), "typed source op"))
    return error;
  if (llvm::Error error = requireBoundarySourceIdentityAttr(
          kernel, boundary, kBoundaryEmitCLowerableOpInterfaceAttrName,
          kEmitCLowerableOpInterfaceName,
          "generated EmitC lowerable interface"))
    return error;

  std::string sourceKind;
  if (llvm::Error error = requireSafeStringAttr(
          kernel, boundary.getOperation(),
          kBoundarySelectedBinarySourceKindAttrName,
          "tcrv_rvv.lowering_boundary", sourceKind))
    return error;
  if (sourceKind != "frontend-lowering" &&
      sourceKind != "default-i32-vadd-typed-body-materialization" &&
      sourceKind != "direct-typed-microkernel-body")
    return makeMicrokernelError(
        kernel,
        llvm::Twine("tcrv_rvv.lowering_boundary selected_binary_source_kind "
                    "'") +
            sourceKind +
            "' is not a recognized typed RVV binary source boundary for @" +
            getPathVariantSymbol(path));
  if (selectedBinarySourceKind)
    *selectedBinarySourceKind = std::move(sourceKind);

  return llvm::Error::success();
}

bool hasAnyMicrokernelBinarySourceIdentity(mlir::Operation *microkernel) {
  return microkernel &&
         (microkernel->hasAttr(kBoundarySelectedBinarySourceKindAttrName) ||
          microkernel->hasAttr(kBoundarySelectedBinaryDTypeAttrName) ||
          microkernel->hasAttr(kBoundarySelectedBinaryFamilyAttrName) ||
          microkernel->hasAttr(kBoundarySelectedBinaryOperatorAttrName) ||
          microkernel->hasAttr(kBoundarySelectedBinaryMicrokernelOpAttrName) ||
          microkernel->hasAttr(kBoundaryEmitCSourceOpAttrName) ||
          microkernel->hasAttr(kBoundaryEmitCLowerableOpInterfaceAttrName));
}

llvm::Error requireMicrokernelSourceIdentityAttr(
    KernelOp kernel, mlir::Operation *microkernel, llvm::StringRef attrName,
    llvm::StringRef expectedValue, llvm::StringRef description) {
  std::string value;
  if (llvm::Error error = requireSafeStringAttr(
          kernel, microkernel, attrName, microkernel->getName().getStringRef(),
          value))
    return error;
  if (value != expectedValue)
    return makeMicrokernelError(
        kernel, llvm::Twine(microkernel->getName().getStringRef()) + " " +
                    attrName + " '" + value +
                    "' does not match selected RVV " + description + " '" +
                    expectedValue + "'");
  return llvm::Error::success();
}

llvm::Error validateMicrokernelSourceIdentityForRecord(
    KernelOp kernel, mlir::Operation *microkernel,
    const RVVBinaryIntrinsicRoute &descriptor,
    llvm::StringRef selectedBinarySourceKind) {
  if (!microkernel)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV path requires matching ") +
                    descriptor.getRVVMicrokernelOpName());

  bool hasIdentity = hasAnyMicrokernelBinarySourceIdentity(microkernel);
  if (!hasIdentity) {
    if (selectedBinarySourceKind == "frontend-lowering")
      return makeMicrokernelError(
          kernel,
          llvm::Twine(descriptor.getRVVMicrokernelOpName()) +
              " requires op-owned selected RVV binary source identity for "
              "frontend-lowering before target artifact export");
    return llvm::Error::success();
  }

  if (selectedBinarySourceKind.empty())
    return makeMicrokernelError(
        kernel,
        llvm::Twine(descriptor.getRVVMicrokernelOpName()) +
            " carries op-owned selected RVV binary source identity without a "
            "matching selected lowering-boundary source identity");

  if (llvm::Error error = requireMicrokernelSourceIdentityAttr(
          kernel, microkernel, kBoundarySelectedBinarySourceKindAttrName,
          selectedBinarySourceKind, "typed source kind"))
    return error;
  if (llvm::Error error = requireMicrokernelSourceIdentityAttr(
          kernel, microkernel, kBoundarySelectedBinaryDTypeAttrName,
          descriptor.getDTypeID(), "typed source dtype"))
    return error;
  if (llvm::Error error = requireMicrokernelSourceIdentityAttr(
          kernel, microkernel, kBoundarySelectedBinaryFamilyAttrName,
          descriptor.getArithmeticFamilyID(), "typed source family"))
    return error;
  if (llvm::Error error = requireMicrokernelSourceIdentityAttr(
          kernel, microkernel, kBoundarySelectedBinaryOperatorAttrName,
          descriptor.family.arithmeticVerb, "typed source operator"))
    return error;
  if (llvm::Error error = requireMicrokernelSourceIdentityAttr(
          kernel, microkernel, kBoundarySelectedBinaryMicrokernelOpAttrName,
          descriptor.getRVVMicrokernelOpName(), "typed microkernel op"))
    return error;
  if (llvm::Error error = requireMicrokernelSourceIdentityAttr(
          kernel, microkernel, kBoundaryEmitCSourceOpAttrName,
          descriptor.getRVVOperationName(), "typed source op"))
    return error;
  return requireMicrokernelSourceIdentityAttr(
      kernel, microkernel, kBoundaryEmitCLowerableOpInterfaceAttrName,
      kEmitCLowerableOpInterfaceName, "generated EmitC lowerable interface");
}

bool selectedVariantRequiresBoundarySourceIdentity(const SelectedPath &path) {
  auto sourceKind =
      getPathVariantOperation(path)->getAttrOfType<mlir::StringAttr>(
          getRVVSelectedBinarySourceKindMetadataName());
  return sourceKind && sourceKind.getValue().trim() == "frontend-lowering";
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
    RVVBinaryDataflowEmissionPlan &dataflowPlan) {
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

  RVVBinaryIntrinsicRoute route =
      getI32BinaryIntrinsicRouteForMicrokernel(family, selectedConfig);
  llvm::SmallVector<support::RuntimeABIParameter, 4> callableABIParameters =
      route.getCallableRuntimeABIParameters();
  RVVBinaryMicrokernelBodyValidationRequest request;
  request.kernel = kernel;
  request.microkernel = microkernel;
  request.descriptor = route;
  request.selectedPolicy = expectedPolicy;
  request.activeRouteID = activeRouteID;
  request.callableABIParameters = callableABIParameters;
  request.expectedComponentCapacityElementCount = getSelectedVariantElementCount(path);
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
    RVVBinaryDataflowEmissionPlan &dataflowPlan) {
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
    const RVVBinaryIntrinsicRoute &descriptor,
    const RVVI32VectorShapeConfig &selectedConfig,
    llvm::StringRef activeRouteID,
    std::int64_t &elementCount, std::int64_t &controlPlaneSEW,
    std::string &controlPlaneLMUL, RVVIntrinsicConfig &intrinsicConfig,
    RVVBinaryDataflowEmissionPlan &dataflowPlan) {
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
  request.expectedComponentCapacityElementCount = getSelectedVariantElementCount(path);
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
    const RVVBinaryIntrinsicRoute &descriptor,
    const RVVI32VectorShapeConfig &selectedConfig,
    llvm::StringRef activeRouteID,
    std::int64_t &elementCount, std::int64_t &controlPlaneSEW,
    std::string &controlPlaneLMUL, RVVIntrinsicConfig &intrinsicConfig,
    RVVBinaryDataflowEmissionPlan &dataflowPlan) {
  unsigned matches = 0;
  for (mlir::Operation &op : kernel.getBody().front()) {
    const RVVBinaryFamilyRecord *microkernelFamily =
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
    KernelOp kernel, const RVVBinaryIntrinsicRoute &descriptor,
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
    const RVVBinaryIntrinsicRoute &descriptor,
    llvm::SmallVectorImpl<support::RuntimeABIParameter> &parameters,
    llvm::SmallVectorImpl<MemWindowOp> &bufferWindows,
    RuntimeParamOp &runtimeElementCountParam,
    llvm::SmallVectorImpl<SelectedPlanMetadataEntry> *selectedPlanMetadata =
        nullptr) {
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
  if (selectedPlanMetadata) {
    selectedPlanMetadata->clear();
    if (llvm::Error error = collectSelectedPlanMetadataEntries(
            kernel, diagnostic, *selectedPlanMetadata))
      return error;
  }

  return llvm::Error::success();
}

llvm::Error validateRVVBinaryCandidateRuntimeABIMirrorsIR(
    const TargetArtifactCandidate &candidate,
    const RVVBinaryIntrinsicRoute &descriptor) {
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

  const RVVBinaryFamilyRecord *registeredFamily =
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
      record.role, record.elementCount, (*runtimeElementCount)->cName,
      /*dispatchAvailabilityGuardCName=*/"rvv_available",
      record.fixedSourceExtent, record.dynamicRuntimeExtent);
}

llvm::Error attachFrontendRuntimeExtentContracts(KernelOp kernel,
                                                 RVVMicrokernelRecord &record) {
  llvm::Expected<std::optional<support::FixedVectorSourceExtentContract>>
      sourceExtent = support::getFixedVectorSourceExtentContract(
          kernel, record.runtimeElementCountParam);
  if (!sourceExtent)
    return sourceExtent.takeError();
  record.fixedSourceExtent = std::move(*sourceExtent);
  llvm::Expected<std::optional<support::DynamicVectorRuntimeExtentContract>>
      runtimeExtent = support::getDynamicVectorRuntimeExtentContract(
          kernel, record.runtimeElementCountParam);
  if (!runtimeExtent)
    return runtimeExtent.takeError();
  record.dynamicRuntimeExtent = std::move(*runtimeExtent);
  if (record.fixedSourceExtent && record.dynamicRuntimeExtent)
    return makeMicrokernelError(
        kernel,
        "fixed source extent and dynamic runtime extent contracts are "
        "mutually exclusive before runtime AVL/VL artifact export");

  if (record.fixedSourceExtent &&
      record.fixedSourceExtent->sourceVectorExtent != record.elementCount)
    return makeMicrokernelError(
        kernel,
        llvm::Twine("fixed vector source extent ") +
            llvm::Twine(record.fixedSourceExtent->sourceVectorExtent) +
            " must match selected artifact-local component capacity " +
            llvm::Twine(record.elementCount) +
            " before runtime AVL/VL artifact export");
  return llvm::Error::success();
}

llvm::Error validateSelectedConfigEmissionMatchesBody(
    KernelOp kernel, const RVVBinarySelectedConfigContract &contract,
    const RVVBinarySelectedConfigEmissionView &selectedConfigEmission,
    const RVVIntrinsicConfig &bodyIntrinsicConfig) {
  if (!selectedConfigEmission.isValid())
    return makeMicrokernelError(
        kernel,
        "selected RVV config emission authority requires complete vector "
        "type, suffix, policy, and intrinsic spelling fields");

  auto failMismatch = [&](llvm::StringRef field, llvm::StringRef expected,
                          llvm::StringRef observed) -> llvm::Error {
    return makeMicrokernelError(
        kernel,
        llvm::Twine("selected RVV config emission authority field '") +
            field + "' expected '" + expected +
            "' from RVVBinarySelectedConfigContract but body verifier "
            "observed '" +
            observed + "'");
  };

  if (selectedConfigEmission.sew != bodyIntrinsicConfig.sew)
    return makeMicrokernelError(
        kernel,
        llvm::Twine("selected RVV config emission authority field 'sew' "
                    "expected ") +
            llvm::Twine(selectedConfigEmission.sew) +
            " from RVVBinarySelectedConfigContract but body verifier "
            "observed " +
            llvm::Twine(bodyIntrinsicConfig.sew));
  if (selectedConfigEmission.lmul != bodyIntrinsicConfig.lmul)
    return failMismatch("lmul", selectedConfigEmission.lmul,
                        bodyIntrinsicConfig.lmul);
  if (selectedConfigEmission.vectorType != bodyIntrinsicConfig.vectorType)
    return failMismatch("vector_type", selectedConfigEmission.vectorType,
                        bodyIntrinsicConfig.vectorType);
  if (selectedConfigEmission.vectorSuffix != bodyIntrinsicConfig.vectorSuffix)
    return failMismatch("vector_suffix", selectedConfigEmission.vectorSuffix,
                        bodyIntrinsicConfig.vectorSuffix);
  if (selectedConfigEmission.setvlSuffix != bodyIntrinsicConfig.setvlSuffix)
    return failMismatch("setvl_suffix", selectedConfigEmission.setvlSuffix,
                        bodyIntrinsicConfig.setvlSuffix);
  if (selectedConfigEmission.setvlIntrinsicName !=
      bodyIntrinsicConfig.setvlIntrinsicName)
    return failMismatch("setvl_intrinsic",
                        selectedConfigEmission.setvlIntrinsicName,
                        bodyIntrinsicConfig.setvlIntrinsicName);
  if (selectedConfigEmission.loadIntrinsicName !=
      bodyIntrinsicConfig.loadIntrinsicName)
    return failMismatch("load_intrinsic",
                        selectedConfigEmission.loadIntrinsicName,
                        bodyIntrinsicConfig.loadIntrinsicName);
  if (selectedConfigEmission.arithmeticIntrinsicName !=
      bodyIntrinsicConfig.arithmeticIntrinsicName)
    return failMismatch("arithmetic_intrinsic",
                        selectedConfigEmission.arithmeticIntrinsicName,
                        bodyIntrinsicConfig.arithmeticIntrinsicName);
  if (selectedConfigEmission.storeIntrinsicName !=
      bodyIntrinsicConfig.storeIntrinsicName)
    return failMismatch("store_intrinsic",
                        selectedConfigEmission.storeIntrinsicName,
                        bodyIntrinsicConfig.storeIntrinsicName);
  if (selectedConfigEmission.tailPolicy != bodyIntrinsicConfig.tailPolicy)
    return failMismatch("tail_policy", selectedConfigEmission.tailPolicy,
                        bodyIntrinsicConfig.tailPolicy);
  if (selectedConfigEmission.maskPolicy != bodyIntrinsicConfig.maskPolicy)
    return failMismatch("mask_policy", selectedConfigEmission.maskPolicy,
                        bodyIntrinsicConfig.maskPolicy);

  if (contract.getVectorType() != selectedConfigEmission.vectorType ||
      contract.getVectorSuffix() != selectedConfigEmission.vectorSuffix ||
      contract.getSetVLSuffix() != selectedConfigEmission.setvlSuffix)
    return makeMicrokernelError(
        kernel,
        "selected RVV config emission authority must be derived from the "
        "selected config contract, not artifact-local or body-only "
        "metadata");
  return llvm::Error::success();
}

llvm::Expected<RVVMicrokernelRecord>
buildMicrokernelRecord(KernelOp kernel, const SelectedPath &path,
                       const TargetCapabilitySet &capabilities,
                       const llvm::StringSet<> &selectedRVVPathKeys,
                       llvm::StringRef activeRouteID,
                       bool requireBoundarySourceIdentity = false) {
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
  const bool effectiveRequireBoundarySourceIdentity =
      requireBoundarySourceIdentity ||
      selectedVariantRequiresBoundarySourceIdentity(path);

  llvm::Expected<const RVVBinaryFamilyRecord *> rvvBinaryFamily =
      resolveSelectedI64FamilyForPath(kernel, path);
  if (!rvvBinaryFamily)
    return rvvBinaryFamily.takeError();

  if (*rvvBinaryFamily) {
    RVVBinaryIntrinsicRoute descriptor =
        getRVVBinaryIntrinsicRoute(**rvvBinaryFamily, **selectedConfig);
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
    RVVBinaryDataflowEmissionPlan dataflowPlan;
    if (llvm::Error error = findAndValidateI64Microkernel(
            kernel, path, selectedRVVPathKeys, *selectedMarch, selectedMABI,
            policy, i64Microkernel, descriptor, **selectedConfig,
            activeRouteID, elementCount, controlPlaneSEW, controlPlaneLMUL,
            intrinsicConfig, dataflowPlan))
      return std::move(error);

    llvm::SmallVector<support::RuntimeABIParameter, 4> runtimeABIParameters;
    llvm::SmallVector<SelectedPlanMetadataEntry, 24> selectedPlanMetadata;
    llvm::SmallVector<MemWindowOp, 3> bufferWindows;
    RuntimeParamOp runtimeElementCountParam;
    if (llvm::Error error = resolveRVVBinaryRuntimeABIParametersForPath(
            kernel, path, descriptor, runtimeABIParameters, bufferWindows,
            runtimeElementCountParam, &selectedPlanMetadata))
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
    record.selectedPlanMetadata = std::move(selectedPlanMetadata);
    record.bufferWindows = std::move(bufferWindows);
    record.runtimeElementCountParam = runtimeElementCountParam;
    record.dataflowPlan = std::move(dataflowPlan);
    record.intrinsicConfig = std::move(intrinsicConfig);
    record.selectedShape = *selectedConfig;
    record.elementCount = elementCount;
    record.controlPlaneSEW = controlPlaneSEW;
    record.controlPlaneLMUL = std::move(controlPlaneLMUL);
    if (llvm::Error error =
            attachFrontendRuntimeExtentContracts(kernel, record))
      return std::move(error);
    llvm::Expected<RVVBinarySelectedConfigContract> selectedConfigContract =
        buildRVVMicrokernelSelectedConfigContract(kernel, record);
    if (!selectedConfigContract)
      return selectedConfigContract.takeError();
    record.selectedConfigContract = std::move(*selectedConfigContract);
    llvm::Expected<RVVBinarySelectedConfigEmissionView> selectedEmission =
        buildRVVBinarySelectedConfigEmissionView(
            record.selectedConfigContract);
    if (!selectedEmission)
      return selectedEmission.takeError();
    if (llvm::Error error = validateSelectedConfigEmissionMatchesBody(
            kernel, record.selectedConfigContract, *selectedEmission,
            record.intrinsicConfig))
      return std::move(error);
    record.selectedConfigEmission = std::move(*selectedEmission);
    if (llvm::Error error =
            validateBoundarySourceIdentityForRecord(
                kernel, path, boundary, record.descriptor,
                effectiveRequireBoundarySourceIdentity,
                &record.selectedBinarySourceKind))
      return std::move(error);
    if (llvm::Error error = validateMicrokernelSourceIdentityForRecord(
            kernel, i64Microkernel, record.descriptor,
            record.selectedBinarySourceKind))
      return std::move(error);
    llvm::Expected<RVVBinaryEmitCBodyMapping> emitcBodyMapping =
        buildRVVEmitCBodyMappingForRecord(kernel, record,
                                          record.emitcBodyMappingSource);
    if (!emitcBodyMapping)
      return emitcBodyMapping.takeError();
    record.emitcBodyMapping = std::move(*emitcBodyMapping);
    return record;
  }

  mlir::Operation *microkernel = nullptr;
  const RVVI32MicrokernelFamilySpec *microkernelFamily = nullptr;
  std::int64_t elementCount = 0;
  std::int64_t controlPlaneSEW = 0;
  std::string controlPlaneLMUL;
  RVVIntrinsicConfig intrinsicConfig;
  RVVBinaryDataflowEmissionPlan dataflowPlan;
  if (llvm::Error error = findAndValidateMicrokernel(
          kernel, path, selectedRVVPathKeys, *selectedMarch, selectedMABI,
          policy, microkernel, microkernelFamily, **selectedConfig,
          activeRouteID, elementCount, controlPlaneSEW, controlPlaneLMUL,
          intrinsicConfig, dataflowPlan))
    return std::move(error);

  RVVMicrokernelRecord record;
  record.family = microkernelFamily;
  record.descriptor =
      getI32BinaryIntrinsicRouteForMicrokernel(*microkernelFamily,
                                              **selectedConfig);
  llvm::SmallVector<support::RuntimeABIParameter, 4> runtimeABIParameters;
  llvm::SmallVector<SelectedPlanMetadataEntry, 24> selectedPlanMetadata;
  llvm::SmallVector<MemWindowOp, 3> bufferWindows;
  RuntimeParamOp runtimeElementCountParam;
  if (llvm::Error error = resolveRVVBinaryRuntimeABIParametersForPath(
          kernel, path, record.descriptor, runtimeABIParameters, bufferWindows,
          runtimeElementCountParam, &selectedPlanMetadata))
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
  record.selectedPlanMetadata = std::move(selectedPlanMetadata);
  record.bufferWindows = std::move(bufferWindows);
  record.runtimeElementCountParam = runtimeElementCountParam;
  record.dataflowPlan = std::move(dataflowPlan);
  record.intrinsicConfig = std::move(intrinsicConfig);
  record.selectedShape = *selectedConfig;
  record.elementCount = elementCount;
  record.controlPlaneSEW = controlPlaneSEW;
  record.controlPlaneLMUL = std::move(controlPlaneLMUL);
  if (llvm::Error error =
          attachFrontendRuntimeExtentContracts(kernel, record))
    return std::move(error);
  llvm::Expected<RVVBinarySelectedConfigContract> selectedConfigContract =
      buildRVVMicrokernelSelectedConfigContract(kernel, record);
  if (!selectedConfigContract)
    return selectedConfigContract.takeError();
  record.selectedConfigContract = std::move(*selectedConfigContract);
  llvm::Expected<RVVBinarySelectedConfigEmissionView> selectedEmission =
      buildRVVBinarySelectedConfigEmissionView(record.selectedConfigContract);
  if (!selectedEmission)
    return selectedEmission.takeError();
  if (llvm::Error error = validateSelectedConfigEmissionMatchesBody(
          kernel, record.selectedConfigContract, *selectedEmission,
          record.intrinsicConfig))
    return std::move(error);
  record.selectedConfigEmission = std::move(*selectedEmission);
  if (llvm::Error error =
          validateBoundarySourceIdentityForRecord(
              kernel, path, boundary, record.descriptor,
              effectiveRequireBoundarySourceIdentity,
              &record.selectedBinarySourceKind))
    return std::move(error);
  if (llvm::Error error = validateMicrokernelSourceIdentityForRecord(
          kernel, microkernel, record.descriptor, record.selectedBinarySourceKind))
    return std::move(error);
  llvm::Expected<RVVBinaryEmitCBodyMapping> emitcBodyMapping =
      buildRVVEmitCBodyMappingForRecord(kernel, record,
                                        record.emitcBodyMappingSource);
  if (!emitcBodyMapping)
    return emitcBodyMapping.takeError();
  record.emitcBodyMapping = std::move(*emitcBodyMapping);
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
    mlir::ModuleOp module, const RVVBinaryFamilyRecord &expectedFamily,
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

llvm::Expected<RVVMicrokernelRecord> buildKernelRecordForRVVBinaryFamily(
    KernelOp kernel, const RVVBinaryFamilyRecord &expectedFamily,
    llvm::StringRef selectedVariant, llvm::StringRef role,
    llvm::StringRef routeID, bool requireBoundarySourceIdentity = false) {
  if (!kernel)
    return makeModuleMicrokernelError(
        "requires a tcrv.exec.kernel operation for selected RVV source "
        "authority resolution");

  llvm::StringMap<VariantOp> directVariants;
  llvm::StringMap<mlir::Operation *> directSymbols;
  collectDirectKernelSymbols(kernel, directVariants, directSymbols);

  llvm::SmallVector<SelectedPath, 4> selectedPaths;
  if (llvm::Error error =
          collectSelectedPaths(kernel, directVariants, directSymbols,
                               selectedPaths))
    return std::move(error);

  llvm::SmallVector<SelectedPath, 2> matchingRVVPaths;
  for (const SelectedPath &path : selectedPaths) {
    const bool matchesSelectedRoute =
        getPathVariantSymbol(path) == selectedVariant &&
        llvm::StringRef(path.role) == role;
    if (!isRVVPluginSelectedPath(path)) {
      if (matchesSelectedRoute && hasRVVLikeOrigin(path))
        return makeMicrokernelError(
            kernel, llvm::Twine("selected RVV-like path @") +
                        getPathVariantSymbol(path) +
                        " uses unknown origin; RVV microkernel source "
                        "authority only accepts registered origin "
                        "'rvv-plugin'");
      continue;
    }
    if (matchesSelectedRoute)
      matchingRVVPaths.push_back(path);
  }

  if (matchingRVVPaths.empty())
    return makeMicrokernelError(
        kernel, llvm::Twine("route '") + routeID +
                    "' requires selected RVV source authority path @" +
                    selectedVariant + " with role '" + role + "'");
  if (matchingRVVPaths.size() != 1)
    return makeMicrokernelError(
        kernel, llvm::Twine("route '") + routeID +
                    "' requires exactly one selected RVV source authority "
                    "path @" +
                    selectedVariant + " with role '" + role + "'");

  llvm::StringSet<> selectedRVVPathKeys;
  selectedRVVPathKeys.insert(
      makePathKey(getPathVariantSymbol(matchingRVVPaths.front()),
                  matchingRVVPaths.front().role));

  llvm::Expected<TargetCapabilitySet> capabilities =
      TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities)
    return capabilities.takeError();

  llvm::Expected<RVVMicrokernelRecord> record = buildMicrokernelRecord(
      kernel, matchingRVVPaths.front(), *capabilities, selectedRVVPathKeys,
      routeID, requireBoundarySourceIdentity);
  if (!record)
    return record.takeError();

  if (!isSameRVVBinaryFamily(record->descriptor.family, expectedFamily))
    return makeMicrokernelError(
        kernel, llvm::Twine("route '") + routeID + "' requires " +
                    expectedFamily.microkernelOpName +
                    " but the selected RVV source authority path @" +
                    selectedVariant + " produced " +
                    record->descriptor.getRVVMicrokernelOpName());
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
getDataflowStepOpName(const RVVBinaryDataflowStep &step,
                      const RVVBinaryIntrinsicRoute &descriptor) {
  if (!step.sourceOpName.empty())
    return step.sourceOpName;
  switch (step.kind) {
  case RVVBinaryDataflowStepKind::Load:
    return (llvm::Twine("tcrv_rvv.") + descriptor.getDTypeID() + "_load")
        .str();
  case RVVBinaryDataflowStepKind::Add:
    return descriptor.getRVVOperationName().str();
  case RVVBinaryDataflowStepKind::Sub:
    return descriptor.getRVVOperationName().str();
  case RVVBinaryDataflowStepKind::Mul:
    return descriptor.getRVVOperationName().str();
  case RVVBinaryDataflowStepKind::Store:
    return (llvm::Twine("tcrv_rvv.") + descriptor.getDTypeID() + "_store")
        .str();
  }
  return "unknown";
}

llvm::StringRef
getDataflowValueCName(RVVBinaryDataflowValue value,
                      const RVVBinaryIntrinsicRoute &descriptor) {
  switch (value) {
  case RVVBinaryDataflowValue::LHSVector:
    return "lhs_vec";
  case RVVBinaryDataflowValue::RHSVector:
    return "rhs_vec";
  case RVVBinaryDataflowValue::ResultVector:
    return descriptor.family.resultCName;
  case RVVBinaryDataflowValue::None:
    return "";
  }
  return "";
}

llvm::Expected<std::string> getEmitCCallOpaqueCalleeForStep(
    const RVVBinaryDataflowStep &step,
    const RVVBinarySelectedConfigEmissionView &selectedConfigEmission,
    const RVVBinaryEmitCBodyMapping &emitcBodyMapping,
    const RVVBinaryIntrinsicRoute &descriptor) {
  switch (step.kind) {
  case RVVBinaryDataflowStepKind::Load:
    return selectedConfigEmission.loadIntrinsicName;
  case RVVBinaryDataflowStepKind::Add:
  case RVVBinaryDataflowStepKind::Sub:
  case RVVBinaryDataflowStepKind::Mul: {
    if (step.sourceOpName.empty() || step.sourceOpRole != "compute" ||
        step.sourceOpInterface != "TCRVEmitCLowerableOpInterface")
      return makeModuleMicrokernelError(
          "RVV family-op to EmitC route requires every compute step to carry "
          "typed source-op and generated op-interface provenance before "
          "choosing an intrinsic callee");
    const RVVBinaryFamilyRecord *sourceFamily =
        lookupRVVBinaryFamilyRegistrationByRVVOperationName(
            step.sourceOpName);
    if (!sourceFamily)
      return makeModuleMicrokernelError(
          llvm::Twine("RVV family-op to EmitC route cannot map typed source "
                      "op '") +
          step.sourceOpName + "' to a registered finite binary family");
    if (!isSameRVVBinaryFamily(*sourceFamily, descriptor.family))
      return makeModuleMicrokernelError(
          llvm::Twine("RVV family-op to EmitC route compute source op '") +
          step.sourceOpName + "' names family '" + sourceFamily->familyID +
          "' but the selected RVV record is '" +
          descriptor.getArithmeticFamilyID() +
          "'; typed source op and selected record must agree before "
          "emitc.call_opaque callee selection");
    std::string expectedIntrinsic =
        (llvm::Twine(sourceFamily->arithmeticIntrinsicPrefix) +
         selectedConfigEmission.vectorSuffix)
            .str();
    if (expectedIntrinsic != selectedConfigEmission.arithmeticIntrinsicName)
      return makeModuleMicrokernelError(
          "selected RVV config emission authority arithmetic intrinsic name "
          "must equal the family-owned suffix-free arithmetic prefix plus the "
          "selected vector suffix before EmitC route construction");
    if (emitcBodyMapping.routeKind != getRVVEmitCRouteKindMetadataValue())
      return makeModuleMicrokernelError(
          "selected RVV EmitC body mapping route kind must match the "
          "plugin-owned selected emission-plan metadata before "
          "emitc.call_opaque callee selection");
    if (emitcBodyMapping.sourceAuthority !=
        getRVVEmitCSourceAuthorityMetadataValue())
      return makeModuleMicrokernelError(
          "selected RVV EmitC body mapping source authority must match the "
          "common EmitC source authority before emitc.call_opaque callee "
          "selection");
    if (emitcBodyMapping.requiredHeader !=
        getRVVEmitCRequiredHeaderMetadataValue())
      return makeModuleMicrokernelError(
          "selected RVV EmitC body mapping required header must remain "
          "riscv_vector.h before RVV intrinsic body emission");
    if (emitcBodyMapping.arithmeticIntrinsicName != expectedIntrinsic)
      return makeModuleMicrokernelError(
          "selected RVV EmitC body mapping arithmetic intrinsic must match "
          "the typed RVV op-family plus selected vector suffix before "
          "emitc.call_opaque callee selection");
    return emitcBodyMapping.arithmeticIntrinsicName;
  }
  case RVVBinaryDataflowStepKind::Store:
    return selectedConfigEmission.storeIntrinsicName;
  }
  return makeModuleMicrokernelError(
      "RVV family-op to EmitC route saw an unknown dataflow step kind");
}

llvm::Expected<BinarySelfCheckArithmeticKind>
getSelfCheckArithmeticFromDataflowPlan(
    const RVVBinaryDataflowEmissionPlan &dataflowPlan,
    llvm::StringRef context) {
  std::optional<BinarySelfCheckArithmeticKind> arithmetic;
  for (const RVVBinaryDataflowStep &step : dataflowPlan.steps) {
    std::optional<BinarySelfCheckArithmeticKind> candidate;
    switch (step.kind) {
    case RVVBinaryDataflowStepKind::Load:
    case RVVBinaryDataflowStepKind::Store:
      continue;
    case RVVBinaryDataflowStepKind::Add:
      candidate = BinarySelfCheckArithmeticKind::Add;
      break;
    case RVVBinaryDataflowStepKind::Sub:
      candidate = BinarySelfCheckArithmeticKind::Sub;
      break;
    case RVVBinaryDataflowStepKind::Mul:
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
      const RVVBinaryIntrinsicRoute &descriptor,
      const RVVBinarySelectedConfigEmissionView &selectedConfigEmission,
      const RVVBinaryEmitCBodyMapping &emitcBodyMapping,
      const RVVBinaryDataflowEmissionPlan &dataflowPlan,
      llvm::ArrayRef<support::RuntimeABIParameter> runtimeABIParameters,
      RVVRuntimeLengthContract runtimeLength, llvm::StringRef loopIndexName)
      : descriptor(descriptor), selectedConfigEmission(selectedConfigEmission),
        emitcBodyMapping(emitcBodyMapping), dataflowPlan(dataflowPlan),
        runtimeLength(std::move(runtimeLength)),
        loopIndexName(loopIndexName.str()) {
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
    if (llvm::Error error = validateRVVRuntimeLengthContract(runtimeLength))
      return std::move(error);
    if (runtimeN->cName != runtimeLength.getRuntimeElementCountCName())
      return makeModuleMicrokernelError(
          llvm::Twine("RVV family-op to EmitC route runtime-length contract "
                      "requires runtime element-count C name '") +
          runtimeLength.getRuntimeElementCountCName() +
          "' but the IR-backed callable ABI parameter uses '" +
          runtimeN->cName + "'");

    TCRVEmitCLowerableRoute route(
        descriptor.getRVVRouteID(), emitcBodyMapping.routeKind);
    route.addHeader("stddef.h");
    route.addHeader("stdint.h");
    route.addHeader(emitcBodyMapping.requiredHeader);
    route.addTypeMapping("!tcrv_rvv.vl", "size_t");
    route.addTypeMapping(
        (llvm::Twine("!tcrv_rvv.") + descriptor.getShapeID()).str(),
        selectedConfigEmission.vectorType);
    for (const support::RuntimeABIParameter &parameter : runtimeABIParameters)
      route.addABIValueMapping(parameter, parameter.cName);

    TCRVEmitCCallOpaqueStep setvlStep;
    setvlStep.sourceOp.opName = "tcrv_rvv.setvl";
    setvlStep.sourceOp.role = "runtime-avl-to-vl";
    setvlStep.callee = selectedConfigEmission.setvlIntrinsicName;
    setvlStep.operands.push_back(
        {runtimeLength.formatRemainingAVLOperandExpression(loopIndexName),
         runtimeN->cType});
    setvlStep.result = TCRVEmitCCallOpaqueResult{"vl", "size_t"};
    route.addCallOpaqueStep(std::move(setvlStep));

    for (const RVVBinaryDataflowStep &step : dataflowPlan.steps) {
      TCRVEmitCCallOpaqueStep emitcStep;
      emitcStep.sourceOp.opName = getDataflowStepOpName(step, descriptor);
      emitcStep.sourceOp.role = getRouteSourceRole(step).str();
      emitcStep.sourceOp.opInterface = step.sourceOpInterface;
      llvm::Expected<std::string> callee =
          getEmitCCallOpaqueCalleeForStep(step, selectedConfigEmission,
                                          emitcBodyMapping, descriptor);
      if (!callee)
        return callee.takeError();
      emitcStep.callee = std::move(*callee);
      if (emitcStep.sourceOp.opName.empty() || emitcStep.callee.empty())
        return makeModuleMicrokernelError(
            "RVV family-op to EmitC route requires every dataflow step to map "
            "to a bounded emitc.call_opaque callee");

      switch (step.kind) {
      case RVVBinaryDataflowStepKind::Load: {
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
            selectedConfigEmission.vectorType};
        break;
      }
      case RVVBinaryDataflowStepKind::Add:
      case RVVBinaryDataflowStepKind::Sub:
      case RVVBinaryDataflowStepKind::Mul:
        emitcStep.operands.push_back(
            {getDataflowValueCName(step.lhs, descriptor).str(),
             selectedConfigEmission.vectorType});
        emitcStep.operands.push_back(
            {getDataflowValueCName(step.rhs, descriptor).str(),
             selectedConfigEmission.vectorType});
        emitcStep.operands.push_back({"vl", "size_t"});
        emitcStep.result = TCRVEmitCCallOpaqueResult{
            getDataflowValueCName(step.result, descriptor).str(),
            selectedConfigEmission.vectorType};
        break;
      case RVVBinaryDataflowStepKind::Store: {
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
             selectedConfigEmission.vectorType});
        emitcStep.operands.push_back({"vl", "size_t"});
        break;
      }
      }
      route.addCallOpaqueStep(std::move(emitcStep));
    }

    return route;
  }

private:
  static llvm::StringRef getRouteSourceRole(const RVVBinaryDataflowStep &step) {
    if (!step.sourceOpRole.empty())
      return step.sourceOpRole;
    switch (step.kind) {
    case RVVBinaryDataflowStepKind::Load:
      return "buffer-load";
    case RVVBinaryDataflowStepKind::Add:
    case RVVBinaryDataflowStepKind::Sub:
    case RVVBinaryDataflowStepKind::Mul:
      return "compute";
    case RVVBinaryDataflowStepKind::Store:
      return "buffer-store";
    }
    return "unknown";
  }

  RVVBinaryIntrinsicRoute descriptor;
  RVVBinarySelectedConfigEmissionView selectedConfigEmission;
  RVVBinaryEmitCBodyMapping emitcBodyMapping;
  RVVBinaryDataflowEmissionPlan dataflowPlan;
  RVVRuntimeLengthContract runtimeLength;
  std::string loopIndexName;
  llvm::SmallVector<support::RuntimeABIParameter, 4> runtimeABIParameters;
};

llvm::Expected<TCRVLowerToEmitCSourceResult> lowerRVVBinaryToEmitCSource(
    const RVVBinaryIntrinsicRoute &descriptor,
    const RVVBinarySelectedConfigEmissionView &selectedConfigEmission,
    const RVVBinaryEmitCBodyMapping &emitcBodyMapping,
    const RVVBinaryDataflowEmissionPlan &dataflowPlan,
    llvm::ArrayRef<support::RuntimeABIParameter> runtimeABIParameters,
    const RVVRuntimeLengthContract &runtimeLength,
    llvm::StringRef functionName,
    std::int64_t fixedRuntimeElementCount = 0) {
  constexpr llvm::StringLiteral kLoopIndexName("offset");
  RVVBinaryEmitCLowerable lowerable(descriptor, selectedConfigEmission,
                                    emitcBodyMapping, dataflowPlan,
                                    runtimeABIParameters, runtimeLength,
                                    kLoopIndexName);
  TCRVLowerToEmitCSourceOptions options;
  options.sourceAuthorityOptions.functionName = functionName.str();
  options.sourceAuthorityOptions.loopIndexName = kLoopIndexName.str();
  options.sourceAuthorityOptions.requireInterfaceBackedCompute = true;
  options.sourceAuthorityOptions.fixedRuntimeElementCount =
      fixedRuntimeElementCount;
  return lowerTCRVEmitCLowerableToEmitCSource(lowerable, options);
}

void printDataflowPlanMetadata(
    llvm::raw_ostream &os,
    const RVVBinaryDataflowEmissionPlan &dataflowPlan,
    const RVVBinaryIntrinsicRoute &descriptor) {
  for (auto [index, step] : llvm::enumerate(dataflowPlan.steps)) {
    os << "/* dataflow_emission_step[" << index
       << "]: op=" << getDataflowStepOpName(step, descriptor);
    switch (step.kind) {
    case RVVBinaryDataflowStepKind::Load:
      os << ", role="
         << support::stringifyRuntimeABIParameterRole(step.bufferRole)
         << ", result=" << getDataflowValueCName(step.result, descriptor);
      break;
    case RVVBinaryDataflowStepKind::Add:
    case RVVBinaryDataflowStepKind::Sub:
    case RVVBinaryDataflowStepKind::Mul:
      os << ", lhs=" << getDataflowValueCName(step.lhs, descriptor)
         << ", rhs=" << getDataflowValueCName(step.rhs, descriptor)
         << ", result=" << getDataflowValueCName(step.result, descriptor);
      if (!step.sourceOpInterface.empty())
        os << ", interface=" << step.sourceOpInterface
           << ", source_role=" << step.sourceOpRole;
      break;
    case RVVBinaryDataflowStepKind::Store:
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
  os << "/* deleted_direct_c_source_route: historical EmitC source text is not "
        "a registered RVV artifact authority */\n";
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
    for (auto [operandIndex, operand] : llvm::enumerate(step.operands))
      os << "/* emitc.call_opaque_operand[" << index << "]["
         << operandIndex << "]: expression=" << operand.expression
         << ", c_type=" << operand.cType << " */\n";
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

llvm::Error printRuntimeABIInvocationContract(
    llvm::raw_ostream &os, llvm::StringRef sourceOwner,
    llvm::StringRef callableSymbol, llvm::StringRef runtimeABIKind,
    llvm::StringRef runtimeABIName, llvm::StringRef runtimeGlueRole,
    llvm::ArrayRef<support::RuntimeABIParameter> parameters,
    llvm::StringRef runtimeElementCountCName, llvm::StringRef productionOwner,
    llvm::StringRef familyID) {
  llvm::Expected<support::RuntimeABIInvocationContract> contract =
      support::buildRuntimeABIInvocationContract(
          /*kernel=*/nullptr, familyID, parameters, sourceOwner,
          callableSymbol, runtimeABIKind, runtimeABIName, runtimeGlueRole,
          runtimeElementCountCName, productionOwner);
  if (!contract)
    return contract.takeError();

  os << "/* "
     << support::formatRuntimeABIInvocationContractCommentBody(
            "runtime_abi_invocation_contract", *contract)
     << " */\n";
  return llvm::Error::success();
}

void printSelectedSourceIdentityContract(
    llvm::raw_ostream &os, const RVVMicrokernelRecord &record) {
  if (record.selectedBinarySourceKind.empty())
    return;
  os << "/* rvv_microkernel_selected_source_identity: "
     << record.selectedConfigContract
            .formatDispatchContractSelectedSourceIdentityMetadataValue(
                record.selectedBinarySourceKind,
                record.descriptor.getRVVMicrokernelOpName())
     << " */\n";
}

llvm::Error printRecordComment(llvm::raw_ostream &os,
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
  printSelectedSourceIdentityContract(os, record);
  os << "/* "
     << record.selectedConfigContract.formatRuntimeVLBoundaryCommentBody()
     << " */\n";
  if (record.fixedSourceExtent) {
    os << "/* " << record.fixedSourceExtent->formatCommentBody() << " */\n";
    os << "/* runtime_element_count_constraint: "
       << record.selectedConfigContract.getRuntimeElementCountCName()
       << " must equal fixed source vector extent "
       << record.fixedSourceExtent->sourceVectorExtent
       << " before runtime AVL/VL execution */\n";
  }
  if (record.dynamicRuntimeExtent) {
    os << "/* " << record.dynamicRuntimeExtent->formatCommentBody()
       << " */\n";
    os << "/* runtime_element_count_source: "
       << record.selectedConfigContract.getRuntimeElementCountCName()
       << " is the source scf.for upper bound and runtime AVL; no fixed "
          "source-extent trap is emitted for this dynamic vector route */\n";
  }
  os << "/* arithmetic_source: typed op "
     << record.descriptor.getRVVOperationName()
     << " via generated EmitC route and IR-backed callable ABI */\n";
  os << "/* emitc_body_mapping_source: "
     << record.emitcBodyMappingSource << " */\n";
  os << "/* emitc_body_mapping: route_kind="
     << record.emitcBodyMapping.routeKind
     << ", source_authority="
     << record.emitcBodyMapping.sourceAuthority
     << ", required_header="
     << record.emitcBodyMapping.requiredHeader
     << ", arithmetic_intrinsic="
     << record.emitcBodyMapping.arithmeticIntrinsicName << " */\n";
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
       << record.selectedConfigContract
              .formatSelectedVectorShapeConfigCommentBody()
       << " */\n";
    os << "/* "
       << record.selectedConfigContract
              .formatSelectedVectorShapeCapabilitiesCommentBody()
       << " */\n";
  }
  os << "/* "
     << record.selectedConfigContract
            .formatSelectedConfigEmissionAuthorityCommentBody()
     << " */\n";
  os << "/* "
     << record.selectedConfigContract.formatSelectedConfigProfileCommentBody()
     << " */\n";
  os << "/* control_plane_config: sew=" << record.controlPlaneSEW
     << ", lmul=" << record.controlPlaneLMUL
     << ", policy=#tcrv_rvv.policy<tail = "
     << record.selectedConfigEmission.tailPolicy
     << ", mask = " << record.selectedConfigEmission.maskPolicy << "> */\n";
  os << "/* intrinsic_config_source: "
        "RVVBinarySelectedConfigContract cross-checked against verified "
        "tcrv_rvv.setvl/tcrv_rvv.with_vl SEW/LMUL/policy metadata */\n";
  os << "/* intrinsic_config: vector_type="
     << record.selectedConfigEmission.vectorType
     << ", vector_suffix=" << record.selectedConfigEmission.vectorSuffix
     << ", setvl_suffix=" << record.selectedConfigEmission.setvlSuffix
     << ", tail_policy=" << record.selectedConfigEmission.tailPolicy
     << ", mask_policy=" << record.selectedConfigEmission.maskPolicy
     << " */\n";
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
  if (llvm::Error error = printRuntimeABIInvocationContract(
          os, "RVVMicrokernel.cpp", functionName,
          record.descriptor.getRVVRuntimeABIKind(),
          record.descriptor.getRVVRuntimeABIName(),
          record.descriptor.getRVVRuntimeGlueRole(), record.runtimeABIParameters,
          record.selectedConfigContract.getRuntimeElementCountCName(),
          "rvv-target-export", record.descriptor.getArithmeticFamilyID()))
    return error;
  return llvm::Error::success();
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
                                      const RVVBinarySelectedConfigEmissionView
                                          &selectedConfigEmission,
                                      std::int64_t elementCount,
                                      std::optional<std::int64_t>
                                          fixedSourceVectorExtent) {
  os << "/* Harness capacity comes from artifact-local component capacity; each "
        "call still supplies runtime n through the generated C ABI. */\n";
  if (fixedSourceVectorExtent)
    os << "/* Harness fixed source extent constraint: runtime_n must equal "
       << *fixedSourceVectorExtent
       << " for this vector-fronted source fixture. */\n";
  os << "/* self_check_expectation_source: " << expectation.provenance
     << "; expected arithmetic and scalar element type come from typed "
        "selected-source metadata. */\n";
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
  os << "  size_t first_vl = " << selectedConfigEmission.setvlIntrinsicName
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
  if (fixedSourceVectorExtent) {
    os << "  int status = " << functionName
       << "_self_check_one((size_t)" << *fixedSourceVectorExtent << ");\n";
    os << "  if (status != 0)\n";
    os << "    return status;\n";
    os << "  printf(\"tcrv_rvv_microkernel_ok runtime_counts=%zu\\n\", "
          "(size_t)"
       << *fixedSourceVectorExtent << ");\n";
    os << "  return 0;\n";
    os << "}\n";
    return;
  }
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

llvm::Error printMicrokernelHeader(const RVVMicrokernelRecord &record,
                                   llvm::raw_ostream &os) {
  std::string includeGuard = makeMicrokernelHeaderIncludeGuard(record);
  std::string functionName = makeMicrokernelFunctionName(record);

  os << "/* TianChen-RV RVV runtime-callable microkernel C header. */\n";
  os << "/* Scope: declaration-only external C ABI for exactly one "
     << record.descriptor.getRVVMicrokernelOpName() << ". */\n";
  os << "/* selected_body_authority: "
     << record.descriptor.getRVVMicrokernelOpName() << " */\n";
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
  printSelectedSourceIdentityContract(os, record);
  os << "/* "
     << record.selectedConfigContract.formatRuntimeVLBoundaryCommentBody()
     << " */\n";
  os << "/* emitc_body_mapping_source: " << record.emitcBodyMappingSource
     << " */\n";
  os << "/* emitc_body_mapping_status: selected RVV EmitC body mapping was "
        "validated before source/header/object artifact export; this header "
        "remains declaration-only and carries no intrinsic include. */\n";
  if (record.fixedSourceExtent) {
    os << "/* " << record.fixedSourceExtent->formatCommentBody() << " */\n";
    os << "/* runtime_element_count_constraint: "
       << record.selectedConfigContract.getRuntimeElementCountCName()
       << " must equal fixed source vector extent "
       << record.fixedSourceExtent->sourceVectorExtent
       << " before runtime AVL/VL execution */\n";
  }
  if (record.dynamicRuntimeExtent) {
    os << "/* " << record.dynamicRuntimeExtent->formatCommentBody()
       << " */\n";
    os << "/* runtime_element_count_source: "
       << record.selectedConfigContract.getRuntimeElementCountCName()
       << " is the source scf.for upper bound and runtime AVL; no fixed "
          "source-extent trap is emitted for this dynamic vector route */\n";
  }
  os << "/* control_plane_runtime_avl: body index argument maps to "
        "target/export-owned runtime "
     << record.selectedConfigContract.getRuntimeElementCountCName()
     << " ABI parameter */\n";
  os << "/* control_plane_vl: !tcrv_rvv.vl value consumed by "
        "tcrv_rvv.with_vl */\n";
  os << "/* dataflow_abi_roles: lhs_load.buffer_role=lhs-input-buffer, "
        "rhs_load.buffer_role=rhs-input-buffer, "
        "store.buffer_role=output-buffer; runtime "
     << record.selectedConfigContract.getRuntimeElementCountCName()
     << " remains the target/export-owned runtime element-count ABI "
        "parameter */\n";
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
  if (llvm::Error error = printRuntimeABIInvocationContract(
          os, "RVVMicrokernel.cpp", functionName,
          record.descriptor.getRVVRuntimeABIKind(),
          record.descriptor.getRVVRuntimeABIName(),
          record.descriptor.getRVVRuntimeGlueRole(), record.runtimeABIParameters,
          record.selectedConfigContract.getRuntimeElementCountCName(),
          "rvv-target-export", record.descriptor.getArithmeticFamilyID()))
    return error;

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
  return llvm::Error::success();
}

void printCStringLiteralEscaped(llvm::raw_ostream &os,
                                llvm::StringRef text) {
  for (char character : text) {
    unsigned char byte = static_cast<unsigned char>(character);
    switch (character) {
    case '\\':
      os << "\\\\";
      break;
    case '"':
      os << "\\\"";
      break;
    case '\n':
      os << "\\n";
      break;
    case '\r':
      os << "\\r";
      break;
    case '\t':
      os << "\\t";
      break;
    default:
      if (std::isprint(byte)) {
        os << character;
      } else {
        const char *hex = "0123456789abcdef";
        os << "\\x" << hex[(byte >> 4) & 0x0f] << hex[byte & 0x0f];
      }
      break;
    }
  }
}

void printObjectEvidenceLine(llvm::raw_ostream &os, llvm::StringRef key,
                             llvm::StringRef value) {
  os << "  \"";
  printCStringLiteralEscaped(os, key);
  os << "=";
  printCStringLiteralEscaped(os, value);
  os << "\\n\"\n";
}

void appendMicrokernelObjectEvidenceSection(
    const RVVMicrokernelRecord &record, std::string &source) {
  std::string section;
  llvm::raw_string_ostream os(section);

  os << "\n/* Object-only RVV artifact evidence: compiler-artifact metadata "
        "for the relocatable object, not runtime/correctness/performance "
        "evidence. */\n";
  os << "#if defined(__clang__) || defined(__GNUC__)\n";
  os << "__attribute__((used, "
        "section(\".rodata.tianchenrv.rvv_artifact\")))\n";
  os << "#endif\n";
  os << "static const char "
        "tianchenrv_rvv_microkernel_object_artifact_evidence[] =\n";

  printObjectEvidenceLine(os, "tianchenrv.rvv.artifact",
                          "rvv-op-owned-object-artifact.v1");
  printObjectEvidenceLine(os, "owner", kRVVPluginName);
  printObjectEvidenceLine(os, "artifact_kind",
                          kMicrokernelObjectArtifactKind);
  printObjectEvidenceLine(os, "object_route",
                          record.descriptor.getRVVObjectRouteID());
  printObjectEvidenceLine(os, "source_route",
                          record.descriptor.getRVVRouteID());
  printObjectEvidenceLine(os, "selected_kernel", record.kernelSymbol);
  printObjectEvidenceLine(os, "selected_variant", record.variantSymbol);
  printObjectEvidenceLine(os, "selected_role", record.role);
  printObjectEvidenceLine(os, "selected_march", record.selectedMarch);
  if (record.selectedMABI)
    printObjectEvidenceLine(os, "selected_mabi", *record.selectedMABI);
  printObjectEvidenceLine(os, "selected_binary_dtype",
                          record.descriptor.getDTypeID());
  if (!record.selectedBinarySourceKind.empty())
    printObjectEvidenceLine(os, "selected_binary_source_kind",
                            record.selectedBinarySourceKind);
  printObjectEvidenceLine(os, "selected_binary_family",
                          record.descriptor.getArithmeticFamilyID());
  printObjectEvidenceLine(os, "selected_binary_operator",
                          record.descriptor.family.arithmeticVerb);
  printObjectEvidenceLine(os, "selected_binary_microkernel_op",
                          record.descriptor.getRVVMicrokernelOpName());
  printObjectEvidenceLine(os, "emitc_source_op",
                          record.descriptor.getRVVOperationName());
  printObjectEvidenceLine(os, "emitc_lowerable_op_interface",
                          kEmitCLowerableOpInterfaceName);
  printObjectEvidenceLine(os, "emitc_body_mapping_source",
                          record.emitcBodyMappingSource);
  printObjectEvidenceLine(os, "emitc_route_kind",
                          record.emitcBodyMapping.routeKind);
  printObjectEvidenceLine(os, "emitc_source_authority",
                          record.emitcBodyMapping.sourceAuthority);
  printObjectEvidenceLine(os, "emitc_required_header",
                          record.emitcBodyMapping.requiredHeader);
  printObjectEvidenceLine(os, "emitc_arithmetic_intrinsic",
                          record.emitcBodyMapping.arithmeticIntrinsicName);
  printObjectEvidenceLine(os, "selected_vector_shape",
                          record.selectedConfigContract.getShapeID());
  printObjectEvidenceLine(
      os, "selected_vector_sew",
      std::to_string(record.selectedConfigContract.getSEWBits()));
  printObjectEvidenceLine(os, "selected_vector_lmul",
                          record.selectedConfigContract.getLMUL());
  printObjectEvidenceLine(os, "selected_tail_policy",
                          record.selectedConfigContract.getTailPolicy());
  printObjectEvidenceLine(os, "selected_mask_policy",
                          record.selectedConfigContract.getMaskPolicy());
  printObjectEvidenceLine(os, "selected_vector_type",
                          record.selectedConfigEmission.vectorType);
  printObjectEvidenceLine(os, "selected_vector_suffix",
                          record.selectedConfigEmission.vectorSuffix);
  printObjectEvidenceLine(os, "selected_setvl_suffix",
                          record.selectedConfigEmission.setvlSuffix);
  printObjectEvidenceLine(
      os, "selected_config_profile_hardware_facts",
      record.selectedConfigContract
          .formatSelectedConfigProfileHardwareFactsMetadataValue());
  printObjectEvidenceLine(
      os, "selected_config_profile_variant_config",
      record.selectedConfigContract
          .formatSelectedConfigProfileVariantConfigMetadataValue());
  printObjectEvidenceLine(
      os, "selected_config_profile_runtime_roles",
      record.selectedConfigContract
          .formatSelectedConfigProfileRuntimeRolesMetadataValue());
  const RVVRuntimeLengthContract &runtimeLength =
      record.selectedConfigContract.getRuntimeLengthContract();
  printObjectEvidenceLine(os, "runtime_element_count_c_name",
                          runtimeLength.getRuntimeElementCountCName());
  printObjectEvidenceLine(os, "runtime_avl_source",
                          runtimeLength.getRuntimeAVLSource());
  printObjectEvidenceLine(os, "runtime_avl_role",
                          runtimeLength.getRuntimeAVLRole());
  printObjectEvidenceLine(os, "runtime_vl_source",
                          runtimeLength.getRuntimeVLSource());
  printObjectEvidenceLine(os, "runtime_vl_scope",
                          runtimeLength.getRuntimeVLScope());
  printObjectEvidenceLine(os, "component_capacity_element_count",
                          std::to_string(
                              runtimeLength.getComponentCapacityElementCount()));
  printObjectEvidenceLine(os, "runtime_abi",
                          record.descriptor.getRVVRuntimeABI());
  printObjectEvidenceLine(os, "runtime_abi_kind",
                          record.descriptor.getRVVRuntimeABIKind());
  printObjectEvidenceLine(os, "runtime_abi_name",
                          record.descriptor.getRVVRuntimeABIName());
  printObjectEvidenceLine(os, "runtime_glue_role",
                          record.descriptor.getRVVRuntimeGlueRole());
  printObjectEvidenceLine(os, "runtime_abi_invocation_contract",
                          "production-cpp-ir-backed-callable-abi");
  printObjectEvidenceLine(os, "runtime_abi_callable_symbol",
                          makeMicrokernelFunctionName(record));
  printObjectEvidenceLine(os, "runtime_abi_ordered_roles",
                          support::formatRuntimeABIOrderedRoles(
                              record.runtimeABIParameters));
  printObjectEvidenceLine(
      os, "runtime_abi_production_owner", "rvv-target-export");
  printObjectEvidenceLine(
      os, "descriptor_compute_authority",
      "quarantined-after-typed-rvv-source-authority");

  for (auto [index, parameter] :
       llvm::enumerate(record.runtimeABIParameters)) {
    std::string value;
    llvm::raw_string_ostream valueStream(value);
    valueStream << "c_name=" << parameter.cName
                << ",c_type=" << parameter.cType
                << ",role="
                << support::stringifyRuntimeABIParameterRole(parameter.role)
                << ",ownership="
                << support::stringifyRuntimeABIParameterOwnership(
                       parameter.ownership);
    valueStream.flush();
    printObjectEvidenceLine(
        os, (llvm::Twine("runtime_abi_parameter[") + llvm::Twine(index) + "]")
                .str(),
        value);
  }

  os << "  ;\n";
  os.flush();
  source += section;
}

TargetArtifactRouteMetadata
buildRVVMicrokernelSourceRouteMetadata(
    const RVVBinaryFamilyRecord &family);

void addRVVMicrokernelConservativeRouteClaims(
    TargetArtifactRouteMetadata &metadata);

TargetArtifactRouteMetadata buildRVVMicrokernelArtifactRouteMetadata(
    const RVVMicrokernelDirectRouteManifestEntry &route);

tianchenrv::target::TargetArtifactExportFn
getRVVMicrokernelExactExportFn(const RVVBinaryFamilyRecord &family,
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

  const RVVBinaryFamilyRecord &family = *route->family;
  llvm::Expected<const RVVVectorShapeConfig *> selectedShape =
      resolveRVVMicrokernelCandidateSelectedShape(candidate, family);
  if (!selectedShape)
    return selectedShape.takeError();

  RVVBinaryIntrinsicRoute descriptor =
      getRVVBinaryIntrinsicRoute(family, **selectedShape);
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
  if (candidate.kernel) {
    if (!candidate.kernel->getParentOfType<mlir::ModuleOp>())
      return makeMicrokernelError(
          candidate.kernel,
          "selected RVV target artifact candidate requires an enclosing "
          "builtin.module for selected config/runtime AVL contract "
          "validation");
    // Every finite typed RVV binary artifact route must prove op-owned source
    // identity at the selected lowering boundary; this is not vadd-specific.
    bool requireBoundarySourceIdentity =
        family.dtype == RVVBinaryDTypeKind::I32 ||
        family.dtype == RVVBinaryDTypeKind::I64;
    llvm::Expected<RVVMicrokernelRecord> sourceAuthority =
        buildKernelRecordForRVVBinaryFamily(
            candidate.kernel, family, candidate.selectedVariant,
            candidate.role, candidate.routeID, requireBoundarySourceIdentity);
    if (!sourceAuthority)
      return sourceAuthority.takeError();
    if (llvm::Error error = validateRVVMicrokernelSelectedPlanMetadata(
            candidate, sourceAuthority->selectedConfigContract))
      return error;
    if (llvm::Error error = validateRVVMicrokernelSelectedSourceIdentityMetadata(
            candidate, *sourceAuthority))
      return error;
  }
  return validateRVVBinaryCandidateRuntimeABIMirrorsIR(candidate, descriptor);
}

TargetArtifactRouteMetadata
buildRVVMicrokernelSourceRouteMetadata(
    const RVVBinaryFamilyRecord &family) {
  TargetArtifactRouteMetadata metadata(
      family.runtimeABI, family.runtimeABIKind, family.runtimeABIName,
      family.runtimeGlueRole);

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
  metadata.addSelectedPlanMetadataRequirement(
      getRVVEmitCRouteKindMetadataName(),
      getRVVEmitCRouteKindMetadataValue(), getRVVEmitCRouteMetadataRole());
  metadata.addSelectedPlanMetadataRequirement(
      getRVVEmitCSourceAuthorityMetadataName(),
      getRVVEmitCSourceAuthorityMetadataValue(),
      getRVVEmitCRouteMetadataRole());
  metadata.addSelectedPlanMetadataRequirement(
      getRVVEmitCRequiredHeaderMetadataName(),
      getRVVEmitCRequiredHeaderMetadataValue(),
      getRVVEmitCRouteMetadataRole());
  metadata.addSelectedPlanMetadataPresenceRequirement(
      getRVVEmitCArithmeticIntrinsicMetadataName(),
      getRVVEmitCRouteMetadataRole());
  metadata.addSelectedPlanMetadataPresenceRequirement(
      getRVVRuntimeElementCountCNameMetadataName(),
      getRVVRuntimeControlNameMetadataRole());
  metadata.addSelectedPlanMetadataPresenceRequirement(
      getRVVComponentCapacityElementCountMetadataName(),
      getRVVComponentCapacityElementCountMetadataRole());

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

  if (family.dtype == RVVBinaryDTypeKind::I32 ||
      family.dtype == RVVBinaryDTypeKind::I64) {
    llvm::StringRef sourceIdentityRole =
        getRVVTypedBinarySourceIdentityMetadataRole();
    metadata.addSelectedPlanMetadataPresenceRequirement(
        getRVVSelectedBinarySourceKindMetadataName(), sourceIdentityRole);
    metadata.addSelectedPlanMetadataRequirement(
        getRVVSelectedBinaryMicrokernelOpMetadataName(),
        family.microkernelOpName, sourceIdentityRole);
  }

  llvm::StringRef profileRole = getRVVSelectedConfigProfileMetadataRole();
  metadata.addSelectedPlanMetadataPresenceRequirement(
      getRVVSelectedConfigProfileHardwareFactsMetadataName(), profileRole);
  metadata.addSelectedPlanMetadataPresenceRequirement(
      getRVVSelectedConfigProfileVariantConfigMetadataName(), profileRole);
  metadata.addSelectedPlanMetadataPresenceRequirement(
      getRVVSelectedConfigProfileRuntimeRolesMetadataName(), profileRole);

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
  metadata.addClaimField("descriptor_compute_authority",
                         "quarantined-after-typed-rvv-source-authority");
  metadata.addClaimField("descriptor_config_authority",
                         "quarantined-after-selected-rvv-config-contract");
  metadata.addClaimField("descriptor_runtime_authority",
                         "quarantined-runtime-avl-from-ir-backed-abi");
}

TargetArtifactRouteMetadata buildRVVMicrokernelArtifactRouteMetadata(
    const RVVMicrokernelDirectRouteManifestEntry &route) {
  if (route.routeKind == RVVMicrokernelDirectRouteKind::Source)
    return buildRVVMicrokernelSourceRouteMetadata(*route.family);

  return buildRVVMicrokernelSourceRouteMetadata(*route.family);
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
    const RVVBinaryIntrinsicRoute &descriptor) {
  if (candidates.size() != 1)
    return false;
  return candidateMatchesRVVRouteRegistration(candidates.front(), descriptor);
}

llvm::Expected<bool> matchRVVMicrokernelI64VAddObjectCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  return matchRVVMicrokernelI64FamilyCandidate(
      candidates, getI64VAddIntrinsicRoute());
}

llvm::Expected<bool> matchRVVMicrokernelI64VSubObjectCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  return matchRVVMicrokernelI64FamilyCandidate(
      candidates, getI64VSubIntrinsicRoute());
}

llvm::Expected<bool> matchRVVMicrokernelI64VMulObjectCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  return matchRVVMicrokernelI64FamilyCandidate(
      candidates, getI64VMulIntrinsicRoute());
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
      candidates, getI64VAddIntrinsicRoute());
}

llvm::Expected<bool> matchRVVMicrokernelI64VSubHeaderCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  return matchRVVMicrokernelI64FamilyCandidate(
      candidates, getI64VSubIntrinsicRoute());
}

llvm::Expected<bool> matchRVVMicrokernelI64VMulHeaderCandidate(
    llvm::ArrayRef<tianchenrv::target::TargetArtifactCandidate> candidates) {
  return matchRVVMicrokernelI64FamilyCandidate(
      candidates, getI64VMulIntrinsicRoute());
}

TargetArtifactCompositeMatchFn
getRVVMicrokernelHeaderMatchFn(const RVVBinaryFamilyRecord &family) {
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
getRVVMicrokernelObjectMatchFn(const RVVBinaryFamilyRecord &family) {
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
getRVVMicrokernelExactExportFn(const RVVBinaryFamilyRecord &family,
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
        for (const RVVBinaryFamilyRecord *family :
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
    const RVVBinaryFamilyRecord &family,
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
  (void)os;
  if (!route.family)
    return makeModuleMicrokernelError(
        "RVV microkernel direct artifact export requires an exact typed "
        "binary family route");
  if (route.routeKind == RVVMicrokernelDirectRouteKind::Source)
    return makeModuleMicrokernelError(kDirectCSourceRouteDeletedReason);
  if (route.routeKind == RVVMicrokernelDirectRouteKind::Object)
    return makeModuleMicrokernelObjectError(kDirectCSourceRouteDeletedReason);

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
  case RVVMicrokernelDirectRouteKind::Source:
    return makeModuleMicrokernelError(kDirectCSourceRouteDeletedReason);
  case RVVMicrokernelDirectRouteKind::Header:
    return printMicrokernelHeader(*record, os);
  case RVVMicrokernelDirectRouteKind::Object:
    return makeModuleMicrokernelObjectError(kDirectCSourceRouteDeletedReason);
  }
  llvm_unreachable("unknown RVV microkernel direct route kind");
}

llvm::Error exportRVVMicrokernelCForBinaryFamily(
    mlir::ModuleOp module, const RVVBinaryFamilyRecord &family,
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
    mlir::ModuleOp module, const RVVBinaryFamilyRecord &family,
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
    mlir::ModuleOp module, const RVVBinaryFamilyRecord &family,
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

llvm::Expected<RVVBinarySelectedConfigContract>
resolveRVVMicrokernelSelectedConfigContractAuthority(
    KernelOp kernel, const RVVBinaryFamilyRecord &family,
    llvm::StringRef selectedVariant, llvm::StringRef role,
    llvm::StringRef routeID) {
  llvm::Expected<RVVMicrokernelRecord> record =
      buildKernelRecordForRVVBinaryFamily(kernel, family, selectedVariant, role,
                                          routeID);
  if (!record)
    return record.takeError();

  if (llvm::Error error = requireRVVSourceAuthorityField(
          "selected variant", record->variantSymbol, selectedVariant, routeID,
          selectedVariant))
    return std::move(error);
  if (llvm::Error error = requireRVVSourceAuthorityField(
          "role", record->role, role, routeID, selectedVariant))
    return std::move(error);
  if (llvm::Error error = requireRVVSourceAuthorityField(
          "active route", record->activeRouteID, routeID, routeID,
          selectedVariant))
    return std::move(error);

  if (record->descriptor.getRVVMicrokernelOpName() !=
      family.microkernelOpName)
    return makeMicrokernelError(
        kernel, llvm::Twine("selected RVV component authority for route '") +
                    routeID + "' requires " + family.microkernelOpName +
                    " but typed RVV record is " +
                    record->descriptor.getRVVMicrokernelOpName());
  return record->selectedConfigContract;
}

llvm::Error exportRVVMicrokernelSelfCheckC(mlir::ModuleOp module,
                                           llvm::raw_ostream &os) {
  (void)module;
  (void)os;
  return makeModuleMicrokernelError(kDirectCSourceRouteDeletedReason);
}

llvm::Error exportRVVMicrokernelHeaderForBinaryFamily(
    mlir::ModuleOp module, const RVVBinaryFamilyRecord &family,
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
    mlir::ModuleOp module, const RVVBinaryFamilyRecord &family,
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
  (void)registry;
  return llvm::Error::success();
}

llvm::Error registerRVVMicrokernelPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry) {
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      kRVVPluginName, registerRVVMicrokernelTargetExporters));
}

llvm::Error registerRVVMicrokernelTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry) {
  (void)registry;
  return llvm::Error::success();
}

} // namespace tianchenrv::target::rvv
