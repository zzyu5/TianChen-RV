#include "TianChenRV/Target/RVV/RVVMicrokernel.h"

#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Support/RuntimeABICallablePlan.h"
#include "TianChenRV/Support/RuntimeABIContract.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"
#include "TianChenRV/Target/I32BinaryFamilyRegistry.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"
#include "TianChenRV/Target/RVV/RVVBinaryDescriptor.h"
#include "TianChenRV/Target/RVV/RVVVectorShape.h"

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
using tianchenrv::tcrv::exec::DiagnosticOp;
using tianchenrv::tcrv::exec::DispatchCaseOp;
using tianchenrv::tcrv::exec::DispatchOp;
using tianchenrv::tcrv::exec::FallbackOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::MemWindowOp;
using tianchenrv::tcrv::exec::RuntimeParamOp;
using tianchenrv::tcrv::exec::VariantOp;
using tianchenrv::tcrv::rvv::I32AddOp;
using tianchenrv::tcrv::rvv::I32M1VectorType;
using tianchenrv::tcrv::rvv::I32M2VectorType;
using tianchenrv::tcrv::rvv::I32LoadOp;
using tianchenrv::tcrv::rvv::I32MulOp;
using tianchenrv::tcrv::rvv::I32StoreOp;
using tianchenrv::tcrv::rvv::I32SubOp;
using tianchenrv::tcrv::rvv::I32VAddMicrokernelOp;
using tianchenrv::tcrv::rvv::I32VMulMicrokernelOp;
using tianchenrv::tcrv::rvv::I32VSubMicrokernelOp;
using tianchenrv::tcrv::rvv::I64AddOp;
using tianchenrv::tcrv::rvv::I64LoadOp;
using tianchenrv::tcrv::rvv::I64M1VectorType;
using tianchenrv::tcrv::rvv::I64MulOp;
using tianchenrv::tcrv::rvv::I64StoreOp;
using tianchenrv::tcrv::rvv::I64SubOp;
using tianchenrv::tcrv::rvv::LoweringBoundaryOp;
using tianchenrv::tcrv::rvv::MaskPolicy;
using tianchenrv::tcrv::rvv::PolicyAttr;
using tianchenrv::tcrv::rvv::SetVLOp;
using tianchenrv::tcrv::rvv::TailPolicy;
using tianchenrv::tcrv::rvv::WithVLOp;

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
constexpr llvm::StringLiteral kElementCountAttrName("element_count");
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
constexpr llvm::StringLiteral kSEWAttrName("sew");
constexpr llvm::StringLiteral kLMULAttrName("lmul");
constexpr llvm::StringLiteral kPolicyAttrName("policy");
constexpr llvm::StringLiteral kBufferRoleAttrName("buffer_role");
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

enum class RVVI32VAddDataflowStepKind {
  Load,
  Add,
  Sub,
  Mul,
  Store,
};

enum class RVVI32VAddDataflowValue {
  None,
  LHSVector,
  RHSVector,
  ResultVector,
};

struct RVVI32VAddDataflowStep {
  RVVI32VAddDataflowStepKind kind = RVVI32VAddDataflowStepKind::Load;
  support::RuntimeABIParameterRole bufferRole =
      support::RuntimeABIParameterRole::LHSInputBuffer;
  RVVI32VAddDataflowValue result = RVVI32VAddDataflowValue::None;
  RVVI32VAddDataflowValue lhs = RVVI32VAddDataflowValue::None;
  RVVI32VAddDataflowValue rhs = RVVI32VAddDataflowValue::None;
  RVVI32VAddDataflowValue value = RVVI32VAddDataflowValue::None;
};

struct RVVI32VAddDataflowEmissionPlan {
  llvm::SmallVector<RVVI32VAddDataflowStep, 4> steps;
};

struct RVVIntrinsicConfig {
  std::int64_t sew = 0;
  std::string lmul;
  std::string vectorType;
  std::string vectorSuffix;
  std::string setvlSuffix;
  std::string setvlIntrinsicName;
  std::string loadIntrinsicName;
  std::string arithmeticIntrinsicName;
  std::string storeIntrinsicName;
  std::string tailPolicy;
  std::string maskPolicy;
};

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
  return tianchenrv::target::rvv::getI32VAddFamilyDescriptor();
}

const RVVI32MicrokernelFamilySpec &getI32VSubFamilySpec() {
  return tianchenrv::target::rvv::getI32VSubFamilyDescriptor();
}

const RVVI32MicrokernelFamilySpec &getI32VMulFamilySpec() {
  return tianchenrv::target::rvv::getI32VMulFamilyDescriptor();
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

const RVVI32MicrokernelFamilySpec *
getI32MicrokernelFamilyForSourceRoute(llvm::StringRef routeID) {
  const RVVI32MicrokernelFamilySpec &addFamily = getI32VAddFamilySpec();
  if (routeID == addFamily.routeID)
    return &addFamily;
  const RVVI32MicrokernelFamilySpec &subFamily = getI32VSubFamilySpec();
  if (routeID == subFamily.routeID)
    return &subFamily;
  const RVVI32MicrokernelFamilySpec &mulFamily = getI32VMulFamilySpec();
  if (routeID == mulFamily.routeID)
    return &mulFamily;
  return nullptr;
}

const RVVBinaryFamilyDescriptor *
getI64MicrokernelFamilyForOp(mlir::Operation *op) {
  if (!op)
    return nullptr;
  llvm::StringRef opName = op->getName().getStringRef();
  for (const RVVBinaryFamilyDescriptor *family :
       getRVVBinaryFamilyDescriptors()) {
    if (family->dtype == RVVBinaryDTypeKind::I64 &&
        family->microkernelOpName == opName)
      return family;
  }
  return nullptr;
}

const RVVBinaryFamilyDescriptor *
getI64MicrokernelFamilyForSourceRoute(llvm::StringRef routeID) {
  for (const RVVBinaryFamilyDescriptor *family :
       getRVVBinaryFamilyDescriptors()) {
    if (family->dtype == RVVBinaryDTypeKind::I64 &&
        family->routeID == routeID)
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

bool candidateMatchesRVVBinaryDescriptor(
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

llvm::Error makeRuntimeABICallablePlanError(
    KernelOp kernel, const RVVBinaryIntrinsicDescriptor &descriptor,
    llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "runtime ABI callable plan validation failed for family '"
         << descriptor.getArithmeticFamilyID() << "'";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
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

llvm::Expected<support::RuntimeABIParameterRole> getDataflowRoleAttr(
    KernelOp kernel, mlir::Operation *op, llvm::StringRef attrName,
    llvm::StringRef context) {
  std::string value;
  if (llvm::Error error =
          requireSafeStringAttr(kernel, op, attrName, context, value))
    return std::move(error);

  std::optional<support::RuntimeABIParameterRole> parsedRole =
      support::symbolizeRuntimeABIParameterRole(value);
  if (!parsedRole)
    return makeMicrokernelError(
        kernel, llvm::Twine(context) + " attribute '" +
                    attrName +
                    "' must reference a supported runtime ABI parameter role");

  return *parsedRole;
}

RVVI32VAddDataflowStep makeLoadStep(support::RuntimeABIParameterRole role,
                                    RVVI32VAddDataflowValue result) {
  RVVI32VAddDataflowStep step;
  step.kind = RVVI32VAddDataflowStepKind::Load;
  step.bufferRole = role;
  step.result = result;
  return step;
}

RVVI32VAddDataflowStep makeAddStep(RVVI32VAddDataflowValue lhs,
                                   RVVI32VAddDataflowValue rhs,
                                   RVVI32VAddDataflowValue result) {
  RVVI32VAddDataflowStep step;
  step.kind = RVVI32VAddDataflowStepKind::Add;
  step.lhs = lhs;
  step.rhs = rhs;
  step.result = result;
  return step;
}

RVVI32VAddDataflowStep makeSubStep(RVVI32VAddDataflowValue lhs,
                                   RVVI32VAddDataflowValue rhs,
                                   RVVI32VAddDataflowValue result) {
  RVVI32VAddDataflowStep step;
  step.kind = RVVI32VAddDataflowStepKind::Sub;
  step.lhs = lhs;
  step.rhs = rhs;
  step.result = result;
  return step;
}

RVVI32VAddDataflowStep makeMulStep(RVVI32VAddDataflowValue lhs,
                                   RVVI32VAddDataflowValue rhs,
                                   RVVI32VAddDataflowValue result) {
  RVVI32VAddDataflowStep step;
  step.kind = RVVI32VAddDataflowStepKind::Mul;
  step.lhs = lhs;
  step.rhs = rhs;
  step.result = result;
  return step;
}

RVVI32VAddDataflowStep makeStoreStep(support::RuntimeABIParameterRole role,
                                     RVVI32VAddDataflowValue value) {
  RVVI32VAddDataflowStep step;
  step.kind = RVVI32VAddDataflowStepKind::Store;
  step.bufferRole = role;
  step.value = value;
  return step;
}

llvm::StringRef stringifyTailPolicyValue(TailPolicy policy) {
  switch (policy) {
  case TailPolicy::Agnostic:
    return "agnostic";
  case TailPolicy::Undisturbed:
    return "undisturbed";
  }
  return "unknown";
}

llvm::StringRef stringifyMaskPolicyValue(MaskPolicy policy) {
  switch (policy) {
  case MaskPolicy::Agnostic:
    return "agnostic";
  case MaskPolicy::Undisturbed:
    return "undisturbed";
  }
  return "unknown";
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

llvm::Error makeIntrinsicConfigError(
    KernelOp kernel, llvm::StringRef activeRouteID,
    const RVVBinaryIntrinsicDescriptor &descriptor, llvm::Twine message) {
  std::string routeID = getEffectiveRouteID(activeRouteID, descriptor);
  return makeMicrokernelError(
      kernel, llvm::Twine("route '") + routeID + "' selected family '" +
                  descriptor.getRVVMicrokernelOpName() +
                  "' has invalid RVV intrinsic metadata: " + message);
}

llvm::Expected<RVVIntrinsicConfig> buildRVVIntrinsicConfig(
    KernelOp kernel, llvm::StringRef activeRouteID,
    const RVVBinaryIntrinsicDescriptor &descriptor, SetVLOp setvl,
    WithVLOp withVL, PolicyAttr selectedPolicy,
    const RVVI32VectorShapeConfig &selectedConfig) {
  if (!selectedPolicy)
    return makeIntrinsicConfigError(
        kernel, activeRouteID, descriptor,
        "missing selected variant tcrv_rvv.policy metadata");

  if (setvl.getPolicy() != selectedPolicy)
    return makeIntrinsicConfigError(
        kernel, activeRouteID, descriptor,
        "tcrv_rvv.setvl policy must match selected variant "
        "tcrv_rvv.policy metadata before C intrinsic emission");

  auto withVLSew = withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto withVLLMUL = withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  auto withVLPolicy = withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName);
  if (!withVLSew || !withVLLMUL || !withVLPolicy)
    return makeIntrinsicConfigError(
        kernel, activeRouteID, descriptor,
        "tcrv_rvv.with_vl must carry explicit SEW/LMUL/policy metadata "
        "before C intrinsic emission");

  if (withVLSew.getInt() != static_cast<std::int64_t>(setvl.getSew()) ||
      withVLLMUL.getValue() != setvl.getLmul() ||
      withVLPolicy != setvl.getPolicy())
    return makeIntrinsicConfigError(
        kernel, activeRouteID, descriptor,
        "tcrv_rvv.with_vl SEW/LMUL/policy metadata must match the "
        "defining tcrv_rvv.setvl metadata before C intrinsic emission");

  llvm::StringRef tail = stringifyTailPolicyValue(selectedPolicy.getTail());
  llvm::StringRef mask = stringifyMaskPolicyValue(selectedPolicy.getMask());
  if (selectedPolicy.getTail() != TailPolicy::Agnostic ||
      selectedPolicy.getMask() != MaskPolicy::Agnostic)
    return makeIntrinsicConfigError(
        kernel, activeRouteID, descriptor,
        llvm::Twine("unsupported policy tail=") + tail + ", mask=" + mask +
            "; supported RVV C intrinsic emission requires tail=agnostic, "
            "mask=agnostic");

  if (setvl.getSew() != static_cast<std::uint64_t>(selectedConfig.sewBits) ||
      setvl.getLmul() != selectedConfig.lmul)
    return makeIntrinsicConfigError(
        kernel, activeRouteID, descriptor,
        llvm::Twine("unsupported SEW/LMUL sew=") +
            llvm::Twine(setvl.getSew()) + ", lmul=" + setvl.getLmul() +
            "; selected capability config requires sew=" +
            llvm::Twine(selectedConfig.sewBits) + ",lmul=" +
            selectedConfig.lmul);

  RVVIntrinsicConfig config;
  config.sew = descriptor.getSEWBits();
  config.lmul = setvl.getLmul().str();
  config.vectorType = descriptor.getVectorType().str();
  config.vectorSuffix = descriptor.getVectorSuffix().str();
  config.setvlSuffix = descriptor.getSetVLSuffix().str();
  config.setvlIntrinsicName = descriptor.getSetVLIntrinsicName();
  config.loadIntrinsicName = descriptor.getLoadIntrinsicName();
  config.arithmeticIntrinsicName = descriptor.getArithmeticIntrinsicName();
  config.storeIntrinsicName = descriptor.getStoreIntrinsicName();
  config.tailPolicy = tail.str();
  config.maskPolicy = mask.str();
  return config;
}

llvm::StringRef getRVVVectorLMUL(mlir::Type type) {
  if (llvm::isa<I32M1VectorType>(type))
    return "m1";
  if (llvm::isa<I32M2VectorType>(type))
    return "m2";
  if (llvm::isa<I64M1VectorType>(type))
    return "m1";
  return {};
}

llvm::Error requireDataflowValueLMUL(KernelOp kernel, mlir::Value value,
                                     const RVVIntrinsicConfig &config,
                                     llvm::StringRef context) {
  llvm::StringRef valueLMUL = getRVVVectorLMUL(value.getType());
  if (valueLMUL.empty())
    return makeMicrokernelError(
        kernel, llvm::Twine(context) +
                    " must use a supported !tcrv_rvv RVV vector typed "
                    "dataflow values before C intrinsic emission");
  if (valueLMUL != config.lmul)
    return makeMicrokernelError(
        kernel, llvm::Twine(context) +
                    " type LMUL must match validated setvl/with_vl LMUL '" +
                    config.lmul + "' before C intrinsic emission");
  return llvm::Error::success();
}

using DataflowValueBinding =
    std::pair<mlir::Value, RVVI32VAddDataflowValue>;

llvm::Expected<RVVI32VAddDataflowValue>
getLoadResultForBufferRole(KernelOp kernel,
                           support::RuntimeABIParameterRole role) {
  switch (role) {
  case support::RuntimeABIParameterRole::LHSInputBuffer:
    return RVVI32VAddDataflowValue::LHSVector;
  case support::RuntimeABIParameterRole::RHSInputBuffer:
    return RVVI32VAddDataflowValue::RHSVector;
  case support::RuntimeABIParameterRole::OutputBuffer:
  case support::RuntimeABIParameterRole::RuntimeElementCount:
  case support::RuntimeABIParameterRole::DispatchAvailabilityGuard:
    return makeMicrokernelError(
        kernel, "tcrv_rvv typed load buffer_role must reference "
                "'lhs-input-buffer' or 'rhs-input-buffer' for this bounded "
                "RVV binary microkernel export route");
  }
  return makeMicrokernelError(kernel,
                              "unsupported tcrv_rvv typed load buffer_role");
}

bool dataflowPlanAlreadyDefines(
    const RVVI32VAddDataflowEmissionPlan &dataflowPlan,
    RVVI32VAddDataflowValue value) {
  return llvm::any_of(dataflowPlan.steps, [&](const auto &step) {
    return step.result == value;
  });
}

llvm::Expected<RVVI32VAddDataflowValue>
lookupTypedDataflowValue(KernelOp kernel,
                         llvm::ArrayRef<DataflowValueBinding> bindings,
                         mlir::Value value, llvm::StringRef context) {
  for (const auto &[candidate, symbolicValue] : bindings)
    if (candidate == value)
      return symbolicValue;
  return makeMicrokernelError(
      kernel, llvm::Twine(context) +
                  " must consume a value produced by a preceding verified "
                  "tcrv_rvv typed dataflow op");
}

llvm::Error buildDataflowEmissionPlanFromTypedBody(
    KernelOp kernel, WithVLOp withVL,
    llvm::ArrayRef<mlir::Operation *> dataflowOps,
    const RVVI32MicrokernelFamilySpec &family,
    const RVVIntrinsicConfig &intrinsicConfig,
    RVVI32VAddDataflowEmissionPlan &dataflowPlan) {
  dataflowPlan.steps.clear();
  llvm::SmallVector<DataflowValueBinding, 3> bindings;

  for (mlir::Operation *op : dataflowOps) {
    if (auto load = llvm::dyn_cast<I32LoadOp>(op)) {
      if (load.getVl() != withVL.getVl())
        return makeMicrokernelError(
            kernel, "tcrv_rvv.i32_load must consume the !tcrv_rvv.vl token "
                    "owned by the surrounding tcrv_rvv.with_vl");
      if (llvm::Error error = requireDataflowValueLMUL(
              kernel, load.getLoaded(), intrinsicConfig,
              "tcrv_rvv.i32_load result"))
        return error;

      llvm::Expected<support::RuntimeABIParameterRole> role =
          getDataflowRoleAttr(kernel, load.getOperation(), kBufferRoleAttrName,
                              "tcrv_rvv.i32_load");
      if (!role)
        return role.takeError();

      llvm::Expected<RVVI32VAddDataflowValue> result =
          getLoadResultForBufferRole(kernel, *role);
      if (!result)
        return result.takeError();
      if (dataflowPlanAlreadyDefines(dataflowPlan, *result))
        return makeMicrokernelError(
            kernel, "RVV i32 microkernel dataflow body has a "
                    "duplicate load role before C emission");

      dataflowPlan.steps.push_back(makeLoadStep(*role, *result));
      bindings.push_back({load.getLoaded(), *result});
      continue;
    }

    if (auto add = llvm::dyn_cast<I32AddOp>(op)) {
      if (family.arithmetic != RVVI32MicrokernelKind::Add)
        return makeMicrokernelError(
            kernel, "tcrv_rvv.i32_add does not match the selected RVV i32 "
                    "microkernel family before C emission");
      if (add.getVl() != withVL.getVl())
        return makeMicrokernelError(
            kernel, "tcrv_rvv.i32_add must consume the !tcrv_rvv.vl token "
                    "owned by the surrounding tcrv_rvv.with_vl");
      if (llvm::Error error = requireDataflowValueLMUL(
              kernel, add.getLhs(), intrinsicConfig, "tcrv_rvv.i32_add lhs"))
        return error;
      if (llvm::Error error = requireDataflowValueLMUL(
              kernel, add.getRhs(), intrinsicConfig, "tcrv_rvv.i32_add rhs"))
        return error;
      if (llvm::Error error = requireDataflowValueLMUL(
              kernel, add.getSum(), intrinsicConfig,
              "tcrv_rvv.i32_add result"))
        return error;

      llvm::Expected<RVVI32VAddDataflowValue> lhs =
          lookupTypedDataflowValue(kernel, bindings, add.getLhs(),
                                   "tcrv_rvv.i32_add lhs");
      if (!lhs)
        return lhs.takeError();
      llvm::Expected<RVVI32VAddDataflowValue> rhs =
          lookupTypedDataflowValue(kernel, bindings, add.getRhs(),
                                   "tcrv_rvv.i32_add rhs");
      if (!rhs)
        return rhs.takeError();
      if (*lhs != RVVI32VAddDataflowValue::LHSVector ||
          *rhs != RVVI32VAddDataflowValue::RHSVector)
        return makeMicrokernelError(
            kernel, "tcrv_rvv.i32_add operands must be derived from the "
                    "preceding lhs-input-buffer and rhs-input-buffer "
                    "tcrv_rvv.i32_load ops before C emission");
      if (dataflowPlanAlreadyDefines(dataflowPlan,
                                     RVVI32VAddDataflowValue::ResultVector))
        return makeMicrokernelError(
            kernel, "tcrv_rvv.i32_vadd_microkernel dataflow body has a "
                    "duplicate add result before C emission");

      dataflowPlan.steps.push_back(
          makeAddStep(*lhs, *rhs, RVVI32VAddDataflowValue::ResultVector));
      bindings.push_back({add.getSum(), RVVI32VAddDataflowValue::ResultVector});
      continue;
    }

    if (auto sub = llvm::dyn_cast<I32SubOp>(op)) {
      if (family.arithmetic != RVVI32MicrokernelKind::Sub)
        return makeMicrokernelError(
            kernel, "tcrv_rvv.i32_sub does not match the selected RVV i32 "
                    "microkernel family before C emission");
      if (sub.getVl() != withVL.getVl())
        return makeMicrokernelError(
            kernel, "tcrv_rvv.i32_sub must consume the !tcrv_rvv.vl token "
                    "owned by the surrounding tcrv_rvv.with_vl");
      if (llvm::Error error = requireDataflowValueLMUL(
              kernel, sub.getLhs(), intrinsicConfig, "tcrv_rvv.i32_sub lhs"))
        return error;
      if (llvm::Error error = requireDataflowValueLMUL(
              kernel, sub.getRhs(), intrinsicConfig, "tcrv_rvv.i32_sub rhs"))
        return error;
      if (llvm::Error error = requireDataflowValueLMUL(
              kernel, sub.getDifference(), intrinsicConfig,
              "tcrv_rvv.i32_sub result"))
        return error;

      llvm::Expected<RVVI32VAddDataflowValue> lhs =
          lookupTypedDataflowValue(kernel, bindings, sub.getLhs(),
                                   "tcrv_rvv.i32_sub lhs");
      if (!lhs)
        return lhs.takeError();
      llvm::Expected<RVVI32VAddDataflowValue> rhs =
          lookupTypedDataflowValue(kernel, bindings, sub.getRhs(),
                                   "tcrv_rvv.i32_sub rhs");
      if (!rhs)
        return rhs.takeError();
      if (*lhs != RVVI32VAddDataflowValue::LHSVector ||
          *rhs != RVVI32VAddDataflowValue::RHSVector)
        return makeMicrokernelError(
            kernel, "tcrv_rvv.i32_sub operands must be derived from the "
                    "preceding lhs-input-buffer and rhs-input-buffer "
                    "tcrv_rvv.i32_load ops before C emission");
      if (dataflowPlanAlreadyDefines(dataflowPlan,
                                     RVVI32VAddDataflowValue::ResultVector))
        return makeMicrokernelError(
            kernel, "tcrv_rvv.i32_vsub_microkernel dataflow body has a "
                    "duplicate subtract result before C emission");

      dataflowPlan.steps.push_back(
          makeSubStep(*lhs, *rhs, RVVI32VAddDataflowValue::ResultVector));
      bindings.push_back(
          {sub.getDifference(), RVVI32VAddDataflowValue::ResultVector});
      continue;
    }

    if (auto mul = llvm::dyn_cast<I32MulOp>(op)) {
      if (family.arithmetic != RVVI32MicrokernelKind::Mul)
        return makeMicrokernelError(
            kernel, "tcrv_rvv.i32_mul does not match the selected RVV i32 "
                    "microkernel family before C emission");
      if (mul.getVl() != withVL.getVl())
        return makeMicrokernelError(
            kernel, "tcrv_rvv.i32_mul must consume the !tcrv_rvv.vl token "
                    "owned by the surrounding tcrv_rvv.with_vl");
      if (llvm::Error error = requireDataflowValueLMUL(
              kernel, mul.getLhs(), intrinsicConfig, "tcrv_rvv.i32_mul lhs"))
        return error;
      if (llvm::Error error = requireDataflowValueLMUL(
              kernel, mul.getRhs(), intrinsicConfig, "tcrv_rvv.i32_mul rhs"))
        return error;
      if (llvm::Error error = requireDataflowValueLMUL(
              kernel, mul.getProduct(), intrinsicConfig,
              "tcrv_rvv.i32_mul result"))
        return error;

      llvm::Expected<RVVI32VAddDataflowValue> lhs =
          lookupTypedDataflowValue(kernel, bindings, mul.getLhs(),
                                   "tcrv_rvv.i32_mul lhs");
      if (!lhs)
        return lhs.takeError();
      llvm::Expected<RVVI32VAddDataflowValue> rhs =
          lookupTypedDataflowValue(kernel, bindings, mul.getRhs(),
                                   "tcrv_rvv.i32_mul rhs");
      if (!rhs)
        return rhs.takeError();
      if (*lhs != RVVI32VAddDataflowValue::LHSVector ||
          *rhs != RVVI32VAddDataflowValue::RHSVector)
        return makeMicrokernelError(
            kernel, "tcrv_rvv.i32_mul operands must be derived from the "
                    "preceding lhs-input-buffer and rhs-input-buffer "
                    "tcrv_rvv.i32_load ops before C emission");
      if (dataflowPlanAlreadyDefines(dataflowPlan,
                                     RVVI32VAddDataflowValue::ResultVector))
        return makeMicrokernelError(
            kernel, "tcrv_rvv.i32_vmul_microkernel dataflow body has a "
                    "duplicate multiply result before C emission");

      dataflowPlan.steps.push_back(
          makeMulStep(*lhs, *rhs, RVVI32VAddDataflowValue::ResultVector));
      bindings.push_back(
          {mul.getProduct(), RVVI32VAddDataflowValue::ResultVector});
      continue;
    }

    if (auto store = llvm::dyn_cast<I32StoreOp>(op)) {
      if (store.getVl() != withVL.getVl())
        return makeMicrokernelError(
            kernel, "tcrv_rvv.i32_store must consume the !tcrv_rvv.vl token "
                    "owned by the surrounding tcrv_rvv.with_vl");
      if (llvm::Error error = requireDataflowValueLMUL(
              kernel, store.getValue(), intrinsicConfig,
              "tcrv_rvv.i32_store value"))
        return error;

      llvm::Expected<support::RuntimeABIParameterRole> role =
          getDataflowRoleAttr(kernel, store.getOperation(), kBufferRoleAttrName,
                              "tcrv_rvv.i32_store");
      if (!role)
        return role.takeError();
      if (*role != support::RuntimeABIParameterRole::OutputBuffer)
        return makeMicrokernelError(
            kernel, "tcrv_rvv.i32_store buffer_role must reference "
                    "'output-buffer' for this bounded RVV i32 microkernel export "
                    "route");

      llvm::Expected<RVVI32VAddDataflowValue> value =
          lookupTypedDataflowValue(kernel, bindings, store.getValue(),
                                   "tcrv_rvv.i32_store value");
      if (!value)
        return value.takeError();
      if (*value != RVVI32VAddDataflowValue::ResultVector)
        return makeMicrokernelError(
            kernel, "tcrv_rvv.i32_store value must be derived from the "
                    "preceding arithmetic result before C emission");

      dataflowPlan.steps.push_back(makeStoreStep(*role, *value));
      continue;
    }

    return makeMicrokernelError(
        kernel,
        llvm::Twine("RVV i32 microkernel dataflow body has "
                    "unexpected operation '") +
            op->getName().getStringRef() +
            "'; C emission consumes only tcrv_rvv.i32_load, "
            "tcrv_rvv.i32_add, tcrv_rvv.i32_sub, or tcrv_rvv.i32_mul, and "
            "tcrv_rvv.i32_store");
  }

  RVVI32VAddDataflowStepKind expectedArithmeticKind =
      RVVI32VAddDataflowStepKind::Mul;
  if (family.arithmetic == RVVI32MicrokernelKind::Add)
    expectedArithmeticKind = RVVI32VAddDataflowStepKind::Add;
  else if (family.arithmetic == RVVI32MicrokernelKind::Sub)
    expectedArithmeticKind = RVVI32VAddDataflowStepKind::Sub;

  if (dataflowPlan.steps.size() != 4 ||
      dataflowPlan.steps[0].kind != RVVI32VAddDataflowStepKind::Load ||
      dataflowPlan.steps[1].kind != RVVI32VAddDataflowStepKind::Load ||
      dataflowPlan.steps[2].kind != expectedArithmeticKind ||
      dataflowPlan.steps[3].kind != RVVI32VAddDataflowStepKind::Store) {
    return makeMicrokernelError(
        kernel, llvm::Twine(family.microkernelOpName) +
                    " dataflow emission plan must come from exactly the "
                    "verified load/load/" +
                    family.arithmeticVerb + "/store tcrv_rvv body order");
  }

  return llvm::Error::success();
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
  if (metadataParameters.empty())
    return makeRuntimeABICallablePlanError(
        kernel, descriptor,
        llvm::Twine(metadataSource) +
            " requires runtime_abi_parameters metadata mirroring the "
            "IR-backed callable ABI plan");

  std::size_t expectedParameterCount =
      descriptor.getCallableRuntimeABIParameters().size();
  if (irBackedParameters.size() != expectedParameterCount)
    return makeRuntimeABICallablePlanError(
        kernel, descriptor,
        llvm::Twine("IR-backed RVV binary callable ABI plan must contain "
                    "exactly ") +
            llvm::Twine(expectedParameterCount) + " parameters for family '" +
            descriptor.getArithmeticFamilyID() + "'");

  for (const support::RuntimeABIParameter &expected : irBackedParameters) {
    const support::RuntimeABIParameter *actual = nullptr;
    unsigned count = 0;
    for (const support::RuntimeABIParameter &candidate : metadataParameters) {
      if (candidate.role != expected.role)
        continue;
      actual = &candidate;
      ++count;
    }

    if (count == 0)
      return makeRuntimeABICallablePlanError(
          kernel, descriptor,
          llvm::Twine(metadataSource) +
              " requires runtime ABI parameter role '" +
              support::stringifyRuntimeABIParameterRole(expected.role) +
              "' to mirror the IR-backed callable ABI plan");
    if (count > 1)
      return makeRuntimeABICallablePlanError(
          kernel, descriptor,
          llvm::Twine(metadataSource) +
              " contains duplicate runtime ABI parameter role '" +
              support::stringifyRuntimeABIParameterRole(expected.role) + "'");

    if (actual->cName != expected.cName || actual->cType != expected.cType ||
        actual->role != expected.role ||
        actual->ownership != expected.ownership)
      return makeRuntimeABICallablePlanError(
          kernel, descriptor,
          llvm::Twine(metadataSource) + " runtime ABI parameter role '" +
              support::stringifyRuntimeABIParameterRole(expected.role) +
              "' must mirror IR-backed callable ABI parameter c_name='" +
              expected.cName + "', c_type='" + expected.cType +
              "', ownership='" +
              support::stringifyRuntimeABIParameterOwnership(
                  expected.ownership) +
              "'");
  }

  for (const support::RuntimeABIParameter &actual : metadataParameters) {
    bool expectedRole = llvm::any_of(
        irBackedParameters, [&](const support::RuntimeABIParameter &param) {
          return param.role == actual.role;
        });
    if (!expectedRole)
      return makeRuntimeABICallablePlanError(
          kernel, descriptor,
          llvm::Twine(metadataSource) +
              " contains unsupported runtime ABI parameter role '" +
              support::stringifyRuntimeABIParameterRole(actual.role) + "'");
  }

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

  mlir::Region &body = microkernel->getRegion(0);
  if (body.empty() || !llvm::hasSingleElement(body))
    return makeMicrokernelError(
        kernel, llvm::Twine(family.microkernelOpName) +
                    " requires exactly one structured RVV control-plane body "
                    "block");

  mlir::Block &block = body.front();
  if (block.getNumArguments() != 1 ||
      !block.getArgument(0).getType().isIndex()) {
    return makeMicrokernelError(
        kernel, llvm::Twine(family.microkernelOpName) +
                    " control-plane body must expose one runtime index block "
                    "argument for target/export-owned n/AVL");
  }

  SetVLOp setvl;
  WithVLOp withVL;
  unsigned setvlCount = 0;
  unsigned withVLCount = 0;
  for (mlir::Operation &bodyOp : block) {
    if (auto candidate = llvm::dyn_cast<SetVLOp>(bodyOp)) {
      setvl = candidate;
      ++setvlCount;
      continue;
    }
    if (auto candidate = llvm::dyn_cast<WithVLOp>(bodyOp)) {
      withVL = candidate;
      ++withVLCount;
      continue;
    }
    return makeMicrokernelError(
        kernel,
        llvm::Twine(family.microkernelOpName) +
            " control-plane body has unexpected operation '" +
            bodyOp.getName().getStringRef() +
            "'; exporter consumes only tcrv_rvv.setvl and tcrv_rvv.with_vl");
  }

  if (setvlCount != 1 || withVLCount != 1)
    return makeMicrokernelError(
        kernel, llvm::Twine(family.microkernelOpName) +
                    " control-plane body requires exactly one "
                    "tcrv_rvv.setvl and exactly one tcrv_rvv.with_vl");
  if (setvl.getAvl() != block.getArgument(0))
    return makeMicrokernelError(
        kernel, llvm::Twine(family.microkernelOpName) +
                    " control-plane body requires setvl AVL to come from the "
                    "runtime index body argument, not descriptor-local "
                    "element_count or a constant");
  if (withVL.getVl() != setvl.getVl())
    return makeMicrokernelError(
        kernel, llvm::Twine(family.microkernelOpName) +
                    " control-plane body requires with_vl to consume the "
                    "!tcrv_rvv.vl token produced by setvl");

  RVVBinaryIntrinsicDescriptor descriptor =
      getI32BinaryIntrinsicDescriptorForMicrokernel(family, selectedConfig);
  llvm::Expected<RVVIntrinsicConfig> config = buildRVVIntrinsicConfig(
      kernel, activeRouteID, descriptor, setvl, withVL, expectedPolicy,
      selectedConfig);
  if (!config)
    return config.takeError();

  mlir::Region &withVLBody = withVL.getBody();
  if (withVLBody.empty() || !llvm::hasSingleElement(withVLBody))
    return makeMicrokernelError(
        kernel, llvm::Twine(family.microkernelOpName) +
                    " control-plane with_vl body must be present for this "
                    "bounded i32 RVV export slice");
  if (withVLBody.front().getNumArguments() != 0)
    return makeMicrokernelError(
        kernel, llvm::Twine(family.microkernelOpName) +
                    " control-plane with_vl body must not have block "
                    "arguments");

  llvm::SmallVector<mlir::Operation *, 4> dataflowOps;
  for (mlir::Operation &withVLOp : withVLBody.front())
    dataflowOps.push_back(&withVLOp);
  if (dataflowOps.size() != 4)
    return makeMicrokernelError(
        kernel, llvm::Twine(family.microkernelOpName) +
                    " control-plane with_vl body requires exactly the finite "
                    "tcrv_rvv.i32_load, tcrv_rvv.i32_load, " +
                    family.arithmeticOpName +
                    ", tcrv_rvv.i32_store dataflow sequence");

  auto lhsLoad = llvm::dyn_cast<I32LoadOp>(dataflowOps[0]);
  auto rhsLoad = llvm::dyn_cast<I32LoadOp>(dataflowOps[1]);
  auto store = llvm::dyn_cast<I32StoreOp>(dataflowOps[3]);
  mlir::Value arithmeticLHS;
  mlir::Value arithmeticRHS;
  mlir::Value arithmeticVL;
  mlir::Value arithmeticResult;
  if (auto add = llvm::dyn_cast<I32AddOp>(dataflowOps[2])) {
    if (family.arithmetic != RVVI32MicrokernelKind::Add)
      return makeMicrokernelError(
          kernel, "RVV microkernel arithmetic op does not match selected "
                  "family");
    arithmeticLHS = add.getLhs();
    arithmeticRHS = add.getRhs();
    arithmeticVL = add.getVl();
    arithmeticResult = add.getSum();
  } else if (auto sub = llvm::dyn_cast<I32SubOp>(dataflowOps[2])) {
    if (family.arithmetic != RVVI32MicrokernelKind::Sub)
      return makeMicrokernelError(
          kernel, "RVV microkernel arithmetic op does not match selected "
                  "family");
    arithmeticLHS = sub.getLhs();
    arithmeticRHS = sub.getRhs();
    arithmeticVL = sub.getVl();
    arithmeticResult = sub.getDifference();
  } else if (auto mul = llvm::dyn_cast<I32MulOp>(dataflowOps[2])) {
    if (family.arithmetic != RVVI32MicrokernelKind::Mul)
      return makeMicrokernelError(
          kernel, "RVV microkernel arithmetic op does not match selected "
                  "family");
    arithmeticLHS = mul.getLhs();
    arithmeticRHS = mul.getRhs();
    arithmeticVL = mul.getVl();
    arithmeticResult = mul.getProduct();
  }
  if (!lhsLoad || !rhsLoad || !arithmeticResult || !store)
    return makeMicrokernelError(
        kernel, llvm::Twine(family.microkernelOpName) +
                    " control-plane with_vl body requires exactly the finite "
                    "tcrv_rvv.i32_load, tcrv_rvv.i32_load, " +
                    family.arithmeticOpName +
                    ", tcrv_rvv.i32_store dataflow sequence");

  if (lhsLoad.getVl() != withVL.getVl() ||
      rhsLoad.getVl() != withVL.getVl() ||
      arithmeticVL != withVL.getVl() ||
      store.getVl() != withVL.getVl())
    return makeMicrokernelError(
        kernel, llvm::Twine(family.microkernelOpName) +
                    " requires every finite RVV i32 dataflow op to consume "
                    "the !tcrv_rvv.vl token owned by with_vl");
  if (arithmeticLHS != lhsLoad.getLoaded() ||
      arithmeticRHS != rhsLoad.getLoaded() ||
      store.getValue() != arithmeticResult)
    return makeMicrokernelError(
        kernel, llvm::Twine(family.microkernelOpName) +
                    " requires finite RVV i32 dataflow SSA chain "
                    "lhs-load,rhs-load -> " +
                    family.arithmeticVerb + " -> store");

  if (llvm::Error error =
          buildDataflowEmissionPlanFromTypedBody(kernel, withVL, dataflowOps,
                                                 family, *config,
                                                 dataflowPlan))
    return error;

  intrinsicConfig = std::move(*config);
  controlPlaneSEW = intrinsicConfig.sew;
  controlPlaneLMUL = intrinsicConfig.lmul;
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

  auto elementCountAttr =
      microkernel->getAttrOfType<mlir::IntegerAttr>(kElementCountAttrName);
  if (!elementCountAttr)
    return makeMicrokernelError(
        kernel, llvm::Twine(descriptor.getRVVMicrokernelOpName()) +
                    " requires integer element_count metadata");
  elementCount = elementCountAttr.getInt();
  if (elementCount <= 0 || elementCount > 64)
    return makeMicrokernelError(
        kernel, llvm::Twine(descriptor.getRVVMicrokernelOpName()) +
                    " element_count must be in the bounded smoke range "
                    "[1, 64]");

  mlir::Region &body = microkernel->getRegion(0);
  if (body.empty() || !llvm::hasSingleElement(body))
    return makeMicrokernelError(
        kernel, llvm::Twine(descriptor.getRVVMicrokernelOpName()) +
                    " requires exactly one structured RVV control-plane body "
                    "block");

  mlir::Block &block = body.front();
  if (block.getNumArguments() != 1 ||
      !block.getArgument(0).getType().isIndex())
    return makeMicrokernelError(
        kernel, llvm::Twine(descriptor.getRVVMicrokernelOpName()) +
                    " control-plane body must expose one runtime index block "
                    "argument for target/export-owned n/AVL");

  SetVLOp setvl;
  WithVLOp withVL;
  unsigned setvlCount = 0;
  unsigned withVLCount = 0;
  for (mlir::Operation &bodyOp : block) {
    if (auto candidate = llvm::dyn_cast<SetVLOp>(bodyOp)) {
      setvl = candidate;
      ++setvlCount;
      continue;
    }
    if (auto candidate = llvm::dyn_cast<WithVLOp>(bodyOp)) {
      withVL = candidate;
      ++withVLCount;
      continue;
    }
    return makeMicrokernelError(
        kernel,
        llvm::Twine(descriptor.getRVVMicrokernelOpName()) +
            " control-plane body has unexpected operation '" +
            bodyOp.getName().getStringRef() +
            "'; exporter consumes only tcrv_rvv.setvl and tcrv_rvv.with_vl");
  }

  if (setvlCount != 1 || withVLCount != 1)
    return makeMicrokernelError(
        kernel, llvm::Twine(descriptor.getRVVMicrokernelOpName()) +
                    " control-plane body requires exactly one "
                    "tcrv_rvv.setvl and exactly one tcrv_rvv.with_vl");
  if (setvl.getAvl() != block.getArgument(0))
    return makeMicrokernelError(
        kernel, llvm::Twine(descriptor.getRVVMicrokernelOpName()) +
                    " control-plane body requires setvl AVL to come from the "
                    "runtime index body argument, not descriptor-local "
                    "element_count or a constant");
  if (withVL.getVl() != setvl.getVl())
    return makeMicrokernelError(
        kernel, llvm::Twine(descriptor.getRVVMicrokernelOpName()) +
                    " control-plane body requires with_vl to consume the "
                    "!tcrv_rvv.vl token produced by setvl");

  llvm::Expected<RVVIntrinsicConfig> config = buildRVVIntrinsicConfig(
      kernel, activeRouteID, descriptor, setvl, withVL, expectedPolicy,
      selectedConfig);
  if (!config)
    return config.takeError();

  mlir::Region &withVLBody = withVL.getBody();
  if (withVLBody.empty() || !llvm::hasSingleElement(withVLBody))
    return makeMicrokernelError(
        kernel, llvm::Twine(descriptor.getRVVMicrokernelOpName()) +
                    " control-plane with_vl body must be present for this "
                    "bounded i64 RVV export slice");
  if (withVLBody.front().getNumArguments() != 0)
    return makeMicrokernelError(
        kernel, llvm::Twine(descriptor.getRVVMicrokernelOpName()) +
                    " control-plane with_vl body must not have block "
                    "arguments");

  llvm::SmallVector<mlir::Operation *, 4> dataflowOps;
  for (mlir::Operation &withVLOp : withVLBody.front())
    dataflowOps.push_back(&withVLOp);
  if (dataflowOps.size() != 4)
    return makeMicrokernelError(
        kernel, llvm::Twine(descriptor.getRVVMicrokernelOpName()) +
                    " control-plane with_vl body requires exactly the finite "
                    "tcrv_rvv.i64_load, tcrv_rvv.i64_load, " +
                    descriptor.getRVVOperationName() +
                    ", tcrv_rvv.i64_store dataflow sequence");

  auto lhsLoad = llvm::dyn_cast<I64LoadOp>(dataflowOps[0]);
  auto rhsLoad = llvm::dyn_cast<I64LoadOp>(dataflowOps[1]);
  auto store = llvm::dyn_cast<I64StoreOp>(dataflowOps[3]);
  mlir::Value arithmeticLHS;
  mlir::Value arithmeticRHS;
  mlir::Value arithmeticVL;
  mlir::Value arithmeticResult;
  if (auto add = llvm::dyn_cast<I64AddOp>(dataflowOps[2])) {
    if (descriptor.family.arithmetic != RVVI32MicrokernelKind::Add)
      return makeMicrokernelError(
          kernel, "RVV i64 microkernel arithmetic op does not match selected "
                  "family");
    arithmeticLHS = add.getLhs();
    arithmeticRHS = add.getRhs();
    arithmeticVL = add.getVl();
    arithmeticResult = add.getSum();
  } else if (auto sub = llvm::dyn_cast<I64SubOp>(dataflowOps[2])) {
    if (descriptor.family.arithmetic != RVVI32MicrokernelKind::Sub)
      return makeMicrokernelError(
          kernel, "RVV i64 microkernel arithmetic op does not match selected "
                  "family");
    arithmeticLHS = sub.getLhs();
    arithmeticRHS = sub.getRhs();
    arithmeticVL = sub.getVl();
    arithmeticResult = sub.getDifference();
  } else if (auto mul = llvm::dyn_cast<I64MulOp>(dataflowOps[2])) {
    if (descriptor.family.arithmetic != RVVI32MicrokernelKind::Mul)
      return makeMicrokernelError(
          kernel, "RVV i64 microkernel arithmetic op does not match selected "
                  "family");
    arithmeticLHS = mul.getLhs();
    arithmeticRHS = mul.getRhs();
    arithmeticVL = mul.getVl();
    arithmeticResult = mul.getProduct();
  }
  if (!lhsLoad || !rhsLoad || !arithmeticResult || !store)
    return makeMicrokernelError(
        kernel, llvm::Twine(descriptor.getRVVMicrokernelOpName()) +
                    " control-plane with_vl body requires exactly the finite "
                    "tcrv_rvv.i64_load, tcrv_rvv.i64_load, " +
                    descriptor.getRVVOperationName() +
                    ", tcrv_rvv.i64_store dataflow sequence");

  if (lhsLoad.getVl() != withVL.getVl() ||
      rhsLoad.getVl() != withVL.getVl() ||
      arithmeticVL != withVL.getVl() ||
      store.getVl() != withVL.getVl())
    return makeMicrokernelError(
        kernel, llvm::Twine(descriptor.getRVVMicrokernelOpName()) +
                    " requires every finite RVV i64 dataflow op to consume "
                    "the !tcrv_rvv.vl token owned by with_vl");
  if (arithmeticLHS != lhsLoad.getLoaded() ||
      arithmeticRHS != rhsLoad.getLoaded() ||
      store.getValue() != arithmeticResult)
    return makeMicrokernelError(
        kernel, llvm::Twine(descriptor.getRVVMicrokernelOpName()) +
                    " requires finite RVV i64 dataflow SSA chain "
                    "lhs-load,rhs-load -> " +
                    descriptor.family.arithmeticVerb + " -> store");

  if (llvm::Error error = requireDataflowValueLMUL(
          kernel, lhsLoad.getLoaded(), *config, "tcrv_rvv.i64_load result"))
    return error;
  if (llvm::Error error = requireDataflowValueLMUL(
          kernel, rhsLoad.getLoaded(), *config, "tcrv_rvv.i64_load result"))
    return error;
  std::string arithmeticLHSContext =
      (llvm::Twine(descriptor.getRVVOperationName()) + " lhs").str();
  std::string arithmeticRHSContext =
      (llvm::Twine(descriptor.getRVVOperationName()) + " rhs").str();
  std::string arithmeticResultContext =
      (llvm::Twine(descriptor.getRVVOperationName()) + " result").str();
  if (llvm::Error error = requireDataflowValueLMUL(
          kernel, arithmeticLHS, *config, arithmeticLHSContext))
    return error;
  if (llvm::Error error = requireDataflowValueLMUL(
          kernel, arithmeticRHS, *config, arithmeticRHSContext))
    return error;
  if (llvm::Error error = requireDataflowValueLMUL(
          kernel, arithmeticResult, *config, arithmeticResultContext))
    return error;
  if (llvm::Error error = requireDataflowValueLMUL(
          kernel, store.getValue(), *config, "tcrv_rvv.i64_store value"))
    return error;

  llvm::Expected<support::RuntimeABIParameterRole> lhsRole =
      getDataflowRoleAttr(kernel, lhsLoad.getOperation(), kBufferRoleAttrName,
                          "tcrv_rvv.i64_load");
  if (!lhsRole)
    return lhsRole.takeError();
  if (*lhsRole != support::RuntimeABIParameterRole::LHSInputBuffer)
    return makeMicrokernelError(
        kernel, "first tcrv_rvv.i64_load buffer_role must reference "
                "'lhs-input-buffer'");

  llvm::Expected<support::RuntimeABIParameterRole> rhsRole =
      getDataflowRoleAttr(kernel, rhsLoad.getOperation(), kBufferRoleAttrName,
                          "tcrv_rvv.i64_load");
  if (!rhsRole)
    return rhsRole.takeError();
  if (*rhsRole != support::RuntimeABIParameterRole::RHSInputBuffer)
    return makeMicrokernelError(
        kernel, "second tcrv_rvv.i64_load buffer_role must reference "
                "'rhs-input-buffer'");

  llvm::Expected<support::RuntimeABIParameterRole> storeRole =
      getDataflowRoleAttr(kernel, store.getOperation(), kBufferRoleAttrName,
                          "tcrv_rvv.i64_store");
  if (!storeRole)
    return storeRole.takeError();
  if (*storeRole != support::RuntimeABIParameterRole::OutputBuffer)
    return makeMicrokernelError(
        kernel, "tcrv_rvv.i64_store buffer_role must reference "
                "'output-buffer'");

  dataflowPlan.steps.clear();
  dataflowPlan.steps.push_back(
      makeLoadStep(*lhsRole, RVVI32VAddDataflowValue::LHSVector));
  dataflowPlan.steps.push_back(
      makeLoadStep(*rhsRole, RVVI32VAddDataflowValue::RHSVector));
  switch (descriptor.family.arithmetic) {
  case RVVI32MicrokernelKind::Add:
    dataflowPlan.steps.push_back(
        makeAddStep(RVVI32VAddDataflowValue::LHSVector,
                    RVVI32VAddDataflowValue::RHSVector,
                    RVVI32VAddDataflowValue::ResultVector));
    break;
  case RVVI32MicrokernelKind::Sub:
    dataflowPlan.steps.push_back(
        makeSubStep(RVVI32VAddDataflowValue::LHSVector,
                    RVVI32VAddDataflowValue::RHSVector,
                    RVVI32VAddDataflowValue::ResultVector));
    break;
  case RVVI32MicrokernelKind::Mul:
    dataflowPlan.steps.push_back(
        makeMulStep(RVVI32VAddDataflowValue::LHSVector,
                    RVVI32VAddDataflowValue::RHSVector,
                    RVVI32VAddDataflowValue::ResultVector));
    break;
  }
  dataflowPlan.steps.push_back(
      makeStoreStep(*storeRole, RVVI32VAddDataflowValue::ResultVector));

  intrinsicConfig = std::move(*config);
  controlPlaneSEW = intrinsicConfig.sew;
  controlPlaneLMUL = intrinsicConfig.lmul;
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
  llvm::SmallVector<MemWindowOp, 3> windows;
  if (llvm::Error error = support::collectRuntimeABIBufferMemWindows(
          kernel, descriptor.getBufferMemWindowSpecs(), windows))
    return error;

  llvm::SmallVector<RuntimeParamOp, 1> runtimeParams;
  if (llvm::Error error = support::collectRuntimeABIParams(
          kernel, descriptor.getRuntimeElementCountParamSpecs(/*cName=*/""),
          runtimeParams))
    return error;

  llvm::SmallVector<support::RuntimeABIParameter, 4> descriptorParameters =
      descriptor.getCallableRuntimeABIParameters(getAttrValue(
          runtimeParams.front().getOperation(),
          support::kRuntimeParamCNameAttrName));

  parameters.clear();
  for (auto [index, window] : llvm::enumerate(windows)) {
    llvm::StringRef cType =
        getAttrValue(window.getOperation(), support::kMemWindowCTypeAttrName);
    llvm::StringRef ownership = getAttrValue(
        window.getOperation(), support::kMemWindowOwnershipAttrName);
    std::optional<support::RuntimeABIParameterOwnership> parsedOwnership =
        support::symbolizeRuntimeABIParameterOwnership(ownership);
    if (!parsedOwnership)
      return makeMicrokernelError(
          kernel, llvm::Twine("RVV binary callable ABI mem_window @") +
                      window.getSymName() + " has unsupported ownership '" +
                      ownership + "'");
    parameters.push_back(support::RuntimeABIParameter(
        descriptorParameters[index].cName, cType,
        descriptorParameters[index].role, *parsedOwnership));
  }

  RuntimeParamOp runtimeParam = runtimeParams.front();
  llvm::StringRef runtimeOwnership = getAttrValue(
      runtimeParam.getOperation(), support::kRuntimeParamOwnershipAttrName);
  std::optional<support::RuntimeABIParameterOwnership> parsedOwnership =
      support::symbolizeRuntimeABIParameterOwnership(runtimeOwnership);
  if (!parsedOwnership)
    return makeMicrokernelError(
        kernel, llvm::Twine("RVV binary callable ABI runtime_param @") +
                    runtimeParam.getSymName() +
                    " has unsupported ownership '" + runtimeOwnership + "'");
  parameters.push_back(support::RuntimeABIParameter(
      getAttrValue(runtimeParam.getOperation(),
                   support::kRuntimeParamCNameAttrName),
      getAttrValue(runtimeParam.getOperation(),
                   support::kRuntimeParamCTypeAttrName),
      support::RuntimeABIParameterRole::RuntimeElementCount, *parsedOwnership));

  bufferWindows.append(windows.begin(), windows.end());
  runtimeElementCountParam = runtimeParam;
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

  const RVVBinaryFamilyDescriptor *rvvBinaryFamily = nullptr;
  if (auto descriptorAttr = getPathVariant(path)->getAttrOfType<mlir::StringAttr>(
          kRVVLoweringDescriptorAttrName)) {
    const RVVBinaryFamilyDescriptor *candidateFamily =
        lookupRVVBinaryFamilyByLoweringDescriptor(
        descriptorAttr.getValue().trim());
    if (candidateFamily && candidateFamily->dtype == RVVBinaryDTypeKind::I64)
      rvvBinaryFamily = candidateFamily;
  }

  if (rvvBinaryFamily) {
    RVVBinaryIntrinsicDescriptor descriptor =
        getRVVBinaryIntrinsicDescriptor(*rvvBinaryFamily, **selectedConfig);
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

llvm::Expected<RVVMicrokernelRecord>
buildModuleRecordForFamily(mlir::ModuleOp module,
                           RVVI32MicrokernelKind expectedFamily,
                           llvm::StringRef routeID) {
  llvm::Expected<RVVMicrokernelRecord> record =
      buildModuleRecord(module, routeID);
  if (!record)
    return record.takeError();

  const RVVI32MicrokernelFamilySpec &expected =
      getI32MicrokernelFamilySpec(expectedFamily);
  if (!record->family || record->family->arithmetic != expected.arithmetic) {
    llvm::StringRef actual =
        record->family ? record->family->microkernelOpName : "<missing>";
    return makeModuleMicrokernelError(
        llvm::Twine("route '") + routeID + "' requires " +
        expected.microkernelOpName + " but the selected RVV record is " +
        actual);
  }

  return std::move(*record);
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

struct RVVBinaryCallableRuntimeABIParameterBindings {
  const support::RuntimeABIParameter *lhs = nullptr;
  const support::RuntimeABIParameter *rhs = nullptr;
  const support::RuntimeABIParameter *out = nullptr;
  const support::RuntimeABIParameter *runtimeElementCount = nullptr;
};

llvm::Expected<RVVBinaryCallableRuntimeABIParameterBindings>
bindRVVBinaryCallableRuntimeABIParametersByRole(
    llvm::ArrayRef<support::RuntimeABIParameter> parameters,
    llvm::StringRef context) {
  RVVBinaryCallableRuntimeABIParameterBindings bindings;
  for (const support::RuntimeABIParameter &parameter : parameters) {
    const support::RuntimeABIParameter **slot = nullptr;
    switch (parameter.role) {
    case support::RuntimeABIParameterRole::LHSInputBuffer:
      slot = &bindings.lhs;
      break;
    case support::RuntimeABIParameterRole::RHSInputBuffer:
      slot = &bindings.rhs;
      break;
    case support::RuntimeABIParameterRole::OutputBuffer:
      slot = &bindings.out;
      break;
    case support::RuntimeABIParameterRole::RuntimeElementCount:
      slot = &bindings.runtimeElementCount;
      break;
    case support::RuntimeABIParameterRole::DispatchAvailabilityGuard:
      return makeModuleMicrokernelError(
          llvm::Twine(context) +
          " does not accept dispatch availability guard ABI parameters");
    }

    if (*slot)
      return makeModuleMicrokernelError(
          llvm::Twine(context) + " has duplicate ABI parameter role '" +
          support::stringifyRuntimeABIParameterRole(parameter.role) + "'");
    *slot = &parameter;
  }

  if (!bindings.lhs || !bindings.rhs || !bindings.out ||
      !bindings.runtimeElementCount)
    return makeModuleMicrokernelError(
        llvm::Twine(context) +
        " requires lhs, rhs, output, and runtime element-count ABI roles");
  return bindings;
}

const support::RuntimeABIParameter *lookupBoundBufferParameter(
    const RVVBinaryCallableRuntimeABIParameterBindings &bindings,
    support::RuntimeABIParameterRole role) {
  switch (role) {
  case support::RuntimeABIParameterRole::LHSInputBuffer:
    return bindings.lhs;
  case support::RuntimeABIParameterRole::RHSInputBuffer:
    return bindings.rhs;
  case support::RuntimeABIParameterRole::OutputBuffer:
    return bindings.out;
  case support::RuntimeABIParameterRole::RuntimeElementCount:
  case support::RuntimeABIParameterRole::DispatchAvailabilityGuard:
    return nullptr;
  }
  return nullptr;
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
                        llvm::StringRef functionName) {
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
  os << "/* arithmetic_c_operator: " << record.descriptor.getCOperator()
     << " */\n";
  os << "/* active_route: " << record.activeRouteID << " */\n";
  os << "/* control_plane_body: tcrv_rvv.setvl -> tcrv_rvv.with_vl */\n";
  os << "/* control_plane_runtime_avl: body index argument maps to "
        "target/export-owned runtime n ABI parameter */\n";
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
        "store.buffer_role=output-buffer; runtime n remains the "
        "target/export-owned runtime element-count ABI parameter */\n";
  printDataflowPlanMetadata(os, record.dataflowPlan, record.descriptor);
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

llvm::Error printMicrokernelFunction(
    llvm::raw_ostream &os, llvm::StringRef functionName,
    llvm::ArrayRef<support::RuntimeABIParameter> parameters,
    const RVVBinaryIntrinsicDescriptor &descriptor,
    const RVVIntrinsicConfig &intrinsicConfig,
    const RVVI32VAddDataflowEmissionPlan &dataflowPlan) {
  if (dataflowPlan.steps.size() != 4)
    return makeModuleMicrokernelError(
        "validated RVV dataflow emission plan requires exactly four "
        "load/load/arithmetic/store steps");

  llvm::Expected<RVVBinaryCallableRuntimeABIParameterBindings>
      bindings = bindRVVBinaryCallableRuntimeABIParametersByRole(
          parameters, "RVV direct microkernel C emission");
  if (!bindings)
    return bindings.takeError();

  const support::RuntimeABIParameter &runtimeN =
      *bindings->runtimeElementCount;

  os << "void " << functionName << "(";
  for (auto [index, parameter] : llvm::enumerate(parameters)) {
    if (index != 0)
      os << ", ";
    support::printRuntimeABIParameterCDeclaration(os, parameter);
  }
  os << ") {\n";
  os << "  size_t offset = 0;\n";
  os << "  while (offset < " << runtimeN.cName << ") {\n";
  os << "    size_t vl = " << intrinsicConfig.setvlIntrinsicName << "("
     << runtimeN.cName << " - offset);\n";
  for (const RVVI32VAddDataflowStep &step : dataflowPlan.steps) {
    switch (step.kind) {
    case RVVI32VAddDataflowStepKind::Load: {
      const support::RuntimeABIParameter *parameter =
          lookupBoundBufferParameter(*bindings, step.bufferRole);
      if (!parameter)
        return makeModuleMicrokernelError(
            "RVV dataflow load step references a non-buffer ABI role");
      os << "    " << intrinsicConfig.vectorType << " "
         << getDataflowValueCName(step.result, descriptor)
         << " = " << intrinsicConfig.loadIntrinsicName << "(&"
         << parameter->cName << "[offset], vl);\n";
      break;
    }
    case RVVI32VAddDataflowStepKind::Add:
    case RVVI32VAddDataflowStepKind::Sub:
    case RVVI32VAddDataflowStepKind::Mul:
      os << "    " << intrinsicConfig.vectorType << " "
         << getDataflowValueCName(step.result, descriptor)
         << " = " << intrinsicConfig.arithmeticIntrinsicName << "("
         << getDataflowValueCName(step.lhs, descriptor) << ", "
         << getDataflowValueCName(step.rhs, descriptor) << ", vl);\n";
      break;
    case RVVI32VAddDataflowStepKind::Store: {
      const support::RuntimeABIParameter *parameter =
          lookupBoundBufferParameter(*bindings, step.bufferRole);
      if (!parameter)
        return makeModuleMicrokernelError(
            "RVV dataflow store step references a non-buffer ABI role");
      os << "    " << intrinsicConfig.storeIntrinsicName << "(&"
         << parameter->cName << "[offset], "
         << getDataflowValueCName(step.value, descriptor) << ", vl);\n";
      break;
    }
    }
  }
  os << "    offset += vl;\n";
  os << "  }\n";
  os << "}\n\n";
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
                                      const RVVBinaryIntrinsicDescriptor
                                          &descriptor,
                                      const RVVIntrinsicConfig &intrinsicConfig,
                                      std::int64_t elementCount) {
  os << "/* Harness capacity comes from descriptor-local element_count; each "
        "call still supplies runtime n through the generated C ABI. */\n";
  os << "static int " << functionName
     << "_self_check_one(size_t runtime_n) {\n";
  os << "  enum { kTCRVMicrokernelCapacity = " << elementCount << " };\n";
  os << "  int32_t lhs[kTCRVMicrokernelCapacity];\n";
  os << "  int32_t rhs[kTCRVMicrokernelCapacity];\n";
  os << "  int32_t out[kTCRVMicrokernelCapacity];\n\n";
  os << "  if (runtime_n == 0 || runtime_n > (size_t)kTCRVMicrokernelCapacity) "
        "{\n";
  os << "    fprintf(stderr, \"invalid rvv microkernel runtime_n=%zu\\n\", "
        "runtime_n);\n";
  os << "    return 1;\n";
  os << "  }\n\n";
  os << "  for (size_t index = 0; index < (size_t)kTCRVMicrokernelCapacity; "
        "++index) {\n";
  os << "    lhs[index] = index + 1;\n";
  os << "    rhs[index] = 100 - index;\n";
  os << "    out[index] = -12345;\n";
  os << "  }\n\n";
  os << "  size_t first_vl = " << intrinsicConfig.setvlIntrinsicName
     << "(runtime_n);\n";
  os << "  if (first_vl == 0 || first_vl > runtime_n) {\n";
  os << "    fprintf(stderr, \"invalid rvv microkernel vl=%zu\\n\", first_vl);\n";
  os << "    return 2;\n";
  os << "  }\n\n";
  os << "  " << functionName << "(lhs, rhs, out, runtime_n);\n\n";
  os << "  for (size_t index = 0; index < runtime_n; ++index) {\n";
  os << "    int32_t expected = "
     << descriptor.getCArithmeticCheckExpression("lhs[index]", "rhs[index]")
     << ";\n";
  os << "    if (out[index] != expected) {\n";
  os << "      fprintf(stderr, \"rvv microkernel mismatch at %zu\\n\", index);\n";
  os << "      return 3;\n";
  os << "    }\n";
  os << "  }\n";
  os << "  for (size_t index = runtime_n; "
        "index < (size_t)kTCRVMicrokernelCapacity; ++index) {\n";
  os << "    if (out[index] != -12345) {\n";
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

  os << "/* TianChen-RV RVV runtime-callable microkernel C export. */\n";
  os << "/* Scope: library-style C source for exactly one "
     << record.descriptor.getRVVMicrokernelOpName() << ". */\n";
  os << "/* Default artifact shape: runtime-callable C ABI function with no "
        "embedded main or self-check harness. */\n";
  if (includeHarness)
    os << "/* Harness mode: adds a bounded self-check main for explicit ssh rvv "
          "evidence only. */\n";
  os << "/* Correctness claims require the explicit self-check harness and ssh "
        "rvv evidence; this source is not generic TianChen-RV lowering or "
        "performance evidence. */\n\n";
  os << "#include <stddef.h>\n";
  os << "#include <stdint.h>\n";
  if (includeHarness)
    os << "#include <stdio.h>\n";
  os << "#include <riscv_vector.h>\n\n";

  printRecordComment(os, record, functionName);
  if (llvm::Error error =
          printMicrokernelFunction(os, functionName,
                                   record.runtimeABIParameters,
                                   record.descriptor, record.intrinsicConfig,
                                   record.dataflowPlan))
    return error;
  if (includeHarness) {
    if (record.descriptor.family.dtype != RVVBinaryDTypeKind::I32)
      return makeModuleMicrokernelError(
          "RVV self-check harness export is currently bounded to i32 "
          "microkernel records");
    printMicrokernelSelfCheckHarness(os, functionName, record.descriptor,
                                     record.intrinsicConfig,
                                     record.elementCount);
  }
  return llvm::Error::success();
}

llvm::Error validateRVVMicrokernelSourceCandidate(
    const TargetArtifactCandidate &candidate) {
  if (const RVVBinaryFamilyDescriptor *i64Family =
          getI64MicrokernelFamilyForSourceRoute(candidate.routeID)) {
    RVVBinaryIntrinsicDescriptor descriptor =
        getRVVBinaryIntrinsicDescriptor(*i64Family, getI64M1VectorShapeConfig());
    if (!candidateMatchesRVVBinaryDescriptor(candidate, descriptor))
      return makeModuleMicrokernelError(
          llvm::Twine("target artifact route '") + candidate.routeID +
          "' does not match supported RVV i64 microkernel ABI metadata; "
          "expected emission_kind '" +
          descriptor.family.emissionKind + "', artifact_kind '" +
          kMicrokernelArtifactKind + "', runtime_abi '" +
          descriptor.getRVVRuntimeABI() + "', runtime_abi_kind '" +
          descriptor.getRVVRuntimeABIKind() + "', runtime_abi_name '" +
          descriptor.getRVVRuntimeABIName() + "', runtime_glue_role '" +
          descriptor.getRVVRuntimeGlueRole() + "'");

    TargetArtifactExporter sourceExporter(
        descriptor.getRVVRouteID(), kMicrokernelArtifactKind,
        kRVVPluginName, descriptor.family.emissionKind, exportRVVMicrokernelC,
        descriptor.getCallableRuntimeABIRoleRequirements(),
        /*directHelperRoute=*/true, /*handoffKind=*/{},
        /*candidateValidationFn=*/nullptr,
        descriptor.getRVVExternalABIComponentGroup(),
        descriptor.getRVVRuntimeABIName());
    if (llvm::Error error = validateTargetArtifactCandidateAgainstExporter(
            candidate, sourceExporter))
      return error;
    return validateRVVBinaryCandidateRuntimeABIMirrorsIR(candidate, descriptor);
  }

  const RVVI32MicrokernelFamilySpec *family =
      getI32MicrokernelFamilyForSourceRoute(candidate.routeID);
  if (!family)
    return makeModuleMicrokernelError(
        llvm::Twine("target artifact route '") + candidate.routeID +
        "' is not a supported RVV i32 or i64 microkernel source route");

  if (!candidateMatchesRVVMicrokernelFamily(candidate, *family))
    return makeModuleMicrokernelError(
        llvm::Twine("target artifact route '") + candidate.routeID +
        "' does not match supported RVV i32 microkernel family ABI metadata; "
        "expected emission_kind '" +
        family->emissionKind + "', artifact_kind '" +
        kMicrokernelArtifactKind + "', runtime_abi '" + family->runtimeABI +
        "', runtime_abi_kind '" + family->runtimeABIKind +
        "', runtime_abi_name '" + family->runtimeABIName +
        "', runtime_glue_role '" + family->runtimeGlueRole + "'");

  TargetArtifactExporter sourceExporter(
      family->routeID, kMicrokernelArtifactKind, kRVVPluginName,
      family->emissionKind, exportRVVMicrokernelC,
      getRVVBinaryCallableRuntimeABIRoleRequirements(*family),
      family->arithmetic == RVVI32MicrokernelKind::Add,
      /*handoffKind=*/{}, /*candidateValidationFn=*/nullptr,
      family->externalABIComponentGroup, family->runtimeABIName);
  if (llvm::Error error =
          validateTargetArtifactCandidateAgainstExporter(candidate,
                                                        sourceExporter))
    return error;
  RVVBinaryIntrinsicDescriptor descriptor =
      getRVVBinaryIntrinsicDescriptor(*family, getI32M1VectorShapeConfig());
  return validateRVVBinaryCandidateRuntimeABIMirrorsIR(candidate, descriptor);
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
  return candidateMatchesRVVBinaryDescriptor(candidates.front(), descriptor);
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

static bool isLegacyGenericRVVMicrokernelDirectRoute(
    const RVVMicrokernelDirectRouteManifestEntry &route) {
  return route.family &&
         isSameRVVBinaryFamily(*route.family, getI32VAddFamilyDescriptor());
}

llvm::ArrayRef<RVVMicrokernelDirectRouteManifestEntry>
getRVVMicrokernelDirectRouteManifest() {
  static const RVVMicrokernelDirectRouteManifestEntry routes[] = {
      {&getI32VAddFamilyDescriptor(), RVVMicrokernelDirectRouteKind::Source},
      {&getI32VAddFamilyDescriptor(), RVVMicrokernelDirectRouteKind::Header},
      {&getI32VAddFamilyDescriptor(), RVVMicrokernelDirectRouteKind::Object},
      {&getI32VSubFamilyDescriptor(), RVVMicrokernelDirectRouteKind::Source},
      {&getI32VSubFamilyDescriptor(), RVVMicrokernelDirectRouteKind::Header},
      {&getI32VSubFamilyDescriptor(), RVVMicrokernelDirectRouteKind::Object},
      {&getI32VMulFamilyDescriptor(), RVVMicrokernelDirectRouteKind::Source},
      {&getI32VMulFamilyDescriptor(), RVVMicrokernelDirectRouteKind::Header},
      {&getI32VMulFamilyDescriptor(), RVVMicrokernelDirectRouteKind::Object},
      {&getI64VAddFamilyDescriptor(), RVVMicrokernelDirectRouteKind::Source},
      {&getI64VAddFamilyDescriptor(), RVVMicrokernelDirectRouteKind::Header},
      {&getI64VAddFamilyDescriptor(), RVVMicrokernelDirectRouteKind::Object},
      {&getI64VSubFamilyDescriptor(), RVVMicrokernelDirectRouteKind::Source},
      {&getI64VSubFamilyDescriptor(), RVVMicrokernelDirectRouteKind::Header},
      {&getI64VSubFamilyDescriptor(), RVVMicrokernelDirectRouteKind::Object},
      {&getI64VMulFamilyDescriptor(), RVVMicrokernelDirectRouteKind::Source},
      {&getI64VMulFamilyDescriptor(), RVVMicrokernelDirectRouteKind::Header},
      {&getI64VMulFamilyDescriptor(), RVVMicrokernelDirectRouteKind::Object},
  };
  return routes;
}

llvm::Error exportRVVMicrokernelDirectRoute(
    mlir::ModuleOp module, const RVVMicrokernelDirectRouteManifestEntry &route,
    llvm::raw_ostream &os) {
  llvm::Expected<RVVMicrokernelRecord> record =
      isLegacyGenericRVVMicrokernelDirectRoute(route)
          ? buildModuleRecord(module)
          : buildModuleRecordForRVVBinaryFamily(module, *route.family,
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

llvm::Error exportRVVMicrokernelC(mlir::ModuleOp module,
                                  llvm::raw_ostream &os) {
  llvm::Expected<RVVMicrokernelRecord> record = buildModuleRecord(module);
  if (!record)
    return record.takeError();

  std::string source;
  llvm::raw_string_ostream stream(source);
  RVVMicrokernelCExportMode mode =
      RVVMicrokernelCExportMode::RuntimeCallableLibrary;
  if (llvm::Error error = printMicrokernelSource(*record, stream, mode))
    return error;
  stream.flush();
  os << source;
  return llvm::Error::success();
}

llvm::Error exportRVVMicrokernelCForFamily(
    mlir::ModuleOp module, i32_binary::I32BinaryFamilyKind family,
    llvm::raw_ostream &os) {
  RVVI32MicrokernelKind descriptorKind = convertI32BinaryFamilyKind(family);
  const RVVI32MicrokernelFamilySpec &expected =
      getI32MicrokernelFamilySpec(descriptorKind);
  llvm::Expected<RVVMicrokernelRecord> record =
      buildModuleRecordForFamily(module, descriptorKind, expected.routeID);
  if (!record)
    return record.takeError();

  std::string source;
  llvm::raw_string_ostream stream(source);
  RVVMicrokernelCExportMode mode =
      RVVMicrokernelCExportMode::RuntimeCallableLibrary;
  if (llvm::Error error = printMicrokernelSource(*record, stream, mode))
    return error;
  stream.flush();
  os << source;
  return llvm::Error::success();
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

llvm::Error exportRVVMicrokernelHeader(mlir::ModuleOp module,
                                       llvm::raw_ostream &os) {
  llvm::Expected<RVVMicrokernelRecord> record = buildModuleRecord(module);
  if (!record)
    return record.takeError();

  printMicrokernelHeader(*record, os);
  return llvm::Error::success();
}

llvm::Error exportRVVMicrokernelHeaderForFamily(
    mlir::ModuleOp module, i32_binary::I32BinaryFamilyKind family,
    llvm::raw_ostream &os) {
  RVVI32MicrokernelKind descriptorKind = convertI32BinaryFamilyKind(family);
  const RVVI32MicrokernelFamilySpec &expected =
      getI32MicrokernelFamilySpec(descriptorKind);
  llvm::Expected<RVVMicrokernelRecord> record =
      buildModuleRecordForFamily(module, descriptorKind, expected.headerRouteID);
  if (!record)
    return record.takeError();

  printMicrokernelHeader(*record, os);
  return llvm::Error::success();
}

llvm::Error exportRVVMicrokernelObject(mlir::ModuleOp module,
                                       llvm::raw_ostream &os) {
  llvm::Expected<RVVMicrokernelRecord> record = buildModuleRecord(module);
  if (!record) {
    std::string message = llvm::toString(record.takeError());
    return makeModuleMicrokernelObjectError(message);
  }

  std::string source;
  llvm::raw_string_ostream stream(source);
  RVVMicrokernelCExportMode mode =
      RVVMicrokernelCExportMode::RuntimeCallableLibrary;
  if (llvm::Error error = printMicrokernelSource(*record, stream, mode))
    return error;
  stream.flush();
  if (source.empty())
    return makeMicrokernelObjectError(
        record->kernelSymbol,
        "validated RVV microkernel C source must be non-empty before object "
        "export");

  return compileGeneratedMicrokernelSourceToObject(*record, source, os);
}

llvm::Error exportRVVMicrokernelObjectForFamily(
    mlir::ModuleOp module, i32_binary::I32BinaryFamilyKind family,
    llvm::raw_ostream &os) {
  RVVI32MicrokernelKind descriptorKind = convertI32BinaryFamilyKind(family);
  const RVVI32MicrokernelFamilySpec &expected =
      getI32MicrokernelFamilySpec(descriptorKind);
  llvm::Expected<RVVMicrokernelRecord> record =
      buildModuleRecordForFamily(module, descriptorKind, expected.objectRouteID);
  if (!record) {
    std::string message = llvm::toString(record.takeError());
    return makeModuleMicrokernelObjectError(message);
  }

  std::string source;
  llvm::raw_string_ostream stream(source);
  RVVMicrokernelCExportMode mode =
      RVVMicrokernelCExportMode::RuntimeCallableLibrary;
  if (llvm::Error error = printMicrokernelSource(*record, stream, mode))
    return error;
  stream.flush();
  if (source.empty())
    return makeMicrokernelObjectError(
        record->kernelSymbol,
        "validated RVV microkernel C source must be non-empty before object "
        "export");

  return compileGeneratedMicrokernelSourceToObject(*record, source, os);
}

llvm::Error registerRVVMicrokernelTargetExporters(
    TargetArtifactExporterRegistry &registry) {
  for (const RVVMicrokernelDirectRouteManifestEntry &route :
       getRVVMicrokernelDirectRouteManifest()) {
    const RVVBinaryFamilyDescriptor &family = *route.family;
    switch (route.routeKind) {
    case RVVMicrokernelDirectRouteKind::Source:
      if (llvm::Error error = registry.registerExporter(TargetArtifactExporter(
              route.getRouteID(), kMicrokernelArtifactKind, kRVVPluginName,
              family.emissionKind, exportRVVMicrokernelC,
              getRVVBinaryCallableRuntimeABIRoleRequirements(family),
              /*directHelperRoute=*/true, /*handoffKind=*/{},
              validateRVVMicrokernelSourceCandidate,
              family.externalABIComponentGroup, family.runtimeABIName)))
        return error;
      break;
    case RVVMicrokernelDirectRouteKind::Header:
      if (llvm::Error error =
              registry.registerCompositeExporter(TargetArtifactCompositeExporter(
                  route.getRouteID(), kMicrokernelHeaderArtifactKind,
                  getRVVMicrokernelHeaderMatchFn(family),
                  exportRVVMicrokernelHeader, kRVVPluginName,
                  family.runtimeABIKind, family.runtimeABIName,
                  resolveRVVMicrokernelRuntimeABIParameters,
                  /*directHelperRoute=*/true, family.externalABIComponentGroup,
                  family.runtimeABIName,
                  validateRVVMicrokernelCallableCandidatePreflight)))
        return error;
      break;
    case RVVMicrokernelDirectRouteKind::Object:
      if (llvm::Error error =
              registry.registerCompositeExporter(TargetArtifactCompositeExporter(
                  route.getRouteID(), kMicrokernelObjectArtifactKind,
                  getRVVMicrokernelObjectMatchFn(family),
                  exportRVVMicrokernelObject, kRVVPluginName,
                  family.runtimeABIKind, family.runtimeABIName,
                  resolveRVVMicrokernelRuntimeABIParameters,
                  /*directHelperRoute=*/true, family.externalABIComponentGroup,
                  family.runtimeABIName,
                  validateRVVMicrokernelCallableCandidatePreflight)))
        return error;
      break;
    }
  }

  return llvm::Error::success();
}

llvm::Error registerRVVMicrokernelTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry) {
  for (const RVVMicrokernelDirectRouteManifestEntry &route :
       getRVVMicrokernelDirectRouteManifest()) {
    const RVVMicrokernelDirectRouteManifestEntry *routePtr = &route;
    if (llvm::Error error = registry.registerRoute(TargetTranslateRoute(
            route.getRouteID(), route.getDescription(),
            [routePtr](mlir::ModuleOp module, llvm::raw_ostream &os) {
              return exportRVVMicrokernelDirectRoute(module, *routePtr, os);
            },
            route.requiresBinaryStdout())))
      return error;
  }
  return llvm::Error::success();
}

} // namespace tianchenrv::target::rvv
