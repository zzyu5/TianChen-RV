#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <optional>

using namespace tianchenrv::tcrv::rvv;

#include "TianChenRV/Dialect/RVV/IR/RVVOpsDialect.cpp.inc"

#include "TianChenRV/Dialect/RVV/IR/RVVEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "TianChenRV/Dialect/RVV/IR/RVVAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "TianChenRV/Dialect/RVV/IR/RVVTypes.cpp.inc"

#define GET_OP_CLASSES
#include "TianChenRV/Dialect/RVV/IR/RVVOps.cpp.inc"

namespace {

constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kCapabilitySummaryAttrName(
    "capability_summary");
constexpr llvm::StringLiteral kBoundaryVLenBBytesAttrName("vlenb_bytes");
constexpr llvm::StringLiteral kBoundaryI32M1LanesAttrName(
    "base_i32_m1_lanes");
constexpr llvm::StringLiteral kSelectedVectorShapeAttrName(
    "selected_vector_shape");
constexpr llvm::StringLiteral kSelectedVectorSEWAttrName(
    "selected_vector_sew");
constexpr llvm::StringLiteral kSelectedVectorLMULAttrName(
    "selected_vector_lmul");
constexpr llvm::StringLiteral kSelectedTailPolicyAttrName(
    "selected_tail_policy");
constexpr llvm::StringLiteral kSelectedMaskPolicyAttrName(
    "selected_mask_policy");
constexpr llvm::StringLiteral kSelectedVectorTypeAttrName(
    "selected_vector_type");
constexpr llvm::StringLiteral kSelectedVectorSuffixAttrName(
    "selected_vector_suffix");
constexpr llvm::StringLiteral kSelectedSetVLSuffixAttrName(
    "selected_setvl_suffix");
constexpr llvm::StringLiteral kSelectedBinarySourceKindAttrName(
    "selected_binary_source_kind");
constexpr llvm::StringLiteral kSelectedBinaryDTypeAttrName(
    "selected_binary_dtype");
constexpr llvm::StringLiteral kSelectedBinaryFamilyAttrName(
    "selected_binary_family");
constexpr llvm::StringLiteral kSelectedBinaryOperatorAttrName(
    "selected_binary_operator");
constexpr llvm::StringLiteral kSelectedBinaryMicrokernelOpAttrName(
    "selected_binary_microkernel_op");
constexpr llvm::StringLiteral kEmitCSourceOpAttrName("emitc_source_op");
constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceAttrName(
    "emitc_lowerable_op_interface");
constexpr llvm::StringLiteral kUnsupportedReasonAttrName(
    "unsupported_reason");
constexpr llvm::StringLiteral kAVLAttrName("avl");
constexpr llvm::StringLiteral kVLAttrName("vl");
constexpr llvm::StringLiteral kSEWAttrName("sew");
constexpr llvm::StringLiteral kLMULAttrName("lmul");
constexpr llvm::StringLiteral kPolicyAttrName("policy");
constexpr llvm::StringLiteral kElementCountAttrName("element_count");
constexpr llvm::StringLiteral kRequiredMarchAttrName("required_march");
constexpr llvm::StringLiteral kSelectedMABIAttrName("selected_mabi");
constexpr llvm::StringLiteral kBufferRoleAttrName("buffer_role");
constexpr llvm::StringLiteral kVLenAttrName("vlen");
constexpr llvm::StringLiteral kVLenBAttrName("vlenb");
constexpr llvm::StringLiteral kRVVVariantRequiredMarchAttrName(
    "tcrv_rvv.required_march");
constexpr llvm::StringLiteral kRVVRequiredCapabilitiesAttrName(
    "tcrv_rvv.required_capabilities");
constexpr llvm::StringLiteral kRVVElementCountAttrName(
    "tcrv_rvv.element_count");
constexpr llvm::StringLiteral kRVVVLenAttrName("tcrv_rvv.vlen");
constexpr llvm::StringLiteral kRVVVLenBAttrName("tcrv_rvv.vlenb");
constexpr llvm::StringLiteral kArchitectureAttrName("architecture");
constexpr llvm::StringLiteral kISAVectorHintsAttrName("isa_vector_hints");
constexpr llvm::StringLiteral kHartCountAttrName("hart_count");
constexpr llvm::StringLiteral kSelectedMarchAttrName("selected_march");
constexpr llvm::StringLiteral kCapabilityFactsAttrName("capability_facts");
constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kRVVCapabilityID("rvv");
constexpr llvm::StringLiteral kUnsupportedStatusValue("unsupported");
constexpr llvm::StringLiteral kDirectVariantRoleValue("direct variant");
constexpr llvm::StringLiteral kDispatchCaseRoleValue("dispatch case");

bool hasMissingOrEmptyStringAttr(mlir::Operation *op,
                                 llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  return !attr || attr.getValue().trim().empty();
}

bool hasPresentButEmptyStringAttr(mlir::Operation *op,
                                  llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  return attr && attr.getValue().trim().empty();
}

bool containsExecutableClaimWording(llvm::StringRef text) {
  std::string lowerText = text.lower();
  llvm::StringRef lower(lowerText);
  return lower.contains("executable emission supported") ||
         lower.contains("rvv emission supported") ||
         lower.contains("runtime evidence") ||
         lower.contains("correctness evidence") ||
         lower.contains("performance evidence") ||
         lower.contains("emits object") ||
         lower.contains("generated object") ||
         lower.contains("rvv intrinsic emitted");
}

bool containsForbiddenMetadataText(llvm::StringRef text) {
  std::string lowerText = text.lower();
  llvm::StringRef lower(lowerText);
  return lower.contains("password") || lower.contains("passwd") ||
         lower.contains("token") || lower.contains("secret") ||
         lower.contains("private key") ||
         lower.contains("authorization:") || lower.contains("api_key") ||
         lower.contains("access_key");
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

bool isAllowedLoweringBoundaryRole(llvm::StringRef role) {
  return role == kDirectVariantRoleValue || role == kDispatchCaseRoleValue;
}

bool hasAnySelectedBinarySourceIdentityMetadata(mlir::Operation *op) {
  return op && (op->hasAttr(kSelectedBinarySourceKindAttrName) ||
                op->hasAttr(kSelectedBinaryDTypeAttrName) ||
                op->hasAttr(kSelectedBinaryFamilyAttrName) ||
                op->hasAttr(kSelectedBinaryOperatorAttrName) ||
                op->hasAttr(kSelectedBinaryMicrokernelOpAttrName) ||
                op->hasAttr(kEmitCSourceOpAttrName) ||
                op->hasAttr(kEmitCLowerableOpInterfaceAttrName));
}

mlir::LogicalResult verifyBoundedMetadata(mlir::Operation *op,
                                          llvm::StringRef attrName,
                                          llvm::StringRef value) {
  constexpr std::size_t kMaxTextLength = 512;
  if (value.empty() || value.size() > kMaxTextLength)
    return op->emitOpError()
           << "attribute '" << attrName
           << "' must be bounded non-empty single-line metadata";

  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      return op->emitOpError()
             << "attribute '" << attrName
             << "' must be bounded non-empty single-line metadata";
    if (byte < 0x20 && character != '\t')
      return op->emitOpError()
             << "attribute '" << attrName
             << "' must be bounded non-empty single-line metadata";
  }

  if (value.contains("/*") || value.contains("*/"))
    return op->emitOpError()
           << "attribute '" << attrName
           << "' must not contain C comment delimiter text";

  if (containsForbiddenMetadataText(value))
    return op->emitOpError()
           << "attribute '" << attrName
           << "' must not contain secret-like or raw credential text";

  return mlir::success();
}

bool arrayAttrsEqual(mlir::ArrayAttr lhs, mlir::ArrayAttr rhs) {
  if (!lhs || !rhs || lhs.size() != rhs.size())
    return false;

  for (auto [lhsAttr, rhsAttr] : llvm::zip(lhs, rhs)) {
    if (lhsAttr != rhsAttr)
      return false;
  }
  return true;
}

bool isAllowedMicrokernelAttr(llvm::StringRef name) {
  return name == kSourceKernelAttrName || name == kSelectedVariantAttrName ||
         name == kOriginAttrName || name == kRoleAttrName ||
         name == kElementCountAttrName ||
         name == kRequiredCapabilitiesAttrName ||
         name == kRequiredMarchAttrName ||
         name == kSelectedVectorShapeAttrName ||
         name == kSelectedVectorSEWAttrName ||
         name == kSelectedVectorLMULAttrName ||
         name == kSelectedTailPolicyAttrName ||
         name == kSelectedMaskPolicyAttrName ||
         name == kSelectedVectorTypeAttrName ||
         name == kSelectedVectorSuffixAttrName ||
         name == kSelectedSetVLSuffixAttrName ||
         name == kSelectedMABIAttrName;
}

bool isAllowedSetVLAttr(llvm::StringRef name) {
  return name == kSEWAttrName || name == kLMULAttrName ||
         name == kPolicyAttrName;
}

bool isAllowedWithVLAttr(llvm::StringRef name) {
  return name == kSEWAttrName || name == kLMULAttrName ||
         name == kPolicyAttrName;
}

bool isAllowedI32LoadAttr(llvm::StringRef name) {
  return name == kBufferRoleAttrName;
}

bool isAllowedI32AddAttr(llvm::StringRef name) {
  return false;
}

bool isAllowedI32SubAttr(llvm::StringRef name) {
  return false;
}

bool isAllowedI32MulAttr(llvm::StringRef name) {
  return false;
}

bool isAllowedI32StoreAttr(llvm::StringRef name) {
  return name == kBufferRoleAttrName;
}

bool isAllowedI64LoadAttr(llvm::StringRef name) {
  return name == kBufferRoleAttrName;
}

bool isAllowedI64AddAttr(llvm::StringRef name) {
  return false;
}

bool isAllowedI64SubAttr(llvm::StringRef name) {
  return false;
}

bool isAllowedI64MulAttr(llvm::StringRef name) {
  return false;
}

bool isAllowedI64StoreAttr(llvm::StringRef name) {
  return name == kBufferRoleAttrName;
}

bool isForbiddenSetVLParameterAttr(llvm::StringRef name) {
  return name == kAVLAttrName || name == kVLenAttrName ||
         name == kVLenBAttrName || name == kElementCountAttrName ||
         name == kRequiredMarchAttrName ||
         name == kRequiredCapabilitiesAttrName ||
         name == kRVVVariantRequiredMarchAttrName ||
         name == kRVVRequiredCapabilitiesAttrName ||
         name == kRVVElementCountAttrName || name == kRVVVLenAttrName ||
         name == kRVVVLenBAttrName;
}

bool isForbiddenWithVLParameterAttr(llvm::StringRef name) {
  return isForbiddenSetVLParameterAttr(name) || name == kVLAttrName ||
         name == kCapabilitySummaryAttrName ||
         name == kArchitectureAttrName || name == kISAVectorHintsAttrName ||
         name == kHartCountAttrName || name == kSelectedMarchAttrName ||
         name == kCapabilityFactsAttrName;
}

bool isForbiddenDataflowParameterAttr(llvm::StringRef name) {
  return isForbiddenWithVLParameterAttr(name) || name == kSEWAttrName ||
         name == kLMULAttrName || name == kPolicyAttrName;
}

mlir::LogicalResult verifyBoundedDataflowRoleAttr(
    mlir::Operation *op, llvm::StringRef attrName,
    llvm::ArrayRef<tianchenrv::support::RuntimeABIParameterRole> expectedRoles,
    llvm::StringSet<> *seenRoles = nullptr) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr || attr.getValue().trim().empty())
    return op->emitOpError()
           << "requires non-empty runtime ABI role string attribute '"
           << attrName << "'";

  llvm::StringRef value = attr.getValue().trim();
  if (mlir::failed(verifyBoundedMetadata(op, attrName, value)))
    return mlir::failure();

  std::optional<tianchenrv::support::RuntimeABIParameterRole> parsedRole =
      tianchenrv::support::symbolizeRuntimeABIParameterRole(value);
  if (!parsedRole)
    return op->emitOpError()
           << "attribute '" << attrName
           << "' must reference a supported runtime ABI parameter role";

  if (seenRoles && !seenRoles->insert(value).second)
    return op->emitOpError()
           << "requires each dataflow runtime ABI role reference to be unique; "
              "duplicate role '" << value << "'";

  if (llvm::is_contained(expectedRoles, *parsedRole))
    return mlir::success();

  std::string expected;
  llvm::raw_string_ostream stream(expected);
  llvm::interleave(
      expectedRoles,
      [&](tianchenrv::support::RuntimeABIParameterRole role) {
        stream << "'"
               << tianchenrv::support::stringifyRuntimeABIParameterRole(role)
               << "'";
      },
      [&] { stream << " or "; });
  stream.flush();
  return op->emitOpError()
         << "attribute '" << attrName
         << "' must reference runtime ABI role " << expected;
}

bool isI32M1Vector(mlir::Type type) {
  return llvm::isa<I32M1VectorType>(type);
}

bool isI32M2Vector(mlir::Type type) {
  return llvm::isa<I32M2VectorType>(type);
}

llvm::StringRef getI32VectorLMUL(mlir::Type type) {
  if (isI32M1Vector(type))
    return "m1";
  if (isI32M2Vector(type))
    return "m2";
  return {};
}

bool isSupportedI32Vector(mlir::Type type) {
  return !getI32VectorLMUL(type).empty();
}

bool isSupportedI32LMUL(llvm::StringRef lmul) {
  return lmul == "m1" || lmul == "m2";
}

bool isI64M1Vector(mlir::Type type) {
  return llvm::isa<I64M1VectorType>(type);
}

llvm::StringRef getI64VectorLMUL(mlir::Type type) {
  if (isI64M1Vector(type))
    return "m1";
  return {};
}

bool isSupportedRVVFirstSliceConfig(std::int64_t sew, llvm::StringRef lmul) {
  if (sew == 32)
    return isSupportedI32LMUL(lmul);
  if (sew == 64)
    return lmul == "m1";
  return false;
}

mlir::LogicalResult verifyI32VectorTypeForWithVL(mlir::Operation *op,
                                                 mlir::Value value,
                                                 llvm::StringRef role) {
  llvm::StringRef valueLMUL = getI32VectorLMUL(value.getType());
  if (valueLMUL.empty())
    return op->emitOpError()
           << "requires " << role
           << " type to be !tcrv_rvv.i32m1 or !tcrv_rvv.i32m2";

  auto withVL = llvm::dyn_cast_or_null<WithVLOp>(op->getParentOp());
  if (!withVL)
    return mlir::success();

  auto expectedLMUL =
      withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (expectedLMUL && expectedLMUL.getValue() != valueLMUL)
    return op->emitOpError()
           << "requires " << role << " type " << value.getType()
           << " to agree with enclosing tcrv_rvv.with_vl LMUL metadata '"
           << expectedLMUL.getValue() << "'";

  return mlir::success();
}

mlir::LogicalResult verifyI64VectorTypeForWithVL(mlir::Operation *op,
                                                 mlir::Value value,
                                                 llvm::StringRef role) {
  llvm::StringRef valueLMUL = getI64VectorLMUL(value.getType());
  if (valueLMUL.empty())
    return op->emitOpError()
           << "requires " << role << " type to be !tcrv_rvv.i64m1";

  auto withVL = llvm::dyn_cast_or_null<WithVLOp>(op->getParentOp());
  if (!withVL)
    return mlir::success();

  auto expectedSEW =
      withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  if (expectedSEW && expectedSEW.getInt() != 64)
    return op->emitOpError()
           << "requires " << role
           << " type !tcrv_rvv.i64m1 to agree with enclosing "
              "tcrv_rvv.with_vl SEW metadata '"
           << expectedSEW.getInt() << "'";

  auto expectedLMUL =
      withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (expectedLMUL && expectedLMUL.getValue() != valueLMUL)
    return op->emitOpError()
           << "requires " << role << " type " << value.getType()
           << " to agree with enclosing tcrv_rvv.with_vl LMUL metadata '"
           << expectedLMUL.getValue() << "'";

  return mlir::success();
}

enum class I32MicrokernelArithmetic {
  Add,
  Sub,
  Mul,
};

struct I32MicrokernelFamilySpec {
  I32MicrokernelArithmetic arithmetic;
  llvm::StringRef microkernelOpName;
  llvm::StringRef arithmeticOpName;
  llvm::StringRef arithmeticVerb;
  llvm::StringRef resultNoun;
};

const I32MicrokernelFamilySpec &
getI32MicrokernelFamilySpec(I32MicrokernelArithmetic arithmetic) {
  static const I32MicrokernelFamilySpec addSpec{
      I32MicrokernelArithmetic::Add, "tcrv_rvv.i32_vadd_microkernel",
      "tcrv_rvv.i32_add", "add", "sum"};
  static const I32MicrokernelFamilySpec subSpec{
      I32MicrokernelArithmetic::Sub, "tcrv_rvv.i32_vsub_microkernel",
      "tcrv_rvv.i32_sub", "subtract", "difference"};
  static const I32MicrokernelFamilySpec mulSpec{
      I32MicrokernelArithmetic::Mul, "tcrv_rvv.i32_vmul_microkernel",
      "tcrv_rvv.i32_mul", "multiply", "product"};
  switch (arithmetic) {
  case I32MicrokernelArithmetic::Add:
    return addSpec;
  case I32MicrokernelArithmetic::Sub:
    return subSpec;
  case I32MicrokernelArithmetic::Mul:
    return mulSpec;
  }
  llvm_unreachable("unknown RVV i32 microkernel arithmetic");
}

std::optional<I32MicrokernelArithmetic>
getEnclosingI32MicrokernelArithmetic(mlir::Operation *op) {
  if (op->getParentOfType<I32VAddMicrokernelOp>())
    return I32MicrokernelArithmetic::Add;
  if (op->getParentOfType<I32VSubMicrokernelOp>())
    return I32MicrokernelArithmetic::Sub;
  if (op->getParentOfType<I32VMulMicrokernelOp>())
    return I32MicrokernelArithmetic::Mul;
  return std::nullopt;
}

enum class I64MicrokernelArithmetic {
  Add,
  Sub,
  Mul,
};

struct I64MicrokernelFamilySpec {
  I64MicrokernelArithmetic arithmetic;
  llvm::StringRef microkernelOpName;
  llvm::StringRef arithmeticOpName;
  llvm::StringRef arithmeticVerb;
};

const I64MicrokernelFamilySpec &
getI64MicrokernelFamilySpec(I64MicrokernelArithmetic arithmetic) {
  static const I64MicrokernelFamilySpec addSpec{
      I64MicrokernelArithmetic::Add, "tcrv_rvv.i64_vadd_microkernel",
      "tcrv_rvv.i64_add", "add"};
  static const I64MicrokernelFamilySpec subSpec{
      I64MicrokernelArithmetic::Sub, "tcrv_rvv.i64_vsub_microkernel",
      "tcrv_rvv.i64_sub", "subtract"};
  static const I64MicrokernelFamilySpec mulSpec{
      I64MicrokernelArithmetic::Mul, "tcrv_rvv.i64_vmul_microkernel",
      "tcrv_rvv.i64_mul", "multiply"};
  switch (arithmetic) {
  case I64MicrokernelArithmetic::Add:
    return addSpec;
  case I64MicrokernelArithmetic::Sub:
    return subSpec;
  case I64MicrokernelArithmetic::Mul:
    return mulSpec;
  }
  llvm_unreachable("unknown RVV i64 microkernel arithmetic");
}

std::optional<I64MicrokernelArithmetic>
getEnclosingI64MicrokernelArithmetic(mlir::Operation *op) {
  if (op->getParentOfType<I64VAddMicrokernelOp>())
    return I64MicrokernelArithmetic::Add;
  if (op->getParentOfType<I64VSubMicrokernelOp>())
    return I64MicrokernelArithmetic::Sub;
  if (op->getParentOfType<I64VMulMicrokernelOp>())
    return I64MicrokernelArithmetic::Mul;
  return std::nullopt;
}

mlir::LogicalResult verifyNestedI64DataflowOp(
    mlir::Operation *op,
    std::optional<I64MicrokernelArithmetic> requiredArithmetic =
        std::nullopt) {
  if (!llvm::isa_and_nonnull<WithVLOp>(op->getParentOp()))
    return op->emitOpError()
           << "must be nested directly in a tcrv_rvv.with_vl body";
  std::optional<I64MicrokernelArithmetic> enclosingArithmetic =
      getEnclosingI64MicrokernelArithmetic(op);
  if (!enclosingArithmetic)
    return op->emitOpError()
           << "must be nested under tcrv_rvv.i64_vadd_microkernel, "
              "tcrv_rvv.i64_vsub_microkernel, or "
              "tcrv_rvv.i64_vmul_microkernel; it is a finite i64 microkernel "
              "dataflow op, not a standalone RVV compute op";
  if (requiredArithmetic && *requiredArithmetic != *enclosingArithmetic) {
    const I64MicrokernelFamilySpec &requiredSpec =
        getI64MicrokernelFamilySpec(*requiredArithmetic);
    return op->emitOpError()
           << "must be nested under " << requiredSpec.microkernelOpName
           << "; the bounded RVV i64 family keeps arithmetic semantics tied "
              "to the enclosing microkernel op";
  }
  if (op->getNumRegions() != 0)
    return op->emitOpError() << "does not own regions";
  return mlir::success();
}

mlir::LogicalResult verifyNestedI32DataflowOp(
    mlir::Operation *op,
    std::optional<I32MicrokernelArithmetic> requiredArithmetic =
        std::nullopt) {
  if (!llvm::isa_and_nonnull<WithVLOp>(op->getParentOp()))
    return op->emitOpError()
           << "must be nested directly in a tcrv_rvv.with_vl body";
  std::optional<I32MicrokernelArithmetic> enclosingArithmetic =
      getEnclosingI32MicrokernelArithmetic(op);
  if (!enclosingArithmetic)
    return op->emitOpError()
           << "must be nested under tcrv_rvv.i32_vadd_microkernel, "
              "tcrv_rvv.i32_vsub_microkernel, or "
              "tcrv_rvv.i32_vmul_microkernel; it is a finite microkernel "
              "dataflow op, not a standalone RVV compute op";
  if (requiredArithmetic && *requiredArithmetic != *enclosingArithmetic) {
    const I32MicrokernelFamilySpec &requiredSpec =
        getI32MicrokernelFamilySpec(*requiredArithmetic);
    return op->emitOpError()
           << "must be nested under " << requiredSpec.microkernelOpName
           << "; the bounded RVV i32 family keeps arithmetic semantics tied "
              "to the enclosing microkernel op";
  }

  if (op->getNumRegions() != 0)
    return op->emitOpError() << "does not own regions";

  return mlir::success();
}

bool matchArithmeticOp(mlir::Operation *op,
                       I32MicrokernelArithmetic arithmetic,
                       mlir::Value &lhs, mlir::Value &rhs, mlir::Value &vl,
                       mlir::Value &result) {
  switch (arithmetic) {
  case I32MicrokernelArithmetic::Add:
    if (auto add = llvm::dyn_cast<I32AddOp>(op)) {
      lhs = add.getLhs();
      rhs = add.getRhs();
      vl = add.getVl();
      result = add.getSum();
      return true;
    }
    return false;
  case I32MicrokernelArithmetic::Sub:
    if (auto sub = llvm::dyn_cast<I32SubOp>(op)) {
      lhs = sub.getLhs();
      rhs = sub.getRhs();
      vl = sub.getVl();
      result = sub.getDifference();
      return true;
    }
    return false;
  case I32MicrokernelArithmetic::Mul:
    if (auto mul = llvm::dyn_cast<I32MulOp>(op)) {
      lhs = mul.getLhs();
      rhs = mul.getRhs();
      vl = mul.getVl();
      result = mul.getProduct();
      return true;
    }
    return false;
  }
  llvm_unreachable("unknown RVV i32 microkernel arithmetic");
}

bool matchI64ArithmeticOp(mlir::Operation *op,
                          I64MicrokernelArithmetic arithmetic,
                          mlir::Value &lhs, mlir::Value &rhs, mlir::Value &vl,
                          mlir::Value &result) {
  switch (arithmetic) {
  case I64MicrokernelArithmetic::Add:
    if (auto add = llvm::dyn_cast<I64AddOp>(op)) {
      lhs = add.getLhs();
      rhs = add.getRhs();
      vl = add.getVl();
      result = add.getSum();
      return true;
    }
    return false;
  case I64MicrokernelArithmetic::Sub:
    if (auto sub = llvm::dyn_cast<I64SubOp>(op)) {
      lhs = sub.getLhs();
      rhs = sub.getRhs();
      vl = sub.getVl();
      result = sub.getDifference();
      return true;
    }
    return false;
  case I64MicrokernelArithmetic::Mul:
    if (auto mul = llvm::dyn_cast<I64MulOp>(op)) {
      lhs = mul.getLhs();
      rhs = mul.getRhs();
      vl = mul.getVl();
      result = mul.getProduct();
      return true;
    }
    return false;
  }
  llvm_unreachable("unknown RVV i64 microkernel arithmetic");
}

mlir::LogicalResult
verifyMicrokernelStructuredControlPlane(
    mlir::Operation *microkernel,
    const I32MicrokernelFamilySpec &family) {
  mlir::Region &body = microkernel->getRegion(0);
  if (body.empty() || !llvm::hasSingleElement(body))
    return microkernel->emitOpError()
           << "requires exactly one structured RVV control-plane body block";

  mlir::Block &block = body.front();
  if (block.getNumArguments() != 1)
    return microkernel->emitOpError()
           << "requires structured control-plane body to have exactly one "
              "runtime index block argument for target/export-owned n/AVL";
  if (!block.getArgument(0).getType().isIndex())
    return microkernel->emitOpError()
           << "requires structured control-plane body argument to have index "
              "type for runtime n/AVL";

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
    return microkernel->emitOpError()
           << "structured control-plane body accepts only "
              "tcrv_rvv.setvl and tcrv_rvv.with_vl direct operations; "
              "unexpected operation '"
           << bodyOp.getName().getStringRef() << "'";
  }

  if (setvlCount != 1)
    return microkernel->emitOpError()
           << "requires exactly one tcrv_rvv.setvl in the structured "
              "control-plane body";
  if (withVLCount != 1)
    return microkernel->emitOpError()
           << "requires exactly one tcrv_rvv.with_vl in the structured "
              "control-plane body";

  if (setvl.getAvl() != block.getArgument(0))
    return microkernel->emitOpError()
           << "requires tcrv_rvv.setvl AVL operand to be the runtime index "
              "body argument, not descriptor-local element_count or a "
              "constant";
  if (withVL.getVl() != setvl.getVl())
    return microkernel->emitOpError()
           << "requires tcrv_rvv.with_vl to consume the !tcrv_rvv.vl token "
              "produced by the body tcrv_rvv.setvl";

  if (!withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName) ||
      !withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName) ||
      !withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName)) {
    return microkernel->emitOpError()
           << "requires tcrv_rvv.with_vl to carry explicit SEW/LMUL/policy "
              "control metadata so emission can consume the structured body";
  }

  mlir::Region &withVLBody = withVL.getBody();
  if (withVLBody.empty() || !llvm::hasSingleElement(withVLBody))
    return microkernel->emitOpError()
           << "requires tcrv_rvv.with_vl to own one body block";
  if (withVLBody.front().getNumArguments() != 0)
    return microkernel->emitOpError()
           << "requires tcrv_rvv.with_vl body to have no block arguments; "
              "runtime n/AVL/VL is carried by the enclosing control-plane "
              "surface";

  llvm::SmallVector<mlir::Operation *, 4> ops;
  for (mlir::Operation &withVLOp : withVLBody.front())
    ops.push_back(&withVLOp);
  if (ops.size() != 4)
    return microkernel->emitOpError()
           << "requires tcrv_rvv.with_vl body to contain exactly the finite "
              "tcrv_rvv.i32_load, tcrv_rvv.i32_load, "
           << family.arithmeticOpName << ", "
              "tcrv_rvv.i32_store dataflow sequence";

  auto lhsLoad = llvm::dyn_cast<I32LoadOp>(ops[0]);
  auto rhsLoad = llvm::dyn_cast<I32LoadOp>(ops[1]);
  auto store = llvm::dyn_cast<I32StoreOp>(ops[3]);
  mlir::Value arithmeticLHS;
  mlir::Value arithmeticRHS;
  mlir::Value arithmeticVL;
  mlir::Value arithmeticResult;
  bool hasArithmetic =
      matchArithmeticOp(ops[2], family.arithmetic, arithmeticLHS,
                        arithmeticRHS, arithmeticVL, arithmeticResult);
  if (!lhsLoad || !rhsLoad || !hasArithmetic || !store)
    return microkernel->emitOpError()
           << "requires tcrv_rvv.with_vl body to contain exactly the finite "
              "tcrv_rvv.i32_load, tcrv_rvv.i32_load, "
           << family.arithmeticOpName << ", "
              "tcrv_rvv.i32_store dataflow sequence";

  if (lhsLoad.getVl() != withVL.getVl() || rhsLoad.getVl() != withVL.getVl() ||
      arithmeticVL != withVL.getVl() || store.getVl() != withVL.getVl())
    return microkernel->emitOpError()
           << "requires every finite RVV i32 dataflow op to consume the "
              "!tcrv_rvv.vl token owned by the surrounding tcrv_rvv.with_vl";

  auto lhsRole = lhsLoad->getAttrOfType<mlir::StringAttr>(kBufferRoleAttrName);
  auto rhsRole = rhsLoad->getAttrOfType<mlir::StringAttr>(kBufferRoleAttrName);
  auto outRole = store->getAttrOfType<mlir::StringAttr>(kBufferRoleAttrName);
  if (!lhsRole || lhsRole.getValue() !=
                      tianchenrv::support::stringifyRuntimeABIParameterRole(
                          tianchenrv::support::RuntimeABIParameterRole::
                              LHSInputBuffer))
    return microkernel->emitOpError()
           << "requires first tcrv_rvv.i32_load to reference runtime ABI role "
              "'lhs-input-buffer'";
  if (!rhsRole || rhsRole.getValue() !=
                      tianchenrv::support::stringifyRuntimeABIParameterRole(
                          tianchenrv::support::RuntimeABIParameterRole::
                              RHSInputBuffer))
    return microkernel->emitOpError()
           << "requires second tcrv_rvv.i32_load to reference runtime ABI role "
              "'rhs-input-buffer'";
  if (!outRole || outRole.getValue() !=
                      tianchenrv::support::stringifyRuntimeABIParameterRole(
                          tianchenrv::support::RuntimeABIParameterRole::
                              OutputBuffer))
    return microkernel->emitOpError()
           << "requires tcrv_rvv.i32_store to reference runtime ABI role "
              "'output-buffer'";

  if (arithmeticLHS != lhsLoad.getLoaded() ||
      arithmeticRHS != rhsLoad.getLoaded() ||
      store.getValue() != arithmeticResult)
    return microkernel->emitOpError()
           << "requires finite RVV i32 dataflow SSA chain "
              "lhs-load,rhs-load -> "
           << family.arithmeticVerb << " -> store";

  return mlir::success();
}

mlir::LogicalResult verifyI64MicrokernelStructuredControlPlane(
    mlir::Operation *microkernel, const I64MicrokernelFamilySpec &family) {
  mlir::Region &body = microkernel->getRegion(0);
  if (body.empty() || !llvm::hasSingleElement(body))
    return microkernel->emitOpError()
           << "requires exactly one structured RVV control-plane body block";

  mlir::Block &block = body.front();
  if (block.getNumArguments() != 1)
    return microkernel->emitOpError()
           << "requires structured control-plane body to have exactly one "
              "runtime index block argument for target/export-owned n/AVL";
  if (!block.getArgument(0).getType().isIndex())
    return microkernel->emitOpError()
           << "requires structured control-plane body argument to have index "
              "type for runtime n/AVL";

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
    return microkernel->emitOpError()
           << "structured control-plane body accepts only "
              "tcrv_rvv.setvl and tcrv_rvv.with_vl direct operations; "
              "unexpected operation '"
           << bodyOp.getName().getStringRef() << "'";
  }

  if (setvlCount != 1 || withVLCount != 1)
    return microkernel->emitOpError()
           << "requires exactly one tcrv_rvv.setvl and exactly one "
              "tcrv_rvv.with_vl in the structured control-plane body";
  if (setvl.getAvl() != block.getArgument(0))
    return microkernel->emitOpError()
           << "requires tcrv_rvv.setvl AVL operand to be the runtime index "
              "body argument, not descriptor-local element_count or a "
              "constant";
  if (withVL.getVl() != setvl.getVl())
    return microkernel->emitOpError()
           << "requires tcrv_rvv.with_vl to consume the !tcrv_rvv.vl token "
              "produced by the body tcrv_rvv.setvl";

  auto withVLSew = withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto withVLLMUL = withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!withVLSew || !withVLLMUL ||
      !withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName)) {
    return microkernel->emitOpError()
           << "requires tcrv_rvv.with_vl to carry explicit SEW/LMUL/policy "
              "control metadata so emission can consume the structured body";
  }
  if (setvl.getSew() != 64 || setvl.getLmul() != "m1" ||
      withVLSew.getInt() != 64 || withVLLMUL.getValue() != "m1")
    return microkernel->emitOpError()
           << "requires finite RVV i64 control-plane config SEW64/LMUL m1";

  mlir::Region &withVLBody = withVL.getBody();
  if (withVLBody.empty() || !llvm::hasSingleElement(withVLBody))
    return microkernel->emitOpError()
           << "requires tcrv_rvv.with_vl to own one body block";
  if (withVLBody.front().getNumArguments() != 0)
    return microkernel->emitOpError()
           << "requires tcrv_rvv.with_vl body to have no block arguments; "
              "runtime n/AVL/VL is carried by the enclosing control-plane "
              "surface";

  llvm::SmallVector<mlir::Operation *, 4> ops;
  for (mlir::Operation &withVLOp : withVLBody.front())
    ops.push_back(&withVLOp);
  if (ops.size() != 4)
    return microkernel->emitOpError()
           << "requires tcrv_rvv.with_vl body to contain exactly the finite "
              "tcrv_rvv.i64_load, tcrv_rvv.i64_load, "
           << family.arithmeticOpName << ", "
              "tcrv_rvv.i64_store dataflow sequence";

  auto lhsLoad = llvm::dyn_cast<I64LoadOp>(ops[0]);
  auto rhsLoad = llvm::dyn_cast<I64LoadOp>(ops[1]);
  mlir::Value arithmeticLHS;
  mlir::Value arithmeticRHS;
  mlir::Value arithmeticVL;
  mlir::Value arithmeticResult;
  bool hasArithmetic = matchI64ArithmeticOp(ops[2], family.arithmetic,
                                            arithmeticLHS, arithmeticRHS,
                                            arithmeticVL, arithmeticResult);
  auto store = llvm::dyn_cast<I64StoreOp>(ops[3]);
  if (!lhsLoad || !rhsLoad || !hasArithmetic || !store)
    return microkernel->emitOpError()
           << "requires tcrv_rvv.with_vl body to contain exactly the finite "
              "tcrv_rvv.i64_load, tcrv_rvv.i64_load, "
           << family.arithmeticOpName << ", "
              "tcrv_rvv.i64_store dataflow sequence";

  if (lhsLoad.getVl() != withVL.getVl() || rhsLoad.getVl() != withVL.getVl() ||
      arithmeticVL != withVL.getVl() || store.getVl() != withVL.getVl())
    return microkernel->emitOpError()
           << "requires every finite RVV i64 dataflow op to consume the "
              "!tcrv_rvv.vl token owned by the surrounding tcrv_rvv.with_vl";

  auto lhsRole = lhsLoad->getAttrOfType<mlir::StringAttr>(kBufferRoleAttrName);
  auto rhsRole = rhsLoad->getAttrOfType<mlir::StringAttr>(kBufferRoleAttrName);
  auto outRole = store->getAttrOfType<mlir::StringAttr>(kBufferRoleAttrName);
  if (!lhsRole || lhsRole.getValue() !=
                      tianchenrv::support::stringifyRuntimeABIParameterRole(
                          tianchenrv::support::RuntimeABIParameterRole::
                              LHSInputBuffer))
    return microkernel->emitOpError()
           << "requires first tcrv_rvv.i64_load to reference runtime ABI role "
              "'lhs-input-buffer'";
  if (!rhsRole || rhsRole.getValue() !=
                      tianchenrv::support::stringifyRuntimeABIParameterRole(
                          tianchenrv::support::RuntimeABIParameterRole::
                              RHSInputBuffer))
    return microkernel->emitOpError()
           << "requires second tcrv_rvv.i64_load to reference runtime ABI role "
              "'rhs-input-buffer'";
  if (!outRole || outRole.getValue() !=
                      tianchenrv::support::stringifyRuntimeABIParameterRole(
                          tianchenrv::support::RuntimeABIParameterRole::
                              OutputBuffer))
    return microkernel->emitOpError()
           << "requires tcrv_rvv.i64_store to reference runtime ABI role "
              "'output-buffer'";

  if (arithmeticLHS != lhsLoad.getLoaded() ||
      arithmeticRHS != rhsLoad.getLoaded() ||
      store.getValue() != arithmeticResult)
    return microkernel->emitOpError()
           << "requires finite RVV i64 dataflow SSA chain "
              "lhs-load,rhs-load -> "
           << family.arithmeticVerb << " -> store";

  return mlir::success();
}

} // namespace

mlir::LogicalResult SetVLOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (attrName == kAVLAttrName)
      return emitOpError()
             << "requires AVL to be a runtime SSA operand; attribute '"
             << kAVLAttrName
             << "' is not accepted as an AVL substitute";

    if (isForbiddenSetVLParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.setvl keeps VLEN/vlenb as target capability "
                "facts, element_count as descriptor-local metadata, and "
                "required_march/required_capabilities as selected-path "
                "metadata";

    if (!isAllowedSetVLAttr(attrName))
      return emitOpError()
             << "only accepts bounded compile-time config attributes '"
             << kSEWAttrName << "', '" << kLMULAttrName << "', and '"
             << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 1)
    return emitOpError()
           << "requires exactly one runtime AVL SSA operand";
  if (!getAvl().getType().isIndex())
    return emitOpError()
           << "requires runtime AVL operand to have index type";

  if (op->getNumResults() != 1)
    return emitOpError() << "requires exactly one VL result";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError()
           << "requires result type to be !tcrv_rvv.vl";

  if (!isSupportedRVVFirstSliceConfig(static_cast<std::int64_t>(getSew()),
                                      getLmul()))
    return emitOpError()
           << "requires bounded RVV first-slice compile-time config to be "
              "SEW32 with LMUL \"m1\"/\"m2\" or SEW64 with LMUL \"m1\"";

  if (!getPolicy())
    return emitOpError()
           << "requires finite #tcrv_rvv.policy compile-time policy metadata";

  return mlir::success();
}

mlir::LogicalResult WithVLOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenWithVLParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.with_vl keeps VLEN/vlenb as target capability "
                "facts, element_count as descriptor-local metadata, "
                "required_march/required_capabilities as selected-path "
                "metadata, and AVL/VL as runtime SSA/control values";

    if (!isAllowedWithVLAttr(attrName))
      return emitOpError()
             << "only accepts optional bounded compile-time config "
                "attributes '"
             << kSEWAttrName << "', '" << kLMULAttrName << "', and '"
             << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 1)
    return emitOpError() << "requires exactly one runtime VL SSA operand";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError()
           << "requires runtime VL operand to have !tcrv_rvv.vl type";

  if (op->getNumRegions() != 1)
    return emitOpError() << "requires exactly one VL scope region";

  mlir::Region &body = getBody();
  if (body.empty() || !llvm::hasSingleElement(body))
    return emitOpError() << "requires a single-block VL scope region";
  if (body.front().getNumArguments() != 0)
    return emitOpError()
           << "requires VL scope region to have no region arguments; the "
              "consumed !tcrv_rvv.vl operand is the scope control value";

  auto sew = op->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto lmul = op->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (sew && lmul &&
      !isSupportedRVVFirstSliceConfig(sew.getInt(), lmul.getValue()))
    return emitOpError()
           << "requires bounded RVV first-slice compile-time config to be "
              "SEW32 with LMUL \"m1\"/\"m2\" or SEW64 with LMUL \"m1\"";
  if (sew && !lmul)
    return emitOpError()
           << "requires optional 'lmul' metadata when optional 'sew' "
              "metadata is present";
  if (!sew && lmul)
    return emitOpError()
           << "requires optional 'sew' metadata when optional 'lmul' "
              "metadata is present";

  auto policy = op->getAttrOfType<PolicyAttr>(kPolicyAttrName);
  if (op->hasAttr(kPolicyAttrName) && !policy)
    return emitOpError()
           << "requires optional policy metadata to be #tcrv_rvv.policy";

  if (auto setvl = getVl().getDefiningOp<SetVLOp>()) {
    if (sew && static_cast<int64_t>(setvl.getSew()) != sew.getInt())
      return emitOpError()
             << "requires optional 'sew' metadata to match defining "
                "tcrv_rvv.setvl";
    if (lmul && setvl.getLmul() != lmul.getValue())
      return emitOpError()
             << "requires optional 'lmul' metadata to match defining "
                "tcrv_rvv.setvl";
    if (policy && setvl.getPolicy() != policy)
      return emitOpError()
             << "requires optional 'policy' metadata to match defining "
                "tcrv_rvv.setvl";
  }

  return mlir::success();
}

mlir::LogicalResult I32LoadOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.i32_load keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and element_count as descriptor-local "
                "microkernel metadata";

    if (!isAllowedI32LoadAttr(attrName))
      return emitOpError()
             << "only accepts finite input buffer runtime ABI role attribute '"
             << kBufferRoleAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return emitOpError()
           << "requires exactly one !tcrv_rvv.vl operand and one "
              "bounded RVV i32 vector result";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedI32DataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(
          verifyI32VectorTypeForWithVL(op, getLoaded(), "result")))
    return mlir::failure();

  llvm::StringSet<> seenRoles;
  if (mlir::failed(verifyBoundedDataflowRoleAttr(
          op, kBufferRoleAttrName,
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer},
          &seenRoles)))
    return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult I32AddOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.i32_add keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "element_count as descriptor-local microkernel metadata";

    if (!isAllowedI32AddAttr(attrName))
      return emitOpError()
             << "does not accept dataflow attributes; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs/rhs bounded RVV i32 vector operands, one "
              "!tcrv_rvv.vl operand, and one bounded RVV i32 vector result";
  if (!isSupportedI32Vector(getLhs().getType()) ||
      !isSupportedI32Vector(getRhs().getType()) ||
      !isSupportedI32Vector(getSum().getType()))
    return emitOpError()
           << "requires lhs, rhs, and result types to be !tcrv_rvv.i32m1 "
              "or !tcrv_rvv.i32m2";
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getSum().getType())
    return emitOpError()
           << "requires lhs, rhs, and result to have the same bounded RVV "
              "i32 vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedI32DataflowOp(
          op, I32MicrokernelArithmetic::Add)))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  return verifyI32VectorTypeForWithVL(op, getSum(), "result");
}

mlir::LogicalResult I32SubOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.i32_sub keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "element_count as descriptor-local microkernel metadata";

    if (!isAllowedI32SubAttr(attrName))
      return emitOpError()
             << "does not accept dataflow attributes; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs/rhs bounded RVV i32 vector operands, one "
              "!tcrv_rvv.vl operand, and one bounded RVV i32 vector result";
  if (!isSupportedI32Vector(getLhs().getType()) ||
      !isSupportedI32Vector(getRhs().getType()) ||
      !isSupportedI32Vector(getDifference().getType()))
    return emitOpError()
           << "requires lhs, rhs, and result types to be !tcrv_rvv.i32m1 "
              "or !tcrv_rvv.i32m2";
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getDifference().getType())
    return emitOpError()
           << "requires lhs, rhs, and result to have the same bounded RVV "
              "i32 vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedI32DataflowOp(
          op, I32MicrokernelArithmetic::Sub)))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  return verifyI32VectorTypeForWithVL(op, getDifference(), "result");
}

mlir::LogicalResult I32MulOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.i32_mul keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "element_count as descriptor-local microkernel metadata";

    if (!isAllowedI32MulAttr(attrName))
      return emitOpError()
             << "does not accept dataflow attributes; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs/rhs bounded RVV i32 vector operands, one "
              "!tcrv_rvv.vl operand, and one bounded RVV i32 vector result";
  if (!isSupportedI32Vector(getLhs().getType()) ||
      !isSupportedI32Vector(getRhs().getType()) ||
      !isSupportedI32Vector(getProduct().getType()))
    return emitOpError()
           << "requires lhs, rhs, and result types to be !tcrv_rvv.i32m1 "
              "or !tcrv_rvv.i32m2";
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getProduct().getType())
    return emitOpError()
           << "requires lhs, rhs, and result to have the same bounded RVV "
              "i32 vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedI32DataflowOp(
          op, I32MicrokernelArithmetic::Mul)))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  return verifyI32VectorTypeForWithVL(op, getProduct(), "result");
}

mlir::LogicalResult I32StoreOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.i32_store keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and element_count as descriptor-local "
                "microkernel metadata";

    if (!isAllowedI32StoreAttr(attrName))
      return emitOpError()
             << "only accepts finite output buffer runtime ABI role attribute '"
             << kBufferRoleAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (op->getNumOperands() != 2 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one bounded RVV i32 vector value operand, one "
              "!tcrv_rvv.vl operand, and no results";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedI32DataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getValue(), "stored value")))
    return mlir::failure();

  llvm::StringSet<> seenRoles;
  if (mlir::failed(verifyBoundedDataflowRoleAttr(
          op, kBufferRoleAttrName,
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer},
          &seenRoles)))
    return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult I64LoadOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.i64_load keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and element_count as descriptor-local "
                "microkernel metadata";

    if (!isAllowedI64LoadAttr(attrName))
      return emitOpError()
             << "only accepts finite input buffer runtime ABI role attribute '"
             << kBufferRoleAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return emitOpError()
           << "requires exactly one !tcrv_rvv.vl operand and one "
              "bounded RVV i64 vector result";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError()
           << "requires runtime VL operand to have !tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedI64DataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(
          verifyI64VectorTypeForWithVL(op, getLoaded(), "result")))
    return mlir::failure();

  llvm::StringSet<> seenRoles;
  return verifyBoundedDataflowRoleAttr(
      op, kBufferRoleAttrName,
      {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer,
       tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer},
      &seenRoles);
}

mlir::LogicalResult I64AddOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.i64_add keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "element_count as descriptor-local microkernel metadata";

    if (!isAllowedI64AddAttr(attrName))
      return emitOpError()
             << "does not accept dataflow attributes; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs/rhs bounded RVV i64 vector operands, one "
              "!tcrv_rvv.vl operand, and one bounded RVV i64 vector result";
  if (!isI64M1Vector(getLhs().getType()) ||
      !isI64M1Vector(getRhs().getType()) ||
      !isI64M1Vector(getSum().getType()))
    return emitOpError()
           << "requires lhs, rhs, and result types to be !tcrv_rvv.i64m1";
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getSum().getType())
    return emitOpError()
           << "requires lhs, rhs, and result to have the same bounded RVV "
              "i64 vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError()
           << "requires runtime VL operand to have !tcrv_rvv.vl type";
  if (mlir::failed(
          verifyNestedI64DataflowOp(op, I64MicrokernelArithmetic::Add)))
    return mlir::failure();
  if (mlir::failed(verifyI64VectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyI64VectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  return verifyI64VectorTypeForWithVL(op, getSum(), "result");
}

mlir::LogicalResult I64SubOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.i64_sub keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "element_count as descriptor-local microkernel metadata";

    if (!isAllowedI64SubAttr(attrName))
      return emitOpError()
             << "does not accept dataflow attributes; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs/rhs bounded RVV i64 vector operands, one "
              "!tcrv_rvv.vl operand, and one bounded RVV i64 vector result";
  if (!isI64M1Vector(getLhs().getType()) ||
      !isI64M1Vector(getRhs().getType()) ||
      !isI64M1Vector(getDifference().getType()))
    return emitOpError()
           << "requires lhs, rhs, and result types to be !tcrv_rvv.i64m1";
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getDifference().getType())
    return emitOpError()
           << "requires lhs, rhs, and result to have the same bounded RVV "
              "i64 vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError()
           << "requires runtime VL operand to have !tcrv_rvv.vl type";
  if (mlir::failed(
          verifyNestedI64DataflowOp(op, I64MicrokernelArithmetic::Sub)))
    return mlir::failure();
  if (mlir::failed(verifyI64VectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyI64VectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  return verifyI64VectorTypeForWithVL(op, getDifference(), "result");
}

mlir::LogicalResult I64MulOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.i64_mul keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "element_count as descriptor-local microkernel metadata";

    if (!isAllowedI64MulAttr(attrName))
      return emitOpError()
             << "does not accept dataflow attributes; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs/rhs bounded RVV i64 vector operands, one "
              "!tcrv_rvv.vl operand, and one bounded RVV i64 vector result";
  if (!isI64M1Vector(getLhs().getType()) ||
      !isI64M1Vector(getRhs().getType()) ||
      !isI64M1Vector(getProduct().getType()))
    return emitOpError()
           << "requires lhs, rhs, and result types to be !tcrv_rvv.i64m1";
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getProduct().getType())
    return emitOpError()
           << "requires lhs, rhs, and result to have the same bounded RVV "
              "i64 vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError()
           << "requires runtime VL operand to have !tcrv_rvv.vl type";
  if (mlir::failed(
          verifyNestedI64DataflowOp(op, I64MicrokernelArithmetic::Mul)))
    return mlir::failure();
  if (mlir::failed(verifyI64VectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyI64VectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  return verifyI64VectorTypeForWithVL(op, getProduct(), "result");
}

mlir::LogicalResult I64StoreOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.i64_store keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and element_count as descriptor-local "
                "microkernel metadata";

    if (!isAllowedI64StoreAttr(attrName))
      return emitOpError()
             << "only accepts finite output buffer runtime ABI role attribute '"
             << kBufferRoleAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (op->getNumOperands() != 2 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one bounded RVV i64 vector value operand, one "
              "!tcrv_rvv.vl operand, and no results";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError()
           << "requires runtime VL operand to have !tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedI64DataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyI64VectorTypeForWithVL(op, getValue(), "stored value")))
    return mlir::failure();

  llvm::StringSet<> seenRoles;
  return verifyBoundedDataflowRoleAttr(
      op, kBufferRoleAttrName,
      {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer},
      &seenRoles);
}

mlir::LogicalResult LoweringBoundaryOp::verify() {
  mlir::Operation *op = getOperation();

  if (hasMissingOrEmptyStringAttr(op, kSourceKernelAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kSourceKernelAttrName
           << "'";
  if (hasMissingOrEmptyStringAttr(op, kRoleAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kRoleAttrName << "'";
  if (hasMissingOrEmptyStringAttr(op, kOriginAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kOriginAttrName
           << "'";
  if (hasMissingOrEmptyStringAttr(op, kStatusAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kStatusAttrName
           << "'";
  if (hasMissingOrEmptyStringAttr(op, kUnsupportedReasonAttrName))
    return emitOpError()
           << "requires non-empty string attribute '"
           << kUnsupportedReasonAttrName << "'";
  if (hasPresentButEmptyStringAttr(op, kCapabilitySummaryAttrName))
    return emitOpError()
           << "requires non-empty string attribute '"
           << kCapabilitySummaryAttrName << "' when present";

  auto status = op->getAttrOfType<mlir::StringAttr>(kStatusAttrName);
  if (status.getValue() != kUnsupportedStatusValue)
    return emitOpError()
           << "status must be '" << kUnsupportedStatusValue
           << "' because tcrv_rvv.lowering_boundary is pre-executable "
              "metadata and not an RVV executable lowering claim";

  auto origin = op->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (origin.getValue() != kRVVPluginName)
    return emitOpError()
           << "origin must be '" << kRVVPluginName
           << "' because this is the RVV plugin boundary surface";

  auto role = op->getAttrOfType<mlir::StringAttr>(kRoleAttrName);
  if (!isAllowedLoweringBoundaryRole(role.getValue()))
    return emitOpError()
           << "role must be '" << kDirectVariantRoleValue << "' or '"
           << kDispatchCaseRoleValue
           << "'; dispatch fallback lowering boundaries are not materialized "
              "by this RVV first slice";

  auto unsupportedReason =
      op->getAttrOfType<mlir::StringAttr>(kUnsupportedReasonAttrName);
  if (containsExecutableClaimWording(unsupportedReason.getValue()))
    return emitOpError()
           << "unsupported_reason must not claim executable RVV emission, "
              "runtime, correctness, or performance evidence";

  auto selectedVariant =
      op->getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
  if (!selectedVariant || selectedVariant.getValue().trim().empty())
    return emitOpError()
           << "requires non-empty variant symbol reference attribute '"
           << kSelectedVariantAttrName << "'";

  auto requiredCapabilities =
      op->getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName);
  if (!requiredCapabilities || requiredCapabilities.empty())
    return emitOpError()
           << "requires non-empty array attribute '"
           << kRequiredCapabilitiesAttrName
           << "' containing capability symbol references";

  auto kernel = op->getParentOfType<tianchenrv::tcrv::exec::KernelOp>();
  if (!kernel)
    return emitOpError()
           << "must be nested directly in a tcrv.exec.kernel";
  if (op->getParentOp() != kernel.getOperation())
    return emitOpError()
           << "must be a direct child of the enclosing tcrv.exec.kernel";

  auto sourceKernel = op->getAttrOfType<mlir::StringAttr>(
      kSourceKernelAttrName);
  if (sourceKernel.getValue() != kernel.getSymName())
    return emitOpError()
           << "source_kernel must match enclosing tcrv.exec.kernel symbol @"
           << kernel.getSymName();

  if (kernel.getBody().empty())
    return emitOpError()
           << "requires enclosing tcrv.exec.kernel to have a body block";

  llvm::Expected<tianchenrv::support::TargetCapabilitySet>
      capabilitiesOrError =
          tianchenrv::support::TargetCapabilitySet::buildFromKernelChecked(
              kernel);
  if (!capabilitiesOrError) {
    std::string message = llvm::toString(capabilitiesOrError.takeError());
    return emitOpError() << message;
  }
  const tianchenrv::support::TargetCapabilitySet &capabilities =
      *capabilitiesOrError;

  tianchenrv::tcrv::exec::VariantOp resolvedVariant;
  for (mlir::Operation &sibling : kernel.getBody().front()) {
    auto variant =
        llvm::dyn_cast<tianchenrv::tcrv::exec::VariantOp>(sibling);
    if (variant && variant.getSymName() == selectedVariant.getValue()) {
      resolvedVariant = variant;
      break;
    }
  }

  if (!resolvedVariant)
    return emitOpError()
           << "selected_variant @" << selectedVariant.getValue()
           << " must resolve to a direct sibling tcrv.exec.variant in the "
              "enclosing tcrv.exec.kernel";

  for (mlir::Attribute requiredCapability : requiredCapabilities) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return emitOpError()
             << "attribute '" << kRequiredCapabilitiesAttrName
             << "' must contain only capability symbol references";

    if (!capabilities.lookupBySymbolName(symbolRef.getValue()))
      return emitOpError()
             << "requires unknown capability @" << symbolRef.getValue()
             << " in enclosing tcrv.exec.kernel";
  }

  auto variantRequires =
      resolvedVariant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!arrayAttrsEqual(requiredCapabilities, variantRequires))
    return emitOpError()
           << "required_capabilities must match selected variant requires "
              "metadata";

  auto vlenbBytes = op->getAttrOfType<mlir::IntegerAttr>(
      kBoundaryVLenBBytesAttrName);
  auto i32M1Lanes = op->getAttrOfType<mlir::IntegerAttr>(
      kBoundaryI32M1LanesAttrName);
  if (static_cast<bool>(vlenbBytes) != static_cast<bool>(i32M1Lanes))
    return emitOpError()
           << "selected capacity metadata requires both '"
           << kBoundaryVLenBBytesAttrName << "' and '"
           << kBoundaryI32M1LanesAttrName << "'";
  if (vlenbBytes) {
    std::int64_t vlenb = vlenbBytes.getInt();
    std::int64_t lanes = i32M1Lanes.getInt();
    if (vlenb <= 0 || lanes <= 0 || vlenb < 4 || vlenb % 4 != 0 ||
        vlenb / 4 != lanes)
      return emitOpError()
             << "selected capacity metadata requires base_i32_m1_lanes to "
                "equal vlenb_bytes divided by four";
  }

  if (hasAnySelectedBinarySourceIdentityMetadata(op)) {
    for (llvm::StringRef attrName :
         {kSelectedBinarySourceKindAttrName, kSelectedBinaryDTypeAttrName,
          kSelectedBinaryFamilyAttrName, kSelectedBinaryOperatorAttrName,
          kSelectedBinaryMicrokernelOpAttrName, kEmitCSourceOpAttrName,
          kEmitCLowerableOpInterfaceAttrName}) {
      auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
      if (!attr || attr.getValue().trim().empty())
        return emitOpError()
               << "selected binary source identity metadata requires "
                  "non-empty string attribute '"
               << attrName << "'";
      if (mlir::failed(
              verifyBoundedMetadata(op, attrName, attr.getValue().trim())))
        return mlir::failure();
    }

    auto interface = op->getAttrOfType<mlir::StringAttr>(
        kEmitCLowerableOpInterfaceAttrName);
    if (interface.getValue() != "TCRVEmitCLowerableOpInterface")
      return emitOpError()
             << "selected binary source identity metadata requires '"
             << kEmitCLowerableOpInterfaceAttrName
             << "' to be 'TCRVEmitCLowerableOpInterface'";
  }

  return mlir::success();
}

mlir::LogicalResult
verifyI32MicrokernelOp(mlir::Operation *op,
                       const I32MicrokernelFamilySpec &family) {
  auto emitOpError = [&]() { return op->emitOpError(); };

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    if (!isAllowedMicrokernelAttr(attr.getName().getValue()))
      return emitOpError()
             << "does not accept generic tensor/tile/benchmark or unknown "
                "attribute '"
             << attr.getName()
             << "'; this op is exactly a bounded RVV i32 vector-"
             << family.arithmeticVerb << " microkernel";
  }

  if (hasMissingOrEmptyStringAttr(op, kSourceKernelAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kSourceKernelAttrName
           << "'";
  if (hasMissingOrEmptyStringAttr(op, kOriginAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kOriginAttrName
           << "'";
  if (hasMissingOrEmptyStringAttr(op, kRoleAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kRoleAttrName << "'";
  if (hasMissingOrEmptyStringAttr(op, kRequiredMarchAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kRequiredMarchAttrName
           << "'";
  if (hasPresentButEmptyStringAttr(op, kSelectedMABIAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kSelectedMABIAttrName
           << "' when present";

  for (llvm::StringRef attrName :
       {kSourceKernelAttrName, kOriginAttrName, kRoleAttrName,
        kRequiredMarchAttrName, kSelectedMABIAttrName}) {
    if (auto attr = op->getAttrOfType<mlir::StringAttr>(attrName))
      if (mlir::failed(
              verifyBoundedMetadata(op, attrName, attr.getValue().trim())))
        return mlir::failure();
  }

  auto origin = op->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (origin.getValue() != kRVVPluginName)
    return emitOpError()
           << "origin must be '" << kRVVPluginName
           << "' because this executable microkernel is RVV plugin-local";

  auto role = op->getAttrOfType<mlir::StringAttr>(kRoleAttrName);
  if (!isAllowedLoweringBoundaryRole(role.getValue()))
    return emitOpError()
           << "role must be '" << kDirectVariantRoleValue << "' or '"
           << kDispatchCaseRoleValue
           << "'; dispatch fallback executable RVV microkernels are not "
              "admitted by this first slice";

  auto elementCount =
      op->getAttrOfType<mlir::IntegerAttr>(kElementCountAttrName);
  if (!elementCount)
    return emitOpError()
           << "requires integer attribute '" << kElementCountAttrName << "'";
  int64_t count = elementCount.getInt();
  if (count <= 0 || count > 64)
    return emitOpError()
           << "element_count must be in the bounded smoke range [1, 64]";

  auto requiredMarch =
      op->getAttrOfType<mlir::StringAttr>(kRequiredMarchAttrName);
  if (!hasRVVVectorHint(requiredMarch.getValue()))
    return emitOpError()
           << "required_march must contain RVV vector evidence";

  auto selectedVariant =
      op->getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
  if (!selectedVariant || selectedVariant.getValue().trim().empty())
    return emitOpError()
           << "requires non-empty variant symbol reference attribute '"
           << kSelectedVariantAttrName << "'";

  auto requiredCapabilities =
      op->getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName);
  if (!requiredCapabilities || requiredCapabilities.empty())
    return emitOpError()
           << "requires non-empty array attribute '"
           << kRequiredCapabilitiesAttrName
           << "' containing capability symbol references";

  auto kernel = op->getParentOfType<tianchenrv::tcrv::exec::KernelOp>();
  if (!kernel)
    return emitOpError() << "must be nested directly in a tcrv.exec.kernel";
  if (op->getParentOp() != kernel.getOperation())
    return emitOpError()
           << "must be a direct child of the enclosing tcrv.exec.kernel";

  auto sourceKernel =
      op->getAttrOfType<mlir::StringAttr>(kSourceKernelAttrName);
  if (sourceKernel.getValue() != kernel.getSymName())
    return emitOpError()
           << "source_kernel must match enclosing tcrv.exec.kernel symbol @"
           << kernel.getSymName();

  if (kernel.getBody().empty())
    return emitOpError()
           << "requires enclosing tcrv.exec.kernel to have a body block";

  llvm::Expected<tianchenrv::support::TargetCapabilitySet>
      capabilitiesOrError =
          tianchenrv::support::TargetCapabilitySet::buildFromKernelChecked(
              kernel);
  if (!capabilitiesOrError) {
    std::string message = llvm::toString(capabilitiesOrError.takeError());
    return emitOpError() << message;
  }
  const tianchenrv::support::TargetCapabilitySet &capabilities =
      *capabilitiesOrError;

  tianchenrv::tcrv::exec::VariantOp resolvedVariant;
  for (mlir::Operation &sibling : kernel.getBody().front()) {
    auto variant =
        llvm::dyn_cast<tianchenrv::tcrv::exec::VariantOp>(sibling);
    if (variant && variant.getSymName() == selectedVariant.getValue()) {
      resolvedVariant = variant;
      break;
    }
  }
  if (!resolvedVariant)
    return emitOpError()
           << "selected_variant @" << selectedVariant.getValue()
           << " must resolve to a direct sibling tcrv.exec.variant in the "
              "enclosing tcrv.exec.kernel";

  auto variantOrigin =
      resolvedVariant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!variantOrigin || variantOrigin.getValue() != kRVVPluginName)
    return emitOpError()
           << "selected_variant must be owned by origin '" << kRVVPluginName
           << "'";

  auto variantRequiredMarch =
      resolvedVariant->getAttrOfType<mlir::StringAttr>(
          kRVVVariantRequiredMarchAttrName);
  if (!variantRequiredMarch || variantRequiredMarch.getValue().trim().empty())
    return emitOpError()
           << "selected_variant requires non-empty string metadata '"
           << kRVVVariantRequiredMarchAttrName << "'";
  if (variantRequiredMarch.getValue().trim() != requiredMarch.getValue().trim())
    return emitOpError()
           << "required_march must match selected variant '"
           << kRVVVariantRequiredMarchAttrName << "' metadata";

  auto variantRequires =
      resolvedVariant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!arrayAttrsEqual(requiredCapabilities, variantRequires))
    return emitOpError()
           << "required_capabilities must match selected variant requires "
              "metadata";

  bool requiresRVV = false;
  for (mlir::Attribute requiredCapability : requiredCapabilities) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return emitOpError()
             << "attribute '" << kRequiredCapabilitiesAttrName
             << "' must contain only capability symbol references";

    const tianchenrv::support::CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbolRef.getValue());
    if (!capability)
      return emitOpError()
             << "requires unknown capability @" << symbolRef.getValue()
             << " in enclosing tcrv.exec.kernel";

    if (!capability->isAvailable())
      return emitOpError()
             << "requires unavailable capability @" << symbolRef.getValue();

    if (capability->satisfiesID(kRVVCapabilityID))
      requiresRVV = true;
  }

  if (!requiresRVV)
    return emitOpError()
           << "required_capabilities must include capability id 'rvv'";

  if (mlir::failed(verifyMicrokernelStructuredControlPlane(op, family)))
    return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult verifyI64MicrokernelOp(
    mlir::Operation *op, const I64MicrokernelFamilySpec &family) {
  auto emitOpError = [&]() { return op->emitOpError(); };

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    if (!isAllowedMicrokernelAttr(attr.getName().getValue()))
      return emitOpError()
             << "does not accept generic tensor/tile/benchmark or unknown "
                "attribute '"
             << attr.getName()
             << "'; this op is exactly a bounded RVV i64 vector "
             << family.arithmeticVerb << " microkernel";
  }

  if (hasMissingOrEmptyStringAttr(op, kSourceKernelAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kSourceKernelAttrName
           << "'";
  if (hasMissingOrEmptyStringAttr(op, kOriginAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kOriginAttrName
           << "'";
  if (hasMissingOrEmptyStringAttr(op, kRoleAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kRoleAttrName << "'";
  if (hasMissingOrEmptyStringAttr(op, kRequiredMarchAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kRequiredMarchAttrName
           << "'";
  if (hasPresentButEmptyStringAttr(op, kSelectedMABIAttrName))
    return emitOpError()
           << "requires non-empty string attribute '" << kSelectedMABIAttrName
           << "' when present";

  for (llvm::StringRef attrName :
       {kSourceKernelAttrName, kOriginAttrName, kRoleAttrName,
        kRequiredMarchAttrName, kSelectedMABIAttrName}) {
    if (auto attr = op->getAttrOfType<mlir::StringAttr>(attrName))
      if (mlir::failed(
              verifyBoundedMetadata(op, attrName, attr.getValue().trim())))
        return mlir::failure();
  }

  auto origin = op->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (origin.getValue() != kRVVPluginName)
    return emitOpError()
           << "origin must be '" << kRVVPluginName
           << "' because this executable microkernel is RVV plugin-local";

  auto role = op->getAttrOfType<mlir::StringAttr>(kRoleAttrName);
  if (!isAllowedLoweringBoundaryRole(role.getValue()))
    return emitOpError()
           << "role must be '" << kDirectVariantRoleValue << "' or '"
           << kDispatchCaseRoleValue
           << "'; dispatch fallback executable RVV microkernels are not "
              "admitted by this first slice";

  auto elementCount =
      op->getAttrOfType<mlir::IntegerAttr>(kElementCountAttrName);
  if (!elementCount)
    return emitOpError()
           << "requires integer attribute '" << kElementCountAttrName << "'";
  int64_t count = elementCount.getInt();
  if (count <= 0 || count > 64)
    return emitOpError()
           << "element_count must be in the bounded smoke range [1, 64]";

  auto requiredMarch =
      op->getAttrOfType<mlir::StringAttr>(kRequiredMarchAttrName);
  if (!hasRVVVectorHint(requiredMarch.getValue()))
    return emitOpError()
           << "required_march must contain RVV vector evidence";

  auto selectedVariant =
      op->getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
  if (!selectedVariant || selectedVariant.getValue().trim().empty())
    return emitOpError()
           << "requires non-empty variant symbol reference attribute '"
           << kSelectedVariantAttrName << "'";

  auto requiredCapabilities =
      op->getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName);
  if (!requiredCapabilities || requiredCapabilities.empty())
    return emitOpError()
           << "requires non-empty array attribute '"
           << kRequiredCapabilitiesAttrName
           << "' containing capability symbol references";

  auto kernel = op->getParentOfType<tianchenrv::tcrv::exec::KernelOp>();
  if (!kernel)
    return emitOpError() << "must be nested directly in a tcrv.exec.kernel";
  if (op->getParentOp() != kernel.getOperation())
    return emitOpError()
           << "must be a direct child of the enclosing tcrv.exec.kernel";

  auto sourceKernel =
      op->getAttrOfType<mlir::StringAttr>(kSourceKernelAttrName);
  if (sourceKernel.getValue() != kernel.getSymName())
    return emitOpError()
           << "source_kernel must match enclosing tcrv.exec.kernel symbol @"
           << kernel.getSymName();

  if (kernel.getBody().empty())
    return emitOpError()
           << "requires enclosing tcrv.exec.kernel to have a body block";

  llvm::Expected<tianchenrv::support::TargetCapabilitySet>
      capabilitiesOrError =
          tianchenrv::support::TargetCapabilitySet::buildFromKernelChecked(
              kernel);
  if (!capabilitiesOrError) {
    std::string message = llvm::toString(capabilitiesOrError.takeError());
    return emitOpError() << message;
  }
  const tianchenrv::support::TargetCapabilitySet &capabilities =
      *capabilitiesOrError;

  tianchenrv::tcrv::exec::VariantOp resolvedVariant;
  for (mlir::Operation &sibling : kernel.getBody().front()) {
    auto variant =
        llvm::dyn_cast<tianchenrv::tcrv::exec::VariantOp>(sibling);
    if (variant && variant.getSymName() == selectedVariant.getValue()) {
      resolvedVariant = variant;
      break;
    }
  }
  if (!resolvedVariant)
    return emitOpError()
           << "selected_variant @" << selectedVariant.getValue()
           << " must resolve to a direct sibling tcrv.exec.variant in the "
              "enclosing tcrv.exec.kernel";

  auto variantOrigin =
      resolvedVariant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!variantOrigin || variantOrigin.getValue() != kRVVPluginName)
    return emitOpError()
           << "selected_variant must be owned by origin '" << kRVVPluginName
           << "'";

  auto variantRequiredMarch =
      resolvedVariant->getAttrOfType<mlir::StringAttr>(
          kRVVVariantRequiredMarchAttrName);
  if (!variantRequiredMarch || variantRequiredMarch.getValue().trim().empty())
    return emitOpError()
           << "selected_variant requires non-empty string metadata '"
           << kRVVVariantRequiredMarchAttrName << "'";
  if (variantRequiredMarch.getValue().trim() != requiredMarch.getValue().trim())
    return emitOpError()
           << "required_march must match selected variant '"
           << kRVVVariantRequiredMarchAttrName << "' metadata";

  auto variantRequires =
      resolvedVariant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!arrayAttrsEqual(requiredCapabilities, variantRequires))
    return emitOpError()
           << "required_capabilities must match selected variant requires "
              "metadata";

  bool requiresRVV = false;
  for (mlir::Attribute requiredCapability : requiredCapabilities) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return emitOpError()
             << "attribute '" << kRequiredCapabilitiesAttrName
             << "' must contain only capability symbol references";

    const tianchenrv::support::CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbolRef.getValue());
    if (!capability)
      return emitOpError()
             << "requires unknown capability @" << symbolRef.getValue()
             << " in enclosing tcrv.exec.kernel";

    if (!capability->isAvailable())
      return emitOpError()
             << "requires unavailable capability @" << symbolRef.getValue();

    if (capability->satisfiesID(kRVVCapabilityID))
      requiresRVV = true;
  }

  if (!requiresRVV)
    return emitOpError()
           << "required_capabilities must include capability id 'rvv'";

  if (mlir::failed(verifyI64MicrokernelStructuredControlPlane(op, family)))
    return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult I32VAddMicrokernelOp::verify() {
  return verifyI32MicrokernelOp(
      getOperation(),
      getI32MicrokernelFamilySpec(I32MicrokernelArithmetic::Add));
}

mlir::LogicalResult I32VSubMicrokernelOp::verify() {
  return verifyI32MicrokernelOp(
      getOperation(),
      getI32MicrokernelFamilySpec(I32MicrokernelArithmetic::Sub));
}

mlir::LogicalResult I32VMulMicrokernelOp::verify() {
  return verifyI32MicrokernelOp(
      getOperation(),
      getI32MicrokernelFamilySpec(I32MicrokernelArithmetic::Mul));
}

mlir::LogicalResult I64VAddMicrokernelOp::verify() {
  return verifyI64MicrokernelOp(
      getOperation(),
      getI64MicrokernelFamilySpec(I64MicrokernelArithmetic::Add));
}

mlir::LogicalResult I64VSubMicrokernelOp::verify() {
  return verifyI64MicrokernelOp(
      getOperation(),
      getI64MicrokernelFamilySpec(I64MicrokernelArithmetic::Sub));
}

mlir::LogicalResult I64VMulMicrokernelOp::verify() {
  return verifyI64MicrokernelOp(
      getOperation(),
      getI64MicrokernelFamilySpec(I64MicrokernelArithmetic::Mul));
}

void TCRVRVVDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TianChenRV/Dialect/RVV/IR/RVVOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "TianChenRV/Dialect/RVV/IR/RVVAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "TianChenRV/Dialect/RVV/IR/RVVTypes.cpp.inc"
      >();
}
