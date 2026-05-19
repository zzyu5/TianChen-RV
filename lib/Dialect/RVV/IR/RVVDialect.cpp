#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

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

constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kSelectedPathRoleAttrName("selected_path_role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRVVConstructionProtocolAttrName(
    "rvv_construction_protocol");
constexpr llvm::StringLiteral kRVVEmitCRouteMappingAttrName(
    "rvv_emitc_route_mapping");
constexpr llvm::StringLiteral kCapabilitySummaryAttrName(
    "capability_summary");
constexpr llvm::StringLiteral kAVLAttrName("avl");
constexpr llvm::StringLiteral kVLAttrName("vl");
constexpr llvm::StringLiteral kSEWAttrName("sew");
constexpr llvm::StringLiteral kLMULAttrName("lmul");
constexpr llvm::StringLiteral kPolicyAttrName("policy");
constexpr llvm::StringLiteral kElementCountAttrName("element_count");
constexpr llvm::StringLiteral kRequiredMarchAttrName("required_march");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kCNameAttrName("c_name");
constexpr llvm::StringLiteral kCTypeAttrName("c_type");
constexpr llvm::StringLiteral kOwnershipAttrName("ownership");
constexpr llvm::StringLiteral kPurposeAttrName("purpose");
constexpr llvm::StringLiteral kOpKindAttrName("op_kind");
constexpr llvm::StringLiteral kMemoryFormAttrName("memory_form");
constexpr llvm::StringLiteral kVLenAttrName("vlen");
constexpr llvm::StringLiteral kVLenBAttrName("vlenb");
constexpr llvm::StringLiteral kRVVVariantRequiredMarchAttrName(
    "tcrv_rvv.required_march");
constexpr llvm::StringLiteral kRVVRequiredCapabilitiesAttrName(
    "tcrv_rvv.required_capabilities");
constexpr llvm::StringLiteral kRVVVLenAttrName("tcrv_rvv.vlen");
constexpr llvm::StringLiteral kRVVVLenBAttrName("tcrv_rvv.vlenb");
constexpr llvm::StringLiteral kArchitectureAttrName("architecture");
constexpr llvm::StringLiteral kISAVectorHintsAttrName("isa_vector_hints");
constexpr llvm::StringLiteral kHartCountAttrName("hart_count");
constexpr llvm::StringLiteral kSelectedMarchAttrName("selected_march");
constexpr llvm::StringLiteral kCapabilityFactsAttrName("capability_facts");

bool containsForbiddenMetadataText(llvm::StringRef text) {
  std::string lowerText = text.lower();
  llvm::StringRef lower(lowerText);
  return lower.contains("password") || lower.contains("passwd") ||
         lower.contains("token") || lower.contains("secret") ||
         lower.contains("private key") ||
         lower.contains("authorization:") || lower.contains("api_key") ||
         lower.contains("access_key");
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

bool isAllowedSetVLAttr(llvm::StringRef name) {
  return name == kSEWAttrName || name == kLMULAttrName ||
         name == kPolicyAttrName;
}

bool isAllowedWithVLAttr(llvm::StringRef name) {
  return name == kSEWAttrName || name == kLMULAttrName ||
         name == kPolicyAttrName || name == kSourceKernelAttrName ||
         name == kSelectedVariantAttrName || name == kOriginAttrName ||
         name == kSelectedPathRoleAttrName || name == kStatusAttrName ||
         name == kRequiredCapabilitiesAttrName ||
         name == kRVVConstructionProtocolAttrName ||
         name == kRVVEmitCRouteMappingAttrName;
}

bool isAllowedI32LoadAttr(llvm::StringRef) {
  return false;
}

bool isAllowedI32BroadcastLoadAttr(llvm::StringRef) {
  return false;
}

bool isAllowedTypedBinaryPreRealizedBodyAttr(llvm::StringRef name) {
  return name == kOpKindAttrName || name == kMemoryFormAttrName ||
         name == kSEWAttrName || name == kLMULAttrName ||
         name == kPolicyAttrName;
}

bool isAllowedLoadAttr(llvm::StringRef) { return false; }

bool isAllowedBroadcastLoadAttr(llvm::StringRef) { return false; }

bool isAllowedStridedLoadAttr(llvm::StringRef) { return false; }

bool isAllowedBinaryAttr(llvm::StringRef name) {
  return name == "kind";
}

bool isAllowedMaskedBinaryAttr(llvm::StringRef name) {
  return name == "kind";
}

bool isAllowedCompareAttr(llvm::StringRef name) { return name == "kind"; }

bool isAllowedSelectAttr(llvm::StringRef) { return false; }

bool isAllowedReduceAttr(llvm::StringRef name) { return name == "kind"; }

bool isAllowedMAccAttr(llvm::StringRef name) { return name == "kind"; }

bool isAllowedStoreAttr(llvm::StringRef) { return false; }

bool isAllowedStridedStoreAttr(llvm::StringRef) { return false; }

bool isSupportedTypedBinaryPreRealizedBodyOpKind(llvm::StringRef opKind) {
  return opKind == "add" || opKind == "sub" || opKind == "mul";
}

bool isSupportedGenericBinaryKind(llvm::StringRef kind) {
  return kind == "add" || kind == "sub" || kind == "mul";
}

bool isSupportedGenericMaskedBinaryKind(llvm::StringRef kind) {
  return kind == "add";
}

bool isSupportedGenericCompareKind(llvm::StringRef kind) {
  return kind == "eq";
}

bool isSupportedGenericReduceKind(llvm::StringRef kind) {
  return kind == "add";
}

bool isSupportedGenericMAccKind(llvm::StringRef kind) {
  return kind == "add";
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

bool isAllowedI32CmpEqAttr(llvm::StringRef) {
  return false;
}

bool isAllowedI32SelectAttr(llvm::StringRef) {
  return false;
}

bool isAllowedI32StoreAttr(llvm::StringRef) {
  return false;
}

bool isAllowedRuntimeABIValueAttr(llvm::StringRef name) {
  return name == kRoleAttrName || name == kCNameAttrName ||
         name == kCTypeAttrName || name == kOwnershipAttrName ||
         name == kPurposeAttrName;
}

bool isForbiddenSetVLParameterAttr(llvm::StringRef name) {
  return name == kAVLAttrName || name == kVLenAttrName ||
         name == kVLenBAttrName || name == kElementCountAttrName ||
         name == kRequiredMarchAttrName ||
         name == kRequiredCapabilitiesAttrName ||
         name == kRVVVariantRequiredMarchAttrName ||
         name == kRVVRequiredCapabilitiesAttrName ||
         name == kRVVVLenAttrName || name == kRVVVLenBAttrName;
}

bool isForbiddenWithVLParameterAttr(llvm::StringRef name) {
  return name == kAVLAttrName || name == kVLAttrName ||
         name == kVLenAttrName || name == kVLenBAttrName ||
         name == kElementCountAttrName || name == kRequiredMarchAttrName ||
         name == kRVVVariantRequiredMarchAttrName ||
         name == kRVVRequiredCapabilitiesAttrName ||
         name == kRVVVLenAttrName || name == kRVVVLenBAttrName ||
         name == kCapabilitySummaryAttrName || name == kArchitectureAttrName ||
         name == kISAVectorHintsAttrName || name == kHartCountAttrName ||
         name == kSelectedMarchAttrName || name == kCapabilityFactsAttrName;
}

bool isForbiddenDataflowParameterAttr(llvm::StringRef name) {
  return isForbiddenWithVLParameterAttr(name) || name == kSEWAttrName ||
         name == kLMULAttrName || name == kPolicyAttrName;
}

bool isForbiddenPreRealizedBodyAuthorityAttr(llvm::StringRef name) {
  return name == kAVLAttrName || name == kVLAttrName ||
         name == kElementCountAttrName || name == kRequiredMarchAttrName ||
         name == kRVVVariantRequiredMarchAttrName ||
         name == kRVVRequiredCapabilitiesAttrName ||
         name == kRVVVLenAttrName || name == kRVVVLenBAttrName ||
         name == kCapabilitySummaryAttrName || name == kArchitectureAttrName ||
         name == kISAVectorHintsAttrName || name == kHartCountAttrName ||
         name == kSelectedMarchAttrName || name == kCapabilityFactsAttrName ||
         name == kSourceKernelAttrName || name == kSelectedVariantAttrName ||
         name == kOriginAttrName || name == kSelectedPathRoleAttrName ||
         name == kStatusAttrName || name == kRequiredCapabilitiesAttrName ||
         name == kRVVConstructionProtocolAttrName ||
         name == kRVVEmitCRouteMappingAttrName;
}

bool isSafeCIdentifier(llvm::StringRef value) {
  if (value.empty() || value.size() > 128)
    return false;
  auto isHead = [](char c) {
    unsigned char byte = static_cast<unsigned char>(c);
    return std::isalpha(byte) || c == '_';
  };
  auto isTail = [&](char c) {
    unsigned char byte = static_cast<unsigned char>(c);
    return isHead(c) || std::isdigit(byte);
  };
  if (!isHead(value.front()))
    return false;
  return llvm::all_of(value.drop_front(), isTail);
}

llvm::StringRef getBoundedRuntimeABIValueCType(
    tianchenrv::support::RuntimeABIParameterRole role) {
  using Role = tianchenrv::support::RuntimeABIParameterRole;
  switch (role) {
  case Role::LHSInputBuffer:
  case Role::RHSInputBuffer:
    return "const int32_t *";
  case Role::OutputBuffer:
    return "int32_t *";
  case Role::RuntimeElementCount:
  case Role::LHSInputStride:
  case Role::RHSInputStride:
  case Role::OutputStride:
    return "size_t";
  case Role::DispatchAvailabilityGuard:
    return {};
  }
  return {};
}

bool isBoundedInputBufferRole(
    tianchenrv::support::RuntimeABIParameterRole role) {
  using Role = tianchenrv::support::RuntimeABIParameterRole;
  return role == Role::LHSInputBuffer || role == Role::RHSInputBuffer;
}

bool isBoundedBufferRole(tianchenrv::support::RuntimeABIParameterRole role) {
  using Role = tianchenrv::support::RuntimeABIParameterRole;
  return isBoundedInputBufferRole(role) || role == Role::OutputBuffer;
}

bool isBoundedRuntimeIndexRole(
    tianchenrv::support::RuntimeABIParameterRole role) {
  using Role = tianchenrv::support::RuntimeABIParameterRole;
  return role == Role::RuntimeElementCount || role == Role::LHSInputStride ||
         role == Role::RHSInputStride || role == Role::OutputStride;
}

mlir::FailureOr<RuntimeABIValueOp>
verifyRuntimeABIValueOperand(mlir::Operation *op, mlir::Value value,
                             llvm::StringRef operandName) {
  if (!llvm::isa<RuntimeABIValueType>(value.getType()))
    return op->emitOpError()
           << "requires " << operandName
           << " operand to have !tcrv_rvv.runtime_abi_value type";

  auto binding = value.getDefiningOp<RuntimeABIValueOp>();
  if (!binding)
    return op->emitOpError()
           << "requires " << operandName
           << " operand to be defined by tcrv_rvv.runtime_abi_value";

  return binding;
}

mlir::LogicalResult verifyRuntimeABIValueOperandRole(
    mlir::Operation *op, mlir::Value value, llvm::StringRef operandName,
    llvm::ArrayRef<tianchenrv::support::RuntimeABIParameterRole>
        expectedRoles) {
  mlir::FailureOr<RuntimeABIValueOp> binding =
      verifyRuntimeABIValueOperand(op, value, operandName);
  if (mlir::failed(binding))
    return mlir::failure();

  std::optional<tianchenrv::support::RuntimeABIParameterRole> parsedRole =
      tianchenrv::support::symbolizeRuntimeABIParameterRole(
          (*binding).getRole());
  if (!parsedRole)
    return op->emitOpError()
           << "requires " << operandName
           << " operand runtime ABI role to be supported";

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
         << "requires " << operandName << " operand to bind runtime ABI role "
         << expected;
}

mlir::LogicalResult verifyRuntimeABIIndexOperandRole(
    mlir::Operation *op, mlir::Value value, llvm::StringRef operandName,
    llvm::ArrayRef<tianchenrv::support::RuntimeABIParameterRole>
        expectedRoles) {
  if (!value.getType().isIndex())
    return op->emitOpError()
           << "requires " << operandName << " operand to have index type";

  auto binding = value.getDefiningOp<RuntimeABIValueOp>();
  if (!binding)
    return op->emitOpError()
           << "requires " << operandName
           << " operand to be defined by tcrv_rvv.runtime_abi_value";

  std::optional<tianchenrv::support::RuntimeABIParameterRole> parsedRole =
      tianchenrv::support::symbolizeRuntimeABIParameterRole(
          binding.getRole());
  if (!parsedRole)
    return op->emitOpError()
           << "requires " << operandName
           << " operand runtime ABI role to be supported";

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
         << "requires " << operandName << " operand to bind runtime ABI role "
         << expected;
}

mlir::LogicalResult verifyRuntimeElementCountOperand(mlir::Operation *op,
                                                     mlir::Value value) {
  if (!value.getType().isIndex())
    return op->emitOpError()
           << "requires runtime n/AVL operand to have index type";

  auto binding = value.getDefiningOp<RuntimeABIValueOp>();
  if (!binding)
    return op->emitOpError()
           << "requires runtime n/AVL operand to be defined by "
              "tcrv_rvv.runtime_abi_value";

  std::optional<tianchenrv::support::RuntimeABIParameterRole> parsedRole =
      tianchenrv::support::symbolizeRuntimeABIParameterRole(
          binding.getRole());
  if (!parsedRole)
    return op->emitOpError()
           << "requires runtime n/AVL operand runtime ABI role to be "
              "supported";

  if (*parsedRole ==
      tianchenrv::support::RuntimeABIParameterRole::RuntimeElementCount)
    return mlir::success();

  return op->emitOpError()
         << "requires runtime n/AVL operand to bind runtime ABI role "
         << "'"
         << tianchenrv::support::stringifyRuntimeABIParameterRole(
                tianchenrv::support::RuntimeABIParameterRole::
                    RuntimeElementCount)
         << "'";
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

bool isI32M1Mask(mlir::Type type) {
  return llvm::isa<I32M1MaskType>(type);
}

llvm::StringRef getGenericRVVVectorLMUL(mlir::Type type) {
  auto vector = llvm::dyn_cast<tianchenrv::tcrv::rvv::VectorType>(type);
  if (!vector)
    return {};
  if (!vector.getElementType().isInteger(32))
    return {};
  if (vector.getLmul() == "m1" || vector.getLmul() == "m2")
    return vector.getLmul();
  return {};
}

llvm::StringRef getGenericRVVMaskLMUL(mlir::Type type) {
  auto mask = llvm::dyn_cast<tianchenrv::tcrv::rvv::MaskType>(type);
  if (!mask)
    return {};
  if (!mask.getElementType().isInteger(32))
    return {};
  if (mask.getLmul() == "m1" || mask.getLmul() == "m2")
    return mask.getLmul();
  return {};
}

mlir::LogicalResult verifyGenericVectorTypeForWithVL(mlir::Operation *op,
                                                     mlir::Value value,
                                                     llvm::StringRef role) {
  auto vector =
      llvm::dyn_cast<tianchenrv::tcrv::rvv::VectorType>(value.getType());
  if (!vector)
    return op->emitOpError()
           << "requires " << role
           << " type to be generic !tcrv_rvv.vector";
  if (!vector.getElementType().isInteger(32))
    return op->emitOpError()
           << "currently requires " << role
           << " element type to be i32 for the retained Stage 1 arithmetic "
              "route";

  llvm::StringRef valueLMUL = getGenericRVVVectorLMUL(value.getType());
  if (valueLMUL.empty())
    return op->emitOpError()
           << "requires " << role
           << " LMUL to be \"m1\" or \"m2\" for the retained Stage 1 "
              "arithmetic route";

  auto withVL = llvm::dyn_cast_or_null<WithVLOp>(op->getParentOp());
  if (!withVL)
    return mlir::success();

  auto expectedSEW =
      withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  if (!expectedSEW)
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit SEW "
              "metadata for generic RVV vector dataflow";
  if (expectedSEW.getInt() != getRVVFirstSliceSEWBits())
    return op->emitOpError()
           << "requires " << role
           << " element type i32 to agree with enclosing tcrv_rvv.with_vl "
              "SEW32 metadata";

  auto expectedLMUL =
      withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedLMUL)
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit LMUL "
              "metadata for generic RVV vector dataflow";
  if (expectedLMUL.getValue() != valueLMUL)
    return op->emitOpError()
           << "requires " << role << " type " << value.getType()
           << " to agree with enclosing tcrv_rvv.with_vl LMUL metadata '"
           << expectedLMUL.getValue() << "'";

  if (!withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for generic RVV vector dataflow";

  return mlir::success();
}

mlir::LogicalResult verifyGenericMaskTypeForWithVL(mlir::Operation *op,
                                                   mlir::Value value,
                                                   llvm::StringRef role) {
  auto mask =
      llvm::dyn_cast<tianchenrv::tcrv::rvv::MaskType>(value.getType());
  if (!mask)
    return op->emitOpError()
           << "requires " << role << " type to be generic !tcrv_rvv.mask";
  if (!mask.getElementType().isInteger(32))
    return op->emitOpError()
           << "currently requires " << role
           << " element type to be i32 for the bounded Stage 2 predicate "
              "route";

  llvm::StringRef valueLMUL = getGenericRVVMaskLMUL(value.getType());
  if (valueLMUL.empty())
    return op->emitOpError()
           << "requires " << role
           << " LMUL to be \"m1\" or \"m2\" for the bounded Stage 2 "
              "predicate route";

  auto withVL = llvm::dyn_cast_or_null<WithVLOp>(op->getParentOp());
  if (!withVL)
    return mlir::success();

  auto expectedSEW =
      withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  if (!expectedSEW)
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit SEW "
              "metadata for generic RVV mask dataflow";
  if (expectedSEW.getInt() != getRVVFirstSliceSEWBits())
    return op->emitOpError()
           << "requires " << role
           << " element type i32 to agree with enclosing tcrv_rvv.with_vl "
              "SEW32 metadata";

  auto expectedLMUL =
      withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedLMUL)
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit LMUL "
              "metadata for generic RVV mask dataflow";
  if (expectedLMUL.getValue() != valueLMUL)
    return op->emitOpError()
           << "requires " << role << " type " << value.getType()
           << " to agree with enclosing tcrv_rvv.with_vl LMUL metadata '"
           << expectedLMUL.getValue() << "'";

  if (!withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for generic RVV mask dataflow";

  return mlir::success();
}

mlir::LogicalResult verifyGenericMaskMatchesVector(mlir::Operation *op,
                                                   mlir::Value maskValue,
                                                   mlir::Value vectorValue,
                                                   llvm::StringRef maskRole,
                                                   llvm::StringRef vectorRole) {
  auto mask =
      llvm::dyn_cast<tianchenrv::tcrv::rvv::MaskType>(maskValue.getType());
  auto vector =
      llvm::dyn_cast<tianchenrv::tcrv::rvv::VectorType>(vectorValue.getType());
  if (!mask || !vector)
    return mlir::success();
  if (mask.getElementType() != vector.getElementType() ||
      mask.getLmul() != vector.getLmul())
    return op->emitOpError()
           << "requires " << maskRole << " type " << maskValue.getType()
           << " to agree with " << vectorRole << " type "
           << vectorValue.getType();
  return mlir::success();
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

  auto expectedSEW =
      withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  if (!expectedSEW)
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit SEW "
              "metadata for bounded RVV i32 dataflow";
  if (expectedSEW.getInt() != getRVVFirstSliceSEWBits())
    return op->emitOpError()
           << "requires " << role
           << " type to agree with enclosing tcrv_rvv.with_vl SEW32 "
              "metadata";

  auto expectedLMUL =
      withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedLMUL)
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit LMUL "
              "metadata for bounded RVV i32 dataflow";
  if (expectedLMUL.getValue() != valueLMUL)
    return op->emitOpError()
           << "requires " << role << " type " << value.getType()
           << " to agree with enclosing tcrv_rvv.with_vl LMUL metadata '"
           << expectedLMUL.getValue() << "'";

  if (!withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for bounded RVV i32 dataflow";

  return mlir::success();
}

mlir::LogicalResult verifyI32M1VectorTypeForWithVL(mlir::Operation *op,
                                                   mlir::Value value,
                                                   llvm::StringRef role) {
  if (!isI32M1Vector(value.getType()))
    return op->emitOpError()
           << "requires " << role << " type to be !tcrv_rvv.i32m1";
  return verifyI32VectorTypeForWithVL(op, value, role);
}

mlir::FailureOr<WithVLOp> verifyNestedDataflowOp(mlir::Operation *op) {
  auto withVL = llvm::dyn_cast_or_null<WithVLOp>(op->getParentOp());
  if (!withVL)
    return op->emitOpError()
           << "must be nested directly in a tcrv_rvv.with_vl body";

  if (op->getNumRegions() != 0)
    return op->emitOpError() << "does not own regions";

  return withVL;
}

mlir::LogicalResult verifyDataflowVLOperandMatchesWithVL(mlir::Operation *op,
                                                         mlir::Value vl) {
  auto withVL = llvm::dyn_cast_or_null<WithVLOp>(op->getParentOp());
  if (!withVL)
    return mlir::success();

  if (vl != withVL.getVl())
    return op->emitOpError()
           << "requires RVV dataflow op to consume the !tcrv_rvv.vl token "
              "owned by the surrounding tcrv_rvv.with_vl";

  return mlir::success();
}

mlir::LogicalResult verifyNoDataflowAttrs(mlir::Operation *op,
                                          llvm::StringRef opName,
                                          bool (*isAllowed)(llvm::StringRef)) {
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return op->emitOpError()
             << "does not accept attribute '" << attr.getName() << "'; "
             << opName
             << " keeps SEW/LMUL/policy on setvl/with_vl, runtime n/AVL/VL "
                "in the surrounding control-plane IR, and rejects deleted "
                "local element_count metadata";

    if (!isAllowed(attrName))
      return op->emitOpError()
             << "does not accept dataflow attributes; unexpected attribute '"
             << attr.getName() << "'";
  }
  return mlir::success();
}

} // namespace

llvm::StringRef RuntimeABIValueOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef RuntimeABIValueOp::getTCRVEmitCLowerableSourceRole() {
  return "runtime_abi";
}

llvm::StringRef SetVLOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef SetVLOp::getTCRVEmitCLowerableSourceRole() {
  return "configure";
}

llvm::StringRef WithVLOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef WithVLOp::getTCRVEmitCLowerableSourceRole() {
  return "scope";
}

llvm::StringRef I32LoadOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef I32LoadOp::getTCRVEmitCLowerableSourceRole() {
  return "load";
}

llvm::StringRef I32BroadcastLoadOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef I32BroadcastLoadOp::getTCRVEmitCLowerableSourceRole() {
  return "load";
}

llvm::StringRef LoadOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef LoadOp::getTCRVEmitCLowerableSourceRole() {
  return "load";
}

llvm::StringRef BroadcastLoadOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef BroadcastLoadOp::getTCRVEmitCLowerableSourceRole() {
  return "load";
}

llvm::StringRef StridedLoadOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef StridedLoadOp::getTCRVEmitCLowerableSourceRole() {
  return "load";
}

llvm::StringRef StoreOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef StoreOp::getTCRVEmitCLowerableSourceRole() {
  return "store";
}

llvm::StringRef StridedStoreOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef StridedStoreOp::getTCRVEmitCLowerableSourceRole() {
  return "store";
}

llvm::StringRef I32StoreOp::getTCRVEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef I32StoreOp::getTCRVEmitCLowerableSourceRole() {
  return "store";
}

mlir::LogicalResult RuntimeABIValueOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (!isAllowedRuntimeABIValueAttr(attrName))
      return emitOpError()
             << "only accepts runtime ABI binding attributes '" << kRoleAttrName
             << "', '" << kCNameAttrName << "', '" << kCTypeAttrName
             << "', '" << kOwnershipAttrName << "', and optional '"
             << kPurposeAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumResults() != 1)
    return emitOpError() << "requires exactly one SSA result";

  if (mlir::failed(verifyBoundedMetadata(op, kRoleAttrName, getRole())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kCNameAttrName, getCName())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kCTypeAttrName, getCType())))
    return mlir::failure();
  if (mlir::failed(
          verifyBoundedMetadata(op, kOwnershipAttrName, getOwnership())))
    return mlir::failure();
  if (auto purpose = op->getAttrOfType<mlir::StringAttr>(kPurposeAttrName))
    if (mlir::failed(
            verifyBoundedMetadata(op, kPurposeAttrName, purpose.getValue())))
      return mlir::failure();

  if (!isSafeCIdentifier(getCName()))
    return emitOpError()
           << "requires attribute '" << kCNameAttrName
           << "' to be a valid bounded C identifier";

  std::optional<tianchenrv::support::RuntimeABIParameterRole> parsedRole =
      tianchenrv::support::symbolizeRuntimeABIParameterRole(getRole());
  if (!parsedRole)
    return emitOpError() << "attribute '" << kRoleAttrName
                         << "' must reference a supported runtime ABI "
                            "parameter role";

  std::optional<tianchenrv::support::RuntimeABIParameterOwnership>
      parsedOwnership =
          tianchenrv::support::symbolizeRuntimeABIParameterOwnership(
              getOwnership());
  if (!parsedOwnership)
    return emitOpError() << "attribute '" << kOwnershipAttrName
                         << "' must reference a supported runtime ABI "
                            "parameter ownership";
  if (*parsedOwnership !=
      tianchenrv::support::RuntimeABIParameterOwnership::TargetExportABIOwned)
    return emitOpError()
           << "requires ownership '"
           << tianchenrv::support::stringifyRuntimeABIParameterOwnership(
                  tianchenrv::support::RuntimeABIParameterOwnership::
                      TargetExportABIOwned)
           << "' for the bounded RVV callable C ABI";

  llvm::StringRef expectedCType =
      getBoundedRuntimeABIValueCType(*parsedRole);
  if (expectedCType.empty())
    return emitOpError()
           << "does not support runtime ABI role '" << getRole()
           << "' in the bounded RVV callable ABI";
  if (getCType() != expectedCType)
    return emitOpError()
           << "requires runtime ABI role '" << getRole()
           << "' to use C type '" << expectedCType << "'";

  if (isBoundedRuntimeIndexRole(*parsedRole)) {
    if (!getValue().getType().isIndex())
      return emitOpError() << "requires runtime ABI role '" << getRole()
                           << "' result to have index type";
    return mlir::success();
  }

  if (isBoundedBufferRole(*parsedRole) &&
      llvm::isa<RuntimeABIValueType>(getValue().getType()))
    return mlir::success();

  return emitOpError()
         << "requires buffer ABI value result to have "
            "!tcrv_rvv.runtime_abi_value type";
}

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
                "facts, rejects deleted local element_count metadata, and "
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

  if (!isRVVFirstSliceDataflowConfig(static_cast<std::int64_t>(getSew()),
                                     getLmul()))
    return emitOpError()
           << "requires bounded RVV first-slice compile-time config to be "
              "SEW32 with LMUL \"m1\" or \"m2\"";

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
                "facts, rejects deleted local element_count metadata, "
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
      !isRVVFirstSliceDataflowConfig(sew.getInt(), lmul.getValue()))
    return emitOpError()
           << "requires bounded RVV first-slice compile-time config to be "
              "SEW32 with LMUL \"m1\" or \"m2\"";
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

  for (llvm::StringRef attrName :
       {kSourceKernelAttrName, kOriginAttrName, kSelectedPathRoleAttrName,
        kStatusAttrName, kRVVConstructionProtocolAttrName,
        kRVVEmitCRouteMappingAttrName}) {
    if (auto attr = op->getAttrOfType<mlir::StringAttr>(attrName))
      if (mlir::failed(verifyBoundedMetadata(op, attrName, attr.getValue())))
        return mlir::failure();
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
                "control-plane IR, and rejects deleted local element_count "
                "metadata";

    if (!isAllowedI32LoadAttr(attrName))
      return emitOpError()
             << "does not accept dataflow attributes; input buffer ABI "
                "provenance must come from the explicit buffer SSA operand; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires exactly one explicit input buffer ABI operand, one "
              "!tcrv_rvv.vl operand, and one bounded RVV i32 vector result";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "input buffer",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(
          verifyI32VectorTypeForWithVL(op, getLoaded(), "result")))
    return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult I32BroadcastLoadOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.i32_broadcast_load keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";

    if (!isAllowedI32BroadcastLoadAttr(attrName))
      return emitOpError()
             << "does not accept dataflow attributes; broadcast RHS ABI "
                "provenance must come from the explicit buffer SSA operand; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires exactly one explicit RHS buffer ABI operand, one "
              "!tcrv_rvv.vl operand, and one bounded RVV i32 vector result";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "broadcast RHS buffer",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(
          verifyI32VectorTypeForWithVL(op, getBroadcast(), "result")))
    return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult TypedBinaryPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected bodies carry only typed RVV "
                "operation/config/memory/runtime SSA facts and must be "
                "realized by the RVV plugin before route construction";

    if (!isAllowedTypedBinaryPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs, out, and runtime n/AVL operands and no "
              "results";

  if (!isSupportedTypedBinaryPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind \"add\", \"sub\", or "
              "\"mul\" for the bounded selected-body realization hook";
  if (getMemoryForm() != "vector-rhs-load")
    return emitOpError()
           << "currently supports only memory_form \"vector-rhs-load\" for "
              "the bounded selected-body realization hook";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body realization hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult LoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.load",
                                         isAllowedLoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires exactly one explicit buffer ABI operand, one "
              "!tcrv_rvv.vl operand, and one generic RVV vector result";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "buffer",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getLoaded(), "result");
}

mlir::LogicalResult BroadcastLoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.broadcast_load",
                                         isAllowedBroadcastLoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires exactly one explicit RHS buffer ABI operand, one "
              "!tcrv_rvv.vl operand, and one generic RVV vector result";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "broadcast RHS buffer",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getBroadcast(), "result");
}

mlir::LogicalResult StridedLoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.strided_load",
                                         isAllowedStridedLoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one explicit input buffer ABI operand, one runtime "
              "stride operand, one !tcrv_rvv.vl operand, and one generic RVV "
              "vector result";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "strided load buffer",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIIndexOperandRole(
          op, getStride(), "strided load stride",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputStride,
           tianchenrv::support::RuntimeABIParameterRole::RHSInputStride})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getLoaded(), "result");
}

mlir::LogicalResult BinaryOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.binary keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedBinaryAttr(attrName))
      return emitOpError()
             << "only accepts generic binary attribute 'kind"
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!isSupportedGenericBinaryKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"add\", \"sub\", or \"mul\" "
              "for the retained Stage 1 arithmetic route";

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs/rhs generic RVV vector operands, one "
              "!tcrv_rvv.vl operand, and one generic RVV vector result";
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getResult().getType())
    return emitOpError()
           << "requires lhs, rhs, and result to have the same generic RVV "
              "vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getResult(), "result");
}

mlir::LogicalResult MaskedBinaryOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.masked_binary keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";

    if (!isAllowedMaskedBinaryAttr(attrName))
      return emitOpError()
             << "only accepts generic masked binary attribute 'kind"
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!isSupportedGenericMaskedBinaryKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"add\" for the bounded Stage 2 "
              "masked arithmetic route";

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one generic RVV mask predicate, one passthrough "
              "generic RVV vector, lhs/rhs generic RVV vector operands, one "
              "!tcrv_rvv.vl operand, and one generic RVV vector result";
  if (getPassthrough().getType() != getLhs().getType() ||
      getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getResult().getType())
    return emitOpError()
           << "requires passthrough, lhs, rhs, and result to have the same "
              "generic RVV vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto compare = getMask().getDefiningOp<CompareOp>();
  if (!compare)
    return emitOpError()
           << "requires mask operand to be produced by tcrv_rvv.compare "
              "inside the selected RVV typed body";
  if (compare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to consume the same "
              "!tcrv_rvv.vl token as tcrv_rvv.masked_binary";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to be in the same "
              "tcrv_rvv.with_vl body as tcrv_rvv.masked_binary";

  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getPassthrough(),
                                                    "passthrough")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getResult(),
                                                    "result")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getResult(), "mask",
                                        "result");
}

mlir::LogicalResult CompareOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.compare keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedCompareAttr(attrName))
      return emitOpError()
             << "only accepts generic compare attribute 'kind"
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!isSupportedGenericCompareKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"eq\" for the bounded Stage 2 "
              "predicate route";

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs/rhs generic RVV vector operands, one "
              "!tcrv_rvv.vl operand, and one generic RVV mask result";
  if (getLhs().getType() != getRhs().getType())
    return emitOpError()
           << "requires lhs and rhs to have the same generic RVV vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "result")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getLhs(), "result",
                                        "lhs");
}

mlir::LogicalResult SelectOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.select",
                                         isAllowedSelectAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one generic RVV mask predicate, true/false generic "
              "RVV vector operands, one !tcrv_rvv.vl operand, and one "
              "generic RVV vector result";
  if (getTrueValue().getType() != getFalseValue().getType() ||
      getTrueValue().getType() != getSelected().getType())
    return emitOpError()
           << "requires true, false, and result to have the same generic RVV "
              "vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto compare = getMask().getDefiningOp<CompareOp>();
  if (!compare)
    return emitOpError()
           << "requires mask operand to be produced by tcrv_rvv.compare "
              "inside the selected RVV typed body";
  if (compare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to consume the same "
              "!tcrv_rvv.vl token as tcrv_rvv.select";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to be in the same "
              "tcrv_rvv.with_vl body as tcrv_rvv.select";

  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getTrueValue(),
                                                    "true value")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getFalseValue(),
                                                    "false value")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getSelected(),
                                                    "result")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getSelected(), "mask",
                                        "result");
}

mlir::LogicalResult ReduceOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.reduce keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedReduceAttr(attrName))
      return emitOpError()
             << "only accepts generic reduction attribute 'kind"
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!isSupportedGenericReduceKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"add\" for the bounded Stage 2 "
              "reduction/accumulation route";

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one input generic RVV vector operand, one "
              "accumulator generic RVV vector operand, one !tcrv_rvv.vl "
              "operand, and one generic RVV vector result";
  if (getInput().getType() != getAccumulator().getType() ||
      getInput().getType() != getResult().getType())
    return emitOpError()
           << "requires input, accumulator, and result to have the same "
              "generic RVV vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getInput(), "input")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getAccumulator(),
                                                    "accumulator")))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getResult(), "result");
}

mlir::LogicalResult MAccOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.macc keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedMAccAttr(attrName))
      return emitOpError()
             << "only accepts generic multiply-accumulate attribute 'kind"
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!isSupportedGenericMAccKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"add\" for the bounded Stage 2 "
              "multiply-accumulate route";

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs, rhs, and accumulator generic RVV vector "
              "operands, one !tcrv_rvv.vl operand, and one generic RVV "
              "vector result";
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getAccumulator().getType() ||
      getLhs().getType() != getResult().getType())
    return emitOpError()
           << "requires lhs, rhs, accumulator, and result to have the same "
              "generic RVV vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getAccumulator(),
                                                    "accumulator")))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getResult(), "result");
}

mlir::LogicalResult StoreOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.store",
                                         isAllowedStoreAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 3 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one explicit output buffer ABI operand, one generic "
              "RVV vector value operand, one !tcrv_rvv.vl operand, and no "
              "results";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "output buffer",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getValue(), "stored value");
}

mlir::LogicalResult StridedStoreOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.strided_store",
                                         isAllowedStridedStoreAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one explicit output buffer ABI operand, one generic "
              "RVV vector value operand, one runtime output stride operand, "
              "one !tcrv_rvv.vl operand, and no results";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "strided store output buffer",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIIndexOperandRole(
          op, getStride(), "strided store stride",
          {tianchenrv::support::RuntimeABIParameterRole::OutputStride})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getValue(), "stored value");
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
                "rejects deleted local element_count metadata";

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
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
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
                "rejects deleted local element_count metadata";

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
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
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
                "rejects deleted local element_count metadata";

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
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  return verifyI32VectorTypeForWithVL(op, getProduct(), "result");
}

mlir::LogicalResult I32CmpEqOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(
          verifyNoDataflowAttrs(op, "tcrv_rvv.i32_cmp_eq",
                                isAllowedI32CmpEqAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs/rhs !tcrv_rvv.i32m1 operands, one "
              "!tcrv_rvv.vl operand, and one !tcrv_rvv.i32m1_mask result";
  if (getLhs().getType() != getRhs().getType())
    return emitOpError()
           << "requires lhs and rhs to have the same bounded RVV i32m1 "
              "vector type";
  if (!isI32M1Mask(getMask().getType()))
    return emitOpError()
           << "requires result type to be !tcrv_rvv.i32m1_mask";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyI32M1VectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  return verifyI32M1VectorTypeForWithVL(op, getRhs(), "rhs");
}

mlir::LogicalResult I32SelectOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(
          verifyNoDataflowAttrs(op, "tcrv_rvv.i32_select",
                                isAllowedI32SelectAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one !tcrv_rvv.i32m1_mask predicate, true/false "
              "!tcrv_rvv.i32m1 operands, one !tcrv_rvv.vl operand, and one "
              "!tcrv_rvv.i32m1 result";
  if (!isI32M1Mask(getMask().getType()))
    return emitOpError()
           << "requires mask operand type to be !tcrv_rvv.i32m1_mask";
  if (!isI32M1Vector(getTrueValue().getType()) ||
      !isI32M1Vector(getFalseValue().getType()) ||
      !isI32M1Vector(getSelected().getType()))
    return emitOpError()
           << "requires true, false, and result types to be "
              "!tcrv_rvv.i32m1";
  if (getTrueValue().getType() != getFalseValue().getType() ||
      getTrueValue().getType() != getSelected().getType())
    return emitOpError()
           << "requires true, false, and result to have the same bounded "
              "RVV i32m1 vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto compare = getMask().getDefiningOp<I32CmpEqOp>();
  if (!compare)
    return emitOpError()
           << "requires mask operand to be produced by tcrv_rvv.i32_cmp_eq "
              "inside the selected RVV typed body";
  if (compare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.i32_cmp_eq to consume the "
              "same !tcrv_rvv.vl token as tcrv_rvv.i32_select";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.i32_cmp_eq to be in the "
              "same tcrv_rvv.with_vl body as tcrv_rvv.i32_select";

  if (mlir::failed(
          verifyI32M1VectorTypeForWithVL(op, getTrueValue(), "true value")))
    return mlir::failure();
  if (mlir::failed(
          verifyI32M1VectorTypeForWithVL(op, getFalseValue(), "false value")))
    return mlir::failure();
  return verifyI32M1VectorTypeForWithVL(op, getSelected(), "result");
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
                "control-plane IR, and rejects deleted local element_count "
                "metadata";

    if (!isAllowedI32StoreAttr(attrName))
      return emitOpError()
             << "does not accept dataflow attributes; output buffer ABI "
                "provenance must come from the explicit buffer SSA operand; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 3 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one explicit output buffer ABI operand, one bounded "
              "RVV i32 vector value operand, one !tcrv_rvv.vl operand, and "
              "no results";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "output buffer",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getValue(), "stored value")))
    return mlir::failure();

  return mlir::success();
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
