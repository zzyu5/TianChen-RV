#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
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
         name == kPolicyAttrName;
}

bool isAllowedI32LoadAttr(llvm::StringRef) {
  return false;
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

bool isSupportedRVVFirstSliceConfig(std::int64_t sew, llvm::StringRef lmul) {
  return sew == 32 && isSupportedI32LMUL(lmul);
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
  if (expectedSEW.getInt() != 32)
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

} // namespace

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

  if (*parsedRole ==
      tianchenrv::support::RuntimeABIParameterRole::RuntimeElementCount) {
    if (!getValue().getType().isIndex())
      return emitOpError()
             << "requires runtime-element-count result to have index type";
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

  if (!isSupportedRVVFirstSliceConfig(static_cast<std::int64_t>(getSew()),
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
      !isSupportedRVVFirstSliceConfig(sew.getInt(), lmul.getValue()))
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
