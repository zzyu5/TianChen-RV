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
constexpr llvm::StringLiteral kBoundaryI32M1LanesAttrName("i32_m1_lanes");
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
         name == kRequiredMarchAttrName || name == kSelectedMABIAttrName;
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

bool isAllowedI32StoreAttr(llvm::StringRef name) {
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

mlir::LogicalResult verifyNestedI32VAddDataflowOp(mlir::Operation *op) {
  if (!llvm::isa_and_nonnull<WithVLOp>(op->getParentOp()))
    return op->emitOpError()
           << "must be nested directly in a tcrv_rvv.with_vl body";
  if (!op->getParentOfType<I32VAddMicrokernelOp>())
    return op->emitOpError()
           << "must be nested under tcrv_rvv.i32_vadd_microkernel; it is a "
              "finite microkernel dataflow op, not a standalone RVV compute op";

  if (op->getNumRegions() != 0)
    return op->emitOpError() << "does not own regions";

  return mlir::success();
}

mlir::LogicalResult
verifyMicrokernelStructuredControlPlane(I32VAddMicrokernelOp microkernel) {
  mlir::Region &body = microkernel.getBody();
  if (body.empty() || !llvm::hasSingleElement(body))
    return microkernel.emitOpError()
           << "requires exactly one structured RVV control-plane body block";

  mlir::Block &block = body.front();
  if (block.getNumArguments() != 1)
    return microkernel.emitOpError()
           << "requires structured control-plane body to have exactly one "
              "runtime index block argument for target/export-owned n/AVL";
  if (!block.getArgument(0).getType().isIndex())
    return microkernel.emitOpError()
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
    return microkernel.emitOpError()
           << "structured control-plane body accepts only "
              "tcrv_rvv.setvl and tcrv_rvv.with_vl direct operations; "
              "unexpected operation '"
           << bodyOp.getName().getStringRef() << "'";
  }

  if (setvlCount != 1)
    return microkernel.emitOpError()
           << "requires exactly one tcrv_rvv.setvl in the structured "
              "control-plane body";
  if (withVLCount != 1)
    return microkernel.emitOpError()
           << "requires exactly one tcrv_rvv.with_vl in the structured "
              "control-plane body";

  if (setvl.getAvl() != block.getArgument(0))
    return microkernel.emitOpError()
           << "requires tcrv_rvv.setvl AVL operand to be the runtime index "
              "body argument, not descriptor-local element_count or a "
              "constant";
  if (withVL.getVl() != setvl.getVl())
    return microkernel.emitOpError()
           << "requires tcrv_rvv.with_vl to consume the !tcrv_rvv.vl token "
              "produced by the body tcrv_rvv.setvl";

  if (!withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName) ||
      !withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName) ||
      !withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName)) {
    return microkernel.emitOpError()
           << "requires tcrv_rvv.with_vl to carry explicit SEW/LMUL/policy "
              "control metadata so emission can consume the structured body";
  }

  mlir::Region &withVLBody = withVL.getBody();
  if (withVLBody.empty() || !llvm::hasSingleElement(withVLBody))
    return microkernel.emitOpError()
           << "requires tcrv_rvv.with_vl to own one body block";
  if (withVLBody.front().getNumArguments() != 0)
    return microkernel.emitOpError()
           << "requires tcrv_rvv.with_vl body to have no block arguments; "
              "runtime n/AVL/VL is carried by the enclosing control-plane "
              "surface";

  llvm::SmallVector<mlir::Operation *, 4> ops;
  for (mlir::Operation &withVLOp : withVLBody.front())
    ops.push_back(&withVLOp);
  if (ops.size() != 4)
    return microkernel.emitOpError()
           << "requires tcrv_rvv.with_vl body to contain exactly the finite "
              "tcrv_rvv.i32_load, tcrv_rvv.i32_load, tcrv_rvv.i32_add, "
              "tcrv_rvv.i32_store dataflow sequence";

  auto lhsLoad = llvm::dyn_cast<I32LoadOp>(ops[0]);
  auto rhsLoad = llvm::dyn_cast<I32LoadOp>(ops[1]);
  auto add = llvm::dyn_cast<I32AddOp>(ops[2]);
  auto store = llvm::dyn_cast<I32StoreOp>(ops[3]);
  if (!lhsLoad || !rhsLoad || !add || !store)
    return microkernel.emitOpError()
           << "requires tcrv_rvv.with_vl body to contain exactly the finite "
              "tcrv_rvv.i32_load, tcrv_rvv.i32_load, tcrv_rvv.i32_add, "
              "tcrv_rvv.i32_store dataflow sequence";

  if (lhsLoad.getVl() != withVL.getVl() || rhsLoad.getVl() != withVL.getVl() ||
      add.getVl() != withVL.getVl() || store.getVl() != withVL.getVl())
    return microkernel.emitOpError()
           << "requires every finite RVV i32 dataflow op to consume the "
              "!tcrv_rvv.vl token owned by the surrounding tcrv_rvv.with_vl";

  auto lhsRole = lhsLoad->getAttrOfType<mlir::StringAttr>(kBufferRoleAttrName);
  auto rhsRole = rhsLoad->getAttrOfType<mlir::StringAttr>(kBufferRoleAttrName);
  auto outRole = store->getAttrOfType<mlir::StringAttr>(kBufferRoleAttrName);
  if (!lhsRole || lhsRole.getValue() !=
                      tianchenrv::support::stringifyRuntimeABIParameterRole(
                          tianchenrv::support::RuntimeABIParameterRole::
                              LHSInputBuffer))
    return microkernel.emitOpError()
           << "requires first tcrv_rvv.i32_load to reference runtime ABI role "
              "'lhs-input-buffer'";
  if (!rhsRole || rhsRole.getValue() !=
                      tianchenrv::support::stringifyRuntimeABIParameterRole(
                          tianchenrv::support::RuntimeABIParameterRole::
                              RHSInputBuffer))
    return microkernel.emitOpError()
           << "requires second tcrv_rvv.i32_load to reference runtime ABI role "
              "'rhs-input-buffer'";
  if (!outRole || outRole.getValue() !=
                      tianchenrv::support::stringifyRuntimeABIParameterRole(
                          tianchenrv::support::RuntimeABIParameterRole::
                              OutputBuffer))
    return microkernel.emitOpError()
           << "requires tcrv_rvv.i32_store to reference runtime ABI role "
              "'output-buffer'";

  if (add.getLhs() != lhsLoad.getLoaded() || add.getRhs() != rhsLoad.getLoaded() ||
      store.getValue() != add.getSum())
    return microkernel.emitOpError()
           << "requires finite RVV i32 dataflow SSA chain "
              "lhs-load,rhs-load -> add -> store";

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

  if (getSew() != 32)
    return emitOpError()
           << "requires SEW compile-time config 'sew' to be 32 for the "
              "bounded RVV first slice";

  if (getLmul() != "m1")
    return emitOpError()
           << "requires LMUL compile-time config 'lmul' to be \"m1\" for the "
              "bounded RVV first slice";

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
  if (sew && sew.getInt() != 32)
    return emitOpError()
           << "requires optional SEW compile-time config 'sew' to be 32 for "
              "the bounded RVV first slice";

  auto lmul = op->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (lmul && lmul.getValue() != "m1")
    return emitOpError()
           << "requires optional LMUL compile-time config 'lmul' to be "
              "\"m1\" for the bounded RVV first slice";

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
              "!tcrv_rvv.i32m1 result";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (!isI32M1Vector(getLoaded().getType()))
    return emitOpError() << "requires result type to be !tcrv_rvv.i32m1";
  if (mlir::failed(verifyNestedI32VAddDataflowOp(op)))
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
           << "requires lhs/rhs !tcrv_rvv.i32m1 operands, one "
              "!tcrv_rvv.vl operand, and one !tcrv_rvv.i32m1 result";
  if (!isI32M1Vector(getLhs().getType()) ||
      !isI32M1Vector(getRhs().getType()) ||
      !isI32M1Vector(getSum().getType()))
    return emitOpError()
           << "requires lhs, rhs, and result types to be !tcrv_rvv.i32m1";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  return verifyNestedI32VAddDataflowOp(op);
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
           << "requires one !tcrv_rvv.i32m1 value operand, one !tcrv_rvv.vl "
              "operand, and no results";
  if (!isI32M1Vector(getValue().getType()))
    return emitOpError() << "requires stored value type to be !tcrv_rvv.i32m1";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedI32VAddDataflowOp(op)))
    return mlir::failure();

  llvm::StringSet<> seenRoles;
  if (mlir::failed(verifyBoundedDataflowRoleAttr(
          op, kBufferRoleAttrName,
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer},
          &seenRoles)))
    return mlir::failure();

  return mlir::success();
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
             << "selected capacity metadata requires i32_m1_lanes to equal "
                "vlenb_bytes divided by four";
  }

  return mlir::success();
}

mlir::LogicalResult I32VAddMicrokernelOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    if (!isAllowedMicrokernelAttr(attr.getName().getValue()))
      return emitOpError()
             << "does not accept generic tensor/tile/benchmark or unknown "
                "attribute '"
             << attr.getName()
             << "'; this op is exactly a bounded RVV i32 vector-add "
                "microkernel";
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

  if (mlir::failed(verifyMicrokernelStructuredControlPlane(*this)))
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
