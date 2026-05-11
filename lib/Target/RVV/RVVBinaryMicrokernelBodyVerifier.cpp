#include "TianChenRV/Target/RVV/RVVBinaryMicrokernelBodyVerifier.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>

namespace tianchenrv::target::rvv {
namespace {

using tianchenrv::support::RuntimeABIParameterRole;
using tianchenrv::conversion::emitc::TCRVEmitCLowerableOpInterface;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::rvv::I32AddOp;
using tianchenrv::tcrv::rvv::I32LoadOp;
using tianchenrv::tcrv::rvv::I32M1VectorType;
using tianchenrv::tcrv::rvv::I32M2VectorType;
using tianchenrv::tcrv::rvv::I32MulOp;
using tianchenrv::tcrv::rvv::I32StoreOp;
using tianchenrv::tcrv::rvv::I32SubOp;
using tianchenrv::tcrv::rvv::I64AddOp;
using tianchenrv::tcrv::rvv::I64LoadOp;
using tianchenrv::tcrv::rvv::I64M1VectorType;
using tianchenrv::tcrv::rvv::I64MulOp;
using tianchenrv::tcrv::rvv::I64StoreOp;
using tianchenrv::tcrv::rvv::I64SubOp;
using tianchenrv::tcrv::rvv::MaskPolicy;
using tianchenrv::tcrv::rvv::PolicyAttr;
using tianchenrv::tcrv::rvv::SetVLOp;
using tianchenrv::tcrv::rvv::TailPolicy;
using tianchenrv::tcrv::rvv::WithVLOp;

constexpr llvm::StringLiteral kBufferRoleAttrName("buffer_role");
constexpr llvm::StringLiteral kElementCountAttrName("element_count");
constexpr llvm::StringLiteral kSEWAttrName("sew");
constexpr llvm::StringLiteral kLMULAttrName("lmul");
constexpr llvm::StringLiteral kPolicyAttrName("policy");
constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "TCRVEmitCLowerableOpInterface");

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

std::string stringifyType(mlir::Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  type.print(os);
  os.flush();
  return text;
}

llvm::Error makeBodyVerifierError(
    const RVVBinaryMicrokernelBodyValidationRequest &request,
    llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "TianChen-RV RVV microkernel body verifier failed";
  KernelOp kernel = request.kernel;
  if (kernel)
    os << " for kernel @" << kernel.getSymName();
  if (!request.descriptor.getRVVMicrokernelOpName().empty())
    os << " family '" << request.descriptor.getRVVMicrokernelOpName() << "'";
  if (!request.activeRouteID.empty())
    os << " on route '" << request.activeRouteID << "'";
  os << ": ";
  message.print(os);
  os.flush();
  return llvm::createStringError(llvm::errc::invalid_argument, text);
}

bool isBoundedPrintable(llvm::StringRef value) {
  if (value.empty() || value.size() > 128)
    return false;
  for (char c : value)
    if (!std::isprint(static_cast<unsigned char>(c)))
      return false;
  return true;
}

llvm::Error validateBoundedText(
    const RVVBinaryMicrokernelBodyValidationRequest &request,
    llvm::StringRef layer, llvm::StringRef value) {
  if (isBoundedPrintable(value))
    return llvm::Error::success();
  return makeBodyVerifierError(
      request, llvm::Twine(layer) +
                   " metadata must be non-empty bounded printable text");
}

mlir::Type getExpectedVectorType(mlir::MLIRContext *context,
                                 const RVVBinaryIntrinsicDescriptor &descriptor) {
  if (!descriptor.shape)
    return {};
  if (descriptor.getDTypeID() == "i32") {
    if (descriptor.getLMUL() == "m2")
      return I32M2VectorType::get(context);
    if (descriptor.getLMUL() == "m1")
      return I32M1VectorType::get(context);
    return {};
  }
  if (descriptor.getDTypeID() == "i64" && descriptor.getLMUL() == "m1")
    return I64M1VectorType::get(context);
  return {};
}

llvm::Error validateVectorValueType(
    const RVVBinaryMicrokernelBodyValidationRequest &request,
    mlir::Value value, mlir::Type expectedType, llvm::StringRef context) {
  if (value.getType() == expectedType)
    return llvm::Error::success();
  return makeBodyVerifierError(
      request, llvm::Twine(context) +
                   " must use selected RVV vector token type '" +
                   stringifyType(expectedType) +
                   "' from the selected descriptor/config layer before "
                   "artifact export");
}

RVVBinaryDataflowStep makeLoadStep(llvm::StringRef sourceOpName,
                                   RuntimeABIParameterRole role,
                                   RVVBinaryDataflowValue result) {
  RVVBinaryDataflowStep step;
  step.kind = RVVBinaryDataflowStepKind::Load;
  step.sourceOpName = sourceOpName.str();
  step.bufferRole = role;
  step.result = result;
  return step;
}

RVVBinaryDataflowStep makeArithmeticStep(llvm::StringRef sourceOpName,
                                         llvm::StringRef sourceOpRole,
                                         llvm::StringRef sourceOpInterface,
                                         RVVBinaryDataflowStepKind kind,
                                         RVVBinaryDataflowValue lhs,
                                         RVVBinaryDataflowValue rhs,
                                         RVVBinaryDataflowValue result) {
  RVVBinaryDataflowStep step;
  step.kind = kind;
  step.sourceOpName = sourceOpName.str();
  step.sourceOpRole = sourceOpRole.str();
  step.sourceOpInterface = sourceOpInterface.str();
  step.lhs = lhs;
  step.rhs = rhs;
  step.result = result;
  return step;
}

RVVBinaryDataflowStep makeStoreStep(llvm::StringRef sourceOpName,
                                    RuntimeABIParameterRole role,
                                    RVVBinaryDataflowValue value) {
  RVVBinaryDataflowStep step;
  step.kind = RVVBinaryDataflowStepKind::Store;
  step.sourceOpName = sourceOpName.str();
  step.bufferRole = role;
  step.value = value;
  return step;
}

unsigned countRole(llvm::ArrayRef<support::RuntimeABIParameter> parameters,
                   RuntimeABIParameterRole role) {
  return llvm::count_if(parameters, [&](const support::RuntimeABIParameter &p) {
    return p.role == role;
  });
}

llvm::Error validateCallableABIContract(
    const RVVBinaryMicrokernelBodyValidationRequest &request) {
  if (request.callableABIParameters.empty())
    return makeBodyVerifierError(
        request,
        "runtime ABI role layer is missing; body verifier requires "
        "lhs-input-buffer, rhs-input-buffer, output-buffer, and "
        "runtime-element-count roles from the descriptor/exporter plan");

  for (RuntimeABIParameterRole role : {
           RuntimeABIParameterRole::LHSInputBuffer,
           RuntimeABIParameterRole::RHSInputBuffer,
           RuntimeABIParameterRole::OutputBuffer,
           RuntimeABIParameterRole::RuntimeElementCount,
       }) {
    unsigned matches = countRole(request.callableABIParameters, role);
    llvm::StringRef roleName = support::stringifyRuntimeABIParameterRole(role);
    if (matches == 0)
      return makeBodyVerifierError(
          request, llvm::Twine("runtime ABI role layer is missing required '") +
                       roleName + "' binding from the descriptor/exporter plan");
    if (matches > 1)
      return makeBodyVerifierError(
          request,
          llvm::Twine("runtime ABI role layer has duplicate '") + roleName +
              "' bindings in the descriptor/exporter plan");
  }

  if (countRole(request.callableABIParameters,
                RuntimeABIParameterRole::DispatchAvailabilityGuard) != 0)
    return makeBodyVerifierError(
        request,
        "runtime ABI callable plan for a bounded microkernel body must not "
        "contain dispatch-availability-guard; dispatch guards stay outside "
        "the runtime-callable body ABI");

  return llvm::Error::success();
}

llvm::Error ensureRoleInCallablePlan(
    const RVVBinaryMicrokernelBodyValidationRequest &request,
    RuntimeABIParameterRole role, llvm::StringRef context) {
  if (countRole(request.callableABIParameters, role) == 1)
    return llvm::Error::success();
  return makeBodyVerifierError(
      request, llvm::Twine(context) +
                   " references runtime ABI role '" +
                   support::stringifyRuntimeABIParameterRole(role) +
                   "' that is not uniquely bound by the descriptor/exporter "
                   "runtime ABI plan");
}

llvm::Expected<RuntimeABIParameterRole> requireBufferRole(
    const RVVBinaryMicrokernelBodyValidationRequest &request,
    mlir::Operation *op, RuntimeABIParameterRole expectedRole,
    llvm::StringRef context) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(kBufferRoleAttrName);
  if (!attr)
    return makeBodyVerifierError(
        request, llvm::Twine(context) +
                     " is missing body-local buffer_role runtime ABI binding "
                     "metadata");

  llvm::StringRef roleText = attr.getValue().trim();
  if (llvm::Error error =
          validateBoundedText(request, "body buffer_role", roleText))
    return std::move(error);

  std::optional<RuntimeABIParameterRole> parsedRole =
      support::symbolizeRuntimeABIParameterRole(roleText);
  if (!parsedRole)
    return makeBodyVerifierError(
        request, llvm::Twine(context) +
                     " has unsupported body buffer_role '" + roleText + "'");

  if (*parsedRole != expectedRole)
    return makeBodyVerifierError(
        request, llvm::Twine(context) +
                     " must reference runtime ABI role '" +
                     support::stringifyRuntimeABIParameterRole(expectedRole) +
                     "' and agree with the descriptor/exporter runtime ABI "
                     "plan");

  if (llvm::Error error = ensureRoleInCallablePlan(request, *parsedRole, context))
    return std::move(error);

  return *parsedRole;
}

llvm::Expected<RVVIntrinsicConfig> buildIntrinsicConfig(
    const RVVBinaryMicrokernelBodyValidationRequest &request, SetVLOp setvl,
    WithVLOp withVL) {
  if (!request.selectedPolicy)
    return makeBodyVerifierError(
        request,
        "selected compile-time vector policy layer is missing "
        "tcrv_rvv.policy metadata");

  if (setvl.getPolicy() != request.selectedPolicy)
    return makeBodyVerifierError(
        request,
        "tcrv_rvv.setvl policy layer is stale: body policy must match "
        "selected variant tcrv_rvv.policy metadata before artifact export");

  auto withVLSew = withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto withVLLMUL = withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  auto withVLPolicy = withVL->getAttrOfType<PolicyAttr>(kPolicyAttrName);
  if (!withVLSew || !withVLLMUL || !withVLPolicy)
    return makeBodyVerifierError(
        request,
        "tcrv_rvv.with_vl compile-time vector config layer must carry "
        "explicit SEW/LMUL/policy metadata before artifact export");

  if (withVLSew.getInt() != static_cast<std::int64_t>(setvl.getSew()) ||
      withVLLMUL.getValue() != setvl.getLmul() ||
      withVLPolicy != setvl.getPolicy())
    return makeBodyVerifierError(
        request,
        "tcrv_rvv.with_vl compile-time vector config layer is stale: "
        "SEW/LMUL/policy must match the defining tcrv_rvv.setvl metadata");

  if (!request.descriptor.shape)
    return makeBodyVerifierError(
        request,
        "selected descriptor/config layer is missing RVV vector shape metadata");

  llvm::StringRef tail =
      stringifyTailPolicyValue(request.selectedPolicy.getTail());
  llvm::StringRef mask =
      stringifyMaskPolicyValue(request.selectedPolicy.getMask());
  if (tail != request.descriptor.getTailPolicy() ||
      mask != request.descriptor.getMaskPolicy())
    return makeBodyVerifierError(
        request,
        llvm::Twine("selected policy layer is stale: selected variant "
                    "tail/mask policy is tail=") +
            tail + ", mask=" + mask +
            " but the selected descriptor/config requires tail=" +
            request.descriptor.getTailPolicy() +
            ", mask=" + request.descriptor.getMaskPolicy());

  if (request.selectedPolicy.getTail() != TailPolicy::Agnostic ||
      request.selectedPolicy.getMask() != MaskPolicy::Agnostic)
    return makeBodyVerifierError(
        request,
        llvm::Twine("unsupported policy tail=") + tail + ", mask=" + mask +
            "; bounded RVV C intrinsic emission requires tail=agnostic, "
            "mask=agnostic");

  if (setvl.getSew() != static_cast<std::uint64_t>(request.descriptor.getSEWBits()) ||
      setvl.getLmul() != request.descriptor.getLMUL())
    return makeBodyVerifierError(
        request,
        llvm::Twine("tcrv_rvv.setvl compile-time vector config layer is "
                    "stale: body sew=") +
            llvm::Twine(setvl.getSew()) + ", lmul=" + setvl.getLmul() +
            " but selected descriptor/config requires sew=" +
            llvm::Twine(request.descriptor.getSEWBits()) +
            ", lmul=" + request.descriptor.getLMUL());

  RVVIntrinsicConfig config;
  config.sew = request.descriptor.getSEWBits();
  config.lmul = request.descriptor.getLMUL().str();
  config.vectorType = request.descriptor.getVectorType().str();
  config.vectorSuffix = request.descriptor.getVectorSuffix().str();
  config.setvlSuffix = request.descriptor.getSetVLSuffix().str();
  config.setvlIntrinsicName = request.descriptor.getSetVLIntrinsicName();
  config.loadIntrinsicName = request.descriptor.getLoadIntrinsicName();
  config.arithmeticIntrinsicName = request.descriptor.getArithmeticIntrinsicName();
  config.storeIntrinsicName = request.descriptor.getStoreIntrinsicName();
  config.tailPolicy = tail.str();
  config.maskPolicy = mask.str();
  return config;
}

void appendArithmeticStep(RVVBinaryDataflowEmissionPlan &plan,
                          llvm::StringRef sourceOpName,
                          llvm::StringRef sourceOpRole,
                          llvm::StringRef sourceOpInterface,
                          RVVBinaryDataflowStepKind kind) {
  plan.steps.push_back(makeArithmeticStep(
      sourceOpName, sourceOpRole, sourceOpInterface, kind,
      RVVBinaryDataflowValue::LHSVector,
      RVVBinaryDataflowValue::RHSVector,
      RVVBinaryDataflowValue::ResultVector));
}

llvm::Error requireGeneratedEmitCLowerableInterface(
    const RVVBinaryMicrokernelBodyValidationRequest &request,
    mlir::Operation *op, llvm::StringRef expectedSourceOpName,
    llvm::StringRef &sourceOpName, llvm::StringRef &sourceOpRole) {
  auto lowerable = llvm::dyn_cast<TCRVEmitCLowerableOpInterface>(op);
  if (!lowerable)
    return makeBodyVerifierError(
        request,
        llvm::Twine("finite dataflow arithmetic op '") +
            op->getName().getStringRef() +
            "' must implement generated TCRVEmitCLowerableOpInterface "
            "before the RVV EmitC route is constructed");

  sourceOpName = lowerable.getTCRVEmitCLowerableSourceOpName();
  sourceOpRole = lowerable.getTCRVEmitCLowerableSourceRole();
  if (llvm::Error error =
          validateBoundedText(request, "EmitC lowerable source op",
                              sourceOpName))
    return error;
  if (llvm::Error error =
          validateBoundedText(request, "EmitC lowerable source role",
                              sourceOpRole))
    return error;
  if (sourceOpName != expectedSourceOpName)
    return makeBodyVerifierError(
        request,
        llvm::Twine("generated TCRVEmitCLowerableOpInterface source op '") +
            sourceOpName + "' must match selected RVV family operation '" +
            expectedSourceOpName + "' before artifact export");
  if (sourceOpRole != "compute")
    return makeBodyVerifierError(
        request,
        llvm::Twine("generated TCRVEmitCLowerableOpInterface source role '") +
            sourceOpRole +
            "' must be 'compute' for bounded RVV arithmetic ops");
  return llvm::Error::success();
}

llvm::Error validateI32DataflowBody(
    const RVVBinaryMicrokernelBodyValidationRequest &request, WithVLOp withVL,
    mlir::Type expectedVectorType, RVVBinaryDataflowEmissionPlan &plan) {
  mlir::Region &withVLBody = withVL.getBody();
  if (withVLBody.empty() || !llvm::hasSingleElement(withVLBody))
    return makeBodyVerifierError(
        request,
        "tcrv_rvv.with_vl body layer must contain one finite i32 dataflow "
        "block for this bounded RVV export route");
  if (withVLBody.front().getNumArguments() != 0)
    return makeBodyVerifierError(
        request, "tcrv_rvv.with_vl dataflow block must not take block "
                 "arguments; runtime AVL/VL control stays in setvl/with_vl");

  llvm::SmallVector<mlir::Operation *, 4> ops;
  for (mlir::Operation &op : withVLBody.front())
    ops.push_back(&op);
  if (ops.size() != 4)
    return makeBodyVerifierError(
        request,
        llvm::Twine("finite dataflow body for selected family '") +
            request.descriptor.getRVVMicrokernelOpName() +
            "' requires exactly tcrv_rvv.i32_load, tcrv_rvv.i32_load, " +
            request.descriptor.getRVVOperationName() +
            ", tcrv_rvv.i32_store before artifact export");

  auto lhsLoad = llvm::dyn_cast<I32LoadOp>(ops[0]);
  auto rhsLoad = llvm::dyn_cast<I32LoadOp>(ops[1]);
  auto store = llvm::dyn_cast<I32StoreOp>(ops[3]);
  mlir::Value arithmeticLHS;
  mlir::Value arithmeticRHS;
  mlir::Value arithmeticVL;
  mlir::Value arithmeticResult;
  llvm::StringRef arithmeticSourceOpName;
  llvm::StringRef arithmeticSourceOpRole;
  RVVBinaryDataflowStepKind arithmeticStepKind =
      RVVBinaryDataflowStepKind::Mul;
  if (auto add = llvm::dyn_cast<I32AddOp>(ops[2])) {
    if (request.descriptor.family.arithmetic != RVVBinaryArithmeticKind::Add)
      return makeBodyVerifierError(
          request,
          "finite dataflow arithmetic op 'tcrv_rvv.i32_add' does not match "
          "the selected microkernel family before artifact export");
    arithmeticLHS = add.getLhs();
    arithmeticRHS = add.getRhs();
    arithmeticVL = add.getVl();
    arithmeticResult = add.getSum();
    arithmeticStepKind = RVVBinaryDataflowStepKind::Add;
  } else if (auto sub = llvm::dyn_cast<I32SubOp>(ops[2])) {
    if (request.descriptor.family.arithmetic != RVVBinaryArithmeticKind::Sub)
      return makeBodyVerifierError(
          request,
          "finite dataflow arithmetic op 'tcrv_rvv.i32_sub' does not match "
          "the selected microkernel family before artifact export");
    arithmeticLHS = sub.getLhs();
    arithmeticRHS = sub.getRhs();
    arithmeticVL = sub.getVl();
    arithmeticResult = sub.getDifference();
    arithmeticStepKind = RVVBinaryDataflowStepKind::Sub;
  } else if (auto mul = llvm::dyn_cast<I32MulOp>(ops[2])) {
    if (request.descriptor.family.arithmetic != RVVBinaryArithmeticKind::Mul)
      return makeBodyVerifierError(
          request,
          "finite dataflow arithmetic op 'tcrv_rvv.i32_mul' does not match "
          "the selected microkernel family before artifact export");
    arithmeticLHS = mul.getLhs();
    arithmeticRHS = mul.getRhs();
    arithmeticVL = mul.getVl();
    arithmeticResult = mul.getProduct();
    arithmeticStepKind = RVVBinaryDataflowStepKind::Mul;
  }

  if (!lhsLoad || !rhsLoad || !arithmeticResult || !store)
    return makeBodyVerifierError(
        request,
        llvm::Twine("finite dataflow body for selected family '") +
            request.descriptor.getRVVMicrokernelOpName() +
            "' requires exactly tcrv_rvv.i32_load, tcrv_rvv.i32_load, " +
            request.descriptor.getRVVOperationName() +
            ", tcrv_rvv.i32_store before artifact export");

  if (llvm::Error error = requireGeneratedEmitCLowerableInterface(
          request, ops[2], request.descriptor.getRVVOperationName(),
          arithmeticSourceOpName, arithmeticSourceOpRole))
    return error;

  if (lhsLoad.getVl() != withVL.getVl() || rhsLoad.getVl() != withVL.getVl() ||
      arithmeticVL != withVL.getVl() || store.getVl() != withVL.getVl())
    return makeBodyVerifierError(
        request,
        "finite i32 dataflow ops must all consume the !tcrv_rvv.vl token "
        "owned by the surrounding tcrv_rvv.with_vl");

  if (arithmeticLHS != lhsLoad.getLoaded() ||
      arithmeticRHS != rhsLoad.getLoaded() ||
      store.getValue() != arithmeticResult)
    return makeBodyVerifierError(
        request,
        llvm::Twine("finite i32 dataflow SSA chain must be lhs-load,"
                    "rhs-load -> ") +
            request.descriptor.family.arithmeticVerb +
            " -> store before artifact export");

  if (llvm::Error error =
          validateVectorValueType(request, lhsLoad.getLoaded(),
                                  expectedVectorType,
                                  "first tcrv_rvv.i32_load result"))
    return error;
  if (llvm::Error error =
          validateVectorValueType(request, rhsLoad.getLoaded(),
                                  expectedVectorType,
                                  "second tcrv_rvv.i32_load result"))
    return error;
  if (llvm::Error error =
          validateVectorValueType(request, arithmeticLHS, expectedVectorType,
                                  "finite i32 arithmetic lhs"))
    return error;
  if (llvm::Error error =
          validateVectorValueType(request, arithmeticRHS, expectedVectorType,
                                  "finite i32 arithmetic rhs"))
    return error;
  if (llvm::Error error =
          validateVectorValueType(request, arithmeticResult, expectedVectorType,
                                  "finite i32 arithmetic result"))
    return error;
  if (llvm::Error error =
          validateVectorValueType(request, store.getValue(), expectedVectorType,
                                  "tcrv_rvv.i32_store value"))
    return error;

  llvm::Expected<RuntimeABIParameterRole> lhsRole = requireBufferRole(
      request, lhsLoad.getOperation(), RuntimeABIParameterRole::LHSInputBuffer,
      "first tcrv_rvv.i32_load");
  if (!lhsRole)
    return lhsRole.takeError();
  llvm::Expected<RuntimeABIParameterRole> rhsRole = requireBufferRole(
      request, rhsLoad.getOperation(), RuntimeABIParameterRole::RHSInputBuffer,
      "second tcrv_rvv.i32_load");
  if (!rhsRole)
    return rhsRole.takeError();
  llvm::Expected<RuntimeABIParameterRole> storeRole = requireBufferRole(
      request, store.getOperation(), RuntimeABIParameterRole::OutputBuffer,
      "tcrv_rvv.i32_store");
  if (!storeRole)
    return storeRole.takeError();

  plan.steps.clear();
  plan.steps.push_back(
      makeLoadStep(lhsLoad->getName().getStringRef(), *lhsRole,
                   RVVBinaryDataflowValue::LHSVector));
  plan.steps.push_back(
      makeLoadStep(rhsLoad->getName().getStringRef(), *rhsRole,
                   RVVBinaryDataflowValue::RHSVector));
  appendArithmeticStep(plan, arithmeticSourceOpName, arithmeticSourceOpRole,
                       kEmitCLowerableOpInterfaceName, arithmeticStepKind);
  plan.steps.push_back(
      makeStoreStep(store->getName().getStringRef(), *storeRole,
                    RVVBinaryDataflowValue::ResultVector));
  return llvm::Error::success();
}

llvm::Error validateI64DataflowBody(
    const RVVBinaryMicrokernelBodyValidationRequest &request, WithVLOp withVL,
    mlir::Type expectedVectorType, RVVBinaryDataflowEmissionPlan &plan) {
  mlir::Region &withVLBody = withVL.getBody();
  if (withVLBody.empty() || !llvm::hasSingleElement(withVLBody))
    return makeBodyVerifierError(
        request,
        "tcrv_rvv.with_vl body layer must contain one finite i64 dataflow "
        "block for this bounded RVV export route");
  if (withVLBody.front().getNumArguments() != 0)
    return makeBodyVerifierError(
        request, "tcrv_rvv.with_vl dataflow block must not take block "
                 "arguments; runtime AVL/VL control stays in setvl/with_vl");

  llvm::SmallVector<mlir::Operation *, 4> ops;
  for (mlir::Operation &op : withVLBody.front())
    ops.push_back(&op);
  if (ops.size() != 4)
    return makeBodyVerifierError(
        request,
        llvm::Twine("finite dataflow body for selected family '") +
            request.descriptor.getRVVMicrokernelOpName() +
            "' requires exactly tcrv_rvv.i64_load, tcrv_rvv.i64_load, " +
            request.descriptor.getRVVOperationName() +
            ", tcrv_rvv.i64_store before artifact export");

  auto lhsLoad = llvm::dyn_cast<I64LoadOp>(ops[0]);
  auto rhsLoad = llvm::dyn_cast<I64LoadOp>(ops[1]);
  auto store = llvm::dyn_cast<I64StoreOp>(ops[3]);
  mlir::Value arithmeticLHS;
  mlir::Value arithmeticRHS;
  mlir::Value arithmeticVL;
  mlir::Value arithmeticResult;
  llvm::StringRef arithmeticSourceOpName;
  llvm::StringRef arithmeticSourceOpRole = "compute";
  llvm::StringRef arithmeticSourceOpInterface;
  RVVBinaryDataflowStepKind arithmeticStepKind =
      RVVBinaryDataflowStepKind::Mul;
  if (auto add = llvm::dyn_cast<I64AddOp>(ops[2])) {
    if (request.descriptor.family.arithmetic != RVVBinaryArithmeticKind::Add)
      return makeBodyVerifierError(
          request,
          "finite dataflow arithmetic op 'tcrv_rvv.i64_add' does not match "
          "the selected microkernel family before artifact export");
    arithmeticLHS = add.getLhs();
    arithmeticRHS = add.getRhs();
    arithmeticVL = add.getVl();
    arithmeticResult = add.getSum();
    arithmeticStepKind = RVVBinaryDataflowStepKind::Add;
  } else if (auto sub = llvm::dyn_cast<I64SubOp>(ops[2])) {
    if (request.descriptor.family.arithmetic != RVVBinaryArithmeticKind::Sub)
      return makeBodyVerifierError(
          request,
          "finite dataflow arithmetic op 'tcrv_rvv.i64_sub' does not match "
          "the selected microkernel family before artifact export");
    arithmeticLHS = sub.getLhs();
    arithmeticRHS = sub.getRhs();
    arithmeticVL = sub.getVl();
    arithmeticResult = sub.getDifference();
    arithmeticStepKind = RVVBinaryDataflowStepKind::Sub;
  } else if (auto mul = llvm::dyn_cast<I64MulOp>(ops[2])) {
    if (request.descriptor.family.arithmetic != RVVBinaryArithmeticKind::Mul)
      return makeBodyVerifierError(
          request,
          "finite dataflow arithmetic op 'tcrv_rvv.i64_mul' does not match "
          "the selected microkernel family before artifact export");
    arithmeticLHS = mul.getLhs();
    arithmeticRHS = mul.getRhs();
    arithmeticVL = mul.getVl();
    arithmeticResult = mul.getProduct();
    arithmeticStepKind = RVVBinaryDataflowStepKind::Mul;
  }

  if (!lhsLoad || !rhsLoad || !arithmeticResult || !store)
    return makeBodyVerifierError(
        request,
        llvm::Twine("finite dataflow body for selected family '") +
            request.descriptor.getRVVMicrokernelOpName() +
            "' requires exactly tcrv_rvv.i64_load, tcrv_rvv.i64_load, " +
            request.descriptor.getRVVOperationName() +
            ", tcrv_rvv.i64_store before artifact export");

  if (request.descriptor.family.familyID == "i64-vadd") {
    if (llvm::Error error = requireGeneratedEmitCLowerableInterface(
            request, ops[2], request.descriptor.getRVVOperationName(),
            arithmeticSourceOpName, arithmeticSourceOpRole))
      return error;
    arithmeticSourceOpInterface = kEmitCLowerableOpInterfaceName;
  } else {
    arithmeticSourceOpName = ops[2]->getName().getStringRef();
  }

  if (lhsLoad.getVl() != withVL.getVl() || rhsLoad.getVl() != withVL.getVl() ||
      arithmeticVL != withVL.getVl() || store.getVl() != withVL.getVl())
    return makeBodyVerifierError(
        request,
        "finite i64 dataflow ops must all consume the !tcrv_rvv.vl token "
        "owned by the surrounding tcrv_rvv.with_vl");

  if (arithmeticLHS != lhsLoad.getLoaded() ||
      arithmeticRHS != rhsLoad.getLoaded() ||
      store.getValue() != arithmeticResult)
    return makeBodyVerifierError(
        request,
        llvm::Twine("finite i64 dataflow SSA chain must be lhs-load,"
                    "rhs-load -> ") +
            request.descriptor.family.arithmeticVerb +
            " -> store before artifact export");

  if (llvm::Error error =
          validateVectorValueType(request, lhsLoad.getLoaded(),
                                  expectedVectorType,
                                  "first tcrv_rvv.i64_load result"))
    return error;
  if (llvm::Error error =
          validateVectorValueType(request, rhsLoad.getLoaded(),
                                  expectedVectorType,
                                  "second tcrv_rvv.i64_load result"))
    return error;
  if (llvm::Error error =
          validateVectorValueType(request, arithmeticLHS, expectedVectorType,
                                  "finite i64 arithmetic lhs"))
    return error;
  if (llvm::Error error =
          validateVectorValueType(request, arithmeticRHS, expectedVectorType,
                                  "finite i64 arithmetic rhs"))
    return error;
  if (llvm::Error error =
          validateVectorValueType(request, arithmeticResult, expectedVectorType,
                                  "finite i64 arithmetic result"))
    return error;
  if (llvm::Error error =
          validateVectorValueType(request, store.getValue(), expectedVectorType,
                                  "tcrv_rvv.i64_store value"))
    return error;

  llvm::Expected<RuntimeABIParameterRole> lhsRole = requireBufferRole(
      request, lhsLoad.getOperation(), RuntimeABIParameterRole::LHSInputBuffer,
      "first tcrv_rvv.i64_load");
  if (!lhsRole)
    return lhsRole.takeError();
  llvm::Expected<RuntimeABIParameterRole> rhsRole = requireBufferRole(
      request, rhsLoad.getOperation(), RuntimeABIParameterRole::RHSInputBuffer,
      "second tcrv_rvv.i64_load");
  if (!rhsRole)
    return rhsRole.takeError();
  llvm::Expected<RuntimeABIParameterRole> storeRole = requireBufferRole(
      request, store.getOperation(), RuntimeABIParameterRole::OutputBuffer,
      "tcrv_rvv.i64_store");
  if (!storeRole)
    return storeRole.takeError();

  plan.steps.clear();
  plan.steps.push_back(
      makeLoadStep(lhsLoad->getName().getStringRef(), *lhsRole,
                   RVVBinaryDataflowValue::LHSVector));
  plan.steps.push_back(
      makeLoadStep(rhsLoad->getName().getStringRef(), *rhsRole,
                   RVVBinaryDataflowValue::RHSVector));
  appendArithmeticStep(plan, arithmeticSourceOpName, arithmeticSourceOpRole,
                       arithmeticSourceOpInterface, arithmeticStepKind);
  plan.steps.push_back(
      makeStoreStep(store->getName().getStringRef(), *storeRole,
                    RVVBinaryDataflowValue::ResultVector));
  return llvm::Error::success();
}

} // namespace

llvm::Expected<RVVBinaryMicrokernelBodyValidationResult>
validateRVVBinaryMicrokernelBody(
    const RVVBinaryMicrokernelBodyValidationRequest &request) {
  if (!request.kernel)
    return makeBodyVerifierError(request, "missing enclosing tcrv.exec.kernel");
  if (!request.microkernel)
    return makeBodyVerifierError(request, "missing selected RVV microkernel op");
  if (request.microkernel->getName().getStringRef() !=
      request.descriptor.getRVVMicrokernelOpName())
    return makeBodyVerifierError(
        request, llvm::Twine("selected body op '") +
                     request.microkernel->getName().getStringRef() +
                     "' does not match descriptor family '" +
                     request.descriptor.getRVVMicrokernelOpName() + "'");

  if (llvm::Error error = validateCallableABIContract(request))
    return std::move(error);

  auto elementCountAttr =
      request.microkernel->getAttrOfType<mlir::IntegerAttr>(
          kElementCountAttrName);
  if (!elementCountAttr)
    return makeBodyVerifierError(
        request, "descriptor-local element_count layer is missing from the "
                 "structured RVV microkernel body");
  std::int64_t elementCount = elementCountAttr.getInt();
  if (elementCount <= 0 || elementCount > 64)
    return makeBodyVerifierError(
        request,
        "descriptor-local element_count layer must stay in the bounded smoke "
        "range [1, 64]");
  if (request.expectedDescriptorElementCount &&
      elementCount != *request.expectedDescriptorElementCount)
    return makeBodyVerifierError(
        request,
        llvm::Twine("descriptor-local element_count layer is stale: body "
                    "element_count=") +
            llvm::Twine(elementCount) +
            " but selected variant metadata tcrv_rvv.element_count=" +
            llvm::Twine(*request.expectedDescriptorElementCount));

  mlir::Region &body = request.microkernel->getRegion(0);
  if (body.empty() || !llvm::hasSingleElement(body))
    return makeBodyVerifierError(
        request, "structured RVV control-plane body must contain exactly one "
                 "block before artifact export");

  mlir::Block &block = body.front();
  if (block.getNumArguments() != 1 ||
      !block.getArgument(0).getType().isIndex())
    return makeBodyVerifierError(
        request, "runtime AVL/VL control layer must expose exactly one index "
                 "body argument for target/export-owned n/AVL");

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
    return makeBodyVerifierError(
        request,
        llvm::Twine("structured RVV control-plane body has unexpected "
                    "operation '") +
            bodyOp.getName().getStringRef() +
            "'; body verifier consumes only tcrv_rvv.setvl and "
            "tcrv_rvv.with_vl at this layer");
  }

  if (setvlCount != 1 || withVLCount != 1)
    return makeBodyVerifierError(
        request, "runtime AVL/VL control layer requires exactly one "
                 "tcrv_rvv.setvl and exactly one tcrv_rvv.with_vl");
  if (setvl.getAvl() != block.getArgument(0))
    return makeBodyVerifierError(
        request, "runtime AVL/VL control layer is stale: tcrv_rvv.setvl AVL "
                 "must come from the runtime index body argument, not "
                 "descriptor-local element_count, hardware metadata, or a "
                 "constant");
  if (withVL.getVl() != setvl.getVl())
    return makeBodyVerifierError(
        request, "runtime AVL/VL control layer is stale: tcrv_rvv.with_vl "
                 "must consume the !tcrv_rvv.vl token produced by setvl");

  llvm::Expected<RVVIntrinsicConfig> config =
      buildIntrinsicConfig(request, setvl, withVL);
  if (!config)
    return config.takeError();

  mlir::Type expectedVectorType =
      getExpectedVectorType(request.microkernel->getContext(),
                            request.descriptor);
  if (!expectedVectorType)
    return makeBodyVerifierError(
        request,
        llvm::Twine("unsupported selected descriptor/config shape '") +
            request.descriptor.getShapeID() + "' for dtype '" +
            request.descriptor.getDTypeID() +
            "'; finite RVV binary body verifier fails closed");

  RVVBinaryDataflowEmissionPlan plan;
  switch (request.descriptor.family.dtype) {
  case RVVBinaryDTypeKind::I32:
    if (llvm::Error error =
            validateI32DataflowBody(request, withVL, expectedVectorType, plan))
      return std::move(error);
    break;
  case RVVBinaryDTypeKind::I64:
    if (llvm::Error error =
            validateI64DataflowBody(request, withVL, expectedVectorType, plan))
      return std::move(error);
    break;
  }

  RVVBinaryMicrokernelBodyValidationResult result;
  result.dataflowPlan = std::move(plan);
  result.intrinsicConfig = std::move(*config);
  result.elementCount = elementCount;
  result.controlPlaneSEW = result.intrinsicConfig.sew;
  result.controlPlaneLMUL = result.intrinsicConfig.lmul;
  return result;
}

} // namespace tianchenrv::target::rvv
