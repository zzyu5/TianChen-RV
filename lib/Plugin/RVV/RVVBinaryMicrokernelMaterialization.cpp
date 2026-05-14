#include "TianChenRV/Plugin/RVV/RVVBinaryMicrokernelMaterialization.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Support/RuntimeABICallablePlan.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/Support/Errc.h"

#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kElementCountAttrName("element_count");
constexpr llvm::StringLiteral kMicrokernelRequiredMarchAttrName(
    "required_march");
constexpr llvm::StringLiteral kSelectedMABIAttrName("selected_mabi");
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
constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "TCRVEmitCLowerableOpInterface");
constexpr llvm::StringLiteral kSEWAttrName("sew");
constexpr llvm::StringLiteral kLMULAttrName("lmul");
constexpr llvm::StringLiteral kPolicyAttrName("policy");
constexpr llvm::StringLiteral kBufferRoleAttrName("buffer_role");

llvm::Error makeRVVBinaryMicrokernelMaterializationError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV extension plugin first slice failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Expected<tcrv::rvv::PolicyAttr> getSelectedConfigPolicyAttr(
    mlir::MLIRContext *context,
    const target::rvv::RVVBinarySelectedConfigContract &contract) {
  if (contract.getTailPolicy() != "agnostic" ||
      contract.getMaskPolicy() != "agnostic")
    return makeRVVBinaryMicrokernelMaterializationError(
        llvm::Twine("selected RVV binary config contract for family '") +
        contract.getFamilyID() +
        "' has unsupported finite policy tail='" +
        contract.getTailPolicy() + "', mask='" + contract.getMaskPolicy() +
        "'");
  return tcrv::rvv::PolicyAttr::get(context, tcrv::rvv::TailPolicy::Agnostic,
                                    tcrv::rvv::MaskPolicy::Agnostic);
}

mlir::StringAttr getStringAttr(mlir::Operation *op, llvm::StringRef name) {
  return op ? op->getAttrOfType<mlir::StringAttr>(name) : mlir::StringAttr();
}

mlir::Type getRVVBinaryVectorType(
    mlir::MLIRContext *context,
    const target::rvv::RVVBinarySelectedConfigContract &contract) {
  if (contract.getDTypeID() == "i32") {
    if (contract.getLMUL() == "m2")
      return tcrv::rvv::I32M2VectorType::get(context);
    return tcrv::rvv::I32M1VectorType::get(context);
  }

  if (contract.getDTypeID() == "i64" && contract.getLMUL() == "m1")
    return tcrv::rvv::I64M1VectorType::get(context);

  return {};
}

llvm::StringRef
getRVVBinaryLoadOpName(
    const target::rvv::RVVBinarySelectedConfigContract &contract) {
  if (contract.getDTypeID() == "i32")
    return tcrv::rvv::I32LoadOp::getOperationName();
  if (contract.getDTypeID() == "i64")
    return tcrv::rvv::I64LoadOp::getOperationName();
  return {};
}

llvm::StringRef
getRVVBinaryStoreOpName(
    const target::rvv::RVVBinarySelectedConfigContract &contract) {
  if (contract.getDTypeID() == "i32")
    return tcrv::rvv::I32StoreOp::getOperationName();
  if (contract.getDTypeID() == "i64")
    return tcrv::rvv::I64StoreOp::getOperationName();
  return {};
}

llvm::Error validateRVVBinaryVLDataflowPlanMatchesContract(
    const RVVBinarySelectedPlan &selectedPlan,
    const target::rvv::RVVBinarySelectedConfigContract &contract) {
  if (!selectedPlan.family)
    return makeRVVBinaryMicrokernelMaterializationError(
        "selected RVV binary VL dataflow materialization requires a finite "
        "binary family descriptor");

  auto failMismatch = [&](llvm::StringRef field, llvm::Twine expected,
                          llvm::Twine actual) -> llvm::Error {
    return makeRVVBinaryMicrokernelMaterializationError(
        llvm::Twine("selected RVV binary VL dataflow materialization "
                    "requires selected intrinsic descriptor field '") +
        field + "' to match selected-config contract; expected '" + expected +
        "' but found '" + actual + "'");
  };

  const target::rvv::RVVBinaryIntrinsicRoute &descriptor =
      selectedPlan.descriptor;
  if (selectedPlan.family->familyID != contract.getFamilyID())
    return failMismatch("family", contract.getFamilyID(),
                        selectedPlan.family->familyID);
  if (descriptor.getArithmeticFamilyID() != contract.getFamilyID())
    return failMismatch("descriptor-family", contract.getFamilyID(),
                        descriptor.getArithmeticFamilyID());
  if (descriptor.getLegacyLoweringTokenMirror() !=
      contract.getLegacyLoweringTokenMirror())
    return failMismatch("legacy-lowering-descriptor-mirror",
                        contract.getLegacyLoweringTokenMirror(),
                        descriptor.getLegacyLoweringTokenMirror());
  if (descriptor.getRVVMicrokernelOpName() !=
      contract.getFamily().microkernelOpName)
    return failMismatch("microkernel-op",
                        contract.getFamily().microkernelOpName,
                        descriptor.getRVVMicrokernelOpName());
  if (descriptor.getRVVOperationName() != contract.getArithmeticOpName())
    return failMismatch("arithmetic-op", contract.getArithmeticOpName(),
                        descriptor.getRVVOperationName());
  if (descriptor.getShapeID() != contract.getShapeID())
    return failMismatch("shape", contract.getShapeID(),
                        descriptor.getShapeID());
  if (descriptor.getSEWBits() != contract.getSEWBits())
    return failMismatch("sew", llvm::Twine(contract.getSEWBits()),
                        llvm::Twine(descriptor.getSEWBits()));
  if (descriptor.getLMUL() != contract.getLMUL())
    return failMismatch("lmul", contract.getLMUL(), descriptor.getLMUL());
  if (descriptor.getTailPolicy() != contract.getTailPolicy())
    return failMismatch("tail-policy", contract.getTailPolicy(),
                        descriptor.getTailPolicy());
  if (descriptor.getMaskPolicy() != contract.getMaskPolicy())
    return failMismatch("mask-policy", contract.getMaskPolicy(),
                        descriptor.getMaskPolicy());
  if (descriptor.getVectorType() != contract.getVectorType())
    return failMismatch("vector-type", contract.getVectorType(),
                        descriptor.getVectorType());
  if (descriptor.getVectorSuffix() != contract.getVectorSuffix())
    return failMismatch("vector-suffix", contract.getVectorSuffix(),
                        descriptor.getVectorSuffix());
  if (descriptor.getSetVLSuffix() != contract.getSetVLSuffix())
    return failMismatch("setvl-suffix", contract.getSetVLSuffix(),
                        descriptor.getSetVLSuffix());
  if (selectedPlan.elementCount != contract.getComponentCapacityElementCount())
    return failMismatch("descriptor-element-count",
                        llvm::Twine(contract.getComponentCapacityElementCount()),
                        llvm::Twine(selectedPlan.elementCount));

  return llvm::Error::success();
}

} // namespace

llvm::Expected<RVVBinaryVLDataflowMaterialization>
buildRVVBinaryVLDataflowMaterialization(
    mlir::MLIRContext *context, const RVVBinarySelectedPlan &selectedPlan) {
  if (!context)
    return makeRVVBinaryMicrokernelMaterializationError(
        "selected RVV binary VL dataflow materialization requires an MLIR "
        "context");

  const target::rvv::RVVBinarySelectedConfigContract &contract =
      selectedPlan.getSelectedConfig().getContract();
  if (llvm::Error error =
          target::rvv::validateRVVBinarySelectedConfigContract(contract))
    return std::move(error);
  if (llvm::Error error =
          validateRVVBinaryVLDataflowPlanMatchesContract(selectedPlan,
                                                         contract))
    return std::move(error);

  RVVBinaryVLDataflowMaterialization dataflow;
  dataflow.selectedConfig = &contract;
  dataflow.vectorType = getRVVBinaryVectorType(context, contract);
  dataflow.microkernelOpName = contract.getFamily().microkernelOpName;
  dataflow.loadOpName = getRVVBinaryLoadOpName(contract);
  dataflow.arithmeticOpName = contract.getArithmeticOpName();
  dataflow.storeOpName = getRVVBinaryStoreOpName(contract);
  dataflow.sewBits = contract.getSEWBits();
  dataflow.lmul = contract.getLMUL();
  dataflow.tailPolicy = contract.getTailPolicy();
  dataflow.maskPolicy = contract.getMaskPolicy();
  dataflow.vectorSuffix = contract.getVectorSuffix();
  dataflow.setvlSuffix = contract.getSetVLSuffix();
  dataflow.componentCapacityElementCount = contract.getComponentCapacityElementCount();

  if (!dataflow.vectorType || dataflow.loadOpName.empty() ||
      dataflow.arithmeticOpName.empty() || dataflow.storeOpName.empty())
    return makeRVVBinaryMicrokernelMaterializationError(
        llvm::Twine("selected RVV binary config contract for family '") +
        contract.getFamilyID() +
        "' requires one supported finite VL dataflow shape");

  return dataflow;
}

llvm::Expected<llvm::SmallVector<support::RuntimeABIParameter, 4>>
buildRVVBinaryCallableRuntimeABIParameters(
    tcrv::exec::KernelOp kernel,
    const target::rvv::RVVBinaryIntrinsicRoute &descriptor) {
  llvm::Expected<support::FiniteBinaryCallableABIPlan> callablePlan =
      support::buildFiniteBinaryCallableABIPlan(
          kernel,
          target::rvv::getRVVBinaryRuntimeABIContract(descriptor.family));
  if (!callablePlan)
    return callablePlan.takeError();
  return std::move(callablePlan->parameters);
}

const target::rvv::RVVBinaryFamilyRecord *
getRVVBinaryMicrokernelFamilyForOp(mlir::Operation *op) {
  if (!op)
    return nullptr;

  llvm::StringRef opName = op->getName().getStringRef();
  for (const target::rvv::RVVBinaryFamilyRecord *family :
       target::rvv::getRVVBinaryFamilyRegistrationRecords()) {
    if (family->microkernelOpName == opName)
      return family;
  }
  return nullptr;
}

llvm::Error rejectExistingRVVBinaryMicrokernelForSelectedPath(
    tcrv::exec::KernelOp kernel, tcrv::exec::VariantOp variant,
    VariantEmissionRole role) {
  if (!kernel || kernel.getBody().empty())
    return llvm::Error::success();

  llvm::StringRef expectedRole = stringifyVariantEmissionRole(role);
  for (mlir::Operation &op : kernel.getBody().front()) {
    const target::rvv::RVVBinaryFamilyRecord *family =
        getRVVBinaryMicrokernelFamilyForOp(&op);
    if (!family)
      continue;

    auto target =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    auto microkernelRole = getStringAttr(&op, kRoleAttrName);
    llvm::StringRef targetSymbol =
        target ? target.getValue() : llvm::StringRef("<missing>");
    llvm::StringRef roleValue = microkernelRole
                                    ? microkernelRole.getValue()
                                    : llvm::StringRef("<missing>");
    if (targetSymbol != variant.getSymName() || roleValue != expectedRole)
      continue;

    return makeRVVBinaryMicrokernelMaterializationError(
        llvm::Twine("requires no pre-existing ") + family->microkernelOpName +
        " for target @" + targetSymbol + " as " + expectedRole);
  }

  return llvm::Error::success();
}

llvm::Expected<mlir::Operation *> materializeRVVBinaryMicrokernelOp(
    mlir::OpBuilder &builder, tcrv::exec::KernelOp kernel,
    tcrv::exec::VariantOp variant, VariantEmissionRole role,
    const RVVBinaryMicrokernelMaterializationPlan &plan) {
  const RVVBinarySelectedPlan &selectedPlan = plan.selectedPlan;
  llvm::Expected<RVVBinaryVLDataflowMaterialization> dataflow =
      buildRVVBinaryVLDataflowMaterialization(builder.getContext(),
                                             selectedPlan);
  if (!dataflow)
    return dataflow.takeError();

  auto requiredCapabilities =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);

  mlir::OperationState state(variant.getLoc(), dataflow->microkernelOpName);
  state.addAttribute(kSourceKernelAttrName,
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(kSelectedVariantAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  variant.getSymName()));
  state.addAttribute(kOriginAttrName, builder.getStringAttr(kRVVPluginName));
  state.addAttribute(kRoleAttrName,
                     builder.getStringAttr(stringifyVariantEmissionRole(role)));
  state.addAttribute(kElementCountAttrName,
                     builder.getI64IntegerAttr(
                         dataflow->componentCapacityElementCount));
  state.addAttribute(kRequiredCapabilitiesAttrName, requiredCapabilities);
  state.addAttribute(kMicrokernelRequiredMarchAttrName,
                     builder.getStringAttr(selectedPlan.requiredMarch));
  addRVVSelectedVectorShapeMetadataToOperationState(state, builder.getContext(),
                                                    dataflow->selectedConfig
                                                        ->getShape());
  if (selectedPlan.selectedMABI)
    state.addAttribute(kSelectedMABIAttrName,
                       builder.getStringAttr(*selectedPlan.selectedMABI));
  llvm::StringRef sourceKind = plan.sourceKind.empty()
                                   ? getRVVDefaultTypedBinarySourceKind()
                                   : llvm::StringRef(plan.sourceKind);
  state.addAttribute(kSelectedBinarySourceKindAttrName,
                     builder.getStringAttr(sourceKind));
  state.addAttribute(kSelectedBinaryDTypeAttrName,
                     builder.getStringAttr(selectedPlan.getDTypeID()));
  state.addAttribute(kSelectedBinaryFamilyAttrName,
                     builder.getStringAttr(selectedPlan.getFamilyID()));
  state.addAttribute(kSelectedBinaryOperatorAttrName,
                     builder.getStringAttr(
                         selectedPlan.family->arithmeticVerb));
  state.addAttribute(
      kSelectedBinaryMicrokernelOpAttrName,
      builder.getStringAttr(selectedPlan.getMicrokernelOpName()));
  state.addAttribute(kEmitCSourceOpAttrName,
                     builder.getStringAttr(selectedPlan.getArithmeticOpName()));
  state.addAttribute(kEmitCLowerableOpInterfaceAttrName,
                     builder.getStringAttr(kEmitCLowerableOpInterfaceName));

  mlir::Region *body = state.addRegion();
  auto *block = new mlir::Block();
  body->push_back(block);
  mlir::Value runtimeN =
      block->addArgument(builder.getIndexType(), variant.getLoc());

  mlir::OpBuilder bodyBuilder(builder.getContext());
  bodyBuilder.setInsertionPointToStart(block);

  llvm::Expected<tcrv::rvv::PolicyAttr> policy =
      getSelectedConfigPolicyAttr(builder.getContext(),
                                  *dataflow->selectedConfig);
  if (!policy)
    return policy.takeError();

  mlir::OperationState setvlState(variant.getLoc(),
                                  tcrv::rvv::SetVLOp::getOperationName());
  setvlState.addOperands(runtimeN);
  setvlState.addTypes(tcrv::rvv::VLType::get(builder.getContext()));
  setvlState.addAttribute(kSEWAttrName,
                          builder.getI64IntegerAttr(dataflow->sewBits));
  setvlState.addAttribute(kLMULAttrName,
                          builder.getStringAttr(dataflow->lmul));
  setvlState.addAttribute(kPolicyAttrName, *policy);
  mlir::Operation *setvlOp = bodyBuilder.create(setvlState);
  mlir::Value vl = setvlOp->getResult(0);

  mlir::OperationState withVLState(variant.getLoc(),
                                   tcrv::rvv::WithVLOp::getOperationName());
  withVLState.addOperands(vl);
  withVLState.addAttribute(kSEWAttrName,
                           builder.getI64IntegerAttr(dataflow->sewBits));
  withVLState.addAttribute(kLMULAttrName,
                           builder.getStringAttr(dataflow->lmul));
  withVLState.addAttribute(kPolicyAttrName, *policy);
  mlir::Region *withVLBody = withVLState.addRegion();
  auto *withVLBlock = new mlir::Block();
  withVLBody->push_back(withVLBlock);

  mlir::OpBuilder withVLBodyBuilder(builder.getContext());
  withVLBodyBuilder.setInsertionPointToStart(withVLBlock);

  mlir::OperationState lhsLoadState(variant.getLoc(), dataflow->loadOpName);
  lhsLoadState.addOperands(vl);
  lhsLoadState.addTypes(dataflow->vectorType);
  lhsLoadState.addAttribute(
      kBufferRoleAttrName,
      builder.getStringAttr(support::stringifyRuntimeABIParameterRole(
          support::RuntimeABIParameterRole::LHSInputBuffer)));
  mlir::Operation *lhsLoad = withVLBodyBuilder.create(lhsLoadState);

  mlir::OperationState rhsLoadState(variant.getLoc(), dataflow->loadOpName);
  rhsLoadState.addOperands(vl);
  rhsLoadState.addTypes(dataflow->vectorType);
  rhsLoadState.addAttribute(
      kBufferRoleAttrName,
      builder.getStringAttr(support::stringifyRuntimeABIParameterRole(
          support::RuntimeABIParameterRole::RHSInputBuffer)));
  mlir::Operation *rhsLoad = withVLBodyBuilder.create(rhsLoadState);

  mlir::OperationState arithmeticState(variant.getLoc(),
                                       dataflow->arithmeticOpName);
  arithmeticState.addOperands({lhsLoad->getResult(0), rhsLoad->getResult(0), vl});
  arithmeticState.addTypes(dataflow->vectorType);
  mlir::Operation *arithmetic = withVLBodyBuilder.create(arithmeticState);

  mlir::OperationState storeState(variant.getLoc(), dataflow->storeOpName);
  storeState.addOperands({arithmetic->getResult(0), vl});
  storeState.addAttribute(
      kBufferRoleAttrName,
      builder.getStringAttr(support::stringifyRuntimeABIParameterRole(
          support::RuntimeABIParameterRole::OutputBuffer)));
  withVLBodyBuilder.create(storeState);

  bodyBuilder.create(withVLState);
  return builder.create(state);
}

} // namespace tianchenrv::plugin::rvv
