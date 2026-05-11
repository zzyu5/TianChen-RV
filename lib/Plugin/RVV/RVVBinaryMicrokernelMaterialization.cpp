#include "TianChenRV/Plugin/RVV/RVVBinaryMicrokernelMaterialization.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Support/RuntimeABICallablePlan.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"

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

tcrv::rvv::PolicyAttr getExpectedRVVPolicyAttr(mlir::MLIRContext *context) {
  return tcrv::rvv::PolicyAttr::get(context, tcrv::rvv::TailPolicy::Agnostic,
                                    tcrv::rvv::MaskPolicy::Agnostic);
}

mlir::StringAttr getStringAttr(mlir::Operation *op, llvm::StringRef name) {
  return op ? op->getAttrOfType<mlir::StringAttr>(name) : mlir::StringAttr();
}

llvm::StringRef getStringAttrValue(mlir::Operation *op,
                                   llvm::StringRef name) {
  mlir::StringAttr attr = getStringAttr(op, name);
  if (!attr)
    return {};
  return attr.getValue();
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

  const target::rvv::RVVBinaryIntrinsicDescriptor &descriptor =
      selectedPlan.descriptor;
  if (selectedPlan.family->familyID != contract.getFamilyID())
    return failMismatch("family", contract.getFamilyID(),
                        selectedPlan.family->familyID);
  if (descriptor.getArithmeticFamilyID() != contract.getFamilyID())
    return failMismatch("descriptor-family", contract.getFamilyID(),
                        descriptor.getArithmeticFamilyID());
  if (descriptor.getLoweringDescriptor() != contract.getLoweringDescriptor())
    return failMismatch("lowering-descriptor",
                        contract.getLoweringDescriptor(),
                        descriptor.getLoweringDescriptor());
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
  if (selectedPlan.elementCount != contract.getDescriptorElementCount())
    return failMismatch("descriptor-element-count",
                        llvm::Twine(contract.getDescriptorElementCount()),
                        llvm::Twine(selectedPlan.elementCount));

  return llvm::Error::success();
}

llvm::Error appendRuntimeABIWindowParameter(
    llvm::SmallVectorImpl<support::RuntimeABIParameter> &parameters,
    tcrv::exec::MemWindowOp window, support::RuntimeABIParameterRole role,
    llvm::StringRef cName) {
  llvm::StringRef cType =
      getStringAttrValue(window.getOperation(), support::kMemWindowCTypeAttrName);
  llvm::StringRef ownership = getStringAttrValue(
      window.getOperation(), support::kMemWindowOwnershipAttrName);
  std::optional<support::RuntimeABIParameterOwnership> parsedOwnership =
      support::symbolizeRuntimeABIParameterOwnership(ownership);
  if (!parsedOwnership)
    return makeRVVBinaryMicrokernelMaterializationError(
        llvm::Twine("RVV binary callable ABI mem_window @") +
        window.getSymName() + " has unsupported ownership '" + ownership + "'");

  parameters.push_back(
      support::RuntimeABIParameter(cName, cType, role, *parsedOwnership));
  return llvm::Error::success();
}

} // namespace

llvm::Expected<std::optional<RVVBinaryMicrokernelMaterializationPlan>>
buildRVVBinaryMicrokernelMaterializationPlanFromVariant(
    tcrv::exec::VariantOp variant,
    const target::rvv::RVVVectorShapeConfig &shape,
    llvm::StringRef expectedDTypeID,
    std::optional<std::string> selectedMABI) {
  llvm::Expected<std::optional<RVVBinarySelectedPlan>> selectedPlan =
      buildRVVBinarySelectedPlanFromVariant(
          variant, shape, expectedDTypeID, std::move(selectedMABI));
  if (!selectedPlan)
    return selectedPlan.takeError();
  if (!*selectedPlan)
    return std::optional<RVVBinaryMicrokernelMaterializationPlan>();

  RVVBinaryMicrokernelMaterializationPlan plan;
  plan.selectedPlan = std::move(**selectedPlan);
  return plan;
}

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
  dataflow.descriptorElementCount = contract.getDescriptorElementCount();

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
    const target::rvv::RVVBinaryIntrinsicDescriptor &descriptor) {
  if (descriptor.family.dtype == target::rvv::RVVBinaryDTypeKind::I32) {
    const target::i32_binary::I32BinaryFamilyDescriptor *family =
        target::i32_binary::lookupI32BinaryFamilyByID(
            descriptor.getArithmeticFamilyID());
    if (!family)
      return makeRVVBinaryMicrokernelMaterializationError(
          llvm::Twine("RVV i32 binary callable ABI requires shared i32 "
                      "binary family descriptor for '") +
          descriptor.getArithmeticFamilyID() + "'");

    llvm::Expected<support::I32BinaryCallableABIPlan> callablePlan =
        support::buildI32BinaryCallableABIPlan(kernel, *family);
    if (!callablePlan)
      return callablePlan.takeError();
    return std::move(callablePlan->parameters);
  }

  llvm::SmallVector<tcrv::exec::MemWindowOp, 3> windows;
  if (llvm::Error error = support::collectRuntimeABIBufferMemWindows(
          kernel, descriptor.getBufferMemWindowSpecs(), windows))
    return std::move(error);

  llvm::SmallVector<support::RuntimeABIParamSpec, 1> countSpecs =
      descriptor.getRuntimeElementCountParamSpecs(/*cName=*/"");
  llvm::SmallVector<tcrv::exec::RuntimeParamOp, 1> runtimeParams;
  if (llvm::Error error =
          support::collectRuntimeABIParams(kernel, countSpecs, runtimeParams))
    return std::move(error);

  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters;
  if (llvm::Error error = appendRuntimeABIWindowParameter(
          parameters, windows[0],
          support::RuntimeABIParameterRole::LHSInputBuffer, "lhs"))
    return std::move(error);
  if (llvm::Error error = appendRuntimeABIWindowParameter(
          parameters, windows[1],
          support::RuntimeABIParameterRole::RHSInputBuffer, "rhs"))
    return std::move(error);
  if (llvm::Error error = appendRuntimeABIWindowParameter(
          parameters, windows[2], support::RuntimeABIParameterRole::OutputBuffer,
          "out"))
    return std::move(error);

  tcrv::exec::RuntimeParamOp runtimeCount = runtimeParams.front();
  std::optional<support::RuntimeABIParameterOwnership> parsedOwnership =
      support::symbolizeRuntimeABIParameterOwnership(getStringAttrValue(
          runtimeCount.getOperation(), support::kRuntimeParamOwnershipAttrName));
  if (!parsedOwnership)
    return makeRVVBinaryMicrokernelMaterializationError(
        llvm::Twine("RVV binary callable ABI runtime_param @") +
        runtimeCount.getSymName() + " has unsupported ownership");

  parameters.push_back(support::RuntimeABIParameter(
      getStringAttrValue(runtimeCount.getOperation(),
                         support::kRuntimeParamCNameAttrName),
      getStringAttrValue(runtimeCount.getOperation(),
                         support::kRuntimeParamCTypeAttrName),
      support::RuntimeABIParameterRole::RuntimeElementCount, *parsedOwnership));

  return parameters;
}

const target::rvv::RVVBinaryFamilyDescriptor *
getRVVBinaryMicrokernelFamilyForOp(mlir::Operation *op) {
  if (!op)
    return nullptr;

  llvm::StringRef opName = op->getName().getStringRef();
  for (const target::rvv::RVVBinaryFamilyDescriptor *family :
       target::rvv::getRVVBinaryFamilyDescriptors()) {
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
    const target::rvv::RVVBinaryFamilyDescriptor *family =
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
                         dataflow->descriptorElementCount));
  state.addAttribute(kRequiredCapabilitiesAttrName, requiredCapabilities);
  state.addAttribute(kMicrokernelRequiredMarchAttrName,
                     builder.getStringAttr(selectedPlan.requiredMarch));
  addRVVSelectedVectorShapeMetadataToOperationState(state, builder.getContext(),
                                                    dataflow->selectedConfig
                                                        ->getShape());
  if (selectedPlan.selectedMABI)
    state.addAttribute(kSelectedMABIAttrName,
                       builder.getStringAttr(*selectedPlan.selectedMABI));

  mlir::Region *body = state.addRegion();
  auto *block = new mlir::Block();
  body->push_back(block);
  mlir::Value runtimeN =
      block->addArgument(builder.getIndexType(), variant.getLoc());

  mlir::OpBuilder bodyBuilder(builder.getContext());
  bodyBuilder.setInsertionPointToStart(block);

  tcrv::rvv::PolicyAttr policy =
      getExpectedRVVPolicyAttr(builder.getContext());

  mlir::OperationState setvlState(variant.getLoc(),
                                  tcrv::rvv::SetVLOp::getOperationName());
  setvlState.addOperands(runtimeN);
  setvlState.addTypes(tcrv::rvv::VLType::get(builder.getContext()));
  setvlState.addAttribute(kSEWAttrName,
                          builder.getI64IntegerAttr(dataflow->sewBits));
  setvlState.addAttribute(kLMULAttrName,
                          builder.getStringAttr(dataflow->lmul));
  setvlState.addAttribute(kPolicyAttrName, policy);
  mlir::Operation *setvlOp = bodyBuilder.create(setvlState);
  mlir::Value vl = setvlOp->getResult(0);

  mlir::OperationState withVLState(variant.getLoc(),
                                   tcrv::rvv::WithVLOp::getOperationName());
  withVLState.addOperands(vl);
  withVLState.addAttribute(kSEWAttrName,
                           builder.getI64IntegerAttr(dataflow->sewBits));
  withVLState.addAttribute(kLMULAttrName,
                           builder.getStringAttr(dataflow->lmul));
  withVLState.addAttribute(kPolicyAttrName, policy);
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
