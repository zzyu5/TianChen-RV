#include "TianChenRV/Plugin/RVV/RVVBinaryMicrokernelMaterialization.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
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
    const target::rvv::RVVVectorShapeConfig &shape) {
  if (shape.dtypeID == "i32") {
    if (shape.lmul == "m2")
      return tcrv::rvv::I32M2VectorType::get(context);
    return tcrv::rvv::I32M1VectorType::get(context);
  }

  if (shape.dtypeID == "i64" && shape.lmul == "m1")
    return tcrv::rvv::I64M1VectorType::get(context);

  return {};
}

llvm::StringRef
getRVVBinaryLoadOpName(const target::rvv::RVVVectorShapeConfig &shape) {
  if (shape.dtypeID == "i32")
    return tcrv::rvv::I32LoadOp::getOperationName();
  if (shape.dtypeID == "i64")
    return tcrv::rvv::I64LoadOp::getOperationName();
  return {};
}

llvm::StringRef
getRVVBinaryStoreOpName(const target::rvv::RVVVectorShapeConfig &shape) {
  if (shape.dtypeID == "i32")
    return tcrv::rvv::I32StoreOp::getOperationName();
  if (shape.dtypeID == "i64")
    return tcrv::rvv::I64StoreOp::getOperationName();
  return {};
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

llvm::Expected<llvm::SmallVector<support::RuntimeABIParameter, 4>>
buildRVVBinaryCallableRuntimeABIParameters(
    tcrv::exec::KernelOp kernel,
    const target::rvv::RVVBinaryIntrinsicDescriptor &descriptor) {
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
  const target::rvv::RVVBinaryIntrinsicDescriptor &descriptor =
      selectedPlan.descriptor;
  const target::rvv::RVVVectorShapeConfig &shape = selectedPlan.getShape();

  mlir::Type vectorType =
      getRVVBinaryVectorType(builder.getContext(), shape);
  llvm::StringRef loadOpName = getRVVBinaryLoadOpName(shape);
  llvm::StringRef storeOpName = getRVVBinaryStoreOpName(shape);
  if (!vectorType || loadOpName.empty() || storeOpName.empty())
    return makeRVVBinaryMicrokernelMaterializationError(
        llvm::Twine("selected RVV binary microkernel descriptor '") +
        selectedPlan.getLoweringDescriptor() +
        "' requires one supported finite vector-shape config");

  auto requiredCapabilities =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);

  mlir::OperationState state(variant.getLoc(),
                             descriptor.getRVVMicrokernelOpName());
  state.addAttribute(kSourceKernelAttrName,
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(kSelectedVariantAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  variant.getSymName()));
  state.addAttribute(kOriginAttrName, builder.getStringAttr(kRVVPluginName));
  state.addAttribute(kRoleAttrName,
                     builder.getStringAttr(stringifyVariantEmissionRole(role)));
  state.addAttribute(kElementCountAttrName,
                     builder.getI64IntegerAttr(selectedPlan.elementCount));
  state.addAttribute(kRequiredCapabilitiesAttrName, requiredCapabilities);
  state.addAttribute(kMicrokernelRequiredMarchAttrName,
                     builder.getStringAttr(selectedPlan.requiredMarch));
  addRVVSelectedVectorShapeMetadataToOperationState(state, builder.getContext(),
                                                    shape);
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
                          builder.getI64IntegerAttr(descriptor.getSEWBits()));
  setvlState.addAttribute(kLMULAttrName,
                          builder.getStringAttr(descriptor.getLMUL()));
  setvlState.addAttribute(kPolicyAttrName, policy);
  mlir::Operation *setvlOp = bodyBuilder.create(setvlState);
  mlir::Value vl = setvlOp->getResult(0);

  mlir::OperationState withVLState(variant.getLoc(),
                                   tcrv::rvv::WithVLOp::getOperationName());
  withVLState.addOperands(vl);
  withVLState.addAttribute(kSEWAttrName,
                           builder.getI64IntegerAttr(descriptor.getSEWBits()));
  withVLState.addAttribute(kLMULAttrName,
                           builder.getStringAttr(descriptor.getLMUL()));
  withVLState.addAttribute(kPolicyAttrName, policy);
  mlir::Region *withVLBody = withVLState.addRegion();
  auto *withVLBlock = new mlir::Block();
  withVLBody->push_back(withVLBlock);

  mlir::OpBuilder withVLBodyBuilder(builder.getContext());
  withVLBodyBuilder.setInsertionPointToStart(withVLBlock);

  mlir::OperationState lhsLoadState(variant.getLoc(), loadOpName);
  lhsLoadState.addOperands(vl);
  lhsLoadState.addTypes(vectorType);
  lhsLoadState.addAttribute(
      kBufferRoleAttrName,
      builder.getStringAttr(support::stringifyRuntimeABIParameterRole(
          support::RuntimeABIParameterRole::LHSInputBuffer)));
  mlir::Operation *lhsLoad = withVLBodyBuilder.create(lhsLoadState);

  mlir::OperationState rhsLoadState(variant.getLoc(), loadOpName);
  rhsLoadState.addOperands(vl);
  rhsLoadState.addTypes(vectorType);
  rhsLoadState.addAttribute(
      kBufferRoleAttrName,
      builder.getStringAttr(support::stringifyRuntimeABIParameterRole(
          support::RuntimeABIParameterRole::RHSInputBuffer)));
  mlir::Operation *rhsLoad = withVLBodyBuilder.create(rhsLoadState);

  mlir::OperationState arithmeticState(
      variant.getLoc(), descriptor.getRVVOperationName());
  arithmeticState.addOperands({lhsLoad->getResult(0), rhsLoad->getResult(0), vl});
  arithmeticState.addTypes(vectorType);
  mlir::Operation *arithmetic = withVLBodyBuilder.create(arithmeticState);

  mlir::OperationState storeState(variant.getLoc(), storeOpName);
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
