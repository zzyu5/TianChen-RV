#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/ExtensionBundle.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"
#include "TianChenRV/Plugin/RVV/RVVVectorSourceFrontDoor.h"
#include "TianChenRV/Target/RVV/RVVTargetSupportBundle.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace tianchenrv::plugin {
namespace {

namespace construction = tianchenrv::plugin::construction;

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kRVVPluginVersion("0.1.0");
constexpr llvm::StringLiteral kRVVCapabilityID("rvv");
constexpr llvm::StringLiteral kRVVCapabilityKind("isa-vector");
constexpr llvm::StringLiteral kRVVPreferredCapabilitySymbol("rvv");
constexpr llvm::StringLiteral kRVVPolicyAttrName("tcrv_rvv.policy");
constexpr llvm::StringLiteral kOriginAttrName("origin");

llvm::Error makeRVVPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV extension plugin first slice failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool hasAvailableRVVCapability(const VariantProposalRequest &request) {
  return request.getKernel() &&
         request.getCapabilities().isCapabilityAvailableByID(kRVVCapabilityID);
}

bool variantContainsExplicitTypedRVVBody(tcrv::exec::VariantOp variant) {
  if (!variant || variant.getBody().empty())
    return false;

  bool found = false;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (found || op == variant.getOperation())
      return;
    if (op->getName().getDialectNamespace() == "tcrv_rvv")
      found = true;
  });
  return found;
}

bool variantContainsPreRealizedRVVSelectedBody(tcrv::exec::VariantOp variant) {
  if (!variant || variant.getBody().empty())
    return false;

  bool found = false;
  variant.getBody().walk([&](tcrv::rvv::TypedBinaryPreRealizedBodyOp) {
    found = true;
  });
  return found;
}

bool isSupportedPreRealizedArithmeticOpKind(llvm::StringRef opKind) {
  return opKind == "add" || opKind == "sub" || opKind == "mul";
}

llvm::Error requireExplicitTypedRVVBody(tcrv::exec::VariantOp variant) {
  if (variantContainsExplicitTypedRVVBody(variant))
    return llvm::Error::success();
  return makeRVVPluginError(
      "materialized RVV variant requires explicit typed RVV "
      "extension-family body");
}

llvm::Error requireRVVSelectedVariant(tcrv::exec::VariantOp variant) {
  if (!variant)
    return makeRVVPluginError(
        "selected RVV lowering boundary requires a materialized "
        "tcrv.exec.variant");

  auto originAttr = variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue() != kRVVPluginName)
    return makeRVVPluginError(
        "materialized RVV variant must be owned by origin 'rvv-plugin'");

  return requireExplicitTypedRVVBody(variant);
}

llvm::Expected<tcrv::rvv::WithVLOp>
findSelectedRVVSelectedBodyBoundary(tcrv::exec::VariantOp variant) {
  if (!variant)
    return makeRVVPluginError(
        "selected RVV lowering boundary requires a materialized "
        "tcrv.exec.variant");

  llvm::SmallVector<tcrv::rvv::SetVLOp, 2> setvls;
  llvm::SmallVector<tcrv::rvv::WithVLOp, 2> withVLs;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (auto setvl = llvm::dyn_cast<tcrv::rvv::SetVLOp>(op))
      setvls.push_back(setvl);
    if (auto withVL = llvm::dyn_cast<tcrv::rvv::WithVLOp>(op))
      withVLs.push_back(withVL);
  });

  if (setvls.size() != 1)
    return makeRVVPluginError(
        "selected RVV typed lowering boundary requires exactly one "
        "tcrv_rvv.setvl op");
  if (withVLs.size() != 1)
    return makeRVVPluginError(
        "selected RVV typed lowering boundary requires exactly one "
        "tcrv_rvv.with_vl op");

  tcrv::rvv::RVVConfigContractDiagnostic configDiagnostic =
      tcrv::rvv::validateRVVSelectedBodyConfigVLStructure(setvls.front(),
                                                          withVLs.front());
  if (!configDiagnostic.ok)
    return makeRVVPluginError(configDiagnostic.message);

  return withVLs.front();
}

mlir::FlatSymbolRefAttr symbolRef(mlir::OpBuilder &builder,
                                  llvm::StringRef symbol) {
  return mlir::FlatSymbolRefAttr::get(builder.getContext(), symbol);
}

llvm::Expected<tcrv::rvv::RuntimeABIValueOp>
requirePreRealizedRuntimeABIValue(
    mlir::Value value, llvm::StringRef context,
    support::RuntimeABIParameterRole expectedRole) {
  auto binding = value.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (!binding)
    return makeRVVPluginError(llvm::Twine(context) +
                              " must be defined by explicit "
                              "tcrv_rvv.runtime_abi_value");

  std::optional<support::RuntimeABIParameterRole> role =
      support::symbolizeRuntimeABIParameterRole(binding.getRole());
  if (!role)
    return makeRVVPluginError(llvm::Twine(context) +
                              " carries unsupported runtime ABI role '" +
                              binding.getRole() + "'");
  if (*role != expectedRole)
    return makeRVVPluginError(
        llvm::Twine(context) + " must bind runtime ABI role '" +
        support::stringifyRuntimeABIParameterRole(expectedRole) +
        "' before RVV selected-body realization");
  return binding;
}

llvm::Expected<tcrv::rvv::TypedBinaryPreRealizedBodyOp>
findUniquePreRealizedRVVSelectedBody(tcrv::exec::VariantOp variant) {
  if (!variant)
    return makeRVVPluginError(
        "selected RVV realization requires a materialized tcrv.exec.variant");

  llvm::SmallVector<tcrv::rvv::TypedBinaryPreRealizedBodyOp, 2> bodies;
  variant.getBody().walk([&](tcrv::rvv::TypedBinaryPreRealizedBodyOp body) {
    bodies.push_back(body);
  });

  if (bodies.size() != 1)
    return makeRVVPluginError(
        "selected RVV realization requires exactly one "
        "tcrv_rvv.typed_binary_pre_realized_body op when no realized "
        "setvl/with_vl body is present");
  return bodies.front();
}

llvm::Error validatePreRealizedRVVSelectedBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedBinaryPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV realization requires a pre-realized body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected body must be a direct child of the "
        "selected tcrv.exec.variant");

  if (!isSupportedPreRealizedArithmeticOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected body currently supports only op_kind "
        "'add', 'sub', or 'mul'");
  if (body.getMemoryForm() != "vector-rhs-load")
    return makeRVVPluginError(
        "pre-realized RVV selected body currently supports only memory_form "
        "'vector-rhs-load'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected body requires SEW32 LMUL m1");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected body requires tail agnostic, mask agnostic "
        "policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(), "pre-realized RVV lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedRuntimeABIValue(
          body.getRhs(), "pre-realized RVV rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(), "pre-realized RVV runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  unsigned realizedSetVLCount = 0;
  unsigned realizedWithVLCount = 0;
  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp ||
        op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedBinaryPreRealizedBodyOp>(op))
      return;
    if (llvm::isa<tcrv::rvv::SetVLOp>(op)) {
      ++realizedSetVLCount;
      unexpectedRVVOp = op;
      return;
    }
    if (llvm::isa<tcrv::rvv::WithVLOp>(op)) {
      ++realizedWithVLCount;
      unexpectedRVVOp = op;
      return;
    }
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected body must not be mixed with "
                    "already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");
  (void)realizedSetVLCount;
  (void)realizedWithVLCount;

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected-body realization requires non-empty "
        "selected variant requires metadata");

  return llvm::Error::success();
}

mlir::Operation *createRealizedSetVL(mlir::OpBuilder &builder,
                                     mlir::Location loc, mlir::Value nValue) {
  mlir::OperationState state(loc, "tcrv_rvv.setvl");
  state.addOperands(nValue);
  state.addTypes(tcrv::rvv::VLType::get(builder.getContext()));
  tcrv::rvv::populateRVVSelectedBodyDefaultConfigAttrs(builder, state);
  return builder.create(state);
}

tcrv::rvv::WithVLOp createRealizedWithVL(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value vlValue,
    tcrv::exec::KernelOp kernel, tcrv::exec::VariantOp variant,
    VariantEmissionRole role, mlir::ArrayAttr requires) {
  mlir::OperationState state(loc, "tcrv_rvv.with_vl");
  state.addOperands(vlValue);
  tcrv::rvv::populateRVVSelectedBodyDefaultConfigAttrs(builder, state);
  state.addAttribute(rvv::getRVVSourceKernelAttrName(),
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(rvv::getRVVSelectedVariantAttrName(),
                     symbolRef(builder, variant.getSymName()));
  state.addAttribute(rvv::getRVVOriginAttrName(),
                     builder.getStringAttr(kRVVPluginName));
  state.addAttribute(rvv::getRVVSelectedPathRoleAttrName(),
                     builder.getStringAttr(stringifyVariantEmissionRole(role)));
  state.addAttribute(rvv::getRVVStatusAttrName(),
                     builder.getStringAttr(rvv::getRVVLoweringBoundaryStatus()));
  state.addAttribute(rvv::getRVVRequiredCapabilitiesAttrName(), requires);
  state.addAttribute(rvv::getRVVConstructionProtocolMetadataName(),
                     builder.getStringAttr(
                         rvv::getRVVConstructionProtocolVersion()));
  state.addRegion();
  auto withVL = llvm::cast<tcrv::rvv::WithVLOp>(builder.create(state));
  withVL.getBody().emplaceBlock();
  return withVL;
}

mlir::Type getStage1GenericVectorType(mlir::OpBuilder &builder) {
  return tcrv::rvv::VectorType::get(
      builder.getContext(), builder.getI32Type(),
      tcrv::rvv::getRVVLMULM1());
}

mlir::Operation *createRealizedGenericLoad(mlir::OpBuilder &builder,
                                           mlir::Location loc,
                                           mlir::Value buffer,
                                           mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.load");
  state.addOperands({buffer, vl});
  state.addTypes(getStage1GenericVectorType(builder));
  return builder.create(state);
}

llvm::Expected<mlir::Operation *>
createRealizedGenericBinaryCompute(mlir::OpBuilder &builder,
                                   mlir::Location loc,
                                   llvm::StringRef opKind, mlir::Value lhs,
                                   mlir::Value rhs, mlir::Value vl) {
  if (!isSupportedPreRealizedArithmeticOpKind(opKind))
    return makeRVVPluginError(
        "pre-realized RVV selected-body realization supports only op_kind "
        "'add', 'sub', or 'mul'");

  mlir::OperationState state(loc, "tcrv_rvv.binary");
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr(opKind));
  state.addTypes(lhs.getType());
  return builder.create(state);
}

void createRealizedGenericStore(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value out, mlir::Value value,
                                mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.store");
  state.addOperands({out, value, vl});
  (void)builder.create(state);
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVSelectedBody(
    const VariantLoweringBoundaryRequest &request) {
  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV selected-body realization requires materialized "
        "kernel and variant");

  llvm::Expected<tcrv::rvv::TypedBinaryPreRealizedBodyOp> body =
      findUniquePreRealizedRVVSelectedBody(variant);
  if (!body)
    return body.takeError();
  if (llvm::Error error = validatePreRealizedRVVSelectedBody(request, *body))
    return std::move(error);

  auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  mlir::OpBuilder &builder = request.getBuilder();
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = body->getLoc();
  builder.setInsertionPoint(body->getOperation());

  auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
      createRealizedSetVL(builder, loc, body->getN()));
  tcrv::rvv::WithVLOp withVL =
      createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                           request.getRole(), requires);

  builder.setInsertionPointToStart(&withVL.getBody().front());
  auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(
      createRealizedGenericLoad(builder, loc, body->getLhs(), setvl.getVl()));
  auto rhsLoad = llvm::cast<tcrv::rvv::LoadOp>(
      createRealizedGenericLoad(builder, loc, body->getRhs(), setvl.getVl()));
  llvm::Expected<mlir::Operation *> compute = createRealizedGenericBinaryCompute(
      builder, loc, body->getOpKind(), lhsLoad.getLoaded(),
      rhsLoad.getLoaded(), setvl.getVl());
  if (!compute)
    return compute.takeError();
  createRealizedGenericStore(builder, loc, body->getOut(),
                             (*compute)->getResult(0), setvl.getVl());
  body->erase();
  return withVL;
}

llvm::Error verifySelectedRVVLoweringBoundaryConformance(
    tcrv::exec::KernelOp kernel, tcrv::exec::VariantOp variant,
    VariantEmissionRole role, tcrv::rvv::WithVLOp boundary) {
  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "selected RVV lowering-boundary validation requires non-empty "
        "selected variant requires metadata");

  const construction::SelectedBoundaryStringAttrExpectation extraAttributes[] =
      {{rvv::getRVVConstructionProtocolMetadataName(),
        rvv::getRVVConstructionProtocolVersion()}};
  construction::SelectedLoweringBoundaryConformanceSpec spec;
  spec.boundaryDescription = "selected RVV lowering-boundary validation";
  spec.selectedVariantSymbol = variant.getSymName();
  spec.sourceKernelSymbol = kernel.getSymName();
  spec.originPlugin = kRVVPluginName;
  spec.pathRole = stringifyVariantEmissionRole(role);
  spec.status = rvv::getRVVLoweringBoundaryStatus();
  spec.requiredCapabilities = variantRequires;
  spec.extraStringAttributes = extraAttributes;
  spec.sourceKernelAttrName = rvv::getRVVSourceKernelAttrName();
  spec.selectedVariantAttrName = rvv::getRVVSelectedVariantAttrName();
  spec.originAttrName = rvv::getRVVOriginAttrName();
  spec.roleAttrName = rvv::getRVVSelectedPathRoleAttrName();
  spec.statusAttrName = rvv::getRVVStatusAttrName();
  spec.requiredCapabilitiesAttrName = rvv::getRVVRequiredCapabilitiesAttrName();
  return construction::verifySelectedLoweringBoundaryConformance(
      boundary.getOperation(), spec);
}

llvm::Error validateSelectedRVVSelectedBodyBoundary(
    const VariantLoweringBoundaryValidationRequest &request) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeRVVPluginError(
        "selected RVV lowering-boundary validation requires a materialized "
        "tcrv.exec.variant");
  if (!request.getKernel())
    return makeRVVPluginError(
        "selected RVV lowering-boundary validation requires an enclosing "
        "tcrv.exec.kernel");

  if (llvm::Error error = requireRVVSelectedVariant(variant))
    return error;

  auto boundary =
      llvm::dyn_cast_if_present<tcrv::rvv::WithVLOp>(request.getBoundary());
  if (!boundary)
    return makeRVVPluginError(
        "selected RVV typed lowering boundary must be the existing "
        "tcrv_rvv.with_vl operation");

  llvm::Expected<tcrv::rvv::WithVLOp> expectedBoundary =
      findSelectedRVVSelectedBodyBoundary(variant);
  if (!expectedBoundary)
    return expectedBoundary.takeError();
  if (expectedBoundary->getOperation() != boundary.getOperation())
    return makeRVVPluginError(
        "selected RVV typed lowering boundary must be the unique "
        "tcrv_rvv.with_vl operation in the selected variant body");
  if (llvm::Error error = verifySelectedRVVLoweringBoundaryConformance(
          request.getKernel(), variant, request.getRole(), boundary))
    return error;

  conversion::emitc::TCRVEmitCLowerableRoute route;
  VariantEmitCLowerableRequest routeRequest(variant, request.getKernel(),
                                            request.getCapabilities(),
                                            request.getRole());
  return rvv::buildRVVSelectedBodyEmitCLowerableRoute(routeRequest, route);
}

const rvv::RVVExtensionPlugin &getBuiltinRVVExtensionPlugin() {
  static const rvv::RVVExtensionPlugin plugin;
  return plugin;
}

} // namespace

namespace rvv {

llvm::StringRef getRVVExtensionPluginName() { return kRVVPluginName; }

llvm::StringRef getRVVExtensionPluginVersion() { return kRVVPluginVersion; }

llvm::StringRef getRVVCapabilityID() { return kRVVCapabilityID; }

llvm::StringRef getRVVCapabilityKind() { return kRVVCapabilityKind; }

llvm::StringRef getRVVPreferredCapabilitySymbol() {
  return kRVVPreferredCapabilitySymbol;
}

llvm::StringRef getRVVPolicyAttrName() { return kRVVPolicyAttrName; }

RVVExtensionPlugin::RVVExtensionPlugin() {
  capabilities.push_back(PluginCapability(
      kRVVCapabilityID, kRVVCapabilityKind,
      "RVV first-slice vector ISA capability participation; target "
      "availability is supplied by tcrv.exec.capability metadata"));
}

llvm::StringRef RVVExtensionPlugin::getName() const { return kRVVPluginName; }

llvm::StringRef RVVExtensionPlugin::getVersion() const {
  return kRVVPluginVersion;
}

llvm::ArrayRef<PluginCapability> RVVExtensionPlugin::getCapabilities() const {
  return capabilities;
}

void RVVExtensionPlugin::registerDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<tcrv::rvv::TCRVRVVDialect>();
}

llvm::Error
RVVExtensionPlugin::verifyExecutableConstructionConformance() const {
  return rvv::verifyRVVConstructionProtocolReady();
}

llvm::Error RVVExtensionPlugin::registerSourceFrontDoorPasses(
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) const {
  out.push_back(SourceFrontDoorPassRegistration(
      kRVVPluginName, "tcrv-rvv-fail-closed-legacy-vector-source-front-door",
      "Fail closed for the legacy RVV source-front-door materializer "
      "during Stage 1; use explicit generic typed tcrv_rvv bodies instead",
      [] { return createFailClosedRVVLegacyVectorSourceFrontDoorPass(); },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          ExplicitOnly));
  return llvm::Error::success();
}

bool RVVExtensionPlugin::supportsOperation(
    const VariantProposalRequest &request) const {
  return hasAvailableRVVCapability(request);
}

llvm::Error RVVExtensionPlugin::proposeVariants(
    const VariantProposalRequest &request,
    llvm::SmallVectorImpl<VariantProposal> &out) const {
  (void)request;
  (void)out;
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::collectVariantProposals(
    const VariantProposalRequest &request,
    VariantProposalCollectionResult &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  out.addRecoverableDecline(
      kRVVPluginName,
      "RVV proposal requires explicit typed tcrv_rvv extension-family IR "
      "before selecting an RVV variant");
  return llvm::Error::success();
}

llvm::Expected<support::TargetCapabilitySet>
RVVExtensionPlugin::buildTargetCapabilitiesFromProbeFacts(
    const RVVProbeCapabilityFacts &facts) const {
  return buildRVVTargetCapabilitiesFromProbeFacts(facts);
}

llvm::Error RVVExtensionPlugin::verifyVariantLegality(
    const VariantLegalityRequest &request) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeRVVPluginError(
        "legality verification requires a materialized tcrv.exec.variant");

  auto originAttr = variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue() != kRVVPluginName)
    return makeRVVPluginError(
        "materialized RVV variant must be owned by origin 'rvv-plugin'");

  return requireExplicitTypedRVVBody(variant);
}

llvm::Error
RVVExtensionPlugin::estimateVariantCost(const VariantCostRequest &request,
                                        VariantCostEstimate &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "cost estimation requires a materialized tcrv.exec.variant");
  if (llvm::Error error = requireExplicitTypedRVVBody(request.getVariant()))
    return error;

  out = VariantCostEstimate();
  out.setScore(1.0);
  out.setExplicitPreference(true);
  out.setOriginPlugin(kRVVPluginName);
  out.setVariantSymbol(request.getVariant().getSymName());
  out.setExplanation("explicit typed RVV variant body; no runtime performance "
                     "claim");
  out.setPolicy("plugin-local typed RVV extension-family IR");
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::checkVariantEmissionReadiness(
    const VariantEmissionRequest &request, VariantEmissionStatus &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "emission readiness requires a materialized tcrv.exec.variant");
  if (!request.getKernel())
    return makeRVVPluginError(
        "emission readiness requires an enclosing tcrv.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string diagnostic = llvm::toString(std::move(error));
    out = VariantEmissionStatus::getUnsupported(
        kRVVPluginName, request.getVariant().getSymName(), diagnostic);
    return llvm::Error::success();
  }

  llvm::Expected<tcrv::rvv::WithVLOp> selectedBoundary =
      findSelectedRVVSelectedBodyBoundary(request.getVariant());
  if (!selectedBoundary) {
    std::string diagnostic = llvm::toString(selectedBoundary.takeError());
    out = VariantEmissionStatus::getUnsupported(
        kRVVPluginName, request.getVariant().getSymName(), diagnostic);
    return llvm::Error::success();
  }
  VariantLoweringBoundaryValidationRequest boundaryRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole(), selectedBoundary->getOperation());
  if (llvm::Error error =
          validateSelectedRVVSelectedBodyBoundary(boundaryRequest)) {
    std::string diagnostic = llvm::toString(std::move(error));
    out = VariantEmissionStatus::getUnsupported(
        kRVVPluginName, request.getVariant().getSymName(), diagnostic);
    return llvm::Error::success();
  }

  VariantEmitCLowerableRequest routeRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole());
  llvm::Expected<RVVSelectedBodyEmitCRouteDescription> routeDescription =
      describeRVVSelectedBodyEmitCRoute(routeRequest);
  if (!routeDescription) {
    std::string diagnostic = llvm::toString(routeDescription.takeError());
    out = VariantEmissionStatus::getUnsupported(
        kRVVPluginName, request.getVariant().getSymName(), diagnostic);
    return llvm::Error::success();
  }

  out = VariantEmissionStatus::getSupported(
      kRVVPluginName, request.getVariant().getSymName(),
      routeDescription->targetArtifactRouteID);
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::buildVariantEmissionPlan(
    const VariantEmissionRequest &request, VariantEmissionPlan &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "emission planning requires a materialized tcrv.exec.variant");

  if (!request.getKernel())
    return makeRVVPluginError(
        "emission planning requires an enclosing tcrv.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality))
    return error;

  llvm::Expected<tcrv::rvv::WithVLOp> selectedBoundary =
      findSelectedRVVSelectedBodyBoundary(request.getVariant());
  if (!selectedBoundary)
    return selectedBoundary.takeError();
  VariantLoweringBoundaryValidationRequest boundaryRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole(), selectedBoundary->getOperation());
  if (llvm::Error error =
          validateSelectedRVVSelectedBodyBoundary(boundaryRequest))
    return error;

  conversion::emitc::TCRVEmitCLowerableRoute route;
  VariantEmitCLowerableRequest routeRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole());
  llvm::Expected<RVVSelectedBodyEmitCRouteDescription> routeDescription =
      describeRVVSelectedBodyEmitCRoute(routeRequest, &route);
  if (!routeDescription)
    return routeDescription.takeError();

  llvm::StringRef runtimeABIName = routeDescription->runtimeABIName;
  out = VariantEmissionPlan::getSupported(
      kRVVPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      getRVVSelectedBodyEmissionKind(), routeDescription->targetArtifactRouteID,
      runtimeABIName, routeDescription->targetArtifactKind,
      "RVV selected typed body route materializes a verified EmitC "
      "module through the common TCRVEmitCLowerableRoute materializer, then "
      "uses the MLIR EmitC C/C++ emitter before RISC-V object packaging");
  out.setRuntimeABIKind(getRVVSelectedBodyRuntimeABIKind());
  out.setRuntimeABIName(runtimeABIName);
  out.setRuntimeGlueRole(getRVVSelectedBodyRuntimeGlueRole());
  out.setLoweringBoundaryOpName(getRVVSelectedBodyLoweringBoundaryOpName());
  out.addRuntimeABIParameters(routeDescription->runtimeABIParameters);
  RVVSelectedBodyConstructionMetadataFacts constructionFacts =
      getRVVSelectedBodyConstructionMetadataFacts(*routeDescription);
  llvm::Expected<llvm::SmallVector<support::ArtifactMetadataEntry, 16>>
      constructionMetadata = getRVVSelectedBodyConstructionArtifactMetadata(
          constructionFacts);
  if (!constructionMetadata)
    return constructionMetadata.takeError();
  for (const support::ArtifactMetadataEntry &entry : *constructionMetadata)
    out.addArtifactMetadata(entry.key, entry.value);
  for (const support::ArtifactMetadataEntry &entry :
       getRVVSelectedBodyConfigArtifactMetadata(*routeDescription))
    out.addArtifactMetadata(entry.key, entry.value);
  if (llvm::Error error =
          out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
    return error;
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::materializeSelectedLoweringBoundary(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "lowering-boundary materialization requires a materialized "
        "tcrv.exec.variant");
  if (!request.getKernel())
    return makeRVVPluginError(
        "lowering-boundary materialization requires an enclosing "
        "tcrv.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality))
    return error;

  llvm::Expected<tcrv::rvv::WithVLOp> boundary =
      findSelectedRVVSelectedBodyBoundary(request.getVariant());
  if (!boundary) {
    llvm::Error boundaryError = boundary.takeError();
    if (!variantContainsPreRealizedRVVSelectedBody(request.getVariant()))
      return boundaryError;
    llvm::consumeError(std::move(boundaryError));
    boundary = realizePreRealizedRVVSelectedBody(request);
    if (!boundary)
      return boundary.takeError();
  } else if (variantContainsPreRealizedRVVSelectedBody(request.getVariant())) {
    return makeRVVPluginError(
        "pre-realized RVV selected body must not be mixed with an already "
        "realized setvl/with_vl body before route construction");
  }

  VariantLoweringBoundaryValidationRequest validationRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole(), boundary->getOperation());
  if (llvm::Error error =
          validateSelectedRVVSelectedBodyBoundary(validationRequest))
    return error;

  out = VariantLoweringBoundaryResult::getMaterialized(
      kRVVPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      boundary->getOperation());
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::validateSelectedLoweringBoundary(
    const VariantLoweringBoundaryValidationRequest &request) const {
  return validateSelectedRVVSelectedBodyBoundary(request);
}

llvm::Error RVVExtensionPlugin::buildVariantEmitCLowerableRoute(
    const VariantEmitCLowerableRequest &request,
    conversion::emitc::TCRVEmitCLowerableRoute &out) const {
  return buildRVVSelectedBodyEmitCLowerableRoute(request, out);
}

llvm::Error RVVExtensionPlugin::configureTargetSupportExtensionBundle(
    ExtensionBundle &bundle) const {
  bundle.addRequiredDialectName("tcrv_rvv");
  return target::rvv::configureRVVTargetSupportExtensionBundle(bundle);
}

llvm::Error RVVExtensionPlugin::registerTargetSupportTranslateRoutes(
    target::TargetTranslateRouteRegistry &registry) const {
  return target::rvv::registerRVVTargetSupportTargetTranslateRoutes(registry);
}

} // namespace rvv

llvm::Error registerRVVExtensionPlugin(ExtensionPluginRegistry &registry) {
  return registry.registerPlugin(getBuiltinRVVExtensionPlugin());
}

} // namespace tianchenrv::plugin
