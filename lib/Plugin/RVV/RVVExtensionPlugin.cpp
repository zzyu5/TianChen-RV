#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Target/RVV/RVVTargetSupportBundle.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string>

namespace tianchenrv::plugin {
namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kRVVPluginVersion("0.1.0");
constexpr llvm::StringLiteral kRVVCapabilityID("rvv");
constexpr llvm::StringLiteral kRVVCapabilityKind("isa-vector");
constexpr llvm::StringLiteral kRVVPreferredCapabilitySymbol("rvv");
constexpr llvm::StringLiteral kRVVPolicyAttrName("tcrv_rvv.policy");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "TCRVEmitCLowerableOpInterface");

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

llvm::Error requireExplicitTypedRVVBody(tcrv::exec::VariantOp variant) {
  if (variantContainsExplicitTypedRVVBody(variant))
    return llvm::Error::success();
  return makeRVVPluginError(
      "materialized RVV variant requires explicit typed RVV "
      "extension-family body");
}

struct RVVI32M1AddSlice {
  tcrv::rvv::SetVLOp setvl;
  tcrv::rvv::WithVLOp withVL;
  tcrv::rvv::I32LoadOp lhsLoad;
  tcrv::rvv::I32LoadOp rhsLoad;
  tcrv::rvv::I32AddOp add;
  tcrv::rvv::I32StoreOp store;
};

llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance>
getEmitCSourceProvenance(mlir::Operation *op, llvm::StringRef expectedRole) {
  auto lowerable =
      llvm::dyn_cast<conversion::emitc::TCRVEmitCLowerableOpInterface>(op);
  if (!lowerable)
    return makeRVVPluginError(llvm::Twine("operation '") +
                              op->getName().getStringRef() +
                              "' must implement " +
                              kEmitCLowerableOpInterfaceName +
                              " before RVV EmitC route construction");

  llvm::StringRef sourceRole =
      lowerable.getTCRVEmitCLowerableSourceRole();
  if (sourceRole != expectedRole)
    return makeRVVPluginError(llvm::Twine("operation '") +
                              op->getName().getStringRef() +
                              "' reports EmitC source role '" + sourceRole +
                              "' but RVV route expected '" + expectedRole +
                              "'");

  conversion::emitc::TCRVEmitCSourceOpProvenance source;
  source.opName = lowerable.getTCRVEmitCLowerableSourceOpName().str();
  source.role = sourceRole.str();
  source.opInterface = kEmitCLowerableOpInterfaceName.str();
  return source;
}

llvm::Error requireAgnosticPolicy(tcrv::rvv::PolicyAttr policy,
                                  llvm::StringRef context) {
  if (!policy)
    return makeRVVPluginError(llvm::Twine(context) +
                              " requires finite RVV policy metadata");
  if (policy.getTail() != tcrv::rvv::TailPolicy::Agnostic ||
      policy.getMask() != tcrv::rvv::MaskPolicy::Agnostic)
    return makeRVVPluginError(
        llvm::Twine(context) +
        " supports only tail agnostic, mask agnostic policy for the bounded "
        "i32m1 EmitC route");
  return llvm::Error::success();
}

llvm::Expected<RVVI32M1AddSlice>
collectRVVI32M1AddSlice(tcrv::exec::VariantOp variant) {
  llvm::SmallVector<tcrv::rvv::SetVLOp, 2> setvls;
  llvm::SmallVector<tcrv::rvv::WithVLOp, 2> withVLs;
  unsigned rvvOpCount = 0;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    ++rvvOpCount;
    if (auto setvl = llvm::dyn_cast<tcrv::rvv::SetVLOp>(op))
      setvls.push_back(setvl);
    if (auto withVL = llvm::dyn_cast<tcrv::rvv::WithVLOp>(op))
      withVLs.push_back(withVL);
  });

  if (setvls.size() != 1)
    return makeRVVPluginError(
        "bounded RVV EmitC route requires exactly one tcrv_rvv.setvl op");
  if (withVLs.size() != 1)
    return makeRVVPluginError(
        "bounded RVV EmitC route requires exactly one tcrv_rvv.with_vl op");

  RVVI32M1AddSlice slice;
  slice.setvl = setvls.front();
  slice.withVL = withVLs.front();

  if (slice.setvl.getSew() != 32 || slice.setvl.getLmul() != "m1")
    return makeRVVPluginError(
        "bounded RVV EmitC route supports only SEW32 LMUL m1 i32 add");
  if (llvm::Error error =
          requireAgnosticPolicy(slice.setvl.getPolicy(), "tcrv_rvv.setvl"))
    return std::move(error);

  auto withVLSEW =
      slice.withVL->getAttrOfType<mlir::IntegerAttr>("sew");
  auto withVLLMUL =
      slice.withVL->getAttrOfType<mlir::StringAttr>("lmul");
  auto withVLPolicy =
      slice.withVL->getAttrOfType<tcrv::rvv::PolicyAttr>("policy");
  if (!withVLSEW || withVLSEW.getInt() != 32 || !withVLLMUL ||
      withVLLMUL.getValue() != "m1")
    return makeRVVPluginError(
        "bounded RVV EmitC route requires tcrv_rvv.with_vl SEW32 LMUL m1 "
        "metadata");
  if (llvm::Error error =
          requireAgnosticPolicy(withVLPolicy, "tcrv_rvv.with_vl"))
    return std::move(error);
  if (slice.withVL.getVl() != slice.setvl.getVl())
    return makeRVVPluginError(
        "bounded RVV EmitC route requires tcrv_rvv.with_vl to consume the "
        "visible tcrv_rvv.setvl result");

  llvm::SmallVector<tcrv::rvv::I32LoadOp, 2> loads;
  unsigned addCount = 0;
  unsigned storeCount = 0;
  for (mlir::Operation &op : slice.withVL.getBody().front()) {
    if (auto load = llvm::dyn_cast<tcrv::rvv::I32LoadOp>(op)) {
      loads.push_back(load);
      continue;
    }
    if (auto add = llvm::dyn_cast<tcrv::rvv::I32AddOp>(op)) {
      slice.add = add;
      ++addCount;
      continue;
    }
    if (auto store = llvm::dyn_cast<tcrv::rvv::I32StoreOp>(op)) {
      slice.store = store;
      ++storeCount;
      continue;
    }
    return makeRVVPluginError(
        llvm::Twine("bounded RVV EmitC route does not support op '") +
        op.getName().getStringRef() +
        "' inside tcrv_rvv.with_vl; expected load-add-store only");
  }

  if (loads.size() != 2)
    return makeRVVPluginError(
        "bounded RVV EmitC route requires exactly two tcrv_rvv.i32_load ops");
  if (addCount != 1)
    return makeRVVPluginError(
        "bounded RVV EmitC route requires exactly one tcrv_rvv.i32_add op");
  if (storeCount != 1)
    return makeRVVPluginError(
        "bounded RVV EmitC route requires exactly one tcrv_rvv.i32_store op");
  if (rvvOpCount != 6)
    return makeRVVPluginError(
        "bounded RVV EmitC route supports only setvl/with_vl/i32_load/"
        "i32_load/i32_add/i32_store");

  for (tcrv::rvv::I32LoadOp load : loads) {
    llvm::StringRef role = load.getBufferRole();
    if (role == tianchenrv::support::stringifyRuntimeABIParameterRole(
                    tianchenrv::support::RuntimeABIParameterRole::
                        LHSInputBuffer)) {
      if (slice.lhsLoad)
        return makeRVVPluginError(
            "bounded RVV EmitC route requires a unique lhs-input-buffer "
            "load");
      slice.lhsLoad = load;
      continue;
    }
    if (role == tianchenrv::support::stringifyRuntimeABIParameterRole(
                    tianchenrv::support::RuntimeABIParameterRole::
                        RHSInputBuffer)) {
      if (slice.rhsLoad)
        return makeRVVPluginError(
            "bounded RVV EmitC route requires a unique rhs-input-buffer "
            "load");
      slice.rhsLoad = load;
      continue;
    }
    return makeRVVPluginError(llvm::Twine("unsupported RVV i32 load role '") +
                              role + "' for bounded EmitC route");
  }

  if (!slice.lhsLoad || !slice.rhsLoad)
    return makeRVVPluginError(
        "bounded RVV EmitC route requires lhs-input-buffer and "
        "rhs-input-buffer loads");
  if (slice.store.getBufferRole() !=
      tianchenrv::support::stringifyRuntimeABIParameterRole(
          tianchenrv::support::RuntimeABIParameterRole::OutputBuffer))
    return makeRVVPluginError(
        "bounded RVV EmitC route requires output-buffer store role");
  if (slice.add.getLhs() != slice.lhsLoad.getLoaded() ||
      slice.add.getRhs() != slice.rhsLoad.getLoaded())
    return makeRVVPluginError(
        "bounded RVV EmitC route requires tcrv_rvv.i32_add to consume lhs "
        "and rhs load results");
  if (slice.store.getValue() != slice.add.getSum())
    return makeRVVPluginError(
        "bounded RVV EmitC route requires tcrv_rvv.i32_store to consume the "
        "add result");

  return slice;
}

llvm::Error addCallStepFromSource(
    conversion::emitc::TCRVEmitCLowerableRoute &route, mlir::Operation *op,
    llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> source =
      getEmitCSourceProvenance(op, expectedRole);
  if (!source)
    return source.takeError();

  conversion::emitc::TCRVEmitCCallOpaqueStep step;
  step.sourceOp = std::move(*source);
  step.callee = callee.str();
  step.operands.append(operands.begin(), operands.end());
  step.result = std::move(result);
  route.addCallOpaqueStep(std::move(step));
  return llvm::Error::success();
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

llvm::StringRef RVVExtensionPlugin::getName() const {
  return kRVVPluginName;
}

llvm::StringRef RVVExtensionPlugin::getVersion() const {
  return kRVVPluginVersion;
}

llvm::ArrayRef<PluginCapability>
RVVExtensionPlugin::getCapabilities() const {
  return capabilities;
}

void RVVExtensionPlugin::registerDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<tcrv::rvv::TCRVRVVDialect>();
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

llvm::Error RVVExtensionPlugin::estimateVariantCost(
    const VariantCostRequest &request, VariantCostEstimate &out) const {
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

  out = VariantEmissionStatus::getUnsupported(
      kRVVPluginName, request.getVariant().getSymName(),
      "RVV selected path has a bounded explicit-op EmitC materialization path "
      "but no runtime ABI, C/C++ emitter handoff, or target artifact route");
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

  (void)out;
  return makeRVVPluginError(
      "RVV emission planning requires runtime ABI, C/C++ emitter handoff, and "
      "target artifact route after the bounded explicit-op EmitC "
      "materialization path; no artifact metadata is available");
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

  out = VariantLoweringBoundaryResult::getNoBoundary(
      kRVVPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      "RVV explicit typed IR uses the bounded EmitC lowerable route directly; "
      "no selected lowering-boundary op is materialized");
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::validateSelectedLoweringBoundary(
    const VariantLoweringBoundaryValidationRequest &request) const {
  (void)request;
  return makeRVVPluginError(
      "RVV selected lowering-boundary validation requires a materialized "
      "plugin lowering boundary for explicit typed RVV IR");
}

llvm::Error RVVExtensionPlugin::buildVariantEmitCLowerableRoute(
    const VariantEmitCLowerableRequest &request,
    conversion::emitc::TCRVEmitCLowerableRoute &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "EmitC route construction requires a materialized tcrv.exec.variant");
  if (!request.getKernel())
    return makeRVVPluginError(
        "EmitC route construction requires an enclosing tcrv.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality))
    return error;

  llvm::Expected<RVVI32M1AddSlice> slice =
      collectRVVI32M1AddSlice(request.getVariant());
  if (!slice)
    return slice.takeError();

  conversion::emitc::TCRVEmitCLowerableRoute route(
      "rvv-i32m1-add-emitc-route",
      "extension-family-ops-to-emitc-call-opaque");
  route.addHeader("stddef.h");
  route.addHeader("stdint.h");
  route.addHeader("riscv_vector.h");
  route.addTypeMapping("!tcrv_rvv.vl", "size_t");
  route.addTypeMapping("!tcrv_rvv.i32m1", "vint32m1_t");
  route.addABIValueMapping(
      support::makeTargetExportABIParameter(
          "lhs", "const int32_t *",
          support::RuntimeABIParameterRole::LHSInputBuffer),
      "lhs");
  route.addABIValueMapping(
      support::makeTargetExportABIParameter(
          "rhs", "const int32_t *",
          support::RuntimeABIParameterRole::RHSInputBuffer),
      "rhs");
  route.addABIValueMapping(
      support::makeTargetExportABIParameter(
          "out", "int32_t *",
          support::RuntimeABIParameterRole::OutputBuffer),
      "out");
  route.addABIValueMapping(
      support::makeTargetExportABIParameter(
          "n", "size_t",
          support::RuntimeABIParameterRole::RuntimeElementCount),
      "n");

  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance>
      withVLSource =
          getEmitCSourceProvenance(slice->withVL.getOperation(), "scope");
  if (!withVLSource)
    return withVLSource.takeError();
  route.addSourceOpProvenance(std::move(*withVLSource));

  using conversion::emitc::TCRVEmitCCallOpaqueOperand;
  using conversion::emitc::TCRVEmitCCallOpaqueResult;
  if (llvm::Error error = addCallStepFromSource(
          route, slice->setvl.getOperation(), "configure",
          "__riscv_vsetvl_e32m1",
          {TCRVEmitCCallOpaqueOperand{"n", "size_t"}},
          TCRVEmitCCallOpaqueResult{"vl", "size_t"}))
    return error;
  if (llvm::Error error = addCallStepFromSource(
          route, slice->lhsLoad.getOperation(), "load",
          "__riscv_vle32_v_i32m1",
          {TCRVEmitCCallOpaqueOperand{"lhs", "const int32_t *"},
           TCRVEmitCCallOpaqueOperand{"vl", "size_t"}},
          TCRVEmitCCallOpaqueResult{"lhs_vec", "vint32m1_t"}))
    return error;
  if (llvm::Error error = addCallStepFromSource(
          route, slice->rhsLoad.getOperation(), "load",
          "__riscv_vle32_v_i32m1",
          {TCRVEmitCCallOpaqueOperand{"rhs", "const int32_t *"},
           TCRVEmitCCallOpaqueOperand{"vl", "size_t"}},
          TCRVEmitCCallOpaqueResult{"rhs_vec", "vint32m1_t"}))
    return error;
  if (llvm::Error error = addCallStepFromSource(
          route, slice->add.getOperation(), "compute",
          "__riscv_vadd_vv_i32m1",
          {TCRVEmitCCallOpaqueOperand{"lhs_vec", "vint32m1_t"},
           TCRVEmitCCallOpaqueOperand{"rhs_vec", "vint32m1_t"},
           TCRVEmitCCallOpaqueOperand{"vl", "size_t"}},
          TCRVEmitCCallOpaqueResult{"sum_vec", "vint32m1_t"}))
    return error;
  if (llvm::Error error = addCallStepFromSource(
          route, slice->store.getOperation(), "store",
          "__riscv_vse32_v_i32m1",
          {TCRVEmitCCallOpaqueOperand{"out", "int32_t *"},
           TCRVEmitCCallOpaqueOperand{"sum_vec", "vint32m1_t"},
           TCRVEmitCCallOpaqueOperand{"vl", "size_t"}}))
    return error;

  out = std::move(route);
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::configureTargetSupportExtensionBundle(
    target::ExtensionBundle &bundle) const {
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
