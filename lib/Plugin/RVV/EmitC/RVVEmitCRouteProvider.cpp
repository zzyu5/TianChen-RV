#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"

#include <optional>
#include <string>
#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kRVVI32M1AddEmitCRouteID(
    "rvv-i32m1-add-emitc-route");
constexpr llvm::StringLiteral kRVVI32M1AddEmissionKind(
    "materialized-emitc-cpp-rvv-intrinsic-object");
constexpr llvm::StringLiteral kRVVI32M1AddLoweringBoundaryOpName(
    "tcrv_rvv.with_vl");
constexpr llvm::StringLiteral kRVVI32M1AddRuntimeABIKind(
    "plugin-owned-runtime-abi");
constexpr llvm::StringLiteral kRVVI32M1AddRuntimeABIName(
    "rvv-i32m1-add-callable-c-abi.v1");
constexpr llvm::StringLiteral kRVVI32M1AddRuntimeGlueRole(
    "emitc-cpp-rvv-intrinsic-runtime-glue");
constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "TCRVEmitCLowerableOpInterface");

llvm::Error makeRVVEmitCRouteProviderError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV plugin-owned EmitC route provider failed: ") +
          message,
      llvm::errc::invalid_argument);
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

llvm::Error requireRVVVariantLegality(tcrv::exec::VariantOp variant) {
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "requires a materialized tcrv.exec.variant");

  auto originAttr = variant->getAttrOfType<mlir::StringAttr>("origin");
  if (!originAttr || originAttr.getValue() != kRVVPluginName)
    return makeRVVEmitCRouteProviderError(
        "materialized RVV variant must be owned by origin 'rvv-plugin'");

  if (variantContainsExplicitTypedRVVBody(variant))
    return llvm::Error::success();

  return makeRVVEmitCRouteProviderError(
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

llvm::Error requireAgnosticPolicy(tcrv::rvv::PolicyAttr policy,
                                  llvm::StringRef context) {
  if (!policy)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " requires finite RVV policy metadata");
  if (policy.getTail() != tcrv::rvv::TailPolicy::Agnostic ||
      policy.getMask() != tcrv::rvv::MaskPolicy::Agnostic)
    return makeRVVEmitCRouteProviderError(
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
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one tcrv_rvv.setvl op");
  if (withVLs.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one tcrv_rvv.with_vl op");

  RVVI32M1AddSlice slice;
  slice.setvl = setvls.front();
  slice.withVL = withVLs.front();

  if (slice.setvl.getSew() != 32 || slice.setvl.getLmul() != "m1")
    return makeRVVEmitCRouteProviderError(
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
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires tcrv_rvv.with_vl SEW32 LMUL m1 "
        "metadata");
  if (llvm::Error error =
          requireAgnosticPolicy(withVLPolicy, "tcrv_rvv.with_vl"))
    return std::move(error);
  if (slice.withVL.getVl() != slice.setvl.getVl())
    return makeRVVEmitCRouteProviderError(
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
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("bounded RVV EmitC route does not support op '") +
        op.getName().getStringRef() +
        "' inside tcrv_rvv.with_vl; expected load-add-store only");
  }

  if (loads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly two tcrv_rvv.i32_load ops");
  if (addCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one tcrv_rvv.i32_add op");
  if (storeCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one tcrv_rvv.i32_store op");
  if (rvvOpCount != 6)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route supports only setvl/with_vl/i32_load/"
        "i32_load/i32_add/i32_store");

  for (tcrv::rvv::I32LoadOp load : loads) {
    llvm::StringRef role = load.getBufferRole();
    if (role == support::stringifyRuntimeABIParameterRole(
                    support::RuntimeABIParameterRole::LHSInputBuffer)) {
      if (slice.lhsLoad)
        return makeRVVEmitCRouteProviderError(
            "bounded RVV EmitC route requires a unique lhs-input-buffer "
            "load");
      slice.lhsLoad = load;
      continue;
    }
    if (role == support::stringifyRuntimeABIParameterRole(
                    support::RuntimeABIParameterRole::RHSInputBuffer)) {
      if (slice.rhsLoad)
        return makeRVVEmitCRouteProviderError(
            "bounded RVV EmitC route requires a unique rhs-input-buffer "
            "load");
      slice.rhsLoad = load;
      continue;
    }
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV i32 load role '") + role +
        "' for bounded EmitC route");
  }

  if (!slice.lhsLoad || !slice.rhsLoad)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires lhs-input-buffer and "
        "rhs-input-buffer loads");
  if (slice.store.getBufferRole() !=
      support::stringifyRuntimeABIParameterRole(
          support::RuntimeABIParameterRole::OutputBuffer))
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires output-buffer store role");
  if (slice.add.getLhs() != slice.lhsLoad.getLoaded() ||
      slice.add.getRhs() != slice.rhsLoad.getLoaded())
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires tcrv_rvv.i32_add to consume lhs "
        "and rhs load results");
  if (slice.store.getValue() != slice.add.getSum())
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires tcrv_rvv.i32_store to consume the "
        "add result");

  return slice;
}

llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance>
getEmitCSourceProvenance(mlir::Operation *op, llvm::StringRef expectedRole) {
  auto lowerable =
      llvm::dyn_cast<conversion::emitc::TCRVEmitCLowerableOpInterface>(op);
  if (!lowerable)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("operation '") + op->getName().getStringRef() +
        "' must implement " + kEmitCLowerableOpInterfaceName +
        " before RVV EmitC route construction");

  llvm::StringRef sourceRole =
      lowerable.getTCRVEmitCLowerableSourceRole();
  if (sourceRole != expectedRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("operation '") + op->getName().getStringRef() +
        "' reports EmitC source role '" + sourceRole +
        "' but RVV route expected '" + expectedRole + "'");

  conversion::emitc::TCRVEmitCSourceOpProvenance source;
  source.opName = lowerable.getTCRVEmitCLowerableSourceOpName().str();
  source.role = sourceRole.str();
  source.opInterface = kEmitCLowerableOpInterfaceName.str();
  return source;
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

} // namespace

llvm::StringRef getRVVI32M1AddEmitCRouteID() {
  return kRVVI32M1AddEmitCRouteID;
}

llvm::StringRef getRVVI32M1AddEmissionKind() {
  return kRVVI32M1AddEmissionKind;
}

llvm::StringRef getRVVI32M1AddLoweringBoundaryOpName() {
  return kRVVI32M1AddLoweringBoundaryOpName;
}

llvm::StringRef getRVVI32M1AddRuntimeABIKind() {
  return kRVVI32M1AddRuntimeABIKind;
}

llvm::StringRef getRVVI32M1AddRuntimeABIName() {
  return kRVVI32M1AddRuntimeABIName;
}

llvm::StringRef getRVVI32M1AddRuntimeGlueRole() {
  return kRVVI32M1AddRuntimeGlueRole;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVI32M1AddRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs", "const int32_t *",
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "n", "size_t", support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::Error buildRVVI32M1AddEmitCLowerableRoute(
    const VariantEmitCLowerableRequest &request,
    conversion::emitc::TCRVEmitCLowerableRoute &out) {
  if (!request.getVariant())
    return makeRVVEmitCRouteProviderError(
        "EmitC route construction requires a materialized tcrv.exec.variant");
  if (!request.getKernel())
    return makeRVVEmitCRouteProviderError(
        "EmitC route construction requires an enclosing tcrv.exec.kernel");

  if (llvm::Error error = requireRVVVariantLegality(request.getVariant()))
    return error;

  llvm::Expected<RVVI32M1AddSlice> slice =
      collectRVVI32M1AddSlice(request.getVariant());
  if (!slice)
    return slice.takeError();

  conversion::emitc::TCRVEmitCLowerableRoute route(
      kRVVI32M1AddEmitCRouteID, "extension-family-ops-to-emitc-call-opaque");
  route.addHeader("stddef.h");
  route.addHeader("stdint.h");
  route.addHeader("riscv_vector.h");
  route.addTypeMapping("!tcrv_rvv.vl", "size_t");
  route.addTypeMapping("!tcrv_rvv.i32m1", "vint32m1_t");
  for (const support::RuntimeABIParameter &parameter :
       getRVVI32M1AddRuntimeABIParameters())
    route.addABIValueMapping(parameter, parameter.cName);

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

} // namespace tianchenrv::plugin::rvv
