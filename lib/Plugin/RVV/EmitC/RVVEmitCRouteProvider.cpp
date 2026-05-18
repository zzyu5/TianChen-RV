#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Support/RuntimeABIContract.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kRVVI32M1ArithmeticEmissionKind(
    "materialized-emitc-cpp-rvv-intrinsic-object");
constexpr llvm::StringLiteral kRVVI32M1ArithmeticLoweringBoundaryOpName(
    "tcrv_rvv.with_vl");
constexpr llvm::StringLiteral kRVVI32M1ArithmeticRuntimeABIKind(
    "plugin-owned-runtime-abi");
constexpr llvm::StringLiteral kRVVI32M1ArithmeticRuntimeGlueRole(
    "emitc-cpp-rvv-intrinsic-runtime-glue");
constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "TCRVEmitCLowerableOpInterface");

struct RVVI32M1ArithmeticRouteSpec {
  RVVI32M1ArithmeticOp op;
  llvm::StringLiteral mnemonic;
  llvm::StringLiteral intrinsic;
  llvm::StringLiteral resultName;
};

constexpr RVVI32M1ArithmeticRouteSpec kRVVI32M1ArithmeticRoutes[] = {
    {RVVI32M1ArithmeticOp::Add, "add", "__riscv_vadd_vv_i32m1", "sum_vec"},
    {RVVI32M1ArithmeticOp::Sub, "sub", "__riscv_vsub_vv_i32m1",
     "difference_vec"},
    {RVVI32M1ArithmeticOp::Mul, "mul", "__riscv_vmul_vv_i32m1",
     "product_vec"},
};

constexpr RVVI32M1ArithmeticOp kRVVI32M1ArithmeticOps[] = {
    RVVI32M1ArithmeticOp::Add, RVVI32M1ArithmeticOp::Sub,
    RVVI32M1ArithmeticOp::Mul};

const RVVI32M1ArithmeticRouteSpec &
getRVVI32M1ArithmeticRouteSpec(RVVI32M1ArithmeticOp op) {
  for (const RVVI32M1ArithmeticRouteSpec &spec :
       kRVVI32M1ArithmeticRoutes)
    if (spec.op == op)
      return spec;
  llvm_unreachable("unknown RVV i32m1 arithmetic op");
}

const RVVI32M1ArithmeticConstructionRoute &
getRVVI32M1ArithmeticConstructionRouteOrDie(RVVI32M1ArithmeticOp op) {
  const RVVI32M1ArithmeticRouteSpec &spec =
      getRVVI32M1ArithmeticRouteSpec(op);
  llvm::Expected<const RVVI32M1ArithmeticConstructionRoute *> route =
      lookupRVVI32M1ArithmeticConstructionRouteByMnemonic(
          spec.mnemonic);
  if (!route) {
    std::string message = llvm::toString(route.takeError());
    llvm::report_fatal_error(llvm::StringRef(message));
  }
  return **route;
}

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

struct RVVI32M1ArithmeticSlice {
  tcrv::rvv::SetVLOp setvl;
  tcrv::rvv::WithVLOp withVL;
  tcrv::rvv::I32LoadOp lhsLoad;
  tcrv::rvv::I32LoadOp rhsLoad;
  tcrv::rvv::I32BroadcastLoadOp rhsBroadcastLoad;
  mlir::Operation *arithmeticOp = nullptr;
  mlir::Value arithmeticLhs;
  mlir::Value arithmeticRhs;
  mlir::Value arithmeticResult;
  RVVI32M1ArithmeticOp arithmeticKind;
  tcrv::rvv::I32StoreOp store;
  support::RuntimeABIParameter lhsABI;
  support::RuntimeABIParameter rhsABI;
  support::RuntimeABIParameter outABI;
  support::RuntimeABIParameter runtimeElementCountABI;
};

std::string formatRuntimeABIExpectedRoles(
    llvm::ArrayRef<support::RuntimeABIParameterRole> expectedRoles) {
  std::string expected;
  llvm::raw_string_ostream stream(expected);
  llvm::interleave(
      expectedRoles,
      [&](support::RuntimeABIParameterRole role) {
        stream << "'"
               << support::stringifyRuntimeABIParameterRole(role) << "'";
      },
      [&] { stream << " or "; });
  stream.flush();
  return expected;
}

llvm::Expected<support::RuntimeABIParameter>
getRuntimeABIParameterBindingFromValue(
    mlir::Value value, llvm::StringRef context,
    llvm::ArrayRef<support::RuntimeABIParameterRole> expectedRoles) {
  auto binding = value.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (!binding)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " must be defined by explicit tcrv_rvv.runtime_abi_value before "
        "RVV EmitC route construction");
  if (llvm::Error error =
          verifyRVVRuntimeABIValueRoleOpInterface(binding.getOperation()))
    return std::move(error);

  std::optional<support::RuntimeABIParameterRole> role =
      support::symbolizeRuntimeABIParameterRole(binding.getRole());
  if (!role)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " carries unsupported runtime ABI role '" +
        binding.getRole() + "'");
  if (!llvm::is_contained(expectedRoles, *role)) {
    std::string expectedRolesText =
        formatRuntimeABIExpectedRoles(expectedRoles);
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " must bind runtime ABI role " +
        expectedRolesText + "; got '" +
        binding.getRole() + "'");
  }

  std::optional<support::RuntimeABIParameterOwnership> ownership =
      support::symbolizeRuntimeABIParameterOwnership(binding.getOwnership());
  if (!ownership)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " carries unsupported runtime ABI ownership '" +
        binding.getOwnership() + "'");

  return support::RuntimeABIParameter(binding.getCName(), binding.getCType(),
                                      *role, *ownership);
}

llvm::Error assignRVVLoadBinding(
    RVVI32M1ArithmeticSlice &slice, tcrv::rvv::I32LoadOp load,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role == support::RuntimeABIParameterRole::LHSInputBuffer) {
    if (slice.lhsLoad)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV EmitC route requires a unique lhs-input-buffer load");
    slice.lhsLoad = load;
    slice.lhsABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role == support::RuntimeABIParameterRole::RHSInputBuffer) {
    if (slice.rhsLoad)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV EmitC route requires a unique rhs-input-buffer load");
    slice.rhsLoad = load;
    slice.rhsABI = parameter;
    return llvm::Error::success();
  }

  return makeRVVEmitCRouteProviderError(
      llvm::Twine("unsupported RVV i32 load runtime ABI role '") +
      support::stringifyRuntimeABIParameterRole(parameter.role) +
      "' for bounded EmitC route");
}

llvm::Error assignRVVBroadcastLoadBinding(
    RVVI32M1ArithmeticSlice &slice, tcrv::rvv::I32BroadcastLoadOp load,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role != support::RuntimeABIParameterRole::RHSInputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV i32 broadcast load runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) +
        "' for bounded EmitC route");
  if (slice.rhsBroadcastLoad)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires a unique rhs broadcast load");
  slice.rhsBroadcastLoad = load;
  slice.rhsABI = parameter;
  return llvm::Error::success();
}

llvm::Error validateRVVI32M1ArithmeticRuntimeABIParameters(
    RVVI32M1ArithmeticSlice &slice,
    const RVVI32M1ArithmeticConstructionRoute &constructionRoute,
    const support::RuntimeABIParameter &runtimeElementCountABI,
    const support::RuntimeABIParameter &outABI) {
  slice.runtimeElementCountABI = runtimeElementCountABI;
  slice.outABI = outABI;

  llvm::SmallVector<support::RuntimeABIParameter, 4> ordered;
  ordered.push_back(slice.lhsABI);
  ordered.push_back(slice.rhsABI);
  ordered.push_back(slice.outABI);
  ordered.push_back(slice.runtimeElementCountABI);

  support::FiniteBinaryRuntimeABIContract contract(
      support::FiniteBinaryRuntimeABIContractSpec{
          constructionRoute.runtimeABIContractName, "const int32_t *",
          "int32_t *"});
  llvm::Expected<support::FiniteBinaryCallableRuntimeABIParameterBindings>
      bindings = support::bindFiniteBinaryCallableRuntimeABIParametersByRole(
          ordered, "RVV i32m1 arithmetic explicit runtime ABI values",
          contract);
  if (!bindings)
    return bindings.takeError();

  if (llvm::Error error =
          tcrv::rvv::verifyRVVI32M1ArithmeticRuntimeABIParameters(
              ordered, "bounded RVV EmitC route explicit runtime ABI values"))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));

  return llvm::Error::success();
}

llvm::Error recordRVVI32M1ArithmeticOp(RVVI32M1ArithmeticSlice &slice,
                                       mlir::Operation *op,
                                       RVVI32M1ArithmeticOp kind,
                                       mlir::Value lhs, mlir::Value rhs,
                                       mlir::Value result) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one supported "
        "tcrv_rvv.i32_add, tcrv_rvv.i32_sub, or tcrv_rvv.i32_mul op");
  slice.arithmeticOp = op;
  slice.arithmeticKind = kind;
  slice.arithmeticLhs = lhs;
  slice.arithmeticRhs = rhs;
  slice.arithmeticResult = result;
  return llvm::Error::success();
}

llvm::Expected<RVVI32M1ArithmeticSlice>
collectRVVI32M1ArithmeticSlice(tcrv::exec::VariantOp variant) {
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

  RVVI32M1ArithmeticSlice slice;
  slice.setvl = setvls.front();
  slice.withVL = withVLs.front();

  tcrv::rvv::RVVConfigContractDiagnostic configDiagnostic =
      tcrv::rvv::validateRVVI32M1ArithmeticConfigVLContract(slice.setvl,
                                                            slice.withVL);
  if (!configDiagnostic.ok)
    return makeRVVEmitCRouteProviderError(configDiagnostic.message);

  llvm::Expected<support::RuntimeABIParameter> runtimeElementCountABI =
      getRuntimeABIParameterBindingFromValue(
          slice.setvl.getAvl(), "tcrv_rvv.setvl AVL operand",
          {support::RuntimeABIParameterRole::RuntimeElementCount});
  if (!runtimeElementCountABI)
    return runtimeElementCountABI.takeError();

  llvm::SmallVector<tcrv::rvv::I32LoadOp, 2> loads;
  llvm::SmallVector<tcrv::rvv::I32BroadcastLoadOp, 1> broadcastLoads;
  unsigned storeCount = 0;
  for (mlir::Operation &op : slice.withVL.getBody().front()) {
    if (auto load = llvm::dyn_cast<tcrv::rvv::I32LoadOp>(op)) {
      loads.push_back(load);
      continue;
    }
    if (auto broadcastLoad =
            llvm::dyn_cast<tcrv::rvv::I32BroadcastLoadOp>(op)) {
      broadcastLoads.push_back(broadcastLoad);
      continue;
    }
    if (auto add = llvm::dyn_cast<tcrv::rvv::I32AddOp>(op)) {
      if (llvm::Error error =
              recordRVVI32M1ArithmeticOp(slice, add.getOperation(),
                                         RVVI32M1ArithmeticOp::Add,
                                         add.getLhs(), add.getRhs(),
                                         add.getSum()))
        return std::move(error);
      continue;
    }
    if (auto sub = llvm::dyn_cast<tcrv::rvv::I32SubOp>(op)) {
      if (llvm::Error error =
              recordRVVI32M1ArithmeticOp(slice, sub.getOperation(),
                                         RVVI32M1ArithmeticOp::Sub,
                                         sub.getLhs(), sub.getRhs(),
                                         sub.getDifference()))
        return std::move(error);
      continue;
    }
    if (auto mul = llvm::dyn_cast<tcrv::rvv::I32MulOp>(op)) {
      if (llvm::Error error =
              recordRVVI32M1ArithmeticOp(slice, mul.getOperation(),
                                         RVVI32M1ArithmeticOp::Mul,
                                         mul.getLhs(), mul.getRhs(),
                                         mul.getProduct()))
        return std::move(error);
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
        "' inside tcrv_rvv.with_vl; expected load or broadcast-load, "
        "arithmetic, and store only");
  }

  if (loads.size() != 2) {
    if (!(loads.size() == 1 && broadcastLoads.size() == 1))
      return makeRVVEmitCRouteProviderError(
          "bounded RVV EmitC route requires either two tcrv_rvv.i32_load ops "
          "or one tcrv_rvv.i32_load plus one "
          "tcrv_rvv.i32_broadcast_load op");
  }
  if (loads.size() == 2 && !broadcastLoads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route must not mix two vector loads with an "
        "additional tcrv_rvv.i32_broadcast_load op");
  if (loads.empty() || broadcastLoads.size() > 1)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires a unique lhs load and at most one "
        "rhs broadcast load");
  if (!slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one supported "
        "tcrv_rvv.i32_add, tcrv_rvv.i32_sub, or tcrv_rvv.i32_mul op");
  if (storeCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one tcrv_rvv.i32_store op");
  if (rvvOpCount != 10)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route supports only runtime_abi_value/"
        "runtime_abi_value/runtime_abi_value/runtime_abi_value/setvl/"
        "with_vl/i32_load/i32_load_or_i32_broadcast_load/"
        "i32_add_or_i32_sub_or_i32_mul/i32_store");

  for (tcrv::rvv::I32LoadOp load : loads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            load.getBuffer(), "tcrv_rvv.i32_load buffer operand",
            {support::RuntimeABIParameterRole::LHSInputBuffer,
             support::RuntimeABIParameterRole::RHSInputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error = assignRVVLoadBinding(slice, load, *parameter))
      return error;
  }
  for (tcrv::rvv::I32BroadcastLoadOp broadcastLoad : broadcastLoads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            broadcastLoad.getBuffer(),
            "tcrv_rvv.i32_broadcast_load buffer operand",
            {support::RuntimeABIParameterRole::RHSInputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVBroadcastLoadBinding(slice, broadcastLoad, *parameter))
      return error;
  }

  if (!slice.lhsLoad || (!slice.rhsLoad && !slice.rhsBroadcastLoad))
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires a lhs-input-buffer load and either "
        "a rhs-input-buffer load or rhs broadcast load");
  llvm::Expected<support::RuntimeABIParameter> outABI =
      getRuntimeABIParameterBindingFromValue(
          slice.store.getBuffer(), "tcrv_rvv.i32_store buffer operand",
          {support::RuntimeABIParameterRole::OutputBuffer});
  if (!outABI)
    return outABI.takeError();
  const RVVI32M1ArithmeticRouteSpec &spec =
      getRVVI32M1ArithmeticRouteSpec(slice.arithmeticKind);
  const RVVI32M1ArithmeticConstructionRoute &constructionRoute =
      getRVVI32M1ArithmeticConstructionRouteOrDie(slice.arithmeticKind);
  if (llvm::Error error = validateRVVI32M1ArithmeticRuntimeABIParameters(
          slice, constructionRoute, *runtimeElementCountABI, *outABI))
    return error;
  mlir::Value rhsValue = slice.rhsBroadcastLoad
                             ? slice.rhsBroadcastLoad.getBroadcast()
                             : slice.rhsLoad.getLoaded();
  if (slice.arithmeticLhs != slice.lhsLoad.getLoaded() ||
      slice.arithmeticRhs != rhsValue)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("bounded RVV EmitC route requires tcrv_rvv.i32_") +
        spec.mnemonic + " to consume lhs load and explicit rhs vector or "
                        "broadcast results");
  if (slice.store.getValue() != slice.arithmeticResult)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires tcrv_rvv.i32_store to consume the "
        "arithmetic result");

  return slice;
}

llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance>
getEmitCSourceProvenance(mlir::Operation *op, llvm::StringRef expectedRole) {
  if (llvm::Error error = verifyRVVRoleOperationInterface(op, expectedRole))
    return std::move(error);

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

llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep>
makeCallStepFromSource(
    mlir::Operation *op,
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
  return step;
}

llvm::Error addCallStepFromSource(
    conversion::emitc::TCRVEmitCLowerableRoute &route, mlir::Operation *op,
    llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> step =
      makeCallStepFromSource(op, expectedRole, callee, operands,
                             std::move(result));
  if (!step)
    return step.takeError();
  route.addCallOpaqueStep(std::move(*step));
  return llvm::Error::success();
}

struct RVVOrderedRoleOperations {
  llvm::SmallVector<mlir::Operation *, 10> operations;
  llvm::SmallVector<unsigned, 10> constructionOrders;
};

unsigned getRVVCanonicalRoleOrder(RVVI32M1ArithmeticSlice &slice,
                                  mlir::Operation *op) {
  auto lhsABI = slice.lhsLoad.getBuffer()
                    .getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  mlir::Value rhsBuffer = slice.rhsBroadcastLoad
                              ? slice.rhsBroadcastLoad.getBuffer()
                              : slice.rhsLoad.getBuffer();
  auto rhsABI = rhsBuffer.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  auto outABI = slice.store.getBuffer()
                    .getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  auto nABI =
      slice.setvl.getAvl().getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (lhsABI && op == lhsABI.getOperation())
    return 0;
  if (rhsABI && op == rhsABI.getOperation())
    return 1;
  if (outABI && op == outABI.getOperation())
    return 2;
  if (nABI && op == nABI.getOperation())
    return 3;
  if (op == slice.setvl.getOperation())
    return 4;
  if (op == slice.withVL.getOperation())
    return 5;
  if (op == slice.lhsLoad.getOperation())
    return 6;
  if (slice.rhsLoad && op == slice.rhsLoad.getOperation())
    return 7;
  if (slice.rhsBroadcastLoad && op == slice.rhsBroadcastLoad.getOperation())
    return 7;
  if (op == slice.arithmeticOp)
    return 8;
  if (op == slice.store.getOperation())
    return 9;
  return 10;
}

RVVOrderedRoleOperations
collectRVVRoleOperationsInBodyOrder(tcrv::exec::VariantOp variant,
                                    RVVI32M1ArithmeticSlice &slice) {
  RVVOrderedRoleOperations ordered;
  if (!variant || variant.getBody().empty())
    return ordered;

  for (mlir::Operation &op : variant.getBody().front()) {
    if (op.getName().getDialectNamespace() != "tcrv_rvv")
      continue;
    ordered.operations.push_back(&op);
    ordered.constructionOrders.push_back(getRVVCanonicalRoleOrder(slice, &op));
    if (auto withVL = llvm::dyn_cast<tcrv::rvv::WithVLOp>(op))
      for (mlir::Operation &nested : withVL.getBody().front())
        if (nested.getName().getDialectNamespace() == "tcrv_rvv") {
          ordered.operations.push_back(&nested);
          ordered.constructionOrders.push_back(
              getRVVCanonicalRoleOrder(slice, &nested));
        }
  }
  return ordered;
}

llvm::Error verifySelectedRVVRoleSequence(
    RVVI32M1ArithmeticSlice &slice, const VariantEmitCLowerableRequest &request,
    const RVVI32M1ArithmeticConstructionRoute &constructionRoute) {
  auto lhsABI = slice.lhsLoad.getBuffer()
                    .getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  mlir::Value rhsBuffer = slice.rhsBroadcastLoad
                              ? slice.rhsBroadcastLoad.getBuffer()
                              : slice.rhsLoad.getBuffer();
  auto rhsABI = rhsBuffer.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  auto outABI = slice.store.getBuffer()
                    .getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  auto nABI =
      slice.setvl.getAvl().getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (!lhsABI || !rhsABI || !outABI || !nABI)
    return makeRVVEmitCRouteProviderError(
        "selected RVV construction role sequence requires runtime ABI values "
        "to be explicit tcrv_rvv.runtime_abi_value ops");

  mlir::ArrayAttr requires =
      request.getVariant()->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!requires || requires.empty())
    return makeRVVEmitCRouteProviderError(
        "selected RVV construction role sequence requires non-empty selected "
        "variant requires metadata");

  RVVOrderedRoleOperations ordered =
      collectRVVRoleOperationsInBodyOrder(request.getVariant(), slice);
  llvm::StringRef rhsSourceOperationName =
      slice.rhsBroadcastLoad ? "tcrv_rvv.i32_broadcast_load"
                             : "tcrv_rvv.i32_load";
  return verifyRVVI32M1ArithmeticSelectedRoleSequence(
      ordered.operations, ordered.constructionOrders,
      request.getVariant().getSymName(),
      stringifyVariantEmissionRole(request.getRole()),
      constructionRoute.operationName, rhsSourceOperationName,
      "selected RVV EmitC route");
}

} // namespace

llvm::ArrayRef<RVVI32M1ArithmeticOp> getRVVI32M1ArithmeticOps() {
  return kRVVI32M1ArithmeticOps;
}

llvm::StringRef stringifyRVVI32M1ArithmeticOp(RVVI32M1ArithmeticOp op) {
  return getRVVI32M1ArithmeticRouteSpec(op).mnemonic;
}

llvm::Expected<RVVI32M1ArithmeticOp>
symbolizeRVVI32M1ArithmeticOpFromEmitCRouteID(llvm::StringRef routeID) {
  llvm::Expected<const RVVI32M1ArithmeticConstructionRoute *>
      constructionRoute =
          lookupRVVI32M1ArithmeticConstructionRouteByEmitCRouteID(routeID);
  if (!constructionRoute)
    return constructionRoute.takeError();
  for (const RVVI32M1ArithmeticRouteSpec &spec :
       kRVVI32M1ArithmeticRoutes)
    if ((*constructionRoute)->mnemonic == spec.mnemonic)
      return spec.op;
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("RVV construction route mnemonic '") +
      (*constructionRoute)->mnemonic + "' has no route provider operation");
}

llvm::StringRef
getRVVI32M1ArithmeticEmitCRouteID(RVVI32M1ArithmeticOp op) {
  return getRVVI32M1ArithmeticConstructionRouteOrDie(op).emitCRouteID;
}

llvm::StringRef getRVVI32M1ArithmeticEmissionKind() {
  return kRVVI32M1ArithmeticEmissionKind;
}

llvm::StringRef getRVVI32M1ArithmeticLoweringBoundaryOpName() {
  return kRVVI32M1ArithmeticLoweringBoundaryOpName;
}

llvm::StringRef getRVVI32M1ArithmeticRuntimeABIKind() {
  return kRVVI32M1ArithmeticRuntimeABIKind;
}

llvm::StringRef
getRVVI32M1ArithmeticRuntimeABIName(RVVI32M1ArithmeticOp op) {
  return getRVVI32M1ArithmeticConstructionRouteOrDie(op).runtimeABIName;
}

llvm::StringRef getRVVI32M1ArithmeticRuntimeGlueRole() {
  return kRVVI32M1ArithmeticRuntimeGlueRole;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVI32M1ArithmeticRuntimeABIParameters() {
  return tcrv::rvv::getRVVI32M1ArithmeticRuntimeABIParameters();
}

llvm::Error buildRVVI32M1ArithmeticEmitCLowerableRouteForOperation(
    RVVI32M1ArithmeticOp op, const VariantEmitCLowerableRequest &request,
    conversion::emitc::TCRVEmitCLowerableRoute &out) {
  const RVVI32M1ArithmeticRouteSpec &expectedSpec =
      getRVVI32M1ArithmeticRouteSpec(op);
  if (!request.getVariant())
    return makeRVVEmitCRouteProviderError(
        "EmitC route construction requires a materialized tcrv.exec.variant");
  if (!request.getKernel())
    return makeRVVEmitCRouteProviderError(
        "EmitC route construction requires an enclosing tcrv.exec.kernel");

  if (llvm::Error error = requireRVVVariantLegality(request.getVariant()))
    return error;
  if (llvm::Error error = verifyRVVConstructionProtocolReady())
    return error;

  llvm::Expected<RVVI32M1ArithmeticSlice> slice =
      collectRVVI32M1ArithmeticSlice(request.getVariant());
  if (!slice)
    return slice.takeError();

  if (slice->arithmeticKind != op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV i32m1 arithmetic route expected i32_") +
        expectedSpec.mnemonic + " but variant body contains i32_" +
        stringifyRVVI32M1ArithmeticOp(slice->arithmeticKind));

  const RVVI32M1ArithmeticConstructionRoute &constructionRoute =
      getRVVI32M1ArithmeticConstructionRouteOrDie(op);
  if (slice->arithmeticOp->getName().getStringRef() !=
      constructionRoute.operationName)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV i32m1 arithmetic route expected typed op '") +
        constructionRoute.operationName + "' from the construction mapping");
  if (llvm::Error error = verifyRVVI32M1ArithmeticConstructionRouteMapping(
          expectedSpec.mnemonic, constructionRoute.operationName,
          constructionRoute.emitCRouteID, constructionRoute.runtimeABIName))
    return error;
  if (llvm::Error error =
          verifySelectedRVVRoleSequence(*slice, request, constructionRoute))
    return error;

  conversion::emitc::TCRVEmitCLowerableRoute route(
      constructionRoute.emitCRouteID,
      "extension-family-ops-to-emitc-call-opaque");
  route.addHeader("stddef.h");
  route.addHeader("stdint.h");
  route.addHeader("riscv_vector.h");
  route.addTypeMapping("!tcrv_rvv.vl", "size_t");
  route.addTypeMapping("!tcrv_rvv.i32m1", "vint32m1_t");
  const support::RuntimeABIParameter runtimeABIParameters[] = {
      slice->lhsABI, slice->rhsABI, slice->outABI,
      slice->runtimeElementCountABI};
  for (const support::RuntimeABIParameter &parameter : runtimeABIParameters)
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
          {TCRVEmitCCallOpaqueOperand{
              slice->runtimeElementCountABI.cName,
              slice->runtimeElementCountABI.cType}},
          TCRVEmitCCallOpaqueResult{"full_chunk_vl", "size_t"}))
    return error;

  conversion::emitc::TCRVEmitCForLoop loop;
  llvm::StringRef inductionName =
      tcrv::rvv::getRVVI32M1ArithmeticEmitCLoopInductionName();
  llvm::StringRef fullChunkVLName =
      tcrv::rvv::getRVVI32M1ArithmeticEmitCFullChunkVLName();
  llvm::StringRef loopVLName =
      tcrv::rvv::getRVVI32M1ArithmeticEmitCLoopVLName();
  loop.inductionVarName = inductionName.str();
  loop.lowerBound = TCRVEmitCCallOpaqueOperand{"0", "size_t"};
  loop.upperBound = TCRVEmitCCallOpaqueOperand{
      slice->runtimeElementCountABI.cName, slice->runtimeElementCountABI.cType};
  loop.step = TCRVEmitCCallOpaqueOperand{fullChunkVLName.str(), "size_t"};

  auto addLoopStep = [&](mlir::Operation *op, llvm::StringRef role,
                         llvm::StringRef callee,
                         llvm::ArrayRef<TCRVEmitCCallOpaqueOperand> operands,
                         std::optional<TCRVEmitCCallOpaqueResult> result =
                             std::nullopt) -> llvm::Error {
    llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> step =
        makeCallStepFromSource(op, role, callee, operands, std::move(result));
    if (!step)
      return step.takeError();
    loop.bodySteps.push_back(std::move(*step));
    return llvm::Error::success();
  };

  if (llvm::Error error = addLoopStep(
          slice->setvl.getOperation(), "configure", "__riscv_vsetvl_e32m1",
          {TCRVEmitCCallOpaqueOperand{
              tcrv::rvv::getRVVI32M1ArithmeticEmitCRemainingAVLExpression(
                  slice->runtimeElementCountABI.cName, inductionName),
              "size_t"}},
          TCRVEmitCCallOpaqueResult{loopVLName.str(), "size_t"}))
    return error;
  if (llvm::Error error = addLoopStep(
          slice->lhsLoad.getOperation(), "load", "__riscv_vle32_v_i32m1",
          {TCRVEmitCCallOpaqueOperand{
               (llvm::StringRef(slice->lhsABI.cName) + " + " + inductionName)
                   .str(),
               slice->lhsABI.cType},
           TCRVEmitCCallOpaqueOperand{loopVLName.str(), "size_t"}},
          TCRVEmitCCallOpaqueResult{"lhs_vec", "vint32m1_t"}))
    return error;
  if (slice->rhsBroadcastLoad) {
    if (llvm::Error error = addLoopStep(
            slice->rhsBroadcastLoad.getOperation(), "load",
            "__riscv_vmv_v_x_i32m1",
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->rhsABI.cName) + "[0]").str(),
                 "int32_t"},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), "size_t"}},
            TCRVEmitCCallOpaqueResult{"rhs_vec", "vint32m1_t"}))
      return error;
  } else {
    if (llvm::Error error = addLoopStep(
            slice->rhsLoad.getOperation(), "load", "__riscv_vle32_v_i32m1",
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->rhsABI.cName) + " + " +
                  inductionName)
                     .str(),
                 slice->rhsABI.cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), "size_t"}},
            TCRVEmitCCallOpaqueResult{"rhs_vec", "vint32m1_t"}))
      return error;
  }
  if (llvm::Error error = addLoopStep(
          slice->arithmeticOp, "compute", expectedSpec.intrinsic,
          {TCRVEmitCCallOpaqueOperand{"lhs_vec", "vint32m1_t"},
           TCRVEmitCCallOpaqueOperand{"rhs_vec", "vint32m1_t"},
           TCRVEmitCCallOpaqueOperand{loopVLName.str(), "size_t"}},
          TCRVEmitCCallOpaqueResult{expectedSpec.resultName.str(),
                                    "vint32m1_t"}))
    return error;
  if (llvm::Error error = addLoopStep(
          slice->store.getOperation(), "store", "__riscv_vse32_v_i32m1",
          {TCRVEmitCCallOpaqueOperand{
               (llvm::StringRef(slice->outABI.cName) + " + " + inductionName)
                   .str(),
               slice->outABI.cType},
           TCRVEmitCCallOpaqueOperand{expectedSpec.resultName.str(),
                                      "vint32m1_t"},
           TCRVEmitCCallOpaqueOperand{loopVLName.str(), "size_t"}}))
    return error;

  route.addForLoop(std::move(loop));

  out = std::move(route);
  return llvm::Error::success();
}

llvm::Error buildRVVI32M1ArithmeticEmitCLowerableRoute(
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

  llvm::Expected<RVVI32M1ArithmeticSlice> slice =
      collectRVVI32M1ArithmeticSlice(request.getVariant());
  if (!slice)
    return slice.takeError();
  return buildRVVI32M1ArithmeticEmitCLowerableRouteForOperation(
      slice->arithmeticKind, request, out);
}

llvm::Error buildRVVI32M1AddEmitCLowerableRoute(
    const VariantEmitCLowerableRequest &request,
    conversion::emitc::TCRVEmitCLowerableRoute &out) {
  return buildRVVI32M1ArithmeticEmitCLowerableRouteForOperation(
      RVVI32M1ArithmeticOp::Add, request, out);
}

llvm::Error buildRVVI32M1SubEmitCLowerableRoute(
    const VariantEmitCLowerableRequest &request,
    conversion::emitc::TCRVEmitCLowerableRoute &out) {
  return buildRVVI32M1ArithmeticEmitCLowerableRouteForOperation(
      RVVI32M1ArithmeticOp::Sub, request, out);
}

llvm::Error buildRVVI32M1MulEmitCLowerableRoute(
    const VariantEmitCLowerableRequest &request,
    conversion::emitc::TCRVEmitCLowerableRoute &out) {
  return buildRVVI32M1ArithmeticEmitCLowerableRouteForOperation(
      RVVI32M1ArithmeticOp::Mul, request, out);
}

} // namespace tianchenrv::plugin::rvv
