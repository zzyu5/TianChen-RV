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
constexpr llvm::StringLiteral
    kRVVSelectedBodyEmissionKind("materialized-emitc-cpp-rvv-intrinsic-object");
constexpr llvm::StringLiteral
    kRVVSelectedBodyLoweringBoundaryOpName("tcrv_rvv.with_vl");
constexpr llvm::StringLiteral
    kRVVSelectedBodyRuntimeABIKind("plugin-owned-runtime-abi");
constexpr llvm::StringLiteral
    kRVVSelectedBodyRuntimeGlueRole("emitc-cpp-rvv-intrinsic-runtime-glue");
constexpr llvm::StringLiteral
    kEmitCLowerableOpInterfaceName("TCRVEmitCLowerableOpInterface");

llvm::Error makeRVVEmitCRouteProviderError(llvm::Twine message);

struct RVVSelectedBodySpecializationMapping {
  RVVSelectedBodyOperationKind operation;
  std::int64_t sew;
  llvm::StringLiteral lmul;
  llvm::StringLiteral tailPolicy;
  llvm::StringLiteral maskPolicy;
  bool supportsRHSBroadcastLoad;
  llvm::StringLiteral operationMnemonic;
  llvm::StringLiteral vlCType;
  llvm::StringLiteral vectorTypeName;
  llvm::StringLiteral maskTypeName;
  llvm::StringLiteral vectorCType;
  llvm::StringLiteral maskCType;
  llvm::StringLiteral setVLIntrinsic;
  llvm::StringLiteral vectorLoadIntrinsic;
  llvm::StringLiteral rhsBroadcastIntrinsic;
  llvm::StringLiteral storeIntrinsic;
  llvm::StringLiteral computeIntrinsic;
  llvm::StringLiteral compareIntrinsic;
  llvm::StringLiteral resultName;
  llvm::StringLiteral maskName;
};

constexpr RVVSelectedBodySpecializationMapping
    kRVVSelectedBodySpecializationMappings[] = {
        {RVVSelectedBodyOperationKind::Add,
         32,
         "m1",
         "agnostic",
         "agnostic",
         true,
         "add",
         "size_t",
         "!tcrv_rvv.i32m1",
         "!tcrv_rvv.i32m1_mask",
         "vint32m1_t",
         "vbool32_t",
         "__riscv_vsetvl_e32m1",
         "__riscv_vle32_v_i32m1",
         "__riscv_vmv_v_x_i32m1",
         "__riscv_vse32_v_i32m1",
         "__riscv_vadd_vv_i32m1",
         "",
         "sum_vec",
         ""},
        {RVVSelectedBodyOperationKind::Sub,
         32,
         "m1",
         "agnostic",
         "agnostic",
         true,
         "sub",
         "size_t",
         "!tcrv_rvv.i32m1",
         "!tcrv_rvv.i32m1_mask",
         "vint32m1_t",
         "vbool32_t",
         "__riscv_vsetvl_e32m1",
         "__riscv_vle32_v_i32m1",
         "__riscv_vmv_v_x_i32m1",
         "__riscv_vse32_v_i32m1",
         "__riscv_vsub_vv_i32m1",
         "",
         "difference_vec",
         ""},
        {RVVSelectedBodyOperationKind::Mul,
         32,
         "m1",
         "agnostic",
         "agnostic",
         true,
         "mul",
         "size_t",
         "!tcrv_rvv.i32m1",
         "!tcrv_rvv.i32m1_mask",
         "vint32m1_t",
         "vbool32_t",
         "__riscv_vsetvl_e32m1",
         "__riscv_vle32_v_i32m1",
         "__riscv_vmv_v_x_i32m1",
         "__riscv_vse32_v_i32m1",
         "__riscv_vmul_vv_i32m1",
         "",
         "product_vec",
         ""},
        {RVVSelectedBodyOperationKind::CmpSelect,
         32,
         "m1",
         "agnostic",
         "agnostic",
         false,
         "cmp_select",
         "size_t",
         "!tcrv_rvv.i32m1",
         "!tcrv_rvv.i32m1_mask",
         "vint32m1_t",
         "vbool32_t",
         "__riscv_vsetvl_e32m1",
         "__riscv_vle32_v_i32m1",
         "",
         "__riscv_vse32_v_i32m1",
         "__riscv_vmerge_vvm_i32m1",
         "__riscv_vmseq_vv_i32m1_b32",
         "selected_vec",
         "cmp_mask"},
};

constexpr RVVSelectedBodyOperationKind kRVVSelectedBodyOperationKinds[] = {
    RVVSelectedBodyOperationKind::Add, RVVSelectedBodyOperationKind::Sub,
    RVVSelectedBodyOperationKind::Mul, RVVSelectedBodyOperationKind::CmpSelect};

const RVVSelectedBodySpecializationMapping &
getRVVSelectedBodySpecializationMappingByOperation(
    RVVSelectedBodyOperationKind op) {
  for (const RVVSelectedBodySpecializationMapping &spec :
       kRVVSelectedBodySpecializationMappings)
    if (spec.operation == op)
      return spec;
  llvm_unreachable("unknown RVV selected-body operation");
}

bool supportsRVVSelectedBodyMemoryForm(
    const RVVSelectedBodySpecializationMapping &mapping,
    RVVSelectedBodyMemoryForm memoryForm) {
  switch (memoryForm) {
  case RVVSelectedBodyMemoryForm::VectorRHSLoad:
    return true;
  case RVVSelectedBodyMemoryForm::RHSBroadcastLoad:
    return mapping.supportsRHSBroadcastLoad;
  }
  llvm_unreachable("unknown RVV selected-body memory form");
}

llvm::Expected<const RVVSelectedBodySpecializationMapping *>
lookupRVVSelectedBodySpecializationMapping(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  for (const RVVSelectedBodySpecializationMapping &mapping :
       kRVVSelectedBodySpecializationMappings) {
    if (mapping.operation == description.operation &&
        mapping.sew == description.sew && mapping.lmul == description.lmul &&
        mapping.tailPolicy == description.tailPolicy &&
        mapping.maskPolicy == description.maskPolicy &&
        supportsRVVSelectedBodyMemoryForm(mapping, description.memoryForm))
      return &mapping;
  }

  return makeRVVEmitCRouteProviderError(
      llvm::Twine(
          "unsupported RVV selected-body route specialization: operation=") +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      ", memory_form=" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) +
      ", SEW=" + llvm::Twine(description.sew) + ", LMUL=" + description.lmul +
      ", tail_policy=" + description.tailPolicy +
      ", mask_policy=" + description.maskPolicy);
}

llvm::Error requireRouteDescriptionText(llvm::StringRef context,
                                        llvm::StringRef field,
                                        llvm::StringRef value) {
  if (!value.trim().empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(llvm::Twine(context) + " " + field +
                                        " must be provider-derived and "
                                        "non-empty");
}

llvm::Error requireRouteDescriptionField(llvm::StringRef context,
                                         llvm::StringRef field,
                                         llvm::StringRef actual,
                                         llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) + " " + field +
      " must mirror selected-body specialization fact '" + expected +
      "' but was '" + actual + "'");
}

llvm::StringRef stringifyRVVTailPolicy(tcrv::rvv::TailPolicy policy) {
  switch (policy) {
  case tcrv::rvv::TailPolicy::Agnostic:
    return "agnostic";
  case tcrv::rvv::TailPolicy::Undisturbed:
    return "undisturbed";
  }
  llvm_unreachable("unknown RVV tail policy");
}

llvm::StringRef stringifyRVVMaskPolicy(tcrv::rvv::MaskPolicy policy) {
  switch (policy) {
  case tcrv::rvv::MaskPolicy::Agnostic:
    return "agnostic";
  case tcrv::rvv::MaskPolicy::Undisturbed:
    return "undisturbed";
  }
  llvm_unreachable("unknown RVV mask policy");
}

const RVVSelectedBodyConstructionRoute &
getRVVSelectedBodyConstructionRouteOrDie(RVVSelectedBodyOperationKind op) {
  const RVVSelectedBodySpecializationMapping &spec =
      getRVVSelectedBodySpecializationMappingByOperation(op);
  llvm::Expected<const RVVSelectedBodyConstructionRoute *> route =
      lookupRVVSelectedBodyConstructionRouteByOperationMnemonic(
          spec.operationMnemonic);
  if (!route) {
    std::string message = llvm::toString(route.takeError());
    llvm::report_fatal_error(llvm::StringRef(message));
  }
  return **route;
}

llvm::Error makeRVVEmitCRouteProviderError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine(
          "TianChen-RV RVV plugin-owned EmitC route provider failed: ") +
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

struct RVVSelectedBodyRouteSlice {
  tcrv::rvv::SetVLOp setvl;
  tcrv::rvv::WithVLOp withVL;
  tcrv::rvv::I32LoadOp lhsLoad;
  tcrv::rvv::I32LoadOp rhsLoad;
  tcrv::rvv::I32BroadcastLoadOp rhsBroadcastLoad;
  mlir::Operation *arithmeticOp = nullptr;
  mlir::Value arithmeticLhs;
  mlir::Value arithmeticRhs;
  mlir::Value arithmeticResult;
  RVVSelectedBodyOperationKind arithmeticKind;
  tcrv::rvv::I32CmpEqOp compareOp;
  tcrv::rvv::I32SelectOp selectOp;
  mlir::Value compareLhs;
  mlir::Value compareRhs;
  mlir::Value compareMask;
  mlir::Value selectMask;
  mlir::Value selectTrueValue;
  mlir::Value selectFalseValue;
  mlir::Value selectResult;
  tcrv::rvv::I32StoreOp store;
  support::RuntimeABIParameter lhsABI;
  support::RuntimeABIParameter rhsABI;
  support::RuntimeABIParameter outABI;
  support::RuntimeABIParameter runtimeElementCountABI;
};

struct RVVSelectedBodyRouteAnalysis {
  RVVSelectedBodyRouteSlice slice;
  const RVVSelectedBodySpecializationMapping *specializationMapping = nullptr;
  const RVVSelectedBodyConstructionRoute *constructionRoute = nullptr;
  RVVSelectedBodyEmitCRouteDescription description;
};

std::string formatRuntimeABIExpectedRoles(
    llvm::ArrayRef<support::RuntimeABIParameterRole> expectedRoles) {
  std::string expected;
  llvm::raw_string_ostream stream(expected);
  llvm::interleave(
      expectedRoles,
      [&](support::RuntimeABIParameterRole role) {
        stream << "'" << support::stringifyRuntimeABIParameterRole(role) << "'";
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
        expectedRolesText + "; got '" + binding.getRole() + "'");
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

llvm::Error
assignRVVLoadBinding(RVVSelectedBodyRouteSlice &slice,
                     tcrv::rvv::I32LoadOp load,
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

llvm::Error
assignRVVBroadcastLoadBinding(RVVSelectedBodyRouteSlice &slice,
                              tcrv::rvv::I32BroadcastLoadOp load,
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

llvm::Error validateRVVSelectedBodyRuntimeABIParameters(
    RVVSelectedBodyRouteSlice &slice,
    const RVVSelectedBodyConstructionRoute &constructionRoute,
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

  if (llvm::Error error = tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
          ordered, "selected RVV EmitC route explicit runtime ABI values"))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));

  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyOperation(RVVSelectedBodyRouteSlice &slice,
                                           mlir::Operation *op,
                                           RVVSelectedBodyOperationKind kind,
                                           mlir::Value lhs, mlir::Value rhs,
                                           mlir::Value result) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one supported "
        "tcrv_rvv.i32_add, tcrv_rvv.i32_sub, tcrv_rvv.i32_mul, or "
        "tcrv_rvv.i32_select op");
  slice.arithmeticOp = op;
  slice.arithmeticKind = kind;
  slice.arithmeticLhs = lhs;
  slice.arithmeticRhs = rhs;
  slice.arithmeticResult = result;
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyCompareOp(RVVSelectedBodyRouteSlice &slice,
                                           tcrv::rvv::I32CmpEqOp compare) {
  if (slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV compare/select EmitC route requires exactly one "
        "tcrv_rvv.i32_cmp_eq op");
  slice.compareOp = compare;
  slice.compareLhs = compare.getLhs();
  slice.compareRhs = compare.getRhs();
  slice.compareMask = compare.getMask();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodySelectOp(RVVSelectedBodyRouteSlice &slice,
                                          tcrv::rvv::I32SelectOp select) {
  if (llvm::Error error = recordRVVSelectedBodyOperation(
          slice, select.getOperation(), RVVSelectedBodyOperationKind::CmpSelect,
          select.getTrueValue(), select.getFalseValue(), select.getSelected()))
    return error;
  slice.selectOp = select;
  slice.selectMask = select.getMask();
  slice.selectTrueValue = select.getTrueValue();
  slice.selectFalseValue = select.getFalseValue();
  slice.selectResult = select.getSelected();
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyRouteSlice>
collectRVVSelectedBodyRouteSlice(tcrv::exec::VariantOp variant) {
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

  RVVSelectedBodyRouteSlice slice;
  slice.setvl = setvls.front();
  slice.withVL = withVLs.front();

  tcrv::rvv::RVVConfigContractDiagnostic configDiagnostic =
      tcrv::rvv::validateRVVSelectedBodyConfigVLStructure(slice.setvl,
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
      if (llvm::Error error = recordRVVSelectedBodyOperation(
              slice, add.getOperation(), RVVSelectedBodyOperationKind::Add,
              add.getLhs(), add.getRhs(), add.getSum()))
        return std::move(error);
      continue;
    }
    if (auto sub = llvm::dyn_cast<tcrv::rvv::I32SubOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodyOperation(
              slice, sub.getOperation(), RVVSelectedBodyOperationKind::Sub,
              sub.getLhs(), sub.getRhs(), sub.getDifference()))
        return std::move(error);
      continue;
    }
    if (auto mul = llvm::dyn_cast<tcrv::rvv::I32MulOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodyOperation(
              slice, mul.getOperation(), RVVSelectedBodyOperationKind::Mul,
              mul.getLhs(), mul.getRhs(), mul.getProduct()))
        return std::move(error);
      continue;
    }
    if (auto compare = llvm::dyn_cast<tcrv::rvv::I32CmpEqOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodyCompareOp(slice, compare))
        return std::move(error);
      continue;
    }
    if (auto select = llvm::dyn_cast<tcrv::rvv::I32SelectOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodySelectOp(slice, select))
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
        "arithmetic or compare/select compute, and store only");
  }

  const bool isCompareSelect = slice.compareOp || slice.selectOp;
  if (isCompareSelect) {
    if (!slice.compareOp || !slice.selectOp)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV compare/select EmitC route requires explicit "
          "tcrv_rvv.i32_cmp_eq and tcrv_rvv.i32_select ops");
    if (loads.size() != 2 || !broadcastLoads.empty())
      return makeRVVEmitCRouteProviderError(
          "bounded RVV compare/select EmitC route requires exactly two "
          "tcrv_rvv.i32_load ops and does not support broadcast predicates in "
          "this slice");
  } else if (loads.size() != 2) {
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
        "tcrv_rvv.i32_add, tcrv_rvv.i32_sub, tcrv_rvv.i32_mul, or "
        "tcrv_rvv.i32_select op");
  if (storeCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one tcrv_rvv.i32_store op");
  unsigned expectedRVVOpCount = isCompareSelect ? 11 : 10;
  if (rvvOpCount != expectedRVVOpCount)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route supports only runtime_abi_value/"
        "runtime_abi_value/runtime_abi_value/runtime_abi_value/setvl/"
        "with_vl/i32_load/i32_load_or_i32_broadcast_load/"
        "i32_add_or_i32_sub_or_i32_mul_or_i32_cmp_eq_i32_select/i32_store");

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
  llvm::StringRef operationMnemonic =
      stringifyRVVSelectedBodyOperationKind(slice.arithmeticKind);
  slice.runtimeElementCountABI = *runtimeElementCountABI;
  slice.outABI = *outABI;
  mlir::Value rhsValue = slice.rhsBroadcastLoad
                             ? slice.rhsBroadcastLoad.getBroadcast()
                             : slice.rhsLoad.getLoaded();
  if (slice.arithmeticKind == RVVSelectedBodyOperationKind::CmpSelect) {
    auto isLoadedDataValue = [&](mlir::Value value) {
      return value == slice.lhsLoad.getLoaded() ||
             value == slice.rhsLoad.getLoaded();
    };
    if (!isLoadedDataValue(slice.compareLhs) ||
        !isLoadedDataValue(slice.compareRhs))
      return makeRVVEmitCRouteProviderError(
          "bounded RVV compare/select EmitC route requires "
          "tcrv_rvv.i32_cmp_eq operands to be explicit lhs/rhs vector load "
          "results from the selected typed body");
    if (slice.selectMask != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV compare/select EmitC route requires "
          "tcrv_rvv.i32_select to consume the typed mask produced by "
          "tcrv_rvv.i32_cmp_eq");
    if (!isLoadedDataValue(slice.selectTrueValue) ||
        !isLoadedDataValue(slice.selectFalseValue))
      return makeRVVEmitCRouteProviderError(
          "bounded RVV compare/select EmitC route requires "
          "tcrv_rvv.i32_select true/false operands to be explicit lhs/rhs "
          "vector load results from the selected typed body");
    if (slice.store.getValue() != slice.selectResult)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV compare/select EmitC route requires tcrv_rvv.i32_store "
          "to consume the select result");
  } else {
    if (slice.arithmeticLhs != slice.lhsLoad.getLoaded() ||
        slice.arithmeticRhs != rhsValue)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("bounded RVV EmitC route requires tcrv_rvv.i32_") +
          operationMnemonic +
          " to consume lhs load and explicit rhs vector or broadcast results");
    if (slice.store.getValue() != slice.arithmeticResult)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV EmitC route requires tcrv_rvv.i32_store to consume the "
          "arithmetic result");
  }

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

  llvm::StringRef sourceRole = lowerable.getTCRVEmitCLowerableSourceRole();
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
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
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

unsigned getRVVCanonicalRoleOrder(RVVSelectedBodyRouteSlice &slice,
                                  mlir::Operation *op) {
  auto lhsABI =
      slice.lhsLoad.getBuffer().getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  mlir::Value rhsBuffer = slice.rhsBroadcastLoad
                              ? slice.rhsBroadcastLoad.getBuffer()
                              : slice.rhsLoad.getBuffer();
  auto rhsABI = rhsBuffer.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  auto outABI =
      slice.store.getBuffer().getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
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
  if (slice.compareOp && op == slice.compareOp.getOperation())
    return 8;
  if (slice.arithmeticKind == RVVSelectedBodyOperationKind::CmpSelect &&
      op == slice.arithmeticOp)
    return 9;
  if (op == slice.arithmeticOp)
    return 8;
  if (op == slice.store.getOperation())
    return slice.arithmeticKind == RVVSelectedBodyOperationKind::CmpSelect ? 10
                                                                           : 9;
  return 10;
}

RVVOrderedRoleOperations
collectRVVRoleOperationsInBodyOrder(tcrv::exec::VariantOp variant,
                                    RVVSelectedBodyRouteSlice &slice) {
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
    RVVSelectedBodyRouteSlice &slice,
    const VariantEmitCLowerableRequest &request,
    const RVVSelectedBodyConstructionRoute &constructionRoute) {
  auto lhsABI =
      slice.lhsLoad.getBuffer().getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  mlir::Value rhsBuffer = slice.rhsBroadcastLoad
                              ? slice.rhsBroadcastLoad.getBuffer()
                              : slice.rhsLoad.getBuffer();
  auto rhsABI = rhsBuffer.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  auto outABI =
      slice.store.getBuffer().getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  auto nABI =
      slice.setvl.getAvl().getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (!lhsABI || !rhsABI || !outABI || !nABI)
    return makeRVVEmitCRouteProviderError(
        "selected RVV construction role sequence requires runtime ABI values "
        "to be explicit tcrv_rvv.runtime_abi_value ops");

  mlir::ArrayAttr
    requires
  = request.getVariant()->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!requires || requires.empty())
    return makeRVVEmitCRouteProviderError(
        "selected RVV construction role sequence requires non-empty selected "
        "variant requires metadata");

  RVVOrderedRoleOperations ordered =
      collectRVVRoleOperationsInBodyOrder(request.getVariant(), slice);
  llvm::StringRef rhsSourceOperationName = slice.rhsBroadcastLoad
                                               ? "tcrv_rvv.i32_broadcast_load"
                                               : "tcrv_rvv.i32_load";
  return verifyRVVSelectedBodySelectedRoleSequence(
      ordered.operations, ordered.constructionOrders,
      request.getVariant().getSymName(),
      stringifyVariantEmissionRole(request.getRole()),
      constructionRoute.typedComputeOpName, rhsSourceOperationName,
      "selected RVV EmitC route");
}

llvm::Expected<RVVSelectedBodyRouteAnalysis>
analyzeRVVSelectedBodyRoute(const VariantEmitCLowerableRequest &request) {
  if (!request.getVariant())
    return makeRVVEmitCRouteProviderError(
        "EmitC route construction requires a materialized tcrv.exec.variant");
  if (!request.getKernel())
    return makeRVVEmitCRouteProviderError(
        "EmitC route construction requires an enclosing tcrv.exec.kernel");

  if (llvm::Error error = requireRVVVariantLegality(request.getVariant()))
    return std::move(error);
  if (llvm::Error error = verifyRVVConstructionProtocolReady())
    return std::move(error);

  llvm::Expected<RVVSelectedBodyRouteSlice> slice =
      collectRVVSelectedBodyRouteSlice(request.getVariant());
  if (!slice)
    return slice.takeError();

  tcrv::rvv::RVVCompileTimeConfig config =
      tcrv::rvv::getRVVSetVLCompileTimeConfig(slice->setvl);
  const auto &configContract = tcrv::rvv::getRVVSelectedBodyConfigVLContract();

  RVVSelectedBodyRouteAnalysis analysis;
  analysis.slice = std::move(*slice);
  analysis.description.operation = analysis.slice.arithmeticKind;
  analysis.description.memoryForm =
      analysis.slice.rhsBroadcastLoad
          ? RVVSelectedBodyMemoryForm::RHSBroadcastLoad
          : RVVSelectedBodyMemoryForm::VectorRHSLoad;
  analysis.description.sew = config.sew;
  analysis.description.lmul = config.lmul;
  analysis.description.tailPolicy =
      stringifyRVVTailPolicy(config.policy.getTail());
  analysis.description.maskPolicy =
      stringifyRVVMaskPolicy(config.policy.getMask());
  analysis.description.configContractID = configContract.configContractID;
  analysis.description.runtimeVLContractID = configContract.runtimeVLContractID;
  analysis.description.runtimeAVLASource = configContract.runtimeAVLASource;
  analysis.description.runtimeABIOrder = configContract.runtimeABIOrder;
  analysis.description.vlDefOpName = configContract.vlDefOpName;
  analysis.description.vlScopeOpName = configContract.vlScopeOpName;
  analysis.description.vlUses = configContract.vlUses;
  analysis.description.emitCLoopKind = configContract.emitCLoopKind;
  analysis.description.emitCLoopInductionName =
      configContract.emitCLoopInductionName;
  analysis.description.emitCFullChunkVLName =
      configContract.emitCFullChunkVLName;
  analysis.description.emitCLoopVLName =
      tcrv::rvv::getRVVSelectedBodyEmitCLoopVLName();
  analysis.description.remainingAVLMetadata =
      configContract.remainingAVLMetadata;
  analysis.description.pointerAdvanceMetadata =
      configContract.pointerAdvanceMetadata;
  analysis.description.boundedSlice = configContract.boundedSlice;
  analysis.description.multiVL = configContract.multiVL;
  analysis.description.boundaryOpName = kRVVSelectedBodyLoweringBoundaryOpName;
  analysis.description.targetArtifactRouteID =
      getRVVSelectedBodyTargetArtifactRouteID();
  analysis.description.targetArtifactKind =
      getRVVSelectedBodyTargetArtifactKind();
  analysis.description.runtimeABIParameters.push_back(analysis.slice.lhsABI);
  analysis.description.runtimeABIParameters.push_back(analysis.slice.rhsABI);
  analysis.description.runtimeABIParameters.push_back(analysis.slice.outABI);
  analysis.description.runtimeABIParameters.push_back(
      analysis.slice.runtimeElementCountABI);

  llvm::Expected<const RVVSelectedBodySpecializationMapping *>
      specializationMapping =
          lookupRVVSelectedBodySpecializationMapping(analysis.description);
  if (!specializationMapping)
    return specializationMapping.takeError();
  analysis.specializationMapping = *specializationMapping;

  llvm::Expected<const RVVSelectedBodyConstructionRoute *> constructionRoute =
      lookupRVVSelectedBodyConstructionRouteByOperationMnemonic(
          analysis.specializationMapping->operationMnemonic);
  if (!constructionRoute)
    return constructionRoute.takeError();
  analysis.constructionRoute = *constructionRoute;

  analysis.description.typedComputeOpName =
      analysis.constructionRoute->typedComputeOpName;
  analysis.description.emitCRouteID = analysis.constructionRoute->emitCRouteID;
  analysis.description.runtimeABIName =
      analysis.constructionRoute->runtimeABIName;
  analysis.description.runtimeABIContractName =
      analysis.constructionRoute->runtimeABIContractName;
  analysis.description.vlCType = analysis.specializationMapping->vlCType;
  analysis.description.vectorTypeName =
      analysis.specializationMapping->vectorTypeName;
  analysis.description.maskTypeName =
      analysis.specializationMapping->maskTypeName;
  analysis.description.vectorCType =
      analysis.specializationMapping->vectorCType;
  analysis.description.maskCType = analysis.specializationMapping->maskCType;
  analysis.description.setVLIntrinsic =
      analysis.specializationMapping->setVLIntrinsic;
  analysis.description.vectorLoadIntrinsic =
      analysis.specializationMapping->vectorLoadIntrinsic;
  analysis.description.rhsBroadcastIntrinsic =
      analysis.specializationMapping->rhsBroadcastIntrinsic;
  analysis.description.storeIntrinsic =
      analysis.specializationMapping->storeIntrinsic;
  analysis.description.intrinsic =
      analysis.specializationMapping->computeIntrinsic;
  analysis.description.compareIntrinsic =
      analysis.specializationMapping->compareIntrinsic;
  analysis.description.resultName = analysis.specializationMapping->resultName;
  analysis.description.maskName = analysis.specializationMapping->maskName;

  if (llvm::Error error = validateRVVSelectedBodyRuntimeABIParameters(
          analysis.slice, *analysis.constructionRoute,
          analysis.slice.runtimeElementCountABI, analysis.slice.outABI))
    return error;
  if (analysis.slice.arithmeticOp->getName().getStringRef() !=
      analysis.constructionRoute->typedComputeOpName)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected typed RVV body route expected compute op '") +
        analysis.constructionRoute->typedComputeOpName +
        "' from the selected-body specialization mapping");
  if (llvm::Error error = verifyRVVSelectedBodyConstructionRouteMapping(
          analysis.specializationMapping->operationMnemonic,
          analysis.constructionRoute->typedComputeOpName,
          analysis.constructionRoute->emitCRouteID,
          analysis.constructionRoute->runtimeABIName))
    return std::move(error);
  if (llvm::Error error = verifySelectedRVVRoleSequence(
          analysis.slice, request, *analysis.constructionRoute))
    return std::move(error);
  if (llvm::Error error = verifyRVVSelectedBodyEmitCRouteDescription(
          analysis.description, "selected RVV EmitC route description"))
    return std::move(error);
  return analysis;
}

} // namespace

llvm::ArrayRef<RVVSelectedBodyOperationKind>
getRVVSelectedBodyOperationKinds() {
  return kRVVSelectedBodyOperationKinds;
}

llvm::StringRef
stringifyRVVSelectedBodyOperationKind(RVVSelectedBodyOperationKind op) {
  return getRVVSelectedBodySpecializationMappingByOperation(op)
      .operationMnemonic;
}

llvm::StringRef
stringifyRVVSelectedBodyMemoryForm(RVVSelectedBodyMemoryForm form) {
  switch (form) {
  case RVVSelectedBodyMemoryForm::VectorRHSLoad:
    return "vector-rhs-load";
  case RVVSelectedBodyMemoryForm::RHSBroadcastLoad:
    return "rhs-broadcast-load";
  }
  llvm_unreachable("unknown RVV selected-body memory form");
}

llvm::StringRef
getRVVSelectedBodyEmitCRouteID(RVVSelectedBodyOperationKind op) {
  return getRVVSelectedBodyConstructionRouteOrDie(op).emitCRouteID;
}

llvm::StringRef getRVVSelectedBodyEmissionKind() {
  return kRVVSelectedBodyEmissionKind;
}

llvm::StringRef getRVVSelectedBodyLoweringBoundaryOpName() {
  return kRVVSelectedBodyLoweringBoundaryOpName;
}

llvm::StringRef getRVVSelectedBodyRuntimeABIKind() {
  return kRVVSelectedBodyRuntimeABIKind;
}

llvm::StringRef
getRVVSelectedBodyRuntimeABIName(RVVSelectedBodyOperationKind op) {
  return getRVVSelectedBodyConstructionRouteOrDie(op).runtimeABIName;
}

llvm::StringRef getRVVSelectedBodyRuntimeGlueRole() {
  return kRVVSelectedBodyRuntimeGlueRole;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVSelectedBodyRuntimeABIParameters() {
  return tcrv::rvv::getRVVSelectedBodyRuntimeABIParameters();
}

RVVSelectedBodyConstructionMetadataFacts
getRVVSelectedBodyConstructionMetadataFacts(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  RVVSelectedBodyConstructionMetadataFacts facts;
  facts.operationMnemonic =
      stringifyRVVSelectedBodyOperationKind(description.operation);
  facts.typedComputeOpName = description.typedComputeOpName;
  facts.emitCRouteID = description.emitCRouteID;
  facts.targetArtifactRouteID = description.targetArtifactRouteID;
  facts.targetArtifactKind = description.targetArtifactKind;
  facts.runtimeABIName = description.runtimeABIName;
  facts.runtimeABIContractName = description.runtimeABIContractName;
  facts.runtimeABIParameters = description.runtimeABIParameters;
  return facts;
}

llvm::Error verifyRVVSelectedBodyEmitCRouteDescription(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (context.trim().empty())
    return makeRVVEmitCRouteProviderError(
        "selected-body route description verification requires a non-empty "
        "context");

  llvm::Expected<const RVVSelectedBodySpecializationMapping *>
      specialization = lookupRVVSelectedBodySpecializationMapping(description);
  if (!specialization)
    return specialization.takeError();
  const RVVSelectedBodySpecializationMapping &mapping = **specialization;

  llvm::Expected<const RVVSelectedBodyConstructionRoute *> route =
      lookupRVVSelectedBodyConstructionRouteByOperationMnemonic(
          mapping.operationMnemonic);
  if (!route)
    return route.takeError();
  const RVVSelectedBodyConstructionRoute &constructionRoute = **route;

  if (llvm::Error error = requireRouteDescriptionField(
          context, "typed compute op", description.typedComputeOpName,
          constructionRoute.typedComputeOpName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "EmitC route id", description.emitCRouteID,
          constructionRoute.emitCRouteID))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "target artifact route id",
          description.targetArtifactRouteID,
          getRVVSelectedBodyTargetArtifactRouteID()))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "target artifact kind", description.targetArtifactKind,
          getRVVSelectedBodyTargetArtifactKind()))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "runtime ABI name", description.runtimeABIName,
          constructionRoute.runtimeABIName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "runtime ABI contract", description.runtimeABIContractName,
          constructionRoute.runtimeABIContractName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "lowering boundary op", description.boundaryOpName,
          kRVVSelectedBodyLoweringBoundaryOpName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "VL C type", description.vlCType, mapping.vlCType))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "vector type", description.vectorTypeName,
          mapping.vectorTypeName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "mask type", description.maskTypeName, mapping.maskTypeName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "vector C type", description.vectorCType,
          mapping.vectorCType))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "mask C type", description.maskCType, mapping.maskCType))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "setvl intrinsic", description.setVLIntrinsic,
          mapping.setVLIntrinsic))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "vector-load intrinsic", description.vectorLoadIntrinsic,
          mapping.vectorLoadIntrinsic))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "store intrinsic", description.storeIntrinsic,
          mapping.storeIntrinsic))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "compute intrinsic", description.intrinsic,
          mapping.computeIntrinsic))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "compare intrinsic", description.compareIntrinsic,
          mapping.compareIntrinsic))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "result value name", description.resultName,
          mapping.resultName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "mask value name", description.maskName, mapping.maskName))
    return error;
  if (description.memoryForm == RVVSelectedBodyMemoryForm::RHSBroadcastLoad)
    if (llvm::Error error = requireRouteDescriptionText(
            context, "RHS broadcast intrinsic",
            description.rhsBroadcastIntrinsic))
      return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "RHS broadcast intrinsic",
          description.rhsBroadcastIntrinsic, mapping.rhsBroadcastIntrinsic))
    return error;

  if (llvm::Error error = verifyRVVSelectedBodyConstructionRuntimeABIParameters(
          description.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));

  return llvm::Error::success();
}

llvm::SmallVector<support::ArtifactMetadataEntry, 16>
getRVVSelectedBodyConfigArtifactMetadata(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  llvm::SmallVector<support::ArtifactMetadataEntry, 16> metadata;
  metadata.push_back(
      {"tcrv_rvv.config_contract", description.configContractID});
  metadata.push_back({"tcrv_rvv.sew", llvm::Twine(description.sew).str()});
  metadata.push_back({"tcrv_rvv.lmul", description.lmul});
  metadata.push_back({"tcrv_rvv.tail_policy", description.tailPolicy});
  metadata.push_back({"tcrv_rvv.mask_policy", description.maskPolicy});
  metadata.push_back(
      {"tcrv_rvv.runtime_vl_contract", description.runtimeVLContractID});
  metadata.push_back(
      {"tcrv_rvv.runtime_avl_source", description.runtimeAVLASource});
  metadata.push_back({"tcrv_rvv.vl_def", description.vlDefOpName});
  metadata.push_back({"tcrv_rvv.vl_scope", description.vlScopeOpName});
  metadata.push_back({"tcrv_rvv.vl_uses", description.vlUses});
  metadata.push_back(
      {"tcrv_rvv.runtime_abi_order", description.runtimeABIOrder});
  metadata.push_back({"tcrv_rvv.runtime_avl_abi_parameter",
                      tcrv::rvv::getRVVSelectedBodyRuntimeAVLParameterName()});
  metadata.push_back({"tcrv_rvv.emitc_loop", description.emitCLoopKind});
  metadata.push_back(
      {"tcrv_rvv.loop_induction", description.emitCLoopInductionName});
  metadata.push_back({"tcrv_rvv.loop_step", description.emitCFullChunkVLName});
  metadata.push_back(
      {"tcrv_rvv.remaining_avl", description.remainingAVLMetadata});
  metadata.push_back(
      {"tcrv_rvv.pointer_advance", description.pointerAdvanceMetadata});
  metadata.push_back({"tcrv_rvv.bounded_slice", description.boundedSlice});
  metadata.push_back({"tcrv_rvv.multi_vl", description.multiVL});
  return metadata;
}

static llvm::Error buildRVVSelectedBodyEmitCLowerableRouteFromAnalysis(
    RVVSelectedBodyRouteAnalysis &analysis,
    conversion::emitc::TCRVEmitCLowerableRoute &out) {
  RVVSelectedBodyRouteSlice *slice = &analysis.slice;
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;

  conversion::emitc::TCRVEmitCLowerableRoute route(
      analysis.description.emitCRouteID,
      "extension-family-ops-to-emitc-call-opaque");
  route.addHeader("stddef.h");
  route.addHeader("stdint.h");
  route.addHeader("riscv_vector.h");
  route.addTypeMapping("!tcrv_rvv.vl", description.vlCType);
  route.addTypeMapping(description.vectorTypeName, description.vectorCType);
  route.addTypeMapping(description.maskTypeName, description.maskCType);
  const support::RuntimeABIParameter runtimeABIParameters[] = {
      slice->lhsABI, slice->rhsABI, slice->outABI,
      slice->runtimeElementCountABI};
  for (const support::RuntimeABIParameter &parameter : runtimeABIParameters)
    route.addABIValueMapping(parameter, parameter.cName);

  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> withVLSource =
      getEmitCSourceProvenance(slice->withVL.getOperation(), "scope");
  if (!withVLSource)
    return withVLSource.takeError();
  route.addSourceOpProvenance(std::move(*withVLSource));

  using conversion::emitc::TCRVEmitCCallOpaqueOperand;
  using conversion::emitc::TCRVEmitCCallOpaqueResult;
  if (llvm::Error error = addCallStepFromSource(
          route, slice->setvl.getOperation(), "configure",
          description.setVLIntrinsic,
          {TCRVEmitCCallOpaqueOperand{slice->runtimeElementCountABI.cName,
                                      slice->runtimeElementCountABI.cType}},
          TCRVEmitCCallOpaqueResult{description.emitCFullChunkVLName.str(),
                                    description.vlCType.str()}))
    return error;

  conversion::emitc::TCRVEmitCForLoop loop;
  llvm::StringRef inductionName = description.emitCLoopInductionName;
  llvm::StringRef fullChunkVLName = description.emitCFullChunkVLName;
  llvm::StringRef loopVLName = description.emitCLoopVLName;
  loop.inductionVarName = inductionName.str();
  loop.lowerBound = TCRVEmitCCallOpaqueOperand{"0", description.vlCType.str()};
  loop.upperBound = TCRVEmitCCallOpaqueOperand{
      slice->runtimeElementCountABI.cName, slice->runtimeElementCountABI.cType};
  loop.step = TCRVEmitCCallOpaqueOperand{fullChunkVLName.str(),
                                         description.vlCType.str()};

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
          slice->setvl.getOperation(), "configure", description.setVLIntrinsic,
          {TCRVEmitCCallOpaqueOperand{
              tcrv::rvv::getRVVSelectedBodyEmitCRemainingAVLExpression(
                  slice->runtimeElementCountABI.cName, inductionName),
              description.vlCType.str()}},
          TCRVEmitCCallOpaqueResult{loopVLName.str(),
                                    description.vlCType.str()}))
    return error;
  if (llvm::Error error = addLoopStep(
          slice->lhsLoad.getOperation(), "load",
          description.vectorLoadIntrinsic,
          {TCRVEmitCCallOpaqueOperand{
               (llvm::StringRef(slice->lhsABI.cName) + " + " + inductionName)
                   .str(),
               slice->lhsABI.cType},
           TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                      description.vlCType.str()}},
          TCRVEmitCCallOpaqueResult{"lhs_vec", description.vectorCType.str()}))
    return error;
  if (slice->rhsBroadcastLoad) {
    if (llvm::Error error = addLoopStep(
            slice->rhsBroadcastLoad.getOperation(), "load",
            description.rhsBroadcastIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->rhsABI.cName) + "[0]").str(),
                 "int32_t"},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"rhs_vec",
                                      description.vectorCType.str()}))
      return error;
  } else {
    if (llvm::Error error = addLoopStep(
            slice->rhsLoad.getOperation(), "load",
            description.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->rhsABI.cName) + " + " + inductionName)
                     .str(),
                 slice->rhsABI.cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"rhs_vec",
                                      description.vectorCType.str()}))
      return error;
  }
  if (slice->arithmeticKind == RVVSelectedBodyOperationKind::CmpSelect) {
    auto loadedVectorOperand =
        [&](mlir::Value value,
            llvm::StringRef context) -> llvm::Expected<TCRVEmitCCallOpaqueOperand> {
      if (value == slice->lhsLoad.getLoaded())
        return TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                          description.vectorCType.str()};
      if (value == slice->rhsLoad.getLoaded())
        return TCRVEmitCCallOpaqueOperand{"rhs_vec",
                                          description.vectorCType.str()};
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("bounded RVV compare/select EmitC route cannot map ") +
          context +
          " because it is not one of the explicit lhs/rhs vector load "
          "results validated from the selected typed body");
    };

    llvm::Expected<TCRVEmitCCallOpaqueOperand> compareLhs =
        loadedVectorOperand(slice->compareLhs, "compare lhs operand");
    if (!compareLhs)
      return compareLhs.takeError();
    llvm::Expected<TCRVEmitCCallOpaqueOperand> compareRhs =
        loadedVectorOperand(slice->compareRhs, "compare rhs operand");
    if (!compareRhs)
      return compareRhs.takeError();
    llvm::Expected<TCRVEmitCCallOpaqueOperand> selectTrue =
        loadedVectorOperand(slice->selectTrueValue, "select true operand");
    if (!selectTrue)
      return selectTrue.takeError();
    llvm::Expected<TCRVEmitCCallOpaqueOperand> selectFalse =
        loadedVectorOperand(slice->selectFalseValue, "select false operand");
    if (!selectFalse)
      return selectFalse.takeError();

    if (llvm::Error error = addLoopStep(
            slice->compareOp.getOperation(), "compute",
            description.compareIntrinsic,
            {*compareLhs, *compareRhs,
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.maskName.str(),
                                      description.maskCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", description.intrinsic,
            {*selectFalse, *selectTrue,
             TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        description.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      description.vectorCType.str()}))
      return error;
  } else {
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", description.intrinsic,
            {TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"rhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      description.vectorCType.str()}))
      return error;
  }
  if (llvm::Error error = addLoopStep(
          slice->store.getOperation(), "store", description.storeIntrinsic,
          {TCRVEmitCCallOpaqueOperand{
               (llvm::StringRef(slice->outABI.cName) + " + " + inductionName)
                   .str(),
               slice->outABI.cType},
           TCRVEmitCCallOpaqueOperand{description.resultName.str(),
                                      description.vectorCType.str()},
           TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                      description.vlCType.str()}}))
    return error;

  route.addForLoop(std::move(loop));

  out = std::move(route);
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyEmitCRouteDescription>
describeRVVSelectedBodyEmitCRoute(
    const VariantEmitCLowerableRequest &request,
    conversion::emitc::TCRVEmitCLowerableRoute *verifiedRoute) {
  llvm::Expected<RVVSelectedBodyRouteAnalysis> analysis =
      analyzeRVVSelectedBodyRoute(request);
  if (!analysis)
    return analysis.takeError();

  if (verifiedRoute)
    if (llvm::Error error = buildRVVSelectedBodyEmitCLowerableRouteFromAnalysis(
            *analysis, *verifiedRoute))
      return std::move(error);

  return analysis->description;
}

llvm::Error buildRVVSelectedBodyEmitCLowerableRoute(
    const VariantEmitCLowerableRequest &request,
    conversion::emitc::TCRVEmitCLowerableRoute &out) {
  llvm::Expected<RVVSelectedBodyRouteAnalysis> analysis =
      analyzeRVVSelectedBodyRoute(request);
  if (!analysis)
    return analysis.takeError();
  return buildRVVSelectedBodyEmitCLowerableRouteFromAnalysis(*analysis, out);
}

} // namespace tianchenrv::plugin::rvv
