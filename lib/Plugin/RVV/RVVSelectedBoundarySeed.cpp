#include "TianChenRV/Plugin/RVV/RVVSelectedBoundarySeed.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace tianchenrv::plugin::rvv {
namespace {

using tianchenrv::support::RuntimeABIParameterOwnership;
using tianchenrv::support::RuntimeABIParameterRole;

constexpr llvm::StringLiteral kSeedAttrName("tcrv_rvv.lowering_seed");
constexpr llvm::StringLiteral kSeedAttrValue("i32m1_add");
constexpr llvm::StringLiteral kRVVCapabilitySymbol("rvv");
constexpr llvm::StringLiteral kSelectedMessage(
    "bounded MLIR vector i32 add seed selected RVV i32m1 boundary");

constexpr std::int64_t kSourceVectorLaneCount = 4;

bool isRank1DynamicI32MemRef(mlir::Type type) {
  auto memrefType = llvm::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType || memrefType.getRank() != 1)
    return false;
  if (!memrefType.isDynamicDim(0))
    return false;
  auto elementType = llvm::dyn_cast<mlir::IntegerType>(
      memrefType.getElementType());
  return elementType && elementType.getWidth() == 32;
}

bool isBoundedSourceVectorI32(mlir::Type type) {
  auto vectorType = llvm::dyn_cast<mlir::VectorType>(type);
  if (!vectorType || vectorType.getRank() != 1)
    return false;
  if (vectorType.getShape().front() != kSourceVectorLaneCount)
    return false;
  auto elementType =
      llvm::dyn_cast<mlir::IntegerType>(vectorType.getElementType());
  return elementType && elementType.getWidth() == 32;
}

std::optional<std::int64_t> getConstantIndexValue(mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantIndexOp>())
    return constant.value();
  return std::nullopt;
}

bool isFuncReturnWithoutOperands(mlir::Operation *op) {
  auto returnOp = llvm::dyn_cast<mlir::func::ReturnOp>(op);
  return returnOp && returnOp.getNumOperands() == 0;
}

bool usesOnlyIndex(mlir::Operation::operand_range indices,
                   mlir::Value expectedIndex) {
  return llvm::hasSingleElement(indices) && *indices.begin() == expectedIndex;
}

mlir::LogicalResult failSeed(mlir::Operation *op, llvm::StringRef message) {
  op->emitError() << "bounded RVV i32m1 selected-boundary seed failed: "
                  << message;
  return mlir::failure();
}

mlir::LogicalResult requireSourceOnlyModule(mlir::ModuleOp module) {
  mlir::Operation *staleOp = nullptr;
  module.walk([&](mlir::Operation *op) {
    if (staleOp)
      return;
    if (op->getName().getDialectNamespace() == "tcrv" ||
        op->getName().getDialectNamespace() == "tcrv_rvv")
      staleOp = op;
  });
  if (!staleOp)
    return mlir::success();

  return failSeed(staleOp,
                  "source seed pass requires source-only MLIR input; "
                  "pre-existing tcrv.exec/tcrv_rvv selected-boundary or "
                  "unselected variant residue is not accepted");
}

mlir::LogicalResult matchBoundedI32AddSeed(mlir::func::FuncOp func) {
  auto seedAttr = func->getAttrOfType<mlir::StringAttr>(kSeedAttrName);
  if (!seedAttr)
    return mlir::success();
  if (seedAttr.getValue() != kSeedAttrValue)
    return failSeed(func, "unsupported RVV lowering seed attribute value");

  mlir::FunctionType functionType = func.getFunctionType();
  if (functionType.getNumResults() != 0)
    return failSeed(func, "source function must not return values");
  if (functionType.getNumInputs() != 4)
    return failSeed(func,
                    "source function must expose exactly four runtime ABI "
                    "operands: lhs, rhs, out, and n");
  for (unsigned index = 0; index < 3; ++index) {
    if (!isRank1DynamicI32MemRef(functionType.getInput(index)))
      return failSeed(func,
                      "lhs, rhs, and out operands must be memref<?xi32>");
  }
  if (!functionType.getInput(3).isIndex())
    return failSeed(func, "runtime n operand must have index type");

  if (!llvm::hasSingleElement(func.getBody()))
    return failSeed(func, "source function must have one block");
  mlir::Block &entry = func.getBody().front();

  mlir::scf::ForOp loop;
  unsigned returnCount = 0;
  for (mlir::Operation &op : entry) {
    if (llvm::isa<mlir::arith::ConstantIndexOp>(op))
      continue;
    if (isFuncReturnWithoutOperands(&op)) {
      ++returnCount;
      continue;
    }
    if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(op)) {
      if (loop)
        return failSeed(func, "source function must contain one scf.for");
      loop = forOp;
      continue;
    }
    return failSeed(&op,
                    "source function may contain only index constants, one "
                    "scf.for, and one empty return");
  }
  if (!loop)
    return failSeed(func, "source function must contain one scf.for");
  if (returnCount != 1)
    return failSeed(func, "source function must contain one empty return");

  std::optional<std::int64_t> lower =
      getConstantIndexValue(loop.getLowerBound());
  if (!lower || *lower != 0)
    return failSeed(loop, "scf.for lower bound must be constant index 0");
  if (loop.getUpperBound() != entry.getArgument(3))
    return failSeed(loop, "scf.for upper bound must be the runtime n operand");
  std::optional<std::int64_t> step = getConstantIndexValue(loop.getStep());
  if (!step || *step != kSourceVectorLaneCount)
    return failSeed(loop, "scf.for step must match the bounded vector chunk");

  mlir::Block &loopBody = loop.getRegion().front();
  llvm::SmallVector<mlir::Operation *, 4> bodyOps;
  for (mlir::Operation &op : loopBody) {
    if (llvm::isa<mlir::scf::YieldOp>(op))
      continue;
    bodyOps.push_back(&op);
  }
  if (bodyOps.size() != 4)
    return failSeed(loop, "scf.for body must contain exactly two vector.load "
                          "ops, one arith.addi, and one vector.store");

  auto lhsLoad = llvm::dyn_cast<mlir::vector::LoadOp>(bodyOps[0]);
  auto rhsLoad = llvm::dyn_cast<mlir::vector::LoadOp>(bodyOps[1]);
  auto add = llvm::dyn_cast<mlir::arith::AddIOp>(bodyOps[2]);
  auto store = llvm::dyn_cast<mlir::vector::StoreOp>(bodyOps[3]);
  if (!lhsLoad || !rhsLoad || !add || !store)
    return failSeed(loop, "scf.for body operation order must be "
                          "vector.load, vector.load, arith.addi, vector.store");

  mlir::Value iv = loop.getInductionVar();
  if (lhsLoad.getBase() != entry.getArgument(0) ||
      !usesOnlyIndex(lhsLoad.getIndices(), iv))
    return failSeed(lhsLoad, "first vector.load must read lhs at the loop iv");
  if (rhsLoad.getBase() != entry.getArgument(1) ||
      !usesOnlyIndex(rhsLoad.getIndices(), iv))
    return failSeed(rhsLoad, "second vector.load must read rhs at the loop iv");
  if (!isBoundedSourceVectorI32(lhsLoad.getResult().getType()) ||
      !isBoundedSourceVectorI32(rhsLoad.getResult().getType()))
    return failSeed(loop,
                    "source vector loads must produce vector<4xi32>");

  if (add.getLhs() != lhsLoad.getResult() || add.getRhs() != rhsLoad.getResult())
    return failSeed(add, "arith.addi must consume the two source vector loads");
  if (!isBoundedSourceVectorI32(add.getResult().getType()))
    return failSeed(add, "arith.addi must produce vector<4xi32>");

  if (store.getBase() != entry.getArgument(2) ||
      !usesOnlyIndex(store.getIndices(), iv))
    return failSeed(store, "vector.store must write out at the loop iv");
  if (store.getValueToStore() != add.getResult())
    return failSeed(store, "vector.store must store the arith.addi result");

  return mlir::success();
}

mlir::FlatSymbolRefAttr symbolRef(mlir::OpBuilder &builder,
                                  llvm::StringRef symbol) {
  return mlir::FlatSymbolRefAttr::get(builder.getContext(), symbol);
}

mlir::Operation *createRuntimeABIValue(mlir::OpBuilder &builder,
                                       mlir::Location loc,
                                       llvm::StringRef role,
                                       llvm::StringRef cName,
                                       llvm::StringRef cType,
                                       mlir::Type resultType) {
  mlir::OperationState state(loc, "tcrv_rvv.runtime_abi_value");
  state.addTypes(resultType);
  state.addAttribute("role", builder.getStringAttr(role));
  state.addAttribute("c_name", builder.getStringAttr(cName));
  state.addAttribute("c_type", builder.getStringAttr(cType));
  state.addAttribute(
      "ownership",
      builder.getStringAttr(
          tianchenrv::support::stringifyRuntimeABIParameterOwnership(
              RuntimeABIParameterOwnership::TargetExportABIOwned)));
  return builder.create(state);
}

mlir::Operation *createSetVL(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value nValue,
                             tcrv::rvv::PolicyAttr policy) {
  mlir::OperationState state(loc, "tcrv_rvv.setvl");
  state.addOperands(nValue);
  state.addTypes(tcrv::rvv::VLType::get(builder.getContext()));
  state.addAttribute(
      "sew", builder.getI64IntegerAttr(tcrv::rvv::getRVVFirstSliceSEWBits()));
  state.addAttribute("lmul",
                     builder.getStringAttr(tcrv::rvv::getRVVI32M1LMUL()));
  state.addAttribute("policy", policy);
  return builder.create(state);
}

mlir::Operation *createWithVL(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value vlValue,
                              tcrv::rvv::PolicyAttr policy) {
  mlir::OperationState state(loc, "tcrv_rvv.with_vl");
  state.addOperands(vlValue);
  state.addAttribute(
      "sew", builder.getI64IntegerAttr(tcrv::rvv::getRVVFirstSliceSEWBits()));
  state.addAttribute("lmul",
                     builder.getStringAttr(tcrv::rvv::getRVVI32M1LMUL()));
  state.addAttribute("policy", policy);
  state.addRegion();
  mlir::Operation *operation = builder.create(state);
  operation->getRegion(0).emplaceBlock();
  return operation;
}

mlir::Operation *createI32Load(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value buffer, mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.i32_load");
  state.addOperands({buffer, vl});
  state.addTypes(tcrv::rvv::I32M1VectorType::get(builder.getContext()));
  return builder.create(state);
}

mlir::Operation *createI32Add(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value lhs, mlir::Value rhs,
                              mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.i32_add");
  state.addOperands({lhs, rhs, vl});
  state.addTypes(tcrv::rvv::I32M1VectorType::get(builder.getContext()));
  return builder.create(state);
}

void createI32Store(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value out, mlir::Value value, mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.i32_store");
  state.addOperands({out, value, vl});
  (void)builder.create(state);
}

void materializeSeedKernel(mlir::OpBuilder &builder, mlir::func::FuncOp func) {
  mlir::Location loc = func.getLoc();
  std::string kernelName = (func.getSymName() + "_kernel").str();
  std::string variantName = (func.getSymName() + "_rvv_i32_add").str();

  mlir::OperationState kernelState(loc, "tcrv.exec.kernel");
  kernelState.addAttribute("sym_name", builder.getStringAttr(kernelName));
  kernelState.addRegion();
  auto kernel =
      llvm::cast<tcrv::exec::KernelOp>(builder.create(kernelState));
  kernel.getBody().emplaceBlock();

  mlir::OpBuilder::InsertionGuard kernelGuard(builder);
  builder.setInsertionPointToStart(&kernel.getBody().front());

  mlir::OperationState capabilityState(loc, "tcrv.exec.capability");
  capabilityState.addAttribute("sym_name",
                               builder.getStringAttr(kRVVCapabilitySymbol));
  capabilityState.addAttribute("id", builder.getStringAttr(getRVVCapabilityID()));
  capabilityState.addAttribute("kind",
                               builder.getStringAttr(getRVVCapabilityKind()));
  capabilityState.addAttribute("status", builder.getStringAttr("available"));
  (void)builder.create(capabilityState);

  mlir::ArrayAttr requires =
      builder.getArrayAttr({symbolRef(builder, kRVVCapabilitySymbol)});
  auto policy = tcrv::rvv::PolicyAttr::get(
      builder.getContext(), tcrv::rvv::TailPolicy::Agnostic,
      tcrv::rvv::MaskPolicy::Agnostic);

  mlir::OperationState variantState(loc, "tcrv.exec.variant");
  variantState.addAttribute("sym_name", builder.getStringAttr(variantName));
  variantState.addAttribute("origin",
                            builder.getStringAttr(getRVVExtensionPluginName()));
  variantState.addAttribute("requires", requires);
  variantState.addAttribute(getRVVPolicyAttrName(), policy);
  variantState.addRegion();
  auto variant =
      llvm::cast<tcrv::exec::VariantOp>(builder.create(variantState));
  variant.getBody().emplaceBlock();

  {
    mlir::OpBuilder::InsertionGuard variantGuard(builder);
    builder.setInsertionPointToStart(&variant.getBody().front());
    mlir::Type runtimeABIValueType =
        tcrv::rvv::RuntimeABIValueType::get(builder.getContext());
    mlir::Operation *lhs = createRuntimeABIValue(
        builder, loc,
        tianchenrv::support::stringifyRuntimeABIParameterRole(
            RuntimeABIParameterRole::LHSInputBuffer),
        "lhs", "const int32_t *", runtimeABIValueType);
    mlir::Operation *rhs = createRuntimeABIValue(
        builder, loc,
        tianchenrv::support::stringifyRuntimeABIParameterRole(
            RuntimeABIParameterRole::RHSInputBuffer),
        "rhs", "const int32_t *", runtimeABIValueType);
    mlir::Operation *out = createRuntimeABIValue(
        builder, loc,
        tianchenrv::support::stringifyRuntimeABIParameterRole(
            RuntimeABIParameterRole::OutputBuffer),
        "out", "int32_t *", runtimeABIValueType);
    mlir::Operation *n = createRuntimeABIValue(
        builder, loc,
        tianchenrv::support::stringifyRuntimeABIParameterRole(
            RuntimeABIParameterRole::RuntimeElementCount),
        "n", "size_t", builder.getIndexType());
    mlir::Operation *setvl = createSetVL(builder, loc, n->getResult(0), policy);
    mlir::Operation *withVL =
        createWithVL(builder, loc, setvl->getResult(0), policy);

    builder.setInsertionPointToStart(&withVL->getRegion(0).front());
    mlir::Operation *lhsLoad =
        createI32Load(builder, loc, lhs->getResult(0), setvl->getResult(0));
    mlir::Operation *rhsLoad =
        createI32Load(builder, loc, rhs->getResult(0), setvl->getResult(0));
    mlir::Operation *sum = createI32Add(builder, loc, lhsLoad->getResult(0),
                                        rhsLoad->getResult(0),
                                        setvl->getResult(0));
    createI32Store(builder, loc, out->getResult(0), sum->getResult(0),
                   setvl->getResult(0));
  }

  mlir::OperationState diagnosticState(loc, "tcrv.exec.diagnostic");
  diagnosticState.addAttribute("reason",
                               builder.getStringAttr("variant-selected"));
  diagnosticState.addAttribute("message",
                               builder.getStringAttr(kSelectedMessage));
  diagnosticState.addAttribute("severity", builder.getStringAttr("note"));
  diagnosticState.addAttribute("status", builder.getStringAttr("selected"));
  diagnosticState.addAttribute("selection_kind",
                               builder.getStringAttr("fallback-only"));
  diagnosticState.addAttribute("target", symbolRef(builder, variantName));
  (void)builder.create(diagnosticState);
}

class MaterializeRVVI32M1SelectedBoundarySeedPass final
    : public mlir::PassWrapper<
          MaterializeRVVI32M1SelectedBoundarySeedPass,
          mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const final {
    return "tcrv-rvv-materialize-i32m1-selected-boundary-seed";
  }

  llvm::StringRef getDescription() const final {
    return "Materialize one bounded MLIR vector i32 add seed into the RVV "
           "i32m1 selected-boundary form";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::scf::SCFDialect, mlir::vector::VectorDialect,
                    tcrv::exec::TCRVExecDialect, tcrv::rvv::TCRVRVVDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    llvm::SmallVector<mlir::func::FuncOp, 2> seeds;
    module.walk([&](mlir::func::FuncOp func) {
      if (func->hasAttr(kSeedAttrName))
        seeds.push_back(func);
    });
    if (seeds.empty())
      return;

    if (mlir::failed(requireSourceOnlyModule(module))) {
      signalPassFailure();
      return;
    }

    for (mlir::func::FuncOp func : seeds) {
      if (mlir::failed(matchBoundedI32AddSeed(func))) {
        signalPassFailure();
        return;
      }
    }

    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(module.getBody());
    for (mlir::func::FuncOp func : seeds)
      materializeSeedKernel(builder, func);

    for (mlir::func::FuncOp func : seeds)
      func.erase();
  }
};

} // namespace

std::unique_ptr<::mlir::Pass>
createMaterializeRVVI32M1SelectedBoundarySeedPass() {
  return std::make_unique<MaterializeRVVI32M1SelectedBoundarySeedPass>();
}

} // namespace tianchenrv::plugin::rvv
