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
constexpr llvm::StringLiteral kLoweringSeedAttrSuffix(".lowering_seed");
constexpr llvm::StringLiteral kRVVCapabilitySymbol("rvv");
constexpr llvm::StringLiteral kScalarPluginName("scalar-plugin");
constexpr llvm::StringLiteral kScalarFallbackCapabilitySymbol(
    "scalar_fallback");
constexpr llvm::StringLiteral kScalarFallbackCapabilityID("scalar.fallback");
constexpr llvm::StringLiteral kScalarFallbackCapabilityKind("fallback");
constexpr llvm::StringLiteral kScalarFallbackPolicy(
    "portable_scalar_fallback_first_slice");
constexpr llvm::StringLiteral kRVVCasePolicy(
    "source-pattern-selected-rvv-case");
constexpr llvm::StringLiteral kFallbackPolicy(
    "source-pattern-conservative-fallback-envelope");

constexpr std::int64_t kSourceVectorLaneCount = 4;

struct SourceRuntimeABIValue {
  RuntimeABIParameterRole role;
  llvm::StringLiteral cName;
  llvm::StringLiteral cType;
  mlir::BlockArgument sourceArgument;
};

struct BoundedI32AddSourcePattern {
  mlir::func::FuncOp func;
  llvm::SmallVector<SourceRuntimeABIValue, 4> runtimeABIValues;
};

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

mlir::LogicalResult failMaterializer(mlir::Operation *op,
                                     llvm::StringRef message) {
  op->emitError() << "bounded RVV i32m1 selected-boundary materializer failed: "
                  << message;
  return mlir::failure();
}

mlir::FailureOr<BoundedI32AddSourcePattern>
failMaterializerMatch(mlir::Operation *op, llvm::StringRef message) {
  (void)failMaterializer(op, message);
  return mlir::failure();
}

bool hasForeignLoweringSeedAttr(mlir::func::FuncOp func) {
  for (mlir::NamedAttribute attr : func->getAttrs()) {
    llvm::StringRef name = attr.getName().getValue();
    if (name != kSeedAttrName && name.ends_with(kLoweringSeedAttrSuffix))
      return true;
  }
  return false;
}

llvm::SmallVector<SourceRuntimeABIValue, 4>
deriveSourceRuntimeABIValues(mlir::Block &entry) {
  llvm::SmallVector<SourceRuntimeABIValue, 4> values;
  values.push_back({RuntimeABIParameterRole::LHSInputBuffer, "lhs",
                    "const int32_t *", entry.getArgument(0)});
  values.push_back({RuntimeABIParameterRole::RHSInputBuffer, "rhs",
                    "const int32_t *", entry.getArgument(1)});
  values.push_back(
      {RuntimeABIParameterRole::OutputBuffer, "out", "int32_t *",
       entry.getArgument(2)});
  values.push_back(
      {RuntimeABIParameterRole::RuntimeElementCount, "n", "size_t",
       entry.getArgument(3)});
  return values;
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

  return failMaterializer(staleOp,
                          "source materializer requires source-only MLIR "
                          "input; pre-existing tcrv.exec/tcrv_rvv "
                          "selected-boundary or unselected variant residue is "
                          "not accepted");
}

mlir::FailureOr<BoundedI32AddSourcePattern>
matchBoundedI32AddSourcePattern(mlir::func::FuncOp func) {
  if (func->hasAttr(kSeedAttrName))
    return failMaterializerMatch(
        func,
        "stale tcrv_rvv.lowering_seed metadata is not accepted as "
        "source-route authority");

  mlir::FunctionType functionType = func.getFunctionType();
  if (functionType.getNumResults() != 0)
    return failMaterializerMatch(func, "source function must not return values");
  if (functionType.getNumInputs() != 4)
    return failMaterializerMatch(
        func,
        "source function must expose exactly four runtime ABI operands: lhs, "
        "rhs, out, and n");
  for (unsigned index = 0; index < 3; ++index) {
    if (!isRank1DynamicI32MemRef(functionType.getInput(index)))
      return failMaterializerMatch(
          func, "lhs, rhs, and out operands must be memref<?xi32>");
  }
  if (!functionType.getInput(3).isIndex())
    return failMaterializerMatch(func,
                                 "runtime n operand must have index type");

  if (!llvm::hasSingleElement(func.getBody()))
    return failMaterializerMatch(func, "source function must have one block");
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
        return failMaterializerMatch(func,
                                     "source function must contain one scf.for");
      loop = forOp;
      continue;
    }
    return failMaterializerMatch(
        &op,
        "source function may contain only index constants, one scf.for, and "
        "one empty return");
  }
  if (!loop)
    return failMaterializerMatch(func,
                                 "source function must contain one scf.for");
  if (returnCount != 1)
    return failMaterializerMatch(func,
                                 "source function must contain one empty "
                                 "return");

  std::optional<std::int64_t> lower =
      getConstantIndexValue(loop.getLowerBound());
  if (!lower || *lower != 0)
    return failMaterializerMatch(
        loop, "scf.for lower bound must be constant index 0");
  if (loop.getUpperBound() != entry.getArgument(3))
    return failMaterializerMatch(
        loop, "scf.for upper bound must be the runtime n operand");
  std::optional<std::int64_t> step = getConstantIndexValue(loop.getStep());
  if (!step || *step != kSourceVectorLaneCount)
    return failMaterializerMatch(
        loop, "scf.for step must match the bounded vector chunk");
  if (loop.getNumRegionIterArgs() != 0 || loop->getNumResults() != 0)
    return failMaterializerMatch(
        loop,
        "scf.for must not use loop-carried iter_args or yield values");

  mlir::Block &loopBody = loop.getRegion().front();
  llvm::SmallVector<mlir::Operation *, 4> bodyOps;
  for (mlir::Operation &op : loopBody) {
    if (auto yield = llvm::dyn_cast<mlir::scf::YieldOp>(op)) {
      if (yield->getNumOperands() != 0)
        return failMaterializerMatch(
            yield, "scf.for yield must not carry loop values");
      continue;
    }
    bodyOps.push_back(&op);
  }
  if (bodyOps.size() != 4)
    return failMaterializerMatch(
        loop,
        "scf.for body must contain exactly two vector.load ops, one "
        "arith.addi, and one vector.store");

  auto lhsLoad = llvm::dyn_cast<mlir::vector::LoadOp>(bodyOps[0]);
  auto rhsLoad = llvm::dyn_cast<mlir::vector::LoadOp>(bodyOps[1]);
  auto add = llvm::dyn_cast<mlir::arith::AddIOp>(bodyOps[2]);
  auto store = llvm::dyn_cast<mlir::vector::StoreOp>(bodyOps[3]);
  if (!lhsLoad || !rhsLoad || !add || !store)
    return failMaterializerMatch(
        loop,
        "scf.for body operation order must be vector.load, vector.load, "
        "arith.addi, vector.store");

  mlir::Value iv = loop.getInductionVar();
  if (lhsLoad.getBase() != entry.getArgument(0) ||
      !usesOnlyIndex(lhsLoad.getIndices(), iv))
    return failMaterializerMatch(
        lhsLoad, "first vector.load must read lhs at the loop iv");
  if (rhsLoad.getBase() != entry.getArgument(1) ||
      !usesOnlyIndex(rhsLoad.getIndices(), iv))
    return failMaterializerMatch(
        rhsLoad, "second vector.load must read rhs at the loop iv");
  if (!isBoundedSourceVectorI32(lhsLoad.getResult().getType()) ||
      !isBoundedSourceVectorI32(rhsLoad.getResult().getType()))
    return failMaterializerMatch(
        loop, "source vector loads must produce vector<4xi32>");

  if (add.getLhs() != lhsLoad.getResult() || add.getRhs() != rhsLoad.getResult())
    return failMaterializerMatch(
        add, "arith.addi must consume the two source vector loads");
  if (!isBoundedSourceVectorI32(add.getResult().getType()))
    return failMaterializerMatch(add,
                                 "arith.addi must produce vector<4xi32>");

  if (store.getBase() != entry.getArgument(2) ||
      !usesOnlyIndex(store.getIndices(), iv))
    return failMaterializerMatch(store,
                                 "vector.store must write out at the loop iv");
  if (store.getValueToStore() != add.getResult())
    return failMaterializerMatch(
        store, "vector.store must store the arith.addi result");

  BoundedI32AddSourcePattern source;
  source.func = func;
  source.runtimeABIValues = deriveSourceRuntimeABIValues(entry);
  return source;
}

mlir::FlatSymbolRefAttr symbolRef(mlir::OpBuilder &builder,
                                  llvm::StringRef symbol) {
  return mlir::FlatSymbolRefAttr::get(builder.getContext(), symbol);
}

std::string formatSourceABIPurpose(const SourceRuntimeABIValue &value) {
  std::string purpose = "source-arg-";
  purpose += std::to_string(value.sourceArgument.getArgNumber());
  purpose += ":";
  purpose += value.cName.str();
  return purpose;
}

mlir::Operation *createRuntimeABIValue(
    mlir::OpBuilder &builder, mlir::Location loc,
    const SourceRuntimeABIValue &sourceValue, mlir::Type resultType) {
  mlir::OperationState state(loc, "tcrv_rvv.runtime_abi_value");
  state.addTypes(resultType);
  state.addAttribute(
      "role",
      builder.getStringAttr(tianchenrv::support::stringifyRuntimeABIParameterRole(
          sourceValue.role)));
  state.addAttribute("c_name", builder.getStringAttr(sourceValue.cName));
  state.addAttribute("c_type", builder.getStringAttr(sourceValue.cType));
  state.addAttribute(
      "ownership",
      builder.getStringAttr(
          tianchenrv::support::stringifyRuntimeABIParameterOwnership(
              RuntimeABIParameterOwnership::TargetExportABIOwned)));
  state.addAttribute("purpose",
                     builder.getStringAttr(formatSourceABIPurpose(sourceValue)));
  return builder.create(state);
}

void createCapability(mlir::OpBuilder &builder, mlir::Location loc,
                      llvm::StringRef symbol, llvm::StringRef id,
                      llvm::StringRef kind) {
  mlir::OperationState state(loc, "tcrv.exec.capability");
  state.addAttribute("sym_name", builder.getStringAttr(symbol));
  state.addAttribute("id", builder.getStringAttr(id));
  state.addAttribute("kind", builder.getStringAttr(kind));
  state.addAttribute("status", builder.getStringAttr("available"));
  (void)builder.create(state);
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

void createScalarFallbackVariant(mlir::OpBuilder &builder, mlir::Location loc,
                                 llvm::StringRef variantName,
                                 mlir::ArrayAttr requires) {
  mlir::OperationState state(loc, "tcrv.exec.variant");
  state.addAttribute("sym_name", builder.getStringAttr(variantName));
  state.addAttribute("origin", builder.getStringAttr(kScalarPluginName));
  state.addAttribute("requires", requires);
  state.addAttribute("policy", builder.getStringAttr(kScalarFallbackPolicy));
  state.addAttribute("fallback_role", builder.getStringAttr("conservative"));
  state.addRegion();
  auto variant = llvm::cast<tcrv::exec::VariantOp>(builder.create(state));
  variant.getBody().emplaceBlock();
}

void createSelectedDispatchEnvelope(mlir::OpBuilder &builder,
                                    mlir::Location loc,
                                    llvm::StringRef rvvVariantName,
                                    llvm::StringRef fallbackVariantName) {
  mlir::OperationState dispatchState(loc, "tcrv.exec.dispatch");
  dispatchState.addRegion();
  mlir::Operation *dispatch = builder.create(dispatchState);
  dispatch->getRegion(0).emplaceBlock();

  mlir::OpBuilder::InsertionGuard dispatchGuard(builder);
  builder.setInsertionPointToStart(&dispatch->getRegion(0).front());

  mlir::OperationState caseState(loc, "tcrv.exec.case");
  caseState.addAttribute("target", symbolRef(builder, rvvVariantName));
  caseState.addAttribute("origin",
                         builder.getStringAttr(getRVVExtensionPluginName()));
  caseState.addAttribute("policy", builder.getStringAttr(kRVVCasePolicy));
  (void)builder.create(caseState);

  mlir::OperationState fallbackState(loc, "tcrv.exec.fallback");
  fallbackState.addAttribute("target", symbolRef(builder, fallbackVariantName));
  fallbackState.addAttribute("origin", builder.getStringAttr(kScalarPluginName));
  fallbackState.addAttribute("policy", builder.getStringAttr(kFallbackPolicy));
  fallbackState.addAttribute("fallback_role",
                             builder.getStringAttr("conservative"));
  (void)builder.create(fallbackState);
}

void materializeSourceKernel(mlir::OpBuilder &builder,
                             const BoundedI32AddSourcePattern &source) {
  mlir::func::FuncOp func = source.func;
  mlir::Location loc = func.getLoc();
  std::string kernelName = (func.getSymName() + "_kernel").str();
  std::string variantName = (func.getSymName() + "_rvv_i32_add").str();
  std::string fallbackVariantName =
      (func.getSymName() + "_scalar_fallback").str();

  mlir::OperationState kernelState(loc, "tcrv.exec.kernel");
  kernelState.addAttribute("sym_name", builder.getStringAttr(kernelName));
  kernelState.addRegion();
  auto kernel =
      llvm::cast<tcrv::exec::KernelOp>(builder.create(kernelState));
  kernel.getBody().emplaceBlock();

  mlir::OpBuilder::InsertionGuard kernelGuard(builder);
  builder.setInsertionPointToStart(&kernel.getBody().front());

  createCapability(builder, loc, kRVVCapabilitySymbol, getRVVCapabilityID(),
                   getRVVCapabilityKind());
  createCapability(builder, loc, kScalarFallbackCapabilitySymbol,
                   kScalarFallbackCapabilityID, kScalarFallbackCapabilityKind);

  mlir::ArrayAttr requires =
      builder.getArrayAttr({symbolRef(builder, kRVVCapabilitySymbol)});
  mlir::ArrayAttr fallbackRequires = builder.getArrayAttr(
      {symbolRef(builder, kScalarFallbackCapabilitySymbol)});
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
        builder, loc, source.runtimeABIValues[0], runtimeABIValueType);
    mlir::Operation *rhs = createRuntimeABIValue(
        builder, loc, source.runtimeABIValues[1], runtimeABIValueType);
    mlir::Operation *out = createRuntimeABIValue(
        builder, loc, source.runtimeABIValues[2], runtimeABIValueType);
    mlir::Operation *n = createRuntimeABIValue(
        builder, loc, source.runtimeABIValues[3], builder.getIndexType());
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

  createScalarFallbackVariant(builder, loc, fallbackVariantName,
                              fallbackRequires);
  createSelectedDispatchEnvelope(builder, loc, variantName,
                                 fallbackVariantName);
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
    return "Materialize one bounded MLIR vector i32 add source pattern into "
           "the RVV i32m1 selected-boundary form";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::scf::SCFDialect, mlir::vector::VectorDialect,
                    tcrv::exec::TCRVExecDialect, tcrv::rvv::TCRVRVVDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    llvm::SmallVector<mlir::func::FuncOp, 2> sources;
    module.walk([&](mlir::func::FuncOp func) {
      if (!hasForeignLoweringSeedAttr(func))
        sources.push_back(func);
    });
    if (sources.empty())
      return;
    if (sources.size() != 1) {
      (void)failMaterializer(
          module,
          "source module must contain exactly one RVV source function "
          "candidate");
      signalPassFailure();
      return;
    }

    if (mlir::failed(requireSourceOnlyModule(module))) {
      signalPassFailure();
      return;
    }

    mlir::FailureOr<BoundedI32AddSourcePattern> source =
        matchBoundedI32AddSourcePattern(sources.front());
    if (mlir::failed(source)) {
      signalPassFailure();
      return;
    }

    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(module.getBody());
    materializeSourceKernel(builder, *source);

    source->func.erase();
  }
};

} // namespace

std::unique_ptr<::mlir::Pass>
createMaterializeRVVI32M1SelectedBoundarySeedPass() {
  return std::make_unique<MaterializeRVVI32M1SelectedBoundarySeedPass>();
}

} // namespace tianchenrv::plugin::rvv
