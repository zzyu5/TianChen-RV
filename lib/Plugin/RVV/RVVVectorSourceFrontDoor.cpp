#include "TianChenRV/Plugin/RVV/RVVVectorSourceFrontDoor.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace tianchenrv::plugin::rvv {
namespace {

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
  tianchenrv::support::RuntimeABIParameter parameter;
  mlir::BlockArgument sourceArgument;
};

struct BoundedI32ArithmeticSourcePattern {
  mlir::func::FuncOp func;
  RVVSelectedBodyOperationKind arithmeticOp;
  llvm::SmallVector<SourceRuntimeABIValue, 4> runtimeABIValues;
};

struct SourceArithmeticOp {
  mlir::Operation *op;
  RVVSelectedBodyOperationKind kind;
  llvm::StringLiteral sourceName;
  mlir::Value lhs;
  mlir::Value rhs;
  mlir::Value result;
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

bool isBoundedSourceTailMask(mlir::Type type) {
  auto vectorType = llvm::dyn_cast<mlir::VectorType>(type);
  if (!vectorType || vectorType.getRank() != 1)
    return false;
  if (vectorType.getShape().front() != kSourceVectorLaneCount)
    return false;
  auto elementType =
      llvm::dyn_cast<mlir::IntegerType>(vectorType.getElementType());
  return elementType && elementType.getWidth() == 1;
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

bool isMinorIdentityTransfer(mlir::AffineMap map,
                             mlir::ShapedType sourceType,
                             mlir::VectorType vectorType) {
  if (!map)
    return false;
  return map == mlir::AffineMap::getMinorIdentityMap(
                    sourceType.getRank(), vectorType.getRank(),
                    map.getContext());
}

bool hasTrueInBounds(mlir::ArrayAttr inBounds) {
  if (!inBounds)
    return false;
  for (mlir::Attribute attr : inBounds) {
    auto boolAttr = llvm::dyn_cast<mlir::BoolAttr>(attr);
    if (boolAttr && boolAttr.getValue())
      return true;
  }
  return false;
}

mlir::LogicalResult failMaterializer(mlir::Operation *op, llvm::Twine message) {
  op->emitError() << "bounded RVV i32m1 vector-source front door failed: "
                  << message;
  return mlir::failure();
}

mlir::FailureOr<BoundedI32ArithmeticSourcePattern>
failMaterializerMatch(mlir::Operation *op, llvm::Twine message) {
  (void)failMaterializer(op, message);
  return mlir::failure();
}

std::optional<SourceArithmeticOp>
matchSupportedSourceArithmeticOp(mlir::Operation *op) {
  if (auto add = llvm::dyn_cast<mlir::arith::AddIOp>(op))
    return SourceArithmeticOp{op, RVVSelectedBodyOperationKind::Add, "arith.addi",
                              add.getLhs(), add.getRhs(), add.getResult()};
  if (auto sub = llvm::dyn_cast<mlir::arith::SubIOp>(op))
    return SourceArithmeticOp{op, RVVSelectedBodyOperationKind::Sub, "arith.subi",
                              sub.getLhs(), sub.getRhs(), sub.getResult()};
  if (auto mul = llvm::dyn_cast<mlir::arith::MulIOp>(op))
    return SourceArithmeticOp{op, RVVSelectedBodyOperationKind::Mul, "arith.muli",
                              mul.getLhs(), mul.getRhs(), mul.getResult()};
  return std::nullopt;
}

mlir::LogicalResult requireTailSafeTransferRead(
    mlir::vector::TransferReadOp read, mlir::Value expectedBase,
    mlir::Value expectedIndex, mlir::Value expectedMask,
    llvm::StringRef operandName) {
  if (read.getSource() != expectedBase ||
      !usesOnlyIndex(read.getIndices(), expectedIndex))
    return failMaterializer(read,
                            llvm::Twine(operandName) +
                                " vector.transfer_read must read at the loop "
                                "iv from the matching source ABI buffer");
  if (!isBoundedSourceVectorI32(read.getVector().getType()))
    return failMaterializer(
        read, llvm::Twine(operandName) +
                  " vector.transfer_read must produce vector<4xi32>");
  if (!isMinorIdentityTransfer(read.getPermutationMap(),
                               read.getSource().getType(),
                               read.getVector().getType()))
    return failMaterializer(
        read, llvm::Twine(operandName) +
                  " vector.transfer_read must use the minor-identity transfer "
                  "map");
  if (hasTrueInBounds(read.getInBounds()))
    return failMaterializer(
        read, llvm::Twine(operandName) +
                  " vector.transfer_read must not claim in_bounds; tail "
                  "behavior must be carried by the explicit mask");
  if (read.getMask() != expectedMask)
    return failMaterializer(
        read, llvm::Twine(operandName) +
                  " vector.transfer_read must consume the explicit tail mask");
  auto paddingType = llvm::dyn_cast<mlir::IntegerType>(
      read.getPadding().getType());
  if (!paddingType || paddingType.getWidth() != 32)
    return failMaterializer(
        read, llvm::Twine(operandName) +
                  " vector.transfer_read padding must be an i32 value");

  return mlir::success();
}

mlir::LogicalResult requireTailSafeTransferWrite(
    mlir::vector::TransferWriteOp write, mlir::Value expectedValue,
    mlir::Value expectedBase, mlir::Value expectedIndex,
    mlir::Value expectedMask) {
  if (write.getVector() != expectedValue)
    return failMaterializer(
        write, "vector.transfer_write must write the supported arithmetic "
               "result");
  if (write.getSource() != expectedBase ||
      !usesOnlyIndex(write.getIndices(), expectedIndex))
    return failMaterializer(
        write, "vector.transfer_write must write out at the loop iv");
  if (!isBoundedSourceVectorI32(write.getVector().getType()))
    return failMaterializer(
        write, "vector.transfer_write must consume vector<4xi32>");
  if (!isMinorIdentityTransfer(write.getPermutationMap(),
                               write.getSource().getType(),
                               write.getVector().getType()))
    return failMaterializer(
        write,
        "vector.transfer_write must use the minor-identity transfer map");
  if (hasTrueInBounds(write.getInBounds()))
    return failMaterializer(
        write, "vector.transfer_write must not claim in_bounds; tail behavior "
               "must be carried by the explicit mask");
  if (write.getMask() != expectedMask)
    return failMaterializer(
        write, "vector.transfer_write must consume the explicit tail mask");

  return mlir::success();
}

llvm::StringRef
getRVVSelectedBodyOperationName(RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::Add:
    return "tcrv_rvv.i32_add";
  case RVVSelectedBodyOperationKind::Sub:
    return "tcrv_rvv.i32_sub";
  case RVVSelectedBodyOperationKind::Mul:
    return "tcrv_rvv.i32_mul";
  default:
    break;
  }
  llvm_unreachable("unsupported RVV i32m1 source-front-door op");
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
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4> parameters =
      tcrv::rvv::getRVVI32M1ArithmeticRuntimeABIParameters();
  for (auto [index, parameter] : llvm::enumerate(parameters))
    values.push_back({parameter, entry.getArgument(index)});
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

mlir::FailureOr<BoundedI32ArithmeticSourcePattern>
matchBoundedI32ArithmeticSourcePattern(mlir::func::FuncOp func) {
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
    if (llvm::isa<mlir::arith::ConstantOp>(op))
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
        "source function may contain only arith constants, one scf.for, and "
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
  if (bodyOps.size() != 6)
    return failMaterializerMatch(
        loop,
        "scf.for body must explicitly compute remaining AVL, create a tail "
        "mask, perform two masked vector.transfer_read ops, one supported "
        "arith.addi/arith.subi/arith.muli op, and one masked "
        "vector.transfer_write");

  auto remaining = llvm::dyn_cast<mlir::arith::SubIOp>(bodyOps[0]);
  auto tailMask = llvm::dyn_cast<mlir::vector::CreateMaskOp>(bodyOps[1]);
  auto lhsRead = llvm::dyn_cast<mlir::vector::TransferReadOp>(bodyOps[2]);
  auto rhsRead = llvm::dyn_cast<mlir::vector::TransferReadOp>(bodyOps[3]);
  std::optional<SourceArithmeticOp> arithmetic =
      matchSupportedSourceArithmeticOp(bodyOps[4]);
  auto write = llvm::dyn_cast<mlir::vector::TransferWriteOp>(bodyOps[5]);
  if (!remaining || !tailMask || !lhsRead || !rhsRead || !arithmetic || !write)
    return failMaterializerMatch(
        loop,
        "scf.for body operation order must be arith.subi n-iv, "
        "vector.create_mask, masked vector.transfer_read, masked "
        "vector.transfer_read, supported arith.addi/arith.subi/arith.muli, "
        "masked vector.transfer_write");

  mlir::Value iv = loop.getInductionVar();
  if (remaining.getLhs() != entry.getArgument(3) ||
      remaining.getRhs() != iv || !remaining.getResult().getType().isIndex())
    return failMaterializerMatch(
        remaining, "tail mask remaining AVL must be computed as runtime n "
                   "minus the loop iv");
  if (!llvm::hasSingleElement(tailMask.getOperands()) ||
      *tailMask.getOperands().begin() != remaining.getResult() ||
      !isBoundedSourceTailMask(tailMask.getResult().getType()))
    return failMaterializerMatch(
        tailMask, "source tail behavior must be explicit as vector.create_mask "
                  "from remaining AVL with type vector<4xi1>");

  if (mlir::failed(requireTailSafeTransferRead(
          lhsRead, entry.getArgument(0), iv, tailMask.getResult(), "lhs")))
    return mlir::failure();
  if (mlir::failed(requireTailSafeTransferRead(
          rhsRead, entry.getArgument(1), iv, tailMask.getResult(), "rhs")))
    return mlir::failure();

  if (arithmetic->lhs != lhsRead.getResult() ||
      arithmetic->rhs != rhsRead.getResult())
    return failMaterializerMatch(
        arithmetic->op,
        llvm::Twine(arithmetic->sourceName) +
            " must consume the two source vector transfers");
  if (!isBoundedSourceVectorI32(arithmetic->result.getType()))
    return failMaterializerMatch(
        arithmetic->op,
        llvm::Twine(arithmetic->sourceName) + " must produce vector<4xi32>");

  if (mlir::failed(requireTailSafeTransferWrite(
          write, arithmetic->result, entry.getArgument(2), iv,
          tailMask.getResult())))
    return mlir::failure();

  BoundedI32ArithmeticSourcePattern source;
  source.func = func;
  source.arithmeticOp = arithmetic->kind;
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
  purpose += value.parameter.cName;
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
          sourceValue.parameter.role)));
  state.addAttribute("c_name",
                     builder.getStringAttr(sourceValue.parameter.cName));
  state.addAttribute("c_type",
                     builder.getStringAttr(sourceValue.parameter.cType));
  state.addAttribute(
      "ownership",
      builder.getStringAttr(
          tianchenrv::support::stringifyRuntimeABIParameterOwnership(
              sourceValue.parameter.ownership)));
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
                             mlir::Value nValue) {
  mlir::OperationState state(loc, "tcrv_rvv.setvl");
  state.addOperands(nValue);
  state.addTypes(tcrv::rvv::VLType::get(builder.getContext()));
  tcrv::rvv::populateRVVI32M1ArithmeticConfigAttrs(builder, state);
  return builder.create(state);
}

mlir::Operation *createWithVL(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value vlValue,
                              llvm::StringRef kernelName,
                              llvm::StringRef variantName,
                              mlir::ArrayAttr requires) {
  mlir::OperationState state(loc, "tcrv_rvv.with_vl");
  state.addOperands(vlValue);
  tcrv::rvv::populateRVVI32M1ArithmeticConfigAttrs(builder, state);
  state.addAttribute(getRVVSourceKernelAttrName(),
                     builder.getStringAttr(kernelName));
  state.addAttribute(getRVVSelectedVariantAttrName(),
                     symbolRef(builder, variantName));
  state.addAttribute(getRVVOriginAttrName(),
                     builder.getStringAttr(getRVVExtensionPluginName()));
  state.addAttribute(getRVVSelectedPathRoleAttrName(),
                     builder.getStringAttr("dispatch case"));
  state.addAttribute(getRVVStatusAttrName(),
                     builder.getStringAttr(getRVVLoweringBoundaryStatus()));
  state.addAttribute(getRVVRequiredCapabilitiesAttrName(), requires);
  state.addAttribute(getRVVConstructionProtocolMetadataName(),
                     builder.getStringAttr(getRVVConstructionProtocolVersion()));
  state.addAttribute(getRVVEmitCRouteMappingMetadataName(),
                     builder.getStringAttr(
                         getRVVConstructionManifest().emitcRoute.routeID));
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

mlir::Operation *createI32Arithmetic(mlir::OpBuilder &builder,
                                     mlir::Location loc,
                                     RVVSelectedBodyOperationKind arithmeticOp,
                                     mlir::Value lhs, mlir::Value rhs,
                                     mlir::Value vl) {
  mlir::OperationState state(loc,
                            getRVVSelectedBodyOperationName(arithmeticOp));
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
                             const BoundedI32ArithmeticSourcePattern &source) {
  mlir::func::FuncOp func = source.func;
  mlir::Location loc = func.getLoc();
  std::string kernelName = (func.getSymName() + "_kernel").str();
  std::string variantName =
      (func.getSymName() + "_rvv_i32_" +
       stringifyRVVSelectedBodyOperationKind(source.arithmeticOp))
          .str();
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
  auto policy = tcrv::rvv::getRVVI32M1ArithmeticPolicy(builder.getContext());

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
    mlir::Operation *setvl = createSetVL(builder, loc, n->getResult(0));
    mlir::Operation *withVL = createWithVL(
        builder, loc, setvl->getResult(0), kernelName, variantName, requires);

    builder.setInsertionPointToStart(&withVL->getRegion(0).front());
    mlir::Operation *lhsLoad =
        createI32Load(builder, loc, lhs->getResult(0), setvl->getResult(0));
    mlir::Operation *rhsLoad =
        createI32Load(builder, loc, rhs->getResult(0), setvl->getResult(0));
    mlir::Operation *result = createI32Arithmetic(
        builder, loc, source.arithmeticOp, lhsLoad->getResult(0),
        rhsLoad->getResult(0), setvl->getResult(0));
    createI32Store(builder, loc, out->getResult(0), result->getResult(0),
                   setvl->getResult(0));
  }

  createScalarFallbackVariant(builder, loc, fallbackVariantName,
                              fallbackRequires);
  createSelectedDispatchEnvelope(builder, loc, variantName,
                                 fallbackVariantName);
}

class MaterializeRVVI32M1VectorSourceFrontDoorPass final
    : public mlir::PassWrapper<
          MaterializeRVVI32M1VectorSourceFrontDoorPass,
          mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const final {
    return "tcrv-rvv-materialize-i32m1-vector-source-front-door";
  }

  llvm::StringRef getDescription() const final {
    return "Materialize one bounded MLIR vector i32 add/sub/mul source "
           "pattern into the RVV i32m1 selected boundary and dispatch front "
           "door";
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

    mlir::FailureOr<BoundedI32ArithmeticSourcePattern> source =
        matchBoundedI32ArithmeticSourcePattern(sources.front());
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
createMaterializeRVVI32M1VectorSourceFrontDoorPass() {
  return std::make_unique<MaterializeRVVI32M1VectorSourceFrontDoorPass>();
}

} // namespace tianchenrv::plugin::rvv
