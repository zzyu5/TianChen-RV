#include "Weft/Plugin/Scalar/ScalarBackendEmissionDriver.h"

#include "Weft/Conversion/EmitC/BackendEmissionRegistry.h"
#include "Weft/Conversion/EmitC/TypedBackendEmissionDriver.h"
#include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.h"
#include "Weft/Dialect/Scalar/IR/ScalarDialect.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>

namespace weft {
namespace plugin {
namespace scalar {

namespace {

namespace emitc = ::mlir::emitc;
namespace weftemitc = ::weft::conversion::emitc;

constexpr llvm::StringLiteral kScalarSkeletonCallee(
    "weft_scalar_compute_skeleton");
constexpr llvm::StringLiteral kEmitCLowerableInterfaceName(
    "WEFTEmitCLowerableOpInterface");

std::string routeSourceComment(llvm::StringRef opName, llvm::StringRef role,
                               llvm::StringRef opInterface) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// weft_emitc.route_source_op=" << opName << " role=" << role
     << " op_interface=" << opInterface;
  os.flush();
  return text;
}

std::string stepComment(llvm::StringRef opName, llvm::StringRef role,
                        llvm::StringRef opInterface, llvm::StringRef callee) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// weft_emitc.source_op=" << opName << " role=" << role
     << " op_interface=" << opInterface << " callee=" << callee;
  os.flush();
  return text;
}

std::string kernelStepComment(llvm::StringRef opName, llvm::StringRef role,
                              llvm::StringRef step) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// weft_emitc.source_op=" << opName << " role=" << role
     << " step=" << step;
  os.flush();
  return text;
}

/// The tracer-bullet final body has no internal loop plan. Its operation
/// identity and immediate are already the complete Scalar computation body.
class ScalarImmediateCallBodyToEmitCFunc final
    : public mlir::OpConversionPattern<weft::scalar::ImmediateCallBodyOp> {
public:
  using mlir::OpConversionPattern<
      weft::scalar::ImmediateCallBodyOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(weft::scalar::ImmediateCallBodyOp body,
                  OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = body.getLoc();
    auto variant = body.getSelectedVariantAttr();
    auto sourceKernel = body.getSourceKernelAttr();
    if (!variant || !sourceKernel)
      return rewriter.notifyMatchFailure(
          body, "immediate_call_body requires selected_variant and "
                "source_kernel attributes");

    std::string functionName =
        ("weft_emitc_" + sourceKernel.getValue() + "_" + variant.getValue())
            .str();
    auto lowerable =
        llvm::dyn_cast<weftemitc::WEFTEmitCLowerableOpInterface>(
            body.getOperation());
    if (!lowerable)
      return rewriter.notifyMatchFailure(
          body, "weft_scalar.immediate_call_body must implement "
                "WEFTEmitCLowerableOpInterface");
    llvm::StringRef callee = kScalarSkeletonCallee;

    auto module = body->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(body, "body has no module");

    mlir::Type i32 = rewriter.getI32Type();
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<emitc::IncludeOp>(loc, "stdint.h",
                                        /*is_standard_include=*/true);
    }

    mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());
    mlir::FunctionType calleeType = rewriter.getFunctionType({i32}, {i32});
    llvm::SmallVector<mlir::NamedAttribute, 1> calleeAttrs;
    calleeAttrs.push_back(rewriter.getNamedAttr(
        mlir::SymbolTable::getVisibilityAttrName(),
        rewriter.getStringAttr("private")));
    rewriter.create<emitc::FuncOp>(loc, callee, calleeType, calleeAttrs);

    mlir::FunctionType functionType =
        rewriter.getFunctionType(/*inputs=*/{}, /*results=*/{});
    llvm::SmallVector<mlir::NamedAttribute, 1> funcAttrs;
    funcAttrs.push_back(rewriter.getNamedAttr(
        "specifiers", rewriter.getStrArrayAttr({"extern", "\"C\""})));
    auto func = rewriter.create<emitc::FuncOp>(loc, functionName, functionType,
                                               funcAttrs);
    mlir::Block *entry = func.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);

    llvm::StringRef sourceOpName =
        lowerable.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef sourceRole = lowerable.getWEFTEmitCLowerableSourceRole();
    rewriter.create<emitc::VerbatimOp>(
        loc, routeSourceComment(sourceOpName, sourceRole,
                                kEmitCLowerableInterfaceName));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(sourceOpName, sourceRole,
                         kEmitCLowerableInterfaceName, callee));

    auto constant = rewriter.create<emitc::ConstantOp>(
        loc, i32, rewriter.getI32IntegerAttr(body.getScalarImmediate()));
    rewriter.create<emitc::CallOpaqueOp>(
        loc, mlir::TypeRange{i32}, callee,
        mlir::ValueRange{constant.getResult()});
    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());

    rewriter.eraseOp(body);
    return mlir::success();
  }
};

struct TernaryProjectionState {
  mlir::ConversionPatternRewriter &rewriter;
  mlir::Location loc;
  llvm::StringRef sourceOpName;
  llvm::StringRef sourceRole;

  mlir::Type sizeType;
  mlir::Type intType;
  mlir::Type floatType;
  mlir::Type constU8Type;
  mlir::Type constI8Type;
  mlir::Type constFloatType;
  mlir::Type constFloatPtrType;

  mlir::Value nArg;
  mlir::Value outArg;
  mlir::Value weightsArg;
  mlir::Value activationsArg;
  mlir::Value sumfVar;
  mlir::Value blockIndex;
  mlir::Value weightBlockBase;
  mlir::Value activationBlockBase;
  mlir::Value packedWeights;
  mlir::Value quantActivations;
  mlir::Value blockAccumulatorVar;
  mlir::Value groupIndex;
  mlir::Value planeIndex;
  mlir::Value laneIndex;
  mlir::Value shift;

  mlir::Value sizeLiteral(int64_t value) {
    return rewriter
        .create<emitc::LiteralOp>(loc, sizeType, std::to_string(value))
        .getResult();
  }

  mlir::Value intLiteral(int64_t value) {
    return rewriter
        .create<emitc::LiteralOp>(loc, intType, std::to_string(value))
        .getResult();
  }

  void emitStep(llvm::StringRef step) {
    rewriter.create<emitc::VerbatimOp>(
        loc, kernelStepComment(sourceOpName, sourceRole, step));
  }
};

mlir::LogicalResult projectTernaryRegion(mlir::Block &planBlock,
                                         TernaryProjectionState &state);

mlir::LogicalResult projectTernaryNode(mlir::Operation *node,
                                       TernaryProjectionState &state) {
  mlir::ConversionPatternRewriter &rewriter = state.rewriter;
  mlir::Location loc = state.loc;

  if (auto block =
          llvm::dyn_cast<weft::scalar::TernaryBlockLoopOp>(node)) {
    state.emitStep("super_block_count");
    mlir::Value nSize =
        rewriter.create<emitc::CastOp>(loc, state.sizeType, state.nArg)
            .getResult();
    mlir::Value blockCount =
        rewriter
            .create<emitc::DivOp>(loc, state.sizeType, nSize,
                                  state.sizeLiteral(block.getQk()))
            .getResult();

    state.sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(state.floatType),
        emitc::OpaqueAttr::get(rewriter.getContext(), ""));
    rewriter.create<emitc::AssignOp>(
        loc, state.sumfVar,
        rewriter
            .create<emitc::LiteralOp>(loc, state.floatType, "0.0f")
            .getResult());

    state.emitStep("super_block_loop");
    auto loop = rewriter.create<emitc::ForOp>(
        loc, state.sizeLiteral(0), blockCount,
        state.sizeLiteral(block.getStep()), /*bodyBuilder=*/nullptr);
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(loop.getBody());
    TernaryProjectionState nested = state;
    nested.blockIndex = loop.getInductionVar();

    mlir::Value weightOffset =
        rewriter
            .create<emitc::MulOp>(
                loc, state.sizeType, nested.blockIndex,
                state.sizeLiteral(block.getWeightBlockStride()))
            .getResult();
    nested.weightBlockBase =
        rewriter
            .create<emitc::AddOp>(loc, state.weightsArg.getType(),
                                  state.weightsArg, weightOffset)
            .getResult();
    nested.packedWeights = nested.weightBlockBase;

    mlir::Value activationOffset =
        rewriter
            .create<emitc::MulOp>(
                loc, state.sizeType, nested.blockIndex,
                state.sizeLiteral(block.getActivationBlockStride()))
            .getResult();
    nested.activationBlockBase =
        rewriter
            .create<emitc::AddOp>(loc, state.activationsArg.getType(),
                                  state.activationsArg, activationOffset)
            .getResult();
    nested.quantActivations = nested.activationBlockBase;
    if (block.getActivationQuantByteOffset() != 0)
      nested.quantActivations =
          rewriter
              .create<emitc::AddOp>(
                  loc, state.activationsArg.getType(),
                  nested.activationBlockBase,
                  state.sizeLiteral(block.getActivationQuantByteOffset()))
              .getResult();

    nested.blockAccumulatorVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(state.intType),
        emitc::OpaqueAttr::get(rewriter.getContext(), ""));
    rewriter.create<emitc::AssignOp>(
        loc, nested.blockAccumulatorVar,
        rewriter.create<emitc::LiteralOp>(loc, state.intType, "0").getResult());
    return projectTernaryRegion(block.getBody().front(), nested);
  }

  if (auto group =
          llvm::dyn_cast<weft::scalar::TernaryPlaneGroupLoopOp>(node)) {
    state.emitStep("plane_group_loop");
    auto loop = rewriter.create<emitc::ForOp>(
        loc, state.sizeLiteral(0), state.sizeLiteral(group.getUpperBound()),
        state.sizeLiteral(group.getStep()), /*bodyBuilder=*/nullptr);
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(loop.getBody());
    TernaryProjectionState nested = state;
    nested.groupIndex = loop.getInductionVar();
    return projectTernaryRegion(group.getBody().front(), nested);
  }

  if (auto plane =
          llvm::dyn_cast<weft::scalar::TernaryPlaneLoopOp>(node)) {
    auto loop = rewriter.create<emitc::ForOp>(
        loc, state.sizeLiteral(0), state.sizeLiteral(plane.getUpperBound()),
        state.sizeLiteral(plane.getStep()), /*bodyBuilder=*/nullptr);
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(loop.getBody());
    TernaryProjectionState nested = state;
    nested.planeIndex = loop.getInductionVar();
    nested.shift =
        rewriter
            .create<emitc::CastOp>(
                loc, state.intType,
                rewriter
                    .create<emitc::MulOp>(
                        loc, state.sizeType, nested.planeIndex,
                        state.sizeLiteral(plane.getFieldBits()))
                    .getResult())
            .getResult();
    return projectTernaryRegion(plane.getBody().front(), nested);
  }

  if (auto lane = llvm::dyn_cast<weft::scalar::TernaryLaneLoopOp>(node)) {
    auto loop = rewriter.create<emitc::ForOp>(
        loc, state.sizeLiteral(0), state.sizeLiteral(lane.getUpperBound()),
        state.sizeLiteral(lane.getStep()), /*bodyBuilder=*/nullptr);
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(loop.getBody());
    TernaryProjectionState nested = state;
    nested.laneIndex = loop.getInductionVar();
    return projectTernaryRegion(lane.getBody().front(), nested);
  }

  if (auto decode =
          llvm::dyn_cast<weft::scalar::TernaryDecodeMacOp>(node)) {
    state.emitStep("ternary_decode_mac");
    mlir::Value weightIndex =
        rewriter
            .create<emitc::AddOp>(loc, state.sizeType, state.groupIndex,
                                  state.laneIndex)
            .getResult();
    auto packedWeights =
        llvm::cast<mlir::TypedValue<emitc::PointerType>>(state.packedWeights);
    mlir::Value weightElement =
        rewriter.create<emitc::SubscriptOp>(loc, packedWeights, weightIndex)
            .getResult();
    mlir::Value weightByte =
        rewriter
            .create<emitc::LoadOp>(loc, state.constU8Type, weightElement)
            .getResult();
    mlir::Value weightInt =
        rewriter.create<emitc::CastOp>(loc, state.intType, weightByte)
            .getResult();
    mlir::Value shifted =
        rewriter
            .create<emitc::BitwiseRightShiftOp>(loc, state.intType, weightInt,
                                                state.shift)
            .getResult();
    mlir::Value masked =
        rewriter
            .create<emitc::BitwiseAndOp>(
                loc, state.intType, shifted,
                state.intLiteral(decode.getFieldMask()))
            .getResult();
    mlir::Value weight =
        rewriter
            .create<emitc::SubOp>(
                loc, state.intType, masked,
                state.intLiteral(decode.getDecodeZeroPoint()))
            .getResult();

    mlir::Value groupTerm =
        rewriter
            .create<emitc::MulOp>(
                loc, state.sizeType, state.groupIndex,
                state.sizeLiteral(decode.getActivationPlaneStride()))
            .getResult();
    mlir::Value planeTerm =
        rewriter
            .create<emitc::MulOp>(
                loc, state.sizeType, state.planeIndex,
                state.sizeLiteral(decode.getPlaneLanes()))
            .getResult();
    mlir::Value activationIndex =
        rewriter
            .create<emitc::AddOp>(
                loc, state.sizeType,
                rewriter
                    .create<emitc::AddOp>(loc, state.sizeType, groupTerm,
                                          planeTerm)
                    .getResult(),
                state.laneIndex)
            .getResult();
    auto quantActivations =
        llvm::cast<mlir::TypedValue<emitc::PointerType>>(
            state.quantActivations);
    mlir::Value activationElement =
        rewriter
            .create<emitc::SubscriptOp>(loc, quantActivations, activationIndex)
            .getResult();
    mlir::Value activationByte =
        rewriter
            .create<emitc::LoadOp>(loc, state.constI8Type, activationElement)
            .getResult();
    mlir::Value activationInt =
        rewriter.create<emitc::CastOp>(loc, state.intType, activationByte)
            .getResult();
    mlir::Value product =
        rewriter
            .create<emitc::MulOp>(loc, state.intType, activationInt, weight)
            .getResult();
    mlir::Value current =
        rewriter
            .create<emitc::LoadOp>(loc, state.intType,
                                   state.blockAccumulatorVar)
            .getResult();
    rewriter.create<emitc::AssignOp>(
        loc, state.blockAccumulatorVar,
        rewriter.create<emitc::AddOp>(loc, state.intType, current, product)
            .getResult());
    return mlir::success();
  }

  if (auto fold = llvm::dyn_cast<weft::scalar::TernaryScaleFoldOp>(node)) {
    state.emitStep("fold_activation_d");
    mlir::Value activationScaleAddress = state.activationBlockBase;
    if (fold.getActivationDByteOffset() != 0)
      activationScaleAddress =
          rewriter
              .create<emitc::AddOp>(
                  loc, state.activationsArg.getType(),
                  state.activationBlockBase,
                  state.sizeLiteral(fold.getActivationDByteOffset()))
              .getResult();
    mlir::Value activationScalePointer =
        rewriter
            .create<emitc::CastOp>(loc, state.constFloatPtrType,
                                   activationScaleAddress)
            .getResult();
    mlir::Value zeroIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value activationScaleElement =
        rewriter
            .create<emitc::SubscriptOp>(
                loc,
                llvm::cast<mlir::TypedValue<emitc::PointerType>>(
                    activationScalePointer),
                zeroIndex)
            .getResult();
    mlir::Value activationScale =
        rewriter
            .create<emitc::LoadOp>(loc, state.constFloatType,
                                   activationScaleElement)
            .getResult();

    state.emitStep("fold_weight_d");
    mlir::Value weightScaleAddress = state.weightBlockBase;
    if (fold.getWeightDByteOffset() != 0)
      weightScaleAddress =
          rewriter
              .create<emitc::AddOp>(
                  loc, state.weightsArg.getType(), state.weightBlockBase,
                  state.sizeLiteral(fold.getWeightDByteOffset()))
              .getResult();
    mlir::Value weightScale =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{state.floatType},
                "(float)*(const _Float16 *)",
                mlir::ValueRange{weightScaleAddress})
            .getResult(0);
    mlir::Value combinedScale =
        rewriter
            .create<emitc::MulOp>(loc, state.floatType, activationScale,
                                  weightScale)
            .getResult();

    state.emitStep("scalar_fold");
    mlir::Value accumulator =
        rewriter
            .create<emitc::LoadOp>(loc, state.intType,
                                   state.blockAccumulatorVar)
            .getResult();
    mlir::Value currentSum =
        rewriter.create<emitc::LoadOp>(loc, state.floatType, state.sumfVar)
            .getResult();
    auto expression = rewriter.create<emitc::ExpressionOp>(
        loc, state.floatType, /*do_not_inline=*/false);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      mlir::Block *expressionBlock =
          rewriter.createBlock(&expression.getRegion());
      rewriter.setInsertionPointToStart(expressionBlock);
      mlir::Value accumulatorFloat =
          rewriter.create<emitc::CastOp>(loc, state.floatType, accumulator)
              .getResult();
      mlir::Value term =
          rewriter
              .create<emitc::MulOp>(loc, state.floatType, accumulatorFloat,
                                    combinedScale)
              .getResult();
      mlir::Value next =
          rewriter
              .create<emitc::AddOp>(loc, state.floatType, currentSum, term)
              .getResult();
      rewriter.create<emitc::YieldOp>(loc, next);
    }
    rewriter.create<emitc::AssignOp>(loc, state.sumfVar,
                                     expression.getResult());
    return mlir::success();
  }

  if (llvm::isa<weft::scalar::TernaryStoreOp>(node)) {
    state.emitStep("store_s");
    mlir::Value sum =
        rewriter.create<emitc::LoadOp>(loc, state.floatType, state.sumfVar)
            .getResult();
    mlir::Value zeroIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    auto output =
        llvm::cast<mlir::TypedValue<emitc::PointerType>>(state.outArg);
    mlir::Value element =
        rewriter.create<emitc::SubscriptOp>(loc, output, zeroIndex).getResult();
    rewriter.create<emitc::AssignOp>(loc, element, sum);
    return mlir::success();
  }

  return node->emitError(
      "unsupported node in exact Scalar ternary computation plan");
}

mlir::LogicalResult projectTernaryRegion(mlir::Block &planBlock,
                                         TernaryProjectionState &state) {
  for (mlir::Operation &node : planBlock)
    if (mlir::failed(projectTernaryNode(&node, state)))
      return mlir::failure();
  return mlir::success();
}

class ScalarPackedTernaryDotBodyToEmitCFunc final
    : public mlir::OpConversionPattern<weft::scalar::PackedTernaryDotBodyOp> {
public:
  using mlir::OpConversionPattern<
      weft::scalar::PackedTernaryDotBodyOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(weft::scalar::PackedTernaryDotBodyOp body,
                  OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto variant = body.getSelectedVariantAttr();
    auto sourceKernel = body.getSourceKernelAttr();
    auto lowerable =
        llvm::dyn_cast<weftemitc::WEFTEmitCLowerableOpInterface>(
            body.getOperation());
    auto module = body->getParentOfType<mlir::ModuleOp>();
    if (!variant || !sourceKernel || !lowerable || !module)
      return rewriter.notifyMatchFailure(
          body, "packed_ternary_dot_body lacks exact ownership or module "
                "context");

    mlir::Location loc = body.getLoc();
    mlir::MLIRContext *context = rewriter.getContext();
    mlir::Type sizeType = emitc::OpaqueType::get(context, "size_t");
    mlir::Type intType = emitc::OpaqueType::get(context, "int");
    mlir::Type floatType = emitc::OpaqueType::get(context, "float");
    mlir::Type constU8Type = emitc::OpaqueType::get(context, "const uint8_t");
    mlir::Type constI8Type = emitc::OpaqueType::get(context, "const int8_t");
    mlir::Type constFloatType =
        emitc::OpaqueType::get(context, "const float");
    mlir::Type floatPtrType = emitc::PointerType::get(floatType);
    mlir::Type constU8PtrType = emitc::PointerType::get(constU8Type);
    mlir::Type constI8PtrType = emitc::PointerType::get(constI8Type);
    mlir::Type constFloatPtrType = emitc::PointerType::get(constFloatType);

    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<emitc::VerbatimOp>(
          loc, "#include <stddef.h>\n#include <stdint.h>");
    }

    mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());
    std::string functionName =
        ("weft_emitc_" + sourceKernel.getValue() + "_" + variant.getValue())
            .str();
    mlir::FunctionType functionType = rewriter.getFunctionType(
        {intType, floatPtrType, constU8PtrType, constI8PtrType}, {});
    llvm::SmallVector<mlir::NamedAttribute, 1> funcAttrs;
    funcAttrs.push_back(rewriter.getNamedAttr(
        "specifiers", rewriter.getStrArrayAttr({"extern", "\"C\""})));
    auto func = rewriter.create<emitc::FuncOp>(loc, functionName, functionType,
                                               funcAttrs);
    mlir::Block *entry = func.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);

    llvm::StringRef sourceOpName =
        lowerable.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef sourceRole = lowerable.getWEFTEmitCLowerableSourceRole();
    rewriter.create<emitc::VerbatimOp>(
        loc, routeSourceComment(sourceOpName, sourceRole,
                                kEmitCLowerableInterfaceName));
    TernaryProjectionState state{
        rewriter,
        loc,
        sourceOpName,
        sourceRole,
        sizeType,
        intType,
        floatType,
        constU8Type,
        constI8Type,
        constFloatType,
        constFloatPtrType,
        entry->getArgument(0),
        entry->getArgument(1),
        entry->getArgument(2),
        entry->getArgument(3),
    };
    if (mlir::failed(projectTernaryRegion(body.getBody().front(), state)))
      return rewriter.notifyMatchFailure(
          body, "failed to project exact Scalar ternary computation plan");
    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
    rewriter.eraseOp(body);
    return mlir::success();
  }
};

struct AffineProjectionState {
  mlir::ConversionPatternRewriter &rewriter;
  mlir::Location loc;
  llvm::StringRef sourceOpName;
  llvm::StringRef sourceRole;

  mlir::Type sizeType;
  mlir::Type intType;
  mlir::Type floatType;
  mlir::Type constU8Type;

  mlir::Value nArg;
  mlir::Value outArg;
  mlir::Value weightsArg;
  mlir::Value blockIndex;
  mlir::Value weightBlockBase;
  mlir::Value outputBlockBase;
  mlir::Value blockScale;
  mlir::Value quantBase;
  mlir::Value packedIndex;
  int64_t outputBlockStride = 0;

  mlir::Value sizeLiteral(int64_t value) {
    return rewriter
        .create<emitc::LiteralOp>(loc, sizeType, std::to_string(value))
        .getResult();
  }

  mlir::Value intLiteral(int64_t value) {
    return rewriter
        .create<emitc::LiteralOp>(loc, intType, std::to_string(value))
        .getResult();
  }

  void emitStep(llvm::StringRef step) {
    rewriter.create<emitc::VerbatimOp>(
        loc, kernelStepComment(sourceOpName, sourceRole, step));
  }
};

mlir::LogicalResult projectAffineRegion(mlir::Block &planBlock,
                                        AffineProjectionState &state);

mlir::LogicalResult projectAffineNode(mlir::Operation *node,
                                      AffineProjectionState &state) {
  mlir::ConversionPatternRewriter &rewriter = state.rewriter;
  mlir::Location loc = state.loc;

  if (auto block = llvm::dyn_cast<weft::scalar::AffineBlockLoopOp>(node)) {
    state.emitStep("block_count");
    mlir::Value nSize =
        rewriter.create<emitc::CastOp>(loc, state.sizeType, state.nArg)
            .getResult();
    mlir::Value blockCount =
        rewriter
            .create<emitc::DivOp>(loc, state.sizeType, nSize,
                                  state.sizeLiteral(block.getQk()))
            .getResult();

    state.emitStep("block_loop");
    auto loop = rewriter.create<emitc::ForOp>(
        loc, state.sizeLiteral(0), blockCount,
        state.sizeLiteral(block.getStep()), /*bodyBuilder=*/nullptr);
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(loop.getBody());
    AffineProjectionState nested = state;
    nested.blockIndex = loop.getInductionVar();
    mlir::Value weightOffset =
        rewriter
            .create<emitc::MulOp>(
                loc, state.sizeType, nested.blockIndex,
                state.sizeLiteral(block.getWeightBlockStride()))
            .getResult();
    nested.weightBlockBase =
        rewriter
            .create<emitc::AddOp>(loc, state.weightsArg.getType(),
                                  state.weightsArg, weightOffset)
            .getResult();
    nested.outputBlockStride = static_cast<int64_t>(block.getQk());
    return projectAffineRegion(block.getBody().front(), nested);
  }

  if (auto scale = llvm::dyn_cast<weft::scalar::AffineBlockScaleOp>(node)) {
    state.emitStep("block_scale");
    mlir::Value address = state.weightBlockBase;
    if (scale.getWeightDByteOffset() != 0)
      address =
          rewriter
              .create<emitc::AddOp>(
                  loc, state.weightsArg.getType(), state.weightBlockBase,
                  state.sizeLiteral(scale.getWeightDByteOffset()))
              .getResult();
    state.blockScale =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{state.floatType},
                "(float)*(const _Float16 *)", mlir::ValueRange{address})
            .getResult(0);
    return mlir::success();
  }

  if (auto quantBase =
          llvm::dyn_cast<weft::scalar::AffineQuantBaseOp>(node)) {
    state.quantBase = state.weightBlockBase;
    if (quantBase.getWeightQuantByteOffset() != 0)
      state.quantBase =
          rewriter
              .create<emitc::AddOp>(
                  loc, state.weightsArg.getType(), state.weightBlockBase,
                  state.sizeLiteral(quantBase.getWeightQuantByteOffset()))
              .getResult();
    return mlir::success();
  }

  if (auto packed =
          llvm::dyn_cast<weft::scalar::AffinePackedByteLoopOp>(node)) {
    state.outputBlockBase =
        rewriter
            .create<emitc::MulOp>(loc, state.sizeType, state.blockIndex,
                                  state.sizeLiteral(state.outputBlockStride))
            .getResult();
    state.emitStep("nibble_loop");
    auto loop = rewriter.create<emitc::ForOp>(
        loc, state.sizeLiteral(0), state.sizeLiteral(packed.getUpperBound()),
        state.sizeLiteral(packed.getStep()), /*bodyBuilder=*/nullptr);
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(loop.getBody());
    AffineProjectionState nested = state;
    nested.packedIndex = loop.getInductionVar();
    return projectAffineRegion(packed.getBody().front(), nested);
  }

  if (auto decode =
          llvm::dyn_cast<weft::scalar::AffineDecodeScaleScatterOp>(node)) {
    state.emitStep("nibble_decode");
    auto quantBase =
        llvm::cast<mlir::TypedValue<emitc::PointerType>>(state.quantBase);
    mlir::Value quantElement =
        rewriter
            .create<emitc::SubscriptOp>(loc, quantBase, state.packedIndex)
            .getResult();
    mlir::Value quantByte =
        rewriter.create<emitc::LoadOp>(loc, state.constU8Type, quantElement)
            .getResult();
    mlir::Value quantInt =
        rewriter.create<emitc::CastOp>(loc, state.intType, quantByte)
            .getResult();

    mlir::Value lowSource = quantInt;
    if (decode.getLowFieldShift() != 0)
      lowSource =
          rewriter
              .create<emitc::BitwiseRightShiftOp>(
                  loc, state.intType, quantInt,
                  state.intLiteral(decode.getLowFieldShift()))
              .getResult();
    mlir::Value lowField =
        rewriter
            .create<emitc::BitwiseAndOp>(
                loc, state.intType, lowSource,
                state.intLiteral(decode.getFieldMask()))
            .getResult();
    mlir::Value lowValue =
        rewriter
            .create<emitc::SubOp>(
                loc, state.intType, lowField,
                state.intLiteral(decode.getDecodeZeroPoint()))
            .getResult();
    mlir::Value highField =
        rewriter
            .create<emitc::BitwiseRightShiftOp>(
                loc, state.intType, quantInt,
                state.intLiteral(decode.getHighFieldShift()))
            .getResult();
    mlir::Value highValue =
        rewriter
            .create<emitc::SubOp>(
                loc, state.intType, highField,
                state.intLiteral(decode.getDecodeZeroPoint()))
            .getResult();

    state.emitStep("scatter_low");
    mlir::Value outputBase =
        rewriter
            .create<emitc::AddOp>(loc, state.sizeType, state.outputBlockBase,
                                  state.packedIndex)
            .getResult();
    mlir::Value lowIndex = outputBase;
    if (decode.getLowOutputDelta() != 0)
      lowIndex =
          rewriter
              .create<emitc::AddOp>(
                  loc, state.sizeType, outputBase,
                  state.sizeLiteral(decode.getLowOutputDelta()))
              .getResult();
    mlir::Value lowFloat =
        rewriter.create<emitc::CastOp>(loc, state.floatType, lowValue)
            .getResult();
    mlir::Value lowScaled =
        rewriter
            .create<emitc::MulOp>(loc, state.floatType, lowFloat,
                                  state.blockScale)
            .getResult();
    auto output =
        llvm::cast<mlir::TypedValue<emitc::PointerType>>(state.outArg);
    mlir::Value lowElement =
        rewriter.create<emitc::SubscriptOp>(loc, output, lowIndex).getResult();
    rewriter.create<emitc::AssignOp>(loc, lowElement, lowScaled);

    state.emitStep("scatter_high");
    mlir::Value highIndex = outputBase;
    if (decode.getHighOutputDelta() != 0)
      highIndex =
          rewriter
              .create<emitc::AddOp>(
                  loc, state.sizeType, outputBase,
                  state.sizeLiteral(decode.getHighOutputDelta()))
              .getResult();
    mlir::Value highFloat =
        rewriter.create<emitc::CastOp>(loc, state.floatType, highValue)
            .getResult();
    mlir::Value highScaled =
        rewriter
            .create<emitc::MulOp>(loc, state.floatType, highFloat,
                                  state.blockScale)
            .getResult();
    mlir::Value highElement =
        rewriter.create<emitc::SubscriptOp>(loc, output, highIndex).getResult();
    rewriter.create<emitc::AssignOp>(loc, highElement, highScaled);
    return mlir::success();
  }

  return node->emitError(
      "unsupported node in exact Scalar affine computation plan");
}

mlir::LogicalResult projectAffineRegion(mlir::Block &planBlock,
                                        AffineProjectionState &state) {
  for (mlir::Operation &node : planBlock)
    if (mlir::failed(projectAffineNode(&node, state)))
      return mlir::failure();
  return mlir::success();
}

class ScalarPackedAffineDequantBodyToEmitCFunc final
    : public mlir::OpConversionPattern<
          weft::scalar::PackedAffineDequantBodyOp> {
public:
  using mlir::OpConversionPattern<
      weft::scalar::PackedAffineDequantBodyOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(weft::scalar::PackedAffineDequantBodyOp body,
                  OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto variant = body.getSelectedVariantAttr();
    auto sourceKernel = body.getSourceKernelAttr();
    auto lowerable =
        llvm::dyn_cast<weftemitc::WEFTEmitCLowerableOpInterface>(
            body.getOperation());
    auto module = body->getParentOfType<mlir::ModuleOp>();
    if (!variant || !sourceKernel || !lowerable || !module)
      return rewriter.notifyMatchFailure(
          body, "packed_affine_dequant_body lacks exact ownership or module "
                "context");

    mlir::Location loc = body.getLoc();
    mlir::MLIRContext *context = rewriter.getContext();
    mlir::Type sizeType = emitc::OpaqueType::get(context, "size_t");
    mlir::Type intType = emitc::OpaqueType::get(context, "int");
    mlir::Type floatType = emitc::OpaqueType::get(context, "float");
    mlir::Type constU8Type = emitc::OpaqueType::get(context, "const uint8_t");
    mlir::Type floatPtrType = emitc::PointerType::get(floatType);
    mlir::Type constU8PtrType = emitc::PointerType::get(constU8Type);

    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<emitc::VerbatimOp>(
          loc, "#include <stddef.h>\n#include <stdint.h>");
    }

    mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());
    std::string functionName =
        ("weft_emitc_" + sourceKernel.getValue() + "_" + variant.getValue())
            .str();
    mlir::FunctionType functionType = rewriter.getFunctionType(
        {intType, floatPtrType, constU8PtrType}, {});
    llvm::SmallVector<mlir::NamedAttribute, 1> funcAttrs;
    funcAttrs.push_back(rewriter.getNamedAttr(
        "specifiers", rewriter.getStrArrayAttr({"extern", "\"C\""})));
    auto func = rewriter.create<emitc::FuncOp>(loc, functionName, functionType,
                                               funcAttrs);
    mlir::Block *entry = func.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);

    llvm::StringRef sourceOpName =
        lowerable.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef sourceRole = lowerable.getWEFTEmitCLowerableSourceRole();
    rewriter.create<emitc::VerbatimOp>(
        loc, routeSourceComment(sourceOpName, sourceRole,
                                kEmitCLowerableInterfaceName));
    AffineProjectionState state{
        rewriter,
        loc,
        sourceOpName,
        sourceRole,
        sizeType,
        intType,
        floatType,
        constU8Type,
        entry->getArgument(0),
        entry->getArgument(1),
        entry->getArgument(2),
    };
    if (mlir::failed(projectAffineRegion(body.getBody().front(), state)))
      return rewriter.notifyMatchFailure(
          body, "failed to project exact Scalar affine computation plan");
    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
    rewriter.eraseOp(body);
    return mlir::success();
  }
};

class ScalarBackendEmissionDriver final
    : public weftemitc::TypedBackendEmissionDriver {
public:
  llvm::StringRef getBackendName() const override { return "scalar"; }

  void populateTypeConversions(
      mlir::TypeConverter & /*typeConverter*/) const override {
    // Scalar plan nodes carry no SSA dataflow values; the typed operation tree
    // itself is the exact computation plan.
  }

  void configureConversionTarget(mlir::ConversionTarget &target) const override {
    target.addIllegalOp<
        weft::scalar::ComputeSkeletonOp,
        weft::scalar::TernaryQ2Q8BlockDotOp,
        weft::scalar::DequantizeRowQ4Op,
        weft::scalar::ImmediateCallBodyOp,
        weft::scalar::PackedTernaryDotBodyOp,
        weft::scalar::TernaryBlockLoopOp,
        weft::scalar::TernaryPlaneGroupLoopOp,
        weft::scalar::TernaryPlaneLoopOp,
        weft::scalar::TernaryLaneLoopOp,
        weft::scalar::TernaryDecodeMacOp,
        weft::scalar::TernaryScaleFoldOp,
        weft::scalar::TernaryStoreOp,
        weft::scalar::PackedAffineDequantBodyOp,
        weft::scalar::AffineBlockLoopOp,
        weft::scalar::AffineBlockScaleOp,
        weft::scalar::AffineQuantBaseOp,
        weft::scalar::AffinePackedByteLoopOp,
        weft::scalar::AffineDecodeScaleScatterOp>();
    target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
  }

  void
  populateLoweringPatterns(mlir::TypeConverter &typeConverter,
                           mlir::RewritePatternSet &patterns) const override {
    patterns.add<ScalarImmediateCallBodyToEmitCFunc,
                 ScalarPackedTernaryDotBodyToEmitCFunc,
                 ScalarPackedAffineDequantBodyToEmitCFunc>(
        typeConverter, patterns.getContext());
  }

  llvm::LogicalResult postConversionCleanup(mlir::ModuleOp module) const override;

  bool moduleHasBackendBody(mlir::ModuleOp module) const override {
    bool hasScalar = false;
    module.walk([&](mlir::Operation *op) {
      if (op->getName().getDialectNamespace() ==
          weft::scalar::WEFTScalarDialect::getDialectNamespace()) {
        hasScalar = true;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    return hasScalar;
  }
};

llvm::LogicalResult
ScalarBackendEmissionDriver::postConversionCleanup(mlir::ModuleOp module) const {
  bool producedFunc = false;
  module.walk([&](emitc::FuncOp) { producedFunc = true; });
  if (!producedFunc)
    return llvm::success();

  llvm::SmallVector<mlir::Operation *, 2> drainedTopLevel;
  for (mlir::Operation &op : module.getBody()->getOperations()) {
    llvm::StringRef dialect = op.getName().getDialectNamespace();
    if (dialect != emitc::EmitCDialect::getDialectNamespace())
      drainedTopLevel.push_back(&op);
  }
  for (mlir::Operation *op : drainedTopLevel)
    op->erase();
  return llvm::success();
}

} // namespace

void registerScalarBackendEmitter(
    weftemitc::BackendEmissionRegistry &registry) {
  static const ScalarBackendEmissionDriver driver;
  registry.registerBackend(driver);
}

} // namespace scalar
} // namespace plugin
} // namespace weft
