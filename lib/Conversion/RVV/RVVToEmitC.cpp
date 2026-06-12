#include "TianChenRV/Conversion/RVV/RVVToEmitC.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Transforms/Passes.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <string>

namespace tianchenrv {
namespace transforms {

#define GEN_PASS_DEF_RVVLOWERTOEMITC
#include "TianChenRV/Transforms/Passes.h.inc"

} // namespace transforms
} // namespace tianchenrv

namespace tianchenrv {
namespace conversion {
namespace rvv {
namespace {

namespace tcrvrvv = ::tianchenrv::tcrv::rvv;
namespace emitc = ::mlir::emitc;

//===----------------------------------------------------------------------===//
// Pure C-type derivation (the typed-fact replacement for the legacy string
// re-parser's getEmitCTypeForCType): turn the runtime ABI value's `c_type`
// spelling into a concrete emitc type. Pointers nest; everything else is opaque.
//===----------------------------------------------------------------------===//

mlir::Type emitCTypeForCTypeSpelling(mlir::MLIRContext *context,
                                     llvm::StringRef cType) {
  cType = cType.trim();
  if (cType.ends_with("*")) {
    llvm::StringRef pointee = cType.drop_back().rtrim();
    return emitc::PointerType::get(context,
                                   emitCTypeForCTypeSpelling(context, pointee));
  }
  return emitc::OpaqueType::get(context, cType);
}

//===----------------------------------------------------------------------===//
// Pure SEW/LMUL/dtype intrinsic name mangler. This is the ONE legitimate
// survivor of the legacy `Twine("__riscv_...")+sew+lmul` assembly: a pure
// function over operand TYPE facts (sew/lmul/dtype) and the op mnemonic, with
// no string plan and no operand-expression concatenation. `mnemonic` is the
// RVV intrinsic verb (vsetvl/vle/vse/vadd/vsub/vmul); the caller derives it
// from the typed source op (binary `kind` -> vadd/vsub/vmul, load -> vle,
// store -> vse, setvl -> vsetvl).
//===----------------------------------------------------------------------===//

std::string riscvIntrinsicName(llvm::StringRef mnemonic, unsigned sew,
                               llvm::StringRef lmul, llvm::StringRef dtype) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "__riscv_" << mnemonic;
  if (mnemonic == "vsetvl") {
    // __riscv_vsetvl_e<sew><lmul>
    os << "_e" << sew << lmul;
  } else if (mnemonic == "vle" || mnemonic == "vse") {
    // __riscv_vle<sew>_v_<dtype><lmul> / __riscv_vse<sew>_v_<dtype><lmul>
    os << sew << "_v_" << dtype << lmul;
  } else if (mnemonic == "vlse" || mnemonic == "vsse") {
    // strided load/store: __riscv_vlse<sew>_v_<dtype><lmul> /
    // __riscv_vsse<sew>_v_<dtype><lmul>
    os << sew << "_v_" << dtype << lmul;
  } else if (mnemonic == "vmv_v_x") {
    // scalar splat: __riscv_vmv_v_x_<dtype><lmul>
    os << "_" << dtype << lmul;
  } else if (mnemonic == "vmerge") {
    // masked merge: __riscv_vmerge_vvm_<dtype><lmul>
    os << "_vvm_" << dtype << lmul;
  } else {
    // arithmetic vv form: __riscv_v<op>_vv_<dtype><lmul>
    os << "_vv_" << dtype << lmul;
  }
  os.flush();
  return name;
}

/// The compare-producing mask intrinsic name:
///   __riscv_v<cmp>_vv_<dtype><lmul>_b<maskbits>
/// where maskbits = sew/lmul-derived predicate width (i32/m1 -> b32). This is
/// the same `<dtype><lmul>` + `_b<maskbits>` shape the legacy
/// getElementwiseMaskIntrinsicSuffix produces.
std::string riscvCompareIntrinsicName(llvm::StringRef mnemonic, unsigned sew,
                                      llvm::StringRef lmul, llvm::StringRef dtype,
                                      unsigned maskBits) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "__riscv_" << mnemonic << "_vv_" << dtype << lmul << "_b" << maskBits;
  os.flush();
  return name;
}

/// Predicate mask width for an (sew, lmul) config, mirroring the legacy
/// getElementwiseMaskIntrinsicSuffix maskBits table. Returns 0 for an
/// unsupported pair (so the caller fails the match and falls back).
unsigned maskWidthForConfig(unsigned sew, llvm::StringRef lmul) {
  if (sew == 32 && lmul == "m1")
    return 32;
  if (sew == 32 && lmul == "m2")
    return 16;
  if (sew == 64 && lmul == "m1")
    return 64;
  if (sew == 64 && lmul == "m2")
    return 32;
  return 0;
}

/// The C dtype token ("i32"/"i64") for the vector element, used by the load/
/// store/arithmetic intrinsic suffix and the `vint<sew>m<lmul>_t` opaque type.
llvm::StringRef vectorDType(tcrvrvv::VectorType type) {
  if (type.getElementType().isSignlessInteger(32))
    return "i32";
  if (type.getElementType().isSignlessInteger(64))
    return "i64";
  return "";
}

//===----------------------------------------------------------------------===//
// Provenance comments, byte-identical to the legacy materializer
// (TCRVEmitCLowerableMaterializer.cpp makeRouteSourceProvenanceComment /
// makeStepProvenanceComment). Reproduced so the rendered C carries the same
// `// tcrv_emitc.*` lines and stays byte-equivalent to the hardware-validated
// golden.
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kOpInterface = "TCRVEmitCLowerableOpInterface";

std::string routeSourceComment(llvm::StringRef opName, llvm::StringRef role) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// tcrv_emitc.route_source_op=" << opName << " role=" << role
     << " op_interface=" << kOpInterface;
  os.flush();
  return text;
}

std::string stepComment(llvm::StringRef opName, llvm::StringRef role,
                        llvm::StringRef callee) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// tcrv_emitc.source_op=" << opName << " role=" << role
     << " op_interface=" << kOpInterface << " callee=" << callee;
  os.flush();
  return text;
}

//===----------------------------------------------------------------------===//
// VariantOp -> emitc.func driver.
//
// The runtime ABI values, setvl, the with_vl scope, and the body ops all live
// as siblings/children inside one tcrv.exec.variant region. The runtime ABI
// values become the C function parameters (not in-body values), so the whole
// variant body is restructured into a single top-level emitc.func with the
// with_vl scope rendered as an emitc.for loop. The per-beachhead-op emit logic
// (setvl/load/binary/store, the for-loop, the AVL `n - i` and pointer `base + i`
// arithmetic) is owned here and is sourced ENTIRELY from the typed op SSA
// Values + typed `!tcrv_rvv.*` types — never from a string plan.
//===----------------------------------------------------------------------===//

struct AbiParam {
  tcrvrvv::RuntimeABIValueOp op;
  std::string cType;
  mlir::Type emitcType;
};

class VariantToEmitCFunc final
    : public mlir::OpConversionPattern<tcrv::exec::VariantOp> {
public:
  VariantToEmitCFunc(const mlir::TypeConverter &typeConverter,
                     mlir::MLIRContext *context)
      : mlir::OpConversionPattern<tcrv::exec::VariantOp>(typeConverter,
                                                         context) {}

  mlir::LogicalResult
  matchAndRewrite(tcrv::exec::VariantOp variant, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MLIRContext *context = variant.getContext();

    // Only the selected lowering boundary (a variant carrying a with_vl scope)
    // is a beachhead body. Variants without a with_vl scope (e.g. the scalar
    // fallback) are left for the legacy path / other families.
    tcrvrvv::WithVLOp scope;
    for (mlir::Operation &op : variant.getBody().front()) {
      if (auto withVL = llvm::dyn_cast<tcrvrvv::WithVLOp>(op)) {
        scope = withVL;
        break;
      }
    }
    if (!scope)
      return rewriter.notifyMatchFailure(variant, "no with_vl scope to lower");

    tcrvrvv::SetVLOp preLoopSetVL =
        scope.getVl().getDefiningOp<tcrvrvv::SetVLOp>();
    if (!preLoopSetVL)
      return rewriter.notifyMatchFailure(
          variant, "with_vl vl token must be defined by tcrv_rvv.setvl");
    mlir::Value avlSource = preLoopSetVL.getAvl();
    auto avlAbi = avlSource.getDefiningOp<tcrvrvv::RuntimeABIValueOp>();
    if (!avlAbi)
      return rewriter.notifyMatchFailure(
          variant, "setvl avl must be a runtime ABI value");

    // Collect the runtime ABI values as ordered function parameters.
    llvm::SmallVector<AbiParam, 4> params;
    for (mlir::Operation &op : variant.getBody().front()) {
      if (auto abi = llvm::dyn_cast<tcrvrvv::RuntimeABIValueOp>(op)) {
        AbiParam param;
        param.op = abi;
        param.cType = abi.getCType().str();
        param.emitcType = emitCTypeForCTypeSpelling(context, param.cType);
        params.push_back(param);
      }
    }
    if (params.empty())
      return rewriter.notifyMatchFailure(variant, "no runtime ABI parameters");

    // Derive the function name exactly as the export path does:
    // tcrv_emitc_<kernel>_<variant>.
    auto kernel = variant->getParentOfType<tcrv::exec::KernelOp>();
    if (!kernel)
      return rewriter.notifyMatchFailure(variant, "variant has no kernel");
    std::string functionName =
        ("tcrv_emitc_" + kernel.getSymName() + "_" + variant.getSymName())
            .str();

    // Build a standalone top-level emitc module: the standard headers the RVV
    // intrinsic body needs, then the function. This mirrors the legacy
    // materializer's module shape (includes + func) so the rendered C is
    // byte-equivalent to the hardware-validated golden.
    mlir::Location loc = variant.getLoc();
    auto module = variant->getParentOfType<mlir::ModuleOp>();
    mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    for (llvm::StringRef header :
         {"stddef.h", "stdint.h", "riscv_vector.h"})
      rewriter.create<emitc::IncludeOp>(loc, header,
                                        /*is_standard_include=*/true);

    rewriter.setInsertionPointToEnd(module.getBody());

    llvm::SmallVector<mlir::Type, 4> paramTypes;
    for (const AbiParam &param : params)
      paramTypes.push_back(param.emitcType);
    mlir::FunctionType functionType =
        rewriter.getFunctionType(paramTypes, /*results=*/{});

    llvm::SmallVector<mlir::NamedAttribute, 1> funcAttrs;
    funcAttrs.push_back(rewriter.getNamedAttr(
        "specifiers", rewriter.getStrArrayAttr({"extern", "\"C\""})));
    auto func = rewriter.create<emitc::FuncOp>(loc, functionName, functionType,
                                               funcAttrs);
    mlir::Block *entry = func.addEntryBlock();

    // Map each runtime ABI value SSA result to its function block argument.
    llvm::DenseMap<mlir::Value, mlir::Value> valueMap;
    for (auto [index, param] : llvm::enumerate(params))
      valueMap[param.op.getResult()] = entry->getArgument(index);

    rewriter.setInsertionPointToStart(entry);

    // Vector/VL/AVL config facts from the typed scope and types.
    unsigned sew = static_cast<unsigned>(preLoopSetVL.getSew());
    llvm::StringRef lmul = preLoopSetVL.getLmul();
    mlir::Type sizeType = emitc::OpaqueType::get(context, "size_t");

    mlir::Value avlArg = valueMap.lookup(avlAbi.getResult());

    // Scope provenance: // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope
    rewriter.create<emitc::VerbatimOp>(
        loc, routeSourceComment(scope.getTCRVEmitCLowerableSourceOpName(),
                                scope.getTCRVEmitCLowerableSourceRole()));

    // Pre-loop full-chunk setvl: __riscv_vsetvl_e<sew><lmul>(n).
    std::string setvlCallee = riscvIntrinsicName("vsetvl", sew, lmul, "");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(preLoopSetVL.getTCRVEmitCLowerableSourceOpName(),
                         preLoopSetVL.getTCRVEmitCLowerableSourceRole(),
                         setvlCallee));
    mlir::Value vlmax =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                         setvlCallee, mlir::ValueRange{avlArg})
            .getResult(0);

    // for (size_t i = 0; i < n; i += vlmax) { ... }
    mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    auto forOp = rewriter.create<emitc::ForOp>(loc, zero, avlArg, vlmax,
                                               /*bodyBuilder=*/nullptr);
    mlir::Value inductionVar = forOp.getInductionVar();

    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());

      // Remaining-AVL setvl: size_t v = n - i; __riscv_vsetvl_e...(v).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(preLoopSetVL.getTCRVEmitCLowerableSourceOpName(),
                           preLoopSetVL.getTCRVEmitCLowerableSourceRole(),
                           setvlCallee));
      mlir::Value remaining =
          rewriter.create<emitc::SubOp>(loc, sizeType, avlArg, inductionVar);
      mlir::Value bodyVL =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                           setvlCallee,
                                           mlir::ValueRange{remaining})
              .getResult(0);

      // Convert each body op in order. Body holds the typed dataflow ops for
      // the elementwise families (plain/broadcast/splat-store/masked/strided).
      for (mlir::Operation &op : scope.getBody().front()) {
        if (auto load = llvm::dyn_cast<tcrvrvv::LoadOp>(op)) {
          if (mlir::failed(emitLoad(rewriter, loc, load, valueMap, inductionVar,
                                    bodyVL)))
            return mlir::failure();
        } else if (auto broadcast =
                       llvm::dyn_cast<tcrvrvv::BroadcastLoadOp>(op)) {
          if (mlir::failed(emitBroadcastLoad(rewriter, loc, broadcast, valueMap,
                                             bodyVL)))
            return mlir::failure();
        } else if (auto splat = llvm::dyn_cast<tcrvrvv::SplatOp>(op)) {
          if (mlir::failed(emitSplat(rewriter, loc, splat, valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto stridedLoad =
                       llvm::dyn_cast<tcrvrvv::StridedLoadOp>(op)) {
          if (mlir::failed(emitStridedLoad(rewriter, loc, stridedLoad, valueMap,
                                           inductionVar, bodyVL)))
            return mlir::failure();
        } else if (auto compare = llvm::dyn_cast<tcrvrvv::CompareOp>(op)) {
          if (mlir::failed(emitCompare(rewriter, loc, compare, valueMap,
                                       bodyVL)))
            return mlir::failure();
        } else if (auto maskedBinary =
                       llvm::dyn_cast<tcrvrvv::MaskedBinaryOp>(op)) {
          if (mlir::failed(emitMaskedBinary(rewriter, loc, maskedBinary,
                                            valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto binary = llvm::dyn_cast<tcrvrvv::BinaryOp>(op)) {
          if (mlir::failed(emitBinary(rewriter, loc, binary, valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto stridedStore =
                       llvm::dyn_cast<tcrvrvv::StridedStoreOp>(op)) {
          if (mlir::failed(emitStridedStore(rewriter, loc, stridedStore,
                                            valueMap, inductionVar, bodyVL)))
            return mlir::failure();
        } else if (auto store = llvm::dyn_cast<tcrvrvv::StoreOp>(op)) {
          if (mlir::failed(emitStore(rewriter, loc, store, valueMap,
                                     inductionVar, bodyVL)))
            return mlir::failure();
        } else {
          return rewriter.notifyMatchFailure(
              variant, "unsupported op in with_vl beachhead body");
        }
      }
    }

    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());

    rewriter.eraseOp(variant);
    return mlir::success();
  }

private:
  /// load(%abi, %vl) -> ptr = base + i; __riscv_vle<sew>_v_<dtype><lmul>(ptr, vl)
  mlir::LogicalResult
  emitLoad(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
           tcrvrvv::LoadOp load,
           llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
           mlir::Value inductionVar, mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(load.getLoaded().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(load, "load result not typed vector");
    mlir::Value base = valueMap.lookup(load.getBuffer());
    if (!base)
      return rewriter.notifyMatchFailure(load, "load buffer not an ABI param");

    // Resolve the EmitC vector type FIRST, before creating any emitc op, so a
    // family the beachhead converter does not cover (e.g. lmul m2) fails the
    // match cleanly and rolls back, instead of leaving a half-converted
    // call_opaque whose result is still a `!tcrv_rvv.vector<...>` type.
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(load, "vector type not convertible");
    std::string callee =
        riscvIntrinsicName("vle", vectorElementWidth(vectorType),
                           vectorType.getLmul(), vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(load.getTCRVEmitCLowerableSourceOpName(),
                         load.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value ptr = rewriter.create<emitc::AddOp>(loc, base.getType(), base,
                                                    inductionVar);
    mlir::Value loaded =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{vecType}, callee,
                                         mlir::ValueRange{ptr, bodyVL})
            .getResult(0);
    valueMap[load.getLoaded()] = loaded;
    return mlir::success();
  }

  /// binary{kind}(%lhs,%rhs,%vl) -> __riscv_v<op>_vv_<dtype><lmul>(lhs,rhs,vl)
  mlir::LogicalResult
  emitBinary(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
             tcrvrvv::BinaryOp binary,
             llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
             mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(binary.getResult().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(binary,
                                         "binary result not typed vector");
    mlir::Value lhs = valueMap.lookup(binary.getLhs());
    mlir::Value rhs = valueMap.lookup(binary.getRhs());
    if (!lhs || !rhs)
      return rewriter.notifyMatchFailure(binary, "binary operand unmapped");

    std::optional<llvm::StringRef> mnemonic = binaryMnemonic(binary.getKind());
    if (!mnemonic)
      return rewriter.notifyMatchFailure(binary, "unsupported binary kind");
    // Resolve the EmitC vector type FIRST (see emitLoad): a non-beachhead lmul
    // must fail the match and roll back, not emit an un-lowered result type.
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(binary,
                                         "vector type not convertible");
    std::string callee =
        riscvIntrinsicName(*mnemonic, vectorElementWidth(vectorType),
                           vectorType.getLmul(), vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(binary.getTCRVEmitCLowerableSourceOpName(),
                         binary.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value result =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{vecType}, callee,
                                         mlir::ValueRange{lhs, rhs, bodyVL})
            .getResult(0);
    valueMap[binary.getResult()] = result;
    return mlir::success();
  }

  /// store(%abi,%val,%vl) -> ptr = base + i; __riscv_vse<sew>_v_<dtype><lmul>(...)
  mlir::LogicalResult
  emitStore(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
            tcrvrvv::StoreOp store,
            llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
            mlir::Value inductionVar, mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(store.getValue().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(store, "store value not typed vector");
    mlir::Value base = valueMap.lookup(store.getBuffer());
    mlir::Value value = valueMap.lookup(store.getValue());
    if (!base || !value)
      return rewriter.notifyMatchFailure(store, "store operand unmapped");
    if (!convertVectorTypeToEmitC(vectorType))
      return rewriter.notifyMatchFailure(store, "vector type not convertible");

    std::string callee =
        riscvIntrinsicName("vse", vectorElementWidth(vectorType),
                           vectorType.getLmul(), vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(store.getTCRVEmitCLowerableSourceOpName(),
                         store.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value ptr = rewriter.create<emitc::AddOp>(loc, base.getType(), base,
                                                    inductionVar);
    rewriter.create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{}, callee,
                                         mlir::ValueRange{ptr, value, bodyVL});
    return mlir::success();
  }

  /// broadcast_load(%abi,%vl) -> scalar = base[0]; __riscv_vmv_v_x_<dtype><lmul>
  /// The legacy materializer renders the RHS broadcast operand `rhs[0]` via an
  /// emitc.subscript + emitc.load reading the first element, then splats that
  /// scalar with vmv_v_x. Reproduced exactly so the rendered C carries the
  /// `const int32_t vN = base[0];` temp + the vmv_v_x splat.
  mlir::LogicalResult
  emitBroadcastLoad(mlir::ConversionPatternRewriter &rewriter,
                    mlir::Location loc, tcrvrvv::BroadcastLoadOp broadcast,
                    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                    mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(broadcast.getBroadcast().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(broadcast,
                                         "broadcast result not typed vector");
    mlir::Value base = valueMap.lookup(broadcast.getBuffer());
    if (!base)
      return rewriter.notifyMatchFailure(broadcast,
                                         "broadcast buffer not an ABI param");
    auto pointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(base);
    if (!pointer)
      return rewriter.notifyMatchFailure(
          broadcast, "broadcast buffer must be a pointer-typed ABI param");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(broadcast,
                                         "vector type not convertible");
    std::string callee =
        riscvIntrinsicName("vmv_v_x", vectorElementWidth(vectorType),
                           vectorType.getLmul(), vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(broadcast.getTCRVEmitCLowerableSourceOpName(),
                         broadcast.getTCRVEmitCLowerableSourceRole(), callee));
    // base[0]: subscript -> lvalue -> load reads the first scalar element.
    mlir::Value index = rewriter.create<emitc::LiteralOp>(
        loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp subscriptOp =
        rewriter.create<emitc::SubscriptOp>(loc, pointer, index);
    auto lvalueType =
        llvm::cast<emitc::LValueType>(subscriptOp.getResult().getType());
    mlir::Value scalar =
        rewriter
            .create<emitc::LoadOp>(loc, lvalueType.getValueType(),
                                   subscriptOp.getResult())
            .getResult();
    mlir::Value result =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{vecType}, callee,
                                         mlir::ValueRange{scalar, bodyVL})
            .getResult(0);
    valueMap[broadcast.getBroadcast()] = result;
    return mlir::success();
  }

  /// splat(%scalar,%vl) -> __riscv_vmv_v_x_<dtype><lmul>(scalar, vl). The scalar
  /// is a runtime ABI value mapped to a function parameter directly.
  mlir::LogicalResult
  emitSplat(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
            tcrvrvv::SplatOp splat,
            llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
            mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(splat.getBroadcast().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(splat, "splat result not typed vector");
    mlir::Value scalar = valueMap.lookup(splat.getScalar());
    if (!scalar)
      return rewriter.notifyMatchFailure(splat, "splat scalar unmapped");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(splat, "vector type not convertible");
    std::string callee =
        riscvIntrinsicName("vmv_v_x", vectorElementWidth(vectorType),
                           vectorType.getLmul(), vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(splat.getTCRVEmitCLowerableSourceOpName(),
                         splat.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value result =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{vecType}, callee,
                                         mlir::ValueRange{scalar, bodyVL})
            .getResult(0);
    valueMap[splat.getBroadcast()] = result;
    return mlir::success();
  }

  /// compare(%lhs,%rhs,%vl){kind} ->
  ///   __riscv_v<cmp>_vv_<dtype><lmul>_b<maskbits>(lhs, rhs, vl) -> vbool<n>_t
  mlir::LogicalResult
  emitCompare(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
              tcrvrvv::CompareOp compare,
              llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
              mlir::Value bodyVL) const {
    auto maskType =
        llvm::dyn_cast<tcrvrvv::MaskType>(compare.getMask().getType());
    if (!maskType)
      return rewriter.notifyMatchFailure(compare, "compare result not typed mask");
    auto operandVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(compare.getLhs().getType());
    if (!operandVecType)
      return rewriter.notifyMatchFailure(compare,
                                         "compare operand not typed vector");
    mlir::Value lhs = valueMap.lookup(compare.getLhs());
    mlir::Value rhs = valueMap.lookup(compare.getRhs());
    if (!lhs || !rhs)
      return rewriter.notifyMatchFailure(compare, "compare operand unmapped");
    std::optional<llvm::StringRef> mnemonic = compareMnemonic(compare.getKind());
    if (!mnemonic)
      return rewriter.notifyMatchFailure(compare, "unsupported compare kind");
    mlir::Type maskEmitCType = getTypeConverter()->convertType(maskType);
    if (!maskEmitCType ||
        maskEmitCType.getDialect().getNamespace() !=
            emitc::EmitCDialect::getDialectNamespace())
      return rewriter.notifyMatchFailure(compare, "mask type not convertible");
    unsigned sew = vectorElementWidth(operandVecType);
    unsigned maskBits = maskWidthForConfig(sew, operandVecType.getLmul());
    if (maskBits == 0)
      return rewriter.notifyMatchFailure(compare, "unsupported mask config");
    std::string callee = riscvCompareIntrinsicName(
        *mnemonic, sew, operandVecType.getLmul(),
        vectorDType(operandVecType), maskBits);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(compare.getTCRVEmitCLowerableSourceOpName(),
                         compare.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value result =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{maskEmitCType},
                                         callee,
                                         mlir::ValueRange{lhs, rhs, bodyVL})
            .getResult(0);
    valueMap[compare.getMask()] = result;
    return mlir::success();
  }

  /// masked_binary(%mask,%passthrough,%lhs,%rhs,%vl){kind} lowers to TWO calls,
  /// byte-identical to the legacy masked merge sequence:
  ///   active = __riscv_v<op>_vv_<dtype><lmul>(lhs, rhs, vl);
  ///   result = __riscv_vmerge_vvm_<dtype><lmul>(passthrough, active, mask, vl);
  mlir::LogicalResult
  emitMaskedBinary(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                   tcrvrvv::MaskedBinaryOp masked,
                   llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                   mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(masked.getResult().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(masked,
                                         "masked result not typed vector");
    mlir::Value passthrough = valueMap.lookup(masked.getPassthrough());
    mlir::Value lhs = valueMap.lookup(masked.getLhs());
    mlir::Value rhs = valueMap.lookup(masked.getRhs());
    mlir::Value mask = valueMap.lookup(masked.getMask());
    if (!passthrough || !lhs || !rhs || !mask)
      return rewriter.notifyMatchFailure(masked, "masked operand unmapped");
    std::optional<llvm::StringRef> mnemonic = binaryMnemonic(masked.getKind());
    if (!mnemonic)
      return rewriter.notifyMatchFailure(masked, "unsupported masked kind");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(masked, "vector type not convertible");
    unsigned sew = vectorElementWidth(vectorType);
    llvm::StringRef lmul = vectorType.getLmul();
    llvm::StringRef dtype = vectorDType(vectorType);

    // active = vadd/vsub/vmul over the two operand vectors (unmasked).
    std::string arithCallee = riscvIntrinsicName(*mnemonic, sew, lmul, dtype);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(masked.getTCRVEmitCLowerableSourceOpName(),
                         masked.getTCRVEmitCLowerableSourceRole(), arithCallee));
    mlir::Value active =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{vecType},
                                         arithCallee,
                                         mlir::ValueRange{lhs, rhs, bodyVL})
            .getResult(0);

    // result = vmerge(passthrough, active, mask, vl) -- inactive lanes keep the
    // passthrough vector.
    std::string mergeCallee = riscvIntrinsicName("vmerge", sew, lmul, dtype);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(masked.getTCRVEmitCLowerableSourceOpName(),
                         masked.getTCRVEmitCLowerableSourceRole(), mergeCallee));
    mlir::Value result =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{vecType}, mergeCallee,
                mlir::ValueRange{passthrough, active, mask, bodyVL})
            .getResult(0);
    valueMap[masked.getResult()] = result;
    return mlir::success();
  }

  /// strided_load(%abi,%stride,%vl) ->
  ///   off = i * stride; ptr = base + off;
  ///   bytestride = (ptrdiff_t)stride * (ptrdiff_t)4;
  ///   __riscv_vlse<sew>_v_<dtype><lmul>(ptr, bytestride, vl)
  mlir::LogicalResult
  emitStridedLoad(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                  tcrvrvv::StridedLoadOp load,
                  llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                  mlir::Value inductionVar, mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(load.getLoaded().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(load, "strided load result not vector");
    mlir::Value base = valueMap.lookup(load.getBuffer());
    mlir::Value stride = valueMap.lookup(load.getStride());
    if (!base || !stride)
      return rewriter.notifyMatchFailure(load, "strided load operand unmapped");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(load, "vector type not convertible");
    std::string callee =
        riscvIntrinsicName("vlse", vectorElementWidth(vectorType),
                           vectorType.getLmul(), vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(load.getTCRVEmitCLowerableSourceOpName(),
                         load.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value ptr =
        emitScaledPointer(rewriter, loc, base, inductionVar, stride);
    mlir::Value byteStride = emitByteStride(rewriter, loc, stride, vectorType);
    mlir::Value loaded =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{vecType}, callee,
                                         mlir::ValueRange{ptr, byteStride,
                                                          bodyVL})
            .getResult(0);
    valueMap[load.getLoaded()] = loaded;
    return mlir::success();
  }

  /// strided_store(%abi,%val,%stride,%vl) ->
  ///   off = i * stride; ptr = base + off;
  ///   bytestride = (ptrdiff_t)stride * (ptrdiff_t)4;
  ///   __riscv_vsse<sew>_v_<dtype><lmul>(ptr, bytestride, val, vl)
  mlir::LogicalResult
  emitStridedStore(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                   tcrvrvv::StridedStoreOp store,
                   llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                   mlir::Value inductionVar, mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(store.getValue().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(store, "strided store value not vector");
    mlir::Value base = valueMap.lookup(store.getBuffer());
    mlir::Value value = valueMap.lookup(store.getValue());
    mlir::Value stride = valueMap.lookup(store.getStride());
    if (!base || !value || !stride)
      return rewriter.notifyMatchFailure(store, "strided store operand unmapped");
    if (!convertVectorTypeToEmitC(vectorType))
      return rewriter.notifyMatchFailure(store, "vector type not convertible");
    std::string callee =
        riscvIntrinsicName("vsse", vectorElementWidth(vectorType),
                           vectorType.getLmul(), vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(store.getTCRVEmitCLowerableSourceOpName(),
                         store.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value ptr =
        emitScaledPointer(rewriter, loc, base, inductionVar, stride);
    mlir::Value byteStride = emitByteStride(rewriter, loc, stride, vectorType);
    rewriter.create<emitc::CallOpaqueOp>(
        loc, mlir::TypeRange{}, callee,
        mlir::ValueRange{ptr, byteStride, value, bodyVL});
    return mlir::success();
  }

  /// Scaled element pointer: off = induction * stride; ptr = base + off.
  /// Mirrors the legacy materializer parseScaledPointerExpression path
  /// (`base + (induction * stride)`): an emitc.mul of the size_t induction and
  /// stride, then an emitc.add onto the pointer base.
  mlir::Value emitScaledPointer(mlir::ConversionPatternRewriter &rewriter,
                                mlir::Location loc, mlir::Value base,
                                mlir::Value inductionVar,
                                mlir::Value stride) const {
    mlir::Value scaledOffset = rewriter.create<emitc::MulOp>(
        loc, inductionVar.getType(), inductionVar, stride);
    return rewriter.create<emitc::AddOp>(loc, base.getType(), base,
                                         scaledOffset);
  }

  /// Byte stride: (ptrdiff_t)stride * (ptrdiff_t)4. Mirrors the legacy
  /// materializer parseSimpleProductExpression path for `stride * 4` with
  /// cType "ptrdiff_t": cast the runtime stride and the element-size literal to
  /// ptrdiff_t, then multiply.
  mlir::Value emitByteStride(mlir::ConversionPatternRewriter &rewriter,
                             mlir::Location loc, mlir::Value stride,
                             tcrvrvv::VectorType vectorType) const {
    mlir::Type ptrdiffType =
        emitc::OpaqueType::get(rewriter.getContext(), "ptrdiff_t");
    unsigned byteWidth = vectorElementWidth(vectorType) / 8;
    mlir::Value strideCast =
        rewriter.create<emitc::CastOp>(loc, ptrdiffType, stride);
    mlir::Value sizeLiteral = rewriter.create<emitc::LiteralOp>(
        loc, stride.getType(), llvm::Twine(byteWidth).str());
    mlir::Value sizeCast =
        rewriter.create<emitc::CastOp>(loc, ptrdiffType, sizeLiteral);
    return rewriter.create<emitc::MulOp>(loc, ptrdiffType, strideCast, sizeCast);
  }

  static unsigned vectorElementWidth(tcrvrvv::VectorType type) {
    if (auto intType =
            llvm::dyn_cast<mlir::IntegerType>(type.getElementType()))
      return intType.getWidth();
    return 0;
  }

  /// Convert a `!tcrv_rvv.vector<...>` to its EmitC type, but ONLY accept a
  /// result the beachhead converter genuinely lowered (an `emitc` type). The
  /// driver registers an identity fallback conversion so unrelated IR is never
  /// illegalized; that identity would otherwise pass an unhandled vector type
  /// (e.g. lmul m2) straight through, letting a half-converted call_opaque keep
  /// a `!tcrv_rvv.vector<...>` result and silently corrupt the module. Rejecting
  /// any non-emitc result here makes a non-beachhead family fail the match and
  /// roll back cleanly, so the export seam falls back to the legacy path.
  mlir::Type convertVectorTypeToEmitC(tcrvrvv::VectorType type) const {
    mlir::Type converted = getTypeConverter()->convertType(type);
    if (!converted)
      return nullptr;
    if (converted.getDialect().getNamespace() !=
        mlir::emitc::EmitCDialect::getDialectNamespace())
      return nullptr;
    return converted;
  }

  static std::optional<llvm::StringRef>
  binaryMnemonic(llvm::StringRef kind) {
    if (kind == "add")
      return llvm::StringRef("vadd");
    if (kind == "sub")
      return llvm::StringRef("vsub");
    if (kind == "mul")
      return llvm::StringRef("vmul");
    return std::nullopt;
  }

  static std::optional<llvm::StringRef>
  compareMnemonic(llvm::StringRef kind) {
    if (kind == "eq")
      return llvm::StringRef("vmseq");
    return std::nullopt;
  }
};

} // namespace

void populateRVVToEmitCTypeConversions(mlir::TypeConverter &typeConverter) {
  // !tcrv_rvv.vl -> emitc.opaque<"size_t"> (the RVV vector-length token is the
  // C size_t produced by __riscv_vsetvl_*).
  typeConverter.addConversion(
      [](tcrvrvv::VLType type) -> std::optional<mlir::Type> {
        return emitc::OpaqueType::get(type.getContext(), "size_t");
      });

  // !tcrv_rvv.vector<i<sew>, "m<lmul>"> -> emitc.opaque<"vint<sew>m<lmul>_t">.
  // The elementwise family covers the bounded {i32,i64} x {m1,m2} grid the
  // selected-body rungs use (e.g. i32/m1 -> vint32m1_t, i64/m1 -> vint64m1_t,
  // i32/m2 -> vint32m2_t, i64/m2 -> vint64m2_t). Other (dtype, lmul) pairs are
  // left unconverted on purpose so later families extend the grid explicitly.
  typeConverter.addConversion(
      [](tcrvrvv::VectorType type) -> std::optional<mlir::Type> {
        unsigned sew = 0;
        if (type.getElementType().isSignlessInteger(32))
          sew = 32;
        else if (type.getElementType().isSignlessInteger(64))
          sew = 64;
        else
          return std::nullopt;
        llvm::StringRef lmul = type.getLmul();
        if (lmul != "m1" && lmul != "m2")
          return std::nullopt;
        std::string name = ("vint" + llvm::Twine(sew) + lmul + "_t").str();
        return emitc::OpaqueType::get(type.getContext(), name);
      });

  // !tcrv_rvv.mask<i<sew>, "m<lmul>"> -> emitc.opaque<"vbool<maskbits>_t">. The
  // predicate mask C type is vbool<maskbits>_t where maskbits = SEW/LMUL_ratio
  // (the sew/lmul-derived width matching maskWidthForConfig): i32/m1 -> 32,
  // i64/m1 -> 64, i32/m2 -> 16, i64/m2 -> 32. Pairs maskWidthForConfig does not
  // know are left unconverted so the masked converter stays scoped to the grid.
  typeConverter.addConversion(
      [](tcrvrvv::MaskType type) -> std::optional<mlir::Type> {
        unsigned sew = 0;
        if (type.getElementType().isSignlessInteger(32))
          sew = 32;
        else if (type.getElementType().isSignlessInteger(64))
          sew = 64;
        else
          return std::nullopt;
        unsigned maskBits = maskWidthForConfig(sew, type.getLmul());
        if (maskBits == 0)
          return std::nullopt;
        std::string name = ("vbool" + llvm::Twine(maskBits) + "_t").str();
        return emitc::OpaqueType::get(type.getContext(), name);
      });

  // !tcrv_rvv.runtime_abi_value carries its concrete C type in the defining
  // op's c_type attribute (e.g. "const int32_t *", "int32_t *", "size_t"),
  // which a pure type-keyed conversion cannot recover. The VariantToEmitCFunc
  // pattern derives the function parameter type from that attr directly; the
  // token type itself maps to an opaque placeholder so it never blocks
  // legalization.
  typeConverter.addConversion(
      [](tcrvrvv::RuntimeABIValueType type) -> std::optional<mlir::Type> {
        return emitc::OpaqueType::get(type.getContext(), "void");
      });
}

void populateRVVElementwiseToEmitCPatterns(mlir::TypeConverter &typeConverter,
                                           mlir::RewritePatternSet &patterns) {
  patterns.add<VariantToEmitCFunc>(typeConverter, patterns.getContext());
}

bool convertRVVModuleToEmitC(mlir::ModuleOp module) {
  mlir::MLIRContext *context = module.getContext();

  // The conversion patterns construct emitc ops/types. When this driver runs
  // outside the pass framework (the live artifact-export materialization seam),
  // the EmitC dialect is only registered, not loaded, in the translate context
  // -- so eagerly load it here. This mirrors the `--tcrv-rvv-lower-to-emitc`
  // pass's `dependentDialects` and makes the driver self-sufficient for both
  // callers. Loading is idempotent.
  context->loadDialect<mlir::emitc::EmitCDialect>();

  mlir::TypeConverter typeConverter;
  // Identity for any type the beachhead conversions do not rewrite, so the
  // pass never illegalizes unrelated IR.
  typeConverter.addConversion([](mlir::Type type) { return type; });
  populateRVVToEmitCTypeConversions(typeConverter);

  mlir::ConversionTarget target(*context);
  // emitc is the lowering destination dialect.
  target.addLegalDialect<mlir::emitc::EmitCDialect>();
  // A tcrv.exec.variant that carries a tcrv_rvv.with_vl selected-lowering
  // boundary is illegal and must be converted into an emitc.func. Variants
  // without a with_vl scope (e.g. scalar fallbacks) stay legal so unconverted
  // families fall through unchanged.
  target.addDynamicallyLegalOp<tcrv::exec::VariantOp>(
      [](tcrv::exec::VariantOp variant) {
        for (mlir::Operation &op : variant.getBody().front())
          if (llvm::isa<tcrv::rvv::WithVLOp>(op))
            return false;
        return true;
      });
  // Everything else (kernels, dispatch, capabilities, other dialects) stays
  // legal; the beachhead conversion rewrites the variant subtree atomically.
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });

  mlir::RewritePatternSet patterns(context);
  populateRVVElementwiseToEmitCPatterns(typeConverter, patterns);

  if (mlir::failed(
          mlir::applyPartialConversion(module, target, std::move(patterns))))
    return false;

  // The beachhead conversion lowers the selected variant body into a
  // standalone emitc.func + headers. Once a function was produced, drop the
  // now-emptied tcrv.exec scaffolding (kernel/capability/dispatch) for that
  // kernel so the module is a clean, translatable EmitC module matching the
  // legacy materializer's output shape. Kernels that still carry a tcrv_rvv
  // body (unconverted families) are left untouched.
  bool producedFunc = false;
  module.walk([&](mlir::emitc::FuncOp) { producedFunc = true; });
  if (producedFunc) {
    llvm::SmallVector<tcrv::exec::KernelOp, 1> drainedKernels;
    module.walk([&](tcrv::exec::KernelOp kernel) {
      bool hasRVVBody = false;
      kernel.walk([&](tcrv::rvv::WithVLOp) { hasRVVBody = true; });
      if (!hasRVVBody)
        drainedKernels.push_back(kernel);
    });
    for (tcrv::exec::KernelOp kernel : drainedKernels)
      kernel.erase();
  }

  // Strangler-fig gate: report a FULL legalization only when no RVV dataflow op,
  // no `tcrv_rvv` leftover TYPE (a half-converted op whose operand/result still
  // carries `!tcrv_rvv.*`), and no unrealized_conversion_cast survives. A
  // partial/failed conversion (any leftover) returns false so the export seam
  // falls back to the legacy string path unchanged.
  auto carriesRVVType = [](mlir::Operation *op) {
    auto isRVV = [](mlir::Type type) {
      return type.getDialect().getNamespace() ==
             tcrvrvv::TCRVRVVDialect::getDialectNamespace();
    };
    return llvm::any_of(op->getOperandTypes(), isRVV) ||
           llvm::any_of(op->getResultTypes(), isRVV);
  };
  bool fullyConverted = true;
  module.walk([&](mlir::Operation *op) {
    if (op->getName().getDialectNamespace() ==
            tcrvrvv::TCRVRVVDialect::getDialectNamespace() ||
        llvm::isa<mlir::UnrealizedConversionCastOp>(op) || carriesRVVType(op)) {
      fullyConverted = false;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return fullyConverted;
}

} // namespace rvv
} // namespace conversion
} // namespace tianchenrv

namespace tianchenrv {
namespace transforms {

namespace {

class RVVLowerToEmitCPass final
    : public impl::RVVLowerToEmitCBase<RVVLowerToEmitCPass> {
public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Run the single shared conversion driver (the same one the live
    // artifact-export materialization seam calls). It runs the
    // TypeConverter/ConversionTarget/patterns + applyPartialConversion and
    // drains the emptied tcrv.exec scaffolding for converted kernels.
    if (conversion::rvv::convertRVVModuleToEmitC(module))
      return;

    // The driver returns false either for a clean structural no-op (an
    // unconverted family whose ops the target keeps legal and which the patterns
    // leave untouched) or for a real conversion failure that left an illegal
    // with_vl-carrying variant behind. Only the latter is a pass failure.
    bool unlegalizedScopeRemains = false;
    module.walk([&](tcrv::rvv::WithVLOp) {
      unlegalizedScopeRemains = true;
      return mlir::WalkResult::interrupt();
    });
    if (unlegalizedScopeRemains)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createRVVLowerToEmitCPass() {
  return std::make_unique<RVVLowerToEmitCPass>();
}

} // namespace transforms
} // namespace tianchenrv
