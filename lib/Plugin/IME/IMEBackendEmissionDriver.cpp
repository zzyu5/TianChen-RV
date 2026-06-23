#include "TianChenRV/Plugin/IME/IMEBackendEmissionDriver.h"

#include "TianChenRV/Conversion/EmitC/BackendEmissionRegistry.h"
#include "TianChenRV/Conversion/EmitC/TypedBackendEmissionDriver.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/IME/IR/IMEDialect.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace tianchenrv {
namespace plugin {
namespace ime {

namespace {

namespace emitc = ::mlir::emitc;
namespace tcrvemitc = ::tianchenrv::conversion::emitc;

constexpr llvm::StringLiteral kOpInterface = "TCRVEmitCLowerableOpInterface";
// The self-contained asm-leaf helper name. ALL dataflow is structured emitc;
// this helper holds the SINGLE justified instruction leaf (the `vmadot` asm) —
// there is no IME intrinsic header, so per the design note we claim ONE leaf,
// not raw()==0. Reached only by a structured emitc.call_opaque on the A/B/C
// pointer block-args of the wrapping emitc.func (no SSA name is interpolated
// into the asm text; the helper has its own fixed parameter names).
constexpr llvm::StringLiteral kVmadotHelperName("tcrv_ime_vmadot_mma_4x4x8");

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

/// The FOUNDATION-validated (real-K1 bit-exact, FOUNDATION.md task 3) int8->int32
/// `vmadot` MAC kernel, emitted verbatim as ONE self-contained `static inline`
/// helper. This is the single justified asm leaf: there is no IME intrinsic
/// header, so the instruction leaf is unavoidably one asm block. The helper
/// signature is fixed (its own parameter names), so the asm never references any
/// translator-generated SSA name — the dataflow into it is structured.
///
///   A: (M,K)=(4,8) int8 row-major   -> vs1 = v0
///   B: stored (N,K)=(4,8) int8       -> vs2 = v1  (== B^T of the math matrix)
///   C: (M,N)=(4,4) int32 (even VD pair v2/v3), C += A . B_stored^T
std::string vmadotHelperBody() {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// tcrv_ime.asm_leaf=" << kVmadotHelperName
     << " mac=4x4x8 elem_in=int8 accum=int32 ime_op=vmadot\n";
  os << "static inline void " << kVmadotHelperName
     << "(const int8_t *A, const int8_t *B, int32_t *C) {\n";
  os << "  __asm__ volatile(\n";
  os << "      \"vsetvli   t0, zero, e8, m1, ta, ma   \\n\\t\"\n";
  os << "      \"vle8.v    v0, (%[pa])                \\n\\t\"\n";
  os << "      \"vle8.v    v1, (%[pb])                \\n\\t\"\n";
  os << "      \"vmv.v.i   v2, 0                      \\n\\t\"\n";
  os << "      \"vmv.v.i   v3, 0                      \\n\\t\"\n";
  os << "      \"vmadot    v2, v0, v1                 \\n\\t\"\n";
  os << "      \"vsetvli   t0, zero, e32, m1, ta, ma  \\n\\t\"\n";
  os << "      \"vse32.v   v2, (%[pc])                \\n\\t\"\n";
  os << "      \"addi      t1, %[pc], 32              \\n\\t\"\n";
  os << "      \"vse32.v   v3, (t1)                   \\n\\t\"\n";
  os << "      :\n";
  os << "      : [pa] \"r\"(A), [pb] \"r\"(B), [pc] \"r\"(C)\n";
  os << "      : \"t0\", \"t1\", \"v0\", \"v1\", \"v2\", \"v3\", \"memory\");\n";
  os << "}";
  os.flush();
  return text;
}

/// Lowers a selected `tcrv.ime.mma` boundary into a standalone EmitC module:
///   #include <stdint.h>
///   static inline void tcrv_ime_vmadot_mma_4x4x8(const int8_t*, const int8_t*,
///                                                 int32_t*) { __asm__(...vmadot...) }
///   extern "C" void tcrv_emitc_<kernel>_<variant>(const int8_t *A,
///                                                  const int8_t *B,
///                                                  int32_t *C) {
///     // route_source_op + source_op provenance comments
///     tcrv_ime_vmadot_mma_4x4x8(A, B, C);
///   }
/// The wrapper is structured emitc (emitc.func + emitc.call_opaque on the A/B/C
/// block args); the single asm leaf is confined to the helper.
class IMEMMAToEmitCFunc final
    : public mlir::OpConversionPattern<tcrv::ime::MMAOp> {
public:
  using mlir::OpConversionPattern<tcrv::ime::MMAOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tcrv::ime::MMAOp mma, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MLIRContext *context = mma.getContext();
    mlir::Location loc = mma.getLoc();

    auto variant =
        mma->getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
    auto sourceKernel =
        mma->getAttrOfType<mlir::StringAttr>("source_kernel");
    if (!variant || !sourceKernel)
      return rewriter.notifyMatchFailure(
          mma, "tcrv.ime.mma requires selected_variant and source_kernel "
               "attributes");
    std::string functionName =
        ("tcrv_emitc_" + sourceKernel.getValue() + "_" + variant.getValue())
            .str();

    llvm::StringRef sourceOpName = mma.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef sourceRole = mma.getTCRVEmitCLowerableSourceRole();

    auto module = mma->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(mma, "tcrv.ime.mma has no module");

    auto i8PtrType = emitc::PointerType::get(
        context, emitc::OpaqueType::get(context, "const int8_t"));
    auto i32PtrType = emitc::PointerType::get(
        context, emitc::OpaqueType::get(context, "int32_t"));

    // Module-scope prologue: include + the self-contained asm-leaf helper.
    {
      mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<emitc::IncludeOp>(loc, "stdint.h",
                                        /*is_standard_include=*/true);
      rewriter.create<emitc::VerbatimOp>(loc, vmadotHelperBody());
    }

    mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());

    // Exported wrapper: extern "C" void <name>(const int8_t*, const int8_t*,
    // int32_t*).
    llvm::SmallVector<mlir::Type, 3> paramTypes{i8PtrType, i8PtrType,
                                                i32PtrType};
    mlir::FunctionType functionType =
        rewriter.getFunctionType(paramTypes, /*results=*/{});
    llvm::SmallVector<mlir::NamedAttribute, 1> funcAttrs;
    funcAttrs.push_back(rewriter.getNamedAttr(
        "specifiers", rewriter.getStrArrayAttr({"extern", "\"C\""})));
    auto func = rewriter.create<emitc::FuncOp>(loc, functionName, functionType,
                                               funcAttrs);
    mlir::Block *entry = func.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);

    rewriter.create<emitc::VerbatimOp>(
        loc, routeSourceComment(sourceOpName, sourceRole));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(sourceOpName, sourceRole, kVmadotHelperName));

    // Structured dataflow into the leaf: call_opaque on the A/B/C block args.
    llvm::SmallVector<mlir::Value, 3> callOperands;
    for (mlir::BlockArgument arg : entry->getArguments())
      callOperands.push_back(arg);
    rewriter.create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{},
                                         kVmadotHelperName, callOperands);

    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());

    rewriter.eraseOp(mma);
    return mlir::success();
  }
};

class IMEBackendEmissionDriver final
    : public tcrvemitc::TypedBackendEmissionDriver {
public:
  llvm::StringRef getBackendName() const override { return "ime"; }

  void populateTypeConversions(
      mlir::TypeConverter & /*typeConverter*/) const override {}

  void
  configureConversionTarget(mlir::ConversionTarget &target) const override {
    target.addIllegalOp<tcrv::ime::MMAOp>();
    target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
  }

  void
  populateLoweringPatterns(mlir::TypeConverter &typeConverter,
                           mlir::RewritePatternSet &patterns) const override {
    patterns.add<IMEMMAToEmitCFunc>(typeConverter, patterns.getContext());
  }

  llvm::LogicalResult
  postConversionCleanup(mlir::ModuleOp module) const override;

  bool moduleHasBackendBody(mlir::ModuleOp module) const override {
    bool hasIME = false;
    module.walk([&](mlir::Operation *op) {
      if (op->getName().getDialectNamespace() ==
          tcrv::ime::TCRVIMEDialect::getDialectNamespace()) {
        hasIME = true;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    return hasIME;
  }
};

llvm::LogicalResult
IMEBackendEmissionDriver::postConversionCleanup(mlir::ModuleOp module) const {
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

void registerIMEBackendEmitter(tcrvemitc::BackendEmissionRegistry &registry) {
  static const IMEBackendEmissionDriver driver;
  registry.registerBackend(driver);
}

} // namespace ime
} // namespace plugin
} // namespace tianchenrv
