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
// The SECOND (unsigned) asm-leaf helper — same structure, but the single
// justified instruction leaf is `vmadotu` (unsigned*unsigned int8 MAC), a
// GENUINELY different instruction (encoding 0xe210022b vs vmadot's 0xe210312b)
// with unsigned numeric semantics. There is no IME intrinsic header, so this is
// likewise ONE justified verbatim leaf reached by a structured call_opaque.
constexpr llvm::StringLiteral kVmadotuHelperName("tcrv_ime_vmadotu_mma_4x4x8");
// The FOURTH (mixed-sign) asm-leaf helper — same structure, but the single
// justified instruction leaf is `vmadotsu` (signed*unsigned int8 MAC), a
// GENUINELY different instruction (encoding 0xe210232b vs vmadot's 0xe210312b
// and vmadotu's 0xe210022b) with mixed-sign numeric semantics (signed A,
// unsigned B). There is no IME intrinsic header, so this is likewise ONE
// justified verbatim leaf reached by a structured call_opaque.
constexpr llvm::StringLiteral kVmadotsuHelperName(
    "tcrv_ime_vmadotsu_mma_4x4x8");

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
/// IME MAC kernel, emitted verbatim as ONE self-contained `static inline` helper.
/// This is the single justified asm leaf: there is no IME intrinsic header, so
/// the instruction leaf is unavoidably one asm block. The helper signature is
/// fixed (its own parameter names), so the asm never references any
/// translator-generated SSA name — the dataflow into it is structured.
///
///   A: (M,K)=(4,8) int8 row-major   -> vs1 = v0
///   B: stored (N,K)=(4,8) int8       -> vs2 = v1  (== B^T of the math matrix)
///   C: (M,N)=(4,4) int32 (even VD pair v2/v3), C += A . B_stored^T
///
/// `helperName` / `mnemonic` parameterize the ONLY two things that differ
/// between the signed (vmadot) and unsigned (vmadotu) MAC: the helper name and
/// the single instruction leaf. Everything structural (load/store/clear) is
/// identical, so the signed vs unsigned divergence is exactly the instruction.
std::string macHelperBody(llvm::StringRef helperName, llvm::StringRef mnemonic) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// tcrv_ime.asm_leaf=" << helperName
     << " mac=4x4x8 elem_in=int8 accum=int32 ime_op=" << mnemonic << "\n";
  os << "static inline void " << helperName
     << "(const int8_t *A, const int8_t *B, int32_t *C) {\n";
  os << "  __asm__ volatile(\n";
  os << "      \"vsetvli   t0, zero, e8, m1, ta, ma   \\n\\t\"\n";
  os << "      \"vle8.v    v0, (%[pa])                \\n\\t\"\n";
  os << "      \"vle8.v    v1, (%[pb])                \\n\\t\"\n";
  os << "      \"vmv.v.i   v2, 0                      \\n\\t\"\n";
  os << "      \"vmv.v.i   v3, 0                      \\n\\t\"\n";
  os << "      \"" << mnemonic << "    v2, v0, v1                 \\n\\t\"\n";
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

std::string vmadotHelperBody() {
  return macHelperBody(kVmadotHelperName, "vmadot");
}

std::string vmadotuHelperBody() {
  return macHelperBody(kVmadotuHelperName, "vmadotu");
}

std::string vmadotsuHelperBody() {
  return macHelperBody(kVmadotsuHelperName, "vmadotsu");
}

// The tiled whole-matrix micro-kernel helper names. The single justified asm
// leaf is the same `vmadot`/`vmadotu` instruction, here driven IN-REGISTER over
// the whole K reduction so the per-output reduce (RVV's vredsum) is
// structurally absent — that is exactly the IME advantage the bench measures.
constexpr llvm::StringLiteral kMatmulHelperName("tcrv_ime_vmadot_matmul");
constexpr llvm::StringLiteral kMatmulUHelperName("tcrv_ime_vmadotu_matmul");

/// The tiled int8->int32 whole-matrix kernel, emitted as ONE self-contained
/// `static inline` helper. It computes C[M,N] += A[M,K] . B[K,N] by looping the
/// FOUNDATION-validated 4x4x8 `vmadot` fragment over the (M/4)x(N/4)x(K/8) tile
/// grid. The inputs are PRE-PACKED into FRAGMENT-MAJOR tile-contiguous layout
/// (the same repack policy applied to BOTH the IME and the RVV baseline; weight
/// repack is amortized offline in e2e), so each 4x8 fragment of A and B is a
/// contiguous 32-byte `vle8` — exactly the FOUNDATION load (`vle8 v0,(A)` over a
/// 4x8 row-major tile loads [r0(8),r1(8),r2(8),r3(8)] = one fragment):
///   Apack: (M/4) row-tiles. Tile mi = K/8 fragments concatenated; fragment f =
///          [Arow0[8f..8f+7], Arow1[..], Arow2[..], Arow3[..]] (32B).
///   Bpack: (N/4) col-tiles, same fragment-major layout over B^T's 4 rows.
///   C:     (M,N) int32 row-major.
/// Per output tile (mi,nj), the 4x4 int32 accumulator lives in the v2/v3 VD
/// pair across the WHOLE K loop (vmadot accumulates C += A.B^T in-register), so
/// the per-output reduce RVV needs (vredsum) is structurally absent. The result
/// is stored ONCE to a contiguous 16-int32 scratch (v2 = rows 0,1; v3 = rows
/// 2,3 — exactly FOUNDATION's validated store), then the structured C wrapper
/// scatters it into the N-strided C tile. The signed/unsigned divergence is
/// exactly the instruction mnemonic; everything structural is identical. The
/// helper has fixed parameter names, so no translator-generated SSA name is
/// interpolated into the asm.
std::string matmulHelperBody(llvm::StringRef helperName,
                             llvm::StringRef mnemonic) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// tcrv_ime.asm_leaf=" << helperName
     << " tiled_matmul mac=4x4x8 elem_in=int8 accum=int32 ime_op=" << mnemonic
     << " in_register_K_accumulate=1\n";
  os << "static inline void " << helperName
     << "(const int8_t *Apack, const int8_t *Bpack, int32_t *C,\n";
  os << "    long M, long N, long K) {\n";
  os << "  const long mt = M / 4, nt = N / 4, kt = K / 8;\n";
  os << "  for (long mi = 0; mi < mt; ++mi) {\n";
  // Apack row-tile (mi): K/8 fragments of 32B, contiguous = mi*4*K bytes in.
  os << "    const int8_t *Atile = Apack + (long)mi * 4 * K;\n";
  os << "    for (long nj = 0; nj < nt; ++nj) {\n";
  // Bpack col-tile (nj): K/8 fragments of 32B (B^T col-tile), contiguous.
  os << "      const int8_t *Btile = Bpack + (long)nj * 4 * K;\n";
  // 16-int32 scratch for the single per-tile store (rows 0,1 in v2; 2,3 in v3),
  // exactly the FOUNDATION store shape. The strided scatter into C is below, in
  // STRUCTURED C (not asm), keeping the asm leaf the validated store.
  os << "      int32_t Cs[16];\n";
  os << "      __asm__ volatile(\n";
  // Clear the 4x4 int32 accumulator (v2/v3) once for this output tile.
  os << "          \"vsetvli   t0, zero, e32, m1, ta, ma  \\n\\t\"\n";
  os << "          \"vmv.v.i   v2, 0                      \\n\\t\"\n";
  os << "          \"vmv.v.i   v3, 0                      \\n\\t\"\n";
  os << "          \"vsetvli   t0, zero, e8, m1, ta, ma   \\n\\t\"\n";
  os << "          \"mv        t1, %[ka]                  \\n\\t\"\n";
  os << "          \"mv        t2, %[kb]                  \\n\\t\"\n";
  os << "          \"mv        t3, %[kt]                  \\n\\t\"\n";
  // K loop: each iter consumes one 4x8 A fragment + one 4x8 B fragment (32B
  // each), accumulating into v2/v3. Pointers advance by 32 bytes per fragment.
  os << "          \"1:                                  \\n\\t\"\n";
  os << "          \"vle8.v    v0, (t1)                   \\n\\t\"\n";
  os << "          \"vle8.v    v1, (t2)                   \\n\\t\"\n";
  os << "          \"" << mnemonic << "    v2, v0, v1                 \\n\\t\"\n";
  os << "          \"addi      t1, t1, 32                 \\n\\t\"\n";
  os << "          \"addi      t2, t2, 32                 \\n\\t\"\n";
  os << "          \"addi      t3, t3, -1                 \\n\\t\"\n";
  os << "          \"bnez      t3, 1b                     \\n\\t\"\n";
  // Store the 4x4 int32 result ONCE to the contiguous scratch (FOUNDATION
  // store: v2 -> Cs[0..7] rows 0,1; v3 -> Cs[8..15] rows 2,3).
  os << "          \"vsetvli   t0, zero, e32, m1, ta, ma  \\n\\t\"\n";
  os << "          \"vse32.v   v2, (%[pcs])               \\n\\t\"\n";
  os << "          \"addi      t4, %[pcs], 32             \\n\\t\"\n";
  os << "          \"vse32.v   v3, (t4)                   \\n\\t\"\n";
  os << "          :\n";
  os << "          : [ka] \"r\"(Atile), [kb] \"r\"(Btile), [kt] \"r\"(kt),\n";
  os << "            [pcs] \"r\"(Cs)\n";
  os << "          : \"t0\", \"t1\", \"t2\", \"t3\", \"t4\",\n";
  os << "            \"v0\", \"v1\", \"v2\", \"v3\", \"memory\");\n";
  // Structured strided scatter: Cs[r*4+c] -> C[(mi*4+r)*N + nj*4+c].
  os << "      for (long r = 0; r < 4; ++r)\n";
  os << "        for (long c = 0; c < 4; ++c)\n";
  os << "          C[(long)(mi * 4 + r) * N + (nj * 4 + c)] += Cs[r * 4 + c];\n";
  os << "    }\n";
  os << "  }\n";
  os << "}";
  os.flush();
  return text;
}

std::string matmulHelperBodySigned() {
  return matmulHelperBody(kMatmulHelperName, "vmadot");
}

std::string matmulHelperBodyUnsigned() {
  return matmulHelperBody(kMatmulUHelperName, "vmadotu");
}

/// Lowers a selected IME MAC boundary (`tcrv.ime.mma` signed / `tcrv.ime.mma_u`
/// unsigned) into a standalone EmitC module:
///   #include <stdint.h>
///   static inline void tcrv_ime_vmadot[u]_mma_4x4x8(const int8_t*,
///                              const int8_t*, int32_t*) { __asm__(...vmadot[u]...) }
///   extern "C" void tcrv_emitc_<kernel>_<variant>(const int8_t *A,
///                                                  const int8_t *B,
///                                                  int32_t *C) {
///     // route_source_op + source_op provenance comments
///     tcrv_ime_vmadot[u]_mma_4x4x8(A, B, C);
///   }
/// The wrapper is structured emitc (emitc.func + emitc.call_opaque on the A/B/C
/// block args); the single asm leaf is confined to the helper. The pattern is
/// templated over the op type; the static traits pin the helper name + body, so
/// the signed/unsigned divergence is ONLY the emitted instruction leaf.
template <typename OpT>
class IMEMACToEmitCFunc final : public mlir::OpConversionPattern<OpT> {
public:
  using mlir::OpConversionPattern<OpT>::OpConversionPattern;
  using OpAdaptor = typename mlir::OpConversionPattern<OpT>::OpAdaptor;

  mlir::LogicalResult
  matchAndRewrite(OpT mma, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MLIRContext *context = mma.getContext();
    mlir::Location loc = mma.getLoc();

    llvm::StringRef helperName = IMEMACToEmitCFunc::helperName();
    std::string helperBody = IMEMACToEmitCFunc::helperBody();

    auto variant =
        mma->template getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
    auto sourceKernel =
        mma->template getAttrOfType<mlir::StringAttr>("source_kernel");
    if (!variant || !sourceKernel)
      return rewriter.notifyMatchFailure(
          mma, "IME MAC boundary requires selected_variant and source_kernel "
               "attributes");
    std::string functionName =
        ("tcrv_emitc_" + sourceKernel.getValue() + "_" + variant.getValue())
            .str();

    llvm::StringRef sourceOpName = mma.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef sourceRole = mma.getTCRVEmitCLowerableSourceRole();

    auto module = mma->template getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(mma, "IME MAC boundary has no module");

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
      rewriter.create<emitc::VerbatimOp>(loc, helperBody);
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
        loc, stepComment(sourceOpName, sourceRole, helperName));

    // Structured dataflow into the leaf: call_opaque on the A/B/C block args.
    llvm::SmallVector<mlir::Value, 3> callOperands;
    for (mlir::BlockArgument arg : entry->getArguments())
      callOperands.push_back(arg);
    rewriter.create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{}, helperName,
                                         callOperands);

    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());

    rewriter.eraseOp(mma);
    return mlir::success();
  }

private:
  static llvm::StringRef helperName();
  static std::string helperBody();
};

// Signed surface: tcrv.ime.mma -> the vmadot asm leaf (the first slice).
template <>
llvm::StringRef IMEMACToEmitCFunc<tcrv::ime::MMAOp>::helperName() {
  return kVmadotHelperName;
}
template <> std::string IMEMACToEmitCFunc<tcrv::ime::MMAOp>::helperBody() {
  return vmadotHelperBody();
}

// Unsigned surface: tcrv.ime.mma_u -> the vmadotu asm leaf (the second op).
template <>
llvm::StringRef IMEMACToEmitCFunc<tcrv::ime::MMAUOp>::helperName() {
  return kVmadotuHelperName;
}
template <> std::string IMEMACToEmitCFunc<tcrv::ime::MMAUOp>::helperBody() {
  return vmadotuHelperBody();
}

// Mixed-sign surface: tcrv.ime.mma_su -> the vmadotsu asm leaf (the fourth op).
template <>
llvm::StringRef IMEMACToEmitCFunc<tcrv::ime::MMASUOp>::helperName() {
  return kVmadotsuHelperName;
}
template <> std::string IMEMACToEmitCFunc<tcrv::ime::MMASUOp>::helperBody() {
  return vmadotsuHelperBody();
}

/// Lowers the tiled whole-matrix IME boundary (`tcrv.ime.matmul`) into a
/// standalone EmitC module:
///   #include <stdint.h>
///   static inline void tcrv_ime_vmadot[u]_matmul(const int8_t*, const int8_t*,
///                                 int32_t*, long M, long N, long K) { ...tiled... }
///   extern "C" void tcrv_emitc_<kernel>_<variant>(const int8_t *Apack,
///                                                  const int8_t *Bpack,
///                                                  int32_t *C) {
///     tcrv_ime_vmadot[u]_matmul(Apack, Bpack, C, <M>, <N>, <K>);
///   }
/// M/N/K are the op's capability-bound problem-dim FACTS (mat_m/mat_n/mat_k),
/// baked into the wrapper as constants (they are compile-time facts of the
/// selected variant). The wrapper is structured emitc (emitc.func +
/// emitc.call_opaque on the Apack/Bpack/C block args + emitc.constant M/N/K);
/// the verbatim helper holds the validated vmadot/vmadotu plus its
/// load/store/K-loop scaffold. Signed vs unsigned is the `ime_op` fact (NOT a
/// family-name branch): vmadot => signed helper, vmadotu => unsigned helper.
class IMEMatMulToEmitCFunc final
    : public mlir::OpConversionPattern<tcrv::ime::MatMulOp> {
public:
  using mlir::OpConversionPattern<tcrv::ime::MatMulOp>::OpConversionPattern;
  using OpAdaptor =
      typename mlir::OpConversionPattern<tcrv::ime::MatMulOp>::OpAdaptor;

  mlir::LogicalResult
  matchAndRewrite(tcrv::ime::MatMulOp matmul, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MLIRContext *context = matmul.getContext();
    mlir::Location loc = matmul.getLoc();

    // Signed/unsigned is a capability-derived FACT carried on the op, not a
    // family-name branch: ime_op == "vmadotu" => unsigned helper, else signed.
    bool isUnsigned = matmul.getImeOp() == "vmadotu";
    llvm::StringRef helperName =
        isUnsigned ? kMatmulUHelperName : kMatmulHelperName;
    std::string helperBody =
        isUnsigned ? matmulHelperBodyUnsigned() : matmulHelperBodySigned();

    auto variant =
        matmul->getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
    auto sourceKernel =
        matmul->getAttrOfType<mlir::StringAttr>("source_kernel");
    if (!variant || !sourceKernel)
      return rewriter.notifyMatchFailure(
          matmul, "IME matmul boundary requires selected_variant and "
                  "source_kernel attributes");
    std::string functionName =
        ("tcrv_emitc_" + sourceKernel.getValue() + "_" + variant.getValue())
            .str();

    int64_t matM = matmul.getMatM();
    int64_t matN = matmul.getMatN();
    int64_t matK = matmul.getMatK();

    llvm::StringRef sourceOpName = matmul.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef sourceRole = matmul.getTCRVEmitCLowerableSourceRole();

    auto module = matmul->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(matmul,
                                         "IME matmul boundary has no module");

    auto i8PtrType = emitc::PointerType::get(
        context, emitc::OpaqueType::get(context, "const int8_t"));
    auto i32PtrType = emitc::PointerType::get(
        context, emitc::OpaqueType::get(context, "int32_t"));
    auto longType = emitc::OpaqueType::get(context, "long");

    // Module-scope prologue: include + the self-contained tiled-kernel helper.
    {
      mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<emitc::IncludeOp>(loc, "stdint.h",
                                        /*is_standard_include=*/true);
      rewriter.create<emitc::VerbatimOp>(loc, helperBody);
    }

    mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());

    // Exported wrapper: extern "C" void <name>(const int8_t* Apack,
    // const int8_t* Bpack, int32_t* C).
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
        loc, stepComment(sourceOpName, sourceRole, helperName));

    // Structured dataflow into the leaf: call_opaque on the Apack/Bpack/C block
    // args plus the M/N/K problem-dim constants (compile-time variant facts).
    llvm::SmallVector<mlir::Value, 6> callOperands;
    for (mlir::BlockArgument arg : entry->getArguments())
      callOperands.push_back(arg);
    for (int64_t dim : {matM, matN, matK}) {
      auto dimAttr = emitc::OpaqueAttr::get(context, std::to_string(dim));
      auto constOp = rewriter.create<emitc::ConstantOp>(loc, longType, dimAttr);
      callOperands.push_back(constOp.getResult());
    }
    rewriter.create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{}, helperName,
                                         callOperands);

    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());

    rewriter.eraseOp(matmul);
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
    target.addIllegalOp<tcrv::ime::MMAOp, tcrv::ime::MMAUOp,
                        tcrv::ime::MMASUOp, tcrv::ime::MatMulOp>();
    target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
  }

  void
  populateLoweringPatterns(mlir::TypeConverter &typeConverter,
                           mlir::RewritePatternSet &patterns) const override {
    patterns.add<IMEMACToEmitCFunc<tcrv::ime::MMAOp>,
                 IMEMACToEmitCFunc<tcrv::ime::MMAUOp>,
                 IMEMACToEmitCFunc<tcrv::ime::MMASUOp>, IMEMatMulToEmitCFunc>(
        typeConverter, patterns.getContext());
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
