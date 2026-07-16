#include "Weft/Conversion/RVV/RVVToEmitC.h"

#include "Weft/Conversion/EmitC/BackendEmissionRegistry.h"
#include "Weft/Conversion/EmitC/TypedBackendEmissionDriver.h"
#include "Weft/Conversion/RVV/RVVBackendEmissionDriver.h"
#include "RVVToEmitCInternal.h"
#include "Weft/Conversion/RVV/RVVToEmitCSupport.h"
#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Transforms/Passes.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace weft {
namespace transforms {

#define GEN_PASS_DEF_RVVLOWERTOEMITC
#include "Weft/Transforms/Passes.h.inc"

} // namespace transforms
} // namespace weft

namespace weft {
namespace conversion {
namespace rvv {

// The free functions below (type conversions, pattern registration, the
// backend driver) live at rvv scope and reference the typed RVV / emitc names
// unqualified, so re-establish the aliases the former anonymous namespace
// provided. The VariantToEmitCFunc method definitions live in `namespace
// detail` (the named-namespace identity that lets them split across TUs); their
// own aliases come from the internal header's `detail` block.
namespace weftrvv = ::weft::rvv;
namespace emitc = ::mlir::emitc;

//===----------------------------------------------------------------------===//
// VariantOp -> emitc.func driver method definitions.
//
// VariantToEmitCFunc (declared in RVVToEmitCInternal.h) lowers a weft.exec
// .variant beachhead body to a top-level emitc.func. Its method DEFINITIONS are
// split across this TU (the registration + pattern + matchAndRewrite dispatch
// home, plus the methods not yet moved to a family TU) and the family-grouped
// RVVToEmitC*.cpp TUs. Each definition is a pure out-of-line move of the former
// in-class body: the emitted C is byte-identical (the block-dot fingerprint and
// lit suite prove it).
//===----------------------------------------------------------------------===//

namespace detail {

mlir::LogicalResult
VariantToEmitCFunc::matchAndRewrite(weft::exec::VariantOp variant, OpAdaptor /*adaptor*/,
                mlir::ConversionPatternRewriter &rewriter) const {
    mlir::MLIRContext *context = variant.getContext();

    // Only the selected lowering boundary (a variant carrying a with_vl scope)
    // is a beachhead body. Variants without a with_vl scope (e.g. the scalar
    // fallback) are left for the legacy path / other families.
    weftrvv::WithVLOp scope;
    for (mlir::Operation &op : variant.getBody().front()) {
      if (auto withVL = llvm::dyn_cast<weftrvv::WithVLOp>(op)) {
        scope = withVL;
        break;
      }
    }
    if (!scope)
      return rewriter.notifyMatchFailure(variant, "no with_vl scope to lower");

    weftrvv::SetVLOp preLoopSetVL =
        scope.getVl().getDefiningOp<weftrvv::SetVLOp>();
    if (!preLoopSetVL)
      return rewriter.notifyMatchFailure(
          variant, "with_vl vl token must be defined by weft_rvv.setvl");
    mlir::Value avlSource = preLoopSetVL.getAvl();
    auto avlAbi = avlSource.getDefiningOp<weftrvv::RuntimeABIValueOp>();
    if (!avlAbi)
      return rewriter.notifyMatchFailure(
          variant, "setvl avl must be a runtime ABI value");

    // Tail/mask policy scope guard. The converter emits tail/mask-AGNOSTIC RVV
    // intrinsic forms (e.g. __riscv_vadd_vv_i32m1) and does NOT model an
    // undisturbed destination for compute/load ops (which would need a
    // passthrough / `_tu`/`_tum` form). A body that requests an undisturbed tail
    // or mask policy must NOT be silently lowered as agnostic — that would emit
    // semantically wrong C. Fail the match so the unsupported-policy body falls
    // through unchanged (the legacy path rejected it explicitly; the conversion
    // simply does not take it over).
    //
    // EXCEPTION — the masked unit-store family. A pure masked-store body
    // (mask_load + payload load + masked_store, the ONLY store being a
    // masked_store) carries an undisturbed SCOPE policy because the masked store
    // skips inactive/tail lanes and leaves their destination memory contents
    // untouched. The masked-store `_m` intrinsic form HONORS that undisturbed
    // semantics by construction, so this specific body shape is correctly
    // lowered even under an undisturbed scope policy. Allow undisturbed ONLY for
    // that shape; every other op (which would need an agnostic or `_tu`/`_tum`
    // form) still forces the agnostic-only refusal so no compute/load is
    // silently mislowered.
    //
    // EXCEPTION 2 — the computed-mask masked-store family. A body whose only
    // store is a single weft_rvv.masked_store but whose mask is computed by a
    // weft_rvv.compare in the same scope (the runtime-scalar / vector
    // computed-mask store shape: load[+splat] -> compare -> masked_store) also
    // carries an undisturbed scope policy that the masked-store `_m` form
    // honors. The compare/splat/load steps emit agnostic intrinsics whose
    // results are fully defined over the active VL, so the undisturbed semantics
    // live entirely in the `_m` store -- correctly lowered. Allow undisturbed
    // for that shape too; anything else still forces the agnostic-only refusal.
    weftrvv::PolicyAttr policy = preLoopSetVL.getPolicy();
    if (policy.getTail() != weftrvv::TailPolicy::Agnostic ||
        policy.getMask() != weftrvv::MaskPolicy::Agnostic) {
      if (!isPureMaskedStoreBody(scope) &&
          !isComputedMaskMaskedStoreBody(scope))
        return rewriter.notifyMatchFailure(
            variant, "only tail/mask-agnostic policy is convertible (except a "
                     "pure masked-store body under undisturbed policy)");
    }

    // Collect the runtime ABI values as ordered function parameters.
    llvm::SmallVector<AbiParam, 4> params;
    for (mlir::Operation &op : variant.getBody().front()) {
      if (auto abi = llvm::dyn_cast<weftrvv::RuntimeABIValueOp>(op)) {
        AbiParam param;
        param.op = abi;
        param.cType = abi.getCType().str();
        param.emitcType = emitCTypeForCTypeSpelling(context, param.cType);
        params.push_back(param);
      }
    }
    if (params.empty())
      return rewriter.notifyMatchFailure(variant, "no runtime ABI parameters");

    // Duplicate runtime ABI c_name guard. Every runtime ABI value becomes a
    // distinct C function parameter, so two ABI values sharing a c_name is a
    // malformed callable contract the legacy route path rejects (e.g. an
    // indexed gather whose `data` and `index` buffers are both named "data").
    // The conversion renders parameters positionally (vN), so it would silently
    // accept the collision and bypass that rejection; refuse a duplicate c_name
    // so the malformed body falls back to the legacy validator.
    {
      llvm::StringSet<> seenCNames;
      for (const AbiParam &param : params) {
        weftrvv::RuntimeABIValueOp abiOp = param.op;
        if (!seenCNames.insert(abiOp.getCName()).second)
          return rewriter.notifyMatchFailure(
              variant, "duplicate runtime ABI c_name (malformed callable "
                       "contract; legacy validator owns the diagnostic)");
      }
    }

    // Derive the function name exactly as the export path does:
    // weft_emitc_<kernel>_<variant>.
    auto kernel = variant->getParentOfType<weft::exec::KernelOp>();
    if (!kernel)
      return rewriter.notifyMatchFailure(variant, "variant has no kernel");
    std::string functionName =
        ("weft_emitc_" + kernel.getSymName() + "_" + variant.getSymName())
            .str();

    // exec-binding contract gate (family-generic). When the selected variant
    // requests exec ABI bindings (`weft_rvv.require_exec_abi_bindings = true`),
    // every runtime ABI value MUST carry an `exec_binding` symbol to its
    // weft.exec ABI declaration -- the legacy route-family path rejects a
    // missing binding. If a body that opts into the contract has an unbound ABI
    // value, this conversion must NOT take it over (it would materialize C
    // without honoring the contract the legacy validator enforces). Fall back so
    // the legacy validator still rejects it.
    if (auto requireBindings = variant->getAttrOfType<mlir::BoolAttr>(
            "weft_rvv.require_exec_abi_bindings");
        requireBindings && requireBindings.getValue()) {
      for (const AbiParam &param : params)
        if (!param.op->hasAttr("exec_binding"))
          return rewriter.notifyMatchFailure(
              variant, "runtime ABI value missing required exec_binding "
                       "(contract enforced by the legacy validator)");
    }

    // Capability config gate (family-generic, I1-honoring). The selected
    // variant's `requires` names the RVV capability provider; that provider is a
    // queryable weft.exec.capability / weft.exec.target MLIR object that may
    // declare `supported_sew` / `supported_lmul`. If present and they EXCLUDE
    // the typed body's (sew, lmul), the capability gates this body out -- the
    // legacy route-family path rejects it ("supported_sew fact ... does not
    // include typed body SEW"). The conversion must respect that legality gate
    // and fall back, not materialize C the capability forbids. Reading the attrs
    // straight off the provider op keeps capability the authority (no string
    // model). When the provider declares no restriction the gate is silent.
    {
      unsigned bodySEW = static_cast<unsigned>(preLoopSetVL.getSew());
      llvm::StringRef bodyLMUL = preLoopSetVL.getLmul();
      // The typed body requires the ratified RVV1.0 tail/mask-agnostic (ta/ma)
      // policy iff its setvl policy is agnostic on both axes -- the form this
      // converter renders into the agnostic `_v` intrinsic spelling. That form
      // is illegal on the RVV0.7 ISA generation (xtheadvector / C920), so the
      // version gate reasons over this fact below.
      bool bodyRequiresAgnosticPolicy =
          policy.getTail() == weftrvv::TailPolicy::Agnostic &&
          policy.getMask() == weftrvv::MaskPolicy::Agnostic;
      if (mlir::failed(checkCapabilityConfigGate(
              rewriter, variant, kernel, bodySEW, bodyLMUL,
              bodyRequiresAgnosticPolicy)))
        return mlir::failure();
    }

    // Build a standalone top-level emitc module: the standard headers the RVV
    // intrinsic body needs, then the function. This mirrors the legacy
    // materializer's module shape (includes + func) so the rendered C is
    // byte-equivalent to the hardware-validated golden.
    mlir::Location loc = variant.getLoc();
    auto module = variant->getParentOfType<mlir::ModuleOp>();
    mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    llvm::SmallVector<llvm::StringRef, 4> headers = {"stddef.h", "stdint.h",
                                                     "riscv_vector.h"};
    // The bodies that call scalar libm need <math.h> so the emitted TU is
    // self-contained: F3 rms_norm (1/sqrtf(mean+eps)), F6 rope (cosf/sinf per
    // dim-pair angle), and nvfp4 (ldexpf in the UE4M3 scale decode -- the
    // constructed typed_flat_block_dot_loop_body nvfp4 codebook-core brick; the
    // monolith op was retired at the nvfp4 flip). ONLY these bodies add the header
    // -- every other (quant-dot / elementwise) kernel keeps the original
    // three-header list byte-identical (additivity).
    bool hasNvfp4CodebookCore = false;
    scope.getBody().walk(
        [&](weftrvv::GgmlBlockDotNVFP4Q80CodebookCoreOp) {
          hasNvfp4CodebookCore = true;
        });
    // rms_norm (now CONSTRUCTED through the reduce-model scaffold) still calls
    // scalar libm (1/sqrtf(mean+eps)), so its constructed body -- a
    // typed_elementwise_loop_body carrying the reduce core brick -- adds <math.h>
    // exactly as the retired monolith did (byte-exact self-include behavior).
    bool hasRmsNormReduceCore = false;
    scope.getBody().walk([&](weftrvv::ElementwiseRmsNormReduceCoreOp) {
      hasRmsNormReduceCore = true;
    });
    // rope (now CONSTRUCTED through the rotate-model scaffold) still calls scalar
    // libm (cosf/sinf per dim-pair angle), so its constructed body -- a
    // typed_elementwise_loop_body carrying the rope rotate core brick -- adds
    // <math.h> exactly as the retired monolith did (byte-exact self-include).
    bool hasRopeRotateCore = false;
    scope.getBody().walk([&](weftrvv::ElementwiseRopeRotateCoreOp) {
      hasRopeRotateCore = true;
    });
    // G.0.3 same-precision-tier gelu variant: the elementwise_gelu_map brick
    // carrying `gelu_precision = "f16lut"` lowers its per-element body to the
    // `weft_gelu_f16lut_scalar` opaque seam (byte-exact to ggml's as-shipped
    // GGML_GELU_FP16 f16 lookup table). The seam calls tanhf => needs <math.h>, and
    // its definition is emitted below as a module preamble.
    bool hasGeluF16Lut = false;
    scope.getBody().walk([&](weftrvv::ElementwiseGeluMapOp g) {
      if (auto prec = g->getAttrOfType<mlir::StringAttr>("gelu_precision"))
        if (prec.getValue() == "f16lut")
          hasGeluF16Lut = true;
    });
    if (hasRmsNormReduceCore || hasRopeRotateCore || hasNvfp4CodebookCore ||
        hasGeluF16Lut)
      headers.push_back("math.h");
    for (llvm::StringRef header : headers)
      rewriter.create<emitc::IncludeOp>(loc, header,
                                        /*is_standard_include=*/true);

    // Emit the f16-LUT gelu seam as a module preamble (once), between the includes
    // and the kernel func. This reproduces ggml's as-shipped GGML_GELU_FP16 path
    // EXACTLY (vec.h:46/968/988): a 65536-entry f16 lookup table (built once by
    // weft_gelu_f16lut_init, mirroring ggml_init's ggml_table_gelu_f16 build) + a
    // per-element clamp/convert/gather/convert. The f16<->f32 conversions and
    // ggml_gelu_f32 are a VERBATIM transcription of ggml's simd-mappings scalar
    // path. Same numeric TIER *and* same runtime ALGORITHM (table gather, no
    // per-element tanhf) as the opponent => a fair same-tier speed A/B, byte-exact.
    // The table build lives OFF the timed path (weft_gelu_f16lut_init, called once
    // by the harness -- symmetric to the opponent's opp_gelu_init).
    if (hasGeluF16Lut) {
      rewriter.create<emitc::VerbatimOp>(
          loc,
          "/* G.0.3 f16-LUT gelu seam: byte-exact + same-algorithm to ggml "
          "as-shipped GGML_GELU_FP16 (vec.h:46/968/988). Table gather, no runtime "
          "tanhf. Call weft_gelu_f16lut_init() once before use (mirrors "
          "ggml_init's ggml_table_gelu_f16 build). */\n"
          "static unsigned short weft_gelu_f16_table[1<<16];\n"
          "static int weft_gelu_f16_ready=0;\n"
          "static inline unsigned short weft_f32_to_f16(float f){\n"
          "  unsigned int x; __builtin_memcpy(&x,&f,4);\n"
          "  unsigned int sign=(x>>16)&0x8000u; int e=(int)((x>>23)&0xff)-127+15; "
          "unsigned int man=x&0x7fffffu;\n"
          "  if(e<=0){ if(e<-10) return (unsigned short)sign; man|=0x800000u; "
          "unsigned int sh=(unsigned int)(14-e);\n"
          "    unsigned short r=(unsigned short)(man>>sh); if((man>>(sh-1))&1) r++; "
          "return (unsigned short)(sign|r); }\n"
          "  if(e>=31) return (unsigned short)(sign|0x7c00u);\n"
          "  unsigned short r=(unsigned short)(sign|((unsigned int)e<<10)|"
          "(man>>13)); if((man>>12)&1) r++; return r;\n"
          "}\n"
          "static inline float weft_f16_to_f32(unsigned short h){\n"
          "  unsigned int sign=(unsigned int)(h&0x8000)<<16; unsigned int "
          "e=(h>>10)&0x1f; unsigned int man=h&0x3ff; unsigned int o;\n"
          "  if(e==0){ if(man==0){o=sign;} else { e=127-15+1; "
          "while(!(man&0x400)){man<<=1;e--;} man&=0x3ff; o=sign|(e<<23)|(man<<13);} "
          "}\n"
          "  else if(e==31){ o=sign|0x7f800000u|(man<<13); }\n"
          "  else { o=sign|((e+112)<<23)|(man<<13); }\n"
          "  float f; __builtin_memcpy(&f,&o,4); return f;\n"
          "}\n"
          "static inline float weft_ggml_gelu_f32(float x){ return 0.5f*x*(1.0f + "
          "tanhf(0.79788456080286535587989211986876f*x*(1.0f + "
          "0.044715f*x*x))); }\n"
          "extern \"C\" void weft_gelu_f16lut_init(void){\n"
          "  for(int i=0;i<(1<<16);++i){ float f=weft_f16_to_f32((unsigned "
          "short)i); weft_gelu_f16_table[i]=weft_f32_to_f16(weft_ggml_gelu_f32(f)); "
          "}\n"
          "  weft_gelu_f16_ready=1;\n"
          "}\n"
          "static inline float weft_gelu_f16lut_scalar(float x){\n"
          "  if(x <= -10.0f) return 0.0f;\n"
          "  if(x >=  10.0f) return x;\n"
          "  unsigned short t = weft_f32_to_f16(x);\n"
          "  return weft_f16_to_f32(weft_gelu_f16_table[t]);\n"
          "}");
    }

    rewriter.setInsertionPointToEnd(module.getBody());

    llvm::SmallVector<mlir::Type, 4> paramTypes;
    for (const AbiParam &param : params)
      paramTypes.push_back(param.emitcType);
    // Every forward-pass / quant-dot kernel returns void (results = {}) EXCEPT
    // the F5b soft_max, which faithfully matches ggml's bare
    // `ggml_float ggml_vec_soft_max_f32(...)` signature: it RETURNS the f64 sum.
    // The CONSTRUCTED soft_max body is the typed elementwise loop op carrying the
    // elementwise_soft_max_reduce_core reduce brick (keyed here). Guard the result
    // type tightly so every other body builds the function type byte-identically
    // (additivity); only the soft_max body carries a `double` result, paired with
    // the single `return (double)...;` emitted in its branch.
    llvm::SmallVector<mlir::Type, 1> resultTypes;
    if (isTypedElementwiseSoftMaxReduceLoopBody(scope))
      resultTypes.push_back(emitc::OpaqueType::get(context, "double"));
    mlir::FunctionType functionType =
        rewriter.getFunctionType(paramTypes, resultTypes);

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

    // segment2 deinterleave: maps a segment2_load field result SSA value to the
    // (loaded tuple value, field index). The downstream weft_rvv.move that
    // sources the field emits the __riscv_vget extract from this tuple.
    llvm::DenseMap<mlir::Value, std::pair<mlir::Value, unsigned>>
        segmentFieldMap;

    rewriter.setInsertionPointToStart(entry);

    // Vector/VL/AVL config facts from the typed scope and types.
    unsigned sew = static_cast<unsigned>(preLoopSetVL.getSew());
    llvm::StringRef lmul = preLoopSetVL.getLmul();
    mlir::Type sizeType = emitc::OpaqueType::get(context, "size_t");

    mlir::Value avlArg = valueMap.lookup(avlAbi.getResult());

    // Scope provenance: // weft_emitc.route_source_op=weft_rvv.with_vl role=scope
    rewriter.create<emitc::VerbatimOp>(
        loc, routeSourceComment(scope.getWEFTEmitCLowerableSourceOpName(),
                                scope.getWEFTEmitCLowerableSourceRole()));

    // Pre-loop full-chunk setvl: __riscv_vsetvl_e<sew><lmul>(n).
    std::string setvlCallee = riscvIntrinsicName("vsetvl", sew, lmul, "");
    mlir::Value vlmax = emitOpaqueCall(
        rewriter, loc, sizeType, setvlCallee, mlir::ValueRange{avlArg},
        preLoopSetVL.getWEFTEmitCLowerableSourceOpName(),
        preLoopSetVL.getWEFTEmitCLowerableSourceRole());

    // The low-precision Gearbox product-reduce-dequantize body owns a dedicated
    // routine: a function-scoped i32 accumulator variable carried across the
    // main (unroll-N) + scalar tail loops, then the dequant epilogue. It is NOT
    // the single-scope `out[0]`-memory-carry standalone reduction path.
    // The DEFERRED-WIDE low-precision contraction (N3 resource-aware
    // max-legal-LMUL schedule, the measured ssh-rvv winner) owns a dedicated
    // routine: a function-scoped i32m8 vector accumulator zero-seeded at its own
    // VLMAX, a strip loop that defers the i16m4 widening product into it via
    // vwadd.wv, ONE trailing vredsum + scalar acc[0] add, then the dequant
    // epilogue. The structural marker is weft_rvv.widening_accumulate.
    // The 2nd-family (i16 dot-reduce) DEFERRED-WIDE schedule (N3 resource-aware
    // max-legal-LMUL, the measured ssh-rvv winner dot_wide_deferred) owns a
    // dedicated routine: a function-scoped i32m8 vector accumulator zero-seeded
    // at its own VLMAX, a strip loop that defers the i32m8 widening product into
    // it via NON-widening vadd.vv, ONE trailing vredsum + scalar acc[0] add,
    // then an i32 lane-0 store (NO dequant). The structural marker is
    // weft_rvv.deferred_accumulate.
    // ggml block-dot / GEMM kernel dispatch (WS-D: zero-core-branch kernel
    // axis). Each tunable block-quant contraction body is recognized by a
    // structural marker (its op identity) and lowered by a dedicated emitter.
    // Every branch here was byte-identical boilerplate varying ONLY in the
    // (recognizer, emitter) pair, so the former 28-way if-chain is collapsed to
    // a first-match-wins table + loop. ORDER IS SEMANTIC: the recognizers are
    // not all mutually exclusive (the Q4_0 gemm-tile / gemm / block-dot and the
    // Q6_K/Q4_K aux32-partial / block-dot markers are checked specific-before-
    // general), so the table preserves the exact original source order. The
    // per-kernel structural docs live at each emitter's definition.
    using BlockDotRecognizer = bool (*)(weftrvv::WithVLOp);
    using BlockDotEmitter = mlir::LogicalResult (VariantToEmitCFunc::*)(
        mlir::ConversionPatternRewriter &, mlir::Location, weftrvv::WithVLOp,
        mlir::Value, mlir::Type,
        llvm::DenseMap<mlir::Value, mlir::Value> &) const;
    struct BlockDotKernel {
      BlockDotRecognizer recognize;
      BlockDotEmitter emit;
    };
    static constexpr BlockDotKernel kBlockDotKernels[] = {
        {&isQ4_0Q8_0BlockDotBody,
         &VariantToEmitCFunc::emitQ4_0Q8_0BlockDot},
        {&isQ4_0Q8_0GemmTileBody,
         &VariantToEmitCFunc::emitQ4_0Q8_0GemmTile},
        {&isQ4_0Q8_0GemmBody,
         &VariantToEmitCFunc::emitQ4_0Q8_0Gemm},
        // NOTE: the q4_0 16x1-repacked GEMM's monolithic op
        // (weft_rvv.repack_gemm_q4_0_q8_0) is RETIRED, as is the GEVM's
        // (weft_rvv.repack_gemv_q4_0_q8_0); the repack front door now constructs
        // the typed weft_rvv.typed_repack_gemm_loop_body / typed_repack_gemv_loop_body
        // regions, lowered below by isTypedRepackGemmLoopBody ->
        // emitTypedRepackGemmLoopBody / isTypedRepackGemvLoopBody ->
        // emitTypedRepackGemvLoopBody.
        // NOTE (G3-lode-flat FLAT-2): the q5_0 16x1-repacked GEVM direct emitter
        // (emitRepackGemvQ5_0Q8_0) is RETIRED from the production dispatch (the
        // FIVE-bit family retired to the front door, the SECOND flat tracer after
        // q4_1): the repack front door now CONSTRUCTS the typed
        // weft_rvv.typed_repack_gem{v,m}_loop_body region (fold_model
        // "lane_wise_vector_scale", d-only no min) out of the SHARED q4_0 bricks --
        // the core stamping weight_nibble_unsigned + weight_qh_byte_offset (@288) +
        // weight_offset_bias (16) (the q5_0 5th-bit `((nibble)|(qh_bit<<4))-16`
        // decode leaf), lowered below by isTypedRepackGem{v,m}LoopBody ->
        // emitTypedRepackGem{v,m}LoopBody (RVVLowerQuantContraction.cpp
        // lowerToRepackGem{v,m}Q50; byte-exact ZERO-MODEL host cert
        // tools/e2e-harness/g3-lode-flat-q50). q8_0 follows via the SAME
        // shared-brick template.
        // NOTE (G3-lode-flat FLAT-3): the q5_1 16x1-repacked GEVM direct emitter
        // (emitRepackGemvQ5_1Q8_1) is RETIRED from the production dispatch (the
        // FIVE-bit-with-min family, the THIRD flat tracer after q4_1/q5_0): the
        // repack front door now CONSTRUCTS the typed
        // weft_rvv.typed_repack_gem{v,m}_loop_body region (fold_model
        // "lane_wise_vector_scale_min", the q4_1 min fold) out of the SHARED q4_0
        // bricks -- the core stamping weight_nibble_unsigned + weight_qh_byte_offset
        // (@320) with NO weight_offset_bias (the q5_1 UNSIGNED 5-bit
        // `(nibble)|(qh_bit<<4)` decode leaf) + the fold stamping the single
        // MIN-fold offset pair, lowered below by isTypedRepackGem{v,m}LoopBody ->
        // emitTypedRepackGem{v,m}LoopBody (RVVLowerQuantContraction.cpp
        // lowerToRepackGem{v,m}Q51; byte-exact ZERO-MODEL host cert
        // tools/e2e-harness/g3-lode-flat-q51). q5_1 = q5_0's qh gather (unsigned/no
        // -16) + q4_1's min fold. q8_0 follows via the SAME shared-brick template.
        {&isPackQ4_0ToX16Body,
         &VariantToEmitCFunc::emitPackQ4_0ToX16},
        // NOTE (G3-lode-flat 曳光弹): the q4_1 16x1-repacked GEVM+GEMM direct
        // emitters (emitRepackGem{v,m}Q4_1Q8_1) are RETIRED from the production
        // dispatch (the FLAT-family FIRST tracer retired to the front door): the
        // repack front door now CONSTRUCTS the typed
        // weft_rvv.typed_repack_gem{v,m}_loop_body region (fold_model
        // "lane_wise_vector_scale_min") out of the SHARED q4_0 bricks -- the core
        // stamping weight_nibble_unsigned (the q4_1 RAW unsigned nibble decode) + the
        // fold stamping the single MIN-fold offset pair (RVVLowerQuantContraction.cpp
        // lowerToRepackGem{v,m}Q41), lowered below by isTypedRepackGem{v,m}LoopBody ->
        // emitTypedRepackGem{v,m}LoopBody (byte-exact, ZERO-MODEL host cert
        // experiments/active/g3-lode-flat-q41). q5_0/q5_1/q8_0 follow via the SAME
        // shared-brick template.
        // NOTE (G3 主线A T2-construct): the q4_K 16x1-repacked GEMM direct emitter
        // (emitRepackGemmQ4KQ8K) + its monolith op (weft_rvv.repack_gemm_q4_K_q8_K)
        // + recognizer (isRepackGemmQ4KQ8KBody) are RETIRED (the FIRST K-quant
        // super-block repack retired to the front door): the repack front door now
        // constructs the typed weft_rvv.typed_repack_gemm_loop_body region
        // (fold_model "kquant_dmin_bsums_min") carrying the
        // weft_rvv.repack_gemm_kquant_core brick (decode_model "q4_K"), lowered
        // below by isTypedRepackGemmLoopBody -> emitTypedRepackGemmLoopBody (K-quant
        // branch) -> emitRepackKQuantGemmBodyQ4K (byte-exact to the retired direct
        // emitter). q6_K/q2_K/q3_K/q5_K ALSO retired (batch5 COMPLETE).
        // NOTE (G3 主线A T3 format4): q5_K GEMM direct emitter (emitRepackGemmQ5KQ8K) +
        // its monolith op + recognizer RETIRED to the front door (constructed +
        // S6-tiled typed_repack_gemm_loop_body, fold_model "kquant_dmin_bsums_min"
        // SHARED with q4_K -> emitRepackKQuantGemmBodyQ5K; the register-cliff lever
        // transfers like q2_K -- q5_K = q4_K min fold + the qh 5th-bit plane).
        // NOTE (G3 主线A T3): q6_K GEMM direct emitter (emitRepackGemmQ6KQ8K) + its
        // monolith op + recognizer RETIRED to the front door (constructed +
        // S6-tiled typed_repack_gemm_loop_body, fold_model "kquant_single_scale_no_min").
        // NOTE (G3 主线A T3 format2): q2_K GEMM direct emitter (emitRepackGemmQ2KQ8K) +
        // its monolith op + recognizer RETIRED to the front door (constructed +
        // S6-tiled typed_repack_gemm_loop_body, fold_model "kquant_dmin_bsums_min"
        // SHARED with q4_K -> emitRepackKQuantGemmBodyQ2K).
        // NOTE (G3 主线A T3 format3): q3_K GEMM direct emitter (emitRepackGemmQ3KQ8K) +
        // its monolith op + recognizer RETIRED to the front door (constructed PLAIN
        // typed_repack_gemm_loop_body, fold_model "kquant_single_scale_no_min" SHARED
        // with q6_K -> emitRepackKQuantGemmBodyQ3K; S6 tiling NULL for weight-bound q3_K).
        // NOTE: the tq2_0 (G3 主线B batch1) AND tq1_0 (G3 主线B batch2) 16x1-repacked
        // GEMM direct emitters (emitRepackGemm{TQ20,TQ10}Q8K) + their monolith ops
        // (weft_rvv.repack_gemm_tq{2,1}_0_q8_K) + recognizers are RETIRED (the whole
        // ternary batch): the repack front door now constructs the typed
        // weft_rvv.typed_repack_gemm_loop_body region carrying the
        // weft_rvv.repack_gemm_ternary_core brick, lowered below by
        // isTypedRepackGemmLoopBody -> emitTypedRepackGemmLoopBody (ternary branch)
        // -> emitRepackTernaryGemmBodyTQ{20,10} (byte-exact to the retired direct
        // emitters; the tq1_0 branch reads the base-3 qh SECOND weight plane).
        // NOTE (G3 M2 iq4_nl codebook front-door): the iq4_nl 16x1-repacked GEMM direct
        // emitter (emitRepackGemmIq4NlQ80) + its monolith op
        // (weft_rvv.repack_gemm_iq4_nl_q8_0) + recognizer (isRepackGemmIq4NlQ80Body) are
        // RETIRED: the repack front door now constructs the typed
        // weft_rvv.typed_repack_gemm_loop_body region (fold_model
        // "codebook_flat_single_scale") carrying the weft_rvv.repack_gemm_codebook_core
        // brick (decode_model "iq4_nl"), lowered below by isTypedRepackGemmLoopBody ->
        // emitTypedRepackGemmLoopBody (codebook branch) -> emitRepackCodebookGemmBodyIq4Nl
        // (byte-exact PLAIN untiled body to the retired direct emitter).
        // NOTE (G3 retirement_batch 4 mxfp4 codebook front-door): the mxfp4 16x1-repacked
        // GEMM direct emitter (emitRepackGemmMxfp4Q8) + its monolith op
        // (weft_rvv.repack_gemm_mxfp4_q8_0) + recognizer (isRepackGemmMxfp4Q8Body) are
        // RETIRED: the repack front door now constructs the typed
        // weft_rvv.typed_repack_gemm_loop_body region (fold_model "codebook_flat_e8m0_scale")
        // carrying the weft_rvv.repack_gemm_codebook_core brick (decode_model "mxfp4"),
        // lowered below by isTypedRepackGemmLoopBody -> emitTypedRepackGemmLoopBody (codebook
        // branch) -> emitRepackCodebookGemmBodyMxfp4 (byte-exact PLAIN untiled body to the
        // retired direct emitter; the E8M0 sibling of iq4_nl).
        // NOTE (G3 M2-后 iq4_xs codebook front-door): the iq4_xs 16x1-repacked GEMM direct
        // emitter (emitRepackGemmIq4XsQ8K) + its monolith op
        // (weft_rvv.repack_gemm_iq4_xs_q8_K) + recognizer (isRepackGemmIq4XsQ8KBody) are
        // RETIRED: the repack front door now constructs the typed
        // weft_rvv.typed_repack_gemm_loop_body region (fold_model
        // "codebook_superblock_signed6_no_min") carrying the weft_rvv.repack_gemm_codebook_core
        // brick (decode_model "iq4_xs"), lowered below by isTypedRepackGemmLoopBody ->
        // emitTypedRepackGemmLoopBody (codebook branch) -> emitRepackCodebookGemmBodyIq4Xs
        // (byte-exact PLAIN untiled body to the retired direct emitter, COMPLETING the iq4
        // codebook pair with iq4_nl).
        // NOTE (G3 M4 iq2-grid front-door, first cell iq2_xxs): the iq2_xxs 16x1-repacked
        // GEMM direct emitter (emitRepackGemmIq2XxsQ8K) + its monolith op
        // (weft_rvv.repack_gemm_iq2_xxs_q8_K) + recognizer (isRepackGemmIq2XxsQ8KBody) are
        // RETIRED: the repack front door now constructs the typed
        // weft_rvv.typed_repack_gemm_loop_body region (fold_model
        // "grid_sign_single_scale_eighth") carrying the NEW weft_rvv.repack_gemm_grid_core
        // brick (decode_model "iq2_xxs"), lowered below by isTypedRepackGemmLoopBody ->
        // emitTypedRepackGemmLoopBody (grid branch) -> emitRepackGridGemmBodyIq2Xxs
        // (byte-exact PLAIN untiled body to the retired direct emitter; the FIRST GRID
        // decode family).
        // NOTE (G3 M4 iq2-grid front-door, cells iq2_xs + iq2_s): the iq2_xs / iq2_s
        // DUAL-scale 16x1-repacked GEMM direct emitters (emitRepackGemmIq2{Xs,S}Q8K) + their
        // monolith ops (weft_rvv.repack_gemm_iq2_{xs,s}_q8_K) + recognizers
        // (isRepackGemmIq2{Xs,S}Q8KBody) are RETIRED, COMPLETING the iq2 grid family: the
        // repack front door now constructs the typed weft_rvv.typed_repack_gemm_loop_body
        // region (fold_model "grid_sign_dualscale_eighth") carrying the SAME
        // weft_rvv.repack_gemm_grid_core brick (decode_model "iq2_xs" / "iq2_s"), lowered
        // below by isTypedRepackGemmLoopBody -> emitTypedRepackGemmLoopBody (grid branch) ->
        // emitRepackGemmIq2DualScaleQ8K (byte-exact PLAIN untiled dual-ls body leaf to the
        // retired direct emitters).
        // NOTE (G3-lode-flat FLAT-4 收官格): the q8_0 16x1-repacked GEVM direct
        // emitter (emitRepackGemvQ8_0Q8_0) + recognizer (isRepackGemvQ8_0Q8_0Body)
        // are RETIRED from the production dispatch (the monolith op
        // weft_rvv.repack_gemv_q8_0_q8_0 + emitter remain defined as lit-only
        // scaffolding): the repack front door now CONSTRUCTS the typed
        // weft_rvv.typed_repack_gem{v,m}_loop_body region (fold_model
        // "lane_wise_vector_scale", the q4_0 d-only fold WHOLE) carrying the SHARED
        // weft_rvv.repack_lane_wise_q4_x_i8_dot / repack_gemm_lane_wise_q4_x_i8_dot
        // CORE brick stamping weight_full_i8 (the FULL-int8 variant: vle8 i8 +
        // per-position vwmul/vwadd_wv i32 in-block accumulation, NO nibble decode),
        // lowered by isTypedRepackGem{v,m}LoopBody -> emitTypedRepackGem{v,m}LoopBody's
        // default path -> emitRepackQ4LaneWiseIntegerCore / emitRepackGemmQ4LaneWise
        // IntegerCore's fullI8 branch (byte-exact GEVM to the retired direct emitter;
        // NET-NEW oracle-validated GEMM). The SIMPLEST flat family, the LAST FLAT-4.
        // NOTE (G3 主线A T2-construct): the q4_K 16x1-repacked GEVM direct emitter
        // (emitRepackGemvQ4KQ8K) + its monolith op (weft_rvv.repack_gemv_q4_K_q8_K)
        // + recognizer (isRepackGemvQ4KQ8KBody) are RETIRED: the repack front door
        // now constructs the typed weft_rvv.typed_repack_gemv_loop_body region
        // (fold_model "kquant_dmin_bsums_min") carrying the
        // weft_rvv.repack_gemv_kquant_core brick (decode_model "q4_K"), lowered
        // below by isTypedRepackGemvLoopBody -> emitTypedRepackGemvLoopBody (K-quant
        // branch) -> emitRepackKQuantGemvBodyQ4K (byte-exact to the retired direct
        // emitter).
        // NOTE (G3 主线A T3 format4): q5_K GEVM direct emitter (emitRepackGemvQ5KQ8K) +
        // its monolith op + recognizer RETIRED to the front door (constructed
        // typed_repack_gemv_loop_body, fold_model "kquant_dmin_bsums_min" SHARED with
        // q4_K -> emitRepackKQuantGemvBodyQ5K; q5_K = q4_K min fold + the qh 5th-bit
        // plane). This COMPLETES the K-quant repack family (q4_K/q6_K/q2_K/q3_K/q5_K).
        // NOTE (G3 主线A T3): q6_K GEVM direct emitter (emitRepackGemvQ6KQ8K) + its
        // monolith op + recognizer RETIRED to the front door (constructed
        // typed_repack_gemv_loop_body, fold_model "kquant_single_scale_no_min").
        // NOTE (G3 主线A T3 format2): q2_K GEVM direct emitter (emitRepackGemvQ2KQ8K) +
        // its monolith op + recognizer RETIRED to the front door (constructed
        // typed_repack_gemv_loop_body, fold_model "kquant_dmin_bsums_min" SHARED with
        // q4_K -> emitRepackKQuantGemvBodyQ2K).
        // NOTE (G3 主线A T3 format3): q3_K GEVM direct emitter (emitRepackGemvQ3KQ8K) +
        // its monolith op + recognizer RETIRED to the front door (constructed
        // typed_repack_gemv_loop_body, fold_model "kquant_single_scale_no_min" SHARED
        // with q6_K -> emitRepackKQuantGemvBodyQ3K).
        // NOTE: the tq2_0 (G3 主线B batch1) AND tq1_0 (G3 主线B batch2) 16x1-repacked
        // GEVM direct emitters (emitRepackGemv{TQ20,TQ10}Q8K) + their monolith ops
        // (weft_rvv.repack_gemv_tq{2,1}_0_q8_K) + recognizers are RETIRED (the whole
        // ternary batch): the repack front door now constructs the typed
        // weft_rvv.typed_repack_gemv_loop_body region carrying the
        // weft_rvv.repack_gemv_ternary_core brick, lowered below by
        // isTypedRepackGemvLoopBody -> emitTypedRepackGemvLoopBody (ternary branch)
        // -> emitRepackTernaryGemvBodyTQ{20,10} (byte-exact to the retired direct
        // emitters; the tq1_0 branch reads the base-3 qh SECOND weight plane).
        // NOTE (G3 M2 iq4_nl codebook front-door): the iq4_nl 16x1-repacked GEVM direct
        // emitter (emitRepackGemvIq4NlQ80) + its monolith op
        // (weft_rvv.repack_gemv_iq4_nl_q8_0) + recognizer (isRepackGemvIq4NlQ80Body) are
        // RETIRED: the repack front door now constructs the typed
        // weft_rvv.typed_repack_gemv_loop_body region (fold_model
        // "codebook_flat_single_scale") carrying the weft_rvv.repack_gemv_codebook_core
        // brick (decode_model "iq4_nl"), lowered below by isTypedRepackGemvLoopBody ->
        // emitTypedRepackGemvLoopBody (codebook branch) -> emitRepackCodebookGemvBodyIq4Nl
        // (byte-exact to the retired direct emitter).
        // NOTE (G3 retirement_batch 4 mxfp4 codebook front-door): the mxfp4 16x1-repacked
        // GEVM direct emitter (emitRepackGemvMxfp4Q8) + its monolith op
        // (weft_rvv.repack_gemv_mxfp4_q8_0) + recognizer (isRepackGemvMxfp4Q8Body) are
        // RETIRED: the repack front door now constructs the typed
        // weft_rvv.typed_repack_gemv_loop_body region (fold_model "codebook_flat_e8m0_scale")
        // carrying the weft_rvv.repack_gemv_codebook_core brick (decode_model "mxfp4"),
        // lowered below by isTypedRepackGemvLoopBody -> emitTypedRepackGemvLoopBody (codebook
        // branch) -> emitRepackCodebookGemvBodyMxfp4 (byte-exact to the retired direct
        // emitter; the E8M0 sibling of iq4_nl).
        // NOTE (G3 M2-后 iq4_xs codebook front-door): the iq4_xs 16x1-repacked GEVM direct
        // emitter (emitRepackGemvIq4XsQ8K) + its monolith op
        // (weft_rvv.repack_gemv_iq4_xs_q8_K) + recognizer (isRepackGemvIq4XsQ8KBody) are
        // RETIRED: the repack front door now constructs the typed
        // weft_rvv.typed_repack_gemv_loop_body region (fold_model
        // "codebook_superblock_signed6_no_min") carrying the weft_rvv.repack_gemv_codebook_core
        // brick (decode_model "iq4_xs"), lowered below by isTypedRepackGemvLoopBody ->
        // emitTypedRepackGemvLoopBody (codebook branch) -> emitRepackCodebookGemvBodyIq4Xs
        // (byte-exact to the retired direct emitter; the SECOND codebook decode family, the
        // codebook super-block sibling of iq4_nl).
        // NOTE (G3 M4 iq2-grid front-door, first cell iq2_xxs): the iq2_xxs 16x1-repacked
        // GEVM direct emitter (emitRepackGemvIq2XxsQ8K) + its monolith op
        // (weft_rvv.repack_gemv_iq2_xxs_q8_K) + recognizer (isRepackGemvIq2XxsQ8KBody) are
        // RETIRED: the repack front door now constructs the typed
        // weft_rvv.typed_repack_gemv_loop_body region (fold_model
        // "grid_sign_single_scale_eighth") carrying the NEW weft_rvv.repack_gemv_grid_core
        // brick (decode_model "iq2_xxs"), lowered below by isTypedRepackGemvLoopBody ->
        // emitTypedRepackGemvLoopBody (grid branch) -> emitRepackGridGemvBodyIq2Xxs
        // (byte-exact to the retired direct emitter; the FIRST GRID decode family).
        // NOTE (G3 M4 iq2-grid front-door, cells iq2_xs + iq2_s): the iq2_xs / iq2_s
        // DUAL-scale 16x1-repacked GEVM direct emitters (emitRepackGemvIq2{Xs,S}Q8K) + their
        // monolith ops (weft_rvv.repack_gemv_iq2_{xs,s}_q8_K) + recognizers
        // (isRepackGemvIq2{Xs,S}Q8KBody) are RETIRED, COMPLETING the iq2 grid family: the
        // repack front door now constructs the typed weft_rvv.typed_repack_gemv_loop_body
        // region (fold_model "grid_sign_dualscale_eighth") carrying the SAME
        // weft_rvv.repack_gemv_grid_core brick (decode_model "iq2_xs" / "iq2_s"), lowered
        // below by isTypedRepackGemvLoopBody -> emitTypedRepackGemvLoopBody (grid branch) ->
        // emitRepackGemvIq2DualScaleQ8K (byte-exact dual-ls body leaf to the retired direct
        // emitters).
        {&isTypedFlatBlockDotLoopBody,
         &VariantToEmitCFunc::emitTypedFlatBlockDotLoopBody},
        // The FRONT-DOOR CONSTRUCTED streaming dequantize_row family-head (q8_0):
        // a pre-constructed weft_rvv.typed_dequantize_row_loop_body region (from an
        // explicit-region lit, or the in-pass construction below) lowers here via
        // the shared q8_0 body -- byte-exact to the retired dispatch-wired monolith.
        {&isTypedDequantizeRowLoopBody,
         &VariantToEmitCFunc::emitTypedDequantizeRowLoopBody},
        // The PRE-EMITC FRONT-DOOR CONSTRUCTED streaming quantize_row family
        // {q8_0/q8_1/q8_K}: a pre-constructed weft_rvv.typed_quantize_row_loop_body
        // region (from the RVVQuantizeRowStreamFrontDoor pass -- the abstract
        // quantize_row_q8_{0,1,K} is already rewritten away) lowers here via the
        // shared per-format body -- byte-exact to the in-emitc construct+emit path
        // (constructQuantizeRowRegionAndLower). The f32->QUANT mirror of the dequant
        // stream entry above.
        {&isTypedQuantizeRowLoopBody,
         &VariantToEmitCFunc::emitTypedQuantizeRowLoopBody},
        // M-FLAT forward-elementwise scaffold (line C, ① 之后): the typed
        // elementwise strip-loop body (constructed sibling of the flat block-dot
        // loop body). It replaced the retired monolith weft_rvv.ggml_vec_scale_f32
        // (its {isGgmlVecScaleF32Body, emitGgmlVecScaleF32} dispatch branch was
        // RETIRED at the scale flip): the constructed body carries the
        // weft_rvv.elementwise_scale_map map core brick, lowered by
        // isTypedElementwiseLoopBody -> emitTypedElementwiseLoopBody, byte-exact
        // to the monolith emit modulo the source-op provenance token.
        {&isTypedElementwiseLoopBody,
         &VariantToEmitCFunc::emitTypedElementwiseLoopBody},
        {&isTypedSuperBlockBlockDotLoopBody,
         &VariantToEmitCFunc::emitTypedSuperBlockBlockDotLoopBody},
        {&isTypedRepackGemvLoopBody,
         &VariantToEmitCFunc::emitTypedRepackGemvLoopBody},
        // The INDEPENDENT q4_K colgroup-tiled GEVM Emission Plan ([K-10]
        // structural-level · [PAT-2] P9): a distinct plan op (NOT a knob on the
        // GEVM loop body), lowered by isTypedRepackGemvColgroupTiledLoopBody ->
        // emitTypedRepackGemvColgroupTiledLoopBody ->
        // emitRepackKQuantGemvColgroupTiledBodyQ4K (byte-exact to the sibling GEVM
        // plan's q4_K body; only the column-group-tiled / block-streaming /
        // register-resident-bank ENVELOPE differs).
        {&isTypedRepackGemvColgroupTiledLoopBody,
         &VariantToEmitCFunc::emitTypedRepackGemvColgroupTiledLoopBody},
        {&isTypedRepackGemmLoopBody,
         &VariantToEmitCFunc::emitTypedRepackGemmLoopBody},
        // NOTE: the monolith iq4_xs kernel {isIQ4XSQ8KBlockDotBody,
        // emitIQ4XSQ8KBlockDot} was RETIRED at the iq4_xs flip (C_construct 23->24):
        // the front door now constructs the typed super-block SCALAR-accumulator loop
        // body (fold_model "scalar_delta_grid", stride 136) carrying the iq4_xs
        // codebook-core brick, lowered by isTypedSuperBlockBlockDotLoopBody ->
        // emitTypedSuperBlockBlockDotLoopBody ->
        // emitTypedSuperBlockScalarDeltaGridLoopBodyIq4xs.
        // NOTE: the monolith iq2_xxs kernel {isIQ2XXSQ8KBlockDotBody,
        // emitIQ2XXSQ8KBlockDot}, the monolith iq2_xs kernel {isIQ2XSQ8KBlockDotBody,
        // emitIQ2XSQ8KBlockDot} AND the monolith iq2_s kernel {isIQ2SQ8KBlockDotBody,
        // emitIQ2SQ8KBlockDot} were RETIRED at their flips (L3 coverage): the front door
        // now constructs the typed super-block SCALAR-accumulator GRID loop body
        // (fold_model "scalar_delta_grid", stride 66 for iq2_xxs / 74 for iq2_xs / 82 for
        // iq2_s), lowered by isTypedSuperBlockBlockDotLoopBody ->
        // emitTypedSuperBlockBlockDotLoopBody ->
        // emitTypedSuperBlockScalarDeltaGridLoopBodyIq2xxs / ...Iq2xs / ...Iq2s.
        // NOTE: the monolith iq3_xxs kernel {isIQ3XXSQ8KBlockDotBody,
        // emitIQ3XXSQ8KBlockDot} was RETIRED at the iq3_xxs flip (L3 coverage): the
        // front door now constructs the typed super-block SCALAR-accumulator GRID loop
        // body (fold_model "scalar_delta_grid", stride 98), lowered by
        // isTypedSuperBlockBlockDotLoopBody -> emitTypedSuperBlockBlockDotLoopBody ->
        // emitTypedSuperBlockScalarDeltaGridLoopBodyIq3xxs.
        // NOTE: the monolith iq3_s kernel {isIQ3SQ8KBlockDotBody,
        // emitIQ3SQ8KBlockDot} was RETIRED at the iq3_s flip (C_construct 22->23): the
        // front door now constructs the typed super-block SCALAR-accumulator GRID loop
        // body (fold_model "scalar_delta_grid", stride 110), lowered by
        // isTypedSuperBlockBlockDotLoopBody -> emitTypedSuperBlockBlockDotLoopBody ->
        // emitTypedSuperBlockScalarDeltaGridLoopBodyIq3s.
        {&isMXFP4Q8_0BlockDotBody,
         &VariantToEmitCFunc::emitMXFP4Q8_0BlockDot},
        // NOTE: the monolith nvfp4 kernel {isNVFP4Q8_0BlockDotBody,
        // emitNVFP4Q8_0BlockDot} was RETIRED at the nvfp4 flip (C_construct 27->28):
        // the front door now constructs the typed FLAT block-dot loop body
        // (fold_model "flat_nvfp4_codebook") carrying the nvfp4 codebook-core brick,
        // lowered by isTypedFlatBlockDotLoopBody -> emitTypedFlatBlockDotLoopBody ->
        // the flat_nvfp4_codebook branch, which re-emits the byte-exact body through
        // the SHARED emitNVFP4BlockDotBodyShared (the sole live caller). mxfp4 stays
        // a monolith (the FP4-class negative control).
        {&isQ1_0Q8_0BlockDotBody,
         &VariantToEmitCFunc::emitQ1_0Q8_0BlockDot},
        {&isQ6_KQ8_KAux32PartialBody,
         &VariantToEmitCFunc::emitQ6_KQ8_KAux32Partial},
        {&isQ4_KNibbleUnpackBody,
         &VariantToEmitCFunc::emitQ4_KNibbleUnpack},
        {&isQ4_KScaleMinBitDanceBody,
         &VariantToEmitCFunc::emitQ4_KScaleMinBitDance},
        {&isQ4_KScaledDotBody,
         &VariantToEmitCFunc::emitQ4_KScaledDot},
        {&isQ4_KMinTermBody,
         &VariantToEmitCFunc::emitQ4_KMinTerm},
        {&isQ4_KSumsFoldScaleDBody,
         &VariantToEmitCFunc::emitQ4_KSumsFoldScaleD},
        {&isQ4_KHorizontalFoldBody,
         &VariantToEmitCFunc::emitQ4_KHorizontalFold},
        {&isQ4_KQ8_KAux32PartialBody,
         &VariantToEmitCFunc::emitQ4_KQ8_KAux32Partial},
        // NOTE: the monolith tq2_0 kernel {isTQ2_0Q8_KBlockDotBody,
        // emitTQ2_0Q8_KBlockDot} was RETIRED at the tq2_0 flip (C_construct 24->25,
        // the FIRST TQ-family member): the front door now constructs the typed
        // super-block SCALAR-accumulator TERNARY loop body (fold_model
        // "scalar_delta_grid", stride 66) carrying the tq2_0 ternary-core brick,
        // lowered by isTypedSuperBlockBlockDotLoopBody ->
        // emitTypedSuperBlockBlockDotLoopBody ->
        // emitTypedSuperBlockScalarDeltaGridLoopBodyTQ20.
        // NOTE: the monolith tq1_0 kernel {isTQ1_0Q8_KBlockDotBody,
        // emitTQ1_0Q8_KBlockDot} was RETIRED at the tq1_0 flip (C_construct 25->26,
        // the SECOND TQ-family member): the front door now constructs the typed
        // super-block SCALAR-accumulator BASE-3 TERNARY loop body (fold_model
        // "scalar_delta_grid", stride 54) carrying the tq1_0 base-3 ternary-core
        // brick, lowered by isTypedSuperBlockBlockDotLoopBody ->
        // emitTypedSuperBlockBlockDotLoopBody ->
        // emitTypedSuperBlockScalarDeltaGridLoopBodyTQ10.
    };
    for (const BlockDotKernel &kernel : kBlockDotKernels) {
      if (kernel.recognize(scope)) {
        if (mlir::failed((this->*kernel.emit)(rewriter, loc, scope, avlArg,
                                              sizeType, valueMap)))
          return mlir::failure();
        rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
        rewriter.eraseOp(variant);
        return mlir::success();
      }
    }
    // NOTE: the monolith forward-pass F1 kernel {isGgmlVecScaleF32Body,
    // emitGgmlVecScaleF32} was RETIRED at the scale flip (C_construct 28->29,
    // the FIRST forward-elementwise operator constructed): the constructed body
    // is the typed elementwise strip-loop op (weft_rvv.typed_elementwise_loop_body,
    // reduce_map_model "map") carrying the weft_rvv.elementwise_scale_map map core
    // brick, dispatched via the {isTypedElementwiseLoopBody,
    // emitTypedElementwiseLoopBody} entry in kBlockDotKernels above (byte-exact to
    // the monolith emit modulo the source-op provenance token).

    // NOTE: the monolith forward-pass F3 kernel {isGgmlRmsNormF32Body,
    // emitGgmlRmsNormF32} was RETIRED at the rms_norm flip (C_construct 30->31,
    // the FIRST forward REDUCE operator constructed): the constructed body is the
    // typed elementwise strip-loop op (weft_rvv.typed_elementwise_loop_body,
    // reduce_map_model "reduce") carrying the weft_rvv.elementwise_rms_norm_reduce_core
    // reduce core brick (the loop-carried f64 accumulator model), dispatched via
    // the {isTypedElementwiseLoopBody, emitTypedElementwiseLoopBody} entry in
    // kBlockDotKernels above (which now resolves the scale/silu MAP bricks AND the
    // rms_norm REDUCE brick). The reduce branch re-emits the byte-exact
    // scalar-double Σx² fold + scalar 1/sqrtf + vectorized normalize strip via
    // emitElementwiseRmsNormReduceStrip, byte-exact to the monolith emit modulo the
    // source-op provenance token.

    // NOTE: the monolith forward-pass F5 kernel {isGgmlVecSiluF32Body,
    // emitGgmlVecSiluF32} was RETIRED at the silu flip (C_construct 29->30, the
    // SECOND forward-elementwise operator constructed): the constructed body is
    // the SAME typed elementwise strip-loop op (weft_rvv.typed_elementwise_loop_body,
    // reduce_map_model "map") carrying the weft_rvv.elementwise_silu_map map core
    // brick, dispatched via the {isTypedElementwiseLoopBody,
    // emitTypedElementwiseLoopBody} entry in kBlockDotKernels above (which now
    // resolves BOTH the scale and silu map bricks). The per-strip m2 exp
    // polynomial is emitted by emitElementwiseSiluMapStrip, byte-exact to the
    // monolith emit modulo the source-op provenance token. The SHARED
    // emitGgmlVExpfM2 exp replication is UNCHANGED (soft_max F5b still consumes it).

    // The forward-pass F5b op (soft_max) COMBINES F5's vectorized transcendental
    // (the SHARED exact ggml_v_expf_m2 polynomial) with a NEW reduction shape:
    // y[i] = e^{x[i]-max} written per strip, and the f64 sum accumulated via the
    // WIDENING reduce vfwredusum_vs_f32m2_f64m1 into a loop-carried f64m1
    // accumulator (NOT F3's scalar-ascending fold) -- ggml's EXACT method
    // (vec.cpp:584-592). It is now CONSTRUCTED through the reduce-model scaffold
    // (the SAME typed_elementwise_loop_body rms_norm built, reduce_map_model
    // "reduce", carrying the weft_rvv.elementwise_soft_max_reduce_core reduce
    // brick; the monolith weft_rvv.ggml_vec_soft_max_f32 op + emitter + recognizer
    // + verifier were RETIRED). Unlike the void-return map/reduce bodies handled
    // by the kBlockDotKernels table above, soft_max is the ONLY forward-pass op
    // whose function RETURNS a scalar (the f64 sum), so it is dispatched HERE
    // (outside the void-return table): the branch wraps emitElementwiseSoftMaxReduceStrip's
    // f64 sum in the function `return`. Marker: the loop op carrying the soft_max
    // reduce brick (isTypedElementwiseLoopBody EXCLUDES it, so the table skips it).
    if (isTypedElementwiseSoftMaxReduceLoopBody(scope)) {
      mlir::FailureOr<mlir::Value> sum = emitElementwiseSoftMaxReduceStrip(
          rewriter, loc, scope, avlArg, sizeType, valueMap);
      if (mlir::failed(sum))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, *sum);
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // The forward-pass F4 op (weft_rvv.quantize_row_q8_0) is the f32 -> QUANT
    // BRIDGE: the per-32-block max-abs REDUCTION (vfredmax of vfabs) + the scalar
    // scale (d = amax/127, id = d ? 1/d : 0) + the f32->i16->i8 NARROWING CONVERT
    // (vfncvt round-to-nearest-even + vncvt truncate), written into the AoS
    // block_q8_0 buffer (the fp16 d at byte 0, the 32 int8 qs at byte 2). It owns
    // a dedicated routine -- an outer block loop, a single e32m8 strip per block
    // (vl=32, relying on Zvl128b => VLEN>=128), the structured d?1/d:0
    // emitc.cmp/emitc.if conditional, the native (_Float16)d AoS store -- a NEW
    // shape vs F1/F3/F5/F5b (a reduction + a narrowing convert + a structured
    // scalar branch) and vs the integer block-dot ops. Marker: the op identity.
    if (isGgmlQuantizeRowQ80Body(scope)) {
      if (mlir::failed(emitGgmlQuantizeRowQ80(rewriter, loc, scope, avlArg,
                                              sizeType, valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // The q8_1 SIBLING quantizer (weft_rvv.quantize_row_q8_1): the SAME amax/
    // scale/narrow shape as q8_0 PLUS the extra vwredsum integer block sum stored
    // as the fp16 block_q8_1.s. DISPATCH-WIRED, void-return like q8_0. Marker: the
    // op identity ([L-6] wiring != construction).
    if (isGgmlQuantizeRowQ81Body(scope)) {
      if (mlir::failed(emitGgmlQuantizeRowQ81(rewriter, loc, scope, avlArg,
                                              sizeType, valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // The q8_K K-quant activation quantizer (weft_rvv.quantize_row_q8_K): the
    // heaviest quantizer -- a QK_K=256 super-block min/max symmetric scale, the
    // vfcvt/vnclip RNE narrowing, the float d store, the per-16 vwredsum bsums,
    // and the zero-block memset special case. DISPATCH-WIRED, void-return. Marker:
    // the op identity ([L-6] wiring != construction).
    if (isGgmlQuantizeRowQ8KBody(scope)) {
      if (mlir::failed(emitGgmlQuantizeRowQ8K(rewriter, loc, scope, avlArg,
                                              sizeType, valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // NOTE: the four forward-elementwise f32 SUPPORT ops (add/mul/cpy/gelu) were
    // RETIRED at the support flip (dispatch-wired -> constructed, C_construct 73->77):
    // the retired monolith recognizer {isGgmlForwardElementwiseF32Body,
    // emitGgmlForwardElementwiseF32} + the support ops weft_rvv.{vec_add,vec_mul,
    // vec_cpy,gelu}_f32 are GONE. They are now CONSTRUCTED through the SAME abstract
    // weft_rvv.ggml_forward_elementwise source op + pre-emitc front door as
    // scale/silu/rms_norm/soft_max/rope: the front door constructs the typed
    // weft_rvv.typed_elementwise_loop_body region (reduce_map_model "map") carrying
    // the per-op elementwise_binary_map (add/mul) / elementwise_copy_map (cpy) /
    // elementwise_gelu_map (gelu) core brick, dispatched via the
    // {isTypedElementwiseLoopBody, emitTypedElementwiseLoopBody} entry in
    // kBlockDotKernels above. The map re-emit reuses the SHARED byte-exact strip/loop
    // helpers (emitForwardVecMapStrip / emitForwardGeluScalarLoop), so the emitted C
    // is byte-identical to the retired monolith modulo ONLY the source-op provenance
    // token.

    // The dequantize_row family (block_qX -> f32 row): the FAMILY-HEAD q8_0 is
    // FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites the
    // abstract weft_rvv.dequantize_row into the typed
    // weft_rvv.typed_dequantize_row_loop_body region { dequantize_row_decode_core;
    // typed_dequantize_row_loop_yield } and lowers it (emission DRIVEN by the typed
    // region op-identity + decode_model, [L-6]/[L-8] construction, byte-exact to the
    // retired q8_0 monolith). The other 22 formats stay DISPATCH-WIRED (the op
    // identity + bounded `format` route to a hand-written per-format monolith decode
    // reproducing ggml's reference dequantize_row_<format>). Void-return, like the
    // forward/quantize bridges above.
    if (isGgmlDequantizeRowBody(scope)) {
      if (mlir::failed(constructOrEmitGgmlDequantizeRow(rewriter, loc, scope,
                                                        avlArg, sizeType,
                                                        valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // NOTE: the monolith forward-pass F6 kernel {isGgmlRopeNormF32Body,
    // emitGgmlRopeNormF32} was RETIRED at the rope flip (C_construct 32->33, the
    // FIRST (and only) forward ROTATE operator constructed): the constructed body
    // is the typed elementwise strip-loop op (weft_rvv.typed_elementwise_loop_body,
    // reduce_map_model "rotate") carrying the weft_rvv.elementwise_rope_rotate_core
    // rotate core brick (the per-pair loop-carried f32 theta recurrence + the
    // scalar cos/sin angle seam + the position-dependent 2x2 rotation), dispatched
    // via the {isTypedElementwiseLoopBody, emitTypedElementwiseLoopBody} entry in
    // kBlockDotKernels above (which now resolves the scale/silu MAP bricks, the
    // rms_norm REDUCE brick, AND the rope ROTATE brick). The rotate branch re-emits
    // the byte-exact scalar per-pair loop via emitElementwiseRopeRotateStrip,
    // byte-exact to the monolith emit modulo the source-op provenance token.

    // The DEFERRED-WIDE / low-precision dequant-contraction family shares ONE
    // emitter signature (the variant + preLoopSetVL/vlmax/setvlCallee carried in,
    // distinct from the block-dot table's shorter signature above), so the three
    // former byte-identical if-branches collapse to a first-match table + loop --
    // the SAME structural-marker dispatch shape as kBlockDotKernels. ORDER IS
    // SEMANTIC: the markers are checked in the original source order (the
    // recognizers are not all mutually exclusive), so the table preserves it. The
    // per-body structural docs live at each emitter's declaration.
    using DequantRecognizer = bool (*)(weftrvv::WithVLOp);
    using DequantEmitter = mlir::LogicalResult (VariantToEmitCFunc::*)(
        mlir::ConversionPatternRewriter &, mlir::Location,
        weft::exec::VariantOp, weftrvv::WithVLOp, weftrvv::SetVLOp, mlir::Value,
        mlir::Value, mlir::Type, llvm::StringRef,
        llvm::DenseMap<mlir::Value, mlir::Value> &) const;
    struct DequantKernel {
      DequantRecognizer recognize;
      DequantEmitter emit;
    };
    static constexpr DequantKernel kDequantKernels[] = {
        {&isDeferredWideDotReduceBody,
         &VariantToEmitCFunc::emitDeferredWideDotReduceBody},
        {&isDeferredWideDequantBody,
         &VariantToEmitCFunc::emitDeferredWideDequantBody},
        {&isLowPrecisionDequantBody,
         &VariantToEmitCFunc::emitLowPrecisionDequantBody},
    };
    for (const DequantKernel &kernel : kDequantKernels) {
      if (kernel.recognize(scope)) {
        if (mlir::failed((this->*kernel.emit)(rewriter, loc, variant, scope,
                                              preLoopSetVL, avlArg, vlmax,
                                              sizeType, setvlCallee, valueMap)))
          return mlir::failure();
        rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
        rewriter.eraseOp(variant);
        return mlir::success();
      }
    }

    // The STANDALONE i32->f32 runtime-scale dequant body (load -> dequantize ->
    // store, no product/reduce/accumulator) owns a dedicated Gearbox-unrolled
    // two-slice setvl loop emitter -- the same VL-loop machinery, expanded
    // `weft_rvv.gearbox.unroll` times. It is NOT the product-reduce dequant
    // path (no accumulator) nor the single-slice emitScopeForLoop (which the
    // emitDequantize guard would refuse). Detect and emit it here.
    if (isStandaloneDequantBody(scope)) {
      if (mlir::failed(emitStandaloneDequantBody(
              rewriter, loc, scope, preLoopSetVL, avlArg, vlmax, sizeType,
              setvlCallee, valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // Standalone reduction pre-loop seed: out[0] = acc[0]. Runs BEFORE the loop
    // (between the pre-loop full-chunk setvl and the for-loop), seeding the
    // scalar accumulator carried through the output cell across runtime chunks.
    bool standaloneReduction = isStandaloneReductionBody(scope);
    if (standaloneReduction) {
      if (mlir::failed(
              emitStandaloneReductionPreLoopSeed(rewriter, loc, scope, valueMap)))
        return mlir::failure();
    }

    // Emit the scope's runtime VL for-loop (setvl-tail chunk loop) + body walk.
    // Extracted into a reusable per-scope helper so a multi-scope body (the
    // Gearbox dequant producer/tail/consumer scopes) can drive it once per
    // scope; the single-scope families call it exactly once, unchanged.
    if (mlir::failed(emitScopeForLoop(rewriter, loc, variant, scope,
                                      preLoopSetVL, avlArg, vlmax, sizeType,
                                      setvlCallee, valueMap, segmentFieldMap,
                                      standaloneReduction)))
      return mlir::failure();

    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());

    rewriter.eraseOp(variant);
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitScopeForLoop(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weft::exec::VariantOp variant, weftrvv::WithVLOp scope,
    weftrvv::SetVLOp preLoopSetVL, mlir::Value avlArg, mlir::Value vlmax,
    mlir::Type sizeType, llvm::StringRef setvlCallee,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    llvm::DenseMap<mlir::Value, std::pair<mlir::Value, unsigned>>
        &segmentFieldMap,
    bool standaloneReduction) const {
    // for (size_t i = 0; i < n; i += vlmax) { ... }
    mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    auto forOp = rewriter.create<emitc::ForOp>(loc, zero, avlArg, vlmax,
                                               /*bodyBuilder=*/nullptr);
    mlir::Value inductionVar = forOp.getInductionVar();

    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());

      // Remaining-AVL setvl: size_t v = n - i; __riscv_vsetvl_e...(v).
      mlir::Value bodyVL = emitOpaqueCallBuilt(
          rewriter, loc, sizeType, setvlCallee,
          preLoopSetVL.getWEFTEmitCLowerableSourceOpName(),
          preLoopSetVL.getWEFTEmitCLowerableSourceRole(),
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value remaining =
                b.create<emitc::SubOp>(l, sizeType, avlArg, inductionVar);
            return {remaining};
          });

      // Build the emit order. Bodies emit in IR order, EXCEPT the pure
      // masked-store family: the legacy route path emits the payload `load`
      // BEFORE the `mask_load` two-step (its realized IR carries them in the
      // opposite order). Reorder ONLY that shape so the rendered C stays
      // byte-identical to the legacy oracle; every other body keeps IR order.
      llvm::SmallVector<mlir::Operation *, 8> orderedOps;
      for (mlir::Operation &op : scope.getBody().front())
        orderedOps.push_back(&op);
      if (isPureMaskedStoreBody(scope)) {
        // Move the single plain payload load ahead of the mask_load.
        auto maskLoadIt = llvm::find_if(orderedOps, [](mlir::Operation *op) {
          return llvm::isa<weftrvv::MaskLoadOp>(op);
        });
        auto loadIt = llvm::find_if(orderedOps, [](mlir::Operation *op) {
          return llvm::isa<weftrvv::LoadOp>(op);
        });
        if (maskLoadIt != orderedOps.end() && loadIt != orderedOps.end() &&
            loadIt > maskLoadIt) {
          mlir::Operation *loadOp = *loadIt;
          orderedOps.erase(loadIt);
          orderedOps.insert(
              llvm::find_if(orderedOps,
                            [](mlir::Operation *op) {
                              return llvm::isa<weftrvv::MaskLoadOp>(op);
                            }),
              loadOp);
        }
      }

      // Computed-mask indexed memory ordering. The string-plan owner (the
      // byte-order the harness ordered-token validator depends on) emits the
      // index_load + its element->byte scale EARLY -- right after the first
      // compare-LHS load, before the splat / remaining loads / compare. The
      // realized IR instead carries index_load after those loads. Reorder ONLY a
      // computed-mask indexed body (one carrying a masked_indexed load/store) so
      // the rendered C keeps the legacy index-early order; every other body
      // keeps IR order. The byte-scale is emitted at index_load time (see
      // emitIndexLoad) so it immediately follows the index_load in that order.
      bool hasMaskedIndexed = llvm::any_of(orderedOps, [](mlir::Operation *op) {
        return llvm::isa<weftrvv::MaskedIndexedLoadOp,
                         weftrvv::MaskedIndexedStoreOp>(op);
      });
      if (hasMaskedIndexed) {
        auto indexLoadIt = llvm::find_if(orderedOps, [](mlir::Operation *op) {
          return llvm::isa<weftrvv::IndexLoadOp>(op);
        });
        auto firstLoadIt = llvm::find_if(orderedOps, [](mlir::Operation *op) {
          return llvm::isa<weftrvv::LoadOp>(op);
        });
        if (indexLoadIt != orderedOps.end() &&
            firstLoadIt != orderedOps.end() && indexLoadIt > firstLoadIt) {
          mlir::Operation *indexLoadOp = *indexLoadIt;
          orderedOps.erase(indexLoadIt);
          // Re-find the first load (the erase may have shifted iterators) and
          // insert the index_load immediately after it.
          auto insertAfter = llvm::find_if(orderedOps, [](mlir::Operation *op) {
            return llvm::isa<weftrvv::LoadOp>(op);
          });
          orderedOps.insert(std::next(insertAfter), indexLoadOp);
        }
      }

      // Computed-mask widening dot-reduce ordering. The string-plan owner emits
      // the compare-produced mask IMMEDIATELY after its two compare-input loads
      // (the byte order the harness ordered-token validator depends on), BEFORE
      // the two dot-product input loads. The realized IR instead carries the
      // compare AFTER all four loads (its operands and the dot inputs are
      // siblings). Reorder ONLY a computed-mask dot-reduce body (one carrying a
      // masked_widening_dot_reduce) so the rendered C keeps the legacy
      // mask-early order; every other body keeps IR order. The compare is moved
      // to immediately follow the last of its own (load/strided_load) operands.
      bool hasMaskedDotReduce = llvm::any_of(orderedOps, [](mlir::Operation *op) {
        return llvm::isa<weftrvv::MaskedWideningDotReduceOp>(op);
      });
      if (hasMaskedDotReduce) {
        auto compareIt = llvm::find_if(orderedOps, [](mlir::Operation *op) {
          return llvm::isa<weftrvv::CompareOp>(op);
        });
        if (compareIt != orderedOps.end()) {
          mlir::Operation *compareOp = *compareIt;
          // Find the last position among the compare's operand-defining ops in
          // the current order; the compare reorders to immediately after it.
          long lastOperandPos = -1;
          for (mlir::Value operand : compareOp->getOperands()) {
            mlir::Operation *def = operand.getDefiningOp();
            if (!def)
              continue;
            auto pos = llvm::find(orderedOps, def);
            if (pos != orderedOps.end())
              lastOperandPos = std::max<long>(
                  lastOperandPos, std::distance(orderedOps.begin(), pos));
          }
          if (lastOperandPos >= 0) {
            orderedOps.erase(compareIt);
            // The erase shifts positions after compareIt; recompute the insert
            // point by re-finding the last operand def.
            mlir::Operation *anchor = nullptr;
            long anchorPos = -1;
            for (mlir::Value operand : compareOp->getOperands()) {
              mlir::Operation *def = operand.getDefiningOp();
              if (!def)
                continue;
              auto pos = llvm::find(orderedOps, def);
              if (pos != orderedOps.end()) {
                long p = std::distance(orderedOps.begin(), pos);
                if (p > anchorPos) {
                  anchorPos = p;
                  anchor = def;
                }
              }
            }
            if (anchor) {
              auto anchorIt = llvm::find(orderedOps, anchor);
              orderedOps.insert(std::next(anchorIt), compareOp);
            } else {
              orderedOps.insert(orderedOps.begin(), compareOp);
            }
          }
        }
      }

      // The standalone reduction in-loop running seed reads back out[0]; the
      // store cell (the output buffer base) is the body's store target. Resolve
      // it once so the standalone reduce ops can read/seed it.
      mlir::Value standaloneOutBuffer;
      if (standaloneReduction) {
        for (mlir::Operation *opPtr : orderedOps)
          if (auto store = llvm::dyn_cast<weftrvv::StoreOp>(opPtr))
            standaloneOutBuffer = valueMap.lookup(store.getBuffer());
        if (!standaloneOutBuffer)
          return rewriter.notifyMatchFailure(
              variant, "standalone reduction output buffer unmapped");
      }

      // Convert each body op in emit order. Body holds the typed dataflow ops
      // for the elementwise/memory families.
      for (mlir::Operation *opPtr : orderedOps) {
        mlir::Operation &op = *opPtr;
        if (auto load = llvm::dyn_cast<weftrvv::LoadOp>(op)) {
          if (mlir::failed(emitLoad(rewriter, loc, load, valueMap, inductionVar,
                                    bodyVL)))
            return mlir::failure();
        } else if (auto broadcast =
                       llvm::dyn_cast<weftrvv::BroadcastLoadOp>(op)) {
          if (mlir::failed(emitBroadcastLoad(rewriter, loc, broadcast, valueMap,
                                             bodyVL)))
            return mlir::failure();
        } else if (auto splat = llvm::dyn_cast<weftrvv::SplatOp>(op)) {
          if (mlir::failed(emitSplat(rewriter, loc, splat, valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto stridedLoad =
                       llvm::dyn_cast<weftrvv::StridedLoadOp>(op)) {
          if (mlir::failed(emitStridedLoad(rewriter, loc, stridedLoad, valueMap,
                                           inductionVar, bodyVL)))
            return mlir::failure();
        } else if (auto indexLoad = llvm::dyn_cast<weftrvv::IndexLoadOp>(op)) {
          if (mlir::failed(emitIndexLoad(rewriter, loc, indexLoad, valueMap,
                                         inductionVar, bodyVL)))
            return mlir::failure();
        } else if (auto indexedLoad =
                       llvm::dyn_cast<weftrvv::IndexedLoadOp>(op)) {
          if (mlir::failed(emitIndexedLoad(rewriter, loc, indexedLoad, valueMap,
                                           bodyVL)))
            return mlir::failure();
        } else if (auto maskLoad = llvm::dyn_cast<weftrvv::MaskLoadOp>(op)) {
          if (mlir::failed(emitMaskLoad(rewriter, loc, maskLoad, valueMap,
                                        inductionVar, bodyVL)))
            return mlir::failure();
        } else if (auto maskedLoad =
                       llvm::dyn_cast<weftrvv::MaskedLoadOp>(op)) {
          if (mlir::failed(emitMaskedLoad(rewriter, loc, maskedLoad, valueMap,
                                          inductionVar, bodyVL)))
            return mlir::failure();
        } else if (auto maskedStridedLoad =
                       llvm::dyn_cast<weftrvv::MaskedStridedLoadOp>(op)) {
          if (mlir::failed(emitMaskedStridedLoad(rewriter, loc,
                                                 maskedStridedLoad, valueMap,
                                                 inductionVar, bodyVL)))
            return mlir::failure();
        } else if (auto maskedIndexedLoad =
                       llvm::dyn_cast<weftrvv::MaskedIndexedLoadOp>(op)) {
          if (mlir::failed(emitMaskedIndexedLoad(rewriter, loc,
                                                 maskedIndexedLoad, valueMap,
                                                 bodyVL)))
            return mlir::failure();
        } else if (auto move = llvm::dyn_cast<weftrvv::MoveOp>(op)) {
          if (mlir::failed(
                  emitMove(rewriter, loc, move, valueMap, segmentFieldMap)))
            return mlir::failure();
        } else if (auto segLoad =
                       llvm::dyn_cast<weftrvv::Segment2LoadOp>(op)) {
          if (mlir::failed(emitSegment2Load(rewriter, loc, segLoad, valueMap,
                                            segmentFieldMap, inductionVar,
                                            bodyVL)))
            return mlir::failure();
        } else if (auto segStore =
                       llvm::dyn_cast<weftrvv::Segment2StoreOp>(op)) {
          if (mlir::failed(emitSegment2Store(rewriter, loc, segStore, valueMap,
                                             inductionVar, bodyVL)))
            return mlir::failure();
        } else if (auto maskedSegLoad =
                       llvm::dyn_cast<weftrvv::MaskedSegment2LoadOp>(op)) {
          if (mlir::failed(emitMaskedSegment2Load(rewriter, loc, maskedSegLoad,
                                                  valueMap, inductionVar,
                                                  bodyVL)))
            return mlir::failure();
        } else if (auto maskedSegStore =
                       llvm::dyn_cast<weftrvv::MaskedSegment2StoreOp>(op)) {
          if (mlir::failed(emitMaskedSegment2Store(rewriter, loc,
                                                   maskedSegStore, valueMap,
                                                   inductionVar, bodyVL)))
            return mlir::failure();
        } else if (auto compare = llvm::dyn_cast<weftrvv::CompareOp>(op)) {
          if (mlir::failed(emitCompare(rewriter, loc, compare, valueMap,
                                       bodyVL)))
            return mlir::failure();
        } else if (auto maskedBinary =
                       llvm::dyn_cast<weftrvv::MaskedBinaryOp>(op)) {
          if (mlir::failed(emitMaskedBinary(rewriter, loc, maskedBinary,
                                            valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto maskedMacc =
                       llvm::dyn_cast<weftrvv::MaskedMAccOp>(op)) {
          if (mlir::failed(emitMaskedMAcc(rewriter, loc, maskedMacc, valueMap,
                                          bodyVL)))
            return mlir::failure();
        } else if (auto macc = llvm::dyn_cast<weftrvv::MAccOp>(op)) {
          if (mlir::failed(emitMAcc(rewriter, loc, macc, valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto maskAnd = llvm::dyn_cast<weftrvv::MaskAndOp>(op)) {
          if (mlir::failed(emitMaskAnd(rewriter, loc, maskAnd, valueMap,
                                       bodyVL)))
            return mlir::failure();
        } else if (auto widenConvert =
                       llvm::dyn_cast<weftrvv::WideningConvertOp>(op)) {
          if (mlir::failed(emitWideningConvert(rewriter, loc, widenConvert,
                                               valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto dequantize =
                       llvm::dyn_cast<weftrvv::DequantizeOp>(op)) {
          if (mlir::failed(emitDequantize(rewriter, loc, dequantize, valueMap,
                                          bodyVL)))
            return mlir::failure();
        } else if (auto select = llvm::dyn_cast<weftrvv::SelectOp>(op)) {
          if (mlir::failed(emitSelect(rewriter, loc, select, valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto reduce = llvm::dyn_cast<weftrvv::ReduceOp>(op)) {
          if (mlir::failed(emitReduce(rewriter, loc, reduce, valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto standaloneReduce =
                       llvm::dyn_cast<weftrvv::StandaloneReduceOp>(op)) {
          if (mlir::failed(emitStandaloneReduce(rewriter, loc, standaloneReduce,
                                                valueMap, standaloneOutBuffer,
                                                bodyVL)))
            return mlir::failure();
        } else if (auto maskedStandaloneReduce =
                       llvm::dyn_cast<weftrvv::MaskedStandaloneReduceOp>(op)) {
          if (mlir::failed(emitMaskedStandaloneReduce(
                  rewriter, loc, maskedStandaloneReduce, valueMap,
                  standaloneOutBuffer, bodyVL)))
            return mlir::failure();
        } else if (auto wproduct =
                       llvm::dyn_cast<weftrvv::WideningProductOp>(op)) {
          if (mlir::failed(emitWideningProduct(rewriter, loc, wproduct,
                                               valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto scaleProduct =
                       llvm::dyn_cast<weftrvv::BlockFp16ScaleProductOp>(op)) {
          if (mlir::failed(emitBlockFp16ScaleProduct(rewriter, loc, scaleProduct,
                                                     valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto computedScaleDequant =
                       llvm::dyn_cast<weftrvv::BlockComputedScaleDequantOp>(
                           op)) {
          if (mlir::failed(emitBlockComputedScaleDequant(
                  rewriter, loc, computedScaleDequant, valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto crossBlockAccumulate =
                       llvm::dyn_cast<weftrvv::CrossBlockF32AccumulateOp>(op)) {
          if (mlir::failed(emitCrossBlockF32Accumulate(
                  rewriter, loc, crossBlockAccumulate, valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto lane0Extract =
                       llvm::dyn_cast<weftrvv::TypedVectorLane0ToScalarExtractOp>(
                           op)) {
          if (mlir::failed(emitTypedVectorLane0ToScalarExtract(
                  rewriter, loc, lane0Extract, valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto offsetBinaryProduct = llvm::dyn_cast<
                       weftrvv::PackedI4OffsetBinaryXI8ProductOp>(op)) {
          if (mlir::failed(emitPackedI4OffsetBinaryXI8Product(
                  rewriter, loc, offsetBinaryProduct, valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto codebookTable =
                       llvm::dyn_cast<weftrvv::CodebookTableBroadcastOp>(op)) {
          if (mlir::failed(emitCodebookTableBroadcast(rewriter, loc,
                                                      codebookTable, valueMap)))
            return mlir::failure();
        } else if (auto codebookGather =
                       llvm::dyn_cast<weftrvv::CodebookGatherXI8ProductOp>(op)) {
          if (mlir::failed(emitCodebookGatherXI8Product(
                  rewriter, loc, codebookGather, valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto unsignedNibbleProduct =
                       llvm::dyn_cast<weftrvv::UnsignedNibbleXI8ProductOp>(op)) {
          if (mlir::failed(emitUnsignedNibbleXI8Product(
                  rewriter, loc, unsignedNibbleProduct, valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto qhSource =
                       llvm::dyn_cast<weftrvv::BlockFiveBitQhSourceOp>(op)) {
          if (mlir::failed(emitBlockFiveBitQhSource(rewriter, loc, qhSource,
                                                    valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto fiveBitProduct =
                       llvm::dyn_cast<weftrvv::FiveBitOffsetBinaryXI8ProductOp>(
                           op)) {
          if (mlir::failed(emitFiveBitOffsetBinaryXI8Product(
                  rewriter, loc, fiveBitProduct, valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto wmacc =
                       llvm::dyn_cast<weftrvv::WideningMAccOp>(op)) {
          if (mlir::failed(
                  emitWideningMAcc(rewriter, loc, wmacc, valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto dotReduce =
                       llvm::dyn_cast<weftrvv::WideningDotReduceOp>(op)) {
          if (mlir::failed(emitWideningDotReduce(rewriter, loc, dotReduce,
                                                 valueMap, standaloneOutBuffer,
                                                 bodyVL)))
            return mlir::failure();
        } else if (auto maskedDotReduce =
                       llvm::dyn_cast<weftrvv::MaskedWideningDotReduceOp>(op)) {
          if (mlir::failed(emitMaskedWideningDotReduce(
                  rewriter, loc, maskedDotReduce, valueMap, standaloneOutBuffer,
                  bodyVL)))
            return mlir::failure();
        } else if (auto binary = llvm::dyn_cast<weftrvv::BinaryOp>(op)) {
          if (mlir::failed(emitBinary(rewriter, loc, binary, valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto stridedStore =
                       llvm::dyn_cast<weftrvv::StridedStoreOp>(op)) {
          if (mlir::failed(emitStridedStore(rewriter, loc, stridedStore,
                                            valueMap, inductionVar, bodyVL)))
            return mlir::failure();
        } else if (auto indexedStore =
                       llvm::dyn_cast<weftrvv::IndexedStoreOp>(op)) {
          if (mlir::failed(emitIndexedStore(rewriter, loc, indexedStore,
                                            valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto maskedStore =
                       llvm::dyn_cast<weftrvv::MaskedStoreOp>(op)) {
          if (mlir::failed(emitMaskedStore(rewriter, loc, maskedStore, valueMap,
                                           inductionVar, bodyVL)))
            return mlir::failure();
        } else if (auto maskedStridedStore =
                       llvm::dyn_cast<weftrvv::MaskedStridedStoreOp>(op)) {
          if (mlir::failed(emitMaskedStridedStore(rewriter, loc,
                                                  maskedStridedStore, valueMap,
                                                  inductionVar, bodyVL)))
            return mlir::failure();
        } else if (auto maskedIndexedStore =
                       llvm::dyn_cast<weftrvv::MaskedIndexedStoreOp>(op)) {
          if (mlir::failed(emitMaskedIndexedStore(rewriter, loc,
                                                  maskedIndexedStore, valueMap,
                                                  bodyVL)))
            return mlir::failure();
        } else if (auto store = llvm::dyn_cast<weftrvv::StoreOp>(op)) {
          // The standalone reduction stores its lane-0 scalar result back to the
          // output buffer BASE (no `+ i` offset, VL=1): the running scalar lives
          // in out[0] across chunks. Mirror the legacy scalar-output store.
          if (isStandaloneReductionOp(store.getValue().getDefiningOp())) {
            auto resultVecType = llvm::dyn_cast<weftrvv::VectorType>(
                store.getValue().getType());
            mlir::Value value = valueMap.lookup(store.getValue());
            if (!resultVecType || !value || !standaloneOutBuffer)
              return rewriter.notifyMatchFailure(
                  store, "standalone reduction store operand unmapped");
            if (mlir::failed(emitStandaloneReductionScalarStore(
                    rewriter, loc, store, standaloneOutBuffer, value,
                    resultVecType)))
              return mlir::failure();
          } else {
            // The reduce family stores only lane 0 of the reduction result back
            // to the output chunk base, so its store VL is the literal 1 (not
            // the running chunk VL). Detect a reduce-sourced store and emit
            // VL=1; every other (elementwise) store keeps the chunk VL. This
            // mirrors the legacy `weft_rvv.reduction_store_vl = "1"` fact.
            mlir::Value storeVL = bodyVL;
            if (auto reduceDef =
                    store.getValue().getDefiningOp<weftrvv::ReduceOp>()) {
              mlir::StringAttr layout = reduceDef.getResultLayoutAttr();
              if (layout && layout.getValue() ==
                                "store-reduction-lane0-to-output-chunk-base")
                storeVL =
                    rewriter.create<emitc::LiteralOp>(loc, sizeType, "1");
            }
            if (mlir::failed(emitStore(rewriter, loc, store, valueMap,
                                       inductionVar, storeVL)))
              return mlir::failure();
          }
        } else {
          return rewriter.notifyMatchFailure(
              variant, "unsupported op in with_vl beachhead body");
        }
      }
    }

    return mlir::success();
  }

bool VariantToEmitCFunc::isLowPrecisionDequantBody(weftrvv::WithVLOp scope) {
    bool sawDequantChain = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      auto dequant = llvm::dyn_cast<weftrvv::DequantizeOp>(op);
      if (!dequant)
        continue;
      mlir::Operation *reduceDef = dequant.getSource().getDefiningOp();
      auto reduce = llvm::dyn_cast_or_null<weftrvv::StandaloneReduceOp>(
          reduceDef);
      if (!reduce)
        return false;
      mlir::Operation *productDef = reduce.getInput().getDefiningOp();
      if (!llvm::isa_and_nonnull<weftrvv::WideningProductOp,
                                 weftrvv::PackedI4NibbleUnpackProductOp>(
              productDef))
        return false;
      sawDequantChain = true;
    }
    return sawDequantChain;
  }

bool VariantToEmitCFunc::isDeferredWideDequantBody(weftrvv::WithVLOp scope) {
    bool sawDeferredChain = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      auto dequant = llvm::dyn_cast<weftrvv::DequantizeOp>(op);
      if (!dequant)
        continue;
      auto reduce = dequant.getSource().getDefiningOp<weftrvv::StandaloneReduceOp>();
      if (!reduce)
        return false;
      if (!reduce.getInput().getDefiningOp<weftrvv::WideningAccumulateOp>())
        return false;
      sawDeferredChain = true;
    }
    return sawDeferredChain;
  }

bool VariantToEmitCFunc::isQ4_0Q8_0BlockDotBody(weftrvv::WithVLOp scope) {
    bool sawBlockDot = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::GgmlBlockDotQ40Q80Op>(op)) {
        if (sawBlockDot)
          return false;
        sawBlockDot = true;
      } else {
        return false;
      }
    }
    return sawBlockDot;
  }

bool VariantToEmitCFunc::isQ4_0Q8_0GemmTileBody(weftrvv::WithVLOp scope) {
    bool sawTile = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::GgmlGemmTileQ40Q80Op>(op)) {
        if (sawTile)
          return false;
        sawTile = true;
      } else {
        return false;
      }
    }
    return sawTile;
  }

bool VariantToEmitCFunc::isQ4_0Q8_0GemmBody(weftrvv::WithVLOp scope) {
    bool sawGemm = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::GgmlGemmQ40Q80Op>(op)) {
        if (sawGemm)
          return false;
        sawGemm = true;
      } else {
        return false;
      }
    }
    return sawGemm;
  }

bool VariantToEmitCFunc::isRepackGemvQ5_0Q8_0Body(weftrvv::WithVLOp scope) {
    bool sawGemv = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::GgmlRepackGemvQ50Q80Op>(op)) {
        if (sawGemv)
          return false;
        sawGemv = true;
      } else {
        return false;
      }
    }
    return sawGemv;
  }

bool VariantToEmitCFunc::isRepackGemvQ5_1Q8_1Body(weftrvv::WithVLOp scope) {
    bool sawGemv = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::GgmlRepackGemvQ51Q81Op>(op)) {
        if (sawGemv)
          return false;
        sawGemv = true;
      } else {
        return false;
      }
    }
    return sawGemv;
  }

bool VariantToEmitCFunc::isPackQ4_0ToX16Body(weftrvv::WithVLOp scope) {
    bool sawPack = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::GgmlPackQ40ToX16Op>(op)) {
        if (sawPack)
          return false;
        sawPack = true;
      } else {
        return false;
      }
    }
    return sawPack;
  }

bool VariantToEmitCFunc::isRepackGemvQ4_1Q8_1Body(weftrvv::WithVLOp scope) {
    bool sawGemv = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::GgmlRepackGemvQ41Q81Op>(op)) {
        if (sawGemv)
          return false;
        sawGemv = true;
      } else {
        return false;
      }
    }
    return sawGemv;
  }

bool VariantToEmitCFunc::isRepackGemmQ4_1Q8_1Body(weftrvv::WithVLOp scope) {
    bool sawGemm = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::GgmlRepackGemmQ41Q81Op>(op)) {
        if (sawGemm)
          return false;
        sawGemm = true;
      } else {
        return false;
      }
    }
    return sawGemm;
  }

// NOTE (G3 主线A T2-construct): isRepackGemmQ4KQ8KBody RETIRED with its monolith op
// (the K-quant repack GEMM is now the constructed typed_repack_gemm_loop_body region).

bool VariantToEmitCFunc::isRepackGemvQ8_0Q8_0Body(weftrvv::WithVLOp scope) {
    bool sawGemv = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::GgmlRepackGemvQ80Q80Op>(op)) {
        if (sawGemv)
          return false;
        sawGemv = true;
      } else {
        return false;
      }
    }
    return sawGemv;
  }

// NOTE (G3 主线A T2-construct): isRepackGemvQ4KQ8KBody RETIRED with its monolith op
// (the K-quant repack GEVM is now the constructed typed_repack_gemv_loop_body region).

// NOTE (G3 主线A T3 format4): the q5_K 16x1-repacked GEVM + GEMM recognizers
// (isRepackGem{v,m}Q5KQ8KBody) are RETIRED with their direct emitters + monolith ops
// (the FIFTH K-quant super-block, the THIRD min-fold, COMPLETING the K-quant repack
// family). The q5_K repack is now CONSTRUCTED as the typed
// weft_rvv.typed_repack_gem{v,m}_loop_body region (fold_model "kquant_dmin_bsums_min",
// SHARED with q4_K) carrying the weft_rvv.repack_gem{v,m}_kquant_core brick (decode_model
// "q5_K"), lowered by isTypedRepackGem{v,m}LoopBody -> emitTypedRepackGem{v,m}LoopBody's
// K-quant MIN branch -> emitRepackKQuantGem{v,m}BodyQ5K (byte-exact GEVM / S6-tiled
// byte-exact GEMM; the register-cliff lever transfers like q2_K). q5_K == q4_K + the qh
// 5th-bit plane.

// NOTE (G3 主线A T3): the q6_K 16x1-repacked GEVM + GEMM recognizers
// (isRepackGem{v,m}Q6KQ8KBody) are RETIRED with their direct emitters + monolith ops.
// The q6_K repack is now CONSTRUCTED as the typed weft_rvv.typed_repack_gem{v,m}_loop_body
// region (fold_model "kquant_single_scale_no_min") carrying the
// weft_rvv.repack_gem{v,m}_kquant_core brick (decode_model "q6_K"), lowered by
// isTypedRepackGem{v,m}LoopBody -> emitTypedRepackGem{v,m}LoopBody's K-quant no-min
// branch -> emitRepackKQuantGem{v,m}BodyQ6K (byte-exact GEVM / S6-tiled byte-exact GEMM).

// NOTE (G3 主线A T3 format2): the q2_K 16x1-repacked GEVM + GEMM recognizers
// (isRepackGem{v,m}Q2KQ8KBody) are RETIRED with their direct emitters + monolith ops.
// The q2_K repack is now CONSTRUCTED as the typed weft_rvv.typed_repack_gem{v,m}_loop_body
// region (fold_model "kquant_dmin_bsums_min", SHARED with q4_K) carrying the
// weft_rvv.repack_gem{v,m}_kquant_core brick (decode_model "q2_K"), lowered by
// isTypedRepackGem{v,m}LoopBody -> emitTypedRepackGem{v,m}LoopBody's K-quant branch ->
// emitRepackKQuantGem{v,m}BodyQ2K (byte-exact GEVM / S6-tiled byte-exact GEMM).

// NOTE (G3 主线A T3 format3): the q3_K 16x1-repacked GEVM + GEMM recognizers
// (isRepackGem{v,m}Q3KQ8KBody) are RETIRED with their direct emitters + monolith ops
// (the SECOND no-min K-quant super-block, the LAST K-quant repack sibling). The q3_K
// repack is now CONSTRUCTED as the typed weft_rvv.typed_repack_gem{v,m}_loop_body region
// (fold_model "kquant_single_scale_no_min", SHARED with q6_K) carrying the
// weft_rvv.repack_gem{v,m}_kquant_core brick (decode_model "q3_K"), lowered by
// isTypedRepackGem{v,m}LoopBody -> emitTypedRepackGem{v,m}LoopBody's K-quant no-min
// branch -> emitRepackKQuantGem{v,m}BodyQ3K (byte-exact GEVM / PLAIN byte-exact GEMM;
// S6 tiling NULL for weight-reconstruction-bound q3_K).

// NOTE: isRepackGem{v,m}TQ20Q8KBody (G3 主线B batch1) AND isRepackGem{v,m}TQ10Q8KBody
// (G3 主线B batch2) -- the tq2_0 2-bit + tq1_0 base-3 direct-emitter recognizers --
// are RETIRED with their direct emitters (the whole ternary batch); the tq{2,1}_0
// repack GEVM/GEMM now flow through the typed_repack_gem{v,m}_loop_body front door
// (isTypedRepackGem{v,m}LoopBody -> the ternary branch, keyed off the in-region
// repack_gem{v,m}_ternary_core brick's decode_model).

// NOTE (G3 M2 iq4_nl codebook front-door): isRepackGem{v,m}Iq4NlQ80Body -- the iq4_nl
// flat CODEBOOK direct-emitter recognizers -- are RETIRED with their direct emitters + the
// monolith ops; the iq4_nl repack GEVM/GEMM now flow through the
// typed_repack_gem{v,m}_loop_body front door (isTypedRepackGem{v,m}LoopBody -> the codebook
// branch, keyed off the in-region repack_gem{v,m}_codebook_core brick's decode_model
// "iq4_nl" + its 16-entry non-linear codebook).

// NOTE (G3 retirement_batch 4 mxfp4 codebook front-door): isRepackGem{v,m}Mxfp4Q8Body -- the
// mxfp4 flat CODEBOOK + E8M0 direct-emitter recognizers -- are RETIRED with their direct
// emitters + the monolith ops; the mxfp4 repack GEVM/GEMM now flow through the
// typed_repack_gem{v,m}_loop_body front door (isTypedRepackGem{v,m}LoopBody -> the codebook
// branch, keyed off the in-region repack_gem{v,m}_codebook_core brick's decode_model "mxfp4"
// + fold_model "codebook_flat_e8m0_scale" + its 16-entry doubled-e2m1 codebook).

// NOTE (G3 M2-后 iq4_xs codebook front-door): isRepackGemvIq4XsQ8KBody +
// isRepackGemmIq4XsQ8KBody (+ the emitRepackGem{v,m}Iq4XsQ8K direct emitters + the
// GgmlRepackGem{v,m}Iq4XsQ8KOp monolith ops) are RETIRED: the iq4_xs repack now flows
// through the typed_repack_gem{v,m}_loop_body front door (isTypedRepackGem{v,m}LoopBody ->
// emitTypedRepackGem{v,m}LoopBody codebook branch, fold_model
// "codebook_superblock_signed6_no_min", decode_model "iq4_xs").

// NOTE (G3 M4 iq2-grid front-door): isRepackGem{v,m}Iq2XxsQ8KBody (first cell) AND
// isRepackGem{v,m}Iq2{Xs,S}Q8KBody (dual-scale siblings, COMPLETING the iq2 grid family) are
// RETIRED with their monolith ops -- the whole iq2 grid family (iq2_xxs / iq2_xs / iq2_s)
// repack GEVM/GEMM now flows through the typed_repack_gem{v,m}_loop_body front door (grid
// branch of emitTypedRepackGem{v,m}LoopBody -> emitRepackGridGem{v,m}BodyIq2Xxs for iq2_xxs
// single-ls / emitRepackGem{v,m}Iq2DualScaleQ8K for iq2_xs / iq2_s dual-ls).

bool VariantToEmitCFunc::isTypedFlatBlockDotLoopBody(weftrvv::WithVLOp scope) {
    bool sawLoopBody = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::TypedFlatBlockDotLoopBodyOp>(op)) {
        if (sawLoopBody)
          return false;
        sawLoopBody = true;
      } else {
        return false;
      }
    }
    return sawLoopBody;
  }

bool VariantToEmitCFunc::isTypedDequantizeRowLoopBody(weftrvv::WithVLOp scope) {
    bool sawLoopBody = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::TypedDequantizeRowLoopBodyOp>(op)) {
        if (sawLoopBody)
          return false;
        sawLoopBody = true;
      } else {
        return false;
      }
    }
    return sawLoopBody;
  }

bool VariantToEmitCFunc::isTypedQuantizeRowLoopBody(weftrvv::WithVLOp scope) {
    bool sawLoopBody = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::TypedQuantizeRowLoopBodyOp>(op)) {
        if (sawLoopBody)
          return false;
        sawLoopBody = true;
      } else {
        return false;
      }
    }
    return sawLoopBody;
  }

bool VariantToEmitCFunc::isTypedSuperBlockBlockDotLoopBody(
    weftrvv::WithVLOp scope) {
    bool sawLoopBody = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(op)) {
        if (sawLoopBody)
          return false;
        sawLoopBody = true;
      } else {
        return false;
      }
    }
    return sawLoopBody;
  }

bool VariantToEmitCFunc::isTypedRepackGemvLoopBody(weftrvv::WithVLOp scope) {
    bool sawLoopBody = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::TypedRepackGemvLoopBodyOp>(op)) {
        if (sawLoopBody)
          return false;
        sawLoopBody = true;
      } else {
        return false;
      }
    }
    return sawLoopBody;
  }

bool VariantToEmitCFunc::isTypedRepackGemvColgroupTiledLoopBody(
    weftrvv::WithVLOp scope) {
    bool sawLoopBody = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::TypedRepackGemvColgroupTiledLoopBodyOp>(op)) {
        if (sawLoopBody)
          return false;
        sawLoopBody = true;
      } else {
        return false;
      }
    }
    return sawLoopBody;
  }

bool VariantToEmitCFunc::isTypedRepackGemmLoopBody(weftrvv::WithVLOp scope) {
    bool sawLoopBody = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::TypedRepackGemmLoopBodyOp>(op)) {
        if (sawLoopBody)
          return false;
        sawLoopBody = true;
      } else {
        return false;
      }
    }
    return sawLoopBody;
  }

bool VariantToEmitCFunc::isQ1_0Q8_0BlockDotBody(weftrvv::WithVLOp scope) {
    bool sawBlockDot = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::GgmlBlockDotQ10Q80Op>(op)) {
        if (sawBlockDot)
          return false;
        sawBlockDot = true;
      } else {
        return false;
      }
    }
    return sawBlockDot;
  }

// NOTE: the monolith recognizer isIQ4XSQ8KBlockDotBody + emitter emitIQ4XSQ8KBlockDot
// were RETIRED at the iq4_xs flip (C_construct 23->24): the front door now constructs the
// typed super-block SCALAR-accumulator loop body (fold_model "scalar_delta_grid", stride
// 136) carrying the iq4_xs codebook-core brick, lowered by
// emitTypedSuperBlockScalarDeltaGridLoopBodyIq4xs (the byte-exact code-move of the retired
// monolith body, dispatched from emitTypedSuperBlockScalarDeltaGridLoopBody on the iq4_xs
// codebook-core brick identity). iq4_xs is the SUPER-BLOCK CODEBOOK sibling of the flat
// iq4_nl codebook (it REUSES iq4_nl's 16-entry vrgather codebook gather as its per-sub-block
// integer core, wrapped in the q4_K-style super-block signed 6-bit scale machinery).

// NOTE: the monolith recognizer isIQ2XXSQ8KBlockDotBody + emitter emitIQ2XXSQ8KBlockDot
// were RETIRED at the iq2_xxs flip (L3 coverage): the front door now constructs the
// typed super-block SCALAR-accumulator GRID loop body (fold_model "scalar_delta_grid",
// stride 66), lowered by emitTypedSuperBlockScalarDeltaGridLoopBodyIq2xxs, which reuses
// the SHARED byte-exact anchors emitIQ2XXSCanonicalGridTableDecl +
// emitIQ2XXSCanonicalSigns64TableDecl + emitIQ2XXSSuperBlockGridBody.

// NOTE: the monolith recognizer isIQ2XSQ8KBlockDotBody + emitter emitIQ2XSQ8KBlockDot
// were RETIRED at the iq2_xs flip (L3 coverage): the front door now constructs the typed
// super-block SCALAR-accumulator GRID loop body (fold_model "scalar_delta_grid", stride
// 74), lowered by emitTypedSuperBlockScalarDeltaGridLoopBodyIq2xs, which reuses the SHARED
// byte-exact anchors emitIQ2XSCanonicalGridTableDecl + emitIQ2XSCanonicalSigns64TableDecl +
// emitIQ2XSSuperBlockGridBody.

// NOTE: the monolith recognizer isIQ2SQ8KBlockDotBody + emitter emitIQ2SQ8KBlockDot
// were RETIRED at the iq2_s flip (L3 coverage): the front door now constructs the typed
// super-block SCALAR-accumulator GRID loop body (fold_model "scalar_delta_grid", stride
// 82), lowered by emitTypedSuperBlockScalarDeltaGridLoopBodyIq2s, which reuses the SHARED
// byte-exact anchors emitIQ2SCanonicalGridTableDecl + emitIQ2SCanonicalSigns256TableDecl +
// emitIQ2SSuperBlockGridBody.

// NOTE: the monolith recognizer isIQ3XXSQ8KBlockDotBody + emitter emitIQ3XXSQ8KBlockDot
// were RETIRED at the iq3_xxs flip (L3 coverage): the front door now constructs the
// typed super-block SCALAR-accumulator GRID loop body (fold_model "scalar_delta_grid",
// stride 98), lowered by emitTypedSuperBlockScalarDeltaGridLoopBodyIq3xxs, which reuses
// the SHARED byte-exact anchors emitIQ3XXSCanonicalGridTableDecl +
// emitIQ3XXSCanonicalKsignsTableDecl + emitIQ3XXSSuperBlockGridBody.

// NOTE: the monolith recognizer isIQ3SQ8KBlockDotBody + emitter emitIQ3SQ8KBlockDot
// were RETIRED at the iq3_s flip (C_construct 22->23): the front door now constructs the
// typed super-block SCALAR-accumulator GRID loop body (fold_model "scalar_delta_grid",
// stride 110), lowered by emitTypedSuperBlockScalarDeltaGridLoopBodyIq3s, which reuses
// the SHARED byte-exact anchors emitIQ3SCanonicalGridTableDecl +
// emitIQ3SSuperBlockGridBody.

// NOTE: the monolith recognizer isIQ1MQ8KBlockDotBody + emitter emitIQ1MQ8KBlockDot
// were RETIRED at the iq1_m flip (L3): the front door now constructs the typed
// super-block SCALAR-accumulator GRID loop body (fold_model "scalar_delta_grid",
// stride 56), lowered by emitTypedSuperBlockScalarDeltaGridLoopBodyIq1M, which reuses
// the SHARED byte-exact anchors emitIQ1MCanonicalGridTableDecl + emitIQ1MSuperBlockGridBody.

bool VariantToEmitCFunc::isMXFP4Q8_0BlockDotBody(weftrvv::WithVLOp scope) {
    bool sawBlockDot = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::GgmlBlockDotMXFP4Q80Op>(op)) {
        if (sawBlockDot)
          return false;
        sawBlockDot = true;
      } else {
        return false;
      }
    }
    return sawBlockDot;
  }

// NOTE: the monolith recognizer isNVFP4Q8_0BlockDotBody + emitter
// emitNVFP4Q8_0BlockDot were RETIRED at the nvfp4 flip (C_construct 27->28): the
// front door now constructs the typed FLAT block-dot loop body (fold_model
// "flat_nvfp4_codebook", stride 36) carrying the nvfp4 codebook-core brick, lowered
// by emitTypedFlatBlockDotLoopBody's flat_nvfp4_codebook branch through the SHARED
// emitNVFP4BlockDotBodyShared (the byte-exact code-move of the retired monolith
// body). nvfp4 is the SECOND FP4-CODEBOOK sibling of the flat q1_0 scaffold (its
// activation is a block_q8_0 stream); it REUSES mxfp4's 16-entry DOUBLED e2m1
// codebook gather wrapped in the per-sub-block UE4M3 fp8 weight scale. mxfp4 stays a
// monolith (the FP4-class negative control).

bool VariantToEmitCFunc::isQ6_KQ8_KAux32PartialBody(weftrvv::WithVLOp scope) {
    bool sawBlockDot = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::GgmlBlockDotQ6KQ8KAux32Op>(op)) {
        if (sawBlockDot)
          return false;
        sawBlockDot = true;
      } else {
        return false;
      }
    }
    return sawBlockDot;
  }

bool VariantToEmitCFunc::isQ4_KNibbleUnpackBody(weftrvv::WithVLOp scope) {
    bool sawUnpack = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::Q4KNibbleUnpackOp>(op)) {
        if (sawUnpack)
          return false;
        sawUnpack = true;
      } else {
        return false;
      }
    }
    return sawUnpack;
  }

bool VariantToEmitCFunc::isQ4_KScaleMinBitDanceBody(weftrvv::WithVLOp scope) {
    bool sawBitDance = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::Q4KScaleMinBitDanceOp>(op)) {
        if (sawBitDance)
          return false;
        sawBitDance = true;
      } else {
        return false;
      }
    }
    return sawBitDance;
  }

bool VariantToEmitCFunc::isQ4_KScaledDotBody(weftrvv::WithVLOp scope) {
    bool sawScaledDot = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::Q4KScaledDotOp>(op)) {
        if (sawScaledDot)
          return false;
        sawScaledDot = true;
      } else {
        return false;
      }
    }
    return sawScaledDot;
  }

bool VariantToEmitCFunc::isQ4_KMinTermBody(weftrvv::WithVLOp scope) {
    bool sawMinTerm = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::Q4KMinTermOp>(op)) {
        if (sawMinTerm)
          return false;
        sawMinTerm = true;
      } else {
        return false;
      }
    }
    return sawMinTerm;
  }

bool VariantToEmitCFunc::isQ4_KSumsFoldScaleDBody(weftrvv::WithVLOp scope) {
    bool sawFold = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::Q4KSumsFoldScaleDOp>(op)) {
        if (sawFold)
          return false;
        sawFold = true;
      } else {
        return false;
      }
    }
    return sawFold;
  }

bool VariantToEmitCFunc::isQ4_KHorizontalFoldBody(weftrvv::WithVLOp scope) {
    bool sawFold = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::Q4KHorizontalFoldOp>(op)) {
        if (sawFold)
          return false;
        sawFold = true;
      } else {
        return false;
      }
    }
    return sawFold;
  }

bool VariantToEmitCFunc::isQ4_KQ8_KAux32PartialBody(weftrvv::WithVLOp scope) {
    bool sawBlockDot = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::GgmlBlockDotQ4KQ8KAux32Op>(op)) {
        if (sawBlockDot)
          return false;
        sawBlockDot = true;
      } else {
        return false;
      }
    }
    return sawBlockDot;
  }


// NOTE: isTQ2_0Q8_KBlockDotBody (the monolith tq2_0 recognizer) was RETIRED at the
// tq2_0 flip (C_construct 24->25). The constructed typed super-block ternary loop body
// is recognized by isTypedSuperBlockBlockDotLoopBody + resolved via
// emitTypedSuperBlockScalarDeltaGridLoopBody -> emitTypedSuperBlockScalarDeltaGridLoopBodyTQ20.

// NOTE: isTQ1_0Q8_KBlockDotBody (the monolith tq1_0 recognizer) was RETIRED at the
// tq1_0 flip (C_construct 25->26). The constructed typed super-block BASE-3 ternary loop
// body is recognized by isTypedSuperBlockBlockDotLoopBody + resolved via
// emitTypedSuperBlockScalarDeltaGridLoopBody -> emitTypedSuperBlockScalarDeltaGridLoopBodyTQ10.

bool VariantToEmitCFunc::isTypedElementwiseLoopBody(weftrvv::WithVLOp scope) {
    bool sawLoopBody = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::TypedElementwiseLoopBodyOp>(op)) {
        if (sawLoopBody)
          return false;
        sawLoopBody = true;
      } else {
        return false;
      }
    }
    // EXCLUDE the soft_max reduce body: it RETURNS the f64 sum, so it is dispatched
    // by its own return-carrying branch, NOT this void-return kBlockDotKernels
    // table entry (which appends emitc.return with no operand). Every other
    // map/reduce body (scale/silu/rms_norm) returns void and stays in the table.
    if (sawLoopBody && isTypedElementwiseSoftMaxReduceLoopBody(scope))
      return false;
    return sawLoopBody;
  }

bool VariantToEmitCFunc::isTypedElementwiseSoftMaxReduceLoopBody(
    weftrvv::WithVLOp scope) {
    // EXACTLY one weft_rvv.typed_elementwise_loop_body whose region carries a
    // weft_rvv.elementwise_soft_max_reduce_core brick (the CONSTRUCTED soft_max
    // exp-sum-reduce body that RETURNS the f64 sum). The reduce brick identity is
    // the dispatch key.
    weftrvv::TypedElementwiseLoopBodyOp loopBody;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto lb = llvm::dyn_cast<weftrvv::TypedElementwiseLoopBodyOp>(op)) {
        if (loopBody)
          return false;
        loopBody = lb;
      } else {
        return false;
      }
    }
    if (!loopBody)
      return false;
    bool sawSoftMaxCore = false;
    loopBody.getBody().walk(
        [&](weftrvv::ElementwiseSoftMaxReduceCoreOp) { sawSoftMaxCore = true; });
    return sawSoftMaxCore;
  }

bool VariantToEmitCFunc::isGgmlQuantizeRowQ80Body(weftrvv::WithVLOp scope) {
    bool sawQuantize = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::GgmlQuantizeRowQ80Op>(op)) {
        if (sawQuantize)
          return false;
        sawQuantize = true;
      } else {
        return false;
      }
    }
    return sawQuantize;
  }

bool VariantToEmitCFunc::isGgmlQuantizeRowQ81Body(weftrvv::WithVLOp scope) {
    bool sawQuantize = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::GgmlQuantizeRowQ81Op>(op)) {
        if (sawQuantize)
          return false;
        sawQuantize = true;
      } else {
        return false;
      }
    }
    return sawQuantize;
  }

bool VariantToEmitCFunc::isGgmlQuantizeRowQ8KBody(weftrvv::WithVLOp scope) {
    bool sawQuantize = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::GgmlQuantizeRowQ8KOp>(op)) {
        if (sawQuantize)
          return false;
        sawQuantize = true;
      } else {
        return false;
      }
    }
    return sawQuantize;
  }

// NOTE: isGgmlForwardElementwiseF32Body was RETIRED at the support flip
// (dispatch-wired -> constructed, C_construct 73->77): add/mul/cpy/gelu are now
// CONSTRUCTED through the abstract weft_rvv.ggml_forward_elementwise source op +
// the pre-emitc front door (the SAME path as scale/silu/rms_norm/soft_max/rope),
// recognized by isTypedElementwiseLoopBody, so no dedicated support recognizer
// remains.

bool VariantToEmitCFunc::isGgmlDequantizeRowBody(weftrvv::WithVLOp scope) {
    // DISPATCH-WIRED marker: the body is EXACTLY one weft_rvv.dequantize_row op.
    // The op identity (+ its bounded `format`) is the dispatch key; the emitter
    // owns the hand-written per-format decode body (NOT a typed loop brick).
    bool sawDequant = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<weftrvv::GgmlDequantizeRowOp>(op)) {
        if (sawDequant)
          return false;
        sawDequant = true;
      } else {
        return false;
      }
    }
    return sawDequant;
  }

bool VariantToEmitCFunc::isDeferredWideDotReduceBody(weftrvv::WithVLOp scope) {
    bool sawDeferredChain = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      auto store = llvm::dyn_cast<weftrvv::StoreOp>(op);
      if (!store)
        continue;
      auto reduce =
          store.getValue().getDefiningOp<weftrvv::StandaloneReduceOp>();
      if (!reduce)
        return false;
      if (!reduce.getInput().getDefiningOp<weftrvv::DeferredAccumulateOp>())
        return false;
      sawDeferredChain = true;
    }
    return sawDeferredChain;
  }

bool VariantToEmitCFunc::isStandaloneDequantBody(weftrvv::WithVLOp scope) {
    weftrvv::LoadOp load;
    weftrvv::DequantizeOp dequant;
    weftrvv::StoreOp store;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto l = llvm::dyn_cast<weftrvv::LoadOp>(op)) {
        if (load)
          return false; // more than one load is not the standalone shape
        load = l;
      } else if (auto d = llvm::dyn_cast<weftrvv::DequantizeOp>(op)) {
        if (dequant)
          return false;
        dequant = d;
      } else if (auto s = llvm::dyn_cast<weftrvv::StoreOp>(op)) {
        if (store)
          return false;
        store = s;
      } else {
        return false; // any other op (product/reduce/select/...) excludes it
      }
    }
    if (!load || !dequant || !store)
      return false;
    // The dequantize must source the load and feed the store, kind/dtype pinned.
    if (dequant.getKind() != "i32_to_f32_scaled")
      return false;
    if (dequant.getSource() != load.getLoaded())
      return false;
    if (store.getValue() != dequant.getResult())
      return false;
    auto srcVec =
        llvm::dyn_cast<weftrvv::VectorType>(load.getLoaded().getType());
    auto resVec =
        llvm::dyn_cast<weftrvv::VectorType>(dequant.getResult().getType());
    if (!srcVec || !resVec)
      return false;
    if (!srcVec.getElementType().isSignlessInteger(32) ||
        !resVec.getElementType().isF32())
      return false;
    return true;
  }

bool VariantToEmitCFunc::isPureMaskedStoreBody(weftrvv::WithVLOp scope) {
    bool sawMaskedStore = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      // Compute / plain-store ops are not part of the pure masked-store shape.
      if (llvm::isa<weftrvv::BinaryOp, weftrvv::MaskedBinaryOp, weftrvv::MAccOp,
                    weftrvv::MaskedMAccOp, weftrvv::CompareOp, weftrvv::SelectOp,
                    weftrvv::ReduceOp, weftrvv::DequantizeOp, weftrvv::MaskAndOp,
                    weftrvv::StoreOp, weftrvv::StridedStoreOp,
                    weftrvv::IndexedStoreOp>(op))
        return false;
      if (llvm::isa<weftrvv::MaskedStoreOp>(op)) {
        if (sawMaskedStore)
          return false; // more than one store is not the bounded shape
        sawMaskedStore = true;
      }
    }
    return sawMaskedStore;
  }

bool VariantToEmitCFunc::isComputedMaskMaskedStoreBody(weftrvv::WithVLOp scope) {
    bool sawComputedMaskedStore = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      // Any compute op other than the mask chain (compare/splat) is out of this
      // shape, as is any other store-like op or a masked LOAD (a load-merge body
      // needs a `_tumu` masked load whose passthrough this exception does not
      // cover).
      if (llvm::isa<weftrvv::BinaryOp, weftrvv::MaskedBinaryOp, weftrvv::MAccOp,
                    weftrvv::MaskedMAccOp, weftrvv::SelectOp, weftrvv::ReduceOp,
                    weftrvv::DequantizeOp, weftrvv::MaskAndOp,
                    weftrvv::MaskedLoadOp, weftrvv::MaskedStridedLoadOp,
                    weftrvv::MaskedIndexedLoadOp, weftrvv::StoreOp,
                    weftrvv::StridedStoreOp, weftrvv::IndexedStoreOp,
                    weftrvv::MaskedStridedStoreOp,
                    weftrvv::MaskedIndexedStoreOp>(op))
        return false;
      if (auto maskedStore = llvm::dyn_cast<weftrvv::MaskedStoreOp>(op)) {
        if (sawComputedMaskedStore)
          return false; // more than one store is not the bounded shape
        auto compare =
            maskedStore.getMask().getDefiningOp<weftrvv::CompareOp>();
        if (!compare)
          return false; // a buffer-mask store is the isPureMaskedStoreBody shape
        // The legitimate unit store-only computed-mask family is runtime-scalar:
        // its compare RHS is a splat of a runtime scalar. Refuse a vector-vector
        // compare unit store-only body (not a real family).
        if (!compare.getRhs().getDefiningOp<weftrvv::SplatOp>())
          return false;
        sawComputedMaskedStore = true;
      }
    }
    return sawComputedMaskedStore;
  }

mlir::LogicalResult
VariantToEmitCFunc::checkCapabilityConfigGate(mlir::ConversionPatternRewriter &rewriter,
                          weft::exec::VariantOp variant,
                          weft::exec::KernelOp kernel, unsigned bodySEW,
                          llvm::StringRef bodyLMUL,
                          bool bodyRequiresAgnosticPolicy) const {
    auto requiresAttr = variant->getAttrOfType<mlir::ArrayAttr>("requires");
    if (!requiresAttr)
      return mlir::success();
    std::string bodySEWToken = llvm::Twine(bodySEW).str();
    // The provider property is a comma-separated allow-list (e.g. "32,64").
    auto listIncludes = [](llvm::StringRef list, llvm::StringRef token) {
      llvm::SmallVector<llvm::StringRef, 4> entries;
      list.split(entries, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
      for (llvm::StringRef entry : entries)
        if (entry.trim() == token)
          return true;
      return false;
    };
    for (mlir::Attribute entry : requiresAttr) {
      auto symbolRef = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(entry);
      if (!symbolRef)
        continue;
      // Resolve the requires symbol to a provider op in the kernel body. The
      // provider is a capability or target op carrying the optional supported_*
      // allow-lists.
      mlir::Operation *provider = nullptr;
      for (mlir::Operation &op : kernel.getBody().front()) {
        auto sym = op.getAttrOfType<mlir::StringAttr>(
            mlir::SymbolTable::getSymbolAttrName());
        if (sym && sym.getValue() == symbolRef.getValue()) {
          provider = &op;
          break;
        }
      }
      if (!provider)
        continue;
      if (auto supportedSEW =
              provider->getAttrOfType<mlir::StringAttr>("supported_sew")) {
        llvm::StringRef value = supportedSEW.getValue().trim();
        if (!value.empty() && !listIncludes(value, bodySEWToken))
          return rewriter.notifyMatchFailure(
              variant, "capability provider supported_sew excludes typed body "
                       "SEW (capability gates this body out)");
      }
      if (auto supportedLMUL =
              provider->getAttrOfType<mlir::StringAttr>("supported_lmul")) {
        llvm::StringRef value = supportedLMUL.getValue().trim();
        if (!value.empty() && !listIncludes(value, bodyLMUL))
          return rewriter.notifyMatchFailure(
              variant, "capability provider supported_lmul excludes typed body "
                       "LMUL (capability gates this body out)");
      }
      // RVV ISA-generation gate (the deepest N1 divergence axis). The
      // tail/mask-agnostic (ta/ma) vector policy is a RATIFIED RVV1.0 feature
      // that RVV0.7 (xtheadvector / C920) does NOT have. So a body that requires
      // the agnostic policy is RVV1.0-only: if the resolved provider declares
      // `rvv_version` = "0.7", the capability gates this body out, exactly as the
      // supported_sew/lmul allow-lists gate an unsupported (sew, lmul). This is
      // gated on the version CAPABILITY FACT read off the provider op (I3: no
      // family-name / march-string branch in the gate). The gate is silent when
      // the provider declares no `rvv_version` or declares "1.0" (the agnostic
      // policy is legal there) -- so rv64gcv behaviour is byte-identical.
      if (bodyRequiresAgnosticPolicy) {
        if (auto rvvVersion =
                provider->getAttrOfType<mlir::StringAttr>("rvv_version")) {
          if (rvvVersion.getValue().trim() == "0.7")
            return rewriter.notifyMatchFailure(
                variant,
                "capability provider rvv_version=0.7 lacks the ratified "
                "tail/mask-agnostic policy the typed body requires (RVV0.7 "
                "ISA generation gates this body out)");
        }
      }
    }
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitLoad(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
         weftrvv::LoadOp load,
         llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
         mlir::Value inductionVar, mlir::Value bodyVL,
         mlir::Value extraOffset) const {
    auto vectorType =
        llvm::dyn_cast<weftrvv::VectorType>(load.getLoaded().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(load, "load result not typed vector");
    mlir::Value base = valueMap.lookup(load.getBuffer());
    if (!base)
      return rewriter.notifyMatchFailure(load, "load buffer not an ABI param");
    if (!bufferPointeeMatchesVectorElement(base, vectorType))
      return rewriter.notifyMatchFailure(
          load, "load buffer C type disagrees with loaded vector element");

    // Resolve the EmitC vector type FIRST, before creating any emitc op, so a
    // family the beachhead converter does not cover (e.g. lmul m2) fails the
    // match cleanly and rolls back, instead of leaving a half-converted
    // call_opaque whose result is still a `!weft_rvv.vector<...>` type.
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(load, "vector type not convertible");
    std::string callee =
        riscvIntrinsicName("vle", vectorElementWidth(vectorType),
                           vectorType.getLmul(), vectorDType(vectorType));
    mlir::Value loaded = emitOpaqueCallBuilt(
        rewriter, loc, vecType, callee,
        load.getWEFTEmitCLowerableSourceOpName(),
        load.getWEFTEmitCLowerableSourceRole(),
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value ptr =
              b.create<emitc::AddOp>(l, base.getType(), base, inductionVar);
          if (extraOffset)
            ptr = b.create<emitc::AddOp>(l, base.getType(), ptr, extraOffset);
          return {ptr, bodyVL};
        });
    valueMap[load.getLoaded()] = loaded;
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitBinary(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
           weftrvv::BinaryOp binary,
           llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
           mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<weftrvv::VectorType>(binary.getResult().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(binary,
                                         "binary result not typed vector");
    mlir::Value lhs = valueMap.lookup(binary.getLhs());
    mlir::Value rhs = valueMap.lookup(binary.getRhs());
    if (!lhs || !rhs)
      return rewriter.notifyMatchFailure(binary, "binary operand unmapped");

    std::optional<llvm::StringRef> mnemonic =
        binaryMnemonic(binary.getKind(), isFloatVector(vectorType));
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
    mlir::Value result = emitOpaqueCall(
        rewriter, loc, vecType, callee, mlir::ValueRange{lhs, rhs, bodyVL},
        binary.getWEFTEmitCLowerableSourceOpName(),
        binary.getWEFTEmitCLowerableSourceRole());
    valueMap[binary.getResult()] = result;
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitReduce(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
           weftrvv::ReduceOp reduce,
           llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
           mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<weftrvv::VectorType>(reduce.getResult().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(reduce, "reduce result not vector");
    // The accumulator seed must be a typed vector (the rhs-load chunk seed).
    if (!llvm::isa<weftrvv::VectorType>(reduce.getAccumulator().getType()))
      return rewriter.notifyMatchFailure(
          reduce, "reduce accumulator is not a typed vector seed");
    // Malformed-body guard (mirrors the legacy `isReduction && hasRHSBroadcastLike`
    // rejection, RVVEmitCRoutePlanning.cpp:20663-20667): the reduction route
    // requires an EXPLICIT vector input AND accumulator LOAD. A
    // broadcast/splat-seeded accumulator (or input) is NOT in this bounded slice
    // -- the running per-chunk lane-0 seed must be a real loaded vector, not a
    // scalar splat. Reject any input/accumulator not produced by a plain
    // weft_rvv.load so a broadcast/splat-seeded reduce body falls back to the
    // legacy validator (which errors) instead of being silently mislowered.
    if (!reduce.getInput().getDefiningOp<weftrvv::LoadOp>() ||
        !reduce.getAccumulator().getDefiningOp<weftrvv::LoadOp>())
      return rewriter.notifyMatchFailure(
          reduce, "reduce input/accumulator must be explicit vector loads "
                  "(broadcast/splat seed is outside the convertible slice)");
    // Only the per-chunk-base result layout is the `reduce` family. The
    // scalar-carry standalone layout is a DIFFERENT (still-owned) family.
    mlir::StringAttr resultLayout = reduce.getResultLayoutAttr();
    if (!resultLayout ||
        resultLayout.getValue() !=
            "store-reduction-lane0-to-output-chunk-base")
      return rewriter.notifyMatchFailure(
          reduce, "reduce result layout outside the convertible reduce family");

    mlir::Value input = valueMap.lookup(reduce.getInput());
    mlir::Value accumulator = valueMap.lookup(reduce.getAccumulator());
    if (!input || !accumulator)
      return rewriter.notifyMatchFailure(reduce, "reduce operand unmapped");

    std::optional<llvm::StringRef> mnemonic =
        reductionMnemonic(reduce.getKind());
    if (!mnemonic)
      return rewriter.notifyMatchFailure(reduce, "unsupported reduce kind");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(reduce, "vector type not convertible");
    std::string callee = riscvReductionIntrinsicName(
        *mnemonic, vectorElementWidth(vectorType), vectorType.getLmul(),
        vectorDType(vectorType));
    mlir::Value result = emitOpaqueCall(
        rewriter, loc, vecType, callee,
        mlir::ValueRange{input, accumulator, bodyVL},
        reduce.getWEFTEmitCLowerableSourceOpName(),
        reduce.getWEFTEmitCLowerableSourceRole());
    valueMap[reduce.getResult()] = result;
    return mlir::success();
  }

bool VariantToEmitCFunc::isStandaloneReductionOp(mlir::Operation *op) {
    if (!op)
      return false;
    if (auto standalone = llvm::dyn_cast<weftrvv::StandaloneReduceOp>(op)) {
      mlir::StringAttr layout = standalone.getResultLayoutAttr();
      return layout && layout.getValue() ==
                           "store-standalone-reduction-lane0-to-output-scalar";
    }
    if (auto masked = llvm::dyn_cast<weftrvv::MaskedStandaloneReduceOp>(op)) {
      mlir::StringAttr layout = masked.getResultLayoutAttr();
      return layout && layout.getValue() ==
                           "store-standalone-reduction-lane0-to-output-scalar";
    }
    // The widening dot-product reduction (plain / strided-input / computed-mask)
    // carries the SAME scalar-carry-through-output structure: an i32 seed read
    // from the accumulator-input buffer pre-loop, a running seed read back from
    // out[0] each chunk, and a lane-0 result stored to the output base (VL=1).
    // Its result layout is `store-dot-reduction-lane0-to-output-scalar`.
    if (auto dot = llvm::dyn_cast<weftrvv::WideningDotReduceOp>(op)) {
      mlir::StringAttr layout = dot.getResultLayoutAttr();
      return layout && layout.getValue() ==
                           "store-dot-reduction-lane0-to-output-scalar";
    }
    if (auto maskedDot =
            llvm::dyn_cast<weftrvv::MaskedWideningDotReduceOp>(op)) {
      mlir::StringAttr layout = maskedDot.getResultLayoutAttr();
      return layout && layout.getValue() ==
                           "store-dot-reduction-lane0-to-output-scalar";
    }
    return false;
  }

bool VariantToEmitCFunc::isStandaloneReductionBody(weftrvv::WithVLOp scope) {
    for (mlir::Operation &op : scope.getBody().front())
      if (isStandaloneReductionOp(&op))
        return true;
    return false;
  }

std::optional<llvm::StringRef>
VariantToEmitCFunc::standaloneReductionMnemonic(llvm::StringRef kind) {
    if (kind == "add")
      return llvm::StringRef("vredsum");
    if (kind == "min")
      return llvm::StringRef("vredmin");
    if (kind == "max")
      return llvm::StringRef("vredmax");
    if (kind == "signed_widening_reduce_add")
      return llvm::StringRef("vwredsum");
    if (kind == "unsigned_widening_reduce_add")
      return llvm::StringRef("vwredsumu");
    return std::nullopt;
  }

std::optional<llvm::StringRef>
VariantToEmitCFunc::maskedStandaloneReductionNeutral(llvm::StringRef kind, unsigned sew) {
    if (kind == "add")
      return llvm::StringRef("0");
    if (kind == "min")
      return sew == 64 ? llvm::StringRef("9223372036854775807")
                       : llvm::StringRef("2147483647");
    if (kind == "max")
      return sew == 64 ? llvm::StringRef("(-9223372036854775807-1)")
                       : llvm::StringRef("(-2147483647-1)");
    return std::nullopt;
  }

mlir::Value VariantToEmitCFunc::emitScalarSeedSplat(mlir::ConversionPatternRewriter &rewriter,
                                mlir::Location loc, mlir::Value buffer,
                                weftrvv::VectorType resultVecType,
                                llvm::StringRef sourceOpName,
                                llvm::StringRef sourceRole) const {
    auto pointer = llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(buffer);
    if (!pointer)
      return nullptr;
    if (!bufferPointeeMatchesVectorElement(buffer, resultVecType))
      return nullptr;
    mlir::Type vecType = convertVectorTypeToEmitC(resultVecType);
    if (!vecType)
      return nullptr;
    if (isFloatVector(resultVecType))
      return nullptr; // standalone reduction is the integer-accumulator slice.
    std::string callee =
        riscvIntrinsicName("vmv_v_x", vectorElementWidth(resultVecType),
                           resultVecType.getLmul(), vectorDType(resultVecType));
    // base[0]: subscript -> lvalue -> load reads the first scalar element. The
    // pointee const-ness (const int32_t* acc vs int32_t* out) flows through the
    // lvalue value type, so the seed temp prints `const int32_t` / `int32_t` to
    // match the legacy oracle automatically.
    return emitOpaqueCallBuilt(
        rewriter, loc, vecType, callee, sourceOpName, sourceRole,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value index =
              b.create<emitc::LiteralOp>(l, b.getIndexType(), "0");
          emitc::SubscriptOp subscriptOp =
              b.create<emitc::SubscriptOp>(l, pointer, index);
          auto lvalueType =
              llvm::cast<emitc::LValueType>(subscriptOp.getResult().getType());
          mlir::Value scalar =
              b.create<emitc::LoadOp>(l, lvalueType.getValueType(),
                                      subscriptOp.getResult())
                  .getResult();
          mlir::Value one =
              b.create<emitc::LiteralOp>(l, getSizeType(rewriter), "1");
          return {scalar, one};
        });
  }

mlir::Type VariantToEmitCFunc::getSizeType(mlir::ConversionPatternRewriter &rewriter) {
    return emitc::OpaqueType::get(rewriter.getContext(), "size_t");
  }

mlir::LogicalResult VariantToEmitCFunc::emitStandaloneReductionPreLoopSeed(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::WithVLOp scope,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    mlir::Operation *reduceOp = nullptr;
    weftrvv::StoreOp storeOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (isStandaloneReductionOp(&op))
        reduceOp = &op;
      if (auto store = llvm::dyn_cast<weftrvv::StoreOp>(op))
        storeOp = store;
    }
    if (!reduceOp || !storeOp)
      return rewriter.notifyMatchFailure(scope,
                                         "standalone reduction body missing "
                                         "reduce/store");

    mlir::Value accSeed;
    mlir::Value resultValue;
    llvm::StringRef sourceOpName;
    llvm::StringRef sourceRole;
    if (auto standalone = llvm::dyn_cast<weftrvv::StandaloneReduceOp>(reduceOp)) {
      accSeed = standalone.getAccumulatorSeed();
      resultValue = standalone.getResult();
      sourceOpName = standalone.getWEFTEmitCLowerableSourceOpName();
      sourceRole = standalone.getWEFTEmitCLowerableSourceRole();
    } else if (auto masked =
                   llvm::dyn_cast<weftrvv::MaskedStandaloneReduceOp>(reduceOp)) {
      accSeed = masked.getAccumulatorSeed();
      resultValue = masked.getResult();
      sourceOpName = masked.getWEFTEmitCLowerableSourceOpName();
      sourceRole = masked.getWEFTEmitCLowerableSourceRole();
    } else if (auto dot =
                   llvm::dyn_cast<weftrvv::WideningDotReduceOp>(reduceOp)) {
      accSeed = dot.getAccumulatorSeed();
      resultValue = dot.getResult();
      sourceOpName = dot.getWEFTEmitCLowerableSourceOpName();
      sourceRole = dot.getWEFTEmitCLowerableSourceRole();
    } else {
      auto maskedDot =
          llvm::cast<weftrvv::MaskedWideningDotReduceOp>(reduceOp);
      accSeed = maskedDot.getAccumulatorSeed();
      resultValue = maskedDot.getResult();
      sourceOpName = maskedDot.getWEFTEmitCLowerableSourceOpName();
      sourceRole = maskedDot.getWEFTEmitCLowerableSourceRole();
    }
    auto resultVecType =
        llvm::dyn_cast<weftrvv::VectorType>(resultValue.getType());
    if (!resultVecType || resultVecType.getLmul() != "m1")
      return rewriter.notifyMatchFailure(
          scope, "standalone reduction result must be an m1 vector");

    mlir::Value accBuffer = valueMap.lookup(accSeed);
    mlir::Value outBuffer = valueMap.lookup(storeOp.getBuffer());
    if (!accBuffer || !outBuffer)
      return rewriter.notifyMatchFailure(
          scope, "standalone reduction acc/out buffers unmapped");

    // out[0] = acc[0]: splat the accumulator seed read, store to out base VL=1.
    mlir::Value seedSplat = emitScalarSeedSplat(rewriter, loc, accBuffer,
                                                resultVecType, sourceOpName,
                                                sourceRole);
    if (!seedSplat)
      return rewriter.notifyMatchFailure(
          scope, "standalone reduction pre-loop seed not convertible");
    if (mlir::failed(emitStandaloneReductionScalarStore(
            rewriter, loc, storeOp, outBuffer, seedSplat, resultVecType)))
      return mlir::failure();
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitStandaloneReductionScalarStore(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::StoreOp store, mlir::Value outBuffer, mlir::Value value,
    weftrvv::VectorType resultVecType) const {
    if (!bufferPointeeMatchesVectorElement(outBuffer, resultVecType))
      return rewriter.notifyMatchFailure(
          store, "standalone reduction output buffer element mismatch");
    if (!convertVectorTypeToEmitC(resultVecType))
      return rewriter.notifyMatchFailure(store,
                                         "standalone result type not convertible");
    std::string callee =
        riscvIntrinsicName("vse", vectorElementWidth(resultVecType),
                           resultVecType.getLmul(), vectorDType(resultVecType));
    // Void interleave (the VL=1 literal is built between the comment and the
    // call): no full-callee void-built helper exists, so split the mangler
    // string back into mnemonic + suffix at the first underscore. Split-then-
    // rejoin with one underscore is byte-exact identity (emitVCallVoidBuilt
    // rebuilds "__riscv_" + mnemonic + "_" + suffix); do not grep-replace.
    auto [vseMnemonic, vseSuffix] =
        llvm::StringRef(callee)
            .drop_front(llvm::StringLiteral("__riscv_").size())
            .split('_');
    emitVCallVoidBuilt(rewriter, loc, vseMnemonic, vseSuffix,
                       store.getWEFTEmitCLowerableSourceOpName(),
                       store.getWEFTEmitCLowerableSourceRole(),
                       [&](mlir::OpBuilder &b, mlir::Location l)
                           -> llvm::SmallVector<mlir::Value> {
                         mlir::Value one = b.create<emitc::LiteralOp>(
                             l, getSizeType(rewriter), "1");
                         return {outBuffer, value, one};
                       });
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitStandaloneReduce(mlir::ConversionPatternRewriter &rewriter,
                     mlir::Location loc, weftrvv::StandaloneReduceOp reduce,
                     llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                     mlir::Value outBuffer, mlir::Value bodyVL) const {
    auto resultVecType =
        llvm::dyn_cast<weftrvv::VectorType>(reduce.getResult().getType());
    auto srcVecType =
        llvm::dyn_cast<weftrvv::VectorType>(reduce.getInput().getType());
    if (!resultVecType || !srcVecType)
      return rewriter.notifyMatchFailure(reduce,
                                         "standalone reduce types not vectors");
    if (resultVecType.getLmul() != "m1")
      return rewriter.notifyMatchFailure(
          reduce, "standalone reduce result must be m1");
    // The input must be an explicit vector load (no broadcast/splat seed) --
    // mirrors the legacy reduction input-load requirement. The
    // widening-product-reduce family chains the reduction directly onto the
    // weft_rvv.widening_product result (load -> load -> vwmul -> vwredsum), so a
    // widening_product input is the OTHER convertible producer: it is a real
    // typed vector dataflow value, not a broadcast/splat scalar seed. The
    // asymmetric offset-binary packed-i4 x plain-i8 product (ggml Q4_0 x Q8_0
    // integer core) chains the SAME way (load x3 -> one-sided decode + vwmul/
    // vwmacc -> vwredsum), so its i16mf2 result is likewise a real typed vector
    // producer for the reduction.
    {
      mlir::Operation *inputDef = reduce.getInput().getDefiningOp();
      if (!inputDef ||
          !llvm::isa<weftrvv::LoadOp, weftrvv::WideningProductOp,
                     weftrvv::PackedI4OffsetBinaryXI8ProductOp,
                     weftrvv::CodebookGatherXI8ProductOp,
                     weftrvv::UnsignedNibbleXI8ProductOp,
                     weftrvv::FiveBitOffsetBinaryXI8ProductOp>(inputDef))
        return rewriter.notifyMatchFailure(
            reduce, "standalone reduce input must be an explicit vector load, "
                    "widening_product, packed-i4 offset-binary x i8 product, "
                    "codebook-gather x i8 product, unsigned-nibble x i8 "
                    "product, or five-bit offset-binary x i8 product result");
    }
    std::optional<llvm::StringRef> mnemonic =
        standaloneReductionMnemonic(reduce.getKind());
    if (!mnemonic)
      return rewriter.notifyMatchFailure(reduce,
                                         "unsupported standalone reduce kind");
    mlir::Value input = valueMap.lookup(reduce.getInput());
    if (!input)
      return rewriter.notifyMatchFailure(reduce, "standalone reduce input "
                                                 "unmapped");
    mlir::Type vecType = convertVectorTypeToEmitC(resultVecType);
    if (!vecType || !convertVectorTypeToEmitC(srcVecType))
      return rewriter.notifyMatchFailure(reduce,
                                         "standalone reduce type not convertible");
    // In-loop running-seed read: out[0] -> splat m1.
    mlir::Value seed = emitScalarSeedSplat(
        rewriter, loc, outBuffer, resultVecType,
        reduce.getWEFTEmitCLowerableSourceOpName(),
        reduce.getWEFTEmitCLowerableSourceRole());
    if (!seed)
      return rewriter.notifyMatchFailure(
          reduce, "standalone reduce running seed not convertible");
    std::string callee = riscvWideningReductionIntrinsicName(
        *mnemonic, vectorDType(srcVecType), srcVecType.getLmul(),
        vectorDType(resultVecType));
    mlir::Value result = emitOpaqueCall(
        rewriter, loc, vecType, callee, mlir::ValueRange{input, seed, bodyVL},
        reduce.getWEFTEmitCLowerableSourceOpName(),
        reduce.getWEFTEmitCLowerableSourceRole());
    valueMap[reduce.getResult()] = result;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitMaskedStandaloneReduce(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::MaskedStandaloneReduceOp reduce,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap, mlir::Value outBuffer,
    mlir::Value bodyVL) const {
    auto resultVecType =
        llvm::dyn_cast<weftrvv::VectorType>(reduce.getResult().getType());
    auto srcVecType =
        llvm::dyn_cast<weftrvv::VectorType>(reduce.getInput().getType());
    if (!resultVecType || !srcVecType)
      return rewriter.notifyMatchFailure(
          reduce, "masked standalone reduce types not vectors");
    if (resultVecType.getLmul() != "m1")
      return rewriter.notifyMatchFailure(
          reduce, "masked standalone reduce result must be m1");
    std::optional<llvm::StringRef> mnemonic =
        standaloneReductionMnemonic(reduce.getKind());
    if (!mnemonic)
      return rewriter.notifyMatchFailure(
          reduce, "unsupported masked standalone reduce kind");
    std::optional<llvm::StringRef> neutral = maskedStandaloneReductionNeutral(
        reduce.getKind(), vectorElementWidth(srcVecType));
    if (!neutral)
      return rewriter.notifyMatchFailure(
          reduce, "masked standalone reduce neutral not derivable");
    mlir::Value mask = valueMap.lookup(reduce.getMask());
    mlir::Value source = valueMap.lookup(reduce.getInput());
    if (!mask || !source)
      return rewriter.notifyMatchFailure(reduce,
                                         "masked standalone reduce operand "
                                         "unmapped");
    mlir::Type resultEmitC = convertVectorTypeToEmitC(resultVecType);
    mlir::Type srcEmitC = convertVectorTypeToEmitC(srcVecType);
    if (!resultEmitC || !srcEmitC)
      return rewriter.notifyMatchFailure(
          reduce, "masked standalone reduce type not convertible");

    // Neutral splat over the masked-out lanes (input lmul, running bodyVL).
    std::string neutralCallee =
        riscvIntrinsicName("vmv_v_x", vectorElementWidth(srcVecType),
                           srcVecType.getLmul(), vectorDType(srcVecType));
    mlir::Value neutralVec = emitOpaqueCallBuilt(
        rewriter, loc, srcEmitC, neutralCallee,
        reduce.getWEFTEmitCLowerableSourceOpName(),
        reduce.getWEFTEmitCLowerableSourceRole(),
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value neutralLiteral = b.create<emitc::LiteralOp>(
              l, resultIntScalarType(rewriter), neutral->str());
          return {neutralLiteral, bodyVL};
        });

    // Merge active source lanes over the neutral background.
    std::string mergeCallee =
        riscvIntrinsicName("vmerge", vectorElementWidth(srcVecType),
                           srcVecType.getLmul(), vectorDType(srcVecType));
    mlir::Value masked = emitOpaqueCall(
        rewriter, loc, srcEmitC, mergeCallee,
        mlir::ValueRange{neutralVec, source, mask, bodyVL},
        reduce.getWEFTEmitCLowerableSourceOpName(),
        reduce.getWEFTEmitCLowerableSourceRole());

    // In-loop running-seed read: out[0] -> splat m1.
    mlir::Value seed = emitScalarSeedSplat(
        rewriter, loc, outBuffer, resultVecType,
        reduce.getWEFTEmitCLowerableSourceOpName(),
        reduce.getWEFTEmitCLowerableSourceRole());
    if (!seed)
      return rewriter.notifyMatchFailure(
          reduce, "masked standalone reduce running seed not convertible");

    std::string callee = riscvWideningReductionIntrinsicName(
        *mnemonic, vectorDType(srcVecType), srcVecType.getLmul(),
        vectorDType(resultVecType));
    mlir::Value result = emitOpaqueCall(
        rewriter, loc, resultEmitC, callee,
        mlir::ValueRange{masked, seed, bodyVL},
        reduce.getWEFTEmitCLowerableSourceOpName(),
        reduce.getWEFTEmitCLowerableSourceRole());
    valueMap[reduce.getResult()] = result;
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitWideningProduct(mlir::ConversionPatternRewriter &rewriter,
                    mlir::Location loc, weftrvv::WideningProductOp product,
                    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                    mlir::Value bodyVL) const {
    auto resultVecType =
        llvm::dyn_cast<weftrvv::VectorType>(product.getResult().getType());
    auto lhsVecType =
        llvm::dyn_cast<weftrvv::VectorType>(product.getLhs().getType());
    if (!resultVecType || !lhsVecType)
      return rewriter.notifyMatchFailure(product,
                                         "widening product types not vectors");
    // The signed rung emits vwmul; the unsigned low-precision rung (ui8 source ->
    // ui16 result, kind=unsigned_widening_product) emits vwmulu, byte-identical
    // to the legacy unsigned widening-product oracle. Both must agree: a signed
    // kind on unsigned vectors (or vice versa) is a malformed body -> fall back.
    const bool unsignedProduct =
        product.getKind() == "unsigned_widening_product";
    if (product.getKind() != "signed_widening_product" && !unsignedProduct)
      return rewriter.notifyMatchFailure(
          product, "only the signed/unsigned widening product is convertible");
    if (unsignedProduct != isUnsignedVector(resultVecType) ||
        unsignedProduct != isUnsignedVector(lhsVecType))
      return rewriter.notifyMatchFailure(
          product, "widening product kind/signedness mismatch with vector types");
    mlir::Type resultEmitC = convertVectorTypeToEmitC(resultVecType);
    if (!resultEmitC || !convertVectorTypeToEmitC(lhsVecType))
      return rewriter.notifyMatchFailure(
          product, "widening product type not convertible");
    mlir::Value lhs = valueMap.lookup(product.getLhs());
    mlir::Value rhs = valueMap.lookup(product.getRhs());
    if (!lhs || !rhs)
      return rewriter.notifyMatchFailure(product,
                                         "widening product operand unmapped");
    // The widened product intrinsic dtype/lmul derive from the RESULT vector.
    std::string callee = riscvIntrinsicName(
        unsignedProduct ? "vwmulu" : "vwmul", vectorElementWidth(resultVecType),
        resultVecType.getLmul(), vectorDType(resultVecType));
    mlir::Value result = emitOpaqueCall(
        rewriter, loc, resultEmitC, callee,
        mlir::ValueRange{lhs, rhs, bodyVL},
        product.getWEFTEmitCLowerableSourceOpName(),
        product.getWEFTEmitCLowerableSourceRole());
    valueMap[product.getResult()] = result;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitPackedI4NibbleUnpackProduct(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::PackedI4NibbleUnpackProductOp packed,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    mlir::Value bodyVL) const {
    auto resultVecType =
        llvm::dyn_cast<weftrvv::VectorType>(packed.getResult().getType());
    auto srcVecType =
        llvm::dyn_cast<weftrvv::VectorType>(packed.getLhs().getType());
    if (!resultVecType || !srcVecType)
      return rewriter.notifyMatchFailure(packed,
                                         "packed-i4 product types not vectors");
    mlir::Type resultEmitC = convertVectorTypeToEmitC(resultVecType);
    mlir::Type srcEmitC = convertVectorTypeToEmitC(srcVecType);
    if (!resultEmitC || !srcEmitC)
      return rewriter.notifyMatchFailure(
          packed, "packed-i4 product type not convertible");
    mlir::Value lhs = valueMap.lookup(packed.getLhs());
    mlir::Value rhs = valueMap.lookup(packed.getRhs());
    if (!lhs || !rhs)
      return rewriter.notifyMatchFailure(packed,
                                         "packed-i4 product operand unmapped");
    llvm::StringRef opName = packed.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = packed.getWEFTEmitCLowerableSourceRole();
    unsigned srcSEW = vectorElementWidth(srcVecType);
    llvm::StringRef srcLmul = srcVecType.getLmul();
    llvm::StringRef srcDtype = vectorDType(srcVecType);
    unsigned resSEW = vectorElementWidth(resultVecType);
    llvm::StringRef resLmul = resultVecType.getLmul();
    llvm::StringRef resDtype = vectorDType(resultVecType);
    mlir::Type u8Type = emitc::OpaqueType::get(rewriter.getContext(), "uint8_t");

    // Shift-by-immediate intrinsics spell as __riscv_<mnemonic>_<dtype><lmul>
    // (the `vx` form is part of the mnemonic), distinct from the `_vv_` form
    // riscvIntrinsicName builds for the binary product intrinsics.
    auto shift = [&](llvm::StringRef mnemonic, llvm::StringRef dtype,
                     llvm::StringRef lmul, mlir::Type vecType, mlir::Value src,
                     llvm::StringRef amount) -> mlir::Value {
      std::string callee =
          riscvScalarImmediateIntrinsicName(mnemonic, dtype, lmul);
      return emitOpaqueCallBuilt(
          rewriter, loc, vecType, callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value amt =
                b.create<emitc::LiteralOp>(l, u8Type, amount.str());
            return {src, amt, bodyVL};
          });
    };
    (void)srcSEW;
    (void)resSEW;

    // Low nibble: vsll(4) both sources into the high nibble.
    mlir::Value lhsLow =
        shift("vsll_vx", srcDtype, srcLmul, srcEmitC, lhs, "4");
    mlir::Value rhsLow =
        shift("vsll_vx", srcDtype, srcLmul, srcEmitC, rhs, "4");
    // vwmul widening product of the shifted low nibbles.
    std::string mulCallee = riscvIntrinsicName("vwmul", resSEW, resLmul, resDtype);
    mlir::Value lowProduct = emitOpaqueCall(
        rewriter, loc, resultEmitC, mulCallee,
        mlir::ValueRange{lhsLow, rhsLow, bodyVL}, opName, role);
    // vsra(8) rescales the i16 product (sign-extends + undoes the 2x4 shift).
    mlir::Value product =
        shift("vsra_vx", resDtype, resLmul, resultEmitC, lowProduct, "8");
    // High nibble: vsra(4) sign-extends the high nibble of each source in place.
    mlir::Value lhsHigh =
        shift("vsra_vx", srcDtype, srcLmul, srcEmitC, lhs, "4");
    mlir::Value rhsHigh =
        shift("vsra_vx", srcDtype, srcLmul, srcEmitC, rhs, "4");
    // vwmacc adds the high-nibble widening product into the accumulator.
    std::string maccCallee =
        riscvIntrinsicName("vwmacc", resSEW, resLmul, resDtype);
    mlir::Value pairSum = emitOpaqueCall(
        rewriter, loc, resultEmitC, maccCallee,
        mlir::ValueRange{product, lhsHigh, rhsHigh, bodyVL}, opName, role);
    valueMap[packed.getResult()] = pairSum;
    return mlir::success();
  }

mlir::FailureOr<mlir::Value> VariantToEmitCFunc::emitOffsetBinaryDecodeProductValue(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weight, mlir::Value actLow, mlir::Value actHigh,
    mlir::Value bodyVL, mlir::Type srcEmitC, mlir::Type resultEmitC,
    llvm::StringRef srcDtype, llvm::StringRef srcLmul, unsigned resSEW,
    llvm::StringRef resLmul, llvm::StringRef resDtype, llvm::StringRef opName,
    llvm::StringRef role) const {
    // The combined decode + product is the back-to-back composition of the two
    // factored halves (decode -> v0/v1, then the asymmetric widening product),
    // emitting the SAME six nodes in the SAME order INC-1's symmetric packed-i4
    // emitter does. The block-dot G1 GEMM tile (INC-14) hoists the decode half
    // ABOVE the M-column loop and replays the product half per column, reusing
    // the decoded v0/v1 lanes -- that weight-decode-reuse split is exactly why
    // these are factored. Callers that do NOT reuse the decode (the per-row
    // strip core, the standalone packed-i4 op) get byte-identical node output.
    std::pair<mlir::Value, mlir::Value> decoded = emitOffsetBinaryDecodeValue(
        rewriter, loc, weight, bodyVL, srcEmitC, srcDtype, srcLmul, opName, role);
    return emitOffsetBinaryProductFromDecodedValue(
        rewriter, loc, decoded.first, decoded.second, actLow, actHigh, bodyVL,
        resultEmitC, resSEW, resLmul, resDtype, opName, role);
  }

std::pair<mlir::Value, mlir::Value> VariantToEmitCFunc::emitOffsetBinaryDecodeValue(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weight, mlir::Value bodyVL, mlir::Type srcEmitC,
    llvm::StringRef srcDtype, llvm::StringRef srcLmul, llvm::StringRef opName,
    llvm::StringRef role) const {
    mlir::Type i32Type = emitc::OpaqueType::get(rewriter.getContext(), "int");
    mlir::Type u8Type = emitc::OpaqueType::get(rewriter.getContext(), "uint8_t");

    // Scalar-immediate intrinsics spell __riscv_<mnemonic>_<dtype><lmul> (the
    // `vx` form is part of the mnemonic), as in the symmetric packed-i4 emitter.
    auto immOp = [&](llvm::StringRef mnemonic, mlir::Value src,
                     llvm::StringRef amount,
                     mlir::Type amtType) -> mlir::Value {
      std::string callee =
          riscvScalarImmediateIntrinsicName(mnemonic, srcDtype, srcLmul);
      return emitOpaqueCallBuilt(
          rewriter, loc, srcEmitC, callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value amt =
                b.create<emitc::LiteralOp>(l, amtType, amount.str());
            return {src, amt, bodyVL};
          });
    };

    // Offset-binary -> two's-complement: xor both nibbles of every packed byte
    // with 0x88 (the vxor.vx scalar is a signed int).
    mlir::Value weightXor = immOp("vxor_vx", weight, "0x88", i32Type);
    // Low nibble: shift into the high nibble then arithmetic-shift back to
    // sign-extend it into a plain signed [-8,7] i8 lane.
    mlir::Value weightLowShifted = immOp("vsll_vx", weightXor, "4", u8Type);
    mlir::Value v0 = immOp("vsra_vx", weightLowShifted, "4", u8Type);
    // High nibble: arithmetic-shift sign-extends it in place.
    mlir::Value v1 = immOp("vsra_vx", weightXor, "4", u8Type);
    return {v0, v1};
  }

mlir::FailureOr<mlir::Value> VariantToEmitCFunc::emitOffsetBinaryProductFromDecodedValue(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value v0, mlir::Value v1, mlir::Value actLow, mlir::Value actHigh,
    mlir::Value bodyVL, mlir::Type resultEmitC, unsigned resSEW,
    llvm::StringRef resLmul, llvm::StringRef resDtype, llvm::StringRef opName,
    llvm::StringRef role) const {
    // Asymmetric widening product: decoded i8 weight x plain i8 activation.
    std::string mulCallee = riscvIntrinsicName("vwmul", resSEW, resLmul, resDtype);
    mlir::Value lowProduct = emitOpaqueCall(
        rewriter, loc, resultEmitC, mulCallee,
        mlir::ValueRange{v0, actLow, bodyVL}, opName, role);
    // vwmacc accumulates the high-nibble x high-activation widening product.
    std::string maccCallee =
        riscvIntrinsicName("vwmacc", resSEW, resLmul, resDtype);
    mlir::Value pairSum = emitOpaqueCall(
        rewriter, loc, resultEmitC, maccCallee,
        mlir::ValueRange{lowProduct, v1, actHigh, bodyVL}, opName, role);
    return pairSum;
  }

mlir::FailureOr<mlir::Value> VariantToEmitCFunc::emitUnsignedNibbleDecodeProductValue(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weightU8, mlir::Value actLow, mlir::Value actHigh,
    mlir::Value bodyVL, mlir::Type srcI8EmitC, mlir::Type srcU8EmitC,
    mlir::Type resultEmitC, llvm::StringRef srcLmul, unsigned resSEW,
    llvm::StringRef resLmul, llvm::StringRef resDtype, llvm::StringRef opName,
    llvm::StringRef role) const {
    mlir::Type i32Type = emitc::OpaqueType::get(rewriter.getContext(), "int");

    // Scalar-immediate intrinsics spell __riscv_<mnemonic>_u8<lmul> (the `vx`
    // form is part of the mnemonic). The decode runs on the UNSIGNED weight lane.
    auto immOp = [&](llvm::StringRef mnemonic, mlir::Value src,
                     llvm::StringRef amount) -> mlir::Value {
      std::string callee =
          riscvScalarImmediateIntrinsicName(mnemonic, "u8", srcLmul);
      return emitOpaqueCallBuilt(
          rewriter, loc, srcU8EmitC, callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value amt =
                b.create<emitc::LiteralOp>(l, i32Type, amount.str());
            return {src, amt, bodyVL};
          });
    };

    // Low nibble: mask the low 4 bits ([0,15]). High nibble: logical-shift the
    // top 4 bits down ([0,15]). Both on the UNSIGNED lane.
    mlir::Value xLow = immOp("vand_vx", weightU8, "0x0F");
    mlir::Value xHigh = immOp("vsrl_vx", weightU8, "0x04");

    // Reinterpret the unsigned nibble lanes to signed i8 (value-identity for
    // 0..15) so they feed the SAME signed widening product the offset-binary core
    // uses. The reinterpret intrinsic is __riscv_vreinterpret_v_u8<L>_i8<L>.
    auto reinterpret = [&](mlir::Value src) -> mlir::Value {
      std::string callee =
          riscvReinterpretIntrinsicName("u8", srcLmul, "i8", srcLmul);
      return emitOpaqueCall(rewriter, loc, srcI8EmitC, callee,
                            mlir::ValueRange{src}, opName, role);
    };
    mlir::Value v0 = reinterpret(xLow);
    mlir::Value v1 = reinterpret(xHigh);

    // Asymmetric widening product: decoded i8 weight x plain i8 activation.
    std::string mulCallee = riscvIntrinsicName("vwmul", resSEW, resLmul, resDtype);
    mlir::Value lowProduct = emitOpaqueCall(
        rewriter, loc, resultEmitC, mulCallee,
        mlir::ValueRange{v0, actLow, bodyVL}, opName, role);
    // vwmacc accumulates the high-nibble x high-activation widening product.
    std::string maccCallee =
        riscvIntrinsicName("vwmacc", resSEW, resLmul, resDtype);
    mlir::Value pairSum = emitOpaqueCall(
        rewriter, loc, resultEmitC, maccCallee,
        mlir::ValueRange{lowProduct, v1, actHigh, bodyVL}, opName, role);
    return pairSum;
  }

mlir::FailureOr<mlir::Value> VariantToEmitCFunc::emitFiveBitOffsetBinaryDecodeProductValue(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weightU8, mlir::Value actLow, mlir::Value actHigh,
    mlir::Value qhLow16, mlir::Value qhHigh16, mlir::Value chunkOffset,
    mlir::Value bodyVL, mlir::Type srcI8EmitC, mlir::Type srcU8EmitC,
    mlir::Type wideU16EmitC, mlir::Type resultEmitC, llvm::StringRef srcLmul,
    llvm::StringRef wideLmul, unsigned resSEW, llvm::StringRef resLmul,
    llvm::StringRef resDtype, llvm::StringRef opName, llvm::StringRef role,
    bool applyOffsetBias) const {
    mlir::Type i32Type = emitc::OpaqueType::get(rewriter.getContext(), "int");

    // u8 scalar-immediate op on the nibble lane (vand 0x0F / vsrl 0x04).
    auto u8ImmOp = [&](llvm::StringRef mnemonic, mlir::Value src,
                       llvm::StringRef amount) -> mlir::Value {
      std::string callee =
          riscvScalarImmediateIntrinsicName(mnemonic, "u8", srcLmul);
      return emitOpaqueCallBuilt(
          rewriter, loc, srcU8EmitC, callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value amt =
                b.create<emitc::LiteralOp>(l, i32Type, amount.str());
            return {src, amt, bodyVL};
          });
    };

    // u16 (wide-LMUL) ops for the per-lane 5th-bit extraction.
    auto u16ImmOp = [&](llvm::StringRef mnemonic, mlir::Value src,
                        llvm::StringRef amount) -> mlir::Value {
      std::string callee =
          riscvScalarImmediateIntrinsicName(mnemonic, "u16", wideLmul);
      return emitOpaqueCallBuilt(
          rewriter, loc, wideU16EmitC, callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value amt =
                b.create<emitc::LiteralOp>(l, i32Type, amount.str());
            return {src, amt, bodyVL};
          });
    };

    // The per-element 5th-bit lane from a broadcast 16-bit qh half: a vid+c shift
    // vector selects each lane's bit, isolate {0,1}, place at bit 4 -> {0,16},
    // narrow u16->u8. Identical for both halves off their respective qh scalar.
    auto fifthBitLane = [&](mlir::Value qhHalf16) -> mlir::Value {
      // idx = vid_v_u16<W>(vl);  sh = vadd_vx_u16<W>(idx, c, vl);
      std::string vidCallee = riscvIotaIntrinsicName("u16", wideLmul);
      mlir::Value idx = emitOpaqueCall(rewriter, loc, wideU16EmitC, vidCallee,
                                       mlir::ValueRange{bodyVL}, opName, role);
      std::string addCallee =
          riscvScalarImmediateIntrinsicName("vadd_vx", "u16", wideLmul);
      mlir::Value sh = emitOpaqueCall(
          rewriter, loc, wideU16EmitC, addCallee,
          mlir::ValueRange{idx, chunkOffset, bodyVL}, opName, role);
      // bits = vmv_v_x_u16<W>(qhHalf16, vl);  (broadcast the 16-bit qh half)
      std::string bcastCallee =
          riscvIntrinsicName("vmv_v_x", 16, wideLmul, "u16");
      mlir::Value bits = emitOpaqueCall(
          rewriter, loc, wideU16EmitC, bcastCallee,
          mlir::ValueRange{qhHalf16, bodyVL}, opName, role);
      // bits = vsrl_vv_u16<W>(bits, sh, vl);  (per-lane bit -> b0)
      std::string srlCallee = riscvIntrinsicName("vsrl", 16, wideLmul, "u16");
      mlir::Value srl = emitOpaqueCall(rewriter, loc, wideU16EmitC, srlCallee,
                                       mlir::ValueRange{bits, sh, bodyVL},
                                       opName, role);
      // bits = vand 1 -> {0,1}; bits = vsll 4 -> {0,16}.
      mlir::Value masked = u16ImmOp("vand_vx", srl, "0x1");
      mlir::Value shifted = u16ImmOp("vsll_vx", masked, "0x4");
      // narrow u16<W> -> u8<core>.
      std::string ncvtCallee = riscvNarrowingConvertIntrinsicName("u8", srcLmul);
      return emitOpaqueCall(rewriter, loc, srcU8EmitC, ncvtCallee,
                            mlir::ValueRange{shifted, bodyVL}, opName, role);
    };

    // Low/high 4-bit nibbles (unsigned), then OR in each half's 5th bit.
    mlir::Value xLow = u8ImmOp("vand_vx", weightU8, "0x0F");
    mlir::Value xHigh = u8ImmOp("vsrl_vx", weightU8, "0x04");
    mlir::Value lowHB = fifthBitLane(qhLow16);
    mlir::Value hiHB = fifthBitLane(qhHigh16);

    auto u8VVOp = [&](llvm::StringRef mnemonic, mlir::Value a,
                      mlir::Value b) -> mlir::Value {
      // VV form: riscvIntrinsicName's default arm spells __riscv_<op>_vv_u8<L>.
      std::string callee = riscvIntrinsicName(mnemonic, 8, srcLmul, "u8");
      return emitOpaqueCall(rewriter, loc, srcU8EmitC, callee,
                            mlir::ValueRange{a, b, bodyVL}, opName, role);
    };
    mlir::Value fiveLow = u8VVOp("vor", xLow, lowHB);   // [0,31]
    mlir::Value fiveHigh = u8VVOp("vor", xHigh, hiHB);  // [0,31]

    // Reinterpret to signed (value-identity for 0..31), then -- for the
    // offset-binary q5_0 path only -- apply the `-16` bias -> i8 [-16,15],
    // feeding the SAME signed widening product. When applyOffsetBias is false
    // (the q5_1 UNSIGNED 5-bit path: ggml's q5_1 reconstructs an unsigned q5 in
    // [0,31] with NO `-16`, mirroring q4_1's unsigned nibble) the bias `vsub` is
    // skipped, so the 5th-bit injection is SHARED byte-for-byte and only the
    // final bias op differs. The two halves still differ ONLY in which qh bits
    // feed them (already handled upstream).
    auto reinterpretBias = [&](mlir::Value u) -> mlir::Value {
      std::string reCallee =
          riscvReinterpretIntrinsicName("u8", srcLmul, "i8", srcLmul);
      mlir::Value s = emitOpaqueCall(rewriter, loc, srcI8EmitC, reCallee,
                                     mlir::ValueRange{u}, opName, role);
      if (!applyOffsetBias)
        return s;
      std::string subCallee =
          riscvScalarImmediateIntrinsicName("vsub_vx", "i8", srcLmul);
      return emitOpaqueCallBuilt(
          rewriter, loc, srcI8EmitC, subCallee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value bias = b.create<emitc::LiteralOp>(l, i32Type, "16");
            return {s, bias, bodyVL};
          });
    };
    mlir::Value v0 = reinterpretBias(fiveLow);
    mlir::Value v1 = reinterpretBias(fiveHigh);

    // Asymmetric widening product: decoded i8 weight x plain i8 activation.
    std::string mulCallee = riscvIntrinsicName("vwmul", resSEW, resLmul, resDtype);
    mlir::Value lowProduct = emitOpaqueCall(
        rewriter, loc, resultEmitC, mulCallee,
        mlir::ValueRange{v0, actLow, bodyVL}, opName, role);
    std::string maccCallee =
        riscvIntrinsicName("vwmacc", resSEW, resLmul, resDtype);
    mlir::Value pairSum = emitOpaqueCall(
        rewriter, loc, resultEmitC, maccCallee,
        mlir::ValueRange{lowProduct, v1, actHigh, bodyVL}, opName, role);
    return pairSum;
  }

mlir::LogicalResult VariantToEmitCFunc::emitPackedI4OffsetBinaryXI8Product(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::PackedI4OffsetBinaryXI8ProductOp packed,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    mlir::Value bodyVL) const {
    auto resultVecType =
        llvm::dyn_cast<weftrvv::VectorType>(packed.getResult().getType());
    auto srcVecType =
        llvm::dyn_cast<weftrvv::VectorType>(packed.getWeight().getType());
    if (!resultVecType || !srcVecType)
      return rewriter.notifyMatchFailure(
          packed, "offset-binary packed-i4 x i8 product types not vectors");
    mlir::Type resultEmitC = convertVectorTypeToEmitC(resultVecType);
    mlir::Type srcEmitC = convertVectorTypeToEmitC(srcVecType);
    if (!resultEmitC || !srcEmitC)
      return rewriter.notifyMatchFailure(
          packed, "offset-binary packed-i4 x i8 product type not convertible");
    mlir::Value weight = valueMap.lookup(packed.getWeight());
    mlir::Value actLow = valueMap.lookup(packed.getActivationLow());
    mlir::Value actHigh = valueMap.lookup(packed.getActivationHigh());
    if (!weight || !actLow || !actHigh)
      return rewriter.notifyMatchFailure(
          packed, "offset-binary packed-i4 x i8 product operand unmapped");
    llvm::StringRef opName = packed.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = packed.getWEFTEmitCLowerableSourceRole();
    mlir::FailureOr<mlir::Value> pairSum = emitOffsetBinaryDecodeProductValue(
        rewriter, loc, weight, actLow, actHigh, bodyVL, srcEmitC, resultEmitC,
        vectorDType(srcVecType), srcVecType.getLmul(),
        vectorElementWidth(resultVecType), resultVecType.getLmul(),
        vectorDType(resultVecType), opName, role);
    if (mlir::failed(pairSum))
      return mlir::failure();
    valueMap[packed.getResult()] = *pairSum;
    return mlir::success();
  }

mlir::FailureOr<mlir::Value> VariantToEmitCFunc::emitCodebookGatherDecodeProductValue(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value weightU8, mlir::Value table, mlir::Value actLow,
    mlir::Value actHigh, mlir::Value bodyVL, mlir::Type srcI8EmitC,
    mlir::Type srcU8EmitC, mlir::Type resultEmitC, llvm::StringRef srcLmul,
    unsigned resSEW, llvm::StringRef resLmul, llvm::StringRef resDtype,
    llvm::StringRef opName, llvm::StringRef role) const {
    mlir::Type i32Type = emitc::OpaqueType::get(rewriter.getContext(), "int");

    // u8 scalar-immediate op on the nibble lane (vand 0x0F low / vsrl 0x04 high).
    // The callee mangle + the "int"-typed immediate literal are byte-identical to
    // the monolithic codebook block-dot's per-strip decode (RVVToEmitCCodebookFp4).
    auto u8ImmOp = [&](llvm::StringRef mnemonic, mlir::Value src,
                       llvm::StringRef amount) -> mlir::Value {
      std::string callee = ("__riscv_" + mnemonic + "_u8" + srcLmul).str();
      return emitOpaqueCallBuilt(
          rewriter, loc, srcU8EmitC, callee, opName, role,
          [&](mlir::OpBuilder &b,
              mlir::Location l) -> llvm::SmallVector<mlir::Value> {
            mlir::Value amt =
                rewriter.create<emitc::LiteralOp>(loc, i32Type, amount.str());
            return {src, amt, bodyVL};
          });
    };
    mlir::Value idxLow = u8ImmOp("vand_vx", weightU8, "0x0F");
    mlir::Value idxHigh = u8ImmOp("vsrl_vx", weightU8, "0x04");

    // The codebook gather: map each UNSIGNED index lane through the broadcast
    // table into signed-i8 weight lanes v0/v1 (vrgather_vv_i8<L>(values, idx)).
    std::string gatherCallee = ("__riscv_vrgather_vv_i8" + srcLmul).str();
    auto gather = [&](mlir::Value idx) -> mlir::Value {
      return emitOpaqueCall(rewriter, loc, srcI8EmitC, gatherCallee,
                            mlir::ValueRange{table, idx, bodyVL}, opName, role);
    };
    mlir::Value v0 = gather(idxLow);
    mlir::Value v1 = gather(idxHigh);

    // The SAME asymmetric signed widening product the offset-binary sibling uses
    // (vwmul low <-> q8[0..15], vwmacc + high <-> q8[16..31]) -> i16 product.
    return emitOffsetBinaryProductFromDecodedValue(rewriter, loc, v0, v1, actLow,
                                                   actHigh, bodyVL, resultEmitC,
                                                   resSEW, resLmul, resDtype,
                                                   opName, role);
  }

mlir::LogicalResult VariantToEmitCFunc::emitCodebookTableBroadcast(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::CodebookTableBroadcastOp table,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    mlir::MLIRContext *ctx = rewriter.getContext();
    auto resultVecType =
        llvm::dyn_cast<weftrvv::VectorType>(table.getResult().getType());
    if (!resultVecType)
      return rewriter.notifyMatchFailure(table,
                                         "codebook table result not a vector");
    mlir::Type resultEmitC = convertVectorTypeToEmitC(resultVecType);
    if (!resultEmitC)
      return rewriter.notifyMatchFailure(
          table, "codebook table vector type not convertible");
    llvm::StringRef opName = table.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = table.getWEFTEmitCLowerableSourceRole();
    llvm::ArrayRef<int8_t> codebook = table.getCodebook();
    llvm::StringRef tableSymbol = table.getTableSymbol();

    // The 16-entry non-linear codebook as a structured `static const int8_t[16]`
    // decl ONCE (the task-sanctioned structured const), byte-identical to the
    // monolithic codebook emitter's table decl.
    std::string decl = ("static const int8_t " + tableSymbol + "[" +
                        std::to_string(codebook.size()) + "] = {")
                           .str();
    for (size_t i = 0; i < codebook.size(); ++i) {
      if (i)
        decl += ", ";
      decl += std::to_string(static_cast<int>(codebook[i]));
    }
    decl += "};";
    rewriter.create<emitc::VerbatimOp>(loc, decl);

    // vint8<L> values = __riscv_vle8_v_i8<L>(<table_symbol>, 16);  (the codebook
    // table broadcast into a vreg; reused by every gather). The table pointer is
    // the structured-const decl above, cast to const int8_t *.
    std::string tableLoadCallee =
        riscvIntrinsicName("vle", 8, resultVecType.getLmul(), "i8");
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Value values = emitOpaqueCallBuilt(
        rewriter, loc, resultEmitC, tableLoadCallee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value tableName = rewriter.create<emitc::LiteralOp>(
              loc, i8PtrType, tableSymbol.str());
          mlir::Value count = rewriter.create<emitc::LiteralOp>(
              loc, getSizeType(rewriter), std::to_string(codebook.size()));
          return {tableName, count};
        },
        llvm::StringRef("codebook_table_load"));
    valueMap[table.getResult()] = values;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitCodebookGatherXI8Product(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::CodebookGatherXI8ProductOp gather,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    mlir::Value bodyVL) const {
    auto resultVecType =
        llvm::dyn_cast<weftrvv::VectorType>(gather.getResult().getType());
    auto weightVecType =
        llvm::dyn_cast<weftrvv::VectorType>(gather.getWeight().getType());
    auto tableVecType =
        llvm::dyn_cast<weftrvv::VectorType>(gather.getTable().getType());
    if (!resultVecType || !weightVecType || !tableVecType)
      return rewriter.notifyMatchFailure(
          gather, "codebook gather x i8 product types not vectors");
    mlir::Type resultEmitC = convertVectorTypeToEmitC(resultVecType);
    mlir::Type srcU8EmitC = convertVectorTypeToEmitC(weightVecType);
    mlir::Type srcI8EmitC = convertVectorTypeToEmitC(tableVecType);
    if (!resultEmitC || !srcU8EmitC || !srcI8EmitC)
      return rewriter.notifyMatchFailure(
          gather, "codebook gather x i8 product type not convertible");
    mlir::Value weight = valueMap.lookup(gather.getWeight());
    mlir::Value table = valueMap.lookup(gather.getTable());
    mlir::Value actLow = valueMap.lookup(gather.getActivationLow());
    mlir::Value actHigh = valueMap.lookup(gather.getActivationHigh());
    if (!weight || !table || !actLow || !actHigh)
      return rewriter.notifyMatchFailure(
          gather, "codebook gather x i8 product operand unmapped");
    llvm::StringRef opName = gather.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = gather.getWEFTEmitCLowerableSourceRole();
    mlir::FailureOr<mlir::Value> product = emitCodebookGatherDecodeProductValue(
        rewriter, loc, weight, table, actLow, actHigh, bodyVL, srcI8EmitC,
        srcU8EmitC, resultEmitC, weightVecType.getLmul(),
        vectorElementWidth(resultVecType), resultVecType.getLmul(),
        vectorDType(resultVecType), opName, role);
    if (mlir::failed(product))
      return mlir::failure();
    valueMap[gather.getResult()] = *product;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitUnsignedNibbleXI8Product(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::UnsignedNibbleXI8ProductOp product,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    mlir::Value bodyVL) const {
    auto resultVecType =
        llvm::dyn_cast<weftrvv::VectorType>(product.getResult().getType());
    auto weightVecType =
        llvm::dyn_cast<weftrvv::VectorType>(product.getWeight().getType());
    if (!resultVecType || !weightVecType)
      return rewriter.notifyMatchFailure(
          product, "unsigned-nibble packed-i4 x i8 product types not vectors");
    // The weight is UNSIGNED u8; build the signed-i8 EmitC type the reinterpret
    // decode feeds off the SAME weight LMUL (no i8-typed operand carries it).
    auto weightI8VecType = weftrvv::VectorType::get(
        rewriter.getContext(), rewriter.getI8Type(), weightVecType.getLmul());
    mlir::Type resultEmitC = convertVectorTypeToEmitC(resultVecType);
    mlir::Type srcU8EmitC = convertVectorTypeToEmitC(weightVecType);
    mlir::Type srcI8EmitC = convertVectorTypeToEmitC(weightI8VecType);
    if (!resultEmitC || !srcU8EmitC || !srcI8EmitC)
      return rewriter.notifyMatchFailure(
          product, "unsigned-nibble packed-i4 x i8 product type not convertible");
    mlir::Value weight = valueMap.lookup(product.getWeight());
    mlir::Value actLow = valueMap.lookup(product.getActivationLow());
    mlir::Value actHigh = valueMap.lookup(product.getActivationHigh());
    if (!weight || !actLow || !actHigh)
      return rewriter.notifyMatchFailure(
          product, "unsigned-nibble packed-i4 x i8 product operand unmapped");
    llvm::StringRef opName = product.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = product.getWEFTEmitCLowerableSourceRole();
    mlir::FailureOr<mlir::Value> pairSum = emitUnsignedNibbleDecodeProductValue(
        rewriter, loc, weight, actLow, actHigh, bodyVL, srcI8EmitC, srcU8EmitC,
        resultEmitC, weightVecType.getLmul(),
        vectorElementWidth(resultVecType), resultVecType.getLmul(),
        vectorDType(resultVecType), opName, role);
    if (mlir::failed(pairSum))
      return mlir::failure();
    valueMap[product.getResult()] = *pairSum;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitBlockFiveBitQhSource(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::BlockFiveBitQhSourceOp qhSource,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    mlir::Value /*bodyVL*/) const {
    // Gate-only edge (mirrors block_fp16_min_product): the qh bytes are re-read by
    // the naming five-bit product op from THIS brick's qh_base + qh_byte_offset, so
    // NOTHING is materialized into the valueMap here. Validate the base mapping
    // only so the op-by-op walk does not fail on the op; the whole variant body
    // (including this op) is erased wholesale after the walk.
    mlir::Value qhBase = valueMap.lookup(qhSource.getQhBase());
    if (!qhBase)
      return rewriter.notifyMatchFailure(
          qhSource, "block_five_bit_qh_source qh_base unmapped");
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitFiveBitOffsetBinaryXI8Product(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::FiveBitOffsetBinaryXI8ProductOp product,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    mlir::Value bodyVL) const {
    mlir::MLIRContext *ctx = rewriter.getContext();
    auto resultVecType =
        llvm::dyn_cast<weftrvv::VectorType>(product.getResult().getType());
    auto weightVecType =
        llvm::dyn_cast<weftrvv::VectorType>(product.getWeight().getType());
    if (!resultVecType || !weightVecType)
      return rewriter.notifyMatchFailure(
          product, "five-bit offset-binary packed x i8 product types not vectors");
    // The weight is UNSIGNED u8; build the signed-i8 EmitC type the reinterpret
    // decode feeds off the SAME weight LMUL (no i8-typed operand carries it).
    auto weightI8VecType = weftrvv::VectorType::get(
        ctx, rewriter.getI8Type(), weightVecType.getLmul());
    mlir::Type resultEmitC = convertVectorTypeToEmitC(resultVecType);
    mlir::Type srcU8EmitC = convertVectorTypeToEmitC(weightVecType);
    mlir::Type srcI8EmitC = convertVectorTypeToEmitC(weightI8VecType);
    if (!resultEmitC || !srcU8EmitC || !srcI8EmitC)
      return rewriter.notifyMatchFailure(
          product,
          "five-bit offset-binary packed x i8 product type not convertible");
    // The u16 wide-LMUL type for the per-lane 5th-bit extraction lives at the
    // RESULT (wide) LMUL -- built as vuint16<W>_t (byte-exact to the monolith).
    llvm::StringRef wideLmul = resultVecType.getLmul();
    mlir::Type wideU16EmitC =
        emitc::OpaqueType::get(ctx, ("vuint16" + wideLmul + "_t").str());

    mlir::Value weight = valueMap.lookup(product.getWeight());
    mlir::Value actLow = valueMap.lookup(product.getActivationLow());
    mlir::Value actHigh = valueMap.lookup(product.getActivationHigh());
    if (!weight || !actLow || !actHigh)
      return rewriter.notifyMatchFailure(
          product, "five-bit offset-binary packed x i8 product operand unmapped");

    // The qh 5th bit is SOURCE-driven: re-read the two aligned 16-bit qh halves
    // from the block_five_bit_qh_source brick that DEFINES the qh_source operand,
    // at THAT brick's own qh_base + qh_byte_offset (single-block form -- the only
    // form reachable in the op-by-op walk). Mutating the brick's qh_base operand or
    // qh_byte_offset attr changes the emitted `(xb + qhOffset)` address -> the
    // emitted bytes (anti-bypass); no byte offset is baked on this product op.
    auto qhBrick =
        product.getQhSource().getDefiningOp<weftrvv::BlockFiveBitQhSourceOp>();
    if (!qhBrick)
      return rewriter.notifyMatchFailure(
          product, "five-bit product qh_source must be defined by a "
                   "weft_rvv.block_five_bit_qh_source brick");
    if (qhBrick.getBlockIndex())
      return rewriter.notifyMatchFailure(
          product, "op-by-op five-bit qh brick lowers the single-block form "
                   "(the block_index-sourced form is the full-body driver path)");
    mlir::Value qhBase = valueMap.lookup(qhBrick.getQhBase());
    if (!qhBase)
      return rewriter.notifyMatchFailure(product,
                                         "five-bit product qh_base unmapped");

    mlir::Type sizeType = getSizeType(rewriter);
    mlir::Type u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
    mlir::Type weightPtrType = qhBase.getType();
    llvm::StringRef u16ReadCallee = "(uint16_t)*(const uint16_t *)";
    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType,
                                               std::to_string(v));
    };
    // Two aligned 16-bit halves off qh_base + qh_byte_offset (LE low @ off, high @
    // off+2), byte-exact to the monolith qhRead's `(uint16_t)*(const uint16_t *)`.
    int64_t qhByteOffset =
        static_cast<int64_t>(qhBrick.getQhByteOffset().value_or(0));
    auto u16ReadAt = [&](int64_t byteOffset) -> mlir::Value {
      mlir::Value ptr = qhBase;
      if (byteOffset != 0)
        ptr = rewriter.create<emitc::AddOp>(loc, weightPtrType, qhBase,
                                            sizeLit(byteOffset));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u32Type},
                                       u16ReadCallee, mlir::ValueRange{ptr})
          .getResult(0);
    };
    mlir::Value qhLow16 = u16ReadAt(qhByteOffset);
    mlir::Value qhHigh16 = u16ReadAt(qhByteOffset + 2);
    // Elided single-strip typed body: the per-lane 5th-bit alignment offset is 0.
    mlir::Value chunkOffset = sizeLit(0);

    llvm::StringRef opName = product.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef role = product.getWEFTEmitCLowerableSourceRole();
    // Route to the ALREADY-BYTE-EXACT five-bit decode arithmetic; applyOffsetBias
    // = true is the q5_0 `-16` offset-binary bias.
    mlir::FailureOr<mlir::Value> pairSum =
        emitFiveBitOffsetBinaryDecodeProductValue(
            rewriter, loc, weight, actLow, actHigh, qhLow16, qhHigh16,
            chunkOffset, bodyVL, srcI8EmitC, srcU8EmitC, wideU16EmitC,
            resultEmitC, weightVecType.getLmul(), wideLmul,
            vectorElementWidth(resultVecType), resultVecType.getLmul(),
            vectorDType(resultVecType), opName, role, /*applyOffsetBias=*/true);
    if (mlir::failed(pairSum))
      return mlir::failure();
    valueMap[product.getResult()] = *pairSum;
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitWideningMAcc(mlir::ConversionPatternRewriter &rewriter,
                 mlir::Location loc, weftrvv::WideningMAccOp macc,
                 llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                 mlir::Value bodyVL) const {
    auto resultVecType =
        llvm::dyn_cast<weftrvv::VectorType>(macc.getResult().getType());
    auto lhsVecType =
        llvm::dyn_cast<weftrvv::VectorType>(macc.getLhs().getType());
    if (!resultVecType || !lhsVecType)
      return rewriter.notifyMatchFailure(macc,
                                         "widening macc types not vectors");
    if (macc.getKind() != "signed_widening_macc_add")
      return rewriter.notifyMatchFailure(
          macc, "only the signed widening macc-add is convertible");
    mlir::Type resultEmitC = convertVectorTypeToEmitC(resultVecType);
    if (!resultEmitC || !convertVectorTypeToEmitC(lhsVecType))
      return rewriter.notifyMatchFailure(macc,
                                         "widening macc type not convertible");
    mlir::Value lhs = valueMap.lookup(macc.getLhs());
    mlir::Value rhs = valueMap.lookup(macc.getRhs());
    mlir::Value accumulator = valueMap.lookup(macc.getAccumulator());
    if (!lhs || !rhs || !accumulator)
      return rewriter.notifyMatchFailure(macc, "widening macc operand unmapped");
    std::string callee =
        riscvIntrinsicName("vwmacc", vectorElementWidth(resultVecType),
                           resultVecType.getLmul(), vectorDType(resultVecType));
    // vwmacc destination read-modify-writes the accumulator: (acc, lhs, rhs, vl).
    mlir::Value result = emitOpaqueCall(
        rewriter, loc, resultEmitC, callee,
        mlir::ValueRange{accumulator, lhs, rhs, bodyVL},
        macc.getWEFTEmitCLowerableSourceOpName(),
        macc.getWEFTEmitCLowerableSourceRole());
    valueMap[macc.getResult()] = result;
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitWideningDotReduce(mlir::ConversionPatternRewriter &rewriter,
                      mlir::Location loc, weftrvv::WideningDotReduceOp dot,
                      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                      mlir::Value outBuffer, mlir::Value bodyVL) const {
    auto resultVecType =
        llvm::dyn_cast<weftrvv::VectorType>(dot.getResult().getType());
    auto lhsVecType =
        llvm::dyn_cast<weftrvv::VectorType>(dot.getLhs().getType());
    if (!resultVecType || !lhsVecType)
      return rewriter.notifyMatchFailure(dot, "dot reduce types not vectors");
    if (resultVecType.getLmul() != "m1")
      return rewriter.notifyMatchFailure(dot, "dot reduce result must be m1");
    if (dot.getKind() != "signed_widening_dot_reduce_add")
      return rewriter.notifyMatchFailure(
          dot, "only the signed widening dot-reduce-add is convertible");
    mlir::Type resultEmitC = convertVectorTypeToEmitC(resultVecType);
    if (!resultEmitC || !convertVectorTypeToEmitC(lhsVecType))
      return rewriter.notifyMatchFailure(dot, "dot reduce type not convertible");
    mlir::Value lhs = valueMap.lookup(dot.getLhs());
    mlir::Value rhs = valueMap.lookup(dot.getRhs());
    if (!lhs || !rhs)
      return rewriter.notifyMatchFailure(dot, "dot reduce operand unmapped");

    // Widened product (result dtype/lmul): vwmul_vv_i32m1(lhs, rhs, vl).
    std::string productCallee =
        riscvIntrinsicName("vwmul", vectorElementWidth(resultVecType),
                           resultVecType.getLmul(), vectorDType(resultVecType));
    mlir::Value product = emitOpaqueCall(
        rewriter, loc, resultEmitC, productCallee,
        mlir::ValueRange{lhs, rhs, bodyVL},
        dot.getWEFTEmitCLowerableSourceOpName(),
        dot.getWEFTEmitCLowerableSourceRole());

    // In-loop running-seed read: out[0] -> splat m1.
    mlir::Value seed = emitScalarSeedSplat(
        rewriter, loc, outBuffer, resultVecType,
        dot.getWEFTEmitCLowerableSourceOpName(),
        dot.getWEFTEmitCLowerableSourceRole());
    if (!seed)
      return rewriter.notifyMatchFailure(dot,
                                         "dot reduce running seed not "
                                         "convertible");

    // Plain horizontal reduction of the i32 product over the running seed.
    std::string reduceCallee = riscvReductionIntrinsicName(
        "vredsum", vectorElementWidth(resultVecType), resultVecType.getLmul(),
        vectorDType(resultVecType));
    mlir::Value result = emitOpaqueCall(
        rewriter, loc, resultEmitC, reduceCallee,
        mlir::ValueRange{product, seed, bodyVL},
        dot.getWEFTEmitCLowerableSourceOpName(),
        dot.getWEFTEmitCLowerableSourceRole());
    valueMap[dot.getResult()] = result;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitMaskedWideningDotReduce(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::MaskedWideningDotReduceOp dot,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap, mlir::Value outBuffer,
    mlir::Value bodyVL) const {
    auto resultVecType =
        llvm::dyn_cast<weftrvv::VectorType>(dot.getResult().getType());
    auto lhsVecType =
        llvm::dyn_cast<weftrvv::VectorType>(dot.getLhs().getType());
    if (!resultVecType || !lhsVecType)
      return rewriter.notifyMatchFailure(dot,
                                         "masked dot reduce types not vectors");
    if (resultVecType.getLmul() != "m1")
      return rewriter.notifyMatchFailure(dot,
                                         "masked dot reduce result must be m1");
    if (dot.getKind() != "signed_masked_widening_dot_reduce_add")
      return rewriter.notifyMatchFailure(
          dot, "only the signed masked widening dot-reduce-add is convertible");
    mlir::Type resultEmitC = convertVectorTypeToEmitC(resultVecType);
    if (!resultEmitC || !convertVectorTypeToEmitC(lhsVecType))
      return rewriter.notifyMatchFailure(dot,
                                         "masked dot reduce type not "
                                         "convertible");
    mlir::Value mask = valueMap.lookup(dot.getMask());
    mlir::Value lhs = valueMap.lookup(dot.getLhs());
    mlir::Value rhs = valueMap.lookup(dot.getRhs());
    if (!mask || !lhs || !rhs)
      return rewriter.notifyMatchFailure(dot,
                                         "masked dot reduce operand unmapped");

    // Zero background over the running VL (the inactive-lane neutral for add).
    std::string zeroCallee =
        riscvIntrinsicName("vmv_v_x", vectorElementWidth(resultVecType),
                           resultVecType.getLmul(), vectorDType(resultVecType));
    mlir::Value zeroVec = emitOpaqueCallBuilt(
        rewriter, loc, resultEmitC, zeroCallee,
        dot.getWEFTEmitCLowerableSourceOpName(),
        dot.getWEFTEmitCLowerableSourceRole(),
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value zeroLiteral =
              b.create<emitc::LiteralOp>(l, resultIntScalarType(rewriter), "0");
          return {zeroLiteral, bodyVL};
        });

    // Masked widened product: vwmul_vv_<rd>m1_m(mask, lhs, rhs, vl).
    std::string productCallee =
        riscvIntrinsicName("vwmul", vectorElementWidth(resultVecType),
                           resultVecType.getLmul(), vectorDType(resultVecType)) +
        "_m";
    mlir::Value maskedProduct = emitOpaqueCall(
        rewriter, loc, resultEmitC, productCallee,
        mlir::ValueRange{mask, lhs, rhs, bodyVL},
        dot.getWEFTEmitCLowerableSourceOpName(),
        dot.getWEFTEmitCLowerableSourceRole());

    // Merge the masked product over the zero background (inactive lanes -> 0).
    std::string mergeCallee =
        riscvIntrinsicName("vmerge", vectorElementWidth(resultVecType),
                           resultVecType.getLmul(), vectorDType(resultVecType));
    mlir::Value merged = emitOpaqueCall(
        rewriter, loc, resultEmitC, mergeCallee,
        mlir::ValueRange{zeroVec, maskedProduct, mask, bodyVL},
        dot.getWEFTEmitCLowerableSourceOpName(),
        dot.getWEFTEmitCLowerableSourceRole());

    // In-loop running-seed read: out[0] -> splat m1.
    mlir::Value seed = emitScalarSeedSplat(
        rewriter, loc, outBuffer, resultVecType,
        dot.getWEFTEmitCLowerableSourceOpName(),
        dot.getWEFTEmitCLowerableSourceRole());
    if (!seed)
      return rewriter.notifyMatchFailure(
          dot, "masked dot reduce running seed not convertible");

    std::string reduceCallee = riscvReductionIntrinsicName(
        "vredsum", vectorElementWidth(resultVecType), resultVecType.getLmul(),
        vectorDType(resultVecType));
    mlir::Value result = emitOpaqueCall(
        rewriter, loc, resultEmitC, reduceCallee,
        mlir::ValueRange{merged, seed, bodyVL},
        dot.getWEFTEmitCLowerableSourceOpName(),
        dot.getWEFTEmitCLowerableSourceRole());
    valueMap[dot.getResult()] = result;
    return mlir::success();
  }

mlir::Type VariantToEmitCFunc::resultIntScalarType(mlir::ConversionPatternRewriter &r) {
    return emitc::OpaqueType::get(r.getContext(), "int");
  }

mlir::LogicalResult
VariantToEmitCFunc::emitStore(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
          weftrvv::StoreOp store,
          llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
          mlir::Value inductionVar, mlir::Value storeVL,
          mlir::Value extraOffset) const {
    auto vectorType =
        llvm::dyn_cast<weftrvv::VectorType>(store.getValue().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(store, "store value not typed vector");
    mlir::Value base = valueMap.lookup(store.getBuffer());
    mlir::Value value = valueMap.lookup(store.getValue());
    if (!base || !value)
      return rewriter.notifyMatchFailure(store, "store operand unmapped");
    if (!bufferPointeeMatchesVectorElement(base, vectorType))
      return rewriter.notifyMatchFailure(
          store, "store buffer C type disagrees with stored vector element");
    if (!convertVectorTypeToEmitC(vectorType))
      return rewriter.notifyMatchFailure(store, "vector type not convertible");

    std::string callee =
        riscvIntrinsicName("vse", vectorElementWidth(vectorType),
                           vectorType.getLmul(), vectorDType(vectorType));
    // Void interleave (pointer adds built between comment and call): split the
    // mangler string at the first underscore -- split-then-rejoin identity
    // (emitVCallVoidBuilt rebuilds "__riscv_" + mnemonic + "_" + suffix).
    auto [vseMnemonic, vseSuffix] =
        llvm::StringRef(callee)
            .drop_front(llvm::StringLiteral("__riscv_").size())
            .split('_');
    emitVCallVoidBuilt(rewriter, loc, vseMnemonic, vseSuffix,
                       store.getWEFTEmitCLowerableSourceOpName(),
                       store.getWEFTEmitCLowerableSourceRole(),
                       [&](mlir::OpBuilder &b, mlir::Location l)
                           -> llvm::SmallVector<mlir::Value> {
                         mlir::Value ptr = b.create<emitc::AddOp>(
                             l, base.getType(), base, inductionVar);
                         if (extraOffset)
                           ptr = b.create<emitc::AddOp>(l, base.getType(), ptr,
                                                        extraOffset);
                         return {ptr, value, storeVL};
                       });
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitBroadcastLoad(mlir::ConversionPatternRewriter &rewriter,
                  mlir::Location loc, weftrvv::BroadcastLoadOp broadcast,
                  llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                  mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<weftrvv::VectorType>(broadcast.getBroadcast().getType());
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
    if (!bufferPointeeMatchesVectorElement(base, vectorType))
      return rewriter.notifyMatchFailure(
          broadcast,
          "broadcast buffer C type disagrees with broadcast vector element");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(broadcast,
                                         "vector type not convertible");
    std::string callee =
        riscvIntrinsicName("vmv_v_x", vectorElementWidth(vectorType),
                           vectorType.getLmul(), vectorDType(vectorType));
    // base[0]: subscript -> lvalue -> load reads the first scalar element.
    mlir::Value result = emitOpaqueCallBuilt(
        rewriter, loc, vecType, callee,
        broadcast.getWEFTEmitCLowerableSourceOpName(),
        broadcast.getWEFTEmitCLowerableSourceRole(),
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value index =
              b.create<emitc::LiteralOp>(l, b.getIndexType(), "0");
          emitc::SubscriptOp subscriptOp =
              b.create<emitc::SubscriptOp>(l, pointer, index);
          auto lvalueType =
              llvm::cast<emitc::LValueType>(subscriptOp.getResult().getType());
          mlir::Value scalar =
              b.create<emitc::LoadOp>(l, lvalueType.getValueType(),
                                      subscriptOp.getResult())
                  .getResult();
          return {scalar, bodyVL};
        });
    valueMap[broadcast.getBroadcast()] = result;
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitSplat(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
          weftrvv::SplatOp splat,
          llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
          mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<weftrvv::VectorType>(splat.getBroadcast().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(splat, "splat result not typed vector");
    mlir::Value scalar = valueMap.lookup(splat.getScalar());
    if (!scalar)
      return rewriter.notifyMatchFailure(splat, "splat scalar unmapped");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(splat, "vector type not convertible");
    // Float splat uses vfmv_v_f (float scalar); integer splat uses vmv_v_x.
    llvm::StringRef splatMnemonic =
        isFloatVector(vectorType) ? "vfmv_v_f" : "vmv_v_x";
    std::string callee =
        riscvIntrinsicName(splatMnemonic, vectorElementWidth(vectorType),
                           vectorType.getLmul(), vectorDType(vectorType));
    mlir::Value result = emitOpaqueCall(
        rewriter, loc, vecType, callee, mlir::ValueRange{scalar, bodyVL},
        splat.getWEFTEmitCLowerableSourceOpName(),
        splat.getWEFTEmitCLowerableSourceRole());
    valueMap[splat.getBroadcast()] = result;
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitCompare(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
            weftrvv::CompareOp compare,
            llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
            mlir::Value bodyVL) const {
    auto maskType =
        llvm::dyn_cast<weftrvv::MaskType>(compare.getMask().getType());
    if (!maskType)
      return rewriter.notifyMatchFailure(compare, "compare result not typed mask");
    auto operandVecType =
        llvm::dyn_cast<weftrvv::VectorType>(compare.getLhs().getType());
    if (!operandVecType)
      return rewriter.notifyMatchFailure(compare,
                                         "compare operand not typed vector");
    mlir::Value lhs = valueMap.lookup(compare.getLhs());
    mlir::Value rhs = valueMap.lookup(compare.getRhs());
    if (!lhs || !rhs)
      return rewriter.notifyMatchFailure(compare, "compare operand unmapped");
    std::optional<llvm::StringRef> mnemonic =
        compareMnemonic(compare.getKind(), isFloatVector(operandVecType));
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
    mlir::Value result = emitOpaqueCall(
        rewriter, loc, maskEmitCType, callee,
        mlir::ValueRange{lhs, rhs, bodyVL},
        compare.getWEFTEmitCLowerableSourceOpName(),
        compare.getWEFTEmitCLowerableSourceRole());
    valueMap[compare.getMask()] = result;
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitSelect(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
           weftrvv::SelectOp select,
           llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
           mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<weftrvv::VectorType>(select.getSelected().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(select,
                                         "select result not typed vector");
    mlir::Value mask = valueMap.lookup(select.getMask());
    mlir::Value trueValue = valueMap.lookup(select.getTrueValue());
    mlir::Value falseValue = valueMap.lookup(select.getFalseValue());
    if (!mask || !trueValue || !falseValue)
      return rewriter.notifyMatchFailure(select, "select operand unmapped");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(select, "vector type not convertible");
    std::string callee =
        riscvIntrinsicName("vmerge", vectorElementWidth(vectorType),
                           vectorType.getLmul(), vectorDType(vectorType));
    mlir::Value result = emitOpaqueCall(
        rewriter, loc, vecType, callee,
        mlir::ValueRange{falseValue, trueValue, mask, bodyVL},
        select.getWEFTEmitCLowerableSourceOpName(),
        select.getWEFTEmitCLowerableSourceRole());
    valueMap[select.getSelected()] = result;
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitMaskAnd(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
            weftrvv::MaskAndOp maskAnd,
            llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
            mlir::Value bodyVL) const {
    auto maskType =
        llvm::dyn_cast<weftrvv::MaskType>(maskAnd.getMask().getType());
    if (!maskType)
      return rewriter.notifyMatchFailure(maskAnd, "mask_and result not mask");
    mlir::Value lhs = valueMap.lookup(maskAnd.getLhs());
    mlir::Value rhs = valueMap.lookup(maskAnd.getRhs());
    if (!lhs || !rhs)
      return rewriter.notifyMatchFailure(maskAnd, "mask_and operand unmapped");
    std::optional<llvm::StringRef> mnemonic = maskAndMnemonic(maskAnd.getKind());
    if (!mnemonic)
      return rewriter.notifyMatchFailure(maskAnd, "unsupported mask_and kind");
    mlir::Type maskEmitCType = getTypeConverter()->convertType(maskType);
    if (!maskEmitCType ||
        maskEmitCType.getDialect().getNamespace() !=
            emitc::EmitCDialect::getDialectNamespace())
      return rewriter.notifyMatchFailure(maskAnd, "mask type not convertible");
    unsigned sew = 0;
    if (maskType.getElementType().isSignlessInteger(32))
      sew = 32;
    else if (maskType.getElementType().isSignlessInteger(64))
      sew = 64;
    unsigned maskBits = maskWidthForConfig(sew, maskType.getLmul());
    if (maskBits == 0)
      return rewriter.notifyMatchFailure(maskAnd, "unsupported mask config");
    std::string callee = riscvMaskComposeIntrinsicName(*mnemonic, maskBits);
    mlir::Value result = emitOpaqueCall(
        rewriter, loc, maskEmitCType, callee,
        mlir::ValueRange{lhs, rhs, bodyVL},
        maskAnd.getWEFTEmitCLowerableSourceOpName(),
        maskAnd.getWEFTEmitCLowerableSourceRole());
    valueMap[maskAnd.getMask()] = result;
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitDequantize(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
               weftrvv::DequantizeOp dequantize,
               llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
               mlir::Value bodyVL) const {
    // Scope guard: only the dequant-CLAMP epilogue (a compare-select body that
    // begins with one dequantize and then clamps via compare/select) is in this
    // family. The STANDALONE dequantize (load -> dequantize -> store, no select)
    // is the Dequantization owner's consumer and is Gearbox-unrolled by a
    // separate schedule pass the simple for-loop here does NOT reproduce. Refuse
    // to convert a dequantize whose enclosing body carries no weft_rvv.select so
    // the standalone body fails to fully legalize and falls back unchanged.
    bool bodyHasSelect = false;
    if (mlir::Block *block = dequantize->getBlock()) {
      for (mlir::Operation &sibling : *block)
        if (llvm::isa<weftrvv::SelectOp>(sibling)) {
          bodyHasSelect = true;
          break;
        }
    }
    if (!bodyHasSelect)
      return rewriter.notifyMatchFailure(
          dequantize, "standalone dequantize (no select) is out of scope");
    mlir::Value source = valueMap.lookup(dequantize.getSource());
    mlir::Value scale = valueMap.lookup(dequantize.getScale());
    if (!source || !scale)
      return rewriter.notifyMatchFailure(dequantize,
                                         "dequantize operand unmapped");
    return emitDequantizeChain(rewriter, loc, dequantize, source, scale,
                               valueMap, bodyVL);
  }

mlir::LogicalResult
VariantToEmitCFunc::emitDequantizeChain(mlir::ConversionPatternRewriter &rewriter,
                    mlir::Location loc, weftrvv::DequantizeOp dequantize,
                    mlir::Value source, mlir::Value scale,
                    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                    mlir::Value bodyVL) const {
    auto resultVecType =
        llvm::dyn_cast<weftrvv::VectorType>(dequantize.getResult().getType());
    if (!resultVecType || !isFloatVector(resultVecType))
      return rewriter.notifyMatchFailure(dequantize,
                                         "dequantize result not f32 vector");
    mlir::Type vecType = convertVectorTypeToEmitC(resultVecType);
    if (!vecType)
      return rewriter.notifyMatchFailure(dequantize,
                                         "vector type not convertible");
    unsigned sew = vectorElementWidth(resultVecType);
    llvm::StringRef lmul = resultVecType.getLmul();
    llvm::StringRef dtype = vectorDType(resultVecType);

    // converted = vfcvt_f_x_v(source_i32, vl) -- int->float reinterpret-convert.
    std::string convertCallee = riscvIntrinsicName("vfcvt_f_x_v", sew, lmul,
                                                   dtype);
    mlir::Value converted = emitOpaqueCall(
        rewriter, loc, vecType, convertCallee,
        mlir::ValueRange{source, bodyVL},
        dequantize.getWEFTEmitCLowerableSourceOpName(),
        dequantize.getWEFTEmitCLowerableSourceRole());

    // result = vfmul_vf(converted, scale, vl) -- runtime f32 scale multiply.
    std::string scaleCallee = riscvIntrinsicName("vfmul_vf", sew, lmul, dtype);
    mlir::Value result = emitOpaqueCall(
        rewriter, loc, vecType, scaleCallee,
        mlir::ValueRange{converted, scale, bodyVL},
        dequantize.getWEFTEmitCLowerableSourceOpName(),
        dequantize.getWEFTEmitCLowerableSourceRole());
    valueMap[dequantize.getResult()] = result;
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitWideningConvert(mlir::ConversionPatternRewriter &rewriter,
                    mlir::Location loc, weftrvv::WideningConvertOp convert,
                    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                    mlir::Value bodyVL) const {
    llvm::StringRef kind = convert.getKind();
    if (kind != "sign_extend_widen_vf2" && kind != "widen_i32_to_i64")
      return rewriter.notifyMatchFailure(
          convert, "unsupported widening_convert kind");

    auto sourceVecType =
        llvm::dyn_cast<weftrvv::VectorType>(convert.getSource().getType());
    auto resultVecType =
        llvm::dyn_cast<weftrvv::VectorType>(convert.getResult().getType());
    if (!sourceVecType || !resultVecType)
      return rewriter.notifyMatchFailure(
          convert, "widening_convert source/result not typed vector");

    // Bounded (source,result) grid the signed vwcvt_x_x_v oracle covers. Refuse
    // any other pairing (including unsigned, which would need vwcvtu) so a
    // malformed/out-of-grid body falls back instead of mislowering.
    auto isSignedIntVec = [](weftrvv::VectorType vec, unsigned sew,
                             llvm::StringRef lmul) {
      return vec.getElementType().isSignlessInteger(sew) &&
             vec.getLmul() == lmul;
    };
    bool gridOk = false;
    if (kind == "sign_extend_widen_vf2")
      gridOk = isSignedIntVec(sourceVecType, 16, "mf2") &&
               isSignedIntVec(resultVecType, 32, "m1");
    else // widen_i32_to_i64
      gridOk = isSignedIntVec(sourceVecType, 32, "m1") &&
               isSignedIntVec(resultVecType, 64, "m2");
    if (!gridOk)
      return rewriter.notifyMatchFailure(
          convert, "widening_convert source/result outside the bounded signed "
                   "widening grid");

    mlir::Value source = valueMap.lookup(convert.getSource());
    if (!source)
      return rewriter.notifyMatchFailure(convert,
                                         "widening_convert source unmapped");
    mlir::Type vecType = convertVectorTypeToEmitC(resultVecType);
    if (!vecType)
      return rewriter.notifyMatchFailure(
          convert, "widening_convert result type not convertible");

    // The callee suffix is the WIDENED RESULT type (i32m1 / i64m2).
    std::string callee = riscvIntrinsicName(
        "vwcvt_x_x_v", vectorElementWidth(resultVecType),
        resultVecType.getLmul(), vectorDType(resultVecType));
    mlir::Value result = emitOpaqueCall(
        rewriter, loc, vecType, callee, mlir::ValueRange{source, bodyVL},
        convert.getWEFTEmitCLowerableSourceOpName(),
        convert.getWEFTEmitCLowerableSourceRole());
    valueMap[convert.getResult()] = result;
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitMaskedBinary(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                 weftrvv::MaskedBinaryOp masked,
                 llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                 mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<weftrvv::VectorType>(masked.getResult().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(masked,
                                         "masked result not typed vector");
    mlir::Value passthrough = valueMap.lookup(masked.getPassthrough());
    mlir::Value lhs = valueMap.lookup(masked.getLhs());
    mlir::Value rhs = valueMap.lookup(masked.getRhs());
    mlir::Value mask = valueMap.lookup(masked.getMask());
    if (!passthrough || !lhs || !rhs || !mask)
      return rewriter.notifyMatchFailure(masked, "masked operand unmapped");
    std::optional<llvm::StringRef> mnemonic =
        binaryMnemonic(masked.getKind(), isFloatVector(vectorType));
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
    mlir::Value active = emitOpaqueCall(
        rewriter, loc, vecType, arithCallee,
        mlir::ValueRange{lhs, rhs, bodyVL},
        masked.getWEFTEmitCLowerableSourceOpName(),
        masked.getWEFTEmitCLowerableSourceRole());

    // result = vmerge(passthrough, active, mask, vl) -- inactive lanes keep the
    // passthrough vector.
    std::string mergeCallee = riscvIntrinsicName("vmerge", sew, lmul, dtype);
    mlir::Value result = emitOpaqueCall(
        rewriter, loc, vecType, mergeCallee,
        mlir::ValueRange{passthrough, active, mask, bodyVL},
        masked.getWEFTEmitCLowerableSourceOpName(),
        masked.getWEFTEmitCLowerableSourceRole());
    valueMap[masked.getResult()] = result;
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitMAcc(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
         weftrvv::MAccOp macc,
         llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
         mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<weftrvv::VectorType>(macc.getResult().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(macc, "macc result not typed vector");
    // Layout/kind contract guard (mirrors MAccOp::verify + the plain-macc
    // route-family plan): only the bounded add / separate-vector-accumulator /
    // output-store slice is convertible. A body with another kind/layout falls
    // back unchanged rather than being mislowered as a plain fused macc.
    if (macc.getKind() != "add")
      return rewriter.notifyMatchFailure(macc, "unsupported macc kind");
    std::optional<llvm::StringRef> accumulatorLayout =
        macc.getAccumulatorLayout();
    std::optional<llvm::StringRef> resultLayout = macc.getResultLayout();
    if (!accumulatorLayout ||
        *accumulatorLayout != "separate-i32-vector-accumulator-input" ||
        !resultLayout ||
        *resultLayout != "store-multiply-accumulate-result-to-output-buffer")
      return rewriter.notifyMatchFailure(
          macc, "macc accumulator/result layout outside the convertible slice");
    // The fused vmacc intrinsic is SEW32-only in the legacy derivation; an i64
    // (or any non-i32) macc has no __riscv_vmacc_vv_i64* form and must fall back
    // unchanged instead of emitting a non-existent intrinsic.
    if (!vectorType.getElementType().isSignlessInteger(32))
      return rewriter.notifyMatchFailure(
          macc, "macc only lowers the SEW32 fused vmacc slice");

    // Operand-source structural guards (mirror the legacy plain/scalar-broadcast
    // MAcc route-family slice). The bounded slice requires:
    //   - lhs and accumulator are explicit weft_rvv.load results,
    //   - rhs is EITHER an explicit weft_rvv.load (plain macc) OR a
    //     weft_rvv.splat (scalar-broadcast macc),
    //   - NO weft_rvv.broadcast_load feeds the macc (broadcast/splat-load macc is
    //     out of the bounded slice -- legacy: "broadcast/splat macc is not in
    //     this bounded slice"),
    //   - if a weft_rvv.splat exists in the body the macc MUST consume it as rhs
    //     (a body that splats but bypasses it is the malformed scalar-broadcast
    //     composition the legacy validator rejects).
    // A body outside this shape is NOT lowered here; notifyMatchFailure rolls the
    // conversion back so the legacy validator still sees (and rejects) it.
    mlir::Operation *lhsDef = macc.getLhs().getDefiningOp();
    mlir::Operation *rhsDef = macc.getRhs().getDefiningOp();
    mlir::Operation *accDef = macc.getAccumulator().getDefiningOp();
    if (!llvm::isa_and_present<weftrvv::LoadOp>(lhsDef) ||
        !llvm::isa_and_present<weftrvv::LoadOp>(accDef))
      return rewriter.notifyMatchFailure(
          macc, "macc lhs/accumulator must be explicit vector loads");
    bool rhsIsLoad = llvm::isa_and_present<weftrvv::LoadOp>(rhsDef);
    bool rhsIsSplat = llvm::isa_and_present<weftrvv::SplatOp>(rhsDef);
    if (!rhsIsLoad && !rhsIsSplat)
      return rewriter.notifyMatchFailure(
          macc, "macc rhs must be an explicit vector load or scalar splat "
                "(broadcast/splat-load macc is out of the bounded slice)");
    // If the enclosing body carries a weft_rvv.splat, the macc must consume it
    // as rhs (scalar-broadcast composition contract). A splatting body whose macc
    // bypasses the splat is the malformed scalar-broadcast macc.
    if (mlir::Block *block = macc->getBlock())
      for (mlir::Operation &sibling : *block)
        if (llvm::isa<weftrvv::SplatOp>(sibling) && !rhsIsSplat)
          return rewriter.notifyMatchFailure(
              macc, "scalar-broadcast macc body must consume the splat result "
                    "as rhs");
    // Accumulator ABI-role binding guard: the accumulator load must read the
    // accumulator-input-buffer ABI value, NOT the output buffer (legacy:
    // "accumulator load to bind accumulator-input-buffer, not output buffer").
    auto accLoad = llvm::cast<weftrvv::LoadOp>(accDef);
    auto accBufferAbi =
        accLoad.getBuffer().getDefiningOp<weftrvv::RuntimeABIValueOp>();
    if (!accBufferAbi ||
        accBufferAbi.getRole() != "accumulator-input-buffer")
      return rewriter.notifyMatchFailure(
          macc, "macc accumulator load must bind the accumulator-input-buffer "
                "ABI role");

    mlir::Value lhs = valueMap.lookup(macc.getLhs());
    mlir::Value rhs = valueMap.lookup(macc.getRhs());
    mlir::Value accumulator = valueMap.lookup(macc.getAccumulator());
    if (!lhs || !rhs || !accumulator)
      return rewriter.notifyMatchFailure(macc, "macc operand unmapped");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(macc, "vector type not convertible");
    std::string callee =
        riscvMAccIntrinsicName(vectorElementWidth(vectorType),
                               vectorType.getLmul(), vectorDType(vectorType));
    mlir::Value result = emitOpaqueCall(
        rewriter, loc, vecType, callee,
        mlir::ValueRange{accumulator, lhs, rhs, bodyVL},
        macc.getWEFTEmitCLowerableSourceOpName(),
        macc.getWEFTEmitCLowerableSourceRole());
    valueMap[macc.getResult()] = result;
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitMaskedMAcc(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
               weftrvv::MaskedMAccOp masked,
               llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
               mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<weftrvv::VectorType>(masked.getResult().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(masked,
                                         "masked_macc result not typed vector");
    if (masked.getKind() != "add")
      return rewriter.notifyMatchFailure(masked, "unsupported masked_macc kind");
    if (masked.getAccumulatorLayout() !=
            "separate-i32-vector-accumulator-input" ||
        masked.getResultLayout() !=
            "store-multiply-accumulate-result-to-output-buffer")
      return rewriter.notifyMatchFailure(
          masked,
          "masked_macc accumulator/result layout outside the convertible slice");
    // SEW32-only fused vmacc (see emitMAcc); the masked rung shares the same
    // vmacc derivation.
    if (!vectorType.getElementType().isSignlessInteger(32))
      return rewriter.notifyMatchFailure(
          masked, "masked_macc only lowers the SEW32 fused vmacc slice");
    auto maskType =
        llvm::dyn_cast<weftrvv::MaskType>(masked.getMask().getType());
    if (!maskType)
      return rewriter.notifyMatchFailure(masked,
                                         "masked_macc mask not typed mask");
    mlir::Value mask = valueMap.lookup(masked.getMask());
    mlir::Value lhs = valueMap.lookup(masked.getLhs());
    mlir::Value rhs = valueMap.lookup(masked.getRhs());
    mlir::Value accumulator = valueMap.lookup(masked.getAccumulator());
    if (!mask || !lhs || !rhs || !accumulator)
      return rewriter.notifyMatchFailure(masked, "masked_macc operand unmapped");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(masked, "vector type not convertible");
    unsigned sew = vectorElementWidth(vectorType);
    llvm::StringRef lmul = vectorType.getLmul();
    llvm::StringRef dtype = vectorDType(vectorType);

    // active = vmacc_vv(accumulator, lhs, rhs, vl) -- fused multiply-accumulate
    // on every lane.
    std::string maccCallee = riscvMAccIntrinsicName(sew, lmul, dtype);
    mlir::Value active = emitOpaqueCall(
        rewriter, loc, vecType, maccCallee,
        mlir::ValueRange{accumulator, lhs, rhs, bodyVL},
        masked.getWEFTEmitCLowerableSourceOpName(),
        masked.getWEFTEmitCLowerableSourceRole());

    // result = vmerge(accumulator, active, mask, vl) -- inactive lanes keep the
    // accumulator passthrough vector.
    std::string mergeCallee = riscvIntrinsicName("vmerge", sew, lmul, dtype);
    mlir::Value result = emitOpaqueCall(
        rewriter, loc, vecType, mergeCallee,
        mlir::ValueRange{accumulator, active, mask, bodyVL},
        masked.getWEFTEmitCLowerableSourceOpName(),
        masked.getWEFTEmitCLowerableSourceRole());
    valueMap[masked.getResult()] = result;
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitStridedLoad(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                weftrvv::StridedLoadOp load,
                llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                mlir::Value inductionVar, mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<weftrvv::VectorType>(load.getLoaded().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(load, "strided load result not vector");
    mlir::Value base = valueMap.lookup(load.getBuffer());
    mlir::Value stride = valueMap.lookup(load.getStride());
    if (!base || !stride)
      return rewriter.notifyMatchFailure(load, "strided load operand unmapped");
    if (!bufferPointeeMatchesVectorElement(base, vectorType))
      return rewriter.notifyMatchFailure(
          load, "strided load buffer C type disagrees with loaded element");
    // Byte-stride contract guard. The base-memory strided family (its loaded
    // vector flows through a weft_rvv.move into a plain store) is driven by a
    // runtime BYTE stride; a body in that shape whose stride ABI value is NOT a
    // byte-stride role is malformed (the legacy validator rejects "source
    // byte-strided load requires source-byte-stride runtime ABI value"). Refuse
    // it so the malformed body falls back rather than being mislowered as an
    // element-strided load.
    if (loadedFeedsMove(load) && !isByteStride(load.getStride()))
      return rewriter.notifyMatchFailure(
          load, "base-memory strided load requires a byte-stride ABI role");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(load, "vector type not convertible");
    std::string callee =
        riscvIntrinsicName("vlse", vectorElementWidth(vectorType),
                           vectorType.getLmul(), vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(load.getWEFTEmitCLowerableSourceOpName(),
                         load.getWEFTEmitCLowerableSourceRole(), callee));
    // Two strided rungs share weft_rvv.strided_load: the base-memory family
    // passes a runtime BYTE stride (uint8_t-cast addressing, stride AS-IS), the
    // elementwise family passes an ELEMENT stride (scaled-element pointer +
    // ptrdiff_t byte stride). Pick the addressing from the typed ABI role.
    mlir::Value ptr;
    mlir::Value byteStride;
    if (isByteStride(load.getStride())) {
      ptr = emitByteStridedPointer(rewriter, loc, base, inductionVar, stride);
      if (!ptr)
        return rewriter.notifyMatchFailure(
            load, "strided load base must be a pointer-typed ABI param");
      byteStride = stride;
    } else {
      ptr = emitScaledPointer(rewriter, loc, base, inductionVar, stride);
      byteStride = emitByteStride(rewriter, loc, stride, vectorType);
    }
    mlir::Value loaded =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{vecType}, callee,
                                         mlir::ValueRange{ptr, byteStride,
                                                          bodyVL})
            .getResult(0);
    valueMap[load.getLoaded()] = loaded;
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitStridedStore(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                 weftrvv::StridedStoreOp store,
                 llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                 mlir::Value inductionVar, mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<weftrvv::VectorType>(store.getValue().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(store, "strided store value not vector");
    mlir::Value base = valueMap.lookup(store.getBuffer());
    mlir::Value value = valueMap.lookup(store.getValue());
    mlir::Value stride = valueMap.lookup(store.getStride());
    if (!base || !value || !stride)
      return rewriter.notifyMatchFailure(store, "strided store operand unmapped");
    if (!bufferPointeeMatchesVectorElement(base, vectorType))
      return rewriter.notifyMatchFailure(
          store, "strided store buffer C type disagrees with stored element");
    // Byte-stride contract guard (see emitStridedLoad): the base-memory strided
    // store family (its stored value comes from a weft_rvv.move of a unit-stride
    // load) requires a byte-stride ABI role. Refuse a non-byte-stride role in
    // that shape so the malformed body falls back instead of being mislowered.
    if (storedValueFromMove(store) && !isByteStride(store.getStride()))
      return rewriter.notifyMatchFailure(
          store, "base-memory strided store requires a byte-stride ABI role");
    if (!convertVectorTypeToEmitC(vectorType))
      return rewriter.notifyMatchFailure(store, "vector type not convertible");
    std::string callee =
        riscvIntrinsicName("vsse", vectorElementWidth(vectorType),
                           vectorType.getLmul(), vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(store.getWEFTEmitCLowerableSourceOpName(),
                         store.getWEFTEmitCLowerableSourceRole(), callee));
    // See emitStridedLoad: select the byte-stride vs element-stride addressing
    // from the stride ABI role.
    mlir::Value ptr;
    mlir::Value byteStride;
    if (isByteStride(store.getStride())) {
      ptr = emitByteStridedPointer(rewriter, loc, base, inductionVar, stride);
      if (!ptr)
        return rewriter.notifyMatchFailure(
            store, "strided store base must be a pointer-typed ABI param");
      byteStride = stride;
    } else {
      ptr = emitScaledPointer(rewriter, loc, base, inductionVar, stride);
      byteStride = emitByteStride(rewriter, loc, stride, vectorType);
    }
    rewriter.create<emitc::CallOpaqueOp>(
        loc, mlir::TypeRange{}, callee,
        mlir::ValueRange{ptr, byteStride, value, bodyVL});
    return mlir::success();
  }

bool VariantToEmitCFunc::resolveSegment2Facts(mlir::ConversionPatternRewriter &rewriter,
                          weftrvv::VectorType fieldType,
                          Segment2Facts &out) const {
    // Only the signed-integer field grid is in scope for the bounded segment2
    // slice (the tuple type spelling is vint<sew>m<lmul>x2_t). A float field
    // would need a different tuple/intrinsic family.
    if (isFloatVector(fieldType))
      return false;
    out.sew = vectorElementWidth(fieldType);
    out.lmul = fieldType.getLmul();
    out.dtype = vectorDType(fieldType);
    if (out.dtype.empty() || out.sew == 0)
      return false;
    // The per-field vector type must lower through the bounded converter grid;
    // reject otherwise so a non-beachhead (sew, lmul) falls back.
    out.fieldVecType = convertVectorTypeToEmitC(fieldType);
    if (!out.fieldVecType)
      return false;
    out.tupleType = emitc::OpaqueType::get(
        rewriter.getContext(), riscvSegment2TupleCType(out.sew, out.lmul));
    return true;
  }

mlir::Value VariantToEmitCFunc::emitSegment2InterleavedPointer(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value base, mlir::Value inductionVar) const {
    mlir::Value two = rewriter.create<emitc::LiteralOp>(
        loc, inductionVar.getType(), "2");
    mlir::Value off = rewriter.create<emitc::MulOp>(
        loc, inductionVar.getType(), inductionVar, two);
    return rewriter.create<emitc::AddOp>(loc, base.getType(), base, off);
  }

mlir::LogicalResult VariantToEmitCFunc::emitSegment2Load(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::Segment2LoadOp segLoad,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    llvm::DenseMap<mlir::Value, std::pair<mlir::Value, unsigned>>
        &segmentFieldMap,
    mlir::Value inductionVar, mlir::Value bodyVL) const {
    if (segLoad.getSegmentCount() != 2)
      return rewriter.notifyMatchFailure(segLoad,
                                         "only segment_count = 2 is in scope");
    if (segLoad.getSourceMemoryForm() != "segment2-interleaved-unit-stride-load")
      return rewriter.notifyMatchFailure(
          segLoad, "segment2_load source memory form outside the slice");
    auto field0Type =
        llvm::dyn_cast<weftrvv::VectorType>(segLoad.getField0().getType());
    auto field1Type =
        llvm::dyn_cast<weftrvv::VectorType>(segLoad.getField1().getType());
    if (!field0Type || !field1Type || field0Type != field1Type)
      return rewriter.notifyMatchFailure(
          segLoad, "segment2_load fields must be matching typed vectors");
    Segment2Facts facts;
    if (!resolveSegment2Facts(rewriter, field0Type, facts))
      return rewriter.notifyMatchFailure(
          segLoad, "segment2_load field vector type not convertible");
    mlir::Value base = valueMap.lookup(segLoad.getSource());
    if (!base)
      return rewriter.notifyMatchFailure(segLoad,
                                         "segment2_load source not an ABI param");
    if (!bufferPointeeMatchesVectorElement(base, field0Type))
      return rewriter.notifyMatchFailure(
          segLoad, "segment2_load source C type disagrees with field element");

    std::string callee =
        riscvSegment2LoadIntrinsicName(facts.sew, facts.lmul, facts.dtype);
    mlir::Value tuple = emitOpaqueCallBuilt(
        rewriter, loc, facts.tupleType, callee,
        segLoad.getWEFTEmitCLowerableSourceOpName(),
        segLoad.getWEFTEmitCLowerableSourceRole(),
        [&](mlir::OpBuilder &,
            mlir::Location) -> llvm::SmallVector<mlir::Value> {
          mlir::Value ptr =
              emitSegment2InterleavedPointer(rewriter, loc, base, inductionVar);
          return {ptr, bodyVL};
        });
    // The two move ops sourced from these field results emit the vget extracts.
    segmentFieldMap[segLoad.getField0()] = {tuple, 0};
    segmentFieldMap[segLoad.getField1()] = {tuple, 1};
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitSegment2Store(mlir::ConversionPatternRewriter &rewriter,
                  mlir::Location loc, weftrvv::Segment2StoreOp segStore,
                  llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                  mlir::Value inductionVar, mlir::Value bodyVL) const {
    if (segStore.getSegmentCount() != 2)
      return rewriter.notifyMatchFailure(segStore,
                                         "only segment_count = 2 is in scope");
    if (segStore.getDestinationMemoryForm() !=
        "segment2-interleaved-unit-stride-store")
      return rewriter.notifyMatchFailure(
          segStore, "segment2_store destination memory form outside the slice");
    auto field0Type =
        llvm::dyn_cast<weftrvv::VectorType>(segStore.getField0().getType());
    auto field1Type =
        llvm::dyn_cast<weftrvv::VectorType>(segStore.getField1().getType());
    if (!field0Type || !field1Type || field0Type != field1Type)
      return rewriter.notifyMatchFailure(
          segStore, "segment2_store fields must be matching typed vectors");
    Segment2Facts facts;
    if (!resolveSegment2Facts(rewriter, field0Type, facts))
      return rewriter.notifyMatchFailure(
          segStore, "segment2_store field vector type not convertible");
    mlir::Value base = valueMap.lookup(segStore.getDestination());
    mlir::Value field0 = valueMap.lookup(segStore.getField0());
    mlir::Value field1 = valueMap.lookup(segStore.getField1());
    if (!base || !field0 || !field1)
      return rewriter.notifyMatchFailure(segStore,
                                         "segment2_store operand unmapped");
    if (!bufferPointeeMatchesVectorElement(base, field0Type))
      return rewriter.notifyMatchFailure(
          segStore, "segment2_store dst C type disagrees with field element");
    // Field-binding guard: the interleave field0/field1 operands must bind the
    // field0/field1 input loads (structural authority via the load buffer role).
    // A body that swaps them (segment2_store %dst, %field1, %field0) is the
    // operand-binding negative the legacy provider rejects -- fall back so it is
    // not silently mislowered.
    if (!fieldVectorBindsLoadRole(segStore.getField0(),
                                  segStore.getField0Role()) ||
        !fieldVectorBindsLoadRole(segStore.getField1(),
                                  segStore.getField1Role()))
      return rewriter.notifyMatchFailure(
          segStore, "segment2_store must consume matching field0/field1 load "
                    "results bound to the field0/field1 input roles");

    // Step 1: pack the two fields into one tuple.
    std::string createCallee =
        riscvSegment2TupleCreateIntrinsicName(facts.dtype, facts.lmul);
    mlir::Value tuple = emitOpaqueCall(
        rewriter, loc, facts.tupleType, createCallee,
        mlir::ValueRange{field0, field1},
        segStore.getWEFTEmitCLowerableSourceOpName(),
        segStore.getWEFTEmitCLowerableSourceRole());

    // Step 2: store the tuple to the interleaved destination.
    std::string storeCallee =
        riscvSegment2StoreIntrinsicName(facts.sew, facts.lmul, facts.dtype);
    // Void interleave (interleaved pointer built between comment and call):
    // split the mangler string at the first underscore -- split/rejoin identity.
    auto [storeMnemonic, storeSuffix] =
        llvm::StringRef(storeCallee)
            .drop_front(llvm::StringLiteral("__riscv_").size())
            .split('_');
    emitVCallVoidBuilt(rewriter, loc, storeMnemonic, storeSuffix,
                       segStore.getWEFTEmitCLowerableSourceOpName(),
                       segStore.getWEFTEmitCLowerableSourceRole(),
                       [&](mlir::OpBuilder &,
                           mlir::Location) -> llvm::SmallVector<mlir::Value> {
                         mlir::Value ptr = emitSegment2InterleavedPointer(
                             rewriter, loc, base, inductionVar);
                         return {ptr, tuple, bodyVL};
                       });
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitMaskedSegment2Load(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::MaskedSegment2LoadOp segLoad,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    mlir::Value inductionVar, mlir::Value bodyVL) const {
    if (segLoad.getSegmentCount() != 2)
      return rewriter.notifyMatchFailure(segLoad,
                                         "only segment_count = 2 is in scope");
    if (segLoad.getSourceMemoryForm() !=
            "segment2-interleaved-unit-stride-load" ||
        segLoad.getInactiveLanePolicy() !=
            "preserve-passthrough-on-false-lanes")
      return rewriter.notifyMatchFailure(
          segLoad, "masked_segment2_load form/policy outside the slice");
    if (!isMaskFromMaskLoadOrCompare(segLoad.getMask()))
      return rewriter.notifyMatchFailure(
          segLoad, "masked_segment2_load mask must come from a compare in the "
                   "same scope");
    auto field0Type =
        llvm::dyn_cast<weftrvv::VectorType>(segLoad.getField0().getType());
    auto field1Type =
        llvm::dyn_cast<weftrvv::VectorType>(segLoad.getField1().getType());
    if (!field0Type || !field1Type || field0Type != field1Type)
      return rewriter.notifyMatchFailure(
          segLoad, "masked_segment2_load fields must be matching typed vectors");
    Segment2Facts facts;
    if (!resolveSegment2Facts(rewriter, field0Type, facts))
      return rewriter.notifyMatchFailure(
          segLoad, "masked_segment2_load field vector type not convertible");
    mlir::Value base = valueMap.lookup(segLoad.getSource());
    mlir::Value mask = valueMap.lookup(segLoad.getMask());
    mlir::Value pass0 = valueMap.lookup(segLoad.getPassthrough0());
    mlir::Value pass1 = valueMap.lookup(segLoad.getPassthrough1());
    if (!base || !mask || !pass0 || !pass1)
      return rewriter.notifyMatchFailure(segLoad,
                                         "masked_segment2_load operand unmapped");
    if (!bufferPointeeMatchesVectorElement(base, field0Type))
      return rewriter.notifyMatchFailure(
          segLoad, "masked_segment2_load src C type disagrees with field "
                   "element");

    // Step 1: pack the two passthroughs into the passthrough tuple.
    std::string createCallee =
        riscvSegment2TupleCreateIntrinsicName(facts.dtype, facts.lmul);
    mlir::Value passTuple = emitOpaqueCall(
        rewriter, loc, facts.tupleType, createCallee,
        mlir::ValueRange{pass0, pass1},
        segLoad.getWEFTEmitCLowerableSourceOpName(),
        segLoad.getWEFTEmitCLowerableSourceRole());

    // Step 2: masked tuple load from the interleaved source.
    std::string loadCallee =
        riscvMaskedSegment2LoadIntrinsicName(facts.sew, facts.lmul, facts.dtype);
    mlir::Value tuple = emitOpaqueCallBuilt(
        rewriter, loc, facts.tupleType, loadCallee,
        segLoad.getWEFTEmitCLowerableSourceOpName(),
        segLoad.getWEFTEmitCLowerableSourceRole(),
        [&](mlir::OpBuilder &,
            mlir::Location) -> llvm::SmallVector<mlir::Value> {
          mlir::Value ptr =
              emitSegment2InterleavedPointer(rewriter, loc, base, inductionVar);
          return {mask, passTuple, ptr, bodyVL};
        });

    // Steps 3 & 4: extract the two field vectors.
    std::string getCallee =
        riscvSegment2FieldExtractIntrinsicName(facts.dtype, facts.lmul);
    mlir::Value field0 = emitSegment2FieldExtract(
        rewriter, loc, segLoad.getWEFTEmitCLowerableSourceOpName(),
        segLoad.getWEFTEmitCLowerableSourceRole(), getCallee, tuple,
        facts.fieldVecType, 0);
    mlir::Value field1 = emitSegment2FieldExtract(
        rewriter, loc, segLoad.getWEFTEmitCLowerableSourceOpName(),
        segLoad.getWEFTEmitCLowerableSourceRole(), getCallee, tuple,
        facts.fieldVecType, 1);
    valueMap[segLoad.getField0()] = field0;
    valueMap[segLoad.getField1()] = field1;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitMaskedSegment2Store(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::MaskedSegment2StoreOp segStore,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    mlir::Value inductionVar, mlir::Value bodyVL) const {
    if (segStore.getSegmentCount() != 2)
      return rewriter.notifyMatchFailure(segStore,
                                         "only segment_count = 2 is in scope");
    if (segStore.getDestinationMemoryForm() !=
            "segment2-interleaved-unit-stride-store" ||
        segStore.getInactiveLanePolicy() != "preserve-output-on-false-lanes")
      return rewriter.notifyMatchFailure(
          segStore, "masked_segment2_store form/policy outside the slice");
    if (!isMaskFromMaskLoadOrCompare(segStore.getMask()))
      return rewriter.notifyMatchFailure(
          segStore, "masked_segment2_store mask must come from a compare in the "
                    "same scope");
    auto field0Type =
        llvm::dyn_cast<weftrvv::VectorType>(segStore.getField0().getType());
    auto field1Type =
        llvm::dyn_cast<weftrvv::VectorType>(segStore.getField1().getType());
    if (!field0Type || !field1Type || field0Type != field1Type)
      return rewriter.notifyMatchFailure(
          segStore,
          "masked_segment2_store fields must be matching typed vectors");
    Segment2Facts facts;
    if (!resolveSegment2Facts(rewriter, field0Type, facts))
      return rewriter.notifyMatchFailure(
          segStore, "masked_segment2_store field vector type not convertible");
    mlir::Value base = valueMap.lookup(segStore.getDestination());
    mlir::Value mask = valueMap.lookup(segStore.getMask());
    mlir::Value field0 = valueMap.lookup(segStore.getField0());
    mlir::Value field1 = valueMap.lookup(segStore.getField1());
    if (!base || !mask || !field0 || !field1)
      return rewriter.notifyMatchFailure(segStore,
                                         "masked_segment2_store operand unmapped");
    if (!bufferPointeeMatchesVectorElement(base, field0Type))
      return rewriter.notifyMatchFailure(
          segStore, "masked_segment2_store dst C type disagrees with field "
                    "element");

    // Step 1: pack the two payload fields into one tuple.
    std::string createCallee =
        riscvSegment2TupleCreateIntrinsicName(facts.dtype, facts.lmul);
    mlir::Value tuple = emitOpaqueCall(
        rewriter, loc, facts.tupleType, createCallee,
        mlir::ValueRange{field0, field1},
        segStore.getWEFTEmitCLowerableSourceOpName(),
        segStore.getWEFTEmitCLowerableSourceRole());

    // Step 2: masked tuple store to the interleaved destination.
    std::string storeCallee =
        riscvMaskedSegment2StoreIntrinsicName(facts.sew, facts.lmul,
                                              facts.dtype);
    // Void interleave (interleaved pointer built between comment and call):
    // split the mangler string at the first underscore -- split/rejoin identity.
    auto [storeMnemonic, storeSuffix] =
        llvm::StringRef(storeCallee)
            .drop_front(llvm::StringLiteral("__riscv_").size())
            .split('_');
    emitVCallVoidBuilt(rewriter, loc, storeMnemonic, storeSuffix,
                       segStore.getWEFTEmitCLowerableSourceOpName(),
                       segStore.getWEFTEmitCLowerableSourceRole(),
                       [&](mlir::OpBuilder &,
                           mlir::Location) -> llvm::SmallVector<mlir::Value> {
                         mlir::Value ptr = emitSegment2InterleavedPointer(
                             rewriter, loc, base, inductionVar);
                         return {mask, ptr, tuple, bodyVL};
                       });
    return mlir::success();
  }

mlir::Value VariantToEmitCFunc::emitSegment2FieldExtract(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    llvm::StringRef opName, llvm::StringRef role, llvm::StringRef callee,
    mlir::Value tuple, mlir::Type fieldVecType, unsigned index) const {
    return emitOpaqueCallBuilt(
        rewriter, loc, fieldVecType, callee, opName, role,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value indexLiteral = b.create<emitc::LiteralOp>(
              l, b.getIndexType(), llvm::Twine(index).str());
          return {tuple, indexLiteral};
        });
  }

mlir::LogicalResult
VariantToEmitCFunc::emitMove(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
         weftrvv::MoveOp move,
         llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
         llvm::DenseMap<mlir::Value, std::pair<mlir::Value, unsigned>>
             &segmentFieldMap) const {
    if (move.getKind() != "copy")
      return rewriter.notifyMatchFailure(move, "unsupported move kind");
    // segment2 deinterleave: the move sources a segment2_load field result.
    auto segIt = segmentFieldMap.find(move.getSource());
    if (segIt != segmentFieldMap.end()) {
      auto vectorType =
          llvm::dyn_cast<weftrvv::VectorType>(move.getResult().getType());
      if (!vectorType)
        return rewriter.notifyMatchFailure(
            move, "segment2 move result not a typed vector");
      Segment2Facts facts;
      if (!resolveSegment2Facts(rewriter, vectorType, facts))
        return rewriter.notifyMatchFailure(
            move, "segment2 move field vector type not convertible");
      mlir::Value tuple = segIt->second.first;
      unsigned index = segIt->second.second;
      std::string callee =
          riscvSegment2FieldExtractIntrinsicName(facts.dtype, facts.lmul);
      mlir::Value field = emitSegment2FieldExtract(
          rewriter, loc, move.getWEFTEmitCLowerableSourceOpName(),
          move.getWEFTEmitCLowerableSourceRole(), callee, tuple,
          facts.fieldVecType, index);
      valueMap[move.getResult()] = field;
      return mlir::success();
    }
    mlir::Value source = valueMap.lookup(move.getSource());
    if (!source)
      return rewriter.notifyMatchFailure(move, "move source unmapped");
    valueMap[move.getResult()] = source;
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitIndexLoad(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
              weftrvv::IndexLoadOp indexLoad,
              llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
              mlir::Value inductionVar, mlir::Value bodyVL) const {
    auto indexVecType =
        llvm::dyn_cast<weftrvv::IndexVectorType>(indexLoad.getLoaded().getType());
    if (!indexVecType)
      return rewriter.notifyMatchFailure(indexLoad,
                                         "index_load result not index vector");
    mlir::Value base = valueMap.lookup(indexLoad.getIndex());
    if (!base)
      return rewriter.notifyMatchFailure(indexLoad,
                                         "index_load buffer not an ABI param");
    // The index buffer MUST be an unsigned-32 pointer (the index vectors are
    // u32). A mismatched index pointee width would dereference at the wrong
    // element width -- reject so the malformed body falls back.
    if (!indexBufferIsU32(base))
      return rewriter.notifyMatchFailure(
          indexLoad, "index_load buffer C type is not a uint32_t pointer");
    mlir::Type vecType = convertIndexVectorTypeToEmitC(indexVecType);
    if (!vecType)
      return rewriter.notifyMatchFailure(indexLoad,
                                         "index vector type not convertible");
    unsigned eew = static_cast<unsigned>(indexLoad.getIndexEew());
    if (eew != 32)
      return rewriter.notifyMatchFailure(indexLoad,
                                         "only EEW=32 index loads are in scope");
    std::string callee = riscvIntrinsicName("vle", eew, indexVecType.getLmul(),
                                            "u32");
    mlir::Value loaded = emitOpaqueCallBuilt(
        rewriter, loc, vecType, callee,
        indexLoad.getWEFTEmitCLowerableSourceOpName(),
        indexLoad.getWEFTEmitCLowerableSourceRole(),
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value ptr =
              b.create<emitc::AddOp>(l, base.getType(), base, inductionVar);
          return {ptr, bodyVL};
        });

    // Computed-mask indexed path: the index_load feeds a masked indexed
    // gather/scatter. The string-plan byte-order (which the harness ordered-
    // token validator depends on) emits the element->byte scale EARLY, right
    // after the index_load and before the splat/loads/compare. Emit the scale
    // here and map the index_load result to the SCALED byte-offset vector so the
    // masked-indexed emitter consumes it directly (it skips its own scale). The
    // base-memory plain indexed path leaves the raw index here and scales inside
    // its own emitter (no early ordering constraint).
    if (mlir::Operation *maskedConsumer = maskedIndexedConsumer(indexLoad)) {
      weftrvv::VectorType dataVectorType = maskedIndexedDataVectorType(
          maskedConsumer);
      if (!dataVectorType)
        return rewriter.notifyMatchFailure(
            indexLoad, "masked indexed consumer data vector not typed");
      llvm::StringRef consumerOpName;
      llvm::StringRef consumerRole;
      if (auto load =
              llvm::dyn_cast<weftrvv::MaskedIndexedLoadOp>(maskedConsumer)) {
        consumerOpName = load.getWEFTEmitCLowerableSourceOpName();
        consumerRole = load.getWEFTEmitCLowerableSourceRole();
      } else {
        auto store = llvm::cast<weftrvv::MaskedIndexedStoreOp>(maskedConsumer);
        consumerOpName = store.getWEFTEmitCLowerableSourceOpName();
        consumerRole = store.getWEFTEmitCLowerableSourceRole();
      }
      mlir::Value byteIndices = emitIndexByteScale(
          rewriter, loc, consumerOpName, consumerRole, loaded, vecType,
          indexVecType, dataVectorType, bodyVL);
      valueMap[indexLoad.getLoaded()] = byteIndices;
      return mlir::success();
    }

    valueMap[indexLoad.getLoaded()] = loaded;
    return mlir::success();
  }

mlir::Operation *VariantToEmitCFunc::maskedIndexedConsumer(weftrvv::IndexLoadOp indexLoad) {
    for (mlir::Operation *user : indexLoad.getLoaded().getUsers())
      if (llvm::isa<weftrvv::MaskedIndexedLoadOp,
                    weftrvv::MaskedIndexedStoreOp>(user))
        return user;
    return nullptr;
  }

weftrvv::VectorType
VariantToEmitCFunc::maskedIndexedDataVectorType(mlir::Operation *maskedConsumer) {
    if (auto load = llvm::dyn_cast<weftrvv::MaskedIndexedLoadOp>(maskedConsumer))
      return llvm::dyn_cast<weftrvv::VectorType>(load.getLoaded().getType());
    auto store = llvm::cast<weftrvv::MaskedIndexedStoreOp>(maskedConsumer);
    return llvm::dyn_cast<weftrvv::VectorType>(store.getValue().getType());
  }

mlir::LogicalResult
VariantToEmitCFunc::emitIndexedLoad(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                weftrvv::IndexedLoadOp indexedLoad,
                llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<weftrvv::VectorType>(indexedLoad.getLoaded().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(indexedLoad,
                                         "indexed_load result not typed vector");
    if (indexedLoad.getOffsetUnit() != "element")
      return rewriter.notifyMatchFailure(
          indexedLoad, "only element-offset indexed loads are in scope");
    auto indexVecType = llvm::dyn_cast<weftrvv::IndexVectorType>(
        indexedLoad.getIndices().getType());
    if (!indexVecType)
      return rewriter.notifyMatchFailure(indexedLoad,
                                         "indexed_load indices not index vector");
    mlir::Value dataBase = valueMap.lookup(indexedLoad.getData());
    mlir::Value indices = valueMap.lookup(indexedLoad.getIndices());
    if (!dataBase || !indices)
      return rewriter.notifyMatchFailure(indexedLoad,
                                         "indexed_load operand unmapped");
    if (!bufferPointeeMatchesVectorElement(dataBase, vectorType))
      return rewriter.notifyMatchFailure(
          indexedLoad, "indexed_load data C type disagrees with loaded element");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    mlir::Type indexEmitCType = convertIndexVectorTypeToEmitC(indexVecType);
    if (!vecType || !indexEmitCType)
      return rewriter.notifyMatchFailure(indexedLoad,
                                         "indexed_load type not convertible");
    unsigned eew = static_cast<unsigned>(indexedLoad.getIndexEew());
    if (eew != 32)
      return rewriter.notifyMatchFailure(
          indexedLoad, "only EEW=32 indexed loads are in scope");

    mlir::Value byteIndices = emitIndexByteScale(
        rewriter, loc, indexedLoad.getWEFTEmitCLowerableSourceOpName(),
        indexedLoad.getWEFTEmitCLowerableSourceRole(), indices, indexEmitCType,
        indexVecType, vectorType, bodyVL);
    std::string callee = riscvIndexedMemoryIntrinsicName(
        "vloxei", eew, vectorDType(vectorType), vectorType.getLmul());
    mlir::Value loaded = emitOpaqueCall(
        rewriter, loc, vecType, callee,
        mlir::ValueRange{dataBase, byteIndices, bodyVL},
        indexedLoad.getWEFTEmitCLowerableSourceOpName(),
        indexedLoad.getWEFTEmitCLowerableSourceRole());
    valueMap[indexedLoad.getLoaded()] = loaded;
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitIndexedStore(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                 weftrvv::IndexedStoreOp indexedStore,
                 llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                 mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<weftrvv::VectorType>(indexedStore.getValue().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(indexedStore,
                                         "indexed_store value not typed vector");
    if (indexedStore.getOffsetUnit() != "element")
      return rewriter.notifyMatchFailure(
          indexedStore, "only element-offset indexed stores are in scope");
    if (indexedStore.getIndexUniqueness() != "unique")
      return rewriter.notifyMatchFailure(
          indexedStore, "only unique-index indexed stores are in scope");
    auto indexVecType = llvm::dyn_cast<weftrvv::IndexVectorType>(
        indexedStore.getIndices().getType());
    if (!indexVecType)
      return rewriter.notifyMatchFailure(
          indexedStore, "indexed_store indices not index vector");
    mlir::Value dstBase = valueMap.lookup(indexedStore.getDestination());
    mlir::Value indices = valueMap.lookup(indexedStore.getIndices());
    mlir::Value value = valueMap.lookup(indexedStore.getValue());
    if (!dstBase || !indices || !value)
      return rewriter.notifyMatchFailure(indexedStore,
                                         "indexed_store operand unmapped");
    if (!bufferPointeeMatchesVectorElement(dstBase, vectorType))
      return rewriter.notifyMatchFailure(
          indexedStore,
          "indexed_store destination C type disagrees with stored element");
    mlir::Type indexEmitCType = convertIndexVectorTypeToEmitC(indexVecType);
    if (!convertVectorTypeToEmitC(vectorType) || !indexEmitCType)
      return rewriter.notifyMatchFailure(indexedStore,
                                         "indexed_store type not convertible");
    unsigned eew = static_cast<unsigned>(indexedStore.getIndexEew());
    if (eew != 32)
      return rewriter.notifyMatchFailure(
          indexedStore, "only EEW=32 indexed stores are in scope");

    mlir::Value byteIndices = emitIndexByteScale(
        rewriter, loc, indexedStore.getWEFTEmitCLowerableSourceOpName(),
        indexedStore.getWEFTEmitCLowerableSourceRole(), indices, indexEmitCType,
        indexVecType, vectorType, bodyVL);
    std::string callee = riscvIndexedMemoryIntrinsicName(
        "vsoxei", eew, vectorDType(vectorType), vectorType.getLmul());
    emitOpaqueCallVoid(rewriter, loc, callee,
                       mlir::ValueRange{dstBase, byteIndices, value, bodyVL},
                       indexedStore.getWEFTEmitCLowerableSourceOpName(),
                       indexedStore.getWEFTEmitCLowerableSourceRole());
    return mlir::success();
  }

mlir::Value
VariantToEmitCFunc::emitIndexByteScale(mlir::ConversionPatternRewriter &rewriter,
                   mlir::Location loc, llvm::StringRef sourceOpName,
                   llvm::StringRef sourceRole, mlir::Value indices,
                   mlir::Type indexEmitCType,
                   weftrvv::IndexVectorType indexVecType,
                   weftrvv::VectorType dataVectorType,
                   mlir::Value bodyVL) const {
    std::string scaleCallee =
        riscvIndexScaleIntrinsicName("u32", indexVecType.getLmul());
    unsigned elemBytes = vectorElementWidth(dataVectorType) / 8;
    return emitOpaqueCallBuilt(
        rewriter, loc, indexEmitCType, scaleCallee, sourceOpName, sourceRole,
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value bytesLiteral = b.create<emitc::LiteralOp>(
              l, emitc::OpaqueType::get(b.getContext(), "size_t"),
              llvm::Twine(elemBytes).str());
          return {indices, bytesLiteral, bodyVL};
        });
  }

mlir::LogicalResult
VariantToEmitCFunc::emitMaskLoad(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
             weftrvv::MaskLoadOp maskLoad,
             llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
             mlir::Value inductionVar, mlir::Value bodyVL) const {
    auto maskType =
        llvm::dyn_cast<weftrvv::MaskType>(maskLoad.getLoaded().getType());
    if (!maskType)
      return rewriter.notifyMatchFailure(maskLoad,
                                         "mask_load result not typed mask");
    if (maskLoad.getMaskMemoryForm() != "unit-stride-mask-load")
      return rewriter.notifyMatchFailure(
          maskLoad, "only unit-stride mask loads are in scope");
    mlir::Value base = valueMap.lookup(maskLoad.getMask());
    if (!base)
      return rewriter.notifyMatchFailure(maskLoad,
                                         "mask_load buffer not an ABI param");
    // The mask buffer is read at the predicate element width (the data vector
    // element width); a mismatched pointee would test the wrong width.
    unsigned sew = 0;
    if (maskType.getElementType().isSignlessInteger(32))
      sew = 32;
    else if (maskType.getElementType().isSignlessInteger(64))
      sew = 64;
    else
      return rewriter.notifyMatchFailure(maskLoad, "unsupported mask element");
    llvm::StringRef dtype;
    if (sew == 32)
      dtype = "i32";
    else
      dtype = "i64";
    if (!maskBufferPointeeMatches(base, dtype))
      return rewriter.notifyMatchFailure(
          maskLoad, "mask_load buffer C type disagrees with mask element width");
    unsigned maskBits = maskWidthForConfig(sew, maskType.getLmul());
    if (maskBits == 0)
      return rewriter.notifyMatchFailure(maskLoad, "unsupported mask config");
    mlir::Type maskEmitCType = getTypeConverter()->convertType(maskType);
    if (!maskEmitCType ||
        maskEmitCType.getDialect().getNamespace() !=
            emitc::EmitCDialect::getDialectNamespace())
      return rewriter.notifyMatchFailure(maskLoad, "mask type not convertible");
    mlir::Type dataVecEmitCType =
        emitc::OpaqueType::get(rewriter.getContext(),
                               ("vint" + llvm::Twine(sew) +
                                maskType.getLmul() + "_t")
                                   .str());

    // Step 1: unit-stride load the mask buffer as a data vector.
    std::string loadCallee = riscvIntrinsicName("vle", sew, maskType.getLmul(),
                                                dtype);
    mlir::Value maskVec = emitOpaqueCallBuilt(
        rewriter, loc, dataVecEmitCType, loadCallee,
        maskLoad.getWEFTEmitCLowerableSourceOpName(),
        maskLoad.getWEFTEmitCLowerableSourceRole(),
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value ptr =
              b.create<emitc::AddOp>(l, base.getType(), base, inductionVar);
          return {ptr, bodyVL};
        });

    // Step 2: lane != 0 -> predicate mask.
    std::string maskCallee =
        riscvMaskNonzeroIntrinsicName(sew, maskType.getLmul(), dtype, maskBits);
    mlir::Value mask = emitOpaqueCallBuilt(
        rewriter, loc, maskEmitCType, maskCallee,
        maskLoad.getWEFTEmitCLowerableSourceOpName(),
        maskLoad.getWEFTEmitCLowerableSourceRole(),
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value zeroLiteral = b.create<emitc::LiteralOp>(
              l, emitc::OpaqueType::get(b.getContext(), "int"), "0");
          return {maskVec, zeroLiteral, bodyVL};
        });
    valueMap[maskLoad.getLoaded()] = mask;
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitMaskedLoad(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
               weftrvv::MaskedLoadOp maskedLoad,
               llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
               mlir::Value inductionVar, mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<weftrvv::VectorType>(maskedLoad.getLoaded().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(maskedLoad,
                                         "masked_load result not typed vector");
    if (maskedLoad.getMemoryForm() != "masked-unit-load" ||
        maskedLoad.getInactiveLanePolicy() !=
            "preserve-passthrough-on-false-lanes")
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_load memory form/policy outside the slice");
    // The predicate authority is structural: the BASE-memory masked family reads
    // its mask from an explicit weft_rvv.mask_load buffer; the computed-mask
    // memory family produces it from a weft_rvv.compare in the same VL scope.
    // Both lower to the byte-identical `_tumu` masked-load form. Accept either
    // authority, but refuse a mask from any other op so a malformed body (an
    // unmaterialized or out-of-family mask producer) falls back to the legacy
    // validator rather than being mislowered.
    if (!isMaskFromMaskLoadOrCompare(maskedLoad.getMask()))
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_load mask must come from explicit mask_load "
                      "buffer authority or a compare in the same scope");
    mlir::Value base = valueMap.lookup(maskedLoad.getBuffer());
    mlir::Value mask = valueMap.lookup(maskedLoad.getMask());
    mlir::Value passthrough = valueMap.lookup(maskedLoad.getPassthrough());
    if (!base || !mask || !passthrough)
      return rewriter.notifyMatchFailure(maskedLoad,
                                         "masked_load operand unmapped");
    if (!bufferPointeeMatchesVectorElement(base, vectorType))
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_load buffer C type disagrees with loaded element");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(maskedLoad,
                                         "vector type not convertible");
    std::string callee =
        riscvMaskedLoadIntrinsicName(vectorElementWidth(vectorType),
                                     vectorType.getLmul(),
                                     vectorDType(vectorType));
    mlir::Value loaded = emitOpaqueCallBuilt(
        rewriter, loc, vecType, callee,
        maskedLoad.getWEFTEmitCLowerableSourceOpName(),
        maskedLoad.getWEFTEmitCLowerableSourceRole(),
        [&](mlir::OpBuilder &b,
            mlir::Location l) -> llvm::SmallVector<mlir::Value> {
          mlir::Value ptr =
              b.create<emitc::AddOp>(l, base.getType(), base, inductionVar);
          return {mask, passthrough, ptr, bodyVL};
        });
    valueMap[maskedLoad.getLoaded()] = loaded;
    return mlir::success();
  }

mlir::LogicalResult
VariantToEmitCFunc::emitMaskedStore(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                weftrvv::MaskedStoreOp maskedStore,
                llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                mlir::Value inductionVar, mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<weftrvv::VectorType>(maskedStore.getValue().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(maskedStore,
                                         "masked_store value not typed vector");
    if (maskedStore.getMemoryForm() != "masked-unit-store" ||
        maskedStore.getInactiveLanePolicy() !=
            "preserve-output-on-false-lanes")
      return rewriter.notifyMatchFailure(
          maskedStore, "masked_store memory form/policy outside the slice");
    // See emitMaskedLoad: accept a mask from explicit mask_load buffer authority
    // (base-memory family) OR a compare in the same scope (computed-mask family);
    // refuse any other producer so a malformed body falls back.
    if (!isMaskFromMaskLoadOrCompare(maskedStore.getMask()))
      return rewriter.notifyMatchFailure(
          maskedStore, "masked_store mask must come from explicit mask_load "
                       "buffer authority or a compare in the same scope");
    mlir::Value base = valueMap.lookup(maskedStore.getBuffer());
    mlir::Value mask = valueMap.lookup(maskedStore.getMask());
    mlir::Value value = valueMap.lookup(maskedStore.getValue());
    if (!base || !mask || !value)
      return rewriter.notifyMatchFailure(maskedStore,
                                         "masked_store operand unmapped");
    if (!bufferPointeeMatchesVectorElement(base, vectorType))
      return rewriter.notifyMatchFailure(
          maskedStore,
          "masked_store buffer C type disagrees with stored element");
    if (!convertVectorTypeToEmitC(vectorType))
      return rewriter.notifyMatchFailure(maskedStore,
                                         "vector type not convertible");
    std::string callee =
        riscvMaskedStoreIntrinsicName(vectorElementWidth(vectorType),
                                      vectorType.getLmul(),
                                      vectorDType(vectorType));
    // Void interleave (pointer add built between comment and call): split the
    // mangler string at the first underscore -- split/rejoin identity.
    auto [storeMnemonic, storeSuffix] =
        llvm::StringRef(callee)
            .drop_front(llvm::StringLiteral("__riscv_").size())
            .split('_');
    emitVCallVoidBuilt(rewriter, loc, storeMnemonic, storeSuffix,
                       maskedStore.getWEFTEmitCLowerableSourceOpName(),
                       maskedStore.getWEFTEmitCLowerableSourceRole(),
                       [&](mlir::OpBuilder &b, mlir::Location l)
                           -> llvm::SmallVector<mlir::Value> {
                         mlir::Value ptr = b.create<emitc::AddOp>(
                             l, base.getType(), base, inductionVar);
                         return {mask, ptr, value, bodyVL};
                       });
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitMaskedStridedLoad(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::MaskedStridedLoadOp maskedLoad,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    mlir::Value inductionVar, mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<weftrvv::VectorType>(maskedLoad.getLoaded().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_strided_load result not typed vector");
    if (maskedLoad.getMemoryForm() != "masked-strided-load" ||
        maskedLoad.getStrideUnit() != "byte" ||
        maskedLoad.getInactiveLanePolicy() !=
            "preserve-passthrough-on-false-lanes")
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_strided_load form/unit/policy outside the slice");
    if (!isMaskFromMaskLoadOrCompare(maskedLoad.getMask()))
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_strided_load mask must come from explicit "
                      "mask_load buffer authority or a compare in the same "
                      "scope");
    if (!isByteStride(maskedLoad.getStride()))
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_strided_load requires a byte-stride ABI role");
    mlir::Value base = valueMap.lookup(maskedLoad.getBuffer());
    mlir::Value mask = valueMap.lookup(maskedLoad.getMask());
    mlir::Value passthrough = valueMap.lookup(maskedLoad.getPassthrough());
    mlir::Value stride = valueMap.lookup(maskedLoad.getStride());
    if (!base || !mask || !passthrough || !stride)
      return rewriter.notifyMatchFailure(maskedLoad,
                                         "masked_strided_load operand unmapped");
    if (!bufferPointeeMatchesVectorElement(base, vectorType))
      return rewriter.notifyMatchFailure(
          maskedLoad,
          "masked_strided_load buffer C type disagrees with loaded element");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(maskedLoad,
                                         "vector type not convertible");
    std::string callee = riscvMaskedStridedLoadIntrinsicName(
        vectorElementWidth(vectorType), vectorType.getLmul(),
        vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(maskedLoad.getWEFTEmitCLowerableSourceOpName(),
                         maskedLoad.getWEFTEmitCLowerableSourceRole(), callee));
    mlir::Value ptr =
        emitByteStridedPointer(rewriter, loc, base, inductionVar, stride);
    if (!ptr)
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_strided_load base must be a pointer-typed ABI "
                      "param");
    mlir::Value loaded =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{vecType}, callee,
                mlir::ValueRange{mask, passthrough, ptr, stride, bodyVL})
            .getResult(0);
    valueMap[maskedLoad.getLoaded()] = loaded;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitMaskedStridedStore(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::MaskedStridedStoreOp maskedStore,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    mlir::Value inductionVar, mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<weftrvv::VectorType>(maskedStore.getValue().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(
          maskedStore, "masked_strided_store value not typed vector");
    if (maskedStore.getMemoryForm() != "masked-strided-store" ||
        maskedStore.getStrideUnit() != "byte" ||
        maskedStore.getInactiveLanePolicy() != "preserve-output-on-false-lanes")
      return rewriter.notifyMatchFailure(
          maskedStore,
          "masked_strided_store form/unit/policy outside the slice");
    if (!isMaskFromMaskLoadOrCompare(maskedStore.getMask()))
      return rewriter.notifyMatchFailure(
          maskedStore, "masked_strided_store mask must come from explicit "
                       "mask_load buffer authority or a compare in the same "
                       "scope");
    if (!isByteStride(maskedStore.getStride()))
      return rewriter.notifyMatchFailure(
          maskedStore, "masked_strided_store requires a byte-stride ABI role");
    mlir::Value base = valueMap.lookup(maskedStore.getBuffer());
    mlir::Value mask = valueMap.lookup(maskedStore.getMask());
    mlir::Value value = valueMap.lookup(maskedStore.getValue());
    mlir::Value stride = valueMap.lookup(maskedStore.getStride());
    if (!base || !mask || !value || !stride)
      return rewriter.notifyMatchFailure(
          maskedStore, "masked_strided_store operand unmapped");
    if (!bufferPointeeMatchesVectorElement(base, vectorType))
      return rewriter.notifyMatchFailure(
          maskedStore,
          "masked_strided_store buffer C type disagrees with stored element");
    if (!convertVectorTypeToEmitC(vectorType))
      return rewriter.notifyMatchFailure(maskedStore,
                                         "vector type not convertible");
    std::string callee = riscvMaskedStridedStoreIntrinsicName(
        vectorElementWidth(vectorType), vectorType.getLmul(),
        vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(maskedStore.getWEFTEmitCLowerableSourceOpName(),
                         maskedStore.getWEFTEmitCLowerableSourceRole(), callee));
    mlir::Value ptr =
        emitByteStridedPointer(rewriter, loc, base, inductionVar, stride);
    if (!ptr)
      return rewriter.notifyMatchFailure(
          maskedStore, "masked_strided_store base must be a pointer-typed ABI "
                       "param");
    rewriter.create<emitc::CallOpaqueOp>(
        loc, mlir::TypeRange{}, callee,
        mlir::ValueRange{mask, ptr, stride, value, bodyVL});
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitMaskedIndexedLoad(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::MaskedIndexedLoadOp maskedLoad,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<weftrvv::VectorType>(maskedLoad.getLoaded().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_indexed_load result not typed vector");
    if (maskedLoad.getMemoryForm() != "masked-indexed-load" ||
        maskedLoad.getOffsetUnit() != "element" ||
        maskedLoad.getInactiveLanePolicy() !=
            "preserve-passthrough-on-false-lanes")
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_indexed_load form/unit/policy outside the slice");
    if (!isMaskFromMaskLoadOrCompare(maskedLoad.getMask()))
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_indexed_load mask must come from explicit "
                      "mask_load buffer authority or a compare in the same "
                      "scope");
    auto indexVecType = llvm::dyn_cast<weftrvv::IndexVectorType>(
        maskedLoad.getIndices().getType());
    if (!indexVecType)
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_indexed_load indices not index vector");
    mlir::Value dataBase = valueMap.lookup(maskedLoad.getData());
    mlir::Value indices = valueMap.lookup(maskedLoad.getIndices());
    mlir::Value mask = valueMap.lookup(maskedLoad.getMask());
    mlir::Value passthrough = valueMap.lookup(maskedLoad.getPassthrough());
    if (!dataBase || !indices || !mask || !passthrough)
      return rewriter.notifyMatchFailure(maskedLoad,
                                         "masked_indexed_load operand unmapped");
    if (!bufferPointeeMatchesVectorElement(dataBase, vectorType))
      return rewriter.notifyMatchFailure(
          maskedLoad,
          "masked_indexed_load data C type disagrees with loaded element");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    mlir::Type indexEmitCType = convertIndexVectorTypeToEmitC(indexVecType);
    if (!vecType || !indexEmitCType)
      return rewriter.notifyMatchFailure(maskedLoad,
                                         "masked_indexed_load type not "
                                         "convertible");
    unsigned eew = static_cast<unsigned>(maskedLoad.getIndexEew());
    if (eew != 32)
      return rewriter.notifyMatchFailure(
          maskedLoad, "only EEW=32 masked indexed loads are in scope");

    // The element->byte index scale was emitted early at index_load time (the
    // computed-mask index-early order), so `indices` is already the byte-offset
    // vector -- consume it directly.
    mlir::Value byteIndices = indices;
    std::string callee = riscvMaskedIndexedLoadIntrinsicName(
        eew, vectorDType(vectorType), vectorType.getLmul());
    mlir::Value loaded = emitOpaqueCall(
        rewriter, loc, vecType, callee,
        mlir::ValueRange{mask, passthrough, dataBase, byteIndices, bodyVL},
        maskedLoad.getWEFTEmitCLowerableSourceOpName(),
        maskedLoad.getWEFTEmitCLowerableSourceRole());
    valueMap[maskedLoad.getLoaded()] = loaded;
    return mlir::success();
  }

mlir::LogicalResult VariantToEmitCFunc::emitMaskedIndexedStore(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    weftrvv::MaskedIndexedStoreOp maskedStore,
    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
    mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<weftrvv::VectorType>(maskedStore.getValue().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(
          maskedStore, "masked_indexed_store value not typed vector");
    if (maskedStore.getMemoryForm() != "masked-indexed-store" ||
        maskedStore.getOffsetUnit() != "element" ||
        maskedStore.getIndexUniqueness() != "unique" ||
        maskedStore.getInactiveLanePolicy() != "preserve-output-on-false-lanes")
      return rewriter.notifyMatchFailure(
          maskedStore,
          "masked_indexed_store form/unit/uniqueness/policy outside the slice");
    if (!isMaskFromMaskLoadOrCompare(maskedStore.getMask()))
      return rewriter.notifyMatchFailure(
          maskedStore, "masked_indexed_store mask must come from explicit "
                       "mask_load buffer authority or a compare in the same "
                       "scope");
    auto indexVecType = llvm::dyn_cast<weftrvv::IndexVectorType>(
        maskedStore.getIndices().getType());
    if (!indexVecType)
      return rewriter.notifyMatchFailure(
          maskedStore, "masked_indexed_store indices not index vector");
    mlir::Value dstBase = valueMap.lookup(maskedStore.getDestination());
    mlir::Value indices = valueMap.lookup(maskedStore.getIndices());
    mlir::Value mask = valueMap.lookup(maskedStore.getMask());
    mlir::Value value = valueMap.lookup(maskedStore.getValue());
    if (!dstBase || !indices || !mask || !value)
      return rewriter.notifyMatchFailure(
          maskedStore, "masked_indexed_store operand unmapped");
    if (!bufferPointeeMatchesVectorElement(dstBase, vectorType))
      return rewriter.notifyMatchFailure(
          maskedStore,
          "masked_indexed_store destination C type disagrees with stored "
          "element");
    mlir::Type indexEmitCType = convertIndexVectorTypeToEmitC(indexVecType);
    if (!convertVectorTypeToEmitC(vectorType) || !indexEmitCType)
      return rewriter.notifyMatchFailure(maskedStore,
                                         "masked_indexed_store type not "
                                         "convertible");
    unsigned eew = static_cast<unsigned>(maskedStore.getIndexEew());
    if (eew != 32)
      return rewriter.notifyMatchFailure(
          maskedStore, "only EEW=32 masked indexed stores are in scope");

    // The element->byte index scale was emitted early at index_load time (the
    // computed-mask index-early order), so `indices` is already the byte-offset
    // vector -- consume it directly.
    mlir::Value byteIndices = indices;
    std::string callee = riscvMaskedIndexedStoreIntrinsicName(
        eew, vectorDType(vectorType), vectorType.getLmul());
    emitOpaqueCallVoid(
        rewriter, loc, callee,
        mlir::ValueRange{mask, dstBase, byteIndices, value, bodyVL},
        maskedStore.getWEFTEmitCLowerableSourceOpName(),
        maskedStore.getWEFTEmitCLowerableSourceRole());
    return mlir::success();
  }

bool VariantToEmitCFunc::indexBufferIsU32(mlir::Value bufferValue) {
    auto pointerType =
        llvm::dyn_cast<emitc::PointerType>(bufferValue.getType());
    if (!pointerType)
      return false;
    auto pointeeOpaque =
        llvm::dyn_cast<emitc::OpaqueType>(pointerType.getPointee());
    if (!pointeeOpaque)
      return false;
    return pointeeOpaque.getValue().contains("uint32_t");
  }

bool VariantToEmitCFunc::maskBufferPointeeMatches(mlir::Value bufferValue,
                                     llvm::StringRef dtype) {
    auto pointerType =
        llvm::dyn_cast<emitc::PointerType>(bufferValue.getType());
    if (!pointerType)
      return false;
    auto pointeeOpaque =
        llvm::dyn_cast<emitc::OpaqueType>(pointerType.getPointee());
    if (!pointeeOpaque)
      return false;
    llvm::StringRef scalar = (dtype == "i32") ? "int32_t" : "int64_t";
    return pointeeOpaque.getValue().contains(scalar);
  }

mlir::Type
VariantToEmitCFunc::convertIndexVectorTypeToEmitC(weftrvv::IndexVectorType type) const {
    mlir::Type converted = getTypeConverter()->convertType(type);
    if (!converted)
      return nullptr;
    if (converted.getDialect().getNamespace() !=
        mlir::emitc::EmitCDialect::getDialectNamespace())
      return nullptr;
    return converted;
  }

mlir::Value VariantToEmitCFunc::emitScaledPointer(mlir::ConversionPatternRewriter &rewriter,
                              mlir::Location loc, mlir::Value base,
                              mlir::Value inductionVar,
                              mlir::Value stride) const {
    mlir::Value scaledOffset = rewriter.create<emitc::MulOp>(
        loc, inductionVar.getType(), inductionVar, stride);
    return rewriter.create<emitc::AddOp>(loc, base.getType(), base,
                                         scaledOffset);
  }

mlir::Value VariantToEmitCFunc::emitByteStride(mlir::ConversionPatternRewriter &rewriter,
                           mlir::Location loc, mlir::Value stride,
                           weftrvv::VectorType vectorType) const {
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

bool VariantToEmitCFunc::isByteStride(mlir::Value strideToken) {
    auto abi = strideToken.getDefiningOp<weftrvv::RuntimeABIValueOp>();
    if (!abi)
      return false;
    return abi.getRole().ends_with("byte-stride");
  }

bool VariantToEmitCFunc::loadedFeedsMove(weftrvv::StridedLoadOp load) {
    return llvm::any_of(load.getLoaded().getUsers(), [](mlir::Operation *user) {
      return llvm::isa<weftrvv::MoveOp>(user);
    });
  }

bool VariantToEmitCFunc::storedValueFromMove(weftrvv::StridedStoreOp store) {
    return llvm::isa_and_present<weftrvv::MoveOp>(
        store.getValue().getDefiningOp());
  }

bool VariantToEmitCFunc::isMaskFromMaskLoadOrCompare(mlir::Value mask) {
    mlir::Operation *def = mask.getDefiningOp();
    return llvm::isa_and_present<weftrvv::MaskLoadOp, weftrvv::CompareOp>(def);
  }

bool VariantToEmitCFunc::fieldVectorBindsLoadRole(mlir::Value fieldVector,
                                     llvm::StringRef expectedRole) {
    auto load = fieldVector.getDefiningOp<weftrvv::LoadOp>();
    if (!load)
      return false;
    auto abi = load.getBuffer().getDefiningOp<weftrvv::RuntimeABIValueOp>();
    if (!abi)
      return false;
    return abi.getRole() == expectedRole;
  }

mlir::Value VariantToEmitCFunc::emitByteStridedPointer(mlir::ConversionPatternRewriter &rewriter,
                                   mlir::Location loc, mlir::Value base,
                                   mlir::Value inductionVar,
                                   mlir::Value stride) const {
    auto pointerType = llvm::dyn_cast<emitc::PointerType>(base.getType());
    if (!pointerType)
      return nullptr;
    auto pointeeOpaque =
        llvm::dyn_cast<emitc::OpaqueType>(pointerType.getPointee());
    if (!pointeeOpaque)
      return nullptr;
    // Preserve a leading `const` qualifier on the byte pointer so a
    // `const int32_t *` base becomes `const uint8_t *` (not `uint8_t *`).
    llvm::StringRef pointee = pointeeOpaque.getValue();
    std::string bytePointee =
        (pointee.contains("const") ? "const uint8_t" : "uint8_t");
    mlir::Type bytePtrType = emitc::PointerType::get(
        emitc::OpaqueType::get(rewriter.getContext(), bytePointee));
    mlir::Value byteBase =
        rewriter.create<emitc::CastOp>(loc, bytePtrType, base);
    mlir::Value byteOffset = rewriter.create<emitc::MulOp>(
        loc, inductionVar.getType(), inductionVar, stride);
    mlir::Value bytePtr =
        rewriter.create<emitc::AddOp>(loc, bytePtrType, byteBase, byteOffset);
    return rewriter.create<emitc::CastOp>(loc, base.getType(), bytePtr);
  }

unsigned VariantToEmitCFunc::vectorElementWidth(weftrvv::VectorType type) {
    if (auto intType =
            llvm::dyn_cast<mlir::IntegerType>(type.getElementType()))
      return intType.getWidth();
    if (auto floatType =
            llvm::dyn_cast<mlir::FloatType>(type.getElementType()))
      return floatType.getWidth();
    return 0;
  }

mlir::Type VariantToEmitCFunc::convertVectorTypeToEmitC(weftrvv::VectorType type) const {
    mlir::Type converted = getTypeConverter()->convertType(type);
    if (!converted)
      return nullptr;
    if (converted.getDialect().getNamespace() !=
        mlir::emitc::EmitCDialect::getDialectNamespace())
      return nullptr;
    return converted;
  }

std::optional<llvm::StringRef>
VariantToEmitCFunc::binaryMnemonic(llvm::StringRef kind, bool isFloat) {
    if (isFloat) {
      if (kind == "add")
        return llvm::StringRef("vfadd");
      if (kind == "sub")
        return llvm::StringRef("vfsub");
      if (kind == "mul")
        return llvm::StringRef("vfmul");
      return std::nullopt;
    }
    if (kind == "add")
      return llvm::StringRef("vadd");
    if (kind == "sub")
      return llvm::StringRef("vsub");
    if (kind == "mul")
      return llvm::StringRef("vmul");
    return std::nullopt;
  }

std::optional<llvm::StringRef> VariantToEmitCFunc::reductionMnemonic(llvm::StringRef kind) {
    if (kind == "add")
      return llvm::StringRef("vredsum");
    if (kind == "min")
      return llvm::StringRef("vredmin");
    if (kind == "max")
      return llvm::StringRef("vredmax");
    return std::nullopt;
  }

std::optional<llvm::StringRef> VariantToEmitCFunc::compareMnemonic(llvm::StringRef kind,
                                                      bool isFloat) {
    if (isFloat) {
      if (kind == "eq")
        return llvm::StringRef("vmfeq");
      if (kind == "slt")
        return llvm::StringRef("vmflt");
      if (kind == "sle")
        return llvm::StringRef("vmfle");
      return std::nullopt;
    }
    if (kind == "eq")
      return llvm::StringRef("vmseq");
    if (kind == "slt")
      return llvm::StringRef("vmslt");
    if (kind == "sle")
      return llvm::StringRef("vmsle");
    return std::nullopt;
  }

std::optional<llvm::StringRef> VariantToEmitCFunc::maskAndMnemonic(llvm::StringRef kind) {
    if (kind == "and")
      return llvm::StringRef("vmand");
    return std::nullopt;
  }

} // namespace detail

// The free functions below run at rvv scope and call the typed type/dtype
// support helpers (isUnsignedVector, maskWidthForConfig, ...) unqualified, as
// they did when the former anonymous namespace pulled `detail` in. Re-establish
// that visibility for this TU's rvv-scope code.
using namespace detail;

void populateRVVToEmitCTypeConversions(mlir::TypeConverter &typeConverter) {
  // !weft_rvv.vl -> emitc.opaque<"size_t"> (the RVV vector-length token is the
  // C size_t produced by __riscv_vsetvl_*).
  typeConverter.addConversion(
      [](weftrvv::VLType type) -> std::optional<mlir::Type> {
        return emitc::OpaqueType::get(type.getContext(), "size_t");
      });

  // !weft_rvv.vector<i<sew>, "m<lmul>"> -> emitc.opaque<"vint<sew>m<lmul>_t">,
  // !weft_rvv.vector<f<sew>, "m<lmul>"> -> emitc.opaque<"vfloat<sew>m<lmul>_t">.
  // The elementwise family covers the bounded {i32,i64} x {m1,m2} grid the
  // selected-body rungs use (e.g. i32/m1 -> vint32m1_t, i64/m1 -> vint64m1_t,
  // i32/m2 -> vint32m2_t, i64/m2 -> vint64m2_t). The compare-select family adds
  // the f32/m1 float grid (f32/m1 -> vfloat32m1_t) for the f32-clamp /
  // dequant-clamp rungs. Other (dtype, lmul) pairs are left unconverted on
  // purpose so later families extend the grid explicitly.
  typeConverter.addConversion(
      [](weftrvv::VectorType type) -> std::optional<mlir::Type> {
        llvm::StringRef lmul = type.getLmul();
        if (type.getElementType().isF32()) {
          // Float grid: only f32/m1 is in scope for the compare-select family.
          if (lmul != "m1")
            return std::nullopt;
          return emitc::OpaqueType::get(type.getContext(), "vfloat32m1_t");
        }
        if (type.getElementType().isF64()) {
          // f64/m1 -> vfloat64m1_t: the SEW=64 double-precision elementwise
          // rung. Gated to m1 (the bounded coverage increment); other LMUL
          // widths stay unconverted so later families extend explicitly. This
          // rung is reachable only on a capability profile whose supported_sew
          // allow-list includes 64 (full-V); a SEW=32-capped (zve32*) profile
          // gates the SEW=64 body out before it reaches here.
          if (lmul != "m1")
            return std::nullopt;
          return emitc::OpaqueType::get(type.getContext(), "vfloat64m1_t");
        }
        // Unsigned low-precision rung: ui8/mf4, ui16/mf2, ui32/m1 ->
        // vuint<sew>m<lmul>_t (the legacy unsigned widening-product/reduce oracle
        // loads u8 sources into vuint8mf4_t, widens to vuint16mf2_t, and reduces
        // into vuint32m1_t). Only the grid those families use is in scope.
        if (isUnsignedVector(type)) {
          auto intType = llvm::cast<mlir::IntegerType>(type.getElementType());
          unsigned uSew = intType.getWidth();
          bool inScope = false;
          if (uSew == 8)
            // mf4 = the legacy unsigned widening-product first-slice load; m1 =
            // the codebook i8 gather anchor at VLEN128 (the packed-i4 weight loads
            // UNSIGNED for the vand/vsrl index split); mf2 = the codebook anchor at
            // VLEN256 (the m1/mf2 capability flip).
            inScope = lmul == "mf4" || lmul == "m1" || lmul == "mf2";
          else if (uSew == 16)
            inScope = lmul == "mf2" || lmul == "m1" || lmul == "m2";
          else if (uSew == 32)
            inScope = lmul == "m1" || lmul == "m2";
          else if (uSew == 64)
            inScope = lmul == "m1" || lmul == "m2";
          if (!inScope)
            return std::nullopt;
          std::string name = ("vuint" + llvm::Twine(uSew) + lmul + "_t").str();
          return emitc::OpaqueType::get(type.getContext(), name);
        }
        unsigned sew = 0;
        if (type.getElementType().isSignlessInteger(8))
          sew = 8;
        else if (type.getElementType().isSignlessInteger(16))
          sew = 16;
        else if (type.getElementType().isSignlessInteger(32))
          sew = 32;
        else if (type.getElementType().isSignlessInteger(64))
          sew = 64;
        else
          return std::nullopt;
        // The widening standalone-reduce source is a FRACTIONAL-LMUL i16
        // vector (i16/mf2 -> vint16mf2_t). The full-LMUL grid stays {m1,m2};
        // i16 also admits its fractional mf2 rung. The signed widening-product
        // / widening-dot contraction family adds the FRACTIONAL-LMUL i8 source
        // rung (i8/mf4 -> vint8mf4_t): the low-precision multiplicand loaded
        // before the vwmul widening to i16/mf2. Other (sew, lmul) pairs are
        // left unconverted on purpose so later families extend explicitly.
        // The deferred-wide max-legal-LMUL contraction (N3, the measured ssh-rvv
        // winner var_v_m2_a1.c) adds the wide rungs i8/m2 (strip load), i16/m4
        // (wide widening product), and i32/m8 (the loop-carried deferred vector
        // accumulator). These extend the in-scope grid for the deferred-wide
        // path; the narrow i8mf4/i16mf2/i32m1 rungs are unchanged.
        // The all-compiler LMUL-width ablation for the i16 dot-reduce family adds
        // the NARROW deferred rungs the budget knob selects: i16 source
        // {mf2,m1,m2,m4} -> i32 accumulator {m1,m2,m4,m8} (one EMUL step wider).
        // So the i32 grid spans {m1,m2,m4,m8} (the m4 accumulator is the budget-12
        // narrow rung) and the i16 grid spans {mf2,m1,m2,m4}.
        bool inScope = false;
        if (sew == 8)
          // mf4 = the narrow first-slice i8 load; m2 = the deferred-wide /
          // byte-anchor VLEN128 strip; m1 = the Track B byte-anchor VLEN256
          // strip (e8m1) -- the net-new rung for the capability flip; mf2 = the
          // codebook i8 gather anchor at VLEN256 (the plain-i8 activations + the
          // gathered codebook lanes, the m1/mf2 codebook flip).
          inScope = lmul == "mf4" || lmul == "m1" || lmul == "m2" ||
                    lmul == "mf2";
        else if (sew == 16)
          inScope = lmul == "mf2" || lmul == "m1" || lmul == "m2" ||
                    lmul == "m4";
        else if (sew == 32)
          inScope = lmul == "m1" || lmul == "m2" || lmul == "m4" ||
                    lmul == "m8";
        else
          inScope = lmul == "m1" || lmul == "m2";
        if (!inScope)
          return std::nullopt;
        std::string name = ("vint" + llvm::Twine(sew) + lmul + "_t").str();
        return emitc::OpaqueType::get(type.getContext(), name);
      });

  // !weft_rvv.index_vector<i<sew>, "m<lmul>"> ->
  // emitc.opaque<"vuint<sew>m<lmul>_t">. The bounded indexed gather/scatter
  // slice loads an UNSIGNED index/offset vector (the element offsets the
  // ordered indexed access scales to bytes), so the C type is the unsigned
  // vector form (i32/m1 -> vuint32m1_t) -- byte-identical to the legacy indexed
  // oracle's vuint32m1_t index vector. Only the m1 grid the slice uses is in
  // scope; other (sew, lmul) pairs stay unconverted so the converter is scoped.
  typeConverter.addConversion(
      [](weftrvv::IndexVectorType type) -> std::optional<mlir::Type> {
        llvm::StringRef lmul = type.getLmul();
        if (lmul != "m1")
          return std::nullopt;
        unsigned sew = 0;
        if (type.getElementType().isSignlessInteger(32))
          sew = 32;
        else
          return std::nullopt;
        std::string name = ("vuint" + llvm::Twine(sew) + lmul + "_t").str();
        return emitc::OpaqueType::get(type.getContext(), name);
      });

  // !weft_rvv.mask<i<sew>, "m<lmul>"> -> emitc.opaque<"vbool<maskbits>_t">. The
  // predicate mask C type is vbool<maskbits>_t where maskbits = SEW/LMUL_ratio
  // (the sew/lmul-derived width matching maskWidthForConfig): i32/m1 -> 32,
  // i64/m1 -> 64, i32/m2 -> 16, i64/m2 -> 32. Pairs maskWidthForConfig does not
  // know are left unconverted so the masked converter stays scoped to the grid.
  typeConverter.addConversion(
      [](weftrvv::MaskType type) -> std::optional<mlir::Type> {
        unsigned sew = 0;
        if (type.getElementType().isSignlessInteger(32) ||
            type.getElementType().isF32())
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

  // !weft_rvv.runtime_abi_value carries its concrete C type in the defining
  // op's c_type attribute (e.g. "const int32_t *", "int32_t *", "size_t"),
  // which a pure type-keyed conversion cannot recover. The VariantToEmitCFunc
  // pattern derives the function parameter type from that attr directly; the
  // token type itself maps to an opaque placeholder so it never blocks
  // legalization.
  typeConverter.addConversion(
      [](weftrvv::RuntimeABIValueType type) -> std::optional<mlir::Type> {
        return emitc::OpaqueType::get(type.getContext(), "void");
      });
}

void populateRVVElementwiseToEmitCPatterns(mlir::TypeConverter &typeConverter,
                                           mlir::RewritePatternSet &patterns) {
  patterns.add<detail::VariantToEmitCFunc>(typeConverter, patterns.getContext());
}

//===----------------------------------------------------------------------===//
// RVV typed-emission backend driver. This is the RVV implementation of the
// shared `TypedBackendEmissionDriver` seam: it supplies ONLY the RVV-specific
// pieces (type conversions, target legality, patterns, the post-conversion
// kernel drain, the RVV-body pre/post-check), while the generic harness
// `convertModuleWithBackendEmitter` owns the boilerplate. A future RVM family
// adds a sibling driver and registers it — no core edit. `convertRVVModuleToEmitC`
// stays the same entry point (pass + plugin probe still call it directly), now
// implemented by delegating to the shared harness with this driver.
//===----------------------------------------------------------------------===//

namespace {

class RVVBackendEmissionDriver final
    : public ::weft::conversion::emitc::TypedBackendEmissionDriver {
public:
  llvm::StringRef getBackendName() const override { return "rvv"; }

  void
  populateTypeConversions(mlir::TypeConverter &typeConverter) const override {
    populateRVVToEmitCTypeConversions(typeConverter);
  }

  void configureConversionTarget(mlir::ConversionTarget &target) const override {
    // A weft.exec.variant that carries a weft_rvv.with_vl selected-lowering
    // boundary is illegal and must be converted into an emitc.func. Variants
    // without a with_vl scope (e.g. scalar fallbacks) stay legal so unconverted
    // families fall through unchanged.
    target.addDynamicallyLegalOp<weft::exec::VariantOp>(
        [](weft::exec::VariantOp variant) {
          for (mlir::Operation &op : variant.getBody().front())
            if (llvm::isa<weft::rvv::WithVLOp>(op))
              return false;
          return true;
        });
    // Everything else (kernels, dispatch, capabilities, other dialects) stays
    // legal; the beachhead conversion rewrites the variant subtree atomically.
    target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
  }

  void
  populateLoweringPatterns(mlir::TypeConverter &typeConverter,
                           mlir::RewritePatternSet &patterns) const override {
    populateRVVElementwiseToEmitCPatterns(typeConverter, patterns);
  }

  llvm::LogicalResult postConversionCleanup(mlir::ModuleOp module) const override;

  bool moduleHasBackendBody(mlir::ModuleOp module) const override {
    // The module carries an RVV body if any op is a weft_rvv op OR still carries
    // a weft_rvv-typed operand/result (a half-converted op). This is both the
    // registry pre-check and the harness's per-backend "no RVV leftover" gate.
    auto isRVVType = [](mlir::Type type) {
      return type.getDialect().getNamespace() ==
             weftrvv::WEFTRVVDialect::getDialectNamespace();
    };
    bool hasRVV = false;
    module.walk([&](mlir::Operation *op) {
      if (op->getName().getDialectNamespace() ==
              weftrvv::WEFTRVVDialect::getDialectNamespace() ||
          llvm::any_of(op->getOperandTypes(), isRVVType) ||
          llvm::any_of(op->getResultTypes(), isRVVType)) {
        hasRVV = true;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    return hasRVV;
  }
};

llvm::LogicalResult
RVVBackendEmissionDriver::postConversionCleanup(mlir::ModuleOp module) const {
  // The beachhead conversion lowers the selected variant body into a
  // standalone emitc.func + headers. Once a function was produced, drop the
  // now-emptied weft.exec scaffolding (kernel/capability/dispatch) for that
  // kernel so the module is a clean, translatable EmitC module matching the
  // legacy materializer's output shape. Kernels that still carry a weft_rvv
  // body (unconverted families) are left untouched.
  bool producedFunc = false;
  module.walk([&](mlir::emitc::FuncOp) { producedFunc = true; });
  if (producedFunc) {
    llvm::SmallVector<weft::exec::KernelOp, 1> drainedKernels;
    module.walk([&](weft::exec::KernelOp kernel) {
      bool hasRVVBody = false;
      kernel.walk([&](weft::rvv::WithVLOp) { hasRVVBody = true; });
      if (!hasRVVBody)
        drainedKernels.push_back(kernel);
    });
    for (weft::exec::KernelOp kernel : drainedKernels)
      kernel.erase();

    // Module-level weft.exec capability/target scaffolding (e.g. a
    // `weft.exec.target @rvv_profile` provider declared at module scope and
    // referenced by the kernel's `target = @...`) is description-source IR the
    // legacy materializer discarded when it built its fresh emitc-only module.
    // Once the converted kernel(s) are drained it dangles, and a leftover
    // non-emitc top-level op makes the export handoff reject the module as
    // not-clean and fall back to the (now-retired) string route. Drop any
    // top-level weft.exec op that carries NO RVV body (the same drain criterion
    // used for kernels), so the materialized module is the clean emitc-only
    // shape the handoff expects. A top-level weft.exec op that still carries a
    // with_vl boundary (an unconverted family) is preserved so the
    // fullyConverted walk still reports a partial conversion.
    llvm::SmallVector<mlir::Operation *, 1> drainedExecOps;
    for (mlir::Operation &op : module.getBody()->getOperations()) {
      if (op.getName().getDialectNamespace() !=
          weft::exec::WEFTExecDialect::getDialectNamespace())
        continue;
      if (llvm::isa<weft::exec::KernelOp>(op))
        continue; // kernels handled above
      bool hasRVVBody = false;
      op.walk([&](weft::rvv::WithVLOp) { hasRVVBody = true; });
      if (!hasRVVBody)
        drainedExecOps.push_back(&op);
    }
    for (mlir::Operation *op : drainedExecOps)
      op->erase();

    // The materialized artifact is a STANDALONE emitc module (the legacy string
    // route built a fresh emitc-only module from scratch and discarded the
    // source IR). The source-front-door families (e.g. the bounded vector
    // source) leave a top-level non-RVV `func.func` source alongside the
    // converted kernel; the conversion correctly never touches it (it is not
    // RVV), but it must NOT ride along in the materialized emitc module — a
    // leftover non-emitc op makes the export handoff / `translateToCpp` reject
    // the module as not-clean and fall back to the (now-retired) legacy string
    // route. Drop the source body ops (anything that is neither an emitc op nor
    // a weft op) so the materialized module matches the legacy materializer's
    // clean emitc-only output. weft leftovers are deliberately preserved so the
    // fullyConverted walk below can still detect a genuinely partial conversion
    // and report a fall-back.
    llvm::SmallVector<mlir::Operation *, 2> drainedSourceOps;
    for (mlir::Operation &op : module.getBody()->getOperations()) {
      llvm::StringRef dialect = op.getName().getDialectNamespace();
      if (dialect != mlir::emitc::EmitCDialect::getDialectNamespace() &&
          dialect != weft::exec::WEFTExecDialect::getDialectNamespace() &&
          dialect != weftrvv::WEFTRVVDialect::getDialectNamespace())
        drainedSourceOps.push_back(&op);
    }
    for (mlir::Operation *op : drainedSourceOps)
      op->erase();
  }

  // The strangler-fig success gate (producedFunc + no RVV leftover op/type + no
  // unrealized_conversion_cast) is owned by the shared harness
  // (convertModuleWithBackendEmitter): `producedFunc` and the unrealized-cast
  // check are dialect-agnostic, and the RVV-leftover check is supplied by this
  // driver's `moduleHasBackendBody`. Cleanup itself never fails.
  return llvm::success();
}

} // namespace

void registerRVVBackendEmitter(
    ::weft::conversion::emitc::BackendEmissionRegistry &registry) {
  // Function-local static: owned by this translation unit, outlives the
  // registry, no global-init-order hazard.
  static const RVVBackendEmissionDriver driver;
  registry.registerBackend(driver);
}

bool convertRVVModuleToEmitC(mlir::ModuleOp module) {
  // The RVV->emitc conversion is now the shared `TypedBackendEmissionDriver`
  // harness parameterized by the RVV driver. The `--weft-rvv-lower-to-emitc`
  // pass and the plugin route probe still call this entry point directly (they
  // need the in-place gate, not the registry's clone-and-try). The behavior is
  // IDENTICAL to the pre-extraction monolithic driver.
  static const RVVBackendEmissionDriver driver;
  return ::weft::conversion::emitc::convertModuleWithBackendEmitter(module,
                                                                          driver);
}

} // namespace rvv
} // namespace conversion
} // namespace weft

namespace weft {
namespace transforms {

namespace {

// M-FLAT loop-scaffold step 4/6: the loop-aware allowlist RECURSIVE validator --
// the strong-form gate ([L-8], core-invariants I5/I7). It RECURSIVELY walks the
// weft_rvv.typed_flat_block_dot_loop_body region tree and asserts every op is a
// member of the forward-compatible typed-primitive allowlist (the bricks + the
// vector-core primitives + the scalar-extract bridge + the loop op/yield + the
// setvl/with_vl/load/store structural ops). It FAIL-CLOSES on any op outside the
// allowlist -- in particular an opaque emitc.call_opaque helper or a hand-written
// *_block_dot monolith leaking into the region at ANY nesting depth. This is the
// machine-checkable provenance basis a future q8_0 constructed(strong) flip
// depends on: it certifies the loop body is built entirely from pattern-library
// primitives with no sanctioned-opaque piece.
//
// It is deliberately the DUAL of the single-block rejectMixedPreRealizedContraction
// body BLOCKLIST (RVVEmitCContractionRouteFamilyInternal.h): that template rejects
// a variant that MIXES already-realized ops with the flat pre-realized op and does
// NOT recurse into any region; feeding the bricks into that blocklist would wrongly
// reject the brick BODY (the anti-write trap). This new allowlist walk is additive
// and never touches rejectMixed, so the 3 single-block strong paths are structurally
// unchanged.
// The shared forward-compatible typed-primitive allowlist membership test (the
// bricks + the vector-core primitives + the scalar-extract bridge + the loop
// ops/yields + the setvl/with_vl/load/store structural ops). Both the flat
// single-accumulator loop body and the q4_K/q5_K DUAL-accumulator super-block
// loop body share ONE union allowlist (the flat validator already unions all
// four flat folds' primitives regardless of which fold a given body uses; adding
// the q4_K super-block bricks keeps that union design -- it only ever makes the
// allowlist MORE permissive, never rejecting an op it used to accept, so the flat
// paths are structurally unchanged).
bool isTypedBlockDotLoopBodyAllowlistOp(mlir::Operation *op) {
  return llvm::isa<
      // brick 1 (per-block fp16 scale product / per-block-source) +
      // the Family-B (q4_1) per-block MIN/SUM correction product
      weft::rvv::BlockFp16ScaleProductOp, weft::rvv::BlockFp16MinProductOp,
      // vector integer core primitives (q8_0 plain signed widening
      // product + q4_0 asymmetric offset-binary packed-i4 x i8 product
      // + q4_1 asymmetric unsigned-nibble packed-i4 x i8 product)
      weft::rvv::WideningProductOp, weft::rvv::PackedI4OffsetBinaryXI8ProductOp,
      weft::rvv::UnsignedNibbleXI8ProductOp,
      // q5_0 five-bit offset-binary packed-i4 (+ qh 5th bit) x i8 product
      // + its per-block qh 32-bit-field source brick
      weft::rvv::FiveBitOffsetBinaryXI8ProductOp,
      weft::rvv::BlockFiveBitQhSourceOp, weft::rvv::StandaloneReduceOp,
      // iq4_nl / FP4 codebook class (2nd primitive class): the 16-entry non-linear
      // int8 lookup-table broadcast brick + the asymmetric codebook-gather packed-i4
      // x plain-i8 widening product brick (vrgather decode, NOT offset-binary)
      weft::rvv::CodebookTableBroadcastOp,
      weft::rvv::CodebookGatherXI8ProductOp,
      // step 2 scalar-lane extract bridge (integer core -> scalar fold)
      weft::rvv::TypedVectorLane0ToScalarExtractOp,
      // brick 2 (per-block computed-scale dequant term)
      weft::rvv::BlockComputedScaleDequantOp,
      // brick 3 (cross-block f32 loop-carried accumulate)
      weft::rvv::CrossBlockF32AccumulateOp,
      // W4: the q4_K/q5_K super-block bricks (nibble unpack, 6-bit scale/min
      // bit-dance, per-sub-block scaled i32 dot, MIN term, deferred positive
      // fold, post-loop horizontal fold)
      weft::rvv::Q4KNibbleUnpackOp, weft::rvv::Q4KScaleMinBitDanceOp,
      weft::rvv::Q4KScaledDotOp, weft::rvv::Q4KMinTermOp,
      weft::rvv::Q4KSumsFoldScaleDOp, weft::rvv::Q4KHorizontalFoldOp,
      // W-C (q6_K milestone-1): the q6_K no-min super-block INTEGER core (the
      // 2-bit qh + 8-bit signed scale unpack + per-sub-block i32 aux32 dot); it
      // is the single-accumulator body's integer brick (paired with the reused
      // q4_K Q4KSumsFoldScaleDOp for the no-min positive fold)
      weft::rvv::GgmlBlockDotQ6KQ8KAux32Op,
      // q3_K first flip: the q3_K no-min super-block INTEGER core (the 2-bit +
      // SUBTRACTIVE-hmask unpack + SIGNED 6-bit scale dance + per-sub-block i32
      // aux32 dot); the q3_K sibling of the q6_K aux32 brick, driving the SAME
      // single-accumulator no-min body (paired with the reused positive fold)
      weft::rvv::GgmlBlockDotQ3KQ8KAux32Op,
      // W-C' (q2_K milestone-1): the q2_K scalar super-block INTEGER core (the
      // 2-bit unpack + plain uint4-nibble scale/min + per-sub-block scalar i32
      // dot producing the two SCALAR states isum + summs); it is the
      // scalar-accumulator body's integer brick (the byte-exact scalar fold
      // sumf += dall*isum - dmin*summs is a later step)
      weft::rvv::GgmlBlockDotQ2KQ8KIntegerCoreOp,
      // iq1_s milestone-1 (first super-block GRID/codebook body): the iq1_s
      // scalar super-block TERNARY-grid INTEGER core (decode_model=lookup -- the
      // 11-bit grid index build from qs+qh, the vluxei16 ternary-grid gather, the
      // signed widening grid dot, the qh-encoded per-sub-block scale, the
      // delta-bsum sum, producing the two SCALAR states sumi + sumi1); it is the
      // scalar-accumulator body's grid brick under fold_model "scalar_delta_grid"
      // (the byte-exact scalar fold sumf += d*(sumi + IQ1S_DELTA*sumi1) is a
      // later step, milestone-2)
      weft::rvv::GgmlBlockDotIQ1SQ8KGridCoreOp,
      // iq1_m (iq1_s sibling): the iq1_m scalar super-block TERNARY-grid INTEGER
      // core (decode_model=lookup -- the SAME 2048-entry iq1s_grid, the packed
      // iq1m_scale fp16 reconstruct, the half-split per-half vluxei16 grid dot with
      // two half scales ls1/ls2, the per-group four-sign delta via a fresh Σq8,
      // producing the two SCALAR states sumi1 + sumi2); it is the scalar-accumulator
      // body's grid brick under fold_model "scalar_delta_grid" (disambiguated from
      // iq1_s by weight_block_stride 56 vs 50)
      weft::rvv::GgmlBlockDotIQ1MQ8KGridCoreOp,
      // iq3_xxs (iq1_s grid sibling): the iq3_xxs scalar super-block GRID-of-4
      // INTEGER core (decode_model=lookup -- the 256-entry uint32 iq3xxs_grid
      // gathered via vluxei16_v_i32m1, the aux32 4-bit-scale + 4-sign-group
      // ksigns decode, producing the ONE SCALAR state bsum); it is the
      // scalar-accumulator body's grid brick under fold_model "scalar_delta_grid"
      // (disambiguated from iq1_s/iq1_m by weight_block_stride 98 and the i32 grid)
      weft::rvv::GgmlBlockDotIQ3XXSQ8KGridCoreOp,
      // iq2_xxs (iq1_s grid sibling, SIGN-PLANE signs64 variant): the iq2_xxs scalar
      // super-block GRID-of-8 INTEGER core (decode_model=lookup -- the 256-entry uint64
      // iq2xxs_grid gathered via vluxei16_v_i64<core>, the SECOND vluxei16 gather over
      // the DERIVED keven_signs_q2xs signs64 sign plane, the aux1 4-bit-scale +
      // 4-sign-group decode, producing the ONE SCALAR state bsum); it is the
      // scalar-accumulator body's grid brick under fold_model "scalar_delta_grid"
      // (disambiguated from iq1_s/iq1_m/iq3_xxs by weight_block_stride 66; carries the
      // Win-A integer_core_lmul m2/m1 gearbox)
      weft::rvv::GgmlBlockDotIQ2XXSQ8KGridCoreOp,
      // iq2_xs (iq2_xxs grid sibling, SIGN-PLANE signs64 variant, PER-HALF explicit scale):
      // the iq2_xs scalar super-block per-half-scale GRID INTEGER core (decode_model=lookup
      // -- the 512-entry uint64 iq2xs_grid gathered via vluxei16_v_i64m1 indexed by
      // `w & 511`, the SECOND vluxei16 gather over the DERIVED keven_signs_q2xs signs64 sign
      // plane keyed by `w >> 9`, the EXPLICIT per-sub-block 4-bit scales[8] two-half split
      // ls1/ls2, producing the ONE SCALAR state bsum); it is the scalar-accumulator body's
      // grid brick under fold_model "scalar_delta_grid" (disambiguated from
      // iq1_s/iq1_m/iq3_xxs/iq2_xxs by weight_block_stride 74; NO gearbox -- fixed 16-lane
      // per-half shape)
      weft::rvv::GgmlBlockDotIQ2XSQ8KGridCoreOp,
      // iq2_s (iq2_xs grid sibling, SIGN-PLANE explicit-signs variant, PER-HALF explicit
      // scale): the iq2_s scalar super-block per-half-scale GRID INTEGER core
      // (decode_model=lookup -- the 1024-entry uint64 iq2s_grid gathered via
      // vluxei16_v_i64m1 indexed by `qs[l] | ((qh<<(8-2l))&0x300)`, the SECOND vluxei16
      // gather over the UNIVERSAL signs256 explicit-sign-byte plane keyed by the raw sign
      // byte, the EXPLICIT per-sub-block 4-bit scales[8] two-half split ls1/ls2, producing
      // the ONE SCALAR state bsum); it is the scalar-accumulator body's grid brick under
      // fold_model "scalar_delta_grid" (disambiguated from
      // iq1_s/iq1_m/iq3_xxs/iq2_xxs/iq2_xs by weight_block_stride 82; NO gearbox -- fixed
      // 16-lane per-half shape)
      weft::rvv::GgmlBlockDotIQ2SQ8KGridCoreOp,
      // iq3_s (iq3_xxs grid sibling, EXPLICIT-SIGNS variant, qh 9th-bit inject, explicit
      // two-nibble scale): the iq3_s scalar super-block GRID-of-4 INTEGER core
      // (decode_model=lookup -- the 512-entry uint32 iq3s_grid gathered via
      // vluxei16_v_i32m1 indexed by the qh-9th-bit-injected index `qs[l] |
      // ((qh<<(8-2l))&256)`, the EXPLICIT per-sub-block sign bytes read from the signs
      // region at offset 74 folded via the inline kmask {1<<j}, the EXPLICIT per-sub-block
      // two-nibble scales[4], producing the ONE SCALAR state bsum); it is the
      // scalar-accumulator body's grid brick under fold_model "scalar_delta_grid"
      // (disambiguated from iq1_s/iq1_m/iq3_xxs/iq2_xxs/iq2_xs/iq2_s by weight_block_stride
      // 110; NO gearbox -- fixed grid-of-4 shape, NO ksigns plane)
      weft::rvv::GgmlBlockDotIQ3SQ8KGridCoreOp,
      // iq4_xs (flat iq4_nl CODEBOOK sibling, SUPER-BLOCK rung): the iq4_xs scalar
      // super-block CODEBOOK INTEGER core (decode_model=lookup -- REUSES iq4_nl's
      // 16-entry non-linear int8 codebook gathered via vrgather_vv_i8m1, the per-sub-block
      // SIGNED 6-bit scale from scales_l[4]+scales_h biased -32 applied in the FLOAT
      // domain, producing the 8 per-sub-block sumi that fold into the scalar sumf); it is
      // the scalar-accumulator body's codebook brick under fold_model "scalar_delta_grid"
      // (disambiguated from the grid siblings by weight_block_stride 136; NO gearbox --
      // the codebook gather pins m1). UNLIKE the grid siblings the fold runs per-sub-block
      // in float, but the single-scalar accumulator arity is identical.
      weft::rvv::GgmlBlockDotIQ4XSQ8KCodebookCoreOp,
      // tq2_0 (the FIRST TQ-family member, ARITHMETIC 2-bit ternary decode): the tq2_0
      // FUSED 2-bit TERNARY super-block INTEGER core (decode_model=arithmetic -- q2_K's
      // 2-bit `(qs>>shift)&3` unpack over the 4 shifts {0,2,4,6}, the per-element `-1`
      // ternary bias vsub, the fused-plane vwmacc against q8 into a wide i16 accumulator,
      // ONE vwredsum per 32-byte chunk, producing the ONE SCALAR state sumi -- NO
      // grid/codebook gather); it is the scalar-accumulator body's ternary brick under
      // fold_model "scalar_delta_grid" (it SHARES weight_block_stride 66 with iq2_xxs but
      // the DISTINCT brick op type disambiguates; carries the Win-A integer_core_lmul
      // m2/m1 gearbox, kernel key "tq2_0")
      weft::rvv::GgmlBlockDotTQ20Q8KTernaryCoreOp,
      // tq1_0 (the SECOND TQ-family member, ARITHMETIC BASE-3 ternary decode): the tq1_0
      // BASE-3 TERNARY super-block INTEGER core (decode_model=arithmetic -- the qs main/tail
      // + qh base-3 trit unpack `q=(uint8_t)(byte*pow3[l]); xi=((uint16_t)q*3)>>8; xi-1` into
      // an element-ordered aux8[256], then the flat-256 widened i8*i8 dot -- vle8 i8 x q8 i8
      // -> vwmul i16 -> vwredsum i32 -- producing the ONE SCALAR state sumi; NO grid/codebook
      // gather); it is the scalar-accumulator body's ternary brick under fold_model
      // "scalar_delta_grid" (its weight_block_stride 54 is UNIQUE, so it dispatches by stride;
      // the DISTINCT base-3 brick op type also disambiguates; carries the Win-A
      // integer_core_lmul m2/m1 gearbox, kernel key "tq1_0"). REUSES the tq2_0 ternary
      // scaffold at C2 marginal cost.
      weft::rvv::GgmlBlockDotTQ10Q8KTernaryCoreOp,
      // q1_0 (the BINARY {-1,+1}-sign class, the LAST flat block-dot family
      // member, C_construct 26->27): the q1_0 BINARY-sign FLAT integer core
      // (decode_model=arithmetic -- the four q8_0 sub-blocks' vlm_v_b{ratio}
      // packed-bit sign mask loaded DIRECTLY as the i8 sign mask, i8-domain
      // vneg/vmerge -> ONE vwredsum i8->i16m1 per sub-block, plus the
      // emitter-inlined TWO-LEVEL fp32 fold `d0 * Σ_k(d1_k * sumi_block_k)`; NO
      // grid/codebook gather, NO nibble unpack); it is the flat loop body's
      // binary-sign brick under fold_model "flat_binary_two_level" (the FLAT
      // scaffold sibling of the super-block scalar cores -- q1_0's activation is a
      // FLAT block_q8_0 stream). Carries the Win-A integer_core_lmul m2/m1
      // gearbox, kernel key "q1_0".
      weft::rvv::GgmlBlockDotQ10Q80BinarySignCoreOp,
      // nvfp4 (the SECOND FP4-codebook class, NVIDIA's FP4): the nvfp4 FP4-CODEBOOK
      // FLAT integer core (decode_model=lookup -- REUSES mxfp4's 16-entry DOUBLED
      // e2m1 codebook gathered via vrgather_vv_i8m1, wrapped in the per-sub-block
      // UE4M3 fp8 weight scale + the two-q8_0-block/half addressing, producing the 4
      // per-sub-block sumi that fold into the scalar sumf); it is the flat loop
      // body's codebook brick under fold_model "flat_nvfp4_codebook" (the FLAT
      // scaffold sibling of q1_0 -- nvfp4's activation is a block_q8_0 stream). NO
      // gearbox -- the codebook gather pins m1.
      weft::rvv::GgmlBlockDotNVFP4Q80CodebookCoreOp,
      // structural VL / memory ops
      weft::rvv::SetVLOp, weft::rvv::WithVLOp, weft::rvv::LoadOp,
      weft::rvv::StoreOp,
      // M-FLAT REPACK (q4_0 16x1 GEVM): the per-block lane-wise nibble-dot
      // integer-core brick (seed i16 lo/hi + nibble-step vwmacc + lo/hi vwadd
      // combine) carried inside the repack GEVM loop body region
      weft::rvv::RepackLaneWiseQ4Q8DotOp,
      // M-FLAT REPACK (q4_0 16x1 GEMM, finale): the ONE-strip N-column integer
      // core brick + the per-column dual-fp16 scale FOLD brick carried inside the
      // repack GEMM loop body region (the runtime-strip x interleaved-column
      // transpose of the GEVM bricks)
      weft::rvv::RepackGemmLaneWiseQ4Q8DotOp,
      weft::rvv::RepackGemmDualFp16ScaleFoldOp,
      // the loop ops themselves + their terminators (forward-compatible): the
      // flat single-accumulator loop op, the q4_K DUAL-accumulator super-block
      // loop op, the q4_0 16x1-repacked GEVM per-strip vector-accumulator loop, and
      // the q4_0 16x1-repacked GEMM per-column vector-accumulator loop
      weft::rvv::TypedFlatBlockDotLoopBodyOp,
      weft::rvv::TypedFlatBlockDotLoopYieldOp,
      weft::rvv::TypedSuperBlockBlockDotLoopBodyOp,
      weft::rvv::TypedSuperBlockBlockDotLoopYieldOp,
      weft::rvv::TypedRepackGemvLoopBodyOp,
      weft::rvv::TypedRepackGemvLoopYieldOp,
      weft::rvv::TypedRepackGemmLoopBodyOp,
      weft::rvv::TypedRepackGemmLoopYieldOp,
      // M-FLAT forward-elementwise scaffold (line C, ① 之后): the typed
      // elementwise strip-loop op + its map core brick + terminator. They join
      // the SAME union allowlist (making it strictly MORE permissive, never
      // rejecting an op it used to accept, so every block-dot path is
      // structurally unchanged); the dedicated elementwise validator walk below
      // fires only on the elementwise loop op_kind.
      weft::rvv::TypedElementwiseLoopBodyOp,
      weft::rvv::TypedElementwiseLoopYieldOp,
      weft::rvv::ElementwiseScaleMapOp,
      // The SECOND forward-elementwise map core brick (silu), reusing the SAME
      // loop op + terminator + validator above; only the per-strip map primitive
      // is new. The union stays strictly MORE permissive (zero block-dot
      // regression).
      weft::rvv::ElementwiseSiluMapOp,
      // The FIRST forward-elementwise REDUCE core brick (rms_norm), reusing the
      // SAME loop op + terminator + validator, now under reduce_map_model
      // "reduce" (a loop-carried f64 accumulator). Its Σx² fold + rsqrt +
      // normalize are re-emitted by the loop op's reduce branch. The union stays
      // strictly MORE permissive (zero block-dot / map regression).
      weft::rvv::ElementwiseRmsNormReduceCoreOp,
      // The fused rms_norm->mul EPILOGUE consumer brick, carried inside the
      // rms_norm reduce core's optional $epilogue region. The reduce-body emitter
      // splices its per-lane vfmul_vv into the normalize strip (the register-kept
      // vy flows straight in; the intermediate normalized row never touches
      // memory). The union stays strictly MORE permissive (zero block-dot / map /
      // reduce regression).
      weft::rvv::ElementwiseMulMapOp,
      // The [FMT-PROP] fused-activation-quantize EPILOGUE consumer brick, carried
      // inside the mul_map's optional $quant_epilogue region. The reduce-body
      // emitter runs the per-block q8_0 amax/scale/narrow body on the register-kept
      // weighted vz (the f32 z[] intermediate never touches memory; the downstream
      // independent quantize_row pass is elided). The union stays strictly MORE
      // permissive (zero block-dot / map / reduce regression).
      weft::rvv::ElementwiseQuantizeQ80MapOp,
      // The SECOND forward-elementwise REDUCE core brick (soft_max), reusing the
      // SAME loop op + terminator + validator + reduce_map_model "reduce", EXCEPT
      // the loop-carried accumulator is the f64m1 WIDENING vector (vfwredusum Σe^x
      // fold) not a scalar double. Its fused exp-store-widening-reduce strip is
      // re-emitted by the soft_max return-carrying branch. The union stays
      // strictly MORE permissive (zero block-dot / map / rms_norm regression).
      weft::rvv::ElementwiseSoftMaxReduceCoreOp,
      // The FIRST forward-elementwise ROTATE core brick (rope), reusing the SAME
      // loop op + terminator + validator, now under reduce_map_model "rotate" (a
      // per-pair scalar loop with a loop-carried f32 theta recurrence). Its
      // position-dependent 2x2 rotation + scalar cos/sin angle seam + theta
      // recurrence are re-emitted by the loop op's rotate branch. The union stays
      // strictly MORE permissive (zero block-dot / map / reduce regression).
      weft::rvv::ElementwiseRopeRotateCoreOp,
      // The THREE forward SUPPORT-op MAP core bricks (the add/mul BINARY two-input
      // map, the cpy pass-through COPY map, and the gelu scalar tanh map), reusing
      // the SAME loop op + terminator + validator + reduce_map_model "map". Their
      // per-strip/per-element bodies are re-emitted by the loop op's map branch
      // via the SHARED byte-exact strip/loop helpers. The union stays strictly
      // MORE permissive (zero block-dot / map / reduce / rotate regression).
      weft::rvv::ElementwiseBinaryMapOp,
      weft::rvv::ElementwiseCopyMapOp,
      weft::rvv::ElementwiseGeluMapOp>(op);
}

// Shared recursive allowlist walk over a loop-body region: fail-close on any op
// outside the union allowlist (the strong-form [L-8] gate). `surfaceName` names
// the loop surface in the diagnostic.
llvm::LogicalResult validateLoopBodyAllowlist(mlir::Region &body,
                                              llvm::StringRef surfaceName) {
  mlir::WalkResult walk =
      body.walk([&](mlir::Operation *op) -> mlir::WalkResult {
        if (isTypedBlockDotLoopBodyAllowlistOp(op))
          return mlir::WalkResult::advance();
        op->emitOpError()
            << "is not in the M-FLAT typed " << surfaceName
            << " block-dot loop-body allowlist; the loop-aware allowlist "
               "validator (the strong-form [L-8] gate) recursively certifies "
               "every op in the loop region is a typed pattern-library primitive "
               "and fail-closes on any non-allowlist op (an opaque "
               "emitc.call_opaque or a hand-written *_block_dot helper must never "
               "leak into the constructed-strong body)";
        return mlir::WalkResult::interrupt();
      });
  return walk.wasInterrupted() ? mlir::failure() : mlir::success();
}

llvm::LogicalResult validateTypedFlatBlockDotLoopBodyAllowlist(
    weft::rvv::TypedFlatBlockDotLoopBodyOp loopBody) {
  return validateLoopBodyAllowlist(loopBody.getBody(), "flat");
}

// W4: the q4_K/q5_K DUAL-accumulator super-block loop body's [L-8] allowlist gate
// -- the SAME recursive strong-form certifier, wired to the super-block loop op.
llvm::LogicalResult validateTypedSuperBlockBlockDotLoopBodyAllowlist(
    weft::rvv::TypedSuperBlockBlockDotLoopBodyOp loopBody) {
  return validateLoopBodyAllowlist(loopBody.getBody(), "super-block");
}

// M-FLAT forward-elementwise scaffold: the typed elementwise strip-loop body's
// [L-8] allowlist gate -- the SAME recursive strong-form certifier, wired to the
// forward-pass (non-dot) elementwise loop op. It fail-closes on any op outside
// the union allowlist (an opaque emitc.call_opaque or a hand-written monolith
// forward helper leaking into the constructed body), the machine-checkable
// provenance basis for the scale constructed(strong) flip.
llvm::LogicalResult validateTypedElementwiseLoopBodyAllowlist(
    weft::rvv::TypedElementwiseLoopBodyOp loopBody) {
  return validateLoopBodyAllowlist(loopBody.getBody(), "elementwise");
}

class RVVLowerToEmitCPass final
    : public impl::RVVLowerToEmitCBase<RVVLowerToEmitCPass> {
public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // M-FLAT step 4/6: run the loop-aware allowlist recursive validator on every
    // typed flat block-dot loop body BEFORE the conversion driver. This is the
    // real, always-firing call site for the strong-form gate: the loop op flows
    // through THIS lowering pass (it never reaches the pre-realized realization
    // owner -- it is not a pre-realized cluster op and already lives inside a
    // with_vl), so the honest post-realization validation seam is here at the
    // pass boundary (the blueprint's "new walk / new post-realization check", not
    // an extension of an existing realization-owner call site). Running it before
    // the applyPartialConversion driver keeps the fail-closed diagnostic outside
    // the conversion's rolled-back attempt, so it reaches stderr reliably. The
    // walk only fires on the loop op_kind, so the single-block strong paths are
    // untouched (zero structural regression).
    bool allowlistRejected = false;
    module.walk([&](weft::rvv::TypedFlatBlockDotLoopBodyOp loopBody) {
      if (mlir::failed(validateTypedFlatBlockDotLoopBodyAllowlist(loopBody)))
        allowlistRejected = true;
    });
    // W4: the same strong-form [L-8] gate on the q4_K/q5_K DUAL-accumulator
    // super-block loop body. The walk only fires on the super-block op_kind, so
    // the flat + single-block strong paths are untouched (zero regression).
    module.walk([&](weft::rvv::TypedSuperBlockBlockDotLoopBodyOp loopBody) {
      if (mlir::failed(
              validateTypedSuperBlockBlockDotLoopBodyAllowlist(loopBody)))
        allowlistRejected = true;
    });
    // M-FLAT forward-elementwise scaffold: the SAME strong-form [L-8] gate on the
    // typed elementwise strip-loop body. The walk only fires on the elementwise
    // op_kind, so the flat + super-block + single-block strong paths are untouched
    // (zero regression).
    module.walk([&](weft::rvv::TypedElementwiseLoopBodyOp loopBody) {
      if (mlir::failed(validateTypedElementwiseLoopBodyAllowlist(loopBody)))
        allowlistRejected = true;
    });
    if (allowlistRejected) {
      signalPassFailure();
      return;
    }

    // Run the single shared conversion driver (the same one the live
    // artifact-export materialization seam calls). It runs the
    // TypeConverter/ConversionTarget/patterns + applyPartialConversion and
    // drains the emptied weft.exec scaffolding for converted kernels.
    if (conversion::rvv::convertRVVModuleToEmitC(module))
      return;

    // The driver returns false either for a clean structural no-op (an
    // unconverted family whose ops the target keeps legal and which the patterns
    // leave untouched) or for a real conversion failure that left an illegal
    // with_vl-carrying variant behind. Only the latter is a pass failure.
    bool unlegalizedScopeRemains = false;
    module.walk([&](weft::rvv::WithVLOp) {
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
} // namespace weft

