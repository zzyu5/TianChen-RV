#ifndef WEFT_DIALECT_SCALAR_IR_SCALAROPS_TD
#define WEFT_DIALECT_SCALAR_IR_SCALAROPS_TD

include "mlir/IR/OpBase.td"
include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.td"

def WEFT_Scalar_Dialect : Dialect {
  let name = "weft_scalar";
  let cppNamespace = "::weft::scalar";
  let summary = "Weft-RV scalar fallback extension dialect";
  let description = [{
    The scalar fallback dialect namespace is the portable, no-RISC-V-extension
    fallback extension family. The architectural family is weft.scalar; the
    concrete MLIR dialect namespace is weft_scalar because MLIR dialect
    namespaces cannot contain '.' characters.

    This dialect exposes ODS-backed compute role-op boundaries that the scalar
    backend emission driver lowers directly to a standalone, portable-scalar
    EmitC module (no __riscv_ intrinsics):

      * weft_scalar.compute_skeleton -- the TRIVIAL tracer-bullet boundary that
        proves the scalar family can flow a typed role op through the shared
        typed-emission harness and the emitc->C++ translate route.

      * weft_scalar.tq2_0_q8_k_vec_dot -- a REAL ternary 2-bit vec_dot kernel
        boundary: the tq2_0 (2-bit weights decoded to {-1,0,+1}) x q8_K
        activation dot with scalar int32 accumulation and a per-super-block
        fp16*fp32 scale fold, lowered to a pure-scalar nested-loop C kernel
        (int8xint8 MAC, NO XOR-popcount, NO vector intrinsics).

      * weft_scalar.dequantize_row_q4_0 -- a REAL 4-bit nibble dequantize kernel
        boundary: the ggml `dequantize_row_q4_0` block expansion, decoding each
        packed byte's low/high nibble to `(nibble - 8) * d` and scattering the
        two halves of the block into the float output row. It is the coverage
        fallback for the 4-bit weight format -- pure nibble arithmetic over
        plain C loops, NO XOR-popcount codebook, NO __riscv_ intrinsics.

    The dialect does not model target object export, runtime ABI provisioning,
    or performance.
  }];
}

class WEFTScalar_Op<string mnemonic, list<Trait> traits = []>
    : Op<WEFT_Scalar_Dialect, mnemonic, traits>;

def ComputeSkeletonOp : WEFTScalar_Op<"compute_skeleton", []> {
  let summary = "minimal portable-scalar source problem";
  let description = [{
    Records the minimal scalar extension-family compute role op in ODS. The op
    proves that the portable scalar fallback family can materialize a typed
    MLIR source problem from which family construction creates a typed final
    body. The source problem itself is not EmitC-lowerable.

    Family-local formula construction consumes `source_kernel`,
    `selected_variant`, and `scalar_immediate` and creates a distinct
    `immediate_call_body`. Artifact lowering accepts only that final typed
    body. This is a TRIVIAL tracer-bullet source problem: it does not model
    tensor/tile computation, runtime ABI values, generated objects,
    correctness, or performance.
  }];

  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    I64Attr:$scalar_immediate
  );

  let assemblyFormat = "attr-dict";
}

def TernaryQ2Q8BlockDotOp : WEFTScalar_Op<"tq2_0_q8_k_vec_dot", []> {
  let summary = "portable-scalar tq2_0 x q8_K source problem";
  let description = [{
    Records a REAL portable-scalar ternary 2-bit vec_dot compute role op: the
    ggml `ggml_vec_dot_tq2_0_q8_K` contraction expressed as a typed MLIR source
    problem. Each super-block holds QK_K ternary weights packed 2-bit (four
    weights per byte). The source records external geometry only; it is not an
    artifact-lowerable computation body.

    The typed attributes are formula inputs. Family-local construction
    qualifies the supported canonical layout and creates a distinct
    `packed_ternary_dot_body` containing all loop geometry, decode constants,
    strides, and offsets. Artifact lowering accepts only that body. This source
    problem models the numeric kernel only; it does not model
    target object export, runtime ABI values, or performance.
  }];

  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_d_byte_offset,
    I64Attr:$activation_d_byte_offset,
    I64Attr:$activation_quant_byte_offset
  );

  let assemblyFormat = "attr-dict";
}

def DequantizeRowQ4Op : WEFTScalar_Op<"dequantize_row_q4_0", []> {
  let summary = "portable-scalar q4_0 dequantize-row source problem";
  let description = [{
    Records a REAL portable-scalar 4-bit dequantize compute role op: the ggml
    `dequantize_row_q4_0` block expansion expressed as a typed MLIR source
    problem.
    Each block holds `qk` weights packed 4-bit (two nibbles per byte) after a
    leading fp16 block scale. The source records external block geometry only;
    it is not an artifact-lowerable computation body.

    The typed attributes are formula inputs. Family-local construction
    qualifies the supported canonical layout and creates a distinct
    `packed_affine_dequant_body` containing the block geometry, decode
    constants, strides, and offsets. Artifact lowering accepts only that body.
    This source problem models the numeric kernel only; it does
    not model target object export, runtime ABI values, or performance.
  }];

  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$weight_d_byte_offset,
    I64Attr:$weight_quant_byte_offset
  );

  let assemblyFormat = "attr-dict";
}

def ImmediateCallBodyOp : WEFTScalar_Op<"immediate_call_body",
    [DeclareOpInterfaceMethods<WEFTEmitCLowerableOpInterface,
      ["getWEFTEmitCLowerableSourceOpName",
       "getWEFTEmitCLowerableSourceRole"]>]> {
  let summary = "constructed scalar immediate-call body";
  let description = [{
    Final typed body constructed from compute_skeleton. The artifact lowerer
    consumes the exact operation and projects the fixed
    `i32 weft_scalar_compute_skeleton(i32)` call ABI plus the typed immediate.
    The body identity owns that call topology; no source op or intermediate
    provider participates in lowering.
  }];
  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    I64Attr:$scalar_immediate
  );
  let assemblyFormat = "attr-dict";
}

// These plan operations are deliberately Scalar-family-local and
// topology-specific. They are not a generic loop/dataflow IR: formula
// construction creates the exact closed tree below, and the Scalar artifact
// driver projects each typed node mechanically.

def TernaryBlockLoopOp : WEFTScalar_Op<"ternary_block_loop",
    [NoRegionArguments, NoTerminator, SingleBlock]> {
  let summary = "tq2 block traversal in the final Scalar computation plan";
  let description = [{
    Owns the outer runtime `n / qk` loop and the per-block weight/activation
    base computation for the packed ternary dot. The nested plan owns the
    group/plane/lane traversal and fold; an artifact driver must not recreate
    either from a source format name.
  }];
  let arguments = (ins
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$activation_quant_byte_offset,
    I64Attr:$step
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}

def TernaryPlaneGroupLoopOp : WEFTScalar_Op<"ternary_plane_group_loop",
    [NoRegionArguments, NoTerminator, SingleBlock]> {
  let summary = "packed-byte group traversal in the ternary dot plan";
  let arguments = (ins I64Attr:$upper_bound, I64Attr:$step);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}

def TernaryPlaneLoopOp : WEFTScalar_Op<"ternary_plane_loop",
    [NoRegionArguments, NoTerminator, SingleBlock]> {
  let summary = "2-bit field-plane traversal in the ternary dot plan";
  let arguments = (ins
    I64Attr:$upper_bound,
    I64Attr:$step,
    I64Attr:$field_bits
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}

def TernaryLaneLoopOp : WEFTScalar_Op<"ternary_lane_loop",
    [NoRegionArguments, NoTerminator, SingleBlock]> {
  let summary = "lane traversal in the ternary dot plan";
  let arguments = (ins I64Attr:$upper_bound, I64Attr:$step);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}

def TernaryDecodeMacOp : WEFTScalar_Op<"ternary_decode_mac", []> {
  let summary = "typed shift-mask-affine ternary decode and signed i32 MAC";
  let description = [{
    Exact innermost computation node for the packed ternary dot. It loads the
    packed weight byte at `group + lane`, shifts by the enclosing plane's
    `plane * field_bits`, masks and subtracts `decode_zero_point`, loads the
    activation at
    `group * activation_plane_stride + plane * plane_lanes + lane`, and adds
    their signed i32 product to the block accumulator.
  }];
  let arguments = (ins
    I64Attr:$field_mask,
    I64Attr:$decode_zero_point,
    I64Attr:$activation_plane_stride,
    I64Attr:$plane_lanes
  );
  let assemblyFormat = "attr-dict";
}

def TernaryScaleFoldOp : WEFTScalar_Op<"ternary_scale_fold", []> {
  let summary = "typed fp16/f32 scale load and block-dot f32 fold";
  let description = [{
    Exact post-inner-loop fold node. It reads the activation scale as f32 and
    the weight scale as fp16 at the typed byte offsets, combines them as
    `activation_scale * weight_scale`, and performs
    `sum + (float)i32_accumulator * combined_scale`.
  }];
  let arguments = (ins
    I64Attr:$weight_d_byte_offset,
    I64Attr:$activation_d_byte_offset
  );
  let assemblyFormat = "attr-dict";
}

def TernaryStoreOp : WEFTScalar_Op<"ternary_store", []> {
  let summary = "store the final ternary-dot f32 sum to output element zero";
  let assemblyFormat = "attr-dict";
}

def PackedTernaryDotBodyOp : WEFTScalar_Op<"packed_ternary_dot_body",
    [NoRegionArguments, NoTerminator, SingleBlock,
     DeclareOpInterfaceMethods<WEFTEmitCLowerableOpInterface,
      ["getWEFTEmitCLowerableSourceOpName",
       "getWEFTEmitCLowerableSourceRole"]>]> {
  let summary = "constructed packed ternary scalar dot body";
  let description = [{
    Final family-local computation body for the tq2_0 x q8_K topology. Its
    nested Scalar-local typed plan contains the block/group/plane/lane loop
    tree, decode-MAC node, scale-fold node, and final store node. Every bound,
    stride, offset, and decode constant lives on the node that consumes it.
    The external ABI is fixed to
    `(int n, float *s, const uint8_t *weights, const int8_t *activations)`.
    Artifact lowering walks this exact plan and does not read or reinterpret
    the tq2_0 source problem.
  }];
  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}

def AffineBlockLoopOp : WEFTScalar_Op<"affine_block_loop",
    [NoRegionArguments, NoTerminator, SingleBlock]> {
  let summary = "q4_0 block traversal in the final Scalar computation plan";
  let arguments = (ins
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$step
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}

def AffineBlockScaleOp : WEFTScalar_Op<"affine_block_scale", []> {
  let summary = "load one q4_0 block scale as fp16 and convert it to f32";
  let arguments = (ins I64Attr:$weight_d_byte_offset);
  let assemblyFormat = "attr-dict";
}

def AffineQuantBaseOp : WEFTScalar_Op<"affine_quant_base", []> {
  let summary = "form the packed q4_0 quant-byte base";
  let arguments = (ins I64Attr:$weight_quant_byte_offset);
  let assemblyFormat = "attr-dict";
}

def AffinePackedByteLoopOp : WEFTScalar_Op<"affine_packed_byte_loop",
    [NoRegionArguments, NoTerminator, SingleBlock]> {
  let summary = "packed-byte traversal in the q4_0 dequantization plan";
  let arguments = (ins I64Attr:$upper_bound, I64Attr:$step);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}

def AffineDecodeScaleScatterOp
    : WEFTScalar_Op<"affine_decode_scale_scatter", []> {
  let summary = "typed paired-nibble decode, scale, and scatter";
  let description = [{
    Exact q4_0 inner computation node. It decodes the low and high unsigned
    fields with the typed shifts/mask/zero point, converts both signed results
    to f32, multiplies each by the enclosing block scale, and stores them at
    `block*qk + packed_index + low_output_delta` and
    `block*qk + packed_index + high_output_delta`.
  }];
  let arguments = (ins
    I64Attr:$low_field_shift,
    I64Attr:$high_field_shift,
    I64Attr:$field_mask,
    I64Attr:$decode_zero_point,
    I64Attr:$low_output_delta,
    I64Attr:$high_output_delta
  );
  let assemblyFormat = "attr-dict";
}

def PackedAffineDequantBodyOp : WEFTScalar_Op<"packed_affine_dequant_body",
    [NoRegionArguments, NoTerminator, SingleBlock,
     DeclareOpInterfaceMethods<WEFTEmitCLowerableOpInterface,
      ["getWEFTEmitCLowerableSourceOpName",
       "getWEFTEmitCLowerableSourceRole"]>]> {
  let summary = "constructed packed affine scalar dequant body";
  let description = [{
    Final family-local computation body for q4_0 paired-nibble dequantization.
    Its nested Scalar-local typed plan contains the block loop, scale load,
    packed-byte loop, paired decode/rescale/scatter node, and every consumed
    bound/stride/offset/decode constant. The external ABI is fixed to
    `(int n, float *output, const uint8_t *weights)`. Artifact lowering walks
    that exact plan and does not read or reinterpret the q4_0 source problem.
  }];
  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}

#endif // WEFT_DIALECT_SCALAR_IR_SCALAROPS_TD
