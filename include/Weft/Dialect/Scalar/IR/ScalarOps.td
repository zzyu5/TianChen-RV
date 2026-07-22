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
    The body identity owns that call topology; no source-op or route provider
    participates in lowering.
  }];
  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    I64Attr:$scalar_immediate
  );
  let assemblyFormat = "attr-dict";
}

def PackedTernaryDotBodyOp : WEFTScalar_Op<"packed_ternary_dot_body",
    [DeclareOpInterfaceMethods<WEFTEmitCLowerableOpInterface,
      ["getWEFTEmitCLowerableSourceOpName",
       "getWEFTEmitCLowerableSourceRole"]>]> {
  let summary = "constructed packed ternary scalar dot body";
  let description = [{
    Final family-local computation body for the tq2_0 x q8_K topology. Its
    semantics are a block loop over `n/qk`; within each block it traverses
    packed weight groups, 2-bit planes, and lanes in that order, decodes
    `((weight >> (plane*field_bits)) & field_mask) - decode_zero_point`, maps
    the activation index as
    `group*activation_plane_stride + plane*plane_lanes + lane`, accumulates in
    signed i32, loads the weight scale as fp16 and activation scale as f32,
    folds `(float)sumi * (activation_scale * weight_scale)` into an f32 sum,
    and stores that sum at output element zero. The external ABI is fixed to
    `(int n, float *s, const uint8_t *weights, const int8_t *activations)`.
    Attributes contain every variable bound/stride/offset/decode constant;
    operation identity contains the fixed loop topology, types, maps, fold
    association, and store semantics. Artifact lowering does not read or
    reinterpret the tq2_0 source problem.
  }];
  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_d_byte_offset,
    I64Attr:$activation_d_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    I64Attr:$packed_weight_bytes,
    I64Attr:$planes,
    I64Attr:$plane_lanes,
    I64Attr:$plane_group_stride,
    I64Attr:$field_bits,
    I64Attr:$field_mask,
    I64Attr:$decode_zero_point,
    I64Attr:$activation_plane_stride
  );
  let assemblyFormat = "attr-dict";
}

def PackedAffineDequantBodyOp : WEFTScalar_Op<"packed_affine_dequant_body",
    [DeclareOpInterfaceMethods<WEFTEmitCLowerableOpInterface,
      ["getWEFTEmitCLowerableSourceOpName",
       "getWEFTEmitCLowerableSourceRole"]>]> {
  let summary = "constructed packed affine scalar dequant body";
  let description = [{
    Final family-local computation body for q4_0 paired-nibble dequantization.
    Its semantics are a block loop over `n/qk`; each block loads one fp16 scale,
    visits `half_width` packed bytes, decodes the low field and high field using
    `field_mask`, `field_bits`, and `decode_zero_point`, converts each signed
    value to f32, multiplies by the block scale, and scatters the pair to
    `block*qk+j` and `block*qk+j+half_width`. The external ABI is fixed to
    `(int n, float *output, const uint8_t *weights)`. Attributes contain every
    variable bound/stride/offset/decode constant; operation identity contains
    the fixed loop topology, types, paired index map, and stores. Artifact
    lowering does not read or reinterpret the q4_0 source problem.
  }];
  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$weight_d_byte_offset,
    I64Attr:$weight_quant_byte_offset,
    I64Attr:$half_width,
    I64Attr:$field_bits,
    I64Attr:$field_mask,
    I64Attr:$decode_zero_point
  );
  let assemblyFormat = "attr-dict";
}

#endif // WEFT_DIALECT_SCALAR_IR_SCALAROPS_TD
