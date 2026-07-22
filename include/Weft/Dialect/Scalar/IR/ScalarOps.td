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

def ComputeSkeletonOp : WEFTScalar_Op<"compute_skeleton",
    [DeclareOpInterfaceMethods<WEFTEmitCLowerableOpInterface>]> {
  let summary = "minimal portable-scalar ODS compute role-op boundary";
  let description = [{
    Records the minimal scalar extension-family compute role op in ODS. The op
    proves that the portable scalar fallback family can materialize a typed
    MLIR role operation implementing the generated EmitC-lowerable provenance
    interface and have the scalar backend emission driver consume that IR
    identity to emit a pure-scalar (no __riscv_) EmitC module.

    Family-local formula construction consumes `source_kernel`,
    `selected_variant`, and `scalar_immediate` and records the complete
    conversion-local final plan (exported symbol, callee, and immediate). The
    emitter only materializes that plan; it does not reinterpret the source
    attributes. This is a TRIVIAL tracer-bullet boundary: it does not model
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

def TernaryQ2Q8BlockDotOp : WEFTScalar_Op<"tq2_0_q8_k_vec_dot",
    [DeclareOpInterfaceMethods<WEFTEmitCLowerableOpInterface>]> {
  let summary = "portable-scalar tq2_0 x q8_K ternary 2-bit vec_dot boundary";
  let description = [{
    Records a REAL portable-scalar ternary 2-bit vec_dot compute role op: the
    ggml `ggml_vec_dot_tq2_0_q8_K` contraction expressed as a typed MLIR
    boundary. Each super-block holds QK_K ternary weights packed 2-bit
    (four weights per byte); the scalar backend emission driver lowers this op
    to a standalone pure-scalar nested-loop C kernel that, for every
    super-block, decodes each 2-bit field to the signed ternary value
    `((qs >> shift) & 3) - 1` in {-1,0,+1(,2)}, multiply-accumulates it against
    the matching int8 q8_K activation into a scalar int32 `sumi`, then folds
    `sumf += (float)sumi * (y.d * fp16(x.d))`. It is a pure int8xint8 MAC over
    plain C loops -- NO XOR-popcount codebook, NO __riscv_ intrinsics, NO
    vector machinery.

    The typed attributes are formula inputs. Family-local construction
    qualifies the supported canonical layout and records a complete
    conversion-local final computation plan, including all loop geometry,
    decode constants, strides, and offsets. The emitter consumes that plan
    mechanically; it neither derives geometry nor accepts an unqualified
    layout. This boundary models the numeric kernel only; it does not model
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

def DequantizeRowQ4Op : WEFTScalar_Op<"dequantize_row_q4_0",
    [DeclareOpInterfaceMethods<WEFTEmitCLowerableOpInterface>]> {
  let summary = "portable-scalar q4_0 4-bit nibble dequantize_row boundary";
  let description = [{
    Records a REAL portable-scalar 4-bit dequantize compute role op: the ggml
    `dequantize_row_q4_0` block expansion expressed as a typed MLIR boundary.
    Each block holds `qk` weights packed 4-bit (two nibbles per byte) after a
    leading fp16 block scale; the scalar backend emission driver lowers this op
    to a standalone pure-scalar nested-loop C kernel that, for every block,
    reads the fp16 scale `d`, decodes each byte's low nibble `(qs[j] & 0x0F) - 8`
    and high nibble `(qs[j] >> 4) - 8`, and scatters `x0*d` and `x1*d` into the
    two halves of the output row (`y[j]` and `y[j + qk/2]`), exactly the ggml
    reference layout. It is pure nibble arithmetic over plain C loops -- NO
    XOR-popcount codebook, NO __riscv_ intrinsics, NO vector machinery.

    The typed attributes are formula inputs. Family-local construction
    qualifies the supported canonical layout and records a complete
    conversion-local final computation plan, including the block geometry,
    nibble decode constants, strides, and offsets. The emitter consumes that
    plan mechanically; it neither reconstructs `qk/2` nor accepts an
    unqualified layout. This boundary models the numeric kernel only; it does
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

#endif // WEFT_DIALECT_SCALAR_IR_SCALAROPS_TD
