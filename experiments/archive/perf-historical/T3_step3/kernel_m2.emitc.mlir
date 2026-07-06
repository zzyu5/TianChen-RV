module {
  emitc.include <"stddef.h">
  emitc.include <"stdint.h">
  emitc.include <"riscv_vector.h">
  emitc.func @tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot(%arg0: !emitc.opaque<"size_t">, %arg1: !emitc.ptr<!emitc.opaque<"float">>, %arg2: !emitc.opaque<"size_t">, %arg3: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg4: !emitc.opaque<"size_t">, %arg5: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg6: !emitc.opaque<"size_t">, %arg7: !emitc.opaque<"int32_t">, %arg8: !emitc.ptr<!emitc.opaque<"const int32_t">>) attributes {specifiers = ["extern", "\22C\22"]} {
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2"
    %0 = call_opaque "__riscv_vsetvl_e8m2"(%arg0) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface"
    %1 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"float">>
    %2 = literal "0.0f" : !emitc.opaque<"float">
    assign %2 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count"
    %3 = literal "32" : !emitc.opaque<"size_t">
    %4 = div %arg0, %3 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    %5 = literal "1" : !emitc.opaque<"size_t">
    %6 = literal "0" : !emitc.opaque<"size_t">
    for %arg9 = %6 to %4 step %5  : !emitc.opaque<"size_t"> {
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x"
      %10 = literal "34" : !emitc.opaque<"size_t">
      %11 = mul %arg9, %10 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %12 = add %arg3, %11 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %13 = literal "34" : !emitc.opaque<"size_t">
      %14 = mul %arg9, %13 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %15 = add %arg5, %14 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %16 = call_opaque "(float)*(const _Float16 *)"(%12) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %17 = call_opaque "(float)*(const _Float16 *)"(%15) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %18 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %19 = literal "0" : !emitc.opaque<"int32_t">
      assign %19 : !emitc.opaque<"int32_t"> to %18 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2"
      %20 = literal "32" : !emitc.opaque<"size_t">
      %21 = call_opaque "__riscv_vsetvl_e8m2"(%20) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %22 = literal "0" : !emitc.opaque<"size_t">
      %23 = literal "2" : !emitc.opaque<"size_t">
      %24 = add %12, %23 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %25 = add %24, %22 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %26 = cast %25 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2"
      %27 = call_opaque "__riscv_vle8_v_i8m2"(%26, %21) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m2_t">
      %28 = literal "2" : !emitc.opaque<"size_t">
      %29 = add %15, %28 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %30 = add %29, %22 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %31 = cast %30 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2"
      %32 = call_opaque "__riscv_vle8_v_i8m2"(%31, %21) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4"
      %33 = call_opaque "__riscv_vwmul_vv_i16m4"(%27, %32, %21) : (!emitc.opaque<"vint8m2_t">, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m4_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %34 = literal "0" : !emitc.opaque<"int32_t">
      %35 = literal "1" : !emitc.opaque<"size_t">
      %36 = call_opaque "__riscv_vmv_v_x_i32m1"(%34, %35) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1"
      %37 = call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"(%33, %36, %21) : (!emitc.opaque<"vint16m4_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %38 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%37) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %38 : !emitc.opaque<"int32_t"> to %18 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %39 = load %18 : <!emitc.opaque<"int32_t">>
      %40 = load %1 : <!emitc.opaque<"float">>
      %41 = cast %39 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
      %42 = mul %41, %16 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
      %43 = mul %42, %17 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
      %44 = add %40, %43 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %44 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
    }
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s"
    %7 = literal "0" : index
    %8 = subscript %arg1[%7] : (!emitc.ptr<!emitc.opaque<"float">>, index) -> !emitc.lvalue<!emitc.opaque<"float">>
    %9 = load %1 : <!emitc.opaque<"float">>
    assign %9 : !emitc.opaque<"float"> to %8 : <!emitc.opaque<"float">>
    return
  }
}

