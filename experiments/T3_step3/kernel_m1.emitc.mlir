module {
  emitc.include <"stddef.h">
  emitc.include <"stdint.h">
  emitc.include <"riscv_vector.h">
  emitc.func @tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot(%arg0: !emitc.opaque<"size_t">, %arg1: !emitc.ptr<!emitc.opaque<"float">>, %arg2: !emitc.opaque<"size_t">, %arg3: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg4: !emitc.opaque<"size_t">, %arg5: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg6: !emitc.opaque<"size_t">, %arg7: !emitc.opaque<"int32_t">, %arg8: !emitc.ptr<!emitc.opaque<"const int32_t">>) attributes {specifiers = ["extern", "\22C\22"]} {
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
    %0 = call_opaque "__riscv_vsetvl_e8m1"(%arg0) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
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
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %20 = literal "32" : !emitc.opaque<"size_t">
      %21 = call_opaque "__riscv_vsetvl_e8m1"(%20) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %22 = literal "32" : !emitc.opaque<"size_t">
      %23 = literal "0" : !emitc.opaque<"size_t">
      for %arg10 = %23 to %22 step %21  : !emitc.opaque<"size_t"> {
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
        %30 = literal "32" : !emitc.opaque<"size_t">
        %31 = sub %30, %arg10 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %32 = call_opaque "__riscv_vsetvl_e8m1"(%31) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %33 = literal "2" : !emitc.opaque<"size_t">
        %34 = add %12, %33 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %35 = add %34, %arg10 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %36 = cast %35 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
        %37 = call_opaque "__riscv_vle8_v_i8m1"(%36, %32) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        %38 = literal "2" : !emitc.opaque<"size_t">
        %39 = add %15, %38 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %40 = add %39, %arg10 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %41 = cast %40 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
        %42 = call_opaque "__riscv_vle8_v_i8m1"(%41, %32) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
        %43 = call_opaque "__riscv_vwmul_vv_i16m2"(%37, %42, %32) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
        %44 = load %18 : <!emitc.opaque<"int32_t">>
        %45 = literal "1" : !emitc.opaque<"size_t">
        %46 = call_opaque "__riscv_vmv_v_x_i32m1"(%44, %45) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
        %47 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%43, %46, %32) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
        %48 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%47) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
        verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface"
        assign %48 : !emitc.opaque<"int32_t"> to %18 : <!emitc.opaque<"int32_t">>
      }
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %24 = load %18 : <!emitc.opaque<"int32_t">>
      %25 = load %1 : <!emitc.opaque<"float">>
      %26 = cast %24 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
      %27 = mul %26, %16 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
      %28 = mul %27, %17 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
      %29 = add %25, %28 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %29 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
    }
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s"
    %7 = literal "0" : index
    %8 = subscript %arg1[%7] : (!emitc.ptr<!emitc.opaque<"float">>, index) -> !emitc.lvalue<!emitc.opaque<"float">>
    %9 = load %1 : <!emitc.opaque<"float">>
    assign %9 : !emitc.opaque<"float"> to %8 : <!emitc.opaque<"float">>
    return
  }
}

