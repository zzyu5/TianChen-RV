module {
  emitc.include <"stddef.h">
  emitc.include <"stdint.h">
  emitc.include <"riscv_vector.h">
  emitc.func @tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(%arg0: !emitc.opaque<"size_t">, %arg1: !emitc.ptr<!emitc.opaque<"float">>, %arg2: !emitc.opaque<"size_t">, %arg3: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg4: !emitc.opaque<"size_t">, %arg5: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg6: !emitc.opaque<"size_t">, %arg7: !emitc.opaque<"int32_t">) attributes {specifiers = ["extern", "\22C\22"]} {
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1"
    %0 = call_opaque "__riscv_vsetvl_e32m1"(%arg0) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
    %1 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"float">>
    %2 = literal "0.0f" : !emitc.opaque<"float">
    assign %2 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count"
    %3 = literal "32" : !emitc.opaque<"size_t">
    %4 = div %arg0, %3 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    %5 = literal "4" : !emitc.opaque<"size_t">
    %6 = rem %4, %5 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    %7 = sub %4, %6 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    %8 = literal "0" : !emitc.opaque<"size_t">
    for %arg8 = %8 to %7 step %5  : !emitc.opaque<"size_t"> {
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x"
      %13 = literal "18" : !emitc.opaque<"size_t">
      %14 = mul %arg8, %13 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %15 = add %arg3, %14 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %16 = literal "34" : !emitc.opaque<"size_t">
      %17 = mul %arg8, %16 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %18 = add %arg5, %17 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %19 = call_opaque "(float)*(const _Float16 *)"(%15) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %20 = call_opaque "(float)*(const _Float16 *)"(%18) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %21 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %22 = literal "0" : !emitc.opaque<"int32_t">
      assign %22 : !emitc.opaque<"int32_t"> to %21 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %23 = literal "16" : !emitc.opaque<"size_t">
      %24 = call_opaque "__riscv_vsetvl_e8m1"(%23) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %25 = literal "0" : !emitc.opaque<"size_t">
      %26 = literal "2" : !emitc.opaque<"size_t">
      %27 = add %15, %26 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %28 = add %27, %25 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %29 = cast %28 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %30 = call_opaque "__riscv_vle8_v_i8m1"(%29, %24) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %31 = literal "2" : !emitc.opaque<"size_t">
      %32 = add %18, %31 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %33 = add %32, %25 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %34 = cast %33 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %35 = call_opaque "__riscv_vle8_v_i8m1"(%34, %24) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %36 = literal "18" : !emitc.opaque<"size_t">
      %37 = add %18, %36 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %38 = add %37, %25 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %39 = cast %38 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %40 = call_opaque "__riscv_vle8_v_i8m1"(%39, %24) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vxor_vx_i8m1"
      %41 = literal "0x88" : !emitc.opaque<"int">
      %42 = call_opaque "__riscv_vxor_vx_i8m1"(%30, %41, %24) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1"
      %43 = literal "4" : !emitc.opaque<"uint8_t">
      %44 = call_opaque "__riscv_vsll_vx_i8m1"(%42, %43, %24) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1"
      %45 = literal "4" : !emitc.opaque<"uint8_t">
      %46 = call_opaque "__riscv_vsra_vx_i8m1"(%44, %45, %24) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1"
      %47 = literal "4" : !emitc.opaque<"uint8_t">
      %48 = call_opaque "__riscv_vsra_vx_i8m1"(%42, %47, %24) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %49 = call_opaque "__riscv_vwmul_vv_i16m2"(%46, %35, %24) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2"
      %50 = call_opaque "__riscv_vwmacc_vv_i16m2"(%49, %48, %40, %24) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %51 = literal "0" : !emitc.opaque<"int32_t">
      %52 = literal "1" : !emitc.opaque<"size_t">
      %53 = call_opaque "__riscv_vmv_v_x_i32m1"(%51, %52) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %54 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%50, %53, %24) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %55 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%54) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %55 : !emitc.opaque<"int32_t"> to %21 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x"
      %56 = literal "1" : !emitc.opaque<"size_t">
      %57 = add %arg8, %56 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %58 = literal "18" : !emitc.opaque<"size_t">
      %59 = mul %57, %58 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %60 = add %arg3, %59 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %61 = literal "1" : !emitc.opaque<"size_t">
      %62 = add %arg8, %61 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %63 = literal "34" : !emitc.opaque<"size_t">
      %64 = mul %62, %63 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %65 = add %arg5, %64 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %66 = call_opaque "(float)*(const _Float16 *)"(%60) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %67 = call_opaque "(float)*(const _Float16 *)"(%65) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %68 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %69 = literal "0" : !emitc.opaque<"int32_t">
      assign %69 : !emitc.opaque<"int32_t"> to %68 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %70 = literal "16" : !emitc.opaque<"size_t">
      %71 = call_opaque "__riscv_vsetvl_e8m1"(%70) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %72 = literal "0" : !emitc.opaque<"size_t">
      %73 = literal "2" : !emitc.opaque<"size_t">
      %74 = add %60, %73 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %75 = add %74, %72 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %76 = cast %75 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %77 = call_opaque "__riscv_vle8_v_i8m1"(%76, %71) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %78 = literal "2" : !emitc.opaque<"size_t">
      %79 = add %65, %78 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %80 = add %79, %72 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %81 = cast %80 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %82 = call_opaque "__riscv_vle8_v_i8m1"(%81, %71) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %83 = literal "18" : !emitc.opaque<"size_t">
      %84 = add %65, %83 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %85 = add %84, %72 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %86 = cast %85 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %87 = call_opaque "__riscv_vle8_v_i8m1"(%86, %71) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vxor_vx_i8m1"
      %88 = literal "0x88" : !emitc.opaque<"int">
      %89 = call_opaque "__riscv_vxor_vx_i8m1"(%77, %88, %71) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1"
      %90 = literal "4" : !emitc.opaque<"uint8_t">
      %91 = call_opaque "__riscv_vsll_vx_i8m1"(%89, %90, %71) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1"
      %92 = literal "4" : !emitc.opaque<"uint8_t">
      %93 = call_opaque "__riscv_vsra_vx_i8m1"(%91, %92, %71) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1"
      %94 = literal "4" : !emitc.opaque<"uint8_t">
      %95 = call_opaque "__riscv_vsra_vx_i8m1"(%89, %94, %71) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %96 = call_opaque "__riscv_vwmul_vv_i16m2"(%93, %82, %71) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2"
      %97 = call_opaque "__riscv_vwmacc_vv_i16m2"(%96, %95, %87, %71) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %98 = literal "0" : !emitc.opaque<"int32_t">
      %99 = literal "1" : !emitc.opaque<"size_t">
      %100 = call_opaque "__riscv_vmv_v_x_i32m1"(%98, %99) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %101 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%97, %100, %71) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %102 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%101) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %102 : !emitc.opaque<"int32_t"> to %68 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x"
      %103 = literal "2" : !emitc.opaque<"size_t">
      %104 = add %arg8, %103 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %105 = literal "18" : !emitc.opaque<"size_t">
      %106 = mul %104, %105 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %107 = add %arg3, %106 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %108 = literal "2" : !emitc.opaque<"size_t">
      %109 = add %arg8, %108 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %110 = literal "34" : !emitc.opaque<"size_t">
      %111 = mul %109, %110 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %112 = add %arg5, %111 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %113 = call_opaque "(float)*(const _Float16 *)"(%107) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %114 = call_opaque "(float)*(const _Float16 *)"(%112) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %115 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %116 = literal "0" : !emitc.opaque<"int32_t">
      assign %116 : !emitc.opaque<"int32_t"> to %115 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %117 = literal "16" : !emitc.opaque<"size_t">
      %118 = call_opaque "__riscv_vsetvl_e8m1"(%117) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %119 = literal "0" : !emitc.opaque<"size_t">
      %120 = literal "2" : !emitc.opaque<"size_t">
      %121 = add %107, %120 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %122 = add %121, %119 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %123 = cast %122 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %124 = call_opaque "__riscv_vle8_v_i8m1"(%123, %118) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %125 = literal "2" : !emitc.opaque<"size_t">
      %126 = add %112, %125 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %127 = add %126, %119 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %128 = cast %127 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %129 = call_opaque "__riscv_vle8_v_i8m1"(%128, %118) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %130 = literal "18" : !emitc.opaque<"size_t">
      %131 = add %112, %130 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %132 = add %131, %119 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %133 = cast %132 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %134 = call_opaque "__riscv_vle8_v_i8m1"(%133, %118) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vxor_vx_i8m1"
      %135 = literal "0x88" : !emitc.opaque<"int">
      %136 = call_opaque "__riscv_vxor_vx_i8m1"(%124, %135, %118) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1"
      %137 = literal "4" : !emitc.opaque<"uint8_t">
      %138 = call_opaque "__riscv_vsll_vx_i8m1"(%136, %137, %118) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1"
      %139 = literal "4" : !emitc.opaque<"uint8_t">
      %140 = call_opaque "__riscv_vsra_vx_i8m1"(%138, %139, %118) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1"
      %141 = literal "4" : !emitc.opaque<"uint8_t">
      %142 = call_opaque "__riscv_vsra_vx_i8m1"(%136, %141, %118) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %143 = call_opaque "__riscv_vwmul_vv_i16m2"(%140, %129, %118) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2"
      %144 = call_opaque "__riscv_vwmacc_vv_i16m2"(%143, %142, %134, %118) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %145 = literal "0" : !emitc.opaque<"int32_t">
      %146 = literal "1" : !emitc.opaque<"size_t">
      %147 = call_opaque "__riscv_vmv_v_x_i32m1"(%145, %146) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %148 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%144, %147, %118) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %149 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%148) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %149 : !emitc.opaque<"int32_t"> to %115 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x"
      %150 = literal "3" : !emitc.opaque<"size_t">
      %151 = add %arg8, %150 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %152 = literal "18" : !emitc.opaque<"size_t">
      %153 = mul %151, %152 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %154 = add %arg3, %153 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %155 = literal "3" : !emitc.opaque<"size_t">
      %156 = add %arg8, %155 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %157 = literal "34" : !emitc.opaque<"size_t">
      %158 = mul %156, %157 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %159 = add %arg5, %158 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %160 = call_opaque "(float)*(const _Float16 *)"(%154) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %161 = call_opaque "(float)*(const _Float16 *)"(%159) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %162 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %163 = literal "0" : !emitc.opaque<"int32_t">
      assign %163 : !emitc.opaque<"int32_t"> to %162 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %164 = literal "16" : !emitc.opaque<"size_t">
      %165 = call_opaque "__riscv_vsetvl_e8m1"(%164) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %166 = literal "0" : !emitc.opaque<"size_t">
      %167 = literal "2" : !emitc.opaque<"size_t">
      %168 = add %154, %167 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %169 = add %168, %166 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %170 = cast %169 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %171 = call_opaque "__riscv_vle8_v_i8m1"(%170, %165) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %172 = literal "2" : !emitc.opaque<"size_t">
      %173 = add %159, %172 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %174 = add %173, %166 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %175 = cast %174 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %176 = call_opaque "__riscv_vle8_v_i8m1"(%175, %165) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %177 = literal "18" : !emitc.opaque<"size_t">
      %178 = add %159, %177 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %179 = add %178, %166 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %180 = cast %179 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %181 = call_opaque "__riscv_vle8_v_i8m1"(%180, %165) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vxor_vx_i8m1"
      %182 = literal "0x88" : !emitc.opaque<"int">
      %183 = call_opaque "__riscv_vxor_vx_i8m1"(%171, %182, %165) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1"
      %184 = literal "4" : !emitc.opaque<"uint8_t">
      %185 = call_opaque "__riscv_vsll_vx_i8m1"(%183, %184, %165) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1"
      %186 = literal "4" : !emitc.opaque<"uint8_t">
      %187 = call_opaque "__riscv_vsra_vx_i8m1"(%185, %186, %165) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1"
      %188 = literal "4" : !emitc.opaque<"uint8_t">
      %189 = call_opaque "__riscv_vsra_vx_i8m1"(%183, %188, %165) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %190 = call_opaque "__riscv_vwmul_vv_i16m2"(%187, %176, %165) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2"
      %191 = call_opaque "__riscv_vwmacc_vv_i16m2"(%190, %189, %181, %165) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %192 = literal "0" : !emitc.opaque<"int32_t">
      %193 = literal "1" : !emitc.opaque<"size_t">
      %194 = call_opaque "__riscv_vmv_v_x_i32m1"(%192, %193) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %195 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%191, %194, %165) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %196 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%195) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %196 : !emitc.opaque<"int32_t"> to %162 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %197 = load %21 : <!emitc.opaque<"int32_t">>
      %198 = load %1 : <!emitc.opaque<"float">>
      %199 = expression : !emitc.opaque<"float"> {
        %209 = cast %197 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %210 = mul %209, %19 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %211 = mul %210, %20 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %212 = add %198, %211 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %212 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %199 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %200 = load %68 : <!emitc.opaque<"int32_t">>
      %201 = load %1 : <!emitc.opaque<"float">>
      %202 = expression : !emitc.opaque<"float"> {
        %209 = cast %200 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %210 = mul %209, %66 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %211 = mul %210, %67 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %212 = add %201, %211 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %212 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %202 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %203 = load %115 : <!emitc.opaque<"int32_t">>
      %204 = load %1 : <!emitc.opaque<"float">>
      %205 = expression : !emitc.opaque<"float"> {
        %209 = cast %203 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %210 = mul %209, %113 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %211 = mul %210, %114 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %212 = add %204, %211 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %212 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %205 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %206 = load %162 : <!emitc.opaque<"int32_t">>
      %207 = load %1 : <!emitc.opaque<"float">>
      %208 = expression : !emitc.opaque<"float"> {
        %209 = cast %206 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %210 = mul %209, %160 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %211 = mul %210, %161 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %212 = add %207, %211 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %212 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %208 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
    }
    %9 = literal "1" : !emitc.opaque<"size_t">
    for %arg8 = %7 to %4 step %9  : !emitc.opaque<"size_t"> {
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x"
      %13 = literal "18" : !emitc.opaque<"size_t">
      %14 = mul %arg8, %13 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %15 = add %arg3, %14 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %16 = literal "34" : !emitc.opaque<"size_t">
      %17 = mul %arg8, %16 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %18 = add %arg5, %17 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %19 = call_opaque "(float)*(const _Float16 *)"(%15) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %20 = call_opaque "(float)*(const _Float16 *)"(%18) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %21 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %22 = literal "0" : !emitc.opaque<"int32_t">
      assign %22 : !emitc.opaque<"int32_t"> to %21 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %23 = literal "16" : !emitc.opaque<"size_t">
      %24 = call_opaque "__riscv_vsetvl_e8m1"(%23) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %25 = literal "16" : !emitc.opaque<"size_t">
      %26 = literal "0" : !emitc.opaque<"size_t">
      for %arg9 = %26 to %25 step %24  : !emitc.opaque<"size_t"> {
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
        %30 = literal "16" : !emitc.opaque<"size_t">
        %31 = sub %30, %arg9 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %32 = call_opaque "__riscv_vsetvl_e8m1"(%31) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %33 = literal "2" : !emitc.opaque<"size_t">
        %34 = add %15, %33 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %35 = add %34, %arg9 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %36 = cast %35 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
        %37 = call_opaque "__riscv_vle8_v_i8m1"(%36, %32) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        %38 = literal "2" : !emitc.opaque<"size_t">
        %39 = add %18, %38 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %40 = add %39, %arg9 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %41 = cast %40 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
        %42 = call_opaque "__riscv_vle8_v_i8m1"(%41, %32) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        %43 = literal "18" : !emitc.opaque<"size_t">
        %44 = add %18, %43 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %45 = add %44, %arg9 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %46 = cast %45 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
        %47 = call_opaque "__riscv_vle8_v_i8m1"(%46, %32) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vxor_vx_i8m1"
        %48 = literal "0x88" : !emitc.opaque<"int">
        %49 = call_opaque "__riscv_vxor_vx_i8m1"(%37, %48, %32) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1"
        %50 = literal "4" : !emitc.opaque<"uint8_t">
        %51 = call_opaque "__riscv_vsll_vx_i8m1"(%49, %50, %32) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1"
        %52 = literal "4" : !emitc.opaque<"uint8_t">
        %53 = call_opaque "__riscv_vsra_vx_i8m1"(%51, %52, %32) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1"
        %54 = literal "4" : !emitc.opaque<"uint8_t">
        %55 = call_opaque "__riscv_vsra_vx_i8m1"(%49, %54, %32) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
        %56 = call_opaque "__riscv_vwmul_vv_i16m2"(%53, %42, %32) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2"
        %57 = call_opaque "__riscv_vwmacc_vv_i16m2"(%56, %55, %47, %32) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
        %58 = load %21 : <!emitc.opaque<"int32_t">>
        %59 = literal "1" : !emitc.opaque<"size_t">
        %60 = call_opaque "__riscv_vmv_v_x_i32m1"(%58, %59) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
        %61 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%57, %60, %32) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
        %62 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%61) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
        verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
        assign %62 : !emitc.opaque<"int32_t"> to %21 : <!emitc.opaque<"int32_t">>
      }
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %27 = load %21 : <!emitc.opaque<"int32_t">>
      %28 = load %1 : <!emitc.opaque<"float">>
      %29 = expression : !emitc.opaque<"float"> {
        %30 = cast %27 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %31 = mul %30, %19 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %32 = mul %31, %20 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %33 = add %28, %32 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %33 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %29 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
    }
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s"
    %10 = literal "0" : index
    %11 = subscript %arg1[%10] : (!emitc.ptr<!emitc.opaque<"float">>, index) -> !emitc.lvalue<!emitc.opaque<"float">>
    %12 = load %1 : <!emitc.opaque<"float">>
    assign %12 : !emitc.opaque<"float"> to %11 : <!emitc.opaque<"float">>
    return
  }
}

