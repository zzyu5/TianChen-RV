module {
  emitc.include <"stddef.h">
  emitc.include <"stdint.h">
  emitc.include <"riscv_vector.h">
  emitc.func @tcrv_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0(%arg0: !emitc.opaque<"size_t">, %arg1: !emitc.ptr<!emitc.opaque<"float">>, %arg2: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg3: !emitc.ptr<!emitc.opaque<"const uint8_t">>) attributes {specifiers = ["extern", "\22C\22"]} {
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1"
    %0 = call_opaque "__riscv_vsetvl_e32m1"(%arg0) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
    %1 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"float">>
    %2 = literal "0.0f" : !emitc.opaque<"float">
    assign %2 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count"
    %3 = literal "32" : !emitc.opaque<"size_t">
    %4 = div %arg0, %3 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    %5 = literal "4" : !emitc.opaque<"size_t">
    %6 = rem %4, %5 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    %7 = sub %4, %6 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    %8 = literal "0" : !emitc.opaque<"size_t">
    for %arg4 = %8 to %7 step %5  : !emitc.opaque<"size_t"> {
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x"
      %13 = literal "22" : !emitc.opaque<"size_t">
      %14 = mul %arg4, %13 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %15 = add %arg2, %14 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %16 = literal "34" : !emitc.opaque<"size_t">
      %17 = mul %arg4, %16 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %18 = add %arg3, %17 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %19 = call_opaque "(float)*(const _Float16 *)"(%15) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %20 = call_opaque "(float)*(const _Float16 *)"(%18) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field"
      %21 = literal "2" : !emitc.opaque<"size_t">
      %22 = add %15, %21 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %23 = call_opaque "(uint16_t)*(const uint16_t *)"(%22) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      %24 = literal "4" : !emitc.opaque<"size_t">
      %25 = add %15, %24 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %26 = call_opaque "(uint16_t)*(const uint16_t *)"(%25) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %27 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %28 = literal "0" : !emitc.opaque<"int32_t">
      assign %28 : !emitc.opaque<"int32_t"> to %27 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %29 = literal "16" : !emitc.opaque<"size_t">
      %30 = call_opaque "__riscv_vsetvl_e8m1"(%29) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %31 = literal "0" : !emitc.opaque<"size_t">
      %32 = literal "6" : !emitc.opaque<"size_t">
      %33 = add %15, %32 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %34 = add %33, %31 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %35 = cast %34 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
      %36 = call_opaque "__riscv_vle8_v_u8m1"(%35, %30) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      %37 = literal "2" : !emitc.opaque<"size_t">
      %38 = add %18, %37 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %39 = add %38, %31 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %40 = cast %39 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %41 = call_opaque "__riscv_vle8_v_i8m1"(%40, %30) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %42 = literal "18" : !emitc.opaque<"size_t">
      %43 = add %18, %42 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %44 = add %43, %31 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %45 = cast %44 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %46 = call_opaque "__riscv_vle8_v_i8m1"(%45, %30) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1"
      %47 = literal "0x0F" : !emitc.opaque<"int">
      %48 = call_opaque "__riscv_vand_vx_u8m1"(%36, %47, %30) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1"
      %49 = literal "0x04" : !emitc.opaque<"int">
      %50 = call_opaque "__riscv_vsrl_vx_u8m1"(%36, %49, %30) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
      %51 = call_opaque "__riscv_vid_v_u16m2"(%30) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
      %52 = call_opaque "__riscv_vadd_vx_u16m2"(%51, %31, %30) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
      %53 = call_opaque "__riscv_vmv_v_x_u16m2"(%23, %30) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
      %54 = call_opaque "__riscv_vsrl_vv_u16m2"(%53, %52, %30) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
      %55 = literal "0x1" : !emitc.opaque<"int">
      %56 = call_opaque "__riscv_vand_vx_u16m2"(%54, %55, %30) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
      %57 = literal "0x4" : !emitc.opaque<"int">
      %58 = call_opaque "__riscv_vsll_vx_u16m2"(%56, %57, %30) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %59 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%58, %30) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
      %60 = call_opaque "__riscv_vid_v_u16m2"(%30) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
      %61 = call_opaque "__riscv_vadd_vx_u16m2"(%60, %31, %30) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
      %62 = call_opaque "__riscv_vmv_v_x_u16m2"(%26, %30) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
      %63 = call_opaque "__riscv_vsrl_vv_u16m2"(%62, %61, %30) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
      %64 = literal "0x1" : !emitc.opaque<"int">
      %65 = call_opaque "__riscv_vand_vx_u16m2"(%63, %64, %30) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
      %66 = literal "0x4" : !emitc.opaque<"int">
      %67 = call_opaque "__riscv_vsll_vx_u16m2"(%65, %66, %30) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %68 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%67, %30) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
      %69 = call_opaque "__riscv_vor_vv_u8m1"(%48, %59, %30) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
      %70 = call_opaque "__riscv_vor_vv_u8m1"(%50, %68, %30) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %71 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%69) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1"
      %72 = literal "16" : !emitc.opaque<"int">
      %73 = call_opaque "__riscv_vsub_vx_i8m1"(%71, %72, %30) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %74 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%70) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1"
      %75 = literal "16" : !emitc.opaque<"int">
      %76 = call_opaque "__riscv_vsub_vx_i8m1"(%74, %75, %30) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %77 = call_opaque "__riscv_vwmul_vv_i16m2"(%73, %41, %30) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2"
      %78 = call_opaque "__riscv_vwmacc_vv_i16m2"(%77, %76, %46, %30) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %79 = literal "0" : !emitc.opaque<"int32_t">
      %80 = literal "1" : !emitc.opaque<"size_t">
      %81 = call_opaque "__riscv_vmv_v_x_i32m1"(%79, %80) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %82 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%78, %81, %30) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %83 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%82) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %83 : !emitc.opaque<"int32_t"> to %27 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x"
      %84 = literal "1" : !emitc.opaque<"size_t">
      %85 = add %arg4, %84 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %86 = literal "22" : !emitc.opaque<"size_t">
      %87 = mul %85, %86 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %88 = add %arg2, %87 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %89 = literal "1" : !emitc.opaque<"size_t">
      %90 = add %arg4, %89 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %91 = literal "34" : !emitc.opaque<"size_t">
      %92 = mul %90, %91 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %93 = add %arg3, %92 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %94 = call_opaque "(float)*(const _Float16 *)"(%88) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %95 = call_opaque "(float)*(const _Float16 *)"(%93) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field"
      %96 = literal "2" : !emitc.opaque<"size_t">
      %97 = add %88, %96 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %98 = call_opaque "(uint16_t)*(const uint16_t *)"(%97) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      %99 = literal "4" : !emitc.opaque<"size_t">
      %100 = add %88, %99 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %101 = call_opaque "(uint16_t)*(const uint16_t *)"(%100) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %102 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %103 = literal "0" : !emitc.opaque<"int32_t">
      assign %103 : !emitc.opaque<"int32_t"> to %102 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %104 = literal "16" : !emitc.opaque<"size_t">
      %105 = call_opaque "__riscv_vsetvl_e8m1"(%104) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %106 = literal "0" : !emitc.opaque<"size_t">
      %107 = literal "6" : !emitc.opaque<"size_t">
      %108 = add %88, %107 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %109 = add %108, %106 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %110 = cast %109 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
      %111 = call_opaque "__riscv_vle8_v_u8m1"(%110, %105) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      %112 = literal "2" : !emitc.opaque<"size_t">
      %113 = add %93, %112 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %114 = add %113, %106 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %115 = cast %114 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %116 = call_opaque "__riscv_vle8_v_i8m1"(%115, %105) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %117 = literal "18" : !emitc.opaque<"size_t">
      %118 = add %93, %117 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %119 = add %118, %106 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %120 = cast %119 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %121 = call_opaque "__riscv_vle8_v_i8m1"(%120, %105) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1"
      %122 = literal "0x0F" : !emitc.opaque<"int">
      %123 = call_opaque "__riscv_vand_vx_u8m1"(%111, %122, %105) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1"
      %124 = literal "0x04" : !emitc.opaque<"int">
      %125 = call_opaque "__riscv_vsrl_vx_u8m1"(%111, %124, %105) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
      %126 = call_opaque "__riscv_vid_v_u16m2"(%105) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
      %127 = call_opaque "__riscv_vadd_vx_u16m2"(%126, %106, %105) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
      %128 = call_opaque "__riscv_vmv_v_x_u16m2"(%98, %105) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
      %129 = call_opaque "__riscv_vsrl_vv_u16m2"(%128, %127, %105) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
      %130 = literal "0x1" : !emitc.opaque<"int">
      %131 = call_opaque "__riscv_vand_vx_u16m2"(%129, %130, %105) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
      %132 = literal "0x4" : !emitc.opaque<"int">
      %133 = call_opaque "__riscv_vsll_vx_u16m2"(%131, %132, %105) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %134 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%133, %105) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
      %135 = call_opaque "__riscv_vid_v_u16m2"(%105) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
      %136 = call_opaque "__riscv_vadd_vx_u16m2"(%135, %106, %105) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
      %137 = call_opaque "__riscv_vmv_v_x_u16m2"(%101, %105) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
      %138 = call_opaque "__riscv_vsrl_vv_u16m2"(%137, %136, %105) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
      %139 = literal "0x1" : !emitc.opaque<"int">
      %140 = call_opaque "__riscv_vand_vx_u16m2"(%138, %139, %105) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
      %141 = literal "0x4" : !emitc.opaque<"int">
      %142 = call_opaque "__riscv_vsll_vx_u16m2"(%140, %141, %105) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %143 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%142, %105) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
      %144 = call_opaque "__riscv_vor_vv_u8m1"(%123, %134, %105) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
      %145 = call_opaque "__riscv_vor_vv_u8m1"(%125, %143, %105) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %146 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%144) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1"
      %147 = literal "16" : !emitc.opaque<"int">
      %148 = call_opaque "__riscv_vsub_vx_i8m1"(%146, %147, %105) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %149 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%145) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1"
      %150 = literal "16" : !emitc.opaque<"int">
      %151 = call_opaque "__riscv_vsub_vx_i8m1"(%149, %150, %105) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %152 = call_opaque "__riscv_vwmul_vv_i16m2"(%148, %116, %105) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2"
      %153 = call_opaque "__riscv_vwmacc_vv_i16m2"(%152, %151, %121, %105) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %154 = literal "0" : !emitc.opaque<"int32_t">
      %155 = literal "1" : !emitc.opaque<"size_t">
      %156 = call_opaque "__riscv_vmv_v_x_i32m1"(%154, %155) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %157 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%153, %156, %105) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %158 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%157) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %158 : !emitc.opaque<"int32_t"> to %102 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x"
      %159 = literal "2" : !emitc.opaque<"size_t">
      %160 = add %arg4, %159 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %161 = literal "22" : !emitc.opaque<"size_t">
      %162 = mul %160, %161 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %163 = add %arg2, %162 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %164 = literal "2" : !emitc.opaque<"size_t">
      %165 = add %arg4, %164 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %166 = literal "34" : !emitc.opaque<"size_t">
      %167 = mul %165, %166 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %168 = add %arg3, %167 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %169 = call_opaque "(float)*(const _Float16 *)"(%163) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %170 = call_opaque "(float)*(const _Float16 *)"(%168) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field"
      %171 = literal "2" : !emitc.opaque<"size_t">
      %172 = add %163, %171 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %173 = call_opaque "(uint16_t)*(const uint16_t *)"(%172) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      %174 = literal "4" : !emitc.opaque<"size_t">
      %175 = add %163, %174 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %176 = call_opaque "(uint16_t)*(const uint16_t *)"(%175) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %177 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %178 = literal "0" : !emitc.opaque<"int32_t">
      assign %178 : !emitc.opaque<"int32_t"> to %177 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %179 = literal "16" : !emitc.opaque<"size_t">
      %180 = call_opaque "__riscv_vsetvl_e8m1"(%179) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %181 = literal "0" : !emitc.opaque<"size_t">
      %182 = literal "6" : !emitc.opaque<"size_t">
      %183 = add %163, %182 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %184 = add %183, %181 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %185 = cast %184 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
      %186 = call_opaque "__riscv_vle8_v_u8m1"(%185, %180) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      %187 = literal "2" : !emitc.opaque<"size_t">
      %188 = add %168, %187 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %189 = add %188, %181 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %190 = cast %189 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %191 = call_opaque "__riscv_vle8_v_i8m1"(%190, %180) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %192 = literal "18" : !emitc.opaque<"size_t">
      %193 = add %168, %192 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %194 = add %193, %181 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %195 = cast %194 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %196 = call_opaque "__riscv_vle8_v_i8m1"(%195, %180) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1"
      %197 = literal "0x0F" : !emitc.opaque<"int">
      %198 = call_opaque "__riscv_vand_vx_u8m1"(%186, %197, %180) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1"
      %199 = literal "0x04" : !emitc.opaque<"int">
      %200 = call_opaque "__riscv_vsrl_vx_u8m1"(%186, %199, %180) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
      %201 = call_opaque "__riscv_vid_v_u16m2"(%180) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
      %202 = call_opaque "__riscv_vadd_vx_u16m2"(%201, %181, %180) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
      %203 = call_opaque "__riscv_vmv_v_x_u16m2"(%173, %180) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
      %204 = call_opaque "__riscv_vsrl_vv_u16m2"(%203, %202, %180) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
      %205 = literal "0x1" : !emitc.opaque<"int">
      %206 = call_opaque "__riscv_vand_vx_u16m2"(%204, %205, %180) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
      %207 = literal "0x4" : !emitc.opaque<"int">
      %208 = call_opaque "__riscv_vsll_vx_u16m2"(%206, %207, %180) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %209 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%208, %180) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
      %210 = call_opaque "__riscv_vid_v_u16m2"(%180) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
      %211 = call_opaque "__riscv_vadd_vx_u16m2"(%210, %181, %180) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
      %212 = call_opaque "__riscv_vmv_v_x_u16m2"(%176, %180) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
      %213 = call_opaque "__riscv_vsrl_vv_u16m2"(%212, %211, %180) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
      %214 = literal "0x1" : !emitc.opaque<"int">
      %215 = call_opaque "__riscv_vand_vx_u16m2"(%213, %214, %180) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
      %216 = literal "0x4" : !emitc.opaque<"int">
      %217 = call_opaque "__riscv_vsll_vx_u16m2"(%215, %216, %180) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %218 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%217, %180) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
      %219 = call_opaque "__riscv_vor_vv_u8m1"(%198, %209, %180) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
      %220 = call_opaque "__riscv_vor_vv_u8m1"(%200, %218, %180) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %221 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%219) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1"
      %222 = literal "16" : !emitc.opaque<"int">
      %223 = call_opaque "__riscv_vsub_vx_i8m1"(%221, %222, %180) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %224 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%220) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1"
      %225 = literal "16" : !emitc.opaque<"int">
      %226 = call_opaque "__riscv_vsub_vx_i8m1"(%224, %225, %180) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %227 = call_opaque "__riscv_vwmul_vv_i16m2"(%223, %191, %180) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2"
      %228 = call_opaque "__riscv_vwmacc_vv_i16m2"(%227, %226, %196, %180) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %229 = literal "0" : !emitc.opaque<"int32_t">
      %230 = literal "1" : !emitc.opaque<"size_t">
      %231 = call_opaque "__riscv_vmv_v_x_i32m1"(%229, %230) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %232 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%228, %231, %180) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %233 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%232) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %233 : !emitc.opaque<"int32_t"> to %177 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x"
      %234 = literal "3" : !emitc.opaque<"size_t">
      %235 = add %arg4, %234 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %236 = literal "22" : !emitc.opaque<"size_t">
      %237 = mul %235, %236 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %238 = add %arg2, %237 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %239 = literal "3" : !emitc.opaque<"size_t">
      %240 = add %arg4, %239 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %241 = literal "34" : !emitc.opaque<"size_t">
      %242 = mul %240, %241 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %243 = add %arg3, %242 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %244 = call_opaque "(float)*(const _Float16 *)"(%238) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %245 = call_opaque "(float)*(const _Float16 *)"(%243) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field"
      %246 = literal "2" : !emitc.opaque<"size_t">
      %247 = add %238, %246 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %248 = call_opaque "(uint16_t)*(const uint16_t *)"(%247) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      %249 = literal "4" : !emitc.opaque<"size_t">
      %250 = add %238, %249 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %251 = call_opaque "(uint16_t)*(const uint16_t *)"(%250) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %252 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %253 = literal "0" : !emitc.opaque<"int32_t">
      assign %253 : !emitc.opaque<"int32_t"> to %252 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %254 = literal "16" : !emitc.opaque<"size_t">
      %255 = call_opaque "__riscv_vsetvl_e8m1"(%254) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %256 = literal "0" : !emitc.opaque<"size_t">
      %257 = literal "6" : !emitc.opaque<"size_t">
      %258 = add %238, %257 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %259 = add %258, %256 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %260 = cast %259 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
      %261 = call_opaque "__riscv_vle8_v_u8m1"(%260, %255) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      %262 = literal "2" : !emitc.opaque<"size_t">
      %263 = add %243, %262 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %264 = add %263, %256 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %265 = cast %264 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %266 = call_opaque "__riscv_vle8_v_i8m1"(%265, %255) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %267 = literal "18" : !emitc.opaque<"size_t">
      %268 = add %243, %267 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %269 = add %268, %256 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %270 = cast %269 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %271 = call_opaque "__riscv_vle8_v_i8m1"(%270, %255) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1"
      %272 = literal "0x0F" : !emitc.opaque<"int">
      %273 = call_opaque "__riscv_vand_vx_u8m1"(%261, %272, %255) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1"
      %274 = literal "0x04" : !emitc.opaque<"int">
      %275 = call_opaque "__riscv_vsrl_vx_u8m1"(%261, %274, %255) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
      %276 = call_opaque "__riscv_vid_v_u16m2"(%255) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
      %277 = call_opaque "__riscv_vadd_vx_u16m2"(%276, %256, %255) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
      %278 = call_opaque "__riscv_vmv_v_x_u16m2"(%248, %255) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
      %279 = call_opaque "__riscv_vsrl_vv_u16m2"(%278, %277, %255) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
      %280 = literal "0x1" : !emitc.opaque<"int">
      %281 = call_opaque "__riscv_vand_vx_u16m2"(%279, %280, %255) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
      %282 = literal "0x4" : !emitc.opaque<"int">
      %283 = call_opaque "__riscv_vsll_vx_u16m2"(%281, %282, %255) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %284 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%283, %255) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
      %285 = call_opaque "__riscv_vid_v_u16m2"(%255) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
      %286 = call_opaque "__riscv_vadd_vx_u16m2"(%285, %256, %255) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
      %287 = call_opaque "__riscv_vmv_v_x_u16m2"(%251, %255) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
      %288 = call_opaque "__riscv_vsrl_vv_u16m2"(%287, %286, %255) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
      %289 = literal "0x1" : !emitc.opaque<"int">
      %290 = call_opaque "__riscv_vand_vx_u16m2"(%288, %289, %255) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
      %291 = literal "0x4" : !emitc.opaque<"int">
      %292 = call_opaque "__riscv_vsll_vx_u16m2"(%290, %291, %255) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %293 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%292, %255) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
      %294 = call_opaque "__riscv_vor_vv_u8m1"(%273, %284, %255) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
      %295 = call_opaque "__riscv_vor_vv_u8m1"(%275, %293, %255) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %296 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%294) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1"
      %297 = literal "16" : !emitc.opaque<"int">
      %298 = call_opaque "__riscv_vsub_vx_i8m1"(%296, %297, %255) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %299 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%295) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1"
      %300 = literal "16" : !emitc.opaque<"int">
      %301 = call_opaque "__riscv_vsub_vx_i8m1"(%299, %300, %255) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %302 = call_opaque "__riscv_vwmul_vv_i16m2"(%298, %266, %255) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2"
      %303 = call_opaque "__riscv_vwmacc_vv_i16m2"(%302, %301, %271, %255) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %304 = literal "0" : !emitc.opaque<"int32_t">
      %305 = literal "1" : !emitc.opaque<"size_t">
      %306 = call_opaque "__riscv_vmv_v_x_i32m1"(%304, %305) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %307 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%303, %306, %255) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %308 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%307) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %308 : !emitc.opaque<"int32_t"> to %252 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %309 = load %27 : <!emitc.opaque<"int32_t">>
      %310 = load %1 : <!emitc.opaque<"float">>
      %311 = expression : !emitc.opaque<"float"> {
        %321 = mul %19, %20 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %322 = cast %309 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %323 = mul %321, %322 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %324 = add %310, %323 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %324 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %311 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %312 = load %102 : <!emitc.opaque<"int32_t">>
      %313 = load %1 : <!emitc.opaque<"float">>
      %314 = expression : !emitc.opaque<"float"> {
        %321 = mul %94, %95 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %322 = cast %312 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %323 = mul %321, %322 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %324 = add %313, %323 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %324 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %314 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %315 = load %177 : <!emitc.opaque<"int32_t">>
      %316 = load %1 : <!emitc.opaque<"float">>
      %317 = expression : !emitc.opaque<"float"> {
        %321 = mul %169, %170 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %322 = cast %315 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %323 = mul %321, %322 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %324 = add %316, %323 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %324 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %317 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %318 = load %252 : <!emitc.opaque<"int32_t">>
      %319 = load %1 : <!emitc.opaque<"float">>
      %320 = expression : !emitc.opaque<"float"> {
        %321 = mul %244, %245 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %322 = cast %318 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %323 = mul %321, %322 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %324 = add %319, %323 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %324 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %320 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
    }
    %9 = literal "1" : !emitc.opaque<"size_t">
    for %arg4 = %7 to %4 step %9  : !emitc.opaque<"size_t"> {
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x"
      %13 = literal "22" : !emitc.opaque<"size_t">
      %14 = mul %arg4, %13 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %15 = add %arg2, %14 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %16 = literal "34" : !emitc.opaque<"size_t">
      %17 = mul %arg4, %16 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %18 = add %arg3, %17 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %19 = call_opaque "(float)*(const _Float16 *)"(%15) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %20 = call_opaque "(float)*(const _Float16 *)"(%18) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field"
      %21 = literal "2" : !emitc.opaque<"size_t">
      %22 = add %15, %21 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %23 = call_opaque "(uint16_t)*(const uint16_t *)"(%22) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      %24 = literal "4" : !emitc.opaque<"size_t">
      %25 = add %15, %24 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %26 = call_opaque "(uint16_t)*(const uint16_t *)"(%25) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %27 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %28 = literal "0" : !emitc.opaque<"int32_t">
      assign %28 : !emitc.opaque<"int32_t"> to %27 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %29 = literal "16" : !emitc.opaque<"size_t">
      %30 = call_opaque "__riscv_vsetvl_e8m1"(%29) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %31 = literal "16" : !emitc.opaque<"size_t">
      %32 = literal "0" : !emitc.opaque<"size_t">
      for %arg5 = %32 to %31 step %30  : !emitc.opaque<"size_t"> {
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
        %36 = literal "16" : !emitc.opaque<"size_t">
        %37 = sub %36, %arg5 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %38 = call_opaque "__riscv_vsetvl_e8m1"(%37) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %39 = literal "6" : !emitc.opaque<"size_t">
        %40 = add %15, %39 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %41 = add %40, %arg5 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %42 = cast %41 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
        %43 = call_opaque "__riscv_vle8_v_u8m1"(%42, %38) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        %44 = literal "2" : !emitc.opaque<"size_t">
        %45 = add %18, %44 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %46 = add %45, %arg5 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %47 = cast %46 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
        %48 = call_opaque "__riscv_vle8_v_i8m1"(%47, %38) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        %49 = literal "18" : !emitc.opaque<"size_t">
        %50 = add %18, %49 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %51 = add %50, %arg5 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %52 = cast %51 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
        %53 = call_opaque "__riscv_vle8_v_i8m1"(%52, %38) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1"
        %54 = literal "0x0F" : !emitc.opaque<"int">
        %55 = call_opaque "__riscv_vand_vx_u8m1"(%43, %54, %38) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1"
        %56 = literal "0x04" : !emitc.opaque<"int">
        %57 = call_opaque "__riscv_vsrl_vx_u8m1"(%43, %56, %38) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
        %58 = call_opaque "__riscv_vid_v_u16m2"(%38) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
        %59 = call_opaque "__riscv_vadd_vx_u16m2"(%58, %arg5, %38) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
        %60 = call_opaque "__riscv_vmv_v_x_u16m2"(%23, %38) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
        %61 = call_opaque "__riscv_vsrl_vv_u16m2"(%60, %59, %38) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
        %62 = literal "0x1" : !emitc.opaque<"int">
        %63 = call_opaque "__riscv_vand_vx_u16m2"(%61, %62, %38) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
        %64 = literal "0x4" : !emitc.opaque<"int">
        %65 = call_opaque "__riscv_vsll_vx_u16m2"(%63, %64, %38) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
        %66 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%65, %38) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
        %67 = call_opaque "__riscv_vid_v_u16m2"(%38) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
        %68 = call_opaque "__riscv_vadd_vx_u16m2"(%67, %arg5, %38) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
        %69 = call_opaque "__riscv_vmv_v_x_u16m2"(%26, %38) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
        %70 = call_opaque "__riscv_vsrl_vv_u16m2"(%69, %68, %38) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
        %71 = literal "0x1" : !emitc.opaque<"int">
        %72 = call_opaque "__riscv_vand_vx_u16m2"(%70, %71, %38) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
        %73 = literal "0x4" : !emitc.opaque<"int">
        %74 = call_opaque "__riscv_vsll_vx_u16m2"(%72, %73, %38) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
        %75 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%74, %38) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
        %76 = call_opaque "__riscv_vor_vv_u8m1"(%55, %66, %38) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
        %77 = call_opaque "__riscv_vor_vv_u8m1"(%57, %75, %38) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
        %78 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%76) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1"
        %79 = literal "16" : !emitc.opaque<"int">
        %80 = call_opaque "__riscv_vsub_vx_i8m1"(%78, %79, %38) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
        %81 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%77) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1"
        %82 = literal "16" : !emitc.opaque<"int">
        %83 = call_opaque "__riscv_vsub_vx_i8m1"(%81, %82, %38) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
        %84 = call_opaque "__riscv_vwmul_vv_i16m2"(%80, %48, %38) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2"
        %85 = call_opaque "__riscv_vwmacc_vv_i16m2"(%84, %83, %53, %38) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
        %86 = load %27 : <!emitc.opaque<"int32_t">>
        %87 = literal "1" : !emitc.opaque<"size_t">
        %88 = call_opaque "__riscv_vmv_v_x_i32m1"(%86, %87) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
        %89 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%85, %88, %38) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
        %90 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%89) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
        verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
        assign %90 : !emitc.opaque<"int32_t"> to %27 : <!emitc.opaque<"int32_t">>
      }
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %33 = load %27 : <!emitc.opaque<"int32_t">>
      %34 = load %1 : <!emitc.opaque<"float">>
      %35 = expression : !emitc.opaque<"float"> {
        %36 = mul %19, %20 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %37 = cast %33 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %38 = mul %36, %37 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %39 = add %34, %38 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %39 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %35 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
    }
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s"
    %10 = literal "0" : index
    %11 = subscript %arg1[%10] : (!emitc.ptr<!emitc.opaque<"float">>, index) -> !emitc.lvalue<!emitc.opaque<"float">>
    %12 = load %1 : <!emitc.opaque<"float">>
    assign %12 : !emitc.opaque<"float"> to %11 : <!emitc.opaque<"float">>
    return
  }
}

