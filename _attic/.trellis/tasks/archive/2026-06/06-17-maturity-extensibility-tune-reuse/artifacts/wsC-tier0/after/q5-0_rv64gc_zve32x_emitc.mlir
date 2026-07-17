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
    %5 = literal "2" : !emitc.opaque<"size_t">
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
      %31 = literal "16" : !emitc.opaque<"size_t">
      %32 = literal "0" : !emitc.opaque<"size_t">
      for %arg5 = %32 to %31 step %30  : !emitc.opaque<"size_t"> {
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
        %63 = literal "16" : !emitc.opaque<"size_t">
        %64 = sub %63, %arg5 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %65 = call_opaque "__riscv_vsetvl_e8m1"(%64) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %66 = literal "6" : !emitc.opaque<"size_t">
        %67 = add %15, %66 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %68 = add %67, %arg5 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %69 = cast %68 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
        %70 = call_opaque "__riscv_vle8_v_u8m1"(%69, %65) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        %71 = literal "2" : !emitc.opaque<"size_t">
        %72 = add %18, %71 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %73 = add %72, %arg5 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %74 = cast %73 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
        %75 = call_opaque "__riscv_vle8_v_i8m1"(%74, %65) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        %76 = literal "18" : !emitc.opaque<"size_t">
        %77 = add %18, %76 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %78 = add %77, %arg5 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %79 = cast %78 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
        %80 = call_opaque "__riscv_vle8_v_i8m1"(%79, %65) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1"
        %81 = literal "0x0F" : !emitc.opaque<"int">
        %82 = call_opaque "__riscv_vand_vx_u8m1"(%70, %81, %65) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1"
        %83 = literal "0x04" : !emitc.opaque<"int">
        %84 = call_opaque "__riscv_vsrl_vx_u8m1"(%70, %83, %65) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
        %85 = call_opaque "__riscv_vid_v_u16m2"(%65) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
        %86 = call_opaque "__riscv_vadd_vx_u16m2"(%85, %arg5, %65) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
        %87 = call_opaque "__riscv_vmv_v_x_u16m2"(%23, %65) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
        %88 = call_opaque "__riscv_vsrl_vv_u16m2"(%87, %86, %65) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
        %89 = literal "0x1" : !emitc.opaque<"int">
        %90 = call_opaque "__riscv_vand_vx_u16m2"(%88, %89, %65) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
        %91 = literal "0x4" : !emitc.opaque<"int">
        %92 = call_opaque "__riscv_vsll_vx_u16m2"(%90, %91, %65) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
        %93 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%92, %65) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
        %94 = call_opaque "__riscv_vid_v_u16m2"(%65) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
        %95 = call_opaque "__riscv_vadd_vx_u16m2"(%94, %arg5, %65) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
        %96 = call_opaque "__riscv_vmv_v_x_u16m2"(%26, %65) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
        %97 = call_opaque "__riscv_vsrl_vv_u16m2"(%96, %95, %65) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
        %98 = literal "0x1" : !emitc.opaque<"int">
        %99 = call_opaque "__riscv_vand_vx_u16m2"(%97, %98, %65) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
        %100 = literal "0x4" : !emitc.opaque<"int">
        %101 = call_opaque "__riscv_vsll_vx_u16m2"(%99, %100, %65) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
        %102 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%101, %65) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
        %103 = call_opaque "__riscv_vor_vv_u8m1"(%82, %93, %65) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
        %104 = call_opaque "__riscv_vor_vv_u8m1"(%84, %102, %65) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
        %105 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%103) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1"
        %106 = literal "16" : !emitc.opaque<"int">
        %107 = call_opaque "__riscv_vsub_vx_i8m1"(%105, %106, %65) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
        %108 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%104) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1"
        %109 = literal "16" : !emitc.opaque<"int">
        %110 = call_opaque "__riscv_vsub_vx_i8m1"(%108, %109, %65) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
        %111 = call_opaque "__riscv_vwmul_vv_i16m2"(%107, %75, %65) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2"
        %112 = call_opaque "__riscv_vwmacc_vv_i16m2"(%111, %110, %80, %65) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
        %113 = load %27 : <!emitc.opaque<"int32_t">>
        %114 = literal "1" : !emitc.opaque<"size_t">
        %115 = call_opaque "__riscv_vmv_v_x_i32m1"(%113, %114) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
        %116 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%112, %115, %65) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
        %117 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%116) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
        verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
        assign %117 : !emitc.opaque<"int32_t"> to %27 : <!emitc.opaque<"int32_t">>
      }
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x"
      %33 = literal "1" : !emitc.opaque<"size_t">
      %34 = add %arg4, %33 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %35 = literal "22" : !emitc.opaque<"size_t">
      %36 = mul %34, %35 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %37 = add %arg2, %36 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %38 = literal "1" : !emitc.opaque<"size_t">
      %39 = add %arg4, %38 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %40 = literal "34" : !emitc.opaque<"size_t">
      %41 = mul %39, %40 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %42 = add %arg3, %41 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %43 = call_opaque "(float)*(const _Float16 *)"(%37) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %44 = call_opaque "(float)*(const _Float16 *)"(%42) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field"
      %45 = literal "2" : !emitc.opaque<"size_t">
      %46 = add %37, %45 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %47 = call_opaque "(uint16_t)*(const uint16_t *)"(%46) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      %48 = literal "4" : !emitc.opaque<"size_t">
      %49 = add %37, %48 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %50 = call_opaque "(uint16_t)*(const uint16_t *)"(%49) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %51 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %52 = literal "0" : !emitc.opaque<"int32_t">
      assign %52 : !emitc.opaque<"int32_t"> to %51 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %53 = literal "16" : !emitc.opaque<"size_t">
      %54 = call_opaque "__riscv_vsetvl_e8m1"(%53) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %55 = literal "16" : !emitc.opaque<"size_t">
      %56 = literal "0" : !emitc.opaque<"size_t">
      for %arg5 = %56 to %55 step %54  : !emitc.opaque<"size_t"> {
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
        %63 = literal "16" : !emitc.opaque<"size_t">
        %64 = sub %63, %arg5 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %65 = call_opaque "__riscv_vsetvl_e8m1"(%64) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %66 = literal "6" : !emitc.opaque<"size_t">
        %67 = add %37, %66 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %68 = add %67, %arg5 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %69 = cast %68 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
        %70 = call_opaque "__riscv_vle8_v_u8m1"(%69, %65) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        %71 = literal "2" : !emitc.opaque<"size_t">
        %72 = add %42, %71 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %73 = add %72, %arg5 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %74 = cast %73 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
        %75 = call_opaque "__riscv_vle8_v_i8m1"(%74, %65) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        %76 = literal "18" : !emitc.opaque<"size_t">
        %77 = add %42, %76 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %78 = add %77, %arg5 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %79 = cast %78 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
        %80 = call_opaque "__riscv_vle8_v_i8m1"(%79, %65) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1"
        %81 = literal "0x0F" : !emitc.opaque<"int">
        %82 = call_opaque "__riscv_vand_vx_u8m1"(%70, %81, %65) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1"
        %83 = literal "0x04" : !emitc.opaque<"int">
        %84 = call_opaque "__riscv_vsrl_vx_u8m1"(%70, %83, %65) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
        %85 = call_opaque "__riscv_vid_v_u16m2"(%65) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
        %86 = call_opaque "__riscv_vadd_vx_u16m2"(%85, %arg5, %65) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
        %87 = call_opaque "__riscv_vmv_v_x_u16m2"(%47, %65) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
        %88 = call_opaque "__riscv_vsrl_vv_u16m2"(%87, %86, %65) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
        %89 = literal "0x1" : !emitc.opaque<"int">
        %90 = call_opaque "__riscv_vand_vx_u16m2"(%88, %89, %65) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
        %91 = literal "0x4" : !emitc.opaque<"int">
        %92 = call_opaque "__riscv_vsll_vx_u16m2"(%90, %91, %65) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
        %93 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%92, %65) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
        %94 = call_opaque "__riscv_vid_v_u16m2"(%65) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
        %95 = call_opaque "__riscv_vadd_vx_u16m2"(%94, %arg5, %65) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
        %96 = call_opaque "__riscv_vmv_v_x_u16m2"(%50, %65) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
        %97 = call_opaque "__riscv_vsrl_vv_u16m2"(%96, %95, %65) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
        %98 = literal "0x1" : !emitc.opaque<"int">
        %99 = call_opaque "__riscv_vand_vx_u16m2"(%97, %98, %65) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
        %100 = literal "0x4" : !emitc.opaque<"int">
        %101 = call_opaque "__riscv_vsll_vx_u16m2"(%99, %100, %65) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
        %102 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%101, %65) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
        %103 = call_opaque "__riscv_vor_vv_u8m1"(%82, %93, %65) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
        %104 = call_opaque "__riscv_vor_vv_u8m1"(%84, %102, %65) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
        %105 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%103) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1"
        %106 = literal "16" : !emitc.opaque<"int">
        %107 = call_opaque "__riscv_vsub_vx_i8m1"(%105, %106, %65) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
        %108 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%104) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1"
        %109 = literal "16" : !emitc.opaque<"int">
        %110 = call_opaque "__riscv_vsub_vx_i8m1"(%108, %109, %65) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
        %111 = call_opaque "__riscv_vwmul_vv_i16m2"(%107, %75, %65) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2"
        %112 = call_opaque "__riscv_vwmacc_vv_i16m2"(%111, %110, %80, %65) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
        %113 = load %51 : <!emitc.opaque<"int32_t">>
        %114 = literal "1" : !emitc.opaque<"size_t">
        %115 = call_opaque "__riscv_vmv_v_x_i32m1"(%113, %114) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
        %116 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%112, %115, %65) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
        %117 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%116) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
        verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
        assign %117 : !emitc.opaque<"int32_t"> to %51 : <!emitc.opaque<"int32_t">>
      }
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %57 = load %27 : <!emitc.opaque<"int32_t">>
      %58 = load %1 : <!emitc.opaque<"float">>
      %59 = expression : !emitc.opaque<"float"> {
        %63 = mul %19, %20 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %64 = cast %57 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %65 = mul %63, %64 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %66 = add %58, %65 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %66 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %59 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %60 = load %51 : <!emitc.opaque<"int32_t">>
      %61 = load %1 : <!emitc.opaque<"float">>
      %62 = expression : !emitc.opaque<"float"> {
        %63 = mul %43, %44 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %64 = cast %60 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %65 = mul %63, %64 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %66 = add %61, %65 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %66 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %62 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
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

