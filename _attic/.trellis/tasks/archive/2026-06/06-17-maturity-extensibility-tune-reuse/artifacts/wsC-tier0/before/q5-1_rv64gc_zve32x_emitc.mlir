module {
  emitc.include <"stddef.h">
  emitc.include <"stdint.h">
  emitc.include <"riscv_vector.h">
  emitc.func @tcrv_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1(%arg0: !emitc.opaque<"size_t">, %arg1: !emitc.ptr<!emitc.opaque<"float">>, %arg2: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg3: !emitc.ptr<!emitc.opaque<"const uint8_t">>) attributes {specifiers = ["extern", "\22C\22"]} {
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1"
    %0 = call_opaque "__riscv_vsetvl_e32m1"(%arg0) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
    %1 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"float">>
    %2 = literal "0.0f" : !emitc.opaque<"float">
    assign %2 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count"
    %3 = literal "32" : !emitc.opaque<"size_t">
    %4 = div %arg0, %3 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    %5 = literal "2" : !emitc.opaque<"size_t">
    %6 = rem %4, %5 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    %7 = sub %4, %6 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    %8 = literal "0" : !emitc.opaque<"size_t">
    for %arg4 = %8 to %7 step %5  : !emitc.opaque<"size_t"> {
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x"
      %13 = literal "24" : !emitc.opaque<"size_t">
      %14 = mul %arg4, %13 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %15 = add %arg2, %14 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %16 = literal "36" : !emitc.opaque<"size_t">
      %17 = mul %arg4, %16 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %18 = add %arg3, %17 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %19 = call_opaque "(float)*(const _Float16 *)"(%15) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %20 = call_opaque "(float)*(const _Float16 *)"(%18) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      %21 = literal "2" : !emitc.opaque<"size_t">
      %22 = add %15, %21 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %23 = call_opaque "(float)*(const _Float16 *)"(%22) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      %24 = literal "2" : !emitc.opaque<"size_t">
      %25 = add %18, %24 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %26 = call_opaque "(float)*(const _Float16 *)"(%25) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field"
      %27 = literal "4" : !emitc.opaque<"size_t">
      %28 = add %15, %27 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %29 = call_opaque "(uint16_t)*(const uint16_t *)"(%28) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      %30 = literal "6" : !emitc.opaque<"size_t">
      %31 = add %15, %30 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %32 = call_opaque "(uint16_t)*(const uint16_t *)"(%31) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %33 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %34 = literal "0" : !emitc.opaque<"int32_t">
      assign %34 : !emitc.opaque<"int32_t"> to %33 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %35 = literal "16" : !emitc.opaque<"size_t">
      %36 = call_opaque "__riscv_vsetvl_e8m1"(%35) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %37 = literal "16" : !emitc.opaque<"size_t">
      %38 = literal "0" : !emitc.opaque<"size_t">
      for %arg5 = %38 to %37 step %36  : !emitc.opaque<"size_t"> {
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
        %75 = literal "16" : !emitc.opaque<"size_t">
        %76 = sub %75, %arg5 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %77 = call_opaque "__riscv_vsetvl_e8m1"(%76) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %78 = literal "8" : !emitc.opaque<"size_t">
        %79 = add %15, %78 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %80 = add %79, %arg5 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %81 = cast %80 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
        %82 = call_opaque "__riscv_vle8_v_u8m1"(%81, %77) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        %83 = literal "4" : !emitc.opaque<"size_t">
        %84 = add %18, %83 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %85 = add %84, %arg5 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %86 = cast %85 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
        %87 = call_opaque "__riscv_vle8_v_i8m1"(%86, %77) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        %88 = literal "20" : !emitc.opaque<"size_t">
        %89 = add %18, %88 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %90 = add %89, %arg5 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %91 = cast %90 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
        %92 = call_opaque "__riscv_vle8_v_i8m1"(%91, %77) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1"
        %93 = literal "0x0F" : !emitc.opaque<"int">
        %94 = call_opaque "__riscv_vand_vx_u8m1"(%82, %93, %77) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1"
        %95 = literal "0x04" : !emitc.opaque<"int">
        %96 = call_opaque "__riscv_vsrl_vx_u8m1"(%82, %95, %77) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
        %97 = call_opaque "__riscv_vid_v_u16m2"(%77) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
        %98 = call_opaque "__riscv_vadd_vx_u16m2"(%97, %arg5, %77) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
        %99 = call_opaque "__riscv_vmv_v_x_u16m2"(%29, %77) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
        %100 = call_opaque "__riscv_vsrl_vv_u16m2"(%99, %98, %77) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
        %101 = literal "0x1" : !emitc.opaque<"int">
        %102 = call_opaque "__riscv_vand_vx_u16m2"(%100, %101, %77) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
        %103 = literal "0x4" : !emitc.opaque<"int">
        %104 = call_opaque "__riscv_vsll_vx_u16m2"(%102, %103, %77) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
        %105 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%104, %77) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
        %106 = call_opaque "__riscv_vid_v_u16m2"(%77) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
        %107 = call_opaque "__riscv_vadd_vx_u16m2"(%106, %arg5, %77) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
        %108 = call_opaque "__riscv_vmv_v_x_u16m2"(%32, %77) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
        %109 = call_opaque "__riscv_vsrl_vv_u16m2"(%108, %107, %77) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
        %110 = literal "0x1" : !emitc.opaque<"int">
        %111 = call_opaque "__riscv_vand_vx_u16m2"(%109, %110, %77) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
        %112 = literal "0x4" : !emitc.opaque<"int">
        %113 = call_opaque "__riscv_vsll_vx_u16m2"(%111, %112, %77) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
        %114 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%113, %77) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
        %115 = call_opaque "__riscv_vor_vv_u8m1"(%94, %105, %77) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
        %116 = call_opaque "__riscv_vor_vv_u8m1"(%96, %114, %77) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
        %117 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%115) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
        %118 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%116) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
        %119 = call_opaque "__riscv_vwmul_vv_i16m2"(%117, %87, %77) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2"
        %120 = call_opaque "__riscv_vwmacc_vv_i16m2"(%119, %118, %92, %77) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
        %121 = load %33 : <!emitc.opaque<"int32_t">>
        %122 = literal "1" : !emitc.opaque<"size_t">
        %123 = call_opaque "__riscv_vmv_v_x_i32m1"(%121, %122) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
        %124 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%120, %123, %77) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
        %125 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%124) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
        verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
        assign %125 : !emitc.opaque<"int32_t"> to %33 : <!emitc.opaque<"int32_t">>
      }
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x"
      %39 = literal "1" : !emitc.opaque<"size_t">
      %40 = add %arg4, %39 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %41 = literal "24" : !emitc.opaque<"size_t">
      %42 = mul %40, %41 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %43 = add %arg2, %42 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %44 = literal "1" : !emitc.opaque<"size_t">
      %45 = add %arg4, %44 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %46 = literal "36" : !emitc.opaque<"size_t">
      %47 = mul %45, %46 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %48 = add %arg3, %47 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %49 = call_opaque "(float)*(const _Float16 *)"(%43) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %50 = call_opaque "(float)*(const _Float16 *)"(%48) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      %51 = literal "2" : !emitc.opaque<"size_t">
      %52 = add %43, %51 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %53 = call_opaque "(float)*(const _Float16 *)"(%52) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      %54 = literal "2" : !emitc.opaque<"size_t">
      %55 = add %48, %54 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %56 = call_opaque "(float)*(const _Float16 *)"(%55) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field"
      %57 = literal "4" : !emitc.opaque<"size_t">
      %58 = add %43, %57 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %59 = call_opaque "(uint16_t)*(const uint16_t *)"(%58) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      %60 = literal "6" : !emitc.opaque<"size_t">
      %61 = add %43, %60 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %62 = call_opaque "(uint16_t)*(const uint16_t *)"(%61) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %63 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %64 = literal "0" : !emitc.opaque<"int32_t">
      assign %64 : !emitc.opaque<"int32_t"> to %63 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %65 = literal "16" : !emitc.opaque<"size_t">
      %66 = call_opaque "__riscv_vsetvl_e8m1"(%65) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %67 = literal "16" : !emitc.opaque<"size_t">
      %68 = literal "0" : !emitc.opaque<"size_t">
      for %arg5 = %68 to %67 step %66  : !emitc.opaque<"size_t"> {
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
        %75 = literal "16" : !emitc.opaque<"size_t">
        %76 = sub %75, %arg5 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %77 = call_opaque "__riscv_vsetvl_e8m1"(%76) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %78 = literal "8" : !emitc.opaque<"size_t">
        %79 = add %43, %78 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %80 = add %79, %arg5 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %81 = cast %80 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
        %82 = call_opaque "__riscv_vle8_v_u8m1"(%81, %77) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        %83 = literal "4" : !emitc.opaque<"size_t">
        %84 = add %48, %83 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %85 = add %84, %arg5 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %86 = cast %85 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
        %87 = call_opaque "__riscv_vle8_v_i8m1"(%86, %77) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        %88 = literal "20" : !emitc.opaque<"size_t">
        %89 = add %48, %88 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %90 = add %89, %arg5 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %91 = cast %90 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
        %92 = call_opaque "__riscv_vle8_v_i8m1"(%91, %77) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1"
        %93 = literal "0x0F" : !emitc.opaque<"int">
        %94 = call_opaque "__riscv_vand_vx_u8m1"(%82, %93, %77) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1"
        %95 = literal "0x04" : !emitc.opaque<"int">
        %96 = call_opaque "__riscv_vsrl_vx_u8m1"(%82, %95, %77) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
        %97 = call_opaque "__riscv_vid_v_u16m2"(%77) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
        %98 = call_opaque "__riscv_vadd_vx_u16m2"(%97, %arg5, %77) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
        %99 = call_opaque "__riscv_vmv_v_x_u16m2"(%59, %77) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
        %100 = call_opaque "__riscv_vsrl_vv_u16m2"(%99, %98, %77) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
        %101 = literal "0x1" : !emitc.opaque<"int">
        %102 = call_opaque "__riscv_vand_vx_u16m2"(%100, %101, %77) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
        %103 = literal "0x4" : !emitc.opaque<"int">
        %104 = call_opaque "__riscv_vsll_vx_u16m2"(%102, %103, %77) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
        %105 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%104, %77) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
        %106 = call_opaque "__riscv_vid_v_u16m2"(%77) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
        %107 = call_opaque "__riscv_vadd_vx_u16m2"(%106, %arg5, %77) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
        %108 = call_opaque "__riscv_vmv_v_x_u16m2"(%62, %77) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
        %109 = call_opaque "__riscv_vsrl_vv_u16m2"(%108, %107, %77) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
        %110 = literal "0x1" : !emitc.opaque<"int">
        %111 = call_opaque "__riscv_vand_vx_u16m2"(%109, %110, %77) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
        %112 = literal "0x4" : !emitc.opaque<"int">
        %113 = call_opaque "__riscv_vsll_vx_u16m2"(%111, %112, %77) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
        %114 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%113, %77) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
        %115 = call_opaque "__riscv_vor_vv_u8m1"(%94, %105, %77) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
        %116 = call_opaque "__riscv_vor_vv_u8m1"(%96, %114, %77) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
        %117 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%115) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
        %118 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%116) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
        %119 = call_opaque "__riscv_vwmul_vv_i16m2"(%117, %87, %77) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2"
        %120 = call_opaque "__riscv_vwmacc_vv_i16m2"(%119, %118, %92, %77) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
        %121 = load %63 : <!emitc.opaque<"int32_t">>
        %122 = literal "1" : !emitc.opaque<"size_t">
        %123 = call_opaque "__riscv_vmv_v_x_i32m1"(%121, %122) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
        %124 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%120, %123, %77) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
        %125 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%124) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
        verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
        assign %125 : !emitc.opaque<"int32_t"> to %63 : <!emitc.opaque<"int32_t">>
      }
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %69 = load %33 : <!emitc.opaque<"int32_t">>
      %70 = load %1 : <!emitc.opaque<"float">>
      %71 = expression : !emitc.opaque<"float"> {
        %75 = cast %69 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %76 = mul %19, %20 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %77 = mul %76, %75 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %78 = mul %23, %26 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %79 = add %77, %78 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %80 = add %70, %79 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %80 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %71 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %72 = load %63 : <!emitc.opaque<"int32_t">>
      %73 = load %1 : <!emitc.opaque<"float">>
      %74 = expression : !emitc.opaque<"float"> {
        %75 = cast %72 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %76 = mul %49, %50 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %77 = mul %76, %75 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %78 = mul %53, %56 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %79 = add %77, %78 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %80 = add %73, %79 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %80 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %74 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
    }
    %9 = literal "1" : !emitc.opaque<"size_t">
    for %arg4 = %7 to %4 step %9  : !emitc.opaque<"size_t"> {
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x"
      %13 = literal "24" : !emitc.opaque<"size_t">
      %14 = mul %arg4, %13 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %15 = add %arg2, %14 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %16 = literal "36" : !emitc.opaque<"size_t">
      %17 = mul %arg4, %16 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %18 = add %arg3, %17 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %19 = call_opaque "(float)*(const _Float16 *)"(%15) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %20 = call_opaque "(float)*(const _Float16 *)"(%18) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      %21 = literal "2" : !emitc.opaque<"size_t">
      %22 = add %15, %21 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %23 = call_opaque "(float)*(const _Float16 *)"(%22) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      %24 = literal "2" : !emitc.opaque<"size_t">
      %25 = add %18, %24 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %26 = call_opaque "(float)*(const _Float16 *)"(%25) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field"
      %27 = literal "4" : !emitc.opaque<"size_t">
      %28 = add %15, %27 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %29 = call_opaque "(uint16_t)*(const uint16_t *)"(%28) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      %30 = literal "6" : !emitc.opaque<"size_t">
      %31 = add %15, %30 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %32 = call_opaque "(uint16_t)*(const uint16_t *)"(%31) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %33 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %34 = literal "0" : !emitc.opaque<"int32_t">
      assign %34 : !emitc.opaque<"int32_t"> to %33 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %35 = literal "16" : !emitc.opaque<"size_t">
      %36 = call_opaque "__riscv_vsetvl_e8m1"(%35) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %37 = literal "16" : !emitc.opaque<"size_t">
      %38 = literal "0" : !emitc.opaque<"size_t">
      for %arg5 = %38 to %37 step %36  : !emitc.opaque<"size_t"> {
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
        %42 = literal "16" : !emitc.opaque<"size_t">
        %43 = sub %42, %arg5 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %44 = call_opaque "__riscv_vsetvl_e8m1"(%43) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %45 = literal "8" : !emitc.opaque<"size_t">
        %46 = add %15, %45 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %47 = add %46, %arg5 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %48 = cast %47 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
        %49 = call_opaque "__riscv_vle8_v_u8m1"(%48, %44) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        %50 = literal "4" : !emitc.opaque<"size_t">
        %51 = add %18, %50 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %52 = add %51, %arg5 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %53 = cast %52 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
        %54 = call_opaque "__riscv_vle8_v_i8m1"(%53, %44) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        %55 = literal "20" : !emitc.opaque<"size_t">
        %56 = add %18, %55 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %57 = add %56, %arg5 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %58 = cast %57 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
        %59 = call_opaque "__riscv_vle8_v_i8m1"(%58, %44) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1"
        %60 = literal "0x0F" : !emitc.opaque<"int">
        %61 = call_opaque "__riscv_vand_vx_u8m1"(%49, %60, %44) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1"
        %62 = literal "0x04" : !emitc.opaque<"int">
        %63 = call_opaque "__riscv_vsrl_vx_u8m1"(%49, %62, %44) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
        %64 = call_opaque "__riscv_vid_v_u16m2"(%44) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
        %65 = call_opaque "__riscv_vadd_vx_u16m2"(%64, %arg5, %44) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
        %66 = call_opaque "__riscv_vmv_v_x_u16m2"(%29, %44) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
        %67 = call_opaque "__riscv_vsrl_vv_u16m2"(%66, %65, %44) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
        %68 = literal "0x1" : !emitc.opaque<"int">
        %69 = call_opaque "__riscv_vand_vx_u16m2"(%67, %68, %44) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
        %70 = literal "0x4" : !emitc.opaque<"int">
        %71 = call_opaque "__riscv_vsll_vx_u16m2"(%69, %70, %44) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
        %72 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%71, %44) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
        %73 = call_opaque "__riscv_vid_v_u16m2"(%44) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
        %74 = call_opaque "__riscv_vadd_vx_u16m2"(%73, %arg5, %44) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
        %75 = call_opaque "__riscv_vmv_v_x_u16m2"(%32, %44) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
        %76 = call_opaque "__riscv_vsrl_vv_u16m2"(%75, %74, %44) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
        %77 = literal "0x1" : !emitc.opaque<"int">
        %78 = call_opaque "__riscv_vand_vx_u16m2"(%76, %77, %44) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
        %79 = literal "0x4" : !emitc.opaque<"int">
        %80 = call_opaque "__riscv_vsll_vx_u16m2"(%78, %79, %44) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
        %81 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%80, %44) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
        %82 = call_opaque "__riscv_vor_vv_u8m1"(%61, %72, %44) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
        %83 = call_opaque "__riscv_vor_vv_u8m1"(%63, %81, %44) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
        %84 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%82) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
        %85 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%83) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
        %86 = call_opaque "__riscv_vwmul_vv_i16m2"(%84, %54, %44) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2"
        %87 = call_opaque "__riscv_vwmacc_vv_i16m2"(%86, %85, %59, %44) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
        %88 = load %33 : <!emitc.opaque<"int32_t">>
        %89 = literal "1" : !emitc.opaque<"size_t">
        %90 = call_opaque "__riscv_vmv_v_x_i32m1"(%88, %89) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
        %91 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%87, %90, %44) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
        %92 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%91) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
        verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
        assign %92 : !emitc.opaque<"int32_t"> to %33 : <!emitc.opaque<"int32_t">>
      }
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %39 = load %33 : <!emitc.opaque<"int32_t">>
      %40 = load %1 : <!emitc.opaque<"float">>
      %41 = expression : !emitc.opaque<"float"> {
        %42 = cast %39 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %43 = mul %19, %20 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %44 = mul %43, %42 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %45 = mul %23, %26 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %46 = add %44, %45 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %47 = add %40, %46 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %47 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %41 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
    }
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s"
    %10 = literal "0" : index
    %11 = subscript %arg1[%10] : (!emitc.ptr<!emitc.opaque<"float">>, index) -> !emitc.lvalue<!emitc.opaque<"float">>
    %12 = load %1 : <!emitc.opaque<"float">>
    assign %12 : !emitc.opaque<"float"> to %11 : <!emitc.opaque<"float">>
    return
  }
}

