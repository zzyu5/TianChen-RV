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
    %5 = literal "4" : !emitc.opaque<"size_t">
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
      %37 = literal "0" : !emitc.opaque<"size_t">
      %38 = literal "8" : !emitc.opaque<"size_t">
      %39 = add %15, %38 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %40 = add %39, %37 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %41 = cast %40 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
      %42 = call_opaque "__riscv_vle8_v_u8m1"(%41, %36) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      %43 = literal "4" : !emitc.opaque<"size_t">
      %44 = add %18, %43 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %45 = add %44, %37 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %46 = cast %45 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %47 = call_opaque "__riscv_vle8_v_i8m1"(%46, %36) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %48 = literal "20" : !emitc.opaque<"size_t">
      %49 = add %18, %48 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %50 = add %49, %37 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %51 = cast %50 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %52 = call_opaque "__riscv_vle8_v_i8m1"(%51, %36) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1"
      %53 = literal "0x0F" : !emitc.opaque<"int">
      %54 = call_opaque "__riscv_vand_vx_u8m1"(%42, %53, %36) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1"
      %55 = literal "0x04" : !emitc.opaque<"int">
      %56 = call_opaque "__riscv_vsrl_vx_u8m1"(%42, %55, %36) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
      %57 = call_opaque "__riscv_vid_v_u16m2"(%36) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
      %58 = call_opaque "__riscv_vadd_vx_u16m2"(%57, %37, %36) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
      %59 = call_opaque "__riscv_vmv_v_x_u16m2"(%29, %36) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
      %60 = call_opaque "__riscv_vsrl_vv_u16m2"(%59, %58, %36) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
      %61 = literal "0x1" : !emitc.opaque<"int">
      %62 = call_opaque "__riscv_vand_vx_u16m2"(%60, %61, %36) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
      %63 = literal "0x4" : !emitc.opaque<"int">
      %64 = call_opaque "__riscv_vsll_vx_u16m2"(%62, %63, %36) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %65 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%64, %36) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
      %66 = call_opaque "__riscv_vid_v_u16m2"(%36) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
      %67 = call_opaque "__riscv_vadd_vx_u16m2"(%66, %37, %36) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
      %68 = call_opaque "__riscv_vmv_v_x_u16m2"(%32, %36) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
      %69 = call_opaque "__riscv_vsrl_vv_u16m2"(%68, %67, %36) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
      %70 = literal "0x1" : !emitc.opaque<"int">
      %71 = call_opaque "__riscv_vand_vx_u16m2"(%69, %70, %36) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
      %72 = literal "0x4" : !emitc.opaque<"int">
      %73 = call_opaque "__riscv_vsll_vx_u16m2"(%71, %72, %36) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %74 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%73, %36) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
      %75 = call_opaque "__riscv_vor_vv_u8m1"(%54, %65, %36) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
      %76 = call_opaque "__riscv_vor_vv_u8m1"(%56, %74, %36) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %77 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%75) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %78 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%76) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %79 = call_opaque "__riscv_vwmul_vv_i16m2"(%77, %47, %36) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2"
      %80 = call_opaque "__riscv_vwmacc_vv_i16m2"(%79, %78, %52, %36) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %81 = literal "0" : !emitc.opaque<"int32_t">
      %82 = literal "1" : !emitc.opaque<"size_t">
      %83 = call_opaque "__riscv_vmv_v_x_i32m1"(%81, %82) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %84 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%80, %83, %36) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %85 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%84) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %85 : !emitc.opaque<"int32_t"> to %33 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x"
      %86 = literal "1" : !emitc.opaque<"size_t">
      %87 = add %arg4, %86 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %88 = literal "24" : !emitc.opaque<"size_t">
      %89 = mul %87, %88 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %90 = add %arg2, %89 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %91 = literal "1" : !emitc.opaque<"size_t">
      %92 = add %arg4, %91 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %93 = literal "36" : !emitc.opaque<"size_t">
      %94 = mul %92, %93 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %95 = add %arg3, %94 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %96 = call_opaque "(float)*(const _Float16 *)"(%90) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %97 = call_opaque "(float)*(const _Float16 *)"(%95) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      %98 = literal "2" : !emitc.opaque<"size_t">
      %99 = add %90, %98 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %100 = call_opaque "(float)*(const _Float16 *)"(%99) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      %101 = literal "2" : !emitc.opaque<"size_t">
      %102 = add %95, %101 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %103 = call_opaque "(float)*(const _Float16 *)"(%102) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field"
      %104 = literal "4" : !emitc.opaque<"size_t">
      %105 = add %90, %104 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %106 = call_opaque "(uint16_t)*(const uint16_t *)"(%105) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      %107 = literal "6" : !emitc.opaque<"size_t">
      %108 = add %90, %107 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %109 = call_opaque "(uint16_t)*(const uint16_t *)"(%108) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %110 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %111 = literal "0" : !emitc.opaque<"int32_t">
      assign %111 : !emitc.opaque<"int32_t"> to %110 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %112 = literal "16" : !emitc.opaque<"size_t">
      %113 = call_opaque "__riscv_vsetvl_e8m1"(%112) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %114 = literal "0" : !emitc.opaque<"size_t">
      %115 = literal "8" : !emitc.opaque<"size_t">
      %116 = add %90, %115 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %117 = add %116, %114 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %118 = cast %117 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
      %119 = call_opaque "__riscv_vle8_v_u8m1"(%118, %113) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      %120 = literal "4" : !emitc.opaque<"size_t">
      %121 = add %95, %120 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %122 = add %121, %114 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %123 = cast %122 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %124 = call_opaque "__riscv_vle8_v_i8m1"(%123, %113) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %125 = literal "20" : !emitc.opaque<"size_t">
      %126 = add %95, %125 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %127 = add %126, %114 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %128 = cast %127 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %129 = call_opaque "__riscv_vle8_v_i8m1"(%128, %113) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1"
      %130 = literal "0x0F" : !emitc.opaque<"int">
      %131 = call_opaque "__riscv_vand_vx_u8m1"(%119, %130, %113) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1"
      %132 = literal "0x04" : !emitc.opaque<"int">
      %133 = call_opaque "__riscv_vsrl_vx_u8m1"(%119, %132, %113) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
      %134 = call_opaque "__riscv_vid_v_u16m2"(%113) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
      %135 = call_opaque "__riscv_vadd_vx_u16m2"(%134, %114, %113) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
      %136 = call_opaque "__riscv_vmv_v_x_u16m2"(%106, %113) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
      %137 = call_opaque "__riscv_vsrl_vv_u16m2"(%136, %135, %113) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
      %138 = literal "0x1" : !emitc.opaque<"int">
      %139 = call_opaque "__riscv_vand_vx_u16m2"(%137, %138, %113) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
      %140 = literal "0x4" : !emitc.opaque<"int">
      %141 = call_opaque "__riscv_vsll_vx_u16m2"(%139, %140, %113) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %142 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%141, %113) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
      %143 = call_opaque "__riscv_vid_v_u16m2"(%113) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
      %144 = call_opaque "__riscv_vadd_vx_u16m2"(%143, %114, %113) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
      %145 = call_opaque "__riscv_vmv_v_x_u16m2"(%109, %113) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
      %146 = call_opaque "__riscv_vsrl_vv_u16m2"(%145, %144, %113) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
      %147 = literal "0x1" : !emitc.opaque<"int">
      %148 = call_opaque "__riscv_vand_vx_u16m2"(%146, %147, %113) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
      %149 = literal "0x4" : !emitc.opaque<"int">
      %150 = call_opaque "__riscv_vsll_vx_u16m2"(%148, %149, %113) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %151 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%150, %113) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
      %152 = call_opaque "__riscv_vor_vv_u8m1"(%131, %142, %113) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
      %153 = call_opaque "__riscv_vor_vv_u8m1"(%133, %151, %113) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %154 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%152) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %155 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%153) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %156 = call_opaque "__riscv_vwmul_vv_i16m2"(%154, %124, %113) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2"
      %157 = call_opaque "__riscv_vwmacc_vv_i16m2"(%156, %155, %129, %113) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %158 = literal "0" : !emitc.opaque<"int32_t">
      %159 = literal "1" : !emitc.opaque<"size_t">
      %160 = call_opaque "__riscv_vmv_v_x_i32m1"(%158, %159) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %161 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%157, %160, %113) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %162 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%161) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %162 : !emitc.opaque<"int32_t"> to %110 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x"
      %163 = literal "2" : !emitc.opaque<"size_t">
      %164 = add %arg4, %163 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %165 = literal "24" : !emitc.opaque<"size_t">
      %166 = mul %164, %165 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %167 = add %arg2, %166 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %168 = literal "2" : !emitc.opaque<"size_t">
      %169 = add %arg4, %168 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %170 = literal "36" : !emitc.opaque<"size_t">
      %171 = mul %169, %170 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %172 = add %arg3, %171 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %173 = call_opaque "(float)*(const _Float16 *)"(%167) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %174 = call_opaque "(float)*(const _Float16 *)"(%172) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      %175 = literal "2" : !emitc.opaque<"size_t">
      %176 = add %167, %175 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %177 = call_opaque "(float)*(const _Float16 *)"(%176) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      %178 = literal "2" : !emitc.opaque<"size_t">
      %179 = add %172, %178 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %180 = call_opaque "(float)*(const _Float16 *)"(%179) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field"
      %181 = literal "4" : !emitc.opaque<"size_t">
      %182 = add %167, %181 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %183 = call_opaque "(uint16_t)*(const uint16_t *)"(%182) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      %184 = literal "6" : !emitc.opaque<"size_t">
      %185 = add %167, %184 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %186 = call_opaque "(uint16_t)*(const uint16_t *)"(%185) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %187 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %188 = literal "0" : !emitc.opaque<"int32_t">
      assign %188 : !emitc.opaque<"int32_t"> to %187 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %189 = literal "16" : !emitc.opaque<"size_t">
      %190 = call_opaque "__riscv_vsetvl_e8m1"(%189) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %191 = literal "0" : !emitc.opaque<"size_t">
      %192 = literal "8" : !emitc.opaque<"size_t">
      %193 = add %167, %192 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %194 = add %193, %191 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %195 = cast %194 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
      %196 = call_opaque "__riscv_vle8_v_u8m1"(%195, %190) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      %197 = literal "4" : !emitc.opaque<"size_t">
      %198 = add %172, %197 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %199 = add %198, %191 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %200 = cast %199 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %201 = call_opaque "__riscv_vle8_v_i8m1"(%200, %190) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %202 = literal "20" : !emitc.opaque<"size_t">
      %203 = add %172, %202 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %204 = add %203, %191 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %205 = cast %204 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %206 = call_opaque "__riscv_vle8_v_i8m1"(%205, %190) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1"
      %207 = literal "0x0F" : !emitc.opaque<"int">
      %208 = call_opaque "__riscv_vand_vx_u8m1"(%196, %207, %190) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1"
      %209 = literal "0x04" : !emitc.opaque<"int">
      %210 = call_opaque "__riscv_vsrl_vx_u8m1"(%196, %209, %190) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
      %211 = call_opaque "__riscv_vid_v_u16m2"(%190) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
      %212 = call_opaque "__riscv_vadd_vx_u16m2"(%211, %191, %190) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
      %213 = call_opaque "__riscv_vmv_v_x_u16m2"(%183, %190) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
      %214 = call_opaque "__riscv_vsrl_vv_u16m2"(%213, %212, %190) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
      %215 = literal "0x1" : !emitc.opaque<"int">
      %216 = call_opaque "__riscv_vand_vx_u16m2"(%214, %215, %190) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
      %217 = literal "0x4" : !emitc.opaque<"int">
      %218 = call_opaque "__riscv_vsll_vx_u16m2"(%216, %217, %190) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %219 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%218, %190) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
      %220 = call_opaque "__riscv_vid_v_u16m2"(%190) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
      %221 = call_opaque "__riscv_vadd_vx_u16m2"(%220, %191, %190) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
      %222 = call_opaque "__riscv_vmv_v_x_u16m2"(%186, %190) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
      %223 = call_opaque "__riscv_vsrl_vv_u16m2"(%222, %221, %190) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
      %224 = literal "0x1" : !emitc.opaque<"int">
      %225 = call_opaque "__riscv_vand_vx_u16m2"(%223, %224, %190) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
      %226 = literal "0x4" : !emitc.opaque<"int">
      %227 = call_opaque "__riscv_vsll_vx_u16m2"(%225, %226, %190) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %228 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%227, %190) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
      %229 = call_opaque "__riscv_vor_vv_u8m1"(%208, %219, %190) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
      %230 = call_opaque "__riscv_vor_vv_u8m1"(%210, %228, %190) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %231 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%229) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %232 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%230) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %233 = call_opaque "__riscv_vwmul_vv_i16m2"(%231, %201, %190) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2"
      %234 = call_opaque "__riscv_vwmacc_vv_i16m2"(%233, %232, %206, %190) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %235 = literal "0" : !emitc.opaque<"int32_t">
      %236 = literal "1" : !emitc.opaque<"size_t">
      %237 = call_opaque "__riscv_vmv_v_x_i32m1"(%235, %236) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %238 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%234, %237, %190) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %239 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%238) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %239 : !emitc.opaque<"int32_t"> to %187 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x"
      %240 = literal "3" : !emitc.opaque<"size_t">
      %241 = add %arg4, %240 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %242 = literal "24" : !emitc.opaque<"size_t">
      %243 = mul %241, %242 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %244 = add %arg2, %243 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %245 = literal "3" : !emitc.opaque<"size_t">
      %246 = add %arg4, %245 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %247 = literal "36" : !emitc.opaque<"size_t">
      %248 = mul %246, %247 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %249 = add %arg3, %248 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %250 = call_opaque "(float)*(const _Float16 *)"(%244) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %251 = call_opaque "(float)*(const _Float16 *)"(%249) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      %252 = literal "2" : !emitc.opaque<"size_t">
      %253 = add %244, %252 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %254 = call_opaque "(float)*(const _Float16 *)"(%253) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      %255 = literal "2" : !emitc.opaque<"size_t">
      %256 = add %249, %255 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %257 = call_opaque "(float)*(const _Float16 *)"(%256) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field"
      %258 = literal "4" : !emitc.opaque<"size_t">
      %259 = add %244, %258 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %260 = call_opaque "(uint16_t)*(const uint16_t *)"(%259) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      %261 = literal "6" : !emitc.opaque<"size_t">
      %262 = add %244, %261 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %263 = call_opaque "(uint16_t)*(const uint16_t *)"(%262) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"uint32_t">
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %264 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %265 = literal "0" : !emitc.opaque<"int32_t">
      assign %265 : !emitc.opaque<"int32_t"> to %264 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %266 = literal "16" : !emitc.opaque<"size_t">
      %267 = call_opaque "__riscv_vsetvl_e8m1"(%266) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %268 = literal "0" : !emitc.opaque<"size_t">
      %269 = literal "8" : !emitc.opaque<"size_t">
      %270 = add %244, %269 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %271 = add %270, %268 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %272 = cast %271 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
      %273 = call_opaque "__riscv_vle8_v_u8m1"(%272, %267) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      %274 = literal "4" : !emitc.opaque<"size_t">
      %275 = add %249, %274 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %276 = add %275, %268 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %277 = cast %276 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %278 = call_opaque "__riscv_vle8_v_i8m1"(%277, %267) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %279 = literal "20" : !emitc.opaque<"size_t">
      %280 = add %249, %279 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %281 = add %280, %268 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %282 = cast %281 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %283 = call_opaque "__riscv_vle8_v_i8m1"(%282, %267) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1"
      %284 = literal "0x0F" : !emitc.opaque<"int">
      %285 = call_opaque "__riscv_vand_vx_u8m1"(%273, %284, %267) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1"
      %286 = literal "0x04" : !emitc.opaque<"int">
      %287 = call_opaque "__riscv_vsrl_vx_u8m1"(%273, %286, %267) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
      %288 = call_opaque "__riscv_vid_v_u16m2"(%267) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
      %289 = call_opaque "__riscv_vadd_vx_u16m2"(%288, %268, %267) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
      %290 = call_opaque "__riscv_vmv_v_x_u16m2"(%260, %267) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
      %291 = call_opaque "__riscv_vsrl_vv_u16m2"(%290, %289, %267) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
      %292 = literal "0x1" : !emitc.opaque<"int">
      %293 = call_opaque "__riscv_vand_vx_u16m2"(%291, %292, %267) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
      %294 = literal "0x4" : !emitc.opaque<"int">
      %295 = call_opaque "__riscv_vsll_vx_u16m2"(%293, %294, %267) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %296 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%295, %267) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
      %297 = call_opaque "__riscv_vid_v_u16m2"(%267) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2"
      %298 = call_opaque "__riscv_vadd_vx_u16m2"(%297, %268, %267) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
      %299 = call_opaque "__riscv_vmv_v_x_u16m2"(%263, %267) : (!emitc.opaque<"uint32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
      %300 = call_opaque "__riscv_vsrl_vv_u16m2"(%299, %298, %267) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
      %301 = literal "0x1" : !emitc.opaque<"int">
      %302 = call_opaque "__riscv_vand_vx_u16m2"(%300, %301, %267) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
      %303 = literal "0x4" : !emitc.opaque<"int">
      %304 = call_opaque "__riscv_vsll_vx_u16m2"(%302, %303, %267) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %305 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%304, %267) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
      %306 = call_opaque "__riscv_vor_vv_u8m1"(%285, %296, %267) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
      %307 = call_opaque "__riscv_vor_vv_u8m1"(%287, %305, %267) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %308 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%306) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %309 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%307) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %310 = call_opaque "__riscv_vwmul_vv_i16m2"(%308, %278, %267) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2"
      %311 = call_opaque "__riscv_vwmacc_vv_i16m2"(%310, %309, %283, %267) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %312 = literal "0" : !emitc.opaque<"int32_t">
      %313 = literal "1" : !emitc.opaque<"size_t">
      %314 = call_opaque "__riscv_vmv_v_x_i32m1"(%312, %313) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %315 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%311, %314, %267) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %316 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%315) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %316 : !emitc.opaque<"int32_t"> to %264 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %317 = load %33 : <!emitc.opaque<"int32_t">>
      %318 = load %1 : <!emitc.opaque<"float">>
      %319 = expression : !emitc.opaque<"float"> {
        %329 = cast %317 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %330 = mul %19, %20 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %331 = mul %330, %329 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %332 = mul %23, %26 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %333 = add %331, %332 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %334 = add %318, %333 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %334 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %319 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %320 = load %110 : <!emitc.opaque<"int32_t">>
      %321 = load %1 : <!emitc.opaque<"float">>
      %322 = expression : !emitc.opaque<"float"> {
        %329 = cast %320 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %330 = mul %96, %97 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %331 = mul %330, %329 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %332 = mul %100, %103 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %333 = add %331, %332 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %334 = add %321, %333 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %334 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %322 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %323 = load %187 : <!emitc.opaque<"int32_t">>
      %324 = load %1 : <!emitc.opaque<"float">>
      %325 = expression : !emitc.opaque<"float"> {
        %329 = cast %323 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %330 = mul %173, %174 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %331 = mul %330, %329 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %332 = mul %177, %180 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %333 = add %331, %332 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %334 = add %324, %333 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %334 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %325 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %326 = load %264 : <!emitc.opaque<"int32_t">>
      %327 = load %1 : <!emitc.opaque<"float">>
      %328 = expression : !emitc.opaque<"float"> {
        %329 = cast %326 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %330 = mul %250, %251 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %331 = mul %330, %329 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %332 = mul %254, %257 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %333 = add %331, %332 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %334 = add %327, %333 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %334 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %328 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
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

