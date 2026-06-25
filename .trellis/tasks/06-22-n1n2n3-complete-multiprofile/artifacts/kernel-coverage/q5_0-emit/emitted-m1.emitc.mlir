module {
  emitc.include <"stddef.h">
  emitc.include <"stdint.h">
  emitc.include <"riscv_vector.h">
  emitc.func @tcrv_emitc_ggml_repack_gemv_q5_0_q8_0_kernel_ggml_repack_gemv_q5_0_q8_0(%arg0: !emitc.opaque<"size_t">, %arg1: !emitc.ptr<!emitc.opaque<"float">>, %arg2: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg3: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg4: !emitc.opaque<"size_t">) attributes {specifiers = ["extern", "\22C\22"]} {
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1"
    %0 = call_opaque "__riscv_vsetvl_e32m1"(%arg0) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface"
    %1 = literal "16" : !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count"
    %2 = literal "32" : !emitc.opaque<"size_t">
    %3 = div %arg0, %2 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=col_group_count"
    %4 = literal "16" : !emitc.opaque<"size_t">
    %5 = div %arg4, %4 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    %6 = literal "4" : !emitc.opaque<"size_t">
    %7 = literal "15" : !emitc.opaque<"size_t">
    %8 = literal "16" : !emitc.opaque<"size_t">
    %9 = literal "1" : !emitc.opaque<"size_t">
    %10 = literal "0" : !emitc.opaque<"size_t">
    for %arg5 = %10 to %5 step %9  : !emitc.opaque<"size_t"> {
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_group_base"
      %14 = mul %arg5, %3 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %15 = literal "352" : !emitc.opaque<"size_t">
      %16 = mul %14, %15 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %17 = add %arg2, %16 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %18 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m4_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4"
      %19 = literal "0.0f" : !emitc.opaque<"float">
      %20 = call_opaque "__riscv_vfmv_v_f_f32m4"(%19, %1) : (!emitc.opaque<"float">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m4_t">
      assign %20 : !emitc.opaque<"vfloat32m4_t"> to %18 : <!emitc.opaque<"vfloat32m4_t">>
      %21 = literal "1" : !emitc.opaque<"size_t">
      %22 = literal "0" : !emitc.opaque<"size_t">
      for %arg6 = %22 to %3 step %21  : !emitc.opaque<"size_t"> {
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base"
        %27 = literal "352" : !emitc.opaque<"size_t">
        %28 = mul %arg6, %27 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %29 = add %17, %28 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base"
        %30 = literal "34" : !emitc.opaque<"size_t">
        %31 = mul %arg6, %30 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %32 = add %arg3, %31 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %33 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m2_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2"
        %34 = literal "0" : !emitc.opaque<"int32_t">
        %35 = call_opaque "__riscv_vmv_v_x_i16m2"(%34, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
        assign %35 : !emitc.opaque<"vint16m2_t"> to %33 : <!emitc.opaque<"vint16m2_t">>
        %36 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m2_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2"
        %37 = literal "0" : !emitc.opaque<"int32_t">
        %38 = call_opaque "__riscv_vmv_v_x_i16m2"(%37, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
        assign %38 : !emitc.opaque<"vint16m2_t"> to %36 : <!emitc.opaque<"vint16m2_t">>
        %39 = literal "1" : !emitc.opaque<"size_t">
        %40 = literal "16" : !emitc.opaque<"size_t">
        %41 = literal "0" : !emitc.opaque<"size_t">
        for %arg7 = %41 to %40 step %39  : !emitc.opaque<"size_t"> {
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr"
          %53 = literal "16" : !emitc.opaque<"size_t">
          %54 = mul %arg7, %53 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %55 = literal "32" : !emitc.opaque<"size_t">
          %56 = add %55, %54 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %57 = add %29, %56 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %58 = cast %57 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
          %59 = call_opaque "__riscv_vle8_v_u8m1"(%58, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_lo_addr"
          %60 = literal "2" : !emitc.opaque<"size_t">
          %61 = mul %arg7, %60 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %62 = literal "288" : !emitc.opaque<"size_t">
          %63 = add %62, %61 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_hi_addr"
          %64 = literal "32" : !emitc.opaque<"size_t">
          %65 = literal "288" : !emitc.opaque<"size_t">
          %66 = add %65, %64 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %67 = add %66, %61 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %68 = add %29, %63 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %69 = cast %68 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint16_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_mask_scalar"
          %70 = call_opaque "(uint16_t)*(const uint16_t *)"(%69) : (!emitc.ptr<!emitc.opaque<"const uint16_t">>) -> !emitc.opaque<"int32_t">
          %71 = add %29, %67 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %72 = cast %71 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint16_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_mask_scalar"
          %73 = call_opaque "(uint16_t)*(const uint16_t *)"(%72) : (!emitc.ptr<!emitc.opaque<"const uint16_t">>) -> !emitc.opaque<"int32_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
          %74 = call_opaque "__riscv_vmv_v_x_u16m2"(%70, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
          %75 = call_opaque "__riscv_vid_v_u16m2"(%1) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
          %76 = call_opaque "__riscv_vsrl_vv_u16m2"(%74, %75, %1) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
          %77 = literal "1" : !emitc.opaque<"size_t">
          %78 = call_opaque "__riscv_vand_vx_u16m2"(%76, %77, %1) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
          %79 = call_opaque "__riscv_vsll_vx_u16m2"(%78, %6, %1) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
          %80 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%79, %1) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1"
          %81 = call_opaque "__riscv_vand_vx_u8m1"(%59, %7, %1) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
          %82 = call_opaque "__riscv_vor_vv_u8m1"(%81, %80, %1) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
          %83 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%82) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1"
          %84 = call_opaque "__riscv_vsub_vx_i8m1"(%83, %8, %1) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2"
          %85 = call_opaque "__riscv_vmv_v_x_u16m2"(%73, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2"
          %86 = call_opaque "__riscv_vid_v_u16m2"(%1) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2"
          %87 = call_opaque "__riscv_vsrl_vv_u16m2"(%85, %86, %1) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2"
          %88 = literal "1" : !emitc.opaque<"size_t">
          %89 = call_opaque "__riscv_vand_vx_u16m2"(%87, %88, %1) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2"
          %90 = call_opaque "__riscv_vsll_vx_u16m2"(%89, %6, %1) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
          %91 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%90, %1) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1"
          %92 = call_opaque "__riscv_vsrl_vx_u8m1"(%59, %6, %1) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1"
          %93 = call_opaque "__riscv_vor_vv_u8m1"(%92, %91, %1) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
          %94 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%93) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1"
          %95 = call_opaque "__riscv_vsub_vx_i8m1"(%94, %8, %1) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo"
          %96 = literal "2" : !emitc.opaque<"size_t">
          %97 = add %96, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %98 = add %32, %97 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %99 = cast %98 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar"
          %100 = call_opaque "*(const int8_t *)"(%99) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi"
          %101 = literal "16" : !emitc.opaque<"size_t">
          %102 = literal "2" : !emitc.opaque<"size_t">
          %103 = add %102, %101 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %104 = add %103, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %105 = add %32, %104 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %106 = cast %105 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar"
          %107 = call_opaque "*(const int8_t *)"(%106) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %108 = load %33 : <!emitc.opaque<"vint16m2_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2"
          %109 = call_opaque "__riscv_vwmacc_vx_i16m2"(%108, %100, %84, %1) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
          assign %109 : !emitc.opaque<"vint16m2_t"> to %33 : <!emitc.opaque<"vint16m2_t">>
          %110 = load %36 : <!emitc.opaque<"vint16m2_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2"
          %111 = call_opaque "__riscv_vwmacc_vx_i16m2"(%110, %107, %95, %1) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
          assign %111 : !emitc.opaque<"vint16m2_t"> to %36 : <!emitc.opaque<"vint16m2_t">>
        }
        %42 = load %33 : <!emitc.opaque<"vint16m2_t">>
        %43 = load %36 : <!emitc.opaque<"vint16m2_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m4"
        %44 = call_opaque "__riscv_vwadd_vv_i32m4"(%42, %43, %1) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m4_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr"
        %45 = cast %29 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const _Float16">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2"
        %46 = call_opaque "__riscv_vle16_v_f16m2"(%45, %1) : (!emitc.ptr<!emitc.opaque<"const _Float16">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat16m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar"
        %47 = cast %32 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const _Float16">>
        %48 = call_opaque "*(const _Float16 *)"(%47) : (!emitc.ptr<!emitc.opaque<"const _Float16">>) -> !emitc.opaque<"_Float16">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4"
        %49 = call_opaque "__riscv_vfwmul_vf_f32m4"(%46, %48, %1) : (!emitc.opaque<"vfloat16m2_t">, !emitc.opaque<"_Float16">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m4_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4"
        %50 = call_opaque "__riscv_vfcvt_f_x_v_f32m4"(%44, %1) : (!emitc.opaque<"vint32m4_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m4_t">
        %51 = load %18 : <!emitc.opaque<"vfloat32m4_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4"
        %52 = call_opaque "__riscv_vfmacc_vv_f32m4"(%51, %50, %49, %1) : (!emitc.opaque<"vfloat32m4_t">, !emitc.opaque<"vfloat32m4_t">, !emitc.opaque<"vfloat32m4_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m4_t">
        assign %52 : !emitc.opaque<"vfloat32m4_t"> to %18 : <!emitc.opaque<"vfloat32m4_t">>
      }
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr"
      %23 = literal "16" : !emitc.opaque<"size_t">
      %24 = mul %arg5, %23 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %25 = add %arg1, %24 : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"float">>
      %26 = load %18 : <!emitc.opaque<"vfloat32m4_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4"
      call_opaque "__riscv_vse32_v_f32m4"(%25, %26, %1) : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"vfloat32m4_t">, !emitc.opaque<"size_t">) -> ()
    }
    %11 = literal "0" : !emitc.opaque<"int32_t">
    %12 = literal "1" : !emitc.opaque<"size_t">
    %13 = call_opaque "__riscv_vmv_v_x_i32m1"(%11, %12) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
    return
  }
}

