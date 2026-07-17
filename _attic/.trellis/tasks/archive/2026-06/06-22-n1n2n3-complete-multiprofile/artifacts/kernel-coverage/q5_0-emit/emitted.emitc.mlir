module {
  emitc.include <"stddef.h">
  emitc.include <"stdint.h">
  emitc.include <"riscv_vector.h">
  emitc.func @tcrv_emitc_ggml_repack_gemv_q5_0_q8_0_kernel_ggml_repack_gemv_q5_0_q8_0(%arg0: !emitc.opaque<"size_t">, %arg1: !emitc.ptr<!emitc.opaque<"float">>, %arg2: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg3: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg4: !emitc.opaque<"size_t">) attributes {specifiers = ["extern", "\22C\22"]} {
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1"
    %0 = call_opaque "__riscv_vsetvl_e32m1"(%arg0) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface"
    %1 = literal "8" : !emitc.opaque<"size_t">
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
      %18 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m2_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2"
      %19 = literal "0.0f" : !emitc.opaque<"float">
      %20 = call_opaque "__riscv_vfmv_v_f_f32m2"(%19, %1) : (!emitc.opaque<"float">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
      assign %20 : !emitc.opaque<"vfloat32m2_t"> to %18 : <!emitc.opaque<"vfloat32m2_t">>
      %21 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m2_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2"
      %22 = literal "0.0f" : !emitc.opaque<"float">
      %23 = call_opaque "__riscv_vfmv_v_f_f32m2"(%22, %1) : (!emitc.opaque<"float">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
      assign %23 : !emitc.opaque<"vfloat32m2_t"> to %21 : <!emitc.opaque<"vfloat32m2_t">>
      %24 = literal "1" : !emitc.opaque<"size_t">
      %25 = literal "0" : !emitc.opaque<"size_t">
      for %arg6 = %25 to %3 step %24  : !emitc.opaque<"size_t"> {
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base"
        %36 = literal "352" : !emitc.opaque<"size_t">
        %37 = mul %arg6, %36 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %38 = add %17, %37 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base"
        %39 = literal "34" : !emitc.opaque<"size_t">
        %40 = mul %arg6, %39 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %41 = add %arg3, %40 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %42 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %43 = literal "0" : !emitc.opaque<"int32_t">
        %44 = call_opaque "__riscv_vmv_v_x_i16m1"(%43, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %44 : !emitc.opaque<"vint16m1_t"> to %42 : <!emitc.opaque<"vint16m1_t">>
        %45 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %46 = literal "0" : !emitc.opaque<"int32_t">
        %47 = call_opaque "__riscv_vmv_v_x_i16m1"(%46, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %47 : !emitc.opaque<"vint16m1_t"> to %45 : <!emitc.opaque<"vint16m1_t">>
        %48 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %49 = literal "0" : !emitc.opaque<"int32_t">
        %50 = call_opaque "__riscv_vmv_v_x_i16m1"(%49, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %50 : !emitc.opaque<"vint16m1_t"> to %48 : <!emitc.opaque<"vint16m1_t">>
        %51 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %52 = literal "0" : !emitc.opaque<"int32_t">
        %53 = call_opaque "__riscv_vmv_v_x_i16m1"(%52, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %53 : !emitc.opaque<"vint16m1_t"> to %51 : <!emitc.opaque<"vint16m1_t">>
        %54 = literal "1" : !emitc.opaque<"size_t">
        %55 = literal "16" : !emitc.opaque<"size_t">
        %56 = literal "0" : !emitc.opaque<"size_t">
        for %arg7 = %56 to %55 step %54  : !emitc.opaque<"size_t"> {
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr"
          %79 = literal "16" : !emitc.opaque<"size_t">
          %80 = mul %arg7, %79 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %81 = literal "32" : !emitc.opaque<"size_t">
          %82 = add %81, %80 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %83 = literal "8" : !emitc.opaque<"size_t">
          %84 = add %82, %83 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %85 = add %38, %82 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %86 = cast %85 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %87 = call_opaque "__riscv_vle8_v_u8mf2"(%86, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          %88 = add %38, %84 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %89 = cast %88 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %90 = call_opaque "__riscv_vle8_v_u8mf2"(%89, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_lo_addr"
          %91 = literal "2" : !emitc.opaque<"size_t">
          %92 = mul %arg7, %91 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %93 = literal "288" : !emitc.opaque<"size_t">
          %94 = add %93, %92 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_hi_addr"
          %95 = literal "32" : !emitc.opaque<"size_t">
          %96 = literal "288" : !emitc.opaque<"size_t">
          %97 = add %96, %95 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %98 = add %97, %92 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %99 = add %38, %94 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %100 = cast %99 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint16_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_mask_scalar"
          %101 = call_opaque "(uint16_t)*(const uint16_t *)"(%100) : (!emitc.ptr<!emitc.opaque<"const uint16_t">>) -> !emitc.opaque<"int32_t">
          %102 = add %38, %98 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %103 = cast %102 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint16_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_mask_scalar"
          %104 = call_opaque "(uint16_t)*(const uint16_t *)"(%103) : (!emitc.ptr<!emitc.opaque<"const uint16_t">>) -> !emitc.opaque<"int32_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m1"
          %105 = call_opaque "__riscv_vmv_v_x_u16m1"(%101, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m1"
          %106 = call_opaque "__riscv_vid_v_u16m1"(%1) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m1"
          %107 = call_opaque "__riscv_vsrl_vv_u16m1"(%105, %106, %1) : (!emitc.opaque<"vuint16m1_t">, !emitc.opaque<"vuint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m1"
          %108 = literal "1" : !emitc.opaque<"size_t">
          %109 = call_opaque "__riscv_vand_vx_u16m1"(%107, %108, %1) : (!emitc.opaque<"vuint16m1_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m1"
          %110 = call_opaque "__riscv_vsll_vx_u16m1"(%109, %6, %1) : (!emitc.opaque<"vuint16m1_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf2"
          %111 = call_opaque "__riscv_vncvt_x_x_w_u8mf2"(%110, %1) : (!emitc.opaque<"vuint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %112 = call_opaque "__riscv_vand_vx_u8mf2"(%87, %7, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %113 = call_opaque "__riscv_vor_vv_u8mf2"(%112, %111, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %114 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%113) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2"
          %115 = call_opaque "__riscv_vsub_vx_i8mf2"(%114, %8, %1) : (!emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m1"
          %116 = call_opaque "__riscv_vmv_v_x_u16m1"(%104, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m1"
          %117 = call_opaque "__riscv_vid_v_u16m1"(%1) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m1"
          %118 = call_opaque "__riscv_vsrl_vv_u16m1"(%116, %117, %1) : (!emitc.opaque<"vuint16m1_t">, !emitc.opaque<"vuint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m1"
          %119 = literal "1" : !emitc.opaque<"size_t">
          %120 = call_opaque "__riscv_vand_vx_u16m1"(%118, %119, %1) : (!emitc.opaque<"vuint16m1_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m1"
          %121 = call_opaque "__riscv_vsll_vx_u16m1"(%120, %6, %1) : (!emitc.opaque<"vuint16m1_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf2"
          %122 = call_opaque "__riscv_vncvt_x_x_w_u8mf2"(%121, %1) : (!emitc.opaque<"vuint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %123 = call_opaque "__riscv_vsrl_vx_u8mf2"(%87, %6, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %124 = call_opaque "__riscv_vor_vv_u8mf2"(%123, %122, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %125 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%124) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2"
          %126 = call_opaque "__riscv_vsub_vx_i8mf2"(%125, %8, %1) : (!emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m1"
          %127 = call_opaque "__riscv_vmv_v_x_u16m1"(%101, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m1"
          %128 = call_opaque "__riscv_vid_v_u16m1"(%1) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m1"
          %129 = literal "8" : !emitc.opaque<"size_t">
          %130 = call_opaque "__riscv_vadd_vx_u16m1"(%128, %129, %1) : (!emitc.opaque<"vuint16m1_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m1"
          %131 = call_opaque "__riscv_vsrl_vv_u16m1"(%127, %130, %1) : (!emitc.opaque<"vuint16m1_t">, !emitc.opaque<"vuint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m1"
          %132 = literal "1" : !emitc.opaque<"size_t">
          %133 = call_opaque "__riscv_vand_vx_u16m1"(%131, %132, %1) : (!emitc.opaque<"vuint16m1_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m1"
          %134 = call_opaque "__riscv_vsll_vx_u16m1"(%133, %6, %1) : (!emitc.opaque<"vuint16m1_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf2"
          %135 = call_opaque "__riscv_vncvt_x_x_w_u8mf2"(%134, %1) : (!emitc.opaque<"vuint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %136 = call_opaque "__riscv_vand_vx_u8mf2"(%90, %7, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %137 = call_opaque "__riscv_vor_vv_u8mf2"(%136, %135, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %138 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%137) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2"
          %139 = call_opaque "__riscv_vsub_vx_i8mf2"(%138, %8, %1) : (!emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m1"
          %140 = call_opaque "__riscv_vmv_v_x_u16m1"(%104, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m1"
          %141 = call_opaque "__riscv_vid_v_u16m1"(%1) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m1"
          %142 = literal "8" : !emitc.opaque<"size_t">
          %143 = call_opaque "__riscv_vadd_vx_u16m1"(%141, %142, %1) : (!emitc.opaque<"vuint16m1_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m1"
          %144 = call_opaque "__riscv_vsrl_vv_u16m1"(%140, %143, %1) : (!emitc.opaque<"vuint16m1_t">, !emitc.opaque<"vuint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m1"
          %145 = literal "1" : !emitc.opaque<"size_t">
          %146 = call_opaque "__riscv_vand_vx_u16m1"(%144, %145, %1) : (!emitc.opaque<"vuint16m1_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m1"
          %147 = call_opaque "__riscv_vsll_vx_u16m1"(%146, %6, %1) : (!emitc.opaque<"vuint16m1_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf2"
          %148 = call_opaque "__riscv_vncvt_x_x_w_u8mf2"(%147, %1) : (!emitc.opaque<"vuint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %149 = call_opaque "__riscv_vsrl_vx_u8mf2"(%90, %6, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %150 = call_opaque "__riscv_vor_vv_u8mf2"(%149, %148, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %151 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%150) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2"
          %152 = call_opaque "__riscv_vsub_vx_i8mf2"(%151, %8, %1) : (!emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo"
          %153 = literal "2" : !emitc.opaque<"size_t">
          %154 = add %153, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %155 = add %41, %154 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %156 = cast %155 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar"
          %157 = call_opaque "*(const int8_t *)"(%156) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi"
          %158 = literal "16" : !emitc.opaque<"size_t">
          %159 = literal "2" : !emitc.opaque<"size_t">
          %160 = add %159, %158 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %161 = add %160, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %162 = add %41, %161 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %163 = cast %162 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar"
          %164 = call_opaque "*(const int8_t *)"(%163) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %165 = load %42 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %166 = call_opaque "__riscv_vwmacc_vx_i16m1"(%165, %157, %115, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %166 : !emitc.opaque<"vint16m1_t"> to %42 : <!emitc.opaque<"vint16m1_t">>
          %167 = load %45 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %168 = call_opaque "__riscv_vwmacc_vx_i16m1"(%167, %164, %126, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %168 : !emitc.opaque<"vint16m1_t"> to %45 : <!emitc.opaque<"vint16m1_t">>
          %169 = load %48 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %170 = call_opaque "__riscv_vwmacc_vx_i16m1"(%169, %157, %139, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %170 : !emitc.opaque<"vint16m1_t"> to %48 : <!emitc.opaque<"vint16m1_t">>
          %171 = load %51 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %172 = call_opaque "__riscv_vwmacc_vx_i16m1"(%171, %164, %152, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %172 : !emitc.opaque<"vint16m1_t"> to %51 : <!emitc.opaque<"vint16m1_t">>
        }
        %57 = load %42 : <!emitc.opaque<"vint16m1_t">>
        %58 = load %45 : <!emitc.opaque<"vint16m1_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2"
        %59 = call_opaque "__riscv_vwadd_vv_i32m2"(%57, %58, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        %60 = load %48 : <!emitc.opaque<"vint16m1_t">>
        %61 = load %51 : <!emitc.opaque<"vint16m1_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2"
        %62 = call_opaque "__riscv_vwadd_vv_i32m2"(%60, %61, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr"
        %63 = cast %38 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const _Float16">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1"
        %64 = call_opaque "__riscv_vle16_v_f16m1"(%63, %1) : (!emitc.ptr<!emitc.opaque<"const _Float16">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat16m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr"
        %65 = literal "16" : !emitc.opaque<"size_t">
        %66 = add %38, %65 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %67 = cast %66 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const _Float16">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1"
        %68 = call_opaque "__riscv_vle16_v_f16m1"(%67, %1) : (!emitc.ptr<!emitc.opaque<"const _Float16">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat16m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar"
        %69 = cast %41 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const _Float16">>
        %70 = call_opaque "*(const _Float16 *)"(%69) : (!emitc.ptr<!emitc.opaque<"const _Float16">>) -> !emitc.opaque<"_Float16">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2"
        %71 = call_opaque "__riscv_vfwmul_vf_f32m2"(%64, %70, %1) : (!emitc.opaque<"vfloat16m1_t">, !emitc.opaque<"_Float16">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2"
        %72 = call_opaque "__riscv_vfcvt_f_x_v_f32m2"(%59, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        %73 = load %18 : <!emitc.opaque<"vfloat32m2_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2"
        %74 = call_opaque "__riscv_vfmacc_vv_f32m2"(%73, %72, %71, %1) : (!emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        assign %74 : !emitc.opaque<"vfloat32m2_t"> to %18 : <!emitc.opaque<"vfloat32m2_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2"
        %75 = call_opaque "__riscv_vfwmul_vf_f32m2"(%68, %70, %1) : (!emitc.opaque<"vfloat16m1_t">, !emitc.opaque<"_Float16">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2"
        %76 = call_opaque "__riscv_vfcvt_f_x_v_f32m2"(%62, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        %77 = load %21 : <!emitc.opaque<"vfloat32m2_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2"
        %78 = call_opaque "__riscv_vfmacc_vv_f32m2"(%77, %76, %75, %1) : (!emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        assign %78 : !emitc.opaque<"vfloat32m2_t"> to %21 : <!emitc.opaque<"vfloat32m2_t">>
      }
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr"
      %26 = literal "16" : !emitc.opaque<"size_t">
      %27 = mul %arg5, %26 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %28 = add %arg1, %27 : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"float">>
      %29 = load %18 : <!emitc.opaque<"vfloat32m2_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2"
      call_opaque "__riscv_vse32_v_f32m2"(%28, %29, %1) : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr"
      %30 = literal "16" : !emitc.opaque<"size_t">
      %31 = mul %arg5, %30 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %32 = literal "8" : !emitc.opaque<"size_t">
      %33 = add %31, %32 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %34 = add %arg1, %33 : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"float">>
      %35 = load %21 : <!emitc.opaque<"vfloat32m2_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2"
      call_opaque "__riscv_vse32_v_f32m2"(%34, %35, %1) : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> ()
    }
    %11 = literal "0" : !emitc.opaque<"int32_t">
    %12 = literal "1" : !emitc.opaque<"size_t">
    %13 = call_opaque "__riscv_vmv_v_x_i32m1"(%11, %12) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
    return
  }
}

