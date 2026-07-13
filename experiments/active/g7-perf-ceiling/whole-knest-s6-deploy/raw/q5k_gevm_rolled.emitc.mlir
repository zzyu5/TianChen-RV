module {
  emitc.include <"stddef.h">
  emitc.include <"stdint.h">
  emitc.include <"riscv_vector.h">
  emitc.func @weft_emitc_ggml_repack_gemv_q5_K_q8_K_kernel_ggml_repack_gemv_q5_K_q8_K(%arg0: !emitc.opaque<"size_t">, %arg1: !emitc.ptr<!emitc.opaque<"float">>, %arg2: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg3: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg4: !emitc.opaque<"size_t">) attributes {specifiers = ["extern", "\22C\22"]} {
    verbatim "// weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface"
    verbatim "// weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1"
    %0 = call_opaque "__riscv_vsetvl_e32m1"(%arg0) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// weft_emitc.route_source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface"
    %1 = literal "8" : !emitc.opaque<"size_t">
    verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_count"
    %2 = literal "256" : !emitc.opaque<"size_t">
    %3 = div %arg0, %2 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=col_group_count"
    %4 = literal "16" : !emitc.opaque<"size_t">
    %5 = div %arg4, %4 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    %6 = literal "1" : !emitc.opaque<"size_t">
    %7 = literal "0" : !emitc.opaque<"size_t">
    for %arg5 = %7 to %5 step %6  : !emitc.opaque<"size_t"> {
      verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_group_base"
      %8 = mul %arg5, %3 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %9 = literal "2816" : !emitc.opaque<"size_t">
      %10 = mul %8, %9 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %11 = add %arg2, %10 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %12 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m2_t">>
      verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2"
      %13 = literal "0.0f" : !emitc.opaque<"float">
      %14 = call_opaque "__riscv_vfmv_v_f_f32m2"(%13, %1) : (!emitc.opaque<"float">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
      assign %14 : !emitc.opaque<"vfloat32m2_t"> to %12 : <!emitc.opaque<"vfloat32m2_t">>
      %15 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m2_t">>
      verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2"
      %16 = literal "0.0f" : !emitc.opaque<"float">
      %17 = call_opaque "__riscv_vfmv_v_f_f32m2"(%16, %1) : (!emitc.opaque<"float">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
      assign %17 : !emitc.opaque<"vfloat32m2_t"> to %15 : <!emitc.opaque<"vfloat32m2_t">>
      %18 = literal "1" : !emitc.opaque<"size_t">
      %19 = literal "0" : !emitc.opaque<"size_t">
      for %arg6 = %19 to %3 step %18  : !emitc.opaque<"size_t"> {
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base"
        %30 = literal "2816" : !emitc.opaque<"size_t">
        %31 = mul %arg6, %30 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %32 = add %11, %31 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base"
        %33 = literal "292" : !emitc.opaque<"size_t">
        %34 = mul %arg6, %33 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %35 = add %arg3, %34 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar"
        %36 = cast %35 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const float">>
        %37 = call_opaque "*(const float *)"(%36) : (!emitc.ptr<!emitc.opaque<"const float">>) -> !emitc.opaque<"float">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr"
        %38 = literal "32" : !emitc.opaque<"size_t">
        %39 = add %32, %38 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %40 = cast %39 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const _Float16">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1"
        %41 = call_opaque "__riscv_vle16_v_f16m1"(%40, %1) : (!emitc.ptr<!emitc.opaque<"const _Float16">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2"
        %42 = call_opaque "__riscv_vfwcvt_f_f_v_f32m2"(%41, %1) : (!emitc.opaque<"vfloat16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2"
        %43 = call_opaque "__riscv_vfmul_vf_f32m2"(%42, %37, %1) : (!emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"float">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr"
        %44 = literal "48" : !emitc.opaque<"size_t">
        %45 = add %32, %44 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %46 = cast %45 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const _Float16">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1"
        %47 = call_opaque "__riscv_vle16_v_f16m1"(%46, %1) : (!emitc.ptr<!emitc.opaque<"const _Float16">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2"
        %48 = call_opaque "__riscv_vfwcvt_f_f_v_f32m2"(%47, %1) : (!emitc.opaque<"vfloat16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2"
        %49 = call_opaque "__riscv_vfmul_vf_f32m2"(%48, %37, %1) : (!emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"float">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        %50 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2"
        %51 = literal "0" : !emitc.opaque<"int32_t">
        %52 = call_opaque "__riscv_vmv_v_x_i32m2"(%51, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %52 : !emitc.opaque<"vint32m2_t"> to %50 : <!emitc.opaque<"vint32m2_t">>
        %53 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2"
        %54 = literal "0" : !emitc.opaque<"int32_t">
        %55 = call_opaque "__riscv_vmv_v_x_i32m2"(%54, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %55 : !emitc.opaque<"vint32m2_t"> to %53 : <!emitc.opaque<"vint32m2_t">>
        %56 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2"
        %57 = literal "0" : !emitc.opaque<"int32_t">
        %58 = call_opaque "__riscv_vmv_v_x_i32m2"(%57, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %58 : !emitc.opaque<"vint32m2_t"> to %56 : <!emitc.opaque<"vint32m2_t">>
        %59 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2"
        %60 = literal "0" : !emitc.opaque<"int32_t">
        %61 = call_opaque "__riscv_vmv_v_x_i32m2"(%60, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %61 : !emitc.opaque<"vint32m2_t"> to %59 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_min_unpack_superhalf"
        %62 = literal "64" : !emitc.opaque<"size_t">
        %63 = add %32, %62 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %64 = cast %63 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %65 = call_opaque "__riscv_vle8_v_u8mf2"(%64, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        %66 = literal "192" : !emitc.opaque<"size_t">
        %67 = add %32, %66 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %68 = cast %67 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %69 = call_opaque "__riscv_vle8_v_u8mf2"(%68, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %70 = literal "0x0F" : !emitc.opaque<"int">
        %71 = call_opaque "__riscv_vand_vx_u8mf2"(%65, %70, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %72 = literal "4" : !emitc.opaque<"int">
        %73 = call_opaque "__riscv_vsrl_vx_u8mf2"(%65, %72, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %74 = literal "0x03" : !emitc.opaque<"int">
        %75 = call_opaque "__riscv_vand_vx_u8mf2"(%69, %74, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
        %76 = literal "4" : !emitc.opaque<"int">
        %77 = call_opaque "__riscv_vsll_vx_u8mf2"(%75, %76, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %78 = literal "0x0C" : !emitc.opaque<"int">
        %79 = call_opaque "__riscv_vand_vx_u8mf2"(%69, %78, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
        %80 = literal "2" : !emitc.opaque<"int">
        %81 = call_opaque "__riscv_vsll_vx_u8mf2"(%79, %80, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %82 = call_opaque "__riscv_vor_vv_u8mf2"(%77, %71, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %83 = call_opaque "__riscv_vor_vv_u8mf2"(%81, %73, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %84 = call_opaque "__riscv_vzext_vf2_u16m1"(%82, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %85 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%84) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %86 = call_opaque "__riscv_vzext_vf2_u16m1"(%83, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %87 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%86) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        %88 = literal "80" : !emitc.opaque<"size_t">
        %89 = add %32, %88 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %90 = cast %89 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %91 = call_opaque "__riscv_vle8_v_u8mf2"(%90, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        %92 = literal "208" : !emitc.opaque<"size_t">
        %93 = add %32, %92 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %94 = cast %93 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %95 = call_opaque "__riscv_vle8_v_u8mf2"(%94, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %96 = literal "0x0F" : !emitc.opaque<"int">
        %97 = call_opaque "__riscv_vand_vx_u8mf2"(%91, %96, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %98 = literal "4" : !emitc.opaque<"int">
        %99 = call_opaque "__riscv_vsrl_vx_u8mf2"(%91, %98, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %100 = literal "0x03" : !emitc.opaque<"int">
        %101 = call_opaque "__riscv_vand_vx_u8mf2"(%95, %100, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
        %102 = literal "4" : !emitc.opaque<"int">
        %103 = call_opaque "__riscv_vsll_vx_u8mf2"(%101, %102, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %104 = literal "0x0C" : !emitc.opaque<"int">
        %105 = call_opaque "__riscv_vand_vx_u8mf2"(%95, %104, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
        %106 = literal "2" : !emitc.opaque<"int">
        %107 = call_opaque "__riscv_vsll_vx_u8mf2"(%105, %106, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %108 = call_opaque "__riscv_vor_vv_u8mf2"(%103, %97, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %109 = call_opaque "__riscv_vor_vv_u8mf2"(%107, %99, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %110 = call_opaque "__riscv_vzext_vf2_u16m1"(%108, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %111 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%110) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %112 = call_opaque "__riscv_vzext_vf2_u16m1"(%109, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %113 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%112) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        %114 = literal "96" : !emitc.opaque<"size_t">
        %115 = add %32, %114 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %116 = cast %115 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %117 = call_opaque "__riscv_vle8_v_u8mf2"(%116, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        %118 = literal "224" : !emitc.opaque<"size_t">
        %119 = add %32, %118 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %120 = cast %119 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %121 = call_opaque "__riscv_vle8_v_u8mf2"(%120, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %122 = literal "0x0F" : !emitc.opaque<"int">
        %123 = call_opaque "__riscv_vand_vx_u8mf2"(%117, %122, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %124 = literal "4" : !emitc.opaque<"int">
        %125 = call_opaque "__riscv_vsrl_vx_u8mf2"(%117, %124, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %126 = literal "0x03" : !emitc.opaque<"int">
        %127 = call_opaque "__riscv_vand_vx_u8mf2"(%121, %126, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
        %128 = literal "4" : !emitc.opaque<"int">
        %129 = call_opaque "__riscv_vsll_vx_u8mf2"(%127, %128, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %130 = literal "0x0C" : !emitc.opaque<"int">
        %131 = call_opaque "__riscv_vand_vx_u8mf2"(%121, %130, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
        %132 = literal "2" : !emitc.opaque<"int">
        %133 = call_opaque "__riscv_vsll_vx_u8mf2"(%131, %132, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %134 = call_opaque "__riscv_vor_vv_u8mf2"(%129, %123, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %135 = call_opaque "__riscv_vor_vv_u8mf2"(%133, %125, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %136 = call_opaque "__riscv_vzext_vf2_u16m1"(%134, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %137 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%136) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %138 = call_opaque "__riscv_vzext_vf2_u16m1"(%135, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %139 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%138) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        %140 = literal "112" : !emitc.opaque<"size_t">
        %141 = add %32, %140 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %142 = cast %141 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %143 = call_opaque "__riscv_vle8_v_u8mf2"(%142, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        %144 = literal "240" : !emitc.opaque<"size_t">
        %145 = add %32, %144 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %146 = cast %145 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %147 = call_opaque "__riscv_vle8_v_u8mf2"(%146, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %148 = literal "0x0F" : !emitc.opaque<"int">
        %149 = call_opaque "__riscv_vand_vx_u8mf2"(%143, %148, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %150 = literal "4" : !emitc.opaque<"int">
        %151 = call_opaque "__riscv_vsrl_vx_u8mf2"(%143, %150, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %152 = literal "0x03" : !emitc.opaque<"int">
        %153 = call_opaque "__riscv_vand_vx_u8mf2"(%147, %152, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
        %154 = literal "4" : !emitc.opaque<"int">
        %155 = call_opaque "__riscv_vsll_vx_u8mf2"(%153, %154, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %156 = literal "0x0C" : !emitc.opaque<"int">
        %157 = call_opaque "__riscv_vand_vx_u8mf2"(%147, %156, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
        %158 = literal "2" : !emitc.opaque<"int">
        %159 = call_opaque "__riscv_vsll_vx_u8mf2"(%157, %158, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %160 = call_opaque "__riscv_vor_vv_u8mf2"(%155, %149, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %161 = call_opaque "__riscv_vor_vv_u8mf2"(%159, %151, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %162 = call_opaque "__riscv_vzext_vf2_u16m1"(%160, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %163 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%162) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %164 = call_opaque "__riscv_vzext_vf2_u16m1"(%161, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %165 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%164) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        %166 = literal "72" : !emitc.opaque<"size_t">
        %167 = add %32, %166 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %168 = cast %167 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %169 = call_opaque "__riscv_vle8_v_u8mf2"(%168, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        %170 = literal "200" : !emitc.opaque<"size_t">
        %171 = add %32, %170 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %172 = cast %171 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %173 = call_opaque "__riscv_vle8_v_u8mf2"(%172, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %174 = literal "0x0F" : !emitc.opaque<"int">
        %175 = call_opaque "__riscv_vand_vx_u8mf2"(%169, %174, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %176 = literal "4" : !emitc.opaque<"int">
        %177 = call_opaque "__riscv_vsrl_vx_u8mf2"(%169, %176, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %178 = literal "0x03" : !emitc.opaque<"int">
        %179 = call_opaque "__riscv_vand_vx_u8mf2"(%173, %178, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
        %180 = literal "4" : !emitc.opaque<"int">
        %181 = call_opaque "__riscv_vsll_vx_u8mf2"(%179, %180, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %182 = literal "0x0C" : !emitc.opaque<"int">
        %183 = call_opaque "__riscv_vand_vx_u8mf2"(%173, %182, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
        %184 = literal "2" : !emitc.opaque<"int">
        %185 = call_opaque "__riscv_vsll_vx_u8mf2"(%183, %184, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %186 = call_opaque "__riscv_vor_vv_u8mf2"(%181, %175, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %187 = call_opaque "__riscv_vor_vv_u8mf2"(%185, %177, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %188 = call_opaque "__riscv_vzext_vf2_u16m1"(%186, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %189 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%188) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %190 = call_opaque "__riscv_vzext_vf2_u16m1"(%187, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %191 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%190) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        %192 = literal "88" : !emitc.opaque<"size_t">
        %193 = add %32, %192 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %194 = cast %193 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %195 = call_opaque "__riscv_vle8_v_u8mf2"(%194, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        %196 = literal "216" : !emitc.opaque<"size_t">
        %197 = add %32, %196 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %198 = cast %197 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %199 = call_opaque "__riscv_vle8_v_u8mf2"(%198, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %200 = literal "0x0F" : !emitc.opaque<"int">
        %201 = call_opaque "__riscv_vand_vx_u8mf2"(%195, %200, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %202 = literal "4" : !emitc.opaque<"int">
        %203 = call_opaque "__riscv_vsrl_vx_u8mf2"(%195, %202, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %204 = literal "0x03" : !emitc.opaque<"int">
        %205 = call_opaque "__riscv_vand_vx_u8mf2"(%199, %204, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
        %206 = literal "4" : !emitc.opaque<"int">
        %207 = call_opaque "__riscv_vsll_vx_u8mf2"(%205, %206, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %208 = literal "0x0C" : !emitc.opaque<"int">
        %209 = call_opaque "__riscv_vand_vx_u8mf2"(%199, %208, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
        %210 = literal "2" : !emitc.opaque<"int">
        %211 = call_opaque "__riscv_vsll_vx_u8mf2"(%209, %210, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %212 = call_opaque "__riscv_vor_vv_u8mf2"(%207, %201, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %213 = call_opaque "__riscv_vor_vv_u8mf2"(%211, %203, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %214 = call_opaque "__riscv_vzext_vf2_u16m1"(%212, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %215 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%214) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %216 = call_opaque "__riscv_vzext_vf2_u16m1"(%213, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %217 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%216) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        %218 = literal "104" : !emitc.opaque<"size_t">
        %219 = add %32, %218 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %220 = cast %219 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %221 = call_opaque "__riscv_vle8_v_u8mf2"(%220, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        %222 = literal "232" : !emitc.opaque<"size_t">
        %223 = add %32, %222 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %224 = cast %223 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %225 = call_opaque "__riscv_vle8_v_u8mf2"(%224, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %226 = literal "0x0F" : !emitc.opaque<"int">
        %227 = call_opaque "__riscv_vand_vx_u8mf2"(%221, %226, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %228 = literal "4" : !emitc.opaque<"int">
        %229 = call_opaque "__riscv_vsrl_vx_u8mf2"(%221, %228, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %230 = literal "0x03" : !emitc.opaque<"int">
        %231 = call_opaque "__riscv_vand_vx_u8mf2"(%225, %230, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
        %232 = literal "4" : !emitc.opaque<"int">
        %233 = call_opaque "__riscv_vsll_vx_u8mf2"(%231, %232, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %234 = literal "0x0C" : !emitc.opaque<"int">
        %235 = call_opaque "__riscv_vand_vx_u8mf2"(%225, %234, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
        %236 = literal "2" : !emitc.opaque<"int">
        %237 = call_opaque "__riscv_vsll_vx_u8mf2"(%235, %236, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %238 = call_opaque "__riscv_vor_vv_u8mf2"(%233, %227, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %239 = call_opaque "__riscv_vor_vv_u8mf2"(%237, %229, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %240 = call_opaque "__riscv_vzext_vf2_u16m1"(%238, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %241 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%240) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %242 = call_opaque "__riscv_vzext_vf2_u16m1"(%239, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %243 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%242) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        %244 = literal "120" : !emitc.opaque<"size_t">
        %245 = add %32, %244 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %246 = cast %245 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %247 = call_opaque "__riscv_vle8_v_u8mf2"(%246, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        %248 = literal "248" : !emitc.opaque<"size_t">
        %249 = add %32, %248 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %250 = cast %249 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %251 = call_opaque "__riscv_vle8_v_u8mf2"(%250, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %252 = literal "0x0F" : !emitc.opaque<"int">
        %253 = call_opaque "__riscv_vand_vx_u8mf2"(%247, %252, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %254 = literal "4" : !emitc.opaque<"int">
        %255 = call_opaque "__riscv_vsrl_vx_u8mf2"(%247, %254, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %256 = literal "0x03" : !emitc.opaque<"int">
        %257 = call_opaque "__riscv_vand_vx_u8mf2"(%251, %256, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
        %258 = literal "4" : !emitc.opaque<"int">
        %259 = call_opaque "__riscv_vsll_vx_u8mf2"(%257, %258, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %260 = literal "0x0C" : !emitc.opaque<"int">
        %261 = call_opaque "__riscv_vand_vx_u8mf2"(%251, %260, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
        %262 = literal "2" : !emitc.opaque<"int">
        %263 = call_opaque "__riscv_vsll_vx_u8mf2"(%261, %262, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %264 = call_opaque "__riscv_vor_vv_u8mf2"(%259, %253, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %265 = call_opaque "__riscv_vor_vv_u8mf2"(%263, %255, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %266 = call_opaque "__riscv_vzext_vf2_u16m1"(%264, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %267 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%266) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %268 = call_opaque "__riscv_vzext_vf2_u16m1"(%265, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %269 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%268) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=min_bsums_fold"
        %270 = literal "260" : !emitc.opaque<"size_t">
        %271 = add %35, %270 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %272 = cast %271 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int16_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar"
        %273 = call_opaque "*(const int16_t *)"(%272) : (!emitc.ptr<!emitc.opaque<"const int16_t">>) -> !emitc.opaque<"int32_t">
        %274 = literal "262" : !emitc.opaque<"size_t">
        %275 = add %35, %274 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %276 = cast %275 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int16_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar"
        %277 = call_opaque "*(const int16_t *)"(%276) : (!emitc.ptr<!emitc.opaque<"const int16_t">>) -> !emitc.opaque<"int32_t">
        %278 = add %273, %277 : (!emitc.opaque<"int32_t">, !emitc.opaque<"int32_t">) -> !emitc.opaque<"int32_t">
        %279 = load %53 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2"
        %280 = call_opaque "__riscv_vwmacc_vx_i32m2"(%279, %278, %87, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %280 : !emitc.opaque<"vint32m2_t"> to %53 : <!emitc.opaque<"vint32m2_t">>
        %281 = load %59 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2"
        %282 = call_opaque "__riscv_vwmacc_vx_i32m2"(%281, %278, %191, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %282 : !emitc.opaque<"vint32m2_t"> to %59 : <!emitc.opaque<"vint32m2_t">>
        %283 = literal "264" : !emitc.opaque<"size_t">
        %284 = add %35, %283 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %285 = cast %284 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int16_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar"
        %286 = call_opaque "*(const int16_t *)"(%285) : (!emitc.ptr<!emitc.opaque<"const int16_t">>) -> !emitc.opaque<"int32_t">
        %287 = literal "266" : !emitc.opaque<"size_t">
        %288 = add %35, %287 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %289 = cast %288 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int16_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar"
        %290 = call_opaque "*(const int16_t *)"(%289) : (!emitc.ptr<!emitc.opaque<"const int16_t">>) -> !emitc.opaque<"int32_t">
        %291 = add %286, %290 : (!emitc.opaque<"int32_t">, !emitc.opaque<"int32_t">) -> !emitc.opaque<"int32_t">
        %292 = load %53 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2"
        %293 = call_opaque "__riscv_vwmacc_vx_i32m2"(%292, %291, %113, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %293 : !emitc.opaque<"vint32m2_t"> to %53 : <!emitc.opaque<"vint32m2_t">>
        %294 = load %59 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2"
        %295 = call_opaque "__riscv_vwmacc_vx_i32m2"(%294, %291, %217, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %295 : !emitc.opaque<"vint32m2_t"> to %59 : <!emitc.opaque<"vint32m2_t">>
        %296 = literal "268" : !emitc.opaque<"size_t">
        %297 = add %35, %296 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %298 = cast %297 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int16_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar"
        %299 = call_opaque "*(const int16_t *)"(%298) : (!emitc.ptr<!emitc.opaque<"const int16_t">>) -> !emitc.opaque<"int32_t">
        %300 = literal "270" : !emitc.opaque<"size_t">
        %301 = add %35, %300 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %302 = cast %301 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int16_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar"
        %303 = call_opaque "*(const int16_t *)"(%302) : (!emitc.ptr<!emitc.opaque<"const int16_t">>) -> !emitc.opaque<"int32_t">
        %304 = add %299, %303 : (!emitc.opaque<"int32_t">, !emitc.opaque<"int32_t">) -> !emitc.opaque<"int32_t">
        %305 = load %53 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2"
        %306 = call_opaque "__riscv_vwmacc_vx_i32m2"(%305, %304, %139, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %306 : !emitc.opaque<"vint32m2_t"> to %53 : <!emitc.opaque<"vint32m2_t">>
        %307 = load %59 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2"
        %308 = call_opaque "__riscv_vwmacc_vx_i32m2"(%307, %304, %243, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %308 : !emitc.opaque<"vint32m2_t"> to %59 : <!emitc.opaque<"vint32m2_t">>
        %309 = literal "272" : !emitc.opaque<"size_t">
        %310 = add %35, %309 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %311 = cast %310 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int16_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar"
        %312 = call_opaque "*(const int16_t *)"(%311) : (!emitc.ptr<!emitc.opaque<"const int16_t">>) -> !emitc.opaque<"int32_t">
        %313 = literal "274" : !emitc.opaque<"size_t">
        %314 = add %35, %313 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %315 = cast %314 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int16_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar"
        %316 = call_opaque "*(const int16_t *)"(%315) : (!emitc.ptr<!emitc.opaque<"const int16_t">>) -> !emitc.opaque<"int32_t">
        %317 = add %312, %316 : (!emitc.opaque<"int32_t">, !emitc.opaque<"int32_t">) -> !emitc.opaque<"int32_t">
        %318 = load %53 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2"
        %319 = call_opaque "__riscv_vwmacc_vx_i32m2"(%318, %317, %165, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %319 : !emitc.opaque<"vint32m2_t"> to %53 : <!emitc.opaque<"vint32m2_t">>
        %320 = load %59 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2"
        %321 = call_opaque "__riscv_vwmacc_vx_i32m2"(%320, %317, %269, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %321 : !emitc.opaque<"vint32m2_t"> to %59 : <!emitc.opaque<"vint32m2_t">>
        %322 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %323 = literal "0" : !emitc.opaque<"int32_t">
        %324 = call_opaque "__riscv_vmv_v_x_i16m1"(%323, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %324 : !emitc.opaque<"vint16m1_t"> to %322 : <!emitc.opaque<"vint16m1_t">>
        %325 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %326 = literal "0" : !emitc.opaque<"int32_t">
        %327 = call_opaque "__riscv_vmv_v_x_i16m1"(%326, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %327 : !emitc.opaque<"vint16m1_t"> to %325 : <!emitc.opaque<"vint16m1_t">>
        %328 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %329 = literal "0" : !emitc.opaque<"int32_t">
        %330 = call_opaque "__riscv_vmv_v_x_i16m1"(%329, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %330 : !emitc.opaque<"vint16m1_t"> to %328 : <!emitc.opaque<"vint16m1_t">>
        %331 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %332 = literal "0" : !emitc.opaque<"int32_t">
        %333 = call_opaque "__riscv_vmv_v_x_i16m1"(%332, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %333 : !emitc.opaque<"vint16m1_t"> to %331 : <!emitc.opaque<"vint16m1_t">>
        %334 = literal "1" : !emitc.opaque<"size_t">
        %335 = literal "16" : !emitc.opaque<"size_t">
        %336 = literal "0" : !emitc.opaque<"size_t">
        for %arg7 = %336 to %335 step %334  : !emitc.opaque<"size_t"> {
          %790 = literal "16" : !emitc.opaque<"size_t">
          %791 = mul %arg7, %790 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr"
          %792 = literal "768" : !emitc.opaque<"size_t">
          %793 = add %792, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %794 = add %32, %793 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %795 = cast %794 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %796 = call_opaque "__riscv_vle8_v_u8mf2"(%795, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %797 = literal "0x0F" : !emitc.opaque<"int">
          %798 = call_opaque "__riscv_vand_vx_u8mf2"(%796, %797, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %799 = literal "4" : !emitc.opaque<"int">
          %800 = call_opaque "__riscv_vsrl_vx_u8mf2"(%796, %799, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          %801 = literal "256" : !emitc.opaque<"size_t">
          %802 = add %801, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %803 = add %32, %802 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %804 = cast %803 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %805 = call_opaque "__riscv_vle8_v_u8mf2"(%804, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %806 = literal "0x01" : !emitc.opaque<"int">
          %807 = call_opaque "__riscv_vand_vx_u8mf2"(%805, %806, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %808 = literal "4" : !emitc.opaque<"int">
          %809 = call_opaque "__riscv_vsll_vx_u8mf2"(%807, %808, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %810 = literal "1" : !emitc.opaque<"int">
          %811 = call_opaque "__riscv_vsrl_vx_u8mf2"(%805, %810, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %812 = literal "0x01" : !emitc.opaque<"int">
          %813 = call_opaque "__riscv_vand_vx_u8mf2"(%811, %812, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %814 = literal "4" : !emitc.opaque<"int">
          %815 = call_opaque "__riscv_vsll_vx_u8mf2"(%813, %814, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %816 = call_opaque "__riscv_vor_vv_u8mf2"(%798, %809, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %817 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%816) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %818 = call_opaque "__riscv_vor_vv_u8mf2"(%800, %815, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %819 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%818) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr"
          %820 = literal "4" : !emitc.opaque<"size_t">
          %821 = add %820, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %822 = literal "36" : !emitc.opaque<"size_t">
          %823 = add %822, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %824 = add %35, %821 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %825 = cast %824 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %826 = call_opaque "*(const int8_t *)"(%825) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %827 = add %35, %823 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %828 = cast %827 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %829 = call_opaque "*(const int8_t *)"(%828) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %830 = load %322 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %831 = call_opaque "__riscv_vwmacc_vx_i16m1"(%830, %826, %817, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %831 : !emitc.opaque<"vint16m1_t"> to %322 : <!emitc.opaque<"vint16m1_t">>
          %832 = load %325 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %833 = call_opaque "__riscv_vwmacc_vx_i16m1"(%832, %829, %819, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %833 : !emitc.opaque<"vint16m1_t"> to %325 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr"
          %834 = literal "776" : !emitc.opaque<"size_t">
          %835 = add %834, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %836 = add %32, %835 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %837 = cast %836 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %838 = call_opaque "__riscv_vle8_v_u8mf2"(%837, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %839 = literal "0x0F" : !emitc.opaque<"int">
          %840 = call_opaque "__riscv_vand_vx_u8mf2"(%838, %839, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %841 = literal "4" : !emitc.opaque<"int">
          %842 = call_opaque "__riscv_vsrl_vx_u8mf2"(%838, %841, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          %843 = literal "264" : !emitc.opaque<"size_t">
          %844 = add %843, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %845 = add %32, %844 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %846 = cast %845 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %847 = call_opaque "__riscv_vle8_v_u8mf2"(%846, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %848 = literal "0x01" : !emitc.opaque<"int">
          %849 = call_opaque "__riscv_vand_vx_u8mf2"(%847, %848, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %850 = literal "4" : !emitc.opaque<"int">
          %851 = call_opaque "__riscv_vsll_vx_u8mf2"(%849, %850, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %852 = literal "1" : !emitc.opaque<"int">
          %853 = call_opaque "__riscv_vsrl_vx_u8mf2"(%847, %852, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %854 = literal "0x01" : !emitc.opaque<"int">
          %855 = call_opaque "__riscv_vand_vx_u8mf2"(%853, %854, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %856 = literal "4" : !emitc.opaque<"int">
          %857 = call_opaque "__riscv_vsll_vx_u8mf2"(%855, %856, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %858 = call_opaque "__riscv_vor_vv_u8mf2"(%840, %851, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %859 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%858) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %860 = call_opaque "__riscv_vor_vv_u8mf2"(%842, %857, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %861 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%860) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr"
          %862 = literal "4" : !emitc.opaque<"size_t">
          %863 = add %862, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %864 = literal "36" : !emitc.opaque<"size_t">
          %865 = add %864, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %866 = add %35, %863 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %867 = cast %866 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %868 = call_opaque "*(const int8_t *)"(%867) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %869 = add %35, %865 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %870 = cast %869 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %871 = call_opaque "*(const int8_t *)"(%870) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %872 = load %328 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %873 = call_opaque "__riscv_vwmacc_vx_i16m1"(%872, %868, %859, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %873 : !emitc.opaque<"vint16m1_t"> to %328 : <!emitc.opaque<"vint16m1_t">>
          %874 = load %331 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %875 = call_opaque "__riscv_vwmacc_vx_i16m1"(%874, %871, %861, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %875 : !emitc.opaque<"vint16m1_t"> to %331 : <!emitc.opaque<"vint16m1_t">>
        }
        %337 = load %322 : <!emitc.opaque<"vint16m1_t">>
        %338 = load %325 : <!emitc.opaque<"vint16m1_t">>
        %339 = load %328 : <!emitc.opaque<"vint16m1_t">>
        %340 = load %331 : <!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold"
        %341 = load %50 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %342 = call_opaque "__riscv_vwmacc_vv_i32m2"(%341, %85, %337, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %343 = call_opaque "__riscv_vwmacc_vv_i32m2"(%342, %111, %338, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %343 : !emitc.opaque<"vint32m2_t"> to %50 : <!emitc.opaque<"vint32m2_t">>
        %344 = load %56 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %345 = call_opaque "__riscv_vwmacc_vv_i32m2"(%344, %189, %339, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %346 = call_opaque "__riscv_vwmacc_vv_i32m2"(%345, %215, %340, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %346 : !emitc.opaque<"vint32m2_t"> to %56 : <!emitc.opaque<"vint32m2_t">>
        %347 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %348 = literal "0" : !emitc.opaque<"int32_t">
        %349 = call_opaque "__riscv_vmv_v_x_i16m1"(%348, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %349 : !emitc.opaque<"vint16m1_t"> to %347 : <!emitc.opaque<"vint16m1_t">>
        %350 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %351 = literal "0" : !emitc.opaque<"int32_t">
        %352 = call_opaque "__riscv_vmv_v_x_i16m1"(%351, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %352 : !emitc.opaque<"vint16m1_t"> to %350 : <!emitc.opaque<"vint16m1_t">>
        %353 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %354 = literal "0" : !emitc.opaque<"int32_t">
        %355 = call_opaque "__riscv_vmv_v_x_i16m1"(%354, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %355 : !emitc.opaque<"vint16m1_t"> to %353 : <!emitc.opaque<"vint16m1_t">>
        %356 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %357 = literal "0" : !emitc.opaque<"int32_t">
        %358 = call_opaque "__riscv_vmv_v_x_i16m1"(%357, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %358 : !emitc.opaque<"vint16m1_t"> to %356 : <!emitc.opaque<"vint16m1_t">>
        %359 = literal "1" : !emitc.opaque<"size_t">
        %360 = literal "16" : !emitc.opaque<"size_t">
        %361 = literal "0" : !emitc.opaque<"size_t">
        for %arg7 = %361 to %360 step %359  : !emitc.opaque<"size_t"> {
          %790 = literal "16" : !emitc.opaque<"size_t">
          %791 = mul %arg7, %790 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr"
          %792 = literal "1024" : !emitc.opaque<"size_t">
          %793 = add %792, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %794 = add %32, %793 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %795 = cast %794 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %796 = call_opaque "__riscv_vle8_v_u8mf2"(%795, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %797 = literal "0x0F" : !emitc.opaque<"int">
          %798 = call_opaque "__riscv_vand_vx_u8mf2"(%796, %797, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %799 = literal "4" : !emitc.opaque<"int">
          %800 = call_opaque "__riscv_vsrl_vx_u8mf2"(%796, %799, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          %801 = literal "512" : !emitc.opaque<"size_t">
          %802 = add %801, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %803 = add %32, %802 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %804 = cast %803 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %805 = call_opaque "__riscv_vle8_v_u8mf2"(%804, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %806 = literal "0x01" : !emitc.opaque<"int">
          %807 = call_opaque "__riscv_vand_vx_u8mf2"(%805, %806, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %808 = literal "4" : !emitc.opaque<"int">
          %809 = call_opaque "__riscv_vsll_vx_u8mf2"(%807, %808, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %810 = literal "1" : !emitc.opaque<"int">
          %811 = call_opaque "__riscv_vsrl_vx_u8mf2"(%805, %810, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %812 = literal "0x01" : !emitc.opaque<"int">
          %813 = call_opaque "__riscv_vand_vx_u8mf2"(%811, %812, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %814 = literal "4" : !emitc.opaque<"int">
          %815 = call_opaque "__riscv_vsll_vx_u8mf2"(%813, %814, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %816 = call_opaque "__riscv_vor_vv_u8mf2"(%798, %809, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %817 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%816) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %818 = call_opaque "__riscv_vor_vv_u8mf2"(%800, %815, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %819 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%818) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr"
          %820 = literal "20" : !emitc.opaque<"size_t">
          %821 = add %820, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %822 = literal "52" : !emitc.opaque<"size_t">
          %823 = add %822, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %824 = add %35, %821 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %825 = cast %824 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %826 = call_opaque "*(const int8_t *)"(%825) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %827 = add %35, %823 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %828 = cast %827 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %829 = call_opaque "*(const int8_t *)"(%828) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %830 = load %347 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %831 = call_opaque "__riscv_vwmacc_vx_i16m1"(%830, %826, %817, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %831 : !emitc.opaque<"vint16m1_t"> to %347 : <!emitc.opaque<"vint16m1_t">>
          %832 = load %350 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %833 = call_opaque "__riscv_vwmacc_vx_i16m1"(%832, %829, %819, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %833 : !emitc.opaque<"vint16m1_t"> to %350 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr"
          %834 = literal "1032" : !emitc.opaque<"size_t">
          %835 = add %834, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %836 = add %32, %835 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %837 = cast %836 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %838 = call_opaque "__riscv_vle8_v_u8mf2"(%837, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %839 = literal "0x0F" : !emitc.opaque<"int">
          %840 = call_opaque "__riscv_vand_vx_u8mf2"(%838, %839, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %841 = literal "4" : !emitc.opaque<"int">
          %842 = call_opaque "__riscv_vsrl_vx_u8mf2"(%838, %841, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          %843 = literal "520" : !emitc.opaque<"size_t">
          %844 = add %843, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %845 = add %32, %844 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %846 = cast %845 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %847 = call_opaque "__riscv_vle8_v_u8mf2"(%846, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %848 = literal "0x01" : !emitc.opaque<"int">
          %849 = call_opaque "__riscv_vand_vx_u8mf2"(%847, %848, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %850 = literal "4" : !emitc.opaque<"int">
          %851 = call_opaque "__riscv_vsll_vx_u8mf2"(%849, %850, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %852 = literal "1" : !emitc.opaque<"int">
          %853 = call_opaque "__riscv_vsrl_vx_u8mf2"(%847, %852, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %854 = literal "0x01" : !emitc.opaque<"int">
          %855 = call_opaque "__riscv_vand_vx_u8mf2"(%853, %854, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %856 = literal "4" : !emitc.opaque<"int">
          %857 = call_opaque "__riscv_vsll_vx_u8mf2"(%855, %856, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %858 = call_opaque "__riscv_vor_vv_u8mf2"(%840, %851, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %859 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%858) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %860 = call_opaque "__riscv_vor_vv_u8mf2"(%842, %857, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %861 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%860) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr"
          %862 = literal "20" : !emitc.opaque<"size_t">
          %863 = add %862, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %864 = literal "52" : !emitc.opaque<"size_t">
          %865 = add %864, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %866 = add %35, %863 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %867 = cast %866 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %868 = call_opaque "*(const int8_t *)"(%867) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %869 = add %35, %865 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %870 = cast %869 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %871 = call_opaque "*(const int8_t *)"(%870) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %872 = load %353 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %873 = call_opaque "__riscv_vwmacc_vx_i16m1"(%872, %868, %859, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %873 : !emitc.opaque<"vint16m1_t"> to %353 : <!emitc.opaque<"vint16m1_t">>
          %874 = load %356 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %875 = call_opaque "__riscv_vwmacc_vx_i16m1"(%874, %871, %861, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %875 : !emitc.opaque<"vint16m1_t"> to %356 : <!emitc.opaque<"vint16m1_t">>
        }
        %362 = load %347 : <!emitc.opaque<"vint16m1_t">>
        %363 = load %350 : <!emitc.opaque<"vint16m1_t">>
        %364 = load %353 : <!emitc.opaque<"vint16m1_t">>
        %365 = load %356 : <!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold"
        %366 = load %50 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %367 = call_opaque "__riscv_vwmacc_vv_i32m2"(%366, %85, %362, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %368 = call_opaque "__riscv_vwmacc_vv_i32m2"(%367, %111, %363, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %368 : !emitc.opaque<"vint32m2_t"> to %50 : <!emitc.opaque<"vint32m2_t">>
        %369 = load %56 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %370 = call_opaque "__riscv_vwmacc_vv_i32m2"(%369, %189, %364, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %371 = call_opaque "__riscv_vwmacc_vv_i32m2"(%370, %215, %365, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %371 : !emitc.opaque<"vint32m2_t"> to %56 : <!emitc.opaque<"vint32m2_t">>
        %372 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %373 = literal "0" : !emitc.opaque<"int32_t">
        %374 = call_opaque "__riscv_vmv_v_x_i16m1"(%373, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %374 : !emitc.opaque<"vint16m1_t"> to %372 : <!emitc.opaque<"vint16m1_t">>
        %375 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %376 = literal "0" : !emitc.opaque<"int32_t">
        %377 = call_opaque "__riscv_vmv_v_x_i16m1"(%376, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %377 : !emitc.opaque<"vint16m1_t"> to %375 : <!emitc.opaque<"vint16m1_t">>
        %378 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %379 = literal "0" : !emitc.opaque<"int32_t">
        %380 = call_opaque "__riscv_vmv_v_x_i16m1"(%379, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %380 : !emitc.opaque<"vint16m1_t"> to %378 : <!emitc.opaque<"vint16m1_t">>
        %381 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %382 = literal "0" : !emitc.opaque<"int32_t">
        %383 = call_opaque "__riscv_vmv_v_x_i16m1"(%382, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %383 : !emitc.opaque<"vint16m1_t"> to %381 : <!emitc.opaque<"vint16m1_t">>
        %384 = literal "1" : !emitc.opaque<"size_t">
        %385 = literal "16" : !emitc.opaque<"size_t">
        %386 = literal "0" : !emitc.opaque<"size_t">
        for %arg7 = %386 to %385 step %384  : !emitc.opaque<"size_t"> {
          %790 = literal "16" : !emitc.opaque<"size_t">
          %791 = mul %arg7, %790 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr"
          %792 = literal "1280" : !emitc.opaque<"size_t">
          %793 = add %792, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %794 = add %32, %793 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %795 = cast %794 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %796 = call_opaque "__riscv_vle8_v_u8mf2"(%795, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %797 = literal "0x0F" : !emitc.opaque<"int">
          %798 = call_opaque "__riscv_vand_vx_u8mf2"(%796, %797, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %799 = literal "4" : !emitc.opaque<"int">
          %800 = call_opaque "__riscv_vsrl_vx_u8mf2"(%796, %799, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          %801 = literal "256" : !emitc.opaque<"size_t">
          %802 = add %801, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %803 = add %32, %802 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %804 = cast %803 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %805 = call_opaque "__riscv_vle8_v_u8mf2"(%804, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %806 = literal "2" : !emitc.opaque<"int">
          %807 = call_opaque "__riscv_vsrl_vx_u8mf2"(%805, %806, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %808 = literal "0x01" : !emitc.opaque<"int">
          %809 = call_opaque "__riscv_vand_vx_u8mf2"(%807, %808, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %810 = literal "4" : !emitc.opaque<"int">
          %811 = call_opaque "__riscv_vsll_vx_u8mf2"(%809, %810, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %812 = literal "3" : !emitc.opaque<"int">
          %813 = call_opaque "__riscv_vsrl_vx_u8mf2"(%805, %812, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %814 = literal "0x01" : !emitc.opaque<"int">
          %815 = call_opaque "__riscv_vand_vx_u8mf2"(%813, %814, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %816 = literal "4" : !emitc.opaque<"int">
          %817 = call_opaque "__riscv_vsll_vx_u8mf2"(%815, %816, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %818 = call_opaque "__riscv_vor_vv_u8mf2"(%798, %811, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %819 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%818) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %820 = call_opaque "__riscv_vor_vv_u8mf2"(%800, %817, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %821 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%820) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr"
          %822 = literal "68" : !emitc.opaque<"size_t">
          %823 = add %822, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %824 = literal "100" : !emitc.opaque<"size_t">
          %825 = add %824, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %826 = add %35, %823 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %827 = cast %826 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %828 = call_opaque "*(const int8_t *)"(%827) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %829 = add %35, %825 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %830 = cast %829 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %831 = call_opaque "*(const int8_t *)"(%830) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %832 = load %372 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %833 = call_opaque "__riscv_vwmacc_vx_i16m1"(%832, %828, %819, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %833 : !emitc.opaque<"vint16m1_t"> to %372 : <!emitc.opaque<"vint16m1_t">>
          %834 = load %375 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %835 = call_opaque "__riscv_vwmacc_vx_i16m1"(%834, %831, %821, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %835 : !emitc.opaque<"vint16m1_t"> to %375 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr"
          %836 = literal "1288" : !emitc.opaque<"size_t">
          %837 = add %836, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %838 = add %32, %837 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %839 = cast %838 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %840 = call_opaque "__riscv_vle8_v_u8mf2"(%839, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %841 = literal "0x0F" : !emitc.opaque<"int">
          %842 = call_opaque "__riscv_vand_vx_u8mf2"(%840, %841, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %843 = literal "4" : !emitc.opaque<"int">
          %844 = call_opaque "__riscv_vsrl_vx_u8mf2"(%840, %843, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          %845 = literal "264" : !emitc.opaque<"size_t">
          %846 = add %845, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %847 = add %32, %846 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %848 = cast %847 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %849 = call_opaque "__riscv_vle8_v_u8mf2"(%848, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %850 = literal "2" : !emitc.opaque<"int">
          %851 = call_opaque "__riscv_vsrl_vx_u8mf2"(%849, %850, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %852 = literal "0x01" : !emitc.opaque<"int">
          %853 = call_opaque "__riscv_vand_vx_u8mf2"(%851, %852, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %854 = literal "4" : !emitc.opaque<"int">
          %855 = call_opaque "__riscv_vsll_vx_u8mf2"(%853, %854, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %856 = literal "3" : !emitc.opaque<"int">
          %857 = call_opaque "__riscv_vsrl_vx_u8mf2"(%849, %856, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %858 = literal "0x01" : !emitc.opaque<"int">
          %859 = call_opaque "__riscv_vand_vx_u8mf2"(%857, %858, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %860 = literal "4" : !emitc.opaque<"int">
          %861 = call_opaque "__riscv_vsll_vx_u8mf2"(%859, %860, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %862 = call_opaque "__riscv_vor_vv_u8mf2"(%842, %855, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %863 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%862) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %864 = call_opaque "__riscv_vor_vv_u8mf2"(%844, %861, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %865 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%864) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr"
          %866 = literal "68" : !emitc.opaque<"size_t">
          %867 = add %866, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %868 = literal "100" : !emitc.opaque<"size_t">
          %869 = add %868, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %870 = add %35, %867 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %871 = cast %870 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %872 = call_opaque "*(const int8_t *)"(%871) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %873 = add %35, %869 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %874 = cast %873 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %875 = call_opaque "*(const int8_t *)"(%874) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %876 = load %378 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %877 = call_opaque "__riscv_vwmacc_vx_i16m1"(%876, %872, %863, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %877 : !emitc.opaque<"vint16m1_t"> to %378 : <!emitc.opaque<"vint16m1_t">>
          %878 = load %381 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %879 = call_opaque "__riscv_vwmacc_vx_i16m1"(%878, %875, %865, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %879 : !emitc.opaque<"vint16m1_t"> to %381 : <!emitc.opaque<"vint16m1_t">>
        }
        %387 = load %372 : <!emitc.opaque<"vint16m1_t">>
        %388 = load %375 : <!emitc.opaque<"vint16m1_t">>
        %389 = load %378 : <!emitc.opaque<"vint16m1_t">>
        %390 = load %381 : <!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold"
        %391 = load %50 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %392 = call_opaque "__riscv_vwmacc_vv_i32m2"(%391, %137, %387, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %393 = call_opaque "__riscv_vwmacc_vv_i32m2"(%392, %163, %388, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %393 : !emitc.opaque<"vint32m2_t"> to %50 : <!emitc.opaque<"vint32m2_t">>
        %394 = load %56 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %395 = call_opaque "__riscv_vwmacc_vv_i32m2"(%394, %241, %389, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %396 = call_opaque "__riscv_vwmacc_vv_i32m2"(%395, %267, %390, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %396 : !emitc.opaque<"vint32m2_t"> to %56 : <!emitc.opaque<"vint32m2_t">>
        %397 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %398 = literal "0" : !emitc.opaque<"int32_t">
        %399 = call_opaque "__riscv_vmv_v_x_i16m1"(%398, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %399 : !emitc.opaque<"vint16m1_t"> to %397 : <!emitc.opaque<"vint16m1_t">>
        %400 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %401 = literal "0" : !emitc.opaque<"int32_t">
        %402 = call_opaque "__riscv_vmv_v_x_i16m1"(%401, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %402 : !emitc.opaque<"vint16m1_t"> to %400 : <!emitc.opaque<"vint16m1_t">>
        %403 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %404 = literal "0" : !emitc.opaque<"int32_t">
        %405 = call_opaque "__riscv_vmv_v_x_i16m1"(%404, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %405 : !emitc.opaque<"vint16m1_t"> to %403 : <!emitc.opaque<"vint16m1_t">>
        %406 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %407 = literal "0" : !emitc.opaque<"int32_t">
        %408 = call_opaque "__riscv_vmv_v_x_i16m1"(%407, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %408 : !emitc.opaque<"vint16m1_t"> to %406 : <!emitc.opaque<"vint16m1_t">>
        %409 = literal "1" : !emitc.opaque<"size_t">
        %410 = literal "16" : !emitc.opaque<"size_t">
        %411 = literal "0" : !emitc.opaque<"size_t">
        for %arg7 = %411 to %410 step %409  : !emitc.opaque<"size_t"> {
          %790 = literal "16" : !emitc.opaque<"size_t">
          %791 = mul %arg7, %790 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr"
          %792 = literal "1536" : !emitc.opaque<"size_t">
          %793 = add %792, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %794 = add %32, %793 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %795 = cast %794 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %796 = call_opaque "__riscv_vle8_v_u8mf2"(%795, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %797 = literal "0x0F" : !emitc.opaque<"int">
          %798 = call_opaque "__riscv_vand_vx_u8mf2"(%796, %797, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %799 = literal "4" : !emitc.opaque<"int">
          %800 = call_opaque "__riscv_vsrl_vx_u8mf2"(%796, %799, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          %801 = literal "512" : !emitc.opaque<"size_t">
          %802 = add %801, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %803 = add %32, %802 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %804 = cast %803 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %805 = call_opaque "__riscv_vle8_v_u8mf2"(%804, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %806 = literal "2" : !emitc.opaque<"int">
          %807 = call_opaque "__riscv_vsrl_vx_u8mf2"(%805, %806, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %808 = literal "0x01" : !emitc.opaque<"int">
          %809 = call_opaque "__riscv_vand_vx_u8mf2"(%807, %808, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %810 = literal "4" : !emitc.opaque<"int">
          %811 = call_opaque "__riscv_vsll_vx_u8mf2"(%809, %810, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %812 = literal "3" : !emitc.opaque<"int">
          %813 = call_opaque "__riscv_vsrl_vx_u8mf2"(%805, %812, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %814 = literal "0x01" : !emitc.opaque<"int">
          %815 = call_opaque "__riscv_vand_vx_u8mf2"(%813, %814, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %816 = literal "4" : !emitc.opaque<"int">
          %817 = call_opaque "__riscv_vsll_vx_u8mf2"(%815, %816, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %818 = call_opaque "__riscv_vor_vv_u8mf2"(%798, %811, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %819 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%818) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %820 = call_opaque "__riscv_vor_vv_u8mf2"(%800, %817, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %821 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%820) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr"
          %822 = literal "84" : !emitc.opaque<"size_t">
          %823 = add %822, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %824 = literal "116" : !emitc.opaque<"size_t">
          %825 = add %824, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %826 = add %35, %823 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %827 = cast %826 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %828 = call_opaque "*(const int8_t *)"(%827) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %829 = add %35, %825 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %830 = cast %829 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %831 = call_opaque "*(const int8_t *)"(%830) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %832 = load %397 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %833 = call_opaque "__riscv_vwmacc_vx_i16m1"(%832, %828, %819, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %833 : !emitc.opaque<"vint16m1_t"> to %397 : <!emitc.opaque<"vint16m1_t">>
          %834 = load %400 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %835 = call_opaque "__riscv_vwmacc_vx_i16m1"(%834, %831, %821, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %835 : !emitc.opaque<"vint16m1_t"> to %400 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr"
          %836 = literal "1544" : !emitc.opaque<"size_t">
          %837 = add %836, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %838 = add %32, %837 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %839 = cast %838 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %840 = call_opaque "__riscv_vle8_v_u8mf2"(%839, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %841 = literal "0x0F" : !emitc.opaque<"int">
          %842 = call_opaque "__riscv_vand_vx_u8mf2"(%840, %841, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %843 = literal "4" : !emitc.opaque<"int">
          %844 = call_opaque "__riscv_vsrl_vx_u8mf2"(%840, %843, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          %845 = literal "520" : !emitc.opaque<"size_t">
          %846 = add %845, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %847 = add %32, %846 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %848 = cast %847 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %849 = call_opaque "__riscv_vle8_v_u8mf2"(%848, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %850 = literal "2" : !emitc.opaque<"int">
          %851 = call_opaque "__riscv_vsrl_vx_u8mf2"(%849, %850, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %852 = literal "0x01" : !emitc.opaque<"int">
          %853 = call_opaque "__riscv_vand_vx_u8mf2"(%851, %852, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %854 = literal "4" : !emitc.opaque<"int">
          %855 = call_opaque "__riscv_vsll_vx_u8mf2"(%853, %854, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %856 = literal "3" : !emitc.opaque<"int">
          %857 = call_opaque "__riscv_vsrl_vx_u8mf2"(%849, %856, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %858 = literal "0x01" : !emitc.opaque<"int">
          %859 = call_opaque "__riscv_vand_vx_u8mf2"(%857, %858, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %860 = literal "4" : !emitc.opaque<"int">
          %861 = call_opaque "__riscv_vsll_vx_u8mf2"(%859, %860, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %862 = call_opaque "__riscv_vor_vv_u8mf2"(%842, %855, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %863 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%862) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %864 = call_opaque "__riscv_vor_vv_u8mf2"(%844, %861, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %865 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%864) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr"
          %866 = literal "84" : !emitc.opaque<"size_t">
          %867 = add %866, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %868 = literal "116" : !emitc.opaque<"size_t">
          %869 = add %868, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %870 = add %35, %867 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %871 = cast %870 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %872 = call_opaque "*(const int8_t *)"(%871) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %873 = add %35, %869 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %874 = cast %873 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %875 = call_opaque "*(const int8_t *)"(%874) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %876 = load %403 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %877 = call_opaque "__riscv_vwmacc_vx_i16m1"(%876, %872, %863, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %877 : !emitc.opaque<"vint16m1_t"> to %403 : <!emitc.opaque<"vint16m1_t">>
          %878 = load %406 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %879 = call_opaque "__riscv_vwmacc_vx_i16m1"(%878, %875, %865, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %879 : !emitc.opaque<"vint16m1_t"> to %406 : <!emitc.opaque<"vint16m1_t">>
        }
        %412 = load %397 : <!emitc.opaque<"vint16m1_t">>
        %413 = load %400 : <!emitc.opaque<"vint16m1_t">>
        %414 = load %403 : <!emitc.opaque<"vint16m1_t">>
        %415 = load %406 : <!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold"
        %416 = load %50 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %417 = call_opaque "__riscv_vwmacc_vv_i32m2"(%416, %137, %412, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %418 = call_opaque "__riscv_vwmacc_vv_i32m2"(%417, %163, %413, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %418 : !emitc.opaque<"vint32m2_t"> to %50 : <!emitc.opaque<"vint32m2_t">>
        %419 = load %56 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %420 = call_opaque "__riscv_vwmacc_vv_i32m2"(%419, %241, %414, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %421 = call_opaque "__riscv_vwmacc_vv_i32m2"(%420, %267, %415, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %421 : !emitc.opaque<"vint32m2_t"> to %56 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_min_unpack_superhalf"
        %422 = literal "128" : !emitc.opaque<"size_t">
        %423 = add %32, %422 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %424 = cast %423 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %425 = call_opaque "__riscv_vle8_v_u8mf2"(%424, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        %426 = literal "192" : !emitc.opaque<"size_t">
        %427 = add %32, %426 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %428 = cast %427 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %429 = call_opaque "__riscv_vle8_v_u8mf2"(%428, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %430 = literal "0x0F" : !emitc.opaque<"int">
        %431 = call_opaque "__riscv_vand_vx_u8mf2"(%425, %430, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %432 = literal "4" : !emitc.opaque<"int">
        %433 = call_opaque "__riscv_vsrl_vx_u8mf2"(%425, %432, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %434 = literal "0x30" : !emitc.opaque<"int">
        %435 = call_opaque "__riscv_vand_vx_u8mf2"(%429, %434, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %436 = literal "0xC0" : !emitc.opaque<"int">
        %437 = call_opaque "__riscv_vand_vx_u8mf2"(%429, %436, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %438 = literal "2" : !emitc.opaque<"int">
        %439 = call_opaque "__riscv_vsrl_vx_u8mf2"(%437, %438, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %440 = call_opaque "__riscv_vor_vv_u8mf2"(%435, %431, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %441 = call_opaque "__riscv_vor_vv_u8mf2"(%439, %433, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %442 = call_opaque "__riscv_vzext_vf2_u16m1"(%440, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %443 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%442) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %444 = call_opaque "__riscv_vzext_vf2_u16m1"(%441, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %445 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%444) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        %446 = literal "144" : !emitc.opaque<"size_t">
        %447 = add %32, %446 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %448 = cast %447 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %449 = call_opaque "__riscv_vle8_v_u8mf2"(%448, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        %450 = literal "208" : !emitc.opaque<"size_t">
        %451 = add %32, %450 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %452 = cast %451 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %453 = call_opaque "__riscv_vle8_v_u8mf2"(%452, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %454 = literal "0x0F" : !emitc.opaque<"int">
        %455 = call_opaque "__riscv_vand_vx_u8mf2"(%449, %454, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %456 = literal "4" : !emitc.opaque<"int">
        %457 = call_opaque "__riscv_vsrl_vx_u8mf2"(%449, %456, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %458 = literal "0x30" : !emitc.opaque<"int">
        %459 = call_opaque "__riscv_vand_vx_u8mf2"(%453, %458, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %460 = literal "0xC0" : !emitc.opaque<"int">
        %461 = call_opaque "__riscv_vand_vx_u8mf2"(%453, %460, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %462 = literal "2" : !emitc.opaque<"int">
        %463 = call_opaque "__riscv_vsrl_vx_u8mf2"(%461, %462, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %464 = call_opaque "__riscv_vor_vv_u8mf2"(%459, %455, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %465 = call_opaque "__riscv_vor_vv_u8mf2"(%463, %457, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %466 = call_opaque "__riscv_vzext_vf2_u16m1"(%464, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %467 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%466) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %468 = call_opaque "__riscv_vzext_vf2_u16m1"(%465, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %469 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%468) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        %470 = literal "160" : !emitc.opaque<"size_t">
        %471 = add %32, %470 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %472 = cast %471 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %473 = call_opaque "__riscv_vle8_v_u8mf2"(%472, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        %474 = literal "224" : !emitc.opaque<"size_t">
        %475 = add %32, %474 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %476 = cast %475 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %477 = call_opaque "__riscv_vle8_v_u8mf2"(%476, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %478 = literal "0x0F" : !emitc.opaque<"int">
        %479 = call_opaque "__riscv_vand_vx_u8mf2"(%473, %478, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %480 = literal "4" : !emitc.opaque<"int">
        %481 = call_opaque "__riscv_vsrl_vx_u8mf2"(%473, %480, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %482 = literal "0x30" : !emitc.opaque<"int">
        %483 = call_opaque "__riscv_vand_vx_u8mf2"(%477, %482, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %484 = literal "0xC0" : !emitc.opaque<"int">
        %485 = call_opaque "__riscv_vand_vx_u8mf2"(%477, %484, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %486 = literal "2" : !emitc.opaque<"int">
        %487 = call_opaque "__riscv_vsrl_vx_u8mf2"(%485, %486, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %488 = call_opaque "__riscv_vor_vv_u8mf2"(%483, %479, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %489 = call_opaque "__riscv_vor_vv_u8mf2"(%487, %481, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %490 = call_opaque "__riscv_vzext_vf2_u16m1"(%488, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %491 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%490) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %492 = call_opaque "__riscv_vzext_vf2_u16m1"(%489, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %493 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%492) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        %494 = literal "176" : !emitc.opaque<"size_t">
        %495 = add %32, %494 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %496 = cast %495 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %497 = call_opaque "__riscv_vle8_v_u8mf2"(%496, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        %498 = literal "240" : !emitc.opaque<"size_t">
        %499 = add %32, %498 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %500 = cast %499 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %501 = call_opaque "__riscv_vle8_v_u8mf2"(%500, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %502 = literal "0x0F" : !emitc.opaque<"int">
        %503 = call_opaque "__riscv_vand_vx_u8mf2"(%497, %502, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %504 = literal "4" : !emitc.opaque<"int">
        %505 = call_opaque "__riscv_vsrl_vx_u8mf2"(%497, %504, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %506 = literal "0x30" : !emitc.opaque<"int">
        %507 = call_opaque "__riscv_vand_vx_u8mf2"(%501, %506, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %508 = literal "0xC0" : !emitc.opaque<"int">
        %509 = call_opaque "__riscv_vand_vx_u8mf2"(%501, %508, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %510 = literal "2" : !emitc.opaque<"int">
        %511 = call_opaque "__riscv_vsrl_vx_u8mf2"(%509, %510, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %512 = call_opaque "__riscv_vor_vv_u8mf2"(%507, %503, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %513 = call_opaque "__riscv_vor_vv_u8mf2"(%511, %505, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %514 = call_opaque "__riscv_vzext_vf2_u16m1"(%512, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %515 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%514) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %516 = call_opaque "__riscv_vzext_vf2_u16m1"(%513, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %517 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%516) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        %518 = literal "136" : !emitc.opaque<"size_t">
        %519 = add %32, %518 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %520 = cast %519 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %521 = call_opaque "__riscv_vle8_v_u8mf2"(%520, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        %522 = literal "200" : !emitc.opaque<"size_t">
        %523 = add %32, %522 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %524 = cast %523 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %525 = call_opaque "__riscv_vle8_v_u8mf2"(%524, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %526 = literal "0x0F" : !emitc.opaque<"int">
        %527 = call_opaque "__riscv_vand_vx_u8mf2"(%521, %526, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %528 = literal "4" : !emitc.opaque<"int">
        %529 = call_opaque "__riscv_vsrl_vx_u8mf2"(%521, %528, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %530 = literal "0x30" : !emitc.opaque<"int">
        %531 = call_opaque "__riscv_vand_vx_u8mf2"(%525, %530, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %532 = literal "0xC0" : !emitc.opaque<"int">
        %533 = call_opaque "__riscv_vand_vx_u8mf2"(%525, %532, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %534 = literal "2" : !emitc.opaque<"int">
        %535 = call_opaque "__riscv_vsrl_vx_u8mf2"(%533, %534, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %536 = call_opaque "__riscv_vor_vv_u8mf2"(%531, %527, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %537 = call_opaque "__riscv_vor_vv_u8mf2"(%535, %529, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %538 = call_opaque "__riscv_vzext_vf2_u16m1"(%536, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %539 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%538) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %540 = call_opaque "__riscv_vzext_vf2_u16m1"(%537, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %541 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%540) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        %542 = literal "152" : !emitc.opaque<"size_t">
        %543 = add %32, %542 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %544 = cast %543 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %545 = call_opaque "__riscv_vle8_v_u8mf2"(%544, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        %546 = literal "216" : !emitc.opaque<"size_t">
        %547 = add %32, %546 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %548 = cast %547 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %549 = call_opaque "__riscv_vle8_v_u8mf2"(%548, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %550 = literal "0x0F" : !emitc.opaque<"int">
        %551 = call_opaque "__riscv_vand_vx_u8mf2"(%545, %550, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %552 = literal "4" : !emitc.opaque<"int">
        %553 = call_opaque "__riscv_vsrl_vx_u8mf2"(%545, %552, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %554 = literal "0x30" : !emitc.opaque<"int">
        %555 = call_opaque "__riscv_vand_vx_u8mf2"(%549, %554, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %556 = literal "0xC0" : !emitc.opaque<"int">
        %557 = call_opaque "__riscv_vand_vx_u8mf2"(%549, %556, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %558 = literal "2" : !emitc.opaque<"int">
        %559 = call_opaque "__riscv_vsrl_vx_u8mf2"(%557, %558, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %560 = call_opaque "__riscv_vor_vv_u8mf2"(%555, %551, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %561 = call_opaque "__riscv_vor_vv_u8mf2"(%559, %553, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %562 = call_opaque "__riscv_vzext_vf2_u16m1"(%560, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %563 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%562) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %564 = call_opaque "__riscv_vzext_vf2_u16m1"(%561, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %565 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%564) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        %566 = literal "168" : !emitc.opaque<"size_t">
        %567 = add %32, %566 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %568 = cast %567 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %569 = call_opaque "__riscv_vle8_v_u8mf2"(%568, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        %570 = literal "232" : !emitc.opaque<"size_t">
        %571 = add %32, %570 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %572 = cast %571 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %573 = call_opaque "__riscv_vle8_v_u8mf2"(%572, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %574 = literal "0x0F" : !emitc.opaque<"int">
        %575 = call_opaque "__riscv_vand_vx_u8mf2"(%569, %574, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %576 = literal "4" : !emitc.opaque<"int">
        %577 = call_opaque "__riscv_vsrl_vx_u8mf2"(%569, %576, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %578 = literal "0x30" : !emitc.opaque<"int">
        %579 = call_opaque "__riscv_vand_vx_u8mf2"(%573, %578, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %580 = literal "0xC0" : !emitc.opaque<"int">
        %581 = call_opaque "__riscv_vand_vx_u8mf2"(%573, %580, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %582 = literal "2" : !emitc.opaque<"int">
        %583 = call_opaque "__riscv_vsrl_vx_u8mf2"(%581, %582, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %584 = call_opaque "__riscv_vor_vv_u8mf2"(%579, %575, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %585 = call_opaque "__riscv_vor_vv_u8mf2"(%583, %577, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %586 = call_opaque "__riscv_vzext_vf2_u16m1"(%584, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %587 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%586) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %588 = call_opaque "__riscv_vzext_vf2_u16m1"(%585, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %589 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%588) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        %590 = literal "184" : !emitc.opaque<"size_t">
        %591 = add %32, %590 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %592 = cast %591 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %593 = call_opaque "__riscv_vle8_v_u8mf2"(%592, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        %594 = literal "248" : !emitc.opaque<"size_t">
        %595 = add %32, %594 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %596 = cast %595 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
        %597 = call_opaque "__riscv_vle8_v_u8mf2"(%596, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %598 = literal "0x0F" : !emitc.opaque<"int">
        %599 = call_opaque "__riscv_vand_vx_u8mf2"(%593, %598, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %600 = literal "4" : !emitc.opaque<"int">
        %601 = call_opaque "__riscv_vsrl_vx_u8mf2"(%593, %600, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %602 = literal "0x30" : !emitc.opaque<"int">
        %603 = call_opaque "__riscv_vand_vx_u8mf2"(%597, %602, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
        %604 = literal "0xC0" : !emitc.opaque<"int">
        %605 = call_opaque "__riscv_vand_vx_u8mf2"(%597, %604, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
        %606 = literal "2" : !emitc.opaque<"int">
        %607 = call_opaque "__riscv_vsrl_vx_u8mf2"(%605, %606, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %608 = call_opaque "__riscv_vor_vv_u8mf2"(%603, %599, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
        %609 = call_opaque "__riscv_vor_vv_u8mf2"(%607, %601, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %610 = call_opaque "__riscv_vzext_vf2_u16m1"(%608, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %611 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%610) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1"
        %612 = call_opaque "__riscv_vzext_vf2_u16m1"(%609, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1"
        %613 = call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"(%612) : (!emitc.opaque<"vuint16m1_t">) -> !emitc.opaque<"vint16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=min_bsums_fold"
        %614 = literal "276" : !emitc.opaque<"size_t">
        %615 = add %35, %614 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %616 = cast %615 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int16_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar"
        %617 = call_opaque "*(const int16_t *)"(%616) : (!emitc.ptr<!emitc.opaque<"const int16_t">>) -> !emitc.opaque<"int32_t">
        %618 = literal "278" : !emitc.opaque<"size_t">
        %619 = add %35, %618 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %620 = cast %619 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int16_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar"
        %621 = call_opaque "*(const int16_t *)"(%620) : (!emitc.ptr<!emitc.opaque<"const int16_t">>) -> !emitc.opaque<"int32_t">
        %622 = add %617, %621 : (!emitc.opaque<"int32_t">, !emitc.opaque<"int32_t">) -> !emitc.opaque<"int32_t">
        %623 = load %53 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2"
        %624 = call_opaque "__riscv_vwmacc_vx_i32m2"(%623, %622, %445, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %624 : !emitc.opaque<"vint32m2_t"> to %53 : <!emitc.opaque<"vint32m2_t">>
        %625 = load %59 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2"
        %626 = call_opaque "__riscv_vwmacc_vx_i32m2"(%625, %622, %541, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %626 : !emitc.opaque<"vint32m2_t"> to %59 : <!emitc.opaque<"vint32m2_t">>
        %627 = literal "280" : !emitc.opaque<"size_t">
        %628 = add %35, %627 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %629 = cast %628 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int16_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar"
        %630 = call_opaque "*(const int16_t *)"(%629) : (!emitc.ptr<!emitc.opaque<"const int16_t">>) -> !emitc.opaque<"int32_t">
        %631 = literal "282" : !emitc.opaque<"size_t">
        %632 = add %35, %631 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %633 = cast %632 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int16_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar"
        %634 = call_opaque "*(const int16_t *)"(%633) : (!emitc.ptr<!emitc.opaque<"const int16_t">>) -> !emitc.opaque<"int32_t">
        %635 = add %630, %634 : (!emitc.opaque<"int32_t">, !emitc.opaque<"int32_t">) -> !emitc.opaque<"int32_t">
        %636 = load %53 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2"
        %637 = call_opaque "__riscv_vwmacc_vx_i32m2"(%636, %635, %469, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %637 : !emitc.opaque<"vint32m2_t"> to %53 : <!emitc.opaque<"vint32m2_t">>
        %638 = load %59 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2"
        %639 = call_opaque "__riscv_vwmacc_vx_i32m2"(%638, %635, %565, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %639 : !emitc.opaque<"vint32m2_t"> to %59 : <!emitc.opaque<"vint32m2_t">>
        %640 = literal "284" : !emitc.opaque<"size_t">
        %641 = add %35, %640 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %642 = cast %641 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int16_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar"
        %643 = call_opaque "*(const int16_t *)"(%642) : (!emitc.ptr<!emitc.opaque<"const int16_t">>) -> !emitc.opaque<"int32_t">
        %644 = literal "286" : !emitc.opaque<"size_t">
        %645 = add %35, %644 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %646 = cast %645 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int16_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar"
        %647 = call_opaque "*(const int16_t *)"(%646) : (!emitc.ptr<!emitc.opaque<"const int16_t">>) -> !emitc.opaque<"int32_t">
        %648 = add %643, %647 : (!emitc.opaque<"int32_t">, !emitc.opaque<"int32_t">) -> !emitc.opaque<"int32_t">
        %649 = load %53 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2"
        %650 = call_opaque "__riscv_vwmacc_vx_i32m2"(%649, %648, %493, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %650 : !emitc.opaque<"vint32m2_t"> to %53 : <!emitc.opaque<"vint32m2_t">>
        %651 = load %59 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2"
        %652 = call_opaque "__riscv_vwmacc_vx_i32m2"(%651, %648, %589, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %652 : !emitc.opaque<"vint32m2_t"> to %59 : <!emitc.opaque<"vint32m2_t">>
        %653 = literal "288" : !emitc.opaque<"size_t">
        %654 = add %35, %653 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %655 = cast %654 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int16_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar"
        %656 = call_opaque "*(const int16_t *)"(%655) : (!emitc.ptr<!emitc.opaque<"const int16_t">>) -> !emitc.opaque<"int32_t">
        %657 = literal "290" : !emitc.opaque<"size_t">
        %658 = add %35, %657 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %659 = cast %658 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int16_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar"
        %660 = call_opaque "*(const int16_t *)"(%659) : (!emitc.ptr<!emitc.opaque<"const int16_t">>) -> !emitc.opaque<"int32_t">
        %661 = add %656, %660 : (!emitc.opaque<"int32_t">, !emitc.opaque<"int32_t">) -> !emitc.opaque<"int32_t">
        %662 = load %53 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2"
        %663 = call_opaque "__riscv_vwmacc_vx_i32m2"(%662, %661, %517, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %663 : !emitc.opaque<"vint32m2_t"> to %53 : <!emitc.opaque<"vint32m2_t">>
        %664 = load %59 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2"
        %665 = call_opaque "__riscv_vwmacc_vx_i32m2"(%664, %661, %613, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %665 : !emitc.opaque<"vint32m2_t"> to %59 : <!emitc.opaque<"vint32m2_t">>
        %666 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %667 = literal "0" : !emitc.opaque<"int32_t">
        %668 = call_opaque "__riscv_vmv_v_x_i16m1"(%667, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %668 : !emitc.opaque<"vint16m1_t"> to %666 : <!emitc.opaque<"vint16m1_t">>
        %669 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %670 = literal "0" : !emitc.opaque<"int32_t">
        %671 = call_opaque "__riscv_vmv_v_x_i16m1"(%670, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %671 : !emitc.opaque<"vint16m1_t"> to %669 : <!emitc.opaque<"vint16m1_t">>
        %672 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %673 = literal "0" : !emitc.opaque<"int32_t">
        %674 = call_opaque "__riscv_vmv_v_x_i16m1"(%673, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %674 : !emitc.opaque<"vint16m1_t"> to %672 : <!emitc.opaque<"vint16m1_t">>
        %675 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %676 = literal "0" : !emitc.opaque<"int32_t">
        %677 = call_opaque "__riscv_vmv_v_x_i16m1"(%676, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %677 : !emitc.opaque<"vint16m1_t"> to %675 : <!emitc.opaque<"vint16m1_t">>
        %678 = literal "1" : !emitc.opaque<"size_t">
        %679 = literal "16" : !emitc.opaque<"size_t">
        %680 = literal "0" : !emitc.opaque<"size_t">
        for %arg7 = %680 to %679 step %678  : !emitc.opaque<"size_t"> {
          %790 = literal "16" : !emitc.opaque<"size_t">
          %791 = mul %arg7, %790 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr"
          %792 = literal "1792" : !emitc.opaque<"size_t">
          %793 = add %792, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %794 = add %32, %793 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %795 = cast %794 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %796 = call_opaque "__riscv_vle8_v_u8mf2"(%795, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %797 = literal "0x0F" : !emitc.opaque<"int">
          %798 = call_opaque "__riscv_vand_vx_u8mf2"(%796, %797, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %799 = literal "4" : !emitc.opaque<"int">
          %800 = call_opaque "__riscv_vsrl_vx_u8mf2"(%796, %799, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          %801 = literal "256" : !emitc.opaque<"size_t">
          %802 = add %801, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %803 = add %32, %802 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %804 = cast %803 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %805 = call_opaque "__riscv_vle8_v_u8mf2"(%804, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %806 = literal "4" : !emitc.opaque<"int">
          %807 = call_opaque "__riscv_vsrl_vx_u8mf2"(%805, %806, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %808 = literal "0x01" : !emitc.opaque<"int">
          %809 = call_opaque "__riscv_vand_vx_u8mf2"(%807, %808, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %810 = literal "4" : !emitc.opaque<"int">
          %811 = call_opaque "__riscv_vsll_vx_u8mf2"(%809, %810, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %812 = literal "5" : !emitc.opaque<"int">
          %813 = call_opaque "__riscv_vsrl_vx_u8mf2"(%805, %812, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %814 = literal "0x01" : !emitc.opaque<"int">
          %815 = call_opaque "__riscv_vand_vx_u8mf2"(%813, %814, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %816 = literal "4" : !emitc.opaque<"int">
          %817 = call_opaque "__riscv_vsll_vx_u8mf2"(%815, %816, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %818 = call_opaque "__riscv_vor_vv_u8mf2"(%798, %811, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %819 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%818) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %820 = call_opaque "__riscv_vor_vv_u8mf2"(%800, %817, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %821 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%820) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr"
          %822 = literal "132" : !emitc.opaque<"size_t">
          %823 = add %822, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %824 = literal "164" : !emitc.opaque<"size_t">
          %825 = add %824, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %826 = add %35, %823 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %827 = cast %826 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %828 = call_opaque "*(const int8_t *)"(%827) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %829 = add %35, %825 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %830 = cast %829 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %831 = call_opaque "*(const int8_t *)"(%830) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %832 = load %666 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %833 = call_opaque "__riscv_vwmacc_vx_i16m1"(%832, %828, %819, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %833 : !emitc.opaque<"vint16m1_t"> to %666 : <!emitc.opaque<"vint16m1_t">>
          %834 = load %669 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %835 = call_opaque "__riscv_vwmacc_vx_i16m1"(%834, %831, %821, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %835 : !emitc.opaque<"vint16m1_t"> to %669 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr"
          %836 = literal "1800" : !emitc.opaque<"size_t">
          %837 = add %836, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %838 = add %32, %837 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %839 = cast %838 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %840 = call_opaque "__riscv_vle8_v_u8mf2"(%839, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %841 = literal "0x0F" : !emitc.opaque<"int">
          %842 = call_opaque "__riscv_vand_vx_u8mf2"(%840, %841, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %843 = literal "4" : !emitc.opaque<"int">
          %844 = call_opaque "__riscv_vsrl_vx_u8mf2"(%840, %843, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          %845 = literal "264" : !emitc.opaque<"size_t">
          %846 = add %845, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %847 = add %32, %846 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %848 = cast %847 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %849 = call_opaque "__riscv_vle8_v_u8mf2"(%848, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %850 = literal "4" : !emitc.opaque<"int">
          %851 = call_opaque "__riscv_vsrl_vx_u8mf2"(%849, %850, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %852 = literal "0x01" : !emitc.opaque<"int">
          %853 = call_opaque "__riscv_vand_vx_u8mf2"(%851, %852, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %854 = literal "4" : !emitc.opaque<"int">
          %855 = call_opaque "__riscv_vsll_vx_u8mf2"(%853, %854, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %856 = literal "5" : !emitc.opaque<"int">
          %857 = call_opaque "__riscv_vsrl_vx_u8mf2"(%849, %856, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %858 = literal "0x01" : !emitc.opaque<"int">
          %859 = call_opaque "__riscv_vand_vx_u8mf2"(%857, %858, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %860 = literal "4" : !emitc.opaque<"int">
          %861 = call_opaque "__riscv_vsll_vx_u8mf2"(%859, %860, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %862 = call_opaque "__riscv_vor_vv_u8mf2"(%842, %855, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %863 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%862) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %864 = call_opaque "__riscv_vor_vv_u8mf2"(%844, %861, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %865 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%864) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr"
          %866 = literal "132" : !emitc.opaque<"size_t">
          %867 = add %866, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %868 = literal "164" : !emitc.opaque<"size_t">
          %869 = add %868, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %870 = add %35, %867 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %871 = cast %870 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %872 = call_opaque "*(const int8_t *)"(%871) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %873 = add %35, %869 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %874 = cast %873 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %875 = call_opaque "*(const int8_t *)"(%874) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %876 = load %672 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %877 = call_opaque "__riscv_vwmacc_vx_i16m1"(%876, %872, %863, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %877 : !emitc.opaque<"vint16m1_t"> to %672 : <!emitc.opaque<"vint16m1_t">>
          %878 = load %675 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %879 = call_opaque "__riscv_vwmacc_vx_i16m1"(%878, %875, %865, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %879 : !emitc.opaque<"vint16m1_t"> to %675 : <!emitc.opaque<"vint16m1_t">>
        }
        %681 = load %666 : <!emitc.opaque<"vint16m1_t">>
        %682 = load %669 : <!emitc.opaque<"vint16m1_t">>
        %683 = load %672 : <!emitc.opaque<"vint16m1_t">>
        %684 = load %675 : <!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold"
        %685 = load %50 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %686 = call_opaque "__riscv_vwmacc_vv_i32m2"(%685, %443, %681, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %687 = call_opaque "__riscv_vwmacc_vv_i32m2"(%686, %467, %682, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %687 : !emitc.opaque<"vint32m2_t"> to %50 : <!emitc.opaque<"vint32m2_t">>
        %688 = load %56 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %689 = call_opaque "__riscv_vwmacc_vv_i32m2"(%688, %539, %683, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %690 = call_opaque "__riscv_vwmacc_vv_i32m2"(%689, %563, %684, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %690 : !emitc.opaque<"vint32m2_t"> to %56 : <!emitc.opaque<"vint32m2_t">>
        %691 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %692 = literal "0" : !emitc.opaque<"int32_t">
        %693 = call_opaque "__riscv_vmv_v_x_i16m1"(%692, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %693 : !emitc.opaque<"vint16m1_t"> to %691 : <!emitc.opaque<"vint16m1_t">>
        %694 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %695 = literal "0" : !emitc.opaque<"int32_t">
        %696 = call_opaque "__riscv_vmv_v_x_i16m1"(%695, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %696 : !emitc.opaque<"vint16m1_t"> to %694 : <!emitc.opaque<"vint16m1_t">>
        %697 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %698 = literal "0" : !emitc.opaque<"int32_t">
        %699 = call_opaque "__riscv_vmv_v_x_i16m1"(%698, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %699 : !emitc.opaque<"vint16m1_t"> to %697 : <!emitc.opaque<"vint16m1_t">>
        %700 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %701 = literal "0" : !emitc.opaque<"int32_t">
        %702 = call_opaque "__riscv_vmv_v_x_i16m1"(%701, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %702 : !emitc.opaque<"vint16m1_t"> to %700 : <!emitc.opaque<"vint16m1_t">>
        %703 = literal "1" : !emitc.opaque<"size_t">
        %704 = literal "16" : !emitc.opaque<"size_t">
        %705 = literal "0" : !emitc.opaque<"size_t">
        for %arg7 = %705 to %704 step %703  : !emitc.opaque<"size_t"> {
          %790 = literal "16" : !emitc.opaque<"size_t">
          %791 = mul %arg7, %790 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr"
          %792 = literal "2048" : !emitc.opaque<"size_t">
          %793 = add %792, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %794 = add %32, %793 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %795 = cast %794 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %796 = call_opaque "__riscv_vle8_v_u8mf2"(%795, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %797 = literal "0x0F" : !emitc.opaque<"int">
          %798 = call_opaque "__riscv_vand_vx_u8mf2"(%796, %797, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %799 = literal "4" : !emitc.opaque<"int">
          %800 = call_opaque "__riscv_vsrl_vx_u8mf2"(%796, %799, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          %801 = literal "512" : !emitc.opaque<"size_t">
          %802 = add %801, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %803 = add %32, %802 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %804 = cast %803 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %805 = call_opaque "__riscv_vle8_v_u8mf2"(%804, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %806 = literal "4" : !emitc.opaque<"int">
          %807 = call_opaque "__riscv_vsrl_vx_u8mf2"(%805, %806, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %808 = literal "0x01" : !emitc.opaque<"int">
          %809 = call_opaque "__riscv_vand_vx_u8mf2"(%807, %808, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %810 = literal "4" : !emitc.opaque<"int">
          %811 = call_opaque "__riscv_vsll_vx_u8mf2"(%809, %810, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %812 = literal "5" : !emitc.opaque<"int">
          %813 = call_opaque "__riscv_vsrl_vx_u8mf2"(%805, %812, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %814 = literal "0x01" : !emitc.opaque<"int">
          %815 = call_opaque "__riscv_vand_vx_u8mf2"(%813, %814, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %816 = literal "4" : !emitc.opaque<"int">
          %817 = call_opaque "__riscv_vsll_vx_u8mf2"(%815, %816, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %818 = call_opaque "__riscv_vor_vv_u8mf2"(%798, %811, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %819 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%818) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %820 = call_opaque "__riscv_vor_vv_u8mf2"(%800, %817, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %821 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%820) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr"
          %822 = literal "148" : !emitc.opaque<"size_t">
          %823 = add %822, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %824 = literal "180" : !emitc.opaque<"size_t">
          %825 = add %824, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %826 = add %35, %823 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %827 = cast %826 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %828 = call_opaque "*(const int8_t *)"(%827) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %829 = add %35, %825 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %830 = cast %829 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %831 = call_opaque "*(const int8_t *)"(%830) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %832 = load %691 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %833 = call_opaque "__riscv_vwmacc_vx_i16m1"(%832, %828, %819, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %833 : !emitc.opaque<"vint16m1_t"> to %691 : <!emitc.opaque<"vint16m1_t">>
          %834 = load %694 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %835 = call_opaque "__riscv_vwmacc_vx_i16m1"(%834, %831, %821, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %835 : !emitc.opaque<"vint16m1_t"> to %694 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr"
          %836 = literal "2056" : !emitc.opaque<"size_t">
          %837 = add %836, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %838 = add %32, %837 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %839 = cast %838 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %840 = call_opaque "__riscv_vle8_v_u8mf2"(%839, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %841 = literal "0x0F" : !emitc.opaque<"int">
          %842 = call_opaque "__riscv_vand_vx_u8mf2"(%840, %841, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %843 = literal "4" : !emitc.opaque<"int">
          %844 = call_opaque "__riscv_vsrl_vx_u8mf2"(%840, %843, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          %845 = literal "520" : !emitc.opaque<"size_t">
          %846 = add %845, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %847 = add %32, %846 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %848 = cast %847 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %849 = call_opaque "__riscv_vle8_v_u8mf2"(%848, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %850 = literal "4" : !emitc.opaque<"int">
          %851 = call_opaque "__riscv_vsrl_vx_u8mf2"(%849, %850, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %852 = literal "0x01" : !emitc.opaque<"int">
          %853 = call_opaque "__riscv_vand_vx_u8mf2"(%851, %852, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %854 = literal "4" : !emitc.opaque<"int">
          %855 = call_opaque "__riscv_vsll_vx_u8mf2"(%853, %854, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %856 = literal "5" : !emitc.opaque<"int">
          %857 = call_opaque "__riscv_vsrl_vx_u8mf2"(%849, %856, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %858 = literal "0x01" : !emitc.opaque<"int">
          %859 = call_opaque "__riscv_vand_vx_u8mf2"(%857, %858, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %860 = literal "4" : !emitc.opaque<"int">
          %861 = call_opaque "__riscv_vsll_vx_u8mf2"(%859, %860, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %862 = call_opaque "__riscv_vor_vv_u8mf2"(%842, %855, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %863 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%862) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %864 = call_opaque "__riscv_vor_vv_u8mf2"(%844, %861, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %865 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%864) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr"
          %866 = literal "148" : !emitc.opaque<"size_t">
          %867 = add %866, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %868 = literal "180" : !emitc.opaque<"size_t">
          %869 = add %868, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %870 = add %35, %867 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %871 = cast %870 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %872 = call_opaque "*(const int8_t *)"(%871) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %873 = add %35, %869 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %874 = cast %873 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %875 = call_opaque "*(const int8_t *)"(%874) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %876 = load %697 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %877 = call_opaque "__riscv_vwmacc_vx_i16m1"(%876, %872, %863, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %877 : !emitc.opaque<"vint16m1_t"> to %697 : <!emitc.opaque<"vint16m1_t">>
          %878 = load %700 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %879 = call_opaque "__riscv_vwmacc_vx_i16m1"(%878, %875, %865, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %879 : !emitc.opaque<"vint16m1_t"> to %700 : <!emitc.opaque<"vint16m1_t">>
        }
        %706 = load %691 : <!emitc.opaque<"vint16m1_t">>
        %707 = load %694 : <!emitc.opaque<"vint16m1_t">>
        %708 = load %697 : <!emitc.opaque<"vint16m1_t">>
        %709 = load %700 : <!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold"
        %710 = load %50 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %711 = call_opaque "__riscv_vwmacc_vv_i32m2"(%710, %443, %706, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %712 = call_opaque "__riscv_vwmacc_vv_i32m2"(%711, %467, %707, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %712 : !emitc.opaque<"vint32m2_t"> to %50 : <!emitc.opaque<"vint32m2_t">>
        %713 = load %56 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %714 = call_opaque "__riscv_vwmacc_vv_i32m2"(%713, %539, %708, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %715 = call_opaque "__riscv_vwmacc_vv_i32m2"(%714, %563, %709, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %715 : !emitc.opaque<"vint32m2_t"> to %56 : <!emitc.opaque<"vint32m2_t">>
        %716 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %717 = literal "0" : !emitc.opaque<"int32_t">
        %718 = call_opaque "__riscv_vmv_v_x_i16m1"(%717, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %718 : !emitc.opaque<"vint16m1_t"> to %716 : <!emitc.opaque<"vint16m1_t">>
        %719 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %720 = literal "0" : !emitc.opaque<"int32_t">
        %721 = call_opaque "__riscv_vmv_v_x_i16m1"(%720, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %721 : !emitc.opaque<"vint16m1_t"> to %719 : <!emitc.opaque<"vint16m1_t">>
        %722 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %723 = literal "0" : !emitc.opaque<"int32_t">
        %724 = call_opaque "__riscv_vmv_v_x_i16m1"(%723, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %724 : !emitc.opaque<"vint16m1_t"> to %722 : <!emitc.opaque<"vint16m1_t">>
        %725 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %726 = literal "0" : !emitc.opaque<"int32_t">
        %727 = call_opaque "__riscv_vmv_v_x_i16m1"(%726, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %727 : !emitc.opaque<"vint16m1_t"> to %725 : <!emitc.opaque<"vint16m1_t">>
        %728 = literal "1" : !emitc.opaque<"size_t">
        %729 = literal "16" : !emitc.opaque<"size_t">
        %730 = literal "0" : !emitc.opaque<"size_t">
        for %arg7 = %730 to %729 step %728  : !emitc.opaque<"size_t"> {
          %790 = literal "16" : !emitc.opaque<"size_t">
          %791 = mul %arg7, %790 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr"
          %792 = literal "2304" : !emitc.opaque<"size_t">
          %793 = add %792, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %794 = add %32, %793 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %795 = cast %794 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %796 = call_opaque "__riscv_vle8_v_u8mf2"(%795, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %797 = literal "0x0F" : !emitc.opaque<"int">
          %798 = call_opaque "__riscv_vand_vx_u8mf2"(%796, %797, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %799 = literal "4" : !emitc.opaque<"int">
          %800 = call_opaque "__riscv_vsrl_vx_u8mf2"(%796, %799, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          %801 = literal "256" : !emitc.opaque<"size_t">
          %802 = add %801, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %803 = add %32, %802 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %804 = cast %803 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %805 = call_opaque "__riscv_vle8_v_u8mf2"(%804, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %806 = literal "6" : !emitc.opaque<"int">
          %807 = call_opaque "__riscv_vsrl_vx_u8mf2"(%805, %806, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %808 = literal "0x01" : !emitc.opaque<"int">
          %809 = call_opaque "__riscv_vand_vx_u8mf2"(%807, %808, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %810 = literal "4" : !emitc.opaque<"int">
          %811 = call_opaque "__riscv_vsll_vx_u8mf2"(%809, %810, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %812 = literal "7" : !emitc.opaque<"int">
          %813 = call_opaque "__riscv_vsrl_vx_u8mf2"(%805, %812, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %814 = literal "0x01" : !emitc.opaque<"int">
          %815 = call_opaque "__riscv_vand_vx_u8mf2"(%813, %814, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %816 = literal "4" : !emitc.opaque<"int">
          %817 = call_opaque "__riscv_vsll_vx_u8mf2"(%815, %816, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %818 = call_opaque "__riscv_vor_vv_u8mf2"(%798, %811, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %819 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%818) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %820 = call_opaque "__riscv_vor_vv_u8mf2"(%800, %817, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %821 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%820) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr"
          %822 = literal "196" : !emitc.opaque<"size_t">
          %823 = add %822, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %824 = literal "228" : !emitc.opaque<"size_t">
          %825 = add %824, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %826 = add %35, %823 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %827 = cast %826 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %828 = call_opaque "*(const int8_t *)"(%827) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %829 = add %35, %825 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %830 = cast %829 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %831 = call_opaque "*(const int8_t *)"(%830) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %832 = load %716 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %833 = call_opaque "__riscv_vwmacc_vx_i16m1"(%832, %828, %819, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %833 : !emitc.opaque<"vint16m1_t"> to %716 : <!emitc.opaque<"vint16m1_t">>
          %834 = load %719 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %835 = call_opaque "__riscv_vwmacc_vx_i16m1"(%834, %831, %821, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %835 : !emitc.opaque<"vint16m1_t"> to %719 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr"
          %836 = literal "2312" : !emitc.opaque<"size_t">
          %837 = add %836, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %838 = add %32, %837 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %839 = cast %838 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %840 = call_opaque "__riscv_vle8_v_u8mf2"(%839, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %841 = literal "0x0F" : !emitc.opaque<"int">
          %842 = call_opaque "__riscv_vand_vx_u8mf2"(%840, %841, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %843 = literal "4" : !emitc.opaque<"int">
          %844 = call_opaque "__riscv_vsrl_vx_u8mf2"(%840, %843, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          %845 = literal "264" : !emitc.opaque<"size_t">
          %846 = add %845, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %847 = add %32, %846 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %848 = cast %847 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %849 = call_opaque "__riscv_vle8_v_u8mf2"(%848, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %850 = literal "6" : !emitc.opaque<"int">
          %851 = call_opaque "__riscv_vsrl_vx_u8mf2"(%849, %850, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %852 = literal "0x01" : !emitc.opaque<"int">
          %853 = call_opaque "__riscv_vand_vx_u8mf2"(%851, %852, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %854 = literal "4" : !emitc.opaque<"int">
          %855 = call_opaque "__riscv_vsll_vx_u8mf2"(%853, %854, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %856 = literal "7" : !emitc.opaque<"int">
          %857 = call_opaque "__riscv_vsrl_vx_u8mf2"(%849, %856, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %858 = literal "0x01" : !emitc.opaque<"int">
          %859 = call_opaque "__riscv_vand_vx_u8mf2"(%857, %858, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %860 = literal "4" : !emitc.opaque<"int">
          %861 = call_opaque "__riscv_vsll_vx_u8mf2"(%859, %860, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %862 = call_opaque "__riscv_vor_vv_u8mf2"(%842, %855, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %863 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%862) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %864 = call_opaque "__riscv_vor_vv_u8mf2"(%844, %861, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %865 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%864) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr"
          %866 = literal "196" : !emitc.opaque<"size_t">
          %867 = add %866, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %868 = literal "228" : !emitc.opaque<"size_t">
          %869 = add %868, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %870 = add %35, %867 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %871 = cast %870 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %872 = call_opaque "*(const int8_t *)"(%871) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %873 = add %35, %869 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %874 = cast %873 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %875 = call_opaque "*(const int8_t *)"(%874) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %876 = load %722 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %877 = call_opaque "__riscv_vwmacc_vx_i16m1"(%876, %872, %863, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %877 : !emitc.opaque<"vint16m1_t"> to %722 : <!emitc.opaque<"vint16m1_t">>
          %878 = load %725 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %879 = call_opaque "__riscv_vwmacc_vx_i16m1"(%878, %875, %865, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %879 : !emitc.opaque<"vint16m1_t"> to %725 : <!emitc.opaque<"vint16m1_t">>
        }
        %731 = load %716 : <!emitc.opaque<"vint16m1_t">>
        %732 = load %719 : <!emitc.opaque<"vint16m1_t">>
        %733 = load %722 : <!emitc.opaque<"vint16m1_t">>
        %734 = load %725 : <!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold"
        %735 = load %50 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %736 = call_opaque "__riscv_vwmacc_vv_i32m2"(%735, %491, %731, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %737 = call_opaque "__riscv_vwmacc_vv_i32m2"(%736, %515, %732, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %737 : !emitc.opaque<"vint32m2_t"> to %50 : <!emitc.opaque<"vint32m2_t">>
        %738 = load %56 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %739 = call_opaque "__riscv_vwmacc_vv_i32m2"(%738, %587, %733, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %740 = call_opaque "__riscv_vwmacc_vv_i32m2"(%739, %611, %734, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %740 : !emitc.opaque<"vint32m2_t"> to %56 : <!emitc.opaque<"vint32m2_t">>
        %741 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %742 = literal "0" : !emitc.opaque<"int32_t">
        %743 = call_opaque "__riscv_vmv_v_x_i16m1"(%742, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %743 : !emitc.opaque<"vint16m1_t"> to %741 : <!emitc.opaque<"vint16m1_t">>
        %744 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %745 = literal "0" : !emitc.opaque<"int32_t">
        %746 = call_opaque "__riscv_vmv_v_x_i16m1"(%745, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %746 : !emitc.opaque<"vint16m1_t"> to %744 : <!emitc.opaque<"vint16m1_t">>
        %747 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %748 = literal "0" : !emitc.opaque<"int32_t">
        %749 = call_opaque "__riscv_vmv_v_x_i16m1"(%748, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %749 : !emitc.opaque<"vint16m1_t"> to %747 : <!emitc.opaque<"vint16m1_t">>
        %750 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %751 = literal "0" : !emitc.opaque<"int32_t">
        %752 = call_opaque "__riscv_vmv_v_x_i16m1"(%751, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %752 : !emitc.opaque<"vint16m1_t"> to %750 : <!emitc.opaque<"vint16m1_t">>
        %753 = literal "1" : !emitc.opaque<"size_t">
        %754 = literal "16" : !emitc.opaque<"size_t">
        %755 = literal "0" : !emitc.opaque<"size_t">
        for %arg7 = %755 to %754 step %753  : !emitc.opaque<"size_t"> {
          %790 = literal "16" : !emitc.opaque<"size_t">
          %791 = mul %arg7, %790 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr"
          %792 = literal "2560" : !emitc.opaque<"size_t">
          %793 = add %792, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %794 = add %32, %793 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %795 = cast %794 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %796 = call_opaque "__riscv_vle8_v_u8mf2"(%795, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %797 = literal "0x0F" : !emitc.opaque<"int">
          %798 = call_opaque "__riscv_vand_vx_u8mf2"(%796, %797, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %799 = literal "4" : !emitc.opaque<"int">
          %800 = call_opaque "__riscv_vsrl_vx_u8mf2"(%796, %799, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          %801 = literal "512" : !emitc.opaque<"size_t">
          %802 = add %801, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %803 = add %32, %802 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %804 = cast %803 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %805 = call_opaque "__riscv_vle8_v_u8mf2"(%804, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %806 = literal "6" : !emitc.opaque<"int">
          %807 = call_opaque "__riscv_vsrl_vx_u8mf2"(%805, %806, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %808 = literal "0x01" : !emitc.opaque<"int">
          %809 = call_opaque "__riscv_vand_vx_u8mf2"(%807, %808, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %810 = literal "4" : !emitc.opaque<"int">
          %811 = call_opaque "__riscv_vsll_vx_u8mf2"(%809, %810, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %812 = literal "7" : !emitc.opaque<"int">
          %813 = call_opaque "__riscv_vsrl_vx_u8mf2"(%805, %812, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %814 = literal "0x01" : !emitc.opaque<"int">
          %815 = call_opaque "__riscv_vand_vx_u8mf2"(%813, %814, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %816 = literal "4" : !emitc.opaque<"int">
          %817 = call_opaque "__riscv_vsll_vx_u8mf2"(%815, %816, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %818 = call_opaque "__riscv_vor_vv_u8mf2"(%798, %811, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %819 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%818) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %820 = call_opaque "__riscv_vor_vv_u8mf2"(%800, %817, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %821 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%820) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr"
          %822 = literal "212" : !emitc.opaque<"size_t">
          %823 = add %822, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %824 = literal "244" : !emitc.opaque<"size_t">
          %825 = add %824, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %826 = add %35, %823 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %827 = cast %826 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %828 = call_opaque "*(const int8_t *)"(%827) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %829 = add %35, %825 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %830 = cast %829 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %831 = call_opaque "*(const int8_t *)"(%830) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %832 = load %741 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %833 = call_opaque "__riscv_vwmacc_vx_i16m1"(%832, %828, %819, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %833 : !emitc.opaque<"vint16m1_t"> to %741 : <!emitc.opaque<"vint16m1_t">>
          %834 = load %744 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %835 = call_opaque "__riscv_vwmacc_vx_i16m1"(%834, %831, %821, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %835 : !emitc.opaque<"vint16m1_t"> to %744 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr"
          %836 = literal "2568" : !emitc.opaque<"size_t">
          %837 = add %836, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %838 = add %32, %837 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %839 = cast %838 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %840 = call_opaque "__riscv_vle8_v_u8mf2"(%839, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %841 = literal "0x0F" : !emitc.opaque<"int">
          %842 = call_opaque "__riscv_vand_vx_u8mf2"(%840, %841, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %843 = literal "4" : !emitc.opaque<"int">
          %844 = call_opaque "__riscv_vsrl_vx_u8mf2"(%840, %843, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          %845 = literal "520" : !emitc.opaque<"size_t">
          %846 = add %845, %791 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %847 = add %32, %846 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %848 = cast %847 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %849 = call_opaque "__riscv_vle8_v_u8mf2"(%848, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %850 = literal "6" : !emitc.opaque<"int">
          %851 = call_opaque "__riscv_vsrl_vx_u8mf2"(%849, %850, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %852 = literal "0x01" : !emitc.opaque<"int">
          %853 = call_opaque "__riscv_vand_vx_u8mf2"(%851, %852, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %854 = literal "4" : !emitc.opaque<"int">
          %855 = call_opaque "__riscv_vsll_vx_u8mf2"(%853, %854, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %856 = literal "7" : !emitc.opaque<"int">
          %857 = call_opaque "__riscv_vsrl_vx_u8mf2"(%849, %856, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %858 = literal "0x01" : !emitc.opaque<"int">
          %859 = call_opaque "__riscv_vand_vx_u8mf2"(%857, %858, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2"
          %860 = literal "4" : !emitc.opaque<"int">
          %861 = call_opaque "__riscv_vsll_vx_u8mf2"(%859, %860, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %862 = call_opaque "__riscv_vor_vv_u8mf2"(%842, %855, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %863 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%862) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2"
          %864 = call_opaque "__riscv_vor_vv_u8mf2"(%844, %861, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %865 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%864) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr"
          %866 = literal "212" : !emitc.opaque<"size_t">
          %867 = add %866, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %868 = literal "244" : !emitc.opaque<"size_t">
          %869 = add %868, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %870 = add %35, %867 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %871 = cast %870 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %872 = call_opaque "*(const int8_t *)"(%871) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %873 = add %35, %869 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %874 = cast %873 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar"
          %875 = call_opaque "*(const int8_t *)"(%874) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %876 = load %747 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %877 = call_opaque "__riscv_vwmacc_vx_i16m1"(%876, %872, %863, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %877 : !emitc.opaque<"vint16m1_t"> to %747 : <!emitc.opaque<"vint16m1_t">>
          %878 = load %750 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %879 = call_opaque "__riscv_vwmacc_vx_i16m1"(%878, %875, %865, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %879 : !emitc.opaque<"vint16m1_t"> to %750 : <!emitc.opaque<"vint16m1_t">>
        }
        %756 = load %741 : <!emitc.opaque<"vint16m1_t">>
        %757 = load %744 : <!emitc.opaque<"vint16m1_t">>
        %758 = load %747 : <!emitc.opaque<"vint16m1_t">>
        %759 = load %750 : <!emitc.opaque<"vint16m1_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold"
        %760 = load %50 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %761 = call_opaque "__riscv_vwmacc_vv_i32m2"(%760, %491, %756, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %762 = call_opaque "__riscv_vwmacc_vv_i32m2"(%761, %515, %757, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %762 : !emitc.opaque<"vint32m2_t"> to %50 : <!emitc.opaque<"vint32m2_t">>
        %763 = load %56 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %764 = call_opaque "__riscv_vwmacc_vv_i32m2"(%763, %587, %758, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2"
        %765 = call_opaque "__riscv_vwmacc_vv_i32m2"(%764, %611, %759, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        assign %765 : !emitc.opaque<"vint32m2_t"> to %56 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr"
        %766 = cast %32 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const _Float16">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1"
        %767 = call_opaque "__riscv_vle16_v_f16m1"(%766, %1) : (!emitc.ptr<!emitc.opaque<"const _Float16">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2"
        %768 = call_opaque "__riscv_vfwcvt_f_f_v_f32m2"(%767, %1) : (!emitc.opaque<"vfloat16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2"
        %769 = call_opaque "__riscv_vfmul_vf_f32m2"(%768, %37, %1) : (!emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"float">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        %770 = load %50 : <!emitc.opaque<"vint32m2_t">>
        %771 = load %12 : <!emitc.opaque<"vfloat32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2"
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2"
        %772 = call_opaque "__riscv_vfcvt_f_x_v_f32m2"(%770, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        %773 = call_opaque "__riscv_vfmacc_vv_f32m2"(%771, %772, %769, %1) : (!emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        %774 = load %53 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2"
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2"
        %775 = call_opaque "__riscv_vfcvt_f_x_v_f32m2"(%774, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        %776 = call_opaque "__riscv_vfnmsac_vv_f32m2"(%773, %43, %775, %1) : (!emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        assign %776 : !emitc.opaque<"vfloat32m2_t"> to %12 : <!emitc.opaque<"vfloat32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr"
        %777 = literal "16" : !emitc.opaque<"size_t">
        %778 = add %32, %777 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %779 = cast %778 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const _Float16">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1"
        %780 = call_opaque "__riscv_vle16_v_f16m1"(%779, %1) : (!emitc.ptr<!emitc.opaque<"const _Float16">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat16m1_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2"
        %781 = call_opaque "__riscv_vfwcvt_f_f_v_f32m2"(%780, %1) : (!emitc.opaque<"vfloat16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2"
        %782 = call_opaque "__riscv_vfmul_vf_f32m2"(%781, %37, %1) : (!emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"float">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        %783 = load %56 : <!emitc.opaque<"vint32m2_t">>
        %784 = load %15 : <!emitc.opaque<"vfloat32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2"
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2"
        %785 = call_opaque "__riscv_vfcvt_f_x_v_f32m2"(%783, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        %786 = call_opaque "__riscv_vfmacc_vv_f32m2"(%784, %785, %782, %1) : (!emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        %787 = load %59 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2"
        verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2"
        %788 = call_opaque "__riscv_vfcvt_f_x_v_f32m2"(%787, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        %789 = call_opaque "__riscv_vfnmsac_vv_f32m2"(%786, %49, %788, %1) : (!emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        assign %789 : !emitc.opaque<"vfloat32m2_t"> to %15 : <!emitc.opaque<"vfloat32m2_t">>
      }
      verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr"
      %20 = literal "16" : !emitc.opaque<"size_t">
      %21 = mul %arg5, %20 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %22 = add %arg1, %21 : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"float">>
      %23 = load %12 : <!emitc.opaque<"vfloat32m2_t">>
      verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2"
      call_opaque "__riscv_vse32_v_f32m2"(%22, %23, %1) : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr"
      %24 = literal "16" : !emitc.opaque<"size_t">
      %25 = mul %arg5, %24 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %26 = literal "8" : !emitc.opaque<"size_t">
      %27 = add %25, %26 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %28 = add %arg1, %27 : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"float">>
      %29 = load %15 : <!emitc.opaque<"vfloat32m2_t">>
      verbatim "// weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2"
      call_opaque "__riscv_vse32_v_f32m2"(%28, %29, %1) : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> ()
    }
    return
  }
}

