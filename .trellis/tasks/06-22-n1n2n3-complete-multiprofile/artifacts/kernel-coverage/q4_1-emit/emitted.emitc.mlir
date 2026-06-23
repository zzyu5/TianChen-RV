module {
  emitc.include <"stddef.h">
  emitc.include <"stdint.h">
  emitc.include <"riscv_vector.h">
  emitc.func @tcrv_emitc_ggml_repack_gemv_q4_1_q8_1_kernel_ggml_repack_gemv_q4_1_q8_1(%arg0: !emitc.opaque<"size_t">, %arg1: !emitc.ptr<!emitc.opaque<"float">>, %arg2: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg3: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg4: !emitc.opaque<"size_t">) attributes {specifiers = ["extern", "\22C\22"]} {
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1"
    %0 = call_opaque "__riscv_vsetvl_e32m1"(%arg0) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface"
    %1 = literal "8" : !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count"
    %2 = literal "32" : !emitc.opaque<"size_t">
    %3 = div %arg0, %2 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=col_group_count"
    %4 = literal "16" : !emitc.opaque<"size_t">
    %5 = div %arg4, %4 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    %6 = literal "1" : !emitc.opaque<"size_t">
    %7 = literal "0" : !emitc.opaque<"size_t">
    for %arg5 = %7 to %5 step %6  : !emitc.opaque<"size_t"> {
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_group_base"
      %11 = mul %arg5, %3 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %12 = literal "320" : !emitc.opaque<"size_t">
      %13 = mul %11, %12 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %14 = add %arg2, %13 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %15 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m2_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2"
      %16 = literal "0.0f" : !emitc.opaque<"float">
      %17 = call_opaque "__riscv_vfmv_v_f_f32m2"(%16, %1) : (!emitc.opaque<"float">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
      assign %17 : !emitc.opaque<"vfloat32m2_t"> to %15 : <!emitc.opaque<"vfloat32m2_t">>
      %18 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m2_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2"
      %19 = literal "0.0f" : !emitc.opaque<"float">
      %20 = call_opaque "__riscv_vfmv_v_f_f32m2"(%19, %1) : (!emitc.opaque<"float">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
      assign %20 : !emitc.opaque<"vfloat32m2_t"> to %18 : <!emitc.opaque<"vfloat32m2_t">>
      %21 = literal "1" : !emitc.opaque<"size_t">
      %22 = literal "0" : !emitc.opaque<"size_t">
      for %arg6 = %22 to %3 step %21  : !emitc.opaque<"size_t"> {
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base"
        %33 = literal "320" : !emitc.opaque<"size_t">
        %34 = mul %arg6, %33 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %35 = add %14, %34 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base"
        %36 = literal "36" : !emitc.opaque<"size_t">
        %37 = mul %arg6, %36 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %38 = add %arg3, %37 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %39 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %40 = literal "0" : !emitc.opaque<"int32_t">
        %41 = call_opaque "__riscv_vmv_v_x_i16m1"(%40, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %41 : !emitc.opaque<"vint16m1_t"> to %39 : <!emitc.opaque<"vint16m1_t">>
        %42 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %43 = literal "0" : !emitc.opaque<"int32_t">
        %44 = call_opaque "__riscv_vmv_v_x_i16m1"(%43, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %44 : !emitc.opaque<"vint16m1_t"> to %42 : <!emitc.opaque<"vint16m1_t">>
        %45 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %46 = literal "0" : !emitc.opaque<"int32_t">
        %47 = call_opaque "__riscv_vmv_v_x_i16m1"(%46, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %47 : !emitc.opaque<"vint16m1_t"> to %45 : <!emitc.opaque<"vint16m1_t">>
        %48 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %49 = literal "0" : !emitc.opaque<"int32_t">
        %50 = call_opaque "__riscv_vmv_v_x_i16m1"(%49, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %50 : !emitc.opaque<"vint16m1_t"> to %48 : <!emitc.opaque<"vint16m1_t">>
        %51 = literal "1" : !emitc.opaque<"size_t">
        %52 = literal "16" : !emitc.opaque<"size_t">
        %53 = literal "0" : !emitc.opaque<"size_t">
        for %arg7 = %53 to %52 step %51  : !emitc.opaque<"size_t"> {
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr"
          %92 = literal "16" : !emitc.opaque<"size_t">
          %93 = mul %arg7, %92 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %94 = literal "64" : !emitc.opaque<"size_t">
          %95 = add %94, %93 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %96 = literal "8" : !emitc.opaque<"size_t">
          %97 = add %95, %96 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %98 = add %35, %95 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %99 = cast %98 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %100 = call_opaque "__riscv_vle8_v_u8mf2"(%99, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          %101 = add %35, %97 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %102 = cast %101 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2"
          %103 = call_opaque "__riscv_vle8_v_u8mf2"(%102, %1) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %104 = literal "0x0F" : !emitc.opaque<"int">
          %105 = call_opaque "__riscv_vand_vx_u8mf2"(%100, %104, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %106 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%105) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %107 = literal "0x04" : !emitc.opaque<"int">
          %108 = call_opaque "__riscv_vsrl_vx_u8mf2"(%100, %107, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %109 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%108) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2"
          %110 = literal "0x0F" : !emitc.opaque<"int">
          %111 = call_opaque "__riscv_vand_vx_u8mf2"(%103, %110, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %112 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%111) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2"
          %113 = literal "0x04" : !emitc.opaque<"int">
          %114 = call_opaque "__riscv_vsrl_vx_u8mf2"(%103, %113, %1) : (!emitc.opaque<"vuint8mf2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2"
          %115 = call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"(%114) : (!emitc.opaque<"vuint8mf2_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo"
          %116 = literal "4" : !emitc.opaque<"size_t">
          %117 = add %116, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %118 = add %38, %117 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %119 = cast %118 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar"
          %120 = call_opaque "*(const int8_t *)"(%119) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi"
          %121 = literal "16" : !emitc.opaque<"size_t">
          %122 = literal "4" : !emitc.opaque<"size_t">
          %123 = add %122, %121 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %124 = add %123, %arg7 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %125 = add %38, %124 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %126 = cast %125 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar"
          %127 = call_opaque "*(const int8_t *)"(%126) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %128 = load %39 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %129 = call_opaque "__riscv_vwmacc_vx_i16m1"(%128, %120, %106, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %129 : !emitc.opaque<"vint16m1_t"> to %39 : <!emitc.opaque<"vint16m1_t">>
          %130 = load %42 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %131 = call_opaque "__riscv_vwmacc_vx_i16m1"(%130, %127, %109, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %131 : !emitc.opaque<"vint16m1_t"> to %42 : <!emitc.opaque<"vint16m1_t">>
          %132 = load %45 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %133 = call_opaque "__riscv_vwmacc_vx_i16m1"(%132, %120, %112, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %133 : !emitc.opaque<"vint16m1_t"> to %45 : <!emitc.opaque<"vint16m1_t">>
          %134 = load %48 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %135 = call_opaque "__riscv_vwmacc_vx_i16m1"(%134, %127, %115, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %135 : !emitc.opaque<"vint16m1_t"> to %48 : <!emitc.opaque<"vint16m1_t">>
        }
        %54 = load %39 : <!emitc.opaque<"vint16m1_t">>
        %55 = load %42 : <!emitc.opaque<"vint16m1_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2"
        %56 = call_opaque "__riscv_vwadd_vv_i32m2"(%54, %55, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        %57 = load %45 : <!emitc.opaque<"vint16m1_t">>
        %58 = load %48 : <!emitc.opaque<"vint16m1_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2"
        %59 = call_opaque "__riscv_vwadd_vv_i32m2"(%57, %58, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr"
        %60 = cast %35 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const _Float16">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1"
        %61 = call_opaque "__riscv_vle16_v_f16m1"(%60, %1) : (!emitc.ptr<!emitc.opaque<"const _Float16">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat16m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr"
        %62 = literal "16" : !emitc.opaque<"size_t">
        %63 = add %35, %62 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %64 = cast %63 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const _Float16">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1"
        %65 = call_opaque "__riscv_vle16_v_f16m1"(%64, %1) : (!emitc.ptr<!emitc.opaque<"const _Float16">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat16m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr"
        %66 = literal "32" : !emitc.opaque<"size_t">
        %67 = add %35, %66 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %68 = cast %67 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const _Float16">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1"
        %69 = call_opaque "__riscv_vle16_v_f16m1"(%68, %1) : (!emitc.ptr<!emitc.opaque<"const _Float16">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat16m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr"
        %70 = literal "48" : !emitc.opaque<"size_t">
        %71 = add %35, %70 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %72 = cast %71 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const _Float16">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1"
        %73 = call_opaque "__riscv_vle16_v_f16m1"(%72, %1) : (!emitc.ptr<!emitc.opaque<"const _Float16">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat16m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar"
        %74 = cast %38 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const _Float16">>
        %75 = call_opaque "*(const _Float16 *)"(%74) : (!emitc.ptr<!emitc.opaque<"const _Float16">>) -> !emitc.opaque<"_Float16">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_sum_scalar"
        %76 = literal "2" : !emitc.opaque<"size_t">
        %77 = add %38, %76 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %78 = cast %77 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const _Float16">>
        %79 = call_opaque "*(const _Float16 *)"(%78) : (!emitc.ptr<!emitc.opaque<"const _Float16">>) -> !emitc.opaque<"_Float16">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2"
        %80 = call_opaque "__riscv_vfwmul_vf_f32m2"(%61, %75, %1) : (!emitc.opaque<"vfloat16m1_t">, !emitc.opaque<"_Float16">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2"
        %81 = call_opaque "__riscv_vfcvt_f_x_v_f32m2"(%56, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        %82 = load %15 : <!emitc.opaque<"vfloat32m2_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2"
        %83 = call_opaque "__riscv_vfmacc_vv_f32m2"(%82, %81, %80, %1) : (!emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2"
        %84 = call_opaque "__riscv_vfwmul_vf_f32m2"(%69, %79, %1) : (!emitc.opaque<"vfloat16m1_t">, !emitc.opaque<"_Float16">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2"
        %85 = call_opaque "__riscv_vfadd_vv_f32m2"(%83, %84, %1) : (!emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        assign %85 : !emitc.opaque<"vfloat32m2_t"> to %15 : <!emitc.opaque<"vfloat32m2_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2"
        %86 = call_opaque "__riscv_vfwmul_vf_f32m2"(%65, %75, %1) : (!emitc.opaque<"vfloat16m1_t">, !emitc.opaque<"_Float16">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2"
        %87 = call_opaque "__riscv_vfcvt_f_x_v_f32m2"(%59, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        %88 = load %18 : <!emitc.opaque<"vfloat32m2_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2"
        %89 = call_opaque "__riscv_vfmacc_vv_f32m2"(%88, %87, %86, %1) : (!emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2"
        %90 = call_opaque "__riscv_vfwmul_vf_f32m2"(%73, %79, %1) : (!emitc.opaque<"vfloat16m1_t">, !emitc.opaque<"_Float16">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2"
        %91 = call_opaque "__riscv_vfadd_vv_f32m2"(%89, %90, %1) : (!emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        assign %91 : !emitc.opaque<"vfloat32m2_t"> to %18 : <!emitc.opaque<"vfloat32m2_t">>
      }
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr"
      %23 = literal "16" : !emitc.opaque<"size_t">
      %24 = mul %arg5, %23 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %25 = add %arg1, %24 : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"float">>
      %26 = load %15 : <!emitc.opaque<"vfloat32m2_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2"
      call_opaque "__riscv_vse32_v_f32m2"(%25, %26, %1) : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr"
      %27 = literal "16" : !emitc.opaque<"size_t">
      %28 = mul %arg5, %27 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %29 = literal "8" : !emitc.opaque<"size_t">
      %30 = add %28, %29 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %31 = add %arg1, %30 : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"float">>
      %32 = load %18 : <!emitc.opaque<"vfloat32m2_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2"
      call_opaque "__riscv_vse32_v_f32m2"(%31, %32, %1) : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> ()
    }
    %8 = literal "0" : !emitc.opaque<"int32_t">
    %9 = literal "1" : !emitc.opaque<"size_t">
    %10 = call_opaque "__riscv_vmv_v_x_i32m1"(%8, %9) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
    return
  }
}

