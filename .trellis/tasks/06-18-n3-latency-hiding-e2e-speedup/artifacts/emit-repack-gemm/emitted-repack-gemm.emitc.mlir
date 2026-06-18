module {
  emitc.include <"stddef.h">
  emitc.include <"stdint.h">
  emitc.include <"riscv_vector.h">
  emitc.func @tcrv_emitc_ggml_repack_gemm_q4_0_q8_0_kernel_ggml_repack_gemm_q4_0_q8_0(%arg0: !emitc.opaque<"size_t">, %arg1: !emitc.ptr<!emitc.opaque<"float">>, %arg2: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg3: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg4: !emitc.opaque<"size_t">, %arg5: !emitc.opaque<"size_t">, %arg6: !emitc.opaque<"size_t">) attributes {specifiers = ["extern", "\22C\22"]} {
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1"
    %0 = call_opaque "__riscv_vsetvl_e32m1"(%arg0) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface"
    %1 = literal "8" : !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count"
    %2 = literal "32" : !emitc.opaque<"size_t">
    %3 = div %arg0, %2 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=row_group_count"
    %4 = literal "4" : !emitc.opaque<"size_t">
    %5 = div %arg4, %4 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=col_group_count"
    %6 = literal "16" : !emitc.opaque<"size_t">
    %7 = div %arg5, %6 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    %8 = literal "4" : !emitc.opaque<"size_t">
    %9 = literal "1" : !emitc.opaque<"size_t">
    %10 = literal "0" : !emitc.opaque<"size_t">
    for %arg7 = %10 to %5 step %9  : !emitc.opaque<"size_t"> {
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_group_base"
      %14 = mul %arg7, %3 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %15 = literal "136" : !emitc.opaque<"size_t">
      %16 = mul %14, %15 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %17 = add %arg3, %16 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %18 = literal "1" : !emitc.opaque<"size_t">
      %19 = literal "0" : !emitc.opaque<"size_t">
      for %arg8 = %19 to %7 step %18  : !emitc.opaque<"size_t"> {
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_group_base"
        %20 = mul %arg8, %3 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %21 = literal "288" : !emitc.opaque<"size_t">
        %22 = mul %20, %21 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %23 = add %arg2, %22 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %24 = literal "1" : !emitc.opaque<"size_t">
        %25 = literal "2" : !emitc.opaque<"size_t">
        %26 = literal "0" : !emitc.opaque<"size_t">
        for %arg9 = %26 to %25 step %24  : !emitc.opaque<"size_t"> {
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=half_row_offset"
          %27 = literal "8" : !emitc.opaque<"size_t">
          %28 = mul %arg9, %27 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2"
          %29 = literal "0.0f" : !emitc.opaque<"float">
          %30 = call_opaque "__riscv_vfmv_v_f_f32m2"(%29, %1) : (!emitc.opaque<"float">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2"
          %31 = literal "0.0f" : !emitc.opaque<"float">
          %32 = call_opaque "__riscv_vfmv_v_f_f32m2"(%31, %1) : (!emitc.opaque<"float">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2"
          %33 = literal "0.0f" : !emitc.opaque<"float">
          %34 = call_opaque "__riscv_vfmv_v_f_f32m2"(%33, %1) : (!emitc.opaque<"float">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2"
          %35 = literal "0.0f" : !emitc.opaque<"float">
          %36 = call_opaque "__riscv_vfmv_v_f_f32m2"(%35, %1) : (!emitc.opaque<"float">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
          %37 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m2_t">>
          assign %30 : !emitc.opaque<"vfloat32m2_t"> to %37 : <!emitc.opaque<"vfloat32m2_t">>
          %38 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m2_t">>
          assign %32 : !emitc.opaque<"vfloat32m2_t"> to %38 : <!emitc.opaque<"vfloat32m2_t">>
          %39 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m2_t">>
          assign %34 : !emitc.opaque<"vfloat32m2_t"> to %39 : <!emitc.opaque<"vfloat32m2_t">>
          %40 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m2_t">>
          assign %36 : !emitc.opaque<"vfloat32m2_t"> to %40 : <!emitc.opaque<"vfloat32m2_t">>
          %41 = literal "1" : !emitc.opaque<"size_t">
          %42 = literal "0" : !emitc.opaque<"size_t">
          for %arg10 = %42 to %3 step %41  : !emitc.opaque<"size_t"> {
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base"
            %87 = literal "288" : !emitc.opaque<"size_t">
            %88 = mul %arg10, %87 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
            %89 = add %23, %88 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base"
            %90 = literal "136" : !emitc.opaque<"size_t">
            %91 = mul %arg10, %90 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
            %92 = add %17, %91 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
            %93 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
            %94 = literal "0" : !emitc.opaque<"int32_t">
            %95 = call_opaque "__riscv_vmv_v_x_i16m1"(%94, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
            assign %95 : !emitc.opaque<"vint16m1_t"> to %93 : <!emitc.opaque<"vint16m1_t">>
            %96 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
            %97 = literal "0" : !emitc.opaque<"int32_t">
            %98 = call_opaque "__riscv_vmv_v_x_i16m1"(%97, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
            assign %98 : !emitc.opaque<"vint16m1_t"> to %96 : <!emitc.opaque<"vint16m1_t">>
            %99 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
            %100 = literal "0" : !emitc.opaque<"int32_t">
            %101 = call_opaque "__riscv_vmv_v_x_i16m1"(%100, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
            assign %101 : !emitc.opaque<"vint16m1_t"> to %99 : <!emitc.opaque<"vint16m1_t">>
            %102 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
            %103 = literal "0" : !emitc.opaque<"int32_t">
            %104 = call_opaque "__riscv_vmv_v_x_i16m1"(%103, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
            assign %104 : !emitc.opaque<"vint16m1_t"> to %102 : <!emitc.opaque<"vint16m1_t">>
            %105 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
            %106 = literal "0" : !emitc.opaque<"int32_t">
            %107 = call_opaque "__riscv_vmv_v_x_i16m1"(%106, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
            assign %107 : !emitc.opaque<"vint16m1_t"> to %105 : <!emitc.opaque<"vint16m1_t">>
            %108 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
            %109 = literal "0" : !emitc.opaque<"int32_t">
            %110 = call_opaque "__riscv_vmv_v_x_i16m1"(%109, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
            assign %110 : !emitc.opaque<"vint16m1_t"> to %108 : <!emitc.opaque<"vint16m1_t">>
            %111 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
            %112 = literal "0" : !emitc.opaque<"int32_t">
            %113 = call_opaque "__riscv_vmv_v_x_i16m1"(%112, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
            assign %113 : !emitc.opaque<"vint16m1_t"> to %111 : <!emitc.opaque<"vint16m1_t">>
            %114 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
            %115 = literal "0" : !emitc.opaque<"int32_t">
            %116 = call_opaque "__riscv_vmv_v_x_i16m1"(%115, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
            assign %116 : !emitc.opaque<"vint16m1_t"> to %114 : <!emitc.opaque<"vint16m1_t">>
            %117 = literal "1" : !emitc.opaque<"size_t">
            %118 = literal "16" : !emitc.opaque<"size_t">
            %119 = literal "0" : !emitc.opaque<"size_t">
            for %arg11 = %119 to %118 step %117  : !emitc.opaque<"size_t"> {
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr"
              %169 = literal "16" : !emitc.opaque<"size_t">
              %170 = mul %arg11, %169 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              %171 = literal "32" : !emitc.opaque<"size_t">
              %172 = add %171, %170 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              %173 = add %172, %28 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              %174 = add %89, %173 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
              %175 = cast %174 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2"
              %176 = call_opaque "__riscv_vle8_v_i8mf2"(%175, %1) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8mf2"
              %177 = call_opaque "__riscv_vsll_vx_i8mf2"(%176, %8, %1) : (!emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8mf2"
              %178 = call_opaque "__riscv_vsra_vx_i8mf2"(%177, %8, %1) : (!emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8mf2"
              %179 = call_opaque "__riscv_vsra_vx_i8mf2"(%176, %8, %1) : (!emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
              %180 = literal "4" : !emitc.opaque<"size_t">
              %181 = mul %arg11, %180 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo"
              %182 = literal "0" : !emitc.opaque<"size_t">
              %183 = add %181, %182 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              %184 = literal "8" : !emitc.opaque<"size_t">
              %185 = add %184, %183 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              %186 = add %92, %185 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
              %187 = cast %186 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar"
              %188 = call_opaque "*(const int8_t *)"(%187) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
              %189 = load %93 : <!emitc.opaque<"vint16m1_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
              %190 = call_opaque "__riscv_vwmacc_vx_i16m1"(%189, %188, %178, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
              assign %190 : !emitc.opaque<"vint16m1_t"> to %93 : <!emitc.opaque<"vint16m1_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi"
              %191 = literal "0" : !emitc.opaque<"size_t">
              %192 = add %181, %191 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              %193 = literal "64" : !emitc.opaque<"size_t">
              %194 = literal "8" : !emitc.opaque<"size_t">
              %195 = add %194, %193 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              %196 = add %195, %192 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              %197 = add %92, %196 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
              %198 = cast %197 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar"
              %199 = call_opaque "*(const int8_t *)"(%198) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
              %200 = load %105 : <!emitc.opaque<"vint16m1_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
              %201 = call_opaque "__riscv_vwmacc_vx_i16m1"(%200, %199, %179, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
              assign %201 : !emitc.opaque<"vint16m1_t"> to %105 : <!emitc.opaque<"vint16m1_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo"
              %202 = literal "1" : !emitc.opaque<"size_t">
              %203 = add %181, %202 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              %204 = literal "8" : !emitc.opaque<"size_t">
              %205 = add %204, %203 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              %206 = add %92, %205 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
              %207 = cast %206 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar"
              %208 = call_opaque "*(const int8_t *)"(%207) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
              %209 = load %96 : <!emitc.opaque<"vint16m1_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
              %210 = call_opaque "__riscv_vwmacc_vx_i16m1"(%209, %208, %178, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
              assign %210 : !emitc.opaque<"vint16m1_t"> to %96 : <!emitc.opaque<"vint16m1_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi"
              %211 = literal "1" : !emitc.opaque<"size_t">
              %212 = add %181, %211 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              %213 = literal "64" : !emitc.opaque<"size_t">
              %214 = literal "8" : !emitc.opaque<"size_t">
              %215 = add %214, %213 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              %216 = add %215, %212 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              %217 = add %92, %216 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
              %218 = cast %217 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar"
              %219 = call_opaque "*(const int8_t *)"(%218) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
              %220 = load %108 : <!emitc.opaque<"vint16m1_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
              %221 = call_opaque "__riscv_vwmacc_vx_i16m1"(%220, %219, %179, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
              assign %221 : !emitc.opaque<"vint16m1_t"> to %108 : <!emitc.opaque<"vint16m1_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo"
              %222 = literal "2" : !emitc.opaque<"size_t">
              %223 = add %181, %222 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              %224 = literal "8" : !emitc.opaque<"size_t">
              %225 = add %224, %223 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              %226 = add %92, %225 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
              %227 = cast %226 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar"
              %228 = call_opaque "*(const int8_t *)"(%227) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
              %229 = load %99 : <!emitc.opaque<"vint16m1_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
              %230 = call_opaque "__riscv_vwmacc_vx_i16m1"(%229, %228, %178, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
              assign %230 : !emitc.opaque<"vint16m1_t"> to %99 : <!emitc.opaque<"vint16m1_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi"
              %231 = literal "2" : !emitc.opaque<"size_t">
              %232 = add %181, %231 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              %233 = literal "64" : !emitc.opaque<"size_t">
              %234 = literal "8" : !emitc.opaque<"size_t">
              %235 = add %234, %233 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              %236 = add %235, %232 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              %237 = add %92, %236 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
              %238 = cast %237 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar"
              %239 = call_opaque "*(const int8_t *)"(%238) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
              %240 = load %111 : <!emitc.opaque<"vint16m1_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
              %241 = call_opaque "__riscv_vwmacc_vx_i16m1"(%240, %239, %179, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
              assign %241 : !emitc.opaque<"vint16m1_t"> to %111 : <!emitc.opaque<"vint16m1_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo"
              %242 = literal "3" : !emitc.opaque<"size_t">
              %243 = add %181, %242 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              %244 = literal "8" : !emitc.opaque<"size_t">
              %245 = add %244, %243 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              %246 = add %92, %245 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
              %247 = cast %246 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar"
              %248 = call_opaque "*(const int8_t *)"(%247) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
              %249 = load %102 : <!emitc.opaque<"vint16m1_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
              %250 = call_opaque "__riscv_vwmacc_vx_i16m1"(%249, %248, %178, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
              assign %250 : !emitc.opaque<"vint16m1_t"> to %102 : <!emitc.opaque<"vint16m1_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi"
              %251 = literal "3" : !emitc.opaque<"size_t">
              %252 = add %181, %251 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              %253 = literal "64" : !emitc.opaque<"size_t">
              %254 = literal "8" : !emitc.opaque<"size_t">
              %255 = add %254, %253 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              %256 = add %255, %252 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
              %257 = add %92, %256 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
              %258 = cast %257 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar"
              %259 = call_opaque "*(const int8_t *)"(%258) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
              %260 = load %114 : <!emitc.opaque<"vint16m1_t">>
              verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
              %261 = call_opaque "__riscv_vwmacc_vx_i16m1"(%260, %259, %179, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
              assign %261 : !emitc.opaque<"vint16m1_t"> to %114 : <!emitc.opaque<"vint16m1_t">>
            }
            %120 = load %93 : <!emitc.opaque<"vint16m1_t">>
            %121 = load %105 : <!emitc.opaque<"vint16m1_t">>
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2"
            %122 = call_opaque "__riscv_vwadd_vv_i32m2"(%120, %121, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
            %123 = load %96 : <!emitc.opaque<"vint16m1_t">>
            %124 = load %108 : <!emitc.opaque<"vint16m1_t">>
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2"
            %125 = call_opaque "__riscv_vwadd_vv_i32m2"(%123, %124, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
            %126 = load %99 : <!emitc.opaque<"vint16m1_t">>
            %127 = load %111 : <!emitc.opaque<"vint16m1_t">>
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2"
            %128 = call_opaque "__riscv_vwadd_vv_i32m2"(%126, %127, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
            %129 = load %102 : <!emitc.opaque<"vint16m1_t">>
            %130 = load %114 : <!emitc.opaque<"vint16m1_t">>
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2"
            %131 = call_opaque "__riscv_vwadd_vv_i32m2"(%129, %130, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr"
            %132 = literal "2" : !emitc.opaque<"size_t">
            %133 = mul %28, %132 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
            %134 = add %89, %133 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
            %135 = cast %134 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const _Float16">>
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1"
            %136 = call_opaque "__riscv_vle16_v_f16m1"(%135, %1) : (!emitc.ptr<!emitc.opaque<"const _Float16">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat16m1_t">
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar"
            %137 = literal "0" : !emitc.opaque<"size_t">
            %138 = add %92, %137 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
            %139 = cast %138 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const _Float16">>
            %140 = call_opaque "*(const _Float16 *)"(%139) : (!emitc.ptr<!emitc.opaque<"const _Float16">>) -> !emitc.opaque<"_Float16">
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2"
            %141 = call_opaque "__riscv_vfwmul_vf_f32m2"(%136, %140, %1) : (!emitc.opaque<"vfloat16m1_t">, !emitc.opaque<"_Float16">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2"
            %142 = call_opaque "__riscv_vfcvt_f_x_v_f32m2"(%122, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
            %143 = load %37 : <!emitc.opaque<"vfloat32m2_t">>
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2"
            %144 = call_opaque "__riscv_vfmacc_vv_f32m2"(%143, %142, %141, %1) : (!emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
            assign %144 : !emitc.opaque<"vfloat32m2_t"> to %37 : <!emitc.opaque<"vfloat32m2_t">>
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar"
            %145 = literal "2" : !emitc.opaque<"size_t">
            %146 = add %92, %145 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
            %147 = cast %146 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const _Float16">>
            %148 = call_opaque "*(const _Float16 *)"(%147) : (!emitc.ptr<!emitc.opaque<"const _Float16">>) -> !emitc.opaque<"_Float16">
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2"
            %149 = call_opaque "__riscv_vfwmul_vf_f32m2"(%136, %148, %1) : (!emitc.opaque<"vfloat16m1_t">, !emitc.opaque<"_Float16">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2"
            %150 = call_opaque "__riscv_vfcvt_f_x_v_f32m2"(%125, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
            %151 = load %38 : <!emitc.opaque<"vfloat32m2_t">>
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2"
            %152 = call_opaque "__riscv_vfmacc_vv_f32m2"(%151, %150, %149, %1) : (!emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
            assign %152 : !emitc.opaque<"vfloat32m2_t"> to %38 : <!emitc.opaque<"vfloat32m2_t">>
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar"
            %153 = literal "4" : !emitc.opaque<"size_t">
            %154 = add %92, %153 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
            %155 = cast %154 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const _Float16">>
            %156 = call_opaque "*(const _Float16 *)"(%155) : (!emitc.ptr<!emitc.opaque<"const _Float16">>) -> !emitc.opaque<"_Float16">
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2"
            %157 = call_opaque "__riscv_vfwmul_vf_f32m2"(%136, %156, %1) : (!emitc.opaque<"vfloat16m1_t">, !emitc.opaque<"_Float16">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2"
            %158 = call_opaque "__riscv_vfcvt_f_x_v_f32m2"(%128, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
            %159 = load %39 : <!emitc.opaque<"vfloat32m2_t">>
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2"
            %160 = call_opaque "__riscv_vfmacc_vv_f32m2"(%159, %158, %157, %1) : (!emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
            assign %160 : !emitc.opaque<"vfloat32m2_t"> to %39 : <!emitc.opaque<"vfloat32m2_t">>
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar"
            %161 = literal "6" : !emitc.opaque<"size_t">
            %162 = add %92, %161 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
            %163 = cast %162 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const _Float16">>
            %164 = call_opaque "*(const _Float16 *)"(%163) : (!emitc.ptr<!emitc.opaque<"const _Float16">>) -> !emitc.opaque<"_Float16">
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2"
            %165 = call_opaque "__riscv_vfwmul_vf_f32m2"(%136, %164, %1) : (!emitc.opaque<"vfloat16m1_t">, !emitc.opaque<"_Float16">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2"
            %166 = call_opaque "__riscv_vfcvt_f_x_v_f32m2"(%131, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
            %167 = load %40 : <!emitc.opaque<"vfloat32m2_t">>
            verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2"
            %168 = call_opaque "__riscv_vfmacc_vv_f32m2"(%167, %166, %165, %1) : (!emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
            assign %168 : !emitc.opaque<"vfloat32m2_t"> to %40 : <!emitc.opaque<"vfloat32m2_t">>
          }
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr"
          %43 = literal "4" : !emitc.opaque<"size_t">
          %44 = mul %arg7, %43 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %45 = literal "0" : !emitc.opaque<"size_t">
          %46 = add %44, %45 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %47 = mul %46, %arg6 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %48 = literal "16" : !emitc.opaque<"size_t">
          %49 = mul %arg8, %48 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %50 = add %47, %49 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %51 = add %50, %28 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %52 = add %arg1, %51 : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"float">>
          %53 = load %37 : <!emitc.opaque<"vfloat32m2_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2"
          call_opaque "__riscv_vse32_v_f32m2"(%52, %53, %1) : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> ()
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr"
          %54 = literal "4" : !emitc.opaque<"size_t">
          %55 = mul %arg7, %54 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %56 = literal "1" : !emitc.opaque<"size_t">
          %57 = add %55, %56 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %58 = mul %57, %arg6 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %59 = literal "16" : !emitc.opaque<"size_t">
          %60 = mul %arg8, %59 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %61 = add %58, %60 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %62 = add %61, %28 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %63 = add %arg1, %62 : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"float">>
          %64 = load %38 : <!emitc.opaque<"vfloat32m2_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2"
          call_opaque "__riscv_vse32_v_f32m2"(%63, %64, %1) : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> ()
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr"
          %65 = literal "4" : !emitc.opaque<"size_t">
          %66 = mul %arg7, %65 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %67 = literal "2" : !emitc.opaque<"size_t">
          %68 = add %66, %67 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %69 = mul %68, %arg6 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %70 = literal "16" : !emitc.opaque<"size_t">
          %71 = mul %arg8, %70 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %72 = add %69, %71 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %73 = add %72, %28 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %74 = add %arg1, %73 : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"float">>
          %75 = load %39 : <!emitc.opaque<"vfloat32m2_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2"
          call_opaque "__riscv_vse32_v_f32m2"(%74, %75, %1) : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> ()
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr"
          %76 = literal "4" : !emitc.opaque<"size_t">
          %77 = mul %arg7, %76 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %78 = literal "3" : !emitc.opaque<"size_t">
          %79 = add %77, %78 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %80 = mul %79, %arg6 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %81 = literal "16" : !emitc.opaque<"size_t">
          %82 = mul %arg8, %81 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %83 = add %80, %82 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %84 = add %83, %28 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %85 = add %arg1, %84 : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"float">>
          %86 = load %40 : <!emitc.opaque<"vfloat32m2_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2"
          call_opaque "__riscv_vse32_v_f32m2"(%85, %86, %1) : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> ()
        }
      }
    }
    %11 = literal "0" : !emitc.opaque<"int32_t">
    %12 = literal "1" : !emitc.opaque<"size_t">
    %13 = call_opaque "__riscv_vmv_v_x_i32m1"(%11, %12) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
    return
  }
}

