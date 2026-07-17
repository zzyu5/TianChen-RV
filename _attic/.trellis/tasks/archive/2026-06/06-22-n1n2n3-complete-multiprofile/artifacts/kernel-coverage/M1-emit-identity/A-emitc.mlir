module {
  emitc.include <"stddef.h">
  emitc.include <"stdint.h">
  emitc.include <"riscv_vector.h">
  emitc.func @tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(%arg0: !emitc.opaque<"size_t">, %arg1: !emitc.ptr<!emitc.opaque<"float">>, %arg2: !emitc.opaque<"size_t">, %arg3: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg4: !emitc.opaque<"size_t">, %arg5: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg6: !emitc.opaque<"size_t">, %arg7: !emitc.opaque<"int32_t">) attributes {specifiers = ["extern", "\22C\22"]} {
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1"
    %0 = call_opaque "__riscv_vsetvl_e32m1"(%arg0) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface"
    %1 = literal "8" : !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count"
    %2 = literal "32" : !emitc.opaque<"size_t">
    %3 = div %arg0, %2 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=col_group_count"
    %4 = literal "16" : !emitc.opaque<"size_t">
    %5 = div %arg2, %4 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    %6 = literal "4" : !emitc.opaque<"size_t">
    %7 = literal "1" : !emitc.opaque<"size_t">
    %8 = literal "0" : !emitc.opaque<"size_t">
    for %arg8 = %8 to %5 step %7  : !emitc.opaque<"size_t"> {
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_group_base"
      %12 = mul %arg8, %3 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %13 = literal "288" : !emitc.opaque<"size_t">
      %14 = mul %12, %13 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %15 = add %arg3, %14 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %16 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m2_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2"
      %17 = literal "0.0f" : !emitc.opaque<"float">
      %18 = call_opaque "__riscv_vfmv_v_f_f32m2"(%17, %1) : (!emitc.opaque<"float">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
      assign %18 : !emitc.opaque<"vfloat32m2_t"> to %16 : <!emitc.opaque<"vfloat32m2_t">>
      %19 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m2_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2"
      %20 = literal "0.0f" : !emitc.opaque<"float">
      %21 = call_opaque "__riscv_vfmv_v_f_f32m2"(%20, %1) : (!emitc.opaque<"float">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
      assign %21 : !emitc.opaque<"vfloat32m2_t"> to %19 : <!emitc.opaque<"vfloat32m2_t">>
      %22 = literal "1" : !emitc.opaque<"size_t">
      %23 = literal "0" : !emitc.opaque<"size_t">
      for %arg9 = %23 to %3 step %22  : !emitc.opaque<"size_t"> {
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base"
        %34 = literal "288" : !emitc.opaque<"size_t">
        %35 = mul %arg9, %34 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %36 = add %15, %35 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base"
        %37 = literal "34" : !emitc.opaque<"size_t">
        %38 = mul %arg9, %37 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %39 = add %arg5, %38 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %40 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %41 = literal "0" : !emitc.opaque<"int32_t">
        %42 = call_opaque "__riscv_vmv_v_x_i16m1"(%41, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %42 : !emitc.opaque<"vint16m1_t"> to %40 : <!emitc.opaque<"vint16m1_t">>
        %43 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %44 = literal "0" : !emitc.opaque<"int32_t">
        %45 = call_opaque "__riscv_vmv_v_x_i16m1"(%44, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %45 : !emitc.opaque<"vint16m1_t"> to %43 : <!emitc.opaque<"vint16m1_t">>
        %46 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %47 = literal "0" : !emitc.opaque<"int32_t">
        %48 = call_opaque "__riscv_vmv_v_x_i16m1"(%47, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %48 : !emitc.opaque<"vint16m1_t"> to %46 : <!emitc.opaque<"vint16m1_t">>
        %49 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint16m1_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1"
        %50 = literal "0" : !emitc.opaque<"int32_t">
        %51 = call_opaque "__riscv_vmv_v_x_i16m1"(%50, %1) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        assign %51 : !emitc.opaque<"vint16m1_t"> to %49 : <!emitc.opaque<"vint16m1_t">>
        %52 = literal "1" : !emitc.opaque<"size_t">
        %53 = literal "16" : !emitc.opaque<"size_t">
        %54 = literal "0" : !emitc.opaque<"size_t">
        for %arg10 = %54 to %53 step %52  : !emitc.opaque<"size_t"> {
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr"
          %77 = literal "16" : !emitc.opaque<"size_t">
          %78 = mul %arg10, %77 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %79 = literal "32" : !emitc.opaque<"size_t">
          %80 = add %79, %78 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %81 = literal "8" : !emitc.opaque<"size_t">
          %82 = add %80, %81 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %83 = add %36, %80 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %84 = cast %83 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2"
          %85 = call_opaque "__riscv_vle8_v_i8mf2"(%84, %1) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
          %86 = add %36, %82 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %87 = cast %86 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2"
          %88 = call_opaque "__riscv_vle8_v_i8mf2"(%87, %1) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8mf2"
          %89 = call_opaque "__riscv_vsll_vx_i8mf2"(%85, %6, %1) : (!emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8mf2"
          %90 = call_opaque "__riscv_vsra_vx_i8mf2"(%89, %6, %1) : (!emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8mf2"
          %91 = call_opaque "__riscv_vsra_vx_i8mf2"(%85, %6, %1) : (!emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8mf2"
          %92 = call_opaque "__riscv_vsll_vx_i8mf2"(%88, %6, %1) : (!emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8mf2"
          %93 = call_opaque "__riscv_vsra_vx_i8mf2"(%92, %6, %1) : (!emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8mf2"
          %94 = call_opaque "__riscv_vsra_vx_i8mf2"(%88, %6, %1) : (!emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo"
          %95 = literal "2" : !emitc.opaque<"size_t">
          %96 = add %95, %arg10 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %97 = add %39, %96 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %98 = cast %97 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar"
          %99 = call_opaque "*(const int8_t *)"(%98) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi"
          %100 = literal "16" : !emitc.opaque<"size_t">
          %101 = literal "2" : !emitc.opaque<"size_t">
          %102 = add %101, %100 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %103 = add %102, %arg10 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %104 = add %39, %103 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
          %105 = cast %104 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar"
          %106 = call_opaque "*(const int8_t *)"(%105) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"int32_t">
          %107 = load %40 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %108 = call_opaque "__riscv_vwmacc_vx_i16m1"(%107, %99, %90, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %108 : !emitc.opaque<"vint16m1_t"> to %40 : <!emitc.opaque<"vint16m1_t">>
          %109 = load %43 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %110 = call_opaque "__riscv_vwmacc_vx_i16m1"(%109, %106, %91, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %110 : !emitc.opaque<"vint16m1_t"> to %43 : <!emitc.opaque<"vint16m1_t">>
          %111 = load %46 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %112 = call_opaque "__riscv_vwmacc_vx_i16m1"(%111, %99, %93, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %112 : !emitc.opaque<"vint16m1_t"> to %46 : <!emitc.opaque<"vint16m1_t">>
          %113 = load %49 : <!emitc.opaque<"vint16m1_t">>
          verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1"
          %114 = call_opaque "__riscv_vwmacc_vx_i16m1"(%113, %106, %94, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"int32_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
          assign %114 : !emitc.opaque<"vint16m1_t"> to %49 : <!emitc.opaque<"vint16m1_t">>
        }
        %55 = load %40 : <!emitc.opaque<"vint16m1_t">>
        %56 = load %43 : <!emitc.opaque<"vint16m1_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2"
        %57 = call_opaque "__riscv_vwadd_vv_i32m2"(%55, %56, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        %58 = load %46 : <!emitc.opaque<"vint16m1_t">>
        %59 = load %49 : <!emitc.opaque<"vint16m1_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2"
        %60 = call_opaque "__riscv_vwadd_vv_i32m2"(%58, %59, %1) : (!emitc.opaque<"vint16m1_t">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr"
        %61 = cast %36 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const _Float16">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1"
        %62 = call_opaque "__riscv_vle16_v_f16m1"(%61, %1) : (!emitc.ptr<!emitc.opaque<"const _Float16">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat16m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr"
        %63 = literal "16" : !emitc.opaque<"size_t">
        %64 = add %36, %63 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %65 = cast %64 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const _Float16">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1"
        %66 = call_opaque "__riscv_vle16_v_f16m1"(%65, %1) : (!emitc.ptr<!emitc.opaque<"const _Float16">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat16m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar"
        %67 = cast %39 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const _Float16">>
        %68 = call_opaque "*(const _Float16 *)"(%67) : (!emitc.ptr<!emitc.opaque<"const _Float16">>) -> !emitc.opaque<"_Float16">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2"
        %69 = call_opaque "__riscv_vfwmul_vf_f32m2"(%62, %68, %1) : (!emitc.opaque<"vfloat16m1_t">, !emitc.opaque<"_Float16">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2"
        %70 = call_opaque "__riscv_vfcvt_f_x_v_f32m2"(%57, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        %71 = load %16 : <!emitc.opaque<"vfloat32m2_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2"
        %72 = call_opaque "__riscv_vfmacc_vv_f32m2"(%71, %70, %69, %1) : (!emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        assign %72 : !emitc.opaque<"vfloat32m2_t"> to %16 : <!emitc.opaque<"vfloat32m2_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2"
        %73 = call_opaque "__riscv_vfwmul_vf_f32m2"(%66, %68, %1) : (!emitc.opaque<"vfloat16m1_t">, !emitc.opaque<"_Float16">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2"
        %74 = call_opaque "__riscv_vfcvt_f_x_v_f32m2"(%60, %1) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        %75 = load %19 : <!emitc.opaque<"vfloat32m2_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2"
        %76 = call_opaque "__riscv_vfmacc_vv_f32m2"(%75, %74, %73, %1) : (!emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
        assign %76 : !emitc.opaque<"vfloat32m2_t"> to %19 : <!emitc.opaque<"vfloat32m2_t">>
      }
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr"
      %24 = literal "16" : !emitc.opaque<"size_t">
      %25 = mul %arg8, %24 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %26 = add %arg1, %25 : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"float">>
      %27 = load %16 : <!emitc.opaque<"vfloat32m2_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2"
      call_opaque "__riscv_vse32_v_f32m2"(%26, %27, %1) : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr"
      %28 = literal "16" : !emitc.opaque<"size_t">
      %29 = mul %arg8, %28 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %30 = literal "8" : !emitc.opaque<"size_t">
      %31 = add %29, %30 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %32 = add %arg1, %31 : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"float">>
      %33 = load %19 : <!emitc.opaque<"vfloat32m2_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2"
      call_opaque "__riscv_vse32_v_f32m2"(%32, %33, %1) : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> ()
    }
    %9 = literal "0" : !emitc.opaque<"int32_t">
    %10 = literal "1" : !emitc.opaque<"size_t">
    %11 = call_opaque "__riscv_vmv_v_x_i32m1"(%9, %10) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
    return
  }
}

