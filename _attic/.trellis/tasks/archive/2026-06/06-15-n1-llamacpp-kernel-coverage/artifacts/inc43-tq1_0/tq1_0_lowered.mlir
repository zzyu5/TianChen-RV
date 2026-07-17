module {
  emitc.include <"stddef.h">
  emitc.include <"stdint.h">
  emitc.include <"riscv_vector.h">
  emitc.func @tcrv_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_ggml_vec_dot_tq1_0_q8_K(%arg0: !emitc.opaque<"size_t">, %arg1: !emitc.ptr<!emitc.opaque<"float">>, %arg2: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg3: !emitc.ptr<!emitc.opaque<"const uint8_t">>) attributes {specifiers = ["extern", "\22C\22"]} {
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1"
    %0 = call_opaque "__riscv_vsetvl_e32m1"(%arg0) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count"
    %1 = literal "256" : !emitc.opaque<"size_t">
    %2 = div %arg0, %1 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.local_variable=aux8 source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
    %3 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<256x!emitc.opaque<"int8_t">>
    %4 = literal "0" : index
    %5 = subscript %3[%4] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
    %6 = apply "&"(%5) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
    verbatim "// tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
    %7 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"float">>
    %8 = literal "0.0f" : !emitc.opaque<"float">
    assign %8 : !emitc.opaque<"float"> to %7 : <!emitc.opaque<"float">>
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_loop"
    %9 = literal "1" : !emitc.opaque<"size_t">
    %10 = literal "0" : !emitc.opaque<"size_t">
    for %arg4 = %10 to %2 step %9  : !emitc.opaque<"size_t"> {
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x"
      %14 = literal "54" : !emitc.opaque<"size_t">
      %15 = mul %arg4, %14 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %16 = add %arg2, %15 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_y"
      %17 = literal "292" : !emitc.opaque<"size_t">
      %18 = mul %arg4, %17 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %19 = add %arg3, %18 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=unpack_base3_trit"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2"
      %20 = literal "32" : !emitc.opaque<"size_t">
      %21 = call_opaque "__riscv_vsetvl_e8m2"(%20) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %22 = cast %16 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2"
      %23 = call_opaque "__riscv_vle8_v_u8m2"(%22, %21) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m2"
      %24 = literal "1" : !emitc.opaque<"int">
      %25 = call_opaque "__riscv_vmul_vx_u8m2"(%23, %24, %21) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m4"
      %26 = literal "3" : !emitc.opaque<"int">
      %27 = call_opaque "__riscv_vwmulu_vx_u16m4"(%25, %26, %21) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m4_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m4"
      %28 = literal "8" : !emitc.opaque<"int">
      %29 = call_opaque "__riscv_vsrl_vx_u16m4"(%27, %28, %21) : (!emitc.opaque<"vuint16m4_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m4_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m2"
      %30 = call_opaque "__riscv_vncvt_x_x_w_u8m2"(%29, %21) : (!emitc.opaque<"vuint16m4_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2"
      %31 = call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"(%30) : (!emitc.opaque<"vuint8m2_t">) -> !emitc.opaque<"vint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2"
      %32 = literal "-1" : !emitc.opaque<"int">
      %33 = call_opaque "__riscv_vadd_vx_i8m2"(%31, %32, %21) : (!emitc.opaque<"vint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m2_t">
      %34 = literal "0" : index
      %35 = subscript %3[%34] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %36 = apply "&"(%35) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2"
      call_opaque "__riscv_vse8_v_i8m2"(%36, %33, %21) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2"
      %37 = literal "32" : !emitc.opaque<"size_t">
      %38 = call_opaque "__riscv_vsetvl_e8m2"(%37) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %39 = cast %16 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2"
      %40 = call_opaque "__riscv_vle8_v_u8m2"(%39, %38) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m2"
      %41 = literal "3" : !emitc.opaque<"int">
      %42 = call_opaque "__riscv_vmul_vx_u8m2"(%40, %41, %38) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m4"
      %43 = literal "3" : !emitc.opaque<"int">
      %44 = call_opaque "__riscv_vwmulu_vx_u16m4"(%42, %43, %38) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m4_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m4"
      %45 = literal "8" : !emitc.opaque<"int">
      %46 = call_opaque "__riscv_vsrl_vx_u16m4"(%44, %45, %38) : (!emitc.opaque<"vuint16m4_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m4_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m2"
      %47 = call_opaque "__riscv_vncvt_x_x_w_u8m2"(%46, %38) : (!emitc.opaque<"vuint16m4_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2"
      %48 = call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"(%47) : (!emitc.opaque<"vuint8m2_t">) -> !emitc.opaque<"vint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2"
      %49 = literal "-1" : !emitc.opaque<"int">
      %50 = call_opaque "__riscv_vadd_vx_i8m2"(%48, %49, %38) : (!emitc.opaque<"vint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m2_t">
      %51 = literal "32" : index
      %52 = subscript %3[%51] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %53 = apply "&"(%52) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2"
      call_opaque "__riscv_vse8_v_i8m2"(%53, %50, %38) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2"
      %54 = literal "32" : !emitc.opaque<"size_t">
      %55 = call_opaque "__riscv_vsetvl_e8m2"(%54) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %56 = cast %16 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2"
      %57 = call_opaque "__riscv_vle8_v_u8m2"(%56, %55) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m2"
      %58 = literal "9" : !emitc.opaque<"int">
      %59 = call_opaque "__riscv_vmul_vx_u8m2"(%57, %58, %55) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m4"
      %60 = literal "3" : !emitc.opaque<"int">
      %61 = call_opaque "__riscv_vwmulu_vx_u16m4"(%59, %60, %55) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m4_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m4"
      %62 = literal "8" : !emitc.opaque<"int">
      %63 = call_opaque "__riscv_vsrl_vx_u16m4"(%61, %62, %55) : (!emitc.opaque<"vuint16m4_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m4_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m2"
      %64 = call_opaque "__riscv_vncvt_x_x_w_u8m2"(%63, %55) : (!emitc.opaque<"vuint16m4_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2"
      %65 = call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"(%64) : (!emitc.opaque<"vuint8m2_t">) -> !emitc.opaque<"vint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2"
      %66 = literal "-1" : !emitc.opaque<"int">
      %67 = call_opaque "__riscv_vadd_vx_i8m2"(%65, %66, %55) : (!emitc.opaque<"vint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m2_t">
      %68 = literal "64" : index
      %69 = subscript %3[%68] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %70 = apply "&"(%69) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2"
      call_opaque "__riscv_vse8_v_i8m2"(%70, %67, %55) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2"
      %71 = literal "32" : !emitc.opaque<"size_t">
      %72 = call_opaque "__riscv_vsetvl_e8m2"(%71) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %73 = cast %16 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2"
      %74 = call_opaque "__riscv_vle8_v_u8m2"(%73, %72) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m2"
      %75 = literal "27" : !emitc.opaque<"int">
      %76 = call_opaque "__riscv_vmul_vx_u8m2"(%74, %75, %72) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m4"
      %77 = literal "3" : !emitc.opaque<"int">
      %78 = call_opaque "__riscv_vwmulu_vx_u16m4"(%76, %77, %72) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m4_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m4"
      %79 = literal "8" : !emitc.opaque<"int">
      %80 = call_opaque "__riscv_vsrl_vx_u16m4"(%78, %79, %72) : (!emitc.opaque<"vuint16m4_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m4_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m2"
      %81 = call_opaque "__riscv_vncvt_x_x_w_u8m2"(%80, %72) : (!emitc.opaque<"vuint16m4_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2"
      %82 = call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"(%81) : (!emitc.opaque<"vuint8m2_t">) -> !emitc.opaque<"vint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2"
      %83 = literal "-1" : !emitc.opaque<"int">
      %84 = call_opaque "__riscv_vadd_vx_i8m2"(%82, %83, %72) : (!emitc.opaque<"vint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m2_t">
      %85 = literal "96" : index
      %86 = subscript %3[%85] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %87 = apply "&"(%86) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2"
      call_opaque "__riscv_vse8_v_i8m2"(%87, %84, %72) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2"
      %88 = literal "32" : !emitc.opaque<"size_t">
      %89 = call_opaque "__riscv_vsetvl_e8m2"(%88) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %90 = cast %16 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2"
      %91 = call_opaque "__riscv_vle8_v_u8m2"(%90, %89) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m2"
      %92 = literal "81" : !emitc.opaque<"int">
      %93 = call_opaque "__riscv_vmul_vx_u8m2"(%91, %92, %89) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m4"
      %94 = literal "3" : !emitc.opaque<"int">
      %95 = call_opaque "__riscv_vwmulu_vx_u16m4"(%93, %94, %89) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m4_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m4"
      %96 = literal "8" : !emitc.opaque<"int">
      %97 = call_opaque "__riscv_vsrl_vx_u16m4"(%95, %96, %89) : (!emitc.opaque<"vuint16m4_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m4_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m2"
      %98 = call_opaque "__riscv_vncvt_x_x_w_u8m2"(%97, %89) : (!emitc.opaque<"vuint16m4_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2"
      %99 = call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"(%98) : (!emitc.opaque<"vuint8m2_t">) -> !emitc.opaque<"vint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2"
      %100 = literal "-1" : !emitc.opaque<"int">
      %101 = call_opaque "__riscv_vadd_vx_i8m2"(%99, %100, %89) : (!emitc.opaque<"vint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m2_t">
      %102 = literal "128" : index
      %103 = subscript %3[%102] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %104 = apply "&"(%103) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2"
      call_opaque "__riscv_vse8_v_i8m2"(%104, %101, %89) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %105 = literal "16" : !emitc.opaque<"size_t">
      %106 = call_opaque "__riscv_vsetvl_e8m1"(%105) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %107 = literal "32" : !emitc.opaque<"size_t">
      %108 = add %16, %107 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %109 = cast %108 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
      %110 = call_opaque "__riscv_vle8_v_u8m1"(%109, %106) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1"
      %111 = literal "1" : !emitc.opaque<"int">
      %112 = call_opaque "__riscv_vmul_vx_u8m1"(%110, %111, %106) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m2"
      %113 = literal "3" : !emitc.opaque<"int">
      %114 = call_opaque "__riscv_vwmulu_vx_u16m2"(%112, %113, %106) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2"
      %115 = literal "8" : !emitc.opaque<"int">
      %116 = call_opaque "__riscv_vsrl_vx_u16m2"(%114, %115, %106) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %117 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%116, %106) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %118 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%117) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m1"
      %119 = literal "-1" : !emitc.opaque<"int">
      %120 = call_opaque "__riscv_vadd_vx_i8m1"(%118, %119, %106) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %121 = literal "160" : index
      %122 = subscript %3[%121] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %123 = apply "&"(%122) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m1"
      call_opaque "__riscv_vse8_v_i8m1"(%123, %120, %106) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %124 = literal "16" : !emitc.opaque<"size_t">
      %125 = call_opaque "__riscv_vsetvl_e8m1"(%124) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %126 = literal "32" : !emitc.opaque<"size_t">
      %127 = add %16, %126 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %128 = cast %127 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
      %129 = call_opaque "__riscv_vle8_v_u8m1"(%128, %125) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1"
      %130 = literal "3" : !emitc.opaque<"int">
      %131 = call_opaque "__riscv_vmul_vx_u8m1"(%129, %130, %125) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m2"
      %132 = literal "3" : !emitc.opaque<"int">
      %133 = call_opaque "__riscv_vwmulu_vx_u16m2"(%131, %132, %125) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2"
      %134 = literal "8" : !emitc.opaque<"int">
      %135 = call_opaque "__riscv_vsrl_vx_u16m2"(%133, %134, %125) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %136 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%135, %125) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %137 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%136) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m1"
      %138 = literal "-1" : !emitc.opaque<"int">
      %139 = call_opaque "__riscv_vadd_vx_i8m1"(%137, %138, %125) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %140 = literal "176" : index
      %141 = subscript %3[%140] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %142 = apply "&"(%141) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m1"
      call_opaque "__riscv_vse8_v_i8m1"(%142, %139, %125) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %143 = literal "16" : !emitc.opaque<"size_t">
      %144 = call_opaque "__riscv_vsetvl_e8m1"(%143) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %145 = literal "32" : !emitc.opaque<"size_t">
      %146 = add %16, %145 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %147 = cast %146 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
      %148 = call_opaque "__riscv_vle8_v_u8m1"(%147, %144) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1"
      %149 = literal "9" : !emitc.opaque<"int">
      %150 = call_opaque "__riscv_vmul_vx_u8m1"(%148, %149, %144) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m2"
      %151 = literal "3" : !emitc.opaque<"int">
      %152 = call_opaque "__riscv_vwmulu_vx_u16m2"(%150, %151, %144) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2"
      %153 = literal "8" : !emitc.opaque<"int">
      %154 = call_opaque "__riscv_vsrl_vx_u16m2"(%152, %153, %144) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %155 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%154, %144) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %156 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%155) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m1"
      %157 = literal "-1" : !emitc.opaque<"int">
      %158 = call_opaque "__riscv_vadd_vx_i8m1"(%156, %157, %144) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %159 = literal "192" : index
      %160 = subscript %3[%159] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %161 = apply "&"(%160) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m1"
      call_opaque "__riscv_vse8_v_i8m1"(%161, %158, %144) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %162 = literal "16" : !emitc.opaque<"size_t">
      %163 = call_opaque "__riscv_vsetvl_e8m1"(%162) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %164 = literal "32" : !emitc.opaque<"size_t">
      %165 = add %16, %164 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %166 = cast %165 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
      %167 = call_opaque "__riscv_vle8_v_u8m1"(%166, %163) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1"
      %168 = literal "27" : !emitc.opaque<"int">
      %169 = call_opaque "__riscv_vmul_vx_u8m1"(%167, %168, %163) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m2"
      %170 = literal "3" : !emitc.opaque<"int">
      %171 = call_opaque "__riscv_vwmulu_vx_u16m2"(%169, %170, %163) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2"
      %172 = literal "8" : !emitc.opaque<"int">
      %173 = call_opaque "__riscv_vsrl_vx_u16m2"(%171, %172, %163) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %174 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%173, %163) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %175 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%174) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m1"
      %176 = literal "-1" : !emitc.opaque<"int">
      %177 = call_opaque "__riscv_vadd_vx_i8m1"(%175, %176, %163) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %178 = literal "208" : index
      %179 = subscript %3[%178] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %180 = apply "&"(%179) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m1"
      call_opaque "__riscv_vse8_v_i8m1"(%180, %177, %163) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %181 = literal "16" : !emitc.opaque<"size_t">
      %182 = call_opaque "__riscv_vsetvl_e8m1"(%181) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %183 = literal "32" : !emitc.opaque<"size_t">
      %184 = add %16, %183 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %185 = cast %184 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
      %186 = call_opaque "__riscv_vle8_v_u8m1"(%185, %182) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1"
      %187 = literal "81" : !emitc.opaque<"int">
      %188 = call_opaque "__riscv_vmul_vx_u8m1"(%186, %187, %182) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m2"
      %189 = literal "3" : !emitc.opaque<"int">
      %190 = call_opaque "__riscv_vwmulu_vx_u16m2"(%188, %189, %182) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2"
      %191 = literal "8" : !emitc.opaque<"int">
      %192 = call_opaque "__riscv_vsrl_vx_u16m2"(%190, %191, %182) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %193 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%192, %182) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %194 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%193) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m1"
      %195 = literal "-1" : !emitc.opaque<"int">
      %196 = call_opaque "__riscv_vadd_vx_i8m1"(%194, %195, %182) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %197 = literal "224" : index
      %198 = subscript %3[%197] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %199 = apply "&"(%198) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m1"
      call_opaque "__riscv_vse8_v_i8m1"(%199, %196, %182) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %200 = literal "4" : !emitc.opaque<"size_t">
      %201 = call_opaque "__riscv_vsetvl_e8m1"(%200) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %202 = literal "48" : !emitc.opaque<"size_t">
      %203 = add %16, %202 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %204 = cast %203 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
      %205 = call_opaque "__riscv_vle8_v_u8m1"(%204, %201) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1"
      %206 = literal "1" : !emitc.opaque<"int">
      %207 = call_opaque "__riscv_vmul_vx_u8m1"(%205, %206, %201) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m2"
      %208 = literal "3" : !emitc.opaque<"int">
      %209 = call_opaque "__riscv_vwmulu_vx_u16m2"(%207, %208, %201) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2"
      %210 = literal "8" : !emitc.opaque<"int">
      %211 = call_opaque "__riscv_vsrl_vx_u16m2"(%209, %210, %201) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %212 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%211, %201) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %213 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%212) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m1"
      %214 = literal "-1" : !emitc.opaque<"int">
      %215 = call_opaque "__riscv_vadd_vx_i8m1"(%213, %214, %201) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %216 = literal "240" : index
      %217 = subscript %3[%216] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %218 = apply "&"(%217) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m1"
      call_opaque "__riscv_vse8_v_i8m1"(%218, %215, %201) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %219 = literal "4" : !emitc.opaque<"size_t">
      %220 = call_opaque "__riscv_vsetvl_e8m1"(%219) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %221 = literal "48" : !emitc.opaque<"size_t">
      %222 = add %16, %221 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %223 = cast %222 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
      %224 = call_opaque "__riscv_vle8_v_u8m1"(%223, %220) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1"
      %225 = literal "3" : !emitc.opaque<"int">
      %226 = call_opaque "__riscv_vmul_vx_u8m1"(%224, %225, %220) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m2"
      %227 = literal "3" : !emitc.opaque<"int">
      %228 = call_opaque "__riscv_vwmulu_vx_u16m2"(%226, %227, %220) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2"
      %229 = literal "8" : !emitc.opaque<"int">
      %230 = call_opaque "__riscv_vsrl_vx_u16m2"(%228, %229, %220) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %231 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%230, %220) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %232 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%231) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m1"
      %233 = literal "-1" : !emitc.opaque<"int">
      %234 = call_opaque "__riscv_vadd_vx_i8m1"(%232, %233, %220) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %235 = literal "244" : index
      %236 = subscript %3[%235] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %237 = apply "&"(%236) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m1"
      call_opaque "__riscv_vse8_v_i8m1"(%237, %234, %220) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %238 = literal "4" : !emitc.opaque<"size_t">
      %239 = call_opaque "__riscv_vsetvl_e8m1"(%238) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %240 = literal "48" : !emitc.opaque<"size_t">
      %241 = add %16, %240 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %242 = cast %241 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
      %243 = call_opaque "__riscv_vle8_v_u8m1"(%242, %239) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1"
      %244 = literal "9" : !emitc.opaque<"int">
      %245 = call_opaque "__riscv_vmul_vx_u8m1"(%243, %244, %239) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m2"
      %246 = literal "3" : !emitc.opaque<"int">
      %247 = call_opaque "__riscv_vwmulu_vx_u16m2"(%245, %246, %239) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2"
      %248 = literal "8" : !emitc.opaque<"int">
      %249 = call_opaque "__riscv_vsrl_vx_u16m2"(%247, %248, %239) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %250 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%249, %239) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %251 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%250) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m1"
      %252 = literal "-1" : !emitc.opaque<"int">
      %253 = call_opaque "__riscv_vadd_vx_i8m1"(%251, %252, %239) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %254 = literal "248" : index
      %255 = subscript %3[%254] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %256 = apply "&"(%255) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m1"
      call_opaque "__riscv_vse8_v_i8m1"(%256, %253, %239) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %257 = literal "4" : !emitc.opaque<"size_t">
      %258 = call_opaque "__riscv_vsetvl_e8m1"(%257) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %259 = literal "48" : !emitc.opaque<"size_t">
      %260 = add %16, %259 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %261 = cast %260 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
      %262 = call_opaque "__riscv_vle8_v_u8m1"(%261, %258) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1"
      %263 = literal "27" : !emitc.opaque<"int">
      %264 = call_opaque "__riscv_vmul_vx_u8m1"(%262, %263, %258) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m2"
      %265 = literal "3" : !emitc.opaque<"int">
      %266 = call_opaque "__riscv_vwmulu_vx_u16m2"(%264, %265, %258) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2"
      %267 = literal "8" : !emitc.opaque<"int">
      %268 = call_opaque "__riscv_vsrl_vx_u16m2"(%266, %267, %258) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1"
      %269 = call_opaque "__riscv_vncvt_x_x_w_u8m1"(%268, %258) : (!emitc.opaque<"vuint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1"
      %270 = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%269) : (!emitc.opaque<"vuint8m1_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m1"
      %271 = literal "-1" : !emitc.opaque<"int">
      %272 = call_opaque "__riscv_vadd_vx_i8m1"(%270, %271, %258) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      %273 = literal "252" : index
      %274 = subscript %3[%273] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %275 = apply "&"(%274) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m1"
      call_opaque "__riscv_vse8_v_i8m1"(%275, %272, %258) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> ()
      %276 = literal "4" : !emitc.opaque<"size_t">
      %277 = add %19, %276 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %278 = cast %277 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %279 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int">>
      %280 = literal "0" : !emitc.opaque<"int">
      assign %280 : !emitc.opaque<"int"> to %279 : <!emitc.opaque<"int">>
      %281 = literal "0" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %282 = literal "16" : !emitc.opaque<"size_t">
      %283 = call_opaque "__riscv_vsetvl_e8m1"(%282) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %284 = literal "16" : !emitc.opaque<"size_t">
      %285 = mul %281, %284 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %286 = add %278, %285 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %287 = add %6, %285 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %288 = call_opaque "__riscv_vle8_v_i8m1"(%286, %283) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %289 = call_opaque "__riscv_vle8_v_i8m1"(%287, %283) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %290 = call_opaque "__riscv_vwmul_vv_i16m2"(%288, %289, %283) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %291 = literal "0" : !emitc.opaque<"int">
      %292 = literal "1" : !emitc.opaque<"size_t">
      %293 = call_opaque "__riscv_vmv_v_x_i32m1"(%291, %292) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %294 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%290, %293, %283) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %295 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%294) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %296 = load %279 : <!emitc.opaque<"int">>
      %297 = add %296, %295 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %297 : !emitc.opaque<"int"> to %279 : <!emitc.opaque<"int">>
      %298 = literal "1" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %299 = literal "16" : !emitc.opaque<"size_t">
      %300 = call_opaque "__riscv_vsetvl_e8m1"(%299) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %301 = literal "16" : !emitc.opaque<"size_t">
      %302 = mul %298, %301 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %303 = add %278, %302 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %304 = add %6, %302 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %305 = call_opaque "__riscv_vle8_v_i8m1"(%303, %300) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %306 = call_opaque "__riscv_vle8_v_i8m1"(%304, %300) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %307 = call_opaque "__riscv_vwmul_vv_i16m2"(%305, %306, %300) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %308 = literal "0" : !emitc.opaque<"int">
      %309 = literal "1" : !emitc.opaque<"size_t">
      %310 = call_opaque "__riscv_vmv_v_x_i32m1"(%308, %309) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %311 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%307, %310, %300) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %312 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%311) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %313 = load %279 : <!emitc.opaque<"int">>
      %314 = add %313, %312 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %314 : !emitc.opaque<"int"> to %279 : <!emitc.opaque<"int">>
      %315 = literal "2" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %316 = literal "16" : !emitc.opaque<"size_t">
      %317 = call_opaque "__riscv_vsetvl_e8m1"(%316) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %318 = literal "16" : !emitc.opaque<"size_t">
      %319 = mul %315, %318 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %320 = add %278, %319 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %321 = add %6, %319 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %322 = call_opaque "__riscv_vle8_v_i8m1"(%320, %317) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %323 = call_opaque "__riscv_vle8_v_i8m1"(%321, %317) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %324 = call_opaque "__riscv_vwmul_vv_i16m2"(%322, %323, %317) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %325 = literal "0" : !emitc.opaque<"int">
      %326 = literal "1" : !emitc.opaque<"size_t">
      %327 = call_opaque "__riscv_vmv_v_x_i32m1"(%325, %326) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %328 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%324, %327, %317) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %329 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%328) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %330 = load %279 : <!emitc.opaque<"int">>
      %331 = add %330, %329 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %331 : !emitc.opaque<"int"> to %279 : <!emitc.opaque<"int">>
      %332 = literal "3" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %333 = literal "16" : !emitc.opaque<"size_t">
      %334 = call_opaque "__riscv_vsetvl_e8m1"(%333) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %335 = literal "16" : !emitc.opaque<"size_t">
      %336 = mul %332, %335 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %337 = add %278, %336 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %338 = add %6, %336 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %339 = call_opaque "__riscv_vle8_v_i8m1"(%337, %334) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %340 = call_opaque "__riscv_vle8_v_i8m1"(%338, %334) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %341 = call_opaque "__riscv_vwmul_vv_i16m2"(%339, %340, %334) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %342 = literal "0" : !emitc.opaque<"int">
      %343 = literal "1" : !emitc.opaque<"size_t">
      %344 = call_opaque "__riscv_vmv_v_x_i32m1"(%342, %343) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %345 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%341, %344, %334) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %346 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%345) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %347 = load %279 : <!emitc.opaque<"int">>
      %348 = add %347, %346 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %348 : !emitc.opaque<"int"> to %279 : <!emitc.opaque<"int">>
      %349 = literal "4" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %350 = literal "16" : !emitc.opaque<"size_t">
      %351 = call_opaque "__riscv_vsetvl_e8m1"(%350) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %352 = literal "16" : !emitc.opaque<"size_t">
      %353 = mul %349, %352 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %354 = add %278, %353 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %355 = add %6, %353 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %356 = call_opaque "__riscv_vle8_v_i8m1"(%354, %351) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %357 = call_opaque "__riscv_vle8_v_i8m1"(%355, %351) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %358 = call_opaque "__riscv_vwmul_vv_i16m2"(%356, %357, %351) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %359 = literal "0" : !emitc.opaque<"int">
      %360 = literal "1" : !emitc.opaque<"size_t">
      %361 = call_opaque "__riscv_vmv_v_x_i32m1"(%359, %360) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %362 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%358, %361, %351) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %363 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%362) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %364 = load %279 : <!emitc.opaque<"int">>
      %365 = add %364, %363 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %365 : !emitc.opaque<"int"> to %279 : <!emitc.opaque<"int">>
      %366 = literal "5" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %367 = literal "16" : !emitc.opaque<"size_t">
      %368 = call_opaque "__riscv_vsetvl_e8m1"(%367) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %369 = literal "16" : !emitc.opaque<"size_t">
      %370 = mul %366, %369 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %371 = add %278, %370 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %372 = add %6, %370 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %373 = call_opaque "__riscv_vle8_v_i8m1"(%371, %368) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %374 = call_opaque "__riscv_vle8_v_i8m1"(%372, %368) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %375 = call_opaque "__riscv_vwmul_vv_i16m2"(%373, %374, %368) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %376 = literal "0" : !emitc.opaque<"int">
      %377 = literal "1" : !emitc.opaque<"size_t">
      %378 = call_opaque "__riscv_vmv_v_x_i32m1"(%376, %377) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %379 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%375, %378, %368) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %380 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%379) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %381 = load %279 : <!emitc.opaque<"int">>
      %382 = add %381, %380 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %382 : !emitc.opaque<"int"> to %279 : <!emitc.opaque<"int">>
      %383 = literal "6" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %384 = literal "16" : !emitc.opaque<"size_t">
      %385 = call_opaque "__riscv_vsetvl_e8m1"(%384) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %386 = literal "16" : !emitc.opaque<"size_t">
      %387 = mul %383, %386 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %388 = add %278, %387 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %389 = add %6, %387 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %390 = call_opaque "__riscv_vle8_v_i8m1"(%388, %385) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %391 = call_opaque "__riscv_vle8_v_i8m1"(%389, %385) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %392 = call_opaque "__riscv_vwmul_vv_i16m2"(%390, %391, %385) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %393 = literal "0" : !emitc.opaque<"int">
      %394 = literal "1" : !emitc.opaque<"size_t">
      %395 = call_opaque "__riscv_vmv_v_x_i32m1"(%393, %394) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %396 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%392, %395, %385) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %397 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%396) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %398 = load %279 : <!emitc.opaque<"int">>
      %399 = add %398, %397 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %399 : !emitc.opaque<"int"> to %279 : <!emitc.opaque<"int">>
      %400 = literal "7" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %401 = literal "16" : !emitc.opaque<"size_t">
      %402 = call_opaque "__riscv_vsetvl_e8m1"(%401) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %403 = literal "16" : !emitc.opaque<"size_t">
      %404 = mul %400, %403 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %405 = add %278, %404 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %406 = add %6, %404 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %407 = call_opaque "__riscv_vle8_v_i8m1"(%405, %402) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %408 = call_opaque "__riscv_vle8_v_i8m1"(%406, %402) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %409 = call_opaque "__riscv_vwmul_vv_i16m2"(%407, %408, %402) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %410 = literal "0" : !emitc.opaque<"int">
      %411 = literal "1" : !emitc.opaque<"size_t">
      %412 = call_opaque "__riscv_vmv_v_x_i32m1"(%410, %411) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %413 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%409, %412, %402) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %414 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%413) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %415 = load %279 : <!emitc.opaque<"int">>
      %416 = add %415, %414 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %416 : !emitc.opaque<"int"> to %279 : <!emitc.opaque<"int">>
      %417 = literal "8" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %418 = literal "16" : !emitc.opaque<"size_t">
      %419 = call_opaque "__riscv_vsetvl_e8m1"(%418) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %420 = literal "16" : !emitc.opaque<"size_t">
      %421 = mul %417, %420 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %422 = add %278, %421 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %423 = add %6, %421 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %424 = call_opaque "__riscv_vle8_v_i8m1"(%422, %419) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %425 = call_opaque "__riscv_vle8_v_i8m1"(%423, %419) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %426 = call_opaque "__riscv_vwmul_vv_i16m2"(%424, %425, %419) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %427 = literal "0" : !emitc.opaque<"int">
      %428 = literal "1" : !emitc.opaque<"size_t">
      %429 = call_opaque "__riscv_vmv_v_x_i32m1"(%427, %428) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %430 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%426, %429, %419) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %431 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%430) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %432 = load %279 : <!emitc.opaque<"int">>
      %433 = add %432, %431 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %433 : !emitc.opaque<"int"> to %279 : <!emitc.opaque<"int">>
      %434 = literal "9" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %435 = literal "16" : !emitc.opaque<"size_t">
      %436 = call_opaque "__riscv_vsetvl_e8m1"(%435) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %437 = literal "16" : !emitc.opaque<"size_t">
      %438 = mul %434, %437 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %439 = add %278, %438 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %440 = add %6, %438 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %441 = call_opaque "__riscv_vle8_v_i8m1"(%439, %436) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %442 = call_opaque "__riscv_vle8_v_i8m1"(%440, %436) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %443 = call_opaque "__riscv_vwmul_vv_i16m2"(%441, %442, %436) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %444 = literal "0" : !emitc.opaque<"int">
      %445 = literal "1" : !emitc.opaque<"size_t">
      %446 = call_opaque "__riscv_vmv_v_x_i32m1"(%444, %445) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %447 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%443, %446, %436) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %448 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%447) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %449 = load %279 : <!emitc.opaque<"int">>
      %450 = add %449, %448 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %450 : !emitc.opaque<"int"> to %279 : <!emitc.opaque<"int">>
      %451 = literal "10" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %452 = literal "16" : !emitc.opaque<"size_t">
      %453 = call_opaque "__riscv_vsetvl_e8m1"(%452) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %454 = literal "16" : !emitc.opaque<"size_t">
      %455 = mul %451, %454 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %456 = add %278, %455 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %457 = add %6, %455 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %458 = call_opaque "__riscv_vle8_v_i8m1"(%456, %453) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %459 = call_opaque "__riscv_vle8_v_i8m1"(%457, %453) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %460 = call_opaque "__riscv_vwmul_vv_i16m2"(%458, %459, %453) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %461 = literal "0" : !emitc.opaque<"int">
      %462 = literal "1" : !emitc.opaque<"size_t">
      %463 = call_opaque "__riscv_vmv_v_x_i32m1"(%461, %462) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %464 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%460, %463, %453) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %465 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%464) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %466 = load %279 : <!emitc.opaque<"int">>
      %467 = add %466, %465 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %467 : !emitc.opaque<"int"> to %279 : <!emitc.opaque<"int">>
      %468 = literal "11" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %469 = literal "16" : !emitc.opaque<"size_t">
      %470 = call_opaque "__riscv_vsetvl_e8m1"(%469) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %471 = literal "16" : !emitc.opaque<"size_t">
      %472 = mul %468, %471 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %473 = add %278, %472 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %474 = add %6, %472 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %475 = call_opaque "__riscv_vle8_v_i8m1"(%473, %470) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %476 = call_opaque "__riscv_vle8_v_i8m1"(%474, %470) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %477 = call_opaque "__riscv_vwmul_vv_i16m2"(%475, %476, %470) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %478 = literal "0" : !emitc.opaque<"int">
      %479 = literal "1" : !emitc.opaque<"size_t">
      %480 = call_opaque "__riscv_vmv_v_x_i32m1"(%478, %479) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %481 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%477, %480, %470) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %482 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%481) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %483 = load %279 : <!emitc.opaque<"int">>
      %484 = add %483, %482 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %484 : !emitc.opaque<"int"> to %279 : <!emitc.opaque<"int">>
      %485 = literal "12" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %486 = literal "16" : !emitc.opaque<"size_t">
      %487 = call_opaque "__riscv_vsetvl_e8m1"(%486) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %488 = literal "16" : !emitc.opaque<"size_t">
      %489 = mul %485, %488 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %490 = add %278, %489 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %491 = add %6, %489 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %492 = call_opaque "__riscv_vle8_v_i8m1"(%490, %487) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %493 = call_opaque "__riscv_vle8_v_i8m1"(%491, %487) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %494 = call_opaque "__riscv_vwmul_vv_i16m2"(%492, %493, %487) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %495 = literal "0" : !emitc.opaque<"int">
      %496 = literal "1" : !emitc.opaque<"size_t">
      %497 = call_opaque "__riscv_vmv_v_x_i32m1"(%495, %496) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %498 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%494, %497, %487) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %499 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%498) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %500 = load %279 : <!emitc.opaque<"int">>
      %501 = add %500, %499 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %501 : !emitc.opaque<"int"> to %279 : <!emitc.opaque<"int">>
      %502 = literal "13" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %503 = literal "16" : !emitc.opaque<"size_t">
      %504 = call_opaque "__riscv_vsetvl_e8m1"(%503) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %505 = literal "16" : !emitc.opaque<"size_t">
      %506 = mul %502, %505 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %507 = add %278, %506 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %508 = add %6, %506 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %509 = call_opaque "__riscv_vle8_v_i8m1"(%507, %504) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %510 = call_opaque "__riscv_vle8_v_i8m1"(%508, %504) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %511 = call_opaque "__riscv_vwmul_vv_i16m2"(%509, %510, %504) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %512 = literal "0" : !emitc.opaque<"int">
      %513 = literal "1" : !emitc.opaque<"size_t">
      %514 = call_opaque "__riscv_vmv_v_x_i32m1"(%512, %513) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %515 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%511, %514, %504) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %516 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%515) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %517 = load %279 : <!emitc.opaque<"int">>
      %518 = add %517, %516 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %518 : !emitc.opaque<"int"> to %279 : <!emitc.opaque<"int">>
      %519 = literal "14" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %520 = literal "16" : !emitc.opaque<"size_t">
      %521 = call_opaque "__riscv_vsetvl_e8m1"(%520) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %522 = literal "16" : !emitc.opaque<"size_t">
      %523 = mul %519, %522 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %524 = add %278, %523 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %525 = add %6, %523 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %526 = call_opaque "__riscv_vle8_v_i8m1"(%524, %521) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %527 = call_opaque "__riscv_vle8_v_i8m1"(%525, %521) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %528 = call_opaque "__riscv_vwmul_vv_i16m2"(%526, %527, %521) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %529 = literal "0" : !emitc.opaque<"int">
      %530 = literal "1" : !emitc.opaque<"size_t">
      %531 = call_opaque "__riscv_vmv_v_x_i32m1"(%529, %530) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %532 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%528, %531, %521) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %533 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%532) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %534 = load %279 : <!emitc.opaque<"int">>
      %535 = add %534, %533 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %535 : !emitc.opaque<"int"> to %279 : <!emitc.opaque<"int">>
      %536 = literal "15" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %537 = literal "16" : !emitc.opaque<"size_t">
      %538 = call_opaque "__riscv_vsetvl_e8m1"(%537) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %539 = literal "16" : !emitc.opaque<"size_t">
      %540 = mul %536, %539 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %541 = add %278, %540 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %542 = add %6, %540 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %543 = call_opaque "__riscv_vle8_v_i8m1"(%541, %538) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %544 = call_opaque "__riscv_vle8_v_i8m1"(%542, %538) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %545 = call_opaque "__riscv_vwmul_vv_i16m2"(%543, %544, %538) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %546 = literal "0" : !emitc.opaque<"int">
      %547 = literal "1" : !emitc.opaque<"size_t">
      %548 = call_opaque "__riscv_vmv_v_x_i32m1"(%546, %547) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %549 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%545, %548, %538) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %550 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%549) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %551 = load %279 : <!emitc.opaque<"int">>
      %552 = add %551, %550 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %552 : !emitc.opaque<"int"> to %279 : <!emitc.opaque<"int">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_d"
      %553 = literal "52" : !emitc.opaque<"size_t">
      %554 = add %16, %553 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %555 = call_opaque "(float)*(const _Float16 *)"(%554) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_activation_d"
      %556 = cast %19 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const float">>
      %557 = literal "0" : index
      %558 = subscript %556[%557] : (!emitc.ptr<!emitc.opaque<"const float">>, index) -> !emitc.lvalue<!emitc.opaque<"const float">>
      %559 = load %558 : <!emitc.opaque<"const float">>
      %560 = mul %555, %559 : (!emitc.opaque<"float">, !emitc.opaque<"const float">) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scalar_fold"
      %561 = load %279 : <!emitc.opaque<"int">>
      %562 = load %7 : <!emitc.opaque<"float">>
      %563 = expression : !emitc.opaque<"float"> {
        %564 = cast %561 : !emitc.opaque<"int"> to !emitc.opaque<"float">
        %565 = mul %564, %560 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %566 = add %562, %565 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %566 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %563 : !emitc.opaque<"float"> to %7 : <!emitc.opaque<"float">>
    }
    %11 = load %7 : <!emitc.opaque<"float">>
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s"
    %12 = literal "0" : index
    %13 = subscript %arg1[%12] : (!emitc.ptr<!emitc.opaque<"float">>, index) -> !emitc.lvalue<!emitc.opaque<"float">>
    assign %11 : !emitc.opaque<"float"> to %13 : <!emitc.opaque<"float">>
    return
  }
}

