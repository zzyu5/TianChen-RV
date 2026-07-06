module {
  emitc.include <"stddef.h">
  emitc.include <"stdint.h">
  emitc.include <"riscv_vector.h">
  emitc.func @tcrv_emitc_ggml_vec_dot_q5_K_q8_K_kernel_rvv_q5_K_q8_K_block_dot(%arg0: !emitc.opaque<"size_t">, %arg1: !emitc.ptr<!emitc.opaque<"float">>, %arg2: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg3: !emitc.ptr<!emitc.opaque<"const uint8_t">>) attributes {specifiers = ["extern", "\22C\22"]} {
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1"
    %0 = call_opaque "__riscv_vsetvl_e32m1"(%arg0) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count"
    %1 = literal "256" : !emitc.opaque<"size_t">
    %2 = div %arg0, %1 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.local_variable=aux8 source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface"
    %3 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<256x!emitc.opaque<"int8_t">>
    %4 = literal "0" : index
    %5 = subscript %3[%4] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
    %6 = apply "&"(%5) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
    verbatim "// tcrv_emitc.local_variable=utmp source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface"
    %7 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<4x!emitc.opaque<"uint32_t">>
    verbatim "// tcrv_emitc.local_variable=sums8 source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface"
    %8 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<8x!emitc.opaque<"float">>
    verbatim "// tcrv_emitc.local_variable=sums source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface"
    %9 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m2_t">>
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2"
    %10 = literal "0.0f" : !emitc.opaque<"float">
    %11 = literal "8" : !emitc.opaque<"size_t">
    %12 = call_opaque "__riscv_vfmv_v_f_f32m2"(%10, %11) : (!emitc.opaque<"float">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
    assign %12 : !emitc.opaque<"vfloat32m2_t"> to %9 : <!emitc.opaque<"vfloat32m2_t">>
    verbatim "// tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface"
    %13 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"float">>
    %14 = literal "0.0f" : !emitc.opaque<"float">
    assign %14 : !emitc.opaque<"float"> to %13 : <!emitc.opaque<"float">>
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_loop"
    %15 = literal "1" : !emitc.opaque<"size_t">
    %16 = literal "0" : !emitc.opaque<"size_t">
    for %arg4 = %16 to %2 step %15  : !emitc.opaque<"size_t"> {
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x"
      %57 = literal "176" : !emitc.opaque<"size_t">
      %58 = mul %arg4, %57 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %59 = add %arg2, %58 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_y"
      %60 = literal "292" : !emitc.opaque<"size_t">
      %61 = mul %arg4, %60 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %62 = add %arg3, %61 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=unpack_4bit"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2"
      %63 = literal "32" : !emitc.opaque<"size_t">
      %64 = call_opaque "__riscv_vsetvl_e8m2"(%63) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %65 = literal "48" : !emitc.opaque<"size_t">
      %66 = add %59, %65 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %67 = cast %66 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2"
      %68 = call_opaque "__riscv_vle8_v_u8m2"(%67, %64) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_high_bit_plane"
      %69 = literal "16" : !emitc.opaque<"size_t">
      %70 = add %59, %69 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %71 = cast %70 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2"
      %72 = call_opaque "__riscv_vle8_v_u8m2"(%71, %64) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2"
      %73 = literal "0x0F" : !emitc.opaque<"int">
      %74 = call_opaque "__riscv_vand_vx_u8m2"(%68, %73, %64) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2"
      %75 = literal "0x01" : !emitc.opaque<"int">
      %76 = call_opaque "__riscv_vand_vx_u8m2"(%72, %75, %64) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2"
      %77 = literal "0x04" : !emitc.opaque<"int">
      %78 = call_opaque "__riscv_vsll_vx_u8m2"(%76, %77, %64) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_u8m2"
      %79 = call_opaque "__riscv_vadd_vv_u8m2"(%74, %78, %64) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"vuint8m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2"
      %80 = call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"(%79) : (!emitc.opaque<"vuint8m2_t">) -> !emitc.opaque<"vint8m2_t">
      %81 = literal "0" : index
      %82 = subscript %3[%81] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %83 = apply "&"(%82) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2"
      call_opaque "__riscv_vse8_v_i8m2"(%83, %80, %64) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2"
      %84 = literal "0x04" : !emitc.opaque<"int">
      %85 = call_opaque "__riscv_vsrl_vx_u8m2"(%68, %84, %64) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2"
      %86 = literal "1" : !emitc.opaque<"int">
      %87 = call_opaque "__riscv_vsrl_vx_u8m2"(%72, %86, %64) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2"
      %88 = literal "0x01" : !emitc.opaque<"int">
      %89 = call_opaque "__riscv_vand_vx_u8m2"(%87, %88, %64) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2"
      %90 = literal "0x04" : !emitc.opaque<"int">
      %91 = call_opaque "__riscv_vsll_vx_u8m2"(%89, %90, %64) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_u8m2"
      %92 = call_opaque "__riscv_vadd_vv_u8m2"(%85, %91, %64) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"vuint8m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2"
      %93 = call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"(%92) : (!emitc.opaque<"vuint8m2_t">) -> !emitc.opaque<"vint8m2_t">
      %94 = literal "32" : index
      %95 = subscript %3[%94] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %96 = apply "&"(%95) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2"
      call_opaque "__riscv_vse8_v_i8m2"(%96, %93, %64) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2"
      %97 = literal "32" : !emitc.opaque<"size_t">
      %98 = call_opaque "__riscv_vsetvl_e8m2"(%97) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %99 = literal "80" : !emitc.opaque<"size_t">
      %100 = add %59, %99 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %101 = cast %100 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2"
      %102 = call_opaque "__riscv_vle8_v_u8m2"(%101, %98) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_high_bit_plane"
      %103 = literal "16" : !emitc.opaque<"size_t">
      %104 = add %59, %103 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %105 = cast %104 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2"
      %106 = call_opaque "__riscv_vle8_v_u8m2"(%105, %98) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2"
      %107 = literal "0x0F" : !emitc.opaque<"int">
      %108 = call_opaque "__riscv_vand_vx_u8m2"(%102, %107, %98) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2"
      %109 = literal "2" : !emitc.opaque<"int">
      %110 = call_opaque "__riscv_vsrl_vx_u8m2"(%106, %109, %98) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2"
      %111 = literal "0x01" : !emitc.opaque<"int">
      %112 = call_opaque "__riscv_vand_vx_u8m2"(%110, %111, %98) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2"
      %113 = literal "0x04" : !emitc.opaque<"int">
      %114 = call_opaque "__riscv_vsll_vx_u8m2"(%112, %113, %98) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_u8m2"
      %115 = call_opaque "__riscv_vadd_vv_u8m2"(%108, %114, %98) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"vuint8m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2"
      %116 = call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"(%115) : (!emitc.opaque<"vuint8m2_t">) -> !emitc.opaque<"vint8m2_t">
      %117 = literal "64" : index
      %118 = subscript %3[%117] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %119 = apply "&"(%118) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2"
      call_opaque "__riscv_vse8_v_i8m2"(%119, %116, %98) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2"
      %120 = literal "0x04" : !emitc.opaque<"int">
      %121 = call_opaque "__riscv_vsrl_vx_u8m2"(%102, %120, %98) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2"
      %122 = literal "3" : !emitc.opaque<"int">
      %123 = call_opaque "__riscv_vsrl_vx_u8m2"(%106, %122, %98) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2"
      %124 = literal "0x01" : !emitc.opaque<"int">
      %125 = call_opaque "__riscv_vand_vx_u8m2"(%123, %124, %98) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2"
      %126 = literal "0x04" : !emitc.opaque<"int">
      %127 = call_opaque "__riscv_vsll_vx_u8m2"(%125, %126, %98) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_u8m2"
      %128 = call_opaque "__riscv_vadd_vv_u8m2"(%121, %127, %98) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"vuint8m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2"
      %129 = call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"(%128) : (!emitc.opaque<"vuint8m2_t">) -> !emitc.opaque<"vint8m2_t">
      %130 = literal "96" : index
      %131 = subscript %3[%130] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %132 = apply "&"(%131) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2"
      call_opaque "__riscv_vse8_v_i8m2"(%132, %129, %98) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2"
      %133 = literal "32" : !emitc.opaque<"size_t">
      %134 = call_opaque "__riscv_vsetvl_e8m2"(%133) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %135 = literal "112" : !emitc.opaque<"size_t">
      %136 = add %59, %135 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %137 = cast %136 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2"
      %138 = call_opaque "__riscv_vle8_v_u8m2"(%137, %134) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_high_bit_plane"
      %139 = literal "16" : !emitc.opaque<"size_t">
      %140 = add %59, %139 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %141 = cast %140 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2"
      %142 = call_opaque "__riscv_vle8_v_u8m2"(%141, %134) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2"
      %143 = literal "0x0F" : !emitc.opaque<"int">
      %144 = call_opaque "__riscv_vand_vx_u8m2"(%138, %143, %134) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2"
      %145 = literal "4" : !emitc.opaque<"int">
      %146 = call_opaque "__riscv_vsrl_vx_u8m2"(%142, %145, %134) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2"
      %147 = literal "0x01" : !emitc.opaque<"int">
      %148 = call_opaque "__riscv_vand_vx_u8m2"(%146, %147, %134) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2"
      %149 = literal "0x04" : !emitc.opaque<"int">
      %150 = call_opaque "__riscv_vsll_vx_u8m2"(%148, %149, %134) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_u8m2"
      %151 = call_opaque "__riscv_vadd_vv_u8m2"(%144, %150, %134) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"vuint8m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2"
      %152 = call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"(%151) : (!emitc.opaque<"vuint8m2_t">) -> !emitc.opaque<"vint8m2_t">
      %153 = literal "128" : index
      %154 = subscript %3[%153] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %155 = apply "&"(%154) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2"
      call_opaque "__riscv_vse8_v_i8m2"(%155, %152, %134) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2"
      %156 = literal "0x04" : !emitc.opaque<"int">
      %157 = call_opaque "__riscv_vsrl_vx_u8m2"(%138, %156, %134) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2"
      %158 = literal "5" : !emitc.opaque<"int">
      %159 = call_opaque "__riscv_vsrl_vx_u8m2"(%142, %158, %134) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2"
      %160 = literal "0x01" : !emitc.opaque<"int">
      %161 = call_opaque "__riscv_vand_vx_u8m2"(%159, %160, %134) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2"
      %162 = literal "0x04" : !emitc.opaque<"int">
      %163 = call_opaque "__riscv_vsll_vx_u8m2"(%161, %162, %134) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_u8m2"
      %164 = call_opaque "__riscv_vadd_vv_u8m2"(%157, %163, %134) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"vuint8m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2"
      %165 = call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"(%164) : (!emitc.opaque<"vuint8m2_t">) -> !emitc.opaque<"vint8m2_t">
      %166 = literal "160" : index
      %167 = subscript %3[%166] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %168 = apply "&"(%167) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2"
      call_opaque "__riscv_vse8_v_i8m2"(%168, %165, %134) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2"
      %169 = literal "32" : !emitc.opaque<"size_t">
      %170 = call_opaque "__riscv_vsetvl_e8m2"(%169) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %171 = literal "144" : !emitc.opaque<"size_t">
      %172 = add %59, %171 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %173 = cast %172 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2"
      %174 = call_opaque "__riscv_vle8_v_u8m2"(%173, %170) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_high_bit_plane"
      %175 = literal "16" : !emitc.opaque<"size_t">
      %176 = add %59, %175 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %177 = cast %176 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2"
      %178 = call_opaque "__riscv_vle8_v_u8m2"(%177, %170) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2"
      %179 = literal "0x0F" : !emitc.opaque<"int">
      %180 = call_opaque "__riscv_vand_vx_u8m2"(%174, %179, %170) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2"
      %181 = literal "6" : !emitc.opaque<"int">
      %182 = call_opaque "__riscv_vsrl_vx_u8m2"(%178, %181, %170) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2"
      %183 = literal "0x01" : !emitc.opaque<"int">
      %184 = call_opaque "__riscv_vand_vx_u8m2"(%182, %183, %170) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2"
      %185 = literal "0x04" : !emitc.opaque<"int">
      %186 = call_opaque "__riscv_vsll_vx_u8m2"(%184, %185, %170) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_u8m2"
      %187 = call_opaque "__riscv_vadd_vv_u8m2"(%180, %186, %170) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"vuint8m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2"
      %188 = call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"(%187) : (!emitc.opaque<"vuint8m2_t">) -> !emitc.opaque<"vint8m2_t">
      %189 = literal "192" : index
      %190 = subscript %3[%189] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %191 = apply "&"(%190) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2"
      call_opaque "__riscv_vse8_v_i8m2"(%191, %188, %170) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2"
      %192 = literal "0x04" : !emitc.opaque<"int">
      %193 = call_opaque "__riscv_vsrl_vx_u8m2"(%174, %192, %170) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2"
      %194 = literal "7" : !emitc.opaque<"int">
      %195 = call_opaque "__riscv_vsrl_vx_u8m2"(%178, %194, %170) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2"
      %196 = literal "0x01" : !emitc.opaque<"int">
      %197 = call_opaque "__riscv_vand_vx_u8m2"(%195, %196, %170) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2"
      %198 = literal "0x04" : !emitc.opaque<"int">
      %199 = call_opaque "__riscv_vsll_vx_u8m2"(%197, %198, %170) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_u8m2"
      %200 = call_opaque "__riscv_vadd_vv_u8m2"(%193, %199, %170) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"vuint8m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2"
      %201 = call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"(%200) : (!emitc.opaque<"vuint8m2_t">) -> !emitc.opaque<"vint8m2_t">
      %202 = literal "224" : index
      %203 = subscript %3[%202] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %204 = apply "&"(%203) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2"
      call_opaque "__riscv_vse8_v_i8m2"(%204, %201, %170) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_min_bit_dance"
      %205 = literal "0x3f3f3f3f" : !emitc.opaque<"uint32_t">
      %206 = literal "0x0f0f0f0f" : !emitc.opaque<"uint32_t">
      %207 = literal "0x03030303" : !emitc.opaque<"uint32_t">
      %208 = literal "4" : !emitc.opaque<"uint32_t">
      %209 = literal "6" : !emitc.opaque<"uint32_t">
      %210 = literal "4" : !emitc.opaque<"size_t">
      %211 = add %59, %210 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %212 = cast %211 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint32_t">>
      %213 = literal "0" : index
      %214 = subscript %212[%213] : (!emitc.ptr<!emitc.opaque<"const uint32_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint32_t">>
      %215 = load %214 : <!emitc.opaque<"const uint32_t">>
      %216 = cast %215 : !emitc.opaque<"const uint32_t"> to !emitc.opaque<"uint32_t">
      %217 = literal "1" : index
      %218 = subscript %212[%217] : (!emitc.ptr<!emitc.opaque<"const uint32_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint32_t">>
      %219 = load %218 : <!emitc.opaque<"const uint32_t">>
      %220 = cast %219 : !emitc.opaque<"const uint32_t"> to !emitc.opaque<"uint32_t">
      %221 = literal "2" : index
      %222 = subscript %212[%221] : (!emitc.ptr<!emitc.opaque<"const uint32_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint32_t">>
      %223 = load %222 : <!emitc.opaque<"const uint32_t">>
      %224 = cast %223 : !emitc.opaque<"const uint32_t"> to !emitc.opaque<"uint32_t">
      %225 = bitwise_right_shift %220, %209 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %226 = bitwise_and %225, %207 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %227 = bitwise_left_shift %226, %208 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %228 = bitwise_right_shift %224, %208 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %229 = bitwise_and %228, %206 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %230 = bitwise_or %229, %227 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %231 = bitwise_and %220, %205 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %232 = bitwise_right_shift %216, %209 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %233 = bitwise_and %232, %207 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %234 = bitwise_left_shift %233, %208 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %235 = bitwise_and %224, %206 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %236 = bitwise_or %235, %234 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %237 = bitwise_and %216, %205 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %238 = literal "0" : index
      %239 = subscript %7[%238] : (!emitc.array<4x!emitc.opaque<"uint32_t">>, index) -> !emitc.lvalue<!emitc.opaque<"uint32_t">>
      assign %237 : !emitc.opaque<"uint32_t"> to %239 : <!emitc.opaque<"uint32_t">>
      %240 = literal "1" : index
      %241 = subscript %7[%240] : (!emitc.array<4x!emitc.opaque<"uint32_t">>, index) -> !emitc.lvalue<!emitc.opaque<"uint32_t">>
      assign %236 : !emitc.opaque<"uint32_t"> to %241 : <!emitc.opaque<"uint32_t">>
      %242 = literal "2" : index
      %243 = subscript %7[%242] : (!emitc.array<4x!emitc.opaque<"uint32_t">>, index) -> !emitc.lvalue<!emitc.opaque<"uint32_t">>
      assign %231 : !emitc.opaque<"uint32_t"> to %243 : <!emitc.opaque<"uint32_t">>
      %244 = literal "3" : index
      %245 = subscript %7[%244] : (!emitc.array<4x!emitc.opaque<"uint32_t">>, index) -> !emitc.lvalue<!emitc.opaque<"uint32_t">>
      assign %230 : !emitc.opaque<"uint32_t"> to %245 : <!emitc.opaque<"uint32_t">>
      %246 = literal "0" : index
      %247 = subscript %7[%246] : (!emitc.array<4x!emitc.opaque<"uint32_t">>, index) -> !emitc.lvalue<!emitc.opaque<"uint32_t">>
      %248 = apply "&"(%247) : (!emitc.lvalue<!emitc.opaque<"uint32_t">>) -> !emitc.ptr<!emitc.opaque<"uint32_t">>
      %249 = cast %248 : !emitc.ptr<!emitc.opaque<"uint32_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.local_variable=aux32 source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %250 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint32m2_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2"
      %251 = literal "0" : !emitc.opaque<"int">
      %252 = literal "8" : !emitc.opaque<"size_t">
      %253 = call_opaque "__riscv_vmv_v_x_i32m2"(%251, %252) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
      assign %253 : !emitc.opaque<"vint32m2_t"> to %250 : <!emitc.opaque<"vint32m2_t">>
      %254 = literal "4" : !emitc.opaque<"size_t">
      %255 = add %62, %254 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %256 = cast %255 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_loop"
      %257 = literal "1" : !emitc.opaque<"size_t">
      %258 = literal "8" : !emitc.opaque<"size_t">
      %259 = literal "0" : !emitc.opaque<"size_t">
      for %arg5 = %259 to %258 step %257  : !emitc.opaque<"size_t"> {
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_load"
        %462 = subscript %249[%arg5] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
        %463 = load %462 : <!emitc.opaque<"const uint8_t">>
        %464 = cast %463 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"int">
        %465 = literal "32" : !emitc.opaque<"size_t">
        %466 = mul %arg5, %465 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_quarter"
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2"
        %467 = literal "8" : !emitc.opaque<"size_t">
        %468 = call_opaque "__riscv_vsetvl_e8mf2"(%467) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %469 = add %256, %466 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
        %470 = add %6, %466 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2"
        %471 = call_opaque "__riscv_vle8_v_i8mf2"(%469, %468) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2"
        %472 = call_opaque "__riscv_vle8_v_i8mf2"(%470, %468) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1"
        %473 = call_opaque "__riscv_vwmul_vv_i16m1"(%471, %472, %468) : (!emitc.opaque<"vint8mf2_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2"
        %474 = load %250 : <!emitc.opaque<"vint32m2_t">>
        %475 = call_opaque "__riscv_vwmacc_vx_i32m2"(%474, %464, %473, %468) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"int">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        verbatim "// tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface"
        assign %475 : !emitc.opaque<"vint32m2_t"> to %250 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_quarter"
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2"
        %476 = literal "8" : !emitc.opaque<"size_t">
        %477 = call_opaque "__riscv_vsetvl_e8mf2"(%476) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %478 = literal "8" : !emitc.opaque<"size_t">
        %479 = add %466, %478 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %480 = add %256, %479 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
        %481 = add %6, %479 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2"
        %482 = call_opaque "__riscv_vle8_v_i8mf2"(%480, %477) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2"
        %483 = call_opaque "__riscv_vle8_v_i8mf2"(%481, %477) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1"
        %484 = call_opaque "__riscv_vwmul_vv_i16m1"(%482, %483, %477) : (!emitc.opaque<"vint8mf2_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2"
        %485 = load %250 : <!emitc.opaque<"vint32m2_t">>
        %486 = call_opaque "__riscv_vwmacc_vx_i32m2"(%485, %464, %484, %477) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"int">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        verbatim "// tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface"
        assign %486 : !emitc.opaque<"vint32m2_t"> to %250 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_quarter"
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2"
        %487 = literal "8" : !emitc.opaque<"size_t">
        %488 = call_opaque "__riscv_vsetvl_e8mf2"(%487) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %489 = literal "16" : !emitc.opaque<"size_t">
        %490 = add %466, %489 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %491 = add %256, %490 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
        %492 = add %6, %490 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2"
        %493 = call_opaque "__riscv_vle8_v_i8mf2"(%491, %488) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2"
        %494 = call_opaque "__riscv_vle8_v_i8mf2"(%492, %488) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1"
        %495 = call_opaque "__riscv_vwmul_vv_i16m1"(%493, %494, %488) : (!emitc.opaque<"vint8mf2_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2"
        %496 = load %250 : <!emitc.opaque<"vint32m2_t">>
        %497 = call_opaque "__riscv_vwmacc_vx_i32m2"(%496, %464, %495, %488) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"int">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        verbatim "// tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface"
        assign %497 : !emitc.opaque<"vint32m2_t"> to %250 : <!emitc.opaque<"vint32m2_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_quarter"
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2"
        %498 = literal "8" : !emitc.opaque<"size_t">
        %499 = call_opaque "__riscv_vsetvl_e8mf2"(%498) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %500 = literal "24" : !emitc.opaque<"size_t">
        %501 = add %466, %500 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %502 = add %256, %501 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
        %503 = add %6, %501 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2"
        %504 = call_opaque "__riscv_vle8_v_i8mf2"(%502, %499) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2"
        %505 = call_opaque "__riscv_vle8_v_i8mf2"(%503, %499) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1"
        %506 = call_opaque "__riscv_vwmul_vv_i16m1"(%504, %505, %499) : (!emitc.opaque<"vint8mf2_t">, !emitc.opaque<"vint8mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2"
        %507 = load %250 : <!emitc.opaque<"vint32m2_t">>
        %508 = call_opaque "__riscv_vwmacc_vx_i32m2"(%507, %464, %506, %499) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"int">, !emitc.opaque<"vint16m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
        verbatim "// tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface"
        assign %508 : !emitc.opaque<"vint32m2_t"> to %250 : <!emitc.opaque<"vint32m2_t">>
      }
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_activation_d"
      %260 = cast %62 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const float">>
      %261 = literal "0" : index
      %262 = subscript %260[%261] : (!emitc.ptr<!emitc.opaque<"const float">>, index) -> !emitc.lvalue<!emitc.opaque<"const float">>
      %263 = load %262 : <!emitc.opaque<"const float">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=min_term_bsums"
      %264 = literal "260" : !emitc.opaque<"size_t">
      %265 = add %62, %264 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %266 = cast %265 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int16_t">>
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %267 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int">>
      %268 = literal "0" : !emitc.opaque<"int">
      assign %268 : !emitc.opaque<"int"> to %267 : <!emitc.opaque<"int">>
      %269 = literal "0" : index
      %270 = subscript %266[%269] : (!emitc.ptr<!emitc.opaque<"const int16_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const int16_t">>
      %271 = load %270 : <!emitc.opaque<"const int16_t">>
      %272 = cast %271 : !emitc.opaque<"const int16_t"> to !emitc.opaque<"int">
      %273 = literal "8" : index
      %274 = subscript %249[%273] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %275 = load %274 : <!emitc.opaque<"const uint8_t">>
      %276 = cast %275 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"int">
      %277 = mul %272, %276 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      %278 = load %267 : <!emitc.opaque<"int">>
      %279 = add %278, %277 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %279 : !emitc.opaque<"int"> to %267 : <!emitc.opaque<"int">>
      %280 = literal "1" : index
      %281 = subscript %266[%280] : (!emitc.ptr<!emitc.opaque<"const int16_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const int16_t">>
      %282 = load %281 : <!emitc.opaque<"const int16_t">>
      %283 = cast %282 : !emitc.opaque<"const int16_t"> to !emitc.opaque<"int">
      %284 = literal "8" : index
      %285 = subscript %249[%284] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %286 = load %285 : <!emitc.opaque<"const uint8_t">>
      %287 = cast %286 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"int">
      %288 = mul %283, %287 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      %289 = load %267 : <!emitc.opaque<"int">>
      %290 = add %289, %288 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %290 : !emitc.opaque<"int"> to %267 : <!emitc.opaque<"int">>
      %291 = literal "2" : index
      %292 = subscript %266[%291] : (!emitc.ptr<!emitc.opaque<"const int16_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const int16_t">>
      %293 = load %292 : <!emitc.opaque<"const int16_t">>
      %294 = cast %293 : !emitc.opaque<"const int16_t"> to !emitc.opaque<"int">
      %295 = literal "9" : index
      %296 = subscript %249[%295] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %297 = load %296 : <!emitc.opaque<"const uint8_t">>
      %298 = cast %297 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"int">
      %299 = mul %294, %298 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      %300 = load %267 : <!emitc.opaque<"int">>
      %301 = add %300, %299 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %301 : !emitc.opaque<"int"> to %267 : <!emitc.opaque<"int">>
      %302 = literal "3" : index
      %303 = subscript %266[%302] : (!emitc.ptr<!emitc.opaque<"const int16_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const int16_t">>
      %304 = load %303 : <!emitc.opaque<"const int16_t">>
      %305 = cast %304 : !emitc.opaque<"const int16_t"> to !emitc.opaque<"int">
      %306 = literal "9" : index
      %307 = subscript %249[%306] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %308 = load %307 : <!emitc.opaque<"const uint8_t">>
      %309 = cast %308 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"int">
      %310 = mul %305, %309 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      %311 = load %267 : <!emitc.opaque<"int">>
      %312 = add %311, %310 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %312 : !emitc.opaque<"int"> to %267 : <!emitc.opaque<"int">>
      %313 = literal "4" : index
      %314 = subscript %266[%313] : (!emitc.ptr<!emitc.opaque<"const int16_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const int16_t">>
      %315 = load %314 : <!emitc.opaque<"const int16_t">>
      %316 = cast %315 : !emitc.opaque<"const int16_t"> to !emitc.opaque<"int">
      %317 = literal "10" : index
      %318 = subscript %249[%317] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %319 = load %318 : <!emitc.opaque<"const uint8_t">>
      %320 = cast %319 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"int">
      %321 = mul %316, %320 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      %322 = load %267 : <!emitc.opaque<"int">>
      %323 = add %322, %321 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %323 : !emitc.opaque<"int"> to %267 : <!emitc.opaque<"int">>
      %324 = literal "5" : index
      %325 = subscript %266[%324] : (!emitc.ptr<!emitc.opaque<"const int16_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const int16_t">>
      %326 = load %325 : <!emitc.opaque<"const int16_t">>
      %327 = cast %326 : !emitc.opaque<"const int16_t"> to !emitc.opaque<"int">
      %328 = literal "10" : index
      %329 = subscript %249[%328] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %330 = load %329 : <!emitc.opaque<"const uint8_t">>
      %331 = cast %330 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"int">
      %332 = mul %327, %331 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      %333 = load %267 : <!emitc.opaque<"int">>
      %334 = add %333, %332 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %334 : !emitc.opaque<"int"> to %267 : <!emitc.opaque<"int">>
      %335 = literal "6" : index
      %336 = subscript %266[%335] : (!emitc.ptr<!emitc.opaque<"const int16_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const int16_t">>
      %337 = load %336 : <!emitc.opaque<"const int16_t">>
      %338 = cast %337 : !emitc.opaque<"const int16_t"> to !emitc.opaque<"int">
      %339 = literal "11" : index
      %340 = subscript %249[%339] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %341 = load %340 : <!emitc.opaque<"const uint8_t">>
      %342 = cast %341 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"int">
      %343 = mul %338, %342 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      %344 = load %267 : <!emitc.opaque<"int">>
      %345 = add %344, %343 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %345 : !emitc.opaque<"int"> to %267 : <!emitc.opaque<"int">>
      %346 = literal "7" : index
      %347 = subscript %266[%346] : (!emitc.ptr<!emitc.opaque<"const int16_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const int16_t">>
      %348 = load %347 : <!emitc.opaque<"const int16_t">>
      %349 = cast %348 : !emitc.opaque<"const int16_t"> to !emitc.opaque<"int">
      %350 = literal "11" : index
      %351 = subscript %249[%350] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %352 = load %351 : <!emitc.opaque<"const uint8_t">>
      %353 = cast %352 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"int">
      %354 = mul %349, %353 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      %355 = load %267 : <!emitc.opaque<"int">>
      %356 = add %355, %354 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %356 : !emitc.opaque<"int"> to %267 : <!emitc.opaque<"int">>
      %357 = literal "8" : index
      %358 = subscript %266[%357] : (!emitc.ptr<!emitc.opaque<"const int16_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const int16_t">>
      %359 = load %358 : <!emitc.opaque<"const int16_t">>
      %360 = cast %359 : !emitc.opaque<"const int16_t"> to !emitc.opaque<"int">
      %361 = literal "12" : index
      %362 = subscript %249[%361] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %363 = load %362 : <!emitc.opaque<"const uint8_t">>
      %364 = cast %363 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"int">
      %365 = mul %360, %364 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      %366 = load %267 : <!emitc.opaque<"int">>
      %367 = add %366, %365 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %367 : !emitc.opaque<"int"> to %267 : <!emitc.opaque<"int">>
      %368 = literal "9" : index
      %369 = subscript %266[%368] : (!emitc.ptr<!emitc.opaque<"const int16_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const int16_t">>
      %370 = load %369 : <!emitc.opaque<"const int16_t">>
      %371 = cast %370 : !emitc.opaque<"const int16_t"> to !emitc.opaque<"int">
      %372 = literal "12" : index
      %373 = subscript %249[%372] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %374 = load %373 : <!emitc.opaque<"const uint8_t">>
      %375 = cast %374 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"int">
      %376 = mul %371, %375 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      %377 = load %267 : <!emitc.opaque<"int">>
      %378 = add %377, %376 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %378 : !emitc.opaque<"int"> to %267 : <!emitc.opaque<"int">>
      %379 = literal "10" : index
      %380 = subscript %266[%379] : (!emitc.ptr<!emitc.opaque<"const int16_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const int16_t">>
      %381 = load %380 : <!emitc.opaque<"const int16_t">>
      %382 = cast %381 : !emitc.opaque<"const int16_t"> to !emitc.opaque<"int">
      %383 = literal "13" : index
      %384 = subscript %249[%383] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %385 = load %384 : <!emitc.opaque<"const uint8_t">>
      %386 = cast %385 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"int">
      %387 = mul %382, %386 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      %388 = load %267 : <!emitc.opaque<"int">>
      %389 = add %388, %387 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %389 : !emitc.opaque<"int"> to %267 : <!emitc.opaque<"int">>
      %390 = literal "11" : index
      %391 = subscript %266[%390] : (!emitc.ptr<!emitc.opaque<"const int16_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const int16_t">>
      %392 = load %391 : <!emitc.opaque<"const int16_t">>
      %393 = cast %392 : !emitc.opaque<"const int16_t"> to !emitc.opaque<"int">
      %394 = literal "13" : index
      %395 = subscript %249[%394] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %396 = load %395 : <!emitc.opaque<"const uint8_t">>
      %397 = cast %396 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"int">
      %398 = mul %393, %397 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      %399 = load %267 : <!emitc.opaque<"int">>
      %400 = add %399, %398 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %400 : !emitc.opaque<"int"> to %267 : <!emitc.opaque<"int">>
      %401 = literal "12" : index
      %402 = subscript %266[%401] : (!emitc.ptr<!emitc.opaque<"const int16_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const int16_t">>
      %403 = load %402 : <!emitc.opaque<"const int16_t">>
      %404 = cast %403 : !emitc.opaque<"const int16_t"> to !emitc.opaque<"int">
      %405 = literal "14" : index
      %406 = subscript %249[%405] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %407 = load %406 : <!emitc.opaque<"const uint8_t">>
      %408 = cast %407 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"int">
      %409 = mul %404, %408 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      %410 = load %267 : <!emitc.opaque<"int">>
      %411 = add %410, %409 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %411 : !emitc.opaque<"int"> to %267 : <!emitc.opaque<"int">>
      %412 = literal "13" : index
      %413 = subscript %266[%412] : (!emitc.ptr<!emitc.opaque<"const int16_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const int16_t">>
      %414 = load %413 : <!emitc.opaque<"const int16_t">>
      %415 = cast %414 : !emitc.opaque<"const int16_t"> to !emitc.opaque<"int">
      %416 = literal "14" : index
      %417 = subscript %249[%416] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %418 = load %417 : <!emitc.opaque<"const uint8_t">>
      %419 = cast %418 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"int">
      %420 = mul %415, %419 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      %421 = load %267 : <!emitc.opaque<"int">>
      %422 = add %421, %420 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %422 : !emitc.opaque<"int"> to %267 : <!emitc.opaque<"int">>
      %423 = literal "14" : index
      %424 = subscript %266[%423] : (!emitc.ptr<!emitc.opaque<"const int16_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const int16_t">>
      %425 = load %424 : <!emitc.opaque<"const int16_t">>
      %426 = cast %425 : !emitc.opaque<"const int16_t"> to !emitc.opaque<"int">
      %427 = literal "15" : index
      %428 = subscript %249[%427] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %429 = load %428 : <!emitc.opaque<"const uint8_t">>
      %430 = cast %429 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"int">
      %431 = mul %426, %430 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      %432 = load %267 : <!emitc.opaque<"int">>
      %433 = add %432, %431 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %433 : !emitc.opaque<"int"> to %267 : <!emitc.opaque<"int">>
      %434 = literal "15" : index
      %435 = subscript %266[%434] : (!emitc.ptr<!emitc.opaque<"const int16_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const int16_t">>
      %436 = load %435 : <!emitc.opaque<"const int16_t">>
      %437 = cast %436 : !emitc.opaque<"const int16_t"> to !emitc.opaque<"int">
      %438 = literal "15" : index
      %439 = subscript %249[%438] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %440 = load %439 : <!emitc.opaque<"const uint8_t">>
      %441 = cast %440 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"int">
      %442 = mul %437, %441 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      %443 = load %267 : <!emitc.opaque<"int">>
      %444 = add %443, %442 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %444 : !emitc.opaque<"int"> to %267 : <!emitc.opaque<"int">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_d"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %445 = call_opaque "(float)*(const _Float16 *)"(%59) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      %446 = mul %445, %263 : (!emitc.opaque<"float">, !emitc.opaque<"const float">) -> !emitc.opaque<"float">
      %447 = load %250 : <!emitc.opaque<"vint32m2_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2"
      %448 = literal "8" : !emitc.opaque<"size_t">
      %449 = call_opaque "__riscv_vfcvt_f_x_v_f32m2"(%447, %448) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2"
      %450 = literal "8" : !emitc.opaque<"size_t">
      %451 = call_opaque "__riscv_vfmul_vf_f32m2"(%449, %446, %450) : (!emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"float">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2"
      %452 = load %9 : <!emitc.opaque<"vfloat32m2_t">>
      %453 = literal "8" : !emitc.opaque<"size_t">
      %454 = call_opaque "__riscv_vfadd_vv_f32m2"(%452, %451, %453) : (!emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vfloat32m2_t">
      verbatim "// tcrv_emitc.assign target=sums source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %454 : !emitc.opaque<"vfloat32m2_t"> to %9 : <!emitc.opaque<"vfloat32m2_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_dmin"
      %455 = literal "2" : !emitc.opaque<"size_t">
      %456 = add %59, %455 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %457 = call_opaque "(float)*(const _Float16 *)"(%456) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      %458 = mul %457, %263 : (!emitc.opaque<"float">, !emitc.opaque<"const float">) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=min_subtract"
      %459 = load %267 : <!emitc.opaque<"int">>
      %460 = load %13 : <!emitc.opaque<"float">>
      %461 = expression : !emitc.opaque<"float"> {
        %462 = cast %459 : !emitc.opaque<"int"> to !emitc.opaque<"float">
        %463 = mul %458, %462 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %464 = sub %460, %463 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %464 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %461 : !emitc.opaque<"float"> to %13 : <!emitc.opaque<"float">>
    }
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_sums_lanes"
    %17 = literal "0" : index
    %18 = subscript %8[%17] : (!emitc.array<8x!emitc.opaque<"float">>, index) -> !emitc.lvalue<!emitc.opaque<"float">>
    %19 = apply "&"(%18) : (!emitc.lvalue<!emitc.opaque<"float">>) -> !emitc.ptr<!emitc.opaque<"float">>
    %20 = load %9 : <!emitc.opaque<"vfloat32m2_t">>
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2"
    %21 = literal "8" : !emitc.opaque<"size_t">
    call_opaque "__riscv_vse32_v_f32m2"(%19, %20, %21) : (!emitc.ptr<!emitc.opaque<"float">>, !emitc.opaque<"vfloat32m2_t">, !emitc.opaque<"size_t">) -> ()
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=horizontal_sum"
    %22 = load %13 : <!emitc.opaque<"float">>
    %23 = literal "0" : index
    %24 = subscript %8[%23] : (!emitc.array<8x!emitc.opaque<"float">>, index) -> !emitc.lvalue<!emitc.opaque<"float">>
    %25 = load %24 : <!emitc.opaque<"float">>
    %26 = add %22, %25 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
    %27 = literal "1" : index
    %28 = subscript %8[%27] : (!emitc.array<8x!emitc.opaque<"float">>, index) -> !emitc.lvalue<!emitc.opaque<"float">>
    %29 = load %28 : <!emitc.opaque<"float">>
    %30 = add %26, %29 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
    %31 = literal "2" : index
    %32 = subscript %8[%31] : (!emitc.array<8x!emitc.opaque<"float">>, index) -> !emitc.lvalue<!emitc.opaque<"float">>
    %33 = load %32 : <!emitc.opaque<"float">>
    %34 = add %30, %33 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
    %35 = literal "3" : index
    %36 = subscript %8[%35] : (!emitc.array<8x!emitc.opaque<"float">>, index) -> !emitc.lvalue<!emitc.opaque<"float">>
    %37 = load %36 : <!emitc.opaque<"float">>
    %38 = add %34, %37 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
    %39 = literal "4" : index
    %40 = subscript %8[%39] : (!emitc.array<8x!emitc.opaque<"float">>, index) -> !emitc.lvalue<!emitc.opaque<"float">>
    %41 = load %40 : <!emitc.opaque<"float">>
    %42 = add %38, %41 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
    %43 = literal "5" : index
    %44 = subscript %8[%43] : (!emitc.array<8x!emitc.opaque<"float">>, index) -> !emitc.lvalue<!emitc.opaque<"float">>
    %45 = load %44 : <!emitc.opaque<"float">>
    %46 = add %42, %45 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
    %47 = literal "6" : index
    %48 = subscript %8[%47] : (!emitc.array<8x!emitc.opaque<"float">>, index) -> !emitc.lvalue<!emitc.opaque<"float">>
    %49 = load %48 : <!emitc.opaque<"float">>
    %50 = add %46, %49 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
    %51 = literal "7" : index
    %52 = subscript %8[%51] : (!emitc.array<8x!emitc.opaque<"float">>, index) -> !emitc.lvalue<!emitc.opaque<"float">>
    %53 = load %52 : <!emitc.opaque<"float">>
    %54 = add %50, %53 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s"
    %55 = literal "0" : index
    %56 = subscript %arg1[%55] : (!emitc.ptr<!emitc.opaque<"float">>, index) -> !emitc.lvalue<!emitc.opaque<"float">>
    assign %54 : !emitc.opaque<"float"> to %56 : <!emitc.opaque<"float">>
    return
  }
}

