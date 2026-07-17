module {
  emitc.include <"stddef.h">
  emitc.include <"stdint.h">
  emitc.include <"riscv_vector.h">
  emitc.func @tcrv_emitc_ggml_vec_dot_tq2_0_q8_K_kernel_ggml_vec_dot_tq2_0_q8_K(%arg0: !emitc.opaque<"size_t">, %arg1: !emitc.ptr<!emitc.opaque<"float">>, %arg2: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg3: !emitc.ptr<!emitc.opaque<"const uint8_t">>) attributes {specifiers = ["extern", "\22C\22"]} {
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1"
    %0 = call_opaque "__riscv_vsetvl_e32m1"(%arg0) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count"
    %1 = literal "256" : !emitc.opaque<"size_t">
    %2 = div %arg0, %1 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.local_variable=aux8 source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
    %3 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<256x!emitc.opaque<"int8_t">>
    %4 = literal "0" : index
    %5 = subscript %3[%4] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
    %6 = apply "&"(%5) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
    verbatim "// tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
    %7 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"float">>
    %8 = literal "0.0f" : !emitc.opaque<"float">
    assign %8 : !emitc.opaque<"float"> to %7 : <!emitc.opaque<"float">>
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_loop"
    %9 = literal "1" : !emitc.opaque<"size_t">
    %10 = literal "0" : !emitc.opaque<"size_t">
    for %arg4 = %10 to %2 step %9  : !emitc.opaque<"size_t"> {
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x"
      %14 = literal "66" : !emitc.opaque<"size_t">
      %15 = mul %arg4, %14 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %16 = add %arg2, %15 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_y"
      %17 = literal "292" : !emitc.opaque<"size_t">
      %18 = mul %arg4, %17 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %19 = add %arg3, %18 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=unpack_2bit_ternary"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2"
      %20 = literal "32" : !emitc.opaque<"size_t">
      %21 = call_opaque "__riscv_vsetvl_e8m2"(%20) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %22 = cast %16 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2"
      %23 = call_opaque "__riscv_vle8_v_u8m2"(%22, %21) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2"
      %24 = literal "0x03" : !emitc.opaque<"int">
      %25 = call_opaque "__riscv_vand_vx_u8m2"(%23, %24, %21) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2"
      %26 = call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"(%25) : (!emitc.opaque<"vuint8m2_t">) -> !emitc.opaque<"vint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2"
      %27 = literal "-1" : !emitc.opaque<"int">
      %28 = call_opaque "__riscv_vadd_vx_i8m2"(%26, %27, %21) : (!emitc.opaque<"vint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m2_t">
      %29 = literal "0" : index
      %30 = subscript %3[%29] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %31 = apply "&"(%30) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2"
      call_opaque "__riscv_vse8_v_i8m2"(%31, %28, %21) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2"
      %32 = literal "2" : !emitc.opaque<"int">
      %33 = call_opaque "__riscv_vsrl_vx_u8m2"(%23, %32, %21) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2"
      %34 = literal "0x03" : !emitc.opaque<"int">
      %35 = call_opaque "__riscv_vand_vx_u8m2"(%33, %34, %21) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2"
      %36 = call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"(%35) : (!emitc.opaque<"vuint8m2_t">) -> !emitc.opaque<"vint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2"
      %37 = literal "-1" : !emitc.opaque<"int">
      %38 = call_opaque "__riscv_vadd_vx_i8m2"(%36, %37, %21) : (!emitc.opaque<"vint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m2_t">
      %39 = literal "32" : index
      %40 = subscript %3[%39] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %41 = apply "&"(%40) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2"
      call_opaque "__riscv_vse8_v_i8m2"(%41, %38, %21) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2"
      %42 = literal "4" : !emitc.opaque<"int">
      %43 = call_opaque "__riscv_vsrl_vx_u8m2"(%23, %42, %21) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2"
      %44 = literal "0x03" : !emitc.opaque<"int">
      %45 = call_opaque "__riscv_vand_vx_u8m2"(%43, %44, %21) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2"
      %46 = call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"(%45) : (!emitc.opaque<"vuint8m2_t">) -> !emitc.opaque<"vint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2"
      %47 = literal "-1" : !emitc.opaque<"int">
      %48 = call_opaque "__riscv_vadd_vx_i8m2"(%46, %47, %21) : (!emitc.opaque<"vint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m2_t">
      %49 = literal "64" : index
      %50 = subscript %3[%49] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %51 = apply "&"(%50) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2"
      call_opaque "__riscv_vse8_v_i8m2"(%51, %48, %21) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2"
      %52 = literal "6" : !emitc.opaque<"int">
      %53 = call_opaque "__riscv_vsrl_vx_u8m2"(%23, %52, %21) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2"
      %54 = literal "0x03" : !emitc.opaque<"int">
      %55 = call_opaque "__riscv_vand_vx_u8m2"(%53, %54, %21) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2"
      %56 = call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"(%55) : (!emitc.opaque<"vuint8m2_t">) -> !emitc.opaque<"vint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2"
      %57 = literal "-1" : !emitc.opaque<"int">
      %58 = call_opaque "__riscv_vadd_vx_i8m2"(%56, %57, %21) : (!emitc.opaque<"vint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m2_t">
      %59 = literal "96" : index
      %60 = subscript %3[%59] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %61 = apply "&"(%60) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2"
      call_opaque "__riscv_vse8_v_i8m2"(%61, %58, %21) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2"
      %62 = literal "32" : !emitc.opaque<"size_t">
      %63 = call_opaque "__riscv_vsetvl_e8m2"(%62) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %64 = literal "32" : !emitc.opaque<"size_t">
      %65 = add %16, %64 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %66 = cast %65 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2"
      %67 = call_opaque "__riscv_vle8_v_u8m2"(%66, %63) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2"
      %68 = literal "0x03" : !emitc.opaque<"int">
      %69 = call_opaque "__riscv_vand_vx_u8m2"(%67, %68, %63) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2"
      %70 = call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"(%69) : (!emitc.opaque<"vuint8m2_t">) -> !emitc.opaque<"vint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2"
      %71 = literal "-1" : !emitc.opaque<"int">
      %72 = call_opaque "__riscv_vadd_vx_i8m2"(%70, %71, %63) : (!emitc.opaque<"vint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m2_t">
      %73 = literal "128" : index
      %74 = subscript %3[%73] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %75 = apply "&"(%74) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2"
      call_opaque "__riscv_vse8_v_i8m2"(%75, %72, %63) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2"
      %76 = literal "2" : !emitc.opaque<"int">
      %77 = call_opaque "__riscv_vsrl_vx_u8m2"(%67, %76, %63) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2"
      %78 = literal "0x03" : !emitc.opaque<"int">
      %79 = call_opaque "__riscv_vand_vx_u8m2"(%77, %78, %63) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2"
      %80 = call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"(%79) : (!emitc.opaque<"vuint8m2_t">) -> !emitc.opaque<"vint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2"
      %81 = literal "-1" : !emitc.opaque<"int">
      %82 = call_opaque "__riscv_vadd_vx_i8m2"(%80, %81, %63) : (!emitc.opaque<"vint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m2_t">
      %83 = literal "160" : index
      %84 = subscript %3[%83] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %85 = apply "&"(%84) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2"
      call_opaque "__riscv_vse8_v_i8m2"(%85, %82, %63) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2"
      %86 = literal "4" : !emitc.opaque<"int">
      %87 = call_opaque "__riscv_vsrl_vx_u8m2"(%67, %86, %63) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2"
      %88 = literal "0x03" : !emitc.opaque<"int">
      %89 = call_opaque "__riscv_vand_vx_u8m2"(%87, %88, %63) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2"
      %90 = call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"(%89) : (!emitc.opaque<"vuint8m2_t">) -> !emitc.opaque<"vint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2"
      %91 = literal "-1" : !emitc.opaque<"int">
      %92 = call_opaque "__riscv_vadd_vx_i8m2"(%90, %91, %63) : (!emitc.opaque<"vint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m2_t">
      %93 = literal "192" : index
      %94 = subscript %3[%93] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %95 = apply "&"(%94) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2"
      call_opaque "__riscv_vse8_v_i8m2"(%95, %92, %63) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> ()
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2"
      %96 = literal "6" : !emitc.opaque<"int">
      %97 = call_opaque "__riscv_vsrl_vx_u8m2"(%67, %96, %63) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2"
      %98 = literal "0x03" : !emitc.opaque<"int">
      %99 = call_opaque "__riscv_vand_vx_u8m2"(%97, %98, %63) : (!emitc.opaque<"vuint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2"
      %100 = call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"(%99) : (!emitc.opaque<"vuint8m2_t">) -> !emitc.opaque<"vint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2"
      %101 = literal "-1" : !emitc.opaque<"int">
      %102 = call_opaque "__riscv_vadd_vx_i8m2"(%100, %101, %63) : (!emitc.opaque<"vint8m2_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m2_t">
      %103 = literal "224" : index
      %104 = subscript %3[%103] : (!emitc.array<256x!emitc.opaque<"int8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"int8_t">>
      %105 = apply "&"(%104) : (!emitc.lvalue<!emitc.opaque<"int8_t">>) -> !emitc.ptr<!emitc.opaque<"int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2"
      call_opaque "__riscv_vse8_v_i8m2"(%105, %102, %63) : (!emitc.ptr<!emitc.opaque<"int8_t">>, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> ()
      %106 = literal "4" : !emitc.opaque<"size_t">
      %107 = add %19, %106 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %108 = cast %107 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %109 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int">>
      %110 = literal "0" : !emitc.opaque<"int">
      assign %110 : !emitc.opaque<"int"> to %109 : <!emitc.opaque<"int">>
      %111 = literal "0" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %112 = literal "16" : !emitc.opaque<"size_t">
      %113 = call_opaque "__riscv_vsetvl_e8m1"(%112) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %114 = literal "16" : !emitc.opaque<"size_t">
      %115 = mul %111, %114 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %116 = add %108, %115 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %117 = add %6, %115 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %118 = call_opaque "__riscv_vle8_v_i8m1"(%116, %113) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %119 = call_opaque "__riscv_vle8_v_i8m1"(%117, %113) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %120 = call_opaque "__riscv_vwmul_vv_i16m2"(%118, %119, %113) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %121 = literal "0" : !emitc.opaque<"int">
      %122 = literal "1" : !emitc.opaque<"size_t">
      %123 = call_opaque "__riscv_vmv_v_x_i32m1"(%121, %122) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %124 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%120, %123, %113) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %125 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%124) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %126 = load %109 : <!emitc.opaque<"int">>
      %127 = add %126, %125 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %127 : !emitc.opaque<"int"> to %109 : <!emitc.opaque<"int">>
      %128 = literal "1" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %129 = literal "16" : !emitc.opaque<"size_t">
      %130 = call_opaque "__riscv_vsetvl_e8m1"(%129) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %131 = literal "16" : !emitc.opaque<"size_t">
      %132 = mul %128, %131 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %133 = add %108, %132 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %134 = add %6, %132 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %135 = call_opaque "__riscv_vle8_v_i8m1"(%133, %130) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %136 = call_opaque "__riscv_vle8_v_i8m1"(%134, %130) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %137 = call_opaque "__riscv_vwmul_vv_i16m2"(%135, %136, %130) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %138 = literal "0" : !emitc.opaque<"int">
      %139 = literal "1" : !emitc.opaque<"size_t">
      %140 = call_opaque "__riscv_vmv_v_x_i32m1"(%138, %139) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %141 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%137, %140, %130) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %142 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%141) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %143 = load %109 : <!emitc.opaque<"int">>
      %144 = add %143, %142 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %144 : !emitc.opaque<"int"> to %109 : <!emitc.opaque<"int">>
      %145 = literal "2" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %146 = literal "16" : !emitc.opaque<"size_t">
      %147 = call_opaque "__riscv_vsetvl_e8m1"(%146) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %148 = literal "16" : !emitc.opaque<"size_t">
      %149 = mul %145, %148 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %150 = add %108, %149 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %151 = add %6, %149 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %152 = call_opaque "__riscv_vle8_v_i8m1"(%150, %147) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %153 = call_opaque "__riscv_vle8_v_i8m1"(%151, %147) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %154 = call_opaque "__riscv_vwmul_vv_i16m2"(%152, %153, %147) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %155 = literal "0" : !emitc.opaque<"int">
      %156 = literal "1" : !emitc.opaque<"size_t">
      %157 = call_opaque "__riscv_vmv_v_x_i32m1"(%155, %156) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %158 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%154, %157, %147) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %159 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%158) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %160 = load %109 : <!emitc.opaque<"int">>
      %161 = add %160, %159 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %161 : !emitc.opaque<"int"> to %109 : <!emitc.opaque<"int">>
      %162 = literal "3" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %163 = literal "16" : !emitc.opaque<"size_t">
      %164 = call_opaque "__riscv_vsetvl_e8m1"(%163) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %165 = literal "16" : !emitc.opaque<"size_t">
      %166 = mul %162, %165 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %167 = add %108, %166 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %168 = add %6, %166 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %169 = call_opaque "__riscv_vle8_v_i8m1"(%167, %164) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %170 = call_opaque "__riscv_vle8_v_i8m1"(%168, %164) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %171 = call_opaque "__riscv_vwmul_vv_i16m2"(%169, %170, %164) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %172 = literal "0" : !emitc.opaque<"int">
      %173 = literal "1" : !emitc.opaque<"size_t">
      %174 = call_opaque "__riscv_vmv_v_x_i32m1"(%172, %173) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %175 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%171, %174, %164) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %176 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%175) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %177 = load %109 : <!emitc.opaque<"int">>
      %178 = add %177, %176 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %178 : !emitc.opaque<"int"> to %109 : <!emitc.opaque<"int">>
      %179 = literal "4" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %180 = literal "16" : !emitc.opaque<"size_t">
      %181 = call_opaque "__riscv_vsetvl_e8m1"(%180) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %182 = literal "16" : !emitc.opaque<"size_t">
      %183 = mul %179, %182 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %184 = add %108, %183 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %185 = add %6, %183 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %186 = call_opaque "__riscv_vle8_v_i8m1"(%184, %181) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %187 = call_opaque "__riscv_vle8_v_i8m1"(%185, %181) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %188 = call_opaque "__riscv_vwmul_vv_i16m2"(%186, %187, %181) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %189 = literal "0" : !emitc.opaque<"int">
      %190 = literal "1" : !emitc.opaque<"size_t">
      %191 = call_opaque "__riscv_vmv_v_x_i32m1"(%189, %190) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %192 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%188, %191, %181) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %193 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%192) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %194 = load %109 : <!emitc.opaque<"int">>
      %195 = add %194, %193 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %195 : !emitc.opaque<"int"> to %109 : <!emitc.opaque<"int">>
      %196 = literal "5" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %197 = literal "16" : !emitc.opaque<"size_t">
      %198 = call_opaque "__riscv_vsetvl_e8m1"(%197) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %199 = literal "16" : !emitc.opaque<"size_t">
      %200 = mul %196, %199 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %201 = add %108, %200 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %202 = add %6, %200 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %203 = call_opaque "__riscv_vle8_v_i8m1"(%201, %198) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %204 = call_opaque "__riscv_vle8_v_i8m1"(%202, %198) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %205 = call_opaque "__riscv_vwmul_vv_i16m2"(%203, %204, %198) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %206 = literal "0" : !emitc.opaque<"int">
      %207 = literal "1" : !emitc.opaque<"size_t">
      %208 = call_opaque "__riscv_vmv_v_x_i32m1"(%206, %207) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %209 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%205, %208, %198) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %210 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%209) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %211 = load %109 : <!emitc.opaque<"int">>
      %212 = add %211, %210 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %212 : !emitc.opaque<"int"> to %109 : <!emitc.opaque<"int">>
      %213 = literal "6" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %214 = literal "16" : !emitc.opaque<"size_t">
      %215 = call_opaque "__riscv_vsetvl_e8m1"(%214) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %216 = literal "16" : !emitc.opaque<"size_t">
      %217 = mul %213, %216 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %218 = add %108, %217 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %219 = add %6, %217 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %220 = call_opaque "__riscv_vle8_v_i8m1"(%218, %215) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %221 = call_opaque "__riscv_vle8_v_i8m1"(%219, %215) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %222 = call_opaque "__riscv_vwmul_vv_i16m2"(%220, %221, %215) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %223 = literal "0" : !emitc.opaque<"int">
      %224 = literal "1" : !emitc.opaque<"size_t">
      %225 = call_opaque "__riscv_vmv_v_x_i32m1"(%223, %224) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %226 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%222, %225, %215) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %227 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%226) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %228 = load %109 : <!emitc.opaque<"int">>
      %229 = add %228, %227 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %229 : !emitc.opaque<"int"> to %109 : <!emitc.opaque<"int">>
      %230 = literal "7" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %231 = literal "16" : !emitc.opaque<"size_t">
      %232 = call_opaque "__riscv_vsetvl_e8m1"(%231) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %233 = literal "16" : !emitc.opaque<"size_t">
      %234 = mul %230, %233 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %235 = add %108, %234 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %236 = add %6, %234 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %237 = call_opaque "__riscv_vle8_v_i8m1"(%235, %232) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %238 = call_opaque "__riscv_vle8_v_i8m1"(%236, %232) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %239 = call_opaque "__riscv_vwmul_vv_i16m2"(%237, %238, %232) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %240 = literal "0" : !emitc.opaque<"int">
      %241 = literal "1" : !emitc.opaque<"size_t">
      %242 = call_opaque "__riscv_vmv_v_x_i32m1"(%240, %241) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %243 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%239, %242, %232) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %244 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%243) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %245 = load %109 : <!emitc.opaque<"int">>
      %246 = add %245, %244 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %246 : !emitc.opaque<"int"> to %109 : <!emitc.opaque<"int">>
      %247 = literal "8" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %248 = literal "16" : !emitc.opaque<"size_t">
      %249 = call_opaque "__riscv_vsetvl_e8m1"(%248) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %250 = literal "16" : !emitc.opaque<"size_t">
      %251 = mul %247, %250 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %252 = add %108, %251 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %253 = add %6, %251 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %254 = call_opaque "__riscv_vle8_v_i8m1"(%252, %249) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %255 = call_opaque "__riscv_vle8_v_i8m1"(%253, %249) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %256 = call_opaque "__riscv_vwmul_vv_i16m2"(%254, %255, %249) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %257 = literal "0" : !emitc.opaque<"int">
      %258 = literal "1" : !emitc.opaque<"size_t">
      %259 = call_opaque "__riscv_vmv_v_x_i32m1"(%257, %258) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %260 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%256, %259, %249) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %261 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%260) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %262 = load %109 : <!emitc.opaque<"int">>
      %263 = add %262, %261 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %263 : !emitc.opaque<"int"> to %109 : <!emitc.opaque<"int">>
      %264 = literal "9" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %265 = literal "16" : !emitc.opaque<"size_t">
      %266 = call_opaque "__riscv_vsetvl_e8m1"(%265) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %267 = literal "16" : !emitc.opaque<"size_t">
      %268 = mul %264, %267 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %269 = add %108, %268 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %270 = add %6, %268 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %271 = call_opaque "__riscv_vle8_v_i8m1"(%269, %266) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %272 = call_opaque "__riscv_vle8_v_i8m1"(%270, %266) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %273 = call_opaque "__riscv_vwmul_vv_i16m2"(%271, %272, %266) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %274 = literal "0" : !emitc.opaque<"int">
      %275 = literal "1" : !emitc.opaque<"size_t">
      %276 = call_opaque "__riscv_vmv_v_x_i32m1"(%274, %275) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %277 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%273, %276, %266) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %278 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%277) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %279 = load %109 : <!emitc.opaque<"int">>
      %280 = add %279, %278 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %280 : !emitc.opaque<"int"> to %109 : <!emitc.opaque<"int">>
      %281 = literal "10" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %282 = literal "16" : !emitc.opaque<"size_t">
      %283 = call_opaque "__riscv_vsetvl_e8m1"(%282) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %284 = literal "16" : !emitc.opaque<"size_t">
      %285 = mul %281, %284 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %286 = add %108, %285 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %287 = add %6, %285 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %288 = call_opaque "__riscv_vle8_v_i8m1"(%286, %283) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %289 = call_opaque "__riscv_vle8_v_i8m1"(%287, %283) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %290 = call_opaque "__riscv_vwmul_vv_i16m2"(%288, %289, %283) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %291 = literal "0" : !emitc.opaque<"int">
      %292 = literal "1" : !emitc.opaque<"size_t">
      %293 = call_opaque "__riscv_vmv_v_x_i32m1"(%291, %292) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %294 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%290, %293, %283) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %295 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%294) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %296 = load %109 : <!emitc.opaque<"int">>
      %297 = add %296, %295 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %297 : !emitc.opaque<"int"> to %109 : <!emitc.opaque<"int">>
      %298 = literal "11" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %299 = literal "16" : !emitc.opaque<"size_t">
      %300 = call_opaque "__riscv_vsetvl_e8m1"(%299) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %301 = literal "16" : !emitc.opaque<"size_t">
      %302 = mul %298, %301 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %303 = add %108, %302 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %304 = add %6, %302 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %305 = call_opaque "__riscv_vle8_v_i8m1"(%303, %300) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %306 = call_opaque "__riscv_vle8_v_i8m1"(%304, %300) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %307 = call_opaque "__riscv_vwmul_vv_i16m2"(%305, %306, %300) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %308 = literal "0" : !emitc.opaque<"int">
      %309 = literal "1" : !emitc.opaque<"size_t">
      %310 = call_opaque "__riscv_vmv_v_x_i32m1"(%308, %309) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %311 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%307, %310, %300) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %312 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%311) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %313 = load %109 : <!emitc.opaque<"int">>
      %314 = add %313, %312 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %314 : !emitc.opaque<"int"> to %109 : <!emitc.opaque<"int">>
      %315 = literal "12" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %316 = literal "16" : !emitc.opaque<"size_t">
      %317 = call_opaque "__riscv_vsetvl_e8m1"(%316) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %318 = literal "16" : !emitc.opaque<"size_t">
      %319 = mul %315, %318 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %320 = add %108, %319 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %321 = add %6, %319 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %322 = call_opaque "__riscv_vle8_v_i8m1"(%320, %317) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %323 = call_opaque "__riscv_vle8_v_i8m1"(%321, %317) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %324 = call_opaque "__riscv_vwmul_vv_i16m2"(%322, %323, %317) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %325 = literal "0" : !emitc.opaque<"int">
      %326 = literal "1" : !emitc.opaque<"size_t">
      %327 = call_opaque "__riscv_vmv_v_x_i32m1"(%325, %326) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %328 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%324, %327, %317) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %329 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%328) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %330 = load %109 : <!emitc.opaque<"int">>
      %331 = add %330, %329 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %331 : !emitc.opaque<"int"> to %109 : <!emitc.opaque<"int">>
      %332 = literal "13" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %333 = literal "16" : !emitc.opaque<"size_t">
      %334 = call_opaque "__riscv_vsetvl_e8m1"(%333) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %335 = literal "16" : !emitc.opaque<"size_t">
      %336 = mul %332, %335 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %337 = add %108, %336 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %338 = add %6, %336 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %339 = call_opaque "__riscv_vle8_v_i8m1"(%337, %334) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %340 = call_opaque "__riscv_vle8_v_i8m1"(%338, %334) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %341 = call_opaque "__riscv_vwmul_vv_i16m2"(%339, %340, %334) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %342 = literal "0" : !emitc.opaque<"int">
      %343 = literal "1" : !emitc.opaque<"size_t">
      %344 = call_opaque "__riscv_vmv_v_x_i32m1"(%342, %343) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %345 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%341, %344, %334) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %346 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%345) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %347 = load %109 : <!emitc.opaque<"int">>
      %348 = add %347, %346 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %348 : !emitc.opaque<"int"> to %109 : <!emitc.opaque<"int">>
      %349 = literal "14" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %350 = literal "16" : !emitc.opaque<"size_t">
      %351 = call_opaque "__riscv_vsetvl_e8m1"(%350) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %352 = literal "16" : !emitc.opaque<"size_t">
      %353 = mul %349, %352 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %354 = add %108, %353 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %355 = add %6, %353 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %356 = call_opaque "__riscv_vle8_v_i8m1"(%354, %351) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %357 = call_opaque "__riscv_vle8_v_i8m1"(%355, %351) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %358 = call_opaque "__riscv_vwmul_vv_i16m2"(%356, %357, %351) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %359 = literal "0" : !emitc.opaque<"int">
      %360 = literal "1" : !emitc.opaque<"size_t">
      %361 = call_opaque "__riscv_vmv_v_x_i32m1"(%359, %360) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %362 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%358, %361, %351) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %363 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%362) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %364 = load %109 : <!emitc.opaque<"int">>
      %365 = add %364, %363 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %365 : !emitc.opaque<"int"> to %109 : <!emitc.opaque<"int">>
      %366 = literal "15" : index
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %367 = literal "16" : !emitc.opaque<"size_t">
      %368 = call_opaque "__riscv_vsetvl_e8m1"(%367) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %369 = literal "16" : !emitc.opaque<"size_t">
      %370 = mul %366, %369 : (index, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %371 = add %108, %370 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      %372 = add %6, %370 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %373 = call_opaque "__riscv_vle8_v_i8m1"(%371, %368) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %374 = call_opaque "__riscv_vle8_v_i8m1"(%372, %368) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %375 = call_opaque "__riscv_vwmul_vv_i16m2"(%373, %374, %368) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %376 = literal "0" : !emitc.opaque<"int">
      %377 = literal "1" : !emitc.opaque<"size_t">
      %378 = call_opaque "__riscv_vmv_v_x_i32m1"(%376, %377) : (!emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %379 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%375, %378, %368) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %380 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%379) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate"
      %381 = load %109 : <!emitc.opaque<"int">>
      %382 = add %381, %380 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      assign %382 : !emitc.opaque<"int"> to %109 : <!emitc.opaque<"int">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_activation_d"
      %383 = cast %19 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const float">>
      %384 = literal "0" : index
      %385 = subscript %383[%384] : (!emitc.ptr<!emitc.opaque<"const float">>, index) -> !emitc.lvalue<!emitc.opaque<"const float">>
      %386 = load %385 : <!emitc.opaque<"const float">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_d"
      %387 = literal "64" : !emitc.opaque<"size_t">
      %388 = add %16, %387 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %389 = call_opaque "(float)*(const _Float16 *)"(%388) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      %390 = mul %386, %389 : (!emitc.opaque<"const float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scalar_fold"
      %391 = load %109 : <!emitc.opaque<"int">>
      %392 = load %7 : <!emitc.opaque<"float">>
      %393 = expression : !emitc.opaque<"float"> {
        %394 = cast %391 : !emitc.opaque<"int"> to !emitc.opaque<"float">
        %395 = mul %394, %390 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %396 = add %392, %395 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %396 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %393 : !emitc.opaque<"float"> to %7 : <!emitc.opaque<"float">>
    }
    %11 = load %7 : <!emitc.opaque<"float">>
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s"
    %12 = literal "0" : index
    %13 = subscript %arg1[%12] : (!emitc.ptr<!emitc.opaque<"float">>, index) -> !emitc.lvalue<!emitc.opaque<"float">>
    assign %11 : !emitc.opaque<"float"> to %13 : <!emitc.opaque<"float">>
    return
  }
}

