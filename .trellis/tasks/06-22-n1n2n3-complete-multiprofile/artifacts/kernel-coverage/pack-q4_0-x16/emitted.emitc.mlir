module {
  emitc.include <"stddef.h">
  emitc.include <"stdint.h">
  emitc.include <"riscv_vector.h">
  emitc.func @tcrv_emitc_ggml_pack_q4_0_to_q4_0x16_kernel_ggml_pack_q4_0_to_q4_0x16(%arg0: !emitc.opaque<"size_t">, %arg1: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg2: !emitc.ptr<!emitc.opaque<"uint8_t">>) attributes {specifiers = ["extern", "\22C\22"]} {
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1"
    %0 = call_opaque "__riscv_vsetvl_e32m1"(%arg0) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.pack_q4_0_to_q4_0x16 role=compute op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.pack_q4_0_to_q4_0x16 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=pack_block_loop"
    %1 = literal "1" : !emitc.opaque<"size_t">
    %2 = literal "0" : !emitc.opaque<"size_t">
    for %arg3 = %2 to %arg0 step %1  : !emitc.opaque<"size_t"> {
      %4 = literal "288" : !emitc.opaque<"size_t">
      %5 = mul %arg3, %4 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %6 = literal "288" : !emitc.opaque<"size_t">
      %7 = mul %arg3, %6 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.pack_q4_0_to_q4_0x16 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=pack_scales"
      %8 = literal "1" : !emitc.opaque<"size_t">
      %9 = literal "16" : !emitc.opaque<"size_t">
      %10 = literal "0" : !emitc.opaque<"size_t">
      for %arg4 = %10 to %9 step %8  : !emitc.opaque<"size_t"> {
        %17 = literal "18" : !emitc.opaque<"size_t">
        %18 = mul %arg4, %17 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %19 = add %5, %18 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %20 = literal "2" : !emitc.opaque<"size_t">
        %21 = mul %arg4, %20 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %22 = add %7, %21 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %23 = literal "0" : !emitc.opaque<"size_t">
        %24 = add %19, %23 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %25 = subscript %arg1[%24] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
        %26 = load %25 : <!emitc.opaque<"const uint8_t">>
        %27 = cast %26 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"uint8_t">
        %28 = add %22, %23 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %29 = subscript %arg2[%28] : (!emitc.ptr<!emitc.opaque<"uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.lvalue<!emitc.opaque<"uint8_t">>
        assign %27 : !emitc.opaque<"uint8_t"> to %29 : <!emitc.opaque<"uint8_t">>
        %30 = literal "1" : !emitc.opaque<"size_t">
        %31 = add %19, %30 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %32 = subscript %arg1[%31] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
        %33 = load %32 : <!emitc.opaque<"const uint8_t">>
        %34 = cast %33 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"uint8_t">
        %35 = add %22, %30 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %36 = subscript %arg2[%35] : (!emitc.ptr<!emitc.opaque<"uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.lvalue<!emitc.opaque<"uint8_t">>
        assign %34 : !emitc.opaque<"uint8_t"> to %36 : <!emitc.opaque<"uint8_t">>
      }
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.pack_q4_0_to_q4_0x16 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=pack_quants_xor"
      %11 = literal "136" : !emitc.opaque<"uint8_t">
      %12 = literal "32" : !emitc.opaque<"size_t">
      %13 = add %7, %12 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %14 = literal "1" : !emitc.opaque<"size_t">
      %15 = literal "16" : !emitc.opaque<"size_t">
      %16 = literal "0" : !emitc.opaque<"size_t">
      for %arg4 = %16 to %15 step %14  : !emitc.opaque<"size_t"> {
        %17 = literal "16" : !emitc.opaque<"size_t">
        %18 = mul %arg4, %17 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %19 = add %13, %18 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %20 = literal "2" : !emitc.opaque<"size_t">
        %21 = add %20, %arg4 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %22 = literal "1" : !emitc.opaque<"size_t">
        %23 = literal "16" : !emitc.opaque<"size_t">
        %24 = literal "0" : !emitc.opaque<"size_t">
        for %arg5 = %24 to %23 step %22  : !emitc.opaque<"size_t"> {
          %25 = literal "18" : !emitc.opaque<"size_t">
          %26 = mul %arg5, %25 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %27 = add %5, %26 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %28 = add %27, %21 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %29 = add %19, %arg5 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
          %30 = subscript %arg1[%28] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
          %31 = load %30 : <!emitc.opaque<"const uint8_t">>
          %32 = cast %31 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"uint8_t">
          %33 = bitwise_xor %32, %11 : (!emitc.opaque<"uint8_t">, !emitc.opaque<"uint8_t">) -> !emitc.opaque<"uint8_t">
          %34 = subscript %arg2[%29] : (!emitc.ptr<!emitc.opaque<"uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.lvalue<!emitc.opaque<"uint8_t">>
          assign %33 : !emitc.opaque<"uint8_t"> to %34 : <!emitc.opaque<"uint8_t">>
        }
      }
    }
    %3 = literal "0" : !emitc.opaque<"size_t">
    return
  }
}

