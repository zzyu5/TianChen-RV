module {
  emitc.include <"stddef.h">
  emitc.include <"stdint.h">
  emitc.include <"riscv_vector.h">
  emitc.func @tcrv_emitc_ggml_vec_dot_q1_0_q8_0_kernel_ggml_vec_dot_q1_0_q8_0(%arg0: !emitc.opaque<"size_t">, %arg1: !emitc.ptr<!emitc.opaque<"float">>, %arg2: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg3: !emitc.ptr<!emitc.opaque<"const uint8_t">>) attributes {specifiers = ["extern", "\22C\22"]} {
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1"
    %0 = call_opaque "__riscv_vsetvl_e32m1"(%arg0) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "static const uint8_t tcrv_q1_0_kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};"
    verbatim "// tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
    %1 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"float">>
    %2 = literal "0.0f" : !emitc.opaque<"float">
    assign %2 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count"
    %3 = literal "128" : !emitc.opaque<"size_t">
    %4 = div %arg0, %3 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=kmask_table_load"
    %5 = literal "tcrv_q1_0_kmask" : !emitc.ptr<!emitc.opaque<"const uint8_t">>
    %6 = literal "8" : !emitc.opaque<"size_t">
    %7 = call_opaque "__riscv_vle8_v_u8m1"(%5, %6) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
    %8 = literal "1" : !emitc.opaque<"size_t">
    %9 = literal "0" : !emitc.opaque<"size_t">
    for %arg4 = %9 to %4 step %8  : !emitc.opaque<"size_t"> {
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x"
      %13 = literal "18" : !emitc.opaque<"size_t">
      %14 = mul %arg4, %13 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %15 = add %arg2, %14 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %16 = call_opaque "(float)*(const _Float16 *)"(%15) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %17 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"float">>
      %18 = literal "0.0f" : !emitc.opaque<"float">
      assign %18 : !emitc.opaque<"float"> to %17 : <!emitc.opaque<"float">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %19 = literal "4" : !emitc.opaque<"size_t">
      %20 = mul %arg4, %19 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %21 = literal "0" : !emitc.opaque<"size_t">
      %22 = add %20, %21 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %23 = literal "34" : !emitc.opaque<"size_t">
      %24 = mul %22, %23 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %25 = add %arg3, %24 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %26 = call_opaque "(float)*(const _Float16 *)"(%25) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.local_variable=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %27 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %28 = literal "0" : !emitc.opaque<"int32_t">
      assign %28 : !emitc.opaque<"int32_t"> to %27 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %29 = literal "8" : !emitc.opaque<"size_t">
      %30 = call_opaque "__riscv_vsetvl_e8m1"(%29) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read"
      %31 = literal "2" : !emitc.opaque<"size_t">
      %32 = add %15, %31 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %33 = cast %32 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %34 = literal "0" : index
      %35 = subscript %33[%34] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %36 = load %35 : <!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1"
      %37 = call_opaque "__riscv_vmv_v_x_u8m1"(%36, %30) : (!emitc.opaque<"const uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1"
      %38 = call_opaque "__riscv_vand_vv_u8m1"(%37, %7, %30) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8"
      %39 = literal "0" : !emitc.opaque<"int">
      %40 = call_opaque "__riscv_vmsne_vx_u8m1_b8"(%38, %39, %30) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool8_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr"
      %41 = literal "2" : !emitc.opaque<"size_t">
      %42 = add %25, %41 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %43 = cast %42 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %44 = call_opaque "__riscv_vle8_v_i8m1"(%43, %30) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2"
      %45 = call_opaque "__riscv_vwcvt_x_x_v_i16m2"(%44, %30) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2"
      %46 = call_opaque "__riscv_vneg_v_i16m2"(%45, %30) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2"
      %47 = call_opaque "__riscv_vmerge_vvm_i16m2"(%46, %45, %40, %30) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint16m2_t">, !emitc.opaque<"vbool8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %48 = load %27 : <!emitc.opaque<"int32_t">>
      %49 = literal "1" : !emitc.opaque<"size_t">
      %50 = call_opaque "__riscv_vmv_v_x_i32m1"(%48, %49) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %51 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%47, %50, %30) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %52 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%51) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %52 : !emitc.opaque<"int32_t"> to %27 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read"
      %53 = literal "3" : !emitc.opaque<"size_t">
      %54 = add %15, %53 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %55 = cast %54 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %56 = literal "0" : index
      %57 = subscript %55[%56] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %58 = load %57 : <!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1"
      %59 = call_opaque "__riscv_vmv_v_x_u8m1"(%58, %30) : (!emitc.opaque<"const uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1"
      %60 = call_opaque "__riscv_vand_vv_u8m1"(%59, %7, %30) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8"
      %61 = literal "0" : !emitc.opaque<"int">
      %62 = call_opaque "__riscv_vmsne_vx_u8m1_b8"(%60, %61, %30) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool8_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr"
      %63 = literal "10" : !emitc.opaque<"size_t">
      %64 = add %25, %63 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %65 = cast %64 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %66 = call_opaque "__riscv_vle8_v_i8m1"(%65, %30) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2"
      %67 = call_opaque "__riscv_vwcvt_x_x_v_i16m2"(%66, %30) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2"
      %68 = call_opaque "__riscv_vneg_v_i16m2"(%67, %30) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2"
      %69 = call_opaque "__riscv_vmerge_vvm_i16m2"(%68, %67, %62, %30) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint16m2_t">, !emitc.opaque<"vbool8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %70 = load %27 : <!emitc.opaque<"int32_t">>
      %71 = literal "1" : !emitc.opaque<"size_t">
      %72 = call_opaque "__riscv_vmv_v_x_i32m1"(%70, %71) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %73 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%69, %72, %30) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %74 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%73) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %74 : !emitc.opaque<"int32_t"> to %27 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read"
      %75 = literal "4" : !emitc.opaque<"size_t">
      %76 = add %15, %75 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %77 = cast %76 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %78 = literal "0" : index
      %79 = subscript %77[%78] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %80 = load %79 : <!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1"
      %81 = call_opaque "__riscv_vmv_v_x_u8m1"(%80, %30) : (!emitc.opaque<"const uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1"
      %82 = call_opaque "__riscv_vand_vv_u8m1"(%81, %7, %30) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8"
      %83 = literal "0" : !emitc.opaque<"int">
      %84 = call_opaque "__riscv_vmsne_vx_u8m1_b8"(%82, %83, %30) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool8_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr"
      %85 = literal "18" : !emitc.opaque<"size_t">
      %86 = add %25, %85 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %87 = cast %86 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %88 = call_opaque "__riscv_vle8_v_i8m1"(%87, %30) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2"
      %89 = call_opaque "__riscv_vwcvt_x_x_v_i16m2"(%88, %30) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2"
      %90 = call_opaque "__riscv_vneg_v_i16m2"(%89, %30) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2"
      %91 = call_opaque "__riscv_vmerge_vvm_i16m2"(%90, %89, %84, %30) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint16m2_t">, !emitc.opaque<"vbool8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %92 = load %27 : <!emitc.opaque<"int32_t">>
      %93 = literal "1" : !emitc.opaque<"size_t">
      %94 = call_opaque "__riscv_vmv_v_x_i32m1"(%92, %93) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %95 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%91, %94, %30) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %96 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%95) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %96 : !emitc.opaque<"int32_t"> to %27 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read"
      %97 = literal "5" : !emitc.opaque<"size_t">
      %98 = add %15, %97 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %99 = cast %98 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %100 = literal "0" : index
      %101 = subscript %99[%100] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %102 = load %101 : <!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1"
      %103 = call_opaque "__riscv_vmv_v_x_u8m1"(%102, %30) : (!emitc.opaque<"const uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1"
      %104 = call_opaque "__riscv_vand_vv_u8m1"(%103, %7, %30) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8"
      %105 = literal "0" : !emitc.opaque<"int">
      %106 = call_opaque "__riscv_vmsne_vx_u8m1_b8"(%104, %105, %30) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool8_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr"
      %107 = literal "26" : !emitc.opaque<"size_t">
      %108 = add %25, %107 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %109 = cast %108 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %110 = call_opaque "__riscv_vle8_v_i8m1"(%109, %30) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2"
      %111 = call_opaque "__riscv_vwcvt_x_x_v_i16m2"(%110, %30) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2"
      %112 = call_opaque "__riscv_vneg_v_i16m2"(%111, %30) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2"
      %113 = call_opaque "__riscv_vmerge_vvm_i16m2"(%112, %111, %106, %30) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint16m2_t">, !emitc.opaque<"vbool8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %114 = load %27 : <!emitc.opaque<"int32_t">>
      %115 = literal "1" : !emitc.opaque<"size_t">
      %116 = call_opaque "__riscv_vmv_v_x_i32m1"(%114, %115) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %117 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%113, %116, %30) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %118 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%117) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %118 : !emitc.opaque<"int32_t"> to %27 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate_sub"
      %119 = load %27 : <!emitc.opaque<"int32_t">>
      %120 = load %17 : <!emitc.opaque<"float">>
      %121 = expression : !emitc.opaque<"float"> {
        %434 = cast %119 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %435 = mul %26, %434 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %436 = add %120, %435 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %436 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %121 : !emitc.opaque<"float"> to %17 : <!emitc.opaque<"float">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %122 = literal "4" : !emitc.opaque<"size_t">
      %123 = mul %arg4, %122 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %124 = literal "1" : !emitc.opaque<"size_t">
      %125 = add %123, %124 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %126 = literal "34" : !emitc.opaque<"size_t">
      %127 = mul %125, %126 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %128 = add %arg3, %127 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %129 = call_opaque "(float)*(const _Float16 *)"(%128) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.local_variable=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %130 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %131 = literal "0" : !emitc.opaque<"int32_t">
      assign %131 : !emitc.opaque<"int32_t"> to %130 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %132 = literal "8" : !emitc.opaque<"size_t">
      %133 = call_opaque "__riscv_vsetvl_e8m1"(%132) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read"
      %134 = literal "6" : !emitc.opaque<"size_t">
      %135 = add %15, %134 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %136 = cast %135 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %137 = literal "0" : index
      %138 = subscript %136[%137] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %139 = load %138 : <!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1"
      %140 = call_opaque "__riscv_vmv_v_x_u8m1"(%139, %133) : (!emitc.opaque<"const uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1"
      %141 = call_opaque "__riscv_vand_vv_u8m1"(%140, %7, %133) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8"
      %142 = literal "0" : !emitc.opaque<"int">
      %143 = call_opaque "__riscv_vmsne_vx_u8m1_b8"(%141, %142, %133) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool8_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr"
      %144 = literal "2" : !emitc.opaque<"size_t">
      %145 = add %128, %144 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %146 = cast %145 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %147 = call_opaque "__riscv_vle8_v_i8m1"(%146, %133) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2"
      %148 = call_opaque "__riscv_vwcvt_x_x_v_i16m2"(%147, %133) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2"
      %149 = call_opaque "__riscv_vneg_v_i16m2"(%148, %133) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2"
      %150 = call_opaque "__riscv_vmerge_vvm_i16m2"(%149, %148, %143, %133) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint16m2_t">, !emitc.opaque<"vbool8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %151 = load %130 : <!emitc.opaque<"int32_t">>
      %152 = literal "1" : !emitc.opaque<"size_t">
      %153 = call_opaque "__riscv_vmv_v_x_i32m1"(%151, %152) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %154 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%150, %153, %133) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %155 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%154) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %155 : !emitc.opaque<"int32_t"> to %130 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read"
      %156 = literal "7" : !emitc.opaque<"size_t">
      %157 = add %15, %156 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %158 = cast %157 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %159 = literal "0" : index
      %160 = subscript %158[%159] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %161 = load %160 : <!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1"
      %162 = call_opaque "__riscv_vmv_v_x_u8m1"(%161, %133) : (!emitc.opaque<"const uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1"
      %163 = call_opaque "__riscv_vand_vv_u8m1"(%162, %7, %133) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8"
      %164 = literal "0" : !emitc.opaque<"int">
      %165 = call_opaque "__riscv_vmsne_vx_u8m1_b8"(%163, %164, %133) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool8_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr"
      %166 = literal "10" : !emitc.opaque<"size_t">
      %167 = add %128, %166 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %168 = cast %167 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %169 = call_opaque "__riscv_vle8_v_i8m1"(%168, %133) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2"
      %170 = call_opaque "__riscv_vwcvt_x_x_v_i16m2"(%169, %133) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2"
      %171 = call_opaque "__riscv_vneg_v_i16m2"(%170, %133) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2"
      %172 = call_opaque "__riscv_vmerge_vvm_i16m2"(%171, %170, %165, %133) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint16m2_t">, !emitc.opaque<"vbool8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %173 = load %130 : <!emitc.opaque<"int32_t">>
      %174 = literal "1" : !emitc.opaque<"size_t">
      %175 = call_opaque "__riscv_vmv_v_x_i32m1"(%173, %174) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %176 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%172, %175, %133) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %177 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%176) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %177 : !emitc.opaque<"int32_t"> to %130 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read"
      %178 = literal "8" : !emitc.opaque<"size_t">
      %179 = add %15, %178 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %180 = cast %179 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %181 = literal "0" : index
      %182 = subscript %180[%181] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %183 = load %182 : <!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1"
      %184 = call_opaque "__riscv_vmv_v_x_u8m1"(%183, %133) : (!emitc.opaque<"const uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1"
      %185 = call_opaque "__riscv_vand_vv_u8m1"(%184, %7, %133) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8"
      %186 = literal "0" : !emitc.opaque<"int">
      %187 = call_opaque "__riscv_vmsne_vx_u8m1_b8"(%185, %186, %133) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool8_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr"
      %188 = literal "18" : !emitc.opaque<"size_t">
      %189 = add %128, %188 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %190 = cast %189 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %191 = call_opaque "__riscv_vle8_v_i8m1"(%190, %133) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2"
      %192 = call_opaque "__riscv_vwcvt_x_x_v_i16m2"(%191, %133) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2"
      %193 = call_opaque "__riscv_vneg_v_i16m2"(%192, %133) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2"
      %194 = call_opaque "__riscv_vmerge_vvm_i16m2"(%193, %192, %187, %133) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint16m2_t">, !emitc.opaque<"vbool8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %195 = load %130 : <!emitc.opaque<"int32_t">>
      %196 = literal "1" : !emitc.opaque<"size_t">
      %197 = call_opaque "__riscv_vmv_v_x_i32m1"(%195, %196) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %198 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%194, %197, %133) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %199 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%198) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %199 : !emitc.opaque<"int32_t"> to %130 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read"
      %200 = literal "9" : !emitc.opaque<"size_t">
      %201 = add %15, %200 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %202 = cast %201 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %203 = literal "0" : index
      %204 = subscript %202[%203] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %205 = load %204 : <!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1"
      %206 = call_opaque "__riscv_vmv_v_x_u8m1"(%205, %133) : (!emitc.opaque<"const uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1"
      %207 = call_opaque "__riscv_vand_vv_u8m1"(%206, %7, %133) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8"
      %208 = literal "0" : !emitc.opaque<"int">
      %209 = call_opaque "__riscv_vmsne_vx_u8m1_b8"(%207, %208, %133) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool8_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr"
      %210 = literal "26" : !emitc.opaque<"size_t">
      %211 = add %128, %210 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %212 = cast %211 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %213 = call_opaque "__riscv_vle8_v_i8m1"(%212, %133) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2"
      %214 = call_opaque "__riscv_vwcvt_x_x_v_i16m2"(%213, %133) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2"
      %215 = call_opaque "__riscv_vneg_v_i16m2"(%214, %133) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2"
      %216 = call_opaque "__riscv_vmerge_vvm_i16m2"(%215, %214, %209, %133) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint16m2_t">, !emitc.opaque<"vbool8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %217 = load %130 : <!emitc.opaque<"int32_t">>
      %218 = literal "1" : !emitc.opaque<"size_t">
      %219 = call_opaque "__riscv_vmv_v_x_i32m1"(%217, %218) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %220 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%216, %219, %133) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %221 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%220) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %221 : !emitc.opaque<"int32_t"> to %130 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate_sub"
      %222 = load %130 : <!emitc.opaque<"int32_t">>
      %223 = load %17 : <!emitc.opaque<"float">>
      %224 = expression : !emitc.opaque<"float"> {
        %434 = cast %222 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %435 = mul %129, %434 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %436 = add %223, %435 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %436 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %224 : !emitc.opaque<"float"> to %17 : <!emitc.opaque<"float">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %225 = literal "4" : !emitc.opaque<"size_t">
      %226 = mul %arg4, %225 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %227 = literal "2" : !emitc.opaque<"size_t">
      %228 = add %226, %227 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %229 = literal "34" : !emitc.opaque<"size_t">
      %230 = mul %228, %229 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %231 = add %arg3, %230 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %232 = call_opaque "(float)*(const _Float16 *)"(%231) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.local_variable=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %233 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %234 = literal "0" : !emitc.opaque<"int32_t">
      assign %234 : !emitc.opaque<"int32_t"> to %233 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %235 = literal "8" : !emitc.opaque<"size_t">
      %236 = call_opaque "__riscv_vsetvl_e8m1"(%235) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read"
      %237 = literal "10" : !emitc.opaque<"size_t">
      %238 = add %15, %237 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %239 = cast %238 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %240 = literal "0" : index
      %241 = subscript %239[%240] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %242 = load %241 : <!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1"
      %243 = call_opaque "__riscv_vmv_v_x_u8m1"(%242, %236) : (!emitc.opaque<"const uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1"
      %244 = call_opaque "__riscv_vand_vv_u8m1"(%243, %7, %236) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8"
      %245 = literal "0" : !emitc.opaque<"int">
      %246 = call_opaque "__riscv_vmsne_vx_u8m1_b8"(%244, %245, %236) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool8_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr"
      %247 = literal "2" : !emitc.opaque<"size_t">
      %248 = add %231, %247 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %249 = cast %248 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %250 = call_opaque "__riscv_vle8_v_i8m1"(%249, %236) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2"
      %251 = call_opaque "__riscv_vwcvt_x_x_v_i16m2"(%250, %236) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2"
      %252 = call_opaque "__riscv_vneg_v_i16m2"(%251, %236) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2"
      %253 = call_opaque "__riscv_vmerge_vvm_i16m2"(%252, %251, %246, %236) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint16m2_t">, !emitc.opaque<"vbool8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %254 = load %233 : <!emitc.opaque<"int32_t">>
      %255 = literal "1" : !emitc.opaque<"size_t">
      %256 = call_opaque "__riscv_vmv_v_x_i32m1"(%254, %255) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %257 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%253, %256, %236) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %258 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%257) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %258 : !emitc.opaque<"int32_t"> to %233 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read"
      %259 = literal "11" : !emitc.opaque<"size_t">
      %260 = add %15, %259 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %261 = cast %260 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %262 = literal "0" : index
      %263 = subscript %261[%262] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %264 = load %263 : <!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1"
      %265 = call_opaque "__riscv_vmv_v_x_u8m1"(%264, %236) : (!emitc.opaque<"const uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1"
      %266 = call_opaque "__riscv_vand_vv_u8m1"(%265, %7, %236) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8"
      %267 = literal "0" : !emitc.opaque<"int">
      %268 = call_opaque "__riscv_vmsne_vx_u8m1_b8"(%266, %267, %236) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool8_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr"
      %269 = literal "10" : !emitc.opaque<"size_t">
      %270 = add %231, %269 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %271 = cast %270 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %272 = call_opaque "__riscv_vle8_v_i8m1"(%271, %236) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2"
      %273 = call_opaque "__riscv_vwcvt_x_x_v_i16m2"(%272, %236) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2"
      %274 = call_opaque "__riscv_vneg_v_i16m2"(%273, %236) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2"
      %275 = call_opaque "__riscv_vmerge_vvm_i16m2"(%274, %273, %268, %236) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint16m2_t">, !emitc.opaque<"vbool8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %276 = load %233 : <!emitc.opaque<"int32_t">>
      %277 = literal "1" : !emitc.opaque<"size_t">
      %278 = call_opaque "__riscv_vmv_v_x_i32m1"(%276, %277) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %279 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%275, %278, %236) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %280 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%279) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %280 : !emitc.opaque<"int32_t"> to %233 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read"
      %281 = literal "12" : !emitc.opaque<"size_t">
      %282 = add %15, %281 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %283 = cast %282 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %284 = literal "0" : index
      %285 = subscript %283[%284] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %286 = load %285 : <!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1"
      %287 = call_opaque "__riscv_vmv_v_x_u8m1"(%286, %236) : (!emitc.opaque<"const uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1"
      %288 = call_opaque "__riscv_vand_vv_u8m1"(%287, %7, %236) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8"
      %289 = literal "0" : !emitc.opaque<"int">
      %290 = call_opaque "__riscv_vmsne_vx_u8m1_b8"(%288, %289, %236) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool8_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr"
      %291 = literal "18" : !emitc.opaque<"size_t">
      %292 = add %231, %291 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %293 = cast %292 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %294 = call_opaque "__riscv_vle8_v_i8m1"(%293, %236) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2"
      %295 = call_opaque "__riscv_vwcvt_x_x_v_i16m2"(%294, %236) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2"
      %296 = call_opaque "__riscv_vneg_v_i16m2"(%295, %236) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2"
      %297 = call_opaque "__riscv_vmerge_vvm_i16m2"(%296, %295, %290, %236) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint16m2_t">, !emitc.opaque<"vbool8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %298 = load %233 : <!emitc.opaque<"int32_t">>
      %299 = literal "1" : !emitc.opaque<"size_t">
      %300 = call_opaque "__riscv_vmv_v_x_i32m1"(%298, %299) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %301 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%297, %300, %236) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %302 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%301) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %302 : !emitc.opaque<"int32_t"> to %233 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read"
      %303 = literal "13" : !emitc.opaque<"size_t">
      %304 = add %15, %303 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %305 = cast %304 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %306 = literal "0" : index
      %307 = subscript %305[%306] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %308 = load %307 : <!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1"
      %309 = call_opaque "__riscv_vmv_v_x_u8m1"(%308, %236) : (!emitc.opaque<"const uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1"
      %310 = call_opaque "__riscv_vand_vv_u8m1"(%309, %7, %236) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8"
      %311 = literal "0" : !emitc.opaque<"int">
      %312 = call_opaque "__riscv_vmsne_vx_u8m1_b8"(%310, %311, %236) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool8_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr"
      %313 = literal "26" : !emitc.opaque<"size_t">
      %314 = add %231, %313 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %315 = cast %314 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %316 = call_opaque "__riscv_vle8_v_i8m1"(%315, %236) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2"
      %317 = call_opaque "__riscv_vwcvt_x_x_v_i16m2"(%316, %236) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2"
      %318 = call_opaque "__riscv_vneg_v_i16m2"(%317, %236) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2"
      %319 = call_opaque "__riscv_vmerge_vvm_i16m2"(%318, %317, %312, %236) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint16m2_t">, !emitc.opaque<"vbool8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %320 = load %233 : <!emitc.opaque<"int32_t">>
      %321 = literal "1" : !emitc.opaque<"size_t">
      %322 = call_opaque "__riscv_vmv_v_x_i32m1"(%320, %321) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %323 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%319, %322, %236) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %324 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%323) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %324 : !emitc.opaque<"int32_t"> to %233 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate_sub"
      %325 = load %233 : <!emitc.opaque<"int32_t">>
      %326 = load %17 : <!emitc.opaque<"float">>
      %327 = expression : !emitc.opaque<"float"> {
        %434 = cast %325 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %435 = mul %232, %434 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %436 = add %326, %435 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %436 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %327 : !emitc.opaque<"float"> to %17 : <!emitc.opaque<"float">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %328 = literal "4" : !emitc.opaque<"size_t">
      %329 = mul %arg4, %328 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %330 = literal "3" : !emitc.opaque<"size_t">
      %331 = add %329, %330 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %332 = literal "34" : !emitc.opaque<"size_t">
      %333 = mul %331, %332 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %334 = add %arg3, %333 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %335 = call_opaque "(float)*(const _Float16 *)"(%334) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.local_variable=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %336 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %337 = literal "0" : !emitc.opaque<"int32_t">
      assign %337 : !emitc.opaque<"int32_t"> to %336 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %338 = literal "8" : !emitc.opaque<"size_t">
      %339 = call_opaque "__riscv_vsetvl_e8m1"(%338) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read"
      %340 = literal "14" : !emitc.opaque<"size_t">
      %341 = add %15, %340 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %342 = cast %341 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %343 = literal "0" : index
      %344 = subscript %342[%343] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %345 = load %344 : <!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1"
      %346 = call_opaque "__riscv_vmv_v_x_u8m1"(%345, %339) : (!emitc.opaque<"const uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1"
      %347 = call_opaque "__riscv_vand_vv_u8m1"(%346, %7, %339) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8"
      %348 = literal "0" : !emitc.opaque<"int">
      %349 = call_opaque "__riscv_vmsne_vx_u8m1_b8"(%347, %348, %339) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool8_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr"
      %350 = literal "2" : !emitc.opaque<"size_t">
      %351 = add %334, %350 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %352 = cast %351 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %353 = call_opaque "__riscv_vle8_v_i8m1"(%352, %339) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2"
      %354 = call_opaque "__riscv_vwcvt_x_x_v_i16m2"(%353, %339) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2"
      %355 = call_opaque "__riscv_vneg_v_i16m2"(%354, %339) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2"
      %356 = call_opaque "__riscv_vmerge_vvm_i16m2"(%355, %354, %349, %339) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint16m2_t">, !emitc.opaque<"vbool8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %357 = load %336 : <!emitc.opaque<"int32_t">>
      %358 = literal "1" : !emitc.opaque<"size_t">
      %359 = call_opaque "__riscv_vmv_v_x_i32m1"(%357, %358) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %360 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%356, %359, %339) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %361 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%360) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %361 : !emitc.opaque<"int32_t"> to %336 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read"
      %362 = literal "15" : !emitc.opaque<"size_t">
      %363 = add %15, %362 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %364 = cast %363 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %365 = literal "0" : index
      %366 = subscript %364[%365] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %367 = load %366 : <!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1"
      %368 = call_opaque "__riscv_vmv_v_x_u8m1"(%367, %339) : (!emitc.opaque<"const uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1"
      %369 = call_opaque "__riscv_vand_vv_u8m1"(%368, %7, %339) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8"
      %370 = literal "0" : !emitc.opaque<"int">
      %371 = call_opaque "__riscv_vmsne_vx_u8m1_b8"(%369, %370, %339) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool8_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr"
      %372 = literal "10" : !emitc.opaque<"size_t">
      %373 = add %334, %372 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %374 = cast %373 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %375 = call_opaque "__riscv_vle8_v_i8m1"(%374, %339) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2"
      %376 = call_opaque "__riscv_vwcvt_x_x_v_i16m2"(%375, %339) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2"
      %377 = call_opaque "__riscv_vneg_v_i16m2"(%376, %339) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2"
      %378 = call_opaque "__riscv_vmerge_vvm_i16m2"(%377, %376, %371, %339) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint16m2_t">, !emitc.opaque<"vbool8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %379 = load %336 : <!emitc.opaque<"int32_t">>
      %380 = literal "1" : !emitc.opaque<"size_t">
      %381 = call_opaque "__riscv_vmv_v_x_i32m1"(%379, %380) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %382 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%378, %381, %339) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %383 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%382) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %383 : !emitc.opaque<"int32_t"> to %336 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read"
      %384 = literal "16" : !emitc.opaque<"size_t">
      %385 = add %15, %384 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %386 = cast %385 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %387 = literal "0" : index
      %388 = subscript %386[%387] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %389 = load %388 : <!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1"
      %390 = call_opaque "__riscv_vmv_v_x_u8m1"(%389, %339) : (!emitc.opaque<"const uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1"
      %391 = call_opaque "__riscv_vand_vv_u8m1"(%390, %7, %339) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8"
      %392 = literal "0" : !emitc.opaque<"int">
      %393 = call_opaque "__riscv_vmsne_vx_u8m1_b8"(%391, %392, %339) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool8_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr"
      %394 = literal "18" : !emitc.opaque<"size_t">
      %395 = add %334, %394 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %396 = cast %395 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %397 = call_opaque "__riscv_vle8_v_i8m1"(%396, %339) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2"
      %398 = call_opaque "__riscv_vwcvt_x_x_v_i16m2"(%397, %339) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2"
      %399 = call_opaque "__riscv_vneg_v_i16m2"(%398, %339) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2"
      %400 = call_opaque "__riscv_vmerge_vvm_i16m2"(%399, %398, %393, %339) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint16m2_t">, !emitc.opaque<"vbool8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %401 = load %336 : <!emitc.opaque<"int32_t">>
      %402 = literal "1" : !emitc.opaque<"size_t">
      %403 = call_opaque "__riscv_vmv_v_x_i32m1"(%401, %402) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %404 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%400, %403, %339) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %405 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%404) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %405 : !emitc.opaque<"int32_t"> to %336 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read"
      %406 = literal "17" : !emitc.opaque<"size_t">
      %407 = add %15, %406 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %408 = cast %407 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %409 = literal "0" : index
      %410 = subscript %408[%409] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %411 = load %410 : <!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1"
      %412 = call_opaque "__riscv_vmv_v_x_u8m1"(%411, %339) : (!emitc.opaque<"const uint8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1"
      %413 = call_opaque "__riscv_vand_vv_u8m1"(%412, %7, %339) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8"
      %414 = literal "0" : !emitc.opaque<"int">
      %415 = call_opaque "__riscv_vmsne_vx_u8m1_b8"(%413, %414, %339) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool8_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr"
      %416 = literal "26" : !emitc.opaque<"size_t">
      %417 = add %334, %416 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %418 = cast %417 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %419 = call_opaque "__riscv_vle8_v_i8m1"(%418, %339) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2"
      %420 = call_opaque "__riscv_vwcvt_x_x_v_i16m2"(%419, %339) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2"
      %421 = call_opaque "__riscv_vneg_v_i16m2"(%420, %339) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2"
      %422 = call_opaque "__riscv_vmerge_vvm_i16m2"(%421, %420, %415, %339) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint16m2_t">, !emitc.opaque<"vbool8_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %423 = load %336 : <!emitc.opaque<"int32_t">>
      %424 = literal "1" : !emitc.opaque<"size_t">
      %425 = call_opaque "__riscv_vmv_v_x_i32m1"(%423, %424) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %426 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%422, %425, %339) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %427 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%426) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %427 : !emitc.opaque<"int32_t"> to %336 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate_sub"
      %428 = load %336 : <!emitc.opaque<"int32_t">>
      %429 = load %17 : <!emitc.opaque<"float">>
      %430 = expression : !emitc.opaque<"float"> {
        %434 = cast %428 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %435 = mul %335, %434 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %436 = add %429, %435 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %436 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %430 : !emitc.opaque<"float"> to %17 : <!emitc.opaque<"float">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %431 = load %17 : <!emitc.opaque<"float">>
      %432 = load %1 : <!emitc.opaque<"float">>
      %433 = expression : !emitc.opaque<"float"> {
        %434 = mul %16, %431 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %435 = add %432, %434 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %435 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %433 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
    }
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s"
    %10 = literal "0" : index
    %11 = subscript %arg1[%10] : (!emitc.ptr<!emitc.opaque<"float">>, index) -> !emitc.lvalue<!emitc.opaque<"float">>
    %12 = load %1 : <!emitc.opaque<"float">>
    assign %12 : !emitc.opaque<"float"> to %11 : <!emitc.opaque<"float">>
    return
  }
}

