module {
  emitc.include <"stddef.h">
  emitc.include <"stdint.h">
  emitc.include <"riscv_vector.h">
  emitc.func @tcrv_emitc_ggml_vec_dot_nvfp4_q8_0_kernel_ggml_vec_dot_nvfp4_q8_0(%arg0: !emitc.opaque<"size_t">, %arg1: !emitc.ptr<!emitc.opaque<"float">>, %arg2: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg3: !emitc.ptr<!emitc.opaque<"const uint8_t">>) attributes {specifiers = ["extern", "\22C\22"]} {
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1"
    %0 = call_opaque "__riscv_vsetvl_e32m1"(%arg0) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "static const int8_t tcrv_nvfp4_kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};"
    verbatim "// tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
    %1 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"float">>
    %2 = literal "0.0f" : !emitc.opaque<"float">
    assign %2 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count"
    %3 = literal "64" : !emitc.opaque<"size_t">
    %4 = div %arg0, %3 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=codebook_table_load"
    %5 = literal "tcrv_nvfp4_kvalues" : !emitc.ptr<!emitc.opaque<"const int8_t">>
    %6 = literal "16" : !emitc.opaque<"size_t">
    %7 = call_opaque "__riscv_vle8_v_i8m1"(%5, %6) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
    %8 = literal "1" : !emitc.opaque<"size_t">
    %9 = literal "0" : !emitc.opaque<"size_t">
    for %arg4 = %9 to %4 step %8  : !emitc.opaque<"size_t"> {
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x"
      %13 = literal "36" : !emitc.opaque<"size_t">
      %14 = mul %arg4, %13 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %15 = add %arg2, %14 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_block_base_index"
      %16 = literal "2" : !emitc.opaque<"size_t">
      %17 = mul %arg4, %16 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block"
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_scale_load"
      %18 = cast %15 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %19 = literal "0" : index
      %20 = subscript %18[%19] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %21 = load %20 : <!emitc.opaque<"const uint8_t">>
      %22 = cast %21 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"uint32_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_exp_man_split"
      %23 = literal "3" : !emitc.opaque<"uint32_t">
      %24 = literal "0xF" : !emitc.opaque<"uint32_t">
      %25 = bitwise_right_shift %22, %23 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %26 = bitwise_and %25, %24 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %27 = literal "0x7" : !emitc.opaque<"uint32_t">
      %28 = bitwise_and %22, %27 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %29 = cast %26 : !emitc.opaque<"uint32_t"> to !emitc.opaque<"int">
      %30 = cast %28 : !emitc.opaque<"uint32_t"> to !emitc.opaque<"int">
      %31 = cast %30 : !emitc.opaque<"int"> to !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_ldexpf_branches"
      %32 = literal "-9" : !emitc.opaque<"int">
      %33 = call_opaque "ldexpf"(%31, %32) : (!emitc.opaque<"float">, !emitc.opaque<"int">) -> !emitc.opaque<"float">
      %34 = literal "1.0f" : !emitc.opaque<"float">
      %35 = literal "8.0f" : !emitc.opaque<"float">
      %36 = div %31, %35 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
      %37 = add %34, %36 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
      %38 = literal "7" : !emitc.opaque<"int">
      %39 = sub %29, %38 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      %40 = call_opaque "ldexpf"(%37, %39) : (!emitc.opaque<"float">, !emitc.opaque<"int">) -> !emitc.opaque<"float">
      %41 = literal "0" : !emitc.opaque<"uint32_t">
      %42 = cmp eq, %26, %41 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> i1
      %43 = conditional %42, %33, %40 : !emitc.opaque<"float">
      %44 = literal "0.5f" : !emitc.opaque<"float">
      %45 = mul %43, %44 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_specials"
      %46 = literal "0.0f" : !emitc.opaque<"float">
      %47 = literal "0" : !emitc.opaque<"uint32_t">
      %48 = literal "0x7F" : !emitc.opaque<"uint32_t">
      %49 = cmp eq, %22, %47 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> i1
      %50 = cmp eq, %22, %48 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> i1
      %51 = logical_or %49, %50 : i1, i1
      %52 = conditional %51, %46, %45 : !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_weight_quants"
      %53 = literal "4" : !emitc.opaque<"size_t">
      %54 = add %15, %53 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %55 = cast %54 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_q8_base"
      %56 = literal "34" : !emitc.opaque<"size_t">
      %57 = mul %17, %56 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %58 = add %arg3, %57 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %59 = call_opaque "(float)*(const _Float16 *)"(%58) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_q8_quants"
      %60 = literal "2" : !emitc.opaque<"size_t">
      %61 = add %58, %60 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %62 = cast %61 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %63 = literal "8" : !emitc.opaque<"size_t">
      %64 = call_opaque "__riscv_vsetvl_e8m1"(%63) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
      %65 = call_opaque "__riscv_vle8_v_u8m1"(%55, %64) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      %66 = literal "8" : !emitc.opaque<"size_t">
      %67 = add %62, %66 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %68 = call_opaque "__riscv_vle8_v_i8m1"(%62, %64) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %69 = call_opaque "__riscv_vle8_v_i8m1"(%67, %64) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1"
      %70 = literal "0x0F" : !emitc.opaque<"int">
      %71 = call_opaque "__riscv_vand_vx_u8m1"(%65, %70, %64) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1"
      %72 = literal "0x04" : !emitc.opaque<"int">
      %73 = call_opaque "__riscv_vsrl_vx_u8m1"(%65, %72, %64) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1"
      %74 = call_opaque "__riscv_vrgather_vv_i8m1"(%7, %71, %64) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1"
      %75 = call_opaque "__riscv_vrgather_vv_i8m1"(%7, %73, %64) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %76 = call_opaque "__riscv_vwmul_vv_i16m2"(%74, %68, %64) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2"
      %77 = call_opaque "__riscv_vwmacc_vv_i16m2"(%76, %75, %69, %64) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %78 = literal "0" : !emitc.opaque<"int32_t">
      %79 = literal "1" : !emitc.opaque<"size_t">
      %80 = call_opaque "__riscv_vmv_v_x_i32m1"(%78, %79) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %81 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%77, %80, %64) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %82 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%81) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %83 = load %1 : <!emitc.opaque<"float">>
      %84 = expression : !emitc.opaque<"float"> {
        %296 = cast %82 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %297 = mul %59, %52 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %298 = mul %297, %296 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %299 = add %83, %298 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %299 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %84 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block"
      %85 = literal "1" : !emitc.opaque<"size_t">
      %86 = add %15, %85 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_scale_load"
      %87 = cast %86 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %88 = literal "0" : index
      %89 = subscript %87[%88] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %90 = load %89 : <!emitc.opaque<"const uint8_t">>
      %91 = cast %90 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"uint32_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_exp_man_split"
      %92 = literal "3" : !emitc.opaque<"uint32_t">
      %93 = literal "0xF" : !emitc.opaque<"uint32_t">
      %94 = bitwise_right_shift %91, %92 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %95 = bitwise_and %94, %93 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %96 = literal "0x7" : !emitc.opaque<"uint32_t">
      %97 = bitwise_and %91, %96 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %98 = cast %95 : !emitc.opaque<"uint32_t"> to !emitc.opaque<"int">
      %99 = cast %97 : !emitc.opaque<"uint32_t"> to !emitc.opaque<"int">
      %100 = cast %99 : !emitc.opaque<"int"> to !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_ldexpf_branches"
      %101 = literal "-9" : !emitc.opaque<"int">
      %102 = call_opaque "ldexpf"(%100, %101) : (!emitc.opaque<"float">, !emitc.opaque<"int">) -> !emitc.opaque<"float">
      %103 = literal "1.0f" : !emitc.opaque<"float">
      %104 = literal "8.0f" : !emitc.opaque<"float">
      %105 = div %100, %104 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
      %106 = add %103, %105 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
      %107 = literal "7" : !emitc.opaque<"int">
      %108 = sub %98, %107 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      %109 = call_opaque "ldexpf"(%106, %108) : (!emitc.opaque<"float">, !emitc.opaque<"int">) -> !emitc.opaque<"float">
      %110 = literal "0" : !emitc.opaque<"uint32_t">
      %111 = cmp eq, %95, %110 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> i1
      %112 = conditional %111, %102, %109 : !emitc.opaque<"float">
      %113 = literal "0.5f" : !emitc.opaque<"float">
      %114 = mul %112, %113 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_specials"
      %115 = literal "0.0f" : !emitc.opaque<"float">
      %116 = literal "0" : !emitc.opaque<"uint32_t">
      %117 = literal "0x7F" : !emitc.opaque<"uint32_t">
      %118 = cmp eq, %91, %116 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> i1
      %119 = cmp eq, %91, %117 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> i1
      %120 = logical_or %118, %119 : i1, i1
      %121 = conditional %120, %115, %114 : !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_weight_quants"
      %122 = literal "12" : !emitc.opaque<"size_t">
      %123 = add %15, %122 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %124 = cast %123 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_q8_base"
      %125 = literal "34" : !emitc.opaque<"size_t">
      %126 = mul %17, %125 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %127 = add %arg3, %126 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %128 = call_opaque "(float)*(const _Float16 *)"(%127) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_q8_quants"
      %129 = literal "18" : !emitc.opaque<"size_t">
      %130 = add %127, %129 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %131 = cast %130 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %132 = literal "8" : !emitc.opaque<"size_t">
      %133 = call_opaque "__riscv_vsetvl_e8m1"(%132) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
      %134 = call_opaque "__riscv_vle8_v_u8m1"(%124, %133) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      %135 = literal "8" : !emitc.opaque<"size_t">
      %136 = add %131, %135 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %137 = call_opaque "__riscv_vle8_v_i8m1"(%131, %133) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %138 = call_opaque "__riscv_vle8_v_i8m1"(%136, %133) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1"
      %139 = literal "0x0F" : !emitc.opaque<"int">
      %140 = call_opaque "__riscv_vand_vx_u8m1"(%134, %139, %133) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1"
      %141 = literal "0x04" : !emitc.opaque<"int">
      %142 = call_opaque "__riscv_vsrl_vx_u8m1"(%134, %141, %133) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1"
      %143 = call_opaque "__riscv_vrgather_vv_i8m1"(%7, %140, %133) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1"
      %144 = call_opaque "__riscv_vrgather_vv_i8m1"(%7, %142, %133) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %145 = call_opaque "__riscv_vwmul_vv_i16m2"(%143, %137, %133) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2"
      %146 = call_opaque "__riscv_vwmacc_vv_i16m2"(%145, %144, %138, %133) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %147 = literal "0" : !emitc.opaque<"int32_t">
      %148 = literal "1" : !emitc.opaque<"size_t">
      %149 = call_opaque "__riscv_vmv_v_x_i32m1"(%147, %148) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %150 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%146, %149, %133) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %151 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%150) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %152 = load %1 : <!emitc.opaque<"float">>
      %153 = expression : !emitc.opaque<"float"> {
        %296 = cast %151 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %297 = mul %128, %121 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %298 = mul %297, %296 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %299 = add %152, %298 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %299 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %153 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block"
      %154 = literal "2" : !emitc.opaque<"size_t">
      %155 = add %15, %154 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_scale_load"
      %156 = cast %155 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %157 = literal "0" : index
      %158 = subscript %156[%157] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %159 = load %158 : <!emitc.opaque<"const uint8_t">>
      %160 = cast %159 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"uint32_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_exp_man_split"
      %161 = literal "3" : !emitc.opaque<"uint32_t">
      %162 = literal "0xF" : !emitc.opaque<"uint32_t">
      %163 = bitwise_right_shift %160, %161 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %164 = bitwise_and %163, %162 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %165 = literal "0x7" : !emitc.opaque<"uint32_t">
      %166 = bitwise_and %160, %165 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %167 = cast %164 : !emitc.opaque<"uint32_t"> to !emitc.opaque<"int">
      %168 = cast %166 : !emitc.opaque<"uint32_t"> to !emitc.opaque<"int">
      %169 = cast %168 : !emitc.opaque<"int"> to !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_ldexpf_branches"
      %170 = literal "-9" : !emitc.opaque<"int">
      %171 = call_opaque "ldexpf"(%169, %170) : (!emitc.opaque<"float">, !emitc.opaque<"int">) -> !emitc.opaque<"float">
      %172 = literal "1.0f" : !emitc.opaque<"float">
      %173 = literal "8.0f" : !emitc.opaque<"float">
      %174 = div %169, %173 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
      %175 = add %172, %174 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
      %176 = literal "7" : !emitc.opaque<"int">
      %177 = sub %167, %176 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      %178 = call_opaque "ldexpf"(%175, %177) : (!emitc.opaque<"float">, !emitc.opaque<"int">) -> !emitc.opaque<"float">
      %179 = literal "0" : !emitc.opaque<"uint32_t">
      %180 = cmp eq, %164, %179 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> i1
      %181 = conditional %180, %171, %178 : !emitc.opaque<"float">
      %182 = literal "0.5f" : !emitc.opaque<"float">
      %183 = mul %181, %182 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_specials"
      %184 = literal "0.0f" : !emitc.opaque<"float">
      %185 = literal "0" : !emitc.opaque<"uint32_t">
      %186 = literal "0x7F" : !emitc.opaque<"uint32_t">
      %187 = cmp eq, %160, %185 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> i1
      %188 = cmp eq, %160, %186 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> i1
      %189 = logical_or %187, %188 : i1, i1
      %190 = conditional %189, %184, %183 : !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_weight_quants"
      %191 = literal "20" : !emitc.opaque<"size_t">
      %192 = add %15, %191 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %193 = cast %192 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_q8_base"
      %194 = literal "1" : !emitc.opaque<"size_t">
      %195 = add %17, %194 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %196 = literal "34" : !emitc.opaque<"size_t">
      %197 = mul %195, %196 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %198 = add %arg3, %197 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %199 = call_opaque "(float)*(const _Float16 *)"(%198) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_q8_quants"
      %200 = literal "2" : !emitc.opaque<"size_t">
      %201 = add %198, %200 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %202 = cast %201 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %203 = literal "8" : !emitc.opaque<"size_t">
      %204 = call_opaque "__riscv_vsetvl_e8m1"(%203) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
      %205 = call_opaque "__riscv_vle8_v_u8m1"(%193, %204) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      %206 = literal "8" : !emitc.opaque<"size_t">
      %207 = add %202, %206 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %208 = call_opaque "__riscv_vle8_v_i8m1"(%202, %204) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %209 = call_opaque "__riscv_vle8_v_i8m1"(%207, %204) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1"
      %210 = literal "0x0F" : !emitc.opaque<"int">
      %211 = call_opaque "__riscv_vand_vx_u8m1"(%205, %210, %204) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1"
      %212 = literal "0x04" : !emitc.opaque<"int">
      %213 = call_opaque "__riscv_vsrl_vx_u8m1"(%205, %212, %204) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1"
      %214 = call_opaque "__riscv_vrgather_vv_i8m1"(%7, %211, %204) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1"
      %215 = call_opaque "__riscv_vrgather_vv_i8m1"(%7, %213, %204) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %216 = call_opaque "__riscv_vwmul_vv_i16m2"(%214, %208, %204) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2"
      %217 = call_opaque "__riscv_vwmacc_vv_i16m2"(%216, %215, %209, %204) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %218 = literal "0" : !emitc.opaque<"int32_t">
      %219 = literal "1" : !emitc.opaque<"size_t">
      %220 = call_opaque "__riscv_vmv_v_x_i32m1"(%218, %219) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %221 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%217, %220, %204) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %222 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%221) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %223 = load %1 : <!emitc.opaque<"float">>
      %224 = expression : !emitc.opaque<"float"> {
        %296 = cast %222 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %297 = mul %199, %190 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %298 = mul %297, %296 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %299 = add %223, %298 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %299 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %224 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block"
      %225 = literal "3" : !emitc.opaque<"size_t">
      %226 = add %15, %225 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_scale_load"
      %227 = cast %226 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %228 = literal "0" : index
      %229 = subscript %227[%228] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const uint8_t">>
      %230 = load %229 : <!emitc.opaque<"const uint8_t">>
      %231 = cast %230 : !emitc.opaque<"const uint8_t"> to !emitc.opaque<"uint32_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_exp_man_split"
      %232 = literal "3" : !emitc.opaque<"uint32_t">
      %233 = literal "0xF" : !emitc.opaque<"uint32_t">
      %234 = bitwise_right_shift %231, %232 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %235 = bitwise_and %234, %233 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %236 = literal "0x7" : !emitc.opaque<"uint32_t">
      %237 = bitwise_and %231, %236 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> !emitc.opaque<"uint32_t">
      %238 = cast %235 : !emitc.opaque<"uint32_t"> to !emitc.opaque<"int">
      %239 = cast %237 : !emitc.opaque<"uint32_t"> to !emitc.opaque<"int">
      %240 = cast %239 : !emitc.opaque<"int"> to !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_ldexpf_branches"
      %241 = literal "-9" : !emitc.opaque<"int">
      %242 = call_opaque "ldexpf"(%240, %241) : (!emitc.opaque<"float">, !emitc.opaque<"int">) -> !emitc.opaque<"float">
      %243 = literal "1.0f" : !emitc.opaque<"float">
      %244 = literal "8.0f" : !emitc.opaque<"float">
      %245 = div %240, %244 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
      %246 = add %243, %245 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
      %247 = literal "7" : !emitc.opaque<"int">
      %248 = sub %238, %247 : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.opaque<"int">
      %249 = call_opaque "ldexpf"(%246, %248) : (!emitc.opaque<"float">, !emitc.opaque<"int">) -> !emitc.opaque<"float">
      %250 = literal "0" : !emitc.opaque<"uint32_t">
      %251 = cmp eq, %235, %250 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> i1
      %252 = conditional %251, %242, %249 : !emitc.opaque<"float">
      %253 = literal "0.5f" : !emitc.opaque<"float">
      %254 = mul %252, %253 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_specials"
      %255 = literal "0.0f" : !emitc.opaque<"float">
      %256 = literal "0" : !emitc.opaque<"uint32_t">
      %257 = literal "0x7F" : !emitc.opaque<"uint32_t">
      %258 = cmp eq, %231, %256 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> i1
      %259 = cmp eq, %231, %257 : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">) -> i1
      %260 = logical_or %258, %259 : i1, i1
      %261 = conditional %260, %255, %254 : !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_weight_quants"
      %262 = literal "28" : !emitc.opaque<"size_t">
      %263 = add %15, %262 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %264 = cast %263 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_q8_base"
      %265 = literal "1" : !emitc.opaque<"size_t">
      %266 = add %17, %265 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %267 = literal "34" : !emitc.opaque<"size_t">
      %268 = mul %266, %267 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %269 = add %arg3, %268 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %270 = call_opaque "(float)*(const _Float16 *)"(%269) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_q8_quants"
      %271 = literal "18" : !emitc.opaque<"size_t">
      %272 = add %269, %271 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %273 = cast %272 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1"
      %274 = literal "8" : !emitc.opaque<"size_t">
      %275 = call_opaque "__riscv_vsetvl_e8m1"(%274) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1"
      %276 = call_opaque "__riscv_vle8_v_u8m1"(%264, %275) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      %277 = literal "8" : !emitc.opaque<"size_t">
      %278 = add %273, %277 : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %279 = call_opaque "__riscv_vle8_v_i8m1"(%273, %275) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1"
      %280 = call_opaque "__riscv_vle8_v_i8m1"(%278, %275) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1"
      %281 = literal "0x0F" : !emitc.opaque<"int">
      %282 = call_opaque "__riscv_vand_vx_u8m1"(%276, %281, %275) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1"
      %283 = literal "0x04" : !emitc.opaque<"int">
      %284 = call_opaque "__riscv_vsrl_vx_u8m1"(%276, %283, %275) : (!emitc.opaque<"vuint8m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1"
      %285 = call_opaque "__riscv_vrgather_vv_i8m1"(%7, %282, %275) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1"
      %286 = call_opaque "__riscv_vrgather_vv_i8m1"(%7, %284, %275) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vuint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2"
      %287 = call_opaque "__riscv_vwmul_vv_i16m2"(%285, %279, %275) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2"
      %288 = call_opaque "__riscv_vwmacc_vv_i16m2"(%287, %286, %280, %275) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %289 = literal "0" : !emitc.opaque<"int32_t">
      %290 = literal "1" : !emitc.opaque<"size_t">
      %291 = call_opaque "__riscv_vmv_v_x_i32m1"(%289, %290) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1"
      %292 = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%288, %291, %275) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %293 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%292) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %294 = load %1 : <!emitc.opaque<"float">>
      %295 = expression : !emitc.opaque<"float"> {
        %296 = cast %293 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %297 = mul %270, %261 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %298 = mul %297, %296 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %299 = add %294, %298 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %299 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %295 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
    }
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s"
    %10 = literal "0" : index
    %11 = subscript %arg1[%10] : (!emitc.ptr<!emitc.opaque<"float">>, index) -> !emitc.lvalue<!emitc.opaque<"float">>
    %12 = load %1 : <!emitc.opaque<"float">>
    assign %12 : !emitc.opaque<"float"> to %11 : <!emitc.opaque<"float">>
    return
  }
}

