module {
  emitc.include <"stddef.h">
  emitc.include <"stdint.h">
  emitc.include <"riscv_vector.h">
  emitc.func @tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0(%arg0: !emitc.opaque<"size_t">, %arg1: !emitc.ptr<!emitc.opaque<"float">>, %arg2: !emitc.opaque<"size_t">, %arg3: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg4: !emitc.opaque<"size_t">, %arg5: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg6: !emitc.opaque<"size_t">, %arg7: !emitc.opaque<"int32_t">) attributes {specifiers = ["extern", "\22C\22"]} {
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1"
    %0 = call_opaque "__riscv_vsetvl_e32m1"(%arg0) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    verbatim "// tcrv_emitc.route_source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
    verbatim "// tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
    %1 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"float">>
    %2 = literal "0.0f" : !emitc.opaque<"float">
    assign %2 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count"
    %3 = literal "32" : !emitc.opaque<"size_t">
    %4 = div %arg0, %3 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    %5 = literal "2" : !emitc.opaque<"size_t">
    %6 = rem %4, %5 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    %7 = sub %4, %6 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
    %8 = literal "0" : !emitc.opaque<"size_t">
    for %arg8 = %8 to %7 step %5  : !emitc.opaque<"size_t"> {
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x"
      %13 = literal "34" : !emitc.opaque<"size_t">
      %14 = mul %arg8, %13 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %15 = add %arg3, %14 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %16 = literal "34" : !emitc.opaque<"size_t">
      %17 = mul %arg8, %16 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %18 = add %arg5, %17 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %19 = call_opaque "(float)*(const _Float16 *)"(%15) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %20 = call_opaque "(float)*(const _Float16 *)"(%18) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %21 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %22 = literal "0" : !emitc.opaque<"int32_t">
      assign %22 : !emitc.opaque<"int32_t"> to %21 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2"
      %23 = literal "32" : !emitc.opaque<"size_t">
      %24 = call_opaque "__riscv_vsetvl_e8m2"(%23) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %25 = literal "0" : !emitc.opaque<"size_t">
      %26 = literal "2" : !emitc.opaque<"size_t">
      %27 = add %15, %26 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %28 = add %27, %25 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %29 = cast %28 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2"
      %30 = call_opaque "__riscv_vle8_v_i8m2"(%29, %24) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m2_t">
      %31 = literal "2" : !emitc.opaque<"size_t">
      %32 = add %18, %31 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %33 = add %32, %25 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %34 = cast %33 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2"
      %35 = call_opaque "__riscv_vle8_v_i8m2"(%34, %24) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4"
      %36 = call_opaque "__riscv_vwmul_vv_i16m4"(%30, %35, %24) : (!emitc.opaque<"vint8m2_t">, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m4_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %37 = literal "0" : !emitc.opaque<"int32_t">
      %38 = literal "1" : !emitc.opaque<"size_t">
      %39 = call_opaque "__riscv_vmv_v_x_i32m1"(%37, %38) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1"
      %40 = call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"(%36, %39, %24) : (!emitc.opaque<"vint16m4_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %41 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%40) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %41 : !emitc.opaque<"int32_t"> to %21 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x"
      %42 = literal "1" : !emitc.opaque<"size_t">
      %43 = add %arg8, %42 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %44 = literal "34" : !emitc.opaque<"size_t">
      %45 = mul %43, %44 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %46 = add %arg3, %45 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %47 = literal "1" : !emitc.opaque<"size_t">
      %48 = add %arg8, %47 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %49 = literal "34" : !emitc.opaque<"size_t">
      %50 = mul %48, %49 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %51 = add %arg5, %50 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %52 = call_opaque "(float)*(const _Float16 *)"(%46) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %53 = call_opaque "(float)*(const _Float16 *)"(%51) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %54 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %55 = literal "0" : !emitc.opaque<"int32_t">
      assign %55 : !emitc.opaque<"int32_t"> to %54 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2"
      %56 = literal "32" : !emitc.opaque<"size_t">
      %57 = call_opaque "__riscv_vsetvl_e8m2"(%56) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %58 = literal "0" : !emitc.opaque<"size_t">
      %59 = literal "2" : !emitc.opaque<"size_t">
      %60 = add %46, %59 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %61 = add %60, %58 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %62 = cast %61 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2"
      %63 = call_opaque "__riscv_vle8_v_i8m2"(%62, %57) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m2_t">
      %64 = literal "2" : !emitc.opaque<"size_t">
      %65 = add %51, %64 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %66 = add %65, %58 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      %67 = cast %66 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2"
      %68 = call_opaque "__riscv_vle8_v_i8m2"(%67, %57) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m2_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4"
      %69 = call_opaque "__riscv_vwmul_vv_i16m4"(%63, %68, %57) : (!emitc.opaque<"vint8m2_t">, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m4_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
      %70 = literal "0" : !emitc.opaque<"int32_t">
      %71 = literal "1" : !emitc.opaque<"size_t">
      %72 = call_opaque "__riscv_vmv_v_x_i32m1"(%70, %71) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1"
      %73 = call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"(%69, %72, %57) : (!emitc.opaque<"vint16m4_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
      %74 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%73) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
      verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %74 : !emitc.opaque<"int32_t"> to %54 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %75 = load %21 : <!emitc.opaque<"int32_t">>
      %76 = load %1 : <!emitc.opaque<"float">>
      %77 = expression : !emitc.opaque<"float"> {
        %81 = cast %75 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %82 = mul %19, %20 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %83 = mul %81, %82 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %84 = add %76, %83 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %84 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %77 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %78 = load %54 : <!emitc.opaque<"int32_t">>
      %79 = load %1 : <!emitc.opaque<"float">>
      %80 = expression : !emitc.opaque<"float"> {
        %81 = cast %78 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %82 = mul %52, %53 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %83 = mul %81, %82 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %84 = add %79, %83 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %84 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %80 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
    }
    %9 = literal "1" : !emitc.opaque<"size_t">
    for %arg8 = %7 to %4 step %9  : !emitc.opaque<"size_t"> {
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x"
      %13 = literal "34" : !emitc.opaque<"size_t">
      %14 = mul %arg8, %13 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %15 = add %arg3, %14 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y"
      %16 = literal "34" : !emitc.opaque<"size_t">
      %17 = mul %arg8, %16 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %18 = add %arg5, %17 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %19 = call_opaque "(float)*(const _Float16 *)"(%15) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h"
      %20 = call_opaque "(float)*(const _Float16 *)"(%18) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
      verbatim "// tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      %21 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
      %22 = literal "0" : !emitc.opaque<"int32_t">
      assign %22 : !emitc.opaque<"int32_t"> to %21 : <!emitc.opaque<"int32_t">>
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2"
      %23 = literal "32" : !emitc.opaque<"size_t">
      %24 = call_opaque "__riscv_vsetvl_e8m2"(%23) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
      %25 = literal "32" : !emitc.opaque<"size_t">
      %26 = literal "0" : !emitc.opaque<"size_t">
      for %arg9 = %26 to %25 step %24  : !emitc.opaque<"size_t"> {
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2"
        %30 = literal "32" : !emitc.opaque<"size_t">
        %31 = sub %30, %arg9 : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %32 = call_opaque "__riscv_vsetvl_e8m2"(%31) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
        %33 = literal "2" : !emitc.opaque<"size_t">
        %34 = add %15, %33 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %35 = add %34, %arg9 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %36 = cast %35 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2"
        %37 = call_opaque "__riscv_vle8_v_i8m2"(%36, %32) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m2_t">
        %38 = literal "2" : !emitc.opaque<"size_t">
        %39 = add %18, %38 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %40 = add %39, %arg9 : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
        %41 = cast %40 : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2"
        %42 = call_opaque "__riscv_vle8_v_i8m2"(%41, %32) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8m2_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4"
        %43 = call_opaque "__riscv_vwmul_vv_i16m4"(%37, %42, %32) : (!emitc.opaque<"vint8m2_t">, !emitc.opaque<"vint8m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m4_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1"
        %44 = load %21 : <!emitc.opaque<"int32_t">>
        %45 = literal "1" : !emitc.opaque<"size_t">
        %46 = call_opaque "__riscv_vmv_v_x_i32m1"(%44, %45) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1"
        %47 = call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"(%43, %46, %32) : (!emitc.opaque<"vint16m4_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
        verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32"
        %48 = call_opaque "__riscv_vmv_x_s_i32m1_i32"(%47) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
        verbatim "// tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
        assign %48 : !emitc.opaque<"int32_t"> to %21 : <!emitc.opaque<"int32_t">>
      }
      verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate"
      %27 = load %21 : <!emitc.opaque<"int32_t">>
      %28 = load %1 : <!emitc.opaque<"float">>
      %29 = expression : !emitc.opaque<"float"> {
        %30 = cast %27 : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
        %31 = mul %19, %20 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %32 = mul %30, %31 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        %33 = add %28, %32 : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
        yield %33 : !emitc.opaque<"float">
      }
      verbatim "// tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface"
      assign %29 : !emitc.opaque<"float"> to %1 : <!emitc.opaque<"float">>
    }
    verbatim "// tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s"
    %10 = literal "0" : index
    %11 = subscript %arg1[%10] : (!emitc.ptr<!emitc.opaque<"float">>, index) -> !emitc.lvalue<!emitc.opaque<"float">>
    %12 = load %1 : <!emitc.opaque<"float">>
    assign %12 : !emitc.opaque<"float"> to %11 : <!emitc.opaque<"float">>
    return
  }
}

