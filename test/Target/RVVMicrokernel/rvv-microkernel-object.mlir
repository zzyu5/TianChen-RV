// REQUIRES: tianchenrv-local-rvv-object-clang

// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-rvv-microkernel-object > %t.direct.o
// RUN: test -s %t.direct.o
// RUN: llvm-readobj --file-headers --symbols %t.direct.o | FileCheck %s --check-prefixes=OBJ,DIRECTOBJ --implicit-check-not="Name: main"
// RUN: llvm-readobj --string-dump=.rodata.tianchenrv.rvv_artifact %t.direct.o | FileCheck %s --check-prefix=OBJNOTE
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-artifact > %t.generic.o
// RUN: test -s %t.generic.o
// RUN: llvm-readobj --file-headers --symbols %t.generic.o | FileCheck %s --check-prefixes=OBJ,GENERICOBJ --implicit-check-not="Name: main"
// RUN: llvm-readobj --string-dump=.rodata.tianchenrv.rvv_artifact %t.generic.o | FileCheck %s --check-prefix=OBJNOTE
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=SOURCE --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=tcrv_rvv_microkernel_ok --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | sed -e '/tcrv_rvv.lowering_boundary/s/emitc_lowerable_op_interface = "TCRVEmitCLowerableOpInterface", //' -e '/tcrv_rvv.lowering_boundary/s/emitc_source_op = "tcrv_rvv.i32_add", //' -e '/tcrv_rvv.lowering_boundary/s/selected_binary_dtype = "i32", //' -e '/tcrv_rvv.lowering_boundary/s/selected_binary_family = "i32-vadd", //' -e '/tcrv_rvv.lowering_boundary/s/selected_binary_microkernel_op = "tcrv_rvv.i32_vadd_microkernel", //' -e '/tcrv_rvv.lowering_boundary/s/selected_binary_operator = "add", //' -e '/tcrv_rvv.lowering_boundary/s/selected_binary_source_kind = "default-i32-vadd-typed-body-materialization", //' | not tcrv-translate --tcrv-export-target-artifact 2>&1 | FileCheck %s --check-prefix=OBJECT-MISSING-SOURCE --implicit-check-not="Format: elf" --implicit-check-not="Name: tcrv_rvv_i32_vadd_microkernel_static_object_i32_vadd_rvv_first_slice"
// RUN: tcrv-translate --help | FileCheck %s --check-prefix=HELP

module @rvv_microkernel_object_input {
  tcrv.exec.kernel @static_object_i32_vadd {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    tcrv.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_mabi = "lp64d",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      status = "available",
      value = "rv64gcv"
    }
    tcrv.exec.capability @rvv_toolchain_mabi {
      id = "rvv.toolchain.mabi",
      kind = "toolchain",
      status = "available",
      value = "lp64d"
    }
  }
}

// OBJ: Format: elf
// OBJ: Arch: riscv
// OBJ: Type: Relocatable

// DIRECTOBJ: Name: tcrv_rvv_i32_vadd_microkernel_static_object_i32_vadd_rvv_first_slice
// GENERICOBJ: Name: tcrv_rvv_i32_vadd_microkernel_static_object_i32_vadd_rvv_first_slice

// OBJNOTE: tianchenrv.rvv.artifact=rvv-op-owned-object-artifact.v1
// OBJNOTE: owner=rvv-plugin
// OBJNOTE: artifact_kind=riscv-elf-relocatable-object
// OBJNOTE: object_route=tcrv-export-rvv-microkernel-object
// OBJNOTE: source_route=tcrv-export-rvv-microkernel-c
// OBJNOTE: selected_kernel=static_object_i32_vadd
// OBJNOTE: selected_variant=rvv_first_slice
// OBJNOTE: selected_role=direct variant
// OBJNOTE: selected_march=rv64gcv
// OBJNOTE: selected_mabi=lp64d
// OBJNOTE: selected_binary_dtype=i32
// OBJNOTE: selected_binary_source_kind=default-i32-vadd-typed-body-materialization
// OBJNOTE: selected_binary_family=i32-vadd
// OBJNOTE: selected_binary_operator=add
// OBJNOTE: selected_binary_microkernel_op=tcrv_rvv.i32_vadd_microkernel
// OBJNOTE: emitc_source_op=tcrv_rvv.i32_add
// OBJNOTE: emitc_lowerable_op_interface=TCRVEmitCLowerableOpInterface
// OBJNOTE: selected_vector_shape=i32m1
// OBJNOTE: selected_vector_sew=32
// OBJNOTE: selected_vector_lmul=m1
// OBJNOTE: selected_tail_policy=agnostic
// OBJNOTE: selected_mask_policy=agnostic
// OBJNOTE: selected_vector_type=vint32m1_t
// OBJNOTE: selected_vector_suffix=i32m1
// OBJNOTE: selected_setvl_suffix=e32m1
// OBJNOTE: runtime_avl_source=runtime-element-count-abi-parameter
// OBJNOTE: runtime_avl_role=runtime-element-count
// OBJNOTE: runtime_vl_source=tcrv_rvv.setvl
// OBJNOTE: runtime_vl_scope=tcrv_rvv.with_vl
// OBJNOTE: runtime_abi=rvv-i32-vadd-runtime-callable-c-abi.v1
// OBJNOTE: runtime_abi_kind=rvv-runtime-callable-c-abi
// OBJNOTE: runtime_abi_name=rvv-i32-vadd-runtime-callable-c-function.v1
// OBJNOTE: runtime_glue_role=runtime-callable-i32-vadd-function
// OBJNOTE: descriptor_compute_authority=quarantined-after-typed-rvv-source-authority
// OBJNOTE: runtime_abi_parameter[0]=c_name=lhs,c_type=const int32_t *,role=lhs-input-buffer,ownership=target-export-abi-owned
// OBJNOTE: runtime_abi_parameter[1]=c_name=rhs,c_type=const int32_t *,role=rhs-input-buffer,ownership=target-export-abi-owned
// OBJNOTE: runtime_abi_parameter[2]=c_name=out,c_type=int32_t *,role=output-buffer,ownership=target-export-abi-owned
// OBJNOTE: runtime_abi_parameter[3]=c_name=n,c_type=size_t,role=runtime-element-count,ownership=target-export-abi-owned

// SOURCE: /* TianChen-RV RVV runtime-callable microkernel C export. */
// SOURCE: /* selected_kernel: @static_object_i32_vadd */
// SOURCE: /* selected_role: direct variant */
// SOURCE: /* selected_march: rv64gcv */
// SOURCE: /* selected_mabi: lp64d */
// SOURCE: /* artifact_kind: runtime-callable-c-source */
// SOURCE: // tcrv_emitc.source_authority=mlir_emitc_cpp_emitter
// SOURCE: static void tcrv_rvv_i32_vadd_microkernel_static_object_i32_vadd_rvv_first_slice__tcrv_emitc_body
// SOURCE: __riscv_vadd_vv_i32m1
// SOURCE: void tcrv_rvv_i32_vadd_microkernel_static_object_i32_vadd_rvv_first_slice

// OBJECT-MISSING-SOURCE: TianChen-RV target source artifact export failed:
// OBJECT-MISSING-SOURCE-SAME: composite target artifact route 'tcrv-export-rvv-microkernel-object' runtime ABI role contract preflight failed
// OBJECT-MISSING-SOURCE-SAME: tcrv_rvv.lowering_boundary for @rvv_first_slice requires selected RVV binary source identity before target artifact export

// HELP-DAG: --tcrv-export-rvv-microkernel-object
// HELP-DAG: --tcrv-export-rvv-i32-vsub-microkernel-object
// HELP-DAG: --tcrv-export-rvv-i32-vmul-microkernel-object
