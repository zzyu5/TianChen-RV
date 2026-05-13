// RUN: tcrv-translate --tcrv-export-rvv-microkernel-c %s | FileCheck %s --check-prefix=ALT --implicit-check-not="while (offset < n)"
// RUN: tcrv-translate --tcrv-export-target-source-artifact %s | FileCheck %s --check-prefix=ALT --implicit-check-not="while (offset < n)"
// RUN: tcrv-translate --tcrv-export-rvv-microkernel-header %s | FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv" --implicit-check-not=") {"
// RUN: sed 's#c_name = "lhs", c_type = "const int32_t \*", ownership = "target-export-abi-owned", role = "lhs-input-buffer"#c_name = "a", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"#' %s | not tcrv-translate --tcrv-export-rvv-microkernel-c 2>&1 | FileCheck %s --check-prefix=STALE-NAME --implicit-check-not="#include <riscv_vector.h>"
// RUN: sed 's#c_name = "lhs", c_type = "const int32_t \*", ownership = "target-export-abi-owned", role = "lhs-input-buffer"#c_name = "a", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"#' %s | not tcrv-translate --tcrv-export-rvv-microkernel-header 2>&1 | FileCheck %s --check-prefix=HEADER-STALE-NAME --implicit-check-not="#ifndef"
// RUN: sed 's#c_name = "lhs", c_type = "const int32_t \*", ownership = "target-export-abi-owned", role = "lhs-input-buffer"#c_name = "a", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"#' %s | not tcrv-translate --tcrv-export-target-artifact 2>&1 | FileCheck %s --check-prefix=OBJECT-STALE-NAME --implicit-check-not="ELF"
// RUN: sed 's#c_name = "rhs", c_type = "const int32_t \*", ownership = "target-export-abi-owned", role = "rhs-input-buffer"#c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"#' %s | not tcrv-translate --tcrv-export-rvv-microkernel-c 2>&1 | FileCheck %s --check-prefix=DUPLICATE --implicit-check-not="#include <riscv_vector.h>"
// RUN: sed 's/c_name = "len", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"/c_name = "len", c_type = "size_t", ownership = "target-export-abi-owned", role = "unknown-length"/' %s | not tcrv-translate --tcrv-export-rvv-microkernel-c 2>&1 | FileCheck %s --check-prefix=UNKNOWN --implicit-check-not="#include <riscv_vector.h>"
// RUN: sed 's/c_name = "out", c_type = "int32_t \*", ownership = "target-export-abi-owned", role = "output-buffer"/c_name = "out", c_type = "int32_t *", ownership = "ir-modeled", role = "output-buffer"/' %s | not tcrv-translate --tcrv-export-rvv-microkernel-c 2>&1 | FileCheck %s --check-prefix=OWNERSHIP --implicit-check-not="#include <riscv_vector.h>"
// RUN: sed 's/c_name = "len", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"/c_name = "len", c_type = "uint64_t", ownership = "target-export-abi-owned", role = "runtime-element-count"/' %s | not tcrv-translate --tcrv-export-rvv-microkernel-c 2>&1 | FileCheck %s --check-prefix=TYPE --implicit-check-not="#include <riscv_vector.h>"
// RUN: sed '/^    tcrv.exec.runtime_param @abi_runtime_element_count/,/^    }/d' %s | not tcrv-translate --tcrv-export-rvv-microkernel-c 2>&1 | FileCheck %s --check-prefix=MISSING-RUNTIME-N --implicit-check-not="#include <riscv_vector.h>"
// RUN: sed '/^    tcrv.exec.runtime_param @abi_runtime_element_count/,/^    }/ s/c_name = "len",//' %s | not tcrv-translate --tcrv-export-rvv-microkernel-c 2>&1 | FileCheck %s --check-prefix=MISSING-RUNTIME-CNAME --implicit-check-not="#include <riscv_vector.h>"

module @rvv_runtime_abi_role_binding {
  tcrv.exec.kernel @abi_names {
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
    tcrv.exec.mem_window @abi_lhs_input_buffer {
      abi_role = "lhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.mem_window @abi_rhs_input_buffer {
      abi_role = "rhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.mem_window @abi_output_buffer {
      abi_role = "output-buffer",
      access = "write",
      binding = "kernel-argument",
      c_type = "int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.runtime_param @abi_runtime_element_count {
      abi_role = "runtime-element-count",
      c_name = "len",
      c_type = "size_t",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-scalar"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      condition = "rvv_capability_properties_available",
      guard = "plugin_local_rvv_property_evidence",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    tcrv.exec.diagnostic {
      message = "static RVV microkernel path selected by role-binding fixture",
      origin = "rvv-plugin",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @rvv_first_slice
    }
    tcrv_rvv.lowering_boundary {
      capability_summary = "rvv",
      emitc_lowerable_op_interface = "TCRVEmitCLowerableOpInterface",
      emitc_source_op = "tcrv_rvv.i32_add",
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      role = "direct variant",
      selected_binary_dtype = "i32",
      selected_binary_family = "i32-vadd",
      selected_binary_microkernel_op = "tcrv_rvv.i32_vadd_microkernel",
      selected_binary_operator = "add",
      selected_binary_source_kind = "direct-typed-microkernel-body",
      selected_variant = @rvv_first_slice,
      source_kernel = "abi_names",
      status = "unsupported",
      unsupported_reason = "RVV lowering boundary is pre-executable metadata only"
    }
    tcrv_rvv.i32_vadd_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_mabi = "lp64d",
      selected_variant = @rvv_first_slice,
      source_kernel = "abi_names"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs = tcrv_rvv.i32_load %vl {buffer_role = "lhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1

        %rhs = tcrv_rvv.i32_load %vl {buffer_role = "rhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1

        %sum = tcrv_rvv.i32_add %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1

        tcrv_rvv.i32_store %sum, %vl {buffer_role = "output-buffer"} : !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.diagnostic {
      artifact_kind = "runtime-callable-c-source",
      emission_kind = "rvv-explicit-i32-vadd-microkernel-c-source",
      lowering_boundary = "tcrv_rvv.lowering_boundary",
      lowering_pipeline = "tcrv-export-rvv-microkernel-c",
      message = "callable ABI parameters mirror execution IR boundaries",
      origin = "rvv-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@rvv],
      role = "direct variant",
      runtime_abi = "rvv-i32-vadd-runtime-callable-c-abi.v1",
      runtime_abi_kind = "rvv-runtime-callable-c-abi",
      runtime_abi_name = "rvv-i32-vadd-runtime-callable-c-function.v1",
      runtime_abi_parameters = [{c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"}, {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"}, {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"}, {c_name = "len", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"}],
      runtime_glue_role = "runtime-callable-i32-vadd-function",
      selected_plan_metadata = [
        {name = "tcrv_rvv.selected_binary_dtype", value = "i32", role = "typed-rvv-binary-source", note = "typed RVV family-op metadata selected by the RVV plugin for the common EmitC route; not descriptor-owned computation, runtime correctness evidence, or performance evidence"},
        {name = "tcrv_rvv.selected_binary_family", value = "i32-vadd", role = "typed-rvv-binary-source", note = "typed RVV family-op metadata selected by the RVV plugin for the common EmitC route; not descriptor-owned computation, runtime correctness evidence, or performance evidence"},
        {name = "tcrv_rvv.selected_binary_operator", value = "add", role = "typed-rvv-binary-source", note = "typed RVV family-op metadata selected by the RVV plugin for the common EmitC route; not descriptor-owned computation, runtime correctness evidence, or performance evidence"},
        {name = "tcrv_rvv.emitc_source_op", value = "tcrv_rvv.i32_add", role = "typed-rvv-emitc-source-op", note = "typed RVV arithmetic op used as the source operation for the common EmitC lowerable route; not a lowering descriptor"},
        {name = "tcrv_rvv.emitc_lowerable_op_interface", value = "TCRVEmitCLowerableOpInterface", role = "typed-rvv-emitc-source-op", note = "generated RVV op interface queried before building the common EmitC lowerable route; not descriptor-selected computation"},
        {name = "tcrv_rvv.runtime_element_count_c_name", value = "len", role = "rvv-runtime-control-name-boundary", note = "runtime ABI/control C name resolved from tcrv.exec runtime boundary; not a compile-time vector-shape config or descriptor element_count"},
        {name = "tcrv_rvv.selected_vector_shape", value = "i32m1", role = "selected-rvv-vector-shape-config", note = "compile-time RVV vector-shape config selected by the RVV plugin and validated against capabilities; not base lane capacity, runtime AVL/VL, or descriptor element_count"},
        {name = "tcrv_rvv.selected_vector_sew", value = "32", role = "selected-rvv-vector-shape-config", note = "compile-time RVV vector-shape config selected by the RVV plugin and validated against capabilities; not base lane capacity, runtime AVL/VL, or descriptor element_count"},
        {name = "tcrv_rvv.selected_vector_lmul", value = "m1", role = "selected-rvv-vector-shape-config", note = "compile-time RVV vector-shape config selected by the RVV plugin and validated against capabilities; not base lane capacity, runtime AVL/VL, or descriptor element_count"},
        {name = "tcrv_rvv.selected_tail_policy", value = "agnostic", role = "selected-rvv-vector-shape-config", note = "compile-time RVV vector-shape config selected by the RVV plugin and validated against capabilities; not base lane capacity, runtime AVL/VL, or descriptor element_count"},
        {name = "tcrv_rvv.selected_mask_policy", value = "agnostic", role = "selected-rvv-vector-shape-config", note = "compile-time RVV vector-shape config selected by the RVV plugin and validated against capabilities; not base lane capacity, runtime AVL/VL, or descriptor element_count"},
        {name = "tcrv_rvv.selected_vector_type", value = "vint32m1_t", role = "selected-rvv-vector-shape-config", note = "compile-time RVV vector-shape config selected by the RVV plugin and validated against capabilities; not base lane capacity, runtime AVL/VL, or descriptor element_count"},
        {name = "tcrv_rvv.selected_vector_suffix", value = "i32m1", role = "selected-rvv-vector-shape-config", note = "compile-time RVV vector-shape config selected by the RVV plugin and validated against capabilities; not base lane capacity, runtime AVL/VL, or descriptor element_count"},
        {name = "tcrv_rvv.selected_setvl_suffix", value = "e32m1", role = "selected-rvv-vector-shape-config", note = "compile-time RVV vector-shape config selected by the RVV plugin and validated against capabilities; not base lane capacity, runtime AVL/VL, or descriptor element_count"},
        {name = "tcrv_rvv.selected_vector_sew_capability", value = "rvv.i32_m1.sew32", role = "selected-rvv-vector-shape-capability", note = "capability id backing the selected RVV vector-shape config; compile-time capability evidence, not runtime AVL/VL or descriptor element_count"},
        {name = "tcrv_rvv.selected_vector_lmul_capability", value = "rvv.i32_m1.lmul_m1", role = "selected-rvv-vector-shape-capability", note = "capability id backing the selected RVV vector-shape config; compile-time capability evidence, not runtime AVL/VL or descriptor element_count"},
        {name = "tcrv_rvv.selected_tail_policy_capability", value = "rvv.i32_m1.tail_policy.agnostic", role = "selected-rvv-vector-shape-capability", note = "capability id backing the selected RVV vector-shape config; compile-time capability evidence, not runtime AVL/VL or descriptor element_count"},
        {name = "tcrv_rvv.selected_mask_policy_capability", value = "rvv.i32_m1.mask_policy.agnostic", role = "selected-rvv-vector-shape-capability", note = "capability id backing the selected RVV vector-shape config; compile-time capability evidence, not runtime AVL/VL or descriptor element_count"},
        {name = "tcrv_rvv.runtime_avl_source", value = "runtime-element-count-abi-parameter", role = "rvv-runtime-vl-avl-boundary", note = "runtime AVL enters through the target/export-owned runtime element-count ABI parameter; runtime VL is produced by tcrv_rvv.setvl and consumed by tcrv_rvv.with_vl; neither value is a target capability fact or descriptor-local element_count"},
        {name = "tcrv_rvv.runtime_avl_role", value = "runtime-element-count", role = "rvv-runtime-vl-avl-boundary", note = "runtime AVL enters through the target/export-owned runtime element-count ABI parameter; runtime VL is produced by tcrv_rvv.setvl and consumed by tcrv_rvv.with_vl; neither value is a target capability fact or descriptor-local element_count"},
        {name = "tcrv_rvv.runtime_vl_source", value = "tcrv_rvv.setvl", role = "rvv-runtime-vl-avl-boundary", note = "runtime AVL enters through the target/export-owned runtime element-count ABI parameter; runtime VL is produced by tcrv_rvv.setvl and consumed by tcrv_rvv.with_vl; neither value is a target capability fact or descriptor-local element_count"},
        {name = "tcrv_rvv.runtime_vl_scope", value = "tcrv_rvv.with_vl", role = "rvv-runtime-vl-avl-boundary", note = "runtime AVL enters through the target/export-owned runtime element-count ABI parameter; runtime VL is produced by tcrv_rvv.setvl and consumed by tcrv_rvv.with_vl; neither value is a target capability fact or descriptor-local element_count"},
        {name = "tcrv_rvv.descriptor_element_count", value = "16", role = "rvv-descriptor-local-component-capacity", note = "bounded descriptor-local component capacity cross-checked after typed selected-plan authority; not legacy descriptor mirror, compute, ABI, source, runtime AVL/VL, hardware capacity, or performance authority"}
      ],
      severity = "info",
      status = "supported",
      target = @rvv_first_slice
    }
  }
}

// ALT: /* selected_binary_config: dtype=i32, family=i32-vadd, operator=add, shape=i32m1, sew=32, lmul=m1, tail_policy=agnostic, mask_policy=agnostic, vector_type=vint32m1_t, vector_suffix=i32m1, setvl_suffix=e32m1, runtime_element_count_c_name=len, dispatch_availability_c_name=rvv_available, descriptor_element_count=16, selected_variant=@rvv_first_slice, selected_role=direct variant */
// ALT: /* control_plane_runtime_avl: body index argument maps to target/export-owned runtime len ABI parameter */
// ALT: /* dataflow_abi_roles: lhs_load.buffer_role=lhs-input-buffer, rhs_load.buffer_role=rhs-input-buffer, store.buffer_role=output-buffer; runtime len remains the target/export-owned runtime element-count ABI parameter */
// ALT: /* dataflow_emission_step[0]: op=tcrv_rvv.i32_load, role=lhs-input-buffer, result=lhs_vec */
// ALT: /* dataflow_emission_step[3]: op=tcrv_rvv.i32_store, role=output-buffer, value=sum_vec */
// ALT: /* callable_abi_source: tcrv.exec.mem_window + tcrv.exec.runtime_param */
// ALT: /* callable_mem_window[0]: symbol=@abi_lhs_input_buffer, abi_role=lhs-input-buffer, access=read, ownership=target-export-abi-owned, c_type=const int32_t * */
// ALT: /* callable_runtime_param[0]: symbol=@abi_runtime_element_count, abi_role=runtime-element-count, c_name=len, c_type=size_t, ownership=target-export-abi-owned */
// ALT: /* runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// ALT: /* runtime_abi_parameter[1]: c_name=rhs, c_type=const int32_t *, role=rhs-input-buffer, ownership=target-export-abi-owned */
// ALT: /* runtime_abi_parameter[2]: c_name=out, c_type=int32_t *, role=output-buffer, ownership=target-export-abi-owned */
// ALT: /* runtime_abi_parameter[3]: c_name=len, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// ALT: // tcrv_emitc.source_authority=mlir_emitc_cpp_emitter
// ALT: static void tcrv_rvv_i32_vadd_microkernel_abi_names_rvv_first_slice__tcrv_emitc_body
// ALT: if (
// ALT: __riscv_vsetvl_e32m1
// ALT: __riscv_vle32_v_i32m1
// ALT: __riscv_vle32_v_i32m1
// ALT: // tcrv_emitc.source_op=tcrv_rvv.i32_add role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_i32m1
// ALT: __riscv_vadd_vv_i32m1
// ALT: __riscv_vse32_v_i32m1
// ALT: void tcrv_rvv_i32_vadd_microkernel_abi_names_rvv_first_slice

// HEADER: /* selected_body_authority: tcrv_rvv.i32_vadd_microkernel */
// HEADER: /* selected_binary_config: dtype=i32, family=i32-vadd
// HEADER-SAME: runtime_element_count_c_name=len
// HEADER-SAME: selected_role=direct variant */
// HEADER: /* control_plane_runtime_avl: body index argument maps to target/export-owned runtime len ABI parameter */
// HEADER: /* dataflow_abi_roles: lhs_load.buffer_role=lhs-input-buffer, rhs_load.buffer_role=rhs-input-buffer, store.buffer_role=output-buffer; runtime len remains the target/export-owned runtime element-count ABI parameter */
// HEADER: /* callable_mem_window[0]: symbol=@abi_lhs_input_buffer, abi_role=lhs-input-buffer
// HEADER-SAME: c_type=const int32_t *
// HEADER: /* callable_runtime_param[0]: symbol=@abi_runtime_element_count, abi_role=runtime-element-count, c_name=len, c_type=size_t, ownership=target-export-abi-owned */
// HEADER: /* runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// HEADER: /* runtime_abi_parameter[3]: c_name=len, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// HEADER: void tcrv_rvv_i32_vadd_microkernel_abi_names_rvv_first_slice(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t len);

// STALE-NAME: runtime ABI callable plan validation failed
// STALE-NAME-SAME: runtime ABI parameter role 'lhs-input-buffer' must mirror IR-backed callable ABI parameter c_name='lhs'
// HEADER-STALE-NAME: runtime ABI callable plan validation failed
// HEADER-STALE-NAME-SAME: runtime ABI parameter role 'lhs-input-buffer' must mirror IR-backed callable ABI parameter c_name='lhs'
// OBJECT-STALE-NAME: composite target artifact route 'tcrv-export-rvv-microkernel-object' runtime ABI role contract preflight failed
// OBJECT-STALE-NAME-SAME: runtime ABI parameter role 'lhs-input-buffer' must mirror IR-backed callable ABI parameter c_name='lhs'
// DUPLICATE: duplicate runtime ABI parameter role 'lhs-input-buffer'
// UNKNOWN: unsupported runtime ABI parameter role 'unknown-length'
// OWNERSHIP: route id 'tcrv-export-rvv-microkernel-c' runtime ABI parameter role 'output-buffer' must use c type 'int32_t *' and ownership 'target-export-abi-owned'
// TYPE: route id 'tcrv-export-rvv-microkernel-c' runtime ABI parameter role 'runtime-element-count' must use c type 'size_t'
// MISSING-RUNTIME-N: runtime ABI runtime_param validation failed
// MISSING-RUNTIME-N-SAME: requires exactly one tcrv.exec.runtime_param with ABI role 'runtime-element-count'
// MISSING-RUNTIME-CNAME: 'tcrv.exec.runtime_param' op requires non-empty string attribute 'c_name'
