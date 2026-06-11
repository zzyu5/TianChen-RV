// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/__riscv_vwredsumu_vs_u16mf2_u32m1/s//__riscv_vwredsum_vs_i16mf2_i32m1/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-VWREDSUM
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_primitive.source_signedness\", value = \"unsigned\"/s//tcrv_rvv.low_precision_primitive.source_signedness\", value = \"signed\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SIGN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.source_signedness\", value = \"unsigned\"/s//tcrv_rvv.low_precision_resource.source_signedness\", value = \"signed\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RESOURCE-SIGN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.primitive_product_reduction_chain_relation\", value = \"unsigned-u8mf4xu8mf4-to-u16mf2-reduce-plus-u32-scalar-to-u32\"/s//tcrv_rvv.low_precision_resource.primitive_product_reduction_chain_relation\", value = \"metadata-derived-product-reduction\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RESOURCE-CHAIN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.reduction_candidate_fact\", value = \"resource-candidate-widening-reduction:unsigned-u8mf4xu8mf4-to-u16mf2-reduce-plus-u32-scalar-to-u32:__riscv_vwredsumu_vs_u16mf2_u32m1:store-vl=1\"/s//tcrv_rvv.low_precision_resource.reduction_candidate_fact\", value = \"target-metadata-unsigned-reduction\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RESOURCE-REDUCTION-CANDIDATE
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_primitive.accumulator_dtype\", value = \"u32\"/s//tcrv_rvv.low_precision_primitive.accumulator_dtype\", value = \"i32\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PRIM-ACC
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_primitive.source_extension\", value = \"zero-extend-u8-to-u16-product\"/s//tcrv_rvv.low_precision_primitive.source_extension\", value = \"sign-extend-i8-to-i16-product\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-EXT
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.c_type_mapping\", value = \"vl:size_t,source:unsigned-e8mf4,product:unsigned-e16mf2,seed:unsigned-u32,result:unsigned-e32m1\"/s//tcrv_rvv.c_type_mapping\", value = \"vl:size_t,source:signed-e8mf4,product:signed-e16mf2,seed:signed-i32,result:signed-e32m1\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CTYPE
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_primitive.tail_policy\", value = \"agnostic\"/s//tcrv_rvv.low_precision_primitive.tail_policy\", value = \"undisturbed\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PRIM-TAIL
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_primitive.product_sew\", value = \"16\"/s//tcrv_rvv.low_precision_primitive.product_sew_missing\", value = \"16\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=MISSING-PRIM-PRODUCT-SEW

// Explicit selected-body input for one bounded Stage 2 unsigned low-precision
// product-reduction chain. The typed tcrv_rvv body carries u8 source loads,
// the u16 product intermediate, the u16-to-u32 standalone reduction, the u32
// scalar seed/result boundary, runtime AVL/VL, and provider-derived
// widening-reduction primitive facts before target artifact export.

module {
  tcrv.exec.kernel @explicit_selected_body_unsigned_product_reduce_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_unsigned_product_reduce attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unsigned-widening-product-reduce-add:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unsigned-widening-product-reduce-add:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const uint32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unsigned-widening-product-reduce-add:acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "uint32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unsigned-widening-product-reduce-add:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unsigned-widening-product-reduce-add:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_unsigned_product_reduce, sew = 32 : i64, source_kernel = "explicit_selected_body_unsigned_product_reduce_kernel", status = "selected-lowering-boundary"} {
        %lhs_vec = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<ui8, "mf4">
        %rhs_vec = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<ui8, "mf4">
        %product = tcrv_rvv.widening_product %lhs_vec, %rhs_vec, %vl {kind = "unsigned_widening_product", product_relation = "unsigned-u8mf4xu8mf4-to-u16mf2"} : !tcrv_rvv.vector<ui8, "mf4">, !tcrv_rvv.vector<ui8, "mf4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<ui16, "mf2">
        %reduced = tcrv_rvv.standalone_reduce %product, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "unsigned_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<ui16, "mf2">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<ui32, "m1">
        tcrv_rvv.store %out, %reduced, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<ui32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @explicit_selected_body_rvv_unsigned_product_reduce {origin = "rvv-plugin", policy = "explicit-selected-body-unsigned-widening-product-reduce-add-case"}
      tcrv.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-unsigned-widening-product-reduce-add-fallback-envelope"}
    }
  }
}

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "widening_product_reduce_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.widening_product+tcrv_rvv.standalone_reduce"}
// PLAN-SAME: {key = "tcrv_rvv.config_contract", value = "rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "tcrv_rvv.element_type", value = "u32"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "vector-rhs-load"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs,acc,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:widening_product_reduce_i8_i16_i32.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:widening_product_reduce_i8_i16_i32.v1;lhs=lhs-input-buffer:lhs:abi|src-load|wprod-lhs|src-u8mf4|hdr;rhs=rhs-input-buffer:rhs:abi|src-load|wprod-rhs|src-u8mf4|hdr;acc=accumulator-input-buffer:acc:abi|seed|wred|u32|hdr;out=output-buffer:out:abi|acc-state|store|res-u32m1|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr"}
// PLAN-SAME: {key = "tcrv_rvv.contraction_route_family_plan", value = "rvv-contraction-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-u8mf4-u16mf2-u32m1-product-reduction-contraction-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,source:unsigned-e8mf4,product:unsigned-e16mf2,seed:unsigned-u32,result:unsigned-e32m1"}
// PLAN-SAME: {key = "tcrv_rvv.source_sew", value = "8"}
// PLAN-SAME: {key = "tcrv_rvv.source_lmul", value = "mf4"}
// PLAN-SAME: {key = "tcrv_rvv.product_sew", value = "16"}
// PLAN-SAME: {key = "tcrv_rvv.product_lmul", value = "mf2"}
// PLAN-SAME: {key = "tcrv_rvv.product_vector_c_type", value = "vuint16mf2_t"}
// PLAN-SAME: {key = "tcrv_rvv.accumulator_sew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.accumulator_lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.result_sew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.result_lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.widening_product_relation", value = "unsigned-u8mf4xu8mf4-to-u16mf2"}
// PLAN-SAME: {key = "tcrv_rvv.product_reduction_chain_relation", value = "unsigned-u8mf4xu8mf4-to-u16mf2-reduce-plus-u32-scalar-to-u32"}
// PLAN-SAME: {key = "tcrv_rvv.widening_product_intrinsic", value = "__riscv_vwmulu_vv_u16mf2"}
// PLAN-SAME: {key = "tcrv_rvv.widening_reduction_intrinsic", value = "__riscv_vwredsumu_vs_u16mf2_u32m1"}
// PLAN-SAME: {key = "tcrv_rvv.scalar_seed_splat_intrinsic", value = "__riscv_vmv_v_x_u32m1"}
// PLAN-SAME: {key = "tcrv_rvv.reduction_store_vl", value = "1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.contract", value = "rvv-low-precision-widening-primitive-facts.v1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.kind", value = "unsigned-u8mf4xu8mf4-to-u16mf2-product-u32m1-reduction.v1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.source_dtype", value = "u8"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.source_signedness", value = "unsigned"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.source_load", value = "unit-stride-byte-load"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.source_extension", value = "zero-extend-u8-to-u16-product"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.product_dtype", value = "u16"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.accumulator_dtype", value = "u32"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.result_dtype", value = "u32"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.source_sew", value = "8"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.source_lmul", value = "mf4"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.product_sew", value = "16"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.product_lmul", value = "mf2"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.accumulator_sew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.accumulator_lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.result_sew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.result_lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.tail_policy", value = "agnostic"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.mask_policy", value = "agnostic"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.runtime_avl_source", value = "runtime_abi:n"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.resource_owner_mirror_source", value = "provider-owned-low-precision-contraction-resource-selection.v1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.candidate_set", value = "rvv-low-precision-product-reduction-resource-candidate-set.v1[signed-i8mf4-i16mf2-i32m1:u1-vector-carry,unsigned-u8mf4-u16mf2-u32m1:u1-vector-carry]"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.selected_candidate", value = "rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-add,unsigned-u8mf4-u16mf2-u32m1,u1]"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.source_dtype", value = "u8"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.source_signedness", value = "unsigned"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.accumulator_dtype", value = "u32"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.result_dtype", value = "u32"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.memory_form", value = "unit-stride-widening-product-reduce-add"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.primitive_chain_kind", value = "unsigned-u8mf4xu8mf4-to-u16mf2-product-u32m1-vwredsumu.v1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.widening_product_candidate_fact", value = "resource-candidate-widening-product:unsigned-u8mf4xu8mf4-to-u16mf2:__riscv_vwmulu_vv_u16mf2"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.reduction_candidate_fact", value = "resource-candidate-widening-reduction:unsigned-u8mf4xu8mf4-to-u16mf2-reduce-plus-u32-scalar-to-u32:__riscv_vwredsumu_vs_u16mf2_u32m1:store-vl=1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.primitive_reduction_intrinsic", value = "__riscv_vwredsumu_vs_u16mf2_u32m1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.target_capability_provider_mirror", value = "selected_capability_provider_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-widening-product-reduce-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_unsigned_product_reduce

// HEADER: tianchenrv.rvv.selected_variant: @explicit_selected_body_rvv_unsigned_product_reduce
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-widening-product-reduce-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.config_contract: rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1
// HEADER: tianchenrv.rvv.memory_form: vector-rhs-load
// HEADER: tianchenrv.rvv.low_precision_primitive.payload_mirror.source_sew: 8
// HEADER: tianchenrv.rvv.low_precision_primitive.payload_mirror.product_lmul: mf2
// HEADER: tianchenrv.rvv.low_precision_primitive.payload_mirror.tail_policy: agnostic
// HEADER: tianchenrv.rvv.low_precision_primitive.payload_mirror.runtime_avl_source: runtime_abi:n
// HEADER: tianchenrv.rvv.low_precision_resource.resource_owner_mirror.source: provider-owned-low-precision-contraction-resource-selection.v1
// HEADER: tianchenrv.rvv.low_precision_resource.selected_candidate: rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-add,unsigned-u8mf4-u16mf2-u32m1,u1]
// HEADER: tianchenrv.rvv.low_precision_resource.source_signedness: unsigned
// HEADER: tianchenrv.rvv.low_precision_resource.accumulator_dtype: u32
// HEADER: tianchenrv.rvv.low_precision_resource.primitive_chain_kind: unsigned-u8mf4xu8mf4-to-u16mf2-product-u32m1-vwredsumu.v1
// HEADER: tianchenrv.rvv.low_precision_resource.widening_product_candidate_fact: resource-candidate-widening-product:unsigned-u8mf4xu8mf4-to-u16mf2:__riscv_vwmulu_vv_u16mf2
// HEADER: tianchenrv.rvv.low_precision_resource.reduction_candidate_fact: resource-candidate-widening-reduction:unsigned-u8mf4xu8mf4-to-u16mf2-reduce-plus-u32-scalar-to-u32:__riscv_vwredsumu_vs_u16mf2_u32m1:store-vl=1
// HEADER: tianchenrv.rvv.target_leaf_profile: rvv-v1-u8mf4-u16mf2-u32m1-product-reduction-contraction-leaf-profile.v1
// HEADER: tianchenrv.rvv.c_type_mapping: vl:size_t,source:unsigned-e8mf4,product:unsigned-e16mf2,seed:unsigned-u32,result:unsigned-e32m1
// HEADER: void tcrv_emitc_explicit_selected_body_unsigned_product_reduce_kernel_explicit_selected_body_rvv_unsigned_product_reduce(const uint8_t *lhs, const uint8_t *rhs, const uint32_t *acc, uint32_t *out, size_t n);

// STALE-VWREDSUM: candidate tcrv_rvv.widening_reduction_intrinsic provenance must mirror selected typed RVV product-reduction low-precision widening-reduction primitive widening reduction intrinsic '__riscv_vwredsumu_vs_u16mf2_u32m1'
// STALE-VWREDSUM-SAME: but was '__riscv_vwredsum_vs_i16mf2_i32m1'

// STALE-SIGN: candidate tcrv_rvv.low_precision_primitive.source_signedness provenance must mirror selected typed RVV product-reduction low-precision widening-reduction primitive source signedness 'unsigned'
// STALE-SIGN-SAME: but was 'signed'

// STALE-RESOURCE-SIGN: candidate tcrv_rvv.low_precision_resource.source_signedness provenance must mirror provider-selected low-precision direct-contraction resource source signedness 'unsigned'
// STALE-RESOURCE-SIGN-SAME: but was 'signed'

// STALE-RESOURCE-CHAIN: candidate tcrv_rvv.low_precision_resource.primitive_product_reduction_chain_relation provenance must mirror provider-selected low-precision direct-contraction resource primitive product-reduction chain relation 'unsigned-u8mf4xu8mf4-to-u16mf2-reduce-plus-u32-scalar-to-u32'
// STALE-RESOURCE-CHAIN-SAME: but was 'metadata-derived-product-reduction'

// STALE-RESOURCE-REDUCTION-CANDIDATE: candidate tcrv_rvv.low_precision_resource.reduction_candidate_fact provenance must mirror provider-selected low-precision direct-contraction resource widening reduction candidate fact 'resource-candidate-widening-reduction:unsigned-u8mf4xu8mf4-to-u16mf2-reduce-plus-u32-scalar-to-u32:__riscv_vwredsumu_vs_u16mf2_u32m1:store-vl=1'
// STALE-RESOURCE-REDUCTION-CANDIDATE-SAME: but was 'target-metadata-unsigned-reduction'

// STALE-PRIM-ACC: candidate tcrv_rvv.low_precision_primitive.accumulator_dtype provenance must mirror selected typed RVV product-reduction low-precision widening-reduction primitive accumulator dtype 'u32'
// STALE-PRIM-ACC-SAME: but was 'i32'

// STALE-EXT: candidate tcrv_rvv.low_precision_primitive.source_extension provenance must mirror selected typed RVV product-reduction low-precision widening-reduction primitive source extension 'zero-extend-u8-to-u16-product'
// STALE-EXT-SAME: but was 'sign-extend-i8-to-i16-product'

// STALE-CTYPE: candidate tcrv_rvv.c_type_mapping provenance must mirror selected typed RVV widening dot route type mapping summary 'vl:size_t,source:unsigned-e8mf4,product:unsigned-e16mf2,seed:unsigned-u32,result:unsigned-e32m1'
// STALE-CTYPE-SAME: but was 'vl:size_t,source:signed-e8mf4,product:signed-e16mf2,seed:signed-i32,result:signed-e32m1'

// STALE-PRIM-TAIL: candidate tcrv_rvv.low_precision_primitive.tail_policy provenance must mirror selected typed RVV product-reduction low-precision widening-reduction primitive tail policy 'agnostic'
// STALE-PRIM-TAIL-SAME: but was 'undisturbed'

// MISSING-PRIM-PRODUCT-SEW: candidate metadata must carry tcrv_rvv.low_precision_primitive.product_sew provenance for selected typed RVV product-reduction low-precision widening-reduction primitive product SEW
