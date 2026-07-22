// RUN: sed 's/, minimum_vlen = 128 : i64//' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=MISSING-MIN
// RUN: sed 's/minimum_vlen = 128 : i64/minimum_vlen = "128"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=MALFORMED-MIN
// RUN: sed 's/id = "rvv", kind/id = "not-rvv", kind/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=MISSING-PROVIDER
// RUN: sed -e 's|// ALT ||' -e 's/requires = \[@rvv\]/requires = [@rvv, @rvv_alt]/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=AMBIGUOUS
// RUN: sed 's/status = "available", minimum_vlen/status = "unavailable", minimum_vlen/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=UNAVAILABLE
// RUN: sed -e 's|// BLOCKER ||' -e 's/status = "available", minimum_vlen/status = "available", relations = #weft.capability_relations<conflicts = ["rvv.blocker"]>, minimum_vlen/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=CONFLICT
// RUN: sed 's/supported_sew = "8,16,32,64"/supported_sew = "16,32,64"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=EMPTY-LEGAL
// RUN: sed 's/supported_sew = "8,16,32,64"/supported_sew = "8,16,64"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=EMPTY-LEGAL
// RUN: sed 's/supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8"/supported_lmul = "m1"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=EMPTY-LEGAL
// RUN: sed 's/supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8"/supported_lmul = "m1,m16"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=UNKNOWN-LMUL
// RUN: sed 's/supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8"/supported_lmul = ""/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=EMPTY-FACT
// RUN: sed 's/rvv_version = "1.0"/rvv_version = "2.0"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=UNKNOWN-VERSION
// RUN: sed 's/rvv_version = "1.0"/rvv_version = ""/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=EMPTY-FACT
// RUN: sed 's/rvv_version = "1.0"/rvv_version = "0.7"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=VERSION-CONFLICT
// RUN: sed 's/rvv_version = "1.0"/required_tail_policy = "", rvv_version = "1.0"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=EMPTY-TAIL-POLICY
// RUN: sed 's/rvv_version = "1.0"/required_tail_policy = 0 : i64, rvv_version = "1.0"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=TYPED-TAIL-POLICY
// RUN: sed 's/rvv_version = "1.0"/required_tail_policy = "sideways", rvv_version = "1.0"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=UNKNOWN-TAIL-POLICY
// RUN: sed 's/rvv_version = "1.0"/required_mask_policy = "", rvv_version = "1.0"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=EMPTY-MASK-POLICY
// RUN: sed 's/rvv_version = "1.0"/required_mask_policy = 0 : i64, rvv_version = "1.0"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=TYPED-MASK-POLICY
// RUN: sed 's/rvv_version = "1.0"/required_mask_policy = "sideways", rvv_version = "1.0"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=UNKNOWN-MASK-POLICY
// RUN: weft-opt %s --weft-rvv-materialize-dequantize-row-stream-front-door | sed '0,/decode_model = "mxfp4"/s//decode_model = "nvfp4"/' | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=PARENT-CORE
// RUN: sed 's/, minimum_vlen = 128 : i64//' %s | not weft-opt --weft-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s --check-prefix=REGISTRY-REJECT

// The typed capability/formula boundary is fail-closed. These are real conversion
// entries, not parser-only or formula-only tests. Unknown and explicit-empty
// allow-list facts are invalid; absence has distinct, documented base semantics.

module {
  weft.exec.kernel @codebook_capability_negative {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available", minimum_vlen = 128 : i64, rvv_version = "1.0", supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8", supported_sew = "8,16,32,64"}
    // ALT weft.exec.capability @rvv_alt {id = "rvv.profile.alt", kind = "profile", status = "available", relations = #weft.capability_relations<provides = ["rvv"]>, minimum_vlen = 128 : i64, rvv_version = "1.0", supported_lmul = "m1,m4", supported_sew = "8,32"}
    // BLOCKER weft.exec.capability @rvv_blocker {id = "rvv.blocker", kind = "policy", status = "available"}
    weft.exec.variant @mxfp4 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "mxfp4"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// MISSING-MIN: is missing typed minimum_vlen
// MALFORMED-MIN: minimum_vlen must be a typed signless i64 attribute
// MISSING-PROVIDER: no selected requires entry satisfied RVV capability
// AMBIGUOUS: ambiguous selected providers were @rvv, @rvv_alt
// UNAVAILABLE: selected RVV capability provider @rvv satisfying id 'rvv' is unavailable
// CONFLICT: selected RVV capability provider @rvv conflicts with available provider @rvv_blocker
// EMPTY-LEGAL: codebook legal set is empty: rejected-empty-legal-set
// UNKNOWN-LMUL: property 'supported_lmul' contains unknown token 'm16'
// EMPTY-FACT: must be absent rather than an explicitly empty allow-list/fact
// UNKNOWN-VERSION: has unknown rvv_version '2.0'
// VERSION-CONFLICT: conflicts internally: rvv_version=0.7 but supported_lmul contains a fractional LMUL token
// EMPTY-TAIL-POLICY: property 'required_tail_policy' must be absent rather than an explicitly empty allow-list/fact
// TYPED-TAIL-POLICY: property 'required_tail_policy' must be a typed string attribute
// UNKNOWN-TAIL-POLICY: property 'required_tail_policy' has unknown policy token 'sideways'
// EMPTY-MASK-POLICY: property 'required_mask_policy' must be absent rather than an explicitly empty allow-list/fact
// TYPED-MASK-POLICY: property 'required_mask_policy' must be a typed string attribute
// UNKNOWN-MASK-POLICY: property 'required_mask_policy' has unknown policy token 'sideways'
// PARENT-CORE: requires parent/core decode_model construction coherence; parent carries 'nvfp4' while the core carries 'mxfp4'
// REGISTRY-REJECT: RVV dequantize-row formula rejected construction: selected RVV provider @rvv is missing typed minimum_vlen
