// declared_instance_hash determinism + profile-equivalence, and NoViableVariant
// coverage, for the compile-time selection attribution JSONL ([D-4] (1) / [D-2a]).
// The hash is taken over the EXPANDED normalized fact set sorted by id with the
// local IR symbol name EXCLUDED, so two kernels carrying the SAME facts hash the
// same regardless of declaration order or symbol names (canon 7d781994:
// "profile == explicit list ==> same hash"). No materialize step: capability-only
// kernels plan to NoViableVariant (chosen=null, reason=null) directly.
//
// RUN: weft-opt %s "--weft-select-variants=attribution-jsonl=%t.jsonl attribution-jsonl-no-timestamp" -o /dev/null
// RUN: FileCheck %s --input-file=%t.jsonl

module {
  // No variants at all => NoViableVariant: chosen=null, reason=null, empty
  // candidates / keys_evaluated. Still emitted so the C_attr^CT denominator is
  // honest.
  // CHECK: {"candidates":[],"chosen":null,"declared_instance_hash":"{{[0-9a-f]+}}","kernel":"no_viable_kernel","keys_evaluated":{},"reason":null,"ts":"0"}
  weft.exec.kernel @no_viable_kernel {
    weft.exec.capability @lonely_toolchain {
      id = "generic.only",
      kind = "toolchain",
      status = "available"
    }
  }

  // Same two capability facts (aaa.one, bbb.two) as @hash_order_ba below, but
  // declared in a different order under different symbol names. Capture the hash.
  // CHECK: {"candidates":[],"chosen":null,"declared_instance_hash":"[[IHASH:[0-9a-f]+]]","kernel":"hash_order_ab","keys_evaluated":{},"reason":null,"ts":"0"}
  weft.exec.kernel @hash_order_ab {
    weft.exec.capability @cap_a {
      id = "aaa.one",
      kind = "toolchain",
      status = "available"
    }
    weft.exec.capability @cap_b {
      id = "bbb.two",
      kind = "toolchain",
      status = "available"
    }
  }

  // Reversed declaration order + different symbol names, SAME facts => the hash
  // MUST equal the captured [[IHASH]]. This is the profile-equivalence guarantee.
  // CHECK: {"candidates":[],"chosen":null,"declared_instance_hash":"[[IHASH]]","kernel":"hash_order_ba","keys_evaluated":{},"reason":null,"ts":"0"}
  weft.exec.kernel @hash_order_ba {
    weft.exec.capability @sym_z {
      id = "bbb.two",
      kind = "toolchain",
      status = "available"
    }
    weft.exec.capability @sym_y {
      id = "aaa.one",
      kind = "toolchain",
      status = "available"
    }
  }
}
