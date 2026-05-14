// RUN: not tcrv-opt %s --tcrv-lower-source-rvv-binary-to-exec 2>&1 | FileCheck %s --check-prefix=SOURCE --implicit-check-not=tcrv.exec.kernel
// RUN: not tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec 2>&1 | FileCheck %s --check-prefix=LINALG-RVV --implicit-check-not=tcrv.exec.kernel
// RUN: not tcrv-opt %s --tcrv-lower-linalg-i32-binary-to-exec 2>&1 | FileCheck %s --check-prefix=LINALG-I32 --implicit-check-not=tcrv.exec.kernel
// RUN: not tcrv-opt %s --tcrv-lower-linalg-i32-vadd-to-exec 2>&1 | FileCheck %s --check-prefix=LINALG-VADD --implicit-check-not=tcrv.exec.kernel
// RUN: not tcrv-opt %s --tcrv-lower-vector-rvv-i32-vadd-to-exec 2>&1 | FileCheck %s --check-prefix=VECTOR-VADD --implicit-check-not=tcrv.exec.kernel
// RUN: not tcrv-opt %s --tcrv-lower-vector-rvv-i32-vsub-to-exec 2>&1 | FileCheck %s --check-prefix=VECTOR-VSUB --implicit-check-not=tcrv.exec.kernel
// RUN: not tcrv-opt %s --tcrv-lower-vector-rvv-i32-vmul-to-exec 2>&1 | FileCheck %s --check-prefix=VECTOR-VMUL --implicit-check-not=tcrv.exec.kernel

module @deleted_core_rvv_source_to_exec_pass_family

// SOURCE: Unknown command line argument '--tcrv-lower-source-rvv-binary-to-exec'
// LINALG-RVV: Unknown command line argument '--tcrv-lower-linalg-rvv-binary-to-exec'
// LINALG-I32: Unknown command line argument '--tcrv-lower-linalg-i32-binary-to-exec'
// LINALG-VADD: Unknown command line argument '--tcrv-lower-linalg-i32-vadd-to-exec'
// VECTOR-VADD: Unknown command line argument '--tcrv-lower-vector-rvv-i32-vadd-to-exec'
// VECTOR-VSUB: Unknown command line argument '--tcrv-lower-vector-rvv-i32-vsub-to-exec'
// VECTOR-VMUL: Unknown command line argument '--tcrv-lower-vector-rvv-i32-vmul-to-exec'
