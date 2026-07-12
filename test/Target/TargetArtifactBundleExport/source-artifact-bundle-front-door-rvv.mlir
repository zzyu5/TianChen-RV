// RUN: rm -rf %t.bundle && mkdir %t.bundle
// RUN: not weft-translate --weft-source-artifact-bundle-front-door --weft-target-artifact-bundle-output-dir=%t.bundle %S/Inputs/generic-vector-source.mlir 2>&1 | FileCheck %s --check-prefix=FAIL --implicit-check-not="weft.target_artifact_bundle_export: complete" --implicit-check-not="rvv-generic-binary-add-emitc-route"
// RUN: not test -e %t.bundle/weft-target-artifact-bundle.index

// A generic vector source (no selected execution plan) must not reach default
// target artifact bundles. Positive RVV object/header/bundle coverage lives in
// the materialized typed selected-body target artifact tests.

// FAIL: Weft-RV execution plan coherence check failed for kernel <missing>
// FAIL-SAME: requires at least one weft.exec.kernel
// FAIL: Weft-RV source-artifact bundle front door failed during source-artifact front-door pipeline
