import os
import subprocess
import tempfile
from pathlib import Path

import lit.formats
import lit.llvm

lit.llvm.initialize(lit_config, config)
llvm_config = lit.llvm.llvm_config

config.name = "Weft"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir", ".test"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.weft_obj_root

llvm_config.use_default_substitutions()
llvm_config.with_environment("PATH", config.weft_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

def has_local_rvv_object_clang():
    clang = Path(config.llvm_tools_dir) / "clang"
    if not clang.exists():
        return False
    source = """\
#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
void weft_lit_rvv_probe(const int32_t *lhs, int32_t *out, size_t n) {
  size_t vl = __riscv_vsetvl_e32m1(n);
  vint32m1_t lhs_vec = __riscv_vle32_v_i32m1(lhs, vl);
  __riscv_vse32_v_i32m1(out, lhs_vec, vl);
}
"""
    with tempfile.TemporaryDirectory(prefix="weft-lit-rvv-object-") as tmp:
        src = Path(tmp) / "probe.c"
        obj = Path(tmp) / "probe.o"
        src.write_text(source)
        try:
            result = subprocess.run(
                [
                    str(clang),
                    "-target",
                    "riscv64",
                    "-O2",
                    "-march=rv64gcv",
                    "-mabi=lp64d",
                    "-c",
                    str(src),
                    "-o",
                    str(obj),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0 and obj.exists() and obj.stat().st_size > 0

def has_local_riscv_object_clang():
    clang = Path(config.llvm_tools_dir) / "clang"
    if not clang.exists():
        return False
    source = """\
#include <stddef.h>
#include <stdint.h>
void weft_lit_riscv_probe(const int32_t *lhs, const int32_t *rhs,
                          int32_t *out, size_t n) {
  for (size_t index = 0; index < n; ++index)
    out[index] = lhs[index] + rhs[index];
}
"""
    with tempfile.TemporaryDirectory(prefix="weft-lit-riscv-object-") as tmp:
        src = Path(tmp) / "probe.c"
        obj = Path(tmp) / "probe.o"
        src.write_text(source)
        try:
            result = subprocess.run(
                [
                    str(clang),
                    "-target",
                    "riscv64",
                    "-O2",
                    "-march=rv64gc",
                    "-mabi=lp64d",
                    "-c",
                    str(src),
                    "-o",
                    str(obj),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0 and obj.exists() and obj.stat().st_size > 0

def has_local_native_clangxx():
    clangxx = Path(config.llvm_tools_dir) / "clang++"
    if not clangxx.exists():
        return False
    source = """\
#include <cstdio>
int main() {
  std::printf("weft-lit-native-clangxx-ok\\n");
  return 0;
}
"""
    with tempfile.TemporaryDirectory(prefix="weft-lit-native-clangxx-") as tmp:
        src = Path(tmp) / "probe.cpp"
        exe = Path(tmp) / "probe"
        src.write_text(source)
        try:
            result = subprocess.run(
                [
                    str(clangxx),
                    "-std=c++17",
                    str(src),
                    "-o",
                    str(exe),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
                check=False,
            )
            if result.returncode != 0 or not exe.exists():
                return False
            run = subprocess.run(
                [str(exe)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return run.returncode == 0

if has_local_riscv_object_clang():
    config.available_features.add("weft-local-riscv-object-clang")

if has_local_rvv_object_clang():
    config.available_features.add("weft-local-rvv-object-clang")

if has_local_native_clangxx():
    config.available_features.add("weft-local-native-clangxx")

tool_dirs = [config.weft_tools_dir, config.llvm_tools_dir]
llvm_config.add_tool_substitutions(
    [
        "weft-opt",
        "weft-translate",
        "mlir-translate",
        "FileCheck",
        "clang",
        "clang++",
        "llvm-readobj",
        "weft-capability-model-test",
        "weft-load-time-resolution-test",
        "weft-construction-protocol-common-test",
        "weft-plugin-registry-test",
        "weft-plugin-variant-cost-test",
        "weft-plugin-variant-legality-test",
        "weft-plugin-variant-proposal-test",
        "weft-emission-readiness-test",
        "weft-offload-extension-plugin-test",
        "weft-rvv-dialect-test",
        "weft-rvv-formula-decision-test",
        "weft-rvv-low-precision-lmul-selection-test",
        "weft-rvv-extension-plugin-test",
        "weft-scalar-extension-plugin-test",
        "weft-target-artifact-export-test",
        "weft-variant-dispatch-synthesis-test",
        "weft-variant-materialization-test",
        "weft-variant-selection-test",
    ],
    tool_dirs,
)
