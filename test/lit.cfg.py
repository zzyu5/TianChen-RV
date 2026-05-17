import os
import subprocess
import tempfile
from pathlib import Path

import lit.formats
import lit.llvm

lit.llvm.initialize(lit_config, config)
llvm_config = lit.llvm.llvm_config

config.name = "TianChenRV"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir", ".test"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.tianchenrv_obj_root

llvm_config.use_default_substitutions()
llvm_config.with_environment("PATH", config.tianchenrv_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

def has_local_rvv_object_clang():
    clang = Path(config.llvm_tools_dir) / "clang"
    if not clang.exists():
        return False
    source = """\
#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
void tcrv_lit_rvv_probe(const int32_t *lhs, int32_t *out, size_t n) {
  size_t vl = __riscv_vsetvl_e32m1(n);
  vint32m1_t lhs_vec = __riscv_vle32_v_i32m1(lhs, vl);
  __riscv_vse32_v_i32m1(out, lhs_vec, vl);
}
"""
    with tempfile.TemporaryDirectory(prefix="tcrv-lit-rvv-object-") as tmp:
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
void tcrv_lit_riscv_probe(const int32_t *lhs, const int32_t *rhs,
                          int32_t *out, size_t n) {
  for (size_t index = 0; index < n; ++index)
    out[index] = lhs[index] + rhs[index];
}
"""
    with tempfile.TemporaryDirectory(prefix="tcrv-lit-riscv-object-") as tmp:
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
  std::printf("tcrv-lit-native-clangxx-ok\\n");
  return 0;
}
"""
    with tempfile.TemporaryDirectory(prefix="tcrv-lit-native-clangxx-") as tmp:
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
    config.available_features.add("tianchenrv-local-riscv-object-clang")

if has_local_rvv_object_clang():
    config.available_features.add("tianchenrv-local-rvv-object-clang")

if has_local_native_clangxx():
    config.available_features.add("tianchenrv-local-native-clangxx")

tool_dirs = [config.tianchenrv_tools_dir, config.llvm_tools_dir]
llvm_config.add_tool_substitutions(
    [
        "tcrv-opt",
        "tcrv-translate",
        "FileCheck",
        "clang",
        "clang++",
        "llvm-readobj",
        "tianchenrv-capability-model-test",
        "tianchenrv-construction-protocol-common-test",
        "tianchenrv-plugin-registry-test",
        "tianchenrv-plugin-variant-cost-test",
        "tianchenrv-plugin-variant-legality-test",
        "tianchenrv-plugin-variant-proposal-test",
        "tianchenrv-emission-readiness-test",
        "tianchenrv-offload-extension-plugin-test",
        "tianchenrv-rvv-dialect-test",
        "tianchenrv-rvv-extension-plugin-test",
        "tianchenrv-scalar-extension-plugin-test",
        "tianchenrv-target-artifact-export-test",
        "tianchenrv-variant-dispatch-synthesis-test",
        "tianchenrv-variant-materialization-test",
        "tianchenrv-variant-selection-test",
    ],
    tool_dirs,
)
