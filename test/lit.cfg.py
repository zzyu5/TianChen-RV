import os

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

tool_dirs = [config.tianchenrv_tools_dir, config.llvm_tools_dir]
llvm_config.add_tool_substitutions(
    ["tcrv-opt", "FileCheck", "tianchenrv-plugin-registry-test"], tool_dirs
)
