configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
  TIANCHENRV_SOURCE_DIR
  TIANCHENRV_BINARY_DIR
  LLVM_RUNTIME_OUTPUT_INTDIR
  LLVM_TOOLS_BINARY_DIR
)

add_lit_testsuite(check-tianchenrv "Running TianChen-RV RVV classroom lit tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  tcrv-opt
  tcrv-translate
  ARGS
  -sv
)

set_target_properties(check-tianchenrv PROPERTIES FOLDER "Tests")
