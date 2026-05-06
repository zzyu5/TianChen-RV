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

add_executable(tianchenrv-plugin-registry-test
  Plugin/PluginRegistryTest.cpp
)

target_link_libraries(tianchenrv-plugin-registry-test
  PRIVATE
  TianChenRVInitAll
  TianChenRVPlugin
  MLIRIR
  MLIRSupport
)

add_lit_testsuite(check-tianchenrv "Running TianChen-RV lit tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  tcrv-opt
  tianchenrv-plugin-registry-test
  ARGS
  -sv
)

set_target_properties(check-tianchenrv PROPERTIES FOLDER "Tests")
