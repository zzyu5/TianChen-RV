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

add_executable(tianchenrv-plugin-variant-proposal-test
  Plugin/PluginVariantProposalTest.cpp
)

target_link_libraries(tianchenrv-plugin-variant-proposal-test
  PRIVATE
  TianChenRVInitAll
  TianChenRVPlugin
  TianChenRVSupport
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
)

add_executable(tianchenrv-plugin-variant-legality-test
  Plugin/PluginVariantLegalityTest.cpp
)

target_link_libraries(tianchenrv-plugin-variant-legality-test
  PRIVATE
  TianChenRVInitAll
  TianChenRVPlugin
  TianChenRVSupport
  MLIRIR
  MLIRParser
  MLIRSupport
)

add_executable(tianchenrv-variant-materialization-test
  Transforms/VariantMaterializationTest.cpp
)

target_link_libraries(tianchenrv-variant-materialization-test
  PRIVATE
  TianChenRVInitAll
  TianChenRVPlugin
  TianChenRVSupport
  TianChenRVTransforms
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
)

add_executable(tianchenrv-variant-dispatch-synthesis-test
  Transforms/VariantDispatchSynthesisTest.cpp
)

target_link_libraries(tianchenrv-variant-dispatch-synthesis-test
  PRIVATE
  TianChenRVInitAll
  TianChenRVSupport
  TianChenRVTransforms
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
)

add_executable(tianchenrv-capability-model-test
  Support/CapabilityModelTest.cpp
)

target_link_libraries(tianchenrv-capability-model-test
  PRIVATE
  TianChenRVInitAll
  TianChenRVSupport
  MLIRIR
  MLIRParser
  MLIRSupport
)

add_lit_testsuite(check-tianchenrv "Running TianChen-RV lit tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  tcrv-opt
  tianchenrv-capability-model-test
  tianchenrv-plugin-registry-test
  tianchenrv-plugin-variant-legality-test
  tianchenrv-plugin-variant-proposal-test
  tianchenrv-variant-dispatch-synthesis-test
  tianchenrv-variant-materialization-test
  ARGS
  -sv
)

set_target_properties(check-tianchenrv PROPERTIES FOLDER "Tests")
