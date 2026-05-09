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
  TianChenRVTransforms
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRPass
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

add_executable(tianchenrv-plugin-variant-cost-test
  Plugin/PluginVariantCostTest.cpp
)

target_link_libraries(tianchenrv-plugin-variant-cost-test
  PRIVATE
  TianChenRVInitAll
  TianChenRVPlugin
  TianChenRVSupport
  MLIRIR
  MLIRParser
  MLIRSupport
)

add_executable(tianchenrv-rvv-extension-plugin-test
  Plugin/RVVExtensionPluginTest.cpp
)

target_link_libraries(tianchenrv-rvv-extension-plugin-test
  PRIVATE
  TianChenRVInitAll
  TianChenRVBuiltinPlugins
  TianChenRVPlugin
  TianChenRVRVVPlugin
  TianChenRVSupport
  TianChenRVTransforms
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
)

add_executable(tianchenrv-scalar-extension-plugin-test
  Plugin/ScalarExtensionPluginTest.cpp
)

target_link_libraries(tianchenrv-scalar-extension-plugin-test
  PRIVATE
  TianChenRVInitAll
  TianChenRVBuiltinPlugins
  TianChenRVPlugin
  TianChenRVScalarPlugin
  TianChenRVSupport
  TianChenRVTransforms
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
)

add_executable(tianchenrv-offload-extension-plugin-test
  Plugin/OffloadExtensionPluginTest.cpp
)

target_link_libraries(tianchenrv-offload-extension-plugin-test
  PRIVATE
  TianChenRVInitAll
  TianChenRVBuiltinPlugins
  TianChenRVPlugin
  TianChenRVOffloadPlugin
  TianChenRVScalarPlugin
  TianChenRVSupport
  TianChenRVTransforms
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
)

add_executable(tianchenrv-rvv-dialect-test
  Dialect/RVV/RVVDialectTest.cpp
)

target_link_libraries(tianchenrv-rvv-dialect-test
  PRIVATE
  TianChenRVInitAll
  TianChenRVRVVDialect
  TianChenRVRVVPlugin
  TianChenRVPlugin
  MLIRIR
  MLIRParser
  MLIRSupport
)

add_executable(tianchenrv-variant-materialization-test
  Transforms/VariantMaterialization/VariantMaterializationTest.cpp
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
  Transforms/DispatchSynthesis/VariantDispatchSynthesisTest.cpp
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

add_executable(tianchenrv-emission-readiness-test
  Transforms/EmissionReadiness/EmissionReadinessTest.cpp
)

target_link_libraries(tianchenrv-emission-readiness-test
  PRIVATE
  TianChenRVInitAll
  TianChenRVPlugin
  TianChenRVRVVPlugin
  TianChenRVSupport
  TianChenRVTransforms
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
)

add_executable(tianchenrv-rvv-lowering-boundary-test
  Transforms/LoweringBoundary/RVVLoweringBoundaryTest.cpp
)

target_link_libraries(tianchenrv-rvv-lowering-boundary-test
  PRIVATE
  TianChenRVInitAll
  TianChenRVBuiltinPlugins
  TianChenRVPlugin
  TianChenRVRVVPlugin
  TianChenRVScalarPlugin
  TianChenRVSupport
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
)

add_executable(tianchenrv-variant-selection-test
  Transforms/VariantSelection/VariantSelectionTest.cpp
)

target_link_libraries(tianchenrv-variant-selection-test
  PRIVATE
  TianChenRVInitAll
  TianChenRVBuiltinPlugins
  TianChenRVPlugin
  TianChenRVRVVPlugin
  TianChenRVScalarPlugin
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

add_executable(tianchenrv-target-artifact-export-test
  Target/TargetArtifactExportTest.cpp
)

target_link_libraries(tianchenrv-target-artifact-export-test
  PRIVATE
  TianChenRVInitAll
  TianChenRVBuiltinTargetArtifactExporters
  TianChenRVTarget
  MLIRIR
  MLIRParser
  MLIRSupport
)

add_executable(tianchenrv-i32-binary-family-registry-test
  Target/I32BinaryFamilyRegistryTest.cpp
)

target_link_libraries(tianchenrv-i32-binary-family-registry-test
  PRIVATE
  TianChenRVBuiltinTargetArtifactExporters
  TianChenRVTarget
  MLIRIR
  MLIRSupport
)

add_lit_testsuite(check-tianchenrv "Running TianChen-RV lit tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  tcrv-opt
  tcrv-translate
  tianchenrv-capability-model-test
  tianchenrv-plugin-variant-cost-test
  tianchenrv-plugin-registry-test
  tianchenrv-plugin-variant-legality-test
  tianchenrv-plugin-variant-proposal-test
  tianchenrv-emission-readiness-test
  tianchenrv-i32-binary-family-registry-test
  tianchenrv-offload-extension-plugin-test
  tianchenrv-rvv-dialect-test
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-rvv-lowering-boundary-test
  tianchenrv-scalar-extension-plugin-test
  tianchenrv-target-artifact-export-test
  tianchenrv-variant-dispatch-synthesis-test
  tianchenrv-variant-materialization-test
  tianchenrv-variant-selection-test
  ARGS
  -sv
)

set_target_properties(check-tianchenrv PROPERTIES FOLDER "Tests")
