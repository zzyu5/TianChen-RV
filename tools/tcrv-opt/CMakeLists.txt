add_mlir_tool(tcrv-opt
  tcrv-opt.cpp
)

target_link_libraries(tcrv-opt
  PRIVATE
  TianChenRVBuiltinTargetArtifactExporters
  TianChenRVBuiltinPlugins
  TianChenRVInitAll
  TianChenRVExecDialect
  TianChenRVRVVPlugin
  TianChenRVTarget
  TianChenRVTransforms
  MLIROptLib
  MLIRIR
  MLIRSupport
)
