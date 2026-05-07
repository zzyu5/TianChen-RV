add_mlir_tool(tcrv-opt
  tcrv-opt.cpp
)

target_link_libraries(tcrv-opt
  PRIVATE
  TianChenRVBuiltinPlugins
  TianChenRVInitAll
  TianChenRVExecDialect
  TianChenRVRVVPlugin
  TianChenRVTransforms
  MLIROptLib
  MLIRIR
  MLIRSupport
)
