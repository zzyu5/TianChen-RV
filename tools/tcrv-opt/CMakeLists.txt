add_mlir_tool(tcrv-opt
  tcrv-opt.cpp
)

target_link_libraries(tcrv-opt
  PRIVATE
  TianChenRVBuiltinPlugins
  TianChenRVInitAll
  TianChenRVExecDialect
  TianChenRVTransforms
  MLIROptLib
  MLIRIR
  MLIRSupport
)
