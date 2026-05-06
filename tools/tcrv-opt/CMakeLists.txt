add_mlir_tool(tcrv-opt
  tcrv-opt.cpp
)

target_link_libraries(tcrv-opt
  PRIVATE
  TianChenRVInitAll
  TianChenRVExecDialect
  TianChenRVTransforms
  MLIROptLib
  MLIRIR
  MLIRSupport
)
