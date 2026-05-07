add_mlir_tool(tcrv-translate
  tcrv-translate.cpp
)

target_link_libraries(tcrv-translate
  PRIVATE
  TianChenRVBuiltinPlugins
  TianChenRVInitAll
  TianChenRVRVVTarget
  TianChenRVScalarTarget
  TianChenRVTarget
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
)
