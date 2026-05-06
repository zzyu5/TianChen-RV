add_subdirectory(Dialect)
add_subdirectory(Support)
add_subdirectory(Plugin)
add_subdirectory(Transforms)

add_mlir_library(TianChenRVInitAll
  InitTianChenRVDialects.cpp

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  TianChenRVPlugin
  MLIRIR
)
