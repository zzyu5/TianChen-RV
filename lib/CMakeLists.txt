add_subdirectory(Dialect)
add_subdirectory(Plugin)

add_mlir_library(TianChenRVInitAll
  InitTianChenRVDialects.cpp

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  TianChenRVPlugin
  MLIRIR
)
