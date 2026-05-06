add_subdirectory(Dialect)

add_mlir_library(TianChenRVInitAll
  InitTianChenRVDialects.cpp

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  MLIRIR
)
