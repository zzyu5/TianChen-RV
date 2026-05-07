add_mlir_library(TianChenRVOffloadTarget
  OffloadRuntimeDescriptor.cpp

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  TianChenRVOffloadDialect
  MLIRIR
  MLIRSupport
)
