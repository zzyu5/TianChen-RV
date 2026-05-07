add_mlir_library(TianChenRVScalarTarget
  ScalarMicrokernel.cpp

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  TianChenRVScalarDialect
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)
