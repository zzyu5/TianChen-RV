add_mlir_library(TianChenRVScalarTarget
  ScalarMicrokernel.cpp

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVScalarDialect
  TianChenRVSupport
  TianChenRVTarget
  MLIRIR
  MLIRSupport
)
