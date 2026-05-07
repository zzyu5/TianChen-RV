add_mlir_library(TianChenRVRVVTarget
  RVVSmokeProbe.cpp

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  TianChenRVRVVDialect
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)
