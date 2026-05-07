add_mlir_library(TianChenRVRVVTarget
  RVVMicrokernel.cpp
  RVVSmokeProbe.cpp

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  TianChenRVRVVDialect
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)
