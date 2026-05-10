add_mlir_library(TianChenRVRVVTarget
  RVVBinaryMicrokernelBodyVerifier.cpp
  RVVMicrokernel.cpp
  RVVSmokeProbe.cpp

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  TianChenRVRVVDialect
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)
