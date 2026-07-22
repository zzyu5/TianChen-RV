add_mlir_library(WeftConversionRVV
  RVVToEmitC.cpp
  RVVToEmitCBlockQuantLinear.cpp
  RVVToEmitCCodebookFp4.cpp
  RVVToEmitCDeferredDequant.cpp
  RVVToEmitCForwardElementwise.cpp
  RVVToEmitCGridCodebook.cpp
  RVVToEmitCKQuant.cpp
  RVVToEmitCTernaryBinary.cpp
  RVVToEmitCSupport.cpp

  DEPENDS
  WeftPassesIncGen
  MLIRRVVOpsIncGen
  MLIRExecOpsIncGen

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftRVVDialect
  WeftExecDialect
  WeftRVVConstructionProtocol
  WeftSupport
  MLIREmitCDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
)
