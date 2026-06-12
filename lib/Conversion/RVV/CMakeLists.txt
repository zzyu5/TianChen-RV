add_mlir_library(TianChenRVConversionRVV
  RVVToEmitC.cpp

  DEPENDS
  TianChenRVPassesIncGen
  MLIRRVVOpsIncGen

  LINK_LIBS PUBLIC
  TianChenRVRVVDialect
  MLIREmitCDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
)
