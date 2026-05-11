add_mlir_library(TianChenRVConversionEmitC
  TCRVEmitCLowerableInterface.cpp

  LINK_LIBS PUBLIC
  TianChenRVSupport
  MLIRSupport
)
