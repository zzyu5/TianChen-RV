add_mlir_library(TianChenRVConversionEmitC
  TCRVEmitCLowerableMaterializer.cpp
  TCRVEmitCLowerableOpInterface.cpp
  TCRVEmitCLowerableInterface.cpp

  DEPENDS
  MLIRTCRVEmitCLowerableOpInterfaceIncGen

  LINK_LIBS PUBLIC
  TianChenRVSupport
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
)
