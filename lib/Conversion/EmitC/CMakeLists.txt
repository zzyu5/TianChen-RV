add_mlir_library(TianChenRVConversionEmitC
  TCRVEmitCLowerableOpInterface.cpp
  TCRVEmitCLowerableInterface.cpp

  DEPENDS
  MLIRTCRVEmitCLowerableOpInterfaceIncGen

  LINK_LIBS PUBLIC
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)
