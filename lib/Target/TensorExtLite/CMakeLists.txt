add_mlir_library(TianChenRVTensorExtLiteTarget
  TensorExtLiteMetadataArtifact.cpp

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  TianChenRVTensorExtLiteDialect
  TianChenRVTensorExtLitePlugin
  TianChenRVSupport
  TianChenRVTarget
  MLIRIR
  MLIRSupport
)
