add_mlir_dialect_library(TianChenRVTemplateDialect
  TemplateDialect.cpp

  DEPENDS
  MLIRTemplateOpsIncGen

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  TianChenRVSupport
  MLIRIR
)
