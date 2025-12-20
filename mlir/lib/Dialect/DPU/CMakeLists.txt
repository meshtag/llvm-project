add_subdirectory(Transforms)

add_mlir_dialect_library(MLIRDPU
  DPUOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/DPU

  DEPENDS
  MLIRDPUOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMIR
  MLIRSideEffectInterfaces
  MLIRSupport
)
