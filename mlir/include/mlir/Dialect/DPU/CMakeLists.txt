add_mlir_dialect(DPUOps dpu)
add_mlir_doc(DPUOps -gen-op-doc DPUOps Dialects/)
add_mlir_doc(DPUOps -gen-dialect-doc DPUDialect Dialects/)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name DPU)
add_public_tablegen_target(MLIRDPUTransformsIncGen)
add_dependencies(mlir-headers MLIRDPUTransformsIncGen)

add_mlir_doc(Passes -gen-pass-doc DPUPasses ./)
