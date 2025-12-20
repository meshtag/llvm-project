//===- Passes.td - DPU dialect passes ------------------------*- tablegen
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_DPU_PASSES_TD
#define MLIR_DIALECT_DPU_PASSES_TD

include "mlir/Pass/PassBase.td"

// DPU to LLVM lowering.
def ConvertDPUToLLVM : Pass<"convert-dpu-to-llvm", "ModuleOp"> {
  let summary = "Lower DPU dialect to LLVM dialect";
  let constructor = "mlir::createConvertDPUToLLVMPass()";
}

#endif // MLIR_DIALECT_DPU_PASSES_TD
