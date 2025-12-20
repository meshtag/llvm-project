//===- Passes.h - DPU dialect passes ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the DPU dialect passes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_DPU_PASSES_H
#define MLIR_DIALECT_DPU_PASSES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

/// Creates a pass to lower DPU dialect to LLVM dialect.
std::unique_ptr<Pass> createConvertDPUToLLVMPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/DPU/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_DPU_PASSES_H
