//===- PassDetail.h - DPU pass class details -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_DPU_TRANSFORMS_PASSDETAIL_H
#define MLIR_DIALECT_DPU_TRANSFORMS_PASSDETAIL_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace dpu {
class DPUDialect;
} // namespace dpu

#define GEN_PASS_CLASSES
#include "mlir/Dialect/DPU/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_DPU_TRANSFORMS_PASSDETAIL_H
